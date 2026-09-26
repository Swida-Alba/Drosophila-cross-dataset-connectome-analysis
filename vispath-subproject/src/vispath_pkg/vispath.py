"""
VisualizePath - Visualization Module for Neural Pathways

This module provides standalone visualization capabilities for neural pathways
discovered through connectome analysis. It can be used independently or 
integrated with FindNeuronConnection.

Classes:
    VisualizePath: Main class for pathway visualization

Author: Kun-Da Wu
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import polars as pl
from .fast_graph_core import FastGraph
from .shared_controls import SHARED_JS, js_escape, html_escape, json_safe
import os
import json
import webbrowser
from pathlib import Path
import ast
import re


def _json_default(o):
    """JSON encoder fallback for numpy scalars in embedded visualization data."""
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def parse_color_to_hex_opacity(color_str):
    """
    Parse a color string (hex, rgb, rgba, named) into hex color and opacity.
    
    Parameters
    ----------
    color_str : str
        Color in any CSS format: '#3498db', 'rgb(52, 152, 219)', 
        'rgba(52, 152, 219, 0.5)', 'blue', etc.
    
    Returns
    -------
    tuple
        (hex_color: str, opacity: float) e.g., ('#3498db', 1.0)
    
    Examples
    --------
    >>> parse_color_to_hex_opacity('rgba(44, 160, 44, 0.2)')
    ('#2ca02c', 0.2)
    >>> parse_color_to_hex_opacity('#1f77b4')
    ('#1f77b4', 1.0)
    """
    if not color_str:
        return ('#000000', 1.0)
    
    color_str = color_str.strip()
    
    # Parse rgba(r, g, b, a) or rgb(r, g, b)
    rgba_match = re.match(r'rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)', color_str)
    if rgba_match:
        r = int(rgba_match.group(1))
        g = int(rgba_match.group(2))
        b = int(rgba_match.group(3))
        a = float(rgba_match.group(4)) if rgba_match.group(4) else 1.0
        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        return (hex_color, a)
    
    # Parse hex color #RRGGBB or #RGB
    if color_str.startswith('#'):
        hex_color = color_str
        # Expand #RGB to #RRGGBB
        if len(hex_color) == 4:
            hex_color = '#' + ''.join([c*2 for c in hex_color[1:]])
        return (hex_color, 1.0)
    
    # Named colors (basic support)
    named_colors = {
        'red': '#ff0000', 'green': '#00ff00', 'blue': '#0000ff',
        'white': '#ffffff', 'black': '#000000', 'gray': '#808080', 'grey': '#808080',
        'yellow': '#ffff00', 'cyan': '#00ffff', 'magenta': '#ff00ff',
        'orange': '#ffa500', 'purple': '#800080', 'pink': '#ffc0cb',
        'brown': '#a52a2a', 'navy': '#000080', 'teal': '#008080'
    }
    if color_str.lower() in named_colors:
        return (named_colors[color_str.lower()], 1.0)
    
    # Default fallback
    return (color_str, 1.0)


def blend_with_gray(hex_color, factor=0.4):
    """
    Desaturate a hex color by blending with mid-gray.
    factor=0 returns original color, factor=1 returns gray.
    """
    if not hex_color:
        return '#808080'
    hex_color = hex_color.strip()
    if not hex_color.startswith('#'):
        hex_color, _ = parse_color_to_hex_opacity(hex_color)
    if len(hex_color) == 4:
        hex_color = '#' + ''.join([c * 2 for c in hex_color[1:]])
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except Exception:
        return '#808080'
    gray = 128
    r = int(r * (1 - factor) + gray * factor)
    g = int(g * (1 - factor) + gray * factor)
    b = int(b * (1 - factor) + gray * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


# Neurotransmitter color palette
# Colors chosen based on user requirements
NT_COLORS = {
    # Acetylcholine - Orange
    'acetylcholine': '#FFA500',
    'ACH': '#FFA500',
    'ach': '#FFA500',
    
    # GABA - Blue
    'gaba': '#0000FF',
    'GABA': '#0000FF',
    
    # Glutamate - Green
    'glutamate': '#008000',
    'GLUT': '#008000',
    'glut': '#008000',
    
    # Dopamine - Red
    'dopamine': '#FF0000',
    'DA': '#FF0000',
    'da': '#FF0000',
    
    # Serotonin - Yellow
    'serotonin': '#FFD700',
    'SER': '#FFD700',
    'ser': '#FFD700',
    '5-HT': '#FFD700',
    '5-ht': '#FFD700',
    
    # Octopamine - Purple
    'octopamine': '#800080',
    'OCT': '#800080',
    'oct': '#800080',

    # Histamine - Cyan
    'histamine': '#00FFFF',
    'HISTAMINE': '#00FFFF',
    'his': '#00FFFF',
    
    # Unknown/other
    'unknown': '#95A5A6',        # Gray
    'none': '#95A5A6',
    '': '#95A5A6',
}

# NT type groupings for interactive color adjustment.
# GLUT (glutamate) is treated as inhibitory: in the adult fly CNS
# glutamatergic transmission acts largely through inhibitory receptors.
NT_GROUPS = {
    'excitatory': ['acetylcholine', 'ACH', 'ach'],
    'inhibitory': ['gaba', 'GABA', 'glutamate', 'GLUT', 'glut'],
    'modulatory': ['dopamine', 'DA', 'da', 'serotonin', 'SER', 'ser', '5-HT', '5-ht', 'octopamine', 'OCT', 'oct'],
    'unknown': ['unknown', 'none', ''],
}

# Default colors for NT groups (for interactive color pickers)
NT_GROUP_COLORS = {
    'excitatory': '#F39C12',  # Orange
    'inhibitory': '#27AE60',  # Green
    'modulatory': '#9B59B6',  # Purple
    'unknown': '#95A5A6',     # Gray
}

def get_nt_group(nt_type):
    """Get the group name for a neurotransmitter type."""
    if nt_type is None or pd.isna(nt_type):
        return 'unknown'
    nt_str = str(nt_type).strip()
    for group, members in NT_GROUPS.items():
        if nt_str in members or nt_str.lower() in [m.lower() for m in members]:
            return group
    return 'unknown'

def get_nt_color(nt_type, opacity=0.6):
    """
    Get the color for a neurotransmitter type.
    
    Parameters
    ----------
    nt_type : str or None
        Neurotransmitter type (e.g., 'acetylcholine', 'gaba', 'dopamine')
    opacity : float
        Opacity for the color (0.0 to 1.0)
        
    Returns
    -------
    str
        RGBA color string
    """
    if nt_type is None or pd.isna(nt_type):
        hex_color = NT_COLORS.get('unknown', '#95A5A6')
    else:
        # Try exact match first, then lowercase
        nt_str = str(nt_type).strip()
        hex_color = NT_COLORS.get(nt_str) or NT_COLORS.get(nt_str.lower()) or NT_COLORS.get('unknown', '#95A5A6')
    
    # Convert hex to rgba
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'


class VisualizePath:
    """
    A class for visualizing neural pathways from CSV/Excel files.
    
    This class provides methods to create interactive visualizations (Sankey diagrams
    and network graphs) from pathway data, typically generated by FindNeuronConnection.FindAllPath().
    
    Attributes:
        path_file (str or pd.DataFrame): Path to CSV/Excel file or DataFrame with pathway data
        sheet_name (str): Excel sheet name to read (default: auto-detect)
        output_folder (str): Folder to save visualization outputs
        source_color (str): Color for source nodes
        intermediate_color (str): Color for intermediate nodes
        target_color (str): Color for target nodes
        link_color (str): Color for connections in Sankey
        network_layout (str): Layout algorithm for network graph
        showfig (bool): Whether to automatically open visualizations in browser
        
    Example:
        >>> # Standalone usage
        >>> vp = VisualizePath(path_file='path_type.xlsx')
        >>> vp.visualize()
        
        >>> # With custom colors
        >>> vp = VisualizePath(
        ...     path_file='selected_paths.csv',
        ...     source_color='#FF6B6B',
        ...     intermediate_color='#FFA500',
        ...     target_color='#FFD700',
        ...     output_folder='./my_viz',
        ...     network_layout='spring',
        ...     showfig=True
        ... )
        >>> conn_df, G = vp.visualize()
    """
    
    def __init__(
        self,
        path_file,
        sheet_name=None,
        output_folder=None,
        source_color=None,
        intermediate_color=None,
        target_color=None,
        link_color=None,
        highlight_color=None,    # NEW: Highlight color for selected elements
        node_color=None,  # For backward compatibility
        node_colors=None,  # NEW: Custom node colors
        network_layout='hierarchical',
        showfig=False,
        edge_width_scale='log',  # NEW: Edge width scaling method
        edge_width_factor=1.0,   # NEW: Edge width multiplier
        edge_width_log_base=None, # NEW: Log base for logarithmic scaling (None = natural log e)
        min_edge_width=0.5,      # NEW: Minimum edge width in pixels
        max_edge_width=30,       # NEW: Maximum edge width in pixels
        min_font_size=6,         # NEW: Minimum font size in pixels
        max_font_size=48,        # NEW: Maximum font size in pixels
        edge_label_font_size=9,  # NEW: On-edge weight label size in px (adjustable in the UI)
        min_node_size=20,        # NEW: Minimum node size in pixels
        max_node_size=80,        # NEW: Maximum node size in pixels
        heatmap_row_order=None,  # NEW: Custom row order for heatmap
        heatmap_col_order=None,  # NEW: Custom column order for heatmap
        straight_reciprocal_edges=True,  # NEW: Use straight lines for reciprocal edges
        generate_empty_network=False,  # NEW: Generate empty network HTML template
        edgeN_limit=500,        # NEW: Limit number of edges to show (default 500)
        output_format='xlsx',   # NEW: Output format for data files ('xlsx' or 'csv')
        save_data_matrices=True,  # Write connMatrix sheets/files in save_data()
        verbose=True,           # NEW: Control print output (True=show prints, False=silent)
        edge_labels=None,       # NEW: Custom edge labels dict {(source, target): {'label_name': value, ...}}
        color_edges_by_nt=False, # NEW: Color edges by neurotransmitter type
        dataset_legend=None,    # NEW: Dataset short code legend {code: full_name} for display names
        dataset_legend_meta=None,  # NEW: per-code {'count': int, 'color': str} — chip count + swatch override
        node_dataset_info=None, # NEW: Node-level dataset info {node_label: {code: name_in_dataset}}
        node_groups=None,       # NEW: Declared node groups [{name,label,color}] (dataset groups in the mapping views)
        separate_hemispheres=False,  # NEW: Enable hemisphere-aware coloring/layout
        hemisphere_desaturate_side='R',  # NEW: Hemisphere to desaturate ('L' or 'R')
        hemisphere_desaturate_factor=0.4,  # NEW: Desaturation blend factor (0-1)
        hemisphere_mirror_default=None,  # None = auto-enable with separate_hemispheres
        progress_total=None,  # Optional [DROCAT][progress] step total (web UI runs only)
        edge_weight_label='synapses',  # Unit label for the edge weight in hover info
        sankey_title=None,  # Sankey heading; None = the pathway default
        sankey_label_layers=True,  # Append the (L<n>) hop layer to Sankey labels
        metric_option_label=None,  # 'weight' option text; None = 'Synapse Count'
    ):
        """
        Initialize VisualizePath with pathway data and visualization settings.
        
        Parameters
        ----------
                {
                    selector: 'node.placeholder',
                    style: {
                        'opacity': 0.0,
                        'label': ''
                    }
                },
        path_file : str or pd.DataFrame
            Path to CSV/Excel file or DataFrame containing pathway data.
            Required columns: 'path_block', 'weights'
            Optional columns: 'connection_ratios', 'traversal_probabilities'
            
        sheet_name : str, optional
            Excel sheet name to read. If None, auto-detects 'path_type' or 'path_bodyId'.
            Ignored for CSV files and DataFrames.
            
        output_folder : str, optional
            Directory to save visualization files. If None, creates '[filename]_figure'
            relative to the input file location (for files) or './selected_paths' (for DataFrames).
            Example: 'L3_to_MeVPMe_allpaths_info.xlsx' → 'L3_to_MeVPMe_allpaths_info_figure/'
            
        source_color : str, optional
            Color for source nodes.
            Default: '#1f77b4' (blue)
            Format: Any valid CSS color (hex, rgb, rgba, named)
            
        intermediate_color : str, optional
            Color for intermediate nodes.
            Default: '#2ca02c' (green)
            Format: Any valid CSS color (hex, rgb, rgba, named)
            
        target_color : str, optional
            Color for target nodes.
            Default: '#d62728' (red)
            
        link_color : str, optional
            Color for connections in Sankey diagram and network edges.
            Default: 'rgba(100,100,100,0.3)' (semi-transparent gray)
            
        highlight_color : str, optional
            Color for highlighted/selected nodes and edges in network visualization.
            Used when clicking on nodes or edges to highlight them.
            Default: '#FF9800' (orange)
            Format: Any valid CSS color (hex, rgb, rgba, named)
            
        node_color : list of str, optional
            [DEPRECATED] Colors for [source_nodes, intermediate_nodes].
            Use source_color and intermediate_color instead.
            Kept for backward compatibility.
            
        node_colors : str or pd.DataFrame, optional
            Custom colors for specific nodes. Can be:
            - Sheet name (str): Name of sheet in Excel file with 'node' and 'color' columns
            - File path (str): Path to CSV/Excel file with 'node' and 'color' columns
            - DataFrame: DataFrame with 'node' and 'color' columns
            Color column supports hex (#RRGGBB) and rgba (rgba(r,g,b,a)) formats.
            Nodes not specified will use default source/intermediate/target colors.
            
        network_layout : str, optional
            Layout algorithm for network graph.
            Options: 'hierarchical', 'spring', 'circular', 'distributed'
            Default: 'hierarchical'
            
        showfig : bool, optional
            Whether to automatically open visualizations in web browser.
            Default: False
            
        edge_width_scale : str, optional
            Edge width scaling method for network graph visualization.
            Options: 'linear', 'log', 'sqrt', 'none'
            - 'linear': Direct proportional scaling (width ∝ weight)
            - 'log': Logarithmic scaling (width ∝ log(weight)) - DEFAULT
            - 'sqrt': Square root scaling (width ∝ √weight)
            - 'none': No scaling (constant width)
            Default: 'log'
            Note: For Sankey diagrams, Plotly auto-scales link widths proportionally.
            
        edge_width_factor : float, optional
            Multiplier for edge widths in network graph (applies after scaling).
            Larger values make edges thicker. Default: 1.0
            
        edge_width_log_base : float, optional
            Base for logarithmic scaling when edge_width_scale='log'.
            - None: Natural logarithm (base e ≈ 2.718) - DEFAULT
            - 2: Binary logarithm (log₂)
            - 10: Common logarithm (log₁₀)
            - Any positive number > 1: Custom base
            Only used when edge_width_scale='log'. Ignored for other scaling methods.
            Default: None (natural log)
            
        min_edge_width : float, optional
            Minimum edge width in pixels for network visualization.
            This is a fixed lower bound - the slider controls max width.
            Default: 0.5
            
        max_edge_width : float, optional
            Maximum edge width in pixels for network visualization.
            This value is controlled by the "Edge Width" slider in the UI.
            Default: 30
            
        min_font_size : int, optional
            Minimum font size in pixels for node labels.
            This is the minimum value for the "Font Size" slider.
            Default: 6
            
        max_font_size : int, optional
            Maximum font size in pixels for node labels.
            This is the maximum value for the "Font Size" slider.
            Default: 48
            
        min_node_size : int, optional
            Minimum node size in pixels.
            This is the minimum value for the "Node Size" slider.
            Default: 20
            
        max_node_size : int, optional
            Maximum node size in pixels.
            This is the maximum value for the "Node Size" slider.
            Default: 80
            
        heatmap_row_order : list of str, optional
            Custom row order for heatmap row nodes (sources).
            If None, uses default sorted order.
            Nodes not in the list will be appended at the end (sorted).
            Example: ['PN_A', 'PN_B', 'LHN_X', 'LHN_Y']
            Default: None
            
        heatmap_col_order : list of str, optional
            Custom order for heatmap column nodes (targets).
            If None, uses default sorted order.
            Nodes not in the list will be appended at the end (sorted).
            Example: ['LHN_X', 'MBON_1', 'MBON_2']
            Default: None
            
        straight_reciprocal_edges : bool, optional
            If True, reciprocal (bidirectional) edges in network visualization will be 
            displayed as straight lines instead of curved lines.
            This makes it easier to see both directions clearly without visual overlap.
            Cytoscape.js supports this via 'curve-style' and 'control-point-distances'.
            Default: False (uses curved lines for reciprocal edges)
            
        generate_empty_network : bool, optional
            If True, generates an empty network HTML template without requiring
            path_file data. Useful for creating blank network visualizations that
            can be populated later or used as templates.
            When enabled, path_file can be None.
            Default: False
            
        edgeN_limit : int, optional
            Limit number of edges to show in visualizations.
            Default: 1000
            
        verbose : bool, optional
            Control print output during visualization.
            If True, shows progress messages and file save notifications.
            If False, runs silently (no print output).
            Default: True
            
        edge_labels : dict, optional
            Custom edge labels for hover tooltips. Dictionary mapping edge tuples
            to label dictionaries: {(source, target): {'label_name': value, ...}}
            Each label_name will be shown as a separate line in the hover tooltip.
            Useful for showing synapse strengths from multiple datasets.
            Example: {('PPL101', 'aMe12'): {'HEMI': 45, 'MCNS': 32, 'FAFB': 28}}
            Default: None
        
        separate_hemispheres : bool, optional
            If True, enables hemisphere-aware coloring and layout controls.
            Nodes with suffix _L/_R/_U will be handled specially.
            Default: False
        
        hemisphere_desaturate_side : str, optional
            Hemisphere side to desaturate. Use 'L' or 'R'.
            Default: 'R'
        
        hemisphere_desaturate_factor : float, optional
            Blend factor (0-1) for desaturation (higher = more gray).
            Default: 0.4
        
        hemisphere_mirror_default : bool, optional
            If True, mirror hemispheres by default in the network layout.
            Default: None — automatically follows separate_hemispheres
            (mirroring is enabled whenever Separate Hemispheres is on,
            unless explicitly set to False).
            
        Raises
        ------
        FileNotFoundError
            If path_file is a string and the file doesn't exist
        ValueError
            If required columns are missing from the data
        """
        self.path_file = path_file
        self.sheet_name = sheet_name
        self.output_folder = output_folder
        self.verbose = verbose  # Store verbose flag for controlling print output
        # Step events are opt-in: nested VisualizePath calls inside coana /
        # comparer pipelines must stay silent so the outer tool's protocol
        # owns the bar.  Only the standalone Net-Viz run passes a total.
        self.progress_total = progress_total
        
        self.edgeN_limit = edgeN_limit
        # Set True once any plot (network/heatmap/Sankey) actually trims
        # edges to the Visualization Edge Limit; read by coana to gate the
        # '[edge limit per neuron]' warning note.
        self.edge_limit_trimmed = False
        self.output_format = output_format
        self.save_data_matrices = save_data_matrices
        
        # Custom edge labels for multi-dataset synapse info
        self.edge_labels = edge_labels  # Dict: {(source, target): {label_name: value, ...}}
        
        # Dataset legend for cross-dataset type name display
        # Format: {short_code: full_dataset_name} e.g., {'M': 'male-cns v0.9', 'F': 'FlyWire FAFB v783'}
        self.dataset_legend = dataset_legend or {}
        # Per-code legend extras (§layout unification): {'count': node
        # count, 'color': swatch override} — the chip renders
        # ``CODE: full (count)`` and linker-column entries carry their
        # LINKER_COLORS swatch through 'color'.
        self.dataset_legend_meta = dataset_legend_meta or {}
        
        # Node-level dataset info for hover labels
        # Format: {node_label: {code: name_in_that_dataset}} e.g., {'MeVP(MTe07)': {'M': 'MeVP', 'F': 'MTe07'}}
        self.node_dataset_info = node_dataset_info or {}
        # Declared node groups: [{name, label, color}].  When set, nodes
        # carrying a matching 'group' attribute are grouped and colored BY
        # GROUP (per-dataset groups in the type-mapping views) — quick
        # actions, the color/opacity dropdown, selection, reset, and the
        # footer legend all follow the declared list; the structural
        # node_type role is preserved on the node's 'role' data field for
        # the hover.
        self.node_groups = [dict(g) for g in (node_groups or [])]
        # Unit label shown with the edge weight (e.g. 'synapses' for
        # connectome graphs, 'neurons' for type-mapping graphs).
        self.edge_weight_label = str(edge_weight_label or 'synapses')

        def _js(value: str) -> str:
            # Values that land inside single-quoted JS literals in the
            # templates below; a caller-supplied title may carry an
            # apostrophe.  The HTML/plotly copies stay raw.
            return (str(value).replace('\\', '\\\\')
                    .replace("'", "\\'").replace('\n', ' ').replace('\r', ''))
        # Sankey heading, node-label layer suffix, and the metric selector's
        # 'weight' option text.  A pathway artifact reads "…(L1)" as hop depth
        # and names its metric 'Synapse Count'; a type-mapping artifact has
        # one hop per DATASET column (so the suffix is noise — and wrong once
        # a view reverses a half's link direction to centre its source) and
        # counts NEURONS, not synapses.  Those callers pass all three; the
        # defaults keep every pathway artifact byte-identical.
        self.sankey_title = str(sankey_title
                                or 'Sankey diagram of pathway connections')
        self._sankey_title_js = _js(self.sankey_title)
        self.sankey_label_layers = bool(sankey_label_layers)
        self.metric_option_label = str(metric_option_label
                                       or 'Synapse Count')
        # The Sankey hover names the same metric.
        self._sankey_weight_name = (_js(metric_option_label)
                                    if metric_option_label else 'Synapses')
        # Dagre flow direction: connectome hierarchies read top-to-bottom,
        # mapping artifacts pass 'LR' so source→target reads left-to-right.
        self.dagre_rank_dir = 'TB'

        # Hemisphere visualization options
        self.separate_hemispheres = separate_hemispheres
        self.hemisphere_desaturate_side = (hemisphere_desaturate_side or 'R').upper()
        self.hemisphere_desaturate_factor = max(0.0, min(1.0, hemisphere_desaturate_factor))
        # Mirror layout auto-enables when Separate Hemispheres (L/R) is checked,
        # unless the caller explicitly disabled it (None = follow the checkbox).
        if hemisphere_mirror_default is None:
            hemisphere_mirror_default = separate_hemispheres
        self.hemisphere_mirror_default = bool(hemisphere_mirror_default)
        
        # Neurotransmitter-based edge coloring
        self.color_edges_by_nt = color_edges_by_nt  # If True, color edges by NT type
        
        # Edge width scaling parameters
        self.edge_width_scale = edge_width_scale
        self.edge_width_factor = edge_width_factor
        self.edge_width_log_base = edge_width_log_base if edge_width_log_base is not None else 'e'  # Default to natural log
        self.min_edge_width = min_edge_width  # Minimum edge width in pixels (fixed lower bound)
        self.max_edge_width = max_edge_width  # Maximum edge width in pixels (controlled by slider)
        
        # Font and node size parameters
        self.min_font_size = min_font_size  # Minimum font size in pixels
        self.max_font_size = max_font_size  # Maximum font size in pixels
        self.edge_label_font_size = edge_label_font_size  # On-edge weight label size (px)
        self.min_node_size = min_node_size  # Minimum node size in pixels
        self.max_node_size = max_node_size  # Maximum node size in pixels
        
        # Heatmap ordering parameters
        self.heatmap_row_order = heatmap_row_order  # Custom row order for heatmap
        self.heatmap_col_order = heatmap_col_order  # Custom column order for heatmap
        
        # Empty network generation flag
        self.generate_empty_network = generate_empty_network
        
        # Handle color parameters - support both new (source_color, intermediate_color)
        # and old (node_color) API for backward compatibility
        # Parse colors to separate hex and opacity
        if source_color is not None or intermediate_color is not None:
            # New API: individual colors
            source_hex, source_opacity = parse_color_to_hex_opacity(source_color or '#1f77b4')
            intermediate_hex, intermediate_opacity = parse_color_to_hex_opacity(intermediate_color or '#2ca02c')
            self.source_color = source_hex
            self.source_opacity = source_opacity
            self.intermediate_color = intermediate_hex
            self.intermediate_opacity = intermediate_opacity
        elif node_color is not None:
            # Old API: node_color array for backward compatibility
            if isinstance(node_color, list) and len(node_color) >= 2:
                source_hex, source_opacity = parse_color_to_hex_opacity(node_color[0])
                intermediate_hex, intermediate_opacity = parse_color_to_hex_opacity(node_color[1])
                self.source_color = source_hex
                self.source_opacity = source_opacity
                self.intermediate_color = intermediate_hex
                self.intermediate_opacity = intermediate_opacity
            else:
                self.source_color = '#1f77b4'
                self.source_opacity = 1.0
                self.intermediate_color = '#2ca02c'
                self.intermediate_opacity = 1.0
        else:
            # Defaults
            self.source_color = '#1f77b4'  # Blue
            self.source_opacity = 1.0
            self.intermediate_color = '#2ca02c'  # Green
            self.intermediate_opacity = 1.0
        
        # Parse target color
        target_hex, target_opacity = parse_color_to_hex_opacity(target_color or '#d62728')
        self.target_color = target_hex
        self.target_opacity = target_opacity
        
        # Parse link/edge color - extract hex and opacity for network edges
        link_hex, link_opacity = parse_color_to_hex_opacity(link_color or 'rgba(100,100,100,0.5)')
        self.link_color = link_color or 'rgba(100,100,100,0.5)'  # Keep original for Sankey
        self.edge_color = link_hex  # Hex color for network edges
        self.edge_opacity = link_opacity  # Opacity for network edges
        
        # Parse highlight color - default to a saturated orange that stays
        # clearly visible against the white canvas (the old light-yellow
        # default was nearly invisible)
        highlight_hex, highlight_opacity = parse_color_to_hex_opacity(highlight_color or '#FF9800')
        self.highlight_color = highlight_hex  # Hex color for highlighted elements
        self.highlight_opacity = highlight_opacity  # Opacity for highlighted elements
        
        # Reciprocal edges style
        self.straight_reciprocal_edges = straight_reciprocal_edges
        
        # Create node_color array for compatibility with internal methods
        self.node_color = [self.source_color, self.intermediate_color]
        
        # Store custom node colors (will be loaded later)
        self.node_colors_input = node_colors
        self.custom_node_colors = None  # Will be populated in _load_custom_colors()
        self.custom_edge_colors = None  # Will be populated if edge-list has color column
        self.custom_node_groups = None  # Will be populated if edge-list has source/target_group columns
        self._csv_group_labels = {}  # safe custom group name -> raw edge-list label
        
        self.network_layout = network_layout
        self.showfig = showfig
        
        # Data storage
        self.path_df = None
        self.conn_df = None
        self.G_network = None
        
        # Skip data loading for empty network generation
        if self.generate_empty_network:
            # Generate unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Set default output folder and filename for empty network
            if self.output_folder is None:
                self.output_folder = './empty_network'
                self.base_filename = f'empty_network_{timestamp}'
            else:
                # The UI passes a per-run folder that already ends in a
                # timestamp (for example, ``plot-network_empty_network_...``).
                # Reuse that name so the generated file does not repeat the
                # same timestamp. Standalone callers with an ordinary folder
                # still receive the timestamped filename used historically.
                folder_name = os.path.basename(self.output_folder.rstrip(os.sep))
                if re.search(r'_\d{8}_\d{6}$', folder_name):
                    self.base_filename = folder_name
                else:
                    self.base_filename = f'{folder_name}_{timestamp}'
            os.makedirs(self.output_folder, exist_ok=True)
            return
        
        # Load and validate data
        self._load_data()
        # Accept 'path' / 'path_str' columns as aliases of 'path_block'
        # (FindAllPath saves the path column as 'path' in its CSV output)
        if self.path_df is not None and 'path_block' not in self.path_df.columns:
            for alias in ('path', 'path_str'):
                if alias in self.path_df.columns:
                    self.path_df = self.path_df.rename(columns={alias: 'path_block'})
                    self._vprint(f"Renamed column '{alias}' to 'path_block'")
                    break
        self._validate_data()
        
        # Load custom colors if provided
        if self.node_colors_input is not None:
            self._load_custom_colors()
    
    def _vprint(self, *args, **kwargs):
        """
        Print helper that respects verbose setting.
        Only prints if self.verbose is True.
        
        Parameters
        ----------
        *args : positional arguments
            Arguments to pass to print()
        **kwargs : keyword arguments
            Keyword arguments to pass to print()
        """
        if self.verbose:
            print(*args, **kwargs)

    def _progress(self, step: int, total: int, label: str = ''):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Emitted only when
        the caller opted in via ``progress_total`` (nested pipeline calls
        keep the previous quiet behavior).
        """
        if self.verbose and self.progress_total:
            print(f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip(),
                  flush=True)
        
    def _normalize_column_names(self):
        """
        Normalize column names to standard internal names.
        
        Handles aliases for:
        - connection_ratios: ratio, ratios, connection_ratio, conn_ratio
        - traversal_probabilities: probability, probabilities, prob, traversal_probability, trav_prob
        """
        if self.path_df is None:
            return
            
        # Define aliases mapping
        aliases = {
            'connection_ratios': ['ratio', 'ratios', 'connection_ratio', 'conn_ratio'],
            'traversal_probabilities': ['probability', 'probabilities', 'prob', 'traversal_probability', 'trav_prob']
        }
        
        # Create rename dictionary
        rename_dict = {}
        columns_lower = {str(col).lower(): str(col) for col in self.path_df.columns}
        
        for standard_name, alias_list in aliases.items():
            # If standard name already exists, skip
            if standard_name in self.path_df.columns:
                continue
                
            # Check for aliases (case-insensitive)
            for alias in alias_list:
                if alias.lower() in columns_lower:
                    original_col = columns_lower[alias.lower()]
                    rename_dict[original_col] = standard_name
                    self._vprint(f"  Renaming column '{original_col}' to '{standard_name}'")
                    break  # Use the first matching alias found
        
        # Apply renaming
        if rename_dict:
            self.path_df = self.path_df.rename(columns=rename_dict)

    def _load_data(self):
        """Load pathway data from file or DataFrame. Supports 1. connection matrix input, 2. [source, target, weight, ratio(optional), probability(optional)] 3-column edge list, and 3. path blocks."""
        if isinstance(self.path_file, pd.DataFrame):
            df = self.path_file.copy()
            edge_list_colsets = [
                {"source", "target", "weight"},
                {"from", "to", "weight"},
                {"pre", "post", "weight"},
            ]
            def has_prefixed_cols(cols):
                # Convert columns to strings to handle numeric column names
                str_cols = [str(c) for c in cols]
                has_pre = any(c.endswith('_pre') for c in str_cols)
                has_post = any(c.endswith('_post') for c in str_cols)
                has_weight = 'weight' in str_cols
                return has_pre and has_post and has_weight
            is_edge_list = any(set(df.columns) == s for s in edge_list_colsets) or has_prefixed_cols(df.columns)
            # Detect connection matrix: 2D numeric DataFrame, not edge-list
            is_numeric_matrix = (
                not is_edge_list and
                df.ndim == 2 and
                df.shape[0] > 1 and df.shape[1] > 1 and
                all(np.issubdtype(dtype, np.number) for dtype in df.dtypes)
            )
            if is_numeric_matrix:
                self._vprint("✓ Recognized input format: connection matrix ({}x{} DataFrame)".format(df.shape[0], df.shape[1]))
                # If index/columns are not all strings, auto-generate node names
                if not (all(isinstance(x, str) for x in df.index) and all(isinstance(x, str) for x in df.columns)):
                    n, m = df.shape
                    row_names = [f"N{i}" for i in range(n)]
                    col_names = [f"N{j}" for j in range(m)]
                    df.index = row_names
                    df.columns = col_names
                # Convert matrix to edge list
                edge_list = []
                for src in df.index:
                    for tgt in df.columns:
                        val = df.at[src, tgt]
                        if pd.notna(val) and val != 0:
                            edge_list.append({"source": src, "target": tgt, "weight": val})
                self.path_df = pd.DataFrame(edge_list)
                if self.output_folder is None:
                    self.output_folder = './selected_paths'
                    self.base_filename = 'selected_paths_matrix'
                else:
                    self.base_filename = os.path.basename(self.output_folder.rstrip(os.sep))
            else:
                if is_edge_list:
                    self._vprint("✓ Recognized input format: edge-list DataFrame (columns: {} )".format(list(df.columns)))
                else:
                    self._vprint("✓ Recognized input format: generic DataFrame (columns: {} )".format(list(df.columns)))
                self.path_df = df
                if self.output_folder is None:
                    self.output_folder = './selected_paths'
                    self.base_filename = 'selected_paths'
                else:
                    self.base_filename = os.path.basename(self.output_folder.rstrip(os.sep))
        else:
            # Track if file picker was used (affects sheet selection behavior)
            file_picker_used = False
            
            # Check if file exists, if not or if path_file is None/empty, prompt user to select
            if self.path_file is None or self.path_file == '' or not os.path.exists(self.path_file):
                if self.path_file and not os.path.exists(self.path_file):
                    self._vprint(f"⚠️ Path file not found: {self.path_file}")
                self._vprint("Please select a path file...")
                self.path_file = self._select_file()
                if self.path_file is None:
                    raise ValueError("No file selected. Cannot proceed without pathway data.")
                file_picker_used = True
            
            file_ext = Path(self.path_file).suffix.lower()
            
            if file_ext == '.csv':
                self._vprint(f"Loading CSV file: {self.path_file}")
                self.path_df = pd.read_csv(self.path_file)
                # For CSV files, ignore sheet_name parameter
                if self.sheet_name:
                    self._vprint(f"  Note: sheet_name '{self.sheet_name}' ignored for CSV files")
                    
            elif file_ext in ['.xlsx', '.xls']:
                self._vprint(f"Loading Excel file: {self.path_file}")
                excel_file = pd.ExcelFile(self.path_file)
                
                # If file picker was used, always ask user to confirm/select sheet
                # Otherwise, auto-select if sheet_name is None
                if file_picker_used:
                    # File picker was used - always ask for confirmation
                    self.sheet_name = self._select_sheet(excel_file, auto_confirm=True)
                    if self.sheet_name is None:
                        raise ValueError("No sheet selected. Cannot proceed without sheet selection.")
                elif self.sheet_name is None:
                    # File path provided but no sheet_name - auto-select quietly
                    self.sheet_name = self._select_sheet(excel_file, auto_confirm=False)
                    if self.sheet_name is None:
                        raise ValueError("No sheet selected. Cannot proceed without sheet selection.")
                # else: sheet_name was explicitly provided, use it
                
                self.path_df = pd.read_excel(self.path_file, sheet_name=self.sheet_name)
                self._vprint(f"  Loaded sheet: '{self.sheet_name}'")
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .xlsx, or .xls")
            
            # Check if loaded data is a connection matrix
            # Detect connection matrix: 2D numeric DataFrame, not edge-list
            # Edge list usually has 'source', 'target', 'weight' or similar columns
            edge_list_colsets = [
                {"source", "target", "weight"},
                {"from", "to", "weight"},
                {"pre", "post", "weight"},
            ]
            def has_prefixed_cols(cols):
                str_cols = [str(c) for c in cols]
                has_pre = any(c.endswith('_pre') for c in str_cols)
                has_post = any(c.endswith('_post') for c in str_cols)
                has_weight = 'weight' in str_cols
                return has_pre and has_post and has_weight
                
            is_edge_list = any(set(self.path_df.columns) == s for s in edge_list_colsets) or has_prefixed_cols(self.path_df.columns)
            
            # Check for path format
            path_cols = ['path_block', 'weights']
            has_path_format = all(col in self.path_df.columns for col in path_cols)
            
            is_numeric_matrix = (
                not is_edge_list and
                not has_path_format and
                self.path_df.ndim == 2 and
                self.path_df.shape[0] > 1 and self.path_df.shape[1] > 1 and
                all(np.issubdtype(dtype, np.number) for dtype in self.path_df.dtypes)
            )
            
            if is_numeric_matrix:
                self._vprint("✓ Recognized input format: connection matrix ({}x{} DataFrame)".format(self.path_df.shape[0], self.path_df.shape[1]))
                df = self.path_df
                # If index/columns are not all strings, auto-generate node names
                if not (all(isinstance(x, str) for x in df.index) and all(isinstance(x, str) for x in df.columns)):
                    # If loaded from CSV without index_col, the first column might be the index
                    # Heuristic: if first column is object/string and others are numeric, set it as index
                    if df.shape[1] > 1 and df.iloc[:, 0].dtype == object:
                        self._vprint("  Using first column as index")
                        df = df.set_index(df.columns[0])
                        self.path_df = df
                    else:
                        n, m = df.shape
                        row_names = [f"N{i}" for i in range(n)]
                        col_names = [f"N{j}" for j in range(m)]
                        df.index = row_names
                        df.columns = col_names
                        self.path_df = df
                
                # Convert matrix to edge list
                edge_list = []
                for src in df.index:
                    for tgt in df.columns:
                        val = df.at[src, tgt]
                        if pd.notna(val) and val != 0:
                            edge_list.append({"source": src, "target": tgt, "weight": val})
                self.path_df = pd.DataFrame(edge_list)
                self._vprint(f"  Converted matrix to {len(self.path_df)} edges")

            # Set default output folder relative to input file
            if self.output_folder is None:
                input_dir = os.path.dirname(os.path.abspath(self.path_file))
                input_filename = os.path.splitext(os.path.basename(self.path_file))[0]
                self.output_folder = os.path.join(input_dir, input_filename + '_figure')
                self.base_filename = input_filename  # Store for output file naming
            else:
                # If custom output_folder provided, extract folder name as base
                self.base_filename = os.path.basename(self.output_folder.rstrip(os.sep))
        
        # Normalize column names (handle aliases)
        self._normalize_column_names()
        
        # Create output folder
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _select_file(self):
        """
        Interactive file selection dialog with cross-platform support.
        Uses the fastest available GUI backend (PyQt5 > wxPython > tkinter).
        
        Returns
        -------
        str or None
            Path to selected file, or None if cancelled
            
        Notes
        -----
        - Tries PyQt5 first (fastest, most native)
        - Falls back to wxPython (fast alternative)
        - Falls back to tkinter (slower, but always available)
        - Works on Windows, macOS, and Linux
        """
        # Try PyQt5 first (fastest and most native-looking)
        try:
            from PyQt5.QtWidgets import QApplication, QFileDialog
            import sys
            
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            self._vprint("Please select a path file...")
            
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Pathway Data File",
                os.getcwd(),
                "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
            )
            
            # Process events to ensure clean exit
            app.processEvents()
            
            if file_path:
                file_path = os.path.normpath(file_path)
                self._vprint(f"✓ Selected file: {file_path}")
                return file_path
            else:
                self._vprint("✗ No file selected")
                return None
                
        except ImportError:
            pass
        
        # Try PyQt6 (similar to PyQt5)
        try:
            from PyQt6.QtWidgets import QApplication, QFileDialog  # type: ignore
            import sys
            
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            self._vprint("Please select a path file...")
            
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Pathway Data File",
                os.getcwd(),
                "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
            )
            
            app.processEvents()
            
            if file_path:
                file_path = os.path.normpath(file_path)
                self._vprint(f"✓ Selected file: {file_path}")
                return file_path
            else:
                self._vprint("✗ No file selected")
                return None
                
        except ImportError:
            pass
        
        # Try wxPython (good performance, native look)
        try:
            import wx  # type: ignore
            
            app = wx.App(False)
            
            wildcard = "Excel files (*.xlsx;*.xls)|*.xlsx;*.xls|CSV files (*.csv)|*.csv|All files (*.*)|*.*"
            
            self._vprint("Please select a path file...")
            
            dialog = wx.FileDialog(
                None,
                "Select Pathway Data File",
                defaultDir=os.getcwd(),
                wildcard=wildcard,
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
            )
            
            if dialog.ShowModal() == wx.ID_OK:
                file_path = dialog.GetPath()
                dialog.Destroy()
                app.Destroy()
                
                file_path = os.path.normpath(file_path)
                self._vprint(f"✓ Selected file: {file_path}")
                return file_path
            else:
                dialog.Destroy()
                app.Destroy()
                self._vprint("✗ No file selected")
                return None
                
        except ImportError:
            pass
        
        # Fallback to tkinter (slower but widely available)
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            
            try:
                root.attributes('-topmost', True)
            except tk.TclError:
                pass
            
            root.lift()
            root.focus_force()
            root.update()
            
            initial_dir = os.getcwd()
            
            self._vprint("Please select a path file...")
            
            file_path = filedialog.askopenfilename(
                parent=root,
                title='Select Pathway Data File',
                filetypes=[
                    ('Excel files', '*.xlsx *.xls'),
                    ('CSV files', '*.csv'),
                    ('All files', '*.*')
                ],
                initialdir=initial_dir
            )
            
            root.update()
            root.quit()
            root.destroy()
            
            if file_path:
                file_path = os.path.normpath(file_path)
                self._vprint(f"✓ Selected file: {file_path}")
                return file_path
            else:
                self._vprint("✗ No file selected")
                return None
                
        except ImportError:
            self._vprint("⚠️ No GUI library available (tried PyQt5, PyQt6, wxPython, tkinter)")
            self._vprint("   Install one of:")
            self._vprint("   - pip install PyQt5  (recommended - fastest)")
            self._vprint("   - pip install PyQt6")
            self._vprint("   - pip install wxPython")
            self._vprint("   - python3-tk (tkinter)")
            return None
        except Exception as e:
            self._vprint(f"⚠️ Error opening file dialog: {e}")
            return None
    
    def _select_sheet(self, excel_file, auto_confirm=False):
        """
        Interactive sheet selection for Excel files.
        
        Parameters
        ----------
        excel_file : pd.ExcelFile
            Excel file object
        auto_confirm : bool, optional
            If True, always ask user to confirm even auto-detected sheets
            If False, silently auto-select common sheets (default)
            
        Returns
        -------
        str or None
            Selected sheet name, or None if cancelled
            
        Notes
        -----
        When auto_confirm=True (file picker was used):
        - Shows auto-detected sheet and asks for confirmation
        - User can accept or choose different sheet
        
        When auto_confirm=False (file path provided):
        - Silently auto-selects common sheet names
        - Only prompts if no common names found
        """
        sheet_names = excel_file.sheet_names
        
        # If only one sheet, use it automatically (no confirmation needed)
        if len(sheet_names) == 1:
            sheet_name = sheet_names[0]
            if auto_confirm:
                self._vprint(f"  Only one sheet found: '{sheet_name}'")
                confirm = self._confirm_sheet_selection(sheet_name, sheet_names, excel_file)
                return confirm if confirm else sheet_name
            else:
                self._vprint(f"  Only one sheet found: '{sheet_name}'")
                return sheet_name
        
        # Try to auto-detect common sheet names
        priority_sheets = ['path_type', 'path_bodyId', 'path_block', 'paths']
        auto_detected = None
        for sheet in priority_sheets:
            if sheet in sheet_names:
                auto_detected = sheet
                break
        
        # If auto-detected and confirmation requested, ask user
        if auto_detected and auto_confirm:
            self._vprint(f"  Auto-detected sheet: '{auto_detected}'")
            confirmed = self._confirm_sheet_selection(auto_detected, sheet_names, excel_file)
            return confirmed if confirmed else auto_detected
        
        # If auto-detected and no confirmation needed, use it
        elif auto_detected:
            self._vprint(f"  Auto-selected sheet: '{auto_detected}'")
            return auto_detected
        
        # No auto-detection possible - always ask user
        return self._prompt_sheet_selection(sheet_names, excel_file)
    
    def _confirm_sheet_selection(self, suggested_sheet, all_sheets, excel_file):
        """
        Show sheet selection dialog with auto-detected sheet pre-selected.
        
        Parameters
        ----------
        suggested_sheet : str
            Auto-detected sheet name (will be pre-selected)
        all_sheets : list
            List of all available sheet names
        excel_file : pd.ExcelFile
            Excel file object
            
        Returns
        -------
        str or None
            Selected sheet name, or None if cancelled
        """
        # Use the combined selection dialog with suggested sheet pre-selected
        return self._prompt_sheet_selection(all_sheets, excel_file, default_sheet=suggested_sheet)
    
    def _prompt_sheet_selection(self, sheet_names, excel_file, default_sheet=None):
        """
        Prompt user to select a sheet from available sheets using GUI dialog.
        Uses the fastest available GUI backend (PyQt5 > wxPython > tkinter).
        
        Parameters
        ----------
        sheet_names : list
            List of available sheet names
        excel_file : pd.ExcelFile
            Excel file object
        default_sheet : str, optional
            Sheet name to pre-select in the dialog (e.g., auto-detected sheet)
            
        Returns
        -------
        str or None
            Selected sheet name, or None if cancelled
        """
        # Try PyQt5 first (fastest)
        result = self._prompt_sheet_pyqt5(sheet_names, excel_file, default_sheet)
        if result is not False:
            return result
        
        # Try PyQt6
        result = self._prompt_sheet_pyqt6(sheet_names, excel_file, default_sheet)
        if result is not False:
            return result
        
        # Try wxPython
        result = self._prompt_sheet_wx(sheet_names, excel_file, default_sheet)
        if result is not False:
            return result
        
        # Fallback to tkinter
        result = self._prompt_sheet_tkinter(sheet_names, excel_file, default_sheet)
        if result is not False:
            return result
        
        # If all fail, use terminal
        return self._prompt_sheet_selection_terminal(sheet_names, excel_file, default_sheet)
    
    def _prompt_sheet_pyqt5(self, sheet_names, excel_file, default_sheet=None):
        """PyQt5 implementation - fastest and most responsive"""
        try:
            from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QListWidgetItem
            import sys
            
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            dialog = QDialog()
            dialog.setWindowTitle("Select Excel Sheet")
            dialog.setMinimumWidth(700)
            dialog.setMinimumHeight(400)
            
            layout = QVBoxLayout()
            
            # Title
            if default_sheet:
                title = QLabel(f"<b>Auto-detected: '{default_sheet}'</b>")
                subtitle = QLabel("Press OK to use it, or select a different sheet:")
                layout.addWidget(title)
                layout.addWidget(subtitle)
            else:
                title = QLabel("<b>Select a sheet:</b>")
                layout.addWidget(title)
            
            # List widget
            list_widget = QListWidget()
            list_widget.setAlternatingRowColors(True)
            
            default_idx = 0
            for idx, sheet in enumerate(sheet_names):
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet, nrows=0)
                    row_count = len(pd.read_excel(excel_file, sheet_name=sheet))
                    col_count = len(df.columns)
                    
                    if sheet == default_sheet:
                        text = f"✓ {sheet} ({row_count} rows, {col_count} cols) [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet} ({row_count} rows, {col_count} cols)"
                except:
                    if sheet == default_sheet:
                        text = f"✓ {sheet} [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet}"
                
                item = QListWidgetItem(text)
                list_widget.addItem(item)
            
            list_widget.setCurrentRow(default_idx)
            list_widget.itemDoubleClicked.connect(dialog.accept)
            layout.addWidget(list_widget)
            
            # Tip label
            tip = QLabel("💡 Tip: Double-click or press OK to select")
            tip.setStyleSheet("color: gray;")
            layout.addWidget(tip)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            ok_button = QPushButton("OK")
            ok_button.setDefault(True)
            ok_button.clicked.connect(dialog.accept)
            button_layout.addWidget(ok_button)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_button)
            
            layout.addLayout(button_layout)
            dialog.setLayout(layout)
            
            # Show dialog
            result = dialog.exec_()
            
            if result == QDialog.Accepted:
                selected_idx = list_widget.currentRow()
                selected_sheet = sheet_names[selected_idx]
                
                if default_sheet and selected_sheet == default_sheet:
                    self._vprint(f"✓ Using auto-detected sheet: '{selected_sheet}'")
                else:
                    self._vprint(f"✓ Selected sheet: '{selected_sheet}'")
                
                app.processEvents()
                return selected_sheet
            else:
                self._vprint("✗ Sheet selection cancelled")
                app.processEvents()
                return None
                
        except ImportError:
            return False  # Not available, try next backend
        except Exception as e:
            self._vprint(f"⚠️ PyQt5 error: {e}")
            return False
    
    def _prompt_sheet_pyqt6(self, sheet_names, excel_file, default_sheet=None):
        """PyQt6 implementation - similar to PyQt5"""
        try:
            from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QListWidgetItem  # type: ignore
            import sys
            
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            dialog = QDialog()
            dialog.setWindowTitle("Select Excel Sheet")
            dialog.setMinimumWidth(700)
            dialog.setMinimumHeight(400)
            
            layout = QVBoxLayout()
            
            # Title
            if default_sheet:
                title = QLabel(f"<b>Auto-detected: '{default_sheet}'</b>")
                subtitle = QLabel("Press OK to use it, or select a different sheet:")
                layout.addWidget(title)
                layout.addWidget(subtitle)
            else:
                title = QLabel("<b>Select a sheet:</b>")
                layout.addWidget(title)
            
            # List widget
            list_widget = QListWidget()
            list_widget.setAlternatingRowColors(True)
            
            default_idx = 0
            for idx, sheet in enumerate(sheet_names):
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet, nrows=0)
                    row_count = len(pd.read_excel(excel_file, sheet_name=sheet))
                    col_count = len(df.columns)
                    
                    if sheet == default_sheet:
                        text = f"✓ {sheet} ({row_count} rows, {col_count} cols) [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet} ({row_count} rows, {col_count} cols)"
                except:
                    if sheet == default_sheet:
                        text = f"✓ {sheet} [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet}"
                
                item = QListWidgetItem(text)
                list_widget.addItem(item)
            
            list_widget.setCurrentRow(default_idx)
            list_widget.itemDoubleClicked.connect(dialog.accept)
            layout.addWidget(list_widget)
            
            # Tip label
            tip = QLabel("💡 Tip: Double-click or press OK to select")
            tip.setStyleSheet("color: gray;")
            layout.addWidget(tip)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            ok_button = QPushButton("OK")
            ok_button.setDefault(True)
            ok_button.clicked.connect(dialog.accept)
            button_layout.addWidget(ok_button)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_button)
            
            layout.addLayout(button_layout)
            dialog.setLayout(layout)
            
            # Show dialog
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                selected_idx = list_widget.currentRow()
                selected_sheet = sheet_names[selected_idx]
                
                if default_sheet and selected_sheet == default_sheet:
                    self._vprint(f"✓ Using auto-detected sheet: '{selected_sheet}'")
                else:
                    self._vprint(f"✓ Selected sheet: '{selected_sheet}'")
                
                app.processEvents()
                return selected_sheet
            else:
                self._vprint("✗ Sheet selection cancelled")
                app.processEvents()
                return None
                
        except ImportError:
            return False  # Not available, try next backend
        except Exception as e:
            self._vprint(f"⚠️ PyQt6 error: {e}")
            return False
    
    def _prompt_sheet_wx(self, sheet_names, excel_file, default_sheet=None):
        """wxPython implementation - fast alternative"""
        try:
            import wx  # type: ignore
            
            app = wx.App(False)
            
            # Build sheet info list
            sheet_info = []
            default_idx = 0
            for idx, sheet in enumerate(sheet_names):
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet, nrows=0)
                    row_count = len(pd.read_excel(excel_file, sheet_name=sheet))
                    col_count = len(df.columns)
                    
                    if sheet == default_sheet:
                        text = f"✓ {sheet} ({row_count} rows, {col_count} cols) [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet} ({row_count} rows, {col_count} cols)"
                except:
                    if sheet == default_sheet:
                        text = f"✓ {sheet} [Suggested]"
                        default_idx = idx
                    else:
                        text = f"  {sheet}"
                
                sheet_info.append(text)
            
            if default_sheet:
                message = f"Auto-detected: '{default_sheet}'\n\nPress OK to use it, or select a different sheet:"
            else:
                message = "Select a sheet:"
            
            dialog = wx.SingleChoiceDialog(
                None,
                message,
                "Select Excel Sheet",
                sheet_info,
                wx.CHOICEDLG_STYLE
            )
            
            dialog.SetSelection(default_idx)
            
            if dialog.ShowModal() == wx.ID_OK:
                selected_idx = dialog.GetSelection()
                selected_sheet = sheet_names[selected_idx]
                dialog.Destroy()
                app.Destroy()
                
                if default_sheet and selected_sheet == default_sheet:
                    self._vprint(f"✓ Using auto-detected sheet: '{selected_sheet}'")
                else:
                    self._vprint(f"✓ Selected sheet: '{selected_sheet}'")
                
                return selected_sheet
            else:
                dialog.Destroy()
                app.Destroy()
                self._vprint("✗ Sheet selection cancelled")
                return None
                
        except ImportError:
            return False  # Not available, try next backend
        except Exception as e:
            self._vprint(f"⚠️ wxPython error: {e}")
            return False
    
    def _prompt_sheet_tkinter(self, sheet_names, excel_file, default_sheet=None):
        """Tkinter implementation - slower but widely available fallback"""
        try:
            import tkinter as tk
            from tkinter import simpledialog
        except ImportError:
            return False
        
        try:
            # Create a custom dialog for sheet selection
            class SheetSelectionDialog(simpledialog.Dialog):
                def __init__(self, parent, title, sheet_names, excel_file, default_sheet=None):
                    self.sheet_names = sheet_names
                    self.excel_file = excel_file
                    self.default_sheet = default_sheet
                    self.selected_sheet = None
                    super().__init__(parent, title)
                
                def body(self, master):
                    master.configure(padx=15, pady=15)
                    
                    # Title
                    if self.default_sheet:
                        title_text = f"Auto-detected: '{self.default_sheet}'"
                        title_label = tk.Label(master, text=title_text, font=('Helvetica', 12, 'bold'))
                        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky='w')
                        
                        subtitle = tk.Label(master, text="Press OK to use it, or select a different sheet:", font=('Helvetica', 9))
                        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky='w')
                        current_row = 2
                    else:
                        title_label = tk.Label(master, text="Select a sheet:", font=('Helvetica', 12, 'bold'))
                        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky='w')
                        current_row = 1
                    
                    # List frame
                    list_frame = tk.Frame(master, relief=tk.SUNKEN, borderwidth=1)
                    list_frame.grid(row=current_row, column=0, columnspan=2, pady=10, sticky='nsew')
                    
                    master.grid_rowconfigure(current_row, weight=1)
                    master.grid_columnconfigure(0, weight=1)
                    
                    # Listbox with scrollbar
                    scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
                    self.listbox = tk.Listbox(
                        list_frame, 
                        yscrollcommand=scrollbar.set, 
                        width=70, 
                        height=min(10, max(4, len(self.sheet_names))),
                        selectmode=tk.SINGLE
                    )
                    
                    scrollbar.config(command=self.listbox.yview)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
                    
                    # Add sheet names
                    default_idx = 0
                    for idx, sheet in enumerate(self.sheet_names):
                        try:
                            df = pd.read_excel(self.excel_file, sheet_name=sheet, nrows=0)
                            row_count = len(pd.read_excel(self.excel_file, sheet_name=sheet))
                            col_count = len(df.columns)
                            
                            if sheet == self.default_sheet:
                                text = f"✓ {sheet} ({row_count} rows, {col_count} cols) [Suggested]"
                                default_idx = idx
                            else:
                                text = f"  {sheet} ({row_count} rows, {col_count} cols)"
                        except:
                            if sheet == self.default_sheet:
                                text = f"✓ {sheet} [Suggested]"
                                default_idx = idx
                            else:
                                text = f"  {sheet}"
                        
                        self.listbox.insert(tk.END, text)
                    
                    self.listbox.select_set(default_idx)
                    self.listbox.see(default_idx)
                    self.listbox.bind('<Double-Button-1>', lambda e: self.ok())
                    self.listbox.bind('<Return>', lambda e: self.ok())
                    
                    # Tip
                    tip_label = tk.Label(master, text="💡 Tip: Double-click or press Enter/OK to select", font=('Helvetica', 9))
                    tip_label.grid(row=current_row+1, column=0, columnspan=2, pady=(5, 0))
                    
                    return self.listbox
                
                def apply(self):
                    selection = self.listbox.curselection()
                    if selection:
                        self.selected_sheet = self.sheet_names[selection[0]]
            
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            
            try:
                root.attributes('-topmost', True)
            except tk.TclError:
                pass
            
            root.lift()
            root.focus_force()
            root.update()
            
            dialog = SheetSelectionDialog(root, "Select Excel Sheet", sheet_names, excel_file, default_sheet)
            selected_sheet = dialog.selected_sheet
            
            root.update()
            root.quit()
            root.destroy()
            
            if selected_sheet:
                if default_sheet and selected_sheet == default_sheet:
                    self._vprint(f"✓ Using auto-detected sheet: '{selected_sheet}'")
                else:
                    self._vprint(f"✓ Selected sheet: '{selected_sheet}'")
                return selected_sheet
            else:
                self._vprint("✗ Sheet selection cancelled")
                return None
                
        except Exception as e:
            self._vprint(f"⚠️ Tkinter error: {e}")
            return False
    
    def _prompt_sheet_selection_terminal(self, sheet_names, excel_file, default_sheet=None):
        """
        Terminal fallback for sheet selection (when tkinter unavailable).
        
        Parameters
        ----------
        sheet_names : list
            List of available sheet names
        excel_file : pd.ExcelFile
            Excel file object
        default_sheet : str, optional
            Sheet name to suggest as default
            
        Returns
        -------
        str or None
            Selected sheet name, or None if cancelled
        """
        self._vprint("\n" + "="*60)
        if default_sheet:
            self._vprint(f"Auto-detected sheet: '{default_sheet}'")
            self._vprint("Select a sheet (or press Enter to use auto-detected):")
        else:
            self._vprint("Multiple sheets found. Please select one:")
        self._vprint("="*60)
        
        for idx, sheet in enumerate(sheet_names, 1):
            # Try to get row count for each sheet
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet, nrows=0)
                row_count = len(pd.read_excel(excel_file, sheet_name=sheet))
                col_count = len(df.columns)
                marker = " ✓ [Auto-detected]" if sheet == default_sheet else ""
                self._vprint(f"  [{idx}] {sheet:30s} ({row_count} rows, {col_count} cols){marker}")
            except:
                marker = " ✓ [Auto-detected]" if sheet == default_sheet else ""
                self._vprint(f"  [{idx}] {sheet}{marker}")
        self._vprint("="*60)
        
        # Get user input
        while True:
            try:
                if default_sheet:
                    prompt = f"Enter number (1-{len(sheet_names)}), sheet name, or [Enter] for '{default_sheet}': "
                else:
                    prompt = f"Enter number (1-{len(sheet_names)}) or sheet name: "
                
                choice = input(prompt).strip()
                
                # Enter key - use default if available
                if choice == '' and default_sheet:
                    self._vprint(f"✓ Using auto-detected sheet: '{default_sheet}'")
                    return default_sheet
                
                # Try as number first
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(sheet_names):
                        selected = sheet_names[idx - 1]
                        self._vprint(f"✓ Selected: '{selected}'")
                        return selected
                    else:
                        self._vprint(f"⚠️ Invalid number. Please enter 1-{len(sheet_names)}")
                        
                # Try as sheet name
                elif choice in sheet_names:
                    self._vprint(f"✓ Selected: '{choice}'")
                    return choice
                    
                # Allow cancel
                elif choice.lower() in ['q', 'quit', 'cancel', 'exit']:
                    self._vprint("✗ Selection cancelled")
                    return None
                    
                else:
                    self._vprint(f"⚠️ Invalid input. Enter a number, sheet name, or 'q' to cancel")
                    
            except KeyboardInterrupt:
                self._vprint("\n✗ Selection cancelled")
                return None
            except EOFError:
                self._vprint("\n✗ No input available")
                return None
        
    def _validate_data(self):
        """
        Validate that required columns exist in the data.
        
        Supports two input formats:
        
        1. **Path-based format** (original):
           Required: 'path_block', 'weights'
           Optional: 'connection_ratios', 'traversal_probabilities', 'layer'
           
        2. **Edge-list format** (new - simple network):
           Required: source + target + weight columns
           Column names can be:
           - 'source' / 'target' / 'weight'
           - '{prefix}_pre' / '{prefix}_post' / 'weight' (e.g., 'bodyId_pre', 'bodyId_post')
           - 'from' / 'to' / 'weight'
           - 'pre' / 'post' / 'weight'
           
           Optional (the expanded Edge List CSV export round-trips directly):
           - 'color': per-edge color (hex or rgba)
           - 'nt_type': neurotransmitter type
           - 'source_group' / 'target_group': endpoint node groups
             (source/intermediate/target) used for node classification
           - 'nt_group' / 'custom_groups': informational, ignored
           - 'ratio' / 'probability': mapped to the standard metric columns
           
        The system automatically detects the format and converts edge-list
        to internal path format for visualization.
        """
        # Check for path-based format
        path_cols = ['path_block', 'weights']
        has_path_format = all(col in self.path_df.columns for col in path_cols)
        
        if has_path_format:
            self._vprint(f"✓ Detected path-based format")
            self._vprint(f"  Loaded {len(self.path_df)} pathways from data")
            self._vprint(f"  Output folder: {self.output_folder}")
            return
        
        # Check for edge-list format
        self._vprint("Path-based format not detected, checking for edge-list format...")
        
        # Try to find source/target/weight columns
        source_col = self._find_column(['source', 'from', 'pre'], suffix='_pre')
        target_col = self._find_column(['target', 'to', 'post'], suffix='_post')
        weight_col = self._find_column(['weight', 'weights', 'synapse_count', 'count'])
        
        # Check for optional color column
        color_col = self._find_column(['color', 'edge_color', 'link_color'])
        
        if source_col and target_col and weight_col:
            self._vprint(f"✓ Detected edge-list format")
            self._vprint(f"  Source column: '{source_col}'")
            self._vprint(f"  Target column: '{target_col}'")
            self._vprint(f"  Weight column: '{weight_col}'")
            if color_col:
                self._vprint(f"  Color column: '{color_col}'")
            self._vprint(f"  Converting {len(self.path_df)} edges to path format...")
            
            # Convert edge-list to path format
            self._convert_edgelist_to_paths(source_col, target_col, weight_col, color_col)
            
            self._vprint(f"✓ Converted to {len(self.path_df)} paths")
            self._vprint(f"  Output folder: {self.output_folder}")
            return
        
        # Neither format found - raise error
        available_cols = list(self.path_df.columns)
        raise ValueError(
            f"Invalid data format. Could not find required columns.\n\n"
            f"Supported formats:\n"
            f"1. Path-based: 'path_block' + 'weights'\n"
            f"2. Edge-list: (source/from/pre/*_pre) + (target/to/post/*_post) + (weight/weights)\n"
            f"3. Connection matrix: Row as pre, Col as post (numeric values)\n\n"
            f"Available columns: {available_cols}\n\n"
            f"Examples:\n"
            f"  Path format:    path_block='A -> B -> C', weights='[10, 5]'\n"
            f"  Edge format:    source='A', target='B', weight=10\n"
            f"  BodyId format:  bodyId_pre=123, bodyId_post=456, weight=10\n"
        )
    
    def _load_custom_colors(self):
        """
        Load custom node colors from file, sheet, or DataFrame.
        
        Populates self.custom_node_colors as a dict mapping node name to color.
        Colors can be in hex (#RRGGBB) or rgba (rgba(r,g,b,a)) format.
        """
        node_colors_df = None
        
        # Determine source type and load data
        if isinstance(self.node_colors_input, pd.DataFrame):
            # Direct DataFrame input
            node_colors_df = self.node_colors_input.copy()
            self._vprint("Loading custom node colors from DataFrame...")
            
        elif isinstance(self.node_colors_input, str):
            # Could be sheet name or file path
            if os.path.exists(self.node_colors_input):
                # File path
                self._vprint(f"Loading custom node colors from file: {self.node_colors_input}")
                file_ext = Path(self.node_colors_input).suffix.lower()
                if file_ext == '.csv':
                    node_colors_df = pd.read_csv(self.node_colors_input)
                elif file_ext in ['.xlsx', '.xls']:
                    node_colors_df = pd.read_excel(self.node_colors_input)
                else:
                    raise ValueError(f"Unsupported file type for node_colors: {file_ext}")
            else:
                # Assume it's a sheet name in the main Excel file
                if isinstance(self.path_file, str) and Path(self.path_file).suffix.lower() in ['.xlsx', '.xls']:
                    self._vprint(f"Loading custom node colors from sheet '{self.node_colors_input}'...")
                    try:
                        node_colors_df = pd.read_excel(self.path_file, sheet_name=self.node_colors_input)
                    except Exception as e:
                        raise ValueError(
                            f"Could not load node_colors from sheet '{self.node_colors_input}': {e}\n"
                            f"Make sure the sheet exists in the Excel file."
                        )
                else:
                    raise ValueError(
                        f"node_colors is a string but not a valid file path: {self.node_colors_input}\n"
                        f"And path_file is not an Excel file, so can't interpret as sheet name."
                    )
        else:
            raise ValueError(f"node_colors must be a DataFrame, file path, or sheet name (str), got {type(self.node_colors_input)}")
        
        # Validate required columns (case-insensitive)
        cols_lower = {col.lower(): col for col in node_colors_df.columns}
        
        if 'node' not in cols_lower:
            raise ValueError(f"node_colors must have a 'node' column (case-insensitive). Found columns: {list(node_colors_df.columns)}")
        if 'color' not in cols_lower:
            raise ValueError(f"node_colors must have a 'color' column (case-insensitive). Found columns: {list(node_colors_df.columns)}")
        
        # Get actual column names
        node_col = cols_lower['node']
        color_col = cols_lower['color']
        
        # Build color mapping dictionary
        self.custom_node_colors = {}
        for _, row in node_colors_df.iterrows():
            node_name = str(row[node_col]).strip()
            color_value = str(row[color_col]).strip()
            
            # Validate color format (hex or rgba)
            if color_value.startswith('#') or color_value.startswith('rgb'):
                # Parse to validate and normalize
                hex_color, opacity = parse_color_to_hex_opacity(color_value)
                # Store as rgba format for consistency
                if opacity < 1.0:
                    # Convert hex back to rgb and add opacity
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    self.custom_node_colors[node_name] = f'rgba({r},{g},{b},{opacity})'
                else:
                    self.custom_node_colors[node_name] = hex_color
            else:
                self._vprint(f"⚠️ Warning: Invalid color format for node '{node_name}': {color_value}. Skipping.")
                continue
        
        self._vprint(f"✓ Loaded custom colors for {len(self.custom_node_colors)} nodes")
    
    def _find_column(self, candidates, suffix=None):
        """
        Find a column from list of candidates.
        
        Parameters
        ----------
        candidates : list
            List of possible column names
        suffix : str, optional
            Suffix to also check (e.g., '_pre', '_post')
            Matches any column ending with suffix that has a prefix
            (e.g., 'bodyId_pre', 'type_pre', 'neuron_pre')
            
        Returns
        -------
        str or None
            Found column name, or None
        """
        cols = self.path_df.columns
        
        # Check exact matches
        for candidate in candidates:
            if candidate in cols:
                return candidate
        
        # Check with suffix (e.g., 'bodyId_pre', 'type_pre', 'anything_pre')
        if suffix:
            for col in cols:
                # Must end with suffix AND have a prefix (not just '_pre' alone)
                if col.endswith(suffix) and len(col) > len(suffix) and '_' in col:
                    # Ensure there's actual content before the suffix
                    prefix = col[:col.rfind(suffix)]
                    if prefix and prefix != '_':  # Must have non-empty prefix
                        return col
        
        return None
    
    def _convert_edgelist_to_paths(self, source_col, target_col, weight_col, color_col=None):
        """
        Convert edge-list format to path-based format.
        
        Automatically detects and converts all numeric columns as additional metrics
        (e.g., ratio, probability, or any other numeric metric columns).
        
        Also recognizes the expanded Edge List CSV export columns: 'nt_type'
        (categorical, detected by name so all-empty columns are not misread
        as numeric), 'source_group'/'target_group' (endpoint node groups kept
        in self.custom_node_groups for node classification in build_network),
        and 'nt_group'/'custom_groups' (informational, skipped).

        The hover-info columns 'edge info' / 'source info' / 'target info'
        ({key:val; ...} cells written by the Edge List CSV export) restore
        the hover labels: 'edge info' becomes per-edge custom labels, while
        the source/target info cells are unioned per node (first value per
        key wins) into self.node_dataset_info — a node appears in many rows
        as source or target and keeps one merged info map.
        
        Parameters
        ----------
        source_col : str
            Name of source column
        target_col : str
            Name of target column
        weight_col : str
            Name of weight column (primary metric)
        color_col : str, optional
            Name of edge color column (hex or rgba format)
        """
        # Create path_block and weights columns
        paths = []
        weights = []
        
        # Store edge colors if provided
        custom_edge_colors = {}
        
        # Identify all numeric columns (excluding source, target, and color columns)
        exclude_cols = {source_col, target_col}
        if color_col:
            exclude_cols.add(color_col)
        
        # Columns of the expanded edge-list CSV export (Edge List CSV button).
        # Grouping is metadata, never a numeric metric: nt_group derives from
        # nt_type, source_group/target_group classify the endpoint nodes, and
        # custom_groups preserves in-HTML custom group names.
        nt_alias_names = {'nt_type', 'nt', 'neurotransmitter'}
        group_info_names = {'nt_group', 'custom_groups'}
        hover_info_names = {'edge info', 'source info', 'target info'}
        
        # Look for the nt_type and endpoint-group columns BY NAME - dtype is
        # unreliable: an all-empty nt_type column reads as NaN and looks numeric.
        nt_type_col = None
        source_group_col = None
        target_group_col = None
        edge_info_col = None
        source_info_col = None
        target_info_col = None
        for col in self.path_df.columns:
            if col in exclude_cols:
                continue
            col_lower = str(col).lower()
            if nt_type_col is None and col_lower in nt_alias_names:
                nt_type_col = col
            elif source_group_col is None and col_lower == 'source_group':
                source_group_col = col
            elif target_group_col is None and col_lower == 'target_group':
                target_group_col = col
            elif edge_info_col is None and col_lower == 'edge info':
                edge_info_col = col
            elif source_info_col is None and col_lower == 'source info':
                source_info_col = col
            elif target_info_col is None and col_lower == 'target info':
                target_info_col = col
        
        # Find all numeric columns that could be additional metrics
        numeric_cols = []
        additional_metric_names = []
        
        skip_cols = set(exclude_cols)
        skip_cols_lower = set(group_info_names) | set(hover_info_names)
        if nt_type_col:
            skip_cols.add(nt_type_col)
        if source_group_col:
            skip_cols.add(source_group_col)
        if target_group_col:
            skip_cols.add(target_group_col)
        for info_col in (edge_info_col, source_info_col, target_info_col):
            if info_col:
                skip_cols.add(info_col)
        
        for col in self.path_df.columns:
            if col in skip_cols or str(col).lower() in skip_cols_lower or col == weight_col:
                continue
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.path_df[col]):
                numeric_cols.append(col)
                # Store original column name for later use
                additional_metric_names.append(col)
        
        # Initialize storage for additional metrics
        additional_metrics = {col: [] for col in numeric_cols}
        nt_types_list = []
        # Endpoint group classification from the expanded export (node -> group)
        node_group_overrides = {}
        # Hover-info cells from the expanded export: per-edge labels and the
        # per-node union of every source/target info the node appears with.
        edge_labels_loaded = {}
        node_info_union = {}

        def _parse_info_cell(value):
            text = '' if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()
            if text.startswith('{') and text.endswith('}'):
                text = text[1:-1]
            info = {}
            for part in text.split(';'):
                part = part.strip()
                if not part or ':' not in part:
                    continue
                key, _, val = part.partition(':')
                info[key.strip()] = val.strip()
            return info
        
        self._vprint(f"  Detected numeric columns: {[weight_col] + numeric_cols}")
        if nt_type_col:
            self._vprint(f"  Detected neurotransmitter column: {nt_type_col}")
        if source_group_col or target_group_col:
            self._vprint(f"  Detected endpoint group columns: "
                         f"{[c for c in (source_group_col, target_group_col) if c]}")
        
        for idx, row in self.path_df.iterrows():
            source = str(row[source_col])
            target = str(row[target_col])
            weight = row[weight_col]
            
            # Create path block
            path_block = f"{source} -> {target}"
            paths.append(path_block)
            
            # Create weights list (primary metric)
            weights.append([weight])
            
            # Store additional metrics
            for col in numeric_cols:
                value = row[col] if pd.notna(row[col]) else 0
                additional_metrics[col].append([value])
            
            # Store nt_type if available (empty cells mean unknown)
            if nt_type_col:
                nt_val = row[nt_type_col] if pd.notna(row[nt_type_col]) else None
                if nt_val is not None and not str(nt_val).strip():
                    nt_val = None
                nt_types_list.append([nt_val])
            
            # Record endpoint node groups from the expanded export so
            # build_network can classify nodes (a plain edge list cannot
            # infer source/target identity)
            if source_group_col is not None:
                grp = row[source_group_col]
                if pd.notna(grp) and str(grp).strip():
                    node_group_overrides[source] = str(grp).strip()
            if target_group_col is not None:
                grp = row[target_group_col]
                if pd.notna(grp) and str(grp).strip():
                    node_group_overrides[target] = str(grp).strip()
            
            # Restore hover infos from the {key:val; ...} cells: edge info
            # becomes the edge's custom labels; source/target info merge
            # into one unique-union map per node (first value per key wins).
            if edge_info_col is not None:
                info = _parse_info_cell(row[edge_info_col])
                if info:
                    edge_labels_loaded[(source, target)] = info
            if source_info_col is not None:
                for key, val in _parse_info_cell(row[source_info_col]).items():
                    node_info_union.setdefault(source, {}).setdefault(key, val)
            if target_info_col is not None:
                for key, val in _parse_info_cell(row[target_info_col]).items():
                    node_info_union.setdefault(target, {}).setdefault(key, val)
            
            # Store edge color if provided
            if color_col and color_col in row and pd.notna(row[color_col]):
                edge_key = (source, target)
                color_value = str(row[color_col]).strip()
                # Validate and normalize color
                if color_value.startswith('#') or color_value.startswith('rgb'):
                    custom_edge_colors[edge_key] = color_value
        
        # Add new columns
        self.path_df['path_block'] = paths
        self.path_df['weights'] = weights
        
        # Restore hover infos for the renderer: per-edge custom labels and
        # the per-node unique-union info map (the cytoscape hover reads
        # self.edge_labels / self.node_dataset_info).
        if edge_labels_loaded:
            if not getattr(self, 'edge_labels', None):
                self.edge_labels = {}
            for edge_key, info in edge_labels_loaded.items():
                self.edge_labels.setdefault(edge_key, {}).update(info)
            self._vprint(f"  Restored hover labels for {len(edge_labels_loaded)} edge(s)")
        if node_info_union:
            for node, info in node_info_union.items():
                merged = self.node_dataset_info.setdefault(node, {})
                for key, val in info.items():
                    merged.setdefault(key, val)
            self._vprint(f"  Restored node info for {len(node_info_union)} node(s)")
        
        if nt_type_col:
            self.path_df['nt_types'] = nt_types_list
        
        # Add additional metric columns
        # Map common metric names to standard column names
        metric_mapping = {
            'ratio': 'connection_ratios',
            'connection_ratio': 'connection_ratios',
            'conn_ratio': 'connection_ratios',
            # already-normalized names (renamed by _normalize_column_names)
            'connection_ratios': 'connection_ratios',
            'probability': 'traversal_probabilities',
            'prob': 'traversal_probabilities',
            'traversal_probability': 'traversal_probabilities',
            'trav_prob': 'traversal_probabilities',
            # already-normalized names (renamed by _normalize_column_names)
            'traversal_probabilities': 'traversal_probabilities'
        }
        
        for col in numeric_cols:
            col_lower = col.lower()
            # Check if this matches a standard metric name
            if col_lower in metric_mapping:
                standard_col = metric_mapping[col_lower]
                self.path_df[standard_col] = additional_metrics[col]
                self._vprint(f"  ✓ Mapped '{col}' → '{standard_col}' for toggle support")
            else:
                # Keep original column name for custom metrics
                # These will be available in conn_df but not in toggle
                # (could be extended to support dynamic toggles in future)
                self.path_df[col] = additional_metrics[col]
                self._vprint(f"  ✓ Added metric column '{col}' (custom metric)")
        
        # Store custom edge colors for later use
        if custom_edge_colors:
            self.custom_edge_colors = custom_edge_colors
            self._vprint(f"  ✓ Loaded custom colors for {len(custom_edge_colors)} edges")
        else:
            self.custom_edge_colors = None
        
        # Endpoint node groups from the expanded export drive node
        # classification in build_network
        if node_group_overrides:
            self.custom_node_groups = node_group_overrides
            self._vprint(f"  ✓ Loaded endpoint groups for {len(node_group_overrides)} nodes")
        else:
            self.custom_node_groups = None
        
        # Ensure standard optional columns exist (for compatibility)
        if 'connection_ratios' not in self.path_df.columns:
            self.path_df['connection_ratios'] = [[] for _ in range(len(self.path_df))]
        
        if 'traversal_probabilities' not in self.path_df.columns:
            self.path_df['traversal_probabilities'] = [[] for _ in range(len(self.path_df))]
        
        if 'layer' not in self.path_df.columns:
            # Assign layer 0 to all paths (will be auto-detected in build_network)
            self.path_df['layer'] = 0
        
    def _parse_path_block(self, path_str):
        """
        Parse path_block string into list of nodes.
        
        Parameters
        ----------
        path_str : str
            Path string in format "A -> B -> C -> D"
            
        Returns
        -------
        list
            List of node names
        """
        return [node.strip() for node in path_str.split('->')]
    
    def _safe_eval_list(self, val):
        """
        Safely evaluate string representation of list.
        
        Parameters
        ----------
        val : str or list
            String representation of list or actual list
            
        Returns
        -------
        list
            Evaluated list
        """
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        elif isinstance(val, list):
            return val
        else:
            return []
    
    def _apply_custom_order(self, available_nodes, custom_order):
        """
        Apply custom ordering to a list of nodes.
        
        Parameters
        ----------
        available_nodes : list
            List of all available nodes
        custom_order : list
            Desired order for nodes
            
        Returns
        -------
        list
            Nodes in custom order, with any remaining nodes appended at the end
            
        Notes
        -----
        - Only includes nodes that are in available_nodes
        - Nodes in custom_order but not in available_nodes are skipped
        - Nodes in available_nodes but not in custom_order are appended at the end (sorted)
        """
        available_set = set(available_nodes)
        ordered = []
        
        # Add nodes from custom_order that exist in available_nodes
        for node in custom_order:
            if node in available_set:
                ordered.append(node)
                available_set.remove(node)
        
        # Append remaining nodes (sorted) that weren't in custom_order
        if available_set:
            ordered.extend(sorted(available_set))
        
        return ordered

    # Palette cycled when free-form edge-list group names are auto-declared
    # (kept distinct from each other and from the default node-type colors).
    _CSV_GROUP_PALETTE = (
        '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b',
        '#3b82f6', '#ef4444', '#10b981', '#a855f7',
    )

    def _register_csv_custom_groups(self, group_nodes: dict) -> None:
        """Declare free-form edge-list group names as first-class groups.

        Registered names flow through the same machinery as the
        ``node_groups`` kwarg (legend chips, quick-select buttons, assign
        dropdown, add-node dialog), so registration is idempotent by name
        and a re-declared group keeps its original color.
        """
        existing = {str(g.get('name') or '') for g in (self.node_groups or [])}
        for offset, name in enumerate(sorted(group_nodes)):
            if name in existing:
                continue
            self.node_groups.append({
                'name': name,
                'label': self._csv_group_labels.get(name, name),
                'color': self._CSV_GROUP_PALETTE[len(existing) % len(self._CSV_GROUP_PALETTE)],
            })
            existing.add(name)

    def build_network(self):
        """
        Build NetworkX graph and connection DataFrame from pathway data.
        
        This method processes the pathway data to create:
        1. A connection DataFrame with aggregated edge weights
        2. A NetworkX directed graph for visualization
        
        The method aggregates multiple paths between the same nodes,
        summing weights and averaging ratios/probabilities.
        
        Returns
        -------
        tuple
            (conn_df, G_network) - Connection DataFrame and NetworkX graph
            
        Notes
        -----
        - Connections are aggregated: same source->target get weights summed
        - Ratios and probabilities are averaged across aggregated connections
        - Graph nodes are labeled with 'node_type' (source/intermediate/target)
        """
        self._vprint("\nBuilding network from pathways...")
        
        # Store connections
        connections = []
        
        # Check which optional columns are available
        has_ratios = 'connection_ratios' in self.path_df.columns
        has_probs = 'traversal_probabilities' in self.path_df.columns
        has_nt = 'nt_types' in self.path_df.columns
        
        for idx, row in self.path_df.iterrows():
            path_block = row['path_block']
            weights = self._safe_eval_list(row['weights'])
            
            # Optional columns - only if they exist
            ratios = self._safe_eval_list(row.get('connection_ratios', [])) if has_ratios else []
            probs = self._safe_eval_list(row.get('traversal_probabilities', [])) if has_probs else []
            nt_types = self._safe_eval_list(row.get('nt_types', [])) if has_nt else []
            
            # Parse path
            nodes = self._parse_path_block(path_block)
            
            # Create connections for each hop
            for i in range(len(nodes) - 1):
                source = nodes[i]
                target = nodes[i + 1]
                weight = weights[i] if i < len(weights) else 0
                ratio = ratios[i] if i < len(ratios) else np.nan
                prob = probs[i] if i < len(probs) else np.nan
                nt = nt_types[i] if i < len(nt_types) else None
                
                conn_data = {
                    'source': source,
                    'target': target,
                    'weight': weight
                }
                
                # Only add optional columns if they exist in source data
                if has_ratios:
                    conn_data['ratio'] = ratio
                if has_probs:
                    conn_data['probability'] = prob
                if has_nt:
                    conn_data['nt_type'] = nt
                    
                connections.append(conn_data)
        
        # Create DataFrame
        conn_df = pd.DataFrame(connections)

        # If there are no per-hop connections (e.g., all input paths are single-node
        # or empty), create an empty DataFrame with expected columns to avoid groupby
        # KeyError. Otherwise aggregate duplicate connections as usual.
        if conn_df.empty:
            cols = ['source', 'target', 'weight']
            if has_ratios:
                cols.append('ratio')
            if has_probs:
                cols.append('probability')
            if has_nt:
                cols.append('nt_type')
            conn_df = pd.DataFrame(columns=cols)
        else:
            # Aggregate duplicate connections - only aggregate columns that exist
            # Use 'max' for weight: same edge has same weight regardless of which path it appears in
            agg_dict = {'weight': 'max'}
            if has_ratios:
                agg_dict['ratio'] = 'mean'
            if has_probs:
                agg_dict['probability'] = 'mean'
            if has_nt:
                # Use mode (most frequent) for categorical data like nt_type
                agg_dict['nt_type'] = lambda x: x.mode().iloc[0] if not x.mode().empty else None

            conn_df = conn_df.groupby(['source', 'target'], as_index=False).agg(agg_dict)

        # Drop zero-weight edges: users requested that zero-weight edges be removed from
        # visualizations while still allowing nodes mentioned only in zero-weight rows
        # to show up as orphan nodes. We therefore drop aggregated edges whose weight
        # is exactly 0 here (after aggregation), but will add all nodes that appear in
        # the original path data back into the graph later.
        before_count = len(conn_df)
        conn_df = conn_df.loc[conn_df['weight'] != 0].copy()
        after_count = len(conn_df)

        if before_count != after_count:
            self._vprint(f"Dropped {before_count - after_count} zero-weight aggregated connections")

        self._vprint(f"Created {len(conn_df)} unique connections from pathways")
        
        # Build graph using FastGraph (lightweight NetworkX replacement)
        G = FastGraph()
        
        # Add edges with attributes - only add attributes that exist
        for _, row in conn_df.iterrows():
            edge_attrs = {'weight': row['weight']}
            if 'ratio' in row:
                edge_attrs['ratio'] = row['ratio']
            if 'probability' in row:
                edge_attrs['probability'] = row['probability']
            
            G.add_edge(row['source'], row['target'], **edge_attrs)
        
        # Determine node types based on position in original paths
        # Track which nodes appear at the start or end of ANY path
        path_sources = set()
        path_targets = set()
        
        for idx, row in self.path_df.iterrows():
            path_block = row['path_block']
            nodes = self._parse_path_block(path_block)
            if len(nodes) > 0:
                path_sources.add(nodes[0])   # First node is a source
                path_targets.add(nodes[-1])  # Last node is a target

        # Ensure orphan nodes (nodes that appear in path data but have no
        # non-zero edges after filtering) are present in the graph. This
        # allows edge-list inputs that contain source/target with weight=0
        # to still produce visible nodes in the network and Sankey.
        all_nodes_in_paths = set()
        for idx, row in self.path_df.iterrows():
            nodes = self._parse_path_block(row['path_block'])
            for n in nodes:
                all_nodes_in_paths.add(n)

        missing_nodes = all_nodes_in_paths - set(G.nodes())
        if missing_nodes:
            for n in missing_nodes:
                G.add_node(n)
        
        # Helper to extract base name without hemisphere suffix
        def _get_base_name(label: str) -> str:
            base = label
            if '(' in base:
                base = base.split('(')[0].strip()
            if base.endswith(('_L', '_R', '_U')):
                base = base[:-2]
            return base
        
        # If separate_hemispheres, also collect base names for matching
        # e.g., if aMe12 is in path_sources, aMe12_L and aMe12_R should also be sources
        source_base_names = set(_get_base_name(s) for s in path_sources) if self.separate_hemispheres else set()
        target_base_names = set(_get_base_name(t) for t in path_targets) if self.separate_hemispheres else set()

        # Classify nodes with two strategies:
        # - Multi-hop path data uses path POSITION: the first node of any
        #   path is a source, the last node a target (mirrors the queried
        #   source/target sets of the pathfinding tools).
        # - Edge-list input (every row is a two-node path) carries no path
        #   structure, so position-based classification would mark EVERY
        #   node with any outgoing edge as a source. Use graph DEGREE
        #   instead: nodes with ONLY outgoing edges are sources, nodes with
        #   ONLY incoming edges are targets. Self-loop edges are excluded
        #   from both counts — they neither emit nor receive connections.
        all_nodes = set(G.nodes())
        source_nodes = set()
        target_nodes = set()

        edge_list_input = bool(self.path_df is not None and len(self.path_df) > 0) and all(
            len(self._parse_path_block(row['path_block'])) == 2
            for _idx, row in self.path_df.iterrows()
        )

        if edge_list_input:
            has_outgoing = set()
            has_incoming = set()
            for u, v in G.edges():
                if u != v:  # self-loops never count as in/out
                    has_outgoing.add(u)
                    has_incoming.add(v)
            source_nodes = has_outgoing - has_incoming
            target_nodes = has_incoming - has_outgoing
        else:
            # Position-based classification: prioritize source/target
            # identity. When separate_hemispheres=True, match by base name
            # as well.
            for node in all_nodes:
                if node in path_sources:
                    source_nodes.add(node)
                elif self.separate_hemispheres and _get_base_name(str(node)) in source_base_names:
                    source_nodes.add(node)
                elif node in path_targets:
                    target_nodes.add(node)
                elif self.separate_hemispheres and _get_base_name(str(node)) in target_base_names:
                    target_nodes.add(node)

        intermediate_nodes = all_nodes - source_nodes - target_nodes
        
        # Set node attributes
        for node in source_nodes:
            G.nodes[node]['node_type'] = 'source'
        for node in intermediate_nodes:
            G.nodes[node]['node_type'] = 'intermediate'
        for node in target_nodes:
            G.nodes[node]['node_type'] = 'target'
        
        # Edge lists carrying endpoint-group columns (the expanded Edge List
        # CSV export) override the position-based classification: every edge
        # is a two-node path, so without them ALL nodes would read "source".
        # Values naming a structural role keep that role; any other name
        # becomes a declared custom group (legend chip, quick-select button,
        # assign dropdown) carried by the node's 'group' attribute.
        if getattr(self, 'custom_node_groups', None):
            valid_groups = {'source', 'intermediate', 'target'}
            overridden = 0
            custom_group_nodes = {}
            for node, group in self.custom_node_groups.items():
                if node not in all_nodes:
                    continue
                if group in valid_groups:
                    G.nodes[node]['node_type'] = group
                    overridden += 1
                else:
                    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', str(group)).strip('_')
                    if safe_name:
                        custom_group_nodes.setdefault(safe_name, set()).add(node)
                        raw_label = str(group).strip()
                        self._csv_group_labels[safe_name] = raw_label
            if overridden:
                self._vprint(f"Applied endpoint groups from edge list to {overridden} nodes")
            if custom_group_nodes:
                self._register_csv_custom_groups(custom_group_nodes)
                for group, nodes in custom_group_nodes.items():
                    for node in nodes:
                        G.nodes[node]['group'] = group
                self._vprint(
                    f"Registered {len(custom_group_nodes)} custom group(s) "
                    f"from the edge list: {', '.join(sorted(custom_group_nodes))}"
                )
        
        # Count from the FINAL node attributes so endpoint-group overrides
        # (expanded edge-list export) are reflected in the summary.
        final_counts = {'source': 0, 'intermediate': 0, 'target': 0}
        for node in all_nodes:
            final_counts[G.nodes[node].get('node_type', 'intermediate')] += 1
        self._vprint(f"Network: {final_counts['source']} source, "
                     f"{final_counts['intermediate']} intermediate, "
                     f"{final_counts['target']} target nodes")
        
        self.conn_df = conn_df
        self.G_network = G
        
        return conn_df, G
    
    def create_sankey(self):
        """
        Create multi-layer Sankey diagram from pathway data.
        
        Builds a layered Sankey diagram similar to coana's implementation:
        - Extracts layer information from paths
        - Orders nodes by layer (source → intermediate → target)
        - Assigns colors based on node types
        - Creates connections between adjacent layers
        
        Returns
        -------
        str
            Path to the generated HTML file
            
        Notes
        -----
        - Requires pathway data with path_block column
        - Node widths represent number of connections
        - Link widths represent synapse weights
        - Colors follow source_color, intermediate_color, and target_color settings
        - Builds layers from path_block data to ensure proper ordering
        """
        if self.path_df is None:
            self._vprint("Error: No pathway data available.")
            return None
        
        self._vprint("\nCreating layered Sankey diagram...")
        
        # Use the same complete-path selection as the network and heatmap.
        # This keeps all three visualizations consistent and prevents a
        # weak target-incoming edge from appearing in Sankey without its
        # selected upstream route.
        if self.G_network is None:
            self.build_network()

        path_df_with_score = self.path_df.copy()

        def compute_path_min_weight(weights_value):
            weights_list = self._safe_eval_list(weights_value)
            if not weights_list:
                return 0
            return min(weights_list)

        path_df_with_score['_min_weight'] = path_df_with_score['weights'].apply(compute_path_min_weight)
        path_df_sorted = path_df_with_score.sort_values('_min_weight', ascending=False)
        
        # Extract layer information from paths
        # edge_data: {(source, target): {'weight': ..., 'ratio': ..., 'prob': ..., 'count': ...}}
        edge_data = {}
        node_layers = {}  # {node: set of layer indices}
        
        # Check which optional columns are available
        has_ratios = 'connection_ratios' in self.path_df.columns
        has_probs = 'traversal_probabilities' in self.path_df.columns
        has_nt = 'nt_types' in self.path_df.columns
        
        selected_for_plot = self._select_edges_for_plot()
        if selected_for_plot is not None and selected_for_plot[3] is not None:
            selected_path_indexes = selected_for_plot[3]
            path_df_to_process = self.path_df.loc[selected_path_indexes]
            needs_simplification = True
            unique_edges = set()
            for _idx, row in path_df_to_process.iterrows():
                for edge_key, _data in self._path_edge_records(row):
                    unique_edges.add(edge_key)
            threshold = selected_for_plot[4]
            self._vprint(
                f'  Selected {len(selected_path_indexes)} complete paths → '
                f'{len(unique_edges)} unique edges (applied threshold: '
                f'edge weight >= {threshold:g})')
        else:
            needs_simplification = False
            path_df_to_process = path_df_sorted
        
        # Process paths to extract edge data
        for idx, row in path_df_to_process.iterrows():
            path_block = row['path_block']
            weights = self._safe_eval_list(row['weights'])
            ratios = self._safe_eval_list(row.get('connection_ratios', [])) if has_ratios else []
            probs = self._safe_eval_list(row.get('traversal_probabilities', [])) if has_probs else []
            nt_types = self._safe_eval_list(row.get('nt_types', [])) if has_nt else []
            
            nodes = self._parse_path_block(path_block)
            
            # Track which layers each node appears in
            for layer_idx, node in enumerate(nodes):
                node_layers.setdefault(node, set()).add(layer_idx)
            
            # Create edges with layer information
            for i in range(len(nodes) - 1):
                source = nodes[i]
                target = nodes[i + 1]

                # Edge keys are LAYER-LESS (source, target): every node
                # renders once, so the same node pair reached at different
                # hop depths by different paths (one pair's direct
                # annotation bridge and its longer crosswalk chain) is one
                # biological connection — keyed by layer it drew as
                # PARALLEL ribbons with the weight counted twice.
                edge_key = (source, target)
                weight = weights[i] if i < len(weights) else 0
                ratio = ratios[i] if i < len(ratios) else 0
                prob = probs[i] if i < len(probs) else 0
                nt = nt_types[i] if i < len(nt_types) else None

                if edge_key not in edge_data:
                    edge_data[edge_key] = {'weight': weight, 'ratio': ratio, 'prob': prob, 'nt': nt}
                else:
                    # Same edge in different paths (possibly at different
                    # hop depths): use max since it's the same biological
                    # connection (weight should be identical, but use max
                    # to be safe)
                    edge_data[edge_key]['weight'] = max(edge_data[edge_key]['weight'], weight)
                    edge_data[edge_key]['ratio'] = max(edge_data[edge_key]['ratio'], ratio)
                    edge_data[edge_key]['prob'] = max(edge_data[edge_key]['prob'], prob)
                    # Keep first NT type found (they should be the same for same edge)
                    if edge_data[edge_key]['nt'] is None:
                        edge_data[edge_key]['nt'] = nt
        
        # Remove zero-weight layer edges (user preference: drop zero-weight edges)
        edge_data = {k: v for k, v in edge_data.items() if v.get('weight', 0) != 0}

        if len(edge_data) == 0:
            # Warn but continue: we still want to build a node-only Sankey (or at least
            # include orphan nodes) if no non-zero links are present.
            self._vprint('\033[33mWarning: No non-zero connections found for Sankey diagram. Building node-only Sankey (no links).\033[0m')
        
        # Track whether the shared selector already simplified the path set.
        # A single irreducibly long path may exceed the configured limit; in
        # that case complete-path integrity is intentionally preferred.
        simplification_applied = needs_simplification
        original_edge_count = len(edge_data)
        
        # Build node list ordered by layers (key to proper layering).
        # Each node sits at its EARLIEST hop layer (nodes render once).
        # When the shared selector simplified the path set, nodes whose
        # every edge was filtered out stay excluded (no orphan
        # reintroduction).
        edge_endpoints = set()
        for source, target in edge_data.keys():
            edge_endpoints.add(source)
            edge_endpoints.add(target)
        nodes_by_layer = {}
        for node, layers in node_layers.items():
            if not layers:
                continue
            if simplification_applied and node not in edge_endpoints:
                continue
            nodes_by_layer.setdefault(min(layers), set()).add(node)
        
        # Build network if not already done (to get node types)
        if self.G_network is None:
            self.build_network()
        
        node_list = []
        node_labels = []
        node_colors_list = []
        node_added = set()  # Track which nodes we've already added
        
        # Get node types from graph
        node_types = {node: self.G_network.nodes[node].get('node_type', 'intermediate') 
                     for node in self.G_network.nodes()}
        
        # Build nodes ordered by layer (add each node only once, at its earliest layer)
        for layer_idx in sorted(nodes_by_layer.keys()):
            layer_nodes = sorted(nodes_by_layer[layer_idx])
            for node in layer_nodes:
                if node in node_added:
                    continue  # Skip if already added
                
                node_added.add(node)
                node_list.append(node)
                
                # Create label with layer information (unless the caller
                # opted out — see sankey_label_layers).
                all_layers = sorted(node_layers[node])
                if not self.sankey_label_layers:
                    node_labels.append(node)
                elif len(all_layers) == 1:
                    node_labels.append(f"{node} (L{all_layers[0]})")
                else:
                    layers_str = ','.join(map(str, all_layers))
                    node_labels.append(f"{node} (L{layers_str})")
                
                # Assign color: prioritize custom colors, then default by node type
                if self.custom_node_colors and node in self.custom_node_colors:
                    # Use custom color
                    node_colors_list.append(self.custom_node_colors[node])
                else:
                    # Use default color based on node type
                    node_type = node_types.get(node, 'intermediate')
                    if node_type == 'source':
                        node_colors_list.append(self.source_color)
                    elif node_type == 'target':
                        node_colors_list.append(self.target_color)
                    else:
                        node_colors_list.append(self.intermediate_color)
        
        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Build edge lists
        source_indices = []
        target_indices = []
        weights = []
        original_weights = []  # Store original weights (including negatives) for hover
        ratios = []
        probs = []
        nt_types_list = []  # Store NT types for edges
        edge_colors = []  # Custom edge colors
        has_negative = False  # Track if any negative weights exist
        has_nt_coloring = False  # Track if NT-based coloring is applied
        
        for (source, target), data in edge_data.items():
            source_indices.append(node_to_idx[source])
            target_indices.append(node_to_idx[target])
            
            # Handle negative weights: use absolute value, mark with different color
            weight = data['weight']
            is_negative = weight < 0
            if is_negative:
                has_negative = True
            abs_weight = abs(weight)
            
            weights.append(abs_weight)
            original_weights.append(weight)  # Keep original for hover label
            ratios.append(data['ratio'])
            probs.append(data['prob'])
            
            # Get NT type
            nt = data.get('nt', None)
            nt_types_list.append(nt)
            
            # Determine edge color
            if is_negative:
                # Light blue for negative edges (overrides NT coloring) —
                # the same value the final styling pass applies below
                edge_colors.append('rgba(74, 144, 226, 0.4)')
            elif self.color_edges_by_nt and nt is not None:
                # Use NT-based color
                edge_colors.append(get_nt_color(nt, opacity=0.6))
                has_nt_coloring = True
            elif self.custom_edge_colors and (source, target) in self.custom_edge_colors:
                edge_colors.append(self.custom_edge_colors[(source, target)])
            else:
                edge_colors.append(self.link_color)
        
        if has_negative:
            self._vprint(f"  ℹ️  Found negative edge weights - using absolute values for link width, light blue for negative edges")
        if has_nt_coloring:
            self._vprint(f"  🧬 Applied neurotransmitter-based edge coloring")
        
        # Create custom hover labels that show source, target, and original weights
        hover_labels = []
        for i, (src_idx, tgt_idx, orig_weight, abs_weight) in enumerate(zip(source_indices, target_indices, original_weights, weights)):
            source_name = html_escape(node_list[src_idx])
            target_name = html_escape(node_list[tgt_idx])
            hover_text = f"{source_name} → {target_name}<br>"
            hover_text += f"Weight: {orig_weight:,}"  # Show original (possibly negative)
            if ratios[i] != 0:
                hover_text += f"<br>Ratio: {ratios[i]:.3f}"
            if probs[i] != 0:
                hover_text += f"<br>Probability: {probs[i]:.3f}"
            # Add NT type to hover if available
            if nt_types_list[i] is not None:
                hover_text += f"<br>NT: {html_escape(nt_types_list[i])}"
            hover_labels.append(hover_text)
        
        # Create Sankey figure using Plotly directly (like coana)
        import plotly.graph_objects as go
        
        # Update edge colors: use previously computed edge_colors (which already handles
        # negative weights, NT coloring, and custom colors)
        # Only override if we need to apply the final styling
        edge_colors_updated = []
        for i, orig_weight in enumerate(original_weights):
            if orig_weight < 0:
                edge_colors_updated.append('rgba(74, 144, 226, 0.4)')  # Light blue for negative
            elif self.color_edges_by_nt and nt_types_list[i] is not None:
                # Use NT-based color (already computed in edge_colors)
                edge_colors_updated.append(edge_colors[i])
            elif self.custom_edge_colors:
                # Check if custom color exists for this edge
                src_node = node_list[source_indices[i]]
                tgt_node = node_list[target_indices[i]]
                if (src_node, tgt_node) in self.custom_edge_colors:
                    edge_colors_updated.append(self.custom_edge_colors[(src_node, tgt_node)])
                else:
                    edge_colors_updated.append('rgba(100, 100, 100, 0.4)')  # Gray for positive
            else:
                edge_colors_updated.append('rgba(100, 100, 100, 0.4)')  # Gray for positive
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=5,
                thickness=5,
                line=dict(color="black", width=0),
                label=node_labels,
                color=node_colors_list
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=weights,
                color=edge_colors_updated,
                customdata=hover_labels,  # Store full hover text
                hovertemplate='%{customdata}<extra></extra>'  # Show custom hover text
            )
        )])
        
        # Add legend annotations if there are negative values
        annotations = []
        if has_negative:
            # Add legend for positive and negative edges
            annotations = [
                dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text='<b>Legend:</b>',
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                ),
                dict(
                    x=0.02, y=0.94,
                    xref='paper', yref='paper',
                    text='<span style="color: rgba(100,100,100,0.6);">■</span> Positive weight',
                    showarrow=False,
                    font=dict(size=11, color='black'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                ),
                dict(
                    x=0.02, y=0.90,
                    xref='paper', yref='paper',
                    text='<span style="color: rgba(74,144,226,0.4);">■</span> Negative weight',
                    showarrow=False,
                    font=dict(size=11, color='black'),
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )
            ]
        
        # Add NT legend if NT coloring is applied
        if has_nt_coloring:
            # Collect unique NT types present in the data
            unique_nts = set(nt for nt in nt_types_list if nt is not None)
            y_offset = 0.86 if has_negative else 0.98  # Start below negative legend or at top
            
            if not annotations:
                annotations = [
                    dict(
                        x=0.02, y=y_offset,
                        xref='paper', yref='paper',
                        text='<b>Neurotransmitters:</b>',
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align='left',
                        xanchor='left',
                        yanchor='top'
                    )
                ]
                y_offset -= 0.04
            else:
                annotations.append(
                    dict(
                        x=0.02, y=y_offset,
                        xref='paper', yref='paper',
                        text='<b>Neurotransmitters:</b>',
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        align='left',
                        xanchor='left',
                        yanchor='top'
                    )
                )
                y_offset -= 0.04
            
            # Add each NT type to legend
            for nt in sorted(unique_nts):
                nt_color = get_nt_color(nt, opacity=0.8)
                annotations.append(
                    dict(
                        x=0.02, y=y_offset,
                        xref='paper', yref='paper',
                        text=f'<span style="color: {nt_color};">■</span> {nt}',
                        showarrow=False,
                        font=dict(size=11, color='black'),
                        align='left',
                        xanchor='left',
                        yanchor='top'
                    )
                )
                y_offset -= 0.04
        
        fig.update_layout(
            title_text=self.sankey_title,
            font_size=12,
            height=None,  # Let it fill container
            autosize=True,  # Auto-resize to container
            margin=dict(l=10, r=10, t=50, b=10),  # Minimal margins to maximize diagram space
            annotations=annotations  # Add legend
        )
        
        output_path = os.path.join(self.output_folder, self.base_filename + '_Sankey.html')
        
        # Get the basic Plotly HTML.  The library is embedded, not
        # referenced: a downloaded Sankey has to open from disk with no
        # network (the CDN form left it blank offline, and a Sankey is the
        # artifact reviewers are most likely to open away from their
        # machine).  plotly.py ships the bundle, so this adds no dependency.
        import plotly.io as pio
        basic_html = pio.to_html(fig, include_plotlyjs=True,
                                 full_html=False)
        
        # Create custom HTML with interactive controls
        html_content = self._create_sankey_html_with_controls(
            basic_html, 
            node_list, 
            node_labels, 
            node_colors_list,
            source_indices,
            target_indices,
            weights,
            ratios,
            probs,
            original_weights,
            simplification_applied,
            original_edge_count,
            nt_types_list=nt_types_list,
            edge_colors=edge_colors_updated
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # if self.showfig:
        #     import webbrowser
        #     webbrowser.open('file://' + os.path.abspath(output_path))
        
        self._vprint(f"  Sankey diagram saved with {len(node_list)} nodes and {len(weights)} edges")
        self._vprint(f"  Output: {output_path}")
        
        return output_path
    
    def _create_sankey_html_with_controls(self, plotly_div, node_list, node_labels, node_colors_list, 
                                          source_indices, target_indices, weights, ratios=None, probs=None, original_weights=None,
                                          simplification_applied=False, original_edge_count=0, nt_types_list=None, edge_colors=None):
        """
        Create HTML with interactive controls for Sankey diagram.
        
        Adds control panel with:
        - Node color pickers for each node type
        - Edge color and opacity sliders
        - Metric toggle (weight/ratio/prob)
        - NT group color pickers (excitatory, inhibitory, modulatory, unknown)
        - Hide/show nodes and edges by clicking
        - Reset button
        
        Parameters
        ----------
        plotly_div : str
            Plotly HTML div content
        node_list : list
            List of node names
        node_labels : list
            List of node labels with layer info
        node_colors_list : list
            List of node colors
        source_indices : list
            List of source node indices for edges
        target_indices : list
            List of target node indices for edges
        weights : list
            List of edge weights (synapse counts)
        ratios : list, optional
            List of edge connection ratios
        probs : list, optional
            List of edge traversal probabilities
        simplification_applied : bool, optional
            Whether edge simplification was applied
        original_edge_count : int, optional
            Original number of edges before simplification
        """
        
        # Check which metrics are available
        has_ratios = ratios is not None and len(ratios) > 0 and any(r > 0 for r in ratios)
        has_probs = probs is not None and len(probs) > 0 and any(p > 0 for p in probs)
        
        # Default to empty lists if not provided
        if ratios is None:
            ratios = [0] * len(weights)
        if probs is None:
            probs = [0] * len(weights)
        
        # Process NT types list for edge coloring by NT group
        # Check if we have actual NT data (not all None/empty)
        if nt_types_list is None:
            nt_types_list = ['unknown'] * len(weights)
            has_nt_types = False
        else:
            # Replace None values with 'unknown' for JavaScript compatibility
            nt_types_list = [nt if nt is not None else 'unknown' for nt in nt_types_list]
            # Check if we have any actual NT data (not all unknown)
            has_nt_types = any(nt != 'unknown' for nt in nt_types_list)
        
        # Map NT types to groups for each edge
        nt_groups_for_edges = []
        for nt in nt_types_list:
            nt_groups_for_edges.append(get_nt_group(nt))
        
        # Parse edge color and opacity
        # For Sankey, always use gray for positive edges (negative edges are light blue)
        edge_hex, edge_opacity = parse_color_to_hex_opacity('rgba(100, 100, 100, 0.5)')
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive Sankey Diagram</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            overflow: hidden;
        }}
        #main-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #controls {{
            background: white;
            border-bottom: 1px solid #ddd;
            padding: 12px 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 25vh;  /* Limit control panel height to 25% of viewport */
        }}
        .control-row {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        #sankey-container {{
            flex: 1;
            background: white;
            position: relative;
            overflow: hidden;
            min-height: 0;  /* Important for flex children */
        }}
        /* Make Plotly div fill the container */
        #sankey-container > div {{
            width: 100% !important;
            height: 100% !important;
        }}
        /* Make Plotly plot fill its parent */
        #sankey-container .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
        #sankey-container .plot-container {{
            width: 100% !important;
            height: 100% !important;
        }}
        /* Make hover labels larger and ensure text doesn't overflow */
        .hoverlayer .hovertext {{
            min-width: 300px !important;
            max-width: 500px !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            padding: 15px !important;
        }}
        .hoverlayer .hovertext text {{
            white-space: normal !important;
        }}
        /* Expand the background box */
        .hoverlayer .hovertext path {{
            opacity: 1 !important;
        }}
        .hoverlayer .hovertext rect {{
            width: auto !important;
            min-width: 300px !important;
        }}
        .control-section {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .control-label {{
            font-weight: 500;
            color: #555;
            font-size: 12px;
            white-space: nowrap;
        }}
        .color-input-group {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        input[type="color"] {{
            width: 40px;
            height: 30px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        input[type="range"] {{
            width: 120px;
        }}
        input[type="text"] {{
            width: 90px;
            padding: 5px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            background: #f9f9f9;
        }}
        .slider-value {{
            color: #666;
            font-size: 11px;
            min-width: 35px;
            text-align: right;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .btn-primary {{
            background: #4CAF50;
            color: white;
        }}
        .btn-primary:hover {{
            background: #45a049;
        }}
        .btn-secondary {{
            background: #2196F3;
            color: white;
        }}
        .btn-secondary:hover {{
            background: #0b7dda;
        }}
        .btn-danger {{
            background: #f44336;
            color: white;
        }}
        .btn-danger:hover {{
            background: #da190b;
        }}
        .btn-group {{
            display: flex;
            gap: 8px;
        }}
        .divider {{
            width: 1px;
            background: #ddd;
            margin: 0 10px;
        }}
        .visibility-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.98);
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            max-width: 300px;
            max-height: 70vh;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }}
        .visibility-panel.show {{
            display: block;
        }}
        .visibility-panel h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #333;
        }}
        .node-list, .edge-list {{
            max-height: 150px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            margin: 10px 0;
            background: #fafafa;
        }}
        .list-item {{
            padding: 5px 8px;
            margin: 2px 0;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }}
        }}
        .list-item:hover {{
            background: #e3f2fd;
        }}
        .list-item.hidden {{
            opacity: 0.4;
            text-decoration: line-through;
        }}
        .color-indicator {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid #999;
            display: inline-block;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <div id="controls">
            <!-- Row 1: Node Colors -->
            <div class="control-row">
                <div class="control-group">
                    <label class="control-label">Source</label>
                    <div class="color-input-group">
                        <input type="color" id="sourceColor" value="{self.source_color}">
                        <input type="text" id="sourceColorText" value="{self.source_color}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Intermediate</label>
                    <div class="color-input-group">
                        <input type="color" id="intermediateColor" value="{self.intermediate_color}">
                        <input type="text" id="intermediateColorText" value="{self.intermediate_color}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Target</label>
                    <div class="color-input-group">
                        <input type="color" id="targetColor" value="{self.target_color}">
                        <input type="text" id="targetColorText" value="{self.target_color}" readonly>
                    </div>
                </div>
            </div>
            
            <!-- Row 2: Metric Selection & Edge Settings -->
            <div class="control-row">
                <div class="control-group">
                    <label class="control-label">Connection Metric</label>
                    <div class="color-input-group">
                        <select id="metricSelect" style="padding: 5px 8px; border: 1px solid #ddd; border-radius: 4px; background: white; cursor: pointer; font-size: 12px;">
                            <option value="weight">{self.metric_option_label}</option>
                            <option value="ratio" {"disabled" if not has_ratios else ""}>Connection Ratio{" (N/A)" if not has_ratios else ""}</option>
                            <option value="prob" {"disabled" if not has_probs else ""}>Traversal Probability{" (N/A)" if not has_probs else ""}</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Edge Color</label>
                    <div class="color-input-group">
                        <input type="color" id="edgeColor" value="{edge_hex}">
                        <input type="text" id="edgeColorText" value="{edge_hex}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Edge Opacity</label>
                    <div class="color-input-group">
                        <input type="range" id="edgeOpacity" min="0" max="100" value="{int(edge_opacity * 100)}">
                        <span class="slider-value" id="edgeOpacityValue">{int(edge_opacity * 100)}%</span>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Node Width</label>
                    <div class="color-input-group">
                        <input type="range" id="nodeWidth" min="{self.min_node_size}" max="{self.max_node_size}" value="5">
                        <span class="slider-value" id="nodeWidthValue">5</span>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Font Size</label>
                    <div class="color-input-group">
                        <input type="range" id="fontSize" min="{self.min_font_size}" max="{self.max_font_size}" value="12">
                        <span class="slider-value" id="fontSizeValue">12px</span>
                    </div>
                </div>
            </div>
            
            <!-- Row 3: NT Group Colors (only shown if NT data available) -->
            <div class="control-row" id="ntControlRow" style="display: {'flex' if has_nt_types else 'none'};">
                <div class="control-group">
                    <label class="control-label">
                        <input type="checkbox" id="colorByNt" onchange="toggleNtColoring()" {'checked' if self.color_edges_by_nt else ''}> Color by NT
                    </label>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Excitatory (ACh)</label>
                    <div class="color-input-group">
                        <input type="color" id="excitatoryColor" value="{NT_GROUP_COLORS['excitatory']}" onchange="updateNtGroupColorText('excitatory')">
                        <input type="text" id="excitatoryColorText" value="{NT_GROUP_COLORS['excitatory']}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Inhibitory (GABA, Glut)</label>
                    <div class="color-input-group">
                        <input type="color" id="inhibitoryColor" value="{NT_GROUP_COLORS['inhibitory']}" onchange="updateNtGroupColorText('inhibitory')">
                        <input type="text" id="inhibitoryColorText" value="{NT_GROUP_COLORS['inhibitory']}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Modulatory (DA, 5-HT, OA)</label>
                    <div class="color-input-group">
                        <input type="color" id="modulatoryColor" value="{NT_GROUP_COLORS['modulatory']}" onchange="updateNtGroupColorText('modulatory')">
                        <input type="text" id="modulatoryColorText" value="{NT_GROUP_COLORS['modulatory']}" readonly>
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Unknown NT</label>
                    <div class="color-input-group">
                        <input type="color" id="unknownColor" value="{NT_GROUP_COLORS['unknown']}" onchange="updateNtGroupColorText('unknown')">
                        <input type="text" id="unknownColorText" value="{NT_GROUP_COLORS['unknown']}" readonly>
                    </div>
                </div>
            </div>
            
            <!-- Row 4: Action Buttons -->
            <div class="control-row">
                <div class="btn-group">
                    <button class="btn-primary" onclick="applyColors()">Apply</button>
                    <button class="btn-secondary" onclick="resetToDefaults()">Reset</button>
                    <button class="btn-secondary" onclick="toggleVisibilityPanel()">Show/Hide</button>
                </div>
                
                <div class="btn-group">
                    <button class="btn-secondary" onclick="zoomIn()">🔍 +</button>
                    <button class="btn-secondary" onclick="zoomOut()">🔍 -</button>
                    <button class="btn-secondary" onclick="resetZoom()">⟲</button>
                    <button class="btn-secondary" id="toggleLabelsBtn" onclick="toggleLabels()">🏷️ Hide Labels</button>
                </div>
                
                <div class="btn-group">
                    <button id="bgToggleBtn" class="btn-secondary" onclick="toggleBackground()" title="Cycle the canvas background between the white and dark presets">White</button>
                    <input type="color" id="customBgColor" value="#f5f5f5" style="width: 30px; height: 28px; border: 1px solid #ddd; border-radius: 3px; cursor: pointer;" onchange="applyCustomBackground()">
                </div>
                
                <div class="btn-group">
                    <label style="font-size: 12px; margin-right: 5px;">Scale:</label>
                    <input type="number" id="exportScale" min="1" max="10" value="2" step="0.5" style="width: 50px; padding: 3px; border: 1px solid #ddd; border-radius: 3px;">
                    <button class="btn-secondary" onclick="exportPNG()" title="Export PNG">📸 PNG</button>
                    <button class="btn-secondary" onclick="exportSVG()" title="Export SVG">🎨 SVG</button>
                </div>
            </div>
        </div>
        
        <div id="sankey-container">
            {plotly_div}
            
            <div id="visibility-panel" class="visibility-panel">
                <h4>👁️ Visibility Controls</h4>
                <div>
                    <strong style="font-size: 12px;">Nodes</strong>
                    <div class="node-list" id="nodeList"></div>
                </div>
                <div>
                    <strong style="font-size: 12px;">Edges</strong>
                    <div class="edge-list" id="edgeList"></div>
                </div>
                <div class="btn-group" style="margin-top: 10px;">
                    <button class="btn-secondary" onclick="showAll()">Show All</button>
                    <button class="btn-danger" onclick="hideAll()">Hide All</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Store original data
        const nodeList = {json_safe(node_list)};
        const nodeLabels = {json_safe(node_labels)};
        const nodeColors = {json_safe(node_colors_list)};
        const sourceIndices = {json_safe(source_indices)};
        const targetIndices = {json_safe(target_indices)};
        const weights = {json_safe(weights)};
        const ratios = {json_safe(ratios)};
        const probs = {json_safe(probs)};
        const originalWeights = {json_safe(original_weights)};  // Original (possibly negative) weights
        const hasRatios = {str(has_ratios).lower()};
        const hasProbs = {str(has_probs).lower()};
        
        // NT type data for edge coloring
        const ntTypes = {json_safe(nt_types_list)};  // NT type for each edge
        const ntGroups = {json_safe(nt_groups_for_edges)};  // NT group for each edge (excitatory, inhibitory, modulatory, unknown)
        const hasNtTypes = {str(has_nt_types).lower()};
        
        // NT group colors (can be modified via controls)
        const ntGroupColors = {{
            'excitatory': '{NT_GROUP_COLORS["excitatory"]}',
            'inhibitory': '{NT_GROUP_COLORS["inhibitory"]}',
            'modulatory': '{NT_GROUP_COLORS["modulatory"]}',
            'unknown': '{NT_GROUP_COLORS["unknown"]}'
        }};
        
        // Whether to color edges by NT group (seeded from the Python-side
        // color_edges_by_nt option so server-computed colors are not lost)
        let colorEdgesByNt = {str(self.color_edges_by_nt).lower()};
        
        // Server-computed initial edge colors (NT/custom/link_color aware).
        // Used until the user picks a custom edge color in the UI.
        const initialEdgeColors = {json_safe(edge_colors) if edge_colors is not None else 'null'};
        let edgeColorCustomized = false;
        
        // Current metric being displayed
        let currentMetric = 'weight';
        
        // Store node types
        const nodeTypes = [];
        nodeLabels.forEach((label, idx) => {{
            if (nodeColors[idx] === '{self.source_color}') nodeTypes.push('source');
            else if (nodeColors[idx] === '{self.target_color}') nodeTypes.push('target');
            else nodeTypes.push('intermediate');
        }});
        
        // Track visibility
        let hiddenNodes = new Set();
        let hiddenEdges = new Set();
        
        // Toggle NT-based edge coloring
        function toggleNtColoring() {{
            colorEdgesByNt = document.getElementById('colorByNt').checked;
            updateDiagram();
        }}
        
        // Update NT group color text display
        function updateNtGroupColorText(group) {{
            const colorInput = document.getElementById(group + 'Color');
            const textInput = document.getElementById(group + 'ColorText');
            textInput.value = colorInput.value;
            ntGroupColors[group] = colorInput.value;
            if (colorEdgesByNt) {{
                updateDiagram();
            }}
        }}
        
        // Get color for an edge based on its NT group
        function getEdgeColorByNtGroup(edgeIdx) {{
            const group = ntGroups[edgeIdx];
            return ntGroupColors[group] || ntGroupColors['unknown'];
        }}
        
        // Parse '#rrggbb' or 'rgb(r,g,b)' / 'rgba(r,g,b,a)' into {{r, g, b}}
        function parseColorStringRGB(color) {{
            if (!color || typeof color !== 'string') return null;
            const rgbaMatch = color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
            if (rgbaMatch) {{
                return {{ r: parseInt(rgbaMatch[1], 10), g: parseInt(rgbaMatch[2], 10), b: parseInt(rgbaMatch[3], 10) }};
            }}
            const hexMatch = color.match(/^#?([0-9a-fA-F]{{2}})([0-9a-fA-F]{{2}})([0-9a-fA-F]{{2}})/);
            if (hexMatch) {{
                return {{ r: parseInt(hexMatch[1], 16), g: parseInt(hexMatch[2], 16), b: parseInt(hexMatch[3], 16) }};
            }}
            return null;
        }}
        
        // Toggle visibility panel
        function toggleVisibilityPanel() {{
            const panel = document.getElementById('visibility-panel');
            panel.classList.toggle('show');
        }}
        
        // Initialize UI
        function initializeLists() {{
            const nodeListEl = document.getElementById('nodeList');
            nodeLabels.forEach((label, idx) => {{
                const item = document.createElement('div');
                item.className = 'list-item';
                item.innerHTML = `<span><span class="color-indicator" style="background: ${{nodeColors[idx]}}"></span>${{escapeHtml(label)}}</span>`;
                item.onclick = () => toggleNode(idx);
                item.id = `node-item-${{idx}}`;
                nodeListEl.appendChild(item);
            }});
            
            const edgeListEl = document.getElementById('edgeList');
            sourceIndices.forEach((src, idx) => {{
                const item = document.createElement('div');
                item.className = 'list-item';
                const metricDisplay = getMetricDisplay(idx);
                item.innerHTML = `<span>${{escapeHtml(nodeList[src])}} → ${{escapeHtml(nodeList[targetIndices[idx]])}}</span><span style="color: #999;" id="edge-metric-${{idx}}">${{metricDisplay}}</span>`;
                item.onclick = () => toggleEdge(idx);
                item.id = `edge-item-${{idx}}`;
                edgeListEl.appendChild(item);
            }});
        }}
        
        // Get metric display string for an edge
        function getMetricDisplay(idx) {{
            if (currentMetric === 'ratio') {{
                return ratios[idx].toFixed(4);
            }} else if (currentMetric === 'prob') {{
                return probs[idx].toFixed(4);
            }} else {{
                return weights[idx];
            }}
        }}
        
        // Update edge list metric values
        function updateEdgeListMetrics() {{
            sourceIndices.forEach((src, idx) => {{
                const metricEl = document.getElementById(`edge-metric-${{idx}}`);
                if (metricEl) {{
                    metricEl.textContent = getMetricDisplay(idx);
                }}
            }});
        }}
        
        // Toggle node visibility
        function toggleNode(idx) {{
            const item = document.getElementById(`node-item-${{idx}}`);
            if (hiddenNodes.has(idx)) {{
                hiddenNodes.delete(idx);
                item.classList.remove('hidden');
            }} else {{
                hiddenNodes.add(idx);
                item.classList.add('hidden');
            }}
            updateDiagram();
        }}
        
        // Toggle edge visibility
        function toggleEdge(idx) {{
            const item = document.getElementById(`edge-item-${{idx}}`);
            if (hiddenEdges.has(idx)) {{
                hiddenEdges.delete(idx);
                item.classList.remove('hidden');
            }} else {{
                hiddenEdges.add(idx);
                item.classList.add('hidden');
            }}
            updateDiagram();
        }}
        
        // Show all
        function showAll() {{
            hiddenNodes.clear();
            hiddenEdges.clear();
            document.querySelectorAll('.list-item').forEach(item => item.classList.remove('hidden'));
            updateDiagram();
        }}
        
        // Hide all
        function hideAll() {{
            nodeLabels.forEach((_, idx) => hiddenNodes.add(idx));
            sourceIndices.forEach((_, idx) => hiddenEdges.add(idx));
            document.querySelectorAll('.list-item').forEach(item => item.classList.add('hidden'));
            updateDiagram();
        }}
        
        // Update color inputs
        document.getElementById('sourceColor').addEventListener('input', (e) => {{
            document.getElementById('sourceColorText').value = e.target.value;
        }});
        document.getElementById('intermediateColor').addEventListener('input', (e) => {{
            document.getElementById('intermediateColorText').value = e.target.value;
        }});
        document.getElementById('targetColor').addEventListener('input', (e) => {{
            document.getElementById('targetColorText').value = e.target.value;
        }});
        document.getElementById('edgeColor').addEventListener('input', (e) => {{
            document.getElementById('edgeColorText').value = e.target.value;
            edgeColorCustomized = true;
        }});
        document.getElementById('edgeOpacity').addEventListener('input', (e) => {{
            document.getElementById('edgeOpacityValue').textContent = e.target.value + '%';
        }});
        document.getElementById('nodeWidth').addEventListener('input', (e) => {{
            document.getElementById('nodeWidthValue').textContent = e.target.value;
            updateDiagram();  // Update immediately when slider changes
        }});
        document.getElementById('fontSize').addEventListener('input', (e) => {{
            document.getElementById('fontSizeValue').textContent = e.target.value + 'px';
            updateDiagram();  // Update immediately when slider changes
        }});
        document.getElementById('metricSelect').addEventListener('change', (e) => {{
            currentMetric = e.target.value;
            updateEdgeListMetrics();  // Update edge list display
            updateDiagram();  // Redraw with new metric
        }});
        
        // Apply colors
        function applyColors() {{
            const sourceColor = document.getElementById('sourceColor').value;
            const intermediateColor = document.getElementById('intermediateColor').value;
            const targetColor = document.getElementById('targetColor').value;
            
            // Update node colors based on type
            nodeLabels.forEach((label, idx) => {{
                if (nodeTypes[idx] === 'source') nodeColors[idx] = sourceColor;
                else if (nodeTypes[idx] === 'target') nodeColors[idx] = targetColor;
                else nodeColors[idx] = intermediateColor;
            }});
            
            // Update color indicators in list
            document.querySelectorAll('.color-indicator').forEach((el, idx) => {{
                el.style.background = nodeColors[idx];
            }});
            
            updateDiagram();
        }}
        
        // Reset to defaults
        function resetToDefaults() {{
            document.getElementById('sourceColor').value = '{self.source_color}';
            document.getElementById('sourceColorText').value = '{self.source_color}';
            document.getElementById('intermediateColor').value = '{self.intermediate_color}';
            document.getElementById('intermediateColorText').value = '{self.intermediate_color}';
            document.getElementById('targetColor').value = '{self.target_color}';
            document.getElementById('targetColorText').value = '{self.target_color}';
            document.getElementById('edgeColor').value = '{edge_hex}';
            document.getElementById('edgeColorText').value = '{edge_hex}';
            document.getElementById('edgeOpacity').value = {int(edge_opacity * 100)};
            document.getElementById('edgeOpacityValue').textContent = '{int(edge_opacity * 100)}%';
            document.getElementById('nodeWidth').value = 5;
            document.getElementById('nodeWidthValue').textContent = '5';
            document.getElementById('fontSize').value = 12;
            document.getElementById('fontSizeValue').textContent = '12px';
            
            // Clearing customization restores the server-computed edge colors
            edgeColorCustomized = false;
            
            // Reset NT group colors if available
            if (hasNtTypes) {{
                document.getElementById('colorByNt').checked = {str(self.color_edges_by_nt).lower()};
                colorEdgesByNt = {str(self.color_edges_by_nt).lower()};
                document.getElementById('excitatoryColor').value = '{NT_GROUP_COLORS["excitatory"]}';
                document.getElementById('excitatoryColorText').value = '{NT_GROUP_COLORS["excitatory"]}';
                document.getElementById('inhibitoryColor').value = '{NT_GROUP_COLORS["inhibitory"]}';
                document.getElementById('inhibitoryColorText').value = '{NT_GROUP_COLORS["inhibitory"]}';
                document.getElementById('modulatoryColor').value = '{NT_GROUP_COLORS["modulatory"]}';
                document.getElementById('modulatoryColorText').value = '{NT_GROUP_COLORS["modulatory"]}';
                document.getElementById('unknownColor').value = '{NT_GROUP_COLORS["unknown"]}';
                document.getElementById('unknownColorText').value = '{NT_GROUP_COLORS["unknown"]}';
                ntGroupColors['excitatory'] = '{NT_GROUP_COLORS["excitatory"]}';
                ntGroupColors['inhibitory'] = '{NT_GROUP_COLORS["inhibitory"]}';
                ntGroupColors['modulatory'] = '{NT_GROUP_COLORS["modulatory"]}';
                ntGroupColors['unknown'] = '{NT_GROUP_COLORS["unknown"]}';
            }}
            
            // Reset node colors
            nodeLabels.forEach((label, idx) => {{
                if (nodeTypes[idx] === 'source') nodeColors[idx] = '{self.source_color}';
                else if (nodeTypes[idx] === 'target') nodeColors[idx] = '{self.target_color}';
                else nodeColors[idx] = '{self.intermediate_color}';
            }});
            
            // Reset metric, zoom, labels, and background
            document.getElementById('metricSelect').value = 'weight';
            currentMetric = 'weight';
            zoomLevel = 1.0;
            applyZoom();
            labelsVisible = true;
            const labelsBtn = document.getElementById('toggleLabelsBtn');
            if (labelsBtn) labelsBtn.textContent = '🏷️ Hide Labels';
            if (typeof bgCtrl !== 'undefined' && bgCtrl.reset) bgCtrl.reset();
            
            showAll();
        }}
        
        // Update diagram with current settings
        function updateDiagram() {{
            const edgeColor = document.getElementById('edgeColor').value;
            const edgeOpacity = document.getElementById('edgeOpacity').value / 100;
            const nodeWidth = parseInt(document.getElementById('nodeWidth').value);
            
            // Create new data arrays with visibility applied
            const visibleNodeColors = nodeColors.map((color, idx) => 
                hiddenNodes.has(idx) ? 'rgba(200,200,200,0.1)' : color
            );
            
            const visibleNodeLabels = nodeLabels.map((label, idx) => {{
                if (hiddenNodes.has(idx)) return '';
                if (!labelsVisible) return '';  // Hide labels when toggled off
                return label;
            }});
            
            const visibleSourceIndices = [];
            const visibleTargetIndices = [];
            const visibleWeights = [];
            const visibleLinkColors = [];
            const visibleHoverText = [];
            
            // Get current metric values
            let metricValues;
            if (currentMetric === 'ratio') {{
                metricValues = ratios;
            }} else if (currentMetric === 'prob') {{
                metricValues = probs;
            }} else {{
                metricValues = weights;
            }}
            
            // Metric display name for hover text
            const metricDisplayName = (currentMetric === 'ratio') ? 'Ratio' : (currentMetric === 'prob' ? 'Probability' : '{self._sankey_weight_name}');
            
            // Ratio/probability are 0..1; normalize them to the synapse-weight
            // scale so link widths stay comparable without a magic ×1000 factor.
            const maxAbsWeight = Math.max(...weights.map(Math.abs), 0);
            const metricScaleFactor = (currentMetric === 'ratio' || currentMetric === 'prob') && maxAbsWeight > 0 ? maxAbsWeight : 1;
            
            sourceIndices.forEach((src, idx) => {{
                const tgt = targetIndices[idx];
                if (hiddenEdges.has(idx) || hiddenNodes.has(src) || hiddenNodes.has(tgt)) {{
                    // Make edge nearly invisible
                    visibleSourceIndices.push(src);
                    visibleTargetIndices.push(tgt);
                    visibleWeights.push(0.001);  // Tiny value
                    visibleLinkColors.push('rgba(200,200,200,0.05)');
                    visibleHoverText.push('');
                }} else {{
                    visibleSourceIndices.push(src);
                    visibleTargetIndices.push(tgt);
                    
                    // Use selected metric value
                    const metricValue = metricValues[idx];
                    // Normalize ratio/prob to the weight scale (see above)
                    const scaledValue = (currentMetric === 'ratio' || currentMetric === 'prob') 
                        ? metricValue * metricScaleFactor
                        : metricValue;
                    visibleWeights.push(scaledValue);
                    
                    // Build custom hover text with ORIGINAL (unscaled) values
                    const valueStr = (currentMetric === 'ratio' || currentMetric === 'prob') 
                        ? metricValue.toFixed(4)
                        : Math.round(metricValue).toLocaleString();
                    
                    // Add NT info to hover text if available
                    let ntInfo = '';
                    if (hasNtTypes && ntTypes[idx] && ntTypes[idx] !== 'unknown') {{
                        ntInfo = '<br><b>NT Type:</b> ' + escapeHtml(ntTypes[idx].toUpperCase());
                    }}
                    const hoverStr = '<b>Source:</b> ' + escapeHtml(nodeLabels[src]) + '<br><b>Target:</b> ' + escapeHtml(nodeLabels[tgt]) + '<br><b>' + metricDisplayName + ':</b> ' + valueStr + ntInfo;
                    visibleHoverText.push(hoverStr);
                    
                    // Determine edge color based on NT group (if enabled) or default coloring
                    const isNegative = originalWeights[idx] < 0;
                    let r, g, b;
                    
                    if (colorEdgesByNt && hasNtTypes) {{
                        // Color by NT group
                        const ntColor = getEdgeColorByNtGroup(idx);
                        r = parseInt(ntColor.substr(1,2), 16);
                        g = parseInt(ntColor.substr(3,2), 16);
                        b = parseInt(ntColor.substr(5,2), 16);
                    }} else if (isNegative) {{
                        // Light blue for negative edges: #4A90E2 = rgb(74, 144, 226)
                        r = 74;
                        g = 144;
                        b = 226;
                    }} else {{
                        // Server-computed initial color (NT/custom/link_color aware)
                        // until the user picks a custom edge color in the UI.
                        const serverRGB = (!edgeColorCustomized && initialEdgeColors && initialEdgeColors[idx])
                            ? parseColorStringRGB(initialEdgeColors[idx])
                            : null;
                        if (serverRGB) {{
                            r = serverRGB.r;
                            g = serverRGB.g;
                            b = serverRGB.b;
                        }} else {{
                            // Use selected edge color for positive edges
                            r = parseInt(edgeColor.substr(1,2), 16);
                            g = parseInt(edgeColor.substr(3,2), 16);
                            b = parseInt(edgeColor.substr(5,2), 16);
                        }}
                    }}
                    
                    visibleLinkColors.push(`rgba(${{r}},${{g}},${{b}},${{edgeOpacity}})`);
                }}
            }});

            // Get current font size
            const fontSize = parseInt(document.getElementById('fontSize').value);
            
            // Build title with simplification note if applicable
            let title = '{self._sankey_title_js}';
            const simplificationApplied = {str(simplification_applied).lower()};
            const originalEdgeCount = {original_edge_count};
            if (simplificationApplied) {{
                title += '<br><sub style="font-size: 0.8em; color: #e67e22;">⚠️ Simplified: showing complete strong paths within the visualization edge limit</sub>';
            }}

            Plotly.react(
                document.querySelector('.plotly-graph-div'),
                [{{
                    type: 'sankey',
                    node: {{
                        pad: 5,
                        thickness: nodeWidth,
                        line: {{ color: "black", width: 0 }},
                        label: visibleNodeLabels,
                        color: visibleNodeColors
                    }},
                    link: {{
                        source: visibleSourceIndices,
                        target: visibleTargetIndices,
                        value: visibleWeights,
                        color: visibleLinkColors,
                        customdata: visibleHoverText,
                        hovertemplate: '%{{customdata}}<extra></extra>'
                    }}
                }}],
                {{
                    title: {{
                        text: title,
                        font: {{ size: fontSize }}
                    }},
                    font: {{ size: fontSize }},
                    hoverlabel: {{
                        align: 'left',
                        namelength: -1,
                        font: {{ size: 14, family: 'Arial, sans-serif' }},
                        bgcolor: 'rgba(255, 255, 255, 0.95)',
                        bordercolor: '#333',
                        pad: 15
                    }}
                }}
            );
        }}
        
        // Zoom functionality
        let zoomLevel = 0.7;  // Start zoomed out to see full diagram
        const zoomStep = 0.2;
        const minZoom = 0.3;
        const maxZoom = 3.0;
        
        function zoomIn() {{
            zoomLevel = Math.min(maxZoom, zoomLevel + zoomStep);
            applyZoom();
        }}
        
        function zoomOut() {{
            zoomLevel = Math.max(minZoom, zoomLevel - zoomStep);
            applyZoom();
        }}
        
        function resetZoom() {{
            zoomLevel = 1.0;
            applyZoom();
        }}
        
        function applyZoom() {{
            const container = document.querySelector('#sankey-container > div');
            if (container) {{
                container.style.transform = `scale(${{zoomLevel}})`;
                container.style.transformOrigin = 'center center';
            }}
        }}
        
        // Label toggle functionality
        let labelsVisible = true;
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            const btn = document.getElementById('toggleLabelsBtn');
            if (btn) {{
                btn.textContent = labelsVisible ? '🏷️ Hide Labels' : '🏷️ Show Labels';
            }}
            updateDiagram();
        }}
        
        // Background color toggle (shared controller)
        const bgCtrl = createBackgroundController(['#ffffff', '#000000'], ['White', 'Black'], applyBackground);
        
        function toggleBackground() {{
            bgCtrl.toggle();
        }}
        
        function applyBackground(color) {{
            document.body.style.background = color;
            document.getElementById('sankey-container').style.background = color;
            const gd = document.querySelector('.plotly-graph-div');
            if (gd) {{
                Plotly.relayout(gd, {{
                    'paper_bgcolor': color,
                    'plot_bgcolor': color
                }});
            }}
            // Adjust text color for dark backgrounds
            const isDark = isColorDark(color);
            document.querySelectorAll('.control-label, .slider-value').forEach(el => {{
                el.style.color = isDark ? '#e0e0e0' : '#555';
            }});
        }}
        
        function applyCustomBackground() {{
            bgCtrl.applyCustom();
        }}
        
{SHARED_JS}
        
        // Export functions (shared backend)
        function exportPNG() {{
            const scale = getExportScale('exportScale', 2, 4);
            const gd = document.querySelector('.plotly-graph-div');
            exportPlotlyToImage(gd, 'png', 'sankey_diagram_' + scale + 'x.png', scale, gd.offsetWidth, gd.offsetHeight);
        }}
        
        function exportSVG() {{
            const gd = document.querySelector('.plotly-graph-div');
            exportPlotlyToImage(gd, 'svg', 'sankey_diagram.svg', 1, gd.offsetWidth, gd.offsetHeight);
        }}
        
        // Initialize on load
        initializeLists();
        applyZoom();  // Apply initial zoom level
        updateDiagram();  // Initialize diagram with JavaScript control
    </script>
</body>
</html>"""
        
        return html

    def _path_edge_records(self, row):
        """Return the non-zero edges and per-edge metadata for one path row.

        A path is only useful for visualization when every hop has a usable
        weight.  In particular, this helper deliberately does not return a
        source-outgoing or target-incoming edge by itself: callers select the
        complete path row and then add all of its edges as one unit.
        """
        nodes = self._parse_path_block(row.get('path_block', ''))
        weights = self._safe_eval_list(row.get('weights', []))
        ratios = self._safe_eval_list(row.get('connection_ratios', []))
        probs = self._safe_eval_list(row.get('traversal_probabilities', []))
        if len(nodes) < 2 or len(weights) < len(nodes) - 1:
            return []

        records = []
        for i in range(len(nodes) - 1):
            weight = weights[i]
            if weight == 0:
                # A zero-weight hop is not present in G_network, so retaining
                # the rest of this path would manufacture a broken path.
                return []
            records.append((
                (nodes[i], nodes[i + 1]),
                {
                    'weight': weight,
                    'ratio': ratios[i] if i < len(ratios) else 0,
                    'probability': probs[i] if i < len(probs) else 0,
                },
            ))
        return records

    def _path_selection_score(self, row, records):
        """Return a stable strength key for complete-path selection.

        §6c: ranked through the shared strongest-selection core
        (``strongest_core.path_rank_key``) so display selection and
        StrongestFirst enumeration share one definition of "strongest":
        primary = full-path bottleneck; the interior-weakest-hop strength
        is kept as the first tie-breaker (display policy — boundary hops
        are often weak solely because they touch the query boundary), and
        path probability as the second.
        """
        weights = [data['weight'] for _, data in records]
        if not weights:
            return (0, 0, 0)
        try:
            from .strongest_core import path_rank_key
        except ImportError:  # src laid bare on sys.path
            from strongest_core import path_rank_key
        path_probability = row.get('path_prob', row.get('path_probability', 0))
        try:
            path_probability = float(path_probability)
        except (TypeError, ValueError):
            path_probability = 0.0
        return path_rank_key(weights, path_probability)

    @staticmethod
    def _edges_on_source_target_corridor(edges, source_nodes, target_nodes):
        """Keep only edges on a source-to-target corridor in ``edges``.

        This is the safe fallback for edge-list input, where explicit path
        rows are unavailable.  An edge survives only when its source is
        reachable from a source node and its target can reach a target node.
        Thus a target-incoming edge such as ``Mi1 -> target`` is removed when
        the selected graph has no selected route into ``Mi1``.
        """
        edge_set = set(edges)
        if not edge_set or not source_nodes or not target_nodes:
            return set()

        forward = set(source_nodes)
        changed = True
        while changed:
            changed = False
            for u, v in edge_set:
                if u in forward and v not in forward:
                    forward.add(v)
                    changed = True

        backward = set(target_nodes)
        changed = True
        while changed:
            changed = False
            for u, v in edge_set:
                if v in backward and u not in backward:
                    backward.add(u)
                    changed = True

        return {(u, v) for u, v in edge_set
                if u in forward and v in backward}

    def _select_edges_for_plot(self):
        """Compute the edge set to draw under the Visualization Edge Limit.

        The network and heatmap share this selector.  When path rows are
        available, complete paths are the selection unit: a source-outgoing
        or target-incoming edge is retained only as part of a selected path.
        This prevents endpoint preservation from creating one-sided dead ends.
        Paths are ranked by interior strength first, then path probability and
        complete-path minimum weight, and are added without exceeding the
        unique-edge limit whenever possible.

        For edge-list input, the selector uses a bounded set of strong edges
        and then removes edges that do not lie on a source-to-target corridor.

        Returns None when no trimming is needed, else
        ``(kept_edges, boundary_capped, integrity_relaxed, selected_paths,
        threshold)``.  ``selected_paths`` is a list of original path-row
        indexes for path input and None for the edge-list fallback.
        """
        G = getattr(self, 'G_network', None)
        if self.edgeN_limit <= 0 or G is None \
                or G.number_of_edges() <= self.edgeN_limit:
            return None
        self.edge_limit_trimmed = True

        path_df = getattr(self, 'path_df', None)
        if path_df is not None and 'path_block' in path_df.columns \
                and 'weights' in path_df.columns:
            candidates = []
            for order, (idx, row) in enumerate(path_df.iterrows()):
                records = self._path_edge_records(row)
                if not records:
                    continue
                candidates.append((
                    self._path_selection_score(row, records),
                    order,
                    idx,
                    records,
                ))
            candidates.sort(key=lambda item: item[0], reverse=True)

            edge_data_dict = {}
            selected_paths = []
            integrity_relaxed = False
            for _score, _order, idx, records in candidates:
                candidate_edges = {edge for edge, _data in records}
                new_edges = candidate_edges.difference(edge_data_dict)
                if (len(edge_data_dict) + len(new_edges)
                        > self.edgeN_limit):
                    # Skip this path and continue looking for a complete path
                    # that still fits.  This avoids the old behavior where
                    # the first path crossing the limit overshot it.
                    continue
                for edge_key, data in records:
                    if edge_key not in edge_data_dict:
                        edge_data_dict[edge_key] = dict(data)
                    else:
                        edge_data_dict[edge_key]['weight'] = max(
                            edge_data_dict[edge_key]['weight'], data['weight'])
                        edge_data_dict[edge_key]['ratio'] = max(
                            edge_data_dict[edge_key]['ratio'], data['ratio'])
                        edge_data_dict[edge_key]['probability'] = max(
                            edge_data_dict[edge_key]['probability'],
                            data['probability'])
                selected_paths.append(idx)

            # A single path longer than the configured limit is irreducible:
            # show that complete path rather than an empty/broken graph.  This
            # is recorded explicitly so callers can explain the soft limit.
            if not selected_paths and candidates:
                _score, _order, idx, records = candidates[0]
                edge_data_dict = {edge: dict(data) for edge, data in records}
                selected_paths = [idx]
                integrity_relaxed = len(edge_data_dict) > self.edgeN_limit

            kept_weights = [d['weight'] for d in edge_data_dict.values()]
            try:
                from .strongest_core import selection_threshold
            except ImportError:  # src laid bare on sys.path
                from strongest_core import selection_threshold
            threshold = selection_threshold(kept_weights) or 0
            return (edge_data_dict, False, integrity_relaxed,
                    selected_paths, threshold)

        # Edge-list fallback.  Endpoint edges are candidates, not guaranteed
        # reservations.  Build a bounded candidate graph, then remove edges
        # that cannot participate in a complete source-to-target corridor.
        all_edges = list(G.edges(data=True))
        source_nodes = {
            node for node, attrs in G.node_attrs.items()
            if attrs.get('node_type') == 'source'
        }
        target_nodes = {
            node for node, attrs in G.node_attrs.items()
            if attrs.get('node_type') == 'target'
        }
        boundary_candidates = []
        for u, v, data in all_edges:
            if u in source_nodes or v in target_nodes:
                boundary_candidates.append((u, v, data))
        boundary_candidates.sort(
            key=lambda item: abs(item[2].get('weight', 0)), reverse=True)
        selected_boundary = boundary_candidates[:self.edgeN_limit]
        boundary_capped = len(boundary_candidates) > len(selected_boundary)
        boundary_keys = {(u, v) for u, v, _data in selected_boundary}

        ordinary = sorted(
            ((u, v, data) for u, v, data in all_edges
             if (u, v) not in boundary_keys),
            key=lambda item: abs(item[2].get('weight', 0)), reverse=True,
        )
        candidate_records = selected_boundary + ordinary[:self.edgeN_limit]
        candidate_keys = {(u, v) for u, v, _data in candidate_records}
        corridor_keys = self._edges_on_source_target_corridor(
            candidate_keys, source_nodes, target_nodes)
        if corridor_keys:
            kept_keys = corridor_keys
        else:
            kept_keys = set(list(candidate_keys)[:self.edgeN_limit])

        kept_edges = {}
        for u, v, data in candidate_records:
            if (u, v) in kept_keys:
                kept_edges[(u, v)] = dict(data)
        kept_weights = [d.get('weight', 0) for d in kept_edges.values()]
        threshold = min(kept_weights) if kept_weights else 0
        return (kept_edges, boundary_capped, False, None, threshold)

    def visualized_paths_for_export(self):
        """Return only path rows represented by the trimmed network.

        The raw ``*_data_original_paths.csv`` remains the complete path
        result.  This companion frame is the reproducible path input for the
        rendered, edge-limited network and contains no independently-added
        endpoint edges.
        """
        path_df = getattr(self, 'path_df', None)
        if path_df is None or 'path_block' not in path_df.columns \
                or 'weights' not in path_df.columns:
            return pd.DataFrame()
        selected = self._select_edges_for_plot()
        if selected is None:
            return path_df.copy()
        selected_paths = selected[3]
        if selected_paths is None:
            return pd.DataFrame(columns=path_df.columns)
        return path_df.loc[selected_paths].copy().reset_index(drop=True)

    def _filter_conn_df_for_plot(self, conn_df):
        """Filter a connection DataFrame to the SAME edge set the network
        draws under the Visualization Edge Limit (complete paths or a
        source-to-target corridor). Returns the frame unchanged when no
        trimming applies."""
        selected = self._select_edges_for_plot()
        if selected is None:
            return conn_df
        kept_keys = set(selected[0])
        mask = [(str(s), str(t)) in kept_keys
                for s, t in zip(conn_df['source'], conn_df['target'])]
        return conn_df[mask]

    def _trim_network_for_plot(self):
        """
        Trim G_network to complete strong paths for plotting.

        Endpoint edges are not independently reserved.  They survive only
        when their complete source-to-target path is selected.  Returns the
        graph to plot (the original network when no trimming is needed) and
        prints the trim warning with the applied weight threshold.
        """
        G_to_plot = self.G_network
        selected = self._select_edges_for_plot()
        if selected is None:
            return G_to_plot
        kept_edges, boundary_capped, integrity_relaxed, selected_paths, threshold = selected

        if boundary_capped:
            self._vprint(
                f'  (edge-list endpoint candidates capped at {self.edgeN_limit} strongest edges)')
        if integrity_relaxed:
            self._vprint(
                f'  (complete-path integrity relaxed the {self.edgeN_limit}-edge limit because the strongest path is longer)')
        self._vprint(
            f'\033[33m⚠️ Too many edges ({self.G_network.number_of_edges()}) - simplifying to complete strong paths within the {self.edgeN_limit}-edge limit (endpoint edges are kept only with their paths)\033[0m')

        # Create subgraph with the kept edges
        G_sub = FastGraph()
        for (u, v), data in kept_edges.items():
            G_sub.add_edge(u, v, **data)
            if u in self.G_network.node_attrs:
                G_sub.node_attrs[u].update(self.G_network.node_attrs[u])
            if v in self.G_network.node_attrs:
                G_sub.node_attrs[v].update(self.G_network.node_attrs[v])
        G_to_plot = G_sub

        if selected_paths is not None:
            self._vprint(
                f'  Selected {len(selected_paths)} complete paths → {G_to_plot.number_of_edges()} edges (applied threshold: weight >= {threshold:g})')
        else:
            self._vprint(
                f'  Kept {G_to_plot.number_of_edges()} edge-list edges on a source-to-target corridor (applied threshold: weight >= {threshold:g})')
        return G_to_plot

    def create_network(self):
        """
        Create interactive Cytoscape.js network visualization.
        
        Creates a fully interactive network graph with:
        - Draggable nodes
        - Hover tooltips showing weight/ratio/probability
        - Right-click to hide nodes
        - Keyboard shortcut (H) to hide selected nodes
        - Export to PNG functionality
        
        Returns
        -------
        str
            Path to the generated HTML file
            
        Notes
        -----
        - Layout algorithm specified by network_layout parameter
        - Edge thickness represents connection weight
        - Node colors follow node_type (source/intermediate/target)
        """
        if self.G_network is None:
            self.build_network()
        
        self._vprint("\nCreating interactive network visualization...")

        G_to_plot = self._trim_network_for_plot()

        output_path = os.path.join(self.output_folder, self.base_filename + '_network.html')
        
        self._plot_cytoscape_network(
            G_to_plot,
            output_path=output_path,
            layout=self.network_layout
        )
        
        self._vprint(f"Network graph saved: {output_path}")
        
        return output_path
    
    def generate_empty_network_html(self):
        """
        Generate an empty network HTML template without any data.
        
        Creates a blank interactive network visualization that can be used as a
        template or populated later with custom data via JavaScript.
        
        Returns
        -------
        str
            Path to the generated empty network HTML file
            
        Notes
        -----
        - Creates a minimal Cytoscape.js network with no nodes or edges
        - Includes all interactive controls (layout, filters, export)
        - Useful for creating templates or testing the visualization interface
        - The HTML includes JavaScript code to dynamically add nodes/edges
        
        Examples
        --------
        >>> vp = VisualizePath(path_file=None, generate_empty_network=True)
        >>> vp.generate_empty_network_html()
        'empty_network/empty_network_20251109_143052_network.html'
        """
        self._vprint("\nGenerating empty network HTML template...")
        self._vprint(f"  Filename: {self.base_filename}_network.html")
        
        # Create an empty FastGraph
        G = FastGraph()
        
        output_path = os.path.join(self.output_folder, self.base_filename + '_network.html')
        
        # Use the existing method with an empty graph. The canvas must open in
        # a FRESH tab (handled below), so suppress the same-window open here —
        # otherwise the file opens twice (once per mechanism).
        self._plot_cytoscape_network(
            G,
            output_path=output_path,
            layout=self.network_layout,
            open_browser=False,
        )
        
        self._vprint(f"✓ Empty network HTML generated: {output_path}")
        self._vprint(f"  Timestamp ensures unique filename for each generation")
        self._vprint(f"  This template includes all interactive controls")
        self._vprint(f"  You can populate it with custom nodes/edges via JavaScript")
        
        if self.showfig:
            # Empty canvases are a user-facing drawing surface; always open a
            # fresh browser tab so the generated canvas is immediately ready.
            webbrowser.open_new_tab('file://' + os.path.abspath(output_path))
        
        return output_path
    
    def _calculate_edge_widths(self, weights):
        """
        Calculate edge widths from weights using specified scaling method.
        
        Parameters
        ----------
        weights : list or np.ndarray
            Edge weights to scale
            
        Returns
        -------
        list
            Scaled edge widths
        """
        weights = np.array(weights, dtype=float)
        
        # Handle zero or negative weights
        weights = np.maximum(weights, 1e-6)
        
        if self.edge_width_scale == 'linear':
            # Linear scaling: width ∝ weight
            scaled = weights
        elif self.edge_width_scale == 'log':
            # Logarithmic scaling: width ∝ log_base(weight)
            if self.edge_width_log_base == 'e' or self.edge_width_log_base is None:
                # Natural logarithm (base e)
                scaled = np.log(weights + 1)  # +1 to avoid log(0)
            else:
                # Custom base: log_b(x) = ln(x) / ln(b)
                log_base = float(self.edge_width_log_base)
                if log_base <= 1:
                    self._vprint(f"Warning: Invalid log base {log_base}, using natural log (e)")
                    scaled = np.log(weights + 1)
                else:
                    scaled = np.log(weights + 1) / np.log(log_base)
        elif self.edge_width_scale == 'sqrt':
            # Square root scaling: width ∝ √weight
            scaled = np.sqrt(weights)
        elif self.edge_width_scale == 'none':
            # No scaling: constant width
            scaled = np.ones_like(weights)
        else:
            # Default to log if unknown method
            self._vprint(f"Warning: Unknown edge_width_scale '{self.edge_width_scale}', using 'log'")
            scaled = np.log(weights + 1)
        
        # Apply multiplier factor
        scaled = scaled * self.edge_width_factor
        
        return scaled.tolist()
    
    def _plot_cytoscape_network(self, G, output_path, layout='hierarchical', open_browser=True):
        """
        Create Cytoscape.js network visualization (internal method).
        
        Parameters
        ----------
        G : networkx.DiGraph
            Network graph to visualize
        output_path : str
            Path to save HTML file
        layout : str
            Layout algorithm: 'hierarchical', 'spring', 'circular', 'distributed'
        """
        # Prepare node data
        nodes_data = []
        has_hemisphere_nodes = False

        def _extract_hemisphere(label: str):
            base_label = label
            if '(' in label:
                base_label = label.split('(')[0].strip()
            hemi = None
            if base_label.endswith(('_L', '_R', '_U')):
                hemi = base_label[-1]
                base_label = base_label[:-2]
            return base_label, hemi

        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'intermediate')

            # Assign color: an explicit per-node 'color' attribute on the
            # graph wins (e.g. linker nodes colored by their metadata
            # column), otherwise fall back to the node-type palette.
            if node_type == 'source':
                base_color = self.node_color[0]
            elif node_type == 'target':
                base_color = self.target_color
            else:  # intermediate
                base_color = self.node_color[1]
            override = G.nodes[node].get('color')
            if override:
                base_color = override

            # Declared groups: the node's 'group' attribute names a
            # declared group (the dataset code in the mapping views).  The
            # group name becomes the grouping identity — it drives every
            # [node_type = "..."] group operation — and supplies the
            # color; the structural role is kept in 'role' for the hover.
            role = node_type
            group_name = ''
            if self.node_groups:
                declared = next(
                    (g for g in self.node_groups
                     if g.get('name') == G.nodes[node].get('group')), None)
                if declared is not None:
                    group_name = str(declared.get('name') or '')
                    node_type = group_name
                    if not override:
                        base_color = declared.get('color', base_color)

            base_name, hemisphere = _extract_hemisphere(str(node))
            if hemisphere:
                has_hemisphere_nodes = True

            # Normalize color to hex and apply hemisphere desaturation if needed
            base_hex, _ = parse_color_to_hex_opacity(base_color)
            color = base_hex
            if self.separate_hemispheres and hemisphere:
                if hemisphere == self.hemisphere_desaturate_side:
                    color = blend_with_gray(base_hex, self.hemisphere_desaturate_factor)

            # Get dataset info for this node (for hover labels)
            ds_info = self.node_dataset_info.get(node, {})

            nodes_data.append({
                'data': {
                    'id': node,
                    'label': G.nodes[node].get('label', node),
                    'node_type': node_type,
                    'group': group_name,
                    'role': role,
                    # Single-valued live group membership ('' = Unassigned);
                    # seeded with the structural/declared identity so the
                    # load-time look is unchanged. node_type stays immutable.
                    'assigned_group': node_type,
                    'hemisphere': hemisphere if hemisphere else '',
                    'base_name': base_name,
                    'color': color,
                    'dataset_info': ds_info  # {code: name_in_that_dataset}
                },
                # Preset layouts honor this top-level position verbatim
                # (the mapping bridges preset); every other layout overwrites
                # it when it runs. Nodes without a precomputed position
                # start at the origin and are placed by their layout.
                'position': G.nodes[node].get('position') or {},
                'classes': ''  # For CSS classes
            })

        # Node groups beyond the standard three (the type-mapping views'
        # 'linker' and 'entry' nodes) become first-class groups in the
        # network UI: quick-action button, color/opacity dropdown entry,
        # selection, and reset all honor them — without this, group
        # operations silently skip those nodes.
        standard_node_groups = ('source', 'intermediate', 'target')
        extra_node_groups: list = []
        if self.node_groups:
            # Declared groups are authoritative: dataset groups in the
            # mapping views, with the labels/colors as declared.  Names
            # are restricted to the same selector-safe charset as custom
            # groups — they end up in HTML attributes, onclick strings
            # and cytoscape attribute selectors; labels keep their raw
            # form for the JSON/DOM paths and are html-escaped at the
            # direct template injections below.
            for g in self.node_groups:
                name = re.sub(r'[^A-Za-z0-9_-]', '_',
                              str(g.get('name') or ''))
                if not name or any(x['name'] == name
                                   for x in extra_node_groups):
                    continue
                extra_node_groups.append({
                    'name': name,
                    'label': str(g.get('label') or name),
                    'color': str(g.get('color') or '#94a3b8'),
                })
        else:
            for nd in nodes_data:
                nt = nd['data'].get('node_type')
                if not nt or nt in standard_node_groups:
                    continue
                if all(g['name'] != nt for g in extra_node_groups):
                    extra_node_groups.append({
                        'name': nt,
                        'label': str(nt).capitalize(),
                        'color': nd['data'].get('color', '#94a3b8'),
                    })
        extra_node_group_options = ''.join(
            f'<option value="{g["name"]}">{html_escape(g["label"])} Nodes</option>'
            for g in extra_node_groups)
        extra_group_buttons = ''.join(
            f'<button class="btn" onclick="selectGroup(\'{g["name"]}\')" '
            f'title="Select all {html_escape(g["label"].lower())} nodes" '
            f'style="font-size: 10px; padding: 5px; background: {g["color"]};">'
            f'{html_escape(g["label"])}</button>'
            for g in extra_node_groups)
        if extra_group_buttons:
            extra_group_buttons = (
                '<div style="display: grid; grid-template-columns: 1fr 1fr; '
                'gap: 6px; margin-bottom: 8px;">' + extra_group_buttons + '</div>')
        extra_group_defaults_js = (
            'const extraNodeGroups = ' + json.dumps(extra_node_groups) + ';\n'
            '        const datasetLegendCodes = new Set('
            + json.dumps(sorted((self.dataset_legend or {}).keys())) + ');\n'
            '        extraNodeGroups.forEach(g => {\n'
            '            originalGroupDefaults[g.name] = { color: g.color, opacity: 100 };\n'
            '        });')
        
        # Build NT type lookup from conn_df if available
        nt_lookup = {}
        if self.conn_df is not None and 'nt_type' in self.conn_df.columns:
            for _, row in self.conn_df.iterrows():
                nt_lookup[(row['source'], row['target'])] = row.get('nt_type', None)
        
        # Prepare edge data
        edges_data = []
        edge_weights = []  # Collect weights for scaling
        has_negative = False  # Track if any negative weights exist
        has_nt_coloring_network = False  # Track if NT coloring applies
        unique_nts_network = set()  # Collect unique NT types for CSS
        
        for source, target, data in G.edges(data=True):
            weight = data.get('weight', 0)
            is_negative = weight < 0
            if is_negative:
                has_negative = True
            
            # Convert all to positive for plotting (avoid Cytoscape issues)
            abs_weight = abs(weight)
            edge_weights.append(abs_weight)
            ratio = data.get('ratio', np.nan)
            prob = data.get('probability', np.nan)
            
            # Get NT type from lookup
            nt_type = nt_lookup.get((source, target), None)
            if self.color_edges_by_nt and nt_type is not None:
                has_nt_coloring_network = True
                unique_nts_network.add(nt_type)
            
            # Format tooltip - use actual newline character, not escaped.
            # Integer-valued counts drop the float '.0' (edge lists often
            # parse synapse counts as floats).
            weight_disp = (f"{int(weight):,}"
                           if float(weight).is_integer() else f"{weight:,}")
            tooltip_parts = [f"Weight: {weight_disp} {self.edge_weight_label}"]
            if not np.isnan(ratio):
                tooltip_parts.append(f"Ratio: {ratio:.3f}")
            if not np.isnan(prob):
                tooltip_parts.append(f"Probability: {prob:.3f}")
            if nt_type:
                tooltip_parts.append(f"NT: {nt_type}")
            
            # Add custom edge labels (e.g., multi-dataset synapse strengths)
            if self.edge_labels and (source, target) in self.edge_labels:
                custom_labels = self.edge_labels[(source, target)]
                if isinstance(custom_labels, dict):
                    for label_name, label_value in custom_labels.items():
                        if isinstance(label_value, (int, float)):
                            tooltip_parts.append(f"{label_name}: {label_value:,}")
                        else:
                            tooltip_parts.append(f"{label_name}: {label_value}")
            
            tooltip = "\n".join(tooltip_parts)  # Use actual newline, not escaped
            
            # Store custom labels in edge data for JavaScript hover handling
            custom_labels_json = {}
            if self.edge_labels and (source, target) in self.edge_labels:
                custom_labels_json = self.edge_labels[(source, target)]
            
            # Per-edge color from the uploaded edge list (the optional
            # 'color' column) wins over the UI link color, which is only
            # the fallback for edges the file does not color.
            edge_color = self.edge_color
            if self.custom_edge_colors and (source, target) in self.custom_edge_colors:
                edge_color = self.custom_edge_colors[(source, target)]
            
            edges_data.append({
                'data': {
                    'source': source,
                    'target': target,
                    'weight': abs_weight,  # Store positive for Cytoscape
                    'original_weight': weight,  # Store original for hover modification
                    'is_negative': 1 if is_negative else 0,  # Use 1/0 instead of True/False for JavaScript
                    'nt_type': nt_type if nt_type else '',  # Store NT type for CSS styling
                    'color': edge_color,  # Per-edge color (file color or link_color fallback)
                    'ratio': ratio if not np.isnan(ratio) else 0,
                    'probability': prob if not np.isnan(prob) else 0,
                    'tooltip': tooltip,
                    'custom_labels': custom_labels_json  # Store for JS access
                }
            })
        
        if has_negative:
            self._vprint(f"  ℹ️  Found negative edge weights - using absolute values for width, light blue color for negative edges")
        if has_nt_coloring_network:
            self._vprint(f"  🧬 Applied neurotransmitter-based edge coloring in network")
        
        # Calculate scaled edge widths
        scaled_widths = self._calculate_edge_widths(edge_weights)
        
        # Add scaled width to each edge
        for i, edge in enumerate(edges_data):
            edge['data']['scaled_width'] = scaled_widths[i]
        
        # Calculate min and max scaled widths for mapData function
        min_scaled_width = min(scaled_widths) if scaled_widths else 1
        max_scaled_width = max(scaled_widths) if scaled_widths else 10
        
        # Extract output name and add timestamp for unique storage key
        import os
        from datetime import datetime
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        timestamp_hash = datetime.now().strftime('%Y%m%d%H%M%S')
        storage_key = f"cytoscape_layout_{output_name}#{timestamp_hash}"
        
        # Map layout names to Cytoscape.js layout algorithms
        layout_map = {
            'hierarchical': 'dagre',        # Dagre - Best for hierarchical graphs (crossing minimization)
            'layered': 'dagre',             # Layered (multipartite) - same dagre engine, explicit alias
            'spring': 'cose',               # CoSE - Force-directed layout
            'circular': 'circle',           # Circular layout
            'distributed': 'dagre',         # Changed to Dagre (better for pathway networks)
            'shell': 'concentric',          # Shell - concentric rings around a center
            'dagre': 'dagre',               # Dagre (Sugiyama algorithm)
            'cose-bilkent': 'cose-bilkent', # CoSE Bilkent - Better quality force-directed
            'fcose': 'fcose',               # fCoSE - Fast CoSE with quality
            'klay': 'klay',                 # KLay - Layer-based layout (like dagre)
            'elk': 'elk',                   # ELK - Eclipse Layout Kernel
            'mapping': 'preset'             # Mapping bridges: layered LR preset
        }
        cytoscape_layout = layout_map.get(layout, 'dagre')
        # JS-facing layout name: the dropdown option value getLayoutConfig
        # is keyed by. The mapping bridges keep their friendly name so the
        # The mapping preset is programmatic-only: dagre is the mapping
        # default, so the dropdown lists it ONLY in documents rendered
        # with the preset (the linker-path view) — selected, so
        # changeLayout()/reset and the initial selector assignment stay
        # consistent.
        js_layout_name = (
            layout if cytoscape_layout == 'preset' else cytoscape_layout
        )
        mapping_layout_optgroup = ""
        if js_layout_name == "mapping":
            mapping_layout_optgroup = (
                '<optgroup label="🔀 Type Mapping (preset)">'
                '<option value="mapping" selected>'
                'Mapping (layered L→R, preset)</option></optgroup>')

        # Which dropdown option carries the `selected` marker.  The option
        # values are the JS-facing names, so this compares against
        # js_layout_name.  It cannot be inlined in the HTML template below:
        # every option there is written with doubled braces (the template is
        # an f-string), so an inline conditional would ship its Python source
        # as HTML text and no option would ever pre-select.
        def _layout_selected(option: str) -> str:
            return 'selected' if js_layout_name == option else ''
        
        # Generate NT-based edge styles if enabled
        nt_edge_styles = ""
        nt_edge_group_options = ""  # For dropdown selector
        if has_nt_coloring_network and unique_nts_network:
            nt_style_parts = []
            nt_option_parts = []
            for nt in sorted(unique_nts_network):
                # Get color without opacity for Cytoscape
                nt_color_rgba = get_nt_color(nt, opacity=1.0)
                # Extract hex color from rgba
                if nt_color_rgba.startswith('rgba'):
                    # Convert rgba to hex for Cytoscape (module-level re)
                    match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+)', nt_color_rgba)
                    if match:
                        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        nt_hex = f'#{r:02x}{g:02x}{b:02x}'
                    else:
                        nt_hex = '#888888'
                else:
                    nt_hex = nt_color_rgba
                
                nt_style_parts.append(f"""{{
                    selector: 'edge[nt_type = "{nt}"]',
                    style: {{
                        'line-color': '{nt_hex}',
                        'target-arrow-color': '{nt_hex}'
                    }}
                }},""")
                # Add option for dropdown
                nt_option_parts.append(f'<option value="nt_{nt}">{nt} Edges</option>')
            nt_edge_styles = "\n                ".join(nt_style_parts)
            nt_edge_group_options = "\n                                    ".join(nt_option_parts)
        
        # Hemisphere group options and controls
        has_hemi_controls = self.separate_hemispheres and has_hemisphere_nodes
        hemisphere_group_options = ""
        hemisphere_controls_html = ""
        hemisphere_group_html = ""
        if has_hemi_controls:
            hemisphere_group_options = """
                                <optgroup label="Hemispheres">
                                    <option value="hemi_left">Left Hemisphere (_L)</option>
                                    <option value="hemi_right">Right Hemisphere (_R)</option>
                                    <option value="hemi_unknown">Unknown Hemisphere (_U)</option>
                                </optgroup>
            """
            hemisphere_controls_html = """
                        <button class="btn" id="mirrorHemiBtn" onclick="toggleHemisphereMirror()" title="Arrange left/right hemisphere neurons as mirrored panels" style="font-size: 11px; padding: 6px; background: #64748b;">
                            🪞 Mirror Hemispheres
                        </button>
            """

            hemisphere_group_html = (
                '<div class="vp-ribbon-group">'
                + hemisphere_controls_html
                + '</div>'
            ) if hemisphere_controls_html.strip() else ''

        # Generate dataset legend HTML for cross-dataset type name display
        # Shows one-character codes and their corresponding dataset names;
        # each chip carries its group's color dot when the code is a
        # declared group (§13).  §layout unification (user 2026-09-07):
        # every chip renders ``CODE: full (count)`` — dataset chips carry
        # their node count (the dynamic group chip for the same code is
        # skipped below, killing the duplicated header row), and
        # linker-column entries carry their LINKER_COLORS swatch + node
        # count through dataset_legend_meta.
        group_color_map = {g['name']: g['color'] for g in extra_node_groups}
        legend_meta = getattr(self, 'dataset_legend_meta', None) or {}
        dataset_legend_html = ""
        if self.dataset_legend:
            legend_items = []
            for code, full_name in sorted(self.dataset_legend.items()):
                meta = legend_meta.get(code) or {}
                dot_color = meta.get('color') or group_color_map.get(code)
                dot = ""
                if dot_color:
                    dot = (
                        '<span style="display:inline-block;width:9px;'
                        'height:9px;border-radius:50%;background:'
                        f'{dot_color};margin-right:4px;"></span>')
                count = meta.get('count')
                count_html = (
                    f' <span style="font-size: 10px; color: var(--vp-text-2);">'
                    f'({count})</span>' if count is not None else '')
                legend_items.append(
                    f'<div class="legend-item" data-dataset="{code}" title="{full_name}">{dot}'
                    f'<span style="font-weight: bold; color: #666;">{code}:</span> '
                    f'<span style="font-size: 11px;">{full_name}</span>{count_html}'
                    f'</div>'
                )
            if legend_items:
                # Compact single-line legend: each chip already carries the
                # full dataset name in its hover title, so only a short hint
                # trails the chips instead of a dedicated header row.
                dataset_legend_html = (
                    ''.join(legend_items)
                    + '<span style="font-size: 10px; color: var(--vp-text-2);" '
                    'title="These short codes prefix node names to mark the '
                    'dataset each node comes from">dataset codes in node names</span>'
                )
        # Footer legends: with declared groups the structural role
        # swatches (source/intermediate/target) are REPLACED by the group
        # legend — one color chip per declared group (datasets in the
        # mapping views); roles stay on the node hovers.
        role_legend_html = ""
        group_legend_html = ""
        standard_group_options_html = (
            '<option value="source">Source Nodes</option>'
            '<option value="intermediate">Intermediate Nodes</option>'
            '<option value="target">Target Nodes</option>')
        if self.node_groups:
            # dataset groups replace the structural roles everywhere:
            # legend, quick actions, and the group dropdown
            standard_group_options_html = ""
            group_legend_html = "".join(
                f'<div class="legend-item" data-group="{g["name"]}" title="{html_escape(g["label"])}">'
                f'<div class="legend-color" style="background: {g["color"]};"></div>'
                f'<span>{html_escape(g["label"])}</span></div>'
                for g in extra_node_groups)
            # roles move to the hovers: no Source/Intermed./Target
            # quick-action buttons when dataset groups are declared
            role_quick_actions_html = ""
        else:
            role_legend_html = (
                '<div class="legend-item" data-group="source">'
                f'<div class="legend-color" style="background: {self.node_color[0]};"></div>'
                '<span>Source</span>'
                '</div>'
                '<div class="legend-item" data-group="intermediate">'
                f'<div class="legend-color" style="background: {self.node_color[1]};"></div>'
                '<span>Intermediate</span>'
                '</div>'
                '<div class="legend-item" data-group="target">'
                f'<div class="legend-color" style="background: {self.target_color};"></div>'
                '<span>Target</span>'
                '</div>')
            role_quick_actions_html = (
                '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">'
                f'<button class="btn" onclick="selectGroup(\'source\')" title="Select all source nodes" style="font-size: 10px; padding: 5px; background: {self.node_color[0]};">Source</button>'
                f'<button class="btn" onclick="selectGroup(\'intermediate\')" title="Select all intermediate nodes" style="font-size: 10px; padding: 5px; background: {self.node_color[1]};">Intermed.</button>'
                '</div>'
                '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">'
                f'<button class="btn" onclick="selectGroup(\'target\')" title="Select all target nodes" style="font-size: 10px; padding: 5px; background: {self.target_color};">Target</button>'
                '</div>')

        # Keep the browser-side NT lookup in sync with the Python palette used
        # to color the default network edges. The hover label uses this same
        # lookup, so a palette change cannot make the edge and label diverge.
        nt_colors_js = json_safe({
            str(nt).strip().lower(): color
            for nt, color in NT_COLORS.items()
        })
        
        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Neural Pathway Network - Selected Paths</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    
    <!-- Layout Extensions -->
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    
    <!-- CoSE-based layouts (need dependencies) -->
    <script src="https://unpkg.com/layout-base@1.0.2/layout-base.js"></script>
    <script src="https://unpkg.com/cose-base@1.0.3/cose-base.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/cytoscape-cose-bilkent.js"></script>
    <script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
    
    <!-- KLay layout -->
    <script src="https://unpkg.com/klayjs@0.4.1/klay.js"></script>
    <script src="https://unpkg.com/cytoscape-klay@3.1.4/cytoscape-klay.js"></script>
    
    <!-- Export Extensions -->
    <script src="https://unpkg.com/cytoscape-svg@0.4.0/cytoscape-svg.js"></script>
    <script>
        // CDN fallback guard: show a clear banner instead of a silent blank canvas
        if (typeof cytoscape === 'undefined') {{
            window.addEventListener('DOMContentLoaded', function() {{
                const host = document.getElementById('cy');
                if (host && !host.querySelector('.cdn-error')) {{
                    const div = document.createElement('div');
                    div.className = 'cdn-error';
                    div.style.cssText = 'padding:60px;text-align:center;color:#94a3b8;font-family:sans-serif;font-size:14px;';
                    div.textContent = '⚠️ Cytoscape.js failed to load (CDN unreachable). Check your internet connection and reload this page.';
                    host.appendChild(div);
                }}
            }});
        }}
    </script>
    <style>
        /* ===== Theme tokens: every surface derives from these so the
           White/Dark/Custom background modes flip the whole UI at once. */
        :root {{
            --vp-panel-bg: #ffffff;
            --vp-card-bg: #fafafa;
            --vp-input-bg: #ffffff;
            --vp-border: #ddd;
            --vp-text-1: #333;
            --vp-text-2: #666;
            --vp-hover-bg: rgba(255, 255, 255, 0.97);
            --vp-selected-bg: #f0f8ff;
            --vp-focus: #2196f3;
            --vp-accent-arrange: #1976d2;
            --vp-accent-filter: #2e7d32;
            --vp-accent-style: #7b1fa2;
            --vp-accent-share: #e65100;
            --vp-accent-edit: #ff9800;
        }}
        body.vp-dark {{
            --vp-panel-bg: #252525;
            --vp-card-bg: #2b2b2b;
            --vp-input-bg: #1f1f1f;
            --vp-border: #454545;
            --vp-text-1: #e0e0e0;
            --vp-text-2: #a0a0a0;
            --vp-hover-bg: rgba(37, 37, 37, 0.97);
            --vp-selected-bg: #1d3346;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: var(--vp-text-1);
            user-select: text;
            -webkit-user-select: text;
            -moz-user-select: text;
            -ms-user-select: text;
        }}
        select,
        input[type="text"],
        input[type="number"] {{
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            border: 1px solid var(--vp-border);
        }}
        button:focus-visible,
        input:focus-visible,
        select:focus-visible {{
            outline: 2px solid var(--vp-focus);
            outline-offset: 1px;
        }}
        #cy {{
            width: 100%;
            height: 90vh;
            background: white;
            border: 1px solid #ddd;
            position: relative;
            overflow: hidden;
        }}
        .controls {{
            padding: 6px 296px 6px 10px;
            background: var(--vp-panel-bg);
            border-bottom: 1px solid var(--vp-border);
            display: flex;
            gap: 10px;
            align-items: stretch;
            flex-wrap: wrap;
            /* Constant ribbon height across tabs: switching tabs no longer
               makes the footer and the canvas jump (PPT-style fixed ribbon). */
            min-height: 127px;
            box-sizing: border-box;
        }}
        body.palette-hidden #ribbon,
        body.palette-hidden .controls {{
            padding-right: 10px;
        }}
        /* Ribbon pages: tool cards wrap left-to-right; the Layout & Style
           page forces its style cards onto a second row via the break. */
        .vp-ribbon-page {{
            display: none;
            flex: 1 1 100%;
            flex-wrap: wrap;
            gap: 6px 12px;
            align-items: flex-start;
            align-content: flex-start;
            /* Pack groups left: space-between stretched sparse wrapped rows
               into huge ragged holes mid-row. */
            justify-content: flex-start;
        }}
        .vp-ribbon-page.open {{ display: flex; }}
        /* Zero-height full-width item: forces the next cards onto a new row. */
        .vp-ribbon-break {{
            flex-basis: 100%;
            height: 0;
        }}
        .vp-ribbon-group {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 4px;
            /* Standalone outlined card per group (PPT group boxes). */
            padding: 4px 10px;
            border: 1px solid var(--vp-border);
            border-radius: 6px;
            min-width: 0;
        }}
        .vp-mini-label {{
            font-size: 10px;
            color: var(--vp-text-2);
            white-space: nowrap;
        }}
        .vp-group-title {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            color: var(--vp-text-2);
            margin-bottom: 1px;
        }}
        /* Uniform compact control heights inside the ribbon. !important is
           required to beat the inline paddings on the generated controls. */
        #ribbon select,
        #ribbon input[type="text"] {{
            height: 26px;
            padding: 3px 8px !important;
            box-sizing: border-box;
        }}
        #ribbon input[type="number"] {{
            height: 24px;
            padding: 2px 4px !important;
            box-sizing: border-box;
        }}
        #ribbon .btn {{
            min-height: 26px;
            padding-top: 4px !important;
            padding-bottom: 4px !important;
            white-space: nowrap;
        }}
        /* Ribbon tabs (PPT-style) on the command strip. */
        .vp-tabs {{
            display: flex;
            align-items: stretch;
            gap: 2px;
            margin-left: 8px;
        }}
        .vp-tab {{
            background: var(--vp-card-bg);
            border: 1px solid var(--vp-border);
            border-radius: 14px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.3;
            color: var(--vp-text-2);
            cursor: pointer;
            white-space: nowrap;
        }}
        .vp-tab:hover {{
            color: var(--vp-text-1);
            background: var(--vp-input-bg);
        }}
        .vp-tab.active {{
            color: var(--vp-tab-accent, var(--vp-focus));
            border-color: var(--vp-tab-accent, var(--vp-focus));
        }}
        .btn {{
            padding: 5px 10px;
            min-height: 28px;
            box-sizing: border-box;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.3s;
            width: 100%;
            text-align: center;
        }}
        .btn:hover {{
            background: #45a049;
        }}
        .btn.secondary {{
            background: #2196F3;
        }}
        .btn.secondary:hover {{
            background: #0b7dda;
        }}
        /* Spinner row: label (icon + text, never wraps) + number input with
           native up/down arrows + unit. The value applies LIVE on every
           arrow step ('input' event); no upper cap is imposed. */
        /* Spinner mini: label stacked above the input so several spinners
           share one ribbon row (.vp-spinner-row). */
        .vp-spinner {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 2px 4px;
            /* Cap the spinner: a wrap container's max-content sums label +
               input + unit on one line, which otherwise inflates every
               spinner to ~140px and forces the ribbon groups to wrap. */
            max-width: 88px;
        }}
        .vp-spinner label {{
            flex: 1 1 100%;
            min-width: 0;
            font-size: 11px;
            font-weight: 600;
            color: var(--vp-text-2);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .vp-spinner input[type="number"] {{
            flex: 1 1 auto;
            width: 56px;
            min-width: 0;
            padding: 3px 4px;
            font-size: 12px;
            border-radius: 4px;
            text-align: right;
        }}
        /* Inline variant: label left of the spinner (Sizes group) keeps the
           control row a single 26px line so the merged Layout & Style page
           stays close to the shared ribbon height. */
        .vp-spinner-inline {{ max-width: none; }}
        .vp-spinner-inline label {{ flex: 0 0 auto; }}
        .vp-spinner-row {{
            display: flex;
            align-items: flex-end;
            flex-wrap: wrap;
            gap: 6px 10px;
        }}
        .vp-unit {{
            flex: 0 0 auto;
            font-size: 11px;
            color: var(--vp-text-2);
        }}
        /* Legend + canvas info footer: separated from the tool ribbon. */
        .vp-footer {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 4px 22px;
            padding: 6px 296px 6px 12px;
            background: var(--vp-panel-bg);
            border-bottom: 1px solid var(--vp-border);
            font-size: 11px;
            line-height: 18px;
            color: var(--vp-text-2);
            box-sizing: border-box;
        }}
        body.palette-hidden .vp-footer {{
            padding-right: 12px;
        }}
        .vp-footer .legend {{ margin: 0; gap: 4px 14px; }}
        .vp-footer .info {{ font-size: 11px; }}
        .info {{
            color: var(--vp-text-2);
            font-size: 14px;
            text-align: left;
        }}
        .legend {{
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 4px 16px;
            align-items: center;
            font-size: 13px;
            margin: 10px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .hidden {{
            display: none;
        }}
        
        /* Hover Info Box - Fixed at bottom-left; element tooltips ONLY
           (operation feedback uses the toast stack). */
        #hoverInfo {{
            position: fixed;
            bottom: 15px;
            left: 15px;
            z-index: 10000;
            background: var(--vp-hover-bg);
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-size: 13px;
            color: var(--vp-text-1);
            line-height: 1.6;
            max-width: 400px;
            border: 1px solid var(--vp-border);
        }}
        #hoverInfo b {{
            color: var(--vp-text-1);
            font-weight: 600;
        }}
        
        /* Main layout: network + right-side menubar */
        .main {{
            display: grid;
            grid-template-columns: 1fr 280px;
            gap: 0;
            align-items: stretch;  /* Stretch to full height */
            height: calc(100vh - 0px);  /* Full viewport height minus controls */
            margin: 0;
            width: 100%;
            box-sizing: border-box;
        }}

        .color-palette {{
            position: fixed;  /* Fixed positioning to overlay everything */
            top: 0;  /* Start from very top of page */
            right: 0;  /* Anchor to right side */
            background: var(--vp-panel-bg);
            border-left: 2px solid var(--vp-border);
            padding: 15px;
            box-shadow: -2px 0 8px rgba(0,0,0,0.1);  /* Add shadow for depth */
            z-index: 1000;  /* High z-index to overlay controls */
            height: 100vh;  /* Full viewport height */
            overflow-y: auto;
            width: 280px;
            box-sizing: border-box; /* include border/padding in width calculations */
        }}
        .color-palette h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: var(--vp-text-1);
        }}
        .palette-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--vp-border);
        }}
        .palette-section:last-child {{
            border-bottom: none;
        }}
        .palette-section h4 {{
            margin: 0 0 10px 0;
            font-size: 12px;
            color: var(--vp-text-2);
            text-transform: uppercase;
            font-weight: 600;
        }}
        /* Accordion sections (right panel): header button toggles body. */
        .vp-acc {{
            padding-bottom: 10px;
        }}
        .vp-acc-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            padding: 2px 0 8px 0;
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: var(--vp-text-1);
            text-align: left;
        }}
        .vp-acc-header .vp-acc-chevron {{
            transition: transform 0.15s ease;
            font-size: 12px;
            color: var(--vp-text-2);
        }}
        .vp-acc.open .vp-acc-header .vp-acc-chevron {{
            transform: rotate(180deg);
        }}
        .vp-acc-body {{ display: none; }}
        .vp-acc.open .vp-acc-body {{ display: block; }}
        .vp-acc .palette-section {{
            border-bottom: none;
            margin-bottom: 12px;
            padding-bottom: 8px;
        }}
        /* Selection summary chip (live state, replaces static hint text). */
        .vp-chip {{
            display: inline-block;
            background: var(--vp-selected-bg);
            border: 1px solid var(--vp-border);
            border-radius: 10px;
            padding: 3px 10px;
            font-size: 11px;
            color: var(--vp-text-1);
            margin-bottom: 8px;
        }}
        .color-group {{
            margin-bottom: 12px;
        }}
        .color-group label {{
            display: block;
            font-size: 11px;
            color: var(--vp-text-2);
            margin-bottom: 5px;
            font-weight: 500;
        }}
        .color-input-group {{
            display: flex;
            gap: 5px;
            align-items: center;
        }}
        .color-input-group input[type="color"] {{
            width: 40px;
            height: 30px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            cursor: pointer;
            background: var(--vp-input-bg);
        }}
        .color-input-group input[type="text"] {{
            width: 70px;
            padding: 5px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            font-size: 11px;
        }}
        .color-input-group input[type="range"] {{
            width: 80px;
        }}
        .color-input-group select {{
            width: 130px;
            padding: 5px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            font-size: 11px;
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            cursor: pointer;
        }}
        .alpha-value {{
            font-size: 11px;
            color: var(--vp-text-2);
            min-width: 35px;
        }}
        .font-select {{
            width: 100%;
            padding: 5px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            font-size: 11px;
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
        }}
        .apply-btn {{
            width: 100%;
            padding: 8px;
            margin-top: 10px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }}
        .apply-btn:hover {{
            background: #45a049;
        }}
        .selected-info {{
            background: var(--vp-selected-bg);
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 11px;
            color: var(--vp-text-1);
        }}
        .selected-info strong {{
            color: #2196F3;
        }}
        .clear-selection-btn {{
            width: 100%;
            padding: 6px;
            margin-top: 5px;
            background: #ff9800;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }}
        .clear-selection-btn:hover {{
            background: #f57c00;
        }}

        /* Panel bar: slim always-visible strip above the controls, used to
           collapse/show the two tool panels (top controls + right palette).
           Doubles as the command strip: node search + help live here. */
        #panelBar {{
            display: flex;
            align-items: center;
            gap: 8px;
            /* Right padding clears the fixed 280px palette so the search
               box and help button are never covered by it. */
            padding: 4px 296px 4px 10px;
            background: var(--vp-panel-bg);
            border-bottom: 1px solid var(--vp-border);
            min-height: 34px;
            box-sizing: border-box;
        }}
        body.palette-hidden #panelBar {{
            padding-right: 10px;
        }}
        .panelbar-btn {{
            padding: 4px 10px;
            font-size: 12px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            cursor: pointer;
            transition: background 0.2s;
        }}
        .panelbar-btn:hover {{
            background: var(--vp-card-bg);
        }}
        .vp-flex-spacer {{
            flex: 1;
        }}
        /* Node search (command strip). */
        .vp-search {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        #nodeSearchInput {{
            width: 150px;
            padding: 4px 8px;
            font-size: 12px;
            border-radius: 4px;
        }}
        #nodeSearchInput:focus {{
            width: 200px;
        }}
        #nodeSearchCount {{
            font-size: 11px;
            color: var(--vp-text-2);
            min-width: 34px;
            text-align: right;
        }}
        /* Compact ▲/▼ steppers beside the search box for cycling matches. */
        .vp-search-step {{
            padding: 2px 7px;
            font-size: 10px;
            line-height: 1.2;
            background: var(--vp-card-bg);
            color: var(--vp-text-2);
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            cursor: pointer;
        }}
        .vp-search-step:hover {{
            color: var(--vp-text-1);
            background: var(--vp-hover-bg);
        }}
        /* Toast stack: operation feedback (success/info/warn/error),
           bottom-left above the hover info box. aria-live=polite. */
        #toastStack {{
            position: fixed;
            bottom: 100px;
            left: 15px;
            z-index: 10001;
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-width: 420px;
            pointer-events: none;
        }}
        .vp-toast {{
            background: var(--vp-hover-bg);
            color: var(--vp-text-1);
            border: 1px solid var(--vp-border);
            border-left-width: 4px;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12.5px;
            line-height: 1.45;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            pointer-events: auto;
        }}
        .vp-toast.vp-toast-success {{ border-left-color: #2e7d32; }}
        .vp-toast.vp-toast-info    {{ border-left-color: #2196f3; }}
        .vp-toast.vp-toast-warn    {{ border-left-color: #e65100; }}
        .vp-toast.vp-toast-error   {{ border-left-color: #c62828; }}
        .vp-toast .vp-toast-action {{
            margin-left: 8px;
            padding: 2px 8px;
            font-size: 11px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            cursor: pointer;
        }}

        /* In-page dialog (replaces native prompt/confirm). */
        #vpDialogOverlay {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.45);
            z-index: 20000;
            display: none;
            align-items: center;
            justify-content: center;
        }}
        #vpDialogOverlay.open {{ display: flex; }}
        .vp-dialog {{
            background: var(--vp-panel-bg);
            color: var(--vp-text-1);
            border: 1px solid var(--vp-border);
            border-radius: 8px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            min-width: 320px;
            max-width: 460px;
            padding: 14px 16px;
        }}
        .vp-dialog h3 {{ margin: 0 0 10px 0; font-size: 15px; }}
        .vp-dialog .vp-dialog-message {{
            font-size: 13px;
            color: var(--vp-text-2);
            margin: 0 0 10px 0;
            line-height: 1.5;
        }}
        .vp-dialog .vp-dialog-field {{ margin-bottom: 10px; }}
        .vp-dialog .vp-dialog-field label {{
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--vp-text-2);
            margin-bottom: 4px;
        }}
        .vp-dialog .vp-dialog-field input,
        .vp-dialog .vp-dialog-field select {{
            width: 100%;
            padding: 6px 8px;
            font-size: 13px;
            border-radius: 4px;
            box-sizing: border-box;
        }}
        .vp-dialog .vp-dialog-actions {{
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-top: 12px;
        }}
        .vp-dialog .vp-dialog-btn {{
            padding: 6px 14px;
            font-size: 12px;
            border-radius: 4px;
            border: 1px solid var(--vp-border);
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            cursor: pointer;
        }}
        .vp-dialog .vp-dialog-btn.vp-primary {{
            background: #2196f3;
            border-color: #2196f3;
            color: white;
        }}
        .vp-dialog .vp-dialog-btn.vp-danger {{
            background: #c62828;
            border-color: #c62828;
            color: white;
        }}

        /* Help overlay (? key / ❓ button). */
        #helpOverlay {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.45);
            z-index: 19000;
            display: none;
            align-items: center;
            justify-content: center;
        }}
        #helpOverlay.open {{ display: flex; }}
        .vp-help-card {{
            background: var(--vp-panel-bg);
            color: var(--vp-text-1);
            border: 1px solid var(--vp-border);
            border-radius: 8px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            width: min(880px, 92vw);
            max-height: 86vh;
            overflow-y: auto;
            padding: 16px 18px;
            position: relative;
        }}
        .vp-help-card h3 {{ margin: 0 0 12px 0; font-size: 16px; }}
        .vp-help-card #helpCloseBtn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 2px 10px;
            font-size: 13px;
            border: 1px solid var(--vp-border);
            border-radius: 4px;
            background: var(--vp-input-bg);
            color: var(--vp-text-1);
            cursor: pointer;
        }}
        .vp-help-cols {{
            display: grid;
            grid-template-columns: 1fr 1fr 1.2fr;
            gap: 16px;
            font-size: 12.5px;
            line-height: 1.55;
        }}
        .vp-help-cols h4 {{
            margin: 0 0 6px 0;
            font-size: 13px;
            color: var(--vp-accent-arrange);
        }}
        .vp-help-cols table {{ border-collapse: collapse; }}
        .vp-help-cols td {{
            padding: 2px 8px 2px 0;
            vertical-align: top;
            color: var(--vp-text-1);
        }}
        .vp-help-cols td.vp-key {{
            white-space: nowrap;
            font-weight: 600;
            color: var(--vp-text-2);
        }}
        .vp-help-cols code {{
            background: var(--vp-card-bg);
            border: 1px solid var(--vp-border);
            border-radius: 3px;
            padding: 0 4px;
        }}

        /* Edit-mode canvas affordance. */
        #cy.vp-editmode {{
            border: 2px dashed var(--vp-accent-edit);
        }}
        #editBadge {{
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 50;
            background: var(--vp-accent-edit);
            color: #fff;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 10px;
            display: none;
            pointer-events: none;
        }}
        /* Canvas drag-mode switch at the canvas top-right: Pan (default —
           drag pans, Shift+drag box-selects) vs Select (plain drag
           box-selects — the switch turns user panning off, which cytoscape
           maps to box-select-on-drag). DOM overlay, never in exports. */
        #cyHints {{
            position: absolute;
            top: 8px;
            right: 8px;
            z-index: 50;
            display: flex;
            gap: 6px;
        }}
        .vp-cy-hint {{
            background: var(--vp-hover-bg);
            border: 1px solid var(--vp-border);
            color: var(--vp-text-2);
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 10px;
            white-space: nowrap;
            cursor: pointer;
            font-family: inherit;
        }}
        .vp-cy-hint:hover {{
            color: var(--vp-text-1);
        }}
        .vp-cy-hint.active {{
            color: var(--vp-accent-arrange);
            border-color: var(--vp-accent-arrange);
        }}
        /* Directional box-selection frame drawn over the canvas: SOLID border
           for a left-to-right (FULL containment) frame, DASHED for a
           right-to-left (TOUCH) frame. pointer-events:none keeps the gesture
           on the canvas underneath. */
        #boxOverlay {{
            position: absolute;
            display: none;
            z-index: 40;
            border: 1px solid var(--vp-accent-arrange);
            background: rgba(25, 118, 210, 0.08);
            pointer-events: none;
        }}
        #boxOverlay.vp-touch {{
            border: 2px dashed var(--vp-accent-arrange);
        }}

        /* Collapsed states: hide the top control bar and reclaim the right
           palette column so the canvas grows into the freed space. */
        .controls.collapsed {{
            display: none;
        }}
        .main.palette-hidden {{
            grid-template-columns: 1fr 0;
        }}
        body.controls-collapsed #cy {{
            height: calc(100vh - 76px);
        }}
    </style>
</head>
<body>
    <!-- Panel bar / command strip: panel toggles + ribbon tabs + search + help -->
    <div id="panelBar">
        <div class="vp-tabs" role="tablist" aria-label="Tool tabs">
            <button type="button" id="tabLayout" class="vp-tab active" style="--vp-tab-accent: var(--vp-accent-arrange);" onclick="switchTab('layout')" title="Layout & style tools: algorithm, node gaps, rotation, sizes, width scale, background (click again to collapse the ribbon)">🔧 Layout &amp; Style</button>
            <button type="button" id="tabFilter" class="vp-tab" style="--vp-tab-accent: var(--vp-accent-filter);" onclick="switchTab('filter')" title="Filter tools: connection metric, hide edges/labels/orphans (click again to collapse the ribbon)">👁️ Filter</button>
            <button type="button" id="tabShare" class="vp-tab" style="--vp-tab-accent: var(--vp-accent-share);" onclick="switchTab('share')" title="Import & export: images, graph, layouts, edge list (click again to collapse the ribbon)">💾 Import & Export</button>
        </div>
        <span class="vp-flex-spacer"></span>
        <button id="refreshLayoutTopBtn" class="panelbar-btn" onclick="refreshLayout()" title="Re-run the current layout algorithm — available from every tool tab">🔄 Refresh Layout</button>
        <div class="vp-search">
            <input type="text" id="nodeSearchInput" placeholder="🔍 Find node…" autocomplete="off" oninput="onSearchInput(this.value)" onkeydown="onSearchKeydown(event)" title="Search nodes by id or label — Enter selects and centers, ↑/↓ or the ▲/▼ buttons step through matches, Esc clears" aria-label="Find node">
            <button type="button" id="searchPrevBtn" class="vp-search-step" onclick="cycleSearchMatch(-1)" title="Previous match (↑)">▲</button>
            <button type="button" id="searchNextBtn" class="vp-search-step" onclick="cycleSearchMatch(1)" title="Next match (↓)">▼</button>
            <span id="nodeSearchCount" aria-live="polite"></span>
        </div>
        <button id="toggleControlsBtn" class="panelbar-btn" onclick="toggleTopControls()" title="Collapse or show the ribbon (all tool tabs)">⚙️ Hide Ribbon</button>
        <button id="togglePanelBtn" class="panelbar-btn" onclick="toggleRightPanel()" title="Collapse or show the right side panel">🎨 Hide Panel</button>
        <button id="helpBtn" class="panelbar-btn" onclick="toggleHelp()" title="Help: mouse gestures, keyboard shortcuts, recipes (?)">❓</button>
    </div>
    <!-- Ribbon: tool card rows per active tab (Layout & Style packs two rows) -->
    <div class="controls" id="ribbon">
        <div class="vp-ribbon-page open" id="pageLayout">
            <div class="vp-ribbon-group" style="min-width: 250px;">
            <label class="vp-group-title" for="layoutSelector">Layout Algorithm</label>
            <select id="layoutSelector" onchange="changeLayout()" title="Layout algorithm used to arrange the nodes" style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ddd; font-size: 12px; background: white; cursor: pointer;">
                {mapping_layout_optgroup}
                <optgroup label="🌟 Hierarchical">
                    <option value="dagre" {_layout_selected('dagre')}>Dagre ⭐⭐⭐⭐⭐</option>
                    <option value="klay" {_layout_selected('klay')}>KLay ⭐⭐⭐⭐</option>
                    <option value="breadthfirst" {_layout_selected('breadthfirst')}>Breadth-First ⭐⭐⭐</option>
                </optgroup>
                <optgroup label="🎯 Force-Directed">
                    <option value="fcose" {_layout_selected('fcose')}>fCoSE ⭐⭐⭐⭐⭐</option>
                    <option value="cose-bilkent" {_layout_selected('cose-bilkent')}>CoSE-Bilkent ⭐⭐⭐⭐</option>
                    <option value="cose" {_layout_selected('cose')}>CoSE ⭐⭐⭐</option>
                </optgroup>
                <optgroup label="🧠 Hemisphere-Aware">
                    <option value="hemi-dagre" {_layout_selected('hemi-dagre')}>Hemisphere Dagre 🪞</option>
                    <option value="hemi-fcose" {_layout_selected('hemi-fcose')}>Hemisphere fCoSE 🪞</option>
                </optgroup>
                <optgroup label="📐 Other">
                    <option value="circle" {_layout_selected('circle')}>Circular ⭐⭐</option>
                    <option value="grid" {_layout_selected('grid')}>Grid ⭐⭐</option>
                    <option value="concentric" {_layout_selected('concentric')}>Concentric ⭐⭐</option>
                </optgroup>
            </select>
            </div>
            <div class="vp-ribbon-group" style="min-width: 290px;">
            <label class="vp-group-title">Inter-node Gap</label>
            <div class="vp-spinner-row">
                <div class="vp-spinner vp-spinner-inline">
                    <label for="nodeGapHSlider" title="Horizontal center-to-center distance between neighboring columns">Horizontal ↔</label>
                    <input type="number" id="nodeGapHSlider" min="0" step="1" value="100" oninput="onSpacingInput('x', this.value)" onchange="onSpacingChange()" title="Horizontal center-to-center distance between neighboring nodes in px (node size unchanged)">
                    <span class="vp-unit">px</span>
                </div>
                <div class="vp-spinner vp-spinner-inline">
                    <label for="nodeGapVSlider" title="Vertical center-to-center distance between neighboring rows">Vertical ↕</label>
                    <input type="number" id="nodeGapVSlider" min="0" step="1" value="100" oninput="onSpacingInput('y', this.value)" onchange="onSpacingChange()" title="Vertical center-to-center distance between neighboring nodes in px (node size unchanged)">
                    <span class="vp-unit">px</span>
                </div>
                <button class="btn" onclick="resetSpacing()" style="align-self: flex-end; background: #607d8b; font-size: 11px; padding: 5px 10px; width: auto; min-width: 0;" title="Restore the inter-node gaps captured at the last layout run">↺ Reset Spacing</button>
            </div>
            </div>
            <div class="vp-ribbon-group" style="min-width: 160px;">
            <label class="vp-group-title">Rotate</label>
            <div class="vp-spinner-row">
                <div class="vp-spinner vp-spinner-inline">
                    <label for="rotateSlider" title="Rotation angle of the whole layout">Rotate ↻</label>
                    <input type="number" id="rotateSlider" step="5" value="0" oninput="onRotationInput(this.value)" onchange="onRotationChange()" title="Rotate the current layout around its centroid in degrees (labels stay horizontal; distances are conserved)">
                    <span class="vp-unit">°</span>
                </div>
                <button class="btn" onclick="rotateCounterClockwise()" style="align-self: flex-end; font-size: 13px; padding: 4px 9px; width: auto; min-width: 0; line-height: 1.2;" title="Rotate the layout 90° counter-clockwise per click (the angle display wraps mod 360; any angle can also be typed in the Rotate field)">↺</button>
            </div>
            </div>
            <div class="vp-ribbon-group" style="min-width: 170px;">
            <label class="vp-group-title">Labels</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <button class="btn secondary" id="toggleLabelsBtn" onclick="toggleLabels()" style="font-size: 12px; padding: 6px; width: 100%;" title="Show or hide node labels (shortcut: L)">🏷️ Hide Labels</button>
                <button class="btn" id="showAllBtn" onclick="showAllNodes()" style="background: #ff9800; font-size: 12px; padding: 6px; width: 100%; display: none;" title="Restore all manually hidden nodes">👁️ Show All</button>
                <button class="btn secondary" id="toggleEdgeWeightsBtn" onclick="toggleEdgeWeightLabels()" data-showing="0" style="font-size: 12px; padding: 6px; width: 100%;" title="Show or hide the value painted on each edge (follows the active Connection Metric)">🏋️ Edge Weights</button>
            </div>
            </div>
            <div class="vp-ribbon-group" style="min-width: 120px;">
            <label class="vp-group-title">Canvas</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <button class="btn" onclick="fitGraph()" style="background: #2196f3; font-size: 12px; padding: 6px; width: 100%;" title="Zoom and pan so all visible nodes fit the canvas">⛶ Fit</button>
            </div>
            </div>
            {hemisphere_group_html}
            <div class="vp-ribbon-break" aria-hidden="true"></div>
            <div class="vp-ribbon-group" style="min-width: 170px;">
            <label class="vp-group-title" for="edgeWidthScale">Edge Width Scale</label>
            <select id="edgeWidthScale" onchange="updateEdgeWidths()" title="How edge weights map to widths (linear, logarithmic, square root, or constant)" style="width: 100%; padding: 5px;">
                <option value="linear" {'selected' if self.edge_width_scale == 'linear' else ''}>Linear</option>
                <option value="log_e" {'selected' if self.edge_width_scale == 'log' and str(self.edge_width_log_base) not in ('2', '10') else ''}>Logarithmic (ln)</option>
                <option value="log_2" {'selected' if self.edge_width_scale == 'log' and str(self.edge_width_log_base) == '2' else ''}>Logarithmic (log₂)</option>
                <option value="log_10" {'selected' if self.edge_width_scale == 'log' and str(self.edge_width_log_base) == '10' else ''}>Logarithmic (log₁₀)</option>
                <option value="sqrt" {'selected' if self.edge_width_scale == 'sqrt' else ''}>Square Root</option>
                <option value="none" {'selected' if self.edge_width_scale == 'none' else ''}>None (Constant)</option>
            </select>
            </div>
            <div class="vp-ribbon-group">
            <label class="vp-group-title">Sizes</label>
            <div class="vp-spinner-row">
                <div class="vp-spinner vp-spinner-inline">
                    <label for="fontSizeSlider" title="Text size of the node labels">Font Size</label>
                    <input type="number" id="fontSizeSlider" min="1" step="1" value="12" oninput="updateFontSize(this.value)" title="Text size of the node labels in px">
                    <span class="vp-unit">px</span>
                </div>
                <div class="vp-spinner vp-spinner-inline">
                    <label for="edgeLabelSizeSlider" title="Text size of the on-edge weight labels (the 🏋️ Edge Weights toggle)">Edge Label Size</label>
                    <input type="number" id="edgeLabelSizeSlider" min="1" step="1" value="{self.edge_label_font_size}" oninput="updateEdgeLabelFontSize(this.value)" title="Text size of the weight labels painted on the edges, in px">
                    <span class="vp-unit">px</span>
                </div>
                <div class="vp-spinner vp-spinner-inline">
                    <label for="nodeSizeSlider" title="Diameter of the nodes">Node Size</label>
                    <input type="number" id="nodeSizeSlider" min="1" step="1" value="40" oninput="updateNodeSize(this.value)" title="Diameter of the nodes in px">
                    <span class="vp-unit">px</span>
                </div>
                <div class="vp-spinner vp-spinner-inline">
                    <label for="edgeWidthSlider" title="Base thickness of the edges">Edge Width</label>
                    <input type="number" id="edgeWidthSlider" min="0.5" step="0.5" value="3" oninput="updateEdgeWidth(this.value)" title="Base thickness of the edges in px">
                    <span class="vp-unit">px</span>
                </div>
                <div class="vp-spinner vp-spinner-inline">
                    <label for="arrowSizeSlider" title="Size of the arrowheads on directed edges">Arrow Size</label>
                    <input type="number" id="arrowSizeSlider" min="1" step="1" value="9" oninput="updateArrowSize(this.value)" title="Size of the arrowheads on directed edges in px">
                    <span class="vp-unit">px</span>
                </div>
            </div>
            </div>
            <div id="reciprocalOffsetControls" class="vp-ribbon-group" style="display: none;">
            <label class="vp-group-title" for="reciprocalOffsetSlider">Reciprocal Offset</label>
            <div class="vp-spinner-row">
                <button id="reciprocalModeToggle" onclick="toggleReciprocalMode()" title="Draw reciprocal edge pairs curved (offset) or straight on top of each other" style="padding: 4px 8px; font-size: 11px; border-radius: 4px; border: 1px solid var(--vp-border); background: #4caf50; color: white; cursor: pointer; flex: 0 0 auto;">Straight</button>
                <input type="number" id="reciprocalOffsetSlider" min="0" step="1" value="5" title="Separation in pixels between the opposite edges of a reciprocal pair (live; bound by initializeReciprocalOffsetControls)">
                <span class="vp-unit">px</span>
            </div>
            </div>
            <div class="vp-ribbon-group" style="width: 258px;">
            <label class="vp-group-title">Background & Font color</label>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="vp-mini-label">Background</span>
                <button id="bgToggleBtn" class="btn" onclick="toggleBackground()" style="flex: 0 0 auto; width: auto; padding: 4px 8px; font-size: 11px; background: #ffffff; color: #1f2937; border: 1px solid #9ca3af;" title="Cycle the canvas background between the white and dark presets (the button previews the preset; exports match the visible background)">White</button>
                <input type="color" id="customBgColor" value="#f5f5f5" style="width: 35px; height: 28px; border: 1px solid var(--vp-border); border-radius: 3px; cursor: pointer;" title="Custom background color — applies immediately (exports match the visible background)" onchange="applyCustomBackground()">
                <span class="vp-mini-label" style="margin-left: 2px;">Font</span>
                <input type="color" id="labelFontColor" value="#000000" style="width: 35px; height: 28px; border: 1px solid var(--vp-border); border-radius: 3px; cursor: pointer;" title="Node label text color (node labels never have a background)" onchange="applyLabelFontColor(this.value)">
            </div>
            </div>
        </div>

        <div class="vp-ribbon-page" id="pageFilter">
            <div class="vp-ribbon-group" style="min-width: 150px;">
            <label class="vp-group-title">Connection Metric</label>
            <select id="metricSelect" onchange="updateMetric()" title="Value used by the edge filter below AND by the edge widths: {self.metric_option_label.lower()}, connection ratio, or traversal probability" style="width: 100%; padding: 5px;">
                <option value="weight">{self.metric_option_label}</option>
                <option value="ratio">Connection Ratio</option>
                <option value="probability">Traversal Probability</option>
            </select>
            </div>
            <div class="vp-ribbon-group" style="min-width: 215px;">
            <label class="vp-group-title" for="ignoreEdgesInput">Hide Edges (by metric)</label>
            <input type="text" id="ignoreEdgesInput" placeholder="OR: <5, >100 | AND: (>=5, <=10)" style="width: 100%; padding: 5px; font-size: 11px; border-radius: 3px; box-sizing: border-box;" oninput="updateIgnoredEdges()" title="Hide edges whose ACTIVE metric value matches. Comma = OR, parentheses = AND, e.g. under 5, over 100, (at least 10 and at most 20)">
            <div style="font-size: 9px; color: var(--vp-text-2); line-height: 1.2;">
                Comma = OR, Parentheses = AND · <a href="#" onclick="openHelp('recipes'); return false;" title="Open the filter-syntax recipes in the help overlay" style="color: var(--vp-accent-filter);">…more</a>
            </div>
            </div>
            <div class="vp-ribbon-group" style="min-width: 230px;">
            <label class="vp-group-title">Auto-hide</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
                <button class="btn" id="hideOrphansBtn" onclick="toggleOrphanNodes()" style="font-size: 10px; padding: 6px; background: #9c27b0;" title="Show or hide nodes with no connections">👻 Hide Orphans</button>
                <button class="btn" id="hideSelfLoopsBtn" onclick="toggleSelfLoops()" style="font-size: 10px; padding: 6px; background: #ff5722;" title="Show or hide edges from a node to itself">🔁 Hide Self-Loops</button>
                <button class="btn" id="hideDeadEndsBtn" onclick="toggleDeadEnds()" style="font-size: 10px; padding: 6px; background: #607d8b;" title="Show or hide out-only non-source / in-only non-target nodes">💀 Hide Dead Ends</button>
            </div>
            <div style="font-size: 10px; color: var(--vp-text-2); line-height: 1.3;">
                • Orphans: nodes with no connections<br>
                • Self-Loops: edges from a node to itself<br>
                • Dead Ends: out-only non-source / in-only non-target nodes
            </div>
            </div>
        </div>

        <div class="vp-ribbon-page" id="pageShare">
            <div class="vp-ribbon-group">
            <label class="vp-group-title">Image</label>
            <div style="display: flex; gap: 6px; align-items: center;">
                <label for="exportScale" style="font-weight: bold; font-size: 13px; white-space: nowrap;" title="Image export magnification">Scale:</label>
                <input type="number" id="exportScale" min="1" max="10" value="2" step="0.5" style="width: 56px; padding: 5px; box-sizing: border-box;" title="Image export magnification (1–10; very large values may fail)">
                <button class="btn" onclick="exportPNG()" style="flex: 1; padding: 6px; font-size: 12px;" title="Export the current view as a PNG image at the chosen scale">PNG</button>
                <button class="btn" onclick="exportSVG()" style="flex: 1; padding: 6px; font-size: 12px;" title="Export the current view as a vector SVG image">SVG</button>
            </div>
            </div>
            <div class="vp-ribbon-group">
            <label class="vp-group-title">Graph</label>
            <div style="display: flex; gap: 6px;">
                <button class="btn" onclick="exportGraph()" style="flex: 1; padding: 6px; font-size: 11px; background: #9c27b0;" title="Download the full graph (nodes, edges, styles, positions) as JSON">📤 Export Graph</button>
                <button class="btn" onclick="importGraph()" style="flex: 1; padding: 6px; font-size: 11px; background: #9c27b0;" title="Replace the canvas with a previously exported graph JSON">📥 Import Graph</button>
            </div>
            </div>
            <div class="vp-ribbon-group">
            <label class="vp-group-title">Layout Persistence</label>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <button class="btn" onclick="saveLayout()" style="background: #4caf50; font-size: 12px; padding: 6px; width: 100%;" title="Save current node positions to this browser's storage for this file">💾 Save</button>
                <button class="btn" onclick="loadLayout()" style="background: #2196f3; font-size: 12px; padding: 6px; width: 100%;" title="Load node positions saved earlier from browser storage">📂 Load</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <button class="btn" onclick="exportLayout()" style="padding: 6px; font-size: 11px; background: #607d8b; width: 100%;" title="Download current node positions as a JSON layout file">📤 Export Layout</button>
                <button class="btn" onclick="importLayout()" style="padding: 6px; font-size: 11px; background: #607d8b; width: 100%;" title="Apply node positions from a previously exported JSON layout file">📥 Import Layout</button>
            </div>
            </div>
            <div class="vp-ribbon-group" style="min-width: 220px;">
            <label class="vp-group-title">Data</label>
            <div style="display: flex; gap: 6px;">
                <button class="btn" onclick="exportEdgeListCSV()" title="Export every edge as CSV (source, target, weight, color, NT, grouping)" style="flex: 1; padding: 6px; font-size: 11px; background: #2e7d32;">📋 Edge List CSV</button>
            </div>
            <input type="file" id="graphFileInput" accept=".json" style="display: none;" onchange="loadGraphFile(event)" title="Hidden file picker used by Import Graph">
            <input type="file" id="layoutFileInput" accept=".json" style="display: none;" onchange="loadLayoutFile(event)" title="Hidden file picker used by Import Layout">
            </div>
        </div>
    </div>
    <!-- Legend + canvas info footer: separated from the tool ribbon; stays
         visible even when the ribbon is collapsed -->
    <div class="vp-footer">
        <div class="legend">
            <!-- Group chips are rebuilt live by refreshLegend(); the dataset
                 chips are static and never touched by it. -->
            <span id="groupLegend" style="display: contents;">{role_legend_html}{group_legend_html}</span>
            <span id="datasetLegend" style="display: contents;">{dataset_legend_html}</span>
        </div>
        <div class="info">
            <strong>{G.number_of_nodes()}</strong> nodes, <strong>{G.number_of_edges()}</strong> connections ·
            Press <strong>?</strong> for help
        </div>
    </div>
    
    <!-- Color Palette Panel -->
    <div class="main">
        <div id="cy"><div id="editBadge">✏️ EDIT MODE</div><div id="boxOverlay"></div><div id="cyHints"><button type="button" id="panModeBtn" class="vp-cy-hint active" aria-pressed="true" onclick="setCanvasDragMode('pan')" title="Pan mode (default): drag the blank canvas to pan — Shift+drag still box-selects">✋ Pan</button><button type="button" id="selectModeBtn" class="vp-cy-hint" aria-pressed="false" onclick="setCanvasDragMode('select')" title="Select mode: drag the blank canvas to box-select without Shift — left-to-right frame = full selection, right-to-left dashed frame = touch; dragging over already-selected objects deselects them (⇧+drag toggles)">□ Select</button></div></div>
        <div class="color-palette" id="colorPalette">
            <div class="palette-content">
                <!-- Edit accordion (collapsed by default) -->
                <div class="palette-section vp-acc" id="accEdit">
                    <button class="vp-acc-header" onclick="toggleAccordion('edit')" title="Show or hide the edit-mode tools">
                        <span>✏️ Edit</span><span class="vp-acc-chevron">▾</span>
                    </button>
                    <div class="vp-acc-body" id="accEditBody">
                    <button class="btn" id="editModeBtn" onclick="toggleEditMode()" title="Toggle manual editing: add/delete nodes and drag to connect" style="width: 100%; margin-bottom: 10px; background: #ff9800;">
                        ✏️ Enable Edit Mode
                    </button>
                    
                    <div id="editControls" style="display: none; margin-bottom: 10px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px;">
                            <button class="btn" onclick="addNode()" title="Add a new node at the center of the view (edit mode)" style="font-size: 11px; padding: 6px; background: #4caf50;">➕ Node</button>
                            <button class="btn" onclick="deleteSelected()" title="Delete the selected elements (edit mode; right-click also deletes)" style="font-size: 11px; padding: 6px; background: #f44336;">🗑️ Delete</button>
                        </div>
                        <div style="font-size: 10px; color: var(--vp-text-2); line-height: 1.3;">
                            • Click node → drag to connect<br>
                            • <strong>Double-click to edit properties</strong><br>
                            • Right-click to delete
                        </div>
                    </div>
                    </div>
                </div>
                
                <!-- History accordion (expanded by default) -->
                <div class="palette-section vp-acc open" id="accHistory">
                    <button class="vp-acc-header" onclick="toggleAccordion('history')" title="Show or hide the undo/redo history">
                        <span>↩️ History</span><span class="vp-acc-chevron">▾</span>
                    </button>
                    <div class="vp-acc-body" id="accHistoryBody">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px;">
                        <button class="btn" id="undoBtn" onclick="undo()" style="font-size: 11px; padding: 6px; background: #6b7280;" title="Undo the last operation (⌘Z / ⌃Z)">↩️ Undo</button>
                        <button class="btn" id="redoBtn" onclick="redo()" style="font-size: 11px; padding: 6px; background: #6b7280;" title="Redo the last undone operation (⌘⇧Z / ⌃Y)">↪️ Redo</button>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 6px;">
                        <select id="historyList" onchange="jumpToHistory(this.selectedIndex)" title="Operation history — every action is recorded; select an entry to undo/redo to it" style="width: 100%; font-size: 11px; padding: 4px; border-radius: 3px;">
                            <option disabled>▶ Current state</option>
                        </select>
                    </div>
                    </div>
                </div>
                
                <!-- Selection & Color accordion (expanded by default) -->
                <div class="palette-section vp-acc open" id="accSelection">
                    <button class="vp-acc-header" onclick="toggleAccordion('selection')" title="Show or hide the selection and color tools">
                        <span>🎛️ Selection & Color</span><span class="vp-acc-chevron">▾</span>
                    </button>
                    <div class="vp-acc-body" id="accSelectionBody">
                    <div id="selectionSummary" class="vp-chip">Nothing selected</div>
                <!-- Individual Selection Section -->
                <div class="palette-section">
                    <h4>Selected element(s)</h4>
                    <div id="selectedInfo" class="selected-info vp-chip">
                        Click on a node or edge to customize its color<br>
                        <em>Hold Shift to select multiple elements</em>
                    </div>
                    <div id="individualControls" style="display: none;">
                        <div class="color-group">
                            <label>Color:</label>
                            <div class="color-input-group">
                                <input type="color" id="individualColor" value="#3498db" title="Color for the selected element(s)">
                                <input type="text" id="individualColorText" value="#3498db" readonly title="Selected color in hex">
                            </div>
                        </div>
                        <div class="color-group">
                            <label>Opacity:</label>
                            <div class="color-input-group">
                                <input type="number" id="individualOpacity" min="0" max="100" step="1" value="100" oninput="updateOpacityDisplay('individual', this.value)" title="Opacity for the selected element(s)">
                                <span class="alpha-value" id="individualOpacityValue">100%</span>
                            </div>
                        </div>
                        <div class="color-group" id="assignGroupRow">
                            <label>Group:</label>
                            <div style="display: flex; gap: 6px; align-items: center;">
                                <select id="assignGroupSelect" title="Target group for the selected nodes (every node belongs to exactly one group)" style="flex: 1; width: auto;"></select>
                                <button class="apply-btn" onclick="assignSelectedToGroup()" title="Move the selected nodes into this group; Unassigned empties them from every group" style="flex: 0 0 auto; background: #2196f3; padding: 5px 8px;">Assign</button>
                            </div>
                        </div>
                        <!-- Geometry: precise numeric size/position editing -->
                        <div class="color-group" id="geomNodeGroup" style="display: none;">
                            <label>Position / Size (node):</label>
                            <div style="display: grid; grid-template-columns: auto 1fr auto 1fr; gap: 4px; align-items: center; font-size: 10px; color: #555;">
                                <span>X</span>
                                <input type="number" id="selGeomX" step="1" style="width: 100%; padding: 3px; border-radius: 3px; font-size: 11px;" title="Precise X position of the selected node">
                                <span>Y</span>
                                <input type="number" id="selGeomY" step="1" style="width: 100%; padding: 3px; border-radius: 3px; font-size: 11px;" title="Precise Y position of the selected node">
                            </div>
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px; align-items: center; margin-top: 4px; font-size: 10px; color: #555;">
                                <span>Size&nbsp;(px)</span>
                                <input type="number" id="selGeomSize" min="1" step="1" style="width: 100%; padding: 3px; border-radius: 3px; font-size: 11px;" title="Size in pixels of the selected node">
                            </div>
                        </div>
                        <div class="color-group" id="geomEdgeGroup" style="display: none;">
                            <label>Width (edge):</label>
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px; align-items: center; font-size: 10px; color: #555;">
                                <span>Width&nbsp;(px)</span>
                                <input type="number" id="selGeomWidth" min="0.5" step="0.5" style="width: 100%; padding: 3px; border-radius: 3px; font-size: 11px;" title="Width in pixels of the selected edge">
                            </div>
                            <div style="font-size: 9px; color: var(--vp-text-2); margin-top: 3px;">Edges are anchored to their endpoints — no free position.</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px;">
                            <button class="btn" id="alignHBtn" onclick="alignSelectedNodes('h')" title="Align selected nodes horizontally (same Y)" style="font-size: 10px; padding: 5px; background: #00897b; opacity: 0.4;">⇔ Align H</button>
                            <button class="btn" id="alignVBtn" onclick="alignSelectedNodes('v')" title="Align selected nodes vertically (same X)" style="font-size: 10px; padding: 5px; background: #00897b; opacity: 0.4;">⇕ Align V</button>
                        </div>
                        <button class="apply-btn" onclick="applyIndividualColor()" title="Apply the chosen color and opacity to the selected elements">Apply to Selected</button>
                        <button class="apply-btn" id="applyGeometryBtn" onclick="applySelectedGeometry()" title="Apply the numeric position/size values to the selected element" style="background: #00838f;">Apply Size/Position</button>
                        <button class="clear-selection-btn" onclick="clearSelection()" title="Deselect all elements">Clear Selection</button>
                    </div>
                </div>
                
                <!-- Groups Section -->
                <div class="palette-section">
                    <h4>🎯 Groups</h4>
                    <div class="color-group">
                        <label>Select Group:</label>
                        <div class="color-input-group">
                            <select id="groupSelector" onchange="updateGroupControls()" title="Choose which group of elements to edit" style="width: 100%;">
                                <optgroup label="Nodes">
                                    {standard_group_options_html}
                                    {extra_node_group_options}
                                    <option value="all_nodes">All Nodes</option>
                                </optgroup>
                                <optgroup label="Edges">
                                    <option value="positive_edges">Positive Edges</option>
                                    <option value="negative_edges">Negative Edges</option>
                                    <option value="all_edges">All Edges</option>
                                </optgroup>
                                {hemisphere_group_options}
                                <optgroup label="NT Edges" id="ntEdgeGroup">
                                    {nt_edge_group_options}
                                </optgroup>
                            </select>
                        </div>
                    </div>
                    <div class="color-group">
                        <label id="groupColorLabel">Color:</label>
                        <div class="color-input-group">
                            <input type="color" id="groupColor" value="{self.node_color[0]}" title="Color to apply to the chosen group">
                            <input type="text" id="groupColorText" value="{self.node_color[0]}" readonly title="Group color in hex">
                        </div>
                    </div>
                    <div class="color-group">
                        <label>Opacity:</label>
                        <div class="color-input-group">
                            <input type="number" id="groupOpacity" min="0" max="100" step="1" value="100" oninput="updateOpacityDisplay('group', this.value)" title="Opacity to apply to the chosen group">
                            <span class="alpha-value" id="groupOpacityValue">100%</span>
                        </div>
                    </div>
                    <button class="apply-btn" onclick="applyGroupColor()" title="Apply the chosen color and opacity to every element in the group">Apply to Group</button>
                    <div style="font-size: 10px; color: var(--vp-text-2); margin-top: 8px; line-height: 1.3;">
                        💡 Use dropdown to select which group to edit.<br>
                        Changes apply to all elements in the group.
                    </div>
                </div>
                
                <!-- Custom Groups Section -->
                <div class="palette-section">
                    <h4>📁 Custom groups</h4>
                    <div style="font-size: 11px; color: var(--vp-text-2); margin-bottom: 8px;">
                        Create node groups from selected nodes (single membership;
                        edges cannot join)
                    </div>
                    <div class="color-group">
                        <label>Group Name:</label>
                        <input type="text" id="customGroupName" placeholder="My Group" style="width: 100%; padding: 4px; border-radius: 3px;" title="Name for the new custom group">
                    </div>
                    <div style="display: flex; gap: 6px; margin-top: 8px;">
                        <button class="apply-btn" onclick="createCustomGroup()" title="Create a custom group from the currently selected elements" style="flex: 1; background: #2196F3;">➕ Create</button>
                        <button class="apply-btn" onclick="deleteCustomGroup()" title="Delete the selected custom group" style="flex: 1; background: #f44336;">🗑️ Delete</button>
                    </div>
                    <select id="customGroupList" style="width: 100%; margin-top: 8px; padding: 4px; display: none;" title="Saved custom groups">
                        <option value="">-- Custom Groups --</option>
                    </select>
                </div>
                
                <!-- Quick Actions Section -->
                <div class="palette-section">
                    <h4>⚡ Quick actions</h4>
                    {role_quick_actions_html}
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">
                        <button class="btn" onclick="selectGroup('all_edges')" title="Select all edges" style="font-size: 10px; padding: 5px; background: {self.edge_color};">All edges</button>
                    </div>
                    {extra_group_buttons}
                    <button class="apply-btn" onclick="applyGlobalColors()" title="Restore the default colors for every node and edge group" style="background: #9c27b0;">🔄 Reset All Colors</button>
                </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Hover Info Display (Bottom-Left): element tooltips ONLY -->
    <div id="hoverInfo">
        💡 <b>Hover over nodes or edges</b> for details · Press <b>?</b> for help
    </div>

    <!-- Toast stack: operation feedback (success/info/warn/error) -->
    <div id="toastStack" aria-live="polite" aria-label="Operation notifications"></div>

    <!-- In-page dialog (replaces native prompt/confirm); content built per open -->
    <div id="vpDialogOverlay" role="dialog" aria-modal="true" aria-labelledby="vpDialogTitle" onclick="if (event.target === this) cancelDialog()">
        <div class="vp-dialog">
            <h3 id="vpDialogTitle"></h3>
            <div id="vpDialogMessage" class="vp-dialog-message" style="display: none;"></div>
            <div id="vpDialogBody"></div>
            <div class="vp-dialog-actions">
                <button type="button" id="vpDialogCancel" class="vp-dialog-btn" onclick="cancelDialog()" title="Cancel this dialog (Esc)">Cancel</button>
                <button type="button" id="vpDialogConfirm" class="vp-dialog-btn vp-primary" onclick="confirmDialog()" title="Confirm this dialog (Enter)">OK</button>
            </div>
        </div>
    </div>

    <!-- Help overlay (? key / ❓ button) -->
    <div id="helpOverlay" role="dialog" aria-modal="true" aria-label="Help: gestures, shortcuts, recipes" onclick="if (event.target === this) closeHelp()">
        <div class="vp-help-card">
            <button type="button" id="helpCloseBtn" onclick="closeHelp()" title="Close help (Esc)">✕</button>
            <h3>Help</h3>
            <div class="vp-help-cols">
                <div id="helpMouse">
                    <h4>🖱️ Mouse</h4>
                    <table>
                        <tr><td class="vp-key">Click</td><td>Select a node / edge</td></tr>
                        <tr><td class="vp-key">Shift+Click / drag</td><td>Multi-select (box select) — L→R frame: FULL; R→L dashed frame: TOUCH; frame over selected objects deselects them (⇧ toggles)</td></tr>
                        <tr><td class="vp-key">Double-click</td><td>Highlight a node's connections</td></tr>
                        <tr><td class="vp-key">Right-click</td><td>Hide node / edge</td></tr>
                        <tr><td class="vp-key">Drag node</td><td>Move it (undoable)</td></tr>
                        <tr><td class="vp-key">Scroll / drag bg</td><td>Zoom / pan</td></tr>
                        <tr><td class="vp-key">Edit mode</td><td>Click two nodes to draw an edge; double-click to edit properties; right-click deletes</td></tr>
                    </table>
                </div>
                <div id="helpKeyboard">
                    <h4>⌨️ Keyboard</h4>
                    <table>
                        <tr><td class="vp-key">H</td><td>Hide selected nodes</td></tr>
                        <tr><td class="vp-key">E</td><td>Hide selected edges</td></tr>
                        <tr><td class="vp-key">L</td><td>Toggle label position (center / outside)</td></tr>
                        <tr><td class="vp-key">?</td><td>Toggle this help</td></tr>
                        <tr><td class="vp-key">⌘Z / ⌃Z</td><td>Undo</td></tr>
                        <tr><td class="vp-key">⌘⇧Z / ⌃Y</td><td>Redo</td></tr>
                        <tr><td class="vp-key">Enter / Esc</td><td>Confirm / cancel dialogs, clear search</td></tr>
                        <tr><td class="vp-key">↑ / ↓</td><td>Find node: previous / next match (same as the ▲/▼ buttons)</td></tr>
                    </table>
                </div>
                <div id="helpRecipes">
                    <h4>📖 Recipes</h4>
                    <p><strong>Hide edges by weight</strong> — comma = OR, parentheses = AND:<br>
                    <code>&lt;5, &gt;100</code> hides weak and very strong edges;<br>
                    <code>(&gt;=10, &lt;=20)</code> keeps only a weight band.</p>
                    <p><strong>Save vs Export Layout</strong> — Save keeps positions in this browser for this file; Export Layout downloads them as JSON to share or re-import elsewhere.</p>
                    <p><strong>Horizontal/Vertical Gap &amp; Rotate</strong> — set the center-to-center distance between neighboring nodes in px (node size unchanged); rotation turns the whole arrangement around its center — the gaps are measured along the layout's own columns/rows, so small tilts keep the values stable, while at ±90°/270° the Horizontal and Vertical values swap accordingly (↺ Reset Spacing restores the layout-run gaps for the current rotation); both reset when the layout re-runs and are undoable.</p>
                    <p><strong>Panels</strong> — ⚙️ hides the top tool panel, 🎨 the right panel; both states are remembered.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Register layout extensions
        if (typeof cytoscape !== 'undefined') {{
            // Register dagre
            if (typeof dagre !== 'undefined' && typeof cytoscapeDagre !== 'undefined') {{
                cytoscape.use(cytoscapeDagre);
            }}
            // Register cose-bilkent
            if (typeof cytoscapeCoseBilkent !== 'undefined') {{
                cytoscape.use(cytoscapeCoseBilkent);
            }}
            // Register fcose
            if (typeof cytoscapeFcose !== 'undefined') {{
                cytoscape.use(cytoscapeFcose);
            }}
            // Register klay
            if (typeof cytoscapeKlay !== 'undefined') {{
                cytoscape.use(cytoscapeKlay);
            }}
        }}
        
        const elements = {{
            nodes: {json_safe(nodes_data, default=_json_default)},
            edges: {json_safe(edges_data, default=_json_default)}
        }};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    // Hide the built-in selection frame: the directional box
                    // overlay below draws its own (solid = full, dashed =
                    // touch). Both fill AND border of the built-in frame
                    // multiply by selection-box-opacity, so 0 hides it all.
                    // The background-press circle stays visible as the PAN
                    // affordance — at half the default 30px radius; select
                    // mode hides it (see setCanvasDragMode).
                    selector: 'core',
                    style: {{
                        'selection-box-opacity': 0,
                        'active-bg-size': 15,
                        'active-bg-opacity': 0.15
                    }}
                }},
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'background-color': 'data(color)',
                        'color': '#000',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'font-family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
                        'width': '40px',
                        'height': '40px',
                        'border-width': '0px',  // No border
                        'text-wrap': 'wrap',
                        'text-max-width': '80px'
                    }}
                }},
                {{
                    selector: 'node.labels-outside',
                    style: {{
                        'text-valign': 'bottom',
                        'text-margin-y': '5px'
                    }}
                }},
                {{
                    selector: 'node.labels-hidden',
                    style: {{
                        'label': ''
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        // Visible selection: a thick colored border plus a
                        // soft overlay halo (the overlay alone was the only
                        // feedback and the thin light-yellow border was
                        // invisible against the white canvas)
                        'border-width': '4px',
                        'border-color': '{self.highlight_color}',
                        'overlay-color': '{self.highlight_color}',
                        'overlay-opacity': 0.25,
                        'overlay-padding': '6px'
                    }}
                }},
                {{
                    selector: 'node.hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'node.orphan-hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'node.deadend-hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge.selfloop-hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 'mapData(scaled_width, {min_scaled_width}, {max_scaled_width}, 1, 10)',
                        'line-color': 'data(color)',  // Per-edge color from the uploaded edge list (falls back to link_color)
                        'target-arrow-color': 'data(color)',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'arrow-scale': 1.5,
                        'label': '',
                        'font-size': '10px',
                        'font-family': "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
                        'text-wrap': 'wrap',
                        'text-max-width': '150px',
                        'text-halign': 'left',
                        'text-valign': 'top',
                        'color': '#000',
                        'text-margin-x': '0px',
                        'text-margin-y': '0px'
                    }}
                }},
                {{
                    selector: 'edge[is_negative = 1]',
                    style: {{
                        'line-color': '#4A90E2',  // Light blue for negative
                        'target-arrow-color': '#4A90E2'
                    }}
                }},
                {{
                    // On-edge weight labels — applied via the 'wlabel'
                    // class (the Edge Weights toggle in the Filter tab);
                    // the shared text stack (background pill, border,
                    // wrap) comes from the base edge style.  Size starts
                    // at edge_label_font_size and follows the Edge Label
                    // Size control live.
                    selector: 'edge.wlabel',
                    style: {{
                        'label': 'data(display_label)',
                        'text-rotation': 'autorotate',
                        'font-size': '{self.edge_label_font_size}px',
                        'color': '#333'
                    }}
                }},
                {nt_edge_styles}
                {{
                    selector: 'edge.hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge.filtered',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge.deadend-hidden',
                    style: {{
                        'display': 'none'
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '{self.highlight_color}',
                        'target-arrow-color': '{self.highlight_color}',
                        'width': 'mapData(scaled_width, {min_scaled_width}, {max_scaled_width}, 3, 16)'
                    }}
                }},
                {{
                    selector: 'edge.highlighted',
                    style: {{
                        'line-color': '{self.highlight_color}',
                        'target-arrow-color': '{self.highlight_color}',
                        'width': 'mapData(scaled_width, {min_scaled_width}, {max_scaled_width}, 3, 20)',
                        'z-index': 999
                    }}
                }},
                {{
                    selector: 'node.placeholder',
                    style: {{
                        'display': 'none',
                        'opacity': 0,
                        'width': 0,
                        'height': 0
                    }}
                }},
                {{
                    selector: 'edge.placeholder',
                    style: {{
                        'display': 'none',
                        'opacity': 0
                    }}
                }}
            ],
            layout: {{
                name: 'preset'  // Use preset first, then apply proper layout after initialization
            }},
            wheelSensitivity: 0.3,
            minZoom: 0.1,
            maxZoom: 5,
            selectionType: 'single',
            userZoomingEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: true,
            autoungrabify: false,
            autounselectify: false
        }});

        let straightReciprocalEdgesEnabled = {'true' if self.straight_reciprocal_edges else 'false'};
        const highlightColor = '{self.highlight_color}';
        const highlightOpacity = {self.highlight_opacity};
        const defaultReciprocalOffset = 5;
        let reciprocalOffset = defaultReciprocalOffset;

        // Initialize layout algorithm variable and configuration function
        let currentLayoutAlgorithm = '{js_layout_name}';
        let labelPosition = 'center';  // 'center' or 'outside'
        let labelsVisible = true;
        const hasHemisphereNodes = {'true' if has_hemi_controls else 'false'};
        let hemisphereMirrorEnabled = {'true' if self.hemisphere_mirror_default and has_hemi_controls else 'false'};
        let originalHemispherePositions = null;
        let hemisphereTemplateSide = null;

        // Layout transforms: the inter-node gap is an ABSOLUTE center-to-
        // center distance in px (the median neighbor distance along the
        // axis), independent of node size; rotation composes by applying
        // (target - lastApplied) around the visible centroid. Both
        // transforms anchor on the centroid, which neither moves, so they
        // compose without drift and rotation conserves every pairwise
        // distance (the H and V gap axes swap under a 90-degree rotation).
        // Baseline gaps are captured at each layout run for Reset Spacing.
        let lastGapX = null;
        let lastGapY = null;
        let baselineGapX = null;
        let baselineGapY = null;
        let lastRotationDeg = 0;
        // Pre-transform snapshot captured at the FIRST 'input' of a slider
        // drag and committed to history on 'change' (release), so one slider
        // interaction produces ONE history entry instead of dozens.
        let pendingTransformState = null;
        // Per-drag dirty gates: an 'input' that never altered geometry
        // (e.g. a full-turn rotation wrap) commits no history on 'change'.
        // Declared WITH the tracker state because the initial-layout call
        // below resets them during script evaluation.
        let spacingDirtySinceInput = false;
        let rotationDirtySinceInput = false;
        
        function getLayoutConfig(layoutName) {{
            // Configure layouts with optimal settings for crossing minimization
            const configs = {{
                'mapping': {{
                    name: 'preset',             // Positions precomputed by hop layer
                    fit: true,
                    animate: false,
                    padding: 50
                }},
                'dagre': {{
                    name: 'dagre',
                    rankDir: '{self.dagre_rank_dir}',  // Document flow direction (TB hierarchy, LR mapping)
                    nodeSep: 50,                // Horizontal spacing between nodes
                    edgeSep: 20,                // Spacing for edges
                    rankSep: 100,               // Vertical spacing between ranks
                    ranker: 'network-simplex',  // Best ranker for crossing minimization
                    animate: false,             // No animation on initial load
                    padding: 50
                }},
                'klay': {{
                    name: 'klay',
                    direction: 'DOWN',
                    nodePlacement: 'BRANDES_KOEPF',
                    edgeRouting: 'ORTHOGONAL',
                    spacing: 50,
                    animate: false,
                    padding: 50
                }},
                'fcose': {{
                    name: 'fcose',
                    quality: 'proof',
                    randomize: false,
                    animate: false,
                    idealEdgeLength: 100,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 2500,
                    tile: true,
                    padding: 50
                }},
                'cose-bilkent': {{
                    name: 'cose-bilkent',
                    quality: 'proof',
                    randomize: false,
                    animate: false,
                    idealEdgeLength: 100,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 2500,
                    tile: true,
                    padding: 50
                }},
                'cose': {{
                    name: 'cose',
                    randomize: false,
                    animate: false,
                    idealEdgeLength: 100,
                    nodeOverlap: 20,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    padding: 50
                }},
                'breadthfirst': {{
                    name: 'breadthfirst',
                    directed: true,
                    circle: false,
                    grid: false,
                    spacingFactor: 1.75,
                    animate: false,
                    padding: 50
                }},
                'circle': {{
                    name: 'circle',
                    animate: false,
                    padding: 50,
                    spacingFactor: 1.5
                }},
                'grid': {{
                    name: 'grid',
                    animate: false,
                    padding: 50,
                    spacingFactor: 1.5
                }},
                'concentric': {{
                    name: 'concentric',
                    animate: false,
                    padding: 50,
                    spacingFactor: 1.5,
                    minNodeSpacing: 50
                }}
            }};
            
            return configs[layoutName] || configs['dagre'];
        }}
        
        // Set the layout selector to show the current layout
        const layoutSelector = document.getElementById('layoutSelector');
        if (layoutSelector) {{
            layoutSelector.value = currentLayoutAlgorithm;
        }}
        
        // Elements in the CURRENT view: not hidden by any hide/filter class.
        // Class-based (deterministic within a synchronous block; :visible /
        // el.visible() depend on style recalculation timing right after
        // addClass). Keep in sync with isEdgeInCurrentGraph.
        function isVisibleElement(el) {{
            return !el.hasClass('hidden') && !el.hasClass('filtered') &&
                   !el.hasClass('selfloop-hidden') && !el.hasClass('orphan-hidden') &&
                   !el.hasClass('deadend-hidden');
        }}

        function cacheHemispherePositions() {{
            if (!hasHemisphereNodes) return;
            originalHemispherePositions = {{}};
            cy.nodes().filter(isVisibleElement).forEach(n => {{
                originalHemispherePositions[n.id()] = n.position();
            }});
        }}

        function getHemisphereTemplateSide() {{
            if (!hasHemisphereNodes) return null;
            // Only VISIBLE nodes decide the template side (hidden nodes must
            // not influence the mirrored layout).
            const visibleNodes = cy.nodes().filter(isVisibleElement);
            const hasL = visibleNodes.some(n => n.data('hemisphere') === 'L');
            const hasR = visibleNodes.some(n => n.data('hemisphere') === 'R');
            if (hasL) return 'L';
            if (hasR) return 'R';
            return null;
        }}

        let hemispherePlaceholderIds = [];
        let hemispherePlaceholderEdgeIds = [];

        // Build unique placeholder set as union of all base names (without _L/_R suffix)
        function buildPlaceholderSet() {{
            // Only VISIBLE nodes participate: the mirror layout must ignore
            // hidden nodes and edges (dead-end/orphan/filter/manual hides).
            const nodes = cy.nodes().filter(isVisibleElement);
            const baseNames = new Set();
            nodes.forEach(n => {{
                const base = n.data('base_name');
                if (base) baseNames.add(base);
            }});
            return baseNames;
        }}

        // Create placeholder nodes for layout, one per unique base name
        function createPlaceholderNodes() {{
            hemispherePlaceholderIds = [];
            hemispherePlaceholderEdgeIds = [];
            const baseNames = buildPlaceholderSet();
            const nodes = cy.nodes().filter(isVisibleElement);
            
            // Map base -> node info for copying type/color
            const baseInfo = {{}};
            nodes.forEach(n => {{
                const base = n.data('base_name');
                if (base && !baseInfo[base]) {{
                    baseInfo[base] = {{
                        node_type: n.data('node_type') || 'intermediate',
                        color: n.data('color') || '#888'
                    }};
                }}
            }});
            
            baseNames.forEach(base => {{
                const placeholderId = `__hemi_ph__${{base}}`;
                if (cy.getElementById(placeholderId).length > 0) return;
                const info = baseInfo[base] || {{ node_type: 'intermediate', color: '#888' }};
                cy.add({{
                    group: 'nodes',
                    data: {{
                        id: placeholderId,
                        label: base,
                        node_type: info.node_type,
                        color: info.color,
                        base_name: base,
                        is_placeholder: 1
                    }},
                    classes: 'placeholder'
                }});
                hemispherePlaceholderIds.push(placeholderId);
            }});
            
            // Create placeholder edges mirroring real edges (by base name),
            // using only VISIBLE real edges.
            const edgeSet = new Set();
            cy.edges().filter(isVisibleElement).forEach(e => {{
                const srcBase = e.source().data('base_name');
                const tgtBase = e.target().data('base_name');
                if (srcBase && tgtBase) {{
                    const key = `${{srcBase}}->${{tgtBase}}`;
                    if (!edgeSet.has(key)) {{
                        edgeSet.add(key);
                        const phEdgeId = `__hemi_ph_edge__${{srcBase}}__${{tgtBase}}`;
                        const srcPh = `__hemi_ph__${{srcBase}}`;
                        const tgtPh = `__hemi_ph__${{tgtBase}}`;
                        if (cy.getElementById(srcPh).length && cy.getElementById(tgtPh).length) {{
                            cy.add({{
                                group: 'edges',
                                data: {{
                                    id: phEdgeId,
                                    source: srcPh,
                                    target: tgtPh,
                                    is_placeholder: 1
                                }},
                                classes: 'placeholder'
                            }});
                            hemispherePlaceholderEdgeIds.push(phEdgeId);
                        }}
                    }}
                }}
            }});
        }}

        // Remove all placeholder nodes and edges
        function removePlaceholders() {{
            // First remove by ID arrays
            hemispherePlaceholderEdgeIds.forEach(id => {{
                const el = cy.getElementById(id);
                if (el.length) el.remove();
            }});
            hemispherePlaceholderIds.forEach(id => {{
                const el = cy.getElementById(id);
                if (el.length) el.remove();
            }});
            hemispherePlaceholderIds = [];
            hemispherePlaceholderEdgeIds = [];
            
            // Also remove any remaining elements with placeholder class (safety net)
            cy.elements('.placeholder').remove();
        }}

        // Run hemisphere-aware layout with mirroring
        // 1. Create placeholder set (unique base names)
        // 2. Layout placeholder set using current algorithm
        // 3. Move placeholder layout to RIGHT panel
        // 4. Position _L nodes (and _U, no-suffix) using placeholder positions
        // 5. Mirror _R nodes to LEFT panel
        // 6. Remove placeholders
        function runHemisphereMirrorLayout() {{
            if (!hasHemisphereNodes) return;
            
            // Create placeholder nodes/edges
            createPlaceholderNodes();
            
            // Layout only placeholder elements using class selector
            const phNodes = cy.nodes('.placeholder');
            const phEdges = cy.edges('.placeholder');
            const phElements = phNodes.union(phEdges);
            
            if (phElements.length === 0) {{
                removePlaceholders();
                return;
            }}
            
            // Temporarily make placeholders visible for layout computation
            phElements.style('display', 'element');
            phElements.style('opacity', 1);
            
            const layoutConfig = getLayoutConfig(currentLayoutAlgorithm);
            layoutConfig.animate = false;
            layoutConfig.fit = false;
            
            const layout = phElements.layout(layoutConfig);
            
            // Define the callback function for positioning nodes after layout
            const positionNodesAfterLayout = () => {{
                // Get placeholder positions and compute bounds
                const phPositions = {{}};
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                
                phNodes.forEach(n => {{
                    const base = n.data('base_name');
                    const pos = n.position();
                    phPositions[base] = {{ x: pos.x, y: pos.y }};
                    minX = Math.min(minX, pos.x);
                    maxX = Math.max(maxX, pos.x);
                    minY = Math.min(minY, pos.y);
                    maxY = Math.max(maxY, pos.y);
                }});
                
                const layoutWidth = maxX - minX || 200;
                const layoutCenterX = (minX + maxX) / 2;
                // Gap between L and R panels: use minimum of 50px or 10% of layout width
                const gap = Math.max(50, layoutWidth * 0.1);
                
                // Calculate panel positions: R on LEFT, L on RIGHT
                // Right panel center (for L nodes): layoutCenterX + gap/2 + layoutWidth/2
                // Left panel center (for R nodes): layoutCenterX - gap/2 - layoutWidth/2
                const rightPanelOffset = gap / 2 + layoutWidth / 2;
                const leftPanelOffset = -(gap / 2 + layoutWidth / 2);
                
                // Remove placeholders before positioning real nodes
                removePlaceholders();
                
                // Position real nodes based on placeholder layout
                // (only VISIBLE nodes - hidden nodes keep their positions
                // and must not be moved by the mirrored layout)
                cy.batch(() => {{
                    cy.nodes().filter(isVisibleElement).forEach(n => {{
                        const base = n.data('base_name');
                        const hemi = n.data('hemisphere');
                        if (!base || !phPositions[base]) return;
                        
                        const refPos = phPositions[base];
                        const relX = refPos.x - layoutCenterX;  // Position relative to center
                        
                        if (hemi === 'L' || !hemi || hemi === 'U') {{
                            // L, U, and unsuffixed nodes go to RIGHT panel
                            n.position({{
                                x: layoutCenterX + rightPanelOffset + relX,
                                y: refPos.y
                            }});
                        }} else if (hemi === 'R') {{
                            // R nodes go to LEFT panel, mirrored
                            n.position({{
                                x: layoutCenterX + leftPanelOffset - relX,  // Mirror X
                                y: refPos.y
                            }});
                        }}
                    }});
                }});
                
                cy.fit();
            }};
            
            // Bind the event BEFORE running the layout to ensure we catch layoutstop
            layout.one('layoutstop', positionNodesAfterLayout);
            layout.run();
            
            // Fallback: if layout is synchronous and layoutstop already fired, 
            // use setTimeout as a safety net
            setTimeout(() => {{
                if (cy.nodes('.placeholder').length > 0) {{
                    // Layout didn't trigger our callback, call it manually
                    positionNodesAfterLayout();
                }}
            }}, 100);
        }}

        function restoreHemispherePositions() {{
            if (!hasHemisphereNodes || !originalHemispherePositions) return;
            cy.batch(() => {{
                Object.keys(originalHemispherePositions).forEach(id => {{
                    const node = cy.getElementById(id);
                    if (node.length) node.position(originalHemispherePositions[id]);
                }});
            }});
            cy.fit();
        }}

        function toggleHemisphereMirror() {{
            if (!hasHemisphereNodes) return;
            pushHistory('Toggle mirror');
            hemisphereMirrorEnabled = !hemisphereMirrorEnabled;
            const btn = document.getElementById('mirrorHemiBtn');
            if (btn) {{
                btn.style.background = hemisphereMirrorEnabled ? '#0ea5e9' : '#64748b';
                btn.textContent = hemisphereMirrorEnabled ? '🪞 Mirrored' : '🪞 Mirror Hemispheres';
            }}
            if (hemisphereMirrorEnabled) {{
                if (!originalHemispherePositions) cacheHemispherePositions();
                runHemisphereMirrorLayout();
            }} else {{
                restoreHemispherePositions();
            }}
        }}

        // Apply the initial layout using proper configuration
        const initialLayout = getLayoutConfig(currentLayoutAlgorithm);
        cy.layout(initialLayout).run();
        resetLayoutTransformTrackers();
        setTimeout(() => {{
            cacheHemispherePositions();
            if (hemisphereMirrorEnabled) runHemisphereMirrorLayout();
        }}, 200);
        
        // Fix for click position drift - ensure Cytoscape canvas is properly sized
        cy.resize();
        cy.fit();

        initializeReciprocalOffsetControls();

        const applyReciprocalOffsets = () => refreshEdgeStyles(false);
        setTimeout(applyReciprocalOffsets, 0);
        cy.on('layoutstop', applyReciprocalOffsets);
        cy.on('dragfree', 'node', applyReciprocalOffsets);
        
        // Add window resize handler to keep canvas in sync
        window.addEventListener('resize', function() {{
            cy.resize();
        }});
        
        // Initialize edge widths based on current dropdown selection
        // This ensures the correct log base is applied on page load
        setTimeout(function() {{
            updateEdgeWidths();
        }}, 100);

        // Show hover info in bottom-left box (like coana)
        cy.on('mouseover', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            const info = document.getElementById('hoverInfo');
            let html = `
                <b>Node:</b> ${{escapeHtml(data.label)}}<br>
                <b>Type:</b> ${{escapeHtml(data.role || data.node_type)}}<br>
                ${{data.group ? `<b>Dataset:</b> ${{escapeHtml(data.group)}}<br>` : ''}}
                ${{(data.assigned_group !== undefined && data.assigned_group !== '' && data.assigned_group !== data.node_type) ? `<b>Group:</b> ${{escapeHtml(groupLabel(data.assigned_group))}}<br>` : ''}}
                <b>Color:</b> ${{escapeHtml(data.color)}}
            `;
            if (data.hemisphere) {{
                html += `<br><b>Hemisphere:</b> ${{escapeHtml(data.hemisphere)}}`;
            }}
            // Add dataset info if available
            if (data.dataset_info && Object.keys(data.dataset_info).length > 0) {{
                html += `<br><span style="color: #888; font-size: 0.9em;">─────────────</span><br><b>Names by dataset:</b>`;
                for (const [code, name] of Object.entries(data.dataset_info).sort()) {{
                    html += `<br>&nbsp;&nbsp;${{escapeHtml(code)}}: ${{escapeHtml(name)}}`;
                }}
            }}
            info.innerHTML = html;
        }});

        cy.on('mouseover', 'edge', function(evt) {{
            const edge = evt.target;
            const data = edge.data();
            const source = edge.source().data('label');
            const target = edge.target().data('label');
            const info = document.getElementById('hoverInfo');
            
            // Get display weight (add negative sign if needed)
            const displayWeight = data.is_negative === 1 ? -data.weight : data.weight;
            
            let html = `<b>Connection:</b> ${{escapeHtml(source)}} → ${{escapeHtml(target)}}<br>`;
            
            // Highlight the current metric
            if (currentMetric === 'weight') {{
                html += `<b>Weight:</b> <span style="color: #4CAF50; font-weight: bold;">${{displayWeight.toLocaleString()}} {self.edge_weight_label} ⬅ Current</span>`;
            }} else {{
                html += `<b>Weight:</b> ${{displayWeight.toLocaleString()}} {self.edge_weight_label}`;
            }}
            
            if (data.ratio && !isNaN(data.ratio)) {{
                if (currentMetric === 'ratio') {{
                    html += `<br><b>Ratio:</b> <span style="color: #4CAF50; font-weight: bold;">${{data.ratio.toFixed(4)}} ⬅ Current</span>`;
                }} else {{
                    html += `<br><b>Ratio:</b> ${{data.ratio.toFixed(4)}}`;
                }}
            }}
            if (data.probability && !isNaN(data.probability)) {{
                if (currentMetric === 'probability') {{
                    html += `<br><b>Probability:</b> <span style="color: #4CAF50; font-weight: bold;">${{data.probability.toFixed(4)}} ⬅ Current</span>`;
                }} else {{
                    html += `<br><b>Probability:</b> ${{data.probability.toFixed(4)}}`;
                }}
            }}
            
            // Display NT type if available
            if (data.nt_type && data.nt_type !== '') {{
                const ntColor = getNTColor(data.nt_type);
                html += `<br><b>NT:</b> <span style="color: ${{ntColor}}; font-weight: bold;">${{escapeHtml(data.nt_type)}}</span>`;
            }}
            
            // Display custom edge labels (e.g., multi-dataset synapse strengths)
            if (data.custom_labels && typeof data.custom_labels === 'object' && Object.keys(data.custom_labels).length > 0) {{
                html += `<br><span style="color: #888; font-size: 0.9em;">─────────────</span>`;
                for (const [labelName, labelValue] of Object.entries(data.custom_labels)) {{
                    const formattedValue = typeof labelValue === 'number' ? labelValue.toLocaleString() : labelValue;
                    html += `<br><b>${{escapeHtml(labelName)}}:</b> <span style="white-space:pre-line">${{escapeHtml(formattedValue)}}</span>`;
                }}
            }}
            
            info.innerHTML = html;
        }});

        cy.on('mouseout', 'node, edge', function() {{
            const info = document.getElementById('hoverInfo');
            const hiddenCount = cy.nodes('.hidden').length;
            if (hiddenCount > 0) {{
                info.innerHTML = `
                    💡 <b>${{hiddenCount}}</b> node(s) hidden<br>
                    Click <b>Show All Nodes</b> button to restore
                `;
            }} else {{
                info.innerHTML = `
                    💡 <b>Hover over nodes or edges</b> to see details<br>
                    <b>Drag nodes</b> to reposition • <b>Scroll</b> to zoom • <b>Double-click</b> to highlight
                `;
            }}
        }});

        // Showing the restore button changes the ribbon's layout.  Cytoscape
        // caches the canvas' client rect for pointer projection, so a DOM
        // reflow without resize() makes subsequent clicks/pans use the old
        // canvas origin (the visible pointer and the hit-test point diverge).
        function resizeCanvasAfterVisibilityControlChange() {{
            if (typeof cy !== 'undefined') cy.resize();
        }}

        // Right-click to hide nodes/edges (or delete in edit mode)
        cy.on('cxttap', 'node', function(evt) {{
            if (editMode) {{
                // In edit mode: delete the node
                deleteElement(evt.target);
            }} else {{
                // Normal mode: hide the node
                pushHistory('Hide node');
                const node = evt.target;
                node.addClass('hidden');
                // Hide connected edges
                node.connectedEdges().addClass('hidden');
                document.getElementById('showAllBtn').style.display = 'inline-block';
                resizeCanvasAfterVisibilityControlChange();
                reapplyDeadEndHiding();
            }}
        }});

        // Right-click to hide edges (or delete in edit mode)
        cy.on('cxttap', 'edge', function(evt) {{
            if (editMode) {{
                // In edit mode: delete the edge
                deleteElement(evt.target);
            }} else {{
                // Normal mode: hide the edge
                pushHistory('Hide edge');
                const edge = evt.target;
                edge.addClass('hidden');
                document.getElementById('showAllBtn').style.display = 'inline-block';
                resizeCanvasAfterVisibilityControlChange();
                reapplyDeadEndHiding();
            }}
        }});

        // Keyboard shortcut: H to hide selected nodes
        document.addEventListener('keydown', function(e) {{
            // Ignore shortcuts while typing in inputs/textareas (undo/redo
            // below uses the same guard)
            const tag = e.target && e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) {{
                return;
            }}
            if (e.key === 'h' || e.key === 'H') {{
                if (dialogCtl.isActive()) return;
                const selected = cy.$('node:selected');
                if (selected.length > 0) {{
                    pushHistory('Hide nodes');
                    selected.addClass('hidden');
                    selected.connectedEdges().addClass('hidden');
                    document.getElementById('showAllBtn').style.display = 'inline-block';
                    resizeCanvasAfterVisibilityControlChange();
                    reapplyDeadEndHiding();
                }}
            }}
        }});

        // Keyboard shortcut: E to hide selected edges
        document.addEventListener('keydown', function(e) {{
            const tag = e.target && e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) {{
                return;
            }}
            if (dialogCtl.isActive()) return;
            if (e.key === 'e' || e.key === 'E') {{
                const selected = cy.$('edge:selected');
                if (selected.length > 0) {{
                    pushHistory('Hide edges');
                    selected.addClass('hidden');
                    document.getElementById('showAllBtn').style.display = 'inline-block';
                    resizeCanvasAfterVisibilityControlChange();
                    reapplyDeadEndHiding();
                }}
            }}
        }});

        // Keep inline highlight overrides in sync with Cytoscape's selection
        cy.on('select', 'edge', function(evt) {{
            applyEdgeHighlightOverride(evt.target);
        }});

        cy.on('unselect', 'edge', function(evt) {{
            clearEdgeHighlightOverride(evt.target);
        }});

        // Keyboard shortcut: L to toggle label position (center/outside)
        document.addEventListener('keydown', function(e) {{
            const tag = e.target && e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) {{
                return;
            }}
            if (dialogCtl.isActive()) return;
            if (e.key === 'l' || e.key === 'L') {{
                pushHistory('Toggle label position');
                if (labelPosition === 'center') {{
                    cy.nodes().addClass('labels-outside');
                    labelPosition = 'outside';
                    console.log('Labels moved outside nodes');
                }} else {{
                    cy.nodes().removeClass('labels-outside');
                    labelPosition = 'center';
                    console.log('Labels moved to center');
                }}
            }}
        }});

        // Double-click to highlight connections
        cy.on('dblclick', 'node', function(evt) {{
            removeHighlightFromEdges(cy.edges('.highlighted'), true);
            addHighlightToEdges(evt.target.connectedEdges());
        }});

        // Click empty space to clear highlights
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                removeHighlightFromEdges(cy.edges('.highlighted'), true);
            }}
        }});

        // Layout control functions (getLayoutConfig and currentLayoutAlgorithm defined earlier)
        function resetLayout() {{
            pushHistory('Reset layout');
            const config = getLayoutConfig(currentLayoutAlgorithm);
            // Add animation for reset
            config.animate = true;
            config.animationDuration = 500;
            // Animated layouts interpolate positions asynchronously, so the
            // tracker reset must wait for layoutstop — measuring right after
            // run() captures the OLD arrangement (stale gap spinners and a
            // Reset-Spacing baseline from the previous layout).
            cy.one('layoutstop', () => {{ resetLayoutTransformTrackers(); }});
            cy.layout(config).run();
        }}
        
        function changeLayout() {{
            pushHistory('Change layout');
            const selector = document.getElementById('layoutSelector');
            const newLayout = selector.value;
            currentLayoutAlgorithm = newLayout;
            
            // Update info text based on selected layout
            const layoutInfos = {{
                'dagre': '💡 Dagre uses Sugiyama\\'s algorithm for optimal edge crossing minimization in hierarchical graphs',
                'klay': '💡 KLay provides layer-based layout with advanced crossing reduction techniques',
                'breadthfirst': '💡 Breadth-first layout creates simple hierarchical structure based on graph traversal',
                'fcose': '💡 fCoSE (fast CoSE) balances speed and quality with compound graph support',
                'cose-bilkent': '💡 CoSE-Bilkent offers highest quality force-directed layout with better crossing minimization',
                'cose': '💡 CoSE (Compound Spring Embedder) is a standard force-directed layout algorithm',
                'circle': '💡 Circular layout arranges all nodes in a circle - simple but many crossings',
                'grid': '💡 Grid layout arranges nodes in a matrix - useful for small networks',
                'concentric': '💡 Concentric layout arranges nodes in nested circles based on hierarchy',
                'hemi-dagre': '🪞 Hemisphere-aware Dagre: layouts L/R neurons in mirrored panels',
                'hemi-fcose': '🪞 Hemisphere-aware fCoSE: layouts L/R neurons in mirrored panels'
            }};
            // The layout description surfaces as an operation toast (the
            // old #layoutInfo div was permanently hidden).
            showToast(layoutInfos[newLayout] || '', 'info');
            
            // Check if this is a hemisphere-aware layout
            const isHemiLayout = newLayout.startsWith('hemi-');
            
            if (isHemiLayout && hasHemisphereNodes) {{
                // Extract the base layout algorithm
                const baseLayout = newLayout.replace('hemi-', '');
                currentLayoutAlgorithm = baseLayout;
                // Enable mirroring and run hemisphere layout
                hemisphereMirrorEnabled = true;
                const btn = document.getElementById('mirrorHemiBtn');
                if (btn) {{
                    btn.style.background = '#0ea5e9';
                    btn.textContent = '🪞 Mirrored';
                }}
                if (!originalHemispherePositions) cacheHemispherePositions();
                runHemisphereMirrorLayout();
                resetLayoutTransformTrackers();
                updateHoverInfo(`🪞 Hemisphere-mirrored ${{baseLayout}} layout applied`);
                return;
            }}
            
            // Get visible (non-hidden) nodes and edges (class-based - see
            // isVisibleElement)
            const visibleElements = cy.elements().filter(isVisibleElement);
            
            if (visibleElements.length === 0) {{
                updateHoverInfo('⚠️ No visible elements to layout');
                return;
            }}
            
            // Apply new layout with animation to visible elements only
            const config = getLayoutConfig(newLayout);
            config.animate = true;
            config.animationDuration = 500;
            // Animated layouts interpolate positions asynchronously — defer
            // the tracker reset to layoutstop so the gap spinners and the
            // Reset-Spacing baseline describe the NEW arrangement.
            cy.one('layoutstop', () => {{ resetLayoutTransformTrackers(); }});
            visibleElements.layout(config).run();
            
            updateHoverInfo(`🔄 Layout changed to ${{newLayout}}`);
            console.log(`Layout changed to: ${{newLayout}}`);
        }}

        function fitGraph() {{
            // Fit with extra padding to account for control panel height;
            // only VISIBLE elements participate (hidden nodes still have
            // positions and would otherwise inflate the bounding box).
            const visible = cy.elements().filter(isVisibleElement);
            if (visible.length > 0) cy.fit(visible, 80);
        }}

        // ===== LAYOUT TRANSFORMS: inter-node gap + rotation =====
        // Both transforms are pure geometry on the CURRENT node positions,
        // anchored on the visible CENTROID (mean position). The centroid is
        // preserved by scaling-around-itself and by rotation, so gap and
        // rotation compose without drift. Hidden nodes are excluded (same
        // isVisibleElement rule as the layout re-runs); node size, labels
        // and edge widths never change.

        // Mean position of the visible nodes — the transform anchor.
        function visibleNodeCentroid() {{
            const visNodes = cy.nodes().filter(isVisibleElement);
            if (visNodes.length === 0) return null;
            let sx = 0, sy = 0;
            visNodes.forEach(n => {{
                sx += n.position().x;
                sy += n.position().y;
            }});
            return {{ x: sx / visNodes.length, y: sy / visNodes.length }};
        }}

        // Which pair of LAYOUT axes currently reads as screen H/V: beyond
        // 45° of accumulated rotation the layout's vertical axis is the more
        // horizontal one on screen. Same parity as the tracker swap on exact
        // quarter turns (swapGapAxesIfQuarterTurn).
        function gapAxesSwapped() {{
            return Math.round(lastRotationDeg / 90) % 2 !== 0;
        }}

        // Median successive difference of the visible coordinates on one
        // axis of the LAYOUT FRAME — the coordinates unrotated by the
        // accumulated rotation, i.e. the center-to-center gap between the
        // layout's own neighboring columns/rows. Measuring there (instead
        // of raw screen coordinates) keeps the gap meaningful at ANY angle:
        // a 5° tilt no longer collapses the median diff (the misaligned
        // columns produce tiny row-offset diffs), so the spinners only
        // change when the geometry really does.
        function measureAxisGap(axis) {{
            const center = visibleNodeCentroid();
            const a = lastRotationDeg * Math.PI / 180;
            const c = Math.cos(a), s = Math.sin(a);
            const coords = [];
            cy.nodes().filter(isVisibleElement).forEach(n => {{
                const p = n.position();
                const dx = p.x - center.x, dy = p.y - center.y;
                // offset → layout frame: rotate by -lastRotationDeg
                coords.push(axis === 'x' ? dx * c + dy * s : -dx * s + dy * c);
            }});
            if (coords.length < 2) return 0;
            coords.sort((a, b) => a - b);
            const diffs = [];
            for (let i = 1; i < coords.length; i++) {{
                const d = coords[i] - coords[i - 1];
                if (d > 0.01) diffs.push(d);
            }}
            if (diffs.length === 0) return 0;
            diffs.sort((a, b) => a - b);
            return diffs[Math.floor(diffs.length / 2)];
        }}

        // Set the inter-node gap on one SCREEN axis to an ABSOLUTE
        // center-to-center distance (px). The scale is applied in the
        // LAYOUT frame (unrotate → scale the mapped axis → rotate back), so
        // a tilted arrangement is stretched along its own columns/rows and
        // never sheared — rotating back to 0° restores the axis-aligned
        // layout exactly. Node sizes are irrelevant — only positions
        // participate.
        function applyNodeGap(screenAxis, targetPx) {{
            const target = Number(targetPx);
            if (!isFinite(target) || target <= 0) return false;
            const swapped = gapAxesSwapped();
            const layoutAxis = ((screenAxis === 'x') !== swapped) ? 'x' : 'y';
            const current = measureAxisGap(layoutAxis);
            if (current <= 0.01) return false;  // single row/column
            const factor = target / current;
            const anchor = visibleNodeCentroid();
            const c = Math.cos(lastRotationDeg * Math.PI / 180);
            const s = Math.sin(lastRotationDeg * Math.PI / 180);
            cy.batch(() => {{
                cy.nodes().filter(isVisibleElement).forEach(n => {{
                    const p = n.position();
                    const dx = p.x - anchor.x, dy = p.y - anchor.y;
                    // offset → layout frame (rotate by -lastRotationDeg)
                    const lx = dx * c + dy * s;
                    const ly = -dx * s + dy * c;
                    // scale the mapped axis, then back to the screen frame
                    const rx = layoutAxis === 'x' ? lx * factor : lx;
                    const ry = layoutAxis === 'y' ? ly * factor : ly;
                    n.position({{
                        x: anchor.x + rx * c - ry * s,
                        y: anchor.y + rx * s + ry * c
                    }});
                }});
            }});
            if (screenAxis === 'x') lastGapX = target; else lastGapY = target;
            return true;
        }}

        // Write the normalized tracker into the Rotate field unless the user
        // is editing it right now.
        function syncRotateDisplay() {{
            const rs = document.getElementById('rotateSlider');
            if (rs && document.activeElement !== rs) rs.value = Math.round(lastRotationDeg);
        }}

        // Write the gap trackers into the H/V spinners (no re-measurement).
        function syncGapDisplays() {{
            const gh = document.getElementById('nodeGapHSlider');
            const gv = document.getElementById('nodeGapVSlider');
            if (gh && lastGapX) gh.value = Math.round(lastGapX * 10) / 10;
            if (gv && lastGapY) gv.value = Math.round(lastGapY * 10) / 10;
        }}

        // An exact ±90° (or ±270°) rotation maps horizontal neighbor columns
        // onto vertical rows and vice versa, so the H/V gap trackers must
        // swap to keep describing the same node pairs (the swap is exact:
        // the rotated x-coordinates ARE the unrotated y-coordinates).
        // Arbitrary angles re-measure on commit instead
        // (onRotationChange → syncTransformInputs).
        function swapGapAxesIfQuarterTurn(deltaDeg) {{
            const norm = ((deltaDeg % 180) + 180) % 180;
            if (Math.abs(norm - 90) > 1e-6) return;
            const tmp = lastGapX; lastGapX = lastGapY; lastGapY = tmp;
            syncGapDisplays();
        }}

        function applyRotationDelta(targetDeg) {{
            const raw = Number(targetDeg);
            if (!isFinite(raw)) return false;
            // Angles wrap mod 360: the applied delta stays exact (± full
            // turns are geometrically identical), while the tracker and the
            // Rotate field are kept in [0, 360) so repeated ↺ clicks read
            // 270 → 180 → 90 → 0 instead of drifting to -1440.
            const deltaDeg = raw - lastRotationDeg;
            const delta = deltaDeg * Math.PI / 180;
            if (Math.abs(delta % (2 * Math.PI)) < 1e-9) {{
                lastRotationDeg = ((raw % 360) + 360) % 360;
                syncRotateDisplay();
                return false;
            }}
            const center = visibleNodeCentroid();
            if (!center) return false;
            const cos = Math.cos(delta), sin = Math.sin(delta);
            cy.batch(() => {{
                cy.nodes().filter(isVisibleElement).forEach(n => {{
                    const p = n.position();
                    const dx = p.x - center.x, dy = p.y - center.y;
                    n.position({{
                        x: center.x + dx * cos - dy * sin,
                        y: center.y + dx * sin + dy * cos
                    }});
                }});
            }});
            lastRotationDeg = ((raw % 360) + 360) % 360;
            // Keep the H/V gap trackers describing the same neighbors: they
            // swap under every quarter turn (including via the ↺ button).
            swapGapAxesIfQuarterTurn(deltaDeg);
            syncRotateDisplay();
            return true;
        }}

        // Sync the gap/rotation spinners: the gaps are measured in the
        // LAYOUT frame (rotation-invariant) and mapped onto the SCREEN H/V
        // axes — swapped beyond 45° of rotation, consistent with the
        // quarter-turn tracker swap.
        function syncTransformInputs() {{
            const lx = measureAxisGap('x');
            const ly = measureAxisGap('y');
            const swapped = gapAxesSwapped();
            const sx = swapped ? ly : lx;
            const sy = swapped ? lx : ly;
            if (sx > 0) lastGapX = sx;
            if (sy > 0) lastGapY = sy;
            syncGapDisplays();
            const rs = document.getElementById('rotateSlider');
            if (rs) rs.value = Math.round(lastRotationDeg);
        }}

        // Slider live preview ('input'): capture the pre-transform state once
        // per drag, then apply as the value changes. A change that did not
        // alter geometry (e.g. a 0 or empty value) commits no history.
        function onSpacingInput(axis, value) {{
            if (!pendingTransformState) pendingTransformState = captureState();
            spacingDirtySinceInput = applyNodeGap(axis, parseFloat(value)) || spacingDirtySinceInput;
        }}

        function onRotationInput(value) {{
            if (!pendingTransformState) pendingTransformState = captureState();
            rotationDirtySinceInput = applyRotationDelta(parseFloat(value)) || rotationDirtySinceInput;
        }}

        // Slider release ('change'): commit ONE history entry with the
        // pre-drag snapshot (same contract as the drag history).
        function onSpacingChange() {{
            if (pendingTransformState && spacingDirtySinceInput) {{
                pushStateHistory('Adjust node spacing', pendingTransformState);
                pendingTransformState = null;
            }}
            spacingDirtySinceInput = false;
        }}

        function onRotationChange() {{
            // Same no-op contract as spacing: an edit that did not alter
            // geometry (full-turn wrap, same angle retyped) commits no
            // history.
            if (pendingTransformState && rotationDirtySinceInput) {{
                pushStateHistory('Rotate layout', pendingTransformState);
                pendingTransformState = null;
            }}
            rotationDirtySinceInput = false;
            // Normalize the field display once editing ends (e.g. 750 → 30).
            syncTransformInputs();
        }}

        // Counter-clockwise 90° per click: the angle wraps mod 360, so the
        // Rotate field reads 270 → 180 → 90 → 0 as clicks accumulate; the
        // Rotate field accepts any typed angle as well.
        function rotateCounterClockwise() {{
            pushHistory('Rotate layout');
            applyRotationDelta(lastRotationDeg - 90);
        }}

        // Restore the inter-node gaps captured at the last layout run.
        function resetSpacing() {{
            if (!baselineGapX && !baselineGapY) return;
            pushHistory('Reset spacing');
            // The baselines are layout-frame values captured right after a
            // layout run (rotation 0): map them through the CURRENT screen
            // axis orientation so the layout's own gaps come back exactly.
            const swapped = gapAxesSwapped();
            const targetX = swapped ? baselineGapY : baselineGapX;
            const targetY = swapped ? baselineGapX : baselineGapY;
            if (targetX) applyNodeGap('x', targetX);
            if (targetY) applyNodeGap('y', targetY);
            syncTransformInputs();
        }}

        // Called after a layout (re)run: the algorithm regenerated every
        // position, so the rotation tracker restarts at 0 and the current
        // gaps become the new Reset-Spacing baseline. Only trackers and
        // spinners are touched — NO transform is applied.
        function resetLayoutTransformTrackers() {{
            lastRotationDeg = 0;
            pendingTransformState = null;
            // A fresh layout invalidates any in-flight transform drag:
            // the next change commits only ITS OWN geometry delta.
            spacingDirtySinceInput = false;
            rotationDirtySinceInput = false;
            const rs = document.getElementById('rotateSlider');
            if (rs) rs.value = 0;
            baselineGapX = measureAxisGap('x') || baselineGapX;
            baselineGapY = measureAxisGap('y') || baselineGapY;
            lastGapX = baselineGapX;
            lastGapY = baselineGapY;
            syncTransformInputs();
        }}

        // ===== PANEL COLLAPSE (top tools + right palette) =====
        const PANEL_STATE_KEY = 'vispath_network_panels';

        function applyTopControlsCollapsed(collapsed) {{
            const controls = document.querySelector('.controls');
            const btn = document.getElementById('toggleControlsBtn');
            if (controls) controls.classList.toggle('collapsed', collapsed);
            document.body.classList.toggle('controls-collapsed', collapsed);
            if (btn) btn.textContent = collapsed ? '⚙️ Show Ribbon' : '⚙️ Hide Ribbon';
        }}

        function toggleTopControls() {{
            const controls = document.querySelector('.controls');
            const collapsed = controls ? !controls.classList.contains('collapsed') : false;
            applyTopControlsCollapsed(collapsed);
            persistPanelState();
            updateHoverInfo(collapsed ? '⚙️ Top panel hidden' : '⚙️ Top panel shown');
            // Cytoscape must re-measure the container or clicks drift after
            // the canvas size changes.
            if (typeof cy !== 'undefined') cy.resize();
        }}

        function applyPaletteHidden(hidden) {{
            const palette = document.getElementById('colorPalette');
            const main = document.querySelector('.main');
            const btn = document.getElementById('togglePanelBtn');
            if (palette) palette.style.display = hidden ? 'none' : '';
            if (main) main.classList.toggle('palette-hidden', hidden);
            // Frees the panel bar's palette clearance so search/help slide right
            document.body.classList.toggle('palette-hidden', hidden);
            if (btn) btn.textContent = hidden ? '🎨 Show Panel' : '🎨 Hide Panel';
        }}

        function toggleRightPanel() {{
            const palette = document.getElementById('colorPalette');
            const hidden = palette ? palette.style.display !== 'none' : false;
            applyPaletteHidden(hidden);
            persistPanelState();
            updateHoverInfo(hidden ? '🎨 Right panel hidden' : '🎨 Right panel shown');
            if (typeof cy !== 'undefined') cy.resize();
        }}

        // Panel collapse is a user preference shared by every network file
        // (unlike layouts, which are per-file), so a global key is used.
        function persistPanelState() {{
            const controls = document.querySelector('.controls');
            const palette = document.getElementById('colorPalette');
            saveObjectToStorage(PANEL_STATE_KEY, {{
                controlsCollapsed: controls ? controls.classList.contains('collapsed') : false,
                paletteHidden: palette ? palette.style.display === 'none' : false,
                activeTab: activeTabName,
                ribbonCollapsed: controls ? controls.classList.contains('collapsed') : false,
                accordions: {{
                    edit: document.getElementById('accEdit').classList.contains('open'),
                    history: document.getElementById('accHistory').classList.contains('open'),
                    selection: document.getElementById('accSelection').classList.contains('open')
                }}
            }});
        }}

        // ===== RIBBON TABS (PPT-style): tool card rows per tab =====
        const RIBBON_PAGES = {{ layout: 'pageLayout', filter: 'pageFilter', share: 'pageShare' }};
        let activeTabName = 'layout';

        function switchTab(name) {{
            const page = document.getElementById(RIBBON_PAGES[name]);
            const ribbon = document.getElementById('ribbon');
            if (!page || !ribbon) return;
            if (page.classList.contains('open')) {{
                // Unified toggle (PPT behavior): the first click on the ACTIVE
                // tab collapses the ribbon, the next click on it expands it
                // again. Routing through the same helper as the ⚙️ button
                // keeps canvas height, toggle label and persistence in sync.
                applyTopControlsCollapsed(!ribbon.classList.contains('collapsed'));
                persistPanelState();
                if (typeof cy !== 'undefined') cy.resize();
                return;
            }}
            Object.keys(RIBBON_PAGES).forEach(key => {{
                const p = document.getElementById(RIBBON_PAGES[key]);
                if (p) p.classList.toggle('open', key === name);
                const tab = document.getElementById('tab' + key.charAt(0).toUpperCase() + key.slice(1));
                if (tab) tab.classList.toggle('active', key === name);
            }});
            activeTabName = name;
            // Expanding via a different tab must also re-sync the ⚙️ toggle
            // label and the canvas height class (same helper, one path).
            applyTopControlsCollapsed(false);
            persistPanelState();
            if (typeof cy !== 'undefined') cy.resize();
        }}

        // Restore the last panel collapse state AFTER cy exists so the
        // follow-up resize() is meaningful.
        function initPanelBar() {{
            const saved = loadObjectFromStorage(PANEL_STATE_KEY);
            if (!saved) return;
            if (saved.controlsCollapsed) applyTopControlsCollapsed(true);
            if (saved.paletteHidden) applyPaletteHidden(true);
            if (saved.activeTab && RIBBON_PAGES[saved.activeTab]) {{
                // Restore directly (NOT via switchTab: for the default
                // 'layout' tab the static markup is already open, and
                // switchTab would read that as 'collapse the ribbon').
                activeTabName = saved.activeTab;
                Object.keys(RIBBON_PAGES).forEach(key => {{
                    const p = document.getElementById(RIBBON_PAGES[key]);
                    if (p) p.classList.toggle('open', key === saved.activeTab);
                    const tab = document.getElementById('tab' + key.charAt(0).toUpperCase() + key.slice(1));
                    if (tab) tab.classList.toggle('active', key === saved.activeTab);
                }});
            }}
            if (saved.ribbonCollapsed) document.getElementById('ribbon').classList.add('collapsed');
            if (saved.accordions) {{
                ['edit', 'history', 'selection'].forEach(function(name) {{
                    const section = document.getElementById('acc' + name.charAt(0).toUpperCase() + name.slice(1));
                    if (section) section.classList.toggle('open', !!saved.accordions[name]);
                }});
            }}
            if (saved.controlsCollapsed || saved.paletteHidden) cy.resize();
        }}

        // Accordion sections (right panel). Open flags persist with the
        // other panel preferences.
        function toggleAccordion(name) {{
            const section = document.getElementById('acc' + name.charAt(0).toUpperCase() + name.slice(1));
            if (section) section.classList.toggle('open');
            persistPanelState();
        }}

        // ===== FEEDBACK: toast stack (operation feedback) =====
        // Pure queue logic (testable headless): keeps at most `maxVisible`
        // toasts, entries expire after ttl ms (errors live longer). The DOM
        // layer in showToast/renderToasts renders whatever is active.
        function createToastQueue(options) {{
            const opts = options || {{}};
            const maxVisible = opts.maxVisible || 4;
            const ttl = opts.ttl || 3000;
            const errorTtl = opts.errorTtl || 6000;
            let items = [];  // {{ id, message, type, action, born, ttl }}
            let nextId = 1;
            return {{
                push(message, type, action, now) {{
                    const t = (typeof now === 'number') ? now : Date.now();
                    const isError = type === 'error';
                    items.push({{
                        id: nextId++,
                        message: String(message),
                        type: type || 'info',
                        action: action || null,
                        born: t,
                        ttl: isError ? errorTtl : ttl
                    }});
                    while (items.length > maxVisible) items.shift();
                    return items.slice();
                }},
                expire(now) {{
                    const t = (typeof now === 'number') ? now : Date.now();
                    items = items.filter(item => t - item.born < item.ttl);
                    return items.slice();
                }},
                remove(id) {{
                    items = items.filter(item => item.id !== id);
                    return items.slice();
                }},
                active() {{ return items.slice(); }},
            }};
        }}

        const toastQueue = createToastQueue();

        // Expire + render the active toasts into #toastStack.
        function renderToasts() {{
            toastQueue.expire();
            const stack = document.getElementById('toastStack');
            if (!stack) return;
            stack.innerHTML = '';
            toastQueue.active().forEach(item => {{
                const el = document.createElement('div');
                el.className = 'vp-toast vp-toast-' + item.type;
                el.dataset.toastId = item.id;
                el.textContent = item.message;
                if (item.action) {{
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'vp-toast-action';
                    btn.textContent = item.action.label;
                    btn.addEventListener('click', function() {{
                        item.action.onClick();
                        toastQueue.remove(item.id);
                        renderToasts();
                    }});
                    el.appendChild(btn);
                }}
                stack.appendChild(el);
            }});
        }}

        // Single entry point for operation feedback. Optional action:
        // {{ label, onClick }} renders an inline button on the toast.
        function showToast(message, type, action) {{
            toastQueue.push(message, type, action);
            renderToasts();
            setTimeout(renderToasts, type === 'error' ? 6000 : 3000);
        }}

        // ===== IN-PAGE DIALOGS (replace native prompt/confirm) =====
        // Pure promise state machine (testable headless): one dialog at a
        // time; field values arrive via set(); confirm() resolves with the
        // values, cancel() rejects with 'cancelled'.
        function createDialogController() {{
            let active = null;  // {{ spec, values, resolve, reject }}
            return {{
                isActive() {{ return active !== null; }},
                open(spec) {{
                    if (active) return null;
                    const initial = {{}};
                    (spec.fields || []).forEach(f => {{ initial[f.key] = f.value; }});
                    active = {{ spec: spec, values: initial, resolve: null, reject: null }};
                    return new Promise((resolve, reject) => {{
                        active.resolve = resolve;
                        active.reject = reject;
                    }});
                }},
                set(key, value) {{
                    if (active) active.values[key] = value;
                }},
                confirm() {{
                    if (!active) return null;
                    const done = active;
                    active = null;
                    done.resolve(done.values);
                    return done.values;
                }},
                cancel() {{
                    if (!active) return null;
                    const done = active;
                    active = null;
                    done.reject('cancelled');
                    return null;
                }},
            }};
        }}

        const dialogCtl = createDialogController();
        let lastDialogFocus = null;

        // Build the dialog DOM for one open() call and wire the buttons.
        // spec: {{ title, message?, fields?: [{{key,label,type,value,options?,placeholder?}}],
        //         confirmText?, danger? }}
        function showDialog(spec) {{
            const promise = dialogCtl.open(spec);
            if (promise === null) return Promise.reject('busy');
            lastDialogFocus = document.activeElement;
            const title = document.getElementById('vpDialogTitle');
            const msg = document.getElementById('vpDialogMessage');
            const body = document.getElementById('vpDialogBody');
            const confirmBtn = document.getElementById('vpDialogConfirm');
            title.textContent = spec.title || '';
            if (spec.message) {{
                msg.textContent = spec.message;
                msg.style.display = 'block';
            }} else {{
                msg.style.display = 'none';
            }}
            body.innerHTML = '';
            (spec.fields || []).forEach(field => {{
                const wrap = document.createElement('div');
                wrap.className = 'vp-dialog-field';
                const label = document.createElement('label');
                label.textContent = field.label || field.key;
                wrap.appendChild(label);
                let input;
                if (field.type === 'select') {{
                    input = document.createElement('select');
                    (field.options || []).forEach(opt => {{
                        const option = document.createElement('option');
                        option.value = opt;
                        option.textContent = opt;
                        input.appendChild(option);
                    }});
                    input.value = field.value;
                }} else {{
                    input = document.createElement('input');
                    input.type = field.type || 'text';
                    if (field.value !== undefined && field.value !== null) input.value = field.value;
                    if (field.placeholder) input.placeholder = field.placeholder;
                }}
                input.dataset.key = field.key;
                input.id = 'vpField_' + field.key;
                input.setAttribute('aria-label', field.label || field.key);
                input.addEventListener('input', function() {{ dialogCtl.set(field.key, input.value); }});
                input.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter') {{ e.preventDefault(); confirmDialog(); }}
                    if (e.key === 'Escape') {{ e.preventDefault(); cancelDialog(); }}
                }});
                wrap.appendChild(input);
                body.appendChild(wrap);
            }});
            confirmBtn.textContent = spec.confirmText || 'OK';
            confirmBtn.className = 'vp-dialog-btn ' + (spec.danger ? 'vp-danger' : 'vp-primary');
            document.getElementById('vpDialogOverlay').classList.add('open');
            const first = body.querySelector('input, select');
            if (first) first.focus();
            return promise;
        }}

        function closeDialogOverlay() {{
            document.getElementById('vpDialogOverlay').classList.remove('open');
            if (lastDialogFocus && lastDialogFocus.focus) lastDialogFocus.focus();
            lastDialogFocus = null;
        }}

        function confirmDialog() {{
            if (!dialogCtl.isActive()) return;
            // Harvest the final field values from the DOM before resolving.
            document.querySelectorAll('#vpDialogBody input, #vpDialogBody select').forEach(input => {{
                dialogCtl.set(input.dataset.key, input.value);
            }});
            dialogCtl.confirm();
            closeDialogOverlay();
        }}

        function cancelDialog() {{
            if (!dialogCtl.isActive()) return;
            dialogCtl.cancel();
            closeDialogOverlay();
        }}

        // ===== CANVAS DRAG MODE: pan (default) vs box-select =====
        // Cytoscape pans on plain drag and box-selects on Shift+drag. In
        // select mode user panning is switched OFF while box selection stays
        // on — cytoscape treats a non-pannable canvas as box-select-on-drag,
        // so a plain drag draws the selection box. The ✋ Pan button restores
        // dragging-as-panning.
        let canvasDragMode = 'pan';

        function setCanvasDragMode(mode) {{
            if (mode !== 'pan' && mode !== 'select') return;
            if (typeof cy === 'undefined') return;
            canvasDragMode = mode;
            cy.userPanningEnabled(mode === 'pan');
            cy.boxSelectionEnabled(true);
            // The background-press circle is the pan affordance: visible at
            // half the default radius in pan mode, hidden in select mode so
            // a selection drag never flashes a pan symbol.
            cy.style().selector('core').style({{
                'active-bg-size': 15,
                'active-bg-opacity': mode === 'pan' ? 0.15 : 0
            }}).update();
            const panBtn = document.getElementById('panModeBtn');
            const selBtn = document.getElementById('selectModeBtn');
            if (panBtn) {{
                panBtn.classList.toggle('active', mode === 'pan');
                panBtn.setAttribute('aria-pressed', String(mode === 'pan'));
            }}
            if (selBtn) {{
                selBtn.classList.toggle('active', mode === 'select');
                selBtn.setAttribute('aria-pressed', String(mode === 'select'));
            }}
            updateHoverInfo(mode === 'pan'
                ? '✋ Pan mode: drag the canvas to pan (⇧+drag still box-selects)'
                : '□ Select mode: drag the canvas to box-select (✋ Pan restores panning)');
        }}

        // ===== DIRECTIONAL BOX SELECTION: L→R full / R→L touch =====
        // Design-tool conventions: a LEFT→RIGHT frame selects only elements
        // FULLY inside it (solid frame); a RIGHT→LEFT frame selects anything
        // the frame TOUCHES (dashed frame). The gesture stays cytoscape's
        // (boxstart/tapdrag/boxend) but its built-in frame is hidden via
        // selection-box-opacity and its selection result is re-derived here.
        let boxStartPos = null;        // model-coords anchor of the frame
        let boxStartRendered = null;   // rendered anchor for overlay drawing
        let boxStartSelection = null;  // pre-gesture selection (⇧ additive)
        let boxAdditive = false;

        function boxNormalized(x1, y1, x2, y2) {{
            return {{ x1: Math.min(x1, x2), y1: Math.min(y1, y2), x2: Math.max(x1, x2), y2: Math.max(y1, y2) }};
        }}

        function pointInRect(px, py, r) {{
            return px >= r.x1 && px <= r.x2 && py >= r.y1 && py <= r.y2;
        }}

        // Liang–Barsky clip: does the segment cross the rect at all?
        function segmentIntersectsRect(x1, y1, x2, y2, r) {{
            let t0 = 0, t1 = 1;
            const dx = x2 - x1, dy = y2 - y1;
            const p = [-dx, dx, -dy, dy];
            const q = [x1 - r.x1, r.x2 - x1, y1 - r.y1, r.y2 - y1];
            for (let i = 0; i < 4; i++) {{
                if (p[i] === 0) {{
                    if (q[i] < 0) return false;
                }} else {{
                    const t = q[i] / p[i];
                    if (p[i] < 0) {{ if (t > t1) return false; if (t > t0) t0 = t; }}
                    else {{ if (t < t0) return false; if (t < t1) t1 = t; }}
                }}
            }}
            return true;
        }}

        function nodeRect(node) {{
            const p = node.position();
            const hw = node.width() / 2, hh = node.height() / 2;
            return {{ x1: p.x - hw, y1: p.y - hh, x2: p.x + hw, y2: p.y + hh }};
        }}

        function nodeFullyInRect(node, r) {{
            const nr = nodeRect(node);
            return nr.x1 >= r.x1 && nr.x2 <= r.x2 && nr.y1 >= r.y1 && nr.y2 <= r.y2;
        }}

        function nodeTouchesRect(node, r) {{
            const nr = nodeRect(node);
            return nr.x1 <= r.x2 && nr.x2 >= r.x1 && nr.y1 <= r.y2 && nr.y2 >= r.y1;
        }}

        // Endpoints + curve midpoint: exact for straight edges, a good
        // approximation for the curved reciprocal edges. Headless cores
        // (tests) have no renderer midpoint — the straight-line midpoint
        // is the exact answer there anyway.
        function edgeSamplePoints(edge) {{
            const sp = edge.source().position(), tp = edge.target().position();
            const pts = [[sp.x, sp.y], [tp.x, tp.y]];
            try {{
                const mid = edge.midpoint();
                pts.push([mid.x, mid.y]);
            }} catch (err) {{
                pts.push([(sp.x + tp.x) / 2, (sp.y + tp.y) / 2]);
            }}
            return pts;
        }}

        function edgeFullyInRect(edge, r) {{
            return edgeSamplePoints(edge).every(p => pointInRect(p[0], p[1], r));
        }}

        function edgeTouchesRect(edge, r) {{
            if (edgeSamplePoints(edge).some(p => pointInRect(p[0], p[1], r))) return true;
            const sp = edge.source().position(), tp = edge.target().position();
            return segmentIntersectsRect(sp.x, sp.y, tp.x, tp.y, r);
        }}

        // Re-derive the box selection: full containment for L→R frames,
        // touch for R→L frames — combined select/deselect semantics:
        //   plain drag  → if the frame covers ONLY already-selected
        //                 elements it DESELECTS them from the selection;
        //                 otherwise it replaces the selection with the
        //                 frame contents.
        //   ⇧ drag      → per-element TOGGLE: covered selected elements
        //                 drop out, covered unselected ones join; the
        //                 outside selection is kept.
        // An empty frame is a no-op. `baseSelection` is the PRE-gesture
        // selection — the live state is already tainted by cytoscape's
        // built-in box select when this runs (deferred after boxend).
        function applyDirectionalBoxSelection(x1, y1, x2, y2, additive, baseSelection) {{
            const rect = boxNormalized(x1, y1, x2, y2);
            const full = x2 >= x1;  // left-to-right = full containment
            let inFrame = cy.collection();
            cy.nodes().filter(isVisibleElement).forEach(n => {{
                if (full ? nodeFullyInRect(n, rect) : nodeTouchesRect(n, rect)) inFrame = inFrame.union(n);
            }});
            cy.edges().filter(isVisibleElement).forEach(e => {{
                if (full ? edgeFullyInRect(e, rect) : edgeTouchesRect(e, rect)) inFrame = inFrame.union(e);
            }});
            const base = baseSelection || cy.collection();
            let result;
            if (inFrame.empty()) {{
                result = base;  // caught nothing — leave the selection alone
            }} else if (additive) {{
                result = base.difference(inFrame).union(inFrame.difference(base));
            }} else if (inFrame.difference(base).empty()) {{
                // everything the frame covers was already selected → remove
                result = base.difference(inFrame);
            }} else {{
                result = inFrame;  // plain select: replace with the frame contents
            }}
            cy.batch(() => {{
                cy.elements().unselect();
                result.select();
            }});
            return result;
        }}

        function drawBoxOverlay(rp) {{
            const o = document.getElementById('boxOverlay');
            if (!o || !boxStartRendered) return;
            o.style.left = Math.min(boxStartRendered.x, rp.x) + 'px';
            o.style.top = Math.min(boxStartRendered.y, rp.y) + 'px';
            o.style.width = Math.abs(rp.x - boxStartRendered.x) + 'px';
            o.style.height = Math.abs(rp.y - boxStartRendered.y) + 'px';
            // touch mode (right-to-left) shows the dashed frame — with a
            // small dead-zone so the border style doesn't flicker while the
            // drag direction is still ambiguous at the anchor
            const dx = rp.x - boxStartRendered.x;
            if (Math.abs(dx) > 4) o.classList.toggle('vp-touch', dx < 0);
        }}

        function onBoxFrameStart(e) {{
            boxStartPos = e.position;
            boxStartRendered = e.renderedPosition ||
                {{ x: e.position.x * cy.zoom() + cy.pan().x, y: e.position.y * cy.zoom() + cy.pan().y }};
            boxAdditive = !!(e.originalEvent &&
                (e.originalEvent.shiftKey || e.originalEvent.metaKey || e.originalEvent.ctrlKey));
            boxStartSelection = cy.elements().filter(el => el.selected());
            const o = document.getElementById('boxOverlay');
            if (o) {{
                // zero-size at the anchor: showing last gesture's leftover
                // geometry for a frame reads as a flash at the drag start
                o.classList.remove('vp-touch');
                o.style.display = 'block';
                o.style.left = boxStartRendered.x + 'px';
                o.style.top = boxStartRendered.y + 'px';
                o.style.width = '0px';
                o.style.height = '0px';
            }}
        }}

        function onBoxFrameDrag(e) {{
            if (boxStartPos && e.renderedPosition) drawBoxOverlay(e.renderedPosition);
        }}

        function onBoxFrameEnd(e) {{
            if (!boxStartPos) return;
            const endPos = e.position || boxStartPos;
            const x1 = boxStartPos.x, y1 = boxStartPos.y;
            const additive = boxAdditive, baseSelection = boxStartSelection;
            boxStartPos = null; boxStartRendered = null;
            const o = document.getElementById('boxOverlay');
            if (o) o.style.display = 'none';
            // cytoscape applies its built-in box selection right AFTER
            // emitting boxend — defer past it, then override with the
            // directional result.
            setTimeout(() => {{
                applyDirectionalBoxSelection(x1, y1, endPos.x, endPos.y, additive, baseSelection);
                boxStartSelection = null; boxAdditive = false;
            }}, 0);
        }}

        cy.on('boxstart', onBoxFrameStart);
        cy.on('tapdrag', onBoxFrameDrag);
        cy.on('boxend', onBoxFrameEnd);

        // ===== COMMAND STRIP: node search =====
        let searchMatches = [];
        let searchIndex = -1;

        // Pure matcher: nodes whose id or label contains the query
        // (case-insensitive, minimum 2 characters).
        function matchNodes(query) {{
            const q = String(query || '').trim().toLowerCase();
            if (q.length < 2) return [];
            const hits = [];
            cy.nodes().forEach(n => {{
                const id = n.id();
                const label = String(n.data('label') || id);
                if (id.toLowerCase().includes(q) || label.toLowerCase().includes(q)) hits.push(n);
            }});
            return hits;
        }}

        function onSearchInput(value) {{
            searchMatches = matchNodes(value);
            searchIndex = searchMatches.length > 0 ? 0 : -1;
            updateSearchCount();
        }}

        function updateSearchCount() {{
            const el = document.getElementById('nodeSearchCount');
            if (!el) return;
            el.textContent = searchMatches.length === 0 ? '' : (searchIndex + 1) + '/' + searchMatches.length;
        }}

        function applySearchMatch() {{
            if (searchIndex < 0 || searchIndex >= searchMatches.length) return;
            const node = searchMatches[searchIndex];
            cy.elements().unselect();
            node.select();
            cy.animate({{ center: {{ eles: node }} }}, {{ duration: 250 }});
            if (!isVisibleElement(node)) showToast('Matched node is hidden — use 👁️ Show All to reveal it', 'warn');
        }}

        // Step to the previous/next match (dir = -1/+1) and select+center it
        // — shared by the ▲/▼ buttons and the ↑/↓ keys, so every navigation
        // route moves the view, not just the 1/N counter.
        function cycleSearchMatch(dir) {{
            if (!searchMatches.length) return;
            searchIndex = (searchIndex + dir + searchMatches.length) % searchMatches.length;
            updateSearchCount();
            applySearchMatch();
        }}

        function onSearchKeydown(event) {{
            if (event.key === 'Enter') {{
                event.preventDefault();
                applySearchMatch();
            }} else if (event.key === 'ArrowDown') {{
                event.preventDefault();
                cycleSearchMatch(1);
            }} else if (event.key === 'ArrowUp') {{
                event.preventDefault();
                cycleSearchMatch(-1);
            }} else if (event.key === 'Escape') {{
                event.preventDefault();
                event.target.value = '';
                searchMatches = [];
                searchIndex = -1;
                updateSearchCount();
                event.target.blur();
            }}
        }}

        // ===== HELP OVERLAY (? key / ❓ button) =====
        function openHelp(section) {{
            document.getElementById('helpOverlay').classList.add('open');
            if (section) {{
                const target = document.getElementById('help' + section.charAt(0).toUpperCase() + section.slice(1));
                if (target && target.scrollIntoView) target.scrollIntoView({{ block: 'start' }});
            }}
        }}

        function closeHelp() {{
            document.getElementById('helpOverlay').classList.remove('open');
        }}

        function toggleHelp() {{
            const overlay = document.getElementById('helpOverlay');
            if (overlay.classList.contains('open')) closeHelp(); else openHelp();
        }}

        // Global Esc/? handling for dialogs and help; runs in capture phase
        // before the single-letter shortcuts (which also guard on them).
        document.addEventListener('keydown', function(e) {{
            if (dialogCtl.isActive()) {{
                if (e.key === 'Escape') {{ e.preventDefault(); cancelDialog(); }}
                return;
            }}
            const helpOpen = document.getElementById('helpOverlay').classList.contains('open');
            if (helpOpen) {{
                if (e.key === 'Escape') {{ e.preventDefault(); closeHelp(); }}
                return;
            }}
            const tag = e.target && e.target.tagName;
            const typing = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable);
            if (!typing && (e.key === '?' || (e.shiftKey && e.key === '/'))) {{
                e.preventDefault();
                toggleHelp();
            }}
        }}, true);

        // Export functions (shared backend). Large export scales are
        // confirmed with an in-page dialog (async), not window.confirm.
        function resolveExportScale() {{
            const el = document.getElementById('exportScale');
            let scale = el ? parseFloat(el.value) : NaN;
            if (isNaN(scale) || scale < 1) scale = 2;
            if (scale <= 4) return Promise.resolve(scale);
            return showDialog({{
                title: 'Large export',
                message: 'Exporting at ' + scale + 'x may fail in your browser (very large image). Export at the requested scale or fall back to a safer 4x?',
                fields: [],
                confirmText: 'Export at ' + scale + 'x',
                danger: true
            }}).then(() => scale).catch(() => 4);
        }}

        async function exportPNG() {{
            const scale = await resolveExportScale();
            exportCytoscapeToImage(cy, 'png', 'network_selected_paths_' + scale + 'x.png', scale, bgCtrl.getColor());
        }}
        
        function exportSVG() {{
            exportCytoscapeToImage(cy, 'svg', 'network_selected_paths.svg', 1, bgCtrl.getColor());
        }}

        // Layout Persistence Functions
        // Use file-specific storage key with timestamp so each HTML copy has independent saved layouts
        const LAYOUT_STORAGE_KEY = '{js_escape(storage_key)}';

        // Evict stale saved-layout keys (keep the newest 20 per storage family)
        try {{
            ['cytoscape_layout_', 'heatmap_settings_'].forEach(function(prefix) {{
                const keys = [];
                for (let i = 0; i < localStorage.length; i++) {{
                    const k = localStorage.key(i);
                    if (k && k.startsWith(prefix)) keys.push(k);
                }}
                keys.sort(function(a, b) {{
                    const ta = (a.match(/#(\d+)$/) || ['', '0'])[1];
                    const tb = (b.match(/#(\d+)$/) || ['', '0'])[1];
                    return tb.localeCompare(ta);
                }});
                keys.slice(20).forEach(function(k) {{ localStorage.removeItem(k); }});
            }});
        }} catch (e) {{ /* localStorage unavailable (privacy mode) */ }}
        
        function saveLayout() {{
            try {{
                const state = {{
                    // Node positions
                    positions: cy.nodes().map(n => ({{
                        id: n.id(),
                        position: n.position()
                    }})),
                    // Node colors
                    colors: cy.nodes().map(n => ({{
                        id: n.id(),
                        color: n.style('background-color')
                    }})),
                    // Node visibility
                    visibility: cy.nodes().map(n => ({{
                        id: n.id(),
                        visible: n.visible(),
                        hidden: n.hasClass('hidden')
                    }})),
                    // Edge visibility
                    edgeVisibility: cy.edges().map(e => ({{
                        id: e.id(),
                        visible: e.visible(),
                        hidden: e.hasClass('hidden')
                    }})),
                    // UI state
                    zoom: cy.zoom(),
                    pan: cy.pan(),
                    labelsVisible: labelsVisible,
                    // Control values
                    edgeWidth: document.getElementById('edgeWidthSlider').value,
                    edgeWidthScale: document.getElementById('edgeWidthScale').value,
                    arrowSize: document.getElementById('arrowSizeSlider').value,
                    fontSize: document.getElementById('fontSizeSlider').value,
                    nodeSize: document.getElementById('nodeSizeSlider').value,
                    // Metadata
                    timestamp: new Date().toISOString(),
                    graphName: '{js_escape(output_name)}'
                }};
                
                localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(state));
                showToast('Layout saved', 'success');
                console.log('Layout saved successfully');
            }} catch (error) {{
                showToast('Save failed', 'error');
                console.error('Error saving layout:', error);
            }}
        }}
        
        function loadLayout() {{
            try {{
                const saved = localStorage.getItem(LAYOUT_STORAGE_KEY);
                if (!saved) {{
                    showToast('No saved layout found', 'warn');
                    return;
                }}
                
                const state = JSON.parse(saved);
                
                // Restore node positions
                state.positions.forEach(item => {{
                    const node = cy.getElementById(item.id);
                    if (node.length > 0) {{
                        node.position(item.position);
                    }}
                }});
                
                // Restore node colors
                state.colors.forEach(item => {{
                    const node = cy.getElementById(item.id);
                    if (node.length > 0) {{
                        node.style('background-color', item.color);
                    }}
                }});
                
                // Restore node visibility
                state.visibility.forEach(item => {{
                    const node = cy.getElementById(item.id);
                    if (node.length > 0) {{
                        if (item.hidden) {{
                            node.addClass('hidden');
                        }} else {{
                            node.removeClass('hidden');
                        }}
                    }}
                }});
                
                // Restore edge visibility
                if (state.edgeVisibility) {{
                    state.edgeVisibility.forEach(item => {{
                        const edge = cy.getElementById(item.id);
                        if (edge.length > 0) {{
                            if (item.hidden) {{
                                edge.addClass('hidden');
                            }} else {{
                                edge.removeClass('hidden');
                            }}
                        }}
                    }});
                }}
                
                // Restore zoom and pan (validate the saved state first)
                if (Number.isFinite(state.zoom)) {{
                    cy.zoom(state.zoom);
                }}
                if (state.pan && Number.isFinite(state.pan.x) && Number.isFinite(state.pan.y)) {{
                    cy.pan(state.pan);
                }}
                
                // Restore label visibility
                if (state.labelsVisible !== undefined && state.labelsVisible !== labelsVisible) {{
                    toggleLabels();
                }}
                
                // Restore control values (suppressed from history: loading
                // a layout is one operation, not a series of slider edits)
                restoringHistoryState = true;
                try {{
                    if (state.edgeWidth) {{
                        document.getElementById('edgeWidthSlider').value = state.edgeWidth;
                        updateEdgeWidth(state.edgeWidth);
                    }}
                    if (state.edgeWidthScale) {{
                        document.getElementById('edgeWidthScale').value = state.edgeWidthScale;
                        updateEdgeWidths();
                    }}
                    if (state.arrowSize) {{
                        document.getElementById('arrowSizeSlider').value = state.arrowSize;
                        updateArrowSize(state.arrowSize);
                    }}
                    if (state.fontSize) {{
                        document.getElementById('fontSizeSlider').value = state.fontSize;
                        updateFontSize(state.fontSize);
                    }}
                    if (state.nodeSize) {{
                        document.getElementById('nodeSizeSlider').value = state.nodeSize;
                        updateNodeSize(state.nodeSize);
                    }}
                }} finally {{
                    restoringHistoryState = false;
                }}
                
                showToast('Layout loaded', 'success');
                console.log('Layout loaded successfully:', state);
            }} catch (error) {{
                showToast('Load failed', 'error');
                console.error('Error loading layout:', error);
            }}
        }}

        function showAllNodes() {{
            pushHistory('Show all');
            cy.elements().removeClass('hidden');
            document.getElementById('showAllBtn').style.display = 'none';
            resizeCanvasAfterVisibilityControlChange();
            reapplyDeadEndHiding();
        }}

        // ===== UNDO / REDO =====
        // Full-state snapshots (data + classes + positions + visibility
        // toggles + view + edge filter) captured BEFORE each mutating
        // operation; bounded history keeps memory predictable. Undo/redo
        // executes strictly by walking this history: undo pops the last
        // entry and re-applies its snapshot, redo re-applies the state
        // that was undone.
        let undoStack = [];
        let redoStack = [];
        const HISTORY_LIMIT = 50;
        let lastFilterHistoryValue = '';  // last edge-filter value recorded in history

        function captureStyleBypass(el) {{
            // Style bypasses (per-element overrides set via el.style()) live in
            // _private.style; ele.json() does NOT expose them in Cytoscape
            // 3.28.1, so read them directly and deep-copy: the object is a
            // live reference into the element and later edits would otherwise
            // corrupt earlier snapshots.
            // IMPORTANT: _private.style also receives COMPUTED (non-bypass)
            // entries — e.g. the default :active overlay (black,
            // overlay-opacity 0.25) written while a node is grabbed — and
            // those linger after the drag ends. Copying them would turn the
            // transient drag shading into a permanent bypass on undo, so
            // only entries flagged bypass === true are captured.
            const st = el._private && el._private.style;
            if (!st) return null;
            const keys = Object.keys(st);
            if (keys.length === 0) return null;
            const out = {{}};
            keys.forEach(k => {{
                const v = st[k];
                if (!v || v.bypass !== true) return;
                out[k] = (v && typeof v === 'object' && 'value' in v) ? v.value : v;
            }});
            if (Object.keys(out).length === 0) return null;
            return JSON.parse(JSON.stringify(out));
        }}

        function captureState() {{
            // Deep-copy data()/position(): Cytoscape returns LIVE references to
            // its internal objects, so storing them directly would let later
            // mutations corrupt every previously captured snapshot and break
            // undo. Data is JSON-serializable (comes from the Python side).
            return {{
                nodes: cy.nodes().map(n => ({{
                    data: JSON.parse(JSON.stringify(n.data())),
                    classes: n.classes(),
                    position: {{ x: n.position().x, y: n.position().y }},
                    // Per-element style overrides (color/size bypasses)
                    style: captureStyleBypass(n)
                }})),
                edges: cy.edges().map(e => ({{
                    data: JSON.parse(JSON.stringify(e.data())),
                    classes: e.classes(),
                    style: captureStyleBypass(e)
                }})),
                selected: cy.$(':selected').map(el => el.id()),
                zoom: cy.zoom(),
                pan: {{ x: cy.pan().x, y: cy.pan().y }},
                // Use the last COMMITTED filter value: the input already shows
                // the user's new text when updateIgnoredEdges runs, so reading
                // it here would capture the post-op value instead of the
                // pre-op state the snapshot must represent.
                filterValue: lastFilterHistoryValue,
                selfLoopsHidden: selfLoopsHidden,
                orphansHidden: orphansHidden,
                deadEndsHidden: deadEndsHidden,
                hemisphereMirrorEnabled: hemisphereMirrorEnabled,
                labelPosition: labelPosition,
                // Global style controls (node/edge size, font/arrow size,
                // scaling method, metric, reciprocal offset) so undo/redo
                // also reverts size and width adjustments exactly.
                globalStyles: {{
                    nodeSize: globalNodeSize,
                    edgeWidth: globalEdgeWidth,
                    fontSize: globalFontSize,
                    edgeLabelFontSize: globalEdgeLabelFontSize,
                    arrowSize: globalArrowSize,
                    edgeWidthScale: globalEdgeWidthScale,
                    metric: currentMetric,
                    reciprocalOffset: reciprocalOffset,
                    // Layout transform trackers so undo/redo also restores
                    // the absolute gaps / rotation base exactly.
                    spacingX: lastGapX,
                    spacingY: lastGapY,
                    rotation: lastRotationDeg
                }}
            }};
        }}

        function restoreState(state) {{
            cy.elements().remove();
            cy.add(state.nodes.map(n => ({{ data: n.data, classes: n.classes, position: n.position }})));
            cy.add(state.edges.map(e => ({{ data: e.data, classes: e.classes }})));
            // Re-apply the captured per-element style overrides so undo/redo
            // round-trips individual color/size edits exactly (re-added
            // elements start with stylesheet appearance only).
            state.nodes.forEach(n => {{
                if (n.style) cy.getElementById(n.data.id).style(n.style);
            }});
            state.edges.forEach(e => {{
                if (e.style) cy.getElementById(e.data.id).style(e.style);
            }});
            (state.selected || []).forEach(id => {{
                const el = cy.getElementById(id);
                if (el.length > 0) el.select();
            }});

            // Restore the visibility-toggle flags BEFORE re-applying the
            // edge filter, so later operations (filter changes, hide
            // toggles) stay consistent with the restored classes.
            selfLoopsHidden = !!state.selfLoopsHidden;
            orphansHidden = !!state.orphansHidden;
            deadEndsHidden = !!state.deadEndsHidden;
            hemisphereMirrorEnabled = !!state.hemisphereMirrorEnabled;
            if (state.labelPosition !== undefined) labelPosition = state.labelPosition;
            syncToggleButtons();

            // Restore the edge-filter input; re-applying it re-derives the
            // filtered classes exactly as captured.
            const filterInput = document.getElementById('ignoreEdgesInput');
            const restoredFilter = state.filterValue || '';
            lastFilterHistoryValue = restoredFilter;
            if (filterInput && filterInput.value !== restoredFilter) filterInput.value = restoredFilter;
            parseEdgeFilterInput();
            applyEdgeFilter();

            // Restore view and the manual-hide indicator
            if (state.zoom) cy.zoom(state.zoom);
            if (state.pan) cy.pan(state.pan);
            const showAll = document.getElementById('showAllBtn');
            if (showAll) showAll.style.display = cy.nodes('.hidden').length > 0 ? 'inline-block' : 'none';

            // Restore the global style controls captured in the snapshot
            // (runs before refreshEdgeStyles so self-loops re-curve with
            // the restored node sizes / edge widths).
            restoreGlobalStyles(state.globalStyles);

            refreshEdgeStyles(false);
            updateUndoRedoButtons();
        }}

        // Re-apply a captured set of global style controls (node size,
        // edge width, font/arrow size, scaling method, metric, reciprocal
        // offset) without recording new history entries.
        function restoreGlobalStyles(gs) {{
            if (!gs) return;
            restoringHistoryState = true;
            try {{
                const applySlider = (id, value) => {{
                    const el = document.getElementById(id);
                    if (el && value !== undefined && value !== null) el.value = value;
                }};
                applySlider('nodeSizeSlider', gs.nodeSize);
                applySlider('edgeWidthSlider', gs.edgeWidth);
                applySlider('fontSizeSlider', gs.fontSize);
                applySlider('edgeLabelSizeSlider', gs.edgeLabelFontSize);
                applySlider('arrowSizeSlider', gs.arrowSize);
                const scaleSelect = document.getElementById('edgeWidthScale');
                if (scaleSelect && gs.edgeWidthScale) scaleSelect.value = gs.edgeWidthScale;
                const metricSelect = document.getElementById('metricSelect');
                if (metricSelect && gs.metric) metricSelect.value = gs.metric;
                if (gs.reciprocalOffset !== undefined) {{
                    reciprocalOffset = gs.reciprocalOffset;
                    const rSlider = document.getElementById('reciprocalOffsetSlider');
                    if (rSlider) rSlider.value = gs.reciprocalOffset;
                    const rLabel = document.getElementById('reciprocalOffsetValue');
                    if (rLabel) rLabel.textContent = Math.round(gs.reciprocalOffset) + 'px';
                }}
                // Restore the layout transform trackers (absolute gaps and
                // rotation base) and re-sync their spinners so the next
                // adjustment applies from the restored reference.
                if (gs.spacingX !== undefined) lastGapX = gs.spacingX;
                if (gs.spacingY !== undefined) lastGapY = gs.spacingY;
                if (gs.rotation !== undefined) lastRotationDeg = gs.rotation;
                syncTransformInputs();
                // Re-apply through the update functions (they read the DOM
                // controls); the flag above stops them from pushing new
                // history entries. updateMetric() re-derives currentMetric
                // and re-applies the width mapping last.
                if (gs.nodeSize !== undefined) updateNodeSize(gs.nodeSize);
                if (gs.edgeWidth !== undefined) updateEdgeWidth(gs.edgeWidth);
                if (gs.fontSize !== undefined) updateFontSize(gs.fontSize);
                if (gs.edgeLabelFontSize !== undefined) updateEdgeLabelFontSize(gs.edgeLabelFontSize);
                if (gs.arrowSize !== undefined) updateArrowSize(gs.arrowSize);
                updateMetric();
            }} finally {{
                restoringHistoryState = false;
            }}
        }}

        // Keep the three visibility-toggle buttons and the mirror button in
        // sync with their flags (used by restoreState).
        function syncToggleButtons() {{
            const deadBtn = document.getElementById('hideDeadEndsBtn');
            if (deadBtn) {{
                deadBtn.textContent = deadEndsHidden ? '👁️ Show Dead Ends' : '💀 Hide Dead Ends';
                deadBtn.style.background = deadEndsHidden ? '#e91e63' : '#607d8b';
            }}
            const orphanBtn = document.getElementById('hideOrphansBtn');
            if (orphanBtn) {{
                orphanBtn.textContent = orphansHidden ? '👁️ Show Orphans' : '👻 Hide Orphans';
                orphanBtn.style.background = orphansHidden ? '#e91e63' : '#9c27b0';
            }}
            const loopBtn = document.getElementById('hideSelfLoopsBtn');
            if (loopBtn) {{
                loopBtn.textContent = selfLoopsHidden ? '👁️ Show Self-Loops' : '🔁 Hide Self-Loops';
                loopBtn.style.background = selfLoopsHidden ? '#e91e63' : '#ff5722';
            }}
            const mirrorBtn = document.getElementById('mirrorHemiBtn');
            if (mirrorBtn) {{
                mirrorBtn.textContent = hemisphereMirrorEnabled ? '🪞 Mirrored' : '🪞 Mirror Hemispheres';
                mirrorBtn.style.background = hemisphereMirrorEnabled ? '#0ea5e9' : '#64748b';
            }}
        }}

        function pushStateHistory(label, state) {{
            undoStack.push({{ label: label, state: state }});
            if (undoStack.length > HISTORY_LIMIT) undoStack.shift();
            redoStack = [];
            updateUndoRedoButtons();
        }}

        function pushHistory(label) {{
            pushStateHistory(label, captureState());
        }}

        // Node relocation (drag) is part of the history: the pre-drag state
        // is stashed on grab and committed on dragfree, so undo restores the
        // positions before the move. NOTE: node drags in Cytoscape.js fire
        // grab/drag/free/dragfree — there is no node-level 'dragstart' (that
        // event only exists for core pan gestures), so the stash must happen
        // on 'grab' or drags are never recorded. Click-without-drag and
        // multi-node drags (dragfree can fire once per node) are deduplicated
        // by checking that positions actually changed.
        let pendingDragState = null;
        function registerDragHistory() {{
            cy.on('grab', 'node', function(evt) {{
                pendingDragState = captureState();
            }});
            cy.on('dragfree', 'node', function(evt) {{
                if (!pendingDragState) return;
                const posById = {{}};
                pendingDragState.nodes.forEach(n => {{ posById[n.data.id] = n.position; }});
                let moved = false;
                cy.nodes().forEach(n => {{
                    const p0 = posById[n.id()];
                    const p1 = n.position();
                    if (p0 && (p0.x !== p1.x || p0.y !== p1.y)) moved = true;
                }});
                if (moved) pushStateHistory('Move nodes', pendingDragState);
                pendingDragState = null;
                // Keep the numeric geometry inputs in sync with manual drags
                // (dragged node only when it is still selected).
                if (evt.target.selected()) syncSelectedGeometryInputs(evt.target);
                updateSelectionChip();
            }});
        }}
        registerDragHistory();

        // Keep the selection controls in sync with ANY selection change
        // (tap, shift/box selection, or programmatic selection).  A Cytoscape
        // tap handler can run before the element has been marked selected, so
        // refresh the geometry rows again from the selection event; otherwise
        // updateAlignButtons() sees an empty selection and hides the rows
        // immediately after the tap handler shows them.
        cy.on('select unselect', 'node, edge', function(evt) {{
            const selected = cy.$(':selected');
            if (selected.length === 0) {{
                syncSelectedGeometryInputs(null);
                updateSelectionChip();
                return;
            }}

            // Prefer the last tapped element when it is still selected.  For
            // shift/box/programmatic selection, use the event target or the
            // first remaining selected element as the geometry primary.
            let primary = selectedElement;
            if (!primary || !primary.selected()) {{
                primary = (evt.target && evt.target.selected()) ? evt.target : selected[0];
            }}
            syncSelectedGeometryInputs(primary);
            updateSelectionChip();
        }});

        function undo() {{
            if (undoStack.length === 0) return;
            const entry = undoStack[undoStack.length - 1];
            redoStack.push({{ label: entry.label, state: captureState() }});
            restoreState(undoStack.pop().state);
            const last = undoStack[undoStack.length - 1];
            updateHoverInfo('↩️ Undo' + (last ? ': ' + last.label : ''));
        }}

        function redo() {{
            if (redoStack.length === 0) return;
            const entry = redoStack[redoStack.length - 1];
            undoStack.push({{ label: entry.label, state: captureState() }});
            restoreState(redoStack.pop().state);
            updateHoverInfo('↪️ Redo: ' + entry.label);
        }}

        // Live selection summary chip in the Selection & Color section.
        function updateSelectionChip() {{
            const chip = document.getElementById('selectionSummary');
            if (!chip) return;
            const nodes = cy.$('node:selected').length;
            const edges = cy.$('edge:selected').length;
            if (nodes === 0 && edges === 0) {{
                chip.textContent = 'Nothing selected';
                return;
            }}
            const parts = [];
            if (nodes > 0) parts.push(nodes + (nodes === 1 ? ' node' : ' nodes'));
            if (edges > 0) parts.push(edges + (edges === 1 ? ' edge' : ' edges'));
            chip.textContent = parts.join(' · ') + ' selected';
        }}

        // Category icon for a history entry (display only — the recorded
        // labels are unchanged).
        function historyIcon(label) {{
            const l = String(label || '').toLowerCase();
            if (l.includes('delete')) return '🗑️';
            if (l.includes('hide') || l.includes('show')) return '👁️';
            if (l.includes('import')) return '📥';
            if (l.includes('export')) return '📤';
            if (l.includes('edit') || l.includes('resize') || l.includes('align') || l.includes('add ')) return '✏️';
            if (l.includes('layout') || l.includes('rotate') || l.includes('spacing') || l.includes('refresh') || l.includes('reset')) return '⤢';
            if (l.includes('color') || l.includes('group')) return '🎨';
            if (l.includes('toggle')) return '🔁';
            return '↩';
        }}

        function updateUndoRedoButtons() {{
            const u = document.getElementById('undoBtn');
            const r = document.getElementById('redoBtn');
            if (u) u.style.opacity = undoStack.length > 0 ? '1' : '0.4';
            if (r) r.style.opacity = redoStack.length > 0 ? '1' : '0.4';
            updateHistoryList();
        }}

        // History dropdown: lists every recorded operation (oldest first),
        // marks the current state, and lets the user undo/redo to any entry.
        function updateHistoryList() {{
            const sel = document.getElementById('historyList');
            if (!sel) return;
            sel.innerHTML = '';
            undoStack.forEach((item, i) => {{
                const opt = document.createElement('option');
                opt.textContent = (i + 1) + '. ' + historyIcon(item.label) + ' ' + item.label;
                sel.appendChild(opt);
            }});
            const cur = document.createElement('option');
            cur.disabled = true;
            cur.textContent = '▶ Current state';
            sel.appendChild(cur);
            redoStack.forEach((item, i) => {{
                const opt = document.createElement('option');
                opt.textContent = (undoStack.length + i + 2) + '. ↪ ' + item.label;
                sel.appendChild(opt);
            }});
            sel.selectedIndex = undoStack.length;
        }}

        // Undo/redo to the history entry at the given dropdown index.
        function jumpToHistory(index) {{
            if (index < 0 || index === undoStack.length) return;
            if (index < undoStack.length) {{
                while (undoStack.length > index) undo();
            }} else {{
                const needed = index - undoStack.length;
                for (let i = 0; i < needed && redoStack.length > 0; i++) redo();
            }}
            updateHistoryList();
        }}

        // System shortcuts: Cmd/Ctrl+Z undo, Cmd/Ctrl+Shift+Z or Cmd/Ctrl+Y redo
        // (skipped while typing in inputs/textareas)
        document.addEventListener('keydown', function(e) {{
            if (!(e.metaKey || e.ctrlKey)) return;
            if (dialogCtl.isActive()) return;
            const tag = (e.target.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
            const k = e.key.toLowerCase();
            if (k === 'z') {{
                e.preventDefault();
                if (e.shiftKey) {{ redo(); }} else {{ undo(); }}
            }} else if (k === 'y') {{
                e.preventDefault();
                redo();
            }}
        }});
        
        function clearEdgeEndpointOverrides(edge) {{
            edge.removeStyle('source-endpoint');
            edge.removeStyle('target-endpoint');
            edge.removeStyle('edge-distances');
            edge.removeStyle('source-distance-from-node');
            edge.removeStyle('target-distance-from-node');
        }}

        function applyStraightEdgeStyle(edge) {{
            edge.style('curve-style', 'straight');
            edge.removeStyle('control-point-distances');
            edge.removeStyle('control-point-weights');
        }}

        function refreshEdgeStyles(showStatus) {{
            const shouldShowStatus = (showStatus === undefined) ? true : showStatus;
            const offsetMagnitude = Math.max(0, parseFloat(reciprocalOffset) || 0);  // Keep reciprocal edges parallel but separated

            // Count visible edges per (source -> target) direction so the
            // reciprocal check is O(1) per edge instead of a full selector
            // scan per edge (O(E²) on large graphs). Using a per-direction
            // COUNT (not a set of all edges) is essential: an edge's own
            // direction key must not count as a parallel, or every edge
            // would wrongly take the reciprocal-offset branch and its
            // arrows would miss the node centers.
            const visibleEdgeCounts = new Map();
            cy.edges().forEach(e => {{
                if (!e.hasClass('hidden') && !e.hasClass('filtered')) {{
                    const key = e.source().id() + '→' + e.target().id();
                    visibleEdgeCounts.set(key, (visibleEdgeCounts.get(key) || 0) + 1);
                }}
            }});

            // Recalculate edge styles to make single edges straight (no curve)
            // This is useful when parallel/reciprocal edges are hidden
            cy.edges().forEach(edge => {{
                const source = edge.source().id();
                const target = edge.target().id();
                const canonicalSign = source.localeCompare(target) < 0 ? 1 : -1;
                
                // Check if there's a visible parallel edge (both directions);
                // a count > 0 for the REVERSE direction means a DIFFERENT
                // edge goes the other way (the edge itself contributes to
                // its own direction key, not the reverse one)
                const hasVisibleParallel = (visibleEdgeCounts.get(target + '→' + source) || 0) > 0;

                if (source === target) {{
                    // Self-loop geometry: Cytoscape renders EVERY self-loop
                    // as a single cubic bezier (it forces curve-style
                    // 'bezier' on loops), with two control points at
                    // 1.4 * control-point-step-size from the node center at
                    // angles given by loop-direction/loop-sweep.
                    // control-point-distances is IGNORED for self-loops.
                    // With the default 90deg sweep the endpoints land on
                    // the node circle exactly 90deg apart (top -> left) and
                    // both end tangents pass through the node center. A
                    // control-point distance of 3.0 * nodeRadius is the
                    // closest single-cubic approximation of the ideal 3/4
                    // circle with the same radius as the node (max
                    // deviation ~29% of r); the step size scales with the
                    // ACTUAL rendered node size (global slider or
                    // per-element geometry edits).
                    const loopNodeSize = edge.source().numericStyle('width') ||
                        parseFloat(document.getElementById('nodeSizeSlider')?.value || 40);
                    edge.style('curve-style', 'bezier');
                    edge.style('control-point-step-size', (3.0 * loopNodeSize / 2) / 1.4);
                    edge.style('loop-direction', '-45deg');
                    edge.style('loop-sweep', '-90deg');
                    return;
                }}

                if (!hasVisibleParallel) {{
                    clearEdgeEndpointOverrides(edge);
                    applyStraightEdgeStyle(edge);
                    return;
                }}

                // Anchor distances must follow the ACTUAL size of each
                // endpoint node: nodes can be individually resized via the
                // geometry editor, so the global slider is only a fallback
                // for nodes that still use the default stylesheet width.
                const globalNodeSize = parseFloat(document.getElementById('nodeSizeSlider')?.value || 40);
                const sourceNodeSize = edge.source().numericStyle('width') || globalNodeSize;
                const targetNodeSize = edge.target().numericStyle('width') || globalNodeSize;
                
                // Compute perpendicular offset in CANONICAL direction (smaller ID -> larger ID)
                // to ensure reciprocal edges offset in opposite directions relative to canvas
                const canonicalSourceId = source < target ? source : target;
                const canonicalTargetId = source < target ? target : source;
                const canonicalSourcePos = cy.getElementById(canonicalSourceId).position();
                const canonicalTargetPos = cy.getElementById(canonicalTargetId).position();
                
                const canonicalDx = canonicalTargetPos.x - canonicalSourcePos.x;
                const canonicalDy = canonicalTargetPos.y - canonicalSourcePos.y;
                const canonicalDistance = Math.hypot(canonicalDx, canonicalDy);
                
                let sourceOffsetX = 0;
                let sourceOffsetY = 0;
                let targetOffsetX = 0;
                let targetOffsetY = 0;
                let sourceDistance = sourceNodeSize / 2;
                let targetDistance = targetNodeSize / 2;
                let perpX = 0;
                let perpY = 0;

                if (canonicalDistance > 0) {{
                    // Perpendicular vector to canonical direction
                    perpX = -canonicalDy / canonicalDistance;
                    perpY = canonicalDx / canonicalDistance;
                    
                    // Apply offset: canonicalSign determines which side of the canonical line
                    // This ensures reciprocal edges move to opposite sides relative to canvas
                    sourceOffsetX = perpX * offsetMagnitude * canonicalSign;
                    sourceOffsetY = perpY * offsetMagnitude * canonicalSign;
                    targetOffsetX = perpX * offsetMagnitude * canonicalSign;
                    targetOffsetY = perpY * offsetMagnitude * canonicalSign;
                }} else {{
                    sourceOffsetY = offsetMagnitude * canonicalSign;
                    targetOffsetY = offsetMagnitude * canonicalSign;
                }}

                if (straightReciprocalEdgesEnabled) {{
                    // Reciprocal edges: keep them straight but offset slightly so they don't overlap
                    applyStraightEdgeStyle(edge);

                    edge.style({{
                        'edge-distances': 'node-position',
                        'source-endpoint': `${{sourceOffsetX.toFixed(2)}} ${{sourceOffsetY.toFixed(2)}}`,
                        'target-endpoint': `${{targetOffsetX.toFixed(2)}} ${{targetOffsetY.toFixed(2)}}`,
                        'source-distance-from-node': sourceDistance,
                        'target-distance-from-node': targetDistance
                    }});
                }} else {{
                    // Has visible parallel edge, use curved style (no offsets in curved mode)
                    edge.style('curve-style', 'bezier');
                    edge.style('control-point-distances', 40);
                    edge.style('control-point-weights', 0.5);

                    // Clear any endpoint offsets and distance overrides from straight mode
                    edge.removeStyle('source-endpoint');
                    edge.removeStyle('target-endpoint');
                    edge.removeStyle('edge-distances');
                    edge.removeStyle('source-distance-from-node');
                    edge.removeStyle('target-distance-from-node');
                }}
            }});
            
            if (shouldShowStatus) {{
                updateHoverInfo('✓ Edge styles refreshed - parallel edges updated');
            }}
        }}

        function initializeReciprocalOffsetControls() {{
            const container = document.getElementById('reciprocalOffsetControls');
            const slider = document.getElementById('reciprocalOffsetSlider');
            const valueLabel = document.getElementById('reciprocalOffsetValue');

            if (!container || !slider) {{
                return;
            }}

            // The offset controls only matter when reciprocal pairs exist
            // on the canvas — keep the group hidden otherwise.  One pass
            // over the directed pairs: a per-edge selector scan is O(E²)
            // on large graphs and breaks on ids containing quotes.
            const directedPairs = new Set();
            let hasReciprocalPairs = false;
            cy.edges().forEach(e => {{
                const source = e.source().id();
                const target = e.target().id();
                if (directedPairs.has(target + '\u0000' + source)) {{
                    hasReciprocalPairs = true;
                }}
                directedPairs.add(source + '\u0000' + target);
            }});
            container.style.display = hasReciprocalPairs ? 'flex' : 'none';
            slider.value = defaultReciprocalOffset;
            reciprocalOffset = defaultReciprocalOffset;
            if (valueLabel) valueLabel.textContent = `${{defaultReciprocalOffset}}px`;
            
            // Update slider enabled state based on current mode
            updateReciprocalSliderState();

            if (!slider.dataset.bound) {{
                slider.addEventListener('input', function(event) {{
                    const newOffset = parseFloat(event.target.value) || 0;
                    if (!restoringHistoryState && newOffset !== reciprocalOffset) pushHistory('Adjust reciprocal offset');
                    reciprocalOffset = newOffset;
                    if (valueLabel) valueLabel.textContent = `${{Math.round(reciprocalOffset)}}px`;
                    refreshEdgeStyles(false);
                }});
                slider.dataset.bound = 'true';
            }}
        }}
        
        function updateReciprocalSliderState() {{
            const slider = document.getElementById('reciprocalOffsetSlider');
            if (!slider) return;
            
            if (straightReciprocalEdgesEnabled) {{
                slider.disabled = false;
                slider.style.opacity = '1';
                slider.style.cursor = 'pointer';
            }} else {{
                slider.disabled = true;
                slider.style.opacity = '0.4';
                slider.style.cursor = 'not-allowed';
            }}
        }}
        
        function toggleReciprocalMode() {{
            straightReciprocalEdgesEnabled = !straightReciprocalEdgesEnabled;
            const toggleBtn = document.getElementById('reciprocalModeToggle');
            
            if (straightReciprocalEdgesEnabled) {{
                toggleBtn.textContent = 'Straight';
                toggleBtn.style.background = '#4caf50';
            }} else {{
                toggleBtn.textContent = 'Curved';
                toggleBtn.style.background = '#ff9800';
            }}
            
            updateReciprocalSliderState();
            refreshEdgeStyles(true);
        }}

        // Edge labels follow the ACTIVE connection metric: weight shows a
        // BARE number — no unit suffix; ratio/probability show the plain
        // value.
        function updateEdgeMetricLabels() {{
            cy.edges().forEach(edge => {{
                const fmt = v => (v === undefined || v === null || v === '' || isNaN(Number(v)))
                    ? '' : String(Math.round(Number(v) * 1000) / 1000);
                let label;
                if (currentMetric === 'ratio') label = fmt(edge.data('ratio'));
                else if (currentMetric === 'probability') label = fmt(edge.data('probability'));
                else {{
                    // Integer-valued weights must not render with a
                    // trailing '.0' — edge lists often parse synapse
                    // counts as floats.
                    const raw = (edge.data('original_weight') !== undefined && edge.data('original_weight') !== null)
                        ? Number(edge.data('original_weight')) : Number(edge.data('weight'));
                    if (!isFinite(raw)) label = '';
                    else if (Number.isInteger(raw)) label = raw.toLocaleString('en-US');
                    else label = String(Math.round(raw * 1000) / 1000);
                }}
                edge.data('display_label', label);
            }});
        }}

        function toggleLabels() {{
            const btn = document.getElementById('toggleLabelsBtn');
            
            if (labelsVisible) {{
                // Hide labels
                cy.nodes().addClass('labels-hidden');
                btn.textContent = '🏷️ Show Labels';
                labelsVisible = false;
            }} else {{
                // Show labels
                cy.nodes().removeClass('labels-hidden');
                btn.textContent = '🏷️ Hide Labels';
                labelsVisible = true;
            }}
        }}

        // Paint the preformatted weight on every edge (Filter tab →
        // Labels → Edge Weights). Class-based, so per-edge inline color
        // and width styling is untouched.
        function toggleEdgeWeightLabels() {{
            const btn = document.getElementById('toggleEdgeWeightsBtn');
            const showing = btn.dataset.showing === '1';
            if (showing) {{
                cy.edges().removeClass('wlabel');
                btn.textContent = '🏋️ Edge Weights';
                btn.dataset.showing = '0';
            }} else {{
                updateEdgeMetricLabels();
                cy.edges().addClass('wlabel');
                // The stylesheet default (#333) is only a fallback — the
                // labels follow the theme / the user's label font color.
                cy.edges('.wlabel').style('color', effectiveEdgeLabelColor());
                btn.textContent = '🏋️ Hide Weights';
                btn.dataset.showing = '1';
            }}
        }}
        
        // Background color toggle (shared controller)
        const bgCtrl = createBackgroundController(['#ffffff', '#000000'], ['White', 'Black'], applyBackground);
        
        function toggleBackground() {{
            bgCtrl.toggle();
        }}
        
        function applyBackground(color) {{
            document.body.style.background = color;
            document.getElementById('cy').style.background = color;
            
            // Flip every UI surface at once via the theme custom properties
            // (cards, inputs, panels, hover box, toasts, dialogs).
            const isDark = isColorDark(color);
            document.body.classList.toggle('vp-dark', isDark);
            
            // Node labels NEVER get a background; instead their text color
            // adapts to the theme unless the user picked one explicitly
            // (replaces the old per-label background readability mechanism).
            // Edge-weight labels adapt with them.
            if (!customLabelColor) {{
                const labelColor = isDark ? '#e5e7eb' : '#000000';
                cy.nodes().style('color', labelColor);
                cy.edges('.wlabel').style('color', labelColor);
                const picker = document.getElementById('labelFontColor');
                if (picker) picker.value = labelColor;
            }}
        }}
        
        // Node label font color (Layout & Style tab → Background & Font). The user's
        // choice wins over the automatic theme adaptation. On-edge weight
        // labels (.wlabel) are text too — they follow the same color so
        // both label kinds stay readable when the background changes.
        let customLabelColor = null;
        function applyLabelFontColor(color) {{
            const hex = extractColorHex(color) || '#000000';
            customLabelColor = hex;
            const picker = document.getElementById('labelFontColor');
            if (picker && picker.value.toLowerCase() !== hex.toLowerCase()) picker.value = hex;
            if (!restoringHistoryState) pushHistory('Change label font color');
            cy.nodes().style('color', hex);
            cy.edges('.wlabel').style('color', hex);
        }}

        // The color edge-weight labels currently render with (theme
        // adaptation unless the user picked an explicit label color).
        function effectiveEdgeLabelColor() {{
            const dark = document.body.classList.contains('vp-dark');
            return customLabelColor || (dark ? '#e5e7eb' : '#000000');
        }}
        
        function applyCustomBackground() {{
            bgCtrl.applyCustom();
        }}
        
{SHARED_JS}

        function updateFontSize(size) {{
            const value = parseFloat(size);
            const echo = document.getElementById('fontSizeValue');
            if (echo) echo.textContent = value + 'px';
            if (!restoringHistoryState && value !== globalFontSize) pushHistory('Adjust font size');
            globalFontSize = value;
            cy.style()
                .selector('node')
                .style('font-size', value + 'px')
                .update();
        }}

        // Text size of the on-edge weight labels (the 🏋️ Edge Weights
        // toggle) — independent of the node-label Font Size control.
        function updateEdgeLabelFontSize(size) {{
            const value = parseFloat(size);
            if (!restoringHistoryState && value !== globalEdgeLabelFontSize) pushHistory('Adjust edge label font size');
            globalEdgeLabelFontSize = value;
            cy.edges('.wlabel').style('font-size', value + 'px');
        }}

        function updateNodeSize(size) {{
            const value = parseFloat(size);
            const echo = document.getElementById('nodeSizeValue');
            if (echo) echo.textContent = value + 'px';
            if (!restoringHistoryState && value !== globalNodeSize) pushHistory('Adjust node size');
            globalNodeSize = value;
            cy.style()
                .selector('node')
                .style({{
                    'width': value + 'px',
                    'height': value + 'px'
                }})
                .update();
            
            // Refresh edge styles to update arrow positions when node size changes
            refreshEdgeStyles(false);
        }}

        function updateEdgeWidth(width) {{
            const echo = document.getElementById('edgeWidthValue');
            if (echo) echo.textContent = width + 'px';
            const value = parseFloat(width);
            if (!restoringHistoryState && value !== globalEdgeWidth) pushHistory('Adjust edge width');
            globalEdgeWidth = value;
            const edges = cy.edges();
            
            // Get current scaling method
            const method = document.getElementById('edgeWidthScale').value;
            
            // Calculate min/max widths based on the input width
            const minWidth = parseFloat(width) * 0.2;  // 20% of width
            const maxWidth = parseFloat(width);
            
            // Get scaled widths for current method
            const scaledWidths = edges.map(e => e.data('scaled_width'));
            
            // Calculate correct minScaled/maxScaled for the CURRENT method
            let minScaled, maxScaled;
            
            if (method.startsWith('log_')) {{
                // Use fixed scale ranges for log methods
                minScaled = 0;
                if (method === 'log_2') {{
                    maxScaled = 14;  // log2(10000) ≈ 13.3
                }} else {{ // log_10
                    maxScaled = 4.5;  // log10(10000) = 4
                }}
            }} else {{
                // For non-log methods, use actual data range
                minScaled = Math.min(...scaledWidths);
                maxScaled = Math.max(...scaledWidths);
                
                // Handle edge case where all values are the same
                if (minScaled === maxScaled) {{
                    minScaled = 0;
                    maxScaled = 1;
                }}
            }}
            
            // Update edge widths with normalization
            cy.style()
                .selector('edge')
                .style({{
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{minWidth}}, ${{maxWidth}})`
                }})
                .selector('edge:selected')
                .style({{
                    // selected edges must stay clearly visible: floor the
                    // width range so thin edges do not vanish on white
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{Math.max(minWidth * 1.5, 3)}}, ${{Math.max(maxWidth * 1.5, 5)}})`
                }})
                .selector('edge.highlighted')
                .style({{
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{minWidth * 2}}, ${{maxWidth * 2}})`
                }})
                .update();

            // Re-fit edge styles: self-loop curvature and arrow anchors
            // follow the rendered edge width.
            refreshEdgeStyles(false);
        }}

        function updateArrowSize(size) {{
            const value = parseFloat(size);
            const echo = document.getElementById('arrowSizeValue');
            if (echo) echo.textContent = value + 'px';
            if (!restoringHistoryState && value !== globalArrowSize) pushHistory('Adjust arrow size');
            globalArrowSize = value;
            cy.style()
                .selector('edge')
                .style({{
                    'arrow-scale': parseFloat(size) / 9  // Normalize to default size of 9
                }})
                .update();
        }}

        // ============ Color Palette Functions ============
        let selectedElement = null;  // For single selection info display
        
        // Color palette is a fixed menubar now; collapse toggle removed

        // Update opacity display
        function updateOpacityDisplay(type, value) {{
            const echo = document.getElementById(type + 'OpacityValue');
            if (echo) echo.textContent = value + '%';
        }}
        
        // Recalculate and update all edge widths based on scaling method
        // Current metric being used for edge widths
        let currentMetric = 'weight';

        // Global style state (size sliders and selects). These mirror the
        // DOM controls so history snapshots capture the PRE-change value
        // even though an 'input' event fires after the control already
        // moved — the same role as lastFilterHistoryValue for the edge
        // filter: the shadow holds the last value recorded in history.
        // restoringHistoryState suppresses new entries while undo/redo or
        // a layout/graph import re-applies a captured snapshot.
        let globalNodeSize = 40;
        let globalEdgeWidth = 3;
        let globalFontSize = 12;
        let globalEdgeLabelFontSize = {self.edge_label_font_size};
        let globalArrowSize = 9;
        let globalEdgeWidthScale = document.getElementById('edgeWidthScale')?.value || 'log_e';
        let restoringHistoryState = false;
        
        function updateMetric() {{
            const metric = document.getElementById('metricSelect').value;
            if (!restoringHistoryState && metric !== currentMetric) pushHistory('Change metric');
            currentMetric = metric;

            // Edge labels (when shown) re-format to the active metric
            updateEdgeMetricLabels();
            console.log(`\n========== UPDATE METRIC ==========`);
            console.log(`Metric selected: ${{metric}}`);

            // Update edge widths with new metric
            updateEdgeWidths();

            // The weight filter evaluates against the ACTIVE metric —
            // re-apply it so hidden edges follow the new unit, and adapt
            // the input placeholder so typed numbers match the unit.
            applyEdgeFilter();
            updateIgnoredEdgesPlaceholder();

            // Update hover info if currently hovering over an edge
            const hoverInfo = document.getElementById('hoverInfo');
            if (hoverInfo && hoverInfo.innerHTML.includes('Connection:')) {{
                // Just leave it as is - next hover will show updated info
            }}
        }}
                
        // Placeholder reflects the active metric so the numbers the user
        // types are in the filtered unit.
        function updateIgnoredEdgesPlaceholder() {{
            const input = document.getElementById('ignoreEdgesInput');
            if (!input) return;
            const examples = {{
                weight: 'OR: <5, >100 | AND: (>=5, <=10)',
                ratio: 'OR: <0.2, >0.8 | AND: (>=0.1, <=0.5)',
                probability: 'OR: <0.05, >0.9 | AND: (>=0.1, <=0.6)'
            }};
            input.placeholder = examples[currentMetric] || examples.weight;
        }}
        
        function updateEdgeWidths() {{
            const method = document.getElementById('edgeWidthScale').value;
            if (!restoringHistoryState && method !== globalEdgeWidthScale) pushHistory('Change edge width scale');
            globalEdgeWidthScale = method;
            
            console.log(`\\n========== UPDATE EDGE WIDTHS ==========`);
            console.log(`Method selected: ${{method}}`);
            console.log(`Current metric: ${{currentMetric}}`);
            
            // Get all edge values based on current metric
            const edges = cy.edges();
            let values;
            
            if (currentMetric === 'ratio') {{
                values = edges.map(e => e.data('ratio'));
            }} else if (currentMetric === 'probability') {{
                values = edges.map(e => e.data('probability'));
            }} else {{
                values = edges.map(e => e.data('weight'));
            }}
            
            if (values.length === 0) {{
                console.warn('No edges to update');
                return;
            }}
            
            console.log(`Values: [${{values.slice(0, 5).join(', ')}}...]`);
            
            // For ratio and probability, scale them up for better visualization
            const scaleFactor = (currentMetric === 'ratio' || currentMetric === 'probability') ? 1000 : 1;
            values = values.map(v => v * scaleFactor);
            
            // Calculate scaled widths based on EXACT method string
            let scaledWidths = values.map(w => {{
                let scaled;
                
                switch(method) {{
                    case 'linear':
                        scaled = w;
                        break;
                    
                    case 'log_e':
                        scaled = Math.log(w + 1);  // Natural log
                        break;
                    
                    case 'log_2':
                        scaled = Math.log(w + 1) / Math.log(2);  // Log base 2
                        break;
                    
                    case 'log_10':
                        scaled = Math.log(w + 1) / Math.log(10);  // Log base 10
                        break;
                    
                    case 'sqrt':
                        scaled = Math.sqrt(w);
                        break;
                    
                    case 'none':
                    default:
                        scaled = 1.0;
                        break;
                }}
                
                return scaled;
            }});
            
            console.log(`Scaled widths: [${{scaledWidths.map(v => v.toFixed(3)).join(', ')}}]`);
            
            // For logarithmic methods, use FIXED scale ranges based on typical weight distributions
            // For other methods, normalize to actual data range
            let minScaled, maxScaled;
            
            if (method.startsWith('log_')) {{
                // Use fixed scale ranges - assumes weights typically range from 1 to ~10000
                minScaled = 0;  // log(1) ≈ 0 for all bases
                
                // Set max based on log base
                if (method === 'log_2') {{
                    maxScaled = 14;  // log2(10000) ≈ 13.3
                }} else if (method === 'log_10') {{
                    maxScaled = 4.5;  // log10(10000) = 4
                }} else {{ // log_e (natural log)
                    maxScaled = 9.2;  // ln(10000) ≈ 9.2
                }}
                
                console.log(`Using FIXED scale for ${{method}}: [${{minScaled}}, ${{maxScaled}}]`);
            }} else {{
                // For non-log methods, normalize to actual data range
                minScaled = Math.min(...scaledWidths);
                maxScaled = Math.max(...scaledWidths);
                
                // Handle edge case where all values are the same
                if (minScaled === maxScaled) {{
                    minScaled = 0;
                    maxScaled = 1;
                }}
                
                console.log(`Using DATA scale: [${{minScaled.toFixed(2)}}, ${{maxScaled.toFixed(2)}}]`);
            }}
            
            // Update each edge's scaled_width data
            edges.forEach((edge, i) => {{
                edge.data('scaled_width', scaledWidths[i]);
            }});
            
            // Get edge width slider value
            const edgeWidthSlider = document.getElementById('edgeWidthSlider');
            const edgeWidth = edgeWidthSlider ? parseFloat(edgeWidthSlider.value) : 3;
            const minEdgeWidth = edgeWidth * 0.2;  // 20% of width
            const maxEdgeWidth = edgeWidth;
            
            // Update the stylesheet with new mapping range
            cy.style()
                .selector('edge')
                .style({{
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{minEdgeWidth}}, ${{maxEdgeWidth}})`
                }})
                .selector('edge:selected')
                .style({{
                    // selected edges must stay clearly visible: floor the
                    // width range so thin edges do not vanish on white
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{Math.max(minEdgeWidth * 1.5, 3)}}, ${{Math.max(maxEdgeWidth * 1.5, 5)}})`
                }})
                .selector('edge.highlighted')
                .style({{
                    'width': `mapData(scaled_width, ${{minScaled}}, ${{maxScaled}}, ${{minEdgeWidth * 2}}, ${{maxEdgeWidth * 2}})`
                }})
                .update();
            
            // Re-fit edge styles: self-loop curvature follows the widths
            // produced by the new scaling method / metric.
            refreshEdgeStyles(false);
            
            console.log('Edge widths updated successfully');
        }}
        
        // Get all currently selected elements (supports multi-selection)
        function getSelectedElements() {{
            return cy.$(':selected');
        }}
        
        // Count selected elements
        function getSelectionCount() {{
            const selected = getSelectedElements();
            const nodes = selected.nodes().length;
            const edges = selected.edges().length;
            return {{ nodes: nodes, edges: edges, total: nodes + edges }};
        }}

        // Update color text fields when color picker changes
        document.getElementById('groupColor').addEventListener('input', function(e) {{
            document.getElementById('groupColorText').value = e.target.value;
        }});
        document.getElementById('individualColor').addEventListener('input', function(e) {{
            document.getElementById('individualColorText').value = e.target.value;
        }});

        const EDGE_BASE_COLOR_KEY = '__baseColor';
        const EDGE_BASE_OPACITY_KEY = '__baseOpacity';

        function setEdgeBaseAppearance(edge, color, opacity, applyImmediately = true) {{
            edge.data(EDGE_BASE_COLOR_KEY, color);
            edge.data(EDGE_BASE_OPACITY_KEY, opacity);
            if (applyImmediately) {{
                edge.style({{
                    'line-color': color,
                    'target-arrow-color': color,
                    'opacity': opacity
                }});
            }}
        }}

        function ensureEdgeBaseAppearance(edge) {{
            if (!edge.data(EDGE_BASE_COLOR_KEY)) {{
                const currentColor = edge.style('line-color');
                const currentOpacity = parseFloat(edge.style('opacity')) || 1;
                setEdgeBaseAppearance(edge, currentColor, currentOpacity, false);
            }}
        }}

        function restoreEdgeBaseAppearance(edge) {{
            const color = edge.data(EDGE_BASE_COLOR_KEY) || edge.style('line-color');
            const opacityData = edge.data(EDGE_BASE_OPACITY_KEY);
            const opacity = (opacityData !== undefined) ? opacityData : (parseFloat(edge.style('opacity')) || 1);
            edge.style({{
                'line-color': color,
                'target-arrow-color': color,
                'opacity': opacity
            }});
        }}

        function applyEdgeHighlightOverride(edge) {{
            ensureEdgeBaseAppearance(edge);
            const baseOpacity = edge.data(EDGE_BASE_OPACITY_KEY);
            const fallbackOpacity = (baseOpacity !== undefined) ? baseOpacity : (parseFloat(edge.style('opacity')) || 1);
            const targetOpacity = Math.max(fallbackOpacity, highlightOpacity || 0.85);
            edge.style({{
                'line-color': highlightColor,
                'target-arrow-color': highlightColor,
                'opacity': targetOpacity
            }});
        }}

        function clearEdgeHighlightOverride(edge) {{
            if (edge.selected() || edge.hasClass('highlighted')) {{
                return;
            }}
            restoreEdgeBaseAppearance(edge);
        }}

        function removeHighlightFromEdges(edges, skipSelected = false) {{
            edges.forEach(edge => {{
                edge.removeClass('highlighted');
                if (skipSelected && edge.selected()) {{
                    return;
                }}
                clearEdgeHighlightOverride(edge);
            }});
        }}

        function addHighlightToEdges(edges) {{
            edges.addClass('highlighted');
            edges.forEach(edge => applyEdgeHighlightOverride(edge));
        }}

        function initializeEdgeBaseStyles() {{
            cy.edges().forEach(edge => {{
                const color = edge.style('line-color');
                const opacity = parseFloat(edge.style('opacity')) || 1;
                setEdgeBaseAppearance(edge, color, opacity, false);
            }});
        }}

        // Store default colors for groups (includes original defaults for reset)
        // Declared dataset groups replace the structural roles as the base
        // node groups (legend, quick actions, dropdown).
        const declaredGroupsActive = {'true' if self.node_groups else 'false'};
        const originalGroupDefaults = {{
            source: {{ color: '{self.node_color[0]}', opacity: {int(self.source_opacity * 100)} }},
            intermediate: {{ color: '{self.node_color[1]}', opacity: {int(self.intermediate_opacity * 100)} }},
            target: {{ color: '{self.target_color}', opacity: {int(self.target_opacity * 100)} }},
            hemisphere_left: {{ color: '{self.node_color[0]}', opacity: 100 }},
            hemisphere_right: {{ color: '{blend_with_gray(self.node_color[0], self.hemisphere_desaturate_factor)}', opacity: 100 }},
            hemisphere_unknown: {{ color: '#9ca3af', opacity: 100 }},
            positive_edges: {{ color: '{self.edge_color}', opacity: {int(self.edge_opacity * 100)} }},
            negative_edges: {{ color: '#4A90E2', opacity: 100 }}
        }};
        {extra_group_defaults_js}
        const groupDefaults = JSON.parse(JSON.stringify(originalGroupDefaults));
        
        // Custom group DEFINITIONS only — each entry is {{label, color,
        // opacity, defaultColor, defaultOpacity}} — membership lives on each
        // node's assigned_group data field ('' = Unassigned), so counts and
        // the legend are always live.
        const customGroups = {{}};

        // ===== NODE GROUP MEMBERSHIP =====
        function groupMembers(name) {{
            return cy.nodes('[assigned_group = "' + name + '"]');
        }}

        // Backfill nodes imported from older exports that lack the field.
        function normalizeAssignedGroups() {{
            cy.nodes().forEach(node => {{
                const g = node.data('assigned_group');
                if (g === undefined || g === null) {{
                    node.data('assigned_group', node.data('node_type') || '');
                }}
            }});
        }}
        normalizeAssignedGroups();

        function groupDefaultFor(name) {{
            if (name === '') return groupDefaults.unassigned || {{ color: '#9ca3af', opacity: 100 }};
            if (customGroups[name]) return {{ color: customGroups[name].color, opacity: customGroups[name].opacity }};
            return groupDefaults[name] || originalGroupDefaults[name] || {{ color: '#888888', opacity: 100 }};
        }}

        function applyGroupLook(node, def) {{
            node.style({{ 'background-color': def.color, 'opacity': ((def.opacity === undefined ? 100 : def.opacity) / 100) }});
        }}

        function groupLabel(name) {{
            if (customGroups[name]) return customGroups[name].label || name;
            const extra = extraNodeGroups.find(g => g.name === name);
            if (extra) return extra.label;
            return name ? name.charAt(0).toUpperCase() + name.slice(1) : 'Unassigned';
        }}

        // Assign nodes to a group (single membership). colorize applies the
        // group's current look; Unassigned instead reverts each node to its
        // structural default.
        function assignNodesToGroup(nodes, name, colorize) {{
            if (name === '') {{
                nodes.forEach(node => {{
                    node.data('assigned_group', '');
                    applyGroupLook(node, groupDefaultFor(String(node.data('node_type') || '')));
                }});
            }} else {{
                const def = groupDefaultFor(name);
                nodes.forEach(node => {{
                    node.data('assigned_group', name);
                    if (colorize) applyGroupLook(node, def);
                }});
            }}
            rebuildAssignSelect();
            rebuildCustomGroupUI();
            refreshLegend();
        }}

        // ===== DYNAMIC LEGEND =====
        // Rebuilds the group chips from live state (structural/declared
        // groups, custom groups, Unassigned when non-empty) with live colors
        // and member counts. Dataset-code chips live in #datasetLegend and
        // are never touched here.
        function legendChip(group, label, def) {{
            const count = groupMembers(group).length;
            return '<div class="legend-item" data-group="' + group + '" title="' + escapeHtml(label) + ': ' + count + ' node(s)">'
                + '<div class="legend-color" style="background: ' + def.color + '; opacity: ' + ((def.opacity === undefined ? 100 : def.opacity) / 100) + ';"></div>'
                + '<span>' + escapeHtml(label) + ' (' + count + ')</span></div>';
        }}

        function refreshLegend() {{
            const holder = document.getElementById('groupLegend');
            if (!holder) return;
            let html = '';
            if (!declaredGroupsActive) {{
                ['source', 'intermediate', 'target'].forEach(name => {{
                    html += legendChip(name, name.charAt(0).toUpperCase() + name.slice(1), groupDefaultFor(name));
                }});
            }}
            extraNodeGroups.forEach(g => {{
                if (datasetLegendCodes.has(g.name)) {{
                    // the static dataset chip already carries this code's
                    // count — no duplicated group chip in the header
                    return;
                }}
                html += legendChip(g.name, g.label, groupDefaultFor(g.name));
            }});
            Object.keys(customGroups).forEach(name => {{
                html += legendChip(name, customGroups[name].label || name, groupDefaultFor(name));
            }});
            if (groupMembers('').length > 0) {{
                html += legendChip('', 'Unassigned', groupDefaultFor(''));
            }}
            holder.innerHTML = html;
        }}
        refreshLegend();

        // ===== GROUP ASSIGNMENT UI =====
        function rebuildAssignSelect() {{
            const sel = document.getElementById('assignGroupSelect');
            if (!sel) return;
            const opts = [];
            if (!declaredGroupsActive) {{
                ['source', 'intermediate', 'target'].forEach(n => {{
                    opts.push({{ value: n, label: n.charAt(0).toUpperCase() + n.slice(1) }});
                }});
            }}
            extraNodeGroups.forEach(g => opts.push({{ value: g.name, label: g.label }}));
            Object.keys(customGroups).forEach(n => opts.push({{ value: 'custom_' + n, label: customGroups[n].label || n }}));
            opts.push({{ value: 'unassigned', label: 'Unassigned' }});
            const current = sel.value;
            sel.innerHTML = opts.map(o => '<option value="' + o.value + '">' + escapeHtml(o.label) + '</option>').join('');
            if (opts.some(o => o.value === current)) sel.value = current;
        }}
        rebuildAssignSelect();

        // Preselect the group shared by the current node selection (if any).
        function syncAssignSelectToSelection() {{
            const sel = document.getElementById('assignGroupSelect');
            if (!sel) return;
            const nodes = cy.$(':selected').nodes();
            if (nodes.length === 0) return;
            const first = nodes[0].data('assigned_group') || '';
            const mixed = nodes.toArray().some(n => (n.data('assigned_group') || '') !== first);
            const val = (mixed || first === '') ? 'unassigned' : (customGroups[first] ? 'custom_' + first : first);
            if (Array.from(sel.options).some(o => o.value === val)) sel.value = val;
        }}

        function onAssignSelectChange() {{}}

        function assignSelectedToGroup() {{
            const sel = document.getElementById('assignGroupSelect');
            if (!sel) return;
            const value = sel.value;
            const name = value === 'unassigned' ? '' : (value.startsWith('custom_') ? value.slice(7) : value);
            const nodes = cy.$(':selected').nodes();
            if (nodes.length === 0) {{
                showToast('Select nodes to assign (groups are node memberships)', 'warn');
                return;
            }}
            pushHistory('Assign to group');
            assignNodesToGroup(nodes, name, true);
            showToast('Assigned ' + nodes.length + ' node(s) ' + (name ? 'to ' + groupLabel(name) : 'to Unassigned'), 'success');
        }}

        // Rebuild the custom-group optgroup in the Groups dropdown (live
        // counts) and inject the Unassigned entry when it has members.
        function rebuildCustomGroupUI() {{
            const selector = document.getElementById('groupSelector');
            const old = document.getElementById('customGroupOptgroup');
            if (old) old.remove();
            const names = Object.keys(customGroups);
            if (names.length > 0 && selector) {{
                const og = document.createElement('optgroup');
                og.id = 'customGroupOptgroup';
                og.label = 'Custom Groups';
                selector.appendChild(og);
                names.forEach(name => {{
                    const option = document.createElement('option');
                    option.value = 'custom_' + name;
                    option.textContent = name + ' (' + groupMembers(name).length + ')';
                    og.appendChild(option);
                }});
            }}
            const oldUn = document.getElementById('unassignedOption');
            if (oldUn) oldUn.remove();
            const unCount = groupMembers('').length;
            if (unCount > 0 && selector) {{
                const opt = document.createElement('option');
                opt.id = 'unassignedOption';
                opt.value = 'unassigned';
                opt.textContent = 'Unassigned (' + unCount + ')';
                selector.appendChild(opt);
            }}
            updateCustomGroupList();
        }}
        
        // NT color mapping for JavaScript
        const ntColors = {nt_colors_js};
        
        function getNTColor(nt) {{
            if (!nt) return ntColors['unknown'];
            const ntLower = String(nt).trim().toLowerCase();
            return ntColors[ntLower] || ntColors['unknown'];
        }}

        // Update group controls when dropdown changes
        function updateGroupControls() {{
            const group = document.getElementById('groupSelector').value;
            let defaults = groupDefaults[group] || groupDefaults[group.replace('all_', '')] || {{ color: '#888888', opacity: 100 }};
            
            // Handle NT edge groups (prefixed with 'nt_')
            if (group.startsWith('nt_')) {{
                const ntType = group.replace('nt_', '');
                const ntColor = getNTColor(ntType);
                // Use saved color if exists, otherwise default NT color
                defaults = groupDefaults[group] || {{ color: ntColor, opacity: 100 }};
            }}

            // Handle hemisphere groups
            if (group === 'hemi_left' || group === 'hemi_right' || group === 'hemi_unknown') {{
                const keyMap = {{
                    'hemi_left': 'hemisphere_left',
                    'hemi_right': 'hemisphere_right',
                    'hemi_unknown': 'hemisphere_unknown'
                }};
                defaults = groupDefaults[keyMap[group]] || {{ color: '#888888', opacity: 100 }};
            }}
            
            // Unassigned pseudo-group
            if (group === 'unassigned') {{
                defaults = groupDefaults.unassigned || {{ color: '#9ca3af', opacity: 100 }};
            }}

            // Handle custom groups (prefixed with 'custom_')
            if (group.startsWith('custom_')) {{
                const groupName = group.replace('custom_', '');
                if (customGroups[groupName]) {{
                    defaults = {{ color: customGroups[groupName].color, opacity: customGroups[groupName].opacity }};
                }}
            }}
            
            // Update color picker with current group's default/saved color
            document.getElementById('groupColor').value = defaults.color;
            document.getElementById('groupColorText').value = defaults.color;
            document.getElementById('groupOpacity').value = defaults.opacity;
            document.getElementById('groupOpacityValue').textContent = defaults.opacity + '%';
            
            // Update label based on group type
            const label = document.getElementById('groupColorLabel');
            if (group.includes('edge') || group.startsWith('nt_')) {{
                label.textContent = 'Edge Color:';
            }} else if (group.startsWith('custom_')) {{
                label.textContent = 'Element Color:';
            }} else {{
                label.textContent = 'Node Color:';
            }}
        }}

        // Apply color to selected group
        function applyGroupColor() {{
            pushHistory('Color change');
            const group = document.getElementById('groupSelector').value;
            const color = document.getElementById('groupColor').value;
            const opacity = document.getElementById('groupOpacity').value / 100;
            
            console.log('Applying color to group:', group, color, opacity);
            
            const applyNodeGroupColor = (name) => {{
                groupMembers(name).forEach(node => {{
                    if (!node.selected()) {{
                        node.style({{ 'background-color': color, 'opacity': opacity }});
                    }}
                }});
                groupDefaults[name] = {{ color: color, opacity: opacity * 100 }};
            }};
            if (group === 'source') applyNodeGroupColor('source');
            if (group === 'intermediate') applyNodeGroupColor('intermediate');
            if (group === 'target') applyNodeGroupColor('target');
            // All Nodes recolors EVERY node — structural roles, declared
            // dataset groups (which REPLACE the roles in the membership
            // field) and custom groups alike — and records each member
            // group's look so the legend and later group ops stay
            // consistent with Reset All Colors.
            if (group === 'all_nodes') {{
                const memberGroups = [];
                cy.nodes().forEach(node => {{
                    const name = String(node.data('assigned_group') || '');
                    if (!memberGroups.includes(name)) memberGroups.push(name);
                }});
                memberGroups.forEach(name => {{
                    groupMembers(name).forEach(node => {{
                        if (!node.selected()) {{
                            node.style({{ 'background-color': color, 'opacity': opacity }});
                        }}
                    }});
                    if (customGroups[name]) {{
                        customGroups[name].color = color;
                        customGroups[name].opacity = opacity * 100;
                    }} else if (name === '') {{
                        groupDefaults.unassigned = {{ color: color, opacity: opacity * 100 }};
                    }} else {{
                        groupDefaults[name] = {{ color: color, opacity: opacity * 100 }};
                    }}
                }});
            }}
            const extraGroup = extraNodeGroups.find(g => g.name === group);
            if (extraGroup) applyNodeGroupColor(group);
            if (group === 'unassigned') {{
                groupMembers('').forEach(node => {{
                    if (!node.selected()) {{
                        node.style({{ 'background-color': color, 'opacity': opacity }});
                    }}
                }});
                groupDefaults.unassigned = {{ color: color, opacity: opacity * 100 }};
            }}
            if (group === 'hemi_left') {{
                cy.nodes().filter('[hemisphere = "L"]').forEach(node => {{
                    if (!node.selected()) {{
                        node.style({{ 'background-color': color, 'opacity': opacity }});
                    }}
                }});
                groupDefaults.hemisphere_left = {{ color: color, opacity: opacity * 100 }};
            }}
            if (group === 'hemi_right') {{
                cy.nodes().filter('[hemisphere = "R"]').forEach(node => {{
                    if (!node.selected()) {{
                        node.style({{ 'background-color': color, 'opacity': opacity }});
                    }}
                }});
                groupDefaults.hemisphere_right = {{ color: color, opacity: opacity * 100 }};
            }}
            if (group === 'hemi_unknown') {{
                cy.nodes().filter('[hemisphere = "U"]').forEach(node => {{
                    if (!node.selected()) {{
                        node.style({{ 'background-color': color, 'opacity': opacity }});
                    }}
                }});
                groupDefaults.hemisphere_unknown = {{ color: color, opacity: opacity * 100 }};
            }}
            if (group === 'positive_edges' || group === 'all_edges') {{
                cy.edges().filter('[is_negative = 0]').forEach(edge => {{
                    if (!edge.selected() && !edge.hasClass('highlighted')) {{
                        setEdgeBaseAppearance(edge, color, opacity, true);
                    }}
                }});
                groupDefaults.positive_edges = {{ color: color, opacity: opacity * 100 }};
            }}
            if (group === 'negative_edges' || group === 'all_edges') {{
                cy.edges().filter('[is_negative = 1]').forEach(edge => {{
                    if (!edge.selected() && !edge.hasClass('highlighted')) {{
                        setEdgeBaseAppearance(edge, color, opacity, true);
                    }}
                }});
                groupDefaults.negative_edges = {{ color: color, opacity: opacity * 100 }};
            }}
            
            // Handle NT edge groups (prefixed with 'nt_')
            if (group.startsWith('nt_')) {{
                const ntType = group.replace('nt_', '');
                cy.edges().filter(`[nt_type = "${{ntType}}"]`).forEach(edge => {{
                    if (!edge.selected() && !edge.hasClass('highlighted')) {{
                        setEdgeBaseAppearance(edge, color, opacity, true);
                    }}
                }});
                // Store the custom color for this NT type
                groupDefaults[group] = {{ color: color, opacity: opacity * 100 }};
            }}
            
            // Handle custom groups (prefixed with 'custom_'): color the
            // CURRENT live members
            if (group.startsWith('custom_')) {{
                const groupName = group.replace('custom_', '');
                if (customGroups[groupName]) {{
                    groupMembers(groupName).forEach(node => {{
                        if (!node.selected()) {{
                            node.style({{ 'background-color': color, 'opacity': opacity }});
                        }}
                    }});
                    customGroups[groupName].color = color;
                    customGroups[groupName].opacity = opacity * 100;
                }}
            }}
            
            // The legend chips mirror the live group colors/counts
            refreshLegend();
            
            console.log('✓ Color applied to group:', group);
        }}

        // Select all elements in a group
        function selectGroup(group) {{
            // Unselect all first
            cy.elements().unselect();
            
            if (group === 'source') {{
                groupMembers('source').select();
            }} else if (group === 'intermediate') {{
                groupMembers('intermediate').select();
            }} else if (group === 'target') {{
                groupMembers('target').select();
            }} else if (group === 'unassigned') {{
                groupMembers('').select();
            }} else if (extraNodeGroups.some(g => g.name === group)) {{
                groupMembers(group).select();
            }} else if (group === 'hemi_left') {{
                cy.nodes().filter('[hemisphere = "L"]').select();
            }} else if (group === 'hemi_right') {{
                cy.nodes().filter('[hemisphere = "R"]').select();
            }} else if (group === 'hemi_unknown') {{
                cy.nodes().filter('[hemisphere = "U"]').select();
            }} else if (group === 'positive_edges') {{
                cy.edges().filter('[is_negative = 0]').select();
            }} else if (group === 'negative_edges') {{
                cy.edges().filter('[is_negative = 1]').select();
            }} else if (group === 'all_edges') {{
                cy.edges().select();
            }} else if (group === 'all_nodes') {{
                cy.nodes().select();
            }} else if (group.startsWith('nt_')) {{
                // Handle NT edge groups
                const ntType = group.replace('nt_', '');
                cy.edges().filter(`[nt_type = "${{ntType}}"]`).select();
            }} else if (group.startsWith('custom_')) {{
                // Live membership: select the current members
                groupMembers(group.replace('custom_', '')).select();
            }}
            
            // Update dropdown to match
            document.getElementById('groupSelector').value = group;
            updateGroupControls();
            syncAssignSelectToSelection();
            
            // Update selection info
            const selectionCount = getSelectionCount();
            document.getElementById('selectedInfo').innerHTML = 
                `<strong>Group Selected:</strong><br>` +
                `${{selectionCount.nodes}} node(s), ${{selectionCount.edges}} edge(s)`;
            document.getElementById('individualControls').style.display = 'block';
        }}

        // Create a node group from the current selection: membership is LIVE
        // (each node carries assigned_group), so later assignments update
        // counts, legend and dropdown automatically. Edges cannot join —
        // groups are a node concept.
        function createCustomGroup() {{
            const nameInput = document.getElementById('customGroupName');
            let groupName = nameInput.value.trim();
            
            if (!groupName) {{
                groupName = 'Group_' + (Object.keys(customGroups).length + 1);
            }}
            
            // Sanitize name (remove special characters)
            groupName = groupName.replace(/[^a-zA-Z0-9_-]/g, '_');
            
            const selectedNodes = cy.$(':selected').nodes();
            if (selectedNodes.length === 0) {{
                showToast('Please select some nodes first (groups are node memberships)', 'warn');
                return;
            }}
            if (cy.$(':selected').edges().length > 0) {{
                showToast('Edges in the selection were ignored — groups are node memberships', 'info');
            }}
            
            // Seed the group look from the first selected node; the computed
            // style reports rgb(), so normalize to hex for input[type=color].
            // defaultColor is what Reset All Colors restores.
            const color = extractColorHex(selectedNodes[0].style('background-color')) || '#888888';
            
            pushHistory('Assign to group');
            customGroups[groupName] = {{
                label: groupName,
                color: color,
                opacity: 100,
                defaultColor: color,
                defaultOpacity: 100
            }};
            assignNodesToGroup(selectedNodes, groupName, true);

            rebuildCustomGroupUI();
            
            // Select the new group in dropdown
            document.getElementById('groupSelector').value = 'custom_' + groupName;
            updateGroupControls();
            
            nameInput.value = '';
            showToast('Created group: ' + groupName + ' (' + selectedNodes.length + ' nodes)', 'success');
        }}
        
        // Delete selected custom group
        function deleteCustomGroup() {{
            const selector = document.getElementById('groupSelector');
            const currentValue = selector.value;
            
            if (!currentValue.startsWith('custom_')) {{
                showToast('Please select a custom group to delete', 'warn');
                return;
            }}
            
            const groupName = currentValue.replace('custom_', '');
            
            // Groups are NOT history-recorded, so a real (danger) dialog
            // guards the irreversible delete.
            showDialog({{
                title: 'Delete custom group',
                message: 'Delete custom group "' + groupName + '"? Its ' + groupMembers(groupName).length + ' node(s) become Unassigned. This cannot be undone.',
                fields: [],
                confirmText: 'Delete',
                danger: true
            }}).then(() => {{
                // Members fall back to Unassigned and revert to their
                // structural look (node_type never changes, so the target
                // look is stable).
                groupMembers(groupName).forEach(node => {{
                    node.data('assigned_group', '');
                    applyGroupLook(node, groupDefaultFor(String(node.data('node_type') || '')));
                }});
                delete customGroups[groupName];
                rebuildCustomGroupUI();
                rebuildAssignSelect();
                refreshLegend();
                // Reset the picker to a group that actually exists —
                // declared dataset groups replace the structural roles in
                // the dropdown, so 'source' is not always an option.
                if (selector.querySelector('option[value="source"]')) {{
                    selector.value = 'source';
                }} else {{
                    const firstNodeGroup = selector.querySelector('option[value="all_nodes"]');
                    if (firstNodeGroup) selector.value = firstNodeGroup.value;
                }}
                updateGroupControls();
                showToast('Deleted custom group: ' + groupName, 'success');
            }}).catch(() => {{}});
        }}
        
        // Update custom group list display
        function updateCustomGroupList() {{
            const list = document.getElementById('customGroupList');
            list.innerHTML = '<option value="">-- Custom Groups --</option>';
            
            const groupNames = Object.keys(customGroups);
            if (groupNames.length > 0) {{
                list.style.display = 'block';
                groupNames.forEach(name => {{
                    const opt = document.createElement('option');
                    opt.value = 'custom_' + name;
                    opt.textContent = name + ' (' + groupMembers(name).length + ')';
                    list.appendChild(opt);
                }});
            }} else {{
                list.style.display = 'none';
            }}
        }}

        // Reset all colors to defaults (groups and memberships are KEPT —
        // only the look resets; custom groups restore their creation color)
        function applyGlobalColors() {{
            pushHistory('Reset colors');
            const resetNodeGroup = (name, def) => {{
                groupMembers(name).forEach(node => {{
                    applyGroupLook(node, def);
                    node.data('customColor', false);
                }});
            }};
            ['source', 'intermediate', 'target'].forEach(name => {{
                resetNodeGroup(name, originalGroupDefaults[name]);
                groupDefaults[name] = JSON.parse(JSON.stringify(originalGroupDefaults[name]));
            }});
            extraNodeGroups.forEach(g => {{
                resetNodeGroup(g.name, originalGroupDefaults[g.name]);
                groupDefaults[g.name] = JSON.parse(JSON.stringify(originalGroupDefaults[g.name]));
            }});
            // Unassigned nodes revert to the unassigned gray
            const unassignedDef = groupDefaults.unassigned || {{ color: '#9ca3af', opacity: 100 }};
            resetNodeGroup('', unassignedDef);
            
            // Reset positive edges to original default
            cy.edges().filter('[is_negative = 0]').forEach(edge => {{
                setEdgeBaseAppearance(edge, originalGroupDefaults.positive_edges.color, originalGroupDefaults.positive_edges.opacity / 100, true);
                edge.data('customColor', false);
            }});
            // Reset negative edges to original default
            cy.edges().filter('[is_negative = 1]').forEach(edge => {{
                setEdgeBaseAppearance(edge, originalGroupDefaults.negative_edges.color, originalGroupDefaults.negative_edges.opacity / 100, true);
                edge.data('customColor', false);
            }});
            
            // Reset custom-group looks to their creation color; definitions
            // and memberships survive a reset by design.
            Object.keys(customGroups).forEach(name => {{
                customGroups[name].color = customGroups[name].defaultColor || customGroups[name].color;
                customGroups[name].opacity = customGroups[name].defaultOpacity === undefined ? 100 : customGroups[name].defaultOpacity;
                resetNodeGroup(name, customGroups[name]);
            }});
            
            // Clear any NT group custom colors (reset to NT defaults)
            Object.keys(groupDefaults).forEach(key => {{
                if (key.startsWith('nt_')) {{
                    delete groupDefaults[key];
                }}
            }});
            
            // The legend mirrors the reset colors
            refreshLegend();
            
            // Update group controls to current selection
            updateGroupControls();
            
            console.log('✓ All colors reset to defaults');
        }}

        // Handle element selection for individual coloring
        cy.on('tap', 'node, edge', function(evt) {{
            const element = evt.target;
            selectedElement = element;  // Keep for backward compatibility
            
            // Check if multiple elements are selected
            const selectionCount = getSelectionCount();
            
            // Get current color and opacity from the tapped element
            let currentColor = '#3498db';
            let currentOpacity = 100;
            
            if (element.isNode()) {{
                const bgColor = element.style('background-color');
                currentColor = extractColorHex(bgColor);
                const opacity = element.style('opacity');
                currentOpacity = Math.round(parseFloat(opacity || 1) * 100);
                
                if (selectionCount.total > 1) {{
                    document.getElementById('selectedInfo').innerHTML = 
                        `<strong>Multi-Selection:</strong><br>` +
                        `${{selectionCount.nodes}} node(s), ${{selectionCount.edges}} edge(s)<br>` +
                        `<em>Colors from: ${{escapeHtml(element.data('label'))}}</em>`;
                }} else {{
                    document.getElementById('selectedInfo').innerHTML = 
                        `<strong>Node:</strong> ${{escapeHtml(element.data('label'))}} (${{escapeHtml(element.data('node_type'))}})`;
                }}
            }} else {{
                const lineColor = element.style('line-color');
                currentColor = extractColorHex(lineColor);
                const opacity = element.style('opacity');
                currentOpacity = Math.round(parseFloat(opacity || 1) * 100);
                
                const sourceNode = element.source().data('label');
                const targetNode = element.target().data('label');
                
                if (selectionCount.total > 1) {{
                    document.getElementById('selectedInfo').innerHTML = 
                        `<strong>Multi-Selection:</strong><br>` +
                        `${{selectionCount.nodes}} node(s), ${{selectionCount.edges}} edge(s)<br>` +
                        `<em>Colors from: ${{escapeHtml(sourceNode)}} → ${{escapeHtml(targetNode)}}</em>`;
                }} else {{
                    document.getElementById('selectedInfo').innerHTML = 
                        `<strong>Edge:</strong> ${{escapeHtml(sourceNode)}} → ${{escapeHtml(targetNode)}}`;
                }}
            }}
            
            // Update individual color controls
            document.getElementById('individualColor').value = currentColor;
            document.getElementById('individualColorText').value = currentColor;
            document.getElementById('individualOpacity').value = currentOpacity;
            document.getElementById('individualOpacityValue').textContent = currentOpacity + '%';
            document.getElementById('individualControls').style.display = 'block';
            syncAssignSelectToSelection();
            // Populate the size/position inputs for the tapped element
            syncSelectedGeometryInputs(element);
        }});

        // Extract hex color from rgba/rgb string
        function extractColorHex(colorStr) {{
            if (colorStr.startsWith('rgba') || colorStr.startsWith('rgb')) {{
                const matches = colorStr.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (matches) {{
                    const r = parseInt(matches[1]);
                    const g = parseInt(matches[2]);
                    const b = parseInt(matches[3]);
                    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
                }}
            }} else if (colorStr.startsWith('#')) {{
                return colorStr;
            }}
            return '#3498db';
        }}

        // Apply color and opacity to ALL selected elements (supports multi-selection!)
        function applyIndividualColor() {{
            const selectedElements = getSelectedElements();

            if (selectedElements.length === 0) {{
                showToast('Please select one or more nodes/edges first', 'warn');
                return;
            }}
            
            pushHistory('Color change');
            const color = document.getElementById('individualColor').value;
            const opacity = document.getElementById('individualOpacity').value / 100;
            
            let nodesUpdated = 0;
            let edgesUpdated = 0;
            
            // Apply to all selected elements
            selectedElements.forEach(function(element) {{
                if (element.isNode()) {{
                    element.style({{
                        'background-color': color,
                        'opacity': opacity  // CSS opacity property (coana's approach)
                    }});
                    element.data('customColor', true);  // Mark as customized
                    nodesUpdated++;
                }} else {{
                    const canApplyEdgeStyle = !element.selected() && !element.hasClass('highlighted');
                    setEdgeBaseAppearance(element, color, opacity, canApplyEdgeStyle);
                    if (!canApplyEdgeStyle) {{
                        applyEdgeHighlightOverride(element);
                    }}
                    element.data('customColor', true);  // Mark as customized
                    edgesUpdated++;
                }}
            }});
            
            console.log(`Applied color+opacity to ${{nodesUpdated}} node(s) and ${{edgesUpdated}} edge(s):`, color, opacity);
            
            // Update info display
            if (nodesUpdated + edgesUpdated > 1) {{
                document.getElementById('selectedInfo').innerHTML = 
                    `<strong>✓ Updated:</strong><br>` +
                    `${{nodesUpdated}} node(s), ${{edgesUpdated}} edge(s)`;
            }}
        }}

        // Clear selection (all selected elements)
        function clearSelection() {{
            cy.$(':selected').unselect();  // Deselect all elements
            selectedElement = null;
            document.getElementById('selectedInfo').innerHTML = 
                'Click on a node or edge to customize its color<br>' +
                '<em>Hold Shift to select multiple elements</em>';
            document.getElementById('individualControls').style.display = 'none';
            syncSelectedGeometryInputs(null);
        }}

        // ===== GEOMETRY EDITING (precise size / position) =====
        // Numeric editing of the selected elements, next to color editing.
        // Node positions are model coordinates; node size keeps nodes
        // circular (width == height). Multi-selection: position moves every
        // selected node by the delta of the primary (last-tapped) element,
        // size/width applies absolutely to all selected elements.

        // Fill the geometry inputs from the primary element and show the
        // matching node/edge rows; pass null to hide both rows.
        function syncSelectedGeometryInputs(primary) {{
            const nodeGroup = document.getElementById('geomNodeGroup');
            const edgeGroup = document.getElementById('geomEdgeGroup');
            if (!primary) {{
                if (nodeGroup) nodeGroup.style.display = 'none';
                if (edgeGroup) edgeGroup.style.display = 'none';
            }} else if (primary.isNode()) {{
                if (nodeGroup) nodeGroup.style.display = 'block';
                if (edgeGroup) edgeGroup.style.display = 'none';
                const pos = primary.position();
                document.getElementById('selGeomX').value = Math.round(pos.x * 10) / 10;
                document.getElementById('selGeomY').value = Math.round(pos.y * 10) / 10;
                document.getElementById('selGeomSize').value = Math.round(primary.numericStyle('width'));
            }} else {{
                if (nodeGroup) nodeGroup.style.display = 'none';
                if (edgeGroup) edgeGroup.style.display = 'block';
                document.getElementById('selGeomWidth').value = Math.round(primary.numericStyle('width') * 10) / 10;
            }}
            updateAlignButtons();
        }}

        // Align buttons are only enabled with 2+ selected nodes; the
        // size/position modifiers and their confirm button are hidden
        // entirely while nothing is selected (they would otherwise keep
        // showing stale values after a deselect).
        function updateAlignButtons() {{
            const enabled = cy.$('node:selected').length >= 2;
            ['alignHBtn', 'alignVBtn'].forEach(id => {{
                const btn = document.getElementById(id);
                if (btn) btn.style.opacity = enabled ? '1' : '0.4';
            }});
            const anySelected = cy.$(':selected').length > 0;
            const applyGeom = document.getElementById('applyGeometryBtn');
            if (applyGeom) applyGeom.style.display = anySelected ? 'block' : 'none';
            if (!anySelected) {{
                const ng = document.getElementById('geomNodeGroup');
                const eg = document.getElementById('geomEdgeGroup');
                if (ng) ng.style.display = 'none';
                if (eg) eg.style.display = 'none';
            }}
        }}

        // Apply the numeric size/position values to the selection as ONE
        // history entry ('Move nodes' for position-only, 'Resize element'
        // whenever a size/width changes).
        function applySelectedGeometry() {{
            const selected = getSelectedElements();
            if (selected.length === 0) {{
                showToast('Please select one or more nodes/edges first', 'warn');
                return;
            }}
            let primary = selectedElement;
            if (!primary || !primary.selected()) primary = selected[0];

            if (primary.isNode()) {{
                const nodes = selected.nodes();
                const newX = parseFloat(document.getElementById('selGeomX').value);
                const newY = parseFloat(document.getElementById('selGeomY').value);
                const newSize = parseFloat(document.getElementById('selGeomSize').value);
                const pos = primary.position();
                const wantMove = !isNaN(newX) && !isNaN(newY) && (newX !== pos.x || newY !== pos.y);
                const wantResize = !isNaN(newSize) && newSize > 0 && newSize !== primary.numericStyle('width');
                if (!wantMove && !wantResize) return;

                pushHistory(wantResize ? 'Resize element' : 'Move nodes');
                if (wantMove) {{
                    // primary goes to the exact coordinates; every other
                    // selected node shifts by the same delta so their
                    // relative placement survives
                    const dx = newX - pos.x;
                    const dy = newY - pos.y;
                    cy.batch(() => {{
                        nodes.forEach(n => {{
                            if (n === primary) {{
                                n.position({{ x: newX, y: newY }});
                            }} else {{
                                const p = n.position();
                                n.position({{ x: p.x + dx, y: p.y + dy }});
                            }}
                        }});
                    }});
                }}
                if (wantResize) {{
                    cy.batch(() => {{
                        nodes.forEach(n => n.style({{ 'width': newSize + 'px', 'height': newSize + 'px' }}));
                    }});
                }}
                refreshEdgeStyles(false);  // keep endpoints/offsets attached to resized nodes
                updateHoverInfo('✓ Geometry updated' + (wantMove ? ' (position)' : '') + (wantResize ? ' (size)' : ''));
            }} else {{
                const edges = selected.edges();
                const newWidth = parseFloat(document.getElementById('selGeomWidth').value);
                if (isNaN(newWidth) || newWidth <= 0 || newWidth === primary.numericStyle('width')) return;

                pushHistory('Resize element');
                cy.batch(() => {{
                    edges.forEach(e => {{
                        // style bypass wins over the stylesheet mapData used
                        // by the global width sliders; customSize marks the
                        // edge as manually sized
                        e.style('width', newWidth + 'px');
                        e.data('customSize', true);
                    }});
                }});
                updateHoverInfo('✓ Edge width set to ' + newWidth + 'px');
            }}
            syncSelectedGeometryInputs(primary);
        }}

        // Align the selected nodes horizontally (same Y) or vertically
        // (same X) on the mean coordinate of the selection.
        function alignSelectedNodes(axis) {{
            const nodes = cy.$('node:selected');
            if (nodes.length < 2) {{
                showToast('Select at least two nodes to align', 'warn');
                return;
            }}
            pushHistory('Align nodes');
            const coord = (axis === 'h') ? 'y' : 'x';
            let sum = 0;
            nodes.forEach(n => {{ sum += n.position()[coord]; }});
            const mean = sum / nodes.length;
            cy.batch(() => {{
                nodes.forEach(n => {{
                    const p = n.position();
                    n.position(coord === 'x' ? {{ x: mean, y: p.y }} : {{ x: p.x, y: mean }});
                }});
            }});
            refreshEdgeStyles(false);
            const primary = (selectedElement && selectedElement.selected()) ? selectedElement : nodes[0];
            syncSelectedGeometryInputs(primary);
            updateHoverInfo('✓ Aligned ' + nodes.length + ' nodes ' +
                (axis === 'h' ? 'horizontally (same Y)' : 'vertically (same X)'));
        }}

        // Apply initial opacity values from parsed input colors
        function applyInitialOpacity() {{
            const sourceOpacity = {self.source_opacity};
            const intermediateOpacity = {self.intermediate_opacity};
            const targetOpacity = {self.target_opacity};
            const edgeOpacity = {self.edge_opacity};
            
            console.log('Applying initial opacity values:', {{
                source: sourceOpacity,
                intermediate: intermediateOpacity,
                target: targetOpacity,
                edge: edgeOpacity
            }});
            
            // Apply initial opacity to all nodes based on their type
            cy.nodes().forEach(function(node) {{
                const nodeType = node.data('node_type');
                if (nodeType === 'source') {{
                    node.style('opacity', sourceOpacity);
                }} else if (nodeType === 'intermediate') {{
                    node.style('opacity', intermediateOpacity);
                }} else if (nodeType === 'target') {{
                    node.style('opacity', targetOpacity);
                }}
            }});
            
            // Apply initial opacity to all edges
            cy.edges().forEach(function(edge) {{
                edge.style('opacity', edgeOpacity);
                const currentColor = edge.style('line-color');
                setEdgeBaseAppearance(edge, currentColor, edgeOpacity, false);
            }});
            
            console.log('✓ Initial opacity applied from input colors (nodes + edges)');
        }}

        // Initialize log base visibility on load
        function initializeLogBaseVisibility() {{
            const scalingMethod = document.getElementById('edgeWidthScale').value;
            const logBaseGroup = document.getElementById('logBaseGroup');
            // The log-base picker group is not rendered in this template;
            // guard so initialization never throws on page load.
            if (!logBaseGroup) {{
                return;
            }}
            if (scalingMethod === 'log') {{
                logBaseGroup.style.display = 'flex';
            }} else {{
                logBaseGroup.style.display = 'none';
            }}
        }}

        // ===== INTERACTIVE EDITING FEATURES =====
        
        let editMode = false;
        let edgeDrawMode = false;
        let sourceNodeForEdge = null;
        let nextNodeId = 1000;  // Start custom node IDs from 1000
        let nextEdgeId = 1000;  // Start custom edge IDs from 1000
        
        // Default edge color and opacity from link_color parameter
        const defaultEdgeColor = '{self.edge_color}';
        const defaultEdgeOpacity = {self.edge_opacity};
        
        // Edge filtering variables
        let ignoredEdges = new Set();  // Set of exact weight values to ignore
        let ignoredEdgeExpressions = [];  // Array of comparison expressions for edges
        
        // Toggle edit mode
        function toggleEditMode() {{
            editMode = !editMode;
            const btn = document.getElementById('editModeBtn');
            const controls = document.getElementById('editControls');
            
            if (editMode) {{
                btn.textContent = '🔒 Disable Edit Mode';
                btn.style.background = '#f44336';
                controls.style.display = 'block';
                // Canvas affordance: dashed accent border + mode badge
                document.getElementById('cy').classList.add('vp-editmode');
                document.getElementById('editBadge').style.display = 'block';
                
                // Enable node dragging in edit mode
                cy.autoungrabify(false);
                
                // Add click handlers for edge drawing (use namespace to avoid conflicts)
                cy.on('tap.editmode', 'node', function(evt) {{
                    if (editMode) {{
                        handleNodeClickForEdge(evt.target);
                    }}
                }});
                
                // Add double-click to edit properties (use namespace)
                cy.on('dbltap.editmode', 'node', function(evt) {{
                    if (editMode) {{
                        editNodeProperties(evt.target);
                    }}
                }});
                
                cy.on('dbltap.editmode', 'edge', function(evt) {{
                    if (editMode) {{
                        editEdgeProperties(evt.target);
                    }}
                }});
                
                // Note: Right-click handlers are managed globally above (lines ~4013-4040)
                // They check editMode status to decide between hide/delete
                
            }} else {{
                btn.textContent = '✏️ Enable Edit Mode';
                btn.style.background = '#ff9800';
                controls.style.display = 'none';
                document.getElementById('cy').classList.remove('vp-editmode');
                document.getElementById('editBadge').style.display = 'none';
                
                // Disable special edit handlers only (use namespace to preserve main handlers)
                cy.off('tap.editmode');
                cy.off('dbltap.editmode');
                edgeDrawMode = false;
                sourceNodeForEdge = null;
                
                // Remove any temporary indicators
                cy.nodes().removeClass('edge-source');
            }}
        }}
        
        // Edit node properties (double-click): in-page dialog with label,
        // type and an explicit color (defaults to the type color).
        function editNodeProperties(node) {{
            if (!node || node.length === 0) return;  // nothing to edit
            const currentId = node.id();
            const currentType = node.data('node_type');
            const typeColor = currentType === 'source' ? '{self.node_color[0]}'
                : (currentType === 'target' ? '{self.target_color}' : '{self.node_color[1]}');
            const currentColor = extractColorHex(node.style('background-color')) || node.data('color') || typeColor;

            showDialog({{
                title: 'Edit node: ' + currentId,
                fields: [
                    {{ key: 'label', label: 'Label', type: 'text', value: node.data('label') }},
                    {{ key: 'type', label: 'Type', type: 'select', value: currentType,
                       options: ['source', 'intermediate', 'target'] }},
                    {{ key: 'color', label: 'Color', type: 'color', value: currentColor }},
                ],
                confirmText: 'Apply',
            }}).then(values => {{
                const newLabel = String(values.label);
                const newType = String(values.type);
                const newColor = String(values.color || typeColor);

                // Update node data
                pushHistory('Edit node');
                node.data('label', newLabel);
                node.data('node_type', newType);
                node.data('color', newColor);
                node.style('background-color', newColor);

                showToast('✓ Updated node: ' + currentId + ' → label="' + newLabel + '", type=' + newType, 'success');
            }}).catch(() => {{}});
        }}
        
        // Edit edge properties (double-click): one in-page dialog for
        // weight, ratio and probability (negative weight preserved).
        function editEdgeProperties(edge) {{
            const source = edge.source().id();
            const target = edge.target().id();
            const currentWeight = edge.data('original_weight') || edge.data('weight') || 1;
            const currentRatio = edge.data('ratio') || '';
            const currentProb = edge.data('probability') || '';
        
            showDialog({{
                title: 'Edit edge: ' + source + ' → ' + target,
                fields: [
                    {{ key: 'weight', label: 'Weight', type: 'number', value: currentWeight }},
                    {{ key: 'ratio', label: 'Connection ratio (optional)', type: 'text', value: currentRatio }},
                    {{ key: 'probability', label: 'Traversal probability (optional)', type: 'text', value: currentProb }},
                ],
                confirmText: 'Apply',
            }}).then(values => {{
                const weightNum = parseFloat(values.weight);
                if (isNaN(weightNum)) {{
                    showToast('Invalid weight value. Must be a number.', 'error');
                    return;
                }}
                let newRatio = String(values.ratio === undefined || values.ratio === null ? '' : values.ratio).trim();
                let newProb = String(values.probability === undefined || values.probability === null ? '' : values.probability).trim();
                if (newRatio !== '') {{
                    const r = parseFloat(newRatio);
                    if (isNaN(r)) newRatio = currentRatio; else newRatio = r;
                }}
                if (newProb !== '') {{
                    const p = parseFloat(newProb);
                    if (isNaN(p)) newProb = currentProb; else newProb = p;
                }}
        
                // Update edge data
                pushHistory('Edit edge');
                edge.data('weight', Math.abs(weightNum));
                edge.data('original_weight', weightNum);
                edge.data('is_negative', weightNum < 0 ? 1 : 0);
        
                if (newRatio !== '') {{
                    edge.data('ratio', newRatio);
                }}
                if (newProb !== '') {{
                    edge.data('probability', newProb);
                }}
        
                // Update tooltip
                const tooltipParts = [`Weight: ${{weightNum}}`];
                if (newRatio !== '' && !isNaN(newRatio)) {{
                    tooltipParts.push(`Ratio: ${{newRatio.toFixed(3)}}`);
                }}
                if (newProb !== '' && !isNaN(newProb)) {{
                    tooltipParts.push(`Probability: ${{newProb.toFixed(3)}}`);
                }}
                edge.data('tooltip', tooltipParts.join('\\n'));
        
                // Update visual properties (edge width and color)
                // Recalculate scaled width based on current scaling method
                const scaledWidth = calculateEdgeWidth(Math.abs(weightNum));
                edge.data('scaled_width', scaledWidth);
        
                // The edge-color pickers (edgeColor/negativeEdgeColor) only exist
                // in the Sankey template; keep the edge's current color instead,
                // preferring its NT color when one is available.
                const edgeNT = edge.data('nt_type') || '';
                const currentColor = extractColorHex(edge.style('line-color'));
                const updatedColor = edgeNT ? getNTColor(edgeNT) : currentColor;
                const currentOpacity = edge.data(EDGE_BASE_OPACITY_KEY) !== undefined ? edge.data(EDGE_BASE_OPACITY_KEY) : (parseFloat(edge.style('opacity')) || 1);
                const canApplyNow = !edge.selected() && !edge.hasClass('highlighted');
                setEdgeBaseAppearance(edge, updatedColor, currentOpacity, canApplyNow);
                if (!canApplyNow) {{
                    applyEdgeHighlightOverride(edge);
                }}
        
                // Apply the width update
                updateEdgeWidths();
        
                showToast('✓ Updated edge: ' + source + ' → ' + target + ' (weight=' + weightNum + ')', 'success');
            }}).catch(() => {{}});
        }}
        
        // Calculate edge width based on weight and current scaling method
        function calculateEdgeWidth(weight) {{
            const scalingMethod = document.getElementById('edgeWidthScale').value;
            const baseWidth = parseFloat(document.getElementById('edgeWidthSlider').value);
            
            let scaledValue = weight;
            
            switch(scalingMethod) {{
                case 'linear':
                    scaledValue = weight;
                    break;
                case 'log_e':
                    scaledValue = weight > 0 ? Math.log(weight + 1) : 0;
                    break;
                case 'log_2':
                    scaledValue = weight > 0 ? Math.log2(weight + 1) : 0;
                    break;
                case 'log_10':
                    scaledValue = weight > 0 ? Math.log10(weight + 1) : 0;
                    break;
                case 'sqrt':
                    scaledValue = Math.sqrt(weight);
                    break;
                case 'none':
                    return baseWidth;
            }}
            
            // Scale to reasonable range (1-10 times base width)
            const maxDataValue = Math.max(...cy.edges().map(e => Math.abs(e.data('weight') || 1)));
            const normalized = maxDataValue > 0 ? scaledValue / maxDataValue : 0;
            return Math.max(0.5, baseWidth * (0.3 + normalized * 2));
        }}
        
        // Handle node click for edge drawing
        function handleNodeClickForEdge(node) {{
            if (!edgeDrawMode) {{
                // First click: select source node
                sourceNodeForEdge = node;
                edgeDrawMode = true;
                cy.nodes().removeClass('edge-source');
                node.addClass('edge-source');
                updateHoverInfo('Selected source: ' + node.id() + '. Click target node to create edge.');
            }} else {{
                // Second click: create edge to target
                if (node.id() !== sourceNodeForEdge.id()) {{
                    const sourceId = sourceNodeForEdge.id();
                    const targetId = node.id();
                    
                    // Check if edge already exists
                    const existingEdge = cy.edges(`[source = "${{sourceId}}"][target = "${{targetId}}"]`);
                    if (existingEdge.length > 0) {{
                        updateHoverInfo('⚠️ Edge already exists: ' + sourceId + ' → ' + targetId);
                        // Reset edge draw mode
                        edgeDrawMode = false;
                        sourceNodeForEdge = null;
                        cy.nodes().removeClass('edge-source');
                        return;
                    }}
                    
                    const edgeId = 'e' + nextEdgeId++;
                    pushHistory('Add edge');
                    const newEdge = cy.add({{
                        group: 'edges',
                        data: {{
                            id: edgeId,
                            source: sourceId,
                            target: targetId,
                            weight: 1,
                            original_weight: 1,
                            is_negative: 0,
                            scaled_width: 3,
                            tooltip: 'Weight: 1'
                        }}
                    }});
                    
                    // Apply default edge color and opacity to newly created edge
                    setEdgeBaseAppearance(newEdge, defaultEdgeColor, defaultEdgeOpacity, true);
                    
                    updateHoverInfo('Edge created: ' + sourceId + ' → ' + targetId);
                }} else {{
                    updateHoverInfo('Cannot create self-loop edge');
                }}
                
                // Reset edge draw mode
                edgeDrawMode = false;
                sourceNodeForEdge = null;
                cy.nodes().removeClass('edge-source');
            }}
        }}
        
        // Add new node: in-page dialog for id + type
        // Group options for the Add-node dialog: declared dataset groups
        // REPLACE the structural roles in the membership field, so the
        // dropdown mirrors rebuildAssignSelect's value set instead of
        // hardcoding the structural roles (an assigned group with no
        // dropdown entry or legend chip would be unreachable in the UI).
        function groupFieldOptions() {{
            const opts = [''];
            if (!declaredGroupsActive) opts.push('source', 'intermediate', 'target');
            extraNodeGroups.forEach(g => opts.push(g.name));
            return opts;
        }}

        function addNode() {{
            if (!editMode) {{
                showToast('Please enable Edit Mode first', 'warn');
                return;
            }}

            showDialog({{
                title: 'Add node',
                fields: [
                    {{ key: 'id', label: 'Node ID', type: 'text', value: '', placeholder: 'e.g. Neuron_X' }},
                    {{ key: 'type', label: 'Type', type: 'select', value: 'intermediate',
                       options: ['source', 'intermediate', 'target'] }},
                    {{ key: 'group', label: 'Group', type: 'select', value: '',
                       options: groupFieldOptions() }},
                ],
                confirmText: 'Add',
            }}).then(values => {{
                const nodeId = String(values.id || '').trim();
                if (!nodeId) return;

                // Check if node already exists
                if (cy.getElementById(nodeId).length > 0) {{
                    showToast('Node with ID "' + nodeId + '" already exists', 'error');
                    return;
                }}

                const nodeType = String(values.type || 'intermediate');

                // Determine color based on type
                let color = '{self.node_color[1]}';  // intermediate default
                if (nodeType === 'source') {{
                    color = '{self.node_color[0]}';
                }} else if (nodeType === 'target') {{
                    color = '{self.target_color}';
                }}

                // Add node at center of viewport
                const extent = cy.extent();
                const centerX = (extent.x1 + extent.x2) / 2;
                const centerY = (extent.y1 + extent.y2) / 2;

                pushHistory('Add node');
                const assignedGroup = (values.group === undefined || values.group === null || values.group === '')
                    ? nodeType : String(values.group);
                cy.add({{
                    group: 'nodes',
                    data: {{
                        id: nodeId,
                        label: nodeId,
                        node_type: nodeType,
                        assigned_group: assignedGroup,
                        color: color
                    }},
                    position: {{ x: centerX, y: centerY }}
                }});
                rebuildAssignSelect();
                rebuildCustomGroupUI();
                refreshLegend();

                showToast('Node added: ' + nodeId + ' (' + nodeType + ')', 'success');
            }}).catch(() => {{}});
        }}
        
        // Delete selected element(s): history-recorded, so no confirmation —
        // a toast with the undo hint is enough.
        function deleteSelected() {{
            if (!editMode) {{
                showToast('Please enable Edit Mode first', 'warn');
                return;
            }}
            
            const selected = cy.$(':selected');
            if (selected.length === 0) {{
                showToast('No elements selected. Click to select nodes or edges.', 'info');
                return;
            }}
            
            pushHistory('Delete selection');
            cy.remove(selected);
            showToast('Deleted ' + selected.length + ' element(s) — ⌘Z/⌃Z to undo', 'success');
        }}
        
        // Delete single element (right-click in edit mode); undoable.
        function deleteElement(element) {{
            const type = element.isNode() ? 'node' : 'edge';
            const id = element.id();
            pushHistory('Delete ' + type);
            cy.remove(element);
            showToast('Deleted ' + type + ': ' + id + ' — ⌘Z/⌃Z to undo', 'success');
        }}
        
        // Update hover info display — LEGACY status entry point. Every
        // operation message is a toast now; the hover box itself shows
        // element tooltips only (the mouseover handlers write it directly).
        function updateHoverInfo(text) {{
            showToast(text);
        }}
        
        // ===== EDGE FILTERING =====
        // Supports same AND/OR logic as heatmap filter:
        //   - OR logic: <5, >100  (comma-separated, any condition matches)
        //   - AND logic: (>=5, <=10)  (parentheses, all conditions must match)
        
        let edgeFilterGroups = [];  // Parsed filter groups for edge filtering
        
        // (Re)parse the edge-filter input into edgeFilterGroups.
        function parseEdgeFilterInput() {{
            const input = document.getElementById('ignoreEdgesInput');
            const filterValue = input ? input.value.trim() : '';
            ignoredEdges.clear();
            ignoredEdgeExpressions = [];
            edgeFilterGroups = [];
            if (filterValue) {{
                edgeFilterGroups = parseEdgeFilterExpressions(filterValue);
            }}
        }}

        // Update ignored edges based on input; every distinct filter value is
        // recorded in the history so undo steps back through filter changes.
        function updateIgnoredEdges() {{
            const input = document.getElementById('ignoreEdgesInput');
            const filterValue = input ? input.value.trim() : '';

            if (filterValue !== lastFilterHistoryValue) {{
                pushHistory('Edge filter');
                lastFilterHistoryValue = filterValue;
            }}

            parseEdgeFilterInput();
            applyEdgeFilter();
        }}
        
        function parseEdgeSingleExpression(expr) {{
            // Parse a single comparison expression like ">5" or "<=10"
            const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
            if (compMatch) {{
                return {{ operator: compMatch[1], threshold: parseFloat(compMatch[2]) }};
            }}
            // Try to parse as exact number
            const num = parseFloat(expr);
            if (!isNaN(num)) {{
                return {{ operator: '==', threshold: num }};
            }}
            return null;
        }}
        
        function parseEdgeFilterExpressions(inputString) {{
            // Returns an array of filter groups (same logic as heatmap)
            const result = [];
            let remaining = inputString.trim();
            
            while (remaining.length > 0) {{
                // Skip leading commas and whitespace
                remaining = remaining.replace(/^[,\\s]+/, '');
                if (remaining.length === 0) break;
                
                if (remaining.startsWith('(')) {{
                    // AND group: find matching closing parenthesis
                    const closeIdx = remaining.indexOf(')');
                    if (closeIdx === -1) {{
                        const inner = remaining.substring(1);
                        const andExprs = inner.split(',').map(e => e.trim()).filter(e => e !== '');
                        const parsed = andExprs.map(e => parseEdgeSingleExpression(e)).filter(e => e !== null);
                        if (parsed.length > 0) {{
                            result.push({{ type: 'AND', expressions: parsed }});
                        }}
                        break;
                    }} else {{
                        const inner = remaining.substring(1, closeIdx);
                        const andExprs = inner.split(',').map(e => e.trim()).filter(e => e !== '');
                        const parsed = andExprs.map(e => parseEdgeSingleExpression(e)).filter(e => e !== null);
                        if (parsed.length > 0) {{
                            result.push({{ type: 'AND', expressions: parsed }});
                        }}
                        remaining = remaining.substring(closeIdx + 1);
                    }}
                }} else {{
                    // Single expression (OR)
                    const nextComma = remaining.indexOf(',');
                    const nextParen = remaining.indexOf('(');
                    let endIdx = remaining.length;
                    
                    if (nextComma !== -1 && (nextParen === -1 || nextComma < nextParen)) {{
                        endIdx = nextComma;
                    }} else if (nextParen !== -1 && (nextComma === -1 || nextParen < nextComma)) {{
                        endIdx = nextParen;
                    }}
                    
                    const expr = remaining.substring(0, endIdx).trim();
                    if (expr.length > 0) {{
                        const parsed = parseEdgeSingleExpression(expr);
                        if (parsed !== null) {{
                            result.push({{ type: 'OR', expression: parsed }});
                        }}
                    }}
                    remaining = remaining.substring(endIdx);
                }}
            }}
            
            return result;
        }}
        
        function evaluateEdgeCondition(value, expr) {{
            switch (expr.operator) {{
                case '>': return value > expr.threshold;
                case '<': return value < expr.threshold;
                case '>=': return value >= expr.threshold;
                case '<=': return value <= expr.threshold;
                case '==': return value === expr.threshold;
                case '!=': return value !== expr.threshold;
                default: return false;
            }}
        }}
        
        // Check if an edge weight should be ignored
        function shouldIgnoreEdge(weight) {{
            if (edgeFilterGroups.length === 0) return false;
            
            // OR logic between groups: return true if ANY group matches
            for (const group of edgeFilterGroups) {{
                if (group.type === 'AND') {{
                    // AND logic within group: ALL expressions must match
                    let allMatch = true;
                    for (const expr of group.expressions) {{
                        if (!evaluateEdgeCondition(weight, expr)) {{
                            allMatch = false;
                            break;
                        }}
                    }}
                    if (allMatch) return true;
                }} else {{
                    // Single OR expression
                    if (evaluateEdgeCondition(weight, group.expression)) {{
                        return true;
                    }}
                }}
            }}
            
            return false;
        }}
        
        // Value of the ACTIVE connection metric for one edge — the filter
        // expressions evaluate against this (weight keeps its sign via
        // original_weight; ratio/probability default to 0 when absent).
        function metricEdgeValue(edge) {{
            if (currentMetric === 'ratio') return Number(edge.data('ratio')) || 0;
            if (currentMetric === 'probability') return Number(edge.data('probability')) || 0;
            return edge.data('original_weight') !== undefined
                ? edge.data('original_weight')
                : edge.data('weight') || 0;
        }}

        // Apply edge filter to show/hide edges
        function applyEdgeFilter() {{
            let hiddenCount = 0;
            let shownCount = 0;
            
            cy.edges().forEach(edge => {{
                if (shouldIgnoreEdge(metricEdgeValue(edge))) {{
                    // Use 'filtered' class instead of display:none
                    edge.addClass('filtered');
                    hiddenCount++;
                }} else {{
                    // Remove 'filtered' class
                    edge.removeClass('filtered');
                    shownCount++;
                }}
            }});
            
            // Re-detect dead ends and orphans if their hiding is active:
            // dead ends first so orphan detection sees the current graph
            // with fresh dead-end classes.
            reapplyDeadEndHiding();
            reapplyOrphanHiding();
            
            if (hiddenCount > 0) {{
                updateHoverInfo(`🔍 Edge filter: ${{shownCount}} shown, ${{hiddenCount}} hidden`);
            }} else {{
                updateHoverInfo('🔍 Edge filter: All edges visible');
            }}
        }}
        
        // ===== ORPHAN NODE CONTROLS =====
        
        let orphansHidden = false;
        let selfLoopsHidden = false;
        let deadEndsHidden = false;
        
        // Toggle self-loop edges visibility (edges where source === target)
        function toggleSelfLoops() {{
            pushHistory('Toggle self-loops');
            const btn = document.getElementById('hideSelfLoopsBtn');
            selfLoopsHidden = !selfLoopsHidden;
            
            // Find all self-loop edges
            const selfLoopEdges = cy.edges().filter(edge => {{
                return edge.source().id() === edge.target().id();
            }});
            
            if (selfLoopsHidden) {{
                selfLoopEdges.addClass('selfloop-hidden');
                btn.textContent = '👁️ Show Self-Loops';
                btn.style.background = '#e91e63';
                updateHoverInfo(`🔁 Hidden ${{selfLoopEdges.length}} self-loop edge(s)`);
            }} else {{
                selfLoopEdges.removeClass('selfloop-hidden');
                btn.textContent = '🔁 Hide Self-Loops';
                btn.style.background = '#ff5722';
                updateHoverInfo(`👁️ ${{selfLoopEdges.length}} self-loop edge(s) visible`);
            }}
            // Changing the visible graph can expose new dead ends
            reapplyDeadEndHiding();
        }}
        
        // Toggle orphan nodes visibility (dynamically detect based on current
        // visible edges). Orphans are also dead ends: when dead-end hiding is
        // active they get covered there, and this toggle re-checks after any
        // graph change. Nodes with only self-loop edges are orphans.
        function toggleOrphanNodes() {{
            pushHistory('Toggle orphans');
            const btn = document.getElementById('hideOrphansBtn');
            orphansHidden = !orphansHidden;
            
            if (orphansHidden) {{
                // Hide orphan nodes (nodes with no VISIBLE non-self-loop connections)
                let hiddenCount = 0;
                cy.nodes().forEach(node => {{
                    if (isOrphanNode(node)) {{
                        node.addClass('orphan-hidden');
                        hiddenCount++;
                    }} else {{
                        // Remove orphan-hidden class if node has visible connections
                        node.removeClass('orphan-hidden');
                    }}
                }});
                
                btn.textContent = '👁️ Show Orphans';
                btn.style.background = '#e91e63';
                updateHoverInfo(`👻 Hidden ${{hiddenCount}} orphan node(s) (based on visible edges)`);
            }} else {{
                // Show all orphan nodes
                cy.nodes('.orphan-hidden').removeClass('orphan-hidden');
                btn.textContent = '👻 Hide Orphans';
                btn.style.background = '#9c27b0';
                updateHoverInfo('👁️ All nodes visible');
            }}
            // Changing the visible graph can expose new dead ends (orphans are
            // dead ends too, so the dead-end pass re-runs as well)
            reapplyDeadEndHiding();
        }}

        // Edges that belong to the CURRENT graph: not manually hidden, not
        // filter-hidden, not self-loop-hidden and not dead-end-hidden.
        function isEdgeInCurrentGraph(e) {{
            return !e.hasClass('hidden') && !e.hasClass('filtered') &&
                   !e.hasClass('selfloop-hidden') && !e.hasClass('deadend-hidden');
        }}

        // A node is an orphan when it has NO connections in the current
        // graph. Self-loop edges do not count as connections (a node whose
        // only edges are self-loops is an orphan).
        function isOrphanNode(node) {{
            const visibleEdges = node.connectedEdges().filter(e => {{
                return isEdgeInCurrentGraph(e) && e.source().id() !== e.target().id();
            }});
            return visibleEdges.length === 0;
        }}

        // A node is a dead end in the CURRENT graph when all its visible
        // edges go one way and it is not declared as that endpoint type:
        // out-only non-source, in-only non-target, OR an orphan (a node with
        // no visible connections - orphans are a kind of dead end). Edges to
        // nodes that are already hidden dead ends are removed from the
        // visible graph, so hiding propagates to newly exposed dead ends.
        function isDeadEndNodeIn(node, deadEndSet) {{
            const nodeType = node.data('node_type') || 'intermediate';
            let outCount = 0;
            let inCount = 0;
            node.connectedEdges().forEach(e => {{
                if (e.hasClass('hidden') || e.hasClass('filtered') || e.hasClass('selfloop-hidden')) return;
                if (deadEndSet.has(e.source().id()) || deadEndSet.has(e.target().id())) return;
                if (e.source().id() === e.target().id()) return;  // skip self-loops
                if (e.source().id() === node.id()) outCount++;
                else inCount++;
            }});
            if (outCount === 0 && inCount === 0) return true;  // orphan: no connections in the current graph
            if (nodeType !== 'source' && outCount > 0 && inCount === 0) return true;
            if (nodeType !== 'target' && inCount > 0 && outCount === 0) return true;
            return false;
        }}

        // Recompute dead ends on the current graph (visible edges after the
        // hide / filter / self-loop toggles) and hide them together with
        // their related edges. Hiding edges can create new dead ends, so
        // the detection iterates to a fixpoint. All nodes are evaluated
        // against the SAME dead-end set per pass and additions are applied
        // in batch afterwards, so the result does not depend on the node
        // iteration order (e.g. a chain A→B→C of intermediates must hide
        // both A and C, whichever is visited first).
        function recomputeDeadEnds() {{
            const deadEndSet = new Set();
            let changed = true;
            while (changed) {{
                changed = false;
                const newlyDead = [];
                cy.nodes().forEach(node => {{
                    if (deadEndSet.has(node.id())) return;
                    if (isDeadEndNodeIn(node, deadEndSet)) {{
                        newlyDead.push(node.id());
                        changed = true;
                    }}
                }});
                newlyDead.forEach(id => deadEndSet.add(id));
            }}
            let hiddenNodes = 0;
            let hiddenEdges = 0;
            cy.nodes().forEach(node => {{
                if (deadEndSet.has(node.id())) {{
                    node.addClass('deadend-hidden');
                    hiddenNodes++;
                }} else {{
                    node.removeClass('deadend-hidden');
                }}
            }});
            cy.edges().forEach(edge => {{
                if (deadEndSet.has(edge.source().id()) || deadEndSet.has(edge.target().id())) {{
                    edge.addClass('deadend-hidden');
                    hiddenEdges++;
                }} else {{
                    edge.removeClass('deadend-hidden');
                }}
            }});
            return {{ nodes: hiddenNodes, edges: hiddenEdges }};
        }}

        // Toggle dead-end visibility: hides out-only non-source / in-only
        // non-target nodes AND their related edges, following the current
        // graph (i.e. what the hide-edges filter leaves visible).
        function toggleDeadEnds() {{
            pushHistory('Toggle dead-ends');
            const btn = document.getElementById('hideDeadEndsBtn');
            deadEndsHidden = !deadEndsHidden;

            if (deadEndsHidden) {{
                const counts = recomputeDeadEnds();
                btn.textContent = '👁️ Show Dead Ends';
                btn.style.background = '#e91e63';
                updateHoverInfo(`💀 Hidden ${{counts.nodes}} dead-end node(s) and ${{counts.edges}} edge(s)`);
            }} else {{
                cy.elements().removeClass('deadend-hidden');
                btn.textContent = '💀 Hide Dead Ends';
                btn.style.background = '#607d8b';
                updateHoverInfo('👁️ All nodes visible');
                // Turning dead-end hiding off reveals orphan nodes that were
                // covered by it; re-apply the orphan toggle if it is active.
                reapplyOrphanHiding();
            }}
        }}

        // Re-detect dead ends after the graph changes (edge filter or hide
        // operations); a no-op while dead-end hiding is off.
        function reapplyDeadEndHiding() {{
            if (!deadEndsHidden) return;
            recomputeDeadEnds();
        }}

        // Re-detect orphan nodes after the graph changes; a no-op while
        // orphan hiding is off. Must run after reapplyDeadEndHiding so the
        // current graph (including dead-end-hidden edges) is consistent.
        // Orphans are also dead ends: nodes already deadend-hidden are left
        // to the dead-end pass (avoids redundant class churn).
        function reapplyOrphanHiding() {{
            if (!orphansHidden) return;
            cy.nodes().forEach(node => {{
                if (node.hasClass('deadend-hidden')) return;
                if (isOrphanNode(node)) {{
                    node.addClass('orphan-hidden');
                }} else {{
                    node.removeClass('orphan-hidden');
                }}
            }});
        }}
        
        // Refresh layout after hiding orphans or filtering edges. Only
        // VISIBLE elements participate (class-based - see isVisibleElement).
        function refreshLayout() {{
            pushHistory('Refresh layout');
            const visibleElements = cy.elements().filter(isVisibleElement);
            
            if (visibleElements.length === 0) {{
                updateHoverInfo('⚠️ No visible elements to layout');
                return;
            }}
            
            updateHoverInfo('🔄 Refreshing layout...');
            
            // Use the currently selected layout algorithm
            const layoutConfig = getLayoutConfig(currentLayoutAlgorithm);
            
            // Apply layout only to visible elements
            visibleElements.layout(layoutConfig).run();
            resetLayoutTransformTrackers();
            
            setTimeout(() => {{
                cacheHemispherePositions();
                if (hemisphereMirrorEnabled) runHemisphereMirrorLayout();
                updateHoverInfo('✓ Layout refreshed for visible elements');
            }}, 600);
        }}
        
        // ===== EXPORT/IMPORT GRAPH =====
        
        // Export graph data and layout
        function exportGraph() {{
            // Collect all nodes with their data and positions
            const nodesData = [];
            cy.nodes().forEach(node => {{
                nodesData.push({{
                    data: node.data(),
                    position: node.position(),
                    classes: node.hasClass('hidden') ? ['hidden'] : [],  // Store hidden class
                    style: {{
                        'background-color': node.style('background-color'),
                        'opacity': parseFloat(node.style('opacity'))
                    }}
                }});
            }});
            
            // Collect all edges with their data and classes
            const edgesData = [];
            cy.edges().forEach(edge => {{
                const classes = [];
                if (edge.hasClass('hidden')) classes.push('hidden');
                if (edge.hasClass('filtered')) classes.push('filtered');
                
                edgesData.push({{
                    data: edge.data(),
                    classes: classes,  // Store classes (hidden/filtered)
                    style: {{
                        'line-color': edge.style('line-color'),
                        'opacity': parseFloat(edge.style('opacity'))
                    }}
                }});
            }});
            
            // Collect current settings (with safe null checks)
            const settings = {{
                edgeFilter: {{
                    inputValue: document.getElementById('ignoreEdgesInput')?.value || '',
                    ignoredValues: Array.from(ignoredEdges),
                    expressions: ignoredEdgeExpressions
                }},
                edgeWidthScaling: {{
                    method: document.getElementById('edgeWidthScale')?.value || 'linear',
                    width: parseFloat(document.getElementById('edgeWidthSlider')?.value || 3)
                }},
                arrowSize: parseFloat(document.getElementById('arrowSizeSlider')?.value || 9),
                fontSize: parseFloat(document.getElementById('fontSizeSlider')?.value || 12),
                nodeSize: parseFloat(document.getElementById('nodeSizeSlider')?.value || 40),
                groupDefaults: JSON.parse(JSON.stringify(groupDefaults)),
                customGroups: JSON.parse(JSON.stringify(customGroups)),
                labelFontColor: customLabelColor || ''
            }};
            
            // Create export object
            const exportData = {{
                version: '2.1',
                timestamp: new Date().toISOString(),
                nodes: nodesData,
                edges: edgesData,
                settings: settings,
                metadata: {{
                    nodeCount: nodesData.length,
                    edgeCount: edgesData.length,
                    customGroupCount: Object.keys(customGroups).length
                }}
            }};
            
            // Download as JSON
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'network_graph_' + new Date().toISOString().slice(0,10) + '.json';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            updateHoverInfo('✓ Exported graph with settings: ' + nodesData.length + ' nodes, ' + edgesData.length + ' edges' + (Object.keys(customGroups).length > 0 ? ', ' + Object.keys(customGroups).length + ' custom groups' : ''));
        }}
        
        // ===== EDGE LIST CSV EXPORT =====
        // Exports every edge of the CURRENT graph (in-HTML edits included:
        // changed weights/colors, added/deleted edges, relabeled nodes) as a
        // CSV edge list. The first three columns plus color follow the
        // Net-Viz edge-list input format, so the file can be re-imported
        // directly; nt_type is recognized on import as well. Grouping is
        // recorded as the endpoint node groups (source/intermediate/target)
        // plus any custom-group membership; NT grouping mirrors the Python
        // get_nt_group mapping.
        
        function csvEscapeField(value) {{
            const s = (value === null || value === undefined) ? '' : String(value);
            if (/[",\\n\\r]/.test(s)) {{
                return '"' + s.replace(/"/g, '""') + '"';
            }}
            return s;
        }}
        
        function getNtGroupCSV(ntType) {{
            if (ntType === null || ntType === undefined) return 'unknown';
            const ntStr = String(ntType).trim();
            // Mirrors the Python get_nt_group mapping (GLUT counts as
            // inhibitory there too).
            const groups = {{
                'excitatory': ['acetylcholine', 'ACH', 'ach'],
                'inhibitory': ['gaba', 'GABA', 'glutamate', 'GLUT', 'glut'],
                'modulatory': ['dopamine', 'DA', 'da', 'serotonin', 'SER', 'ser', '5-HT', '5-ht', 'octopamine', 'OCT', 'oct'],
                'unknown': ['unknown', 'none', '']
            }};
            const ntLower = ntStr.toLowerCase();
            for (const group of Object.keys(groups)) {{
                const members = groups[group];
                if (members.indexOf(ntStr) !== -1 || members.some(m => m.toLowerCase() === ntLower)) {{
                    return group;
                }}
            }}
            return 'unknown';
        }}
        
        // Pure CSV builder (no DOM access) - the Node test harness extracts
        // and executes this function against headless Cytoscape.
        function buildEdgeListCSV() {{
            const header = ['source', 'target', 'weight', 'color', 'nt_type', 'nt_group',
                            'source_group', 'target_group', 'custom_groups', 'ratio', 'probability',
                            'edge info', 'source info', 'target info'];
            const edgeWeightLabelJS = '{self.edge_weight_label}';
            // {{key:val; ...}} hover-info cells for the CSV round trip.
            const formatInfoCell = (obj, extraEntries) => {{
                const entries = [...(extraEntries || []), ...Object.entries(obj || {{}})];
                if (!entries.length) return '';
                return '{{' + entries.map(([k, v]) => k + ':' + v).join('; ') + '}}';
            }};
            // node id -> names of the custom groups currently containing it
            // (live membership via each node's assigned_group field)
            const membership = {{}};
            Object.keys(customGroups).forEach(name => {{
                groupMembers(name).forEach(node => {{
                    const id = node.id();
                    if (!membership[id]) membership[id] = [];
                    membership[id].push(name);
                }});
            }});
            const labelOf = node => node.data('label') || node.id();
            const lines = [header.join(',')];
            cy.edges().forEach(edge => {{
                // Prefer the signed original weight (edit mode keeps it in
                // sync with the display weight; drawn edges set both).
                const weightRaw = edge.data('original_weight');
                const weight = (weightRaw !== undefined && weightRaw !== null && !isNaN(parseFloat(weightRaw)))
                    ? parseFloat(weightRaw)
                    : (parseFloat(edge.data('weight')) || 0);
                // Base appearance is seeded from the computed style at load
                // time and kept in sync by every color edit, so it is the
                // authoritative current color; fall back to the live style.
                // Cytoscape reports computed colors as rgb()/rgba() strings, so
                // normalize through extractColorHex to always emit a hex value.
                const rawColor = edge.data(EDGE_BASE_COLOR_KEY) || edge.style('line-color');
                const color = extractColorHex(rawColor);
                const nt = edge.data('nt_type') || '';
                const sourceNode = edge.source();
                const targetNode = edge.target();
                // ratio/probability default to 0 when absent - export them
                // as empty cells so re-import does not invent metrics
                const ratio = edge.data('ratio');
                const prob = edge.data('probability');
                const edgeInfo = formatInfoCell(edge.data('custom_labels') || {{}},
                    [['weight', weight + ' ' + edgeWeightLabelJS]]);
                const sourceInfo = formatInfoCell(sourceNode.data('dataset_info') || {{}});
                const targetInfo = formatInfoCell(targetNode.data('dataset_info') || {{}});
                const row = [
                    labelOf(sourceNode),
                    labelOf(targetNode),
                    weight,
                    color,
                    nt,
                    nt ? getNtGroupCSV(nt) : '',
                    (sourceNode.data('assigned_group') ?? sourceNode.data('node_type') ?? 'intermediate'),
                    (targetNode.data('assigned_group') ?? targetNode.data('node_type') ?? 'intermediate'),
                    Array.from(new Set([
                        ...(membership[sourceNode.id()] || []),
                        ...(membership[targetNode.id()] || [])
                    ])).join(';'),
                    ratio ? ratio : '',
                    prob ? prob : '',
                    edgeInfo,
                    sourceInfo,
                    targetInfo
                ];
                lines.push(row.map(csvEscapeField).join(','));
            }});
            return lines.join('\\n');
        }}
        
        function exportEdgeListCSV() {{
            const edgeCount = cy.edges().length;
            if (edgeCount === 0) {{
                updateHoverInfo('⚠️ No edges to export');
                return;
            }}
            const csv = buildEdgeListCSV();
            const dataBlob = new Blob([csv], {{type: 'text/csv;charset=utf-8'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'network_edge_list_' + new Date().toISOString().slice(0,10) + '.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            updateHoverInfo('✓ Exported ' + edgeCount + ' edge(s) as CSV (source, target, weight, color, NT, grouping)');
        }}
        
        // Import graph (trigger file input)
        function importGraph() {{
            document.getElementById('graphFileInput').click();
        }}
        
        // Load graph from file
        function loadGraphFile(event) {{
            const file = event.target.files[0];
            if (!file) return;
            pushHistory('Import graph');

            const reader = new FileReader();
            reader.onload = async function(e) {{
                try {{
                    const importData = JSON.parse(e.target.result);

                    if (!importData.nodes || !importData.edges) {{
                        showToast('Invalid graph file format — expected a graph export with "nodes" and "edges"', 'error');
                        return;
                    }}

                    // Replace or merge — decided in the in-page dialog
                    // (Confirm = REPLACE, Cancel = MERGE), consistent with
                    // the no-native-modal contract.
                    let replace = false;
                    try {{
                        await showDialog({{
                            title: 'Import graph',
                            message: 'Import ' + importData.nodes.length + ' nodes and ' +
                                importData.edges.length + ' edges — OK to REPLACE the current graph, Cancel to MERGE into it.',
                            fields: [],
                            confirmText: 'Replace',
                        }});
                        replace = true;
                    }} catch (dialogError) {{
                        if (dialogError === 'busy') return;  // another dialog owns the page
                    }}

                    if (replace) {{
                        // Replace: clear current graph
                        cy.elements().remove();
                    }}
                    
                    // Add nodes (with layout positions if available)
                    const nodeIds = new Set();
                    importData.nodes.forEach(nodeData => {{
                        const nodeId = nodeData.data.id;
                        nodeIds.add(nodeId);
                        
                        // Check if node already exists
                        if (cy.getElementById(nodeId).length > 0) {{
                            // Update existing node position if in merge mode
                            if (!replace && nodeData.position) {{
                                cy.getElementById(nodeId).position(nodeData.position);
                            }}
                        }} else {{
                            // Add new node
                            cy.add({{
                                group: 'nodes',
                                data: nodeData.data,
                                position: nodeData.position || {{ x: 0, y: 0 }}
                            }});
                            
                            // Restore classes (e.g., hidden)
                            if (nodeData.classes && nodeData.classes.length > 0) {{
                                const node = cy.getElementById(nodeId);
                                nodeData.classes.forEach(cls => node.addClass(cls));
                            }}
                            
                            // Apply custom styles if available
                            if (nodeData.style) {{
                                const node = cy.getElementById(nodeId);
                                if (nodeData.style['background-color']) {{
                                    node.style('background-color', nodeData.style['background-color']);
                                }}
                                if (nodeData.style['opacity'] !== undefined) {{
                                    node.style('opacity', nodeData.style['opacity']);
                                }}
                            }}
                        }}
                    }});
                    
                    // Add edges (only if both source and target exist)
                    let addedEdges = 0;
                    importData.edges.forEach(edgeData => {{
                        const source = edgeData.data.source;
                        const target = edgeData.data.target;
                        
                        // Check if both nodes exist
                        if (cy.getElementById(source).length > 0 && cy.getElementById(target).length > 0) {{
                            // Check if edge already exists
                            const existingEdge = cy.edges(`[source = "${{source}}"][target = "${{target}}"]`);
                            if (existingEdge.length === 0) {{
                                cy.add({{
                                    group: 'edges',
                                    data: edgeData.data
                                }});
                                
                                // Get the newly added edge
                                const edge = cy.edges(`[source = "${{source}}"][target = "${{target}}"]`);
                                
                                // Restore classes (e.g., hidden, filtered)
                                if (edgeData.classes && edgeData.classes.length > 0 && edge.length > 0) {{
                                    edgeData.classes.forEach(cls => edge.addClass(cls));
                                }}
                                
                                // Apply custom styles if available
                                if (edgeData.style && edge.length > 0) {{
                                    edge.forEach(function(e) {{
                                        const importedColor = edgeData.style['line-color'] || e.style('line-color');
                                        const importedOpacity = (edgeData.style['opacity'] !== undefined)
                                            ? edgeData.style['opacity']
                                            : (parseFloat(e.style('opacity')) || 1);
                                        const canApplyBase = !e.selected() && !e.hasClass('highlighted');
                                        setEdgeBaseAppearance(e, importedColor, importedOpacity, canApplyBase);
                                        if (!canApplyBase) {{
                                            applyEdgeHighlightOverride(e);
                                        }}
                                    }});
                                }}
                                addedEdges++;
                            }}
                        }}
                    }});
                    
                    // Fit to view
                    cy.fit(null, 50);
                    
                    // Memberships: backfill nodes from older exports, then
                    // rebuild every group-driven UI from live state
                    normalizeAssignedGroups();
                    rebuildAssignSelect();
                    rebuildCustomGroupUI();
                    refreshLegend();
                    
                    // Re-detect orphans/dead ends on the imported graph if
                    // their hiding toggles are active
                    reapplyDeadEndHiding();
                    reapplyOrphanHiding();
                    
                    // Restore settings if available
                    if (importData.settings) {{
                        const settings = importData.settings;
                        
                        // Restore edge filter
                        if (settings.edgeFilter) {{
                            const filterInput = document.getElementById('ignoreEdgesInput');
                            if (filterInput && settings.edgeFilter.inputValue) {{
                                filterInput.value = settings.edgeFilter.inputValue;
                                updateIgnoredEdges(); // Apply the filter
                            }}
                        }}
                        
                        // Restore style controls (suppressed from history:
                        // 'Import graph' is the recorded operation)
                        restoringHistoryState = true;
                        try {{
                            // Restore edge width scaling
                            if (settings.edgeWidthScaling) {{
                                const scalingSelect = document.getElementById('edgeWidthScale');
                                if (scalingSelect && settings.edgeWidthScaling.method) {{
                                    scalingSelect.value = settings.edgeWidthScaling.method;
                                    updateEdgeWidths(); // Apply the scaling change
                                }}
                                
                                const widthSlider = document.getElementById('edgeWidthSlider');
                                if (widthSlider && settings.edgeWidthScaling.width !== undefined) {{
                                    widthSlider.value = settings.edgeWidthScaling.width;
                                    updateEdgeWidth(settings.edgeWidthScaling.width);
                                }}
                            }}
                            
                            // Restore arrow size
                            if (settings.arrowSize !== undefined) {{
                                const arrowSlider = document.getElementById('arrowSizeSlider');
                                if (arrowSlider) {{
                                    arrowSlider.value = settings.arrowSize;
                                    updateArrowSize(settings.arrowSize);
                                }}
                            }}
                            
                            // Restore font size
                            if (settings.fontSize !== undefined) {{
                                const fontSlider = document.getElementById('fontSizeSlider');
                                if (fontSlider) {{
                                    fontSlider.value = settings.fontSize;
                                    updateFontSize(settings.fontSize);
                                }}
                            }}
                            
                            // Restore node size
                            if (settings.nodeSize !== undefined) {{
                                const nodeSlider = document.getElementById('nodeSizeSlider');
                                if (nodeSlider) {{
                                    nodeSlider.value = settings.nodeSize;
                                    updateNodeSize(settings.nodeSize);
                                }}
                            }}
                            
                            // Restore the label font color choice
                            if (settings.labelFontColor) {{
                                applyLabelFontColor(settings.labelFontColor);
                            }}
                        }} finally {{
                            restoringHistoryState = false;
                        }}
                        
                        // Restore group defaults (NT groups, etc.)
                        if (settings.groupDefaults) {{
                            Object.keys(settings.groupDefaults).forEach(key => {{
                                groupDefaults[key] = settings.groupDefaults[key];
                            }});
                        }}
                        
                        // Restore custom group DEFINITIONS (membership lives
                        // in each node's assigned_group data field; legacy
                        // exports with static id lists are normalized).
                        if (settings.customGroups) {{
                            Object.keys(settings.customGroups).forEach(groupName => {{
                                const def = settings.customGroups[groupName] || {{}};
                                customGroups[groupName] = {{
                                    label: def.label || groupName,
                                    color: def.color || '#888888',
                                    opacity: def.opacity === undefined ? 100 : def.opacity,
                                    defaultColor: def.defaultColor || def.color || '#888888',
                                    defaultOpacity: def.defaultOpacity === undefined ? 100 : def.defaultOpacity
                                }};
                                // Legacy exports carried static id lists —
                                // re-materialize their memberships AND look
                                // (legacy payloads may lack per-node styles).
                                (def.ids || []).forEach(id => {{
                                    const n = cy.getElementById(id);
                                    if (n.length > 0 && n.isNode()) {{
                                        n.data('assigned_group', groupName);
                                        applyGroupLook(n, customGroups[groupName]);
                                    }}
                                }});
                            }});
                            rebuildCustomGroupUI();
                            refreshLegend();
                        }}
                        
                        const customGroupCount = settings.customGroups ? Object.keys(settings.customGroups).length : 0;
                        updateHoverInfo(
                            '✓ Imported with settings: ' + importData.nodes.length + ' nodes, ' + 
                            addedEdges + ' edges' + (customGroupCount > 0 ? ', ' + customGroupCount + ' custom groups' : '')
                        );
                    }} else {{
                        updateHoverInfo(
                            '✓ Imported: ' + importData.nodes.length + ' nodes, ' + 
                            addedEdges + ' edges (layout preserved)'
                        );
                    }}
                    
                }} catch (error) {{
                    showToast('Error loading graph file: ' + error.message, 'error');
                    console.error('Import error:', error);
                }}
            }};
            reader.readAsText(file);
            
            // Reset file input
            event.target.value = '';
        }}
        
        // ===== EXPORT/IMPORT LAYOUT ONLY =====
        
        // Export only node positions (layout)
        function exportLayout() {{
            // Collect only node IDs and positions
            const layoutData = {{}};
            cy.nodes().forEach(node => {{
                layoutData[node.id()] = {{
                    x: node.position().x,
                    y: node.position().y
                }};
            }});
            
            // Create export object
            const exportData = {{
                version: '1.0',
                type: 'layout',
                timestamp: new Date().toISOString(),
                layout: layoutData,
                metadata: {{
                    nodeCount: Object.keys(layoutData).length,
                    description: 'Node positions only (no edges or properties)'
                }}
            }};
            
            // Download as JSON
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'network_layout_' + new Date().toISOString().slice(0,10) + '.json';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            updateHoverInfo('✓ Exported layout: ' + Object.keys(layoutData).length + ' node positions');
        }}
        
        // Import layout (trigger file input)
        function importLayout() {{
            document.getElementById('layoutFileInput').click();
        }}
        
        // Load layout from file and apply to existing nodes
        function loadLayoutFile(event) {{
            const file = event.target.files[0];
            if (!file) return;
            pushHistory('Import layout');
            
            const reader = new FileReader();
            reader.onload = function(e) {{
                try {{
                    const importData = JSON.parse(e.target.result);
                    
                    // Check if it's a layout file
                    if (!importData.layout) {{
                        showToast('Invalid layout file format. Expected a layout export file.', 'error');
                        return;
                    }}
                    
                    const layoutData = importData.layout;
                    let updatedCount = 0;
                    let notFoundCount = 0;
                    const notFoundNodes = [];
                    
                    // Apply positions to existing nodes
                    for (const nodeId in layoutData) {{
                        const node = cy.getElementById(nodeId);
                        if (node.length > 0) {{
                            // Node exists - update its position
                            node.position({{
                                x: layoutData[nodeId].x,
                                y: layoutData[nodeId].y
                            }});
                            updatedCount++;
                        }} else {{
                            // Node not found in current graph
                            notFoundCount++;
                            notFoundNodes.push(nodeId);
                        }}
                    }}
                    
                    // Show results
                    let message = `✓ Layout applied: ${{updatedCount}} nodes repositioned`;
                    
                    if (notFoundCount > 0) {{
                        message += `\\n⚠️ ${{notFoundCount}} nodes not found in current graph`;
                        if (notFoundCount <= 5) {{
                            message += `\\n  Missing: ${{notFoundNodes.join(', ')}}`;
                        }}
                    }}
                    
                    // Check for nodes in current graph that weren't in layout
                    const currentNodeCount = cy.nodes().length;
                    const unmappedCount = currentNodeCount - updatedCount;
                    if (unmappedCount > 0) {{
                        message += `\\n💡 ${{unmappedCount}} current nodes kept their positions`;
                    }}
                    
                    showToast(message, 'success', {{ label: 'Fit', onClick: function() {{ cy.fit(null, 50); }} }});
                    
                }} catch (error) {{
                    showToast('Error loading layout file: ' + error.message, 'error');
                    console.error('Import error:', error);
                }}
            }};
            reader.readAsText(file);
            
            // Reset file input
            event.target.value = '';
        }}
        
        // Add CSS for edge-source indicator
        cy.style().selector('.edge-source').style({{
            'border-width': '4px',
            'border-color': '#ff9800',
            'border-opacity': 1
        }}).update();
        
        // Initial fit and apply opacity
        cy.fit(null, 50);
        applyInitialOpacity();
        initializeEdgeBaseStyles();
        initializeLogBaseVisibility();
        initPanelBar();
    </script>
</body>
</html>"""
        
        # Save HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if self.showfig and open_browser:
            webbrowser.open('file://' + os.path.abspath(output_path))
    
    def _save_df_to_csv_polars(self, df, path, index=False):
        """Save DataFrame to CSV using Polars for speed"""
        if df is None or df.empty:
            # Create empty file if dataframe is empty, to match pandas behavior
            with open(path, 'w') as f:
                if df is not None:
                    f.write(','.join(df.columns) + '\n')
            return
            
        try:
            # If index is True, reset index to make it a column
            if index:
                df_to_save = df.reset_index()
            else:
                df_to_save = df
                
            pl_df = pl.from_pandas(df_to_save)
            pl_df.write_csv(path)
        except Exception as e:
            # Fallback to Pandas if Polars fails (e.g. object types)
            try:
                df.to_csv(path, index=index)
            except Exception as e2:
                print(f"  Error saving CSV: {e2}", flush=True)

    def _save_dfs_to_excel_polars(self, data_map, path):
        """
        Save multiple DataFrames to Excel using Polars.
        
        Parameters
        ----------
        data_map : dict
            {sheet_name: (df, include_index)}
        path : str
            Output path
        """
        try:
            import xlsxwriter
            with xlsxwriter.Workbook(path, {'nan_inf_to_errors': True}) as workbook:
                for sheet_name, (df, include_index) in data_map.items():
                    if df is None:
                        continue
                        
                    if include_index:
                        df_to_save = df.reset_index()
                    else:
                        df_to_save = df
                    
                    # Convert to Polars
                    pl_df = pl.from_pandas(df_to_save)
                    
                    # Write to worksheet
                    # Polars write_excel supports workbook argument
                    pl_df.write_excel(workbook=workbook, worksheet=sheet_name)
        except Exception as e:
            self._vprint(f"Polars Excel save failed ({e}), falling back to Pandas...")
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                for sheet_name, (df, include_index) in data_map.items():
                    if df is not None:
                        df.to_excel(writer, sheet_name=sheet_name, index=include_index)

    def save_data(self):
        """
        Save connection data and original paths to Excel or CSV files.
        
        If output_format is 'xlsx':
            Creates an Excel file with multiple sheets:
            1. 'connections': Aggregated connection data
            2. 'original_paths': Original pathway data
            3. 'connMatrix_weight': Connection matrix (weights)
            4. 'connMatrix_ratio': Connection matrix (ratios) - if available
            5. 'connMatrix_prob': Connection matrix (probabilities) - if available
            6. 'connMatrix_nt_type': Connection matrix (neurotransmitters) - if available
            
        If output_format is 'csv':
            Creates multiple CSV files in the output folder:
            - [filename]_data_connections.csv
            - [filename]_data_original_paths.csv
            - [filename]_data_connMatrix_weight.csv
            - ...
        
        Returns
        -------
        str or list
            Path to the generated Excel file, or list of paths to CSV files
        """
        if self.conn_df is None:
            self.build_network()
        
        # Create matrices
        all_nodes = sorted(list(set(self.conn_df['source']).union(set(self.conn_df['target']))))
        
        # Weight matrix
        weight_matrix = self.conn_df.pivot_table(
            index='source', 
            columns='target', 
            values='weight', 
            fill_value=0
        ).reindex(index=all_nodes, columns=all_nodes, fill_value=0)
        
        # Ratio matrix
        ratio_matrix = None
        if 'ratio' in self.conn_df.columns:
            ratio_matrix = self.conn_df.pivot_table(
                index='source', 
                columns='target', 
                values='ratio', 
                fill_value=0
            ).reindex(index=all_nodes, columns=all_nodes, fill_value=0)
            
        # Probability matrix
        prob_matrix = None
        if 'probability' in self.conn_df.columns:
            prob_matrix = self.conn_df.pivot_table(
                index='source', 
                columns='target', 
                values='probability', 
                fill_value=0
            ).reindex(index=all_nodes, columns=all_nodes, fill_value=0)
            
        # NT Type matrix
        nt_matrix = None
        if 'nt_type' in self.conn_df.columns:
            nt_matrix = self.conn_df.pivot_table(
                index='source', 
                columns='target', 
                values='nt_type', 
                aggfunc='first',
                fill_value=''
            ).reindex(index=all_nodes, columns=all_nodes, fill_value='')

        if self.output_format == 'csv':
            self._vprint("\nSaving data to CSV files...")
            created_files = []
            
            # Save connections
            conn_path = os.path.join(self.output_folder, self.base_filename + '_data_connections.csv')
            self._save_df_to_csv_polars(self.conn_df, conn_path, index=False)
            created_files.append(conn_path)
            
            # Save original paths
            paths_path = os.path.join(self.output_folder, self.base_filename + '_data_original_paths.csv')
            
            # Create a copy for saving to avoid modifying the original dataframe
            df_to_save = self.path_df.copy()
            
            # Sort by path length (ascending) and path probability (descending)
            sort_cols = []
            ascending_order = []
            # Check for length column (could be 'length' or 'path_length')
            if 'length' in df_to_save.columns:
                sort_cols.append('length')
                ascending_order.append(True)
            elif 'path_length' in df_to_save.columns:
                sort_cols.append('path_length')
                ascending_order.append(True)
                
            # Check for probability column (could be 'path_prob' or 'path_probability')
            if 'path_prob' in df_to_save.columns:
                sort_cols.append('path_prob')
                ascending_order.append(False)
            elif 'path_probability' in df_to_save.columns:
                sort_cols.append('path_probability')
                ascending_order.append(False)
                
            if sort_cols:
                df_to_save = df_to_save.sort_values(sort_cols, ascending=ascending_order)
            
            # Drop redundant columns
            cols_to_drop = ['path_str', 'path_block', 'connection_ratios', 'traversal_probabilities']
            df_to_save = df_to_save.drop(columns=[c for c in cols_to_drop if c in df_to_save.columns])
            
            self._save_df_to_csv_polars(df_to_save, paths_path, index=False)
            created_files.append(paths_path)
            
            # Save matrices (skipped when save_data_matrices=False — e.g.
            # FindAllPath, whose data_details/conn_mat_type_*.csv are the
            # canonical type-level matrices; connections/paths still save)
            if self.save_data_matrices:
                weight_path = os.path.join(self.output_folder, self.base_filename + '_data_connMatrix_weight.csv')
                self._save_df_to_csv_polars(weight_matrix, weight_path, index=True)
                created_files.append(weight_path)
                
                if ratio_matrix is not None:
                    ratio_path = os.path.join(self.output_folder, self.base_filename + '_data_connMatrix_ratio.csv')
                    self._save_df_to_csv_polars(ratio_matrix, ratio_path, index=True)
                    created_files.append(ratio_path)
                    
                if prob_matrix is not None:
                    prob_path = os.path.join(self.output_folder, self.base_filename + '_data_connMatrix_prob.csv')
                    self._save_df_to_csv_polars(prob_matrix, prob_path, index=True)
                    created_files.append(prob_path)
                    
                if nt_matrix is not None:
                    nt_path = os.path.join(self.output_folder, self.base_filename + '_data_connMatrix_nt_type.csv')
                    self._save_df_to_csv_polars(nt_matrix, nt_path, index=True)
                    created_files.append(nt_path)
            
            self._vprint(f"Data saved to {len(created_files)} CSV files in: {self.output_folder}")
            return created_files
            
        else:
            self._vprint("\nSaving data to Excel...")
            
            output_path = os.path.join(self.output_folder, self.base_filename + '_data.xlsx')
            
            data_map = {
                'connections': (self.conn_df, False),
                'original_paths': (self.path_df, False),
            }
            
            if self.save_data_matrices:
                data_map['connMatrix_weight'] = (weight_matrix, True)
                if ratio_matrix is not None:
                    data_map['connMatrix_ratio'] = (ratio_matrix, True)
                if prob_matrix is not None:
                    data_map['connMatrix_prob'] = (prob_matrix, True)
                if nt_matrix is not None:
                    data_map['connMatrix_nt_type'] = (nt_matrix, True)
                
            self._save_dfs_to_excel_polars(data_map, output_path)
            
            self._vprint(f"Data saved: {output_path}")
            
            return output_path
    
    def create_heatmaps(self, conn_matrices, titles=None, color_scales=None):
        """
        Create heatmap visualizations for connection matrices.
        
        Parameters
        ----------
        conn_matrices : dict
            Dictionary of {matrix_name: matrix_dataframe} to visualize.
            Example: {'connMatrix_bodyId': df1, 'connMatrix_type': df2}
        titles : dict, optional
            Dictionary of {matrix_name: title_string}. If None, uses matrix names.
        color_scales : dict, optional
            Dictionary of {matrix_name: color_scale}. If None, uses defaults.
            Color scales should be in Plotly format: [[0, 'color1'], [1, 'color2']]
        
        Returns
        -------
        list
            List of created heatmap file paths
            
        Example
        -------
        >>> matrices = {
        ...     'connMatrix_type': conn_matrix_type,
        ...     'ratioMatrix_type': conn_matrix_ratio_type
        ... }
        >>> titles = {
        ...     'connMatrix_type': 'Connection Matrix by Type',
        ...     'ratioMatrix_type': 'Connection Ratio Matrix by Type'
        ... }
        >>> color_scales = {
        ...     'connMatrix_type': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],
        ...     'ratioMatrix_type': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']]
        ... }
        >>> vp.create_heatmaps(matrices, titles, color_scales)
        """
        if titles is None:
            titles = {}
        if color_scales is None:
            color_scales = {}
        
        # Default color scales for different matrix types
        default_color_scales = {
            'conn': [[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],        # Green for connections
            'transmission': [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],  # Purple for transmission
            'ratio': [[0, 'rgb(255,255,255)'], [1, 'rgb(204,102,0)']]       # Orange for ratios
        }
        
        created_files = []
        
        self._vprint('\nCreating heatmap visualizations...')
        for matrix_name, matrix_df in conn_matrices.items():
            if matrix_df is None or matrix_df.empty:
                continue
            
            # Generate title
            if matrix_name in titles:
                title = titles[matrix_name]
            else:
                title = f'Heatmap: {matrix_name}'
            
            # Select color scale
            if matrix_name in color_scales:
                color_scale = color_scales[matrix_name]
            else:
                # Auto-detect color scale based on matrix name
                if 'ratio' in matrix_name.lower():
                    color_scale = default_color_scales['ratio']
                elif 'transmission' in matrix_name.lower() or 'prob' in matrix_name.lower():
                    color_scale = default_color_scales['transmission']
                else:
                    color_scale = default_color_scales['conn']
            
            # Generate filename
            filename = os.path.join(self.output_folder, f'heatmap_{matrix_name}.html')
            
            # Create heatmap using the standalone VisConnMatInteractive
            VisConnMatInteractive(
                matrix_df,
                filename=filename,
                title=title,
                color_scale=color_scale,
                showfig=False,  # Don't auto-open each heatmap
                verbose=self.verbose
            )
            
            created_files.append(filename)
            self._vprint(f'  Created: heatmap_{matrix_name}.html')
        
        self._vprint('Done\n')
        return created_files

    def create_heatmap(self, custom_row_order=None, custom_col_order=None):
        """
        Create an interactive heatmap from the connection DataFrame.
        
        Converts the connection DataFrame (source, target, weight) into matrices
        and creates an interactive heatmap visualization using VisConnMatInteractive.
        
        Parameters
        ----------
        custom_row_order : list of str, optional
            Custom order for row nodes (sources). If None, uses the class attribute
            `heatmap_row_order` (if set), otherwise uses sorted order.
            Nodes not in the list will be appended at the end.
        custom_col_order : list of str, optional
            Custom order for column nodes (targets). If None, uses the class attribute
            `heatmap_col_order` (if set), otherwise uses sorted order.
            Nodes not in the list will be appended at the end.
        
        Returns
        -------
        str
            Path to the generated HTML file
            
        Notes
        -----
        - Creates matrices for weight, ratio, and probability (if available)
        - Node ordering follows source -> intermediate -> target (or custom order)
        - Missing connections are shown as 0
        - Creates an interactive heatmap with metric toggle, zoom, pan, and export features
        
        Examples
        --------
        >>> # Default sorting
        >>> vis.create_heatmap()
        >>> 
        >>> # Custom row order
        >>> vis.create_heatmap(custom_row_order=['A', 'B', 'C'])
        >>> 
        >>> # Custom both orders
        >>> vis.create_heatmap(
        ...     custom_row_order=['PN1', 'PN2', 'LHN1'],
        ...     custom_col_order=['LHN1', 'LHN2', 'MBON1']
        ... )
        """
        if self.conn_df is None or len(self.conn_df) == 0:
            self._vprint("Warning: No connection data available for heatmap.")
            return None
        
        # Filter edges if limit is set — the SAME complete-path/corridor edge
        # set as the network, so the heatmap never shows edges the network
        # hides or endpoint edges that would be disconnected there.
        conn_df_to_use = self.conn_df
        selected = self._select_edges_for_plot()
        if selected is not None:
            kept_edges, _boundary_capped, _integrity_relaxed, _paths, _thr = selected
            conn_df_to_use = self._filter_conn_df_for_plot(self.conn_df)
            self._vprint(
                f'  Filtering to the complete-path/corridor edge set — same '
                f'set as the network (out of {len(self.conn_df)})')
        elif getattr(self, 'G_network', None) is None and self.edgeN_limit \
                and len(self.conn_df) > self.edgeN_limit:
            # standalone heatmap without a network graph: plain top-N fallback
            self.edge_limit_trimmed = True
            self._vprint(
                f'  Filtering top {self.edgeN_limit} edges by weight (out of {len(self.conn_df)})')
            conn_df_to_use = self.conn_df.sort_values(
                'weight', ascending=False).head(self.edgeN_limit)
        
        self._vprint("\nCreating interactive heatmap...")
        
        # Create weight matrix from connections
        weight_matrix = conn_df_to_use.pivot_table(
            index='source',
            columns='target',
            values='weight',
            fill_value=0
        )
        
        # Determine nodes that actually appear as sources or targets in the connection data
        # This handles cases where a node can be BOTH a source and a target
        actual_sources = set(conn_df_to_use['source'].unique())
        actual_targets = set(conn_df_to_use['target'].unique())
        all_nodes = actual_sources | actual_targets
        
        # Nodes that are ONLY sources (appear as source but never as target)
        source_only = actual_sources - actual_targets
        # Nodes that are ONLY targets (appear as target but never as source)
        target_only = actual_targets - actual_sources
        # Nodes that are BOTH source AND target (intermediate nodes in pathways)
        intermediate_nodes = actual_sources & actual_targets
        
        # For heatmap: rows = nodes that act as sources, cols = nodes that act as targets
        # This ensures we don't miss any connections
        all_row_nodes = list(source_only) + list(intermediate_nodes)
        all_col_nodes = list(intermediate_nodes) + list(target_only)
        
        # Apply custom ordering if provided
        # Priority: parameter > class attribute > default (sorted)
        if custom_row_order is not None:
            row_nodes = self._apply_custom_order(all_row_nodes, custom_row_order)
        elif self.heatmap_row_order is not None:
            row_nodes = self._apply_custom_order(all_row_nodes, self.heatmap_row_order)
        else:
            row_nodes = sorted(all_row_nodes)
            
        if custom_col_order is not None:
            col_nodes = self._apply_custom_order(all_col_nodes, custom_col_order)
        elif self.heatmap_col_order is not None:
            col_nodes = self._apply_custom_order(all_col_nodes, self.heatmap_col_order)
        else:
            col_nodes = sorted(all_col_nodes)
        
        # Reindex weight matrix: rows = sources, columns = targets
        weight_matrix = weight_matrix.reindex(index=row_nodes, columns=col_nodes, fill_value=0)
        
        # Create ratio and probability matrices if those columns exist
        matrices_dict = {'weight': weight_matrix}
        
        if 'ratio' in conn_df_to_use.columns:
            self._vprint("  Creating ratio matrix...")
            ratio_matrix = conn_df_to_use.pivot_table(
                index='source',
                columns='target',
                values='ratio',
                fill_value=0
            )
            ratio_matrix = ratio_matrix.reindex(index=row_nodes, columns=col_nodes, fill_value=0)
            matrices_dict['ratio'] = ratio_matrix
        
        if 'probability' in conn_df_to_use.columns:
            self._vprint("  Creating probability matrix...")
            prob_matrix = conn_df_to_use.pivot_table(
                index='source',
                columns='target',
                values='probability',
                fill_value=0
            )
            prob_matrix = prob_matrix.reindex(index=row_nodes, columns=col_nodes, fill_value=0)
            matrices_dict['probability'] = prob_matrix
        
        # Generate filename
        heatmap_file = os.path.join(self.output_folder, f'{self.base_filename}_heatmap.html')
        
        # Create interactive heatmap with metric toggle
        title = f'Connection Matrix - {self.base_filename}'
        VisConnMatInteractive(
            weight_matrix,  # Default matrix (backward compatibility)
            filename=heatmap_file,
            title=title,
            showfig=False,
            matrices_dict=matrices_dict,  # Pass all matrices for metric toggle
            verbose=self.verbose
        )
        
        self._vprint(f"  Heatmap saved: {heatmap_file}")
        
        return heatmap_file

    def get_heatmap_node_info(self):
        """
        Get information about nodes that will appear in heatmap rows and columns.
        
        This helps users understand and customize the heatmap node ordering.
        Must be called after build_network().
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'row_nodes': list of nodes that appear in rows (sources)
            - 'col_nodes': list of nodes that appear in columns (targets)
            - 'source_only': nodes that only act as sources
            - 'target_only': nodes that only act as targets
            - 'intermediate': nodes that act as both source and target
            
        Examples
        --------
        >>> vis = VisualizePath('data.csv')
        >>> vis.build_network()
        >>> info = vis.get_heatmap_node_info()
        >>> print("Row nodes:", info['row_nodes'])
        >>> print("Column nodes:", info['col_nodes'])
        >>> print("Dual-role nodes:", info['intermediate'])
        """
        if self.conn_df is None or len(self.conn_df) == 0:
            raise ValueError("No connection data available. Run build_network() first.")
        
        # Determine nodes that actually appear as sources or targets
        actual_sources = set(self.conn_df['source'].unique())
        actual_targets = set(self.conn_df['target'].unique())
        
        source_only = actual_sources - actual_targets
        target_only = actual_targets - actual_sources
        intermediate = actual_sources & actual_targets
        
        all_row_nodes = list(source_only) + list(intermediate)
        all_col_nodes = list(intermediate) + list(target_only)
        
        return {
            'row_nodes': sorted(all_row_nodes),
            'col_nodes': sorted(all_col_nodes),
            'source_only': sorted(source_only),
            'target_only': sorted(target_only),
            'intermediate': sorted(intermediate)
        }
    
    def print_heatmap_node_info(self):
        """
        Print heatmap node information in a user-friendly format.
        
        Displays which nodes will appear in rows vs columns and their roles.
        Must be called after build_network().
        
        Examples
        --------
        >>> vis = VisualizePath('data.csv')
        >>> vis.build_network()
        >>> vis.print_heatmap_node_info()
        """
        info = self.get_heatmap_node_info()
        
        self._vprint("\n" + "=" * 80)
        self._vprint("HEATMAP NODE INFORMATION")
        self._vprint("=" * 80)
        self._vprint(f"\nRow nodes (sources): {len(info['row_nodes'])} total")
        self._vprint(f"  {', '.join(info['row_nodes'])}")
        self._vprint(f"\nColumn nodes (targets): {len(info['col_nodes'])} total")
        self._vprint(f"  {', '.join(info['col_nodes'])}")
        self._vprint(f"\nNode roles:")
        self._vprint(f"  Source-only:  {len(info['source_only'])} nodes - {', '.join(info['source_only'])}")
        self._vprint(f"  Target-only:  {len(info['target_only'])} nodes - {', '.join(info['target_only'])}")
        self._vprint(f"  Both (intermediate): {len(info['intermediate'])} nodes - {', '.join(info['intermediate'])}")
        self._vprint("\nNote: Nodes in 'Both' category appear in BOTH rows AND columns")
        self._vprint("=" * 80 + "\n")
    
    def visualize(self, plot_heatmap=True, plot_Sankey=True, plot_network=True):
        """
        Create all visualizations (Heatmap + Sankey + Network + Data export).
        
        This is the main method to call for complete visualization workflow.
        It executes:
        1. Build network graph from pathway data
        2. Create heatmap (shown first if showfig=True)
        3. Create Sankey diagram
        4. Create interactive network graph
        5. Save data to Excel
        
        When showfig=True, opens all three visualizations in browser:
        heatmap → sankey → network
        
        If generate_empty_network=True, only generates an empty network HTML template.
        
        Returns
        -------
        tuple
            (conn_df, G_network) - Connection DataFrame and NetworkX graph
            For empty network: (None, None)
            
        Example
        -------
        >>> vp = VisualizePath('path_type.xlsx', showfig=True)
        >>> conn_df, G = vp.visualize()
        >>> print(f"Created {len(conn_df)} connections")
        
        >>> # Generate empty network template
        >>> vp = VisualizePath(path_file=None, generate_empty_network=True, showfig=True)
        >>> vp.visualize()
        """
        # Handle empty network generation
        if self.generate_empty_network:
            self._progress(1, 4, "Creating empty network template")
            self._vprint("=" * 80)
            self._vprint("VisualizePath - Generating empty network template")
            self._vprint("=" * 80)
            self.generate_empty_network_html()
            self._vprint("=" * 80)
            self._vprint("✓ Empty network generation complete!")
            self._vprint("=" * 80)
            return None, None
        
        self._progress(1, 4, "Loading and normalizing path data")
        self._vprint("=" * 80)
        self._vprint("VisualizePath - Creating pathway visualizations")
        self._vprint("=" * 80)
        
        # Build network
        self._progress(2, 4, "Building the pathway graph")
        self.build_network()
        
        # Create and show visualizations one by one
        # Order: heatmap → sankey → network
        import time
        
        # The view label reflects which views this run actually generates.
        enabled_views = []
        if plot_heatmap:
            enabled_views.append("heatmap")
        if plot_Sankey:
            enabled_views.append("Sankey")
        if plot_network:
            enabled_views.append("network")
        if enabled_views:
            self._progress(3, 4, f"Creating {' and '.join(enabled_views)} views")
        
        # 1. Create heatmap and show immediately
        if plot_heatmap:
            heatmap_path = self.create_heatmap()
            if self.showfig:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(heatmap_path))
                time.sleep(0.5)  # Small delay before next visualization
        
        # 2. Create sankey and show immediately (already handles showfig internally)
        if plot_Sankey:
            sankey_path = self.create_sankey()
            if self.showfig:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(sankey_path))
                time.sleep(0.5)  # Small delay before next visualization
        
        # 3. Create network and show immediately
        if plot_network:
            network_path = self.create_network()
            # ``create_network`` already opens the generated file when
            # ``showfig`` is enabled. Do not open it again here.
        
        # Save data
        self._progress(4, 4, "Saving visualization data and files")
        self.save_data()
        
        self._vprint("\n" + "=" * 80)
        self._vprint("✓ Visualization complete!")
        self._vprint("=" * 80)
        self._vprint(f"\nOutput files in: {self.output_folder}")
        if plot_heatmap:
            self._vprint(f"  • {self.base_filename}_heatmap.html - Connection matrix")
        if plot_Sankey:
            self._vprint(f"  • {self.base_filename}_Sankey.html - Flow-based diagram")
        if plot_network:
            self._vprint(f"  • {self.base_filename}_network.html - Interactive network")
        self._vprint(f"  • {self.base_filename}_data.xlsx - Connection data")
        
        return self.conn_df, self.G_network

    def visualize_heatmap(self, custom_row_order=None, custom_col_order=None):
        """
        Prepare data and create only the heatmap visualization.
        
        This is a convenience method that builds the network (if not already built)
        and creates just the heatmap visualization.
        
        Parameters
        ----------
        custom_row_order : list, optional
            Custom ordering for rows (sources)
        custom_col_order : list, optional
            Custom ordering for columns (targets)
            
        Returns
        -------
        str
            Path to the generated heatmap HTML file
            
        Example
        -------
        >>> vp = VisualizePath('path_type.xlsx', showfig=True)
        >>> heatmap_path = vp.visualize_heatmap()
        >>> print(f"Heatmap saved to: {heatmap_path}")
        """
        self._vprint("=" * 80)
        self._vprint("VisualizePath - Creating heatmap visualization")
        self._vprint("=" * 80)
        
        # Build network if not already built
        if self.conn_df is None or self.G_network is None:
            self.build_network()
        
        # Create heatmap
        heatmap_path = self.create_heatmap(
            custom_row_order=custom_row_order,
            custom_col_order=custom_col_order
        )
        
        # Show if requested
        if self.showfig:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(heatmap_path))
        
        self._vprint("\n" + "=" * 80)
        self._vprint("✓ Heatmap visualization complete!")
        self._vprint("=" * 80)
        self._vprint(f"\nOutput file: {heatmap_path}")
        
        return heatmap_path

    def visualize_sankey(self):
        """
        Prepare data and create only the Sankey diagram visualization.
        
        This is a convenience method that builds the network (if not already built)
        and creates just the Sankey diagram.
        
        Returns
        -------
        str
            Path to the generated Sankey HTML file
            
        Example
        -------
        >>> vp = VisualizePath('path_type.xlsx', showfig=True)
        >>> sankey_path = vp.visualize_sankey()
        >>> print(f"Sankey saved to: {sankey_path}")
        """
        self._vprint("=" * 80)
        self._vprint("VisualizePath - Creating Sankey diagram")
        self._vprint("=" * 80)
        
        # Build network if not already built
        if self.conn_df is None or self.G_network is None:
            self.build_network()
        
        # Create Sankey
        sankey_path = self.create_sankey()
        
        # Show if requested
        if self.showfig:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(sankey_path))
        
        self._vprint("\n" + "=" * 80)
        self._vprint("✓ Sankey visualization complete!")
        self._vprint("=" * 80)
        self._vprint(f"\nOutput file: {sankey_path}")
        
        return sankey_path

    def visualize_network(self):
        """
        Prepare data and create only the network graph visualization.
        
        This is a convenience method that builds the network (if not already built)
        and creates just the interactive network graph.
        
        Returns
        -------
        str
            Path to the generated network HTML file
            
        Example
        -------
        >>> vp = VisualizePath('path_type.xlsx', showfig=True)
        >>> network_path = vp.visualize_network()
        >>> print(f"Network saved to: {network_path}")
        """
        self._vprint("=" * 80)
        self._vprint("VisualizePath - Creating network visualization")
        self._vprint("=" * 80)
        
        # Build network if not already built
        if self.conn_df is None or self.G_network is None:
            self.build_network()
        
        # Create network
        network_path = self.create_network()
        # ``create_network`` already handles the optional browser open.
        
        self._vprint("\n" + "=" * 80)
        self._vprint("✓ Network visualization complete!")
        self._vprint("=" * 80)
        self._vprint(f"\nOutput file: {network_path}")
        
        return network_path


# Convenience function for quick usage
def visualize_paths(
    path_file,
    sheet_name=None,
    output_folder=None,
    source_color=None,
    intermediate_color=None,
    target_color=None,
    link_color=None,
    node_color=None,  # For backward compatibility
    network_layout='hierarchical',
    showfig=False
):
    """
    Convenience function to quickly visualize pathways.
    
    This is a shorthand for creating a VisualizePath instance and calling visualize().
    
    Parameters
    ----------
    path_file : str or pd.DataFrame
        Path to CSV/Excel file or DataFrame with pathway data
    sheet_name : str, optional
        Excel sheet name (default: auto-detect)
    output_folder : str, optional
        Output directory (default: '[filename]_figure' for files, './selected_paths' for DataFrames)
    source_color : str, optional
        Color for source nodes
    intermediate_color : str, optional
        Color for intermediate nodes
    target_color : str, optional
        Color for target nodes
    link_color : str, optional
        Color for Sankey connections
    node_color : list, optional
        [DEPRECATED] Colors for [source, intermediate] nodes.
        Use source_color and intermediate_color instead.
    network_layout : str, optional
        Layout algorithm: 'hierarchical', 'spring', 'circular', 'distributed'
    showfig : bool, optional
        Auto-open in browser (default: False)
        
    Returns
    -------
    tuple
        (conn_df, G_network) - Connection DataFrame and NetworkX graph
        
    Example
    -------
    >>> from vispath import visualize_paths
    >>> conn_df, G = visualize_paths('path_type.xlsx', showfig=True)
    
    >>> # With custom colors
    >>> conn_df, G = visualize_paths(
    ...     'path_type.xlsx',
    ...     source_color='#FF6B6B',
    ...     intermediate_color='#FFA500',
    ...     target_color='#FFD700',
    ...     showfig=True
    ... )
    """
    vp = VisualizePath(
        path_file=path_file,
        sheet_name=sheet_name,
        output_folder=output_folder,
        source_color=source_color,
        intermediate_color=intermediate_color,
        target_color=target_color,
        link_color=link_color,
        node_color=node_color,  # Pass for backward compatibility
        network_layout=network_layout,
        showfig=showfig
    )
    
    return vp.visualize()


def visualize_heatmap(
    path_file,
    sheet_name=None,
    output_folder=None,
    source_color=None,
    intermediate_color=None,
    target_color=None,
    custom_row_order=None,
    custom_col_order=None,
    showfig=False
):
    """
    Convenience function to quickly create a heatmap visualization.
    
    Parameters
    ----------
    path_file : str or pd.DataFrame
        Path to CSV/Excel file or DataFrame with pathway data
    sheet_name : str, optional
        Excel sheet name (default: auto-detect)
    output_folder : str, optional
        Output directory
    source_color : str, optional
        Color for source nodes
    intermediate_color : str, optional
        Color for intermediate nodes
    target_color : str, optional
        Color for target nodes
    custom_row_order : list, optional
        Custom ordering for rows (sources)
    custom_col_order : list, optional
        Custom ordering for columns (targets)
    showfig : bool, optional
        Auto-open in browser (default: False)
        
    Returns
    -------
    str
        Path to the generated heatmap HTML file
        
    Example
    -------
    >>> from vispath import visualize_heatmap
    >>> heatmap_path = visualize_heatmap('path_type.xlsx', showfig=True)
    """
    vp = VisualizePath(
        path_file=path_file,
        sheet_name=sheet_name,
        output_folder=output_folder,
        source_color=source_color,
        intermediate_color=intermediate_color,
        target_color=target_color,
        showfig=showfig
    )
    
    return vp.visualize_heatmap(
        custom_row_order=custom_row_order,
        custom_col_order=custom_col_order
    )


def visualize_sankey(
    path_file,
    sheet_name=None,
    output_folder=None,
    source_color=None,
    intermediate_color=None,
    target_color=None,
    link_color=None,
    showfig=False
):
    """
    Convenience function to quickly create a Sankey diagram visualization.
    
    Parameters
    ----------
    path_file : str or pd.DataFrame
        Path to CSV/Excel file or DataFrame with pathway data
    sheet_name : str, optional
        Excel sheet name (default: auto-detect)
    output_folder : str, optional
        Output directory
    source_color : str, optional
        Color for source nodes
    intermediate_color : str, optional
        Color for intermediate nodes
    target_color : str, optional
        Color for target nodes
    link_color : str, optional
        Color for Sankey connections
    showfig : bool, optional
        Auto-open in browser (default: False)
        
    Returns
    -------
    str
        Path to the generated Sankey HTML file
        
    Example
    -------
    >>> from vispath import visualize_sankey
    >>> sankey_path = visualize_sankey('path_type.xlsx', showfig=True)
    """
    vp = VisualizePath(
        path_file=path_file,
        sheet_name=sheet_name,
        output_folder=output_folder,
        source_color=source_color,
        intermediate_color=intermediate_color,
        target_color=target_color,
        link_color=link_color,
        showfig=showfig
    )
    
    return vp.visualize_sankey()


def visualize_network(
    path_file,
    sheet_name=None,
    output_folder=None,
    source_color=None,
    intermediate_color=None,
    target_color=None,
    network_layout='hierarchical',
    showfig=False
):
    """
    Convenience function to quickly create a network graph visualization.
    
    Parameters
    ----------
    path_file : str or pd.DataFrame
        Path to CSV/Excel file or DataFrame with pathway data
    sheet_name : str, optional
        Excel sheet name (default: auto-detect)
    output_folder : str, optional
        Output directory
    source_color : str, optional
        Color for source nodes
    intermediate_color : str, optional
        Color for intermediate nodes
    target_color : str, optional
        Color for target nodes
    network_layout : str, optional
        Layout algorithm: 'hierarchical', 'spring', 'circular', 'distributed'
    showfig : bool, optional
        Auto-open in browser (default: False)
        
    Returns
    -------
    str
        Path to the generated network HTML file
        
    Example
    -------
    >>> from vispath import visualize_network
    >>> network_path = visualize_network('path_type.xlsx', showfig=True)
    """
    vp = VisualizePath(
        path_file=path_file,
        sheet_name=sheet_name,
        output_folder=output_folder,
        source_color=source_color,
        intermediate_color=intermediate_color,
        target_color=target_color,
        network_layout=network_layout,
        showfig=showfig
    )
    
    return vp.visualize_network()


def VisConnMatInteractive(cmat, filename, title='', color_scale=None, showfig=True, fontsize=12, conn_df=None, matrices_dict=None, verbose=True, zmin=None, zmax=None, init_width=None, init_height=None, init_clustered=True, metric_name=None, square_cells=False):
    """Standalone interactive heatmap page (ordering, color, export controls).

    ``square_cells=True`` opens the page with cells locked square and the
    canvas height auto-fitted to the matrix shape.
    """
    # Remember whether the CALLER passed a colorscale: only then do we seed the
    # heatmap with a custom colorscale (the default below is a legacy leftover).
    provided_color_scale = color_scale
    if color_scale is None:
        color_scale = [[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']]
    '''Create interactive heatmap with comprehensive controls similar to network visualization
    
    Features:
    - Metric toggle: Switch between weight/ratio/probability (if provided)
    - Clustering toggle: Toggle between original and clustered ordering (hierarchical clustering)
    - Scale switcher: Linear / Log2 / Log10 / Sqrt
    - Colorscale selector with presets (Greens, Purples, Oranges, Blues, Reds, Viridis, etc.)
    - Font size slider
    - Export to SVG with adjustable resolution
    - Zoom/pan controls
    - Save/load layout state
    
    Parameters
    ----------
    cmat : pd.DataFrame
        Connection matrix to visualize (weight matrix if matrices_dict not provided)
    filename : str
        Output HTML filename
    title : str, optional
        Title for the heatmap
    color_scale : list, optional
        Plotly color scale (default starting point)
    showfig : bool, optional
        Whether to open in browser
    fontsize : int, optional
        Default font size for labels
    conn_df : pd.DataFrame, optional
        Connection dataframe with type information for enhanced hover labels (bodyId heatmaps only)
    matrices_dict : dict, optional
        Dictionary with keys 'weight', 'ratio', 'probability' containing different metric matrices
        If provided, enables metric toggle. Otherwise uses cmat as weight matrix only.
    verbose : bool, optional
        Control print output. Default True.
    zmin : float, optional
        Minimum value for fixed color scale. If None, uses data range.
    zmax : float, optional
        Maximum value for fixed color scale. If None, uses data range.
    init_width : int, optional
        Initial width of the heatmap in pixels. If None, auto-calculated based on matrix size.
    init_height : int, optional
        Initial height of the heatmap in pixels. If None, auto-calculated based on matrix size.
    init_clustered : bool, optional
        Whether to show clustered ordering by default. Default True.
        Set to True for similarity matrices (connectivity profiling).
        Set to False to show original ordering by default.
    '''
    
    # Helper function for verbose printing
    def _vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    # Handle multiple matrices for metric toggle
    has_multiple_metrics = matrices_dict is not None and isinstance(matrices_dict, dict)
    
    if has_multiple_metrics:
        # Use provided matrices dictionary
        available_metrics = []
        matrices_data = {}
        
        if 'weight' in matrices_dict and matrices_dict['weight'] is not None:
            available_metrics.append('weight')
            matrices_data['weight'] = matrices_dict['weight'].values.copy()
        
        if 'ratio' in matrices_dict and matrices_dict['ratio'] is not None:
            available_metrics.append('ratio')
            matrices_data['ratio'] = matrices_dict['ratio'].values.copy()
        
        if 'probability' in matrices_dict and matrices_dict['probability'] is not None:
            available_metrics.append('probability')
            matrices_data['probability'] = matrices_dict['probability'].values.copy()
        
        # Use first available metric as default
        default_metric = available_metrics[0] if available_metrics else 'weight'
        data_linear = matrices_data.get(default_metric, cmat.values.copy())
        metric_type = default_metric
    else:
        # Single matrix mode - determine metric type from title/filename
        available_metrics = ['weight']  # Only one metric available
        matrices_data = {}
        
        metric_type = 'weight'
        if 'ratio' in title.lower() or 'ratio' in filename.lower():
            metric_type = 'ratio'
            available_metrics = ['ratio']
        elif 'transmission' in title.lower() or 'probability' in title.lower():
            metric_type = 'probability'
            available_metrics = ['probability']
        
        data_linear = cmat.values.copy()
        matrices_data[metric_type] = data_linear
    
    # Display label for the current metric: explicit metric_name wins over the
    # heuristic ('weight' -> "Synapses", ratio/probability -> capitalized).
    metric_label = metric_name if metric_name else metric_type.capitalize()
    
    is_large = cmat.shape[0] > 100 or cmat.shape[1] > 100
    
    # Check sparsity for potential optimization
    zero_count = np.count_nonzero(data_linear == 0)
    sparsity_ratio = zero_count / data_linear.size
    is_sparse = sparsity_ratio > 0.5  # More than 50% zeros
    
    # Compute hierarchical clustering with multiple methods for row/column ordering
    _vprint("  Computing hierarchical clustering...")
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    
    # Store clustering results for all methods
    clustering_methods = ['ward', 'average', 'complete', 'single']
    clustering_results = {}
    
    try:
        for method in clustering_methods:
            method_results = {}
            
            # Cluster rows (source neurons)
            if data_linear.shape[0] > 1:
                # Use euclidean distance (required for ward, good for others)
                row_distances = pdist(data_linear, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(row_distances)):
                    raise ValueError("Non-finite distances in row clustering")
                row_linkage = linkage(row_distances, method=method)
                method_results['row_order'] = leaves_list(row_linkage).tolist()
            else:
                method_results['row_order'] = [0]
            
            # Cluster columns (target neurons)
            if data_linear.shape[1] > 1:
                col_distances = pdist(data_linear.T, metric='euclidean')
                # Check for non-finite values
                if not np.all(np.isfinite(col_distances)):
                    raise ValueError("Non-finite distances in column clustering")
                col_linkage = linkage(col_distances, method=method)
                method_results['col_order'] = leaves_list(col_linkage).tolist()
            else:
                method_results['col_order'] = [0]
            
            clustering_results[method] = method_results
        
        # Use Ward as default (best for most connectome data)
        row_order_clustered = np.array(clustering_results['ward']['row_order'])
        col_order_clustered = np.array(clustering_results['ward']['col_order'])
        
        clustering_successful = True
        _vprint(f"  ✓ Clustering complete: {len(row_order_clustered)} rows, {len(col_order_clustered)} cols")
        _vprint(f"  Available methods: Ward (default), Average, Complete, Single")
    except Exception as e:
        _vprint(f"  ⚠ Clustering failed: {e}")
        _vprint(f"  Using original order")
        row_order_clustered = np.array(range(data_linear.shape[0]))
        col_order_clustered = np.array(range(data_linear.shape[1]))
        clustering_successful = False
        clustering_results = {}
    
    # Store both original and clustered orders
    row_order_original = list(range(data_linear.shape[0]))
    col_order_original = list(range(data_linear.shape[1]))
    
    # For large matrices, reduce precision to save HTML size
    # Keep more precision for ratio/probability metrics
    if is_large:
        if metric_type in ['ratio', 'probability']:
            # Keep 4 decimal places for ratios/probabilities
            data_linear = np.round(data_linear, 4)
        else:
            # For synapse counts, round to integers (no precision loss)
            data_linear = np.round(data_linear, 0)
    
    # Deep optimization: For very large matrices, compute transforms in JavaScript
    # This saves ~75% of HTML file size by not embedding pre-computed transforms
    use_lazy_transforms = is_large and data_linear.size > 50000
    
    # Sparse matrix optimization: For matrices with >70% zeros, use COO format
    use_sparse_format = is_large and sparsity_ratio > 0.7 and data_linear.size > 50000
    sparse_data = None
    
    if use_sparse_format:
        # Convert to COO (Coordinate) format: store only non-zero values
        rows, cols = np.nonzero(data_linear)
        values = data_linear[rows, cols]
        sparse_data = {
            'rows': rows.tolist(),
            'cols': cols.tolist(),
            'values': values.tolist(),
            'shape': list(data_linear.shape)
        }
        _vprint(f"  Using sparse format: {sparsity_ratio*100:.1f}% zeros, storing {len(values)} values instead of {data_linear.size}")
    
    if use_lazy_transforms:
        # Store only linear data; transforms computed client-side
        data_log2 = None
        data_log10 = None
        data_sqrt = None
    else:
        # Pre-compute for small matrices (faster initial display)
        # Handle negative values: sign(v) * transform(|v|)
        # Suppress warnings for edge cases (zeros, negatives handled by np.where)
        with np.errstate(divide='ignore', invalid='ignore'):
            data_log2 = np.where(data_linear >= 0, 
                                 np.log2(data_linear + 1), 
                                 -np.log2(-data_linear + 1))
            data_log10 = np.where(data_linear >= 0, 
                                  np.log10(data_linear + 1), 
                                  -np.log10(-data_linear + 1))
            data_sqrt = np.where(data_linear >= 0, 
                                np.sqrt(data_linear), 
                                -np.sqrt(-data_linear))
            # Replace any NaN/inf values that may have occurred
            data_log2 = np.nan_to_num(data_log2, nan=0.0, posinf=0.0, neginf=0.0)
            data_log10 = np.nan_to_num(data_log10, nan=0.0, posinf=0.0, neginf=0.0)
            data_sqrt = np.nan_to_num(data_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        
        if is_large:
            if metric_type in ['ratio', 'probability']:
                data_log2 = np.round(data_log2, 4)
                data_log10 = np.round(data_log10, 4)
                data_sqrt = np.round(data_sqrt, 4)
            else:
                data_log2 = np.round(data_log2, 2)
                data_log10 = np.round(data_log10, 2)
                data_sqrt = np.round(data_sqrt, 2)
    
    # Create hover text with original values
    # If conn_df is provided, create type lookup for bodyId heatmaps
    type_lookup = None
    
    if conn_df is not None and 'bodyId_pre' in conn_df.columns and 'type_pre' in conn_df.columns:
        # Create lookup dictionaries for bodyId -> type
        # Convert bodyId keys to strings to match matrix index/columns
        type_lookup = {
            'pre': {str(k): v for k, v in conn_df.set_index('bodyId_pre')['type_pre'].to_dict().items()},
            'post': {str(k): v for k, v in conn_df.set_index('bodyId_post')['type_post'].to_dict().items()}
        }
    
    # Generate hover text with actual labels for all matrix sizes
    # No longer use compact mode - always show full information with proper labels
    hover_text = []
    for i, row_label in enumerate(cmat.index):
        hover_row = []
        for j, col_label in enumerate(cmat.columns):
            value = cmat.iloc[i, j]
            # NaN/Inf are documented as "no connection" - show an em dash
            # instead of crashing int() formatting.
            try:
                finite_value = float(value)
                is_finite = np.isfinite(finite_value)
            except (TypeError, ValueError):
                is_finite = False
            if not is_finite:
                value_str = '—'
            elif metric_type == 'ratio' or metric_type == 'probability':
                value_str = f'{value:.4f}'
            else:
                value_str = f'{int(value):,}' if value == int(value) else f'{value:,.2f}'
            
            row_label_safe = html_escape(row_label)
            col_label_safe = html_escape(col_label)
            # Always use actual labels with type info if available
            if type_lookup:
                try:
                    # Labels are already strings, use them directly for type lookup
                    row_id = str(row_label)
                    col_id = str(col_label)
                    row_type = html_escape(type_lookup['pre'].get(row_id, 'Unknown'))
                    col_type = html_escape(type_lookup['post'].get(col_id, 'Unknown'))
                    hover_row.append(f'<b>Source:</b> {row_label_safe} ({row_type})<br><b>Target:</b> {col_label_safe} ({col_type})<br><b>{metric_label}:</b> {value_str}')
                except:
                    # Fall back to label-only display if type lookup fails
                    hover_row.append(f'<b>Source:</b> {row_label_safe}<br><b>Target:</b> {col_label_safe}<br><b>{metric_label}:</b> {value_str}')
            else:
                # No type info available - just show labels
                hover_row.append(f'<b>Source:</b> {row_label_safe}<br><b>Target:</b> {col_label_safe}<br><b>{metric_label}:</b> {value_str}')
        hover_text.append(hover_row)
    
    # Determine axis labels - ALWAYS use actual names, not numeric indices
    # Even for large matrices, show proper labels (optimization only affects hover text)
    x_labels = cmat.columns.astype(str).tolist()
    y_labels = cmat.index.astype(str).tolist()
    
    # Calculate dynamic left margin based on longest row label
    # Approximate character width: ~7px per character at default font size
    max_row_label_length = max(len(str(label)) for label in y_labels) if y_labels else 0
    max_col_label_length = max(len(str(label)) for label in x_labels) if x_labels else 0
    # Calculate left margin: min 120px, max 400px, ~7px per character + 40px padding
    dynamic_left_margin = min(400, max(120, max_row_label_length * 7 + 40))
    # Calculate bottom margin for rotated column labels: min 120px, ~5px per char (due to 45° rotation)
    dynamic_bottom_margin = min(300, max(120, max_col_label_length * 5 + 40))
    
    # Generate unique storage key for this heatmap
    from datetime import datetime
    output_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp_hash = datetime.now().strftime('%Y%m%d%H%M%S')
    storage_key = f"heatmap_settings_{output_name}#{timestamp_hash}"
    
    # Determine default colorscale name
    default_colorscale = 'Greens'
    if 'ratio' in filename.lower():
        default_colorscale = 'Oranges'
    elif 'transmission' in filename.lower() or 'probability' in filename.lower():
        default_colorscale = 'Purples'
    
    # Create HTML with comprehensive interactive controls
    # Optional: open with square cells locked (callers generating similarity
    # heatmaps request this so the matrix cells are square on first render).
    SQUARE_CELLS_ONLOAD = ''
    if square_cells:
        SQUARE_CELLS_ONLOAD = (
            "window.addEventListener('load', function() {"
            " setTimeout(function() {"
            " if (!window.__drocatHasSavedSettings && !squareCellsLocked)"
            " { makeSquareCells(); }"
            " }, 250); });"
        )

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html_escape(title)}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            user-select: text;
        }}
        
        .main-container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .controls {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 8px;
            margin-bottom: 10px;
        }}
        
        .control-section {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        
        .control-section h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        .button-group {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }}
        
        button {{
            padding: 6px 10px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s;
            color: #495057;
        }}
        
        button:hover {{
            background: #f8f9fa;
            border-color: #adb5bd;
        }}
        
        button.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        
        button.export-btn {{
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }}
        
        button.export-btn:hover {{
            background: #1976D2;
            border-color: #1976D2;
        }}
        
        button.save-btn {{
            background: #FF9800;
            color: white;
            border-color: #FF9800;
        }}
        
        button.save-btn:hover {{
            background: #F57C00;
            border-color: #F57C00;
        }}
        
        select {{
            width: 100%;
            padding: 4px 6px;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 11px;
            background: white;
            cursor: pointer;
            color: #495057;
        }}
        
        select:focus {{
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
        }}
        
        .slider-control {{
            margin-bottom: 6px;
        }}
        
        .slider-control label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
            font-size: 10px;
            color: #495057;
            font-weight: 500;
        }}
        
        .slider-value {{
            color: #4CAF50;
            font-weight: 600;
        }}
        
        input[type="range"] {{
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #dee2e6;
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        input[type="range"]::-webkit-slider-thumb:hover {{
            background: #45a049;
            transform: scale(1.15);
        }}
        
        input[type="range"]::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
            border: none;
        }}
        
        #heatmap-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        #heatmap {{
            width: 100%;
            height: 1200px;
        }}
        
        .status-message {{
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            text-align: center;
            margin-top: 8px;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        
        .status-info {{
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        
        .status-error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        
        .info-box {{
            background: #e7f3ff;
            border-left: 3px solid #2196F3;
            padding: 8px;
            border-radius: 3px;
            font-size: 10px;
            color: #1976D2;
            margin-top: 8px;
            line-height: 1.4;
        }}
        
        .info-box strong {{
            display: block;
            margin-bottom: 3px;
            font-size: 11px;
        }}
        
        .drag-item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 6px 8px;
            margin-bottom: 4px;
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }}
        
        .drag-item:hover {{
            background: #f0f0f0;
            border-color: #4CAF50;
        }}
        
        .drag-item.dragging {{
            opacity: 0.5;
            background: #e3f2fd;
        }}
        
        .drag-item.drag-over {{
            border-top: 3px solid #4CAF50;
        }}
        
        .drag-handle {{
            margin-right: 6px;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="controls">
            <div class="controls-grid">
                <!-- Metric, Ordering & Scale Combined Section -->
                {'<div class="control-section" id="metricOrderingSection">' if has_multiple_metrics else '<div class="control-section">'}
                    {'<h3>📊 Metric, Ordering & Scale</h3>' if has_multiple_metrics else '<h3>🔀 Ordering & Scale</h3>'}
                    
                    <!-- Metric Selection (if multiple metrics available) -->
                    {'<div style="margin-bottom: 8px;"><label style="font-size: 10px; display: block; margin-bottom: 2px;">Metric:</label>' if has_multiple_metrics else '<!-- Single metric mode -->'}
                    {'<select id="metricSelect" onchange="updateMetric()">' if has_multiple_metrics else ''}
                        {'<option value="weight">Synapse Count</option>' if has_multiple_metrics and 'weight' in available_metrics else ''}
                        {'<option value="ratio"' + (' selected' if metric_type == 'ratio' else '') + '>Connection Ratio</option>' if has_multiple_metrics and 'ratio' in available_metrics else ''}
                        {'<option value="probability"' + (' selected' if metric_type == 'probability' else '') + '>Traversal Probability</option>' if has_multiple_metrics and 'probability' in available_metrics else ''}
                    {'</select></div>' if has_multiple_metrics else ''}
                    
                    <!-- Clustering Toggle -->
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Ordering:</label>
                        <div class="button-group">
                            <button id="btn-original" class="{'' if init_clustered and clustering_successful else 'active'}" onclick="toggleClustering('original')">Original</button>
                            <button id="btn-clustered" class="{'active' if init_clustered and clustering_successful else ''}" onclick="toggleClustering('clustered')">Clustered</button>
                        </div>
                    </div>
                    
                    <!-- Clustering Method Selection -->
                    <div id="clusteringMethodSection" style="margin-bottom: 8px; display: {'block' if init_clustered and clustering_successful else 'none'};">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Clustering Method:</label>
                        <select id="clusteringMethodSelect" onchange="updateClusteringMethod()" style="width: 100%; font-size: 10px; padding: 4px;">
                            <option value="ward">Ward (Compact Clusters)</option>
                            <option value="average">Average (Balanced)</option>
                            <option value="complete">Complete (Tight Clusters)</option>
                            <option value="single">Single (Loose Clusters)</option>
                        </select>
                    </div>
                    
                    <!-- Scale Selection -->
                    <div>
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">Scale:</label>
                        <div class="button-group">
                            <button id="btn-linear" class="active" onclick="setScale('linear')">Linear</button>
                            <button id="btn-log2" onclick="setScale('log2')">Log₂</button>
                            <button id="btn-log10" onclick="setScale('log10')">Log₁₀</button>
                            <button id="btn-sqrt" onclick="setScale('sqrt')">√</button>
                        </div>
                    </div>
                    
                    <!-- Data Filter -->
                    <div style="margin-top: 8px;">
                        <label style="font-size: 10px; display: block; margin-bottom: 2px;">
                            🔍 Data Filter:
                            <button onclick="resetDataFilter()" style="padding: 2px 6px; font-size: 9px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reset filter">🔄</button>
                        </label>
                        <input type="text" id="dataFilterInput" placeholder="OR: <5, >100 | AND: (>=5, <=10)" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="applyDataFilter()">
                        <div style="font-size: 8px; color: #888; margin-top: 1px;">Comma = OR, Parentheses = AND. E.g., <5, (>=10, <=20), >100</div>
                        <div style="margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap;">
                            <div class="button-group" style="flex: 1; min-width: 120px;">
                                <button id="btn-filter-hide" class="active" onclick="setFilterMode('hide')" title="Hide filtered rows/columns">Hide</button>
                                <button id="btn-filter-zero" onclick="setFilterMode('zero')" title="Show filtered values as 0">Zero</button>
                            </div>
                            <label style="font-size: 9px; display: flex; align-items: center; gap: 2px; cursor: pointer;" title="Show or hide filtered rows and columns">
                                <input type="checkbox" id="showFilteredRowsCols" onchange="toggleFilteredVisibility()" checked>
                                <span>Show rows/cols</span>
                            </label>
                        </div>
                        <div id="filterStatus" style="font-size: 9px; color: #666; margin-top: 2px; min-height: 14px;"></div>
                    </div>
                </div>
                
                <!-- Color -->
                <div class="control-section" id="colorscaleSection">
                    <h3>🎨 Color</h3>
                    <select id="colorscaleSelect" onchange="updateColorscale()" style="margin-bottom: 8px;">
                        <option value="Greens" {'selected' if default_colorscale == 'Greens' else ''}>Greens</option>
                        <option value="Purples" {'selected' if default_colorscale == 'Purples' else ''}>Purples</option>
                        <option value="Oranges" {'selected' if default_colorscale == 'Oranges' else ''}>Oranges</option>
                        <option value="Blues" {'selected' if default_colorscale == 'Blues' else ''}>Blues</option>
                        <option value="Reds">Reds</option>
                        <option value="Viridis">Viridis</option>
                        <option value="Plasma">Plasma</option>
                        <option value="Inferno">Inferno</option>
                        <option value="Magma">Magma</option>
                        <option value="Cividis">Cividis</option>
                        <option value="Hot">Hot</option>
                        <option value="Jet">Jet</option>
                        <option value="RdBu">Red-Blue (Diverging)</option>
                        <option value="RdYlGn">Red-Yellow-Green</option>
                        <option value="Custom" {'selected' if provided_color_scale is not None else ''}>Custom</option>
                    </select>
                    
                    <div id="customColorSection">
                        <div style="margin-bottom: 6px;">
                            <label style="display: block; margin-bottom: 3px; font-size: 10px;">
                                <input type="checkbox" id="use3PointScale" onchange="toggle3PointScale()"> 
                                3-Point Scale (diverging)
                            </label>
                        </div>
                        <div id="twoPointColors">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax" value="#68379c" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax2" placeholder="Auto" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div id="threePointColors" style="display: none;">
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Min (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMin3" value="#0000ff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMin3" value="0" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Mid (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMid3" value="#ffffff" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMid3" value="0.5" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <label style="font-size: 10px; display: block; margin-bottom: 2px;">Max (value):</label>
                                <div style="display: flex; gap: 4px;">
                                    <input type="color" id="colorMax3" value="#ff0000" style="width: 40px; height: 26px; cursor: pointer;">
                                    <input type="number" id="valueMax3" value="1" step="any" style="flex: 1; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;">
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 4px; margin-top: 4px;">
                            <button onclick="applyCustomColors()" style="flex: 1; font-size: 10px;">Apply</button>
                            <button onclick="resetToAutoColors()" style="flex: 1; font-size: 10px;">Auto</button>
                        </div>
                    </div>
                </div>
                
                <!-- Font Size & Colorbar Settings -->
                <div class="control-section">
                    <h3>🎚️ Display</h3>
                    <div class="slider-control">
                        <label>
                            <span>Font Size:</span>
                            <span class="slider-value" id="fontSizeValue">{fontsize}px</span>
                        </label>
                        <input type="range" id="fontSizeSlider" min="8" max="48" value="{fontsize}" step="1" oninput="updateFontSize(this.value)">
                    </div>
                    <div style="margin-top: 8px; display: flex; gap: 4px;">
                        <button id="toggleLabelsBtn" onclick="toggleLabels()" style="flex: 1;">
                            {'🏷️ Hide Labels' if not is_large else '🏷️ Show Labels'}
                        </button>
                        <button id="toggleCellValuesBtn" onclick="toggleCellValues()" style="flex: 1;">
                            🔢 Show Values
                        </button>
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Cell Value Size:</span>
                            <span class="slider-value" id="cellValueSizeValue">10px</span>
                        </label>
                        <input type="range" id="cellValueSizeSlider" min="6" max="48" value="10" step="1" oninput="updateCellValueSize(this.value)">
                    </div>
                    <div style="margin-top: 8px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 4px;">Ignore Values (comma-separated):</label>
                        <input type="text" id="ignoreValuesInput" placeholder="e.g., 0, >20, <=5" style="width: 100%; padding: 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px; box-sizing: border-box;" oninput="updateIgnoredValues()">
                    </div>
                    <div class="slider-control" style="margin-top: 8px;">
                        <label>
                            <span>Contrast Threshold:</span>
                            <span class="slider-value" id="contrastThresholdValue">0.5000</span>
                            <button onclick="reverseContrastColors()" style="padding: 2px 6px; font-size: 10px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 4px;" title="Reverse black/white colors">🔄</button>
                        </label>
                        <input type="range" id="contrastThresholdSlider" min="0" max="1" value="0.5" step="0.0001" oninput="updateContrastThreshold(this.value)">
                    </div>
                </div>
                
                <!-- Plot Dimensions -->
                <div class="control-section">
                    <h3>📐 Plot Size</h3>
                    <div class="slider-control">
                        <label>
                            <span>Width:</span>
                            <span class="slider-value" id="widthValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="widthSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="widthInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                    </div>
                    <div class="slider-control">
                        <label>
                            <span>Height:</span>
                            <span class="slider-value" id="heightValue">800px</span>
                        </label>
                        <div style="display: flex; gap: 4px; align-items: center;">
                            <input type="range" id="heightSlider" min="400" max="2400" value="800" step="20" oninput="updatePlotSize()" style="flex: 1;">
                            <input type="number" id="heightInput" value="800" min="100" step="20" style="width: 70px; padding: 2px 4px; font-size: 10px; border: 1px solid #dee2e6; border-radius: 3px;" oninput="updatePlotSizeFromInput()">
                        </div>
                    </div>
                    <div style="display: flex; gap: 4px;">
                        <button id="squareCellsBtn" onclick="makeSquareCells()" style="flex: 1;">⬜ Square Cells</button>
                        <button onclick="resetPlotSize()" style="flex: 1;">🔄 Reset</button>
                    </div>
                    <div style="margin-top: 8px;">
                        <button id="transposeBtn" onclick="transposeMatrix()" style="width: 100%;">🔄 Swap Rows ↔ Columns</button>
                    </div>
                </div>
                
                <!-- Row/Column Ordering -->
                <div class="control-section">
                    <h3>📋 Row/Column Order</h3>
                    <button onclick="toggleOrderPanel('rows')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Rows</button>
                    <button onclick="toggleOrderPanel('cols')" style="width: 100%; font-size: 10px; margin-bottom: 4px;">📑 Reorder Columns</button>
                    <button onclick="resetOrder()" style="width: 100%; font-size: 10px;">🔄 Reset to Original</button>
                    
                </div>
                
                <!-- Export & Saving -->
                <div class="control-section">
                    <h3>💾 Export & Saving</h3>
                    <div class="slider-control" style="margin-bottom: 8px;">
                        <label>
                            <span>Export Scale (PNG):</span>
                            <span class="slider-value" id="exportScaleValue">2x</span>
                        </label>
                        <input type="range" id="exportScaleSlider" min="1" max="5" value="2" step="0.5" oninput="updateExportScale(this.value)">
                    </div>
                    <div class="button-group" style="flex-direction: column; margin-bottom: 8px;">
                        <button class="export-btn" onclick="exportPNG()" style="width: 100%;">📥 Export PNG</button>
                        <button class="export-btn" onclick="exportSVG()" style="width: 100%;">📥 Export SVG</button>
                    </div>
                    <div class="button-group">
                        <button class="save-btn" onclick="saveSettings()">💾 Save</button>
                        <button class="save-btn" onclick="loadSettings()">📂 Load</button>
                        <button onclick="resetSettings()">🔄 Reset</button>
                    </div>
                    <div id="settingsStatus"></div>
                </div>
                
                <!-- Background Color Toggle -->
                <div class="control-section">
                    <h3>🎨 Background</h3>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <button id="bgToggleBtn" onclick="toggleBackground()" style="flex: 1; padding: 6px; font-size: 11px;">White</button>
                        <input type="color" id="customBgColor" value="#f5f5f5" style="width: 35px; height: 28px; border: 1px solid #ddd; border-radius: 3px; cursor: pointer;" onchange="applyCustomBackground()">
                    </div>
                </div>
            </div>
            
            <div class="info-box">
                <strong>💡 Tips:</strong>
                Use Log₂ or Log₁₀ scales for large dynamic ranges • 
                Adjust plot size with width/height sliders for better visualization • 
                Use export scale (1x-5x) to control SVG resolution • 
                3-point custom colors ideal for diverging data (negative → zero → positive) • 
                Hover over cells for details • 
                Zoom and pan with mouse • 
                Settings persist across sessions
            </div>
        </div>
        
        <div id="heatmap-container">
            <div id="heatmap"></div>
        </div>
    </div>
    
    <!-- Floating Reorder Panel -->
    <div id="orderPanel" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                 background: white; border: 2px solid #333; border-radius: 8px; padding: 16px; 
                                 box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; min-width: 300px; max-width: 400px; 
                                 max-height: 70vh; flex-direction: column; display: none;">
        <div style="margin-bottom: 12px; border-bottom: 2px solid #ddd; padding-bottom: 8px;">
            <label id="orderPanelLabel" style="font-size: 14px; font-weight: bold; color: #333;"></label>
        </div>
        <div id="orderList" style="font-size: 12px; overflow-y: auto; flex: 1; margin-bottom: 12px;"></div>
        <button onclick="closeOrderPanel()" style="width: 100%; font-size: 12px; padding: 8px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">✓ Close</button>
    </div>
    
    <!-- Overlay backdrop for floating panel -->
    <div id="orderPanelBackdrop" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
                                        background: rgba(0,0,0,0.5); z-index: 9999;" onclick="closeOrderPanel()"></div>
    
    <script>
        // Metric toggle support
        const availableMetrics = {json.dumps(available_metrics)};
        const hasMultipleMetrics = availableMetrics.length > 1;
        let currentMetric = '{metric_type}';
        
        // Store all metric matrices
        const metricsData = {{}};
'''
    
    # Add metric data assignments
    for metric in available_metrics:
        html_content += f"        metricsData['{metric}'] = {json.dumps(matrices_data[metric].tolist())};\n"
    
    html_content += f'''
        
        // Data for different scales
        const sparseData = {json.dumps(sparse_data) if sparse_data is not None else 'null'};
        const useSparseFormat = sparseData !== null;
        
        // Get current metric data
        let dataLinear = metricsData[currentMetric];
        
        const dataLog2 = {'null' if data_log2 is None else json.dumps(data_log2.tolist())};
        const dataLog10 = {'null' if data_log10 is None else json.dumps(data_log10.tolist())};
        const dataSqrt = {'null' if data_sqrt is None else json.dumps(data_sqrt.tolist())};
        const xLabels = {json_safe(x_labels)};
        const yLabels = {json_safe(y_labels)};
        const storageKey = '{js_escape(storage_key)}';
        const useLazyTransforms = {json.dumps(use_lazy_transforms)};
        
        // Track current row/column order (for interactive reordering)
        // If init_clustered is true and clustering succeeded, start with clustered order
        let currentXLabels = xLabels.slice();
        let currentYLabels = yLabels.slice();
        
        // Hover text - always use full array with proper labels (no compact mode)
        const hoverText = {json_safe(hover_text)};
        
        // Cache for lazy-computed transforms
        let cachedDataLog2 = null;
        let cachedDataLog10 = null;
        let cachedDataSqrt = null;
        
        // Clustering data - row and column orders for all methods
        const rowOrderOriginal = {json.dumps(row_order_original)};
        const colOrderOriginal = {json.dumps(col_order_original)};
        const clusteringAvailable = {json.dumps(clustering_successful)};
        
        // All clustering method results
        const clusteringResults = {json.dumps(clustering_results)};
        
        // Default to Ward method
        const rowOrderClustered = {json.dumps(row_order_clustered.tolist())};
        const colOrderClustered = {json.dumps(col_order_clustered.tolist())};
        
        // Current settings
        let currentScale = 'linear';
        let currentColorscale = '{'Custom' if provided_color_scale is not None else default_colorscale}';
        let currentFontSize = {fontsize};
        let useAutoRange = {json.dumps(zmin is None and zmax is None)};
        let customZmin = {json.dumps(float(zmin)) if zmin is not None else 'null'};
        let customZmax = {json.dumps(float(zmax)) if zmax is not None else 'null'};
        let customColorScale = {json_safe(provided_color_scale) if provided_color_scale is not None else 'null'};  // Custom color scale (caller-provided or built by the color panel)
        let use3PointScale = false;
        let currentWidth = {init_width if init_width else 800};
        let currentHeight = {init_height if init_height else 800};
        let exportScale = 2;
        let squareCellsLocked = false;  // Track if square cells are locked
        let showLabels = !{json.dumps(is_large)};  // Show labels for small matrices, hide for large
        let showCellValues = false;  // Track if cell values should be displayed in cells (default: false)
        let cellValueFontSize = 10;  // Font size for cell value annotations
        let ignoredValues = new Set();  // Set of values to ignore when displaying cell values
        let contrastThreshold = 0.5;  // Luminance threshold for contrast color (0-1, default: 0.5)
        let reverseContrast = false;  // Whether to reverse black/white contrast colors
        let useClusteredOrder = {json.dumps(init_clustered and clustering_successful)};  // Track current ordering mode (use init_clustered parameter)
        let currentClusteringMethod = 'ward';  // Current clustering method (ward, average, complete, single)
        
        // Dynamic margins based on label lengths
        const dynamicLeftMargin = {dynamic_left_margin};
        const dynamicBottomMargin = {dynamic_bottom_margin};
        let isTransposed = false;  // Track if matrix is transposed
        const metricType = '{metric_type}';
        const isLarge = {json.dumps(is_large)};
        const originalTitle = '{js_escape(title)}';
        
        // Data filter state
        let dataFilterActive = false;
        let dataFilterExpressions = [];
        let filteredRowIndices = [];  // Indices of rows to show after filtering
        let filteredColIndices = [];  // Indices of columns to show after filtering
        let dataFilterMode = 'hide';  // 'hide' = hide rows/cols, 'zero' = show values as 0
        let showFilteredRowsCols = true;  // Whether to show filtered rows/cols (when in 'zero' mode)
        let zeroMaskMatrix = null;  // Boolean matrix for cells to show as 0
        
        // Function to generate hover text dynamically when needed
        // Hover text is pre-generated in Python with proper labels
        // This function regenerates it when switching metrics (multi-metric mode)
        function generateHoverText() {{
            if (!hasMultipleMetrics) {{
                return hoverText;  // Use pre-generated hover text for single-metric mode
            }}
            
            // Generate hover text on-the-fly for multi-metric mode
            const rows = dataLinear.length;
            const cols = dataLinear[0].length;
            const result = new Array(rows);
            
            // Get metric display name
            const metricNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const currentMetricName = metricNames[currentMetric] || currentMetric;
            
            for (let i = 0; i < rows; i++) {{
                result[i] = new Array(cols);
                for (let j = 0; j < cols; j++) {{
                    const value = dataLinear[i][j];
                    let valueStr;
                    if (currentMetric === 'ratio' || currentMetric === 'probability') {{
                        valueStr = value.toFixed(4);
                    }} else {{
                        valueStr = Math.floor(value) === value ? 
                            value.toLocaleString() : 
                            value.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                    }}

                    // Always use actual labels from yLabels and xLabels
                    const srcLabel = yLabels[i];
                    const tgtLabel = xLabels[j];
                    result[i][j] = '<b>Source:</b> ' + srcLabel + '<br><b>Target:</b> ' + tgtLabel + '<br><b>' + currentMetricName + ':</b> ' + valueStr;
                }}
            }}
            return result;
        }}
        
        function getDataForScale(scale) {{
            if (!useLazyTransforms) {{
                // Use pre-computed data for small matrices
                switch(scale) {{
                    case 'log2': return dataLog2;
                    case 'log10': return dataLog10;
                    case 'sqrt': return dataSqrt;
                    default: return dataLinear;
                }}
            }}
            
            // Lazy computation for large matrices
            switch(scale) {{
                case 'log2':
                    if (cachedDataLog2 === null) {{
                        console.log('Computing log₂ transform...');
                        cachedDataLog2 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log2(|v| + 1)
                            if (v < 0) return -Math.log2(-v + 1);
                            return Math.log2(v + 1);
                        }}));
                    }}
                    return cachedDataLog2;
                case 'log10':
                    if (cachedDataLog10 === null) {{
                        console.log('Computing log₁₀ transform...');
                        cachedDataLog10 = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * log10(|v| + 1)
                            if (v < 0) return -Math.log10(-v + 1);
                            return Math.log10(v + 1);
                        }}));
                    }}
                    return cachedDataLog10;
                case 'sqrt':
                    if (cachedDataSqrt === null) {{
                        console.log('Computing √ transform...');
                        cachedDataSqrt = dataLinear.map(row => row.map(v => {{
                            // Handle negative values: sign(v) * sqrt(|v|)
                            if (v < 0) return -Math.sqrt(-v);
                            return Math.sqrt(v);
                        }}));
                    }}
                    return cachedDataSqrt;
                default:
                    return dataLinear;
            }}
        }}
        
        function getScaleLabel(scale) {{
            switch(scale) {{
                case 'log2': return ' (log₂)';
                case 'log10': return ' (log₁₀)';
                case 'sqrt': return ' (√)';
                default: return '';
            }}
        }}
        
        function getDataRange(data) {{
            let min = Infinity;
            let max = -Infinity;
            for (let row of data) {{
                for (let val of row) {{
                    // Skip non-finite cells (documented as "no connection")
                    if (!Number.isFinite(val)) continue;
                    if (val < min) min = val;
                    if (val > max) max = val;
                }}
            }}
            // Constant (or all non-finite) matrices: avoid a zero range
            if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {{
                if (!Number.isFinite(min)) {{ min = 0; max = 1; }}
                else {{ max = min + 1; }}
            }}
            return {{min, max}};
        }}
        
        function reorderData(data, rowOrder, colOrder) {{
            // Reorder rows and columns of the data matrix according to given orders
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = data[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function reorderLabels(labels, order) {{
            // Reorder labels according to given order
            const reordered = new Array(order.length);
            for (let i = 0; i < order.length; i++) {{
                reordered[i] = labels[order[i]];
            }}
            return reordered;
        }}
        
        function reorderHoverText(hoverText, rowOrder, colOrder) {{
            // Reorder hover text according to given orders
            if (hoverText === null) return null;
            const reordered = new Array(rowOrder.length);
            for (let i = 0; i < rowOrder.length; i++) {{
                reordered[i] = new Array(colOrder.length);
                for (let j = 0; j < colOrder.length; j++) {{
                    reordered[i][j] = hoverText[rowOrder[i]][colOrder[j]];
                }}
            }}
            return reordered;
        }}
        
        function createHeatmap() {{
            // Safety check: ensure data is available
            if (!dataLinear || dataLinear.length === 0) {{
                console.error('Cannot create heatmap: data not available');
                return;
            }}
            
            // IMPORTANT: Create deep copies of data to avoid mutating the cached originals
            let data = getDataForScale(currentScale).map(row => row.slice());
            let dataOriginal = dataLinear.map(row => row.slice()); // Keep original for cell values
            const scaleLabel = getScaleLabel(currentScale);
            
            // Determine which labels to use based on transpose state
            let displayXLabels, displayYLabels;
            let currentHoverText = generateHoverText();
            
            if (isTransposed) {{
                // When transposed: rows become columns, columns become rows
                // So we use the swapped tracking variables
                displayXLabels = currentYLabels.slice();
                displayYLabels = currentXLabels.slice();
                
                // Transpose the data matrix
                data = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
                dataOriginal = dataOriginal[0].map((_, colIndex) => dataOriginal.map(row => row[colIndex]));
                
                // Transpose hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = currentHoverText[0].map((_, colIndex) => 
                        currentHoverText.map(row => row[colIndex])
                    );
                }}
                
                // Now apply reordering based on current tracked order (already transposed)
                const baseXLabels = yLabels;
                const baseYLabels = xLabels;
                
                const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                
                // Reorder transposed data
                data = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => data[rowIdx][colIdx])
                );
                dataOriginal = rowMapping.map(rowIdx => 
                    colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                );
                
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                    );
                }}
            }} else {{
                // Normal (non-transposed) mode
                displayXLabels = currentXLabels.slice();
                displayYLabels = currentYLabels.slice();
                
                // Apply reordering if different from base labels
                const baseXLabels = xLabels;
                const baseYLabels = yLabels;
                
                const needsRowReorder = !arraysEqual(displayYLabels, baseYLabels);
                const needsColReorder = !arraysEqual(displayXLabels, baseXLabels);
                
                if (needsRowReorder || needsColReorder) {{
                    const rowMapping = displayYLabels.map(label => baseYLabels.indexOf(label));
                    const colMapping = displayXLabels.map(label => baseXLabels.indexOf(label));
                    
                    // Reorder data matrix
                    data = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => data[rowIdx][colIdx])
                    );
                    dataOriginal = rowMapping.map(rowIdx => 
                        colMapping.map(colIdx => dataOriginal[rowIdx][colIdx])
                    );
                    
                    // Reorder hover text if available
                    if (currentHoverText !== null) {{
                        currentHoverText = rowMapping.map(rowIdx => 
                            colMapping.map(colIdx => currentHoverText[rowIdx][colIdx])
                        );
                    }}
                }}
            }}
            
            // Apply clustering reordering if enabled (after transpose and custom reordering)
            if (useClusteredOrder && clusteringAvailable) {{
                // Get clustering results for the selected method
                const selectedMethod = clusteringResults[currentClusteringMethod];
                let methodRowOrder = rowOrderClustered;
                let methodColOrder = colOrderClustered;
                
                if (selectedMethod) {{
                    methodRowOrder = selectedMethod.row_order;
                    methodColOrder = selectedMethod.col_order;
                }} else {{
                    console.warn('Clustering method not found:', currentClusteringMethod, '- using default');
                }}
                
                // When transposed, swap the cluster orders to match the transposed dimensions
                const effectiveRowOrder = isTransposed ? methodColOrder : methodRowOrder;
                const effectiveColOrder = isTransposed ? methodRowOrder : methodColOrder;
                
                data = reorderData(data, effectiveRowOrder, effectiveColOrder);
                dataOriginal = reorderData(dataOriginal, effectiveRowOrder, effectiveColOrder);
                displayXLabels = reorderLabels(displayXLabels, effectiveColOrder);
                displayYLabels = reorderLabels(displayYLabels, effectiveRowOrder);
                // Reorder hover text if available
                if (currentHoverText !== null) {{
                    currentHoverText = reorderHoverText(currentHoverText, effectiveRowOrder, effectiveColOrder);
                }}
                // Reorder zero mask if active (create a copy to avoid mutating the original)
                if (zeroMaskMatrix !== null) {{
                    const maskCopy = zeroMaskMatrix.map(row => row.slice());
                    zeroMaskMatrix = reorderData(maskCopy, effectiveRowOrder, effectiveColOrder);
                }}
            }}
            
            // Apply data filter based on filter mode
            // Create a local copy of zeroMaskMatrix to avoid mutating the original
            let localZeroMask = zeroMaskMatrix ? zeroMaskMatrix.map(row => row.slice()) : null;
            
            if (dataFilterActive) {{
                if (dataFilterMode === 'hide' && filteredRowIndices.length > 0 && filteredColIndices.length > 0) {{
                    // HIDE MODE: Filter out rows/columns entirely
                    data = filteredRowIndices.map(rowIdx => 
                        filteredColIndices.map(colIdx => data[rowIdx][colIdx])
                    );
                    dataOriginal = filteredRowIndices.map(rowIdx => 
                        filteredColIndices.map(colIdx => dataOriginal[rowIdx][colIdx])
                    );
                    
                    // Filter labels
                    displayXLabels = filteredColIndices.map(idx => displayXLabels[idx]);
                    displayYLabels = filteredRowIndices.map(idx => displayYLabels[idx]);
                    
                    // Filter hover text if available
                    if (currentHoverText !== null) {{
                        currentHoverText = filteredRowIndices.map(rowIdx => 
                            filteredColIndices.map(colIdx => currentHoverText[rowIdx][colIdx])
                        );
                    }}
                    
                    console.log(`Data filter (hide): showing ${{filteredRowIndices.length}} rows × ${{filteredColIndices.length}} cols`);
                }} else if (dataFilterMode === 'zero' && localZeroMask !== null) {{
                    // ZERO MODE: Show filtered values as 0
                    const nRows = data.length;
                    const nCols = data[0].length;
                    
                    // Apply zero mask to data (using local copy to preserve original)
                    for (let i = 0; i < nRows; i++) {{
                        for (let j = 0; j < nCols; j++) {{
                            if (localZeroMask[i][j]) {{
                                data[i][j] = 0;
                                dataOriginal[i][j] = 0;
                            }}
                        }}
                    }}
                    
                    // Optionally hide rows/cols where ALL values are now zero
                    if (!showFilteredRowsCols) {{
                        // Find rows where ALL values are masked (fully filtered rows)
                        const rowsToShow = [];
                        const colsToShow = [];
                        
                        // Check which rows have at least one unmasked value
                        for (let i = 0; i < nRows; i++) {{
                            let hasUnmasked = false;
                            for (let j = 0; j < nCols; j++) {{
                                if (!localZeroMask[i][j]) {{
                                    hasUnmasked = true;
                                    break;
                                }}
                            }}
                            if (hasUnmasked) rowsToShow.push(i);
                        }}
                        
                        // Check which cols have at least one unmasked value
                        for (let j = 0; j < nCols; j++) {{
                            let hasUnmasked = false;
                            for (let i = 0; i < nRows; i++) {{
                                if (!localZeroMask[i][j]) {{
                                    hasUnmasked = true;
                                    break;
                                }}
                            }}
                            if (hasUnmasked) colsToShow.push(j);
                        }}
                        
                        // Apply row/col filtering if some are fully masked
                        if (rowsToShow.length < nRows || colsToShow.length < nCols) {{
                            data = rowsToShow.map(rowIdx => 
                                colsToShow.map(colIdx => data[rowIdx][colIdx])
                            );
                            dataOriginal = rowsToShow.map(rowIdx => 
                                colsToShow.map(colIdx => dataOriginal[rowIdx][colIdx])
                            );
                            displayXLabels = colsToShow.map(idx => displayXLabels[idx]);
                            displayYLabels = rowsToShow.map(idx => displayYLabels[idx]);
                            if (currentHoverText !== null) {{
                                currentHoverText = rowsToShow.map(rowIdx => 
                                    colsToShow.map(colIdx => currentHoverText[rowIdx][colIdx])
                                );
                            }}
                            console.log(`Data filter (zero, hidden): ${{rowsToShow.length}}/${{nRows}} rows × ${{colsToShow.length}}/${{nCols}} cols`);
                        }}
                    }}
                    
                    console.log(`Data filter (zero): masked cells showing as 0`);
                }}
            }}
            
            const range = getDataRange(data);
            
            // Determine which colorscale to use
            let colorscaleToUse;
            
            // Check if we should use custom colorscale
            if (currentColorscale === 'Custom' && customColorScale && Array.isArray(customColorScale) && customColorScale.length > 0) {{
                // Use the custom colorscale array directly
                colorscaleToUse = customColorScale;
                console.log('✓ createHeatmap: Using CUSTOM colorscale:', {{
                    scale: customColorScale,
                    length: customColorScale.length,
                    positions: customColorScale.map(c => c[0])
                }});
            }} else {{
                // For preset colorscales, convert to array format for Plotly compatibility
                // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them explicitly
                colorscaleToUse = getPlotlyColorscaleArray(currentColorscale);
                console.log('createHeatmap: Using preset colorscale:', {{
                    name: currentColorscale,
                    isArray: Array.isArray(colorscaleToUse),
                    length: Array.isArray(colorscaleToUse) ? colorscaleToUse.length : 'N/A'
                }});
            }}
            
            // Get metric display name for colorbar
            const metricDisplayNames = {{
                'weight': 'Synapses',
                'ratio': 'Ratio',
                'probability': 'Probability'
            }};
            const metricNameOverride = '{js_escape(metric_name or '')}';
            if (metricNameOverride) {{
                metricDisplayNames[metricType] = metricNameOverride;
            }}
            const metricDisplayName = metricDisplayNames[currentMetric] || currentMetric;
            
            const trace = {{
                z: data,
                x: displayXLabels.map((_, i) => i),  // Use indices for positioning
                y: displayYLabels.map((_, i) => i),  // Use indices for positioning
                type: 'heatmap',
                colorscale: colorscaleToUse,
                colorbar: {{
                    title: metricDisplayName + scaleLabel,
                    titleside: 'right'
                }}
            }};
            
            // Configure text display for cell values
            console.log('createHeatmap: showCellValues =', showCellValues);
            if (showCellValues) {{
                // Show cell values: use texttemplate to display z values
                console.log('Setting texttemplate to show cell values');
                
                // Create a text array from the data for display
                const textArray = data.map(row => row.map(val => val.toString()));
                
                trace.text = textArray;  // Text array for display
                trace.texttemplate = '%{{text}}';  // Use the text array
                trace.textfont = {{
                    size: Math.max(8, Math.min(16, currentFontSize * 0.8))
                }};
                // For hover, use the detailed hover text
                trace.hovertext = currentHoverText;
                trace.hoverinfo = 'text';
                trace.hovertemplate = '%{{hovertext}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }} else {{
                // Hide cell values: no texttemplate, only hover text
                console.log('NOT setting texttemplate - hiding cell values');
                trace.text = currentHoverText;  // Text for hover only
                trace.hoverinfo = 'text';  // Show hover text on hover
                trace.hovertemplate = '%{{text}}<extra></extra>';  // <extra></extra> hides "trace 0"
            }}
            console.log('trace texttemplate:', trace.texttemplate);
            console.log('trace text sample:', trace.text ? trace.text[0] : 'none');
            
            // Apply custom colorbar range
            // Priority: 1) Custom color range (for cross-heatmap comparison)
            //           2) Manual slider range (if not auto)
            //           3) Auto range (default)
            if (window.customColorRange) {{
                trace.zmin = window.customColorRange.min;
                trace.zmax = window.customColorRange.max;
                console.log('Using custom color range:', window.customColorRange);
            }} else if (!useAutoRange && customZmin !== null && customZmax !== null) {{
                trace.zmin = customZmin;
                trace.zmax = customZmax;
            }}
            
            // Store current range for slider scaling
            window.currentDataRange = range;
            
            // Update 2-point color value inputs to show current data range in auto mode
            if (!window.customColorRange) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Determine axis titles based on transpose state
            const xAxisLabel = isTransposed ? 'Source' : 'Target';
            const yAxisLabel = isTransposed ? 'Target' : 'Source';
            const xAxisCount = displayXLabels.length;
            const yAxisCount = displayYLabels.length;
            
            const layout = {{
                title: originalTitle,
                font: {{size: currentFontSize}},
                autosize: false,
                xaxis: {{
                    title: isLarge ? '<b>' + xAxisLabel + '</b> (' + xAxisCount + ' neurons)' : '<b>' + xAxisLabel + '</b>',
                    side: 'bottom',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    tickangle: displayXLabels.length > 1 ? -45 : 0,  // Always rotate when multiple labels
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayXLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayXLabels  // Use labels as tick text
                }},
                yaxis: {{
                    title: isLarge ? '<b>' + yAxisLabel + '</b> (' + yAxisCount + ' neurons)' : '<b>' + yAxisLabel + '</b>',
                    side: 'left',
                    titlefont: {{size: currentFontSize + 2, color: '#333333'}},
                    autorange: 'reversed',
                    showticklabels: showLabels,
                    tickmode: 'array',  // Use explicit tick values
                    tickvals: displayYLabels.map((_, i) => i),  // Use indices as tick positions
                    ticktext: displayYLabels  // Use labels as tick text
                }},
                hoverlabel: {{
                    bgcolor: 'white',
                    font_size: 12,
                    font_family: 'Arial'
                }},
                width: currentWidth,
                height: currentHeight,
                margin: {{l: dynamicLeftMargin, r: 40, b: dynamicBottomMargin, t: 100, pad: 4}}
            }};
            
            const config = {{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {{
                    format: 'png',
                    filename: 'heatmap_' + currentScale,
                    height: currentHeight,
                    width: currentWidth,
                    scale: exportScale
                }}
            }};
            
            // Add cell value annotations if enabled
            if (showCellValues) {{
                const annotations = [];
                
                // Get the actual zmin/zmax for color mapping
                const actualZmin = trace.zmin !== undefined ? trace.zmin : range.min;
                const actualZmax = trace.zmax !== undefined ? trace.zmax : range.max;
                
                for (let i = 0; i < data.length; i++) {{
                    for (let j = 0; j < data[i].length; j++) {{
                        const scaledValue = data[i][j];  // Scaled value for color
                        const originalValue = dataOriginal[i][j];  // Original value for display
                        
                        // Skip this value if it matches ignore criteria (exact value or expression)
                        if (shouldIgnoreValue(originalValue)) {{
                            continue;
                        }}
                        
                        // Calculate the background color for this cell using scaled value
                        const normalized = (scaledValue - actualZmin) / (actualZmax - actualZmin);
                        
                        // Get color from the colorscale
                        let bgColor = 'rgb(128, 128, 128)';  // default gray
                        if (Array.isArray(colorscaleToUse)) {{
                            // For custom colorscales - interpolate between color stops
                            bgColor = interpolateColorscale(colorscaleToUse, normalized);
                        }} else {{
                            // For named colorscales, get color from Plotly's colorscale
                            bgColor = getColorFromPlotlyScale(colorscaleToUse, normalized);
                        }}
                        
                        // Convert color to RGB and determine contrast color
                        const rgb = hexToRgb(bgColor);
                        const textColor = getContrastColor(rgb);
                        
                        // Debug logging for first few cells
                        if (i === 0 && j < 3) {{
                            const luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                            console.log(`Cell [${{i}},${{j}}] original=${{originalValue}}, scaled=${{scaledValue}}, normalized=${{normalized.toFixed(3)}}, bgColor=${{bgColor}}, rgb=[${{rgb}}], luminance=${{luminance.toFixed(1)}}, threshold=${{(contrastThreshold * 255).toFixed(1)}}, textColor=${{textColor}}`);
                        }}
                        
                        annotations.push({{
                            x: j,  // Use index for positioning
                            y: i,  // Use index for positioning
                            text: String(originalValue),  // Display original value
                            showarrow: false,
                            font: {{
                                size: cellValueFontSize,
                                color: textColor
                            }},
                            xref: 'x',
                            yref: 'y'
                        }});
                    }}
                }}
                layout.annotations = annotations;
                console.log('Added', annotations.length, 'annotations for cell values with adaptive colors');
            }}
            
            Plotly.newPlot('heatmap', [trace], layout, config);
        }}
        
        function toggleClustering(mode) {{
            // Toggle between original and clustered ordering
            useClusteredOrder = (mode === 'clustered');
            
            // Update button states
            document.getElementById('btn-original').classList.toggle('active', mode === 'original');
            document.getElementById('btn-clustered').classList.toggle('active', mode === 'clustered');
            
            // Show/hide clustering method selector
            const methodSection = document.getElementById('clusteringMethodSection');
            if (methodSection) {{
                methodSection.style.display = (mode === 'clustered' && clusteringAvailable) ? 'block' : 'none';
            }}
            
            // If clustering is not available, show message and revert
            if (mode === 'clustered' && !clusteringAvailable) {{
                alert('Clustering is not available for this matrix. Using original order.');
                useClusteredOrder = false;
                document.getElementById('btn-original').classList.add('active');
                document.getElementById('btn-clustered').classList.remove('active');
                return;
            }}
            
            // Update data filter state (disables when clustering is active)
            applyDataFilter();
            
            // Recreate heatmap with new ordering
            createHeatmap();
        }}
        
        function updateClusteringMethod() {{
            // Get selected clustering method
            const methodSelect = document.getElementById('clusteringMethodSelect');
            currentClusteringMethod = methodSelect.value;
            
            console.log('Switching to clustering method:', currentClusteringMethod);
            
            // Update the heatmap with new clustering method
            if (useClusteredOrder) {{
                createHeatmap();
            }}
        }}
        
        function setScale(scale) {{
            currentScale = scale;
            
            // Update button states
            document.querySelectorAll('[id^="btn-"]').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById('btn-' + scale).classList.add('active');
            
            createHeatmap();
        }}
        
        function updateMetric() {{
            // Switch to the selected metric
            currentMetric = document.getElementById('metricSelect').value;
            console.log('Switching to metric:', currentMetric);
            
            // Update dataLinear with the new metric's data (always use original ordering)
            dataLinear = metricsData[currentMetric];
            
            // Clear cached transforms so they're recomputed for new metric
            cachedDataLog2 = null;
            cachedDataLog10 = null;
            cachedDataSqrt = null;
            
            // Recreate the heatmap with new metric data (clustering will be applied in createHeatmap)
            createHeatmap();
        }}
        
        function updateColorscale() {{
            currentColorscale = document.getElementById('colorscaleSelect').value;
            
            // If switching to Custom and no custom scale exists, create default
            if (currentColorscale === 'Custom' && !customColorScale) {{
                applyCustomColors();
            }}
            
            createHeatmap();
        }}
        
        function toggleCustomColorPanel() {{
            const panel = document.getElementById('customColorPanel');
            if (panel.style.display === 'none') {{
                panel.style.display = 'block';
            }} else {{
                panel.style.display = 'none';
            }}
        }}
        
        function toggle3PointScale() {{
            use3PointScale = document.getElementById('use3PointScale').checked;
            const twoPoint = document.getElementById('twoPointColors');
            const threePoint = document.getElementById('threePointColors');
            
            if (use3PointScale) {{
                twoPoint.style.display = 'none';
                threePoint.style.display = 'block';
                
                // Set default values based on current data range
                if (window.currentDataRange) {{
                    const range = window.currentDataRange;
                    const mid = (range.min + range.max) / 2;
                    document.getElementById('valueMin3').value = formatValueDisplay(range.min);
                    document.getElementById('valueMid3').value = formatValueDisplay(mid);
                    document.getElementById('valueMax3').value = formatValueDisplay(range.max);
                }}
            }} else {{
                twoPoint.style.display = 'block';
                threePoint.style.display = 'none';
            }}
        }}
        
        function rgbToPlotlyFormat(hex) {{
            // Convert hex color to RGB format for Plotly
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        // Helper function to compare two arrays for equality
        function arraysEqual(arr1, arr2) {{
            if (arr1.length !== arr2.length) return false;
            for (let i = 0; i < arr1.length; i++) {{
                if (arr1[i] !== arr2[i]) return false;
            }}
            return true;
        }}
        
        function formatValueDisplay(value) {{
            // Format number to remove trailing zeros and unnecessary decimal point
            // Examples: 0.000000 -> "0", 250.123456 -> "250.123456", 1.500000 -> "1.5"
            if (value === 0) return "0";
            const str = value.toFixed(6);
            // Remove trailing zeros and decimal point if not needed
            return str.replace(/\.?0+$/, '');
        }}
        
        function applyCustomColors() {{
            if (use3PointScale) {{
                // 3-point scale with custom value mapping
                const colorMin = document.getElementById('colorMin3').value;
                const colorMid = document.getElementById('colorMid3').value;
                const colorMax = document.getElementById('colorMax3').value;
                
                const valueMin = parseFloat(document.getElementById('valueMin3').value);
                const valueMid = parseFloat(document.getElementById('valueMid3').value);
                const valueMax = parseFloat(document.getElementById('valueMax3').value);
                
                // Get current data range
                const range = window.currentDataRange;
                if (!range) {{
                    alert('Please wait for data to load before applying custom colors.');
                    return;
                }}
                
                // Map custom values to [0, 1] range - allows values beyond actual data range
                const normalizeValue = (val, rangeMin, rangeMax) => {{
                    if (rangeMax === rangeMin) return 0.5;
                    return (val - rangeMin) / (rangeMax - rangeMin);
                }};
                
                // Use custom value range for normalization (allows cross-heatmap comparison)
                const customRangeMin = valueMin;
                const customRangeMax = valueMax;
                
                if (customRangeMax === customRangeMin) {{
                    alert('Custom min and max values cannot be the same.');
                    return;
                }}
                
                // Map custom value points to [0, 1] colorscale positions
                // This defines where each color appears on the scale
                const posMid = normalizeValue(valueMid, customRangeMin, customRangeMax);
                
                // Clamp mid position to [0, 1]
                const clampedPosMid = Math.max(0, Math.min(1, posMid));
                
                // Create color scale array spanning 0 to 1
                // Plotly will map data values to this scale based on customColorRange
                customColorScale = [
                    [0, rgbToPlotlyFormat(colorMin)],
                    [clampedPosMid, rgbToPlotlyFormat(colorMid)],
                    [1, rgbToPlotlyFormat(colorMax)]
                ];
                
                // Set custom range for Plotly to use
                window.customColorRange = {{min: valueMin, max: valueMax}};
                
                // Sort by position (required by Plotly)
                customColorScale.sort((a, b) => a[0] - b[0]);
                
                // Ensure positions are distinct (avoid duplicates)
                const epsilon = 0.001;
                for (let i = 1; i < customColorScale.length; i++) {{
                    if (Math.abs(customColorScale[i][0] - customColorScale[i-1][0]) < epsilon) {{
                        customColorScale[i][0] = customColorScale[i-1][0] + epsilon;
                    }}
                }}
                
                console.log('Applied 3-point scale:', {{
                    inputs: {{
                        min: {{value: valueMin, color: colorMin}},
                        mid: {{value: valueMid, color: colorMid}},
                        max: {{value: valueMax, color: colorMax}}
                    }},
                    customRange: {{min: valueMin, max: valueMax}},
                    midPosition: clampedPosMid,
                    colorScale: customColorScale
                }});
            }} else {{
                // 2-point scale with optional custom value mapping
                const colorMin = document.getElementById('colorMin').value;
                const colorMax = document.getElementById('colorMax').value;
                
                const valueMin2Input = document.getElementById('valueMin2').value;
                const valueMax2Input = document.getElementById('valueMax2').value;
                
                // Check if custom values are specified
                if (valueMin2Input !== '' && valueMax2Input !== '') {{
                    // Use custom values for cross-heatmap comparison
                    const valueMin = parseFloat(valueMin2Input);
                    const valueMax = parseFloat(valueMax2Input);
                    
                    if (valueMax === valueMin) {{
                        alert('Custom min and max values cannot be the same.');
                        return;
                    }}
                    
                    // Colorscale spans from 0 to 1 (representing valueMin to valueMax)
                    // Plotly will map data values to this scale automatically
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Override the data normalization by setting colorscale range
                    window.customColorRange = {{min: valueMin, max: valueMax}};
                    
                    console.log('Applied 2-point scale with custom values:', {{
                        customRange: {{min: valueMin, max: valueMax}},
                        colorScale: customColorScale
                    }});
                }} else {{
                    // Auto mode: use full data range
                    customColorScale = [
                        [0, rgbToPlotlyFormat(colorMin)],
                        [1, rgbToPlotlyFormat(colorMax)]
                    ];
                    
                    // Clear custom range
                    window.customColorRange = null;
                    
                    console.log('Applied 2-point scale (auto):', customColorScale);
                }}
            }}
            
            // Switch to Custom colorscale and update
            currentColorscale = 'Custom';
            
            // Update dropdown without triggering the onchange handler
            const selectElement = document.getElementById('colorscaleSelect');
            const oldOnchange = selectElement.onchange;
            selectElement.onchange = null;
            selectElement.value = 'Custom';
            selectElement.onchange = oldOnchange;
            
            console.log('About to create heatmap with custom scale:', {{
                currentColorscale: currentColorscale,
                customColorScale: customColorScale,
                dropdownValue: selectElement.value
            }});
            
            createHeatmap();
        }}
        
        function resetToAutoColors() {{
            // Clear custom color range
            window.customColorRange = null;
            
            // Update value input boxes to show current data range
            const range = window.currentDataRange;
            if (range) {{
                document.getElementById('valueMin2').value = formatValueDisplay(range.min);
                document.getElementById('valueMax2').value = formatValueDisplay(range.max);
            }}
            
            // Recreate heatmap with auto colors
            createHeatmap();
            
            console.log('Reset to auto color mode');
        }}
        
        function updateFontSize(size) {{
            currentFontSize = parseInt(size);
            document.getElementById('fontSizeValue').textContent = size + 'px';
            createHeatmap();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            const btn = document.getElementById('toggleLabelsBtn');
            btn.textContent = showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels';
            
            // Update the layout to hide/show ALL text elements including colorbar
            const gd = document.getElementById('heatmap');
            
            // Update colorbar text (trace-level property)
            const traceUpdate = {{
                'colorbar.title.text': showLabels ? (metricType.charAt(0).toUpperCase() + metricType.slice(1)) : '',
                'colorbar.showticklabels': showLabels
            }};
            
            // Update layout elements
            const layoutUpdate = {{
                'title.text': showLabels ? originalTitle : '',
                'xaxis.showticklabels': showLabels,
                'yaxis.showticklabels': showLabels,
                'xaxis.title.text': showLabels ? (isLarge ? '<b>Target</b> (' + gd.data[0].x.length + ' neurons)' : '<b>Target</b>') : '',
                'yaxis.title.text': showLabels ? (isLarge ? '<b>Source</b> (' + gd.data[0].y.length + ' neurons)' : '<b>Source</b>') : '',
                'xaxis.ticks': showLabels ? 'outside' : '',
                'yaxis.ticks': showLabels ? 'outside' : '',
                // Prevent autosize from expanding the plot
                'autosize': false,
                // Keep margins fixed to prevent rescaling
                'margin.l': dynamicLeftMargin,
                'margin.r': 40,
                'margin.t': 100,
                'margin.b': 120,
                // Preserve dimensions explicitly
                'width': currentWidth,
                'height': currentHeight
            }};
            
            // Update both trace and layout
            Plotly.restyle(gd, traceUpdate, 0);
            Plotly.relayout(gd, layoutUpdate);
        }}
        
        function toggleCellValues() {{
            showCellValues = !showCellValues;
            const btn = document.getElementById('toggleCellValuesBtn');
            btn.textContent = showCellValues ? '🔢 Hide Values' : '🔢 Show Values';
            
            console.log('toggleCellValues called, showCellValues is now:', showCellValues);
            
            // Recreate heatmap to add/remove cell value annotations
            createHeatmap();
        }}
        
        function updateCellValueSize(size) {{
            cellValueFontSize = parseInt(size);
            document.getElementById('cellValueSizeValue').textContent = cellValueFontSize + 'px';
            
            // Only recreate if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateContrastThreshold(value) {{
            contrastThreshold = parseFloat(value);
            document.getElementById('contrastThresholdValue').textContent = contrastThreshold.toFixed(4);
            console.log('Contrast threshold updated to:', contrastThreshold);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function reverseContrastColors() {{
            reverseContrast = !reverseContrast;
            console.log('Contrast colors reversed:', reverseContrast);
            
            // Recreate heatmap if cell values are currently shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function updateIgnoredValues() {{
            const input = document.getElementById('ignoreValuesInput');
            const expressions = input.value.split(',').map(v => v.trim()).filter(v => v !== '');
            
            // Clear and repopulate the ignored values array
            // Store both exact values and comparison expressions
            ignoredValues.clear();
            ignoredValues.expressions = [];  // Array to store comparison expressions
            
            expressions.forEach(expr => {{
                // Check if it's a comparison expression (>, <, >=, <=)
                const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
                if (compMatch) {{
                    // It's a comparison expression
                    const operator = compMatch[1];
                    const threshold = parseFloat(compMatch[2]);
                    ignoredValues.expressions.push({{ operator, threshold }});
                }} else {{
                    // Try to parse as exact number
                    const num = parseFloat(expr);
                    if (!isNaN(num)) {{
                        ignoredValues.add(num);
                    }}
                }}
            }});
            
            console.log('Ignored exact values:', Array.from(ignoredValues));
            console.log('Ignored expressions:', ignoredValues.expressions);
            
            // Recreate heatmap if cell values are shown
            if (showCellValues) {{
                createHeatmap();
            }}
        }}
        
        function shouldIgnoreValue(value) {{
            // Check if value matches any exact value
            if (ignoredValues.has(value)) {{
                return true;
            }}
            
            // Check if value matches any comparison expression
            if (ignoredValues.expressions && ignoredValues.expressions.length > 0) {{
                for (const expr of ignoredValues.expressions) {{
                    let matches = false;
                    switch (expr.operator) {{
                        case '>':
                            matches = value > expr.threshold;
                            break;
                        case '<':
                            matches = value < expr.threshold;
                            break;
                        case '>=':
                            matches = value >= expr.threshold;
                            break;
                        case '<=':
                            matches = value <= expr.threshold;
                            break;
                        case '==':
                            matches = value === expr.threshold;
                            break;
                        case '!=':
                            matches = value !== expr.threshold;
                            break;
                    }}
                    if (matches) {{
                        return true;
                    }}
                }}
            }}
            
            return false;
        }}
        
        // ===== DATA FILTER FUNCTIONS =====
        // Filter entire rows/columns based on their maximum values
        
        // ===== FILTER EXPRESSION PARSING =====
        // Supports:
        //   - OR logic: <5, >100  (comma-separated, any condition matches)
        //   - AND logic: (>=5, <=10)  (parentheses, all conditions must match)
        //   - Mixed: <5, (>=10, <=20), >100  (filter if <5 OR (>=10 AND <=20) OR >100)
        
        function parseSingleExpression(expr) {{
            // Parse a single comparison expression like ">5" or "<=10"
            const compMatch = expr.match(/^([><]=?|==|!=)\\s*(-?\\d+\\.?\\d*)$/);
            if (compMatch) {{
                return {{ operator: compMatch[1], threshold: parseFloat(compMatch[2]) }};
            }}
            // Try to parse as exact number
            const num = parseFloat(expr);
            if (!isNaN(num)) {{
                return {{ operator: '==', threshold: num }};
            }}
            return null;
        }}
        
        function parseFilterExpressions(inputString) {{
            // Returns an array of filter groups
            // Each group is either:
            //   - A single expression (OR with other groups)
            //   - An array of expressions (AND within group, OR with other groups)
            
            const result = [];
            let remaining = inputString.trim();
            
            while (remaining.length > 0) {{
                // Skip leading commas and whitespace
                remaining = remaining.replace(/^[,\\s]+/, '');
                if (remaining.length === 0) break;
                
                if (remaining.startsWith('(')) {{
                    // AND group: find matching closing parenthesis
                    const closeIdx = remaining.indexOf(')');
                    if (closeIdx === -1) {{
                        // No closing paren, treat rest as single group
                        const inner = remaining.substring(1);
                        const andExprs = inner.split(',').map(e => e.trim()).filter(e => e !== '');
                        const parsed = andExprs.map(e => parseSingleExpression(e)).filter(e => e !== null);
                        if (parsed.length > 0) {{
                            result.push({{ type: 'AND', expressions: parsed }});
                        }}
                        break;
                    }} else {{
                        const inner = remaining.substring(1, closeIdx);
                        const andExprs = inner.split(',').map(e => e.trim()).filter(e => e !== '');
                        const parsed = andExprs.map(e => parseSingleExpression(e)).filter(e => e !== null);
                        if (parsed.length > 0) {{
                            result.push({{ type: 'AND', expressions: parsed }});
                        }}
                        remaining = remaining.substring(closeIdx + 1);
                    }}
                }} else {{
                    // Single expression (OR): read until comma or opening paren
                    const nextComma = remaining.indexOf(',');
                    const nextParen = remaining.indexOf('(');
                    let endIdx = remaining.length;
                    
                    if (nextComma !== -1 && (nextParen === -1 || nextComma < nextParen)) {{
                        endIdx = nextComma;
                    }} else if (nextParen !== -1 && (nextComma === -1 || nextParen < nextComma)) {{
                        endIdx = nextParen;
                    }}
                    
                    const expr = remaining.substring(0, endIdx).trim();
                    if (expr.length > 0) {{
                        const parsed = parseSingleExpression(expr);
                        if (parsed !== null) {{
                            result.push({{ type: 'OR', expression: parsed }});
                        }}
                    }}
                    remaining = remaining.substring(endIdx);
                }}
            }}
            
            return result;
        }}
        
        function evaluateSingleCondition(value, expr) {{
            // Evaluate a single condition against a value
            switch (expr.operator) {{
                case '>': return value > expr.threshold;
                case '<': return value < expr.threshold;
                case '>=': return value >= expr.threshold;
                case '<=': return value <= expr.threshold;
                case '==': return value === expr.threshold;
                case '!=': return value !== expr.threshold;
                default: return false;
            }}
        }}
        
        function shouldFilterValue(value, filterGroups) {{
            // Check if a value should be filtered based on parsed filter groups
            // Returns true if value matches the filter (should be filtered/hidden/zeroed)
            if (filterGroups.length === 0) return false;
            
            // OR logic between groups: return true if ANY group matches
            for (const group of filterGroups) {{
                if (group.type === 'AND') {{
                    // AND logic within group: ALL expressions must match
                    let allMatch = true;
                    for (const expr of group.expressions) {{
                        if (!evaluateSingleCondition(value, expr)) {{
                            allMatch = false;
                            break;
                        }}
                    }}
                    if (allMatch) return true;  // This AND group matched
                }} else {{
                    // Single OR expression
                    if (evaluateSingleCondition(value, group.expression)) {{
                        return true;
                    }}
                }}
            }}
            
            return false;  // No group matched
        }}
        
        function shouldHideRowOrColumn(maxValue, filterGroups) {{
            // Legacy wrapper for row/column filtering (hide mode)
            return shouldFilterValue(maxValue, filterGroups);
        }}
        
        function applyDataFilter() {{
            const input = document.getElementById('dataFilterInput');
            const statusDiv = document.getElementById('filterStatus');
            const filterValue = input.value.trim();
            
            // Disable data filtering when clustering is active
            if (useClusteredOrder && clusteringAvailable) {{
                statusDiv.textContent = '⚠️ Data filter disabled during clustering';
                statusDiv.style.color = '#ff9800';
                input.disabled = true;
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                zeroMaskMatrix = null;
                return;
            }} else {{
                input.disabled = false;
            }}
            
            if (!filterValue) {{
                // No filter - show all rows/columns
                dataFilterActive = false;
                dataFilterExpressions = [];
                filteredRowIndices = [];
                filteredColIndices = [];
                zeroMaskMatrix = null;
                statusDiv.textContent = '';
                createHeatmap();
                return;
            }}
            
            // Parse filter expressions
            dataFilterExpressions = parseFilterExpressions(filterValue);
            
            if (dataFilterExpressions.length === 0) {{
                statusDiv.textContent = '⚠️ Invalid filter format';
                statusDiv.style.color = '#d32f2f';
                return;
            }}
            
            // Use original unscaled data for filtering
            const filterData = metricsData[currentMetric];
            
            const nRows = filterData.length;
            const nCols = filterData[0].length;
            
            // Calculate max value for each row and column
            const rowMaxValues = new Array(nRows).fill(-Infinity);
            const colMaxValues = new Array(nCols).fill(-Infinity);
            
            for (let i = 0; i < nRows; i++) {{
                for (let j = 0; j < nCols; j++) {{
                    const value = filterData[i][j];
                    if (value > rowMaxValues[i]) rowMaxValues[i] = value;
                    if (value > colMaxValues[j]) colMaxValues[j] = value;
                }}
            }}
            
            // Determine which rows and columns to keep (for 'hide' mode)
            // Hide mode: filter based on row/column MAX values
            filteredRowIndices = [];
            filteredColIndices = [];
            
            for (let i = 0; i < nRows; i++) {{
                if (!shouldHideRowOrColumn(rowMaxValues[i], dataFilterExpressions)) {{
                    filteredRowIndices.push(i);
                }}
            }}
            
            for (let j = 0; j < nCols; j++) {{
                if (!shouldHideRowOrColumn(colMaxValues[j], dataFilterExpressions)) {{
                    filteredColIndices.push(j);
                }}
            }}
            
            // Create zero mask matrix (for 'zero' mode)
            // Zero mode: filter based on INDIVIDUAL CELL values
            zeroMaskMatrix = new Array(nRows);
            let maskedCellCount = 0;
            for (let i = 0; i < nRows; i++) {{
                zeroMaskMatrix[i] = new Array(nCols);
                for (let j = 0; j < nCols; j++) {{
                    // Apply filter to individual cell value
                    const cellValue = filterData[i][j];
                    zeroMaskMatrix[i][j] = shouldFilterValue(cellValue, dataFilterExpressions);
                    if (zeroMaskMatrix[i][j]) maskedCellCount++;
                }}
            }}
            
            dataFilterActive = true;
            
            const hiddenRows = nRows - filteredRowIndices.length;
            const hiddenCols = nCols - filteredColIndices.length;
            
            // Update status based on mode
            if (dataFilterMode === 'hide') {{
                if (filteredRowIndices.length === 0 || filteredColIndices.length === 0) {{
                    statusDiv.textContent = '⚠️ Filter hides all data!';
                    statusDiv.style.color = '#d32f2f';
                    dataFilterActive = false;
                    return;
                }}
                statusDiv.textContent = `✓ Showing ${{filteredRowIndices.length}}/${{nRows}} rows, ${{filteredColIndices.length}}/${{nCols}} cols`;
                statusDiv.style.color = '#2e7d32';
                console.log(`Data filter (hide): hiding ${{hiddenRows}} rows and ${{hiddenCols}} cols`);
            }} else {{
                // 'zero' mode
                const totalCells = nRows * nCols;
                const pctMasked = ((maskedCellCount / totalCells) * 100).toFixed(1);
                statusDiv.textContent = `✓ ${{maskedCellCount}}/${{totalCells}} cells (${{pctMasked}}%) shown as 0`;
                statusDiv.style.color = '#2e7d32';
                console.log(`Data filter (zero): masking ${{maskedCellCount}}/${{totalCells}} cells as 0`);
            }}
            
            createHeatmap();
        }}
        
        function setFilterMode(mode) {{
            dataFilterMode = mode;
            
            // Update button states
            document.getElementById('btn-filter-hide').classList.toggle('active', mode === 'hide');
            document.getElementById('btn-filter-zero').classList.toggle('active', mode === 'zero');
            
            // Update checkbox visibility (only relevant for 'zero' mode)
            const checkbox = document.getElementById('showFilteredRowsCols');
            checkbox.disabled = mode !== 'zero';
            
            // Re-apply filter with new mode
            if (dataFilterActive) {{
                applyDataFilter();
            }}
        }}
        
        function toggleFilteredVisibility() {{
            showFilteredRowsCols = document.getElementById('showFilteredRowsCols').checked;
            
            // Re-apply filter with new visibility setting
            if (dataFilterActive && dataFilterMode === 'zero') {{
                createHeatmap();
            }}
        }}
        
        function resetDataFilter() {{
            document.getElementById('dataFilterInput').value = '';
            document.getElementById('filterStatus').textContent = '';
            dataFilterActive = false;
            dataFilterExpressions = [];
            filteredRowIndices = [];
            filteredColIndices = [];
            zeroMaskMatrix = null;
            createHeatmap();
        }}
        
        // ===== END DATA FILTER FUNCTIONS =====
        
        function getPlotlyColorscaleArray(scaleName) {{
            // Return colorscale array for Plotly heatmap
            // Plotly v1.58.5 doesn't recognize all colorscale names, so we define them as arrays
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Return the colorscale array, or fallback to the name string
            return colorscales[scaleName] || scaleName;
        }}
        
        function getColorFromPlotlyScale(scaleName, normalized) {{
            // Map of Plotly colorscales to their RGB interpolations
            // These are approximations of Plotly's built-in scales
            const colorscales = {{
                'Greens': [
                    [0.0, 'rgb(247,252,245)'],
                    [0.125, 'rgb(229,245,224)'],
                    [0.25, 'rgb(199,233,192)'],
                    [0.375, 'rgb(161,217,155)'],
                    [0.5, 'rgb(116,196,118)'],
                    [0.625, 'rgb(65,171,93)'],
                    [0.75, 'rgb(35,139,69)'],
                    [0.875, 'rgb(0,109,44)'],
                    [1.0, 'rgb(0,68,27)']
                ],
                'Blues': [
                    [0.0, 'rgb(247,251,255)'],
                    [0.125, 'rgb(222,235,247)'],
                    [0.25, 'rgb(198,219,239)'],
                    [0.375, 'rgb(158,202,225)'],
                    [0.5, 'rgb(107,174,214)'],
                    [0.625, 'rgb(66,146,198)'],
                    [0.75, 'rgb(33,113,181)'],
                    [0.875, 'rgb(8,81,156)'],
                    [1.0, 'rgb(8,48,107)']
                ],
                'Reds': [
                    [0.0, 'rgb(255,245,240)'],
                    [0.125, 'rgb(254,224,210)'],
                    [0.25, 'rgb(252,187,161)'],
                    [0.375, 'rgb(252,146,114)'],
                    [0.5, 'rgb(251,106,74)'],
                    [0.625, 'rgb(239,59,44)'],
                    [0.75, 'rgb(203,24,29)'],
                    [0.875, 'rgb(165,15,21)'],
                    [1.0, 'rgb(103,0,13)']
                ],
                'Purples': [
                    [0.0, 'rgb(252,251,253)'],
                    [0.125, 'rgb(239,237,245)'],
                    [0.25, 'rgb(218,218,235)'],
                    [0.375, 'rgb(188,189,220)'],
                    [0.5, 'rgb(158,154,200)'],
                    [0.625, 'rgb(128,125,186)'],
                    [0.75, 'rgb(106,81,163)'],
                    [0.875, 'rgb(84,39,143)'],
                    [1.0, 'rgb(63,0,125)']
                ],
                'Oranges': [
                    [0.0, 'rgb(255,245,235)'],
                    [0.125, 'rgb(254,230,206)'],
                    [0.25, 'rgb(253,208,162)'],
                    [0.375, 'rgb(253,174,107)'],
                    [0.5, 'rgb(253,141,60)'],
                    [0.625, 'rgb(241,105,19)'],
                    [0.75, 'rgb(217,72,1)'],
                    [0.875, 'rgb(166,54,3)'],
                    [1.0, 'rgb(127,39,4)']
                ],
                'Viridis': [
                    [0, 'rgb(68,1,84)'],
                    [0.25, 'rgb(59,82,139)'],
                    [0.5, 'rgb(33,145,140)'],
                    [0.75, 'rgb(94,201,98)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Plasma': [
                    [0, 'rgb(13,8,135)'],
                    [0.25, 'rgb(126,3,168)'],
                    [0.5, 'rgb(204,71,120)'],
                    [0.75, 'rgb(248,149,64)'],
                    [1, 'rgb(240,249,33)']
                ],
                'Inferno': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(87,16,110)'],
                    [0.5, 'rgb(188,55,84)'],
                    [0.75, 'rgb(249,142,9)'],
                    [1, 'rgb(252,255,164)']
                ],
                'Magma': [
                    [0, 'rgb(0,0,4)'],
                    [0.25, 'rgb(81,18,124)'],
                    [0.5, 'rgb(182,54,121)'],
                    [0.75, 'rgb(251,136,97)'],
                    [1, 'rgb(252,253,191)']
                ],
                'Cividis': [
                    [0, 'rgb(0,32,76)'],
                    [0.25, 'rgb(0,79,110)'],
                    [0.5, 'rgb(53,133,136)'],
                    [0.75, 'rgb(149,189,161)'],
                    [1, 'rgb(253,231,37)']
                ],
                'Hot': [
                    [0, 'rgb(0,0,0)'],
                    [0.33, 'rgb(255,0,0)'],
                    [0.66, 'rgb(255,255,0)'],
                    [1, 'rgb(255,255,255)']
                ],
                'Jet': [
                    [0, 'rgb(0,0,143)'],
                    [0.25, 'rgb(0,159,255)'],
                    [0.5, 'rgb(0,255,0)'],
                    [0.75, 'rgb(255,159,0)'],
                    [1, 'rgb(143,0,0)']
                ],
                'RdBu': [
                    [0, 'rgb(5,10,172)'],
                    [0.35, 'rgb(106,137,247)'],
                    [0.5, 'rgb(190,190,190)'],
                    [0.65, 'rgb(220,170,132)'],
                    [1, 'rgb(178,10,28)']
                ],
                'RdYlGn': [
                    [0, 'rgb(165,0,38)'],
                    [0.25, 'rgb(253,174,97)'],
                    [0.5, 'rgb(255,255,191)'],
                    [0.75, 'rgb(166,217,106)'],
                    [1, 'rgb(0,104,55)']
                ]
            }};
            
            // Get the colorscale array
            const scale = colorscales[scaleName];
            if (!scale) {{
                // Fallback to grayscale
                const gray = Math.round(normalized * 255);
                return `rgb(${{gray}},${{gray}},${{gray}})`;
            }}
            
            // Find the two color stops to interpolate between
            let lower = scale[0];
            let upper = scale[scale.length - 1];
            
            for (let i = 0; i < scale.length - 1; i++) {{
                if (normalized >= scale[i][0] && normalized <= scale[i + 1][0]) {{
                    lower = scale[i];
                    upper = scale[i + 1];
                    break;
                }}
            }}
            
            // Interpolate between the two colors
            const t = (normalized - lower[0]) / (upper[0] - lower[0]);
            const lowerRgb = hexToRgb(lower[1]);
            const upperRgb = hexToRgb(upper[1]);
            
            const r = Math.round(lowerRgb[0] + t * (upperRgb[0] - lowerRgb[0]));
            const g = Math.round(lowerRgb[1] + t * (upperRgb[1] - lowerRgb[1]));
            const b = Math.round(lowerRgb[2] + t * (upperRgb[2] - lowerRgb[2]));
            
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        function interpolateColorscale(colorscale, normalized) {{
            // Interpolate color from a custom colorscale array
            // colorscale format: [[0, 'color1'], [0.5, 'color2'], [1, 'color3'], ...]
            
            if (!Array.isArray(colorscale) || colorscale.length === 0) {{
                return 'rgb(128, 128, 128)';  // fallback gray
            }}
            
            // Handle edge cases
            if (normalized <= 0 || normalized <= colorscale[0][0]) {{
                return Array.isArray(colorscale[0]) && colorscale[0].length > 1 ? colorscale[0][1] : 'rgb(128, 128, 128)';
            }}
            if (normalized >= 1 || normalized >= colorscale[colorscale.length - 1][0]) {{
                const last = colorscale[colorscale.length - 1];
                return Array.isArray(last) && last.length > 1 ? last[1] : 'rgb(128, 128, 128)';
            }}
            
            // Find the two color stops to interpolate between
            let lower = colorscale[0];
            let upper = colorscale[colorscale.length - 1];
            
            for (let i = 0; i < colorscale.length - 1; i++) {{
                if (normalized >= colorscale[i][0] && normalized <= colorscale[i + 1][0]) {{
                    lower = colorscale[i];
                    upper = colorscale[i + 1];
                    break;
                }}
            }}
            
            // Interpolate between the two colors
            const t = (normalized - lower[0]) / (upper[0] - lower[0]);
            const lowerRgb = hexToRgb(lower[1]);
            const upperRgb = hexToRgb(upper[1]);
            
            const r = Math.round(lowerRgb[0] + t * (upperRgb[0] - lowerRgb[0]));
            const g = Math.round(lowerRgb[1] + t * (upperRgb[1] - lowerRgb[1]));
            const b = Math.round(lowerRgb[2] + t * (upperRgb[2] - lowerRgb[2]));
            
            return `rgb(${{r}},${{g}},${{b}})`;
        }}
        
        function getContrastColor(rgb) {{
            // Calculate luminance from RGB color
            // If luminance is high (light background), use dark text; otherwise use light text
            const r = rgb[0];
            const g = rgb[1];
            const b = rgb[2];
            
            // Calculate relative luminance using the formula for sRGB
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            
            // Convert normalized threshold (0-1) to 0-255 range for comparison
            const threshold = contrastThreshold * 255;
            
            // Compare against the adjustable threshold
            // Normal: high luminance (light bg) → black text, low luminance (dark bg) → white text
            // Reverse: swap the logic
            if (reverseContrast) {{
                return luminance > threshold ? 'white' : 'black';
            }} else {{
                return luminance > threshold ? 'black' : 'white';
            }}
        }}
        
        function getColorForValue(value, zmin, zmax, colorscale) {{
            // Normalize value to 0-1 range
            const normalized = (value - zmin) / (zmax - zmin);
            
            // Get RGB color from the colorscale at the normalized position
            // This is a simplified version - Plotly has complex colorscale interpolation
            // For now, we'll sample the colorscale array
            if (Array.isArray(colorscale) && colorscale.length > 0) {{
                const idx = Math.floor(normalized * (colorscale.length - 1));
                const colorStop = colorscale[Math.max(0, Math.min(idx, colorscale.length - 1))];
                if (Array.isArray(colorStop) && colorStop.length > 1) {{
                    return colorStop[1];
                }}
            }}
            
            // Fallback: return a color based on normalized value
            if (normalized < 0.5) {{
                return `rgb(${{Math.round(normalized * 510)}}, ${{Math.round(normalized * 510)}}, 255)`;
            }} else {{
                return `rgb(255, ${{Math.round((1 - normalized) * 510)}}, ${{Math.round((1 - normalized) * 510)}})`;
            }}
        }}
        
        function hexToRgb(hex) {{
            // Convert hex color to RGB array
            if (hex.startsWith('#')) {{
                const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
                return result ? [
                    parseInt(result[1], 16),
                    parseInt(result[2], 16),
                    parseInt(result[3], 16)
                ] : [128, 128, 128];
            }} else if (hex.startsWith('rgb')) {{
                const match = hex.match(/\\d+/g);
                return match ? match.slice(0, 3).map(Number) : [128, 128, 128];
            }}
            return [128, 128, 128];
        }}
        
        function updatePlotSize() {{
            const gd = document.getElementById('heatmap');
            currentWidth = parseInt(document.getElementById('widthSlider').value);
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: dynamicLeftMargin, r: 40, b: dynamicBottomMargin, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightSlider').value);
            }}
            
            // Sync input boxes with sliders
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            
            // Update the layout without recreating the entire plot
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function updatePlotSizeFromInput() {{
            const gd = document.getElementById('heatmap');
            const widthInput = parseInt(document.getElementById('widthInput').value);
            
            // Update width
            currentWidth = widthInput;
            
            // If square cells are locked, auto-adjust height
            if (squareCellsLocked && gd.data && gd.data[0]) {{
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                const margins = gd.layout.margin || {{l: dynamicLeftMargin, r: 40, b: dynamicBottomMargin, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                currentHeight = Math.round(plotAreaHeight + marginVertical);
            }} else {{
                currentHeight = parseInt(document.getElementById('heightInput').value);
            }}
            
            // Update sliders (clamped to their range) and displays
            document.getElementById('widthSlider').value = Math.min(2400, Math.max(400, currentWidth));
            document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
            document.getElementById('widthInput').value = currentWidth;
            document.getElementById('heightInput').value = currentHeight;
            document.getElementById('widthValue').textContent = currentWidth + 'px';
            document.getElementById('heightValue').textContent = currentHeight + 'px';
            
            // Update the layout
            Plotly.relayout(gd, {{
                width: currentWidth,
                height: currentHeight
            }});
        }}
        
        function makeSquareCells() {{
            const gd = document.getElementById('heatmap');
            if (!gd.data || !gd.data[0]) return;
            
            const btn = document.getElementById('squareCellsBtn');
            squareCellsLocked = !squareCellsLocked;
            
            if (squareCellsLocked) {{
                // Lock to square cells
                const numRows = gd.data[0].y.length;
                const numCols = gd.data[0].x.length;
                
                // Get margins (use dynamic margins calculated based on label lengths)
                const margins = gd.layout.margin || {{l: dynamicLeftMargin, r: 40, b: dynamicBottomMargin, t: 100}};
                const marginHorizontal = margins.l + margins.r;
                const marginVertical = margins.t + margins.b;
                
                // Calculate height for square cells based on current width
                const plotAreaWidth = currentWidth - marginHorizontal;
                const plotAreaHeight = plotAreaWidth * numRows / numCols;
                const targetHeight = Math.round(plotAreaHeight + marginVertical);
                
                // Update height
                currentHeight = targetHeight;
                document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, targetHeight));
                document.getElementById('heightInput').value = targetHeight;
                document.getElementById('heightValue').textContent = targetHeight + 'px';
                
                // Lock aspect ratio
                Plotly.relayout(gd, {{
                    width: currentWidth,
                    height: targetHeight,
                    'xaxis.scaleanchor': 'y',
                    'xaxis.scaleratio': 1,
                    'yaxis.constrain': 'domain'
                }});
                
                btn.textContent = '🔓 Unlock Cells';
                btn.style.backgroundColor = '#28a745';
                
                console.log('Square cells LOCKED:', {{
                    numCols: numCols,
                    numRows: numRows,
                    width: currentWidth,
                    height: targetHeight,
                    cellAspectRatio: 1.0
                }});
            }} else {{
                // Unlock - remove aspect ratio constraint
                Plotly.relayout(gd, {{
                    'xaxis.scaleanchor': null,
                    'xaxis.scaleratio': null,
                    'yaxis.constrain': null
                }});
                
                btn.textContent = '⬜ Square Cells';
                btn.style.backgroundColor = '';
                
                console.log('Square cells UNLOCKED - free adjustment enabled');
            }}
        }}
        
        function transposeMatrix() {{
            isTransposed = !isTransposed;
            
            // Update button text
            const btn = document.getElementById('transposeBtn');
            btn.textContent = isTransposed ? '🔄 Restore Original' : '🔄 Swap Rows ↔ Columns';
            btn.style.backgroundColor = isTransposed ? '#17a2b8' : '';
            
            console.log('Matrix transposed:', isTransposed);
            
            // Recreate heatmap with transposed data
            createHeatmap();
        }}
        
        // Row/Column reordering functions
        function resetOrder() {{
            // Reset to original order (before any reordering operations)
            currentXLabels = xLabels.slice();
            currentYLabels = yLabels.slice();
            console.log('Reset to original order');
            closeOrderPanel();  // Close panel if open
            createHeatmap();
        }}
        
        // Drag and drop ordering
        let currentOrderType = null;  // 'rows' or 'cols'
        let draggedItem = null;
        let tempOrder = [];
        
        function toggleOrderPanel(type) {{
            currentOrderType = type;
            const panel = document.getElementById('orderPanel');
            const backdrop = document.getElementById('orderPanelBackdrop');
            const label = document.getElementById('orderPanelLabel');
            const listContainer = document.getElementById('orderList');
            
            // Get current labels based on type and transpose state
            // We need to show the ACTUAL order displayed on heatmap, including clustering
            let labels;
            if (type === 'rows') {{
                // Visual rows = Y-axis
                labels = isTransposed ? currentXLabels.slice() : currentYLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveRowOrder = isTransposed ? colOrderClustered : rowOrderClustered;
                    labels = reorderLabels(labels, effectiveRowOrder);
                }}
                label.textContent = 'Reorder Rows (Y-axis)';
            }} else {{
                // Visual columns = X-axis
                labels = isTransposed ? currentYLabels.slice() : currentXLabels.slice();
                
                // Apply clustering if enabled
                if (useClusteredOrder && clusteringAvailable) {{
                    const effectiveColOrder = isTransposed ? rowOrderClustered : colOrderClustered;
                    labels = reorderLabels(labels, effectiveColOrder);
                }}
                label.textContent = 'Reorder Columns (X-axis)';
            }}
            
            tempOrder = labels.slice();
            console.log('toggleOrderPanel:', type, 'isTransposed:', isTransposed, 'clustered:', useClusteredOrder, 'labels:', labels);
            
            // Create draggable list
            listContainer.innerHTML = '';
            labels.forEach((item, index) => {{
                const div = document.createElement('div');
                div.className = 'drag-item';
                div.draggable = true;
                div.dataset.label = item;
                div.innerHTML = '<span class="drag-handle">☰</span>' + escapeHtml(item);
                
                div.addEventListener('dragstart', handleDragStart);
                div.addEventListener('dragover', handleDragOver);
                div.addEventListener('drop', handleDrop);
                div.addEventListener('dragend', handleDragEnd);
                div.addEventListener('dragenter', handleDragEnter);
                div.addEventListener('dragleave', handleDragLeave);
                
                listContainer.appendChild(div);
            }});
            
            // Show panel and backdrop
            panel.style.display = 'flex';
            backdrop.style.display = 'block';
        }}
        
        function closeOrderPanel() {{
            document.getElementById('orderPanel').style.display = 'none';
            document.getElementById('orderPanelBackdrop').style.display = 'none';
            currentOrderType = null;
            draggedItem = null;
            tempOrder = [];
        }}
        
        function handleDragStart(e) {{
            draggedItem = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.innerHTML);
        }}
        
        function handleDragOver(e) {{
            if (e.preventDefault) {{
                e.preventDefault();
            }}
            e.dataTransfer.dropEffect = 'move';
            return false;
        }}
        
        function handleDragEnter(e) {{
            if (this !== draggedItem) {{
                this.classList.add('drag-over');
            }}
        }}
        
        function handleDragLeave(e) {{
            this.classList.remove('drag-over');
        }}
        
        function handleDrop(e) {{
            if (e.stopPropagation) {{
                e.stopPropagation();
            }}
            
            if (draggedItem !== this) {{
                // Reorder in DOM - insert before the target
                const draggedLabel = draggedItem.dataset.label;
                const targetLabel = this.dataset.label;
                
                const listContainer = document.getElementById('orderList');
                
                // Always insert before the target element
                // This gives consistent behavior: dropping on X puts item before X
                this.parentNode.insertBefore(draggedItem, this);
                
                // Read the new order from DOM to ensure perfect sync
                const itemsAfter = Array.from(listContainer.children);
                tempOrder = itemsAfter.map(item => item.dataset.label);
                
                console.log('Dragged', draggedLabel, 'before', targetLabel, '| New order:', tempOrder);
                
                // Apply immediately to heatmap
                applyReorderImmediate();
            }}
            
            this.classList.remove('drag-over');
            return false;
        }}
        
        function handleDragEnd(e) {{
            this.classList.remove('dragging');
            
            // Remove drag-over class from all items
            const items = document.querySelectorAll('.drag-item');
            items.forEach(item => item.classList.remove('drag-over'));
        }}
        
        function applyReorderImmediate() {{
            if (!currentOrderType || tempOrder.length === 0) return;
            
            // When user manually reorders, disable clustering to respect their choice
            if (useClusteredOrder) {{
                useClusteredOrder = false;
                const orderBtn = document.getElementById('orderBtn');
                if (orderBtn) {{
                    orderBtn.textContent = '🔀 Clustered Order';
                }}
                console.log('Disabled clustering due to manual reordering');
            }}
            
            if (currentOrderType === 'rows') {{
                if (isTransposed) {{
                    currentXLabels = tempOrder.slice();
                }} else {{
                    currentYLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to rows:', tempOrder);
            }} else {{
                if (isTransposed) {{
                    currentYLabels = tempOrder.slice();
                }} else {{
                    currentXLabels = tempOrder.slice();
                }}
                console.log('Applied immediate reorder to columns:', tempOrder);
            }}
            
            createHeatmap();
        }}
        
        function applyDragOrder() {{
            // Just close the panel - reordering already applied immediately
            closeOrderPanel();
        }}
        
        function resetPlotSize() {{
            currentWidth = 800;
            currentHeight = 800;
            document.getElementById('widthSlider').value = 800;
            document.getElementById('heightSlider').value = 800;
            document.getElementById('widthInput').value = 800;
            document.getElementById('heightInput').value = 800;
            document.getElementById('widthValue').textContent = '800px';
            document.getElementById('heightValue').textContent = '800px';
            updatePlotSize();
        }}
        
        function updateExportScale(value) {{
            exportScale = parseFloat(value);
            document.getElementById('exportScaleValue').textContent = value + 'x';
        }}
        
        // Export functions (shared backend)
        function exportPNG() {{
            const gd = document.getElementById('heatmap');
            const filename = 'heatmap_' + currentScale + '_' + new Date().getTime() + '_' + exportScale + 'x.png';
            exportPlotlyToImage(gd, 'png', filename, exportScale, currentWidth, currentHeight, function() {{
                showStatus('✅ PNG exported: ' + Math.round(currentWidth * exportScale) + 'x' + Math.round(currentHeight * exportScale) + 'px', 'success');
            }});
        }}
        
        function exportSVG() {{
            const gd = document.getElementById('heatmap');
            const filename = 'heatmap_' + currentScale + '_' + new Date().getTime() + '.svg';
            // SVG at native size. Heatmaps with up to 100 cells export as
            // vector <rect> cells (editable shapes); larger heatmaps keep
            // Plotly's embedded pixel image so the colors stay crisp after
            // PowerPoint's Convert-to-Shape.
            Plotly.toImage(gd, {{ format: 'svg', width: currentWidth, height: currentHeight }}).then(function(dataUrl) {{
                const svgString = decodeURIComponent(dataUrl.split(',')[1]);
                const imgMatch = svgString.match(/<image[^>]*xlink:href="data:image\/png;base64,[^"]+"[^>]*>/);
                if (!imgMatch) {{
                    downloadBlob(new Blob([svgString], {{ type: 'image/svg+xml' }}), filename);
                    showStatus('✅ SVG exported: ' + currentWidth + 'x' + currentHeight + 'px', 'success');
                    return;
                }}
                const imgTag = imgMatch[0];
                const pngUrl = imgTag.match(/xlink:href="([^"]+)"/)[1];
                vectorizeHeatmapCells(svgString, imgTag, pngUrl).then(function(finalSvg) {{
                    downloadBlob(new Blob([finalSvg], {{ type: 'image/svg+xml' }}), filename);
                    showStatus('✅ SVG exported: ' + currentWidth + 'x' + currentHeight + 'px', 'success');
                }});
            }}).catch(function(error) {{
                console.error('SVG export failed:', error);
                showStatus('⚠️ SVG export failed. See console.', 'error');
            }});
        }}
        
        // Replace the rasterized heatmap-cell <image> with one <rect> per cell
        // (colors taken from the rendered pixels, so the visual result is
        // identical) - but only for heatmaps with at most 100 cells. Larger
        // heatmaps keep the embedded pixel image to avoid color diffusion.
        function vectorizeHeatmapCells(svgString, imgTag, pngUrl) {{
            return new Promise(function(resolve) {{
                const img = new Image();
                img.onload = function() {{
                    const cols = img.naturalWidth;
                    const rows = img.naturalHeight;
                    if (!rows || !cols || rows * cols > 100) {{
                        resolve(svgString);  // >100 cells: keep the embedded pixel image
                        return;
                    }}
                    const x = parseFloat((imgTag.match(/ x="([^"]+)"/) || [])[1] || '0');
                    const y = parseFloat((imgTag.match(/ y="([^"]+)"/) || [])[1] || '0');
                    const w = parseFloat((imgTag.match(/ width="([^"]+)"/) || [])[1] || '0');
                    const h = parseFloat((imgTag.match(/ height="([^"]+)"/) || [])[1] || '0');
                    const canvas = document.createElement('canvas');
                    canvas.width = cols;
                    canvas.height = rows;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const data = ctx.getImageData(0, 0, cols, rows).data;
                    const cellW = w / cols;
                    const cellH = h / rows;
                    let rects = '<g class="heatmap-cells-vector" shape-rendering="crispEdges">';
                    for (let r = 0; r < rows; r++) {{
                        for (let c = 0; c < cols; c++) {{
                            const i = (r * cols + c) * 4;
                            const a = data[i + 3];
                            if (a === 0) continue;  // fully transparent (masked) cell
                            const fill = a < 255
                                ? 'rgba(' + data[i] + ',' + data[i + 1] + ',' + data[i + 2] + ',' + (a / 255) + ')'
                                : 'rgb(' + data[i] + ',' + data[i + 1] + ',' + data[i + 2] + ')';
                            rects += '<rect x="' + (x + c * cellW).toFixed(2) + '" y="' + (y + r * cellH).toFixed(2) +
                                '" width="' + cellW.toFixed(2) + '" height="' + cellH.toFixed(2) + '" fill="' + fill + '"/>';
                        }}
                    }}
                    rects += '</g>';
                    resolve(svgString.replace(imgTag, rects));
                }};
                img.onerror = function() {{ resolve(svgString); }};
                img.src = pngUrl;
            }});
        }}
        
        function saveSettings() {{
            try {{
                const settings = {{
                    // Scale and colorscale
                    scale: currentScale,
                    colorscale: currentColorscale,
                    fontSize: currentFontSize,
                    useAutoRange: useAutoRange,
                    zminSlider: document.getElementById('zminSlider')?.value,
                    zmaxSlider: document.getElementById('zmaxSlider')?.value,
                    // Custom colorscale settings
                    customColorScale: customColorScale,
                    use3PointScale: use3PointScale,
                    colorMin: document.getElementById('colorMin')?.value,
                    colorMax: document.getElementById('colorMax')?.value,
                    colorMin3: document.getElementById('colorMin3')?.value,
                    colorMid3: document.getElementById('colorMid3')?.value,
                    colorMax3: document.getElementById('colorMax3')?.value,
                    valueMin3: document.getElementById('valueMin3')?.value,
                    valueMid3: document.getElementById('valueMid3')?.value,
                    valueMax3: document.getElementById('valueMax3')?.value,
                    // Layout
                    width: currentWidth,
                    height: currentHeight,
                    exportScale: exportScale,
                    showLabels: showLabels,
                    // Data state
                    currentMetric: currentMetric,
                    useClusteredOrder: useClusteredOrder,
                    clusteringMethod: currentClusteringMethod,
                    isTransposed: isTransposed,
                    // Cell values
                    showCellValues: showCellValues,
                    cellValueFontSize: cellValueFontSize,
                    ignoredValuesInput: document.getElementById('ignoreValuesInput')?.value,
                    contrastThreshold: contrastThreshold,
                    reverseContrast: reverseContrast,
                    // UI state
                    squareCellsLocked: squareCellsLocked,
                    // Row/column order after reordering
                    currentXLabels: currentXLabels,
                    currentYLabels: currentYLabels
                }};
                localStorage.setItem(storageKey, JSON.stringify(settings));
                console.log('Settings saved successfully:', settings);
                showStatus('✅ Settings saved!', 'success');
            }} catch (error) {{
                console.error('Error saving settings:', error);
                showStatus('⚠️ Error saving settings', 'error');
            }}
        }}
        
        function loadSettings(showStatusMsg = true) {{
            const saved = localStorage.getItem(storageKey);
            if (saved) {{
                try {{
                    const settings = JSON.parse(saved);
                currentScale = settings.scale || 'linear';
                currentColorscale = settings.colorscale || '{default_colorscale}';
                currentFontSize = settings.fontSize || {fontsize};
                useAutoRange = settings.useAutoRange !== undefined ? settings.useAutoRange : true;
                customColorScale = settings.customColorScale || null;
                use3PointScale = settings.use3PointScale || false;
                
                // Update UI
                document.querySelectorAll('[id^="btn-"]').forEach(btn => btn.classList.remove('active'));
                document.getElementById('btn-' + currentScale).classList.add('active');
                document.getElementById('colorscaleSelect').value = currentColorscale;
                document.getElementById('fontSizeSlider').value = currentFontSize;
                document.getElementById('fontSizeValue').textContent = currentFontSize + 'px';
                
                // Restore custom colors
                if (settings.colorMin) document.getElementById('colorMin').value = settings.colorMin;
                if (settings.colorMax) document.getElementById('colorMax').value = settings.colorMax;
                if (settings.colorMin3) document.getElementById('colorMin3').value = settings.colorMin3;
                if (settings.colorMid3) document.getElementById('colorMid3').value = settings.colorMid3;
                if (settings.colorMax3) document.getElementById('colorMax3').value = settings.colorMax3;
                if (settings.valueMin3) document.getElementById('valueMin3').value = settings.valueMin3;
                if (settings.valueMid3) document.getElementById('valueMid3').value = settings.valueMid3;
                if (settings.valueMax3) document.getElementById('valueMax3').value = settings.valueMax3;
                document.getElementById('use3PointScale').checked = use3PointScale;
                toggle3PointScale();
                
                // Restore plot size (clamp to valid range)
                if (settings.width) {{
                    currentWidth = Math.min(3000, Math.max(400, settings.width));
                    document.getElementById('widthSlider').value = Math.min(2400, Math.max(400, currentWidth));
                    document.getElementById('widthInput').value = currentWidth;
                    document.getElementById('widthValue').textContent = currentWidth + 'px';
                }}
                if (settings.height) {{
                    currentHeight = Math.min(3000, Math.max(400, settings.height));
                    document.getElementById('heightSlider').value = Math.min(2400, Math.max(400, currentHeight));
                    document.getElementById('heightInput').value = currentHeight;
                    document.getElementById('heightValue').textContent = currentHeight + 'px';
                }}
                if (settings.exportScale) {{
                    exportScale = Math.min(5, Math.max(1, settings.exportScale || 2));
                    document.getElementById('exportScaleSlider').value = exportScale;
                    document.getElementById('exportScaleValue').textContent = exportScale + 'x';
                }}
                
                // Restore label visibility
                if (settings.showLabels !== undefined) {{
                    showLabels = settings.showLabels;
                    document.getElementById('toggleLabelsBtn').textContent = showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels';
                }}
                
                // Restore additional state
                if (settings.currentMetric !== undefined && hasMultipleMetrics) {{
                    currentMetric = settings.currentMetric;
                    document.querySelectorAll('.metric-btn').forEach(btn => btn.classList.remove('active'));
                    const metricBtn = document.getElementById('metric-' + currentMetric);
                    if (metricBtn) {{
                        metricBtn.classList.add('active');
                    }}
                }}
                
                if (settings.useClusteredOrder !== undefined && clusteringAvailable) {{
                    useClusteredOrder = settings.useClusteredOrder;
                    const orderBtn = document.getElementById('orderBtn');
                    if (orderBtn) {{
                        orderBtn.textContent = useClusteredOrder ? '📊 Original Order' : '🔀 Clustered Order';
                    }}
                }}
                
                if (settings.clusteringMethod !== undefined && clusteringAvailable) {{
                    currentClusteringMethod = settings.clusteringMethod;
                    const methodSelect = document.getElementById('clusteringMethodSelect');
                    if (methodSelect) {{
                        methodSelect.value = currentClusteringMethod;
                    }}
                    // Update the method selector visibility based on clustering state
                    const methodSection = document.getElementById('clusteringMethodSection');
                    if (methodSection) {{
                        methodSection.style.display = useClusteredOrder ? 'block' : 'none';
                    }}
                }}
                
                if (settings.isTransposed !== undefined) {{
                    isTransposed = settings.isTransposed;
                    const transposeBtn = document.getElementById('transposeBtn');
                    if (transposeBtn) {{
                        transposeBtn.textContent = isTransposed ? '🔄 Un-Transpose' : '🔄 Transpose';
                    }}
                }}
                
                if (settings.showCellValues !== undefined) {{
                    showCellValues = settings.showCellValues;
                    const cellValuesBtn = document.getElementById('toggleCellValuesBtn');
                    if (cellValuesBtn) {{
                        cellValuesBtn.textContent = showCellValues ? '🔢 Hide Values' : '🔢 Show Values';
                    }}
                }}
                
                if (settings.cellValueFontSize !== undefined) {{
                    cellValueFontSize = settings.cellValueFontSize;
                    const sizeSlider = document.getElementById('cellValueSizeSlider');
                    const sizeValue = document.getElementById('cellValueSizeValue');
                    if (sizeSlider) sizeSlider.value = cellValueFontSize;
                    if (sizeValue) sizeValue.textContent = cellValueFontSize + 'px';
                }}
                
                if (settings.ignoredValuesInput !== undefined) {{
                    const ignoreInput = document.getElementById('ignoreValuesInput');
                    if (ignoreInput) {{
                        ignoreInput.value = settings.ignoredValuesInput;
                        updateIgnoredValues();
                    }}
                }}
                
                if (settings.contrastThreshold !== undefined) {{
                    contrastThreshold = settings.contrastThreshold;
                    const thresholdSlider = document.getElementById('contrastThresholdSlider');
                    const thresholdValue = document.getElementById('contrastThresholdValue');
                    if (thresholdSlider) thresholdSlider.value = contrastThreshold;
                    if (thresholdValue) thresholdValue.textContent = contrastThreshold.toFixed(4);
                }}
                
                if (settings.reverseContrast !== undefined) {{
                    reverseContrast = settings.reverseContrast;
                }}
                
                if (settings.squareCellsLocked !== undefined) {{
                    squareCellsLocked = settings.squareCellsLocked;
                    const lockBtn = document.getElementById('lockSquareCellsBtn');
                    if (lockBtn) {{
                        lockBtn.textContent = squareCellsLocked ? '🔓 Unlock Square Cells' : '🔒 Lock Square Cells';
                    }}
                }}
                
                // Restore row/column order after custom reordering
                if (settings.currentXLabels && Array.isArray(settings.currentXLabels)) {{
                    currentXLabels = settings.currentXLabels.slice();
                }}
                if (settings.currentYLabels && Array.isArray(settings.currentYLabels)) {{
                    currentYLabels = settings.currentYLabels.slice();
                }}
                
                if (!useAutoRange && settings.zminSlider && settings.zmaxSlider) {{
                    document.getElementById('zminSlider').value = settings.zminSlider;
                    document.getElementById('zmaxSlider').value = settings.zmaxSlider;
                    updateColorbarRange();
                }}
                
                    createHeatmap();
                    if (showStatusMsg) {{
                        showStatus('✅ Settings loaded!', 'success');
                    }}
                }} catch (error) {{
                    console.error('Error loading settings:', error);
                    if (showStatusMsg) {{
                        showStatus('⚠️ Error loading settings, using defaults', 'error');
                    }}
                    createHeatmap();
                }}
            }} else {{
                if (showStatusMsg) {{
                    showStatus('ℹ️ No saved settings found', 'info');
                }}
            }}
        }}
        
        function resetSettings() {{
            currentScale = 'linear';
            currentColorscale = '{default_colorscale}';
            currentFontSize = {fontsize};
            customColorScale = null;
            use3PointScale = false;
            currentWidth = 800;
            currentHeight = 800;
            exportScale = 2;
            showLabels = !isLarge;
            
            document.querySelectorAll('[id^="btn-"]').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-linear').classList.add('active');
            document.getElementById('colorscaleSelect').value = currentColorscale;
            document.getElementById('fontSizeSlider').value = currentFontSize;
            document.getElementById('fontSizeValue').textContent = currentFontSize + 'px';
            
            // Reset plot size
            document.getElementById('widthSlider').value = 800;
            document.getElementById('heightSlider').value = 800;
            document.getElementById('widthValue').textContent = '800px';
            document.getElementById('heightValue').textContent = '800px';
            document.getElementById('exportScaleSlider').value = 2;
            document.getElementById('exportScaleValue').textContent = '2x';
            
            // Reset custom color inputs
            document.getElementById('colorMin').value = '#ffffff';
            document.getElementById('colorMax').value = '#68379c';
            document.getElementById('colorMin3').value = '#0000ff';
            document.getElementById('colorMid3').value = '#ffffff';
            document.getElementById('colorMax3').value = '#ff0000';
            document.getElementById('use3PointScale').checked = false;
            toggle3PointScale();
            
            createHeatmap();
            showStatus('✅ Reset to defaults', 'success');
        }}
        
        // Background color toggle (shared controller)
        // White maps to #ffffff for parity with the network/Sankey templates
        // (the page CSS default of #f5f5f5 remains until the first toggle).
        const bgCtrl = createBackgroundController(['#ffffff', '#000000'], ['White', 'Black'], applyBackground);
        
        function toggleBackground() {{
            bgCtrl.toggle('');
        }}
        
        function applyBackground(color) {{
            document.body.style.background = color;
            document.querySelector('.main-container').style.background = color;
            document.getElementById('heatmap-container').style.background = color;
            
            // Update Plotly layout background
            const gd = document.getElementById('heatmap');
            if (gd) {{
                Plotly.relayout(gd, {{
                    'paper_bgcolor': color,
                    'plot_bgcolor': color
                }});
            }}
            
            // Adjust text colors based on background luminance
            const isDark = isColorDark(color);
            
            // Update control panel text colors
            document.querySelectorAll('.control-section h3, .control-section label, .slider-value, .info-box').forEach(el => {{
                el.style.color = isDark ? '#e0e0e0' : (el.classList.contains('slider-value') ? '#4CAF50' : '#495057');
            }});
            
            // Update controls background for dark mode
            document.querySelectorAll('.controls, .control-section').forEach(el => {{
                el.style.background = isDark ? '#2d2d44' : (el.classList.contains('controls') ? 'white' : '#f8f9fa');
                el.style.borderColor = isDark ? '#444' : '#e9ecef';
            }});
        }}
        
        function applyCustomBackground() {{
            bgCtrl.applyCustom();
        }}
        
{SHARED_JS}
        
        function showStatus(message, type) {{
            showStatusInContainer('settingsStatus', message, type || 'info');
        }}
        
        // Try to load saved settings on page load
        window.addEventListener('load', () => {{
            // Initialize custom color range if zmin/zmax were provided from Python
            if (customZmin !== null && customZmax !== null) {{
                window.customColorRange = {{ min: customZmin, max: customZmax }};
                console.log('Initialized custom color range from Python parameters:', window.customColorRange);
            }}
            
            const saved = localStorage.getItem(storageKey);
            if (saved) {{
                loadSettings(false);  // Silent load on initialization
                window.__drocatHasSavedSettings = true;
            }} else {{
                window.__drocatHasSavedSettings = false;
                // createHeatmap() will apply clustering automatically if useClusteredOrder=true
                // No need to pre-set currentYLabels/currentXLabels here
                createHeatmap();
            }}
        }});
        {SQUARE_CELLS_ONLOAD}
    </script>
</body>
</html>
'''
    
    # Write HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    if showfig:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(filename))
