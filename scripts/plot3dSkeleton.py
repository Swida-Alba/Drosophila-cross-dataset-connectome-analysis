# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "plot3d_skeleton" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/visualization.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.
import sys
from pathlib import Path
import bokeh.palettes as bp

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from visualize_skeleton import VisualizeSkeleton

# please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
# dataset = 'hemibrain:v1.2.1'
# Note: optic-lobe:v1.1 is in JRCFIB2018Fraw coordinates (same as hemibrain).
# brain_mesh='native' keeps the scene in that own template space; 'FAFB' draws
# the FLYWIRE outline as the scene frame instead. (The former brain_mesh='whole'
# JRC2018F scene transform was retired.)


vs = VisualizeSkeleton(
    # dataset = 'flywire_FAFB_v783', 
    dataset = 'male-cns:v0.9',
    # dataset = 'hemibrain:v1.2.1',
    token = '',
    output_dir='../local_data/plot_3d',
    neuron_layers = ['aMe12', 'aMe9', 'aMe10'], # or in the format: 'VA1d_adPN -> LHCENT3 -> MBON01'
    custom_layer_names = [],  # Optional custom names for layers
    # layer_map_csv = 'path/to/mapping.csv',  # Optional CSV file mapping neurons to layers
    neuron_colors=bp.Category20[20],
    skip_synapse = True,
    neuron_alpha = 0.2,
    min_synapse_num = 3,
    synapse_size = 'real',
    synapse_alpha = 0.6,
    skeleton_mode = 'tube',
    synapse_mode = 'cone',
    legend_mode='tree',  # 'single', 'type', 'tree', or 'layer' — 'tree' is the
                        # UI default (ui/config.py DEFAULTS); the backend's own
                        # default is 'layer', so this line is what makes the
                        # template reproduce the tab.
    
    export_views=True,
    show_fig=True,
    
    background_color='white',  # Background color: 'white' (default), 'black', or any CSS color
    
    mesh_roi = ['EB', 'LH', 'AL'],
    # mesh_color=(1,1,1,0),
    brain_mesh='native',
    vnc_mesh=False,
    cache_neurons=True,
    cache_synapses=True,
    
    export_method='webdriver',  # 'webdriver' (default, fast, requires Chrome browser version 109+), or 'kaleido' (slower fallback, stable)
    export_scale=3,
)

vs.plot_neurons()

# Export PDF with individual neuron profiles, per-legend plots. 
# legend_mode='single' shows individual neurons, 'type' groups by type, 'layer' groups by layer.
# plot_individuals() should be called AFTER plot_neurons() for correct figure references.

vs.plot_individuals(
    pdf_images_per_page=(3, 2),  # (columns, rows)
    views=['front'],
    summary_format=['pdf', 'pptx'],  # 'pdf', 'pptx', or ['pdf', 'pptx'] for both
)

# Export rotating video: degree_per_frame controls rotation speed
# 1.0 = 360 frames (12 sec at 30fps), 2.0 = 180 frames (6 sec)
# rotate='horizontal' (default) or 'vertical'

vs.export_video(
    fps=30, 
    degree_per_frame=1.0, 
    rotate='horizontal', 
    scale=3,
    export_gif=True,      # Enable GIF export (default: True)
    gif_scale=0.2,        # GIF resolution scale (default: 0.2)
)
