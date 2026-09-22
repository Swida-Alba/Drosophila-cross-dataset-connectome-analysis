"""DROCAT UI Tabs package."""

from .find_path import create_find_path_tab
from .find_shortest import create_find_shortest_tab
from .network import create_network_tab
from .connectivity import create_connectivity_tab
from .morphology import create_morphology_tab
from .inter_dataset import create_inter_dataset_tab
from .type_validation import create_type_validation_tab
from .nb_find_lines import create_nb_find_lines_tab
from .nb_find_neuron import create_nb_find_neuron_tab
from .nb_colabel import create_nb_colabel_tab
from .flylight import create_flylight_tab
from .visualization import (
    create_skeleton_tab,
    create_net_viz_tab,
    create_visualization_tab,
)
from .settings import create_settings_tab

__all__ = [
    "create_find_path_tab",
    "create_find_shortest_tab",
    "create_network_tab",
    "create_connectivity_tab",
    "create_morphology_tab",
    "create_inter_dataset_tab",
    "create_type_validation_tab",
    "create_nb_find_lines_tab",
    "create_nb_find_neuron_tab",
    "create_nb_colabel_tab",
    "create_flylight_tab",
    "create_skeleton_tab",
    "create_net_viz_tab",
    "create_visualization_tab",
    "create_settings_tab",
]
