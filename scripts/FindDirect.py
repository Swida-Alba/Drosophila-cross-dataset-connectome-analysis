# DROCAT edit-to-run TEMPLATE, not a CLI — and with NO UI mirror:
# FindDirectConnections appears nowhere in ui/ (only in
# src/coana.py), so this script is the only surface for it. Values
# below are pinned to ui/config.py DEFAULTS anyway, so a future tab
# agrees with it. Copy to archive/scripts_local/ before a real
# scientific run (skills/drocat-usage/SKILL.md:66); relative paths
# assume a working directory of scripts/.
import sys
from pathlib import Path
import warnings

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from coana import FindNeuronConnection

if __name__ == '__main__':
    
    fc = FindNeuronConnection(
        # Token automatically loaded from config_local.json (recommended) or set token='' here
        output_dir='../local_data/connection_data',
        # dataset='flywire_FAFB_v783',  # Combined dataset and version in one parameter
        dataset = 'male-cns:v0.9',
        # dataset='hemibrain:v1.2.1',
        sourceNeurons=['aMe.*'],  # [] = all neurons; or list of types/instances like ['L3.*_R']
        targetNeurons=[],  # pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist()
        exclude_intra_type_connections=False,
        custom_source_name='',  # Custom name for source neurons (useful when using [] or reading from file)
        custom_target_name='',  # Custom name for target neurons
        min_synapse_num=3,
        min_ratio=0.0,  # Minimum connection ratio (weight/post) - direct ratio without 0.3 scaling
        min_traversal_probability=0.0,  # Minimum traversal probability (ratio/0.3, capped at 1.0)
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        use_cache=True,  # Enable caching for faster subsequent runs
        network_layout='distributed',
        output_format='csv',  # 'xlsx' (default) or 'csv'
        # 500 is both the dataclass default (coana.py:3366) and the UI's; the
        # 50 here silently cut the visualised edge budget below what any UI run
        # would have produced.
        edgeN_limit=500,
    )

    fc.InitializeNeuronInfo()
    fc.FindDirectConnections()
