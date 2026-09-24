import sys
from pathlib import Path
import warnings
import time

# Add project root and src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings("ignore")
from src.coana import FindNeuronConnection

if __name__ == '__main__':
    fc = FindNeuronConnection(
        # Token automatically loaded from config_local.json (recommended) or set token='' here
        output_dir='../local_data/connection_data',
        dataset='male-cns:v0.9', 
        # dataset='hemibrain:v1.2.1',
        # dataset='optic-lobe:v1.1',
        # dataset='flywire_FAFB_v783',
        sourceNeurons=['aMe12'],  # pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
        targetNeurons=['PPL101'],
        custom_source_name='', # you can specify a custom name for the source neurons, especially when you are using a list of many types of neurons or a list of neurons read from a file
        custom_target_name='',  # you can specify a custom name for the target neurons
        custom_source_group_names=[],
        custom_target_group_names=[],
        min_synapse_num=3,
        min_ratio=0.0,
        min_traversal_probability=0,
        filter_by='bodyId',  # 'bodyId' or 'type' level filtering
        showfig=False,
        max_interlayer=2,
        keyword_in_path_to_remove=['None'],
        network_layout='distributed',
        use_cache=True,  # Enable caching for faster subsequent runs
        edgeN_limit=500,
        output_format='csv',  # 'xlsx' (default) or 'csv'
        # StrongestFirst is the dataclass default (src/coana.py:3088) AND the
        # value the Find Path tab sends (ui/tabs/find_path.py:345); this line
        # claimed MemoizedDFS was the default while naming neither.
        pathfinding='StrongestFirst',  # 'StrongestFirst' | 'MemoizedDFS' | 'DFS' | 'MeetInMiddle' | 'DP' | 'Bidirectional'
        # The two budgets the Find Path tab sends (ui/config.py DEFAULTS):
        # 0 = auto, which resolves to the internal 1M cap, and 1,000,000
        # edges. Left unset here they would fall back to the dataclass's None
        # (coana.py:3113,3129), i.e. an unbounded traversal the UI never runs.
        max_paths_bodyid=0,  # "Max Paths (BodyId)"; 0 = auto
        graph_edge_limit_bodyid=1000000,  # "Edge Budget"
        drop_untyped=True,  # untyped neurons never anchor a reported path
        skip_bodyId=True,
    )

    start_time = time.time()
    fc.InitializeNeuronInfo()
    # fc.FindPath()
    fc.FindAllPath(forward_only=True)
    end_time = time.time()
    print(f"Pathfinding completed in {end_time - start_time:.2f} seconds")
