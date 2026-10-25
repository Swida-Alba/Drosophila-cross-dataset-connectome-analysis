#!/usr/bin/env python3
"""
Build Connection Cache

Simplified script that uses FindNeuronConnection.build_connection_cache()
to pre-build the connection cache for a dataset.

The connection cache stores:
- Pre-synaptic and post-synaptic bodyIds
- Connection weights (synapse counts)
- Neuron type information (type_pre, type_post)

Usage:
    python build_connection_cache.py <dataset> [options]
    
Examples:
    # Build cache for hemibrain
    python build_connection_cache.py hemibrain:v1.2.1 --token YOUR_TOKEN
    
    # Build cache for FAFB (local files)
    python build_connection_cache.py flywire_FAFB_v783
    
    # Build for specific neuron types
    python build_connection_cache.py hemibrain:v1.2.1 --types Mi1 T4a aMe12
    
    # Show cache statistics
    python build_connection_cache.py --stats hemibrain:v1.2.1

Author: Hemibrain Analysis Team
Date: December 2024
"""

import sys
import time
import argparse
from pathlib import Path

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# DEFAULT PARAMETERS - Edit these to run the script directly
# =============================================================================
DEFAULT_DATASET = 'hemibrain:v1.2.1'  # Dataset name (use : format for NeuPrint)
DEFAULT_TOKEN = ''  # NeuPrint API token (or set NEUPRINT_TOKEN env var)
DEFAULT_SERVER = 'https://neuprint.janelia.org'  # NeuPrint server
DEFAULT_BATCH_SIZE = 100  # Neurons per batch
DEFAULT_NEURON_TYPES = None  # List of types to cache, or None for all
# =============================================================================


def build_cache(
    dataset: str,
    token: str = None,
    server: str = DEFAULT_SERVER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    neuron_types: list = None
) -> dict:
    """
    Build connection cache using FindNeuronConnection.
    
    Parameters:
        dataset: Dataset name (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')
        token: NeuPrint API token (required for NeuPrint datasets)
        server: NeuPrint server URL
        batch_size: Number of neurons per API batch
        neuron_types: List of neuron types to cache, or None for all
        
    Returns:
        dict: Summary with total_neurons, total_connections, elapsed_time, etc.
    """
    from coana import FindNeuronConnection
    
    print("=" * 60)
    print("Building Connection Cache")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Batch size: {batch_size}")
    if neuron_types:
        print(f"Neuron types: {neuron_types}")
    else:
        print("Neuron types: ALL")
    print()
    
    # Use TokenManager to get token
    try:
        from utils.token_manager import token_manager
        token = token_manager.get_token('NEUPRINT_TOKEN', token)
    except ImportError:
        pass

    # Initialize FNC with cache enabled
    fnc = FindNeuronConnection(
        dataset=dataset,
        server=server,
        token=token,
        use_cache=True,
        verbose_mode='full'
    )
    
    # Build cache using FNC's built-in method
    result = fnc.build_connection_cache(
        neuron_types=neuron_types,
        batch_size=batch_size
    )
    
    # Show summary
    safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    print()
    print("=" * 60)
    print("Cache Build Summary")
    print("=" * 60)
    print(f"Total neurons processed: {result.get('total_neurons', 0):,}")
    print(f"Total connections cached: {result.get('total_connections', 0):,}")
    print(f"Cached neurons: {len(result.get('cached_neurons', [])):,}")
    print(f"Failed neurons: {len(result.get('failed_neurons', [])):,}")
    print(f"Elapsed time: {result.get('elapsed_time', 0):.1f} seconds")
    print(f"Cache location: cache/{safe_name}/")
    
    return result


def complete_neuron_count(index_df, conn_df) -> int:
    """Neurons whose cached rows cover their recorded ``connection_count``.

    A neuron with one cached row out of ten must not count as fully cached;
    the old union of "has any row" + zero-outdegree flags hid exactly that
    truncation. Zero-outdegree neurons are complete only when the flag says
    so.
    """
    if 'downstream_complete' not in index_df.columns:
        return 0
    flagged_zero = set(
        index_df.loc[index_df['downstream_complete'], 'bodyId'].astype(str)
    )
    if 'connection_count' in index_df.columns and conn_df is not None \
            and 'bodyId_pre' in conn_df.columns:
        counts = conn_df['bodyId_pre'].astype(str).value_counts().to_dict()
        complete = 0
        for bid, want in zip(index_df['bodyId'].astype(str),
                             index_df['connection_count'].fillna(0)):
            want = int(want)
            if want <= 0:
                complete += int(bid in flagged_zero)
            elif counts.get(bid, 0) >= want:
                complete += 1
        return complete
    rows_pre = set()
    if conn_df is not None and 'bodyId_pre' in getattr(conn_df, 'columns', ()):
        rows_pre = set(conn_df['bodyId_pre'].astype(str))
    return len(rows_pre | flagged_zero)


def show_stats(dataset: str) -> None:
    """Show cache statistics for a dataset."""
    import pandas as pd
    
    # Normalize dataset name
    safe_name = canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')
    
    # Get paths
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    cache_dir = project_root / 'cache' / safe_name
    
    print()
    print("=" * 60)
    print(f"Connection Cache Statistics: {dataset}")
    print("=" * 60)
    
    conn_file = cache_dir / 'connections.parquet'
    index_file = project_root / 'neuron_indexes' / safe_name / 'neuron_index.parquet'
    
    if not conn_file.exists():
        print(f"\n[ERROR] Connection cache not found: {conn_file}")
        print(f"        Run: python build_connection_cache.py {dataset}")
        return
    
    # Connection stats
    try:
        conn_df = pd.read_parquet(conn_file)
        print(f"\nConnection Cache: {conn_file.name}")
        print(f"  File size: {conn_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"  Last modified: {time.ctime(conn_file.stat().st_mtime)}")
        print(f"  Total connections: {len(conn_df):,}")
        print(f"  Unique upstream neurons: {conn_df['bodyId_pre'].nunique():,}")
        print(f"  Unique downstream neurons: {conn_df['bodyId_post'].nunique():,}")
        if 'weight' in conn_df.columns:
            print(f"  Total synapse count: {conn_df['weight'].sum():,}")
    except Exception as e:
        print(f"[ERROR] Could not read connection cache: {e}")
    
    # Neuron index stats
    if index_file.exists():
        try:
            index_df = pd.read_parquet(index_file)
            print(f"\nNeuron Index: {index_file.name}")
            print(f"  File size: {index_file.stat().st_size / (1024*1024):.2f} MB")
            print(f"  Total neurons indexed: {len(index_df):,}")
            if 'downstream_complete' in index_df.columns:
                complete = complete_neuron_count(
                    index_df, locals().get('conn_df'))
                print(f"  Fully cached neurons: {complete:,} ({100*complete/len(index_df):.1f}%)")
        except Exception as e:
            print(f"[WARNING] Could not read neuron index: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build connection cache using FindNeuronConnection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default parameters (edit DEFAULT_* at top of file)
    python build_connection_cache.py
    
    # Build cache for hemibrain
    python build_connection_cache.py hemibrain:v1.2.1 --token YOUR_TOKEN
    
    # Build cache for specific neuron types
    python build_connection_cache.py hemibrain:v1.2.1 --types Mi1 T4a aMe12
    
    # Build cache for FAFB (uses local files)
    python build_connection_cache.py flywire_FAFB_v783
    
    # Show cache statistics
    python build_connection_cache.py --stats
"""
    )
    
    parser.add_argument('dataset', nargs='?', default=DEFAULT_DATASET,
                        help='Dataset name (e.g., hemibrain:v1.2.1)')
    parser.add_argument('--token', '-t', type=str, default=DEFAULT_TOKEN,
                        help='NeuPrint API token')
    parser.add_argument('--server', '-s', type=str, default=DEFAULT_SERVER,
                        help='NeuPrint server URL')
    parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Neurons per batch')
    parser.add_argument('--types', nargs='+', default=DEFAULT_NEURON_TYPES,
                        help='Specific neuron types to cache')
    parser.add_argument('--stats', action='store_true',
                        help='Show cache statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats(args.dataset)
        return
    
    build_cache(
        dataset=args.dataset,
        token=args.token,
        server=args.server,
        batch_size=args.batch_size,
        neuron_types=args.types
    )


if __name__ == '__main__':
    main()
