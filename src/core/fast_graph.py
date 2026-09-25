"""
FastGraph - Lightweight Graph Implementation for Connectome Analysis

This module re-exports FastGraph from vispath_pkg to maintain a single source
of truth while preserving backward compatibility for existing imports.

The full implementation is in vispath-subproject/src/vispath_pkg/fast_graph.py
which provides:
- NetworkX-compatible API with node/edge attributes
- All pathfinding algorithms (DFS, BFS, meet-in-the-middle, etc.)
- DataFrame construction methods for both Pandas and Polars
- Label aggregation for type-level analysis

Example Usage:
    from core.fast_graph import FastGraph
    
    # Build graph from DataFrame
    G = FastGraph.build_from_dataframe(
        df, source_col='pre', target_col='post', weight_col='weight'
    )
    
    # Find all paths
    for path in G.all_simple_paths(source, target, cutoff=3):
        print(path)
    
    # Aggregate by neuron type
    G_type = G.aggregate_by_label(bodyid_to_type_map)
"""

import sys
from pathlib import Path

# Add vispath-subproject to path for import
_vispath_path = Path(__file__).parent.parent.parent / "vispath-subproject" / "src"
if _vispath_path.is_dir() and str(_vispath_path) not in sys.path:
    sys.path.insert(0, str(_vispath_path))

# Re-export FastGraph/DiGraph from vispath_pkg, resolved lazily (PEP 562):
# vispath_pkg ships from the subproject's own packaging and is not inside
# the drocat wheel, so importing this shim must not hard-require it.
_VSPATH_HINT = (
    "requires the vispath subproject (vispath_pkg). In a repo checkout it "
    "is loaded automatically; with a wheel install run: "
    "pip install ./vispath-subproject")


def __getattr__(name):
    if name in ("FastGraph", "DiGraph"):
        try:
            from vispath_pkg.fast_graph_core import FastGraph, DiGraph
            return {"FastGraph": FastGraph, "DiGraph": DiGraph}[name]
        except ImportError as exc:
            raise ImportError(f"Pathfinding enumeration {_VSPATH_HINT}") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
