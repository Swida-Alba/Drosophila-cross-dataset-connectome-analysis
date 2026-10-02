"""
FastGraph - Lightweight Graph Implementation for VisualizePath

A minimal, efficient graph implementation optimized for visualization tasks.
Replaces NetworkX for simple graph operations to reduce dependencies and overhead.

Date: 2025-12
"""

import sys
import time
from collections import defaultdict


class LineProgress:
    """Single-line progress display that refreshes in place with ``\r``.

    tqdm only rewrites the line when the stream is a TTY; when the output
    is piped or captured it emits one line per update, spamming the log.
    This display always rewrites the SAME line, so progress stays on one
    line in terminals, in captured output and in logs alike.

    Nested displays defer: only the most recently created running bar
    writes the line (e.g. a pathfinding ``L{length}`` bar takes over while
    the consumer's streaming bar wraps its generator); when it closes the
    outer bar regains the line.

    Provides the tqdm API subset the pathfinding loops use (iteration,
    ``update``, ``set_postfix``, ``set_postfix_str`` and ``close``) so it
    can drop in where a bar was displayed.
    """

    _active = []  # running instances, oldest first

    def __init__(self, iterable, desc, total=None, leave=False, unit='it'):
        self._it = iter(iterable)
        self._desc = desc
        self._total = total
        if self._total is None and hasattr(iterable, '__len__'):
            self._total = len(iterable)
        self._leave = leave
        self._unit = unit
        self._n = 0
        self._start = time.time()
        self._last_refresh = 0.0
        self._last_len = 0
        self._postfix = ''
        self._closed = False
        LineProgress._active.append(self)
        # Show the initial state immediately. The first memoized DFS state can
        # be expensive, and without this refresh the log window appears idle
        # until that state finishes.
        self._refresh(force=True)

    # -- tqdm-compatible API ------------------------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._it)
        except StopIteration:
            self.close()
            raise
        self._n += 1
        self._refresh()
        return item

    def update(self, n=1):
        self._n += n
        self._refresh(force=True)

    def set_postfix(self, refresh=False, **kwargs):
        self._postfix = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        if refresh:
            self._refresh(force=True)

    def set_postfix_str(self, s, refresh=True):
        self._postfix = s
        if refresh:
            self._refresh(force=True)

    def close(self):
        if self._closed:
            return
        was_last = bool(LineProgress._active) and self is LineProgress._active[-1]
        if was_last:
            if self._leave:
                # final state stays on the line, then the line is ended
                self._refresh(force=True)
                sys.stdout.write('\n')
            else:
                # leave=False: clear the line (tqdm-style), no residue
                sys.stdout.write('\r' + ' ' * self._last_len + '\r')
            sys.stdout.flush()
        try:
            LineProgress._active.remove(self)
        except ValueError:
            pass
        self._closed = True

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _fmt_secs(secs):
        secs = max(int(secs), 0)
        m, s = divmod(secs, 60)
        return f'{m:02d}:{s:02d}'

    def _refresh(self, force=False):
        if self._closed:
            return
        # A nested (more recent) bar owns the line; defer until it closes.
        if LineProgress._active and self is not LineProgress._active[-1]:
            return
        now = time.time()
        if not force and now - self._last_refresh < 0.1:
            return
        self._last_refresh = now
        elapsed = now - self._start
        rate = self._n / elapsed if elapsed > 0 else 0.0
        if self._total:
            pct = self._n / self._total * 100
            head = f'{self._desc}: {self._n}/{self._total} ({pct:4.1f}%)'
            eta = (self._total - self._n) / rate if rate > 0 else None
            eta_str = f'<{self._fmt_secs(eta)}' if eta is not None else '<?'
        else:
            head = f'{self._desc}: {self._n}{self._unit}'
            eta_str = ''
        msg = (f'\r{head} [{self._fmt_secs(elapsed)}{eta_str}, '
               f'{rate:,.1f}{self._unit}/s]')
        if self._postfix:
            msg += f' | {self._postfix}'
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._last_len = len(msg) - 1  # line length without the \r


def drain_shortest_merge(merged, budget, stats=None):
    """Apply the StrongestFirst budget + tau tie-drain to a merged stream.

    ``merged`` is an iterable of ``(bottleneck, path, target)`` tuples in
    DESCENDING bottleneck order (the merged per-target streams of
    ``FastGraph.find_paths_shortest_strongest_first``, or the outer merge
    of several batch emissions). Emits ``path`` for each item; once
    ``budget`` paths are out, every remaining tie at the achieved
    bottleneck tau is drained and the first strictly-weaker item records
    ``strongest_dropped``. This is the single shared implementation of the
    shortest-mode budget semantics — the monolithic enumerator and the
    batched-discovery pipeline both route through it, so their
    ``stats`` (``emitted`` / ``tau`` / ``budget_bitten`` /
    ``strongest_dropped`` / ``per_target``) are produced identically.

    A generator: stats are finalized when the stream is exhausted (same
    contract as the enumerator's in-place drain).
    """
    emitted = 0
    tau = None
    tau_open = False
    strongest_dropped = None
    per_target = {}
    for bottleneck, path, target in merged:
        if tau_open and bottleneck < tau:
            # Everything left across all streams is strictly weaker:
            # the first sub-tau item in the merged (descending) order
            # is the strongest dropped path (w2).
            strongest_dropped = bottleneck
            break
        yield path
        emitted += 1
        per_target[target] = per_target.get(target, 0) + 1
        tau = bottleneck
        if budget is not None and emitted >= budget:
            tau_open = True
    if stats is not None:
        stats['emitted'] = emitted
        stats['tau'] = tau
        stats['budget_bitten'] = bool(budget is not None and tau_open
                                      and strongest_dropped is not None)
        stats['strongest_dropped'] = strongest_dropped
        stats['per_target'] = per_target


def merge_shortest_batch_emissions(batch_emissions, budget=None, stats=None):
    """Merge per-batch shortest emissions into the global emission order.

    Each entry of ``batch_emissions`` is a list of
    ``(bottleneck, path, target)`` tuples produced by
    ``find_paths_shortest_strongest_first(..., budget=None, payload=True)``
    on one discovery batch's standalone graph (each list is already in
    descending-bottleneck order). The lists are ``heapq.merge``-ed in batch
    order and the global ``budget`` is applied exactly once via
    ``drain_shortest_merge``.

    Equivalence contract (plan-shortest-batched-discovery §3.B): batches
    must be CONTIGUOUS runs of ``sorted(targets, key=str)`` — the
    enumerator's own stream order — so this nested merge reproduces the
    flat monolithic tie-breaking exactly. Do not rely on merge stability
    as an undocumented given: the pipeline's equivalence tests pin it with
    ties across batch boundaries.
    """
    import heapq
    merged = heapq.merge(*batch_emissions, key=lambda item: -item[0])
    yield from drain_shortest_merge(merged, budget, stats)


class FastGraph:
    """
    A lightweight directed graph implementation for visualization.
    
    Provides a NetworkX-compatible API subset optimized for:
    - Building graphs from edge lists
    - Node/edge attribute storage
    - Graph iteration and filtering
    
    This is a simplified version that doesn't include pathfinding algorithms
    (not needed for visualization).
    
    Attributes
    ----------
    adj : dict
        Adjacency list: {node: {neighbor: weight, ...}, ...}
    node_attrs : dict
        Node attributes: {node: {attr_name: value, ...}, ...}
    edge_attrs : dict
        Edge attributes: {(u, v): {attr_name: value, ...}, ...}
    """
    
    def __init__(self):
        """Initialize an empty directed graph."""
        self.adj = {}  # {u: {v: weight}}
        self.node_attrs = {}  # {node: {attr: value}}
        self.edge_attrs = {}  # {(u, v): {attr: value}}
        self._num_edges = 0
        # When False, add_edge skips the per-edge edge_attrs dict (weight
        # still lives in adj).  A pure-pathfinding graph never reads them,
        # and the dicts cost ~350 bytes/edge on million-edge graphs.
        self.store_edge_attrs = True
        # Lazy reverse adjacency: {v: set of predecessors u with u -> v}.
        # Built on first use (backward traversals) instead of materialising a
        # full reversed graph copy per algorithm call; invalidated on edits.
        self._radj = None
        self._radj_dirty = False

    def add_node(self, n, **attrs):
        """
        Add a single node with optional attributes.
        
        Parameters
        ----------
        n : hashable
            Node identifier
        **attrs : keyword arguments
            Node attributes (e.g., node_type='source', color='#ff0000')
        """
        if n not in self.adj:
            self.adj[n] = {}
        if n not in self.node_attrs:
            self.node_attrs[n] = {}
        self.node_attrs[n].update(attrs)

    def add_edge(self, u, v, weight=1.0, **attrs):
        """
        Add an edge from u to v with optional weight and attributes.
        
        Parameters
        ----------
        u : hashable
            Source node
        v : hashable
            Target node
        weight : float, optional
            Edge weight (default: 1.0). If edge exists, weights are summed.
        **attrs : keyword arguments
            Additional edge attributes (e.g., ratio=0.5, probability=0.8)
        """
        # Ensure nodes exist
        if u not in self.adj:
            self.adj[u] = {}
            self.node_attrs[u] = {}
        if v not in self.adj:
            self.adj[v] = {}
            self.node_attrs[v] = {}
        
        # Add/update edge
        if v not in self.adj[u]:
            self.adj[u][v] = 0.0
            self._num_edges += 1
        self.adj[u][v] += weight
        # Topology may have changed; the lazy reverse index is rebuilt on demand.
        self._radj_dirty = True
        
        # Store edge attributes (skippable for slim pathfinding graphs)
        if self.store_edge_attrs:
            edge_key = (u, v)
            if edge_key not in self.edge_attrs:
                self.edge_attrs[edge_key] = {}
            self.edge_attrs[edge_key]['weight'] = self.adj[u][v]
            self.edge_attrs[edge_key].update(attrs)

    def number_of_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self.adj)

    def number_of_edges(self):
        """Return the number of edges in the graph."""
        return self._num_edges

    def edges(self, data=False):
        """
        Return edges, optionally with attributes.
        
        Parameters
        ----------
        data : bool, optional
            If True, return (u, v, attr_dict) tuples. Default: False.
            
        Returns
        -------
        iterator
            Edge iterator
        """
        for u in self.adj:
            for v, w in self.adj[u].items():
                if data:
                    attrs = self.edge_attrs.get((u, v), {'weight': w})
                    yield (u, v, attrs)
                else:
                    yield (u, v)

    def __contains__(self, node):
        """Check if node exists in graph."""
        return node in self.adj

    def has_node(self, n):
        """Check if node n exists in graph."""
        return n in self.adj

    def has_edge(self, u, v):
        """Check if edge (u, v) exists in graph."""
        return u in self.adj and v in self.adj[u]

    def neighbors(self, n):
        """Return iterator over neighbors of node n."""
        return iter(self.adj.get(n, {}))

    def successors(self, n):
        """Return iterator over successors of node n (same as neighbors for DiGraph)."""
        return self.neighbors(n)

    def predecessors(self, n):
        """Return iterator over predecessors of node n (u with u -> n)."""
        return iter(self._ensure_radj().get(n, ()))

    def _ensure_radj(self):
        """Build the reverse adjacency index on first use / after edits.

        O(V + E) once, then O(1) predecessor lookups — replaces the full
        reversed-graph copy (``reverse()``) that pathfinding used to build
        per call, which cost ~250 MB on a 722k-edge connectome.
        """
        if self._radj is None or self._radj_dirty:
            radj = {}
            for u, vs in self.adj.items():
                for v in vs:
                    radj.setdefault(v, set()).add(u)
            self._radj = radj
            self._radj_dirty = False
        return self._radj

    def out_degree(self, n):
        """Return out-degree of node n."""
        return len(self.adj.get(n, {}))

    def in_degree(self, n):
        """Return in-degree of node n (O(1) via the reverse index)."""
        return len(self._ensure_radj().get(n, ()))

    def get_edge_data(self, u, v, default=None):
        """
        Return edge attributes for edge (u, v).
        
        Parameters
        ----------
        u, v : hashable
            Source and target nodes
        default : any, optional
            Value to return if edge doesn't exist
            
        Returns
        -------
        dict or default
            Edge attribute dictionary
        """
        if u in self.adj and v in self.adj[u]:
            return self.edge_attrs.get((u, v), {'weight': self.adj[u][v]})
        return default

    @property
    def _node(self):
        """
        NetworkX-compatible node attribute access.
        Returns a dict-like object for node attributes.
        """
        return self.node_attrs

    @property
    def nodes(self):
        """
        NetworkX-compatible nodes property.
        Returns a NodeView-like object that supports both iteration and attribute access.
        
        Usage:
            - for n in G.nodes: ...  (iteration)
            - for n in G.nodes(): ...  (iteration via call)
            - G.nodes[n]['attr'] = value  (attribute access)
            - G.nodes[n].get('attr', default)  (safe attribute access)
        """
        return _NodeView(self)




    def reverse(self, copy=True):
        """
        Return a graph with all edges reversed.
        
        Parameters
        ----------
        copy : bool, optional
            Ignored (always returns new graph). For API compatibility.
            
        Returns
        -------
        FastGraph
            New graph with reversed edges
        """
        new_G = FastGraph()
        # A slim (attr-free) source graph must stay slim: rebuilding the
        # per-edge attr dicts here would reintroduce the ~350 bytes/edge
        # cost the flag exists to avoid.
        new_G.store_edge_attrs = self.store_edge_attrs
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(v, u, w)
                # Copy edge attributes
                if (u, v) in self.edge_attrs:
                    new_G.edge_attrs[(v, u)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes
        for n, attrs in self.node_attrs.items():
            new_G.node_attrs[n] = attrs.copy()
        return new_G

    def subgraph(self, nodes):
        """
        Return a subgraph containing only the specified nodes.
        
        Parameters
        ----------
        nodes : iterable
            Nodes to include in subgraph
            
        Returns
        -------
        FastGraph
            New graph with only specified nodes and their internal edges
        """
        nodes = set(nodes)
        new_G = FastGraph()
        new_G.store_edge_attrs = self.store_edge_attrs
        for u in nodes:
            if u in self.adj:
                for v, w in self.adj[u].items():
                    if v in nodes:
                        new_G.add_edge(u, v, w)
                        if (u, v) in self.edge_attrs:
                            new_G.edge_attrs[(u, v)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes for included nodes
        for n in nodes:
            if n in self.node_attrs:
                new_G.node_attrs[n] = self.node_attrs.get(n, {}).copy()
        return new_G

    def copy(self):
        """
        Return a deep copy of the graph.
        
        Returns
        -------
        FastGraph
            New graph with copied data
        """
        new_G = FastGraph()
        new_G.store_edge_attrs = self.store_edge_attrs
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(u, v, w)
        # Copy attributes
        for n, attrs in self.node_attrs.items():
            new_G.node_attrs[n] = attrs.copy()
        for edge, attrs in self.edge_attrs.items():
            new_G.edge_attrs[edge] = attrs.copy()
        return new_G

    def build_from_dataframe(self, df, source_col, target_col, weight_col, store_edge_attrs=True, **extra_cols):
        """
        Build graph from a DataFrame (modifies graph in place).

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            DataFrame with edge data
        source_col : str
            Column name for source nodes
        target_col : str
            Column name for target nodes
        weight_col : str
            Column name for edge weights
        store_edge_attrs : bool, optional
            When False, skip the per-edge edge_attrs dicts (weight remains
            available via adj).  Pathfinding-only graphs should pass False:
            the dicts cost ~350 bytes/edge.  Default True preserves the
            attribute-writing behavior for attribute consumers.
        **extra_cols : str
            Additional column names to store as edge attributes
            e.g., ratio_col='ratio', prob_col='probability'

        Returns
        -------
        self : FastGraph
            Returns self for method chaining
        """
        self.store_edge_attrs = store_edge_attrs
        # Check if it's a Polars DataFrame
        is_polars = False
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                is_polars = True
        except ImportError:
            pass

        if is_polars:
            if df.is_empty():
                return self
            # Convert to rows for iteration
            for row in df.iter_rows(named=True):
                u = row[source_col]
                v = row[target_col]
                w = row[weight_col]
                
                # Collect extra attributes
                attrs = {}
                for attr_name, col_name in extra_cols.items():
                    if col_name in row:
                        attrs[attr_name] = row[col_name]
                
                self.add_edge(u, v, w, **attrs)
        else:
            # Pandas implementation
            if df.empty:
                return self
            
            for _, row in df.iterrows():
                u = row[source_col]
                v = row[target_col]
                w = row[weight_col]
                
                # Collect extra attributes
                attrs = {}
                for attr_name, col_name in extra_cols.items():
                    if col_name in df.columns:
                        attrs[attr_name] = row[col_name]
                
                self.add_edge(u, v, w, **attrs)
        
        return self

    def to_dataframe(self, include_attrs=True):
        """
        Convert graph edges to a pandas DataFrame.
        
        Parameters
        ----------
        include_attrs : bool, optional
            If True, include all edge attributes. Default: True.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with columns [source, target, weight, ...]
        """
        import pandas as pd
        
        edges = []
        for u in self.adj:
            for v, w in self.adj[u].items():
                edge_data = {'source': u, 'target': v, 'weight': w}
                if include_attrs:
                    attrs = self.edge_attrs.get((u, v), {})
                    for k, val in attrs.items():
                        if k != 'weight':  # Already included
                            edge_data[k] = val
                edges.append(edge_data)
        return pd.DataFrame(edges)

    def filter_by_weight(self, min_weight):
        """
        Create a new graph with only edges meeting minimum weight threshold.
        
        Parameters
        ----------
        min_weight : float
            Minimum edge weight to include
            
        Returns
        -------
        FastGraph
            New graph with filtered edges
        """
        filtered = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                if w >= min_weight:
                    filtered.add_edge(u, v, w)
                    if (u, v) in self.edge_attrs:
                        filtered.edge_attrs[(u, v)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes for remaining nodes
        for n in filtered.adj:
            if n in self.node_attrs:
                filtered.node_attrs[n] = self.node_attrs[n].copy()
        return filtered

    # =========================================================================
    # Pathfinding Methods (for connectome analysis)
    # =========================================================================

    def aggregate_by_label(self, label_map: dict, return_edge_df: bool = False):
        """
        Aggregate graph nodes by label mapping (e.g., bodyId -> type).
        
        Creates a new graph where nodes are labels (e.g., neuron types) and
        edge weights are the sum of all original edges between nodes with
        those labels.
        
        Parameters
        ----------
        label_map : dict
            Dictionary mapping original node IDs to labels.
            e.g., {bodyId1: 'type_A', bodyId2: 'type_A', bodyId3: 'type_B'}
        return_edge_df : bool, optional
            If True, also return a DataFrame with edge details
                           
        Returns
        -------
        FastGraph
            New graph with aggregated edges by label
        pandas.DataFrame (optional)
            Edge data with columns [type_pre, type_post, weight]
        
        Example
        -------
        >>> G = FastGraph()
        >>> G.add_edge(1, 10, 5)  # bodyId 1 -> 10, weight 5
        >>> G.add_edge(2, 10, 3)  # bodyId 2 -> 10, weight 3
        >>> label_map = {1: 'A', 2: 'A', 10: 'X'}
        >>> G_type = G.aggregate_by_label(label_map)
        >>> # G_type now has edge A -> X with weight 8
        """
        aggregated = FastGraph()
        edge_data = []
        
        # Aggregate edges by label pairs
        label_weights = {}  # (label_pre, label_post) -> total_weight
        
        for u in self.adj:
            label_u = label_map.get(u)
            if label_u is None:
                continue  # Skip nodes without labels
                
            for v, w in self.adj[u].items():
                label_v = label_map.get(v)
                if label_v is None:
                    continue  # Skip nodes without labels
                
                key = (label_u, label_v)
                if key not in label_weights:
                    label_weights[key] = 0.0
                label_weights[key] += w
        
        # Build aggregated graph
        for (label_u, label_v), total_w in label_weights.items():
            aggregated.add_edge(label_u, label_v, total_w)
            if return_edge_df:
                edge_data.append({
                    'type_pre': label_u,
                    'type_post': label_v,
                    'weight': total_w
                })
        
        if return_edge_df:
            import pandas as pd
            return aggregated, pd.DataFrame(edge_data)
        return aggregated

    @classmethod
    def from_dataframe_bodyid(cls, df, pre_col='bodyId_pre', post_col='bodyId_post', 
                               weight_col='weight', type_pre_col='type_pre', 
                               type_post_col='type_post'):
        """
        Build a bodyId-level graph from DataFrame and extract type mapping.
        
        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            DataFrame with bodyId-level connection data
        pre_col : str
            Column name for presynaptic bodyId
        post_col : str
            Column name for postsynaptic bodyId
        weight_col : str
            Column name for edge weight
        type_pre_col : str
            Column name for presynaptic type
        type_post_col : str
            Column name for postsynaptic type
            
        Returns
        -------
        tuple
            (FastGraph, dict) - bodyId graph and bodyId->type mapping
        """
        import pandas as pd
        
        graph = cls()
        label_map = {}
        
        # Check if Polars
        is_polars = False
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                is_polars = True
        except ImportError:
            pass
        
        if is_polars:
            for row in df.iter_rows(named=True):
                u = row[pre_col]
                v = row[post_col]
                w = row[weight_col]
                graph.add_edge(u, v, w)
                
                # Build label map
                if type_pre_col in row and row[type_pre_col]:
                    label_map[u] = row[type_pre_col]
                if type_post_col in row and row[type_post_col]:
                    label_map[v] = row[type_post_col]
        else:
            for _, row in df.iterrows():
                u = row[pre_col]
                v = row[post_col]
                w = row[weight_col]
                graph.add_edge(u, v, w)
                
                # Build label map
                if type_pre_col in df.columns and pd.notna(row.get(type_pre_col)):
                    label_map[u] = row[type_pre_col]
                if type_post_col in df.columns and pd.notna(row.get(type_post_col)):
                    label_map[v] = row[type_post_col]
        
        return graph, label_map

    def all_simple_paths(self, source, target, cutoff):
        """
        Find all simple paths from source to target with length <= cutoff.
        
        Parameters
        ----------
        source : hashable
            Starting node
        target : hashable
            Ending node
        cutoff : int
            Maximum path length (number of edges)
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        if source not in self.adj:
            return
        
        yield from self._dfs_paths(source, target, cutoff, [source], {source})

    def _dfs_paths(self, u, target, cutoff, path, visited):
        """Internal DFS helper for all_simple_paths."""
        if len(path) > cutoff + 1:
            return
        
        if u == target:
            yield list(path)
            return
            
        if u not in self.adj:
            return

        for v in self.adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                yield from self._dfs_paths(v, target, cutoff, path, visited)
                path.pop()
                visited.remove(v)

    def find_paths_dfs_backtracking(self, sources, targets, cutoff, verbose=False):
        """
        Standard Backward DFS with backtracking (no memoization).
        Optimized for lowest memory usage at the cost of CPU time.
        Uses Iterative Deepening to find shortest paths first.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        source_set = set(sources)
        R = self.reverse()

        def dfs_recursive(u, target_depth, path, visited):
            current_len = len(path) - 1
            if current_len == target_depth:
                if u in source_set:
                    yield path[::-1]
                return

            if u in R.adj:
                for v in R.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        yield from dfs_recursive(v, target_depth, path, visited)
                        path.pop()
                        visited.remove(v)

        for length in range(1, cutoff + 1):
            iterator = targets
            if verbose:
                iterator = LineProgress(targets, desc=f"L{length} Backtracking", leave=False)
            
            for t in iterator:
                yield from dfs_recursive(t, length, [t], {t})

    def find_paths_meet_in_the_middle(self, sources, targets, cutoff, verbose=False):
        """
        Bidirectional DFS (Meet-in-the-middle) to find all paths.
        Optimized for memory by storing paths of length L/2 and streaming the rest.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        from collections import defaultdict
        
        source_set = set(sources)
        target_set = set(targets)
        
        def simple_dfs_paths(start_node, target_depth, edge_iter, valid_end_nodes=None):
            # In-place backtracking: the previous implementation copied the
            # whole path list (`path + [v]`) on every edge traversal, which is
            # O(depth) allocation per step and dominates runtime on dense
            # graphs. A shared path/visited pair with one copy at the leaf is
            # equivalent and much cheaper.
            path = [start_node]
            visited = {start_node}

            def dfs(u, depth):
                if depth == target_depth:
                    if valid_end_nodes is None or u in valid_end_nodes:
                        yield u, list(path)
                    return
                for v in edge_iter(u):
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        yield from dfs(v, depth + 1)
                        path.pop()
                        visited.discard(v)

            yield from dfs(start_node, 0)

        def get_reachable_set(start_nodes, depth):
            # Nodes reachable from `start_nodes` walking predecessors
            # (reverse of G) — no reversed-graph copy needed.
            current = set(start_nodes)
            for i in range(depth):
                next_layer = set()
                if not current: break
                for u in current:
                    next_layer.update(self.predecessors(u))
                current = next_layer
            return current

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            valid_mids = get_reachable_set(target_set, rem)
            if not valid_mids:
                continue
                
            fwd_paths_map = defaultdict(list)
            iterator = sources
            if verbose:
                iterator = LineProgress(sources, desc=f"L{length} Fwd(L{mid})", leave=False)
            
            for s in iterator:
                if s not in self.adj: continue
                for end_node, path in simple_dfs_paths(s, mid, lambda u: self.adj.get(u, ()), valid_mids):
                    fwd_paths_map[end_node].append(path)
                    
            if not fwd_paths_map:
                continue
                
            valid_ends_for_backward = set(fwd_paths_map.keys())
            iterator = targets
            if verbose:
                iterator = LineProgress(targets, desc=f"L{length} Bwd(L{rem})", leave=False)
                
            for t in iterator:
                if not list(self.predecessors(t)): continue
                for end_node, r_path in simple_dfs_paths(t, rem, lambda u: self.predecessors(u), valid_ends_for_backward):
                    r_path_rev = r_path[::-1]
                    r_set = set(r_path_rev)
                    for f_path in fwd_paths_map[end_node]:
                        if len(set(f_path) & r_set) == 1:
                            combined = f_path + r_path_rev[1:]
                            yield combined

    def find_paths_backward_dp(self, sources, targets, cutoff, verbose=False):
        """
        Backward Reachability (DP) + Forward DFS.
        Computes sets of nodes reachable from targets, then uses guided DFS.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        target_set = set(targets)
        
        valid_nodes_at_dist = [set() for _ in range(cutoff + 2)]
        valid_nodes_at_dist[0] = target_set
        
        for k in range(1, cutoff + 1):
            prev_set = valid_nodes_at_dist[k-1]
            current_set = valid_nodes_at_dist[k]
            # Predecessors of the previous layer = nodes that can reach a
            # target in exactly k steps (no reversed-graph copy needed).
            for target_node in prev_set:
                current_set.update(self.predecessors(target_node))
        
        def guided_dfs(u, depth, current_path):
            if depth == 0:
                if u in target_set:
                    yield list(current_path)
                return

            if u in self.adj:
                target_set_for_next_step = valid_nodes_at_dist[depth-1]
                for v in self.adj[u]:
                    if v in target_set_for_next_step:
                        if v not in current_path:
                            current_path.append(v)
                            yield from guided_dfs(v, depth - 1, current_path)
                            current_path.pop()

        for length in range(1, cutoff + 1):
            valid_sources = [s for s in sources if s in valid_nodes_at_dist[length]]
            iterator = valid_sources
            if verbose:
                iterator = LineProgress(valid_sources, desc=f"L{length} GuidedDFS", leave=False)
            for s in iterator:
                yield from guided_dfs(s, length, [s])

    def trim_to_strongest(self, keep, reserve_sources=None, reserve_targets=None,
                          reserve_edges=None, reserve_cap=None):
        """
        Keep only the ``keep`` strongest NON-reserved edges (by weight), in place.

        Source/target edges are reserved first and do NOT count toward the
        limit: every edge leaving a node in ``reserve_sources`` and every
        edge entering a node in ``reserve_targets`` (plus the explicit
        ``reserve_edges``) survives regardless of weight, and the strongest
        ``keep`` other edges are added on top — so path integrity is
        preserved (sources keep their outgoing edges, targets their incoming
        edges) and the kept graph can exceed ``keep`` edges.

        The auto-reservation is BOUNDED by ``reserve_cap`` (default: the
        limit itself): when the source/target nodes are so many that their
        incident edges alone would swallow the whole graph (e.g. type-level
        queries or edge-list visualizations where every node is classified
        as source/target), only the strongest ``reserve_cap`` source/target
        edges are reserved and the remaining candidates rejoin the
        non-reserved pool — so the trim always produces a bounded graph
        (at most ``reserve_cap + keep`` edges plus the explicit
        ``reserve_edges``). Edge attributes are preserved for kept edges;
        nodes not touched by any kept edge are removed (nodes that only
        receive kept edges stay discoverable). The reverse adjacency cache
        is invalidated.

        Parameters
        ----------
        keep : int or None
            Maximum number of NON-reserved edges to keep. None / <= 0 / >=
            the number of non-reserved edges => no trimming (returns
            (0, None)).
        reserve_sources : iterable, optional
            Source nodes: ALL their outgoing edges are reserved first (up to
            ``reserve_cap`` total).
        reserve_targets : iterable, optional
            Target nodes: ALL their incoming edges are reserved first (up to
            ``reserve_cap`` total).
        reserve_edges : iterable, optional
            Explicit (u, v) edges that are always kept (never capped).
        reserve_cap : int, optional
            Maximum number of auto-reserved source/target edges (default:
            ``keep`` — the graph stays bounded even for degenerate
            source/target classification).

        Returns
        -------
        tuple
            (removed_edges, threshold): the number of removed edges and the
            applied cutoff — the minimum weight among the kept NON-reserved
            edges (None when no trimming happened or nothing non-reserved
            was kept).
        """
        if keep is None or keep <= 0 or self._num_edges <= keep:
            return 0, None
        reserved = set(reserve_edges or ())
        candidates = []
        if reserve_sources:
            for u in set(reserve_sources):
                for v in self.adj.get(u, ()):
                    candidates.append((u, v))
        if reserve_targets:
            for t in set(reserve_targets):
                for pred in self.predecessors(t):
                    candidates.append((pred, t))
        cap = keep if reserve_cap is None else reserve_cap
        if candidates:
            if cap and cap > 0 and len(candidates) > cap:
                # degenerate classification: reserve only the strongest
                # candidates; the rest rejoin the non-reserved pool
                candidates.sort(key=lambda uv: self.adj[uv[0]][uv[1]], reverse=True)
                reserved.update(candidates[:cap])
                leftover = candidates[cap:]
            else:
                reserved.update(candidates)
                leftover = []
        else:
            leftover = []
        non_reserved = [
            (u, v) for u, neigh in self.adj.items() for v in neigh
            if (u, v) not in reserved
        ] + leftover
        if len(non_reserved) <= keep:
            return 0, None
        ranked = sorted(
            non_reserved,
            key=lambda uv: self.adj[uv[0]][uv[1]], reverse=True,
        )
        kept_non_reserved = ranked[:keep]
        kept_edges = reserved | set(kept_non_reserved)
        threshold = min(self.adj[u][v] for u, v in kept_non_reserved) if kept_non_reserved else None
        removed = self._apply_kept_edges(kept_edges)
        return removed, threshold

    def _apply_kept_edges(self, kept_edges):
        """Drop every edge not in ``kept_edges`` (in place); removes nodes
        that end up without any edge and invalidates the radj cache."""
        kept_nodes = {u for u, v in kept_edges} | {v for u, v in kept_edges}
        removed = 0
        for u in list(self.adj):
            for v in list(self.adj[u]):
                if (u, v) not in kept_edges:
                    del self.adj[u][v]
                    self.edge_attrs.pop((u, v), None)
                    self._num_edges -= 1
                    removed += 1
            if not self.adj[u] and u not in kept_nodes:
                del self.adj[u]
                self.node_attrs.pop(u, None)
        self._radj = None
        self._radj_dirty = False
        return removed

    def find_paths_memoized_dfs(self, sources, targets, cutoff, direction='forward', verbose=False):
        """
        Memoized DFS with valid successor pruning.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        direction : str
            'forward' or 'backward'
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        if direction == 'backward':
            # Walk predecessors instead of building a reversed graph copy;
            # paths are enumerated target -> source and reversed on the way out.
            for path in self._memoized_dfs_impl(
                targets, set(sources), cutoff,
                edge_iter=lambda u: self.predecessors(u),
                verbose=verbose,
            ):
                yield path[::-1]
            return
        yield from self._memoized_dfs_impl(
            sources, set(targets), cutoff,
            edge_iter=lambda u: self.adj.get(u, ()),
            verbose=verbose,
        )

    def _memoized_dfs_impl(self, starts, end_set, cutoff, edge_iter, verbose=False):
        """Shared memoized-DFS core; ``edge_iter(u)`` yields the neighbours
        to walk (successors for forward, predecessors for backward)."""
        valid_successors = {}
        
        def find_valid_successors(u, k):
            state = (u, k)
            if state in valid_successors:
                return valid_successors[state]
            if k == 0:
                return True if u in end_set else None
            successors = []
            for v in edge_iter(u):
                if find_valid_successors(v, k-1) is not None:
                    successors.append(v)
            res = successors if successors else None
            valid_successors[state] = res
            return res

        def reconstruct(u, k, path):
            if k == 0:
                if u in end_set:
                    yield list(path)
                return
            succs = valid_successors.get((u, k))
            if not succs: return
            for v in succs:
                if v not in path:
                    path.append(v)
                    yield from reconstruct(v, k-1, path)
                    path.pop()

        capped_sources = []
        total_paths = 0
        for length in range(1, cutoff + 1):
            iterator = starts
            if verbose:
                iterator = LineProgress(starts, desc=f"L{length} BuildMemo", leave=False)
            valid_starts = []
            for s in iterator:
                if find_valid_successors(s, length) is not None:
                    valid_starts.append(s)
            iterator = valid_starts
            if verbose:
                iterator = LineProgress(valid_starts, desc=f"L{length} Reconstruct", leave=False)
            for s in iterator:
                for p in reconstruct(s, length, [s]):
                    yield p
                    total_paths += 1
                if verbose:
                    # keep the running path total on the SAME bar (no
                    # separate counter line interleaving with the bar)
                    try:
                        iterator.set_postfix(paths=f"{total_paths:,}", refresh=True)
                    except Exception:
                        pass

    def find_paths_shortest(self, sources, targets, cutoff=None, verbose=False):
        """
        Enumerate ALL shortest (minimum hop-count) paths for every reachable
        (source, target) pair. Tied shortest paths are all yielded, each once.

        Mechanism
        ---------
        1. One backward BFS per target over the lazy reverse index ->
           dist[v] = hops from v to that target. Per-target distances (not a
           single nearest-target gradient) are required for correct per-pair
           results when sources and targets overlap: a shortest s->t route
           need not descend the nearest-target distance.
        2. Forward DFS from each source restricted to edges (u, v) with
           dist[v] == dist[u] - 1; a path ends exactly at the target (the
           only distance-0 node of its BFS). The strict distance descent
           makes cycles impossible, so no visited bookkeeping is needed.

        Complexity is O(T*(V + E)) preprocessing plus output size —
        polynomial, unlike the all-path algorithms (no branching^depth
        explosion). Consistent with the sibling algorithms, zero-hop paths
        (a source that is itself a target) are NOT yielded.

        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int or None
            Maximum path length (edges); paths longer than this are not
            yielded. None = unlimited.
        verbose : bool
            Show a single-line progress display

        Yields
        ------
        list
            Each shortest path as a list of nodes
        """
        from collections import deque

        target_set = set(t for t in targets if t in self.adj)
        if not target_set:
            return
        source_list = [s for s in dict.fromkeys(sources) if s in self.adj]
        if not source_list:
            return

        radj = self._ensure_radj()
        adj = self.adj

        target_iter = sorted(target_set, key=str)
        if verbose:
            target_iter = LineProgress(target_iter, desc='Shortest Distances', leave=False)

        total_paths = 0
        for t in target_iter:
            # Phase 1: backward BFS from this target
            dist = {t: 0}
            queue = deque([t])
            while queue:
                u = queue.popleft()
                d = dist[u] + 1
                for p in radj.get(u, ()):
                    if p not in dist:
                        dist[p] = d
                        queue.append(p)

            # Phase 2: walk the shortest-path DAG toward t. The walk is
            # ITERATIVE (explicit stack of (node, path, neighbor iterator)) —
            # the distance descent makes cycles impossible, so no visited
            # bookkeeping is needed, and very long shortest paths cannot hit
            # the Python recursion limit. Neighbor order matches the
            # recursive DFS (depth-first, adjacency order).
            def emit(start):
                stack = [(start, [start], iter(adj.get(start, ())))]
                while stack:
                    node, path, nbr_iter = stack[-1]
                    du = dist[node]
                    advanced = False
                    for v in nbr_iter:
                        dv = dist.get(v)
                        if dv is None or dv != du - 1:
                            continue
                        if v == t:
                            yield path + [v]
                        else:
                            stack.append((v, path + [v], iter(adj.get(v, ()))))
                        advanced = True
                        break
                    if not advanced:
                        stack.pop()

            for s in source_list:
                if s == t:
                    continue  # zero-hop paths excluded (sibling convention)
                ds = dist.get(s)
                if ds is None or (cutoff is not None and ds > cutoff):
                    continue
                for p in emit(s):
                    yield p
                    total_paths += 1
            if verbose:
                # keep the running path total on the SAME bar (no separate
                # counter line interleaving with the bar)
                try:
                    target_iter.set_postfix(paths=f"{total_paths:,}", refresh=True)
                except Exception:
                    pass

    def find_paths_shortest_backward(self, targets, sources, cutoff=None,
                                     verbose=False, target_cutoffs=None):
        """Enumerate shortest paths by walking backward from each target.

        This is the target-rooted counterpart of :meth:`find_paths_shortest`.
        The distance pass still runs over the lazy reverse index, but path
        reconstruction starts at the target and follows only reverse-DAG
        branches that contain one of the requested source nodes.  Branches
        that cannot end at a requested source are never expanded, which is
        important when a target has a large incoming fan-in but only a small
        subset of sources is in scope.

        Returned paths retain the normal source-to-target orientation.  A
        source that is itself a target is excluded at zero hops, matching the
        sibling shortest enumerator's convention.

        ``target_cutoffs`` optionally maps each target to its own maximum hop
        depth.  This is used by target-rooted discovery: a target first found
        at depth ``d`` must not later emit a source-target path longer than
        ``d`` merely because another source was found farther upstream.
        """
        from collections import deque

        target_set = set(t for t in targets if t in self.adj)
        source_list = [s for s in dict.fromkeys(sources) if s in self.adj]
        source_set = set(source_list)
        if not target_set or not source_set:
            return

        radj = self._ensure_radj()
        target_iter = sorted(target_set, key=str)
        if verbose:
            target_iter = LineProgress(target_iter,
                                       desc='Backward shortest paths',
                                       leave=False)

        total_paths = 0
        for target in target_iter:
            target_cutoff = cutoff
            if target_cutoffs is not None:
                configured = target_cutoffs.get(target)
                if configured is None:
                    configured = target_cutoffs.get(str(target))
                if configured is not None:
                    target_cutoff = max(0, int(configured))

            # Reverse BFS: dist[node] is the hop count from node to target.
            dist = {target: 0}
            queue = deque([target])
            while queue:
                node = queue.popleft()
                node_dist = dist[node]
                if target_cutoff is not None and node_dist >= target_cutoff:
                    continue
                next_dist = node_dist + 1
                for predecessor in radj.get(node, ()):
                    if predecessor not in dist:
                        dist[predecessor] = next_dist
                        queue.append(predecessor)

            # Mark only nodes on a shortest-DAG branch that reaches a
            # requested source.  Processing distances from farthest to
            # nearest avoids recursive traversal and keeps unrelated incoming
            # branches out of reconstruction entirely.
            valid_nodes = {
                source for source in source_set
                if source != target
                and source in dist
                and (target_cutoff is None
                     or dist[source] <= target_cutoff)
            }
            by_distance = sorted(dist.items(), key=lambda item: item[1],
                                 reverse=True)
            for node, node_dist in by_distance:
                if node in valid_nodes:
                    continue
                if any(
                    dist.get(predecessor) == node_dist + 1
                    and predecessor in valid_nodes
                    for predecessor in radj.get(node, ())
                ):
                    valid_nodes.add(node)

            if target not in valid_nodes:
                if verbose:
                    try:
                        target_iter.set_postfix(paths=f"{total_paths:,}",
                                                refresh=True)
                    except Exception:
                        pass
                continue

            # Iterative DFS on the reverse shortest-path DAG.  ``path_rev``
            # is target -> ... -> source and is reversed only when yielded.
            stack = [(target, [target], iter(radj.get(target, ()) ))]
            while stack:
                node, path_rev, predecessor_iter = stack[-1]
                node_dist = dist[node]
                advanced = False
                for predecessor in predecessor_iter:
                    if dist.get(predecessor) != node_dist + 1:
                        continue
                    if predecessor not in valid_nodes:
                        continue
                    candidate = path_rev + [predecessor]
                    if predecessor in source_set and predecessor != target:
                        yield list(reversed(candidate))
                        total_paths += 1
                    # A requested source can also be an intermediate node
                    # for another requested source.  Emit its own pair, but
                    # keep walking upstream so that source2 -> source1 ->
                    # target is not lost when both source IDs are enrolled.
                    stack.append(
                        (predecessor, candidate,
                         iter(radj.get(predecessor, ())))
                    )
                    advanced = True
                    break
                if not advanced:
                    stack.pop()

            if verbose:
                try:
                    target_iter.set_postfix(paths=f"{total_paths:,}",
                                            refresh=True)
                except Exception:
                    pass

    def find_paths_shortest_strongest_first(self, targets, sources, cutoff=None,
                                            budget=None, stats=None,
                                            target_cutoffs=None, verbose=False,
                                            target_distances=None,
                                            payload=False):
        """Budgeted best-first enumeration of MINIMUM-HOP paths (§7.2).

        For every target ``t``: one backward BFS gives ``dist_t`` (hops to
        t); the shortest-path DAG is the dist-descending subgraph, and a
        per-target maximin DP ``Wt[v]`` — the best bottleneck over DAG
        paths ``v -> t`` — makes best-first expansion emit that target's
        min-hop paths in DESCENDING-bottleneck order. The per-target
        streams are k-way merged, and the global ``budget`` applies with a
        tau tie-drain: a bitten run is exactly "all min-hop paths with
        bottleneck >= tau" (tau-equivalent to a complete shortest run at
        ``min_synapse_num = tau``). Unbitten runs return the same SET as
        ``find_paths_shortest_backward`` — only the order differs (global
        descending vs target-major). Zero-hop pairs (source == target) are
        excluded, matching the siblings.

        The DAG's strictly decreasing distances make cycles impossible, so
        no visited bookkeeping is needed inside a path.

        ``target_cutoffs`` optionally caps each target's hop depth (same
        contract as ``find_paths_shortest_backward``). ``target_distances``
        optionally seeds each target's distance map from an earlier
        discovery BFS (``{target: {node: hops-to-target}}``), skipping the
        per-target reverse BFS entirely — the maps must label the same
        graph the enumeration runs on (the DROCAT pipeline seeds them from
        its discovery stage). ``stats`` receives ``emitted``, ``tau``,
        ``budget_bitten``, ``strongest_dropped`` and ``per_target``
        ({target: emitted}).

        ``payload=True`` (batched-discovery composition mode) yields the
        raw ``(bottleneck, path, target)`` tuples of the merged stream in
        descending-bottleneck order WITHOUT applying the budget drain —
        ``budget`` must be None. ``merge_shortest_batch_emissions`` then
        merges several such payload lists (one per discovery batch) and
        applies the global drain once, reproducing the monolithic
        emission order (batches must be contiguous runs of
        ``sorted(targets, key=str)``).

        Memory note: the merge keeps every per-target stream (BFS dist +
        DAG maximin DP) alive simultaneously — state is bounded by
        ``hop_cap`` per target and is modest for the depth-explosion
        scenario this budget exists for (few targets, deep), but queries
        with MANY broad targets scale it linearly in the target count.

        Cost: per target O(V_t + E_t) for the BFS/DP plus best-first work
        proportional to the emitted prefix tree — no branching^depth
        explosion over non-shortest branches; with ``target_distances``
        the BFS is skipped and only the DP + emission remain.
        """
        import heapq
        from collections import deque

        INF = float('inf')
        target_set = set(t for t in targets if t in self.adj)
        source_list = [s for s in dict.fromkeys(sources) if s in self.adj]
        if not target_set or not source_list:
            return
        target_distances = target_distances or {}

        def _seed_for(target):
            if not target_distances:
                return None
            return (target_distances.get(target)
                    or target_distances.get(str(target)))

        # The reverse index feeds only the fresh per-target BFS; when every
        # target is seeded from discovery distances it is dead weight
        # (O(V + E) on the whole graph) — skip it.
        radj = None
        if any(_seed_for(target) is None for target in target_set):
            radj = self._ensure_radj()
        adj = self.adj

        def target_stream(target, hop_cap, seeded_dist=None):
            """All min-hop paths to ``target``, descending bottleneck."""
            # Backward BFS: dist[v] = min hops from v to target. A seeded
            # map (from the pipeline's discovery stage) replaces the BFS;
            # it is capped to the same hop bound the BFS would respect.
            if seeded_dist is not None:
                dist = {
                    node: d for node, d in seeded_dist.items()
                    if hop_cap is None or d <= hop_cap
                }
                if target not in dist:
                    dist[target] = 0
            else:
                dist = {target: 0}
                queue = deque([target])
                while queue:
                    node = queue.popleft()
                    node_dist = dist[node]
                    if hop_cap is not None and node_dist >= hop_cap:
                        continue
                    for predecessor in radj.get(node, ()):
                        if predecessor not in dist:
                            dist[predecessor] = node_dist + 1
                            queue.append(predecessor)
            # Per-target maximin DP over the shortest-path DAG (edges with
            # dist[v] == dist[u] - 1): Wt[v] = best bottleneck over DAG
            # paths v -> target. Nodes processed by INCREASING dist so a
            # node's DAG successors are settled first.
            by_dist = {}
            for node, d in dist.items():
                by_dist.setdefault(d, []).append(node)
            Wt = {target: INF}
            for d in sorted(by_dist):
                if d == 0:
                    continue
                for u in by_dist[d]:
                    best = None
                    for v, w in adj.get(u, {}).items():
                        if dist.get(v) != d - 1:
                            continue
                        tail = Wt.get(v)
                        if tail is None:
                            continue
                        cand = w if tail == INF else min(w, tail)
                        if best is None or cand > best:
                            best = cand
                    if best is not None:
                        Wt[u] = best
            # Best-first over DAG prefixes seeded from the sources: the pop
            # bound min(run_b, Wt[node]) is the exact best completion of
            # the prefix, so paths are emitted in descending bottleneck
            # order without any budget.
            heap = []
            counter = 0
            for s in source_list:
                if s == target:
                    continue
                ds = dist.get(s)
                if ds is None or (hop_cap is not None and ds > hop_cap):
                    continue
                bound = Wt.get(s)
                if bound is None:
                    continue
                heapq.heappush(heap, (-bound, counter, s, INF, (s,)))
                counter += 1
            while heap:
                neg_bound, _, node, run_b, path = heapq.heappop(heap)
                if node == target:
                    yield -neg_bound, list(path), target
                d = dist[node]
                if d == 0:
                    continue
                neighbors = sorted(adj.get(node, {}).items(),
                                   key=lambda kv: (-kv[1], kv[0]))
                for v, w in neighbors:
                    if dist.get(v) != d - 1:
                        continue
                    new_run = run_b if run_b < w else w
                    v_bound = Wt.get(v)
                    if v_bound is None:
                        continue
                    new_bound = new_run if new_run < v_bound else v_bound
                    heapq.heappush(heap, (-new_bound, counter, v, new_run,
                                          path + (v,)))
                    counter += 1

        streams = []
        for target in sorted(target_set, key=str):
            hop_cap = cutoff
            if target_cutoffs is not None:
                configured = target_cutoffs.get(target)
                if configured is None:
                    configured = target_cutoffs.get(str(target))
                if configured is not None:
                    hop_cap = max(0, int(configured))
            streams.append(target_stream(
                target, hop_cap, seeded_dist=_seed_for(target)))

        merged = heapq.merge(*streams, key=lambda item: -item[0])
        if payload:
            if budget is not None:
                raise ValueError(
                    'payload mode yields the undrained merged stream; '
                    'budget must be None (apply it once on the outer '
                    'merge via drain_shortest_merge / '
                    'merge_shortest_batch_emissions)')
            yield from merged
            return
        yield from drain_shortest_merge(merged, budget, stats)

    def find_paths_bidirectional_bfs(self, sources, targets, cutoff, verbose=False):
        """
        Bidirectional BFS (Layer-based) pathfinding.
        Builds search trees from both sides and finds intersection.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        from collections import defaultdict
        
        f_layers = [defaultdict(set) for _ in range(cutoff + 1)]
        for s in sources:
            if s in self.adj:
                f_layers[0][s].add(None)
        
        for d in range(cutoff):
            current_layer = f_layers[d]
            next_layer = f_layers[d+1]
            for u in current_layer:
                if u in self.adj:
                    for v in self.adj[u]:
                        next_layer[v].add(u)
            if not next_layer: break

        b_layers = [defaultdict(set) for _ in range(cutoff + 1)]
        target_set = set(targets)
        radj = self._ensure_radj()  # predecessor index (reverse of G)
        
        for t in targets:
            if t in radj:
                b_layers[0][t].add(None)
                
        for d in range(cutoff):
            current_layer = b_layers[d]
            next_layer = b_layers[d+1]
            for u in current_layer:
                if u in radj:
                    for v in radj[u]:
                        next_layer[v].add(u)
            if not next_layer: break

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            f_nodes = set(f_layers[mid].keys())
            b_nodes = set(b_layers[rem].keys())
            meet_nodes = f_nodes & b_nodes
            
            if not meet_nodes: continue
            
            def get_fwd_paths(u, depth):
                if depth == 0:
                    yield [u]
                    return
                for parent in f_layers[depth][u]:
                    if parent is not None:
                        for p in get_fwd_paths(parent, depth-1):
                            if u not in p:
                                yield p + [u]

            def get_bwd_paths(u, depth):
                if depth == 0:
                    yield [u]
                    return
                for child in b_layers[depth][u]:
                    if child is not None:
                        for p in get_bwd_paths(child, depth-1):
                            if u not in p:
                                yield [u] + p

            iterator = meet_nodes
            if verbose:
                iterator = LineProgress(meet_nodes, desc=f"L{length} Reconstruct", leave=False)

            for u in iterator:
                f_paths = list(get_fwd_paths(u, mid))
                b_paths = list(get_bwd_paths(u, rem))
                for fp in f_paths:
                    for bp in b_paths:
                        if len(set(fp) & set(bp)) == 1:
                            yield fp + bp[1:]

    def _widest_path_backward(self, targets, cutoff):
        """Best achievable bottleneck from every node to any target.

        Delegates to the shared strongest-selection core (§6c of
        plan-cross-dataset-report-fixes). ``W[d][v]`` = max over paths
        ``v -> any target`` with at most ``d`` edges of the min edge
        weight (widest-path / maximin value).
        """
        try:
            from .strongest_core import widest_path_backward
        except ImportError:  # src laid bare on sys.path
            from strongest_core import widest_path_backward
        return widest_path_backward(self.adj, targets, int(cutoff))

    def find_paths_strongest_first(self, sources, targets, cutoff,
                                   budget=None, per_pair_k=None, stats=None,
                                   verbose=False):
        """Budgeted best-first enumeration in DESCENDING bottleneck order.

        Emits complete simple paths (source -> ... -> target, at most
        ``cutoff`` edges, no repeated nodes) strongest-first, where the
        strength of a path is its BOTTLENECK (minimum edge weight).

        Semantics:
        - With ``budget=None`` (or a budget >= the total path count) the
          emitted SET is identical to ``find_paths_memoized_dfs`` (any
          simple path of 1..cutoff edges ending at a target; paths may
          pass through other targets). Only the emission order differs.
        - With a budget, emission continues until ``budget`` paths have
          been emitted and then drains every remaining tie at the cutoff
          strength tau (the last emitted bottleneck). The result is
          exactly "all intact paths with bottleneck >= tau"; ``stats``
          reports ``tau``, ``emitted`` and ``budget_bitten``.
        - ``per_pair_k`` (optional) additionally skips emitting paths for
          a (source, target) pair once it has emitted k paths (fair share;
          pairs keep being extended). With per_pair_k set, the result is
          "per-pair top-k", NOT a global bottleneck cutoff, and the
          threshold-monotone prefix property no longer holds.

        Determinism: neighbors are walked in (-weight, node) order and a
        monotonic counter breaks priority ties, so repeated runs on the
        same graph emit the same sequence.
        """
        import heapq

        INF = float('inf')
        cutoff = int(cutoff)
        if cutoff < 1:
            return
        target_set = set(targets)
        W = self._widest_path_backward(target_set, cutoff)

        heap = []  # (-bound, tie, node, hops, run_bottleneck, path_tuple)
        counter = 0
        for s in sources:
            bound = W[cutoff].get(s)
            if bound is None:
                continue
            heapq.heappush(heap, (-bound, counter, s, 0, INF, (s,)))
            counter += 1

        emitted = 0
        tau = None
        tau_open = False  # True once the budget is reached: drain ties at tau
        truncated = False  # True when weaker paths were skipped (tau < full)
        pair_counts = {}
        last_pop_bound = None
        # Canonical-tau support: the strongest DROPPED path's bottleneck.
        # Paths below tau are dropped either by the drain break (this pop's
        # bound is their best completion) or by the push filter (new_bound
        # is the prefix's best completion); the max over both is the exact
        # w2, so the canonical threshold tau_min = w2 + 1 is the minimal
        # query that reproduces this output set.
        strongest_dropped = None

        while heap:
            neg_bound, _, node, hops, run_b, path = heapq.heappop(heap)
            bound = -neg_bound
            last_pop_bound = bound
            if tau_open and bound < tau:
                truncated = True
                if strongest_dropped is None or bound > strongest_dropped:
                    strongest_dropped = bound
                break  # everything left is strictly weaker than tau

            # A length-0 path (a source that is itself a target) is not part
            # of the enumerated universe (matching find_paths_memoized_dfs).
            is_target = node in target_set and hops >= 1
            if is_target:
                if per_pair_k is not None:
                    pair = (path[0], node)
                    if pair_counts.get(pair, 0) >= per_pair_k:
                        is_target = False  # fair share reached: no emission
                    else:
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
                if is_target:
                    yield list(path)
                    emitted += 1
                    tau = run_b
                    if budget is not None and emitted >= budget:
                        tau_open = True

            if hops >= cutoff:
                continue
            remaining = cutoff - hops - 1
            # Deterministic expansion: strongest edge first, then node id.
            neighbors = sorted(self.adj.get(node, {}).items(),
                               key=lambda kv: (-kv[1], kv[0]))
            path_set = set(path)
            for v, w in neighbors:
                if v in path_set:
                    continue
                new_run = run_b if run_b < w else w
                v_bound = W[remaining].get(v)
                if v_bound is None:
                    continue
                new_bound = new_run if new_run < v_bound else v_bound
                if tau_open and new_bound < tau:
                    truncated = True
                    if (strongest_dropped is None
                            or new_bound > strongest_dropped):
                        strongest_dropped = new_bound
                    continue  # weaker than the cutoff being drained
                heapq.heappush(
                    heap,
                    (-new_bound, counter, v, hops + 1, new_run,
                     path + (v,)),
                )
                counter += 1

        if stats is not None:
            stats['emitted'] = emitted
            stats['budget_bitten'] = bool(budget is not None and truncated)
            stats['tau'] = tau
            stats['last_bound'] = last_pop_bound
            stats['strongest_dropped'] = strongest_dropped

    def __repr__(self):
        return f"FastGraph(nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"


# Compatibility alias


class _NodeView:
    """
    A NetworkX-compatible node view that supports both iteration and dict-like access.
    """
    def __init__(self, graph):
        self._graph = graph
    
    def __iter__(self):
        return iter(self._graph.adj)
    
    def __call__(self, data=False):
        """Called when G.nodes() is used."""
        if data:
            for n in self._graph.adj:
                yield (n, self._graph.node_attrs.get(n, {}))
        else:
            yield from self._graph.adj
    
    def __getitem__(self, node):
        """Allow G.nodes[node] access to node attributes."""
        if node not in self._graph.node_attrs:
            self._graph.node_attrs[node] = {}
        return self._graph.node_attrs[node]
    
    def __setitem__(self, node, attrs):
        """Allow G.nodes[node] = {...} assignment."""
        self._graph.node_attrs[node] = attrs
    
    def __contains__(self, node):
        return node in self._graph.adj
    
    def __len__(self):
        return len(self._graph.adj)
    
    def keys(self):
        return self._graph.adj.keys()
    
    def items(self):
        for n in self._graph.adj:
            yield (n, self._graph.node_attrs.get(n, {}))
    
    def update(self, attrs_dict):
        """Update node attributes from a dict."""
        for node, attrs in attrs_dict.items():
            if node not in self._graph.node_attrs:
                self._graph.node_attrs[node] = {}
            self._graph.node_attrs[node].update(attrs)




DiGraph = FastGraph
