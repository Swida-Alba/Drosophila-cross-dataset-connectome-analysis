# Pathfinding Methods in FindAllPath

The `FindAllPath` function in `coana.py` runs on the optimized `FastGraph`
core. Since 2026-09-05, **StrongestFirst is the built-in 'all'-mode
algorithm** — the Algorithm selector was removed from the Find All Paths
and Cross-Dataset tabs. The `pathfinding` parameter survives for
scripts/API callers (legacy complete enumerators are documented below
and remain verified), and `max_paths_bodyid` bounds the output with a
reported τ while `graph_edge_limit_bodyid` is the **Edge Budget** that
floors the discovery cone.

> 📖 **For the end-to-end technical picture** — every pipeline stage,
> guarantee, complexity figure, and measured result — see
> [PATHFINDING_PIPELINE.md](../technical/PATHFINDING_PIPELINE.md).

The complete enumerators are also evaluated theoretically and by measured time/memory in [PATHFINDING_ALGORITHM_EVALUATION.md](../technical/PATHFINDING_ALGORITHM_EVALUATION.md); the benchmark harness has been archived to `archive/examples/performance/benchmark_pathfinding.py`.

**Why pathfinding runs at the bodyId level first:** FindAllPath discovers
all paths on the bodyId-level graph and only then aggregates the in-path
pairs into type-level summaries. Aggregating *all* per-bodyId pairs into a
type-level graph and pathfinding on that alone would hide differences
within a type (the *bundle effect*) and report phantom type paths with no
real neuron chain behind them. See
[TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md](../technical/TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md)
for the rationale and the comparative measurements.

## Available Algorithms

### 0. StrongestFirst (budgeted) — the default since 2026-09-04
**Parameter:** `pathfinding='StrongestFirst'`
**Method:** `FastGraph.find_paths_strongest_first`

Best-first search ordered by each path's **bottleneck** (its weakest
edge), with a widest-path backward DP as the admissible suffix bound.
Emits complete intact paths **strongest-first** under a path budget
(`max_paths_bodyid`, default 1,000,000):

*   **Mechanism**:
    *   Precomputes `W[d][v]` — the best achievable bottleneck from node
        `v` to any target within `d` edges.
    *   Expands path prefixes best-first on `min(running bottleneck,
        W[remaining][node])`, so complete paths are emitted in
        non-increasing bottleneck order.
    *   When the budget is reached, the search drains all ties at the
        cutoff strength **tau** and stops: the output contains exactly
        "all intact paths with bottleneck >= tau". Tau is reported in the
        run log and `user_warning_notes.txt`.
*   **Guarantee**: without a budget bite the emitted SET is identical to
    the complete enumerators below (only the order differs). When the
    budget bites, the cutoff is a well-defined strength threshold — not
    an arbitrary truncation.
*   **Best Use Case**: connectome queries where full enumeration
    explodes (millions of paths); you keep the strongest structure
    within a bounded memory/time envelope.
*   **Deterministic**: same graph, same emission sequence.
*   **Sole UI algorithm (2026-09-05)**: the Algorithm selector was
    removed; every UI run uses StrongestFirst. `0 = auto` → internal 1M
    budget. Complete runs report their **natural tau** (the weakest
    emitted path's bottleneck — "every threshold up to this value yields
    this identical set") alongside `paths_complete=true`; budget-bitten
    runs report both the budget and the emitted count (the tie-drain may
    overshoot the budget).
*   **Unification (Fix C, 2026-09-04)**: a positive `max_paths_bodyid`
    routes to StrongestFirst regardless of the Algorithm selector; the
    lossy bodyId edge trim no longer runs in
    FindAllPath/FindShortestPath. The single bounding mechanism is the path
    budget with its τ report — a budgeted run at threshold t is
    *equivalent to a complete run at min_synapse_num = τ*.
*   **Edge Budget (Fix D, 2026-09-05; budget-fit refined)**:
    `graph_edge_limit_bodyid` (default 1M, 0 = off) caps the cone itself:
    after the lossless prunes, the **budget-fit search** finds the weakest
    weight tier whose lossless-closed cone fits the cap and floors the
    cone there — a pure threshold raise (effective cutoff =
    max(τ_budget, t*)), reported as `edge_weight_floor` /
    `edge_budget_landing` with the residual slack and probe trace. This
    replaces the one-shot `w0 = (N-th strongest edge weight) + 1`
    landing, which could waste most of the cap on boundary-tie mass.
    Shortest mode is never floored.
*   Before enumeration, **lossless hop-budget pruning passes**
    (`prune_layers_hop_budget`, iterated to a fixpoint) remove every
    discovery edge that cannot lie on any source→target path within the
    layer bound — at every depth, in both 'all' and 'shortest' modes. No
    admissible path is lost (each kept-path set is unchanged); the
    removed-row count is logged and noted as lossless in
    `user_warning_notes.txt`.
*   **Ratio/probability filters retired (F9, 2026-09-05)**:
    `connection_ratio` (weight / ALL-POST incoming weight — the
    denominator is now threshold-free) and `traversal_probability`
    (ratio/0.3, capped) are readout columns only; they no longer filter
    the graph. This restores cone nesting under any future ratio
    predicate and makes ratios comparable across thresholds.

### Threshold and bottleneck provenance

Every Complete Paths and Shortest Paths run writes a provenance block to
`parameters.txt`, `all_attributes.json`, and `data_details/parameters.csv`.
`requested_threshold` is the entered Min Synapse Count; `applied_threshold`
is the canonical equivalent threshold for the materialized path set. It is
the requested value for a complete run, `w2 + 1` when the StrongestFirst path
budget bites, and includes the Edge Budget floor when that graph budget fires.
`applied_threshold_source` identifies the mechanism (`requested`,
`strongest_first_budget`, `edge_budget`, or both).

The block also reports `strongest_first_budget` and its bite flag, `tau` /
`strongest_first_tau` (the StrongestFirst landing or natural weakest emitted
bottleneck), `tau_canonical`, `w2` /
`strongest_dropped_bottleneck`, `w0` / `edge_weight_floor`, `w1` /
`edge_budget_landing`, `W*` / `strongest_retained_bottleneck`, and
`paths_complete`. A path bottleneck is the minimum edge weight along that
path. Shortest mode may be bounded by the StrongestFirst path budget but is
never floored by the Edge Budget.

The **Drop Untyped Neurons** option is checked by default in both pathfinding
tabs. It removes edges touching empty, Unknown/None/NaN, or numeric
bodyId-fallback labels after enrichment and before graph construction, and
writes dropped rows to `data_details/untyped_dropped_records.csv` when there
are any.

## Complete enumerators (API / benchmark reference)

The enumerators below are **no longer selectable in the UI**
(StrongestFirst is the pipeline); scripts can still request them via
`pathfinding=...` with `max_paths_bodyid=0` for unbounded complete
enumeration. They all return the identical path set — which is exactly
what a τ-clamped StrongestFirst run equals at its reported tau.

### 1. Bidirectional Search (Layer Intersection)
**Parameter:** `pathfinding='Bidirectional'`
**Method:** `FastGraph.find_paths_bidirectional_bfs`

This algorithm performs a simultaneous Breadth-First Search (BFS) from both the source and target sets.

*   **Mechanism**: 
    *   Expands "layers" of nodes from sources (Forward) and targets (Backward).
    *   Finds the intersection of these layers at the midpoint (e.g., for length 4, intersects Forward Layer 2 and Backward Layer 2).
    *   Reconstructs paths by backtracking from the intersection nodes.
*   **Best Use Case**: Shallow graphs where shortest-first *emission order*
    is wanted. It is **not** what the Shortest Paths tab runs — that tab
    uses the target-rooted StrongestFirst enumerator
    `find_paths_shortest_strongest_first` (per-target reverse BFS,
    maximin bottleneck DP, best-first strength-ordered emission, k-way
    merge across targets), not Bidirectional, and not the
    `find_paths_shortest_backward` exporter (which exists in FastGraph but
    the pipeline does not call it).
*   **Pros**: Emits shortest paths first (an ordering property — the
    returned path set is identical to every other complete enumerator).
*   **Cons**: High memory usage for dense graphs as it stores full layers (measured 452 MB at 4 layers, 1.1 GB at 5).

### 2. Backward Reachability (DP)
**Parameter:** `pathfinding='DP'`
**Method:** `FastGraph.find_paths_backward_dp`

A hybrid approach combining Backward BFS for pruning and Forward DFS for path construction. **Reference complete enumerator** (fastest measured at 2–3 intermediate layers; StrongestFirst is the pipeline default).

*   **Mechanism**:
    *   **Phase 1 (Backward Reachability)**: Computes sets R_k containing all nodes that can reach a target in exactly k steps.
    *   **Phase 2 (Guided DFS)**: Runs a forward DFS from sources, but only visits a neighbor v if v ∈ R_{k-1}.
*   **Best Use Case**: Sparse graphs or queries where many branches lead to dead ends.
*   **Pros**: **Lowest memory footprint**. Aggressively prunes dead ends before the main search.
*   **Cons**: Requires two passes over the graph; degenerates on very deep (5+) queries.

### 3. Memoized DFS (forward) — reference complete enumerator
**Parameter:** `pathfinding='MemoizedDFS'`
**Method:** `FastGraph.find_paths_memoized_dfs` (direction='forward')

A depth-first traversal from the sources that memoizes which successors
can still reach a target at each remaining depth, so every (node, depth)
state is computed once and reconstruction walks only valid successors.

*   **Mechanism**:
    *   Recursively explores paths from sources to targets.
    *   Memoizes valid successors per (node, remaining-depth); dead-end branches are never walked.
*   **Best Use Case**: the common case — fastest measured at **every** depth on real connectomes (no reversed-graph copy; the memo stays inside the source-side cone).
*   **Pros**: Fastest measured; smallest peak allocation at 2–4 layers.
*   **Cons**: Can use significant memory for memoization tables on very large source cones.

### 4. Memoized DFS (backward)
**Parameter:** `pathfinding='DFS'`
**Method:** `FastGraph.find_paths_memoized_dfs` (direction='backward')

The same memoized DFS started from the targets on the reversed graph.

*   **Mechanism**:
    *   Recursively explores paths from targets to sources.
    *   Memoizes which nodes can reach the source within k steps.
*   **Best Use Case**: When targets are fewer than sources.
*   **Pros**: Fast at 4–5 layers with few targets (measured 0.99–1.03 s at 4 layers, fastest at 5).
*   **Cons**: Builds a reversed-graph copy (~250 MB on a 722k-edge graph) and explores the target-side cone.

### 5. Meet-in-the-middle DFS
**Parameter:** `pathfinding='MeetInMiddle'`
**Method:** `FastGraph.find_paths_meet_in_the_middle`

*   **Mechanism**: 
    *   **Forward Phase**: Performs DFS from sources for ⌊L/2⌋ steps. Stores these half-paths in a memory-efficient hash map.
    *   **Backward Phase**: Performs DFS from targets for the remaining steps.
    *   **Join**: When the backward search hits a node existing in the forward map, it stitches the paths together.
*   **Best Use Case**: Fastest at shallow depths (2–3 intermediate layers, within noise of MemoizedDFS).
*   **Pros**: Half-depth exponential in theory.
*   **Cons**: The half-path table is rebuilt for every length, the forward walk is unpruned and the pairwise stitching costs more than a fully-pruned shared memo on real connectomes (measured 2.17 s at 4 layers, 5.88 s at 5).

## Comparison Summary

| Algorithm | Parameter | Underlying Method | Best For | Time (worst) | Memory (worst) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StrongestFirst** | `StrongestFirst` | `find_paths_strongest_first` | **The pipeline default** | O(L·E + explored·log H + P·L) | O(L·V + H·L) |
| **Memoized DFS (fwd)** | `MemoizedDFS` | `find_paths_memoized_dfs` | **Complete runs (API); all depths** | O(L·E + P·L) | O(L·E) |
| **Memoized DFS (bwd)** | `DFS` | `find_paths_memoized_dfs` (backward) | **Deep paths, few targets** | O(L·E + P·L) | O(L·E) |
| **Meet-in-the-middle** | `MeetInMiddle` | `find_paths_meet_in_the_middle` | Shallow; safe mid-ground | O(b^{L/2}·L + P·L) | O(b^{L/2}·L) |
| **Backward DP** | `DP` | `find_paths_backward_dp` | Robust, no reverse copy | O(L·E + P·L) | O(L·V) |
| **Bidirectional BFS** | `Bidirectional` | `find_paths_bidirectional_bfs` | Shortest-first order (API only; the Shortest Paths tab uses target-rooted StrongestFirst) | O(L·E + P·L) | **O(L·(V+E))** |

*L = cutoff (max path length), E = edges, V = nodes, b = branching factor, P = number of found paths.*

**Shortest Paths tab (FindShortestPath):** the tab enumerates through
`find_paths_shortest_strongest_first` — target-rooted StrongestFirst with
a maximin bottleneck DP over the per-target reverse BFS DAG and
best-first strength-ordered emission merged across targets. Its output can
be tau-bounded by `max_paths_bodyid`, and the Edge Budget never applies
(shortest mode is never floored).

## Measured Evaluation (2026-08)

Benchmarked on real hemibrain v1.2.1 queries (connections ≥ 5 synapses) —
all algorithms returned **identical path sets** in every scenario. The
"(benchmarked default)" tag below is what this table measured in 2026-08; the
pipeline's default has since moved to **StrongestFirst** (line 4, and
`coana.py:3088`), which is why the complete enumerators here stay documented
rather than prescribed:

- **2–3 intermediate layers**: LC → MBON/PPL, 10,331 nodes, 301,443 edges, 4 sources × 5 targets (14 / 2,874 paths)
- **4–5 intermediate layers**: LPLC1 → MBON01, 24,576 nodes, 722,143 edges, 2 sources × 2 targets (35,819 paths; 5M+ paths at 5 layers, capped)

| Algorithm | 2 layers | 3 layers | 4 layers | 5 layers |
| :--- | ---: | ---: | ---: | ---: |
| **MemoizedDFS** (benchmarked default) | 0.01 s / 0.1 MB | 0.06 s / 1.7 MB | **0.23 s** / 7.6 MB | 4.97 s / 651 MB |
| **DFS** (backward) | 0.02 s / 0.3 MB | 0.05 s / 1.8 MB | 0.42 s / 9.6 MB | 5.01 s / 612 MB |
| **MeetInMiddle** | **0.00 s** / 0.3 MB | **0.01 s** / 0.6 MB | 1.25 s / 4.8 MB | 6.18 s / 615 MB |
| **DP** | 0.06 s / 0.8 MB | 0.02 s / 1.5 MB | 0.44 s / 9.4 MB | 20.2 s / 651 MB |
| **Bidirectional** | 0.07 s / 18.9 MB† | 0.18 s / 59.4 MB† | 0.64 s / 194.9 MB† | **4.54 s** / 892 MB† |

*Fastest time per column in **bold**; † = highest memory in that column.
**2026-08 optimization:** FastGraph keeps a lazy reverse adjacency index
instead of copying the whole reversed graph per run — the earlier
100–450 MB footprints were mostly those copies and are gone (DP dropped
from 272 MB to 9.4 MB at 4 layers). At 5 layers every value includes
~610 MB of the 5 million collected paths.*

Key findings:

- **MemoizedDFS (forward) is fastest at 4 layers** (0.23 s) and within
  noise elsewhere; all algorithms except Bidirectional now run in
  <10 MB at 2–4 layers thanks to the lazy reverse-adjacency index.
- **MeetInMiddle is fastest at 2–3 layers** (0.00–0.01 s).
- **DFS (backward) is the best alternative for few targets**; it no longer
  pays for a reversed-graph copy (0.3–9.6 MB at 2–4 layers).
- **Bidirectional** remains the memory leader (18.9 → 892 MB) — its full
  layer trees dominate now that the other algorithms no longer copy the
  graph.
- **DP** degenerates on deep queries (20.2 s at 5 layers).

**Recommendation (2026-08, superseded 2026-09-05)**: `MemoizedDFS` was
the fastest complete enumerator; StrongestFirst is now the pipeline
algorithm (bounded output with a reported τ), with MemoizedDFS kept for
unbounded complete runs via the API. Use `DFS` for deep paths with few
targets, `MeetInMiddle` for shallow queries, and `Bidirectional` only for
shortest-first *ordering* with memory to spare — minimum-hop semantics in
the UI come from the Shortest Paths tab's target-rooted StrongestFirst
enumerator, not from Bidirectional.

