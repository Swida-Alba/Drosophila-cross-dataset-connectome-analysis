# Pathfinding Algorithm Evaluation

Evaluation of the complete-enumerator `FastGraph` pathfinding algorithms used by DROCAT
(Find All Paths, Find Path, Cross-Dataset Comparison), both theoretically
(complexity analysis of the implementations in
`vispath-subproject/src/vispath_pkg/fast_graph_core.py`) and practically
(measured wall time + peak memory on synthetic graphs and real
hemibrain v1.2.1 connectome queries).

Benchmark harness: `examples/performance/benchmark_pathfinding.py`
(raw numbers: `cache/benchmark_pathfinding_results.json`).
Machine: macOS, Python 3.11 (drocat conda env).

## 1. What the algorithms have in common

All complete enumerators solve the **all-simple-paths** problem: enumerate every
acyclic path from any source to any target with length ≤ L (the cutoff).
The number of paths P is exponential in L in the worst case (branching
factor b → O(b^L) paths), so *no* algorithm can be faster than O(P·L) —
it must at least write out every path. The algorithms therefore differ in:

- how much **pre-computation / pruning** they do before or during the DFS,
- how much **auxiliary memory** they trade for that pruning,
- whether paths are produced **shortest-first** (useful for early
  termination).

Notation: V = nodes, E = edges (of the search-space graph), S/T = number of
sources/targets, L = cutoff, P = number of output paths, b = effective
branching factor.

## 2. Theoretical analysis

### Bidirectional BFS — UI name `Bidirectional`
`find_paths_bidirectional_bfs`

Builds complete layer maps (node → set of parents) from the source side AND
from the target side (on the reversed graph) for every depth 0..L, then for
each length finds the meeting layer and stitches forward/backward
half-paths.

- **Time**: O(L·E) to build both search trees + O(P·L) to reconstruct.
- **Memory**: O(L·(V+E)) — the parent-pointer layer maps. **Highest of the
  four**; both trees are fully stored, so memory grows linearly with the
  cutoff and with the frontier size.
- **Strengths**: paths are emitted shortest-first; a path of length ℓ only
  needs layers up to ℓ, so a shallow query never expands deeper than
  necessary.
- **Weaknesses**: stores *every* node of every layer of *both* trees — on
  dense connectomes the layer-3/4 frontiers are millions of entries.

### Backward Reachability + Guided DFS — UI name `DP`
`find_paths_backward_dp`

Phase 1 computes the node sets R_k = {nodes that can reach a target in
exactly k steps} for k = 0..L by iterating the reversed graph. Phase 2 runs
a forward DFS from the sources that only follows edges into R_{k-1}, i.e.
only edges that provably lie on some source→target path.

- **Time**: O(L·E) for the reachability pass + O(P·L) for reconstruction.
  Dead-end branches are never explored, so the constant factor of the DFS
  is small.
- **Memory**: O(L·V) — only node *sets* per depth, no edge lists, no
  parent trees. **Lowest of the pruning algorithms**.
- **Strengths**: best all-round on dense graphs with many dead ends (the
  normal connectome case); the pre-computed layers also make the DFS
  branch-free per step (set membership test).
- **Weaknesses**: two passes; reachability sets are built for the full
  cutoff even if the query is sparse.

### Backward Memoized DFS — UI name `DFS`
`find_paths_memoized_dfs` (direction='backward')

Memoizes valid_successors[(u, k)] = the successors of u that can still
reach a target in k−1 steps, so each (node, remaining-depth) state is
computed once; reconstruction then enumerates paths through the memo.

- **Time**: O(L·E) worst case for the memo build + O(P·L) reconstruction.
- **Memory**: O(L·E) worst case — the memo stores a successor *list* per
  (node, depth), more than DP's sets.
- **Strengths**: when |targets| ≪ |sources|, running backward from the
  smaller set shrinks the reachable space (memo states are computed only
  for nodes that can reach a target).
- **Weaknesses**: memo lists can be large; no shortest-first ordering.

### Meet-in-the-Middle DFS — UI name `MemoizedDFS` *(pre-naming-fix label; see the audit's naming-fix table — `MemoizedDFS` now routes to the forward memoized DFS)*
`find_paths_meet_in_the_middle`

For each length ℓ: mid = ⌊ℓ/2⌋. Forward DFS from the sources to depth mid,
storing **half-paths** in a map (end node → list of paths); backward DFS
from the targets for the remaining ℓ−mid steps; stitch pairs that meet at a
single node.

- **Time**: O(b^{ℓ/2}·L) per length for the two half-searches + O(P·L) to
  stitch (each pair is validated with a set intersection).
- **Memory**: O(b^{ℓ/2}·L) — only the *forward half-path table* is stored.
- **Strengths**: the exponential is on the **half-depth** frontier, so in
  theory deep paths cost far less than a full-depth search; measured
  fastest at 2–3 intermediate layers.
- **Weaknesses**: the forward half-path table is **rebuilt for every
  length**, the forward DFS walks every branch without target-guided
  pruning (endpoints are only checked at the leaves), and stitching scans
  all forward half-paths sharing an endpoint with a `set()` construction
  per candidate pair — see §3.3 for why these costs outweigh the
  half-depth gain on real connectomes.

### Why meet-in-the-middle does not beat backward-memoized DFS on real connectomes

Measured (hemibrain, cutoffs 2–5): DFS 0.99 s vs MemoizedDFS 2.29 s at
4 layers, 5.29 s vs 6.21 s at 5 layers. The implementation-level reasons:

1. **Per-length rebuild**: `find_paths_meet_in_the_middle` rebuilds
   `fwd_paths_map` from scratch inside `for length in range(1, cutoff+1)`,
   so half-paths of depth 2 are re-enumerated for lengths 4, 5 and 6.
   `find_paths_memoized_dfs` computes the `valid_successors[(u, k)]` memo
   **once** and shares it across every length.
2. **No pruning on the forward side**: the forward DFS walks *all* simple
   paths to depth mid, only checking at the leaf whether the endpoint is a
   valid mid node. The backward-memoized DFS prunes during the memo build
   (a successor is stored only if it can reach a target in k−1 steps), so
   dead-end branches are never walked.
3. **Pairwise stitching**: for every backward half-path,
   `for f_path in fwd_paths_map[end_node]: if len(set(f_path) & r_set) == 1`
   builds a `set()` per candidate pair and scans all forward half-paths
   sharing the endpoint. On dense connectomes mid nodes have high fan-in
   (thousands of half-paths), so this join dominates — while DFS's
   reconstruction is a pure walk over precomputed successor lists with no
   set operations.
4. **Half-path storage**: every half-path is materialised as a full list
   per length, adding allocation churn that the shared-memo DFS avoids.

Net effect: the theoretical half-depth win (b^{L/2} vs b^L) is real only
on graphs with low shared structure (e.g. deep synthetic chains); on real
connectomes the per-length rebuild + unpruned forward walk + stitching
join cost more than a full-depth but fully-pruned, memo-shared walk.

**Direction matters more than the algorithm.** The 2026-08 audit found
that the *forward* memoized DFS (`find_paths_memoized_dfs` without
`direction='backward'`) beats both of the above at every depth: it makes
no reversed-graph copy and its memo is pruned to the source-side cone
(pre-optimization measurements: 0.48 s / 7.5 MB at 4, 5.20 s / 651 MB at
5; post-2026-08-optimization the §3.1 table reads 0.23 s / 7.6 MB at 4
and 4.97 s / 651 MB at 5, vs 0.99–2.29 s / ~272 MB for the
backward/dual-side variants). It was never benchmarked before because the routing mislabeled
`MemoizedDFS` as meet-in-the-middle — the naming fix (§ audit) exposed
the best algorithm.

### Theoretical summary

| Algorithm | Time (worst) | Memory (worst) | Shortest-first | Verdict |
| --- | --- | --- | --- | --- |
| Bidirectional BFS | O(L·E + P·L) | **O(L·(V+E))** | yes | shallow graphs, sparse |
| Backward DP | O(L·E + P·L) | O(L·V) | no | **best all-round** |
| Memoized DFS | O(L·E + P·L) | O(L·E) | no | few targets; best deep tradeoff on real data |
| Meet-in-middle | O(b^{L/2}·L + P·L) | O(b^{L/2}·L) | no | shallow fast; deep paths only in theory (see §2) |

*Practical note: the meet-in-the-middle's half-depth advantage is offset by
its per-length rebuild, unpruned forward walk and pairwise stitching on
real connectomes — measured DFS (backward memoized) is the best deep
tradeoff when targets are few.*

## 3. Practical (measured) evaluation

### 3.1 Real connectome queries — hemibrain v1.2.1

Two queries, edges filtered to ≥ 5 synapses (the UI default Min Synapse
Count), search space pruned to nodes within cutoff hops of sources/targets
and dead-end-pruned, exactly like the FindAllPath pipeline:

- **2–3 intermediate layers** (cutoff 3/4): 4 source bodyIds (LPLC1, LC4)
  → 5 target bodyIds (MBON01, MBON03, PPL101): **10,331 nodes,
  301,443 edges** — 14 / 2,874 paths.
- **4–5 intermediate layers** (cutoff 5/6): 2 sources (LPLC1) → 2 targets
  (MBON01): **24,576 nodes, 722,143 edges** — 35,819 paths at 4 layers;
  5M+ paths at 5 layers (collection capped at 5M).

Times are untraced wall seconds; peak memory is the tracemalloc peak of a
separate traced run (includes the graph baseline + collected path list +
tracing overhead, so *differences* are the meaningful part — at 5 layers
~610 MB of every value is the 5 million collected paths themselves;
corrected 2026-09-25 from an inconsistent earlier "~870 MB" that exceeded
four of the five measured values below).

| Algorithm | 2 layers (14 paths) | 3 layers (2,874) | 4 layers (35,819) | 5 layers (5M+) |
| --- | --- | --- | --- | --- |
| | time / peak | time / peak | time / peak | time / peak |
| MemoizedDFS (fwd) | 0.01 s / 0.1 MB | 0.06 s / 1.7 MB | **0.23 s** / 7.6 MB | 4.97 s / 651 MB |
| DFS (bwd) | 0.02 s / 0.3 MB | 0.05 s / 1.8 MB | 0.42 s / 9.6 MB | 5.01 s / 612 MB |
| MeetInMiddle | **0.00 s** / 0.3 MB | **0.01 s** / 0.6 MB | 1.25 s / 4.8 MB | 6.18 s / 615 MB |
| DP | 0.06 s / 0.8 MB | 0.02 s / 1.5 MB | 0.44 s / 9.4 MB | 20.2 s / 651 MB |
| Bidirectional | 0.07 s / 18.9 MB† | 0.18 s / 59.4 MB† | 0.64 s / 194.9 MB† | **4.54 s** / 892 MB† |

*Fastest time per column in **bold**; † = highest memory in that column.
**2026-08 optimization:** FastGraph keeps a lazy reverse adjacency index
(`_ensure_radj`, invalidated on edits) instead of copying the whole
reversed graph per run — the earlier 100–450 MB footprints were mostly
those copies and are gone (DP dropped from 272 MB to 9.4 MB at 4 layers;
MeetInMiddle from 2.17 s / 272 MB to 1.25 s / 4.8 MB). Bidirectional's
layer trees now dominate memory (18.9 → 892 MB). At 5 layers every value
includes ~610 MB of the 5 million collected paths themselves. All five
produced identical path sets in every scenario that completed.*

Findings:

- **The 2026-08 reverse-index optimization removed the dominant memory
  cost**: all algorithms except Bidirectional now run in <10 MB at 2–4
  layers (they previously built a ~250 MB reversed-graph copy per call).
  DP: 272 → 9.4 MB; MeetInMiddle: 2.17 s / 272 MB → 1.25 s / 4.8 MB.
- **MeetInMiddle is fastest at 2–3 layers** (0.00–0.01 s); **MemoizedDFS
  (forward) is fastest at 4 layers** (0.23 s); **Bidirectional is fastest
  at 5 layers** (4.54 s) but is the only algorithm with a large footprint
  (18.9 → 892 MB — its full layer trees from both sides).
- **DFS (backward) is the best alternative when targets are few**, now
  without the reversed-graph penalty (0.3–9.6 MB at 2–4 layers).
- **DP degenerates on deep queries** (20.2 s at 5 layers) once its
  reachability sets cover most of the graph.
- At **5 layers** every value includes ~610 MB of the 5 million collected
  paths themselves — the algorithm-relative differences are the 612–892 MB
  spread around that floor.

### 3.2 Synthetic scaling (controlled branching)

Layered DAG, branching 6, 8 layers (320 nodes, 1,680 edges), 3 sources × 3
targets, cutoff 7 → 49,188 paths:

| Algorithm | time (s) | peak (MB) |
| --- | --- | --- |
| DP | 0.16 | 0.9 |
| DFS | 0.16 | 0.6 |
| MemoizedDFS | 0.30 | 0.6 |
| Bidirectional | 0.57 | 0.8 |

On a *well-structured* graph with no dead ends the theoretical differences
shrink (all algorithms within 3.5×, memory differences disappear) — they
show up on graphs with dead ends and large frontiers, i.e. real
connectomes.

Tiny synthetic graphs (≤ 750 edges, ≤ 378 paths): all algorithms complete
in < 10 ms — for small queries the choice is irrelevant.

## 4. Recommendations

*(Numbers corrected 2026-09-25 against the §3.1 measured table — the
block had been written against the pre-optimization audit-era figures
that the §3.1 "2026-08 optimization" note supersedes.)*

1. **Pipeline default: StrongestFirst** (2026-09-04, not in the measured
   tables below) — budgeted best-first on the path bottleneck; the
   enumerators benchmarked here are its unbounded complete references
   (script/API only). **Fastest complete enumerator per depth** (table
   above): MeetInMiddle at 2–3 layers, **MemoizedDFS (forward)** at
   4 layers (0.23 s) and as the best deep tradeoff, Bidirectional at
   5 layers — with MemoizedDFS keeping the smallest peak allocation at
   2–4 layers (no reversed-graph copy).
2. **Few targets, many sources**: **DFS** (backward memoized) starts from
   the smaller set — 5.01 s at 5 layers (Bidirectional's 4.54 s is the
   5-layer winner; see the table), and since the 2026-08 lazy
   reverse-index optimization it no longer pays a reversed-graph copy
   (0.3–9.6 MB at 2–4 layers).
3. **Shallow queries (2–3 layers)**: **MeetInMiddle** or MemoizedDFS —
   within noise of each other.
4. **Shortest paths first**: **Bidirectional** only if memory is no
   concern (194.9 MB at 4 layers, 892 MB at 5 — the highest in every
   column).
5. **DP** remains a robust low-memory fallback at shallow depths but
   degenerates on deep queries (20.2 s at 5 layers).

## 5. Reproducibility

```bash
python examples/performance/benchmark_pathfinding.py --json cache/benchmark_pathfinding_results.json
```

The script verifies algorithm equivalence on every scenario (all
benchmarked algorithms must return the same path set), guards each run
with a 180 s timeout and a 5,000,000-path cap, and measures wall time
(untraced) plus tracemalloc peak (traced) per algorithm. Backtracking is
not benchmarked: it is unusable on real connectomes beyond 3 intermediate
layers (see the 2026-08 history) and remains only as a backend fallback
for extreme memory constraints.

**Date**: August 2026
**Scope**: `FastGraph` algorithms in `vispath-subproject/src/vispath_pkg/fast_graph_core.py`
**Correctness**: equivalence verified on all scenarios plus the existing
randomized test suite in `tests/core/test_pathfinding.py`

---

# 2026-08 Algorithm Audit & Better-Algorithm Research

## Audit: implementations are correct, names now match

**Correctness.** A property-based audit ran all four algorithms (plus both
memoized-DFS directions) against the naive `all_simple_paths` reference on
150 random directed graphs (cycles, 1–3 sources × 1–3 targets, cutoffs
1–7): **750 algorithm-runs, zero mismatches** — every algorithm returns
exactly the positive-length simple-path set. This matches the earlier
300-graph randomized suite in `tests/core/test_pathfinding.py`. The only
contractual difference: FastGraph algorithms enumerate **positive-length**
paths only (a source that is also a target yields no trivial zero-length
path), by design.

**Naming.** The routing previously mislabeled the algorithms:

| Value | Old routing (wrong) | Fixed routing |
| --- | --- | --- |
| `MemoizedDFS` | `find_paths_meet_in_the_middle` | `find_paths_memoized_dfs` (forward) |
| `DFS` | `find_paths_memoized_dfs` (backward) | unchanged (memoized DFS, backward) |
| `MeetInMiddle` | — (did not exist) | `find_paths_meet_in_the_middle` (new) |

`DP` and `Bidirectional` were already accurate. The `MemoizedDFS` label now
runs memoized DFS, and meet-in-the-middle has its own name (`MeetInMiddle`)
in `coana.py` routing, `ComparisonParameters`, the UI Algorithm selector
and the benchmark harness. Results are identical either way (all algorithms
are equivalent); only performance semantics changed.

## Can anything enumerate all simple paths faster?

The problem is output-bound: P paths of length L force O(P·L) work, so no
algorithm can beat the current approach asymptotically on the output term.
The realistic levers are constant factors, and two candidates were
**measured and rejected** on the real 24,576-node / 722k-edge graph
(cutoff 5, 35,819 paths):

1. **Bitmask-visited guided DFS** (integer `mask | (1<<idx[v])` instead of
   the `v not in current_path` list scan): **not faster** — 3.7 s vs
   3.5 s for `find_paths_backward_dp`. Python big-int shifts on 24k-bit
   masks cost more than the ≤5-element list scans they replace; masks only
   win for very deep paths (L ≫ 64) with small graphs.
2. **Degree-2 chain compression** (contract in/out-degree-1 chains before
   searching): **nothing to compress** — only 1% of nodes are pure chain
   nodes on this connectome (hub-and-spoke topology), and the compressed
   search was slower. It would only pay off on chain-dominated graphs.

Conclusion: on real connectomes the current DP/DFS guided searches are
already near the pure-Python practical floor. The remaining levers are not
new algorithms but engines:

- **Native backends** — igraph `get_all_simple_paths` or Networkit's
  all-simple-paths (C/C++) typically cut per-path overhead 5–20×.
- **Parallel per-source enumeration** — the pipeline already chunks
  sources across processes; this is the cheapest real speedup for large
  queries.
- **Tighter graph prefiltering** — raising Min Synapse Count / Edge Limit – BodyIds
  shrinks E, and every algorithm's work is O(E)-driven; this dwarfs any
  constant-factor algorithm change (the UI guide already recommends it).

---

# Code-level audit: where the time and memory actually go

Read line-by-line from `fast_graph_core.py` (line numbers refer to the
2026-08 revision). The question that motivated this audit: **why does
Bidirectional BFS not convert its larger memory into lower time?**

## Per-path enumeration cost (measured, controlled graph)

A 47-node "double-broom" graph (4 sources → 20 mid nodes → 1 meet node →
20 mid nodes → 2 targets, cutoff 4, 3,200 paths) isolates the enumeration
machinery — memory is identical for every algorithm (0.1 MB), so only the
per-path code paths differ:

| Algorithm | time | µs/path |
| --- | --- | --- |
| DFS (backward) / DP | 0.006 s | 1.8 |
| MemoizedDFS (forward) | 0.012 s | 3.7 |
| MeetInMiddle | 0.018 s | 5.6 |
| **Bidirectional** | **0.033 s** | **10.4 (5.8× slower)** |

## Why Bidirectional's enumeration is the most expensive per path

`find_paths_bidirectional_bfs` (lines 888–987) spends its memory on the
full layer trees, but the enumeration phase is implemented with
quadratic copying:

1. **List-copy concatenation at every depth level** — `yield p + [u]`
   (line 965) and `yield [u] + p` (line 975): each recursion level copies
   the whole path so far, so one half-path of depth ℓ costs O(ℓ²)
   element copies. DP/DFS use in-place `path.append(v)` / `path.pop()`
   (lines 793–795, 865–867) with a single `yield list(current_path)`
   (lines 785, 859) — **one O(ℓ) copy per output path**.
2. **Full materialisation per meet node** — `f_paths = list(get_fwd_paths(u, mid))`
   and `b_paths = list(get_bwd_paths(u, rem))` (lines 982–983): every
   meet node's half-path lists are built into memory before the pairing,
   and the same meet node re-enumerates them for **every** length in
   which it participates (no memoisation across lengths, line 947 loop).
3. **Per-pair set machinery** — `len(set(fp) & set(bp)) == 1` (line 986):
   two full set constructions plus an intersection per candidate pair,
   then `fp + bp[1:]` (line 987) copies the whole path again. On graphs
   with a high-fan-in meet node this pairing loop dominates (the
   double-broom graph is exactly this worst case).
4. **Unpruned, full-cutoff layer build** — the f/b layer maps (lines
   916–923, 933–940) store parent-pointer sets for *every* node of *every*
   depth up to the cutoff, re-scanning each node's adjacency once per
   depth it appears in: O(L·(V+E)) entries — the 119 → 159 → 452 →
   1,149 MB measurements. The maps are built for the whole cutoff even
   for the short lengths, and contain nodes that lie on no s→t path at
   all (no pruning to actual reachability).
5. **Why the classic bidirectional win does not apply**: in
   shortest-path search you stop at the first meeting layer, so the
   halved frontier pays off. In all-paths enumeration there is no early
   stop — every path must be emitted, and the stitching machinery is
   dearer per path than a plain pruned DFS walk. Measured consequence:
   Bidirectional is slower than the memoized variants at 2–4 layers
   (1.45 s vs 0.48 s at 4 layers on the real query, 35,819 paths) and
   only ties DFS at 5 layers (4.93 s vs 4.91 s) while using 1.1 GB.

## Where the other algorithms' time and memory go

- **DP** (`find_paths_backward_dp`, lines 748–808): memory = the
  `valid_nodes_at_dist` node *sets* only (line 772, O(L·V) — no parent
  links), built by bulk `set.update(R.adj[...])` (line 780, C-speed).
  Enumeration = in-place guided DFS with an O(1) set-membership prune
  (line 791) and an O(ℓ)-scan cycle check (line 792, effectively constant
  for ℓ ≤ 7). One O(ℓ) copy per path (line 785). Costs: the `reverse()`
  copy (~250 MB) and degenerate deep cutoffs (18.5 s at 5 layers) once
  the reachability sets cover most of the graph.
- **Memoized DFS** (`find_paths_memoized_dfs`, lines 810–886): memory =
  the `valid_successors` (node, depth) → successor-list memo (line 839),
  computed once per state (lines 841–854) and shared across all lengths.
  Enumeration = in-place walk over the memo lists (lines 856–867), one
  O(ℓ) copy per path (line 859). Forward direction additionally avoids
  `reverse()` entirely — the smallest memory and the fastest measured.
- **Meet-in-the-middle** (`find_paths_meet_in_the_middle`, lines
  644–746): rebuilds `fwd_paths_map` from scratch for every length
  (lines 712–728 — half-paths of depth 2 are re-enumerated for lengths
  4, 5, 6); the forward DFS walks branches unpruned, filtering only at
  the leaf (lines 680–684); stitching is the same per-pair set machinery
  as Bidirectional (`len(set(f_path) & r_set) == 1`, line 744, plus
  `f_path + r_path_rev[1:]`, line 745) — hence its 5.6 µs/path on the
  broom graph and 2.17 s at 4 layers on the real query.
- **Backtracking** (`find_paths_dfs_backtracking`, lines 590–642): no
  memo, no pruning — re-walks the whole tree per length per target
  (lines 636–642); O(L) memory but unusable beyond 3 intermediate layers
  on real data. Kept only as a backend fallback.

## Correctness re-verified

All four algorithms (plus both memoized-DFS directions) matched the naive
`all_simple_paths` reference on 150 random directed graphs (cycles,
multi-source/target, cutoffs 1–7): **750 runs, zero mismatches**, under
the documented positive-length contract.
