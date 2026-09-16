# DROCAT Pathfinding Pipeline — Comprehensive Technical Report

**Status:** current as of 2026-09-07. This report summarizes every
pathfinding strategy in DROCAT end-to-end: discovery, the lossless
pruning stack, the budget-fit Edge Budget, the StrongestFirst family,
guarantees, complexity, measured performance, and the design decisions
behind them. Companion documents:
[STRONGEST_FIRST_PATHFINDING.md](STRONGEST_FIRST_PATHFINDING.md)
(the StrongestFirst design record),
[PATHFINDING_ALGORITHM_EVALUATION.md](PATHFINDING_ALGORITHM_EVALUATION.md)
(complete-enumerator benchmarks),
[PathFinding_Methods.md](../core-features/PathFinding_Methods.md)
(user guide). Plan/design history lives in `_plan/plan-pathfinding-doc-audit.md`.

## 1. Scope and entry points

| Entry point | Mode | Pipeline |
| --- | --- | --- |
| `FindNeuronConnection.FindAllPath()` | `'all'` — all simple paths ≤ bound | this report, all stages |
| `FindNeuronConnection.FindShortestPath()` | `'shortest'` — per-pair minimum-hop paths | same pipeline, shortest-only enumeration, never floored |
| `FindNeuronConnection.FindPath()` | legacy type-level | full threshold-filtered cone, type paths via graph search |
| `ComparisonAnalyzer` (`path_mode`) | both | the same engine per dataset/threshold |
| `FindAllPathMultiThreshold()` | `'all'` | enumerates once at t₀, replays higher thresholds from the bottleneck-annotated set |

Everything runs on the bodyId-level `FastGraph`
(`vispath-subproject/src/vispath_pkg/fast_graph_core.py`); type-level
paths are **derived** from bodyId paths (unique type sequences), never
searched separately — the *bundle effect* rationale is in
[TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md](TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md).

## 2. The pipeline, stage by stage

All stage references are `src/coana.py` unless noted. Orchestration:
`FindAllPath` / `FindShortestPath` → `_find_paths_core` (shared), graph
preparation in `_graph_edge_frames`.

### Stage 0 — Resolution and threshold filter
Source/target queries (types, bodyIds, instances, regex) resolve to
bodyIds; the `all_neurons` token forces depth 0. `min_synapse_num`
filters connection rows at fetch time (weight = synapse count).
`min_ratio` / `min_traversal_probability` are **readout columns only**
(F9 retirement) — they never filter the graph.

### Stage 1 — Discovery
- **'all' mode**: forward layer-by-layer fetch with `forward_only=True`
  (each neuron queried once; the frontier is `post − all_discovered`).
  The depth cap `max_interlayer + 1` is the **only** stop condition in
  'all' mode — there is no target early-stop (the log banner is
  mode-conditional since 2026-09-06). A cap-terminated run with a live
  frontier sets `discovery_complete=False` and warns.
- **'shortest' mode**: target-rooted **backward** discovery — each
  target expands incoming edges until the requested sources are seen or
  the hop bound expires. A target's first-appearance depth is its exact
  shortest distance, so discovery **stops early** once every target is
  enrolled (provably safe), and per-target hop limits — the FARTHEST
  requested source's own shortest distance, per the per-pair contract —
  are recorded for enumeration (`_shortest_target_hop_limits`). The
  per-target distance maps are passed to the enumerator, which skips its
  own reverse BFS, and a per-POST incoming cache
  (`incoming_connections.parquet` + `incoming_complete.json` beside the
  source-oriented `connections.parquet`) serves proven-complete frontier
  posts without re-querying (2026-09-08 shortest-path fixation).
- **Comprehensive re-querying** (`forward_only=False`, script-only)
  re-queries all discovered neurons at every layer; since 2026-09-06 the
  duplicate `(pre, post)` rows this produces are **dropped** at fetch
  time (anti-join against the seen-pair set) — previously `add_edge`
  summed them, multiplying the pair's weight by its layer count (§4.1 of
  the audit). The default `forward_only=True` is unaffected (each pair
  is fetched once).

### Stage 1b — Untyped-neuron filter (`drop_untyped`, both modes)

`FindNeuronConnection.drop_untyped` (default `True`; UI checkbox "Drop
Untyped Neurons" in Output Options of both tabs, payload key
`drop_untyped`, configurable in Settings → Default Settings) removes
connection rows whose pre- or post-side label is untyped. It runs in
the shared `_find_paths_core` for BOTH modes — the 'all' forward layer
loop and the 'shortest' target-rooted backward discovery
(`_discover_shortest_backward`) — AFTER label enrichment and BEFORE
graph construction, so an untyped neuron can never be an intermediate
node of a returned path or visualization.

- **Predicate**: `utils.label_utils.is_untyped_type_label` — a label is
  untyped when it is empty after strip, one of the Unknown/None/NaN
  sentinels (case-insensitive), or all-digit (numeric bodyId fallback).
  Cross-Dataset Comparison delegates to the same predicate
  (`ComparisonAnalyzer._is_untyped_type_value`) but keeps its own
  POST-label-mapping timing: the per-dataset `FindNeuronConnection`
  runs execute with `drop_untyped=False` so only the comparison-level
  filter fires (`_drop_untyped_neurons` in
  `src/comparison/comparison_analyzer.py`).
- **Records**: `data_details/untyped_dropped_records.csv` (written only
  when rows were dropped) — columns `dataset`, `threshold`,
  `conn_layer`, then the connection columns present, plus
  `untyped_side` (`pre` | `post` | `pre+post`). `user_warning_notes.txt`
  gains an `[untyped dropped]` entry only when rows were dropped
  (never for a no-op enabled filter).
- Untyped source/target bodyIds may remain enrolled in
  `source_neurons.csv` / `target_neurons.csv` while their incident
  edges were removed.
- `drop_untyped` is part of the FindAllPath graph-cache key
  (`_findallpath_cache_key`).

### Stage 2 — Lossless hop-budget pruning (fixpoint)
`prune_layers_hop_budget` (`max_passes=4` since 2026-09-06):

- One pass computes unweighted BFS distances `distS(u)` (from sources,
  forward) and `distT(v)` (to targets, reverse) over the current layer
  tables and keeps edge (u,v) iff `distS(u) + 1 + distT(v) ≤ bound`,
  `bound = max_interlayer + 1`.
- **Lossless by proof**: any admissible simple path through (u,v)
  certifies `distS(u) ≤ a`, `distT(v) ≤ b` with `a + 1 + b ≤ bound`.
  *Lossless, not complete*: prefix/suffix minima may share vertices a
  simple path cannot reuse.
- **Iterated to a fixpoint** (§7.1 of the audit): distances recomputed
  on the already-pruned tables can only grow, so each pass can only
  surface more dead edges; iteration narrows the gap monotonically and
  is still lossless (every admissible path survives every pass). Stats
  report cumulative `rows_dropped` and `passes`; `max_passes=1`
  reproduces the historical single pass.
- Runs **at every depth, in both modes, unconditionally, before the
  graph is built**. Inputs are never mutated (the FindAllPath graph
  cache shares frames across thresholds).
- Each pass reports the **strongest-retained bottleneck W\*** (widest
  maximin to the targets) — identical before and after lossless passes,
  printed as *"top paths unchanged"*.

Measured effect (FAFB, 8 sources × 1,756 Tm3, bound 4): **5,847,663 →
289,048 rows (−95%)** in one pass; few-target queries prune far harder
(6-target L3 cone: 5.85M → 24k, −99.6%) — for them the Edge Budget
never fires at all.

### Stage 3 — Edge Budget floor (budget-fit search; 'all' mode only)
`fit_edge_budget` (2026-09-06; refines the one-shot Fix D landing).

When the lossless-pruned cone `E₀` still exceeds `graph_edge_limit_bodyid`
(UI default 1M; 0/None = off), the search finds:

```
t* = min { t : C(t) ≤ cap }      C(t) = | fixpoint-closure of {e : w(e) ≥ t} |
```

— the **weakest weight tier whose lossless-closed cone fits the cap**.

- **Probes**: mask `w ≥ tier` + single-pass lossless closure (the same
  operator as the one-shot's second pass), so accept decisions are
  conservative and probe 0 is bit-identical to the historical one-shot.
- **Search**: quickselect landing (N-th strongest weight `w1`,
  canonicalized to the strongest tier it admits — `E(w1+1)` is the same
  cone) → gallop down the tiers (`i₀+1, +2, +4, …`) → on the first
  overshoot, bisect between the last fitting and first overshooting
  tier. Probe budget `max_probes=8`; zero-count probes count as *fits*
  (a weaker tier can revive the cone); the final set must be non-empty
  or the floor is declined honestly.
- **Semantics**: a pure threshold raise — the floored run is exactly a
  complete run at `min_synapse_num = t*`. Reported as
  `edge_weight_floor` (t\*), `edge_budget_landing` (first overshooting
  tier), plus `residual_slack`, `budget_fully_used`, `probes`,
  `truncated`, `search_seconds`, `probe_trace` in
  `graph_pruning_record` / `all_attributes.json` / a loud note.
- **Guarantees**: kept rows ≤ cap; if not truncated, the next weaker
  tier exceeds the cap (budget maximally spent under threshold
  semantics). When even the full closed cone fits, no floor is applied
  (`floor_skipped` — fixing the one-shot's latent over-flooring, whose
  gate fired on the pre-closure count).
- **Why not the one-shot** (`w0 = w1 + 1`): the +1 excludes the
  boundary-tie tier and the closure's slack was discarded — measured
  waste 70–98% of the cap on real cones, and one million-scale case
  where it declined entirely while the fit search found a valid floor
  (`apply_edge_budget_floor` is retained as the tested probe-0
  reference).
- **Shortest mode is never floored** (backend gate + the Shortest tab's
  forced 0): flooring preserves reachability but can inflate shortest
  distances.

Search cost is bounded by `max_probes` × one lossless pass over the cone
and scales linearly: measured +0.2–0.7s at E₀ = 289k, +2–4.6s at E₀ =
2.05M (pure-Python prototype; the production port is vectorized).

### Stage 4 — Graph build
Layer tables feed `FastGraph.build_from_dataframe(...,
store_edge_attrs=False)` (slim mode: weights live in the adjacency
dicts, saving ~350 bytes/edge); duplicate pairs **across layers are
summed** — correct for legitimate merges (e.g. type aggregation), and
with the Stage-1 dedup unreachable as a fetch artifact.

### Stage 5 — Dead-end node prune (post-build)
Backward BFS from the targets; `G.subgraph(nodes_that_can_reach_targets)`,
plus the W\* report. Lossless (no enumerator can use a node that cannot
reach a target). After the §7.1 fixpoint this sweep removes exactly
nothing in theory (any node incident to a kept edge lies on an
admissible walk); it is retained as cheap insurance and can be demoted
to an assertion once the fixpoint settles in.

### Stage 6 — Enumeration

#### 6a. 'all' mode — StrongestFirst (the pipeline algorithm)
`FastGraph.find_paths_strongest_first`:

1. **Widest-path backward DP** (`strongest_core.widest_path_backward`):
   `W[d][v]` = best achievable bottleneck (max over paths of min edge
   weight) from v to any target within d edges; `W[0][t] = ∞`.
2. **A\*-style best-first** over path prefixes keyed by the *exact*
   best completion `min(running bottleneck, W[remaining][node])` —
   pops come in non-increasing bound order, so complete paths are
   emitted in **descending bottleneck order**. `W = None` prefixes are
   never pushed (the DP doubles as an admissible prune). Neighbors are
   walked strongest-first with deterministic tie rules.
3. **Budget + τ drain** (`max_paths_bodyid`; None/0 = auto → internal
   1,000,000 — StrongestFirst never runs unbounded): at the budget the
   drain opens; every tie at the cutoff **τ** is drained and everything
   strictly weaker is dropped, recording the strongest dropped
   bottleneck w2. Output = exactly `{bottleneck ≥ τ}`; canonical τ =
   w2 + 1 is the minimal query reproducing the set.
- **Fix C routing**: any positive `max_paths_bodyid` routes *any*
  algorithm selection through StrongestFirst; the selector only
  distinguishes the unbounded complete enumerators (script/API, all
  verified set-equal): `MemoizedDFS` (forward; fastest complete),
  `DFS` (backward; few targets), `MeetInMiddle`, `DP`, `Bidirectional`,
  `Backtracking` — benchmark table in
  [PATHFINDING_ALGORITHM_EVALUATION.md](PATHFINDING_ALGORITHM_EVALUATION.md).

#### 6b. 'shortest' mode — StrongestFirst on the shortest-path DAG
`FastGraph.find_paths_shortest_strongest_first` (2026-09-06; §7.2):

1. Per target t: backward BFS `dist_t`; the **shortest-path DAG** is the
   dist-descending subgraph (`dist_t[v] = dist_t[u] − 1`) — acyclic by
   construction, so no visited bookkeeping.
2. Per-target maximin DP `Wt[v]` over DAG edges; best-first on
   `min(run_bottleneck, Wt)` emits that target's min-hop paths in
   descending-bottleneck order.
3. Per-target streams are **k-way merged** (global descending order);
   the global budget drains ties at τ exactly as in 'all' mode.
   `target_cutoffs` (per-target hop limits from backward discovery) are
   honored. Zero-hop pairs (source = target) are excluded, matching the
   siblings.
- **Contract**: a bitten run is exactly "all min-hop paths with
  bottleneck ≥ τ" (τ reported, loudly); an unbitten run is the complete
  min-hop set — verified set-identical to `find_paths_shortest_backward`
  (a separate enumerator that exists in the codebase but is **not**
  called by the pipeline) on real data (413,115 paths). The
  per-pair search is polynomial (no non-shortest branch is explored),
  but the *total* min-hop count can still grow quickly at depth — this
  budget is its bound. Stats: `emitted`, `tau`, `budget_bitten`,
  `strongest_dropped`, `per_target`. UI: Shortest-tab **Max Paths
  (BodyId)** (0 = auto → 1M).

### Stage 7 — Enrichment and outputs
Per-path bottleneck annotation, τ / canonical-τ / budget reporting into
`all_attributes.json` + `parameters.txt` + `user_warning_notes.txt`,
replay capture (Feature F), type-path derivation, CSV/XLSX exports,
visualization (display-only caps: `edgeN_limit`, strongest-interior
path ranking). Identical for both modes.

After enumeration, every run (BOTH modes — shortest was previously
never re-stamped) writes the **applied-threshold provenance block** to
`parameters.txt`, `all_attributes.json`, AND `data_details/parameters.csv`
(computed by the shared `applied_threshold_provenance()` in
`src/utils/threshold_state.py` — also delegated to by
`ComparisonAnalyzer._applied_state_for` for the cross-dataset
summary rows; finalized by `_finalize_threshold_provenance` +
`_write_run_metadata`):
`requested_threshold`, `applied_threshold`, `applied_threshold_source`
(`requested` | `strongest_first_budget` | `edge_budget` |
`strongest_first_budget+edge_budget`), `strongest_first_budget`
(effective budget; auto 1,000,000), `strongest_first_budget_bitten`,
`strongest_first_tau` (landing τ), `tau_canonical` (minimal equivalent
threshold: `w2+1` when the bite leaves a gap `[w2+1, τ]`, else the
natural τ), `strongest_dropped_bottleneck` (w2), `edge_budget`,
`edge_budget_applied`, `edge_budget_landing` (w1), `edge_weight_floor`
(w0), `strongest_retained_bottleneck` (W\* — the widest-path ceiling
after lossless pruning), `paths_complete`. `parameters.txt` keeps the
backward-compatible alias lines `applied_tau (min path bottleneck)` and
`edge_weight_floor`. Semantics: `applied_threshold` is the requested
threshold for complete/unbounded runs (the natural τ is reported
separately); when a lossy budget affects the output it is the canonical
minimal threshold reproducing the materialized set, and
`applied_threshold_source` names the mechanism(s). τ is a
landing/collapse bound — the budgeted output is a strength-bounded path
set, never an arbitrary first-N truncation. Replay-materialized folders
(named by the applied-threshold grammar: a genuine requested run keeps the
bare `minsyn_{requested}`; a collapse floor is
`minsyn_{applied}_applied_floor`; an aliased requested level is
`minsyn_{requested}_skipped`; a requested level coinciding with its floor is
skipped silently) and the
Cross-Dataset raw-run folders selected by standard or combination query rows
carry the same block. A
`[combined threshold]` note is appended to `user_warning_notes.txt`
when BOTH budgets affected a run.

## 3. Guarantees (what a run promises)

With the §5 notation (𝒫 / 𝒫_short universes, β the bottleneck, L̂ the
fixpoint-closed cone, t\* the fit floor):

1. **τ-equivalence (P1)**: a budget-bitten run emits exactly
   `{ p : β(p) ≥ τ }` — `{ p ∈ 𝒫 : β(p) ≥ τ }` in 'all' mode,
   `{ p ∈ 𝒫_short : β(p) ≥ τ }` in 'shortest'. Never an arbitrary
   subset (the Fix C line), and the minimal reproducing threshold is
   the reported canonical τ = w2 + 1.
2. **Pruning soundness (P2)**: `{ edges(p) : p ∈ 𝒫 ∪ 𝒫_short } ⊆ L̂`
   (Theorem 1) and the dead-end node prune keeps every path; the floor
   is the only lossy stage and is exactly the threshold raise
   `E ↦ E(t*)`.
3. **Cap honesty (P3)**: when a floor is applied, |kept| = C(t\*) ≤ cap,
   with `residual_slack = cap − C(t*)`, `budget_fully_used` ⇔ the next
   tier exceeds the cap, and `truncated` flagged when the probe budget
   stopped the search early.
4. **Composition (P4)**: a bounded run's effective threshold is
   `max(τ_budget, t*)`; the output equals a complete run at that
   threshold.
5. **Determinism (P5)**: identical graphs yield identical emission
   sequences (total order on heap keys, §5.3).
6. **Threshold nesting (P6)**: s ≤ t ⟹ complete-run output at
   s ⊇ output at t (cones nest via monotone E(·) and Φ), which
   underpins multi-threshold comparison and replay.
7. **Mode separation (P7)**: the Edge Budget never touches shortest
   mode (backend gate + UI), so shortest distances are never distorted
   by a strength floor; shortest mode is bounded only by depth and the
   path budget.

## 4. Complexity and memory

Variables: `V`/`E` = nodes/edges of the pruned cone, `L = B` the
bound, `P = |𝒫|` (or |𝒫_short|) the path count, `H` the live heap peak,
`X` the explored prefix count (≤ prefixes with κ ≥ the effective
cutoff), `D` the discovery depth. Per stage: discovery Θ(D · fetch);
fixpoint prune ≤ 4 passes × Θ(B·E); fit search ≤ 8 probes × Θ(B·E);
DP Θ(L·E). Enumeration:

| Strategy | Time | Aux memory | Order / bound |
| --- | --- | --- | --- |
| **StrongestFirst** ('all', default) | O(L·E) DP + explored·log H + P·L | O(L·V) + O(H·L) | descending bottleneck; **budget + τ** |
| **Shortest-SF** ('shortest', default) | Σ_t O(V_t + E_t) + emitted work | Σ_t O(V_t + E_t) streams | descending bottleneck; **budget + τ** |
| MemoizedDFS fwd (API) | O(L·E) + P·L | O(L·E) memo | complete, unordered |
| DFS bwd (API) | same | O(L·E) | complete; best when T ≪ S |
| MeetInMiddle (API) | O(b^{L/2}·L) + P·L | O(b^{L/2}·L) | complete; shallow |
| DP (API) | O(L·E) + P·L | O(L·V) — lowest | complete; degenerates deep |
| Bidirectional (API) | O(L·E) + P·L | O(L·(V+E)) — highest | complete, shortest-first |
| Backtracking (API) | O(b^L) | O(L) | complete; iterative deepening |

All complete enumerators return the identical path set (property-tested;
750 algorithm-runs, zero mismatches). StrongestFirst's only output
advantage is the budget; its admissible A\* bound (§5.2) means no exact
algorithm can expand fewer prefixes at equal emission order — the
remaining honest speedups are constant-factor and I/O-level.

## 5. Algorithm deep dives

**Notation.** The search graph is the directed weighted graph
`G = (V, E, w)` with `w : E → ℤ≥1` (synapse counts, stored as float64).
`S ⊆ V` sources, `T ⊆ V` targets, `B = max_interlayer + 1` (the bound;
`L ≡ B`). A path is a node sequence `p = (p₀, …, p_k)` with `|p| = k`
edges; `p` is *simple* iff its nodes are pairwise distinct; its
**bottleneck** is `β(p) = min{ w(p_{i−1}, p_i) : i = 1..k }`, with
`β(p₀) = +∞` for the empty path. The admissible universe is

```
𝒫 = { p : p simple, p₀ ∈ S, p_k ∈ T, 1 ≤ |p| ≤ L }        P = |𝒫|
```

(paths may pass through other targets). For shortest mode,
`𝒫_min(t)` = min-hop paths ending at `t`, `𝒫_short = ⋃_{t∈T} 𝒫_min(t)`.
`d_S(v)` / `d_T(v)` are unweighted min-hop distances from `S` / to `T`
(∞ if unreachable). Weights are integral, which is what makes
"τ − 1"-style statements well-defined.

### 5.1 Lossless hop-budget pruning — data flow and proofs

```
pass(tables, S, T, B):
  adj, radj, adj_w ← union of all non-empty layer tables   # intern ids → int32
  d_S ← BFS(S, adj);  d_T ← BFS(T, radj)                   # unweighted, BIG = ∞
  for each table:  keep row (u,v)  iff  d_S(u) + 1 + d_T(v) ≤ B
  strongest_retained ← maximin BFS (best bottleneck S → T)  # reporting only
```

Define the pass operator on edge sets `X ⊆ E`:

```
Φ(X) = { (u,v) ∈ X : d_S^X(u) + 1 + d_T^X(v) ≤ B }
```

with distances computed **within** `X`. The pipeline computes
`L̂ = Φ^k(E₀)` for the smallest `k ≤ max_passes` with
`Φ^k(E₀) = Φ^{k+1}(E₀)` — the **greatest fixpoint** of Φ below E₀.

**Theorem 1 (losslessness).** ∀k ≥ 0: `{ edges(p) : p ∈ 𝒫 } ⊆ Φ^k(E₀)`.
*Proof.* Induction on k. k = 0 is trivial. Assume p ⊆ Φ^k(E₀) and let
(u,v) be p's i-th edge, with prefix length a and suffix length b,
`a + 1 + b = |p| ≤ B`. The prefix is an a-edge walk inside Φ^k(E₀), so
`d_S^{Φ^k}(u) ≤ a`; likewise `d_T^{Φ^k}(v) ≤ b`; hence
`d_S + 1 + d_T ≤ a + 1 + b ≤ B` and the row survives. ∎ (The test is
evaluated against distances *within the current set*, which p inhabits
by the induction hypothesis.)

**Remark (lossless ⊄ complete).** The converse implication holds for
walks but not for simple paths: `d_S(u) + 1 + d_T(v) ≤ B` certifies a
*walk* through (u,v), and the two shortest walks may share interior
nodes. Pruning exactly the edges on some simple path is NP-hard
(directed 2-vertex-disjoint-paths, Fortune–Hopcroft–Wyllie 1980) — the
fixpoint test is the practical ceiling.

**Theorem 2 (fixpoint convergence).** Φ is (i) **monotone**:
`X ⊆ X′ ⟹ Φ(X) ⊆ Φ(X′)`, because growing the edge set can only
decrease distances pointwise (`X ⊆ X′ ⟹ d^X ≥ d^{X′}`); and (ii)
**intensive**: `Φ(X) ⊆ X`. Hence `(Φ^k(E₀))_{k≥0}` is a descending
chain in the finite lattice `(2^E, ⊆)`, stabilizing at `L̂` in
≤ max_passes steps with `Φ(L̂) = L̂`. At the fixpoint every node
incident to a kept edge lies on a concrete S→T walk of ≤ B hops (the
BFS parent chains construct it), so the post-build dead-end node prune
— whose predicate "can reach T at any distance" is weaker — removes
exactly nothing.

**Stranding** motivates iteration: one pass can keep (u,v) with
`d_S(u) + 1 + d_T(v) ≤ B` while every continuation of v died in that
same pass; recomputing on Φ(E₀) raises `d_T(v)` and the edge then
fails. Empirically 2–3 passes converge (cap 4; fit-probes use 1).

**Complexity.** Per pass: two BFS sweeps Θ(|V| + |E|) + a vectorized
mask Θ(|E|) (polars `filter` / pandas `loc`; frames are new objects —
cached graph frames are never mutated). Total
≤ max_passes · Θ(B·|E|). The W\* report is one extra maximin sweep,
Θ(|E| log |V|).

### 5.2 The widest-path DP (shared by both StrongestFirst variants)

Define the maximin value over bounded walks ending in T:

```
𝒲_0(v) = +∞                iff v ∈ T      (the empty completion is free)
𝒲_d(v) = max{ β(q) : q walk v → T, 0 ≤ |q| ≤ d }   (max ∅ = ⊥, "absent")
𝒲_d(v) = max_{(v→w, c) ∈ E} min( c, 𝒲_{d−1}(w) )   for d = 1..L
```

`⊥` (coded as `None`) means no bounded walk exists. Bottom-up
evaluation over d = 0..L costs Θ(L·|E|) time, O(L·|V|) space. Two
consumers:

1. **Pruning**: `𝒲_{L−h}(v) = ⊥` ⟹ no completion within the remaining
   hops ⟹ a prefix at depth h ending in v is never pushed.
2. **Admissible prefix bound** for a prefix π ending at v with running
   bottleneck `ρ(π) = min{ w(e) : e ∈ π }`:

```
κ(π) = min( ρ(π), 𝒲_{L−|π|}(v) )  ≥  max{ β(π∘q) : π∘q ∈ 𝒫 }
```

κ is an **upper bound** on the best simple-completion bottleneck
(walks may revisit π's nodes; bound and truth coincide whenever the
optimal walk is already simple). *Admissibility* — never an
underestimate — is precisely the property the ordering and drain
theorems (§5.3) require; exactness over simple completions is not
needed.

### 5.3 StrongestFirst ('all' mode) — pseudocode and invariants

```
W ← widest_path_backward(T, L)                 # W[d][v] ≡ 𝒲_d(v)
heap ← { (−W[L][s], tie++, s, hops=0, run=+∞, (s,)) : s ∈ S, W[L][s] ≠ ⊥ }
τ = None; drain = false; w2 = None
while heap:
    (−b, _, v, hops, run, path) = heappop(heap)
    if drain and b < τ: w2 = max(w2, b); break          # everything left < τ
    if v ∈ T and hops ≥ 1:
        yield path; τ = run; emitted += 1
        if emitted ≥ budget: drain = true                # open the τ drain
    if hops = L: continue
    for (w, c) in sorted(adj[v]) by (−c, w):             # strongest first, ties by id
        if w ∈ path: continue                            # simple-path constraint
        r' = min(run, c);  vb = W[L−hops−1].get(w)
        if vb = ⊥: continue                              # no completion in depth
        b' = min(r', vb)
        if drain and b' < τ: w2 = max(w2, b'); continue  # below the cutoff
        heappush((−b', tie++, w, hops+1, r', path+(w,)))
stats: emitted, τ, budget_bitten, strongest_dropped = w2, last_bound
```

**Theorem 3 (descending emission).** If complete paths are emitted at
times i < j then `β(p_j) ≤ β(p_i)`. *Proof.* When p_i pops, its
bottleneck `β(p_i) ≤ κ(π_i)` (ρ of a completion is ≤ the prefix bound).
The heap is a max-heap on κ, so every entry present has κ ≤ β(p_i);
and κ is non-increasing along descendants (κ(child) = min(ρ', 𝒲) with
ρ' ≤ ρ, 𝒲 ≤ κ(parent)). Hence every future completion has
β ≤ κ ≤ β(p_i). ∎

**Theorem 4 (completeness without a bite).** With budget = ∞ the
emitted multiset equals 𝒫. *Proof.* A prefix is discarded only if (a)
it repeats a node (not simple, not in 𝒫), or (b) `𝒲 = ⊥` (no
completion — correct by definition), or (c) after a drain (absent
here). Every p ∈ 𝒫 therefore has all its prefixes pushed, and pops
occur in the order of its nodes' depth. ∎

**Theorem 5 (bite ⇒ exact τ-cut).** With budget c < P, let
`τ = β(p_c)` (the c-th emitted bottleneck). Then
`emitted = { p ∈ 𝒫 : β(p) ≥ τ }`, `|emitted| ≥ c` (ties at τ drained),
and with `w2 = max{ β(p) : p ∈ 𝒫, β(p) < τ }` (max ∅ → ⊥),
`emitted = { p ∈ 𝒫 : β(p) ≥ w2 + 1 }` — w2 + 1 is the **minimal**
threshold reproducing the set. *Proof sketch.* After the drain opens,
a prefix is pushed only if its admissible bound `κ ≥ τ`, so every path
with β ≥ τ retains a pushed prefix chain and is emitted before any
break; the break fires at the first pop with κ < τ, and since κ ≥ best
achievable, no path with β ≥ τ remains. w2 is exactly the largest
dropped bottleneck (recorded at both the break and the push filter).
∎ **Canonical τ:** `w2 + 1` is the minimal threshold reproducing the
set (weights integral ⇒ the next integer step is meaningful).

**Determinism.** Heap keys `(−κ, counter)` with counter monotone, plus
neighbor order `sort by (−c, node id)` — a total order; identical
graphs produce identical emission sequences.

**Worked example.** S→A(9)→T, S→D(7)→T, S→B(5)→T (each label = both
edges' weight), budget 2. Seed: κ = 𝒲₂(S) = 9. Pop S → push (S,A) 9,
(S,D) 7, (S,B) 5. Pop (S,A) → push (S,A,T) 9 → pop, emit, τ = 9.
Pop (S,D) 7 → push (S,D,T) 7 → pop, emit, τ = 7, budget reached →
drain. Pop (S,B) 5 < τ → w2 = 5, break. Output
{(S,A,T), (S,D,T)}, τ = 7, canonical τ = 6.

**Complexity.** Θ(L·|E|) DP; heap operations O(X log X) over X
explored prefixes; Θ(P·L) output writing. With a bite at τ, explored
prefixes satisfy κ ≥ τ, so work ≈ output + frontier above τ.

### 5.4 Shortest-mode StrongestFirst — per-target DAG machinery

For each target t (all streams run concurrently, merged globally):

1. **Backward BFS**: `d_t(v)` = min hops v→t, expanded while
   `d_t < hop_cap(t)` (the per-target limit from discovery). Nodes
   beyond the cap or unreachable are absent — `d_t(v) = ⊥`.
2. **Shortest-path DAG**: `DAG_t = { (u,v) ∈ E : d_t(u) = d_t(v) + 1 }`,
   identified lazily during expansion. Acyclic: d_t strictly decreases
   along every kept edge, so `d_t` itself is a topological order
   (reversed). **Characterization**:
   `𝒫_min(t) = { p : p₀ ∈ S, p_k = t, ∀i: (p_{i−1},p_i) ∈ DAG_t,
   p₀ ≠ t }` and every such p has `|p| = d_t(p₀)` — min-hop exactly
   (a dist-decreasing walk of k steps ends at the unique dist-0 node).
3. **Per-target maximin DP** over DAG_t in topological order (dist
   ascending; all DAG-successors of a dist-d node have dist d−1 and
   are settled first):

```
Wt(t) = +∞;   Wt(u) = max_{(u→v) ∈ DAG_t} min( w(u,v), Wt(v) )   (max ∅ = ⊥)
```

   This is §5.2's DP with the depth index collapsed — the distance
   gradient is the depth. Θ(|V_t| + |E_t|) per target.
4. **Per-target best-first**: the §5.3 loop restricted to DAG edges,
   seeded from `{s ∈ S : d_t(s) ≤ hop_cap(t), s ≠ t}` with
   `κ = min(ρ, Wt)`; each target's stream yields 𝒫_min(t) in
   descending-bottleneck order (Theorems 3–4 apply per stream).
5. **k-way merge + global drain**: `heapq.merge(streams, key = −β)`
   produces a globally descending sequence; the first post-bite item
   with β < τ is the globally strongest dropped path (Theorem 5 lifts
   verbatim: `emitted = { p ∈ 𝒫_short : β(p) ≥ τ }`).

`target_cutoffs` are the per-target FARTHEST requested source's own
shortest distance (per-pair contract, 2026-09-08): every reachable
`(source, target)` pair gets its own minimum-hop set, and the discovery
BFS distance maps seed the enumerator directly (no second reverse BFS).
`find_paths_shortest_backward`, the separate backward enumerator that
shares this contract, is **not** called by the pipeline —
`find_paths_shortest_strongest_first` is the pipeline's shortest-mode
enumerator. Memory: all per-target states coexist —
`Σ_t (|V_t| + |E_t|)` — modest for the deep/few-target explosions this
budget targets, linear in |T| for very broad queries. Zero-hop pairs
(source = target) are excluded, matching the sibling enumerator.

### 5.5 Budget-fit search — bracket invariants

Let `𝒯 = (t₁ > t₂ > … > t_M)` be the distinct weights of E₀ (tiers),
`E(t) = { e ∈ E₀ : w(e) ≥ t }`, and

```
C(t) = | Φ^ω(E(t)) |                 # the fixpoint-closed cone at t
fits(t)  ⟺  C(t) ≤ cap
```

**Lemma (monotone predicate).** s ≤ t ⟹ C(s) ≥ C(t) (E(·) and Φ are
monotone, Theorem 2), so `fits` is monotone: ∃ a minimal fitting tier
`t* = min{ t ∈ 𝒯 : fits(t) }` — the search's target.

**Probes.** A probe at tier t computes `C(t)` via the **single-pass**
closure (Φ applied once — the historical one-shot operator), so accept
decisions are conservative: single-pass counts ≥ fixpoint counts, and
probe 0 is bit-identical to the one-shot output. The quickselect
landing `λ_raw = w₍c₎ + 1` (w₍c₎ = c-th largest order statistic of
{w(e)}) is canonicalized to `ι₀ = min{ i : 𝒯_i ≥ λ_raw }` — cones
depend only on tier membership, so `E(λ_raw) = E(𝒯_{ι₀})` and every
probe is a tier weight.

**Bracket invariant.** Maintain indices `lo, hi` with
`fits(𝒯_lo) ∧ ¬fits(𝒯_hi)` (hi may be a virtual boundary M+1).
Initialize `lo = ι₀` (fits by the one-shot guarantee `C(𝒯_{ι₀}) < c`).
Gallop probes `lo+1, lo+2, lo+4, …`; on the first overshoot `hi` is
set, then `mid = ⌊(lo+hi)/2⌋` bisection — each probe either moves lo up
(fits) or hi down (not), strictly shrinking `hi − lo` while preserving
the invariant. Two quirks:

- **Revival**: C can be 0 at a tier and > 0 lower (Theorem 2's
  monotonicity does not bound C away from 0). An empty probe therefore
  counts as *fitting* for the bracket; only the final answer must be
  non-empty (else `declined` — the cone is returned unfloored).
- **Truncation**: at `max_probes` (default 8) with `hi − lo > 1`, the
  returned t\* = 𝒯_lo is cap-valid but possibly above the true minimum;
  `truncated` is reported.

**Termination cases.** `hi = lo + 1` ⟹ `t* = 𝒯_lo` and
`¬fits(𝒯_{lo+1})` certifies maximality (`budget_fully_used`); gallop
reaches the bottom tier fitting ⟹ `floor_skipped` (the full closed
cone fits the cap; the one-shot's pre-closure gate would have floored
anyway — the over-flooring fix). Cost: ≤ max_probes probes ×
Θ(B·|E₀|); the kept set is exactly `Φ¹(E(t*))` (single-pass), reported
with `residual_slack = cap − C(t*)`.

### 5.6 The complete enumerators (mechanics)

All yield the full simple-path set 𝒫; they differ in pruning state and
memory shape.

- **MemoizedDFS** (default `direction='forward'`): memo
  `Valid(u,k) = { v ∈ succ(u) : 𝒲_k(v) ≠ ⊥ }` (mutual recursion,
  `None` = dead state), keyed (node, remaining-depth) and **shared
  across all lengths**; reconstruction walks only memoized successors,
  skipping on-path nodes. Space O(L·E) worst (a successor list per
  (u,k)). The backward variant runs on predecessors from T and reverses
  paths on yield — the memo then covers only the target-reachable cone,
  best when |T| ≪ |S|.
- **DP** (`find_paths_backward_dp`): layer sets `R_0 = T`,
  `R_k = N⁻(R_{k−1})` (predecessor image — no reversed copy); the
  guided DFS from s at remaining depth k steps only to `v ∈ R_{k−1}` —
  one set-membership test per edge, O(L·|V|) space, the lowest of the
  pruning algorithms; degenerates when the R_k cover V (deep dense
  queries).
- **MeetInMiddle**: per length ℓ, forward DFS to depth `m = ⌊ℓ/2⌋`
  fills a map `end ↦ {half-paths}` (rebuilt for every ℓ); backward DFS
  from T walks ℓ−m and emits every stitched pair with
  `|set(π_f) ∩ set(π_b)| = 1`. The b^{ℓ/2} branching win is offset by
  the per-length rebuild, the unpruned forward walk, and the pairwise
  join (measured, see the evaluation doc).
- **Bidirectional**: parent-set layers `F_d, Bd` (d = 0..L) from both
  sides; per ℓ, meet at `mid = ⌊ℓ/2⌋` and enumerate all
  simple forward×backward pairs. Emits shortest lengths first; layer
  maps dominate memory (measured 892 MB at L5).
- **Backtracking**: iterative deepening — for ℓ = 1..L a full backward
  DFS to exactly depth ℓ; O(L) stack, no memo; recomputes shared
  prefixes, trading CPU for minimal memory.

## 6. Measured performance (FAFB v783, weight ≥ 3)

Validation harnesses: `scripts/verify_budget_fit_pruning.py` (synthetic
+ brute force, 11/11), `scripts/compare_budget_fit_real_data.py`
(prototype comparison incl. million-scale), and
`scripts/verify_production_real_data.py` (production code paths).

| Scenario | Cone after lossless prune | Result |
| --- | --- | --- |
| 6 targets, L2/L3 | 204 / 23,967 rows | Edge Budget never fires — lossless prune suffices |
| 1,756 Tm3 targets, L3 (E₀ 289k), cap 50k | one-shot t0=12 → 2,567 kept, **3,758 paths** | fit t\*=6 → 28,025 kept, **154,134 paths** (τ=7, = the no-floor run) — **41×** |
| same, production `fit_edge_budget` | t\*=6 identical, 7 probes, 1.4s; ≥ one-shot utilization **10.9×** | pipeline floors under a 20k cap; shortest untouched |
| 3,337 Tm3+Mi1 targets, L4 (E₀ 2.05M), cap 1M | one-shot w0=6 → 295k (70% slack), 0.94s | fit t\*=5 → 536k, 2.99s (+2s); linear cost, no explosion; at cap 50k the one-shot declined entirely while the fit honored the cap |
| shortest, real graph (289k edges) | **413,115 min-hop paths**, set-identical to the backward enumerator; descending emission; budget-500 bite → τ=10, 646 = exactly `{bn ≥ τ}` | |

Unit/property coverage: `tests/core/test_hop_budget_pruning.py`
(losslessness vs naive enumeration), `test_edge_budget_floor.py`
(one-shot contract), `test_budget_fit_and_shortest_sf.py` (§7.4/§7.2
invariants), `test_pathfinding.py` (pipeline integration). Full core
suite: 3,599 passed.

## 7. Choosing knobs (practical)

- **Few targets, shallow** (≤ ~10 targets): defaults suffice — the
  lossless prune does the work; both budgets stay idle.
- **Broad targets / deep searches ('all')**: keep `graph_edge_limit_bodyid`
  ≈ 1M (floors the cone honestly) and rely on `max_paths_bodyid = 0`
  (auto 1M) for the output; read τ to know what you got. Raise
  `min_synapse_num` for the strongest effect.
- **Shortest at depth ≥ 5–8**: set **Max Paths (BodyId)** on the
  Shortest tab — it is the only bound that mode has (by design).
- Reading a run: `edge_weight_floor` = the strength the cone was
  floored at; τ = the strength the *path list* was cut at; W\* = your
  best route (never removed); effective cutoff = max(τ, t\*).

### Filter and cap distinctions

Five different knobs bound what a run computes vs. shows — do not
conflate them:

- **Min Synapse Count** (`min_synapse_num`): threshold — edges below it
  never enter the graph.
- **Edge Budget** (`graph_edge_limit_bodyid`): graph-level lossy weight
  floor (floor w0 above the landing tier w1), 'all' mode only — exactly
  equivalent to raising the threshold; reported as `edge_weight_floor`
  (w0) / `edge_budget_landing` (w1); shortest mode is never floored.
- **Max Paths (BodyId)** (`max_paths_bodyid`): path-output budget
  (StrongestFirst), both modes — the graph is not trimmed.
- **Drop Untyped Neurons** (`drop_untyped`): neuron-label filter
  (Stage 1b) — rows touching an untyped label are dropped after
  enrichment, before the graph is built.
- **Visualization Edge Limit** (`edgeN_limit`): DRAWING-ONLY cap on the
  unique edges rendered per HTML view — never changes fetching, the
  graph, or path outputs; a single complete path may exceed it to stay
  intact.

## 8. Design decisions and rejected alternatives

- **Exact "keep only edges on a simple path" is NP-complete** (directed
  2-vertex-disjoint-paths) — the fixpoint hop test is the practical
  ceiling; the post-build node prune becomes redundant at fixpoint
  (noted for future demotion).
- **The literal "released-budget" ledger diverges** (the +1 landing
  releases ≥ 1 unit every round → allowance → ∞, cap abandoned). The
  fixed-cap search `t* = min{t : C(t) ≤ cap}` with bisection is the
  convergent form; the slack-spending ratchet survives only as the
  gallop's warm start (accept only if `C ≤ cap`, else bisect).
- **Per-tier partial admission** (splitting a weight tier to fill the
  cap exactly) is rejected: it breaks τ-equivalence — the Fix C line.
  Residual slack `cap − C(t*)` is inherent to threshold semantics.
- **Per-layer strongest-k beam** is rejected as a default: lossy without
  a τ certificate (recorded in §7.3 of the audit); if ever wanted, ship
  as an explicitly approximate mode.
- **Probes use the single-pass closure** (not the fixpoint): accept
  decisions stay conservative (single-pass counts ≥ fixpoint counts)
  and probe 0 remains bit-identical to the historical one-shot; the
  fixpoint remains an optional pre-stage.

## 9. File and test map

| Concern | Location |
| --- | --- |
| Pipeline orchestration, pruning, budget-fit, untyped filter, provenance | `src/coana.py` (`_find_paths_core`, `_graph_edge_frames`, `prune_layers_hop_budget`, `_hop_budget_pass_once`, `fit_edge_budget`, `apply_edge_budget_floor`, `_discover_shortest_backward`, `drop_untyped`, `_finalize_threshold_provenance`, `_write_run_metadata`); shared formula `applied_threshold_provenance` in `src/utils/threshold_state.py` (re-exported by coana, delegated to by the comparison analyzer) |
| Untyped predicate | `src/utils/label_utils.py` (`is_untyped_type_label`; also delegated to by `ComparisonAnalyzer._is_untyped_type_value`) |
| Graph + enumerators | `vispath-subproject/src/vispath_pkg/fast_graph_core.py` (`find_paths_strongest_first`, `find_paths_shortest_strongest_first` — the pipeline's shortest-mode enumerator; `find_paths_shortest_backward` exists but is not called by the pipeline; complete enumerators); shared core `strongest_core.py` |
| UI | `ui/tabs/find_path.py`, `ui/tabs/find_shortest.py` (Max Paths and Drop Untyped Neurons fields), `ui/config.py` DEFAULTS, `ui/runner.py` TOOL_REGISTRY |
| Validation | `tests/core/test_hop_budget_pruning.py`, `test_edge_budget_floor.py`, `test_budget_fit_and_shortest_sf.py`, `test_pathfinding.py`; `scripts/verify_budget_fit_pruning.py`, `scripts/compare_budget_fit_real_data.py`, `scripts/verify_production_real_data.py` |
| Design history | `_plan/plan-pathfinding-doc-audit.md` (§7 planned + status), `_plan/plan-sf-only-edge-budget.md`, `_plan/plan-cross-dataset-pathfinding-optimization.md` |
