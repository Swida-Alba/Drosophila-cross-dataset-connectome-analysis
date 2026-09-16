# StrongestFirst Pathfinding — Technical Report

**Status:** implemented (2026-09-04; revised 2026-09-05 and 2026-09-06).
Scope: the unified path-bounding stack for `FindAllPath` /
`FindShortestPath` and every consumer that ranks or limits paths
(cross-dataset comparison, Complete Paths tab, path visualization).
2026-09-05 revision: **StrongestFirst is the only selectable 'all'-mode
algorithm** (the UI selector was removed; the `pathfinding` parameter
stays for scripts/tests), the retired edge-limit knob was repurposed as
the lossy **Edge Budget** floor (Fix D), and complete runs report their
natural tau.
2026-09-06 revision: the Edge Budget landing was refined into the
**budget-fit search** (`fit_edge_budget` — weakest tier whose closed
cone fits the cap), the lossless prune **iterates to a fixpoint**, the
shortest mode runs **StrongestFirst with a path budget**
(`find_paths_shortest_strongest_first`, Shortest-tab Max Paths field),
and the comprehensive re-query mode deduplicates cross-layer pairs.
2026-09-07 revision: every run writes an **applied-threshold provenance
block** (`applied_threshold` / `applied_threshold_source` /
`tau_canonical` / … — §4c) to its run metadata, and the
**Drop Untyped Neurons** filter (`drop_untyped`, default on) runs in
the shared pipeline for both modes after enrichment and before graph
construction.
See [PATHFINDING_PIPELINE.md](PATHFINDING_PIPELINE.md) for the
end-to-end report.
Companion plans: `plan-cross-dataset-pathfinding-optimization.md`
(implementation), `plan-cross-dataset-report-fixes.md` (Fix A cap
policy, Fix C trim unification, §6b/§6c),
`plan-sf-only-edge-budget.md` (this revision).

---

## 1. The conceptual model

Two different knobs bound a pathfinding run, and they are not the same
kind of object:

| | manual threshold (`min_synapse_num`) | StrongestFirst budget (`max_paths_bodyid`) |
|---|---|---|
| what it trims | the bodyId **graph**, a priori | the ranked **path list**, a posteriori |
| when | before/during discovery | during enumeration |
| chosen by | the user | emerges from the data |
| effect | removes edges (and everything behind them) | removes whole weak paths, keeps the strongest |

Because StrongestFirst ranks paths by **bottleneck** (the minimum edge
weight along a path — a route is as strong as its weakest hop) and stops
at the budget with a fully drained tie group, a budgeted run satisfies an
exact equivalence:

> **τ-equivalence.** `StrongestFirst(threshold=t, budget=B)` with
> reported cutoff τ produces exactly the same path set as a *complete*
> run at `min_synapse_num = τ` (same query, same hop bound).
> *Proof:* the emitted set is `{paths : every edge ≥ τ}` by the drain
> semantics; a complete run at threshold τ enumerates precisely that set,
> since every such path is also a valid ≥t path and nothing weaker
> survives. ∎

So a low threshold with capping in a large graph **behaves as a raised
threshold** — never as a blind trim — and the run reports which raised
threshold it corresponds to. τ is data-dependent (not user-chosen);
users who want a specific τ raise the threshold or the budget.

**The five knobs, and what each one actually bounds:**

| knob | bounds | kind |
|---|---|---|
| Min Synapse Count (`min_synapse_num`) | the graph | threshold — edges below it never enter the graph |
| Edge Budget (`graph_edge_limit_bodyid`) | the graph | lossy weight floor (floor w0 above the landing tier w1), 'all' mode only — exactly equivalent to raising the threshold; shortest mode is never floored |
| Max Paths (BodyId) (`max_paths_bodyid`) | the path list | StrongestFirst output budget, both modes — the graph is not trimmed |
| Drop Untyped Neurons (`drop_untyped`) | neuron labels | filter — rows touching an untyped label are dropped after enrichment, before the graph is built (both modes) |
| Visualization Edge Limit (`edgeN_limit`) | the drawing | drawing-only cap on unique edges rendered per HTML view — never changes fetching, the graph, or path outputs; a single complete path may exceed it to stay intact |

## 2. Pipeline layers (where each mechanism runs)

Everything happens at the **bodyId level**; type-mapped outputs are
downstream projections of the bodyId path set (type-level aggregation
before pathfinding would create phantom type paths — the "bundle
effect"; see TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md).

1. **Discovery** (Phase 1): layer tables fetched from the
   threshold-filtered connection cache — cones are nested across
   thresholds (cone(t=10) ⊆ cone(t=3)).
2. **Untyped-neuron filter** (`drop_untyped=True`, both modes): after
   label enrichment and before graph construction, connection rows
   whose pre- or post-side label is untyped (empty, an
   Unknown/None/NaN sentinel, or all-digit —
   `utils.label_utils.is_untyped_type_label`) are dropped, so an
   untyped neuron can never be an intermediate node of a returned
   path. Drops are exported to
   `data_details/untyped_dropped_records.csv` (with `untyped_side`)
   and an `[untyped dropped]` note only when rows were dropped; the
   flag is part of the FindAllPath graph-cache key
   (`_findallpath_cache_key`).
3. **Lossless hop-budget pruning** (`prune_layers_hop_budget`): drops
   edges with `dist_S(u) + 1 + dist_T(v) > max_interlayer + 1` — they
   cannot lie on any admissible path. Lossless by proof; measured
   39–77% cone reduction on real queries. The pass **iterates on the
   already-pruned tables until a pass drops nothing** (capped passes):
   still lossless — every admissible path survives every pass — and
   strictly tighter than a single pass, since distances recomputed on
   the pruned graph can only grow.
4. **Dead-end node pruning**: nodes that cannot reach any target
   (post-pruning) are removed. Also lossless. Both passes report a
   **strongest-retained bottleneck** — the widest-path maximin value
   W\* = max over source→target paths of the min edge weight — which is
   *identical* before and after the passes (that is what lossless
   means), and is printed with the note *"top paths unchanged"*.
5. **StrongestFirst enumeration** (`find_paths_strongest_first`): A*-style
   best-first on the prefix bound `min(running bottleneck,
   W[remaining][node])`. Emits complete intact paths in descending
   bottleneck order; stops at the budget and drains ties at τ.
6. **Enrichment / type-path derivation / saves / visualization**: unchanged
   downstream stages; per-threshold enrichment denominators are computed
   exactly as before.

**Retired then repurposed (Fix C → Fix D):** the lossy bodyId edge trim
(`_trim_edges_with_path_integrity` via `_graph_edge_frames`) no longer
runs in FindAllPath/FindShortestPath — it bounded the graph but not the
paths, was not nested across thresholds, and could silently remove the
weakest hop of a top path. The arbitrary-order `max_paths_bodyid`
truncation for legacy enumerators is likewise gone: any budget routes
through StrongestFirst. **Fix D (2026-09-05, refined by the budget-fit
search)** repurposed the former knob as the **Edge Budget**: after the
lossless prunes, a cone exceeding the budget is floored by
`fit_edge_budget` at the weakest tier whose closed cone fits the cap —
see §4b. The old `_trim_bodyid_edges` top-N trim was deleted;
`_trim_edges_with_path_integrity` survives only for the FindNetwork
type-level group trims.

## 3. Guarantees

- **No budget bite ⇒ identical set** to complete enumeration (only the
  order differs). Unit-tested against MemoizedDFS.
- **Budget bite ⇒ `all intact paths with bottleneck ≥ τ`** — ties at τ
  fully drained; never an arbitrary subset. τ and the fact of bounding
  are recorded in `user_warning_notes.txt` and `all_attributes.json`
  (`strongest_first_cutoff`, `trim_policy`).
- **Determinism**: same graph ⇒ same emission sequence (stable tie
  rules).
- **Per-threshold comparability**: discovery cones and unbounded runs
  are nested across thresholds; budgeted runs are compared at equal τ
  (`tau` / `paths_complete` columns in `threshold_sensitivity.csv`).

## 4. Efficiency profile

| mechanism | bounds | measured cost |
|---|---|---|
| lossless hop-budget pruning | graph (exact) | O(E); 39–77% of discovery edges removed on real queries |
| strongest-first budget | paths + search | work ≈ output + frontier above τ |
| retired edge trim | graph (lossy) | bounded edges but NOT paths; non-nested across thresholds |

Benchmarks (BANC-era baseline, since re-verified on male-cns):
complete enumeration 533 s / 3.6 GB → StrongestFirst + pruning +
`skip_bodyId=True` 19.5 s / 11 MB on the same query; sensorimotor
long-path matrix (JO→VNC motor, JO→DN, visual→DN at L4–L5): StrongestFirst
54–186 s where complete MemoizedDFS exceeded 300 s.

Worst case (budget never reached, no ties): cost ≈ complete enumeration
plus one O(cutoff·E) DP — never asymptotically worse.

## 4b. Reading the two numbers: W\* (pruning ceiling) and τ (budget)

A run reports two different strength numbers — do not confuse them:

- **W\* — strongest retained path bottleneck** (pruning note,
  §6b): computed BEFORE enumeration, after the lossless prunes. It is
  the widest-path maximin value of the whole cone — the strength of the
  single best source→target route. It is the **ceiling**: no path can be
  stronger than W\*, and because both prunes are lossless, W\* is
  identical before and after them ("top paths unchanged"). Any manual
  threshold ≤ W\* leaves this best path intact; thresholds above W\*
  would remove the best route.
- **τ — budget cutoff** (StrongestFirst note): where the path budget
  landed, if it landed. The output contains all intact paths with
  bottleneck ≥ τ; τ ≤ W\* always.

Reading them together: W\* = "your best route is this strong, and
pruning preserved it"; τ = "you kept everything down to this strength".
A complete run (budget not bitten) reports its natural τ = the weakest
path's bottleneck with `paths_complete = true` — meaning every
threshold up to that value yields the identical set.

**Implemented (Fix D, 2026-09-05; refined by the budget-fit search):**
the Edge Budget floor. After the lossless prunes, if the pruned cone
still exceeds the Edge Budget (`graph_edge_limit_bodyid`, default 1M;
0/None = off) a floor is applied. The original one-shot landing located
the N-th strongest edge weight w1 by quickselect and dropped every edge
with `weight < w0 = w1 + 1` — the **+1 was load-bearing** (edges
strictly heavier than w1 number fewer than N, so the floored cone
always fit the budget), but it *discarded the boundary-tie tier* and
whatever slack the closure left unused, which on real connectomes can
be most of the cap (measured: 95–98% waste; in one case the one-shot
declined entirely where the fit search delivered a valid floor).

The **budget-fit search** (`fit_edge_budget`) now floors at the
*weakest tier whose lossless-closed cone fits the cap*: it probes tiers
downward from the one-shot landing (gallop, then bisect between the
last fitting and first overshooting tier, under a probe budget of 8)
with single-pass closures, and floors at the winner. Every probe is a
pure threshold raise, so the floored run is exactly a complete run at
`min_synapse_num = t*` — **effective cutoff = max(τ_budget, t\*)**,
reported as `edge_weight_floor` / `edge_budget_landing` in the run
attributes, `parameters.txt`, and a lossy-floor note together with the
residual slack, probe trace, and whether the budget is fully used (the
next weaker tier exceeds the cap). When even the full closed cone fits
the cap, no floor is applied (`floor_skipped`); degenerate cones (a
single weight tier, or a budget below the distinct-weight support)
revert with an honest note instead of returning an empty graph.
Shortest mode is never floored.

## 4c. Applied-threshold provenance (2026-09-07)

Every pathfinding run — BOTH modes; shortest was previously never
re-stamped — writes a provenance block to `parameters.txt`,
`all_attributes.json`, AND `data_details/parameters.csv` after
enumeration, computed by `applied_threshold_provenance()` (shared in
`src/utils/threshold_state.py`; re-exported by `src/coana.py` and
delegated to by `ComparisonAnalyzer._applied_state_for`) and
finalized by `_finalize_threshold_provenance` +
`_write_run_metadata`:

| field | meaning |
|---|---|
| `requested_threshold` | the user-entered Min Synapse Count before any budget effect |
| `applied_threshold` | the **canonical equivalent threshold** that reproduces the materialized set: the requested threshold for a complete/unbounded run (natural τ reported separately), or `tau_canonical` (`w2+1` / landing τ) when a lossy budget affected the output. **No `max(canonical, requested)` clamp** — the pipeline order makes it structurally ≥ the request (rev 2, 2026-09-11) |
| `applied_threshold_source` | `requested` \| `strongest_first_budget` \| `edge_budget` \| `strongest_first_budget+edge_budget` |
| `strongest_first_budget` | the effective path budget (0/auto → 1,000,000) |
| `strongest_first_budget_bitten` | whether the budget actually bit |
| `strongest_first_tau` | the landing τ (collapse bound); for a complete run, the natural weakest emitted-path bottleneck |
| `tau_canonical` | the minimal equivalent threshold: `w2+1` when the bite leaves a gap `[w2+1, τ]`, else the landing/natural τ. This IS `applied_threshold` on a budgeted slice |
| `strongest_dropped_bottleneck` | w2 — the strongest path NOT emitted after a bite |
| `edge_budget` / `edge_budget_applied` | the cap and whether the floor is **binding for this threshold** (`w0 > requested`); a floor at or below the request leaves the output untouched |
| `edge_budget_landing` (w1) / `edge_weight_floor` (w0) | the tier that determined the floor and the floor itself |
| `strongest_retained_bottleneck` | W\* — the widest-path ceiling after lossless pruning (§4b) |
| `paths_complete` | true exactly when no lossy budget actually affected THIS threshold's output |

**Semantics (rev 2).** τ (`strongest_first_tau`) is a **landing/collapse**
bound; `tau_canonical` is the **minimal equivalent threshold** when a gap
exists below it. `applied_threshold` **is** that canonical equivalent (or
the requested threshold for a complete run) — there is deliberately no
clamp. The pipeline order — filter edges by the requested threshold, then
the edge budget, then build the graph, then StrongestFirst — guarantees a
per-threshold run's canonical is at or above its own request: a bitten
slice drops only paths with bottleneck ≥ requested (so `w2+1 > requested`),
a binding floor has `w0 > requested`, and a complete slice's natural τ is
`≥ requested`. A canonical **below** the request therefore means the record
inherited a *lower* threshold's provenance; fix that at the source (the
multi-threshold replay's per-slice metadata), never by clamping the label.

The Edge Budget floor is **per-threshold**: `edge_budget_applied` (alias
`edge_floor_binding`) is true only when `w0 > requested_threshold`. When
the floor sits at or below the request the floored graph reproduces the
unfiltered graph's output at that threshold, so the run is untouched
(`paths_complete = true`). The graph-level floor (`edge_weight_floor` /
`edge_budget_landing`) and its pruning record remain recorded as
diagnostics regardless. A replayed slice is labeled from its own slice
state: a slice at/above the t0 floor is a complete run at that value
(floor inert, `paths_complete = true`); a slice below the landing τ0
derives its own dropped bottleneck/canonical.

The bottleneck is the minimum edge weight along a path,
so a budgeted output is a **strength-bounded path set** — never an
arbitrary first-N truncation. `parameters.txt` keeps the
backward-compatible alias lines `applied_tau (min path bottleneck)` and
`edge_weight_floor`. Replay folders materialized by
`_replay_output_folder_for_threshold` (`minsyn_{t}`) and the
Cross-Dataset raw-run folders selected by standard or combination query rows
carry the same block.
Shortest mode may take the StrongestFirst budget but **never** the Edge
Budget floor. A `[combined threshold]` note is appended to
`user_warning_notes.txt` when BOTH budgets affected a run.

## 5. API reference

- `FastGraph.find_paths_strongest_first(sources, targets, cutoff,
  budget=None, per_pair_k=None, stats=None, verbose=False)` — budgeted
  strongest-first enumeration. `stats` receives `emitted`, `tau`,
  `budget_bitten`, `last_bound`.
- `FastGraph.find_paths_shortest_strongest_first(targets, sources,
  cutoff=None, budget=None, stats=None, target_cutoffs=None,
  verbose=False)` — §7.2: budgeted best-first over each target's
  shortest-path DAG with a per-target maximin DP; per-target streams are
  k-way merged and the global budget drains ties at τ. A bitten run is
  exactly "all min-hop paths with bottleneck ≥ τ"; unbitten runs equal
  `find_paths_shortest_backward` (a separate enumerator that exists in
  the codebase but is **not** called by the pipeline) as a set.
  `stats` receives `emitted`,
  `tau`, `budget_bitten`, `strongest_dropped`, `per_target`.
- `FastGraph._widest_path_backward(targets, cutoff)` — delegates to
  `strongest_core.widest_path_backward(adj, targets, cutoff)`.
- `vispath_pkg/strongest_core.py` — the shared core:
  `widest_path_backward`, `path_bottleneck`, `interior_strength`,
  `path_rank_key`, `selection_threshold`, `drain_budget`.
- `coana.prune_layers_hop_budget(conn_layers, sources, targets, bound, …,
  max_passes=4)` — the lossless pass, iterated to a fixpoint (a probe of
  the budget-fit uses `max_passes=1`, the historical single pass);
  returns `(pruned_tables, stats)` with `strongest_retained` and
  `passes`.
- `coana.fit_edge_budget(conn_layers, budget, sources, targets, bound, …,
  max_probes=8)` — the budget-fit search: floors at the weakest tier
  whose closed cone fits the cap; returns `(floored_tables, stats)` with
  `applied`, `floor` (t\*), `landing` (first overshooting tier weight),
  `edges_after`, `residual_slack`, `budget_fully_used`, `probes`,
  `truncated`, `floor_skipped`, `search_seconds`, `probe_trace`,
  `strongest_retained`. Never mutates its inputs; reverts honestly when
  no non-empty tier fits.
- `coana.apply_edge_budget_floor(conn_layers, budget, sources, targets,
  bound, …)` — the original one-shot Fix D floor (w0 = w1 + 1, single
  lossless pass), retained as the probe-0 reference and for tests; the
  pipeline uses `fit_edge_budget`.
- `ComparisonParameters.max_paths_bodyid` — `None`/0 = auto
  (StrongestFirst: internal 1M budget in 'all' and 'shortest' modes;
  legacy enumerators: unbounded); `>0` = that budget, always routed
  through StrongestFirst.
- `ComparisonParameters.graph_edge_limit_bodyid` — the **Edge Budget**
  (default 1M in the UI; `None`/0 = off for API callers). Lossy floor,
  'all' mode only.
- `FindNeuronConnection.drop_untyped` (default `True`) — the
  neuron-label filter applied in `_find_paths_core` for both modes
  after enrichment and before graph build (see §2 layer 2); shared
  predicate `utils.label_utils.is_untyped_type_label`; drops recorded
  in `data_details/untyped_dropped_records.csv` (+ `untyped_side`);
  part of the FindAllPath graph-cache key (`_findallpath_cache_key`).
  Cross-Dataset Comparison delegates to the same predicate but filters
  post label-mapping (`ComparisonAnalyzer._drop_untyped_neurons`;
  the per-dataset runs use `drop_untyped=False`).
- Run metadata: `all_attributes.json`, `parameters.txt`, and
  `data_details/parameters.csv` record the applied-threshold
  provenance block (§4c: `requested_threshold`, `applied_threshold`,
  `applied_threshold_source`, `strongest_first_budget`,
  `strongest_first_budget_bitten`, `strongest_first_tau`,
  `tau_canonical`, `strongest_dropped_bottleneck`, `edge_budget`,
  `edge_budget_applied`, `edge_budget_landing`, `edge_weight_floor`,
  `strongest_retained_bottleneck`, `paths_complete`) plus
  `strongest_first_cutoff` (budget τ or natural τ) and the
  `graph_pruning_record`
  (lossless passes + the floor); `parameters.txt` keeps the alias
  lines `applied_tau (min path bottleneck)` and `edge_weight_floor`.
- `ComparisonAnalyzer._path_taus[(dataset, threshold)]` — per-run τ for
  the sensitivity export; `_path_run_meta[(dataset, threshold)]` adds
  `applied_folder`, `edge_weight_floor`, `skipped`, `duplicate_of`.

## 6. Verification

- Property tests: strongest-first set ≡ complete set without a budget;
  budgeted set ≡ `{bottleneck ≥ τ}` (brute-force checked on randomized
  graphs); lossless pruning keeps exactly the admissible path universe.
- Golden-master: type-level outputs byte-identical with/without the
  optimizations; per-threshold sets nested (`threshold_sensitivity.csv`
  monotone).
- Real-data matrix (male-cns:v1.0 + FAFB v783): JO→VNC motor, JO→DN,
  visual→DN at L4–L5 — see
  `plan-cross-dataset-report-fixes.md` §2b and the sensorimotor harness
  under `local_data/sensorimotor_optim_test/`. BANC is operational for
  connectivity and pathfinding (standalone local-release path, covered by
  `tests/e2e/test_banc_ame12_ppl_e2e.py` with current-data aMe12→PPL runs);
  that sensorimotor matrix simply predates the BANC integration and has not
  been rerun on it. The only deferred BANC features are morphology
  similarity/comparison (vector-quality validation pending).

## 7. Decision log

- 2026-09-05: **StrongestFirst-only UI** — the Algorithm selector was
  removed from the Complete Paths and Cross-Dataset tabs (and Settings);
  the `pathfinding` parameter stays honored for scripts/tests so the
  complete-enumeration test coverage keeps working.
- 2026-09-05: **Fix D Edge Budget implemented** with w0 = w1 + 1 (the
  strict floor guarantees kept edges < N even under a massive tie tier;
  the earlier plan's plain top-N landing kept all boundary ties).
  Degenerate guards: single-tier cones and below-support budgets revert
  with a note. Auto-adaptive escalation (raise w0 on a wall-clock guard)
  remains deferred.
- 2026-09-05: **F5 τ-folder discipline** — replayed multi-threshold
  runs materialize only real thresholds (a fresh tau-folder with tau
  denominators + every asked t > tau); collapsed asked thresholds alias
  the tau folder and the intermediate minsyn_{t0} folder is removed.
- 2026-09-05: **F9 ratio/probability filters retired** — both are
  readout columns (connection_ratio now uses the threshold-free ALL-POST
  denominator); the UI entrances are hidden and the pipeline ignores the
  parameters with a notice. Rationale: the threshold-based denominator
  made ratios jump across thresholds and broke cone nesting when the
  filters were active.
- 2026-09-04: budget routing overrides the Algorithm selector when a
  positive budget is set (Fix C) — one trimming mechanism, no arbitrary
  truncation.
- 2026-09-04: lossy edge trim retired from FindAllPath/FindShortestPath
  (legacy `FindPath` keeps its local behavior; the tabs do not use it).
- 2026-09-04: interior-weakest-hop preference kept only as the display
  tie-breaker in the visualization selector; enumeration ranks by the
  full-path bottleneck.
