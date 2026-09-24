# How Pathfinding Aggregates at the Type Level

*Prepared: 2026-08-22 — an audit of how the DROCAT toolkit aggregates neuron-level paths into neuron-type-level paths, and the trade-offs of that design.*

## TL;DR

The toolkit **does not run pathfinding on a type-level graph.** Instead it runs pathfinding first at the *bodyId* (single-neuron) level, then **derives** the type-level paths by mapping each discovered neuron path to its sequence of neuron types.

Aggregation happens in two separate places and plays two different roles:

1. **Edge-weight aggregation** (`aggregate_by_label`) — sums synapse weights over all neuron pairs belonging to a type pair. This produces the type-level *network graph* (which types connect, and by how much).
2. **Path derivation** (`_derive_type_paths_from_bodyid_paths` in `src/coana.py`) — projects each discovered neuron path onto a type sequence and keeps only sequences whose every hop is a real, in-path connection.

The key design decision is that **paths are derived, never re-searched**, at the type level. This deliberately rejects the "aggregate all neuron pairs → run pathfinding on the type graph" shortcut because it produces largely phantom paths. The rest of this report explains why, how the code does it, and the resulting pros and cons.

---

## 1. Where the pieces live

| Concern | Location |
|---|---|
| Type-level weight aggregation (neuron pairs → type edges) | `vispath-subproject/src/vispath_pkg/fast_graph_core.py` — `FastGraph.aggregate_by_label` |
| bodyId→type label map, path pipeline driver | `src/coana.py` — `FindNeuronConnection` (`FindAllPath`, `FindShortestPath`) |
| **Deriving type-level paths from discovered neuron paths** (the core logic) | `src/coana.py` — `FindNeuronConnection._derive_type_paths_from_bodyid_paths` |
| Pair-wise shortest-path retention before type mapping | `src/coana.py` — `FindNeuronConnection._keep_shortest_bodyid_paths` |
| bodyId-level vs type-level path filters (ratio / traversal probability) | `src/coana.py` — `_apply_bodyid_level_filters`, `_apply_type_level_filters` |
| The five selectable pathfinding algorithms | `fast_graph_core.py` — `find_paths_*` |
| Authoritative design rationale for derivation-first | `docs/technical/TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md` |

---

## 2. Conceptual setup: two levels of graph

A **neuron type** is a class label shared by many individual neurons (bodyIds). This single fact drives everything.

- **bodyId-level graph `G`**: nodes are individual neurons (bodies). An edge `(aᵢ, bⱼ)` means neuron `aᵢ` makes `w` synapses onto `bⱼ`. This is the "ground truth" connectivity.
- **type-level graph**: nodes are neuron types. An edge `A → B` aggregates *every* neuron pair `(aᵢ, bⱼ) where aᵢ has type A and bⱼ has type B`, summing their synapse weights.

The question the toolkit answers is: given a set of source *types* and target *types*, what routes connect them? The crux is whether you answer that on `G` or on the type graph. The toolkit answers it on `G` and only *reports* the type-level answer.

### 2.1 The edge-weight aggregation: `aggregate_by_label`

```python
def aggregate_by_label(self, label_map, return_edge_df=False):
    aggregated = FastGraph()
    label_weights = {}  # (label_pre, label_post) -> total_weight
    for u in self.adj:
        label_u = label_map.get(u)
        if label_u is None:
            continue
        for v, w in self.adj[u].items():
            label_v = label_map.get(v)
            if label_v is None:
                continue
            key = (label_u, label_v)
            label_weights[key] = label_weights.get(key, 0.0) + w
    for (label_u, label_v), total_w in label_weights.items():
        aggregated.add_edge(label_u, label_v, total_w)
    ...
```

This is a straightforward **many-to-one aggregation**: all neuron pairs that share a `(type_pre, type_post)` label are collapsed, their synapse counts summed into a single edge weight.

**Critical caveat this function does NOT encode:** the resulting type edge `A → B` = "types A and B connect" is true, but it says *nothing* about *which* neurons were involved. That is precisely the gap the derivation step below exists to fill. `aggregate_by_label` is still genuinely useful for type-topology visualization (sankey/heatmap/network views) and for the connectivity profiler; it's only the *path* answer that must come from neuron-level discovery.

### 2.2 The core: deriving, not re-searching, type-level paths

`_derive_type_paths_from_bodyid_paths(src/coana.py:10192)` is the heart of the aggregation:

```python
def _derive_type_paths_from_bodyid_paths(self, all_paths, node_label,
                                         kept_type_edges, source_types,
                                         target_types, verbose=False):
    source_set = set(source_types)
    target_set = set(target_types)
    seen = set()
    for p in iterator:
        seen.add(tuple(node_label(n) for n in p))   # map each neuron path -> type sequence
    out = []
    for seq in seen:
        if seq[0] not in source_set or seq[-1] not in target_set:
            continue
        if all((seq[i], seq[i + 1]) in kept_type_edges
               for i in range(len(seq) - 1)):
            out.append(list(seq))
    return out
```

The steps:

1. Take `all_paths`, the set of discovered **neuron** paths (each is a list of bodyIds).
2. Map each path to its **type sequence** via `node_label` (a `bodyId → type` callable), collecting the *unique* type sequences in `seen`.
3. Keep a type sequence only if it
   - starts at a **queried** source type and ends at a **queried** target type, and
   - has **every consecutive type hop present** in `kept_type_edges` (the type-edge table produced by aggregation), as a defensive label-consistency check.

The `kept_type_edges` membership check is a *verification*, not a search: it guards against label-mismatch bugs but doesn't introduce new paths. No pathfinding ever runs on the type graph. Because the derivation is literally the type projection of an already-found path, **every reported type path is the type sequence of a real chain of neurons.**

### 2.3 Repeat-type routes survive — a concrete gain

Because the paths are projected from real neuron chains, routes that revisit a type (`A → B → A`) are kept — as long as a concrete neuron realization exists (e.g. neurone B₁ receives from a type-A neuron, and a *different* type-A neuron B₂ projects onward). A simple-path search on the type graph would drop these silently.

### 2.4 Shortest-paths mode uses pair-wise shortest, not a global cutoff

`_keep_shortest_bodyid_paths(src/coana.py:10240)` keeps the shortest path **independently for every exact (source bodyId, target bodyId) pair**, keyed by *both* endpoints:

```python
pair = (path[0], path[-1])
distance = len(path) - 1
previous = shortest_distance.get(pair)
if previous is None or distance < previous:
    shortest_distance[pair] = distance
return [path for path in all_paths
        if shortest_distance.get((path[0], path[-1])) == len(path) - 1]
```

This matters because targets are discovered at *different distances*; a global hop cutoff would wrongly keep a long path to a close target and drop a valid short path to a far target. Only after per-pair shortest retention are those filtered neuron paths mapped to types. With `skip_bodyId=True`, multiple neuron pairs can map to the same type sequence; the type table dedupes the sequence and emits an enrollment warning — so the type table is deliberately **not** one row per neuron pair.

---

## 3. The algorithm the derivation is built on

Pathfinding itself runs entirely at the neuron level on `G`. Five algorithms are selectable (`ui/config.py` lists `PATHFINDING_ALGORITHMS`, `MemoizedDFS` is the default):

| Algorithm | Idea |
|---|---|
| **Bidirectional BFS** | Search from source and target simultaneously, meet in the middle |
| **Dynamic programming (DP)** | Layer-by-layer reachability via `max_interlayer` depth cap |
| **Memoized DFS (fwd/bwd)** | Depth-first with memoized sub-results, forward and target-rooted |
| **Meet-in-the-middle** | Split the path at its midpoint |
| **DFS** | Plain depth-first enumeration of all paths |

The discovered set of in-path neuron edge pairs (`edges_in_paths`) then feeds `EnrichConnectionTable` (pair dedup + weight summation + type aggregation) to build `conn_types` — type-level edges, every one backed by a real neuron path.

The discovery graph is first trimmed by **path integrity** (`_trim_edges_with_path_integrity`): a reachability filter plus source/target reservation and dead-end refill, plus a pan-graph edge limit (`edgeN_limit`) that keeps only the strongest usable edges. Crucially, that edge limit bounds the *search space* only — it does not cap the number of type-level paths reported.

---

## 4. Pros and cons

### 4.1 The main design: derive from neuron paths, don't re-search on the type graph

**Why:** A neuron type is a class shared by many neurons, and a type-edge `A → B` aggregates *all* neuron pairs `(aᵢ, bⱼ)` into a "bundle" of parallel channels. Once aggregated, the identities of the individual neurons are lost — so a type path `A → B → C` can exist at the type level **even when no single chain of neurons realizes it.**

The canonical failure mode (from the design doc's synthetic example):

```
neuron pairs:              type aggregation:
  a1 → b1   (A → B)         A → B   (w = 18)
  a2 → b2   (A → B)         B → C   (w = 16)
  b2 → c1   (B → C)
  b3 → c2   (B → C)
```

Type-level says `A → B → C` exists. But the only A→B edges land on b1 and b2, and the only B→C edges leave b2 and b3 — and b1 is a dead end, b3 has no incoming A-edge. **No neuron path exists.** It's a phantom.

This is **verified on real data**, measuring the current pipeline (derive) vs the shortcut (aggregate-all → type pathfinding), across queries in male-cns and hemibrain. The instrument was a one-shot (`type_agg_eval.py`) in `local_data/`, the disposable data directory, and is not kept — the table is its recorded output, and re-deriving it means rebuilding the comparison:

| Query | bodyId paths | type OLD (derive) | type NEW (shortcut) | phantom |
|---|--:|--:|--:|--:|
| R1-R6→Tm3 L2 (male-cns, seed 3) | 50 | 11 | 21 | 10 |
| L1→Tm3 L2 (male-cns) | 399 | 27 | 133 | 106 |
| R1-R6→Tm3 L3 (male-cns) | 176 | 35 | 3 281 | 3 151 |
| KCg-m→MBON01 L2 (hemibrain) | 14 749 | 334 | 262 | 137 |
| **TOTAL** | | **443** | **3 749** | |

**97% of the shortcut's extra paths are phantom** — with no realized route, no path metrics, and no neural meaning.

#### Pros of the derive-first design

- **No phantom paths.** Every reported type path is the projection of a real neuron chain. This is the single most important advantage — verified at 97% phantom inflation for the shortcut.
- **Finds repeated-type routes** (`A → B → A`) that a simple-path search on the type graph drops.
- **Honours depth limits.** The neuron-level discovery respects `max_interlayer`, so routes that only exist beyond the depth cap are correctly *not* reported. The type graph has no depth concept and would silently report them.
- **Respects target identity.** The derivation requires ending at a *queried* target type that a real chain actually reaches. (The design doc's R1-R6→Tm3 L2 case: the shortcut reports `R1-R6 → L1 → Tm1 → Tm3`, but the Tm1 that receives from the discovered L1 projects to a Tm3 **that isn't one of the six queried targets** — a phantom resolved only by the derivation.)
- **Path metrics are real.** Because paths correspond to neurons, downstream metrics (lengths, counts, weights) are meaningful. `OLD ⊆ NEW` would be desirable but is *not* guaranteed — the shortcut can also *lose* real paths when its different edge weights re-rank the edge-limit trim (broke on hemibrain).

#### Cons of the derive-first design

- **Higher compute.** Discovery enumerates paths over the full neuron graph, which can reach into the millions (e.g. 14 749 neuron paths on one hemibrain query). Deriving is O(paths) after that, cheap, but the discovery itself is the cost.
- **Memory pressure from the path set.** The `line 10209` doc note flags that paths can number in the millions; the set-building loop (`seen.add(...)`) is the load-bearing step.
- **Weak-route loss.** Some real neuron routes use weak edges that the path-integrity edge-limit trim discards; recovering them would cost hundreds of phantoms (a 33:1 noise-to-signal ratio at L3). The toolkit chooses the clean signal.
- **No second source of truth.** The type-level output is a *summary* of the discovery. It cannot introduce any path that the neuron discovery didn't already see — by design.

### 4.2 The edge-weight aggregation itself (`aggregate_by_label`)

**Pros:** gives a clean, cheap type-topology view (which types connect, by what total strength); drives the connectivity profiler's type-level comparison vectors and matrices; powers the sankey/heatmap/network visualizations.

**Cons:** the summed weight is the weight of *the type edge*, not of any single connection — it conflates distinct neuron pairs, which is exactly the bundling that creates phantoms. On its own it is a topology overview, *not* a path answer.

### 4.3 When the type-level shortcut *is* acceptable

Per the design doc, aggregate-all → type pathfinding is legitimate only as a **different artifact**, never as `"allpaths"`:

- ✅ a **type-topology overview** of the discovered network (the role `network_early` plays) — shows which types connect without claiming any path is real;
- ✅ when **only type-level edges exist** (neuron-level data unavailable);
- ❌ as the `"allpaths"` result — phantom paths have no metrics and silently corrupt downstream analyses.

---

## 5. Summary

The toolkit's aggregation of type-level paths is built on a deliberate trade: **run pathfinding once, cheaply, at the neuron level; derive type-level paths from what you found.**

- Edge weights aggregate via `aggregate_by_label` (sum synapse counts over neuron pairs → type edges).
- Paths are **derived**, not re-searched: each discovered neuron path is projected onto its type sequence and validated hop-by-hop (`_derive_type_paths_from_bodyid_paths`).
- This eliminates the phantom paths that a naive "aggregate-all → type pathfinding" would produce (97% phantom in real-data evaluation), preserves repeat-type routes, and respects depth and target-identity constraints.
- The costs are discovery-side compute/memory, plus loss of some weak routes that the edge-limit trims.

The type graph is a *summary artifact* — a correct and useful one for topology — but it is explicitly not a substitute for the neuron-level discovery when the question is "does a real chain of neurons connect these two types."

---

## 6. Files used for this report

- `src/coana.py` — `FindNeuronConnection` discovery pipeline
- `vispath-subproject/src/vispath_pkg/fast_graph_core.py` — `FastGraph`, `aggregate_by_label`, `find_paths_*`
- `docs/technical/TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md` — design rationale + evaluation data
- `docs/core-features/PathFinding_Methods.md` — algorithm overview
- `scripts/FindPath.py`, `ui/config.py`, `ui/tabs/find_path.py` — entry points and parameters

---

## 7. Addendum (2026-08-22) — corrections and follow-up fixes

An independent review of this report against the code found three
inaccuracies, since fixed in the codebase:

1. **"No pathfinding ever runs on the type graph" (lines 7, 97) was
   overbroad.** The legacy `FindPath` method (still registered in
   `ui/runner.py` TOOL_REGISTRY) performs a type-level graph search
   (`G_type.all_simple_paths` at coana.py:10778), and the current
   `_find_paths_core` used to run a **group-level** graph search
   (`G_group.all_simple_paths` / `find_paths_shortest`) for custom
   groups. The group branch has now been converted to the same
   derivation approach as types (`_derive_label_paths_from_bodyid_paths`),
   so the FindAllPath/FindShortestPath pipeline no longer runs any
   label-level pathfinding. The legacy `FindPath` entry point is
   unchanged and out of scope.
2. **The evaluation table's "type OLD (derive)" column (line 159) was
   mislabeled.** The one-shot harness measured OLD as the
   historical type-graph re-search on in-path-aggregated edges (its
   `type_paths()` runs `find_paths_memoized_dfs` on the type graph); the
   derivation is the eval's third ("agg") pipeline, whose counts are not
   in the table. The 97% figure (3420/3539 = 96.6% of new-only paths) is
   unaffected by the relabeling.
3. **The pan-graph discovery edge limit is `graph_edge_limit_bodyid`,
   not `edgeN_limit` (line 135).** `edgeN_limit` is the
   visualization-only cap (config.py:331; the warning note at
   coana.py:10278-10284 states pathfinding is not trimmed by it). The
   discovery trim also applies only for deep searches
   (`max_interlayer >= 3`) in 'all' mode and is off by default in
   'shortest' mode.

The report's TOTAL row (443 / 3 749) spans the full 7-run battery (the
seed-5 and Tm1 rows appear only in the design doc), so it cannot be
reconciled with the four displayed rows; the 97% needs the NEW-only
column (3 539) to reproduce.

Separately, untyped (and unassigned) neurons are now preserved by the
type/group aggregation via the exclusive label chain `label -> type ->
bodyId` (groups: `custom_group -> type -> bodyId`), first non-empty
wins, instead of being dropped or merged into an 'Unknown' bucket — see
the design doc for details.
