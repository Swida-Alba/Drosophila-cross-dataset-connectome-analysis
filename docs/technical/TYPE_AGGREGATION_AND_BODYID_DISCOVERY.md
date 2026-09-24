# Why FindAllPath Finds Paths at the bodyId Level First

Design rationale for the discovery-first pipeline in `FindAllPath` and
`FindShortestPath`:

> The type-level results are **derived from** discovered bodyId-level paths —
> never computed independently of them. Aggregating all per-bodyId pairs
> into a type-level graph and running pathfinding on that graph alone would
> hide the differences *within* a type and report paths that no real chain
> of neurons can realize.

## 1. The pipeline (current design)

```
per-bodyId pair tables (forward layers for Complete Paths;
                         target-rooted incoming layers for Shortest Paths)
   │
   │ 1. pan-graph edge limit (path-integrity trim: reachability filter +
   │    source/target reservation + adaptive dead-end refill)
   ▼
bodyId graph G ──────────► 2. bodyId-level pathfinding  (THE DISCOVERY)
   │                              │
   │                              ▼
   │                    edges_in_paths — the SET of unique (pre, post)
   │                    pairs that lie on at least one found bodyId path
   ▼                              │
   │                              ▼
   │                    3. each layer table is filtered down to its
   │                       in-path pairs (set intersection)
   ▼                              │
   ▼                              ▼
   │                    4. EnrichConnectionTable (pair dedup + weight
   │                       summation + type aggregation)
   ▼                              │
   ▼                              ▼
conn_types — type-level edges, every one backed by a real bodyId path
   │
   ▼
5. derive the type-level (and custom-group-level) paths: map each
   discovered bodyId path to its type/group sequence, then VERIFY every
   hop against the corresponding edge table (no second pathfinding on a
   label-level graph, and no label-level edge limit — the bodyId
   discovery already bounds the search space; the Visualization Edge
   Limit `edgeN_limit` remains the only type-level cap, applied when
   drawing and shared with the bodyId-level view, §6b). Custom-group paths are derived exactly like type paths, so
   they cannot chain group edges backed by different bodyId pairs into
   phantom routes.
```

## Shortest discovery is target-rooted

`FindShortestPath` does not use the Complete Paths forward source fan-out as
its default discovery graph. For each queried target, it performs a reverse
BFS through incoming connections until the enrolled source bodyIds have been
reached or the configured hop bound is exhausted. A source bodyId that cannot
reach any queried target is never part of the shortest graph. Before graph
construction, incoming rows are filtered to the shortest-DAG branches that
actually connect a requested source to a discovered target.

The target's earliest discovery layer is the minimum over the source bodyIds
that reach it. It must not be used as a global cutoff: another source can have
a longer shortest path to that same target. Shortest filtering is therefore
performed on exact `(source bodyId, target bodyId)` endpoints before mapping
paths to types.

When `skip_bodyId=True`, multiple bodyId pairs can collapse to the same type
sequence. The type CSV intentionally deduplicates that sequence, so the run
writes an enrollment warning and root-level `source_neurons.csv` and
`target_neurons.csv`. Use Shortest Paths with bodyId output enabled for exact
pairs, or Complete Paths for all paths within a chosen depth.

If the reverse BFS reaches the hop limit while a frontier is still active,
the result is shortest only within the explored, threshold-filtered graph.
A path found at that boundary is a constrained solution, not a proof of a
global shortest route. This condition is recorded in `user_warning_notes.txt`.

Every step after the discovery consumes only pairs that were **observed on
an actual bodyId path**. The type-level output in step 5 is a *summary* of
the discovery — the unique type sequences of the discovered bodyId paths —
not an independent source of truth. In shortest mode, the pipeline first
enforces the minimum hop count independently for every exact
`(source bodyId, target bodyId)` pair. It then maps those filtered paths to
types; it does not apply a global distal-target cutoff or a second
per-type-pair minimum. Thus a longer type sequence is retained only when it is
the shortest path for a different target instance. Identical type sequences
remain one type-level row; use the bodyId path output when the exact instance
pair must remain visible.

### Why the type-level paths are derived, not re-searched

Step 5 was historically a second pathfinding on the type-level graph
(built from `conn_types`). Running a search there re-introduces the bundle
effect at the type level: it can report a type chain A→B→C whose hops are
each backed by *some* bodyId pair, but by **different** pairs, so no single
bodyId path realizes the chain. Deriving the type paths from the
discovered bodyId paths instead guarantees every reported type path is the
type sequence of a real bodyId path. It also preserves repeated-type
routes (A→B→A through two distinct B neurons) that a simple-path search on
the type graph silently drops. Measured on the battery in Section 5: the
old re-search added 0–3 phantom paths per query and missed 0–31 real
repeated-type paths; the derivation matches the discovered bodyId paths
exactly.

Custom-group paths follow the exact same rule. Historically the group
branch ran a DFS on the aggregated group graph, which could report a
group chain G1→G2→G3 whose hops were backed by different bodyId pairs
(the same bundle effect); it also crashed once the group tables carried
the `custom_group_pre`/`custom_group_post` schema. The group output is
now derived from the discovered bodyId paths with the same endpoint and
hop verification, and repeated-group routes (G1→G2→G1) are preserved.

### Untyped and unassigned neurons keep their identity

A neuron without a type (or without a custom group) is never dropped
from the type/group aggregation. Both enrichment engines resolve each
bodyId to exactly one label by the exclusive chain

- type level: `std_label -> type -> bodyId`
- group level: `custom_group -> type -> bodyId`

first non-empty wins, so a bodyId never appears under both its custom
group and its raw type. Untyped neurons therefore aggregate under their
bodyId (per-neuron groups) instead of vanishing or merging into an
'Unknown' bucket, and the derivation callables use the same chain, so
the type/group paths they emit stay consistent with the edge tables.

The label resolution treats every absent cell identically: `None`, float
`NaN` (real neuron tables store untyped labels this way), and the literal
strings `''`, `'nan'`, `'None'` are all missing. Without the `NaN` guard,
untyped neurons project to the bogus type `'nan'` and every type path
through them is silently dropped (`_is_missing_type_label` in `coana.py`).
Verified on real male-cns data: untyped targets survive as their bodyId
in the type and group outputs (verified with the pathfinding
`drop_untyped` filter disabled — see below — since the default filter
removes untyped-labeled rows before graph construction).

Pathfinding adds a **separate, later filter** on top of this aggregation
rule: `FindNeuronConnection.drop_untyped=True` (the default; UI "Drop
Untyped Neurons") removes connection rows whose pre- or post-side label
is untyped (`utils.label_utils.is_untyped_type_label`) AFTER enrichment
and BEFORE graph construction, in both Complete and Shortest Paths, so
an untyped neuron can never appear even as an intermediate node of a
returned path or visualization. Dropped rows are exported to
`data_details/untyped_dropped_records.csv` (with an `untyped_side`
column) only when rows were dropped, and an untyped source/target can
remain enrolled in `source_neurons.csv` / `target_neurons.csv` while
its incident edges are removed. The aggregation statement above is
unaffected: it governs how labels resolve, while `drop_untyped` governs
which rows enter the graph.

## 2. Why the type level cannot simply aggregate all pairs (the bundle effect)

A type is a **class label shared by many neurons**, and a type-level edge
A→B aggregates **all** bodyId pairs (aᵢ, bⱼ) with those types. Once
aggregated, the individual identities of aᵢ and bⱼ are lost — the type edge
is a *bundle* of parallel channels. A type-level path therefore exists
whenever each consecutive type pair has **some** channel, but nothing
guarantees that the channels **connect to each other**.

### Minimal synthetic example (the core failure mode)

```
bodyId pairs:                  type aggregation:
  a1 → b1   (A → B)              A → B   (w = 18)
  a2 → b2   (A → B)              B → C   (w = 16)
  b2 → c1   (B → C)
  b3 → c2   (B → C)

type-level graph:  A → B → C   ⇒   type path "A → B → C" EXISTS
```

But **no bodyId path** a → b → c exists:

- the only A→B edges land on **b1** and **b2**;
- the only B→C edges leave **b2** and **b3**;
- b1 has no outgoing edge, so a1 → b1 is a dead end;
- b2 → c1 works for a2, but b3 → c2 has no incoming A-edge.

The aggregation mixed the **B neurons that receive from A** with the **B
neurons that project to C** — two different populations sharing the same
type label. The type path A→B→C is a *phantom*: no neuron of type B is both
reached from an A-neuron and able to reach a C-neuron within one hop.

## 3. Where "hidden differences under the same type" appear

The design intent of the discovery step is exactly to prevent phantom paths
by requiring a **concrete chain of neurons**. The hidden within-type
differences that a pure type-level aggregation would gloss over:

| Hidden difference | Consequence for a type-only pipeline |
|---|---|
| **Split membership** — the B-neurons receiving from A differ from the B-neurons projecting to C | the type path A→B→C exists, no bodyId route does (synthetic example above) |
| **Dead-end members** — some B-neurons have only incoming (or only outgoing) edges | they inflate the type edge weight and can create phantom continuations through neurons that can never reach a target |
| **Cutoff interplay** — an A→B edge exists only at a depth where B→C would exceed `max_interlayer` | the type graph has no depth concept, so it reports routes the discovery correctly rejects as too long |
| **Target identity** — a B-neuron projects to *a* neuron of type C, but not to any **queried** target | the type path ends at the right *type* but at the wrong *neuron* |
| **Weak-route trimming** — the pan-graph edge limit keeps only the strongest usable edges | type aggregation preserves the type-pair presence of edges the discovery pruned as unusable |

The discovery is the only step that can answer the question *"does a
concrete chain of neurons exist, within the depth limit, ending at a
queried target?"* — the type level cannot.

## 4. Real-data example (male-cns:v1.0, R1-R6 → Tm3)

Query: 6 R1-R6 sources → 6 Tm3 targets, `max_interlayer=2`,
`min_synapse_num=3`. The type-only pipeline reports the phantom type path

```
R1-R6 → L1 → Tm1 → Tm3
```

Every hop exists at the bodyId level:

- an R1-R6 neuron (e.g. bodyId `274589`) → an L1 neuron (`34008`),
- that L1 neuron → a Tm1 neuron (`38214`),
- that Tm1 neuron → **a** Tm3 neuron (`117197`).

Yet the old pipeline does **not** report this path — and correctly so: the
Tm3 neuron `117197` is *not one of the six queried targets*. The Tm1 that
receives from the discovered L1 population projects to a different Tm3
neuron than the ones being searched. After aggregation the two Tm3 neurons
are indistinguishable ("same type"), and the phantom path appears.

## 5. Comparative measurements

A one-shot harness (`type_agg_eval.py`, written under the disposable
`local_data/` in 2026-09 and no longer kept) compared the two pipelines on
real cached connectomes: **OLD** (discovery + in-path aggregation, the current
pipeline) vs **NEW** (aggregate all per-bodyId pairs → type graph →
type-level pathfinding). Every new-only type path is then checked for a
simple bodyId route **from a queried source to a queried target** in (a)
the trimmed discovery graph (what the old pipeline searches) and (b) the
full untrimmed bodyId graph (routes the bodyId edge-limit trim discarded).
All queries: 6 sources × 6 targets, `min_synapse_num=3`, seed 7.

| Query (dataset, L) | bodyId paths | type OLD | type NEW | OLD missing | NEW-only | backed (trimmed) | backed (full only) | **phantom** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1-R6→Tm3 L2 (male-cns, seed 3) | 50 | 11 | 21 | 0 | 10 | 0 | 0 | **10** |
| R1-R6→Tm3 L2 (male-cns) | 46 | 12 | 23 | 0 | 11 | 0 | 0 | **11** |
| R1-R6→Tm1 L2 (male-cns) | 101 | 24 | 29 | 0 | 5 | 0 | 0 | **5** |
| L1→Tm3 L2 (male-cns) | 399 | 27 | 133 | 0 | 106 | 0 | 0 | **106** |
| R1-R6→Tm3 L3 (male-cns) | 176 | 35 | 3 281 | 0 | 3 246 | 0 | 95 | **3 151** |
| KCg-m→MBON01 L2 (hemibrain) | 14 749 | 334 | 262 | **233** | 161 | 0 | 24 | **137** |
| **TOTAL** | | 443 | 3 749 | 233 | 3 539 | 0 | 119 | **3 420 (97 %)** |

Three findings:

1. **Phantom inflation dominates (97 % of new-only paths).** The bundle
   effect is not a corner case: on every query the shortcut reports paths
   with no bodyId route from a queried source to a queried target. At L3
   the type graph is dense enough that the shortcut explodes (3 281 vs 35
   backed paths) while staying ~97 % phantom.

2. **The shortcut can also LOSE real paths (hemibrain: 233 of 334 old
   paths missing).** The type-level edge limit re-ranks edges by the
   *new* weights (summed over all pairs instead of in-path pairs), so a
   weak-ranked type edge that the old pipeline keeps can be trimmed from
   the new type graph. `OLD ⊆ NEW` is *not* guaranteed — it happened to
   hold on the male-cns runs and broke on hemibrain.

3. **Deep layers: the recovery-vs-noise trade-off is unfavourable.** At
   L3, 95 new-only paths are backed in the full bodyId graph but not in
   the trimmed discovery graph (their routes use weak edges the bodyId
   edge-limit trim removed). Keeping them costs 3 151 phantoms — a 33:1
   noise-to-signal ratio — and still loses nothing from OLD there.

## 6. When is the type-only shortcut acceptable?

The shortcut (aggregate all pairs → type-level pathfinding, skipping the
bodyId discovery) is legitimate **only as a different artifact**, never as
a drop-in replacement for `allpaths`:

- ✅ as a **type-topology overview** of the discovered network (the role
  the `network_early/` / `network_early_bodyId/` early previews play) —
  it shows which type pairs connect, without claiming any path is real;
- ✅ when bodyId-level data is unavailable and only type-level edges exist;
- ❌ as the "allpaths" result — its phantom paths have no path metrics, no
  real route, and would silently change downstream analyses that assume
  each path is realized by neurons.

## 6b. Visualization naming parity and the shared drawing cap

Both visualization levels a pathfinding run produces follow the same
naming convention (`<run>` = the run-folder name):

- **Type level** — `visualization/`: `Network_<run>.html`,
  `Heatmap_<run>.html`, `Sankey_<run>.html`, and the exported data
  under `visualization/visualization_data/<run>_data_*.csv`.
- **BodyId level** — `bodyId_visualization/`: the same names
  (`Network_<run>.html`, `Heatmap_<run>.html`, `Sankey_<run>.html`,
  `bodyId_visualization/visualization_data/<run>_data_*.csv`).
  The historical raw names (e.g. `bodyId_visualization_network.html`)
  are no longer produced, and missing artifact types stay absent —
  a direct-connection view gains no Sankey from naming parity alone.
- **Early previews** keep their own folders, `network_early/` and
  `network_early_bodyId/`, but now use the same `Network_<run>.html`
  prefix inside.

Both levels also share the same **drawing cap**: the Visualization Edge
Limit `edgeN_limit` is passed to the shared VisualizePath edge selector
(`_select_edges_for_plot` in
`vispath-subproject/src/vispath_pkg/vispath.py`) at both levels. The
cap is drawing-only — it never changes fetching, the graph, or the
path outputs, and a single complete path may exceed it to stay intact.
When the cap actually trims, companion files record the exact path
rows the rendered (edge-limited) network represents:
`visualization/visualization_data/type_paths_visualized.csv` (type
level) and `bodyId_visualization/visualization_data/bodyId_paths_visualized.csv`
(bodyId level).

## 7. Related documents

- [Pathfinding Methods in FindAllPath](../core-features/PathFinding_Methods.md)
- [Pathfinding Algorithm Evaluation](PATHFINDING_ALGORITHM_EVALUATION.md)
- `user_warning_notes.txt` (written per run when edge limits or filters
  tilt the outputs)
