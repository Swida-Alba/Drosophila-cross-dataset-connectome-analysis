---
title: Pathfinding Algorithms
tag: Connectome
---

How DROCAT finds multi-hop pathways and how runs are bounded. <b>StrongestFirst is the built-in 'all'-mode algorithm</b> — the Algorithm selector was removed on 2026-09-05. The other enumerators below remain available to scripts via the <code>pathfinding</code> parameter and are kept here for the benchmark record.

  
  
## StrongestFirst (budgeted) — the pipeline algorithm

<code>pathfinding='StrongestFirst'</code> &middot; <code>FastGraph.find\_paths\_strongest\_first</code>

Best-first search ordered by each path's <strong>bottleneck</strong> (its weakest edge), with a widest-path backward DP as the admissible bound. Emits complete intact paths <strong>strongest-first</strong> under a path budget (<b>Max Paths (BodyId)</b>, <code>max\_paths\_bodyid</code>; 0 = auto → internal 1,000,000 budget): when the budget is reached the search <strong>drains every tie at the cutoff strength <strong>tau</strong></strong> and stops, so the output contains exactly "all intact paths with bottleneck &ge; tau" (the drain may overshoot the budget — both numbers are reported). Without a budget bite the emitted set is identical to the complete enumerators, and the run reports its <strong>natural tau</strong> — the weakest emitted path's bottleneck ("every threshold up to this value yields this identical set").

<strong>Threshold basis</strong>: on <code>weight\_basis='connection\_ratio'</code> runs the edge weight is the F9 connection ratio (synapses / the post neuron's total incoming at min\_weight=1) — the threshold, Edge Budget tiers and StrongestFirst bottlenecks are in ratio units; every synapse column is unchanged. See <code>docs/core-features/ConnectionRatioPaths.md</code>.

Two lossless prunes (hop-budget edge pruning, iterated to a fixpoint, + dead-end node pruning) shrink the cone before enumeration. A lossy <strong>Edge Budget</strong> (<code>graph\_edge\_limit\_bodyid</code>, default 1M, 0 = off) caps the cone itself: when the lossless-pruned cone still exceeds the budget, the <b>budget-fit search</b> floors it at the weakest weight tier whose closed cone fits the cap (<code>t\*</code>) — exactly equivalent to raising the threshold to <code>t\*</code>, the applied floor is reported as <code>edge\_weight\_floor</code> (effective cutoff = max(tau, t\*)), and the residual slack plus probe trace are recorded. When even the full closed cone fits the cap, no floor is applied. Shortest mode is never floored.

<hr/>

## What "pathfinding" means here

Given a set of <b>source neurons</b> and a set of <b>target neurons</b>, DROCAT enumerates every acyclic connection chain (path) from any source to any target with at most <code>Max Intermediate Layers + 1</code> edges (the <b>cutoff</b>). All five algorithms return <b>exactly the same paths</b> — they only differ in how fast they run and how much memory they use.

Path counting is exponential in the cutoff (each extra hop multiplies the search by the average branching factor), so the algorithm choice matters most for <b>deep</b> queries (3+ intermediate layers) on <b>dense</b> connectomes.

  
## The complete enumerators (API / benchmark reference)

<p class="small">All of these return <b>exactly the same path set</b> as
an unbudgeted StrongestFirst run — they differ only in speed and
memory. They are no longer selectable in the UI; scripts can still set
<code>pathfinding='MemoizedDFS'</code> (etc.) with
<code>max_paths_bodyid=0</code> for unbounded complete enumeration,
which is also exactly what a τ-clamped run equals at its reported tau.</p>
  

<div class="grid">
  <div class="card">
    <h3>DP (Backward Reachability)</h3>
    <p>First computes which nodes can reach a target within the cutoff
      (backward DP layers), then runs a forward search that only follows
      edges that provably lead to a target.</p>
    <p class="small">Strongest dead-end pruning; stores only node sets.
      <b>Fastest measured at 3 intermediate layers</b> (0.02 s) with low,
      flat memory. Degrades on very deep (5+) queries. Not the default —
      <b>MemoizedDFS</b> is.</p>
  </div>

<div class="card">
    <h3>DFS (Backward Memoized)</h3>
    <p>A standard depth-first traversal from the targets, remembering which
      nodes can still reach a source at each remaining depth.</p>
    <p class="small">Moderate memory. <b>Fastest at 4 intermediate
      layers</b> and the best time-memory tradeoff at 5, when the target
      group is small (the backward direction starts from the smaller
      set).</p>
    <p class="small"><b>Best for:</b> deep paths with few targets.</p>
  </div>

<div class="card">
    <h3>MemoizedDFS (Forward Memoized)</h3>
    <p>A depth-first traversal from the sources that remembers which
      successors can still reach a target at each remaining depth, so
      each (node, depth) state is computed once.</p>
    <p class="small">Moderate memory and no reversed-graph copy — the
      memo is built from the sources and pruned to successors that can
      reach a target, so the search stays inside the source-side cone.
      <b>Fastest at 4 intermediate layers</b>; the default.</p>
    <p class="small"><b>Best for:</b> balanced queries; deep paths with
      many targets.</p>
  </div>

<div class="card">
    <h3>MeetInMiddle (Meet-in-the-Middle)</h3>
    <p>Stores all half-paths up to the middle depth from the sources, then
      walks backward from the targets and stitches matching halves.</p>
    <p class="small">Fastest at shallow depths (2–3 layers), but its
      half-path table is rebuilt for every length and the pairwise
      stitching costs more than a fully-pruned shared memo on real
      connectomes — so it is <b>not</b> the deep-path winner its design
      suggests.</p>
    <p class="small"><b>Best for:</b> shallow queries; a safe middle
      ground at 4+ layers.</p>
  </div>

<div class="card">
    <h3>Bidirectional (BFS)</h3>
    <p>Searches forward from sources and backward from targets at the same
      time, then stitches the two half-trees at the middle layer.</p>
    <p class="small">Shortest-first <i>emission order</i> (the path set is
      identical to every other complete enumerator). <b>Highest memory</b>:
      stores
      the full search trees from both sides, and memory grows with the
      cutoff (452 MB at 4 layers, 1.1 GB at 5 on a small query). The
      extra memory buys a smaller search frontier, but not faster
      enumeration: its path stitching copies lists at every depth and
      builds sets per candidate pair, so it is the <b>most expensive per
      path</b> of all four (measured 10.4 µs/path vs 1.8 for DFS/DP on a
      controlled graph) — the classic bidirectional win only applies to
      shortest-path search, where you can stop at the first meeting.</p>
    <p class="small"><b>Best for:</b> shallow API queries where you want
      shortest-first ordering and memory is not a concern. Note the
      <b>Shortest Paths</b> tab does not use it — that tab runs the
      target-rooted StrongestFirst enumerator
      <code>find_paths_shortest_strongest_first</code> (see below).</p>
  </div>

</div>

  
## Measured performance

Real hemibrain v1.2.1 queries, connections ≥ 5 synapses, wall time (untraced) + tracemalloc peak memory. All algorithms found the identical path set in every scenario.

- <b>2–3 intermediate layers</b>: LC → MBON/PPL, 10,331 neurons, 301,443 edges, 4 sources × 5 targets (14 / 2,874 paths).
- <b>4–5 intermediate layers</b>: LPLC1 → MBON01, 24,576 neurons, 722,143 edges, 2 sources × 2 targets (35,819 paths; 5M+ paths at 5 layers — runs capped after 5M paths).

  

<table>
  <tr><th>Algorithm</th><th>2 layers</th><th>3 layers</th>
      <th>4 layers</th><th>5 layers</th></tr>
  <tr><td><b>MemoizedDFS</b> (default)</td><td>0.01 s / 0.1 MB</td>
      <td>0.06 s / 1.7 MB</td><td><b>0.23 s</b> / 7.6 MB</td>
      <td>4.97 s / 651 MB</td></tr>
  <tr><td>DFS (backward)</td><td>0.02 s / 0.3 MB</td><td>0.05 s / 1.8 MB</td>
      <td>0.42 s / 9.6 MB</td><td>5.01 s / 612 MB</td></tr>
  <tr><td>MeetInMiddle</td><td><b>0.00 s</b> / 0.3 MB</td>
      <td><b>0.01 s</b> / 0.6 MB</td><td>1.25 s / 4.8 MB</td>
      <td>6.18 s / 615 MB</td></tr>
  <tr><td>DP</td><td>0.06 s / 0.8 MB</td><td>0.02 s / 1.5 MB</td>
      <td>0.44 s / 9.4 MB</td><td>20.2 s / 651 MB</td></tr>
  <tr><td>Bidirectional</td><td>0.07 s / 18.9 MB†</td><td>0.18 s / 59.4 MB†</td>
      <td>0.64 s / 194.9 MB†</td><td><b>4.54 s</b> / 892 MB†</td></tr>
</table>

<p class="small">
Fastest time per column in <b>bold</b>. † = highest memory in that
column (Bidirectional stores full layer trees from both sides).
<b>2026-08 optimization:</b> FastGraph now keeps a lazy reverse
adjacency index instead of copying the whole reversed graph per run —
the earlier 100–450 MB footprints were mostly those copies and are
gone (DP dropped from 272 MB to 9.4 MB at 4 layers). At 5 layers every
value includes ~610 MB of the 5 million collected paths themselves.
Benchmark harness: <code>archive/examples/performance/benchmark_pathfinding.py</code>;
full analysis in
<code>docs/technical/PATHFINDING_ALGORITHM_EVALUATION.md</code>.
</p>
  

<div class="callout info">
  <b>What changed in the 2026-08 audit</b>
  The algorithm names now match the implementations (a routing bug made
  "MemoizedDFS" run meet-in-the-middle; the forward memoized DFS was
  never wired up — and it turns out to be the fastest of all: it makes
  no reversed-graph copy and prunes to the source-side cone). All five
  implementations were verified correct against a naive reference on
  150 random graphs (750 runs, zero mismatches). The default changed
  from DP to <b>MemoizedDFS</b>.
</div>

  
## The Shortest Paths tab

The Shortest Paths tab does not use any 'all'-mode enumerator above. It enumerates through <code>FastGraph.find\_paths\_shortest\_strongest\_first</code>: a per-target reverse BFS outlines the min-hop DAG, a maximin bottleneck DP annotates every branch with the best achievable strength, and tied shortest paths are emitted best-first in strength order with a k-way merge across targets. The <b>Max Paths (BodyId)</b> budget bounds it exactly like an 'all'-mode run — a bitten run is tau-bounded ("all min-hop paths with bottleneck &ge; tau") — and the <b>Edge Budget never applies</b>: shortest mode is never floored. (The <code>find\_paths\_shortest\_backward</code> exporter also exists in FastGraph, but the pipeline does not call it.)

<hr/>
  

<h2 id="corridor-lane">Fast Mode (the corridor lane) vs Complete Mode</h2>

Every pathfinding run has two analysis modes. <strong>Complete mode</strong> is the run itself: the budgeted StrongestFirst pipeline (+ Refill) finding <em>all intact paths &ge; the run's applied threshold</em>. <strong>Fast mode</strong> — the corridor lane — is a <strong>standalone fourth pruning axis</strong> beside the Edge Budget, the StrongestFirst budget and the type-level refill: <strong>route-strength pruning in the type quotient</strong>. After a deep run (Complete Paths or Shortest Paths with Max Intermediate Layers &ge; 4, value <code>auto</code>; or any depth with value <code>both</code> — the Run Mode (Complete / Fast) control values are <code>complete</code>/<code>auto</code>/<code>both</code>/<code>fast</code> — <code>fast</code> runs the lane <strong>standalone</strong>, skipping discovery, enumeration, materialization and the refill entirely, so <code>route\_scope\_paths/route\_paths\_type.csv</code> is that run’s <em>only</em> path table and the pair report opens on the lane’s own views rather than describing it as finding nothing; legacy <code>off</code>/<code>on</code> normalize on read), it re-enumerates at the run's <strong>ASKED</strong> threshold restricted to <em>validated top-M\*</em> type-level corridors — keeping the weak-but-real routes (<code>\[asked, t&#964;)</code> mass) that the main run's budget/floor cut. It never substitutes for the main pipeline and never modifies the run's own outputs; everything lands in <code>route\_scope\_paths/</code> beside the run folder (plus <code>data\_details/route\_plan/plan.json</code>).

- <strong>Plan (G\_type\*)</strong>: the run's edge universe (same filters, hemisphere suffixes, label chain, untyped handling) is aggregated to type pairs; corridors are ranked by <em>UB</em> = the widest-path bottleneck upper bound (min over hops of the type-pair max edge weight). Route floor certificate: dropping walks below the floor never drops a path with bottleneck &ge; the floor.
- <strong>Validate (phantom-checked)</strong>: planned walks are accepted only when a layered-reachability test proves real bodyId paths at the asked threshold (deep plans are ~50% phantom); each (source-target type pair &times; hop length) class carries TWO budgets in TWO currencies (2026-10-16): a <strong>route share</strong> of valid type-level walks (<code>round(top_m_base &times; length_decay^(h-1))</code>, default 15 at h2) that an accepted candidate pays ONCE, and a <strong>realization envelope</strong> (<code>--pair-length-budget</code>, default 100 bodyId witness paths) that the materialization pass spends &mdash; every accepted route keeps at least one witness, so a fat corridor&#39;s tie mass is truncated and the route list never is. Phantoms cost candidate pulls, never the route share (pull cap = 2 &times; share; <code>stop=probe_cap</code> vs <code>stop=budget</code> says which bound closed the class). Enumeration keeps StrongestFirst tie-drain semantics; the walk rows record <em>that walk’s</em> landing <code>applied\_tau</code> (the value the drain reached when the walk was accepted), while the CLASS-level certificate is <code>floor</code>/<code>floor\_next</code> in the provenance — and those are candidate upper bounds, not measured strengths (G11; three scopes share the one key <code>applied\_tau</code>: walk, class, and the run’s own <code>parameters.txt</code> value, F-EX-012). <strong>Planner rule (default)</strong>: a walk that revisits a type is not planned at all &mdash; reciprocal pairs make multi-visit subtrees explode the drain for little gain (restorable with the standalone CLI's <code>--allow-multi-visit</code>; pruned extensions are disclosed as <code>multi\_visit\_pruned</code>). <strong>Certified mode (opt-in)</strong>: the standalone CLI's <code>--certify-actual</code> keeps accepting candidates per class until the first skipped candidate's bound drops below the weakest exported path's real bottleneck &mdash; the exported top-share is then the true strongest actual neuron routes, not bound-guided representatives (slower: cost tracks bound tightness; per-class <code>certified</code>/<code>weakest\_emitted</code> in the provenance). On shortest runs the lane validates against the run's own shortest-discovery store when present (per-layer ROI slices are summed to the physical pair weight — the store's fetches are verbatim per-ROI frames — and layers are then deduplicated keep-first by sorted layer; any cross-layer weight disagreement after summation is disclosed as <code>store\_weight\_conflicts</code>); without a store it falls back to the lossless hop-budget prune (<code>validation\_basis</code> records which).
- <strong>Enumerate per walk</strong>: each accepted walk's own hop-pair subgraph is enumerated strongest-first on a per-position layered graph (each hop's edges appear only at their own layer) &mdash; plan-faithful, no cross-class leakage, no mislabeled hop lengths, and the emitted bodyId paths are <strong>body-simple</strong> (a neuron appears at most once) by construction; the rare knob-restored multi-visit walks enforce raw-body simplicity inside the generator itself (F-PF-003: the budget can no longer be spent on invalid repeated-body walks before a valid weaker realization surfaces).
- <strong>Outputs</strong>: <code>route\_paths\_type.csv</code> (one row per validated walk: engine-style weights/ratios/min\_weight/nt\_types/coverage + applied\_tau, route\_h, route\_ub, route\_floor\_h, plan\_rank, realized\_paths), <code>route\_paths\_bodyId.csv</code> (per-path detail), <code>route\_connection\_type.csv</code> (scope type-pair masses: <code>weight</code>/<code>universe\_type\_weights</code> = the whole admitted pair, <code>selected\_support\_mass</code> = the lane’s emitted witnesses), <strong>the lane’s own second round at synapse &ge; 1</strong> — the same validated chains re-valued through the refill’s forward+backward layered filter as <code>route\_full\_paths\_type.csv</code> / <code>route\_full\_carrying\_paths\_type.csv</code> / <code>route\_full\_type\_pairs.csv</code>, with denominators in <code>route\_type\_incoming.csv</code> (&ge; 1) and <code>route\_type\_incoming\_asked.csv</code> (the lane’s asked tier) and mirror pages <code>route\_full\_type\_network.html</code> / <code>route\_full\_type\_heatmap.html</code>, <code>route\_provenance.json</code> (per-class floors/stop reasons/phantoms/off-plan disclosures, per-stage <code>phases\_s</code> timings, gap partition vs the run's own paths, plan hash), <code>README.md</code>, and <code>plan.json</code> under <code>data\_details/route\_plan/</code> beside the run folder (beside an explicit <code>--out</code> in standalone direct mode).
- <strong>Discipline</strong>: a user\_warning\_notes entry appears only on refusal or failure (e.g. a missing or non-positive Min Connection Ratio on a ratio-basis run; ratio-basis corridors themselves are supported — each edge is admitted on its own F9 input share); multi-threshold replays skip slice folders (the base lowest-threshold folder carries the superset corridors); the lane runs in an isolated subprocess with a 900&nbsp;s wall-clock budget, so it can never stall the run.

Cross-dataset comparisons forward the mode to every per-dataset run (firing only at each dataset's lowest swept threshold) and the report gains a presence-driven corridor section comparing corridors through the canonical type mapping.

<strong>Ratio flavors and visualization bases (shared with the refill lane):</strong> corridor walk rows carry BOTH per-hop share flavors &mdash; plain (realized mass &divide; the post type's full-membership incoming) and adjusted (&divide; the &ge;-threshold incoming; equal to plain when the asked threshold is 1). The same four metric bases are selectable across the pathfinding visualizations &mdash; Synapses / Ratio (all inputs) / the adjusted ratio, labelled Adj. Ratio with the tier its denominator was conditioned at / Refilled &mdash; on the heatmap toggle and the network + Sankey Connection-Metric selectors (options appear when the data carries them; hover always lists every metric; Sankey labels stay weight-based with metrics on hover; NT coloring, when enabled, overrides colors only). BodyId-level visualizations keep weight+ratio only: adjusted/refilled bodyId data lives in the exported CSVs.

  
## Cost profile (time & memory)

Measured on the MCNS long-haul query (ORN* &rarr; DN*, 2,635 sources,
1,350 targets, max intermediate layers = 8, bound 9): the <strong>Complete
mode</strong> is output-linear &mdash; its cost scales with the emitted
bodyId path count P through materialization, the enrichment join and the
CSV/visualization writes (measured: 25&ndash;30 min wall, 11&ndash;18 GB
peak at 354K&ndash;1M paths). The refill's full-scope tail is the layered
per-position body-set filter (2026-10-13): linear in the chains' own
subgraphs, no path materialization, nothing truncates &mdash; on the
long-haul ratio query it replaced a ~115-minute DP/DFS reconstruction
(3.02B walks, 737 chains capped at 2M paths) with a ~3-minute pass at
identical asked-scope output; the label-revisiting chains among them
carry the layered-walk superset semantics, disclosed as
<code>chains_revisit</code>. The <strong>Fast mode</strong> is
input-linear &mdash; one full-cache scan plus per-source drains and
exports (measured: ~4.3 min, a few GB, 12&ndash;22K type walks). The
corridor's per-source drain explores the type graph with a best-first
order whose worst case is super-linear in the graph's density; the
pop/push budgets bound it and a bound exceed is disclosed
(<code>push_capped</code>/<code>stalled</code>), so the lane degrades to
a weaker certificate rather than an unbounded run. Threshold semantics
do not change either mode's complexity class: the synapse cutoff is an
absolute edge filter, while the ratio cutoff is per-neuron adaptive
(each hop needs only t<sub>r</sub> of that target's own input), which
admits a larger cone and a proportionally larger output &mdash; on the
long-haul query the ratio basis kept ~3&times; the bodyId paths and
correspondingly more wall time.

## What bounds a run today?

  

<table>
  <tr><th>Knob</th><th>Bounds</th><th>Reported as</th></tr>
  <tr><td><b>Min Synapse Count</b> (asked threshold)</td><td>the discovery cone, a priori</td><td>the folder <code>minsyn_{t}</code></td></tr>
  <tr><td><b>Edge Budget</b> (<code>graph_edge_limit_bodyid</code>, default 1M)</td><td>the lossless-pruned cone, lossy: budget-fit floor at the weakest tier t* whose closed cone fits the cap</td><td><code>edge_weight_floor</code> (w0) / <code>edge_budget_landing</code> (w1) + residual slack in the run attributes + a lossy-floor note</td></tr>
  <tr><td><b>Max Paths (BodyId)</b> (<code>max_paths_bodyid</code>, 0 = auto → 1M)</td><td>the output path list: keeps all intact paths with bottleneck ≥ tau</td><td><code>strongest_first_tau</code> (tau) + <code>paths_complete</code> in the run notes and exports</td></tr>
</table>

<p class="small">They compose exactly: effective cutoff =
<b>max(tau_budget, t*)</b>, and the run equals a complete run at that
threshold. A complete run reports its natural tau (weakest emitted
bottleneck) — every threshold up to that value yields the identical
set. The <b>Drop Untyped Neurons</b> checkbox is a neuron-label filter,
not a bound: it removes edges touching untyped labels (empty /
Unknown / NaN / bodyId-fallback) before the graph is built and is
distinct from graph budgets, path budgets and the drawing cap.</p>
<p class="small">Every run consolidates these into an
<b>applied-threshold provenance</b> block written to
<code>parameters.txt</code>, <code>all_attributes.json</code> and
<code>data_details/parameters.csv</code>: <code>requested_threshold</code>,
<code>applied_threshold</code>, <code>applied_threshold_source</code>
  ('requested' | 'strongest_first_budget' | 'edge_budget' |
  'strongest_first_budget+edge_budget'),
<code>strongest_first_budget</code> (auto 1,000,000) with
<code>strongest_first_budget_bitten</code>, <code>strongest_first_tau</code>
  (the landing tau), <code>tau_canonical</code>,
<code>strongest_dropped_bottleneck</code> (w2 — the strongest path NOT
kept), <code>edge_budget_applied</code>, <code>edge_budget_landing</code>
  (w1), <code>edge_weight_floor</code> (w0),
<code>strongest_retained_bottleneck</code> (W* — the widest-path ceiling
after lossless pruning; lossless pruning never changes it) and
<code>paths_complete</code>. When a gap exists between w2 and tau, every
threshold in [w2+1, tau] yields the identical set and w2+1
  (= <code>tau_canonical</code>) is the minimal equivalent; for an
unbounded run <code>applied_threshold</code> stays at the requested
threshold and the natural tau is reported separately. A budgeted output
is a strength-bounded path set, never an arbitrary first-N truncation.</p>

  
## Practical tips

- <b>Prefilter the graph</b>: raising <b>Min Synapse Count</b> and lowering the <b>Edge Budget</b> shrinks the search space far more than any other control.
- <b>Deep + dense = expensive</b>: the number of paths grows exponentially with <b>Max Intermediate Layers</b>; the output itself dominates runtime and memory once path counts reach millions — that is what Max Paths bounds, with a reported tau.
- <b>Clamping is honest</b>: a τ-clamped run equals a complete run at its reported tau (complete enumerators — API only — reproduce the identical set when run at that threshold).
