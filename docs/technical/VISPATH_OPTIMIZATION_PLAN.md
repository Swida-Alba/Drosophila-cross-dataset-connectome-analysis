# vispath Deep-Dive: Findings & Optimization Plan (2026-08)

A code-level audit of the visualization stack
(`vispath-subproject/src/vispath_pkg/vispath.py`,
`src/comparison/html_report_generator.py`, `fast_graph_core.py`) performed
while adding the Hide-Dead-Ends toggle, undo/redo and auto-mirror features.

**Status: items F1–F4, F6, F10 are IMPLEMENTED (2026-08); F5, F7, F8, F9
remain planned.** Implemented items are marked ✅ below.

## Findings (verified)

### F1. CDN-only JavaScript — offline runs render nothing
`_plot_cytoscape_network` loads cytoscape + 6 layout extensions + dagre
from cdnjs/unpkg. Verified in a sandboxed browser: when the CDN is
unreachable, `cytoscape` is `undefined`, the whole inline script aborts at
`const cy = cytoscape(...)` and the page shows an empty canvas with **no
error message**. The report generator has the same pattern for vis-network.

### F2. Node/edge data embedded into HTML/JS without escaping
`nodes_data` / `edges_data` are injected into the f-string as Python dict
reprs (`'data': {'id': 'C', ...}`). A node label containing `'` (quote),
`</script>` or other JS-sensitive characters would break the page or
inject markup. All user-controlled neuron names flow through here.

### F3. `FastGraph.reverse()` copies the whole graph
`find_paths_backward_dp`, `find_paths_memoized_dfs(backward)`,
`find_paths_meet_in_the_middle` and `find_paths_bidirectional_bfs` each
build a full reversed FastGraph (~250 MB + ~0.3–0.5 s on a 722k-edge
graph) at the start of the run. Forward MemoizedDFS avoids it entirely
(measured 100× lower peak allocation), which is why it is now the default —
but the other algorithms still pay the cost every call, including repeated
calls per query.

### F4. `in_degree()` / `predecessors()` are O(V) / O(E) per call
`FastGraph.in_degree(n)` scans every node's adjacency; `predecessors(n)`
iterates all nodes. Any per-node loop using them is quadratic (e.g. the
dead-end classification prototypes). A precomputed in/out-degree map
(updated on `add_edge`) would make both O(1)/O(deg).

### F5. `_plot_cytoscape_network` is a ~200 KB single f-string
The entire HTML+JS is one Python f-string with `{{ }}` escaping. It is
error-prone (this session's edits required careful brace doubling),
impossible to lint, and any change re-renders the whole artifact. The
report generator duplicates similar toggle JS for vis-network.

### F6. Orphan re-detection inconsistency
`applyEdgeFilter` re-detects orphans with
`.not('.hidden, .filtered')` while `toggleOrphanNodes` uses
`.not('.hidden, .filtered, .selfloop-hidden')` — a node connected only by
self-loops is treated as an orphan by the filter path but not by the
button path. (The new dead-end re-detection follows the button convention.)

### F7. Report networks: full clear+add + re-layout on every toggle
`updateNetworkDisplay` calls `edges.clear(); edges.add(...)` and re-runs
the hierarchical layout via `setOptions` + 200 ms `setTimeout` on every
filter/dead-end toggle — janky on large graphs and easy to race.

### F8. Duplicated dead-end logic (Python vs JS, two functions)
The recursive dead-end closure exists three times:
`_generate_conservation_network` (Python), `_generate_dataset_network`
(Python, with the simpler degree-based variant), and `updateNetworkDisplay`
(JS). They can drift (they already differ slightly: the dataset view uses
the one-step definition, the conservation view the recursive closure).
Dead code removed this session: unused `has_outgoing`/`has_incoming` in
`_generate_conservation_network`, unused `originalDeadEndNodeIds` in JS.

### F9. Undo/redo snapshots are O(V+E) full-state copies
The new undo/redo captures the complete element state (data + classes +
positions) per operation, capped at 50 entries. Fine for typical graphs
(<5k elements) but heavy for 50k-node networks.

### F10. localStorage persistence accumulates keys
Every generated HTML uses a timestamped storage key
(`cytoscape_layout_<name>#<ts>`), so saved layouts accumulate in
localStorage forever with no eviction.

## Optimization Plan (prioritized)

### P0 — Resilience & safety (done)
1. ✅ **CDN fallback guard + error banner** (F1): both generators now show a
   clear "failed to load (CDN unreachable)" banner instead of a silent
   blank canvas when cytoscape/vis-network are unavailable.
2. ✅ **Escape all embedded data** (F2): the cytoscape `elements` are now
   serialized with `json.dumps` (numpy-safe) — this also fixes a latent bug
   where multi-line tooltips (`\n` in edge data) produced invalid JS. The
   report already used `json.dumps`.

### P1 — Runtime performance (done)
3. ✅ **Lazy reverse-adjacency in FastGraph** (F3): `_ensure_radj()` builds
   a predecessor index on first use and invalidates on edits;
   `find_paths_backward_dp`, `find_paths_memoized_dfs(backward)`,
   `find_paths_meet_in_the_middle` and `find_paths_bidirectional_bfs` now
   walk predecessors instead of building full `reverse()` copies
   (~250 MB saved per call on a 722k-edge graph). Verified: 750 randomized
   equivalence runs, zero mismatches.
4. ✅ **O(1) degree lookups** (F4): `in_degree()` uses the reverse index;
   `predecessors()` no longer scans all nodes.

### P2 — Maintainability & UX
5. **Template the network HTML** (F5): move the static JS/CSS into a
   separate template string (or `.js` asset read at runtime) with
   placeholders; the f-string then only interpolates data. This makes the
   JS lintable and reviewable (and would have prevented the brace-escaping
   risk in this session).
6. ✅ **Fix the orphan re-detection inconsistency** (F6): `applyEdgeFilter`
   now uses the same visible-edge definition as `toggleOrphanNodes`
   (`.hidden, .filtered, .selfloop-hidden`).
7. **Incremental network updates in the report** (F7): diff node/edge
   datasets before `clear()/add()` (vis supports `update()`), debounce the
   filter input, and drop the `setTimeout` re-layout when the topology is
   unchanged.
8. **Single source of truth for dead-end logic** (F8): compute the
   recursive closure once in Python, serialize the resulting role map
   (including dead-end flags) into the JS data, and let the JS only
   *apply* visibility — one implementation, no drift.

### P3 — Scale & hygiene
9. **Command-pattern undo** (F9): replace full-state snapshots with
   inverse operations (e.g. `{type:'hide', ids:[...]}`) when graphs exceed
   ~10k elements; keep snapshots for small graphs (simpler and robust).
10. ✅ **localStorage eviction** (F10): saved-layout keys
    (`cytoscape_layout_*`, `heatmap_settings_*`) are capped at the
    newest 20 per family **when a network page is loaded** (the eviction
    runs in the cytoscape template; heatmap pages do not evict on their
    own).

## Verification status of this session's features

- Hide Dead Ends: definition = out-only non-source / in-only non-target,
  based on visible edges; verified headlessly (20/20 assertions) with the
  actual generated JS: detection, toggle, edge-filter re-detection,
  sources/targets exemption.
- Undo/Redo: verified LIFO round-trips, Cmd/Ctrl+Z / Shift+Z / Ctrl+Y,
  input-focus skip, redo-stack clearing on new mutation.
- Auto-mirror: `hemisphere_mirror_default` now follows
  `separate_hemispheres` (None default) in `VisualizePath`, and the
  cross-dataset report initialises mirroring enabled + auto-applies it
  when Separate Hemispheres is on. Both verified in rendered output.
- Report dead code removed; both generators' JS pass `node --check`.
