# vispath — VisualizePath

Module `vispath-subproject/src/vispath_pkg/vispath.py` (imported as `vispath_pkg`;
not installed into site-packages — add `vispath-subproject/src` to `PATH`).
`VisualizePath` renders interactive Cytoscape network, Sankey, and heatmap HTML
from a path/connection table, and can build a blank editable network canvas.

## Constructor (key params)

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file="/abs/output/paths/path_data.csv",    # CSV or XLSX; or None for empty canvas
    output_folder="/abs/output/network",
    network_layout="hierarchical",      # spring, circular, distributed
    source_color="#4A90E2",
    intermediate_color="#50E3C2",
    target_color="#B8E986",
    link_color="rgba(74,144,226,0.3)",
    showfig=False,
    output_format="csv",
    generate_empty_network=False,       # True → blank editable Cytoscape canvas
)
```

## Key methods

| Method | Purpose |
| --- | --- |
| `visualize(plot_heatmap=True, plot_Sankey=True, plot_network=True)` | Render all three HTML views; returns `(connections, graph)`. |
| `visualize_heatmap(custom_row_order=None, custom_col_order=None)` | Heatmap only. |
| `visualize_sankey()` | Sankey only. |
| `visualize_network()` | Network only. |
| `visualized_paths_for_export()` | Connection table for export. |

## Interactive heatmap page (`VisConnMatInteractive`)

The standalone heatmap HTML behind `plot_heatmap` is built by
`VisConnMatInteractive(cmat, filename, ...)` (also called directly by the
profiling and comparison pipelines). Since 2026-09-29 it ships the
hierarchical-clustering linkage matrices (ward/average/complete/single) in the
page payload and draws **dendrograms** beside the clustered heatmap: column
tree above, row tree right, controlled by the "🌳 Dendrogram" button with tree
line-width and band-size sliders (all persisted per page in localStorage).
Key behaviors:

- The trees re-render on clustering-method switch, transpose, and
  Original/Clustered toggles; leaves are anchored to the displayed order.
- The bands coexist with the square-cells lock; after each constrained render
  the band edges re-align onto the heatmap's true cell edges
  (`alignDendrogramBands`), so the trees stay flush.
- Grid lines behind the trees are hidden while the bands are shown.
- `init_clustered=False` ships the linkage data but opens in original order;
  `square_cells=True` auto-locks cells at first render.
- PNG/SVG export captures the whole figure including the dendrograms.
- The deprecated `statvis._VisConnMatInteractive_local` fallback (used only
  when vispath_pkg cannot be imported) does NOT have the dendrogram feature;
  it emits an ImportWarning when used.

## Library embedding (offline contract)

`VisConnMatInteractive(..., inline_plotly=False)` — the plotly.js source has
two modes:

- `inline_plotly=True` embeds the version-matched `plotly.min.js` (from the
  installed plotly package) INTO the html: a single-file page, immune to any
  later relocation (+~4.8 MB per file). Find-paths `create_heatmap()` /
  `create_heatmaps()` pass `True` — their `Heatmap_<run>.html` files are
  fully self-contained.
- `inline_plotly=False` (default) shares ONE `plotly.min.js` per output
  directory (statvis matrix pages emit many htmls per run) and adds a
  parse-time fallback: if `window.Plotly` is missing the page retries a copy
  one directory up (`../plotly.min.js`).

Relocation contract: htmls and the side-file library must stay co-located —
`coana._organize_vispath_artifacts` moves `plotly.min.js` together with the
relocated heatmap (fixed 2026-10-06: a library left at the run root blanked
every relocated `visualization/Heatmap_*.html`). CDN strings are forbidden
in generated pages (the offline tests regex-scan for `https?://` script
srcs, even inside JS `document.write`).

## Network page (Net-Viz): view behavior + Edge Anchor

The cytoscape page emitted by `_plot_cytoscape_network` (committed 2026-10-07):

- **Open view** — with no layout saved for the generated file, the page fits
  once (`cy.fit(null, 50)`, clamped by `minZoom` on large graphs); a SAVED
  layout auto-restores its own zoom/pan instead (deferred 250 ms, after the
  hemisphere cache). Re-layouts never re-fit (`config.fit = false` choke
  point); ⛶ Fit and the import toasts are the explicit fits. Tests pin
  `js.count("cy.fit(null, 50)") == 4` and `"cy.fit();" not in js`.
- **Edge Anchor ribbon group** (beside Edge Route) — `Center` (default,
  byte-identical classic attachment on the center-line rim) vs `Auto`: every
  edge end snaps to the nearest of 8 anchors — the 4 box corners (ellipses:
  the exact 45° rim points via `rimDistance`) and the 4 mid-edges — chosen
  by 45°-sector direction (`snapAnchorIndex`/`anchorPoint`, shared
  top-level geometry helpers). Applies in ALL edge modes (straight, curved
  pairs keep the cp-40 bow, merged canonical half, flow bows keep their
  center-line obstruction test); waypoint (segments) and self-loop edges
  are excluded.
- **Per-end overrides** — selected edges show draggable ◇ endpoint handles
  (DOM overlays, waypoint-ring pattern). A drag reassigns THAT end only
  (reciprocal halves independently) by writing `anchorSrc`/`anchorDst`
  data + the endpoint style; reciprocal pairs keep their parallel
  offset-line look unless overridden. `↺ Reset anchors` in the selection
  panel clears overrides. Mode + overrides persist through
  `captureNetworkState`/`applyNetworkState` (localStorage saves and the v2
  layout file verbatim). Node shape changes re-derive the rim pins.
- **Waypoints** — bends declare `edge-distances: 'node-position'` so the
  rendered bend lands EXACTLY on the stored point (cytoscape's default
  `'intersection'` basis spans the rim-to-rim chord and drifts the bend by
  up to a rim radius near the ends — fixed 2026-10-07, regression-tested in
  `edge_anchor_harness.js` stage [7]). Selecting an edge reveals slate
  passive rings on its waypoints (no click near the bend needed); the amber
  click-to-bundle behavior is unchanged and suppresses the passive ring.
- **In-page exports** — SVG (`sanitizeSvgForOffice`: every element closed
  filled geometry, no `<line>`, arrowheads as single triangles, waypoint
  bends included), PNG, edge-list CSV, graph JSON, and the v2 layout JSON
  (carries `edgeAnchor: {mode, overrides}` — added 2026-10-07).

## Examples

```python
# Standard: render a completed FindPath/comparison CSV
from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/paths/path_data.csv",
                   output_folder="/abs/output/network",
                   network_layout="hierarchical", showfig=False)
connections, graph = vp.visualize()

# Empty editable canvas
vp = VisualizePath(path_file=None, output_folder="/abs/output/empty_network",
                   generate_empty_network=True, network_layout="hierarchical",
                   showfig=True)
vp.visualize()
```

## Edge cap and artifact naming (pathfinding runs)

- `_select_edges_for_plot` is the shared selector behind the Visualization Edge
  Limit (`edgeN_limit`): a drawing-only cap on the number of unique edges
  rendered per HTML view. It never changes fetch, graph, or path outputs, and a
  single complete path may exceed the cap to stay intact. Type-level and
  bodyId-level visualizations share the same cap.
- `_organize_vispath_artifacts` (generalized in `src/coana.py`) fixes the
  canonical naming contract for both levels: `Network_<run>.html`,
  `Heatmap_<run>.html`, `Sankey_<run>.html`, and
  `visualization_data/<run>_data_*.csv` inside `visualization/` and
  `bodyId_visualization/` (raw legacy names such as
  `bodyId_visualization_network.html` no longer remain). Early previews keep
  `network_early/` and `network_early_bodyId/` with `Network_<run>.html` inside.
- When the drawing cap trims, a companion CSV lists what was rendered:
  `visualization/visualization_data/type_paths_visualized.csv` (type level) and
  `bodyId_visualization/visualization_data/bodyId_paths_visualized.csv`
  (bodyId level). Missing artifact types stay absent.

## Notes

- The input may be CSV or Excel; for Excel use `sheet_name="path_type"` /
  `"path_bodyId"` when autodetection is ambiguous.
- Use a completed FindPath/comparison artifact rather than a hand-written file
  unless the schema has been verified.
- `generate_empty_network=True` requires `path_file=None`.
- For large graphs use `network_layout="distributed"` or `"hierarchical"`; keep
  `showfig=False` for headless runs and open the returned `*_network.html` manually.
