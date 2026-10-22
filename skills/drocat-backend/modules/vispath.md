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
| `create_heatmap(custom_row_order=None, custom_col_order=None, cell_data=None, cell_fill=None, cell_outline=None, fill_scale=None)` | Heatmap only, from the built network; the four optional channels colour/border individual cells (see *Per-cell style*). |
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
- The "🏷️ Show/Hide Labels" toggle writes the **flat** Plotly attribute keys
  (`'colorbar.title'`, `'title'`, `'xaxis.title'`, `'yaxis.title'`), because
  `createHeatmap` stores those titles as plain strings — a nested
  `'xaxis.title.text'` write is two levels into a string container and Plotly
  throws `container is not an object`, in `restyle` and in `relayout` alike. The
  colourbar title the toggle restores is the string `createHeatmap` actually drew
  (`colorbarTitleText`, the selected metric's display name plus the scale label),
  never a recomputed one. Pages over 100 rows or columns open with the tick
  labels hidden; hidden *by the button* additionally clears the colourbar and
  axis titles, while a page that loads hidden keeps them.
- Showing the labels on such a page runs `fitCanvasForLabels(true)`: the canvas
  grows to `ticks × (fontSize + 2)` plus margins on BOTH axes (one pitch for both,
  so the square-cell lock cannot leave one axis under-spaced), capped at
  `LABEL_FIT_MAX_PX = 4000` per side, and `syncPlotSizeControls()` writes the new
  size back into the Plot Size panel — a page that resized itself while the panel
  still said 800 px would be the same lie as the old dead button. Hiding gives the
  room back **unless** the reader touched the size meanwhile: `updatePlotSize`,
  `updatePlotSizeFromInput`, `resetPlotSize` and `loadSettings` clear
  `labelFitSize`, because the square-cell hook re-derives the height after the
  relayout, so the fitted size cannot be compared against to decide authorship.
- PNG/SVG export captures the whole figure including the dendrograms.
- The deprecated `statvis._VisConnMatInteractive_local` fallback (used only
  when vispath_pkg cannot be imported) does NOT have the dendrogram feature;
  it emits an ImportWarning when used. Its label toggle re-renders rather
  than restyling, leaves the figure/axis titles standing, and does not fit
  the canvas for a large matrix. Until 2026-10-14 its emitted script read
  `is_sparse`/`sparsity`/`use_scatter_mode` — Python-only names — so the
  page drew nothing at all; `tests/core/test_statvis_coverage.py`
  (`TestVisConnMatInteractiveLocalRenders`) now renders it in headless
  Chrome, because eight string-level tests had kept it green while it was
  blank.

## Per-cell style channels (2026-10-17)

Filed by `clock-production` (`DROCAT_issue_heatmap_per_cell_style.md`): a Plotly
heatmap colours a cell only from `colorscale(z)` and has no per-cell border, so a
caller could not say "this pair is conserved" or "this one also appears in BANC"
on the matrix page. `VisConnMatInteractive` and `VisualizePath.create_heatmap`
now take four optional channels — rendering only, no computed number moves:

- `cell_data` — `{field_name: {(row_label, col_label): value}}`, one option per
  field in the reader's panel ("any data" at the page level);
- `cell_fill` — the caller's own colouring, and the panel's default source: CSS
  colour strings colour cells directly, anything else is a category priced by
  `fill_scale`;
- `cell_outline` — `{(row_label, col_label): True | {'width', 'color'}}`;
- `fill_scale` — `{category: '#hex'}`. A category with no entry **raises** rather
  than falling through to the unmapped grey, and a key naming a label no axis
  carries warns — silence there would read as an unstyled cell forever.

Keys are the two AXIS LABELS (`normalize_cell_style` joins them with `\t`, and
accepts the `(row, col)` tuple spelling too), never positions: transpose,
clustering, custom order and the axis subset all rewrite `displayXLabels`/
`displayYLabels`, and a node that is both source and target sits on both axes.
The page resolves a cell by `(source-side, target-side)` in the caller's
direction whichever axis those currently sit on.

The reader gets a **🧩 Cell style** panel: colour source (metric / any field),
one colour input per category (numeric fields are binned, 2–10 bins), outline
source, border colour and width, and Reset. A style change re-runs
`createHeatmap()`, which is the one place `Plotly.newPlot` is called — no
`restyle` of nested keys, the trap the label-toggle defect showed. While a fill
channel owns the colour the metric's `zmin`/`zmax` and the reader's colour range
are deliberately not applied (the scale is hard stops, one per category); the
metric keeps its numbers in the cell text, the hover and every table.

With no channel passed, every emitted chunk is the empty string — the default
page is byte-identical to the pre-feature page (certified against the old
module; keeps `tests/core/test_profiling_multi_dataset.py`'s `xgap`/`ygap` law
and "no `layout.shapes`, single heatmap trace" true).

`ComparisonAnalyzer.visualize_conserved_paths(...)` is the wired caller:
`cell_style_fill_datasets` / `cell_style_outline_datasets` read the per-dataset
weights it already built (`'{nickname} wt'` per reporting dataset), fill =
reported by **every** dataset in the fill set, outline = reported by **any** in
the outline set, and a dataset name that reports nothing in that matrix raises
instead of greying the whole page.

Laws: `tests/vispath/test_heatmap_cell_style.py` (normalizer, panel, transpose,
clustering, custom order, axis subset, escaping, default page) and
`tests/core/test_conserved_network_cell_style.py` (the producer's maps).

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
  once (`cy.fit(null, 50)`, lifted to a readable 0.35 zoom floor on large
  graphs where exact framing would clamp to `minZoom`); a SAVED layout
  auto-restores its own zoom/pan instead (deferred 250 ms, after the
  hemisphere cache). Re-layouts never re-fit (`config.fit = false` choke
  point); ⛶ Fit and the import toasts are the explicit fits (exact
  framing). Tests pin `js.count("cy.fit(null, 50)") == 4` and
  `"cy.fit();" not in js`.
- **Edge Anchor ribbon group** (beside Edge Route) — `Center` (default,
  classic center-directed attachment on the center-line rim) vs `Auto`: every
  edge end defaults to the nearest of 8 anchors — the 4 corners and the 4
  mid-edges, each exact for its shape (polar rim for ellipses, corner arcs
  for round-rectangles at cytoscape's draw radius min(width/4, height/4, 8),
  box corners for sharp rectangles — via `rimDistance`), chosen by
  45°-sector direction (`snapAnchorIndex`/`anchorPoint`/`ANCHOR_CENTER`,
  shared top-level geometry helpers). Applies in ALL edge modes (straight,
  curved pairs keep the cp-40 bow, merged canonical half, flow bows keep their
  center-line obstruction test); waypoint (segments) and self-loop edges
  are excluded.
- **Per-end overrides (both modes adjustable)** — selected edges show
  draggable endpoint handles in BOTH modes: a blue diamond for each of the
  8 compass anchors and a round teal badge for the NINTH, center-directed
  anchor (`ANCHOR_CENTER = 8`) — the badge sits AT THE NODE CENTER while
  the edge end anchors to the near rim intersection (the rim point facing
  the other node). Dragging around the node snaps to the nearest compass
  anchor; dragging INTO the node body pins the center-directed attachment. A drag reassigns THAT end on the grabbed
  edge AND, as a SYNCHED MULTI-EDIT, the same anchor on that end of every
  other selected eligible edge (each resolving against its own geometry;
  one coalesced history entry — undo reverts the whole selection) by
  writing `anchorSrc`/`anchorDst` data + the endpoint style; the other
  ends stay untouched, and reciprocal pairs keep their parallel
  offset-line look unless overridden. `↺ Reset anchors` in the selection panel clears
  overrides. Mode + overrides persist through
  `captureNetworkState`/`applyNetworkState` (localStorage saves and the v2
  layout file verbatim). Node shape changes re-derive the rim pins. Rim formulas are shape-exact:
  polar rim for ellipses, corner arcs for round-rectangles (draw radius
  min(width/4, height/4, 8)), ray-vs-rhombus for diamonds (the 45° diagonal
  anchors land on the rhombus edge midpoints), box corners for sharp
  rectangles.
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
- **The trim must carry the pair-level metrics** (audit 2026-10-16). A trimmed
  page's edge records are re-derived from the PATH ROW
  (`_path_edge_records`: weight/ratio/probability), while `ratio_adj` and every
  `pair_extra_metrics` name live on the graph the caller built — so before this
  rule a run over the cap drew both metrics as `N/A` on a folder whose data
  files had them (measured on `R3_complete_and`: 396 emitted pairs, limit 200,
  `refilled` non-null on 0/200 edges and `ratio_adj` on 0/200). The records now
  copy the caller's own metric names (plus `ratio_adj`) off `G_network`. A run
  small enough to skip the trim always showed them, which is why every unit test
  passed: the law is pinned on an EMITTED page whose fixture forces the trim
  (`tests/vispath/test_metric_labels.py`), with the pre-fix record builder
  re-run as the control
  (`_plan/probes/2026-10-16-review/trim_metric_control.py`).

## Metric bases (V1, 2026-10-10)

Edges carry up to four parallel metrics on the same topology:
`weight` (Synapses), `ratio` (all-inputs share), `ratio_adj`
(adjusted — weight over the post type's >= THRESHOLD incoming; equals
`ratio` when the threshold is 1), and `refilled` (pair-level refill
totals, present only when the run's refill mirror exists). The heatmap
metric toggle and the network/Sankey Connection-Metric selectors expose
whichever of these the input provides; hover always lists every metric;
Sankey labels stay weight-based (metrics on hover); NT coloring, when
enabled, overrides edge COLORS only. Feed pair-level metrics via
`VisualizePath(pair_extra_metrics={metric: {(src, dst): value}})` —
METRIC-keyed, the shape both DROCAT callers pass and the constructor's own
comment now states;
per-hop adjusted ratios ride the path frame's `connection_ratios_adj`
list column (positional, like `connection_ratios`).

### The transmitter (NT) buckets on the network page

`color_edges_by_nt=True` writes one Cytoscape rule per transmitter the data
carries (`edge[nt_type = "GLUT"]`), and — since the user report of 2026-10-17 —
one more for the bucket `nt_bucket()` calls `'unknown'`: `None` / `NaN` / blank /
'None' / 'nan' / 'n/a' cells all land there, in the palette's
`NT_COLORS['unknown']` gray (`#95A5A6`), with an `Unknown Edges` entry in the
group selector so the reader can recolour or hide it. Before that rule existed an
unannotated edge matched nothing and kept the page's base link colour — on DROCAT
runs `rgba(100,150,240,.5)`, a light blue — so "this pair carries no transmitter
information" was drawn in the same visual language as a real transmitter.

Two boundaries, both pinned in `tests/vispath/test_nt_unknown_edges.py`: the
bucket rule is emitted ONLY when the page has some real vocabulary to be gray
relative to (a run with no transmitter column at all keeps its uniform edge
colour rather than a page that claims everything is unknown), and it is not
emitted at all when `color_edges_by_nt` is off (the payload then keeps the raw
column, blank included).

### Edge-arrow legend (2026-10-22)

`VisualizePath(network_edge_legend=[…])` draws a caller-supplied legend as
**arrow samples** (a `<line>` + triangular `<polygon>` per item, never colour
swatches) on the network page, in the network's own SVG export and as a
standalone `edge_legend.svg` download:

```python
network_edge_legend=[
    {"title": "Neurotransmitter", "items": [
        {"label": "ACh", "color": "#FFA500"},
        {"label": "Glut", "color": "#008000"},
        {"label": "mixed", "color": "#95A5A6"}]},
    {"title": "Synapse strength", "items": [
        {"label": "1–20 (20%)", "color": "#6b7280", "opacity": 0.2},
        # … 21–50 / 51–100 / >100
    ]},
]
```

- `label` is required; `color` defaults `#333333`; `opacity` clamps to 0–1
  (written as `stroke-opacity` / `fill-opacity`); `line_style` is `solid`
  (default), `dashed` or `dotted`.
- `edge_legend_position` chooses the exported SVG's corner (`bottom-left`
  default; the other three corners are accepted). `edge_legend_page_position`
  (default `top-left`) chooses the on-page card — the canvas bottom sits below
  a 900 px viewport, so the page follows the pilot's top-left placement while
  the export keeps the requested bottom-left.
- The legend is appended **before** `sanitizeSvgForOffice`, so the Office
  rewrite applies to it; every `<text>` carries `font-family="Arial"`
  explicitly and there is no `<image>` / `<foreignObject>`.
- The sanitized export then keeps the legend as ONE standalone root group
  (`edgeLegend`) holding `edgeLegendShapes` and `edgeLegendTexts` subgroups,
  rather than dissolving it into the network's global `#shapes` / `#texts`
  groups. The standalone `edge_legend.svg` uses the same shapes/texts split.
  Item labels stay individual `<text>` boxes (the pre-sanitize interleaving is
  what stops the Office multiline-label merge).
- A page without `network_edge_legend` emits none of the overlay, spec,
  buttons or builder JS and its SVG export is unchanged.
- Pinned by `tests/vispath/test_network_edge_legend.py`: live Chrome checks
  the standalone counts, the sanitized export, all four anchor positions and
  the overlay geometry.

### `metric_note`: a page's own word about a metric it cannot carry

`VisualizePath(metric_note='…')` renders one muted line under the Connection
Metric select on the network and Sankey pages, and renders NOTHING when it is
empty (the default) — so a page that has every metric never shows an
explanation of a gap it does not have. DROCAT uses it where a bodyId-level page
greys out Refilled and the adjusted ratio: both are computed per TYPE pair, so
the page says who measures them and where the records live, rather than leaving
`(N/A)` as the only signal (review 2026-10-17). The caption is caller data: it
goes through `html_escape`, and `tests/vispath/test_metric_labels.py` pins that
it lands under `id="metricSelect"` on both page kinds.

## Notes

- The input may be CSV or Excel; for Excel use `sheet_name="path_type"` /
  `"path_bodyId"` when autodetection is ambiguous.
- Use a completed FindPath/comparison artifact rather than a hand-written file
  unless the schema has been verified.
- `generate_empty_network=True` requires `path_file=None`.
- For large graphs use `network_layout="distributed"` or `"hierarchical"`; keep
  `showfig=False` for headless runs and open the returned `*_network.html` manually.
