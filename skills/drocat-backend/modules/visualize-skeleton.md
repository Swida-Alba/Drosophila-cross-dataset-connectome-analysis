# visualize_skeleton — VisualizeSkeleton

Module `src/visualize_skeleton.py`. `VisualizeSkeleton` produces interactive 3D
plotly HTML for neuron skeletons, synapses, brain-region ROI meshes, and supports
PNG/PDF/PPTX/GIF/video exports. `WebDriverExportSession` (in the same module)
drives Chrome for PNG/video when `export_method="webdriver"`.

## Constructor (key params)

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset="male-cns:v0.9",
    neuron_layers=[["aMe12"], ["aMe10", "aMe9"]],   # nested list model
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    hemisphere="both",
    custom_layer_names=[],              # optional layer labels
    output_dir="/abs/output/skeleton",
    output_format="csv",                # merged synapse export: "csv" | "xlsx"
    skeleton_mode="line",               # "line" | "tube" (start with line)
    brain_mesh="native",                # "native" | "BANC" | "FAFB" | "male-cns" | "none" (all five are the whole vocabulary; pre-rename 'template'/'whole' are rejected)
    vnc_mesh=None,
    legend_mode="layer",                # "single" | "type" | "tree" | "layer"
    freeze_view=True,                   # viewer pages pin the 3D framing on legend toggles
    neuron_alpha=1.0,
    neuron_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    synapse_colors=["#50E3C2"],
    background_color="#ffffff",
    skip_synapse=False,
    min_synapse_num=3,
    synapse_size="real",                # "real" or a numeric size; uniform sizing uses uniform_synapse_size
    uniform_synapse_size=False,
    synapse_alpha=1.0,
    synapse_mode="scatter",             # "scatter" | "sphere" | "cone" | "tetrahedron"
    mesh_roi=["EB", "LH", "AL"],
    mesh_color=["#4A90E2", "#50E3C2", "#B8E986"],
    mesh_alpha=0.1,
    cache_neurons=True,
    cache_synapses=True,
    smooth_skeleton=True,
    show_soma=False,
    show_connectors=False,
    export_method="webdriver",          # or "kaleido"
    export_scale=2,
    export_views=False,
    show_fig=False,
    brain_mesh_color=None,
    neuprint_skeleton_pipeline="fast",  # "fast" | "fine" | "artistic" | "direct" | ...
    skeleton_mesh_simplification=0.0,
)
```

## Key methods

| Method | Purpose |
| --- | --- |
| `plot_neurons()` | Build the interactive HTML (main entry). |
| `plot_individuals(pdf_images_per_page=(3,2), views=["front"], summary_format=["pdf"], granularity="legend", neuron_alpha=None)` | Profile export, one profile per `granularity` level: `"legend"` \| `"layer"` \| `"type"` \| `"body"` (bodyId leaf). Companion soma meshes and pre/post sites follow their owner; meshes, synapse markers and legend swatches are background. `views` are the lowercase camera names (`front` / `back` / `top` / `bottom` / `left` / `right`; `all` expands to the scene's table); capped at `MAX_INDIVIDUAL_PROFILES` (300) renders = groups × views, over which it renders the leading groups that fit and logs the skipped ones. |
| `export_video(fps=30, degree_per_frame=1.0, rotate="horizontal", export_gif=True, gif_scale=0.2, html_file=None, ...)` | Rotating video/GIF export; `html_file` re-exports a stored page. |
| `visualization_manifest()` / `write_visualization_manifest()` | Describe the scene for later re-export (`visualization_manifest.json` in the run folder: `canonical_page`, `legend_mode`, `freeze_view`, render settings, `views` cameras, and a per-trace role table the re-exporter re-reads as an HTML round-trip check). |
| `list_available_rois(refresh=False, fetch_online=True)` | List available ROI meshes for the dataset. |

```python
vs = VisualizeSkeleton(dataset="male-cns:v0.9", neuron_layers=[["aMe12", "aMe10"]],
                       output_dir="/abs/output/skeleton", skeleton_mode="line",
                       mesh_roi=["EB"], show_fig=False, skip_synapse=True)
vs.plot_neurons()
# optional, after the base HTML succeeds:
vs.plot_individuals(pdf_images_per_page=(3, 2), views=["front"], summary_format=["pdf"])
vs.plot_individuals(granularity="body")   # one profile per bodyId leaf
vs.export_video(fps=30, export_gif=True, gif_scale=0.2)
vs.export_video(html_file="/abs/output/skeleton/<run>/scene.html")  # re-export a stored page
```

## Supporting functions

- `figure_payload_from_html(html_path)` — read `(data, layout, config)` back out
  of an exported page. plotly 6.4 has no `plotly.io.read_html`, so DROCAT parses
  the page's own `Plotly.newPlot(...)` call, scanning from the end of the file
  because the embedded plotly.js bundle contains the same literal.
- `figure_from_plotly_html(html_path)` — rebuild a `go.Figure` from a stored
  page (used by both `export_video(html_file=...)` and
  `export_video_from_html(...)`).
- `classify_traces(traces, mesh_roi_names=())` — resolve each trace's role
  (`neuron` / `companion` / `site` / `synapse` / `mesh` / `legend_swatch`) and
  legend identity, `meta.drocatTrace` first (written by `_stamp_trace_identity`
  / `_stamp_site_identity`, with `meta.drocatLegend` as the tree-legend tag),
  then structural signals, then names. Never decide a trace's role from its
  name alone: a neuron type can contain an ROI acronym (`LHPD1L` vs `LH`).
- `PROFILE_GRANULARITIES` — `('legend', 'layer', 'type', 'body')`, the levels
  `plot_individuals(granularity=...)` accepts (anything else raises). `layer` /
  `type` / `body` read the stamped identity, so they work in any `legend_mode`.
- `read_visualization_manifest(path)` — the run's `visualization_manifest.json`
  as a dict, or `{}` for a run written before the manifest existed (page, folder
  or manifest path all accepted).
- `manifest_role_drift(manifest, roles)` — advisory one-liner when a page
  classifies as different roles than the manifest's `traces` table recorded,
  i.e. its `drocatTrace` stamps did not survive the HTML round trip. `None` when
  the run recorded no table or the two tallies agree;
  `export_individuals_from_html()` prints it and exports anyway.
- `resolve_viewer_page(path)` — the canonical viewer page of a run folder or
  page; a `_simplified.html` copy is resolved back to the canonical page
  (`visualization_manifest.json` → `canonical_page`, else the first
  non-`_simplified`, non-`_`-prefixed HTML in the folder sorted by name), and
  raises rather than silently re-exporting degraded geometry when none exists.
- `profile_plan_from_html(html_path, granularity='legend')` —
  `(entries, roles)` for a stored page: the same `classify_traces` +
  `VisualizeSkeleton._build_profile_plan` pair the live renderer uses, so a
  re-export offers exactly the levels that run would have produced.
- `export_individuals_from_html(html_path, output_dir=None, granularity='legend',
  views=None, scale=2, auto_crop=True, crop_margin=30, background_color=None,
  timeout=60)` — re-export a stored page's profiles with no dataset access:
  reads the page, classifies it, and renders one PNG per profile group through
  one `export_individuals_webdriver` session (cameras from the manifest's
  `views`, else `dataset_view_cameras(dataset, brain_mesh)`). Returns that
  helper's `{success, files, failed, error}` dict; PNGs only, no PDF/PPTX
  summary (those are built by `plot_individuals`).
- `export_video_from_html(html_file, fps=30, degree_per_frame=1.0,
  rotate='horizontal', export_gif=True, gif_scale=0.2, auto_crop=False,
  export_method='kaleido', timeout=120, background_color=None, scale=2)` —
  rotating mp4/gif from a stored page. `export_method='webdriver'` renders every
  frame through a single `WebDriverExportSession`
  (`_render_video_frames_via_session`, one page load plus `_rotation_camera`
  per frame, orbit handedness from `_page_z_sign`) and falls back to kaleido
  when no browser can be started; `background_color=None` takes the value the
  run recorded. Writes `pics_{fps}fps_{plane}/`,
  `{stem}_video_{h|v}_{forward|backward}.mp4` and the matching GIFs.
- `reexport_output_dir(path)` — the default destination for everything
  re-exported from a page: `<page stem>_reexport/` beside it, so the source run
  is never rewritten and a second run with the same settings reuses the frames
  already on disk. The UI's re-export card runs these helpers as the
  `plot3d_reexport` tool (`ui/runner.py`).

## Notes

- Line-mode somas: navis draws the soma sphere only at `neuron.soma`, which it
  derives from the SWC `label==1` node — and then loses, because every DROCAT
  skeleton loader stamps `units` after `read_swc` and navis 1.5.0's units
  setter clears the assignment (measured: 0 of 150 sampled male-cns and 0 of
  150 hemibrain cached skeletons arrive with a soma). So before line node
  reduction the renderer resolves the marker itself, in order: the `label==1`
  row still in the node table, provided it has a radius to render (62.7% of
  male-cns files carry the row and every sampled one had positive radius; the
  radius heuristic disagreed with the marker on 14% of those), else the node
  nearest the NeuPrint `somaLocation` annotation when the layer table carries
  one, else the fattest radius node (skipped on BANC — its skeletons have no soma
  signal: labels all `2`, root thinner than neurites) — and at save time any
  tagged sphere below `LINE_SOMA_MIN_VISIBLE_FRACTION` (0.0055) of the frozen
  scene's longest axis is grown to that floor, while a sphere with NO extent is
  left alone: it cannot be scaled into one, and dividing by it once raised
  straight through `save_figure` and cost stage 4 the whole scene (5 of 21
  parents on a 2026-09-24 male-cns family run). Line soma sizes are display
  choices, not measurements. `show_soma=False` disables all of it.
- BANC (`banc_v626` / `banc_v888`) is supported and renders in native BANC
  space from the public release bucket (no token): SWC skeletons, ROI meshes
  from the public `region_outlines` layer, and brain/VNC outline templates
  (`brain_mesh="native"` — or the explicit `"BANC"` — shows the brain
  portion, `vnc_mesh=True` the VNC
  portion; both cut at the neck coordinate y = 350,000 nm). Current mesh
  choices are `native`, `BANC`, `FAFB`, `male-cns`, `none`; the old
  `template`/`whole` names remain accepted as compatibility aliases. Saved
  HTML/PNGs open on the calibrated BANC frontal view (anterior at -Y).
- BANC knobs: the skeleton chain is unified — 888 L2 first, else the 888
  full-resolution skeleton, else the v626-era pcg-skel set (µm scaled to
  nm). L2 tubes skip face decimation (cache-level product); full-res tubes
  decimate FAFB-style, floored at 4,000 kept faces. The deprecated
  `banc_skeleton_resolution` argument is accepted but ignored. Tube radii
  are normalized to a 240 nm median by default
  (`banc_normalize_radius`, `banc_radius_target_nm`).
- BANC synapses default to `skip`: opting in downloads a ~3.9 GB per-synapse
  table once (resumable) and draws pre-synaptic site markers only (the
  release publishes no post-site coordinates).
- Mesh fixes are dataset-specific; verify ROI availability and coordinates before
  changing transforms.
- `export_method="webdriver"` needs Chrome + WebDriver (Kaleido is the slower
  fallback). Use `skeleton_mode="line"` and no exports for a first smoke test.
- `legend_mode="tree"` renders like `"type"` for static exports and adds the
  collapsible group/type/bodyId panel to the permanent HTML pages only; counts
  are unique neuron items, not Plotly traces.
- `brain_mesh`/`vnc_mesh` choose what a page opens *showing*, not what it
  contains, and not the scene's coordinate space either: `_needs_skeleton_transform`
  answers from the dataset's source→render pair, so a `male-cns`/`hemibrain`
  scene still takes the raw-voxel → nm affine (×8) with `brain_mesh='none'`.
  It used to answer False for `'none'` on the reasoning that no envelope means
  no frame to match, which left the whole anatomy 8x too small in the wrong
  corner next to the envelope the page always embeds. A transform that fails
  twice now sets `_skeleton_transform_disabled` rather than rewriting
  `brain_mesh`.
  On `male-cns` and `banc` — whose native template is one volume
  split at the neck — whatever the two checkboxes leave unshown is still embedded,
  hidden (`_embed_unshown_context_mesh` → `_embed_context_mesh`): a
  `brain_mesh='none'` run embeds both halves, a `none` + `vnc_mesh=True` run
  embeds the brain. Decimated to
  `CONTEXT_MESH_EMBEDDED_TARGET_FACES` (20,000 faces; measured halves: native
  VNC 36,670 faces / 0.871 MB of page → 20,001 / 0.471 MB, BANC VNC
  38,200 → 19,999 / 0.481 MB), and absent from `exportable_meshes` so it never
  reaches the GLB/OBJ. It uses `visible=False`, not `'legendonly'`, because the
  tree reads legendonly as shown. Only these two datasets have a second half;
  FAFB, hemibrain, optic-lobe and MANC embed nothing extra (verified on a real
  FAFB page: 11 traces, one shown envelope, no hidden half). A cross-template
  scene never falls back to the native split — wrong coordinates.
  `_is_unshown_context_mesh` is the
  single predicate for "present but not shown"; it keeps such a mesh out of
  the profile-plan background list and the tree's restore baseline, but *not*
  out of the frozen box — see `freeze_view` below. `_note_hidden_context_meshes`
  appends the names to
  `parameters.txt` after the page is written — the parameter block itself is
  composed during initialization, before any trace exists.
- `freeze_view=True` pins the permanent viewer pages' scene axes to the padded
  extents of every trace (hidden ones included, span/32 per axis) so legend and
  tree toggles cannot rescale the scene, and pins `aspectmode='manual'` at the
  ratio those ranges imply — `aspectmode='data'` recomputes the x:y:z box from
  the visible traces on every redraw, so ranges alone let one hidden mesh change
  the anatomy's shape. A relayout/restyle listener re-applies the pin after the
  page's
  reset-camera button or a double-click, which restore the pre-script layout
  snapshot and silently drop it; Plotly's second reset button is removed from
  the toolbar (`VIEWER_MODEBAR_BUTTONS_TO_REMOVE`, keyed by Plotly's
  button-registry name `resetCameraDefault3d`, not the rendered `data-attr`). A
  second ⌖ control (and the
  `C` key) moves `scene.camera.center` onto the visible traces, because a frozen
  box keeps its pivot at the centre of the whole scene; it is inert in fit mode,
  and Fit itself clears that center (the offset is normalized to the pinned box)
  while re-freezing restores it unless the user panned. A page that opens with a
  context half embedded but hidden runs that same control at setup -- the writer
  passes `hidden_half` from `_unshown_context_mesh_indices`, and the script reads
  it as `CONFIG.hiddenHalf` -- so the orbit starts on the half it draws instead
  of on the gap between the two: a 40-degree turn slid the brain 212 px across a
  male-cns frame before, 2 px after. With nothing hidden the flag is false and
  the pivot is left where Plotly puts it.
  A Freeze/Fit button and the `F` key
  hand autoscaling back. All three floating controls -- those two and the
  light/dark switch -- show a hover hint naming the key and the current state,
  drawn from `data-drocat-tip` by page CSS rather than a native `title`, one row
  per hint (`width:max-content` under a `min(560px,88vw)` cap: the 36 px button
  is the tip's containing block, so shrink-to-fit otherwise stacks a word per
  line); the
  switch's copy names the theme the click selects ("Switch to the dark theme
  (T)") and is drawn to its left, since that button owns the top-right corner. The pin is injected page JS that exits early under
  `navigator.webdriver`, so PNG/video/profile exports keep autoscaling and draw
  the same content (measured against the pre-feature generator on one male-cns
  scene: same 3 layer profiles and 2 views, 0.067% of pixels different, every
  one of them within 4 px of pre-existing ink — a hidden 29th trace nudges the
  alpha compositing at edges, and two runs of the same generator are
  byte-identical). An
  embedded context mesh widens the first frame and is then never able to move a
  frozen one: `_scene_data_ranges` spans it, so the page has exactly **one** box
  and it already contains the half the viewer can reveal. There is no second box
  and no box-switching code. Measured on a male-cns brain-only page: 0.3% of the
  embedded cord's vertices fall inside a brain-only box, so the earlier two-box
  design's toggle drew nothing; with the union box the cord comes on and goes off
  with all three axis ranges, `camera.center`, `camera.eye` and `aspectratio`
  unmoved and 0.37% of the frame's pixels different, at a cost of 1.46× zoom-out
  along the longest axis. `repaint()` repairs a lost pin on `plotly_relayout`
  *and* `plotly_restyle` (a tree eye is a restyle), and returns unless
  `drifted()` — because growing the framing under a freeze is a camera move, and
  Fit is the control that follows content.
- `synapse_size` accepts `"real"` or a numeric value; uniform sizing uses the
  `uniform_synapse_size` bool. Invalid/empty values fall back to `"real"`.
- `cache_neurons`/`cache_synapses` persist raw skeletons/synapses, reused by the
  Settings skeleton pull.
