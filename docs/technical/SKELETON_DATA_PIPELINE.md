# Skeleton Data Pipeline — Technical Report

**Scope:** neuron-skeleton handling for the three connectome dataset sources
(FAFB local release, male-cns, and standalone BANC), brain + VNC template
meshes, and ROI meshes —
fetching, caching, storage, simplification, and scene-space handling.

**Diagrams:** [skeleton_pipeline.html](../visualizations/skeleton_pipeline.html)
(data flow across the three dataset families, incl. the ROI-mesh stage),
per-dataset skeleton-handling flowcharts —
[FAFB](../visualizations/fafb_skeleton_flow.html) ·
[male-cns](../visualizations/malecns_skeleton_flow.html) ·
[BANC](../visualizations/banc_skeleton_flow.html) — and
[banc_roi_resolution.html](../visualizations/banc_roi_resolution.html)
(exact decision flow for one `mesh_roi` entry on a BANC scene, first run vs
cache used). Static preview: `skeleton_pipeline.png`.
Generators: `generate_skeleton_flows.py` in the same directory.

This report merges and supersedes
`docs/visualizations/FAFB_NEUPRINT_SKELETON_PIPELINE_PLAN.md` and the
pipeline sections of `docs/visualizations/Skeleton_Data_Pipeline.md`
(both superseded files have since been removed).

---

## 1. Sources, fetching, and fallback chains

The three dataset families (NeuPrint, BANC, FAFB) **deliberately keep
independent fetch pipelines** — their sources, storage, representations, and
formats differ. Sharing is limited to the representation-level layers that
are genuinely common: the compressed-SWC provenance contract
(`skeleton_provenance.py`: `# DROCAT simpl:` / `# DROCAT source:` headers,
parsers, the raw-store path layout), the feature/vector computation, and the
vector-cache machinery. The fetch entry points (`morphology.fetch_skeleton_on_demand` /
`morphology.fetch_skeletons_on_demand_batch`) are skeleton-native for
every dataset kind: FAFB resolves in one call through
`load_local_release_skeletons` (repair caches -> raw cache -> healed zip ->
extrusion check/repair -> CAVE skeletonization), NeuPrint batches through
`_fetch_neuprint_skeleton_batch`, BANC through `_fetch_banc_skeleton_batch`
and the shared raw cache. The fetchers only own the dataset-agnostic cache
transaction around them. (Until 2026-09-29 the FAFB branch served CAVE
prepared MeshNeurons while a skeleton-serving block sat unreachable behind
an unconditional mesh return — the mesh-where-skeleton trap behind the
reverse-scene defect record; it was removed and FAFB now delegates to the
loader.) Do not merge the fetch layers into a single strategy: the
differences (auth, batching, staging, representation) are real.

### 1.1 FAFB / flywire (`flywire_FAFB_v783`)

Decision flow: [fafb_skeleton_flow.html](../visualizations/fafb_skeleton_flow.html).

Every render pipeline (fast/fine/artistic tube and line) consumes
TreeNeurons. Per-body resolution order:

0. Stages 1–2 below are gated by the visualizer's `cache_neurons` flag
   (UI default `True`, class default `False`): with `cache_neurons=False`
   the repair caches and shared raw cache are skipped and bodies resolve
   from the healed zip (and a fresh CAVE repair for flagged bodies).
1. Previously repaired trees, served network-free from the local repair
   caches before the zip is ever read (status-driven, via
   `extrusion_check_results.parquet`): `api_repaired` bodies from
   `cache/<ds>/skeletons/cave_skeletons/`; `local_fallback` bodies from
   `cave_skeletons/` first, then `cache/<ds>/skeletons/extrusion_fixes/`.
2. Shared raw-skeleton cache (`cache/<ds>/skeletons/raw_skeletons/*.swc.zst`;
   legacy frozen reads — the pipeline no longer writes new entries there).
3. Local healed skeleton zip `datasets/<ds>/sk_lod1_783_healed.zip`
   (~13.8 GB, the only skeleton source; served directly, read-only).
4. Extrusion check on the tree (one-time per body; parquet-cached in
   `extrusion_check_results.parquet` — only ids without a recorded row pay
   the detector cost). Flagged or missing bodies are replaced by the
   **CAVE replacement path**: a cached
   `cache/<ds>/skeletons/cave_skeletons/{bodyId}.swc.zst` tree is used first;
   on a miss, the raw CAVE mesh is wavefront-skeletonized (no pre-decimation)
   and the level-0 tree is persisted in that dedicated replacement store
   (status `api_repaired`).
   - a CAVE outage or per-body miss falls back to pruning the diagnosed
     extrusion branch locally (status `local_fallback`); the pruned fix is
     persisted to `cache/<ds>/skeletons/extrusion_fixes/{bodyId}.swc.zst`
     with the `# DROCAT source: local_extrusion_fix` header so later runs
     serve it directly. When no safe cut exists the extruded tree is kept
     and the status stays `api_failed` (retryable on the next run).
5. Token-gated CAVE API fetch for everything still missing.

The former columnar `.zst` bundle pipeline (`sk_lod1_783_healed.zst`,
lazy per-skeleton conversion, bulk `pack`) is **retired**: the application
never creates or appends a `.zst` container. An existing `.zst` is opened
read-only only when no zip is present; the container tooling
(`python -m src.fafb_bundle pack|verify|info|compact|append`) remains
available for manual maintenance only.

Skeletonization calibration (measured on l-LNv, 2026-09): wavefront on the
raw mesh reproduces the healed bundle's node density within ~11% at 1–8 s
per neuron; any mesh pre-decimation is slower end-to-end (decimation costs
more than it saves) and drops thin processes (density collapses to 10–30% of
reference), so the raw mesh is always the skeletonization input.

If CAVE access is unavailable, the user gets an actionable message naming the
missing local bundle/cache and the required CAVE configuration — the resolver
never fails prematurely while a CAVE token exists.

### 1.2 male-cns (`male-cns:v1.0`, NeuPrint client)

Decision flow: [malecns_skeleton_flow.html](../visualizations/malecns_skeleton_flow.html).

1. Shared raw-skeleton cache.
2. Batched parallel NeuPrint `fetch_skeletons` (token via the token-manager
   chain); skeletons arrive in **JRCFIB2022Mraw voxel** coordinates and are
   persisted back into the raw store in that frame.

### 1.3 BANC (`banc_v626` / `banc_v888`, public GCS bucket, no authentication)

Decision flow: [banc_skeleton_flow.html](../visualizations/banc_skeleton_flow.html).

> Note on morphology scope: BANC is deliberately excluded from
> INTRA-dataset Find Similar / Comparison pending vector-quality
> validation of the public L2/full skeletons, yet it DOES participate in
> CROSS-dataset morphology (one-time bootstrap of its population
> artifacts from these same public skeletons, with a standing
> reliability warning). Both stances are intentional; see the
> [cross-dataset morphology guide](../ui_guides/cross_dataset_morphology.html).

Both standalone releases are supported and fetch from the same public
products; every skeleton is served by its 888-namespace file name (v626
bodyIds resolve through the id crosswalk). One SWC per neuron under
`compiled_data/banc_888/banc_banc_space_swc/`, with a
**per-neuron fallback chain** (mutually exclusive products — probed: no neuron
ships both):

1. `{888_id}_l2.swc` — coarse L2 approximation, nanometres (cache-level product).
2. `{888_id}_skeleton.swc` — full resolution, nanometres (proofread subset).
3. v626 pcg-skel SWC (`neuron_skeletons/swcs-from-pcg-skel/`) — **micrometres**,
   scaled ×1000 to nm; v626 ids resolve to 888 stems via the meta-feather
   crosswalk.

## 2. Raw cache and storage rules

FAFB raw reads come from the healed zip directly; the shared raw store layout
`cache/{dataset}/skeletons/raw_skeletons/{bodyId}.swc.zst` (raw level,
simplification 0, native coordinates) is kept for legacy frozen reads and for
NeuPrint/BANC fetch caching. FAFB CAVE replacements use the separate
`skeletons/cave_skeletons/` store and locally pruned fixes the
`skeletons/extrusion_fixes/` store described above.

- Raw SWCs are never modified by render-time simplification.
- A MeshNeuron is never serialized as an SWC to satisfy a skeleton API.
- BANC entries carry a `# DROCAT source:` provenance header
  (`banc_gcs_l2` | `banc_gcs_full` | `banc_gcs_pcg_um_x1000`) that is re-read
  on every cache load to restore the processing class; malformed downloads are
  parsed before caching and never poison the store. The pcg-µm product is
  persisted **scaled to nm** (R1) so a reload cannot mix unit frames.
- FAFB CAVE skeletonization writes a **separate replacement store**,
  `cache/{dataset}/skeletons/cave_skeletons/{bodyId}.swc.zst`, with the
  provenance header `cave_mesh_wavefront`. These trees replace healed-zip
  skeletons that failed the extrusion check, so they must never overwrite
  the legacy mirror in `raw_skeletons`. Level-0 entries left in
  `raw_skeletons` by the pre-unification fetch path remain readable as a
  legacy fallback; simp90-legacy writes there are rejected.
- Locally pruned extrusion fixes write a second replacement store,
  `cache/{dataset}/skeletons/extrusion_fixes/{bodyId}.swc.zst`, with the
  provenance header `local_extrusion_fix`. A recorded `local_fallback`
  status serves from `cave_skeletons` first, then this store.

The former prepared FAFB mesh cache (`cache/{dataset}/meshes/…`, 95%-decimated
`MeshNeuron` pickles) is no longer read or written by the visualization
pipeline or by the on-demand fetchers (which serve skeletons for every
dataset kind); mesh-representation workflows that read it directly —
`FlyWireMeshCache` / `find_similar_flywire_mesh_cache` and
`_fetch_cave_mesh` / `CAVEDataFetcher.fetch_fafb_mesh` — still own it.

## 3. Rendering pipelines and simplification scale

### 3.1 FAFB tube pipelines

```text
fast:     raw SWC → node reduction to 25% retention → tube mesh → fine face decimation
fine:     raw SWC → tube mesh → soma-aware face decimation (no node stage)
artistic: raw SWC → tube mesh → vertex-clustering decimation (no node stage)
```

The 25% target is a node-count target: topology-aware reduction must keep
roots, branch points, and terminals, so achieved retention can exceed the
nominal target. NeuPrint-specific smoothing/resampling/radius transforms are
never applied to native FAFB SWCs.

### 3.2 male-cns tube pipeline

One aggregate preprocessing boundary per render: affine
`JRCFIB2022Mraw → JRCFIB2022M` (nm), tube mesh (6 pts), decimation to the
mesh-cache level. `fast` is the public default; `fine` and `artistic` remain
explicit selections with their own morphology behavior.

### 3.3 BANC tube pipeline

Radius repair (≤0/NaN → 1 nm), median-radius normalization to the shared
240 nm target, tube mesh, then **source-class-dependent decimation**:

- `full` — decimation to the slider target with the 4,000-face full-res floor.
- `l2` / `pcg-µm` — counted as **~90% pre-simplified**
  (`BANC_L2_EQUIVALENT_SIMPLIFICATION = 0.90`); face decimation applies only to
  the *excess* above that baseline, scaled onto the L2 density:
  asked 0.95 → 50% of L2 faces removed; asked 0.99 → 90%; asked ≤ 0.90 →
  untouched.

Measured densities behind the 0.90 baseline (population medians): L2 mean edge
≈ 3,369 nm, pcg-µm ≈ 5,634 nm, full ≈ 376 nm — L2 is 9.0× and pcg-µm 15×
coarser than full. Node-count simplification cannot reproduce L2 coarseness
(topology preservation plateaus at ~57% retention / ~570 nm edges), which is
why the L2 product is treated as pre-simplified rather than re-derived.

### 3.4 Line mode

Local TreeNeuron sources: raw SWC or a CAVE replacement tree → in-memory node
simplification → direct line plotting. Dataset defaults: FAFB 90% reduction;
male-cns 50%; BANC full-res 50% (L2/pcg untouched). FAFB CAVE replacements
arrive as pre-skeletonized trees from the `cave_skeletons` store, so line mode
never skeletonizes at the render boundary.

### 3.5 Simplification scale (all knobs)

| Constant / knob | Value | Applies to |
| --- | --- | --- |
| `FAFB_FAST_NODE_RETENTION` | 0.25 retain | FAFB fast node stage |
| `FAFB_LINE_NODE_REDUCTION` | 0.90 | FAFB line mode |
| `NEUPRINT_LINE_NODE_REDUCTION` | 0.50 | male-cns line mode |
| `BANC_LINE_FULL_NODE_REDUCTION` | 0.50 | BANC line mode, full-res only |
| `skeleton_mesh_simplification` | 0.90 fast / 0.95 fine-artistic | face decimation, all families |
| `NEUPRINT_MESH_CACHE_SIMPLIFICATION` | 0.95 | male-cns mesh-cache level |
| `FLYWIRE_MESH_CACHE_SIMPLIFICATION` | 0.95 | prepared-mesh-cache level (non-visualization consumers) |
| `BANC_FULL_MIN_KEEP_FACES` | 4,000 | BANC full-res decimation floor |
| `BANC_L2_EQUIVALENT_SIMPLIFICATION` | 0.90 | BANC L2/pcg pre-simplified baseline |

## 4. Scene space and brain-mesh selection

`brain_mesh='native'` renders in the dataset's own template space. Explicit
`BANC` / `FAFB` / `male-cns` selections **move the whole scene** — neurons,
synapses, ROIs, and the outline — into that template's space via the flybrains
bridging registry, capped at two hops (longer chains route through elastix
registrations too lossy for rendering; unreachable selections warn and keep the
native scene). The FLYWIRE frame carries the template's left-right tilt, so
whenever the FAFB mesh is in use (`_fafb_tilt_applies`) the −3°/−3° tilt
correction is applied to the outline, neurons, synapses, and ROI meshes
**together** — one global rotation, registration preserved.

**The tilt correction is data-level, not camera-level**: it permanently
rotates the rendered coordinates (mesh vertices, neuron nodes, synapse sites)
around the brain center; the FAFB camera itself is axis-aligned with no tilt
component. Everything rendered in the FLYWIRE frame receives the same rotation,
so alignment between elements is unaffected. Raw caches always stay in native
coordinates; scene moves happen at render time only.

## 5. Brain + VNC template meshes

Outlines come from the `flybrains` package and, for BANC, the release's public
`region_outlines` layer:

| Template | Source mesh | Coordinates | Preparation |
| --- | --- | --- | --- |
| FLYWIRE (FAFB) | `flybrains.FLYWIRE.mesh` | FLYWIRE nm | left-right tilt correction (−3° Z, −3° Y around the brain center) whenever the FAFB mesh is in use |
| JRCFIB2022M (male CNS) | `flybrains.JRCFIB2022M.mesh` (brain + VNC shells) | JRCFIB2022M nm | brain/VNC split at the neck: **Z = 340,000 nm**, boundary faces assigned by centroid |
| BANC | public `region_outlines` segid 1 (`BANC_outline`; render-decimated pair, full res kept as `BANC_outline_full`) | BANC nm | brain/VNC split at the neck: **Y = 350,000 nm** (measured waist 470k; brain lobes end ≈ 240k, VNC somas start ≈ 538k), boundary faces assigned by centroid |

Centroid assignment (rather than the earlier all-vertices filter) keeps the two
portions seamless at the cut plane — the all-vertices filter left a visible
crack in BANC renders and trimmed the male-CNS brain cap. The former
`brain_mesh='whole'` JRC2018F scene-transform mode (H5 transform downloads,
~13 GB) is retired; `FAFB` now renders the FLYWIRE outline as the scene frame.

## 6. ROI meshes

ROI resolution per scene dataset:

- **male-cns / hemibrain / optic-lobe** — native NeuPrint ROI meshes
  (JRCFIB2022Mraw / JRCFIB2018F frames), fetched from NeuPrint when not cached
  under `cache/<dataset>/meshes/`.
- **FAFB** — no native ROI product: named ROIs are fetched from **male-cns**
  (`JRCFIB2022Mraw`) and bridged `JRCFIB2022Mraw → JRCFIB2022M → FLYWIRE` at
  render time, then tilt-leveled; transformed meshes (tilt included) cache
  under `cache/<ds>/meshes_transformed/FLYWIRE/`. The ROI list comes from
  `cache/male-cns_v0_9/available_rois.json`. Decision flow:
  [fafb_roi_resolution.html](../visualizations/fafb_roi_resolution.html).
- **BANC** — no named ROI product either: the public `region_outlines` layer
  carries only four aggregates (`BANC_outline` segid 1, `BANC_neuropil` 2,
  `BANC_brain_neuropil` 3, `BANC_vnc_neuropil` 4, all in BANC nm). Named ROIs
  use the **same male-cns-transformed handling as FAFB**, bridged
  `JRCFIB2022Mraw → JRCFIB2022M → BANC`, cached under
  `cache/<ds>/meshes_transformed/BANC/`. The four native aggregates load in
  place (source `BANC`, no transform). The offered ROI list is the male-cns
  set (5,412 names) plus the four aggregates.

### 6.1 Exact resolution order for one `mesh_roi` entry on a FAFB scene

As coded in `VisualizeSkeleton.plot_mesh()` (decision flow:
[fafb_roi_resolution.html](../visualizations/fafb_roi_resolution.html)):

1. **Dataset cache lookup** — `cache/flywire_FAFB_v783/meshes/{X}.json`.
2. **Transformed-cache check** —
   `cache/flywire_FAFB_v783/meshes_transformed/FLYWIRE/{X}.json`: on hit, load
   and render as-is (tilt included from the caching run).
3. **male-cns fetch** — `neu.fetch_roi(X, client=male-cns:v0.9)`; the raw mesh
   (JRCFIB2022Mraw) is saved into the dataset cache dir, then bridged
   `JRCFIB2022Mraw → JRCFIB2022M → FLYWIRE`, tilt-leveled, and written to the
   transformed cache. Failures are logged. The generic hemibrain fallback is
   **excluded for FAFB** — a name missing from male-cns is skipped with a
   warning rather than mis-framed.

| Run | Path taken |
| --- | --- |
| First | dataset-cache miss → male-cns fetch → bridge + tilt → transformed cache write → render |
| Second+ | transformed-cache hit → render as-is |

### 6.2 Exact resolution order for one `mesh_roi` entry on a male-cns scene

Decision flow: [malecns_roi_resolution.html](../visualizations/malecns_roi_resolution.html).

1. **Dataset cache lookup** — `cache/male-cns_v1_0/meshes/{X}.json`.
2. **Legacy `primary_rois` check** — `navis_roi_meshes_json/primary_rois/{X}.json`
   (JRCFIB2018F frame; coded branch, directory absent locally).
3. **male-cns fetch** — `neu.fetch_roi(X, client=male-cns:v0.9)`; raw mesh
   saved into the dataset cache dir (JRCFIB2022Mraw).
4. **Affine** `JRCFIB2022Mraw → JRCFIB2022M` applied **on every render** —
   male-cns ROI meshes have no transformed cache (the transform is a cheap
   built-in affine).

| Run | Path taken |
| --- | --- |
| Every | dataset-cache miss → male-cns fetch → affine → render; cache hit → affine → render |

### 6.3 Exact resolution order for one `mesh_roi` entry on a BANC scene

As coded in `VisualizeSkeleton.plot_mesh()` (decision flow:
[banc_roi_resolution.html](../visualizations/banc_roi_resolution.html)).
Cache locations differ by mesh kind: **aggregate** outlines are always
stored under the fixed `cache/banc_v888/` folder regardless of the selected
release, while **transformed named-ROI** meshes and raw male-cns downloads
use the selected scene dataset's cache directory (`<ds>` below is
`banc_v626` or `banc_v888`):

1. **Dataset cache lookup** — `cache/banc_v888/meshes/{X}.json` for BANC
   aggregate outlines (case-safe name; `_get_banc_mesh_dir` fixes this to
   the `banc_v888` folder so cross-template selections find the products).
2. **BANC aggregate check** — if `{X}` is in the region-name map
   (`region_name_map.json`, from the release's `segment_properties/info`),
   resolve through `_banc_mesh_file_path` (clean plain names for the
   aggregates — the generic case-safe encoding cannot see names with lowercase
   letters, e.g. `BANC_neuropil`) and fetch from CloudVolume on miss. Loaded
   with `source = BANC` → **no transform**.
3. **Transformed-cache check** — `cache/<ds>/meshes_transformed/BANC/{X}.json`:
   on hit, load and render as-is (no transform).
4. **male-cns fetch** — `neu.fetch_roi(X, client=male-cns:v0.9)`; the raw mesh
   (JRCFIB2022Mraw) is saved into the selected dataset's cache dir, then
   bridged `JRCFIB2022Mraw → JRCFIB2022M → BANC` and written to the
   transformed cache. This step needs a NeuPrint token. Failures are logged.
   The generic hemibrain fallback that non-FAFB datasets
   use is **excluded for BANC** — a name missing from male-cns is skipped with
   a warning rather than mis-framed.

Cache levels across runs (named ROI):

| Run | Path taken |
| --- | --- |
| First | dataset-cache miss → male-cns fetch → bridge → transformed cache write → render |
| Second+ | transformed-cache hit → render as-is (the raw dataset-cache copy is shadowed) |

Native BANC aggregates: first run fetches from CloudVolume into the dataset
cache; later runs load from it; both render without transform.

ROI placement always follows the scene's skeleton transform target — outline
selections never retarget ROIs.

## 7. UI and caller defaults

- Public tube pipeline default: `fast`; `fine` and `artistic` are explicit choices.
- `skeleton_mode='line'` by default for Similarity-tab sub-visualizations and
  NeuronBridge Find Neurons sub-visualizations; the dedicated Skeleton tab stays
  independently configurable (tube default), with an explicit caller override.
- Line mode never triggers tube conversion or surface decimation.
- `use_cache=True` reads applicable source caches first and writes results;
  `use_cache=False` fetches online and reads/writes no caches.

## 8. Further reading

- [3D_Skeleton_Guide.md](../visualizations/3D_Skeleton_Guide.md) — user-facing guide
- [BANC_INTEGRATION.md](../BANC_INTEGRATION.md) — BANC products, crosswalk, radius normalization
- [AVAILABLE_ROIS.md](../AVAILABLE_ROIS.md) — ROI name lists per dataset
- [CacheSystem_Guide_v4.md](../core-features/CacheSystem_Guide_v4.md) — cache system overview
