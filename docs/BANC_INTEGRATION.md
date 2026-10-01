# BANC Data Integration

This project supports the BANC (Brain And Nerve Cord) connectome using its
**public release bucket** — no FlyWire/Codex login, no CAVE token, and no
manual downloads are required. Use the exact dataset identifier that matches
the release: `banc_v626` or `banc_v888`. (Legacy `flywire_BANC_v626` /
`flywire_BANC_v888` spellings are accepted everywhere and resolve to the same
folders and caches.)

## What is supported

| Workflow | Status |
|---|---|
| Connectivity / pathfinding / network analysis | ✅ (local parquet tables) |
| **3D skeleton visualization** (native BANC space) | ✅ (public-bucket SWCs) |
| BANC ROI / neuropil meshes + CNS outline template | ✅ (public `region_outlines` layer) |
| Morphology find-similar | ❌ deferred (vector-quality validation pending) |
| Morphology comparison (vector_v2 / NBLAST) | ✅ runs, with a provisional-scores warning |
| CAVE API queries (`brain_and_nerve_cord`) | ❌ token-gated, not used |

## Data Preparation (automatic)

Selecting `banc_v888` or `banc_v626` in any tool prepares the dataset
automatically from the public bucket (`~134 MB` of products, `~186 MB` total on disk after conversion, one time):

- **Neuron metadata** from `compiled_data/banc_888/banc_888_meta.feather`
  (ids, curated `cell_type`, `Alternative Cell Type(s)`,
  `fafb_cell_type`, `fafb_alignment_cell_type`, `malecns_cell_type`,
  `hemibrain_cell_type`,
  `manc_cell_type`, match bodyIds, classes, neurotransmitters, proofread
  flags). Row counts are release-specific: the 2026-09-04 bucket snapshot
  yields **188,508** rows for `banc_v888` (ids as served) and **185,165**
  rows for `banc_v626` (ids resolved through `root_626`, then
  deduplicated — the coarser v626 materialization carries duplicate ids).
- **Connections** from `neuron_connectivity/<version>/synapses_v1_..._
  connectioncountsperneuropil_countthresh3.parquet` (per-neuropil pair
  counts; synapse size >= 3 and connection count >= 3 thresholds are baked
  into the product — matching what the Codex download provided).

The raw products stay in `datasets/<dataset>/downloads/` so re-runs are
offline — with one exception: the connections product is only fetched
while the merged connections table is missing, so it can be deleted once
that table exists (see *Local files and disk usage* below). The sidecar
is written as `<dataset>_metadata.json` (e.g.
`datasets/banc_v626/banc_v626_metadata.json`,
`datasets/banc_v888/banc_v888_metadata.json`) and records
`source: banc_public_gcs`.
Preparation also fills the neuron table's `post` column (post-synaptic
counts) from the merged connections, matching what the manual Codex path
produced.

## Cross-dataset type bridges

The auto type mapper treats `banc_v626` and `banc_v888` as separate release
namespaces. Their curated labels are direct evidence for the matching target
dataset:

| BANC column | bridge target |
|---|---|
| `fafb_cell_type` | FAFB v783 |
| `fafb_alignment_cell_type` | FAFB v783 — fallback-only lane (2026-09-27) |
| `malecns_cell_type` | male-cns v1.0 |
| `hemibrain_cell_type` | hemibrain v1.2.1 |
| `manc_cell_type` | MANC v1.0/v1.2.1 |

`fafb_alignment_cell_type` is the weaker second FAFB lane (96.5% vs 99.2%
`fafb_match` agreement whole-release; it anchors on the BANC type's own
name in the disagreement cases, e.g. the ORN glomerular swaps). It fills a
mapping only where `fafb_cell_type` yields no winner — 18 types per
release on the 2026-09-27 tables, all cross-name (e.g. `AVLP614` →
`CB1476`) — with the evidence tier `direct alignment label` and the
provenance kind `cross-dataset cell type (alignment)`; curated winners and
curated conflicts are never touched, and its own vote conflicts are
diagnostics, not conflict records.

`auto:` is a provenance prefix, not a separate namespace. A normalized
`auto:<name>` label is eligible when `<name>` resolves to a known target type;
the raw value and auto provenance are retained. Unknown `auto:` labels remain
ineligible, and conflicting label votes remain rejected. FAFB/MCNS match
bodyIds are checked against the target table when available only as optional
provenance diagnostics; contradictory rows are recorded as conflicts, not
removed from the curated type-label vote. The mapper records the winning
label, vote counts, verified votes, verification conflicts, raw/canonical
values, and alternates in bridge provenance. BANC labels are direct-only and
never turn BANC into a connector between unrelated datasets. A label hop is
a derivation endpoint once it lands in the target namespace with the cell
token naming the target type exactly; a rename-resolved token (e.g. cell
`SMP537`, FAFB primary `DN1pD`) terminates at the raw-token name node and
the chain continues through the target namespace's own
`additional_type(s)` edge (2026-09-27 token chaining), and when a curated
chain bridges the same token to the same target as a one-linker
annotation chain, that annotation chain is dropped (curated subsumption) —
see AUTO_TYPE_MAPPING.md for the full five-form FAFB↔BANC semantics.
Because `Alternative Cell Type(s)`
concatenates the other datasets' curated labels, a primary's annotation cell
naming another BANC primary is a cross-reference, not a rename — the
derivation walk refuses to hop through it (this closed the `l-LNv → BM_*`
fan-out where one oddly-labeled neuron reached 1,212 unrelated primaries).

The two BANC releases have one explicit type bridge from
`datasets/banc_v888/downloads/banc_888_meta.feather` (or the compiled local
copy): the full `root_626` ↔ `root_888` relation. Duplicate `root_626` rows
are retained, `root_888` is the type relation key, and equal numeric IDs are
not treated as proof. This bridge is labeled `banc_release_crosswalk`; it
provides independent release coverage participants but never exposes a
bodyId-to-bodyId pairing or licenses generic BANC↔BANC annotation paths.
Both releases sit in the mapper's `DATASET_PRIORITY` walk (v888 right after
v626), so a v888-only type name auto-detects its own namespace.

### Manual downloads are no longer needed

The bucket path fully replaces the old manual Codex downloads
(`neurons.csv.gz` + `connections_princeton.csv.gz`).  The converter keeps a
legacy compatibility path for those files, but no workflow requires them.

### Migrating a pre-rename checkout

Datasets prepared before the `flywire_BANC_*` -> `banc_*` rename are
migrated with:

```bash
python scripts/maintenance/migrate_banc_dataset_names.py [--dry-run]
```

Old spellings keep working even without the migration (every dataset-name
entry point canonicalizes via `utils.naming_utils.canonical_dataset_name`).

## 3D Skeleton Visualization

BANC renders in **native BANC space** (nanometres). Skeletons are the
official release SWCs, fetched on demand from

```
compiled_data/banc_888/banc_banc_space_swc/<root_888_id>_<resolution>.swc
```

- One file per neuron: `_skeleton.swc` (full resolution, proofread neurons)
  or `_l2.swc` (coarse L2 approximation for everything else). A 404 on the
  preferred resolution transparently falls back to the other.
- `banc_v888` body ids resolve directly; `banc_v626` ids resolve through the
  meta-feather crosswalk (`root_626` -> `banc_888_id`). In the current
  snapshot every crosswalk row carries an 888 id (100% id resolution,
  measured over `cache/banc_v626/banc_id_crosswalk.parquet`). Skeleton
  availability is a separate metric: neurons with no skeleton in any chain
  tier below (a small residual of tiny fragments) are skipped with a
  warning.
- Fetched skeletons cache into the shared raw skeleton store
  (`cache/<dataset>/skeletons/raw_skeletons/<id>.swc.zst`, raw level 0 with
  a provenance header that also records which source served the neuron),
  so subsequent renders are instant and offline.
- Every preset view (interactive dropdown, PNG view export, individual
  plots, video) shares one per-dataset camera table, and saved HTML opens on
  the calibrated BANC frontal view (anterior at -Y) on every export path.

### Skeleton source chain (unified)

There is no per-run source selection: every fetch walks one chain and uses
the first source that exists for the neuron —

1. **888 L2** — `compiled_data/banc_888/banc_banc_space_swc/{id}_l2.swc`
   (nanometres; the cache-level product for ~most neurons),
2. **888 full-resolution** — `{id}_skeleton.swc` (proofread neurons; serves
   the few neurons that have no L2 at all, e.g. one aMe12),
3. **v626 pcg-skel** — `neuron_skeletons/swcs-from-pcg-skel/{root_626}.swc`
   (micrometres, scaled ×1000 on fetch; v888 ids are translated through the
   id crosswalk). This covers the neurons absent from the 888 export in
   both resolutions — the loader documents this tier at roughly **1.7% of
   the release** (e.g. one aMe12 and one l-LNv body); only fragments with
   no skeleton in any tier are skipped.

The old **BANC Skeleton Resolution** select (`l2` / `full` / `full_auto`)
was removed from the Skeleton tab — the chain is always L2 → full → pcg.
The deprecated `banc_skeleton_resolution` constructor argument is still
accepted (validated, ignored) for script compatibility.

Rendered tube radii are normalized per neuron (median → a shared 240 nm
target by default; see below). L2 tubes are the cache-level product and
count as ~90% simplified against the full-resolution skeletons: they skip
face decimation unless the asked `skeleton_mesh_simplification` exceeds
that baseline — the excess is scaled onto the L2 density (asked 0.95
removes 50% of L2 tube faces; asked 0.99 removes 90%).
The Skeleton tab's mesh-simplification default stays the FAFB-style 0.90:
it drives the full-resolution sources, and a 4,000-face floor bounds
full-resolution decimation at extreme slider values.

**id namespaces**: `root_626` and `root_888` coincide for ~61 % of the
release (115,737 of 188,508 rows) and differ for the rest; the meta-feather
crosswalk covers 100 % of neurons in both directions, so a v888 body id
always reaches its v626 pcg-skel twin and vice versa.

**Tube radius normalization**: the release products carry inconsistent
radius calibers per neuron (median radii range ~50–172 nm), which shows up
as mismatched tube thicknesses in one scene. The Skeleton tab exposes a
**Normalized Tube Radius** checkbox (default on) with a **Radius Target**
(240 nm median by default): each neuron's radii are rescaled so its median
hits the target, preserving relative branch thickness. Uncheck to use the
raw radii.

## Synapse markers

BANC defaults to skipped synapses. Opting in (any synapse mode other than
`skip`) triggers a one-time ~3.9 GB download of the release per-synapse
table into `datasets/<dataset>/downloads/` (resumable; subsequent renders
are local), then draws **pre-synaptic site markers** at the T-bar
positions — the release publishes only pre-site coordinates, so the
paired cone/sphere markers used for FAFB/NeuPrint are not available. The
UI shows an explicit warning when synapses are enabled for a BANC
dataset. The `pre-post sites` mode is not supported for BANC.

The downloaded raw table is aggregated once into the derived per-pair
table `datasets/<dataset>/<dataset>_synapse_table.parquet` (~1.4 GB,
columns `pre_root_id, post_root_id, syn_count, x_pre, y_pre, z_pre` in
nanometres) and the raw download is then deleted. Because the release has
no post-site coordinates, the reader mirrors the pre-site position onto
`x/y/z_post` in memory after loading; the mirrored duplicates are **not**
persisted. The derived table is stored with a measured **lossless**
layout (`DELTA_BINARY_PACKED` on `pre_root_id`, `BYTE_STREAM_SPLIT` on
`post_root_id`/`syn_count`/`x_pre`/`y_pre`, zstd-7, no dictionary) and
stamped with a `DROCAT.lossless` footer marker — values read back are
bit-identical, only the storage representation changes. Tables produced
by older DROCAT versions still carry the three duplicated post columns
(~40% of the file) and the plain layout; they are re-encoded in place
automatically the next time the table is used (one streaming pass,
~30 s on the v888 release).

## Local files and disk usage

| path | required? | notes |
|---|---|---|
| `datasets/<ds>/downloads/banc_888_meta.feather` | **keep** | read at runtime (neuron index, cross-dataset type mapper) and needed to rebuild the neuron tables |
| `datasets/<ds>/<ds>_allneurons_neuron_df.csv` | **keep** | read directly by ROI screening and the neuron index builder (the `.parquet` twin is preferred elsewhere) |
| `datasets/<ds>/<ds>_allneurons_neuron_df.parquet` | keep | canonical neuron table |
| `datasets/<ds>/<ds>_merged_connections.parquet` | keep | source for the connection cache; used by profiling and comparison |
| `datasets/<ds>/<ds>_synapse_table.parquet` | keep (lossless-encoded) | derived per-pair synapse table; legacy tables self-upgrade on first reuse |
| `datasets/<ds>/downloads/connections_<version>.parquet` | droppable | only consumed to build the merged table; re-downloaded automatically if the merged table is ever missing. **Deletable once `<ds>_merged_connections.parquet` exists** (76 MB) |
| `cache/<ds>/connections.parquet` + `.src` | droppable | derived from the merged table; rebuilt automatically whenever missing or stale (~109 MB) |
| `cache/<ds>/neuron_index_state.parquet` | droppable | rebuilt together with the connection cache (~15 MB) |
| `cache/<ds>/incoming_connections.parquet` + `incoming_complete.json` | droppable | pathfinding caches, recomputed on demand (~52 MB) |
| `cache/<ds>/skeletons/`, `meshes/`, `meshes_transformed/` | droppable | render caches; re-fetched at the cost of network time |

## Pathfinding & connectivity

Connectivity comes from the bucket-prepared, **per-version** local tables —
`banc_v626` and `banc_v888` are distinct datasets with distinct id spaces
(~61 % of ids coincide across releases and are the same neurons; the rest
are remapped; the crosswalk covers 100 % of both directions). FindPath /
FindAllPath read the per-neuron connection cache, which **rebuilds
automatically** from `<dataset>_merged_connections.parquet` whenever the
table is newer — no manual cache step and no CAVE token. Notes:

- BANC is not on the NeuPrint server: pathfinding never contacts it, and
  `use_cache=False` (online-only mode) is unsupported for BANC and raises a
  clear error.
- ROI-based path filters are unavailable (the release ships no primary ROI
  list); `roi_coverage` in the metadata sidecar is marked
  `not_available_for_banc`.
- `<dataset>_metadata.json` statistics are regenerated from the tables on
  every preparation (neuron counts, real type coverage excluding
  `'Unknown'`, synapse totals).

## ROI meshes, brain and VNC outlines

BANC `mesh_roi` options come in two groups with different availability:

**Native aggregate outlines — always offered.** The public
`region_outlines` CloudVolume layer provides four aggregates
(`BANC_outline`, `BANC_neuropil`, `BANC_brain_neuropil`, `BANC_vnc_neuropil`),
all in nanometres — the same frame as the skeletons, so they render without
any transform.

**Named ROIs — conditional.** The release publishes **no named ROI meshes**,
so named ROIs (e.g. `AL(R)`) use the same handling as FAFB: fetched from
**male-cns** (`JRCFIB2022Mraw`) and bridged into BANC space at render time,
cached under `cache/<ds>/meshes_transformed/BANC/`. They appear in the
offered ROI list only when a male-cns `available_rois.json` cache exists,
and a cache miss fetches from the male-cns source, which requires a
NeuPrint token — without one the four aggregates keep working but named-ROI
requests fail.

`brain_mesh='native'` (or the explicit `brain_mesh='BANC'`) draws the
**brain portion** of the whole-CNS outline and `vnc_mesh=True` the **VNC
portion**: the two are segmented by the neck coordinate
(`y = 470 µm` waist; cut at `y = 350 µm` so the neck stays with the VNC),
mirroring the male-cns geometric extraction. Boundary faces are assigned by
centroid so brain and VNC meet without a crack.

## Usage

```python
from coana import FindNeuronConnection

# bodyIds verified against the prepared banc_v888 table (aMe12 -> PPL101,
# the same pair exercised by tests/e2e/test_banc_ame12_ppl_e2e.py)
fnc = FindNeuronConnection()
fnc.dataset = 'banc_v888'
fnc.sourceNeurons = ['720575941596944935']  # aMe12
fnc.targetNeurons = ['720575941416009108']  # PPL101
fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='banc_v888',
    neuron_layers=[['l-LNv']],
    brain_mesh='native',
    mesh_roi=['BANC_neuropil'],
    skip_synapse=True,
)
vs.plot_neurons()
```

## Notes

* **Root IDs** are very large integers; DROCAT keeps them as exact strings.
* **`type` coverage**: roughly **63%** of neurons carry a curated
  `cell_type` (counting non-empty values other than `'Unknown'`); the rest
  are `'Unknown'` (same convention as the Codex tables). Measured on the
  2026-09-04 bucket snapshot: 116,906/185,165 (63.1%) for `banc_v626` and
  118,748/188,508 (63.0%) for `banc_v888` — counts are release- and
  snapshot-dependent, and the same predicate is what the converter's
  regenerated metadata reports. The
  `Alternative Cell Type(s)` column aggregates the curated name plus the
  per-dataset cross-fly type matches and remains fully wired into the
  cross-dataset auto type mapper (which reads the CSV form of the table).
* **Synapse coordinates**: BANC per-synapse tables are nanometres; the local
  merged-connections table aggregates weights per neuropil, so exact
  per-synapse/post-site rendering is not supported. What BANC does support
  is an **opt-in per-pair pre-site marker mode**: a derived synapse table
  (`<dataset>_synapse_table.parquet`) drives approximate pre-site markers
  per connected pair when the option is enabled in the 3D skeleton view —
  useful as a connectivity diagnostic, but it is not a per-synapse display.
* The v626 metadata also carries a `root_626` -> `banc_888_id` skeleton
  resolution crosswalk (cached at `cache/<dataset>/banc_id_crosswalk.parquet`);
  skeletons are always fetched by their 888-namespace file name. This
  skeleton helper is separate from the type mapper's full `root_626` ↔
  `root_888` relation and must not be used as type-mapping evidence.
