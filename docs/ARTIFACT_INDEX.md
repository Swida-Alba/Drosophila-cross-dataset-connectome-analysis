# Artifact citation index — exported file → topic → producing code → skill

Lookup table for **citing a produced file**. Paste any path or filename pattern (an exact
run folder is best) and this map resolves it to: the question that file answers, the
`file:line` that writes it, the `file:line` that reads or verifies it, and the skill that
covers the operation.

Pairs with, and does not replace:

*   `docs/OUTPUT_FILES.md` — per-tool prose about the same files. **Its own stamp reads
    "Verified 2026-08-15; pathfinding updated 2026-09-07"** (`docs/OUTPUT_FILES.md:5`),
    so it is older than this index; trust a `file:line` here over its prose if they differ.
*   The **run guide inside every run folder** (`_UserGuide_please_read_me.html`) — the
    authoritative per-file, per-column description *for that run*, generated from the same
    registry (§8). If you have a run folder, open its guide before consulting this table.
*   `_clock_production/AGENTS.md` — the rules for the article repo's own artifacts (§7).

Paths are relative to the DROCAT repo root unless prefixed `_clock_production/`.
Line numbers were verified against the tree as of 2026-09-28; re-run the check in §10
after any refactor that moves files.

---

## 1. Naming grammar (what a filename can tell you)

| Token | Meaning | Defined at |
| --- | --- | --- |
| `{ts}` | `YYYYMMDD_HHMMSS` of the run | composed inline per tool (34 sites); recognized by `src/utils/naming_utils.py:267` |
| `{ABBREV}` | dataset code in folder names: `MCNS` `FAFB` `HEMI` `MANC` `OLOB` `BANC` (versioned `B_v626`/`B_v888` only when both releases run) | `src/utils/naming_utils.py:27` `DATASET_ABBREVIATIONS`, `:157 dataset_abbrev()`, `:65 canonical_dataset_name()` |
| run-folder prefixes | the grammar that decides whether a directory *is* a run folder | `src/utils/naming_utils.py:221 RUN_FOLDER_PREFIXES`, `:263 _RE`, `:270 is_run_folder_name()`, `:285 run_folder_timestamp()` |

**There is no shared run-folder creator** — each tool composes its own name and `mkdir`s
it, so a new export will not appear here automatically. Two consequences worth knowing:

*   A folder name is **not** always the parameter record: `{query}` is sanitized and
    truncated to 60 chars (`src/comparison/profile_comparator.py:7088`), and a
    caller-supplied `saveas` suppresses the parameter tokens entirely
    (`src/coana.py:12155`, `:13832`, `:15912`). `parameters.json` in the folder is the
    authority, not the name.
*   `similar-connectivity` is a retired prefix that is still registered
    (`docs/OUTPUT_FILES.md:387-392` vs `src/utils/naming_utils.py:232`).

---

## 2. Cross-dataset type mapper (type-level pairs)

Reached headlessly by `_compute_type_mapping` — `ui/components/type_mapping_panel.py`
(module-level, no UI state). In this section **`panel` means
`ui/components/type_mapping_panel.py`**, and a bare `:NNN` belongs to the module named
immediately before it in the same cell (the same convention holds in §4, where every
unqualified line number is `src/visualize_skeleton.py`).
**These are browser downloads: they land in the user's
Downloads folder, never on disk in a run folder, and nothing in-repo verifies them.**

| Pattern | Answers | Producer (writer → naming) | Reader |
| --- | --- | --- | --- |
| `mapping_<src>_<tgt>_<ts>.csv` | the per-pair flow table: adopted target, bridge, bodyId pools, `mapping_status`, `relationship`, `mapping_origin` | `src/comparison/mapping_visualization.py:3136 build_bridges_csv` → `ui/components/type_mapping_panel.py:932 _deliver_pair_csv` | `_clock_production/analysis/02,03,08` |
| `mapping_branch_bodyids_<src>_<tgt>_<ts>.csv` | per-branch pools incl. `target_out_map` bodyIds | panel `:938 _deliver_branch_bodyids` (keys via `mapping_visualization.py:59 mapping_pool_key`, `:86 get_mapping_pool`) | panel `:430` comment only |
| `mapping_all_pairs_<ts>.csv` | all pairs, unfiltered | panel `:1829` (builder `:1810`) | — |
| `mapping_{sankey,network}_<variant>_<foreign>_<ts>.html` | network / bridge-linker / Sankey view of the flows | `mapping_visualization.py:898`, `:1558`, `:2402` → panel `:907 _deliver_flows` | — |
| `mapping_sankey_<ts>.html` / `mapping_graph_<ts>.html` | composed multi-pair view | `mapping_visualization.py:2545`, `:3071` → panel `:897`, `:1574` | — |
| `outputs/type_mapping/source_map_network.html` | which bridge sources are licensed | `scripts/render_source_map_network.py:22` ← `mapping_visualization.py:2669` | git-ignored (`outputs/`) |
| `auto_type_mapping.csv`, `_suspects.csv`, `_conflicts.csv`, `auto_type_mapping.json`, `type_resolution_topology.json` | the mapping a profiling run carried with it | `src/comparison/cross_dataset_type_mapper.py:5044`, `:5236`, `:5444`, `:5713`; run-folder names at `src/comparison/comparison_analyzer.py:5985, 6028, 6045, 6084` | `src/comparison/report_tabbed.py:287` |
| `neuron_indexes/<dataset>/type_mapper_snapshot_v2.pkl` | the mapper's cold-build cache — **not evidence, never cite it** | `src/comparison/cross_dataset_type_mapper.py:1923` (path `:1876`), read `:1955` | — |

The claim/disclosure boundary, which every count on these files depends on:
`src/comparison/mapping_visualization.py:73 flow_is_claimed()` — a flow is a **claim** only
if its adopted target is in the decision's `mapping_target_types`; anything else is
disclosure. `docs/AUTO_TYPE_MAPPING.md` and
`docs/technical/AUTO_TYPE_MAPPING_IMPLEMENTATION.md` carry the vocabulary.

## 3. TM VEV — bodyId-level validation of a mapping (`type-map-validation_*`)

Run folder `type-map-validation_<SRC>_to_<TGT>_<ts>/`, assembled at
`src/comparison/mapping_validation.py:6194-6196`; layout registry `:1053
RUN_FILE_LAYOUT`, schemas `:1122 _RUN_CSV_SCHEMAS`, every CSV through
`:1364 _write_run_csv` (call sites `:6754-6832`). Produced by
`scripts/RunMappingValidation.py`, or queued by
`scripts/maintenance/run_tmvev_matrix.sh:42-45, 74` (queue dir holds `manifest.tsv`,
`queue.log`, `<label>.log`).

| Subfolder | Files | Answers |
| --- | --- | --- |
| `validation/` | `validation_results.csv` `forward_matches.csv` `pair_summary.csv` `pool_categories.csv` `examinees.csv` `deep_candidates.csv` `noise_filtered_candidates.csv` | per-pair verdicts, the metric columns, the Rev-3.12 category partition |
| `expansion/` | `family_candidates.csv` `relatives.csv` `out_map_expansion.csv` `source_candidates.csv` `source_status.csv` `backward_matches.csv` `target_matches.csv` | fan-out/invasion evidence; `target_matches.csv` is stage 5e (every appeared target scanned back over the source universe) |
| `gap_fill/` | `gap_fill_dedup.csv` `gap_fill_levels.csv` `gap_fill_proposals.csv` | which holes got filled at which bar (`level` high/medium/low/type_gated/advice) |
| `pooling/` | `pooling_candidates.csv` `pooling_pool.csv` `pooling_sources.csv` `pooling_cross_validation.json` | the unsupervised parallel mode's ledgers |
| `mapping/` | `mapping_export.csv` (+`tier`) `same_name_excluded.csv` `disclosure_evidence.csv` `suspects_verification.csv` | what was claimed vs declined, and on what evidence |
| root | `report.html` (`src/comparison/mapping_validation_report.py:4712`); `parameters.json`, `set_coverage.json`, `morphology_calibration.json`, `pipeline_progress.jsonl`, `README.txt` are all written by `src/comparison/mapping_validation.py` at `:6833`, `:6916`, `:6911`, `:2162`, `:7054` respectively; plus `user_warning_notes.txt` | the readable summary, the provenance, and per-stage telemetry |
| `visualization/` | `plot-3d_<ABBREV>_branches_<TYPE>_<ts>/branches_<TYPE>.html` (`src/comparison/mapping_validation_visualize.py:605, 1203`; glob documented `src/visualize_skeleton.py:2888`) | 21 scenes, **0.4-0.8 GB per run** — the one part never copied to an article repo |

**Readers/verifiers:** `scripts/verify_tmvev_run_exports.py:54` imports the layout and
schemas (`:92 path_of`, `:140` discovery, `:161 check_layout`, `:170` schema equality,
`:182-190` scene completeness, `:195-203` root deliverables, `:205` old-name ban);
`scripts/verify_tmvev_run_parity.py` for A/B runs; report reader
`src/comparison/mapping_validation_report.py:779-848`; `report.html` is regenerable with
`python -m comparison.mapping_validation_report <run_dir>` (`:4732`).

**Provenance fields to quote** (not the folder name): `parameters.json →
input_fingerprint.{git_rev, git_dirty, git_rev_at_start, git_dirty_at_start}`
(`src/comparison/mapping_validation.py:6071-6095`, `target_vector_store:1005`), and `run_label`
(field `:377`, written `:6840`) — the only place a `--label` survives. `set_coverage.json →
source.source_status`, `target.backward_evidence`, `mapper_gap.{types, untyped_rows}`
(`src/comparison/mapping_validation.py:4739-4762`). `morphology_calibration.json →
track_a_null_bar` / `_lo` / `track_a_null_n` `:4266-4268`, `branch_bars:4493`,
`pool_ref_tiers:4451` — the qualification floors, which is what moves a bin count
between two runs of the same query.

**Two-generation layout:** `run_file_path:1095-1112` falls back to the flat pre-2026-09-19
paths, and `examinees.csv` replaced `suspicious_candidates.csv` (older folders still read
via `src/comparison/mapping_validation_report.py:630-632`). Skill:
`skills/type-mapping-validation/SKILL.md`.

## 4. Visualization, profiles, decks

All of it is `src/visualize_skeleton.py` (there is no `src/visualization/` package; a
`src/volume_rendering.py` module was never shipped).

| Pattern | Answers | Producer | Reader |
| --- | --- | --- | --- |
| `plot-network_<stem>_<ts>/` | the Net-Viz tab's three views of one edge/path list, written at the run-folder **root** as `plot-network_<stem>_<ts>_{network,Sankey,heatmap}.html` plus `_data.xlsx` (`src/coana.py:14799-14801` names the views, `:14806-14811` the `_data` files; the xlsx itself comes from `vispath-subproject/src/vispath_pkg/vispath.py:13928`). The tab also **reorganizes** runs: `src/coana.py:14790-14812` moves the three views into `visualization/` and renames them `{Network,Sankey,Heatmap}_<run>.html`, so the same export exists in two shapes depending on whether it went through the organizer | the bridge/path graph as an interactive network, Sankey (uniform `weight=1` links carry no flow quantity), connection matrix, parsed input | folder created by `ui/tabs/visualization.py:1668 make_plotpath_folder` (`:1682`); its recent-run list is remembered in `ui/local_config.json:14-17`; the guide is `_UserGuide_please_read_me.html` (§8) |
| `plot-3d_<ABBREV>_<stem>_<ts>/` | one skeleton scene run | `src/visualize_skeleton.py:9168-9170` (prefix `:2885`) | `scripts/verify_tmvev_run_exports.py:190`, `src/morphology_comparison.py:1242` |
| `…/*.html` viewer page | the scene itself (`class="plotly-graph-div"`) | `:19511`, writer `:6134 _write_plotly_html` | `:732 resolve_viewer_page` |
| `visualization_manifest.json` | what layers the page actually holds | `:6077-6083` | `:707 read_visualization_manifest`, `tests/core/test_visualize_skeleton_manifest.py` |
| `parameters.txt`, `viz_layer_info.csv` | applied settings / layer table | `:9176`, `:9495` | — |
| `exported_views/{saveas}[_{view}].png` | per-view stills of a scene | `:19709`, `:19733` (folder `:19572`) | — |
| `{view}_{safe_name}.png` + `{safe_name}.html` | **per-neuron profile images** — the exporter an article pipeline wants | `:837 export_individuals_from_html` (also `:2113-2114` webdriver route); live-tab route `:20414`/`:20339` | `tests/core/test_visualize_skeleton_neuron_export.py` |
| `{stem}_profiles/`, `{stem}_reexport/` | default out-dirs | `:910-911`, `:783` | — |
| `individual_profiles_summary{_by_view,_by_name}.pptx` | a deck of profiles, widescreen | `:20891` in `:20792 _create_individual_pptx` | — |
| `aggregated_images.pptx` | generic image-stack deck | `src/utils/report_utils.py:258-260 img2pptx` (twin `src/visualize_skeleton.py:23047`) | `src/neuronbridge_finder.py:9625` |
| `<html_stem>_webdriver.png` | intended HTML→PNG route | `:22653 export_png_webdriver` (name assembled `:22727`) | **broken, §9** |

Framing defaults that decide whether an image is usable: `skeleton_mode "tube"`
(`ui/config.py:523`; analysis default `"line"` `:524`), `brain_mesh "native"`
(`ui/config.py:539`), `brain_mesh_color 'auto'` (`src/visualize_skeleton.py:3687` — **no
`brain_mesh_alpha` field exists**; alpha is embedded in the rgba), `export_views True`
(`ui/config.py:530`), `profile_granularity "legend"` (`:534`, values at
`src/visualize_skeleton.py:292`), `freeze_view True` (`:537`), `export_scale 3`
(`:2900`), `auto_crop True` / `crop_margin 30` (`:1974-1975`), `width/height 900`
(`:1970-1971`), `views='front'` (`:19968`). `neuron_alpha` is **not** in `ui/config.py`:
backend `0.2` (`src/visualize_skeleton.py:3133`) vs tab setting `0.3`
(`ui/components/skeleton_visualization_settings.py:198`, widget `ui/tabs/visualization.py:663`)
— name the value you used.

## 5. Connectivity, profiling, homologs, paths, networks, NeuronBridge, morphology

| Run folder | Citeable files | Answers | Reader |
| --- | --- | --- | --- |
| `profiling_<DS>_<query>_<ts>/` | `parameters.json:12569`; `{type_level\|group_level}/results/{type_similarity\|group_similarity}_{metric}_{dir}.csv:12676`; `bodyid_level/results/bodyid_similarity_*:12685`; `type_avg_bodyid_similarity_*:12694` | connectivity profile similarity at each level | `tests/core/test_profile_comparator_coverage.py:2066` |
| `profiling_<A>_vs_<B>_<ts>/` (cross-dataset) | `cross_dataset/{direction}_{metric}.csv:14163`; multi-dataset `similarity_{dir}_{metric}.csv:13474` | the same, across two releases | — |
| `homologs_<SRC>_to_<TGT>_<query>_<ts>/` | `results/homolog_results.csv:8000`, `bodyid_results.csv:7985`, `type_summary.csv:8335`, `type_level_results.csv:8298`, `intra_type_results.csv:8017`, `shuffle_test.json:8057`, `auto_type_mapping.json:7825` | candidate homologs + the null test | `tests/core/test_profile_comparator_coverage.py:987` |
| `direct_comparison_<neurons>_<ts>.csv` (+`_params.json`) | `src/comparison/profile_comparator.py:3500`, `:3527` | one named pair, no search | — |
| `finddirect_<DS>_…_L{L}w{w}r{r}p{p}_<ts>/` | `data_details/{s}_to_{t}_info_snp{n}_*:12351-12379`, `connectionMatrix_type.csv`, `transmissionMat_type.csv`, bodyId matrices `:12453` | direct connection search | `src/coana.py:12160` |
| `find-paths-{complete,shortest}_<DS>_…_<ts>/` | `{s}_to_{t}_allpaths_type.csv:18143`, `_bodyId_paths.csv:18572`, `all_attributes.json:15931`, `parameters.txt:15936`, `density_meta.json:15268` | pathfinding | `src/comparison/comparison_analyzer.py:9228`, `tests/e2e/run_real_data_e2e.py:315` |
| `find-network_<DS>_…_<ts>/` | `data_details/{connection_type.csv:12825, neurons.csv:12744, parameters.csv:12743}`, `all_attributes.json:12725` | network query | `src/coana.py:12712` |
| `cross-dataset_<src>_to_<tgt>_<codes>_<ts>/` | folder name `src/comparison/comparison_parameters.py:1532` (run stamp cached at `:866`); results written by `src/comparison/comparison_analyzer.py`: `comparison_results/unified_summary.csv:6700`, `edge_presence_matrix.csv:8787`, `threshold_alignment_best_matches.csv:7460`, `run_manifest.json:6302`, `comparison_report_used_data/{similarity_by_query.csv:6743, query_resolution.csv:6349, type_appearance_order.csv:6333}` | the cross-dataset comparison UI's whole run | readers `src/comparison/report_tabbed.py:263, 287, 388-391` |
| `NB-find-lines_<DS>_<query>_<ts>/` | `line_summary.csv:9373`, `{q}_lines.csv:9201`, `gal4_lexa_summary.csv:9401` | NeuronBridge line hits | `src/neuronbridge_finder.py:9104`; pruning `src/neuronbridge_output_policy.py:215` |
| `NB-find-neurons_<line>_<ts>/` | `{line}_type_mapped.csv:6527`, `all_neurons.csv:7781` | neurons within a line | `:7595`; policy `:263` |
| `NB-colabeling_<lines>_<ts>/` | `expression_matrix.csv:2487`, `expression_matrix_merged.csv:2759`, `colabeling_matrix_{method}.csv:8342`, `line_summary.csv:8474` | co-labeling | `:8190`; policy `:303` |
| `NB-find-lines-expanded_<DS>_<q>_<ts>/` | per-chip `line_summary.csv` | expansion of a query | `src/neuronbridge_query_expansion.py:675` |
| `similar-morphology_<DS>_<q[:40]>_<ts>/` | `results.csv:8692`, `type_summary.csv:8693` | morphology similarity search | `src/morphology.py:8688` |
| `morphology_comparison_<DS>_<q>_<ts>/` | `{type_level/type_similarity\|group_level/group_similarity}_{method}.csv:1330`, `bodyid_level/bodyid_similarity_{method}.csv:1333`, `members.csv:1353`, `visualization/heatmap_{level}_{method}.html`, `report.html`, `parameters.json` | grouped morphology comparison | `src/morphology_comparison.py:823`; reader `:1235-1245`, `tests/core/test_morphology.py:1025` |

Bulk that is **not** citable: `profiles/**/*_profile.json`
(`src/comparison/profile_comparator.py:7789`, `:12723`), `overlaps/` (`:7792`),
the Excel-only connectivity matrices `conn_mat_*` (`src/coana.py:2208`, `:2216`, `:2224`),
`visualization/*`, and the NeuronBridge image cache
(`src/neuronbridge_finder.py:4464` → `line_image_mapping.json`). Also: every pathfinding
folder carries an `*_allpaths_info.xlsx` (`src/coana.py:17901`) beside its CSVs — cite the
CSV.

Skills: `drocat-usage` (`tabs/connectivity-profiling.md`, `find-homologs.md`, `find-path.md`,
`find-shortest.md`, `network.md`, `find-similar.md`, `inter-dataset.md`, `nb-find-lines.md`,
`nb-find-neuron.md`, `nb-colabel.md`) and `drocat-backend`
(`modules/profile-comparator.md`, `comparison.md`, `coana-connectivity.md`, `neuronbridge.md`,
`morphology-similarity.md`, `morph-cross-dataset.md`).

## 6. The data layer (on-disk inputs, not run outputs)

Resolved output directory: `ui/config.py:75 get_default_output_dir()` reads
`ui/local_config.json` (`LOCAL_CONFIG_FILE` `ui/config.py:19`, key `default_output_dir`
`:77`, setter `:91`, per-tab override `:20/:138/:143`; UI field
`ui/tabs/settings.py:596-599`). Fallback `local_data/` (`ui/config.py:17`).
**`datasets/`, `cache/`, `neuron_indexes/` are always repo-root-relative** regardless of
that setting (`src/coana.py:2757`, `src/build_connection_cache.py:139`,
`src/neuron_index_builder.py:231`, `src/synapse_cache.py:157`, `src/morphology.py:2367`).

| Artifact | Topic | Writer | Reader (use this accessor) | Ignored at |
| --- | --- | --- | --- | --- |
| `datasets/<safe>/<safe>_allneurons_neuron_df.{csv,parquet}` | the neuron table: ids, `type`, `cell_type`, `Class` | `src/FAFB_file_converter.py:695 ensure_flywire_data` (paths `:716-717`), `src/BANC_file_converter.py:641`, `:569` | `src/flywire_ids.py:118 resolve_flywire_dataset_dir`, `src/fafb_utils.py:21 prepare_flywire_data` | `.gitignore:13` |
| `datasets/<safe>/<safe>_synapse_table.parquet`, `_merged_connections.parquet`, `_allneurons_roi_count_df.parquet` | synapse/connectivity/ROI tables | `src/banc_public_data.py:787 ensure_synapse_derived_table` (`:803`), `src/visualize_skeleton.py:17440` | `src/visualize_skeleton.py:9824 _get_synapse_table_path`, `src/synapse_cache.py:258`, `:200` | `.gitignore:13` |
| `datasets/<safe>/<safe>_metadata.json` | counts/freshness sidecar for the above | `src/BANC_file_converter.py:387`, `:557`; `scripts/generate_dataset_metadata.py:52` | `ui/dataset_service.py:864`, `ui/roi_options.py:71` | `.gitignore:13` |
| `neuron_indexes/<safe>/neuron_index.parquet` (+`_search`) | the queryable index every type-level answer reads | `src/build_seed_indexes.py:102` (`:127-128`), `src/coana.py:5841`, `:11263` | `ui/neuron_index.py:253`, `:385`, `:526 load_cached_neuron_index` | `.gitignore:26` (3 seeds + `manifest.json` re-included `:27-38`) |
| `cache/<safe>/connections.parquet` + `_batch_files/` | consolidated connectivity cache | `src/coana.py:5622 _consolidate_batch_files` | `src/coana.py:5116 _load_connection_db`, `src/comparison/profile_comparator.py:4686` | `.gitignore:15` |
| `cache/<safe>/cache_manifest.json` | `{schema, dataset, distinct_connections, built_at, source}` | `src/coana.py:4139 _write_cache_manifest` (`:4142-4148`) | guards `:4075`, `:4080`, loss tolerance `:4306` | `.gitignore:15` |
| `cache/<safe>/connectivity_profiles.parquet`, `available_rois.json` | profile / ROI caches | `src/build_connectivity_profile_cache.py:162`, `src/visualize_skeleton.py:17440` | `src/coana.py:4046`, `:4230 _check_cache_coverage`, `ui/roi_options.py:153` | `.gitignore:15` |
| `cache/<safe>/skeletons/`, `find_similar/morphology/skeleton__vectors_v2.parquet` + `meta_v2.json` | fetched skeletons and the morphology vector store | `src/morphology.py:2917 build` → `:3207` (V2 class `:3746`, paths `:3780`) | `src/comparison/morph_cross_dataset.py` store readers | `.gitignore:15` |
| `cache/<safe>/synapses/{pre}_{post}.parquet`, `banc_id_crosswalk.parquet`, `cache/dataset_availability.json`, `cache/storage_audit.json` | ROI-granular synapse pairs, id crosswalk, availability/inventory | `src/synapse_cache.py:258`, `src/storage_inventory.py:40` | `ui/dataset_service.py:864` | `.gitignore:15` |
| `local_data/`, `outputs/` | queued runs, scenes, scratch — **disposable by design** | per tool | — | `.gitignore:19`, `:83` |

Sizes on this machine (why none of it is committable): `datasets/flywire_FAFB_v783` 14 GB,
`datasets/banc_v888` 1.6 GB, `cache/male-cns_v1_0` 695 MB, `cache/banc_v888` 644 MB,
`cache/flywire_FAFB_v783` 447 MB, `local_data/` 32 GB. Skill: `drocat-backend`
(`references/util-support.md:48-83`, `modules/coana-connectivity.md`), `drocat-install`
(`references/troubleshooting.md:64, 108`), `drocat-usage` (`references/datasets-and-auth.md:42`).

## 7. `_clock_production/` — the article repo's own exports

Separate git repo (branch `clock-production`), DROCAT-ignored via `.git/info/exclude`;
its rules are `_clock_production/AGENTS.md` (§7 commands, §3 data, §5 records, §8
quotability). `<producer>` there is
`_clock_production/drocat_backend.py:47 PRODUCER = "type_mapper"`, and record folders come
from `:52 new_record()`.

| Pattern | Script → DROCAT-side call | Answers |
| --- | --- | --- |
| `data/raw/type_mapping/FAFB-to-{BANC,MCNS}/type_mapper_results/mapping_*_<stamp>.{csv,html}` | `analysis/01_…:107-131` → `_compute_type_mapping` (`:77,:89`) + `build_bridges_csv` (`:112`) | the §0 clock population's type-level mapping, both volumes |
| `data/raw/type_mapping/FAFB-to-{MCNS,BANC}/TM VEV/<target>_<mode>_<stamp>/…` | `analysis/06_tmvev_ingest.py:232-234` — **the mode label is inserted here, not by DROCAT** (`:122 labelled()`, `:116 stamp_of()`, `:99 mode_of_live()`) | ingested TM VEV results core, scenes excluded |
| `data/processed/type_mapper-clock_profile_groups.json` | `analysis/03_…:218-220` → `flow_is_claimed`/`mapping_pool_key` (`:143`) | per-type claim groups feeding every figure |
| `data/raw/skeleton_profiles/type_mapper-<stamp>/{FAFB,MCNS,BANC}/<TYPE>.png` | `analysis/04_…:123,143,188` → `export_individuals_from_html` (`:175`) | the 63 profile cells |
| `figures/type_mapper-clock_profile_plate.pptx` + `_<n>.png` previews | `analysis/05_…:347,394-401,424,427`; geometry gate `:276 check_geometry` | the A4 plate (deck page 1) |
| `…/type_mapper-population-<stamp>/<COL>/` + deck page 2 | `analysis/07_…:136,151,195,285-297` | all 21 types overlaid per volume |
| `records/<date>_…-banc-clock-sets-<stamp>/banc_clock_set_membership.csv` (+summary, manifest) + deck page 3 | `analysis/08_…:219,227-234,254,274,348,354` | BANC's own `circadian_neuron` vs what FAFB's clock maps onto it |
| `figures/type_mapper-flow_network-<stamp>.html` + `data/raw/type_mapping/bridges/type_mapper-flow_network-<stamp>/{_sankey,_heatmap,_xlsx,_guide}` | `analysis/09_bridge_flow_network.py:136-143` (view → figures, siblings + inputs beside it; layout-node gate `:113`, differing-bytes refusal `:68`) driving the Net-Viz `plot-network_` run of §4 | the mapper's bridge **vocabulary** as one graph (5 dataset codes + 9 bridge columns, 44 edges). Direction-agnostic, and every input `weight = 1`, so it is a schematic of the lanes — never a pair's mapping, never a flow quantity |
| `user_data/*` (indexed in `user_data/README.md`) | hand-curated; scripts read-only | plate row order; curated double-claim assignments; **`type_mapper_bridges_edgelist.csv` + `type_mapper_bridges_layout.json`, tracked precisely because DROCAT cannot regenerate them** — `09` refuses the pair unless the layout's 14 node keys equal the edge list's node set |

Live records right now: claim sets `records/2026-09-28_type_mapper-baseline/`; plate
`records/2026-09-27_type_mapper-clock-profile-plate-full4/`; population
`records/2026-09-27_type_mapper-clock-population-pop1/`; BANC sets
`records/2026-09-27_type_mapper-banc-clock-sets-sets2/`; matrix ingest
`records/2026-09-28_tmvev-ingest-tmvev-clock-matrix-20260928/`; bridge flow network
`records/2026-09-28_type_mapper-flow_network-20260928_113705/`.
`data/` is ignored except force-added small evidence; `archive/**/data_raw/` stays off git;
nothing over 100 MB is ever committed.

## 8. Per-file authority: the run-guide registry

`ui/output_guide.py` `TOOL_GUIDE_SPECS` (`:1145`) is the real file→meaning map, and it
describes files **as they exist in that run** (first-match-wins against the actual
directory, `assemble_run_content:2626`, with a `leftovers` bucket for anything
unrecognized). One entry per tool, 18 tools: `find_path:1146`, `find_shortest:1153`,
`find_network:1161`, `plot3d_skeleton:1163`, `plot3d_reexport:1236`, `plot_path:1264`,
`find_homologs:1283`, `find_similar_morphology:1290`, `connectivity_profiling:1316`,
`morph_cross_dataset:1431`, `morphology_comparison:1488`, `inter_dataset:1547`,
`nb_find_lines:1954`, `nb_find_lines_expanded:1991`, `nb_find_neuron:2039`,
`nb_colabel:2085`, `flylight_download:2163`, `type_mapping_validation:2183`.

Each file spec carries `pattern`, `description`, `columns`, `preview`, `preview_title`,
`matrix`; column meanings come from `COLUMN_GLOSSARY:42` (`glossary_entry:517`), and the
guide also renders applied thresholds (`:2823`, `:2864`, `:2925`), key params (`:3166`),
`user_warning_notes.txt` (`:2765`) and the TM VEV term glossary (`:2614`). It is written by
`write_run_guide:3873` from `ui/runner.py:540` on every successful tab run and from
`scripts/RunMappingValidation.py:378`; format is `run_guide_format`
(`ui/config.py:516`, `:844`, env `DROCAT_RUN_GUIDE_FORMAT`), basename
`_UserGuide_please_read_me` (`ui/output_guide.py:28`). Preview flags feed
`ui/components/result_previews.py:18, 75`.

**So: this index names the file; the run guide explains its columns.**

## 9. Trap register (each has bitten someone)

*   **`export_png_webdriver()` cannot open any DROCAT page.** `:22770` waits
    `By.CLASS_NAME, "plotly"` while `_write_plotly_html` (`:6151`) emits
    `plotly-graph-div`; working sessions wait `js-plotly-plot` (`:1373`) and the JS
    resolvers fall back (`.js-plotly-plot || .plotly-graph-div`: `:4458`, `:4838`, `:5250`).
    It burns its 60 s timeout and returns `None`. Tests hide it by stubbing selenium
    (`tests/core/test_visualize_skeleton_coverage.py:5554-5566`). Use
    `export_individuals_from_html` (`:837`).
*   **Non-uniform rescale.** the live `plot_individuals` kaleido path hardcodes 900×900
    (`src/visualize_skeleton.py:20418`); only
    `export_individuals_from_html` (`:839`) forwards a real viewport.
*   **`_compute_unified_crop_bounds` returns `(row_min, row_max, col_min, col_max)`**
    (`:982`, `:1003`), *not* PIL's `(left, top, right, bottom)`. The wrong unpacking still
    produces a valid-looking deck whose cells are slivers.
*   **Read bodyId columns as strings.** `dtype={"bodyId": str}` — 18-digit ids widen to
    float64 and silently round when targets are blank
    (`src/comparison/comparison_analyzer.py:9049`, `:9082`;
    `src/neuronbridge_finder.py:3917`).
*   **`matched` ≠ `best`.** `matched` is a target-side pool `category`; `best` is the
    mutual-best count in `pair_summary.csv`. Artifacts older than 2026-09-24 use the old
    naming.
*   **Pooled level is `group_level/`, not `type_level/`.** At `aggregation_level='custom'`
    profile similarity lands in `group_level/results/group_similarity_*`
    (`src/comparison/profile_comparator.py:12368 _aggregate_names`, `:12444-12461`), and at
    `'bodyid'` there is **no `type_level/` and no `profiles/aggregated/`** (`:12480`,
    `:12502`); same rule `src/morphology_comparison.py:802-813`.
*   **Rank columns are dropped from every homolog CSV** —
    `_drop_rank_cols:164` strips `rank_corr` / `rank_union_raw`, so a sort key is not
    recoverable from disk.
*   **Volume columns are not cross-dataset comparable**: the nm³ caliber is spelled
    per dataset (`size` / `size_nm` / `Volume (nm^3)`), folded only inside
    `src/comparison/body_id_resolver.py:1035-1075`.
*   **A pool is not a claim set.** Counts like 198 vs 207 vs 242 differ by basis
    (`flow_is_claimed`, §2); never compare a pool figure to a claim figure.
*   **No code checks a cache's id namespace.** Measured 2026-09-28: `cache/hemibrain_v1_2_1/
    connections.parquet` holds 7,369,077 rows, of which **20,583 carry an 18-digit
    FlyWire-space bodyId on at least one side** (20,441 in `bodyId_pre`, 20,400 in
    `bodyId_post`), while `neuron_indexes/hemibrain_v1_2_1/neuron_index.parquet` holds
    186,061 rows and every id there is 9 or 10 digits — so those rows join to nothing.
    Both integrity layers still pass, because `_compute_cache_integrity`
    (`src/coana.py:4091`) only counts flagged rows.
*   **Scene pages are huge**: >50 MB warns (`src/visualize_skeleton.py:9623`, threshold
    `:9637`), simplification caps kaleido at 100 MB / webdriver at 200 MB (`:2965-2972`),
    and Plotly JS is embedded per page (`:6150`).
*   **The validation mode is not in DROCAT's run folder name** — only in
    `parameters.json → validation_mode`. An article repo may add it locally (07 of §7);
    never infer a mode from a stamp.

## 10. Coverage gaps, and how to check this index

Two clusters have **no skill coverage**: the cross-dataset type-mapper export cluster
(`mapping_*_<ts>.csv/html`, `mapping_all_pairs`, `auto_type_mapping*` — the browser-download
path), and the label-mapper / neuron-index export cluster
(`cache/user_mappings/_inline` from `ui/components/mapping_editor.py:30-36`,
`matched_entries_*.csv` from `ui/components/neuron_index_viewer.py:1582, 1652, 2502, 3084`).
Front door for docs is `docs/README.md:1` (structure `:100-141`, navigation `:143-166`,
`OUTPUT_FILES.md` linked at `:176`); sub-indexes are `docs/technical/README.md`,
`docs/core-features/README.md`, `docs/ui_guides/README.html:19-21`,
`docs/visualizations/README.md`. The audit-record cluster is `skills/repository-audit`.

### Audit gate this file was accepted at (2026-09-28)

**192 `file:line` cites resolved and in range (0 out-of-range), 102 symbol assertions
checked against the source line ±3 with 101 passing, and the registry count confirmed by
parsing the module: `TOOL_GUIDE_SPECS` = 18 entries.** The one non-pass was a checker
expectation, not a citation (`src/visualize_skeleton.py:19709` writes into `views_folder`;
the `exported_views` name itself is `:19572`, cited separately).

Recipe to re-run after a refactor — no script is checked in, so have an agent do it:
(1) extract every `path:LINE` and bare `:LINE` from this file; a bare cite belongs to the
nearest preceding **source** file on its line, never to a data filename (`parameters.json`
etc. own no line numbers); (2) assert `1 ≤ line ≤ len(file)`, resolving abbreviated paths by
suffix against the repo; (3) assert the symbol named next to each cite appears within ±3
lines of it. Four cites failed that gate when this file was first written and were
corrected: `_RUN_CSV_SCHEMAS` is `src/comparison/mapping_validation.py:1122` (not `:1236`,
which is inside the dict), `export_png_webdriver` is defined at
`src/visualize_skeleton.py:22653` (broken wait `:22770`, name site `:22727`), the
cross-dataset result CSVs are written by `src/comparison/comparison_analyzer.py` (not
`comparison_parameters.py`, which only names the folder), and `run_label` lives at
`src/comparison/mapping_validation.py:377` / `:6840`. A cited-but-nonexistent
`layer_{i}.csv` was dropped, with two other unverified bulk filenames.
