# Backend module index

Consolidated index of the backend classes/functions and their key methods. For
full parameter lists, open the matching module guide in [`../modules/`](../modules/).
All paths are relative to the repo root unless noted.

## coana (`src/coana.py`)

- `FindNeuronConnection(dataset, sourceNeurons, targetNeurons, output_dir, min_synapse_num, min_ratio, min_traversal_probability, max_interlayer, filter_by, pathfinding, max_paths_bodyid, graph_edge_limit_bodyid, visualize_before_reconstruct, search_columns, network_layout, use_cache, edgeN_limit, output_format, skip_bodyId, showfig, custom_source_name, custom_target_name, keyword_in_path_to_remove, cache_only, saveas, separate_hemispheres, hemisphere_filter, keep_only_hemisphere_conserved_connections, symmetry_analysis, find_reciprocal, custom_mapping_file)`
  - `InitializeNeuronInfo()`
  - `FindDirectConnections()`
  - `FindPath(find_bodyId_path=None)`
  - `FindAllPath(find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None, use_graph_cache=True, find_reciprocal=False)`
  - `FindShortestPath(find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None, use_graph_cache=True, find_reciprocal=False)`
  - `FindNetwork()`
  - `build_connection_cache(neuron_types=None, neuron_bodyIds=None, batch_size=100, force_rebuild=False, quiet=False, progress_callback=None, cancel_event=None, max_workers=None, status_callback=None)`
  - `build_connectivity_profile_cache(neuron_types=None, top_k=10, top_m=5, expand_2hop=True, max_neurons=None, force_refresh=False, progress_callback=None)`

## morphology (`src/morphology.py`)

- `MorphologyComparer(query, dataset, level, method, metric, candidate_cap, candidate_source, visualize_top_n, visualize_by, min_weight, min_shared_partners, roi_filter, ...)` → `find_similar()`
- `SkeletonVectorCache(dataset, project_root=None, ...)` → `build(fetch_missing=0)`, `ensure(fetch_missing=0)`, `coverage()`, `vectors_for(body_ids, compute_missing=True)`
- `find_similar_raw_cache(dataset, ...)`, `find_similar_dataset_cache_v2(dataset, ...)`, `find_similar_flywire_mesh_cache(...)`
- `fetch_skeleton_on_demand(dataset, body_id, ...)` / `fetch_skeletons_on_demand_batch(dataset, body_ids, ...)`: skeleton-native for EVERY dataset kind — FAFB delegates to `load_local_release_skeletons` (TreeNeuron, extrusion-checked), NeuPrint via `neuprint.fetch_skeleton`, BANC via the public SWC bucket. FAFB meshes are a different front-door: `_fetch_cave_mesh` / `CAVEDataFetcher.fetch_fafb_mesh` (the old mesh-returning FAFB branch, and the dead skeleton block behind it, were removed 2026-09-29 — record: `_plan/plan-tmvev-reverse-scene-loader-defects.md`).
- Raw-basis design: the skeleton cache stores RAW (level-0) skeletons — fetch
  pipelines default `simplification=0`; vector caches (V1/V2) build from raw
  trees only (simplified files are skipped, never releveled). Simplification
  is applied at visualization/render time only. Legacy simp90 files can be
  quarantined with `scripts/maintenance/purge_legacy_simp90_cache.py`.

## morphology_comparison (`src/morphology_comparison.py`)

- `MorphologyProfileComparer(dataset, query, method, aggregation_level, custom_mapping_file, max_members_per_type, max_total_neurons, output_dir, saveas, generate_heatmaps, show_figures, use_cache, verbose, n_workers)` → `run()` — intra-dataset N×N morphology comparison; `aggregation_level` picks the matrix row (`type` / `bodyid` / `custom group`), the bodyId matrix is always the scored primitive and the type-level file is absent at bodyId level (`method="vector_v2"` on the whitened vector cache, `"nblast"` on dotprops, which warns past 30 neurons and proceeds; BANC runs with a provisional-scores warning).

## comparison (`src/comparison/__init__.py`)

- `ComparisonParameters(datasets, source_neurons, target_neurons, output_folder, comparison_mode, path_mode, max_interlayer, thresholds, top_edges, max_paths_bodyid, graph_edge_limit_bodyid, edgeN_limit, pathfinding, search_columns, skip_bodyId, cache_only, auto_type_mapping, _min_ratio, _min_prob, _output_format, parallel, max_workers, separate_hemispheres, keep_only_hemisphere_conserved_connections, symmetry_analysis, find_reciprocal, overall_mapping_json)`
- `ComparisonAnalyzer(params, verbose=True)` → `run_comparison()`, `run_all_analyses()`, `run_path_analysis()`, `run_edge_analysis()`, `export_results()`, `generate_report()`, `generate_html_report()`
- `quick_compare(datasets, source_neurons, target_neurons, ...)`
- `CrossDatasetTypeMapper`, `LabelMapper`, `DatasetConfig`, `DataLoader`, `ComparisonVisualizer`

## comparison.report_kit (`src/comparison/report_kit.py`)

- Shared tabbed-report machinery extracted from `ConnectivityProfileComparer`
  and now also driving the cross-dataset morphology comparison:
  `report_css()`, `report_script()`, `cluster_heatmap_matrix(matrix)`
  (Ward/VisPath ordering), `plotly_heatmap_fragment(...)`,
  `append_report_heatmap/_metric_grid/_tab_group(...)`,
  `generate_standalone_heatmaps(...)` (VisPath render + interactive-heatmap
  fallback), `MetricStyle` + `metric_style(key)` (positive/diverging scales),
  and the `REPORT_*_COLORSCALE` constants.

## comparison.profile_comparator (`src/comparison/profile_comparator.py`)

- `ConnectivityProfileComparer(query, dataset, top_k, top_m, min_synapse_threshold, direction, output_dir, generate_heatmaps, show_figures, skip_bodyId_level, verbose, use_cache, aggregation_level, ensure_cache_complete, custom_mapping_file)` → `run()` — intra/multi-dataset connectivity comparison; needs ≥2 neurons in scope (not 2 rows) and raises otherwise; the pooled level files as `type_level/` or, at `custom`, `group_level/`, while a `bodyid` run writes only `bodyid_level/` (its `type_avg_bodyid_*` folded from the same scores).
- `HomologFinder(source, source_dataset, target_dataset, output_dir, top_n, top_k, top_m, min_shared_partners, vector_prune_fraction, similarity_metric, vector_prefiltering, include_untyped_partners, min_synapse_threshold, use_cache, saveas, ensure_cache_complete, output_folder_prefix, visualize_skeleton, visualize_top_n, visualization_settings, use_auto_type_mapping, verbose)`
  - `find_homologs_fast()`, `find_homologs()`, `find_novel_homologs()`, `find_homologs_intra_dataset(...)`, `run_random_control_test(...)`
- `ProfileComparator` (static helpers, no constructor args) → `compare_profiles(...)`, `compare_profiles_simple(...)`
- `ComparisonResult`, `DEFAULT_SCORE_WEIGHTS`

## comparison.mapping_validation (`src/comparison/mapping_validation.py`)

BodyId-level type-mapping **validate-expand-visualize** orchestrator (CLI
`scripts/RunMappingValidation.py`). Rev 3.12:

- `MappingValidationConfig(...)` → `effective_mode` / `mode_rank` /
  `mode_at_least(mode)`; `validation_mode ∈ {restrictive, family,
  aggressive}` (nested).
- `MappingValidator(cfg)` → `resolve_type_pairs()`, `validate_pair(...)`,
  `annotate_invaders(...)`, `run_morphology(...)`,
  `finalize_categories(per_pair_res, sus, deep, fills, pool_detail)`,
  `run()`.
- Category partition (one ordered first-match per branch): tier
  (`matched`/`verified`/`borderline`/`unmatched`) > `sibling` >
  `candidates` > `family` > `relative` > `examinees` (renamed from
  'suspicious' 2026-09-18); helpers
  `classify_category(...)`, `candidate_annotation(...)`,
  `morph_qualified(...)`, `_leaf_token(...)`, `normalize_mode(...)`,
  `_dedup_rows_by_bid(...)`.
- Per-bodyId leaf token, ordered: `{T}(out-map)` (type is an in-map type;
  bodyId-level) > `{T}>{src}` > `{T}(no_source)` > `untyped`.
- Same-name-first consumers: `TypePair.same_name_first` provenance on
  fired selections; `same_name_excluded.csv` accounting for held /
  evidence-only / multivalue types; opt-in `verify_suspects` rival
  verification → `suspects_verification.csv` (advisory, default OFF).
- Exports → `validation_results.csv`, `pool_categories.csv`,
  `examinees.csv` (was `suspicious_candidates.csv`),
  `same_name_excluded.csv`, `source_candidates.csv`
  (out-of-map sources reaching branch pools, null-bar
  morph-qualified), `deep_candidates.csv`,
  `noise_filtered_candidates.csv`, `gap_fill_proposals.csv`,
  `gap_fill_dedup.csv`, `family_candidates.csv`, `relatives.csv`,
  `mapping_export.csv`, `pair_summary.csv`, `set_coverage.json`,
  `morphology_calibration.json`, `parameters.json`, `README.txt`,
  `visualization/*.html`.
- Scene renderer: `comparison.mapping_validation_visualize`
  (`build_category_buckets` reads the exported `category`;
  `_load_scene_skeletons` is the ONE dataset-general skeleton loader for
  every scene lane — FAFB/BANC via `load_local_release_skeletons` with
  `check_extrusions=True`, NeuPrint via the raw-skeleton cache + on-demand
  fetch, non-TreeNeuron objects refused; `check_scene_population` is the
  expected-vs-planted-vs-rendered self-check, and an empty query layer
  writes `SCENE_FAILED.txt` instead of rendering).

## neuronbridge_finder (`src/neuronbridge_finder.py`)

- `NeuronBridgeFinder(verbose=True, separate_splitgal4=False, region=None, max_workers=4)`
  - `find_lines_batch(queries, dataset, output_dir, match_type, ...)`
  - `find_neurons_batch(line_names, output_dir, match_type, ...)`
  - `analyze_colabeling(lines, output_dir, similarity_methods, ...)`
  - `visualize_colabeling_matrix(...)`, `visualize_expression_matrix(...)`, `visualize_expression_matrix_merged(...)`, `visualize_labeling_distribution(...)`, `visualize_colabeling_distribution(...)`

## neuronbridge_output_policy (`src/neuronbridge_output_policy.py`)

- `prune_find_lines_run(output_path, keep_per_match_csv, cleanup_source_images)` / `prune_find_neurons_run(...)` / `prune_colabel_run(...)`
- Idempotent Compact prune pass; audit merged into the run's `cleanup_audit.json`

## neuronbridge_coverage (`src/neuronbridge_coverage.py`)

- `refresh(datasets, client=None, force=False, ttl_days=7, nb_version=None)` → advisory `CoverageSnapshot` (exact / aligned / unavailable / unknown per dataset)
- `load_snapshot()`, `snapshot.covered_datasets(...)`, `snapshot.unavailable_datasets(...)`, `warnings_for(...)`, `sample_body_ids(dataset)`, `probe_coverage(...)`, `classify_records(...)`
- Snapshot: `cache/neuronbridge/coverage_snapshot.json`; offline never invalidates state

## neuronbridge_query_expansion (`src/neuronbridge_query_expansion.py`)

- `ExpandedLineFinder(verbose, separate_splitgal4, region, max_workers).run(queries, dataset, expand_names, coverage_datasets, ...)` → coverage-routed, name-expanded Find Lines (one `find_lines_batch` per chip; `NB-find-lines-expanded_*` run folder with `expansion_map.csv` / `expansion_summary.json` / `user_warning_notes.txt`)

## flylight_downloader (`src/flylight_downloader.py`)

- `FlyLightDownloader(output_dir, formats, image_types, region, collection_category, max_workers, simple_mode, use_boto3, include_vt_lines, verbose)`
  - `download(line_name, output_dir, max_files, flat_structure, add_timestamp, generate_summary, summary_images_per_page, ...)`
  - `list_vt_files(...)`, `list_categories()`

## visualize_skeleton (`src/visualize_skeleton.py`)

- `VisualizeSkeleton(dataset, neuron_layers, search_columns, hemisphere, custom_layer_names, output_dir, output_format, skeleton_mode, brain_mesh, vnc_mesh, legend_mode, neuron_alpha, neuron_colors, synapse_colors, background_color, skip_synapse, min_synapse_num, synapse_size, uniform_synapse_size, synapse_alpha, synapse_mode, mesh_roi, mesh_color, mesh_alpha, cache_neurons, cache_synapses, smooth_skeleton, show_soma, show_connectors, export_method, export_scale, export_views, show_fig, brain_mesh_color, neuprint_skeleton_pipeline, skeleton_mesh_simplification, ...)`
  - `plot_neurons()`, `plot_individuals(pdf_images_per_page, views, summary_format)`, `export_video(fps, degree_per_frame, rotate, export_gif, gif_scale, ...)`, `list_available_rois(refresh=False, fetch_online=True)`
- `WebDriverExportSession(width, height, scale, timeout, render_wait)` (Chrome driven)

## vispath (`vispath-subproject/src/vispath_pkg/vispath.py`)

- `VisualizePath(path_file, output_folder, network_layout, source_color, intermediate_color, target_color, link_color, showfig, output_format, generate_empty_network)`
  - `visualize(plot_heatmap=True, plot_Sankey=True, plot_network=True)`
  - `visualize_heatmap(...)`, `visualize_sankey()`, `visualize_network()`, `visualized_paths_for_export()`

## Supporting scripts (`src/` scripts)

- `python src/build_connection_cache.py <dataset>`
- `python src/build_connectivity_profile_cache.py <dataset>`
- `python src/build_seed_indexes.py`
- `src/FAFB_file_converter.py`, `src/BANC_file_converter.py` — FAFB and standalone BANC release conversion.
