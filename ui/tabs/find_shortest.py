"""
FindShortestPath Tab - Shortest (minimum hop-count) pathfinding between
neuron groups. Backend: FindNeuronConnection.FindShortestPath (the shared
FindAllPath pipeline with shortest-only enumeration).
"""

from nicegui import ui

from ..config import FILTER_OPTIONS, OUTPUT_FORMATS, NETWORK_LAYOUTS, SEARCH_COLUMNS, get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid,
    tool_page, ALL_NEURONS_TOKEN, uses_all_neurons_token,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions


def _has_multiple_source_type_queries(values, search_columns="auto"):
    """Return whether a Shortest Paths source query is a large type set.

    The UI accepts both type/name queries and bodyIds in the same chip field.
    Numeric values are therefore treated as explicit bodyIds and do not raise
    this advisory.  ``auto`` is intentionally included because it is the
    default search scope and most users enter neuron types without changing
    the advanced selector first.
    """
    scope = str(search_columns or "auto").strip().casefold()
    if scope not in {"auto", "type"}:
        return False
    if uses_all_neurons_token(values):
        # ``all_neurons`` is a special single-side mode; any other chips are
        # ignored by the execution path and should not trigger this advisory.
        return False

    query_values = []
    for value in values or []:
        text = str(value or "").strip()
        if text and not text.isdigit():
            query_values.append(text)
    return len(dict.fromkeys(query_values)) > 1


def create_find_shortest_tab():
    """Create the Shortest Paths tab UI."""
    runner = ScriptRunner()
    output_panel = OutputPanel(
        "Shortest Pathfinding Output", state_key="find_shortest"
    )
    dataset = None
    search_columns = None

    def _type_suggest(text):
        """Auto-suggest from the selected dataset's type names. Type matches
        come first for string input; the range expands to instance/bodyId only
        when no type matched and the search scope is 'auto'."""
        ds = dataset.value if dataset is not None else ""
        scope = search_columns.value if search_columns is not None else "auto"
        # Keep the complete candidate pool for local continuation filtering;
        # the input menu, not the backend matcher, limits visible rows.
        return dataset_suggestions(text, ds, scope, limit=None)

    form_col, results_col = tool_page(
        "Shortest Paths",
        "Find only the shortest (minimum hop-count) paths between source and target neuron groups.",
        icon="alt_route",
        doc="find_shortest.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-dataset"'):
            section_header("Dataset", "storage")
            dataset = dataset_selector(
                hint="Select the connectome dataset.",
                allow_custom=True,
                group_recommended=True,
            )
            output_dir = dir_input(scope="find_shortest")

        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-neurons"'):
            section_header("Neuron Selection", "hub")
            source_input = neuron_list_input(
                label="Source Neurons",
                placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10)",
                hint="Enter neuron types, bodyIds, or patterns. Upload CSV/TSV/Excel for large lists. "
                     "Type 'all_neurons' to load every neuron in the dataset.",
                suggestions=_type_suggest,
                available_neurons=lambda: dataset.value if dataset is not None else "",
            ).classes("drocat-fixed-neuron-input")
            target_input = neuron_list_input(
                label="Target Neurons",
                placeholder="Type or upload CSV/TSV/Excel (e.g., PPL101, DN1p)",
                hint="Enter neuron types, bodyIds, or patterns. Upload CSV/TSV/Excel for large lists. "
                     "Type 'all_neurons' to load every neuron in the dataset.",
                suggestions=_type_suggest,
                available_neurons=lambda: dataset.value if dataset is not None else "",
            ).classes("drocat-fixed-neuron-input")
            source_type_warning = ui.label(
                "⚠️ Shortest Paths works best with a small source set. Multiple "
                "source type queries can expand to many bodyIds and make the "
                "target-rooted search larger and harder to interpret. Prefer a "
                "single source type/query, or enable bodyId output for exact "
                "source-target pairs."
            ).classes("text-caption text-amber-8")
            source_type_warning.set_visibility(False)
            ui.label(
                "Tip: type 'all_neurons' as the only chip of one side to load every "
                "neuron in the dataset and fetch all adjacent neurons at your "
                "thresholds. It forces Max Intermediate Layers = 0 (direct "
                "connections only) and cannot be used on both sides."
            ).classes("text-caption drocat-muted")
            mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                label="Custom Grouping",
                tab_key="find_shortest",
                datasets_provider=lambda: [dataset.value] if dataset.value else [],
                watch_elements=[dataset],
                query_inputs={"source": source_input, "target": target_input},
            )

        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                # Intentionally 5 (not the shared max_interlayer default): the
                # shortest-path tab targets deeper searches; keep in sync with
                # the e2e test asserting this divergence.
                max_interlayer = number_input(
                    "Max Intermediate Layers", 5, 0, None,
                    hint="Maximum number of intermediate neuron layers between source "
                         "and target. Higher = more paths but slower.",
                )
                min_synapse = number_input(
                    "Min Synapse Count", get_user_default("min_synapse_num"), 1, 100,
                    hint="Minimum number of synapses for a connection to be included. Filters out weak/noisy connections.",
                )
                max_paths = number_input(
                    "Max Paths (BodyId)", get_user_default("max_paths_bodyid"),
                    0, 100000000,
                    hint="Path-output budget for the StrongestFirst min-hop "
                         "enumeration: when the search exceeds it, ALL min-hop "
                         "paths above the achieved strength cutoff (tau) are "
                         "kept and tau is reported. Filters the emitted PATHS "
                         "only — the graph is not trimmed. 0 = auto (1M budget).",
                )
                edge_limit = number_input(
                    "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                    hint="Drawing-only cap: at most this many unique edges are "
                         "rendered per visualization (network / Sankey / heatmap). "
                         "It never changes fetching, the graph, or the path "
                         "output; a single complete path may still exceed it to "
                         "stay intact.",
                )
            find_reciprocal = checkbox_input(
                "Find Reciprocal Connections", False,
                hint="Enrich the path graph with reciprocal direct connections.",
            )

        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-output"'):
            section_header("Output Options", "output")
            with param_grid(2):
                custom_source_name = ui.input(
                    label="Custom Source Name (optional)",
                    placeholder="e.g., aMe_clock",
                ).classes("w-full").tooltip("Custom label for source group in output files/plots.")
                custom_target_name = ui.input(
                    label="Custom Target Name (optional)",
                    placeholder="e.g., PPL1_dopamine",
                ).classes("w-full").tooltip("Custom label for target group in output files/plots.")
            with param_grid(3):
                output_format = select_input(
                    "Output Format", OUTPUT_FORMATS, get_user_default("output_format"),
                    hint="'csv': faster, smaller. 'xlsx': Excel format with formatting.",
                )
                network_layout = select_input(
                    "Network Layout", NETWORK_LAYOUTS, get_user_default("network_layout"),
                    hint="Layout algorithm for the HTML network visualization.",
                )
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe_clock_shortest",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the unified auto name "
                    "(find-paths-shortest_<dataset>_<src>_to_<tgt>_<params>_<timestamp>)."
                )
            with ui.row().classes("gap-4"):
                skip_bodyid = checkbox_input(
                    "Skip BodyId in Output", get_user_default("skip_bodyId"),
                    hint="Exclude individual bodyId-level results. Only show type-level aggregation.",
                )
                show_fig = checkbox_input(
                    "Show Figure", get_user_default("showfig_analysis"),
                    hint="Open the interactive HTML visualization automatically after completion.",
                )
                drop_untyped = checkbox_input(
                    "Drop Untyped Neurons", get_user_default("drop_untyped"),
                    hint="Neuron-label filter (shared with Cross-Dataset "
                         "Comparison): remove edges touching untyped neurons "
                         "(empty / Unknown / bodyId-fallback type labels) "
                         "BEFORE the path graph is built. Dropped rows: "
                         "data_details/untyped_dropped_records.csv; counts "
                         "in user_warning_notes.txt.",
                )

        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                separate_hemi = checkbox_input(
                    "Hemisphere-aware", False,
                    hint="Split type/group aggregation into _L/_R/_U hemisphere labels.",
                )
                hemi_filter = select_input(
                    "Hemisphere", ["both", "left", "right"], "both",
                    hint="'both': all neurons. 'left'/'right': restrict to that hemisphere. "
                         "Neurons WITHOUT an explicit hemisphere (no _L/_R instance suffix "
                         "or Soma side) are always included in every option.",
                    inline=True,
                )
            with ui.row().classes("items-center gap-4 flex-wrap"):
                symmetry_analysis = checkbox_input(
                    "Symmetry Analysis", False,
                    hint="Generate ipsilateral vs contralateral symmetry outputs.",
                )
            with ui.row().classes("items-center gap-4 flex-wrap"):
                keep_hemi_conserved = checkbox_input(
                    "Keep Only Hemisphere-Conserved Edges", False,
                    hint="Keep only edges conserved between hemispheres (requires Hemisphere-aware).",
                )
            def _sync_hemisphere_options():
                if separate_hemi.value:
                    keep_hemi_conserved.enable()
                    symmetry_analysis.enable()
                    hemi_filter.set_enabled(True)
                else:
                    # dependent options are unchecked AND disabled so a
                    # greyed-out True never reaches the backend
                    keep_hemi_conserved.value = False
                    symmetry_analysis.value = False
                    hemi_filter.value = 'both'
                    keep_hemi_conserved.disable()
                    symmetry_analysis.disable()
                    hemi_filter.set_enabled(False)
            separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
            _sync_hemisphere_options()

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-findshortest-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                keyword_filter = neuron_list_input(
                    label="Keywords to Exclude from Paths",
                    show_filter=False,
                    show_upload=False,
                    hint="Paths containing these keywords in neuron types will be removed. "
                         "Type a keyword and press Enter (or leave the field) to add it as a chip.",
                )

                with param_grid(3):
                    # F9: ratio/probability filters are disabled by default
                    # (ratio is a readout column). Round-13 AND mode makes
                    # the ratio a co-threshold with Min Synapse Count.
                    min_ratio = number_input(
                        "Min Connection Ratio", 0, 0, 1, 0.01,
                        hint="Readout column by default. Under Combine "
                             "Thresholds = Both (AND) it co-thresholds the "
                             "graph: an edge must pass BOTH the synapse "
                             "count and this ratio.",
                    ).set_visibility(False)
                    threshold_combination = select_input(
                        "Combine Thresholds", ["Any one (default)", "Both (AND)"],
                        default=("Both (AND)"
                                 if get_user_default("threshold_combination")
                                 == "and" else "Any one (default)"),
                        hint="How Min Synapse Count and Min Connection Ratio "
                             "combine when both are set. Both (AND): an edge "
                             "enters the graph only when it clears BOTH "
                             "thresholds (round-13).",
                    )
                    min_traversal = number_input(
                        "Min Traversal Prob.", 0, 0, 1, 0.01,
                        hint="Disabled: traversal_probability is a readout column "
                             "(ratio/0.3, capped at 1.0) — it no longer filters.",
                    ).set_visibility(False)
                    # Fix C: the lossy bodyId edge limit is deprecated and
                    # ignored — shortest paths are always complete.

                    def _sync_shortest_combination():
                        min_ratio.set_visibility(
                            threshold_combination.value == "Both (AND)")

                    threshold_combination.on_value_change(
                        lambda _e: _sync_shortest_combination())
                    _sync_shortest_combination()

                with param_grid(3):
                    # plan-shortest-batched-discovery: discovery memory for
                    # many broad targets is bounded per enumeration batch.
                    discovery_batch_budget = number_input(
                        "Discovery Batch Budget", 2_000_000, 0, 100_000_000,
                        hint="Shortest-mode memory control: discovery labels "
                             "stream to shortest_discovery_store/ and "
                             "enumeration runs per batch of targets whose "
                             "combined BFS distance states stay under this "
                             "budget. Discovery memory otherwise grows with "
                             "targets x frontier x depth. 0 = single batch "
                             "(legacy behavior). Results are identical "
                             "either way.",
                    )
                    target_batch_size = number_input(
                        "Targets per Batch", 0, 0, 100_000,
                        hint="Simple alternative to the batch budget: fixed "
                             "number of targets per enumeration batch. "
                             "0 = use the Discovery Batch Budget instead.",
                    )
                    # plan-shortest-store-retention: what happens to the
                    # run's shortest_discovery_store/ after a successful
                    # run (wired like the Complete Paths budget knobs).
                    store_retention = select_input(
                        "Discovery Store", ["keep", "compact", "prune"],
                        "keep",
                        hint="keep: leave shortest_discovery_store/ in the "
                             "run folder (default; full audit surface). "
                             "compact: merge the connection layers into one "
                             "4-column file and drop the derivable dag-edge "
                             "chunks (roughly halves a deep run's store). "
                             "prune: delete all store data files, keeping "
                             "meta.json as the size census. Path outputs are "
                             "identical in every mode.",
                    )
                    # Coverage early-stop (shortest): stop deepening backward
                    # discovery once both sides' per-TYPE coverage
                    # requirements are met, instead of exhausting the depth
                    # bound on per-pair completeness. Emitted pairs keep
                    # exact per-pair minimum hops; the stop is disclosed in
                    # the notes + diagnostics.
                    _COVERAGE_LEVELS = ["Any", "25%", "50%", "75%", "Full",
                                        "Custom %"]
                    _coverage_hint = (
                        "Per queried TYPE: Any = at least 1 enrolled bodyId "
                        "reached; a % = each queried type must individually "
                        "reach that share of its enrolled bodyIds; Full = "
                        "all. Checked at every discovery layer; deepening "
                        "stops once BOTH sides are satisfied (deeper pairs "
                        "within the depth bound are then not searched)."
                    )
                    shortest_source_coverage = select_input(
                        "Source Coverage", _COVERAGE_LEVELS, "Any",
                        hint="Source side of the coverage early-stop. " +
                             _coverage_hint,
                    )
                    shortest_target_coverage = select_input(
                        "Target Coverage", _COVERAGE_LEVELS, "Full",
                        hint="Target side of the coverage early-stop "
                             "(default Full = stop once every queried "
                             "target type is fully reached). " + _coverage_hint,
                    )
                    coverage_custom_pct = number_input(
                        "Custom Coverage %", 50, 1, 100,
                        hint="Used by whichever side is set to 'Custom %': "
                             "each queried type must individually reach "
                             "this share of its enrolled bodyIds.",
                    ).set_visibility(False)

                    def _sync_coverage_custom():
                        coverage_custom_pct.set_visibility(
                            "Custom %" in (shortest_source_coverage.value,
                                           shortest_target_coverage.value))

                    shortest_source_coverage.on_value_change(
                        lambda _e: _sync_coverage_custom())
                    shortest_target_coverage.on_value_change(
                        lambda _e: _sync_coverage_custom())
                    _sync_coverage_custom()

                    def _coverage_payload(value):
                        """UI level -> engine fraction (None = legacy)."""
                        if value == "Any":
                            return 0.0
                        if value == "Full":
                            return 1.0
                        if value.endswith("%"):
                            try:
                                return float(value[:-1]) / 100.0
                            except ValueError:
                                return None
                        return None  # Custom % handled by the caller

                    def _coverage_side_payload(select):
                        raw = _coverage_payload(select.value)
                        if raw is None and select.value == "Custom %":
                            raw = max(
                                1, min(100, int(coverage_custom_pct.value
                                                or 50))) / 100.0
                        return raw

                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
                )
                filter_by = select_input(
                    "Filter By", FILTER_OPTIONS, get_user_default("filter_by"),
                    hint="'bodyId': filter at individual neuron level. 'type': aggregate by neuron type.",
                )

                with ui.row().classes("gap-4"):
                    use_cache = checkbox_input(
                        "Use Cache", get_user_default("use_cache"),
                        hint="Cache neuron data locally for 10-100x speedup on repeated runs.",
                    )
                    cache_only = checkbox_input(
                        "Cache Only (Offline)", get_user_default("cache_only"),
                        hint="Use only local cache and never contact the server. "
                             "Requires the cache to be pre-built.",
                    )

    def _update_source_type_warning(_event=None):
        """Keep the source-size advisory visible while the query is edited."""
        _mode, values = source_input.get_value()
        scope = search_columns.value if search_columns is not None else "auto"
        source_type_warning.set_visibility(
            _has_multiple_source_type_queries(values, scope)
        )

    # The shared input callback covers Enter, removal, upload/viewer
    # synchronization, clear, and blur-committed text.  The filter/search
    # scope can also turn a query into an explicit type search, so refresh on
    # those selectors as well.
    source_input.add_value_change_listener(_update_source_type_warning)
    if source_input.filter_mode is not None:
        source_input.filter_mode.on_value_change(_update_source_type_warning)
    search_columns.on_value_change(_update_source_type_warning)
    _update_source_type_warning()

    with results_col:
        output_panel.create(run_label="Shortest Paths", run_icon="alt_route")

    async def run_shortest():
        # Get values from neuron_list_input (returns (mode, list))
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()
        # Uploads are also included in get_value(); refresh once at execution
        # time so the advisory is correct even if the file picker was used.
        _update_source_type_warning()

        # 'all_neurons' is a special token: it loads the full neuron set on
        # that side (fetch all adjacent neurons at the given thresholds),
        # replaces every other chip, and forces direct connections only.
        # The backend enforces the same rules for script/API callers.
        src_all = uses_all_neurons_token(src_neurons)
        tgt_all = uses_all_neurons_token(tgt_neurons)
        if src_all and tgt_all:
            ui.notify(
                "'all_neurons' cannot be used as both source and target",
                type="warning",
            )
            return

        sources = [ALL_NEURONS_TOKEN] if src_all else apply_filter_mode(src_neurons, src_mode)
        targets = [ALL_NEURONS_TOKEN] if tgt_all else apply_filter_mode(tgt_neurons, tgt_mode)

        if not sources:
            ui.notify("Please enter at least one source neuron", type="warning")
            return
        if not targets:
            ui.notify("Please enter at least one target neuron", type="warning")
            return

        if src_all or tgt_all:
            ui.notify(
                "'all_neurons' used — Max Intermediate Layers is forced to 0 "
                "(direct connections only)",
                type="warning",
            )

        # Resolve custom grouping first: an invalid inline board aborts the
        # run before the output panel enters its running state.
        mapping_path, mapping_ok = resolve_grouping()
        if not mapping_ok:
            return

        output_panel.clear()
        output_panel.set_running(True)

        # Parse keyword filter (chips are already individual keywords); the
        # empty-field default matches Complete Paths — the backend's own
        # 'None' default is overridden by an explicit empty list
        keywords = [str(k) for k in keyword_filter.get_value()[1]] or ['None']

        constructor_params = {
            "dataset": dataset.value,
            "sourceNeurons": sources,
            "targetNeurons": targets,
            "output_dir": output_dir.value,
            "min_synapse_num": int(min_synapse.value),
            # F9 default: ratio/prob are readouts (sent 0). Round-13 AND
            # mode: the ratio co-thresholds with Min Synapse Count.
            "threshold_combination": (
                "and" if threshold_combination.value == "Both (AND)"
                else "or"),
            "min_ratio": (float(min_ratio.value)
                          if threshold_combination.value == "Both (AND)"
                          else 0.0),
            "min_traversal_probability": 0.0,
            "max_interlayer": 0 if (src_all or tgt_all) else int(max_interlayer.value),
            "filter_by": filter_by.value,
            "max_paths_bodyid": int(max_paths.value) or None,
            # Fix C/D: shortest mode is never floored — the Edge Budget
            # does not apply here.
            "graph_edge_limit_bodyid": 0,
            # plan-shortest-batched-discovery: batched discovery +
            # budgeted enumeration (0 = legacy monolithic path).
            "discovery_batch_budget": int(discovery_batch_budget.value) or 0,
            "target_batch_size": int(target_batch_size.value) or 0,
            # plan-shortest-store-retention: post-run store handling.
            "discovery_store_retention": store_retention.value,
            # Coverage early-stop: UI defaults Any source + Full target.
            "shortest_source_coverage": _coverage_side_payload(
                shortest_source_coverage),
            "shortest_target_coverage": _coverage_side_payload(
                shortest_target_coverage),
            # Shortest Paths no longer exposes the early network preview in
            # the UI; keep the backend behavior explicitly disabled.
            "visualize_before_reconstruct": False,
            "search_columns": search_columns.value,
            "network_layout": network_layout.value,
            "use_cache": use_cache.value,
            "edgeN_limit": int(edge_limit.value),
            "output_format": output_format.value,
            "skip_bodyId": skip_bodyid.value,
            "showfig": show_fig.value,
            "custom_source_name": custom_source_name.value or '',
            "custom_target_name": custom_target_name.value or '',
            "keyword_in_path_to_remove": keywords,
            "cache_only": cache_only.value,
            "drop_untyped": drop_untyped.value,
            "saveas": saveas.value.strip() or "",
            "separate_hemispheres": separate_hemi.value,
            "hemisphere_filter": hemi_filter.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }
        if mapping_path:
            constructor_params["custom_mapping_file"] = mapping_path

        result = await output_panel.run(runner, "find_shortest", constructor_params, "find_shortest",
                                        output_dir=output_dir.value)

        # InitializeNeuronInfo runs before the shortest-path search and
        # reports the resolved source/target counts. Record only after that
        # confirmation so zero-match queries stay out of history.
        match_info = result.get("neuron_match") or {}
        if match_info.get("any_pair"):
            from ..history_store import record as _record_history
            _record_history(
                [str(v) for v in src_neurons + tgt_neurons],
                datasets=[dataset.value] if dataset.value else [],
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_shortest)
    output_panel.cancel_button.on_click(runner.cancel)
