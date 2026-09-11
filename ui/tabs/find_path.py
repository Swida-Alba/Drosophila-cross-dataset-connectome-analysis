"""
FindPath Tab - Multi-hop pathfinding between neuron groups.
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


def create_find_path_tab():
    """Create the FindPath tab UI."""
    runner = ScriptRunner()
    output_panel = OutputPanel("Pathfinding Output", state_key="find_path")
    dataset = None
    search_columns = None

    def _type_suggest(text):
        """Auto-suggest from the selected dataset's type names. Type matches
        come first for string input; the range expands to instance/bodyId only
        when no type matched and the search scope is 'auto'."""
        ds = dataset.value if dataset is not None else ""
        scope = search_columns.value if search_columns is not None else "auto"
        # Keep the complete candidate pool for local continuation filtering;
        # neuron_list_input applies the display limit after it narrows the
        # pool, so valid names beyond the first page remain reachable.
        return dataset_suggestions(text, ds, scope, limit=None)

    form_col, results_col = tool_page(
        "Complete Paths",
        "Discover multi-hop pathways between source and target neuron groups.",
        icon="route",
        doc="find_path.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-findpath-dataset"'):
            section_header("Dataset", "storage")
            dataset = dataset_selector(
                hint="Select the connectome dataset.",
                allow_custom=True,
            )
            output_dir = dir_input(scope="find_path")

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-neurons"'):
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
            ui.label(
                "Tip: type 'all_neurons' as the only chip of one side to load every "
                "neuron in the dataset and fetch all adjacent neurons at your "
                "thresholds. It forces Max Intermediate Layers = 0 (direct "
                "connections only) and cannot be used on both sides."
            ).classes("text-caption drocat-muted")
            mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                label="Custom Grouping",
                tab_key="find_path",
                datasets_provider=lambda: [dataset.value] if dataset.value else [],
                watch_elements=[dataset],
                query_inputs={"source": source_input, "target": target_input},
            )

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                max_interlayer = number_input(
                    "Max Intermediate Layers", get_user_default("max_interlayer"), 0, None,
                    hint="Maximum number of intermediate neuron layers between source and target. Higher = more paths but slower.",
                )
                min_synapse = number_input(
                    "Min Synapse Count", get_user_default("min_synapse_num"), 1, 100,
                    hint="Minimum number of synapses for a connection to be included. Filters out weak/noisy connections.",
                )
                edge_limit = number_input(
                    "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                    hint="Drawing-only cap: at most this many unique edges are "
                         "rendered per visualization (network / Sankey / heatmap). "
                         "It never changes fetching, the graph, or the path "
                         "output; a single complete path may still exceed it to "
                         "stay intact.",
                )
            interlayer_warning = ui.label(
                "⚠️ Layers ≥ 4: the path count grows combinatorially (branching^depth) — "
                "reconstruction can take hours and produce billions of paths. Raise "
                "Min Synapse Count, lower Max Paths (BodyId) or the Edge Budget "
                "so StrongestFirst keeps only the strongest routes, or "
                "minimize/batch the source and target sets."
            ).classes("text-caption text-amber-8").set_visibility(False)
            find_reciprocal = checkbox_input(
                "Find Reciprocal Connections", False,
                hint="Enrich the path graph with reciprocal direct connections.",
            )
            def _on_max_interlayer_change(e):
                interlayer_warning.set_visibility((e.value or 0) >= 4)

            max_interlayer.on_value_change(_on_max_interlayer_change)

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-output"'):
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
                    placeholder="e.g., aMe_clock_paths",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the unified auto name "
                    "(find-paths-complete_<dataset>_<src>_to_<tgt>_<params>_<timestamp>)."
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

        with ui.card().classes("w-full drocat-card").props('id="card-findpath-hemisphere"'):
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
        with ui.card().classes("w-full drocat-card").props('id="card-findpath-advanced"'):
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
                    # F9: ratio/probability filters are disabled (ratio is a
                    # readout column now). The entrances stay in the code,
                    # hidden, for the future ratio-weighted mode.
                    min_ratio = number_input(
                        "Min Connection Ratio", 0, 0, 1, 0.01,
                        hint="Disabled: connection_ratio is a readout column "
                             "(weight / all-post incoming weight) — it no longer "
                             "filters.",
                    ).set_visibility(False)
                    min_traversal = number_input(
                        "Min Traversal Prob.", 0, 0, 1, 0.01,
                        hint="Disabled: traversal_probability is a readout column "
                             "(ratio/0.3, capped at 1.0) — it no longer filters.",
                    ).set_visibility(False)
                with ui.row().classes("gap-4"):
                    with ui.column().classes("gap-0"):
                        max_paths_bodyid = number_input(
                            "Max Paths (BodyId)", get_user_default("max_paths_bodyid"),
                            0, 100000000,
                            hint="Path-output budget for StrongestFirst enumeration: "
                                 "when the search exceeds it, ALL paths above the "
                                 "achieved strength cutoff (tau) are kept and tau is "
                                 "reported. Filters the emitted PATHS only — the "
                                 "graph is not trimmed. "
                                 "0 = auto (StrongestFirst: 1M budget).",
                        )
                    with ui.column().classes("gap-0"):
                        edge_budget = number_input(
                            "Edge Budget", get_user_default("graph_edge_limit_bodyid"),
                            0, 100000000,
                            hint="Graph filter: after the lossless prunes, discovery "
                                 "cones exceeding this many bodyId edges are floored "
                                 "just above the N-th strongest edge's weight "
                                 "(w0 = w1 + 1) — exactly equivalent to raising the "
                                 "threshold; the applied floor is reported as "
                                 "edge_weight_floor. This is a graph budget, distinct "
                                 "from the drawing-only Visualization Edge Limit. "
                                 "0 = off. Never applied in Shortest Paths.",
                        )
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

    with results_col:
        output_panel.create(run_label="Complete Paths", run_icon="account_tree")

    async def run_pathfinding():
        # Get values from neuron_list_input (returns (mode, list))
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()

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

        # Parse keyword filter (chips are already individual keywords)
        keywords = [str(k) for k in keyword_filter.get_value()[1]] or ['None']

        constructor_params = {
            "dataset": dataset.value,
            "sourceNeurons": sources,
            "targetNeurons": targets,
            "output_dir": output_dir.value,
            "min_synapse_num": int(min_synapse.value),
            # F9: ratio/probability filters are disabled — hidden UI, sent 0.
            "min_ratio": 0.0,
            "min_traversal_probability": 0.0,
            "max_interlayer": 0 if (src_all or tgt_all) else int(max_interlayer.value),
            "filter_by": filter_by.value,
            # F1: StrongestFirst is the only 'all'-mode algorithm; the
            # selector was removed.
            "pathfinding": "StrongestFirst",
            # Fix D: the Edge Budget (lossy floor above the N-th strongest
            # edge, w0 = w1 + 1). 0 = off.
            "graph_edge_limit_bodyid": int(edge_budget.value),
            "max_paths_bodyid": int(max_paths_bodyid.value) or None,
            # Complete Paths no longer exposes the early network preview in
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

        result = await output_panel.run(runner, "find_path", constructor_params, "find_all_path",
                                        output_dir=output_dir.value)

        # InitializeNeuronInfo runs before the path search and reports the
        # resolved source/target counts. Record only after that confirmation;
        # unknown queries that resolve to zero neurons never enter history.
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

    output_panel.run_button.on_click(run_pathfinding)
    output_panel.cancel_button.on_click(runner.cancel)
