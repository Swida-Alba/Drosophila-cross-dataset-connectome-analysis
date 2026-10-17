"""Network tab - mutual direct connections among the queried neurons.

Backend: FindNeuronConnection.FindNetwork (FindAllPath-style enrichment,
hemisphere-aware analysis, network + heatmap visualizations without Sankey).
"""

from nicegui import ui

from ..config import OUTPUT_FORMATS, NETWORK_LAYOUTS, SEARCH_COLUMNS, get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components import pathfinding_controls as pc
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions


def create_network_tab():
    """Create the Network tab UI (FindNetwork)."""
    runner = ScriptRunner()
    output_panel = OutputPanel("Network Output", state_key="network")
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
        "Network",
        "Mutual direct connections among the queried neurons.",
        icon="schema",
        doc="find_network.md",
    )

    with form_col:
        # Scope notice: the network is limited to the queried neurons.
        ui.label(
            "ℹ️ This tool builds a LIMITED network of direct connections among "
            "the queried neurons only (no intermediate neurons are involved). "
            "For a more complete network that also involves intermediate neurons, "
            "use the Complete Paths tab with Find Reciprocal Connections enabled. "
            "Because the induced sub-network is fetched whole, it has no path "
            "budget to bite: Max Intermediate Layers, Max Paths, the Edge "
            "Budget, the applied tau and the Fast (corridor) lane do not apply "
            "here — their absence is the model, not a missing option."
        ).classes("text-caption text-amber-8 w-full")

        with ui.card().classes("w-full drocat-card").props('id="card-network-dataset"'):
            section_header("Dataset", "storage")
            dataset = dataset_selector(
                hint="Select the connectome dataset.",
                allow_custom=True,
                group_recommended=True,
            )
            output_dir = dir_input(scope="network")

        with ui.card().classes("w-full drocat-card").props('id="card-network-neurons"'):
            section_header("Neuron Selection", "hub")
            query_input = neuron_list_input(
                label="Query Neurons",
                placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, PPL101)",
                hint="The network is built from the direct connections WITHIN this set. "
                     "Enter neuron types, bodyIds, or patterns; upload CSV/TSV/Excel for large lists.",
                suggestions=_type_suggest,
                available_neurons=lambda: dataset.value if dataset is not None else "",
                expands_group_labels=True,
            ).classes("drocat-fixed-neuron-input")
            mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                label="Custom Grouping",
                tab_key="network",
                datasets_provider=lambda: [dataset.value] if dataset.value else [],
                watch_elements=[dataset],
                query_inputs={"query": query_input},
            )

        with ui.card().classes("w-full drocat-card").props('id="card-network-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                # The SAME threshold regime as Complete/Shortest Paths (one
                # definition each; the percent conversion happens in exactly
                # one place). Ratio mode floors the synapse count to 1 — the
                # ratio IS the threshold — and AND keeps both.
                threshold_basis = pc.threshold_basis_select(
                    hint="Which edge-weight basis thresholds the network: raw "
                         "synapse counts, or the connection ratio (synapses / "
                         "all-post incoming weight of the post neuron — "
                         "comparable across datasets).")
                min_synapse = number_input(
                    "Min Synapse Count", get_user_default("min_synapse_num"), 1, 100,
                    hint="Minimum number of synapses for a connection to be included. Filters out weak/noisy connections.",
                )
                min_ratio_threshold = pc.min_ratio_percent_input()
                threshold_combination = pc.threshold_combination_select(
                    hint="How Min Synapse Count and Min Connection Ratio "
                         "combine when both are set. Default: they are "
                         "alternatives — one threshold governs the network. "
                         "Both (AND): an edge enters only when it clears BOTH "
                         "thresholds (each on its own scale); both inputs stay "
                         "visible and the folder token carries both tiers.")
                edge_limit = number_input(
                    "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                    hint="Drawing-only cap: at most this many unique edges are "
                         "rendered per visualization (network / heatmap). It "
                         "never changes which connections are fetched or "
                         "exported; a single complete edge is never trimmed.",
                )

            def _sync_threshold_basis():
                pc.sync_threshold_visibility(threshold_basis, min_synapse,
                                             min_ratio_threshold,
                                             threshold_combination)

            for _el in (threshold_basis, threshold_combination,
                        min_ratio_threshold):
                _el.on_value_change(lambda _e: _sync_threshold_basis())
            _sync_threshold_basis()

        with ui.card().classes("w-full drocat-card").props('id="card-network-output"'):
            section_header("Output Options", "output")
            with param_grid(3):
                custom_group_name = ui.input(
                    label="Custom Group Name (optional)",
                    placeholder="e.g., aMe_clock",
                ).classes("w-full").tooltip("Custom label for the queried neuron group in output files/plots.")
                custom_target_name = ui.input(
                    label="Custom Target Name (optional)",
                    placeholder="e.g., targets",
                ).classes("w-full").tooltip("Custom label for the target side of the output files/plots (same queried set for Network).")
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe_clock_network",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the unified auto name "
                    "(find-network_<dataset>_<group>_<params>_<timestamp>)."
                )
            with param_grid(3):
                output_format = select_input(
                    "Output Format", OUTPUT_FORMATS, get_user_default("output_format"),
                    hint="'csv': faster, smaller. 'xlsx': Excel format with formatting.",
                )
                network_layout = select_input(
                    "Network Layout", NETWORK_LAYOUTS, get_user_default("network_layout"),
                    hint="Layout algorithm for the HTML network visualization.",
                )
                skip_bodyid = checkbox_input(
                    "Skip BodyId in Output", get_user_default("skip_bodyId"),
                    hint="Exclude individual bodyId-level connection tables. Type-level tables are always saved.",
                )
                # The drawing is this tool's only interactive output, so the
                # auto-open switch the path tabs have belongs here too
                # (it was missing, and the engine default silently disabled
                # it: FindNetwork passes showfig straight to VisualizePath).
                show_fig = checkbox_input(
                    "Show Figure", get_user_default("showfig_analysis"),
                    hint="Open the interactive HTML visualization automatically after completion.",
                )

        with ui.card().classes("w-full drocat-card").props('id="card-network-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
            # Identical on all three connection tabs by design (shared module).
            hemi = pc.hemisphere_block()

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-network-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                with param_grid(2):
                    # F9: ratio/probability filters are disabled (ratio is a
                    # readout column now). The entrances stay in the code,
                    # hidden, for the future ratio-weighted mode; the payload's
                    # ratio comes from the percent field in Core (shared with
                    # the path tabs — one unit, one conversion point).
                    min_ratio = number_input(
                        "Min Connection Ratio", 0, 0, 1, 0.01,
                        hint=pc.DORMANT_RATIO_HINT,
                    ).set_visibility(False)
                    min_traversal = number_input(
                        "Min Traversal Prob.", 0, 0, 1, 0.01,
                        hint="Disabled: traversal_probability is a readout column "
                             "(ratio/0.3, capped at 1.0) — it no longer filters.",
                    ).set_visibility(False)
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
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
                drop_untyped = checkbox_input(
                    "Drop Untyped Neurons", get_user_default("drop_untyped"),
                    hint="Neuron-label filter (shared with Complete Paths, "
                         "Shortest Paths and Cross-Dataset Comparison): remove "
                         "edges touching untyped neurons (empty / Unknown / "
                         "bodyId-fallback type labels). Dropped rows: "
                         "data_details/untyped_dropped_records.csv; counts in "
                         "user_warning_notes.txt.",
                )

    with results_col:
        output_panel.create(run_label="Find Network", run_icon="schema")

    async def run_network():
        mode, neurons = query_input.get_value()
        query = apply_filter_mode(neurons, mode)

        if not query:
            ui.notify("Please enter at least one query neuron", type="warning")
            return

        # The token is checked on the RAW chips, exactly as the two path tabs
        # check theirs: under the non-exact filter modes `apply_filter_mode`
        # rewrites a chip into a pattern (`.*all_neurons.*`), and the token is
        # matched by equality — so testing the rewritten list would let the
        # one query set the engine always rejects through the gate.
        if pc.refuses_all_neurons(neurons):
            return

        # Resolve custom grouping first: an invalid inline board aborts the
        # run before the output panel enters its running state.
        mapping_path, mapping_ok = resolve_grouping()
        if not mapping_ok:
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "dataset": dataset.value,
            # FindNetwork uses the queried set as both source and target
            # (mutual direct connections == source == target).
            "sourceNeurons": query,
            "targetNeurons": query,
            "output_dir": output_dir.value,
            # One derivation, shared with the two path tabs: the percent
            # field converts in exactly one place, and 'or' vs Both (AND)
            # means the same thing on all three pages.
            **pc.ratio_payloads(threshold_basis, threshold_combination,
                                min_synapse, min_ratio_threshold),
            # F9: ratio/probability filters disabled — hidden UI, sent 0.
            "min_traversal_probability": 0.0,
            "search_columns": search_columns.value,
            "network_layout": network_layout.value,
            "use_cache": use_cache.value,
            "edgeN_limit": int(edge_limit.value),
            "output_format": output_format.value,
            "skip_bodyId": skip_bodyid.value,
            "showfig": show_fig.value,
            "drop_untyped": drop_untyped.value,
            "custom_source_name": custom_group_name.value or '',
            "custom_target_name": custom_target_name.value or '',
            "cache_only": cache_only.value,
            "saveas": saveas.value.strip() or "",
            **hemi.payload(),
        }
        if mapping_path:
            constructor_params["custom_mapping_file"] = mapping_path

        result = await output_panel.run(runner, "find_network", constructor_params, "find_network",
                                        output_dir=output_dir.value)

        # FindNetwork initializes the queried neuron set before looking for
        # connections. Keep a query only when that resolution found neurons;
        # zero-match strings are not useful history entries.
        match_info = result.get("neuron_match") or {}
        if match_info.get("any_pair"):
            from ..history_store import record as _record_history
            _record_history(
                [str(v) for v in query],
                datasets=[dataset.value] if dataset.value else [],
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_network)
    output_panel.cancel_button.on_click(runner.cancel)
