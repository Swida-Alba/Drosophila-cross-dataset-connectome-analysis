"""Connectivity Tab - Homolog finding and connectivity profile comparison.

Two sub-tabs share this page:
- Find Homolog: homolog-style connectivity search (Target = Source gives the
  intra-dataset similar-neuron search), with an optional cross-dataset
  morphological qualification of the visualized top matches.
- Comparison: profile existing neurons within and across datasets.
"""

from nicegui import ui

from ..config import (
    SIMILARITY_METRICS,
    get_tab_output_dir,
    get_user_default,
)
from ..components.common import (
    apply_filter_mode,
    checkbox_input,
    dataset_multi_selector,
    dataset_selector,
    dir_input,
    neuron_list_input,
    number_input,
    param_grid,
    section_header,
    select_input,
    tool_page,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions, datasets_suggestions


def _resolve_comparison_output_dir(value):
    """Resolve the path selected in the Comparison sub-tab before a run starts."""
    selected = str(value or "").strip()
    return selected or str(get_tab_output_dir("connectivity_profiling")).strip()


def create_connectivity_tab():
    # One runner per output panel: a shared runner would let a second run
    # clobber the first run's process handle (cancel would kill the wrong
    # process and orphan the other).
    similar_runner = ScriptRunner()
    comparison_runner = ScriptRunner()
    similar_output = OutputPanel(
        "Similarity Output", state_key="connectivity_similar"
    )
    comparison_output = OutputPanel(
        "Comparison Output", state_key="connectivity_comparison"
    )
    source_dataset = None
    datasets_select = None

    def _similar_suggest(text):
        dataset_name = source_dataset.value if source_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    def _comparison_suggest(text):
        selected = datasets_select.value if datasets_select is not None else []
        return datasets_suggestions(text, selected, limit=None)

    form_col, results_col = tool_page(
        "Connectivity",
        "Find homologs by connectivity profile and compare profiles "
        "within and across datasets.",
        icon="analytics",
        doc="connectivity.md",
    )

    with form_col:
        # Sub-tab switch: Find Homolog vs Comparison
        mode_value = {"value": "Find Homolog"}
        with ui.row().classes(
            "w-full items-center justify-between gap-8 px-2"
        ):
            similar_mode_button = ui.button("Find Homolog").props(
                "outline no-caps"
            ).classes("w-5/12")
            comparison_mode_button = ui.button("Comparison").props(
                "outline no-caps"
            ).classes("w-5/12")
            for button in (similar_mode_button, comparison_mode_button):
                button.style(
                    "min-height: 3.5rem; font-size: 1.1rem; "
                    "font-weight: 700;"
                )

        # ================= Find Homolog panel (homolog search) =================
        with ui.column().classes("w-full gap-1") as similar_panel:
            with ui.card().classes("w-full drocat-card").props(
                'id="card-connectivity-similar-datasets"'
            ):
                section_header("Datasets", "storage")
                with param_grid(2):
                    source_dataset = dataset_selector(
                        label="Source Dataset",
                        hint="Dataset where the source neuron lives.",
                    )
                    target_dataset = dataset_selector(
                        label="Target Dataset",
                        default=get_user_default("default_target_dataset"),
                        hint="Dataset to search in. Set Target = Source for a "
                             "within-dataset (intra-dataset) similar-neuron "
                             "search.",
                    )
                output_dir = dir_input(scope="find_homologs")

            with ui.card().classes("w-full drocat-card").props(
                'id="card-connectivity-similar-neurons"'
            ):
                section_header("Source Neuron", "search")
                source_input = neuron_list_input(
                    label="Source Neuron(s) (type or bodyId)",
                    show_filter=False,
                    show_upload=True,
                    suggestions=_similar_suggest,
                    available_neurons=lambda: source_dataset.value
                    if source_dataset is not None else "",
                    hint="Enter one or more neuron types, bodyIds, or "
                         "higher-category labels (e.g. cell class). All inputs "
                         "are expanded per type and aggregated into one grouped "
                         "output folder.",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Search Parameters", "tune")
                with param_grid(3):
                    top_n = number_input("Top N Candidates", get_user_default("top_n"), 5, 100, hint="Number of top candidates to return.")
                    top_k = number_input("Top K Partners", get_user_default("top_k"), 5, 50, hint="Top K partners per direction for profile construction.")
                    top_m = number_input("Min Types (M)", get_user_default("top_m"), 3, 20, hint="Minimum unique partner types in profile.")
                with param_grid(2):
                    similarity_metric = select_input(
                        "Sort By", SIMILARITY_METRICS, get_user_default("similarity_metric"),
                        hint="Metric used ONLY for ordering the candidate list (top-N cut). "
                             "Selectable: jaccard (default, conservative — bounded [0,1], "
                             "never degenerate), rank_union, cosine. Every metric is still "
                             "computed by the backend; rank_corr is hidden because it is "
                             "unreliable at bodyId level (see BENCHMARK_RESULTS.md §6c).",
                    )
                with ui.row().classes("gap-4"):
                    use_fast = checkbox_input("Fast Search", get_user_default("fast_search"), hint="Use adjacency expansion for faster candidate discovery.")
                    vector_prefilter = checkbox_input("Vector Pre-filtering", get_user_default("vector_prefilter"), hint="Pre-filter candidates using vector cosine similarity.")
                    expand_2hop = checkbox_input("2-Hop Expansion", get_user_default("expand_2hop"), hint="Include 2-hop typed partners for untyped 1-hop neurons.")
                with param_grid(3):
                    min_synapse_threshold = number_input(
                        "Min Synapse Threshold", get_user_default("min_synapse_num"), 1, 100,
                        hint="Minimum synapse count for a connection to enter a profile.",
                    )
                    use_cache = checkbox_input(
                        "Use Cache", get_user_default("use_cache"),
                        hint="Cache profiles and connections locally for faster repeat searches.",
                    )
                    use_auto_type_mapping = checkbox_input(
                        "Auto Type Mapping", get_user_default("auto_type_mapping"),
                        hint="Resolve partner type names across datasets via the shared "
                             "resolver (licensed renames map, conflicts fail closed, "
                             "unmapped fall back to raw).",
                    )
                with ui.row().classes("w-full items-center gap-4"):
                    visualize = checkbox_input(
                        "Visualize Candidates",
                        True,
                        hint="Generate 3D skeleton visualizations of the top matches.",
                    )
                visualization_settings = skeleton_visualization_settings(
                    default_top_n=5,
                    top_n_label="Visualize Top N Candidates",
                    top_n_hint="Number of top candidates to render as 3D skeletons.",
                    default_visualize_by="type",
                    show_high_quality_warning=True,
                    # Similar searches render many candidates at once; do
                    # not pop the figure open unless the user asks for it.
                    default_show_fig=False,
                    dataset_provider=lambda: [
                        source_dataset.value,
                        target_dataset.value,
                    ],
                    dataset_watchers=[source_dataset, target_dataset],
                    card_id="card-connectivity-advanced-viz",
                )
                saveas = ui.input(
                    label="Save Folder Name (optional)",
                    placeholder="e.g., aMe12_similar",
                ).classes("w-full drocat-input").tooltip(
                    "Custom output folder name. Leave empty for the unified auto name "
                    "(homologs_<source_ds>_to_<target_ds>_<query>_<timestamp>)."
                )
                full_cache = checkbox_input(
                    "Pre-build Full Dataset Cache", False,
                    hint="Fetch connections for EVERY uncached neuron before searching. "
                         "Very slow on first use (can take hours); leave off to fetch only "
                         "the connections the search needs.",
                )

                # --- Morph qualification (cross-dataset; plan-morph-
                #     qualification-find-homolog.md). Off by default; applies
                #     to the visualized top-N only. ---
                morph_qualify = checkbox_input(
                    "Morph Qualification", False,
                    hint="Score the visualized top-N candidates against the "
                         "transformed query (production vector_v2, no NBLAST) "
                         "and gate them by a per-query null bar — the p95 of "
                         "200 seeded random target neurons. Candidates below "
                         "the bar are excluded from the 3D scene; the results "
                         "tables keep every row and gain morph columns. "
                         "Requires Visualize Candidates on and a supported "
                         "target (FAFB / male-cns / BANC); works for "
                         "cross-dataset and same-dataset (intra) runs alike.",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    ui.label("Qualification mode") \
                        .classes("text-caption drocat-muted")
                    morph_mode = ui.select(
                        {'null': 'Null bar (per-query null p95 + offset)',
                         'mapping_ref':
                             'Mapping-referenced floor (floors v3)'},
                        value='null', with_input=False,
                    ).classes("w-72").tooltip(
                        "Mapping-referenced floors need a cross-dataset "
                        "pair (mapper branch pools); disabled when "
                        "Target = Source — the null bar applies.")
                with ui.row().classes("w-full items-center gap-4"):
                    ui.label("Null percentile level") \
                        .classes("text-caption drocat-muted")
                    morph_level = ui.select(
                        {95: 'p95 (default)', 90: 'p90', 75: 'p75',
                         99: 'p99'}, value=95, with_input=False,
                    ).classes("w-40")
                with ui.row().classes("w-full items-center gap-4"):
                    ui.label("Qualification bar offset above the null bar") \
                        .classes("text-caption drocat-muted")
                    morph_bar_offset = ui.slider(
                        min=0.0, max=0.2, step=0.01, value=0.0
                    ).props("label-always").classes("w-48")
                    morph_bar_offset_value = ui.label("0.00 (bar = null p95)")

                    def _sync_offset_label(_e=None):
                        off = float(morph_bar_offset.value or 0)
                        if str(morph_mode.value or 'null') == 'mapping_ref':
                            morph_bar_offset_value.set_text(
                                f"{off:.2f} (bar = pool baseline - offset)")
                        else:
                            lvl = int(morph_level.value or 95)
                            morph_bar_offset_value.set_text(
                                f"{off:.2f} (bar = null p{lvl})")

                    morph_bar_offset.on_value_change(_sync_offset_label)
                    morph_level.on_value_change(_sync_offset_label)
                    morph_mode.on_value_change(_sync_offset_label)
                    _sync_offset_label()
                morph_scope_warning = ui.label("").classes(
                    "text-caption text-amber-8").set_visibility(False)

                def _morph_option_guard(_e=None):
                    src = str(source_dataset.value or "")
                    tgt = str(target_dataset.value or "")
                    allowed = any(k in tgt.lower() for k in (
                        "fafb", "flywire", "banc", "male-cns", "malecns"))
                    cross = bool(src) and bool(tgt) and src != tgt
                    # Same-dataset (intra) runs qualify too — the identity
                    # chain; only the target family is restricted.
                    # mapping_ref needs a cross-dataset pair (mapper branch
                    # pools), so it greys out when Target = Source — and its
                    # value resets, so a greyed-out mapping_ref never reaches
                    # the run record as this run's mode.
                    morph_qualify.set_enabled(allowed)
                    morph_mode.set_enabled(allowed and cross)
                    if allowed and not cross and morph_mode.value == "mapping_ref":
                        morph_mode.value = "null"
                    if tgt and not allowed:
                        morph_scope_warning.set_text(
                            "Morph qualification supports FAFB, "
                            "male-cns and BANC targets only.")
                        morph_scope_warning.set_visibility(True)
                    else:
                        morph_scope_warning.set_visibility(False)

                source_dataset.on_value_change(_morph_option_guard)
                target_dataset.on_value_change(_morph_option_guard)
                _morph_option_guard()

        # ================= Comparison panel (profile comparison) =================
        with ui.column().classes("w-full gap-1") as comparison_panel:
            with ui.card().classes("w-full drocat-card").props(
                'id="card-connectivity-comparison-datasets"'
            ):
                section_header("Datasets", "storage")
                datasets_select = dataset_multi_selector(
                    label="Datasets to compare (select one or more)",
                    hint="Select one or more datasets. One dataset with multiple thresholds "
                         "is also supported. Two or more datasets profile the same query in "
                         "each dataset (names mapped per dataset) and add within-dataset "
                         "(intra) plus across-dataset (inter, same neuron) comparisons. "
                         "The inter-dataset overview puts all queried neurons in rows and "
                         "dataset pairs in columns.",
                )
                comparison_output_dir = dir_input(scope="connectivity_profiling")

            with ui.card().classes("w-full drocat-card").props(
                'id="card-connectivity-comparison-neurons"'
            ):
                section_header("Query Neurons", "search")
                query_input = neuron_list_input(
                    label="Neurons to Compare",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10, aMe9)",
                    hint="Enter 2+ neurons to compare profiles. Upload CSV/TSV/Excel for large lists.",
                    suggestions=_comparison_suggest,
                    available_neurons=lambda: list(datasets_select.value or [])
                    if datasets_select is not None else [],
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Profile Construction", "build")
                with param_grid(2):
                    top_k_cmp = number_input(
                        "Top K Partners", get_user_default("top_k"), 5, 50,
                        hint="Number of top synaptic partners per direction to include in the profile.",
                    )
                    top_m_cmp = number_input(
                        "Min Unique Types (M)", get_user_default("top_m"), 3, 20,
                        hint="Minimum unique partner types. If top_k yields fewer, K is expanded.",
                    )

                # Custom grouping via the inline LabelMapper-format board
                # (only for the 'custom group' aggregation level)
                custom_group_box = ui.card().classes("w-full drocat-card").props('id=card-custom-group')
                with custom_group_box:
                    section_header("Custom Groups (LabelMapper)", "group_work")
                    mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                        label="Custom Grouping Preset",
                        hint="Groups from the history menu or defined inline for this query. "
                             "Each source-side group becomes one row of the comparison "
                             "matrix.",
                        tab_key="profiling",
                        datasets_provider=lambda: list(datasets_select.value or []),
                        watch_elements=[datasets_select],
                        query_inputs={"query": query_input},
                    )
                    ui.label(
                        "Groups are read from the grouping board's source mapping: each custom label "
                        "is one group, and its members for the selected datasets fill the rows. "
                        "The neuron query above is ignored in this mode."
                    ).classes("text-caption drocat-muted")
                custom_group_box.set_visibility(False)

                # --- Advanced Settings (kept at the bottom, in its own card) ---
                with ui.card().classes("w-full drocat-card").props(
                    'id="card-connectivity-advanced"'
                ):
                    with ui.expansion(
                        "Advanced Settings", icon="settings_suggest",
                    ).classes("w-full drocat-section-expansion"):
                        with ui.row().classes("gap-4"):
                            analyze_upstream = checkbox_input("Upstream", True, hint="Include presynaptic (input) partners in profile.")
                            analyze_downstream = checkbox_input("Downstream", True, hint="Include postsynaptic (output) partners in profile.")

                        ui.separator()
                        ui.label(
                            "All six similarity matrices are generated: overall, Jaccard, "
                            "weighted Jaccard, cosine, rank correlation, and rank-correlation "
                            "union (same metric set as the Find Homolog sub-tab). Overall "
                            "combines upstream and downstream connectivity."
                        ).classes("text-caption drocat-muted")
                        cluster_heatmap = checkbox_input(
                            "Generate Heatmaps", True,
                            hint="Create VisPath heatmaps for editing and Plotly heatmaps in the report.",
                        )
                        with param_grid(3):
                            min_synapse_threshold_cmp = number_input(
                                "Min Synapse Threshold", get_user_default("min_synapse_num"), 1, 100,
                                hint="Minimum synapse count for a connection to enter the profile.",
                            )
                            use_cache_cmp = checkbox_input(
                                "Use Cache", get_user_default("use_cache"),
                                hint="Cache profiles and connections locally for faster repeat comparisons "
                                     "(same setting as the Find Homolog panel above).",
                            )
                            aggregation_level = select_input(
                                "Aggregation Level", ["type", "bodyid", "custom group"], "type",
                                hint="'type': each matched neuron type is one row — patterns like "
                                     "'aMe.*' or name-filter inputs expand into their independent "
                                     "types. 'bodyid': every individual neuron is one row. "
                                     "'custom group': rows come from the custom grouping board below.",
                            ).props('id=select-aggregation')
                            skip_bodyid_level = select_input(
                                "BodyId-Level Computation", ["auto", "skip", "compute"], "auto",
                                hint="'auto': skip bodyId matrices only when >1000 bodyIds. "
                                     "'skip': type-level only. 'compute': always include bodyId "
                                     "and type-average-bodyId matrices. Type and bodyId levels "
                                     "are both available in the comparison output.",
                            )
                        with ui.row().classes("gap-4"):
                            show_figures = checkbox_input(
                                "Show Figures", False,
                                hint="Open generated heatmaps in the browser.",
                            )
                        full_cache_cmp = checkbox_input(
                            "Pre-build Full Dataset Cache", False,
                            hint="Fetch connections for EVERY uncached neuron before comparing. "
                                 "Very slow on first use (can take hours); leave off to use the "
                                 "connections already cached.",
                        )

                aggregation_level.on_value_change(
                    lambda e: custom_group_box.set_visibility(
                        (e.value or "") == "custom group"
                    )
                )

    with results_col:
        with ui.column().classes("w-full gap-1") as similar_output_container:
            similar_output.create(run_label="Find Homologs", run_icon="play_arrow")
        with ui.column().classes("w-full gap-1") as comparison_output_container:
            comparison_output.create(run_label="Run Comparison", run_icon="play_arrow")

    def sync_mode():
        is_similar = mode_value["value"] == "Find Homolog"
        similar_panel.set_visibility(is_similar)
        comparison_panel.set_visibility(not is_similar)
        similar_output_container.set_visibility(is_similar)
        comparison_output_container.set_visibility(not is_similar)
        similar_mode_button.props(
            "color=primary" if is_similar else "color=grey-7"
        )
        comparison_mode_button.props(
            "color=grey-7" if is_similar else "color=primary"
        )

    def set_mode(value: str):
        mode_value["value"] = value
        sync_mode()

    similar_mode_button.on_click(lambda _event: set_mode("Find Homolog"))
    comparison_mode_button.on_click(lambda _event: set_mode("Comparison"))

    async def run_similar():
        source_vals = source_input.get_value()[1]
        sources = []
        seen = set()
        for value in source_vals or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            sources.append(value)
        if not sources:
            ui.notify("Please enter at least one source neuron", type="warning")
            return

        similar_output.clear()
        similar_output.set_running(True)
        visualization_values = visualization_settings.values()
        if visualize.value:
            visualization_settings.warn_empty_custom_palettes()

        # Morph qualification is defined on the visualized set: without
        # visualization there is nothing to qualify (the plan's guard).
        morph_qualify_on = bool(morph_qualify.value)
        if morph_qualify_on and not visualize.value:
            ui.notify(
                "Morph qualification requires 'Visualize Candidates'; "
                "running without it.", type="warning")
            morph_qualify_on = False

        # Single combined run: pass the full list of source neurons so the
        # backend resolves each into its real types (coarse labels / other
        # string-column queries expand per type) and writes one output folder
        # grouped by query type.
        constructor_params = {
            "source": sources,
            "source_dataset": source_dataset.value,
            "target_dataset": target_dataset.value,
            "output_dir": output_dir.value,
            "top_n": int(top_n.value),
            "top_k": int(top_k.value),
            "top_m": int(top_m.value),
            "similarity_metric": similarity_metric.value,
            "vector_prefiltering": vector_prefilter.value,
            "include_untyped_partners": expand_2hop.value,
            "visualize_skeleton": visualize.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if visualize.value else 0
            ),
            "visualization_settings": visualization_values,
            "min_synapse_threshold": int(min_synapse_threshold.value),
            "use_cache": use_cache.value,
            "saveas": saveas.value.strip() or "",
            "use_auto_type_mapping": use_auto_type_mapping.value,
            "ensure_cache_complete": full_cache.value,
        }
        method_params = {
            "use_fast": use_fast.value,
            "morph_qualify": morph_qualify_on,
            "morph_bar_offset": float(morph_bar_offset.value or 0.0),
            "morph_mode": str(morph_mode.value or "null"),
            "morph_level": int(morph_level.value or 95),
        }

        try:
            result = await similar_output.run(
                similar_runner, "find_homologs", constructor_params,
                "find_homologs_multi", method_params=method_params,
                output_dir=output_dir.value,
            )
            if result.get("cancelled"):
                similar_output.set_status("Cancelled", "red")
                return
            succeeded = result.get("returncode") == 0
            similar_output.set_status(
                "Completed" if succeeded else "Failed",
                "green" if succeeded else "red",
            )
            if succeeded:
                from ..history_store import record as _record_history
                _record_history(
                    [str(s) for s in sources],
                    datasets=[source_dataset.value]
                    if source_dataset.value else [],
                )
            files = result.get("files", [])
            if files:
                similar_output.show_files(
                    list(files),
                    result.get("output_folder") or output_dir.value,
                )
        finally:
            similar_output.set_running(False)

    async def run_comparison():
        mode, neurons = query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if not query:
            ui.notify("Please enter at least one neuron", type="warning")
            return

        selected_datasets = list(datasets_select.value or [])
        if not selected_datasets:
            ui.notify("Select one or more datasets", type="warning")
            return

        skip_bodyid_param = {
            "auto": "auto",
            "skip": True,
            "compute": False,
        }.get(skip_bodyid_level.value, "auto")

        # Custom-group mode needs a mapping (preset or inline); resolve it
        # before the running state so an invalid board aborts cleanly.
        mapping_path = None
        if aggregation_level.value == "custom group":
            mapping_path, mapping_ok = resolve_grouping()
            if not mapping_ok:
                return
            if not mapping_path:
                ui.notify(
                    "Define inline groups (or load them from the history menu) for "
                    "the custom groups", type="warning")
                return

        # Determine direction from checkboxes
        if analyze_upstream.value and analyze_downstream.value:
            direction = 'both'
        elif analyze_upstream.value:
            direction = 'upstream'
        elif analyze_downstream.value:
            direction = 'downstream'
        else:
            ui.notify("Select upstream, downstream, or both", type="warning")
            return

        comparison_output.clear()
        comparison_output.set_running(True)
        try:

            # Keep the path selected in this sub-tab as the single source of truth
            # for both the backend constructor and the output-file scanner.  The
            # fallback matters when the input has not emitted its first browser
            # change event yet (for example, after opening the tab and clicking
            # Run immediately).
            output_path = _resolve_comparison_output_dir(comparison_output_dir.value)

            constructor_params = {
                "query": query,
                "datasets": selected_datasets,
                "output_dir": output_path,
                "top_k": int(top_k_cmp.value),
                "top_m": int(top_m_cmp.value),
                "min_synapse_threshold": int(min_synapse_threshold_cmp.value),
                "direction": direction,
                "generate_heatmaps": cluster_heatmap.value,
                "show_figures": show_figures.value,
                "verbose": True,
                "use_cache": use_cache_cmp.value,
                "aggregation_level": {
                    "type": "type",
                    "bodyid": "bodyid",
                    "custom group": "custom",
                }[aggregation_level.value],
                "skip_bodyId_level": skip_bodyid_param,
                "ensure_cache_complete": full_cache_cmp.value,
            }

            if aggregation_level.value == "custom group":
                constructor_params["custom_mapping_file"] = mapping_path

            result = await comparison_output.run(
                comparison_runner,
                "connectivity_profiling",
                constructor_params,
                "run",
                output_dir=output_path,
            )

            # The comparison pipeline resolves the query before comparing
            # profiles, so a completed run means the queried chips are useful
            # history entries for the selected datasets.
            if result["returncode"] == 0:
                from ..history_store import record as _record_history
                _record_history(
                    [str(v) for v in query],
                    datasets=list(selected_datasets),
                )

            comparison_output.set_status(
                "Completed" if result["returncode"] == 0 else "Failed",
                "green" if result["returncode"] == 0 else "red",
            )
            comparison_output.show_files(
                result["files"], result.get("output_folder") or output_path
            )
        finally:
            comparison_output.set_running(False)

    similar_output.run_button.on_click(run_similar)
    similar_output.cancel_button.on_click(similar_runner.cancel)
    comparison_output.run_button.on_click(run_comparison)
    comparison_output.cancel_button.on_click(comparison_runner.cancel)

    sync_mode()
