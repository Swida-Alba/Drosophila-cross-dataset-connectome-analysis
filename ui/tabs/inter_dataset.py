"""Cross-Dataset Comparison Tab - runs ComparisonAnalyzer over N datasets."""

import json
import logging
import os

from nicegui import ui
from ..config import COMPARISON_MODES, PATH_MODES, SEARCH_COLUMNS, get_user_default
from ..components.common import (
    dataset_multi_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.mapping_editor import custom_grouping_block
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_aware_suggestions


def create_inter_dataset_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Comparison Output", state_key="inter_dataset")
    datasets_select = None
    search_columns = None

    def _type_suggest(text):
        """Auto-suggest across all selected datasets' type names, with a
        dataset-aware gray hint (``type · male-cns:v1.0``) so cross-dataset
        entries stay traceable. Type matches come first for string input;
        the range expands to instance/bodyId only when no type matched and
        the search scope is 'auto'."""
        ds = datasets_select.value if datasets_select is not None else []
        scope = search_columns.value if search_columns is not None else "auto"
        # Keep the complete candidate pool for local continuation filtering;
        # the input menu, not the backend matcher, limits visible rows.
        return dataset_aware_suggestions(text, ds, scope, limit=None)

    form_col, results_col = tool_page(
        "Cross-Dataset Comparison",
        "Analyze one dataset across thresholds or compare connectivity across datasets.",
        icon="sync_alt",
        doc="cross_dataset.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-datasets"'):
            section_header("Datasets", "storage")
            datasets_select = dataset_multi_selector(
                label="Datasets to compare (one dataset with multiple thresholds is also supported)",
                group_recommended=True,
            )
            # Round 2 entrance: the standalone type-mapping preview. The
            # button stays disabled until >= 2 selected datasets have
            # cached neuron indexes; the popup composes the mapping across
            # every ordered pair of the selection (informational only).
            from ..components.type_mapping_panel import create_type_mapping_entry

            with ui.row().classes("items-center gap-4 flex-wrap"):
                type_mapping_button = create_type_mapping_entry(
                    lambda: list(datasets_select.value or []))

            def _sync_type_mapping_state(_e=None):
                type_mapping_button.refresh_state()

            datasets_select.on_value_change(_sync_type_mapping_state)
            _sync_type_mapping_state()
            output_dir = dir_input(scope="inter_dataset")

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-neurons"'):
            section_header("Neuron Selection", "hub")
            source_input = neuron_list_input(
                label="Source Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Source neurons for pathfinding. Type one query per chip or upload a CSV/TSV/Excel file (first column).",
                suggestions=_type_suggest,
                available_neurons=lambda: datasets_select.value if datasets_select is not None else [],
                # History rows tag the selected datasets where the value
                # actually resolves (exact local-pool membership), so
                # cross-dataset entries stay traceable without claiming
                # datasets a type does not exist in.
                show_history_datasets=True,
            ).classes("drocat-fixed-neuron-input")
            target_input = neuron_list_input(
                label="Target Neurons",
                placeholder="Type or upload CSV/TSV/Excel with neuron types/bodyIds",
                hint="Target neurons for pathfinding. Type one query per chip or upload a CSV/TSV/Excel file (first column).",
                suggestions=_type_suggest,
                available_neurons=lambda: datasets_select.value if datasets_select is not None else [],
                show_history_datasets=True,
            ).classes("drocat-fixed-neuron-input")
            mapping_select, _grouper_card, resolve_grouping = custom_grouping_block(
                label="Custom Mapping",
                datasets_provider=lambda: list(datasets_select.value or []),
                require_names=True,
                tab_key="inter_dataset",
                watch_elements=[datasets_select],
                query_inputs={"source": source_input, "target": target_input},
            )

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-core"'):
            section_header("Core Parameters", "tune")
            with param_grid(3):
                comparison_mode = select_input(
                    "Mode", COMPARISON_MODES, "path",
                    hint="'path': discover edges via path traversal. 'edge': compare edges independently by weight.",
                )
                path_mode = select_input(
                    "Path Enumeration", PATH_MODES, "all",
                    hint="'all': every path within the layer limit (FindAllPath). "
                         "'shortest': only per-pair minimum-hop paths (FindShortestPath) — "
                         "Max Layers is an EXACT depth bound (8 by default; high values "
                         "like 99 give an effectively unlimited search).",
                )
                max_interlayer = number_input(
                    "Max Intermediate Layers", 2, 0, 100,
                    hint="Maximum hops between source and target. In shortest mode this "
                         "is an EXACT depth bound: 0 = direct connections only, 8 = default, "
                         "and a high unreachable number (e.g. 99) gives an effectively "
                         "unlimited search.",
                )
            # Auto is the default mode: it measures each dataset's own
            # threshold window from one bootstrap run and emits BOTH the
            # per-threshold (vertical) and density-matched (horizontal)
            # comparisons — no threshold chips required.
            threshold_mode_value = {"value": "auto"}
            threshold_mode_buttons = {}
            section_header("Threshold Mode", "tune")
            with ui.row().classes(
                "w-full items-center justify-between gap-4 px-2 flex-nowrap"
            ):
                for _mode, _label in (
                    ("auto", "Auto"),
                    ("standard", "Standard"),
                    ("combinations", "Custom combination"),
                ):
                    _button = ui.button(_label).props(
                        "outline no-caps"
                    ).classes("w-1/3")
                    _button.style(
                        "min-height: 3rem; font-size: 1.0rem; font-weight: 700;"
                    )
                    threshold_mode_buttons[_mode] = _button

            thresholds_input = neuron_list_input(
                label="Synapse Thresholds",
                initial=[3, 5, 10],
                unit_label="threshold",
                show_filter=False,
                show_upload=False,
                hint="List of min synapse thresholds to analyze. "
                     "Type one threshold per chip (e.g. 3, 5, 10), or keep the defaults.",
            ).classes("w-full drocat-full-row-control")

            standard_threshold_hint = ui.label(
                "Standard mode: each chip is one comparison query shared by all selected datasets."
            ).classes("text-xs opacity-60 w-full")
            auto_threshold_hint = ui.label(
                "Auto: one bootstrap pathfinding run per dataset (floor = Min "
                "Synapse Count 3) measures each dataset's own threshold window "
                "[w_start, w_star_measured] from its searched graph, then emits "
                "BOTH comparisons — per-threshold rows (vertical: one identical "
                "threshold for every dataset, the like-for-like spine) and "
                "density-matched rows (horizontal: thresholds chosen per "
                "dataset to equalize E(t)/N, the density-matched envelope). "
                "No threshold chips needed."
            ).classes("text-xs opacity-60 w-full")
            auto_threshold_hint.set_visibility(False)
            combination_panel = ui.column().classes("w-full gap-2")
            combination_rows = [
                {"id": "combo_001", "label": "Combination 1", "values": {}}
            ]
            combination_inputs = []
            combination_table = None

            def _capture_combination_rows():
                """Persist current cell editors before rebuilding the table."""
                for row_state, input_map in combination_inputs:
                    for dataset, inp in input_map.items():
                        row_state.setdefault("values", {})[dataset] = str(
                            inp.value or ""
                        ).strip()

            def _combination_datasets():
                return list(datasets_select.value or [])

            def _rebuild_combination_table(_event=None):
                _capture_combination_rows()
                selected = _combination_datasets()
                combination_inputs.clear()
                combination_table.clear()
                if len(selected) < 2:
                    with combination_table:
                        ui.label(
                            "Custom combination requires at least two selected "
                            "datasets."
                        ).classes("text-xs text-amber-8")
                    return

                # The first time Custom combination mode is opened, make a usable row
                # from the first standard threshold. Newly selected datasets
                # in an existing table intentionally remain blank.
                default_values = thresholds_input.get_value()[1]
                default_value = (
                    str(default_values[0]) if default_values else ""
                )
                if not any(row.get("values") for row in combination_rows):
                    combination_rows[0]["values"] = {
                        dataset: default_value for dataset in selected
                    }

                with combination_table:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Query").classes("w-36 text-xs font-medium")
                        for dataset in selected:
                            ui.label(dataset).classes(
                                "flex-1 min-w-[120px] text-xs font-medium"
                            )
                        ui.label("").classes("w-8")

                    for index, row_state in enumerate(combination_rows, start=1):
                        row_state["id"] = row_state.get("id") or f"combo_{index:03d}"
                        row_state["label"] = row_state.get("label") or f"Combination {index}"
                        row_state.setdefault("values", {})
                        input_map = {}
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label(row_state["label"]).classes(
                                "w-36 text-xs font-medium"
                            )
                            for dataset in selected:
                                inp = ui.input(
                                    value=row_state["values"].get(dataset, ""),
                                    placeholder="positive integer",
                                ).props("dense outlined type=number").classes(
                                    "flex-1 min-w-[120px]"
                                )
                                input_map[dataset] = inp
                            remove = ui.button(
                                icon="delete_outline",
                                on_click=lambda _e, target=row_state: _remove_combination_row(target),
                            ).props(
                                'flat dense round color="negative" aria-label="Remove combination"'
                            ).classes("w-8")
                            remove.tooltip("Remove this query row")
                        combination_inputs.append((row_state, input_map))

                    with ui.row().classes("items-center gap-2"):
                        add = ui.button(
                            "Add combination", icon="add", on_click=_add_combination_row
                        ).props("outline dense")
                        add.tooltip("Add one complete cross-dataset threshold query")
                        seed = ui.button(
                            "Suggest from alignment", icon="insights",
                            on_click=_seed_from_alignment,
                        ).props("outline dense")
                        seed.tooltip(
                            "Prefill rows with density-equivalent thresholds from "
                            "a completed run's threshold_alignment output.")

            def _add_combination_row(_event=None):
                _capture_combination_rows()
                existing_indices = []
                for row_state in combination_rows:
                    row_id = str(row_state.get("id", ""))
                    if row_id.startswith("combo_"):
                        try:
                            existing_indices.append(int(row_id.rsplit("_", 1)[-1]))
                        except ValueError:
                            pass
                next_index = max(existing_indices or [0]) + 1
                combination_rows.append({
                    "id": f"combo_{next_index:03d}",
                    "label": f"Combination {next_index}",
                    "values": {},
                })
                _rebuild_combination_table()

            def _remove_combination_row(target):
                if len(combination_rows) <= 1:
                    ui.notify(
                        "Keep at least one threshold combination row.",
                        type="warning",
                    )
                    return
                _capture_combination_rows()
                combination_rows[:] = [
                    row for row in combination_rows if row is not target
                ]
                _rebuild_combination_table()

            def _seed_from_alignment(_event=None):
                """Prefill rows from a completed run's alignment suggestion.

                Reads comparison_results/suggested_threshold_combinations.json
                from the run output folder (solved by a prior standard run).
                """
                import json as _json
                import os as _os
                base = output_dir.value
                if not base:
                    ui.notify("Set an output folder first.", type="warning")
                    return
                path = _os.path.join(
                    base, "comparison_results",
                    "suggested_threshold_combinations.json")
                if not _os.path.exists(path):
                    ui.notify(
                        "No alignment suggestion found. Run a comparison "
                        "first; the suggestion is written to "
                        "comparison_results/suggested_threshold_combinations.json.",
                        type="warning")
                    return
                try:
                    with open(path, encoding="utf-8") as fh:
                        payload = _json.load(fh)
                    rows = payload.get("combinations", [])
                except Exception as exc:
                    ui.notify(f"Could not read suggestion: {exc}", type="negative")
                    return
                if not rows:
                    ui.notify("The suggestion file has no complete rows.",
                              type="warning")
                    return
                _capture_combination_rows()
                selected = set(_combination_datasets())
                combination_rows.clear()
                for i, row in enumerate(rows, start=1):
                    cell = {
                        ds: str(row["thresholds"][ds])
                        for ds in selected
                        if ds in row.get("thresholds", {})
                    }
                    combination_rows.append({
                        "id": row.get("id") or f"combo_{i:03d}",
                        "label": row.get("label") or f"Aligned {i}",
                        "values": cell,
                    })
                _rebuild_combination_table()
                ui.notify(
                    f"Seeded {len(rows)} aligned combination(s) from "
                    f"{_os.path.basename(path)}.", type="positive")

            with combination_panel:
                ui.label(
                    "Custom combination mode: each row is one query; every selected dataset "
                    "must have one positive threshold. Blank cells are invalid."
                ).classes("text-xs opacity-60 w-full")
                combination_table = ui.column().classes("w-full gap-1")
            combination_panel.set_visibility(False)

            def _sync_threshold_mode():
                mode = threshold_mode_value["value"]
                advanced = mode == "combinations"
                is_auto = mode == "auto"
                # Auto mode needs no threshold chips (the bootstrap floor
                # defaults to 3), so the chip editor only belongs to the
                # Standard and Custom combination modes.
                thresholds_input.set_visibility(not advanced and not is_auto)
                standard_threshold_hint.set_visibility(
                    not advanced and not is_auto)
                auto_threshold_hint.set_visibility(is_auto)
                combination_panel.set_visibility(advanced)
                for _mode, _button in threshold_mode_buttons.items():
                    _button.props(
                        "color=primary"
                        if _mode == threshold_mode_value["value"]
                        else "color=grey-7"
                    )
                if advanced:
                    _rebuild_combination_table()

            def _set_threshold_mode(value):
                threshold_mode_value["value"] = value
                _sync_threshold_mode()

            for _mode, _button in threshold_mode_buttons.items():
                _button.on_click(
                    lambda _e, mode=_mode: _set_threshold_mode(mode)
                )
            datasets_select.on_value_change(_rebuild_combination_table)
            _sync_threshold_mode()

            def _collect_threshold_configuration():
                """Return the canonical threshold payload for the active mode."""
                selected = _combination_datasets()
                mode = threshold_mode_value["value"]
                if mode == "auto":
                    # Auto mode never requires threshold chips: an empty box
                    # is valid and the bootstrap floor defaults to 3. Chips,
                    # when present, raise the floor to the lowest chip.
                    try:
                        values = [
                            int(value)
                            for item in thresholds_input.get_value()[1]
                            for value in str(item).replace(" ", "").split(",")
                            if value
                        ]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Invalid thresholds format. Use positive integers."
                        ) from exc
                    if any(value <= 0 for value in values):
                        raise ValueError(
                            "Synapse thresholds must be positive integers."
                        )
                    # The docs' contract (guide + skill): auto mode derives
                    # thresholds from cross-dataset overlap, so it needs at
                    # least two datasets — one selected dataset falls back
                    # to plain per-threshold vertical rows, which is NOT
                    # what the user asked Auto for.
                    if len(selected) < 2:
                        raise ValueError(
                            "Auto threshold mode needs at least two selected "
                            "datasets (it derives thresholds from cross-"
                            "dataset overlap)."
                        )
                    return mode, sorted(set(values)) or [3], None
                if mode == "standard":
                    try:
                        values = [
                            int(value)
                            for item in thresholds_input.get_value()[1]
                            for value in str(item).replace(" ", "").split(",")
                            if value
                        ]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Invalid thresholds format. Use positive integers."
                        ) from exc
                    if not values or any(value <= 0 for value in values):
                        raise ValueError(
                            "Please enter at least one positive synapse "
                            "threshold."
                        )
                    return mode, sorted(set(values)), None
                _capture_combination_rows()
                if len(selected) < 2:
                    raise ValueError(
                        "Custom combination requires at least two selected "
                        "datasets."
                    )
                combinations = []
                signatures = set()
                for index, row_state in enumerate(combination_rows, start=1):
                    values = {}
                    for dataset in selected:
                        raw_value = str(
                            row_state.get("values", {}).get(dataset, "")
                        ).strip()
                        if not raw_value:
                            raise ValueError(
                                f"Combination {index} is missing a threshold "
                                f"for {dataset}."
                            )
                        try:
                            value = int(raw_value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Combination {index} has an invalid threshold "
                                f"for {dataset}: {raw_value!r}."
                            ) from exc
                        if value <= 0:
                            raise ValueError(
                                f"Combination {index} threshold for {dataset} "
                                "must be positive."
                            )
                        values[dataset] = value
                    signature = tuple(values[dataset] for dataset in selected)
                    if signature in signatures:
                        raise ValueError(
                            f"Combination {index} duplicates an existing "
                            "threshold row."
                        )
                    signatures.add(signature)
                    combinations.append({
                        "id": row_state.get("id") or f"combo_{index:03d}",
                        "label": row_state.get("label") or f"Combination {index}",
                        "thresholds": values,
                    })
                return mode, [], combinations

            # Feature B: static rough-range hint (whole-dataset
            # connection-pair density per neuron). Informational only —
            # query-specific alignment is exported per run in the
            # threshold_alignment files.
            ui.label(
                "Rough whole-dataset hint (connection-pair density per neuron, "
                "measured 2026-09-25 on the cached typed universe): "
                "BANC ≥ 3 ≈ FAFB ≥ ~8 ≈ male-cns ≥ ~9; BANC ≥ 5 ≈ FAFB ≥ ~11 ≈ "
                "male-cns ≥ ~15 (FAFB v783, male-cns v1.0; BANC v626/v888). "
                "Query-specific alignment differs — check the threshold_alignment files "
                "in your run output."
            ).classes("text-xs opacity-60 w-full").style("line-height:1.35")
            find_reciprocal = checkbox_input(
                "Find Reciprocal Connections", False,
                hint="Build reciprocal graphs and include them in reports.",
            )

        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-hemisphere"'):
            section_header("Hemisphere Analysis", "sync_alt")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                separate_hemi = checkbox_input(
                    "Hemisphere-aware", False,
                    hint="Split type/group aggregation into _L/_R/_U hemisphere labels.",
                ).props('id=checkbox-separate-hemi')
            with ui.row().classes("items-center gap-4 flex-wrap"):
                symmetry_analysis = checkbox_input(
                    "Symmetry Analysis", True,
                    hint="Generate per-dataset hemisphere symmetry summaries (auto-enabled with Hemisphere-aware).",
                ).props('id=checkbox-symmetry')
            with ui.row().classes("items-center gap-4 flex-wrap"):
                keep_hemi_conserved = checkbox_input(
                    "Keep Only Hemisphere-Conserved Edges", False,
                    hint="Keep only edges conserved between hemispheres (requires Hemisphere-aware).",
                ).props('id=checkbox-hemi-conserved')
            def _sync_hemisphere_options():
                if separate_hemi.value:
                    keep_hemi_conserved.enable()
                    symmetry_analysis.enable()
                    # auto-enabled with Hemisphere-aware (per the hint)
                    symmetry_analysis.value = True
                else:
                    # uncheck + disable the hemisphere-dependent options so a
                    # greyed-out True is never passed to the backend
                    keep_hemi_conserved.disable()
                    keep_hemi_conserved.value = False
                    symmetry_analysis.disable()
                    symmetry_analysis.value = False
            separate_hemi.on_value_change(lambda _e: _sync_hemisphere_options())
            _sync_hemisphere_options()

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-interdataset-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                with param_grid(2):
                    # F1: StrongestFirst is the only 'all'-mode algorithm —
                    # the selector was removed; the payload sends the
                    # constant.
                    max_paths_bodyid = number_input(
                        "Max Paths (BodyId)", get_user_default("max_paths_bodyid"), 0, 100000000,
                        hint="Path budget for StrongestFirst enumeration: when the search "
                             "exceeds it, ALL paths above the achieved strength cutoff "
                             "(tau) are kept and tau is reported in the run notes. "
                             "0 = auto (StrongestFirst: 1M budget).",
                    )
                    edge_budget = number_input(
                        "Edge Budget", get_user_default("graph_edge_limit_bodyid"), 0, 100000000,
                        hint="Graph filter ('all' path mode): after the lossless prunes, "
                             "discovery cones exceeding this many bodyId edges are "
                             "floored just above the N-th strongest edge's weight "
                             "(w0 = w1 + 1) — exactly equivalent to raising the "
                             "threshold; the applied floor is reported as "
                             "edge_weight_floor. Distinct from the drawing-only "
                             "Visualization Edge Limit. 0 = off. Shortest mode "
                             "never floors.",
                    )
                    top_edges = number_input(
                        "Top Edges in Analysis Reports", 500, 10, 5000,
                        hint="Limits top-edge comparison/overlap results and edge/path "
                             "presence-matrix rows and the path-presence data used by "
                             "comparison summary plots. It does not trim the pathfinding "
                             "graph or set the per-visualization drawn-edge cap: see "
                             "Max Paths (BodyId) for the path budget and Visualization "
                             "Edge Limit for plotted edges.",
                    )
                search_columns = select_input(
                    "Search Columns", SEARCH_COLUMNS, get_user_default("search_columns"),
                    hint="Which columns to search when resolving neuron names in every dataset. "
                         "'auto': all columns (bodyId -> type -> instance -> flywireType/others). "
                         "Use 'type'/'instance'/'bodyId' to restrict the search.",
                )
                with ui.row().classes("gap-4"):
                    skip_bodyid = checkbox_input("Skip BodyId Level", get_user_default("skip_bodyId"), hint="Skip bodyId-level results for speed.")
                    cache_only = checkbox_input("Cache Only (Offline)", get_user_default("cache_only"), hint="Use only local cache, no server connection.")
                    auto_type_mapping = checkbox_input(
                        "Auto Type Mapping", get_user_default("auto_type_mapping"),
                        hint="Auto-map type names across datasets via the male-cns v1.0 "
                             "neuron info (e.g. FAFB MTe07 <-> male-cns MeVPLo2) when no "
                             "Custom Mapping is defined.",
                    )
                # Feature F: single enumeration + per-threshold replay.
                replay_paths = checkbox_input(
                    "Replay Paths (single enumeration)", get_user_default("replay_paths"),
                    hint="Path mode 'all': enumerate ONCE at the lowest threshold and "
                         "materialize every higher threshold from the bottleneck-annotated "
                         "path set — identical outputs, no re-enumeration. Disable to "
                         "force legacy per-threshold enumeration. Shortest mode is never "
                         "replayed (min-hop sets are not nested across thresholds).",
                )
                auto_extend_thresholds = checkbox_input(
                    "Auto-extend Collapsed Thresholds", False,
                    hint="F7: when a run's effective tau collapses asked thresholds, extend "
                         "each dataset with k × τ_ref points (τ_ref = max per-dataset tau) "
                         "while ≤ 2× the max asked threshold — the schedule is global, so "
                         "the expanded points stay shared across datasets. Default off; "
                         "suggested by the banner on collapse.",
                )
                drop_untyped = checkbox_input(
                    "Drop Untyped Neurons", get_user_default("drop_untyped"),
                    hint="Remove edges touching untyped neurons — shared "
                         "predicate with the pathfinding tabs (empty / "
                         "Unknown / NaN / bodyId-fallback labels). They can "
                         "never match across datasets. Applied AFTER the "
                         "standardized cross-dataset labels are resolved. "
                         "Dropped rows: comparison_results/"
                         "untyped_dropped_records.csv (the delegated "
                         "per-dataset pathfinding folders intentionally keep "
                         "untyped rows in their data_details/ outputs so "
                         "filtering happens once after standardization); "
                         "counts appended to "
                         "user_warning_notes.txt.",
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
                    min_prob = number_input(
                        "Min Traversal Prob.", 0, 0, 1, 0.01,
                        hint="Disabled: traversal_probability is a readout column "
                             "(ratio/0.3, capped at 1.0) — it no longer filters.",
                    ).set_visibility(False)
                    output_format = select_input(
                        "Output Format", ["csv", "xlsx"], get_user_default("output_format"),
                        hint="Format for exported data tables.",
                    )
                with param_grid(2):
                    parallel = checkbox_input(
                        "Parallel Processing", True,
                        hint="Run per-dataset work in parallel where possible.",
                    )
                    max_workers = number_input(
                        "Max Workers", 4, 1, 16,
                        hint="Number of parallel workers (only used when Parallel Processing is on).",
                    )
                with param_grid(2):
                    # Fix C: the lossy bodyId edge limit was removed — the
                    # StrongestFirst path budget above is the single knob.
                    edge_limit_viz = number_input(
                        "Visualization Edge Limit", get_user_default("edgeN_limit"), 10, 5000,
                        hint="Drawing-only cap: at most this many unique edges are "
                             "rendered per visualization (network / Sankey / heatmap) "
                             "in the delegated Complete/Shortest Paths runs. It never "
                             "changes fetching, the graph, or the path output; a single "
                             "complete path may still exceed it to stay intact. Type- and "
                             "bodyId-level visualizations share the same cap. Same "
                             "default as the Complete Paths tab.",
                    )

                def _apply_path_mode_defaults(notify=False):
                    """A mode switch resets the mode-specific defaults:
                    shortest -> Max Layers 8, Edge Budget 0 (off, never
                    floors); all -> Max Layers 2, Edge Budget 1M (deep
                    searches). The user is warned their values were reset."""
                    if path_mode.value == 'shortest':
                        max_interlayer.value = 8
                        edge_budget.value = 0
                        edge_budget.disable()
                    else:
                        max_interlayer.value = 2
                        edge_budget.enable()
                        edge_budget.value = get_user_default("graph_edge_limit_bodyid") or 1000000
                    if notify:
                        ui.notify(
                            f"Path Enumeration switched to '{path_mode.value}': "
                            "Max Layers and Edge Budget were reset to the "
                            "mode defaults — re-enter custom values if needed.",
                            type="warning",
                        )
                path_mode.on_value_change(lambda _e: _apply_path_mode_defaults(notify=True))
                _apply_path_mode_defaults()

    with results_col:
        output_panel.create(run_label="Run Comparison", run_icon="play_arrow")

    async def run_comparison():
        src_mode, src_neurons = source_input.get_value()
        tgt_mode, tgt_neurons = target_input.get_value()
        sources = apply_filter_mode(src_neurons, src_mode)
        targets = apply_filter_mode(tgt_neurons, tgt_mode)

        if not sources:
            ui.notify("Please provide at least one source neuron", type="warning")
            return

        datasets = datasets_select.value or []
        if not datasets:
            ui.notify("Please add at least 1 dataset to analyze", type="warning")
            return

        try:
            threshold_mode, thresholds, threshold_combinations = (
                _collect_threshold_configuration()
            )
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return

        # Resolve custom grouping (preset or inline); inline group labels are
        # compulsory for cross-dataset comparisons and validated here.
        mapping_path, mapping_ok = resolve_grouping()
        if not mapping_ok:
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "datasets": datasets,
            "source_neurons": sources,
            "target_neurons": targets,
            "output_folder": output_dir.value,
            "comparison_mode": comparison_mode.value,
            "path_mode": path_mode.value,
            "max_interlayer": int(max_interlayer.value),
            "thresholds": thresholds,
            "threshold_mode": threshold_mode,
            "threshold_dataset_order": list(datasets),
            "threshold_combinations": threshold_combinations,
            "density_normalizer": "per_node",
            "replay_paths": replay_paths.value,
            "auto_extend_thresholds": auto_extend_thresholds.value,
            "drop_untyped": drop_untyped.value,
            "top_edges": int(top_edges.value),
            # Fix D: the Edge Budget (lossy floor above the N-th strongest
            # edge, w0 = w1 + 1). 0 = off; shortest mode never floors.
            "graph_edge_limit_bodyid": int(edge_budget.value),
            "max_paths_bodyid": int(max_paths_bodyid.value) or None,
            "edgeN_limit": int(edge_limit_viz.value),
            # F1: StrongestFirst is the only 'all'-mode algorithm; the
            # selector was removed.
            "pathfinding": "StrongestFirst",
            "search_columns": search_columns.value,
            "skip_bodyId": skip_bodyid.value,
            "cache_only": cache_only.value,
            "auto_type_mapping": auto_type_mapping.value,
            # F9: ratio/probability filters are disabled — hidden UI,
            # metadata-only keys, sent 0.
            "_min_ratio": 0.0,
            "_min_prob": 0.0,
            "_output_format": output_format.value,
            "parallel": parallel.value,
            "max_workers": int(max_workers.value) if parallel.value else None,
            "separate_hemispheres": separate_hemi.value,
            "keep_only_hemisphere_conserved_connections": keep_hemi_conserved.value,
            "symmetry_analysis": symmetry_analysis.value,
            "find_reciprocal": find_reciprocal.value,
        }
        if mapping_path:
            constructor_params["overall_mapping_json"] = mapping_path

        result = await output_panel.run(runner, "inter_dataset", constructor_params, "run",
                                        output_dir=output_dir.value)

        # Persistent threshold/bottleneck provenance notice — the analyzer
        # writes effective_thresholds.json for every completed comparison
        # mode, including shortest and complete (non-collapsed) runs.
        try:
            run_folder = result.get("output_folder") or output_dir.value or ""
            notice_path = os.path.join(run_folder,
                                       "effective_thresholds.json")
            if os.path.isfile(notice_path):
                with open(notice_path, "r", encoding="utf-8") as nf:
                    notice = json.load(nf)
                banner = notice.get("banner")
                if banner:
                    output_panel.set_notice(banner)
                else:
                    output_panel.clear_notice()
            else:
                output_panel.clear_notice()
        except Exception as exc:
            # The banner is the only surface for threshold-collapse
            # provenance; dropping it silently hides exactly the confusing
            # outcome. Keep the UI stable but leave a trace.
            logging.debug("threshold provenance notice unavailable: %s", exc)

        # Each dataset-level path analysis initializes its neuron sets before
        # comparing them. Record only when at least one source/target pair was
        # resolved to real neurons in that process.
        match_info = result.get("neuron_match") or {}
        if match_info.get("any_pair"):
            from ..history_store import record as _record_history
            # Per-value dataset provenance: stamp only the selected datasets
            # whose local pools actually contain the value, so a male-cns-only
            # type never carries banc/flywire badges just because they were
            # selected for the same run. Unresolvable chips (regex patterns,
            # names matched through remote-only data) fall back to the whole
            # selection and keep the run-scope meaning.
            values = [str(v) for v in sources + targets]
            try:
                from ..type_suggestions import datasets_resolving
                value_datasets = datasets_resolving(
                    values, list(datasets_select.value or []))
            except Exception:
                value_datasets = {}
            _record_history(
                values,
                datasets=list(datasets_select.value or []),
                value_datasets=value_datasets,
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_comparison)
    output_panel.cancel_button.on_click(runner.cancel)
