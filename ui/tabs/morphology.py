"""Morphology Tab - Morphological similarity search and comparison.

Two sub-tabs share this page (mirroring the Connectivity tab's structure):
- Find Similar: query-vs-all morphological similarity search (intra-dataset).
- Comparison: N×N comparison of queried neurons. One selected dataset runs
  the intra-dataset comparison (vector_v2 or NBLAST scoring); two or more
  datasets run the cross-dataset comparison (FAFB / male-cns / BANC,
  vector_v2 only, transform-based with null baselines).
"""

from nicegui import ui

from ..config import (
    CANDIDATE_SOURCE_OPTIONS,
    DEFAULTS,
    MORPH_LEVEL_OPTIONS,
    MORPH_METHOD_OPTIONS,
    PROJECT_ROOT,
    SRC_DIR,
    get_user_default,
)
from ..components.common import (
    dataset_selector, dataset_multi_selector, neuron_list_input,
    number_input, select_input, checkbox_input, dir_input, section_header,
    param_grid, tool_page, apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions, datasets_suggestions
from ..dataset_service import is_banc_dataset


def _cross_allowed_datasets():
    """Datasets eligible for cross-dataset morphology (FAFB/male-cns v1.0/BANC).

    Mirrors the backend's dataset_scope families: every other dataset has
    no bridging transform within the 2-hop accuracy guard, and male-cns
    v0.9 / optic-lobe lack vector_v2 population artifacts — all refused
    with an explicit banner.
    """
    try:
        from ..dataset_service import get_dataset_service
        names = get_dataset_service().get_all_datasets()
    except Exception:
        # Never leave the allow-list empty on a service hiccup: the backend
        # still validates every pair, so an empty filter would wrongly
        # banner the production FAFB <-> male-cns pair.
        return ["male-cns:v1.0", "flywire_FAFB_v783", "banc_v888",
                "banc_v626"]
    if not names:
        return ["male-cns:v1.0", "flywire_FAFB_v783", "banc_v888",
                "banc_v626"]
    allowed = []
    for name in names:
        low = str(name).lower()
        if "male-cns" in low or "malecns" in low:
            if "v0.9" in low or "optic" in low:
                continue
            allowed.append(name)
            continue
        if "banc" in low:
            allowed.append(name)
            continue
        if "fafb" in low or ("flywire" in low and "banc" not in low):
            allowed.append(name)
    return allowed

# Option lists live centrally in ui/config; labels for the method select
# stay local because the backend only knows the raw keys.
_MORPH_METHOD_LABELS = {
    "vector_v2": "Vector (spatial)",
    "nblast": "NBLAST",
}
MORPH_METHODS = {
    method: _MORPH_METHOD_LABELS.get(method, method)
    for method in MORPH_METHOD_OPTIONS
}


def create_morphology_tab():
    # One runner per output panel: a shared runner would let a second run
    # clobber the first run's process handle (cancel would kill the wrong
    # process and orphan the other).
    similar_runner = ScriptRunner()
    comparison_runner = ScriptRunner()
    output_panel = OutputPanel("Morphology Output", state_key="morphology_similar")
    comparison_output = OutputPanel(
        "Comparison Output", state_key="morphology_comparison"
    )
    dataset = None
    comparison_datasets = None

    def _morph_suggest(text):
        dataset_name = dataset.value if dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    def _comparison_suggest(text):
        selected = (list(comparison_datasets.value or [])
                    if comparison_datasets is not None else [])
        if len(selected) == 1:
            return dataset_suggestions(text, selected[0], limit=None)
        return datasets_suggestions(text, selected, limit=None)

    form_col, results_col = tool_page(
        "Morphology",
        "Find morphologically similar neurons within a dataset.",
        icon="science",
        doc="morphology.md",
    )

    with form_col:
        # Sub-tab switch: Find Similar vs Comparison (the Comparison
        # sub-tab covers intra- AND cross-dataset comparison, dispatched
        # by the number of selected datasets — same pattern as the
        # Connectivity tab's Comparison sub-tab).
        mode_value = {"value": "Find Similar"}
        with ui.row().classes(
            "w-full items-center justify-between gap-8 px-2"
        ):
            find_mode_button = ui.button("Find Similar").props(
                "outline no-caps"
            ).classes("w-5/12")
            comparison_mode_button = ui.button("Comparison").props(
                "outline no-caps"
            ).classes("w-5/12")
            for button in (find_mode_button, comparison_mode_button):
                button.style(
                    "min-height: 3.5rem; font-size: 1.1rem; "
                    "font-weight: 700;"
                )

        # ================= Find Similar panel (morphology search) =================
        with ui.column().classes("w-full gap-1") as find_panel:
            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-findsimilar-dataset"'
            ):
                section_header("Dataset", "storage")
                dataset = dataset_selector(
                    disable_banc=True,
                    hint="Dataset to search for similar neurons in.",
                )
                morph_output_dir = dir_input(scope="find_similar_morphology")
                morph_dataset_warning = ui.label(
                    "⚠️ BANC morphological similarity is deferred: public "
                    "L2/full skeletons still need vector-quality validation. "
                    "3D skeleton visualization for BANC is available."
                ).classes("text-caption text-amber-8").set_visibility(False)
                morph_best_dataset_warning = ui.label(
                    "⚠️ Morphological similarity works best with the "
                    "male-cns:v1.0 dataset; results on other datasets may be "
                    "less reliable, since male-cns:v1.0 includes ROI data "
                    "that is used for efficient candidate screening."
                ).classes("text-caption text-amber-8").set_visibility(False)

            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-findsimilar-neurons"'
            ):
                section_header("Query", "search")
                query_input = neuron_list_input(
                    label="Query Neuron(s)",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, 1005174948)",
                    hint="Neuron types, bodyIds, or patterns. Multiple queries "
                         "are searched independently and saved as separate runs.",
                    suggestions=_morph_suggest,
                    available_neurons=lambda: dataset.value
                    if dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Similarity Parameters", "tune")
                with param_grid(3):
                    level = select_input(
                        "Level", MORPH_LEVEL_OPTIONS, get_user_default("morph_level"),
                        hint="'auto' (recommended): a type query returns "
                             "type-to-type results, a bodyId query returns "
                             "bodyId-to-bodyId results. 'bodyid': rank "
                             "individual neurons. 'type': aggregate candidates "
                             "by neuron type.",
                    )
                    # Legacy saved default ("vector") maps to the current method.
                    saved_method = get_user_default("morph_method")
                    if saved_method not in MORPH_METHODS:
                        saved_method = "vector_v2"
                    method = select_input(
                        "Method", MORPH_METHODS, saved_method,
                        hint="'Vector (spatial)' (default): shape + brain-position/"
                             "expansion blocks with ZCA whitening — finer "
                             "discrimination. 'NBLAST': canonical NBLAST "
                             "(slower; runs on vector-prefiltered candidates).",
                    )
                    # The legacy cosine/Pearson metric selector was removed:
                    # vector_v2 scoring is per-block whitened cosine, and
                    # NBLAST carries its own normalized score.
                with param_grid(2):
                    candidate_source = select_input(
                        "Candidate Source", CANDIDATE_SOURCE_OPTIONS,
                        get_user_default("candidate_source"),
                        hint="'auto' (recommended): NeuPrint datasets screen "
                             "candidates by primary-ROI distribution "
                             "similarity (every neuron reachable; a one-time "
                             "matrix is cached), FlyWire searches the vector "
                             "cache directly. 'roi': ROI-distribution screen "
                             "only. 'combined': union of the ROI screen and "
                             "the connectivity (shared-partner) screen — "
                             "widest pool, most skeleton fetches. 'profile': "
                             "connectivity screen only (misses neurons "
                             "without shared partners). 'cache': full-"
                             "morphology over the local skeleton population "
                             "(download skeletons first).",
                    )
                    candidate_cap = number_input(
                        "Candidate Cap", get_user_default("candidate_cap"), 10, 5000,
                        hint="Maximum number of candidates entering the "
                             "morphological comparison: the sorted candidate "
                             "list is truncated to this many neurons (all "
                             "source modes; also the NBLAST prefilter in "
                             "cache mode). ALL compared candidates are "
                             "returned and written; the visualize Top N "
                             "controls rendering only.",
                    )
                roi_filter = select_input(
                    "ROI Filter", ["All ROIs"], "All ROIs",
                    hint="Restrict candidate discovery to synapse rows in "
                         "the selected ROIs (only shown when the dataset's "
                         "connection cache carries ROI data). 'All ROIs' = "
                         "no restriction.",
                )
                with ui.row().classes("w-full items-center gap-4"):
                    visualize = checkbox_input(
                        "Visualize Top Results",
                        DEFAULTS["morph_visualize_top_n"] > 0,
                        hint="Render optional 3D skeletons for the highest-ranked results.",
                    )
                    visualization_settings = skeleton_visualization_settings(
                        default_top_n=DEFAULTS["morph_visualize_top_n"],
                        top_n_label="Visualize Top N Types / Neurons",
                        top_n_hint=(
                            "Number of top results to render. The grouping choice "
                            "controls whether types or individual bodyIds are shown."
                        ),
                        default_visualize_by=DEFAULTS["morph_visualize_by"],
                        show_high_quality_warning=True,
                        dataset_provider=lambda: dataset.value,
                        dataset_watchers=[dataset],
                    )

            def refresh_roi_options():
                # ROI data availability differs per dataset (male-cns has 114
                # ROIs; hemibrain's connection cache has none).
                try:
                    from pathlib import Path
                    import polars as pl
                    conn_path = (Path(PROJECT_ROOT) / "cache"
                                 / dataset.value.replace(":", "_").replace(".", "_")
                                 / "connections.parquet")
                    rois = ["All ROIs"]
                    if conn_path.exists():
                        conn = pl.read_parquet(conn_path)
                        if "roi" in conn.columns:
                            vals = (conn["roi"].drop_nulls()
                                    .filter(pl.col("roi") != "")
                                    .unique().sort().to_list())
                            if vals:
                                rois = ["All ROIs"] + [str(v) for v in vals]
                    roi_filter.options = rois
                    if roi_filter.value not in rois:
                        roi_filter.value = "All ROIs"
                    roi_filter.set_visibility(len(rois) > 1)
                except Exception:
                    roi_filter.set_visibility(False)

        # ================= Comparison panel (profile comparison) =================
        with ui.column().classes("w-full gap-1") as comparison_panel:
            with ui.row().classes("w-full items-center justify-end gap-4 px-2"):
                ui.link(
                    "Instructions",
                    "docs/ui_guides/morphology_comparison.html",
                ).classes("drocat-doc-link")
                ui.link(
                    "Cross-Dataset Instructions",
                    "docs/ui_guides/cross_dataset_morphology.html",
                ).classes("drocat-doc-link")
            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-comparison-dataset"'
            ):
                section_header("Datasets", "storage")
                comparison_datasets = dataset_multi_selector(
                    label="Datasets to compare",
                    default=["male-cns:v1.0"],
                    hint="One dataset runs the intra-dataset N×N comparison "
                         "(vector_v2 or NBLAST). Two or more run the "
                         "cross-dataset comparison of the queried neurons "
                         "(FAFB / male-cns / BANC only — vector_v2 in each "
                         "target's render space, with null baselines; other "
                         "datasets raise a banner error).",
                )
                comparison_output_dir = dir_input(scope="morphology_comparison")
                comparison_banc_warning = ui.label(
                    "⚠️ BANC morphological comparison is deferred: public "
                    "L2/full skeletons still need vector-quality validation. "
                    "3D skeleton visualization for BANC is available."
                ).classes("text-caption text-amber-8").set_visibility(False)
                cross_banc_warning = ui.label(
                    "⚠️ BANC morphology is experimental: public skeleton "
                    "products mix L2/full/µm sources, so cross-dataset "
                    "scores involving BANC are less reliable. Population "
                    "artifacts are bootstrapped from cached skeletons on "
                    "first use."
                ).classes("text-caption text-amber-8").set_visibility(False)
                cross_rejected_warning = ui.label("").classes(
                    "text-caption text-red-8").set_visibility(False)

            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-comparison-neurons"'
            ):
                section_header("Query Neurons", "search")
                comparison_query_input = neuron_list_input(
                    label="Neurons to Compare",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10, aMe9)",
                    hint="Enter neuron types, bodyIds, or patterns "
                         "(e.g. aMe.*). Each type is one matrix row; its "
                         "members supply the pairwise scores. With two or "
                         "more datasets, types are resolved per dataset "
                         "(same-name/pattern, or via Auto Type Mapping) "
                         "and missing types show up as explicit empty "
                         "rows.",
                    suggestions=_comparison_suggest,
                    available_neurons=lambda: list(
                        comparison_datasets.value or [])
                    if comparison_datasets is not None else [],
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Comparison Parameters", "tune")
                with param_grid(2):
                    comparison_max_members = number_input(
                        "Max Members per Type", 25, 1, 200,
                        hint="Members sampled per type (and dataset) for "
                             "the pairwise scores (large types are "
                             "truncated; the member list is written to "
                             "members.csv).",
                    )
                # --- intra-dataset-only parameters (exactly one dataset) ---
                with ui.column().classes("w-full gap-1") as intra_params_box:
                    with param_grid(2):
                        comparison_method = select_input(
                            "Method", MORPH_METHODS, "vector_v2",
                            hint="'Vector (spatial)' (default): the Find "
                                 "Similar vector_v2 score on whitened "
                                 "vectors — fast, whole-population "
                                 "whitening comes from the dataset cache. "
                                 "'NBLAST': canonical normalized NBLAST on "
                                 "raw-skeleton dotprops; capped at 30 "
                                 "total neurons.",
                        )
                    with ui.row().classes("w-full items-center gap-4"):
                        comparison_visualize = checkbox_input(
                            "3D Skeleton Visualization", False,
                            hint="Render the compared neurons as one "
                                 "skeleton layer per compared type (line "
                                 "rendering by default; written to "
                                 "plot-3d_<dataset> inside the run folder "
                                 "and linked from report.html).",
                        )
                    comparison_visualization_settings = (
                        skeleton_visualization_settings(
                            include_ranking=False,
                            show_high_quality_warning=True,
                            dataset_provider=lambda: (
                                list(comparison_datasets.value or [None])[0]),
                            dataset_watchers=[comparison_datasets],
                        ))
                # --- cross-dataset-only parameters (two or more datasets) ---
                with ui.column().classes("w-full gap-1") as cross_params_box:
                    with param_grid(3):
                        cross_null_k = number_input(
                            "Null Sample Size", 200, 10, 1000,
                            hint="Seeded random target neurons per dataset "
                                 "pair defining the null baseline (p95). "
                                 "The sample is shared across queries and "
                                 "runs.",
                        )
                        cross_scene_members = number_input(
                            "Scene Members per Type", 3, 1, 10,
                            hint="Members rendered per type and dataset in "
                                 "the 3D overlay scenes.",
                        )
                    cross_reference = select_input(
                        "Reference Template (scenes)",
                        ["(first selected)"], "(first selected)",
                        hint="Template the 3D overlay scenes render in. "
                             "Scores always live in each target's render "
                             "space regardless of this choice (frame "
                             "disclosure in the report).",
                    )
                    cross_auto_mapping = checkbox_input(
                        "Auto Type Mapping", True,
                        hint="Resolve query type names across datasets via "
                             "the shared validity-aware mapper: curated "
                             "renames (e.g. APDN3 → SLP249) and valid "
                             "splits resolve, conflicts fail closed, "
                             "unmapped names fall back to the raw name. "
                             "Turn off for strict same-name/pattern "
                             "matching.",
                    )
                    with ui.row().classes("w-full items-center gap-4"):
                        cross_visualize = checkbox_input(
                            "3D Overlay Scenes", True,
                            hint="Render one overlay scene per run: every "
                                 "dataset's members bridged into the "
                                 "reference template (line rendering, "
                                 "layer legend).",
                        )
                        cross_fetch = checkbox_input(
                            "Fetch Missing Skeletons Online", True,
                            hint="Pull skeletons missing from the local "
                                 "cache through the dataset APIs and "
                                 "persist them. Turn off for a strictly "
                                 "offline comparison.",
                        )
                cross_params_box.set_visibility(False)

            # --- Advanced Settings (kept at the bottom, in its own card;
            #     intra-dataset only — hidden in cross-dataset mode) ---
            advanced_card = ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-advanced"'
            )
            with advanced_card:
                with ui.expansion(
                    "Advanced Settings", icon="settings_suggest",
                ).classes("w-full drocat-section-expansion"):
                        comparison_fetch = checkbox_input(
                            "Fetch Missing Skeletons Online", True,
                            hint="Pull skeletons for neurons missing from "
                                 "the vector cache through the API "
                                 "(NeuPrint raw SWC; FAFB healed bundle → "
                                 "CAVE fallback) and persist them into the "
                                 "shared cache. Turn off for a strictly "
                                 "offline comparison.",
                        )
                        comparison_max_total = number_input(
                            "Max Total Neurons", 200, 2, 2000,
                            hint="Safety cap on the vectorized population "
                                 "(NBLAST is always limited to 30 total).",
                        )
                        with ui.row().classes("gap-4"):
                            comparison_heatmaps = checkbox_input(
                                "Generate Heatmaps", True,
                                hint="Create interactive (VisPath) "
                                     "heatmaps for both levels.",
                            )
                            comparison_show_figures = checkbox_input(
                                "Show Figures", False,
                                hint="Open generated heatmaps in the "
                                     "browser.",
                            )

        def sync_mode():
            is_find = mode_value["value"] == "Find Similar"
            find_panel.set_visibility(is_find)
            comparison_panel.set_visibility(not is_find)
            find_output_container.set_visibility(is_find)
            comparison_output_container.set_visibility(not is_find)
            find_mode_button.props(
                "color=primary" if is_find else "color=grey-7"
            )
            comparison_mode_button.props(
                "color=grey-7" if is_find else "color=primary"
            )

        def set_mode(value: str):
            mode_value["value"] = value
            sync_mode()

        def on_dataset_change(_e=None):
            morph_dataset_warning.set_visibility(is_banc_dataset(dataset.value))
            morph_best_dataset_warning.set_visibility(
                str(dataset.value or "").strip().lower() != "male-cns:v1.0"
            )
            refresh_roi_options()

        def _on_comparison_datasets_change(_e=None):
            selected = list(comparison_datasets.value or [])
            intra = len(selected) <= 1
            intra_params_box.set_visibility(intra)
            advanced_card.set_visibility(intra)
            cross_params_box.set_visibility(not intra)
            options = ["(first selected)"] + selected
            cross_reference.options = options
            if cross_reference.value not in options:
                cross_reference.value = options[0]
            has_banc = any(is_banc_dataset(d) for d in selected)
            comparison_banc_warning.set_visibility(intra and has_banc)
            cross_banc_warning.set_visibility((not intra) and has_banc)
            rejected = ([] if intra else
                        [d for d in selected
                         if d not in _cross_allowed_datasets()])
            if rejected:
                cross_rejected_warning.set_text(
                    "⛔ Not supported for cross-dataset morphology "
                    "(no ≤2-hop bridging transform / no population "
                    "artifacts): " + ", ".join(str(d) for d in rejected))
                cross_rejected_warning.set_visibility(True)
            else:
                cross_rejected_warning.set_visibility(False)

        comparison_datasets.on_value_change(_on_comparison_datasets_change)

        find_mode_button.on_click(lambda _event: set_mode("Find Similar"))
        comparison_mode_button.on_click(lambda _event: set_mode("Comparison"))
        dataset.on_value_change(on_dataset_change)

    with results_col:
        with ui.column().classes("w-full gap-1") as find_output_container:
            output_panel.create(run_label="Find Similar Neurons", run_icon="play_arrow")
        with ui.column().classes("w-full gap-1") as comparison_output_container:
            comparison_output.create(run_label="Run Comparison", run_icon="play_arrow")

    def _unique_queries(values):
        """Return the entered queries in order, without duplicate chips."""
        queries = []
        seen = set()
        for value in values or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            queries.append(value)
        return queries

    def _collect_files(results):
        """Merge per-query output files while preserving their paths."""
        files = {}
        for result in results:
            for file_info in result.get("files", []):
                path = file_info.get("path")
                if path:
                    files[path] = file_info
        return list(files.values())

    async def run_find_similar():
        if is_banc_dataset(dataset.value):
            morph_dataset_warning.set_visibility(True)
            ui.notify(
                "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                type="warning",
            )
            return
        mode, neurons = query_input.get_value()
        raw_queries = _unique_queries(neurons)
        queries = _unique_queries(apply_filter_mode(neurons, mode))
        if not queries:
            ui.notify("Please enter at least one query neuron", type="warning")
            return
        if len(queries) > 50:
            ui.notify("Please limit the query to 50 neurons", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        visualization_values = visualization_settings.values()
        if visualize.value:
            visualization_settings.warn_empty_custom_palettes()
        base_params = {
            "dataset": dataset.value,
            "level": level.value,
            "method": method.value,
            "candidate_cap": int(candidate_cap.value),
            "candidate_source": candidate_source.value,
            "roi_filter": None if roi_filter.value == "All ROIs" else [roi_filter.value],
            "visualize_top_n": (
                visualization_values["visualize_top_n"] if visualize.value else 0
            ),
            "visualize_by": visualization_values["visualize_by"],
            "visualization_settings": visualization_values,
            "output_dir": morph_output_dir.value,
            "saveas": "",
            "verbose": True,
            "n_workers": 8,
            "use_cache": get_user_default("use_cache"),
            # Raw skeleton persistence is now unconditional and shared with
            # visualization and Settings cache pulls.
            "cache_fetched_skeletons": True,
        }
        results = []
        last_output_folder = None
        try:
            for index, query in enumerate(queries):
                if len(queries) > 1:
                    output_panel.log(
                        f"--- Morphology query {index + 1}/{len(queries)}: {query} ---",
                        "system",
                    )
                constructor_params = dict(base_params)
                constructor_params["query"] = query
                result = await output_panel.run(
                    similar_runner, "find_similar_morphology", constructor_params,
                    "find_similar", output_dir=morph_output_dir.value,
                )
                results.append(result)
                # A completed per-query run means the query resolved in the
                # dataset; keep the raw chip (pre-pattern) in the history.
                if result.get("returncode") == 0:
                    from ..history_store import record as _record_history
                    raw = raw_queries[index] if index < len(raw_queries) else query
                    _record_history(
                        [str(raw)],
                        datasets=[dataset.value] if dataset.value else [],
                    )
                last_output_folder = result.get("output_folder") or last_output_folder
                if result.get("cancelled"):
                    break

            cancelled = any(result.get("cancelled") for result in results)
            succeeded = bool(results) and all(
                result.get("returncode") == 0 for result in results
            )
            if cancelled:
                output_panel.set_status("Cancelled", "red")
            else:
                output_panel.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            files = _collect_files(results)
            if files:
                output_panel.show_files(
                    files,
                    morph_output_dir.value if len(queries) > 1
                    else last_output_folder or morph_output_dir.value,
                )
        finally:
            output_panel.set_running(False)

    output_panel.run_button.on_click(run_find_similar)
    output_panel.cancel_button.on_click(similar_runner.cancel)

    async def run_comparison():
        selected = list(comparison_datasets.value or [])
        if not selected:
            ui.notify("Please select at least one dataset", type="warning")
            return
        mode, neurons = comparison_query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if len(query) < 2 and len(selected) == 1:
            ui.notify(
                "Please enter at least two neurons to compare",
                type="warning",
            )
            return
        if not query:
            ui.notify("Please enter at least one query neuron",
                      type="warning")
            return

        cross = len(selected) > 1
        if cross:
            rejected = [d for d in selected
                        if d not in _cross_allowed_datasets()]
            if rejected:
                cross_rejected_warning.set_text(
                    "⛔ Not supported for cross-dataset morphology "
                    "(no ≤2-hop bridging transform / no population "
                    "artifacts): " + ", ".join(str(d) for d in rejected))
                cross_rejected_warning.set_visibility(True)
                ui.notify(
                    "Cross-dataset comparison refused: "
                    + ", ".join(str(d) for d in rejected),
                    type="warning",
                )
                return
        elif is_banc_dataset(selected[0]):
            comparison_banc_warning.set_visibility(True)
            ui.notify(
                "BANC morphological comparison is unavailable; select a "
                "non-BANC dataset.",
                type="warning",
            )
            return

        comparison_output.clear()
        comparison_output.set_running(True)
        try:
            if cross:
                reference = cross_reference.value
                if str(reference).startswith("("):
                    reference = None
                constructor_params = {
                    "datasets": selected,
                    "query": query,
                    "output_dir": comparison_output_dir.value,
                    "max_members_per_type": int(comparison_max_members.value),
                    "null_k": int(cross_null_k.value),
                    "reference_template": reference,
                    "scene_members_per_type": int(cross_scene_members.value),
                    "visualize": bool(cross_visualize.value),
                    "fetch_online": bool(cross_fetch.value),
                    "use_auto_type_mapping": bool(cross_auto_mapping.value),
                    "use_cache": True,
                    "verbose": True,
                }
                tool = "morph_cross_dataset"
            else:
                visualization_values = (
                    comparison_visualization_settings.values())
                viz_on = bool(comparison_visualize.value)
                if viz_on:
                    comparison_visualization_settings \
                        .warn_empty_custom_palettes()
                constructor_params = {
                    "dataset": selected[0],
                    "query": query,
                    "method": comparison_method.value,
                    "max_members_per_type": int(comparison_max_members.value),
                    "max_total_neurons": int(comparison_max_total.value),
                    "fetch_online": bool(comparison_fetch.value),
                    "output_dir": comparison_output_dir.value,
                    "saveas": "",
                    "generate_heatmaps": bool(comparison_heatmaps.value),
                    "show_figures": bool(comparison_show_figures.value),
                    "verbose": True,
                    "n_workers": 8,
                    "use_cache": get_user_default("use_cache"),
                    "visualize": viz_on,
                    "visualization_settings": (
                        visualization_values if viz_on else {}),
                }
                tool = "morphology_comparison"

            result = await comparison_output.run(
                comparison_runner, tool, constructor_params,
                "run", output_dir=comparison_output_dir.value,
            )
            succeeded = result.get("returncode") == 0
            if result.get("cancelled"):
                comparison_output.set_status("Cancelled", "red")
            else:
                comparison_output.set_status(
                    "Completed" if succeeded else "Failed",
                    "green" if succeeded else "red",
                )
            if succeeded:
                from ..history_store import record as _record_history
                _record_history(
                    [str(v) for v in query],
                    datasets=selected,
                )
            files = result.get("files", [])
            if files:
                comparison_output.show_files(
                    list(files),
                    result.get("output_folder")
                    or comparison_output_dir.value,
                )
        finally:
            comparison_output.set_running(False)

    comparison_output.run_button.on_click(run_comparison)
    comparison_output.cancel_button.on_click(comparison_runner.cancel)

    sync_mode()
    on_dataset_change()
    _on_comparison_datasets_change()
