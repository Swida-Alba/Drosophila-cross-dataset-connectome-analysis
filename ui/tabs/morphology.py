"""Morphology Tab - Morphological similarity search and comparison.

Two sub-tabs share this page:
- Find Similar: query-vs-all morphological similarity search (intra-dataset).
- Comparison: N×N morphology comparison of already-identified neurons
  (intra-dataset only; vector_v2 or NBLAST scoring).
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
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, section_header, param_grid, tool_page,
    apply_filter_mode,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions
from ..dataset_service import is_banc_dataset

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
    comparison_dataset = None

    def _morph_suggest(text):
        dataset_name = dataset.value if dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    def _comparison_suggest(text):
        dataset_name = comparison_dataset.value if comparison_dataset is not None else ""
        return dataset_suggestions(text, dataset_name, limit=None)

    form_col, results_col = tool_page(
        "Morphology",
        "Find morphologically similar neurons within a dataset.",
        icon="science",
        doc="morphology.md",
    )

    with form_col:
        # Sub-tab switch: Find Similar vs Comparison
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

            with ui.card().classes("w-full drocat-card"):
                section_header("Skeleton Vector Cache", "memory")
                coverage_label = ui.label("Coverage: checking...").classes(
                    "text-caption drocat-muted"
                )
                with ui.row().classes("items-center gap-2"):
                    build_button = ui.button(
                        "Build Vector Cache", icon="auto_awesome"
                    ).props("color=secondary outline").on_click(
                        lambda: build_cache()
                    )
                    ui.label(
                        "One-time build from cached skeletons (auto-triggered on "
                        "first query); incremental afterwards."
                    ).classes("text-caption drocat-muted")

                ui.label(
                    "Raw skeletons fetched by Find Similar, visualization, and "
                    "dataset pulls are always stored as reusable .swc.zst files "
                    "under cache/<dataset>/skeletons/raw_skeletons/ (legacy "
                    ".swc.gz remains readable). Use "
                    "Settings → Dataset Cache → Download All Skeletons "
                    "to prefetch "
                    "the shared population."
                ).classes("text-caption drocat-muted")

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

            def refresh_coverage():
                # Lightweight: no navis/statvis import at page build (they are
                # heavy and would slow the first page response). The vector
                # cache is a plain parquet file, countable with polars.
                try:
                    from pathlib import Path
                    dataset_folder = (Path(PROJECT_ROOT) / "cache"
                                      / dataset.value.replace(":", "_").replace(".", "_"))
                    vector_folder = dataset_folder / "find_similar"
                    # recursive: raw-cache downloads may be grouped in nested
                    # folders by a dataset-specific source.
                    raw_dir = dataset_folder / "skeletons" / "raw_skeletons"
                    raw_files = (list(raw_dir.rglob("*.pkl"))
                                 + list(raw_dir.rglob("*.swc.gz"))
                                 + list(raw_dir.rglob("*.swc.zst")))
                    legacy_raw_dir = vector_folder / "raw_skeletons"
                    raw_files += (list(legacy_raw_dir.rglob("*.pkl"))
                                  + list(legacy_raw_dir.rglob("*.swc.gz"))
                                  + list(legacy_raw_dir.rglob("*.swc.zst")))
                    n_skel = len({
                        p.name.removesuffix(".swc.zst").removesuffix(".swc.gz")
                        .removesuffix(".pkl")
                        for p in raw_files
                    })
                    # FAFB v783: the healed zip is the real skeleton source
                    # (served directly; a legacy .zst is opened read-only
                    # when no zip exists; the pickle cache holds meshes).
                    dataset_folder_name = dataset.value.replace(":", "_").replace(".", "_")
                    dataset_dir = Path(PROJECT_ROOT) / "datasets" / dataset_folder_name
                    bundle_path = dataset_dir / "sk_lod1_783_healed.zst"
                    zip_path = dataset_dir / "sk_lod1_783_healed.zip"
                    if bundle_path.exists() or zip_path.exists():
                        try:
                            import sys as _sys
                            if str(SRC_DIR) not in _sys.path:
                                _sys.path.insert(0, str(SRC_DIR))
                            from fafb_bundle import FAFBSkeletonBundle
                            reader = FAFBSkeletonBundle(
                                bundle_path if bundle_path.exists() else None,
                                zip_path=zip_path if zip_path.exists() else None,
                                lazy_convert=False)
                            try:
                                n_skel = reader.count()
                            finally:
                                reader.close()
                        except Exception:
                            pass
                    n_vec = 0
                    vec_file = (vector_folder / "morphology"
                                / "skeleton__vectors_v2.parquet")
                    if vec_file.exists():
                        import polars as pl
                        n_vec = pl.read_parquet(vec_file).height
                    coverage_label.text = (
                        f"Dataset skeletons: {n_skel}  ·  vectorized: {n_vec}"
                    )
                except Exception:
                    coverage_label.text = "Coverage unavailable."

            async def build_cache():
                if is_banc_dataset(dataset.value):
                    morph_dataset_warning.set_visibility(True)
                    ui.notify(
                        "BANC morphological similarity is unavailable; select a non-BANC dataset.",
                        type="warning",
                    )
                    return
                build_button.disable()
                ui.notify(
                    "Building skeleton vector cache (this can take a few minutes)...",
                    type="info",
                )
                try:
                    import asyncio
                    import sys
                    sys.path.insert(0, str(SRC_DIR))
                    from morphology import find_similar_dataset_cache_v2

                    def _run():
                        cache = find_similar_dataset_cache_v2(
                            dataset.value, n_workers=8, verbose=False
                        )
                        return cache.build(fetch_missing=0)

                    stats = await asyncio.to_thread(_run)
                    ui.notify(
                        f"Vector cache ready: {stats['rows']} rows "
                        f"({stats['new']} new)"
                    )
                    refresh_coverage()
                except Exception as ex:
                    ui.notify(f"Cache build failed: {ex}", type="negative")
                finally:
                    build_button.enable()

        # ================= Comparison panel (profile comparison) =================
        with ui.column().classes("w-full gap-1") as comparison_panel:
            with ui.row().classes("w-full items-center justify-end px-2"):
                ui.link(
                    "Instructions",
                    "docs/ui_guides/morphology_comparison.html",
                ).classes("drocat-doc-link")
            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-comparison-dataset"'
            ):
                section_header("Dataset", "storage")
                comparison_dataset = dataset_selector(
                    disable_banc=True,
                    hint="Dataset whose neurons are compared. Morphological "
                         "comparison is intra-dataset only.",
                )
                comparison_output_dir = dir_input(scope="morphology_comparison")
                comparison_banc_warning = ui.label(
                    "⚠️ BANC morphological comparison is deferred: public "
                    "L2/full skeletons still need vector-quality validation. "
                    "3D skeleton visualization for BANC is available."
                ).classes("text-caption text-amber-8").set_visibility(False)

            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-comparison-neurons"'
            ):
                section_header("Query Neurons", "search")
                comparison_query_input = neuron_list_input(
                    label="Neurons to Compare",
                    placeholder="Type or upload CSV/TSV/Excel (e.g., aMe12, aMe10, aMe9)",
                    hint="Enter 2+ neuron types, bodyIds, or patterns "
                         "(e.g. aMe.*). Each type is one matrix row; its "
                         "members supply the pairwise scores.",
                    suggestions=_comparison_suggest,
                    available_neurons=lambda: comparison_dataset.value
                    if comparison_dataset is not None else "",
                ).classes("drocat-fixed-neuron-input")

            with ui.card().classes("w-full drocat-card"):
                section_header("Comparison Parameters", "tune")
                with param_grid(2):
                    comparison_method = select_input(
                        "Method", MORPH_METHODS, "vector_v2",
                        hint="'Vector (spatial)' (default): the Find Similar "
                             "vector_v2 score on whitened vectors — fast, "
                             "whole-population whitening comes from the "
                             "dataset cache. 'NBLAST': canonical normalized "
                             "NBLAST on raw-skeleton dotprops; capped at 30 "
                             "total neurons.",
                    )
                    comparison_max_members = number_input(
                        "Max Members per Type", 25, 1, 200,
                        hint="Members sampled per type for the pairwise "
                             "scores (large types are truncated; the member "
                             "list is written to members.csv).",
                    )

            # --- Advanced Settings (kept at the bottom, in its own card) ---
            with ui.card().classes("w-full drocat-card").props(
                'id="card-morphology-advanced"'
            ):
                with ui.expansion(
                    "Advanced Settings", icon="settings_suggest",
                ).classes("w-full drocat-section-expansion"):
                    comparison_fetch = checkbox_input(
                        "Fetch Missing Skeletons Online", True,
                        hint="Pull skeletons for neurons missing from the "
                             "vector cache through the API (NeuPrint raw "
                             "SWC; FAFB healed bundle → CAVE fallback) and "
                             "persist them into the shared cache. Turn off "
                             "for a strictly offline comparison.",
                    )
                    comparison_max_total = number_input(
                        "Max Total Neurons", 200, 2, 2000,
                        hint="Safety cap on the vectorized population "
                             "(NBLAST is always limited to 30 total).",
                    )
                    with ui.row().classes("gap-4"):
                        comparison_heatmaps = checkbox_input(
                            "Generate Heatmaps", True,
                            hint="Create interactive (VisPath) heatmaps for "
                                 "both levels.",
                        )
                        comparison_show_figures = checkbox_input(
                            "Show Figures", False,
                            hint="Open generated heatmaps in the browser.",
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
            refresh_coverage()
            refresh_roi_options()

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
        if is_banc_dataset(comparison_dataset.value):
            comparison_banc_warning.set_visibility(True)
            ui.notify(
                "BANC morphological comparison is unavailable; select a non-BANC dataset.",
                type="warning",
            )
            return
        mode, neurons = comparison_query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if len(query) < 2:
            ui.notify(
                "Please enter at least two neurons to compare",
                type="warning",
            )
            return

        comparison_output.clear()
        comparison_output.set_running(True)
        constructor_params = {
            "dataset": comparison_dataset.value,
            "query": query,
            "method": comparison_method.value,
            "max_members_per_type": int(comparison_max_members.value),
            "max_total_neurons": int(comparison_max_total.value),
            "fetch_online": comparison_fetch.value,
            "output_dir": comparison_output_dir.value,
            "saveas": "",
            "generate_heatmaps": comparison_heatmaps.value,
            "show_figures": comparison_show_figures.value,
            "verbose": True,
            "n_workers": 8,
            "use_cache": get_user_default("use_cache"),
        }
        try:
            result = await comparison_output.run(
                comparison_runner, "morphology_comparison", constructor_params,
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
                    datasets=[comparison_dataset.value]
                    if comparison_dataset.value else [],
                )
            files = result.get("files", [])
            if files:
                comparison_output.show_files(
                    list(files),
                    result.get("output_folder") or comparison_output_dir.value,
                )
        finally:
            comparison_output.set_running(False)

    comparison_output.run_button.on_click(run_comparison)
    comparison_output.cancel_button.on_click(comparison_runner.cancel)

    def _on_comparison_dataset_change(_e=None):
        comparison_banc_warning.set_visibility(
            is_banc_dataset(comparison_dataset.value))

    comparison_dataset.on_value_change(_on_comparison_dataset_change)

    sync_mode()
    on_dataset_change()
    _on_comparison_dataset_change()
