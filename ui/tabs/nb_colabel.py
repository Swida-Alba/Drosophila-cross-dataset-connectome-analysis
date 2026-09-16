"""NeuronBridge Colabel Tab - Co-labeling analysis for driver lines."""

from nicegui import ui
from ..config import DATASETS, MATCH_ALGORITHMS, get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input, checkbox_input,
    dir_input, section_header, param_grid, tool_page, output_detail_control,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner


def create_nb_colabel_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Co-Labeling Output", state_key="nb_colabel")

    form_col, results_col = tool_page(
        "Co-Labeling Analysis",
        "Analyze co-labeling patterns between driver lines.",
        icon="layers",
        tag="NeuronBridge",
        doc="nb_colabel.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Driver Lines", "search")
            line_input = neuron_list_input(
                label="Driver Line Names",
                unit_label="line",
                show_filter=False,
                show_upload=False,
                history_kind="line",
                hint="Type a driver line name and press Enter (or leave the field) to add it as a chip.",
            )
            with param_grid(2):
                dataset = dataset_selector(
                    label="3D Dataset (optional)",
                    default=None,
                    datasets=["(all)"] + DATASETS,
                    hint=(
                        "Limits the optional 3D skeleton output. Co-labeling "
                        "statistics always use all datasets in NeuronBridge."
                    ),
                )
                output_dir = dir_input(scope="nb_colabel")

        with ui.card().classes("w-full drocat-card"):
            section_header("Analysis Settings", "tune")
            with ui.row().classes("gap-4"):
                method_jaccard = checkbox_input("Jaccard", True, hint="Binary Jaccard similarity of labeled types.")
                method_weighted = checkbox_input("Weighted Jaccard", True, hint="Score-weighted Jaccard similarity.")
            ui.label(
                "Line-specificity and sparsity metrics are included automatically."
            ).classes("text-caption drocat-muted")
            with param_grid(2):
                match_algo = select_input("Algorithm", MATCH_ALGORITHMS, get_user_default("match_algorithm"))
                top_n = number_input(
                    "Top N Matches Per Line",
                    get_user_default("nb_top_n"),
                    1,
                    2000,
                    hint=(
                        "Always retrieve this many highest-scoring matches per line; "
                        "the score cutoff does not reduce the top-N result list."
                    ),
                )

            with ui.row().classes("w-full items-center gap-4"):
                visualize_3d = checkbox_input(
                    "3D Skeleton",
                    False,
                    hint="Render optional 3D skeletons of top co-labeled types.",
                )
                visualization_settings = skeleton_visualization_settings(
                    default_top_n=5,
                    top_n_label="Visualize Top N Types",
                    top_n_hint="Number of top co-labeled types to visualize in 3D.",
                    default_visualize_by="type",
                    default_show_fig=False,
                    default_export_views=True,
                    dataset_provider=lambda: "" if dataset.value in (None, "(all)") else dataset.value,
                    dataset_watchers=[dataset],
                )
            _, detail_flags = output_detail_control(
                "Full keeps every exported file. Compact drops the row-level "
                "source data (line_labeled_neurons/ and "
                "distribution_data_by_neuron.csv) once the similarity "
                "matrices and report are written — every removal is audited "
                "in the run's cleanup_audit.json. The matrices, expression "
                "data, by-type distributions, and the HTML report always "
                "stay. The NeuronBridge match cache is off by default "
                "(Settings → NeuronBridge Match Cache), so removed tables "
                "regenerate only by re-running the analysis."
            )

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-nb-colabel-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                with ui.row().classes("gap-4"):
                    gen_heatmap = checkbox_input("Heatmaps", True, hint="Generate interactive heatmap visualizations.")
                    gen_report = checkbox_input("HTML Report", True, hint="Generate comprehensive HTML analysis report.")
                with param_grid(2):
                    min_score = number_input(
                        "Score Cutoff",
                        get_user_default("nb_min_score"),
                        0,
                        200000,
                        1000,
                        hint=(
                            "Filters expression-matrix and similarity calculations "
                            "after the top-N matches have been retrieved."
                        ),
                    )
                    min_type_avg_score = number_input(
                        "Min Type Avg Score", get_user_default("nb_min_type_avg_score"), 0, 200000, 1000,
                        hint="Additional filter for co-labeling similarity matrices; does not filter the expression matrix.",
                    )
                with param_grid(3):
                    sort_by = select_input(
                        "Sort By", ["max_score", "type_avg_score"], "max_score",
                        hint="Sorting key for expression matrices.",
                    )
                    background_color = select_input(
                        "Profile Background", ["white", "black"], "white",
                        hint="Background color for individual profile PDFs.",
                    )
                    pdf_cols = number_input("Profile Images Per Page (cols)", 3, 1, 6)
                    pdf_rows = number_input("Profile Images Per Page (rows)", 2, 1, 6)

    with results_col:
        output_panel.create(run_label="Run Co-Labeling", run_icon="play_arrow")

    async def run_colabel():
        lines = line_input.get_value()[1]
        if len(lines) < 2:
            ui.notify("Please enter at least two driver lines", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        ds = "all" if dataset.value in (None, "(all)") else dataset.value

        methods = []
        if method_jaccard.value: methods.append("jaccard")
        if method_weighted.value: methods.append("weighted_jaccard")
        if not methods:
            ui.notify("Select at least one similarity method", type="warning")
            output_panel.set_running(False)
            return

        visualization_values = visualization_settings.values()
        if visualize_3d.value:
            visualization_settings.warn_empty_custom_palettes()
        constructor_params = {
            "verbose": True,
            "use_cache": bool(get_user_default("nb_use_cache")),
        }

        method_params = {
            "lines": lines,
            "match_type": match_algo.value,
            "output_dir": output_dir.value,
            "similarity_methods": methods,
            "generate_report": gen_report.value,
            "visualize": gen_heatmap.value,
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if visualize_3d.value else 0
            ),
            "top_n_neurons": int(top_n.value),
            "min_score": float(min_score.value),
            "min_type_avg_score": float(min_type_avg_score.value),
            "sort_by": sort_by.value,
            "background_color": background_color.value,
            "pdf_images_per_page": (int(pdf_cols.value), int(pdf_rows.value)),
            "datasets_to_visualize": ds,
            "visualize_by": visualization_values["visualize_by"],
            "visualization_settings": visualization_values,
            "keep_per_match_csv": detail_flags()["keep_per_match_csv"],
        }

        result = await output_panel.run(runner, "nb_colabel", constructor_params, "colabel",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        # Co-labeling always searches every NeuronBridge dataset, so the
        # Driver lines use their own history list; co-labeling is not a neuron
        # query and must not pollute the neuron history.
        if result["returncode"] == 0:
            from ..line_history_store import record as _record_history
            _record_history([str(v) for v in lines])

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_colabel)
    output_panel.cancel_button.on_click(runner.cancel)
