"""NeuronBridge FindNeuron Tab - Find EM neurons matching LM driver lines."""

from nicegui import ui
from ..config import MATCH_ALGORITHMS, get_user_default
from ..components.common import (
    neuron_list_input, number_input, select_input, checkbox_input,
    dir_input, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..components.skeleton_visualization_settings import skeleton_visualization_settings
from ..runner import ScriptRunner


def create_nb_find_neuron_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("EM Neurons Output", state_key="nb_find_neuron")

    form_col, results_col = tool_page(
        "Find EM Neurons",
        "Find EM neurons matching GAL4 driver line morphology.",
        icon="search",
        tag="NeuronBridge",
        doc="nb_find_neuron.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Driver Line Query", "search")
            line_input = neuron_list_input(
                label="Driver Line Names",
                unit_label="line",
                show_filter=False,
                show_upload=False,
                history_kind="line",
                hint="Type a driver line name and press Enter (or leave the field) to add it as a chip.",
            )
            output_dir = dir_input(scope="nb_find_neuron")

        with ui.card().classes("w-full drocat-card"):
            section_header("Search Parameters", "tune")
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
                visualize = checkbox_input(
                    "Visualize Top Neurons",
                    True,
                    hint="Generate optional 3D skeleton visualizations of matched neurons.",
                )
                visualization_settings = skeleton_visualization_settings(
                    default_top_n=10,
                    top_n_label="Visualize Top N",
                    top_n_hint="Number of top types or bodyIds to render in 3D.",
                    default_visualize_by="type",
                    show_high_quality_warning=True,
                    default_show_fig=False,
                    default_export_views=True,
                )

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-nb-findneuron-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                with param_grid(2):
                    sort_by = select_input(
                        "Sort By", ["max_score", "type_avg_score"], "max_score",
                        hint="Sorting key for matched neuron results.",
                    )
                    pdf_cols = number_input("Profile Images Per Page (cols)", 4, 1, 6)
                    pdf_rows = number_input("Profile Images Per Page (rows)", 3, 1, 6)
                    background_color = select_input(
                        "Profile Background", ["white", "black"], "white",
                        hint="Background color for individual profile PDFs.",
                    )
                    min_score = number_input(
                        "Score Cutoff",
                        get_user_default("nb_min_score"),
                        0,
                        200000,
                        1000,
                        hint=(
                            "Threshold for score-based views. Top-N result files "
                            "still retain all ranked matches, including lower scores."
                        ),
                    )
                generate_pdf = checkbox_input("PDF Summary", True, hint="Generate PDF/PPTX with individual neuron profiles.")

    with results_col:
        output_panel.create(run_label="Find EM Neurons", run_icon="play_arrow")

    async def run_find_neuron():
        lines = line_input.get_value()[1]
        if not lines:
            ui.notify("Please enter at least one driver line", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)
        visualization_values = visualization_settings.values()
        if visualize.value:
            visualization_settings.warn_empty_custom_palettes()

        constructor_params = {"verbose": True}

        method_params = {
            "line_names": lines,
            "output_dir": output_dir.value,
            "match_type": match_algo.value,
            "top_n": int(top_n.value),
            "min_score": float(min_score.value),
            "visualize_top_n": (
                visualization_values["visualize_top_n"]
                if visualize.value else 0
            ),
            "generate_individual_profiles": ['pdf'] if generate_pdf.value else None,
            "visualize_by": visualization_values["visualize_by"],
            "visualization_settings": visualization_values,
            "sort_by": sort_by.value,
            "pdf_images_per_page": (int(pdf_cols.value), int(pdf_rows.value)),
            "background_color": background_color.value,
        }

        result = await output_panel.run(runner, "nb_find_neuron", constructor_params, "find_neurons",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        # A completed search means the driver lines matched EM neurons; keep
        # them in the query history without dataset provenance (NeuronBridge
        # searches every dataset). Driver-line history is intentionally kept
        # separate from the neuron-query history.
        if result["returncode"] == 0:
            from ..line_history_store import record as _record_history
            _record_history([str(v) for v in lines])

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_find_neuron)
    output_panel.cancel_button.on_click(runner.cancel)
