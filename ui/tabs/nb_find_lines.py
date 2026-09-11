"""NeuronBridge FindLines Tab - Find LM driver lines matching EM neurons."""

from nicegui import ui
from ..config import DATASETS, MATCH_ALGORITHMS, get_user_default
from ..components.common import (
    dataset_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, multi_select_input,
    apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_suggestions, datasets_suggestions


def create_nb_find_lines_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Driver Lines Output", state_key="nb_find_lines")
    dataset = None

    def _type_suggest(text):
        selected = dataset.value if dataset is not None else "(all)"
        if selected == "(all)":
            return datasets_suggestions(text, DATASETS, limit=None)
        return dataset_suggestions(text, selected, limit=None)

    form_col, results_col = tool_page(
        "Find Driver Lines",
        "Find GAL4 / Split-GAL4 driver lines matching EM neurons.",
        icon="biotech",
        tag="NeuronBridge",
        doc="nb_find_lines.md",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Query Neurons", "search")
            query_input = neuron_list_input(
                label="EM Neurons (bodyId, type, or instance)",
                hint="Enter EM neuron identifiers. Use filter mode for pattern matching across types.",
                suggestions=_type_suggest,
                available_neurons=lambda: DATASETS if dataset is not None
                and dataset.value == "(all)" else (dataset.value if dataset is not None else ""),
            )
            with param_grid(2):
                dataset = dataset_selector(
                    label="Dataset (optional)", default=None, datasets=["(all)"] + DATASETS,
                    hint="Restrict search to one dataset, or select (all) to search everywhere.",
                )
                output_dir = dir_input(scope="nb_find_lines")

        with ui.card().classes("w-full drocat-card"):
            section_header("Search Parameters", "tune")
            with param_grid(2):
                match_algo = select_input(
                    "Algorithm", MATCH_ALGORITHMS, get_user_default("match_algorithm"),
                    hint="'cds': Color Depth Search (fast). 'pppm': Point Pattern (precise). 'both': run both.",
                )
                top_image_lines = number_input(
                    "Top Lines for Images",
                    30,
                    1,
                    100,
                    hint=(
                        "When image download is enabled, download only the top N "
                        "ranked lines (per category in separate mode). Search CSVs "
                        "still contain all matches."
                    ),
                )

        with ui.card().classes("w-full drocat-card").props('id="card-nb-image-download"'):
            section_header("Image Download", "image")
            with ui.row().classes("gap-4"):
                download_images = checkbox_input(
                    "Download Images",
                    False,
                    hint=(
                        "Download matched line images. FlyLight is searched first; "
                        "missing lines automatically fall back to NeuronBridge."
                    ),
                )
                download_flylight = checkbox_input(
                    "From FlyLight",
                    True,
                    hint=(
                        "Search FlyLight S3/CDN first (GAL4/LexA → Split-GAL4 → "
                        "MCFO → RawImages); missing lines still use NeuronBridge."
                    ),
                ).props('id=checkbox-flylight')
                generate_pdf = checkbox_input("PDF Summary", True, hint="Create a PDF with downloaded images ordered by score.")
                generate_pptx = checkbox_input("PPTX Summary", False, hint="Create a PowerPoint summary alongside the PDF.")
            with param_grid(2):
                image_formats = multi_select_input(
                    "Image Formats", ["png", "jpg"], ["png", "jpg"],
                    hint="File formats to download (neuronbridge: png/jpg; flylight adds h5j/mp4/json).",
                )
                image_types = multi_select_input(
                    "Image Types", ["cdm", "mip", "aligned", "translation", "metadata"], ["cdm", "mip"],
                    hint="Image types to download (cdm/mip for NeuronBridge).",
                )
            with param_grid(3):
                max_download_images = number_input(
                    "Max Images / Line", 12, 1, 100,
                    hint="Maximum number of images downloaded per driver line.",
                )
                flylight_category = multi_select_input(
                    "FlyLight Collections", ["GAL4/LEXA", "SplitGAL4", "MCFO", "RawImages", "All"],
                    ["GAL4/LEXA", "SplitGAL4"],
                    hint=(
                        "FlyLight collections searched in priority order; MCFO and "
                        "RawImages are automatic fallbacks. 'All' searches every "
                        "collection."
                    ),
                )
                simple_mode = checkbox_input(
                    "Simple Mode (fewer files)", True,
                    hint="Download only representative files (20x/multichannel for Split-GAL4, "
                         "total for GAL4/LexA).",
                )
            with param_grid(3):
                pdf_cols = number_input("PDF Images Per Page (cols)", 3, 1, 6)
                pdf_rows = number_input("PDF Images Per Page (rows)", 2, 1, 6)
                summary_background = select_input(
                    "Summary Background", ["black", "white"], "black",
                    hint="Background color for the PDF/PPTX image summary.",
                )
                organize_by_region = checkbox_input(
                    "Organize by Region", False,
                    hint="Group downloaded FlyLight images into Brain/VNC subfolders.",
                )

        # --- Advanced Settings (kept at the bottom, in its own card) ---
        with ui.card().classes("w-full drocat-card").props('id="card-nb-findlines-advanced"'):
            with ui.expansion(
                "Advanced Settings", icon="settings_suggest",
            ).classes("w-full drocat-section-expansion"):
                separate_split = checkbox_input(
                    "Separate Split-GAL4 Results", True,
                    hint="Generate separate summary CSVs for GAL4/LexA vs Split-GAL4 lines.",
                )
                with param_grid(3):
                    region = select_input(
                        "Region", ["Brain", "VNC", "All"], "Brain",
                        hint="Anatomical region filter for image downloads.",
                    )
                    max_workers = number_input(
                        "Max Workers", 8, 1, 32,
                        hint="Parallel workers for API searches (lower if rate-limited).",
                    )
                    sort_by = select_input(
                        "Sort By", ["max", "completeness"], "max",
                        hint="'max': score-weighted matches (score × coverage). "
                             "'completeness': best coverage of all queries.",
                    )

    with results_col:
        output_panel.create(run_label="Find Driver Lines", run_icon="play_arrow")

    async def run_find_lines():
        mode, neurons = query_input.get_value()
        query = apply_filter_mode(neurons, mode)
        if not query:
            ui.notify("Please enter at least one neuron", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        ds = None if dataset.value == "(all)" else dataset.value

        constructor_params = {
            "verbose": True,
            "separate_splitgal4": separate_split.value,
            "region": region.value,
            "max_workers": int(max_workers.value),
        }

        # Determine download_images source
        dl_source = None
        if download_images.value and download_flylight.value:
            dl_source = 'both'
        elif download_flylight.value:
            dl_source = 'flylight'
        elif download_images.value:
            dl_source = 'neuronbridge'

        # Summary formats: PDF and/or PPTX (the backend accepts a list)
        summary_formats = []
        if generate_pdf.value:
            summary_formats.append("pdf")
        if generate_pptx.value:
            summary_formats.append("pptx")

        method_params = {
            "queries": query,
            "dataset": ds,
            "output_dir": output_dir.value,
            "match_type": match_algo.value,
            "download_images": dl_source,
            "download_img_for_top_n_lines": int(top_image_lines.value) if dl_source else None,
            "summary_format": summary_formats or None,
            "sort_by": sort_by.value,
            "image_formats": image_formats.value or ["png"],
            "image_types": image_types.value or ["cdm"],
            "max_download_images_per_line": int(max_download_images.value) if dl_source else None,
            "flylight_category": flylight_category.value or None,
            "simple_mode": simple_mode.value,
            "organize_by_region": organize_by_region.value,
            "pdf_images_per_page": (int(pdf_cols.value), int(pdf_rows.value)),
            "summary_background_color": summary_background.value,
        }

        result = await output_panel.run(runner, "nb_find_lines", constructor_params, "find_lines",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        # A completed search means the queried EM identifiers resolved. A
        # single-dataset restriction records that dataset; '(all)' leaves the
        # entries unscoped so they appear in every dataset's history list.
        if result["returncode"] == 0:
            from ..history_store import record as _record_history
            _record_history(
                [str(v) for v in query],
                datasets=[ds] if ds else None,
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_find_lines)
    output_panel.cancel_button.on_click(runner.cancel)
