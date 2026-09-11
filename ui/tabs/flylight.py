"""FlyLight Download Tab - search, preview, and download FlyLight imagery."""

from nicegui import run, ui

from ..components.common import (
    checkbox_input, dir_input, multi_select_input, neuron_list_input, number_input,
    param_grid, section_header, select_input, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner

FORMAT_OPTIONS = ["png", "jpg", "h5j", "lsm", "mp4", "json"]
IMAGE_TYPE_OPTIONS = ["mip", "cdm", "aligned", "metadata", "translation"]
CATEGORY_OPTIONS = ["GAL4/LEXA", "SplitGAL4", "MCFO", "RawImages", "All"]
REGION_OPTIONS = ["Brain", "VNC", "All"]
DEFAULT_FLAT_STRUCTURE = True
DEFAULT_IMAGE_SUMMARY = "pdf"


def _parse_lines(value: str) -> list:
    """Split a comma-separated driver-line string into a clean list."""
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def _line_values(lines_input) -> list:
    """Return the committed driver-line chips from the shared list input."""
    return [
        str(name).strip()
        for name in lines_input.get_value()[1]
        if str(name).strip()
    ]


def _build_downloader(
    formats, image_types, region, categories, simple_mode,
    max_workers: int = 4, verbose=False,
):
    """Construct a FlyLightDownloader with the tab's current filters.

    Verbose is disabled for in-process preview calls (search / list) so
    progress prints do not spam the UI thread; the download run streams
    through the runner subprocess instead.
    """
    from flylight_downloader import FlyLightDownloader

    return FlyLightDownloader(
        formats=formats or ["png"],
        image_types=image_types or ["mip"],
        region=region or "All",
        collection_category=categories or None,
        simple_mode=simple_mode,
        max_workers=max_workers,
        verbose=verbose,
        use_boto3=True,
        include_vt_lines=True,
    )


def _fetch_lines(pattern, formats, image_types, region, categories, simple_mode) -> list:
    """io_bound target: search driver lines matching a pattern."""
    return _build_downloader(
        formats, image_types, region, categories, simple_mode
    ).search_lines(pattern)


def _fetch_file_preview(
    line_names, formats, image_types, region, categories, simple_mode,
    max_files_per_line=None,
) -> list:
    """io_bound target: list the files that match the current filters."""
    return _build_downloader(
        formats, image_types, region, categories, simple_mode
    ).get_filtered_files(line_names, max_files_per_line=max_files_per_line)


def create_flylight_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("FlyLight Output", state_key="flylight")

    form_col, results_col = tool_page(
        "FlyLight Downloader",
        "Search, preview, and download FlyLight driver-line imagery "
        "(Janelia S3 bucket + flimg.janelia.org CDN).",
        icon="download",
        tag="FlyLight",
        tag_color="green",
        doc="flylight_downloader.html",
    )

    with form_col:
        with ui.card().classes("w-full drocat-card"):
            section_header("Driver Lines", "route")
            lines_input = neuron_list_input(
                label="Driver line name(s)",
                unit_label="line",
                show_filter=False,
                show_upload=False,
                history_kind="line",
                hint=(
                    "Type a driver line name and press Enter or leave the field "
                    "to add it as a chip. R-lines (Gen1 GAL4) and SS-lines "
                    "(Split-GAL4) come from S3; VT lines use the HTTP CDN."
                ),
            )
            output_dir = dir_input(scope="flylight")

        with ui.card().classes("w-full drocat-card"):
            section_header("File Filters", "filter_alt")
            with param_grid(2):
                formats = multi_select_input(
                    "Formats", FORMAT_OPTIONS, ["png", "jpg"],
                    hint="File formats to download (VT lines primarily have jpg/mp4).",
                )
                image_types = multi_select_input(
                    "Image Types", IMAGE_TYPE_OPTIONS, ["mip", "cdm"],
                    hint="Image types: mip/cdm projections, aligned stacks, metadata, translation videos.",
                )
            with param_grid(2):
                region = select_input(
                    "Region", REGION_OPTIONS, "Brain",
                    hint="Anatomical region filter (Brain / VNC / All).",
                )
                categories = multi_select_input(
                    "Collections", CATEGORY_OPTIONS, ["GAL4/LEXA", "SplitGAL4"],
                    hint="Collection categories to search in priority order (MCFO is the fallback).",
                )

        with ui.card().classes("w-full drocat-card"):
            section_header("Download Options", "tune")
            with param_grid(3):
                simple_mode = checkbox_input(
                    "Simple Mode (fewer files)", True,
                    hint="Download only representative files (20x/multichannel for Split-GAL4, "
                         "total for GAL4/LexA) with a small fallback when nothing matches.",
                )
                max_files = number_input(
                    "Max Files / Line", 6, 1, 200,
                    hint="Maximum number of files downloaded per line (None = no limit).",
                )
                max_workers = number_input(
                    "Max Workers", 4, 1, 32,
                    hint="Parallel download threads.",
                )
            with param_grid(3):
                flat_structure = checkbox_input(
                    "Flat Folder Structure", DEFAULT_FLAT_STRUCTURE,
                    hint="Save files directly instead of preserving the S3 key structure.",
                )
                add_timestamp = checkbox_input(
                    "Timestamp Folder", True,
                    hint="Append a timestamp to the output folder name.",
                )
                summary = select_input(
                    "Image Summary", ["none", "pdf", "pptx", "pdf + pptx"], DEFAULT_IMAGE_SUMMARY,
                    hint="Generate a PDF/PPTX contact sheet of the downloaded images.",
                )
            with param_grid(2):
                summary_cols = number_input("Summary Columns", 3, 1, 6)
                summary_rows = number_input("Summary Rows", 2, 1, 6)

        with ui.card().classes("w-full drocat-card"):
            section_header("Preview", "preview")
            with ui.row().classes("items-center gap-2 w-full"):
                search_input = ui.input(
                    placeholder="Pattern, e.g. R10A0, VT037, SS01",
                ).classes("drocat-input").style("width: 320px")
                search_btn = ui.button("Search Lines", icon="search")
                list_btn = ui.button("List Files", icon="list_alt")
            ui.label(
                "Search finds matching driver lines (click a result to add it); "
                "List shows the files the current filters would download."
            ).classes("text-caption drocat-muted")
            search_results = ui.row().classes("gap-1 w-full flex-wrap")
            file_table = ui.table(
                columns=[
                    {"name": "line", "label": "Line", "field": "line", "align": "left"},
                    {"name": "file", "label": "File", "field": "file", "align": "left"},
                    {"name": "type", "label": "Type", "field": "type", "align": "left"},
                    {"name": "size", "label": "Size", "field": "size", "align": "right"},
                ],
                rows=[],
            ).classes("w-full")
            file_table.add_slot(
                "body",
                r"""
                <q-tr :props="props">
                  <q-td key="line" :props="props">{{ props.row.line }}</q-td>
                  <q-td key="file" :props="props">
                    <a :href="props.row.url" target="_blank" class="text-primary">{{ props.row.file }}</a>
                  </q-td>
                  <q-td key="type" :props="props">{{ props.row.type }}</q-td>
                  <q-td key="size" :props="props">{{ props.row.size }}</q-td>
                </q-tr>
                """,
            )

    with results_col:
        output_panel.create(run_label="Download Images", run_icon="download")

    def _add_line(name: str):
        lines_input.add_values([name])

    async def do_search():
        pattern = search_input.value.strip()
        if not pattern:
            ui.notify("Enter a search pattern first", type="warning")
            return
        search_btn.disable()
        search_btn.text = "Searching…"
        try:
            lines = await run.io_bound(
                _fetch_lines, pattern,
                formats.value, image_types.value, region.value,
                categories.value, simple_mode.value,
            )
            search_results.clear()
            with search_results:
                if not lines:
                    ui.label(f"No driver lines match '{pattern}'.").classes("text-caption drocat-muted")
                for name in lines[:60]:
                    ui.chip(name, icon="add").props("clickable").on(
                        "click", lambda e, n=name: _add_line(n)
                    )
            ui.notify(f"{len(lines)} matching line(s)", type="positive" if lines else "warning")
        except Exception as exc:
            ui.notify(f"Search failed: {exc}", type="negative")
        finally:
            search_btn.enable()
            search_btn.text = "Search Lines"

    async def do_list():
        line_names = _line_values(lines_input)
        if not line_names:
            ui.notify("Enter at least one driver line first", type="warning")
            return
        list_btn.disable()
        list_btn.text = "Listing…"
        try:
            files = await run.io_bound(
                _fetch_file_preview, line_names,
                formats.value, image_types.value, region.value,
                categories.value, simple_mode.value,
                int(max_files.value) if max_files.value else None,
            )
            file_table.rows = [
                {
                    "line": f.line_name,
                    "file": f.filename,
                    "type": f.extension,
                    "size": f"{f.size_mb:.1f} MB",
                    "url": f.url,
                }
                for f in files[:200]
            ]
            ui.notify(
                f"{len(files)} file(s) match the filters"
                if files else "No files match the current filters",
                type="positive" if files else "warning",
            )
        except Exception as exc:
            ui.notify(f"Listing failed: {exc}", type="negative")
        finally:
            list_btn.enable()
            list_btn.text = "List Files"

    async def run_download():
        line_names = _line_values(lines_input)
        if not line_names:
            ui.notify("Enter at least one driver line first", type="warning")
            return

        selected_output_dir = str(output_dir.value or "").strip()
        if not selected_output_dir:
            ui.notify("Choose an output directory first", type="warning")
            return

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "output_dir": selected_output_dir,
            "formats": formats.value or ["png"],
            "image_types": image_types.value or ["mip"],
            "region": region.value,
            "collection_category": categories.value or None,
            "max_workers": int(max_workers.value),
            "simple_mode": simple_mode.value,
            "use_boto3": True,
            "include_vt_lines": True,
            "verbose": "pbar",
        }

        summary_value = None if summary.value == "none" else summary.value
        if summary_value == "pdf + pptx":
            summary_value = ["pdf", "pptx"]

        method_params = {
            "line_name": line_names,
            "output_dir": selected_output_dir,
            "max_files": int(max_files.value) if max_files.value else None,
            "flat_structure": flat_structure.value,
            "add_timestamp": add_timestamp.value,
            "generate_summary": summary_value,
            "summary_images_per_page": (int(summary_cols.value), int(summary_rows.value)),
        }

        result = await output_panel.run(
            runner, "flylight_download", constructor_params, "download",
            method_params=method_params,
            output_dir=selected_output_dir,
        )

        # A completed download means the driver lines existed on FlyLight;
        # Keep them in the separate driver-line history (imagery is not tied
        # to a connectome dataset).
        if result["returncode"] == 0:
            from ..line_history_store import record as _record_history
            _record_history([str(v) for v in line_names])

        output_panel.set_running(False)
        output_panel.set_status(
            "Completed" if result["returncode"] == 0 else "Failed",
            "green" if result["returncode"] == 0 else "red",
        )
        output_panel.show_files(
            result["files"], result.get("output_folder") or selected_output_dir
        )

    output_panel.run_button.on_click(run_download)
    output_panel.cancel_button.on_click(runner.cancel)
    search_btn.on_click(do_search)
    list_btn.on_click(do_list)
