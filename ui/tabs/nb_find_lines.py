"""NeuronBridge FindLines Tab - Find LM driver lines matching EM neurons.

The query box is dataset-aware like the Cross-Dataset tab: suggestions
carry a ``column · dataset`` gray hint, history rows carry per-value
dataset badges, and the dataset input is a multi-select with an exclusive
``(all)`` indicator.  Datasets NeuronBridge does not host are disabled
(advisory, from the auto-refreshing coverage snapshot) and an expansion
toggle routes each query chip through the cross-dataset type mapper into
every hosted release before searching.
"""

import json

from nicegui import ui
from ..config import DATASETS, MATCH_ALGORITHMS, get_user_default
from ..components.common import (
    dataset_multi_selector, neuron_list_input, number_input, select_input,
    checkbox_input, dir_input, multi_select_input, output_detail_control,
    apply_filter_mode, section_header, param_grid, tool_page,
)
from ..components.output_panel import OutputPanel
from ..runner import ScriptRunner
from ..type_suggestions import dataset_aware_suggestions

_ALL_DATASETS = "(all)"
_ALL_LABEL = "(all) — search everywhere"

# The coverage refresh talks to the NeuronBridge server and takes seconds to
# minutes (transient reachability included). Tabs are constructed eagerly for
# EVERY browser session, so kicking it per page build piled identical slow
# requests onto the process each time a client connected. Run it once per
# server process; later tab builds reuse the persisted snapshot.
_COVERAGE_REFRESH_STARTED = False

# Seed default before the first coverage snapshot exists on disk (plan
# `_plan/plan-nb-find-lines-dataset-aware-queries.md` §6.1): the four
# releases NeuronBridge hosted as of the 2026-09-15 v3_10_0 probe. The
# snapshot, not this list, drives disable/warning decisions.
_KNOWN_COVERED_DEFAULT = [
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "manc:v1.2.1",
    "flywire_FAFB_v783",
]


def _coverage_module():
    try:
        from src import neuronbridge_coverage as coverage
    except ImportError:
        from neuronbridge_coverage import coverage  # type: ignore
    return coverage


def create_nb_find_lines_tab():
    runner = ScriptRunner()
    output_panel = OutputPanel("Driver Lines Output", state_key="nb_find_lines")
    datasets_select = None

    def _selected_datasets() -> list:
        value = datasets_select.value or [] if datasets_select is not None else []
        return [str(v) for v in value if v != _ALL_DATASETS]

    def _type_suggest(text):
        """Dataset-aware suggestions over the selected datasets ('(all)' or
        an empty selection searches every dataset's pool), with the same
        ``column · dataset`` gray hint as the Cross-Dataset tab."""
        scope = _selected_datasets() or DATASETS
        return dataset_aware_suggestions(text, scope, "auto", limit=None)

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
                available_neurons=lambda: _selected_datasets() or DATASETS,
                show_history_datasets=True,
            )
            with param_grid(2):
                datasets_select = dataset_multi_selector(
                    label="Datasets ('(all)' or empty searches everywhere)",
                    default=[ds for ds in _KNOWN_COVERED_DEFAULT if ds in DATASETS],
                    datasets=[_ALL_DATASETS] + DATASETS,
                    hint=(
                        "Select one or more EM datasets to search. '(all)' or an "
                        "empty selection searches every dataset. Datasets "
                        "NeuronBridge does not host are disabled below (advisory: "
                        "runs still proceed with a warning)."
                    ),
                )

            # '(all)' is an exclusive indicator: picking it clears dataset
            # chips; picking a dataset chip clears '(all)'. The newest
            # addition wins.
            _exclusivity = {"previous": list(datasets_select.value or [])}

            def _sync_all_exclusive(_event=None):
                value = [str(v) for v in (datasets_select.value or [])]
                previous = set(_exclusivity["previous"])
                has_all = _ALL_DATASETS in value
                others = [v for v in value if v != _ALL_DATASETS]
                new_items = [v for v in value if v not in previous]
                if has_all and others:
                    if _ALL_DATASETS in new_items:
                        datasets_select.value = [_ALL_DATASETS]
                    else:
                        datasets_select.value = others
                _exclusivity["previous"] = [
                    str(v) for v in (datasets_select.value or [])
                ]

            datasets_select.on_value_change(_sync_all_exclusive)
            # Keep the synthetic option's label free of dataset-status tags.
            try:
                labels = dict(datasets_select.options)
                if _ALL_DATASETS in labels:
                    labels[_ALL_DATASETS] = _ALL_LABEL
                    datasets_select.set_options(
                        labels, value=datasets_select.value)
            except Exception:
                pass

            coverage_label = ui.label("").classes("text-xs opacity-60 w-full")
            coverage_label.set_visibility(False)

            def _apply_coverage(snapshot):
                """Advisory coverage UI: disable unhosted datasets, note the
                aligned releases.  Unknown/failed refreshes disable nothing."""
                if snapshot is None or datasets_select is None:
                    return
                unavailable = snapshot.unavailable_datasets(DATASETS)
                covered = snapshot.covered_datasets(DATASETS)
                try:
                    # A JS array literal (Python repr's single quotes keep
                    # the double-quoted prop value intact).
                    datasets_select.props(
                        ':option-disable="opt => '
                        f'{unavailable!r}'
                        '.includes(String(opt.value || opt))"'
                    )
                    datasets_select._drocat_nb_unavailable = list(unavailable)
                except Exception:
                    pass
                parts = []
                aligned = {
                    ds: hosted for ds, hosted in covered.items()
                    if hosted != ds
                }
                if covered:
                    parts.append(
                        f"hosted: {len(covered)}/{len(DATASETS)} datasets")
                if aligned:
                    parts.extend(
                        f"{ds} → {hosted}" for ds, hosted in sorted(aligned.items()))
                if unavailable:
                    parts.append(
                        "not hosted (disabled): " + ", ".join(unavailable))
                if parts:
                    coverage_label.text = (
                        "NeuronBridge coverage: " + " · ".join(parts))
                    coverage_label.set_visibility(True)
                datasets_select.update()

            def _load_persisted_coverage():
                try:
                    coverage = _coverage_module()
                    _apply_coverage(coverage.load_snapshot())
                except Exception:
                    pass

            _load_persisted_coverage()

            def _kick_coverage_refresh():
                """Background snapshot refresh; the UI updates via a short
                polling timer so elements are only touched on the event
                loop. The network refresh itself runs once per server
                process (module-level guard): every later tab build just
                re-applies the persisted snapshot."""
                global _COVERAGE_REFRESH_STARTED
                if _COVERAGE_REFRESH_STARTED:
                    return
                _COVERAGE_REFRESH_STARTED = True
                holder = {"snapshot": None}

                def _worker():
                    try:
                        coverage = _coverage_module()
                        holder["snapshot"] = coverage.refresh(DATASETS)
                    except Exception:
                        holder["snapshot"] = False  # stop the polling

                try:
                    import threading
                    threading.Thread(target=_worker, daemon=True).start()
                except Exception:
                    return

                polls = {"left": 120}

                def _poll():
                    polls["left"] -= 1
                    snapshot = holder["snapshot"]
                    if snapshot is False or polls["left"] <= 0:
                        timer.active = False
                    elif snapshot is not None:
                        timer.active = False
                        _apply_coverage(snapshot)

                try:
                    timer = ui.timer(1.0, _poll)
                except Exception:
                    return

            _kick_coverage_refresh()

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
            expand_names = checkbox_input(
                "Expand query names across datasets", True,
                hint=(
                    "Expand each name through the cross-dataset type mapper "
                    "into every NeuronBridge-hosted release before searching "
                    "(e.g. a male-cns v1.0 type also finds its v0.9 name, and "
                    "FAFB/hemibrain equivalents). Datasets NeuronBridge does "
                    "not host are skipped with a warning — never a block."
                ),
            ).props('id="checkbox-nb-expand-names"')
            _, detail_flags = output_detail_control(
                "Full keeps every exported file. Compact keeps the newest "
                "N runs' match tables (belt-and-braces: the latest query "
                "stays inspectable) and prunes older Compact runs' tables; "
                "downloaded images always go once the PDF/PPTX is "
                "generated — every removal is audited in the runs' "
                "cleanup_audit.json. The NeuronBridge match cache is off "
                "by default (Settings → NeuronBridge Match Cache), so a "
                "pruned match table regenerates only by re-running the "
                "query."
            )
            compact_keep_last_n = number_input(
                "Compact: Keep Last N Runs' Match Tables",
                1, 0, 100,
                hint=(
                    "Rolling window for Compact output detail: the newest N "
                    "expanded Find Lines runs keep their bodyId-level match "
                    "tables; older Compact runs are pruned. 0 deletes this "
                    "run's tables immediately. Full ignores this."
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

        selected = _selected_datasets()
        ds = selected or None  # '(all)' or empty -> search everywhere
        expand = bool(expand_names.value)

        # Advisory coverage warnings from the persisted snapshot: notify and
        # proceed.  A missing snapshot warns about nothing and blocks nothing.
        try:
            coverage = _coverage_module()
            snapshot = coverage.load_snapshot()
            for warning in coverage.warnings_for(selected or DATASETS, snapshot):
                ui.notify(warning, type="warning")
        except Exception:
            pass

        output_panel.clear()
        output_panel.set_running(True)

        constructor_params = {
            "verbose": True,
            "separate_splitgal4": separate_split.value,
            "region": region.value,
            "max_workers": int(max_workers.value),
            "use_cache": bool(get_user_default("nb_use_cache")),
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
        tool_name = "nb_find_lines"
        if expand:
            tool_name = "nb_find_lines_expanded"
            method_params["expand_names"] = True
            method_params["coverage_datasets"] = list(DATASETS)
            # Retention is orchestrator-level: only the expanded tool owns
            # the run folder the window sweeps.
            method_params.update(detail_flags())
            method_params["compact_keep_last_n"] = int(compact_keep_last_n.value)
        else:
            method_params.update(detail_flags())
        method_params["compact_keep_last_n"] = int(compact_keep_last_n.value)

        result = await output_panel.run(runner, tool_name, constructor_params, "find_lines",
                                        method_params=method_params,
                                        output_dir=output_dir.value)

        # A completed search means the queried EM identifiers resolved.
        # A dataset-scoped run records that scope; '(all)' leaves the
        # entries unscoped so they appear in every dataset's history list.
        # Per-value badges use exact local-pool membership, so a
        # male-cns-only type never carries the other datasets' tags.
        if result["returncode"] == 0:
            from ..history_store import record as _record_history
            values = [str(v) for v in query]
            value_datasets = {}
            try:
                from ..type_suggestions import datasets_resolving
                value_datasets = datasets_resolving(
                    values, selected or DATASETS)
            except Exception:
                value_datasets = {}
            _record_history(
                values,
                datasets=selected or None,
                value_datasets=value_datasets,
            )

        output_panel.set_running(False)
        output_panel.set_status("Completed" if result["returncode"] == 0 else "Failed",
                                "green" if result["returncode"] == 0 else "red")
        output_panel.show_files(result["files"], result.get("output_folder") or output_dir.value)

    output_panel.run_button.on_click(run_find_lines)
    output_panel.cancel_button.on_click(runner.cancel)
