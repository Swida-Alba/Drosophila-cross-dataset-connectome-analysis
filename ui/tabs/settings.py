"""Settings Tab - Tokens, dataset status, app settings, and storage
management (Settings → Storage: cache/run-folder scan + guarded
removal via ``src/storage_inventory.py``)."""

import json
import os
import shutil
from pathlib import Path

from nicegui import run, ui

from ..codex_pull import codex_puller, human_bytes

from ..config import (
    CACHE_SIMPLIFICATION_OPTIONS,
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    APP_DOCS_BRANCH,
    APP_DOCS_URL,
    APP_GITHUB_URL,
    APP_VERSION,
    DEFAULT_SETTING_GROUPS,
    DEFAULT_SETTING_SPECS,
    PROJECT_ROOT,
    clear_tab_output_overrides,
    get_auto_suggest_enabled,
    get_default_output_dir,
    get_show_history_enabled,
    get_user_default,
    reset_user_defaults,
    set_auto_suggest_enabled,
    set_default_output_dir,
    set_show_history_enabled,
    set_user_default,
)
from ..components.common import (
    checkbox_input,
    combo_input,
    dataset_multi_selector,
    dataset_status_card,
    dir_input,
    number_input,
    param_grid,
    refresh_dataset_selector_statuses,
    section_header,
    select_input,
    sync_output_dir_fields,
)
from ..components.custom_grouper import to_canonical_dict
from ..components.mapping_editor import custom_grouping_block
from ..components.storage_card import create_storage_card
from ..dataset_service import get_dataset_service
from .. import mapping_store


def create_settings_tab():
    with ui.column().classes("w-full drocat-page gap-3"):
        with ui.row().classes("w-full items-center gap-3 drocat-page-head"):
            with ui.element("div").classes("drocat-page-mark"):
                ui.icon("settings").classes("text-white")
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Settings").classes("drocat-page-title")
                    ui.link(
                        "Instructions",
                        "docs/ui_guides/settings.html",
                    ).classes("drocat-doc-link")
                ui.label("Configure API tokens, view dataset status, and manage preferences.").classes("drocat-page-sub")

        # Skeleton downloads always populate the shared raw .swc.zst cache.
        from ..dataset_pull import DatasetPuller
        from ..skeleton_pull import SkeletonPuller

        puller = DatasetPuller()
        skeleton_puller = SkeletonPuller()

        # Dataset Status. The card already owns its title; adding a second
        # section header here made the Settings page look like it contained
        # two different availability controls.
        dataset_status_card()

        with ui.card().classes("w-full drocat-card"):
            section_header("Dataset Cache", "download")
            ui.label(
                "'Pull Dataset Metadata' downloads/verifies only the neuron table, "
                "ROI table, and the materialized neuron index "
                "(datasets/<dataset>/ + neuron_indexes/<dataset>/neuron_index.parquet) — "
                "no connections are fetched. 'Pull Complete Connections' fetches "
                "the full resumable connection cache "
                "(cache/<dataset>/connections.parquet) and skips already-cached "
                "neurons; 'Force rebuild' clears a broken cache first and rebuilds "
                "it completely."
            ).classes("text-caption drocat-muted")

            def _format_eta(st) -> str:
                """Human-readable remaining-time estimate from fetch progress."""
                fetch_started = st.get("fetch_started_at")
                current, total = st.get("current", 0), st.get("total", 0)
                if not fetch_started or current <= 0 or total <= 0 or current >= total:
                    return "ETA --"
                import time as _time
                elapsed = max(_time.time() - fetch_started, 1e-3)
                rate = current / elapsed  # neurons per second
                remaining = (total - current) / rate
                m, s = divmod(int(remaining), 60)
                h, m = divmod(m, 60)
                if h:
                    return f"ETA ~{h}h {m:02d}m"
                return f"ETA ~{m}m {s:02d}s"

            with ui.row().classes("items-center gap-3 w-full").style("flex-wrap: wrap"):
                ds_select = ui.select(
                    options=DATASETS, value=DATASETS[0], label="Dataset"
                ).props("outlined").classes("drocat-select").style("min-width: 260px")
                force_rebuild = ui.checkbox(
                    "Force rebuild (clear broken cache first)"
                ).tooltip(
                    "Deletes the existing cache/<dataset>/connections.parquet "
                    "before fetching, rebuilding the connection cache from scratch. "
                    "The neuron index in neuron_indexes/ survives with reset progress "
                    "flags. Unchecked, the "
                    "pull only fetches the neurons missing from the cache (resume)."
                )
                batch_input = ui.number(
                    label="Batch size", value=100, min=10, max=5000, step=10
                ).classes("drocat-input").style("width: 150px").tooltip(
                    "Neurons fetched per batch (default 100)."
                )
                parallel_input = ui.number(
                    label="Parallel workers", value=4, min=1, max=32, step=1
                ).classes("drocat-input").style("width: 150px").tooltip(
                    "Batches fetched concurrently (1 = sequential). Only raise if the "
                    "NeuPrint/FlyLight server tolerates parallel requests."
                )

            with ui.row().classes("items-center gap-2").style("flex-wrap: wrap"):
                run_btn = ui.button(
                    "Pull Dataset Metadata", icon="download", color="primary"
                ).tooltip(
                    "Downloads/verifies only the dataset neuron table, ROI table, "
                    "and the materialized neuron index — no connections are "
                    "fetched. Already-present files are never re-downloaded."
                )
                connection_run_btn = ui.button(
                    "Pull Complete Connections", icon="hub", color="secondary"
                ).props("outline").tooltip(
                    "Fetches the complete resumable connection cache "
                    "(cache/<dataset>/connections.parquet) for every neuron of "
                    "the dataset; already-cached neurons are skipped."
                )
                cancel_btn = ui.button("Cancel", icon="stop", color="negative").props("outline")
                cancel_btn.set_enabled(False)

            # The bar shows a live percent label; nicegui's default would
            # render the raw 0-1 fraction on the bar instead.  ``show_value``
            # off also thins the bar, so pin the height back to 20px.
            progress = ui.linear_progress(
                value=0, show_value=False, size="20px"
            ).props("instant-feedback").classes("w-full")
            with progress:
                progress_label = ui.label("0%").classes(
                    "absolute-center text-white"
                ).style("font-size: 0.875rem")
            status_label = ui.label("Idle").classes("text-caption drocat-muted")
            result_label = ui.label("").classes("text-caption")
            pull_done_synced = {"value": False}

            def refresh_pull_state():
                st = puller.state
                skeleton_running = skeleton_puller.running
                storage_panel.set_pull_active(
                    st["running"] or skeleton_running)
                if (
                    not st["running"]
                    and run_btn.enabled
                    and connection_run_btn.enabled
                    and not skeleton_running
                ):
                    return
                run_btn.set_enabled(not st["running"] and not skeleton_running)
                connection_run_btn.set_enabled(not st["running"] and not skeleton_running)
                cancel_btn.set_enabled(st["running"] or skeleton_running)
                ds_select.set_enabled(not st["running"] and not skeleton_running)
                force_rebuild.set_enabled(not st["running"] and not skeleton_running)
                batch_input.set_enabled(not st["running"] and not skeleton_running)
                parallel_input.set_enabled(not st["running"] and not skeleton_running)
                skeleton_simplification.set_enabled(
                    not st["running"] and not skeleton_running
                )
                skeleton_batch_size.set_enabled(
                    not st["running"] and not skeleton_running
                )
                if st["running"]:
                    pull_done_synced["value"] = False
                    operation_label = (
                        "complete connections"
                        if st.get("operation") == "connections"
                        else "dataset metadata"
                    )
                    if st.get("cancel_requested"):
                        # Wind-down after Cancel: the in-flight batch and the
                        # batch consolidation can take a while — show a
                        # persistent hint and an indeterminate bar instead of
                        # the frozen progress text.
                        progress.props(add="indeterminate")
                        progress_label.text = ""
                        status_label.text = (
                            f"Cancelling {operation_label} pull for "
                            f"{st['dataset']} — stopping after the current "
                            f"batch and consolidating fetched data..."
                        )
                        return
                    if st["total"] and st["total"] > 0:
                        # Fetch phase: neuron totals are known, show a
                        # determinate bar with the live percent.
                        progress.props(remove="indeterminate")
                        total = st["total"] or 1
                        frac = min(st["current"] / total, 1.0)
                        progress.set_value(frac)
                        progress_label.text = f"{frac * 100:.2f}%"
                        status_label.text = (
                            f"Pulling {operation_label} for {st['dataset']}: {st['info']} "
                            f"({st['current']:,}/{st['total']:,} neurons remaining, "
                            f"{frac * 100:.2f}%) | {_format_eta(st)}"
                        )
                    else:
                        # Preparation phase (client connect / dataset download
                        # / index build): no totals yet — an indeterminate bar
                        # plus the phase message keeps the pull from looking
                        # frozen at 0/0.
                        progress.props(add="indeterminate")
                        progress_label.text = ""
                        status_label.text = (
                            f"Preparing {operation_label} pull for "
                            f"{st['dataset']}: {st['info']}"
                        )
                    return
                if not st["done"]:
                    return
                if not pull_done_synced["value"]:
                    progress.props(remove="indeterminate")
                    # A pull creates the local cache files directly. Refresh
                    # selector labels immediately so they do not wait for a
                    # full page rebuild (or a separate availability refresh).
                    refresh_dataset_selector_statuses()
                    pull_done_synced["value"] = True
                if st["error"]:
                    status_label.text = "Failed"
                    result_label.text = f"❌ {st['error']}"
                else:
                    s = st["summary"] or {}
                    if st.get("operation") == "full_dataset":
                        # Metadata-only pull: no connection totals exist.
                        head = (
                            "⏹ Cancelled during metadata preparation."
                            if st["cancelled"]
                            else "✅ Dataset metadata ready."
                        )
                        result_label.text = (
                            f"{head} Neuron table: {s.get('neuron_count', 0):,} "
                            f"neurons | index rows: {s.get('index_rows', 0):,} | "
                            f"{s.get('elapsed_time', 0):.1f}s"
                        )
                    else:
                        operation_label = (
                            "Complete connection pull"
                            if st.get("operation") == "connections"
                            else "Dataset metadata pull"
                        )
                        if st["cancelled"]:
                            # Leave the bar where the pull stopped; the result
                            # line says it can be resumed.
                            progress_label.text = ""
                        else:
                            progress.set_value(1.0)
                            progress_label.text = "100%"
                        head = (
                            "⏹ Cancelled - fetched batches consolidated; re-run to resume."
                            if st["cancelled"]
                            else f"✅ {operation_label} complete."
                        )
                        result_label.text = (
                            f"{head} Target: {s.get('total_neurons', 0):,} | "
                            f"newly cached: {s.get('newly_cached', 0):,} | "
                            f"already cached: {s.get('already_cached', 0):,} | "
                            f"failed: {len(s.get('failed_neurons', [])):,} | "
                            f"connections: {s.get('total_connections', 0):,} | "
                            f"{s.get('elapsed_time', 0):.1f}s"
                        )
                    status_label.text = "Idle"

            def _start_pull(operation: str):
                ok = puller.start(
                    str(ds_select.value),
                    force_rebuild=force_rebuild.value,
                    batch_size=int(batch_input.value or 100),
                    max_workers=int(parallel_input.value or 1),
                    operation=operation,
                )
                if ok:
                    run_btn.set_enabled(False)
                    connection_run_btn.set_enabled(False)
                    skeleton_run_btn.set_enabled(False)
                    cancel_btn.set_enabled(True)
                    result_label.text = ""
                    progress.set_value(0)
                    progress_label.text = "0%"
                    pull_done_synced["value"] = False
                else:
                    ui.notify("A dataset pull is already running", type="warning")

            def start_pull():
                _start_pull("full_dataset")

            def start_connection_pull():
                _start_pull("connections")

            def stop_pull():
                if puller.running:
                    puller.cancel()
                    status_label.text = "Cancelling after the current batch..."
                if skeleton_puller.running:
                    skeleton_puller.cancel()
                    skeleton_status.text = "Cancelling after the current batch..."

            run_btn.on_click(start_pull)
            connection_run_btn.on_click(start_connection_pull)
            cancel_btn.on_click(stop_pull)
            ui.timer(0.5, refresh_pull_state)

            # Skeleton downloads sit between the dataset-pull progress and
            # the skeleton-pull progress so each pull's controls are grouped
            # with its own bar.
            with ui.row().classes("items-center gap-2").style("flex-wrap: wrap"):
                skeleton_simplification = ui.select(
                    options=CACHE_SIMPLIFICATION_OPTIONS,
                    value=0,
                    label="Cache Simplification",
                ).props("outlined").classes("drocat-select").style("min-width: 190px").tooltip(
                    "Percent of skeleton nodes REMOVED when 'Download All "
                    "Skeletons' writes the shared .swc.zst cache: 0 = raw "
                    "(default — the cache stores raw skeletons; "
                    "simplification is applied at visualization time). "
                    "Every file records its level in the header."
                )
                skeleton_batch_size = ui.number(
                    label="Skeleton Batch Size", value=64, min=10, max=500,
                    step=10,
                ).classes("drocat-input").style("width: 160px").tooltip(
                    "Neurons requested per navis call (64 default). Progress "
                    "is reported per completed skeleton regardless of this "
                    "value; larger batches only reduce the repeated per-batch "
                    "metadata queries. Concurrent fetches are set by "
                    "Parallel workers."
                )
                skeleton_run_btn = ui.button(
                    "Download All Skeletons", icon="download", color="secondary"
                ).props("outline").tooltip(
                    "Download missing raw skeletons as reusable .swc.zst files "
                    "at the Cache Simplification level selected beside this "
                        "button, using the Dataset and Parallel workers "
                        "selected above. FAFB requires a manual Codex "
                        "download (see the FAFB converter instructions); "
                        "BANC skeletons are fetched from its public release "
                        "bucket."
                )

            skeleton_progress = ui.linear_progress(
                value=0, show_value=False, size="20px"
            ).props("instant-feedback").classes("w-full")
            with skeleton_progress:
                skeleton_progress_label = ui.label("0%").classes(
                    "absolute-center text-white"
                ).style("font-size: 0.875rem")
            skeleton_status = ui.label("Idle").classes("text-caption drocat-muted")
            skeleton_result = ui.label("").classes("text-caption")
            skeleton_done_synced = {"value": False}

            def refresh_skeleton_pull_state():
                st = skeleton_puller.state
                running = bool(st["running"])
                dataset_running = puller.running
                storage_panel.set_pull_active(running or dataset_running)
                skeleton_run_btn.set_enabled(not running and not dataset_running)
                cancel_btn.set_enabled(running or dataset_running)
                # The dataset selector is shared with both dataset-cache pulls.
                ds_select.set_enabled(not running and not puller.running)
                skeleton_simplification.set_enabled(
                    not running and not puller.running
                )
                skeleton_batch_size.set_enabled(
                    not running and not puller.running
                )
                if running:
                    skeleton_done_synced["value"] = False
                    if st.get("cancel_requested"):
                        # Wind-down after Cancel: the in-flight navis batch
                        # and the persist phase can take a while — show a
                        # persistent hint and an indeterminate bar instead of
                        # the frozen progress text.
                        skeleton_progress.props(add="indeterminate")
                        skeleton_progress_label.text = ""
                        skeleton_status.text = (
                            f"Cancelling skeleton pull for {st['dataset']} — "
                            f"finishing the current batch and persisting "
                            f"fetched skeletons..."
                        )
                        return
                    total = st["total"] or 1
                    frac = min(st["current"] / total, 1.0)
                    skeleton_progress.set_value(frac)
                    skeleton_progress_label.text = f"{frac * 100:.2f}%"
                    skeleton_status.text = (
                        f"Pulling {st['dataset']}: "
                        f"{st['info']} ({st['current']:,}/{st['total']:,}) | "
                        f"{_format_eta(st)}"
                    )
                    return
                if not st["done"]:
                    return
                if not skeleton_done_synced["value"]:
                    skeleton_progress.props(remove="indeterminate")
                    refresh_dataset_selector_statuses()
                    skeleton_done_synced["value"] = True
                if st["error"]:
                    skeleton_status.text = "Failed"
                    skeleton_result.text = f"❌ {st['error']}"
                else:
                    summary = st.get("summary") or {}
                    if st["cancelled"]:
                        skeleton_progress_label.text = ""
                    else:
                        skeleton_progress.set_value(1.0)
                        skeleton_progress_label.text = "100%"
                    head = "⏹ Cancelled." if st["cancelled"] else "✅ Skeleton pull complete."
                    skeleton_result.text = (
                        f"{head} "
                        f"fetched: {summary.get('fetched', 0):,}; "
                        f"already available: {summary.get('skipped_existing', 0):,}; "
                        f"errors: {summary.get('errors', 0):,}"
                    )
                    skeleton_status.text = "Idle"

            def start_skeleton_pull():
                if puller.running:
                    ui.notify("Finish the active dataset/connection pull before downloading skeletons", type="warning")
                    return
                dataset = str(ds_select.value)
                # FAFB skeletons are supplied as a manually downloaded healed
                # bundle. BANC is independent: its skeletons are fetched by
                # the puller from the public release bucket.
                # Note: the dataset predicate lives in flywire_ids, not in
                # flywire_readiness (which only defines is_banc/is_fafb).
                try:
                    from src.utils.flywire_readiness import (
                        flywire_manual_skeleton_instruction,
                    )
                    from src.flywire_ids import is_banc_dataset, is_fafb_dataset
                except ImportError:
                    from utils.flywire_readiness import (
                        flywire_manual_skeleton_instruction,
                    )
                    from flywire_ids import is_banc_dataset, is_fafb_dataset
                if is_fafb_dataset(dataset):
                    message = flywire_manual_skeleton_instruction(dataset)
                    skeleton_status.text = "Manual download required (FAFB)"
                    skeleton_result.text = f"❌ {message}"
                    ui.notify(message, type="warning")
                    return
                if is_banc_dataset(dataset):
                    skeleton_status.text = (
                        "Fetching BANC skeletons from the public release "
                        "bucket...")
                ok = skeleton_puller.start(
                    dataset,
                    max_workers=int(parallel_input.value or 1),
                    mode="raw",
                    simplification=int(skeleton_simplification.value),
                    batch_size=int(skeleton_batch_size.value or 64),
                )
                if ok:
                    run_btn.set_enabled(False)
                    connection_run_btn.set_enabled(False)
                    skeleton_run_btn.set_enabled(False)
                    cancel_btn.set_enabled(True)
                    skeleton_result.text = ""
                    skeleton_progress.set_value(0)
                    skeleton_progress_label.text = "0%"
                    skeleton_done_synced["value"] = False
                else:
                    ui.notify("A skeleton pull is already running", type="warning")

            skeleton_run_btn.on_click(start_skeleton_pull)
            ui.timer(0.5, refresh_skeleton_pull_state)

        # Storage (caches + exported data): the scan/delete utility card.
        # The pull pollers below flip its actions off while a pull runs.
        storage_panel = create_storage_card(puller, skeleton_puller)

        # Tokens
        with ui.card().classes("w-full drocat-card"):
            section_header("API Tokens", "key")

            # Full detection chain (config files then env vars) so env-var
            # tokens are recognized as configured; the source is shown in
            # the status labels next to each field.
            token_sources = _token_sources()
            token_state = {
                "neuprint": token_sources["neuprint"][0],
                "cave": token_sources["cave"][0],
                "flywire_codex": token_sources["flywire_codex"][0],
            }

            # Reminder when tokens are missing: the NeuPrint token is
            # required for NeuPrint datasets; the FlyWire Codex token
            # enables the automatic FAFB downloads. Refreshed whenever the
            # saved tokens change.
            token_reminder = ui.element("div").props('id="drocat-token-reminder"').classes(
                "w-full drocat-token-reminder"
            )
            with token_reminder:
                token_reminder_text = ui.label("").classes("text-sm drocat-warn")

            def _refresh_token_reminder():
                neuprint_missing = not token_state.get("neuprint")
                codex_note = ""
                if not token_state.get("flywire_codex"):
                    codex_note = (
                        " ℹ️ FlyWire Codex token not configured - optional; "
                        "it enables the automatic FAFB downloads on this tab "
                        "and lazy skeleton fetching."
                    )
                if not neuprint_missing and not codex_note:
                    token_reminder.set_visibility(False)
                    return
                token_reminder.set_visibility(True)
                base = (
                    "⚠️ NeuPrint token not configured - it is required for "
                    "NeuPrint datasets. Set it below or in config.json."
                    if neuprint_missing else ""
                )
                token_reminder_text.text = (base + codex_note).strip()
                token_reminder_text.update()

            _refresh_token_reminder()

            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("NeuPrint Token (Required for all NeuPrint datasets)").classes("text-caption font-bold")
                    neuprint_status = ui.label(_token_status(token_state["neuprint"], token_sources["neuprint"][1])).classes("text-caption drocat-muted")
                ui.html("Get it from <a href='https://neuprint.janelia.org/account' target='_blank' style='color:var(--drocat-cobalt)'>neuprint.janelia.org/account</a>").classes("text-caption drocat-muted")

            neuprint_token = ui.input(
                label="NeuPrint Token",
                value="",
                placeholder="Leave blank to keep the saved token",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            ui.separator()

            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("FlyWire Codex Token (for automatic FAFB downloads)").classes("text-caption font-bold")
                    codex_status = ui.label(_token_status(token_state["flywire_codex"], token_sources["flywire_codex"][1])).classes("text-caption drocat-muted")
                ui.html("Sign in at <a href='https://codex.flywire.ai/api/download?dataset=fafb' target='_blank' style='color:var(--drocat-cobalt)'>codex.flywire.ai</a> with a Google account, then copy your API token from <a href='https://codex.flywire.ai/account' target='_blank' style='color:var(--drocat-cobalt)'>codex.flywire.ai/account</a>").classes("text-caption drocat-muted")
                ui.label("Needed only for the automatic FAFB data downloads on this tab and lazy per-neuron skeleton fetching. The CAVE token is a separate, contributor-gated credential that no download or analysis needs; power users can still set it via config.json (see the FAFB instructions below).").classes("text-caption drocat-muted")

            codex_token = ui.input(
                label="FlyWire Codex Token",
                value="",
                placeholder="Leave blank to keep the saved token",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            with ui.row().classes("items-center gap-2 flex-wrap"):
                save_btn = ui.button("Save Tokens", icon="save", color="primary")
                test_btn = ui.button("Test NeuPrint", icon="wifi", color="secondary")
                codex_test_btn = ui.button("Test Codex", icon="wifi", color="secondary")
                clear_blank = ui.checkbox(
                    "Clear a saved value when its field is blank",
                    value=False,
                ).classes("text-caption")
            ui.label(
                "Saved token values stay on the server and are never pre-filled in the browser. "
                "Enter a new value only when you want to replace the current one."
            ).classes("text-caption drocat-muted")

        # FlyWire Codex downloads (FAFB)
        import codex_downloader
        with ui.card().classes("w-full drocat-card"):
            section_header("FAFB Data Downloads (FlyWire Codex)", "cloud_download")
            downloads_dir = codex_downloader.downloads_dir_for(
                project_root=PROJECT_ROOT)

            codex_gate_label = ui.label("").classes("text-caption drocat-warn")
            necessary_check = ui.checkbox(
                "1 · Necessary data — always included ("
                f"{human_bytes(codex_downloader.level_bytes(codex_downloader.LEVEL_NECESSARY))}, 7 files: "
                "neuron table, connections and metadata for the FAFB converter)",
                value=True,
            ).props("disable").tooltip(
                "The FAFB converter's required downloads plus the metadata "
                "enrichment files. This level is part of every download.")
            synapse_check = ui.checkbox(
                "2 · Synapse table ("
                f"{human_bytes(codex_downloader.level_bytes(codex_downloader.LEVEL_SYNAPSE))})"
                " — needed for synapse visualization",
                value=False,
            ).tooltip(
                "Downloads the per-synapse table and converts it into the "
                "local synapse parquet that skeleton visualization reads.")
            skeleton_check = ui.checkbox(
                "3 · Skeleton bundle ("
                f"{human_bytes(codex_downloader.level_bytes(codex_downloader.LEVEL_SKELETON))})"
                " — full local skeletons",
                value=False,
            ).tooltip(
                "Without the bundle, skeletons fetch lazily per neuron from "
                "the Codex server during visualization (a Codex token is "
                "still needed).")

            with ui.column().classes("w-full gap-0"):
                codex_status_labels = {
                    key: ui.label("").classes("text-caption drocat-muted")
                    for key in codex_downloader.PRODUCT_CATALOG
                }

            def _refresh_codex_file_status():
                for key, label in codex_status_labels.items():
                    status = codex_downloader.product_status(key, downloads_dir)
                    if status["complete"]:
                        mark = "✓"
                    elif status["local"]:
                        mark = "◐ partial"
                    else:
                        mark = "· missing"
                    label.text = (f"{mark}  {status['filename']} "
                                  f"({human_bytes(status['expected'])})")

            _refresh_codex_file_status()

            with ui.row().classes("items-center gap-2 flex-wrap"):
                download_btn = ui.button(
                    "Download & Convert", icon="cloud_download", color="primary"
                ).tooltip(
                    "Downloads the selected levels into "
                    "datasets/flywire_FAFB_v783/downloads/ (resumable; "
                    "interrupted files continue where they stopped) and runs "
                    "the FAFB converter when the necessary level completes.")
                convert_btn = ui.button(
                    "Convert Only", icon="transform", color="secondary"
                ).props("outline").tooltip(
                    "Runs the FAFB converter over the local downloads "
                    "(no network) — the python src/FAFB_file_converter.py "
                    "equivalent.")
                codex_cancel_btn = ui.button(
                    "Cancel", icon="stop", color="negative").props("outline")
                codex_cancel_btn.set_enabled(False)

            codex_progress = ui.linear_progress(
                value=0, show_value=False, size="20px"
            ).props("instant-feedback").classes("w-full")
            with codex_progress:
                codex_progress_label = ui.label("0%").classes(
                    "absolute-center text-white"
                ).style("font-size: 0.875rem")
            codex_status_label = ui.label("Idle").classes(
                "text-caption drocat-muted")
            codex_result_label = ui.label("").classes("text-caption")
            codex_done_synced = {"value": False}

            def _refresh_codex_gate():
                has_token = bool(token_state.get("flywire_codex"))
                codex_gate_label.text = (
                    "⚠️ Save a FlyWire Codex API token in the API Tokens "
                    "card above to enable Codex downloads (Convert Only "
                    "works without one).")
                codex_gate_label.set_visibility(not has_token)
                download_btn.set_enabled(
                    has_token and not codex_puller.running)
                convert_btn.set_enabled(not codex_puller.running)

            def _selected_levels():
                levels = [codex_downloader.LEVEL_NECESSARY]
                if synapse_check.value:
                    levels.append(codex_downloader.LEVEL_SYNAPSE)
                if skeleton_check.value:
                    levels.append(codex_downloader.LEVEL_SKELETON)
                return levels

            def _free_space_anchor() -> Path:
                anchor = downloads_dir
                while not anchor.exists() and anchor.parent != anchor:
                    anchor = anchor.parent
                return anchor

            def _start_codex_pull(download: bool):
                if codex_puller.running:
                    ui.notify("A Codex download is already running",
                              type="warning")
                    return
                levels = _selected_levels() if download else []
                if levels and not token_state.get("flywire_codex"):
                    ui.notify(
                        "Save a FlyWire Codex token first "
                        "(API Tokens card above)", type="warning")
                    return
                if levels:
                    needed = 0
                    for level in levels:
                        for product in codex_downloader.level_products(level):
                            status = codex_downloader.product_status(
                                product.key, downloads_dir)
                            if not status["complete"]:
                                needed += product.size - status["local"]
                    free = shutil.disk_usage(_free_space_anchor()).free
                    if needed > free:
                        ui.notify(
                            f"Not enough disk space: {human_bytes(needed)} "
                            f"needed, {human_bytes(free)} free",
                            type="negative")
                        return
                if codex_puller.start(levels, run_converter=True,
                                      project_root=PROJECT_ROOT):
                    codex_result_label.text = ""
                    codex_progress.set_value(0)
                    codex_progress_label.text = "0%"
                    codex_done_synced["value"] = False
                    _refresh_codex_gate()
                else:
                    ui.notify("A Codex download is already running",
                              type="warning")

            def refresh_codex_pull_state():
                st = codex_puller.state
                has_token = bool(token_state.get("flywire_codex"))
                download_btn.set_enabled(has_token and not st["running"])
                convert_btn.set_enabled(not st["running"])
                codex_cancel_btn.set_enabled(st["running"])
                synapse_check.set_enabled(not st["running"])
                skeleton_check.set_enabled(not st["running"])
                if not st["running"] and not st["done"]:
                    return
                if st["running"]:
                    codex_done_synced["value"] = False
                    if st.get("cancel_requested"):
                        codex_progress.props(add="indeterminate")
                        codex_progress_label.text = ""
                        codex_status_label.text = (
                            "Cancelling — finishing the current chunk "
                            "(partial files resume)...")
                        return
                    if st["total"] and st["total"] > 0:
                        codex_progress.props(remove="indeterminate")
                        frac = min(st["current"] / st["total"], 1.0)
                        codex_progress.set_value(frac)
                        codex_progress_label.text = f"{frac * 100:.1f}%"
                        codex_status_label.text = (
                            f"{st['info']} "
                            f"({human_bytes(st['current'])} / "
                            f"{human_bytes(st['total'])}) | {_format_eta(st)}")
                    else:
                        codex_progress.props(add="indeterminate")
                        codex_progress_label.text = ""
                        codex_status_label.text = st["info"]
                    return
                if not st["done"]:
                    return
                if not codex_done_synced["value"]:
                    codex_progress.props(remove="indeterminate")
                    _refresh_codex_file_status()
                    refresh_dataset_selector_statuses()
                    codex_done_synced["value"] = True
                if st["error"]:
                    codex_status_label.text = "Failed"
                    codex_result_label.text = f"❌ {st['error']}"
                else:
                    s = st["summary"] or {}
                    if st["cancelled"]:
                        codex_progress_label.text = ""
                    else:
                        codex_progress.set_value(1.0)
                        codex_progress_label.text = "100%"
                    head = ("⏹ Cancelled — partial files resume on the "
                            "next run." if st["cancelled"]
                            else "✅ Codex download complete.")
                    codex_result_label.text = (
                        f"{head} downloaded {len(s.get('downloaded', []))} "
                        f"file(s), already present "
                        f"{len(s.get('skipped', []))} | converter: "
                        f"{s.get('convert_note', 'not run')}")
                    codex_status_label.text = "Idle"

            _refresh_codex_gate()
            download_btn.on_click(lambda: _start_codex_pull(download=True))
            convert_btn.on_click(lambda: _start_codex_pull(download=False))
            codex_cancel_btn.on_click(codex_puller.cancel)
            ui.timer(0.5, refresh_codex_pull_state)

        # Output
        with ui.card().classes("w-full drocat-card"):
            section_header("Output Settings", "folder")
            default_dir = dir_input(
                label="Default Output Directory",
                default=get_default_output_dir(),
                global_default=True,
            )
            ui.label(
                "This directory is inherited by tool tabs until a tab-specific "
                "override is saved. Reset clears every tab override. "
                f"Project root: {PROJECT_ROOT}"
            ).classes("text-caption drocat-muted")
            with ui.row():
                save_default_btn = ui.button("Save Default Directory", icon="save", color="primary")
                reset_default_btn = ui.button("Reset", icon="restart_alt", color="secondary")

            def save_default_dir():
                raw_value = (default_dir.value or "").strip()
                if not raw_value:
                    ui.notify("Choose an output directory first", type="warning")
                    return
                saved, effective = set_default_output_dir(raw_value, create=True)
                if not saved or not effective:
                    ui.notify("Cannot save output directory (check the path)", type="negative")
                    return
                default_dir.value = effective
                sync_output_dir_fields(default_dir, effective)
                ui.notify("Default output directory saved", type="positive")

            def reset_default_dir():
                set_default_output_dir("")
                clear_tab_output_overrides()
                default_dir.value = str(DEFAULT_OUTPUT_DIR)
                sync_output_dir_fields(
                    default_dir,
                    str(DEFAULT_OUTPUT_DIR),
                    force=True,
                )
                ui.notify("Default output directory reset", type="positive")

            save_default_btn.on_click(save_default_dir)
            reset_default_btn.on_click(reset_default_dir)

        # App Settings
        with ui.card().classes("w-full drocat-card"):
            section_header("App Settings", "tune")
            ui.label(
                "Auto-suggestion shows dataset type/instance/bodyId names while "
                "typing in a neuron input; query history shows Recent/Frequent "
                "values when the field is empty. Both drive the pathfinding "
                "neuron boxes AND the advanced layer editor cell dropdown, and "
                "take effect immediately."
            ).classes("text-caption drocat-muted")
            with ui.column().classes("gap-1"):
                auto_suggest_cb = ui.checkbox(
                    "Input Auto-Suggestion",
                    value=get_auto_suggest_enabled(),
                ).tooltip(
                    "Type-ahead suggestions (dataset type names) in the "
                    "pathfinding neuron inputs and the advanced layer editor "
                    "cell dropdown. Disabling restores plain chip inputs (the "
                    "query-history list below is unaffected)."
                )
                history_cb = ui.checkbox(
                    "Show Query History",
                    value=get_show_history_enabled(),
                ).tooltip(
                    "Recent/Frequent history list shown when a neuron input's "
                    "field is empty, in the pathfinding neuron inputs and the "
                    "advanced layer editor cell dropdown. Disabling hides the "
                    "history list but keeps live type-ahead suggestions."
                )

            def _toggle_setting(event, setter, what):
                enabled = bool(event.value)
                if setter(enabled):
                    ui.notify(
                        f"{what} " + ("enabled" if enabled else "disabled"),
                        type="positive" if enabled else "warning",
                    )
                else:
                    ui.notify("Failed to save the setting", type="negative")

            auto_suggest_cb.on_value_change(
                lambda e, s=set_auto_suggest_enabled, w="Input auto-suggestion":
                _toggle_setting(e, s, w)
            )
            history_cb.on_value_change(
                lambda e, s=set_show_history_enabled, w="Query history":
                _toggle_setting(e, s, w)
            )

        # Default Settings (persistent defaults for the one-time tab controls)
        with ui.card().classes("w-full drocat-card"):
            section_header("Default Settings", "settings_suggest")
            ui.label(
                "Tool-tab selections remain one-time; these defaults are applied "
                "every time a tab loads. Save to persist them across sessions; "
                "Reset restores the built-in values."
            ).classes("text-caption drocat-muted")

            default_controls = {}

            def _render_default_control(key):
                spec = DEFAULT_SETTING_SPECS[key]
                value = get_user_default(key)
                if spec["kind"] == "bool":
                    return checkbox_input(
                        spec["label"], value, hint=spec.get("hint", ""),
                    )
                if spec["kind"] == "select":
                    return select_input(
                        spec["label"], spec["options"], value,
                        hint=spec.get("hint", ""),
                    )
                if spec["kind"] == "combo":
                    return combo_input(
                        spec["label"], spec["options"], value,
                        hint=spec.get("hint", ""),
                    )
                return number_input(
                    spec["label"], value,
                    spec.get("min", 0), spec.get("max"),
                    spec.get("step", 1),
                    hint=spec.get("hint", ""),
                )

            for group_id, group_title in DEFAULT_SETTING_GROUPS:
                group_keys = [
                    key for key, spec in DEFAULT_SETTING_SPECS.items()
                    if spec["group"] == group_id
                ]
                if not group_keys:
                    continue
                ui.label(group_title).classes("drocat-mini-label")
                bool_keys = [
                    key for key in group_keys
                    if DEFAULT_SETTING_SPECS[key]["kind"] == "bool"
                ]
                other_keys = [key for key in group_keys if key not in bool_keys]
                if bool_keys:
                    with ui.row().classes("gap-4").style("flex-wrap: wrap"):
                        for key in bool_keys:
                            default_controls[key] = _render_default_control(key)
                if other_keys:
                    with param_grid(3):
                        for key in other_keys:
                            default_controls[key] = _render_default_control(key)

            def _collect_default_values():
                """Read and coerce every control value; None on invalid input."""
                values = {}
                for key, ctrl in default_controls.items():
                    spec = DEFAULT_SETTING_SPECS[key]
                    value = ctrl.value
                    if spec["kind"] == "bool":
                        value = bool(value)
                    elif spec["kind"] == "int":
                        try:
                            value = int(value)
                        except (TypeError, ValueError):
                            return None, spec["label"]
                    elif spec["kind"] == "float":
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            return None, spec["label"]
                    elif spec["kind"] == "combo":
                        value = str(value or "").strip()
                        if not value:
                            return None, spec["label"]
                    values[key] = value
                return values, None

            def save_defaults():
                values, invalid_label = _collect_default_values()
                if invalid_label is not None:
                    ui.notify(
                        f"Enter a valid number for {invalid_label}",
                        type="warning",
                    )
                    return
                for key, value in values.items():
                    if not set_user_default(key, value):
                        ui.notify(
                            f"Invalid value for {DEFAULT_SETTING_SPECS[key]['label']}",
                            type="negative",
                        )
                        return
                ui.notify(
                    f"Defaults saved ({len(values)} settings)",
                    type="positive",
                )

            def reset_defaults():
                reset_user_defaults()
                for key, ctrl in default_controls.items():
                    ctrl.set_value(get_user_default(key))
                ui.notify("Defaults reset to built-in values", type="positive")

            with ui.row().classes("items-center gap-2"):
                save_defaults_btn = ui.button(
                    "Save Defaults", icon="save", color="primary",
                )
                reset_defaults_btn = ui.button(
                    "Reset", icon="restart_alt", color="secondary",
                )
            save_defaults_btn.on_click(save_defaults)
            reset_defaults_btn.on_click(reset_defaults)

        # Custom type mappings (LabelMapper-format groups; saving writes a
        # reusable JSON preset).
        # Use the same Custom Mapping panel as the tool tabs so Settings and
        # query-local mapping edits share the outlined member fields, aligned
        # dataset rows, and available-neuron viewer.
        with ui.card().classes("w-full drocat-card"):
            section_header("Custom Type Mappings", "hub")
            ui.label(
                "Define reusable neuron groups across datasets. The shared Custom Mapping "
                "panel keeps the same aligned member inputs and LabelMapper JSON format "
                "used by the analysis tabs. Saving updates the Custom Mapping "
                "preset in cache/user_mappings.json and exports its JSON — runs never "
                "load it automatically; reuse it from a board's Load… action."
            ).classes("text-caption drocat-muted")

            mapping_row_action_renderers = []

            mapping_dataset_select = None

            def _render_mapping_dataset_selector():
                nonlocal mapping_dataset_select
                mapping_dataset_select = dataset_multi_selector(
                    label="Target datasets",
                    default=[],
                    group_recommended=True,
                    hint=(
                        "Select the datasets this reusable mapping should apply to. "
                        "The editor renders and saves every selected dataset column for "
                        "each group, including empty cells."
                    ),
                )
                return mapping_dataset_select

            def _mapping_datasets():
                values = mapping_dataset_select.value if mapping_dataset_select else []
                values = values or []
                if not isinstance(values, (list, tuple, set)):
                    values = [values]
                return [str(dataset) for dataset in values if dataset]

            mapping_button, mapping_dialog, _resolve_mapping = custom_grouping_block(
                label="Custom Mapping",
                hint="Open the shared LabelMapper group editor with aligned dataset rows "
                     "and available-neuron lookup.",
                datasets_provider=_mapping_datasets,
                tab_key="settings_mapping",
                panel_title="Custom Mapping",
                dataset_selector_renderer=_render_mapping_dataset_selector,
                row_action_renderers=mapping_row_action_renderers,
            )
            mapping_button.classes("drocat-settings-mapping-button")
            grouper = mapping_dialog.inline_grouper

            def _save_custom_mapping():
                # Settings intentionally has one reusable mapping surface now;
                # the old name/description/preset controls are removed from
                # this card. Save to a stable preset name and make it active.
                name = "Custom Mapping"
                if not grouper.datasets():
                    ui.notify(
                        "Select at least one target dataset first",
                        type="warning",
                    )
                    return
                if grouper.is_empty():
                    ui.notify("Add at least one group with neurons first", type="warning")
                    return
                errors = grouper.validate()
                if errors:
                    for err in errors:
                        ui.notify(err, type="negative")
                    return
                payload = to_canonical_dict(
                    grouper._active_rows(),
                    grouper.datasets(),
                    origin="settings",
                )
                sides = {
                    "source_mapping": payload["source_mapping"],
                    "target_mapping": payload["target_mapping"],
                }
                if not mapping_store.save_mapping(name, sides):
                    ui.notify("Failed to save mapping", type="negative")
                    return
                mapping_store.set_active_mapping(name)
                ui.notify("Custom Mapping saved and set active", type="positive")

            def _render_save_mapping_action():
                ui.button(
                    "Save Mapping",
                    icon="save",
                    on_click=_save_custom_mapping,
                ).props("outline no-caps").classes(
                    "drocat-labelmapper-query-action"
                ).tooltip(
                    "Save this mapping using the selected target datasets"
                )

            mapping_row_action_renderers.append(_render_save_mapping_action)
            # The first render happens before the callbacks above are defined;
            # refresh once so an already-open editor receives the new action.
            grouper.resync()

        # Dataset Preparation Guide
        with ui.card().classes("w-full drocat-card"):
            section_header("Dataset Preparation Guide", "menu_book")

            with ui.expansion("NeuPrint Datasets (hemibrain, male-cns, optic-lobe, manc)", icon="cloud").classes("w-full"):
                ui.html("""
                <div style="color:var(--drocat-navy)" class="text-sm">
                    <p><b>NeuPrint datasets are fetched automatically from the server.</b> No manual download required.</p>
                    <ol class="list-decimal ml-4 mt-2">
                        <li>Get your NeuPrint token from <a href="https://neuprint.janelia.org/account" target="_blank" style="color:var(--drocat-cobalt)">neuprint.janelia.org/account</a></li>
                        <li>Enter the token in the API Tokens section above</li>
                        <li>Click "Save Tokens"</li>
                        <li>Click "Test NeuPrint" to verify the connection</li>
                        <li>Datasets will appear in all tool tabs automatically</li>
                    </ol>
                    <p class="mt-2" style="color:var(--drocat-muted)">Available NeuPrint datasets:</p>
                    <ul class="list-disc ml-4" style="color:var(--drocat-muted)">
                        <li>hemibrain:v1.2.1 - Adult fly brain (central)</li>
                        <li>male-cns:v1.0 - Full male CNS (latest)</li>
                        <li>male-cns:v0.9 - Full male CNS</li>
                        <li>optic-lobe:v1.1 - Optic lobe detailed</li>
                        <li>manc:v1.2.1 / manc:v1.0 - Male VNC</li>
                    </ul>
                </div>
                """)

            with ui.expansion("FlyWire FAFB v783 · strict local preparation", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:var(--drocat-navy)" class="text-sm">
                    <p class="mt-3" style="color:var(--drocat-cobalt)"><b>Automatic:</b> save a FlyWire Codex API token in the API Tokens card above, then use the <code>FAFB Data Downloads</code> card on this tab to fetch the levels you need. Skeletons fetch lazily per neuron, so the 13.9 GB bundle is optional. The manual steps below remain the offline fallback.</p>
                    <p style="color:var(--drocat-warn)"><b>Follow the converter layout exactly.</b> Download the raw Codex files; do not rename them to a generated <code>*_allneurons_*</code> filename and do not place raw files in the dataset root.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">1. Create the input folder</p>
                    <p>From the project root, create <code>datasets/flywire_FAFB_v783/downloads/</code>. Keep the original <code>.csv.gz</code> filenames in this folder.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">2. Put these raw files in <code>downloads/</code></p>
                    <p><b>Required for local pathfinding and connection analysis:</b></p>
                    <ul class="list-disc ml-4">
                        <li><code>classification.csv.gz</code></li>
                        <li><code>connections_princeton_no_threshold.csv.gz</code> (preferred), or <code>connections_princeton.csv.gz</code> / <code>connections.csv.gz</code> as the supported fallback</li>
                    </ul>
                    <p><b>Recommended metadata enrichment</b> (the converter can continue without these, but neuron labels and metadata will be incomplete):</p>
                    <ul class="list-disc ml-4">
                        <li><code>names.csv.gz</code>, <code>coordinates.csv.gz</code>, <code>neurons.csv.gz</code></li>
                        <li><code>cell_stats.csv.gz</code>, <code>consolidated_cell_types.csv.gz</code></li>
                    </ul>
                    <p><b>Optional visualization inputs:</b> <code>fafb_v783_princeton_synapse_table.csv.gz</code> for a local synapse table and <code>sk_lod1_783_healed.zip</code> for local skeletons. The converter discovers matching synapse/skeleton filenames and moves the skeleton ZIP to the dataset root.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">3. Convert the files</p>
                    <pre style="white-space:pre-wrap"><code>python src/FAFB_file_converter.py</code></pre>
                    <p>Alternatively, select <code>flywire_FAFB_v783</code> in a tool and run it; the first run invokes the same local preparation automatically.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">4. Verify before running analysis</p>
                    <p>The dataset root should contain generated files named <code>flywire_FAFB_v783_allneurons_neuron_df.parquet</code> (and CSV) and <code>flywire_FAFB_v783_merged_connections.parquet</code>. Click <b>Refresh</b> above and look for <b>ready</b> status chips.</p>
                    <p style="color:var(--drocat-warn)"><b>CAVE token (advanced, optional):</b> a live-CAVE fallback only — no download or analysis needs it, and there is no Settings input for it (set <code>tokens.cave</code> in <code>config.json</code> or the <code>CAVE_TOKEN</code> env var if ever needed; contributor-gated). Local converted tables, the Codex downloads and lazy skeletons all work without it.</p>
                </div>
                """)

            with ui.expansion("BANC v888 / v626 · automatic public-bucket preparation", icon="download").classes("w-full"):
                ui.html("""
                <div style="color:var(--drocat-navy)" class="text-sm">
                    <p><b>BANC metadata + connections now download automatically</b> from the public BANC release bucket (no token, no manual download): just select <code>banc_v888</code> or <code>banc_v626</code> in a tool and run it. Skeletons for 3D visualization fetch on demand from the same bucket.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">Choose one exact dataset identifier</p>
                    <p>Use either <code>banc_v888</code> or <code>banc_v626</code>. Never mix files from one version into the other version's folder. Legacy <code>flywire_BANC_*</code> spellings keep working.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">Everything is automatic</p>
                    <p>Selecting a BANC dataset in any tool triggers the one-time download and conversion — no manual downloads, no CAVE token. Skeletons fetch on demand from the same bucket during 3D visualization.</p>

                    <p class="mt-3 font-bold" style="color:var(--drocat-cobalt)">4. Verify before running analysis</p>
                    <p>The selected dataset root should contain <code>&lt;dataset&gt;_allneurons_neuron_df.parquet</code> (and CSV) and <code>&lt;dataset&gt;_merged_connections.parquet</code>. Click <b>Refresh</b> above and look for <b>ready</b> status chips.</p>
                    <p style="color:var(--drocat-warn)"><b>BANC has no <code>force_API_fetching</code> or CAVE fallback.</b> Pathfinding, network visualization, tabular analysis, and skeleton visualization use the BANC public-release tables/SWCs.</p>
                </div>
                """)

        # About
        with ui.card().classes("w-full drocat-card"):
            section_header("About", "info")
            ui.label("DROCAT - Drosophila Connectome Analysis Toolkit").classes("text-subtitle1 font-bold")
            ui.label(f"Version {APP_VERSION} · branch {APP_DOCS_BRANCH}").classes("text-caption drocat-muted")
            with ui.row().classes("mt-2"):
                ui.link("GitHub", APP_GITHUB_URL, new_tab=True).classes("text-primary")
                ui.link("Docs", APP_DOCS_URL, new_tab=True).classes("text-primary")

    def _update_token_status():
        neuprint_status.text = _token_status(token_state["neuprint"], token_sources["neuprint"][1])
        codex_status.text = _token_status(token_state["flywire_codex"], token_sources["flywire_codex"][1])
        _refresh_codex_gate()
        _refresh_token_reminder()

    def save_tokens():
        # Tokens are written to the gitignored config_local.json override so
        # the committed config.json stays clean on GitHub.
        config_path = PROJECT_ROOT / "config_local.json"
        entered_neuprint = (neuprint_token.value or "").strip()
        entered_codex = (codex_token.value or "").strip()
        # Base the save on the config-file tokens only: an env-var token
        # must never be copied into config_local.json by clicking Save.
        saved_tokens = _load_tokens()

        if entered_neuprint:
            saved_tokens["neuprint"] = entered_neuprint
        elif clear_blank.value:
            saved_tokens.pop("neuprint", None)
        if entered_codex:
            saved_tokens["flywire_codex"] = entered_codex
        elif clear_blank.value:
            saved_tokens.pop("flywire_codex", None)

        # Keep the versioned env map, any other sections, and token keys
        # this tab does not manage (e.g. an advanced CAVE token set
        # directly in config.json); only the two managed keys are
        # replaced.
        data = _read_config_dict(config_path)
        existing_tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
        tokens_section = dict(existing_tokens)
        for key in ("neuprint", "flywire_codex"):
            tokens_section[key] = saved_tokens.get(key, "")
        data["tokens"] = tokens_section
        try:
            config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            config_path.chmod(0o600)
            # Re-resolve from files + env so the status labels keep showing
            # the actual source after the save.
            token_sources.update(_token_sources())
            token_state.clear()
            token_state.update({key: value for key, (value, _source) in token_sources.items()})
            # Clear the client-side fields after saving so a browser DOM
            # snapshot can never retain a secret value.
            neuprint_token.value = ""
            codex_token.value = ""
            clear_blank.value = False
            _update_token_status()

            ui.notify("Tokens saved to config_local.json", type="positive")
            from ..dataset_service import get_dataset_service
            service = get_dataset_service()
            service._token = saved_tokens.get("neuprint") or None
            service._cave_token = _load_tokens().get("cave") or None
            service._cache.clear()
            service._available_neuprint = None
            service._server_datasets = {}
            service._last_fetch_time = 0
        except Exception as e:
            ui.notify(f"Failed to save: {e}", type="negative")

    async def test_connection():
        # The input is intentionally blank while a token is configured.  Use
        # the server-side value for testing unless the user entered a new one.
        token = (neuprint_token.value or "").strip() or token_state.get("neuprint", "")
        if not token:
            ui.notify("Enter a NeuPrint token first", type="warning")
            return
        ui.notify("Testing NeuPrint connection...", type="info")
        test_btn.disable()
        try:
            from neuprint import Client

            def probe_neuprint():
                client = Client(
                    "neuprint.janelia.org",
                    "hemibrain:v1.2.1",
                    token,
                )
                result = client.fetch_custom(
                    "MATCH (n:Neuron) RETURN count(n) as count LIMIT 1"
                )
                return result["count"].iloc[0] if not result.empty else 0

            count = await run.io_bound(probe_neuprint)
            ui.notify(f"Connected! {count:,} neurons in hemibrain.", type="positive")
        except Exception as e:
            ui.notify(f"Connection failed: {str(e)[:80]}", type="negative")
        finally:
            test_btn.enable()

    async def test_codex_connection():
        # Like the NeuPrint test: use the typed value if present, else the
        # saved one.
        token = (codex_token.value or "").strip() or token_state.get("flywire_codex", "")
        if not token:
            ui.notify("Enter a FlyWire Codex token first", type="warning")
            return
        ui.notify("Testing FlyWire Codex access...", type="info")
        codex_test_btn.disable()
        try:
            import codex_downloader

            ok, message = await run.io_bound(
                codex_downloader.verify_codex_token, None, token)
            ui.notify(message, type="positive" if ok else "warning")
        except Exception as e:
            ui.notify(f"Codex test failed: {str(e)[:80]}", type="negative")
        finally:
            codex_test_btn.enable()

    save_btn.on_click(save_tokens)
    test_btn.on_click(test_connection)
    codex_test_btn.on_click(test_codex_connection)


def _read_config_dict(config_path: Path) -> dict:
    """Read a config JSON object, tolerating a missing/corrupt file.

    Uses ``utf-8-sig`` because Windows PowerShell's
    ``Set-Content -Encoding UTF8`` prepends a BOM; plain ``utf-8`` would
    reject the file and the caller would silently drop the sections it
    meant to preserve (envs, output dir, ...).
    """
    if not config_path.exists():
        return {}
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_tokens() -> dict:
    """Load valid tokens from config.json (primary) then config_local.json.

    config.json wins per key; the gitignored config_local.json only fills
    entries that are empty there. A ``YOUR_*`` placeholder is treated as
    unconfigured.
    """
    tokens = {}
    for filename in ("config.json", "config_local.json"):
        config_path = PROJECT_ROOT / filename
        if not config_path.exists():
            continue
        try:
            # utf-8-sig tolerates the UTF-8 BOM that Windows editors
            # prepend to saved JSON files.
            cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except Exception:
            cfg = None
        cfg_tokens = (cfg or {}).get("tokens") if isinstance(cfg, dict) else None
        if isinstance(cfg_tokens, dict):
            for key in ("neuprint", "cave", "flywire_codex"):
                value = cfg_tokens.get(key)
                if isinstance(value, str) and value.strip() and not value.startswith("YOUR_"):
                    # First non-empty value wins: config.json is read first.
                    tokens.setdefault(key, value.strip())
    return tokens


def _token_sources() -> dict:
    """Return {key: (token, source)} for neuprint/cave with the full chain.

    Priority per key: config.json (wins per key), then the gitignored
    config_local.json (empty-entry fallback), then the environment:
    NEUPRINT_APPLICATION_CREDENTIALS (the variable neuprint-python itself
    reads) or NEUPRINT_TOKEN for NeuPrint, CAVE_TOKEN for CAVE. A
    ``YOUR_*`` placeholder is treated as unconfigured. The source string is
    shown in the Settings tab so users see where each token comes from.
    """
    sources = {}
    for key, env_names in (
        ("neuprint", ("NEUPRINT_APPLICATION_CREDENTIALS", "NEUPRINT_TOKEN")),
        ("cave", ("CAVE_TOKEN",)),
        ("flywire_codex", ("FLYWIRE_CODEX_TOKEN",)),
    ):
        token, source = "", ""
        for filename in ("config.json", "config_local.json"):
            config_path = PROJECT_ROOT / filename
            if not config_path.exists():
                continue
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
            except Exception:
                cfg = None
            cfg_tokens = (cfg or {}).get("tokens") if isinstance(cfg, dict) else None
            if isinstance(cfg_tokens, dict):
                value = cfg_tokens.get(key)
                if isinstance(value, str) and value.strip() and not value.startswith("YOUR_"):
                    token, source = value.strip(), filename
                    break
        if not token:
            for env_name in env_names:
                env_value = os.environ.get(env_name, "").strip()
                if env_value and not env_value.startswith("YOUR_"):
                    token, source = env_value, f"env var {env_name}"
                    break
        sources[key] = (token, source)
    return sources


def _token_status(token: str, source: str = "") -> str:
    """Return a non-sensitive status label naming the token's source.

    ``source`` is "config.json", "config_local.json" or "env var <NAME>"
    as resolved by :func:`_token_sources`; the token value itself is never
    shown.
    """
    if not token:
        return "not configured"
    return f"configured ({source or 'config files'})"
