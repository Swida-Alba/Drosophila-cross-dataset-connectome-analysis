"""Settings → Storage card (plan plan-settings-storage-utility.md).

Two tables (caches / exported run folders) rendered from an on-demand
scan, with preview-then-confirm removal actions wired to
``src/storage_inventory.py``. The card never scans during page build:
the first scan starts on a short timer after mount (or via Scan Now) and
runs through ``run.io_bound`` so the du over the cache tree never blocks
the event loop.

Scan coverage (nothing is scanned twice): the default output directory,
the recorded output roots (persisted via ``ui.config``; recorded
automatically when a run exports to a non-default folder, manually via
"Remember", forgettable with a chip's ✕), the remembered per-tab
overrides, the folders seen in the run history, and the typed extra
field. Every table row carries a hover tooltip with its full path(s)
(cache rows list the exact files a clear would remove) and an
"Open folder" button that reveals the folder in the system file manager.

Safety wiring (the core module enforces the hard guards):

* Cache-only (offline) runs get a warning — never a block — when the
  selection includes connection data.
* Actions are disabled while a dataset/connection/skeleton pull runs
  (``set_pull_active``, driven by the pull-state pollers).
* Run folders that are an active run's output folder are locked.
* Every removal is previewed (paths + bytes) and needs an explicit
  "Clean now"; folder deletion additionally needs a checkbox confirm.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from nicegui import run, ui

from .common import section_header

try:
    from src import storage_inventory as si
except ImportError:  # ui importable standalone (tests, tooling)
    from ...src import storage_inventory as si  # type: ignore

try:
    from ..run_state import ACTIVE_STATUSES, RUN_MANAGER
except ImportError:  # pragma: no cover
    from run_state import ACTIVE_STATUSES, RUN_MANAGER  # type: ignore

# Body slot shared by both tables: a native `title` hover on the whole
# row shows the full path(s), and a trailing button opens the folder in
# the system file manager (event → _open_dir).
_BODY_SLOT = r"""
<q-tr :props="props" :title="props.row.__path">
  <q-td auto-width>
    <q-checkbox v-model="props.selected" dense />
  </q-td>
  <q-td v-for="col in props.cols" :key="col.name" :props="props">
    {{ col.value }}
  </q-td>
  <q-td auto-width>
    <q-btn flat dense size="sm" icon="folder_open" aria-label="Open folder"
           @click.stop="$parent.$emit('open_dir', props.row.__open)">
      <q-tooltip>Open folder</q-tooltip>
    </q-btn>
  </q-td>
</q-tr>
"""

_CONNECTION_CLASSES = {"connections", "incoming"}

_CACHES_COLUMNS = [
    {"name": "cls", "label": "Class", "field": "class", "align": "left",
     "sortable": True},
    {"name": "dataset", "label": "Dataset", "field": "dataset",
     "align": "left", "sortable": True},
    {"name": "size", "label": "Size", "field": "size", "align": "right"},
    {"name": "last_used", "label": "Last used", "field": "last_used",
     "align": "left", "sortable": True},
    {"name": "rebuild", "label": "Rebuild cost / notes", "field": "rebuild",
     "align": "left"},
]

_RUNS_COLUMNS = [
    {"name": "name", "label": "Run folder", "field": "name",
     "align": "left", "sortable": True},
    {"name": "tool", "label": "Tool", "field": "tool", "align": "left",
     "sortable": True},
    {"name": "stamp", "label": "Timestamp", "field": "stamp",
     "align": "left", "sortable": True},
    # Formatted byte strings are the display fields; Quasar sorts by
    # `field`, and a string sort on '9.9 MB' would be wrong, so these
    # three columns are display-only.
    {"name": "size", "label": "Total", "field": "size", "align": "right"},
    {"name": "source", "label": "Source data", "field": "source",
     "align": "right"},
    {"name": "deliverable", "label": "Deliverables", "field": "deliverable",
     "align": "right"},
]


def _fmt_ts(raw: str) -> str:
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "—"


def _fmt_mtime(mtime: float) -> str:
    if not mtime:
        return "—"
    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, OverflowError):
        return "—"


def _truncate_lines(values: List[str], limit: int = 12) -> str:
    head = "\n".join(values[:limit])
    if len(values) > limit:
        head += f"\n… and {len(values) - limit} more"
    return head


class StorageCard:
    """State + UI for the Settings → Storage card."""

    def __init__(self, puller, skeleton_puller):
        self.puller = puller
        self.skeleton_puller = skeleton_puller
        self.cache_items: List = []
        self.run_items: List = []
        self._pull_active = False
        self._scanning = False
        self.extra_root: Optional[ui.input] = None
        self.caches_table: Optional[ui.table] = None
        self.runs_table: Optional[ui.table] = None
        self.scan_btn = None
        self.clear_btn = None
        self.prune_btn = None
        self.delete_btn = None
        self.status_label = None
        self.result_label = None
        self._build()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self):
        with ui.card().classes("w-full drocat-card"):
            section_header("Storage", "storage")
            ui.label(
                "Inspect and reclaim disk space. Nothing is ever removed "
                "automatically — every action previews the exact paths and "
                "bytes first and runs only when you confirm. Cache data is "
                "refetched on demand; deleted run folders are gone."
            ).classes("text-caption drocat-muted")

            with ui.row().classes("items-center gap-3 w-full")\
                    .style("flex-wrap: wrap"):
                self.scan_btn = ui.button(
                    "Scan Now", icon="refresh", on_click=self.start_scan
                ).props("outline")
                self.extra_root = ui.input(
                    label="Additional output folder to scan",
                    placeholder="optional",
                ).classes("drocat-input").style("min-width: 300px")\
                    .tooltip(
                        "The default output directory, remembered tab "
                        "folders and recorded output roots are always "
                        "scanned. Type one more folder here to include "
                        "it in the next scan; 'Remember' keeps it for "
                        "every future scan.")
                self.remember_btn = ui.button(
                    "Remember", icon="bookmark_add",
                    on_click=self._remember_folder,
                ).props("outline").tooltip(
                    "Record the folder above so it is always scanned "
                    "(and shown below with a ✕ to forget it).")
                self.status_label = ui.label("").classes(
                    "text-caption drocat-muted")

            with ui.row().classes("items-center gap-2 w-full")\
                    .style("flex-wrap: wrap"):
                ui.label("Recorded output roots:").classes(
                    "text-caption drocat-muted")
                self.roots_row = ui.row().classes(
                    "items-center gap-1").style("flex-wrap: wrap")
            self._render_roots()

            with ui.expansion("Caches", icon="cached").classes("w-full"):
                self.caches_table = ui.table(
                    columns=_CACHES_COLUMNS, rows=[], row_key="key",
                    selection="multiple", pagination=12,
                    on_select=self._sync_action_buttons,
                ).classes("w-full").props("dense flat bordered")
                self.caches_table.add_slot("body", _BODY_SLOT)
                self.caches_table.on("open_dir", self._open_dir)
                with ui.row().classes("items-center gap-2"):
                    self.clear_btn = ui.button(
                        "Clear Selected…", icon="delete_sweep",
                        on_click=self._confirm_clear,
                    ).props("outline color=negative").set_enabled(False)
                    ui.label("Selection is removed only after the preview "
                             "is confirmed.").classes(
                        "text-caption drocat-muted")

            with ui.expansion(
                    "Exported data", icon="folder_delete").classes(
                    "w-full"):
                self.runs_table = ui.table(
                    columns=_RUNS_COLUMNS, rows=[], row_key="name",
                    selection="multiple", pagination=12,
                    on_select=self._sync_action_buttons,
                ).classes("w-full").props("dense flat bordered")
                self.runs_table.add_slot("body", _BODY_SLOT)
                self.runs_table.on("open_dir", self._open_dir)
                with ui.row().classes("items-center gap-2"):
                    self.prune_btn = ui.button(
                        "Prune Source Data…", icon="content_cut",
                        on_click=self._confirm_prune,
                    ).props("outline").set_enabled(False)
                    self.delete_btn = ui.button(
                        "Delete Folder…", icon="delete_forever",
                        on_click=self._confirm_delete_folder,
                    ).props("outline color=negative").set_enabled(False)
                    ui.label(
                        "Prune removes the bodyId-level source data of "
                        "registered tools (deliverables stay). Delete "
                        "removes whole run folders — double-confirmed."
                    ).classes("text-caption drocat-muted")

            self.result_label = ui.label("").classes("text-caption")

        # First scan shortly after mount (page build stays untouched);
        # run.io_bound keeps the du off the event loop. Guarded: when the
        # user reloads before the timer fires, the old client's elements
        # are deleted and any update would raise.
        async def _initial_scan():
            try:
                await self.start_scan()
            except Exception:
                pass  # stale client after a page reload — nothing to scan

        ui.timer(1.5, _initial_scan, once=True)

    # ------------------------------------------------------------------
    # Interlocks
    # ------------------------------------------------------------------

    def set_pull_active(self, active: bool) -> None:
        """Called by the pull-state pollers on the Settings tab.

        The pollers fire twice a second, so this acts only on state
        CHANGES — a repeated 'inactive' call must never wipe the scan
        status line.
        """
        active = bool(active)
        if active == self._pull_active:
            return
        self._pull_active = active
        if self._pull_active:
            for btn in (self.scan_btn, self.clear_btn, self.prune_btn,
                        self.delete_btn):
                if btn is not None:
                    btn.set_enabled(False)
            if self.status_label is not None:
                self.status_label.set_text(
                    "Paused while a dataset/skeleton pull is running.")
            return
        if not self._scanning and self.scan_btn is not None:
            self.scan_btn.set_enabled(True)
        if self.status_label is not None:
            self.status_label.set_text("")
        self._sync_action_buttons()

    def _sync_action_buttons(self, _event=None) -> None:
        caches_sel = self.caches_table.selected if \
            self.caches_table is not None else []
        runs_sel = self.runs_table.selected if \
            self.runs_table is not None else []
        if self.clear_btn is not None:
            self.clear_btn.set_enabled(bool(caches_sel))
        if self.prune_btn is not None:
            self.prune_btn.set_enabled(any(
                r.get("registered") and (r.get("source_bytes") or 0) > 0
                for r in runs_sel if not r.get("locked")))
        if self.delete_btn is not None:
            self.delete_btn.set_enabled(any(
                not r.get("locked") for r in runs_sel))

    # ------------------------------------------------------------------
    # Recorded output roots
    # ------------------------------------------------------------------

    def _render_roots(self) -> None:
        """Rebuild the recorded-root chips (✕ forgets the root)."""
        from ..config import get_storage_scan_roots, remove_storage_scan_root
        self.roots_row.clear()
        roots = get_storage_scan_roots()
        with self.roots_row:
            if not roots:
                ui.label("none yet — they are recorded automatically when "
                         "a run uses a non-default folder").classes(
                    "text-caption drocat-muted")
            for root in roots:
                # This NiceGUI signals chip removal via on_value_change
                # (no on_removal kwarg exists here).
                ui.chip(
                    root, removable=True, icon="folder",
                    on_value_change=lambda _e, r=root: self._forget_root(r),
                ).tooltip("Scanned by the Storage card; ✕ stops scanning "
                          "(nothing is deleted).")

    def _forget_root(self, root: str) -> None:
        from ..config import remove_storage_scan_root
        remove_storage_scan_root(root)
        ui.notify(f"Stopped scanning {root} (nothing was deleted)",
                  type="info")
        self._render_roots()

    async def _remember_folder(self, _event=None) -> None:
        from ..config import add_storage_scan_root
        value = (self.extra_root.value or "").strip()
        if not value:
            ui.notify("Type a folder first", type="warning")
            return
        if not Path(value).expanduser().is_dir():
            ui.notify("That folder does not exist (check the path)",
                      type="negative")
            return
        if not await run.io_bound(add_storage_scan_root, value):
            ui.notify("Could not record that folder (check the path)",
                      type="negative")
            return
        self.extra_root.value = ""
        self._render_roots()
        ui.notify("Folder recorded — it will be scanned from now on",
                  type="positive")
        await self.start_scan()

    async def _open_dir(self, event) -> None:
        """Open a row's folder in the system file manager."""
        path = event.args
        if isinstance(path, dict):
            path = path.get("path")
        if not path:
            return
        from ..runner import open_folder
        await run.io_bound(open_folder, str(path))

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _output_roots(self) -> List[str]:
        """Every location the scanner covers: the default output dir,
        recorded output roots, remembered tab overrides, folders seen in
        the run history, plus the typed extra field. Deduplicated."""
        from ..config import (
            TAB_OUTPUT_DIRS_KEY,
            get_default_output_dir,
            get_storage_scan_roots,
            load_local_config,
        )
        roots: List[str] = [get_default_output_dir()]
        roots.extend(get_storage_scan_roots())
        try:
            overrides = load_local_config().get(TAB_OUTPUT_DIRS_KEY, {})
            if isinstance(overrides, dict):
                roots.extend(
                    value for value in overrides.values()
                    if isinstance(value, str) and value.strip())
        except Exception:
            pass
        try:
            for record in RUN_MANAGER.recent_runs(500):
                folder = record.get("output_dir")
                if folder:
                    roots.append(str(folder))
        except Exception:
            pass
        extra = (self.extra_root.value or "").strip() if \
            self.extra_root is not None else ""
        if extra:
            roots.append(extra)
        seen: set = set()
        unique: List[str] = []
        for root in roots:
            try:
                resolved = str(Path(root).expanduser().resolve())
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        return unique

    def _active_run_folders(self) -> set:
        try:
            records = RUN_MANAGER.recent_runs(200)
        except Exception:
            return set()
        active = set()
        for record in records:
            if record.get("status") in ACTIVE_STATUSES:
                folder = record.get("output_folder")
                if folder:
                    try:
                        active.add(str(Path(folder).resolve()))
                    except OSError:
                        continue
        return active

    async def start_scan(self, _event=None) -> None:
        if self._scanning:
            return
        if self._pull_active:
            ui.notify("Finish the running pull before scanning",
                      type="warning")
            return
        self._scanning = True
        self.scan_btn.set_enabled(False)
        self.status_label.set_text("Scanning…")

        roots = list(self._output_roots())

        def _scan_work():
            caches = si.scan_caches()
            runs = si.scan_run_folders(roots) if roots else []
            return caches, runs, roots

        try:
            caches, runs, used_roots = await run.io_bound(_scan_work)
        except Exception as exc:
            self._scanning = False
            self.scan_btn.set_enabled(True)
            self.status_label.set_text("")
            ui.notify(f"Scan failed: {exc}", type="negative")
            return

        self.cache_items = caches
        self.run_items = runs
        active_folders = self._active_run_folders()

        cache_rows = []
        for item in caches:
            notes = item.rebuild_note
            if item.warnings:
                notes = f"⚠ {notes} — {'; '.join(item.warnings)}"
            shown = [str(p) for p in item.paths]
            tooltip = "\n".join(shown[:12]) + (
                f"\n… and {len(shown) - 12} more" if len(shown) > 12 else "")
            cache_rows.append({
                "key": item.key,
                "class": item.label,
                "dataset": item.dataset or "—",
                "size_bytes": item.size_bytes,
                "size": si.format_bytes(item.size_bytes),
                "last_used": _fmt_mtime(item.last_used),
                "rebuild": notes,
                "__path": tooltip or item.open_path or "",
                "__open": item.open_path or "",
            })
        self.caches_table.rows = cache_rows
        self.caches_table.selected = []
        self.caches_table.update()

        run_rows = []
        for item in sorted(self.run_items, key=lambda r: r.timestamp,
                           reverse=True):
            try:
                locked = str(item.path.resolve()) in active_folders
            except OSError:
                locked = False
            run_rows.append({
                "name": item.name,
                "path": str(item.path),
                "tool": item.tool or "(unregistered)",
                "stamp": _fmt_ts(item.timestamp),
                "size_bytes": item.size_bytes,
                "size": si.format_bytes(item.size_bytes),
                "source_bytes": item.source_bytes,
                "source": ("—" if item.source_bytes < 0
                           else si.format_bytes(item.source_bytes)),
                "deliverable_bytes": item.deliverable_bytes,
                "deliverable": ("—" if item.deliverable_bytes < 0
                                else si.format_bytes(
                                    item.deliverable_bytes)),
                "registered": item.registered,
                "locked": locked,
                "__path": str(item.path),
                "__open": str(item.path),
            })
        self.runs_table.rows = run_rows
        self.runs_table.selected = []
        self.runs_table.update()

        total_cache = sum(i.size_bytes for i in caches)
        self._scanning = False
        self.scan_btn.set_enabled(not self._pull_active)
        locked_note = f"; {len(active_folders)} locked by active runs" \
            if active_folders else ""
        self.status_label.set_text(
            f"{len(caches)} cache rows ({si.format_bytes(total_cache)}), "
            f"{len(runs)} run folders across {len(used_roots)} root(s)"
            f"{locked_note} — scanned "
            f"{datetime.now().strftime('%H:%M:%S')}")
        self._sync_action_buttons()

    # ------------------------------------------------------------------
    # Selection helpers + guards
    # ------------------------------------------------------------------

    def _selected_cache_items(self) -> List:
        selected_keys = {r["key"] for r in
                         (self.caches_table.selected or [])}
        return [i for i in self.cache_items if i.key in selected_keys]

    def _selected_run_rows(self, *, require_registered: bool) -> List[dict]:
        rows = [r for r in (self.runs_table.selected or [])
                if not r.get("locked")]
        if require_registered:
            rows = [r for r in rows if r.get("registered")]
        return rows

    def _interlocked(self) -> bool:
        if self.puller.running or self.skeleton_puller.running:
            ui.notify(
                "Finish or cancel the running dataset/skeleton pull "
                "before removing storage", type="warning")
            return True
        return False

    def _cache_only_warning(self, items: List) -> str:
        from ..config import get_user_default
        if not get_user_default("cache_only"):
            return ""
        if any(i.cls in _CONNECTION_CLASSES for i in items):
            return ("Settings → Cache Only (Offline) is enabled: tools "
                    "will not refetch these connection files until it is "
                    "turned off.")
        return ""

    # ------------------------------------------------------------------
    # Cache clear flow
    # ------------------------------------------------------------------

    async def _confirm_clear(self, _event=None) -> None:
        if self._interlocked():
            return
        items = self._selected_cache_items()
        if not items:
            ui.notify("Select cache rows first", type="warning")
            return
        paths = [p for item in items for p in item.paths if p.exists()]
        total = sum(i.size_bytes for i in items)
        cache_only_note = self._cache_only_warning(items)

        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label("Clear caches").classes("text-subtitle1")
            ui.label(f"{len(items)} cache row(s) · "
                     f"{si.format_bytes(total)}").classes("text-caption")
            ui.label(_truncate_lines([str(p) for p in paths])).classes(
                "text-caption").style(
                "font-family: monospace; white-space: pre-wrap; "
                "max-height: 180px; overflow-y: auto")
            if cache_only_note:
                ui.label(f"⚠ {cache_only_note}").classes(
                    "text-caption drocat-warn")
            ui.label(
                "Refetched on demand. The clear takes full effect after "
                "the app restarts (in-process caches may still serve or "
                "re-save cleared data until then)."
            ).classes("text-caption drocat-muted")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("outline")
                ui.button("Clean Now", icon="delete_sweep",
                          color="negative",
                          on_click=lambda: dialog.submit(True))
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return
        result = await run.io_bound(si.delete_paths, paths)
        self._report(result, restart_note=True)
        await self.start_scan()

    # ------------------------------------------------------------------
    # Run-folder flows
    # ------------------------------------------------------------------

    async def _confirm_prune(self, _event=None) -> None:
        if self._interlocked():
            return
        rows = [r for r in self._selected_run_rows(require_registered=True)
                if (r.get("source_bytes") or 0) > 0]
        if not rows:
            ui.notify(
                "Select registered run folders with source data first",
                type="warning")
            return
        total = sum(r["source_bytes"] for r in rows)
        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label("Prune source data").classes("text-subtitle1")
            ui.label(
                f"{len(rows)} run folder(s) · {si.format_bytes(total)} of "
                f"source data will be removed; summaries, reports and "
                f"plots stay.").classes("text-caption")
            ui.label(_truncate_lines([r["name"] for r in rows])).classes(
                "text-caption").style("white-space: pre-wrap")
            ui.label(
                "This runs the Compact prune regardless of the runs' "
                "original output mode — you are choosing it now. Each "
                "folder's cleanup_audit.json records the removals."
            ).classes("text-caption drocat-muted")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("outline")
                ui.button("Prune Now", icon="content_cut",
                          color="negative",
                          on_click=lambda: dialog.submit(True))
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return
        paths = [r["path"] for r in rows]

        def _work():
            outcomes = []
            for path in paths:
                try:
                    audit = si.prune_run_folder(path)
                    outcomes.append((
                        path,
                        int(audit.get("bytes_reclaimed_this_pass") or 0),
                        None))
                except Exception as exc:
                    outcomes.append((path, 0, str(exc)))
            return outcomes

        outcomes = await run.io_bound(_work)
        ok = [(p, b) for p, b, err in outcomes if err is None]
        failed = [(p, err) for p, b, err in outcomes if err]
        if failed:
            ui.notify(
                f"Pruned {len(ok)} folder(s); {len(failed)} failed "
                f"(first: {failed[0][1]})",
                type="warning" if ok else "negative")
        else:
            ui.notify(
                f"Pruned {len(ok)} folder(s), reclaimed "
                f"{si.format_bytes(sum(b for _, b in ok))}",
                type="positive")
        self.result_label.set_text(
            "Prune removed source data from: "
            + ", ".join(Path(p).name for p, _ in ok[:5])
            + (" …" if len(ok) > 5 else ""))
        await self.start_scan()

    async def _confirm_delete_folder(self, _event=None) -> None:
        if self._interlocked():
            return
        rows = self._selected_run_rows(require_registered=False)
        if not rows:
            ui.notify("Select run folders first", type="warning")
            return
        total = sum(r.get("size_bytes") or 0 for r in rows)
        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label("Delete run folders").classes("text-subtitle1")
            ui.label(
                f"{len(rows)} run folder(s) · {si.format_bytes(total)} "
                f"will be permanently removed, deliverables included."
            ).classes("text-caption")
            ui.label(_truncate_lines([r["name"] for r in rows])).classes(
                "text-caption").style("white-space: pre-wrap")
            understand = ui.checkbox(
                "I understand these run folders cannot be recovered")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("outline")

                def _try_confirm():
                    if not understand.value:
                        ui.notify("Tick the confirmation box first",
                                  type="warning")
                        return
                    dialog.submit(True)

                ui.button("Delete Forever", icon="delete_forever",
                          color="negative", on_click=_try_confirm)
        dialog.open()
        confirmed = await dialog
        if not confirmed:
            return
        paths = [r["path"] for r in rows]
        result = await run.io_bound(si.delete_paths, paths)
        self._report(result, restart_note=False)
        await self.start_scan()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _report(self, result: dict, *, restart_note: bool) -> None:
        removed = result.get("removed", [])
        refused = result.get("refused", [])
        reclaimed = result.get("bytes_reclaimed", 0)
        message = (f"Removed {len(removed)} item(s), reclaimed "
                   f"{si.format_bytes(reclaimed)}.")
        if refused:
            message += f" {len(refused)} path(s) refused: " + "; ".join(
                f"{r['path']} ({r['reason']})" for r in refused[:3])
            if len(refused) > 3:
                message += f" … and {len(refused) - 3} more"
        if restart_note and removed:
            message += " Full effect after the app restarts."
        self.result_label.set_text(message)
        ui.notify(message, type="positive" if removed else "warning")


def create_storage_card(puller, skeleton_puller) -> StorageCard:
    """Build the Settings → Storage card in place."""
    return StorageCard(puller=puller, skeleton_puller=skeleton_puller)
