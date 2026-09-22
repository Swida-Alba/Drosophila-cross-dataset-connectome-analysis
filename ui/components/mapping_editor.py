"""Mapping grid editor: a table grid for one LabelMapper side.

Rows are custom groups (a name column plus one column per dataset); cells
hold comma-separated neuron identifiers. The grid edits the LabelMapper JSON
schema directly:

    {
        "custom_label": ["grp1", "grp2"],
        "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]
    }
"""
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

from nicegui import ui

from .. import group_history, history_store, mapping_store
from ..config import PROJECT_ROOT
from ..type_suggestions import dataset_suggestions
from .common import neuron_list_input
from .custom_grouper import LiteCustomGrouper

_LABEL_KEYS = ("custom_label", "std_label")

NONE_MAPPING = "(No custom mapping)"

# Transient per-run exports of the inline grouper (canonical LabelMapper
# format — directly loadable by scripts); newest per tab wins.
_INLINE_DIR = PROJECT_ROOT / "cache" / "user_mappings" / "_inline"


def inline_mapping_path(tab_key: str) -> str:
    """Timestamped export path for one tab's inline groups."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(_INLINE_DIR / f"{tab_key}_{stamp}.json")


# Local documentation served by the app (ui/app.py registers /docs).
# The guide's "UI workflow" section explains the preset workflow; it is an
# HTML page (like the other ui_guides) so it renders in the browser instead
# of downloading as a markdown file.
MAPPING_GUIDE_URL = "docs/ui_guides/label_mapper.html"


def mapping_selector(
    label: str = "Custom Type Mapping",
    hint: str = "Reuse a saved custom type mapping (manage them in the Settings tab).",
    show_instructions: bool = True,
) -> ui.select:
    """Dropdown of saved mapping presets; defaults to the active one.

    Options refresh when the dropdown is opened, so presets created after
    the tab was built still appear. When *show_instructions* is true a small
    link to the LabelMapper guide (custom-grouping instructions) is rendered
    right below the dropdown.
    """
    options = [NONE_MAPPING] + mapping_store.list_mappings()
    active = mapping_store.get_active_mapping() or NONE_MAPPING
    sel = ui.select(
        options=options,
        value=active if active in options else NONE_MAPPING,
        label=label,
    ).classes("w-full drocat-select").tooltip(hint)

    def _refresh_options():
        current = sel.value
        available = [NONE_MAPPING] + mapping_store.list_mappings()
        if current not in available:
            sel.value = NONE_MAPPING
        sel.options = available

    sel.on("click", _refresh_options)

    if show_instructions:
        with ui.row().classes("items-center w-full"):
            ui.link(
                "📖 Custom grouping instructions",
                MAPPING_GUIDE_URL,
                new_tab=True,
            ).classes("text-caption text-primary")
    return sel


def selected_mapping_file_path(selection: Optional[str]) -> Optional[str]:
    """Export path for the chosen preset, or None when no mapping is chosen."""
    if not selection or selection == NONE_MAPPING:
        return None
    return mapping_store.mapping_file_path(selection)


def custom_grouping_block(
    label: str = "Custom Grouping",
    hint: str = (
        "Open the grouping panel: reuse a saved custom type mapping (manage "
        "them in the Settings tab) or define groups inline for this query."
    ),
    datasets_provider: Optional[Callable[[], List[str]]] = None,
    require_names: bool = False,
    tab_key: str = "tab",
    watch_elements: Optional[List] = None,
    query_inputs: Optional[Dict[str, object]] = None,
    panel_title: Optional[str] = None,
    row_action_renderers: Optional[List[Callable[[], None]]] = None,
    dataset_selector_renderer: Optional[Callable[[], object]] = None,
) -> Tuple[ui.button, ui.dialog, Callable[[], Tuple[Optional[str], bool]]]:
    """Custom Grouping button + popup grouping panel.

    Returns ``(open_button, grouper_dialog, resolve_mapping_path)``:

    - The control is a BUTTON: clicking it opens the panel directly (same
      popup design as the "See available neurons" viewer). The panel hosts
      an optional embedded dataset selector, saved-group loader, and inline
      group board; the button label mirrors the current state (none / inline).
    - ``resolve_mapping_path()`` -> ``(path, ok)``. ``ok=False`` means the
      run must be aborted (errors were already notified). A selected preset
      wins; otherwise a non-empty inline board is validated, exported to
      the canonical JSON (script-loadable), recorded in the group history,
      and its path returned. No preset + empty board: ``(None, True)``.
    """
    grouper = LiteCustomGrouper(tab_key=tab_key, require_names=require_names,
                                query_inputs=query_inputs,
                                row_action_renderers=row_action_renderers)

    open_button = ui.button(icon="group_work").props(
        "outline no-caps").classes("w-full").style(
        "justify-content: flex-start; min-height: 40px")
    open_button.tooltip(hint)

    dialog = ui.dialog()
    dataset_selector = None
    # Persistent: the panel must not close on outside-click / ESC, because the
    # suggestion menus portal to the body (outside the dialog DOM) and clicking
    # them would otherwise dismiss the panel mid-edit. It closes only via X.
    dialog.props("persistent")
    with dialog:
        with ui.card().classes("w-[min(98vw,1400px)] max-w-none"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.label(panel_title or label).classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props(
                    "flat round dense")
            if dataset_selector_renderer is not None:
                with ui.column().classes("w-full gap-1"):
                    dataset_selector = dataset_selector_renderer()
            with ui.row().classes("w-full items-center gap-3"):
                # The former "preset" control is now a HISTORY loader: a button
                # opening a menu of previously-used groups (recorded
                # automatically when a group is pushed to a query / exported).
                # Each entry has a load action and an "x" that removes it from
                # the history after a confirmation popup; nothing is named or
                # saved manually.
                history_button = ui.button(
                    "Load a saved group (history)", icon="history"
                ).props("outline no-caps").style(
                    "min-width: 320px; justify-content: flex-start")
                with history_button:
                    with ui.menu() as history_menu:
                        history_list = ui.column().classes("gap-0").style(
                            "min-width: 300px")
                ui.link(
                    "📖 Custom grouping instructions",
                    MAPPING_GUIDE_URL,
                    new_tab=True,
                ).classes("text-caption text-primary")
            with ui.column().classes("w-full gap-1") as grouper_container:
                pass
    watched_elements = list(watch_elements or [])
    if dataset_selector is not None:
        watched_elements.append(dataset_selector)

    def _set_embedded_datasets(values: List[str]) -> None:
        if dataset_selector is not None:
            dataset_selector.value = list(values)

    grouper.create(
        grouper_container,
        datasets_provider or (lambda: []),
        watch_elements=watched_elements,
        datasets_setter=(
            _set_embedded_datasets if dataset_selector is not None else None
        ),
    )

    # Confirmation popup for history removal (overlays the panel dialog).
    remove_dialog = ui.dialog()
    with remove_dialog:
        with ui.card().classes("q-pa-md"):
            remove_msg = ui.label("").classes("text-body2")
            with ui.row().classes("justify-end gap-2 q-mt-sm"):
                ui.button("Cancel", on_click=remove_dialog.close).props(
                    "flat dense")
                ui.button("Remove", icon="delete",
                          on_click=lambda: _confirm_remove()).props(
                    "flat dense color=negative")

    # Test/DOM hooks.
    dialog.inline_grouper = grouper
    dialog.history_menu = history_menu
    dialog.dataset_selector = dataset_selector

    def _render_history() -> None:
        history_list.clear()
        with history_list:
            recent = group_history.list_recent()
            if not recent:
                ui.label("No group history yet.").classes(
                    "text-caption drocat-muted px-3 py-2")
                return
            for lab in recent:
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.button(
                        lab, on_click=lambda _e, l=lab: _load_group(l)
                    ).props("flat dense no-caps align=left").classes("flex-grow")
                    ui.button(
                        icon="close",
                        on_click=lambda _e, l=lab: _ask_remove(l)
                    ).props("flat dense round").tooltip("Remove from history")

    def _load_group(lab: str) -> None:
        rec = group_history.valid_labels().get(lab)
        if rec:
            # The first panel open seeds one blank editing row. Reuse that
            # placeholder for a loaded history group, but preserve it once a
            # user has entered either a name or a member.
            grouper._collect_rows()
            grouper.handle.load_saved_row(lab, rec.get("members") or {})
            grouper.ensure_row_datasets()
            # Render directly: resync would collect the old widgets one more
            # time and could overwrite the newly loaded row with their blank
            # values before the replacement inputs are created.
            grouper._render()
            _update_button_label()
        history_menu.close()

    def _ask_remove(lab: str) -> None:
        pending_remove["label"] = lab
        remove_msg.set_text(f"Remove group '{lab}' from the history?")
        remove_dialog.open()

    def _confirm_remove() -> None:
        lab = pending_remove["label"]
        if lab:
            group_history.remove_label(lab)
            history_store.remove(lab)
            ui.notify(f"Group '{lab}' removed from history", type="positive")
            pending_remove["label"] = None
            _render_history()
        remove_dialog.close()

    pending_remove = {"label": None}
    dialog.load_history_group = _load_group
    dialog.request_remove_history = _ask_remove
    dialog.confirm_remove_history = _confirm_remove

    def _group_count() -> int:
        return sum(
            1 for row in grouper.handle.rows
            if any(vals for vals in (row.get("cells") or {}).values()))

    def _update_button_label() -> None:
        n = _group_count()
        open_button.text = (
            f"{label} · inline · {n} group(s)" if n else f"{label} · none")

    def _open_panel() -> None:
        dialog.open()

    def _on_dialog_toggle(event) -> None:
        if event.value:  # opening
            _render_history()
            # First-open convenience: seed a single empty row for editing, but
            # only when the board is empty (never on top of existing groups).
            if not grouper.handle.rows:
                grouper.handle.add_row()
            grouper.resync()
        _update_button_label()

    history_button.on_click(lambda: (_render_history(), history_menu.open()))
    open_button.on_click(_open_panel)
    dialog.on_value_change(_on_dialog_toggle)
    _update_button_label()

    def _query_history_groups() -> Tuple[List[str], List[str]]:
        """Materialize custom labels used by the normal query inputs.

        The source/target history intentionally stores a chip as a string.
        If that string is a known custom-group label, restore its member table
        onto the active inline board before exporting the mapping file. This
        makes selecting a custom item directly from a source/target history
        menu equivalent to loading it in the grouping panel first.

        Returns ``(loaded_labels, labels_without_current_members)``. The
        second list prevents a label from silently falling through to the
        ordinary neuron-type resolver when it has no members for the selected
        dataset(s).
        """
        requested = {}
        valid_custom_labels = group_history.valid_labels()
        for query_input in grouper.query_inputs.values():
            getter = getattr(query_input, "get_value", None)
            if not callable(getter):
                continue
            try:
                result = getter()
            except Exception:  # pragma: no cover - defensive UI boundary
                continue
            values = (
                result[1]
                if isinstance(result, (tuple, list)) and len(result) > 1
                else result
            )
            if values is None:
                continue
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            for value in values:
                label_value = str(value or "").strip()
                if label_value in requested:
                    continue
                record = valid_custom_labels.get(label_value)
                if record is not None:
                    requested[label_value] = record

        if not requested:
            return [], []

        grouper._collect_rows()
        existing = {
            str(row.get("name") or "").strip()
            for row in grouper.handle.rows
        }
        for label_value, record in requested.items():
            # Preserve an explicitly edited inline row with the same label.
            if label_value not in existing:
                grouper.handle.add_row(
                    label_value, record.get("members") or {})
        grouper.resync()
        _update_button_label()

        datasets = grouper.datasets()
        missing = []
        if datasets:
            for label_value in requested:
                row = next(
                    (r for r in grouper.handle.rows
                     if str(r.get("name") or "").strip() == label_value),
                    None,
                )
                if row is None or not any(
                    (row.get("cells") or {}).get(ds) for ds in datasets
                ):
                    missing.append(label_value)
        return list(requested), missing

    def resolve_mapping_path() -> Tuple[Optional[str], bool]:
        _loaded, missing = _query_history_groups()
        if missing:
            for label_value in missing:
                ui.notify(
                    f"Custom group '{label_value}' has no members for the "
                    "selected dataset(s)",
                    type="negative",
                )
            return None, False
        if grouper.is_empty():
            return None, True
        errors = grouper.validate()
        if errors:
            for err in errors:
                ui.notify(err, type="negative")
            return None, False
        path = grouper.export_to(inline_mapping_path(tab_key))
        # History stores user intent at export time (label-keyed,
        # cell-granularity upsert; auto-named groups are skipped).
        payload = grouper.history_payload()
        group_history.record(payload, origin="inline")
        for label_value, cells in payload:
            if not str(label_value or "").strip():
                continue
            history_store.record(
                [str(label_value)],
                custom_values=[str(label_value)],
                datasets=[
                    dataset for dataset, values in (cells or {}).items()
                    if any(str(value).strip() for value in (values or []))
                ],
            )
        # The query history may have been recorded just before this mapping
        # was resolved. Mark matching values now so a later group removal can
        # prune stale entries even if the original run predates the registry
        # record.
        history_store.mark_custom(group_history.valid_labels())
        return path, True

    return open_button, dialog, resolve_mapping_path



class MappingGridEditor:
    """Editable grid for one mapping side (source / target / intermediate)."""

    def __init__(self, side: str):
        self.side = side
        self._container: Optional[ui.column] = None
        self._datasets: List[str] = []
        self._groups: List[str] = []
        self._cells: Dict[str, Dict[int, str]] = {}  # dataset -> {row: "a, b"}
        self._available: List[str] = []
        self._cell_widgets: Dict[str, Dict[int, ui.element]] = {}

    # -- data ---------------------------------------------------------------

    def set_data(self, data: Optional[dict]) -> None:
        """Load a LabelMapper side dict into the grid (replaces current)."""
        self._datasets = []
        self._groups = []
        self._cells = {}
        self._cell_widgets = {}
        if not data:
            self._rerender()
            return
        labels = data.get("custom_label") or data.get("std_label") or []
        self._groups = [str(g) for g in labels]
        for ds, groups in data.items():
            if ds in _LABEL_KEYS:
                continue
            self._datasets.append(ds)
            self._cells[ds] = {
                i: ", ".join(str(x) for x in group)
                for i, group in enumerate(groups or [])
            }
        for i in range(len(self._groups)):
            for ds in self._datasets:
                self._cells.setdefault(ds, {}).setdefault(i, "")
        self._available = [d for d in self._available if d not in self._datasets]
        self._rerender()

    @staticmethod
    def _parse_cell_text(value: str) -> List[str]:
        """Convert the legacy comma-separated cell value to input chips."""
        return [part.strip() for part in str(value or "").split(",") if part.strip()]

    @staticmethod
    def _encode_cell_values(values) -> str:
        """Keep the stored LabelMapper representation comma-separated."""
        return ", ".join(
            str(value).strip() for value in (values or []) if str(value).strip()
        )

    @staticmethod
    def _cell_suggest(dataset: str):
        """Dataset-aware type-ahead for one group-member cell.

        Bound per dataset because the grid keeps a column per LabelMapper side, so
        a cell suggests from its own dataset's pools — the same dataset-aware
        search the inline custom grouper wires for its member cells.
        """
        return partial(dataset_suggestions, dataset=dataset, search_columns="auto")

    def _collect_widgets(self) -> None:
        """Copy live chip values back into the legacy cell store."""
        for ds, widgets in self._cell_widgets.items():
            for idx, widget in widgets.items():
                if hasattr(widget, "get_value"):
                    self._cells.setdefault(ds, {})[idx] = self._encode_cell_values(
                        widget.get_value()[1]
                    )

    def get_data(self) -> dict:
        """Build the LabelMapper side dict from the grid state."""
        self._collect_widgets()
        data = {"custom_label": [g.strip() for g in self._groups]}
        for ds in self._datasets:
            data[ds] = [
                [x.strip() for x in self._cells[ds].get(i, "").split(",") if x.strip()]
                for i in range(len(self._groups))
            ]
        return data

    def is_empty(self) -> bool:
        self._collect_widgets()
        return not self._groups or all(
            not self._cells.get(ds, {}).get(i, "").strip()
            for i in range(len(self._groups))
            for ds in self._datasets
        )

    # -- structure ----------------------------------------------------------

    def _add_group(self) -> None:
        self._collect_widgets()
        idx = len(self._groups)
        self._groups.append(f"Group_{idx + 1}")
        for ds in self._datasets:
            self._cells.setdefault(ds, {})[idx] = ""
        self._rerender()

    def _remove_group(self, idx: int) -> None:
        self._collect_widgets()
        if 0 <= idx < len(self._groups):
            self._groups.pop(idx)
            for ds in self._datasets:
                self._cells[ds].pop(idx, None)
        self._rerender()

    def _add_dataset(self, ds: Optional[str]) -> None:
        self._collect_widgets()
        if not ds or ds in self._datasets:
            return
        self._datasets.append(ds)
        self._cells[ds] = {i: "" for i in range(len(self._groups))}
        self._available = [d for d in self._available if d != ds]
        self._rerender()

    # -- rendering ----------------------------------------------------------

    def create(self, container, available_datasets: List[str]) -> None:
        """Attach the grid to *container*; *available_datasets* are offered
        as new dataset columns."""
        self._container = container
        self._available = [d for d in available_datasets if d not in self._datasets]
        self._rerender()

    def _rerender(self) -> None:
        if self._container is None:
            return
        self._container.clear()
        self._cell_widgets = {}
        with self._container:
            ui.label(
                f"{self.side.replace('_mapping', '').title()} mapping - "
                "one group at a time; each dataset has its own member row"
            ).classes("text-caption drocat-muted")
            for i, name in enumerate(self._groups):
                with ui.element("div").classes("w-full drocat-labelmapper-group"):
                    with ui.column().classes("drocat-labelmapper-group-name gap-1"):
                        ui.label("Group name").classes("text-caption drocat-muted")
                        name_input = ui.input(value=name).props(
                            "outlined"
                        ).classes("w-full drocat-input")
                        name_input.on_value_change(
                            lambda e, idx=i: self._set_group_name(idx, e.value)
                        )
                    with ui.column().classes(
                        "w-full min-w-0 gap-2 drocat-labelmapper-group-members"
                    ):
                        with ui.row().classes(
                            "w-full items-center gap-4 flex-wrap "
                            "drocat-labelmapper-row-actions"
                        ):
                            with ui.row().classes(
                                "items-center justify-center gap-4 flex-wrap "
                                "drocat-labelmapper-query-actions"
                            ):
                                pass
                            with ui.row().classes(
                                "items-center gap-1 drocat-labelmapper-group-controls"
                            ):
                                ui.button(
                                    icon="delete",
                                    on_click=lambda idx=i: self._remove_group(idx),
                                ).props("flat dense round").tooltip("Remove group")
                    with ui.column().classes(
                        "w-full gap-2 drocat-labelmapper-datasets"
                    ):
                        for ds in self._datasets:
                            with ui.row().classes(
                                "w-full items-start gap-3 "
                                "drocat-labelmapper-dataset-row"
                            ):
                                ui.label(ds).classes(
                                    "drocat-labelmapper-dataset-label"
                                )
                                with ui.column().classes(
                                    "gap-0 min-w-0 "
                                    "drocat-labelmapper-dataset-input"
                                ):
                                    cell = neuron_list_input(
                                        label="Neuron members",
                                        unit_label="member",
                                        show_filter=False,
                                        show_upload=False,
                                        initial=self._parse_cell_text(
                                            self._cells[ds].get(i, "")
                                        ),
                                        suggestions=self._cell_suggest(ds),
                                        history_hint_datasets=lambda ds=ds: [ds],
                                        available_neurons=lambda dataset=ds: dataset,
                                    )
                                self._cell_widgets.setdefault(ds, {})[i] = cell
                                cell.chip_input.on_value_change(
                                    lambda _e, d=ds, idx=i, widget=cell:
                                    self._cells[d].__setitem__(
                                        idx,
                                        self._encode_cell_values(
                                            widget.get_value()[1]
                                        ),
                                    )
                                )
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.button("Add Group", icon="add", on_click=self._add_group).props("flat dense")
                if self._available:
                    ds_select = ui.select(options=self._available, label="Add dataset column").classes(
                        "drocat-select"
                    ).props("outlined").style("width: 260px")
                    ui.button("Add Column", icon="playlist_add", on_click=lambda: self._add_dataset(ds_select.value)).props(
                        "flat dense"
                    )

    def _set_group_name(self, idx: int, value: str) -> None:
        if 0 <= idx < len(self._groups):
            self._groups[idx] = value or f"Group_{idx + 1}"
