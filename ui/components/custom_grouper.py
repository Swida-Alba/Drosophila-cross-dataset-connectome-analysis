"""Lite custom grouper: query-local, layer-manager-style group editor.

A light sibling of the Skeleton tab's layer manager: rows are GROUPS
(optional label + one neuron cell per dataset column) instead of path
layers. It plugs into every tab that offers Custom Grouping and exports the
LabelMapper ``overall_mapping_json`` schema — the single canonical format
that the backend reads directly and the UI reloads (format uniformity):

    {
        "format": "drocat_custom_groups/v1",
        "groups_meta": {"updated_at": "...", "origin": "inline"},
        "source_mapping": {
            "custom_label": ["aMe", "clock"],
            "male-cns:v0.9": [["aMe12", "aMe10"], ["DN1p"]],
            "hemibrain:v1.2.1": [["aMe12"], []]
        },
        "target_mapping": { "...same content..." }
    }

Rules baked in here:

- One ``custom_label`` entry governs EVERY dataset column; export always
  materializes every dataset from the provider as a key (empty groups
  ``[]`` included), so ``ComparisonAnalyzer.validate_datasets`` passes on
  cross-dataset runs.
- Source and target roles carry identical content (asymmetric per-role
  grouping stays the Settings preset editor's job); no intermediate role.
- Blank labels auto-name to ``Group_N`` at export (mirrors LabelMapper's
  own ``{role}_grp{i}`` auto-naming); cross-dataset tabs make names
  compulsory via ``require_names``.
- ``LiteGroupHandle`` is JS-free so harness tests can drive it directly.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nicegui import ui

from .. import group_history, history_store
from ..type_suggestions import get_dataset_pools, match_suggestions
from .common import neuron_list_input

FORMAT_TAG = "drocat_custom_groups/v1"
_LABEL_KEYS = ("custom_label", "std_label")
_ROLE_PRECEDENCE = ("source_mapping", "target_mapping", "intermediate_mapping")


# ---------------------------------------------------------------------------
# Serializer (shared by export, reload and history)
# ---------------------------------------------------------------------------

def auto_label(index: int) -> str:
    """The export name of an unnamed group row (1-based)."""
    return f"Group_{index}"


def to_canonical_dict(rows: List[dict], datasets: List[str],
                      origin: str = "inline") -> dict:
    """Build the canonical LabelMapper overall-JSON payload.

    Every dataset in *datasets* becomes a key of every role side with one
    group list per label (``[]`` when the row has no members there), keeping
    the grid rectangular as ``_load_from_json`` requires.
    """
    labels = []
    for i, row in enumerate(rows):
        name = str(row.get("name") or "").strip()
        labels.append(name or auto_label(i + 1))

    def build_side() -> dict:
        side: dict = {"custom_label": labels}
        for ds in datasets:
            side[str(ds)] = [
                [str(x).strip()
                 for x in ((row.get("cells") or {}).get(ds) or [])
                 if str(x).strip()]
                for row in rows
            ]
        return side

    return {
        "format": FORMAT_TAG,
        "groups_meta": {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "origin": origin,
        },
        "source_mapping": build_side(),
        "target_mapping": build_side(),
    }


def from_canonical_dict(data: dict) -> List[dict]:
    """Parse ANY canonical/legacy mapping payload back into board rows.

    Side precedence source -> target -> intermediate; accepts both
    ``custom_label`` and legacy ``std_label``; ignores ``format`` /
    ``groups_meta`` and any unknown keys. Raises ``ValueError`` when no
    usable side exists.
    """
    if not isinstance(data, dict):
        raise ValueError("Mapping file must contain a JSON object")
    for side_key in _ROLE_PRECEDENCE:
        side = data.get(side_key)
        if not isinstance(side, dict):
            continue
        label_key = next((k for k in _LABEL_KEYS if k in side), None)
        if not label_key:
            continue
        labels = [str(x) for x in side.get(label_key) or []]
        ds_keys = [k for k in side if k != label_key]
        rows = []
        for i, lab in enumerate(labels):
            cells: Dict[str, List[str]] = {}
            for ds in ds_keys:
                groups = side.get(ds)
                vals = groups[i] if isinstance(groups, list) and i < len(groups) else []
                if isinstance(vals, (str, int, float)):
                    vals = [vals]
                cells[ds] = [str(v) for v in vals if str(v).strip()]
            rows.append({"name": lab, "cells": cells})
        return rows
    raise ValueError(
        "No mapping side with 'custom_label'/'std_label' found in file")


def datasets_with_members(rows: List[dict]) -> List[str]:
    """Return datasets that have at least one member in *rows*.

    Mapping files are often rectangular already, so simply collecting every
    dataset key would select columns that are present only as empty padding.
    The Settings editor uses this helper when loading a map: non-empty
    columns are added to the embedded selector, and the renderer then shows
    every selected column for every group (including empty cells).
    """
    datasets: List[str] = []
    seen = set()
    for row in rows or []:
        for dataset, values in (row.get("cells") or {}).items():
            if not any(str(value).strip() for value in (values or [])):
                continue
            dataset = str(dataset).strip()
            if dataset and dataset not in seen:
                seen.add(dataset)
                datasets.append(dataset)
    return datasets


# ---------------------------------------------------------------------------
# State handle (JS-free, test-driven)
# ---------------------------------------------------------------------------

class LiteGroupHandle:
    """Ordered group rows: ``{"name": str, "cells": {dataset: [neurons]}}``."""

    def __init__(self):
        self.rows: List[dict] = []

    def add_row(self, name: str = "",
                cells: Optional[Dict[str, List[str]]] = None) -> int:
        self.rows.append({
            "name": (name or "").strip(),
            "cells": {str(ds): [str(v) for v in vals]
                      for ds, vals in (cells or {}).items()},
        })
        return len(self.rows) - 1

    def upsert_row(self, name: str,
                   cells: Dict[str, List[str]]) -> int:
        """Replace the row carrying *name* (cover semantics), else append."""
        name = (name or "").strip()
        for i, row in enumerate(self.rows):
            if row["name"] == name:
                self.rows[i] = {
                    "name": name,
                    "cells": {str(ds): [str(v) for v in vals]
                              for ds, vals in (cells or {}).items()},
                }
                return i
        return self.add_row(name, cells)

    @staticmethod
    def _row_is_blank(row: dict) -> bool:
        """Whether a row has neither a label nor any member values."""
        return not str(row.get("name") or "").strip() and not any(
            str(value).strip()
            for values in (row.get("cells") or {}).values()
            for value in (values or [])
        )

    def load_saved_row(self, name: str,
                       cells: Dict[str, List[str]]) -> int:
        """Load one history row, reusing the initial blank placeholder.

        Opening the panel seeds one blank row so it is immediately editable.
        A saved group should occupy that placeholder instead of being pushed
        into a second row. Once the first row contains either a label or a
        member, normal upsert behavior preserves it and appends new groups.
        """
        name = (name or "").strip()
        row = {
            "name": name,
            "cells": {str(ds): [str(v) for v in vals]
                      for ds, vals in (cells or {}).items()},
        }
        if self.rows and self._row_is_blank(self.rows[0]):
            self.rows[0] = row
            # Avoid a duplicate if the same label was already present later.
            self.rows[1:] = [
                existing for existing in self.rows[1:]
                if existing.get("name") != name
            ]
            return 0
        return self.upsert_row(name, cells)

    def remove_row(self, index: int) -> None:
        if 0 <= index < len(self.rows):
            del self.rows[index]

    def move_row(self, index: int, delta: int) -> None:
        target = index + delta
        if 0 <= index < len(self.rows) and 0 <= target < len(self.rows):
            self.rows.insert(target, self.rows.pop(index))

    def replace_rows(self, rows: List[dict]) -> None:
        self.rows = [
            {"name": str(r.get("name") or "").strip(),
             "cells": {str(ds): [str(v) for v in vals]
                       for ds, vals in (r.get("cells") or {}).items()}}
            for r in rows
        ]

    def is_empty(self) -> bool:
        return not any(
            any(vals for vals in (row.get("cells") or {}).values())
            for row in self.rows
        )


# ---------------------------------------------------------------------------
# UI component
# ---------------------------------------------------------------------------

class LiteCustomGrouper:
    """The inline group board; one instance per tab."""

    def __init__(self, tab_key: str, require_names: bool = False,
                 query_inputs: Optional[Dict[str, object]] = None,
                 row_action_renderers: Optional[List[Callable[[], None]]] = None):
        self.tab_key = tab_key
        self.require_names = require_names
        # Optional named query inputs (e.g. {"source": ..., "target": ...})
        # that group rows can push their members into.
        self.query_inputs = dict(query_inputs or {})
        # Optional controls rendered alongside the per-row query actions.
        # Keep the caller's list by reference so Settings can attach actions
        # after the shared editor has been constructed.
        self.row_action_renderers = (
            row_action_renderers
            if row_action_renderers is not None
            else []
        )
        self.handle = LiteGroupHandle()
        self._datasets_provider: Callable[[], List[str]] = lambda: []
        self._datasets_setter: Optional[Callable[[List[str]], None]] = None
        self._suspend_dataset_watch = False
        self._container: Optional[ui.column] = None
        self._name_inputs: List[ui.input] = []
        self._cell_widgets: List[Dict[str, ui.element]] = []

    # ------------------------------------------------------------ lifecycle
    def create(self, container: ui.column,
               datasets_provider: Callable[[], List[str]],
               watch_elements: Optional[List] = None,
               datasets_setter: Optional[Callable[[List[str]], None]] = None,
               ) -> "LiteCustomGrouper":
        self._container = container
        self._datasets_provider = datasets_provider
        self._datasets_setter = datasets_setter
        for el in (watch_elements or []):
            el.on_value_change(lambda _e: self._on_dataset_change())
        self.resync()
        return self

    def datasets(self) -> List[str]:
        return [str(d) for d in (self._datasets_provider() or []) if d]

    def _on_dataset_change(self) -> None:
        if not self._suspend_dataset_watch:
            self.resync()

    def ensure_datasets(self, datasets: List[str]) -> List[str]:
        """Add dataset values to the selector without dropping its choices.

        Loading a map can introduce datasets that were not selected before
        the panel opened. The setter is temporarily guarded so its normal
        value-change watcher cannot collect the old widgets into the newly
        loaded rows before the final render.
        """
        current = self.datasets()
        merged = list(current)
        for dataset in datasets or []:
            dataset = str(dataset).strip()
            if dataset and dataset not in merged:
                merged.append(dataset)
        if merged == current or self._datasets_setter is None:
            return current
        self._suspend_dataset_watch = True
        try:
            self._datasets_setter(merged)
        finally:
            self._suspend_dataset_watch = False
        return self.datasets()

    def ensure_row_datasets(self) -> List[str]:
        """Add every non-empty dataset represented by the current board."""
        return self.ensure_datasets(datasets_with_members(self.handle.rows))

    def load_rows(self, rows: List[dict]) -> None:
        """Replace the board and add its non-empty dataset columns."""
        self._collect_rows()
        self.handle.replace_rows(rows)
        self.ensure_row_datasets()
        self._render()

    def resync(self) -> None:
        """Re-render the board (dataset columns and cell widgets follow)."""
        if self._container is None:
            return
        self._collect_rows()  # preserve typed state before rebuilding
        self._render()

    # ------------------------------------------------------------- rendering
    def _cell_suggest(self, ds: str):
        def _suggest(text: str):
            return match_suggestions(text, get_dataset_pools(ds), "auto")
        return _suggest

    def _render(self) -> None:
        container = self._container
        container.clear()
        self._name_inputs = []
        self._cell_widgets = []
        datasets = self.datasets()
        with container:
            # Single-row toolbar: title + compact functional icons.
            with ui.row().classes("items-center gap-1 w-full flex-nowrap"):
                ui.label("Inline Custom Groups").classes("drocat-card-title")
                ui.space()
                ui.button(icon="add", on_click=self._add_group).props(
                    "flat dense round").tooltip("Add group")
                self._load_button()
            ui.label(
                "Each group starts with a label, followed by one aligned "
                "member row per dataset. Empty boxes export as empty groups."
            ).classes("text-caption drocat-muted")
            for i, row in enumerate(self.handle.rows):
                self._render_row(i, row, datasets)
            if not datasets:
                ui.label(
                    "Select at least one dataset above to add dataset-specific "
                    "members to a group."
                ).classes("drocat-empty")
            if not self.handle.rows:
                ui.label("No groups yet — click 'Add Group' or pull one from "
                         "history.").classes("drocat-empty")

    _QUERY_ICONS = {"source": "login", "target": "logout", "query": "hub"}

    def _render_row(self, i: int, row: dict, datasets: List[str]) -> None:
        widgets: Dict[str, ui.element] = {}
        with ui.element("div").classes("w-full drocat-labelmapper-group"):
            with ui.column().classes("drocat-labelmapper-group-name gap-1"):
                ui.label("Group name").classes("text-caption drocat-muted")
                name_input = ui.input(
                    value=row["name"],
                    placeholder=auto_label(i + 1)
                    + (" (compulsory)" if self.require_names else " (optional)"),
                ).props("outlined").classes("w-full drocat-input")
                self._name_inputs.append(name_input)
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
                        for key, target in self.query_inputs.items():
                            icon = self._QUERY_ICONS.get(key, "add")
                            ui.button(
                                f"Add to {key.title()}", icon=icon,
                                on_click=lambda _e, k=key, idx=i: self.push_to_query(k, idx),
                            ).props("outline no-caps").classes(
                                "drocat-labelmapper-query-action"
                            ).tooltip(
                                f"Add this group's members to the {key.title()} input")
                        # These are global editor actions (for example Settings'
                        # Save Mapping), so render them once in the first group's
                        # action row rather than repeating them for every group.
                        if i == 0:
                            for render_action in self.row_action_renderers:
                                render_action()
                    with ui.row().classes(
                        "items-center gap-1 drocat-labelmapper-group-controls"
                    ):
                        ui.button(
                            icon="delete",
                            on_click=lambda _e, idx=i: self._remove_group(idx),
                        ).props("flat dense round").tooltip("Remove group")
            with ui.column().classes(
                "w-full gap-2 drocat-labelmapper-datasets"
            ):
                for ds in datasets:
                    with ui.row().classes(
                        "w-full items-start gap-3 drocat-labelmapper-dataset-row"
                    ):
                        ui.label(ds).classes("drocat-labelmapper-dataset-label")
                        with ui.column().classes(
                            "gap-0 min-w-0 drocat-labelmapper-dataset-input"
                        ):
                            widget = neuron_list_input(
                                label="Neuron members",
                                unit_label="member",
                                show_filter=False,
                                show_upload=False,
                                initial=row["cells"].get(ds, []),
                                suggestions=self._cell_suggest(ds),
                                history_hint_datasets=lambda ds=ds: [ds],
                                available_neurons=lambda dataset=ds: dataset,
                            )
                        widgets[ds] = widget
        self._cell_widgets.append(widgets)

    def push_to_query(self, key: str, row_index: int) -> List[str]:
        """Add a row's group LABEL to the named query input.

        The label (not the raw members) is pushed so the query chip reads as
        the group; the backend expands the label into its member neurons via
        the active mapping (``FindNeuronConnection._expand_group_labels``)
        and the exported tables/visualizations show the group label.
        Returns the pushed label list.
        """
        target = self.query_inputs.get(key)
        if target is None or not hasattr(target, "add_values"):
            return []
        rows = self._collect_rows()
        if not (0 <= row_index < len(rows)):
            return []
        if not any(vals for vals in (rows[row_index].get("cells") or {}).values()):
            ui.notify("This group has no members to add", type="warning")
            return []
        label = rows[row_index]["name"] or auto_label(row_index + 1)
        target.add_values([label])
        # Record the group into the history the moment it is pushed to a
        # query, so it immediately appears in the panel's history pulldown.
        group_history.record(
            [(rows[row_index]["name"], rows[row_index].get("cells") or {})],
            origin="inline")
        history_store.record(
            [label],
            custom_values=[label],
            datasets=[
                dataset for dataset, values in
                (rows[row_index].get("cells") or {}).items()
                if any(str(value).strip() for value in (values or []))
            ],
        )
        ui.notify(f"Group '{label}' added to {key.title()}", type="positive")
        return [label]

    # --------------------------------------------------------------- actions
    def _add_group(self) -> None:
        self._collect_rows()
        self.handle.add_row()
        self._render()

    def _remove_group(self, index: int) -> None:
        self._collect_rows()
        self.handle.remove_row(index)
        self._render()

    def _move_group(self, index: int, delta: int) -> None:
        self._collect_rows()
        self.handle.move_row(index, delta)
        self._render()

    def _load_button(self) -> None:
        with ui.button(icon="upload_file").props("flat dense round").tooltip(
                "Load… (mapping JSON)"):
            with ui.menu():
                ui.label("Load groups from a mapping JSON (canonical or "
                         "legacy preset format)").classes(
                    "text-caption drocat-muted px-3 pt-2")
                ui.upload(label="Choose .json", auto_upload=True,
                          on_upload=self._handle_load).props(
                    'accept=".json" flat dense').classes("w-72 px-3 pb-2")

    async def _handle_load(self, e) -> None:
        try:
            raw = e.content.read()
            data = json.loads(raw.decode("utf-8-sig", errors="replace"))
            rows = from_canonical_dict(data)
            if not rows:
                raise ValueError("file defines no groups")
            self.load_rows(rows)
            ui.notify(f"Loaded {len(rows)} group(s)", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load mapping file: {exc}", type="negative")

    # ---------------------------------------------------------------- state
    def _collect_rows(self) -> List[dict]:
        """Sync the handle from live widgets; returns the collected rows."""
        datasets = self.datasets()
        rows = []
        for i, row in enumerate(self.handle.rows):
            name = (self._name_inputs[i].value
                    if i < len(self._name_inputs) else row.get("name")) or ""
            widgets = self._cell_widgets[i] if i < len(self._cell_widgets) else {}
            cells: Dict[str, List[str]] = {}
            for ds in datasets:
                widget = widgets.get(ds)
                if widget is not None:
                    cells[ds] = [str(v) for v in widget.get_value()[1]]
                else:
                    cells[ds] = list((row.get("cells") or {}).get(ds) or [])
            # Keep members of datasets that are not currently rendered
            # (e.g. pulled from history) so they survive column changes.
            for ds, vals in (row.get("cells") or {}).items():
                if ds not in cells:
                    cells[ds] = list(vals)
            rows.append({"name": str(name).strip(), "cells": cells})
        self.handle.replace_rows(rows)
        return rows

    # --------------------------------------------------------------- export
    def _active_rows(self) -> List[dict]:
        """Rows that carry a name or members; drops the seeded empty
        placeholder row so it never exports as a stray ``Group_N``."""
        return [r for r in self._collect_rows()
                if r["name"] or any((r.get("cells") or {}).values())]

    def validate(self) -> List[str]:
        """Collect state and report blocking errors (empty when valid)."""
        rows = self._active_rows()
        errors = []
        if not rows:
            errors.append("Define at least one non-empty custom group")
            return errors
        labels = []
        for i, row in enumerate(rows):
            label = row["name"] or auto_label(i + 1)
            labels.append(label)
            if self.require_names and not row["name"]:
                errors.append(
                    f"Group {i + 1}: the label name is compulsory for "
                    "cross-dataset comparisons")
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        if dupes:
            errors.append("Duplicate group labels: " + ", ".join(dupes))
        return errors

    def to_canonical(self, origin: str = "inline") -> dict:
        rows = self._active_rows()
        return to_canonical_dict(rows, self.datasets(), origin=origin)

    def export_to(self, path) -> str:
        """Validate, write the canonical JSON and sweep stale same-tab files.

        Only files older than 24 h are removed: an in-flight run may still
        need the export it was handed (the runner subprocess reads the file
        seconds later), so unconditional pruning could delete a live run's
        mapping out from under it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now().timestamp() - 24 * 3600
        for stale in path.parent.glob(f"{self.tab_key}_*.json"):
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                pass
        payload = self.to_canonical()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return str(path)

    def history_payload(self) -> List[tuple]:
        """(original_label, cells) pairs; blank labels stay blank so the
        history store can skip auto-named groups."""
        return [(row["name"], row["cells"]) for row in self._active_rows()]

    def is_empty(self) -> bool:
        return not self._active_rows()
