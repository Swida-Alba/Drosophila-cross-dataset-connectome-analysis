"""NiceGUI viewer for a locally cached neuron index."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from nicegui import run, ui

from ..config import PROJECT_ROOT
from utils.naming_utils import dataset_abbrev
from .banner import push_banner
from ..neuron_index import (
    NO_DERIVATION_TEXT,
    build_matches_csv,
    collect_zero_hit_matches,
    collect_zero_hit_matches_in_process,
    count_types_in_index,
    is_type_mapper_loaded,
    load_cached_neuron_index,
    mapped_csv_extras,
    neuron_index_path,
    query_match_group_subtypes,
    query_neuron_index,
    resolve_prioritized_bridge_pool,
    run_cross_dataset_scan_in_process,
    run_serialized_cross_dataset_scan,
    CROSS_SCAN_SUPERSEDED,
)


# A one-character query can legitimately produce thousands of deduplicated
# names. Sending every match row to Quasar at once overwhelms the websocket
# even though the underlying server-side query is bounded and responsive.
# Keep the full membership maps for exact selection, but render a fixed page
# of match details; the panel pager still exposes every matched name.
MATCH_GROUP_PAGE_SIZE = 50
# One gesture can produce a value-click, a QTable selection event, and a
# delayed table update. Cover the maximum scroll-settle plus notification
# lifetime so that these events can never start a second focus animation.
FOCUS_DEDUP_SECONDS = 3.2
# The viewer's automatic table search ignores single-character input: one
# letter would re-filter the whole index on every keystroke with mostly noise.
# The explicit Search button can still submit a deliberate one-character query.
MIN_SEARCH_CHARS = 2
# Cross-dataset scans settle behind the search text.  Every render supersedes
# the previous prefix's pending scan, so typing 'CB2438' runs one scan for the
# final query instead of one per keystroke; each scan holds the process-wide
# single flight, and a queued scan behind a cold worker would otherwise run
# every broad intermediate query to completion.
ALIAS_SCAN_SETTLE_SECONDS = 0.6
# A query matching this many local rows also maps into tens of thousands of
# foreign names (the 'CB' prefix on male-cns matched ~37,000 rows and fed
# ~23,000 items through the mapping walk).  Even with a warm in-process
# mapper, run those scans in the spawned worker: their CPU-heavy enrichment
# must never run inside the websocket process, or the client misses its
# heartbeats and the connection drops (the CB2438 crash).
BROAD_QUERY_LOCAL_HITS = 10_000


def _effective_search_text(raw: str, *, force: bool = False) -> str:
    """Return the search text this box sends to the backend.

    Queries shorter than :data:`MIN_SEARCH_CHARS` (measured after stripping)
    stay unfiltered during automatic refreshes, so the first typed character
    alone never matches. An explicit Search button can set ``force`` to allow
    a deliberate one-character query.
    """
    text = str(raw or "").strip()
    if not text or (not force and len(text) < MIN_SEARCH_CHARS):
        return ""
    return text


def _normalized_focus_keys(keys) -> tuple[str, ...]:
    """Return stable, non-empty focus keys without changing their order."""
    result: list[str] = []
    seen: set[str] = set()
    for key in keys or ():
        value = str(key or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _dataset_values(value) -> List[str]:
    """Normalize a single- or multi-dataset getter result."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _column_label(column: str) -> str:
    """Verbatim metadata column name (user 2026-09-27 unification).

    The 'See available neurons' panel displays the neuron index's original
    column names exactly as the dataset tables and the type mapper use
    them — no title-casing — so names read off the panel (e.g.
    ``fafb_cell_type``, ``flywireType``, ``Alternative Cell Type(s)``)
    agree with the exports, the bridge map and the docs.
    """
    return column


def _relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _query_preview_values(getter: Callable[[], object] | None) -> List[str]:
    """Read the current query values for the viewer's compact preview."""
    if getter is None:
        return []
    try:
        value = getter()
    except Exception:
        return []
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    seen = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _render_missing_cache(content, dataset: str, path: Path) -> None:
    with content:
        ui.icon("database", color="orange").classes("text-4xl")
        ui.label("This dataset is not cached locally yet.").classes("text-subtitle1 font-bold")
        ui.label(
            f"The cached neuron index was not found at {_relative_source(path)}. "
            "The viewer does not open or stream the original dataset file."
        ).classes("text-body2")
        ui.label(
            "To make the index available, either run the selected analysis once "
            "(where the workflow supports first-run cache creation), or pull the "
            "dataset metadata and connections from the Settings tab."
        ).classes("text-body2")
        ui.label("Recommended UI flow").classes("font-bold text-primary mt-2")
        ui.label(
            "Open Settings → Dataset Cache, choose "
            f"{dataset}, then click Pull Dataset Metadata, followed by "
            "Pull Complete Connections for the connection cache. "
            "The connection pull is resumable."
        ).classes("text-body2")
        ui.link(
            "Open dataset-cache instructions",
            "docs/ui_guides/settings.html",
            new_tab=True,
        ).classes("drocat-doc-link")
        ui.label("Command-line alternative (from the project root)").classes(
            "font-bold text-primary mt-2"
        )
        ui.code(
            f"python src/build_connection_cache.py {dataset}",
            language="bash",
        ).classes("w-full")
        ui.label(
            "NeuPrint datasets need a configured token. FAFB must be prepared "
            "locally and BANC uses its public release; follow the matching guide in Settings."
        ).classes("text-caption drocat-muted")


def _render_index(
    content,
    dataset: str,
    *,
    header_meta=None,
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    query_edit: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
    defer_apply: bool = False,
    apply_holder: dict | None = None,
    cross_mapping_mode: dict | None = None,
) -> None:
    """Render the current dataset's index or its cache-missing state.

    ``cross_mapping_mode`` is the dialog header's mode holder
    (``{"enabled": bool, "refresh": callable}``): when enabled, the search
    also maps the query into the other cached datasets — a collapsed
    cross-dataset expansion renders above the results alongside the native
    rows. The rendered viewer registers its footer refresh under
    ``"refresh"`` so the header toggle can re-evaluate the panel without
    re-running the query.
    """
    content.clear()
    if header_meta is not None:
        header_meta.clear()
    path = neuron_index_path(dataset)
    try:
        index = load_cached_neuron_index(dataset)
    except FileNotFoundError:
        _render_missing_cache(content, dataset, path)
        return
    except Exception as exc:
        with content:
            ui.icon("error", color="red").classes("text-4xl")
            ui.label("The cached neuron index could not be opened.").classes(
                "text-subtitle1 font-bold"
            )
            ui.label(str(exc)).classes("text-body2 drocat-err")
            ui.label(
                "Use Settings → Dataset Cache → Force rebuild, then open this viewer again."
            ).classes("text-caption drocat-muted")
        return

    columns = list(index.columns)
    if not columns:
        with content:
            ui.label("The cached neuron index is empty.").classes("text-body2 drocat-warn")
        return

    if header_meta is not None:
        with header_meta:
            ui.badge(
                f"{index.frame.height:,} indexed rows", color="primary"
            ).props("outline")
            ui.label(f"Source: {_relative_source(index.path)}").classes(
                "text-caption drocat-muted drocat-neuron-source"
            )
            if index.enriched:
                ui.label(
                    "metadata-enriched"
                ).classes("text-caption drocat-muted drocat-neuron-enriched")

    with content:
        if query_values_getter is not None:
            with ui.element("div").classes("w-full drocat-neuron-intro-row"):
                with ui.element("section").classes("drocat-neuron-query-preview"):
                    query_preview_state = {"expanded": False}

                    def toggle_query_preview() -> None:
                        query_preview_state["expanded"] = not query_preview_state[
                            "expanded"
                        ]
                        if query_preview_state["expanded"]:
                            query_preview_scroll.classes(
                                add="drocat-neuron-query-preview-expanded",
                                remove="drocat-neuron-query-preview-collapsed",
                            )
                            query_preview_toggle.text = "Collapse"
                        else:
                            query_preview_scroll.classes(
                                add="drocat-neuron-query-preview-collapsed",
                                remove="drocat-neuron-query-preview-expanded",
                            )
                            query_preview_toggle.text = "Expand"
                        query_preview_toggle.update()

                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("playlist_add_check", color="primary").classes("text-lg")
                            ui.label(
                                f"{'Selected' if defer_apply else 'Current query'} · {query_label}"
                            ).classes(
                                "text-subtitle2 font-bold"
                            )
                        with ui.row().classes("items-center gap-1"):
                            query_preview_toggle = ui.button(
                                "Expand", on_click=toggle_query_preview
                            ).props("flat dense").classes(
                                "drocat-query-preview-expand-btn"
                            )
                            ui.badge(
                                "selected" if defer_apply else "mirrors input",
                                color="primary",
                            ).props("outline")
                    with ui.element("div").classes(
                        "w-full drocat-neuron-query-preview-list "
                        "drocat-neuron-query-preview-collapsed"
                    ) as query_preview_scroll:
                        query_preview = ui.row().classes(
                            "w-full items-center gap-1 flex-wrap"
                        )
                    query_preview_empty = ui.label(
                        "No values in the query yet. Select a match or body row to add one."
                    ).classes("text-caption drocat-muted mt-1")

                    def refresh_query_preview() -> None:
                        # Deferred mode shows the in-panel pending selection; the
                        # live mode mirrors the owning input's committed values.
                        values = (
                            selected_query_values()
                            if defer_apply
                            else _query_preview_values(query_values_getter)
                        )
                        query_preview.clear()
                        query_preview_empty.set_visibility(not values)
                        query_preview_toggle.set_visibility(bool(values))
                        with query_preview:
                            for value in values:
                                with ui.element("div").classes(
                                    "drocat-neuron-query-chip-wrap"
                                ) as query_chip:
                                    # Keep the value label's historic class so
                                    # the preview remains easy to inspect and
                                    # compatible with existing UI tests.
                                    ui.label(value).classes(
                                        "drocat-neuron-query-chip"
                                    )
                                    query_chip.on(
                                        "dblclick",
                                        lambda _event=None, v=value: edit_query_value(v),
                                    ).tooltip("Double-click to edit this query value")
                                    if query_remove is not None:
                                        ui.button(
                                            icon="close",
                                            on_click=lambda v=value: remove_query_value(v),
                                        ).props("flat round dense").classes(
                                            "drocat-neuron-query-chip-remove"
                                        ).tooltip("Remove from this query")

                    def edit_query_value(value: str) -> None:
                        """Remove the value from viewer selection, then edit it."""
                        remove_query_value(value)
                        if query_edit is not None:
                            query_edit(value)
                        refresh_query_preview()
                ui.label(
                    "Search returns all matches across bodyId, type, instance, and useful "
                    "type/taxonomy fields once the query has at least two characters; "
                    "typing a single character does not auto-filter, but Search can "
                    "force a safe prefix-only search for it. Strict case-sensitive "
                    "prefixes come first, followed by case-insensitive substring "
                    "matches. Choose a target column and match mode to apply that rule "
                    "directly to this search box (Contains matches anywhere, without "
                    "starts-with priority); leave it unset for the global search. "
                    "Numeric input is verified against bodyId. Match details also keeps "
                    "a secondary matched name when the same row matches in another "
                    "field. Select a matched name to select every body sharing it, or "
                    "select individual body rows to add their body IDs."
                ).classes("text-caption drocat-muted drocat-neuron-search-help")
        else:
            ui.label(
                "Search returns all matches across bodyId, type, instance, and useful "
                "type/taxonomy fields once the query has at least two characters; "
                "typing a single character does not auto-filter, but Search can "
                "force a safe prefix-only search for it. Strict case-sensitive "
                "prefixes come first, followed by case-insensitive substring matches. "
                "Choose a target column and match mode to apply that rule directly to "
                "this search box (Contains matches anywhere, without starts-with "
                "priority); leave it unset for the global search. Numeric input is "
                "verified against bodyId."
            ).classes("text-caption drocat-muted drocat-neuron-search-help")

        with ui.row().classes(
            "w-full items-end gap-2 flex-wrap drocat-neuron-search-toolbar"
        ):
            search_input = ui.input(
                "Search identities & taxonomy",
                placeholder="e.g. aMe12 or 5813",
            ).props("outlined clearable input-debounce=180").classes(
                "flex-grow drocat-input drocat-neuron-search-field"
            )
            search_button = ui.button("Search", icon="search").props(
                "unelevated color=primary"
            ).classes("drocat-neuron-search-submit").tooltip(
                "Run this query, including a deliberate one-character search"
            )
            filter_options = {"__none__": "No column filter"}
            filter_options.update({column: _column_label(column) for column in columns})
            target_column = ui.select(
                options=filter_options,
                value="__none__",
                label="Target column",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            filter_operator = ui.select(
                options={
                    "contains": "Contains",
                    "prefix": "Starts with",
                    "suffix": "Ends with",
                    "exact": "Exact",
                    "regex": "Regex",
                },
                value="contains",
                label="Match mode",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            sort_options = {"__match_value__": "Matched value (default)"}
            sort_options.update({column: _column_label(column) for column in columns})
            sort_column = ui.select(
                options=sort_options,
                value="__match_value__",
                label="Sort by",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            direction = ui.select(
                options={"asc": "Ascending", "desc": "Descending"},
                value="asc",
                label="Order",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            page_size = ui.select(
                options={
                    25: "25 / page",
                    50: "50 / page",
                    100: "100 / page",
                    200: "200 / page",
                    500: "500 / page",
                },
                value=50,
                label="Rows",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 120px")

        single_char_warning = ui.label("").classes(
            "text-caption drocat-warn drocat-neuron-search-warning"
        )
        single_char_warning.set_visibility(False)

        filter_operator.set_enabled(False)

        # Cross-dataset search panel: always rendered on a zero-hit query,
        # and — collapsed, with a summary line — above the results while the
        # dialog header's "Cross-dataset type mapping" mode is enabled.
        # Content is strictly informational — other-dataset rows are never
        # merged into this table or selection.
        with ui.element("section").classes(
            "w-full drocat-neuron-alias-panel"
        ) as alias_section:
            alias_container = ui.element("div").classes("w-full")
        alias_section.set_visibility(False)
        # Current-key state for the alias scan: every new query or hidden
        # panel voids the result of a scan still running in the background,
        # so a stale scan can never overwrite newer UI state. The last
        # completed scan is cached per query, so paging and display changes
        # never re-run the mapper.
        alias_scan = {
            "cache": {"key": None, "matches": None},
            "expanded": False,
            # The input change and the explicit Search button can describe
            # the same query.  Coalesce that request while the cold mapper
            # scan is running instead of queueing a second copy.
            "current_key": None,
            "inflight_key": None,
        }
        # Search/filter events can arrive faster than the indexed query can
        # finish. Keep a monotonically increasing generation so an older
        # worker result can never repaint a newer query.
        refresh_generation = {"value": 0}
        # A forced one-character query remains active for paging and display
        # changes until the user edits the search text again.
        search_state = {"forced_text": None}

        initial = query_neuron_index(index, page_size=50)
        match_columns = [
            {
                "name": "match_column",
                "label": "Matched by",
                "field": "match_column",
                "align": "left",
                "classes": "drocat-neuron-match-by",
                "headerClasses": "drocat-neuron-match-by",
                "style": "width: 135px; min-width: 135px",
                "headerStyle": "width: 135px; min-width: 135px",
                "sortable": False,
            },
            {
                "name": "match_value",
                "label": "Matched value",
                "field": "match_value",
                "align": "left",
                "classes": "drocat-neuron-match-value",
                "headerClasses": "drocat-neuron-match-value",
                "style": "width: 190px; min-width: 190px",
                "headerStyle": "width: 190px; min-width: 190px",
                "sortable": False,
            },
            {
                "name": "body_count",
                "label": "Rows",
                "field": "body_count",
                "align": "right",
                "classes": "drocat-neuron-match-count",
                "headerClasses": "drocat-neuron-match-count",
                "style": "width: 64px; min-width: 64px",
                "headerStyle": "width: 64px; min-width: 64px",
                "sortable": False,
            },
        ]
        table_columns = [
            *[
                {
                    "name": column,
                    "label": _column_label(column),
                    "field": column,
                    # Full-index sorting is controlled above; enabling Quasar's
                    # client-side header sort here would sort only the current page.
                    "sortable": False,
                }
                for column in columns
            ],
        ]
        # The match panel is intentionally compact: one row per deduplicated
        # matched value. It is aligned with the metadata panel at the top, but
        # does not reserve blank rows for every repeated bodyId.
        selected_match_values: set[str] = set()
        selected_match_order: List[str] = []
        selected_match_members: dict[str, set[str]] = {}
        selected_match_body_ids: dict[str, tuple[str, ...]] = {}
        selected_body_ids: dict[str, str] = {}
        # Expanded subtype panels, keyed by the match-group value. Entries
        # survive query re-runs (paging, focus jumps) so a click on a member
        # body does not collapse the panel the user is working in. ``members``
        # keeps the payload from being reused when a later search changes the
        # membership of a same-named group.
        subtype_expansions: dict[str, dict] = {}
        # Lazily-computed complete key->bodyId map for the current query, used
        # only by the full metadata table's select-all across all pages.
        full_table_all_keys: dict[str, str] | None = None
        current_rows = list(initial.rows)
        match_groups_all = list(initial.match_groups)
        current_groups = match_groups_all[:MATCH_GROUP_PAGE_SIZE]
        match_state = {"page": 1}
        match_header_selection = {"all_visible": False}
        current_group_body_ids = {
            str(key): tuple(values)
            for key, values in initial.match_group_body_ids.items()
        }
        match_group_related = {
            str(key): tuple(values)
            for key, values in initial.match_group_related.items()
        }
        match_group_primary = {
            str(key): tuple(values)
            for key, values in initial.match_group_primary.items()
        }
        group_members = {
            str(key): set(values)
            for key, values in initial.match_group_members.items()
        }
        match_table = None
        table = None
        selection_status = None
        match_status = None
        match_page_position = None
        match_previous_button = None
        match_next_button = None
        query_callback = query_selection or add_to_query

        # Subtype expansion is only meaningful below a coarse taxonomy entry.
        # A type, instance, or bodyId match is already the leaf identity.
        subtype_leaf_columns = {"bodyId", "type", "instance"}

        def stamp_match_group_flags() -> None:
            """Attach expansion eligibility and persisted panel state."""
            can_expand = "type" in columns
            for group in match_groups_all:
                group["__can_expand"] = (
                    can_expand
                    and str(group.get("match_column_key", "") or "")
                    not in subtype_leaf_columns
                )
            for group in match_groups_all:
                key = str(group.get("__match_group_key", "") or "")
                entry = subtype_expansions.get(key)
                if entry is None:
                    continue
                if set(group_members.get(key, ())) != entry["members"]:
                    continue
                group["__subtypes"] = entry["display"]
                group["__expanded"] = entry["expanded"]
        stamp_match_group_flags()

        def effective_body_keys() -> set[str]:
            keys = set(selected_body_ids)
            for value in selected_match_values:
                keys.update(selected_match_members.get(value, set()))
                keys.update(group_members.get(value, set()))
            return keys

        def current_search_text(*, force_search: bool = False) -> str:
            raw_text = str(search_input.value or "").strip()
            forced = bool(raw_text) and search_state["forced_text"] == raw_text
            return _effective_search_text(
                raw_text,
                force=force_search or forced,
            )

        def current_prefix_only_search(*, force_search: bool = False) -> bool:
            """Keep an explicit one-character global search to prefix hits.

            A letter such as ``L`` appears in the hemisphere suffix of most
            instances. Expanding that deliberate one-character search to all
            case-insensitive substrings creates a result set large enough to
            overwhelm the websocket and the process. Prefix hits still cover
            names such as ``R7`` and ``R8`` across every searchable field.
            """
            raw_text = str(search_input.value or "").strip()
            forced = bool(raw_text) and search_state["forced_text"] == raw_text
            return bool(
                raw_text
                and len(raw_text) == 1
                and (force_search or forced)
            )

        def update_single_char_warning() -> None:
            """Explain the bounded behavior of a one-character query."""
            raw_text = str(search_input.value or "").strip()
            if len(raw_text) != 1:
                single_char_warning.set_visibility(False)
                return
            forced = search_state["forced_text"] == raw_text
            if forced:
                single_char_warning.text = (
                    "Warning: this one-character search returns starts-with "
                    "matches only for performance; substring matches are not "
                    "searched. Cross-dataset mapping, when enabled, follows "
                    "the same starts-with-only rule."
                )
            else:
                single_char_warning.text = (
                    "Warning: one-character searches use starts-with matches "
                    "only for performance; press Search to run this query. "
                    "Cross-dataset mapping, when enabled, follows the same "
                    "starts-with-only rule."
                )
            single_char_warning.update()
            single_char_warning.set_visibility(True)

        def current_query_kwargs(*, force_search: bool = False) -> dict:
            """Return the search/filter/sort kwargs shared by page and key queries."""
            requested_sort = sort_column.value
            if requested_sort == "__match_value__":
                # Ascending matched-value order is the implicit default. Preserve
                # an explicitly chosen descending direction for that same sort.
                requested_sort = (
                    "__match_value__" if direction.value == "desc" else None
                )
            return {
                "search": current_search_text(force_search=force_search),
                "search_column": target_column.value,
                "search_operator": filter_operator.value,
                "sort_by": requested_sort,
                "descending": direction.value == "desc",
                "prefix_only_search": current_prefix_only_search(
                    force_search=force_search
                ),
            }

        def query_kwargs_with_mapped(*, force_search: bool = False) -> dict:
            """Query kwargs; in mapped view the type set replaces the search."""
            if not mapped_view.get("active"):
                return current_query_kwargs(force_search=force_search)
            requested_sort = sort_column.value
            if requested_sort in (None, "", "__match_value__"):
                requested_sort = "type"
            return {
                "types_include": sorted(mapped_view["types"]),
                "sort_by": requested_sort,
                "descending": direction.value == "desc",
            }

        def current_full_table_keys() -> dict[str, str]:
            """Lazily materialise every matching key->bodyId across all pages."""
            nonlocal full_table_all_keys
            if full_table_all_keys is None:
                full_table_all_keys = dict(
                    query_neuron_index(
                        index,
                        **query_kwargs_with_mapped(),
                        page=1,
                        page_size=50,
                        include_all_keys=True,
                    ).all_keys
                )
            return full_table_all_keys

        def related_match_values(value: str) -> List[str]:
            """Return one direct primary/secondary selection bundle.

            The backend intentionally does not return transitive connected
            components here.  Two independent primary names may share a
            taxonomy spelling on different rows; walking a graph would make
            selecting one primary unexpectedly select the other one too.
            """
            value = str(value or "").strip()
            if not value:
                return []
            return [
                linked
                for linked in match_group_related.get(value, (value,))
                if str(linked or "").strip()
            ]

        def match_member_keys(value: str) -> List[str]:
            """Return the exact table rows belonging to one clicked match."""
            keys: List[str] = []
            for linked_value in related_match_values(value):
                members = (
                    selected_match_members.get(linked_value, set())
                    or group_members.get(linked_value, ())
                )
                for member in members:
                    member = str(member or "").strip()
                    if member and member not in keys:
                        keys.append(member)
            return keys

        def remember_match(value: str, *, expand: bool = True) -> None:
            """Persist a selected name and its verified membership.

            Match rows are replaced whenever the search changes, so the
            selection cannot depend on the current page or current search
            result. The exact body-ID snapshot is retained for execution even
            when the selected name is no longer visible in the new search.
            """
            value = str(value or "").strip()
            if not value:
                return
            linked_values = related_match_values(value) if expand else [value]
            for linked_value in linked_values:
                if linked_value not in selected_match_values:
                    selected_match_values.add(linked_value)
                selected_match_members[linked_value] = set(
                    group_members.get(
                        linked_value,
                        selected_match_members.get(linked_value, set()),
                    )
                )
                selected_match_body_ids[linked_value] = tuple(
                    current_group_body_ids.get(
                        linked_value,
                        selected_match_body_ids.get(linked_value, ()),
                    )
                )
                for primary_value in match_group_primary.get(
                    linked_value, (linked_value,)
                ):
                    if (
                        primary_value
                        and primary_value not in selected_match_order
                    ):
                        selected_match_order.append(primary_value)

        def forget_match(value: str) -> None:
            value = str(value or "").strip()
            linked_values = set(related_match_values(value))
            if not linked_values:
                linked_values = {value}
            for linked_value in linked_values:
                selected_match_values.discard(linked_value)
                selected_match_members.pop(linked_value, None)
                selected_match_body_ids.pop(linked_value, None)
            selected_match_order[:] = [
                value for value in selected_match_order
                if value not in linked_values
            ]

        def remember_subtype(
            value: str,
            member_keys,
            body_ids,
        ) -> None:
            """Record one expanded subtype with its exact membership.

            Subtypes reuse the named-selection maps so query chips, body-ID
            resolution, chip removal, and cross-search persistence behave
            exactly like a match group picked from the panel.
            """
            value = str(value or "").strip()
            if not value:
                return
            selected_match_values.add(value)
            selected_match_members[value] = {
                str(key) for key in member_keys or () if str(key or "").strip()
            }
            selected_match_body_ids[value] = tuple(
                str(body_id) for body_id in body_ids or () if str(body_id or "").strip()
            )
            if value not in selected_match_order:
                selected_match_order.append(value)

        def row_body_id(row) -> str:
            """Return a verified, query-safe body ID for an individual row."""
            value = str(row.get("bodyId", "") or "").strip()
            if not value:
                return ""
            integer, dot, fraction = value.partition(".")
            if dot and integer.isdigit() and fraction and set(fraction) == {"0"}:
                return integer
            return value

        def selected_query_values() -> List[str]:
            values: List[str] = []
            seen: set[str] = set()
            # Keep the user's selection order even when a later search no
            # longer displays an earlier selected match group.
            for value in selected_match_order:
                if value in selected_match_values and value not in seen:
                    values.append(value)
                    seen.add(value)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                value = str(selected_body_ids.get(key, "") or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            for value in selected_body_ids.values():
                value = str(value or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            return values

        def selected_query_body_ids() -> List[str]:
            """Resolve selections to exact body IDs for query execution.

            The compact match panel displays human-readable names, but a name
            can occur in multiple metadata columns. Passing the resolved IDs
            separately prevents the eventual query from re-running a priority
            string lookup and selecting a different column by accident.
            """
            values: List[str] = []
            seen: set[str] = set()
            for value in selected_match_order:
                if value not in selected_match_values:
                    continue
                for body_id in selected_match_body_ids.get(value, ()):
                    body_id = str(body_id or "").strip()
                    if body_id and body_id not in seen:
                        values.append(body_id)
                        seen.add(body_id)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                if key not in selected_body_ids:
                    continue
                body_id = row_body_id(row)
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            for body_id in selected_body_ids.values():
                body_id = str(body_id or "").strip()
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            return values

        # Deferred apply (layer editor): hold the selection in the panel, then
        # commit it once when the dialog closes. Live mode pushes on every toggle.
        if defer_apply:
            def apply_pending_selection() -> None:
                if query_selection is not None:
                    query_selection(selected_query_values())
                if query_resolution is not None:
                    query_resolution(selected_query_body_ids())
            if apply_holder is not None:
                apply_holder["fn"] = apply_pending_selection
        if query_values_getter is not None:
            refresh_query_preview()

        def sync_query_selection() -> None:
            if not defer_apply:
                if query_callback is not None:
                    query_callback(selected_query_values())
                if query_resolution is not None:
                    query_resolution(selected_query_body_ids())
            if query_values_getter is not None:
                refresh_query_preview()

        def update_selection_status() -> None:
            if selection_status is not None:
                selected_count = len(selected_match_values) + len(selected_body_ids)
                selection_status.text = f"{selected_count} selected"
                selection_status.update()

        def refresh_table_selection() -> None:
            # Expansion checkboxes render from row data, so their checked
            # flags must be re-stamped before the table pushes an update.
            for group in current_groups:
                display = group.get("__subtypes")
                if not display:
                    continue
                for subtype in display.get("subtypes", ()):
                    subtype["selected"] = (
                        str(subtype.get("match_value", "") or "").strip()
                        in selected_match_values
                    )
            if match_table is not None:
                visible_match_rows = [
                    row for row in current_groups
                    if row.get("match_role") != "secondary"
                    and str(row.get("match_value", "") or "") in selected_match_values
                ]
                if match_header_selection["all_visible"]:
                    selectable_match_values = {
                        str(row.get("match_value", "") or "").strip()
                        for row in current_groups
                        if row.get("match_role") != "secondary"
                    }
                    if (
                        selectable_match_values
                        and selectable_match_values.issubset(selected_match_values)
                    ):
                        # Secondary rows are display-only and have no row
                        # checkbox. Keep them in QTable's internal selection
                        # only after a header select-all so Quasar can show
                        # the header as fully checked.
                        visible_match_rows.extend(
                            row for row in current_groups
                            if row.get("match_role") == "secondary"
                        )
                match_table.selected = visible_match_rows
                match_table.update()
            if table is not None:
                active = effective_body_keys()
                table.selected = [
                    row for row in current_rows
                    if str(row.get("__neuron_key", "") or "") in active
                ]
                table.update()
            update_selection_status()

        def handle_match_selection(event) -> None:
            visible_row_keys = {
                str(row.get("__match_group_key", "") or "")
                for row in current_groups
            }
            selected_row_keys = {
                str(row.get("__match_group_key", "") or "")
                for row in list(getattr(event, "selection", []) or [])
            }
            match_header_selection["all_visible"] = bool(
                visible_row_keys and visible_row_keys.issubset(selected_row_keys)
            )
            previously_selected = set(selected_match_values)
            visible_rows = [
                row for row in current_groups
                if row.get("match_role") != "secondary"
            ]
            visible_values = {
                str(row.get("match_value", "") or "").strip()
                for row in visible_rows
            }
            selected_rows = [
                row for row in list(getattr(event, "selection", []) or [])
                if row.get("match_role") != "secondary"
            ]
            raw_selected_values = {
                str(row.get("match_value", "") or "").strip()
                for row in selected_rows
            }
            selected_values: set[str] = set()
            for value in raw_selected_values:
                selected_values.update(related_match_values(value))
            # The table emits only the current result rows. Update those rows
            # while preserving selections made in an earlier search.
            for value in visible_values - selected_values:
                forget_match(value)
            for row in selected_rows:
                remember_match(str(row.get("match_value", "") or "").strip())
            # A matched-name selection includes every body in that group,
            # including rows on later pages. The full table mirrors the
            # current page of that selection when it is visible.
            sync_query_selection()
            # Selection and clicking a matched value use the same focus path:
            # resolve the first exact member, compute its sorted data page,
            # then scroll the actual metadata row into view.
            # QTable reports the complete selected set, not the row that was
            # just clicked. Focus the newly selected row's members so adding
            # aMe26 after aMe1/aMe13 does not jump back to the first group.
            new_values = [
                value for value in raw_selected_values
                if value not in previously_selected
            ]
            focus_value = new_values[-1] if new_values else (
                next(iter(raw_selected_values), "")
            )
            focus_keys = match_member_keys(focus_value)
            if selected_rows and focus_keys:
                return request_focus(focus_keys, anchor_key=focus_keys[0])
            else:
                refresh_table_selection()

        def handle_match_toggle(event) -> None:
            """Apply one checkbox toggle without trusting stale QTable state.

            The match table is re-rendered after every query and selection
            change.  Binding the slot checkbox directly with ``v-model`` can
            therefore make a second click report the previous selection back
            to the server.  The slot emits the requested next state instead;
            this handler updates the persistent selection sets first and then
            refreshes the table from those sets.
            """
            args = getattr(event, "args", None)
            if not isinstance(args, dict):
                return
            row = args.get("row")
            if not isinstance(row, dict):
                return
            if row.get("match_role") == "secondary":
                return
            match_header_selection["all_visible"] = False
            value = str(row.get("match_value", "") or "").strip()
            if not value:
                return
            if bool(args.get("selected")):
                remember_match(value)
            else:
                forget_match(value)
            sync_query_selection()
            if bool(args.get("selected")):
                focus_keys = match_member_keys(value)
                if focus_keys:
                    return request_focus(focus_keys, anchor_key=focus_keys[0])
            refresh_table_selection()

        def build_subtype_entry(group) -> dict:
            """Compute and cache the subtype panel for one match group.

            The server-side entry keeps the full selection payload (member
            keys and body IDs); only the compact display copy is attached to
            the row so a 500-subtype expansion never ships its whole
            membership over the websocket.
            """
            key = str(group.get("__match_group_key", "") or "")
            members = tuple(sorted(group_members.get(key, ())))
            payload = query_match_group_subtypes(index, members)
            display = {
                "subtypes": [
                    {
                        "match_value": subtype["match_value"],
                        "body_count": subtype["body_count"],
                        "selected": subtype["match_value"] in selected_match_values,
                    }
                    for subtype in payload["subtypes"]
                ],
                "total_types": payload["total_types"],
                "truncated": payload["truncated"],
            }
            entry = {
                "members": set(members),
                "payload": payload,
                "display": display,
                "expanded": False,
            }
            subtype_expansions[key] = entry
            return entry

        def handle_match_expand_toggle(event) -> None:
            key = str(getattr(event, "args", "") or "").strip()
            if not key:
                return
            group = next(
                (
                    candidate
                    for candidate in match_groups_all
                    if str(candidate.get("__match_group_key", "") or "") == key
                ),
                None,
            )
            if group is None or not group.get("__can_expand"):
                return
            entry = subtype_expansions.get(key)
            if entry is None or set(group_members.get(key, ())) != entry["members"]:
                entry = build_subtype_entry(group)
            entry["expanded"] = not entry["expanded"]
            group["__subtypes"] = entry["display"]
            group["__expanded"] = entry["expanded"]
            match_table.update_rows(current_groups)

        def handle_subtype_toggle(event) -> None:
            """Apply one expanded-subtype checkbox toggle."""
            args = getattr(event, "args", None)
            if not isinstance(args, dict):
                return
            group_key = str(args.get("group", "") or "").strip()
            value = str(args.get("value", "") or "").strip()
            if not value:
                return
            if bool(args.get("selected")):
                subtype = None
                entry = subtype_expansions.get(group_key)
                if entry is not None:
                    subtype = next(
                        (
                            candidate
                            for candidate in entry["payload"]["subtypes"]
                            if str(candidate.get("match_value", "")) == value
                        ),
                        None,
                    )
                if subtype is not None:
                    remember_subtype(
                        value,
                        subtype.get("member_keys", ()),
                        subtype.get("body_ids", ()),
                    )
                else:
                    remember_match(value)
            else:
                forget_match(value)
            sync_query_selection()
            refresh_table_selection()
            match_table.update_rows(current_groups)

        def handle_body_selection(event) -> None:
            visible = {
                str(row.get("__neuron_key", "") or ""): row
                for row in current_rows
            }
            selected_keys = {
                str(row.get("__neuron_key", "") or "")
                for row in list(getattr(event, "selection", []) or [])
            }
            # Unchecking one body row breaks a whole-name selection. The
            # remaining checked rows become individual selections.
            for value in list(selected_match_values):
                visible_group = (
                    selected_match_members.get(value, set())
                    | group_members.get(value, set())
                ) & set(visible)
                if visible_group and not visible_group.issubset(selected_keys):
                    forget_match(value)

            active_group_keys = set()
            for value in selected_match_values:
                active_group_keys.update(group_members.get(value, set()))
            for key, row in visible.items():
                if key not in selected_keys:
                    selected_body_ids.pop(key, None)
                elif key not in active_group_keys:
                    value = row_body_id(row)
                    if value:
                        selected_body_ids[key] = value
            sync_query_selection()
            refresh_table_selection()

        def remove_query_value(value: str) -> None:
            """Remove one value from the mirrored/Selected preview and selection state."""
            value = str(value or "").strip()
            if not value:
                return
            forget_match(value)
            for key, body_id in list(selected_body_ids.items()):
                if str(body_id or "").strip() == value:
                    selected_body_ids.pop(key, None)
            sync_query_selection()
            # In live mode the 'x' also removes from the owning input; in
            # deferred mode it only prunes the pending in-panel selection.
            if not defer_apply and query_remove is not None:
                query_remove(value)
            if query_values_getter is not None:
                refresh_query_preview()
            refresh_table_selection()

        def handle_full_table_select_all(_event=None) -> None:
            """Select or deselect the current query's rows across all pages.

            The header checkbox only reports a click; the server decides the
            direction by checking whether the whole result set is already
            selected, so a partially-selected page always resolves to
            "select-all" and a fully-selected result toggles back off.
            Because selections persist across searches, deselecting only
            removes the rows currently matching the query rather than wiping
            selections made against an earlier query.
            """
            keys = current_full_table_keys()
            if not keys:
                return
            if set(keys).issubset(effective_body_keys()):
                for key in keys:
                    selected_body_ids.pop(key, None)
            else:
                selected_body_ids.update(keys)
            sync_query_selection()
            refresh_table_selection()

        with ui.element("div").classes("w-full drocat-neuron-results-layout"):
            with ui.element("section").classes("drocat-neuron-match-panel"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("manage_search", color="primary").classes("text-lg")
                        ui.label("Match details").classes("text-subtitle2 font-bold")
                    with ui.row().classes("items-center gap-2"):
                        if query_callback is not None:
                            selection_status = ui.label("0 selected").classes(
                                "text-caption drocat-muted"
                            )
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap drocat-neuron-panel-toolbar"
                ):
                    match_status = ui.label("No matched names").classes(
                        "text-caption drocat-muted flex-grow"
                    )
                    match_page_position = ui.label("Page 1 of 1").classes(
                        "text-caption drocat-muted"
                    )
                    with ui.row().classes("items-center gap-1"):
                        match_previous_button = ui.button(
                            "Previous matches", icon="chevron_left"
                        ).props("flat dense")
                        match_next_button = ui.button(
                            "Next matches", icon="chevron_right"
                        ).props("flat dense")
                match_table = ui.table(
                    rows=current_groups,
                    columns=match_columns,
                    row_key="__match_group_key",
                    selection="multiple",
                    on_select=handle_match_selection,
                    pagination=None,
                ).classes("w-full drocat-neuron-match-table")
                # The custom body slot adds the selection cell explicitly.
                # Render the matching header cell explicitly as well so it
                # uses the same width and alignment as the row checkboxes.
                match_table.add_slot(
                    "header",
                    r"""
                    <q-tr :props="props">
                      <q-th auto-width class="drocat-neuron-match-select-cell">
                        <q-checkbox
                          v-model="props.selected"
                          :indeterminate="props.selected === null"
                          dense
                        />
                      </q-th>
                      <q-th
                        v-for="col in props.cols"
                        :key="col.name"
                        :props="props"
                        :class="col.headerClasses"
                        :style="col.headerStyle"
                      >
                        {{ col.label }}
                      </q-th>
                    </q-tr>
                    """,
                )
                match_table.add_slot(
                    "body",
                    r"""
                    <q-tr
                      :props="props"
                      :class="{
                        'drocat-neuron-match-secondary-row': props.row.match_role === 'secondary',
                      }"
                    >
                      <q-td auto-width class="drocat-neuron-match-select-cell">
                      <q-checkbox
                        v-if="props.row.match_role !== 'secondary'"
                        :model-value="props.selected"
                        dense
                        @click.stop="$parent.$emit('match-selection-toggle', { row: props.row, selected: !props.selected })"
                      />
                      </q-td>
                      <q-td key="match_column" :props="props" class="drocat-neuron-match-by">
                        <div class="drocat-neuron-match-source">
                          <q-btn
                            v-if="props.row.__can_expand"
                            flat dense round size="xs"
                            class="drocat-neuron-match-expand-btn"
                            :icon="props.row.__expanded ? 'expand_less' : 'expand_more'"
                            @click.stop="$parent.$emit('match-expand-toggle', props.row.__match_group_key)"
                          />
                          <q-icon
                            v-if="props.row.match_role === 'secondary'"
                            name="arrow_right_alt"
                            class="drocat-neuron-match-secondary-arrow"
                            size="18px"
                          />
                          {{ props.row.match_column }}
                        </div>
                      </q-td>
                      <q-td key="match_value" :props="props">
                        <div class="drocat-neuron-match-value-line">
                          <q-btn
                            flat dense no-caps
                            class="drocat-neuron-match-jump"
                            :label="props.row.match_value"
                            @click.stop="$parent.$emit('match-value-click', props.row)"
                          />
                        </div>
                        <div
                          v-if="props.row.first_body_id"
                          class="drocat-neuron-match-first"
                        >first bodyId: {{ props.row.first_body_id }}</div>
                      </q-td>
                      <q-td key="body_count" :props="props" class="text-right">
                        {{ props.row.body_count }}
                      </q-td>
                    </q-tr>
                    <q-tr
                      v-if="props.row.__expanded && props.row.__subtypes"
                      class="drocat-neuron-match-subtype-panel-row"
                    >
                      <q-td colspan="4" class="drocat-neuron-match-subtype-cell">
                        <div class="drocat-neuron-match-subtype-head">
                          {{ props.row.__subtypes.total_types }}
                          {{ props.row.__subtypes.total_types === 1 ? 'type' : 'types' }}
                          in {{ props.row.match_value }}
                        </div>
                        <div class="drocat-neuron-match-subtype-list">
                          <div
                            v-for="subtype in props.row.__subtypes.subtypes"
                            :key="subtype.match_value"
                            class="drocat-neuron-match-subtype-item"
                          >
                            <q-checkbox
                              :model-value="!!subtype.selected"
                              dense
                              @click.stop="$parent.$emit('match-subtype-toggle', { group: props.row.__match_group_key, value: subtype.match_value, selected: !subtype.selected })"
                            />
                            <span class="drocat-neuron-match-subtype-name">{{ subtype.match_value }}</span>
                            <span class="drocat-neuron-match-subtype-count">{{ subtype.body_count }}</span>
                          </div>
                          <div
                            v-if="!props.row.__subtypes.subtypes.length"
                            class="drocat-neuron-match-subtype-note"
                          >
                            No type values in this entry.
                          </div>
                          <div
                            v-if="props.row.__subtypes.truncated"
                            class="drocat-neuron-match-subtype-note"
                          >
                            Showing the first {{ props.row.__subtypes.subtypes.length }} of
                            {{ props.row.__subtypes.total_types }} types. Refine by searching the type name.
                          </div>
                        </div>
                      </q-td>
                    </q-tr>
                    """,
                )
            with ui.element("section").classes("drocat-neuron-full-panel"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.icon("table_view", color="primary").classes("text-lg")
                    ui.label("Full neuron metadata").classes("text-subtitle2 font-bold")
                ui.label(
                    "Scroll horizontally to inspect every retained metadata field."
                ).classes("text-caption drocat-muted")
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap drocat-neuron-panel-toolbar"
                ):
                    page_info = ui.label("").classes(
                        "text-caption drocat-muted flex-grow"
                    )
                    no_results = ui.label(
                        "No rows match the current search/filter."
                    ).classes("text-caption drocat-warn")
                    page_position = ui.label("").classes(
                        "text-caption drocat-muted"
                    )
                    with ui.row().classes("items-center gap-1"):
                        previous_button = ui.button(
                            "Previous page", icon="chevron_left"
                        ).props("flat dense")
                        next_button = ui.button(
                            "Next page", icon="chevron_right"
                        ).props("flat dense")
                        # Global matched-rows export: works in any search
                        # state (hits, zero hits, mapped view). Late-bound
                        # like the banner buttons: the handler is defined
                        # further down in this function.
                        export_rows_button = ui.button(
                            "Export matched rows (CSV)", icon="download"
                        ).props("flat dense").on_click(
                            lambda: _export_matched_rows()
                        )
                        export_rows_button.tooltip(
                            "Download every row matching the current query, "
                            "across all pages")
                    with ui.element("section").classes(
                        "w-full drocat-mapped-warning"
                    ) as mapped_warning_section:
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.icon("warning", color="warning")
                            mapped_warning_label = ui.label("").classes(
                                "text-subtitle2 text-warning flex-grow"
                            )
                            with ui.button(
                                "View Sankey", icon="multiple_stop"
                            ).props("flat dense"):
                                with ui.menu():
                                    ui.menu_item(
                                        "Type-level",
                                        on_click=lambda: (
                                            _view_active_mapping(
                                                "sankey", "type")))
                                    ui.menu_item(
                                        "Linker view",
                                        on_click=lambda: (
                                            _view_active_mapping(
                                                "sankey", "linker")))
                            with ui.button(
                                "View Network", icon="account_tree"
                            ).props("flat dense"):
                                with ui.menu():
                                    ui.menu_item(
                                        "Type-level",
                                        on_click=lambda: (
                                            _view_active_mapping(
                                                "network", "type")))
                                    ui.menu_item(
                                        "Linker view",
                                        on_click=lambda: (
                                            _view_active_mapping(
                                                "network", "linker")))
                            ui.button(
                                "Back to normal search", icon="undo"
                            ).props("flat dense").on_click(lambda: _exit_mapped_view())
                    mapped_warning_section.set_visibility(False)
                with ui.element("div").classes("w-full drocat-data-viewer-scroll"):
                    table = ui.table(
                        rows=initial.rows,
                        columns=table_columns,
                        row_key="__neuron_key",
                        selection="multiple",
                        on_select=handle_body_selection,
                        pagination=None,
                    ).classes("w-full drocat-data-viewer-table")
                    # Keep the table's default selection checkbox at the left
                    # while using a body slot for the row-specific highlight.
                    table.add_slot(
                        "body",
                        r"""
                        <q-tr
                          :data-neuron-key="props.row.__neuron_key"
                          :class="{
                            'drocat-neuron-selected-row': props.selected,
                          }"
                        >
                          <q-td auto-width class="drocat-neuron-select-cell" style="width: 48px; min-width: 48px;">
                            <q-checkbox v-model="props.selected" dense />
                          </q-td>
                          <q-td
                            v-for="col in props.cols"
                            :key="col.name"
                            :props="props"
                            :class="{
                              'drocat-neuron-hit-cell': (
                                props.row.match_column_keys || [props.row.match_column_key]
                              ).includes(col.name),
                              'drocat-neuron-secondary-hit-cell': (
                                props.row.secondary_match_column_keys || []
                              ).includes(col.name),
                              'drocat-neuron-map-cell': (col.name || '').startsWith('__map_'),
                            }"
                            :style="col.style || ''"
                            :data-match-column="(
                              props.row.match_column_keys || [props.row.match_column_key]
                            ).includes(col.name) ? col.name : null"
                            :data-match-role="(
                              props.row.secondary_match_column_keys || []
                            ).includes(col.name) ? 'secondary' : null"
                          >
                            <div
                              v-if="(col.name || '').startsWith('__map_')"
                              class="drocat-neuron-map-value"
                              @click.stop="$event.currentTarget.classList.toggle('drocat-map-cell-expanded')"
                            >
                              <span
                                v-if="props.row.__highlighted_cells && props.row.__highlighted_cells[col.name]"
                                v-html="props.row.__highlighted_cells[col.name]"
                              ></span>
                              <span v-else>{{ props.row[col.field] }}</span>
                            </div>
                            <span
                              v-else-if="props.row.__highlighted_cells && props.row.__highlighted_cells[col.name]"
                              v-html="props.row.__highlighted_cells[col.name]"
                            ></span>
                            <span v-else>{{ props.row[col.field] }}</span>
                            <q-tooltip
                              v-if="(col.name || '').startsWith('__map_') && props.row.__map_bridge"
                              class="drocat-map-cell-tooltip"
                              anchor="top middle"
                              self="bottom middle"
                            >{{ props.row.__map_bridge }}</q-tooltip>
                          </q-td>
                        </q-tr>
                        """,
                    )
                    # The header select-all mirrors the row checkboxes but
                    # selects every matching row across all pages, not just the
                    # current page. The checkbox is view-only (``:model-value``)
                    # so the server can drive the selection and refresh the whole
                    # table from the persistent key set.
                    table.add_slot(
                        "header",
                        r"""
                        <q-tr :props="props">
                          <q-th auto-width class="drocat-neuron-select-cell" style="width: 48px; min-width: 48px;">
                            <q-checkbox
                              :model-value="props.selected"
                              :indeterminate="props.selected === null"
                              dense
                              @click.stop="$parent.$emit('full-table-select-all')"
                            />
                          </q-th>
                          <q-th
                            v-for="col in props.cols"
                            :key="col.name"
                            :props="props"
                            :class="{
                              'drocat-neuron-map-cell': (col.name || '').startsWith('__map_'),
                            }"
                            :style="col.headerStyle || ''"
                          >
                            <div
                              v-if="(col.name || '').startsWith('__map_')"
                              class="drocat-neuron-map-value"
                            >{{ col.label }}</div>
                            <template v-else>{{ col.label }}</template>
                          </q-th>
                        </q-tr>
                        """,
                    )

        state = {"page": initial.page, "page_size": 50}
        # Total of the most recent query. The cross-dataset mode toggle
        # re-evaluates only the results footer with it — the table query
        # itself does not change when the mode flips.
        last_result_total = {"value": 0}
        # (column, value) pairs of the most recent query's match groups.
        # They feed the value-driven mapper fallback: non-type entries
        # (a cell_type value, for example) translate to other datasets'
        # types through the local neurons carrying them.
        last_matched_values: List[tuple] = []

        def _hide_alias_panel() -> None:
            """Hide the panel and void any alias scan still in flight."""
            alias_scan["current_key"] = None
            alias_section.set_visibility(False)

        def _export_matches_csv() -> None:
            """Download every matched entry of the expansion as a CSV file.

            Exported uncapped: all type/label matches (including the ones the
            panel summarizes behind "+N more") plus the query-alias
            candidates, for every dataset with native matches.
            """
            try:
                csv_text = build_matches_csv(
                    dataset, str(search_input.value or "").strip(),
                    matched_values=list(last_matched_values),
                    prefix_only_search=current_prefix_only_search())
            except Exception as exc:
                ui.notify(f"Export failed: {exc}", type="negative")
                return
            if not csv_text:
                ui.notify("No matched entries to export.", type="info")
                return
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = (
                f"matched_entries_{dataset.replace(':', '_')}_{stamp}.csv")
            ui.download.content(csv_text, name, "text/csv")
            # Same persistent orange banner as every other download
            # artifact (user 2026-09-10: unified, collapsible notices)
            push_banner(
                f"Matched entries exported as {name} — check your "
                "browser's default downloads folder. Informational "
                "only, please double check.")

        # Broad queries must not freeze the server building a giant CSV.
        EXPORT_ROWS_CAP = 100_000

        def _export_matched_rows() -> None:
            """Download every row matching the current query (all pages).

            Same filter/sort as the on-screen table (the mapped view exports
            its type set); the CSV carries every retained metadata column.
            """
            try:
                result = query_neuron_index(
                    index,
                    **query_kwargs_with_mapped(),
                    page=1,
                    page_size=50,
                    include_all_rows=True,
                )
            except Exception as exc:
                ui.notify(f"Export failed: {exc}", type="negative")
                return
            if result.total > EXPORT_ROWS_CAP:
                ui.notify(
                    f"{result.total:,} matching rows exceed the "
                    f"{EXPORT_ROWS_CAP:,}-row export cap — refine the query.",
                    type="warning",
                )
                return
            if not result.rows:
                ui.notify("No rows match the current search/filter.",
                          type="info")
                return
            import csv
            import io

            # Mapped view: PREPEND the mapping provenance columns (§9.3) —
            # the foreign dataset/types, the matched column, and one
            # bridge-<column> cell per standardized linker column — so the
            # foreign/bridge columns lead the CSV, ahead of this dataset's
            # metadata columns.
            extra_fieldnames: List[str] = []
            extras: List[Dict[str, str]] = []
            if mapped_view.get("active"):
                extra_fieldnames, extras = mapped_csv_extras(
                    result.rows, mapped_view.get("provenance", {}),
                    mapped_view.get("foreign_dataset", ""))
            buffer = io.StringIO()
            fieldnames = (
                extra_fieldnames
                + [column for column in columns if column in result.rows[0]]
            )
            writer = csv.DictWriter(
                buffer, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            if extras:
                for row, extra in zip(result.rows, extras):
                    writer.writerow({**row, **extra})
            else:
                writer.writerows(result.rows)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"{dataset.replace(':', '_')}_matched_rows_{stamp}.csv"
            ui.download.content(buffer.getvalue(), name, "text/csv")
            # Same persistent orange banner as every other download
            # artifact (user 2026-09-10: unified, collapsible notices)
            push_banner(
                f"Matched rows exported as {name} — check your "
                "browser's default downloads folder. Informational "
                "only, please double check.")

        def _search_local_alias(name: str) -> None:
            """Refill the local search with an alias found in this dataset."""
            if str(search_input.value or "").strip() == name:
                # set_value would not fire the change handler for an equal
                # value; refresh explicitly (and leave any mapped view).
                mapped_view.clear()
                _restore_map_columns()
                mapped_warning_section.set_visibility(False)
                return _dispatch_refresh(reset_page=True)
            else:
                # reset_and_refresh clears the mapped view via the value
                # change handler.
                search_input.set_value(name)

        def _render_alias_status(text: str, *, busy: bool) -> None:
            """One-line zero-hit status row with an optional busy spinner."""
            alias_section.set_visibility(True)
            alias_container.clear()
            with alias_container:
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    ui.icon("travel_explore", color="warning").classes("text-lg")
                    ui.label(text).classes("text-subtitle2 font-bold")
                    if busy:
                        ui.spinner(type="hourglass", size="lg")

        def render_alias_matches() -> None:
            """Cross-dataset panel: instant status, mapper scan off-loop.

            A zero-hit query renders the panel expanded (it is the only
            content there is); with hits, the cross-dataset mode renders the
            same scan as a collapsed expansion above the results. The auto
            type mapper initializes lazily and its first scan can take a
            while, so the panel shows an explicit status immediately and the
            scan runs in a dedicated worker (a spawned process for a cold
            mapper, otherwise a thread). The current-key check voids
            results superseded by a newer query or a hidden panel; the last
            completed scan is cached per query so paging and display changes
            never re-run it. A forced one-character query carries the same
            starts-with-only bound into the native cross-dataset scan.
            """
            query_text = str(search_input.value or "").strip()
            prefix_only_search = current_prefix_only_search()
            zero_hit = last_result_total["value"] == 0
            matched_pairs = list(last_matched_values)
            cache_key = (
                dataset,
                query_text,
                prefix_only_search,
                zero_hit,
                tuple(sorted((c.casefold(), v.casefold())
                             for c, v in matched_pairs)),
            )
            alias_scan["current_key"] = cache_key

            def _apply(matches) -> None:
                native = (matches or {}).get("native", [])
                mapped = (matches or {}).get("mapped", [])
                value_blocks = (matches or {}).get("value_mapped", [])
                guidance = (matches or {}).get("guidance", [])
                native_useful = any(
                    entry.get("types") or entry.get("labels") for entry in native
                )
                mapped_useful = any(
                    entry["outcome"] == "matched" and entry["candidates"]
                    for entry in mapped
                )
                value_useful = any(
                    entry.get("types") or entry.get("labels")
                    for entry in value_blocks
                )
                if not (native_useful or mapped_useful or value_useful):
                    # The scan finished without a single counterpart anywhere;
                    # keep the state explicit instead of hiding it.
                    guidance_note = (
                        " For datasets without an automatic mapping, track "
                        "them in the Cross-Dataset tab's type mapping panel "
                        "(download their metadata first)."
                        if guidance else ""
                    )
                    if zero_hit:
                        _render_alias_status(
                            f"No matches for '{query_text}' in {dataset}, and the "
                            "other cached datasets have no cross-dataset "
                            f"counterparts either.{guidance_note}",
                            busy=False,
                        )
                    else:
                        _render_alias_status(
                            f"No cross-dataset counterparts for '{query_text}' "
                            f"in the other cached datasets.{guidance_note}",
                            busy=False,
                        )
                    return
                # Datasets with something to show drive the collapsed
                # expansion's caption and the closing "no counterpart" line.
                matched_datasets = {
                    entry["dataset"]
                    for entry in mapped
                    if entry["outcome"] == "matched" and entry["candidates"]
                }
                matched_datasets.update(
                    entry["dataset"]
                    for entry in native
                    if entry.get("types") or entry.get("labels")
                )
                matched_datasets.update(
                    entry["dataset"] for entry in value_blocks
                )
                # A cached re-render can follow a hidden panel (mapped-type
                # view exit) — visibility is the section's, not the cache's.
                alias_section.set_visibility(True)
                alias_container.clear()
                with alias_container:
                    if not zero_hit:
                        def _remember_expansion(event) -> None:
                            alias_scan["expanded"] = bool(event.value)

                        # Several cached versions can share one abbreviation
                        # (banc_v626 + banc_v888); the caption lists each
                        # abbreviation once.
                        caption_abbrevs: List[str] = []
                        for matched_dataset in sorted(matched_datasets):
                            abbrev = dataset_abbrev(matched_dataset)
                            if abbrev not in caption_abbrevs:
                                caption_abbrevs.append(abbrev)
                        co_expansion = ui.expansion(
                            f"Cross-dataset type mapping for '{query_text}'",
                            caption=(
                                "matched in "
                                + ", ".join(caption_abbrevs)
                                + " — informational only, please double check"
                            ),
                            icon="travel_explore",
                            value=bool(alias_scan["expanded"]),
                            on_value_change=_remember_expansion,
                        ).props("dense").classes("w-full drocat-neuron-alias-expansion")
                        slot = co_expansion
                    else:
                        slot = alias_container
                    with slot:
                        if zero_hit:
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"
                            ):
                                ui.icon("travel_explore", color="warning").classes(
                                    "text-lg")
                                ui.label(
                                    "No rows here. Cross-dataset matches — "
                                    "informational only, please double check."
                                ).classes("text-subtitle2 font-bold")
                                ui.button(
                                    "Export matched entries (CSV)",
                                    icon="download",
                                ).props("flat dense").on_click(_export_matches_csv)
                        else:
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"
                            ):
                                ui.button(
                                    "Export matched entries (CSV)",
                                    icon="download",
                                ).props("flat dense").on_click(_export_matches_csv)

                    def _multivalue_marker(type_name: str,
                                           dataset: str) -> str:
                        """` 🧩 multi (a|b)` when the release's own `type`
                        cell lists several candidate names (plan §5.3/D5).
                        Display-only."""
                        try:
                            from comparison.cross_dataset_type_mapper import (
                                get_type_mapper,
                            )
                            parts = get_type_mapper().multivalue_parts(
                                type_name, dataset)
                        except Exception:
                            parts = None
                        if not parts:
                            return ""
                        return (" 🧩 multi ("
                                + "|".join(str(x) for x in parts) + ")")

                    def _annotation_text(ann) -> str:
                        if not ann:
                            return "— no counterpart in this dataset"
                        # RES-6: a `claimed` annotation is evidence, not a
                        # mapping — the claimed target does not exist in
                        # this dataset and must not read as "maps to".
                        if ann.get("status") == "claimed":
                            return ("— crosswalk claims "
                                    f"'{ann['targets'][0]}', but this "
                                    "dataset carries no such type "
                                    "(evidence only)")
                        # RES-16: the suspects note comes before the kind
                        # dispatch — a FIRED selection with a rename kind
                        # used to display as a bare "maps to" with no
                        # rivals note.
                        if ann.get("suspects"):
                            _n = len(ann.get("suspect_rivals") or [])
                            _sel = (ann["targets"][0]
                                    if ann.get("targets") else "selected")
                            return (f"— same-name-first selection of "
                                    f"'{_sel}'; {_n} rival candidate(s) "
                                    "not selected (expand Suspects)")
                        if ann["kind"] == "one of N":
                            return "— here: one of " + ", ".join(ann["targets"])
                        if ann["kind"] == "renamed":
                            return "— here: maps to '" + ann["targets"][0] + "'"
                        if ann["kind"] == "conflict":
                            # A HELD same-name fan-out is a conflict with a
                            # reason (plan §5.2): say WHY nothing was
                            # selected instead of leaving a bare "conflict".
                            _riv = ann.get("suspect_rivals") or []
                            if _riv:
                                return (f"— none selected: the same-name "
                                        f"candidate was held; {len(_riv)} "
                                        "rival candidate(s) could not be "
                                        "excluded (see Suspects below)")
                            return "— unresolved mapping conflict"
                        if ann["kind"] == "same name":
                            # Three DIFFERENT evidential situations used to
                            # share one caveat, so curated identities and
                            # same-name-first selections were both labelled
                            # "no metadata verification" (plan-ui-type-
                            # mapper-alignment §5.1 / U7).
                            if ann.get("curated_identity"):
                                return ("— same name, backed by a curated "
                                        "cross-dataset relation")
                            return ("— same name in this dataset "
                                    "(no metadata verification — "
                                    "please double check)")
                        return f"— here: {ann['kind']} {', '.join(ann['targets'])}"

                    with slot:
                        def _render_block(entry) -> None:
                            """One dataset block: badge, action buttons, details."""
                            if not (entry.get("types") or entry.get("labels")):
                                return
                            mapped_names = entry.get("mapped_type_names", [])
                            if mapped_names:
                                # Button rows center the small badge on
                                # the flat dense buttons' axis (see the
                                # -centered rule in the app CSS).
                                with ui.element("div").classes(
                                    "drocat-neuron-alias-col-badge "
                                    "drocat-neuron-alias-badge-centered"
                                ):
                                    ui.badge(entry["dataset"]).props(
                                        "outline")
                            else:
                                ui.badge(entry["dataset"]).props(
                                    "outline").classes(
                                    "drocat-neuron-alias-col-badge"
                                )
                            # The button sits beside its dataset badge so it
                            # always names the block it expands. The mapped
                            # type count rides on the tooltip to keep the
                            # button compact.
                            if mapped_names:
                                ui.button(
                                    "Mapped types",
                                    icon="table_view",
                                ).props("flat dense").tooltip(
                                    f"Show these {len(mapped_names)} mapped "
                                    "types in the current dataset"
                                ).classes(
                                    "drocat-neuron-alias-col-mapped"
                                ).on_click(
                                    lambda _e=None, entry_ref=entry:
                                    _enter_mapped_view(entry_ref)
                                )
                                with ui.button(
                                    "Sankey", icon="multiple_stop"
                                ).props("flat dense").classes(
                                    "drocat-neuron-alias-col-sankey"
                                ):
                                    with ui.menu():
                                        ui.menu_item(
                                            "Type-level",
                                            on_click=lambda _e=None,
                                            ref=entry: (
                                                _view_mapping_visualization(
                                                    "sankey", "type", ref)))
                                        ui.menu_item(
                                            "Linker view",
                                            on_click=lambda _e=None,
                                            ref=entry: (
                                                _view_mapping_visualization(
                                                    "sankey", "linker", ref)))
                                with ui.button(
                                    "Network", icon="account_tree"
                                ).props("flat dense").classes(
                                    "drocat-neuron-alias-col-network"
                                ):
                                    with ui.menu():
                                        ui.menu_item(
                                            "Type-level",
                                            on_click=lambda _e=None,
                                            ref=entry: (
                                                _view_mapping_visualization(
                                                    "network", "type", ref)))
                                        ui.menu_item(
                                            "Linker view",
                                            on_click=lambda _e=None,
                                            ref=entry: (
                                                _view_mapping_visualization(
                                                    "network", "linker", ref)))
                            with ui.element("div").classes(
                                "drocat-neuron-alias-col-details"
                            ):
                                for cand in entry.get("types", []):
                                    text = (
                                        f"'{cand['name']}'"
                                        + _multivalue_marker(
                                            cand["name"], entry["dataset"])
                                        + f" ({cand['count']:,} neurons) "
                                        + _annotation_text(cand.get("mapped"))
                                    )
                                    ui.label(text).classes("text-caption")
                                if entry.get("types_truncated"):
                                    ui.label(
                                        f"+{entry['types_truncated']} more types"
                                    ).classes("text-caption drocat-muted")
                                for label in entry.get("labels", []):
                                    ui.label(
                                        f"label '{label['label']}' · "
                                        f"{label['column']} "
                                        f"({label['count']:,} neurons)"
                                    ).classes("text-caption")
                                    covered = [
                                        f"'{t['name']}' ({t['count']:,}) "
                                        + _annotation_text(t.get("mapped"))
                                        for t in label.get("types", [])
                                    ]
                                    if covered:
                                        ui.label(
                                            "    types under this label: "
                                            + "; ".join(covered)
                                        ).classes(
                                            "text-caption drocat-muted")
                                    if label.get("types_truncated"):
                                        ui.label(
                                            f"    +{label['types_truncated']} "
                                            "more types under this label"
                                        ).classes("text-caption drocat-muted")
                                if entry.get("labels_truncated"):
                                    ui.label(
                                        f"+{entry['labels_truncated']} more labels"
                                    ).classes("text-caption drocat-muted")

                        if native_useful:
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"):
                                ui.label(
                                    "Type-name matches in other datasets "
                                    "(name-similar — not necessarily the same type):"
                                ).classes("text-caption font-bold drocat-muted")
                            # One shared grid for every dataset block: the badge, the
                            # action buttons, and the detail text each keep their own
                            # column, so the same items align vertically across rows.
                            with ui.element("div").classes(
                                "w-full drocat-neuron-alias-grid"
                            ):
                                for entry in native:
                                    _render_block(entry)
                        if value_useful:
                            value_names = sorted({
                                str(value)
                                for _column, value in matched_pairs
                            })
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"):
                                ui.label(
                                    "Auto-mapped counterparts of the matched "
                                    f"value(s) {', '.join(value_names)} "
                                    "(via the type mapper — informational "
                                    "only, please double check):"
                                ).classes("text-caption font-bold drocat-muted")
                            with ui.element("div").classes(
                                "w-full drocat-neuron-alias-grid"
                            ):
                                for entry in value_blocks:
                                    _render_block(entry)
                        if guidance:
                            unmapped = [
                                f"{item['dataset']}"
                                + ("" if item.get("cached")
                                   else " (metadata not downloaded)")
                                for item in guidance
                            ]
                            ui.label(
                                "No automatic mapping for "
                                + ", ".join(unmapped)
                                + " — track them in the Cross-Dataset tab's "
                                "type mapping panel."
                            ).classes("text-caption drocat-muted")

                        if mapped_useful:
                            with ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"):
                                ui.label(
                                    "Auto type mapping:"
                                ).classes("text-caption font-bold drocat-muted")
                            # Same shared-column idea as the block grid above, reduced
                            # to badge + text so every dataset's text starts at the
                            # same x position.
                            with ui.element("div").classes(
                                "w-full drocat-neuron-alias-grid "
                                "drocat-neuron-alias-grid-mapped"
                            ):
                                for entry in mapped:
                                    if entry["outcome"] != "matched" or not entry["candidates"]:
                                        continue
                                    # The selected dataset's candidates carry
                                    # "Search here" buttons: center the badge on
                                    # that first row's axis like the native grid.
                                    if entry["is_selected"]:
                                        with ui.element("div").classes(
                                            "drocat-neuron-alias-col-badge "
                                            "drocat-neuron-alias-badge-centered"
                                        ):
                                            ui.badge(
                                                entry["dataset"] + " (this dataset)"
                                            ).props("outline")
                                    else:
                                        ui.badge(
                                            entry["dataset"]
                                        ).props("outline").classes(
                                            "drocat-neuron-alias-col-badge"
                                        )
                                    with ui.element("div").classes("min-w-0"):
                                        for cand in entry["candidates"]:
                                            text = ("'" + cand['name'] + "'"
                                                    + _multivalue_marker(
                                                        cand["name"],
                                                        entry["dataset"])
                                                    + f" — {cand['kind']}")
                                            if cand["kind"] == "same name":
                                                # Same three-way split as
                                                # _annotation_text (§5.1).
                                                if cand.get("suspects"):
                                                    text += (" (same-name-first "
                                                             "selection; rivals "
                                                             "not selected)")
                                                elif cand.get("curated_identity"):
                                                    text += (" (backed by a "
                                                             "curated cross-dataset "
                                                             "relation)")
                                                else:
                                                    text += (" (no metadata verification — "
                                                             "please double check)")
                                            if cand.get("asymmetry_note"):
                                                text += ("; "
                                                         + cand["asymmetry_note"])
                                            if cand.get("aggregates"):
                                                text += (
                                                    "; a match also covers: "
                                                    + ", ".join(cand["aggregates"])
                                                )
                                            if cand.get("count") is not None:
                                                text += f" ({cand['count']:,} neurons)"
                                            if cand["kind"] == "one of N":
                                                text += (
                                                    "; the parent family "
                                                    "splits by neuron")
                                            with ui.row().classes(
                                                "items-center gap-2 flex-wrap"
                                            ):
                                                ui.label(text).classes("text-caption")
                                                if entry["is_selected"]:
                                                    ui.button(
                                                        f"Search '{cand['name']}' here",
                                                        icon="search",
                                                    ).props("flat dense").on_click(
                                                        lambda _e=None, name=cand["name"]:
                                                        _search_local_alias(name)
                                                    )
                                            # COLLAPSED same-name-first details
                                            # (plan §5.2/D1): the rivals that were
                                            # NOT selected, one row each, from the
                                            # mapper's own evidence record.
                                            if cand.get("suspect_rivals"):
                                                _rivals = list(cand["suspect_rivals"])
                                                with ui.expansion(
                                                        f"Suspects — "
                                                        f"{len(_rivals)} rival "
                                                        f"candidate(s) not selected",
                                                        icon="warning").classes(
                                                            "text-caption"):
                                                    ui.label(
                                                        "These rival names were in "
                                                        "the fan-out's candidate "
                                                        "set but were NOT selected "
                                                        "(the same-name candidate "
                                                        "was). Rivals are never "
                                                        "merged; add a custom "
                                                        "label mapping if one "
                                                        "belongs to your analysis."
                                                    ).classes(
                                                        "text-caption drocat-muted")
                                                    for _r in _rivals:
                                                        ui.label(
                                                            f"• '{_r}'"
                                                        ).classes("text-caption")

                        unknown = [
                            entry["dataset"]
                            for entry in mapped
                            if entry["dataset"] not in matched_datasets
                        ]
                        if unknown:
                            ui.label(
                                "No known counterpart in: " + ", ".join(unknown)
                            ).classes("text-caption drocat-muted")

            # A completed scan is cached per query: paging and display-only
            # refreshes re-render the stored result instead of re-running
            # the mapper scan.
            cached = alias_scan["cache"]
            if cached["key"] == cache_key and cached["matches"] is not None:
                _apply(cached["matches"])
                return

            if alias_scan["inflight_key"] == cache_key:
                # A newer table refresh rebuilt the footer while the same
                # expansion is still running.  Keep one visible busy row and
                # let the existing worker publish its result for this key.
                _render_alias_status(
                    f"Checking the other cached datasets for '{query_text}'…",
                    busy=True,
                )
                return

            alias_scan["inflight_key"] = cache_key

            async def _scan_async() -> None:
                # Heavy scan off the event loop; NiceGUI elements are only
                # touched here, back on the loop, after the await.  A cold
                # mapper is process-isolated because its pandas/Python load
                # can starve the websocket even from a thread.
                # Let rapid typing settle first: a newer query re-rendered
                # the footer and replaced current_key, so a superseded
                # pending scan exits before it reaches the single-flight
                # boundary instead of queueing behind a cold worker and
                # running every broad intermediate query to completion.
                await asyncio.sleep(ALIAS_SCAN_SETTLE_SECONDS)
                if alias_scan["current_key"] != cache_key:
                    if alias_scan["inflight_key"] == cache_key:
                        alias_scan["inflight_key"] = None
                    return
                try:
                    # The generation check belongs inside the process-wide
                    # single-flight boundary: stale workers may be queued
                    # behind a cold R1 scan, and checking only after the call
                    # still lets each stale worker allocate its own mapper /
                    # Parquet working set.  A broad query is dispatched to
                    # the worker even with a warm local mapper: its
                    # enrichment is CPU-heavy enough to starve the
                    # websocket from a thread.
                    broad_query = (
                        last_result_total["value"] > BROAD_QUERY_LOCAL_HITS)
                    if not is_type_mapper_loaded() or broad_query:
                        matches = await run_cross_dataset_scan_in_process(
                            collect_zero_hit_matches_in_process,
                            dataset,
                            query_text,
                            matched_values=matched_pairs,
                            prefix_only_search=prefix_only_search,
                            # carry the caller's index root across the spawn
                            # boundary — a spawned worker re-imports this
                            # module with the real PROJECT_ROOT, which would
                            # otherwise scan indexes the caller never sees
                            index_root=str(PROJECT_ROOT),
                            is_current=lambda: (
                                alias_scan["current_key"] == cache_key),
                        )
                    else:
                        matches = await run.io_bound(
                            run_serialized_cross_dataset_scan,
                            collect_zero_hit_matches,
                            dataset,
                            query_text,
                            matched_values=matched_pairs,
                            prefix_only_search=prefix_only_search,
                            is_current=lambda: (
                                alias_scan["current_key"] == cache_key),
                            # Polled inside the warm in-process scan so a
                            # query typed while it runs abandons the stale
                            # enrichment instead of finishing it.
                            should_abort=lambda: (
                                alias_scan["current_key"] != cache_key),
                        )
                except Exception:
                    matches = None
                if alias_scan["inflight_key"] == cache_key:
                    alias_scan["inflight_key"] = None
                if matches is CROSS_SCAN_SUPERSEDED:
                    return
                if alias_scan["current_key"] != cache_key:
                    return  # a newer query or a hidden panel superseded this
                if matches is None:
                    _render_alias_status(
                        f"No matches for '{query_text}' in {dataset}. The "
                        "cross-dataset mapping check could not be completed.",
                        busy=False,
                    )
                else:
                    alias_scan["cache"]["key"] = cache_key
                    alias_scan["cache"]["matches"] = matches
                    _apply(matches)
                    if not is_type_mapper_loaded():
                        # §efficiency (user 2026-09-10): the isolated
                        # worker keeps its own mapper copy; without this
                        # warm the FIRST mapping-artifact download paid a
                        # synchronous ~7s mapper load inside the click
                        # handler.  Warm the UI-process copy off the
                        # event loop; the mapper's load lock makes a
                        # racing click safe.
                        async def _warm_ui_mapper() -> None:
                            from comparison.cross_dataset_type_mapper \
                                import get_type_mapper

                            await run.io_bound(get_type_mapper().load)

                        asyncio.ensure_future(_warm_ui_mapper())

            if zero_hit:
                if is_type_mapper_loaded():
                    status = (
                        f"No matches for '{query_text}' in {dataset}. Mapping "
                        "other datasets…"
                    )
                else:
                    status = (
                        f"No matches for '{query_text}' in {dataset}. Warming "
                        "the isolated auto type mapper; the app stays "
                        "responsive…"
                    )
                _render_alias_status(status, busy=True)
            else:
                if is_type_mapper_loaded():
                    status = (
                        f"Checking the other cached datasets for "
                        f"'{query_text}'…"
                    )
                else:
                    status = (
                        f"Checking the other cached datasets for "
                        f"'{query_text}' — warming the isolated auto type "
                        "mapper; the app stays responsive…"
                    )
                _render_alias_status(status, busy=True)
            # On the app loop (production), scan off-loop and render the
            # result back on the loop; without a running loop (direct /
            # test invocation) execute inline so results land
            # deterministically.
            try:
                asyncio.get_running_loop().create_task(_scan_async())
            except RuntimeError:
                asyncio.run(_scan_async())


        # Mapped-type view state: entered from the expansion panel's
        # "Mapped types" buttons; cleared by any query change or
        # the explicit exit button.  While active, the main table shows the
        # current dataset's neurons of the mapped types with the two floating
        # provenance columns (foreign types, matched column); the bridge
        # derivation rides on the cells as a hover title.  Rows remain
        # current-dataset neurons — selection adds them to the query
        # normally; foreign data only appears in the provenance columns.
        mapped_view: Dict[str, Any] = {}
        map_columns = [
            {"name": "__map_foreign", "label": "Foreign type(s)",
             "field": "__map_foreign", "sortable": False, "align": "left"},
            {"name": "__map_origin", "label": "Matched column",
             "field": "__map_origin", "sortable": False, "align": "left"},
        ]

        def _restore_map_columns() -> None:
            base = [c for c in table.columns
                    if not str(c.get("name", "")).startswith("__map_")]
            if len(base) != len(table.columns):
                table.columns = base
                table.update()

        def _apply_map_columns() -> None:
            foreign = mapped_view.get("foreign_dataset", "")
            widths = mapped_view.get("widths", {})
            foreign_w = widths.get("__map_foreign", 200)
            origin_w = widths.get("__map_origin", 200)
            styles = {
                "__map_foreign": (
                    f"position:sticky;right:{origin_w}px;z-index:6;"
                    f"width:{foreign_w}px;min-width:{foreign_w}px;"
                    f"max-width:{foreign_w}px;"
                    "background: var(--drocat-map-cell-bg) !important;"
                ),
                "__map_origin": (
                    "position:sticky;right:0px;z-index:6;"
                    f"width:{origin_w}px;min-width:{origin_w}px;"
                    f"max-width:{origin_w}px;"
                    "background: var(--drocat-map-cell-bg) !important;"
                ),
            }
            # Header cells carry the same sticky geometry; their background
            # is left to the CSS rule so the distinct head tint applies.
            header_styles = {
                "__map_foreign": (
                    f"position:sticky;right:{origin_w}px;z-index:9;"
                    f"width:{foreign_w}px;min-width:{foreign_w}px;"
                    f"max-width:{foreign_w}px;"
                ),
                "__map_origin": (
                    "position:sticky;right:0px;z-index:9;"
                    f"width:{origin_w}px;min-width:{origin_w}px;"
                    f"max-width:{origin_w}px;"
                ),
            }
            from utils.naming_utils import dataset_abbrev
            map_columns[0]["label"] = (
                f"Foreign type(s) · {dataset_abbrev(foreign)}")
            for col in map_columns:
                col["style"] = styles.get(col["name"], "")
                col["headerStyle"] = header_styles.get(col["name"], "")
            base = [c for c in table.columns
                    if not str(c.get("name", "")).startswith("__map_")]
            # the floating provenance columns are the table's LAST columns,
            # pinned to the right edge of the scroll area
            table.columns = base + map_columns
            table.update()

        def _exit_mapped_view() -> None:
            was_active = bool(mapped_view.get("active"))
            mapped_view.clear()
            _restore_map_columns()
            mapped_warning_section.set_visibility(False)
            if was_active:
                return _dispatch_refresh(reset_page=True)

        # §efficiency (user 2026-09-10): the four artifact buttons of one
        # block used to rebuild flows + pools on EVERY click.  Cache per
        # entry for the panel's lifetime; holding the entry in the value
        # keeps `id(entry)` keys unique while the panel is alive.
        _artifact_source_cache: Dict[int, tuple] = {}

        def _mapping_flows_and_pools(entry) -> tuple:
            """Flows + per-bridge bodyId pools for one foreign block.

            Flows drive every mapping artifact; pools (standardized
            linkers → pooled bodyIds per (source type, foreign type))
            feed the sankey ribbons and the linker-path graph.
            """
            cached = _artifact_source_cache.get(id(entry))
            if cached is not None and cached[0] is entry:
                return cached[1], cached[2]
            from comparison.mapping_visualization import build_mapping_flows

            source_counts = count_types_in_index(
                index, entry.get("mapped_type_names", []))
            flows = build_mapping_flows(
                [entry], dataset, source_counts=source_counts)
            if not flows:
                _artifact_source_cache[id(entry)] = (entry, [], {})
                return [], {}
            pools: Dict[tuple, Dict[str, Any]] = {}
            foreign_index = None
            foreign_ds = entry.get("dataset", "")
            if foreign_ds:
                try:
                    # Mapping artifacts need only endpoint bodyIds, types,
                    # and linker columns for coverage.  Loading the foreign
                    # wide display index here duplicated the largest table
                    # precisely when the user opened an R1 mapped view.
                    from ..neuron_index import _load_coverage_index
                    foreign_index = _load_coverage_index(foreign_ds)
                except Exception:
                    foreign_index = None
            for flow in flows:
                chains = [c for c in (flow.get("bridges") or [])
                          if c and c[-1].get("value") == flow.get(
                              "foreign_type")]
                if not chains:
                    continue
                pool = resolve_prioritized_bridge_pool(
                    dataset, foreign_ds, chains,
                    flow["source_type"], flow["foreign_type"],
                    indexes={
                        ds_name: ds_index for ds_name, ds_index in (
                            (dataset, index), (foreign_ds, foreign_index))
                        if ds_index is not None
                    })
                if pool.get("resolution_status") != "supported":
                    continue
                pools[(flow["source_type"], flow["foreign_type"])] = pool
            _artifact_source_cache[id(entry)] = (entry, flows, pools)
            return flows, pools

        def _render_mapping_artifact(kind: str, variant: str,
                                     flows, pools, foreign_ds: str
                                     ) -> Optional[str]:
            """Build one mapping artifact's HTML in memory.

            (kind, variant) is one of: sankey/type (two-band type-level
            flows) and sankey/linker (standardized linker bands) — both
            through the vispath sankey backend with the shared
            interactive control panel (user-adjustable node/edge
            colors) —, network/type (vispath dagre type-level graph),
            network/linker (vispath colored linker paths).  Nothing is
            written to the repository.
            """
            from comparison.mapping_visualization import (
                render_bridge_linker_html,
                render_mapping_network_html,
                render_mapping_sankey_html,
            )

            if kind == "sankey":
                return render_mapping_sankey_html(
                    flows, pools=pools, variant=variant)
            if variant == "linker":
                return render_bridge_linker_html(
                    flows, source_dataset=dataset,
                    target_dataset=foreign_ds, pools=pools)
            return render_mapping_network_html(flows, pools=pools)

        def _build_mapping_visualization(kind: str, variant: str,
                                         entry) -> Optional[tuple]:
            """Build one mapping artifact for a foreign block.

            Returns ``(html_text, file_name)``; delivery is a browser
            download with a persistent banner, so no server-side file is
            kept (the old ``outputs/`` writes are gone).
            """
            flows, pools = _mapping_flows_and_pools(entry)
            if not flows:
                return None
            foreign_ds = entry.get("dataset", "")
            html_text = _render_mapping_artifact(
                kind, variant, flows, pools, foreign_ds)
            if not html_text:
                return None
            stamp = time.strftime("%Y%m%d_%H%M%S")
            foreign = foreign_ds.replace(":", "_") or "dataset"
            name = f"mapping_{kind}_{variant}_{foreign}_{stamp}.html"
            return html_text, name

        def _deliver_mapping_visualization(html_text: str, name: str,
                                           label: str) -> None:
            """Browser download + persistent banner.  The copy lands
            in the browser's default downloads folder; nothing is saved
            server-side."""
            ui.download.content(html_text, name, "text/html")
            push_banner(
                f"{label} saved as {name} — check your browser's default "
                "downloads folder. Informational only, please double check.")

        def _view_mapping_visualization(kind: str, variant: str,
                                        entry) -> None:
            """Expansion-panel dispatch: build + deliver one artifact."""
            label = f"{kind.capitalize()} ({variant} view)"
            try:
                built = _build_mapping_visualization(kind, variant, entry)
            except Exception as exc:
                ui.notify(f"{label} failed: {exc}", type="negative")
                return
            if not built:
                ui.notify("Nothing to visualize.", type="info")
                return
            _deliver_mapping_visualization(built[0], built[1], label)

        def _enter_mapped_view(entry) -> None:
            """Run the equivalent search for a block's mapped types here.

            The main table switches to the current dataset's neurons of the
            mapped type names, with provenance columns.  Rows remain
            current-dataset neurons — selection adds them to the query
            normally; foreign data only appears in the provenance columns.
            """
            foreign = entry.get("dataset", "")
            types = sorted(entry.get("mapped_type_names", []))
            if not types:
                return
            written = entry.get("matched_written") or (
                str(search_input.value or "").strip()
            )
            # Value-driven blocks come from a non-type column entry (e.g.
            # cell_type · circadian_clock); the provenance origin names
            # that column instead of the type column.
            value_column = str(entry.get("value_source") or "").strip()
            origin_prefix = (
                f"{value_column} · '{written}'" if value_column
                else f"type · '{written}'"
            )
            provenance: Dict[str, List[Dict[str, Any]]] = {}
            column_texts: Dict[str, List[str]] = {
                "__map_foreign": [], "__map_origin": [],
            }

            def _collect(items: List[Dict[str, Any]], matched_origin: str) -> None:
                # Provenance per (local target, foreign type) pair, deduplicated
                # and built from the standardized linkers so every dataset
                # behaves identically (values included, hub routes flagged).
                from comparison.cross_dataset_type_mapper import bridge_linker_text

                # Cold cross-dataset scans run in an isolated worker.  Their
                # result already carries the bridge chains needed for this
                # view; loading the mapper again in the websocket process
                # would put the original cold-start failure back on the
                # user's click.  Warm, in-process results retain the fallback
                # for compatibility with older callers.
                mapper = None
                mapper_loaded = is_type_mapper_loaded()
                if mapper_loaded:
                    try:
                        from comparison.cross_dataset_type_mapper import get_type_mapper

                        mapper = get_type_mapper()
                    except Exception:
                        mapper = None
                for item in items:
                    ann = item.get("mapped")
                    if not ann:
                        continue
                    for target in ann.get("targets", []):
                        pair_key = (target, item["name"])
                        if pair_key in provenance_pairs:
                            continue
                        provenance_pairs.add(pair_key)
                        stored_bridges = item.get("bridges_by_target") or {}
                        if target in stored_bridges:
                            chains = stored_bridges.get(target) or []
                        elif mapper is not None:
                            try:
                                chains = mapper.get_type_bridges(
                                    target, dataset, foreign)
                            except Exception:
                                chains = []
                        else:
                            chains = []
                        linker_info = bridge_linker_text(
                            chains, dataset, foreign, item["name"])
                        row_entry = {
                            "foreign_type": item["name"],
                            "matched": matched_origin,
                            "origins": linker_info["entries"],
                            "source_text": linker_info["text"] or
                            NO_DERIVATION_TEXT,
                            "entry_text": item["name"],
                            "foreign_text": item["name"],
                            "origin_text": matched_origin,
                        }
                        provenance.setdefault(target, []).append(row_entry)
                        column_texts["__map_foreign"].append(
                            row_entry["foreign_text"])
                        column_texts["__map_origin"].append(
                            row_entry["origin_text"])

            provenance_pairs: set = set()
            _collect(entry.get("types_all") or entry.get("types", []),
                     origin_prefix)
            for label in (entry.get("labels_all")
                          or entry.get("labels", [])):
                # provenance must cover the FULL covered list (covered_all)
                # — the display cap must not leave beyond-cap rows (e.g.
                # DN1a, l-LNv, s-LNv) with empty provenance cells
                _collect(label.get("covered_all")
                         or label.get("types", []),
                         f"{label['column']} · {label['label']}")

            def _fit(texts: List[str], min_w: int, max_w: int) -> int:
                longest = max((len(t) for t in texts), default=0)
                return max(min_w, min(max_w, int(longest * 7.4) + 30))

            widths = {
                "__map_foreign": _fit(column_texts["__map_foreign"], 150, 320),
                "__map_origin": _fit(column_texts["__map_origin"], 170, 360),
            }
            try:
                from comparison.mapping_visualization import (
                    build_mapping_flows,
                )

                flows, pools = _mapping_flows_and_pools(entry)
            except Exception:
                flows, pools = [], {}
            mapped_view.clear()
            mapped_view.update({
                "active": True,
                "foreign_dataset": foreign,
                "types": set(types),
                "provenance": provenance,
                "widths": widths,
                "flows": flows,
                "pools": pools,
            })
            _apply_map_columns()
            return _dispatch_refresh(reset_page=True)

        # A QTable gesture may emit both a value-click and a selection event.
        # Coalesce those duplicate events by their exact anchor while still
        # allowing a different matched entry to be selected immediately.
        focus_request = {"requested_at": 0.0, "anchor": ""}

    def scroll_to_table_rows(focus_keys, anchor_key: str | None = None) -> None:
        """Focus every visible member row after the table finishes scrolling.

        ``scrollIntoView({behavior: 'smooth'})`` is asynchronous. Starting the
        animation on the same tick makes the shade disappear while the row is
        still moving, and repeated NiceGUI/QTable events can queue a second
        flash. A client-side duplicate cooldown and settle poll make one click
        produce one post-scroll flash. Rows outside the current page are
        intentionally ignored by the DOM lookup; the server first moves the
        page to the first member, so every member on that focused page is
        shaded together.
        """
        keys = list(_normalized_focus_keys(focus_keys))
        if not keys:
            return
        encoded_keys = json.dumps(keys)
        anchor = str(anchor_key or keys[0]).strip() or keys[0]
        encoded_anchor = json.dumps(anchor)
        focus_dedup_ms = int(FOCUS_DEDUP_SECONDS * 1000)
        ui.run_javascript(
            f"""
            setTimeout(() => {{
                const rawKeys = {encoded_keys};
                const keys = Array.from(new Set(rawKeys)).sort();
                const anchor = {encoded_anchor};
                const root = document.querySelector('.drocat-neuron-full-panel .drocat-data-viewer-scroll');
                if (!root) return;
                const state = window.__drocatNeuronFocusState || (window.__drocatNeuronFocusState = {{
                    token: 0, signature: '', requestedAt: 0, blockedUntil: 0, timer: null
                }});
                const signature = anchor;
                const now = performance.now();
                // Coalesce duplicate click/selection events for the same
                // matched entry, but allow a deliberate selection of a
                // different entry immediately. The anchor is the exact row
                // the user selected, not the first previously selected row.
                if (now < state.blockedUntil && state.signature === signature) return;
                state.signature = signature;
                state.requestedAt = now;
                state.blockedUntil = now + {focus_dedup_ms};
                state.token += 1;
                const token = state.token;
                if (state.timer) window.clearTimeout(state.timer);
                root.querySelectorAll('.drocat-neuron-focus-flash').forEach(row =>
                    row.classList.remove('drocat-neuron-focus-flash'));

                let attempts = 0;
                const locate = () => Array.from(root.querySelectorAll('tr[data-neuron-key]'))
                    .filter(row => keys.includes(row.dataset.neuronKey));
                const findRows = () => {{
                    if (state.token !== token) return;
                    const rows = locate();
                    if (!rows.length && attempts++ < 15) {{
                        window.setTimeout(findRows, 60);
                        return;
                    }}
                    if (!rows.length) return;
                    const anchorRow = rows.find(row => row.dataset.neuronKey === anchor) || rows[0];
                    anchorRow.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
                    let lastRect = '';
                    let stableFrames = 0;
                    const started = performance.now();
                    const flash = () => {{
                        if (state.token !== token) return;
                        root.querySelectorAll('.drocat-neuron-focus-flash').forEach(row =>
                            row.classList.remove('drocat-neuron-focus-flash'));
                        rows.forEach(row => {{
                            row.classList.remove('drocat-neuron-focus-flash');
                            void row.offsetWidth;
                            row.classList.add('drocat-neuron-focus-flash');
                        }});
                        state.timer = window.setTimeout(() => {{
                            if (state.token !== token) return;
                            rows.forEach(row => row.classList.remove('drocat-neuron-focus-flash'));
                            state.timer = null;
                        }}, 1400);
                    }};
                    const waitForSettle = () => {{
                        if (state.token !== token) return;
                        const rect = anchorRow.getBoundingClientRect();
                        const currentRect = `${{Math.round(rect.top)}}:${{Math.round(rect.left)}}`;
                        stableFrames = currentRect === lastRect ? stableFrames + 1 : 0;
                        lastRect = currentRect;
                        if ((stableFrames >= 3 && performance.now() - started >= 120)
                            || performance.now() - started >= 1200) {{
                            flash();
                            return;
                        }}
                        window.requestAnimationFrame(waitForSettle);
                    }};
                    window.requestAnimationFrame(waitForSettle);
                }};
                findRows();
            }}, 80);
            """
        )

    def render_match_page() -> None:
        """Render one bounded match-detail page without re-running the query."""
        total = len(match_groups_all)
        pages = max(1, (total + MATCH_GROUP_PAGE_SIZE - 1) // MATCH_GROUP_PAGE_SIZE)
        match_state["page"] = max(1, min(match_state["page"], pages))
        start = (match_state["page"] - 1) * MATCH_GROUP_PAGE_SIZE
        end = min(total, start + MATCH_GROUP_PAGE_SIZE)
        current_groups[:] = match_groups_all[start:end]
        match_table.update_rows(current_groups)
        if total:
            match_status.text = (
                f"Showing {start + 1:,}–{end:,} of {total:,} matched names"
            )
        else:
            match_status.text = "No matched names"
        match_status.update()
        match_page_position.text = (
            f"Page {match_state['page']:,} of {pages:,}"
        )
        match_page_position.update()
        match_previous_button.set_enabled(match_state["page"] > 1)
        match_next_button.set_enabled(match_state["page"] < pages)

    def _next_refresh_generation() -> int:
        refresh_generation["value"] += 1
        return refresh_generation["value"]

    def refresh(
        _event=None,
        *,
        reset_page: bool = False,
        focus_key: str | None = None,
        focus_keys=None,
        focus_anchor_key: str | None = None,
        force_search: bool = False,
        _result=None,
        _generation: int | None = None,
    ):
        nonlocal full_table_all_keys
        if _generation is None:
            _next_refresh_generation()
        if reset_page:
            state["page"] = 1
            match_state["page"] = 1
        try:
            current_page_size = int(page_size.value or 50)
        except (TypeError, ValueError):
            current_page_size = 50
        state["page_size"] = current_page_size
        # Selecting every row needs the complete result set; fetching it here
        # would make every keystroke build the whole key map, so it stays lazy.
        full_table_all_keys = None
        if _result is None:
            def run_query(requested_page: int, requested_focus_key: str | None = None):
                return query_neuron_index(
                    index,
                    **query_kwargs_with_mapped(force_search=force_search),
                    page=requested_page,
                    page_size=current_page_size,
                    focus_key=requested_focus_key,
                )

            result = run_query(state["page"], focus_key)
            if focus_key and result.focus_page and result.focus_page != result.page:
                # The first pass computes the sorted position; the second fetches
                # only the page containing that position.
                result = run_query(result.focus_page)
        else:
            result = _result
        state["page"] = result.page
        current_rows[:] = list(result.rows)
        if mapped_view.get("active"):
            # Provenance columns: precomputed at enter time — which foreign
            # types mapped to this row's type, and where the match came
            # from. The cells keep their fixed pinned width, so the full
            # cell text rides on them as a hover title (__map_bridge)
            # together with the bridge derivation.
            provenance = mapped_view.get("provenance", {})
            pools = mapped_view.get("pools", {})
            for row in current_rows:
                entries = provenance.get(str(row.get("type", "")), [])
                if entries:
                    # One origin often reaches the row through several
                    # foreign types; repeating it in the cell buries the
                    # information and overflows the pinned width, so join
                    # first occurrences only.
                    foreign_texts = list(dict.fromkeys(
                        str(e["foreign_text"]) for e in entries))
                    origin_texts = list(dict.fromkeys(
                        str(e["origin_text"]) for e in entries))
                    row["__map_foreign"] = "; ".join(foreign_texts)
                    row["__map_origin"] = "; ".join(origin_texts)
                    # §9.4: the per-bridge pool granularity reads next to
                    # the derivation (e.g. pool 4 to 4 bodyIds)
                    hover_lines = [
                        f"Foreign type(s): {row['__map_foreign']}",
                        f"Matched column: {row['__map_origin']}",
                    ]
                    for e in entries:
                        pool = pools.get(
                            (str(row.get("type", "")),
                             e.get("foreign_type", "")))
                        gran = ""
                        if pool:
                            src = len(pool.get("source_body_ids") or [])
                            tgt = len(pool.get("target_body_ids") or [])
                            if src or tgt:
                                gran = f" (pool {src} to {tgt} bodyIds)"
                        hover_lines.append(
                            f"{e['foreign_text']}: {e['source_text']}{gran}")
                    row["__map_bridge"] = "\n".join(hover_lines)
        match_groups_all[:] = list(result.match_groups)
        last_matched_values.clear()
        for group in result.match_groups:
            if group.get("match_role") == "secondary":
                continue
            column = str(group.get("match_column_key") or "").strip()
            value = str(group.get("match_value") or "").strip()
            if not column or not value:
                continue
            key = (column.casefold(), value.casefold())
            if key in {(c.casefold(), v.casefold())
                       for c, v in last_matched_values}:
                continue
            last_matched_values.append((column, value))
            if len(last_matched_values) >= 8:
                break
        current_group_body_ids.clear()
        current_group_body_ids.update({
            str(key): tuple(values)
            for key, values in result.match_group_body_ids.items()
        })
        match_group_related.clear()
        match_group_related.update({
            str(key): tuple(values)
            for key, values in result.match_group_related.items()
        })
        match_group_primary.clear()
        match_group_primary.update({
            str(key): tuple(values)
            for key, values in result.match_group_primary.items()
        })
        group_members.clear()
        group_members.update({
            str(key): set(values)
            for key, values in result.match_group_members.items()
        })
        stamp_match_group_flags()
        render_match_page()
        table.update_rows(current_rows)
        if focus_keys is None and focus_key:
            focus_keys = (focus_key,)
        # Restore selections after replacing the rows. On a different page,
        # only matching visible rows are checked; the underlying selection
        # sets still retain entries selected on other pages.
        refresh_table_selection()
        if focus_keys:
            scroll_to_table_rows(
                focus_keys,
                anchor_key=focus_anchor_key or focus_key,
            )
        page_position.text = f"Page {result.page:,} of {result.pages:,}"
        page_position.update()
        previous_button.set_enabled(result.page > 1)
        next_button.set_enabled(result.page < result.pages)
        if result.total:
            start = (result.page - 1) * result.page_size + 1
            end = min(result.total, result.page * result.page_size)
            page_info.text = f"Showing {start:,}–{end:,} of {result.total:,} matching rows"
        else:
            page_info.text = "0 matching rows"
        page_info.update()
        no_results.set_visibility(result.total == 0 and not mapped_view.get("active"))
        last_result_total["value"] = result.total
        if mapped_view.get("active"):
            # Mapped-type view: the warning banner replaces the alias panel.
            _hide_alias_panel()
            mapped_warning_label.text = (
                f"Mapped-type view — showing {result.total:,} {dataset} "
                f"neurons whose types map to "
                f"{dataset_abbrev(mapped_view.get('foreign_dataset', ''))} types "
                "(auto mapping + name similarity — please double check). "
                "Selection adds these neurons to the query."
            )
            mapped_warning_label.update()
            mapped_warning_section.set_visibility(True)
            page_info.text = (
                f"Mapped view — showing {result.total:,} neurons of "
                f"{len(mapped_view.get('types', set()))} mapped types"
            )
            page_info.update()
        elif result.total == 0:
            render_alias_matches()
            mapped_warning_section.set_visibility(False)
        elif (
            cross_mapping_mode is not None
            and cross_mapping_mode.get("enabled")
            and current_search_text()
        ):
            # Cross-dataset type mapping mode: the native rows stay, and the
            # cross-dataset scan co-displays as a collapsed expansion above
            # the results.
            render_alias_matches()
            mapped_warning_section.set_visibility(False)
        else:
            _hide_alias_panel()
            mapped_warning_section.set_visibility(False)

    async def refresh_async(
        _event=None,
        *,
        reset_page: bool = False,
        focus_key: str | None = None,
        focus_keys=None,
        focus_anchor_key: str | None = None,
        force_search: bool = False,
    ):
        """Run an interactive index query away from NiceGUI's event loop.

        Polars does the heavy filtering and match-group construction in
        ``query_neuron_index``. Running it inline blocks websocket heartbeats;
        a search on a large cached index can then look like a disconnected
        app. Capture the request before yielding to the worker and discard
        results superseded by a later event.
        """
        nonlocal full_table_all_keys
        generation = _next_refresh_generation()
        if reset_page:
            state["page"] = 1
            match_state["page"] = 1
        try:
            current_page_size = int(page_size.value or 50)
        except (TypeError, ValueError):
            current_page_size = 50
        state["page_size"] = current_page_size
        full_table_all_keys = None
        query_kwargs = query_kwargs_with_mapped(force_search=force_search)

        async def run_query(requested_page: int,
                            requested_focus_key: str | None = None):
            return await run.io_bound(
                query_neuron_index,
                index,
                **query_kwargs,
                page=requested_page,
                page_size=current_page_size,
                focus_key=requested_focus_key,
            )

        try:
            result = await run_query(state["page"], focus_key)
            if result is None or refresh_generation["value"] != generation:
                return
            if focus_key and result.focus_page and result.focus_page != result.page:
                # The first pass computes the sorted position; the second
                # fetches only the page containing that position.
                result = await run_query(result.focus_page)
                if result is None:
                    return
        except asyncio.CancelledError:
            return

        if refresh_generation["value"] != generation:
            return
        refresh(
            reset_page=reset_page,
            focus_key=focus_key,
            focus_keys=focus_keys,
            focus_anchor_key=focus_anchor_key,
            force_search=force_search,
            _result=result,
            _generation=generation,
        )

    def _dispatch_refresh(
        *,
        reset_page: bool = False,
        focus_key: str | None = None,
        focus_keys=None,
        focus_anchor_key: str | None = None,
        force_search: bool = False,
    ):
        """Use the worker-backed refresh for app events, sync fallback for tests."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # NiceGUI's direct test client has no running app loop. Keep this
            # deterministic path synchronous so existing component tests can
            # invoke event listeners without an application server.
            return refresh(
                reset_page=reset_page,
                focus_key=focus_key,
                focus_keys=focus_keys,
                focus_anchor_key=focus_anchor_key,
                force_search=force_search,
            )
        return refresh_async(
            reset_page=reset_page,
            focus_key=focus_key,
            focus_keys=focus_keys,
            focus_anchor_key=focus_anchor_key,
            force_search=force_search,
        )

    def _view_active_mapping(kind: str, variant: str) -> None:
        """Mapped-view banner dispatch: render the stored flows + pools
        and deliver (browser download + persistent banner)."""
        flows = mapped_view.get("flows", [])
        pools = mapped_view.get("pools", {})
        foreign_ds = mapped_view.get("foreign_dataset", "")
        if not flows:
            ui.notify("Nothing to visualize.", type="info")
            return
        label = f"{kind.capitalize()} ({variant} view)"
        try:
            html_text = _render_mapping_artifact(
                kind, variant, flows, pools, foreign_ds)
        except Exception as exc:
            ui.notify(f"{label} failed: {exc}", type="negative")
            return
        if not html_text:
            ui.notify("Nothing to visualize.", type="info")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        foreign = (foreign_ds or "dataset").replace(":", "_")
        _deliver_mapping_visualization(
            html_text, f"mapping_{kind}_{variant}_{foreign}_{stamp}.html",
            label)

    def refresh_result_panels() -> None:
        """Re-evaluate only the results footer (cross-dataset panel/warnings).

        The header's cross-dataset mode toggle uses this: the table query is
        unchanged, so flipping the mode must not re-run it. A mapped-type
        view stays exactly as it is — the mode only takes effect once the
        view is exited.
        """
        if mapped_view.get("active"):
            return
        mapped_warning_section.set_visibility(False)
        if last_result_total["value"] == 0:
            render_alias_matches()
        elif (
            cross_mapping_mode is not None
            and cross_mapping_mode.get("enabled")
            and current_search_text()
        ):
            render_alias_matches()
        else:
            _hide_alias_panel()

    if cross_mapping_mode is not None:
        cross_mapping_mode["refresh"] = refresh_result_panels

    def _clear_mapped_view_for_query() -> None:
        if mapped_view.get("active"):
            mapped_view.clear()
            _restore_map_columns()
            mapped_warning_section.set_visibility(False)

    def handle_search_change(_event=None):
        """Handle automatic search without launching a one-character query.

        A single character is intentionally not an automatic search. Returning
        before dispatching a refresh is important for large indexes: even an
        otherwise unfiltered refresh needlessly competes with the websocket
        while the user is still deciding whether to press Search.
        """
        _clear_mapped_view_for_query()
        raw_text = str(search_input.value or "").strip()
        if raw_text and search_state["forced_text"] == raw_text:
            # An explicit Search click can be followed by the input's
            # debounced value event for the same text.  The click already
            # submitted this query; suppressing the duplicate avoids a
            # second full-index scan while the first R1 request is warming
            # the cross-dataset mapper.
            update_single_char_warning()
            return None
        search_state["forced_text"] = None
        update_single_char_warning()
        if raw_text and not _effective_search_text(raw_text):
            # Do not leave a completed cross-dataset expansion for the prior
            # query visible while the user is composing this guarded input.
            _hide_alias_panel()
            return None
        return _dispatch_refresh(reset_page=True)

    def reset_and_refresh(_event=None, *, clear_forced_search: bool = True):
        # Any query change leaves the mapped-type view: the expansion panel
        # re-evaluates from scratch on the next refresh.
        _clear_mapped_view_for_query()
        if clear_forced_search:
            search_state["forced_text"] = None
        update_single_char_warning()
        return _dispatch_refresh(reset_page=True)

    def force_search(_event=None):
        """Submit the current text, including a deliberate one-character query."""
        _clear_mapped_view_for_query()
        search_state["forced_text"] = str(search_input.value or "").strip() or None
        update_single_char_warning()
        return _dispatch_refresh(reset_page=True, force_search=True)

    def display_refresh(_event=None):
        """Sort / Order / Rows are display controls, not query changes:
        the mapped-type view is a display state and must survive them."""
        return _dispatch_refresh(reset_page=True)

    def request_focus(focus_keys, anchor_key: str | None = None) -> None:
        """Run one page-jump/focus request for one user action.

        NiceGUI can deliver a matched-value click and the QTable selection
        update independently. Suppress duplicate focus requests for the same
        exact anchor during the current scroll/breathe interval, while a
        different selected entry gets its own focus immediately.
        """
        normalized = tuple(_normalized_focus_keys(focus_keys))
        if not normalized:
            refresh_table_selection()
            return
        now = time.monotonic()
        anchor = str(anchor_key or normalized[0]).strip()
        if anchor not in normalized:
            anchor = normalized[0]
        if (
            now - focus_request["requested_at"] < FOCUS_DEDUP_SECONDS
            and focus_request["anchor"] == anchor
        ):
            refresh_table_selection()
            return
        focus_request["requested_at"] = now
        focus_request["anchor"] = anchor
        return _dispatch_refresh(
            focus_key=anchor,
            focus_keys=normalized,
            focus_anchor_key=anchor,
        )

    def handle_match_value_click(event):
        row = getattr(event, "args", None)
        if not isinstance(row, dict):
            return
        value = str(row.get("match_value", "") or "").strip()
        # ``group_members`` contains the private table-row keys (body ID plus
        # source ordinal), not display body IDs. Sort the set so a multi-row
        # click chooses the same anchor every time.
        member_keys = tuple(sorted(group_members.get(value, ())))
        if member_keys:
            return request_focus(member_keys, anchor_key=member_keys[0])

    search_input.on_value_change(handle_search_change)
    search_button.on_click(force_search)
    def handle_filter_column_change(event):
        has_target = target_column.value not in {None, "", "__none__"}
        filter_operator.set_enabled(has_target)
        return reset_and_refresh(event, clear_forced_search=False)

    target_column.on_value_change(handle_filter_column_change)
    filter_operator.on_value_change(
        lambda event: reset_and_refresh(event, clear_forced_search=False)
    )
    sort_column.on_value_change(display_refresh)
    direction.on_value_change(display_refresh)
    page_size.on_value_change(display_refresh)

    def change_page(delta):
        state["page"] = max(1, state["page"] + delta)
        return _dispatch_refresh()

    def change_match_page(delta):
        match_state["page"] = max(1, match_state["page"] + delta)
        render_match_page()
        refresh_table_selection()

    previous_button.on_click(lambda: change_page(-1))
    next_button.on_click(lambda: change_page(1))
    match_previous_button.on_click(lambda: change_match_page(-1))
    match_next_button.on_click(lambda: change_match_page(1))
    match_table.on("match-value-click", handle_match_value_click)
    match_table.on("match-selection-toggle", handle_match_toggle)
    match_table.on("match-expand-toggle", handle_match_expand_toggle)
    match_table.on("match-subtype-toggle", handle_subtype_toggle)
    table.on("full-table-select-all", handle_full_table_select_all)
    refresh()


def create_neuron_index_viewer_link(
    dataset_getter: Callable[[], object],
    *,
    label: str = "See available neurons",
    on_open: Callable[[], object] | None = None,
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    query_edit: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
    defer_apply: bool = False,
):
    """Create a link-like control that opens the cached-index viewer.

    ``dataset_getter`` is evaluated at click time, so changing the dataset in
    a tool tab immediately changes the viewer target.  A multi-dataset getter
    (the Cross-Dataset tab) gets a dataset selector inside the dialog.
    When query callbacks are supplied, the match panel supports multi-select
    and synchronizes selected matched values with the owning query input.
    ``query_remove`` makes the mirrored query preview editable by removing
    one value at a time; ``query_edit`` lets a double-click return that value
    to the owning chip editor.

    With ``defer_apply=True`` (the layer editor), matching values are NOT
    pushed to the caller on every toggle. Instead they accumulate in the
    panel's "Selected" preview (which shows ``selected_query_values()`` and is
    renamed from "Current query"/"mirrors input") and are committed once, when
    the dialog closes via its OK or 'x' button. The preview chips' 'x' then only
    prunes the pending selection rather than the caller's value list.

    ``on_open`` runs immediately before the viewer resolves its dataset. It is
    useful for callers that need to start a fresh selection session without
    changing the viewer's query-selection semantics.
    """
    dialog = ui.dialog()
    # Persistent: clicking the empty backdrop must not dismiss the panel;
    # it closes only via its OK / 'x' buttons.
    dialog.props("persistent")
    # Deferred apply: the OK / 'x' buttons commit the latest pending selection
    # once, then close. ``_render_index`` (re)sets the apply callback per open.
    _apply_holder: dict = {"fn": None}

    def finish() -> None:
        try:
            if _apply_holder["fn"] is not None:
                _apply_holder["fn"]()
        finally:
            dialog.close()

    # Cross-dataset type mapping mode: when enabled, the viewer's search also
    # maps the query into the other cached datasets. The holder survives
    # dataset switches; the rendered viewer registers its footer refresh
    # under "refresh" so toggling re-evaluates the cross-dataset panel
    # without re-running the table query.
    cross_mapping_mode: dict = {"enabled": False, "refresh": None}

    def _paint_cross_toggle() -> None:
        cross_toggle.props(
            "color=primary" if cross_mapping_mode["enabled"] else "color=grey-7")

    def _toggle_cross_mapping() -> None:
        cross_mapping_mode["enabled"] = not cross_mapping_mode["enabled"]
        _paint_cross_toggle()
        refresh_footer = cross_mapping_mode.get("refresh")
        if refresh_footer is not None:
            try:
                refresh_footer()
            except Exception:
                # A stale viewer registration (content re-rendered since)
                # must never break the toggle itself.
                pass

    with dialog:
        with ui.card().classes(
            "w-[min(98vw,1800px)] max-w-none drocat-neuron-viewer-card"
        ):
            with ui.row().classes(
                "w-full items-center justify-between gap-2 drocat-neuron-dialog-header"
            ):
                with ui.row().classes("items-center gap-2 min-w-0 flex-grow"):
                    title = ui.label("Available neurons").classes(
                        "text-h6 drocat-neuron-dialog-title"
                    )
                    header_meta = ui.row().classes(
                        "items-center gap-2 flex-wrap min-w-0 drocat-neuron-header-meta"
                    )
                # Cross-dataset mode toggle sits left of OK: enabled, the
                # search co-displays its cross-dataset matches (collapsed)
                # with the native rows.
                cross_toggle = ui.button(
                    "Cross-dataset type mapping",
                    icon="travel_explore",
                    on_click=_toggle_cross_mapping,
                ).props("outline dense no-caps color=grey-7").classes(
                    "drocat-neuron-cross-mapping-toggle"
                ).tooltip(
                    "Type mapping mode: the search also maps the query into "
                    "the other cached datasets (auto mapping + name "
                    "similarity — informational only, please double check)."
                )
                # OK finishes the selection and exits the panel; the 'x' is a
                # secondary dismiss. Both commit any deferred (pending) selection.
                ui.button(
                    "OK", icon="check",
                    on_click=finish,
                ).props("color=primary dense").classes("drocat-neuron-ok")
                ui.button(icon="close", on_click=finish).props("flat round dense")
            dataset_picker_slot = ui.row().classes("w-full items-center")
            content = ui.column().classes(
                "w-full gap-2 drocat-neuron-viewer-content"
            )

    def open_viewer():
        if on_open is not None:
            try:
                on_open()
            except Exception:
                # Opening the viewer should remain usable even when a caller's
                # session bookkeeping is being torn down during a tab switch.
                pass
        try:
            datasets = _dataset_values(dataset_getter())
        except Exception as exc:
            datasets = []
            error = str(exc)
        else:
            error = ""

        dataset_picker_slot.clear()
        content.clear()
        if error:
            title.text = "Available neurons"
            with content:
                ui.label(f"Could not determine the selected dataset: {error}").classes(
                    "text-body2 drocat-err"
                )
        elif not datasets:
            title.text = "Available neurons"
            with content:
                ui.label("Choose a dataset before opening the neuron index.").classes(
                    "text-body2 drocat-warn"
                )
        else:
            if len(datasets) > 1:
                with dataset_picker_slot:
                    picker = ui.select(
                        options=datasets,
                        value=datasets[0],
                        label="Dataset to view",
                    ).props("outlined").classes("drocat-select").style("min-width: 280px")
                picker.on_value_change(lambda event: _open_dataset(str(event.value)))
            _open_dataset(datasets[0])
        dialog.open()

    def _open_dataset(dataset: str):
        title.text = f"Available neurons · {dataset}"
        title.update()
        _render_index(
            content,
            dataset,
            header_meta=header_meta,
            query_values_getter=query_values_getter,
            query_selection=query_selection,
            query_resolution=query_resolution,
            query_remove=query_remove,
            query_edit=query_edit,
            add_to_query=add_to_query,
            query_label=query_label,
            defer_apply=defer_apply,
            apply_holder=_apply_holder,
            cross_mapping_mode=cross_mapping_mode,
        )

    link = ui.button(label, icon="search", on_click=open_viewer).props(
        "flat dense no-caps"
    ).classes("drocat-inline-link")
    # Expose the dialog for component-level tests and for callers that want
    # to close it after changing tabs.
    link.neuron_index_dialog = dialog
    return link
