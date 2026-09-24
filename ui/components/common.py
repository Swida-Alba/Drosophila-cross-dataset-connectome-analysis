"""Shared UI components for DROCAT.

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, rounded corners and
a focus-panel + contact-sheet workspace layout.
"""

import asyncio
import itertools
import os
import re
import shutil
import time

from nicegui import ui
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
import inspect
import json
import platform
import subprocess
import weakref

from src.utils.dataset_release_registry import get_release_recommendation
from src.flywire_ids import is_banc_dataset, is_fafb_dataset
from ..dataset_service import (
    CAP_MISSING,
    CAP_ON_DEMAND,
    CAP_PARTIAL,
    CAP_READY,
    SERVER_AVAILABLE,
    SERVER_HIDDEN,
    SERVER_NO_TOKEN,
    SERVER_TIMEOUT,
    SERVER_UNKNOWN,
    SERVER_UNREACHABLE,
)

from .. import group_history
from .suggestion_list import (
    ADDED_CLASS,
    CHECK_CLASS,
    MARKED_ROW_TITLE,
    REMOVE_CLASS,
    SUGGESTION_LIMIT,
    chip_is_marked,
    marked_rows,
    without_chip,
)
from ..config import (
    PROJECT_ROOT,
    get_default_output_dir,
    get_tab_output_dir,
    get_user_default,
    has_tab_output_override,
    set_default_output_dir,
    set_tab_output_dir,
)


# Registered output-directory fields let the Settings tab update inherited
# values without overwriting a tab-specific override.
_OUTPUT_DIR_INPUTS = []

# Per-input token linking a suggestion menu to its chip-input anchor in the
# browser. The keyboard-navigation script resolves the open menu for the
# focused input through the matching ``drocat-suggest-anchor-<n>`` /
# ``drocat-suggest-menu-<n>`` class pair.
_SUGGEST_TOKEN = itertools.count(1)

# Client-side Expand/Collapse toggle for the chip editor. The click flips the
# shell class (which drives the field's max-height) and the button label in
# the browser immediately, so a long chip list answers instantly even while
# the connection is busy; the resulting state is emitted so the server-side
# mirror (chip_list_expanded) stays authoritative for clear-all and rebuilds.
# The label edit must write the existing text NODE (firstChild.nodeValue):
# setting textContent would replace the node Vue renders from the QBtn label
# prop, and every later server-side text push would patch the orphaned node
# while the visible label stayed frozen.
_CHIP_LIST_TOGGLE_JS = (
    "(function (event) {"
    "  const t = event.target;"
    "  const btn = t && t.closest ? t.closest('button') : null;"
    "  if (!btn) return;"
    "  const shell = btn.closest('.drocat-chip-input-shell');"
    "  if (!shell) return;"
    "  const expanded = !shell.classList.contains('drocat-chip-list-expanded');"
    "  shell.classList.add(expanded ? 'drocat-chip-list-expanded'"
    "                              : 'drocat-chip-list-collapsed');"
    "  shell.classList.remove(expanded ? 'drocat-chip-list-collapsed'"
    "                                  : 'drocat-chip-list-expanded');"
    "  const label = btn.querySelector('.q-btn__content .block')"
    "                || btn.querySelector('.q-btn__content');"
    "  if (label && label.firstChild && label.firstChild.nodeType === 3) {"
    "    label.firstChild.nodeValue = expanded ? 'Collapse' : 'Expand';"
    "  }"
    "  emit(expanded);"
    "})"
)

# Client-side arrow-key navigation for the suggestion/history dropdown.
# The menu never takes focus (no-focus), so keystrokes land in the QSelect
# editor; this document-level capture handler moves a highlight between the
# menu rows: ArrowDown enters the list from the input box, ArrowUp leaves it
# from the first row, and Enter or Tab picks the highlighted row. A pick
# leaves the editor text alone (the server clears it in the same pass that
# adds the chip) and keeps the highlight ON the list, advancing to the next
# row, so repeated presses keep selecting entries.
_SUGGEST_KEYNAV_SCRIPT = """
<script>
(function () {
  if (window.__drocatSuggestNav) return;
  window.__drocatSuggestNav = true;

  function selectableRows(menu) {
    var rows = [];
    menu.querySelectorAll('.q-item').forEach(function (row) {
      if (row.classList.contains('drocat-suggest-header')) return;
      if (row.offsetParent === null) return;
      rows.push(row);
    });
    return rows;
  }

  function setHighlight(rows, index) {
    rows.forEach(function (row) {
      row.classList.remove('drocat-suggest-active');
    });
    if (index >= 0 && index < rows.length) {
      rows[index].classList.add('drocat-suggest-active');
      rows[index].scrollIntoView({ block: 'nearest' });
    }
  }

  function observeMenu(menu) {
    // After a keyboard pick the server rebuilds the list (a held suggestion
    // query re-renders its own rows, a history pick re-offers Recent).
    // Re-apply the pending highlight to the rebuilt rows so the cursor stays
    // on the list; closing the menu drops any pending highlight.
    if (menu.__drocatNav) return;
    var state = { pending: -1 };
    menu.__drocatNav = state;
    var observer = new MutationObserver(function () {
      if (menu.style.display === 'none') {
        state.pending = -1;
        return;
      }
      if (state.pending < 0) return;
      var rows = selectableRows(menu);
      if (!rows.length) return;
      setHighlight(rows, Math.min(state.pending, rows.length - 1));
      state.pending = -1;
    });
    observer.observe(menu, {
      childList: true,
      attributes: true,
      attributeFilter: ['style'],
    });
  }

  document.addEventListener('keydown', function (event) {
    if (!['ArrowDown', 'ArrowUp', 'Enter', 'Tab'].includes(event.key)) return;
    var active = document.activeElement;
    if (!active || typeof active.closest !== 'function') return;
    var shell = active.closest('.drocat-chip-input-shell');
    if (!shell) return;
    var menuClass = null;
    shell.classList.forEach(function (cls) {
      if (cls.indexOf('drocat-suggest-anchor-') === 0) {
        menuClass = cls.replace('drocat-suggest-anchor-', 'drocat-suggest-menu-');
      }
    });
    if (!menuClass) return;
    var menu = document.querySelector('.' + menuClass);
    if (!menu || menu.style.display === 'none') return;
    observeMenu(menu);
    var rows = selectableRows(menu);
    if (!rows.length) return;
    var current = -1;
    rows.forEach(function (row, i) {
      if (row.classList.contains('drocat-suggest-active')) current = i;
    });
    if (event.key === 'ArrowDown') {
      // Enter the list from the input box, or move down one row.
      if (current < rows.length - 1) {
        event.preventDefault();
        setHighlight(rows, current + 1);
      }
    } else if (event.key === 'ArrowUp') {
      if (current === 0) {
        // First row -> back to the input box (highlight removed).
        event.preventDefault();
        setHighlight(rows, -1);
      } else if (current > 0) {
        event.preventDefault();
        setHighlight(rows, current - 1);
      }
    } else if (event.key === 'Enter' || event.key === 'Tab') {
      // With a highlighted row, both keys pick it instead of their native
      // behavior (commit typed text / move focus to the next control).
      // Without a highlight the native behavior is left untouched.
      if (current === -1) return;
      event.preventDefault();
      event.stopPropagation();
      // The editor text is deliberately NOT wiped here. A local wipe blanks
      // the field while the pick is still on the wire, which reads as the
      // text vanishing before the chip arrives; the server clears it in the
      // same pass that adds the chip. preventDefault/stopPropagation keep
      // Quasar from seeing this keystroke, so the leftover text cannot be
      // committed as an extra chip.
      window.__drocatSuggestEnterPick = true;
      // Keep the cursor ON the list: advance the highlight locally and let
      // the mutation observer re-apply it once the server rebuilds the
      // rows, so further Enter/Tab presses keep selecting entries.
      menu.__drocatNav.pending = current + 1;
      rows[current].click();
      setHighlight(rows, Math.min(current + 1, rows.length - 1));
    }
  }, true);
  // Quasar also commits the editor text on the Enter keyup; suppress it for
  // the pick above so only the highlighted row lands in the chip list.
  document.addEventListener('keyup', function (event) {
    if (event.key !== 'Enter' || !window.__drocatSuggestEnterPick) return;
    event.preventDefault();
    event.stopPropagation();
    window.__drocatSuggestEnterPick = false;
  }, true);
  // A real (cursor) click on a suggestion row is portaled to <body>, so it would
  // blur the host QSelect editor and close the list. preventDefault the mousedown
  // so the editor keeps focus and the menu stays open after a pick; the row's
  // own click handler still fires (preventDefault only stops the focus move).
  document.addEventListener('mousedown', function (ev) {
    var t = ev.target;
    if (!t || typeof t.closest !== 'function') return;
    if (!t.closest('.drocat-suggest-menu')) return;
    ev.preventDefault();
  }, true);
})();
</script>
"""

# Quasar's QMenu positions itself once at open time and only repositions on
# window resize — scrolling an inner container (the fixed-height type-mapping
# dialog, the viewer card) leaves the open suggestion/history menu behind at
# its old viewport position, visually detached from the query box. QMenu's
# own component instance is not reachable from the portaled DOM, so anchor
# the menu by hand: the per-input token pair (drocat-suggest-anchor-<token> /
# drocat-suggest-menu-<token>) identifies the query box each menu belongs to,
# and the scroll handler re-writes the same inline top/left that Quasar's
# position engine set at open time (anchor="bottom start" self="top start",
# flipping above the box when the viewport bottom would be overflowed).
# Capture phase reaches scroll events of inner overflow containers, which
# never bubble to the document.
_SUGGEST_SCROLL_SCRIPT = """
<script>
(function () {
  if (window.__drocatSuggestScrollAnchor) return;
  window.__drocatSuggestScrollAnchor = true;

  function reposition(menu) {
    if (menu.style.display === 'none') return;
    var token = null;
    menu.classList.forEach(function (cls) {
      if (cls.indexOf('drocat-suggest-menu-') === 0) {
        token = cls.slice('drocat-suggest-menu-'.length);
      }
    });
    if (!token) return;
    var anchor = document.querySelector('.drocat-suggest-anchor-' + token);
    if (!anchor) return;
    var rect = anchor.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return;
    var top = rect.bottom;
    var left = rect.left;
    // Skip the write when the anchor has not moved (scroll events can fire
    // many times per frame; the positioning work itself is trivial).
    if (menu.style.top === top + 'px' && menu.style.left === left + 'px') {
      return;
    }
    var height = menu.offsetHeight;
    if (top + height > window.innerHeight - 8 && rect.top - height > 8) {
      top = rect.top - height;
    }
    menu.style.top = top + 'px';
    menu.style.left = left + 'px';
  }

  // Positioning runs synchronously per scroll event: the work is one anchor
  // lookup plus an inline style write, and scroll events must still land
  // when the page's rendering loop is throttled.
  document.addEventListener('scroll', function () {
    document.querySelectorAll('.drocat-suggest-menu').forEach(reposition);
  }, true);
})();
</script>
"""


# =============================================================================
# Page layout helper (focus panel + results / contact sheet)
# =============================================================================

def tool_page(
    title: str,
    subtitle: str = "",
    icon: str = "science",
    tag: Optional[str] = None,
    tag_color: str = "purple",
    doc: Optional[str] = None,
) -> tuple:
    """
    Create the two-column workspace used by every tool tab.

    Returns (form_column, results_column). The form is the "focus panel"
    where the user configures a run; the results column is the "contact
    sheet" showing status, log and output files.
    """
    with ui.column().classes("w-full drocat-page gap-0") as page_col:
        with ui.row().classes("w-full items-center gap-3 drocat-page-head"):
            with ui.element("div").classes("drocat-page-mark"):
                ui.icon(icon).classes("text-white")
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(title).classes("drocat-page-title")
                    if tag:
                        ui.badge(tag, color=tag_color).classes("drocat-tag-badge")
                    if doc:
                        ui.link(
                            "Instructions",
                            f"docs/ui_guides/{doc.replace('.md', '.html')}",
                        ).classes("drocat-doc-link")
                if subtitle:
                    ui.label(subtitle).classes("drocat-page-sub")
        with ui.row().classes("w-full drocat-workspace items-start"):
            form_col = ui.column().classes("drocat-form gap-3")
            results_col = ui.column().classes("drocat-results gap-3")
    # Let full-width blocks (e.g. the inline custom grouper) mount themselves
    # below the two-column workspace instead of squeezing into the form column.
    form_col._drocat_page = page_col
    results_col._drocat_page = page_col
    return form_col, results_col


def section_header(title: str, icon: str = "settings"):
    """Create a section header with icon."""
    with ui.row().classes("items-center gap-2 drocat-section-head"):
        with ui.element("div").classes("drocat-section-icon"):
            ui.icon(icon).classes("text-primary")
        ui.label(title).classes("drocat-section-title")


def param_grid(columns: int = 2):
    """Create a responsive grid container for parameters."""
    return ui.grid(columns=columns).classes("w-full drocat-param-grid")


# =============================================================================
# Dataset Selector
# =============================================================================

# Dataset selectors are created before the Settings tab's availability card.
# Keep weak references so a completed availability refresh can update every
# live selector without keeping closed NiceGUI clients alive.
_DATASET_SELECTORS = weakref.WeakSet()


def _register_dataset_selector(selector, options, service) -> None:
    """Track a status-aware selector for later availability refreshes."""
    selector._drocat_dataset_options = tuple(options)
    selector._drocat_dataset_service = service
    _DATASET_SELECTORS.add(selector)


def refresh_dataset_selector_statuses(service=None) -> int:
    """Refresh local/server suffixes on all live dataset selectors.

    Select option labels are static after construction.  The Settings
    availability card therefore calls this after a refresh so a newly pulled
    dataset is not left displayed as ``☁ server`` until the page is rebuilt.

    Returns the number of selectors updated.  A selector whose client was
    already torn down is ignored; the weak set drops it on its own.
    """
    if service is None:
        from ..dataset_service import get_dataset_service

        service = get_dataset_service()

    updated = 0
    # An explicit refresh must re-derive: a pull finishing moments ago has to
    # flip the tags now, not when the build-time label memo expires.
    _invalidate_dataset_label_cache()
    for selector in tuple(_DATASET_SELECTORS):
        if getattr(selector, "_drocat_dataset_service", service) is not service:
            continue
        options = getattr(selector, "_drocat_dataset_options", ())
        if not options:
            continue
        try:
            labels = {
                dataset: "  ".join(_dataset_label_parts(dataset, service))
                for dataset in options
            }
            if getattr(selector, "options", None) == labels:
                continue
            selector.set_options(labels, value=selector.value)
            updated += 1
        except RuntimeError:
            # NiceGUI can delete a client between the timer callback and this
            # update. Its weak reference will disappear after teardown.
            continue
    return updated


# Short-TTL memo behind _dataset_label_parts. Every dataset selector built
# anywhere in the app resolves per-option status labels (2 fs probes each),
# and one page build constructs ~300 selector-option rounds × 14 datasets —
# thousands of synchronous directory scans on the event loop for labels that
# only need to agree with the 5s status poller.
_DATASET_LABEL_TTL = 2.0
_DATASET_LABEL_CACHE: Dict[str, tuple] = {}


def _invalidate_dataset_label_cache() -> None:
    """Drop memoized selector status labels before an explicit refresh."""
    _DATASET_LABEL_CACHE.clear()


def _dataset_label_parts(ds: str, service) -> List[str]:
    """Build the option label parts with source + local status tags."""
    if is_banc_dataset(ds):
        src_tag = "[BANC]"
    elif is_fafb_dataset(ds):
        src_tag = "[FAFB]"
    else:
        src_tag = "[NP]"
    now = time.monotonic()
    cached = _DATASET_LABEL_CACHE.get(ds)
    # The memo is only valid for the same service instance: tests (and any
    # future multi-service setup) swap services, and stale tags from another
    # instance must never leak.
    if cached is not None and cached[1] > now and cached[2] is service:
        return list(cached[0])
    info = service._cache.get(ds)
    # The filesystem is authoritative for local state.  DatasetInfo can be a
    # persisted or server-backed snapshot, so trusting its old local flags
    # would leave a selector showing "cached" after the cache is deleted.
    if service._check_local_prepared(ds):
        status_tag = "✓ local"
    elif service._check_local_cache(ds):
        status_tag = "◐ cached"
    elif info and info.available:
        status_tag = "☁ server"
    else:
        status_tag = ""
    parts = [ds, src_tag] + ([status_tag] if status_tag else [])
    _DATASET_LABEL_CACHE[ds] = (list(parts), now + _DATASET_LABEL_TTL, service)
    return parts


def _refresh_local_dataset_flags(results, service) -> bool:
    """Re-derive local readiness for the availability rows (no network).

    All local dimensions (metadata/connectivity/visualization, counts) come
    from disk, so the Settings timer can refresh them while the page stays
    open.  Only the persisted server dimension is left untouched.
    """
    changed = False
    for name, info in (results or {}).items():
        fresh = service.check_dataset_availability(name)
        signature = (
            info.metadata_state, info.connectivity_state,
            info.visualization_state, info.visualization_source,
            info.local_prepared, info.local_cache, info.neuron_count,
            info.typed_count, info.available,
        )
        new_signature = (
            fresh.metadata_state, fresh.connectivity_state,
            fresh.visualization_state, fresh.visualization_source,
            fresh.local_prepared, fresh.local_cache, fresh.neuron_count,
            fresh.typed_count, fresh.available,
        )
        if signature != new_signature:
            changed = True
        results[name] = fresh
    return changed


def _resolve_default_dataset(
    options: List[str],
    datasets: Optional[List[str]],
    disable_banc: bool = False,
) -> Optional[str]:
    """Pick the initial single-dataset selection.

    The saved user default applies only to selectors using the standard
    dataset list (``datasets is None``); custom option lists such as
    "(all)" helpers keep their first entry.  A disabled BANC default is
    skipped so the selector never starts on an unselectable option.
    """
    if datasets is None and options:
        preferred = get_user_default("default_dataset")
        if preferred in options:
            if not (disable_banc and "banc" in str(preferred).strip().lower()):
                return preferred
    return options[0] if options else None


def _attach_release_recommendation_notice(selector, options, multiple=False):
    """Attach an inline, action-oriented release recommendation.

    The selector's public return value and option values stay unchanged.  The
    notice is a sibling element so all existing callers keep their current
    NiceGUI bindings while old release selections receive an explicit, user-
    initiated upgrade path.
    """
    notice = ui.element("div").classes("w-full drocat-release-notice")
    notice.props(
        'data-testid="dataset-release-notice" role="status" '
        'aria-live="polite"'
    )
    state = {"dismissed": None}
    option_values = tuple(str(value) for value in options)

    def _selected_values():
        value = selector.value
        if multiple:
            if value is None:
                return []
            if isinstance(value, (tuple, list, set)):
                return [str(item) for item in value]
            return [str(value)]
        return [str(value)] if value not in (None, "") else []

    def _recommendations(selected):
        selected_set = set(selected)
        records = []
        for dataset in selected:
            record = get_release_recommendation(dataset, option_values)
            if not record:
                continue
            # Selecting both releases is intentional and should not nag the
            # user; the newer release is already part of the comparison.
            if record["target"] in selected_set:
                continue
            records.append(record)
        return records

    def _render(_event=None):
        selected = _selected_values()
        current_key = tuple(selected)
        records = _recommendations(selected)
        notice.clear()
        if not records or state["dismissed"] == current_key:
            notice.set_visibility(False)
            return

        notice.set_visibility(True)
        with notice:
            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                ui.icon("info").classes("text-primary")
                if len(records) == 1:
                    record = records[0]
                    ui.label(record["message"]).classes("drocat-release-notice-text")
                    if record.get("actionable"):
                        def _use_newer(record=record):
                            replacement = record["target"]
                            if multiple:
                                values = _selected_values()
                                values = [
                                    replacement if value == record["source"] else value
                                    for value in values
                                ]
                                selector.set_value(values)
                            else:
                                selector.set_value(replacement)
                            state["dismissed"] = None
                            _render()

                        ui.button(
                            f"Use {record['target']}",
                            on_click=_use_newer,
                        ).props("flat dense").classes("drocat-release-action")
                    else:
                        ui.label("Access or prepare the newer release first.").classes(
                            "drocat-release-notice-muted"
                        )
                else:
                    labels = ", ".join(
                        f"{record['source']} → {record['target']}"
                        for record in records
                    )
                    ui.label(
                        f"Newer supported releases are available: {labels}."
                    ).classes("drocat-release-notice-text")
                ui.button(
                    "Keep current",
                    on_click=lambda: (
                        state.__setitem__("dismissed", current_key), _render()
                    ),
                ).props("flat dense").classes("drocat-release-dismiss")

    selector._drocat_release_notice = notice
    selector._drocat_release_notice_render = _render
    selector.on_value_change(_render)
    _render()
    return notice


def dataset_selector(
    label: str = "Dataset",
    default: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    on_change: Optional[Callable] = None,
    hint: str = "NeuPrint: fetched from server with token. FAFB: uses converted local files and optional CAVE access. BANC: uses its public release bucket; no CAVE token is needed.",
    allow_custom: bool = False,
    show_local_status: bool = True,
    disable_banc: bool = False,
    show_release_notice: bool = True,
) -> ui.select:
    """Create a dataset dropdown selector with local status labels.

    ``disable_banc`` keeps BANC visible for discoverability but marks every
    BANC option as disabled in the Quasar popup.  The backend still validates
    the dataset because a value can be supplied programmatically or by a
    previously saved UI state.
    """
    from ..dataset_service import get_dataset_service

    service = get_dataset_service()
    options = datasets if datasets is not None else service.get_all_datasets()

    if show_local_status:
        labeled_options = {
            ds: "  ".join(_dataset_label_parts(ds, service)) for ds in options
        }
        sel_options = labeled_options
    else:
        sel_options = options

    default_val = default if default is not None else _resolve_default_dataset(
        options, datasets, disable_banc
    )
    sel = ui.select(
        options=sel_options,
        value=default_val,
        label=label,
    ).props("outlined").classes("w-full drocat-select").tooltip(hint)
    if show_local_status:
        _register_dataset_selector(sel, options, service)
    if show_release_notice:
        _attach_release_recommendation_notice(sel, options, multiple=False)
    if disable_banc:
        # NiceGUI converts its Python option mapping to QSelect options with
        # ``label`` and an internal index.  Use the rendered label as the
        # stable predicate so the BANC rows are disabled whether local-status
        # suffixes are shown or not.  Quasar renders disabled options grey and
        # prevents selecting them from the popup.
        disabled_labels = [
            str(dataset_name)
            for dataset_name in options
            if "banc" in str(dataset_name).strip().lower()
        ]
        if disabled_labels:
            sel.props(
                ":option-disable=\"option => "
                "String(option.value || option.label).toLowerCase().includes('banc')\""
            )
        # Keep a small Python-side marker for tests and server-side callers;
        # it does not replace the backend guard.
        sel._drocat_disabled_datasets = disabled_labels
    if allow_custom:
        sel.props('use-input new-value-mode="add-unique"')
    if on_change:
        sel.on_value_change(on_change)
    return sel


def dataset_multi_selector(
    label: str = "Datasets",
    default: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    hint: str = (
        "Select one or more datasets. One dataset with multiple thresholds is "
        "also supported; "
        "multiple datasets enable cross-dataset comparison. Shows [NP]=NeuPrint, "
        "[FAFB]=FAFB local release, [BANC]=public BANC bucket, ✓ local / ☁ server status."
    ),
    show_local_status: bool = True,
    show_release_notice: bool = True,
) -> ui.select:
    """Create a multi-select dataset dropdown with local status labels."""
    from ..dataset_service import get_dataset_service

    service = get_dataset_service()
    options = datasets if datasets is not None else service.get_all_datasets()

    if show_local_status:
        sel_options = {
            ds: "  ".join(_dataset_label_parts(ds, service)) for ds in options
        }
    else:
        sel_options = options

    # ``None`` means "use the saved user default dataset, else the first two
    # datasets" for callers that want a convenient default.  An explicit empty
    # list means "start unselected".
    if default is not None:
        default_val = default
    else:
        preferred = get_user_default("default_dataset")
        default_val = (
            [preferred]
            if datasets is None and preferred in options
            else (options[:2] if len(options) >= 2 else options)
        )
    sel = ui.select(
        options=sel_options,
        value=default_val,
        label=label,
        multiple=True,
    ).props("outlined").classes("w-full drocat-select").props(
        "use-chips use-input"
    ).tooltip(hint)
    _clear_native_select_editor_after_selection(sel)
    if show_local_status:
        _register_dataset_selector(sel, options, service)
    if show_release_notice:
        _attach_release_recommendation_notice(sel, options, multiple=True)
    return sel


# =============================================================================
# Advanced Neuron Input (with filter mode)
# =============================================================================

def advanced_neuron_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p",
    hint: str = "Enter neuron types, bodyIds, or patterns. Use filter mode for pattern matching.",
) -> ui.element:
    """Create an advanced neuron input with filter mode selector."""
    with ui.column().classes("w-full gap-0") as container:
        with ui.row().classes("w-full items-end gap-2"):
            textarea = ui.textarea(
                label=label,
                placeholder=placeholder,
            ).props("autogrow").classes("flex-grow drocat-input").tooltip(hint)
            filter_mode = ui.select(
                options={
                    "exact": "Exact",
                    "startswith": "Starts with",
                    "contains": "Contains",
                    "endswith": "Ends with",
                    "regex": "Regex",
                },
                value="exact",
                label="Filter",
            ).classes("w-32 drocat-select").props("dense outlined").tooltip(
                "Exact: match type/bodyId exactly\n"
                "Starts with: e.g. 'DN' → DN1p, DN2\n"
                "Contains: e.g. 'PN' → adPN, lPN\n"
                "Ends with: e.g. '_R' → right hemisphere\n"
                "Regex: e.g. 'KC.*' → all KC types"
            )

    container.get_value = lambda: (filter_mode.value, textarea.value)
    container.textarea = textarea
    container.filter_mode = filter_mode
    return container


def output_detail_control(
    hint: str,
) -> Tuple[Any, Callable[[], Dict[str, bool]]]:
    """Output detail segmented control for the NeuronBridge tool tabs.

    Full (default) keeps every exported file. Compact drops the bodyId-
    level source-data tables after their summaries are written and, where
    applicable, the downloaded images after the integrated PDF/PPTX exists
    (every removal audited in the run's ``cleanup_audit.json``).

    Returns ``(control, flags_getter)``; ``flags_getter()`` maps the
    selection onto the ``keep_per_match_csv`` / ``cleanup_source_images``
    method parameters.
    """
    with ui.column().classes("w-full gap-1"):
        detail = ui.toggle(
            {"full": "Full", "compact": "Compact"}, value="full",
        ).props("outline no-caps").classes("drocat-select")
        ui.label(hint).classes("text-xs opacity-60 w-full").style(
            "line-height:1.35")

    def _flags() -> Dict[str, bool]:
        compact = detail.value == "compact"
        return {
            "keep_per_match_csv": not compact,
            "cleanup_source_images": compact,
        }

    return detail, _flags


def neuron_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p",
    hint: str = "Comma or newline separated. Supports types, bodyIds, regex patterns like KC.*",
) -> ui.textarea:
    """Create a simple neuron query input textarea."""
    return ui.textarea(
        label=label,
        placeholder=placeholder,
    ).props("autogrow").classes("w-full drocat-input").tooltip(hint)


async def read_upload_event(event) -> tuple[str, bytes]:
    """Read a NiceGUI 3 upload event, with compatibility for older events."""
    upload = getattr(event, "file", None)
    if upload is not None:
        return upload.name, await upload.read()

    name = getattr(event, "name", "upload")
    content = getattr(event, "content", None)
    if content is None:
        raise ValueError("Upload event contains no file")
    data = content.read()
    if inspect.isawaitable(data):
        data = await data
    return name, data


def parse_neuron_upload(filename: str, content: bytes) -> List:
    """Parse the first column of a CSV/TSV/Excel neuron-list upload."""
    import csv
    import io
    from numbers import Integral, Real

    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".tsv", ".xlsx", ".xls"}:
        raise ValueError("Use a CSV, TSV, XLSX, or XLS neuron-list file")
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        frame = pd.read_excel(io.BytesIO(content), header=None)
        values = frame.iloc[:, 0].dropna().tolist()
    else:
        text = content.decode("utf-8-sig", errors="replace")
        delimiter = "\t" if suffix == ".tsv" else ","
        values = [
            row[0]
            for row in csv.reader(io.StringIO(text), delimiter=delimiter)
            if row
        ]

    loaded = [
        (value, str(value).strip().strip("\"'"))
        for value in values
        if str(value).strip().strip("\"'")
    ]
    if loaded and loaded[0][1].lower() in {
        "type", "neuron", "name", "bodyid", "id", "body_id",
    }:
        loaded = loaded[1:]
    parsed = []
    for raw_value, value in loaded:
        if isinstance(raw_value, Integral):
            parsed.append(int(raw_value))
            continue
        if isinstance(raw_value, Real) and float(raw_value).is_integer():
            parsed.append(int(raw_value))
            continue
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def _normalize_neuron_value(item):
    """Normalize body IDs to integers while preserving neuron type strings."""
    value = str(item)
    return int(value) if value.isdigit() else value


def neuron_list_input(
    label: str = "Neurons",
    placeholder: str = "e.g., aMe12, aMe10, DN1p (or upload CSV/TSV/Excel)",
    hint: str = (
        "Type a name and press Enter or leave the field to add it as a chip. "
        "Commas and spaces are kept as part of the name. "
        "Upload a CSV/TSV/Excel file (first column) for a larger list."
    ),
    unit_label: str = "neuron",
    show_filter: bool = True,
    show_upload: bool = True,
    max_items: Optional[int] = None,
    initial: Optional[List] = None,
    suggestions: Optional[Callable[[str], List[Tuple[str, str]]]] = None,
    available_neurons: Optional[Callable[[], object]] = None,
    history_datasets: Optional[Callable[[], object]] = None,
    history_kind: Optional[str] = None,
    history_hint_datasets: Optional[Callable[[], object]] = None,
    show_history_datasets: bool = False,
    suggestion_min_chars: int = 1,
    suggestion_limit: int = SUGGESTION_LIMIT,
    show_count: bool = True,
    show_clear: bool = True,
    filter_dense: bool = True,
    input_actions: Optional[Callable[[], None]] = None,
) -> ui.element:
    """
    Create a chip-based list input for neurons or driver lines.

    - Type a name (type, bodyId or pattern) and press Enter or leave the field
      to add it as a chip. The whole typed text becomes ONE chip — commas and
      spaces inside names are preserved, never treated as separators.
    - Upload a CSV/TSV/Excel file (first column) for a larger list.
    - ``initial`` seeds the chip list with pre-existing values.
    - ``max_items`` caps the list (used for single-input tabs); additional
      values are rejected once the cap is reached.
    - A live count badge and a Clear button keep the list manageable. Both can
      be hidden independently with ``show_count=False`` and
      ``show_clear=False`` for compact one-row editors. The badge names items
      with ``unit_label`` ("neuron" by default; pass e.g. "threshold" for
      non-neuron chip inputs so the counter reads "3 thresholds" instead of
      "3 neurons").
    - ``filter_dense`` (default True) keeps the Match-by select compact;
      pass ``False`` to render it at the chip input's height (same-row
      layouts that place other controls beside it).
    - ``input_actions``: optional callback invoked INSIDE the input row,
      right of the Match-by select — create action buttons there (they
      share the row with the chip input, which narrows accordingly).
    - ``suggestions``: optional provider ``typed_text -> [(value, hint)]``
      powering the auto-suggest dropdown (dataset type/instance/bodyId names
      with the searched column as a gray hint). The provider is prefiltered
      from the first character and suggestions are shown from the first
      character by default (``suggestion_min_chars=1``). A blank focused field
      opens the persistent query history (last 10 + most frequent), filtered
      to the dataset(s) selected in the current tab (via ``history_datasets``
      or ``available_neurons``); history is not mixed into a nonblank dataset
      search. History rows carry the same gray category hint as suggestion
      rows (the searched column, or the cached instance for bodyIds). Set
      ``history_kind="line"`` for a driver-line input; line history is stored
      separately from neuron history and works even without a suggestion
      provider. Set ``history_kind="type_mapping"`` for the Type Mapping
      panel's query box, which keeps its own standalone history store so
      panel searches never mix with the analysis tabs' query history. With
      ``history_hint_datasets`` (a dataset getter), history rows render the
      same gray hint as auto-suggestion rows — the matched column and the
      dataset(s) it came from (``type · male-cns:v1.0``) — computed with
      exact membership over those datasets' local pools; the history list
      itself stays unscoped. With
      ``show_history_datasets=True`` (the cross-dataset tab), history rows
      additionally show a gray tag per selected dataset whose local pool
      actually contains the value — exact membership, restricted to the
      datasets currently selected in the tab's dataset input, so a
      male-cns-only type never carries badges for the other datasets that
      merely shared the same run. Values resolving in none of the selected
      datasets (patterns, not-pulled datasets) show no tags. Arrow keys
      navigate the open dropdown: ArrowDown enters the list
      from the editor, ArrowUp returns to the editor from the first row, and
      Enter or Tab picks the highlighted row — the highlight then advances
      to the next entry and stays on the list, so repeated presses keep
      selecting. At most ``suggestion_limit`` entries
      are shown. With a provider, the native QSelect popup is replaced by
      the custom suggestion menu.

    Picking a row adds the chip first and clears the editor text after it, and
    the query's rows stay listed in the same order — a value that is already in
    the field is tinted and ticked rather than dropped, so the list never
    shifts under the pointer while several entries are handled in a row (type
    ``PPL1``, click ``PPL101``, then ``PPL102``; clicking the ticked ``PPL101``
    again takes that chip back out, the same as its own ``x``). The
    Recent/Frequent history carries the same marks, and a marked history row
    deselects too — its prune ``x`` remains a separate action, dropping the
    entry from the history rather than from the query, and is shown only while
    that row is hovered or focused so the two removals never sit side by side.
    Any click that closes
    the list ends the held query: clicking in the query box dismisses it, and so
    does an outside click, ESC or focusing another input, after which the next
    opening shows the history list. New typed text starts its own round, and
    picking from the history list keeps showing that list.

    - ``available_neurons``: optional zero-argument dataset getter. When
      supplied, a ``See available neurons`` link opens the rendered,
      searchable cached neuron-index viewer for the current dataset.
      The viewer's mirrored query chips can remove values through the same
      input when it supplies a ``query_remove`` callback. Both the editor and
      mirrored chips can be double-clicked to re-edit a value.

    Long chip lists collapse to a scrollable three-row editor; the Expand
    action shows the full list without changing the query (Collapse restores
    the compact view). The toggle is applied client-side for an instant
    response; the expanded/collapsed state clips on the chip container, so
    chips never render outside the field border while scrolling.

    The Recent/Frequent history list is click-to-toggle on the focused,
    empty editor: the click that focuses the field opens it, clicking the
    editor again hides it, and clicking once more shows it again. While
    typed text shows the suggestion list, clicks never hide it; a HELD
    suggestion list (still showing after a pick, editor already cleared)
    closes on its own dismissal click, and the click after that opens the
    history list. Clicks on chips or their remove buttons never toggle
    anything.

    Returns container with .get_value() -> (filter_mode, neuron_list).
    """
    if history_kind not in (None, "neuron", "line", "type_mapping"):
        raise ValueError(
            "history_kind must be 'neuron', 'line', 'type_mapping', or None")
    history_enabled = suggestions is not None or history_kind is not None

    def _unit(count: int) -> str:
        """Pluralized unit label for count displays."""
        return f"{unit_label}{'s' if count != 1 else ''}"

    uploaded_neurons: List = []
    value_change_callbacks: List[Callable[[], None]] = []
    chip_list_expanded = {"value": False}

    async def handle_upload(e):
        """Parse the first column of an uploaded CSV/TSV/Excel neuron list."""
        try:
            filename, raw = await read_upload_event(e)
            loaded = parse_neuron_upload(filename, raw)
            # Mutate in place so container.uploaded_neurons remains current.
            uploaded_neurons[:] = loaded
            if upload_label is not None:
                upload_label.text = (
                    f"✓ {len(uploaded_neurons)} {_unit(len(uploaded_neurons))} loaded from {filename}"
                )
                upload_label.classes(replace="text-caption drocat-ok")
        except Exception as exc:
            uploaded_neurons.clear()
            if upload_label is not None:
                upload_label.text = f"Error: {exc}"
                upload_label.classes(replace="text-caption drocat-err")
        if upload_label is not None:
            upload_label.set_visibility(True)
        update_status()
        upload_menu.close()

    with ui.column().classes("w-full gap-1") as container:
        with ui.row().classes("w-full items-end gap-2 drocat-neuron-input-row"):
            # Chip-based list input. Quasar normally commits a new value only
            # on Enter; the input event below preserves the editor text so it
            # can also be committed when the user moves focus elsewhere.
            # Seed values are kept in the options list: NiceGUI's QSelect only
            # renders chips whose values exist in its options (model-value is
            # filtered against them), so committed values must be added there.
            initial_values = [_normalize_neuron_value(item) for item in (initial or [])]
            # Keep a private anchor around each QSelect. The custom QMenu is
            # rendered into this wrapper below, rather than into the shared
            # input row; otherwise two neuron inputs on the same page would
            # anchor their menus to the same row and one popup could obscure
            # the other input during a focus change.
            with ui.element("div").classes(
                "relative flex-grow drocat-chip-input-shell drocat-chip-list-collapsed"
            ) as chip_input_anchor:
                chip_input = ui.select(
                    options=list(initial_values),
                    value=list(initial_values),
                    label=label,
                    multiple=True,
                ).props(
                    'use-chips use-input new-value-mode="add-unique" '
                    'input-debounce="0" outlined'
                ).classes("w-full drocat-select drocat-chip-input").tooltip(hint)

            # Keep the layout control inside the field itself. A history or
            # suggestion menu is anchored below the field and therefore cannot
            # cover this button or turn an Expand click into a history pick.
            # The click flips the layout client-side (see _CHIP_LIST_TOGGLE_JS)
            # so a long chip list answers instantly even while the connection
            # is busy; the emitted state only syncs the server mirror.
            with chip_input.add_slot("append"):
                expand_button = ui.button(
                    "Expand",
                ).props("flat dense no-focus").classes("drocat-chip-expand-btn")
                expand_button.on(
                    "click",
                    lambda event: _sync_chip_list_state(event),
                    js_handler=_CHIP_LIST_TOGGLE_JS,
                )
                expand_button.set_visibility(bool(initial_values))

            filter_mode = None
            if show_filter:
                filter_mode = ui.select(
                    options={
                        "exact": "Exact",
                        "startswith": "Starts with",
                        "contains": "Contains",
                        "endswith": "Ends with",
                        "regex": "Regex",
                    },
                    value="exact",
                    label="Match by",
                ).classes("w-32 drocat-select drocat-neuron-match-filter").props(
                    "dense outlined" if filter_dense else "outlined"
                ).tooltip(
                    "How the query matches the Search Columns: exact, prefix "
                    "(starts with), substring (contains), suffix (ends with) "
                    "or regex pattern"
                )

            if input_actions is not None:
                # Extra row widgets (e.g. an action button) placed to the
                # RIGHT of the Match-by select — callers keep their own
                # handlers; ``filter_dense=False`` lets them match the
                # chip input's height.
                input_actions()

            if show_upload:
                # Compact upload: hidden inside a dropdown attached to the input row
                with ui.button(icon="upload_file").props("flat dense round").classes(
                    "drocat-upload-trigger"
                ).tooltip("Upload CSV/TSV/Excel (first column = neurons)"):
                    with ui.menu() as upload_menu:
                        ui.label("Load neuron list from file").classes(
                            "text-caption drocat-muted px-3 pt-2"
                        )
                        ui.label("CSV / TSV / XLSX / XLS · first column is read").classes(
                            "text-caption drocat-muted px-3 pb-1"
                        )
                        ui.upload(
                            label="Choose neuron file",
                            on_upload=handle_upload,
                            auto_upload=True,
                        ).props('accept=".csv,.xlsx,.xls,.tsv" flat dense').classes("w-72")
                        ui.link(
                            "File format instructions",
                            "docs/ui_guides/input_formats.html",
                        ).classes("drocat-doc-link px-3 pb-2")

        def sync_viewer_selection(values):
            """Synchronize viewer-owned chips without replacing user values.

            The viewer sends its complete current selection on every checkbox
            change. Values that were already in the input (or uploaded) are
            never owned by the viewer, so deselecting a viewer group cannot
            remove an unrelated pre-existing query chip.
            """
            current = list(chip_input.value or [])
            previous_viewer_values = set(viewer_owned_values)
            base = [
                value for value in current
                if str(value) not in previous_viewer_values
            ]
            base_keys = {str(value) for value in base}
            uploaded_keys = {str(value) for value in uploaded_neurons}
            merged = list(base)
            owned = set()
            selected_display = set()
            for item in values or []:
                value = _normalize_neuron_value(item)
                key = str(value)
                if key in base_keys or key in uploaded_keys:
                    # The viewer may select a value that the user already
                    # entered. Keep the existing chip, but still mark it as
                    # viewer-selected so execution uses the verified body IDs
                    # supplied by the index rather than resolving the name
                    # through a possibly colliding metadata column.
                    selected_display.add(key)
                    continue
                if max_items is not None and len(merged) >= max_items:
                    break
                if key in {str(existing) for existing in merged}:
                    continue
                merged.append(value)
                owned.add(key)
                selected_display.add(key)
            viewer_owned_values.clear()
            viewer_owned_values.update(owned)
            viewer_selected_values.clear()
            viewer_selected_values.update(selected_display)
            sync_options(merged)
            _suppress_history_popup["value"] = True
            chip_input.run_method("updateInputValue", "")
            chip_input.set_value(merged)
            pending_input["value"] = ""
            update_status()
            return len(owned)

        def sync_viewer_body_selection(values):
            """Keep exact body-ID resolution for viewer-owned display chips."""
            viewer_owned_body_ids[:] = [
                _normalize_neuron_value(item)
                for item in (values or [])
                if str(item or "").strip()
            ]

        def remove_viewer_query_value(value):
            """Remove a value from the input when its viewer chip is closed."""
            target = str(value or "").strip()
            if not target:
                return
            uploaded_neurons[:] = [
                item for item in uploaded_neurons
                if str(item or "").strip() != target
            ]
            current = list(chip_input.value or [])
            remaining = [
                item for item in current
                if str(item or "").strip() != target
            ]
            viewer_owned_values.discard(target)
            viewer_selected_values.discard(target)
            viewer_owned_body_ids[:] = [
                item for item in viewer_owned_body_ids
                if str(item or "").strip() != target
            ]
            if remaining != current:
                _suppress_history_popup["value"] = True
                chip_input.run_method("updateInputValue", "")
                sync_options(remaining)
                chip_input.set_value(remaining)
            update_status()

        # Status row: optional live count + upload status + clear. Compact
        # callers can omit the entire row; the advanced layer editor uses its
        # table selection/deletion controls instead.
        count_badge = None
        upload_label = None
        clear_button = None
        viewer_link = None
        if show_count or show_upload or show_clear or available_neurons is not None:
            with ui.row().classes("w-full items-center gap-2"):
                if show_count:
                    count_badge = ui.badge(
                        f"0 {_unit(0)}", color="grey-6"
                    ).props("outline")
                if show_upload:
                    upload_label = ui.label("").classes("text-caption drocat-muted")
                    upload_label.set_visibility(False)
                if show_clear:
                    # Keep this action text-only across every neuron input. The
                    # label is already short and the icon made the status row
                    # look different from the other Clear actions in the UI.
                    clear_button = ui.button("Clear").props(
                        "flat dense"
                    ).classes("drocat-clear-btn")
                if available_neurons is not None:
                    # Import lazily to keep the common input component independent
                    # from the optional viewer's Polars-backed data layer.
                    from .neuron_index_viewer import create_neuron_index_viewer_link

                    viewer_link = create_neuron_index_viewer_link(
                        available_neurons,
                        query_values_getter=lambda: [
                            *uploaded_neurons,
                            *(chip_input.value or []),
                        ],
                        query_selection=sync_viewer_selection,
                        query_resolution=sync_viewer_body_selection,
                        query_remove=remove_viewer_query_value,
                        query_edit=lambda value: start_edit_value(value),
                        query_label=label,
                    )

    pending_input = {"value": ""}
    viewer_owned_values = set()
    viewer_selected_values = set()
    viewer_owned_body_ids = []
    # Programmatic list changes (uploads, viewer selections, and clear) must
    # not pop the Recent list open.
    _suppress_history_popup = {"value": False}
    # The suggestion query whose rows are currently held on screen. A pick
    # keeps it so the next entry can be added without retyping; any click that
    # closes the list, or new typed text, drops it back to the history list.
    # Defined here (not inside the menu block) because the value-commit
    # handlers below run even when no suggestion menu exists.
    _sticky = {"query": None}

    def normalize_neuron(item):
        return _normalize_neuron_value(item)

    def sync_options(values):
        """Keep committed values in the option list so they render as chips.

        NiceGUI's QSelect derives the rendered model-value from its options
        (``Select._value_to_model_value`` skips values not in the options
        list), so any value committed programmatically — blur, upload, seed —
        must be added to ``options`` or the chip stays invisible.
        """
        for value in values:
            if value not in chip_input.options:
                chip_input.options.append(value)

    def update_status():
        combined = [normalize_neuron(item) for item in uploaded_neurons]
        combined.extend(normalize_neuron(item) for item in (chip_input.value or []))
        count = len(dict.fromkeys(combined))
        if count_badge is not None:
            count_badge.text = f"{count} {_unit(count)}"
            count_badge.props(f"color={'primary' if count else 'grey-6'}")
        expand_button.set_visibility(bool(chip_input.value))
        for callback in list(value_change_callbacks):
            try:
                callback()
            except Exception:
                # A page-level advisory must not break the shared input when
                # a consumer is torn down during a tab switch.
                pass

    def _apply_chip_list_state(expanded: bool) -> None:
        """Push the expanded/collapsed layout state onto the server mirror.

        Idempotent: the browser usually applied the same state already (see
        _CHIP_LIST_TOGGLE_JS), so the class/text updates are no-ops.
        """
        chip_list_expanded["value"] = expanded
        if expanded:
            chip_input_anchor.classes(
                add="drocat-chip-list-expanded",
                remove="drocat-chip-list-collapsed",
            )
            expand_button.text = "Collapse"
        else:
            chip_input_anchor.classes(
                add="drocat-chip-list-collapsed",
                remove="drocat-chip-list-expanded",
            )
            expand_button.text = "Expand"
        expand_button.update()

    def toggle_chip_list() -> None:
        """Toggle the compact three-row view of the chip editor.

        Server-driven path (programmatic use). The user-facing button flips
        the layout in the browser first and adopts the state here afterwards.
        """
        # Expanding is a layout action, not a new query interaction. If the
        # suggestion/history menu is open because the editor still owns focus,
        # close it before the button takes focus so the Recent list cannot
        # flash back over the expanded editor.
        if history_enabled and suggest_menu is not None:
            _suppress_history_popup["value"] = True
            _sticky["query"] = None
            _close_suggest()
        _apply_chip_list_state(not chip_list_expanded["value"])

    def _sync_chip_list_state(event) -> None:
        """Adopt the state the browser's click handler already applied.

        A click event without an emitted boolean (plain click wiring, tests)
        falls back to toggling the current server-side state.
        """
        args = getattr(event, "args", None)
        if isinstance(args, bool):
            expanded = args
        else:
            expanded = not chip_list_expanded["value"]
        _apply_chip_list_state(expanded)

    def start_edit_value(value) -> None:
        """Remove one chip and put its exact text back into the editor."""
        raw_value = getattr(value, "args", value)
        target = str(raw_value or "").strip()
        if not target:
            return
        current = list(chip_input.value or [])
        remaining = [item for item in current if str(item).strip() != target]
        if len(remaining) == len(current):
            # The mirrored viewer can race a query refresh; still let the
            # user recover the text they double-clicked.
            remaining = current
        viewer_owned_values.discard(target)
        viewer_selected_values.discard(target)
        viewer_owned_body_ids[:] = [
            item for item in viewer_owned_body_ids
            if str(item).strip() != target
        ]
        sync_options(remaining)
        _suppress_history_popup["value"] = True
        _sticky["query"] = None
        chip_input.set_value(remaining)
        pending_input["value"] = target
        chip_input.run_method("focus")
        chip_input.run_method("updateInputValue", target)
        update_status()

    def clear_all():
        uploaded_neurons.clear()
        viewer_owned_values.clear()
        viewer_selected_values.clear()
        viewer_owned_body_ids.clear()
        _suppress_history_popup["value"] = True
        _sticky["query"] = None
        chip_input.set_value([])
        if chip_list_expanded["value"]:
            _apply_chip_list_state(False)
        if upload_label is not None:
            upload_label.text = ""
            upload_label.set_visibility(False)
        update_status()

    def remember_user_input(event):
        """Track the native editor input — user typing AND deletions.

        The native ``input`` event fires only for user edits (Quasar resets the
        editor programmatically, which never emits it), so an empty value here
        means the user cleared the field and the pending text must be dropped.
        """
        pending_input["value"] = str(getattr(event, "args", "") or "")

    def remember_quasar_input(event):
        """Track Quasar's ``input-value``, ignoring its blur-reset emission.

        Quasar emits ``input-value ''`` when it clears the editor after focus
        loss; that must not wipe the text typed just before the blur handler
        commits it. Non-empty values always replace the pending text.
        """
        value = str(getattr(event, "args", "") or "")
        if value.strip():
            pending_input["value"] = value

    def commit_pending_text(event=None):
        """Commit editor text when focus leaves the chip selector.

        The whole editor text becomes a single chip: commas and spaces are
        legal inside names (e.g. driver line names, 'A -> B -> C' layers) and
        are never treated as separators. Enter and focus loss are the only
        commit triggers.
        """
        # While the suggestion menu is open the blur comes from clicking a
        # suggestion — the click commits the picked value, not the typed text.
        if history_enabled and suggest_menu.value:
            return
        # Committing typed text is a finished query round, so a held suggestion
        # list must not survive into the next render.
        _sticky["query"] = None
        args = getattr(event, "args", None) if event is not None else None
        text = str(args or "") or pending_input["value"]
        pending_input["value"] = ""
        text = text.strip()
        if not text:
            return
        current = list(chip_input.value or [])
        # Single-item inputs reject any additional value once at capacity.
        if max_items is not None and len(current) >= max_items:
            update_status()
            return
        value = _normalize_neuron_value(text)
        if value not in current:
            merged = current + [value]
            if max_items is not None:
                merged = merged[:max_items]
            sync_options(merged)
            chip_input.set_value(merged)
        update_status()

    def handle_value_change(_event):
        # Enter, chip removal, paste, and uploads all update the model value.
        # In each case there is no longer an uncommitted editor value.
        pending_input["value"] = ""
        current = list(chip_input.value or [])
        # Enforce the item cap even for values Quasar added natively (Enter).
        if max_items is not None and len(current) > max_items:
            current = current[:max_items]
        sync_options(current)
        if current != list(chip_input.value or []):
            chip_input.set_value(current)
            return
        update_status()

    # ------------------------------------------------------------------
    # Auto-suggest + query history. A provider enables both features for
    # neuron inputs; an explicit ``history_kind`` also enables the history
    # menu for inputs such as driver-line fields that do not have a dataset
    # suggestion provider. The selected store keeps line and neuron values
    # in separate namespaces.
    # ------------------------------------------------------------------
    suggest_menu = None
    if history_enabled:
        # The native QSelect popup would open empty on focus/typing; hide it
        # (CSS rule in ui/app.py) and drive the custom menu instead.
        chip_input.props('hide-dropdown-icon '
                         'popup-content-class="drocat-native-popup-hidden"')
        # Re-enter the per-input wrapper so QMenu uses this field as its
        # anchor. NiceGUI/Quasar portals the menu to the body at runtime, but
        # its position and outside-focus lifecycle still come from the parent.
        with chip_input_anchor:
            with ui.menu() as suggest_menu:
                pass
        suggest_menu.classes("drocat-suggest-menu")
        # Unique token pair ties THIS input's anchor to THIS menu in the
        # browser (class-scoped: NiceGUI 3.15 renders no DOM ids). The
        # keyboard-navigation script uses the pair to resolve the open menu
        # of the focused editor.
        suggest_token = next(_SUGGEST_TOKEN)
        chip_input_anchor.classes(f"drocat-suggest-anchor-{suggest_token}")
        suggest_menu.classes(f"drocat-suggest-menu-{suggest_token}")
        # The arrow-key navigation + scroll-anchoring scripts are page-global;
        # register them once per client connection.
        if not getattr(suggest_menu.client,
                       "_drocat_suggest_scripts_added", False):
            ui.add_head_html(_SUGGEST_KEYNAV_SCRIPT)
            ui.add_head_html(_SUGGEST_SCROLL_SCRIPT)
            suggest_menu.client._drocat_suggest_scripts_added = True
        # Anchor the popup to THIS input's wrapper explicitly. Parent-component
        # anchoring alone can detach (menu renders at the page origin) when the
        # input is rebuilt inside nested containers (e.g. the inline grouper's
        # cells); a selector target resolves the anchor element directly.
        suggest_menu.props(
            f'target="#{chip_input_anchor.html_id}" '
            'anchor="bottom start" self="top start" fit '
            'max-height=240px'
        )
        # The menu must NEVER take focus from the editor: QMenu focuses itself
        # on open by default, which blurs the QSelect, clears the typed text
        # and swallows further keystrokes. no-focus keeps the editor focused
        # while the menu overlays. (No explicit target: NiceGUI 3.15 renders
        # no DOM ids, so the menu anchors to the per-input wrapper instead of
        # relying on a selector target.)
        # ``no-focus`` keeps the popup from taking the editor focus when it
        # opens. ``no-refocus`` is equally important when a second QSelect is
        # clicked: closing the first popup must not put focus back on its old
        # editor after the new field has already been selected. The target
        # wrapper normally gives QMenu its own parent-click toggle; disable
        # that toggle because it can run after the QSelect focus event and
        # close a history menu just opened by the server.
        suggest_menu.props('no-focus no-refocus no-parent-event')
        suggest_menu.style("overflow-y: auto;")

        # Server-side closes (1-char gating, rebuilds, picks) must NOT commit
        # the pending editor text; only genuine client-side closes (outside
        # click / ESC) commit it, like a plain blur.
        _close_guard = {"value": False}
        # Whether the pointer is inside the menu: a blur caused by clicking a
        # suggestion must not hide the menu before the click lands.
        _pointer_in_menu = {"value": False}
        # Whether the editor currently has focus (the Recent list only
        # reopens after a finished input while the field is still in use).
        _focused = {"value": False}
        # QSelect forwards editor changes through ``input-value`` while the
        # native input also emits ``input``. Both are wired below because
        # either event can be skipped by a browser/Quasar transition; this
        # guard prevents the same keystroke from rebuilding the menu twice.
        _last_suggest_text = {"value": None}
        # Each input owns its own candidate history. Appended characters can
        # filter this list locally; a full backend match is needed only after
        # the continuation no longer has candidates.
        _candidate_state = {"text": None, "entries": []}
        # Whether the open menu currently shows the Recent/Frequent history
        # (as opposed to type-ahead suggestions). Drives the editor
        # click-to-toggle: click shows the history, the next click hides it,
        # the next shows it again. A held suggestion query (``_sticky``) is
        # dismissed by its own closing click, which also drops the hold.
        _menu_showing_history = {"value": False}
        # Set by the focus handler: the click that caused the focus must not
        # also toggle the history it just opened. Cleared by the next editor
        # click and on blur.
        _skip_next_editor_click = {"value": False}

        def _feature_enabled() -> bool:
            """Whether any part of the suggest/history UX is live. Either the
            type-ahead suggestions or the query history can be on alone, so the
            menu lifecycle runs when either is enabled; an empty editor still
            needs to decide between the two below."""
            from ..config import get_auto_suggest_enabled, get_show_history_enabled
            return bool(get_auto_suggest_enabled() or get_show_history_enabled())

        def _suggestions_enabled() -> bool:
            """Settings toggle: live type-ahead suggestions are on."""
            from ..config import get_auto_suggest_enabled
            return bool(get_auto_suggest_enabled())

        def _history_enabled() -> bool:
            """Settings toggle: the Recent/Frequent history list is on."""
            from ..config import get_show_history_enabled
            return bool(get_show_history_enabled())

        def _reset_candidate_state():
            _candidate_state["text"] = None
            _candidate_state["entries"] = []

        def _get_suggestions(text: str) -> List[Tuple[str, str]]:
            """Reuse a case-sensitive continuation candidate list when it
            still narrows; otherwise ask the backend for a fresh staged set.
            """
            query = str(text).strip()
            if not query:
                _reset_candidate_state()
                return []

            previous_text = _candidate_state["text"]
            previous_entries = _candidate_state["entries"]
            if previous_text == query:
                return list(previous_entries)

            if previous_text and query.startswith(previous_text):
                from ..type_suggestions import filter_candidate_entries

                narrowed = filter_candidate_entries(query, previous_entries)
                if narrowed:
                    _candidate_state["text"] = query
                    _candidate_state["entries"] = narrowed
                    return narrowed

            entries = list(suggestions(query) or []) if suggestions else []
            _candidate_state["text"] = query
            _candidate_state["entries"] = entries
            return entries

        def _refresh_menu():
            """Show freshly rebuilt content even when the menu is already
            open: clearing an open q-menu empties it (Quasar hides an empty
            menu) and open() on an open menu is a no-op — so close it
            (guarded, no commit) and reopen it. The menu's no-focus and
            no-refocus props keep the active QSelect editor in control during
            this close/open cycle without issuing a delayed focus command that
            could steal focus from the other input during a rapid handoff."""
            if suggest_menu.value:
                _close_suggest()
            suggest_menu.open()

        def _close_suggest():
            _close_guard["value"] = True
            suggest_menu.close()
            _menu_showing_history["value"] = False
            _close_guard["value"] = False

        def _clear_editor_text():
            """Drop the typed text so it cannot be committed as an extra chip.

            The wipe is enqueued before the model update on purpose. NiceGUI's
            outbox flushes coalesced element updates ahead of any run_method
            message, so the chip still reaches the browser first and the field
            empties after it, while Quasar's new-value-mode never sees leftover
            text alongside a foreign model change. The empty input-value event
            that the wipe produces is pre-empted too: unguarded it would
            rebuild the menu a second time and drop the highlight the client
            re-applied after the pick.
            """
            _last_suggest_text["value"] = ""
            chip_input.run_method("updateInputValue", "")

        def _commit_suggestion(value):
            """Add a clicked row's chip, or take it back out when it is marked.

            The held query (``_sticky``) decides what the finished-input
            handler renders afterwards: a suggestion pick keeps its own rows so
            the next entry can be added without retyping, while a history pick
            re-offers the Recent/Frequent list. Both lists mark the rows that
            stand for a current chip, so a second click on the same row is the
            reverse of the first -- the tick is a checkbox, not a receipt. The
            model shrinks through the same loopback as a chip's own ``x``, so
            the list re-renders itself with that row unmarked.
            """
            current = list(chip_input.value or [])
            # Only a suggestion row holds its query; a history row must not keep
            # the type-ahead list alive under a blank editor.
            held = None if _menu_showing_history["value"] else _sticky["query"]
            pending_input["value"] = ""
            if chip_is_marked(current, value):
                # Deselect. Checked before the capacity clamp on purpose: a
                # field that is full is exactly when taking a chip back out
                # needs to work.
                kept = without_chip(current, value)
                _focused["value"] = True
                _clear_editor_text()
                _sticky["query"] = held
                chip_input.set_value(kept)
                chip_input.run_method("focus")
                update_status()
                return
            if max_items is not None and len(current) >= max_items:
                # No room for another chip: holding rows that cannot be added
                # would only leave a list that answers to nothing.
                _sticky["query"] = None
                _clear_editor_text()
                _close_suggest()
                return
            merged = current + [value]
            if max_items is not None:
                merged = merged[:max_items]
            _clear_editor_text()
            # The field is still in use after a pick: keep it marked focused so
            # the finished-input handler re-offers a list instead of closing
            # it. Without this the very FIRST pick (before the field is already
            # flagged focused) could close the menu while later picks keep it
            # open.
            _focused["value"] = True
            sync_options(merged)
            _sticky["query"] = held
            if max_items is not None and len(merged) >= max_items:
                _sticky["query"] = None
            # A held query keeps its candidate pool: the re-render narrows the
            # cached rows instead of asking the provider again.
            if _sticky["query"] is None:
                _reset_candidate_state()
            chip_input.set_value(merged)
            # Keep the editor focused after a pick so focus does not escape to
            # <body> (the menu is portaled); the finished-input handler then
            # re-offers the held list instead of closing it.
            chip_input.run_method("focus")
            update_status()

        def _show_suggestions(entries, query=None):
            """Render type-ahead rows and hold them for the next pick.

            ``query`` is recorded as the held suggestion, so a pick re-renders
            the same rows instead of falling back to the history list. Which
            rows read as already-added is decided by ``marked_rows`` -- shared
            with the layer editor's overlay so the two lists cannot drift apart
            -- and clicking such a row deselects it.
            """
            # Always discard the previous query before handling the new one.
            # This matters when a narrower query has no candidates: the menu
            # is hidden, but its old rows must not survive into a later reopen.
            _menu_showing_history["value"] = False
            _sticky["query"] = query
            rows = marked_rows(entries, chip_input.value or [],
                               limit=suggestion_limit)
            suggest_menu.clear()
            if not rows:
                # Nothing left to offer for this query: close and let the next
                # focus re-offer the history list.
                _sticky["query"] = None
                _close_suggest()
                return
            with suggest_menu:
                for value, hint, added in rows:
                    row = ui.item().props("dense")
                    if added:
                        row.classes(ADDED_CLASS)
                        row.props(f'title="{MARKED_ROW_TITLE}"')
                    with row.on_click(lambda v=value: _commit_suggestion(v)):
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.label(value).classes("text-body2")
                            if hint:
                                ui.label(hint).classes(
                                    "text-caption drocat-muted")
                            if added:
                                # Stays listed so the rows never move under the
                                # pointer; the tick says this value is already a
                                # chip, and clicking the row again takes it back.
                                ui.icon("check").classes(
                                    f"text-caption drocat-muted {CHECK_CLASS}")
            _refresh_menu()

        def _show_history(query: str = ""):
            # Rendering the history list replaces any held suggestion query,
            # so a later pick has nothing to hold on to.
            _sticky["query"] = None
            if history_kind == "line":
                from ..line_history_store import (
                    datasets_of as _datasets_of,
                    frequent as _frequent,
                    prune_orphaned_custom as _prune_orphaned_custom,
                    recent as _recent,
                    remove as _remove,
                )
            elif history_kind == "type_mapping":
                from ..type_mapping_history import (
                    datasets_of as _datasets_of,
                    frequent as _frequent,
                    prune_orphaned_custom as _prune_orphaned_custom,
                    recent as _recent,
                    remove as _remove,
                )
            else:
                from ..history_store import (
                    datasets_of as _datasets_of,
                    frequent as _frequent,
                    prune_orphaned_custom as _prune_orphaned_custom,
                    recent as _recent,
                    remove as _remove,
                )

            valid_custom_labels = group_history.valid_labels()
            _prune_orphaned_custom(valid_custom_labels)

            def _history_dataset_scope():
                getter = history_datasets or available_neurons
                if getter is None:
                    return None
                try:
                    raw = getter()
                except Exception:
                    return None
                if isinstance(raw, str):
                    values = [raw]
                else:
                    try:
                        values = list(raw)
                    except TypeError:
                        values = [raw]
                values = [
                    str(value).strip()
                    for value in values
                    if isinstance(value, str) and str(value).strip()
                ]
                return values or None

            # Driver-line history is shared by NeuronBridge and FlyLight and
            # has no dataset dimension, even if a caller supplies a dataset
            # getter for another purpose.
            dataset_scope = (
                None if history_kind == "line" else _history_dataset_scope()
            )

            def matches_query(value: str) -> bool:
                # History filtering follows the same strict prefix behavior
                # as the first suggestion stage. It is intentionally kept
                # to history only while the editor has fewer than the
                # minimum number of characters for dataset suggestions.
                return not query or str(value).startswith(query)

            recents = [
                v for v in _recent(datasets=dataset_scope)
                if matches_query(v)
            ]
            freqs = [
                v for v in _frequent(datasets=dataset_scope)
                if v not in recents and matches_query(v)
            ]
            if not recents and not freqs:
                # Nothing left to offer (the prune x can empty the list while it
                # is open): rebuild the menu empty before closing it, exactly as
                # the type-ahead branch below does, so no stale row survives.
                suggest_menu.clear()
                _close_suggest()
                return

            # Suggestion-style hints (Type Mapping panel): one batched,
            # exact-membership lookup over the getter's datasets annotates
            # every row with the matched column and dataset(s) — the same
            # gray hint its auto-suggestion rows show. Built once per menu
            # render; a dataset without local pools simply contributes no
            # hint, and the history list itself stays unscoped.
            _history_hints = None
            if history_hint_datasets is not None:
                try:
                    hint_datasets = [
                        str(dataset).strip()
                        for dataset in (history_hint_datasets() or [])
                        if str(dataset).strip()
                    ]
                except Exception:
                    hint_datasets = []
                if hint_datasets:
                    from ..type_suggestions import (
                        dataset_aware_history_hints,
                    )

                    _history_hints = dataset_aware_history_hints(
                        [*recents, *freqs], hint_datasets)

            # Dataset provenance tags (cross-dataset tab): one batched,
            # exact-membership lookup over the selected datasets' local pools.
            # A badge then claims only datasets where the value actually
            # resolves — the recorded run scope alone would tag every chip
            # of a multi-dataset run with the whole selection. Only a failed
            # lookup (None) falls back to the recorded scope below.
            _resolved_datasets = None
            if show_history_datasets and history_kind != "line" \
                    and dataset_scope is not None:
                try:
                    from ..type_suggestions import datasets_resolving

                    _resolved_datasets = datasets_resolving(
                        [*recents, *freqs], dataset_scope)
                except Exception:
                    _resolved_datasets = None

            def _remove_history_value(value: str):
                # Removing an item is deliberately independent from picking
                # it. The client-side stopPropagation below keeps the parent
                # q-item from committing the value before this rerender.
                _remove(value)
                _show_history(query)

            def _history_hint(value: str) -> str | None:
                """Category hint for a history row, mirroring suggestion rows.

                With ``history_hint_datasets`` the row carries the full
                suggestion-style hint (matched column · dataset(s)), numeric
                rows included (cached instance per dataset). Otherwise
                body-ID rows use the cached instance as their hint and other
                values resolve their searched column (type/instance/bodyId)
                from the active dataset pools, so history rows read like
                auto-suggestion rows.
                """
                text = str(value).strip()
                if not text:
                    return None
                if _history_hints is not None:
                    return _history_hints.get(text)
                if re.fullmatch(r"\d+(?:\.0+)?", text):
                    if suggestions is None:
                        return None
                    try:
                        for candidate, hint in suggestions(text) or []:
                            if str(candidate) == text and hint:
                                # Body-ID pools use the corresponding
                                # instance as their hint; do not display the
                                # generic fallback.
                                if str(hint).casefold() != "bodyid":
                                    return str(hint)
                    except Exception:
                        # History must remain usable when a dataset is not
                        # local.
                        return None
                    return None
                # Non-bodyId values: resolve the searched column from the
                # selected dataset pools (lookup built lazily, once per menu
                # render, only when a non-bodyId row needs a hint).
                if dataset_scope is None:
                    return None
                lookup = _column_lookup()
                return lookup.get(text) if lookup is not None else None

            def _history_datasets(value: str) -> List[str]:
                """Dataset tags for one row, restricted to the datasets
                currently selected in the tab's dataset input.

                Tags come from exact membership over the selected datasets'
                local pools: values that resolve in none of them (regex
                patterns, names only in datasets without local tables) stay
                unannotated rather than re-claiming the recorded run scope.
                Only a failed membership lookup falls back to the recorded
                scope.
                """
                if (
                    history_kind == "line"
                    or not show_history_datasets
                    or dataset_scope is None
                ):
                    return []
                scope = set(dataset_scope)
                if _resolved_datasets is not None:
                    return [
                        dataset
                        for dataset in _resolved_datasets.get(
                            str(value).strip(), [])
                        if dataset in scope
                    ]
                return [
                    dataset for dataset in _datasets_of(value)
                    if dataset in scope
                ]

            _column_lookup_state = {"built": False, "lookup": None}

            def _column_lookup():
                """Value -> searched column over the selected datasets'
                pools, for history category hints. Built lazily so menus
                without non-bodyId rows never pay for the pool scan."""
                if _column_lookup_state["built"]:
                    return _column_lookup_state["lookup"]
                _column_lookup_state["built"] = True
                try:
                    from ..type_suggestions import (
                        get_dataset_pools,
                        suggestion_pool,
                    )

                    if len(dataset_scope) == 1:
                        pools = get_dataset_pools(dataset_scope[0])
                    else:
                        pools = suggestion_pool(dataset_scope)
                    lookup = {}
                    # bodyId is intentionally skipped: numeric history rows
                    # resolve through the instance-hint path above, and the
                    # bodyId pool is by far the largest column.
                    for column in ("type", "instance"):
                        for candidate, _ in pools.get(column, []):
                            # First matching column wins (type priority).
                            lookup.setdefault(str(candidate), column)
                    _column_lookup_state["lookup"] = lookup
                except Exception:
                    # History must remain usable without local pools.
                    _column_lookup_state["lookup"] = None
                return _column_lookup_state["lookup"]

            suggest_menu.clear()
            # History rows carry the same marks as suggestion rows: a value the
            # field already holds is ticked, and clicking it again takes that
            # chip back out. Only chips count -- an uploaded file's neurons are
            # not something a row click can remove, so ticking their rows would
            # promise a deselect that cannot happen.
            chipped = chip_input.value or []
            with suggest_menu:
                if recents:
                    ui.item("Recent").props("dense disabled").classes(
                        "text-caption drocat-muted drocat-suggest-header")
                    for v in recents:
                        _history_item(
                            v,
                            v in valid_custom_labels,
                            _history_hint(v),
                            _remove_history_value,
                            _history_datasets(v),
                            marked=chip_is_marked(chipped, v),
                        )
                if freqs:
                    ui.item("Frequent").props("dense disabled").classes(
                        "text-caption drocat-muted drocat-suggest-header")
                    for v in freqs:
                        _history_item(
                            v,
                            v in valid_custom_labels,
                            _history_hint(v),
                            _remove_history_value,
                            _history_datasets(v),
                            marked=chip_is_marked(chipped, v),
                        )
            _refresh_menu()
            # After the reopen cycle (refresh closes and reopens the menu),
            # mark the visible list as history so the editor click-to-toggle
            # knows a click should hide it.
            _menu_showing_history["value"] = True

        def _history_item(value, is_custom=False, hint=None, remove_handler=None,
                          datasets=None, marked=False):
            item = ui.item().props("dense")
            if marked:
                item.classes(ADDED_CLASS)
                # The row and its prune button now both look like a removal, so
                # say which is which: the row takes the chip back, the x takes
                # the entry out of the history.
                item.props(f'title="{MARKED_ROW_TITLE}"')
            with item.on_click(lambda v=value: _commit_suggestion(v)):
                with ui.row().classes("items-center gap-2 no-wrap w-full"):
                    ui.label(str(value)).classes("text-body2 flex-grow")
                    if hint:
                        ui.label(str(hint)).classes("text-caption drocat-muted")
                    # Dataset provenance tags (cross-dataset tab): one gray
                    # tag per SELECTED dataset whose local pool actually
                    # contains the value. Values resolving in none of the
                    # selected datasets (patterns, not-pulled datasets) show
                    # no tags here.
                    for dataset_name in datasets or []:
                        ui.badge(str(dataset_name)).props(
                            "outline dense").classes(
                            "text-caption drocat-muted "
                            "drocat-history-dataset-badge")
                    if is_custom:
                        ui.badge("custom", color="grey-6").props(
                            "outline dense")
                    if marked:
                        ui.icon("check").classes(
                            f"text-caption drocat-muted {CHECK_CLASS}")
                    if remove_handler is not None:
                        remove_button = ui.button(icon="close")
                        remove_button.props("flat round dense size=sm")
                        # Built into the row, not added on hover: ui.row() would
                        # reflow the list under the pointer mid-pick. The CSS in
                        # ui/app.py is what keeps it out of the way until the
                        # row is hovered or focused.
                        remove_button.classes(REMOVE_CLASS)
                        remove_button.tooltip("Remove from query history")
                        remove_button.on(
                            "click",
                            lambda _event, v=value: remove_handler(v),
                            js_handler=(
                                "(event) => { event.stopPropagation(); "
                                "emit(null); }"
                            ),
                        )

        def _on_suggest_input(event):
            # The editor text changed: refresh the list immediately (the
            # settings toggles switch suggestions and history off at runtime).
            if not _feature_enabled():
                _reset_candidate_state()
                _close_suggest()
                return
            text = str(getattr(event, "args", "") or "")
            if text == _last_suggest_text["value"]:
                return
            _last_suggest_text["value"] = text
            if len(text.strip()) < suggestion_min_chars:
                # This branch is configurable for callers that intentionally
                # want a longer minimum. The default is one character, so a
                # blank focused editor is the only default path that renders
                # query history instead of dataset suggestions.
                if text.strip():
                    # Under the minimum length, dataset suggestions need at
                    # least one typed character; history is the fallback.
                    if _suggestions_enabled():
                        _get_suggestions(text.strip())
                else:
                    # A manual clear starts a new query. Do not let the next
                    # query reuse candidates from the text that was erased.
                    _reset_candidate_state()
                if _focused["value"] and _history_enabled():
                    _show_history(text.strip())
                else:
                    _sticky["query"] = None
                    _close_suggest()
                    # Nothing was shown for this text: clear the dedup key so a
                    # later re-enable (or re-focus) of the feature re-processes
                    # the same keystroke.
                    _last_suggest_text["value"] = None
                return
            if _suggestions_enabled():
                typed = text.strip()
                _show_suggestions(_get_suggestions(typed), typed)
            else:
                _sticky["query"] = None
                _close_suggest()
                # Suggestions are toggled off: forget this text so re-enabling
                # the toggle lets the very next keystroke reopen the list.
                _last_suggest_text["value"] = None

        def _on_suggest_input_value(event):
            """Handle Quasar's component event only while this field owns
            focus.

            The event can be delivered after a QSelect-to-QSelect focus
            transition. In that case the shared ``document.activeElement``
            value belongs to the other field; ignoring the event prevents it
            from resurrecting this field's menu.
            """
            if not _focused["value"]:
                _reset_candidate_state()
                _close_suggest()
                return
            _on_suggest_input(event)

        def _on_suggest_focus(_event):
            _close_other_suggestion_menus()
            _focused["value"] = True
            # The focus itself was (almost always) caused by the click that
            # is still being processed; that click must not immediately
            # toggle the history the focus handler just opened.
            _skip_next_editor_click["value"] = True
            if not _feature_enabled():
                return
            # The menu is already showing suggestions (e.g. after a pick) —
            # do not flip it back to the history list.
            if suggest_menu.value:
                return
            # No editor text yet -> offer the persistent query history.
            if not pending_input["value"]:
                _reset_candidate_state()
                if _history_enabled():
                    _show_history()

        def _on_suggest_blur(event):
            # Quasar can emit a component-level blur while it is moving focus
            # from the QSelect shell to its internal search input. That is not
            # a real focus change, and closing here causes the history menu to
            # flash on the first click. The client-side blur listener waits
            # one tick and reports whether the labelled editor still owns
            # focus; an actual outside click reports ``still_inside=False``.
            blur_args = getattr(event, "args", None)
            if isinstance(blur_args, dict) and blur_args.get("still_inside"):
                return
            if not _feature_enabled():
                # A setting can be changed from the Settings tab while a
                # menu is open. Treat the next focus change as a real blur
                # even in that disabled state so no stale popup survives.
                _focused["value"] = False
                _sticky["query"] = None
                _reset_candidate_state()
                _close_suggest()
                return
            # Focus moved INTO the menu: a pick is about to happen; keep the
            # field "focused" so the finished-input handler offers Recent.
            if _pointer_in_menu["value"]:
                return
            _focused["value"] = False
            # Leaving the field is a closing click: the next focus re-offers
            # the history list rather than the query that was left behind.
            _sticky["query"] = None
            _reset_candidate_state()
            # Focus left the field: hide the list automatically, then commit
            # the pending text like a plain blur.
            _close_suggest()
            commit_pending_text()

        def _deactivate_for_focus_change():
            """Hide this menu when another neuron input receives focus."""
            _focused["value"] = False
            _pointer_in_menu["value"] = False
            _sticky["query"] = None
            _reset_candidate_state()
            _close_suggest()
            # If Quasar skipped the old field's blur event, preserve the
            # normal finished-input behavior and commit its editor text now.
            commit_pending_text()

        def _close_other_suggestion_menus():
            """Make this input the only active suggestion-menu owner."""
            client = suggest_menu.client
            # Store the registry on the client itself: it is naturally scoped
            # to one page connection and disappears with that client, while
            # a module-level registry could retain page closures.
            registrations = getattr(client, "_drocat_suggestion_menus", [])
            alive = []
            for menu, deactivate in registrations:
                if getattr(menu, "_deleted", False):
                    continue
                alive.append((menu, deactivate))
                if menu is not suggest_menu:
                    deactivate()
            client._drocat_suggestion_menus = alive

        def _on_menu_hide(_event):
            _pointer_in_menu["value"] = False
            if suggest_menu.value or _close_guard["value"]:
                # Reopened, or closed by a server rebuild (_refresh_menu):
                # neither ends the held suggestion query.
                return
            # The browser closed the list (ESC or an outside click) without a
            # pick: the held query is finished so the next focus offers the
            # history list, and any text still in the editor commits like a
            # plain blur. Suggestion commits clear pending_input first.
            _sticky["query"] = None
            if pending_input["value"]:
                commit_pending_text()

        def _finished_input(_event):
            # A chip was added or removed (Enter / pick / x): the input
            # finished. A held suggestion query keeps its own rows on screen so
            # the next entry can be picked without retyping (a removed chip
            # returns to the list, since the render drops only current chips).
            # Otherwise an empty editor in a still-used field re-offers the
            # Recent list, and anything else hides the menu.
            held = _sticky["query"]
            if (held and _focused["value"] and _suggestions_enabled()
                    and not _suppress_history_popup["value"]
                    and _feature_enabled()):
                _show_suggestions(_get_suggestions(held), held)
                return
            _reset_candidate_state()
            if not _feature_enabled():
                _close_suggest()
            elif _suppress_history_popup["value"]:
                _suppress_history_popup["value"] = False
                _close_suggest()
            elif not pending_input["value"] and _focused["value"] and _history_enabled():
                _show_history()
            else:
                _close_suggest()

        def _toggle_history_on_editor_click(event):
            """Toggle the Recent/Frequent list from the focused editor.

            Runs only for clicks that landed on the editor surface itself
            (the js_handler emits a boolean); clicks on chips, remove
            buttons or the field border are ignored. While the editor holds
            typed text (suggestion list showing), clicks never toggle — the
            suggestion rows follow the text, and hiding them mid-typing
            would read as a glitch. The click that CAUSED the focus is
            skipped: focus already opened the list.
            """
            if _skip_next_editor_click["value"]:
                _skip_next_editor_click["value"] = False
                return
            if not getattr(event, "args", False):
                return
            if not _focused["value"] or pending_input["value"]:
                return
            if suggest_menu.value and _sticky["query"]:
                # A held suggestion list is closed by the click that dismisses
                # it; the held query goes with it, so the next click opens the
                # history list instead of the query left behind by a pick.
                _sticky["query"] = None
                _reset_candidate_state()
                _close_suggest()
                return
            if not _history_enabled():
                return
            if suggest_menu.value and _menu_showing_history["value"]:
                _close_suggest()
            else:
                _show_history()

        registry = getattr(suggest_menu.client, "_drocat_suggestion_menus", [])
        registry.append((suggest_menu, _deactivate_for_focus_change))
        suggest_menu.client._drocat_suggestion_menus = registry

        chip_input.on("input", _on_suggest_input,
                      js_handler="(event) => emit(event?.target?.value ?? '')")
        # This is Quasar's canonical QSelect editor event. Keep it as a
        # second input source so updates continue even when the native DOM
        # input event is swallowed during popup/selection transitions.
        chip_input.on(
            "input-value",
            _on_suggest_input_value,
            # Quasar can emit this Vue event with the value from the previous
            # render while the native editor already contains the next
            # character. Read the focused editor at dispatch time so a stale
            # payload cannot reopen the previous query's rows.
            js_handler="() => emit(document.activeElement?.value ?? '')",
        )
        chip_input.on("focus", _on_suggest_focus)
        chip_input.on(
            "blur",
            _on_suggest_blur,
            js_handler=(
                "(event) => {"
                f"const label = {json.dumps(label)};"
                "setTimeout(() => {"
                "const active = document.activeElement;"
                "const stillInside = active?.getAttribute?.('aria-label') === label;"
                "emit({still_inside: Boolean(stillInside)});"
                "}, 0);"
                "}"
            ),
        )
        # The menu itself is not focusable, but keep its pointer guard so a
        # suggestion click is not treated as an outside blur before its item
        # handler commits the selected value.
        suggest_menu.on("mousedown",
                        lambda _e: _pointer_in_menu.__setitem__("value", True),
                        js_handler="(event) => emit(0)")
        suggest_menu.on_value_change(_on_menu_hide)

    # Capture the editor text while the user types. The native ``input`` event
    # (trusted typing sets the DOM value, and the js_handler ships the text) is
    # user-only, so it also tracks deletions; Quasar's ``input-value`` Vue event
    # mirrors the text but re-emits an empty value on blur-reset, which must be
    # ignored so the typed text survives until the commit handler runs.
    chip_input.on(
        "input",
        remember_user_input,
        js_handler="(event) => emit(event?.target?.value ?? '')",
    )
    chip_input.on(
        "input-value",
        remember_quasar_input,
        js_handler="() => emit(document.activeElement?.value ?? '')",
    )
    chip_input.on_value_change(handle_value_change)
    # Registered AFTER handle_value_change so the pending text is already
    # cleared when the "finished input" decision runs.
    if history_enabled:
        chip_input.on_value_change(_finished_input)
    # Commit the remembered editor text when the field loses focus, so a value
    # is added as a chip without requiring Enter. ``blur`` is a Quasar field
    # event and fires reliably when focus moves elsewhere.
    chip_input.on("blur", commit_pending_text)
    chip_input.on(
        "dblclick",
        start_edit_value,
        js_handler=(
            "(event) => {"
            "const chip = event.target.closest?.('.q-chip');"
            "if (!chip || event.target.closest?.('.q-chip__remove')) return;"
            "const content = chip.querySelector('.q-chip__content') || chip;"
            "const value = (content.textContent || '').trim();"
            "if (value) emit(value);"
            "}"
        ),
    )
    # Click-to-toggle the history list on the focused, empty editor: click
    # shows it, the next click hides it, the next shows it again — the same
    # dropdown convention as a plain combobox. Only clicks on the editor
    # surface (input / empty field area) participate; chips, their remove
    # buttons and suggestion rows keep their own behavior.
    if history_enabled:
        chip_input.on(
            "click",
            lambda event: _toggle_history_on_editor_click(event),
            js_handler=(
                "(event) => {"
                "  const t = event.target;"
                "  const onEditor = !!t && ("
                "    t.classList.contains('q-field__input') ||"
                "    t.classList.contains('q-field__native'));"
                "  emit(onEditor);"
                "}"
            ),
        )
    if clear_button is not None:
        clear_button.on_click(clear_all)
    update_status()

    def get_value():
        display_values = [*uploaded_neurons, *(chip_input.value or [])]
        combined = []
        # Keep the query surface readable.  A viewer selection may have an
        # exact body-ID snapshot in ``viewer_owned_body_ids`` for UI display
        # and safeguards, but execution must receive the selected name here.
        # The shared resolver converts that name to body IDs later, after it
        # has applied bodyId -> type -> instance -> metadata priority.  This
        # prevents type queries from leaking body IDs into logs, parameters,
        # attributes, and auto-named output folders.
        combined.extend(normalize_neuron(item) for item in display_values)
        combined = list(dict.fromkeys(combined))
        if max_items is not None:
            combined = combined[:max_items]
        mode = filter_mode.value if filter_mode is not None else "exact"
        return (mode, combined)

    container.get_value = get_value
    container.chip_input = chip_input
    container.chip_input_anchor = chip_input_anchor
    container.expand_button = expand_button
    container.filter_mode = filter_mode
    container.uploaded_neurons = uploaded_neurons
    container.suggest_menu = suggest_menu
    container.neuron_index_link = viewer_link

    def add_value_change_listener(callback: Callable[[], None]) -> None:
        """Register a callback for chip, upload, viewer, or clear changes."""
        if callable(callback) and callback not in value_change_callbacks:
            value_change_callbacks.append(callback)

    container.add_value_change_listener = add_value_change_listener

    def add_values(values) -> List:
        """Merge *values* into the chip list (programmatic append)."""
        current = list(chip_input.value or [])
        existing = {str(c) for c in current}
        for v in values:
            v = _normalize_neuron_value(v)
            if str(v) not in existing:
                current.append(v)
                existing.add(str(v))
        if max_items is not None:
            current = current[:max_items]
        _suppress_history_popup["value"] = True
        _sticky["query"] = None
        sync_options(current)
        chip_input.set_value(current)
        update_status()
        return current

    container.add_values = add_values
    return container


# =============================================================================
# Standard Form Inputs
# =============================================================================

def number_input(
    label: str,
    value: float = 0,
    min_val: float = 0,
    max_val: Optional[float] = 1000,
    step: float = 1,
    hint: str = "",
) -> ui.number:
    """Create a numeric input field with tooltip.

    Pass ``max_val=None`` when the input should have no artificial upper
    bound.
    """
    inp = ui.number(
        label=label,
        value=value,
        min=min_val,
        max=max_val,
        step=step,
    ).classes("w-full drocat-input")
    # A cleared field yields None, which crashes the int()/float()
    # coercions in run handlers; restore the last valid value instead so
    # every consumer keeps receiving a number.
    _last_valid = {"value": value}

    def _restore_last_valid(event):
        if event.value is None:
            inp.set_value(_last_valid["value"])
        else:
            _last_valid["value"] = event.value

    inp.on_value_change(_restore_last_valid)
    if hint:
        inp.tooltip(hint)
    return inp


def select_input(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    hint: str = "",
    help_doc: Optional[str] = None,
    inline: bool = False,
) -> ui.select:
    """Create a dropdown select input with tooltip.

    help_doc: name of a docs/ui_guides/*.html file; a small "guide" link is
    rendered under the select so the option list can be explained in depth.
    inline: use a compact fixed-width wrapper for placing the select beside
    a related checkbox or control in a flex row.
    """
    wrapper_classes = "gap-0 drocat-inline-select" if inline else "gap-0 w-full"
    with ui.column().classes(wrapper_classes):
        select_classes = "drocat-select" if inline else "w-full drocat-select"
        sel = ui.select(
            options=options,
            value=default or options[0],
            label=label,
        ).classes(select_classes)
        if hint:
            sel.tooltip(hint)
        if help_doc:
            ui.link(
                "Algorithm guide",
                f"docs/ui_guides/{help_doc}",
                new_tab=True,
            ).classes("text-caption drocat-doc-link")
    return sel


def checkbox_input(
    label: str,
    value: bool = True,
    hint: str = "",
) -> ui.checkbox:
    """Create a checkbox input with tooltip."""
    cb = ui.checkbox(label, value=value)
    if hint:
        cb.tooltip(hint)
    return cb


def _list_subdirs(path: str) -> List[str]:
    """Sorted subdirectory names under *path* ('' when not a directory)."""
    try:
        return sorted(
            (e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))),
            key=str.lower,
        )
    except OSError:
        return []


def _shell_quote_applescript(value: str) -> str:
    """Quote a path for an AppleScript string literal."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _macos_native_directory_picker_sync(
    title: str,
    initial: str,
) -> Tuple[bool, Optional[str]]:
    """Run the native macOS folder picker (``choose folder``) via osascript.

    The dialog runs as a subprocess so the UI event loop is never blocked.

    A session-level failure mode exists where the panel cannot be presented
    at all: osascript then returns -128 "User canceled" almost instantly
    without ever showing a window. A genuine cancel takes at least a moment
    (the user has to see and dismiss the dialog), so an instant -128 is
    reported as "picker unavailable" to let callers fall back to the
    in-app browser dialog instead of silently doing nothing.
    """
    executable = shutil.which("osascript")
    if not executable:
        return False, None
    script = (
        f"set startFolder to POSIX file {_shell_quote_applescript(str(initial))}\n"
        f"set chosenFolder to choose folder with prompt "
        f"{_shell_quote_applescript(title)} default location startFolder\n"
        "POSIX path of chosenFolder"
    )
    try:
        started = time.monotonic()
        completed = subprocess.run(
            [executable, "-e", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        elapsed = time.monotonic() - started
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    if completed.returncode == 0:
        selected = completed.stdout.strip()
        return True, selected or None
    # macOS uses -128 for the normal Cancel action.
    if "-128" in completed.stderr or "User canceled" in completed.stderr:
        if elapsed < 2.0:
            # No dialog could have been shown and dismissed this fast:
            # the panel never appeared, so treat it as unavailable.
            return False, None
        return True, None
    return False, None


def _shell_quote_powershell(value: str) -> str:
    """Quote a path for a single-quoted PowerShell string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _native_directory_picker_sync(
    title: str,
    initial: str,
) -> Tuple[bool, Optional[str]]:
    """Run a desktop folder picker without blocking NiceGUI's event loop.

    The boolean reports whether a supported picker was available. A supported
    picker returning no path means the user cancelled it; callers should not
    silently open a second dialog in that case.
    """
    initial_path = (
        Path(os.path.expanduser(initial)) if str(initial or "").strip()
        else Path.home()
    )
    if not initial_path.is_dir():
        initial_path = initial_path.parent if initial_path.parent.is_dir() else Path.home()
    system = platform.system()

    if system == "Darwin":
        # Native ``choose folder`` panel via osascript. In rare broken
        # sessions the panel cannot be presented at all; that failure is
        # detected inside the helper (instant -128) so the caller falls back
        # to the in-app browser dialog instead of silently doing nothing.
        return _macos_native_directory_picker_sync(title, str(initial_path))

    if system == "Windows":
        executable = shutil.which("powershell") or shutil.which("pwsh")
        if not executable:
            return False, None
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$dialog.Description = {_shell_quote_powershell(title)}; "
            f"$dialog.SelectedPath = {_shell_quote_powershell(str(initial_path))}; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ [Console]::WriteLine($dialog.SelectedPath) }"
        )
        try:
            completed = subprocess.run(
                [executable, "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False, None
        if completed.returncode == 0:
            selected = completed.stdout.strip()
            return True, selected or None
        return True, None

    # Linux desktop environments commonly expose one of these helpers. A
    # browser-only deployment normally has neither, so the in-app fallback is
    # still the portable path.
    zenity = shutil.which("zenity")
    kdialog = shutil.which("kdialog")
    if zenity:
        command = [
            zenity,
            "--file-selection",
            "--directory",
            f"--title={title}",
            f"--filename={str(initial_path)}{os.sep}",
        ]
    elif kdialog:
        command = [kdialog, "--getexistingdirectory", str(initial_path), title]
    else:
        return False, None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    if completed.returncode == 0:
        selected = completed.stdout.strip()
        return True, selected or None
    return True, None


async def native_directory_picker(
    title: str = "Select Directory",
    initial: str = "",
) -> Tuple[bool, Optional[str]]:
    """Open a local desktop directory chooser without blocking the UI."""
    return await asyncio.to_thread(
        _native_directory_picker_sync,
        title,
        initial,
    )


async def dir_browser_dialog(title: str = "Select Directory",
                             initial: str = "",
                             default_output: str = "") -> Optional[str]:
    """Open a polished in-browser directory picker.

    This is the portable fallback for browsers that cannot access the server's
    filesystem. It provides path navigation, filtering, quick roots, and an
    optional desktop-picker button without blocking the NiceGUI event loop.
    """
    state = {"path": ""}
    result = {"path": None}
    done = asyncio.Event()

    def _norm(p: str) -> str:
        p = os.path.expanduser((p or "").strip())
        if not p:
            p = default_output or str(Path.home())
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            p = os.path.dirname(p) or str(Path.home())
        return p

    def _render(p: str):
        p = _norm(p)
        state["path"] = p
        path_input.value = p
        current_path.text = p
        status.text = ""
        dir_list.clear()
        filter_text = str(folder_filter.value or "").strip().casefold()
        parent = os.path.dirname(p)
        if parent != p:
            with dir_list:
                ui.button(
                    f"Up to {parent}",
                    icon="arrow_upward",
                    on_click=lambda pp=parent: _render(pp),
                ).props("flat align=left dense").classes("w-full justify-start")
        subs = [entry for entry in _list_subdirs(p)
                if not filter_text or filter_text in entry.casefold()]
        folder_count.text = f"{len(subs):,} folder{'s' if len(subs) != 1 else ''}"
        if not subs:
            status.text = "No matching subfolders"
        for entry in subs:
            full = os.path.join(p, entry)
            with dir_list:
                ui.button(
                    entry,
                    icon="folder",
                    on_click=lambda f=full: _render(f),
                ).props("flat align=left dense").classes("w-full justify-start")

    def _go():
        raw = os.path.expanduser(str(path_input.value or "").strip())
        if not raw:
            _render(default_output or str(Path.home()))
            return
        p = os.path.abspath(raw)
        if os.path.isdir(p):
            _render(p)
        else:
            status.text = f"Not a directory: {p}"
            status.update()

    def _select():
        result["path"] = state["path"]
        dialog.close()
        done.set()

    def _cancel():
        dialog.close()
        done.set()

    async def _use_system_picker():
        available, selected = await native_directory_picker(title, state["path"])
        if selected:
            result["path"] = _norm(selected)
            dialog.close()
            done.set()
        elif not available:
            status.text = (
                "No desktop picker is available here. Use the path field or "
                "choose a folder from the list."
            )
            status.update()

    with ui.dialog() as dialog, ui.card().classes(
        "w-[min(94vw,760px)] max-w-none drocat-card drocat-dir-picker"
    ):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("folder_open", color="primary").classes("text-xl")
                with ui.column().classes("gap-0"):
                    ui.label(title).classes("text-h6")
                    ui.label(
                        "Choose where this tool will save its results."
                    ).classes("text-caption drocat-muted")
            ui.button(icon="close", on_click=_cancel).props("flat round dense")

        with ui.row().classes("w-full items-end gap-2"):
            path_input = ui.input("Current folder", value="").classes(
                "flex-grow drocat-input"
            )
            ui.button("Go", icon="arrow_forward", on_click=_go).props(
                "outline dense"
            )
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.button(
                "Home", icon="home", on_click=lambda: _render(str(Path.home()))
            ).props("flat dense")
            ui.button(
                "Project", icon="folder_special",
                on_click=lambda: _render(str(PROJECT_ROOT)),
            ).props("flat dense")
            ui.button(
                "System picker", icon="open_in_new", on_click=_use_system_picker
            ).props("flat dense").tooltip(
                "Use the macOS/Windows/Linux folder chooser when the UI runs on your desktop"
            )
        with ui.element("div").classes("w-full drocat-dir-current"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("folder", color="primary")
                current_path = ui.label("").classes("font-medium break-all")
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label("Folders in this location").classes("text-subtitle2")
            folder_count = ui.label("").classes("text-caption drocat-muted")
            folder_filter = ui.input(
                "Filter folders", placeholder="Type to narrow the list"
            ).props("dense clearable").classes("w-56 drocat-input")
        with ui.element("div").classes(
            "w-full h-72 overflow-auto drocat-dir-list"
        ):
            dir_list = ui.column().classes("w-full gap-1")
        status = ui.label("").classes("text-caption drocat-muted")
        with ui.row().classes("w-full items-center justify-end gap-2"):
            ui.button("Select this folder", icon="check", on_click=_select).props(
                "color=primary"
            )
            ui.button("Cancel", on_click=_cancel).props("flat")

    folder_filter.on_value_change(lambda _event: _render(state["path"]))

    _render(initial)
    dialog.open()
    await done.wait()
    return result["path"]


def dir_input(
    label: str = "Output Directory",
    default: Optional[str] = None,
    hint: str = "Where results will be saved. Use the folder button to choose a directory.",
    scope: Optional[str] = None,
    global_default: bool = False,
) -> ui.input:
    """Create a tab-local output directory input.

    Tool tabs persist their own override under ``scope`` and otherwise inherit
    the Settings-tab default. The Settings tab passes ``global_default=True``;
    only that field may change the global default for inherited paths.
    """
    scope = str(scope or "").strip() or None
    default_path = default or (
        get_default_output_dir()
        if global_default
        else get_tab_output_dir(scope)
    )

    tooltip_text = hint
    if scope and not global_default:
        tooltip_text = (
            f"{hint} This value overrides the Settings default for this tab; "
            "use the reset button to inherit it again."
        )
    elif global_default:
        tooltip_text = (
            f"{hint} This field controls the default used by tabs without "
            "their own override."
        )

    inp = ui.input(
        label=label,
        value=default_path,
    ).classes("w-full drocat-input drocat-output-dir").tooltip(tooltip_text)

    inherited = global_default or not has_tab_output_override(scope)
    inp.classes(
        add="drocat-output-dir-inherited" if inherited
        else "drocat-output-dir-override"
    )
    _OUTPUT_DIR_INPUTS.append({
        "input": inp,
        "scope": scope,
        "global": global_default,
    })

    def _persist_output_dir():
        raw = (inp.value or "").strip()
        if global_default:
            saved, effective = set_default_output_dir(raw, create=False)
            if saved and effective:
                inp.value = effective
                sync_output_dir_fields(inp, effective)
            return
        if scope:
            saved, effective = set_tab_output_dir(scope, raw, create=False)
            if saved and effective:
                inp.value = effective
                inp.classes(
                    add="drocat-output-dir-override" if raw
                    else "drocat-output-dir-inherited",
                    remove="drocat-output-dir-inherited" if raw
                    else "drocat-output-dir-override",
                )

    def _set_selected(selected: str):
        inp.value = selected
        _persist_output_dir()

    async def browse_system():
        available, selected = await native_directory_picker(
            title=f"Select {label}",
            initial=inp.value or str(PROJECT_ROOT),
        )
        if selected:
            _set_selected(selected)
        elif not available:
            await browse_panel()

    async def browse_panel():
        selected = await dir_browser_dialog(
            title=f"Select {label}",
            initial=inp.value or str(PROJECT_ROOT),
            default_output=inp.value or get_default_output_dir(),
        )
        if selected:
            _set_selected(selected)

    def reset_tab_override():
        if not scope or global_default:
            return
        saved, effective = set_tab_output_dir(scope, "", create=False)
        if saved and effective:
            inp.value = effective
            inp.classes(
                add="drocat-output-dir-inherited",
                remove="drocat-output-dir-override",
            )
            inp.update()
            ui.notify("This tab now uses the Settings default", type="positive")

    # One folder action: the native desktop picker first, with the in-app
    # browser dialog as its fallback when no desktop picker is available.
    with inp.add_slot("append"):
        ui.button(icon="folder_open", on_click=browse_system).props(
            "flat dense round"
        ).classes("drocat-dir-icon-btn").tooltip(
            "Open the system folder picker"
        )
        if scope and not global_default:
            ui.button(icon="restart_alt", on_click=reset_tab_override).props(
                "flat dense round"
            ).classes("drocat-dir-reset-btn").tooltip(
                "Use the Settings default for this tab"
            )

    inp.on("blur", _persist_output_dir)

    return inp


def sync_output_dir_fields(source, value: str, force: bool = False) -> None:
    """Update inherited output fields after a Settings-tab change.

    A normal global save updates fields that still inherit the default. A
    forced reset clears the distinction and updates every tab, including old
    persisted overrides.
    """
    for record in _OUTPUT_DIR_INPUTS:
        other = record["input"]
        eligible = (
            force
            or record["global"]
            or not has_tab_output_override(record["scope"])
        )
        if eligible and other is not source:
            try:
                if getattr(other, "value", None) != value:
                    other.value = value
                other.classes(
                    add="drocat-output-dir-inherited",
                    remove="drocat-output-dir-override",
                )
            except Exception:
                # Lazy tab panels keep their elements unmounted until first
                # activation: nicegui updates the server-side props but then
                # raises while building the change event (no client yet), so
                # the value is already correct when the panel mounts.
                pass
    try:
        selector = ".drocat-output-dir input" if force else ".drocat-output-dir-inherited input"
        ui.run_javascript(
            f"""
            document.querySelectorAll({json.dumps(selector)}).forEach(inp => {{
                if (inp.value !== {json.dumps(value)}) {{
                    inp.value = {json.dumps(value)};
                    inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }});
            """
        )
    except Exception:
        # No client connected (e.g. unit tests): backend values are enough.
        pass


# =============================================================================
# Chip List Input
# =============================================================================

def _clear_native_select_editor_after_selection(select: ui.select) -> None:
    """Clear a QSelect's filter text after a chip is added or removed.

    Quasar clears the editor for keyboard commits, but a mouse selection in a
    multiple QSelect can leave the text used to find the option in the editor.
    Query-like controls should expose that text only while searching, not as a
    second value beside the newly-created chip.
    """

    def clear_editor(_event) -> None:
        select.run_method("updateInputValue", "")

    select.on("add", clear_editor)
    select.on("remove", clear_editor)

def chip_list_input(
    label: str = "Items",
    placeholder: str = "Type and press Enter to add",
    hint: str = "Enter items one by one. Press Enter or comma to add a chip. Click X to remove.",
    initial: Optional[List[str]] = None,
) -> ui.select:
    """Create a chip-based input where each entry becomes a removable chip."""
    sel = ui.select(
        options=initial or [],
        value=initial or [],
        label=label,
        multiple=True,
    ).classes("w-full drocat-select drocat-chip-input").props('use-chips use-input new-value-mode="add-unique"').tooltip(hint)
    sel.props('input-debounce="0"')
    _clear_native_select_editor_after_selection(sel)
    return sel


def multi_select_input(
    label: str,
    options: List[str],
    default: Optional[List[str]] = None,
    hint: str = "",
    with_search: bool = True,
) -> ui.select:
    """Create a multi-select dropdown from predefined options."""
    sel = ui.select(
        options=options,
        value=default or [],
        label=label,
        multiple=True,
    ).classes("w-full drocat-select")
    if with_search:
        sel.props("use-input use-chips")
        _clear_native_select_editor_after_selection(sel)
    if hint:
        sel.tooltip(hint)
    return sel


def combo_input(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    hint: str = "",
) -> ui.select:
    """Create a combo box: dropdown with predefined options + free text."""
    sel = ui.select(
        options=options,
        value=default or (options[0] if options else ""),
        label=label,
    ).classes("w-full drocat-select").props('use-input new-value-mode="add-unique" fill-input hide-selected')
    if hint:
        sel.tooltip(hint)
    return sel


# =============================================================================
# Utility Functions
# =============================================================================

ALL_NEURONS_TOKEN = "all_neurons"
"""Special chip accepted as source or target in the Complete/Shortest Paths
tabs: loads the full neuron set on that side so the run fetches all adjacent
neurons at the given thresholds. It replaces every other chip and forces
Max Intermediate Layers = 0 (direct connections only)."""


def uses_all_neurons_token(values: List) -> bool:
    """True when the 'all_neurons' token is among the input chips.

    The token is matched case-insensitively and, when present, replaces every
    other chip in the same input (enforced by the UI and the backend).
    """
    return any(
        str(value or "").strip().casefold() == ALL_NEURONS_TOKEN
        for value in (values or [])
    )


def apply_filter_mode(neurons: List, mode: str) -> List:
    """Convert an input mode into the backend's query representation.

    Exact (including a missing/unknown mode) deliberately stays a bare value;
    :func:`statvis._process_single_neuron` treats bare values as strict,
    case-sensitive matches.  The other UI modes are made explicit as regex
    patterns so the backend cannot silently reinterpret an exact query as a
    prefix or substring search.
    """
    normalized = str(mode or "exact").strip().lower()
    normalized = normalized.replace(" ", "")
    aliases = {
        "startwith": "startswith",
        "startswith": "startswith",
        "endwith": "endswith",
        "endswith": "endswith",
        "contains": "contains",
        "regex": "regex",
        "exact": "exact",
    }
    normalized = aliases.get(normalized, "exact")
    if normalized == "exact":
        return list(neurons or [])
    result = []
    for n in neurons:
        # Starts/contains/ends-with are literal text modes.  Escape user
        # punctuation so their backend representation cannot accidentally
        # become a different regex query; the explicit Regex mode remains
        # available for callers who intentionally need regular expressions.
        s = re.escape(str(n))
        if normalized == "startswith":
            result.append(f"{s}.*")
        elif normalized == "contains":
            result.append(f".*{s}.*")
        elif normalized == "endswith":
            result.append(f".*{s}$")
        else:
            result.append(s)
    return result


def open_folder(path: str):
    """Open a folder in the system file manager (shared impl in ui.runner)."""
    from ..runner import open_folder as _open_folder
    _open_folder(path)


# =============================================================================
# Dataset Status Card (lazy - only fetches on button click)
# =============================================================================

def dataset_status_card() -> ui.card:
    """
    Create a card showing dataset availability status.
    Does NOT auto-fetch on page load. Click Refresh to fetch server metadata
    and re-check local converted tables.
    """
    from ..dataset_service import get_dataset_service
    import threading

    service = get_dataset_service()
    cached_results, cached_updated_at = service.get_cached_availability()
    state = {
        "results": cached_results or None,
        "updated_at": cached_updated_at,
        "error": None,
        "done": False,
        "running": False,
    }

    with ui.card().classes("w-full drocat-card") as card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dataset Availability").classes("drocat-card-title")
            with ui.row().classes("items-center gap-3"):
                last_updated = ui.label("Last updated at: not yet refreshed").classes(
                    "text-caption drocat-muted"
                )
                refresh_btn = ui.button(
                    "Refresh", icon="refresh", color="primary"
                ).props("flat dense").tooltip(
                    "Check local converted tables and server availability.\n"
                    "NeuPrint server status requires a valid token; FAFB and BANC use local release files."
                )

        ui.separator()
        status_container = ui.column().classes("w-full gap-1")

        with status_container:
            ui.label(
                "Click Refresh to query NeuPrint/Codex; local dataset folders are shown immediately."
            ).classes("text-caption drocat-muted")

        def _format_updated_at(updated_at):
            if not updated_at:
                return "Last updated at: not yet refreshed"
            # ISO timestamps are persisted with a timezone. Keep the display
            # compact while retaining the local offset for clarity.
            return f"Last updated at: {str(updated_at).replace('T', ' ')}"

        def render_results(results, updated_at=None):
            last_updated.text = _format_updated_at(
                updated_at if updated_at is not None else state.get("updated_at")
            )
            status_container.clear()
            with status_container:
                if not results:
                    ui.label(
                        "No datasets found. NeuPrint status needs a token; "
                        "FAFB/BANC status needs the converted local tables."
                    ).classes("text-caption drocat-warn")
                    return

                def count_badge(info, color):
                    text = f"{info.neuron_count:,}" if info.neuron_count else "n/a"
                    ui.badge(text, color=color).props("outline")

                def name_icon(info):
                    if info.local_prepared:
                        return "check_circle", "green"
                    if info.local_cache:
                        return "cached", "orange"
                    if info.server_state == SERVER_AVAILABLE:
                        return "cloud_done", "blue"
                    return "cloud_off", "grey"

                def chip(label, state, color_map, tooltip):
                    text, color = color_map.get(state, (str(state), "grey"))
                    ui.badge(f"{label}: {text}", color=color).props("outline") \
                        .tooltip(tooltip)

                SERVER_TEXT = {
                    SERVER_AVAILABLE: ("available", "green"),
                    SERVER_NO_TOKEN: ("no token", "grey"),
                    SERVER_TIMEOUT: ("timeout", "red"),
                    SERVER_UNREACHABLE: ("unreachable", "red"),
                    SERVER_HIDDEN: ("hidden", "grey"),
                    SERVER_UNKNOWN: ("unknown", "grey"),
                }
                CAP_TEXT = {
                    CAP_READY: ("ready", "green"),
                    CAP_ON_DEMAND: ("on-demand", "orange"),
                    CAP_PARTIAL: ("partial", "orange"),
                    CAP_MISSING: ("missing", "grey"),
                }

                for name, info in results.items():
                    # Trust the resolved source field.  BANC is a standalone
                    # public release; it must never be rendered as FlyWire.
                    family = getattr(info, "family", "") or str(
                        getattr(info, "source", "") or "").lower()
                    if family == "banc":
                        src_badge_text, src_badge_color = "BANC", "orange"
                    elif family == "fafb":
                        src_badge_text, src_badge_color = "FAFB", "purple"
                    else:
                        src_badge_text, src_badge_color = "NeuPrint", "blue"

                    icon_name, icon_color = name_icon(info)
                    with ui.row().classes("items-center gap-2 w-full drocat-status-row"):
                        ui.icon(icon_name, color=icon_color)
                        ui.label(info.display_name or name).classes(
                            "font-medium flex-grow").tooltip(
                            f"family: {family} | access: "
                            f"{getattr(info, 'access_mode', '') or 'n/a'}")
                        ui.badge(src_badge_text, color=src_badge_color).props("outline")
                        chip("server", info.server_state, SERVER_TEXT,
                             "NeuPrint/bucket/CAVE reachability (checked on Refresh)")
                        chip("metadata", info.metadata_state, CAP_TEXT,
                             "Analysis basics: neuron table + ROI table + index")
                        chip("connectivity", info.connectivity_state, CAP_TEXT,
                             "Connection tables (local) or server-streamed")
                        viz_state = info.visualization_state
                        viz_src = getattr(info, "visualization_source", None)
                        chip("viz", viz_state, CAP_TEXT,
                             f"Skeletons/meshes (source: {viz_src or 'n/a'})")
                        count_badge(info, icon_color)

        # The composed read is network-free: it returns persisted server
        # state (if any) plus disk-derived local readiness, so it is safe on
        # page load with no token and no prior refresh.
        initial_results = cached_results or service.get_cached_availability()[0]
        state["results"] = initial_results
        render_results(initial_results, cached_updated_at)
        refresh_dataset_selector_statuses(service)

        def render_error(msg):
            status_container.clear()
            with status_container:
                ui.label(f"Error: {msg}").classes("text-caption drocat-err")

        def do_refresh():
            try:
                results = service.refresh_availability()
                state["results"] = results
                state["updated_at"] = service.get_availability_updated_at()
                state["error"] = None
            except Exception as e:
                state["results"] = None
                state["error"] = str(e)
            finally:
                state["done"] = True
                state["running"] = False

        def on_refresh_click():
            if state["running"]:
                return
            state.update(running=True, done=False, results=None, error=None)
            refresh_btn.disable()
            status_container.clear()
            with status_container:
                ui.spinner("dots", size="sm")
                ui.label(
                    "Checking local files and NeuPrint/Codex availability... (may take 10-30s)"
                ).classes("text-caption drocat-muted")
            thread = threading.Thread(target=do_refresh, daemon=True)
            thread.start()

        refresh_btn.on_click(on_refresh_click)

        def poll_results():
            if state["done"]:
                state["done"] = False
                if state["results"] is not None:
                    render_results(state["results"], state.get("updated_at"))
                elif state["error"] is not None:
                    render_error(state["error"])
                refresh_btn.enable()

            # Keep local status live without issuing another network request.
            # This catches both a pull creating connections.parquet and an
            # external cleanup/removal while the Settings page is open.
            # Deliberately a 5s cadence (was 1s): the card is built for every
            # browser session and its timer keeps firing while the user works
            # in other tabs, so a 1s re-scan of every dataset directory on
            # the shared event loop multiplied into constant UI lag.
            if not state["running"] and state["results"] is not None:
                if _refresh_local_dataset_flags(state["results"], service):
                    render_results(state["results"], state.get("updated_at"))
            refresh_dataset_selector_statuses(service)

        ui.timer(5.0, poll_results)

    return card
