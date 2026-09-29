"""Interactive layer-style editor for the Skeleton tab (advanced table).

The editor keeps its rows in a disk-backed draft (see ``ui.layer_style_store``):
every change is auto-saved after a short debounce, so an accidental UI/port
shutdown never loses edits. A draft stays "dirty" (pending export) until the
user explicitly exports the CSV.

Each row is one neuron with an optional per-neuron color and a synapse /
pre-synaptic / post-synaptic color, exactly matching the Skeleton backend's
``layer_map_csv`` contract (``layer`` + ``neuron`` + optional color columns).
The table doubles as the in-page equivalent of uploading such a CSV: rows can
be loaded from an uploaded CSV and exported back to CSV.
"""
from datetime import datetime
import json
import tempfile
import time
from typing import Callable, List, Optional

from nicegui import ui

from .. import layer_style_store
from ..type_suggestions import dataset_suggestions
from .suggestion_list import (
    MARKED_ROW_TITLE,
    chip_is_marked,
    marked_rows,
    without_chip,
)

AUTOSAVE_DELAY = 0.6  # seconds between last edit and disk flush
# Auto-save is only worth persisting once the table holds a real style; a handful
# of half-typed scaffolding rows would otherwise create noisy/throwaway drafts.
AUTOSAVE_MIN_NON_EMPTY_ROWS = 5


def _notify(message: str, type: str = "info") -> None:
    """ui.notify that tolerates being called without an active UI slot
    (background tasks, direct calls from unit tests)."""
    try:
        ui.notify(message, type=type)
    except RuntimeError:
        pass


def _columns_for_mode(mode: str) -> list:
    """Table column definitions for a synapse mode.

    'synapse' shows a single ``synapse_color`` column; 'pre-post sites' shows
    the ``pre_synaptic_color`` / ``post_synaptic_color`` columns instead.
    """
    def column(name: str, label: str, *, sortable: bool = False) -> dict:
        # Column widths come from a per-column CSS variable (``--wc-<name>``) so the
        # user can drag the header resizer to resize the column interactively. The
        # width is applied to both the header and the body cells.
        return {
            "name": name,
            "label": label,
            "field": name,
            "align": "left",
            "sortable": sortable,
            "classes": f"drocat-{name.replace('_', '-')}-column",
            "headerClasses": f"drocat-{name.replace('_', '-')}-column",
            "style": f"width:var(--wc-{name})",
            "headerStyle": f"width:var(--wc-{name})",
        }

    cols = [
        # Layer is kept narrow (no sort icon) so the checkbox/layer columns stay
        # slim, leaving the remaining width for Neuron and the colour columns.
        column("layer", "Layer"),
        # ``neuron`` is a chip list in the body slot (``props.row.neurons``);
        # a list has no natural sort key, so disable sorting for that column.
        column("neuron", "Neuron"),
        column("color", "Color"),
    ]
    if mode == "pre-post sites":
        cols += [
            column("pre_synaptic_color", "Pre-synaptic color"),
            column("post_synaptic_color", "Post-synaptic color"),
        ]
    else:
        cols += [
            column("synapse_color", "Synapse color"),
        ]
    return cols


_TABLE_HEADER_SLOT = r"""
<q-tr :props="props" class="drocat-edge-header-row">
  <q-th class="drocat-edge-select-cell">
    <q-checkbox
      v-model="props.selected"
      :indeterminate="props.selected === null"
      dense
    />
    <span class="drocat-col-resizer" data-col="select"></span>
  </q-th>
  <q-th
    v-for="col in props.cols"
    :key="col.name"
    :props="props"
    class="drocat-edge-header-cell"
    :class="[col.headerClasses || '', { 'drocat-edge-divider': col.name !== 'color' }]"
  >
    {{ col.label }}
    <span class="drocat-col-resizer" :data-col="col.name"></span>
  </q-th>
</q-tr>
"""


# Drag-to-resize helper for the table headers. It updates ``--wc-<col>`` on the
# table element (driving each column's width), purely client-side so resizing never
# rebuilds the table or steals focus from a cell being edited. Listeners are
# attached to ``document`` (not ``window``) so a real mouse drag keeps firing even
# when the pointer travels outside the header; pointer events cover both mice and
# trackpads.
_COL_RESIZE_JS = r"""
if (!window.drocatStartColResize) {
  // ``event`` is the real mousedown; the header cell is read from ``event.target``
  // because the resizer span is the target of the delegated listener.
  window.drocatStartColResize = function (event, col) {
    event.preventDefault();
    const span = event.target && event.target.closest
      ? event.target.closest('.drocat-col-resizer')
      : null;
    const th = span ? span.closest('th') : null;
    if (!th) return;
    const table = th.closest('.q-table');
    if (!table) return;
    const startX = event.clientX;
    const startW = th.getBoundingClientRect().width;
    // Column cell class mirrors the Python ``_columns_for_mode`` ``classes`` value.
    const colClass = 'drocat-' + String(col).replace(/_/g, '-') + '-column';
    const onMove = function (ev) {
      const w = Math.max(42, startW + (ev.clientX - startX));
      table.style.setProperty('--wc-' + col, w + 'px');
      // ``table-layout:auto`` treats ``width`` as a hint; force the rendered width
      // by setting ``min-width`` on every cell of the column so the column actually
      // grows (or shrinks) and the table (implicitly) widens around it.
      const cells = table.querySelectorAll('.' + colClass);
      for (let i = 0; i < cells.length; i++) {
        cells[i].style.minWidth = w + 'px';
      }
    };
    const onUp = function () {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
    };
    document.body.style.cursor = 'col-resize';
    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };
}
// Delegate the mousedown to ``document`` in the capture phase: the resizer spans'
// own Vue ``@mousedown`` binding does not fire in the table's compiled slot, but
// a document-level capture listener always sees the press. The span carries the
// column name in ``data-col`` so a single handler covers every header cell.
if (!window.drocatResizeDelegated) {
  window.drocatResizeDelegated = true;
  document.addEventListener('mousedown', function (ev) {
    const span = ev.target && ev.target.closest
      ? ev.target.closest('.drocat-col-resizer')
      : null;
    if (!span) return;
    const col = span.getAttribute('data-col');
    if (!col) return;
    // Swallow the press so the header's sort/click and text selection do not
    // fire while the user is resizing (capture phase, before the th handles it).
    ev.preventDefault();
    ev.stopPropagation();
    window.drocatStartColResize(ev, col);
  }, true);
}
"""


# Client-side suggestion overlay renderer. The overlay DIV is a single static NiceGUI
# element; its items are pure DOM injected here (never NiceGUI elements) so typing or
# focusing never re-renders the q-table body (which would remount the focused
# q-select and wipe typed text). Three socket listeners drive it from Python:
# ``pick`` (commit a row), ``remove`` (prune a history row) and ``toggle``
# (a click on the focused cell, or Escape, opening/closing the list).
# ``__OVERLAY_ID__``/``__PICK_LID__``/``__REMOVE_LID__``/``__TOGGLE_LID__`` are
# substituted at build time with the overlay's element id + those listener ids,
# and ``__MARKED_TITLE_JSON__`` with the shared marked-row hover text.
_SUGGESTION_JS = r"""
function drocatSuggestEsc(text) {
  // Rows are built as raw DOM from dataset/history values, which include names
  // the user typed and later committed to history.
  return String(text == null ? '' : text).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
            "'": '&#39;'}[c];
  });
}
// Hover text for a marked row, shared with the query box via suggestion_list.
var __drocatMarkedTitle = __MARKED_TITLE_JSON__;
// The q-select input of the cell the overlay is anchored to. ARIA state that
// belongs to the combobox (expanded, active descendant) has to be written here,
// because focus never leaves this input while the list is open.
function drocatSuggestInput() {
  var rowId = window.__drocatSuggestAnchorRow;
  if (rowId === undefined || rowId === null) return null;
  var cell = document.getElementById('neuron-cell-' + rowId);
  return cell ? cell.querySelector('input') : null;
}
function drocatSuggestSetHighlight(rows, index) {
  for (var i = 0; i < rows.length; i++) {
    rows[i].classList.remove('drocat-suggest-active');
  }
  var input = drocatSuggestInput();
  if (index >= 0 && index < rows.length) {
    rows[index].classList.add('drocat-suggest-active');
    rows[index].scrollIntoView({ block: 'nearest' });
    // The highlight is visual only, so a screen reader needs the pointer to
    // say which option the next Enter commits.
    if (input) input.setAttribute('aria-activedescendant', rows[index].id);
  } else if (input) {
    input.removeAttribute('aria-activedescendant');
  }
}
// Anchor the overlay just below the cell. The overlay is position:fixed, so it
// uses viewport coords; re-applying the rect on scroll keeps it glued to the
// cell as the page moves (matching the standard query box's menu).
function drocatSuggestPosition(el, o) {
  var r = el.getBoundingClientRect();
  o.style.top = r.bottom + 'px';
  o.style.left = r.left + 'px';
  o.style.width = Math.max(r.width, 180) + 'px';
}
// Re-glue the overlay to its anchor cell (no-op when hidden). Used by the
// scroll/resize listeners and by the requestAnimationFrame loop below, so it
// keeps tracking the cell even if a scroll event is not dispatched.
function drocatSuggestReposition() {
  var o = document.getElementById('drocat-suggest-overlay');
  if (!o || o.style.display === 'none') return;
  var rowId = window.__drocatSuggestAnchorRow;
  if (rowId === undefined) return;
  var el = document.getElementById('neuron-cell-' + rowId);
  if (el) drocatSuggestPosition(el, o);
}
// A rAF loop that re-anchors every frame while the overlay is open. Quasar
// menus and real browsers fire scroll events, but programmatic/automated
// scrolls can skip them; polling the cell rect every frame is cheap and keeps
// the overlay glued in every case.
function drocatSuggestStartTrack() {
  var tick = function () {
    var o = document.getElementById('drocat-suggest-overlay');
    if (!o || o.style.display === 'none') {
      window.__drocatSuggestRaf = null;
      return;
    }
    drocatSuggestReposition();
    window.__drocatSuggestRaf = requestAnimationFrame(tick);
  };
  if (!window.__drocatSuggestRaf) {
    window.__drocatSuggestRaf = requestAnimationFrame(tick);
  }
}
function drocatSuggestStopTrack() {
  if (window.__drocatSuggestRaf) {
    cancelAnimationFrame(window.__drocatSuggestRaf);
    window.__drocatSuggestRaf = null;
  }
}
window.drocatSuggest = {
  render: function (rowId, items, isHistory) {
    var el = document.getElementById('neuron-cell-' + rowId);
    var o = document.getElementById('drocat-suggest-overlay');
    if (!el || !o) return;
    // A transformed ancestor (the tab-panel slide transition keeps a persistent
    // transform) becomes the containing block for position:fixed descendants, so
    // absolute viewport coordinates would be displaced. Relocate the overlay to
    // <body> so 'fixed' resolves against the viewport where the cell rect lives.
    if (o.parentElement !== document.body) {
      document.body.appendChild(o);
    }
    // Keep the anchored row so the scroll listener can re-glue the overlay to it.
    window.__drocatSuggestAnchorRow = rowId;
    drocatSuggestPosition(el, o);
    // The list starts flush at the cell; the switch-cell mousedown guard already
    // distinguishes an item click (pick) from a click on a lower cell (focus
    // switch), so no transparent spacer is needed and no empty gap is shown.
    // The cell's q-select is the combobox; this list is its popup. ARIA state
    // lives on the input because focus never moves here (moving it would
    // remount the cell), so the list is announced through aria-controls and
    // aria-activedescendant rather than by being focusable itself.
    o.setAttribute('role', 'listbox');
    o.setAttribute('aria-multiselectable', 'true');
    o.setAttribute('aria-label', isHistory ? 'Recent neurons' : 'Neuron suggestions');
    var combobox = el.querySelector('input');
    if (combobox) {
      combobox.setAttribute('aria-expanded', 'true');
      combobox.setAttribute('aria-controls', 'drocat-suggest-overlay');
    }
    var html = '';
    if (isHistory) {
      // A listbox may only contain options, so the caption is presentational.
      html += '<div class="drocat-suggest-header" role="presentation">Recent</div>';
    }
    for (var i = 0; i < items.length; i++) {
      var value = items[i][0];
      var hint = items[i][1] || '';
      // Third element: the value is already a chip in this cell, in the
      // type-ahead list and in Recent alike. Like the query box, such a row is
      // marked rather than dropped -- the list never shifts under the pointer
      // -- and clicking it takes that chip back out.
      var added = items[i][2] === true;
      html += '<div class="drocat-suggest-item'
              + (added ? ' drocat-suggest-added' : '')
              + '" id="drocat-suggest-opt-' + i + '" role="option"'
              + ' aria-selected="' + (added ? 'true' : 'false') + '"'
              + (added ? ' title="' + drocatSuggestEsc(__drocatMarkedTitle) + '"'
                       : '')
              + ' data-value="' + drocatSuggestEsc(value) + '">';
      html += '<span class="drocat-suggest-label">' + drocatSuggestEsc(value)
              + '</span>';
      if (hint) html += '<span class="drocat-suggest-hint">'
                        + drocatSuggestEsc(hint) + '</span>';
      if (added) {
        // Decorative: aria-selected already carries the state.
        html += '<span class="drocat-suggest-check material-icons"'
                + ' aria-hidden="true">check</span>';
      }
      // History rows can be pruned individually, mirroring the query box. The
      // control is mouse-only, so it stays out of the option's name, and the CSS
      // holds it at opacity 0 until the row is hovered or highlighted: a marked
      // row's tick takes the chip back and this takes the history entry, so the
      // two must not be on screen together.
      if (isHistory) {
        html += '<span class="drocat-suggest-remove" aria-hidden="true"'
                + ' data-value="' + drocatSuggestEsc(value)
                + '" title="Remove from history">' + '×</span>';
      }
      html += '</div>';
    }
    o.innerHTML = html;
    o.style.display = 'block';
    // A keyboard pick asks for the next row to be highlighted once the rebuilt
    // list lands, so repeated Enter presses keep selecting down the list. The
    // row index is client state, so it is consumed here and always cleared.
    var pending = window.__drocatSuggestPending;
    window.__drocatSuggestPending = -1;
    if (pending >= 0) {
      var rows = o.querySelectorAll('.drocat-suggest-item');
      drocatSuggestSetHighlight(rows, Math.min(pending, rows.length - 1));
    }
    drocatSuggestStartTrack();
  },
  hide: function () {
    var o = document.getElementById('drocat-suggest-overlay');
    if (o) { o.style.display = 'none'; }
    var input = drocatSuggestInput();
    if (input) { input.setAttribute('aria-expanded', 'false'); }
    window.__drocatSuggestPending = -1;
    drocatSuggestStopTrack();
  },
  emit: function (listenerId, value) {
    if (window.socket && window.did_handshake) {
      window.socket.emit('event', {
        id: __OVERLAY_ID__,
        client_id: window.clientId,
        listener_id: listenerId,
        // Always exactly one JSON argument: Client.handle_event iterates
        // msg['args'], so an argument-less event must still send a list.
        args: [JSON.stringify(value)]
      });
    }
  },
  pick: function (value) {
    window.drocatSuggest.emit('__PICK_LID__', value);
  },
  remove: function (value) {
    window.drocatSuggest.emit('__REMOVE_LID__', value);
  },
  // A press on the already-focused cell, or Escape: the Python side decides
  // whether that closes the list or brings it back (the query box's
  // click-to-toggle), so this only reports -- and hides, when a list was up.
  toggle: function (wasVisible) {
    if (wasVisible) { window.drocatSuggest.hide(); }
    window.drocatSuggest.emit('__TOGGLE_LID__', wasVisible === true);
  }
};
// Delegated (capture) click listener: attached at page load so it persists even
// though the overlay div is created later (a IIFE that runs in <head> would find
// the overlay already gone/not-yet-rendered and skip binding, leaving the items
// unclickable). A document-level listener catches every .drocat-suggest-item.
// Scoped to THIS overlay, in both branches: the shared classes also dress the
// query box's rows and history prune (one reveal rule, one row contract). A
// capture-phase stopPropagation below document level runs before the query
// box's own click handlers, so an unscoped match swallowed every prune click
// there — the event died at the document and the button's own handler never
// ran, making "Remove from query history" a silent no-op while this overlay's
// remove(null) fired harmlessly in its place.
if (!window.drocatSuggestDelegated) {
  window.drocatSuggestDelegated = true;
  document.addEventListener('click', function (ev) {
    var o = document.getElementById('drocat-suggest-overlay');
    if (!o || o.style.display === 'none') return;
    var rem = ev.target && ev.target.closest ? ev.target.closest('.drocat-suggest-remove') : null;
    if (rem && o.contains(rem)) {
      ev.preventDefault();
      ev.stopPropagation();
      window.drocatSuggest.remove(rem.getAttribute('data-value'));
      return;
    }
    var t = ev.target && ev.target.closest ? ev.target.closest('.drocat-suggest-item') : null;
    if (t && o.contains(t)) {
      ev.preventDefault();
      ev.stopPropagation();
      window.drocatSuggest.pick(t.getAttribute('data-value'));
    }
  }, true);
}
// Down-key navigation into the open overlay: ArrowDown moves the highlight down
// (entering the list from the cell), ArrowUp leaves it from the first row, Enter
// picks the highlighted row (and asks for the next one, mirroring the standard
// query box), and Escape closes the list the way an outside click would.
if (!window.drocatSuggestNav) {
  window.drocatSuggestNav = true;
  document.addEventListener('keydown', function (event) {
    // The pick's simulated chip entry presses Enter programmatically; that
    // keypress must not come back around as another pick.
    if (window.__drocatSuggestSim) return;
    if (!['ArrowDown', 'ArrowUp', 'Enter', 'Tab', 'Escape'].includes(event.key)) return;
    var o = document.getElementById('drocat-suggest-overlay');
    if (!o || o.style.display === 'none') return;
    var rows = o.querySelectorAll('.drocat-suggest-item');
    if (!rows.length) return;
    var current = -1;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].classList.contains('drocat-suggest-active')) current = i;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      window.drocatSuggest.toggle(true);
      return;
    }
    if (event.key === 'ArrowDown') {
      if (current < rows.length - 1) {
        event.preventDefault();
        drocatSuggestSetHighlight(rows, current + 1);
      }
    } else if (event.key === 'ArrowUp') {
      if (current === 0) {
        event.preventDefault();
        drocatSuggestSetHighlight(rows, -1);
      } else if (current > 0) {
        event.preventDefault();
        drocatSuggestSetHighlight(rows, current - 1);
      }
    } else if (event.key === 'Enter' || event.key === 'Tab') {
      if (current === -1) return;
      event.preventDefault();
      event.stopPropagation();
      // Remember where the highlight was headed so the server's rebuild of the
      // same rows lands with the highlight already on the next entry.
      window.__drocatSuggestPending = current + 1;
      window.drocatSuggest.pick(rows[current].getAttribute('data-value'));
    }
  }, true);
}
// Re-anchor on scroll/resize as a first-line response (the rAF loop in
// drocatSuggestStartTrack covers the case where a scroll event is not
// dispatched). Capture-phase scroll catches scrolls from nested containers as
// well as the window; both are a no-op when the overlay is hidden.
if (!window.drocatSuggestReposition) {
  window.drocatSuggestReposition = true;
  document.addEventListener('scroll', drocatSuggestReposition, true);
  window.addEventListener('resize', drocatSuggestReposition);
}
// Switch-cell guard: the overlay is an inline dropdown below the focused cell,
// so its suggestion items overlap the neuron cells of the rows underneath. A
// real click on one of those lower cells would otherwise land on an item and
// commit a stray suggestion to the still-focused cell instead of switching
// focus. On mousedown, if the pointer is over a DIFFERENT neuron cell (and the
// overlay is open), close the overlay and focus that cell so the intended
// switch happens.
if (!window.drocatSuggestCellSwitch) {
  window.drocatSuggestCellSwitch = true;
  document.addEventListener('mousedown', function (ev) {
    var o = document.getElementById('drocat-suggest-overlay');
    if (!o || o.style.display === 'none') return;
    // A click on a suggestion item / header / remove button is an intentional
    // pick (or history prune), never a cell switch. The item overlaps a lower
    // row's cell rect, so decide by element type, not coords. preventDefault on
    // mousedown also stops the focused cell from blurring: the overlay is a
    // plain div (not a no-focus Quasar menu), so without it the click would drop
    // focus, firing commit/blur and hiding the list midway through a pick.
    var t = ev.target;
    if (t && t.closest && (t.closest('.drocat-suggest-item') ||
        t.closest('.drocat-suggest-header') ||
        t.closest('.drocat-suggest-remove'))) {
      ev.preventDefault();
      return;
    }
    var or = o.getBoundingClientRect();
    if (ev.clientX < or.left || ev.clientX > or.right ||
        ev.clientY < or.top || ev.clientY > or.bottom) return;
    var focusedRow = window.__drocatSuggestAnchorRow;
    var cells = document.querySelectorAll('.drocat-neuron-cell');
    for (var i = 0; i < cells.length; i++) {
      var cell = cells[i];
      if (cell.id === 'neuron-cell-' + focusedRow) continue;
      var r = cell.getBoundingClientRect();
      if (ev.clientX >= r.left && ev.clientX <= r.right &&
          ev.clientY >= r.top && ev.clientY <= r.bottom) {
        ev.preventDefault();
        ev.stopPropagation();
        window.drocatSuggest.hide();
        var input = cell.querySelector('.q-field__input') || cell.querySelector('input');
        if (input) input.focus();
        return;
      }
    }
  }, true);
}
// Click-to-toggle on the focused cell, mirroring the query box: a press on the
// cell that owns the list reports itself and Python decides whether that closes
// the list or brings it back (a dismissal click never re-opens it in the same
// breath). The switch-cell guard above deliberately skips this cell, and chips
// keep their own meaning, so a press on a chip's x never toggles. Only a press
// that closes a visible list swallows the native click; with the list already
// hidden the cell keeps its ordinary click behavior (focus, text selection) and
// this merely reports the press, which Python ignores unless it was the
// dismissal that hid the list.
if (!window.drocatSuggestToggle) {
  window.drocatSuggestToggle = true;
  document.addEventListener('mousedown', function (ev) {
    var t = ev.target;
    if (!t || !t.closest) return;
    var cell = t.closest('.drocat-neuron-cell');
    if (!cell || cell.id !== 'neuron-cell-' + window.__drocatSuggestAnchorRow) return;
    if (t.closest('.q-chip')) return;
    var o = document.getElementById('drocat-suggest-overlay');
    if (o && o.style.display !== 'none') {
      ev.preventDefault();
      window.drocatSuggest.toggle(true);
    } else {
      window.drocatSuggest.toggle(false);
    }
  }, true);
}
// Focus continuity for the editor table: in this NiceGUI/Quasar stack EVERY
// table re-render rebuilds all body rows (rows prop, a selection toggle, any
// element update landing in the same render cycle), which drops the focus and
// the caret mid-edit. When a blur inside the table is immediately followed by
// row removals AND the focus fell to <body> (i.e. the user did not go
// somewhere else), put the caret back into the same cell of the rebuilt row
// and restore the field content the rebuild wiped.
if (!window.__drocatCellFocusGuard) {
  window.__drocatCellFocusGuard = true;
  var drocatLastCellFocus = null;
  var drocatRowsRebuiltAt = 0;
  document.addEventListener('focusout', function (ev) {
    var inp = ev.target;
    if (!inp || (inp.tagName !== 'INPUT' && inp.tagName !== 'TEXTAREA' && inp.tagName !== 'BUTTON')) return;
    var td = inp.closest ? inp.closest('.drocat-edge-table td') : null;
    if (!td) { drocatLastCellFocus = null; return; }
    var tr = td.closest('tr');
    var tbody = tr ? tr.parentElement : null;
    var rec = {
      at: Date.now(),
      rowId: tr ? tr.getAttribute('data-row-id') : null,
      rowIndex: tbody ? Array.prototype.indexOf.call(tbody.children, tr) : -1,
      cellIndex: Array.prototype.indexOf.call(tr.children, td),
      tag: inp.tagName,
      caret: null, value: '',
    };
    try { rec.caret = inp.selectionStart; rec.value = inp.value; } catch (e) {}
    drocatLastCellFocus = rec;
    setTimeout(function () {
      var rec2 = drocatLastCellFocus;
      drocatLastCellFocus = null;
      if (!rec2) return;
      var ae = document.activeElement;
      if (ae && ae !== document.body) return;              // user went elsewhere
      if (Date.now() - rec2.at > 300) return;              // not this blur
      if (Date.now() - drocatRowsRebuiltAt > 300) return;  // no rebuild happened
      var tb = document.querySelector('.drocat-edge-table tbody');
      if (!tb) return;
      var tr2 = rec2.rowId != null
        ? tb.querySelector('tr[data-row-id="' + rec2.rowId + '"]')
        : tb.children[rec2.rowIndex];
      if (!tr2) return;
      var td2 = tr2.children[rec2.cellIndex];
      var inp2 = td2 && td2.querySelector(rec2.tag.toLowerCase());
      if (!inp2) return;
      inp2.focus();
      if (inp2.tagName === 'BUTTON') return;
      if (!inp2.value && rec2.value) {
        // Replay the wiped text only into the SAME logical row (the body
        // slots stamp data-row-id): a purely positional restore after a
        // row shift would commit it into whatever row now occupies the
        // recorded index, and a deleted row restores nothing at all.
        var sameRow = rec2.rowId != null ||
          (Array.prototype.indexOf.call(tb.children, tr2) === rec2.rowIndex);
        if (sameRow) {
          inp2.value = rec2.value;
          inp2.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
      if (rec2.caret != null && inp2.setSelectionRange) {
        try {
          var len = (inp2.value || '').length;
          var pos = Math.min(rec2.caret, len);
          inp2.setSelectionRange(pos, pos);
        } catch (e) {}
      }
    }, 0);
  }, true);
  var drocatRowMo = new MutationObserver(function (muts) {
    for (var i = 0; i < muts.length; i++) {
      var removed = muts[i].removedNodes;
      for (var j = 0; j < removed.length; j++) {
        if (removed[j].nodeName === 'TR') { drocatRowsRebuiltAt = Date.now(); return; }
      }
    }
  });
  drocatRowMo.observe(document.documentElement, { childList: true, subtree: true });
}
// Enter-to-next-row navigation for the editor tables: from the pressed input,
// focus the same column of the row below (no server round trip, so no remount
// risk; on the last row it is a no-op). The cell templates call this AFTER
// emitting their commit event. Shared with the Net-Viz edge editor via the
// window.drocatTableMove name.
document.addEventListener('drocat-enter-move', function (ev) {
  // Vue's production render proxy hides `window` from template expressions, so
  // the cell templates dispatch this DOM event instead of calling the helper.
  if (window.drocatTableMove) window.drocatTableMove(ev);
});
// Neuron chip cell: Enter with an EMPTY field moves to the next row's same
// column; Enter with text keeps adding chips (Quasar add-unique). Bound in the
// capture phase because QSelect consumes the Enter keydown at its input before
// a template (bubble) handler would see it — and deferred to a timeout so
// Quasar's own handling of the key is never disturbed.
if (!window.drocatNeuronEnterNav) {
  window.drocatNeuronEnterNav = true;
  document.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter') return;
    var inp = ev.target;
    if (!inp || !inp.closest || !inp.closest('.drocat-neuron-cell')) return;
    if (String(inp.value || '') !== '') return;
    var o = document.getElementById('drocat-suggest-overlay');
    if (o && o.style.display !== 'none') {
      var rows = o.querySelectorAll('.drocat-suggest-item');
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].classList.contains('drocat-suggest-active')) return; // pick
      }
    }
    setTimeout(function () {
      var live = document.activeElement;
      if (live && live !== document.body && live.closest && live.closest('.drocat-neuron-cell')) {
        // only move when the field is still empty and focus stayed put
        if (String(live.value || '') === '') window.drocatTableMove({ target: live });
      }
    }, 0);
  }, true);
}
if (!window.drocatTableMove) {
  window.drocatTableMove = function (ev) {
    var inp = ev && ev.target;
    if (!inp || !inp.closest) return;
    var td = inp.closest('td');
    var tr = td ? td.closest('tr') : null;
    if (!td || !tr || !tr.parentElement) return;
    var ci = Array.prototype.indexOf.call(tr.children, td);
    var next = tr.nextElementSibling;
    if (!next) return;
    var td2 = next.children[ci];
    var inp2 = td2 && td2.querySelector('input, textarea');
    if (!inp2) return;
    inp2.focus();
    if (inp2.setSelectionRange) {
      try {
        var len = (inp2.value || '').length;
        inp2.setSelectionRange(len, len);
      } catch (e) {}
    }
  };
}
"""


# Generic body slot: it iterates over the active columns (``props.cols``) so the
# cells follow the table's column set. The layer cell is a bare numeric input
# (free integer layer numbers — gaps are legitimate; a numbering jump breaks
# the inter-layer synapse pair); every other colour column renders a picker
# button that opens the single-color picker.
_BODY_SLOT = r"""
<q-tr
  :props="props"
  :key="props.row.id"
  :data-row-id="props.row.id"
  :class="props.rowIndex % 2 === 0 ? 'drocat-edge-row-even' : 'drocat-edge-row-odd'
">
  <q-td class="drocat-edge-select-cell">
    <q-checkbox v-model="props.selected" dense />
  </q-td>
  <q-td v-for="col in props.cols" :key="col.name" :props="props"
    :class="['drocat-edge-cell', col.classes || '', col.name !== 'color' ? 'drocat-edge-divider' : '']">
    <template v-if="col.name === 'layer'">
      <q-input v-model="props.row.layer" dense borderless hide-bottom-space
        type="number" step="1" placeholder="Layer"
        @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: 'layer', value: $event })"
        @blur="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'layer', value: props.row.layer })"
        @keydown.enter="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'layer', value: props.row.layer }); (function(i){var d=i.ownerDocument,e=d.createEvent('Event');e.initEvent('drocat-enter-move',true,false);i.dispatchEvent(e);})($event.target)"
        @keydown.tab="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'layer', value: props.row.layer })" />
    </template>
    <template v-else-if="col.name === 'neuron'">
      <div class="drocat-neuron-cell" :id="'neuron-cell-' + props.row.id">
        <q-select v-model="props.row.neurons" :options="props.row.neuron_options"
          multiple use-chips use-input new-value-mode="add-unique"
          hide-dropdown-icon popup-content-class="drocat-native-popup-hidden"
          dense borderless hide-bottom-space
          placeholder="Neuron"
          @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: 'neuron', value: $event })"
          @focus="$parent.$emit('layer-cell-focus', { id: props.row.id })"
          @input-value="(v) => $parent.$emit('layer-cell-suggest', { id: props.row.id, text: v })"
          @blur="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'neuron', value: props.row.neurons })"
          @keydown.tab="$parent.$emit('layer-cell-commit', { id: props.row.id, field: 'neuron', value: props.row.neurons })" />
      </div>
    </template>
    <template v-else>
      <div class="row items-center no-wrap gap-1 drocat-color-cell">
        <q-btn
          :icon="props.row[col.name] ? null : 'palette'"
          :round="!props.row[col.name]"
          :style="props.row[col.name] ? { backgroundColor: props.row[col.name] } : {}"
          flat dense size="xs"
          :class="['drocat-color-cell-picker', { 'drocat-color-cell-picker-set': !!props.row[col.name] }]"
          @click.stop="$parent.$emit('layer-color-pick', { id: props.row.id, field: col.name })"
          title="Pick color"
        />
        <q-input v-model="props.row[col.name]" dense borderless hide-bottom-space
          placeholder="(auto)"
          @update:model-value="$parent.$emit('layer-cell-change', { id: props.row.id, field: col.name, value: $event })"
          @blur="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })"
          @keydown.enter="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] }); (function(i){var d=i.ownerDocument,e=d.createEvent('Event');e.initEvent('drocat-enter-move',true,false);i.dispatchEvent(e);})($event.target)"
          @keydown.tab="$parent.$emit('layer-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })" />
      </div>
    </template>
  </q-td>
</q-tr>
"""


class LayerStyleEditorHandle:
    """State + actions of one editor card; usable from tests without JS."""

    def __init__(
        self,
        export_dir_provider: Optional[Callable[[], str]] = None,
        dataset_provider: Optional[Callable[[], object]] = None,
        search_columns_provider: Optional[Callable[[], str]] = None,
    ):
        self.rows: List[dict] = self._scaffold_rows()
        self.current_name: str = ""
        self.export_dir_provider = export_dir_provider
        self._dataset_provider = dataset_provider or (lambda: "")
        self._search_columns_provider = search_columns_provider or (lambda: "auto")
        # NiceGUI elements, assigned while the card is built.
        self.name_input: Optional[ui.input] = None
        self.table: Optional[ui.table] = None
        self.table_container: Optional[ui.element] = None
        self.status_label: Optional[ui.label] = None
        self.validation_panel: Optional[ui.element] = None
        self.validation_label: Optional[ui.label] = None
        self.available_neurons_link: Optional[ui.element] = None
        self._selected_ids: List[int] = []
        # Auto-save debounce deadline (polled by the one tick timer from render)
        # and that tick timer itself, created once during the card build.
        self._autosave_due: Optional[float] = None
        self._autosave_tick = None
        self.expansion: Optional[ui.expansion] = None
        self._transient_csv_path: Optional[str] = None
        self.synapse_mode: str = "synapse"
        self._pick_popup: Optional[object] = None
        self._pending_pick: Optional[dict] = None
        self._available_next_layer: Optional[int] = None
        self._available_query_values: List[str] = []
        # In-table neuron-cell auto-suggestion is shown in a plain positioned overlay
        # div (not a Quasar menu, so it never blurs the cell or clears typed text).
        # The overlay DIV ITSELF is a single static NiceGUI element; its suggestion
        # items are rendered on the CLIENT via ``run_javascript`` (never as NiceGUI
        # elements). Creating NiceGUI elements for the items during typing would
        # re-render the q-table body and remount the focused q-select, wiping the
        # typed text — so the items are pure DOM and committed through a single
        # ``pick`` listener emitted from JS. ``_suggest_suppress`` is armed on a
        # commit/blur; the re-render's synthetic focus/reset is ignored only for a
        # short window (``_suggest_suppress_until``), after which a genuine
        # refocus/typing clears it and re-opens the overlay.
        self._suggest_overlay: Optional[ui.element] = None
        self._suggest_row: Optional[int] = None
        self._suggest_suppress: bool = False
        self._suggest_suppress_until: float = 0.0
        # Post-pick hold: a refreshed table can emit a synthetic blur right after a
        # suggestion pick; within this window on_inline_commit keeps the overlay
        # open, so a pick never closes the list (matching the query box).
        self._suggest_keep_open_until: float = 0.0
        # The query whose rows the overlay is currently holding, with the rows
        # themselves, mirroring the shared query box: a pick re-renders the same
        # rows so the next entry can be added without retyping. Any close drops it.
        self._suggest_query: Optional[str] = None
        self._suggest_pool: List[tuple] = []
        # Whether the overlay is showing right now, and whether the last thing
        # the user did was dismiss it with a click/Escape (the next such press
        # brings the history list back instead of doing nothing).
        self._suggest_visible: bool = False
        self._suggest_toggled_off: bool = False
        # A table refresh remounts the cells, and the remounted q-select re-emits
        # an empty ``input-value``. While a list is on screen that reset must not
        # be read as "the user cleared the field", or it would swap the held query
        # for the history list right after a pick.
        self._suggest_ignore_reset_until: float = 0.0
        self._suggest_pick_listener_id: Optional[str] = None
        self._suggest_remove_listener_id: Optional[str] = None
        self._suggest_toggle_listener_id: Optional[str] = None
        # Value -> searched column (type/instance) cache for history category tags.
        self._history_cat_lookup: dict = {}
        self._history_cat_key: Optional[str] = None
        # After a programmatic chip commit, the re-rendered q-select re-emits
        # ``@update:model-value`` with a possibly-stale chip list; ignore those
        # for a short window so they cannot overwrite the authoritative rows.
        self._suppress_neuron_value_until: float = 0.0
        # While the pick's simulated chip entry is in flight, the cell field's
        # value changes are ours (the picked name, then the entry clearing the
        # field), not the user's typing: ``on_neuron_suggest`` ignores them so
        # the held list is not swapped for a fresh query mid-pick.
        self._simulate_until: float = 0.0
        # Exact signatures of the values the chip entry typed into the field.
        # On a slow socket the synthetic events land seconds after the pick,
        # when the coarse window has lapsed; an event whose text exactly equals
        # a simulated value is still ours (a user typing produces incremental
        # prefixes, never the full name in one event).
        self._simulated_values: dict = {}

    def _empty_logical_row(self) -> dict:
        """One empty scaffolding row (no layer / neuron / colour)."""
        return {
            "layer": "",
            "neurons": [],
            "color": "",
            "synapse_color": "",
            "pre_synaptic_color": "",
            "post_synaptic_color": "",
        }

    def _scaffold_rows(self) -> List[dict]:
        """Seed the editor with empty rows the user can type straight into."""
        return [self._empty_logical_row() for _ in range(3)]

    def _ensure_scaffolding(self) -> None:
        """Keep at least 3 rows present so the table always has empty cells."""
        while len(self.rows) < 3:
            self.rows.append(self._empty_logical_row())

    def _dataset_value(self) -> str:
        """Resolve the current Skeleton dataset for viewer/suggestions."""
        try:
            value = self._dataset_provider()
        except Exception:
            return ""
        if isinstance(value, (list, tuple, set)):
            value = next(iter(value), "")
        return str(value or "").strip()

    def _suggestions_enabled(self) -> bool:
        """Settings toggle: live type-ahead suggestions are on."""
        from ..config import get_auto_suggest_enabled
        return bool(get_auto_suggest_enabled())

    def _history_enabled(self) -> bool:
        """Settings toggle: the Recent/Frequent history list is on."""
        from ..config import get_show_history_enabled
        return bool(get_show_history_enabled())

    def _suggest_neurons(self, text: str):
        """Return the shared dataset-aware suggestion list for the add field."""
        dataset = self._dataset_value()
        if not dataset:
            return []
        try:
            scope = self._search_columns_provider()
        except Exception:
            scope = "auto"
        return dataset_suggestions(
            text,
            dataset,
            str(scope or "auto"),
            limit=None,
        )

    # ------------------------------------------------------------------ rows
    def _row_dicts(self) -> List[dict]:
        # Each cell keeps its ``neurons`` chip list and a stable id used as the
        # table row_key. The layer cell is a bare numeric input (no option list
        # to carry); suggestions are shown in the separate overlay instead of
        # the native dropdown, so typing never interrupts the q-select.
        row_dicts = []
        for i, row in enumerate(self.rows):
            row_dicts.append({
                **row, "id": i,
                "neuron_options": list(dict.fromkeys(list(row.get("neurons") or []))),
            })
        return row_dicts

    def refresh_table(self, *, preserve_selection: bool = False) -> None:
        """Refresh the table while keeping Python and QTable selection in sync."""
        row_dicts = self._row_dicts()
        valid_ids = {
            int(row["id"])
            for row in row_dicts
            if isinstance(row.get("id"), int)
        }
        if preserve_selection:
            self._selected_ids = [
                idx for idx in self._selected_ids if idx in valid_ids
            ]
        else:
            self._selected_ids = []

        if self.table is not None:
            self.table.rows = row_dicts
            self.table.selected = [
                row for row in row_dicts if row["id"] in self._selected_ids
            ]
            self.table.update()
        # The remount this triggers reports an empty cell input; while a list is
        # on screen that reset is not a user action, so it is ignored briefly.
        if self._suggest_visible:
            self._suggest_ignore_reset_until = time.time() + 0.6

    def set_rows(self, rows: List[dict], name: Optional[str] = None) -> None:
        # Loading real rows replaces the working set as-is (no scaffolding), so a
        # CSV round-trips to exactly its rows; the fresh-editor 3 empty rows are
        # seeded only in ``__init__`` (and restored by ``delete_selected``).
        self.rows = layer_style_store.logical_normalize(rows)
        self._available_next_layer = None
        if name is not None:
            self.current_name = name
            if self.name_input is not None:
                self.name_input.value = name
        self.refresh_table()

    def set_synapse_mode(self, mode: str) -> None:
        """Switch to a synapse mode (updates table columns + CSV format)."""
        self.synapse_mode = mode if mode in layer_style_store.MODE_COLUMNS else "synapse"
        if self.table is not None:
            self.table.columns = _columns_for_mode(self.synapse_mode)
            self.table.update()
        self.refresh_table()

    def render(self) -> None:
        """Create the table once; the columns are updated in place on mode change."""
        if self.table_container is None:
            return
        self.table_container.clear()
        # Per-column widths are controlled by these CSS variables (the header
        # resizers update them drag-to-resize). Column definitions reference them as
        # ``width:var(--wc-<name>)``. The defaults are deliberately modest so the
        # whole table (up to 6 columns in pre-post mode) fits inside the card; the
        # checkbox/layer stay capped narrow while Neuron/colors take the rest.
        _col_width_vars = (
            "--wc-select:44px; --wc-layer:68px; --wc-neuron:400px; --wc-color:280px; "
            "--wc-synapse_color:280px; --wc-pre_synaptic_color:280px; "
            "--wc-post_synaptic_color:280px"
        )
        with self.table_container:
            self.table = ui.table(
                columns=_columns_for_mode(self.synapse_mode),
                rows=[],
                row_key="id",
                selection="multiple",
                on_select=self.on_select,
            ).classes("w-full drocat-edge-table").props("dense flat bordered").style(_col_width_vars)
            self.table.add_slot("header", _TABLE_HEADER_SLOT)
            self.table.add_slot("body", _BODY_SLOT)
            self.table.on("layer-cell-change", self.on_inline_edit)
            self.table.on("layer-cell-commit", self.on_inline_commit)
            self.table.on("layer-color-pick", self.on_color_pick)
            self.table.on("layer-cell-focus", self.on_neuron_focus)
            self.table.on("layer-cell-suggest", self.on_neuron_suggest)
            # The `table_container.clear()` above deleted the previous
            # timer element (NiceGUI cancels timers on element deletion),
            # so recreate it unconditionally — an `is None` guard would
            # keep the stale, cancelled Timer object and leave a re-rendered
            # editor with a dead debounce poller.
            self._autosave_tick = ui.timer(0.25, self._autosave_poll)
        self.refresh_table()

    def _update_validation(self) -> list:
        """Render the persistent in-page validation panel and return errors."""
        errors = layer_style_store.validate_rows(self.rows)
        if self.validation_panel is None or self.validation_label is None:
            return errors
        if errors:
            self.validation_label.text = "Layer editor errors:\n" + "\n".join(
                f"• {error}" for error in errors
            )
            self.validation_panel.set_visibility(True)
        else:
            self.validation_label.text = ""
            self.validation_panel.set_visibility(False)
        self.validation_label.update()
        return errors

    def load_csv_path(self, path: str) -> bool:
        """Load rows from a CSV file into the table; returns success."""
        try:
            from pathlib import Path
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return False
        return self.load_csv_text(text)

    def load_csv_text(self, text: str) -> bool:
        """Load rows from CSV *text* into the table; returns success."""
        try:
            self.set_rows(layer_style_store.load_rows_from_csv_text(text))
        except Exception:
            return False
        if self.status_label is not None:
            self.status_label.text = f"Loaded {len(self.rows)} rows from CSV"
        return True

    # ------------------------------------------------------ available neurons
    def begin_available_batch(self) -> None:
        """Start a fresh viewer-selection batch."""
        self._available_next_layer = None
        self._available_query_values = []

    def available_query_values(self) -> List[str]:
        """Return the current selection mirrored by the available-neuron viewer."""
        return list(self._available_query_values)

    def apply_available_neurons(self, values) -> int:
        """Commit the viewer's final selection as one row per entry.

        The layer-editor viewer defers its selection to panel close, so this is
        called once with the final "Selected" list. Because the table is not
        touched while the panel is open, deselecting a matched value simply means
        it never gets committed; existing rows are preserved and only genuinely
        new entries are appended (filling empty scaffolding rows first, then
        growing the table). Each new entry gets its OWN layer: the session
        starts at the table's next free number and every added entry takes the
        following one (entries skipped because they already exist in the table
        consume no number), so a five-entry selection lands on consecutive
        layers like 2, 3, 4, 5, 6.
        """
        cleaned = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if text and text not in seen:
                cleaned.append(text)
                seen.add(text)
        if not cleaned:
            self.begin_available_batch()
            return 0

        # The viewer sends its complete selection after every toggle. Keep that
        # selection separate from the table rows so the viewer can mirror it in
        # its own boxed preview while the table continues to append only new
        # entries.
        self._available_query_values = list(cleaned)

        if self._available_next_layer is None:
            self._available_next_layer = layer_style_store.next_layer_number(self.rows)

        existing = {
            neuron
            for row in self.rows
            for neuron in row.get("neurons", [])
            if str(neuron).strip()
        }
        # Fill the existing empty (scaffolding) rows first — one neuron per empty
        # row — then append new rows for any remaining neurons, so a selection
        # uses the rows already waiting on screen before growing the table.
        empty_indices = [
            i for i, row in enumerate(self.rows)
            if layer_style_store._logical_is_empty(row)
        ]
        added_ids = []
        added_layers = []
        fill_pos = 0
        for value in cleaned:
            if value in existing:
                continue
            layer_number = self._available_next_layer
            self._available_next_layer += 1
            new_row = layer_style_store.logical_normalize([{
                "layer": str(layer_number),
                "neurons": [value],
            }])[0]
            if fill_pos < len(empty_indices):
                idx = empty_indices[fill_pos]
                self.rows[idx] = new_row
                fill_pos += 1
                added_ids.append(idx)
            else:
                self.rows.append(new_row)
                added_ids.append(len(self.rows) - 1)
            existing.add(value)
            added_layers.append(layer_number)

        if not added_ids:
            return 0

        self._ensure_scaffolding()
        self._selected_ids = added_ids
        self.refresh_table(preserve_selection=True)
        if len(added_layers) == 1:
            self._update_status(f"Added 1 neuron to layer {added_layers[0]}")
        else:
            self._update_status(
                f"Added {len(added_layers)} neurons to layers "
                f"{added_layers[0]}–{added_layers[-1]}"
            )
        self.schedule_autosave()
        return len(added_ids)

    # ------------------------------------------------------------- selection
    def on_select(self, event) -> None:
        # The client click already updated the checkbox state; only record the
        # ids. Re-assigning ``table.selected`` here would push fresh row dicts
        # and remount every cell under the cursor (a refresh_table does that).
        self._selected_ids = [row.get("id") for row in getattr(event, "selection", []) or []]

    def on_inline_edit(self, event) -> None:
        """Update the in-memory logical row from a table-cell change.

        The body slot keeps the live QSelect/QInput model on the client and sends
        only this small payload to Python. On every keystroke we mutate just the
        row model — never the table rows, validation panel or autosave timer — so
        the active input keeps its cursor and focus. Validation and auto-save are
        deferred to ``on_inline_commit`` (blur / Enter / Tab; Enter also moves
        the focus to the next row's same column) and to any layer edit.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in layer_style_store.LAYER_STYLE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return

        value = args.get("value")
        if field == "neuron":
            # A programmatic commit re-renders the cell and the q-select re-emits
            # a (possibly stale) chip list; ignore it briefly so it cannot
            # overwrite the authoritative rows.
            if time.time() < self._suppress_neuron_value_until:
                return
            # The chip cell reports its whole list; a plain string (tests) is
            # treated as a single chip.
            if isinstance(value, str):
                neurons = [value] if value.strip() else []
            else:
                neurons = [
                    str(v).strip()
                    for v in (value or [])
                    if str(v).strip()
                ]
            changed = self.rows[row_id]["neurons"] != neurons
            self.rows[row_id]["neurons"] = neurons
            # Marks follow the chips: dropping one clears its tick while the
            # list is held, as it does in the query box.
            if self._suggest_visible and row_id == self._suggest_row:
                self._rehold_suggestions(row_id)
            if changed:
                # A chip added by typing + Enter lands here through the
                # model update, never through a commit event; without
                # arming the debounced save, closing the tab before a
                # blur/Tab loses the chip — the module's never-lose-edits
                # guarantee.
                self.schedule_autosave()
            return

        scalar = str(value or "").strip()
        self.rows[row_id][field] = scalar
        if field == "layer":
            # A pure model update: the layer cell has no option list to re-render
            # (discrete layer numbers are legitimate), and validation is
            # intentionally NOT run here — incomplete fields only surface when
            # the user runs/exports.
            self.schedule_autosave()

    def on_inline_commit(self, event) -> None:
        """Auto-save once focus leaves a cell (blur / Enter / Tab).

        Enter additionally moves the focus to the same column of the next row
        (client-side; the neuron chip cell moves only when its field is empty,
        so Enter with text keeps adding chips).

        Validation is deliberately left to run/export, so a half-typed row never
        flashes a red error while the user is still editing. Leaving the cell also
        dismisses the suggestion overlay — EXCEPT during a short post-pick window,
        when the table refresh can emit a synthetic blur; that must not hide the
        list that was just re-offered (issue: pick/remove should keep it open).
        """
        if time.time() >= self._suggest_keep_open_until:
            self._close_suggest_overlay()
        self.on_inline_edit(event)
        self.schedule_autosave()

    # ------------------------------------------------- neuron-cell suggestions
    def on_neuron_suggest(self, event) -> None:
        """Compute dataset-aware suggestions for the focused neuron cell.

        The suggestions are shown in the pointing overlay (``_suggest_overlay``)
        below the focused cell, so typing inside the cell keeps its focus and text
        (the overlay is a plain div, not a Quasar menu).
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        text = str(args.get("text") or "").strip()
        if not 0 <= row_id < len(self.rows):
            return
        # The pick's simulated chip entry moves the field through the picked
        # name and then clears it; neither is the user typing. The coarse
        # window covers fast sockets; the exact-match map covers slow ones,
        # where the synthetic events land after it has lapsed. The trailing
        # field clear follows within one round trip of the value event, so
        # swallowing the value arms the empty-reset window.
        now = time.time()
        self._simulated_values = {
            v: until for v, until in self._simulated_values.items() if until > now
        }
        if text and self._simulated_values.pop(text, None) is not None:
            self._suggest_ignore_reset_until = now + 3.0
            return
        if now < self._simulate_until:
            return
        # A commit/blur closes the overlay. Ignore the re-render's empty reset
        # ``input-value`` only while it arrives inside the suppression window; a
        # non-empty query (or a later empty reset) clears suppression and re-opens.
        if (
            self._suggest_suppress
            and not text
            and time.time() < self._suggest_suppress_until
        ):
            return
        # Same story for a refresh while a list is on screen: the remounted cell
        # reports an empty field, which is not the user clearing their query.
        if (
            not text
            and self._suggest_query
            and time.time() < self._suggest_ignore_reset_until
        ):
            return
        self._suggest_suppress = False
        self._show_neuron_suggestions(row_id, text)

    def on_neuron_focus(self, event) -> None:
        """Record the focused cell and open history for an empty field.

        A refocus of the cell that already owns a held query is not a request to
        re-open anything: it is the pick's own ``_refocus_neuron_cell`` coming
        back around, and re-rendering here would replace the held rows with the
        history list.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        if self._suggest_row is not None and self._suggest_row != row_id:
            # Genuine switch to a different cell: close the old overlay and let the
            # new cell open its own history/suggestions.
            self._close_suggest_overlay()
            self._suggest_suppress = False
        elif (
            self._suggest_visible
            and self._suggest_query
            and self._suggest_row == row_id
            and time.time() < self._suggest_keep_open_until
        ):
            # The cell that owns a held query is reporting focus again. That is
            # the refocus a pick performs after refreshing the table, and the
            # held rows are already on screen: re-opening here would replace
            # them with the history list and end the hold mid-run, which is the
            # one thing the whole sticky-list feature exists to prevent.
            return
        elif self._suggest_suppress:
            # Same cell re-focused. Only the re-render's synthetic focus (which
            # arrives inside the suppression window) is ignored; a genuine later
            # refocus clears suppression and re-opens the history overlay.
            if time.time() < self._suggest_suppress_until:
                return
            self._suggest_suppress = False
        self._suggest_row = row_id
        self._show_neuron_suggestions(row_id, "")

    def _show_neuron_suggestions(
        self, row_id: int, text: str, pool: Optional[List[tuple]] = None
    ) -> None:
        """Fill the non-focus-stealing overlay below the focused neuron cell.

        The overlay div is populated entirely on the CLIENT via ``run_javascript``
        (pure DOM, never NiceGUI elements) — creating NiceGUI items here would
        re-render the q-table body and remount the focused q-select, wiping typed
        text. Clicking an item commits through the registered ``pick`` listener.
        History rows carry a ``Recent`` header and a right-aligned category tag,
        matching the standard query box.

        ``text`` is also the query held for the next pick: a blank ``text`` offers
        the Recent list and holds nothing, while a query records its rows (``pool``
        reuses ones already computed) so several entries can be picked from one
        list without retyping. In either list, values that are already chips stay
        listed and are tinted and ticked, and clicking such a row takes its chip
        back out -- exactly as ``neuron_list_input`` does.
        """
        if self._suggest_overlay is None:
            return
        if text:
            # Type-ahead dataset suggestions, independently toggled in Settings.
            if not self._suggestions_enabled():
                self._close_suggest_overlay()
                return
            entries = list(pool if pool is not None else self._suggest_neurons(text))
            is_history = False
        else:
            # Query history, independently toggled in Settings.
            if not self._history_enabled():
                self._close_suggest_overlay()
                return
            entries = self._recent_neuron_history()
            is_history = True
        if not entries:
            self._close_suggest_overlay()
            return
        self._suggest_suppress = False
        self._suggest_toggled_off = False
        self._suggest_query = None if is_history else text
        self._suggest_pool = [] if is_history else entries
        # Which rows read as already-added is decided in suggestion_list, the
        # same helper the shared query box uses, so the two lists cannot drift.
        # Both lists mark: a Recent row that is also a chip deselects on click.
        items = marked_rows(entries, self.rows[row_id].get("neurons", []))
        js = (
            f"window.drocatSuggest && window.drocatSuggest.render({int(row_id)}, "
            f"{json.dumps(items)}, {json.dumps(is_history)});"
        )
        try:
            self.table.client.run_javascript(js)
            self._suggest_visible = True
        except Exception:
            self._close_suggest_overlay()

    def _on_suggest_pick(self, event) -> None:
        """Commit a clicked suggestion to the currently focused cell."""
        value = getattr(event, "args", None)
        self._commit_neuron_suggestion(self._suggest_row, value)

    def _on_suggest_remove(self, event) -> None:
        """Remove one history entry from the Recent overlay."""
        value = str(getattr(event, "args", None) or "").strip()
        if not value:
            return
        try:
            from ..history_store import remove as history_remove
            history_remove(value)
        except Exception:
            return
        if self._suggest_row is not None:
            self._show_neuron_suggestions(self._suggest_row, "")

    def _recent_neuron_history(self):
        """Recent + frequent neuron query history; each carries a category tag.

        History rows mirror the standard query box: the gray tag is the searched
        column (type/instance/etc.) resolved from the dataset pools rather than the
        literal ``history`` label.
        """
        try:
            from ..history_store import frequent, recent
            dataset = self._dataset_value()
            scope = [dataset] if dataset else None
            lookup = self._history_category_lookup(dataset)
            seen = set()
            entries = []
            for value in list(recent(datasets=scope)) + list(frequent(datasets=scope)):
                item = str(value).strip()
                if item and item not in seen:
                    seen.add(item)
                    entries.append((item, lookup.get(item, "")))
            return entries
        except Exception:
            return []

    def _history_category_lookup(self, dataset: str) -> dict:
        """Value -> searched column (type/instance) over the dataset pools, cached."""
        if self._history_cat_key == dataset:
            return self._history_cat_lookup
        lookup = {}
        if dataset:
            try:
                from ..type_suggestions import get_dataset_pools
                pools = get_dataset_pools(dataset)
                for column in ("type", "instance"):
                    for candidate, _ in pools.get(column, []):
                        lookup.setdefault(str(candidate), column)
            except Exception:
                pass
        self._history_cat_key = dataset
        self._history_cat_lookup = lookup
        return lookup

    def _close_suggest_overlay(self) -> None:
        # Closing (commit / blur / no matches / a dismissal click) ends the held
        # query, so the next opening offers the history list rather than the old
        # rows. Re-opening from the cell's reset events is suppressed only for a
        # short window; after it a genuine refocus or a non-empty query clears
        # suppression (see ``on_neuron_suggest``/``on_neuron_focus``). The
        # client-side renderer hides the overlay so no NiceGUI elements rebuild.
        self._suggest_suppress = True
        self._suggest_suppress_until = time.time() + 0.4
        self._suggest_visible = False
        self._suggest_query = None
        self._suggest_pool = []
        if self._suggest_overlay is not None:
            try:
                self._suggest_overlay.client.run_javascript(
                    "window.drocatSuggest && window.drocatSuggest.hide();"
                )
            except Exception:
                pass

    def _rehold_suggestions(self, row_id: int) -> None:
        """Re-render what the overlay showed before, so a pick never closes it.

        A held query keeps its own rows (from the cached pool, so a round of
        picks costs one provider call); a pick from the history list re-offers
        the history list.
        """
        held = self._suggest_query
        if held:
            self._show_neuron_suggestions(row_id, held, pool=self._suggest_pool)
        else:
            self._show_neuron_suggestions(row_id, "")

    def _commit_neuron_suggestion(self, row_id: int, value: str) -> None:
        """Add a clicked row's chip to the cell, or take it back out when marked.

        Mirrors the standard query box: the row model changes first, the list
        stays on the same query (its rows re-render with the new value ticked),
        so entries can be handled one after another without retyping, and
        clicking a ticked row removes that chip again -- the tick is a checkbox,
        not a receipt.

        An ADD never touches the table rows: the chip is entered through the
        cell's own q-select (a simulated type + Enter, the same public path a
        user's keypress takes), so no cell remounts, the caret and any
        in-flight typing elsewhere survive, and the field clears itself after
        the chip lands. Only a REMOVAL needs a table refresh to sync the
        shorter chip list, and its refocus uses the settle-retry JS.
        """
        if row_id is None or not 0 <= row_id < len(self.rows):
            return
        value = str(value or "").strip()
        if not value:
            return
        neurons = list(self.rows[row_id]["neurons"])
        if chip_is_marked(neurons, value):
            neurons = without_chip(neurons, value)
            removing = True
        else:
            neurons.append(value)
            removing = False
        self.rows[row_id]["neurons"] = neurons
        # Ignore the cell's re-emit for a moment so neither the simulated chip
        # entry nor a remount's stale echo can clobber the authoritative rows --
        # in either direction, since a stale list would put a removed chip
        # straight back.
        self._suppress_neuron_value_until = time.time() + 0.6
        # The chip entry (or, on removal, the refresh) produces a synthetic
        # field change; it must not be read as the user clearing their query.
        self._suggest_ignore_reset_until = time.time() + 0.6
        if not removing:
            self._simulate_until = time.time() + 0.6
        self.schedule_autosave()
        # The hold keeps a synthetic blur from closing the list mid-pick.
        self._suggest_keep_open_until = time.time() + 0.6
        if removing:
            self.refresh_table(preserve_selection=True)
            self._rehold_suggestions(row_id)
            self._refocus_neuron_cell(row_id)
        else:
            self._rehold_suggestions(row_id)
            self._simulate_chip_entry(row_id, value)

    def _simulate_chip_entry(self, row_id: int, value: str) -> None:
        """Type a picked name into the cell's own q-select and press Enter.

        The simulated keypress goes through Quasar's public entry path
        (``new-value-mode="add-unique"``): the chip renders, the field clears
        itself, and ``@update:model-value`` syncs the client copy -- all
        without replacing ``table.rows``, so no cell remounts and the focus
        stays where the user has it. ``__drocatSuggestSim`` keeps the overlay's
        own Enter navigation from treating the simulated keypress as a pick.
        """
        self._simulated_values[str(value)] = time.time() + 4.0
        try:
            self.table.client.run_javascript(
                "(function(){"
                f"var c=document.getElementById('neuron-cell-{int(row_id)}');"
                "if(!c)return;"
                "var inp=c.querySelector('.q-field__input')||c.querySelector('input');"
                "if(!inp)return;"
                "var ae=document.activeElement;"
                "if(!ae||ae===document.body||c.contains(ae)){inp.focus();}"
                "window.__drocatSuggestSim=true;"
                "try{"
                f"inp.value={json.dumps(str(value))};"
                "inp.dispatchEvent(new Event('input',{bubbles:true}));"
                "inp.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',"
                "keyCode:13,which:13,bubbles:true}));"
                "}finally{window.__drocatSuggestSim=false;}"
                "})()"
            )
        except Exception:
            pass

    def _on_suggest_toggle(self, _event) -> None:
        """A press on the focused cell, or Escape: close the list, or bring it back.

        The dismissal never re-opens in the same breath; the next such press
        offers the history list, so a held query cannot leak into a fresh round.
        The client reports what it had rendered, but ``_suggest_visible`` is the
        one the refresh cycle maintains, so that is what decides here.
        """
        row_id = self._suggest_row
        if row_id is None or not 0 <= row_id < len(self.rows):
            return
        if self._suggest_visible:
            self._close_suggest_overlay()
            self._suggest_toggled_off = True
            return
        if not self._suggest_toggled_off:
            return
        self._show_neuron_suggestions(row_id, "")

    def _refocus_neuron_cell(self, row_id: int) -> None:
        """Refocus a neuron cell's input after a pick, clearing the typed prefix.

        A removal pick's table refresh remounts the cells (every re-render does
        in this stack), which drops the focus and wipes the typed query — the
        refocus puts the caret back, and the clear here keeps a leftover prefix
        from committing as a second chip on blur. The
        whole routine is guarded inside the JS (where the live focus state is):
        if the user is already typing into some other input, the pick's
        round-trip must not yank the caret back to the picked cell. The refocus
        also retries for a short while: the refresh's re-render lands
        asynchronously, so a single focus() can land on the input that is about
        to be replaced, and the remount would drop the focus to <body>.
        """
        try:
            cell_id = int(row_id)
            self.table.client.run_javascript(
                "(function(){"
                f"var cell='neuron-cell-{cell_id}';"
                "var c=document.getElementById(cell);"
                "if(!c)return;"
                "var ae=document.activeElement;"
                "if(ae&&ae!==document.body&&!c.contains(ae))return;"
                "var deadline=Date.now()+1500;var cleared=false;"
                "function step(){"
                "var cc=document.getElementById(cell);"
                "if(!cc)return;"
                "var ae2=document.activeElement;"
                "if(ae2&&ae2!==document.body&&!cc.contains(ae2))return;"
                "var inp=cc.querySelector('.q-field__input')||cc.querySelector('input');"
                "if(!inp)return;"
                "if(!cleared){cleared=true;"
                "if(inp.value){inp.value='';"
                "inp.dispatchEvent(new Event('input',{bubbles:true}));}}"
                "inp.focus();"
                # Focus can land on the input that the refresh's async re-render
                # is about to replace, so settling is confirmed only after a
                # grace period: if the same input still holds focus, we are past
                # the remount; if it dropped, retry until the deadline.
                "setTimeout(function(){"
                "if(document.activeElement===inp)return;"
                "if(Date.now()<deadline){step();}"
                "},100);"
                "}"
                "setTimeout(step,150);"
                "})()"
            )
        except Exception:
            pass

    def on_color_pick(self, event) -> None:
        """Open the single-color picker popup for a color cell.

        The table body slot emits ``layer-color-pick`` with {id, field} when a
        cell's swatch is clicked; this opens the shared popup seeded with the
        current cell value and applies the committed color back to that row.
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in layer_style_store.LAYER_STYLE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return
        self._pending_pick = {"row_id": row_id, "field": field}
        if self._pick_popup is None:
            return
        initial = self.rows[row_id].get(field) or "#145cff"
        self._pick_popup.open(initial)

    def _apply_picked_color(self, value: str) -> None:
        """Apply a committed color from the popup back to the table cell."""
        pending = self._pending_pick
        self._pending_pick = None
        if not pending:
            return
        field = pending.get("field")
        row_id = pending.get("row_id")
        if row_id is None or not 0 <= row_id < len(self.rows):
            return
        self.rows[row_id][field] = value
        self.refresh_table(preserve_selection=True)
        self.schedule_autosave()

    # --------------------------------------------------------------- editing
    def add_empty_row(self) -> None:
        """Append an empty scaffolding row for direct in-table editing."""
        self.rows.append(self._empty_logical_row())
        self._selected_ids = [len(self.rows) - 1]
        self.refresh_table(preserve_selection=True)
        self.schedule_autosave()

    def delete_selected(self) -> None:
        if not self._selected_ids:
            _notify("Select rows to delete", type="warning")
            return
        for idx in sorted(set(self._selected_ids), reverse=True):
            if 0 <= idx < len(self.rows):
                del self.rows[idx]
        self._selected_ids = []
        self._ensure_scaffolding()
        self.refresh_table()
        self.schedule_autosave()

    # ------------------------------------------------------------- auto-save
    def _non_empty_row_count(self) -> int:
        """Number of rows holding a layer, neuron or colour (scaffolding excluded)."""
        return sum(
            1 for row in self.rows
            if not layer_style_store._logical_is_empty(row)
        )

    def schedule_autosave(self) -> None:
        """Debounce edits, then flush to disk.

        The deadline is kept in Python and polled by the editor's one tick
        timer (created once in ``render``): creating a one-shot ``ui.timer``
        per commit mounts a new element in a live slot and deleting it at
        flush time unmounts it again, and either patch can remount the table's
        cells — which is exactly the input interruption under editing.

        Auto-save is gated on a minimum number of filled rows so partial tables
        do not create throwaway drafts. A manual export still flushes explicitly
        regardless of the row count.
        """
        if self._non_empty_row_count() < AUTOSAVE_MIN_NON_EMPTY_ROWS:
            self._update_status(
                f"Auto-save needs {AUTOSAVE_MIN_NON_EMPTY_ROWS} filled rows"
            )
            return
        self._update_status("Editing… (auto-save pending)")
        self._autosave_due = time.time() + AUTOSAVE_DELAY

    def _autosave_poll(self) -> None:
        """Fire the debounced flush once the deadline passes (tick timer)."""
        if self._autosave_due is None or time.time() < self._autosave_due:
            return
        self._autosave_due = None
        self.flush_autosave()

    def flush_autosave(self) -> Optional[str]:
        """Write the current rows to the draft store; returns the CSV path."""
        self._autosave_due = None
        name = str(self.name_input.value or "").strip() if self.name_input else self.current_name
        if not name:
            self._update_status("Enter a draft name to enable auto-save")
            return None
        if self.current_name and name != self.current_name:
            layer_style_store.delete_draft(self.current_name)
        slug = layer_style_store.save_draft(name, self.rows, dirty=True)
        if slug is None:
            self._update_status("Auto-save failed (invalid name or disk error)")
            return None
        self.current_name = name
        self._update_status(f"Auto-saved {datetime.now().strftime('%H:%M:%S')} · pending export")
        return layer_style_store.draft_csv_path(name)

    # ---------------------------------------------------------------- export
    def export_csv(self) -> Optional[str]:
        """Download the current layer style and optionally copy it to the output dir."""
        name = self._draft_name()
        csv_path = self.flush_autosave() if name else None
        csv_text = layer_style_store.rows_to_csv_for_mode(self.rows, self.synapse_mode)
        errors = self._update_validation()
        if errors:
            _notify("CSV downloaded, but has validation errors: " + errors[0], type="warning")

        slug = layer_style_store.sanitize_name(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{slug}_layers" if slug else "layers"
        filename = f"{filename_base}_{timestamp}.csv"
        export_dir = self.export_dir_provider() if self.export_dir_provider else None
        target = None
        if export_dir:
            from pathlib import Path
            try:
                Path(export_dir).mkdir(parents=True, exist_ok=True)
                target = Path(export_dir) / filename
                target.write_text(csv_text, encoding="utf-8")
            except OSError as ex:
                _notify(f"Could not save a local copy: {ex}", type="warning")

        self._download_csv(csv_text, filename)
        if name and csv_path is not None:
            layer_style_store.mark_exported(self.current_name)
            self._update_status("Downloaded · no unsaved changes")
        else:
            self._update_status(f"Downloaded {filename}")
        _notify("Layer style CSV downloaded", type="positive")
        return str(target) if target else (csv_path or filename)

    # ------------------------------------------------------------- reporting
    def _update_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.text = text

    def _draft_name(self) -> str:
        return (
            str(self.name_input.value or "").strip()
            if self.name_input is not None
            else self.current_name
        )

    @property
    def transient_csv_path(self) -> Optional[str]:
        """Path of the unnamed run file, if one is currently staged."""
        return self._transient_csv_path

    def _download_csv(self, csv_text: str, filename: str) -> None:
        """Trigger a browser download without requiring a filesystem dialog."""
        payload = csv_text.encode("utf-8")
        try:
            if self.table is not None:
                self.table.client.download(payload, filename, "text/csv")
            else:
                ui.download(payload, filename, media_type="text/csv")
        except RuntimeError:
            # Direct handle calls in tests or scripts may not have an active
            # NiceGUI request context; the optional local copy still remains.
            pass

    def _write_transient_csv(self) -> Optional[str]:
        self.cleanup_transient_csv()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix="drocat_layer_style_",
                suffix=".csv",
                delete=False,
            ) as handle:
                handle.write(layer_style_store.rows_to_csv_for_mode(self.rows, self.synapse_mode))
                self._transient_csv_path = handle.name
        except OSError as ex:
            self._update_status(f"Could not prepare layer list: {ex}")
            return None
        self._update_status("Ready to run · draft name is optional")
        return self._transient_csv_path

    def cleanup_transient_csv(self) -> Optional[str]:
        """Remove the temporary run CSV, returning its former path."""
        path = self._transient_csv_path
        self._transient_csv_path = None
        if path:
            try:
                from pathlib import Path
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        return path

    def runnable_csv_path(self) -> Optional[str]:
        """Return a VisualizeSkeleton-ready CSV; a draft name is optional for runs."""
        errors = self._update_validation()
        if errors:
            self._update_status("Fix the layer editor errors before running")
            return None
        complete = layer_style_store.complete_rows(self.rows)
        if not complete:
            return None
        return self.flush_autosave() if self._draft_name() else self._write_transient_csv()


def layer_style_editor(
    export_dir_provider: Optional[Callable[[], str]] = None,
    dataset_provider: Optional[Callable[[], object]] = None,
    search_columns_provider: Optional[Callable[[], str]] = None,
    card_id: str = "card-skeleton-layer-style-editor",
    on_expand: Optional[Callable[[], None]] = None,
) -> LayerStyleEditorHandle:
    """Build the collapsed editor panel and return its handle."""
    handle = LayerStyleEditorHandle(
        export_dir_provider,
        dataset_provider=dataset_provider,
        search_columns_provider=search_columns_provider,
    )

    def on_panel_change(event) -> None:
        if getattr(event, "value", False) and on_expand:
            on_expand()

    # Auto-suggestion overlay for the in-table neuron chip cells: a plain positioned
    # div (not a Quasar menu) that shows server-computed suggestions below the
    # focused cell without stealing its focus or clearing text. It is created as a
    # SIBLING of the expansion panel (not inside it) so that populating it never
    # re-renders the expansion's content slot (the table), which would otherwise
    # remount the focused q-select and wipe any typed text.
    handle._suggest_overlay = ui.element("div").props(
        'id="drocat-suggest-overlay"'
    ).classes("drocat-suggest-overlay").style("display: none;")
    # Register the commit listener (a clicked suggestion commits to the focused cell)
    # and capture its id so the client JS can emit directly over the socket.
    handle._suggest_overlay.on("pick", handle._on_suggest_pick)
    for _listener in handle._suggest_overlay._event_listeners.values():
        if _listener.type == "pick":
            handle._suggest_pick_listener_id = _listener.id
            break
    # History-row removal uses its own listener so an 'x' click never commits a pick.
    handle._suggest_overlay.on("remove", handle._on_suggest_remove)
    for _listener in handle._suggest_overlay._event_listeners.values():
        if _listener.type == "remove":
            handle._suggest_remove_listener_id = _listener.id
            break
    # A press on the focused cell / Escape: the click-to-toggle the query box has.
    handle._suggest_overlay.on("toggle", handle._on_suggest_toggle)
    for _listener in handle._suggest_overlay._event_listeners.values():
        if _listener.type == "toggle":
            handle._suggest_toggle_listener_id = _listener.id
            break
    _suggest_js = (
        _SUGGESTION_JS.replace("__OVERLAY_ID__", str(handle._suggest_overlay.id))
        .replace("__MARKED_TITLE_JSON__", json.dumps(MARKED_ROW_TITLE))
        .replace("__PICK_LID__", str(handle._suggest_pick_listener_id))
        .replace("__REMOVE_LID__", str(handle._suggest_remove_listener_id))
        .replace("__TOGGLE_LID__", str(handle._suggest_toggle_listener_id))
    )
    ui.add_head_html(f"<script>{_suggest_js}</script>")

    with ui.expansion(
        "Advanced Layer Editor",
        icon="edit_note",
        value=False,
        on_value_change=on_panel_change,
    ).classes("w-full drocat-edge-editor").props(f'id="{card_id}"') as panel:
        handle.expansion = panel
        ui.label(
            "Edit the Skeleton layers and per-neuron colors directly (the in-page "
            "equivalent of the custom-layer CSV). One row is one layer/color group; "
            "add several neurons as chips in a cell and they are written as separate "
            "rows on that layer at export. Layer numbers are free integers — gaps "
            "are legitimate, and a numbering jump breaks the inter-layer synapse "
            "pair (e.g. 1, 2, 4, 5 renders pairs 1→2 and 4→5). Changes are "
            "auto-saved, so edits survive a shutdown."
        ).classes("text-caption drocat-muted")

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            handle.name_input = ui.input(
                "Draft Name", placeholder="my_custom_layers",
            ).props('outlined dense').classes("grow min-w-[240px]")

        # The table is rebuilt per synapse mode (columns + body slot) inside this
        # container; the layer cell is a bare numeric input.
        handle.table_container = ui.column().classes("w-full")
        handle.render()

        # The header resizers call this client-side helper (injected once per page).
        try:
            if not getattr(handle.table.client,
                          "_drocat_layer_editor_js_added", False):
                ui.add_head_html(f"<script>{_COL_RESIZE_JS}</script>")
                handle.table.client._drocat_layer_editor_js_added = True
        except Exception:
            pass

        with ui.card().classes("w-full drocat-layer-validation").props(
            f'id="{card_id}-validation"'
        ) as validation_panel:
            with ui.row().classes("items-start gap-2"):
                ui.icon("error", color="negative").classes("mt-1")
                handle.validation_label = ui.label().classes(
                    "text-caption text-negative drocat-layer-validation-label"
                )
        handle.validation_panel = validation_panel
        validation_panel.set_visibility(False)

        # Shared single-color picker popup used by the colour cells' swatches.
        from .color_picker_popup import color_picker_popup

        handle._pick_popup = color_picker_popup(card_id=f"{card_id}-picker")
        handle._pick_popup.on_submit(handle._apply_picked_color)

        from .neuron_index_viewer import create_neuron_index_viewer_link

        handle.available_neurons_link = create_neuron_index_viewer_link(
            handle._dataset_value,
            label="See available neurons",
            on_open=handle.begin_available_batch,
            query_values_getter=handle.available_query_values,
            query_selection=handle.apply_available_neurons,
            query_label="Advanced layer selection",
            # Hold the selection in the panel ("Selected") and commit it only
            # when the dialog closes, so deselecting a matched value removes it.
            defer_apply=True,
        )

        with ui.row().classes(
            "w-full items-center gap-2 drocat-layer-add-actions"
        ):
            ui.button("Add Row", icon="add").props("dense").on_click(handle.add_empty_row)
            ui.button(
                "Delete Selected", icon="delete"
            ).props("dense outline").on_click(handle.delete_selected)

        with ui.row().classes("w-full items-center gap-3"):
            handle.status_label = ui.label("Empty draft").classes(
                "text-caption drocat-muted grow"
            )
            ui.button("Export CSV", icon="file_download").props("outline dense").on_click(
                handle.export_csv
            )
            with ui.button(icon="upload_file").props("outline dense").classes(
                "drocat-upload-trigger"
            ).tooltip("Load a custom-layer CSV into the table"):
                with ui.menu() as upload_menu:
                    ui.label("Load a custom-layer CSV").classes(
                        "text-caption drocat-muted px-3 pt-2"
                    )
                    ui.label(
                        "Columns: layer, neuron, color, synapse_color, "
                        "pre_synaptic_color, post_synaptic_color"
                    ).classes("text-caption drocat-muted px-3 pb-1")
                    ui.upload(
                        label="Choose CSV",
                        auto_upload=True,
                        on_upload=lambda e: _handle_csv_upload(handle, e),
                    ).props('accept=".csv" flat dense').classes("w-72")
                    upload_menu.update()

    return handle


async def _handle_csv_upload(handle: LayerStyleEditorHandle, event) -> None:
    """Load an uploaded CSV into the editor table."""
    from ..components.common import read_upload_event
    try:
        _filename, data = await read_upload_event(event)
        text = data.decode("utf-8")
    except Exception as ex:
        _notify(f"CSV upload failed: {ex}", type="negative")
        return
    if not handle.load_csv_text(text):
        _notify("Could not parse the uploaded CSV", type="negative")
