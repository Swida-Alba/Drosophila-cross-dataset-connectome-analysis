"""Interactive edge-list editor for the Net-Viz tab.

Mirrors the Advanced Layer Editor's editing model (Visualization > Skeleton):
inline-only table editing, empty scaffolding rows, a gated debounced
auto-save, a deferred validation panel, CSV import and drag-resizable
columns. The editor keeps its rows in a disk-backed draft (see
``ui.edge_list_store``): every change is auto-saved after a short debounce,
so an accidental UI/port shutdown never loses edits. A draft stays "dirty"
(pending export) until the user explicitly exports the CSV.
"""
from datetime import datetime
import tempfile
import time
from typing import Callable, List, Optional

from nicegui import ui

from .. import edge_list_store

AUTOSAVE_DELAY = 0.6  # seconds between last edit and disk flush
# Auto-save is only worth persisting once the table holds a real network; a
# handful of half-typed scaffolding rows would otherwise create noisy
# throwaway drafts. A manual export still flushes explicitly regardless.
AUTOSAVE_MIN_NON_EMPTY_ROWS = 5
# Empty rows the editor starts with (and never drops below on delete).
SCAFFOLD_ROW_COUNT = 3


def _notify(message: str, type: str = "info") -> None:
    """ui.notify that tolerates being called without an active UI slot
    (background tasks, direct calls from unit tests)."""
    try:
        ui.notify(message, type=type)
    except RuntimeError:
        pass


def _columns_for_mode(mode: str) -> list:
    """Table column definitions for a column set.

    ``basic`` shows the core edge columns; ``full`` adds the node-group
    columns and the hover-info columns VisualizePath reads from the expanded
    edge-list CSV (``{key:val; ...}`` cells).
    """
    def column(name: str, label: str, *, sortable: bool = False,
               align: str = "left") -> dict:
        # Column widths come from a per-column CSS variable (``--wc-<name>``,
        # spaces mapped to underscores so the name is a valid CSS identifier)
        # so the user can drag the header resizer to resize the column
        # interactively. The width is applied to header and body cells alike.
        var = name.replace(" ", "_")
        return {
            "name": name,
            "label": label,
            "field": name,
            "align": align,
            "sortable": sortable,
            "classes": f"drocat-{var.replace('_', '-')}-column",
            "headerClasses": f"drocat-{var.replace('_', '-')}-column",
            "style": f"width:var(--wc-{var})",
            "headerStyle": f"width:var(--wc-{var})",
        }

    cols = [
        column("source", "Source", sortable=True),
        column("target", "Target", sortable=True),
        column("weight", "Weight", sortable=True, align="right"),
        column("color", "Color (optional)"),
    ]
    if mode == "full":
        cols += [
            column("source_group", "Source group"),
            column("target_group", "Target group"),
            column("edge info", "Edge info"),
            column("source info", "Source info"),
            column("target info", "Target info"),
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
    :class="[col.headerClasses || '', { 'drocat-edge-divider': props.cols[props.cols.length - 1].name !== col.name }]"
  >
    {{ col.label }}
    <span class="drocat-col-resizer" :data-col="col.name.split(' ').join('_')"></span>
  </q-th>
</q-tr>
"""


# Drag-to-resize helper for the table headers. It updates ``--wc-<col>`` on the
# table element (driving each column's width), purely client-side so resizing never
# rebuilds the table or steals focus from a cell being edited. Listeners are
# attached to ``document`` (not ``window``) so a real mouse drag keeps firing even
# when the pointer travels outside the header; pointer events cover both mice and
# trackpads. Shared with the Advanced Layer Editor (injected once per client).
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
// Focus continuity for the editor table: in this NiceGUI/Quasar stack EVERY
// table re-render rebuilds all body rows (rows prop, a selection toggle, any
// element update landing in the same render cycle), which drops the focus and
// the caret mid-edit. When a blur inside the table is immediately followed by
// row removals AND the focus fell to <body> (i.e. the user did not go
// somewhere else), put the caret back into the same cell of the rebuilt row
// and restore the field content the rebuild wiped. Shared with the layer
// editor's table via the __drocatCellFocusGuard flag — whichever page loads
// first installs it, and it covers both tables.
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
// Enter-to-next-row navigation for the editor table: from the pressed input,
// focus the same column of the row below (no server round trip, so no remount
// risk; on the last row it is a no-op). The cell templates call this AFTER
// emitting their commit event. Shared with the Skeleton layer editor via the
// window.drocatTableMove name.
document.addEventListener('drocat-enter-move', function (ev) {
  // Vue's production render proxy hides `window` from template expressions, so
  // the cell templates dispatch this DOM event instead of calling the helper.
  if (window.drocatTableMove) window.drocatTableMove(ev);
});
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


_EDGE_BODY_SLOT = r"""
<q-tr
  :props="props"
  :key="props.row.id"
  :data-row-id="props.row.id"
  :class="props.rowIndex % 2 === 0 ? 'drocat-edge-row-even' : 'drocat-edge-row-odd'"
>
  <q-td class="drocat-edge-select-cell">
    <q-checkbox v-model="props.selected" dense />
  </q-td>
  <q-td v-for="col in props.cols" :key="col.name" :props="props"
    :class="['drocat-edge-cell', col.classes || '', props.cols[props.cols.length - 1].name !== col.name ? 'drocat-edge-divider' : '']">
    <template v-if="col.name === 'color'">
      <div class="row items-center no-wrap gap-1 drocat-color-cell">
        <q-btn
          :icon="props.row.color ? null : 'palette'"
          :round="!props.row.color"
          :style="props.row.color ? { backgroundColor: props.row.color } : {}"
          flat dense size="xs"
          :class="['drocat-color-cell-picker', { 'drocat-color-cell-picker-set': !!props.row.color }]"
          @click.stop="$parent.$emit('edge-color-pick', { id: props.row.id, field: 'color' })"
          title="Pick color"
        />
        <q-input v-model="props.row.color" dense borderless hide-bottom-space
          placeholder="(auto)"
          @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: 'color', value: $event })"
          @blur="$parent.$emit('edge-cell-commit', { id: props.row.id, field: 'color', value: props.row.color })"
          @keydown.enter="$parent.$emit('edge-cell-commit', { id: props.row.id, field: 'color', value: props.row.color }); (function(i){var d=i.ownerDocument,e=d.createEvent('Event');e.initEvent('drocat-enter-move',true,false);i.dispatchEvent(e);})($event.target)"
          @keydown.tab="$parent.$emit('edge-cell-commit', { id: props.row.id, field: 'color', value: props.row.color })" />
      </div>
    </template>
    <template v-else>
      <q-input v-model="props.row[col.name]" dense borderless hide-bottom-space
        :placeholder="col.name === 'source' ? 'Source' : col.name === 'target' ? 'Target' : col.name === 'weight' ? 'Weight' : (col.name === 'edge info' || col.name === 'source info' || col.name === 'target info') ? 'key: val; …' : 'Optional'"
        :inputmode="col.name === 'weight' ? 'decimal' : undefined"
        :input-class="col.name === 'weight' ? 'text-right' : undefined"
        @update:model-value="$parent.$emit('edge-cell-change', { id: props.row.id, field: col.name, value: $event })"
        @blur="$parent.$emit('edge-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })"
        @keydown.enter="$parent.$emit('edge-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] }); (function(i){var d=i.ownerDocument,e=d.createEvent('Event');e.initEvent('drocat-enter-move',true,false);i.dispatchEvent(e);})($event.target)"
        @keydown.tab="$parent.$emit('edge-cell-commit', { id: props.row.id, field: col.name, value: props.row[col.name] })" />
    </template>
  </q-td>
</q-tr>
"""


class EdgeListEditorHandle:
    """State + actions of one editor card; usable from tests without JS."""

    def __init__(self, export_dir_provider: Optional[Callable[[], str]] = None):
        self.rows: List[dict] = self._scaffold_rows()
        self.current_name: str = ""
        self.export_dir_provider = export_dir_provider
        self.columns_mode: str = "basic"
        # NiceGUI elements, assigned while the card is built.
        self.name_input: Optional[ui.input] = None
        self.table: Optional[ui.table] = None
        self.table_container: Optional[ui.column] = None
        self.status_label: Optional[ui.label] = None
        self.validation_panel: Optional[ui.card] = None
        self.validation_label: Optional[ui.label] = None
        self._selected_ids: List[int] = []
        # Auto-save debounce deadline (polled by the one tick timer from render)
        # and that tick timer itself, created once during the card build.
        self._autosave_due: Optional[float] = None
        self._autosave_tick = None
        self.expansion: Optional[ui.expansion] = None
        self._transient_csv_path: Optional[str] = None
        self._pending_pick: Optional[dict] = None
        self._pick_popup = None

    # ------------------------------------------------------------ scaffolding
    def _empty_row(self) -> dict:
        return {col: "" for col in edge_list_store.EDGE_COLUMNS}

    def _scaffold_rows(self) -> List[dict]:
        """Seed the editor with empty rows the user can type straight into."""
        return [self._empty_row() for _ in range(SCAFFOLD_ROW_COUNT)]

    def _ensure_scaffolding(self) -> None:
        """Keep enough rows present so the table always has empty cells."""
        while len(self.rows) < SCAFFOLD_ROW_COUNT:
            self.rows.append(self._empty_row())

    # ------------------------------------------------------------------ rows
    def _row_dicts(self) -> List[dict]:
        return [{**row, "id": i} for i, row in enumerate(self.rows)]

    def refresh_table(self, *, preserve_selection: bool = False) -> None:
        """Refresh the table while keeping Python and QTable selection in sync.

        QTable clears its visual selection when ``rows`` is replaced, so
        selection is cleared on data loads and explicitly restored only for
        edits that should keep the active rows selected.
        """
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

    def set_rows(self, rows: List[dict], name: Optional[str] = None) -> None:
        # Loading real rows replaces the working set as-is (no scaffolding), so
        # a CSV round-trips to exactly its rows; the fresh-editor scaffolding
        # rows are seeded only in ``__init__`` (and restored by delete).
        self.rows = edge_list_store.normalize_rows(rows)
        if name is not None:
            self.current_name = name
            if self.name_input is not None:
                self.name_input.value = name
        self.refresh_table()

    def set_columns_mode(self, mode: str) -> None:
        """Switch the visible column set ('basic' or 'full') in place."""
        self.columns_mode = mode if mode in ("basic", "full") else "basic"
        if self.table is not None:
            self.table.columns = _columns_for_mode(self.columns_mode)
            self.table.update()
        self.refresh_table(preserve_selection=True)

    def render(self) -> None:
        """Create the table once; the columns are updated in place on mode change."""
        if self.table_container is None:
            return
        self.table_container.clear()
        # Per-column widths are controlled by these CSS variables (the header
        # resizers update them drag-to-resize). Defaults keep the basic table
        # inside its card; the resizer overrides them per column.
        _col_width_vars = (
            "--wc-select:44px; --wc-source:180px; --wc-target:180px; "
            "--wc-weight:90px; --wc-color:150px; --wc-source_group:140px; "
            "--wc-target_group:140px; --wc-edge_info:230px; "
            "--wc-source_info:210px; --wc-target_info:210px"
        )
        with self.table_container:
            self.table = ui.table(
                columns=_columns_for_mode(self.columns_mode),
                rows=[],
                row_key="id",
                selection="multiple",
                on_select=self.on_select,
            ).classes("w-full drocat-edge-table").props(
                "dense flat bordered"
            ).style(_col_width_vars)
            self.table.add_slot("header", _TABLE_HEADER_SLOT)
            self.table.add_slot("body", _EDGE_BODY_SLOT)
            self.table.on("edge-cell-change", self.on_inline_edit)
            self.table.on("edge-cell-commit", self.on_inline_commit)
            self.table.on("edge-color-pick", self.on_color_pick)
            # The `table_container.clear()` above deleted the previous
            # timer element (NiceGUI cancels timers on element deletion),
            # so recreate it unconditionally — an `is None` guard would
            # keep the stale, cancelled Timer object and leave a re-rendered
            # editor with a dead debounce poller.
            self._autosave_tick = ui.timer(0.25, self._autosave_poll)
        self.refresh_table()

    def _update_validation(self) -> list:
        """Render the persistent in-page validation panel and return errors."""
        errors = edge_list_store.validate_rows(self.rows)
        if self.validation_panel is None or self.validation_label is None:
            return errors
        if errors:
            self.validation_label.text = "Edge list errors:\n" + "\n".join(
                f"• {error}" for error in errors
            )
            self.validation_panel.set_visibility(True)
        else:
            self.validation_label.text = ""
            self.validation_panel.set_visibility(False)
        self.validation_label.update()
        return errors

    def load_csv_text(self, text: str) -> bool:
        """Load rows from CSV *text* into the table; returns success."""
        try:
            self.set_rows(edge_list_store.load_rows_from_csv_text(text))
        except Exception:
            return False
        if self.status_label is not None:
            self.status_label.text = f"Loaded {len(self.rows)} rows from CSV"
        return True

    # ------------------------------------------------------------- selection
    def on_select(self, event) -> None:
        # The client click already updated the checkbox state; only record the
        # ids. Re-assigning ``table.selected`` here would push fresh row dicts
        # and remount every cell under the cursor (a refresh_table does that).
        self._selected_ids = [row.get("id") for row in getattr(event, "selection", []) or []]

    def on_inline_edit(self, event) -> None:
        """Update the in-memory row from a table-cell change.

        The body slot keeps the live QInput model on the client and sends only
        this small payload to Python. On every keystroke we mutate just the
        row model — never the table rows, validation panel or autosave timer —
        so the active input keeps its cursor and focus. Auto-save is deferred
        to ``on_inline_commit`` (blur / Enter / Tab; Enter also moves the
        focus to the next row's same column).
        """
        args = getattr(event, "args", event)
        if not isinstance(args, dict):
            return
        try:
            row_id = int(args.get("id"))
        except (TypeError, ValueError):
            return
        field = args.get("field")
        if field not in edge_list_store.EDGE_COLUMNS:
            return
        if not 0 <= row_id < len(self.rows):
            return

        value = str(args.get("value") or "").strip()
        self.rows[row_id][field] = value

    def on_inline_commit(self, event) -> None:
        """Auto-save once focus leaves a cell (blur / Enter / Tab).

        Enter additionally moves the focus to the same column of the next row
        (client-side, no server round trip).

        Validation is deliberately left to run/export, so a half-typed row
        never flashes a red error while the user is still editing.
        """
        self.on_inline_edit(event)
        self.schedule_autosave()

    def on_color_pick(self, event) -> None:
        """Open the single-color picker popup for a color cell.

        The table body slot emits ``edge-color-pick`` with {id, field} when a
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
        if field != "color":
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
        self.rows.append(self._empty_row())
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
        """Number of rows holding any value (scaffolding excluded)."""
        return sum(
            1 for row in self.rows if not edge_list_store.is_empty_row(row)
        )

    def schedule_autosave(self) -> None:
        """Debounce edits, then flush to disk.

        The deadline is kept in Python and polled by the editor's one tick
        timer (created once in ``render``): creating a one-shot ``ui.timer``
        per commit mounts a new element in a live slot and deleting it at
        flush time unmounts it again, and either patch can remount the table's
        cells — which is exactly the input interruption under editing.

        Auto-save is gated on a minimum number of filled rows so partial
        tables do not create throwaway drafts. A manual export still flushes
        explicitly regardless of the row count.
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
        # Rename semantics: leaving the current draft name saves under the new
        # name and removes the stale draft file.
        if self.current_name and name != self.current_name:
            edge_list_store.delete_draft(self.current_name)
        slug = edge_list_store.save_draft(name, self.rows, dirty=True)
        if slug is None:
            self._update_status("Auto-save failed (invalid name or disk error)")
            return None
        self.current_name = name
        self._update_status(f"Auto-saved {datetime.now().strftime('%H:%M:%S')} · pending export")
        return edge_list_store.draft_csv_path(name)

    # ---------------------------------------------------------------- export
    def export_csv(self) -> Optional[str]:
        """Download the current edge list and optionally copy it to the output dir."""
        name = self._draft_name()
        csv_path = self.flush_autosave() if name else None
        csv_text = edge_list_store.rows_to_csv(self.rows)
        errors = self._update_validation()
        if errors:
            _notify("CSV downloaded, but has validation errors: " + errors[0], type="warning")

        slug = edge_list_store.sanitize_name(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{slug}_edge_list" if slug else "edge_list"
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
            edge_list_store.mark_exported(self.current_name)
            self._update_status("Downloaded · no unsaved changes")
        else:
            self._update_status(f"Downloaded {filename}")
        _notify("Edge list CSV downloaded", type="positive")
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
                prefix="drocat_edge_list_",
                suffix=".csv",
                delete=False,
            ) as handle:
                handle.write(edge_list_store.rows_to_csv(self.rows))
                self._transient_csv_path = handle.name
        except OSError as ex:
            self._update_status(f"Could not prepare edge list: {ex}")
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

    def runnable_path_file(self) -> Optional[str]:
        """Return a PlotPath-ready CSV; a draft name is optional for runs."""
        errors = self._update_validation()
        if errors:
            self._update_status("Fix the edge list errors before running")
            return None
        complete = edge_list_store.complete_rows(self.rows)
        if not complete:
            return None
        return self.flush_autosave() if self._draft_name() else self._write_transient_csv()


def edge_list_editor(
    export_dir_provider: Optional[Callable[[], str]] = None,
    card_id: str = "card-net-viz-edge-editor",
    on_expand: Optional[Callable[[], None]] = None,
) -> EdgeListEditorHandle:
    """Build the collapsed editor panel and return its handle."""
    handle = EdgeListEditorHandle(export_dir_provider)

    def on_panel_change(event) -> None:
        if getattr(event, "value", False) and on_expand:
            on_expand()

    with ui.expansion(
        "Edge List Editor",
        icon="edit_note",
        value=False,
        on_value_change=on_panel_change,
    ).classes("w-full drocat-edge-editor").props(f'id="{card_id}"') as panel:
        handle.expansion = panel
        ui.label(
            "Edit the edge list directly (source → target with weight); use "
            "Add Row for an extra row. Columns switches to the group and "
            "hover-info columns VisualizePath understands. Changes are "
            "auto-saved, so edits survive an app/port shutdown; the draft "
            "stays marked as unsaved until you export it."
        ).classes("text-caption drocat-muted")

        # Authoring notice: complex enriched graphs (hover labels, custom
        # groups, extra metric columns) are easier to author in a local CSV
        # than in this table.
        with ui.row().classes("w-full items-start gap-2 flex-nowrap").props(
            'id="card-net-viz-enriched-notice"'
        ):
            ui.icon("tips_and_updates", color="primary").classes("mt-1")
            with ui.column().classes("gap-1"):
                ui.label(
                    "Tip: for complex, enriched graphs — per-edge hover-label "
                    "info, custom node groups, NT and metric columns — edit a "
                    "local CSV file with the extended columns and import it "
                    "with the upload button below, or run it directly through "
                    "the File upload source. The Auto Type Mapping network "
                    "views showcase these features."
                ).classes("text-caption drocat-muted")
                with ui.row().classes("gap-3 flex-wrap"):
                    ui.link(
                        "Enriched edge-list format",
                        "docs/ui_guides/network.html#enriched-edge-list",
                    ).classes("drocat-doc-link")
                    ui.link(
                        "Auto Type Mapping guide",
                        "docs/ui_guides/cross_dataset.html#type-mapping-panel",
                    ).classes("drocat-doc-link")

        with ui.row().classes("w-full items-end gap-2 flex-wrap"):
            handle.name_input = ui.input(
                "Draft Name", placeholder="my_custom_network",
            ).props('outlined dense').classes("grow min-w-[240px]")
            ui.select(
                {"basic": "Basic columns", "full": "All columns"},
                value="basic",
                label="Columns",
                on_change=lambda e: handle.set_columns_mode(e.value),
            ).props("outlined dense").classes("w-44")

        handle.table_container = ui.column().classes("w-full")
        handle.render()

        # The header resizers call this client-side helper (injected once per
        # page). This editor's copy also carries the focus guard and the
        # Enter-move machinery, so it must have its OWN per-client flag:
        # sharing one with the layer editor let whichever editor rendered
        # first silently skip the other's script (the inner window guards
        # dedupe the bindings, so double injection is safe).
        try:
            if not getattr(handle.table.client,
                           "_drocat_edge_editor_js_added", False):
                ui.add_head_html(f"<script>{_COL_RESIZE_JS}</script>")
                handle.table.client._drocat_edge_editor_js_added = True
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

        # Shared single-color picker popup used by the color cells' swatches.
        from .color_picker_popup import color_picker_popup

        handle._pick_popup = color_picker_popup(card_id=f"{card_id}-picker")
        handle._pick_popup.on_submit(handle._apply_picked_color)

        with ui.row().classes("w-full items-center gap-2 drocat-layer-add-actions"):
            ui.button("Add Row", icon="add").props("dense").on_click(
                handle.add_empty_row
            )
            ui.button("Delete Selected", icon="delete").props(
                "dense outline"
            ).on_click(handle.delete_selected)

        with ui.row().classes("w-full items-center gap-3"):
            handle.status_label = ui.label("Empty draft").classes(
                "text-caption drocat-muted grow"
            )
            ui.button("Export CSV", icon="file_download").props("outline dense").on_click(
                handle.export_csv
            )
            with ui.button(icon="upload_file").props("outline dense").classes(
                "drocat-upload-trigger"
            ).tooltip("Load an edge-list CSV into the table"):
                with ui.menu() as upload_menu:
                    ui.label("Load an edge-list CSV").classes(
                        "text-caption drocat-muted px-3 pt-2"
                    )
                    ui.label(
                        "Columns: source, target, weight, color, "
                        "source_group, target_group, edge info, "
                        "source info, target info"
                    ).classes("text-caption drocat-muted px-3 pb-1")
                    ui.upload(
                        label="Choose CSV",
                        auto_upload=True,
                        on_upload=lambda e: _handle_csv_upload(handle, e),
                    ).props('accept=".csv" flat dense').classes("w-72")
                    upload_menu.update()

    return handle


async def _handle_csv_upload(handle: EdgeListEditorHandle, event) -> None:
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
