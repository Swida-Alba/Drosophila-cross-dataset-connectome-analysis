"""Tests for the edge-list editor backend (auto-save draft store) and its
UI integration in the Net-Viz tab.

Covers:
- ui/edge_list_store.py: draft CRUD, atomic auto-save, dirty tracking,
  crash recovery, validation, and PlotPath-ready CSV layout.
- ui/components/edge_list_editor.py: editor state, add/delete/inline edit
  operations, debounced auto-save flush, and export.
- ui/tabs/visualization.py: Net-Viz source mode buttons and editor expansion wiring.
"""
import json
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ui.edge_list_store as store


# =============================================================================
# Store fixtures & helpers
# =============================================================================

@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the draft store at a temp directory for every test."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "edge_list_drafts")
    yield


ROWS = [
    {"source": "aMe12", "target": "aMe10", "weight": "128"},
    {"source": "aMe10", "target": "MBON01", "weight": "47"},
]


def meta(name):
    return store.get_meta(name)


# =============================================================================
# Store: naming & validation
# =============================================================================

class TestNaming:
    def test_sanitize_basic(self):
        assert store.sanitize_name("my network") == "my_network"

    def test_sanitize_special_chars(self):
        assert store.sanitize_name("a/b:c*d?.csv") == "a_b_c_d_csv"

    def test_sanitize_empty(self):
        assert store.sanitize_name("   ") == ""
        assert store.sanitize_name("///") == ""

    def test_sanitize_truncates(self):
        assert len(store.sanitize_name("x" * 300)) <= 80


class TestValidation:
    def test_valid_rows(self):
        assert store.validate_rows(ROWS) == []

    def test_missing_fields_reported(self):
        errors = store.validate_rows([{"source": "A", "target": "", "weight": ""}])
        assert any("missing target" in e for e in errors)
        assert any("missing weight" in e for e in errors)

    def test_non_numeric_weight(self):
        errors = store.validate_rows([{"source": "A", "target": "B", "weight": "many"}])
        assert any("not a number" in e for e in errors)

    def test_negative_weight(self):
        errors = store.validate_rows([{"source": "A", "target": "B", "weight": "-3"}])
        assert any(">= 0" in e for e in errors)

    def test_completely_empty_rows_ignored(self):
        assert store.validate_rows([{}, {"source": "", "target": "", "weight": ""}]) == []

    def test_normalize_strips_and_fills(self):
        rows = store.normalize_rows([{"source": " A ", "target": "B"}])
        assert rows[0]["source"] == "A"
        assert rows[0]["target"] == "B"
        assert rows[0]["weight"] == ""
        assert set(rows[0]) == set(store.EDGE_COLUMNS)

    def test_complete_rows(self):
        rows = ROWS + [{"source": "X", "target": "", "weight": "5"}]
        assert store.complete_rows(rows) == store.normalize_rows(ROWS)


# =============================================================================
# Store: save / load / metadata round-trip
# =============================================================================

class TestSaveLoad:
    def test_save_and_load_round_trip(self):
        slug = store.save_draft("net", ROWS)
        assert slug == "net"
        assert store.load_draft("net") == store.normalize_rows(ROWS)

    def test_order_preserved(self):
        rows = [{"source": f"N{i}", "target": f"N{i+1}", "weight": str(i)} for i in range(5)]
        store.save_draft("ordered", rows)
        assert [r["source"] for r in store.load_draft("ordered")] == [f"N{i}" for i in range(5)]

    def test_invalid_name_rejected(self):
        assert store.save_draft("///", ROWS) is None
        assert store.load_draft("///") is None

    def test_load_missing_returns_none(self):
        assert store.load_draft("ghost") is None

    def test_meta_created_dirty(self):
        store.save_draft("net", ROWS)
        m = meta("net")
        assert m["name"] == "net"
        assert m["dirty"] is True
        assert m["row_count"] == 2
        assert m["created_at"] and m["updated_at"]

    def test_meta_keeps_created_at_on_update(self):
        store.save_draft("net", ROWS)
        created = meta("net")["created_at"]
        store.save_draft("net", ROWS[:1])
        m = meta("net")
        assert m["created_at"] == created
        assert m["row_count"] == 1

    def test_overwrite_replaces_rows(self):
        store.save_draft("net", ROWS)
        store.save_draft("net", ROWS[:1])
        assert len(store.load_draft("net")) == 1

    def test_empty_draft_allowed(self):
        assert store.save_draft("empty", []) == "empty"
        assert store.load_draft("empty") == []
        assert meta("empty")["row_count"] == 0

    def test_csv_quoting_survives_commas_and_quotes(self):
        rows = [{"source": 'A,B', "target": 'say "hi"', "weight": "1"}]
        store.save_draft("quoted", rows)
        loaded = store.load_draft("quoted")
        assert loaded[0]["source"] == "A,B"
        assert loaded[0]["target"] == 'say "hi"'


class TestCsvLayout:
    """The draft CSV must be directly consumable by VisualizePath."""

    def test_columns_without_color(self):
        store.save_draft("net", ROWS)
        header = store.draft_csv_path("net")
        first_line = Path(header).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "source,target,weight"

    def test_color_column_only_when_used(self):
        rows = ROWS + [{"source": "P", "target": "Q", "weight": "1", "color": "#ff0000"}]
        store.save_draft("colored", rows)
        first_line = Path(store.draft_csv_path("colored")).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "source,target,weight,color"

    def test_pandas_edge_list_detection(self):
        """Exact column set {source, target, weight} is vispath's edge-list format."""
        import pandas as pd
        store.save_draft("net", ROWS)
        df = pd.read_csv(store.draft_csv_path("net"))
        assert set(df.columns) == {"source", "target", "weight"}
        assert {"source", "target", "weight"} in [
            {"source", "target", "weight"}, {"from", "to", "weight"}, {"pre", "post", "weight"},
        ]

    def test_extended_columns_only_when_used(self):
        """Group / hover-info columns are written only when a row uses them."""
        extended = ROWS + [{
            "source": "P", "target": "Q", "weight": "1",
            "source_group": "PAM", "target_group": "MBON",
            "edge info": "nt:ACH", "source info": "dataset:FAFB",
            "target info": "dataset:BANC",
        }]
        store.save_draft("extended", extended)
        first_line = Path(store.draft_csv_path("extended")).read_text(
            encoding="utf-8").splitlines()[0]
        # `color` stays absent: every optional column is written only when used.
        assert first_line == (
            "source,target,weight,source_group,target_group,"
            "edge info,source info,target info"
        )
        store.save_draft("basic", ROWS)
        header = Path(store.draft_csv_path("basic")).read_text(
            encoding="utf-8").splitlines()[0]
        assert header == "source,target,weight"

    def test_empty_scaffolding_rows_dropped_from_csv(self):
        text = store.rows_to_csv(ROWS + [{}])
        assert text.splitlines() == [
            "source,target,weight",
            "aMe12,aMe10,128",
            "aMe10,MBON01,47",
        ]
        store.save_draft("scaffolded", ROWS + [{}])
        assert len(store.load_draft("scaffolded")) == 2
        assert meta("scaffolded")["row_count"] == 2

    def test_load_rows_from_csv_text(self):
        rows = store.load_rows_from_csv_text(
            "source,target,weight,color,source_group\nA,B,1,#fff000,PAM\n"
        )
        assert rows[0]["source"] == "A"
        assert rows[0]["color"] == "#fff000"
        assert rows[0]["source_group"] == "PAM"
        assert set(rows[0]) == set(store.EDGE_COLUMNS)
        # Unknown columns are ignored; missing ones are filled empty.
        rows = store.load_rows_from_csv_text("source,target,weight,junk\nA,B,1,x\n")
        assert rows[0]["weight"] == "1"
        assert "junk" not in rows[0]

    def test_no_temp_files_left_behind(self):
        store.save_draft("net", ROWS)
        leftovers = list(store._store_dir.glob("*.tmp"))
        assert leftovers == []


# =============================================================================
# Store: dirty tracking & recovery
# =============================================================================

class TestDirtyTracking:
    def test_save_marks_dirty(self):
        store.save_draft("net", ROWS)
        assert meta("net")["dirty"] is True

    def test_mark_exported_clears_dirty(self):
        store.save_draft("net", ROWS)
        assert store.mark_exported("net") is True
        assert meta("net")["dirty"] is False

    def test_mark_dirty_again(self):
        store.save_draft("net", ROWS)
        store.mark_exported("net")
        assert store.mark_dirty("net") is True
        assert meta("net")["dirty"] is True

    def test_save_with_dirty_false(self):
        store.save_draft("net", ROWS, dirty=False)
        assert meta("net")["dirty"] is False

    def test_set_dirty_missing_draft(self):
        assert store.set_dirty("ghost", True) is False
        assert store.mark_exported("ghost") is False

    def test_pending_drafts_only_dirty(self):
        store.save_draft("dirty1", ROWS)
        store.save_draft("clean1", ROWS, dirty=False)
        store.save_draft("dirty2", ROWS)
        pending = store.pending_drafts()
        names = {m["name"] for m in pending}
        assert names == {"dirty1", "dirty2"}

    def test_list_drafts_newest_first(self):
        store.save_draft("old", ROWS)
        store.save_draft("new", ROWS)
        # Touch 'old' so it becomes the newest.
        store.save_draft("old", ROWS[:1])
        names = [m["name"] for m in store.list_drafts()]
        assert names[0] == "old"
        assert set(names) == {"old", "new"}


class TestCrashRecovery:
    """Simulate a previous session: files exist on disk, app restarts."""

    def test_recover_files_written_by_previous_session(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "session_draft.csv").write_text(
            "source,target,weight\nA,B,10\n", encoding="utf-8"
        )
        (draft_dir / "session_draft.meta.json").write_text(
            json.dumps({
                "name": "session draft", "slug": "session_draft",
                "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:01",
                "dirty": True, "row_count": 1,
            }),
            encoding="utf-8",
        )
        pending = store.pending_drafts()
        assert [m["name"] for m in pending] == ["session draft"]
        rows = store.load_draft("session draft")
        assert rows[0]["source"] == "A"
        assert rows[0]["target"] == "B"
        assert rows[0]["weight"] == "10"
        assert set(rows[0]) == set(store.EDGE_COLUMNS)

    def test_corrupt_meta_ignored(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "broken.csv").write_text("source,target,weight\n", encoding="utf-8")
        (draft_dir / "broken.meta.json").write_text("{not json", encoding="utf-8")
        assert store.list_drafts() == []
        assert store.pending_drafts() == []

    def test_orphan_meta_without_csv_ignored(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "orphan.meta.json").write_text(
            json.dumps({"name": "orphan", "slug": "orphan", "dirty": True, "row_count": 0}),
            encoding="utf-8",
        )
        assert store.list_drafts() == []

    def test_corrupt_csv_returns_none(self):
        draft_dir = store._store_dir
        draft_dir.mkdir(parents=True, exist_ok=True)
        (draft_dir / "bad.csv").write_bytes(b"\xff\xfe\x00broken")
        assert store.load_draft("bad") is None


# =============================================================================
# Store: delete & paths
# =============================================================================

class TestDeleteAndPaths:
    def test_delete_removes_both_files(self):
        store.save_draft("net", ROWS)
        assert store.delete_draft("net") is True
        assert store.load_draft("net") is None
        assert meta("net") is None
        assert store.draft_csv_path("net") is None

    def test_delete_missing_returns_false(self):
        assert store.delete_draft("ghost") is False

    def test_draft_csv_path(self):
        store.save_draft("net", ROWS)
        path = store.draft_csv_path("net")
        assert path and Path(path).exists()
        assert path.endswith("net.csv")

    def test_draft_csv_path_missing(self):
        assert store.draft_csv_path("ghost") is None


# =============================================================================
# UI component: editor handle behavior
# =============================================================================

from nicegui import Client
from nicegui.page import page


@pytest.fixture()
def store_patch_for_component(monkeypatch, tmp_path):
    """The component imports the store module; patch its store dir too."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "comp_drafts")
    return tmp_path / "comp_drafts"


def build_editor(store_dir, export_dir=None):
    from ui.components.edge_list_editor import edge_list_editor
    client = Client(page(f"/edge-editor-{uuid.uuid4().hex}"))
    with client:
        handle = edge_list_editor(export_dir_provider=lambda: str(export_dir) if export_dir else None)
    return client, handle


class TestEditorHandle:
    def test_card_elements_exist(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ids = [
            (getattr(el, "_props", None) or {}).get("id")
            for el in client.elements.values()
        ]
        assert "card-net-viz-edge-editor" in ids
        assert "card-net-viz-edge-editor-validation" in ids
        assert handle.table is not None and handle.status_label is not None
        assert handle.validation_panel is not None
        assert handle.validation_panel.visible is False
        # The editor seeds 3 empty scaffolding rows for direct in-table typing.
        assert len(handle.rows) == 3
        assert all(store.is_empty_row(row) for row in handle.rows)
        button_texts = [getattr(el, "text", "") for el in client.elements.values()]
        assert "Add Row" in button_texts
        assert "Add Edge" not in button_texts

    def test_add_empty_row_and_inline_edit_autosave(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "my draft"
        handle.add_empty_row()
        assert handle._selected_ids == [len(handle.rows) - 1]
        new_id = len(handle.rows) - 1
        for field, value in (("source", "A"), ("target", "B"), ("weight", "12")):
            handle.on_inline_commit(
                SimpleNamespace(args={"id": new_id, "field": field, "value": value})
            )
        handle.rows[new_id]["color"] = "#123456"
        # Debounced timer does not run in unit tests; an explicit flush also
        # bypasses the filled-rows gate.
        csv_path = handle.flush_autosave()
        assert csv_path and Path(csv_path).exists()
        rows = store.load_draft("my draft")
        assert rows == [{
            "source": "A", "target": "B", "weight": "12", "color": "#123456",
            "source_group": "", "target_group": "",
            "edge info": "", "source info": "", "target info": "",
        }]
        assert meta("my draft")["dirty"] is True
        assert "Auto-saved" in handle.status_label.text

    def test_inline_edit_updates_row_without_rebuilding_table(self, store_patch_for_component, monkeypatch):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        rebuilds = []
        monkeypatch.setattr(handle.table, "update", lambda *a, **k: rebuilds.append(1))
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "target", "value": " MBON02 "})
        )
        assert handle.rows[1]["target"] == "MBON02"
        # Keystroke edits mutate the row model only, never the table.
        assert rebuilds == []

    def test_delete_selected_rows(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS + [
            {"source": "X", "target": "Y", "weight": "9"},
            {"source": "Z", "target": "W", "weight": "8"},
        ])
        handle._selected_ids = [1]
        handle.delete_selected()
        assert len(handle.rows) == 3
        assert handle.rows[1]["source"] == "X"

    def test_delete_keeps_scaffolding_rows(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle._selected_ids = [0]
        handle.delete_selected()
        # Deleting below the scaffolding floor restores empty rows.
        assert len(handle.rows) == 3
        assert handle.rows[0]["source"] == "aMe10"
        assert all(store.is_empty_row(row) for row in handle.rows[1:])

    def test_add_empty_row_appends_and_selects(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.add_empty_row()
        assert len(handle.rows) == 4
        assert handle._selected_ids == [3]

    def test_refresh_keeps_python_and_table_selection_aligned(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle.on_select(SimpleNamespace(selection=[{**ROWS[1], "id": 1}]))
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "weight", "value": "99"})
        )
        assert handle._selected_ids == [1]
        assert [row["id"] for row in handle.table.selected] == [1]
        assert handle.rows[1]["weight"] == "99"

        # Loading/replacing rows clears both representations of selection.
        handle.set_rows(ROWS[:1])
        assert handle._selected_ids == []
        assert handle.table.selected == []

    def test_rename_deletes_previous_draft(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="old name")
        handle.flush_autosave()
        assert store.get_meta("old name") is not None
        handle.name_input.value = "new name"
        handle.flush_autosave()
        assert store.get_meta("old name") is None
        assert store.get_meta("new name") is not None
        assert handle.current_name == "new name"

    def test_flush_without_name_does_nothing(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        assert handle.flush_autosave() is None
        assert store.list_drafts() == []

    def test_autosave_gated_on_min_non_empty_rows(
        self, store_patch_for_component, monkeypatch
    ):
        """Auto-save only arms once the table holds >=5 filled rows.

        A handful of scaffolding rows must not create a throwaway draft: below
        the threshold schedule_autosave does not arm the debounce timer and
        surfaces a gate status instead, while the manual export path is
        unaffected.
        """
        from ui.components import edge_list_editor as editor_module
        from ui.components.edge_list_editor import AUTOSAVE_MIN_NON_EMPTY_ROWS

        client, handle = build_editor(store_patch_for_component)

        # Fresh editor starts with 3 empty scaffolding rows.
        assert handle._non_empty_row_count() == 0
        # Grow the table so it can hold up to the threshold.
        while len(handle.rows) < AUTOSAVE_MIN_NON_EMPTY_ROWS:
            handle.rows.append(handle._empty_row())

        def _filled(i):
            row = handle._empty_row()
            row.update({"source": f"N{i}", "target": f"M{i}", "weight": str(i + 1)})
            return row

        # 4 filled rows is still below the threshold.
        for i in range(AUTOSAVE_MIN_NON_EMPTY_ROWS - 1):
            handle.rows[i] = _filled(i)
        assert handle._non_empty_row_count() == AUTOSAVE_MIN_NON_EMPTY_ROWS - 1

        armed = []

        class _FakeTimer:
            def __init__(self, *_a, **_k):
                pass

            def cancel(self):
                pass

        def _fake_timer(*a, **k):
            armed.append(a)
            return _FakeTimer(*a, **k)

        monkeypatch.setattr(editor_module.ui, "timer", _fake_timer)

        handle.name_input.value = "draft"
        handle.schedule_autosave()
        # Below threshold: no debounce timer is armed and the gate status is shown.
        assert armed == []
        assert "filled rows" in (handle.status_label.text or "")

        # At exactly the threshold, auto-save arms the debounce timer.
        handle.rows[AUTOSAVE_MIN_NON_EMPTY_ROWS - 1] = _filled(
            AUTOSAVE_MIN_NON_EMPTY_ROWS
        )
        assert handle._non_empty_row_count() == AUTOSAVE_MIN_NON_EMPTY_ROWS
        armed.clear()
        handle.schedule_autosave()
        assert len(armed) == 1

    def test_column_mode_swap_adds_group_and_info_columns(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        assert [c["name"] for c in handle.table.columns] == [
            "source", "target", "weight", "color",
        ]
        handle.set_columns_mode("full")
        assert [c["name"] for c in handle.table.columns] == list(store.EDGE_COLUMNS)
        handle.set_columns_mode("basic")
        assert len(handle.table.columns) == 4

    def test_load_csv_text_into_table(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        text = "source,target,weight,color,source_group\nA,B,1,#fff000,PAM\n"
        assert handle.load_csv_text(text) is True
        assert handle.rows[0]["source"] == "A"
        assert handle.rows[0]["color"] == "#fff000"
        assert handle.rows[0]["source_group"] == "PAM"
        assert "Loaded 1 rows from CSV" in handle.status_label.text

    def test_color_cell_picker_opens_popup(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        opened = []

        class _FakePopup:
            def open(self, initial):
                opened.append(initial)

        handle._pick_popup = _FakePopup()
        handle.on_color_pick(SimpleNamespace(args={"id": 0, "field": "color"}))
        assert opened == ["#145cff"]
        # Non-color fields never open the popup.
        handle.on_color_pick(SimpleNamespace(args={"id": 0, "field": "weight"}))
        assert len(opened) == 1

    def test_enriched_csv_notice_present(self, store_patch_for_component):
        """The editor carries the in-page notice steering complex enriched
        graphs (hover labels, custom groups, more columns) to local CSV
        files, linked to the format docs and the Type Mapping guide."""
        client, handle = build_editor(store_patch_for_component)
        notice = next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id")
            == "card-net-viz-enriched-notice"
        )
        assert notice is not None
        texts = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if getattr(el, "text", "")
        ]
        joined = "\n".join(texts)
        assert "local CSV file" in joined
        assert "Auto Type Mapping" in joined
        assert "Enriched edge-list format" in texts
        assert "Auto Type Mapping guide" in texts

    def test_apply_picked_color_updates_table_cell(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle._pending_pick = {"row_id": 1, "field": "color"}
        handle._apply_picked_color("#ff0000")
        assert handle.rows[1]["color"] == "#ff0000"
        assert [row.get("color") for row in handle.table.rows] == ["", "#ff0000"]

    def test_runnable_path_file_requires_complete_edge(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"source": "A", "target": "", "weight": ""}], name="incomplete")
        assert handle.runnable_path_file() is None
        handle.set_rows(ROWS, name="complete")
        assert handle.runnable_path_file() is not None

    def test_runnable_blocks_on_validation_errors(self, store_patch_for_component):
        """A validation error blocks the run and is surfaced in the panel."""
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"source": "A", "target": "B", "weight": "many"}], name="bad")
        assert handle.runnable_path_file() is None
        assert handle.validation_panel.visible is True
        assert "not a number" in handle.validation_label.text
        assert "Fix the edge list errors" in handle.status_label.text
        handle.rows[0]["weight"] = "5"
        assert handle.runnable_path_file() is not None
        assert handle.validation_panel.visible is False

    def test_runnable_path_file_without_draft_name_uses_transient_csv(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        path = handle.runnable_path_file()
        try:
            assert path == handle.transient_csv_path
            assert path and Path(path).exists()
            assert "aMe12,aMe10,128" in Path(path).read_text(encoding="utf-8")
            assert "draft name is optional" in handle.status_label.text
            assert store.list_drafts() == []
        finally:
            handle.cleanup_transient_csv()
        assert not Path(path).exists()

    def test_export_downloads_without_draft_name(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        downloads = []
        handle.table.client.download = lambda src, filename, media_type: downloads.append(
            (src, filename, media_type)
        )

        exported = handle.export_csv()

        assert re.fullmatch(r"edge_list_\d{8}_\d{6}\.csv", exported)
        assert len(downloads) == 1
        content, filename, media_type = downloads[0]
        assert content.decode("utf-8").startswith("source,target,weight\n")
        assert re.fullmatch(r"edge_list_\d{8}_\d{6}\.csv", filename)
        assert media_type == "text/csv"
        assert store.list_drafts() == []

    def test_export_marks_clean_and_copies(self, store_patch_for_component, tmp_path):
        export_dir = tmp_path / "run_output"
        client, handle = build_editor(store_patch_for_component, export_dir=export_dir)
        handle.set_rows(ROWS, name="exp")
        handle.flush_autosave()
        exported = handle.export_csv()
        assert exported and Path(exported).parent == export_dir
        assert re.fullmatch(r"exp_edge_list_\d{8}_\d{6}\.csv", Path(exported).name)
        assert Path(exported).exists()
        assert meta("exp")["dirty"] is False
        assert "no unsaved changes" in handle.status_label.text

    def test_editor_has_no_draft_load_or_delete_controls(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        labels = [
            (getattr(el, "_props", None) or {}).get("label")
            for el in client.elements.values()
        ]
        assert "Load Draft" not in labels
        assert "Load" not in labels
        assert "Delete Draft" not in labels
        assert "Apply to Selected" not in labels


# =============================================================================
# Network tab integration
# =============================================================================

class TestNetworkTabIntegration:
    def _patch_store(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "_store_dir", tmp_path / "tab_drafts")

    def _build_tab(self, monkeypatch, tmp_path):
        from ui.tabs.visualization import create_net_viz_tab
        client = Client(page(f"/network-tab-{uuid.uuid4().hex}"))
        with client:
            create_net_viz_tab()
        return client

    def _labels(self, client):
        return [
            (getattr(el, "_props", None) or {}).get("label")
            for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label")
        ]

    def _by_id(self, client, card_id):
        return next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id") == card_id
        )

    def _mode_button(self, client, text):
        matches = [
            el for el in client.elements.values()
            if getattr(el, "text", None) == text
        ]
        assert len(matches) == 1, f"button {text!r}: {len(matches)}"
        return matches[0]

    @staticmethod
    def _click_button(button):
        for listener in (getattr(button, "_event_listeners", None) or {}).values():
            if getattr(listener, "type", None) == "click":
                listener.handler(SimpleNamespace())

    def test_mode_buttons_replace_canvas_source_select(self, monkeypatch, tmp_path):
        """The Canvas Source dropdown is replaced by three segmented mode
        buttons (Edge list editor / Empty canvas / File upload)."""
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        selects = [
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label") == "Canvas Source"
        ]
        assert selects == []
        for name in ("Edge list editor", "Empty canvas", "File upload"):
            self._mode_button(client, name)

    def test_editor_card_present(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-net-viz-edge-editor" in ids

    def test_net_viz_editor_does_not_receive_output_dir_provider(self, monkeypatch, tmp_path):
        """Net-Viz edge-list exports should only trigger the browser download."""
        self._patch_store(monkeypatch, tmp_path)
        from ui.tabs import visualization

        original_editor = visualization.edge_list_editor
        captured = {}

        def capture_editor(*args, **kwargs):
            captured["kwargs"] = kwargs
            handle = original_editor(*args, **kwargs)
            captured["handle"] = handle
            return handle

        monkeypatch.setattr(visualization, "edge_list_editor", capture_editor)
        self._build_tab(monkeypatch, tmp_path)

        assert "export_dir_provider" not in captured["kwargs"]
        assert captured["handle"].export_dir_provider is None

    def test_no_reminder_without_dirty_drafts(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        store.save_draft("clean", ROWS, dirty=False)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" not in ids

    def test_dirty_drafts_do_not_render_a_recovery_card(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        store.save_draft("unfinished", ROWS)
        client = self._build_tab(monkeypatch, tmp_path)
        ids = [(getattr(el, "_props", None) or {}).get("id") for el in client.elements.values()]
        assert "card-edge-draft-recovery" not in ids

    def test_source_switch_toggles_path_input(self, monkeypatch, tmp_path):
        """The mode buttons swap exactly one visible source panel and keep
        the editor table expanded whenever the editor mode is active."""
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        editor_panel = self._by_id(client, "card-net-viz-editor-panel")
        empty_panel = self._by_id(client, "card-net-viz-empty-canvas")
        path_panel = self._by_id(client, "net-viz-path-input")
        editor_card = self._by_id(client, "card-net-viz-edge-editor")
        # Default mode is Edge list editor: editor panel visible with the
        # table expanded; the other two panels hidden.
        assert editor_panel.visible is True
        assert empty_panel.visible is False
        assert path_panel.visible is False
        assert editor_card.value is True
        self._click_button(self._mode_button(client, "Empty canvas"))
        assert editor_panel.visible is False
        assert empty_panel.visible is True
        assert path_panel.visible is False
        self._click_button(self._mode_button(client, "File upload"))
        assert editor_panel.visible is False
        assert empty_panel.visible is False
        assert path_panel.visible is True
        self._click_button(self._mode_button(client, "Edge list editor"))
        assert editor_panel.visible is True
        assert empty_panel.visible is False
        assert path_panel.visible is False
        assert editor_card.value is True
