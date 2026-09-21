"""Tests for the layer-style editor store + component (Skeleton advanced table).

Covers:
- ui/layer_style_store.py: draft CRUD, CSV layout, validation, and upload
  parsing (including unquoted CSS color functions).
- ui/components/layer_style_editor.py: editor state, add/delete/inline edit,
  auto-save flush, CSV export/upload.
- ui/tabs/visualization.py: Skeleton Layer Editor dropdown wiring.
"""
import re
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ui.layer_style_store as store
from ui.layer_style_store import LAYER_STYLE_COLUMNS


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the draft store at a temp directory for every test."""
    monkeypatch.setattr(store, "_store_dir", tmp_path / "layer_style_drafts")
    yield


ROWS = [
    {"layer": "0", "neuron": "aMe12", "color": "#ff0000"},
    {"layer": "0", "neuron": "aMe13", "color": "rgb(0,255,0)"},
    {"layer": "1", "neuron": "dn1", "color": "rgba(0,0,255,0.5)"},
]


def meta(name):
    return store.get_meta(name)


class TestStoreValidation:
    def test_normalize_fills_all_columns(self):
        assert store.normalize_rows([{"layer": "0", "neuron": " aMe12 "}]) == [{
            "layer": "0", "neuron": "aMe12", "color": "",
            "synapse_color": "", "pre_synaptic_color": "", "post_synaptic_color": "",
        }]

    def test_valid_rows(self):
        assert store.validate_rows(ROWS) == []

    def test_missing_layer_or_neuron_reported(self):
        errors = store.validate_rows([{"layer": "", "neuron": "aMe12"}])
        assert any("missing layer" in e for e in errors)
        errors = store.validate_rows([{"layer": "0", "neuron": ""}])
        assert any("missing neuron" in e for e in errors)

    def test_non_numeric_layer_reported(self):
        errors = store.validate_rows([{"layer": "x", "neuron": "aMe12"}])
        assert any("not a number" in e for e in errors)

    def test_complete_rows(self):
        # All rows carry layer + neuron, so all are complete (normalized).
        assert store.complete_rows(ROWS) == store.normalize_rows(ROWS)
        # A row missing a neuron is dropped.
        partial = ROWS + [{"layer": "2", "neuron": ""}]
        assert len(store.complete_rows(partial)) == len(ROWS)

    def test_empty_rows_ignored(self):
        assert store.validate_rows([{}, {"layer": "", "neuron": ""}]) == []

    def test_layers_must_be_continuous(self):
        errors = store.validate_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ])
        assert any("missing layer(s) 2" in error for error in errors)
        assert store.validate_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "2", "neuron": "b"},
        ]) == []


class TestStoreSaveLoad:
    def test_round_trip(self):
        slug = store.save_draft("layers", ROWS)
        assert slug == "layers"
        assert store.load_draft("layers") == store.normalize_rows(ROWS)

    def test_csv_header_has_all_columns(self):
        store.save_draft("layers", ROWS)
        header = Path(store.draft_csv_path("layers")).read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(LAYER_STYLE_COLUMNS)

    def test_meta_dirty(self):
        store.save_draft("layers", ROWS)
        m = meta("layers")
        assert m["dirty"] is True
        assert m["row_count"] == 3

    def test_quoted_color_round_trips(self):
        store.save_draft("q", [{"layer": "0", "neuron": "n", "color": "rgba(1,2,3,0.5)"}])
        loaded = store.load_draft("q")
        assert loaded[0]["color"] == "rgba(1,2,3,0.5)"

    def test_load_rows_from_csv_text_unquoted_color(self):
        text = "layer,neuron,color\n0,aMe12,rgba(74,144,226,0.3)\n"
        rows = store.load_rows_from_csv_text(text)
        assert rows[0]["color"] == "rgba(74,144,226,0.3)"

    def test_delete_draft(self):
        store.save_draft("layers", ROWS)
        assert store.delete_draft("layers") is True
        assert store.load_draft("layers") is None


class TestModeColumnsAndLayers:
    def test_available_layers_contiguous_from_1(self):
        assert store.available_layers([]) == [1]
        assert store.available_layers([{"layer": "1"}]) == [1, 2]
        assert store.available_layers([{"layer": "1"}, {"layer": "2"}]) == [1, 2, 3]
        # If no row uses layer 1, only 1 is offered (never a higher number).
        assert store.available_layers([{"layer": "2"}]) == [1]

    def test_next_layer_number_starts_at_one_then_uses_maximum(self):
        assert store.next_layer_number([]) == 1
        assert store.next_layer_number([{"layer": "1", "neuron": "a"}]) == 2
        assert store.next_layer_number([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ]) == 4

    def test_mode_columns(self):
        assert store.mode_columns("synapse") == ("layer", "neuron", "color", "synapse_color")
        assert store.mode_columns("pre-post sites") == (
            "layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"
        )

    def test_rows_to_csv_for_mode(self):
        syn = store.rows_to_csv_for_mode(ROWS, "synapse")
        assert syn.splitlines()[0] == "layer,neuron,color,synapse_color"
        pp = store.rows_to_csv_for_mode(ROWS, "pre-post sites")
        assert pp.splitlines()[0] == "layer,neuron,color,pre_synaptic_color,post_synaptic_color"


class TestLogicalRows:
    def test_logical_normalize_accepts_flat_and_logical(self):
        flat = store.logical_normalize([{"layer": "0", "neuron": "aMe12"}])
        assert flat == [{
            "layer": "0", "neurons": ["aMe12"], "color": "",
            "synapse_color": "", "pre_synaptic_color": "", "post_synaptic_color": "",
        }]
        logical = store.logical_normalize([{"layer": "1", "neurons": ["a", "b"]}])
        assert logical[0]["neurons"] == ["a", "b"]

    def test_flatten_rows_expands_chips_and_drops_empty_rows(self):
        rows = store.flatten_rows([
            {"layer": "1", "neurons": ["aMe12", "aMe13"], "color": "#111"},
            {"layer": "", "neurons": [], "color": ""},  # scaffolding -> dropped
        ])
        assert [
            (r["layer"], r["neuron"], r["color"]) for r in rows
        ] == [("1", "aMe12", "#111"), ("1", "aMe13", "#111")]

    def test_validate_ignores_only_fully_empty_rows(self):
        assert store.validate_rows([
            {"layer": "", "neurons": []},          # empty scaffolding -> ok
            {"layer": "1", "neurons": []},          # layer only -> missing neuron
            {"layer": "", "neurons": ["x"]},       # neuron only -> missing layer
        ]) == ["Row 2: missing neuron", "Row 3: missing layer"]

    def test_multi_chip_same_layer_is_not_a_gap(self):
        # Two cells on the same layer with multiple chips must not look like a gap.
        assert store.validate_rows([
            {"layer": "1", "neurons": ["a"]},
            {"layer": "1", "neurons": ["b", "c"]},
        ]) == []

    def test_rows_to_csv_for_mode_flattens_multi_chip_cell(self):
        csv_text = store.rows_to_csv_for_mode(
            [{"layer": "1", "neurons": ["a", "b"], "color": "#111"}], "synapse"
        )
        assert csv_text.splitlines() == [
            "layer,neuron,color,synapse_color",
            "1,a,#111,",
            "1,b,#111,",
        ]

    def test_save_and_load_round_trips_flattened_rows(self):
        slug = store.save_draft("multi", [
            {"layer": "1", "neurons": ["a", "b"], "color": "#111"},
            {"layer": "", "neurons": [], "color": ""},
        ])
        assert slug == "multi"
        rows = store.load_draft("multi")
        assert [(r["layer"], r["neuron"], r["color"]) for r in rows] == [
            ("1", "a", "#111"), ("1", "b", "#111")
        ]


# =============================================================================
# UI component: editor handle behavior
# =============================================================================

from nicegui import Client
from nicegui.page import page


@pytest.fixture()
def store_patch_for_component(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_store_dir", tmp_path / "comp_drafts")
    return tmp_path / "comp_drafts"


def build_editor(store_dir, export_dir=None):
    from ui.components.layer_style_editor import layer_style_editor
    client = Client(page(f"/layer-style-editor-{uuid.uuid4().hex}"))
    with client:
        handle = layer_style_editor(
            export_dir_provider=lambda: str(export_dir) if export_dir else None
        )
    return client, handle


class TestEditorHandle:
    def test_card_elements_exist(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ids = [
            (getattr(el, "_props", None) or {}).get("id")
            for el in client.elements.values()
        ]
        assert "card-skeleton-layer-style-editor" in ids
        assert handle.table is not None and handle.status_label is not None
        assert handle.name_input is not None
        assert handle._pick_popup is not None
        # Suggestions are rendered in a plain overlay below the focused cell (no
        # Quasar menu, so typing keeps focus).
        assert handle._suggest_overlay is not None
        # The fresh editor starts with 3 empty scaffolding rows.
        assert len(handle.rows) == 3
        assert all(store._logical_is_empty(row) for row in handle.rows)

    def test_add_neuron_control_omits_counter_and_clear_action(
        self, store_patch_for_component
    ):
        client, _handle = build_editor(store_patch_for_component)
        # The pending add-form (with Clear / count badge) was removed in favour of
        # direct in-table editing, so neither control should be present.
        assert not [
            element for element in client.elements.values()
            if getattr(element, "text", "") == "Clear"
        ]
        assert not [
            element for element in client.elements.values()
            if type(element).__name__ == "Badge"
            and "neuron" in str(getattr(element, "text", "")).lower()
        ]

    def test_add_empty_row_and_inline_edit_autosave(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "my layers"
        handle.add_empty_row()
        # Edit the first cell as if the chip input committed a neuron list, with
        # a layer and a color, then flush the debounced auto-save.
        handle.on_inline_commit(SimpleNamespace(args={
            "id": 0, "field": "neuron", "value": ["aMe12"],
        }))
        handle.on_inline_commit(SimpleNamespace(args={
            "id": 0, "field": "layer", "value": "0",
        }))
        handle.on_inline_commit(SimpleNamespace(args={
            "id": 0, "field": "color", "value": "#123456",
        }))
        csv_path = handle.flush_autosave()
        assert csv_path and Path(csv_path).exists()
        rows = store.load_draft("my layers")
        assert rows[0]["neuron"] == "aMe12"
        assert rows[0]["color"] == "#123456"
        assert meta("my layers")["dirty"] is True

    def test_inline_edit_updates_row(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "neuron", "value": [" aMe13 "]})
        )
        assert handle.rows[1]["neurons"] == ["aMe13"]

    def test_inline_text_edit_does_not_rebuild_table(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        table_updates = []
        handle.table.update = lambda: table_updates.append(True)
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "neuron", "value": ["aMe13x"]})
        )
        assert handle.rows[1]["neurons"] == ["aMe13x"]
        assert table_updates == []

    def test_available_neurons_append_one_row_per_entry_to_one_batch_layer(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"layer": "1", "neuron": "existing"}])
        handle.begin_available_batch()
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        assert handle.available_query_values() == ["n1", "n2"]
        assert [(row["layer"], row["neurons"]) for row in handle.rows] == [
            ("1", ["existing"]), ("2", ["n1"]), ("2", ["n2"])
        ]
        # The viewer reports its complete selection after every toggle; only
        # the new entry is appended, and it stays on the same batch layer.
        assert handle.apply_available_neurons(["n1", "n2", "n3"]) == 1
        assert handle.rows[-1]["layer"] == "2"
        handle.begin_available_batch()
        assert handle.available_query_values() == []
        assert handle.apply_available_neurons(["n4"]) == 1
        assert handle.rows[-1]["layer"] == "3"

    def test_empty_available_neuron_table_starts_batch_at_layer_one(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        # The pristine 3 scaffolding rows are replaced by the batch on layer 1.
        assert {row["layer"] for row in handle.rows if row["neurons"]} == {"1"}

    def test_validation_panel_reports_and_clears_layer_gap(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ])
        # Editing must NOT surface validation live (only run/export does).
        assert handle.validation_panel.visible is False
        errors = handle._update_validation()  # what run/export triggers
        assert errors
        assert handle.validation_panel.visible is True
        assert "missing layer(s) 2" in handle.validation_label.text
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "layer", "value": "2"})
        )
        # A valid table clears the panel on the next run/export check.
        assert handle._update_validation() == []
        assert handle.validation_panel.visible is False

    def test_mode_switch_hides_prepost_color_column(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_synapse_mode("pre-post sites")
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"
        ]
        handle.set_synapse_mode("synapse")
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "synapse_color"
        ]

    def test_advanced_neuron_input_uses_dataset_suggestions(
        self, store_patch_for_component, monkeypatch
    ):
        from ui.components import layer_style_editor as editor_module

        calls = []
        monkeypatch.setattr(
            editor_module,
            "dataset_suggestions",
            lambda text, dataset, columns, limit=None: calls.append(
                (text, dataset, columns, limit)
            ) or [("aMe12", "type")],
        )
        client = Client(page(f"/layer-style-suggest-{uuid.uuid4().hex}"))
        with client:
            handle = editor_module.layer_style_editor(
                dataset_provider=lambda: "male-cns:v1.0",
                search_columns_provider=lambda: "type",
            )
        assert handle._suggest_neurons("aMe") == [("aMe12", "type")]
        assert calls == [("aMe", "male-cns:v1.0", "type", None)]

    def test_suggestion_commit_adds_chip_and_reoffers_without_closing(
        self, store_patch_for_component, monkeypatch
    ):
        """Picking a suggestion adds the chip and keeps the list open.

        Like the standard query box, the overlay is re-offered for the
        still-focused cell instead of being closed (no suppression window is
        armed), so subsequent focus/typing on the same cell continues to show
        the list. Deliberate divergence: this per-cell overlay re-offers the
        Recent/history list, while the shared query box holds the picked
        query's own rows for the next entry.
        """
        from ui.components import layer_style_editor as editor_module

        client = Client(page(f"/layer-style-suggest-life-{uuid.uuid4().hex}"))
        with client:
            handle = editor_module.layer_style_editor(
                dataset_provider=lambda: "male-cns:v1.0",
                search_columns_provider=lambda: "type",
            )
        # Record which cell the overlay opened for (instead of rendering DOM).
        showed = []
        monkeypatch.setattr(
            handle, "_show_neuron_suggestions",
            lambda row_id, text: showed.append((row_id, text)),
        )
        monkeypatch.setattr(handle, "_refocus_neuron_cell", lambda row_id: None)

        # Focusing an empty cell opens suggestions for that cell.
        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        assert showed == [(0, "")]
        assert handle._suggest_row == 0

        # Committing a picked suggestion adds the chip and re-offers the Recent
        # list for the still-focused cell (the list stays open, not suppressed).
        showed.clear()
        handle._commit_neuron_suggestion(0, "aMe12")
        assert handle.rows[0]["neurons"] == ["aMe12"]
        assert showed == [(0, "")]
        assert handle._suggest_suppress is False

        # A synthetic re-focus of the same row re-opens (no suppression window).
        showed.clear()
        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        assert showed == [(0, "")]

        # A reset empty input-value event on the same row also re-opens.
        showed.clear()
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": ""}))
        assert showed == [(0, "")]

        # Typing a non-empty query shows filtered results for that cell.
        showed.clear()
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "aMe"}))
        assert showed == [(0, "aMe")]

        # Switching to a different cell is a genuine action: it re-opens there.
        showed.clear()
        handle.on_neuron_focus(SimpleNamespace(args={"id": 1}))
        assert showed == [(1, "")]
        assert handle._suggest_suppress is False

    def test_autosave_gated_on_min_non_empty_rows(
        self, store_patch_for_component, monkeypatch
    ):
        """Auto-save only arms once the table holds >=5 filled rows.

        A handful of scaffolding rows must not create a throwaway draft: below the
        threshold schedule_autosave does not arm the debounce timer and surfaces a
        gate status instead, while the manual export path is unaffected.
        """
        from ui.components import layer_style_editor as editor_module
        from ui.components.layer_style_editor import AUTOSAVE_MIN_NON_EMPTY_ROWS

        client, handle = build_editor(store_patch_for_component)

        # Fresh editor starts with 3 empty scaffolding rows.
        assert handle._non_empty_row_count() == 0
        # Grow the table so it can hold up to the threshold.
        while len(handle.rows) < AUTOSAVE_MIN_NON_EMPTY_ROWS:
            handle.rows.append(handle._empty_logical_row())

        fill = {
            "color": "", "synapse_color": "",
            "pre_synaptic_color": "", "post_synaptic_color": "",
        }
        # 4 filled rows is still below the threshold.
        for i in range(AUTOSAVE_MIN_NON_EMPTY_ROWS - 1):
            handle.rows[i] = {
                "layer": str(i + 1), "neurons": [f"n{i}"], **fill,
            }
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
        handle.rows[AUTOSAVE_MIN_NON_EMPTY_ROWS - 1] = {
            "layer": str(AUTOSAVE_MIN_NON_EMPTY_ROWS), "neurons": ["n5"], **fill,
        }
        assert handle._non_empty_row_count() == AUTOSAVE_MIN_NON_EMPTY_ROWS
        armed.clear()
        handle.schedule_autosave()
        assert len(armed) == 1

    def test_suggestion_settings_gate_typing_overlay(
        self, store_patch_for_component, monkeypatch
    ):
        """Unchecking Input Auto-Suggestion hides the type-ahead suggestions.

        A non-empty query must close the overlay instead of rendering the
        dataset suggestion list, but the empty-field history list is unaffected.
        """
        from ui.components import layer_style_editor as editor_module
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        # Track whether the suggestion resolver was ever consulted (it should
        # not be while the setting is off).
        consulted = []
        monkeypatch.setattr(
            handle, "_suggest_neurons",
            lambda _text: consulted.append(
                "suggest") or [("aMe12", "type")],
        )
        monkeypatch.setattr(_config, "get_auto_suggest_enabled", lambda: False)

        def _catch_close(*_a, **_k):
            # _close_suggest_overlay arms the suppress window; record it but do
            # not hit the socket in the test.
            handle._suggest_suppress = True
            handle._suggest_suppress_until = time.time() + 0.4

        monkeypatch.setattr(handle, "_close_suggest_overlay", _catch_close)

        assert handle._suggestions_enabled() is False
        handle._show_neuron_suggestions(0, "aMe")
        # Suggestions were suppressed and no keyword search ran.
        assert consulted == []
        assert handle._suggest_suppress is True

    def test_suggestion_settings_gate_history_overlay(
        self, store_patch_for_component, monkeypatch
    ):
        """Unchecking Show Query History hides the empty-field history list.

        An empty field must close the overlay instead of rendering a Recent/
        Frequent list, but live self._suggest_neurons typing is unaffected.
        """
        from ui.components import layer_style_editor as editor_module
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        consulted = []
        monkeypatch.setattr(
            handle, "_recent_neuron_history",
            lambda: consulted.append("history") or [("l-LNv", "type")],
        )
        monkeypatch.setattr(_config, "get_show_history_enabled", lambda: False)

        def _catch_close(*_a, **_k):
            handle._suggest_suppress = True
            handle._suggest_suppress_until = time.time() + 0.4

        monkeypatch.setattr(handle, "_close_suggest_overlay", _catch_close)

        assert handle._history_enabled() is False
        handle._show_neuron_suggestions(0, "")
        # History was suppressed and the store was never queried.
        assert consulted == []
        assert handle._suggest_suppress is True

    def test_suggestion_settings_render_respects_history_flag(
        self, store_patch_for_component, monkeypatch
    ):
        """Enabled history renders the Recent payload (isHistory=true), while a
        typed query renders the dataset suggestion payload (isHistory=false)."""
        from ui.components import layer_style_editor as editor_module
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(_config, "get_auto_suggest_enabled", lambda: True)
        monkeypatch.setattr(_config, "get_show_history_enabled", lambda: True)
        monkeypatch.setattr(
            handle, "_suggest_neurons", lambda _t: [("aMe12", "type")],
        )
        monkeypatch.setattr(
            handle, "_recent_neuron_history", lambda: [("l-LNv", "type")],
        )
        seen = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript",
            lambda js: seen.append(js),
        )

        # Typing a query → suggestion payload, not history.
        handle._show_neuron_suggestions(0, "aMe")
        assert seen and "aMe12" in seen[-1]
        assert "true" not in seen[-1]  # not a history render

        # Empty field → history payload with the Recent flag.
        seen.clear()
        handle._show_neuron_suggestions(0, "")
        assert seen and "l-LNv" in seen[-1]
        assert "true" in seen[-1]  # isHistory=true

    def test_delete_selected_rows(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "del"
        handle.set_rows(ROWS + [{"layer": "2", "neuron": "X"}])
        handle._selected_ids = [1]
        handle.delete_selected()
        assert len(handle.rows) == 3
        assert handle.rows[1]["neurons"] == ["dn1"]

    def test_color_cell_picker_opens_popup(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="c")
        handle.on_color_pick(SimpleNamespace(args={"id": 0, "field": "color"}))
        assert handle._pending_pick == {"row_id": 0, "field": "color"}
        assert handle._pick_popup is not None

    def test_apply_picked_color_updates_table_cell(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="c")
        handle._pending_pick = {"row_id": 1, "field": "color"}
        handle._apply_picked_color("rgba(1, 2, 3, 0.4)")
        assert handle.rows[1]["color"] == "rgba(1, 2, 3, 0.4)"

    def test_set_synapse_mode_changes_columns(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "synapse_color"
        ]
        handle.set_synapse_mode("pre-post sites")
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "pre_synaptic_color", "post_synaptic_color"
        ]
        handle.set_synapse_mode("synapse")
        assert [c["name"] for c in handle.table.columns] == [
            "layer", "neuron", "color", "synapse_color"
        ]

    def test_table_row_dicts_carry_layer_opts(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(
            [{"layer": "1", "neuron": "a"}, {"layer": "2", "neuron": "b"}], name="x"
        )
        row_dicts = handle._row_dicts()
        assert row_dicts[0]["layer_opts"] == ["1", "2", "3"]
        assert row_dicts[1]["layer_opts"] == ["1", "2", "3"]

    def test_load_csv_text_into_table(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        ok = handle.load_csv_text("layer,neuron,color\n0,aMe12,rgba(74,144,226,0.3)\n")
        assert ok is True
        assert len(handle.rows) == 1
        assert handle.rows[0]["color"] == "rgba(74,144,226,0.3)"
        assert "Loaded 1 rows" in handle.status_label.text

    def test_runnable_csv_requires_layer_and_neuron(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"layer": "0", "neuron": ""}], name="incomplete")
        assert handle.runnable_csv_path() is None
        handle.set_rows(ROWS, name="complete")
        assert handle.runnable_csv_path() is not None

    def test_runnable_csv_without_name_uses_transient(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS, name="")
        path = handle.runnable_csv_path()
        try:
            assert path == handle.transient_csv_path
            assert path and Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            # The transient CSV uses the synapse-mode columns (default 'synapse').
            assert content.startswith(",".join(store.mode_columns("synapse")) + "\n")
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
        assert re.fullmatch(r"layers_\d{8}_\d{6}\.csv", exported)
        assert len(downloads) == 1
        content, filename, media_type = downloads[0]
        assert content.decode("utf-8").startswith(",".join(store.mode_columns("synapse")) + "\n")
        assert media_type == "text/csv"
        assert store.list_drafts() == []


# =============================================================================
# Skeleton tab integration
# =============================================================================

class TestSkeletonTabIntegration:
    def _patch_store(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "_store_dir", tmp_path / "tab_drafts")

    def _build_tab(self, monkeypatch, tmp_path):
        from ui.tabs.visualization import create_skeleton_tab
        client = Client(page(f"/skeleton-tab-{uuid.uuid4().hex}"))
        with client:
            create_skeleton_tab()
        return client

    def _by_label(self, client, label):
        matches = [
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("label") == label
        ]
        assert len(matches) == 1, f"label {label!r}: {len(matches)}"
        return matches[0]

    def _by_id(self, client, card_id):
        return next(
            el for el in client.elements.values()
            if (getattr(el, "_props", None) or {}).get("id") == card_id
        )

    def _mode_button(self, client, text):
        matches = [el for el in client.elements.values() if getattr(el, "text", None) == text]
        assert len(matches) == 1, f"button {text!r}: {len(matches)}"
        return matches[0]

    @staticmethod
    def _click_button(button):
        for listener in (getattr(button, "_event_listeners", None) or {}).values():
            if getattr(listener, "type", None) == "click":
                listener.handler(SimpleNamespace())

    def test_layer_editor_buttons_exist(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        for name in ("Standard", "Advanced", "File upload"):
            self._mode_button(client, name)

    def test_layer_editor_toggles_panels(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        standard = self._by_id(client, "card-skeleton-layers")
        advanced = self._by_id(client, "card-skeleton-layer-style")
        file_upload = self._by_id(client, "card-skeleton-layer-upload")
        assert standard.visible is True
        assert advanced.visible is False
        assert file_upload.visible is False
        self._click_button(self._mode_button(client, "Advanced"))
        assert standard.visible is False
        assert advanced.visible is True
        assert file_upload.visible is False
        self._click_button(self._mode_button(client, "File upload"))
        assert standard.visible is False
        assert advanced.visible is False
        assert file_upload.visible is True
        self._click_button(self._mode_button(client, "Standard"))
        assert standard.visible is True
        assert advanced.visible is False
        assert file_upload.visible is False

    def test_color_editor_panels_only_in_standard_mode(self, monkeypatch, tmp_path):
        """The neuron/synapse palette editors are visible only in Standard
        mode (the Advanced table and the uploaded CSV supply their own
        color columns). The ROI palette is independent of the layer editor
        — it defines neuron/synapse colors, never layer colors — and
        deliberately stays visible in every mode (bfa2180)."""
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        for palette_id in (
            "card-skeleton-neuron-palette",
            "card-skeleton-synapse-palette",
        ):
            palette = self._by_id(client, palette_id)
            assert palette.visible is True
            self._click_button(self._mode_button(client, "Advanced"))
            assert palette.visible is False
            self._click_button(self._mode_button(client, "File upload"))
            assert palette.visible is False
            self._click_button(self._mode_button(client, "Standard"))
            assert palette.visible is True
        roi = self._by_id(client, "card-skeleton-roi-palette")
        for mode in ("Advanced", "File upload", "Standard"):
            self._click_button(self._mode_button(client, mode))
            assert roi.visible is True, mode

    def test_synapse_view_mode_options(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        view = self._by_label(client, "Synapse Mode")
        assert view.options == ["synapse", "pre-post sites", "skip"]
        assert view.value == "synapse"
        shape = self._by_label(client, "Synapse Shape")
        assert "cone" in shape.options and "pre_post" not in shape.options
        pre_post_shape = self._by_label(client, "Pre/post shape")
        assert len(pre_post_shape.options) == 2
        # The threshold is renamed from "Min Synapse Count"; the old label is gone.
        self._by_label(client, "Synapse Threshold")
        labels = [(getattr(el, "_props", None) or {}).get("label") for el in client.elements.values()]
        assert "Min Synapse Count" not in labels

    def test_shape_defaults_follow_skeleton_mode(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        mode = self._by_label(client, "Skeleton Mode")
        synapse_shape = self._by_label(client, "Synapse Shape")
        pre_post_shape = self._by_label(client, "Pre/post shape")

        assert mode.value == "tube"
        assert synapse_shape.value == "cone"
        # Pre/post sites default to scatter markers so the HTML stays small,
        # regardless of the skeleton/morphology mode.
        assert pre_post_shape.value == "scatter (circles + diamonds)"

        mode.set_value("line")
        assert synapse_shape.value == "scatter"
        assert pre_post_shape.value == "scatter (circles + diamonds)"

        mode.set_value("tube")
        assert synapse_shape.value == "cone"
        assert pre_post_shape.value == "scatter (circles + diamonds)"

    def test_synapse_view_mode_toggles_shapes_and_warning(self, monkeypatch, tmp_path):
        self._patch_store(monkeypatch, tmp_path)
        client = self._build_tab(monkeypatch, tmp_path)
        view = self._by_label(client, "Synapse Mode")
        shape = self._by_label(client, "Synapse Shape")
        pre_post_shape = self._by_label(client, "Pre/post shape")
        warning = self._by_id(client, "card-skeleton-pre-post-warning")
        # Default: synapse mode -> shape visible, pre/post shape + warning hidden.
        assert shape.visible is True
        assert pre_post_shape.visible is False
        assert warning.visible is False
        view.set_value("pre-post sites")
        assert shape.visible is False
        assert pre_post_shape.visible is True
        assert warning.visible is True
        view.set_value("skip")
        assert shape.visible is False
        assert pre_post_shape.visible is False
        assert warning.visible is False
