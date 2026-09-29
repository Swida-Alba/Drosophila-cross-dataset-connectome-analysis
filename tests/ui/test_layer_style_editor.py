"""Tests for the layer-style editor store + component (Skeleton advanced table).

Covers:
- ui/layer_style_store.py: draft CRUD, CSV layout, validation, and upload
  parsing (including unquoted CSS color functions).
- ui/components/layer_style_editor.py: editor state, add/delete/inline edit,
  auto-save flush, CSV export/upload.
- ui/tabs/visualization.py: Skeleton Layer Editor wiring.
"""
import json
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

    def test_layers_may_be_discrete(self):
        # Layer numbers are free whole numbers: gaps and any starting number
        # are legitimate (a numbering jump breaks the inter-layer synapse pair
        # at render time, it is not a table error).
        assert store.validate_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "3", "neuron": "b"},
        ]) == []
        assert store.validate_rows([
            {"layer": "5", "neuron": "a"},
            {"layer": "7", "neuron": "b"},
        ]) == []
        assert store.validate_rows([
            {"layer": "0", "neuron": "a"},
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


def render_call(js):
    """The (row_id, items, is_history) triple behind one overlay render.

    ``items`` entries are ``[value, hint, already_added]``: a picked value keeps
    its row and is marked, exactly as the shared query box does.
    """
    payload = js.split("render(", 1)[1].rsplit(");", 1)[0]
    return json.loads("[" + payload + "]")


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

    def test_layer_edit_is_a_pure_model_update(self, store_patch_for_component):
        """A layer edit never refreshes the table.

        The layer cell is a bare numeric input (no option list to re-render) and
        discrete numbers are legitimate, so any layer edit is a model update
        plus the autosave schedule — a refresh would remount every cell and
        drop focus and half-typed text elsewhere.
        """
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "1", "neuron": "b"},
            {"layer": "2", "neuron": "c"},
        ])
        refreshes = []
        handle.refresh_table = lambda *a, **k: refreshes.append(k)
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 0, "field": "layer", "value": "2"})
        )
        assert handle.rows[0]["layer"] == "2"
        assert refreshes == []
        # Discrete jumps are ordinary values too.
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 0, "field": "layer", "value": "10"})
        )
        assert handle.rows[0]["layer"] == "10"
        assert refreshes == []

    def test_on_select_records_ids_without_rebuilding_rows(
        self, store_patch_for_component
    ):
        """The selection handler must not push fresh row dicts to the table.

        Re-assigning table.selected on every checkbox click remounts all cells;
        the client click already applied the state, so only the ids are recorded
        (the next refresh_table re-syncs from them).
        """
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(ROWS)
        table = handle.table
        rows_before, selected_before = table.rows, table.selected
        handle.on_select(SimpleNamespace(selection=[{"id": 1}]))
        assert handle._selected_ids == [1]
        assert table.rows is rows_before
        assert table.selected is selected_before

    def test_schedule_autosave_debounce_polls_to_a_single_flush(
        self, store_patch_for_component, monkeypatch
    ):
        """The debounce is a Python deadline polled by one persistent timer.

        Creating a one-shot ui.timer per commit (and deleting it at flush)
        mounts/unmounts elements in a live slot, which remounts the table's
        cells under the user's cursor; the deadline now just ages out through
        ``_autosave_poll``, so rescheduling never touches the element tree.
        """
        from ui.components import layer_style_editor as editor_module

        client, handle = build_editor(store_patch_for_component)
        handle.name_input.value = "draft"
        handle.set_rows([
            {"layer": str(i + 1), "neurons": [f"n{i}"]} for i in range(5)
        ])

        flushed = []
        monkeypatch.setattr(handle, "flush_autosave", lambda: flushed.append(1))

        handle.schedule_autosave()
        assert handle._autosave_due is not None
        # The deadline has not passed: the poll must not flush yet.
        handle._autosave_poll()
        assert flushed == []
        # Once it passes, exactly one poll flushes and disarms the deadline.
        monkeypatch.setattr(
            editor_module.time, "time", lambda: handle._autosave_due + 1
        )
        handle._autosave_poll()
        assert flushed == [1]
        assert handle._autosave_due is None
        # Further polls are no-ops until the next commit re-arms.
        handle._autosave_poll()
        assert flushed == [1]

    def test_refocus_js_never_yanks_focus_from_another_input(
        self, store_patch_for_component, monkeypatch
    ):
        """The pick's programmatic refocus is guarded inside the JS.

        The Python side always sends the focus snippet (so the pick keeps the
        overlay anchored), but the snippet itself must stand down when the user
        is already typing into some other input.
        """
        client, handle = build_editor(store_patch_for_component)
        sent = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript", lambda js: sent.append(js)
        )
        handle._refocus_neuron_cell(0)
        js = sent[-1]
        assert ".focus()" in js  # still refocuses when nothing else holds focus
        assert "document.activeElement" in js
        assert "c.contains(ae))return" in js
        # The refocus retries until the input RETAINS focus: the refresh's
        # re-render lands asynchronously, so a single focus() can land on the
        # input that is about to be replaced.
        assert "setTimeout(step" in js
        # The typed prefix is cleared here (a keyed refresh no longer remounts
        # the cell, so nothing else wipes it before the next blur commit).
        assert "inp.value=''" in js

    def test_body_slot_rows_are_keyed(self, store_patch_for_component):
        """The body slot keys each row so a refresh patches cells in place.

        Without a per-row key, pushing fresh row dicts remounts every cell input
        and drops focus plus uncommitted text (the reported auto-save
        interruption); the stable row id turns the refresh into a keyed patch.
        """
        from ui.components.layer_style_editor import _BODY_SLOT
        assert ':key="props.row.id"' in _BODY_SLOT

    def test_focus_continuity_guard_is_installed(self, store_patch_for_component):
        """The page installs the client-side focus-continuity guard.

        In this NiceGUI/Quasar stack every table re-render rebuilds all body
        rows, dropping focus mid-edit; the guard restores the caret (and the
        field content when the rebuild wiped it) whenever the focus fell to
        <body> right after row removals — covering the remounts no server-side
        refresh reduction can eliminate (e.g. a selection toggle).
        """
        from ui.components.layer_style_editor import _SUGGESTION_JS
        assert "__drocatCellFocusGuard" in _SUGGESTION_JS
        assert "drocatRowsRebuiltAt" in _SUGGESTION_JS

    def test_available_neurons_append_one_row_per_entry_on_its_own_layer(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([{"layer": "1", "neuron": "existing"}])
        handle.begin_available_batch()
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        assert handle.available_query_values() == ["n1", "n2"]
        # Each entry gets its OWN layer: consecutive numbers from the table's
        # next free layer, so n1 lands on 2 and n2 on 3.
        assert [(row["layer"], row["neurons"]) for row in handle.rows] == [
            ("1", ["existing"]), ("2", ["n1"]), ("3", ["n2"])
        ]
        # The viewer reports its complete selection after every toggle; only
        # the new entry is appended, on the next consecutive layer.
        assert handle.apply_available_neurons(["n1", "n2", "n3"]) == 1
        assert handle.rows[-1]["layer"] == "4"
        handle.begin_available_batch()
        assert handle.available_query_values() == []
        assert handle.apply_available_neurons(["n4"]) == 1
        assert handle.rows[-1]["layer"] == "5"

    def test_empty_available_neuron_table_starts_entries_at_layer_one(
        self, store_patch_for_component
    ):
        client, handle = build_editor(store_patch_for_component)
        assert handle.apply_available_neurons(["n1", "n2"]) == 2
        # The pristine 3 scaffolding rows are replaced by the batch: one entry
        # per row, each on its own consecutive layer.
        assert {row["layer"] for row in handle.rows if row["neurons"]} == {"1", "2"}

    def test_validation_panel_reports_and_clears_bad_layer(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows([
            {"layer": "1", "neuron": "a"},
            {"layer": "abc", "neuron": "b"},
        ])
        # Editing must NOT surface validation live (only run/export does).
        assert handle.validation_panel.visible is False
        errors = handle._update_validation()  # what run/export triggers
        assert errors
        assert handle.validation_panel.visible is True
        assert "not a number" in handle.validation_label.text
        handle.on_inline_edit(
            SimpleNamespace(args={"id": 1, "field": "layer", "value": "3"})
        )
        # A valid table (discrete layers included) clears the panel on the next
        # run/export check.
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

    def test_suggestion_commit_holds_the_query_list_and_ticks_picks(
        self, store_patch_for_component, monkeypatch
    ):
        """The in-table overlay follows the query box: pick, then keep picking.

        A pick adds the chip and re-renders the same query's rows, marking the
        values that are already chips instead of dropping them, so entries can be
        added one after another without retyping. A pick from the history list
        re-offers the history list, and any close ends the held query.
        """
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(_config, "get_auto_suggest_enabled", lambda: True)
        monkeypatch.setattr(_config, "get_show_history_enabled", lambda: True)
        monkeypatch.setattr(
            handle, "_recent_neuron_history", lambda: [("l-LNv", "type")],
        )
        calls = []

        def provider(text):
            calls.append(text)
            return [
                ("PPL101", "type"), ("PPL102", "type"), ("PPL103", "type"),
            ] if str(text).startswith("PPL1") else []

        monkeypatch.setattr(handle, "_suggest_neurons", provider)
        monkeypatch.setattr(handle, "_refocus_neuron_cell", lambda row_id: None)
        sent = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript", lambda js: sent.append(js),
        )

        def renders():
            """Overlay re-renders in order (add picks also send chip-entry JS)."""
            return [s for s in sent if "drocatSuggest.render(" in s]

        def marks():
            """Values the newest render offered, with their tick flags."""
            _, items, _ = render_call(renders()[-1])
            return {item[0]: item[2] for item in items}

        def added():
            return sorted(value for value, tick in marks().items() if tick)

        # Focusing an empty cell offers the history list, and nothing is held.
        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        assert render_call(renders()[-1])[2] is True
        assert handle._suggest_query is None
        assert handle._suggest_visible is True

        # A pick from the history list adds the chip and re-offers that list --
        # with the row ticked now, because a history row marks a current chip
        # exactly like a type-ahead row does.
        handle._commit_neuron_suggestion(0, "l-LNv")
        assert handle.rows[0]["neurons"] == ["l-LNv"]
        assert render_call(renders()[-1])[2] is True
        assert handle._suggest_query is None
        assert added() == ["l-LNv"]

        # Clicking that marked row is the reverse of the first click: the chip
        # goes away and the list keeps its place, unmarked.
        handle._commit_neuron_suggestion(0, "l-LNv")
        assert handle.rows[0]["neurons"] == []
        assert render_call(renders()[-1])[2] is True
        assert added() == []
        handle._commit_neuron_suggestion(0, "l-LNv")
        assert handle.rows[0]["neurons"] == ["l-LNv"]

        # Past the chip-entry window, field events are the user's again.
        handle._simulate_until = 0.0
        # A typed query holds: the pick keeps its rows and ticks the new chip.
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "PPL1"}))
        assert calls == ["PPL1"]
        assert added() == []  # the held chip is not in this pool
        handle._commit_neuron_suggestion(0, "PPL101")
        assert handle.rows[0]["neurons"] == ["l-LNv", "PPL101"]
        assert render_call(renders()[-1])[2] is False
        assert added() == ["PPL101"]
        assert marks()["PPL102"] is False
        # The held rows come from the cached pool, so the provider is not asked
        # again for the same query.
        assert calls == ["PPL1"]

        # The list stayed put, so the next pick needs no new text.
        handle._commit_neuron_suggestion(0, "PPL102")
        assert added() == ["PPL101", "PPL102"]

        # And a third click on a ticked row takes that one back out. A removal
        # cannot go through the cell's own entry path, so it is the one pick
        # that still refreshes the table (its refocus is stubbed away here).
        handle._commit_neuron_suggestion(0, "PPL101")
        assert handle.rows[0]["neurons"] == ["l-LNv", "PPL102"]
        assert added() == ["PPL102"]
        assert marks()["PPL101"] is False
        assert handle._suggest_query == "PPL1"
        assert handle._suggest_keep_open_until > time.time()

        # Removing a chip through the cell's own x clears its tick too.
        handle._suppress_neuron_value_until = 0.0
        handle.on_inline_edit(SimpleNamespace(args={
            "id": 0, "field": "neuron", "value": ["l-LNv"],
        }))
        assert added() == []

        # A commit (focus leaving the cell) ends the held query: the next focus
        # offers the history list again.
        handle._suggest_keep_open_until = 0.0
        handle.on_inline_commit(SimpleNamespace(args={
            "id": 0, "field": "neuron", "value": ["l-LNv"],
        }))
        assert handle._suggest_visible is False
        assert handle._suggest_query is None
        sent.clear()
        # Past the short window that swallows the refresh's synthetic refocus, a
        # genuine refocus of the cell offers the history list again.
        handle._suggest_suppress_until = 0.0
        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        assert render_call(renders()[-1])[2] is True

    def test_suggestion_pick_never_touches_the_table_and_holds_the_list(
        self, store_patch_for_component, monkeypatch
    ):
        """An add pick enters the chip through the cell, not a table refresh.

        The refresh the old flow ran remounted every cell of the table: focus
        dropped, typing elsewhere was wiped, and a one-shot refocus had to race
        the async re-render (and lost whenever the socket lagged). The add now
        goes through the cell's own entry path (simulated type + Enter), so the
        table is untouched -- and the held list survives both the pick and a
        genuine refocus replay.
        """
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(_config, "get_auto_suggest_enabled", lambda: True)
        monkeypatch.setattr(_config, "get_show_history_enabled", lambda: True)
        monkeypatch.setattr(
            handle, "_recent_neuron_history", lambda: [("l-LNv", "type")],
        )
        monkeypatch.setattr(handle, "_suggest_neurons", lambda text: [
            ("PPL101", "type"), ("PPL102", "type")])

        sent = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript", lambda js: sent.append(js),
        )

        def renders():
            return [s for s in sent if "drocatSuggest.render(" in s]

        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "PPL1"}))
        render_count = len(renders())
        handle._commit_neuron_suggestion(0, "PPL101")

        sims = [s for s in sent if "__drocatSuggestSim" in s]
        assert sims, "the add pick must enter the chip through the cell's own q-select"
        assert handle.rows[0]["neurons"] == ["PPL101"]
        assert not any("setTimeout(step" in s for s in sent), (
            "an add pick must not need the settle-refocus: nothing remounts"
        )
        assert len(renders()) == render_count + 1
        _, items, is_history = render_call(renders()[-1])
        assert is_history is False
        assert handle._suggest_query == "PPL1"
        assert {item[0]: item[2] for item in items} == {
            "PPL101": True, "PPL102": False}

        # A genuine refocus replay one round trip later keeps the held rows
        # (the keep-open window logic still guards real refocuses).
        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        assert len(renders()) == render_count + 1
        assert handle._suggest_query == "PPL1"

        # A removal pick is the one path that still refreshes the table, and
        # its refocus uses the settle-retry JS (the remount drops the focus).
        handle._commit_neuron_suggestion(0, "PPL101")
        assert handle.rows[0]["neurons"] == []
        assert any("setTimeout(step" in s for s in sent), (
            "the removal's refocus must retry until the re-render settles"
        )

    def test_suggestion_overlay_ignores_the_refresh_empty_reset(
        self, store_patch_for_component, monkeypatch
    ):
        """A pick's synthetic field reset must not swap the held query for history.

        An add pick clears the cell's field (the chip entry's own behavior), and
        a removal pick's refresh remounts the cell — either way a synthetic
        empty ``input-value`` arrives. Read as typing, that would clear the
        held query, so the reset is ignored while a query is held.
        """
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(_config, "get_auto_suggest_enabled", lambda: True)
        monkeypatch.setattr(handle, "_suggest_neurons", lambda text: [
            ("PPL101", "type"), ("PPL102", "type")])
        monkeypatch.setattr(handle, "_refocus_neuron_cell", lambda row_id: None)
        renders = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript",
            lambda js: renders.append(js),
        )

        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "PPL1"}))
        assert handle._suggest_query == "PPL1"
        handle._commit_neuron_suggestion(0, "PPL101")
        renders.clear()
        # The reset the chip entry's field-clear reports, one round trip later.
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": ""}))
        assert renders == []
        assert handle._suggest_query == "PPL1"

        # On a slow socket the chip entry's synthetic value event lands after
        # the coarse window has lapsed; the exact-match map still swallows it
        # (and arms the window for the trailing field clear), so the held query
        # is not swapped for a fresh search of the picked name.
        handle._simulate_until = 0.0
        handle._suggest_ignore_reset_until = 0.0
        handle._simulated_values["PPL101"] = time.time() + 4.0
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "PPL101"}))
        assert handle._suggest_query == "PPL1"
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": ""}))
        assert renders == []
        assert handle._suggest_query == "PPL1"
        assert handle._suggest_ignore_reset_until > time.time()

        # Once the window passes, clearing the field really is a user action.
        handle._suggest_ignore_reset_until = 0.0
        handle._simulate_until = 0.0
        handle._recent_neuron_history = lambda: [("l-LNv", "type")]
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": ""}))
        assert render_call(renders[-1])[2] is True
        assert handle._suggest_query is None

    def test_suggestion_overlay_click_and_escape_toggle_the_list(
        self, store_patch_for_component, monkeypatch
    ):
        """A press on the focused cell closes the list; the next one reopens it.

        Mirrors the query box's click-to-toggle: the dismissal click never
        reopens in the same breath, it ends the held query, and the following
        press offers the history list.
        """
        import ui.config as _config

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(_config, "get_show_history_enabled", lambda: True)
        monkeypatch.setattr(
            handle, "_recent_neuron_history", lambda: [("l-LNv", "type")],
        )
        monkeypatch.setattr(handle, "_suggest_neurons", lambda text: [
            ("PPL101", "type"), ("PPL102", "type")])
        press = SimpleNamespace(args=None)
        renders = []
        monkeypatch.setattr(
            handle.table.client, "run_javascript",
            lambda js: renders.append(js),
        )

        handle.on_neuron_focus(SimpleNamespace(args={"id": 0}))
        handle.on_neuron_suggest(SimpleNamespace(args={"id": 0, "text": "PPL1"}))
        assert handle._suggest_visible is True
        handle._on_suggest_toggle(press)
        assert handle._suggest_visible is False
        assert handle._suggest_toggled_off is True

        # A second press reopens, and it is the history list (the hold ended).
        handle._on_suggest_toggle(press)
        assert handle._suggest_visible is True
        assert handle._suggest_toggled_off is False
        assert handle._suggest_query is None
        assert render_call(renders[-1])[2] is True

        # A close (blur/commit) is not a dismissal, so a press after it does
        # not reopen the list on its own: refocusing the cell is what does that.
        handle._close_suggest_overlay()
        renders.clear()
        handle._on_suggest_toggle(press)
        assert renders == []
        assert handle._suggest_visible is False

    def test_suggestion_overlay_listeners_survive_the_socket_payload(
        self, store_patch_for_component, monkeypatch
    ):
        """The overlay's listeners are reachable over the real event path.

        Calling the handlers directly cannot catch a bad wire shape, and
        NiceGUI's own normalization is strict: ``Client.handle_event`` iterates
        ``msg['args']``, so an argument-less emit raises before the handler runs
        and the click-to-toggle dies silently. Every listener therefore sends
        exactly one JSON argument, which is what this drives here.
        """
        from ui.components import layer_style_editor as editor_module

        client, handle = build_editor(store_patch_for_component)
        monkeypatch.setattr(
            handle, "_recent_neuron_history", lambda: [("l-LNv", "type")],
        )
        monkeypatch.setattr(handle, "_suggest_neurons", lambda text: [
            ("PPL101", "type")])
        overlay = handle._suggest_overlay
        assert handle._suggest_pick_listener_id
        assert handle._suggest_toggle_listener_id
        # The renderer has no argument-less branch left to send.
        assert "args: [JSON.stringify(value)]" in editor_module._SUGGESTION_JS

        client.handle_event({
            "id": overlay.id,
            "listener_id": handle._suggest_pick_listener_id,
            "args": [json.dumps("PPL101")],
        })
        assert handle.rows[0]["neurons"] == []

        # Focusing a cell is what names the row the overlay belongs to.
        handle._suggest_row = 0
        client.handle_event({
            "id": overlay.id,
            "listener_id": handle._suggest_pick_listener_id,
            "args": [json.dumps("PPL101")],
        })
        assert handle.rows[0]["neurons"] == ["PPL101"]

        # The very same message deselects, because the row is now marked: one
        # wire shape carries both directions of the toggle.
        client.handle_event({
            "id": overlay.id,
            "listener_id": handle._suggest_pick_listener_id,
            "args": [json.dumps("PPL101")],
        })
        assert handle.rows[0]["neurons"] == []

        # A blank payload matches no chip, so it cannot remove one.
        handle.rows[0]["neurons"] = ["PPL101"]
        client.handle_event({
            "id": overlay.id,
            "listener_id": handle._suggest_pick_listener_id,
            "args": [json.dumps("  ")],
        })
        assert handle.rows[0]["neurons"] == ["PPL101"]

        handle._suggest_visible = True
        client.handle_event({
            "id": overlay.id,
            "listener_id": handle._suggest_toggle_listener_id,
            "args": [json.dumps(True)],
        })
        assert handle._suggest_visible is False
        assert handle._suggest_toggled_off is True

        # The shape the fix replaced: NiceGUI rejects it before dispatch.
        with pytest.raises(TypeError):
            client.handle_event({
                "id": overlay.id,
                "listener_id": handle._suggest_toggle_listener_id,
                "args": None,
            })

    def test_suggestion_overlay_script_marks_holds_and_dismisses(
        self, store_patch_for_component
    ):
        """The client renderer carries the synced affordances."""
        from ui.components import layer_style_editor as editor_module

        js = editor_module._SUGGESTION_JS
        # A row already in the cell is marked, not removed.
        assert "drocat-suggest-added" in js
        assert 'drocat-suggest-check' in js
        # Rows are raw DOM built from dataset/user text, so they are escaped.
        assert "drocatSuggestEsc" in js
        # A keyboard pick advances the highlight into the rebuilt list.
        assert "__drocatSuggestPending" in js
        # Escape closes the list, and a press on the focused cell toggles it.
        assert "'Escape'" in js
        assert "__TOGGLE_LID__" in js
        assert "drocatSuggestToggle" in js
        # Only a press that closes a visible list swallows the native click, so
        # clicking the focused cell still focuses and text-selects as usual.
        assert "toggle(true)" in js
        assert "toggle(false)" in js
        # Chip presses keep their own meaning and never toggle the list.
        assert "q-chip" in js
        # The rows are exposed as a listbox of options, and because focus never
        # leaves the cell's input the combobox state is written there: the
        # highlight is otherwise visual only and invisible to a screen reader.
        assert "'listbox'" in js
        assert 'role="option"' in js
        assert "aria-selected" in js
        assert "aria-multiselectable" in js
        assert "aria-activedescendant" in js
        assert "aria-expanded" in js
        # The tick repeats aria-selected and the mouse-only history prune has no
        # keyboard meaning, so neither may add to an option's accessible name.
        assert 'aria-hidden="true">check<' in js
        assert 'drocat-suggest-remove" aria-hidden="true"' in js
        # The delegated capture listener acts only on ITS OWN overlay: the
        # shared classes also dress the query box's rows and history prune,
        # and an unscoped capture-phase stopPropagation runs before that
        # button's own click handler — it killed every query-box prune click
        # at the document ("Remove from query history" became a no-op) while
        # this overlay's remove(null) fired harmlessly in its place.
        assert "getElementById('drocat-suggest-overlay')" in js
        assert "o.style.display === 'none'" in js
        assert "o.contains(rem)" in js
        assert "o.contains(t)" in js
        # A marked row says out loud what its next click does, because a marked
        # history row also carries a prune x that means something else. The text
        # arrives as a JSON literal substituted at build time.
        assert "__drocatMarkedTitle" in js
        assert "__MARKED_TITLE_JSON__" in js
        assert 'title="' in js

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
        # Below threshold: no debounce is armed and the gate status is shown.
        assert armed == []
        assert handle._autosave_due is None
        assert "filled rows" in (handle.status_label.text or "")

        # At exactly the threshold, auto-save sets the debounce deadline.
        handle.rows[AUTOSAVE_MIN_NON_EMPTY_ROWS - 1] = {
            "layer": str(AUTOSAVE_MIN_NON_EMPTY_ROWS), "neurons": ["n5"], **fill,
        }
        assert handle._non_empty_row_count() == AUTOSAVE_MIN_NON_EMPTY_ROWS
        armed.clear()
        handle.schedule_autosave()
        assert handle._autosave_due is not None
        # The tick timer is the only ui.timer the editor ever creates.
        assert armed == []

    def test_suggestion_settings_gate_typing_overlay(
        self, store_patch_for_component, monkeypatch
    ):
        """Unchecking Input Auto-Suggestion hides the type-ahead suggestions.

        A non-empty query must close the overlay instead of rendering the
        dataset suggestion list, but the empty-field history list is unaffected.
        """
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

    def test_table_row_dicts_carry_neuron_options_only(self, store_patch_for_component):
        client, handle = build_editor(store_patch_for_component)
        handle.set_rows(
            [{"layer": "1", "neuron": "a"}, {"layer": "2", "neuron": "b"}], name="x"
        )
        row_dicts = handle._row_dicts()
        # The layer cell is a bare numeric input now: no option list is attached.
        assert "layer_opts" not in row_dicts[0]
        assert row_dicts[0]["neuron_options"] == ["a"]
        assert row_dicts[1]["neuron_options"] == ["b"]

    def test_body_slot_layer_cell_is_numeric_with_enter_move(
        self, store_patch_for_component
    ):
        """The layer cell is a bare number input and Enter walks down a column.

        The old dropdown only offered contiguous layers; discrete numbers are
        legitimate now. Enter on a single-value cell commits and moves to the
        same column of the next row; the neuron cell moves only when its field
        is empty (Enter with text adds a chip instead). The move is requested
        by dispatching a ``drocat-enter-move`` DOM event — Vue's production
        render proxy hides ``window`` from template expressions, so the
        templates cannot call the helper directly.
        """
        from ui.components.layer_style_editor import _BODY_SLOT
        assert 'col.name === \'layer\'' in _BODY_SLOT
        assert '<q-input v-model="props.row.layer"' in _BODY_SLOT
        assert "type=\"number\"" in _BODY_SLOT
        assert _BODY_SLOT.count("drocat-enter-move") == 2  # layer + color cells
        # The neuron cell's Enter-move lives in the JS block (capture phase):
        # QSelect consumes the Enter keydown at its input before any template
        # bubble handler, and a capture binding on the QSelect itself raced
        # Quasar's own Enter handling (chip add). The JS rule moves only when
        # the field is empty and no suggestion row is highlighted.
        from ui.components.layer_style_editor import _SUGGESTION_JS
        assert "addEventListener('drocat-enter-move'" in _SUGGESTION_JS
        assert "drocatNeuronEnterNav" in _SUGGESTION_JS
        assert "drocat-suggest-active" in _SUGGESTION_JS

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
