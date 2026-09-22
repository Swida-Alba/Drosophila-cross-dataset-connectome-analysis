"""Tests for the lite custom grouper (component, serializer, gate wiring).

Covers the refined plan contract:
- canonical LabelMapper overall-JSON as the single format (format
  uniformity): the UI exports it, reloads it, and the real LabelMapper
  reads it directly — including validate_datasets for cross-dataset runs;
- rectangular export: every selected dataset becomes a key of every role
  side (empty groups [] included), blank labels auto-name to Group_N;
- reload tolerance: legacy std_label, side precedence, unknown keys;
- selector gate: inline sentinel expands the board, other choices collapse;
- require_names (Cross-Dataset) blocks blank labels; other tabs auto-name;
- run resolution: empty inline board blocks, valid board exports + records
  the group history, presets resolve to their exported file.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.components import custom_grouper as cg  # noqa: E402
from ui.components import mapping_editor as me  # noqa: E402
import ui.group_history as gh  # noqa: E402

DS_A = "male-cns:v0.9"
DS_B = "hemibrain:v1.2.1"


# =============================================================================
# Serializer (pure)
# =============================================================================
class TestCanonicalSerializer:
    def test_auto_names_blank_labels(self):
        rows = [{"name": "", "cells": {DS_A: ["aMe12"]}},
                {"name": "clock", "cells": {DS_A: ["DN1p"]}}]
        data = cg.to_canonical_dict(rows, [DS_A])
        assert data["source_mapping"]["custom_label"] == ["Group_1", "clock"]

    def test_every_dataset_materialized_with_rectangular_groups(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12"]}}]
        data = cg.to_canonical_dict(rows, [DS_A, DS_B])
        for side in ("source_mapping", "target_mapping"):
            assert data[side][DS_A] == [["aMe12"]]
            # DS_B has no members but MUST exist with one (empty) group per
            # label so LabelMapper.validate_datasets passes.
            assert data[side][DS_B] == [[]]

    def test_format_and_meta_tags(self):
        data = cg.to_canonical_dict(
            [{"name": "g", "cells": {DS_A: ["x"]}}], [DS_A], origin="inline")
        assert data["format"] == cg.FORMAT_TAG
        assert data["groups_meta"]["origin"] == "inline"
        assert data["groups_meta"]["updated_at"]

    def test_source_and_target_roles_identical(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"]}}]
        data = cg.to_canonical_dict(rows, [DS_A])
        assert data["source_mapping"] == data["target_mapping"]
        assert "intermediate_mapping" not in data

    def test_roundtrip_identity(self):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"], DS_B: ["aMe12"]}},
                {"name": "", "cells": {DS_A: [], DS_B: ["DN1p"]}}]
        data = cg.to_canonical_dict(rows, [DS_A, DS_B])
        back = cg.from_canonical_dict(data)
        assert [r["name"] for r in back] == ["aMe", "Group_2"]
        assert back[0]["cells"][DS_A] == ["aMe12", "aMe10"]
        assert back[0]["cells"][DS_B] == ["aMe12"]
        assert back[1]["cells"][DS_A] == []
        assert back[1]["cells"][DS_B] == ["DN1p"]

    def test_datasets_with_members_ignores_empty_padding_columns(self):
        rows = [
            {"name": "aMe", "cells": {DS_A: ["aMe12"], DS_B: []}},
            {"name": "clock", "cells": {DS_A: [], DS_B: ["DN1p"]}},
        ]
        assert cg.datasets_with_members(rows) == [DS_A, DS_B]
        assert cg.datasets_with_members([
            {"name": "empty", "cells": {DS_A: [], DS_B: []}}
        ]) == []

    def test_reload_legacy_std_label_and_side_precedence(self):
        legacy = {
            "source_mapping": {"std_label": ["grp"], DS_A: [["aMe12"]]},
        }
        rows = cg.from_canonical_dict(legacy)
        assert rows == [{"name": "grp", "cells": {DS_A: ["aMe12"]}}]
        # source wins when several sides exist
        both = {
            "source_mapping": {"custom_label": ["s"], DS_A: [["a"]]},
            "target_mapping": {"custom_label": ["t"], DS_A: [["b"]]},
        }
        assert cg.from_canonical_dict(both)[0]["name"] == "s"

    def test_reload_rejects_unusable_payload(self):
        with pytest.raises(ValueError):
            cg.from_canonical_dict({"nothing": 1})
        with pytest.raises(ValueError):
            cg.from_canonical_dict({"source_mapping": {DS_A: [["x"]]}})


# =============================================================================
# Format uniformity: the exported file loads through the REAL LabelMapper
# =============================================================================
class TestLabelMapperEquivalence:
    def test_exported_file_loads_directly(self, tmp_path):
        rows = [{"name": "aMe", "cells": {DS_A: ["aMe12", "aMe10"], DS_B: ["aMe12"]}},
                {"name": "clock", "cells": {DS_A: ["DN1p"], DS_B: []}}]
        path = tmp_path / "inline.json"
        path.write_text(json.dumps(cg.to_canonical_dict(rows, [DS_A, DS_B])),
                        encoding="utf-8")

        from comparison.label_mapper import LabelMapper
        mapper = LabelMapper(overall_mapping_json=str(path))
        for role in ("source", "target"):
            assert sorted(mapper.get_neurons_for_label("aMe", DS_A, role)) == \
                ["aMe10", "aMe12"]
            assert mapper.get_neurons_for_label("aMe", DS_B, role) == ["aMe12"]
            assert mapper.get_neurons_for_label("clock", DS_A, role) == ["DN1p"]
            assert mapper.get_neurons_for_label("clock", DS_B, role) == []
        # Cross-dataset guard: both datasets present in both roles.
        mapper.validate_datasets([DS_A, DS_B], role="both")

    def test_auto_named_file_equivalent_to_named_file(self, tmp_path):
        """Blank labels auto-name; structurally identical to named presets."""
        rows = [{"name": "", "cells": {DS_A: ["aMe12"]}}]
        auto = cg.to_canonical_dict(rows, [DS_A])
        named = cg.to_canonical_dict(
            [{"name": "Group_1", "cells": {DS_A: ["aMe12"]}}], [DS_A])
        assert auto["source_mapping"] == named["source_mapping"]
        from comparison.label_mapper import LabelMapper
        m_auto = LabelMapper(overall_mapping_json=None,
                             source_mapping_dict={DS_A: [["aMe12"]]},
                             source_labels=["Group_1"])
        path = tmp_path / "auto.json"
        path.write_text(json.dumps(auto), encoding="utf-8")
        m_file = LabelMapper(overall_mapping_json=str(path))
        assert (m_file.get_neurons_for_label("Group_1", DS_A, "source") ==
                m_auto.get_neurons_for_label("Group_1", DS_A, "source") ==
                ["aMe12"])


# =============================================================================
# Handle (JS-free state)
# =============================================================================
class TestLiteGroupHandle:
    def test_add_remove_move(self):
        h = cg.LiteGroupHandle()
        h.add_row("a", {DS_A: ["1"]})
        h.add_row("b", {DS_A: ["2"]})
        h.add_row("c", {DS_A: ["3"]})
        h.move_row(0, 1)
        assert [r["name"] for r in h.rows] == ["b", "a", "c"]
        h.remove_row(1)
        assert [r["name"] for r in h.rows] == ["b", "c"]

    def test_upsert_covers_same_label(self):
        h = cg.LiteGroupHandle()
        h.add_row("aMe", {DS_A: ["aMe12"]})
        idx = h.upsert_row("aMe", {DS_A: ["aMe9"], DS_B: ["aMe12_R"]})
        assert idx == 0 and len(h.rows) == 1
        assert h.rows[0]["cells"][DS_A] == ["aMe9"]
        assert h.rows[0]["cells"][DS_B] == ["aMe12_R"]

    def test_saved_row_reuses_blank_placeholder_only(self):
        h = cg.LiteGroupHandle()
        h.add_row()
        assert h.load_saved_row("ortho", {DS_A: ["aMe12"]}) == 0
        assert [row["name"] for row in h.rows] == ["ortho"]

        h = cg.LiteGroupHandle()
        h.add_row("draft")
        assert h.load_saved_row("ortho", {DS_A: ["aMe12"]}) == 1
        assert [row["name"] for row in h.rows] == ["draft", "ortho"]

        h = cg.LiteGroupHandle()
        h.add_row("", {DS_A: ["already typed"]})
        assert h.load_saved_row("ortho", {DS_A: ["aMe12"]}) == 1
        assert [row["name"] for row in h.rows] == ["", "ortho"]

    def test_is_empty(self):
        h = cg.LiteGroupHandle()
        assert h.is_empty()
        h.add_row("x", {DS_A: []})
        assert h.is_empty()
        h.add_row("y", {DS_A: ["1"]})
        assert not h.is_empty()


# =============================================================================
# UI wiring (structural harness tests)
# =============================================================================
@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect history file and inline export dir into tmp."""
    monkeypatch.setattr(gh, "HISTORY_PATH", tmp_path / "group_history.json")
    inline_dir = tmp_path / "_inline"
    monkeypatch.setattr(me, "_INLINE_DIR", inline_dir)
    return tmp_path


class TestSelectorGate:
    def test_button_opens_panel_and_label_mirrors_state(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-gate"))
        with client:
            button, dialog, _resolve = me.custom_grouping_block(
                tab_key="gate", datasets_provider=lambda: [DS_A])
        assert dialog.value is False
        assert "none" in button.text
        # The panel hosts the history loader menu and the board.
        assert dialog.history_menu is not None
        assert dialog.inline_grouper is not None
        with client:
            dialog.open()
        assert dialog.value is True

    def test_open_seeds_single_empty_row_when_empty(self, isolated_store):
        """First open with an empty board seeds exactly one empty row for
        editing; a second open must not stack more empty rows."""
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-seed"))
        with client:
            _button, dialog, _resolve = me.custom_grouping_block(
                tab_key="seed", datasets_provider=lambda: [DS_A])
        grouper = dialog.inline_grouper
        with client:
            dialog.open()
        assert len(grouper.handle.rows) == 1
        assert grouper.handle.rows[0]["name"] == ""
        with client:
            dialog.close()
            dialog.open()
        assert len(grouper.handle.rows) == 1  # no stacking

    def test_button_label_tracks_inline_groups(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-gate2"))
        with client:
            button, dialog, _resolve = me.custom_grouping_block(
                tab_key="gate2", datasets_provider=lambda: [DS_A])
        grouper = dialog.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        with client:
            dialog.open()
            dialog.close()  # closing refreshes the button label
        assert "inline · 1 group(s)" in button.text


class TestRunResolution:
    def _build(self, page_path, require_names=False):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page(page_path))
        with client:
            selector, card, resolve = me.custom_grouping_block(
                tab_key=f"res_{page_path.strip('/').replace('-', '_')}",
                datasets_provider=lambda: [DS_A, DS_B],
                require_names=require_names)
        return client, selector, card, resolve

    def test_none_selection_resolves_no_mapping(self, isolated_store):
        client, selector, _card, resolve = self._build("/res-none")
        with client:
            path, ok = resolve()
        assert ok is True and path is None

    def test_empty_board_resolves_to_no_mapping(self, isolated_store):
        """No preset + empty board = simply no mapping (no run blocking)."""
        client, selector, _card, resolve = self._build("/res-empty")
        with client:
            path, ok = resolve()
        assert ok is True and path is None

    def test_inline_valid_board_exports_and_records_history(self, isolated_store):
        client, selector, card, resolve = self._build("/res-valid")
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"], DS_B: ["aMe12"]})
        grouper.handle.add_row("", {DS_A: ["DN1p"]})  # auto-named
        grouper.resync()
        with client:
            path, ok = resolve()
        assert ok is True and path and Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["source_mapping"]["custom_label"] == ["aMe", "Group_2"]
        assert data["source_mapping"][DS_B][1] == []  # padded empty group
        # History: named label recorded, auto label skipped.
        assert gh.list_recent() == ["aMe"]
        assert gh.get_label("aMe")["members"][DS_A] == ["aMe12"]
        # Recent same-tab exports are KEPT (an in-flight run may still read
        # them); only files older than 24 h are swept on export.
        with client:
            path2, ok2 = resolve()
        assert ok2 and path2 != path
        exports = set(Path(path).parent.glob("*.json"))
        assert {Path(path), Path(path2)} <= exports
        import os, time
        old = Path(path)
        past = time.time() - 25 * 3600
        os.utime(old, (past, past))
        with client:
            path3, ok3 = resolve()
        assert ok3 and not old.exists(), "stale export must be swept"
        assert Path(path2).exists() and Path(path3).exists()

    def test_history_select_loads_group_onto_board(self, isolated_store):
        """The history loader (former preset) loads a recorded group's members
        onto the board; nothing is named or saved manually."""
        gh.record([("ortho", {DS_A: ["aMe12", "aMe10"]})], origin="inline")
        client, _button, dialog, resolve = self._build("/res-history")
        grouper = dialog.inline_grouper
        with client:
            dialog.open()
        # Load via the history menu's load action.
        with client:
            dialog.load_history_group("ortho")
        names = [r["name"] for r in grouper.handle.rows]
        assert names == ["ortho"]
        row = next(r for r in grouper.handle.rows if r["name"] == "ortho")
        assert row["cells"][DS_A] == ["aMe12", "aMe10"]
        # Resolve now exports the loaded group.
        with client:
            path, ok = resolve()
        assert ok and path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["source_mapping"]["custom_label"] == ["ortho"]

    def test_history_removal_requires_confirmation(self, isolated_store):
        """The 'x' opens a confirmation; the label is removed only on confirm."""
        gh.record([("gone", {DS_A: ["aMe12"]})], origin="inline")
        client, _button, dialog, _resolve = self._build("/res-remove")
        with client:
            dialog.open()
        # Requesting removal opens the confirm dialog but does NOT delete yet.
        with client:
            dialog.request_remove_history("gone")
        assert gh.list_recent() == ["gone"]
        # Confirming deletes it from the history store.
        with client:
            dialog.confirm_remove_history()
        assert gh.list_recent() == []

    def test_history_removal_also_removes_query_history(
        self, isolated_store, tmp_path, monkeypatch
    ):
        """Deleting a custom group invalidates its normal input-history chip."""
        import ui.history_store as hs

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        gh.record([("gone", {DS_A: ["aMe12"]})], origin="inline")
        hs.record(["gone"], now="2026-08-11T10:00:00")
        client, _button, dialog, _resolve = self._build("/res-remove-query")
        with client:
            dialog.open()
            dialog.request_remove_history("gone")
            dialog.confirm_remove_history()
        assert gh.list_recent() == []
        assert hs.recent() == []

    def test_orphaned_custom_query_history_is_hidden(
        self, isolated_store, tmp_path, monkeypatch
    ):
        """A removed custom label cannot remain in the input history menu."""
        import ui.history_store as hs
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        gh.record([("test", {DS_A: ["aMe12"]})], origin="inline")
        hs.record(["test"], now="2026-08-11T10:00:00",
                  custom_values=["test"])
        gh.remove_label("test")

        client = Client(page("/res-orphaned-query-history"))
        with client:
            box = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [],
            )
        focus = next(
            listener for listener in box.chip_input._event_listeners.values()
            if listener.type == "focus"
        )
        box.chip_input._handle_event({"listener_id": focus.id, "args": None})

        assert hs.recent() == []
        assert hs.frequent() == []

    def test_custom_label_from_query_history_materializes_mapping(
        self, isolated_store, tmp_path, monkeypatch
    ):
        """A custom chip picked in source/target history remains executable."""
        import ui.history_store as hs
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        gh.record([("ortho", {DS_A: ["aMe12", "aMe10"]})], origin="inline")
        hs.record(["ortho"], now="2026-08-11T10:00:00")

        client = Client(page("/res-query-history-custom"))
        with client:
            source = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [],
            )
            target = neuron_list_input(
                label="Target Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [],
            )
            _button, dialog, resolve = me.custom_grouping_block(
                tab_key="query_history_custom",
                datasets_provider=lambda: [DS_A],
                query_inputs={"source": source, "target": target},
            )

        from types import SimpleNamespace

        def subtree_texts(element):
            values = [getattr(element, "text", "")]
            values.extend(
                text for child in element.default_slot.children
                for text in subtree_texts(child)
            )
            return values

        def pick_custom_history(box):
            focus = next(
                listener for listener in box.chip_input._event_listeners.values()
                if listener.type == "focus"
            )
            box.chip_input._handle_event({"listener_id": focus.id, "args": None})
            item = next(
                element for element in box.suggest_menu.default_slot.children
                if type(element).__name__ == "Item"
                and "ortho" in subtree_texts(element)
            )
            click = next(
                listener for listener in item._event_listeners.values()
                if listener.type == "click"
            )
            click.handler(SimpleNamespace())

        pick_custom_history(source)
        pick_custom_history(target)
        assert source.get_value()[1] == ["ortho"]
        assert target.get_value()[1] == ["ortho"]
        with client:
            path, ok = resolve()

        assert ok and path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["source_mapping"]["custom_label"] == ["ortho"]
        assert data["source_mapping"][DS_A] == [["aMe12", "aMe10"]]
        assert data["target_mapping"][DS_A] == [["aMe12", "aMe10"]]
        assert dialog.inline_grouper.handle.rows[0]["name"] == "ortho"

    def test_custom_history_entry_has_gray_custom_tag(
        self, isolated_store, tmp_path, monkeypatch
    ):
        """Known custom labels are visibly distinguished in query history."""
        import ui.config as cfg
        import ui.history_store as hs
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
        gh.record([("custom_a", {DS_A: ["aMe12"]})], origin="inline")
        hs.record(["custom_a", "ordinary"], now="2026-08-11T10:00:00")

        client = Client(page("/query-history-custom-tag"))
        with client:
            box = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda _text: [],
            )
        focus = next(
            listener for listener in box.chip_input._event_listeners.values()
            if listener.type == "focus"
        )
        box.chip_input._handle_event({"listener_id": focus.id, "args": None})

        def subtree_texts(el):
            values = [getattr(el, "text", "")]
            values.extend(
                text for child in el.default_slot.children
                for text in subtree_texts(child)
            )
            return values

        items = [
            el for el in client.elements.values()
            if type(el).__name__ == "Item"
        ]
        custom_item = next(el for el in items if "custom_a" in subtree_texts(el))
        ordinary_item = next(el for el in items if "ordinary" in subtree_texts(el))
        assert "custom" in subtree_texts(custom_item)
        assert "custom" not in subtree_texts(ordinary_item)

    def test_push_records_group_into_history(self, isolated_store):
        """Pushing a group to a query records it into the history immediately."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input
        client = Client(page("/res-pushhist"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            _b, dialog, _r = me.custom_grouping_block(
                tab_key="pushhist", datasets_provider=lambda: [DS_A],
                query_inputs={"source": source})
        grouper = dialog.inline_grouper
        grouper.handle.add_row("clock", {DS_A: ["DN1p"]})
        grouper.resync()
        assert gh.list_recent() == []
        with client:
            grouper.push_to_query("source", 0)
        assert gh.list_recent() == ["clock"]
        assert gh.get_label("clock")["members"][DS_A] == ["DN1p"]

    def test_require_names_blocks_blank_labels(self, isolated_store):
        client, selector, card, resolve = self._build("/res-names", require_names=True)
        grouper = card.inline_grouper
        grouper.handle.add_row("", {DS_A: ["aMe12"]})
        grouper.resync()
        with client:
            path, ok = resolve()
        assert ok is False and path is None
        # Naming the group makes it pass (the name input is the source of
        # truth; resync collects from the widgets).
        grouper._name_inputs[0].value = "aMe"
        with client:
            path, ok = resolve()
        assert ok is True and path

    def test_dataset_columns_follow_watched_selector(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from nicegui import ui as _ui
        client = Client(page("/grouper-sync"))
        with client:
            ds_sel = _ui.select(options=[DS_A, DS_B], value=DS_A)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="sync",
                datasets_provider=lambda: [ds_sel.value],
                watch_elements=[ds_sel])
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        assert list(grouper._cell_widgets[0].keys()) == [DS_A]
        ds_sel.value = DS_B  # watch fires -> resync re-renders columns
        assert grouper.datasets() == [DS_B]
        assert list(grouper._cell_widgets[0].keys()) == [DS_B]
        # Members of the hidden dataset survive the column change.
        assert grouper.handle.rows[0]["cells"][DS_A] == ["aMe12"]


class TestReloadIntoBoard:
    def test_load_replaces_rows_from_canonical_payload(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        client = Client(page("/grouper-reload"))
        with client:
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="reload", datasets_provider=lambda: [DS_A, DS_B])
        grouper = card.inline_grouper
        payload = cg.to_canonical_dict(
            [{"name": "aMe", "cells": {DS_A: ["aMe12"], DS_B: ["aMe12"]}}],
            [DS_A, DS_B])
        grouper.handle.replace_rows(cg.from_canonical_dict(payload))
        grouper.resync()
        assert grouper.handle.rows[0]["name"] == "aMe"
        assert grouper.handle.rows[0]["cells"][DS_B] == ["aMe12"]
        # Export from the reloaded board reproduces the payload's sides.
        out = grouper.to_canonical()
        assert out["source_mapping"] == payload["source_mapping"]

    def test_embedded_selector_load_adds_nonempty_datasets_and_rectangularizes_rows(
        self, isolated_store
    ):
        from nicegui import Client, ui
        from nicegui.page import page

        selected = None

        def render_dataset_selector():
            nonlocal selected
            selected = ui.select(
                options=[DS_A, DS_B],
                value=[],
                label="Target datasets",
                multiple=True,
            )
            return selected

        client = Client(page("/grouper-embedded-datasets"))
        with client:
            _button, dialog, _resolve = me.custom_grouping_block(
                tab_key="embedded-datasets",
                datasets_provider=lambda: list(selected.value or [])
                if selected is not None else [],
                dataset_selector_renderer=render_dataset_selector,
            )

        assert dialog.dataset_selector is selected
        assert selected.value == []
        rows = cg.from_canonical_dict(cg.to_canonical_dict(
            [
                {"name": "first", "cells": {DS_A: ["aMe12"], DS_B: []}},
                {"name": "second", "cells": {DS_A: [], DS_B: ["DN1p"]}},
            ],
            [DS_A, DS_B],
        ))
        grouper = dialog.inline_grouper
        grouper.load_rows(rows)

        assert selected.value == [DS_A, DS_B]
        assert grouper.datasets() == [DS_A, DS_B]
        assert set(grouper._cell_widgets[0]) == {DS_A, DS_B}
        assert set(grouper._cell_widgets[1]) == {DS_A, DS_B}
        assert grouper._cell_widgets[0][DS_B].get_value()[1] == []
        assert grouper._cell_widgets[1][DS_A].get_value()[1] == []


# =============================================================================
# Workflow: pushing group members into query inputs
# =============================================================================
class TestQueryPushWorkflow:
    def test_push_to_query_adds_union_of_members(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            target = neuron_list_input(label="Target Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push",
                datasets_provider=lambda: [DS_A, DS_B],
                query_inputs={"source": source, "target": target})
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12", "aMe10"],
                                        DS_B: ["aMe12", "aMe9"]})
        grouper.resync()
        with client:
            pushed = grouper.push_to_query("source", 0)
        # The group LABEL is pushed (backend expands it into members).
        assert pushed == ["aMe"]
        assert source.get_value()[1] == ["aMe"]
        assert target.get_value()[1] == []

    def test_push_twice_does_not_duplicate(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push2"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push2",
                datasets_provider=lambda: [DS_A],
                query_inputs={"source": source})
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        with client:
            grouper.push_to_query("source", 0)
            grouper.push_to_query("source", 0)
        assert source.get_value()[1] == ["aMe"]

    def test_push_empty_group_warns_and_adds_nothing(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-push3"))
        with client:
            source = neuron_list_input(label="Source Neurons",
                                       show_filter=False, show_upload=False)
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="push3",
                datasets_provider=lambda: [DS_A],
                query_inputs={"source": source})
        grouper = card.inline_grouper
        grouper.handle.add_row("empty", {DS_A: []})
        grouper.resync()
        with client:
            assert grouper.push_to_query("source", 0) == []
        assert source.get_value()[1] == []


class TestSuggestionAnchoring:
    def test_suggest_menu_targets_its_input_wrapper(self, isolated_store):
        """The custom suggestion menu must carry an explicit target selector
        for its own input wrapper (robust anchoring inside nested rebuilds)."""
        from nicegui import Client
        from nicegui.page import page
        from ui.components.common import neuron_list_input

        client = Client(page("/grouper-anchor"))
        with client:
            box = neuron_list_input(
                label="Source Neurons", show_filter=False, show_upload=False,
                suggestions=lambda text: [("aMe12", "type")])
        menu = box.suggest_menu
        anchor = box.chip_input.parent_slot.parent
        assert menu._props.get("target") == f"#{anchor.html_id}"

    def test_grouper_cell_menus_are_anchored(self, isolated_store):
        from nicegui import Client
        from nicegui.page import page

        client = Client(page("/grouper-anchor2"))
        with client:
            _selector, card, _resolve = me.custom_grouping_block(
                tab_key="anchor2", datasets_provider=lambda: [DS_A])
        grouper = card.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"]})
        grouper.resync()
        cell = grouper._cell_widgets[0][DS_A]
        menu = cell.suggest_menu
        anchor = cell.chip_input.parent_slot.parent
        assert menu._props.get("target") == f"#{anchor.html_id}"
        assert menu._props.get("anchor") == "bottom start"
        assert menu._props.get("self") == "top start"
        assert menu._props.get("fit") is True
        assert menu._props.get("max-height") == "240px"


class TestLabelMapperEditorSurface:
    def test_optional_row_action_renders_once_in_first_group(self, isolated_store):
        """Shared editors can add a global action beside query-row actions."""
        from nicegui import Client
        from nicegui import ui
        from nicegui.page import page

        row_actions = []

        def render_save_action():
            ui.button("Save Mapping", icon="save").props(
                "outline no-caps"
            ).classes("drocat-labelmapper-query-action")

        client = Client(page("/grouper-row-action"))
        with client:
            _button, dialog, _resolve = me.custom_grouping_block(
                tab_key="grouper-row-action",
                datasets_provider=lambda: [DS_A],
                row_action_renderers=row_actions,
            )

        row_actions.append(render_save_action)
        grouper = dialog.inline_grouper
        grouper.handle.add_row("first", {DS_A: ["aMe12"]})
        grouper.handle.add_row("second", {DS_A: ["aMe10"]})
        grouper.resync()

        save_buttons = [
            el for el in client.elements.values()
            if getattr(el, "text", None) == "Save Mapping"
        ]
        assert len(save_buttons) == 1
        assert save_buttons[0]._props.get("outline") is True
        assert "drocat-labelmapper-query-action" in save_buttons[0]._classes

    def test_custom_grouper_uses_aligned_dataset_rows_and_viewers(self, isolated_store):
        """Each dataset gets an isolated outlined chip field and index viewer."""
        from nicegui import Client
        from nicegui.page import page

        client = Client(page("/grouper-layout"))
        with client:
            _button, dialog, _resolve = me.custom_grouping_block(
                tab_key="grouper-layout",
                datasets_provider=lambda: [DS_A, DS_B],
                require_names=True,
            )
        grouper = dialog.inline_grouper
        grouper.handle.add_row("aMe", {DS_A: ["aMe12"], DS_B: ["MTe07"]})
        grouper.resync()

        dataset_rows = [
            el for el in client.elements.values()
            if "drocat-labelmapper-dataset-row" in getattr(el, "_classes", set())
        ]
        assert len(dataset_rows) == 2
        for dataset in (DS_A, DS_B):
            cell = grouper._cell_widgets[0][dataset]
            assert cell.chip_input._props.get("outlined") is True
            assert cell.neuron_index_link.text == "See available neurons"
        assert not any(
            getattr(el, "_props", {}).get("icon") in {
                "arrow_upward", "arrow_downward"
            }
            for el in client.elements.values()
        )

    def test_settings_mapping_editor_uses_the_same_member_surface(
        self, isolated_store, tmp_path, monkeypatch
    ):
        """The Settings LabelMapper editor exposes the same viewer/chip UX."""
        from nicegui import Client
        from nicegui.page import page

        import ui.config as cfg

        # Gate the toggles on a redirected file so the run does not depend on
        # (or overwrite) the developer's own ui/local_config.json.
        monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
        cfg.set_auto_suggest_enabled(True)
        cfg.set_show_history_enabled(True)
        # The member cell resolves suggestions through the shared dataset pool
        # helper; stub it so the test proves the wiring, not the data.
        calls = []
        monkeypatch.setattr(
            me, "dataset_suggestions",
            lambda text, dataset=None, search_columns="auto": calls.append(
                (text, dataset, search_columns)
            ) or [("aMe12", "type")],
        )

        client = Client(page("/settings-mapping-layout"))
        with client:
            from nicegui import ui as _ui

            container = _ui.column()
            editor = me.MappingGridEditor("source_mapping")
            editor.create(container, [DS_A, DS_B])
            editor._add_group()
            editor._add_dataset(DS_A)

        cell = editor._cell_widgets[DS_A][0]
        cell.add_values(["aMe12", "aMe10"])
        assert cell.chip_input._props.get("outlined") is True
        assert cell.neuron_index_link.text == "See available neurons"
        assert editor.get_data()[DS_A] == [["aMe12", "aMe10"]]
        assert any(
            "drocat-labelmapper-dataset-row" in getattr(el, "_classes", set())
            for el in client.elements.values()
        )

        # Type-ahead reaches the member cell the same way it reaches the inline
        # grouper's: typing consults the dataset-aware provider for THIS column's
        # dataset and lists its rows.
        for listener in [
            listener for listener in cell.chip_input._event_listeners.values()
            if listener.type == "input"
        ]:
            cell.chip_input._handle_event(
                {"listener_id": listener.id, "args": "aM"})
        assert calls and calls[0] == ("aM", DS_A, "auto")
        assert "aMe12" in [
            element.text for element in client.elements.values()
            if getattr(element, "text", "")
        ]

    def test_both_member_cells_offer_the_same_input_affordances(self):
        """The inline grouper and the Settings grid are one surface in two
        places, so neither may quietly offer less than the other: type-ahead
        from the column's own dataset, and a Recent list whose rows say whether
        that value exists in the column's dataset at all.
        """
        import inspect

        for module in (cg, me):
            source = inspect.getsource(module)
            assert "suggestions=self._cell_suggest(ds)" in source
            assert "history_hint_datasets=lambda ds=ds: [ds]" in source
