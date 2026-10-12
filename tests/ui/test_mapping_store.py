"""Tests for the custom type-mapping store (cache/user_mappings.json) and
the mapping grid editor's data round-trip."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.mapping_store as store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the store at a temp file and clean up after each test."""
    store_dir = tmp_path / "user_mappings"
    monkeypatch.setattr(store, "_store_dir", store_dir)
    monkeypatch.setattr(store, "_store_file", store_dir / "user_mappings.json")
    yield
    import shutil
    shutil.rmtree(store_dir, ignore_errors=True)


SAMPLE = {
    "source_mapping": {
        "custom_label": ["grpA", "grpB"],
        "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]],
        "male-cns:v0.9": [["aMe12"], ["aMe12-like"]],
    },
    "target_mapping": {
        "custom_label": ["tg1"],
        "hemibrain:v1.2.1": [["MBON01"]],
    },
}


class TestStoreCrud:
    def test_empty_store(self):
        assert store.list_mappings() == []
        assert store.get_active_mapping() is None

    def test_save_and_get(self):
        assert store.save_mapping("my map", SAMPLE, "test mapping")
        preset = store.get_mapping("my map")
        assert preset["name"] == "my map"
        assert preset["description"] == "test mapping"
        assert preset["source_mapping"]["custom_label"] == ["grpA", "grpB"]

    def test_update_existing(self):
        store.save_mapping("m", {"source_mapping": {"custom_label": ["a"], "d": [["x"]]}})
        store.save_mapping("m", {"source_mapping": {"custom_label": ["a", "b"], "d": [["x"], ["y"]]}}, "v2")
        preset = store.get_mapping("m")
        assert preset["description"] == "v2"
        assert preset["source_mapping"]["custom_label"] == ["a", "b"]

    def test_rename_and_active_follows(self):
        store.save_mapping("old", SAMPLE)
        store.set_active_mapping("old")
        assert store.rename_mapping("old", "new")
        assert store.get_mapping("old") is None
        assert store.get_mapping("new") is not None
        assert store.get_active_mapping() == "new"

    def test_delete(self):
        store.save_mapping("gone", SAMPLE)
        store.set_active_mapping("gone")
        assert store.delete_mapping("gone")
        assert store.get_mapping("gone") is None
        assert store.get_active_mapping() is None

    def test_invalid_name_rejected(self):
        assert not store.save_mapping("   ", SAMPLE)
        assert not store.rename_mapping("x", "  ")

    def test_persistence_across_instances(self):
        store.save_mapping("persist", SAMPLE)
        store.set_active_mapping("persist")
        # fresh load from disk
        assert store.list_mappings() == ["persist"]
        assert store.get_active_mapping() == "persist"


class TestExport:
    def test_export_file_carries_the_mapping_keys_and_grammar_tag(self):
        store.save_mapping("export me", SAMPLE)
        path = store.mapping_file_path("export me")
        assert path is not None and Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 'format' is what the backend loader gates presets on, so the export
        # a run receives has to declare it (see comparison.label_mapper).
        assert set(data.keys()) == {"format", "source_mapping", "target_mapping"}
        assert data["format"] == "drocat_custom_groups/v1"
        assert "name" not in data and "description" not in data

    def test_export_roundtrip_through_label_mapper(self):
        store.save_mapping("rt", SAMPLE)
        path = store.mapping_file_path("rt")
        from comparison.label_mapper import LabelMapper
        mapper = LabelMapper(overall_mapping_json=path)
        assert mapper.get_label("hemibrain:v1.2.1", "aMe12") == "grpA"
        assert mapper.get_label("male-cns:v0.9", "aMe12-like") == "grpB"
        assert mapper.get_label("hemibrain:v1.2.1", "MBON01") == "tg1"

    def test_export_removed_on_delete_and_rename(self):
        store.save_mapping("old name", SAMPLE)
        old_path = Path(store.mapping_file_path("old name"))
        assert old_path.exists()
        store.rename_mapping("old name", "new name")
        assert not old_path.exists()
        new_path = Path(store.mapping_file_path("new name"))
        assert new_path.exists()
        store.delete_mapping("new name")
        assert not new_path.exists()


class TestValidation:
    def test_valid_mapping_passes(self):
        assert store.validate_mapping(SAMPLE) == []

    def test_group_count_mismatch(self):
        bad = {"source_mapping": {"custom_label": ["a", "b"], "d": [["x"]]}}
        errors = store.validate_mapping(bad)
        assert any("must have 2 groups" in e for e in errors)

    def test_duplicate_group_names(self):
        bad = {"source_mapping": {"custom_label": ["a", "a"], "d": [["x"], ["y"]]}}
        assert any("unique" in e for e in store.validate_mapping(bad))

    def test_missing_label_list(self):
        bad = {"source_mapping": {"d": [["x"]]}}
        assert any("custom_label" in e for e in store.validate_mapping(bad))

    def test_empty_cell_groups_allowed(self):
        # Empty groups are legal (a group may have no neurons in one dataset,
        # like an empty CSV cell); only non-list groups or non-string items
        # are rejected.
        ok = {"source_mapping": {"custom_label": ["a"], "d": [[]]}}
        assert store.validate_mapping(ok) == []
        bad = {"source_mapping": {"custom_label": ["a"], "d": [["x", 42]]}}
        assert any("neuron identifiers" in e for e in store.validate_mapping(bad))


class TestGridEditorDataRoundtrip:
    def test_set_get_roundtrip(self):
        from ui.components.mapping_editor import MappingGridEditor
        editor = MappingGridEditor("source_mapping")
        editor.set_data(SAMPLE["source_mapping"])
        data = editor.get_data()
        assert data == SAMPLE["source_mapping"]

    def test_empty_editor(self):
        from ui.components.mapping_editor import MappingGridEditor
        editor = MappingGridEditor("source_mapping")
        editor.set_data({})
        assert editor.is_empty()
        assert editor.get_data() == {"custom_label": []}

    def test_cells_parse_commas(self):
        from ui.components.mapping_editor import MappingGridEditor
        editor = MappingGridEditor("source_mapping")
        editor.set_data({"custom_label": ["g"], "ds": [["a", "b"]]})
        editor._cells["ds"][0] = " a , b , "
        data = editor.get_data()
        assert data["ds"] == [["a", "b"]]
