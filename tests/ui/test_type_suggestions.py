"""Tests for dataset auto-suggestion pools and staged matching.

Covers strict type-first prefixes, cross-column prefix expansion, the final
case-insensitive substring fallback, bodyId -> instance hints, and the local
pool sources (cache neuron_index.parquet first, dataset tables as fallback,
mtime-keyed caching).
"""

import polars as pl
import pytest

from ui.type_suggestions import (
    _POOL_CACHE, _folder_pools, dataset_aware_suggestions, entry_hint,
    filter_candidate_entries, get_dataset_pools, match_suggestions,
    suggestion_pool,
)

# Synthetic pools mirroring a cache neuron_index (type/instance/bodyId).
POOLS = {
    "type": [("APL", "type"), ("APL2", "type"), ("aMe12", "type"),
             ("MBON01", "type")],
    "instance": [("APL_1", "instance"), ("aMe12_1", "instance")],
    "bodyId": [("5813012345", "aMe12_1"), ("5813012346", "bodyId"),
               ("5813012399", "bodyId")],
}


@pytest.fixture(autouse=True)
def _clear_pool_cache():
    """Pools are cached in a module-level dict; keep tests independent."""
    _POOL_CACHE.clear()
    yield
    _POOL_CACHE.clear()


@pytest.fixture
def local_dirs(tmp_path, monkeypatch):
    """Point the pool sources at a temporary indexes/datasets pair."""
    indexes = tmp_path / "neuron_indexes"
    datasets = tmp_path / "datasets"
    indexes.mkdir(parents=True)
    datasets.mkdir(parents=True)
    monkeypatch.setattr("ui.type_suggestions._INDEX_DIR", indexes)
    monkeypatch.setattr("ui.type_suggestions._DATASETS_DIR", datasets)
    return indexes, datasets


class TestMatchSuggestions:
    def test_empty_input_or_pools(self):
        assert match_suggestions("", POOLS) == []
        assert match_suggestions("   ", POOLS) == []
        assert match_suggestions("aMe", {}) == []

    def test_numeric_input_auto_scopes_to_bodyid(self):
        out = match_suggestions("58130123", POOLS)
        assert [v for v, _ in out] == ["5813012345", "5813012346", "5813012399"]
        # The hint is the instance for known bodyIds, else the column name.
        assert out[0][1] == "aMe12_1"
        assert out[1][1] == "bodyId"

    def test_string_input_type_first(self):
        assert match_suggestions("AP", POOLS) == [("APL", "type"), ("APL2", "type")]

    def test_case_sensitive_prefix(self):
        """Strict prefixes preserve capitalization; fallback substrings do not."""
        assert match_suggestions("aMe", POOLS) == [("aMe12", "type")]
        assert match_suggestions("ame", POOLS) == [
            ("aMe12", "type"), ("aMe12_1", "instance"),
        ]
        assert match_suggestions("Me12", POOLS) == [
            ("aMe12", "type"), ("aMe12_1", "instance"),
        ]
        assert match_suggestions("AMMC", POOLS) == []
        assert match_suggestions("APL", POOLS)[0] == ("APL", "type")
        assert match_suggestions("Apl2", POOLS) == [("APL2", "type")]

    def test_type_prefix_is_strictly_first(self):
        """A type prefix suppresses every less-specific column candidate."""
        pools = {
            "type": [("aMe12", "type"), ("aMeClock", "type")],
            "instance": [("aMe12_L", "instance")],
            "bodyId": [("aMe12_body", "aMe12_L")],
            "flywireType": [("aMe12_fw", "flywireType")],
        }
        assert match_suggestions("aMe", pools) == [
            ("aMe12", "type"), ("aMeClock", "type"),
        ]

    def test_full_viewer_prefix_mode_keeps_every_search_column(self):
        pools = {
            "type": [("aMe12", "type")],
            "instance": [("aMe12_L", "instance")],
            "flywireType": [("aMe12_fw", "flywireType")],
        }
        assert match_suggestions("aMe", pools) == [("aMe12", "type")]
        assert match_suggestions(
            "aMe", pools, limit=None, all_prefix_matches=True
        ) == [
            ("aMe12", "type"),
            ("aMe12_L", "instance"),
            ("aMe12_fw", "flywireType"),
        ]

    def test_full_viewer_mode_keeps_numeric_bodyid_guard(self):
        pools = {
            "type": [("Cell123", "type")],
            "instance": [("123_instance", "instance")],
            "bodyId": [("123", "bodyId")],
        }
        assert match_suggestions(
            "123", pools, limit=None, all_prefix_matches=True
        ) == [("123", "bodyId")]

    def test_other_column_prefixes_expand_only_after_type_prefixes_fail(self):
        """BodyId/instance/extra prefixes are returned before substrings."""
        pools = {
            "type": [("CellAlpha", "type")],
            "instance": [("aMe12_L", "instance")],
            "bodyId": [("aMe12_body", "aMe12_L")],
            "flywireType": [("aMe12_fw", "flywireType")],
        }
        assert match_suggestions("aMe", pools) == [
            ("aMe12_body", "aMe12_L"),
            ("aMe12_L", "instance"),
            ("aMe12_fw", "flywireType"),
        ]

    def test_prefix_stage_wins_over_substring_stage(self):
        """A non-type prefix suppresses a type substring fallback."""
        pools = {
            "type": [("CellAlpha", "type")],
            "instance": [("alpha_instance", "instance")],
        }
        assert match_suggestions("alpha", pools) == [("alpha_instance", "instance")]

    def test_substring_fallback_is_last_and_type_preferred(self):
        """Only when every prefix fails do case-insensitive substrings appear."""
        pools = {
            "type": [("CellAlpha", "type"), ("BetaCell", "type")],
            "instance": [("xAlpha_instance", "instance")],
            "bodyId": [("9900", "Alpha_1")],
        }
        assert match_suggestions("Alpha", pools) == [
            ("CellAlpha", "type"), ("xAlpha_instance", "instance"),
        ]

    def test_prefix_miss_uses_case_insensitive_substring(self):
        assert match_suggestions("Me12_", POOLS) == [("aMe12_1", "instance")]
        assert match_suggestions("AME", POOLS) == [
            ("aMe12", "type"), ("aMe12_1", "instance"),
        ]

    def test_filter_candidate_entries_narrows_case_sensitively(self):
        candidates = [
            ("aMe12", "type"), ("aMe10", "type"),
            ("aMap", "type"), ("Ame12", "type"),
        ]
        assert filter_candidate_entries("aM", candidates) == candidates[:3]
        assert filter_candidate_entries("aMe1", candidates) == [
            ("aMe12", "type"), ("aMe10", "type"),
        ]
        assert filter_candidate_entries("am", candidates) == []

    def test_expand_only_when_no_type_match_and_auto(self):
        # "aMe12_" matches no type -> the range expands to instance.
        assert match_suggestions("aMe12_", POOLS) == [("aMe12_1", "instance")]
        # "aMe12" itself DOES match a type -> type stays first and alone.
        assert match_suggestions("aMe12", POOLS) == [("aMe12", "type")]

    def test_expansion_order_instance_then_bodyid_then_extra(self):
        extras = dict(POOLS)
        extras["bodyId"] = extras["bodyId"] + [("FL5813", "bodyId")]
        extras["flywireType"] = [("FLaMe12_x", "flywireType")]
        out = match_suggestions("FL", extras)
        # no type match -> instance, then bodyId, then extra type columns
        assert out == [("FL5813", "bodyId"), ("FLaMe12_x", "flywireType")]

    def test_explicit_scope_restricts_column(self):
        assert match_suggestions("5813", POOLS, "type") == []
        assert match_suggestions("mbon", POOLS, "instance") == []
        assert match_suggestions("mbon", POOLS, "bodyId") == []
        assert match_suggestions("APL", POOLS, "type") == [("APL", "type"), ("APL2", "type")]

    def test_explicit_scope_uses_prefix_then_substring_in_that_column(self):
        assert match_suggestions("PL_", POOLS, "instance") == [("APL_1", "instance")]
        assert match_suggestions("Me12_", POOLS, "instance") == [("aMe12_1", "instance")]

    def test_auto_bodyid_entries_keep_instance_hints(self):
        out = match_suggestions("58130123", POOLS)
        assert out == [
            ("5813012345", "aMe12_1"),
            ("5813012346", "bodyId"),
            ("5813012399", "bodyId"),
        ]

    def test_scope_case_insensitive_and_unknown_falls_back_to_auto(self):
        assert match_suggestions("APL", POOLS, "TYPE") == [("APL", "type"), ("APL2", "type")]
        assert match_suggestions("AP", POOLS, "bogus") == [("APL", "type"), ("APL2", "type")]

    def test_limit(self):
        assert len(match_suggestions("5813", POOLS, limit=2)) == 2
        assert len(match_suggestions("AP", POOLS, limit=1)) == 1

    def test_unbounded_limit_keeps_all_continuation_candidates(self):
        pools = {
            "type": [(f"aType{i:02d}", "type") for i in range(60)],
        }
        assert len(match_suggestions("a", pools)) == 50
        all_entries = match_suggestions("a", pools, limit=None)
        assert len(all_entries) == 60
        assert filter_candidate_entries("aType59", all_entries) == [
            ("aType59", "type"),
        ]


class TestEntryHint:
    def test_type_instance_bodyid_priority(self):
        assert entry_hint("APL", POOLS) == "type"
        assert entry_hint("APL_1", POOLS) == "instance"
        assert entry_hint("5813012345", POOLS) == "bodyId"

    def test_unknown_value(self):
        assert entry_hint("nope", POOLS) is None
        assert entry_hint("", POOLS) is None


class TestPoolSources:
    def test_index_parquet_pools(self, local_dirs):
        """neuron_indexes/<folder>/neuron_index.parquet builds
        type/instance/bodyId pools (bodyId hints = the corresponding
        instance)."""
        indexes, _ = local_dirs
        folder = "test_v1"
        (indexes / folder).mkdir()
        pl.DataFrame({
            "bodyId": [5813012345, 5813012346],
            "type": ["APL", "MBON01"],
            "instance": ["APL_1", "MBON01_1"],
            "flywireType": ["FW_APL", "FW_MBON"],
            "hemilineage": ["AL", "MB"],
            "last_fetched": ["2026-08-12", "2026-08-12"],
            "roiInfo": ["large payload", "large payload"],
        }).write_parquet(indexes / folder / "neuron_index.parquet")

        pools = _folder_pools(folder)
        assert pools["type"] == [("APL", "type"), ("MBON01", "type")]
        assert pools["bodyId"] == [
            ("5813012345", "APL_1"), ("5813012346", "MBON01_1"),
        ]
        assert pools["flywireType"] == [
            ("FW_APL", "flywireType"), ("FW_MBON", "flywireType"),
        ]
        assert "hemilineage" not in pools
        assert "last_fetched" not in pools
        assert "roiInfo" not in pools

    def test_table_fallback_pools(self, local_dirs):
        """datasets/<folder> neuron tables (no index file) build pools
        for type/class taxonomy columns, not arbitrary metadata."""
        _, datasets = local_dirs
        folder = "test_v1"
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "test_neuron_df.csv").write_text(
            "bodyId,type,instance,flywireType,class,hemilineage\n"
            "1,APL,APL_1,FW_APL,Kenyon,AL\n"
            "2,MBON01,MBON01_1,FW_MBON,Projection,MB\n"
        )

        pools = _folder_pools(folder)
        assert pools["type"] == [("APL", "type"), ("MBON01", "type")]
        assert pools["flywireType"] == [("FW_APL", "flywireType"), ("FW_MBON", "flywireType")]
        assert pools["class"] == [("Kenyon", "class"), ("Projection", "class")]
        assert "hemilineage" not in pools
        assert pools["bodyId"] == [("1", "APL_1"), ("2", "MBON01_1")]

    def test_table_fallback_accepts_flywire_body_id_alias(self, local_dirs):
        _, datasets = local_dirs
        folder = "flywire_test_v1"
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "flywire_test_neuron_df.csv").write_text(
            "root_id,type,instance,super_class\n"
            "720575940000000001,aMe12,aMe12_L,optic\n"
        )

        pools = _folder_pools(folder)
        assert pools["bodyId"] == [
            ("720575940000000001", "aMe12_L"),
        ]

    def test_cache_index_preferred_over_table(self, local_dirs):
        indexes, datasets = local_dirs
        folder = "test_v1"
        (indexes / folder).mkdir()
        pl.DataFrame({
            "bodyId": [1], "type": ["APL"], "instance": ["APL_1"],
        }).write_parquet(indexes / folder / "neuron_index.parquet")
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "x_neuron_df.csv").write_text(
            "bodyId,type,instance\n1,MBON01,MBON01_1\n"
        )

        # Index values stay first, while the table fills missing names.
        assert _folder_pools(folder)["type"] == [
            ("APL", "type"), ("MBON01", "type"),
        ]

    def test_sparse_cache_index_falls_back_to_table_names(self, local_dirs):
        """A metadata-empty index must not hide the dataset's type table."""
        indexes, datasets = local_dirs
        folder = "test_v1"
        (indexes / folder).mkdir()
        pl.DataFrame({
            "bodyId": [1, 2], "type": ["", ""], "instance": ["", ""],
        }).write_parquet(indexes / folder / "neuron_index.parquet")
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "x_neuron_df.csv").write_text(
            "bodyId,type,instance\n"
            "1,aMe12,aMe12_L\n"
            "2,aMe10,aMe10_R\n"
        )

        pools = _folder_pools(folder)
        assert pools["type"] == [("aMe10", "type"), ("aMe12", "type")]
        assert pools["instance"] == [
            ("aMe10_R", "instance"), ("aMe12_L", "instance"),
        ]

    def test_newer_metadata_replaces_stale_rich_index(self, local_dirs):
        """A refreshed source table must remove stale indexed suggestions."""
        import os

        indexes, datasets = local_dirs
        folder = "test_v1"
        index_dir = indexes / folder
        index_dir.mkdir()
        index = index_dir / "neuron_index.parquet"
        pl.DataFrame({
            "bodyId": ["1"],
            "type": ["old-type"],
            "instance": ["old-type_L"],
            "flywireType": ["old-flywire"],
        }).write_parquet(index)
        source_dir = datasets / folder
        source_dir.mkdir()
        table = source_dir / "test_allneurons_neuron_df.csv"
        table.write_text(
            "bodyId,type,instance,flywireType\n"
            "1,new-type,new-type_L,new-flywire\n",
            encoding="utf-8",
        )
        os.utime(index, ns=(1_000, 1_000))
        os.utime(table, ns=(2_000, 2_000))

        pools = _folder_pools(folder)
        assert pools["type"] == [("new-type", "type")]
        assert pools["flywireType"] == [("new-flywire", "flywireType")]

    def test_partial_cache_index_is_augmented_for_aMe_prefix(self, local_dirs):
        """An index with unrelated names still exposes table-only aMe types."""
        indexes, datasets = local_dirs
        folder = "test_v1"
        (indexes / folder).mkdir()
        pl.DataFrame({
            "bodyId": [1], "type": ["hDeltaM"], "instance": ["hDeltaM_C1"],
        }).write_parquet(indexes / folder / "neuron_index.parquet")
        ds_dir = datasets / folder
        ds_dir.mkdir()
        (ds_dir / "x_neuron_df.csv").write_text(
            "bodyId,type,instance\n"
            "1,hDeltaM,hDeltaM_C1\n"
            "2,aMe12,aMe12_L\n"
        )

        pools = _folder_pools(folder)
        assert match_suggestions("aM", pools) == [("aMe12", "type")]
        assert match_suggestions("aMe", pools) == [("aMe12", "type")]

    def test_type_prefix_suppresses_substring_candidates(self):
        pools = {
            "type": [("aMe12", "type")],
            "instance": [("hDeltaM_C1", "instance")],
        }
        assert match_suggestions("aM", pools) == [("aMe12", "type")]

    def test_missing_folder_returns_empty(self, local_dirs, monkeypatch):
        assert get_dataset_pools("hemibrain:v1.2.1") == {}
        assert get_dataset_pools("") == {}

    def test_pool_cache_invalidates_on_file_change(self, local_dirs):
        """Rewriting the index file (new mtime) refreshes the cached pools."""
        indexes, _ = local_dirs
        folder = "test_v1"
        (indexes / folder).mkdir()
        index = indexes / folder / "neuron_index.parquet"
        pl.DataFrame({
            "bodyId": [1], "type": ["APL"], "instance": ["APL_1"],
        }).write_parquet(index)
        assert _folder_pools(folder)["type"] == [("APL", "type")]

        import time
        time.sleep(0.01)  # guarantee a different mtime
        pl.DataFrame({
            "bodyId": [1], "type": ["MBON01"], "instance": ["MBON01_1"],
        }).write_parquet(index)
        assert _folder_pools(folder)["type"] == [("MBON01", "type")]


class TestSuggestionPool:
    def test_union_dedup_across_datasets(self, monkeypatch):
        a = {"type": [("APL", "type"), ("MBON01", "type")]}
        b = {"type": [("APL", "type"), ("KC", "type")],
             "instance": [("KC_1", "instance")]}
        monkeypatch.setattr(
            "ui.type_suggestions.get_dataset_pools",
            lambda ds: a if ds == "A" else b,
        )
        assert suggestion_pool(["A", "B"]) == {
            "type": [("APL", "type"), ("MBON01", "type"), ("KC", "type")],
            "instance": [("KC_1", "instance")],
        }


class TestDatasetAwareSuggestions:
    """The cross-dataset tab's suggestions carry the matched column AND the
    dataset(s) it came from in the gray hint."""

    POOLS_A = {
        "type": [("APL", "type"), ("aMe12", "type")],
        "instance": [("APL_1", "instance")],
        "bodyId": [("5813012345", "aMe12_1")],
    }
    POOLS_B = {
        "type": [("APL", "type"), ("KC", "type")],
        "instance": [("KC_1", "instance")],
        "bodyId": [("5813012345", "aMe12_2")],
    }

    @pytest.fixture
    def pools(self, monkeypatch):
        monkeypatch.setattr(
            "ui.type_suggestions.get_dataset_pools",
            lambda ds: self.POOLS_A if ds == "A" else self.POOLS_B,
        )

    def test_hint_groups_shared_category_with_all_datasets(self, pools):
        assert dataset_aware_suggestions("AP", ["A", "B"]) == [
            ("APL", "type · A, B"),
        ]
        # The type-prefix stage already matched, so the instance pool is not
        # consulted (same staged semantics as the plain merged suggestions).
        assert dataset_aware_suggestions("KC", ["A", "B"]) == [
            ("KC", "type · B"),
        ]

    def test_mixed_categories_keep_per_dataset_segments(self, pools, monkeypatch):
        # A value that is a type in one dataset and an instance in another
        # lists both categories, each tagged with its dataset.
        mixed = {
            "type": [],
            "instance": [("aMe12", "instance")],
            "bodyId": [],
        }
        monkeypatch.setattr(
            "ui.type_suggestions.get_dataset_pools",
            lambda ds: self.POOLS_A if ds == "A" else mixed,
        )
        assert dataset_aware_suggestions("aMe", ["A", "B"]) == [
            ("aMe12", "type · A · instance · B"),
        ]

    def test_only_selected_datasets_contribute(self, pools):
        assert dataset_aware_suggestions("KC", ["A"]) == []
        assert dataset_aware_suggestions("AP", ["A"]) == [
            ("APL", "type · A"),
        ]

    def test_bodyid_hints_carry_each_dataset_instance(self, pools):
        assert dataset_aware_suggestions("5813012345", ["A", "B"]) == [
            ("5813012345", "aMe12_1 · A · aMe12_2 · B"),
        ]

    def test_empty_input_and_limit(self, pools):
        assert dataset_aware_suggestions("", ["A", "B"]) == []
        assert dataset_aware_suggestions("AP", ["A", "B"], limit=1) == [
            ("APL", "type · A, B"),
        ]


class TestDatasetsResolving:
    """Per-value resolving-dataset lists behind the cross-dataset history
    badge tags: exact membership, the caller's dataset order, and values
    found nowhere (patterns) absent from the mapping."""

    POOLS_A = {
        "type": [("APL", "type"), ("aMe12", "type")],
        "instance": [("APL_1", "instance")],
        "bodyId": [("5813012345", "aMe12_1")],
    }
    POOLS_B = {
        "type": [("APL", "type"), ("KC", "type")],
        "instance": [("KC_1", "instance")],
        "bodyId": [("5813012345", "aMe12_2")],
    }

    @pytest.fixture
    def pools(self, monkeypatch):
        monkeypatch.setattr(
            "ui.type_suggestions.get_dataset_pools",
            lambda ds: self.POOLS_A if ds == "A" else self.POOLS_B,
        )

    def test_membership_order_and_absent_values(self, pools):
        from ui.type_suggestions import datasets_resolving

        assert datasets_resolving(
            ["APL", "KC", "5813012345", "AP.*"], ["A", "B"]
        ) == {
            "APL": ["A", "B"],
            "KC": ["B"],
            "5813012345": ["A", "B"],
        }

    def test_restricted_to_requested_datasets(self, pools):
        from ui.type_suggestions import datasets_resolving

        assert datasets_resolving(["APL", "KC"], ["A"]) == {"APL": ["A"]}
        assert datasets_resolving(["APL", "KC"], []) == {}

    def test_empty_and_blank_values_ignored(self, pools):
        from ui.type_suggestions import datasets_resolving

        assert datasets_resolving(["", "  ", None], ["A", "B"]) == {}


def test_clean_split_breaks_combined_cells_into_names():
    """Combined additional_type(s) cells become individual suggestions."""
    from ui.type_suggestions import _clean_split

    assert _clean_split(['vDeltaB, vDeltaC', 'APDN3', None, '']) == [
        'APDN3', 'vDeltaB', 'vDeltaC',
    ]
    assert _clean_split(['aMe12']) == ['aMe12']
    assert _clean_split(['A, , B']) == ['A', 'B']
    assert _clean_split([None, '']) == []
    # whitespace around parts is stripped
    assert _clean_split(['X,  Y ,Z']) == ['X', 'Y', 'Z']
