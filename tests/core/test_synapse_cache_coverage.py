"""Coverage tests for src/synapse_cache.py.

Targets the uncovered branches in the coverage report:
_json_value normalization branches (numpy scalars/arrays, tolist-only objects,
sets, criteria-like attribute objects, repr fallback), the atomic-writer OSError
fallbacks, _read_manifest schema/dataset/corrupt handling, the disabled-cache
early returns, _valid_frame edge cases, and the full load/save query + pair
paths including cache-hit, empty-entry, missing-file and corrupt-file branches.

Hermetic: all state lives under pytest's tmp_path; no network, no real cache.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import synapse_cache as sc  # noqa: E402


DATASET = "hemibrain:v1.2.1"


def _pair_frame(pre="1", post="2"):
    return pd.DataFrame({
        "bodyId_pre": [pre],
        "bodyId_post": [post],
        "x_pre": [1.0], "y_pre": [2.0], "z_pre": [3.0],
        "x_post": [4.0], "y_post": [5.0], "z_post": [6.0],
    })


def _spec():
    return {"source_ids": ["1"], "target_ids": ["2"], "min_total_weight": 1}


# ---------------------------------------------------------------------------
# _json_value normalization branches
# ---------------------------------------------------------------------------

def test_json_value_passthrough_scalars():
    assert sc._json_value(None) is None
    assert sc._json_value("x") == "x"
    assert sc._json_value(3) == 3
    assert sc._json_value(2.5) == 2.5
    assert sc._json_value(True) is True


def test_json_value_numpy_scalar_uses_item():
    # covers the `value.item()` branch
    assert sc._json_value(np.int64(5)) == 5
    assert sc._json_value(np.float64(1.5)) == 1.5


def test_json_value_tolist_only_object():
    # covers the `value.tolist()` branch for objects without .item()
    class ToListOnly:
        def tolist(self):
            return [1, 2, 3]

    assert sc._json_value(ToListOnly()) == [1, 2, 3]


def test_json_value_numpy_array_item_raises_falls_through():
    # multi-element ndarray: .item() raises -> except -> repr fallback
    result = sc._json_value(np.array([1, 2, 3]))
    assert result["class"] == "ndarray"
    assert "repr" in result


def test_json_value_mapping_sorted_keys():
    assert sc._json_value({"b": 1, "a": 2}) == {"a": 2, "b": 1}


def test_json_value_list_tuple_and_set():
    assert sc._json_value((1, 2)) == [1, 2]
    assert sc._json_value([3, "a"]) == [3, "a"]
    # set/frozenset are sorted deterministically
    assert sc._json_value({2, 1}) == [1, 2]
    assert sc._json_value(frozenset({"b", "a"})) == ["a", "b"]


def test_json_value_criteria_like_attrs():
    class Criteria:
        def __init__(self):
            self.bodyId = [1, 2]
            self.type = "aMe"
            self.confidence = None  # None-valued attr is skipped
            self.status = "active"

    result = sc._json_value(Criteria())
    assert result["class"] == "Criteria"
    assert result["attrs"]["bodyId"] == [1, 2]
    assert result["attrs"]["type"] == "aMe"
    assert result["attrs"]["status"] == "active"
    assert "confidence" not in result["attrs"]


def test_json_value_attr_that_raises_is_skipped():
    class Raising:
        @property
        def bodyId(self):
            raise RuntimeError("boom")

        type = "x"

    result = sc._json_value(Raising())
    assert result["attrs"] == {"type": "x"}


def test_json_value_callable_attr_skipped_and_repr_fallback():
    class OnlyCallables:
        def bodyId(self):  # callable -> ignored
            return 1

    # callable-only attrs produce an empty attr dict -> repr fallback
    result = sc._json_value(OnlyCallables())
    assert result["class"] == "OnlyCallables"
    assert "repr" in result


def test_stable_query_spec_and_query_key_deterministic():
    spec = {"b": np.int64(2), "a": [1, 2]}
    stable = sc.stable_query_spec(spec)
    assert stable == {"a": [1, 2], "b": 2}
    # same logical content, different order -> identical key
    assert sc.query_key({"a": 1, "b": 2}) == sc.query_key({"b": 2, "a": 1})
    assert len(sc.query_key(spec)) == 32
    # the class-level staticmethod delegates to the module function
    assert sc.SynapseCache.query_key(spec) == sc.query_key(spec)


def test_dataset_folder_and_pair_key():
    assert sc.dataset_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
    assert sc._pair_key(1, 2) == "1\t2"


# ---------------------------------------------------------------------------
# atomic writers: OSError fallback in the cleanup finally-block
# ---------------------------------------------------------------------------

def test_atomic_parquet_cleanup_oserror_swallowed(tmp_path, monkeypatch):
    def _raise(self, missing_ok=False):
        raise OSError("locked")

    monkeypatch.setattr("pathlib.Path.unlink", _raise)
    target = tmp_path / "out.parquet"
    sc._atomic_parquet(_pair_frame(), target)
    assert target.exists()


def test_atomic_json_cleanup_oserror_swallowed(tmp_path, monkeypatch):
    def _raise(self, missing_ok=False):
        raise OSError("locked")

    monkeypatch.setattr("pathlib.Path.unlink", _raise)
    target = tmp_path / "out.json"
    sc._atomic_json({"a": 1}, target)
    assert target.exists()


# ---------------------------------------------------------------------------
# _read_manifest branches
# ---------------------------------------------------------------------------

def test_read_manifest_missing_returns_default(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    manifest = cache._read_manifest()
    assert manifest["dataset"] == DATASET
    assert manifest["pairs"] == {}
    # cached on second call
    assert cache._read_manifest() is manifest


def test_read_manifest_valid_existing(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(_pair_frame(), [("1", "2")])
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    manifest = fresh._read_manifest()
    assert "1\t2" in manifest["pairs"]


def test_two_instances_merge_instead_of_overwriting_manifest(tmp_path):
    first = sc.SynapseCache(DATASET, str(tmp_path))
    second = sc.SynapseCache(DATASET, str(tmp_path))
    first.save_pairs(_pair_frame("1", "2"), [("1", "2")])
    second.save_pairs(_pair_frame("3", "4"), [("3", "4")])
    manifest = sc.SynapseCache(DATASET, str(tmp_path))._read_manifest()
    assert "1\t2" in manifest["pairs"]
    assert "3\t4" in manifest["pairs"]


def test_read_manifest_wrong_schema_resets(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text(json.dumps({
        "schema_version": 999, "dataset": DATASET, "pairs": {"x": {}},
    }))
    manifest = cache._read_manifest()
    assert manifest["schema_version"] == sc.SYNAPSE_CACHE_SCHEMA_VERSION
    assert manifest["pairs"] == {}


def test_read_manifest_wrong_dataset_resets(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text(json.dumps({
        "schema_version": sc.SYNAPSE_CACHE_SCHEMA_VERSION,
        "dataset": "other:v1", "pairs": {},
    }))
    manifest = cache._read_manifest()
    assert manifest["dataset"] == DATASET


def test_read_manifest_corrupt_json_falls_back(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache.manifest_path.write_text("{ not json")
    manifest = cache._read_manifest()
    assert manifest["pairs"] == {}


def test_write_manifest_disabled_is_noop(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path), enabled=False)
    cache._write_manifest()
    assert not cache.manifest_path.exists()


# ---------------------------------------------------------------------------
# _valid_frame edge cases
# ---------------------------------------------------------------------------

def test_valid_frame_edges():
    assert sc.SynapseCache._valid_frame(None) is False
    assert sc.SynapseCache._valid_frame("nope") is False
    assert sc.SynapseCache._valid_frame(pd.DataFrame()) is True
    assert sc.SynapseCache._valid_frame(_pair_frame()) is True
    assert sc.SynapseCache._valid_frame(pd.DataFrame({"a": [1]})) is False


# ---------------------------------------------------------------------------
# query save/load paths
# ---------------------------------------------------------------------------

def test_load_and_save_query_disabled(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path), enabled=False)
    assert cache.load_query(_spec()) is None
    cache.save_query(_spec(), _pair_frame())  # no-op
    assert not cache.query_dir.exists()


def test_save_query_none_frame_writes_empty(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), None)
    loaded = cache.load_query(_spec())
    assert loaded is not None and loaded.empty


def test_save_query_non_dataframe_ignored(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), "not-a-frame")
    assert cache.load_query(_spec()) is None


def test_save_query_invalid_columns_rejected(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), pd.DataFrame({"a": [1]}))
    assert cache.load_query(_spec()) is None


def test_load_query_reads_from_disk_in_fresh_cache(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), _pair_frame(), source="neuprint")
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    loaded = fresh.load_query(_spec())
    assert loaded is not None and len(loaded) == 1
    # now served from the in-memory cache
    assert fresh.load_query(_spec()) is not None


def test_load_query_missing_meta_or_data(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    assert cache.load_query(_spec()) is None  # nothing on disk
    # data exists but meta missing
    cache.query_dir.mkdir(parents=True, exist_ok=True)
    _pair_frame().to_parquet(cache.query_path(_spec()), index=False)
    assert cache.load_query(_spec()) is None


def test_load_query_bad_schema_and_dataset_and_spec(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), _pair_frame())
    key = sc.query_key(_spec())

    def _rewrite(**overrides):
        meta = json.loads(cache.query_meta_path(_spec()).read_text())
        meta.update(overrides)
        cache.query_meta_path(_spec()).write_text(json.dumps(meta))

    # wrong schema version
    _rewrite(schema_version=999)
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None

    cache.save_query(_spec(), _pair_frame())
    _rewrite(dataset="other:v1")
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None

    cache.save_query(_spec(), _pair_frame())
    _rewrite(query_key="0" * 32)
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None

    cache.save_query(_spec(), _pair_frame())
    _rewrite(spec={"different": True})
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None
    assert key == sc.query_key(_spec())


def test_load_query_corrupt_parquet_returns_none(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), _pair_frame())
    # corrupt the parquet payload but keep a valid meta sidecar
    path = cache.query_path(_spec())
    path.write_bytes(b"not-a-parquet-file")
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None


def test_load_query_invalid_columns_returns_none(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_query(_spec(), _pair_frame())
    # overwrite with a parquet that lacks the required columns
    pd.DataFrame({"a": [1]}).to_parquet(cache.query_path(_spec()), index=False)
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    assert fresh.load_query(_spec()) is None


# ---------------------------------------------------------------------------
# pair save/load paths
# ---------------------------------------------------------------------------

def test_load_pairs_disabled_lists_all_missing(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path), enabled=False)
    rows, missing = cache.load_pairs([1, 2], [3, 4])
    assert rows is None
    assert missing == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_save_pairs_disabled_is_noop(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path), enabled=False)
    cache.save_pairs(_pair_frame(), [("1", "2")])
    assert not cache.root.exists()


def test_save_pairs_invalid_frame_rejected(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(pd.DataFrame({"a": [1]}), [("1", "2")])
    # invalid frame returns early before indexing or writing the manifest
    assert not cache.manifest_path.exists()


def test_load_pairs_cache_hit_nonempty_and_empty(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(_pair_frame(), [("1", "2")])
    # first read populates _pair_frames, second serves from memory
    rows1, missing1 = cache.load_pairs([1], [2])
    rows2, missing2 = cache.load_pairs([1], [2])
    assert missing1 == [] and missing2 == []
    assert rows1 is not None and rows2 is not None
    pd.testing.assert_frame_equal(rows1, rows2)

    # empty indexed pair served from memory as known-without-file
    cache.save_pairs(None, [("7", "8")])
    rows, missing = cache.load_pairs([7], [8])
    assert rows is None and missing == []
    rows_again, missing_again = cache.load_pairs([7], [8])
    assert rows_again is None and missing_again == []


def test_load_pairs_missing_file(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    rows, missing = cache.load_pairs([100], [200])
    assert rows is None
    assert missing == [("100", "200")]


def test_load_pairs_corrupt_parquet(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(_pair_frame(), [("1", "2")])
    # corrupt the pair file
    path = cache.pair_path("1", "2")
    path.write_bytes(b"garbage")
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    rows, missing = fresh.load_pairs([1], [2])
    assert rows is None
    assert missing == [("1", "2")]


def test_load_pairs_invalid_columns(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(_pair_frame(), [("1", "2")])
    # overwrite with a parquet lacking the required pair columns
    pd.DataFrame({"a": [1]}).to_parquet(cache.pair_path("1", "2"), index=False)
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    rows, missing = fresh.load_pairs([1], [2])
    assert rows is None
    assert missing == [("1", "2")]


def test_load_pairs_multi_pair_concat(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    frame = pd.concat([_pair_frame("1", "2"), _pair_frame("1", "4")],
                      ignore_index=True)
    cache.save_pairs(frame, [("1", "2"), ("1", "4"), ("5", "6")])
    fresh = sc.SynapseCache(DATASET, str(tmp_path))
    # one source against its two targets -> both rows concatenated
    rows, missing = fresh.load_pairs([1], [2, 4])
    assert rows is not None and len(rows) == 2
    assert missing == []
    # the empty-indexed pair resolves as known-without-rows
    rows_none, missing_empty = fresh.load_pairs([5], [6])
    assert rows_none is None and missing_empty == []


def test_save_pairs_skips_already_saved_empty_index(tmp_path):
    cache = sc.SynapseCache(DATASET, str(tmp_path))
    cache.save_pairs(_pair_frame(), [("1", "2"), ("9", "9")])
    manifest = json.loads(cache.manifest_path.read_text())
    assert manifest["pairs"]["1\t2"]["empty"] is False
    assert manifest["pairs"]["1\t2"]["row_count"] == 1
    assert manifest["pairs"]["9\t9"]["empty"] is True
