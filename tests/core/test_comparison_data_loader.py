"""Coverage tests for comparison.data_loader.DataLoader.

Hermetic: all file I/O happens inside pytest tmp_path. No real dataset
folders are read and no network access is performed.
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.data_loader import DataLoader


class _StubConfig:
    """Minimal stand-in for DatasetConfig with neuron groups."""

    def __init__(self, dataset, source_neurons, target_neurons):
        self.dataset = dataset
        self.source_neurons = source_neurons
        self.target_neurons = target_neurons


class _StubMapper:
    """Minimal stand-in for LabelMapper."""

    def __init__(self, has_mapping=True):
        self._has = has_mapping

    def has_mapping(self):
        return self._has

    def export_to_parameters(self):
        return {"source": {"grp": {"ds": ["a"]}}}


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path / "results"))


def test_init_paths(loader, tmp_path):
    assert loader.base_path == os.path.abspath(str(tmp_path / "results"))
    assert loader.dataset_data_path.endswith("dataset_data")
    assert loader.comparison_results_path.endswith("comparison_results")
    assert loader.visualizations_path.endswith("comparison_visualizations")


def test_ensure_directories(loader):
    loader.ensure_directories()
    assert os.path.isdir(loader.base_path)
    assert os.path.isdir(loader.dataset_data_path)
    assert os.path.isdir(loader.comparison_results_path)
    # idempotent
    loader.ensure_directories()


def test_sanitize_name(loader):
    assert loader._sanitize_name("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
    # The writer (ComparisonParameters._sanitize_name) and this loader
    # replace ':' and '.' but preserve '-' for readability, so the folder
    # is male-cns_v0_9 — not the column-safe male_cns_v0_9 used by the
    # label mapper.
    assert loader._sanitize_name("male-cns:v0.9") == "male-cns_v0_9"


def test_get_dataset_path(loader):
    path = loader.get_dataset_path("hemibrain:v1.2.1", 5)
    assert path.endswith(os.path.join("hemibrain_v1_2_1", "minsyn_5"))


def test_save_parameters_with_mapper(loader):
    loader.ensure_directories()
    configs = [
        _StubConfig("hemibrain:v1.2.1", ["aMe12"], ["PN"]),
        _StubConfig("male-cns:v0.9", ["aMe12"], ["PN"]),
    ]
    params_dict = {"datasets": ["hemibrain:v1.2.1", "male-cns:v0.9"], "thresholds": [1, 3]}
    path = loader.save_parameters(params_dict, configs, label_mapper=_StubMapper(True))
    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["source_groups"]["hemibrain:v1.2.1"] == ["aMe12"]
    assert loaded["target_groups"]["male-cns:v0.9"] == ["PN"]
    assert "label_mapping" in loaded


def test_save_parameters_without_mapper(loader):
    loader.ensure_directories()
    configs = [_StubConfig("hemibrain:v1.2.1", [], [])]
    path = loader.save_parameters({"thresholds": [1]}, configs, label_mapper=_StubMapper(False))
    with open(path) as f:
        loaded = json.load(f)
    assert "label_mapping" not in loaded
    # no mapper at all
    path2 = loader.save_parameters({"thresholds": [1]}, configs)
    assert os.path.exists(path2)


def test_load_parameters_missing_raises(loader):
    loader.ensure_directories()
    with pytest.raises(FileNotFoundError):
        loader.load_parameters()


def test_load_parameters_roundtrip(loader):
    loader.ensure_directories()
    loader.save_parameters({"thresholds": [1, 3]}, [])
    loaded = loader.load_parameters()
    assert loaded["thresholds"] == [1, 3]


def test_dataset_results_exist_and_load(loader):
    assert loader.dataset_results_exist("ds", 1) is False
    df = pd.DataFrame({"path": ["A>B"], "hop_count": [1]})
    path = loader.save_dataset_result(df, "hemibrain:v1.2.1", 3, "connection_type_paths.csv")
    assert os.path.exists(path)
    assert loader.dataset_results_exist("hemibrain:v1.2.1", 3) is True

    loaded = loader.load_dataset_results("hemibrain:v1.2.1", 3, file_type="paths")
    assert not loaded.empty
    assert list(loaded.columns) == ["path", "hop_count"]

    # missing path -> empty
    empty = loader.load_dataset_results("nope", 1)
    assert empty.empty


def test_load_dataset_results_fallback_match(loader):
    # Pattern '*paths*' does not match, but generic CSV scan finds file_type in name
    folder = loader.get_dataset_path("ds", 1)
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_csv(os.path.join(folder, "results_summary.csv"), index=False)
    # direct glob for '*summary*' should match
    df = loader.load_dataset_results("ds", 1, file_type="summary")
    assert not df.empty
    # unmatched file_type: glob finds no pattern match, fallback loop keeps
    # the unmatched csv list, so the first csv is still returned
    df2 = loader.load_dataset_results("ds", 1, file_type="zzz")
    assert not df2.empty


def test_load_all_dataset_results(loader):
    df = pd.DataFrame({"path": ["A>B"]})
    loader.save_dataset_result(df, "d1", 1, "all_paths.csv")
    results = loader.load_all_dataset_results(["d1", "d2"], [1], file_type="paths")
    assert "d1" in results and 1 in results["d1"]
    assert results["d2"] == {}


def test_save_load_comparison_result(loader):
    loader.ensure_directories()
    df = pd.DataFrame({"w": [1.0, 2.0]}, index=["a -> b", "c -> d"])
    path = loader.save_comparison_result(df, "aligned.csv")
    assert os.path.exists(path)
    loaded = loader.load_comparison_result("aligned.csv")
    assert list(loaded.index) == ["a -> b", "c -> d"]

    # timestamped variant
    path_ts = loader.save_comparison_result(df, "aligned.csv", include_timestamp=True)
    assert "aligned_" in os.path.basename(path_ts)

    # missing file -> empty
    assert loader.load_comparison_result("missing.csv").empty


def test_metadata_comparison_roundtrip(loader):
    loader.ensure_directories()
    assert loader.load_metadata_comparison().empty
    df = pd.DataFrame({"total_neurons": [100]}, index=["ds1"])
    path = loader.save_metadata_comparison(df)
    assert os.path.exists(path)
    loaded = loader.load_metadata_comparison()
    assert loaded.loc["ds1", "total_neurons"] == 100


def test_list_available_datasets_and_thresholds(loader):
    assert loader.list_available_datasets() == []
    assert loader.list_available_thresholds("nope") == []

    loader.save_dataset_result(pd.DataFrame({"x": [1]}), "hemibrain:v1.2.1", 3, "a.csv")
    loader.save_dataset_result(pd.DataFrame({"x": [1]}), "hemibrain:v1.2.1", 1, "a.csv")
    datasets = loader.list_available_datasets()
    assert "hemibrain_v1_2_1" in datasets
    thresholds = loader.list_available_thresholds("hemibrain:v1.2.1")
    assert thresholds == [1, 3]

    # non-parsable minsyn dir is ignored
    ds_dir = os.path.join(loader.dataset_data_path, "hemibrain_v1_2_1")
    os.makedirs(os.path.join(ds_dir, "minsyn_abc"), exist_ok=True)
    assert loader.list_available_thresholds("hemibrain:v1.2.1") == [1, 3]


def test_get_summary(loader):
    loader.save_dataset_result(pd.DataFrame({"x": [1]}), "d1", 2, "a.csv")
    summary = loader.get_summary()
    assert summary["datasets_count"] == 1
    assert summary["datasets"]["d1"]["thresholds"] == [2]
    assert summary["datasets"]["d1"]["threshold_count"] == 1


def test_export_summary_report(loader):
    loader.ensure_directories()
    analysis_results = {
        "datasets": ["d1", "d2"],
        "thresholds": [1, 3],
        "key_findings": ["finding one"],
    }
    path = loader.export_summary_report(analysis_results)
    assert os.path.exists(path)
    content = Path(path).read_text()
    assert "d1" in content and "finding one" in content

    # minimal results without key findings
    path2 = loader.export_summary_report({"datasets": [], "thresholds": []})
    assert os.path.exists(path2)
