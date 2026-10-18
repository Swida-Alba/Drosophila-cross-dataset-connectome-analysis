"""Contract tests for the multi-dataset morphology suite.

The real intra/cross scoring is covered by the two backend test modules; this
module pins the ORCHESTRATION the UI now relies on: one run folder, one
shared member set, intra per dataset + ordered cross pairs under it, and a
top-level report/parameters bridge. Both backends are replaced with fakes so
the layout assertions never pay for skeleton scoring.
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comparison import morph_multi_dataset as mmd  # noqa: E402


class _FakeCross:
    instances = []
    resolved = {}
    warnings = ["cross warning"]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_dir = kwargs.get("output_dir")
        self.saveas = kwargs.get("saveas")
        self.query = kwargs.get("query")
        self.members_override = kwargs.get("members_override")
        self.warnings = []
        _FakeCross.instances.append(self)

    def _resolve_members(self, tokens, dataset):
        return ({k: list(v) for k, v in
                 (self.resolved.get(dataset) or {}).items()},
                [f"{dataset}: resolved"])

    def run(self):
        out = Path(self.output_dir) / self.saveas
        for pair in ("MCNS_to_FAFB", "FAFB_to_MCNS"):
            (out / pair / "results").mkdir(parents=True, exist_ok=True)
        (out / "parameters.json").write_text(
            json.dumps({"warnings": ["cross warning"],
                        "queries": list(self.query or [])}),
            encoding="utf-8")
        (out / "README.txt").write_text("cross", encoding="utf-8")
        (out / "report.html").write_text("<html>cross</html>",
                                         encoding="utf-8")
        (out / "overview.csv").write_text("queried_type\nA\n",
                                          encoding="utf-8")
        (out / "members_summary.csv").write_text(
            "queried_type,MCNS,FAFB\nA,2,2\n", encoding="utf-8")
        return {"output_folder": str(out), "files": [],
                "warnings": list(_FakeCross.warnings)}


class _FakeIntra:
    instances = []
    custom_groups = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataset = kwargs.get("dataset")
        self.method = kwargs.get("method")
        self.methods = (["vector_v2", "nblast"] if self.method == "both"
                        else [self.method])
        self.aggregation_level = kwargs.get("aggregation_level")
        self.resolved_members = kwargs.get("resolved_members")
        self.output_dir = kwargs.get("output_dir")
        self.saveas = kwargs.get("saveas")
        _FakeIntra.instances.append(self)

    def _resolve_members(self):
        return {k: list(v) for k, v in
                (self.custom_groups.get(self.dataset) or {}).items()}

    @staticmethod
    def _body_id(value):
        return int(value)

    def run(self):
        out = Path(self.output_dir) / self.saveas
        out.mkdir(parents=True, exist_ok=True)
        for method in self.methods:
            (out / "type_level").mkdir(exist_ok=True)
            (out / "type_level"
             / f"type_similarity_{method}.csv").write_text(
                "x\n", encoding="utf-8")
            (out / "bodyid_level").mkdir(exist_ok=True)
            (out / "bodyid_level"
             / f"bodyid_similarity_{method}.csv").write_text(
                "x\n", encoding="utf-8")
        (out / "members.csv").write_text("bodyId\n", encoding="utf-8")
        (out / "report.html").write_text("<html>intra</html>",
                                         encoding="utf-8")
        (out / "parameters.json").write_text("{}", encoding="utf-8")
        members = self.resolved_members or {}
        return {
            "output_folder": str(out),
            "method": self.method,
            "methods": list(self.methods),
            "aggregation_level": self.aggregation_level,
            "rows_compared": len(members),
            "neurons_compared": sum(len(v) for v in members.values()),
            "neurons_compared_by_method": {
                method: sum(len(v) for v in members.values())
                for method in self.methods},
            "files": [],
        }


@pytest.fixture(autouse=True)
def _isolate_fakes(monkeypatch):
    _FakeCross.instances = []
    _FakeCross.resolved = {
        "male-cns:v1.0": {"A": [1, 2], "B": [3]},
        "flywire_FAFB_v783": {"A": [11], "B": [12, 13]},
    }
    _FakeCross.warnings = ["cross warning"]
    _FakeIntra.instances = []
    _FakeIntra.custom_groups = {
        "male-cns:v1.0": {"G1": [1, 2, 9], "G2": [3]},
        "flywire_FAFB_v783": {"G1": [11], "G2": [12, 13, 99]},
    }
    monkeypatch.setattr(mmd, "CrossDatasetMorphComparer", _FakeCross)
    monkeypatch.setattr(mmd, "MorphologyProfileComparer", _FakeIntra)


def _suite(tmp_path, **kwargs):
    params = dict(
        datasets=["male-cns:v1.0", "flywire_FAFB_v783"],
        query=["A", "B"], method="both", aggregation_level="type",
        output_dir=str(tmp_path), verbose=False)
    params.update(kwargs)
    return mmd.MultiDatasetMorphComparer(**params)


def test_one_folder_holds_intra_and_cross(tmp_path):
    result = _suite(tmp_path).run()
    run = Path(result["output_folder"])

    assert run.parent == tmp_path
    assert run.name.startswith("morphology_comparison_MCNS_FAFB_")
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1
    assert (run / "intra_dataset" / "MCNS" / "report.html").is_file()
    assert (run / "intra_dataset" / "FAFB" / "report.html").is_file()
    assert (run / "cross_dataset" / "report.html").is_file()
    assert (run / "cross_dataset" / "MCNS_to_FAFB" / "results").is_dir()
    assert (run / "report.html").is_file()
    assert (run / "parameters.json").is_file()
    assert (run / "README.txt").is_file()

    params = json.loads((run / "parameters.json").read_text())
    assert params["methods_intra"] == ["vector_v2", "nblast"]
    assert params["method_cross"] == "vector_v2"
    assert params["resolved_population"] == {
        "male-cns:v1.0": {"rows": 2, "neurons": 3},
        "flywire_FAFB_v783": {"rows": 2, "neurons": 3},
    }
    assert params["intra_dataset"]["MCNS"]["methods"] == [
        "vector_v2", "nblast"]
    assert params["cross_dataset"]["pairs"] == [
        "FAFB_to_MCNS", "MCNS_to_FAFB"]
    assert params["cross_parameters"]["queries"] == ["A", "B"]

    hub = (run / "report.html").read_text(encoding="utf-8")
    assert "intra_dataset/MCNS/report.html" in hub
    assert "intra_dataset/FAFB/report.html" in hub
    assert "cross_dataset/report.html" in hub

    warnings = (run / "user_warning_notes.txt").read_text(encoding="utf-8")
    assert "[cross] cross warning" in warnings


def test_both_phases_receive_the_same_member_set(tmp_path):
    _suite(tmp_path).run()
    cross_runs = [c for c in _FakeCross.instances if c.members_override]
    assert len(cross_runs) == 1
    shared = cross_runs[0].members_override
    assert shared == {
        "male-cns:v1.0": {"A": [1, 2], "B": [3]},
        "flywire_FAFB_v783": {"A": [11], "B": [12, 13]},
    }
    intra_runs = [c for c in _FakeIntra.instances
                  if c.aggregation_level == "type"]
    assert {c.dataset for c in intra_runs} == {
        "male-cns:v1.0", "flywire_FAFB_v783"}
    for run in intra_runs:
        assert run.resolved_members == shared[run.dataset]


def test_custom_groups_are_filtered_to_the_shared_population(tmp_path):
    result = _suite(
        tmp_path, query=["G1", "G2"], aggregation_level="custom",
        custom_mapping_file="/tmp/preset.json").run()
    run = Path(result["output_folder"])

    # 9 (MCNS) and 99 (FAFB) are in the preset but not in the shared cross
    # population; the intra matrices must not compare a larger pool.
    custom_runs = [c for c in _FakeIntra.instances
                   if c.aggregation_level == "custom"
                   and c.resolved_members is not None]
    assert {c.dataset for c in custom_runs} == {
        "male-cns:v1.0", "flywire_FAFB_v783"}
    by_dataset = {c.dataset: c.resolved_members for c in custom_runs}
    assert by_dataset["male-cns:v1.0"] == {"G1": [1, 2], "G2": [3]}
    assert by_dataset["flywire_FAFB_v783"] == {
        "G1": [11], "G2": [12, 13]}
    assert run.name.startswith("morphology_comparison_MCNS_FAFB_G1_G2_")


def test_total_cap_is_applied_once_before_both_phases(tmp_path):
    result = _suite(tmp_path, max_total_neurons=2).run()
    params = json.loads(
        (Path(result["output_folder"]) / "parameters.json").read_text())
    assert params["resolved_population"] == {
        "male-cns:v1.0": {"rows": 1, "neurons": 2},
        "flywire_FAFB_v783": {"rows": 2, "neurons": 2},
    }
    assert any("max_total_neurons=2" in warning
               for warning in params["warnings"])
    cross_run = [c for c in _FakeCross.instances if c.members_override][0]
    assert sum(len(v) for v in cross_run.members_override[
        "male-cns:v1.0"].values()) == 2
    assert sum(len(v) for v in cross_run.members_override[
        "flywire_FAFB_v783"].values()) == 2


def test_needs_two_datasets():
    with pytest.raises(ValueError, match="at least two datasets"):
        mmd.MultiDatasetMorphComparer(
            datasets=["male-cns:v1.0"], query=["A"])
