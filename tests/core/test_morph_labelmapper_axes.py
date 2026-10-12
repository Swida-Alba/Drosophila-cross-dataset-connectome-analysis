"""LabelMapper axis compatibility across the morphology comparison family.

The issue report (`clock-production/DROCAT_issue_morph_comparison_labelmapper_axes.md`)
established: the cross-dataset morph comparer had NO LabelMapper entry point,
and bodyId-level axes never carried the standard labels. These tests pin the
fix contract:

* bodyId axes upgrade to ``{bodyId}_{label}({native})`` on resolved ids and
  stay byte-identical to the legacy form otherwise (compatibility gate);
* the cross run writes ``morph_labeltype_matrix.csv`` BESIDE the native
  ``morph_type_matrix.csv`` (never replacing it), with multi-label native
  types expanded to one row per label and unmapped members keeping their
  native name;
* ``overview.csv`` gains a ``label`` column disclosing the resolved
  labels ('/'-joined for double-claim types);
* mapper-absent runs produce no label artifacts at all.

Pure-unit: ``__new__``-built comparers, a two-label preset, no dataset
folders (the dataset label maps are monkeypatched where the axis helper
needs them).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.morph_cross_dataset import CrossDatasetMorphComparer  # noqa: E402
from comparison.label_mapper import PRESET_FORMAT_TAG  # noqa: E402

DS_A = "male-cns:v1.0"
DS_B = "flywire_FAFB_v783"


def _write_preset(tmp_path: Path) -> str:
    """Two labels over DS_A members; bodyId 3 stays unmapped (native key)."""
    preset = {
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {
            "custom_label": ["La", "Lb"],
            DS_A: [["1", "2"], ["5"]],
        }
    }
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset), encoding="utf-8")
    return str(path)


def _cross_comparer(preset_path):
    comp = CrossDatasetMorphComparer.__new__(CrossDatasetMorphComparer)
    comp.datasets = [DS_A, DS_B]
    comp.max_members_per_type = 25
    comp.null_k = 10
    comp.reference_template = None
    comp.scene_members_per_type = 3
    comp.visualize = False
    comp.generate_heatmaps = False
    comp.use_auto_type_mapping = False
    comp.label_mapping_file = preset_path or ""
    comp._label_mapper = None
    comp._label_mapper_loaded = False
    comp._axis_label_of_cache = None
    comp.warnings = []
    comp._abbrevs = {}
    comp._log = lambda message: None
    return comp


def _rows_frame():
    # bodyId 1+2 are native type T1 (label La and La); bodyId 3 is T2
    # (unmapped -> native key); bodyId 5 is T3 (label Lb)
    return pd.DataFrame({
        "source_type": ["T1", "T1", "T2", "T3"],
        "target_type": ["U1", "U1", "U2", "U1"],
        "source_bodyId": [1, 2, 3, 5],
        "target_bodyId": [11, 12, 13, 11],
        "morph_v2": [0.5, 0.7, 0.25, 0.9],
    })


# ------------------------------------------------------- label type matrix
def test_label_type_matrix_expands_and_folds(tmp_path):
    comp = _cross_comparer(_write_preset(tmp_path))
    matrix = comp._label_type_matrix(_rows_frame(), DS_A, DS_B)
    # bodyIds 1,2 -> label La (mean 0.6); bodyId 5 -> Lb; bodyId 3 unmapped
    # -> keeps its native type name T2
    assert set(matrix.index) == {"La", "Lb", "T2"}
    assert set(matrix.columns) == {"U1", "U2"}
    assert matrix.loc["La", "U1"] == pytest.approx(0.6)
    assert matrix.loc["Lb", "U1"] == pytest.approx(0.9)
    assert matrix.loc["T2", "U2"] == pytest.approx(0.25)


def test_label_type_matrix_empty_without_preset(tmp_path):
    comp = _cross_comparer(None)
    assert comp._label_type_matrix(_rows_frame(), DS_A, DS_B).empty


# ------------------------------------------------------- bodyId matrix axes
def test_member_matrix_axes_formatted(tmp_path, monkeypatch):
    import comparison.profile_comparator as pc

    monkeypatch.setattr(
        pc, "_load_body_id_label_maps",
        lambda ds: ({"1": "T1", "11": "U1"},
                    {"1": "x_L", "11": "y_R"},
                    {"1": "left", "11": "right"}))
    comp = _cross_comparer(_write_preset(tmp_path))
    matrix = comp._member_matrix(
        _rows_frame(), DS_A,
        {"T1": [1, 2], "T2": [3], "T3": [5]},
        DS_B, {"U1": [11], "U2": [13]})
    # bodyId 1 resolves to La -> '{bodyId}_{label}({native})' where the
    # parenthetical is exactly the legacy suffix (male-cns: the instance);
    # bodyIds 2/3 are not in the preset -> '{bodyId}_{label}({native})' is
    # skipped for 2?  No: 2 IS in the preset (La covers 1 AND 2) -> its
    # fallback type rides in the parentheses; 3 stays fully native.
    assert "1_La(x_L)" in matrix.index
    assert "2_La(T1)" in matrix.index
    assert "3_T2" in matrix.index


def test_member_matrix_axes_legacy_without_preset(monkeypatch):
    import comparison.profile_comparator as pc

    monkeypatch.setattr(
        pc, "_load_body_id_label_maps",
        lambda ds: ({"1": "T1"}, {"1": "x_L"}, {"1": "left"}))
    comp = _cross_comparer(None)
    matrix = comp._member_matrix(
        _rows_frame().iloc[[0]], DS_A, {"T1": [1]}, DS_B, {"U1": [11]})
    assert list(matrix.index) == ["1_x_L"]


# ---------------------------------------------------------------- overview
def test_overview_label_column_discloses_resolution(tmp_path):
    comp = _cross_comparer(_write_preset(tmp_path))
    members = {DS_A: {"T1": [1, 2], "T3": [5], "T2": [3]}, DS_B: {}}
    overview = comp._overview_frame(["T1", "T3", "T2"], [DS_A, DS_B],
                                    members, {})
    assert overview.columns[0] == "queried_type"
    assert overview.columns[1] == "label"
    by_type = dict(zip(overview["queried_type"], overview["label"]))
    assert by_type["T1"] == "La"          # both members -> one label
    assert by_type["T3"] == "Lb"
    assert by_type["T2"] == ""            # unmapped -> empty, not a lie


# ------------------------------------------------------------ write outputs
def test_write_outputs_writes_labeltype_beside_native(tmp_path):
    comp = _cross_comparer(_write_preset(tmp_path))
    run_path = tmp_path / "run"
    run_path.mkdir()
    pair_info = {
        "rows": None,
        "type_matrix": pd.DataFrame({"U1": [0.4]}, index=["T1"]),
        "bodyid_matrix": pd.DataFrame(),
        "labeltype_matrix": pd.DataFrame({"U1": [0.4]}, index=["La"]),
        "baseline": {},
    }
    comp._write_outputs(run_path, ["T1"], [DS_A, DS_B],
                        {DS_A: {"T1": [1]}}, {(DS_A, DS_B): pair_info},
                        pd.DataFrame({"queried_type": ["T1"]}), [])
    results = run_path / f"MCNS_to_FAFB" / "results"
    assert (results / "morph_type_matrix.csv").exists()
    assert (results / "morph_labeltype_matrix.csv").exists()
    params = json.loads(
        (run_path / "parameters.json").read_text(encoding="utf-8"))
    assert params["label_mapping_file"] == _write_preset(tmp_path)
    assert "morph_labeltype_matrix.csv" in params["label_axes"]


def test_write_outputs_no_label_artifacts_without_preset(tmp_path):
    comp = _cross_comparer(None)
    run_path = tmp_path / "run"
    run_path.mkdir()
    pair_info = {
        "rows": None,
        "type_matrix": pd.DataFrame({"U1": [0.4]}, index=["T1"]),
        "bodyid_matrix": pd.DataFrame(),
        "labeltype_matrix": pd.DataFrame(),
        "baseline": {},
    }
    comp._write_outputs(run_path, ["T1"], [DS_A, DS_B],
                        {DS_A: {"T1": [1]}}, {(DS_A, DS_B): pair_info},
                        pd.DataFrame({"queried_type": ["T1"]}), [])
    results = run_path / "MCNS_to_FAFB" / "results"
    assert (results / "morph_type_matrix.csv").exists()
    assert not (results / "morph_labeltype_matrix.csv").exists()
    params = json.loads(
        (run_path / "parameters.json").read_text(encoding="utf-8"))
    assert "label_mapping_file" not in params
    assert "label_axes" not in params


# ------------------------------------------------- intra comparer (F2 fix)
def test_intra_display_labels_formatted(tmp_path, monkeypatch):
    from morphology_comparison import MorphologyProfileComparer

    comp = MorphologyProfileComparer.__new__(MorphologyProfileComparer)
    comp.dataset = DS_A
    comp.project_root = PROJECT_ROOT
    comp.label_mapping_file = _write_preset(tmp_path)
    comp._label_mapper = None
    comp._label_mapper_loaded = False
    comp._axis_label_of_cache = None
    comp._type_map = {"1": "T1", "5": "T3"}
    comp._instance_map = {"1": "x", "5": "y"}
    comp._body_id = lambda value: int(value)
    comp._log = lambda message: None
    comp._resolution_notes = []
    # bodyIds 1 (La) and 5 (Lb) resolve; the parenthetical is exactly the
    # legacy suffix (male-cns: the instance)
    labels = comp._display_labels([1, 5])
    assert labels == ["1_La(x)", "5_Lb(y)"]


def test_intra_display_labels_legacy_without_preset():
    from morphology_comparison import MorphologyProfileComparer

    comp = MorphologyProfileComparer.__new__(MorphologyProfileComparer)
    comp.dataset = DS_A
    comp.project_root = PROJECT_ROOT
    comp.label_mapping_file = ""
    comp._label_mapper = None
    comp._label_mapper_loaded = False
    comp._axis_label_of_cache = None
    comp._type_map = {"1": "T1"}
    comp._instance_map = {"1": "x"}
    comp._body_id = lambda value: int(value)
    comp._resolution_notes = []
    assert comp._display_labels([1]) == ["1_x"]
