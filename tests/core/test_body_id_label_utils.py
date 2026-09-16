"""Coverage tests for the shared bodyId display-label helpers.

These helpers format bodyId-level export rows/axes as
'{bodyId}_{instance}' (NeuPrint-style datasets) or
'{bodyId}_{type}_{L|R}' (FAFB/BANC), mirroring
``VisualizeSkeleton._tree_neuron_label``. Pure-unit: no dataset folders
are touched (``body_id_label_map`` is exercised through monkeypatched
loaders).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.label_utils import (
    body_id_label_map,
    build_body_id_label,
)

MALE_CNS = "male-cns:v1.0"
FAFB = "flywire_FAFB_v783"
BANC = "banc_v888"


# ------------------------------------------------- build_body_id_label
def test_neuprint_dataset_uses_instance_suffix():
    label = build_body_id_label(
        MALE_CNS, 11309,
        type_map={11309: "aMe12"}, instance_map={11309: "aMe4_L"})
    assert label == "11309_aMe4_L"


def test_local_release_uses_type_and_hemisphere():
    label = build_body_id_label(
        FAFB, "7205759406", type_map={"7205759406": "aMe4"},
        instance_map={"7205759406": "AA042_L"}, side_map={"7205759406": "left"})
    assert label == "7205759406_aMe4_L"


def test_local_release_hemisphere_from_instance_suffix():
    # No side map: the instance's '_R' suffix resolves the hemisphere.
    label = build_body_id_label(
        BANC, "9001", type_map={"9001": "aMe4"}, instance_map={"9001": "X_R"})
    assert label == "9001_aMe4_R"


def test_local_release_type_without_side_keeps_bare_type():
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "aMe4"}, instance_map={"7": "AA"})
    assert label == "7_aMe4"


def test_instance_fallback_on_local_release_without_type():
    label = build_body_id_label(FAFB, "9002", instance_map={"9002": "AA_L"})
    assert label == "9002_AA_L"


def test_neuprint_type_fallback_appends_hemisphere():
    label = build_body_id_label(
        MALE_CNS, 42, type_map={42: "aMe12"},
        side_map={42: "right"})
    assert label == "42_aMe12_R"


def test_fallback_type_is_last_named_resort():
    label = build_body_id_label(
        MALE_CNS, 1, fallback_type="aMe12")
    assert label == "1_aMe12"
    # Local releases append the hemisphere to the fallback type when the
    # side map resolves it (the instance map had nothing for the neuron).
    label = build_body_id_label(
        FAFB, "5", side_map={"5": "right"}, fallback_type="aMe4")
    assert label == "5_aMe4_R"


def test_bare_bodyid_is_final_fallback():
    assert build_body_id_label(MALE_CNS, 7) == "7"
    assert build_body_id_label(FAFB, "7") == "7"


def test_lookup_tolerates_int_str_key_mismatch():
    # Maps keyed by strings, id passed as int (and vice versa).
    assert build_body_id_label(
        MALE_CNS, 11309, instance_map={"11309": "aMe4_L"}) == "11309_aMe4_L"
    assert build_body_id_label(
        MALE_CNS, "11309", instance_map={11309: "aMe4_L"}) == "11309_aMe4_L"


# ------------------------------------------------- body_id_label_map
def test_body_id_label_map_uses_preloaded_maps(monkeypatch):
    monkeypatch.setattr(
        "utils.label_utils._local_connectome_dataset",
        lambda dataset: False, raising=True)
    labels = body_id_label_map(
        MALE_CNS, [11309, 42],
        type_map={11309: "aMe12", 42: "aMe10"},
        instance_map={11309: "aMe4_L"},
        side_map={})
    assert labels == {"11309": "11309_aMe4_L", "42": "42_aMe10"}


def test_body_id_label_map_loads_missing_maps(monkeypatch):
    captured = {}

    def _fake_type_map(dataset, project_root=None):
        captured["dataset"] = dataset
        return ({5: "aMe12"}, {5: "aMe5_L"})

    def _fake_side_map(dataset, project_root=None):
        return {}

    import utils.label_utils as lu
    monkeypatch.setattr(
        lu, "_load_neuron_type_map", None, raising=False)
    # body_id_label_map imports the loaders from morphology lazily; patch
    # the morphology module attributes it resolves.
    import morphology
    monkeypatch.setattr(morphology, "_load_neuron_type_map", _fake_type_map)
    monkeypatch.setattr(morphology, "_dataset_soma_side_map", _fake_side_map)

    labels = body_id_label_map(MALE_CNS, [5])
    assert labels == {"5": "5_aMe5_L"}
    assert captured["dataset"] == MALE_CNS
