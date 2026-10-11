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


# ---------------------------------------- LabelMapper axis format (F2/F3)
def test_label_format_wraps_native_in_parentheses():
    """A resolved bodyId reads {bodyId}_{label}({native}) — the legacy
    suffix, side marker included, moves into the parentheses."""
    label = build_body_id_label(
        FAFB, "7205759406", type_map={"7205759406": "aMe4"},
        instance_map={"7205759406": "AA042_L"}, side_map={"7205759406": "left"},
        label_of=lambda ds, bid: "5th-LNv")
    assert label == "7205759406_5th-LNv(aMe4_L)"


def test_label_format_keeps_instance_suffix_in_parentheses():
    # instance-based suffix (no side appended): the instance rides unchanged
    label = build_body_id_label(
        MALE_CNS, 11309, type_map={11309: "aMe12"},
        instance_map={11309: "aMe4_L"}, label_of=lambda ds, bid: "aMe12_std")
    assert label == "11309_aMe12_std(aMe4_L)"


def test_label_format_bodyid_stays_head_token():
    # axis parsers split on '_' and take the head — the bodyId must remain
    # the token before the first underscore even when the label carries '_'
    label = build_body_id_label(
        FAFB, "7205759406", type_map={"7205759406": "aMe4"},
        side_map={"7205759406": "left"},
        label_of=lambda ds, bid: "LNd_CRY+_ITP+")
    assert label.split("_", 1)[0] == "7205759406"
    assert label == "7205759406_LNd_CRY+_ITP+(aMe4_L)"


def test_unmapped_bodyid_keeps_legacy_form_byte_identical():
    legacy = build_body_id_label(
        FAFB, "7205759406", type_map={"7205759406": "aMe4"},
        instance_map={"7205759406": "AA042_L"}, side_map={"7205759406": "left"})
    # resolver returns None for this id -> byte-identical to no resolver
    assert build_body_id_label(
        FAFB, "7205759406", type_map={"7205759406": "aMe4"},
        instance_map={"7205759406": "AA042_L"},
        side_map={"7205759406": "left"},
        label_of=lambda ds, bid: None) == legacy
    assert legacy == "7205759406_aMe4_L"


def test_no_resolver_is_byte_identical():
    kwargs = dict(type_map={"7": "aMe4"}, side_map={"7": "left"})
    assert (build_body_id_label(FAFB, "7", **kwargs)
            == build_body_id_label(FAFB, "7", label_of=None, **kwargs))


def test_resolver_exception_degrades_to_legacy():
    def bad(_ds, _bid):
        raise RuntimeError("boom")
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "aMe4"}, label_of=bad)
    assert label == "7_aMe4"


def test_label_without_any_native_suffix():
    # bare bodyId fallback (no maps at all): the label still names the row
    assert build_body_id_label(
        FAFB, "7", label_of=lambda ds, bid: "5th-LNv") == "7_5th-LNv"


def test_body_id_label_map_passes_label_through():
    labels = body_id_label_map(
        FAFB, ["9"], type_map={"9": "aMe4"}, instance_map={"9": "AA_L"},
        side_map={"9": "left"}, label_of=lambda ds, bid: "std-aMe4")
    assert labels["9"] == "9_std-aMe4(aMe4_L)"


def test_body_id_label_map_without_label_is_legacy():
    labels = body_id_label_map(
        FAFB, ["9"], type_map={"9": "aMe4"}, instance_map={"9": "AA_L"},
        side_map={"9": "left"})
    assert labels["9"] == "9_aMe4_L"


# ------------------------------------------------- label == native dedup
def test_label_equal_native_drops_the_parenthetical():
    # label == native type (no side): nothing to add -> plain {bodyId}_{label}
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "l-LNv"}, label_of=lambda ds, bid: "l-LNv")
    assert label == "7_l-LNv"


def test_label_equal_native_keeps_only_the_side():
    # label == native type but a side marker exists: keep the side, drop the
    # repeated name (house '{bodyId}_{type}_{L|R}' shape, label substituted)
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "l-LNv"}, side_map={"7": "left"},
        label_of=lambda ds, bid: "l-LNv")
    assert label == "7_l-LNv_L"


def test_label_equal_instance_drops_the_parenthetical():
    # NeuPrint-style: the instance is the native suffix
    label = build_body_id_label(
        MALE_CNS, 11309, instance_map={11309: "aMe4_L"},
        label_of=lambda ds, bid: "aMe4_L")
    assert label == "11309_aMe4_L"


def test_label_equal_with_fallback_type_also_dedups():
    # the fallback type is the native suffix of last resort — same rule
    label = build_body_id_label(
        FAFB, "7", fallback_type="DN1a", label_of=lambda ds, bid: "DN1a")
    assert label == "7_DN1a"


def test_differing_label_still_wraps_native():
    # near-miss must NOT dedup (side-only difference is handled above)
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "l-LNv"}, side_map={"7": "left"},
        label_of=lambda ds, bid: "l-LNv_std")
    assert label == "7_l-LNv_std(l-LNv_L)"


def test_label_equal_instance_modulo_side_keeps_only_side():
    # NeuPrint-style: the native suffix is the INSTANCE spelling the side
    # out ('l-LNv_L'); a label differing only by that marker dedups too
    label = build_body_id_label(
        MALE_CNS, 11309, instance_map={11309: "l-LNv_L"},
        label_of=lambda ds, bid: "l-LNv")
    assert label == "11309_l-LNv_L"


def test_type_and_side_map_dedup_still_fires():
    label = build_body_id_label(
        FAFB, "7", type_map={"7": "DN1a"}, side_map={"7": "right"},
        label_of=lambda ds, bid: "DN1a")
    assert label == "7_DN1a_R"
