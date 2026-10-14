"""Coverage tests for comparison.label_mapper.LabelMapper.

Hermetic: all mapping files are written into pytest tmp_path. No access to
real mappings/ or datasets/ folders.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.label_mapper import (
    PRESET_FORMAT_TAG, PresetFormatError, LabelMapper)

DS1 = "hemibrain:v1.2.1"
DS2 = "male-cns:v0.9"


def _dict_mapper():
    return LabelMapper(
        source_mapping_dict={DS1: [["aMe12", "aMe12_R"], ["aMe12_L"]], DS2: [["aMe12"], []]},
        source_labels=["aMe12_grp1", "aMe12_grp2"],
        target_mapping_dict={DS1: [["PN1"]], DS2: [["PN1", "PN2"]]},
        target_labels=["pn_grp"],
        intermediate_mapping_dict={DS1: [["IN1"]], DS2: [["IN2"]]},
        intermediate_labels=["in_grp"],
    )


def test_empty_mapper():
    mapper = LabelMapper()
    assert mapper.is_empty
    assert not mapper.has_mapping()
    assert "LabelMapper" in repr(mapper)


def test_dict_mapper_basics():
    mapper = _dict_mapper()
    assert not mapper.is_empty
    assert mapper.has_mapping()
    assert mapper.get_std_label(DS1, "aMe12", "source") == "aMe12_grp1"
    assert mapper.get_std_label(DS1, "aMe12_L", "source") == "aMe12_grp2"
    # hemisphere suffix fallback: aMe12_R not directly mapped in DS2 but base aMe12 is
    assert mapper.get_std_label(DS2, "aMe12_R", "source") == "aMe12_grp1_R"
    # unmapped -> auto-generated
    assert mapper.get_std_label(DS1, "Unknown1", "source") == "Unknown1"
    assert LabelMapper.auto_generate_label(123) == "123"
    # target role
    assert mapper.get_std_label(DS2, "PN2", "target") == "pn_grp"
    # intermediate role
    assert mapper.get_std_label(DS1, "IN1", "intermediate") == "in_grp"
    # sanitized dataset fallback: mapping stored under sanitized key,
    # queried with the original dataset identifier
    san_mapper = LabelMapper(source_mapping_dict={"hemibrain_v1_2_1": [["aMe12"]]}, source_labels=["g"])
    assert san_mapper.get_std_label("hemibrain:v1.2.1", "aMe12", "source") == "g"
    assert san_mapper.get_label("hemibrain:v1.2.1", "aMe12") == "g"
    assert san_mapper.get_neurons_for_label("g", "hemibrain:v1.2.1", "source") == ["aMe12"]
    assert san_mapper.get_all_neurons_for_dataset("hemibrain:v1.2.1", "source") == ["aMe12"]


def test_get_label_priority_and_get_mapped_label():
    mapper = _dict_mapper()
    # get_label checks source first
    assert mapper.get_label(DS1, "aMe12") == "aMe12_grp1"
    assert mapper.get_label(DS1, "PN1") == "pn_grp"
    assert mapper.get_label(DS1, "IN1") == "in_grp"
    assert mapper.get_label(DS1, "nope") == "nope"

    assert mapper.get_mapped_label("aMe12_grp1", DS1) == ["aMe12", "aMe12_R"]
    assert mapper.get_mapped_label("aMe12_grp2", DS2) is None or mapper.get_mapped_label("aMe12_grp2", DS2) == []
    assert mapper.get_mapped_label("pn_grp", DS2) == ["PN1", "PN2"]
    assert mapper.get_mapped_label("in_grp", DS1) == "IN1"
    assert mapper.get_mapped_label("missing", DS1) is None


def test_validate_datasets():
    mapper = _dict_mapper()
    mapper.validate_datasets([DS1, DS2], role="both")
    mapper.validate_datasets([DS1], role="source")
    with pytest.raises(ValueError):
        mapper.validate_datasets([DS1, "unknown-ds"], role="source")
    with pytest.raises(ValueError):
        mapper.validate_datasets(["unknown-ds"], role="target")


def test_merge():
    a = LabelMapper(source_mapping_dict={DS1: [["aMe12"]]}, source_labels=["grp"])
    b = LabelMapper(source_mapping_dict={DS1: [["aMe12", "aMe13"]], DS2: [["aMe12"]]}, source_labels=["grp"])
    a.merge(b)
    assert a.get_neurons_for_label("grp", DS1, "source") == ["aMe12", "aMe13"]
    assert a.get_neurons_for_label("grp", DS2, "source") == ["aMe12"]


def test_queries():
    mapper = _dict_mapper()
    assert mapper.get_all_std_labels("source") == ["aMe12_grp1", "aMe12_grp2"] or set(
        mapper.get_all_std_labels("source")
    ) == {"aMe12_grp1", "aMe12_grp2"}
    assert set(mapper.get_datasets("source")) == {DS1, DS2}
    assert set(mapper.get_all_neurons_for_dataset(DS1, "source")) == {"aMe12", "aMe12_R", "aMe12_L"}
    assert mapper.get_all_neurons_for_dataset("nope", "source") == []
    assert mapper.get_neurons_for_label("aMe12_grp1", DS1, "source") == ["aMe12", "aMe12_R"]
    assert mapper.get_neurons_for_label("missing", DS1, "source") == []
    # hemi suffix fallback on label
    assert mapper.get_neurons_for_label("aMe12_grp1_R", DS1, "source") == ["aMe12", "aMe12_R"]
    # _split_hemi_suffix
    assert LabelMapper._split_hemi_suffix("X_R") == ("X", "_R")
    assert LabelMapper._split_hemi_suffix("X_U") == ("X", "_U")
    assert LabelMapper._split_hemi_suffix("X") == ("X", "")
    assert LabelMapper._split_hemi_suffix(5) == (5, "")


def test_apply_to_dataframe():
    mapper = _dict_mapper()
    df = pd.DataFrame({
        "type_pre": ["aMe12", None],
        "type_post": ["PN1", "PN2"],
        "weight": [5, 2],
    })
    out = mapper.apply_to_dataframe(df, DS1)
    assert out.loc[0, "std_label_pre"] == "aMe12_grp1"
    assert out.loc[1, "std_label_pre"] == ""
    assert out.loc[0, "std_label_post"] == "pn_grp"
    # empty df unchanged
    assert mapper.apply_to_dataframe(pd.DataFrame(), DS1).empty
    # bodyId-based df
    df_body = pd.DataFrame({"bodyId_pre": [1001], "bodyId_post": [2002], "weight": [1]})
    out_body = mapper.apply_to_dataframe(df_body, DS1)
    assert out_body.loc[0, "std_label_pre"] == "1001"


def test_mapping_summary_and_export_parameters():
    mapper = _dict_mapper()
    summary = mapper.get_mapping_summary()
    assert not summary.empty
    assert set(summary["role"]) == {"source", "target", "intermediate"}
    params = mapper.export_to_parameters()
    assert "resolved_source_mapping" in params
    assert params["resolved_source_mapping"]["aMe12_grp1"][DS1] == ["aMe12", "aMe12_R"]


def test_to_dict_export_json_roundtrip(tmp_path):
    mapper = _dict_mapper()
    d = mapper.to_dict()
    assert "source_mapping" in d and "target_mapping" in d and "intermediate_mapping" in d

    path = tmp_path / "mapping.json"
    mapper.export_to_json(str(path))
    assert path.exists()

    reloaded = LabelMapper(overall_mapping_json=str(path))
    assert reloaded.get_std_label(DS1, "aMe12", "source") == "aMe12_grp1"
    assert reloaded.get_std_label(DS2, "PN2", "target") == "pn_grp"
    assert reloaded.get_std_label(DS1, "IN1", "intermediate") == "in_grp"


def test_load_from_json_missing_file():
    with pytest.raises(FileNotFoundError):
        LabelMapper(source_mapping_file="/nonexistent/nowhere.json")


def test_load_from_json_errors(tmp_path):
    # no mapping keys at all
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": PRESET_FORMAT_TAG, "foo": 1}))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad))

    # missing label key
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                "source_mapping": {"ds1": [["a"]]}}))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad2))

    # group count mismatch
    bad3 = tmp_path / "bad3.json"
    bad3.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["l1", "l2"], "ds1": [["a"]]}
    }))
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad3))

    # unified file legitimately missing target role -> ok
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]}
    }))
    mapper = LabelMapper(overall_mapping_json=str(ok))
    assert mapper.get_std_label("ds1", "a", "source") == "l1"

    # non-list group value gets wrapped
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["l1"], "ds1": ["a"]}
    }))
    mapper2 = LabelMapper(source_mapping_file=str(scalar))
    assert mapper2.get_std_label("ds1", "a", "source") == "l1"


def test_load_from_csv_expanded_format(tmp_path):
    csv = tmp_path / "source.csv"
    csv.write_text(
        "custom_label,std_pattern,flywire_FAFB,notes\n"
        "grpA,A.*,100,first\n"
        "grpA,A.*_R,101,second\n"
        ",B.*,200,auto label from pattern\n"
        ",,\n"
    )
    mapper = LabelMapper(source_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 100, "source") == "grpA"
    assert mapper.get_std_label("flywire_FAFB", 101, "source") == "grpA"
    assert mapper.get_std_label("flywire_FAFB", 200, "source") == "B.*"
    assert not mapper.is_empty


def test_load_from_csv_legacy_format(tmp_path):
    csv = tmp_path / "legacy.csv"
    csv.write_text(
        "std_label,flywire_FAFB,other_ds\n"
        "grp1,10;abc,30\n"
        ",40;\n"
    )
    mapper = LabelMapper(target_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 10, "target") == "grp1"
    assert mapper.get_std_label("flywire_FAFB", "abc", "target") == "grp1"
    assert mapper.get_std_label("other_ds", 30, "target") == "grp1"
    assert mapper.get_std_label("other_ds", 40, "target") == "40"


def test_load_from_csv_errors(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        LabelMapper(source_mapping_file=str(missing))

    bad = tmp_path / "bad.csv"
    bad.write_text("col_a,col_b\n1,2\n")
    with pytest.raises(ValueError):
        LabelMapper(source_mapping_file=str(bad))


def test_verify_csv_format(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("custom_label,flywire_FAFB\ngrpA,100\ngrpA,101\n,200\n")
    res = LabelMapper.verify_csv_format(str(good))
    assert res["valid"]
    assert res["summary"]["rows"] == 3
    assert res["warnings"]  # empty label warning + duplicate label warning

    res_missing = LabelMapper.verify_csv_format(str(tmp_path / "nope.csv"))
    assert not res_missing["valid"]

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    res_bad = LabelMapper.verify_csv_format(str(bad))
    assert not res_bad["valid"]

    no_datasets = tmp_path / "nodata.csv"
    no_datasets.write_text("custom_label,notes\ngrpA,x\n")
    res_nd = LabelMapper.verify_csv_format(str(no_datasets))
    assert not res_nd["valid"]


def test_verify_json_format(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["a"], "ds1": [["1"]]},
        "target_mapping": {"std_label": ["b"], "ds1": [["2"]]},
    }))
    res = LabelMapper.verify_json_format(str(good))
    assert res["valid"]

    res_missing = LabelMapper.verify_json_format(str(tmp_path / "nope.json"))
    assert not res_missing["valid"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    assert not LabelMapper.verify_json_format(str(bad_json))["valid"]

    no_sections = tmp_path / "empty.json"
    no_sections.write_text("{}")
    assert not LabelMapper.verify_json_format(str(no_sections))["valid"]

    bad_section = tmp_path / "badsec.json"
    bad_section.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                       "source_mapping": {"ds1": [["a"]]}}))
    assert not LabelMapper.verify_json_format(str(bad_section))["valid"]

    bad_labels_type = tmp_path / "badtype.json"
    bad_labels_type.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                           "source_mapping": {"custom_label": "notalist", "ds1": [["a"]]}}))
    assert not LabelMapper.verify_json_format(str(bad_labels_type))["valid"]

    bad_ds_type = tmp_path / "baddstype.json"
    bad_ds_type.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                       "source_mapping": {"custom_label": ["a"], "ds1": "notalist"}}))
    assert not LabelMapper.verify_json_format(str(bad_ds_type))["valid"]

    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                    "source_mapping": {"custom_label": ["a", "b"], "ds1": [["1"]]}}))
    assert not LabelMapper.verify_json_format(str(mismatch))["valid"]

    # Shape is not a version gate: a preset that omits the tag — or declares a
    # grammar this build does not read — is refused on the tag, not parsed.
    no_tag = tmp_path / "notag.json"
    no_tag.write_text(json.dumps({
        "source_mapping": {"custom_label": ["a"], "ds1": [["1"]]},
    }))
    res_no_tag = LabelMapper.verify_json_format(str(no_tag))
    assert not res_no_tag["valid"] and "no 'format' key" in res_no_tag["errors"][0]

    other_grammar = tmp_path / "v2.json"
    other_grammar.write_text(json.dumps({
        "format": "drocat_custom_groups/v2",
        "source_mapping": {"custom_label": ["a"], "ds1": [["1"]]},
    }))
    res_v2 = LabelMapper.verify_json_format(str(other_grammar))
    assert not res_v2["valid"]
    assert "drocat_custom_groups/v2" in res_v2["errors"][0]


def test_preset_format_gate_on_load(tmp_path):
    """The loader refuses the same two cases verify_json_format reports."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]},
    }))
    assert LabelMapper(overall_mapping_json=str(good)).get_std_label(
        "ds1", "a", "source") == "l1"

    no_tag = tmp_path / "notag.json"
    no_tag.write_text(json.dumps({
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]},
    }))
    with pytest.raises(PresetFormatError, match="no 'format' key"):
        LabelMapper(overall_mapping_json=str(no_tag))

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({
        "format": "drocat_custom_groups/v2",
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]},
    }))
    with pytest.raises(PresetFormatError, match="drocat_custom_groups/v2"):
        LabelMapper(overall_mapping_json=str(unknown))

    # An in-memory dict makes no file-format claim, so it stays ungated.
    assert LabelMapper(source_mapping_dict={"ds1": [["a"]]},
                       source_labels=["l1"]).get_std_label(
        "ds1", "a", "source") == "l1"


def test_exported_preset_reload_is_self_consistent(tmp_path):
    """Whatever the backend exports, it must be able to read back."""
    mapper = _dict_mapper()
    path = tmp_path / "roundtrip.json"
    mapper.export_to_json(str(path))
    assert json.loads(path.read_text())["format"] == PRESET_FORMAT_TAG
    assert LabelMapper(overall_mapping_json=str(path)).get_std_label(
        DS1, "aMe12", "source") == "aMe12_grp1"


def test_load_label_mapper_gate_and_swallow(tmp_path):
    """load_label_mapper degrades a BROKEN preset to None (a run keeps its
    native keys) but re-raises the version gate — silently dropping the
    overlay would publish labels that were never applied."""
    from comparison.label_mapper import load_label_mapper

    logs = []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_label_mapper(str(broken), log=logs.append) is None
    assert any("could not load the label mapper" in line for line in logs)

    no_tag = tmp_path / "notag.json"
    no_tag.write_text(json.dumps({
        "source_mapping": {"custom_label": ["l1"], "ds1": [["a"]]},
    }))
    with pytest.raises(PresetFormatError):
        load_label_mapper(str(no_tag), log=logs.append)


def test_verify_mapping_file_dispatch(tmp_path):
    good_json = tmp_path / "m.json"
    good_json.write_text(json.dumps({"format": PRESET_FORMAT_TAG,
                                     "source_mapping": {"custom_label": ["a"], "ds1": [["1"]]}}))
    assert LabelMapper.verify_mapping_file(str(good_json))["valid"]

    good_csv = tmp_path / "m.csv"
    good_csv.write_text("custom_label,ds1\na,1\n")
    assert LabelMapper.verify_mapping_file(str(good_csv))["valid"]

    unsupported = LabelMapper.verify_mapping_file(str(tmp_path / "m.txt"))
    assert not unsupported["valid"]


# ---------------------------------------------------------------------------
# Appended coverage batch: target/intermediate roles, dict/CSV/JSON load
# branches, unsanitization, sanitized reverse lookups, verify error paths.
# ---------------------------------------------------------------------------


def test_merge_target_and_intermediate():
    a = LabelMapper(
        target_mapping_dict={DS1: [["PN1"]]}, target_labels=["t"],
        intermediate_mapping_dict={DS1: [["IN1"]]}, intermediate_labels=["i"],
    )
    b = LabelMapper(
        target_mapping_dict={DS1: [["PN2"]], DS2: [["PN3"]]}, target_labels=["t"],
        intermediate_mapping_dict={DS1: [["IN2"]]}, intermediate_labels=["i"],
    )
    a.merge(b)
    assert a.get_neurons_for_label("t", DS1, "target") == ["PN1", "PN2"]
    assert a.get_neurons_for_label("t", DS2, "target") == ["PN3"]
    assert a.get_neurons_for_label("i", DS1, "intermediate") == ["IN1", "IN2"]


def test_load_intermediate_csv_expanded(tmp_path):
    csv = tmp_path / "inter.csv"
    csv.write_text(
        "custom_label,flywire_FAFB,male_cns_v0_9,notes\n"
        "i1,500,510,x\n"
        ",600,,y\n"
    )
    mapper = LabelMapper(intermediate_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 500, "intermediate") == "i1"
    assert mapper.get_std_label(DS2, 510, "intermediate") == "i1"
    # missing label + no std_pattern -> auto-generated from first dataset value
    assert mapper.get_std_label("flywire_FAFB", 600, "intermediate") == "600"


def test_load_expanded_target_and_legacy_source_csv(tmp_path):
    target_csv = tmp_path / "target_exp.csv"
    target_csv.write_text("custom_label,flywire_FAFB\ntg1,800\n")
    mapper = LabelMapper(target_mapping_file=str(target_csv))
    assert mapper.get_std_label("flywire_FAFB", 800, "target") == "tg1"

    source_csv = tmp_path / "source_legacy.csv"
    source_csv.write_text("std_label,flywire_FAFB\nsg1,900;901\n")
    mapper2 = LabelMapper(source_mapping_file=str(source_csv))
    assert mapper2.get_std_label("flywire_FAFB", 900, "source") == "sg1"
    assert mapper2.get_std_label("flywire_FAFB", 901, "source") == "sg1"


def test_load_legacy_intermediate_csv(tmp_path):
    csv = tmp_path / "legacy_inter.csv"
    csv.write_text(
        "std_label,flywire_FAFB\n"
        "ig1,70;71\n"
        ",\n"  # row with no valid data -> skipped
    )
    mapper = LabelMapper(intermediate_mapping_file=str(csv))
    assert mapper.get_std_label("flywire_FAFB", 70, "intermediate") == "ig1"
    assert mapper.get_std_label("flywire_FAFB", 71, "intermediate") == "ig1"


def test_load_json_std_label_key(tmp_path):
    p = tmp_path / "sl.json"
    p.write_text(json.dumps(
        {"format": PRESET_FORMAT_TAG,
         "source_mapping": {"std_label": ["g1"], "ds1": [["9"]]}}))
    mapper = LabelMapper(source_mapping_file=str(p))
    assert mapper.get_std_label("ds1", "9", "source") == "g1"


def test_load_from_dict_branches():
    # empty dataset entry is skipped
    m = LabelMapper(source_mapping_dict={DS1: [], DS2: [["x"]]}, source_labels=["g"])
    assert m.get_std_label(DS2, "x", "source") == "g"
    assert m.get_datasets("source") == [DS2]
    # flat list with len(labels) == len(neurons) > 1 -> one group each
    m2 = LabelMapper(source_mapping_dict={DS1: ["n1", "n2"]}, source_labels=["l1", "l2"])
    assert m2.get_std_label(DS1, "n1", "source") == "l1"
    assert m2.get_std_label(DS1, "n2", "source") == "l2"
    # flat list without labels -> single group with auto-generated label
    m3 = LabelMapper(source_mapping_dict={DS1: ["n3", "n4"]})
    assert m3.get_all_std_labels("source") == ["source_grp1"]
    # more groups than labels -> labels extended
    m4 = LabelMapper(source_mapping_dict={DS1: [["a"], ["b"], ["c"]]},
                     source_labels=["only1"])
    assert set(m4.get_all_std_labels("source")) == {
        "only1", "source_grp2", "source_grp3"}


def test_unsanitize_dataset_names():
    m = LabelMapper()
    # hemibrain conversion is documented as best-effort/lossy:
    # only the first remaining underscore is restored to a dot
    assert m._unsanitize_dataset_name("hemibrain_v1_2_1") == "hemibrain:v1.2_1"
    assert m._unsanitize_dataset_name("male_cns_v0_9") == "male-cns:v0.9"
    assert m._unsanitize_dataset_name("flywire_FAFB") == "flywire_FAFB"
    assert m._unsanitize_dataset_name("other_ds") == "other_ds"


def test_get_std_label_hemi_suffix_sanitized():
    m = LabelMapper(source_mapping_dict={"hemibrain_v1_2_1": [["aMe12"]]},
                    source_labels=["g"])
    assert m.get_std_label("hemibrain:v1.2.1", "aMe12_R", "source") == "g_R"


def test_role_query_helpers():
    mapper = _dict_mapper()
    # get_neurons_for_label across roles + base-label sanitized fallback
    assert mapper.get_neurons_for_label("pn_grp", DS1, "target") == ["PN1"]
    assert mapper.get_neurons_for_label("in_grp", DS2, "intermediate") == ["IN2"]
    m = LabelMapper(target_mapping_dict={"hemibrain_v1_2_1": [["x"]]},
                    target_labels=["base"])
    assert m.get_neurons_for_label("base_L", "hemibrain:v1.2.1", "target") == ["x"]
    # get_all_std_labels / get_datasets for target and intermediate
    assert mapper.get_all_std_labels("target") == ["pn_grp"]
    assert mapper.get_all_std_labels("intermediate") == ["in_grp"]
    assert set(mapper.get_datasets("target")) == {DS1, DS2}
    assert set(mapper.get_datasets("intermediate")) == {DS1, DS2}
    # get_all_neurons_for_dataset for target and intermediate
    assert mapper.get_all_neurons_for_dataset(DS1, "target") == ["PN1"]
    assert mapper.get_all_neurons_for_dataset(DS2, "intermediate") == ["IN2"]


def test_get_label_sanitized_target_intermediate():
    m = LabelMapper(
        target_mapping_dict={"hemibrain_v1_2_1": [["tt"]]}, target_labels=["tg"],
        intermediate_mapping_dict={"hemibrain_v1_2_1": [["ii"]]},
        intermediate_labels=["ig"],
    )
    assert m.get_label("hemibrain:v1.2.1", "tt") == "tg"
    assert m.get_label("hemibrain:v1.2.1", "ii") == "ig"


def test_verify_csv_format_read_error(tmp_path):
    d = tmp_path / "adir.csv"
    d.mkdir()
    res = LabelMapper.verify_csv_format(str(d))
    assert not res["valid"]
    assert "Failed to read CSV" in res["errors"][0]


def test_verify_json_format_read_error(tmp_path):
    d = tmp_path / "adir.json"
    d.mkdir()
    res = LabelMapper.verify_json_format(str(d))
    assert not res["valid"]
    assert "Failed to read file" in res["errors"][0]


# --- user-first alignment wrapper (comparison_analyzer, Phase A) ----------


def _wrap_user_mapper():
    """User mapper: two bodyId-keyed labels (the settled-type shape)."""
    return LabelMapper(
        source_mapping_dict={DS2: [["13531", "12254"], ["14633"]]},
        source_labels=["5th-LNv", "LNd_CRY+_ITP+"],
        target_mapping_dict={DS2: [["13531", "12254"], ["14633"]]},
        target_labels=["5th-LNv", "LNd_CRY+_ITP+"],
    )


def _wrap_policy_mapper():
    """Synthesized-policy stand-in: maps raw names (including a USER
    label — the re-resolution hazard the wrapper exists to stop)."""
    return LabelMapper(
        source_mapping_dict={DS2: [["5th-LNv", "5thsLNv_LNd6"]]},
        source_labels=["5thsLNv_family"],
    )


class TestUserFirstLabelMapper:
    def _wrap(self, user=True, policy=True):
        from comparison.comparison_analyzer import _UserFirstLabelMapper

        return _UserFirstLabelMapper(
            _wrap_user_mapper() if user else None,
            _wrap_policy_mapper() if policy else None,
        )

    def test_identity_tier_beats_policy(self):
        """A user label passes through verbatim even when the policy tier
        would re-resolve it — tier 0 is the whole point."""
        w = self._wrap()
        assert w.get_label(DS2, "5th-LNv") == "5th-LNv"
        assert w.is_user_label("5th-LNv")

    def test_identity_is_hemisphere_suffix_aware(self):
        w = self._wrap()
        assert w.is_user_label("5th-LNv_L")
        assert w.get_label(DS2, "5th-LNv_R") == "5th-LNv_R"

    def test_user_member_lookup(self):
        """BodyId-keyed user members resolve to their labels (tier 0b)."""
        w = self._wrap()
        assert w.get_label(DS2, "13531") == "5th-LNv"
        assert w.get_label(DS2, "14633") == "LNd_CRY+_ITP+"

    def test_user_priority_over_policy(self):
        """A name both mappers govern resolves by the USER mapper."""
        user = LabelMapper(
            source_mapping_dict={DS2: [["99999"]]},
            source_labels=["user_label"],
        )
        policy = LabelMapper(
            source_mapping_dict={DS2: [["99999"]]},
            source_labels=["policy_label"],
        )
        from comparison.comparison_analyzer import _UserFirstLabelMapper

        assert _UserFirstLabelMapper(user, policy).get_label(
            DS2, "99999") == "user_label"

    def test_policy_fills_unmapped_names(self):
        w = self._wrap()
        assert w.get_label(DS2, "5thsLNv_LNd6") == "5thsLNv_family"

    def test_total_miss_returns_input(self):
        w = self._wrap()
        assert w.get_label(DS2, "unknown_type") == "unknown_type"
        assert not w.is_user_label("unknown_type")

    def test_tier_chain_is_a_lookup_not_a_column_writer(self):
        """The wrapper answers name questions; it no longer writes a frame.

        ``apply_to_dataframe`` was deleted with the 2026-10-13 boundary: its
        vocabulary is label-or-identity, so re-deriving it over a frame that
        already carries the run's effective key erased the `T*` the run had
        emitted (and was not idempotent). What replaces it is asserted in
        ``test_residual_registry.py::TestVerifierLane`` and in
        ``test_comparison_analyzer_coverage.py`` (loaded keys survive
        verbatim); here the tier chain itself stays the contract.
        """
        w = self._wrap()
        assert not hasattr(w, 'apply_to_dataframe')
        assert w.get_label(DS2, "5th-LNv") == "5th-LNv"        # tier 0 identity
        assert w.get_label(DS2, "13531") == "5th-LNv"          # tier 0b member
        assert w.get_label(DS2, "14633") == "LNd_CRY+_ITP+"    # the other member
        assert w.get_label(DS2, "5thsLNv_LNd6") == "5thsLNv_family"  # tier 1

    def test_is_empty(self):
        w = self._wrap()
        assert not w.is_empty
        from comparison.comparison_analyzer import _UserFirstLabelMapper

        assert _UserFirstLabelMapper(None, None).is_empty

    def test_no_user_mapper_delegates_to_policy(self):
        w = self._wrap(user=False)
        assert w.get_label(DS2, "5thsLNv_LNd6") == "5thsLNv_family"
        assert w.get_label(DS2, "5th-LNv") == "5thsLNv_family"


# --- bodyId fallback lane in apply_to_dataframe (Phase A) -----------------


def test_apply_to_dataframe_bodyid_fallback():
    """A bodyId-keyed mapper renames bodyId-level frames even though its
    members are not type names: rows whose NAME lookup missed retry via
    the matching bodyId column."""
    m = LabelMapper(
        source_mapping_dict={DS2: [["13531", "12254"]]},
        source_labels=["5th-LNv"],
        target_mapping_dict={DS2: [["13531", "12254"]]},
        target_labels=["5th-LNv"],
    )
    df = pd.DataFrame({
        "type_pre": ["5thsLNv_LNd6", "known_type"],
        "bodyId_pre": ["13531", "555"],
        "type_post": ["other", "5thsLNv_LNd6"],
        "bodyId_post": ["777", "12254"],
    })
    out = m.apply_to_dataframe(df, DS2)
    # name miss + mapped bodyId → the bodyId's label
    assert out["std_label_pre"].iloc[0] == "5th-LNv"
    # name miss + unmapped bodyId → the raw type name survives (an
    # unmapped bodyId must never displace the type name)
    assert out["std_label_pre"].iloc[1] == "known_type"
    assert out["std_label_post"].iloc[0] == "other"
    assert out["std_label_post"].iloc[1] == "5th-LNv"


def test_apply_to_dataframe_bodyid_fallback_name_mapped_wins():
    """When the NAME maps, the bodyId is not consulted (name-first)."""
    m = LabelMapper(
        source_mapping_dict={DS2: [["13531"], ["legacy_name"]]},
        source_labels=["5th-LNv", "by_name"],
        target_mapping_dict={DS2: [["13531"], ["legacy_name"]]},
        target_labels=["5th-LNv", "by_name"],
    )
    df = pd.DataFrame({
        "type_pre": ["legacy_name"],
        "bodyId_pre": ["13531"],
    })
    out = m.apply_to_dataframe(df, DS2)
    assert out["std_label_pre"].iloc[0] == "by_name"


def test_apply_to_dataframe_bodyid_fallback_noop_for_name_mappers():
    """A name-keyed mapper's members are not bodyIds — the fallback lane
    never fires, so behavior is unchanged."""
    m = _dict_mapper()
    df = pd.DataFrame({
        "type_pre": ["aMe12"],
        "bodyId_pre": ["13531"],
    })
    out = m.apply_to_dataframe(df, DS1)
    assert out["std_label_pre"].iloc[0] == "aMe12_grp1"
