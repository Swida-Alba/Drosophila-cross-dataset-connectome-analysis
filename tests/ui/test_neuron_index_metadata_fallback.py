"""A downloaded neuron-metadata table stands in for a missing index.

Fresh installs ship only the MCNS / FAFB / BANC seed indexes; every other
dataset has to be pulled first.  The metadata pull itself already contains the
type/bodyId columns the cross-dataset type-mapping preview needs, so the
index readers fall back to it until the materialized index exists.
"""

import polars as pl

from ui import neuron_index as ni


HEMI = "hemibrain:v1.2.1"
FOLDER = "hemibrain_v1_2_1"


def _write_metadata_csv(root, rows=None):
    path = root / "datasets" / FOLDER / f"{FOLDER}_allneurons_neuron_df.csv"
    path.parent.mkdir(parents=True)
    frame = pl.DataFrame(
        rows or {
            "bodyId": ["100", "101", "102"],
            "type": ["l-LNv", "l-LNv", "aMe12"],
            "instance": ["l-LNv_R1", "l-LNv_R2", "aMe12_R1"],
            "class": ["clock", "clock", "clock"],
            "hemibrainType": ["l-LNv", "l-LNv", "aMe12"],
            "flywireType": ["l-LNv", "l-LNv", "aMe12"],
        }
    )
    frame.write_csv(path)
    return path


def test_metadata_only_dataset_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(tmp_path)

    assert ni.neuron_index_available(HEMI) is True
    assert ni.neuron_index_available("missing:v1.0") is False


def test_cross_match_and_coverage_read_the_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(tmp_path)

    index = ni._load_cross_match_index(HEMI)
    assert index is not None
    assert index.frame.height == 3
    assert "type" in index.frame.columns
    assert "class" in index.frame.columns
    assert ni.count_type_in_index(index, "l-LNv") == 2

    coverage = ni._load_coverage_index(HEMI)
    assert coverage is not None
    assert {"bodyId", "type", "hemibrainType"}.issubset(
        set(coverage.frame.columns))
    assert coverage.frame.filter(
        pl.col("bodyId") == "100").height == 1

    alias = ni._load_alias_index(HEMI)
    assert alias is not None
    assert {"bodyId", "type", "instance", "class"}.issubset(
        set(alias.frame.columns))


def test_parquet_metadata_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "datasets" / FOLDER / f"{FOLDER}_allneurons_neuron_df.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "bodyId": ["100", "101"],
            "type": ["l-LNv", "aMe12"],
            "instance": ["l-LNv_R1", "aMe12_R1"],
        }
    ).write_parquet(path)

    assert ni.neuron_index_available(HEMI) is True
    index = ni._load_cross_match_index(HEMI)
    assert index is not None
    assert ni.count_type_in_index(index, "l-LNv") == 1


def test_materialized_index_wins_over_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(tmp_path)
    index_path = ni.neuron_index_path(HEMI)
    index_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {"bodyId": ["900"], "type": ["FromIndex"]}
    ).write_parquet(index_path)

    assert ni.neuron_index_source_path(HEMI) == index_path
    index = ni._load_cross_match_index(HEMI)
    assert index is not None
    assert ni.count_type_in_index(index, "FromIndex") == 1
    assert ni.count_type_in_index(index, "l-LNv") == 0


def test_explicit_cache_dir_stays_hermetic(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(tmp_path)

    assert ni.neuron_index_available(
        HEMI, cache_dir=tmp_path / "indexes") is False


def test_dataset_list_includes_metadata_only_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(tmp_path)
    monkeypatch.setattr("ui.config.DATASETS", [HEMI])

    assert ni.datasets_with_cached_indexes() == [HEMI]
    assert ni.datasets_with_cached_indexes(cache_dir=tmp_path / "indexes") == []


def test_metadata_without_a_type_column_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ni, "PROJECT_ROOT", tmp_path)
    _write_metadata_csv(
        tmp_path,
        rows={
            "bodyId": ["100", "101"],
            "instance": ["l-LNv_R1", "l-LNv_R2"],
        },
    )

    assert ni.neuron_index_available(HEMI) is False
    assert ni._load_cross_match_index(HEMI) is None