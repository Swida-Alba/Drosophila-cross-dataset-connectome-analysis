"""Regression tests for lossless FlyWire body-ID handling."""

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flywire_ids import (  # noqa: E402
    FlyWireBodyIdError,
    body_id_to_api_int,
    neuprint_dataset_name,
    normalize_flywire_body_id,
    normalize_flywire_id_columns,
    resolve_flywire_dataset_dir,
)


LARGE_ID = "72057594037927937"


def test_normalizer_preserves_exact_large_decimal_ids():
    assert normalize_flywire_body_id(LARGE_ID) == LARGE_ID
    assert normalize_flywire_body_id(int(LARGE_ID)) == LARGE_ID
    assert normalize_flywire_body_id("000" + LARGE_ID) == LARGE_ID
    assert normalize_flywire_body_id("123.0") == "123"
    assert body_id_to_api_int(LARGE_ID) == int(LARGE_ID)


@pytest.mark.parametrize("value", [float(LARGE_ID), "72057594037927937.0", 1.5, -1, None])
def test_normalizer_rejects_lossy_or_invalid_ids(value):
    with pytest.raises(FlyWireBodyIdError):
        normalize_flywire_body_id(value)


def test_dataframe_normalization_uses_string_ids_without_rounding():
    frame = pd.DataFrame({"bodyId": [LARGE_ID, "000123"], "weight": [1, 2]})

    normalize_flywire_id_columns(frame, ["bodyId"])

    assert frame["bodyId"].tolist() == [LARGE_ID, "123"]
    assert frame["bodyId"].map(type).eq(str).all()


def test_banc_dataset_resolution_does_not_fallback_to_fafb(tmp_path):
    fafb_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
    fafb_dir.mkdir(parents=True)

    assert resolve_flywire_dataset_dir(tmp_path, "banc_v626") is None
    assert resolve_flywire_dataset_dir(tmp_path, "flywire_FAFB_v783") == fafb_dir


def test_flywire_prepared_tables_are_read_without_raw_fafb_fallback(tmp_path):
    import fafb_utils

    banc_dir = tmp_path / "banc_v626"
    banc_dir.mkdir()
    neuron_path = banc_dir / "banc_v626_allneurons_neuron_df.parquet"
    connection_path = banc_dir / "banc_v626_merged_connections.parquet"
    pd.DataFrame({"bodyId": [LARGE_ID], "type": ["T"]}).to_parquet(
        neuron_path, index=False
    )
    pd.DataFrame({
        "bodyId_pre": [LARGE_ID],
        "bodyId_post": ["123"],
        "weight": [1],
    }).to_parquet(connection_path, index=False)

    neuron_file, connection_file = fafb_utils.prepare_flywire_data(banc_dir)

    assert neuron_file.endswith("_allneurons_neuron_df.parquet")
    assert connection_file.endswith("_merged_connections.parquet")


def test_missing_banc_tables_raise_instead_of_using_fafb_raw_names(tmp_path):
    import fafb_utils

    banc_dir = tmp_path / "banc_v626"
    (banc_dir / "downloads").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Prepared BANC tables"):
        fafb_utils.prepare_flywire_data(banc_dir)


def test_morphology_type_map_reads_flywire_ids_as_strings(tmp_path):
    import morphology

    dataset = "banc_v626"
    table_dir = tmp_path / "datasets" / dataset
    table_dir.mkdir(parents=True)
    pd.DataFrame({
        "bodyId": [LARGE_ID],
        "type": ["T"],
        "instance": ["T_L"],
    }).to_parquet(
        table_dir / f"{dataset}_allneurons_neuron_df.parquet", index=False
    )

    type_map, instance_map = morphology._load_neuron_type_map(
        dataset, str(tmp_path)
    )

    assert type_map == {LARGE_ID: "T"}
    assert instance_map == {LARGE_ID: "T_L"}


def test_morphology_vector_cache_load_keeps_flywire_body_ids_as_strings(tmp_path):
    import morphology

    dataset = "flywire_FAFB_v783"
    cache = morphology.SkeletonVectorCache(
        dataset, project_root=str(tmp_path), raw_only=True, verbose=False
    )
    row = {
        "bodyId": LARGE_ID,
        "rep": "skeleton",
        **{name: 0.0 for name in morphology.MORPHOMETRIC_FEATURES},
        **{f"pv_{index}": 0.0 for index in range(morphology.PERSISTENCE_DIM)},
        "type": "T",
        "instance": "T_L",
    }
    cache.parquet_path.parent.mkdir(parents=True)
    pd.DataFrame([row]).to_parquet(cache.parquet_path, index=False)

    loaded = cache.load()

    assert loaded["bodyIds"].dtype == object
    assert loaded["bodyIds"].tolist() == [LARGE_ID]
    assert loaded["df"]["bodyId"].map(type).eq(str).all()


def test_statvis_polars_reader_validates_flywire_ids(tmp_path):
    import polars as pl
    import statvis

    path = tmp_path / "neurons.parquet"
    pd.DataFrame({"bodyId": [LARGE_ID], "type": ["T"]}).to_parquet(
        path, index=False
    )

    frame = statvis._load_local_neuron_df_cached(str(path), True)

    assert frame.schema["bodyId"] == pl.Utf8
    assert frame["bodyId"].to_list() == [LARGE_ID]


def test_visualization_synapse_lookup_does_not_cross_fallback_to_fafb(tmp_path):
    from visualize_skeleton import VisualizeSkeleton

    datasets = tmp_path / "datasets"
    fafb_dir = datasets / "flywire_FAFB_v783"
    banc_dir = datasets / "banc_v626"
    fafb_dir.mkdir(parents=True)
    banc_dir.mkdir(parents=True)
    (fafb_dir / "flywire_FAFB_v783_synapse_table.parquet").touch()

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.script_path = str(tmp_path)
    visualizer.dataset = "banc_v626"

    assert visualizer._get_synapse_table_path() is None


def test_neuronbridge_reads_exact_flywire_parquet_table(tmp_path):
    from neuronbridge_finder import NeuronBridgeFinder

    datasets = tmp_path / "datasets"
    banc_dir = datasets / "banc_v626"
    banc_dir.mkdir(parents=True)
    pd.DataFrame({"bodyId": [LARGE_ID], "type": ["T"]}).to_parquet(
        banc_dir / "banc_v626_allneurons_neuron_df.parquet",
        index=False,
    )

    finder = NeuronBridgeFinder(
        datasets_path=str(datasets), use_cache=False, verbose=False
    )
    frame = finder._load_neuron_df_for_dataset("banc_v626")

    assert frame is not None
    assert frame["bodyId"].tolist() == [LARGE_ID]
    assert frame["bodyId"].map(type).eq(str).all()


def test_folder_spellings_resolve_to_the_names_the_neuprint_server_knows():
    """A NeuPrint release's folder spelling resolves against everything local
    and nothing remote: r25 ran ``--target hemibrain_v1_2_1`` and the server
    answered "Dataset 'hemibrain_v1_2_1' does not exist", so the run lost every
    target skeleton — zero Track-A scores, no null-bar sample, scenes with no
    target neurons — while its tables, profile cache and folder names all worked.
    """
    cases = {
        "hemibrain_v1_2_1": "hemibrain:v1.2.1",
        "male-cns_v0_9": "male-cns:v0.9",
        "male-cns_v1_0": "male-cns:v1.0",
        "manc_v1_2_3": "manc:v1.2.3",
        "optic-lobe_v1_1": "optic-lobe:v1.1",
        # already-correct and version-less server names pass through, so the
        # translation is idempotent
        "hemibrain:v1.2.1": "hemibrain:v1.2.1",
        "mushroombody": "mushroombody",
        # local releases are NOT NeuPrint datasets and must keep their own
        # identifiers (FAFB's name even contains a version suffix)
        "flywire_FAFB_v783": "flywire_FAFB_v783",
        "banc_v888": "banc_v888",
        "flywire_BANC_v626": "banc_v626",
    }
    for folder, expected in cases.items():
        once = neuprint_dataset_name(folder)
        assert once == expected
        assert neuprint_dataset_name(once) == once
