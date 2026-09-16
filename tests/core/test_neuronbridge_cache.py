from pathlib import Path

import pandas as pd

from neuronbridge_cache import NeuronBridgeParquetCache
import neuronbridge_finder as neuronbridge_finder_module
from neuronbridge_finder import NeuronBridgeFinder


class _DummyClient:
    version = "test-version"


def _finder(tmp_path, monkeypatch):
    monkeypatch.setattr(neuronbridge_finder_module, "NBClient", _DummyClient)
    return NeuronBridgeFinder(
        cache_folder=str(tmp_path),
        # This module exercises the cache paths, which are off by default
        # since the output-modes change.
        use_cache=True,
        verbose=False,
        max_workers=1,
    )


def test_image_parquet_schema_and_deduplication(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "bodyId": 123,
                "score": 0.4,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "dataset": "hemibrain:v1.2.1",
                "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "T1",
                "instance": "I1",
                "status": "Traced",
                "dataset_folder": "discarded",
            },
            {
                "bodyId": 123,
                "score": 0.9,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "dataset": "hemibrain:v1.2.1",
                "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "T1",
                "instance": "I1",
                "status": "Traced",
            },
        ]
    )

    path = cache.save_image("lm-1", "cds", frame)
    assert path is not None and path.suffix == ".parquet"
    loaded = cache.load_image("lm-1", "cds")

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["score"] == 0.9
    assert list(loaded.columns) == [
        "bodyId",
        "score",
        "image_id",
        "lm_sample",
        "match_type",
        "dataset",
        "library",
        "type",
        "instance",
        "status",
    ]


def test_flywire_image_cache_preserves_large_body_ids(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path)
    body_id = "72057594037927937"
    frame = pd.DataFrame([
        {
            "bodyId": body_id,
            "score": 0.8,
            "image_id": "em-1",
            "lm_sample": "lm-1",
            "match_type": "cds",
            "dataset": "flywire_FAFB_v783",
        }
    ])

    cache.save_image("lm-1", "cds", frame)
    loaded = cache.load_image("lm-1", "cds")

    assert loaded is not None
    assert loaded.iloc[0]["bodyId"] == body_id
    assert isinstance(loaded.iloc[0]["bodyId"], str)


def test_id_parquet_reload_preserves_score_order(tmp_path):
    cache = NeuronBridgeParquetCache(tmp_path)
    cache.save_id(
        "123_cds_any",
        pd.DataFrame(
            [
                {"line": "LH-low", "library": "lib", "score": 0.2, "image_id": "lm-2", "match_type": "cds"},
                {"line": "LH-high", "library": "lib", "score": 0.9, "image_id": "lm-1", "match_type": "cds"},
            ]
        ),
    )
    loaded = cache.load_id("123_cds_any")
    assert loaded is not None
    assert loaded["line"].tolist() == ["LH-high", "LH-low"]


def test_id_key_ignores_non_id_parameters(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    finder.region = "Brain"
    finder.max_api_images_per_line = 3
    first = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    finder.region = "VNC"
    finder.max_api_images_per_line = 50
    second = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    assert first == second == "123_cds_hemibrain:v1.2.1"

    old_cache = NeuronBridgeParquetCache(tmp_path, version="old-version")
    old_cache.save_id(
        first,
        pd.DataFrame(
            [{"line": "LH1", "library": "lib", "score": 0.8, "image_id": "lm1", "match_type": "cds"}]
        ),
    )
    assert not finder._is_cached("id_to_lines", first)


def test_both_id_results_are_reconstructed_from_algorithm_tables(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)

    def fake_em_matches(body_id, match_type=None, expected_dataset=None, **kwargs):
        return [
            {
                "line": "LH1",
                "library": "lib",
                "score": 0.8 if match_type == "cds" else 0.7,
                "image_id": f"{match_type}-image",
                "match_type": match_type,
            }
        ]

    monkeypatch.setattr(finder, "_get_em_matches", fake_em_matches)
    combined = finder.id_to_lines(123, match_type="both", expected_dataset="ds:v1")

    assert len(combined) == 1
    assert combined.iloc[0]["match_type"] == "both"
    parquet_root = Path(tmp_path) / "parquet" / "v_test-version"
    assert not list((parquet_root / "id_to_lines").glob("*both*"))
    assert (parquet_root / "id_to_lines" / "id_to_lines_123_cds_ds_v1.parquet").exists()
    assert (parquet_root / "id_to_lines" / "id_to_lines_123_pppm_ds_v1.parquet").exists()

    monkeypatch.setattr(
        finder,
        "_get_em_matches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    cached_combined = finder.id_to_lines(123, match_type="both", expected_dataset="ds:v1")
    assert cached_combined[["line", "combined_rank"]].equals(
        combined[["line", "combined_rank"]]
    )


def test_line_results_use_image_cache_without_line_snapshot(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    monkeypatch.setattr(
        finder,
        "_get_lm_matches",
        lambda line_name, match_type=None: [
            {
                "bodyId": "123",
                "dataset": "ds:v1",
                "instance": "I1",
                "type": "T1",
                "status": "Traced",
                "score": 0.8,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": match_type,
                "library": "lib",
            },
            {
                "bodyId": "124",
                "dataset": "ds:v1",
                "instance": "I2",
                "type": "T2",
                "status": "Traced",
                "score": 0.7,
                "image_id": "em-2",
                "lm_sample": "lm-1",
                "match_type": match_type,
                "library": "lib",
            },
        ],
    )

    top_one = finder.line_to_neuron("LH1", top_n=1)
    all_rows = finder.line_to_neuron("LH1", top_n=-1)

    assert len(top_one) == 1
    assert len(all_rows) == 2
    assert not list(Path(tmp_path).glob("line_to_neuron_*.parquet"))
    assert not list(Path(tmp_path).glob("line_to_neuron_*.csv"))


def test_legacy_csv_is_not_loaded_and_clear_cache_removes_known_legacy_files(
    tmp_path, monkeypatch
):
    finder = _finder(tmp_path, monkeypatch)
    image_dir = Path(tmp_path) / "image_cache"
    image_dir.mkdir(parents=True, exist_ok=True)

    line_csv = Path(tmp_path) / "line_to_neuron_LH1_cds_Brain_all.csv"
    pd.DataFrame(
        [
            {
                "bodyId": 123,
                "dataset": "ds:v1",
                "instance": "I1",
                "type": "T1",
                "status": "Traced",
                "score": 0.8,
                "image_id": "em-1",
                "lm_sample": "lm-1",
                "match_type": "cds",
                "library": "lib",
            }
        ]
    ).to_csv(line_csv, index=False)

    id_csv = Path(tmp_path) / "id_to_lines_123_cds_ds_v1_Brain_all.csv"
    pd.DataFrame(
        [
            {
                "line": "LH1",
                "library": "lib",
                "score": 0.8,
                "image_id": "lm-1",
                "match_type": "cds",
            }
        ]
    ).to_csv(id_csv, index=False)

    image_csv = image_dir / "cds_lm-1.csv"
    pd.DataFrame([{"bodyId": 123, "score": 0.8}]).to_csv(image_csv, index=False)

    # Legacy files do not participate in cache hits.
    assert finder._load_from_cache(
        "id_to_lines", "123_cds_ds:v1_Brain_all"
    ) is None
    assert finder._load_image_cache("lm-1", "cds") is None

    finder.clear_cache()
    assert not line_csv.exists()
    assert not id_csv.exists()
    assert not image_csv.exists()


class _EMImage:
    type = "EMImage"
    files = object()

    def __init__(self, published_name):
        self.publishedName = published_name


class _MetadataClient:
    version = "test-version"

    def __init__(self):
        self.calls = 0
        self.images = [
            _EMImage("male-cns:v0.9:12211"),
            _EMImage("male-cns:v1.0:12211"),
        ]

    def get_em_images(self, body_id):
        self.calls += 1
        return list(self.images)


def test_dataset_version_selection_does_not_fall_back_to_wrong_image(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    client = _MetadataClient()
    finder._client = client

    selected = finder._get_em_image_for_dataset(12211, "male-cns:v1.0")

    assert selected is client.images[1]
    assert finder._get_em_image_for_dataset(12211, "male-cns:v2.0") is None


def test_batched_metadata_validation_carries_selected_image(tmp_path, monkeypatch):
    finder = _finder(tmp_path, monkeypatch)
    client = _MetadataClient()
    finder._client = client

    result = finder._validate_body_ids_parallel(
        [{"bodyId": 12211, "dataset": "male-cns:v1.0"}],
        max_workers=1,
    )

    assert client.calls == 1
    assert result[12211]["dataset"] == "male-cns:v1.0"
    assert result[12211]["em_image"] is client.images[1]
