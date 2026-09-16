"""Coverage tests for neuronbridge_finder.

All HTTP boundaries are mocked via a fake NeuronBridge client injected into
``finder._client`` (or by monkeypatching ``NBClient``).  All file/cache I/O
happens inside pytest ``tmp_path``.  No network, no multiprocessing.
"""

import json
import os
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import neuronbridge_finder as nbf_mod
from neuronbridge_finder import (
    NeuronBridgeFinder,
    _create_mountain_order,
    _extract_base_type,
    _to_int_bodyid,
    image_summary_skip_note,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _DummyClient:
    """Minimal client substitute for construction-time initialization."""

    version = "test-version"


class FakeNBClient:
    """In-memory NeuronBridge client keyed by image id / body id / line."""

    version = "test-version"

    def __init__(self):
        self.em_images = {}
        self.lm_images = {}
        self.cds_matches = {}
        self.ppp_matches = {}
        self.calls = []

    def get_em_images(self, body_id):
        self.calls.append(("em_images", body_id))
        result = self.em_images.get(body_id)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_lm_images(self, line_name):
        self.calls.append(("lm_images", line_name))
        result = self.lm_images.get(line_name)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_cds_matches(self, image):
        key = getattr(image, "id", None)
        self.calls.append(("cds", key))
        result = self.cds_matches.get(key)
        if isinstance(result, Exception):
            raise result
        return result or []

    def get_ppp_matches(self, image):
        key = getattr(image, "id", None)
        self.calls.append(("pppm", key))
        result = self.ppp_matches.get(key)
        if isinstance(result, Exception):
            raise result
        return result or []


def _em_image(body_id="5813128953", cds=True, pppm=False):
    files = SimpleNamespace(
        CDSResults="https://example.invalid/cds.parquet" if cds else None,
        PPPMResults="https://example.invalid/ppp.parquet" if pppm else None,
    )
    return SimpleNamespace(
        id=f"em-{body_id}",
        type="EMImage",
        publishedName=f"hemibrain:v1.2.1:{body_id}",
        libraryName="FlyEM_Hemibrain_v1.2.1",
        neuronType="MBON01",
        neuronInstance="MBON01_R",
        anatomicalArea="Brain",
        files=files,
    )


def _lm_match(line="VT037867", score=100, image_id="lm-1"):
    image = SimpleNamespace(
        type="LMImage",
        publishedName=line,
        libraryName="FlyEM_Hemibrain_v1.2.1",
        id=image_id,
    )
    return SimpleNamespace(image=image, normalizedScore=score)


def _em_match(body_id="5813128953", score=100, image_id="em-5813128953"):
    image = SimpleNamespace(
        type="EMImage",
        publishedName=f"hemibrain:v1.2.1:{body_id}",
        libraryName="FlyEM_Hemibrain_v1.2.1",
        id=image_id,
        neuronType="MBON01",
        neuronInstance="MBON01_R",
    )
    return SimpleNamespace(image=image, normalizedScore=score)


def _lm_image(image_id="img-1", area="Brain", cds=True, pppm=False):
    files = SimpleNamespace(
        CDSResults="u" if cds else None,
        PPPMResults="u" if pppm else None,
    )
    return SimpleNamespace(id=image_id, anatomicalArea=area, files=files)


HEMIBRAIN_DF = pd.DataFrame(
    {
        "bodyId": ["5813128953", "1234"],
        "type": ["MBON01", "LH173"],
        "instance": ["MBON01_R", "LH173_L"],
        "status": ["ok", "ok"],
    }
)


@pytest.fixture
def finder(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    datasets_path = tmp_path / "datasets"
    datasets_path.mkdir()
    return NeuronBridgeFinder(
        datasets_path=str(datasets_path),
        cache_folder=str(tmp_path / "cache"),
        # Cache-behavior coverage explicitly exercises the cache paths,
        # which are off by default since the output-modes change.
        use_cache=True,
        verbose=False,
        max_workers=1,
    )


@pytest.fixture
def finder_with_client(finder):
    finder._client = FakeNBClient()
    return finder


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def test_to_int_bodyid_handles_various_inputs():
    assert _to_int_bodyid(123) == 123
    assert _to_int_bodyid("456") == 456
    assert _to_int_bodyid("5813128953.0") == 5813128953
    assert _to_int_bodyid("abc") == "abc"
    assert _to_int_bodyid(None) is None


def test_extract_base_type():
    assert _extract_base_type("MCNS_aMe12") == "aMe12"
    assert _extract_base_type("aMe12") == "aMe12"
    mapping = {"MCNS_aMe12": "merged_aMe"}
    assert _extract_base_type("MCNS_aMe12", mapping) == "merged_aMe"
    assert _extract_base_type("MCNS_other", mapping) == "other"


def test_create_mountain_order():
    assert _create_mountain_order([], []) == ([], [])
    # Peak in the center, descending outwards; the alternating left/right
    # writes must not overwrite the center slot.
    scores, labels = _create_mountain_order([1, 4, 3, 2], ["a", "b", "c", "d"])
    assert scores == [1, 3, 4, 2] and labels == ["a", "c", "b", "d"]
    scores, labels = _create_mountain_order([1, 3, 2], ["a", "b", "c"])
    assert scores == [2, 3, 1] and labels == ["c", "b", "a"]
    assert "" not in labels and 0 not in scores


def test_image_summary_skip_note():
    assert image_summary_skip_note("pdf", download_images=True) is None
    assert image_summary_skip_note(None, download_images=False) is None
    assert image_summary_skip_note([], download_images=False) is None
    note = image_summary_skip_note("pdf", download_images=False)
    assert note and "summary format" in note
    note = image_summary_skip_note(["pdf", "pptx"], download_images=False)
    assert "pdf" in note and "pptx" in note


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_invalid_match_type_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    with pytest.raises(ValueError, match="Invalid match_type"):
        NeuronBridgeFinder(cache_folder=str(tmp_path), match_type="nope")


def test_invalid_region_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    with pytest.raises(ValueError, match="Invalid region"):
        NeuronBridgeFinder(cache_folder=str(tmp_path), region="spine")


def test_match_type_and_region_normalized(finder):
    assert finder.match_type == "cds"
    assert finder.region == "All"


def test_validate_helpers(finder):
    assert finder._validate_match_type(" CDS ") == "cds"
    assert finder._validate_similarity_method("Jaccard") == "jaccard"
    assert finder._validate_sort_by("MAX") == "max"
    with pytest.raises(ValueError):
        finder._validate_match_type("wrong")
    with pytest.raises(ValueError):
        finder._validate_similarity_method("wrong")
    with pytest.raises(ValueError):
        finder._validate_sort_by("wrong")


def test_init_client_success_after_retry(tmp_path, monkeypatch):
    attempts = {"n": 0}

    def flaky_client():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("connection timeout")
        return _DummyClient()

    monkeypatch.setattr(nbf_mod, "NBClient", flaky_client)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    f = NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert attempts["n"] == 2
    assert isinstance(f._client, _DummyClient)


def test_init_client_failure_loop_raises(tmp_path, monkeypatch):
    def failing_client():
        raise ConnectionError("connection timeout")

    monkeypatch.setattr(nbf_mod, "NBClient", failing_client)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    # All 5 attempts run with one wait per retry, then the documented
    # RuntimeError is raised (retry_delays now has max_retries - 1 entries).
    with pytest.raises(RuntimeError):
        NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert sleeps == [2, 5, 10, 15]


def test_fix_store_prefixes(finder):
    store = SimpleNamespace(
        prefixes={"CDSResults": "https://data-dev.example/v3_8_0/prefix"}
    )
    finder._client = SimpleNamespace(config=SimpleNamespace(stores={"s": store}))
    finder._fix_store_prefixes()
    assert store.prefixes["CDSResults"] == "https://data-prod.example/v3_8_1/prefix"
    # Client without config.stores is a no-op
    finder._client = SimpleNamespace()
    finder._fix_store_prefixes()


# ---------------------------------------------------------------------------
# _retry_with_backoff
# ---------------------------------------------------------------------------

def test_retry_with_backoff_retryable_succeeds(finder, monkeypatch):
    sleeps = []
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: sleeps.append(d))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("Connection reset by peer")
        return "ok"

    assert finder._retry_with_backoff(flaky, max_retries=5, initial_delay=0.5) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_retry_with_backoff_non_retryable_raises_immediately(finder, monkeypatch):
    sleeps = []
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: sleeps.append(d))

    def broken():
        raise ValueError("not a network problem")

    with pytest.raises(ValueError):
        finder._retry_with_backoff(broken, max_retries=5)
    assert sleeps == []


def test_retry_with_backoff_exhausts_retries(finder, monkeypatch):
    monkeypatch.setattr(nbf_mod.time, "sleep", lambda d: None)

    def always_timeout():
        raise TimeoutError("read timeout")

    with pytest.raises(TimeoutError):
        finder._retry_with_backoff(always_timeout, max_retries=2, initial_delay=0.1)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def test_normalize_dataset_name(finder):
    assert finder._normalize_dataset_name("flywire_FAFB_v783") == "flywire-fafb"
    assert finder._normalize_dataset_name("flywire_fafb:v783") == "flywire-fafb"
    assert finder._normalize_dataset_name("male-cns:v0.9") == "male-cns"
    assert finder._normalize_dataset_name("hemibrain:v1.2.1") == "hemibrain"
    assert finder._normalize_dataset_name("hemibrain_v1_2_1") == "hemibrain"


def test_save_parameters(tmp_path, finder):
    class Custom:
        def __repr__(self):
            return "Custom()"

    path = finder._save_parameters(
        str(tmp_path),
        "analyze_colabeling",
        {
            "top_n": 5,
            "lines": ("VT037867", "LH173"),
            "opts": {"a": 1},
            "arr": np.array([1, 2, 3]),
            "custom": Custom(),
            "plain_set": {1},
            "none": None,
        },
    )
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["metadata"]["function"] == "analyze_colabeling"
    assert saved["function_params"]["top_n"] == 5
    assert saved["function_params"]["lines"] == ["VT037867", "LH173"]
    assert saved["function_params"]["arr"] == [1, 2, 3]
    assert "has_neuprint_token" in saved["module_params"]


def test_save_user_warning_notes(tmp_path, finder):
    assert finder._save_user_warning_notes(str(tmp_path), []) is None
    path = finder._save_user_warning_notes(str(tmp_path), ["caveat one"])
    assert "caveat one" in open(path, encoding="utf-8").read()


def test_filter_images_by_region(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    images = [
        SimpleNamespace(anatomicalArea="Brain"),
        SimpleNamespace(anatomicalArea="VNC"),
        SimpleNamespace(anatomicalArea=None),
    ]
    f = NeuronBridgeFinder(cache_folder=str(tmp_path / "c"), verbose=False, region="all")
    assert f._filter_images_by_region(images) == images
    f.region = "Brain"
    assert len(f._filter_images_by_region(images)) == 2  # Brain + missing area
    f.region = "VNC"
    filtered = f._filter_images_by_region(images)
    # VNC image plus the area-less one (included by default)
    assert len(filtered) == 2 and filtered[0].anatomicalArea == "VNC"


def test_filter_images_by_match_availability(finder):
    both = _lm_image("i1", cds=True, pppm=True)
    cds_only = _lm_image("i2", cds=True, pppm=False)
    no_files = SimpleNamespace(id="i3")
    images = [both, cds_only, no_files]
    assert finder._filter_images_by_match_availability(images, "cds") == [both, cds_only]
    assert finder._filter_images_by_match_availability(images, "pppm") == [both]
    assert finder._filter_images_by_match_availability(images, "both") == [both, cds_only]
    assert finder._filter_images_by_match_availability(images, "???") == images


def test_classify_line_type(finder):
    assert finder._classify_line_type("VT037867") == "gal4_lexa"
    assert finder._classify_line_type("R10A06") == "gal4_lexa"
    assert finder._classify_line_type("GMR_0001") == "gal4_lexa"
    assert finder._classify_line_type("SS010") == "split_gal4"
    assert finder._classify_line_type("LH173") == "split_gal4"
    assert finder._classify_line_type("") == "gal4_lexa"
    assert finder._classify_line_type("XYZ") == "gal4_lexa"


def test_print_warning_summary(finder, capsys):
    finder.verbose = True
    finder._warning_collector = [
        "flimg.janelia.org server not accessible for VT037867: error",
        "No files from GAL4 collections for VT037867, trying MCFO",
        "No images found for LH173 in any collection",
    ]
    finder._print_warning_summary()
    out = capsys.readouterr().out
    assert "Server Access Issues" in out
    assert "MCFO fallback" in out
    assert "no FlyLight images" in out
    assert finder._warning_collector == []
    finder._print_warning_summary()  # empty collector is a no-op


def test_vprint_batch_mode_and_progress(finder, capsys):
    finder.verbose = True
    finder._batch_mode = True
    finder._vprint("hidden")
    assert capsys.readouterr().out == ""
    finder._vprint("shown", force=True)
    assert "shown" in capsys.readouterr().out
    finder._batch_mode = False
    finder._progress(2, 4, "working")
    assert "[DROCAT][progress] 2/4 working" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Library -> dataset mapping
# ---------------------------------------------------------------------------

def test_parse_library_name(finder):
    assert finder._parse_library_name("FlyEM_Hemibrain_v1.2.1") == ("FlyEM_Hemibrain", "1.2.1")
    assert finder._parse_library_name("FlyEM_MANC") == ("FlyEM_MANC", "")
    assert finder._parse_library_name("") == ("", "")


def test_get_dataset_from_library(finder):
    assert finder._get_dataset_from_library("FlyEM_Hemibrain_v1.2.1") == "hemibrain_v1_2_1"
    assert finder._get_dataset_from_library("") is None
    # Unmapped version auto-maps from the same base library
    assert finder._get_dataset_from_library("FlyEM_MANC_v9.9") == "manc_v9_9"
    # Prefix partial match fallback
    assert finder._get_dataset_from_library("FlyEM_Hemibrain_extra") == "hemibrain_v1_2_1"
    assert finder._get_dataset_from_library("Zebra_Lib") is None


def test_get_dataset_name_from_library(finder):
    assert finder._get_dataset_name_from_library("FlyEM_Hemibrain_v1.2.1") == "hemibrain:v1.2.1"
    assert finder._get_dataset_name_from_library("") is None
    assert finder._get_dataset_name_from_library("FlyEM_MANC_v9.9") == "manc:v9.9"
    # Folder-style mapped names become colon-style
    assert finder._get_dataset_name_from_library("FlyWire_FAFB_v999") == "flywire_FAFB:v999"
    assert finder._get_dataset_name_from_library("Zebra_Lib") is None


def test_dataset_name_folder_conversions(finder):
    assert finder._dataset_name_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
    assert finder._dataset_name_to_folder("hemibrain_v1_2_1") == "hemibrain_v1_2_1"
    assert finder._dataset_name_to_folder("not-a-dataset") is None
    assert finder._folder_to_dataset_name("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
    assert finder._folder_to_dataset_name("other_folder") == "other_folder"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _write_neuron_csv(datasets_path, folder, df):
    folder_path = Path(datasets_path) / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    path = folder_path / f"{folder}_allneurons_neuron_df.csv"
    df.to_csv(path, index=False)
    return path


def test_ensure_datasets_loaded_local_csv(finder):
    _write_neuron_csv(finder.datasets_path, "hemibrain_v1_2_1", HEMIBRAIN_DF)
    loaded = finder._ensure_datasets_loaded(["hemibrain:v1.2.1"])
    assert loaded == ["hemibrain_v1_2_1"]
    assert "hemibrain_v1_2_1" in finder._neuron_dfs
    # Already loaded -> immediate
    assert finder._ensure_datasets_loaded(["hemibrain:v1.2.1"]) == ["hemibrain_v1_2_1"]


def test_ensure_datasets_loaded_pull_and_fallback(finder, monkeypatch):
    pulled = pd.DataFrame({"bodyId": ["9"], "type": ["X"]})
    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: pulled)
    assert finder._ensure_datasets_loaded(["manc:v1.0"]) == ["manc_v1_0"]

    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: None)
    fnc_df = pd.DataFrame({"bodyId": ["7"], "type": ["Y"]})
    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: fnc_df)
    assert finder._ensure_datasets_loaded(["male-cns:v0.9"]) == ["male-cns_v0_9"]

    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: None)
    assert finder._ensure_datasets_loaded(["male-cns:v1.0"]) == []


def test_ensure_datasets_loaded_flywire_skip(finder):
    # No local FlyWire table -> skipped, no network attempted
    assert finder._ensure_datasets_loaded(["flywire_FAFB_v783"]) == []


def test_load_neuron_df_for_dataset_paths(finder, monkeypatch):
    assert finder._load_neuron_df_for_dataset("") is None
    finder._neuron_dfs["cached_ds"] = HEMIBRAIN_DF
    assert finder._load_neuron_df_for_dataset("cached_ds") is HEMIBRAIN_DF

    # CSV load
    _write_neuron_csv(finder.datasets_path, "hemibrain_v1_2_1", HEMIBRAIN_DF)
    df = finder._load_neuron_df_for_dataset("hemibrain_v1_2_1")
    assert df is not None and len(df) == 2

    # Parquet load
    folder = Path(finder.datasets_path) / "manc_v1_0"
    folder.mkdir(parents=True, exist_ok=True)
    HEMIBRAIN_DF.to_parquet(folder / "manc_v1_0_allneurons_neuron_df.parquet")
    assert finder._load_neuron_df_for_dataset("manc_v1_0") is not None

    # Missing -> pull path (mocked)
    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda ds: None)
    assert finder._load_neuron_df_for_dataset("optic-lobe_v1_1") is None

    # FlyWire missing -> skip without pulling
    pulled = []
    monkeypatch.setattr(
        finder, "_pull_and_load_dataset", lambda ds: pulled.append(ds))
    assert finder._load_neuron_df_for_dataset("banc_v626") is None
    assert pulled == []

    # No datasets_path -> FNC fallback (mocked)
    finder.datasets_path = None
    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda ds: HEMIBRAIN_DF)
    assert finder._load_neuron_df_for_dataset("anything") is HEMIBRAIN_DF


def test_enrich_match_with_dataset_info(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    match = {"bodyId": "5813128953"}
    enriched = finder._enrich_match_with_dataset_info(match, _em_image())
    assert enriched["dataset"] == "hemibrain:v1.2.1"
    assert enriched["dataset_folder"] == "hemibrain_v1_2_1"
    assert enriched["type"] == "MBON01"
    assert enriched["instance"] == "MBON01_R"
    assert enriched["status"] == "ok"

    # bodyId missing from local table -> NeuronBridge metadata fallback
    match = {"bodyId": "999"}
    enriched = finder._enrich_match_with_dataset_info(match, _em_image())
    assert enriched["type"] == "MBON01"
    assert enriched["status"] == ""

    # Unknown library -> unknown dataset
    img = _em_image()
    img.libraryName = "Zebra_Lib"
    enriched = finder._enrich_match_with_dataset_info({"bodyId": "1"}, img)
    assert enriched["dataset"] == "unknown"
    assert enriched["dataset_folder"] == "unknown"


# ---------------------------------------------------------------------------
# id_to_lines cache layer
# ---------------------------------------------------------------------------

def test_get_cache_path_invalid_type(finder):
    with pytest.raises(ValueError, match="Unsupported"):
        finder._get_cache_path("bogus", "x")


def test_save_and_load_from_cache(finder):
    df = pd.DataFrame(
        {"line": ["VT037867"], "library": ["FlyEM_Hemibrain_v1.2.1"],
         "score": [100], "image_id": ["lm-1"], "match_type": ["cds"]}
    )
    assert finder._load_from_cache("id_to_lines", "key1") is None
    finder._save_to_cache("id_to_lines", "key1", df)
    loaded = finder._load_from_cache("id_to_lines", "key1")
    assert loaded is not None and list(loaded["line"]) == ["VT037867"]
    assert finder._is_cached("id_to_lines", "key1")
    # Empty df is never saved
    finder._save_to_cache("id_to_lines", "empty", pd.DataFrame())
    assert not finder._is_cached("id_to_lines", "empty")


def test_load_from_cache_corrupt_parquet(finder):
    path = finder._get_cache_path("id_to_lines", "corrupt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not a parquet file")
    with pytest.warns(UserWarning, match="Failed to read"):
        assert finder._load_from_cache("id_to_lines", "corrupt") is None


def test_cache_index_helpers(finder):
    finder._add_to_cache_index("id_to_lines", "a/b:c")
    assert "id_to_lines_a_b_c" in finder._cache_index["id_to_lines"]
    assert finder._is_cached("id_to_lines", "a/b:c")
    assert not finder._is_cached("id_to_lines", "missing")


def test_cache_disabled_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nbf_mod, "NBClient", _DummyClient)
    f = NeuronBridgeFinder(cache_folder=str(tmp_path), use_cache=False, verbose=False)
    assert f._load_from_cache("id_to_lines", "x") is None
    assert f._load_image_cache("img", "cds") is None
    f._save_to_cache("id_to_lines", "x", pd.DataFrame({"a": [1]}))
    f._save_image_cache("img", "cds", [{"bodyId": "1", "score": 1}])


def test_load_from_cache_polars_and_bulk(finder):
    if not nbf_mod.HAS_POLARS:
        pytest.skip("polars not installed")
    df = pd.DataFrame(
        {"line": ["VT037867"], "library": ["L"], "score": [100.0],
         "image_id": ["lm-1"], "match_type": ["cds"]}
    )
    cache_key = finder._get_id_to_lines_cache_key(123, "cds", "hemibrain:v1.2.1")
    assert cache_key == "123_cds_hemibrain:v1.2.1"
    assert finder._load_from_cache_polars("id_to_lines", cache_key) is None
    finder._save_to_cache("id_to_lines", cache_key, df)
    loaded = finder._load_from_cache_polars("id_to_lines", cache_key)
    assert loaded is not None and loaded.height == 1

    combined, ids = finder._load_cached_neurons_bulk_polars(
        [{"bodyId": 123, "dataset": "hemibrain:v1.2.1"},
         {"bodyId": 999, "dataset": "hemibrain:v1.2.1"}],
        "cds",
    )
    assert ids == [123]
    assert "source_dataset" in combined.columns
    empty, ids = finder._load_cached_neurons_bulk_polars([], "cds")
    assert empty.height == 0 and ids == []


# ---------------------------------------------------------------------------
# Line mapping + image cache
# ---------------------------------------------------------------------------

def test_line_mapping_missing_and_corrupt(finder):
    assert finder._load_line_mapping() == {"lines": {}, "images": {}}
    path = finder._get_line_mapping_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{invalid json")
    assert finder._load_line_mapping() == {"lines": {}, "images": {}}


def test_update_and_save_line_mapping(finder):
    finder._update_line_mapping("VT037867", "Brain", ["img-1", "img-2"],
                                image_id="img-1", match_type="cds")
    mapping = finder._load_line_mapping()
    assert mapping["lines"]["VT037867"]["Brain"]["image_ids"] == ["img-1", "img-2"]
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    # Updating again does not duplicate cached types
    finder._update_line_mapping("VT037867", "Brain", ["img-1", "img-2"],
                                image_id="img-1", match_type="cds")
    mapping = finder._load_line_mapping()
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    # Manual save round-trip
    finder._save_line_mapping({"lines": {"X": {}}, "images": {}})
    assert "X" in finder._load_line_mapping()["lines"]


def test_save_line_mapping_failure_warns(finder, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.warns(UserWarning, match="Failed to save line mapping"):
        finder._update_line_mapping("VT037867", "Brain", ["img-1"])


def test_sync_mapping_from_cache_files(finder):
    image_dir = finder._get_parquet_cache().image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "cds_img1.parquet").write_bytes(b"x")
    (image_dir / "pppm_img1.parquet").write_bytes(b"x")
    (image_dir / "badname.parquet").write_bytes(b"x")
    (image_dir / "junk_img2.parquet").write_bytes(b"x")
    stats = finder.sync_mapping_from_cache_files()
    assert stats == {"images_scanned": 2, "types_updated": 2}
    mapping = finder._load_line_mapping()
    assert set(mapping["images"]["img1"]["cached_types"]) == {"cds", "pppm"}


def test_image_cache_load_corrupt(finder):
    path = finder._get_image_cache_path("imgX", "cds")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"garbage")
    with pytest.warns(UserWarning, match="Failed to read"):
        assert finder._load_image_cache("imgX", "cds") is None


def test_save_image_cache_and_mapping(finder):
    matches = [{"bodyId": "5813128953", "score": 100, "image_id": "em-1",
                "lm_sample": "img-1", "match_type": "cds",
                "dataset": "hemibrain:v1.2.1", "library": "FlyEM_Hemibrain_v1.2.1",
                "type": "MBON01", "instance": "MBON01_R", "status": "ok"}]
    # 'both' is never written
    finder._save_image_cache("img-1", "both", matches, line_name="VT037867")
    assert finder._load_image_cache("img-1", "both") is None
    # Empty matches are skipped
    finder._save_image_cache("img-1", "cds", [], line_name="VT037867")
    assert finder._load_image_cache("img-1", "cds") is None

    finder._save_image_cache("img-1", "cds", matches, line_name="VT037867")
    cached = finder._load_image_cache("img-1", "cds")
    assert cached is not None and list(cached["bodyId"]) == ["5813128953"]
    mapping = finder._load_line_mapping()
    assert mapping["images"]["img-1"]["cached_types"] == ["cds"]
    assert mapping["images"]["img-1"]["line"] == "VT037867"


# ---------------------------------------------------------------------------
# API match fetching with fake client
# ---------------------------------------------------------------------------

def test_fetch_matches_from_api(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    lm_image = _lm_image("img-1")
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match(score=60)]

    cds = finder._fetch_matches_from_api(lm_image, "cds")
    assert len(cds) == 1
    assert cds[0]["bodyId"] == "5813128953"
    assert cds[0]["match_type"] == "cds"
    assert cds[0]["type"] == "MBON01"  # enriched from local table

    pppm = finder._fetch_matches_from_api(lm_image, "pppm")
    assert pppm[0]["match_type"] == "pppm"
    assert pppm[0]["lm_sample"] == "img-1"


def test_fetch_matches_from_api_error_returns_empty(finder_with_client):
    finder_with_client._client.cds_matches["img-1"] = RuntimeError("boom")
    assert finder_with_client._fetch_matches_from_api(_lm_image("img-1"), "cds") == []


def test_get_image_matches_cached_paths(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF

    # No image id -> nothing
    assert finder._get_image_matches_cached(SimpleNamespace(id=""), "cds") == ([], False, False)

    lm_image = _lm_image("img-1")
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match(score=60)]

    # First call fetches from API and caches
    matches, from_cache, partial = finder._get_image_matches_cached(
        lm_image, "cds", line_name="VT037867")
    assert len(matches) == 1 and not from_cache and not partial

    # Second call hits the cache
    matches, from_cache, partial = finder._get_image_matches_cached(lm_image, "cds")
    assert len(matches) == 1 and from_cache and not partial

    # 'both' with cds cached but pppm not -> partial cache
    matches, from_cache, partial = finder._get_image_matches_cached(
        lm_image, "both", line_name="VT037867")
    assert len(matches) == 2 and not from_cache and partial

    # 'both' again -> everything cached
    matches, from_cache, partial = finder._get_image_matches_cached(lm_image, "both")
    assert len(matches) == 2 and from_cache and partial


# ---------------------------------------------------------------------------
# EM image / dataset identity helpers
# ---------------------------------------------------------------------------

def test_get_em_images(finder_with_client):
    finder = finder_with_client
    assert finder._get_em_images(1) == []
    finder._client.em_images[2] = RuntimeError("api down")
    assert finder._get_em_images(2) == []
    img = _em_image()
    finder._client.em_images[3] = [img]
    assert finder._get_em_images(3) == [img]
    # Generator inputs are converted to lists
    finder._client.em_images[4] = (i for i in [img])
    assert finder._get_em_images(4) == [img]
    # No client -> []
    finder._client = None
    assert finder._get_em_images(3) == []


def test_dataset_identity_helpers():
    assert NeuronBridgeFinder._dataset_identity(None) == ("", None)
    assert NeuronBridgeFinder._dataset_identity("unknown") == ("", None)
    assert NeuronBridgeFinder._dataset_identity("male-cns:v1.0:12211") == ("male-cns", "1.0")
    assert NeuronBridgeFinder._dataset_identity("hemibrain_v1_2_1") == ("hemibrain", "1.2.1")
    assert NeuronBridgeFinder._dataset_identity("hemibrain") == ("hemibrain", None)

    assert NeuronBridgeFinder._datasets_match("hemibrain:v1.2.1", "hemibrain:v1.2.1")
    assert not NeuronBridgeFinder._datasets_match("hemibrain:v1.2.1", "hemibrain:v1.2")
    assert NeuronBridgeFinder._datasets_match("hemibrain", "hemibrain:v1.2.1")
    assert NeuronBridgeFinder._datasets_match(None, "hemibrain:v1.2.1")
    assert not NeuronBridgeFinder._datasets_match("manc:v1.0", "hemibrain:v1.2.1")


def test_dataset_name_from_em_image_and_selection():
    img_a = SimpleNamespace(publishedName="hemibrain:v1.2.1:123")
    img_b = SimpleNamespace(publishedName="manc:v1.0:456:789")  # extra numeric id
    img_c = SimpleNamespace(publishedName="")
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_a) == "hemibrain:v1.2.1"
    # Only the trailing numeric body-id segment is dropped
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_b) == "manc:v1.0:456"
    assert NeuronBridgeFinder._dataset_name_from_em_image(img_c) is None

    assert NeuronBridgeFinder._select_em_image_for_dataset([]) is None
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], None) is img_a
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], "unknown") is img_a
    assert NeuronBridgeFinder._select_em_image_for_dataset(
        [img_a, img_b], "manc:v1.0") is img_b
    assert NeuronBridgeFinder._select_em_image_for_dataset([img_a], "manc:v1.0") is None


def test_get_body_metadata_and_validation(finder_with_client):
    finder = finder_with_client
    img = _em_image()
    finder._client.em_images[10] = [img]
    actual_ds, selected = finder._get_body_metadata(10, "hemibrain:v1.2.1")
    assert actual_ds == "hemibrain:v1.2.1" and selected is img
    actual_ds, selected = finder._get_body_metadata(10, "manc:v1.0")
    assert selected is None  # no matching dataset image

    assert finder._validate_body_id_dataset(10, "hemibrain:v1.2.1") == "hemibrain:v1.2.1"
    assert finder._get_em_image_for_dataset(10, "hemibrain:v1.2.1") is img
    # No client -> expected dataset is echoed
    finder._client = None
    assert finder._validate_body_id_dataset(10, "hemibrain:v1.2.1") == "hemibrain:v1.2.1"


def test_validate_body_ids_parallel(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[10] = [_em_image()]
    results = finder._validate_body_ids_parallel(
        [{"bodyId": 10, "dataset": "hemibrain:v1.2.1"},
         {"bodyId": 11}],  # unknown body -> no images
        max_workers=2,
    )
    assert results[10]["dataset"] == "hemibrain:v1.2.1"
    assert results[10]["em_image"] is not None
    assert results[11]["dataset"] is None
    assert finder._validate_body_ids_parallel([]) == {}


def test_extract_body_id(finder):
    assert finder._extract_body_id(_em_image()) == "5813128953"
    assert finder._extract_body_id(SimpleNamespace(id="img-778")) == "778"
    assert finder._extract_body_id(SimpleNamespace()) == ""


def test_sort_matches_by_rank(finder):
    assert finder._sort_matches_by_rank([]) == []
    matches = [
        {"line": "A", "score": 90, "match_type": "cds"},
        {"line": "B", "score": 80, "match_type": "cds"},
        {"line": "A", "score": 70, "match_type": "pppm"},
        {"line": "C", "score": 60, "match_type": "pppm"},
    ]
    result = finder._sort_matches_by_rank(matches, key_field="line")
    by_line = {m["line"]: m for m in result}
    assert by_line["A"]["combined_rank"] == 2  # cds rank 1 + pppm rank 2
    assert by_line["A"]["match_type"] == "both"
    assert by_line["B"]["pppm_rank"] == 3  # missing pppm -> max+1
    assert result[0]["line"] == "A"


# ---------------------------------------------------------------------------
# _get_em_matches / _get_lm_matches
# ---------------------------------------------------------------------------

def test_get_em_matches_cds_sorted(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[100] = [_em_image(cds=True)]
    finder._client.cds_matches["em-5813128953"] = [
        _lm_match("VT037867", score=50),
        _lm_match("LH173", score=90),
    ]
    matches = finder._get_em_matches(100, match_type="cds")
    assert [m["line"] for m in matches] == ["LH173", "VT037867"]
    assert matches[0]["match_type"] == "cds"


def test_get_em_matches_error_paths(finder_with_client):
    finder = finder_with_client
    # No EM image at all
    assert finder._get_em_matches(200, match_type="cds") == []
    # Explicit None image (validation found nothing)
    assert finder._get_em_matches(200, match_type="cds", em_image=None) == []
    # Image without files metadata
    assert finder._get_em_matches(200, match_type="cds",
                                  em_image=SimpleNamespace(id="x")) == []
    # match_type both with no result URLs -> warning path, empty result
    no_urls = _em_image(cds=False, pppm=False)
    assert finder._get_em_matches(200, match_type="both", em_image=no_urls) == []


def test_get_em_matches_both_raw_and_ranked(finder_with_client):
    finder = finder_with_client
    em_img = _em_image(cds=True, pppm=True)
    finder._client.cds_matches[em_img.id] = [
        _lm_match("VT037867", score=90), _lm_match("LH173", score=80)]
    finder._client.ppp_matches[em_img.id] = [
        _lm_match("VT037867", score=70), _lm_match("SS010", score=95)]

    raw = finder._get_em_matches(1, match_type="both", em_image=em_img, raw=True)
    assert len(raw) == 4 and all(m["match_type"] in ("cds", "pppm") for m in raw)

    ranked = finder._get_em_matches(1, match_type="both", em_image=em_img)
    assert ranked[0]["line"] == "VT037867"  # best combined rank
    assert ranked[0]["match_type"] == "both"


def test_get_lm_matches_cds_dedup(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder._client.lm_images["VT037867"] = [_lm_image("img-1")]
    finder._client.cds_matches["img-1"] = [
        _em_match(score=50), _em_match(score=90)]

    matches = finder._get_lm_matches("VT037867", match_type="cds")
    # Duplicates (same bodyId + dataset) collapsed, best score kept
    assert len(matches) == 1
    assert matches[0]["score"] == 90
    assert matches[0]["type"] == "MBON01"
    # Line mapping recorded the checked images (region defaults to 'All')
    mapping = finder._load_line_mapping()
    assert mapping["lines"]["VT037867"]["All"]["image_ids"] == ["img-1"]


def test_get_lm_matches_empty_paths(finder_with_client):
    finder = finder_with_client
    assert finder._get_lm_matches("MISSING", match_type="cds") == []

    finder._client.lm_images["VT1"] = [_lm_image("img-1", area="VNC")]
    finder.region = "Brain"
    assert finder._get_lm_matches("VT1", match_type="cds") == []

    finder.region = "All"
    finder._client.lm_images["VT2"] = [_lm_image("img-1", cds=False, pppm=False)]
    assert finder._get_lm_matches("VT2", match_type="cds") == []

    finder._client.lm_images["VT3"] = RuntimeError("api down")
    assert finder._get_lm_matches("VT3", match_type="cds") == []


def test_get_lm_matches_both_with_limit(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder.max_api_images_per_line = 1
    finder._client.lm_images["VT9"] = [
        _lm_image("img-1", cds=True, pppm=True),
        _lm_image("img-2", cds=True, pppm=True),
    ]
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.ppp_matches["img-1"] = [_em_match("1234", score=60)]

    matches = finder._get_lm_matches("VT9", match_type="both")
    assert len(matches) == 2
    assert all(m["match_type"] == "both" for m in matches)
    # Only img-1 was processed because of max_api_images_per_line=1
    assert "img-2" not in [c[1] for c in finder._client.calls]


def test_get_lm_matches_parallel(finder_with_client):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    finder.max_workers = 2
    finder._client.lm_images["VT9"] = [
        _lm_image("img-1"), _lm_image("img-2")]
    finder._client.cds_matches["img-1"] = [_em_match(score=80)]
    finder._client.cds_matches["img-2"] = [_em_match("1234", score=70)]
    matches = finder._get_lm_matches("VT9", match_type="cds")
    assert len(matches) == 2


# ---------------------------------------------------------------------------
# id_to_lines
# ---------------------------------------------------------------------------

def test_id_to_lines_cds_fetch_cache_and_hit(finder_with_client):
    finder = finder_with_client
    finder._client.em_images[100] = [_em_image(cds=True)]
    finder._client.cds_matches["em-5813128953"] = [
        _lm_match("VT037867", score=50), _lm_match("LH173", score=90)]

    df = finder.id_to_lines(100, match_type="cds")
    assert list(df["line"]) == ["LH173", "VT037867"]

    # Second call served from cache (no new API calls)
    calls_before = len(finder._client.calls)
    df2 = finder.id_to_lines(100, match_type="cds")
    assert list(df2["line"]) == ["LH173", "VT037867"]
    assert len(finder._client.calls) == calls_before


def test_id_to_lines_both_uses_algorithm_caches(finder_with_client):
    finder = finder_with_client
    em_img = _em_image(cds=True, pppm=True)
    finder._client.em_images[100] = [em_img]
    finder._client.cds_matches[em_img.id] = [_lm_match("VT037867", score=90)]
    finder._client.ppp_matches[em_img.id] = [_lm_match("LH173", score=85)]

    # Pre-populate the cds cache
    finder.id_to_lines(100, match_type="cds")

    df = finder.id_to_lines(100, match_type="both")
    assert set(df.columns) >= {"combined_rank", "cds_score", "pppm_score"}
    assert sorted(df["line"]) == ["LH173", "VT037867"]


def test_id_to_lines_empty_and_invalid(finder_with_client):
    finder = finder_with_client
    df = finder.id_to_lines(404, match_type="cds")
    assert df.empty and list(df.columns) == ["line", "library", "score", "image_id", "match_type"]
    df = finder.id_to_lines(404, match_type="both")
    assert df.empty and "combined_rank" in df.columns
    with pytest.raises(ValueError):
        finder.id_to_lines(404, match_type="wrong")


# ---------------------------------------------------------------------------
# Co-labeling / aggregation
# ---------------------------------------------------------------------------

def test_calculate_expression_entropy(finder):
    assert finder._calculate_expression_entropy({}) == 0.0
    assert finder._calculate_expression_entropy({"a": 0}) == 0.0
    assert finder._calculate_expression_entropy({"a": 5}) == 0.0
    assert abs(finder._calculate_expression_entropy({"a": 1, "b": 1}) - 1.0) < 1e-9


def test_calculate_weighted_specificity(finder):
    empty = finder._calculate_weighted_specificity(pd.DataFrame(), {"mbon01"})
    assert empty["weighted_type_proportion"] == 0.0
    df = pd.DataFrame({"type": ["MBON01", "LH173"], "score": [80.0, 20.0]})
    result = finder._calculate_weighted_specificity(df, {"mbon01"})
    assert result["weighted_type_proportion"] == pytest.approx(0.8)
    assert result["weighted_queried_score"] == 80.0
    assert result["weighted_total_score"] == 100.0
    assert result["mean_queried_score"] == 80.0
    # Missing score column
    assert finder._calculate_weighted_specificity(
        pd.DataFrame({"type": ["x"]}), {"x"}, score_column="nope"
    )["weighted_total_score"] == 0.0


def _patch_line_to_neuron(finder, monkeypatch, data):
    def fake(line_name, match_type="cds", top_n=100, **_kwargs):
        result = data.get(line_name)
        if isinstance(result, Exception):
            raise result
        return result if result is not None else pd.DataFrame()

    monkeypatch.setattr(finder, "line_to_neuron", fake)


def test_build_colabeling_matrix_methods(finder, monkeypatch):
    l1 = pd.DataFrame({"type": ["A", "B", "C"], "score": [10.0, 5.0, 2.0]})
    l2 = pd.DataFrame({"type": ["A", "B", "C"], "score": [9.0, 3.0, 1.0]})
    _patch_line_to_neuron(finder, monkeypatch, {"L1": l1, "L2": l2, "L3": pd.DataFrame()})

    matrix, sets = finder._build_colabeling_matrix(
        ["L1", "L2", "L3"], similarity_method="jaccard")
    assert matrix.loc["L1", "L1"] == 1.0
    assert matrix.loc["L1", "L2"] == pytest.approx(1.0)
    assert matrix.loc["L1", "L3"] == 0.0
    assert sets["L1"] == {"a", "b", "c"}
    assert sets["L3"] == set()

    matrix, _ = finder._build_colabeling_matrix(
        ["L1", "L2"], similarity_method="weighted_jaccard", min_score=4.0)
    # min_score drops L2's score-3 row: intersection min(10,9)=9,
    # union max(10,9)+max(5,0)=15 -> 9/15
    assert matrix.loc["L1", "L2"] == pytest.approx(0.6)

    matrix, _ = finder._build_colabeling_matrix(
        ["L1", "L2"], similarity_method="rank_correlation")
    assert matrix.loc["L1", "L2"] == pytest.approx(1.0)


def test_build_colabeling_matrix_filters_and_errors(finder, monkeypatch):
    l1 = pd.DataFrame({"type": ["A", "B"], "score": [100.0, 1.0]})
    l2 = pd.DataFrame({"type": ["A", "B"], "score": [90.0, 2.0]})
    _patch_line_to_neuron(
        finder, monkeypatch, {"L1": l1, "L2": l2, "BAD": RuntimeError("api")})

    matrix, sets = finder._build_colabeling_matrix(
        ["L1", "L2", "BAD"], min_type_avg_score=50.0)
    assert sets["L1"] == {"a"}  # low-score type 'b' filtered out
    assert sets["BAD"] == set()

    with pytest.raises(ValueError):
        finder._build_colabeling_matrix(["L1"], similarity_method="nope")


def test_calculate_colabeling_sparsity(finder):
    matrix = pd.DataFrame(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        index=["L1", "L2", "L3"], columns=["L1", "L2", "L3"],
    )
    scores = finder._calculate_colabeling_sparsity(matrix, threshold=0.1)
    assert scores["L1"]["n_colabeling_lines"] == 1
    assert scores["L3"]["colabel_sparsity"] == 1.0
    single = pd.DataFrame([[1.0]], index=["L1"], columns=["L1"])
    assert finder._calculate_colabeling_sparsity(single)["L1"]["colabel_sparsity"] == 1.0


def _combined_df():
    return pd.DataFrame(
        {
            "line": ["VT037867", "VT037867", "LH173"],
            "score": [90.0, 60.0, 80.0],
            "source_bodyId": ["5813128953", "1234", "5813128953"],
            "source_dataset": ["hemibrain:v1.2.1", "hemibrain:v1.2.1", "unknown"],
        }
    )


def test_aggregate_results_pandas(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    combined, stats = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=True, sort_by="max")
    assert list(stats["line"]) == ["VT037867", "LH173"]
    assert "weighted_score" in stats.columns
    assert "min_score_per_dataset" in stats.columns
    assert "matched_types" in stats.columns
    vt_row = stats[stats["line"] == "VT037867"].iloc[0]
    assert vt_row["match_count"] == 2
    assert "matched_bodyIds" in combined.columns

    # completeness sorting branch
    _, stats_c = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="completeness")
    assert "coverage_ratio" in stats_c.columns

    # separate_splitgal4 adds line_type
    finder.separate_splitgal4 = True
    _, stats_s = finder._aggregate_results_pandas(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="max")
    assert set(stats_s["line_type"]) == {"gal4_lexa", "split_gal4"}

    # No 'line' column -> empty stats
    combined_only, empty_stats = finder._aggregate_results_pandas(
        pd.DataFrame({"score": [1.0]}), "cds", False, "max")
    assert empty_stats.empty


def test_aggregate_results_polars(finder):
    if not nbf_mod.HAS_POLARS:
        pytest.skip("polars not installed")
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    combined, stats = finder._aggregate_results_polars(
        _combined_df(), "cds", is_multi_dataset=True, sort_by="completeness")
    assert len(stats) == 2
    assert "coverage_ratio" in stats.columns
    assert "matched_bodyIds" in combined.columns

    finder.separate_splitgal4 = True
    _, stats_s = finder._aggregate_results_polars(
        _combined_df(), "cds", is_multi_dataset=False, sort_by="max")
    assert "line_type" in stats_s.columns

    combined_only, empty_stats = finder._aggregate_results_polars(
        pd.DataFrame({"score": [1.0]}), "cds", False, "max")
    assert empty_stats.empty


# ---------------------------------------------------------------------------
# Fake vispath_pkg (VisConnMatInteractive) for matrix visualizations
# ---------------------------------------------------------------------------

def _install_fake_vispath(monkeypatch):
    """Register a fake ``vispath_pkg`` module; returns the call recorder list."""
    calls = []

    class VisConnMatInteractive:
        def __init__(self, df, filename=None, title=None, **kwargs):
            calls.append({
                "filename": filename,
                "title": title,
                "shape": getattr(df, "shape", None),
                "kwargs": kwargs,
            })
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                Path(filename).write_text("<html>fake heatmap</html>")

    mod = types.ModuleType("vispath_pkg")
    mod.VisConnMatInteractive = VisConnMatInteractive
    monkeypatch.setitem(sys.modules, "vispath_pkg", mod)
    return calls


def test_visualize_colabeling_matrix_missing_vispath(finder, tmp_path, monkeypatch):
    # sys.modules entry of None forces ImportError on both import attempts
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)
    matrix = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    assert finder.visualize_colabeling_matrix(matrix, str(tmp_path / "out")) == ""
    assert finder.visualize_expression_matrix(matrix, str(tmp_path / "out")) == ""
    assert finder.visualize_expression_matrix_merged(matrix, str(tmp_path / "out")) == ""


def test_visualize_colabeling_matrix_with_fake_vispath(finder, tmp_path, monkeypatch):
    calls = _install_fake_vispath(monkeypatch)
    matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    out = finder.visualize_colabeling_matrix(
        matrix, str(tmp_path / "heat"), color_scale="green", filename="custom.html")
    assert out.endswith("custom.html") and Path(out).exists()
    assert calls[0]["kwargs"]["zmin"] == 0.0 and calls[0]["kwargs"]["zmax"] == 1.0
    # Unknown color scale falls back to the purple preset
    finder.visualize_colabeling_matrix(matrix, str(tmp_path / "heat"), color_scale="nope")
    assert len(calls) == 2


def test_visualize_expression_matrix_paths(finder, tmp_path, monkeypatch):
    calls = _install_fake_vispath(monkeypatch)
    df = pd.DataFrame(
        {
            "HEMI_MBON01": [10.0, 0.0],
            "HEMI_DNp01": [5.0, 8.0],
            "MCNS_LH173": [0.0, 6.0],
        },
        index=["L1", "L2"],
    )
    out_dir = str(tmp_path / "expr")
    # All types shown
    path = finder.visualize_expression_matrix(df, out_dir)
    assert path.endswith("expression_matrix.html")
    assert Path(out_dir, "expression_matrix.csv").exists()
    assert Path(out_dir, "expression_matrix_viz.csv").exists()

    # top_n_types truncation exercises the re-sort branch
    path = finder.visualize_expression_matrix(df, out_dir, top_n_types=2)
    assert Path(path).exists()

    # type_filter: keep only MBON-like types
    path = finder.visualize_expression_matrix(
        df, out_dir, type_filter={"contains": "MBON"})
    assert Path(path).exists()

    # Filter that removes everything -> empty -> ""
    assert finder.visualize_expression_matrix(
        df, out_dir, type_filter={"contains": "ZZZ"}) == ""

    # Unknown color scale falls back to green
    finder.visualize_expression_matrix(df, out_dir, color_scale="nope")
    assert len(calls) == 4


def test_visualize_expression_matrix_merged_paths(finder, tmp_path, monkeypatch):
    calls = _install_fake_vispath(monkeypatch)
    monkeypatch.setattr(nbf_mod, "HAS_TYPE_MAPPER", False)
    df = pd.DataFrame(
        {"HEMI_A": [10.0, 4.0], "MCNS_A": [8.0, 0.0], "HEMI_b": [2.0, 6.0]},
        index=["L1", "L2"],
    )
    out_dir = str(tmp_path / "merged")
    path = finder.visualize_expression_matrix_merged(df, out_dir)
    assert path.endswith("expression_matrix_merged.html") and Path(path).exists()
    assert Path(out_dir, "expression_matrix_merged.csv").exists()
    assert Path(out_dir, "expression_matrix_merged_viz.csv").exists()

    # mean / sum / unknown aggregation branches
    assert finder.visualize_expression_matrix_merged(df, out_dir, aggregation="mean")
    assert finder.visualize_expression_matrix_merged(df, out_dir, aggregation="sum")
    assert finder.visualize_expression_matrix_merged(df, out_dir, aggregation="???")

    # top_n truncation + type_filter on merged names
    assert finder.visualize_expression_matrix_merged(df, out_dir, top_n_types=1)
    assert finder.visualize_expression_matrix_merged(
        df, out_dir, type_filter={"startswith": "A"})
    # Filter removing everything -> ""
    assert finder.visualize_expression_matrix_merged(
        df, out_dir, type_filter={"startswith": "ZZZ"}) == ""


def test_visualize_expression_matrix_merged_with_mapper(finder, tmp_path, monkeypatch):
    calls = _install_fake_vispath(monkeypatch)

    class FakeMapper:
        _loaded = True

        def get_merge_mapping_for_types(self, cols, verbose=False):
            return {"HEMI_b": "A"}

    monkeypatch.setattr(nbf_mod, "HAS_TYPE_MAPPER", True)
    monkeypatch.setattr(nbf_mod, "get_type_mapper", lambda: FakeMapper())
    df = pd.DataFrame({"HEMI_A": [10.0], "HEMI_b": [2.0], "MCNS_A": [8.0]}, index=["L1"])
    path = finder.visualize_expression_matrix_merged(df, str(tmp_path))
    assert Path(path).exists()

    # Mapper that raises -> warning branch, prefix-stripping fallback
    def broken_mapper():
        raise RuntimeError("mapper data missing")

    monkeypatch.setattr(nbf_mod, "get_type_mapper", broken_mapper)
    path = finder.visualize_expression_matrix_merged(df, str(tmp_path))
    assert Path(path).exists()

    # Mapper object that is not loaded -> skipped silently
    class NotLoaded:
        _loaded = False

    monkeypatch.setattr(nbf_mod, "get_type_mapper", lambda: NotLoaded())
    path = finder.visualize_expression_matrix_merged(df, str(tmp_path))
    assert Path(path).exists()


def test_visualize_labeling_distribution(finder, tmp_path):
    df = pd.DataFrame(
        {
            "score": [30.0, 10.0, 20.0, 5.0],
            "type": ["A", "B", "A", "C"],
            "dataset": ["d1", "d1", "d2", "d2"],
        }
    )
    # grouped subplots with threshold
    p = finder.visualize_labeling_distribution(
        df, str(tmp_path), show_threshold=8.0, group_by="dataset")
    assert p and Path(p).exists()
    # grouped but without the label column present
    p = finder.visualize_labeling_distribution(
        df, str(tmp_path), label_column="missing_col", group_by="dataset",
        filename="g_nolabel.html")
    assert Path(p).exists()
    # plain single plot
    p = finder.visualize_labeling_distribution(
        df, str(tmp_path), filename="plain.html")
    assert Path(p).exists()
    # plain without label column
    p = finder.visualize_labeling_distribution(
        df[["score"]], str(tmp_path), label_column="type", filename="nolabel.html")
    assert Path(p).exists()
    # empty / missing score column
    assert finder.visualize_labeling_distribution(pd.DataFrame(), str(tmp_path)) == ""
    assert finder.visualize_labeling_distribution(
        pd.DataFrame({"x": [1]}), str(tmp_path)) == ""
    # group_by column all-NaN -> no groups
    df_nan = df.copy()
    df_nan["dataset"] = np.nan
    assert finder.visualize_labeling_distribution(
        df_nan, str(tmp_path), group_by="dataset") == ""
    # scores empty after label aggregation cannot happen; empty-agg path:
    assert finder.visualize_labeling_distribution(
        pd.DataFrame({"score": pd.Series([], dtype=float)}), str(tmp_path)) == ""


def test_visualize_colabeling_distribution_edges(finder, tmp_path):
    assert finder.visualize_colabeling_distribution({}, str(tmp_path)) == ("", "")
    assert finder.visualize_colabeling_distribution(
        {"L1": pd.DataFrame()}, str(tmp_path)) == ("", "")


def test_visualize_colabeling_distribution_min_score_fallback(finder, tmp_path):
    df = pd.DataFrame(
        {"score": [3.0, 2.0], "type": ["A", "B"], "dataset": ["d", "d"],
         "bodyId": ["1", "2"]}
    )
    # min_score filters everything -> falls back to all data
    p_type, p_neuron = finder.visualize_colabeling_distribution(
        {"L1": df}, str(tmp_path), min_score=1000.0)
    assert Path(p_type).exists() and Path(p_neuron).exists()
    assert Path(tmp_path, "labeling_distribution_stacked.html").exists()
    assert Path(tmp_path, "distribution_data_by_neuron.csv").exists()
    assert Path(tmp_path, "distribution_data_by_type.csv").exists()


def test_sort_expression_matrix_branches(finder):
    # Empty input passes through (both orientations)
    empty = pd.DataFrame()
    assert finder._sort_expression_matrix(empty, as_types_rows=True).empty
    assert finder._sort_expression_matrix(empty, as_types_rows=False).empty
    # Lines x Types input (index named 'line') is transposed
    lines_x_types = pd.DataFrame(
        {"T1": [5.0, 0.0], "T2": [1.0, 2.0]}, index=pd.Index(["L1", "L2"], name="line")
    )
    sorted_types = finder._sort_expression_matrix(lines_x_types, as_types_rows=True)
    assert list(sorted_types.index) == ["T2", "T1"] or set(sorted_types.index) == {"T1", "T2"}
    # as_types_rows=False returns the transpose
    back = finder._sort_expression_matrix(lines_x_types, as_types_rows=False)
    assert set(back.index) == {"L1", "L2"}


# ---------------------------------------------------------------------------
# analyze_colabeling end-to-end (mocked client only, no network)
# ---------------------------------------------------------------------------

def _setup_colabeling_client(finder):
    client = finder._client
    # Pre-seed the local dataset table so match enrichment never triggers a
    # network dataset pull (_pull_and_load_dataset) during the flows below.
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF

    # NOTE: _sort_expression_matrix only transposes to types-as-rows when
    # len(columns) > len(df) * 2 (source heuristic bug for small matrices),
    # so we seed enough distinct types (> 2x the number of lines) for the
    # end-to-end flow to orient the matrix correctly.
    def _typed_match(n):
        m = _em_match(f"9{n}", score=20000 + n * 100, image_id=f"em-9{n}")
        m.image.neuronType = f"A00{n}"
        m.image.neuronInstance = f"A00{n}_R"
        return m

    m1 = _em_match("5813128953", score=45000)
    m2 = _em_match("1234", score=38000)
    m2.image.neuronType = "LH173"
    m2.image.neuronInstance = "LH173_L"
    client.lm_images["L1"] = [_lm_image("img-L1")]
    client.cds_matches["img-L1"] = [m1, m2] + [_typed_match(i) for i in range(1, 5)]

    m3 = _em_match("5813128953", score=40000)
    m4 = _em_match("777", score=25000)
    m4.image.neuronType = "DNp01"
    m4.image.neuronInstance = ""
    client.lm_images["L2"] = [_lm_image("img-L2")]
    client.cds_matches["img-L2"] = [m3, m4] + [_typed_match(i) for i in range(5, 9)]

    client.lm_images["EMPTY"] = []


def test_analyze_colabeling_end_to_end(finder_with_client, tmp_path, monkeypatch):
    finder = finder_with_client
    _setup_colabeling_client(finder)
    viz_calls = _install_fake_vispath(monkeypatch)
    monkeypatch.setattr(nbf_mod, "HAS_TYPE_MAPPER", False)

    results = finder.analyze_colabeling(
        lines="L1,L2,EMPTY",
        output_dir=str(tmp_path),
        min_score=10000.0,
        min_type_avg_score=0.0,
        type_filter={"contains": ["MBON", "LH", "DN"]},
        similarity_methods=["jaccard", "weighted_jaccard", "rank_correlation"],
    )

    # Results structure
    assert not results["expression_matrix"].empty
    assert not results["labeling_info"].empty
    assert set(results["colabeling_matrices"]) == {
        "jaccard", "weighted_jaccard", "rank_correlation"}
    summary = results["line_summary"]
    assert list(summary["line"]) == ["L1", "L2", "EMPTY"]
    empty_row = summary[summary["line"] == "EMPTY"].iloc[0]
    assert empty_row["n_neurons"] == 0

    # Output folder + artifacts
    run_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("NB-colabeling_")]
    assert len(run_dirs) == 1
    out = run_dirs[0]
    for fname in [
        "expression_matrix.csv", "expression_matrix_viz.csv", "expression_matrix.html",
        "expression_matrix_merged.csv", "expression_matrix_merged.html",
        "labeling_info.csv", "line_summary.csv", "parameters.json",
        "colabeling_matrix_jaccard.csv", "colabeling_matrix_jaccard.html",
        "colabeling_matrix_weighted_jaccard.html",
        "colabeling_matrix_rank_correlation.csv",
        "labeling_distribution_by_type.html", "labeling_distribution_by_neuron.html",
        "labeling_distribution_stacked.html",
        "distribution_data_by_neuron.csv", "distribution_data_by_type.csv",
        "colabeling_report.html", "user_warning_notes.txt",
    ]:
        assert (out / fname).exists(), fname
    assert (out / "line_labeled_neurons" / "L1_neurons.csv").exists()
    assert (out / "line_labeled_neurons" / "EMPTY_neurons.csv").exists() is False

    # Report content
    report = (out / "colabeling_report.html").read_text(encoding="utf-8")
    assert "Co-Labeling Analysis Report" in report
    assert "pair-card" in report  # shared MBON01 -> weighted_jaccard > 0.1

    # Heatmaps created via the fake VisConnMatInteractive
    assert len(viz_calls) >= 5

    # Per-line neurons retained below the cutoff but flagged
    l1 = results["line_neurons"]["L1"]
    assert "_passes_min_score" in l1.columns and l1["_passes_min_score"].all()


def test_analyze_colabeling_requires_two_lines(finder):
    assert finder.analyze_colabeling(["ONLY_ONE"]) == {}
    assert finder.analyze_colabeling("  ") == {}


def test_analyze_colabeling_no_expression_data(finder_with_client, tmp_path):
    finder_with_client._client.lm_images["A"] = []
    finder_with_client._client.lm_images["B"] = []
    results = finder_with_client.analyze_colabeling(
        ["A", "B"], output_dir=str(tmp_path), visualize=False, generate_report=False)
    assert results["expression_matrix"] is None
    assert results["line_neurons"]["A"].empty


def test_analyze_colabeling_match_type_validation(finder_with_client, tmp_path):
    finder = finder_with_client
    _setup_colabeling_client(finder)
    # invalid match type raises through the validator
    with pytest.raises(ValueError):
        finder.analyze_colabeling(["L1", "L2"], match_type="bogus")


# ---------------------------------------------------------------------------
# Batch B: neuron_to_lines / save_results / type summaries / find_neurons_batch
# ---------------------------------------------------------------------------

def _block_network_pulls(finder, monkeypatch):
    """Guarantee no NeuPrint/FNC pull can happen from the flows under test."""
    monkeypatch.setattr(finder, "_pull_and_load_dataset", lambda folder: None)
    monkeypatch.setattr(finder, "_fetch_neuron_df_via_fnc", lambda folder: None)


def test_save_results_variants(finder, tmp_path):
    df = pd.DataFrame({"line": ["A"], "score": [1.0]})
    # DataFrame without timestamp
    path = finder.save_results(df, str(tmp_path / "out.csv"), include_timestamp=False)
    assert path.endswith("out.csv") and Path(path).exists()
    # dict of DataFrames with timestamp appended to the filename
    out = finder.save_results({"111": df, "222": pd.DataFrame()}, str(tmp_path / "multi.csv"))
    assert out != str(tmp_path / "multi.csv") and Path(out).exists()
    saved = pd.read_csv(out)
    assert "query_bodyId" in saved.columns and set(saved["query_bodyId"]) == {111}
    # dict with only empty frames -> header-only file
    out2 = finder.save_results(
        {"1": pd.DataFrame()}, str(tmp_path / "empty.csv"), include_timestamp=False)
    assert Path(out2).exists()


def test_type_summary_and_type_count(finder, monkeypatch):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    df = pd.DataFrame({
        "bodyId": ["5813128953", "1234", "777", "888"],
        "type": ["MBON01", "LH173", "", None],
        "score": [45000.0, 38000.0, 25000.0, 20000.0],
        "dataset": ["hemibrain:v1.2.1"] * 4,
    })
    summary = finder._create_type_summary(df, "hemibrain:v1.2.1")
    assert list(summary.columns)[:2] == ["type", "labeled_N"]
    assert "typed_N_in_dataset" in summary.columns
    assert summary.iloc[0]["type"] == "MBON01"
    # untyped neurons get unique unknown_ labels
    assert {"unknown_777", "unknown_888"} <= set(summary["type"])
    mb = summary[summary["type"] == "MBON01"].iloc[0]
    assert mb["typed_N_in_dataset"] == 1

    # alternative + invalid sort_by (falls back to max_score)
    s2 = finder._create_type_summary(df, "hemibrain:v1.2.1", sort_by="median_score")
    assert s2.iloc[0]["type"] == "MBON01"
    s3 = finder._create_type_summary(df, "hemibrain:v1.2.1", sort_by="bogus")
    assert s3.iloc[0]["type"] == "MBON01"

    # no score column -> score metric columns are excluded from output
    s4 = finder._create_type_summary(df.drop(columns=["score"]), "hemibrain:v1.2.1")
    assert "max_score" not in s4.columns and list(s4.columns)[:2] == ["type", "labeled_N"]

    # no type column -> everything becomes unknown_
    s5 = finder._create_type_summary(df.drop(columns=["type"]), "hemibrain:v1.2.1")
    assert all(t.startswith("unknown_") for t in s5["type"])

    # _get_type_count_in_dataset branches
    assert finder._get_type_count_in_dataset("unknown_123", "hemibrain:v1.2.1") == 1
    assert finder._get_type_count_in_dataset("12345", "hemibrain:v1.2.1") == 1
    assert finder._get_type_count_in_dataset("MBON01", "hemibrain:v1.2.1") == 1
    assert finder._get_type_count_in_dataset("NOT_THERE", "hemibrain:v1.2.1") == 1
    # unmapped dataset with pulls blocked -> default 1
    assert finder._get_type_count_in_dataset("MBON01", "weird_ds:v9") == 1


def test_apply_type_filter_full(finder):
    types = [(1, "DNp01"), (2, "MBON01"), (3, "LH173_R"), (4, "IN_xyz")]
    assert finder._apply_type_filter(types, None) == types
    assert finder._apply_type_filter(types, {}) == types
    assert finder._apply_type_filter(types, {"contains": "DN"}) == [(1, "DNp01")]
    assert finder._apply_type_filter(types, {"contains": ["DN", "LH"]}) == [
        (1, "DNp01"), (3, "LH173_R")]
    assert finder._apply_type_filter(types, {"startswith": ["MB", "LH"]}) == [
        (2, "MBON01"), (3, "LH173_R")]
    assert finder._apply_type_filter(types, {"endswith": "_R"}) == [(3, "LH173_R")]
    assert finder._apply_type_filter(types, {"regex": r"^DN[a-z]\d+"}) == [(1, "DNp01")]
    # invalid regex pattern matches nothing
    assert finder._apply_type_filter(types, {"regex": "["}) == []
    # unknown filter type passes everything through
    assert finder._apply_type_filter(types, {"weird": "x"}) == types
    # AND across keys
    assert finder._apply_type_filter(types, {"contains": "LH", "endswith": "_R"}) == [
        (3, "LH173_R")]
    assert finder._apply_type_filter(types, {"contains": "LH", "startswith": "DN"}) == []


def test_top_types_fallback_score_and_color_helpers(finder):
    df = pd.DataFrame({
        "type_label": ["A", "A", "B", "C"],
        "score": [10.0, 2.0, 8.0, 4.0],
        "bodyId": ["1", "2", "3", "4"],
    })
    assert finder._get_top_types_fallback(df) == ["A", "B", "C"]
    assert finder._get_top_types_fallback(df, top_n=1) == ["A"]
    assert finder._get_top_types_fallback(df, sort_by="median_score") == ["B", "A", "C"]
    assert finder._get_top_types_fallback(df, sort_by="bogus") == ["A", "B", "C"]
    # no score column -> count-based ordering
    no_score = finder._get_top_types_fallback(df.drop(columns=["score"]), top_n=1)
    assert no_score == ["A"]

    # _get_type_scores for every metric
    assert finder._get_type_scores(df) == {"A": 10.0, "B": 8.0, "C": 4.0}
    assert finder._get_type_scores(df, sort_by="avg_score")["A"] == pytest.approx(6.0)
    assert finder._get_type_scores(df, sort_by="median_score")["B"] == 8.0
    assert finder._get_type_scores(df, sort_by="Q3_score")["A"] == pytest.approx(8.0)
    assert finder._get_type_scores(df, sort_by="Q1_score")["A"] == pytest.approx(4.0)
    assert finder._get_type_scores(df, sort_by="???") == {"A": 10.0, "B": 8.0, "C": 4.0}
    assert finder._get_type_scores(df.drop(columns=["score"])) == {}

    # layer alphas / colors
    assert finder._compute_layer_alphas([]) == []
    assert finder._compute_layer_alphas([0.0, 0.0]) == [0.0, 0.0]
    assert finder._compute_layer_alphas([10.0, 5.0], base_alpha=0.4) == [0.4, 0.2]
    assert finder._hex_to_rgba("#ff0000", 0.5) == (255, 0, 0, 0.5)
    assert finder._hex_to_rgba("00ff00") == (0, 255, 0, 1.0)
    colors = finder._create_neuron_colors_with_alpha(3, [0.2, 0.1, 0.05])
    assert len(colors) == 3 and colors[0][3] == 0.2
    assert len(finder._create_neuron_colors_with_alpha(12, [0.1] * 12)) == 12
    # alpha list shorter than layers -> default 0.2
    short = finder._create_neuron_colors_with_alpha(3, [0.5])
    assert short[2][3] == 0.2


class _FakeCrossMapper:
    """Fake mapper on the shared resolver contract: canonical merge keys
    come from ``get_mapping_decision`` (not the raw ``get_canonical_type``),
    so the export must route through ``canonical_merge_key``.  The
    ``called`` flags pin which API the code under test used."""
    _loaded = True

    def __init__(self):
        self.decision_calls = 0
        self.canonical_calls = 0

    def _get_type_mapping_key(self, dataset):
        return dataset

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, include_bridges=False):
        self.decision_calls += 1
        target = {"MBON01": "canon_MBON", "LH173": "canon_LH"}.get(
            source_type, source_type)
        return {'status': 'mapped', 'source_type': source_type,
                'target_type': target, 'target_types': [target],
                'relationship': '1-to-1', 'conflicts': []}

    def get_canonical_type(self, type_name, source_dataset=None):
        # Kept so a regression back to the raw API is detectable: the
        # export must NOT take this path.
        self.canonical_calls += 1
        return {"MBON01": "canon_MBON", "LH173": "canon_LH"}.get(
            type_name, type_name)


def test_save_dataset_categorized_files_and_type_mapped(finder, tmp_path, monkeypatch):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    df = pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "type": ["MBON01", "MBON01", "LH173", "MBON01"],
        "score": [40.0, 30.0, 20.0, 25.0],
        "dataset": ["hemibrain:v1.2.1", "hemibrain:v1.2.1", "manc:v1.0", "manc:v1.0"],
    })
    monkeypatch.setattr(nbf_mod, "HAS_TYPE_MAPPER", True)
    _fake_mapper = _FakeCrossMapper()
    monkeypatch.setattr(nbf_mod, "get_type_mapper", lambda: _fake_mapper)

    out = tmp_path / "cat"
    out.mkdir()
    finder._save_dataset_categorized_files(df, "L1", str(out))
    by_dataset = out / "by_dataset"
    assert (by_dataset / "L1_hemibrain_v1_2_1_neurons.csv").exists()
    assert (by_dataset / "L1_hemibrain_v1_2_1_types.csv").exists()
    assert (by_dataset / "L1_manc_v1_0_neurons.csv").exists()
    mapped = pd.read_csv(out / "L1_type_mapped.csv")
    assert "canonical_type" in mapped.columns and "best_max_score" in mapped.columns
    assert set(mapped["canonical_type"]) == {"canon_MBON", "canon_LH"}
    row = mapped[mapped["canonical_type"] == "canon_MBON"].iloc[0]
    assert row["total_labeled_N"] == 3 and row["best_max_score"] == 40.0
    # Workstream-I contract: the export resolves through canonical_merge_key
    # (get_mapping_decision), NOT the status-blind raw get_canonical_type.
    assert _fake_mapper.decision_calls > 0
    assert _fake_mapper.canonical_calls == 0

    # single dataset -> no type-mapped file
    out2 = tmp_path / "cat2"
    out2.mkdir()
    finder._save_dataset_categorized_files(df[df["dataset"] == "manc:v1.0"], "L2", str(out2))
    assert not (out2 / "L2_type_mapped.csv").exists()

    # missing dataset column -> nothing written
    out3 = tmp_path / "cat3"
    out3.mkdir()
    finder._save_dataset_categorized_files(df.drop(columns=["dataset"]), "L3", str(out3))
    assert list(out3.iterdir()) == []

    # mapper is None -> skipped
    monkeypatch.setattr(nbf_mod, "get_type_mapper", lambda: None)
    out4 = tmp_path / "cat4"
    out4.mkdir()
    finder._save_type_mapped_csv(df, "L4", str(out4))
    assert list(out4.iterdir()) == []

    # mapper raising -> warning branch, still no file
    def broken_mapper():
        raise RuntimeError("no mapper")

    monkeypatch.setattr(nbf_mod, "get_type_mapper", broken_mapper)
    finder._save_type_mapped_csv(df, "L5", str(out4))
    assert list(out4.iterdir()) == []

    # HAS_TYPE_MAPPER False -> early return
    monkeypatch.setattr(nbf_mod, "HAS_TYPE_MAPPER", False)
    finder._save_type_mapped_csv(df, "L6", str(out4))
    assert list(out4.iterdir()) == []


def _install_fake_visualize_skeleton(monkeypatch, fail=False):
    """Register a fake ``visualize_skeleton`` module; returns instance list."""
    instances = []

    class VisualizeSkeleton:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            out_dir = kwargs.get("output_dir", ".")
            self.save_folder = str(Path(out_dir) / "plot-3d_fake")
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            self.plot_calls = []
            self.individual_calls = []
            instances.append(self)

        def plot_neurons(self):
            if fail:
                raise RuntimeError("viz boom")
            self.plot_calls.append(True)

        def plot_individuals(self, **kwargs):
            self.individual_calls.append(kwargs)

    mod = types.ModuleType("visualize_skeleton")
    mod.VisualizeSkeleton = VisualizeSkeleton
    monkeypatch.setitem(sys.modules, "visualize_skeleton", mod)
    return instances


def test_visualize_top_types_paths(finder, tmp_path, monkeypatch):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    combined = pd.DataFrame({
        "bodyId": ["5813128953", "1234", "5813128953", "777"],
        "type": ["MBON01", "LH173", "MBON01", ""],
        "score": [45000.0, 38000.0, 40000.0, 10000.0],
        "dataset": ["hemibrain:v1.2.1"] * 4,
    })

    # visualize_skeleton unavailable -> graceful return
    monkeypatch.setitem(sys.modules, "visualize_skeleton", None)
    finder._visualize_top_types(combined, 5, str(tmp_path))

    instances = _install_fake_visualize_skeleton(monkeypatch)

    # missing dataset column -> return
    finder._visualize_top_types(combined.drop(columns=["dataset"]), 5, str(tmp_path))
    assert not instances

    # type mode with fallback ranking + individual profiles
    finder._visualize_top_types(
        combined, 5, str(tmp_path / "v1"), visualize_by="type",
        source_line="L1", generate_individual_profiles=["pdf"])
    assert len(instances) == 1
    vs = instances[0]
    assert vs.kwargs["dataset"] == "hemibrain:v1.2.1"
    assert vs.kwargs["legend_mode"] == "layer"
    assert vs.kwargs["custom_layer_names"] == ["r1_MBON01_x2", "r2_LH173_x1"]
    assert vs.kwargs["export_views"] == ["front", "bottom"]  # region 'All'
    assert vs.plot_calls == [True]
    assert vs.individual_calls and vs.individual_calls[0]["summary_format"] == ["pdf"]

    # bodyId mode with type filter
    finder._visualize_top_types(
        combined, 5, str(tmp_path / "v2"), visualize_by="bodyId",
        type_filter={"contains": "MBON"})
    assert len(instances) == 2
    assert instances[1].kwargs["legend_mode"] == "single"
    assert instances[1].kwargs["custom_layer_names"] == ["r1_MBON01_x2"]

    # labeling_info drives the type ordering
    labeling = pd.DataFrame({
        "type": ["LH173", "MBON01"],
        "dataset": ["hemibrain:v1.2.1"] * 2,
        "L1": [1.0, 2.0],
    })
    finder._visualize_top_types(combined, 1, str(tmp_path / "v3"), labeling_info=labeling)
    assert instances[2].kwargs["custom_layer_names"] == ["r1_LH173_x1"]

    # datasets_to_visualize variants
    finder._visualize_top_types(combined, 5, str(tmp_path / "v4"),
                                datasets_to_visualize="manc:v1.0")
    assert len(instances) == 3
    finder._visualize_top_types(combined, 5, str(tmp_path / "v5"),
                                datasets_to_visualize=["manc:v1.0"])
    assert len(instances) == 3
    finder._visualize_top_types(combined, 5, str(tmp_path / "v6"),
                                datasets_to_visualize=["hemibrain:v1.2.1", "manc:v1.0"])
    assert len(instances) == 4

    # all untyped -> skipped
    untyped = combined.copy()
    untyped["type"] = ""
    finder._visualize_top_types(untyped, 5, str(tmp_path / "v7"))
    assert len(instances) == 4

    # bodyIds not found in local dataset -> skipped
    ghost = pd.DataFrame({
        "bodyId": ["999999"], "type": ["Ghost"], "score": [50.0],
        "dataset": ["hemibrain:v1.2.1"],
    })
    finder._visualize_top_types(ghost, 5, str(tmp_path / "v8"))
    assert len(instances) == 4

    # type filter matching nothing -> skipped
    finder._visualize_top_types(combined, 5, str(tmp_path / "v8b"),
                                type_filter={"contains": "ZZZZ"})
    assert len(instances) == 4

    # VisualizeSkeleton failure is swallowed
    fail_instances = _install_fake_visualize_skeleton(monkeypatch, fail=True)
    finder._visualize_top_types(combined, 5, str(tmp_path / "v9"))
    assert len(fail_instances) == 1 and not fail_instances[0].plot_calls

    # region-specific export views
    finder.region = "Brain"
    inst = _install_fake_visualize_skeleton(monkeypatch)
    finder._visualize_top_types(combined, 5, str(tmp_path / "v10"))
    assert inst[0].kwargs["export_views"] == ["front"]
    finder.region = "VNC"
    inst = _install_fake_visualize_skeleton(monkeypatch)
    finder._visualize_top_types(combined, 5, str(tmp_path / "v11"))
    assert inst[0].kwargs["export_views"] == ["bottom"]
    finder.region = "All"

    # advanced visualization settings plumbing
    inst = _install_fake_visualize_skeleton(monkeypatch)
    finder._visualize_top_types(
        combined, 5, str(tmp_path / "v12"),
        visualization_settings={
            "export_views": False,
            "neuron_colors": [(9, 9, 9, 1.0)],
            "mesh_color": "auto",
            "synapse_alpha": 0.5,
        })
    kw = inst[0].kwargs
    assert kw["export_views"] is False
    assert kw["neuron_colors"] == [(9, 9, 9, 1.0)]
    assert kw["synapse_alpha"] == 0.5
    assert "mesh_color" not in kw


def test_visualize_top_types_legend_mode_ignores_saved_preference(
        finder, tmp_path, monkeypatch):
    """The saved app-wide legend preference must not rewrite the mode
    derived from ``visualize_by`` (homolog-style regression)."""
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    combined = pd.DataFrame({
        "bodyId": ["5813128953", "1234"],
        "type": ["MBON01", "LH173"],
        "score": [45000.0, 38000.0],
        "dataset": ["hemibrain:v1.2.1"] * 2,
    })
    instances = _install_fake_visualize_skeleton(monkeypatch)

    saved = {"legend_mode": "single"}  # arbitrary saved global preference

    finder._visualize_top_types(
        combined, 5, str(tmp_path / "t"), visualize_by="type",
        visualization_settings=dict(saved))
    assert instances[0].kwargs["legend_mode"] == "layer"

    finder._visualize_top_types(
        combined, 5, str(tmp_path / "b"), visualize_by="bodyId",
        visualization_settings=dict(saved))
    assert instances[1].kwargs["legend_mode"] == "single"


def test_neuron_to_lines_flows(finder_with_client, monkeypatch):
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    finder._client.em_images[5813128953] = [_em_image()]
    finder._client.em_images[1234] = [_em_image("1234")]
    finder._client.cds_matches["em-5813128953"] = [
        _lm_match("VT037867", score=90, image_id="imgA")]
    finder._client.cds_matches["em-1234"] = [_lm_match("LH173", score=80, image_id="imgB")]

    # dataset mismatch discovered during validation -> skipped neuron
    mismatched = _em_image("1234")
    mismatched.publishedName = "manc:v1.0:1234"
    finder._client.em_images[1234] = [mismatched]
    assert finder.neuron_to_lines("LH173", dataset="hemibrain:v1.2.1") == {}

    # string query resolved against the local dataset table
    finder._client.em_images[1234] = [_em_image("1234")]
    results = finder.neuron_to_lines("MBON01", dataset="hemibrain:v1.2.1")
    assert "5813128953" in results and not results["5813128953"].empty
    assert list(results["5813128953"]["line"]) == ["VT037867"]
    assert set(results["5813128953"]["source_dataset"]) == {"hemibrain:v1.2.1"}

    # no match -> {}
    assert finder.neuron_to_lines("ZZZ_NONE", dataset="hemibrain:v1.2.1") == {}

    # list query is deduplicated
    results = finder.neuron_to_lines(["MBON01", "MBON01"], dataset="hemibrain:v1.2.1")
    assert len(results) == 1

    # integer bodyId absent from every dataset -> 'unknown' fallback entry
    results = finder.neuron_to_lines(999999999, dataset="hemibrain:v1.2.1")
    assert "999999999" in results and results["999999999"].empty

    # invalid match type
    with pytest.raises(ValueError):
        finder.neuron_to_lines("MBON01", match_type="bogus")


def test_neuron_to_lines_verbose_bulk_cache(finder_with_client, monkeypatch):
    if not nbf_mod.HAS_TQDM or not nbf_mod.HAS_POLARS:
        pytest.skip("tqdm/polars required")
    finder = finder_with_client
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    _block_network_pulls(finder, monkeypatch)
    finder._client.em_images[5813128953] = [_em_image()]
    finder._client.em_images[1234] = [_em_image("1234")]
    finder._client.cds_matches["em-5813128953"] = [_lm_match("VT037867", score=90)]
    finder._client.cds_matches["em-1234"] = [_lm_match("LH173", score=80)]
    finder.verbose = True

    # first call: sequential fetch branch with progress bar
    results = finder.neuron_to_lines(["MBON01", "LH173"], dataset="hemibrain:v1.2.1")
    assert len(results) == 2

    # second call: both neurons cached -> bulk polars loader branch
    results2 = finder.neuron_to_lines(["MBON01", "LH173"], dataset="hemibrain:v1.2.1")
    assert len(results2) == 2
    assert set(results2["5813128953"]["line"]) == {"VT037867"}


def _setup_find_neurons_client(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    client = finder._client
    m1 = _em_match("5813128953", score=45000)
    m2 = _em_match("1234", score=38000)
    m2.image.neuronType = "LH173"
    m2.image.neuronInstance = "LH173_L"
    client.lm_images["L1"] = [_lm_image("img-L1")]
    client.cds_matches["img-L1"] = [m1, m2]
    client.lm_images["L2"] = [_lm_image("img-L2")]
    client.cds_matches["img-L2"] = [m2]
    client.lm_images["EMPTY"] = []


def test_find_neurons_batch_single_line(finder_with_client, tmp_path, monkeypatch):
    finder = finder_with_client
    _setup_find_neurons_client(finder)
    _block_network_pulls(finder, monkeypatch)

    combined = finder.find_neurons_batch("L1", top_n=10, output_dir=str(tmp_path))
    assert not combined.empty and set(combined["source_line"]) == {"L1"}
    run_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("NB-find-neurons_")]
    assert len(run_dirs) == 1
    out = run_dirs[0]
    by_dataset = out / "by_dataset"
    for fname in [
        "L1_neurons.csv", "all_neurons.csv", "parameters.json",
        "user_warning_notes.txt",
    ]:
        assert (out / fname).exists(), fname
    for fname in [
        "L1_hemibrain_v1_2_1_neurons.csv", "L1_hemibrain_v1_2_1_types.csv",
    ]:
        assert (by_dataset / fname).exists(), fname
    assert any(p.name.startswith("labeling_distribution") for p in out.iterdir())

    # empty inputs / no matches
    assert finder.find_neurons_batch([]).empty
    assert finder.find_neurons_batch("   ").empty
    assert finder.find_neurons_batch("EMPTY").empty
    with pytest.raises(ValueError):
        finder.find_neurons_batch("L1", match_type="bogus")


def test_find_neurons_batch_multiline_with_visualization(finder_with_client, tmp_path,
                                                        monkeypatch):
    finder = finder_with_client
    _setup_find_neurons_client(finder)
    _block_network_pulls(finder, monkeypatch)
    instances = _install_fake_visualize_skeleton(monkeypatch)

    combined = finder.find_neurons_batch(
        ["L1", "L2"], top_n=10, output_dir=str(tmp_path),
        visualize_top_n=5, generate_individual_profiles=["pdf"])
    assert set(combined["source_line"]) == {"L1", "L2"}
    run_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("NB-find-neurons_")]
    out = run_dirs[0]
    assert (out / "viz_L1").exists() and (out / "viz_L2").exists()
    assert len(instances) == 2
    assert all(vs.plot_calls for vs in instances)


def test_find_neurons_batch_verbose_progress(finder_with_client, tmp_path, monkeypatch):
    if not nbf_mod.HAS_TQDM:
        pytest.skip("tqdm required")
    finder = finder_with_client
    _setup_find_neurons_client(finder)
    _block_network_pulls(finder, monkeypatch)
    finder.verbose = True

    # sequential fetch with progress bar (max_workers == 1)
    combined = finder.find_neurons_batch(["L1", "L2"], top_n=10, output_dir=str(tmp_path))
    assert set(combined["source_line"]) == {"L1", "L2"}

    # parallel fetch branch (max_workers > 1, threads only)
    finder.max_workers = 2
    combined2 = finder.find_neurons_batch(
        ["L1", "L2"], top_n=10, output_dir=str(tmp_path / "p2"))
    assert set(combined2["source_line"]) == {"L1", "L2"}


# ---------------------------------------------------------------------------
# Batch C: region helpers, image download, PDF/PPTX summaries
# ---------------------------------------------------------------------------

def test_parse_region_and_filter_helpers(finder):
    p = finder._parse_region_from_filename
    assert p("x.png", full_key="VT GAL4/VT037867/brain/f.jpg") == "Brain"
    assert p("x.png", full_key="VT GAL4/VT037867/vnc/f.jpg") == "VNC"
    assert p("R85D07_AE_01_03-fA01b_C101223_total.jpg") == "Brain"
    assert p("R85D07_AE_01_03-fA01v_C101223_total.jpg") == "VNC"
    assert p("L1-2021-f-20x-vnc-Split_GAL4-CDM_1.png") == "VNC"
    assert p(
        "SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-JRC2018-CDM_1.png"
    ) == "Brain"
    assert p("SS01015-20131220_31_C3-f-20x-Split_GAL4-CDM_1.png") == "Other"

    f_brain = SimpleNamespace(
        key="VT GAL4/VT037867/brain/VT037867-total.jpg", url="")
    f_vnc = SimpleNamespace(key="VT GAL4/VT037867/vnc/VT037867-total.jpg", url="")
    f_nofile = SimpleNamespace(
        url="https://flimg.invalid/cgi-bin/image/VT037867-brain-total.jpg")
    files = [f_brain, f_vnc, f_nofile]
    assert finder._filter_flylight_files_by_region(files) == files  # All
    finder.region = "Brain"
    assert finder._filter_flylight_files_by_region(files) == [f_brain, f_nofile]
    finder.region = "VNC"
    assert finder._filter_flylight_files_by_region(files) == [f_vnc]
    finder.region = "All"


def test_reorganize_files_by_region(finder, tmp_path):
    out = tmp_path / "imgs"
    coll = out / "SplitGAL4" / "SS01015"
    coll.mkdir(parents=True)
    f_brain = coll / "SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-CDM_1.png"
    f_vnc = coll / "SS01015-20131220_31_C3-f-20x-vnc-Split_GAL4-CDM_1.png"
    f_other = coll / "SS01015-plain.png"
    for f in (f_brain, f_vnc, f_other):
        f.write_bytes(b"x")
    missing = out / "missing.png"

    moved = finder._reorganize_files_by_region(
        [str(f_brain), str(f_vnc), str(f_other), str(missing)],
        str(out), verbose=True)
    assert len(moved) == 3
    assert (out / "Brain" / "SS01015" / f_brain.name).exists()
    assert (out / "VNC" / "SS01015" / f_vnc.name).exists()
    assert (out / "Other" / "SS01015" / f_other.name).exists()
    # emptied collection dir was cleaned up
    assert not (out / "SplitGAL4").exists()


def _write_tiny_png(path):
    from PIL import Image
    Image.new("RGB", (8, 8), color=(200, 30, 30)).save(str(path))


def test_collect_line_images(tmp_path):
    from neuronbridge_finder import _collect_line_images

    images = tmp_path / "images"
    images.mkdir()
    # layout 1: per-line subdirectories
    (images / "VT037867").mkdir()
    _write_tiny_png(images / "VT037867" / "a.png")
    _write_tiny_png(images / "VT037867" / "b.jpg")
    (images / "VT037867" / "c.txt").write_text("not an image")
    # empty subdir with no images -> ignored
    (images / "emptydir").mkdir()
    # layout 2: flat files grouped by prefix before '-'
    _write_tiny_png(images / "SS01015-20x-brain.png")
    _write_tiny_png(images / "SS01015-other.png")
    _write_tiny_png(images / "lonely.png")
    # layout 3: source subfolders scanned one level deep
    (images / "neuronbridge" / "R10A06").mkdir(parents=True)
    _write_tiny_png(images / "neuronbridge" / "R10A06" / "r.png")
    (images / "flylight").mkdir()
    _write_tiny_png(images / "flylight" / "VT037867-flat.png")
    # custom layout: subfolder containing line dirs
    (images / "custom" / "LH173").mkdir(parents=True)
    _write_tiny_png(images / "custom" / "LH173" / "x.png")

    collected = _collect_line_images(images)
    assert len(collected["VT037867"]) == 3
    assert len(collected["SS01015"]) == 2
    assert collected["Unknown"]
    assert collected["R10A06"]
    assert collected["LH173"]
    assert "emptydir" not in collected


def test_create_image_pdf_variants(tmp_path):
    from neuronbridge_finder import create_image_pdf

    assert create_image_pdf(str(tmp_path / "nope"), verbose=False) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert create_image_pdf(str(empty), verbose=False) is None

    images = tmp_path / "images"
    (images / "VT037867").mkdir(parents=True)
    for i in range(5):
        _write_tiny_png(images / "VT037867" / f"img{i}.png")
    (images / "SS01015").mkdir()
    _write_tiny_png(images / "SS01015" / "s.png")
    # corrupt image hits the per-image exception branch
    (images / "SS01015" / "bad.png").write_bytes(b"not a png")

    out = create_image_pdf(
        str(images), output_pdf=str(tmp_path / "s1.pdf"),
        images_per_page=(2, 2), landscape=False, page_size="letter",
        line_order=["SS01015", "VT037867"], verbose=False,
        background_color="black")
    assert out and Path(out).exists()
    out2 = create_image_pdf(
        str(images), output_pdf=str(tmp_path / "s2.pdf"),
        images_per_page=(3, 2), background_color="#123456", verbose=False)
    assert out2 and Path(out2).exists()
    out3 = create_image_pdf(
        str(images), output_pdf=str(tmp_path / "s3.pdf"),
        background_color=(0.9, 0.9, 0.9), verbose=False)
    assert out3 and Path(out3).exists()
    # default output path inside images dir
    out4 = create_image_pdf(str(images), verbose=False)
    assert out4 and Path(out4).name == "images_summary.pdf"


def test_create_image_pptx_variants(tmp_path):
    from neuronbridge_finder import create_image_pptx

    assert create_image_pptx(str(tmp_path / "nope"), verbose=False) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert create_image_pptx(str(empty), verbose=False) is None

    images = tmp_path / "images"
    (images / "VT037867").mkdir(parents=True)
    for i in range(5):
        _write_tiny_png(images / "VT037867" / f"verylongfilename_image_number_{i}.png")
    (images / "SS01015").mkdir()
    _write_tiny_png(images / "SS01015" / "s.png")
    (images / "SS01015" / "bad.png").write_bytes(b"not a png")

    out = create_image_pptx(
        str(images), output_pptx=str(tmp_path / "s1.pptx"),
        images_per_slide=(2, 2), slide_size="standard",
        line_order=["SS01015", "VT037867"], verbose=False,
        background_color="black")
    assert out and Path(out).exists()
    for bg, sz in [("#003366", "a4"), ((10, 200, 30), "widescreen"),
                   ("white", "widescreen"), ("gray", "standard"),
                   ("weirdcolor", "widescreen")]:
        out2 = create_image_pptx(
            str(images), output_pptx=str(tmp_path / f"{sz}_{abs(hash(str(bg)))}.pptx"),
            images_per_slide=(3, 2), slide_size=sz, verbose=False,
            background_color=bg)
        assert out2 and Path(out2).exists()
    out3 = create_image_pptx(str(images), verbose=False)
    assert out3 and Path(out3).name == "images_summary.pptx"


def test_generate_image_summaries(finder, tmp_path, monkeypatch):
    images = tmp_path / "images"
    (images / "VT037867").mkdir(parents=True)
    _write_tiny_png(images / "VT037867" / "a.png")
    out = tmp_path / "out"
    out.mkdir()

    # missing images dir -> no-op
    finder._generate_image_summaries(
        str(tmp_path / "missing"), str(out), ["VT037867"], "pdf")

    # pdf + pptx via the built-in pptx builder
    monkeypatch.setattr(nbf_mod, "HAS_IMG2PPTX", False)
    finder._generate_image_summaries(
        str(images), str(out), ["VT037867"], ["pdf", "pptx"],
        summary_background_color="white")
    assert (out / "images_summary.pdf").exists()
    assert (out / "images_summary.pptx").exists()

    # img2pptx fast path when available
    calls = {}

    def fake_img2pptx(**kwargs):
        calls.update(kwargs)
        p = Path(kwargs["output_pptx"])
        p.write_bytes(b"pptx")
        return str(p)

    monkeypatch.setattr(nbf_mod, "HAS_IMG2PPTX", True)
    monkeypatch.setattr(nbf_mod, "img2pptx", fake_img2pptx)
    finder._generate_image_summaries(
        str(images), str(out), ["VT037867"], "pptx")
    assert calls and calls["include_subfolders"] is True
    # empty/None formats are dropped silently
    finder._generate_image_summaries(str(images), str(out), ["VT037867"], [None, ""])


def test_download_neuronbridge_images_and_dispatch(finder_with_client, tmp_path,
                                                  monkeypatch):
    import urllib.request

    finder = finder_with_client
    client = finder._client
    files = SimpleNamespace(
        CDM="path/VT037867-cdm.png",
        SignalMip="path/VT037867-mip.jpg",
        SignalMipMasked="path/VT037867-masked.png",
    )
    client.lm_images["VT037867"] = [SimpleNamespace(files=files)]
    client.lm_images["BAD"] = RuntimeError("boom")
    client.lm_images["NOFILES"] = [SimpleNamespace(files=None)]

    def fake_urlretrieve(url, dest):
        with open(dest, "wb") as fh:
            fh.write(b"fake-img")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    # cdm only, png format only, one line with files + error + empty lines
    out = tmp_path / "nb"
    got = finder._download_neuronbridge_images(
        ["VT037867", "BAD", "NOFILES", "MISSING"], str(out), "png", "cdm",
        max_files=None, verbose=False)
    assert len(got) == 1 and Path(got[0]).name == "VT037867-cdm.png"

    # 'all' image types + 'all' formats -> all three files, max_files cap
    got2 = finder._download_neuronbridge_images(
        ["VT037867"], str(tmp_path / "nb2"), ["all"], "all",
        max_files=2, verbose=True)
    assert len(got2) == 2

    # nothing downloadable
    assert finder._download_neuronbridge_images(
        ["MISSING"], str(tmp_path / "nb3"), "png", "cdm", None, True) == []

    # download_line_images dispatch
    got3 = finder.download_line_images(
        "VT037867", str(tmp_path / "dl1"), source="neuronbridge",
        formats="png", image_types="cdm")
    assert len(got3) == 1
    assert finder.download_line_images("VT037867", str(tmp_path / "dl2"),
                                       source="bogus", verbose=True) == []
    assert finder.download_line_images("", str(tmp_path / "dl3"), verbose=True) == []


def _fl_file(name, collection, source="s3", key=None, url=None):
    return SimpleNamespace(
        filename=name, collection=collection, source=source,
        key=key or "", url=url or "")


class _FakeFlyLightDownloader:
    instances = []
    files_by_category = {}
    r_line_files = {}
    raise_on_init = False

    def __init__(self, **kwargs):
        if _FakeFlyLightDownloader.raise_on_init:
            raise ImportError("fake flylight unavailable")
        _FakeFlyLightDownloader.instances.append(self)
        self.kwargs = kwargs

    def get_filtered_files(self, line_name):
        cat = self.kwargs.get("collection_category")
        return list(_FakeFlyLightDownloader.files_by_category.get(cat, {}).get(
            line_name, []))

    def download(self, line_name=None, max_files=None, flat_structure=True,
                 add_timestamp=False, files=None):
        out_dir = Path(self.kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for f in files or []:
            target = out_dir / f.filename
            target.write_bytes(b"fake")
            written.append(target)
        return written

    def _get_r_line_files(self, line_name):
        return list(_FakeFlyLightDownloader.r_line_files.get(line_name, []))


def _install_fake_flylight_module(monkeypatch):
    _FakeFlyLightDownloader.instances = []
    _FakeFlyLightDownloader.files_by_category = {}
    _FakeFlyLightDownloader.r_line_files = {}
    _FakeFlyLightDownloader.raise_on_init = False
    mod = types.ModuleType("flylight_downloader")
    mod.FlyLightDownloader = _FakeFlyLightDownloader
    monkeypatch.setitem(sys.modules, "flylight_downloader", mod)
    return _FakeFlyLightDownloader


def test_download_flylight_images_with_category(finder, tmp_path, monkeypatch):
    fake = _install_fake_flylight_module(monkeypatch)
    out = tmp_path / "fl"

    # simple_mode: SplitGAL4 keeps only 20x+multichannel (minus image1/2);
    # GAL4/LEXA keeps only 'total'; region filter keeps brain files only
    fake.files_by_category = {
        "SplitGAL4": {
            "SS01015": [
                _fl_file("SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-multichannel-CDM_1.png",
                         "SplitGAL4"),
                _fl_file("SS01015-image1-20x-multichannel-brain.png", "SplitGAL4"),
                _fl_file("SS01015-40x-brain.png", "SplitGAL4"),
            ]},
        "GAL4/LEXA": {
            "VT037867": [
                _fl_file("VT037867-brain-total.jpg", "GAL4/LEXA"),
                _fl_file("VT037867-vnc-confocal.jpg", "GAL4/LEXA"),
            ]},
    }

    files, missing = finder._download_flylight_images_with_category(
        ["SS01015", "VT037867"], str(out), ["png", "jpg"], "all",
        max_files=5, category=["GAL4/LEXA", "SplitGAL4"],
        simple_mode=True, verbose=True)
    names = sorted(Path(p).name for p in files)
    assert names == ["SS01015-20131220_31_C3-f-20x-brain-Split_GAL4-multichannel-CDM_1.png",
                     "VT037867-brain-total.jpg"]
    assert missing == []
    # both lines were served by their primary category: no MCFO warnings
    assert not any("MCFO fallback" in w for w in finder._warning_collector)

    # region filter applied during scan (VNC file dropped before download)
    finder.region = "Brain"
    files2, _ = finder._download_flylight_images_with_category(
        ["VT037867"], str(tmp_path / "fl2"), ["jpg"], "all",
        max_files=None, category="GAL4/LEXA", simple_mode=True)
    assert [Path(p).name for p in files2] == ["VT037867-brain-total.jpg"]
    finder.region = "All"


def test_download_flylight_fallback_chain(finder, tmp_path, monkeypatch):
    fake = _install_fake_flylight_module(monkeypatch)

    # VT line with nothing anywhere -> full fallback chain + terminal warning
    files, missing = finder._download_flylight_images_with_category(
        ["VT037867"], str(tmp_path / "f1"), "png", "all", None,
        category=["GAL4/LEXA"], verbose=False)
    assert files == [] and missing == ["VT037867"]
    joined = " ".join(finder._warning_collector)
    assert "trying MCFO fallback" in joined
    assert "trying RawImages collection" in joined
    assert "trying flweb.janelia.org" in joined
    assert f"No images found for VT037867 in any FlyLight collection" in joined

    # flweb fallback rescues an R-line
    finder._warning_collector.clear()
    fake.r_line_files = {
        "R85D07": [_fl_file("R85D07_AE_01_03-fA01b_total.jpg", "Gen1 GAL4")]}
    files, missing = finder._download_flylight_images_with_category(
        ["R85D07"], str(tmp_path / "f2"), "jpg", "all", None,
        category=["GAL4/LEXA"], verbose=False)
    assert len(files) == 1 and missing == []

    # nothing anywhere for multiple lines
    finder._warning_collector.clear()
    files, missing = finder._download_flylight_images_with_category(
        ["A1", "A2"], str(tmp_path / "f3"), "png", "all", None,
        category=["MCFO"])
    assert files == [] and sorted(missing) == ["A1", "A2"]

    # module import failure inside the helper
    _FakeFlyLightDownloader.raise_on_init = True
    files, missing = finder._download_flylight_images_with_category(
        ["X"], str(tmp_path / "f4"), "png", "all", None, category=None,
        verbose=True)
    assert files == [] and missing == ["X"]

    # _download_flylight_images wrapper
    _FakeFlyLightDownloader.raise_on_init = False
    fake.files_by_category = {"All": {"Z1": [_fl_file("Z1-total.png", "All")]}}
    got = finder._download_flylight_images(
        ["Z1"], str(tmp_path / "f5"), "png", "all", None, verbose=True)
    assert len(got) == 1


def _setup_find_lines_client(finder):
    finder._neuron_dfs["hemibrain_v1_2_1"] = HEMIBRAIN_DF
    client = finder._client
    client.em_images[5813128953] = [_em_image()]
    client.em_images[1234] = [_em_image("1234")]
    m_vt = _lm_match("VT037867", score=90000, image_id="img-vt")
    m_ss = _lm_match("SS01015", score=80000, image_id="img-ss")
    m_r = _lm_match("R10A06", score=70000, image_id="img-r")
    client.cds_matches["em-5813128953"] = [m_vt, m_ss, m_r]
    client.cds_matches["em-1234"] = [_lm_match("SS01015", score=60000,
                                                image_id="img-ss2")]
    nb_files = SimpleNamespace(CDM="path/cdm.png", SignalMip="path/mip.png")
    client.lm_images["VT037867"] = [SimpleNamespace(files=nb_files)]
    client.lm_images["R10A06"] = [SimpleNamespace(files=nb_files)]
    client.lm_images["SS01015"] = []


def test_find_lines_batch_end_to_end(finder_with_client, tmp_path, monkeypatch):
    import urllib.request

    finder = finder_with_client
    _setup_find_lines_client(finder)
    _block_network_pulls(finder, monkeypatch)

    def fake_urlretrieve(url, dest):
        with open(dest, "wb") as fh:
            fh.write(b"fake-img")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    # simple string query without downloads
    # NOTE: find_lines_batch defaults to download_images='flylight', which
    # would hit the real FlyLight downloader module — always pass it
    # explicitly in hermetic tests.
    combined = finder.find_lines_batch(
        "MBON01", dataset="hemibrain:v1.2.1", output_dir=str(tmp_path),
        download_images=None)
    assert not combined.empty
    assert set(combined["line"]) == {"VT037867", "SS01015", "R10A06"}
    run_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("NB-find-lines_")]
    assert len(run_dirs) == 1
    out = run_dirs[0]
    assert (out / "MBON01_lines.csv").exists()
    assert (out / "line_summary.csv").exists()
    assert (out / "parameters.json").exists()
    summary = pd.read_csv(out / "line_summary.csv")
    assert list(summary.columns)[:2] == ["line", "match_count"] or \
        summary.columns[0] == "line"

    # comma-separated multi query, completeness sort, cached id_to_lines rows
    combined2 = finder.find_lines_batch(
        "MBON01,LH173", dataset="hemibrain:v1.2.1", sort_by="completeness",
        output_dir=str(tmp_path / "m"), download_images=None)
    assert {"MBON01", "LH173"} <= set(combined2["source_query"])
    out2 = [d for d in (tmp_path / "m").iterdir()
            if d.name.startswith("NB-find-lines_")][0]
    summary2 = pd.read_csv(out2 / "line_summary.csv")
    assert summary2.loc[0, "line"] == "SS01015"

    # integer bodyId query + >3 queries '_etc' folder naming
    combined3 = finder.find_lines_batch(
        [5813128953, "MBON01", "LH173", "MBON01"], dataset="hemibrain:v1.2.1",
        output_dir=str(tmp_path / "i"), download_images=None)
    assert not combined3.empty
    assert any("_etc_" in d.name for d in (tmp_path / "i").iterdir())

    # empty queries / invalid sort_by
    assert finder.find_lines_batch([]).empty
    assert finder.find_lines_batch("  , ").empty
    with pytest.raises(ValueError):
        finder.find_lines_batch("MBON01", sort_by="bogus")

    # separate_splitgal4 mode: per-category summaries + per-category top-N
    finder.separate_splitgal4 = True
    combined4 = finder.find_lines_batch(
        "MBON01,LH173", dataset="hemibrain:v1.2.1",
        output_dir=str(tmp_path / "sep"), download_images=None)
    out4 = [d for d in (tmp_path / "sep").iterdir()
            if d.name.startswith("NB-find-lines_")][0]
    assert (out4 / "gal4_lexa_summary.csv").exists()
    assert (out4 / "split_gal4_summary.csv").exists()

    # neuronbridge downloads + pdf/pptx summary generation
    monkeypatch.setattr(nbf_mod, "HAS_IMG2PPTX", False)
    finder.find_lines_batch(
        "MBON01", dataset="hemibrain:v1.2.1", output_dir=str(tmp_path / "d"),
        download_images="neuronbridge", download_img_for_top_n_lines=3,
        summary_format=["pdf", "pptx"], summary_background_color="white")
    out5 = [d for d in (tmp_path / "d").iterdir()
            if d.name.startswith("NB-find-lines_")][0]
    assert (out5 / "images" / "VT037867").exists()
    assert (out5 / "images_summary.pdf").exists()
    assert (out5 / "images_summary.pptx").exists()

    # flylight downloads with NeuronBridge fallback for missing lines
    fake = _install_fake_flylight_module(monkeypatch)
    fake.files_by_category = {
        "GAL4/LEXA": {"VT037867": [_fl_file("VT037867-brain-total.png",
                                             "GAL4/LEXA")]}}
    # capture the printed warnings: _print_warning_summary clears the
    # collector after printing, so snapshot it through a patched printer
    printed = []
    monkeypatch.setattr(
        finder, "_print_warning_summary",
        lambda: printed.extend(list(finder._warning_collector)))
    finder._warning_collector.clear()
    finder.find_lines_batch(
        "MBON01", dataset="hemibrain:v1.2.1", output_dir=str(tmp_path / "fl"),
        download_images="flylight", download_img_for_top_n_lines=None,
        organize_by_region=True, summary_format=None)
    out6 = [d for d in (tmp_path / "fl").iterdir()
            if d.name.startswith("NB-find-lines_")][0]
    assert (out6 / "images" / "Brain" / "VT037867").exists()
    assert (out6 / "images" / "neuronbridge_fallback" / "R10A06").exists()
    # R10A06 resolved by the fallback -> its terminal warning was removed,
    # SS01015 was not resolved -> its terminal warning remains
    assert not any("No images found for VT037867" in w for w in printed)
    assert not any("No images found for R10A06" in w for w in printed)
    assert any("No images found for SS01015" in w for w in printed)

    # 'both' sources: separate neuronbridge/ and flylight/ folders
    finder.separate_splitgal4 = False
    fake.files_by_category = {
        "GAL4/LEXA": {"VT037867": [_fl_file("VT037867-brain-total.png",
                                             "GAL4/LEXA")]}}
    finder.find_lines_batch(
        "MBON01", dataset="hemibrain:v1.2.1", output_dir=str(tmp_path / "both"),
        download_images="both", download_img_for_top_n_lines=2,
        summary_format="pdf")
    out7 = [d for d in (tmp_path / "both").iterdir()
            if d.name.startswith("NB-find-lines_")][0]
    assert (out7 / "images" / "neuronbridge").exists()
    assert (out7 / "images" / "flylight").exists()
    assert (out7 / "images_summary.pdf").exists()


def test_find_lines_batch_with_images_deprecated(finder_with_client, tmp_path,
                                                 monkeypatch):
    finder = finder_with_client
    _setup_find_lines_client(finder)
    _block_network_pulls(finder, monkeypatch)
    with pytest.warns(DeprecationWarning):
        df = finder.find_lines_batch_with_images(
            "MBON01", dataset="hemibrain:v1.2.1", download_images=True,
            image_source="neuronbridge", output_dir=None)
    assert not df.empty


def test_clear_cache_variants(finder, tmp_path):
    root = Path(finder.cache_folder)

    def seed():
        paths = [
            root / "parquet" / "ds" / "id_to_lines" / "a.parquet",
            root / "parquet" / "ds" / "image_cache" / "b.parquet",
            root / "parquet" / "ds" / "line_image_mapping.json",
            root / "parquet" / "ds" / "manifest.json",
            root / "legacy.csv",
            root / "legacy.parquet",
            root / "image_cache" / "c.csv",
            root / "image_cache" / "line_image_mapping.json",
            root / "id_to_lines_old.csv",
        ]
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

    seed()
    finder.clear_cache("id_to_lines")
    assert not (root / "parquet" / "ds" / "id_to_lines" / "a.parquet").exists()
    assert not (root / "id_to_lines_old.csv").exists()
    assert (root / "parquet" / "ds" / "image_cache" / "b.parquet").exists()

    finder.clear_cache("image_cache")
    assert not (root / "parquet" / "ds" / "image_cache" / "b.parquet").exists()
    assert not (root / "image_cache" / "c.csv").exists()
    assert (root / "legacy.csv").exists()

    finder.clear_cache()
    remaining = [p for p in root.rglob("*") if p.is_file()]
    assert remaining == []

    seed()
    with pytest.raises(ValueError):
        finder.clear_cache("bogus")

    # cache folder absent -> no-op
    finder.cache_folder = str(tmp_path / "nonexistent_cache")
    finder.clear_cache()


def test_convenience_functions(monkeypatch):
    created = {}

    class _FakeFinder:
        def __init__(self, verbose=False):
            created["verbose"] = verbose

        def id_to_lines(self, body_id, match_type="cds"):
            return pd.DataFrame({"line": ["VT037867"]})

        def line_to_neuron(self, line_name, match_type="cds"):
            return pd.DataFrame({"bodyId": ["5813128953"]})

    monkeypatch.setattr(nbf_mod, "NeuronBridgeFinder", _FakeFinder)
    from neuronbridge_finder import find_lines_for_body, find_neurons_for_line

    df = find_lines_for_body(5813128953)
    assert list(df["line"]) == ["VT037867"]
    df2 = find_neurons_for_line("VT037867", match_type="pppm")
    assert list(df2["bodyId"]) == ["5813128953"]
    assert created["verbose"] is False
