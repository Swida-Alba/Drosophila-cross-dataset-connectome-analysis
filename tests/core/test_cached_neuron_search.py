"""Coverage for the parquet-backed neuron resolver used by analysis code."""

import pandas as pd
import polars as pl


def _write_search_cache(tmp_path):
    from src.neuron_index_builder import build_search_cache_frame, search_cache_path

    dataset = "cache-test:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    frame = pl.DataFrame(
        {
        "bodyId": ["100", "200", "300", "400", "500", "600"],
        "type": ["MTe01a", "Other", "MeVPaMe2", "", "aMe17a", "aMe17e"],
        "instance": ["MTe01a_L", "Other_L", "MeVPaMe2_R", "MTe_R", "aMe17a_L", "aMe17e_L"],
        "flywireType": ["MTe07", "MTe12", "aMe19a", "MTe27", "aMe17a1", ""],
        "hemibrainType": ["", "", "", "", "", "aMe17a"],
        }
    )
    frame.write_parquet(index_path)
    build_search_cache_frame(frame).write_parquet(search_cache_path(index_path))
    return dataset, frame


def test_cached_resolver_preserves_priority_and_explicit_prefix_short_circuit(tmp_path):
    from src.neuron_search import get_cached_neuron_search, resolve_neuron_query
    from src.statvis import _process_single_neuron

    dataset, frame = _write_search_cache(tmp_path)
    cache = get_cached_neuron_search(dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None

    # Bare names are strict and priority ordered.
    ids, info = resolve_neuron_query(cache, "MTe01a")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # The cached resolver applies the same first-column rule as the
    # dataframe fallback: the type-owned aMe17a identity does not expand to
    # the later hemibrainType value on the independent aMe17e row.
    ids, info = resolve_neuron_query(cache, "aMe17a")
    assert ids == ["500"]
    assert info["matched_column"] == "type"
    ids, info = resolve_neuron_query(cache, "aMe17a.*")
    assert ids == ["500"]
    assert info["matched_column"] == "type"

    # A literal prefix stops at the first priority column. Later-column names
    # remain available to the viewer as secondary evidence, but must not
    # expand a pathfinding query after a type match exists.
    ids, info = resolve_neuron_query(cache, "MTe.*")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # The plain token is not silently promoted to a prefix query.
    ids, info = resolve_neuron_query(cache, "MeVPa")
    assert ids == []
    assert info["matched_column"] is None

    # Numeric automatic searches are identity-only and prefix-restricted.
    ids, info = resolve_neuron_query(cache, "20")
    assert ids == []
    assert info["matched_column"] is None
    ids, info = resolve_neuron_query(cache, "20.*")
    assert ids == ["200"]
    assert info["matched_column"] == "bodyId"

    # A type scope may intentionally search a numeric-looking type name.
    ids, info = resolve_neuron_query(cache, "200", search_columns="type")
    assert ids == []
    assert info["matched_column"] is None

    # Compare the cache resolver with the authoritative dataframe matcher for
    # the legacy pathfinding forms. Ordering may differ because the sidecar is
    # priority/value sorted, but the resolved neuron set must be identical.
    pandas_frame = frame.to_pandas()
    body_ids = pandas_frame["bodyId"].tolist()
    for query in ("MTe01a", "MTe.*", "MeVPa", "MeVPa.*", ".*aMe.*", "20", "20.*"):
        cached_ids, _ = resolve_neuron_query(cache, query)
        dataframe_ids, _ = _process_single_neuron(
            query,
            pandas_frame,
            body_ids,
            verbose=False,
        )
        assert set(cached_ids) == {str(value) for value in dataframe_ids}


def test_get_neurons_uses_cache_resolver_before_dataframe_scan(tmp_path, monkeypatch):
    import src.statvis as statvis
    from src.neuron_search import get_cached_neuron_search

    dataset, frame = _write_search_cache(tmp_path)
    cache = get_cached_neuron_search(dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None
    pandas_frame = frame.to_pandas()
    roi_frame = pd.DataFrame({"bodyId": pandas_frame["bodyId"]})

    monkeypatch.setattr(
        statvis,
        "_ensure_local_dataset_files",
        lambda *args, **kwargs: ("cache-test_v1_0", "unused-prefix"),
    )
    monkeypatch.setattr(
        statvis,
        "_get_cached_neuron_df",
        lambda *args, **kwargs: (pandas_frame.copy(), roi_frame.copy()),
    )
    monkeypatch.setattr(statvis, "_get_cached_neuron_search", lambda _dataset: cache)

    def fail_dataframe_match(*args, **kwargs):
        raise AssertionError("cached query unexpectedly fell back to dataframe scan")

    monkeypatch.setattr(statvis, "_process_single_neuron", fail_dataframe_match)

    neurons, _, _, _ = statvis.getNeurons(
        ["MTe.*"],
        dataset=dataset,
        verbose=False,
    )
    assert neurons["bodyId"].tolist() == ["100"]


def test_an_emitted_T_star_key_is_read_as_the_key_before_the_glob(tmp_path):
    """``SMP540*`` is the key the typing registry emits, not a family glob.

    Copying that key out of a typed export used to ask for every name STARTING
    with ``SMP540`` — which includes the bare native type wherever it is
    legitimately untouched (_plan/pending_issues.md). A frame carrying the
    emitted key answers it exactly; a frame that has none keeps the prefix
    reading, because this lane holds no label mapper and must not invent the
    class either. Both resolvers carry the rule, so the sidecar and the
    authoritative dataframe cannot disagree about one query.
    """
    from src.neuron_index_builder import (
        build_search_cache_frame,
        search_cache_path,
    )
    from src.neuron_search import (
        get_cached_neuron_search,
        resolve_dataframe_query,
        resolve_neuron_query,
    )

    typed = pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "type": ["SMP540*", "SMP540*", "SMP540", "SMP5401"],
        "instance": ["a", "b", "c", "d"],
    })
    ids, info = resolve_dataframe_query(typed, "SMP540*")
    assert [str(value) for value in ids] == ["1", "2"]
    assert info["matched_column"] == "type"
    # the hemisphere spelling addresses the same key
    ids_l, _info_l = resolve_dataframe_query(typed, "SMP540*_L")
    assert [str(value) for value in ids_l] == ["1", "2"]

    # nothing carries the key: the prefix reading still answers, unchanged
    plain = pd.DataFrame({"bodyId": ["3", "4"],
                          "type": ["SMP540", "SMP5401"],
                          "instance": ["c", "d"]})
    ids2, info2 = resolve_dataframe_query(plain, "SMP540*")
    assert [str(value) for value in ids2] == ["3", "4"]
    assert info2["matched_column"] == "type"

    # the cached sidecar path agrees
    dataset = "residual-key-test:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    plframe = pl.from_pandas(typed)
    plframe.write_parquet(index_path)
    build_search_cache_frame(plframe).write_parquet(
        search_cache_path(index_path))
    cache = get_cached_neuron_search(
        dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None
    cached_ids, cached_info = resolve_neuron_query(cache, "SMP540*")
    assert {str(value) for value in cached_ids} == {"1", "2"}
    assert cached_info["matched_column"] == "type"


def test_stale_sidecar_rebuilds_for_new_priority_fields(tmp_path):
    from src.neuron_index_builder import build_search_cache_frame, search_cache_path
    from src.neuron_search import get_cached_neuron_search, resolve_neuron_query

    dataset = "stale-sidecar:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    frame = pl.DataFrame({
        "bodyId": ["100"],
        "type": [""],
        "instance": [""],
        "flywireType": [""],
        "locationType": ["new-location-type"],
    })
    frame.write_parquet(index_path)
    # Simulate a sidecar produced before locationType was added to the
    # canonical *Type projection. It is readable, but semantically stale.
    build_search_cache_frame(
        frame, ["bodyId", "type", "instance", "flywireType"]
    ).write_parquet(search_cache_path(index_path))

    cache = get_cached_neuron_search(dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None
    assert cache.search_path is None
    ids, info = resolve_neuron_query(cache, "new-location-type")
    assert ids == ["100"]
    assert info["matched_column"] == "locationType"


def test_missing_cache_falls_back_to_authoritative_dataframe(tmp_path):
    """A missing parquet cache must not change lookup semantics or write files."""
    from src.neuron_search import (
        get_cached_neuron_search,
        resolve_cached_or_dataframe_query,
        resolve_dataframe_query,
    )

    dataset = "metadata-only:v1.0"
    frame = _write_search_cache(tmp_path)[1].to_pandas()
    index_root = tmp_path / "empty-indexes"

    assert get_cached_neuron_search(dataset, index_root=index_root) is None
    ids, info = resolve_cached_or_dataframe_query(
        None, frame, "MTe07"
    )
    expected, expected_info = resolve_dataframe_query(frame, "MTe07")

    assert {str(value) for value in ids} == {
        str(value) for value in expected
    }
    assert info["cache"] is False
    assert info["matched_column"] == expected_info["matched_column"]
    assert not index_root.exists()


def test_legacy_cache_missing_metadata_columns_falls_back(tmp_path):
    """Same IDs do not make a legacy, narrow index safe for new fields."""
    from src.neuron_index_builder import build_search_cache_frame
    from src.neuron_search import (
        get_cached_neuron_search,
        resolve_cached_or_dataframe_query,
    )

    dataset = "legacy-schema:v1.0"
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    legacy = pl.DataFrame({
        "bodyId": ["100"],
        "type": [""],
        "instance": [""],
    })
    legacy.write_parquet(index_path)
    build_search_cache_frame(legacy).write_parquet(
        cache_dir / "neuron_index_search.parquet"
    )

    source = pd.DataFrame({
        "bodyId": ["100"],
        "type": [""],
        "instance": [""],
        "flywireType": ["MTe07"],
    })
    cache = get_cached_neuron_search(dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None

    ids, info = resolve_cached_or_dataframe_query(cache, source, "MTe07")
    assert ids == ["100"]
    assert info["cache"] is False
    assert info["matched_column"] == "flywireType"


def test_fast_metadata_loader_does_not_use_stale_parquet(tmp_path):
    """The CSV pulled by the dataset pipeline remains authoritative."""
    import os

    import src.statvis as statvis

    csv = tmp_path / "sample_neuron_df.csv"
    parquet = tmp_path / "sample_neuron_df.parquet"
    pd.DataFrame({"bodyId": ["1"], "type": ["from-csv"]}).to_csv(
        csv, index=False
    )
    pl.DataFrame({"bodyId": ["1"], "type": ["from-parquet"]}).write_parquet(
        parquet
    )

    os.utime(parquet, ns=(1_000, 1_000))
    os.utime(csv, ns=(2_000, 2_000))
    assert statvis._load_dataframe_fast(str(csv))["type"].tolist() == [
        "from-csv"
    ]

    os.utime(parquet, ns=(3_000, 3_000))
    assert statvis._load_dataframe_fast(str(csv))["type"].tolist() == [
        "from-parquet"
    ]


def test_fast_loader_drops_serialized_index_column(tmp_path):
    """CSVs written with to_csv(index=True) must not leak their index."""
    import src.statvis as statvis

    csv = tmp_path / "indexed_neuron_df.csv"
    pd.DataFrame({"bodyId": ["1", "2"], "type": ["a", "b"]}).to_csv(
        csv, index=True
    )

    df = statvis._load_dataframe_fast(str(csv))
    assert list(df.columns) == ["bodyId", "type"]


def test_drop_leading_index_columns_variants():
    import src.statvis as statvis

    # pandas names an unnamed index 'Unnamed: 0'; polars may auto-name it.
    for name in ("", "Unnamed: 0", "column_1", "index"):
        df = pd.DataFrame({name: [0, 1], "bodyId": ["1", "2"]})
        assert list(statvis.drop_leading_index_columns(df).columns) == ["bodyId"]

    # Polars frames are supported too.
    pl_df = pl.DataFrame({"column_1": [0, 1], "bodyId": ["1", "2"]})
    assert statvis.drop_leading_index_columns(pl_df).columns == ["bodyId"]

    # An unnamed leading column holding real (non-integer) data is kept.
    df = pd.DataFrame({"": ["x", "y"], "bodyId": ["1", "2"]})
    assert list(statvis.drop_leading_index_columns(df).columns) == ["", "bodyId"]

    # Non-leading columns with index-like names are left alone.
    df = pd.DataFrame({"bodyId": ["1"], "index": [7]})
    assert list(statvis.drop_leading_index_columns(df).columns) == [
        "bodyId",
        "index",
    ]
