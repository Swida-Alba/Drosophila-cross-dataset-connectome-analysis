"""Regression tests for connection-cache completeness and no-cache runs."""

import sys
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_positive_completion_flag_without_rows_is_refetched():
    """A historical positive count must not hide a missing current edge set."""
    from coana import FindNeuronConnection

    fc = object.__new__(FindNeuronConnection)
    fc.use_cache = True
    fc._conn_db_pre_id_cache = None
    fc._conn_index = {"B": [0]}
    fc._neuron_index_dict = {
        "A": {"downstream_complete": True, "connection_count": 5},
        "B": {"downstream_complete": True, "connection_count": 1},
        "C": {"downstream_complete": True, "connection_count": 0},
    }
    fc._load_connection_db = lambda: pl.DataFrame({
        "bodyId_pre": ["B"],
        "bodyId_post": ["T"],
        "weight": [5],
    })
    fc._load_neuron_index = lambda: pd.DataFrame()
    fc._vprint = lambda *args, **kwargs: None

    cached, uncached, partially_cached = fc._query_connection_db(
        ["A", "B", "C"]
    )

    assert uncached == ["A"]
    assert partially_cached == []
    assert set(cached["bodyId_pre"].to_list()) == {"B"}


def test_use_cache_false_does_not_write_connection_or_state_files(
    tmp_path, monkeypatch, capsys
):
    """No-cache path keeps fetched data in memory and writes no cache files."""
    import coana
    import neuprint
    from coana import FindNeuronConnection

    monkeypatch.chdir(tmp_path)

    fc = object.__new__(FindNeuronConnection)
    fc.use_cache = False
    fc.cache_only = False
    fc.cache_folder = ""
    fc.dataset = "fake:v1"
    fc.client_type = "neuprint"
    fc.force_API_fetching = False
    fc.simple_fetch = False
    fc.kwargs_fetch = {}
    fc.filter_by = "bodyId"
    fc.exclude_intra_type_connections = False
    fc.label_mapper = None
    fc.min_synapse_num = 1
    fc.min_ratio = 0.0
    fc.min_traversal_probability = 0.0
    fc._vprint = lambda *args, **kwargs: None
    fc._ensure_neuprint_client = lambda: None
    fc._query_connection_db = lambda upstream_bodyIds, downstream_bodyIds=None: (
        pl.DataFrame(), list(upstream_bodyIds), []
    )
    fc._enrich_connections_with_neuron_info = lambda frame: frame.assign(
        type_pre="A",
        instance_pre="A_1",
        type_post="B",
        instance_post="B_1",
    )
    fc._apply_hemisphere_suffix_to_conn_df = lambda frame: frame
    fc._save_connections_only = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("no-cache run attempted to save connections")
    )
    fc._mark_neurons_as_cached = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("no-cache run attempted to save index state")
    )

    def retry(func, **kwargs):
        return func()

    monkeypatch.setattr(
        coana,
        "_get_api_retry_utils",
        # 4-tuple: api_call_with_retry, APITimeoutError,
        # APIRetryExhaustedError, APICancelError
        lambda: (retry, RuntimeError, RuntimeError, RuntimeError),
    )

    def fake_fetch_adjacencies(**kwargs):
        from tqdm import tqdm

        # Simulate neuprint's internal request progress.  The analysis should
        # expose only its own outer bar.
        for _ in tqdm(range(5), desc="nested fake"):
            pass
        return pd.DataFrame(), pd.DataFrame({
            "bodyId_pre": ["1"],
            "bodyId_post": ["2"],
            "weight": [4],
            "roi": ["fake"],
        })

    monkeypatch.setattr(neuprint, "fetch_adjacencies", fake_fetch_adjacencies)

    result = fc._fetch_connections_with_cache(
        upstream_bodyIds=["1"],
        downstream_bodyIds=None,
        min_weight=1,
        min_conn_ratio=0.0,
        min_traversal_prob=0.0,
    )

    assert len(result) == 1
    assert not (tmp_path / "connections.parquet").exists()
    assert not (tmp_path / "neuron_index_state.parquet").exists()
    output = capsys.readouterr()
    assert "nested fake" not in output.out + output.err
    assert "Pulling connections" in output.out + output.err


def test_resume_pull_refetches_complete_flags_missing_from_connection_cache(
    tmp_path, monkeypatch
):
    """Settings-tab resume must repair stale completion flags automatically."""
    import neuprint
    from coana import FindNeuronConnection

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(neuprint, "Client", FakeClient)
    monkeypatch.setattr(
        FindNeuronConnection, "_ensure_complete_dataset", lambda self: None
    )
    monkeypatch.setattr(
        FindNeuronConnection,
        "_ensure_neuron_index_from_metadata",
        lambda self: False,
    )

    dataset = "fake:stale"
    index_path = tmp_path / "neuron_indexes" / "fake_stale" / "neuron_index.parquet"
    index_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "bodyId": ["1"],
        "type": ["aMe26"],
        "instance": ["aMe26_L"],
        "post": [10],
        "downstream_complete": [True],
        "last_fetched": ["2026-08-17 21:00:00"],
        "connection_count": [5],
    }).write_parquet(index_path)

    fc = FindNeuronConnection(
        dataset=dataset,
        use_cache=True,
        cache_only=False,
        verbose=False,
        script_path=str(tmp_path),
    )
    fetched = []

    def fake_bulk(upstream_bodyIds, downstream_bodyIds=None, **kwargs):
        fetched.extend(str(value) for value in upstream_bodyIds)
        return pd.DataFrame({
            "bodyId_pre": [str(value) for value in upstream_bodyIds],
            "bodyId_post": ["2" for _ in upstream_bodyIds],
            "weight": [5 for _ in upstream_bodyIds],
            "roi": ["fake" for _ in upstream_bodyIds],
        })

    monkeypatch.setattr(fc, "_fetch_connections_bulk", fake_bulk)

    summary = fc.build_connection_cache(
        neuron_bodyIds=["1"],
        batch_size=1,
        quiet=True,
    )

    assert fetched == ["1"]
    assert summary["already_cached"] == 0
    assert summary["newly_cached"] == 1


def test_analysis_reloads_connection_snapshot_after_settings_pull(tmp_path):
    """An existing analysis object must see a newly replaced connections file."""
    import coana
    from coana import FindNeuronConnection

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    db_path = cache_dir / "connections.parquet"
    pl.DataFrame({
        "bodyId_pre": ["old"],
        "bodyId_post": ["target"],
        "weight": [1],
    }).write_parquet(db_path)

    fc = object.__new__(FindNeuronConnection)
    fc.use_cache = True
    fc.dataset = "fake:reload"
    fc.client_type = "neuprint"
    fc.cache_folder = str(cache_dir)
    fc._dataset_safe = "fake_reload"
    fc._conn_df_cache = None
    fc._conn_index = None
    fc._conn_index_post = None
    fc._conn_db_pre_id_cache = None
    fc._vprint = lambda *args, **kwargs: None

    try:
        first = fc._load_connection_db()
        assert first["bodyId_pre"].to_list() == ["old"]

        # Simulate the Settings pull's final atomic replacement.
        pl.DataFrame({
            "bodyId_pre": ["new"],
            "bodyId_post": ["target"],
            "weight": [2],
        }).write_parquet(db_path)

        second = fc._load_connection_db()
        assert second["bodyId_pre"].to_list() == ["new"]
    finally:
        coana._FNC_CACHE.pop("fake_reload", None)


def test_analysis_reloads_completion_state_sidecar_after_settings_pull(tmp_path):
    """Completion flags written by the pull must reach an existing analysis."""
    import coana
    from coana import FindNeuronConnection

    dataset = "fake:state-reload"
    fc = object.__new__(FindNeuronConnection)
    fc.use_cache = True
    fc.dataset = dataset
    fc.client_type = "neuprint"
    fc.script_path = str(tmp_path)
    fc.cache_folder = str(tmp_path / "cache" / "fake_state-reload")
    fc._dataset_safe = "fake_state-reload"
    fc._neuron_index_cache = None
    fc._neuron_index_dict = None
    fc._vprint = lambda *args, **kwargs: None

    index_path = Path(fc._get_neuron_index_path())
    index_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "bodyId": ["1"],
        "type": ["T"],
        "instance": ["T_L"],
        "post": [1],
        "downstream_complete": [False],
        "last_fetched": [""],
        "connection_count": [0],
    }).write_parquet(index_path)

    try:
        first = fc._load_neuron_index()
        assert not bool(first.loc[0, "downstream_complete"])

        state_path = Path(fc._get_neuron_index_state_path())
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "bodyId": ["1"],
            "downstream_complete": [True],
            "last_fetched": ["2026-08-18 12:00:00"],
            "connection_count": [7],
        }).write_parquet(state_path)

        second = fc._load_neuron_index()
        assert bool(second.loc[0, "downstream_complete"])
        assert int(second.loc[0, "connection_count"]) == 7
    finally:
        coana._FNC_CACHE.pop("fake_state-reload", None)


def test_namespace_audit_counts_foreign_bodyids(tmp_path):
    """I-5 (2026-09-28): connection rows whose bodyId is not in the
    dataset's own neuron index join to nothing — counted and disclosed
    (never rewritten), instead of passing both integrity layers silently."""
    from coana import FindNeuronConnection

    idx_dir = tmp_path / "neuron_indexes" / "dsX"
    idx_dir.mkdir(parents=True)
    pl.DataFrame({
        "bodyId": [1, 2],
        "downstream_complete": [True, False],
        "connection_count": [2, 0],
    }).write_parquet(idx_dir / "neuron_index.parquet")
    cache_dir = tmp_path / "cache" / "dsX"
    cache_dir.mkdir(parents=True)
    pl.DataFrame({
        "bodyId_pre": [1, 1, 99],
        "bodyId_post": [2, 88, 2],
        "weight": [1, 1, 5],
    }).write_parquet(cache_dir / "connections.parquet")

    fc = object.__new__(FindNeuronConnection)
    fc.dataset = "dsX"
    fc._dataset_safe = "dsX"
    fc.script_path = str(tmp_path)
    prints = []
    fc._vprint = lambda message, level="full", **kw: prints.append(message)

    result = fc._compute_cache_integrity()
    assert result["distinct_connections"] == 3
    assert result["foreign_id_rows"] == 2  # (1, 88) and (99, 2)
    assert result["foreign_id_rows_pre"] == 1
    assert result["foreign_id_rows_post"] == 1

    # the disclosure is surfaced, not just computed
    fc._check_cache_coverage()
    assert any("namespace check" in p for p in prints), prints
