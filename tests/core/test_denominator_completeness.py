"""Ratio-denominator completeness (engine round 2026-10-05).

connection_ratio / connection_ratio_adj need the post's COMPLETE incoming
mass. The connection cache may be absent or incomplete; the two
``_fetch_total_incoming_weight*`` fetchers must auto-pull whatever is
missing (local release table first, then the dataset's API) and disclose
it — never silently serve a partial map. Online-only (``use_cache=False``)
FAFB keeps its CAVE doctrine (test_fafb_no_cache.py).
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET = "flywire_FAFB_v783"

# Synthetic local release: three post neurons, one background pre.
#   totals @1: 101=100, 102=37+12=49, 103=5+4=9 ; types: TA=101+103=109, TB=102=49
#   totals @5: identical except P3=5 (the 4-edge drops)
_EDGES = [
    ("9001", "101", 100),
    ("9001", "102", 37),
    ("9001", "103", 5),
    ("101", "102", 12),
    ("102", "103", 4),
]
_NEURONS = [
    ("101", "TA"), ("102", "TB"), ("103", "TA"), ("9001", "TB"),
]


def _write_release(root: Path, csv_conn: bool = False):
    ds = root / "datasets" / DATASET
    ds.mkdir(parents=True, exist_ok=True)
    if csv_conn:
        # Legacy pre-converter column names (pre_root_id / syn_count) —
        # the aggregate must adapt, not assume bodyId_pre/weight.
        pd.DataFrame({
            "pre_root_id": [e[0] for e in _EDGES],
            "post_root_id": [e[1] for e in _EDGES],
            "syn_count": [e[2] for e in _EDGES],
        }).to_csv(ds / f"{DATASET}_merged_connections.csv", index=False)
    else:
        pl.DataFrame({
            "bodyId_pre": [e[0] for e in _EDGES],
            "bodyId_post": [e[1] for e in _EDGES],
            "weight": [e[2] for e in _EDGES],
        }).write_parquet(ds / f"{DATASET}_merged_connections.parquet")
    pd.DataFrame(_NEURONS, columns=["bodyId", "type"]).to_csv(
        ds / f"{DATASET}_allneurons_neuron_df.csv", index=False)


def _make_finder(tmp_path: Path, *, cache_edges=None, cache_neurons=None,
                 csv_conn: bool = False):
    """Finder over the synthetic release with a configurable cache.

    ``cache_edges``/``cache_neurons`` None = no cache files at all;
    a list = write exactly those rows to the cache parquet/index
    (an INCOMPLETE cache when shorter than the release)."""
    from coana import FindNeuronConnection

    _write_release(tmp_path, csv_conn=csv_conn)
    cache_dir = tmp_path / "cache" / DATASET
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "connections.parquet"
    index_path = tmp_path / "neuron_indexes" / DATASET / "neuron_index.parquet"

    if cache_edges is not None:
        pl.DataFrame({
            "bodyId_pre": [e[0] for e in cache_edges],
            "bodyId_post": [e[1] for e in cache_edges],
            "weight": [e[2] for e in cache_edges],
        }).write_parquet(db_path)
    if cache_neurons is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            [{"bodyId": b, "type": t} for b, t in cache_neurons],
            schema={"bodyId": pl.Utf8, "type": pl.Utf8},
        ).write_parquet(index_path)

    finder = object.__new__(FindNeuronConnection)
    finder.use_cache = True
    finder.client_type = "flywire"
    finder.dataset = DATASET
    finder.cache_folder = str(cache_dir)
    finder.script_path = str(tmp_path)
    finder.verbose_mode = "silent"
    finder._conn_df_cache = None
    finder._connection_maps = {}
    finder._local_neuron_df_cache = {}
    finder._warn_notes = []
    finder._vprint = lambda *a, **k: None
    finder._get_connection_db_path = lambda: str(db_path)
    finder._get_neuron_index_path = lambda: str(index_path)
    finder._min_synapse_excluded = False
    finder.build_connection_cache = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("denominator lookup attempted to build a cache"))
    return finder


def _as_map(frame, key):
    if frame is None or len(frame) == 0:
        return {}
    return {str(k): float(v) for k, v in
            zip(frame[key], frame["total_incoming_weight"])}


def test_incomplete_cache_bodyid_map_is_completed_from_release(tmp_path):
    """The cache serves only P1; P2/P3 totals are auto-pulled from the
    local release table and merged into one complete map."""
    finder = _make_finder(
        tmp_path, cache_edges=[("9001", "101", 100)])
    result = finder._fetch_total_incoming_weight(["101", "102", "103"], 1)
    assert _as_map(result, "bodyId_post") == {"101": 100.0, "102": 49.0,
                                              "103": 9.0}
    assert any("ratio denominators" in n and "2 post neurons" in n
               for n in finder._warn_notes), finder._warn_notes


def test_incomplete_cache_type_map_is_completed_from_release(tmp_path):
    """The cache index knows only TA; TB's mass is pulled via the local
    neuron table members and merged."""
    finder = _make_finder(
        tmp_path,
        cache_edges=[("9001", "101", 100), ("102", "103", 4), ("9001", "103", 5)],
        cache_neurons=[("101", "TA"), ("103", "TA")],
    )
    result = finder._fetch_total_incoming_weight_by_type(["TA", "TB"], 1)
    assert _as_map(result, "type_post") == {"TA": 109.0, "TB": 49.0}
    assert any("ratio denominators" in n and "1 post types" in n
               for n in finder._warn_notes), finder._warn_notes


def test_absent_cache_serves_from_local_release_without_building(tmp_path):
    """No cache files at all: the complete release table serves every
    denominator (bodyId AND type), and the network cache build never
    runs (the stub raises if it does)."""
    finder = _make_finder(tmp_path)
    by_body = finder._fetch_total_incoming_weight(["101", "102", "103"], 1)
    assert _as_map(by_body, "bodyId_post") == {"101": 100.0, "102": 49.0,
                                               "103": 9.0}
    by_type = finder._fetch_total_incoming_weight_by_type(["TA", "TB"], 1)
    assert _as_map(by_type, "type_post") == {"TA": 109.0, "TB": 49.0}
    assert any("local release table" in n for n in finder._warn_notes)


def test_adjusted_ratio_denominator_conditioned_and_complete(tmp_path):
    """connection_ratio_adj at Min Synapse 5: the P3 denominator counts
    only edges >= 5 (5, not 9) — pulled from the release when the cache
    has no >=5 edge for P3 (partial cache with only the 4-edge)."""
    finder = _make_finder(tmp_path, cache_edges=[("102", "103", 4)])
    finder.min_synapse_num = 5
    frame = pl.DataFrame({
        "bodyId_pre": ["102"], "bodyId_post": ["103"], "weight": [4],
    })
    out = finder._attach_ratio_adj_columns(
        frame, "bodyId_pre", "bodyId_post").to_pandas()
    # implied conditioned denominator: weight / adj == 5 (complete map);
    # the incomplete-cache behavior would give 4 (the in-frame sum).
    assert 4 / out["connection_ratio_adj"].iloc[0] == pytest.approx(5.0)


def test_complete_cache_never_pulls(tmp_path):
    """A cache covering every requested post produces no pull and no
    disclosure note."""
    finder = _make_finder(
        tmp_path,
        cache_edges=list(_EDGES),
        cache_neurons=list(_NEURONS),
    )
    result = finder._fetch_total_incoming_weight(["101", "102"], 1)
    assert _as_map(result, "bodyId_post") == {"101": 100.0, "102": 49.0}
    assert not any("ratio denominators" in n for n in finder._warn_notes)


def test_zero_mass_post_is_confirmed_once_not_repulled(tmp_path):
    """A post with no incoming mass at the threshold is absent from every
    source: it is pulled once (memoized), never per call, and the result
    simply omits it."""
    finder = _make_finder(
        tmp_path, cache_edges=[("9001", "101", 100)])
    calls = []
    original = finder._local_incoming_totals_by_bodyid
    finder._local_incoming_totals_by_bodyid = (
        lambda posts, mw=1: calls.append(list(posts))
        or original(posts, mw))
    first = finder._fetch_total_incoming_weight(["101", "777"], 1)
    second = finder._fetch_total_incoming_weight(["101", "777"], 1)
    assert _as_map(first, "bodyId_post") == {"101": 100.0}
    assert _as_map(second, "bodyId_post") == {"101": 100.0}
    assert len(calls) == 1, calls   # PX pulled once, memoized as absent


def test_csv_release_column_names_are_adapted(tmp_path):
    """A legacy CSV release (pre_root_id/syn_count columns) feeds the same
    complete map as the parquet release (regression: the aggregate used
    to select bodyId_pre unconditionally and crash on CSV releases)."""
    finder = _make_finder(tmp_path, csv_conn=True)
    by_body = finder._fetch_total_incoming_weight(["101", "102", "103"], 1)
    assert _as_map(by_body, "bodyId_post") == {"101": 100.0, "102": 49.0,
                                               "103": 9.0}
    by_type = finder._fetch_total_incoming_weight_by_type(["TA", "TB"], 1)
    assert _as_map(by_type, "type_post") == {"TA": 109.0, "TB": 49.0}


# ---------------------------------------------------------------------------
# Cache-sourced profile completeness (profiler supplement)
# ---------------------------------------------------------------------------

def test_profiler_supplement_merges_incomplete_neurons(tmp_path, monkeypatch):
    """A cache-sourced bodyId profile for a neuron flagged
    downstream_complete=False is supplemented from the dataset API and
    dedup-merged; complete neurons and release-sourced datasets are
    untouched."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison.connectivity_profiler import (
        ConnectivityProfiler as CP, _PROFILER_CONN_CACHE)

    index_dir = tmp_path / "neuron_indexes" / "ds"
    index_dir.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["101", "102"],
                  "downstream_complete": [True, False]}).write_parquet(
        index_dir / "neuron_index.parquet")

    profiler = object.__new__(CP)
    profiler._index_flags_cache = {}
    profiler.neuron_index_dir = tmp_path / "neuron_indexes"
    profiler._log = lambda *a, **k: None

    saved_entry = _PROFILER_CONN_CACHE.get("ds")
    _PROFILER_CONN_CACHE["ds"] = {"conn_df": None, "source": "cache"}
    try:
        local_up = pd.DataFrame({
            "partner_bodyId": ["900"], "partner_type": ["TX"],
            "neuron_bodyId": ["102"], "weight": [4]})
        api_up = pd.DataFrame({
            "partner_bodyId": ["901", "900"], "partner_type": ["TY", "TX"],
            "neuron_bodyId": ["102", "102"], "weight": [6, 4]})
        calls = []
        monkeypatch.setattr(
            CP, "_query_connections_neuprint",
            lambda self, ids, ds: calls.append(list(ids))
            or (api_up, pd.DataFrame(columns=[
                "partner_bodyId", "partner_type", "neuron_bodyId",
                "weight"])))

        # flags resolve via the overridden index dir; but the SAFE name of
        # dataset "ds" is "ds" -> index at neuron_indexes/ds staged above
        up, down = profiler._supplement_incomplete_cache(
            102, local_up, pd.DataFrame(), "ds")
        assert calls == [['102']]
        assert sorted(up["partner_bodyId"].astype(str)) == ["900", "901"]
        assert down.empty

        # complete neuron: no API call
        up2, _ = profiler._supplement_incomplete_cache(
            101, local_up, pd.DataFrame(), "ds")
        assert calls == [['102']]
        assert len(up2) == 1

        # local-release dataset: never supplements (release is complete)
        up3, _ = profiler._supplement_incomplete_cache(
            102, local_up, pd.DataFrame(), "flywire_FAFB_v783")
        assert calls == [['102']]
        assert len(up3) == 1

        # release-sourced cache entry: no supplement either
        _PROFILER_CONN_CACHE["ds"]["source"] = "release"
        up4, _ = profiler._supplement_incomplete_cache(
            102, local_up, pd.DataFrame(), "ds")
        assert calls == [['102']]
        assert len(up4) == 1
    finally:
        if saved_entry is None:
            _PROFILER_CONN_CACHE.pop("ds", None)
        else:
            _PROFILER_CONN_CACHE["ds"] = saved_entry
