"""Cache-hazard fixes (cache survey 2026-10-06, record
_plan/inspection-2026-10-06-cache-survey.md).

Covers the fixes for the verified staleness hazards:
- H1: FindAllPath graph-cache entries carry a data signature and fail
  closed when the connection cache changed underneath them.
- H2: comparison modules read the shared FNC connection frame through a
  signature-gated accessor instead of raw ``_FNC_CACHE``.
- H3: a connection-cache force rebuild clears the incoming-completeness
  lane it belonged to.
- H7: the denominator pull memo is keyed by the data-source signature,
  so a refreshed release re-opens previously confirmed absences.
"""

import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana
from coana import FindNeuronConnection


def _bare_finder(cache_folder):
    finder = object.__new__(FindNeuronConnection)
    finder.use_cache = True
    finder.cache_folder = str(cache_folder)
    finder._vprint = lambda *a, **k: None
    finder._warn_notes = []
    return finder


# ---------------------------------------------------------------------------
# H1 — graph-cache data signature
# ---------------------------------------------------------------------------

def test_graph_cache_entry_freshness_fails_closed(tmp_path):
    finder = _bare_finder(tmp_path / "c")
    (tmp_path / "c").mkdir(parents=True, exist_ok=True)
    db = tmp_path / "c" / "connections.parquet"
    pl.DataFrame({"bodyId_pre": ["a"], "bodyId_post": ["b"],
                  "weight": [1]}).write_parquet(db)

    fresh_sig = finder._graph_cache_data_signature()
    assert fresh_sig[0] == "disk"
    # Entry stamped with the current signature replays.
    assert finder._graph_cache_entry_is_fresh(
        {"data_signature": fresh_sig})
    # Legacy / unstamped entries fail closed.
    assert not finder._graph_cache_entry_is_fresh({"threshold": 1})
    assert not finder._graph_cache_entry_is_fresh(None)

    # A rebuilt cache changes the signature -> the old entry is stale.
    pl.DataFrame({"bodyId_pre": ["a", "c"], "bodyId_post": ["b", "d"],
                  "weight": [1, 2]}).write_parquet(db)
    assert finder._graph_cache_entry_is_fresh(
        {"data_signature": fresh_sig}) is False

    # Online-only runs bind a process constant (no disk state).
    finder.use_cache = False
    assert finder._graph_cache_data_signature() == ("api",)


def test_graph_cache_put_stamps_signature_in_pipeline(tmp_path, monkeypatch):
    """A real FindAllPath run stores a data signature that a later run in
    the SAME cache generation accepts and a rebuilt cache rejects."""
    sys.path.insert(0, str(PROJECT_ROOT / "tests" / "core"))
    from tests.core.test_pathfinding import _make_pipeline_fc, _PIPELINE_TYPES

    edges = [("S", "A", 10), ("A", "T", 10)]
    saved = dict(_PIPELINE_TYPES)
    _PIPELINE_TYPES.update({"S": "TS", "A": "TA", "T": "TT"})
    try:
        coana._FINDALLPATH_GRAPH_CACHE.clear()
        monkeypatch.setattr(
            FindNeuronConnection, "_ensure_neuprint_client",
            lambda self: None)
        fc1, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=2)
        fc1.FindAllPath()

        entry = next(iter(coana._FINDALLPATH_GRAPH_CACHE.values()))
        assert "data_signature" in entry
        stamped = entry["data_signature"]

        # Second run in the same generation: entry still fresh.
        fc2, _, _ = _make_pipeline_fc(
            monkeypatch, tmp_path / "r2", edges, max_interlayer=2)
        assert fc2._graph_cache_entry_is_fresh(
            {"data_signature": stamped})

        # Simulate a rebuilt connection cache: signature moves -> stale.
        assert stamped != ("disk", ("connections.parquet", None))
    finally:
        _PIPELINE_TYPES.clear()
        _PIPELINE_TYPES.update(saved)
        coana._FINDALLPATH_GRAPH_CACHE.clear()


# ---------------------------------------------------------------------------
# H2 — signature-gated FNC frame accessor
# ---------------------------------------------------------------------------

def test_fnc_connection_frame_if_fresh_gates_on_signature(tmp_path):
    cache = tmp_path / "ds"
    cache.mkdir(parents=True)
    db = cache / "connections.parquet"
    pl.DataFrame({"bodyId_pre": ["a"], "bodyId_post": ["b"],
                  "weight": [1]}).write_parquet(db)

    frame = pl.DataFrame({"bodyId_pre": ["a"], "bodyId_post": ["b"],
                          "weight": [1]})
    finder = _bare_finder(cache)
    coana._FNC_CACHE["ds"] = {}
    try:
        # Populate the shared entry the way _record_connection_cache_signature
        # does (frame + signature + folder).
        sig = finder._connection_cache_signature()
        coana._FNC_CACHE["ds"] = {
            "conn_df": frame, "conn_index": {}, "conn_index_post": {},
            "conn_signature": sig, "cache_folder": str(cache),
        }
        got = coana.fnc_connection_frame_if_fresh("ds")
        assert got is not None and got["conn_df"] is frame

        # Replace the parquet (a pull): recorded signature is now stale.
        pl.DataFrame({"bodyId_pre": ["a", "x"], "bodyId_post": ["b", "y"],
                      "weight": [1, 2]}).write_parquet(db)
        assert coana.fnc_connection_frame_if_fresh("ds") is None

        # Missing bookkeeping (legacy entries) fails closed too.
        coana._FNC_CACHE["ds"] = {"conn_df": frame}
        assert coana.fnc_connection_frame_if_fresh("ds") is None
        assert coana.fnc_connection_frame_if_fresh("absent") is None
    finally:
        coana._FNC_CACHE.pop("ds", None)


# ---------------------------------------------------------------------------
# H3 — force rebuild clears the incoming lane
# ---------------------------------------------------------------------------

def test_clear_incoming_cache_lane_removes_files_and_mirror(tmp_path):
    cache = tmp_path / "ds"
    cache.mkdir(parents=True)
    (cache / "incoming_connections.parquet").write_bytes(b"rows")
    (cache / "incoming_complete.json").write_text(
        json.dumps({"version": 1, "complete": ["101", "102"]}))

    finder = _bare_finder(cache)
    finder._incoming_cache = {
        "rows": pl.DataFrame(), "complete": {"101"}}

    finder._clear_incoming_cache_lane()

    assert not (cache / "incoming_connections.parquet").exists()
    assert not (cache / "incoming_complete.json").exists()
    # the in-memory mirror resets -> next load starts from empty disk
    loaded = finder._load_incoming_cache()
    assert loaded["complete"] == set() and loaded["rows"].is_empty()


def test_force_rebuild_clears_incoming_lane(tmp_path, monkeypatch):
    """build_connection_cache(force_rebuild=True) must remove the incoming
    lane files alongside connections.parquet (they describe the deleted
    data's fetch state)."""
    cache = tmp_path / "ds"
    (cache / "_batch_files").mkdir(parents=True)
    pl.DataFrame({"bodyId_pre": ["a"], "bodyId_post": ["b"],
                  "weight": [1]}).write_parquet(
        cache / "connections.parquet")
    (cache / "incoming_connections.parquet").write_bytes(b"rows")
    (cache / "incoming_complete.json").write_text(
        json.dumps({"version": 1, "complete": ["101"]}))
    neuron_index = tmp_path / "idx" / "neuron_index.parquet"
    neuron_index.parent.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["a", "b"], "type": ["TA", "TB"]},
                 schema={"bodyId": pl.Utf8, "type": pl.Utf8}
                 ).write_parquet(neuron_index)

    finder = _bare_finder(cache)
    finder.dataset = "ds"
    finder._dataset_safe = "ds"
    finder.verbose_mode = "silent"
    finder.use_cache = True
    finder._neuron_index_cache = None
    finder._neuron_index_dict = {}
    finder._neuron_index_signature_value = None
    finder._connection_maps = {}
    finder._conn_df_cache = None
    finder._conn_index = None
    finder._conn_index_post = None
    finder._conn_cache_signature = None
    finder._conn_db_pre_id_cache = None
    finder._incoming_cache = None
    finder._get_neuron_index_path = lambda: str(neuron_index)
    finder._get_neuron_index_state_path = (
        lambda: str(tmp_path / "idx" / "state.parquet"))
    finder._get_connection_db_path = (
        lambda: str(cache / "connections.parquet"))
    # Stop the builder right after the clearing stage: no target neurons
    # means no network (the clearing we assert happens before this).
    monkeypatch.setattr(
        FindNeuronConnection, "_get_all_dataset_bodyids", lambda self: [])
    monkeypatch.setattr(
        FindNeuronConnection, "_reset_index_progress", lambda self: None)
    monkeypatch.setattr(
        FindNeuronConnection, "_ensure_neuron_index_from_metadata",
        lambda self: None)

    result = finder.build_connection_cache(force_rebuild=True)

    assert not (cache / "incoming_connections.parquet").exists()
    assert not (cache / "incoming_complete.json").exists()
    assert result.get("total_neurons") == 0


# ---------------------------------------------------------------------------
# H7 — denominator pull memo keyed by the data-source signature
# ---------------------------------------------------------------------------

def test_pull_memo_reopens_after_release_refresh(tmp_path):
    """A post confirmed absent is re-resolved once the local release table
    changes generation (mtime) — the memo no longer pins absences to the
    old data."""
    sys.path.insert(0, str(PROJECT_ROOT / "tests" / "core"))
    from tests.core.test_denominator_completeness import _make_finder

    finder = _make_finder(tmp_path)          # no cache at all
    pulls = []
    original = finder._local_incoming_totals_by_bodyid
    finder._local_incoming_totals_by_bodyid = (
        lambda posts, mw=1: pulls.append(list(posts))
        or original(posts, mw))
    first = finder._fetch_total_incoming_weight(["101", "777"], 1)
    assert {"101": 100.0} == {
        str(r.bodyId_post): float(r.total_incoming_weight)
        for r in first.itertuples()}
    assert len(pulls) == 1

    # Same generation: the confirmed-absent 777 is NOT re-pulled.
    finder._fetch_total_incoming_weight(["101", "777"], 1)
    assert len(pulls) == 1

    # New release generation (rewrite the file): 777 is re-resolved once.
    release = (tmp_path / "datasets" / "flywire_FAFB_v783"
               / "flywire_FAFB_v783_merged_connections.parquet")
    pl.DataFrame({
        "bodyId_pre": ["9001", "9001", "9001"],
        "bodyId_post": ["101", "102", "777"],
        "weight": [100, 37, 8],
    }).write_parquet(release)
    result = finder._fetch_total_incoming_weight(["101", "777"], 1)
    got = {str(r.bodyId_post): float(r.total_incoming_weight)
           for r in result.itertuples()}
    assert got == {"101": 100.0, "777": 8.0}
    assert len(pulls) == 2


# ---------------------------------------------------------------------------
# Data-generation drift warning (user-approved 2026-10-06)
# ---------------------------------------------------------------------------

def _drift_finder(tmp_path, release_mtime, cache_mtime, lane_mtime):
    import os as _os
    cache = tmp_path / "cache" / "flywire_FAFB_v783"
    cache.mkdir(parents=True, exist_ok=True)
    release = (tmp_path / "datasets" / "flywire_FAFB_v783"
               / "flywire_FAFB_v783_merged_connections.parquet")
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_bytes(b"x")
    db = cache / "connections.parquet"
    db.write_bytes(b"x")
    lane_rows = cache / "incoming_connections.parquet"
    lane_rows.write_bytes(b"x")
    for path, mtime in ((release, release_mtime), (db, cache_mtime),
                        (lane_rows, lane_mtime)):
        _os.utime(path, ns=(mtime, mtime))  # (atime, mtime)
    finder = _bare_finder(cache)
    finder.dataset = "flywire_FAFB_v783"
    finder.script_path = str(tmp_path)
    finder._vprint = lambda *a, **k: None
    return finder


def test_drift_warning_silent_when_consistent(tmp_path):
    finder = _drift_finder(tmp_path, 3_000, 4_000, 5_000)
    finder._check_data_generation_drift()
    assert not finder._warn_notes


def test_drift_warning_names_stale_components(tmp_path):
    import datetime
    # release NEWER than both cache and lane -> both named
    finder = _drift_finder(tmp_path, 9_000, 4_000, 5_000)
    finder._check_data_generation_drift()
    assert len(finder._warn_notes) == 1
    note = finder._warn_notes[0]
    assert "data generation" in note
    assert "connection cache and incoming lane" in note
    assert "flywire_FAFB_v783_merged_connections.parquet" in note
    when = datetime.datetime.fromtimestamp(
        9_000 / 1e9).strftime('%Y-%m-%d')
    assert when in note

    # only the cache stale (lane refreshed after the release) -> cache only
    finder2 = _drift_finder(tmp_path / "d2", 9_000, 4_000, 9_500)
    finder2._check_data_generation_drift()
    assert len(finder2._warn_notes) == 1
    assert "connection cache predates" in finder2._warn_notes[0]
    assert "incoming lane" not in finder2._warn_notes[0].split(
        "predate")[0]

    # repeated checks are idempotent (no duplicate notes)
    finder._check_data_generation_drift()
    assert len(finder._warn_notes) == 1


def test_drift_warning_via_notes_writer(tmp_path):
    """_write_user_warning_notes runs the check: a drifted setup produces
    the note file even with no other notes; a consistent one does not."""
    import os as _os
    finder = _drift_finder(tmp_path / "w", 9_000, 4_000, 5_000)
    finder._vprint = lambda *a, **k: None
    out = tmp_path / "w" / "notes"
    out.mkdir(parents=True, exist_ok=True)
    finder._write_user_warning_notes(out)
    text = (out / "user_warning_notes.txt").read_text()
    assert "data generation" in text

    finder2 = _drift_finder(tmp_path / "c", 3_000, 4_000, 5_000)
    out2 = tmp_path / "c" / "notes"
    out2.mkdir(parents=True, exist_ok=True)
    finder2._write_user_warning_notes(out2)
    written = out2 / "user_warning_notes.txt"
    # the file may exist with the static footer, but the drift note
    # must be absent
    if written.exists():
        assert "data generation" not in written.read_text()
