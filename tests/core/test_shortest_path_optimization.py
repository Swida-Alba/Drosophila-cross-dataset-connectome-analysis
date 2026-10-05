"""Plan Section 5 (2026-09-07 audit) shortest-path fixation tests.

Covers the audit's §5.8 acceptance items that the fixes implement:

1. Per-pair correctness: same target, sources at different hop distances —
   both pairs' minimum-hop paths (plus tied alternatives) are returned.
2. Depth cap: a farther source at or below its own valid shortest distance
   stays included; paths beyond the cap stay excluded.
3. Direction reuse: seeding the enumerator with the discovery distances
   yields the identical output set as its own fresh reverse BFS.
4. Shared frontier: two targets with overlapping reverse frontiers produce
   the same path set as independent single-target searches.
5. Incoming-cache completeness: reverse discovery fetches online only for
   posts that are not proven incoming-complete, persists the fetched rows
   with a completeness marker, and a repeated run re-queries nothing.
6. Post-filter removal: the pipeline verifies (never re-filters) the
   per-pair minimum-hop invariant.
"""

import json
import os

import pandas as pd
import polars as pl
import pytest

import coana
from coana import FindNeuronConnection
from vispath_pkg.fast_graph_core import FastGraph

from tests.core.test_pathfinding import _make_pipeline_fc


def _bodyid_paths(fc):
    paths_csv = os.path.join(
        fc.allpath_folder, "src_to_tgt_allpaths_bodyId_paths.csv")
    paths = pl.read_csv(paths_csv)
    return {
        str(row[0]) if len(row) == 1 else " ".join(str(c) for c in row)
        for row in paths.iter_rows()
    }


def _path_column(fc):
    paths_csv = os.path.join(
        fc.allpath_folder, "src_to_tgt_allpaths_bodyId_paths.csv")
    return [str(p) for p in pl.read_csv(paths_csv)["path"].to_list()]


def _pipeline(monkeypatch, tmp_path, edges, **kwargs):
    """Offline shortest-pipeline builder with bodyId output enabled (the
    materialization's interlayer-info fetch is kept offline like the
    existing TestFindShortestPathPipeline tests)."""
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    monkeypatch.setattr(
        coana.FindNeuronConnection, "_ensure_neuprint_client",
        lambda self: None)
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, edges, **kwargs)
    fc.skip_bodyId = False
    return fc, fetch_calls, logs


# ---------------------------------------------------------------------------
# 1 + 2. Per-pair correctness and the depth cap
# ---------------------------------------------------------------------------

_PAIR_EDGES = [
    ("S1", "A", 10), ("A", "T", 10),                    # S1->T in 2 hops
    ("S2", "B", 10), ("B", "C", 10), ("C", "T", 10),    # S2->T in 3 hops
]

_CHAIN_SIMPLE = [("S", "A", 10), ("A", "T", 10)]


def test_per_pair_contract_returns_both_distances(monkeypatch, tmp_path):
    fc, _, _ = _pipeline(
        monkeypatch, tmp_path, _PAIR_EDGES, max_interlayer=99,
        source_ids=("S1", "S2"))
    fc.FindShortestPath()

    paths = _path_column(fc)
    assert any(p.startswith("S1->A->T") for p in paths), paths
    assert any(p.startswith("S2->B->C->T") for p in paths), paths
    # discovery diagnostics (audit fixation 8) are exported to metadata
    attrs = json.loads(open(os.path.join(
        fc.allpath_folder, "all_attributes.json"),
        encoding="utf-8").read())
    diag = attrs["shortest_discovery_diagnostics"]
    assert diag["targets_found"] == 1
    assert diag["dag_edges"] >= 4
    assert diag["per_target_distance_states"]["T"] >= 4


def test_per_pair_contract_keeps_tied_alternatives(monkeypatch, tmp_path):
    edges = _PAIR_EDGES + [
        ("S1", "X1", 10), ("X1", "T", 10),   # tied 2-hop alternative for S1
    ]
    fc, _, _ = _pipeline(
        monkeypatch, tmp_path, edges, max_interlayer=99,
        source_ids=("S1", "S2"))
    fc.FindShortestPath()

    paths = _path_column(fc)
    assert sum(1 for p in paths if p.startswith("S1->") and p.endswith("T")
               and p.count("->") == 2) == 2, paths
    assert any(p.startswith("S2->B->C->T") for p in paths), paths


def test_depth_cap_keeps_farther_source_within_its_own_distance(
        monkeypatch, tmp_path):
    """With a 3-edge cap (2 interlayers) both pairs exist; with a 2-edge
    cap (1 interlayer) the 3-edge pair is out of scope while the 2-edge
    pair remains."""
    fc3, _, _ = _pipeline(
        monkeypatch, tmp_path, _PAIR_EDGES, max_interlayer=3,
        source_ids=("S1", "S2"))
    fc3.FindShortestPath()
    paths3 = _path_column(fc3)
    assert any(p.startswith("S1->A->T") for p in paths3)
    assert any(p.startswith("S2->B->C->T") for p in paths3)

    fc2, _, _ = _pipeline(
        monkeypatch, tmp_path, _PAIR_EDGES, max_interlayer=1,
        source_ids=("S1", "S2"))
    fc2.FindShortestPath()
    paths2 = _path_column(fc2)
    assert any(p.startswith("S1->A->T") for p in paths2)
    assert not any(p.startswith("S2->") for p in paths2)


# ---------------------------------------------------------------------------
# 3. Enumerator distance-seeding parity (fresh BFS vs discovery-seeded)
# ---------------------------------------------------------------------------

def test_enumerator_seeded_distances_match_fresh_bfs():
    G = FastGraph()
    edges = [
        ("T", "C", 4), ("C", "B", 9), ("B", "S2", 7),      # 3-hop, bn 4
        ("T", "A", 8), ("A", "S1", 5),                     # 2-hop, bn 5
        ("T", "A", 8), ("A", "S3", 2),                     # 2-hop, bn 2
    ]
    for u, v, w in edges:
        G.add_edge(u, v, w)

    fresh_stats, seeded_stats = {}, {}
    fresh = list(G.find_paths_shortest_strongest_first(
        ["T"], ["S1", "S2", "S3"], 4, budget=1000000, stats=fresh_stats))
    # discovery BFS on the same graph gives each source's own distance
    seeded_distances = {"T": {"T": 0, "C": 1, "B": 2, "S2": 3,
                              "A": 1, "S1": 2, "S3": 2}}
    seeded = list(G.find_paths_shortest_strongest_first(
        ["T"], ["S1", "S2", "S3"], 4, budget=1000000, stats=seeded_stats,
        target_distances=seeded_distances))

    assert sorted(map(tuple, fresh)) == sorted(map(tuple, seeded))
    assert fresh_stats["tau"] == seeded_stats["tau"]
    assert fresh_stats["per_target"] == seeded_stats["per_target"]
    # strength order is preserved with seeded distances too
    assert [p for _, p in
            sorted(((min(G.adj[u][v] for u, v in zip(p, p[1:])), p)
                    for p in seeded), key=lambda item: -item[0])] == \
        [list(p) for p in seeded]


# ---------------------------------------------------------------------------
# 4. Shared frontier parity (overlapping targets)
# ---------------------------------------------------------------------------

def test_shared_frontier_matches_independent_searches(monkeypatch, tmp_path):
    """T1 and T2 share the intermediate C <- B <- S cone; the combined run
    must produce the same per-pair path set as two single-target runs."""
    edges = [
        ("S", "B", 10), ("B", "C", 10),
        ("C", "T1", 10), ("C", "T2", 10),
    ]

    def run(target_ids, label):
        fc, _, _ = _pipeline(
            monkeypatch, tmp_path / label, edges, max_interlayer=99,
            target_ids=target_ids)
        fc.FindShortestPath()
        return set(_path_column(fc))

    combined = run(("T1", "T2"), "combined")
    single1 = run(("T1",), "t1")
    single2 = run(("T2",), "t2")
    assert combined == single1 | single2


# ---------------------------------------------------------------------------
# 5. Incoming-direction cache completeness (F-PERF-03 / F-PERF-04)
# ---------------------------------------------------------------------------

def test_incoming_cache_fetches_once_then_reuses(monkeypatch, tmp_path):
    """Direction-aware cache (audit F-PERF-03/F-PERF-04 fix): rows are
    fetched online once, persisted with a completeness marker, and a
    repeated query for the same posts re-queries nothing — even when the
    SOURCE side is the only thing the old cache could prove complete."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    fc = object.__new__(FindNeuronConnection)
    fc.dataset = "test:v1"
    fc.use_cache = True
    fc.cache_only = False
    fc.cache_folder = str(cache_root)
    fc.min_ratio = 0.0
    fc.min_traversal_probability = 0.0
    fc.drop_untyped = True
    fc.label_mapper = None
    fc.verbose_mode = "silent"
    fc._warn_notes = []
    fc._vprint = lambda *a, **k: None
    fc._reset_untyped_drop_tracking()
    fc._conn_index_post = {}
    # empty source-oriented cache (S is NOT even cached here: the old
    # implementation's source-completeness gate is gone entirely)
    monkeypatch.setattr(
        FindNeuronConnection, "_load_connection_db",
        lambda self, force_reload=False: pl.DataFrame(
            schema={"bodyId_pre": pl.Utf8, "bodyId_post": pl.Utf8,
                    "weight": pl.Int64, "roi": pl.Utf8}))
    monkeypatch.setattr(
        FindNeuronConnection, "_build_conn_index", lambda self: None)
    monkeypatch.setattr(
        FindNeuronConnection, "_finalize_path_connection_frame",
        lambda self, combined: combined)

    online_calls = []

    def fake_online(posts):
        online_calls.append(sorted(posts))
        rows = []
        for post in posts:
            for pre, post_, w in _CHAIN_SIMPLE:
                if str(post_) == str(post):
                    rows.append({"bodyId_pre": pre, "bodyId_post": post,
                                 "weight": w, "roi": "WholeBrain"})
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        FindNeuronConnection, "_fetch_incoming_connections_online",
        staticmethod(fake_online))

    out1 = fc._fetch_path_connections_backward(["T"])
    assert online_calls == [["T"]]
    assert len(out1) == 1  # S->A->T chain: A->T incoming row
    assert os.path.exists(cache_root / "incoming_connections.parquet")
    complete = json.loads(
        (cache_root / "incoming_complete.json").read_text())["complete"]
    assert complete == ["T"]

    # Repeated query: T is incoming-complete -> no online call.
    out2 = fc._fetch_path_connections_backward(["T"])
    assert online_calls == [["T"]]
    assert len(out2) == 1
    stats = fc._shortest_cache_stats
    assert stats["posts_cache_complete"] == 1
    assert stats["posts_online"] == 1  # only the very first fetch

    # A NEW post is fetched individually; complete posts are not re-queried.
    fc._fetch_path_connections_backward(["T", "A"])
    assert online_calls == [["T"], ["A"]]
    fc._fetch_path_connections_backward(["T", "A"])
    assert online_calls == [["T"], ["A"]]
    # ...and the persisted rows serve it
    assert len(fc._fetch_path_connections_backward(["A"])) == 1


def test_incoming_cache_missing_post_is_refetched(monkeypatch, tmp_path):
    """A new frontier post not covered by the completeness markers is
    fetched individually; already-complete posts are not re-queried."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    fc = object.__new__(FindNeuronConnection)
    fc.dataset = "test:v1"
    fc.use_cache = True
    fc.cache_only = False
    fc.cache_folder = str(cache_root)
    fc.min_ratio = 0.0
    fc.min_traversal_probability = 0.0
    fc.drop_untyped = True
    fc.label_mapper = None
    fc.verbose_mode = "silent"
    fc._warn_notes = []
    fc._vprint = lambda *a, **k: None
    fc._reset_untyped_drop_tracking()
    fc._conn_index_post = {}
    monkeypatch.setattr(
        FindNeuronConnection, "_load_connection_db",
        lambda self, force_reload=False: pl.DataFrame(
            schema={"bodyId_pre": pl.Utf8, "bodyId_post": pl.Utf8,
                    "weight": pl.Int64, "roi": pl.Utf8}))
    monkeypatch.setattr(
        FindNeuronConnection, "_build_conn_index", lambda self: None)
    monkeypatch.setattr(
        FindNeuronConnection, "_finalize_path_connection_frame",
        lambda self, combined: combined)
    # pre-seed the cache: T is already incoming-complete
    fc._load_incoming_cache()["complete"] |= {"T"}

    online_calls = []

    def fake_online(posts):
        online_calls.append(sorted(posts))
        return pd.DataFrame([
            {"bodyId_pre": "A", "bodyId_post": "T", "weight": 10,
             "roi": "WholeBrain"},
        ])

    monkeypatch.setattr(
        FindNeuronConnection, "_fetch_incoming_connections_online",
        staticmethod(fake_online))
    fc._fetch_path_connections_backward(["T", "A"])
    # T was proven complete; only A may be queried online.
    assert online_calls == [["A"]]


# ---------------------------------------------------------------------------
# 6. Post-filter removal: verify-only, never re-filtered
# ---------------------------------------------------------------------------

def test_pipeline_no_longer_second_filters(monkeypatch, tmp_path, capsys):
    fc, _, logs = _pipeline(
        monkeypatch, tmp_path, _PAIR_EDGES, max_interlayer=99,
        source_ids=("S1", "S2"))
    fc.FindShortestPath()

    # the old 'Shortest bodyId filter: kept X of Y' pass is gone...
    assert not any("Shortest bodyId filter" in m for m in logs)
    # ...and the invariant verifier reported no violation
    assert not any("shortest invariant violation" in m
                   for m in fc._warn_notes)
    # a genuine violation WOULD be reported without touching the output
    fc2 = object.__new__(FindNeuronConnection)
    fc2._warn_notes = []
    fc2._vprint = lambda *a, **k: None
    # pair (S, T) has minimum hop count 2; the 3-hop detour violates it
    fc2._verify_shortest_bodyid_paths([["S", "A", "T"], ["S", "B", "C", "T"]])
    assert any("shortest invariant violation" in n
               for n in fc2._warn_notes)


# ---------------------------------------------------------------------------
# 7. Engine issue 2026-10-05 (I2): one type, one ratio denominator per run
# ---------------------------------------------------------------------------

_DENOM_EDGES = [
    # S->A2->T1 (2 hops) and S->A->X->T2 (3 hops): the (TS, TA) type pair
    # realizes at TWO conn layers with DIFFERENT masses (9 at the nearer
    # depth, 7 at the farther) — the audited aMe3->s-CPDN3A split, where
    # the type-path per-hop Ratio showed the MEAN over the pair's layer
    # rows while connection_type.csv showed the single-type-mass value.
    ("S", "A2", 9), ("A2", "T1", 12),
    ("S", "A", 7), ("A", "X", 4), ("X", "T2", 5),
    # never-reachable background mass: belongs to the type denominators
    # (full-membership incoming mass) but never to any path
    ("B1", "T1", 100), ("B1", "T2", 37), ("B1", "A2", 50),
    ("B1", "A", 11), ("B1", "X", 13),
]

# dataset-wide full-membership incoming masses implied by _DENOM_EDGES
_DENOM_TYPE_MASS = {"TA": 77.0, "TX": 17.0, "TT": 154.0}
_DENOM_NEURON_TOTAL = {
    "A": 18.0, "A2": 59.0, "X": 17.0, "T1": 112.0, "T2": 42.0,
}


@pytest.fixture()
def denom_types():
    """Type the _DENOM_EDGES nodes for the shared pipeline fixture."""
    from tests.core.test_pathfinding import _PIPELINE_TYPES
    saved = dict(_PIPELINE_TYPES)
    _PIPELINE_TYPES.update({
        "S": "TS", "A": "TA", "A2": "TA", "X": "TX",
        "T1": "TT", "T2": "TT", "B1": "TB"})
    yield _PIPELINE_TYPES
    _PIPELINE_TYPES.clear()
    _PIPELINE_TYPES.update(saved)


def test_type_path_ratios_share_conn_table_denominator(
        monkeypatch, tmp_path, denom_types):
    """The type-path per-hop Ratio and connection_type.csv must imply the
    SAME per-type mass. The realized readout (hop mass / the post type's
    full-membership incoming mass) applies on the synapse basis too — the
    old join value was the MEAN over the pair's per-layer rows (here
    mean(9/77, 7/77) = 8/77 for a hop that realizes 9/77)."""
    fc, _, _ = _pipeline(
        monkeypatch, tmp_path, _DENOM_EDGES, max_interlayer=3,
        source_ids=("S",), target_ids=("T1", "T2"))

    # Real denominators: dataset-wide type masses + per-neuron totals
    # (the same F9 fetches the production conn tables use).
    monkeypatch.setattr(
        FindNeuronConnection, "_fetch_total_incoming_weight_by_type",
        lambda self, types, *a, **k: pd.DataFrame({
            "type_post": [str(t) for t in types],
            "total_incoming_weight": [
                _DENOM_TYPE_MASS.get(str(t), 0.0) for t in types]}))
    monkeypatch.setattr(
        FindNeuronConnection, "_fetch_total_incoming_weight",
        lambda self, posts, *a, **k: pd.DataFrame({
            "bodyId_post": [str(p) for p in posts],
            "total_incoming_weight": [
                _DENOM_NEURON_TOTAL.get(str(p), 0.0) for p in posts]}))

    # Enrichment with REAL per-row ratios: weight / the post type's mass
    # (what EnrichConnectionTable computes in production from the same
    # fetch, instead of the shared harness's constant 0.5).
    def enrich(conn, **kwargs):
        def _attach_ratio(frame, post_col):
            posts = [str(v) for v in frame[post_col].to_list()]
            weights = [float(v) for v in frame["weight"].to_list()]
            return frame.with_columns(
                pl.Series("connection_ratio",
                          [(w / _DENOM_TYPE_MASS[p])
                           if p in _DENOM_TYPE_MASS else None
                           for w, p in zip(weights, posts)],
                          dtype=pl.Float64))

        conn_e = _attach_ratio(conn, "type_post").with_columns(
            pl.lit(0.5).alias("traversal_probability"))
        conn_t = (_attach_ratio(
            conn_e.group_by(["type_pre", "type_post"])
                  .agg(pl.col("weight").sum()), "type_post")
            .with_columns(pl.lit(0.5).alias("traversal_probability")))
        return conn_e, conn_t, None

    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", enrich)
    fc.FindShortestPath()

    # Surface 1: the type paths. TS->TA->TT realizes S->A2->T1, so its
    # hops carry the realized masses 9 and 12 over the type masses 77 and
    # 154 — NOT the cross-layer mean (8/77) or the pair sum (16/77).
    import ast
    type_csv = os.path.join(
        fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
    type_rows = {
        row[0]: row for row in pl.read_csv(type_csv).iter_rows()}
    assert "TS->TA->TT" in type_rows, sorted(type_rows)
    hop_ratios = ast.literal_eval(type_rows["TS->TA->TT"][3])
    assert hop_ratios[0] == pytest.approx(9 / 77)
    assert hop_ratios[1] == pytest.approx(12 / 154)

    # Surface 2: the connection table rows backing those hops imply the
    # SAME denominators (weight / connection_ratio == the type mass).
    conn_csv = os.path.join(
        fc.allpath_folder, "data_details", "connection_type.csv")
    conn = pl.read_csv(conn_csv)
    ta_tt = conn.filter(
        (pl.col("type_pre") == "TA") & (pl.col("type_post") == "TT"))
    assert ta_tt.height == 1
    assert ta_tt["weight"][0] == 12
    assert ta_tt["connection_ratio"][0] == pytest.approx(12 / 154)
    assert 12 / ta_tt["connection_ratio"][0] == pytest.approx(
        _DENOM_TYPE_MASS["TT"])
    # and the hop ratio equals the conn-table row's ratio exactly — one
    # run, one denominator
    assert hop_ratios[1] == pytest.approx(ta_tt["connection_ratio"][0])
    # the multi-layer pair (TS, TA) keeps BOTH per-layer rows with the
    # same single-fetch mass on each
    ts_ta = conn.filter(
        (pl.col("type_pre") == "TS") & (pl.col("type_post") == "TA"))
    assert sorted(ts_ta["weight"].to_list()) == [7, 9]
    for w, r in zip(ts_ta["weight"].to_list(),
                    ts_ta["connection_ratio"].to_list()):
        assert w / r == pytest.approx(_DENOM_TYPE_MASS["TA"])
    assert hop_ratios[0] == pytest.approx(
        9 / _DENOM_TYPE_MASS["TA"])
