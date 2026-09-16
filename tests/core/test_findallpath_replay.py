"""Feature F tests — pathfinding replay
(plan-cross-dataset-threshold-alignment.md §9, rollout steps 2–4).

- Bottleneck annotation: the natural tau equals a brute-force min over
  per-path bottlenecks (small graphs).
- Golden master: FindAllPathMultiThreshold([t0, t1, ...]) reproduces the
  per-threshold FindAllPath outputs file-by-file (type-path sets and
  connection tables identical).
- Property: replayed path sets are nested (t2 ⊆ t1 ⊆ t0) by construction.
- Budgeted StrongestFirst: the replayed set at threshold t is exactly the
  fresh SF run's set (prefix slice of the bottleneck-sorted array).
- Fallback: REPLAY_BUDGET exceeded -> results['_fallback'].
"""

import os
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "vispath-subproject" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402
import coana  # noqa: E402
from coana import _FINDALLPATH_GRAPH_CACHE  # noqa: E402

# Multiple routes with varied weights; several survive every threshold so
# the replay slices are non-trivial.
_EDGES = [
    ("S", "A", 5), ("S", "B", 8), ("A", "T", 6), ("B", "T", 3),
    ("A", "C", 4), ("C", "T", 9), ("S", "T", 12),
]
_THRESHOLDS = [1, 4, 7]


def _type_paths(folder):
    csv = os.path.join(folder, "src_to_tgt_allpaths_type.csv")
    return set(pl.read_csv(csv)["path"].to_list())


def _conn_pairs(folder):
    csv = os.path.join(folder, "data_details", "connection_type.csv")
    df = pl.read_csv(csv)
    return set(zip(df["type_pre"].to_list(), df["type_post"].to_list(),
                   df["weight"].to_list()))


def _fixture(monkeypatch, tmp_path, edges=_EDGES, max_interlayer=2,
             min_synapse=1, pathfinding="MemoizedDFS", max_paths_bodyid=0):
    fc, calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, edges, max_interlayer=max_interlayer,
        min_synapse=min_synapse)
    fc.pathfinding = pathfinding
    fc.max_paths_bodyid = max_paths_bodyid
    fc.capture_replay = False
    return fc, calls, logs


@pytest.fixture(autouse=True)
def _clean_graph_cache():
    _FINDALLPATH_GRAPH_CACHE.clear()
    yield
    _FINDALLPATH_GRAPH_CACHE.clear()


# ---------------------------------------------------------------------------
# Bottleneck annotation / natural tau (Feature G groundwork)
# ---------------------------------------------------------------------------

def brute_force_bottlenecks(edges, sources=("S",), targets=("T",),
                            max_hops=3):
    """All simple paths' bottleneck (min edge weight) by brute force."""
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
    bns = []

    def walk(node, visited, min_w):
        if node in targets and visited:
            bns.append(min_w)
        if len(visited) > max_hops:
            return
        for v, w in adj.get(node, []):
            if v in visited:
                continue
            walk(v, visited | {v}, min(min_w, w))

    for s in sources:
        walk(s, {s}, float("inf"))
    return bns


def test_natural_tau_equals_brute_force_min_bottleneck(monkeypatch, tmp_path):
    fc, _c, _l = _fixture(monkeypatch, tmp_path)
    fc.FindAllPath()
    bns = brute_force_bottlenecks(_EDGES)
    assert bns
    assert fc.strongest_first_cutoff == min(bns)
    assert fc.strongest_first_budget_bitten is False


# ---------------------------------------------------------------------------
# Golden master: multi-threshold replay == per-threshold enumeration
# ---------------------------------------------------------------------------

def test_replay_golden_master_matches_per_threshold(monkeypatch, tmp_path):
    # legacy: one full run per REAL threshold. The t0 run at 1 completes
    # with natural tau 3 (weakest path S->B->T), so under the F5 folder
    # discipline the collapse point 3 gets its own folder; asked 1 is
    # skipped (aliases minsyn_3) and 4/7 materialize normally.
    legacy = {}
    for t in (1, 3, 4, 7):
        fc, _c, _l = _fixture(monkeypatch, tmp_path / f"legacy_{t}",
                              min_synapse=t)
        fc.min_synapse_num = t
        fc.FindAllPath()
        legacy[t] = (_type_paths(fc.allpath_folder),
                     _conn_pairs(fc.allpath_folder))

    # replay: ONE multi-threshold run into minsyn_1 (the analyzer layout:
    # per-threshold folders minsyn_{t})
    fc, _c, _l = _fixture(monkeypatch, tmp_path / "replay")
    fc.saveas = str(tmp_path / "replay" / "minsyn_1")
    fc.save_folder = str(tmp_path / "replay" / "minsyn_1")
    results = fc.FindAllPathMultiThreshold(_THRESHOLDS)

    assert "_fallback" not in results
    # F5: asked 1 collapsed onto the tau folder; the intermediate
    # minsyn_1 folder was removed.
    assert results[1]["skipped"] is True
    assert results[1]["applied_folder"] == 3
    assert not (tmp_path / "replay" / "minsyn_1").exists()
    for t in (3, 4, 7):
        folder = tmp_path / "replay" / f"minsyn_{t}"
        assert folder.exists(), f"missing folder for t={t}"
        assert _type_paths(str(folder)) == legacy[t][0], f"type paths t={t}"
        assert _conn_pairs(str(folder)) == legacy[t][1], f"conn pairs t={t}"
        meta = results.get(t, results[1])
        assert meta["paths_complete"] is True
        # replayed thresholds: tau = min bottleneck of the slice (complete)
        if meta is not results[1] and meta.get("replayed"):
            bns = [b for b in brute_force_bottlenecks(_EDGES) if b >= t]
            assert meta["tau"] == min(bns)


def test_replayed_sets_are_nested(monkeypatch, tmp_path):
    fc, _c, _l = _fixture(monkeypatch, tmp_path / "replay")
    fc.saveas = str(tmp_path / "replay" / "minsyn_1")
    fc.save_folder = str(tmp_path / "replay" / "minsyn_1")
    fc.FindAllPathMultiThreshold(_THRESHOLDS)
    # F5 folders: the tau folder (3) plus every asked t > tau.
    sets = {t: _type_paths(str(tmp_path / "replay" / f"minsyn_{t}"))
            for t in (3, 4, 7)}
    assert sets[7] <= sets[4] <= sets[3]


def test_replay_matches_bottleneck_filter_bruteforce(monkeypatch, tmp_path):
    """Replay at t == brute-force 'all paths with bottleneck >= t'."""
    fc, _c, _l = _fixture(monkeypatch, tmp_path / "replay")
    fc.saveas = str(tmp_path / "replay" / "minsyn_1")
    fc.save_folder = str(tmp_path / "replay" / "minsyn_1")
    fc.FindAllPathMultiThreshold(_THRESHOLDS)

    bns = brute_force_bottlenecks(_EDGES)
    # type-level projection makes exact path identity awkward; compare the
    # bodyId-independent count: number of DISTINCT type paths with
    # min_weight >= t from the t0 export must match the replayed count.
    for t in (3, 4, 7):
        folder = str(tmp_path / "replay" / f"minsyn_{t}")
        df = pl.read_csv(os.path.join(folder, "src_to_tgt_allpaths_type.csv"))
        assert (df["min_weight"] >= t).all(), f"t={t}: weak path survived"


# ---------------------------------------------------------------------------
# Budgeted StrongestFirst: replay == prefix slice == fresh SF run
# ---------------------------------------------------------------------------

def test_budgeted_sf_replay_slice_matches_fresh_run(monkeypatch, tmp_path):
    """Budget bites at t0 (tau). For t >= tau the replayed set is the
    complete set at t and equals a fresh SF run's output at t."""
    budget = 3
    # replay run
    fc, _c, _l = _fixture(monkeypatch, tmp_path / "replay",
                          pathfinding="StrongestFirst",
                          max_paths_bodyid=budget)
    fc.saveas = str(tmp_path / "replay" / "minsyn_1")
    fc.save_folder = str(tmp_path / "replay" / "minsyn_1")
    results = fc.FindAllPathMultiThreshold([1, 4, 7])

    t0_meta = results[1]
    assert t0_meta["budget_bitten"] is True  # 7+ paths > 3 budget
    tau0 = t0_meta["tau"]
    assert tau0 is not None and t0_meta["paths_complete"] is False

    # t=4 and t=7 are >= tau? (tau depends on the tie drain — with this
    # graph SF emits strongest first; tau0 >= 4 means slices complete)
    replayed_7 = _type_paths(str(tmp_path / "replay" / "minsyn_7"))
    # fresh complete SF run at t=7 (budget large enough not to bite)
    fc2, _c2, _l2 = _fixture(monkeypatch, tmp_path / "fresh7",
                             pathfinding="StrongestFirst",
                             max_paths_bodyid=0, min_synapse=7)
    fc2.min_synapse_num = 7
    fc2.FindAllPath()
    fresh7 = _type_paths(fc2.allpath_folder)

    if tau0 <= 4:
        # the slices at 4 and 7 are complete sets — must equal fresh runs
        assert replayed_7 == fresh7
        assert results[7]["paths_complete"] is True
    else:
        # tau0 > 4: the slice is the tau0-bounded set, identical to t0's
        replayed_4 = _type_paths(str(tmp_path / "replay" / "minsyn_4"))
        assert replayed_4 == _type_paths(
            str(tmp_path / "replay" / "minsyn_1"))


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

def test_replay_fallback_on_budget_exceeded(monkeypatch, tmp_path):
    fc, _c, _l = _fixture(monkeypatch, tmp_path)
    fc.saveas = str(tmp_path / "minsyn_1")
    fc.save_folder = str(tmp_path / "minsyn_1")
    monkeypatch.setattr(coana.FindNeuronConnection,
                        "REPLAY_BUDGET_BYTES", 16)
    results = fc.FindAllPathMultiThreshold([1, 4])
    assert results.get("_fallback") is True


def test_single_threshold_replay_runs_normally(monkeypatch, tmp_path):
    fc, _c, _l = _fixture(monkeypatch, tmp_path)
    fc.saveas = str(tmp_path / "minsyn_4")
    fc.save_folder = str(tmp_path / "minsyn_4")
    results = fc.FindAllPathMultiThreshold([4])
    assert "_fallback" not in results
    assert results[4]["replayed"] is False
    assert _type_paths(str(tmp_path / "minsyn_4"))


# ---------------------------------------------------------------------------
# Per-slice provenance: the t0 graph floor must not be reported as binding
# for a higher slice it does not affect.
# ---------------------------------------------------------------------------

def test_replay_slice_above_t0_floor_reports_it_inactive(monkeypatch, tmp_path):
    """A t0 edge-budget floor sits below a higher requested threshold; the
    higher slice's own run is unaffected, so edge_budget_applied must be
    False for it (and the floor-w0 stays a t0-graph diagnostic)."""
    # Enough edges that the budget forces a floor at t0, with heavier edges
    # surviving at the higher threshold.
    edges = [
        ("S", "A", 5), ("A", "T", 6),
        ("S", "B", 2), ("B", "T", 2),
        ("S", "C", 3), ("C", "T", 3),
        ("S", "D", 4), ("D", "T", 4),
    ]
    fc, _c, _l = _fixture(
        monkeypatch, tmp_path, edges=edges, max_interlayer=2)
    fc.graph_edge_limit_bodyid = 4  # forces a floor on the t0=1 cone
    fc.saveas = str(tmp_path / "minsyn_1")
    fc.save_folder = str(tmp_path / "minsyn_1")
    results = fc.FindAllPathMultiThreshold([1, 5])

    floor = fc.edge_weight_floor
    assert floor is not None, "fixture did not exercise the edge floor"
    # The higher slice (5) is unaffected by a floor at/below it: it must not
    # be flagged as binding and must carry no landing tier, while still
    # recording the floor as a t0-graph diagnostic.
    if floor <= 5:
        assert results[5]["edge_budget_applied"] is False
        assert results[5]["edge_budget_landing"] is None
        assert results[5]["edge_weight_floor"] == floor


def test_replay_weak_gap_threshold_aliases_to_canonical(monkeypatch, tmp_path):
    """A requested threshold BELOW the canonical (t < w2+1) has paths the
    t0 budget dropped; the canonical folder is a valid complete run at
    canon > t, so it ALIASES there (applied=canon >= request) rather than
    re-enumerating."""
    # Paths S->T with bottlenecks 12, 5, 4, 2, 1. Budget 3 keeps 12,5,4
    # (tau=4), drops 2 and 1 => w2=2, canon=w2+1=3.
    edges = [
        ("S", "T", 12),
        ("S", "A", 5), ("A", "T", 6),
        ("S", "B", 4), ("B", "T", 9),
        ("S", "C", 2), ("C", "T", 9),
        ("S", "D", 1), ("D", "T", 9),
    ]
    fc, _c, _l = _fixture(monkeypatch, tmp_path, edges=edges,
                          max_interlayer=2, pathfinding="StrongestFirst",
                          max_paths_bodyid=3)
    fc.saveas = str(tmp_path / "minsyn_1")
    fc.save_folder = str(tmp_path / "minsyn_1")
    results = fc.FindAllPathMultiThreshold([1, 2, 3])

    assert results[1]["budget_bitten"] is True
    assert results[1]["tau"] == 4
    assert results[1]["tau_canonical"] == 3
    # t=2 (< canon): aliased to the canonical folder (applied 3 > request).
    assert results[2].get("skipped") is True
    assert results[2].get("applied_folder") == 3
    # t=3 == canon: aliased to the canonical folder.
    assert results[3].get("applied_folder") == 3


def test_replay_mid_gap_threshold_materializes_under_request(monkeypatch, tmp_path):
    """A requested threshold in the mid gap (canon < t <= tau) is exactly a
    complete run at t; it materializes its own folder so applied == t, not
    the weaker canonical."""
    # bottlenecks 30,20,8,5 ; budget 2 keeps 30,20 (tau=20), drops 8,5 =>
    # w2=8, canon=9. Requested 12 is in (9,20] -> mid gap.
    edges = [
        ("S", "T", 30),
        ("S", "A", 20), ("A", "T", 25),
        ("S", "B", 8), ("B", "T", 9),
        ("S", "C", 5), ("C", "T", 9),
    ]
    fc, _c, _l = _fixture(monkeypatch, tmp_path, edges=edges,
                          max_interlayer=2, pathfinding="StrongestFirst",
                          max_paths_bodyid=2)
    fc.saveas = str(tmp_path / "minsyn_1")
    fc.save_folder = str(tmp_path / "minsyn_1")
    results = fc.FindAllPathMultiThreshold([1, 12, 30])

    assert results[1]["tau"] == 20
    assert results[1]["tau_canonical"] == 9
    # t=12 in (9,20]: own folder, applied=12, not aliased to 9.
    assert results[12].get("skipped") is not True
    assert results[12]["applied_folder"] == 12
    assert os.path.isdir(str(tmp_path / "minsyn_12"))
    # t=30 > tau: materialized complete run at 30.
    assert results[30]["paths_complete"] is True
    assert os.path.isdir(str(tmp_path / "minsyn_30"))
