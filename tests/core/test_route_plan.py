"""Systematic tests for the corridor lane core (plan-corridor-lane.md)."""
import json
import time
from pathlib import Path

import polars as pl
import pytest

from pathlib import Path as _P
_VP = _P(__file__).resolve().parents[2] / "vispath-subproject" / "src"
import sys as _sys
if str(_VP) not in _sys.path:
    _sys.path.insert(0, str(_VP))

from route_plan import (ClassBudget, CorridorDrain, CorridorRefusal,
                        build_type_pairs, realize_count, run_lane,
                        validate_and_fill, widest_to_targets, wmax_map)


def frame(edges):
    return pl.DataFrame(
        {"bodyId_pre": [e[0] for e in edges],
         "bodyId_post": [e[1] for e in edges],
         "weight": [e[2] for e in edges]})

def tmap(m):
    return pl.DataFrame({"bid": [k for k in m], "t": [m[k] for k in m]})

A = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"}


def test_mass_conservation():
    edges = frame([("a1", "b1", 4), ("a2", "b2", 6), ("b1", "c1", 3)])
    typed, pairs = build_type_pairs(edges, tmap(A), 1)
    assert typed.height == 3
    assert pairs["w_sum"].sum() == edges["weight"].sum()
    row = pairs.filter((pl.col("type_pre") == "A")
                       & (pl.col("type_post") == "B"))
    assert row["w_max"][0] == 6 and row["n"][0] == 2


def test_min_synapse_and_untyped():
    m = dict(A, u1="nan")
    edges = frame([("a1", "b1", 2), ("a1", "b2", 5), ("b1", "u1", 9)])
    typed, pairs = build_type_pairs(edges, tmap(m), 3, drop_untyped=True)
    assert typed.height == 1 and pairs.height == 1
    assert pairs["w_max"][0] == 5


def test_ub_admissibility_and_certificate():
    # graph: a1 -10-> b1 -9-> c1 ; a1 -5-> b2 ; b2 -7-> c1
    adj = {"A": [("B", 10), ("B", 5)], "B": [("C", 9), ("C", 7)]}
    # w* per pair from a real type-pair table
    pairs = pl.DataFrame({"type_pre": ["A", "B"],
                          "type_post": ["B", "C"],
                          "w_max": [10, 9]})
    wstar = wmax_map(pairs)
    W = widest_to_targets(adj, {"C"}, 2)
    # drain emits walks descending by UB = min hop w*
    drain = CorridorDrain(adj, "A", {"C"}, 2)
    ubs = [ub for ub, _ in drain]
    assert ubs == sorted(ubs, reverse=True)
    # UB of A->B->C via (10,9) is 9; via (5,7)... w*(A,B)=10 so only one AB
    # pair-max: both AB walks share UB min(10, w*(B,C)) = 9/7 hopwise
    for ub, walk in CorridorDrain(adj, "A", {"C"}, 2):
        for a, b in zip(walk, walk[1:]):
            assert ub <= wstar.get((a, b), 0) + 1e-12


def test_phantom_validation():
    # type walk A->B->C exists in G_type* but b-layer is disconnected:
    # a1->b1 only; b2->c1 only  => both A->B->C walks are phantoms
    edges = frame([("a1", "b1", 9), ("b2", "c1", 8), ("a1", "c1", 1)])
    typed, pairs = build_type_pairs(edges, tmap(A), 1)
    # build per-hop dicts for the walk (A,B,C)
    hop_dicts = []
    for a, b in [("A", "B"), ("B", "C")]:
        sub = typed.filter((pl.col("type_pre") == a)
                           & (pl.col("type_post") == b))
        d = {}
        for u, v in zip(sub["bodyId_pre"], sub["bodyId_post"]):
            d.setdefault(u, []).append(v)
        hop_dicts.append(d)
    n, truncated = realize_count(hop_dicts, ["a1"], ["c1"], 100)
    assert n == 0 and not truncated  # phantom: no a->b->c chain


def test_class_budget_binds_and_fills():
    # dense star: 3 distinct A->B->C walks, each realizing 2 paths
    edges = [("a1", "b1", 9), ("b1", "c1", 9), ("a2", "b1", 8),
             ("a1", "b2", 7), ("b2", "c1", 7), ("a2", "b2", 6),
             ("a3", "b1", 5), ("a3", "b2", 5)]
    adj = {"A": [("B", 9), ("B", 7), ("B", 5)],
           "B": [("C", 9), ("C", 7)]}
    drain = CorridorDrain(adj, "A", {"C"}, 3)
    hop_pairs = {("A", "B"): {}, ("B", "C"): {}}
    for u, v, w in edges:
        tu, tv = A[u], A[v]
        if (tu, tv) in hop_pairs:
            hop_pairs[(tu, tv)].setdefault(u, []).append(v)

    # h2 = the realizable class; h3 keeps the loop alive (feasible by the
    # conservative d<=h rule but barren — no 3-walks), so the full h2
    # class captures its post-stop floor witness before drain exhaustion
    classes = {(("A", "C"), 2): ClassBudget(limit=3),
               (("A", "C"), 3): ClassBudget(limit=3)}
    def realize_fn(walk, cap):
        n, _tr = realize_count(
            [hop_pairs[h] for h in zip(walk, walk[1:])],
            ["a1", "a2", "a3"], {"c1"}, cap)
        return min(cap, n), False
    prov = validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "C"), 2)]
    assert 1 <= len(cb.walks) <= 3
    assert cb.used <= 3
    assert cb.stop == "budget"
    assert prov["drain_pops"] > 0
    # descending-UB acceptance within the class
    ubs = [ub for ub, _ in cb.walks]
    assert ubs == sorted(ubs, reverse=True)
    # honest floor witness: the drain yielded one more candidate (UB 7)
    # after the class filled — floor_next records it (plan §5: floor =
    # UB(M+1); the last ACCEPTED UB alone would overstate by a tie)
    assert cb.floor_next == 7 and cb.floor == 9
    # the barren-but-feasible class dried out without accepting anything
    assert classes[(("A", "C"), 3)].stop == "dry"


def test_validate_and_fill_probe_cap_and_infeasible():
    adj = {"A": [("B", 9)], "B": [("C", 9)], "C": []}
    drain = CorridorDrain(adj, "A", {"C"}, 2)
    classes = {(("A", "C"), 2): ClassBudget(limit=100, mh_prov=1),
               (("A", "C"), 1): ClassBudget(limit=100, feasible=False)}
    def realize_fn(walk, cap):  # thin walks never fill
        return min(cap, 10), False
    validate_and_fill(drain, classes, realize_fn)
    probe = classes[(("A", "C"), 2)]
    assert probe.pulled == 1 and probe.stop == "probe_cap"
    assert probe.satisfied() is True
    inf = classes[(("A", "C"), 1)]
    assert inf.stop == "infeasible" and inf.pulled == 0
    # the probe-capped class satisfies the global stop right after its
    # pull — almost no drain work (§22 cost control)
    assert drain.pops <= 5


def test_validate_and_fill_terminates_when_satisfied_via_phantoms():
    # bound-5 in-hook anchor regression: a class that satisfies via
    # PROBE-CAP on phantoms (never an acceptance) must still terminate
    # the drain — the acceptance-only recompute version spun to the pop
    # cap (>40 min on the real FAFB anchor)
    adj = {"A": [("B", 9)], "B": [("C", 9)], "C": []}
    drain = CorridorDrain(adj, "A", {"C"}, 2)
    classes = {(("A", "C"), 2): ClassBudget(limit=100, mh_prov=1)}

    def realize_fn(walk, cap):
        return 0, False  # every candidate a phantom

    validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "C"), 2)]
    assert cb.pulled == 1 and cb.phantoms == 1 and cb.stop == "probe_cap"
    assert drain.pops <= 4  # terminated; did not drain to the cap


def test_realize_count_work_cap_truncated():
    # fat phantom chain: the work cap cuts exploration before any path;
    # (0, True) means UNVERIFIED, never a proven phantom
    hop1 = {"s": [f"m{i}" for i in range(50)]}
    hop2 = {f"m{i}": [f"x{i}_{j}" for j in range(50)]
            for i in range(50)}
    hop3 = {f"x{i}_{j}": [] for i in range(50) for j in range(50)}
    n, truncated = realize_count([hop1, hop2, hop3], ["s"], set(),
                                 100, work_cap=1000)
    assert n == 0 and truncated
    # small chains stay exact
    n2, t2 = realize_count(
        [{"s": ["m"]}, {"m": ["t"]}], ["s"], {"t"}, 10)
    assert n2 == 1 and not t2


def test_drain_resumable_all_targets():
    adj = {"A": [("B", 5), ("D", 6)], "B": [("C", 4)],
           "D": [("C", 3)], "C": []}
    drain = CorridorDrain(adj, "A", {"C"}, 3)
    got = [(tuple(w)) for _ub, w in drain]
    assert ("A", "B", "C") in got and ("A", "D", "C") in got
    # exhausted drain raises StopIteration again (resumable semantics)
    with pytest.raises(StopIteration):
        next(drain)


def test_widest_fast_parity_with_pure_python():
    from route_plan import widest_to_targets_fast
    pairs = pl.DataFrame({
        "type_pre": ["A", "A", "B", "D", "B", "E"],
        "type_post": ["B", "D", "C", "C", "E", "C"],
        "w_max": [10, 6, 4, 3, 8, 2]})
    adj = {}
    for a, b, w in zip(pairs["type_pre"], pairs["type_post"],
                       pairs["w_max"]):
        adj.setdefault(a, []).append((b, w))
    W_pure = widest_to_targets(adj, {"C"}, 3)
    W_fast = widest_to_targets_fast(pairs, {"C"}, 3)
    assert len(W_pure) == len(W_fast) == 4
    for r in range(len(W_pure)):
        for node in set(W_pure[r]) | set(W_fast[r]):
            assert float(W_pure[r].get(node, 0)) == \
                float(W_fast[r].get(node, 0)), (r, node)


def test_shared_W_across_sources():
    adj = {"A": [("B", 5)], "D": [("B", 4)], "B": [("C", 3)], "C": []}
    W = widest_to_targets(adj, {"C"}, 3)
    d1 = [w for _ub, w in CorridorDrain(adj, "A", {"C"}, 3, W=W)]
    d2 = [w for _ub, w in CorridorDrain(adj, "D", {"C"}, 3, W=W)]
    assert ("A", "B", "C") in d1 and ("D", "B", "C") in d2


# ----------------------------------------------------- run parameters
def test_read_run_params_spaced_keys_and_list_repr(tmp_path):
    from route_plan import _parse_list, read_run_params
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "Parameters for processing aMe12 to PPL101:\n"
        "source neurons:             ['aMe12', 'aMe02']\n"
        "target neurons:             ['PPL101']\n"
        "min synapse number:         3\n"
        "max interlayer:             5\n"
        "dataset:                    flywire:v783\n"
        "separate hemispheres:       True\n"
        "exclude intra-type connections: False\n"
        "path_mode:                  all\n")
    kv = read_run_params(run)
    assert kv["min synapse number"] == "3"
    assert kv["max interlayer"] == "5"
    assert _parse_list(kv["source neurons"]) == ["aMe12", "aMe02"]
    assert _parse_list(kv["target neurons"]) == ["PPL101"]
    assert _parse_list("a,b, c") == ["a", "b", "c"]


def test_match_type_labels_hemisphere_suffixes():
    from route_plan import _match_type_labels
    labels = ["aMe12", "aMe12_L", "aMe12_R", "R7p_L", "APL"]
    # F-PF-005: the helper returns (matched labels, explicit bodyId tokens)
    assert _match_type_labels(labels, ["aMe12"]) == ({"aMe12"}, set())
    assert _match_type_labels(labels, ["aMe12"], body2type={}) == \
        ({"aMe12"}, set())
    labels2 = ["aMe12_L", "aMe12_R", "R7p_L"]
    assert _match_type_labels(labels2, ["aMe12"]) == \
        ({"aMe12_L", "aMe12_R"}, set())
    # a token absent from the label universe cannot enroll (refused later)
    assert _match_type_labels(labels2, ["APL"],
                              body2type={"12345": "APL"}) == (set(), set())
    # bodyId-routed token enrolls via its effective label AND keeps its
    # explicit bodyId membership (the whole-type expansion is gone)
    assert _match_type_labels(labels2, ["12345"],
                              body2type={"12345": "APL"}) == \
        ({"APL"}, {"12345"})


# ------------------------------------------------------- end-to-end lane
def _write_synth_dataset(tmp_path, n_intermediate=200, int_ids=False):
    """a1 -> {b_i} -> c1 with strictly ordered weights; A->B->C is the
    only corridor and realizes n_intermediate distinct-bottleneck paths."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pre, post, w = [], [], []
    bid = (lambda i: str(1000 + i)) if int_ids else (lambda i: f"b{i}")
    a1, c1 = ("1", "2") if int_ids else ("a1", "c1")
    for i in range(n_intermediate):
        b = bid(i)
        pre += [a1, b]; post += [b, c1]
        w += [100 + i, 200 + i]
    pl.DataFrame({"bodyId_pre": pre, "bodyId_post": post,
                  "weight": w}).write_parquet(root / "connections.parquet")
    bodies = [a1, c1] + [bid(i) for i in range(n_intermediate)]
    types = ["A", "C"] + ["B"] * n_intermediate
    pd.DataFrame({"bodyId": bodies, "type": types}).to_parquet(
        root / "neurons.parquet", index=False)
    return root


def test_run_lane_direct_end_to_end_plan_faithful(tmp_path):
    root = _write_synth_dataset(tmp_path)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    rows = pl.read_csv(out / "route_paths_type.csv")
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["status"] == "complete"
    assert prov["trigger"] == "driver-direct"
    # §23.5 contract: ONE type row per validated walk; per-path detail in
    # the bodyId csv. Budget = the class share (200 realizations available,
    # limit 100) — the strongest 100 paths, tau = the 100th bottleneck.
    assert rows.height == 1 and body.height == 100
    row = rows.row(0, named=True)
    assert row["path"] == "A->B->C" and row["route_h"] == row["length"] == 2
    assert row["route_ub"] == 299.0        # w*(A,B)=299 top walk UB
    assert row["plan_rank"] == 1
    assert row["applied_tau"] == 200.0     # 100th bottleneck (i=100)
    assert row["realized_paths"] == 100
    # engine column semantics: summed per-hop realized masses + per-node
    # coverage over type membership (the emitted 100 = i in 100..199)
    assert json.loads(row["weights"]) == [24950, 34950]
    assert row["coverage"] == "[1/1, 100/200, 1/1]"  # display syntax
    assert row["min_weight"] == 24950
    assert len(json.loads(row["ratios"])) == 2
    # C1.2: adj ratios present; at asked=1 adj == plain BY CONSTRUCTION
    assert json.loads(row["ratios_adj"]) == json.loads(row["ratios"])
    assert row["min_ratio_adj"] == row["min_ratio"]
    cls = prov["classes"]["A->C@h2"]
    assert cls["stop"] == "budget" and cls["emitted"] == 100
    assert cls["off_plan"] == 0
    # connection table (C1.3): mirror family + corridor tail; realized
    # ratios == plain at asked 1
    conn = pl.read_csv(out / "route_connection_type.csv")
    assert conn.columns[:7] == ["type_pre", "type_post", "weight",
                                "nt_type", "connection_ratio", "conn_layer",
                                "connection_ratio_adj"]
    assert set(conn["conn_layer"]) == {"route_scope"}
    assert set(zip(conn["type_pre"], conn["type_post"])) == {
        ("A", "B"), ("B", "C")}
    ab = conn.filter((pl.col("type_pre") == "A")
                     & (pl.col("type_post") == "B"))
    assert ab["realized_mass"][0] == 24950 and ab["realizing_paths"][0] == 100
    assert ab["connection_ratio"][0] == ab["connection_ratio_adj"][0]
    plain = json.loads(row["ratios"])[0]
    assert abs(ab["connection_ratio"][0] - round(plain, 6)) < 1e-9
    # plan.json: content-hashed serialized plan (direct mode -> beside out)
    plan = json.loads((out / "data_details" / "route_plan"
                       / "plan.json").read_text())
    assert plan["plan_hash"] == prov["plan_hash"]
    assert plan["classes"]["A->C@h2"]["walks"] == [[299, ["A", "B", "C"]]]
    assert prov["validated_walks"] == 1 and prov["bodyId_paths"] == 100
    assert prov["scope_pairs"] == 2


def test_run_lane_int_bodyids(tmp_path):
    # int64 bodyIds in both tables must survive the Utf8 cast chain
    root = _write_synth_dataset(tmp_path, n_intermediate=5, int_ids=True)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    rows = pl.read_csv(out / "route_paths_type.csv")
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    assert rows.height == 1 and body.height == 5
    assert set(rows["path"]) == {"A->B->C"}


def test_run_lane_run_folder_parameters_parsing(tmp_path):
    # run-folder mode must read SPACED parameters.txt keys (the engine's
    # grammar), not underscore names
    root = _write_synth_dataset(tmp_path, n_intermediate=5)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "Parameters for processing A to C:\n"
        f"source neurons:             ['A']\n"
        f"target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "dataset:                    synth\n")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    rows = pl.read_csv(run / "route_scope_paths" / "route_paths_type.csv")
    body = pl.read_csv(run / "route_scope_paths" / "route_paths_bodyId.csv")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert rows.height == 1 and body.height == 5   # 1 walk, 5 realizations
    assert prov["asked"] == 1 and prov["bound"] == 3
    assert prov["gap_partition"] is None or isinstance(
        prov["gap_partition"], dict)  # no allpaths here -> None


def test_run_lane_ratio_basis_refused(tmp_path):
    root = _write_synth_dataset(tmp_path, n_intermediate=3)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "weight basis:              connection_ratio\n"
        "min synapse number:         1\n")
    with pytest.raises(CorridorRefusal):
        run_lane(run, connections_path=root / "connections.parquet",
                 neuron_table=root / "neurons.parquet")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["status"] == "refused"
    assert "ratio" in prov["reason"]


def test_run_lane_missing_inputs_refused(tmp_path):
    with pytest.raises(CorridorRefusal):
        run_lane(None, asked=1, max_interlayer=2, source_types=["A"],
                 target_types=["C"], out_dir=tmp_path / "o")


def test_run_lane_no_duplicates_across_classes(tmp_path):
    # multi-length corridor: A->B->C (h2) and A->C (h1) both realizable;
    # per-class enumeration must not duplicate rows across h classes
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "b1", "a2", "a1"],
        "bodyId_post": ["b1", "c1", "b1", "c1"],
        "weight": [10, 9, 8, 5]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "a2", "b1", "c1"],
                  "type": ["A", "A", "B", "C"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "o"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out, per_pair_length_budget=100)
    rows = pl.read_csv(out / "route_paths_type.csv")
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    # per-walk type rows: unique walks; path decomposes to exactly h hops
    assert (rows["route_h"] == rows["length"]).all()
    assert rows.height == rows["path"].n_unique()
    for p, h in zip(rows["path"].to_list(), rows["route_h"].to_list()):
        assert len(p.split("->")) - 1 == h
    # per-path bodyId rows: unique realizations; totals match walk rows
    bkeys = list(zip(body["path"].to_list(), body["weights"].to_list()))
    assert len(bkeys) == len(set(bkeys))
    assert (body["route_h"] == body["length"]).all()
    assert body.height == sum(rows["realized_paths"].to_list())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_cache_hit_and_invalidation(tmp_path):
    """P1.1: same inputs → cache hit (byte-identical universe); touching
    the connections file → new signature → rebuild (fail-closed)."""
    root = _write_synth_dataset(tmp_path, n_intermediate=30)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out1)
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out2)
    # identical outputs across the cached run
    t1 = pl.read_csv(out1 / "route_paths_type.csv")
    t2 = pl.read_csv(out2 / "route_paths_type.csv")
    assert t1.equals(t2)
    # cache hit: rerunning the SAME inputs creates NO new cache entry
    # (the dir is shared across tests — diff the file set)
    def entries():
        return {f.name for f in Path(route_plan_cache_dir()).glob(
            "universe__*.parquet")}
    before = entries()
    assert before, "first run must have written a cache entry"
    import os
    os.utime(root / "connections.parquet", (time.time() + 5,
                                            time.time() + 5))
    out3 = tmp_path / "o3"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out3)
    after = entries()
    new_keys = after - before
    assert len(new_keys) == 1  # touched table → new signature → new entry
    t3 = pl.read_csv(out3 / "route_paths_type.csv")
    assert t3.equals(t1)  # rebuilt universe is byte-equivalent in output


def route_plan_cache_dir():
    import route_plan
    from pathlib import Path
    return (Path(route_plan.__file__).resolve().parents[1] / "cache"
            / "route_plan" / "ds")


def test_layered_enumeration_multi_visit(tmp_path):
    """P1.2 acceptance + planner rule (2026-10-11): a multi-visit walk
    (A->B->A->C) is NOT planned by default (user default — reciprocals
    make multi-visit subtrees explode); allow_multi_visit=True restores
    the layered enumeration of its exact realizations."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    # a1 -10-> b1 -9-> a2 -8-> c1 ; types A->B->A->C (A revisited)
    pl.DataFrame({
        "bodyId_pre": ["a1", "b1", "a2"],
        "bodyId_post": ["b1", "a2", "c1"],
        "weight": [10, 9, 8]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "b1", "a2", "c1"],
                  "type": ["A", "B", "A", "C"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "o"
    assert run_lane(None, asked=1, max_interlayer=3,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out, per_pair_length_budget=100)
    rows = pl.read_csv(out / "route_paths_type.csv")
    # default: the multi-visit walk is not planned
    assert rows.filter(pl.col("route_h") == 3).height == 0
    # knob restores it
    out2 = tmp_path / "o2"
    assert run_lane(None, asked=1, max_interlayer=3,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out2, per_pair_length_budget=100,
                    allow_multi_visit=True)
    rows2 = pl.read_csv(out2 / "route_paths_type.csv")
    h3 = rows2.filter(pl.col("route_h") == 3)
    assert h3.height == 1 and h3["realized_paths"][0] == 1
    assert h3["path"][0] == "A->B->A->C"
    body = pl.read_csv(out2 / "route_paths_bodyId.csv")
    assert body.filter(pl.col("route_h") == 3).height == 1


def test_shortest_store_validation_basis(tmp_path):
    """P1.3: a shortest run folder with a store-like layer layout gets
    validation_basis='shortest_dag_store'; without it, the fallback is
    disclosed."""
    root = _write_synth_dataset(tmp_path, n_intermediate=5)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  shortest\n"
        "dataset:                    synth\n")
    # no store -> fallback disclosed
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["validation_basis"] == "lossless_prune_fallback"
    readme = (run / "route_scope_paths" / "README.md").read_text()
    assert "FALLBACK" in readme


# ------------------------------------------------ plan-2026-10-11 rounds
def test_drain_deadline_bites_immediately():
    """F2: the drain's wall-clock check must fire against a monotonic
    deadline (the old epoch-derived deadline made this check dead code —
    the store-basis lane burned 4,471s against a 900s budget)."""
    adj = {"A": [("B", 5.0)], "B": [("C", 5.0)], "C": []}
    W = widest_to_targets(adj, {"C"}, 2)
    d = CorridorDrain(adj, "A", {"C"}, 2, W=W,
                      deadline=time.monotonic() - 1.0)
    with pytest.raises(StopIteration):
        next(iter(d))
    assert d.time_capped and d.exhausted


def test_validate_and_fill_deadline_bites_inside_loop():
    """F2: the validate loop must stop INSIDE the pull loop when the
    monotonic deadline passes (previously only post-validate checks
    worked, so the expensive phase ran unbudgeted)."""
    adj = {"A": [(f"X{i}", 5.0) for i in range(40)]}
    for i in range(40):
        adj[f"X{i}"] = [("C", 5.0)]
    adj["C"] = []
    W = widest_to_targets(adj, {"C"}, 2)
    drain = CorridorDrain(adj, "A", {"C"}, 2, W=W)
    classes = {(("A", "C"), 2): ClassBudget(limit=10_000)}

    def slow_realize(_walk, _cap):
        time.sleep(0.02)
        return 0, False          # phantom: the class stays hungry

    prov = validate_and_fill(
        drain, classes, slow_realize,
        deadline=time.monotonic() + 0.15)
    assert prov["time_capped"] is True
    assert 1 <= prov["pulls"] <= 15            # ~20ms per pull
    assert prov["classes"]["A->C@h2"]["stop"] == "capped"


def test_planner_rule_blocks_first_hop_self_pair():
    """F1: a SELF type pair at the first hop (X->X, present whenever
    exclude_intra is off) must not enter as walk (X, X) — the init push
    previously had no multi-visit check, which is exactly how
    R8->R8->aMe12->DN1-l->5th-LNv survived the planner rule."""
    adj = {"A": [("A", 9.0), ("B", 5.0)], "B": [("C", 5.0)], "C": []}
    W = widest_to_targets(adj, {"C"}, 4)
    d = CorridorDrain(adj, "A", {"C"}, 4, W=W, allow_multi_visit=False)
    walks = [w for _ub, w in d]
    assert ("A", "A", "B", "C") not in walks
    assert ("A", "B", "C") in walks
    assert d.multi_visit_pruned >= 1
    W2 = widest_to_targets(adj, {"C"}, 4)
    d2 = CorridorDrain(adj, "A", {"C"}, 4, W=W2, allow_multi_visit=True)
    assert ("A", "A", "B", "C") in [w for _u, w in d2]


def test_dp_count_work_cap_lower_bound():
    """F3: a capped forward pass returns (lower bound, True) — sums only
    grow, so a partial pass under-counts but never over-counts; the
    uncapped pass is exact."""
    from route_plan import _dp_count
    # s -> u1..u10 -> v1..v10 -> t : exactly 10 realizations, 30 relaxations
    hop0 = {"s": [f"u{i}" for i in range(10)]}
    hop1 = {f"u{i}": [f"v{i}"] for i in range(10)}
    hop2 = {f"v{i}": ["t"] for i in range(10)}
    n, truncated = _dp_count([hop0, hop1, hop2], {"s"}, {"t"})
    assert (n, truncated) == (10, False)
    n2, truncated2 = _dp_count([hop0, hop1, hop2], {"s"}, {"t"},
                               work_cap=25)
    assert truncated2 is True and 0 < n2 < 10


def test_pair_edges_views_dedup_canonical_order():
    """F4/F5: the batched pair structure serves dedup'd hop dicts
    (validation) and canonically ordered tuples (enumeration); duplicate
    rows collapse keep-first (plan F8 determinism)."""
    from route_plan import _PairEdges
    loss = pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "a2", "b1"],
        "bodyId_post": ["b1", "b1", "b1", "c1"],
        "weight": [5.0, 9.0, 7.0, 3.0],
        "type_pre": ["A", "A", "A", "B"],
        "type_post": ["B", "B", "B", "C"]})
    pe = _PairEdges(loss)
    assert pe.tuples("A", "B") == [("a1", "b1", 5.0),
                                   ("a2", "b1", 7.0)]
    assert pe.hop("A", "B") == {"a1": ["b1"], "a2": ["b1"]}
    assert pe.hop("B", "C") == {"b1": ["c1"]}
    assert pe.tuples("C", "A") == [] and pe.hop("C", "A") == {}


def test_store_universe_dedup_keep_first_sorted_layer(tmp_path):
    """F8: store edges recurring across layers (91% on the real MCNS
    store) collapse keep-first over SORTED layer labels — the lane's
    edge weights become deterministic instead of last-row-wins, and the
    vectorized reader replaces the 27M-row Python loader pass."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "b1"],
        "bodyId_post": ["b1", "c1"],
        "weight": [9, 3]}).write_parquet(root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "b1", "c1"],
                  "type": ["A", "B", "C"]}).to_parquet(
        root / "neurons.parquet", index=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  shortest\n"
        "dataset:                    synth\n")
    # real store layout with per-ROI SLICE rows: (a1,b1) carries two L0
    # slices (2+7=9) and one L1 row (5) — the slice SUM is the physical
    # weight, sorted-first (L0) wins, and the cross-layer disagreement
    # after summation is 0 (the fetch is self-consistent). (b1,c1) is a
    # single row repeated verbatim in both layers (refetch multiplicity).
    store = run / "shortest_discovery_store"
    (store / "connections").mkdir(parents=True)
    (store / "meta.json").write_text('{"targets_found": ["c1"]}')
    pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "b1", "b1"],
        "bodyId_post": ["b1", "b1", "c1", "c1"],
        "roi": ["A", "B", "A", "A"],
        "weight": [2.0, 7.0, 3.0, 3.0]}).write_parquet(
        store / "connections" / "connections_L0.parquet")
    pl.DataFrame({
        "bodyId_pre": ["a1", "b1"],
        "bodyId_post": ["b1", "c1"],
        "roi": ["A", "A"],
        "weight": [5.0, 3.0]}).write_parquet(
        store / "connections" / "connections_L1.parquet")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    out = run / "route_scope_paths"
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["validation_basis"] == "shortest_dag_store"
    assert prov["store_dedup"] == "ROI-slice sum per layer; keep-first by sorted layer"
    # the L1 refetch carries ONE slice (5) vs L0's summed 9 — a simulated
    # incomplete refetch is exactly what the conflict disclosure exists
    # to surface (the real MCNS store discloses 0)
    assert prov["store_weight_conflicts"] == 1
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    assert body["weights"].to_list() == ["9,3"]


def test_layered_permutation_paths_excluded(tmp_path):
    """F5b sequence fidelity (knob mode): a multi-visit walk's pair set
    can compose a DIFFERENT type sequence of the same length (here
    A->B->C->B->A->D also composes A->B->A->B->A->D over the same pair
    set). Per-position placement forces the walk's own sequence — the
    old all-layer placement emitted the permutation under the wrong
    walk, aggregating its masses under wrong hop labels."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "a2", "b1", "c1", "b1", "b2", "b2",
                       "a2", "a3"],
        "bodyId_post": ["b1", "b2", "c1", "b2", "a2", "a2", "a3",
                        "d1", "d1"],
        "weight": [10, 10, 9, 8, 7, 6, 6, 5, 5]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "a2", "a3", "b1", "b2", "c1",
                             "d1"],
                  "type": ["A", "A", "A", "B", "B", "C",
                           "D"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "o"
    assert run_lane(None, asked=1, max_interlayer=5,
                    source_types=["A"], target_types=["D"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out, per_pair_length_budget=100,
                    allow_multi_visit=True)
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    b2t = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B",
           "c1": "C", "d1": "D"}

    def seq(path):
        return "->".join(b2t[b] for b in path.split("->"))

    h5 = body.filter(pl.col("route_h") == 5)
    cbacd = h5.filter(
        pl.col("path").map_elements(seq, return_dtype=pl.Utf8)
        == "A->B->C->B->A->D")
    # exactly the walk's own 2 realizations — the permutation
    # a1->b1->a2->b2->a3->d1 must NOT appear under this walk's type
    # sequence (old all-layer placement emitted it here)
    assert cbacd.height == 2
    assert "a1->b1->a2->b2->a3->d1" not in set(
        p for p in cbacd["path"].to_list())


def test_pair_edges_matches_reference_filter_recipe():
    """Literal twin test (review round): _PairEdges must agree with the
    OLD per-pair full-frame-filter recipe (the deleted _HopIndex: filter
    loss per pair, append rows preserving duplicates) on a seeded
    randomized universe — adjacency as sets, tuples as the dedup'd
    canonical order, duplicate store rows collapsing keep-first."""
    import random
    from route_plan import _PairEdges
    rng = random.Random(20261011)
    types = ["A", "B", "C", "D"]
    bodies = [f"b{i}" for i in range(40)]
    rows = []
    for _ in range(400):
        a, b = rng.choice(bodies), rng.choice(bodies)
        rows.append((a, b, float(rng.randint(1, 20)),
                     f"T{rng.randrange(4)}", f"T{rng.randrange(4)}"))
    # force duplicates across AND within pairs (the store-basis shape)
    rows.extend(rows[:25])
    loss = pl.DataFrame(
        rows, schema={"bodyId_pre": pl.Utf8, "bodyId_post": pl.Utf8,
                      "weight": pl.Float64, "type_pre": pl.Utf8,
                      "type_post": pl.Utf8}, orient="row")
    pe = _PairEdges(loss)
    for a in types:
        for b in types:
            # OLD recipe: full-frame filter per pair, duplicates kept
            sub = loss.filter((pl.col("type_pre") == a)
                              & (pl.col("type_post") == b))
            old_pairs = list(zip(sub["bodyId_pre"].to_list(),
                                 sub["bodyId_post"].to_list()))
            old_adj = {}
            for u, v in old_pairs:
                old_adj.setdefault(u, []).append(v)
            new_adj = pe.hop(a, b)
            # adjacency as sets must match exactly (both dedup'd)
            assert {u: set(vs) for u, vs in old_adj.items()} \
                == {u: set(vs) for u, vs in new_adj.items()}, (a, b)
            # tuples = dedup'd rows; weights = keep-first over the
            # canonical (u, v) sort (duplicates collapse deterministically)
            seen = {}
            for u, v, w in sorted(set(zip(sub["bodyId_pre"].to_list(),
                                          sub["bodyId_post"].to_list(),
                                          sub["weight"].to_list()))):
                seen[(u, v)] = w
            assert pe.tuples(a, b) == [(u, v, w) for (u, v), w
                                       in sorted(seen.items())], (a, b)


# ---------------------------------------------------------------------
# Pathfinding-audit fixes (issues-2026-10-13; plan ...-audit-fixes.md)
# ---------------------------------------------------------------------

def test_and_fractional_ratio_refused(tmp_path):
    """F-PF-004: a fractional min connection ratio must fire the
    dual-threshold refusal — the old integer parser defaulted '0.01' to 0
    and the lane searched a graph violating the parent's own predicate
    (a 1-synapse edge into a 1000-synapse cell passed the corridor)."""
    root = _write_synth_dataset(tmp_path, n_intermediate=3)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "min synapse number:         1\n"
        "min connection ratio:       0.01\n")
    with pytest.raises(CorridorRefusal):
        run_lane(run, connections_path=root / "connections.parquet",
                 neuron_table=root / "neurons.parquet")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["status"] == "refused"
    assert "dual-threshold" in prov["reason"]


def test_and_malformed_ratio_refused(tmp_path):
    """F-PF-004: a non-numeric threshold refuses — never silently zero."""
    root = _write_synth_dataset(tmp_path, n_intermediate=3)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "min synapse number:         1\n"
        "min connection ratio:       strong\n")
    with pytest.raises(CorridorRefusal):
        run_lane(run, connections_path=root / "connections.parquet",
                 neuron_table=root / "neurons.parquet")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert "not a number" in prov["reason"]


def test_and_zero_ratio_passes(tmp_path):
    """F-PF-004: an explicit zero is not AND mode."""
    root = _write_synth_dataset(tmp_path, n_intermediate=3)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "min connection ratio:       0\n")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")


def test_missing_mapping_file_refused(tmp_path):
    """F-PF-010: a recorded-but-absent mapping file refuses instead of
    silently falling back to native labels (which changes the settled
    group membership and every exported label)."""
    root = _write_synth_dataset(tmp_path, n_intermediate=3)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "custom mapping file:        /tmp/drocat-audit-missing-mapper.json\n")
    with pytest.raises(CorridorRefusal):
        run_lane(run, connections_path=root / "connections.parquet",
                 neuron_table=root / "neurons.parquet")
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["status"] == "refused"
    assert "does not exist" in prov["reason"]


def test_bodyid_token_enrollment_keeps_membership(tmp_path):
    """F-PF-005 (direct mode): an explicit bodyId source token keeps its
    membership — the unrequested sibling body's much stronger edge must
    not enter the query (the old expansion enrolled the whole type)."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "a2"],
        "bodyId_post": ["c1", "c1"],
        "weight": [2, 100]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "a2", "c1"],
                  "type": ["A", "A", "C"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=1,
                    source_types=["a1"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    assert body["path"].to_list() == ["a1->c1"]
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["enrollment"]["sources"] == "bodyids"


def test_run_folder_parent_csv_membership(tmp_path):
    """F-PF-005 (run-folder mode): the parent's enrollment CSVs are the
    query's resolved membership — a partially Checked target set must
    survive into the corridor instead of being re-expanded to the type."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["s1", "s1"],
        "bodyId_post": ["t1", "t2"],
        "weight": [5, 50]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["s1", "t1", "t2"],
                  "type": ["S", "C", "C"]}).to_csv(
        root / "neurons.csv", index=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['S']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "dataset:                    synth\n")
    pd.DataFrame({"bodyId": ["s1"]}).to_csv(
        run / "source_neurons.csv", index=False)
    pd.DataFrame({"bodyId": ["t1", "t2"],
                  "Checked": ["True", "False"]}).to_csv(
        run / "target_neurons.csv", index=False)
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.csv")
    body = pl.read_csv(run / "route_scope_paths" / "route_paths_bodyId.csv")
    assert body["path"].to_list() == ["s1->t1"]
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["enrollment"]["targets"] == "types+parent_csv"
