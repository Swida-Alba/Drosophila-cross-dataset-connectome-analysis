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
    # class captures its post-stop floor witness before drain exhaustion.
    # ROUTE SHARE (2026-10-16): acceptance is counted in published
    # type-level ROUTES. Every candidate here realizes the SAME 6 bodyId
    # paths, so the old realization-counted budget stopped this class after
    # ONE walk (6 >= limit 3); with a share of 2 it stops after two.
    classes = {(("A", "C"), 2): ClassBudget(limit=3, route_share=2),
               (("A", "C"), 3): ClassBudget(limit=3, route_share=2)}
    def realize_fn(walk, cap):
        n, _tr = realize_count(
            [hop_pairs[h] for h in zip(walk, walk[1:])],
            ["a1", "a2", "a3"], {"c1"}, cap)
        return min(cap, n), False
    prov = validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "C"), 2)]
    assert len(cb.walks) == 2
    assert cb.used == 2, "used counts routes, never realizations"
    assert cb.stop == "budget"
    assert prov["drain_pops"] > 0
    # descending-UB acceptance within the class
    ubs = [ub for ub, _ in cb.walks]
    assert ubs == sorted(ubs, reverse=True)
    # honest floor witness: the drain yielded one more candidate (UB 7)
    # after the class filled — floor_next records it (plan §5: floor =
    # UB(M+1); the last ACCEPTED UB alone would overstate safety by a tie).
    # With the share counted in routes the class fills on the 9 and the
    # first 7, so the witness is a TIE at 7 — the case §5 exists for.
    assert cb.floor == 7 and cb.floor_next == 7
    # the barren-but-feasible class dried out without accepting anything
    assert classes[(("A", "C"), 3)].stop == "dry"


def test_validate_and_fill_probe_cap_and_infeasible():
    adj = {"A": [("B", 9)], "B": [("C", 9)], "C": []}
    drain = CorridorDrain(adj, "A", {"C"}, 2)
    classes = {(("A", "C"), 2): ClassBudget(limit=100, route_share=100,
                                            pull_cap=1),
               (("A", "C"), 1): ClassBudget(limit=100, route_share=100,
                                            feasible=False)}
    def realize_fn(walk, cap):  # existence probe: any realization counts
        return min(cap, 10), False
    validate_and_fill(drain, classes, realize_fn)
    probe = classes[(("A", "C"), 2)]
    assert probe.pulled == 1 and probe.stop == "probe_cap"
    assert probe.used == 1 < probe.share, "share unfilled at the pull cap"
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
    classes = {(("A", "C"), 2): ClassBudget(limit=100, route_share=100,
                                            pull_cap=1)}

    def realize_fn(walk, cap):
        return 0, False  # every candidate a phantom

    validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "C"), 2)]
    assert cb.pulled == 1 and cb.phantoms == 1 and cb.stop == "probe_cap"
    assert drain.pops <= 4  # terminated; did not drain to the cap


def test_fat_walk_cannot_buy_the_class_route_share():
    """2026-10-16 (user-measured): aMe12->PPL103@h2 published ONE route.

    The acceptance loop spent the class budget on the walk's REALIZATION
    COUNT, so the single walk aMe12->KCg-d->PPL103 (>= 100 bodyId paths)
    filled the class and the candidate at UB 15 behind it was never
    accepted (accepted=1, used=100, pulled=1, stop='budget'). Acceptance
    now probes existence and spends ONE route per accepted walk; the
    realization envelope (`limit`) belongs to the materialization pass.
    """
    adj = {"A": [("B", 9), ("C", 8), ("D", 7)],
           "B": [("T", 9)], "C": [("T", 8)], "D": [("T", 7)], "T": []}
    drain = CorridorDrain(adj, "A", {"T"}, 2)
    # B is the fat corridor (a huge realization set), C/D are thin.
    realizing = {"A->B->T": 5_000, "A->C->T": 2, "A->D->T": 1}
    probed = []

    def realize_fn(walk, cap):
        n = realizing["->".join(walk)]
        probed.append(cap)
        return min(cap, n), False

    classes = {(("A", "T"), 2): ClassBudget(limit=3, route_share=3)}
    validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "T"), 2)]
    assert len(cb.walks) == 3, "the fat walk must not crowd out its siblings"
    assert cb.used == 3 == cb.share
    assert set(probed) == {1}, (
        "acceptance probes EXISTENCE (cap 1); counting realizations here is "
        "how one corridor ate the whole class")
    assert cb.stop == "budget"


def test_phantom_still_rejected_under_route_counting():
    """The existence probe must not become an accept-anything: a walk with
    no layered realization is still a phantom and still costs a pull, and a
    run of phantoms still terminates the class (pull cap, §22 cost control).
    """
    adj = {"A": [("B", 9), ("C", 8)], "B": [("T", 9)], "C": [("T", 8)],
           "T": []}
    drain = CorridorDrain(adj, "A", {"T"}, 2)
    classes = {(("A", "T"), 2): ClassBudget(limit=10, route_share=2,
                                            pull_cap=2)}

    def realize_fn(walk, cap):
        return 0, False  # every candidate is a phantom

    validate_and_fill(drain, classes, realize_fn)
    cb = classes[(("A", "T"), 2)]
    assert cb.walks == [] and cb.phantoms == 2
    assert cb.used == 0 and cb.stop == "probe_cap"


def test_lane_publishes_every_accepted_route(tmp_path):
    """End-to-end half of the 2026-10-16 finding: the class's realization
    envelope bounds the tie MASS an enumeration may carry, never the route
    list. Here A->B->T realizes 50 bodyId paths against a 20-path envelope
    and A->C->T is a weaker-but-valid sibling: BOTH must appear in
    `route_paths_type.csv`, each with at least one witness. The old pass
    `break`ed on the spent envelope and silently dropped the sibling."""
    import pandas as pd
    root = tmp_path / "ds_fat"
    root.mkdir()
    pre, post, w = [], [], []
    for i in range(50):
        pre += ["a1", f"b{i}"]
        post += [f"b{i}", "t1"]
        w += [100 + i, 200 + i]
    pre += ["a1", "c1"]
    post += ["c1", "t1"]
    w += [10, 10]
    pl.DataFrame({"bodyId_pre": pre, "bodyId_post": post,
                  "weight": w}).write_parquet(root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "t1", "c1"]
                         + [f"b{i}" for i in range(50)],
                  "type": ["A", "T", "C"] + ["B"] * 50}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_fat"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=20,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out)
    rows = pl.read_csv(out / "route_paths_type.csv")
    assert sorted(rows["path"].to_list()) == ["A->B->T", "A->C->T"]
    by = {r["path"]: r for r in rows.to_dicts()}
    # the envelope truncated the FAT walk's witness sample; the sibling
    # still got its guaranteed one
    assert by["A->B->T"]["realized_paths"] == 20
    assert by["A->C->T"]["realized_paths"] == 1
    prov = json.loads((out / "route_provenance.json").read_text())
    cls = prov["classes"]["A->T@h2"]
    assert cls["accepted"] == 2 == cls["used"]
    assert cls["counted"] == "routes" and cls["route_share"] == 15
    assert cls["realization_envelope"] == 20
    assert cls["stop"] == "exhausted"
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    assert len(body) == 21


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


def test_lane_adj_denominator_is_all_pres(tmp_path):
    """F-EX-003 D3b: the lane's CONDITIONED denominator must be the ALL-PRES
    incoming of the edges it admits, exactly like `connection_type.csv`'s
    `connection_ratio_adj` — not its own `typed` frame, which has already
    dropped untyped presynaptic bodies. Measured on MCNS: PPL101 38,866
    (typed-pres) vs 39,137 (all-pres) at asked=1, which broke the lane's own
    documented law that adj == plain there.

    The fixture therefore gives the intermediate body an UNTYPED presynaptic
    neighbour: a typed-only universe cannot express the difference, which is
    why the existing asked=1 assertion below stayed green against the buggy
    denominator."""
    import pandas as pd
    root = tmp_path / "ds_untyped"
    root.mkdir()
    # a1 -> b1 -> c1, plus u1 (NO type) -> b1. u1's 90 synapses are part of
    # B's dataset incoming but are dropped from the lane's `typed` frame.
    pl.DataFrame({"bodyId_pre": ["a1", "b1", "u1"],
                  "bodyId_post": ["b1", "c1", "b1"],
                  "weight": [100, 80, 90]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "b1", "c1", "u1"],
                  "type": ["A", "B", "C", ""]}).to_parquet(
        root / "neurons.parquet", index=False)

    def _lane(asked, name):
        out = tmp_path / name
        assert run_lane(None, asked=asked, max_interlayer=2,
                        source_types=["A"], target_types=["C"],
                        connections_path=root / "connections.parquet",
                        neuron_table=root / "neurons.parquet",
                        out_dir=out)
        return pl.read_csv(out / "route_paths_type.csv").row(0, named=True)

    # asked = 1: nothing is conditioned away, so the adjusted list IS the
    # plain one (both divide by B's 190 / C's 80).
    one = _lane(1, "lane_t1")
    assert json.loads(one["ratios"]) == json.loads(one["ratios_adj"])
    assert float(one["min_ratio"]) == float(one["min_ratio_adj"])
    w = json.loads(one["weights"])
    assert abs(w[0] / json.loads(one["ratios"])[0] - 190.0) < 1e-6, one
    assert abs(w[1] / json.loads(one["ratios"])[1] - 80.0) < 1e-6, one

    # asked = 60: conditioning keeps a1->b1 (100) AND u1->b1 (90) — the
    # untyped body's mass belongs to the denominator even though the body
    # can never be a graph node — so adj > plain and its implied denominator
    # stays the all-pres 190, not the typed-only 100.
    six = _lane(60, "lane_t60")
    r_plain, r_adj = (json.loads(six["ratios"]), json.loads(six["ratios_adj"]))
    w60 = json.loads(six["weights"])
    assert abs(w60[0] / r_plain[0] - 190.0) < 1e-6, six
    assert abs(w60[0] / r_adj[0] - 190.0) < 1e-6, six
    assert r_adj[0] >= r_plain[0]


def test_lane_denominator_keeps_excluded_autapses(tmp_path):
    """An excluded body autapse still belongs to the engine's all-pres
    denominator, exactly like connection_type.csv's ratio."""
    import pandas as pd
    root = tmp_path / "ds_autapse"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "b1", "c1"],
        "bodyId_post": ["b1", "c1", "c1"],
        "weight": [100.0, 80.0, 100.0],
    }).write_parquet(root / "connections.parquet")
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
        "path_mode:                  all\n"
        "dataset:                    synth\n"
        "exclude autapses:           True\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    row = pl.read_csv(
        run / "route_scope_paths" / "route_paths_type.csv").row(
            0, named=True)
    assert json.loads(row["weights"]) == [100.0, 80.0]
    # C's all-pres incoming is 80 (b1->c1) + 100 (c1->c1 autapse) = 180.
    assert json.loads(row["ratios"])[-1] == 80.0 / 180.0
    assert json.loads(row["ratios_adj"])[-1] == 80.0 / 180.0


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
    # R3: the lane records BOTH the caller's mode and the resolved run mode
    assert prov["corridor_mode"] == "both" and prov["run_mode"] == "both"
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
    # Decisions 3 (F-EX-001): the walk row's mass columns are the CHAIN's
    # per-position restricted mass — every realizing bodyId edge of this type
    # sequence, not the sum over the budgeted sample of paths. The class kept
    # 100 of 200 realizations, so the SAMPLED witness union (24950/34950, the
    # strongest 100 paths' distinct edges) is now a separate, named column and
    # must never exceed the chain mass beside it (gate L6a).
    assert json.loads(row["weights"]) == [39900, 59900]
    assert row["coverage"] == "[1/1, 200/200, 1/1]"   # chain-level n/N
    assert row["min_weight"] == 39900
    assert json.loads(row["sampled_support_mass"]) == [24950, 34950]
    assert row["sampled_coverage"] == "[1/1, 100/200, 1/1]"
    assert row["realized_paths"] == 100                # the sample, disclosed
    s_w = json.loads(row["sampled_support_mass"])
    assert all(s <= c for s, c in zip(s_w, json.loads(row["weights"])))
    assert len(json.loads(row["ratios"])) == 2
    # ratios are the CHAIN mass over the post type's all-pres incoming:
    # B's incoming is the a1->b_i family (39900) and C's the b_i->c1 family
    # (59900), so this walk saturates both shares at 1.0.
    assert json.loads(row["ratios"]) == [1.0, 1.0]
    # C1.2: adj ratios present; at asked=1 adj == plain BY CONSTRUCTION
    assert json.loads(row["ratios_adj"]) == json.loads(row["ratios"])
    assert row["min_ratio_adj"] == row["min_ratio"]
    cls = prov["classes"]["A->C@h2"]
    # ROUTE-COUNTED ACCEPTANCE (2026-10-16): this fixture has ONE type walk
    # (its 200 realizations no longer buy the class), so the honest stop is
    # 'exhausted' — the drain held nothing else — while the realization
    # envelope still bites where it always did: in `emitted` and in the
    # row's `realized_paths` above.
    assert cls["stop"] == "exhausted" and cls["emitted"] == 100
    assert cls["accepted"] == 1 == cls["used"]
    assert cls["counted"] == "routes"
    assert cls["realization_envelope"] == 100
    assert cls["route_share"] == 15
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
    plain = json.loads(row["sampled_ratios"])[0]
    # The PAIR table's ratios divide the lane's selected support (its own
    # documented `selected_support_mass`/`realized_mass`), so they match the
    # witness readout, not the chain-level `ratios` column (F-EX-001: level 3
    # belongs on the pair table, level chain belongs on the row).
    assert abs(ab["connection_ratio"][0] - round(plain, 6)) < 1e-9
    assert ab["selected_support_mass"][0] == ab["realized_mass"][0] == 24950
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
    assert prov["validation_basis"] == "shortest_store_edge_union"
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


def test_tie_truncation_confirmed_on_real_tie(tmp_path):
    """F-PF-006: a share break that truly cuts an equal-strength route
    reports ties_truncated_confirmed — and the FINAL provenance snapshot
    carries it (the audit's hard-cut-serialized-as-zero bug)."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "a", "b1", "b2"],
        "bodyId_post": ["b1", "b2", "t", "t"],
        "weight": [10, 10, 10, 10]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b1", "b2", "t"],
                  "type": ["A", "B", "B", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    rows = pl.read_csv(out / "route_paths_type.csv")
    assert rows.height == 1
    prov = json.loads((out / "route_provenance.json").read_text())
    cls = prov["classes"]["A->T@h2"]
    assert cls["ties_truncated"] == 1
    assert cls["ties_truncated_confirmed"] == 1
    assert cls["ties_truncated_potential"] == 0


def test_tie_truncation_absent_single_route(tmp_path):
    """F-PF-006: exactly one realization — the share cut nothing, so no
    tie flag may fire."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "b"],
        "bodyId_post": ["b", "t"],
        "weight": [10, 10]}).write_parquet(root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b", "t"],
                  "type": ["A", "B", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    prov = json.loads((out / "route_provenance.json").read_text())
    cls = prov["classes"]["A->T@h2"]
    assert cls["ties_truncated"] == 0
    assert cls["ties_truncated_confirmed"] == 0
    assert cls["ties_truncated_potential"] == 0


def test_tie_truncation_cleared_on_unequal_next(tmp_path):
    """F-PF-006: the next realization strictly weaker than the share's
    tau is NOT a tie cut — the flag must clear (the old code set it on
    every share break)."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "a", "b1", "b2"],
        "bodyId_post": ["b1", "b2", "t", "t"],
        "weight": [10, 5, 10, 5]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b1", "b2", "t"],
                  "type": ["A", "B", "B", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    prov = json.loads((out / "route_provenance.json").read_text())
    cls = prov["classes"]["A->T@h2"]
    assert cls["ties_truncated"] == 0
    assert cls["ties_truncated_confirmed"] == 0
    assert cls["ties_truncated_potential"] == 0
    rows = pl.read_csv(out / "route_paths_type.csv")
    assert rows.height == 1          # the strong route; weak one over share


def test_pop_cap_reports_pop_capped_never_exhausted():
    """F-PF-007: a drain stopped by the pop budget names the cause —
    a pulled class is 'pop_capped', an unpulled hungry class is 'capped'
    (never 'exhausted'/'dry', which claim a complete search / proven
    absence)."""
    adj = {"A": [("B", 10), ("C", 9)], "B": [("T", 8)], "C": [("T", 7)]}
    drain = CorridorDrain(adj, "A", {"T"}, 2, max_pops=3)
    classes = {(("A", "T"), 1): ClassBudget(limit=5),
               (("A", "T"), 2): ClassBudget(limit=5)}
    prov = validate_and_fill(drain, classes,
                             lambda walk, cap: (1, False),
                             max_pulls=drain.max_pops)
    assert prov["drain_pop_capped"] is True
    assert prov["drain_exhausted"] is False
    stops = {k: v["stop"] for k, v in prov["classes"].items()}
    # both classes name the cause (the pulled/unpulled distinction lives
    # in the class's pulled counter, not in a different stop label)
    assert stops["A->T@h2"] == "pop_capped"
    assert stops["A->T@h1"] == "pop_capped"
    assert "exhausted" not in stops.values()
    assert "dry" not in stops.values()


def test_pair_export_names_both_support_sets(tmp_path):
    """F-PF-008: route_connection_type.csv explicitly names its two
    support sets — weight/universe_type_weights (the whole admitted
    type-pair mass; renamed from universe_pair_mass by
    plan-corridor-refill-and-mode-tabs F4/Q3, same quantity) vs
    selected_support_mass/realized_mass (the lane's emitted witnesses) —
    and they differ exactly when the universe is bigger than the selected
    support."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "a", "b1", "b2"],
        "bodyId_post": ["b1", "b2", "t", "t"],
        "weight": [10, 10, 10, 10]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b1", "b2", "t"],
                  "type": ["A", "B", "B", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    conn = pl.read_csv(out / "route_connection_type.csv")
    for col in ("universe_type_weights", "selected_support_mass",
                "realized_mass"):
        assert col in conn.columns
    ab = conn.filter((pl.col("type_pre") == "A")
                     & (pl.col("type_post") == "B")).row(0, named=True)
    # universe: a->b1 + a->b2 = 20; the selected walk witnesses a->b1 = 10
    assert ab["weight"] == ab["universe_type_weights"] == 20
    assert ab["selected_support_mass"] == ab["realized_mass"] == 10
    bt = conn.filter((pl.col("type_pre") == "B")
                     & (pl.col("type_post") == "T")).row(0, named=True)
    assert bt["universe_type_weights"] == 20
    assert bt["selected_support_mass"] == 10


def test_raw_body_simple_engine_yields_only_simple_paths():
    """F-PF-003 (engine level): with raw_body_simple the layered
    generator spends its budget on body-simple paths only — the audit's
    fixture (invalid 100 walk a1->b1->a1->t vs valid 90 walk
    a1->b2->a2->t) yields the 90 path under budget 1 instead of
    exhausting the budget on the invalid 100 walk."""
    from vispath_pkg.fast_graph_core import FastGraph
    g = FastGraph()
    g.build_from_edges([
        # layered, position-encoded: hop0 A->B, hop1 B->A, hop2 A->T
        ("a1\x1f0", "b1\x1f1", 100.0), ("a1\x1f0", "b2\x1f1", 90.0),
        ("b1\x1f1", "a1\x1f2", 100.0), ("b2\x1f1", "a2\x1f2", 90.0),
        ("a1\x1f2", "t\x1f3", 100.0), ("a2\x1f2", "t\x1f3", 90.0),
    ])
    paths = list(g.find_paths_strongest_first(
        ["a1\x1f0"], {"t\x1f3"}, 3, budget=1, raw_body_simple=True))
    assert len(paths) == 1
    assert [n.split("\x1f")[0] for n in paths[0]] == \
        ["a1", "b2", "a2", "t"]
    # and the raw weights follow the simple route, not the invalid one
    st: dict = {}
    list(g.find_paths_strongest_first(
        ["a1\x1f0"], {"t\x1f3"}, 3, budget=1, stats=st,
        raw_body_simple=True))
    # tau = 90 (the only SIMPLE realization); nothing valid was dropped,
    # so the budget did not bite -- the old code burned the budget on the
    # invalid 100 walk and emitted nothing at all
    assert st["tau"] == 90.0 and st["budget_bitten"] is False


def test_corridor_multi_visit_emits_valid_realization(tmp_path):
    """F-PF-003 (corridor level): allow_multi_visit=True with share 1 —
    the valid weaker realization of the repeated-type walk is emitted
    (the audit probe recorded accepted=1, used=1, emitted=0)."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "b1", "b2", "a1", "a2"],
        "bodyId_post": ["b1", "b2", "a1", "a2", "t", "t"],
        "weight": [100, 90, 100, 90, 100, 90]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a1", "a2", "b1", "b2", "t"],
                  "type": ["A", "A", "B", "B", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    out = tmp_path / "lane_out"
    assert run_lane(None, asked=1, max_interlayer=3,
                    per_pair_length_budget=1, allow_multi_visit=True,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    body = pl.read_csv(out / "route_paths_bodyId.csv")
    emitted = [p.split("->") for p in body["path"].to_list()]
    # the strong invalid walk (a1->b1->a1->t) must NOT appear; the valid
    # 90 realization must
    assert ["a1", "b1", "a1", "t"] not in emitted
    assert ["a1", "b2", "a2", "t"] in emitted


def test_shortest_store_enforces_per_endpoint_minimum_hops(tmp_path):
    """F-PF-009 (D3-B): with the store's per-target distance maps the
    lane enforces per-endpoint minimum hops — the audit's two-target
    fixture emits a->t1, b->t1 and the MINIMUM a->b->t2, and rejects the
    nonminimum a->b->t1 that the flattened union used to emit while
    claiming shortest semantics."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "b", "b", "a"],
        "bodyId_post": ["t1", "t1", "t2", "b"],
        "weight": [5, 5, 5, 5]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b", "t1", "t2"],
                  "type": ["S", "S", "T", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['S']\n"
        "target neurons:             ['T']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  shortest\n"
        "dataset:                    synth\n")
    store = run / "shortest_discovery_store"
    (store / "connections").mkdir(parents=True)
    (store / "node_distances").mkdir(parents=True)
    (store / "meta.json").write_text('{"targets_found": ["t1", "t2"]}')
    pl.DataFrame({
        "bodyId_pre": ["a", "b", "b", "a"],
        "bodyId_post": ["t1", "t1", "t2", "b"],
        "weight": [5.0, 5.0, 5.0, 5.0]}).write_parquet(
        store / "connections" / "connections_L0.parquet")
    pl.DataFrame({
        "target": ["t1", "t1", "t2", "t2"],
        "node": ["a", "b", "b", "a"],
        "dist": [1, 1, 1, 2]}).write_parquet(
        store / "node_distances" / "node_distances_L0_c0.parquet")
    # the nonminimum a->b->t1 route rides the type-revisiting walk
    # S->S->T, which only exists under the multi-visit knob (the default
    # planner rule never plans type-revisiting walks at all)
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    allow_multi_visit=True)
    prov = json.loads(
        (run / "route_scope_paths" / "route_provenance.json").read_text())
    assert prov["validation_basis"] == "shortest_dag_store_enforced"
    assert prov["shortest_enforced"] is True
    body = pl.read_csv(run / "route_scope_paths" / "route_paths_bodyId.csv")
    paths = set(body["path"].to_list())
    assert "a->t1" in paths and "b->t1" in paths
    assert "a->b->t2" in paths          # the minimum 2-hop route to t2
    assert "a->b->t1" not in paths      # nonminimum: d_t1(a) == 1


def test_certified_mode_exports_the_strongest_actual_route(tmp_path):
    """F-PF-002 (D2-B): with certify_actual the class keeps accepting
    candidates past the share until the first skipped candidate's bound
    drops below the weakest exported path's ACTUAL bottleneck -- the
    audit's displacement fixture (a bound-100 walk whose realization is
    bottleneck-1, against a bound-90 walk realizing bottleneck-90, share
    1) exports the 90 route.

    The bound-guided default no longer DROPS the weaker-bound route: since
    the 2026-10-16 fix its budget counts type-level routes and every
    accepted route keeps one witness, so BOTH corridors are published and
    the displacement shows up as RANK (the bound-100 walk leads with a
    realized bottleneck of 1) instead of as an absence. certify_actual is
    the mode that resolves the share to the strongest ACTUAL route."""
    import pandas as pd
    root = tmp_path / "ds"
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": ["a", "b1", "a", "b2", "a", "c"],
        "bodyId_post": ["b1", "t", "b2", "t", "c", "t"],
        "weight": [100, 1, 1, 100, 90, 90]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({"bodyId": ["a", "b1", "b2", "c", "t"],
                  "type": ["A", "B", "B", "C", "T"]}).to_parquet(
        root / "neurons.parquet", index=False)

    out_bound = tmp_path / "bound"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out_bound)
    body_bound = pl.read_csv(
        out_bound / "route_paths_bodyId.csv")["path"].to_list()
    assert body_bound == ["a->b1->t", "a->c->t"], (
        "UB-descending acceptance, one witness per accepted route")
    rows_bound = pl.read_csv(out_bound / "route_paths_type.csv").to_dicts()
    assert [r["path"] for r in rows_bound] == ["A->B->T", "A->C->T"]
    assert rows_bound[0]["plan_rank"] == 1 and rows_bound[0]["route_ub"] == 100
    # the displacement is now a RANKING fact, disclosed by the realized
    # bottleneck beside the optimistic bound
    assert rows_bound[0]["applied_tau"] == 1 and rows_bound[1]["applied_tau"] == 90
    cls_bound = json.loads(
        (out_bound / "route_provenance.json").read_text())["classes"]["A->T@h2"]
    assert cls_bound["accepted"] == 2 and cls_bound["used"] == 2
    assert cls_bound["realization_envelope"] == 1

    out_cert = tmp_path / "cert"
    assert run_lane(None, asked=1, max_interlayer=2,
                    per_pair_length_budget=1,
                    certify_actual=True,
                    source_types=["A"], target_types=["T"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out_cert)
    body_cert = pl.read_csv(
        out_cert / "route_paths_bodyId.csv")["path"].to_list()
    assert body_cert == ["a->c->t"]            # the strongest ACTUAL route
    prov = json.loads((out_cert / "route_provenance.json").read_text())
    assert prov["knobs"]["certify_actual"] is True
    cls = prov["classes"]["A->T@h2"]
    assert cls["certified"] is True
    assert cls["weakest_emitted"] == 90


def test_lane_ratio_basis_adj_equals_plain_by_construction(tmp_path):
    """Gate L1 on the RATIO basis. The engine conditions
    `connection_ratio_adj` at the MIN-SYNAPSE tier (`thr = max(1,
    min_synapse_num)`), and a ratio-basis run forces that to 1 — so its
    adjusted denominator is the plain all-inputs one and adj == plain for
    EVERY row. Dividing by the ratio-gated cone instead (what `conn_r` is)
    is the emitted-subset mistake F-EX-003 was: measured on the real
    L1r0.001 lane, 0 of 8 rows had adj == plain.

    The fixture cuts the ratio gate at 0.005, which admits only the
    stronger half of the b_i->c1 family — so the cone denominator (34,950)
    differs from the dataset one (59,900) and the old code cannot pass."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "parameters.txt").write_text(
        "weight basis:              connection_ratio\n"
        "min connection ratio:      0.005\n"
        "min synapse number:        1\n"
        "max interlayer:            2\n"
        "source neurons:            ['a1']\n"
        "target neurons:            ['c1']\n", encoding="utf-8")
    root = _write_synth_dataset(tmp_path)
    assert run_lane(out, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = out / "route_scope_paths"
    rows = pl.read_csv(lane / "route_paths_type.csv")
    assert rows.height >= 1
    prov = json.loads((lane / "route_provenance.json").read_text())
    assert prov["weight_basis"] == "connection_ratio"
    assert rows.filter(pl.col("ratios") != pl.col("ratios_adj")).height == 0
    for row in rows.to_dicts():
        assert json.loads(row["ratios_adj"]) == json.loads(row["ratios"])
        # the plain share is a DATASET share: chain mass / all-pres
        # incoming. C's incoming is the FULL b_i->c1 family (59,900),
        # never the ratio-gated 34,950.
        w = json.loads(row["weights"])
        r = json.loads(row["ratios"])
        assert abs(w[1] / r[1] - 59900.0) < 1e-6, row


def test_lane_full_scope_refill_second_round(tmp_path):
    """Q2 (plan-corridor-refill-and-mode-tabs): the lane's second round
    re-values ITS OWN chains at synapse >= 1 through the refill's layered
    filter, so a Fast run carries the same scope pair the Complete mode
    does. The fixture asks 150, which cuts the weaker half of both hop
    families, so there IS mass below the lane's tier to recover."""
    out = tmp_path / "lane_full"
    assert run_lane(None, asked=150, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=(root := _write_synth_dataset(
                        tmp_path, n_intermediate=60)) / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    out_dir=out)
    asked_rows = pl.read_csv(out / "route_paths_type.csv")
    comp = pl.read_csv(out / "route_full_paths_type.csv")
    carr = pl.read_csv(out / "route_full_carrying_paths_type.csv")
    pairs = pl.read_csv(out / "route_full_type_pairs.csv")
    inc = pl.read_csv(out / "route_type_incoming.csv")
    prov = json.loads((out / "route_provenance.json").read_text())

    # the SAME chain set the lane validated at its own tier
    assert set(comp["path"]) == set(asked_rows["path"])
    assert comp.height > 0 and carr.height > 0
    # mirror column contract (Q5): the complete mode's names, lane namespace
    assert comp.columns[-2:] == ["emitted_weights", "refill_weights"]
    assert carr.columns[-2:] == ["emitted_weights", "total_weights"]
    assert pairs.columns == ["type_pre", "type_post", "nt_type",
                             "emitted_weight",
                             "refill_weight", "refilled_total",
                             "refilled_connection_ratio",
                             "universe_type_weights",
                             "pair_offchain_mass"]
    # `Universe type weights` (F4/Q3 rename of universe_pair_mass) is the
    # >= 1 universe the hop pair is drawn from: realized + off-chain == it,
    # so every printed chain mass is provably a lower bound of its pair.
    for row in pairs.iter_rows(named=True):
        assert (float(row["refilled_total"])
                + float(row["pair_offchain_mass"])
                == pytest.approx(float(row["universe_type_weights"]))), row
        assert float(row["emitted_weight"]) + float(row["refill_weight"]) \
            == pytest.approx(float(row["refilled_total"])), row

    # gate G3: emitted + refill == total, hop for hop, integer-valued
    for row in comp.iter_rows(named=True):
        e = json.loads(row["emitted_weights"])
        rf = json.loads(row["refill_weights"])
        w = json.loads(row["weights"])
        assert [a + b for a, b in zip(e, rf)] == w, row
        assert all(float(v).is_integer() for v in e + rf + w), row
        assert any(b > 0 for b in rf) or row["path"] not in set(carr["path"])
    # the carrying table states its SUBJECT and reconciles beside it
    for row in carr.iter_rows(named=True):
        e = json.loads(row["emitted_weights"])
        sub = json.loads(row["weights"])
        tot = json.loads(row["total_weights"])
        assert [a + b for a, b in zip(e, sub)] == tot, row
        assert any(b > 0 for b in sub), row

    # gate G4/G5: one denominator file reproduces every ratio, and the
    # >= 1 scope carries NO adjusted column (conditioning at 1 is a no-op)
    assert "ratios_adj" not in comp.columns
    assert "ratios_adj" not in carr.columns
    den = dict(zip(inc["type_post"], inc["incoming_threshold"]))
    assert den["C"] > 0
    for row in comp.iter_rows(named=True):
        seq = row["path"].split("->")
        w = json.loads(row["weights"])
        r = json.loads(row["ratios"])
        for i, m in enumerate(w):
            assert abs(m / den[seq[i + 1]] - r[i]) < 1e-12, row

    full = prov["full"]
    assert full["status"] == "ok" and full["scope"] == "synapse >= 1"
    assert full["chains"] == comp.height
    assert full["chains_carrying"] == carr.height
    assert full["refill_weight_total"] > 0
    # the pair table agrees with the path tables' totals, per hop pair
    assert (pairs["emitted_weight"] + pairs["refill_weight"]
            == pairs["refilled_total"]).all()
    assert (pairs["pair_offchain_mass"] >= 0).all()
    # the mirror pages exist for the second round too
    assert (out / "route_full_type_network.html").exists()
    assert (out / "route_full_type_heatmap.html").exists()
    assert not (out / ".mirror_visualizations_failed").exists()


def test_lane_full_scope_refill_at_floor_is_a_no_op(tmp_path):
    """asked == 1: the lane's tier IS the floor, so nothing lies below it —
    the second round must say so rather than print a table of zeros or
    fail. (Gate G3's other face.)"""
    out = tmp_path / "lane_floor"
    root = _write_synth_dataset(tmp_path, n_intermediate=20)
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out)
    prov = json.loads((out / "route_provenance.json").read_text())
    comp = pl.read_csv(out / "route_full_paths_type.csv")
    carr = pl.read_csv(out / "route_full_carrying_paths_type.csv")
    assert carr.height == 0
    assert prov["full"]["chains_carrying"] == 0
    assert prov["full"]["status"] == "nothing_below_tier"
    # the asked-tier masses are unchanged by the >= 1 pass at the floor
    for row in comp.iter_rows(named=True):
        assert all(v == 0 for v in json.loads(row["refill_weights"])), row


# ---------------------------------------------------------------------------
# Round-2 parity laws (plan §4.2 cache guard, §4.3 time budget): the lane's
# synapse >= 1 pass must refuse an incomplete connection source and must say
# when the clock cut it — never publish an undercounting universe or a
# partial table that reads as complete.
# ---------------------------------------------------------------------------

def _lane_full_fixture(tmp_path, emitted_incoming):
    """A run root with an emitted-tier incoming file beside the lane folder,
    plus a lane run that produces a second round."""
    ds = _write_synth_dataset(tmp_path, n_intermediate=60)
    (tmp_path / "data_details").mkdir(exist_ok=True)
    (tmp_path / "data_details" / "type_incoming_emitted.csv").write_text(
        "type,incoming\n" + "".join(f"{t},{v}\n"
                                    for t, v in emitted_incoming.items()),
        encoding="utf-8")
    out = tmp_path / "route_scope_paths"
    ok = run_lane(None, asked=150, max_interlayer=2,
                  source_types=["A"], target_types=["C"],
                  connections_path=ds / "connections.parquet",
                  neuron_table=ds / "neurons.parquet", out_dir=out)
    return out, ok


def test_lane_full_scope_cache_guard_passes_and_is_stamped(tmp_path):
    """§4.2 key parity: the >= 1 all-post incoming must be checked against
    the run's own emitted-tier incoming, and the result stamped. A clean run
    records `ok`; the comparison is the SAME invariant the complete mode's
    full scope enforces."""
    out, ok = _lane_full_fixture(tmp_path, {"C": 1.0})
    assert ok
    prov = json.loads((out / "route_provenance.json").read_text())
    full = prov.get("full") or {}
    assert full.get("cache_guard") == "ok", full
    assert full.get("time_capped") is False
    assert (out / "route_full_paths_type.csv").exists()


def test_lane_full_scope_cache_guard_refuses_an_undercounting_source(
        tmp_path):
    """The negative control: when the run's own emitted-tier incoming EXCEEDS
    what the lane's connection source can account for, the source is
    incomplete relative to the run. The tables must NOT be written (they
    would undercount the universe silently) and the refusal must name the
    type and both numbers."""
    out, ok = _lane_full_fixture(tmp_path, {"C": 1e12})
    assert ok, "the lane itself still completes; only the round refuses"
    prov = json.loads((out / "route_provenance.json").read_text())
    full = prov.get("full") or {}
    assert full.get("status") == "failed", full
    assert full.get("cache_guard") == "FAILED"
    assert "C" in full.get("error", "")
    assert "cache guard FAILED" in full.get("error", "")
    for name in ("route_full_paths_type.csv",
                 "route_full_carrying_paths_type.csv",
                 "route_full_type_pairs.csv"):
        assert not (out / name).exists(), \
            f'{name} was written despite the guard refusal'


def test_lane_full_scope_skips_on_a_spent_time_budget(tmp_path):
    """§4.3: the pass shares the lane's clock. A lane whose budget is spent
    before the round starts must DISCLOSE the skip (never silently drop the
    round), and it must return before touching the connections."""
    import time as _time

    import route_plan as rp

    out = tmp_path / "route_scope_paths"
    out.mkdir(parents=True)
    block = rp._full_scope_refill(
        out, conn=None, tmap=None, body2type=None, src_ids=set(),
        tgt_ids=set(), walk_rows=[{"path": "A->B->C"}], typed=None,
        type_membership={}, type_incoming={}, asked=3,
        time_budget_s=0.001, t_start=_time.monotonic() - 1.0)
    assert block and block.get("status") == "skipped_time_budget", block
    assert "budget" in (block.get("note") or "")
    assert not (out / "route_full_paths_type.csv").exists()


def test_lane_mirror_pages_skip_when_the_run_has_its_own(tmp_path):
    """REDUNDANCY (user, 2026-10-15): the lane's mirror pages drew a strict
    SUBSET of the run's own Network in the same style (measured on the
    user's run: 26 of 38 pairs), so they are skipped while that page exists —
    and the skip is DISCLOSED, because a reader who finds no html in
    route_scope_paths/ must not think the lane forgot it. A fast-only run
    (which has no main visualization) still gets them."""
    import pandas as pd

    import route_plan
    out = tmp_path / "route_scope_paths"
    out.mkdir()
    pd.DataFrame([
        {'type_pre': 'A', 'type_post': 'B', 'selected_support_mass': 5.0,
         'connection_ratio': 0.1, 'nt_type': 'gaba'},
    ]).to_csv(out / "route_connection_type.csv", index=False)

    outcome = route_plan._write_mirror_visualizations(
        out, src_types=["A"], tgt_types=["B"], adj_ratio_tier=3)
    assert outcome == "built", outcome
    asked = (out / "route_type_network.html").read_text(encoding="utf-8")
    assert '"nt_type"' in asked and 'gaba' in asked, \
        'the lane mirror lost the transmitter (NT parity, issue 6)'

    # the run grows its own visualization -> the second call skips
    viz = tmp_path / "visualization"
    viz.mkdir()
    (viz / "Network_run.html").write_text("<html></html>", encoding="utf-8")
    for stale in out.glob("route_type_*.html"):
        stale.unlink()
    outcome2 = route_plan._write_mirror_visualizations(
        out, src_types=["A"], tgt_types=["B"], adj_ratio_tier=3)
    assert "skipped" in outcome2, outcome2
    assert not (out / "route_type_network.html").exists()
    assert not (out / "route_type_heatmap.html").exists()

def test_lane_full_scope_refill_on_the_ratio_basis(tmp_path):
    """G3's "integer-valued ON ANY BASIS", for the lane's second round: the
    ratio basis changes ADMISSION, never the unit. At the >= 1 floor a ratio
    lane's asked tier IS the floor (the run forces min synapse 1), so the
    honest outcomes are: masses integer-valued, no adjusted column, and — when
    the lane already saw every edge — the round says `nothing_below_tier`
    instead of inventing a band."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "parameters.txt").write_text(
        "weight basis:              connection_ratio\n"
        "min connection ratio:      0.005\n"
        "min synapse number:        1\n"
        "max interlayer:            2\n"
        "source neurons:            ['a1']\n"
        "target neurons:            ['c1']\n", encoding="utf-8")
    root = _write_synth_dataset(tmp_path)
    assert run_lane(out, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = out / "route_scope_paths"
    prov = json.loads((lane / "route_provenance.json").read_text())
    assert prov["weight_basis"] == "connection_ratio"
    full = prov.get("full") or {}
    assert full, "the second round did not run on a ratio lane"
    assert full.get("status") in ("ok", "nothing_below_tier")
    comp = pl.read_csv(lane / "route_full_paths_type.csv")
    assert comp.height >= 1
    assert "ratios_adj" not in comp.columns
    for row in comp.iter_rows(named=True):
        e = json.loads(row["emitted_weights"])
        rf = json.loads(row["refill_weights"])
        w = json.loads(row["weights"])
        assert [a + b for a, b in zip(e, rf)] == w, row
        assert all(float(v).is_integer() for v in e + rf + w), row
    carr = pl.read_csv(lane / "route_full_carrying_paths_type.csv")
    if carr.height == 0:
        assert full.get("status") == "nothing_below_tier", full
    else:
        for row in carr.iter_rows(named=True):
            e = json.loads(row["emitted_weights"])
            sub = json.loads(row["weights"])
            tot = json.loads(row["total_weights"])
            assert [a + b for a, b in zip(e, sub)] == tot, row


def test_lane_universe_is_the_dataset_ge1_pair_mass(tmp_path):
    """G4's substance, recomputed rather than read back: the corridor's
    `universe_type_weights` for a hop pair IS the dataset's whole >= 1 mass
    for that pair. Both lanes take their universe from the same definition, so
    this is what "one denominator across both lanes" means at the floor — and
    computing it here from the fixture's parquet (not from the table) is what
    keeps the law honest."""
    root = _write_synth_dataset(tmp_path, n_intermediate=40)
    out = tmp_path / "run"
    out.mkdir()
    (out / "parameters.txt").write_text(
        "min synapse number:        3\n"
        "max interlayer:            2\n"
        "source neurons:            ['a1']\n"
        "target neurons:            ['c1']\n", encoding="utf-8")
    assert run_lane(out, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    pairs = pl.read_csv(out / "route_scope_paths"
                        / "route_full_type_pairs.csv")
    assert pairs.height >= 1

    def type_of(bid):
        return 'A' if bid == 'a1' else ('C' if bid == 'c1' else 'B')

    want = {}
    for u, v, w in pl.read_parquet(root / "connections.parquet").iter_rows():
        key = (type_of(u), type_of(v))
        want[key] = want.get(key, 0.0) + float(w)
    for row in pairs.iter_rows(named=True):
        key = (row["type_pre"], row["type_post"])
        assert key in want, (key, sorted(want))
        assert float(row["universe_type_weights"]) == want[key], row


# ---------------------------------------------------------------- Phase B
def _typed_dataset(tmp_path, edges, types, name="ds"):
    """Write a bodyId edge table + neuron table for the typed-lane tests."""
    import pandas as pd
    root = tmp_path / name
    root.mkdir()
    pl.DataFrame({
        "bodyId_pre": [e[0] for e in edges],
        "bodyId_post": [e[1] for e in edges],
        "weight": [e[2] for e in edges],
    }).write_parquet(root / "connections.parquet")
    pd.DataFrame({
        "bodyId": list(types),
        "type": [types[b] for b in types],
    }).to_parquet(root / "neurons.parquet", index=False)
    return root


def _run_typed(root, out, **kw):
    opts = dict(asked=1, max_interlayer=2, source_types=["A"],
                target_types=["C"], connections_path=root / "connections.parquet",
                neuron_table=root / "neurons.parquet", out_dir=out,
                type_pruned=True, type_prune_basis="synapse",
                type_min_synapse=1, type_min_ratio_adj=0.0)
    opts.update(kw)
    assert run_lane(None, **opts)
    rows = pl.read_csv(out / "route_paths_type.csv")
    manifest = json.loads((out / "type_prune_manifest.json").read_text())
    return rows, manifest


def test_type_pruned_phantom_rejected(tmp_path):
    """Phase D.1: A->B and B->C exist, but the B bodies carrying the
    incoming hop are disjoint from the B bodies carrying the outgoing
    hop. No single B body realizes A->B->C, so the walk is a phantom and
    must not be exported. The realizable A->D->C stays."""
    # Every type pair below sits on SOME source->target body path, so the
    # lossless cone keeps it. But the A->B edge's B body (b1) and the
    # B->C edge's B body (b2) are disjoint — no single B body composes
    # A->B->C, so the TYPE walk is a phantom. A->B->D->C and A->D->B->C
    # are realizable; A->E->C is the clean control.
    edges = [
        ("a1", "b1", 100), ("b1", "d1", 90), ("d1", "c1", 80),
        ("a2", "d2", 70), ("d2", "b2", 60), ("b2", "c1", 50),
        ("a1", "e1", 40), ("e1", "c2", 30),
    ]
    types = {"a1": "A", "a2": "A", "b1": "B", "b2": "B",
             "d1": "D", "d2": "D", "e1": "E", "c1": "C", "c2": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out", max_interlayer=3)
    assert "A->B->C" not in set(rows["path"])
    assert {"A->B->D->C", "A->D->B->C", "A->E->C"} <= set(rows["path"])
    assert manifest["phantoms"] >= 1
    assert manifest["accepted_paths"] == rows.height
    assert manifest["type_pruned_complete"] is True
    assert manifest["completeness_reason"] == "exhausted"


def test_type_pruned_prunes_pair_and_enumerates_complete(tmp_path):
    """Phase D.2: the pair below the mass floor is removed from the
    candidate graph; the surviving A->E->C walk is enumerated and
    validated. `type_pruned_complete` is claimed only on exhaustion."""
    edges = [
        ("a1", "b1", 100), ("b1", "c1", 30),    # pruned (30 < 100)
        ("a1", "e1", 200), ("e1", "c1", 150),   # kept
    ]
    types = {"a1": "A", "b1": "B", "e1": "E", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out", type_min_synapse=100)
    assert set(rows["path"]) == {"A->E->C"}
    assert manifest["pairs_before"] == 4
    assert manifest["pairs_after"] == 3
    assert manifest["pairs_removed"] == 1
    assert manifest["candidate_walks"] == 1
    assert manifest["validated_walks"] == 1
    assert manifest["type_pruned_complete"] is True


def test_type_pruned_cap_forbids_completeness(tmp_path):
    """Phase D.6: a type-walk cap is a disclosed stop, never a complete
    claim."""
    edges = [
        ("a1", "e1", 200), ("e1", "c1", 150),
        ("a1", "f1", 180), ("f1", "c1", 140),
    ]
    types = {"a1": "A", "e1": "E", "f1": "F", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out", type_walk_cap=1)
    assert manifest["enumeration_capped"] is True
    assert manifest["type_pruned_complete"] is False
    assert manifest["completeness_reason"] == "type_walk_cap"
    assert rows.height == 1


def test_type_pruned_exact_realized_filter(tmp_path):
    """Phase D.2/P0-B: the pair prune is coarse; the exact gate is the
    REALIZED min_ratio_adj. A realizable but weak path is counted as
    filtered, not exported."""
    edges = [
        ("a1", "e1", 200), ("e1", "c1", 150),   # C share 150/170
        ("a1", "g1", 30), ("g1", "c1", 20),     # C share 20/170
    ]
    types = {"a1": "A", "e1": "E", "g1": "G", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, type_min_ratio_adj=0.5)
    assert set(rows["path"]) == {"A->E->C"}
    assert manifest["validated_walks"] == 2
    assert manifest["filtered_weak"] == 1
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["validated_walks"] == manifest["validated_walks"]
    assert all(float(v) > 0.5 for v in rows["min_ratio_adj"])
    # the exact gate is strictly greater; the weak path is absent
    assert "A->G->C" not in set(rows["path"])


def test_type_pruned_first_target_stops_the_walk(tmp_path):
    """Phase B.3: exactly one clock token at the final node — a target is
    recorded and never expanded through."""
    from route_plan import _enumerate_type_walks
    pairs = pl.DataFrame({
        "type_pre": ["A", "B", "C"],
        "type_post": ["B", "C", "D"],
        "w_sum": [1.0, 1.0, 1.0],
        "w_max": [1.0, 1.0, 1.0],
        "n": [1, 1, 1],
    })
    walks, stats = _enumerate_type_walks(
        pairs, ["A"], {"C"}, bound=3)
    assert stats["capped"] is False
    assert walks == [("A", "B", "C")]


def test_type_pruned_complete_matches_bruteforce_type_walk_set(tmp_path):
    """Phase D.2: on a toy graph the lane's candidate set equals an
    independent exhaustive simple-walk DFS over the pruned pair table."""
    edges = [
        ("a1", "e1", 200), ("e1", "c1", 150),
        ("a1", "f1", 180), ("f1", "c1", 140),
        ("a1", "e1", 10), ("e1", "f1", 5), ("f1", "c1", 5),
    ]
    types = {"a1": "A", "e1": "E", "f1": "F", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out", type_min_synapse=6)
    # independent type-level brute force over the same kept pairs
    pairs = [("A", "E"), ("E", "C"), ("A", "F"), ("F", "C")]
    adj = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
    found = set()

    def dfs(node, walk):
        for nxt in adj.get(node, ()):
            if nxt == "C":
                found.add(tuple(walk + (nxt,)))
                continue
            if nxt in walk:
                continue
            dfs(nxt, walk + (nxt,))

    dfs("A", ("A",))
    assert manifest["candidate_walks"] == len(found)
    assert manifest["enumeration_capped"] is False
    assert set(rows["path"]) <= {"->".join(w) for w in found}


def test_type_pruned_bounded_revisit_is_enumerated(tmp_path):
    """P1-I: the article's Complete tables carry type revisits realized by
    DIFFERENT bodies. The lane's default `bounded` policy enumerates
    A->B->A->C; `simple` does not. The revisit is a type-level event, not
    a bodyId autapse."""
    edges = [
        ("a1", "b1", 100), ("b1", "a2", 90), ("a2", "c1", 80),
    ]
    types = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(
        root, tmp_path / "out_bounded", max_interlayer=3)
    assert "A->B->A->C" in set(rows["path"])
    assert manifest["revisit_policy"] == "bounded"
    assert manifest["chains_revisit"] >= 1
    assert manifest["type_pruned_complete"] is True
    rows_s, manifest_s = _run_typed(
        root, tmp_path / "out_simple", max_interlayer=3,
        type_revisit_policy="simple")
    assert "A->B->A->C" not in set(rows_s["path"])
    assert manifest_s["revisit_policy"] == "simple"


def test_type_pruned_revisit_uses_layered_superset(tmp_path):
    """Issue 2026-10-23: a label-revisiting walk is validated by ONE
    layered forward/backward pass. The layered intersection overcounts a
    body reused at two positions, so its masses are a disclosed superset
    of body-simple realizations; no witness sample or envelope exists."""
    edges = [
        ("a1", "b1", 100), ("b1", "a2", 100), ("a2", "c1", 1),
        # The a1-returning route makes the layered A->C hop mass 101 and
        # the layered gate 101/101; it is not body-simple but IS in the
        # layered corridor semantics.
        ("b1", "a1", 100), ("a1", "c1", 100),
    ]
    types = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out_layered"
    out.mkdir()
    stale_full = out / "route_full_paths_type.csv"
    stale_full.write_text("stale\n", encoding="utf-8")
    rows, manifest = _run_typed(
        root, out, max_interlayer=3, type_min_ratio_adj=0.01)
    assert "A->B->A->C" in set(rows["path"])
    row = rows.filter(pl.col("path") == "A->B->A->C").row(0, named=True)
    assert row["chain_revisit"] is True
    assert abs(float(row["min_ratio_adj"]) - 1.0) < 1e-9
    assert row["ratio_adj_complete"] is True
    simple = rows.filter(pl.col("path") == "A->C").row(0, named=True)
    assert simple["chain_revisit"] is False
    # A->C is type-simple; the layered pass keeps both source bodies'
    # A->C edges, so its hop mass and conditioned denominator are 101/101.
    assert abs(float(simple["min_ratio_adj"]) - 1.0) < 1e-9
    # The layered contract has no sampled witness fields or envelope.
    for removed in ("exact_gate_basis", "realization_sample_truncated",
                    "sampled_support_mass", "sampled_ratios",
                    "sampled_coverage", "realized_paths",
                    "gate_min_ratio_adj"):
        assert removed not in rows.columns
    assert manifest["validation_basis"] == "layered_fwd_bwd"
    assert manifest["revisit_semantics"] == "layered_fwd_bwd_superset"
    assert manifest["chains_revisit"] >= 1
    assert "realization_truncated" not in manifest
    assert "realization_envelope" not in manifest
    assert not (out / "route_paths_bodyId.csv").exists()
    # The additive synapse >= 1 round is off by default for the typed lane;
    # stale artifacts from an older in-place run are removed.
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["full"]["status"] == "skipped_not_requested"
    assert not stale_full.exists()
    assert not (out / "route_full_carrying_paths_type.csv").exists()


def test_type_pruned_layered_revisit_needs_no_body_simple_witness(tmp_path):
    """The chosen contract is corridor/refill layered semantics: a walk
    whose forward∩backward corridor exists is validated even when every
    realization reuses a body (no body-simple witness). The old lane
    classified such a walk as a phantom."""
    edges = [
        ("a1", "b1", 100), ("b1", "a1", 100), ("a1", "c1", 100),
    ]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(
        root, tmp_path / "out_reuse", max_interlayer=3,
        type_min_ratio_adj=0.01)
    assert "A->B->A->C" in set(rows["path"])
    row = rows.filter(pl.col("path") == "A->B->A->C").row(0, named=True)
    assert row["chain_revisit"] is True
    # a1 is the only A body: the A->B->A->C route reuses it at positions
    # 0 and 2, so no body-simple realization exists.
    assert manifest["phantoms"] == 0
    assert manifest["unverified"] == 0
    assert manifest["revisit_semantics"] == "layered_fwd_bwd_superset"


def test_corridor_diverse_type_rows_spreads_and_trims_tie_mass():
    """The corridor/budgeted selector uses type edges, respects the class
    budget even when cutoff ties emitted more rows, and reports overflow."""
    from route_plan import _select_diverse_type_rows
    rows = [{"path": f"A->B->X{i}->C", "min_ratio_adj": 0.08}
            for i in range(10)]
    rows += [{"path": f"A->D{i}->C", "min_ratio_adj": 0.06}
             for i in range(3)]
    selected, stats = _select_diverse_type_rows(
        rows, quota=1, clock_types={"C"}, target=6)
    paths = [r["path"] for r in selected]
    assert len(paths) == 6
    assert sum(p.startswith("A->B->") for p in paths) == 3
    assert sum(p.startswith("A->D") for p in paths) == 3
    assert stats["overflow"] >= 1
    assert stats["max_edge_reuse_observed"] >= 2
    assert stats["clock_edge_quota"] == 1
    assert stats["max_clock_edge_reuse_observed"] == 0


def test_type_pruned_diverse_topk_spreads_a_dominant_edge(tmp_path):
    """Updated issue 2026-10-23 §2.3: pure top-k fills with the one strong
    first-hop edge; the diverse output defers/quota-fills across the
    alternatives and discloses overflow/strength loss. The pure table is
    unchanged (strength reference)."""
    edges = [("a1", "b1", 100)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    for i in range(10):
        x = f"x{i}"
        types[x] = f"X{i}"
        edges += [("b1", x, 90), (x, "c1", 80)]
    for i in range(3):
        d = f"d{i}"
        types[d] = f"D{i}"
        edges += [("a1", d, 70), (d, "c1", 60)]
    root = _typed_dataset(tmp_path, edges, types)
    pure_rows, _ = _run_typed(root, tmp_path / "out_pure", type_top_k=6)
    rows, manifest = _run_typed(
        root, tmp_path / "out_div", type_top_k=6,
        type_diversity="edge_quota", type_edge_reuse_fraction=0.1,
        type_diversity_pool=3.0)
    pure = set(pure_rows["path"])
    assert all(p.startswith("A->B->") for p in pure)
    diverse = pl.read_csv(tmp_path / "out_div" /
                           "route_paths_type_diverse.csv")
    dpaths = set(diverse["path"])
    assert any(p.startswith("A->D") for p in dpaths)
    assert len(dpaths) == 6
    div = manifest["diversity"]
    assert div["mode"] == "edge_quota"
    assert div["max_edge_reuse"] == 1
    assert div["overflow_paths"] >= 1
    assert div["max_edge_reuse_observed"] >= 2
    assert div["strength_loss_abs_max"] > 0
    assert div["distinct_edges_per_depth"]
    assert div["jaccard_basis"] == "bodyId_edges"
    assert (tmp_path / "out_div" / "route_topk_pairs_diverse.csv").exists()
    assert manifest["diversity_mode"] == "edge_quota"
    assert manifest["type_pruned_complete"] is False
    # The pure table is exactly the no-diversity top-k.
    assert set(rows["path"]) == pure
    assert not (tmp_path / "out_div" / "route_paths_bodyId.csv").exists()
    # Turning diversity off removes stale diverse artifacts in a re-run.
    _run_typed(root, tmp_path / "out_div", type_top_k=6)
    assert not (tmp_path / "out_div" /
                "route_paths_type_diverse.csv").exists()


def test_type_pruned_topk_matches_the_sorted_full_set(tmp_path):
    """Issue 2026-10-23 §4: a top-k run returns exactly the first k rows of
    the fully sorted exhaustive accepted set, certifies the bound, records
    the next candidate bound, and writes no witness/envelope artifacts."""
    edges = []
    types = {"a1": "A", "c1": "C"}
    for i, w in enumerate((100, 80, 70)):
        b = f"b{i}"
        types[b] = f"B{i}"
        edges += [("a1", b, w), (b, "c1", w - 5)]
    root = _typed_dataset(tmp_path, edges, types)
    full_rows, _ = _run_typed(root, tmp_path / "out_full")
    full = full_rows.sort(
        ["min_ratio_adj", "path"], descending=[True, False]).head(2)
    rows, manifest = _run_typed(
        root, tmp_path / "out_topk", type_top_k=2)
    assert rows["path"].to_list() == full["path"].to_list()
    assert rows["plan_rank"].to_list() == [1, 2]
    assert manifest["selection_mode"] == "strongest_first"
    assert manifest["type_top_k"] == 2
    assert manifest["topk_complete"] is True
    assert manifest["topk_certified"] is True
    assert manifest["selection_complete"] is True
    assert manifest["type_pruned_complete"] is False
    assert manifest["completeness_reason"] == "topk_certified"
    assert manifest["topk_bound"] is not None
    assert manifest["topk_next_bound"] is not None
    assert manifest["topk_next_bound"] < manifest["topk_bound"]
    assert manifest["accepted_paths"] == 2
    assert manifest["candidates_pulled"] >= 2
    topk_pairs = pl.read_csv(tmp_path / "out_topk" / "route_topk_pairs.csv")
    assert topk_pairs.height == 1
    pair = topk_pairs.row(0, named=True)
    assert pair["accepted"] == 2 and pair["topk"] == 2
    assert pair["stop_reason"] == "certified"
    assert pair["next_bound"] == manifest["topk_next_bound"]
    assert not (tmp_path / "out_topk" / "route_paths_bodyId.csv").exists()
    for removed in ("realization_envelope", "realization_truncated",
                    "sampled_support_mass", "realized_paths"):
        assert removed not in manifest
        assert removed not in rows.columns


def test_clock_relay_exports_both_views_from_one_search(tmp_path):
    """Issue 2026-10-23 §3: clock types may be intermediates; the first-
    clock, relay, no-intra and intra views come from one run."""
    edges = [
        ("s1", "a1", 100), ("a1", "b1", 90), ("b1", "c1", 80),
        ("s1", "c1", 70), ("a1", "n1", 60), ("n1", "b1", 50),
    ]
    types = {"s1": "S", "a1": "A", "b1": "B", "c1": "C", "n1": "N"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out_relay"
    rows, manifest = _run_typed(
        root, out, max_interlayer=4, clock_relay_mode=True,
        source_types=["S"], target_types=["A", "B", "C"])
    paths = set(rows["path"])
    assert "S->A->B->C" in paths
    relay = pl.read_csv(out / "route_paths_type_clock_relay.csv")
    row = relay.filter(pl.col("path") == "S->A->B->C").row(0, named=True)
    assert row["clock_sequence"] == "A,B,C"
    assert row["intra_clock_edges"] == 2
    assert row["first_clock"] == "A"
    assert row["clock_relay"] is True
    first = pl.read_csv(out / "route_paths_type_first_clock.csv")
    assert "S->A->B->C" not in set(first["path"])
    assert set(first["path"]) == {"S->A", "S->C"}
    no_intra = pl.read_csv(
        out / "route_paths_type_clock_relay_no_intra.csv")
    intra = pl.read_csv(out / "route_paths_type_clock_relay_intra.csv")
    assert "S->A->N->B" in set(no_intra["path"])
    assert "S->A->B->C" in set(intra["path"])
    assert "S->A->B->C" not in set(no_intra["path"])
    assert manifest["clock_relay_mode"] == "all"
    assert manifest["clock_relay_counts"]["first_clock_paths"] == 2
    assert manifest["clock_relay_counts"]["relay_paths"] == 4
    assert manifest["clock_sequence_counts"]["A,B,C"] == 2
    conn = pl.read_csv(out / "route_connection_type.csv")
    assert {"intra_clock", "clock_relay"} <= set(conn.columns)
    ab = conn.filter((pl.col("type_pre") == "A")
                     & (pl.col("type_post") == "B")).row(0, named=True)
    assert ab["intra_clock"] is True and ab["clock_relay"] is True


def test_clock_relay_first_clock_is_byte_identical_when_off(tmp_path):
    """The first-clock export is the legacy lane's file, byte-for-byte."""
    edges = [
        ("s1", "a1", 100), ("a1", "b1", 90), ("b1", "c1", 80),
        ("s1", "c1", 70), ("a1", "n1", 60), ("n1", "b1", 50),
    ]
    types = {"s1": "S", "a1": "A", "b1": "B", "c1": "C", "n1": "N"}
    root = _typed_dataset(tmp_path, edges, types)
    off = tmp_path / "out_off"
    relay = tmp_path / "out_relay"
    _run_typed(root, off, max_interlayer=4,
               source_types=["S"], target_types=["A", "B", "C"])
    _run_typed(root, relay, max_interlayer=4, clock_relay_mode=True,
               source_types=["S"], target_types=["A", "B", "C"])
    assert ((off / "route_paths_type.csv").read_bytes()
            == (relay / "route_paths_type_first_clock.csv").read_bytes())


def test_excluded_pairs_and_clock_clock_switch(tmp_path):
    """Explicit pair exclusions (including the clock-clock convenience)
    remove those hops from enumeration and are disclosed in the manifest."""
    edges = [
        ("s1", "a1", 100), ("a1", "b1", 90), ("b1", "c1", 80),
        ("s1", "c1", 70), ("a1", "n1", 60), ("n1", "b1", 50),
    ]
    types = {"s1": "S", "a1": "A", "b1": "B", "c1": "C", "n1": "N"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out_excl"
    rows, manifest = _run_typed(
        root, out, max_interlayer=4, clock_relay_mode=True,
        source_types=["S"], target_types=["A", "B", "C"],
        exclude_pairs="A->B")
    assert "S->A->B" not in set(rows["path"])
    assert "S->A->N->B" in set(rows["path"])
    assert manifest["exclusions"]["applied"] == ["A->B"]
    out2 = tmp_path / "out_excl_clock"
    rows2, manifest2 = _run_typed(
        root, out2, max_interlayer=4, clock_relay_mode=True,
        source_types=["S"], target_types=["A", "B", "C"],
        exclude_clock_clock=True)
    for path in rows2["path"]:
        hops = set(zip(path.split("->"), path.split("->")[1:]))
        assert ("A", "B") not in hops and ("B", "C") not in hops
    assert set(manifest2["exclusions"]["applied"]) == {"A->B", "B->C"}
    with pytest.raises(CorridorRefusal):
        _run_typed(root, tmp_path / "out_bad",
                   source_types=["S"], target_types=["A", "B", "C"],
                   exclude_pairs="A-B")


def test_typed_only_features_refuse_the_fast_lane(tmp_path):
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    with pytest.raises(CorridorRefusal):
        run_lane(None, asked=1, max_interlayer=2, source_types=["A"],
                 target_types=["C"],
                 connections_path=root / "connections.parquet",
                 neuron_table=root / "neurons.parquet",
                 out_dir=tmp_path / "out_fast", type_pruned=False,
                 type_top_k=2)


def test_type_pruned_manifest_records_filter_and_label_parity(tmp_path):
    """P1-K/P2-S: the manifest carries the universe's scope, filters, label
    map digest and enrollment, so the walk set can be reproduced."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out")
    assert manifest["scope"] == "full_floor"
    assert manifest["type_prune_scope"] == "full_floor"
    assert manifest["drop_untyped"] is True
    assert manifest["exclude_intra_type_connections"] is False
    assert manifest["separate_hemispheres"] is False
    assert manifest["hemisphere_filter"] == "both"
    assert manifest["weight_basis"] == "synapse"
    assert len(manifest["type_map_sha"]) == 12
    assert manifest["type_map_size"] == 3
    assert manifest["enrollment"]["sources"] == "types"
    assert manifest["requested_floor"] == 1
    assert manifest["typed_lane_floor"] == 1
    assert manifest["type_pruned_floor"] == 1
    assert manifest["complete_tau"] is None  # direct mode has no run tau
    assert manifest["mass_source"] == "lossless_cone"
    assert manifest["mass_before_source"] == (
        "lossless_discovery_cone_pair_mass")
    prov = json.loads((tmp_path / "out" / "route_provenance.json").read_text())
    assert prov["type_map_sha"] == manifest["type_map_sha"]
    assert prov["type_map_size"] == 3


def test_type_pruned_covers_exhaustive_bodyid_paths(tmp_path):
    """Phase D.5 (budget independence): with a zero exact filter, the lane's
    exported type paths must equal an exhaustive bodyId simple-path search
    projected to types — no path budget, no class share, no corridor quota.
    A phantom type walk that no single body realizes must not appear."""
    edges = [
        ("a1", "b1", 100), ("b1", "c1", 80),      # A->B->C
        ("a1", "d1", 90), ("d1", "c1", 70),       # A->D->C
        ("a1", "b2", 60), ("b2", "d1", 50),       # A->B->D->C
        ("a1", "e1", 40), ("e1", "b1", 30),       # A->E->B->C
        ("a1", "e2", 20), ("f1", "c1", 10),       # keeps E2/F on no A path
    ]
    types = {"a1": "A", "b1": "B", "b2": "B", "d1": "D",
             "e1": "E", "e2": "E", "f1": "F", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(
        root, tmp_path / "out", max_interlayer=3, type_min_ratio_adj=0.0)
    # exhaustive body-simple DFS from every A body to every C body
    adj = {}
    for u, v, _w in edges:
        adj.setdefault(u, []).append(v)
    found = set()

    def dfs(body, walk):
        for nxt in adj.get(body, ()):
            if types.get(nxt) == "C":
                found.add(tuple(types[b] for b in walk + (nxt,)))
                continue
            if nxt in walk or len(walk) >= 4:
                continue
            dfs(nxt, walk + (nxt,))

    for body, typ in types.items():
        if typ == "A":
            dfs(body, (body,))
    assert found == {
        ("A", "B", "C"), ("A", "D", "C"),
        ("A", "B", "D", "C"), ("A", "E", "B", "C")}
    assert set(rows["path"]) == {"->".join(x) for x in found}
    assert manifest["accepted_paths"] == len(found)
    assert manifest["type_pruned_complete"] is True


def test_typed_knobs_round_trip_from_parameters_txt(tmp_path):
    """P2-S: a folder stamped by a typed run reproduces the lane on a bare
    run-folder invocation (no CLI flags)."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "dataset:                    synth\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type prune scope:           full_floor\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "type min wmax:              0\n"
        "type walk mode:             complete\n"
        "type walk cap:              1000\n"
        "type revisit policy:        bounded\n"
        "type max type visits:       0\n"
        "type validate work cap:     2000000\n"
        "per pair scope:             True\n"
        "scope lt mode:              discover\n"
        "scope pad:                  2\n"
        "scope hard cap:             6\n"
        "scope discovery cap:        6\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    rows = pl.read_csv(lane / "route_paths_type.csv")
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert set(rows["path"]) == {"A->B->C"}
    assert manifest["basis"] == "synapse"
    assert manifest["type_walk_cap"] == 1000
    assert manifest["type_pruned_complete"] is True
    assert manifest["scope_mode"] == "per_pair_shortest_plus_2"
    assert manifest["scope_lt"] == "max_local_validated"
    assert manifest["scope_complete"] is True
    assert manifest["scope_complete_by_pair"] == {"A->C": "exhausted"}
    assert (lane / "route_scope_pairs.csv").exists()
    assert (lane / "scope_table.csv").exists()
    prov = json.loads((lane / "route_provenance.json").read_text())
    assert prov["type_pruned"] is True
    assert prov["bound_source"] == "per_pair_scope"
    assert prov["scope_mode"] == "per_pair_shortest_plus_2"
    assert prov["scope_complete"] is True
    assert prov["scope_complete_by_pair"] == {"A->C": "exhausted"}


def test_topk_relay_exclusions_round_trip_from_parameters_txt(tmp_path):
    """The new typed knobs replay from a bare run-folder invocation."""
    edges = [
        ("s1", "a1", 100), ("a1", "b1", 90), ("b1", "c1", 80),
        ("s1", "c1", 70), ("a1", "n1", 60), ("n1", "b1", 50),
    ]
    types = {"s1": "S", "a1": "A", "b1": "B", "c1": "C", "n1": "N"}
    root = _typed_dataset(tmp_path, edges, types)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['S']\n"
        "target neurons:             ['A', 'B', 'C']\n"
        "min synapse number:         1\n"
        "max interlayer:             4\n"
        "dataset:                    synth\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type prune scope:           full_floor\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "type min wmax:              0\n"
        "type walk mode:             complete\n"
        "type walk cap:              1000\n"
        "type revisit policy:        simple\n"
        "type max type visits:       0\n"
        "type validate work cap:     2000000\n"
        "type top k:                 2\n"
        "type diversity:             edge_quota\n"
        "type edge reuse fraction:   0.5\n"
        "type diversity pool:        2\n"
        "clock relay mode:           all\n"
        "type exclude pairs:         A->B\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert manifest["selection_mode"] == "strongest_first"
    assert manifest["type_top_k"] == 2
    assert manifest["diversity_mode"] == "edge_quota"
    assert manifest["diversity"]["mode"] == "edge_quota"
    assert manifest["clock_relay_mode"] == "all"
    assert manifest["exclusions"]["applied"] == ["A->B"]
    assert (lane / "route_topk_pairs.csv").exists()
    assert (lane / "route_paths_type_diverse.csv").exists()
    assert (lane / "route_paths_type_first_clock.csv").exists()
    assert (lane / "route_paths_type_clock_relay.csv").exists()


def test_fast_lane_does_not_adopt_a_typed_stamp(tmp_path):
    """The in-run Fast subprocess must keep route_scope_paths/ Fast.

    It runs on a typed-stamped folder without --type-pruned; adopting the
    stamp would write typed tables into the Fast namespace (and skip the
    Fast lane entirely), which the report then reads as Fast.
    """
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "dataset:                    synth\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n",
        encoding="utf-8")
    assert run_lane(
        run, connections_path=root / "connections.parquet",
        neuron_table=root / "neurons.parquet", infer_type_pruned=False)
    fast = run / "route_scope_paths"
    assert (fast / "route_provenance.json").exists()
    prov = json.loads((fast / "route_provenance.json").read_text())
    assert prov["type_pruned"] is False
    assert (fast / "route_paths_type.csv").exists()
    assert not (run / "route_scope_paths_type_pruned").exists()


def test_typed_lane_honors_exclude_autapses(tmp_path):
    """P1-I: bodyId autapses are removed by the run's rule, while a TYPE
    self-edge realized by two different bodies stays. The A->A pair must
    carry n=1, not 2, and A->A->B->C must still be realizable."""
    edges = [
        ("a1", "a1", 100),   # body autapse — removed
        ("a1", "a2", 10),    # A->A across two bodies — kept
        ("a2", "b1", 5), ("b1", "c1", 4),
    ]
    types = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)

    def _lane(name, exclude):
        run = tmp_path / name
        run.mkdir()
        (run / "parameters.txt").write_text(
            "source neurons:             ['A']\n"
            "target neurons:             ['C']\n"
            "min synapse number:         1\n"
            "max interlayer:             3\n"
            "path_mode:                  all\n"
            f"exclude autapses:           {exclude}\n",
            encoding="utf-8")
        assert run_lane(run, connections_path=root / "connections.parquet",
                        neuron_table=root / "neurons.parquet",
                        type_pruned=True, type_prune_basis="synapse",
                        type_min_synapse=1, type_min_ratio_adj=0.0,
                        type_revisit_policy="bounded")
        lane = run / "route_scope_paths_type_pruned"
        pairs = pl.read_csv(lane / "route_type_pairs_pruned.csv")
        aa = pairs.filter((pl.col("type_pre") == "A")
                          & (pl.col("type_post") == "A"))
        rows = pl.read_csv(lane / "route_paths_type.csv")
        manifest = json.loads(
            (lane / "type_prune_manifest.json").read_text())
        return aa["n"][0] if aa.height else 0, set(rows["path"]), manifest

    kept, paths, manifest = _lane("with_rule", True)
    assert kept == 1
    assert "A->A->B->C" in paths
    assert manifest["exclude_autapses"] is True
    both, paths2, manifest2 = _lane("without_rule", False)
    # With the rule off, the lossless edge union retains the body autapse
    # in the A->A pair. What must survive either way is the TYPE self-edge
    # realized by two different bodies.
    assert both == 2
    assert "A->A->B->C" in paths2
    assert manifest2["exclude_autapses"] is False


def test_type_prune_completeness_includes_unresolved_ratios():
    """Issue 2026-10-22 §2.2: a run with an unresolved conditioned
    denominator cannot be labelled complete, even when every walk was
    validated. `filtered_weak` is deliberate and stays complete."""
    from route_plan import _type_prune_completeness

    ok = _type_prune_completeness(
        enum_capped=False, time_capped=False, unverified=0,
        ratio_unresolved=0)
    assert ok == (True, True, "exhausted")
    bad = _type_prune_completeness(
        enum_capped=False, time_capped=False, unverified=0,
        ratio_unresolved=3)
    assert bad == (False, False, "ratio_unresolved")
    capped = _type_prune_completeness(
        enum_capped=True, time_capped=False, unverified=0,
        ratio_unresolved=0)
    assert capped[0] is False and capped[1] is True
    unverified = _type_prune_completeness(
        enum_capped=False, time_capped=False, unverified=1,
        ratio_unresolved=0)
    assert unverified == (False, True, "unverified")


def test_type_pruned_label_mapper_witness_is_real(tmp_path):
    """Issue 2026-10-22 §2.3: the recorded mapping file is hashed, never
    the empty-map MD5 (`99914b932bd3`)."""
    import hashlib
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    map_path = tmp_path / "map.json"
    base_map = {
        "format": "drocat_custom_groups/v1",
        "source_mapping": {"custom_label": [], "synth": []},
        "target_mapping": {"custom_label": [], "synth": []},
    }
    map_path.write_text(json.dumps(base_map), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "dataset:                    synth\n"
        f"custom mapping file:        {map_path}\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    type_pruned=True, type_prune_basis="synapse",
                    type_min_synapse=1, type_min_ratio_adj=0.0)
    prov = json.loads((run / "route_scope_paths_type_pruned"
                       / "route_provenance.json").read_text())
    assert prov["label_mapper_source"] == f"file:{map_path}"
    assert prov["label_mapper_sha"] == hashlib.md5(
        map_path.read_bytes()).hexdigest()[:12]
    assert prov["label_mapper_sha"] != "99914b932bd3"

    # a changed mapping file changes the witness
    base_map["groups_meta"] = {"rev": 2}
    map_path.write_text(json.dumps(base_map), encoding="utf-8")
    run2 = tmp_path / "run2"
    run2.mkdir()
    (run2 / "parameters.txt").write_text(
        (run / "parameters.txt").read_text(), encoding="utf-8")
    assert run_lane(run2, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet",
                    type_pruned=True, type_prune_basis="synapse",
                    type_min_synapse=1, type_min_ratio_adj=0.0)
    prov2 = json.loads((run2 / "route_scope_paths_type_pruned"
                        / "route_provenance.json").read_text())
    assert prov2["label_mapper_sha"] != prov["label_mapper_sha"]


def test_type_pruned_nt_is_canonical_lowercase(tmp_path):
    """Issue 2026-10-22 §2.4: FAFB `ACH` exports as `acetylcholine`."""
    import pandas as pd
    root = tmp_path / "nt_ds"
    root.mkdir()
    pl.DataFrame({"bodyId_pre": ["a1", "b1"],
                  "bodyId_post": ["b1", "c1"],
                  "weight": [100, 80]}).write_parquet(
        root / "connections.parquet")
    pd.DataFrame({
        "bodyId": ["a1", "b1", "c1"],
        "type": ["A", "B", "C"],
        "nt_type": ["ACH", "GLUT", "GABA"],
    }).to_parquet(root / "neurons.parquet", index=False)
    out = tmp_path / "out"
    assert run_lane(None, asked=1, max_interlayer=2,
                    source_types=["A"], target_types=["C"],
                    connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out,
                    type_pruned=True, type_prune_basis="synapse",
                    type_min_synapse=1, type_min_ratio_adj=0.0)
    row = pl.read_csv(out / "route_paths_type.csv").row(0, named=True)
    assert json.loads(row["nt_types"]) == ["acetylcholine", "glutamate"]
    from utils.label_utils import canonical_neurotransmitter
    assert canonical_neurotransmitter("ACH") == "acetylcholine"
    assert canonical_neurotransmitter("GABA") == "gaba"


def test_effective_type_metadata_aggregates_group_votes():
    """Issue 2026-10-23: a LabelMapper group is one effective label; its
    NT is a deterministic majority over member bodies with votes,
    agreement, tie/ambiguity and native membership."""
    import json as _json
    import pandas as pd
    from type_level_refill import build_effective_type_metadata
    frame = pd.DataFrame({
        "bodyId": ["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2",
                   "d1"],
        "type": ["R7y", "R7p", "R7d", "M1", "M2", "M3", "T1", "T2",
                 "U1"],
        "consensusNt": ["histamine", "histamine", "histamine",
                        "histamine", "histamine", "acetylcholine",
                        "gaba", "histamine", None],
    })
    mapping = {b: "R7" for b in ("a1", "a2", "a3")}
    mapping.update({"b1": "M", "b2": "M", "b3": "M",
                    "c1": "T", "c2": "T", "d1": "U"})
    meta = build_effective_type_metadata(frame, mapping)
    r7 = meta["R7"]
    assert r7["nt_majority"] == "histamine"
    assert r7["nt_votes"] == {"histamine": 3}
    assert r7["nt_n"] == 3 and r7["nt_agreement"] == 1.0
    assert r7["nt_ambiguous"] is False
    assert r7["nt_source_column"] == "consensusNt"
    assert r7["n_bodies"] == 3
    assert r7["native_types"] == ["R7d", "R7p", "R7y"]
    mixed = meta["M"]
    assert mixed["nt_majority"] == "histamine"
    assert mixed["nt_votes"] == {"histamine": 2, "acetylcholine": 1}
    assert abs(mixed["nt_agreement"] - 2 / 3) < 1e-9
    assert mixed["nt_ambiguous"] is False
    tie = meta["T"]
    # Sorted canonical order breaks the tie: gaba < histamine.
    assert tie["nt_majority"] == "gaba"
    assert tie["nt_ambiguous"] is True
    missing = meta["U"]
    assert missing["nt_majority"] == "unknown"
    assert missing["nt_votes"] == {} and missing["nt_n"] == 0
    assert missing["nt_ambiguous"] is True
    assert missing["native_types"] == ["U1"]
    # The helper is JSON-safe for the lane's audit table.
    _json.dumps(meta)


def test_type_nt_map_votes_over_the_effective_group():
    """A custom group's NT is the majority over member bodies, not the
    native subtype keys (MCNS R7*/R8* -> R7/R8)."""
    import pandas as pd
    from route_plan import _type_nt_map
    frame = pd.DataFrame({
        "bodyId": ["a1", "a2", "a3", "b1"],
        "type": ["R7y", "R7p", "R7d", "R8y"],
        "consensusNt": ["histamine", "histamine", "acetylcholine",
                        "histamine"],
    })
    body2type = {"a1": "R7", "a2": "R7", "a3": "R7", "b1": "R8"}
    assert _type_nt_map(frame, body2type=body2type) == {
        "R7": "histamine", "R8": "histamine"}
    native = _type_nt_map(frame)
    assert native["R7y"] == "histamine"
    assert native["R7p"] == "histamine"
    assert native["R7d"] == "acetylcholine"


def test_type_pruned_missing_nt_is_unknown_not_empty(tmp_path):
    """A label with no NT writes `unknown` in the connection table and the
    consensus audit, never NaN/empty."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out_missing_nt"
    rows, _ = _run_typed(root, out)
    conn = pl.read_csv(out / "route_connection_type.csv")
    assert set(conn["nt_type"]) == {"unknown"}
    consensus = pl.read_csv(out / "type_nt_consensus.csv")
    assert set(consensus["nt_majority"]) == {"unknown"}
    assert set(consensus["nt_n"]) == {0}
    assert not conn["nt_type"].is_null().any()
    assert json.loads(rows["nt_types"][0]) == ["unknown", "unknown"]


def test_grouped_source_nt_consensus_end_to_end(tmp_path):
    """MCNS-like fixture: three native R7 subtypes fold to R7; the source
    edge carries histamine and the consensus table records 1.0 agreement."""
    import json as _json
    import pandas as pd
    edges = [("a1", "b1", 100), ("a2", "b1", 90), ("a3", "b1", 80),
             ("b1", "c1", 70)]
    types = {"a1": "R7y", "a2": "R7p", "a3": "R7d",
             "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types, name="nt_group")
    pd.DataFrame({
        "bodyId": ["a1", "a2", "a3", "b1", "c1"],
        "type": ["R7y", "R7p", "R7d", "B", "C"],
        "consensusNt": ["histamine", "histamine", "histamine",
                        "acetylcholine", "gaba"],
    }).to_parquet(root / "neurons.parquet", index=False)
    mapping = {
        "format": "drocat_custom_groups/v1",
        "source_mapping": {"custom_label": ["R7"],
                           "synth": [["R7y", "R7p", "R7d"]]},
        "target_mapping": {"custom_label": [], "synth": []},
    }
    map_path = tmp_path / "nt_mapping.json"
    map_path.write_text(_json.dumps(mapping), encoding="utf-8")
    run = tmp_path / "nt_run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "dataset:                    synth\n"
        "source neurons:             ['R7']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        f"custom mapping file:        {map_path}\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    conn = pl.read_csv(lane / "route_connection_type.csv")
    source = conn.filter(pl.col("type_pre") == "R7").row(0, named=True)
    assert source["nt_type"] == "histamine"
    assert source["nt_agreement"] == 1.0
    assert source["nt_n"] == 3
    assert source["nt_ambiguous"] is False
    assert json.loads(source["nt_votes"]) == {"histamine": 3}
    consensus = pl.read_csv(lane / "type_nt_consensus.csv")
    row = consensus.filter(pl.col("type") == "R7").row(0, named=True)
    assert row["nt_majority"] == "histamine"
    assert row["nt_agreement"] == 1.0
    assert row["n_bodies"] == 3
    assert _json.loads(row["native_types"]) == ["R7d", "R7p", "R7y"]
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert manifest["nt_consensus_table"] == "type_nt_consensus.csv"
    assert manifest["nt_consensus_labels"] >= 2


def test_type_pruned_synapse_basis_defaults_to_the_asked_floor(tmp_path):
    """Issue 2026-10-22 §2.4: an unset pair mass floor on the synapse basis
    is the run's asked floor (3), never the old hard-coded 1."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(
        root, tmp_path / "out", asked=3, type_min_synapse=0,
        type_min_ratio_adj=0.0)
    assert manifest["min_synapse"] == 3
    assert rows.height == 1


# ------------------------------------------------ per-pair shortest scope
def _scope_table(tmp_path, rows, name="scope.csv"):
    import pandas as pd
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_per_pair_scope_phantom_shortest_discovers_validated_lt(tmp_path):
    """Acceptance 1/2: the raw type-shortest A->B->C is a phantom (the B
    bodies are disjoint), so discovery must advance to the first
    bodyId-validated accepted length (4) and bound the pair at Lt+2, not
    at the raw BFS length."""
    # A->B->C is a real TYPE walk (both pair edges lie on some
    # source->target body path) but has no bodyId realization: the A->B
    # edge ends at b1 and the B->C edge starts at b2. Each pair edge is
    # kept alive by its own 4-hop route, so the phantom survives the
    # lossless cone and only bodyId validation can reject it.
    edges = [
        ("a1", "b1", 100),                    # A->B, ends at b1
        ("b1", "x1", 90), ("x1", "y1", 80),
        ("y1", "c1", 70),                     # A->B->X->Y->C (4 hops)
        ("b2", "c1", 50),                     # B->C, starts at b2
        ("a1", "p1", 60), ("p1", "q1", 40),
        ("q1", "b2", 30),                     # A->P->Q->B->C (4 hops)
    ]
    types = {"a1": "A", "b1": "B", "b2": "B", "x1": "X",
             "y1": "Y", "p1": "P", "q1": "Q", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, asked=3, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="discover", scope_pad=2, scope_hard_cap=6)
    assert set(rows["path"]) == {"A->B->X->Y->C", "A->P->Q->B->C"}
    assert manifest["scope_mode"] == "per_pair_shortest_plus_2"
    assert manifest["scope_lt"] == "max_local_validated"
    assert manifest["scope_complete_by_pair"]["A->C"] == "exhausted"
    assert manifest["scope_complete"] is True
    assert manifest["type_pruned_complete"] is True
    assert manifest["scope_pad"] == 2
    assert manifest["scope_hard_cap"] == 6
    assert manifest["scope_bound_source"] == "Lt_plus_pad"
    scope = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()
    assert len(scope) == 1
    pair = scope[0]
    assert pair["Lt_i"] == 4 and pair["Lt"] == 4
    assert pair["scope_bound"] == 6
    assert pair["scope_pad_effective"] == 2
    assert pair["scope_mode"] == "per_pair_shortest_plus_2"
    assert pair["applied_floor"] == 3
    assert pair["status"] == "exhausted"
    assert pair["first_phantom_type_path"] == "A->B->C"
    assert pair["witness_type_path"] == "A->B->X->Y->C"
    # Layered validation produces no bodyId witness column at all.
    assert "witness_body_path" not in pair
    assert pair["bodyid_paths"] == 0
    assert pair["paths"] == 2
    # The reusable Pass-A table carries the same validated length.
    table = pl.read_csv(out / "scope_table.csv").to_dicts()[0]
    assert table["Lt_i"] == 4 and table["max_hops"] == 6
    assert table["type_map_sha"] == manifest["type_map_sha"]
    # The scoped universe is built at the discovered bound, not at the
    # run's global max_interlayer+1 (3): the 4-hop route is enumerated.
    assert manifest["scope_working_bound"] == 6
    assert manifest["scope_final_max_bound"] == 6


def test_per_pair_scope_hard_cap_marks_pair_capped(tmp_path):
    """Acceptance 5: Lt+2 above the hard cap bounds the pair at the cap,
    keeps the in-scope paths, and forces the scoped completeness false."""
    edges = [
        ("a1", "b1", 100), ("b1", "d1", 90),
        ("d1", "e1", 80), ("e1", "c1", 70),
    ]
    types = {"a1": "A", "b1": "B", "d1": "D",
             "e1": "E", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="discover", scope_pad=2, scope_hard_cap=4)
    assert set(rows["path"]) == {"A->B->D->E->C"}
    pair = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()[0]
    assert pair["Lt_i"] == 4
    assert pair["scope_bound"] == 4
    assert pair["hard_capped"] is True
    assert pair["status"] == "capped"
    assert manifest["scope_complete_by_pair"]["A->C"] == "capped"
    assert manifest["scope_complete"] is False
    assert manifest["type_pruned_complete"] is False
    assert manifest["completeness_reason"] == "per_pair_capped"
    assert manifest["scope_capped_pairs"] == ["A->C"]


def test_per_pair_scope_external_table_scope_and_outside_target(tmp_path):
    """Acceptance 3/4: a 1-hop target is in scope at bound 3; a target
    reachable only at 4 hops is reported no_path and absent. The +1
    sensitivity table includes it."""
    edges = [
        ("a1", "c1", 5),                       # A->C, 1 hop
        ("a1", "b1", 10), ("b1", "e1", 9),
        ("e1", "f1", 8), ("f1", "d1", 7),      # A->D, 4 hops
    ]
    types = {"a1": "A", "c1": "C", "b1": "B", "e1": "E",
             "f1": "F", "d1": "D"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C",
         "max_hops": 3, "Lt_i": 1},
        {"source_type": "A", "target_type": "D",
         "max_hops": None, "Lt_i": None},
    ])
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, max_interlayer=2, source_types=["A"],
        target_types=["C", "D"], per_pair_scope=True,
        scope_lt_mode="external", scope_table=str(table),
        scope_pad=2, scope_hard_cap=6)
    assert set(rows["path"]) == {"A->C"}
    assert manifest["scope_lt"] == "external_table"
    assert manifest["scope_complete_by_pair"] == {
        "A->C": "exhausted", "A->D": "no_path"}
    assert manifest["scope_complete"] is False
    assert manifest["type_pruned_complete"] is False
    assert manifest["scope_no_path_pairs"] == ["A->D"]
    scope = {f'{r["source_type"]}->{r["target_type"]}': r
             for r in pl.read_csv(out / "route_scope_pairs.csv").to_dicts()}
    assert scope["A->C"]["scope_bound"] == 3
    assert scope["A->D"]["scope_bound"] is None
    assert scope["A->D"]["status"] == "no_path"

    # +1 sensitivity: the 4-hop target's own bound now admits it.
    table2 = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C",
         "max_hops": 3, "Lt_i": 1},
        {"source_type": "A", "target_type": "D",
         "max_hops": 4, "Lt_i": None},
    ], name="scope_sensitivity.csv")
    out2 = tmp_path / "out2"
    rows2, manifest2 = _run_typed(
        root, out2, max_interlayer=2, source_types=["A"],
        target_types=["C", "D"], per_pair_scope=True,
        scope_lt_mode="external", scope_table=str(table2),
        scope_pad=2, scope_hard_cap=6)
    assert set(rows2["path"]) == {"A->C", "A->B->E->F->D"}
    assert manifest2["scope_complete_by_pair"] == {
        "A->C": "exhausted", "A->D": "exhausted"}
    assert manifest2["scope_complete"] is True


def test_per_pair_scope_table_refusals(tmp_path):
    """Acceptance 4: missing/extra/duplicate pairs, a bound below the
    declared Lt_i, and stale mapping/asked metadata all refuse."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out"

    def _run(table):
        return run_lane(
            None, asked=1, max_interlayer=2, source_types=["A"],
            target_types=["C"], connections_path=root / "connections.parquet",
            neuron_table=root / "neurons.parquet", out_dir=out,
            type_pruned=True, type_prune_basis="synapse",
            type_min_synapse=1, type_min_ratio_adj=0.0,
            per_pair_scope=True, scope_lt_mode="external",
            scope_table=str(table), scope_pad=2, scope_hard_cap=6)

    missing = _scope_table(tmp_path, [], name="empty.csv")
    if missing.stat().st_size == 0:
        missing.write_text("source_type,target_type,max_hops\n",
                           encoding="utf-8")
    with pytest.raises(CorridorRefusal, match="no rows|does not cover"):
        _run(missing)
    extra = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C",
         "max_hops": 3, "Lt_i": 2},
        {"source_type": "A", "target_type": "Z",
         "max_hops": 3, "Lt_i": None},
    ], name="extra.csv")
    with pytest.raises(CorridorRefusal, match="did not select"):
        _run(extra)
    dup = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C",
         "max_hops": 3, "Lt_i": 2},
        {"source_type": "A", "target_type": "C",
         "max_hops": 3, "Lt_i": 2},
    ], name="dup.csv")
    with pytest.raises(CorridorRefusal, match="duplicate"):
        _run(dup)
    below = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C",
         "max_hops": 2, "Lt_i": 3},
    ], name="below.csv")
    with pytest.raises(CorridorRefusal, match="below.*Lt_i"):
        _run(below)
    stale_hash = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C", "max_hops": 3,
         "Lt_i": 2, "type_map_sha": "deadbeef0000"},
    ], name="stale_hash.csv")
    with pytest.raises(CorridorRefusal, match="stale"):
        _run(stale_hash)
    stale_asked = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C", "max_hops": 3,
         "Lt_i": 2, "asked": 999},
    ], name="stale_asked.csv")
    with pytest.raises(CorridorRefusal, match="stale"):
        _run(stale_asked)
    stale_thr = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C", "max_hops": 3,
         "Lt_i": 2, "threshold": 0.5},
    ], name="stale_threshold.csv")
    with pytest.raises(CorridorRefusal, match="stale"):
        _run(stale_thr)
    zero_bound = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C", "max_hops": 0,
         "Lt": 1},
    ], name="zero_bound.csv")
    with pytest.raises(CorridorRefusal, match="no max_hops"):
        _run(zero_bound)
    low_common = _scope_table(tmp_path, [
        {"source_type": "A", "target_type": "C", "max_hops": 4,
         "Lt": 2, "Lt_BANC": 4},
    ], name="low_common.csv")
    with pytest.raises(CorridorRefusal, match="max_i Lt_i"):
        _run(low_common)


def test_per_pair_scope_discover_table_replays_as_external(tmp_path):
    """The article workflow: discover writes scope_table.csv; a separate
    external-mode run consumes it and reproduces the same walks."""
    edges = [
        ("a1", "b1", 100), ("b1", "c1", 80),
        ("a1", "d1", 90), ("d1", "c1", 70),
    ]
    types = {"a1": "A", "b1": "B", "c1": "C", "d1": "D"}
    root = _typed_dataset(tmp_path, edges, types)
    out1 = tmp_path / "discover"
    rows1, manifest1 = _run_typed(
        root, out1, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="discover", scope_pad=2, scope_hard_cap=6)
    table = out1 / "scope_table.csv"
    assert table.exists()
    out2 = tmp_path / "external"
    rows2, manifest2 = _run_typed(
        root, out2, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="external", scope_table=str(table),
        scope_pad=2, scope_hard_cap=6)
    assert set(rows1["path"]) == set(rows2["path"]) == {"A->B->C", "A->D->C"}
    assert manifest1["scope_complete"] is True
    assert manifest2["scope_lt"] == "external_table"
    assert len(manifest2["scope_table_sha"]) == 12
    assert manifest2["scope_complete"] is True


def test_enumerate_type_walks_honours_per_pair_bounds():
    """The per-pair DFS records a target only inside its own bound and
    prunes prefixes that cannot reach any target inside that bound."""
    from route_plan import _enumerate_type_walks
    pairs = pl.DataFrame({
        "type_pre": ["A", "B", "A", "D"],
        "type_post": ["B", "C", "D", "E"],
        "w_sum": [1.0, 1.0, 1.0, 1.0],
        "w_max": [1.0, 1.0, 1.0, 1.0],
        "n": [1, 1, 1, 1],
    })
    pairs2 = pl.concat([pairs, pl.DataFrame({
        "type_pre": ["E"], "type_post": ["C"],
        "w_sum": [1.0], "w_max": [1.0], "n": [1]})])
    walks, stats = _enumerate_type_walks(
        pairs2, ["A"], {"C"}, bound=4,
        pair_bounds={("A", "C"): 2})
    assert walks == [("A", "B", "C")]
    assert stats["scope"] == "per_pair"
    walks_all, stats_all = _enumerate_type_walks(
        pairs2, ["A"], {"C"}, bound=4,
        pair_bounds={("A", "C"): 4})
    assert set(walks_all) == {("A", "B", "C"), ("A", "D", "E", "C")}
    assert stats_all["max_bound"] == 4
    none, stats_none = _enumerate_type_walks(
        pairs2, ["A"], {"C"}, bound=4,
        pair_bounds={("A", "C"): None})
    assert none == []
    assert stats_none["walks"] == 0


def test_per_pair_scope_off_has_no_scope_contract(tmp_path):
    """The global-bound path is unchanged: no scope table, no scope fields."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    rows, manifest = _run_typed(root, tmp_path / "out", max_interlayer=2)
    assert rows.height == 1
    assert "scope_mode" not in manifest
    assert not (tmp_path / "out" / "route_scope_pairs.csv").exists()
    prov = json.loads((tmp_path / "out" / "route_provenance.json").read_text())
    assert prov["bound_source"] == "max_interlayer"
    assert prov["scope_mode"] is None


def test_per_pair_scope_three_dataset_lt_max_local_contract(tmp_path):
    """Acceptance 1: three datasets with per-pair Lt_i 2/3/4 and a
    conserved 4-hop path. A global-min+1 table (3) excludes it; the
    max-local+1 (5) and max-local+2 (6) scopes include it."""
    # Dataset A: shortest A->C is 2 hops.
    d2 = _typed_dataset(tmp_path, [
        ("a1", "b1", 100), ("b1", "c1", 80),
    ], {"a1": "A", "b1": "B", "c1": "C"}, name="ds2")
    # Dataset B: shortest A->C is 3 hops.
    d3 = _typed_dataset(tmp_path, [
        ("a1", "b1", 100), ("b1", "x1", 90), ("x1", "c1", 80),
    ], {"a1": "A", "b1": "B", "x1": "X", "c1": "C"}, name="ds3")
    # Dataset C: the conserved path is 4 hops; this is the slowest dataset.
    d4 = _typed_dataset(tmp_path, [
        ("a1", "b1", 100), ("b1", "x1", 90),
        ("x1", "y1", 80), ("y1", "c1", 70),
        ("a1", "d1", 60),  # a direct control target, so a 3-hop bound
                           # still has a non-empty lossless cone
    ], {"a1": "A", "b1": "B", "x1": "X", "y1": "Y",
        "c1": "C", "d1": "D"}, name="ds4")
    lts = {}
    for name, root in (("ds2", d2), ("ds3", d3), ("ds4", d4)):
        out = tmp_path / f"discover_{name}"
        _run_typed(root, out, max_interlayer=2, per_pair_scope=True,
                   scope_lt_mode="discover", scope_pad=2,
                   scope_hard_cap=6)
        table = pl.read_csv(out / "scope_table.csv").to_dicts()[0]
        lts[name] = int(table["Lt_i"])
        assert table["max_hops"] == min(int(table["Lt_i"]) + 2, 6)
    assert lts == {"ds2": 2, "ds3": 3, "ds4": 4}
    common_lt = max(lts.values())
    assert common_lt == 4

    def _external(max_hops, tag, lt_i=None, lt_value=None):
        table = _scope_table(tmp_path, [
            {"source_type": "A", "target_type": "C",
             "max_hops": max_hops, "Lt_i": lt_i,
             "Lt": common_lt if lt_value is None else lt_value},
            {"source_type": "A", "target_type": "D",
             "max_hops": 3, "Lt_i": 1, "Lt": 1},
        ], name=f"scope_{tag}.csv")
        out = tmp_path / f"external_{tag}"
        rows, manifest = _run_typed(
            d4, out, max_interlayer=2, source_types=["A"],
            target_types=["C", "D"], per_pair_scope=True,
            scope_lt_mode="external", scope_table=str(table),
            scope_pad=2, scope_hard_cap=6)
        return rows, manifest

    # global-min Lt + 1 = 3: the 4-hop conserved path is outside scope.
    rows_min, man_min = _external(
        3, "global_min_plus_1", lt_i=None, lt_value=min(lts.values()))
    assert "A->B->X->Y->C" not in set(rows_min["path"])
    assert "A->D" in set(rows_min["path"])
    assert man_min["scope_complete_by_pair"] == {
        "A->C": "no_path", "A->D": "exhausted"}
    assert man_min["scope_complete"] is False
    # max-local Lt + 1 = 5: included.
    rows_1, man_1 = _external(5, "max_local_plus_1", lt_i=common_lt)
    assert set(rows_1["path"]) == {"A->B->X->Y->C", "A->D"}
    assert man_1["scope_complete_by_pair"] == {
        "A->C": "exhausted", "A->D": "exhausted"}
    assert man_1["scope_complete"] is True
    _pair_1 = {r["target_type"]: r for r in pl.read_csv(
        tmp_path / "external_max_local_plus_1"
        / "route_scope_pairs.csv").to_dicts()}["C"]
    assert _pair_1["Lt_i"] == 4
    assert _pair_1["Lt"] == common_lt == 4
    assert _pair_1["scope_bound"] == 5
    assert _pair_1["scope_pad_effective"] == 1
    # max-local Lt + 2 = 6: included (and equal to the full 1 % set here).
    rows_2, man_2 = _external(6, "max_local_plus_2", lt_i=common_lt)
    assert set(rows_2["path"]) == {"A->B->X->Y->C", "A->D"}
    assert man_2["scope_complete"] is True


def test_per_pair_scope_empty_cone_completes_as_no_path(tmp_path):
    """A scope too small for every target is a legitimate no_path result,
    not a refusal: the run reports the unreached pair and claims no
    completeness."""
    edges = [
        ("a1", "b1", 100), ("b1", "x1", 90),
        ("x1", "y1", 80), ("y1", "c1", 70),   # A->C is 4 hops
    ]
    types = {"a1": "A", "b1": "B", "x1": "X",
             "y1": "Y", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [{
        "source_type": "A", "target_type": "C", "max_hops": 3,
    }], name="small_scope.csv")
    out = tmp_path / "out"
    assert run_lane(
        None, asked=1, max_interlayer=2, source_types=["A"],
        target_types=["C"], connections_path=root / "connections.parquet",
        neuron_table=root / "neurons.parquet", out_dir=out,
        type_pruned=True, type_prune_basis="synapse",
        type_min_synapse=1, type_min_ratio_adj=0.0,
        per_pair_scope=True, scope_lt_mode="external",
        scope_table=str(table), scope_pad=2, scope_hard_cap=6)
    manifest = json.loads((out / "type_prune_manifest.json").read_text())
    assert manifest["scope_complete_by_pair"] == {"A->C": "no_path"}
    assert manifest["scope_complete"] is False
    assert manifest["type_pruned_complete"] is False
    assert manifest["completeness_reason"] == "per_pair_no_path"
    assert pl.read_csv(out / "route_paths_type.csv").height == 0
    assert pl.read_csv(out / "route_scope_pairs.csv").height == 1
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["status"] == "complete"
    assert prov["scope_no_path_pairs"] == ["A->C"]


def test_per_pair_scope_run_folder_external_round_trip(tmp_path):
    """A run-folder replay reads `per pair scope`/`scope table` from
    parameters.txt; an external scope table reproduces discover's paths."""
    edges = [
        ("a1", "b1", 100), ("b1", "c1", 80),
        ("a1", "d1", 90), ("d1", "c1", 70),
    ]
    types = {"a1": "A", "b1": "B", "c1": "C", "d1": "D"}
    root = _typed_dataset(tmp_path, edges, types)
    out_discover = tmp_path / "discover"
    rows_d, _ = _run_typed(
        root, out_discover, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="discover", scope_pad=2, scope_hard_cap=6)
    table = out_discover / "scope_table.csv"
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "dataset:                    synth\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "per pair scope:             True\n"
        "scope lt mode:              external\n"
        f"scope table:                {table}\n"
        "scope pad:                  2\n"
        "scope hard cap:             6\n"
        "scope discovery cap:        6\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    rows_r = pl.read_csv(lane / "route_paths_type.csv")
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert set(rows_r["path"]) == set(rows_d["path"])
    assert manifest["scope_lt"] == "external_table"
    assert manifest["scope_complete"] is True
    assert (lane / "route_scope_pairs.csv").exists()
    # The supplied table is preserved beside the lane outputs, byte-equal
    # to the input whose sha the manifest records.
    assert (lane / "scope_table.csv").read_bytes() == table.read_bytes()

    # Portable replay: if the originally recorded table path is gone, the
    # lane's byte-identical copy is the fallback audit witness.
    table.unlink()
    rerun = run / "rerun"
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=rerun)
    rows_rerun = pl.read_csv(rerun / "route_paths_type.csv")
    manifest2 = json.loads((rerun / "type_prune_manifest.json").read_text())
    assert set(rows_rerun["path"]) == set(rows_d["path"])
    assert manifest2["scope_table_sha"] == manifest["scope_table_sha"]
    assert manifest2["scope_lt"] == "external_table"


def test_per_pair_scope_all_unreached_table_completes(tmp_path):
    """Every pair may be unreached: a table with no bounds completes with
    no_path rows instead of refusing or inventing a global bound."""
    edges = [
        ("a1", "b1", 100), ("b1", "x1", 90),
        ("x1", "y1", 80), ("y1", "c1", 70),
    ]
    types = {"a1": "A", "b1": "B", "x1": "X",
             "y1": "Y", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [{
        "source_type": "A", "target_type": "C", "max_hops": None,
    }], name="all_unreached.csv")
    out = tmp_path / "out"
    assert run_lane(
        None, asked=1, max_interlayer=2, source_types=["A"],
        target_types=["C"], connections_path=root / "connections.parquet",
        neuron_table=root / "neurons.parquet", out_dir=out,
        type_pruned=True, type_prune_basis="synapse",
        type_min_synapse=1, type_min_ratio_adj=0.0,
        per_pair_scope=True, scope_lt_mode="external",
        scope_table=str(table), scope_pad=2, scope_hard_cap=6)
    manifest = json.loads((out / "type_prune_manifest.json").read_text())
    assert manifest["scope_complete_by_pair"] == {"A->C": "no_path"}
    assert manifest["scope_complete"] is False
    assert manifest["type_pruned_complete"] is False
    assert pl.read_csv(out / "route_paths_type.csv").height == 0
    assert pl.read_csv(out / "route_scope_pairs.csv").height == 1


def test_fast_subprocess_ignores_the_per_pair_scope_stamp(tmp_path):
    """The Fast subprocess on a scoped typed folder passes
    --no-infer-type-pruned; it must ignore the scope stamp too, or the
    Fast lane would refuse as a typed run."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    run = tmp_path / "run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "source neurons:             ['A']\n"
        "target neurons:             ['C']\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "per pair scope:             True\n"
        "scope lt mode:              discover\n"
        "scope pad:                  2\n"
        "scope hard cap:             6\n"
        "scope discovery cap:        6\n",
        encoding="utf-8")
    out = tmp_path / "fast"
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet", out_dir=out,
                    type_pruned=False, infer_type_pruned=False)
    assert (out / "route_paths_type.csv").exists()
    assert not (out / "route_scope_pairs.csv").exists()
    prov = json.loads((out / "route_provenance.json").read_text())
    assert prov["type_pruned"] is False
    assert prov["scope_mode"] is None
    assert not (run / "route_scope_paths_type_pruned").exists()


def test_per_pair_scope_external_lt_plus_pad_over_cap_is_capped(tmp_path):
    """External mode: a pair whose Lt + pad exceeds the hard cap is capped
    even when the table already clamped max_hops to the cap."""
    edges = [
        ("a1", "b1", 100), ("b1", "x1", 90), ("x1", "c1", 80),
    ]
    types = {"a1": "A", "b1": "B", "x1": "X", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [{
        "source_type": "A", "target_type": "C", "max_hops": 4,
        "Lt_i": 3, "Lt": 3,
    }], name="cap_table.csv")
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="external", scope_table=str(table),
        scope_pad=2, scope_hard_cap=4)
    assert set(rows["path"]) == {"A->B->X->C"}
    pair = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()[0]
    assert pair["scope_bound"] == 4
    assert pair["scope_pad_effective"] == 1
    assert pair["hard_capped"] is True
    assert pair["status"] == "capped"
    assert manifest["scope_bound_source"] == "external_table"
    assert manifest["scope_complete"] is False
    assert manifest["completeness_reason"] == "per_pair_capped"


def test_per_pair_scope_report_fills_the_local_volume_column(tmp_path):
    """The per-pair report names Lt_FAFB/Lt_MCNS/Lt_BANC; a FAFB run fills
    its own column even when the input table carries no Lt_<volume>."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, dataset="flywire_FAFB_v783", max_interlayer=2,
        per_pair_scope=True, scope_lt_mode="discover",
        scope_pad=2, scope_hard_cap=6)
    pair = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()[0]
    assert pair["Lt_FAFB"] == pair["Lt_i"] == 2
    assert pair["Lt_MCNS"] is None
    assert pair["Lt_BANC"] is None
    assert manifest["scope_dataset"] == "flywire_FAFB_v783"


def test_per_pair_scope_external_validation_cap_is_not_upgraded(tmp_path):
    """A supplied table whose scope discovery was validation-capped cannot
    be reported as exhausted just because this run's enumeration drains."""
    edges = [("a1", "b1", 100), ("b1", "c1", 80)]
    types = {"a1": "A", "b1": "B", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [{
        "source_type": "A", "target_type": "C", "max_hops": 4,
        "Lt_i": 2, "Lt": 2, "status": "validation_cap",
    }], name="validation_cap_table.csv")
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, max_interlayer=2, per_pair_scope=True,
        scope_lt_mode="external", scope_table=str(table),
        scope_pad=2, scope_hard_cap=6)
    assert set(rows["path"]) == {"A->B->C"}
    pair = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()[0]
    assert pair["status"] == "validation_cap"
    assert manifest["scope_complete_by_pair"] == {
        "A->C": "validation_cap"}
    assert manifest["scope_complete"] is False
    assert manifest["completeness_reason"] == "per_pair_validation_cap"
    assert manifest["type_pruned_complete"] is False


def test_per_pair_scope_external_volume_columns_derive_common_lt(tmp_path):
    """A multi-dataset table carrying only Lt_FAFB/Lt_MCNS/Lt_BANC takes
    Lt = max(volumes) and derives max_hops = Lt + pad (issue §3 item 2)."""
    edges = [
        ("a1", "b1", 100), ("b1", "x1", 90),
        ("x1", "y1", 80), ("y1", "c1", 70),
    ]
    types = {"a1": "A", "b1": "B", "x1": "X",
             "y1": "Y", "c1": "C"}
    root = _typed_dataset(tmp_path, edges, types)
    table = _scope_table(tmp_path, [{
        "source_type": "A", "target_type": "C",
        "Lt_FAFB": 2, "Lt_MCNS": 3, "Lt_BANC": 4,
    }], name="multi_volume.csv")
    out = tmp_path / "out"
    rows, manifest = _run_typed(
        root, out, dataset="flywire_FAFB_v783", max_interlayer=2,
        per_pair_scope=True, scope_lt_mode="external",
        scope_table=str(table), scope_pad=2, scope_hard_cap=8)
    assert set(rows["path"]) == {"A->B->X->Y->C"}
    pair = pl.read_csv(out / "route_scope_pairs.csv").to_dicts()[0]
    assert pair["Lt"] == 4
    assert pair["scope_bound"] == 6
    assert pair["scope_pad_effective"] == 2
    assert pair["Lt_FAFB"] == 2
    assert pair["Lt_MCNS"] == 3
    assert pair["Lt_BANC"] == 4
    assert pair["status"] == "exhausted"
    assert manifest["scope_complete"] is True


def test_grouped_source_mapping_end_to_end(tmp_path):
    """Issue 2026-10-22 (grouped source scope): a custom mapping that folds
    several native R7/R8 subtypes into R7/R8 must build the typed universe
    with the GROUPED labels. A previous native run's cached universe must
    not be reused (the cache key now carries the effective type-map digest).

    Fixture: two native R7 subtypes + two R8 subtypes mapped to R7/R8,
    one intermediate type B and 21 targets. Expected 2 x 21 = 42 pairs,
    all exhausted, full-floor rows > 0 and a non-empty path table.
    """
    targets = [f"C{i:02d}" for i in range(1, 22)]
    edges = [
        ("a1", "b1", 100), ("a2", "b1", 90),
        ("a3", "b2", 80), ("a4", "b2", 70),
    ]
    for c in targets:
        edges.append(("b1", f"c_{c}", 50))
        edges.append(("b2", f"c_{c}", 40))
    types = {"a1": "R7p", "a2": "R7d", "a3": "R8p", "a4": "R8d",
             "b1": "B", "b2": "B"}
    types.update({f"c_{c}": c for c in targets})
    root = _typed_dataset(tmp_path, edges, types, name="grouped")

    # 1) A native run first, so its cached universe has native labels.
    out_native = tmp_path / "native"
    assert run_lane(
        None, asked=1, max_interlayer=2,
        source_types=["R7p", "R7d", "R8p", "R8d"],
        target_types=targets,
        connections_path=root / "connections.parquet",
        neuron_table=root / "neurons.parquet", out_dir=out_native,
        type_pruned=True, type_prune_basis="synapse",
        type_min_synapse=1, type_min_ratio_adj=0.0)
    native_rows = pl.read_csv(out_native / "route_paths_type.csv")
    assert native_rows.height > 0
    assert all(p.split("->")[0] in {"R7p", "R7d", "R8p", "R8d"}
               for p in native_rows["path"])

    # 2) The grouped mapping must rebuild with R7/R8, not reuse native.
    mapping = {
        "format": "drocat_custom_groups/v1",
        "source_mapping": {
            "custom_label": ["R7", "R8"],
            "synth": [["R7p", "R7d"], ["R8p", "R8d"]],
        },
        "target_mapping": {"custom_label": [], "synth": []},
    }
    mapping_path = tmp_path / "grouped_mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    run = tmp_path / "grouped_run"
    run.mkdir()
    (run / "parameters.txt").write_text(
        "dataset:                    synth\n"
        # The article's 80 run passes the resolved source bodyIds, not the
        # grouped labels; enrollment must still resolve them to R7/R8.
        "source neurons:             ['a1', 'a2', 'a3', 'a4']\n"
        f"target neurons:             {targets!r}\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "per pair scope:             True\n"
        "scope lt mode:              discover\n"
        "scope pad:                  2\n"
        "scope hard cap:             6\n"
        "scope discovery cap:        6\n"
        f"custom mapping file:        {mapping_path}\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    rows = pl.read_csv(lane / "route_paths_type.csv")
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert rows.height > 0
    assert manifest["full_floor_rows"] > 0
    assert manifest["scope_pairs_total"] == 42
    assert manifest["scope_complete"] is True
    assert set(manifest["scope_complete_by_pair"].values()) == {"exhausted"}
    assert manifest["sources"] == ["R7", "R8"]
    assert manifest["targets"] == targets
    assert manifest["source_bodies_effective"] == 4
    assert manifest["source_group_counts"] == {"R7": 2, "R8": 2}
    assert manifest["source_group_counts_all"] == {"R7": 2, "R8": 2}
    assert manifest["source_unenrolled_counts"] == {}
    assert manifest["source_native_counts"] == {
        "R7d": 1, "R7p": 1, "R8d": 1, "R8p": 1}
    assert len(manifest["source_group_mapping_sha"]) == 12
    # Both grouped sources are present in the exported paths.
    assert any(p.startswith("R7->B->") for p in rows["path"])
    assert any(p.startswith("R8->B->") for p in rows["path"])
    # No native source label leaks into the grouped path table.
    assert not any(p.startswith(("R7p", "R7d", "R8p", "R8d"))
                   for p in rows["path"])


def test_universe_cache_key_carries_effective_type_map(tmp_path):
    """Issue 2026-10-22 (grouped source scope): the typed-universe cache
    key must change with the effective bodyId->type map. Previously every
    LabelMapper hashed as {} (no `overall_mapping` attribute), so a grouped
    run reused a native-label universe."""
    from route_plan import _label_mapper_payload, _universe_signature
    root = _typed_dataset(
        tmp_path, [("a1", "b1", 10), ("b1", "c1", 5)],
        {"a1": "A", "b1": "B", "c1": "C"})
    args = (root / "connections.parquet", root / "neurons.parquet",
            1, True, False, False, "both", None, False)
    native = _universe_signature(*args, type_map_sha="aaaa")
    grouped = _universe_signature(*args, type_map_sha="bbbb")
    assert native != grouped

    from comparison.label_mapper import LabelMapper
    mapper = LabelMapper(
        source_mapping_dict={"synth": [["R7p"], ["R7d"]]},
        source_labels=["R7", "R8"],
    )
    payload = _label_mapper_payload(mapper)
    assert payload and "_source_mapping" in payload
    assert payload["_source_mapping"]["R7"]["synth"] == [["R7p"]]
    assert payload["_source_mapping"]["R8"]["synth"] == [["R7d"]]


def test_grouped_source_prefix_reports_unenrolled_group(tmp_path):
    """Prefix-source grouped run (the article's 80-folder shape): the
    effective label R7R8_unclear exists but no body is enrolled. The
    manifest must keep it in source_group_counts_all and disclose the
    unenrolled count, while the 42 enrolled R7/R8 pairs still work."""
    targets = [f"C{i:02d}" for i in range(1, 22)]
    edges = [
        ("a1", "b1", 100), ("a2", "b1", 90),
        ("a3", "b2", 80), ("a4", "b2", 70),
    ]
    for c in targets:
        edges.append(("b1", f"c_{c}", 50))
        edges.append(("b2", f"c_{c}", 40))
    types = {"a1": "R7p", "a2": "R7d", "a3": "R8p", "a4": "R8d",
             "a5": "R7R8_unclear", "b1": "B", "b2": "B"}
    types.update({f"c_{c}": c for c in targets})
    root = _typed_dataset(tmp_path, edges, types, name="grouped_prefix")
    mapping = {
        "format": "drocat_custom_groups/v1",
        "source_mapping": {
            "custom_label": ["R7", "R8"],
            "synth": [["R7p", "R7d"], ["R8p", "R8d"]],
        },
        "target_mapping": {"custom_label": [], "synth": []},
    }
    mapping_path = tmp_path / "grouped_mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    run = tmp_path / "grouped_prefix_run"
    run.mkdir()
    (run / "source_neurons.csv").write_text(
        "bodyId\na1\na2\na3\na4\n", encoding="utf-8")
    (run / "parameters.txt").write_text(
        "dataset:                    synth\n"
        "source neurons:             {'startswith': ['R7', 'R8']}\n"
        f"target neurons:             {targets!r}\n"
        "min synapse number:         1\n"
        "max interlayer:             2\n"
        "path_mode:                  all\n"
        "type_pruned:                True\n"
        "type prune basis:           synapse\n"
        "type min ratio adj:         0.0\n"
        "type min synapse:           1\n"
        "per pair scope:             True\n"
        "scope lt mode:              discover\n"
        "scope pad:                  2\n"
        "scope hard cap:             6\n"
        "scope discovery cap:        6\n"
        f"custom mapping file:        {mapping_path}\n",
        encoding="utf-8")
    assert run_lane(run, connections_path=root / "connections.parquet",
                    neuron_table=root / "neurons.parquet")
    lane = run / "route_scope_paths_type_pruned"
    manifest = json.loads((lane / "type_prune_manifest.json").read_text())
    assert manifest["sources"] == ["R7", "R7R8_unclear", "R8"]
    assert manifest["scope_pairs_total"] == 63
    assert manifest["source_bodies_effective"] == 4
    assert manifest["source_group_counts"] == {"R7": 2, "R8": 2}
    assert manifest["source_group_counts_all"] == {
        "R7": 2, "R7R8_unclear": 1, "R8": 2}
    assert manifest["source_unenrolled_counts"] == {"R7R8_unclear": 1}
    status = manifest["scope_complete_by_pair"]
    assert all(status[f"R7R8_unclear->{c}"] == "no_path" for c in targets)
    assert all(status[f"R7->{c}"] == "exhausted" for c in targets)
    assert all(status[f"R8->{c}"] == "exhausted" for c in targets)
