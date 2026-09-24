#!/usr/bin/env python3
"""Real-data validation of the PRODUCTION implementations (not the prototype):

  1. coana.fit_edge_budget vs the prototype fit_budget and the one-shot
     apply_edge_budget_floor on a real broad-target FAFB cone
     (same tier decision, same kept rows, cap invariant, supersets);
  2. the pipeline integration: FindNeuronConnection._graph_edge_frames
     floors in 'all' mode (floor <= cap, budget_fully_used reported) and
     never floors in 'shortest' mode;
  3. FastGraph.find_paths_shortest_strongest_first on the real graph:
     unbudgeted set == find_paths_shortest_backward, descending
     bottleneck emission, and a bitten run keeps exactly
     {bottleneck >= tau} (verified against an unbudgeted enumeration).

Run:  python3 scripts/harness/verify_production_real_data.py [scenario A|C|all]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "harness"))

import polars as pl                                        # noqa: E402
import coana                                               # noqa: E402
from core.fast_graph import FastGraph                       # noqa: E402
import verify_budget_fit_pruning as bfp                      # noqa: E402

DATASET = "flywire_FAFB_v783"
CONN = ROOT / "datasets" / DATASET / f"{DATASET}_merged_connections.parquet"
INDEX = ROOT / "neuron_indexes" / DATASET / "neuron_index.parquet"

FAILS = []


def check(name, fn):
    try:
        detail = fn() or ''
        print(f"  PASS  {name} {detail}")
    except AssertionError as exc:
        FAILS.append(name)
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:                                # noqa: BLE001
        FAILS.append(name)
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


def pick(index, types, per_type=None):
    ids = []
    for t in types:
        sub = index.filter(pl.col("type") == t)["bodyId"].to_list()
        ids.extend(str(x) for x in (sub if per_type is None else sub[:per_type]))
    return ids


def discover(conn, sources, max_interlayer):
    reached, frontier, tables = set(sources), list(sources), []
    for _ in range(max_interlayer + 1):
        t = (conn.filter(pl.col("bodyId_pre").is_in(frontier))
                 .select("bodyId_pre", "bodyId_post", "weight"))
        tables.append(t)
        posts = set(t["bodyId_post"].to_list()) - reached
        reached |= posts
        frontier = list(posts)
    return tables, reached


def to_edges(tables):
    union = pl.concat(tables)
    ids = (pl.concat([union.select(pl.col("bodyId_pre").alias("id")),
                      union.select(pl.col("bodyId_post").alias("id"))])
           .unique()["id"].to_list())
    code = {x: i for i, x in enumerate(ids)}
    pre = union["bodyId_pre"].replace_strict(code, return_dtype=pl.Int64).to_list()
    post = union["bodyId_post"].replace_strict(code, return_dtype=pl.Int64).to_list()
    return list(zip(pre, post, union["weight"].to_list())), code


def to_tables(edges):
    rows = {"bodyId_pre": [], "bodyId_post": [], "weight": []}
    for u, v, w in edges:
        rows["bodyId_pre"].append(u)
        rows["bodyId_post"].append(v)
        rows["weight"].append(w)
    return [pl.DataFrame(rows).with_columns(
        pl.lit("0->1").alias("conn_layer"))]


def main():
    only = sys.argv[1].upper() if len(sys.argv) > 1 else "C"
    index = pl.read_parquet(INDEX)
    conn = (pl.scan_parquet(CONN)
            .filter(pl.col("weight") >= 3)
            .with_columns(pl.col("bodyId_pre").cast(pl.Utf8),
                          pl.col("bodyId_post").cast(pl.Utf8))
            .select("bodyId_pre", "bodyId_post", "weight")
            .collect())

    scenarios = {
        "C": dict(max_interlayer=3, target_types=["Tm3"], caps=[50_000]),
        "D": dict(max_interlayer=4, target_types=["Tm3", "Mi1"],
                  caps=[1_000_000]),
    }[only]

    sources = pick(index, ["LC4", "LPLC1"], 4)
    targets = pick(index, scenarios["target_types"])
    bound = scenarios["max_interlayer"] + 1
    print(f"cone: {len(sources)} sources x {len(targets)} targets "
          f"(bound {bound})")
    tables, reached = discover(conn, sources, scenarios["max_interlayer"])
    targets_found = [t for t in targets if t in reached]
    edges, code = to_edges(tables)
    src = [code[s] for s in sources if s in code]
    tgt = [code[s] for s in targets_found]
    t0 = time.perf_counter()
    e0, _ = bfp.hop_pass(edges, src, tgt, bound)
    print(f"cone rows {len(edges):,} -> lossless-pruned E0 {len(e0):,} "
          f"({time.perf_counter() - t0:.1f}s)")
    tables0 = to_tables(e0)
    g0 = FastGraph()
    for u, v, w in e0:
        g0.add_edge(u, v, w)

    # ---- 1. production fit_edge_budget vs prototype vs one-shot ----------
    for cap in scenarios["caps"]:
        def prod_fit():
            out, st = coana.fit_edge_budget(tables0, cap, src, tgt, bound,
                                            max_probes=8)
            assert st["status"] == "floored", st["status"]
            rows = sum(t.height for t in out)
            assert rows <= cap, rows
            return (f"t*={st['floor']:.0f} kept {rows:,} "
                    f"slack {st['residual_slack']:,} "
                    f"probes {st['probes']} fully_used "
                    f"{st['budget_fully_used']} {st['search_seconds']:.1f}s")

        def proto_match():
            out, st = coana.fit_edge_budget(tables0, cap, src, tgt, bound,
                                            max_probes=8)
            fit = bfp.fit_budget(e0, cap, src, tgt, bound,
                                 max_probes=8, max_passes=1)
            assert st["floor"] == fit["threshold"], (
                f"{st['floor']} != {fit['threshold']}")
            prod_rows = sum(t.height for t in out)
            assert prod_rows == len(fit["kept"]), (
                f"{prod_rows} != {len(fit['kept'])}")
            return f"tier {st['floor']:.0f} matches prototype"

        def oneshot_compare():
            one_out, one = coana.apply_edge_budget_floor(
                tables0, cap, src, tgt, bound)
            fit_out, fit = coana.fit_edge_budget(tables0, cap, src, tgt,
                                                 bound, max_probes=8)
            one_rows = sum(t.height for t in one_out)
            fit_rows = sum(t.height for t in fit_out)
            assert fit_rows >= one_rows
            assert fit["floor"] <= one["floor"]
            return (f"one-shot t0={one['floor']:.0f} kept {one_rows:,} | "
                    f"fit kept {fit_rows:,} "
                    f"(x{fit_rows / max(one_rows, 1):.1f})")

        print(f"\n-- cap {cap:,} --")
        check("production fit_edge_budget", prod_fit)
        check("tier matches prototype", proto_match)
        check(">= one-shot utilization", oneshot_compare)

    # ---- 2. pipeline integration (_graph_edge_frames) ---------------------
    def pipeline_all_floors():
        fc = object.__new__(coana.FindNeuronConnection)
        fc.max_interlayer = bound - 1
        fc.graph_edge_limit_bodyid = 20_000
        fc._vprint = lambda *a, **k: None
        fc._warn_notes = []
        fc.edge_weight_floor = None
        fc.edge_budget_landing = None
        frames = fc._graph_edge_frames(
            [t.clone() for t in tables0], [str(s) for s in src],
            [str(t) for t in tgt], path_mode="all")
        rows = sum(f.height for f in frames)
        assert rows <= 20_000, rows
        assert fc.edge_weight_floor is not None
        assert fc.graph_pruning_record.get("stage") == "edge_budget_fit"
        return (f"floor {fc.edge_weight_floor:.0f}, {rows:,} rows <= cap, "
                f"stage recorded")

    def pipeline_shortest_never_floors():
        fc = object.__new__(coana.FindNeuronConnection)
        fc.max_interlayer = bound - 1
        fc.graph_edge_limit_bodyid = 20_000
        fc._vprint = lambda *a, **k: None
        fc._warn_notes = []
        fc.edge_weight_floor = None
        fc.edge_budget_landing = None
        frames = fc._graph_edge_frames(
            [t.clone() for t in tables0], [str(s) for s in src],
            [str(t) for t in tgt], path_mode="shortest")
        rows = sum(f.height for f in frames)
        assert fc.edge_weight_floor is None, "shortest mode was floored"
        assert rows == len(e0), (rows, len(e0))
        return f"untouched: {rows:,} rows, no floor"

    print()
    check("pipeline 'all' floors under cap (20k)", pipeline_all_floors)
    check("pipeline 'shortest' never floors", pipeline_shortest_never_floors)

    # ---- 3. production shortest StrongestFirst on the real graph ----------
    if only == "C":
        print(f"\nshortest StrongestFirst on the real graph "
              f"({g0.number_of_edges():,} edges / "
              f"{g0.number_of_nodes():,} nodes):")

        def unbudgeted_set_matches():
            stats = {}
            t0 = time.perf_counter()
            sf = set(map(tuple, g0.find_paths_shortest_strongest_first(
                tgt, src, bound, budget=None, stats=stats)))
            dt = time.perf_counter() - t0
            bwd = set(map(tuple, g0.find_paths_shortest_backward(
                tgt, src, bound, verbose=False)))
            assert sf == bwd, (f"{len(sf)} vs {len(bwd)}")
            return f"{len(sf):,} min-hop paths identical in {dt:.1f}s"

        def path_bn(g, path):
            return min(g.adj[u][v] for u, v in zip(path, path[1:]))

        def descending_order():
            bns = []
            for p in g0.find_paths_shortest_strongest_first(tgt, src, bound):
                bns.append(path_bn(g0, p))
                if len(bns) >= 20_000:      # prefix order is the claim
                    break
            assert bns == sorted(bns, reverse=True)
            return f"first {len(bns):,} in descending order " \
                   f"({bns[-1]:g}..{bns[0]:g})"

        def bitten_tau_property():
            stats = {}
            budget = 500
            bitten = list(g0.find_paths_shortest_strongest_first(
                tgt, src, bound, budget=budget, stats=stats))
            tau = stats["tau"]
            assert stats["budget_bitten"] and len(bitten) >= budget
            # the bitten set is exactly {bn >= tau} of the unbudgeted run
            full = list(g0.find_paths_shortest_strongest_first(
                tgt, src, bound, budget=None))
            expect_set = {tuple(p) for p in full
                          if path_bn(g0, p) >= tau}
            assert {tuple(p) for p in bitten} == expect_set
            return (f"budget {budget} -> tau={tau:g}, emitted "
                    f"{stats['emitted']:,} == |{{bn >= tau}}| "
                    f"{len(expect_set):,}")

        check("unbudgeted set == shortest_backward", unbudgeted_set_matches)
        check("descending bottleneck order", descending_order)
        check("bitten run = {bn >= tau} prefix", bitten_tau_property)

    print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILED: {FAILS}")
        return 1
    print("RESULT: all production real-data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
