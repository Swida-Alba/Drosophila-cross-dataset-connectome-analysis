#!/usr/bin/env python3
"""Real-data comparison: one-shot edge budget (Fix D) vs budget-fit search
(§7.4), on the local flywire_FAFB_v783 merged-connections cache.

Pipeline mirrors coana's `_find_paths_core` forward discovery (forward_only,
per-layer fetch from the newly-discovered frontier, threshold filter, no
early stop in 'all' mode), the initial lossless hop-budget prune, then:

  - one-shot: floor at w1 + 1, drop, ONE lossless pass   (apply_edge_budget_floor)
  - budget-fit: gallop + bisection over weight tiers, single-pass probes
    (fit_budget from scripts/harness/verify_budget_fit_pruning.py)

Payoff metric: run the real FastGraph.find_paths_strongest_first on each
kept cone with the SAME path budget and compare emitted paths / tau —
more paths at the same memory cap is the utilization win.

Run:  python3 scripts/harness/compare_budget_fit_real_data.py
"""
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "harness"))

import polars as pl                                       # noqa: E402
from vispath_pkg.fast_graph_core import FastGraph          # noqa: E402
import verify_budget_fit_pruning as bfp                    # noqa: E402

DATASET = "flywire_FAFB_v783"
CONN = ROOT / "datasets" / DATASET / f"{DATASET}_merged_connections.parquet"
INDEX = ROOT / "neuron_indexes" / DATASET / "neuron_index.parquet"
SOURCE_TYPES = ["LC4", "LPLC1"]
SOURCE_PER_TYPE = 4
TARGET_TYPES = ["MBON01", "MBON03", "PPL101"]
PATH_BUDGET = 100_000
CAPS = [1_000_000, 250_000, 50_000]
MAX_PROBES = 8


def pick_neurons(index: pl.DataFrame, types, per_type=None):
    ids = []
    for t in types:
        sub = index.filter(pl.col("type") == t)["bodyId"].to_list()
        if not sub:
            raise SystemExit(f"type {t!r} not in index")
        take = sub if per_type is None else sub[:per_type]
        ids.extend(str(x) for x in take)
    return ids


def discover(conn: pl.DataFrame, sources, max_interlayer):
    """Forward layer-by-layer discovery (forward_only=True): table l =
    outgoing connections of the neurons newly discovered at layer l.
    'all' mode never stops early — the depth cap is the only bound."""
    reached = set(sources)
    frontier = list(sources)
    tables = []
    for layer in range(max_interlayer + 1):
        t = (conn.filter(pl.col("bodyId_pre").is_in(frontier))
                 .select("bodyId_pre", "bodyId_post", "weight"))
        tables.append(t)
        posts = set(t["bodyId_post"].to_list()) - reached
        reached |= posts
        frontier = list(posts)
        print(f"    layer {layer}: {t.height:,} rows, "
              f"{len(posts):,} new neurons")
    return tables, reached


def cone_to_edges(tables):
    """Intern bodyIds -> int codes; return (edge list, id -> code map)."""
    union = pl.concat(tables)
    ids = (pl.concat([union.select(pl.col("bodyId_pre").alias("id")),
                      union.select(pl.col("bodyId_post").alias("id"))])
           .unique()["id"].to_list())
    code = {x: i for i, x in enumerate(ids)}
    pre = (union["bodyId_pre"].replace_strict(code, default=None,
                                              return_dtype=pl.Int64).to_list())
    post = (union["bodyId_post"].replace_strict(code, default=None,
                                                return_dtype=pl.Int64).to_list())
    w = union["weight"].to_list()
    return list(zip(pre, post, w)), code


def build_graph(kept):
    g = FastGraph()
    for u, v, w in kept:
        g.add_edge(u, v, w)
    return g


def enumerate_paths(kept, sources, targets, bound, label):
    g = build_graph(kept)
    stats = {}
    t0 = time.perf_counter()
    for _ in g.find_paths_strongest_first(sources, targets, bound,
                                          budget=PATH_BUDGET, stats=stats):
        pass
    dt = time.perf_counter() - t0
    print(f"      {label:10s} graph {g.number_of_edges():>10,} edges / "
          f"{g.number_of_nodes():>8,} nodes | paths {stats.get('emitted'):>9,} | "
          f"tau {stats.get('tau')} | bitten {stats.get('budget_bitten')} | {dt:.1f}s")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_scenario(tag, conn, index, max_interlayer, target_types,
                 targets_all=False, caps=None, enumerate=True):
    caps = caps or CAPS
    bound = max_interlayer + 1
    sources = pick_neurons(index, SOURCE_TYPES, SOURCE_PER_TYPE)
    targets = pick_neurons(index, target_types, None if targets_all else 2)
    print(f"\n=== scenario {tag}: max_interlayer={max_interlayer} "
          f"(bound {bound}), min_syn=3, targets={target_types}"
          f"{' (all)' if targets_all else ''} ===")
    print(f"  sources ({len(sources)}): {SOURCE_TYPES} | "
          f"targets ({len(targets)}) | rss {rss_mb():,.0f} MB")

    t0 = time.perf_counter()
    tables, reached = discover(conn, sources, max_interlayer)
    targets_found = [t for t in targets if t in reached]
    print(f"  discovery: {time.perf_counter() - t0:.1f}s | "
          f"targets found {len(targets_found)}/{len(targets)}")
    if not targets_found:
        print("  no targets reachable — skipping scenario")
        return

    edges, code = cone_to_edges(tables)
    src = [code[s] for s in sources if s in code]
    tgt = [code[s] for s in targets_found]
    print(f"  cone: {len(edges):,} rows, {len(code):,} neurons | "
          f"rss {rss_mb():,.0f} MB")

    t0 = time.perf_counter()
    e0, _ = bfp.hop_pass(edges, src, tgt, bound)
    t_prune = time.perf_counter() - t0
    print(f"  initial lossless prune: {len(edges):,} -> {len(e0):,} rows "
          f"({t_prune:.1f}s)")
    if not e0:
        print("  cone empty after pruning — skipping scenario")
        return

    for cap in CAPS:
        print(f"  -- cap {cap:,} --")
        t0 = time.perf_counter()
        one = bfp.oneshot_fix_d(e0, cap, src, tgt, bound)
        t_one = time.perf_counter() - t0
        if one['status'] == 'floored':
            print(f"    one-shot : floored  w0={one['threshold']:.0f}  "
                  f"kept {len(one['kept']):,}  slack {cap - len(one['kept']):,}  "
                  f"[{t_one:.2f}s]")
        else:
            print(f"    one-shot : {one['status']}  kept {len(one['kept']):,}  "
                  f"[{t_one:.2f}s]")

        t0 = time.perf_counter()
        fit = bfp.fit_budget(e0, cap, src, tgt, bound,
                             max_probes=MAX_PROBES, max_passes=1)
        t_fit = time.perf_counter() - t0
        traj = ", ".join(f"{t}:{c:,}" for t, c, _ in fit['probes'])
        if fit['status'] == 'floored':
            print(f"    fit      : floored  t*={fit['threshold']:.0f}  "
                  f"kept {len(fit['kept']):,}  slack {cap - len(fit['kept']):,}  "
                  f"truncated={fit['truncated']}  [{t_fit:.2f}s]")
        else:
            print(f"    fit      : {fit['status']}  kept {len(fit['kept']):,}  "
                  f"truncated={fit['truncated']}  [{t_fit:.2f}s]")
        print(f"    fit trajectory (weight:closed-edges): {traj}")
        print(f"    probe wall times: "
              + ", ".join(f"{t}:{s:.2f}s" for (t, _c, _p), s
                          in zip(fit['probes'], fit['probe_seconds'])))
        print(f"    search overhead: fit {t_fit:.2f}s = "
              f"{t_fit / max(t_one, 1e-9):.1f}x one-shot "
              f"(initial prune above: {t_prune:.2f}s) | rss {rss_mb():,.0f} MB")

        if not enumerate:
            continue
        print(f"    path yield (StrongestFirst budget {PATH_BUDGET:,}, "
              f"cutoff {bound}):")
        enumerate_paths(e0, src, tgt, bound, "no floor")
        if one['status'] == 'floored':
            enumerate_paths(one['kept'], src, tgt, bound, "one-shot")
        enumerate_paths(fit['kept'], src, tgt, bound, "fit")


def main():
    print(f"dataset: {DATASET}")
    index = pl.read_parquet(INDEX)
    conn = (pl.scan_parquet(CONN)
            .filter(pl.col("weight") >= 3)
            .with_columns(pl.col("bodyId_pre").cast(pl.Utf8),
                          pl.col("bodyId_post").cast(pl.Utf8))
            .select("bodyId_pre", "bodyId_post", "weight")
            .collect())
    print(f"connections loaded: {len(conn):,} rows (weight >= 3)")
    scenarios = {
        "A": lambda: run_scenario("A: shallow, 6 targets", conn, index,
                                  max_interlayer=2, target_types=TARGET_TYPES),
        "B": lambda: run_scenario("B: deep, 6 targets", conn, index,
                                  max_interlayer=3, target_types=TARGET_TYPES),
        "C": lambda: run_scenario("C: deep, broad targets", conn, index,
                                  max_interlayer=3, target_types=["Tm3"],
                                  targets_all=True),
        "D": lambda: run_scenario("D: deep+wide, million-scale", conn, index,
                                  max_interlayer=4,
                                  target_types=["Tm3", "Mi1"], targets_all=True,
                                  caps=[1_000_000, 250_000], enumerate=False),
    }
    only = sys.argv[1].upper() if len(sys.argv) > 1 else None
    for key, fn in scenarios.items():
        if only in (None, key):
            fn()


if __name__ == "__main__":
    main()
