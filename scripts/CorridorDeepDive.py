"""Corridor lane CLI (plan-corridor-lane.md §6/§23.3): run the standalone
route-scoped lane against a run folder or a direct query. Both modes
delegate to route_plan.run_lane — the driver never re-implements lane
logic (the duplicated body diverged from the library once already).

Run-folder mode (reads parameters.txt; mirror of scripts/TypeLevelRefill.py):
    python scripts/CorridorDeepDive.py <run_dir> \
        --connections <connections.parquet> --neuron-table <neurons.csv|parquet>

Direct mode:
    python scripts/CorridorDeepDive.py \
        --dataset-dir cache/flywire_FAFB_v783 \
        --neuron-table datasets/flywire_FAFB_v783/flywire_FAFB_v783_allneurons_neuron_df.parquet \
        --sources aMe12 --targets PPL101 --bound 4 --asked 1 --out <dir>

Outputs land in <run_dir>/route_scope_paths/ (run-folder mode) or --out
(direct mode): route_paths_type.csv (one row per validated walk),
route_paths_bodyId.csv, route_connection_type.csv, route_type_incoming_asked.csv,
route_provenance.json, README.md — plus the lane's OWN second round at
synapse >= 1 (route_full_paths_type.csv, route_full_carrying_paths_type.csv,
route_full_type_pairs.csv, route_type_incoming.csv and the
route_full_type_{network,heatmap}.html mirrors) — plus plan.json under
data_details/route_plan/ beside the output root. Exit codes: 0 complete,
2 refused (deterministic reason).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vispath-subproject" / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="run folder (parameters.txt)")
    ap.add_argument("--connections",
                    help="connections parquet (default: resolved from the "
                         "run / fnc / dataset layout)")
    ap.add_argument("--neuron-table",
                    help="neuron table parquet/csv for direct mode")
    ap.add_argument("--dataset-dir",
                    help="direct mode: dir holding connections.parquet")
    ap.add_argument("--dataset", help="dataset name for label parity")
    ap.add_argument("--sources", help="comma-separated source types")
    ap.add_argument("--targets", help="comma-separated target types")
    ap.add_argument("--bound", type=int, default=4,
                    help="hop bound = max_interlayer + 1 (default 4)")
    ap.add_argument("--asked", type=int, default=1)
    ap.add_argument("--top-m-base", type=int, default=10,
                    help="route share base: each (pair, hop-length) class "
                         "publishes up to round(top_m_base * "
                         "length_decay**(h-1)) VALID TYPE-LEVEL ROUTES "
                         "(15 at h2 by default). Counted in accepted routes, "
                         "never in a route's bodyId realizations "
                         "(2026-10-16).")
    ap.add_argument("--length-decay", type=float, default=1.5,
                    help="per-hop decay applied to --top-m-base")
    ap.add_argument("--pair-length-budget", type=int, default=100,
                    help="the class's REALIZATION envelope: how many bodyId "
                         "witness paths the materialization pass may "
                         "enumerate per (pair, length) class. Every accepted "
                         "route keeps at least one witness, so this bounds "
                         "the tie mass, never the route list.")
    ap.add_argument("--no-bodyid", action="store_true",
                    help="skip route_paths_bodyId.csv")
    ap.add_argument("--time-budget", type=float, default=900.0,
                    help="wall-clock budget in seconds (0 = unlimited)")
    ap.add_argument("--allow-multi-visit", action="store_true",
                    help="plan walks that revisit a type (default: "
                         "disabled — reciprocal-heavy subtrees explode "
                         "the drain for little gain)")
    ap.add_argument("--certify-actual", action="store_true",
                    help="certify actual-strength selection: keep "
                         "accepting candidates per class until the first "
                         "skipped candidate's bound drops below the "
                         "weakest exported path's real bottleneck -- the "
                         "exported top-share is then the true strongest "
                         "actual routes (slower: cost tracks bound "
                         "tightness; F-PF-002 D2-B)")
    ap.add_argument("--trigger", default="run-folder",
                    help="provenance trigger label (the in-run hook "
                         "passes 'in-run (subprocess)')")
    ap.add_argument("--out", help="direct-mode output dir")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from route_plan import CorridorRefusal, run_lane

    common = dict(corridor_mode="both",
                  allow_multi_visit=args.allow_multi_visit,
                  certify_actual=args.certify_actual,
                  trigger=args.trigger,
                  top_m_base=args.top_m_base,
                  length_decay=args.length_decay,
                  per_pair_length_budget=args.pair_length_budget,
                  write_bodyid=not args.no_bodyid,
                  time_budget_s=(args.time_budget or None))
    try:
        if args.run_dir:
            run_lane(args.run_dir, connections_path=args.connections,
                     neuron_table=args.neuron_table, **common)
            out = Path(args.run_dir) / "route_scope_paths"
        else:
            if not (args.dataset_dir and args.sources and args.targets
                    and args.out):
                ap.error("direct mode needs --dataset-dir --sources "
                         "--targets --out")
            run_lane(None,
                     connections_path=args.connections
                     or (Path(args.dataset_dir) / "connections.parquet"),
                     neuron_table=args.neuron_table,
                     dataset=args.dataset,
                     asked=args.asked,
                     max_interlayer=args.bound - 1,
                     source_types=args.sources.split(","),
                     target_types=args.targets.split(","),
                     out_dir=args.out, **common)
            out = Path(args.out)
    except CorridorRefusal as exc:
        print(f"corridor lane refused: {exc}")
        raise SystemExit(2)
    if not args.quiet:
        print(f"corridor lane complete: {out}")


if __name__ == "__main__":
    main()
