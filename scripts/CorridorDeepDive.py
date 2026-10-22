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

Outputs land in <run_dir>/route_scope_paths/ (Fast) or
<run_dir>/route_scope_paths_type_pruned/ (typed) in run-folder mode, or
--out (both modes): route_paths_type.csv (one row per validated walk),
route_paths_bodyId.csv (Fast lane), route_connection_type.csv,
route_type_incoming_asked.csv, route_provenance.json, README.md — plus the
lane's OWN second round at synapse >= 1 (route_full_paths_type.csv,
route_full_carrying_paths_type.csv, route_full_type_pairs.csv,
route_type_incoming.csv and the route_full_type_{network,heatmap}.html
mirrors; opt-in for the typed lane via --full-scope) — plus plan.json under
data_details/route_plan/ beside the output root.

The typed lane validates each type walk with ONE layered forward/backward
pass (issue 2026-10-23). It writes no bodyId witness rows and has no
realization envelope; label-revisiting walks carry the disclosed layered
superset masses.

Per-pair shortest scope (typed lane only, issue 2026-10-22):
--per-pair-scope replaces the global bound with min(Lt + --scope-pad,
--scope-hard-cap) per (source,target) pair. --scope-lt-mode discover
computes Lt locally and writes scope_table.csv; external reads
--scope-table. route_scope_pairs.csv carries the per-pair statuses.
Exit codes: 0 complete, 2 refused (deterministic reason).
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
                    help="hop bound = max_interlayer + 1 (default 4); "
                         "ignored when --per-pair-scope is on (each pair "
                         "uses min(Lt + pad, hard cap))")
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
                    help="Fast lane's REALIZATION envelope per (pair, "
                         "length) class. The typed lane validates with one "
                         "layered forward/backward pass and ignores this "
                         "(issue 2026-10-23).")
    ap.add_argument("--no-bodyid", action="store_true",
                    help="skip route_paths_bodyId.csv (Fast lane only; the "
                         "typed lane never writes bodyId witness rows)")
    ap.add_argument("--full-scope", action="store_true",
                    default=None,
                    help="also run the synapse >= 1 second round for the "
                         "typed lane. It is off by default there (layered "
                         "validation is the production contract; the Fast "
                         "lane keeps its existing default).")
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
    ap.add_argument("--type-pruned", action="store_true",
                    help="Phase B: prune the type graph coarsely, enumerate "
                         "EVERY type walk over it, validate each at bodyId "
                         "level and apply the exact realized "
                         "min_ratio_adj filter. No top-M share.")
    ap.add_argument("--type-prune-basis",
                    choices=["ratio_adj", "connection_ratio", "synapse"],
                    default="ratio_adj",
                    help="coarse pair-level prune basis (default ratio_adj)")
    ap.add_argument("--type-prune-scope", default="full_floor",
                    choices=["full_floor"],
                    help="universe the type graph is built from (only "
                         "full_floor carries a completeness claim)")
    ap.add_argument("--type-min-ratio-adj", type=float, default=0.01,
                    help="pair-level adjusted-ratio floor for ratio bases "
                         "AND the exact realized-path gate (default 0.01)")
    ap.add_argument("--type-min-synapse", type=int, default=0,
                    help="optional pair mass floor (w_sum) on the prune")
    ap.add_argument("--type-min-wmax", type=int, default=0,
                    help="optional bodyId realization floor (w_max) on the "
                         "prune")
    ap.add_argument("--type-walk-cap", type=int, default=2_000_000,
                    help="disclosed cap on the complete type-walk "
                         "enumeration (a cap forbids completeness)")
    ap.add_argument("--type-revisit-policy", choices=["simple", "bounded"],
                    default="bounded",
                    help="bounded (default) is the complete revisit set "
                         "for the hop bound when --type-max-type-visits=0; "
                         "simple forbids repeated types")
    ap.add_argument("--type-max-type-visits", type=int, default=0,
                    help="bounded-policy visit cap per type; 0 = the hop "
                         "bound (the complete revisit set for that bound)")
    ap.add_argument("--type-validate-work-cap", type=int, default=2_000_000,
                    help="legacy typed-lane knobs: the layered "
                         "forward/backward validation has no witness work "
                         "cap, so this is accepted but ignored")
    ap.add_argument("--per-pair-scope", action="store_true",
                    help="replace the global max_interlayer+1 bound with a "
                         "per-(source,target) scope derived from validated "
                         "shortest paths: bound = min(Lt + pad, hard cap). "
                         "Requires --type-pruned.")
    ap.add_argument("--scope-lt-mode", choices=["external", "discover"],
                    default=None,
                    help="external: read --scope-table; discover: compute "
                         "Lt_i locally and write scope_table.csv (default: "
                         "external when --scope-table is given, else discover)")
    ap.add_argument("--scope-table",
                    help="CSV with source_type,target_type and max_hops "
                         "(or scope_bound/bound); optional Lt_i/Lt metadata")
    ap.add_argument("--scope-pad", type=int, default=2,
                    help="hops added to the validated shortest length "
                         "(default 2)")
    ap.add_argument("--scope-hard-cap", type=int, default=6,
                    help="absolute hop cap on every per-pair bound; 0 = no "
                         "cap (default 6)")
    ap.add_argument("--scope-discovery-cap", type=int, default=6,
                    help="deepest hop count Pass A may search when "
                         "discovering Lt_i (default 6)")
    ap.add_argument("--trigger", default="run-folder",
                    help="provenance trigger label (the in-run hook "
                         "passes 'in-run (subprocess)')")
    ap.add_argument("--out",
                    help="output dir (both modes; default: the run folder's "
                         "route_scope_paths/ or "
                         "route_scope_paths_type_pruned/)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-infer-type-pruned", action="store_true",
                    help="do not adopt type_pruned from the run folder's "
                         "parameters.txt (the in-run Fast subprocess uses "
                         "this so the typed lane keeps its own namespace)")
    args = ap.parse_args()
    if args.scope_table and not args.per_pair_scope:
        ap.error("--scope-table requires --per-pair-scope")
    if args.scope_lt_mode and not args.per_pair_scope:
        ap.error("--scope-lt-mode requires --per-pair-scope")

    from route_plan import (FAST_LANE_DIR, TYPED_LANE_DIR, CorridorRefusal,
                            run_lane)

    common = dict(corridor_mode=("typed" if args.type_pruned else "both"),
                  allow_multi_visit=args.allow_multi_visit,
                  certify_actual=args.certify_actual,
                  trigger=args.trigger,
                  top_m_base=args.top_m_base,
                  length_decay=args.length_decay,
                  per_pair_length_budget=args.pair_length_budget,
                  write_bodyid=not args.no_bodyid,
                  full_scope=(True if args.full_scope else None),
                  time_budget_s=(args.time_budget or None),
                  type_pruned=args.type_pruned,
                  type_prune_basis=args.type_prune_basis,
                  type_prune_scope=args.type_prune_scope,
                  type_min_ratio_adj=args.type_min_ratio_adj,
                  type_min_synapse=args.type_min_synapse,
                  type_min_wmax=args.type_min_wmax,
                  type_walk_cap=args.type_walk_cap,
                  type_revisit_policy=args.type_revisit_policy,
                  type_max_type_visits=args.type_max_type_visits,
                  type_validate_work_cap=args.type_validate_work_cap,
                  per_pair_scope=args.per_pair_scope,
                  scope_lt_mode=args.scope_lt_mode,
                  scope_table=args.scope_table,
                  scope_pad=args.scope_pad,
                  scope_hard_cap=args.scope_hard_cap,
                  scope_discovery_cap=args.scope_discovery_cap,
                  infer_type_pruned=not args.no_infer_type_pruned)
    try:
        if args.run_dir:
            out = (Path(args.out) if args.out else
                   Path(args.run_dir) / (TYPED_LANE_DIR if args.type_pruned
                                         else FAST_LANE_DIR))
            run_lane(args.run_dir, connections_path=args.connections,
                     neuron_table=args.neuron_table, out_dir=out, **common)
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
