#!/usr/bin/env python
"""F8 golden-master harness (plan-sf-only-edge-budget.md §13).

Runs the SAME cross-dataset query twice — once with the replay batch
(replay_paths=True) and once with legacy per-threshold enumeration
(replay_paths=False) — then compares the two output trees:

- file inventory equality (every relative path in the legacy tree must
  exist in the replay tree and vice versa, modulo per-run empty
  previews);
- value comparison for every shared CSV/JSON: numeric cells with a
  1e-12 tolerance, string cells exactly; timestamp-ish fields
  ('run date', 'fetched_at', ...) are ignored.

Usage (repo root, drocat env):
    python scripts/run_replay_golden_master.py \
        --datasets "male-cns:v1.0" "flywire_FAFB_v783" \
        --source Mi1,Tm3 --target l-LNv \
        --thresholds 3,5,10 --max-interlayer 2

Outputs land in local_data/replay_golden_master/<stamp>_{replay,legacy}/
and the diff report is printed and saved next to them. Exit code 0 =
zero semantic differences.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from comparison.comparison_parameters import ComparisonParameters  # noqa: E402

TIMESTAMP_KEYS = {"run date", "fetched_at", "density_generated_at",
                  "generated", "generated_at", "run_date"}
FLOAT_TOL = 1e-12


def run_variant(root: Path, replay: bool, args) -> Path:
    params = ComparisonParameters(
        datasets=args.datasets,
        source_neurons=[s for s in args.source.split(",") if s],
        target_neurons=[t for t in args.target.split(",") if t],
        output_folder=str(root),
        comparison_mode="path",
        path_mode="all",
        max_interlayer=args.max_interlayer,
        thresholds=args.thresholds,
        replay_paths=replay,
        auto_extend_thresholds=False,
        pathfinding="StrongestFirst",
        graph_edge_limit_bodyid=args.edge_budget,
        max_paths_bodyid=None,
        skip_bodyId=True,
        cache_only=True,
    )
    analyzer = ComparisonAnalyzer(params, verbose=True)
    analyzer.run_comparison()
    analyzer.export_results()
    return root


def _norm_key(key):
    return str(key).strip().lower().replace(" ", "_")


def _values_equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
            if math.isnan(fa) and math.isnan(fb):
                return True
            return abs(fa - fb) <= FLOAT_TOL
        except (TypeError, ValueError):
            return str(a) == str(b)
    return str(a) == str(b)


def compare_csv(path_a: Path, path_b: Path) -> list:
    diffs = []
    try:
        da = pd.read_csv(path_a)
        db = pd.read_csv(path_b)
    except Exception as exc:  # unreadable -> byte-compare fallback
        return ([f"{path_a.name}: unreadable CSV ({exc})"]
                if path_a.read_bytes() != path_b.read_bytes() else [])
    if list(da.columns) != list(db.columns):
        diffs.append(f"{path_a.name}: columns differ")
        return diffs
    if len(da) != len(db):
        diffs.append(f"{path_a.name}: row count {len(da)} != {len(db)}")
        return diffs
    for col in da.columns:
        if _norm_key(col) in TIMESTAMP_KEYS:
            continue
        for i in range(len(da)):
            if not _values_equal(da.at[i, col], db.at[i, col]):
                diffs.append(
                    f"{path_a.name}[{i}].{col}: "
                    f"{da.at[i, col]!r} != {db.at[i, col]!r}")
                if len(diffs) > 50:
                    diffs.append("... (truncated)")
                    break
    return diffs


def compare_json(path_a: Path, path_b: Path) -> list:
    diffs = []

    def walk(a, b, trail):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                if _norm_key(k) in TIMESTAMP_KEYS:
                    continue
                if k not in a or k not in b:
                    diffs.append(f"{path_a.name}:{trail}/{k}: missing on "
                                 f"one side")
                else:
                    walk(a[k], b[k], f"{trail}/{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{path_a.name}:{trail}: list length "
                             f"{len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{trail}[{i}]")
        elif not _values_equal(a, b):
            diffs.append(f"{path_a.name}:{trail}: {a!r} != {b!r}")

    try:
        walk(json.loads(path_a.read_text()),
             json.loads(path_b.read_text()), "")
    except Exception as exc:
        diffs.append(f"{path_a.name}: JSON compare failed ({exc})")
    return diffs


def compare_trees(root_a: Path, root_b: Path) -> list:
    diffs = []
    rel_a = {str(p.relative_to(root_a)) for p in root_a.rglob("*") if p.is_file()}
    rel_b = {str(p.relative_to(root_b)) for p in root_b.rglob("*") if p.is_file()}
    for missing in sorted(rel_a - rel_b):
        diffs.append(f"only in replay tree: {missing}")
    for missing in sorted(rel_b - rel_a):
        diffs.append(f"only in legacy tree: {missing}")
    for rel in sorted(rel_a & rel_b):
        pa, pb = root_a / rel, root_b / rel
        if pa.stat().st_size != pb.stat().st_size:
            diffs.append(f"{rel}: size {pa.stat().st_size} != "
                         f"{pb.stat().st_size}")
            continue
        if pa.suffix == ".csv":
            diffs.extend(compare_csv(pa, pb))
        elif pa.suffix == ".json":
            diffs.extend(compare_json(pa, pb))
        elif pa.read_bytes() != pb.read_bytes():
            diffs.append(f"{rel}: binary content differs")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["male-cns:v1.0", "flywire_FAFB_v783"])
    parser.add_argument("--source", default="Mi1,Tm3")
    parser.add_argument("--target", default="l-LNv")
    parser.add_argument("--thresholds", default="3,5,10",
                        help="comma-separated ascending integers")
    parser.add_argument("--max-interlayer", type=int, default=2)
    parser.add_argument("--edge-budget", type=int, default=0,
                        help="Edge Budget for BOTH variants (0 = off)")
    args = parser.parse_args()
    args.thresholds = sorted(int(t) for t in args.thresholds.split(","))
    args.datasets = list(args.datasets)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = PROJECT_ROOT / "local_data" / "replay_golden_master"
    base.mkdir(parents=True, exist_ok=True)
    root_replay = base / f"{stamp}_replay"
    root_legacy = base / f"{stamp}_legacy"
    print(f"[F8] replay tree: {root_replay}")
    run_variant(root_replay, True, args)
    print(f"[F8] legacy tree: {root_legacy}")
    run_variant(root_legacy, False, args)

    diffs = compare_trees(root_replay, root_legacy)
    report = (f"F8 golden-master report ({stamp})\n"
              f"datasets={args.datasets} thresholds={args.thresholds} "
              f"L{args.max_interlayer}\n"
              f"semantic differences: {len(diffs)}\n"
              + "\n".join(f"  - {d}" for d in diffs[:200]))
    print(report)
    (base / f"{stamp}_report.txt").write_text(report, encoding="utf-8")
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
