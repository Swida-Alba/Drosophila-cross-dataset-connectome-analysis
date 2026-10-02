#!/usr/bin/env python3
"""DROCAT CLI — connection-ratio pathfinding lane (Phase 1 standalone).

Runs the ratio-basis FindAllPath analog (plan-connection-ratio-
pathfinding §2/§5) on a full-dataset connection table and writes a
standalone output folder. Examples:

    # explicit enrollment
    python scripts/ConnectionRatioPaths.py \
        --connections cache/flywire_FAFB_v783/connections.parquet \
        --sources 720575940609627403,720575940622446106 \
        --targets 720575940617691170 --min-ratio 0.001 \
        --max-interlayer 2 --out ratio_out

    # mirror an existing synapse run's enrollment (READ-ONLY: output
    # lands in a SIBLING folder, never inside the run)
    python scripts/ConnectionRatioPaths.py \
        --connections ... --in-run <find-paths run folder> \
        --min-ratio 0.0005
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from connection_ratio_paths import (  # noqa: E402
        ConnectionRatioPathError,
        compute_connection_ratio_paths,
        load_connections,
    )
    from type_level_refill import (  # noqa: E402
        TypeLevelRefillError,
        read_enrollment,
        read_run_provenance,
    )
except ImportError:  # pragma: no cover - direct src/ execution
    from src.connection_ratio_paths import (  # noqa: E402
        ConnectionRatioPathError,
        compute_connection_ratio_paths,
        load_connections,
    )
    from src.type_level_refill import (  # noqa: E402
        TypeLevelRefillError,
        read_enrollment,
        read_run_provenance,
    )


def _flag(text: str, pattern: str, default=None):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip().strip("'\"") if m else default


def read_run_flags(run_dir: Path):
    """dataset / hemisphere filter / mapping file from parameters.txt
    (max_interlayer, separate-hemispheres, intra-type, aggregate method
    come from read_run_provenance)."""
    text = (run_dir / 'parameters.txt').read_text(
        encoding='utf-8', errors='replace')
    separate = _flag(text, r'^separate hemispheres:\s*(\S+)', 'False')
    return (
        _flag(text, r'^dataset:\s*(.+)$', ''),
        _flag(text, r'^hemisphere filter:\s*(\S+)', 'both'),
        str(separate).strip().lower() in ('true', '1'),
        _flag(text, r'^custom mapping file:\s*(.+)$', None),
    )


def load_neuron_table(path: Path):
    import pandas as pd

    if str(path).lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _id_list(raw):
    return [x.strip() for x in raw.split(',') if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--connections', required=True, type=Path,
                    help='full-dataset connection table (parquet/csv, '
                         'natural weights) — the F9 denominator universe')
    ap.add_argument('--sources', help='comma-separated source bodyIds')
    ap.add_argument('--targets', help='comma-separated target bodyIds')
    ap.add_argument('--in-run', type=Path,
                    help='read enrollment + flags from an existing '
                         'find-paths run folder (READ-ONLY; output goes '
                         'to a sibling folder)')
    ap.add_argument('--min-ratio', type=float, required=True,
                    help='ratio threshold t_r, 0 < t_r <= 1')
    ap.add_argument('--max-interlayer', type=int)
    ap.add_argument('--edge-budget', type=int, default=0,
                    help='Edge Budget cap on the closed cone (0 = off)')
    ap.add_argument('--path-budget', type=int, default=0,
                    help='StrongestFirst path budget (0 = 1,000,000)')
    ap.add_argument('--max-probes', type=int, default=8,
                    help='edge-budget tier-probe cap (production default 8)')
    ap.add_argument('--neuron-table', type=Path,
                    help='dataset neuron table (bodyId/type[/hemisphere]) '
                         'for the effective type map + admission guards')
    ap.add_argument('--dataset', default='')
    ap.add_argument('--mapping-file', type=Path, default=None,
                    help='recorded custom mapping JSON (label mapper)')
    ap.add_argument('--separate-hemispheres', action='store_true')
    ap.add_argument('--hemisphere-filter', default='both',
                    choices=('both', 'left', 'right'))
    ap.add_argument('--exclude-intra-type', action='store_true')
    ap.add_argument('--keep-untyped', action='store_true',
                    help='disable the drop-untyped admission guard '
                         '(default ON, production parity)')
    ap.add_argument('--aggregate-method', default='product',
                    choices=('product', 'average', 'ratio'))
    ap.add_argument('--out', type=Path, default=None,
                    help='output folder (default: sibling of --in-run)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    sources = targets = None
    dataset = args.dataset
    hemi_filter = args.hemisphere_filter
    separate = args.separate_hemispheres
    mapping_file = args.mapping_file
    max_interlayer = args.max_interlayer
    exclude_intra = args.exclude_intra_type
    aggregate = args.aggregate_method
    source_note = 'CLI (explicit enrollment)'
    try:
        if args.in_run is not None:
            sources, targets = read_enrollment(args.in_run)
            prov = read_run_provenance(args.in_run)
            r_dataset, hemi_filter, separate, mapping_file = \
                read_run_flags(args.in_run)
            dataset = dataset or r_dataset
            if max_interlayer is None:
                max_interlayer = prov.get('max_interlayer')
            exclude_intra = (
                str(prov.get('exclude_intra_type_connections',
                             'False')).strip().lower() in ('true', '1'))
            aggregate = prov.get('aggregate_method') or aggregate
            source_note = f'CLI (--in-run {args.in_run.name})'
        else:
            if not args.sources or not args.targets:
                ap.error('--sources and --targets (or --in-run) required.')
            sources = _id_list(args.sources)
            targets = _id_list(args.targets)
        if not max_interlayer or int(max_interlayer) < 1:
            ap.error('--max-interlayer >= 1 required (directly or via '
                     '--in-run).')
        if not sources or not targets:
            raise ConnectionRatioPathError(
                'empty enrollment: no sources or no Checked targets.')

        neuron_frame = None
        if args.neuron_table is not None:
            neuron_frame = load_neuron_table(args.neuron_table)
        elif mapping_file is not None:
            raise ConnectionRatioPathError(
                '--mapping-file without --neuron-table: the effective '
                'type map needs the neuron table.')

        connections = load_connections(args.connections)
        rec = compute_connection_ratio_paths(
            connections=connections, sources=sources, targets=targets,
            min_ratio=args.min_ratio, max_interlayer=int(max_interlayer),
            neuron_frame=neuron_frame, dataset=dataset,
            mapping_file=mapping_file, separate_hemispheres=separate,
            hemi_filter=hemi_filter,
            edge_budget=(args.edge_budget or None),
            path_budget=(args.path_budget or 1_000_000),
            aggregate_method=aggregate,
            drop_untyped=(not args.keep_untyped),
            exclude_intra_type=exclude_intra,
            max_probes=args.max_probes,
            out_dir=args.out, run_dir=args.in_run,
            source_note=source_note)

        if not args.quiet:
            prov = rec['provenance']
            disc = rec['discovery']
            print(f"status: {rec['status']}")
            print(f"cone: {disc.get('cone_rows', 0):,} edges "
                  f"({disc.get('cone_nodes', 0):,} nodes)")
            print(f"paths: {len(rec['paths']):,}  "
                  f"tau: {prov.get('strongest_first_tau')}  "
                  f"bitten: {prov.get('strongest_first_budget_bitten')}  "
                  f"W*: {prov.get('strongest_retained_bottleneck')}")
            print(f"applied threshold: {prov.get('applied_threshold')} "
                  f"({prov.get('applied_threshold_source')})")
            if rec.get('out_dir'):
                print(f"output: {rec['out_dir']}")
        return 0
    except (ConnectionRatioPathError, TypeLevelRefillError) as exc:
        print(f'REFUSED — {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
