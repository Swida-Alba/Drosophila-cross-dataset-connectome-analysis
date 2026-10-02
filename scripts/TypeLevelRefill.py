# CLI for the type-level connection-strength refill (plan:
# _plan/plan-type-level-refill.md, Phase 1).
#
# Reads a budget-bitten FindAllPath ('all' mode) run folder and writes the
# standalone refill records OUTSIDE the run folder (additive only; the
# run's own files are never touched — same convention as the homolog CLI).
#
#   python scripts/TypeLevelRefill.py <run_dir> [<run_dir> ... | scan root]
#           --connections <connections.parquet|csv>
#           --neuron-table <*_allneurons_neuron_df.csv>
#           [--out DIR] [--detail-cap N] [--path-budget N] [--quiet]
#
# The connection source must be the FULL dataset connections (natural
# weights, no threshold) and the neuron table the same one the run used
# (labels are resolved to the run's effective convention: hemisphere
# suffixes are applied when the run separated hemispheres, untyped ids
# are dropped).
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd  # noqa: E402

try:
    from type_level_refill import (  # noqa: E402
        DEFAULT_DETAIL_CAP,
        DEFAULT_REFILL_PATH_BUDGET,
        TypeLevelRefillError,
        build_effective_type_map,
        compute_type_level_refill,
        filter_edges_by_hemisphere,
    )
except ImportError:  # pragma: no cover - direct src/ execution
    from src.type_level_refill import (  # noqa: E402
        DEFAULT_DETAIL_CAP,
        DEFAULT_REFILL_PATH_BUDGET,
        TypeLevelRefillError,
        build_effective_type_map,
        compute_type_level_refill,
        filter_edges_by_hemisphere,
    )

try:
    from utils.naming_utils import is_run_folder_name  # noqa: E402
except ImportError:  # pragma: no cover
    from src.utils.naming_utils import is_run_folder_name  # noqa: E402

DEFAULT_OUT = 'type_level_refill_output'


def scan_run_folders(root: Path):
    """Yield run folders: known prefix+timestamp names, then renamed
    folders that directly hold the refill's required artifacts."""
    found = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and is_run_folder_name(entry.name):
            found.append(entry)
    if found:
        yield from found
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / 'parameters.txt').exists() \
                and (entry / 'data_details' / 'connection_type.csv').exists():
            yield entry


def load_connections(path: Path):
    """Full connection source as (pre, post, weight) string-typed triples."""
    if str(path).lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    missing = {'bodyId_pre', 'bodyId_post', 'weight'} - set(df.columns)
    if missing:
        raise SystemExit(
            f'--connections {path}: missing columns {sorted(missing)}.')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df[df['weight'].notna()]
    return [(str(u), str(v), float(w))
            for u, v, w in zip(df['bodyId_pre'], df['bodyId_post'],
                               df['weight'])]


def _run_bool(value) -> bool:
    return str(value).strip().lower() in ('true', '1')


def read_run_flags(run_dir: Path):
    """separate_hemispheres / hemisphere_filter / mapping file / dataset
    from parameters.txt."""
    import re
    text = (run_dir / 'parameters.txt').read_text(
        encoding='utf-8', errors='replace')

    def _flag(pattern, default=None):
        m = re.search(pattern, text, re.MULTILINE)
        return m.group(1).strip().strip("'\"") if m else default

    return (
        _run_bool(_flag(r'^separate hemispheres:\s*(\S+)', 'False')),
        _flag(r'^hemisphere filter:\s*(\S+)', 'both'),
        _flag(r'^custom mapping file:\s*(.+)$'),   # may contain spaces
        _flag(r'^dataset:\s*(\S+)'),
    )


def build_type_map(neuron_table: Path, separate_hemispheres: bool,
                   mapping_file=None, dataset=None):
    """bodyId -> EFFECTIVE label via the shared module builder (label
    mapping with the same semantics as the run's type-level aggregation,
    hemisphere suffixes, untyped drop)."""
    frame = pd.read_csv(neuron_table, dtype=str, low_memory=False)
    if 'bodyId' not in frame.columns or 'type' not in frame.columns:
        raise SystemExit(
            f'--neuron-table {neuron_table}: needs bodyId and type '
            f'columns (found {list(frame.columns)[:8]}...).')
    if mapping_file and mapping_file.lower() in ('none', 'n/a', ''):
        mapping_file = None
    return build_effective_type_map(
        frame, dataset=dataset or 'unknown',
        mapping_file=(Path(mapping_file) if mapping_file else None),
        separate_hemispheres=separate_hemispheres)




def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Compute standalone type-level refill records for '
                    'budget-bitten FindAllPath runs (applied threshold > '
                    'asked threshold).')
    parser.add_argument(
        'run_dirs', nargs='+',
        help='Pathfinding run folders (find-paths-complete_*) or parent '
             'directories to scan.')
    parser.add_argument(
        '--connections', required=True,
        help='FULL dataset connections (.parquet or .csv) with '
             'bodyId_pre/bodyId_post/weight at natural weights.')
    parser.add_argument(
        '--neuron-table', required=True,
        help='Neuron table CSV (bodyId, type[, hemisphere|somaSide|'
             'rootSide|instance]) — the same table the run used.')
    parser.add_argument(
        '--out', default=DEFAULT_OUT,
        help=f'Output root (per-run subfolders). Default ./{DEFAULT_OUT}. '
             f'The run folders themselves are never modified.')
    parser.add_argument(
        '--in-run', action='store_true',
        help='Write the records INTO each run folder under '
             'data_details/type_level_refill/ instead of --out (additive '
             'only — existing run files are never touched; the run guide '
             'surfaces them when present).')
    parser.add_argument('--detail-cap', type=int, default=DEFAULT_DETAIL_CAP,
                        help=f'Row cap for refill_bodyId_pairs.csv '
                             f'(summary never capped). '
                             f'Default {DEFAULT_DETAIL_CAP}.')
    parser.add_argument(
        '--path-budget', type=int, default=DEFAULT_REFILL_PATH_BUDGET,
        help=f'Enumeration guard for the asked-threshold re-exploration. '
             f'Default {DEFAULT_REFILL_PATH_BUDGET}.')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    connections_path = Path(args.connections)
    neuron_table = Path(args.neuron_table)
    for path, flag in ((connections_path, '--connections'),
                       (neuron_table, '--neuron-table')):
        if not path.exists():
            raise SystemExit(f'{flag}: {path} does not exist.')

    folders = []
    for entry in args.run_dirs:
        path = Path(entry)
        if not path.is_dir():
            raise SystemExit(f'{entry}: not a directory.')
        if (path / 'parameters.txt').exists() and (
                path / 'data_details' / 'connection_type.csv').exists():
            folders.append(path)          # a direct run folder
        else:
            folders.extend(scan_run_folders(path))
    if not folders:
        raise SystemExit('No run folders found (need parameters.txt + '
                         'data_details/connection_type.csv).')

    out_root = Path(args.out)
    failures = 0
    for run_dir in folders:
        try:
            separate, hemi_filter, mapping_file, dataset = read_run_flags(
                run_dir)
            if mapping_file and not Path(mapping_file).exists():
                raise SystemExit(
                    f'{run_dir.name}: parameters.txt records custom mapping '
                    f'file {mapping_file!r} which does not exist — the '
                    f'refill cannot reproduce the run\'s labels without it.')
            type_map = build_type_map(neuron_table, separate,
                                      mapping_file=mapping_file,
                                      dataset=dataset)
            edges = filter_edges_by_hemisphere(
                load_connections(connections_path), type_map, hemi_filter)
            rec = compute_type_level_refill(
                run_dir, edges=edges, type_map=type_map,
                detail_cap=args.detail_cap, path_budget=args.path_budget,
                out_dir=(None if args.in_run else out_root / run_dir.name),
                source_note=f'CLI: connections={connections_path.name}, '
                            f'neuron_table={neuron_table.name}')
            if rec['status'] == 'refilled':
                msg = (f"refilled: {rec['refill_edges']} bodyId edges / "
                       f"{rec['refill_weight_total']} syn "
                       f"(w0={rec['edge_weight_floor']}, "
                       f"bitten={rec['strongest_first_budget_bitten']})")
            else:
                msg = rec['status']
            if not args.quiet:
                print(f'{run_dir.name}: {msg}')
        except TypeLevelRefillError as exc:
            failures += 1
            print(f'{run_dir.name}: REFUSED — {exc}', file=sys.stderr)
    if failures:
        print(f'{failures} run folder(s) refused.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
