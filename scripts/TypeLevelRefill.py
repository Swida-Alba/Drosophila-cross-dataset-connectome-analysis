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
        compute_type_level_refill,
    )
except ImportError:  # pragma: no cover - direct src/ execution
    from src.type_level_refill import (  # noqa: E402
        DEFAULT_DETAIL_CAP,
        DEFAULT_REFILL_PATH_BUDGET,
        TypeLevelRefillError,
        compute_type_level_refill,
    )

try:
    from utils.label_utils import is_untyped_type_label  # noqa: E402
except ImportError:  # pragma: no cover
    from src.utils.label_utils import is_untyped_type_label  # noqa: E402

try:
    from utils.naming_utils import is_run_folder_name  # noqa: E402
except ImportError:  # pragma: no cover
    from src.utils.naming_utils import is_run_folder_name  # noqa: E402

DEFAULT_OUT = 'type_level_refill_output'
_HEMI_ALIASES = {
    'r': 'R', 'right': 'R', 'rhs': 'R', 'right hemisphere': 'R',
    'l': 'L', 'left': 'L', 'lhs': 'L', 'left hemisphere': 'L',
}


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


def _hemi_codes(frame: pd.DataFrame):
    """Hemisphere codes per the coana rules: hemisphere > somaSide/
    soma side > rootSide, then instance _L/_R, defaulting to 'U'."""
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    col = next((lowered[c] for c in
                ('hemisphere', 'soma side', 'somaside', 'rootside')
                if c in lowered), None)
    codes = pd.Series('U', index=frame.index, dtype=object)
    if col is not None:
        vals = frame[col].fillna('').astype(str).str.strip().str.lower()
        codes = vals.map(_HEMI_ALIASES).fillna('U')
    elif 'instance' in frame.columns:
        inst = frame['instance'].fillna('').astype(str)
        codes = pd.Series('U', index=frame.index, dtype=object)
        codes[inst.str.endswith('_R')] = 'R'
        codes[inst.str.endswith('_L')] = 'L'
    return codes


def build_type_map(neuron_table: Path, separate_hemispheres: bool):
    """bodyId -> EFFECTIVE label: base type (untyped dropped), suffixed
    _L/_R/_U when the run separated hemispheres (coana suffix rules)."""
    frame = pd.read_csv(neuron_table, dtype=str, low_memory=False)
    if 'bodyId' not in frame.columns or 'type' not in frame.columns:
        raise SystemExit(
            f'--neuron-table {neuron_table}: needs bodyId and type '
            f'columns (found {list(frame.columns)[:8]}...).')
    labels = frame['type'].fillna('Unknown').astype(str)
    if separate_hemispheres:
        labels = labels + '_' + _hemi_codes(frame)
    keep = ~labels.map(is_untyped_type_label)
    return dict(zip(frame.loc[keep, 'bodyId'].astype(str),
                    labels[keep]))


def _run_bool(value) -> bool:
    return str(value).strip().lower() in ('true', '1')


def read_run_flags(run_dir: Path):
    """separate_hemispheres / hemisphere_filter from parameters.txt."""
    text = (run_dir / 'parameters.txt').read_text(
        encoding='utf-8', errors='replace')
    import re
    sep = re.search(r'^separate hemispheres:\s*(\S+)', text, re.MULTILINE)
    filt = re.search(r'^hemisphere filter:\s*(\S+)', text, re.MULTILINE)
    return (_run_bool(sep.group(1)) if sep else False,
            filt.group(1) if filt else 'both')


def filter_edges_by_hemisphere(edges, type_map, hemi_filter):
    """Fetch-time edge rule (coana._apply_hemisphere_suffix_to_conn_df):
    'left'/'right' keep an edge only when BOTH endpoints are that side or
    'U'; 'both' keeps everything."""
    if hemi_filter not in ('left', 'right'):
        return edges
    keep_code = hemi_filter[0].upper()
    code_of = {}
    for body_id, label in type_map.items():
        code = label.rsplit('_', 1)[-1] if label.endswith(('_L', '_R', '_U')) \
            else 'U'
        code_of[body_id] = code
    return [e for e in edges
            if code_of.get(e[0], 'U') in (keep_code, 'U')
            and code_of.get(e[1], 'U') in (keep_code, 'U')]


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
            separate, hemi_filter = read_run_flags(run_dir)
            type_map = build_type_map(neuron_table, separate)
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
