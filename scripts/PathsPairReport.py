# CLI for the paths pair report (plan: _plan/plan-paths-pair-report.md).
# Reads the paths tables a run already produced and writes the per-pair
# HTML report + breakdown CSVs INTO the run folder (additive only).
#
#   python scripts/PathsPairReport.py <run_dir> [<run_dir> ...]
#           [--top-per-length 10] [--matrix-rows 50]
#           [--rank-by min_weight|path_prob|length]
import argparse
import sys
from pathlib import Path

# Add project root and src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from paths_pair_report import (  # noqa: E402
    DEFAULT_GLOBAL_EDGES,
    DEFAULT_GLOBAL_PAIRS,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_TOP_PER_LENGTH,
    RANK_KEYS,
    generate_paths_pair_report,
)

try:
    from utils.naming_utils import is_run_folder_name  # noqa: E402
except ImportError:  # pragma: no cover - direct src/ execution
    from src.utils.naming_utils import is_run_folder_name  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Generate the per-source-target-pair HTML report '
                    '(paths_pair_report.html + paths_pair_breakdown/) '
                    'inside each pathfinding run folder.')
    parser.add_argument(
        'run_dirs', nargs='+',
        help='Pathfinding run folders (find-paths-complete_*, '
             'find-paths-shortest_*, cross-dataset_*) or a parent '
             'directory to scan.')
    parser.add_argument(
        '--top-per-length', type=int, default=DEFAULT_TOP_PER_LENGTH,
        help=f'Capped table: top-N paths per (pair, length). '
             f'Default {DEFAULT_TOP_PER_LENGTH}.')
    parser.add_argument(
        '--matrix-rows', type=int, default=DEFAULT_MATRIX_ROWS,
        help=f'Presence matrix rows shown per pair. '
             f'Default {DEFAULT_MATRIX_ROWS}.')
    parser.add_argument(
        '--global-pairs', type=int, default=DEFAULT_GLOBAL_PAIRS,
        help=f'Global tab: pairs shown in the pair x unit matrix. '
             f'Default {DEFAULT_GLOBAL_PAIRS}.')
    parser.add_argument(
        '--global-edges', type=int, default=DEFAULT_GLOBAL_EDGES,
        help=f'Global tab: top-N edges drawn in the global route network. '
             f'Default {DEFAULT_GLOBAL_EDGES}.')
    parser.add_argument(
        '--rank-by', choices=RANK_KEYS, default='min_weight',
        help='Primary rank key (bottleneck-first by default).')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    def scan(root: Path):
        """Yield RUN folders: known prefix+timestamp names first, then a
        fallback for renamed folders that hold a paths table directly.

        A cross-dataset run is yielded whole — its dataset_data/*/minsyn_*/
        delegates become units of ONE report — so we never descend into a
        folder once it is recognized as a run."""
        if not root.is_dir():
            return
        if is_run_folder_name(root.name) or any(
                root.glob('*_allpaths_type.csv')):
            yield root
            return
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith('.'):
                yield from scan(child)

    failures = 0
    for raw in args.run_dirs:
        root = Path(raw)
        candidates = list(scan(root)) if root.is_dir() else [root]
        if not candidates:
            print(f'!! no run folder found: {root}')
            failures += 1
            continue
        for run_dir in candidates:
            try:
                report = generate_paths_pair_report(
                    run_dir,
                    top_per_length=args.top_per_length,
                    matrix_rows=args.matrix_rows,
                    rank_by=args.rank_by,
                    global_pairs=args.global_pairs,
                    global_edges=args.global_edges,
                    log=None if args.quiet else print,
                )
                print(f'✅ {report}')
            except Exception as exc:  # noqa: BLE001 - CLI boundary
                print(f'!! {run_dir}: {exc}')
                failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
