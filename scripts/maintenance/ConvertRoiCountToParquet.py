"""One-time local migration: ROI-count CSVs -> zstd parquet.

``pull_dataset`` now writes ``_roi_count_df.parquet`` (~5x smaller than the
CSV). This script converts CSVs that already exist on disk from older pulls,
verifies the round-trip, and keeps the source CSV unless ``--delete-source``
is passed explicitly. Readers keep CSV fallback, so skipping or interrupting
this migration is always safe.

Usage (from anywhere):
    python scripts/maintenance/ConvertRoiCountToParquet.py --dry-run
    python scripts/maintenance/ConvertRoiCountToParquet.py            # convert, keep CSV
    python scripts/maintenance/ConvertRoiCountToParquet.py --delete-source
    python scripts/maintenance/ConvertRoiCountToParquet.py male-cns_v1_0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

DATASETS_DIR = Path(__file__).resolve().parents[2] / 'datasets'


def convert(csv_path: Path, delete_source: bool = False) -> bool:
    parquet_path = csv_path.with_suffix('.parquet')
    if parquet_path.exists():
        if delete_source:
            try:
                pd.read_parquet(parquet_path, columns=['bodyId', 'roi'])
            except Exception as exc:
                print(f'[FAIL] {csv_path}: existing parquet is unreadable '
                      f'({exc}); CSV kept')
                return False
            csv_path.unlink()
            print(f'[skip] {csv_path.name} (parquet already exists; '
                  'CSV deleted)')
        else:
            print(f'[skip] {csv_path.name} (parquet already exists)')
        return True

    csv_size = csv_path.stat().st_size
    df = pl.read_csv(csv_path, infer_schema_length=10000).to_pandas()
    # The first column is the meaningless RangeIndex written by
    # ``to_csv(index=True)``; pulls no longer store it.
    if len(df.columns) and (df.columns[0] in ('', 'Unnamed: 0')):
        df = df.drop(columns=df.columns[0])
    for col in ('bodyId', 'pre', 'post', 'downstream', 'upstream'):
        if col in df.columns:
            df[col] = df[col].astype('int64')

    tmp_path = parquet_path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp_path, index=False, compression='zstd')

    back = pd.read_parquet(tmp_path)
    if not (back.shape == df.shape and list(back.columns) == list(df.columns)
            and bool((back['bodyId'] == df['bodyId']).all())
            and bool((back['roi'] == df['roi']).all())):
        tmp_path.unlink(missing_ok=True)
        print(f'[FAIL] {csv_path}: round-trip mismatch, CSV kept')
        return False

    tmp_path.replace(parquet_path)
    pq_size = parquet_path.stat().st_size
    action = 'CSV deleted' if delete_source else 'CSV kept'
    if delete_source:
        csv_path.unlink()
    print(f'[ok]   {csv_path.parent.name}: {csv_size / 2**20:8.1f} MiB -> '
          f'{pq_size / 2**20:6.1f} MiB ({csv_size / pq_size:.1f}x smaller; '
          f'{action})')
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert ROI-count CSVs to zstd parquet after a '
                    'verified round-trip. The source CSV is kept by default.')
    parser.add_argument('folders', nargs='*',
                        help='dataset folder names (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the CSVs that would be converted')
    parser.add_argument('--keep-csv', action='store_true',
                        help='keep the source CSV (default)')
    parser.add_argument('--delete-source', action='store_true',
                        help='delete each source CSV only after verification')
    args = parser.parse_args(argv)
    if args.keep_csv and args.delete_source:
        parser.error('--keep-csv and --delete-source are mutually exclusive')
    folders = args.folders
    csvs = sorted(DATASETS_DIR.glob('*/*_allneurons_roi_count_df.csv'))
    if folders:
        csvs = [c for c in csvs if c.parent.name in folders]
    if not csvs:
        scope = f" in {', '.join(folders)}" if folders else ""
        print(f'No ROI-count CSVs{scope} found under {DATASETS_DIR}')
        return 0
    if args.dry_run:
        for csv_path in csvs:
            print(f'[would convert] {csv_path}')
        return 0

    print(f'Converting {len(csvs)} ROI-count CSV(s) to zstd parquet...')
    failed = [c for c in csvs
              if not convert(c, delete_source=args.delete_source)]
    if failed:
        print(f'{len(failed)} conversion(s) failed; their CSVs were kept.')
        return 1
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
