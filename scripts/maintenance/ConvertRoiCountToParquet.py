"""One-time local migration: ROI-count CSVs -> zstd parquet.

``pull_dataset`` now writes ``_roi_count_df.parquet`` (~5x smaller than the
CSV). This script converts CSVs that already exist on disk from older pulls,
verifies the round-trip, and only then deletes the CSV. Readers keep CSV
fallback, so skipping or interrupting this migration is always safe.

Usage (from anywhere):
    python scripts/maintenance/ConvertRoiCountToParquet.py            # all datasets
    python scripts/maintenance/ConvertRoiCountToParquet.py male-cns_v1_0   # one dataset folder
"""

import sys
from pathlib import Path

import pandas as pd
import polars as pl

DATASETS_DIR = Path(__file__).resolve().parents[2] / 'datasets'


def convert(csv_path: Path) -> bool:
    parquet_path = csv_path.with_suffix('.parquet')
    if parquet_path.exists():
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
    csv_path.unlink()
    pq_size = parquet_path.stat().st_size
    print(f'[ok]   {csv_path.parent.name}: {csv_size / 2**20:8.1f} MiB -> '
          f'{pq_size / 2**20:6.1f} MiB ({csv_size / pq_size:.1f}x smaller)')
    return True


def main() -> int:
    folders = sys.argv[1:]
    csvs = sorted(DATASETS_DIR.glob('*/*_allneurons_roi_count_df.csv'))
    if folders:
        csvs = [c for c in csvs if c.parent.name in folders]
    if not csvs:
        scope = f" in {', '.join(folders)}" if folders else ""
        print(f'No ROI-count CSVs{scope} found under {DATASETS_DIR}')
        return 0

    print(f'Converting {len(csvs)} ROI-count CSV(s) to zstd parquet...')
    failed = [c for c in csvs if not convert(c)]
    if failed:
        print(f'{len(failed)} conversion(s) failed; their CSVs were kept.')
        return 1
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
