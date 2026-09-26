#!/usr/bin/env python
"""§C2/§C3/§O13: prepare the BANC v888 tables from its public bucket.

Runs the product's own ``banc_public_data.prepare_dataset_tables`` exactly the
way the first real BANC query does, and reports the three things the round
cares about: it completes, the derived synapse table carries the lossless
footer marker, and how much size the re-encode reclaimed.

Pass --guard to import coana first (which installs ensure_utf8_stdio); without
it this reproduces the module's real standalone import path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / 'drocat'
for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', action='store_true',
                        help='import coana first (installs the UTF-8 stdio guard)')
    parser.add_argument('--dataset', default='banc_v888')
    args = parser.parse_args(argv)

    print(f'stdout encoding at start: {sys.stdout.encoding}')
    print(f'PYTHONIOENCODING        : {os.environ.get("PYTHONIOENCODING", "unset")}')
    print(f'coana imported first    : {args.guard}')
    if args.guard:
        import coana  # noqa: F401
        print(f'stdout encoding after coana import: {sys.stdout.encoding}')
    print()

    import banc_public_data
    dataset_dir = PROJECT / 'datasets' / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        ok = banc_public_data.prepare_dataset_tables(args.dataset, str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        print(f'PREPARE RAISED {type(exc).__name__}: {exc}')
        import traceback
        traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - started
    print(f'prepare ok={ok} in {elapsed:.1f} s')

    import polars as pl
    import pyarrow.parquet as pq
    safe = banc_public_data._dataset_folder(args.dataset)
    print(f'canonical dataset folder: {safe}')
    for name in (f'{safe}_allneurons_neuron_df.parquet',
                 f'{safe}_merged_connections.parquet',
                 f'{safe}_synapse_table.parquet'):
        path = dataset_dir / name
        if not path.exists():
            print(f'  MISSING {name}')
            continue
        size = path.stat().st_size / 1e6
        if name.endswith('synapse_table.parquet'):
            meta = pq.read_schema(str(path)).metadata or {}
            marker = meta.get(b'DROCAT.lossless')
            print(f'  {name}: {size:,.1f} MB  '
                  f'rows={pl.read_parquet(path, n_rows=1).height and pq.ParquetFile(str(path)).metadata.num_rows:,}  '
                  f'DROCAT.lossless={marker!r}')
        else:
            frame = pl.read_parquet(path)
            print(f'  {name}: {size:,.1f} MB  rows={frame.height:,}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
