"""One-time repair of a poisoned neuron index (plan R3-b, 2026-09-15).

A failed/transient connection fetch could stamp neurons
``downstream_complete=True, connection_count=0`` — the "verified
zero-outdegree" marker — even though the server reports real partners.
``_query_connection_db`` now revalidates suspect zero markers before
trusting them, but the stale markers remain in the stored index.

This script recomputes the EFFECTIVE index (base
``neuron_indexes/<safe>/neuron_index.parquet`` merged with the progress
sidecar ``cache/<safe>/neuron_index_state.parquet``) and flips every
``downstream_complete=True & connection_count==0`` row to incomplete.
Nothing else changes: rows in the connection cache still prove
completeness, and genuinely isolated neurons are re-verified lazily by
the runtime revalidation (one batched count query) and re-marked.

Usage:
    python scripts/maintenance/repair_neuron_index.py --dataset hemibrain:v1.2.1           # dry run
    python scripts/maintenance/repair_neuron_index.py --dataset hemibrain:v1.2.1 --apply   # write
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402
from utils.parquet_utils import atomic_replace  # noqa: E402

NEURON_INDEXES = PROJECT / 'neuron_indexes'
CACHE = PROJECT / 'cache'


def _safe_name(dataset: str) -> str:
    from utils.naming_utils import canonical_dataset_name
    return canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')


def load_effective_index(base_path: str, sidecar_path: str):
    """Effective index = base rows overridden by sidecar rows (bodyId key).

    Mirrors the merge semantics of
    ``ComparisonAnalyzer._load_neuron_index``: the sidecar carries the
    frequently-updated progress (downstream_complete / connection_count /
    last_fetched) and wins on conflicts. Sidecar-only neurons are appended
    so nothing is hidden.
    """
    base = pd.read_parquet(base_path)
    effective = base.copy()
    if not os.path.exists(sidecar_path):
        return effective
    side = pd.read_parquet(sidecar_path)
    side = side.copy()
    side['_key'] = side['bodyId'].astype(str)
    effective['_key'] = effective['bodyId'].astype(str)
    # sidecar rows override base rows
    merged = effective.merge(
        side.drop(columns=['bodyId']), on='_key', how='outer',
        suffixes=('', '_side'), indicator=True)
    key_series = merged['_key']
    for col in ('downstream_complete', 'connection_count', 'last_fetched'):
        if f'{col}_side' in merged.columns:
            merged[col] = merged[f'{col}_side'].where(
                merged[f'{col}_side'].notna(), merged[col])
            merged = merged.drop(columns=[f'{col}_side'])
    merged = merged.drop(columns=['_key'])
    # sidecar-only rows have no base bodyId — restore it from the key and
    # keep the column integer-typed like the base index (float NaN fill
    # would corrupt the big bodyIds when stringified)
    if 'bodyId' in merged.columns:
        merged['bodyId'] = merged['bodyId'].where(
            merged['bodyId'].notna(), key_series)
        try:
            as_num = pd.to_numeric(merged['bodyId'])
            if as_num.notna().all():
                merged['bodyId'] = as_num.astype('int64')
        except (ValueError, TypeError):
            pass
    return merged


def repair_effective_index(effective: pd.DataFrame):
    """Flip True+0 completion markers to incomplete. Returns (frame, flipped)."""
    flipped = effective.copy()
    mask = ((flipped['downstream_complete'].astype(bool))
            & (pd.to_numeric(flipped['connection_count'],
                             errors='coerce').fillna(0) == 0))
    flipped.loc[mask, 'downstream_complete'] = False
    return flipped, int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', default='hemibrain:v1.2.1',
                    help='Dataset name, e.g. hemibrain:v1.2.1')
    ap.add_argument('--apply', action='store_true',
                    help='Write the repair (default: dry run)')
    args = ap.parse_args()

    safe = _safe_name(args.dataset)
    base_path = str(NEURON_INDEXES / safe / 'neuron_index.parquet')
    sidecar_path = str(CACHE / safe / 'neuron_index_state.parquet')
    if not os.path.exists(base_path):
        print(f'No neuron index at {base_path}')
        sys.exit(1)

    effective = load_effective_index(base_path, sidecar_path)
    before = int(((effective['downstream_complete'].astype(bool))
                  & (pd.to_numeric(effective['connection_count'],
                                   errors='coerce').fillna(0) == 0)).sum())
    flipped, flipped_n = repair_effective_index(effective)
    print(f'{args.dataset}: effective rows={len(effective):,}, '
          f'stale True+0 markers={before:,} -> {flipped_n:,} flipped')

    if not args.apply:
        print('DRY RUN — re-run with --apply to write.')
        return

    # Write the repaired EFFECTIVE state: full base rewrite + sidecar reset
    # to the repaired rows (the sidecar remains the live progress store).
    tmp = base_path + '.repair_tmp'
    flipped.to_parquet(tmp, index=False)
    atomic_replace(tmp, base_path)
    if os.path.exists(sidecar_path):
        keep = flipped[flipped['bodyId'].astype(str).isin(
            pd.read_parquet(sidecar_path)['bodyId'].astype(str))][
            ['bodyId', 'downstream_complete', 'last_fetched',
             'connection_count']]
        tmp2 = sidecar_path + '.repair_tmp'
        keep.to_parquet(tmp2, index=False)
        atomic_replace(tmp2, sidecar_path)
    print(f'Wrote repaired index to {base_path}'
          + (f' and reset sidecar {sidecar_path}' if os.path.exists(sidecar_path)
             else ''))
    print('Runtime note: genuinely isolated neurons are re-verified once by '
          'the zero-marker revalidation and re-marked automatically.')


if __name__ == '__main__':
    main()
