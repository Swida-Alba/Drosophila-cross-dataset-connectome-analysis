#!/usr/bin/env python3
"""One-time metadata enrichment: median synapse density per dataset.

Threshold-alignment spec, Feature A. For each ``datasets/<safe_name>/``
this script rewrites ``<safe_name>_metadata.json`` preserving every
existing key (incl. ``fetched_at``) and adding:

- ``synapse_density.per_neuron_median`` — median over ALL neurons of
  (pre + post) — the whole-dataset connection-density denominator used by
  the static threshold-equivalence note;
- ``per_neuron_mean`` / ``per_neuron_median_post`` / ``per_neuron_median_pre``;
- ``neuron_universe: "all_neurons"`` and ``pre_source``
  ("neuron_table" for NeuPrint-hosted datasets, "derived_from_connections"
  for local FlyWire/BANC downloads whose neuron tables carry post only —
  pre is summed from merged_connections.parquet).

Fresh fetches compute the same fields via ComparisonAnalyzer
(``_fetch_local_metadata`` / ``_fetch_neuprint_metadata``), so this script
only needs to run once per checkout (or after adding a dataset).

Usage:
    python scripts/generate_dataset_metadata.py            # all datasets
    python scripts/generate_dataset_metadata.py male-cns   # name filter
"""

import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, 'src'),
):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from comparison.metadata_density import (
        compute_synapse_density,
        load_neuron_table,
    )
except ImportError:  # pragma: no cover - direct src imports
    from src.comparison.metadata_density import (
        compute_synapse_density,
        load_neuron_table,
    )


def enrich_dataset(datasets_root: str, safe_name: str, verbose: bool = True) -> bool:
    """Enrich one dataset's metadata JSON in place. Returns True on success."""
    import json

    dataset_path = os.path.join(datasets_root, safe_name)
    metadata_path = os.path.join(dataset_path, f'{safe_name}_metadata.json')

    if not os.path.exists(metadata_path):
        if verbose:
            print(f'  ⚠️  {safe_name}: no metadata JSON, skipping')
        return False

    neuron_df = load_neuron_table(dataset_path, safe_name)
    if neuron_df is None:
        if verbose:
            print(f'  ⚠️  {safe_name}: no neuron table found, skipping')
        return False

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    density = compute_synapse_density(neuron_df, dataset_path=dataset_path)
    if not density:
        if verbose:
            print(f'  ⚠️  {safe_name}: neuron table has no synapse columns, skipping')
        return False

    metadata['synapse_density'] = density
    metadata['density_generated_at'] = datetime.now().isoformat()

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    if verbose:
        print(
            f'  ✅ {safe_name}: median (pre+post)/neuron = '
            f"{density['per_neuron_median']:.1f} "
            f"(pre source: {density['pre_source']})")
    return True


def main(argv):
    name_filter = argv[1].lower() if len(argv) > 1 else None
    datasets_root = os.path.join(PROJECT_ROOT, 'datasets')

    if not os.path.isdir(datasets_root):
        print(f'Datasets folder not found: {datasets_root}')
        return 1

    safe_names = sorted(
        d for d in os.listdir(datasets_root)
        if os.path.isdir(os.path.join(datasets_root, d))
        and not d.startswith('.')
    )
    if name_filter:
        safe_names = [d for d in safe_names if name_filter in d.lower()]

    print(f'=== Dataset metadata enrichment (synapse density) ===')
    print(f'Folder: {datasets_root}')
    ok = 0
    for safe_name in safe_names:
        if enrich_dataset(datasets_root, safe_name):
            ok += 1
    print(f'--- enriched {ok}/{len(safe_names)} datasets ---')
    return 0


if __name__ == '__main__':
    pd.set_option('display.width', 160)
    sys.exit(main(sys.argv))
