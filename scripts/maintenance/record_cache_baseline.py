#!/usr/bin/env python
"""Certify a local connection cache offline (2026-09-18 retest, finding F5).

Why this exists: a cache restored from a backup — or copied from a
colleague's machine — has no ``cache/<dataset>/cache_manifest.json``, and
every *automatic* writer of one needs a live NeuPrint connection.  Cache-only
runs therefore refuse a complete cache with nothing offline to fix it with.
``FindNeuronConnection.record_cache_baseline()`` is the product's single
remedy, and this script is its command-line surface (the other surface is
Settings -> Storage -> "Certify local cache").

The script never weakens the check: the per-neuron integrity check has to
pass first, so a truncated cache still cannot certify itself (finding F2),
and an existing manifest is never overwritten silently — re-recording it is
an explicit ``--force``.

Usage:
    python scripts/maintenance/record_cache_baseline.py --list
    python scripts/maintenance/record_cache_baseline.py --dataset <id>
    python scripts/maintenance/record_cache_baseline.py --dataset <id> --force

(``<id>`` is the identifier your runs use, e.g. hemibrain:v1.2.1 — the
same one the manifest records.)

Exit codes: 0 certified (or listed), 1 the product refused, 2 bad usage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def build_offline_connection(dataset: str, script_path):
    """Return a ``FindNeuronConnection`` that provably cannot connect.

    Every connected construction route is a dead end for this exact job:
    ``cache_only=True`` runs ``_enforce_cache_coverage()`` inside
    ``__post_init__`` — that is the refusal this tool exists to cure, so it
    raises before certification is ever reached — and any route that lets the
    client block run either builds a NeuPrint client (needs a token and a
    network) or rebuilds the neuron index as an init side effect.  The
    certification path (``record_cache_baseline`` ->
    ``_check_cache_coverage`` -> ``_compute_cache_integrity`` /
    ``_load_cache_manifest`` / ``_write_cache_manifest``) reads only the
    attributes set below, so the instance is assembled directly — the same
    shape ``tests/core/test_cache_coverage.py`` uses.
    """
    import coana

    dataset_safe = coana.dataset_folder(dataset)
    connection = object.__new__(coana.FindNeuronConnection)
    connection.script_path = str(script_path)
    connection.dataset = dataset
    connection._dataset_safe = dataset_safe
    connection.use_cache = True
    connection.cache_only = True
    connection.cache_folder = os.path.join(str(script_path), 'cache',
                                           dataset_safe)
    # 'silent' keeps the product's own level='always' certification line and
    # drops the verbose scan chatter a maintenance run does not need.
    connection.verbose_mode = 'silent'
    return connection


def read_manifest_state(dataset: str,
                        script_path) -> Tuple[Optional[dict], str]:
    """(current manifest or None, manifest path) via the product's reader."""
    connection = build_offline_connection(dataset, script_path)
    return connection._load_cache_manifest(), connection._cache_manifest_path()


def _print_state(dataset: str, manifest: Optional[dict], path: str) -> None:
    print(f'Dataset: {dataset}')
    print(f'Manifest: {path}')
    if manifest is None:
        if os.path.exists(path):
            print('  current state: present but rejected (schema, dataset or '
                  'unreadable) — cache-only runs refuse it, and --force '
                  'replaces it.')
        else:
            print('  current state: none — cache-only runs cannot verify this '
                  'cache, so they refuse it.')
        return
    print('  current state: '
          f"source={manifest.get('source')}, "
          f"distinct_connections="
          f"{int(manifest.get('distinct_connections') or 0):,}, "
          f"built_at={manifest.get('built_at')}")


def certify(dataset: str, *, script_path=PROJECT,
            force: bool = False) -> Tuple[dict, str]:
    """Report the current manifest state, then certify this dataset's cache.

    Returns ``(manifest, manifest_path)``.  Raises ``RuntimeError`` carrying
    the product's own refusal text — nothing is re-worded here, and no
    manifest is written on a refusal.
    """
    connection = build_offline_connection(dataset, script_path)
    manifest, path = read_manifest_state(dataset, script_path)
    _print_state(dataset, manifest, path)
    written = connection.record_cache_baseline(force=force)
    return written, connection._cache_manifest_path()


def _has_connection_cache(dataset_dir: Path) -> bool:
    if (dataset_dir / 'connections.parquet').exists():
        return True
    batch_dir = dataset_dir / '_batch_files'
    try:
        return any(batch_dir.glob('batch_*.parquet'))
    except OSError:
        return False


def list_certifiable(script_path) -> int:
    """Report every local connection cache and whether it can be certified.

    Filesystem presence only: the per-neuron check that actually gates
    certification still runs when a dataset is certified.
    """
    from neuron_index_builder import dataset_identifier_from_folder

    script_path = Path(script_path)
    cache_root = script_path / 'cache'
    if not cache_root.is_dir():
        print(f'No cache folder at {cache_root}')
        return 0
    print(f'Scanning {cache_root} for connection caches...\n')
    certifiable = 0
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir() or not _has_connection_cache(entry):
            continue
        index_path = (script_path / 'neuron_indexes' / entry.name
                      / 'neuron_index.parquet')
        manifest_path = entry / 'cache_manifest.json'
        if manifest_path.exists():
            state = 'has a manifest — re-record it only with --force'
        elif not index_path.exists():
            state = ('has no neuron index — completeness cannot be checked, '
                     'nothing to certify')
        else:
            state = 'CERTIFIABLE (no manifest yet)'
            certifiable += 1
        print(f'  {dataset_identifier_from_folder(entry.name)}')
        print(f'      {state}')
        print(f'      cache: {entry}')
    print(f'\n{certifiable} connection cache(s) can be certified with '
          f'--dataset <id>.')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Certify a local connection cache offline by recording '
                    'its integrity manifest (cache_manifest.json). Reports '
                    'the current manifest state first and never overwrites '
                    'an existing manifest without --force.')
    parser.add_argument('--dataset',
                        help='Dataset identifier, e.g. hemibrain:v1.2.1')
    parser.add_argument('--project-root', default=str(PROJECT),
                        help='DROCAT project root holding cache/ and '
                             'neuron_indexes/ (default: this checkout)')
    parser.add_argument('--force', action='store_true',
                        help='Re-record the manifest when the cache already '
                             'has one (the new count is taken from the '
                             'current cache)')
    parser.add_argument('--list', action='store_true', dest='list_datasets',
                        help='List the local connection caches and whether '
                             'each one can be certified, then exit')
    args = parser.parse_args(argv)
    root = Path(args.project_root)

    if args.list_datasets:
        return list_certifiable(root)
    if not args.dataset:
        parser.error('--dataset is required (or run with --list to see what '
                     'can be certified)')

    try:
        manifest, path = certify(args.dataset, script_path=root,
                                 force=args.force)
    except RuntimeError as refusal:
        # The product owns this wording — it is the same text the cache-only
        # refusal and the UI show — so it is printed verbatim.
        print(f'\n{refusal}', file=sys.stderr)
        if not args.force:
            known, _state = read_manifest_state(args.dataset, root)
            if known is not None:
                print('Hint: this cache already has a manifest — re-record '
                      'it with --force.', file=sys.stderr)
        return 1

    print('  Certified.')
    print(f"  distinct_connections: "
          f"{int(manifest.get('distinct_connections') or 0):,}")
    print(f"  source: {manifest.get('source')}")
    print(f'  manifest: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
