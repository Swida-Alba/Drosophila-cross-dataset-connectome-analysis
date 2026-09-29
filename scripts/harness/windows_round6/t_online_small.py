#!/usr/bin/env python
"""§O2/§O3/§F1: one small ONLINE male-cns query against real NeuPrint.

Deliberately tiny (direct connections only, two named types) so it fetches a
handful of neurons instead of rebuilding a brain-wide cache. It is the cheapest
way to prove three things at once:

  O2  live neuron resolution with a real token, no cache refusal, no SystemExit
  O3  an online run writes cache_manifest.json itself with source="baseline"
      and, once a user_certified manifest exists, must not silently overwrite it
  F1  FindNeuron/getNeurons resolve a type query without terminating the process
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Resolve the repo root from THIS file's location: committed at
# scripts/harness/windows_round6/ the root is 3 up; when staged
# beside a checkout (round-6/7 hosts) fall back to the sibling
# 'drocat' folder that host layout used.
PROJECT = HERE
for _ in range(5):
    if (PROJECT / 'src' / 'coana.py').exists():
        break
    PROJECT = PROJECT.parent
else:
    _sibling = HERE.parent / 'drocat'
    if (_sibling / 'src' / 'coana.py').exists():
        PROJECT = _sibling

for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DATASET = 'male-cns:v1.0'
SAFE = 'male-cns_v1_0'
MANIFEST = PROJECT / 'cache' / SAFE / 'cache_manifest.json'


def show_manifest(tag):
    if MANIFEST.exists():
        print(f'  {tag}: {json.loads(MANIFEST.read_text(encoding="utf-8"))}')
    else:
        print(f'  {tag}: no manifest')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.chdir(PROJECT)
    import coana
    from coana import FindNeuronConnection

    print('=== pre-state ===')
    show_manifest('before')
    print(f'  cache dir present: {(PROJECT / "cache" / SAFE).is_dir()}')

    print('\n=== F9-style SystemExit probe: a run must never raise SystemExit ===')
    started = time.perf_counter()
    try:
        fc = FindNeuronConnection(
            dataset=DATASET,
            sourceNeurons=['aMe12'],
            targetNeurons=['PPL101'],
            output_dir=str(PROJECT / 'local_data' / 'r6_online_small'),
            min_synapse_num=10,
            max_interlayer=0,          # direct connections only: a tiny fetch
            filter_by='type',
            use_cache=True,
            cache_only=False,
            pathfinding='StrongestFirst',
            drop_untyped=True,
            skip_bodyId=True,
            showfig=False,
            verbose_mode='simple',
        )
        fc.InitializeNeuronInfo()
        fc.FindAllPath(forward_only=True)
        elapsed = time.perf_counter() - started
        print(f'  run completed in {elapsed:.1f} s')
    except SystemExit as exc:
        print(f'  BUG: SystemExit({exc.code}) escaped instead of an exception')
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f'  run raised {type(exc).__name__}: {str(exc)[:400]}')

    print('\n=== O3: did the online run write the baseline manifest? ===')
    show_manifest('after')
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text(encoding='utf-8'))
        print(f'  source == "baseline"     : {data.get("source") == "baseline"}')
        print(f'  dataset                  : {data.get("dataset")}')
        print(f'  distinct_connections     : {data.get("distinct_connections"):,}')

    print('\n=== O3 leg 2: a user_certified manifest must not be overwritten ===')
    if MANIFEST.exists():
        original = json.loads(MANIFEST.read_text(encoding='utf-8'))
        certified = dict(original, source='user_certified',
                         distinct_connections=original['distinct_connections'])
        MANIFEST.write_text(json.dumps(certified, indent=2), encoding='utf-8')
        print(f'  planted source=user_certified, distinct='
              f'{certified["distinct_connections"]:,}')
        coana._CACHE_COVERAGE_CACHE.clear()
        fc2 = FindNeuronConnection(
            dataset=DATASET, sourceNeurons=['aMe12'], targetNeurons=['PPL101'],
            output_dir=str(PROJECT / 'local_data' / 'r6_online_second'),
            min_synapse_num=10, max_interlayer=0, filter_by='type',
            use_cache=True, cache_only=False, showfig=False,
            verbose_mode='silent')
        fc2.InitializeNeuronInfo()
        fc2.FindAllPath(forward_only=True)
        after = json.loads(MANIFEST.read_text(encoding='utf-8'))
        print(f'  manifest after a 2nd online run: source={after["source"]}, '
              f'distinct={after["distinct_connections"]:,}')
        print(f'  [write-once held: {after.get("source") == "user_certified"}]')

    print('\n=== cache folder produced ===')
    cache_dir = PROJECT / 'cache' / SAFE
    if cache_dir.is_dir():
        for p in sorted(cache_dir.iterdir()):
            print(f'  {p.name}: '
                  f'{"dir" if p.is_dir() else f"{p.stat().st_size:,} bytes"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
