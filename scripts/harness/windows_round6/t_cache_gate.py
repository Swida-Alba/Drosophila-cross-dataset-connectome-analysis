#!/usr/bin/env python
"""Round-6 §B harness: the cache-integrity gate and offline certification.

Why this exists on this host: the 2026-09-25 notes assume a round-5 machine with
a seeded 10.2 M-connection ``cache/male-cns_v1_0``. That cache is off-limits
here (only the manual FAFB download may be copied), and the gate itself is
NeuPrint-only -- ``_enforce_cache_coverage`` is unreachable for FAFB/BANC, so
the gate cannot be exercised on the one dataset we are allowed to use.

This harness therefore builds a scratch DROCAT root whose connection cache is
*self-consistent with its own neuron index* and drives the product's real
surfaces against it: ``FindNeuronConnection(cache_only=True)`` construction, a
normal run against an unreachable NeuPrint host, and the
``record_cache_baseline`` CLI. Every refusal is the product's own text.

    python t_cache_gate.py setup   [--mode pristine]
    python t_cache_gate.py refuse-cacheonly
    python t_cache_gate.py refuse-normal-offline
    python t_cache_gate.py certify [--force]
    python t_cache_gate.py truncate [--rows N]
    python t_cache_gate.py restore
    python t_cache_gate.py append  [--rows N]
    python t_cache_gate.py manifest --set-dataset <id> | --set-schema <n>
    python t_cache_gate.py status

Modes for setup:
    pristine         cache matches the index, no manifest          (B1, B3a)
    truncated        cache holds fewer rows than the index records (B2, B3b/c)
    partial-clear    connections.parquet deleted, batches kept     (B3g)
    missing-dataset  neuron table absent                            (B3e)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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

SCRATCH = HERE.parent / 'gate_scratch'
DATASET = 'probe-gate:v1.0'
NEURON_COUNT = 40
ROWS_PER_NEURON = 25
UNREACHABLE_SERVER = 'http://127.0.0.1:59999'

for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def dataset_safe() -> str:
    import coana
    return coana.dataset_folder(DATASET)


def paths():
    safe = dataset_safe()
    return {
        'root': SCRATCH,
        'cache': SCRATCH / 'cache' / safe,
        'batches': SCRATCH / 'cache' / safe / '_batch_files',
        'conn': SCRATCH / 'cache' / safe / 'connections.parquet',
        'manifest': SCRATCH / 'cache' / safe / 'cache_manifest.json',
        'index': SCRATCH / 'neuron_indexes' / safe / 'neuron_index.parquet',
        'datasets': SCRATCH / 'datasets' / safe,
        'neuron_df': SCRATCH / 'datasets' / safe / f'{safe}_allneurons_neuron_df.parquet',
        'safe': safe,
    }


def write_tables(p, mode):
    """Write a neuron index whose recorded counts the cache can satisfy."""
    import polars as pl

    body_ids = [1000 + i for i in range(NEURON_COUNT)]
    keep = NEURON_COUNT if mode != 'truncated' else NEURON_COUNT
    rows = []
    for i, bid in enumerate(body_ids):
        # In truncated mode the cache will only hold the first 6 neurons.
        recorded = ROWS_PER_NEURON
        rows.append((bid, True, recorded))
    p['index'].parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        'bodyId': [r[0] for r in rows],
        'type': [f'ProbeType{i:02d}' for i in range(len(rows))],
        'downstream_complete': [r[1] for r in rows],
        'connection_count': [r[2] for r in rows],
    }).write_parquet(p['index'])

    p['datasets'].mkdir(parents=True, exist_ok=True)
    if mode != 'missing-dataset':
        pl.DataFrame({
            'bodyId': body_ids,
            'type': [f'ProbeType{i:02d}' for i in range(len(body_ids))],
            'instance': [f'Probe{i:02d}' for i in range(len(body_ids))],
        }).write_parquet(p['neuron_df'])
    elif p['neuron_df'].exists():
        p['neuron_df'].unlink()

    connection_rows = []
    cached_neurons = 6 if mode == 'truncated' else NEURON_COUNT
    for i, bid in enumerate(body_ids[:cached_neurons]):
        for k in range(ROWS_PER_NEURON):
            connection_rows.append((bid, 5000 + i * 100 + k, k + 1, f'ROI_{k % 3}'))
    p['cache'].mkdir(parents=True, exist_ok=True)
    p['batches'].mkdir(parents=True, exist_ok=True)
    if mode == 'partial-clear':
        if p['conn'].exists():
            p['conn'].unlink()
        half = len(connection_rows) // 2
        pl.DataFrame({
            'bodyId_pre': [r[0] for r in connection_rows[:half]],
            'bodyId_post': [r[1] for r in connection_rows[:half]],
            'weight': [r[2] for r in connection_rows[:half]],
            'roi': [r[3] for r in connection_rows[:half]],
        }).write_parquet(p['batches'] / 'batch_probe_0001.parquet')
        return
    if p['batches'].is_dir():
        shutil.rmtree(p['batches'])
        p['batches'].mkdir(parents=True)
    pl.DataFrame({
        'bodyId_pre': [r[0] for r in connection_rows],
        'bodyId_post': [r[1] for r in connection_rows],
        'weight': [r[2] for r in connection_rows],
        'roi': [r[3] for r in connection_rows],
    }).write_parquet(p['conn'])


def cmd_setup(args):
    p = paths()
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / 'local_data').mkdir(parents=True)
    (SCRATCH / 'config.json').write_text(
        json.dumps({'tokens': {'neuprint': '', 'cave': ''}}), encoding='utf-8')
    write_tables(p, args.mode)
    print(f'setup mode={args.mode}')
    print(f'  scratch root : {p["root"]}')
    print(f'  dataset      : {DATASET} -> {p["safe"]}')
    print(f'  connections  : {p["conn"].exists()}')
    print(f'  neuron index : {p["index"]}')
    print(f'  neuron table : {p["neuron_df"].exists()}')
    print(f'  manifest     : {p["manifest"].exists()}')
    return 0


def _construct(cache_only, expect_paths=False):
    """Build a FindNeuronConnection against the scratch root."""
    p = paths()
    from coana import FindNeuronConnection
    fc = FindNeuronConnection(
        dataset=DATASET,
        script_path=str(SCRATCH),
        sourceNeurons=['ProbeType00', 'ProbeType01'],
        targetNeurons=['ProbeType02'],
        output_dir=str(SCRATCH / 'local_data'),
        min_synapse_num=1,
        max_interlayer=1,
        use_cache=True,
        cache_only=cache_only,
        server=UNREACHABLE_SERVER,
        showfig=False,
        verbose_mode='simple',
    )
    if expect_paths:
        fc.InitializeNeuronInfo()
        fc.FindAllPath(forward_only=True)
    return fc


def _report(action, exc=None):
    if exc is None:
        print(f'{action}: NO REFUSAL (construction/run succeeded)')
        return 0
    text = str(exc)
    print(f'{action}: refused with {type(exc).__name__}')
    print('--- begin verbatim message ---')
    print(text)
    print('--- end message ---')
    for probe in ('cannot be verified', 'no integrity manifest',
                  'record_cache_baseline', 'Certify local cache',
                  'rows were lost', 'allow_incomplete_cache',
                  'Run once with cache_only=False'):
        print(f'  contains {probe!r}: {probe in text}')
    return 1


def cmd_refuse_cacheonly(_args):
    import coana
    coana._CACHE_COVERAGE_CACHE.clear()
    try:
        _construct(cache_only=True)
    except Exception as exc:  # noqa: BLE001 - the refusal is the result
        return _report('B3 cache_only=True construction', exc)
    return _report('B3 cache_only=True construction')


def cmd_refuse_normal_offline(_args):
    import coana
    coana._CACHE_COVERAGE_CACHE.clear()
    try:
        _construct(cache_only=False)
    except Exception as exc:  # noqa: BLE001
        return _report('B1 normal run, server unreachable', exc)
    return _report('B1 normal run, server unreachable')


def cmd_certify(args):
    """Run the product's own certification CLI against the scratch root."""
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    script = PROJECT / 'scripts' / 'maintenance' / 'record_cache_baseline.py'
    command = [sys.executable, str(script), '--dataset', DATASET,
               '--project-root', str(SCRATCH)]
    if args.force:
        command.append('--force')
    print('$ ' + ' '.join(command))
    proc = subprocess.run(command, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', cwd=str(PROJECT))
    print(f'exit code: {proc.returncode}')
    print('--- stdout ---')
    print(proc.stdout)
    if proc.stderr:
        print('--- stderr ---')
        print(proc.stderr)
    if p_manifest_exists():
        print('--- manifest on disk ---')
        print(Path(paths()['manifest']).read_text(encoding='utf-8'))
    else:
        print('--- no manifest written ---')
    for probe in ('Refusing to certify', 'recorded / cached'):
        print(f'  combined output contains {probe!r}: '
              f'{probe in proc.stdout or probe in proc.stderr}')
    return 0


def p_manifest_exists():
    return paths()['manifest'].exists()


def cmd_truncate(args):
    import polars as pl
    p = paths()
    backup = p['conn'].with_suffix('.parquet.pretruncate')
    if p['conn'].exists() and not backup.exists():
        shutil.copy2(p['conn'], backup)
        df = pl.read_parquet(p['conn'])
        kept = max(1, int(args.rows))
        df.head(kept).write_parquet(p['conn'])
        print(f'truncated {p["conn"]} to {kept} rows (backup {backup.name})')
    else:
        print(f'nothing to truncate (conn={p["conn"].exists()}, '
              f'backup={backup.exists()})')
    return 0


def cmd_restore(_args):
    p = paths()
    backup = p['conn'].with_suffix('.parquet.pretruncate')
    if backup.exists():
        shutil.copy2(backup, p['conn'])
        print(f'restored {p["conn"]} from {backup.name}')
    else:
        print('no pre-truncation backup present')
    return 0


def cmd_append(args):
    """Legal growth: append distinct rows beyond what the index records."""
    import polars as pl
    p = paths()
    df = pl.read_parquet(p['conn'])
    start = df.height
    extra = pl.DataFrame({
        'bodyId_pre': [1000 + (i % NEURON_COUNT) for i in range(args.rows)],
        'bodyId_post': [9000 + i for i in range(args.rows)],
        'weight': [1] * args.rows,
        'roi': [f'ROI_NEW_{i % 3}' for i in range(args.rows)],
    })
    pl.concat([df, extra]).write_parquet(p['conn'])
    print(f'appended {args.rows} distinct rows: {start} -> {start + args.rows}')
    return 0


def cmd_manifest(args):
    """Tamper with the manifest: foreign dataset id, bad schema, or hold a copy."""
    p = Path(paths()['manifest'])
    if not p.exists():
        print('no manifest present')
        return 1
    data = json.loads(p.read_text(encoding='utf-8'))
    if args.set_dataset:
        data['dataset'] = args.set_dataset
    if args.set_schema is not None:
        data['schema'] = args.set_schema
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('manifest now:')
    print(p.read_text(encoding='utf-8'))
    return 0


def cmd_status(_args):
    import coana
    coana._CACHE_COVERAGE_CACHE.clear()
    p = paths()
    fc = object.__new__(coana.FindNeuronConnection)
    fc.script_path = str(SCRATCH)
    fc.dataset = DATASET
    fc._dataset_safe = p['safe']
    fc.use_cache = True
    fc.cache_only = True
    fc.cache_folder = str(p['cache'])
    fc.verbose_mode = 'silent'
    coverage = fc._check_cache_coverage()
    print(f'cache files      : {fc._connection_cache_files()}')
    print(f'manifest exists  : {p["manifest"].exists()}')
    if coverage is None:
        print('coverage         : None (nothing to check)')
        return 0
    print(f'distinct         : {coverage["distinct_connections"]:,}')
    print(f'complete_flagged : {coverage["complete_flagged"]:,}')
    print(f'mismatches       : {len(coverage["mismatches"])}')
    if coverage['mismatches']:
        print(f'  sample         : {coverage["mismatches"][:3]}')
    if p['manifest'].exists():
        print(Path(p['manifest']).read_text(encoding='utf-8'))
    import time
    t0 = time.perf_counter()
    coana._CACHE_COVERAGE_CACHE.clear()
    fc._check_cache_coverage()
    t1 = time.perf_counter()
    fc._check_cache_coverage()
    t2 = time.perf_counter()
    print(f'B3j cold scan      : {t1 - t0:.3f} s')
    print(f'B3j memoized recheck: {t2 - t1:.4f} s')
    return 0


def cmd_b3matrix(_args):
    """Run the whole B3 gate matrix against the scratch root, in order.

    Each leg reports the verbatim first line of the refusal and whether the
    run left an output folder behind (a refused run must not, finding F1).
    """
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import coana
    from coana import FindNeuronConnection

    p = paths()
    results = []

    def leg(label, setup_mode, action, rows=None):
        coana._CACHE_COVERAGE_CACHE.clear()
        if setup_mode is not None:
            cmd_setup(argparse.Namespace(mode=setup_mode))
            if setup_mode in ('pristine', 'truncated'):
                # certification is the offline remedy; re-run it so every leg
                # below starts from a known manifest state
                if setup_mode == 'pristine':
                    subprocess.run([sys.executable, str(
                        PROJECT / 'scripts' / 'maintenance' /
                        'record_cache_baseline.py'), '--dataset', DATASET,
                        '--project-root', str(SCRATCH), '--force'],
                        capture_output=True, text=True, cwd=str(PROJECT))
        if rows is not None:
            cmd_truncate(argparse.Namespace(rows=rows))
        before = sorted(os.listdir(p['root'] / 'local_data')) \
            if (p['root'] / 'local_data').is_dir() else []
        coana._CACHE_COVERAGE_CACHE.clear()
        outcome, message = 'COMPLETED', ''
        try:
            _construct(cache_only=True)
        except TypeError as exc:
            outcome, message = 'TypeError', str(exc)
        except Exception as exc:  # noqa: BLE001
            outcome, message = 'REFUSED', f'{type(exc).__name__}: {exc}'
        after = sorted(os.listdir(p['root'] / 'local_data')) \
            if (p['root'] / 'local_data').is_dir() else []
        created = [d for d in after if d not in before]
        first = message.splitlines()
        headline = first[0] if first else ''
        print(f'\n--- {label}')
        print(f'    outcome      : {outcome}')
        print(f'    first line   : {headline}')
        print(f'    folders left : {created if created else "none"}')
        for probe in ('cannot be verified', 'no integrity manifest',
                      'rows were lost', 'Restore', 'cache is insufficient',
                      'dataset_csv=False', 'allow_incomplete_cache'):
            if probe in message:
                print(f'    message has  : {probe!r}')
        results.append((label, outcome, bool(created), headline))
        return message

    leg('B3b truncated cache, cache-only, max_interlayer=2', 'pristine',
        'cacheonly', rows=300)
    leg('B3c truncated cache, shallow query', None, 'cacheonly', rows=None)
    msg = leg('B3e dataset table missing', 'missing-dataset', 'cacheonly')
    leg('B3g partial clear (connections.parquet gone, batches kept)',
        'partial-clear', 'cacheonly')
    # B3f legal growth: rebuild pristine + certified, then append beyond the
    # recorded counts.
    leg('B3f pre-state (pristine + certified, cache-only)', 'pristine',
        'cacheonly')
    cmd_append(argparse.Namespace(rows=500))
    leg('B3f after appending 500 distinct rows (growth is legal)', None,
        'cacheonly')
    # B3i a manifest copied in from another dataset
    manifest = Path(p['manifest'])
    data = json.loads(manifest.read_text(encoding='utf-8'))
    data['dataset'] = 'flywire_FAFB_v783'
    manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    leg('B3i manifest from a different dataset', None, 'cacheonly')
    # B3d the deleted opt-in
    coana._CACHE_COVERAGE_CACHE.clear()
    print('\n--- B3d allow_incomplete_cache=True (parameter was deleted)')
    try:
        FindNeuronConnection(**{k: v for k, v in {}.items()},
                             dataset=DATASET, script_path=str(SCRATCH),
                             sourceNeurons=['ProbeType00'],
                             targetNeurons=['ProbeType02'],
                             output_dir=str(SCRATCH / 'local_data' / 'b3d'),
                             allow_incomplete_cache=True)
        print('    outcome      : CONSTRUCTED (parameter still exists!)')
    except TypeError as exc:
        print(f'    outcome      : TypeError (AS DESIGNED)')
        print(f'    message      : {exc}')
    except Exception as exc:  # noqa: BLE001
        print(f'    outcome      : {type(exc).__name__}: {exc}')

    print('\n===== B3 matrix summary =====')
    for label, outcome, left_folder, headline in results:
        print(f'{label}\n    {outcome}  folder_left={left_folder}')
    print(f'\nB3e dataset_csv mention present: '
          f'{"dataset_csv=False" in (msg or "")}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=[
        'setup', 'refuse-cacheonly', 'refuse-normal-offline', 'certify',
        'truncate', 'restore', 'append', 'manifest', 'status', 'b3matrix'])
    parser.add_argument('--mode', default='pristine',
                        choices=['pristine', 'truncated', 'partial-clear',
                                 'missing-dataset'])
    parser.add_argument('--rows', type=int, default=25)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--set-dataset', dest='set_dataset')
    parser.add_argument('--set-schema', dest='set_schema', type=int)
    args = parser.parse_args(argv)
    return {
        'setup': cmd_setup,
        'refuse-cacheonly': cmd_refuse_cacheonly,
        'refuse-normal-offline': cmd_refuse_normal_offline,
        'certify': cmd_certify,
        'truncate': cmd_truncate,
        'restore': cmd_restore,
        'append': cmd_append,
        'manifest': cmd_manifest,
        'status': cmd_status,
        'b3matrix': cmd_b3matrix,
    }[args.action](args)


if __name__ == '__main__':
    sys.exit(main())
