#!/usr/bin/env python
"""Round-6 §C harness: the Windows file layer (F6 re-encode, temp hygiene).

C1  standalone lossless re-encode repro (last round it silently did nothing:
    both readers stayed open across os.replace -> WinError 32 -> False)
C4  temp-file hygiene sweep over cache/ and datasets/
C5  an orphaned temp from a dead writer is reclaimed immediately (psutil
    liveness, not the old 6-hour age rule)
C6  a concurrent preparation must not delete another process's in-flight temp
C7  the FAFB/BANC converter compaction actually lands (footer marker)

    python t_windows_file_layer.py c1
    python t_windows_file_layer.py c4
    python t_windows_file_layer.py c5
    python t_windows_file_layer.py c6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
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

import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# Importing `coana` installs utils.console_encoding.ensure_utf8_stdio(), which
# is what makes the product's own scripts able to print ✓ on a cp936 console.
# `utils.parquet_utils` does not install it, so without this line the re-encode
# status print aborts the run (round-6 finding, see r6_reencode_repro.log).
import coana  # noqa: E402,F401

from utils.parquet_utils import (  # noqa: E402
    parquet_lossless_marker, reencode_parquet_lossless,
    remove_stale_temp_files, temp_sibling, write_parquet_atomic)


def _table(path, rows=200_000):
    pl.DataFrame({
        'pre': pl.Series([i % 977 for i in range(rows)], dtype=pl.Int64),
        'post': pl.Series([i % 7919 for i in range(rows)], dtype=pl.Int64),
        'weight': pl.Series([(i % 50) + 1 for i in range(rows)], dtype=pl.Int64),
        'roi': pl.Series([f'ROI_{i % 11}' for i in range(rows)], dtype=pl.Utf8),
    }).write_parquet(path)


def cmd_c1(_args):
    tmp = Path(tempfile.mkdtemp(prefix='r6_c1_'))
    target = tmp / 'table.parquet'
    _table(target)
    before = target.stat().st_size
    marker_before = parquet_lossless_marker(str(target))
    checksum_before = (pl.read_parquet(target)['weight'].sum()
                       + pl.read_parquet(target)['pre'].sum())

    warnings = []
    import logging
    class _Capture(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())
    log = logging.getLogger()
    handler = _Capture()
    log.addHandler(handler)
    try:
        started = time.perf_counter()
        reencoded = reencode_parquet_lossless(
            str(target), {'compression': 'zstd', 'use_dictionary': False})
        elapsed = time.perf_counter() - started
    finally:
        log.removeHandler(handler)

    after = target.stat().st_size
    leftovers = sorted(p.name for p in tmp.iterdir() if p.name.endswith('.tmp'))
    print(f'reencode returned      : {reencoded}')
    print(f'size before/after      : {before:,} -> {after:,} '
          f'({(after - before) / before:+.1%})')
    print(f'marker before/after    : {marker_before} -> '
          f'{parquet_lossless_marker(str(target))}')
    print(f'elapsed                : {elapsed:.2f} s')
    print(f'temp leftovers         : {leftovers}')
    print(f'"skipped" warnings     : {[w for w in warnings if "skipp" in w.lower()]}')
    values_ok = (pl.read_parquet(target)['weight'].sum()
                 + pl.read_parquet(target)['pre'].sum()) == checksum_before
    print(f'reread ok              : {values_ok}')
    rc = 0 if reencoded is True and after != before and not leftovers else 1
    print(f'C1 verdict             : {"PASS" if rc == 0 else "FAIL"}')
    return rc


def cmd_c4(_args):
    patterns = ('*.compact.*.tmp', '*.build.*.tmp', '*.tmp')
    found = []
    for root in ('cache', 'datasets'):
        base = PROJECT / root
        if not base.is_dir():
            continue
        for pattern in patterns:
            found.extend(sorted(str(p) for p in base.rglob(pattern)))
    print(f'scanned {PROJECT / "cache"} and {PROJECT / "datasets"}')
    if found:
        print('TEMP LEFTOVERS:')
        for item in found:
            print('  ' + item)
    else:
        print('no temp leftovers')
    print(f'C4 verdict: {"FAIL" if found else "PASS"}')
    return 1 if found else 0


def _dead_pid():
    """A PID that provably no longer exists."""
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    pid = proc.pid
    proc.wait()
    try:
        import psutil
        if psutil.pid_exists(pid):
            return None
    except ImportError:
        pass
    return pid


def cmd_c5(_args):
    tmp = Path(tempfile.mkdtemp(prefix='r6_c5_'))
    target = tmp / 'connections.parquet'
    _table(target, rows=50_000)
    pid = _dead_pid()
    if pid is None:
        print('C5 verdict: BLOCKED (could not obtain a dead PID)')
        return 2
    orphan = tmp / f'.connections.parquet.compact.{pid}.tmp'
    orphan.write_bytes(b'orphaned by a killed run')
    started = time.perf_counter()
    removed = reencode_parquet_lossless(
        str(target), {'compression': 'zstd', 'use_dictionary': False})
    elapsed = time.perf_counter() - started
    print(f'dead writer PID        : {pid}')
    print(f'orphan planted         : {orphan.name}')
    print(f'reencode returned      : {removed}')
    print(f'elapsed                : {elapsed:.2f} s')
    print(f'orphan still present   : {orphan.exists()}  '
          f'(6-hour age rule would have kept it)')
    rc = 0 if not orphan.exists() else 1
    print(f'C5 verdict             : {"PASS" if rc == 0 else "FAIL"}')
    return rc


def cmd_c6(_args):
    """A live writer's in-progress temp must survive another run's sweep."""
    tmp = Path(tempfile.mkdtemp(prefix='r6_c6_'))
    target = tmp / 'connections.parquet'
    _table(target, rows=50_000)
    holder = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)'])
    time.sleep(0.4)
    live = tmp / f'.connections.parquet.compact.{holder.pid}.tmp'
    live.write_bytes(b'in-progress by a live process')
    remove_stale_temp_files(str(target), 'compact')
    survived = live.exists()
    print(f'live holder PID        : {holder.pid}')
    print(f'live temp survived     : {survived}')
    holder.terminate()
    holder.wait(timeout=10)
    remove_stale_temp_files(str(target), 'compact')
    print(f'after holder exits     : {live.exists()}')
    rc = 0 if survived and not live.exists() else 1
    print(f'C6 verdict             : {"PASS" if rc == 0 else "FAIL"}')
    return rc


def cmd_probe_atomic(_args):
    """The verifier's own probe, run standalone for the A2 evidence."""
    tmp = Path(tempfile.mkdtemp(prefix='r6_a2_'))
    target = tmp / 'roundtrip.parquet'
    write_parquet_atomic(
        str(target),
        lambda temp: pl.DataFrame({'v': [1, 2, 3]}).write_parquet(temp))
    got = pl.read_parquet(target)['v'].to_list()
    ok_encode = reencode_parquet_lossless(
        str(target), {'compression': 'zstd', 'use_dictionary': False})
    marker = pq.read_schema(str(target)).metadata.get(b'DROCAT.lossless')
    print(f'round-trip values      : {got}')
    print(f're-encode returned     : {ok_encode}')
    print(f'footer marker          : {marker}')
    print(f'temp leftovers         : '
          f'{[p.name for p in tmp.iterdir() if p.name.endswith(".tmp")]}')
    return 0 if got == [1, 2, 3] and ok_encode and marker == b'v1' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['c1', 'c4', 'c5', 'c6', 'atomic'])
    args = parser.parse_args(argv)
    return {'c1': cmd_c1, 'c4': cmd_c4, 'c5': cmd_c5, 'c6': cmd_c6,
            'atomic': cmd_probe_atomic}[args.action](args)


if __name__ == '__main__':
    sys.exit(main())
