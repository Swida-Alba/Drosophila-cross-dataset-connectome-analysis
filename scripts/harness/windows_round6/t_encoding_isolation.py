#!/usr/bin/env python
"""Round-6 §D/§C isolation: the re-encode status print on a cp936 console.

`utils/parquet_utils.reencode_parquet_lossless` prints a `✓` on success and a
`⚠️` in its own exception handler, but unlike `coana`, `statvis` and
`visualize_skeleton` it never calls
`utils.console_encoding.ensure_utf8_stdio()`.  On a legacy Windows code page
this makes the *success* path abort.

Run each leg as its own process:

    python t_encoding_isolation.py bare          # import utils.parquet_utils only
    python t_encoding_isolation.py with_coana     # the documented entry point
    python t_encoding_isolation.py child_utf8     # child forced to UTF-8, parent cp936
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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

ENV_PY = os.environ.get('DROCAT_PY', sys.executable)

CHILD = r"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(%r, 'src'))
if %r == 'with_coana':
    import coana  # installs ensure_utf8_stdio()
import polars as pl
from utils.parquet_utils import reencode_parquet_lossless, parquet_lossless_marker
tmp = tempfile.mkdtemp(prefix='enc_')
target = os.path.join(tmp, 'x.parquet')
pl.DataFrame({'a': list(range(5000)), 'b': [f'R{i %% 7}' for i in range(5000)]}
             ).write_parquet(target)
try:
    r = reencode_parquet_lossless(target, {'compression': 'zstd',
                                           'use_dictionary': False})
    print('RESULT returned=%%s marker=%%s' %% (r, parquet_lossless_marker(target)))
except Exception as exc:
    print('RAISED %%s: %%s' %% (type(exc).__name__, exc))
    raise
""" % (str(PROJECT), '__MODE__')


def run(mode, child_io=None, label=''):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env.pop('PYTHONIOENCODING', None)
    if child_io:
        env['PYTHONIOENCODING'] = child_io
    code = CHILD.replace('__MODE__', mode)
    print(f'--- {label or mode}  (PYTHONIOENCODING={child_io or "unset"}) ---')
    proc = subprocess.run([ENV_PY, '-c', code], capture_output=True,
                          cwd=str(PROJECT), env=env)
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    print(out.strip())
    if err:
        tail = [ln for ln in err.splitlines() if ln.strip()][-4:]
        print('  stderr tail:')
        for ln in tail:
            print('   ', ln)
    print(f'  exit={proc.returncode}\n')
    return proc.returncode


def main(argv=None):
    leg = (argv or ['all'])[0]
    # This harness must survive printing the very characters it is testing.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(f'console code page: '
          f'{subprocess.run(["cmd", "/c", "chcp"], capture_output=True).stdout.decode("mbcs", "replace").strip()}')
    print(f'stream encoding of THIS parent: {sys.stdout.encoding}\n')
    if leg == 'bare':
        return run('bare')
    if leg == 'with_coana':
        return run('with_coana')
    if leg == 'child_utf8':
        return run('bare', child_io='utf-8', label='child forced UTF-8 (the V2 case)')
    rc = 0
    rc |= run('bare') == 0  # bare SHOULD succeed but does not -> invert
    rc |= run('with_coana')
    rc |= run('bare', child_io='utf-8', label='child forced UTF-8 (the V2 case)')
    return 0 if rc else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
