#!/usr/bin/env python3
"""Run the repository test suite only while no other session is busy.

This working tree is shared by several live agents, and two pytest sessions at
once produce phantom FileNotFoundError failures here. The script therefore:

1. polls for competing processes (any pytest, the NiceGUI app, a DROCAT
   pipeline script, or a second http.server) and requires QUIET_STREAK
   consecutive clean polls before it starts;
2. runs the suite in sequential stages, so a stage boundary re-checks for a
   session that started while the previous stage ran;
3. writes a timestamped log plus a machine-readable summary the caller can
   poll without touching the terminal.

Usage:  python scripts/maintenance/wait_then_full_suite.py [--dry-run] [--with-e2e] [--stages=...]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / 'local_data'
POLL_SECONDS = 20
QUIET_STREAK = 3          # ~60 s of quiet before anything is launched
MAX_WAIT_SECONDS = 6 * 3600

STAGES = [
    ('core', ['tests/core']),
    ('ui', ['tests/ui']),
    ('docs-and-misc', ['tests/docs', 'tests/vispath',
                       'tests/test_drocat_usage_skill.py']),
]
E2E_STAGE = ('e2e', ['tests/e2e'])

# The project env, not whatever python happens to be on PATH: the base env
# carries pytest 7.4 while this one carries 9.1, and the suite is maintained
# against the project env the app itself runs in.
def _project_env_name() -> str:
    # The versioned env name follows ui/config.py's APP_VERSION, so a bump
    # does not silently detach this script from the env it should run in.
    env = 'drocat-4.5.0'
    cfg = Path(__file__).resolve().parents[2] / 'ui' / 'config.py'
    try:
        m = re.search(r'^APP_VERSION\s*=\s*["\']([\d.]+)["\']',
                      cfg.read_text(), re.M)
        if m:
            env = f'drocat-{m.group(1)}'
    except OSError:
        pass
    return env


def _project_env_python() -> str:
    """The project env's interpreter, derived per platform.

    macOS (the dev host) keeps envs at ~/anaconda3/envs/<env>/bin/python;
    the Windows test hosts (see the retest notes) use
    %USERPROFILE%\anaconda3\envs\<env>\python.exe. Both derive the env
    name from APP_VERSION so a version bump cannot detach the script.
    """
    name = _project_env_name()
    if sys.platform.startswith('win'):
        return str(Path.home() / 'anaconda3' / 'envs' / name / 'python.exe')
    return str(Path.home() / 'anaconda3' / 'envs' / name / 'bin' / 'python')


_PROJECT_PY = _project_env_python()
PYTEST_PY = os.environ.get('DROCAT_TEST_PYTHON') or (
    _PROJECT_PY if Path(_PROJECT_PY).exists() else sys.executable)

# Anything that means "another session is mid-flight". Rather than list every
# pipeline entry point, any interpreter from the project env counts - except the
# two long-lived servers below, which cannot collide with a test run (the
# user's calls, 2026-09-22, after an idle `ui/app.py` held the queue for 10 h
# and a static `http.server` held it again for minutes). My own process tree is
# excluded in competing(), so the suite never blocks on itself.
# Any interpreter from the project env counts as a competitor: match the
# env directory itself (envs[/\]drocat-<ver>) so the posix and the Windows
# command-line spellings both hit.
COMPETITOR = re.compile(
    r'(?:[-/]m\s+pytest(?:\s|$)|(?<![\w./])pytest\s+tests|'
    'envs[/\\\\]' + re.escape(_project_env_name()) + ')')

IDLE_SERVER = re.compile(
    r'(?:ui[/\\]app\.py|-m http\.server|wait_then_full_suite|'
    r'multiprocessing\.(?:resource_tracker|spawn|popen_fork)'
    r'|spawn_main|_main_loop)')


def now() -> str:
    return datetime.now().strftime('%H:%M:%S')


def all_processes() -> list[tuple[int, int, str]]:
    """(pid, ppid, command) for every process, from one ps call."""
    out = subprocess.run(['ps', '-eo', 'pid=,ppid=,command='],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def descendants(root: int) -> set[int]:
    rows = all_processes()
    children: dict[int, list[int]] = {}
    for pid, ppid, _ in rows:
        children.setdefault(ppid, []).append(pid)
    seen, stack = {root}, [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def competing() -> list[str]:
    """Other sessions' live work; my own process tree is never a competitor."""
    mine = descendants(os.getpid())
    hits = []
    for pid, _, cmd in all_processes():
        if pid in mine or IDLE_SERVER.search(cmd):
            continue
        if COMPETITOR.search(cmd):
            hits.append(f'{pid}: {cmd[:140]}')
    return hits


def wait_for_quiet(log) -> None:
    streak, waited = 0, 0
    while True:
        hits = competing()
        if not hits:
            streak += 1
            if streak >= QUIET_STREAK:
                log(f'[{now()}] quiet for {QUIET_STREAK} polls '
                    f'({waited}s waited); starting')
                return
        else:
            streak = 0
            log(f'[{now()}] busy ({len(hits)}): ' + ' ;; '.join(hits[:4]))
        waited += POLL_SECONDS
        if waited > MAX_WAIT_SECONDS:
            raise SystemExit(f'gave up after {MAX_WAIT_SECONDS}s of busy tree')
        time.sleep(POLL_SECONDS)


def run_stage(name: str, targets: list[str], log_path: Path, log) -> dict:
    wait_for_quiet(log)
    out_log = log_path.with_suffix(f'.{name}.log')
    cmd = [PYTEST_PY, '-m', 'pytest', *targets, '-q', '-rf',
           '--durations=15', '-p', 'no:cacheprovider']
    t0 = time.time()
    with open(out_log, 'w', encoding='utf-8') as fh:
        fh.write('$ ' + ' '.join(cmd) + '\n\n')
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    tail = out_log.read_text(encoding='utf-8', errors='replace').strip().splitlines()
    summary = next((l for l in reversed(tail)
                    if re.search(r'\d+ (?:passed|failed|error|no tests ran)', l)),
                   tail[-1] if tail else '')
    log(f'[{now()}] stage {name}: rc={proc.returncode} in {dur/60:.1f}m — {summary}')
    return {'stage': name, 'rc': proc.returncode, 'seconds': round(dur, 1),
            'summary': summary, 'log': str(out_log)}


def main() -> int:
    dry = '--dry-run' in sys.argv
    with_e2e = '--with-e2e' in sys.argv
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'full_suite_{stamp}'
    lines_path = log_path.with_suffix('.runner.log')

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(lines_path, 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n')

    if dry:
        hits = competing()
        print('competitors detected now:' if hits else 'quiet')
        for h in hits:
            print('  ', h)
        return 0

    stages = list(STAGES) + ([E2E_STAGE] if with_e2e else [])
    only = next((a.split('=')[1] for a in sys.argv if a.startswith('--stages=')), '')
    if only:
        # Re-run a subset after a stage already passed, e.g.
        # --stages=docs-and-misc,recheck
        named = dict(stages)
        named['recheck'] = ['tests/core/test_parquet_utils.py']
        stages = [(k, named[k]) for k in only.split(',') if k in named]
    log(f'[{now()}] waiting for a quiet tree; stages={[s[0] for s in stages]}; '
        f'interpreter={PYTEST_PY}')
    results = []
    for name, targets in stages:
        results.append(run_stage(name, targets, log_path, log))
    with open(log_path.with_suffix('.summary.json'), 'w', encoding='utf-8') as fh:
        json.dump({'started': stamp, 'stages': results,
                   'e2e_included': with_e2e}, fh, indent=2)
    bad = [r for r in results if r['rc'] != 0]
    log(f'[{now()}] done; {len(results) - len(bad)}/{len(results)} stages clean')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
