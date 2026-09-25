#!/usr/bin/env python3
"""Stage ONLY my hunks from a file another session also edited.

A shared working tree means `git add <file>` can commit someone else's
work-in-progress. This splits each listed file's unstaged diff into hunks, calls
a hunk MINE when its body contains one of the identifying substrings I supply,
and stages just those — so the commit that follows carries my change and
nothing else.

    python scripts/maintenance/stage_own_hunks.py \
        --select 'ui/output_guide.py=morph_pool_ref' \
        --select 'docs/OUTPUT_FILES.md=the deciding pair'[,more,substrings]
        [--apply] [--dump-dir DIR]

Dry-run by default: it prints every hunk decision (`MINE` / `foreign`) and writes
the selected patch per file to a temporary directory (or --dump-dir) so the
selection can be READ before anything is staged. --apply runs
`git apply --cached` on those hunks.

Deliberate limits:
* only paths named by --select are ever looked at, and only their UNSTAGED
  diff — staged work and untracked files are left alone;
* a hunk that matches nothing is NOT staged, so a partial match leaves the
  rest for the other session rather than guessing;
* it exits non-zero when a named path has no changed hunks or none matched —
  the usual sign that the substring is stale, not that the change is gone.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def hunks_of(path):
    """The file's unstaged diff, split into (header, [hunk lines...])."""
    diff = subprocess.run(['git', 'diff', '--', path], cwd=REPO,
                          capture_output=True, text=True).stdout
    if not diff.strip():
        return [], []
    lines = diff.splitlines(keepends=True)
    head, out, cur = [], [], None
    for ln in lines:
        if ln.startswith('@@'):
            if cur:
                out.append(cur)
            cur = [ln]
        elif cur is None:
            head.append(ln)
        else:
            cur.append(ln)
    if cur:
        out.append(cur)
    return head, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--select', action='append', default=[],
                    metavar='PATH=SUBSTR[,SUBSTR…]',
                    help='a file to split and what marks a hunk as mine')
    ap.add_argument('--apply', action='store_true',
                    help='stage the selected hunks (default: dry run)')
    ap.add_argument('--dump-dir', default=None,
                    help='where to write the selected patches (default: a '
                         'temporary dir; nothing is written under local_data/)')
    args = ap.parse_args(argv)
    if not args.select:
        ap.error('nothing to do: pass at least one --select PATH:SUBSTR')
    dump = Path(args.dump_dir) if args.dump_dir \
        else Path(tempfile.mkdtemp(prefix='stage_own_hunks_'))
    if not args.dump_dir:
        print(f'# dry-run patches go to {dump}')
    problems = []
    for spec in args.select:
        path, _, marks = spec.partition('=')
        if not marks:
            ap.error(f'--select needs PATH=SUBSTR[,SUBSTR]: {spec!r}')
        path = path.strip()
        marks = [m.strip() for m in marks.split(',') if m.strip()]
        head, hs = hunks_of(path)
        print(f'==== {path}  ({len(hs)} changed hunks; marks: {marks})')
        if not hs:
            problems.append(f'{path}: no unstaged hunks')
            continue
        keep = []
        for i, h in enumerate(hs):
            body = ''.join(h)
            mine = any(m in body for m in marks)
            span = h[0].split('@@')[1].strip() if h[0].count('@@') > 1 else '?'
            print(f"    hunk {i:<2} {span:<30} {'MINE' if mine else 'foreign'}")
            if mine:
                keep.append(h)
        if not keep:
            problems.append(f'{path}: no hunk matched {marks}')
            continue
        patch = ''.join(head) + ''.join(''.join(h) for h in keep)
        if args.apply:
            subprocess.run(['git', 'apply', '--cached', '--allow-empty', '-'],
                           cwd=REPO, input=patch, text=True, check=True)
            print(f'    -> staged {len(keep)}/{len(hs)} hunks')
        else:
            out = dump / (path.replace('/', '_') + '.patch')
            out.write_text(patch, encoding='utf-8')
            print(f'    -> {len(keep)}/{len(hs)} hunks selected; patch {out}')
    if problems:
        print('\n== unresolved ==')
        for p in problems:
            print('  !', p)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
