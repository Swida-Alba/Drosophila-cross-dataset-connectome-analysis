#!/usr/bin/env python3
"""Verify ``path:LINE`` citations in docs/*.md against the live tree.

I-7 (2026-09-28): line-number cites in a moving tree decay within hours —
``docs/OUTPUT_FILES.md``'s own stamp reads "Verified 2026-08-15", and
sibling commits shifted hot files by hundreds of lines inside one session.
This tool extracts every ``path:LINE`` cite from the docs, asserts the
line is in range, and — when the citing line names a symbol in backticks —
asserts that symbol appears within ±3 lines of the cite (the recipe used
to build ``docs/ARTIFACT_INDEX.md``, which deliberately cites by symbol or
grep-able string instead of line).  Every stale cite is listed; the exit
code is 1 when any cite is stale, so it can gate a docs change in CI.

A cite is "stale" when the path is missing, the line is out of range, or
the cited line's neighbourhood no longer carries the named symbol.

Usage:
    python scripts/maintenance/verify_doc_cites.py [--docs docs] [--quiet]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# `relative/path.ext:LINE` — extension-bounded so prose like "2026-09-28:5"
# never matches; the leading boundary rejects a preceding path character.
CITE_RE = re.compile(
    r'(?<![\w./\\-])'
    r'((?:[\w.-]+/)*[\w.-]+\.(?:py|js|md|txt|json|html)):(\d+)')

# A backticked identifier-ish token on the citing line — the symbol the
# cite is supposed to sit next to.
TICK_RE = re.compile(r'`([A-Za-z_][\w.]*)`')

SYMBOL_WINDOW = 3  # ± lines the symbol may have drifted


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_paths(raw: str, doc_dir: Path, root: Path):
    yield root / raw
    yield doc_dir / raw
    yield doc_dir / '..' / raw


def collect_cites(doc: Path):
    """All (line_no, raw_path, line_no_in_target, symbols) cites of one
    doc, skipping fenced code blocks."""
    cites = []
    in_fence = False
    for lineno, line in enumerate(
            doc.read_text(encoding='utf-8').splitlines(), 1):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in CITE_RE.finditer(line):
            symbols = [t for t in TICK_RE.findall(line)
                       if '.' in t or not t[0].isdigit()]
            cites.append((lineno, m.group(1), int(m.group(2)), symbols))
    return cites


def check_cite(root: Path, doc: Path, raw: str, target_line: int,
               symbols) -> str:
    """'ok' or a short stale reason for one cite."""
    target = None
    for cand in _candidate_paths(raw, doc.parent, root):
        if cand.is_file():
            target = cand
            break
    if target is None:
        return f'missing file {raw!r}'
    lines = target.read_text(encoding='utf-8', errors='replace').splitlines()
    if not 1 <= target_line <= len(lines):
        return (f'line {target_line} out of range '
                f'({len(lines)} lines in {raw})')
    if not symbols:
        return 'ok'
    window = lines[max(0, target_line - 1 - SYMBOL_WINDOW):
                   target_line + SYMBOL_WINDOW]
    window_text = '\n'.join(window)
    for sym in symbols:
        leaf = sym.split('.')[-1]
        if leaf and leaf in window_text:
            return 'ok'
    return (f'symbol {symbols[0]!r} not within ±{SYMBOL_WINDOW} lines '
            f'of {raw}:{target_line}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Check docs/*.md path:LINE cites against the tree.')
    ap.add_argument('--docs', type=Path, default=Path('docs'),
                    help='docs directory to scan (default docs/)')
    ap.add_argument('--quiet', action='store_true',
                    help='print only the summary')
    args = ap.parse_args(argv)
    root = repo_root()
    docs_dir = args.docs if args.docs.is_absolute() else root / args.docs
    total = ok = 0
    problems = []
    for doc in sorted(docs_dir.rglob('*.md')):
        for lineno, raw, target_line, symbols in collect_cites(doc):
            total += 1
            verdict = check_cite(root, doc, raw, target_line, symbols)
            if verdict == 'ok':
                ok += 1
            elif not args.quiet:
                problems.append(
                    f'{doc.relative_to(root)}:{lineno}  '
                    f'{raw}:{target_line}  — {verdict}')
    for line in problems:
        print(line)
    stale = total - ok
    print(f'{total} cites checked: {ok} ok, {stale} stale')
    return 1 if stale else 0


if __name__ == '__main__':
    raise SystemExit(main())
