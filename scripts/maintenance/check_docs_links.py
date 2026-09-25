#!/usr/bin/env python3
"""Fail when active documentation references files that do not exist.

Checks markdown links and backticked ``docs/``-, ``scripts/``- and
``tests/``-style repo paths inside README.md, docs/ (excluding docs/archive
and the audit records under docs/audits, which quote broken paths as
evidence) and skills/. A broken reference is one that names a file without
an accompanying "(removed)" / "(never shipped)" annotation — historical
mentions of retired artifacts are legitimate prose.

Usage
    python scripts/maintenance/check_docs_links.py
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MD_LINK = re.compile(r'\]\(([^)#\s]+\.md)\)')
CODE_PATH = re.compile(
    r'`((?:docs|scripts|tests|skills|src|ui)/[A-Za-z0-9_./-]+\.(?:md|py|sh|json))`')
REMOVED_HINT = re.compile(
    r'removed|never shipped|retired|superseded|planned; never|'
    r'has since been removed|since moved|historical|now lives at',
    re.I)


def _target_in_repo(ref: str, base: Path) -> Path:
    return (base / ref).resolve() if not ref.startswith('/') else Path(ref)


def check_file(path: Path) -> list:
    broken = []
    text = path.read_text(encoding='utf-8')
    for match in MD_LINK.finditer(text):
        ref = match.group(1)
        if ref.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        if not _target_in_repo(ref, path.parent).exists():
            line = text[:match.start()].count('\n') + 1
            broken.append(f"{path}:{line} -> {ref}")
    for match in CODE_PATH.finditer(text):
        ref = match.group(1)
        # A mention in the same sentence as a removal annotation is
        # historical prose, not a live reference.
        window = text[max(0, match.start() - 120):match.end() + 160]
        if REMOVED_HINT.search(window):
            continue
        if not (REPO / ref).exists():
            line = text[:match.start()].count('\n') + 1
            broken.append(f"{path}:{line} -> {ref}")
    return broken


def main() -> int:
    targets = [REPO / 'README.md']
    for sub in ('docs', 'skills'):
        for f in (REPO / sub).rglob('*.md'):
            parts = set(f.parts)
            if 'archive' in parts or 'audits' in parts or 'checked' in parts:
                continue
            targets.append(f)
    broken = []
    for target in targets:
        if target.exists():
            broken.extend(check_file(target))
    if broken:
        print(f"{len(broken)} broken documentation reference(s):",
              file=sys.stderr)
        for line in broken:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"docs link check OK ({len(targets)} files)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
