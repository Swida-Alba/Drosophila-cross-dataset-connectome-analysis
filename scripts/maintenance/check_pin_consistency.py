#!/usr/bin/env python3
"""Fail when dependency pins disagree between packaging files.

`pyproject.toml` ([project].dependencies), `requirements.txt`, and
`requirements-windows.txt` describe the same tested environment; when the
same package is pinned to different versions in two of them, `pip install
drocat` and `pip install -r requirements.txt` silently resolve different
stacks (audit F-CONFIG-002). Same-package conflicts fail the check; a pin
present in one file only, or a range in pyproject vs an exact pin, is
reported informationally (requirements.txt stays the source of truth for
the tested environment).

Usage
    python scripts/maintenance/check_pin_consistency.py
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - 3.10 needs tomli
    try:
        import tomli as tomllib
    except ImportError:
        print("check_pin_consistency: tomllib/tomli unavailable "
              "(Python <3.11 without tomli)", file=sys.stderr)
        sys.exit(2)

REQ_NAME = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s#]+)')
SPEC_NAME = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$')


def _canonical(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _parse_requirements(path: Path) -> dict:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        match = REQ_NAME.match(line)
        if match:
            pins[_canonical(match.group(1))] = match.group(2)
    return pins


def _parse_pyproject() -> tuple:
    data = tomllib.loads((REPO / 'pyproject.toml').read_text())
    project = data.get('project', {})
    exact, ranged = {}, {}
    sections = {'dependencies': project.get('dependencies', [])}
    for extra, deps in (project.get('optional-dependencies') or {}).items():
        sections[f'optional-dependencies:{extra}'] = deps
    for section, specs in sections.items():
        for spec in specs:
            match = SPEC_NAME.match(spec.strip())
            if not match:
                continue
            name, rest = _canonical(match.group(1)), match.group(2).strip()
            if '==' in rest and ',' not in rest.split('==')[0]:
                exact[name] = (rest.split('==')[1].strip(), section)
            elif rest:
                ranged[name] = (rest, section)
    return exact, ranged


def main() -> int:
    req = _parse_requirements(REPO / 'requirements.txt')
    req_win = _parse_requirements(REPO / 'requirements-windows.txt')
    exact, ranged = _parse_pyproject()

    conflicts = []
    # requirements files must agree with each other and with pyproject pins
    for name, version in sorted(req.items()):
        other = req_win.get(name)
        if other and other != version:
            conflicts.append(
                f"{name}: requirements.txt=={version} but "
                f"requirements-windows.txt=={other}")
        py = exact.get(name)
        if py and py[0] != version:
            conflicts.append(
                f"{name}: requirements.txt=={version} but "
                f"pyproject {py[1]}=={py[0]}")
    for name, (spec, section) in sorted(ranged.items()):
        if name in req:
            print(f"note: {name} is a range in pyproject ({section}: {spec}) "
                  f"and an exact pin in requirements.txt "
                  f"(=={req[name]}); requirements.txt is the tested env")

    if conflicts:
        for conflict in conflicts:
            print(f"CONFLICT: {conflict}", file=sys.stderr)
        print(f"{len(conflicts)} pin conflict(s) — the files describe "
              "different environments", file=sys.stderr)
        return 1
    print(f"pin consistency OK "
          f"({len(req)} requirements pins, {len(exact)} pyproject exact pins, "
          f"{len(ranged)} pyproject ranges)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
