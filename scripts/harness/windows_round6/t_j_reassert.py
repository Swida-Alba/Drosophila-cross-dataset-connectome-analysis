#!/usr/bin/env python
"""Re-assert the §J1/J2/J3 invariants over run folders already on disk.

The first pass mis-located the run folders, so this reads them by explicit
path and reports the [invariant] outcomes plus the counts the notes want.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'j_runs'

RUNS = {
    'J1 both hemispheres': ROOT / 'j1',
    'J2 hemisphere_filter=left': ROOT / 'j2_left',
    'J3 find_reciprocal=True': ROOT / 'j3_reciprocal',
}


def run_dir(base: Path) -> Path:
    subs = [p for p in base.iterdir() if p.is_dir()]
    return max(subs, key=lambda p: p.stat().st_mtime)


def csv_rows(path: Path) -> int:
    return max(len(path.read_text(encoding='utf-8', errors='replace')
                 .splitlines()) - 1, 0)


def notes(path: Path) -> str:
    p = path / 'user_warning_notes.txt'
    return p.read_text(encoding='utf-8', errors='replace') if p.exists() else ''


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    snapshots = {}
    for label, base in RUNS.items():
        if not base.is_dir():
            print(f'{label}: MISSING {base}')
            continue
        run = run_dir(base)
        type_csvs = sorted(run.glob('*allpaths_type.csv'))
        rows = {p.name: csv_rows(p) for p in type_csvs}
        data_details = run / 'data_details'
        detail = {}
        if data_details.is_dir():
            for p in sorted(data_details.glob('*.csv')):
                detail[p.name] = csv_rows(p)
        recip = {}
        if (run / 'find_reciprocal').is_dir():
            recip = {p.name: csv_rows(p)
                     for p in sorted((run / 'find_reciprocal').glob('*.csv'))}
        attrs = {}
        if (run / 'all_attributes.json').exists():
            try:
                attrs = json.loads((run / 'all_attributes.json')
                                   .read_text(encoding='utf-8'))
            except json.JSONDecodeError as exc:
                attrs = {'__error__': str(exc)}
        snapshots[label] = {
            'run': run, 'paths': sum(rows.values()), 'type_csvs': rows,
            'details': detail, 'reciprocal': recip, 'notes': notes(run),
            'threshold_source': attrs.get('applied_threshold_source'),
            'provenance_incomplete': any('.provenance_incomplete' in p.name
                                         for p in run.rglob('*')),
        }
        print(f'\n===== {label}')
        print(f'  run folder        : {run.name}')
        print(f'  type path rows    : {rows}  (total {sum(rows.values())})')
        print(f'  applied_threshold_source: {snapshots[label]["threshold_source"]}')
        print(f'  .provenance_incomplete  : {snapshots[label]["provenance_incomplete"]}')
        print(f'  data_details CSVs :')
        for name, count in detail.items():
            print(f'      {name}: {count}')
        if recip:
            print(f'  find_reciprocal/  :')
            for name, count in recip.items():
                print(f'      {name}: {count}')
        note_lines = [ln.strip() for ln in snapshots[label]['notes'].splitlines()
                      if ln.strip().startswith('- [')]
        print(f'  bracketed notes   : {note_lines}')

    j1 = snapshots.get('J1 both hemispheres')
    j2 = snapshots.get('J2 hemisphere_filter=left')
    j3 = snapshots.get('J3 find_reciprocal=True')

    print('\n\n================ INVARIANT ASSERTIONS ================')
    def check(label, ok, detail=''):
        print(f'[{"PASS" if ok else "FAIL"}] {label}' + (f' — {detail}' if detail else ''))
        return ok

    rc = 0
    if j1 and j2:
        rc |= not check(
            'J2 [invariant] hemisphere change RE-DISCOVERS the graph cache',
            j2['paths'] < j1['paths'],
            f"paths {j1['paths']} -> {j2['paths']} (Mac 83 -> 26)")
        left_csv = next(iter(j2['type_csvs'].values()), None)
        body = ''.join((j2['run'] / n).read_text(encoding='utf-8', errors='replace')
                       for n in j2['type_csvs'])
        r_rows = [ln for ln in body.splitlines()[1:] if re.search(r'_R\b|_R_|_R,', ln)]
        rc |= not check('J2 left filter held: zero _R labels',
                        not r_rows, f'{len(r_rows)} rows')
        rc |= not check('J2 hemisphere note appears',
                        '[hemisphere]' in j2['notes']
                        and 'hemisphere_filter=left' in j2['notes'])
        rc |= not check('J1 has no hemisphere note (control)',
                        '[hemisphere]' not in j1['notes'])
    if j3:
        rc |= not check(
            'J3 [invariant] find_reciprocal/ exists from the constructor field alone',
            bool(j3['reciprocal']))
        populated = {k: v for k, v in (j3['reciprocal'] or {}).items()
                     if k.startswith('reciprocal_connection')}
        rc |= not check('J3 both reciprocal connection CSVs populated',
                        bool(populated) and all(v > 0 for v in populated.values()),
                        str(populated))
        rc |= not check('J3 [enrichment] note present',
                        '[enrichment]' in j3['notes'])
        rc |= not check('J3 path rows unchanged vs J1 (enrichment is additive)',
                        j3['paths'] == (j1['paths'] if j1 else j3['paths']),
                        f"{j1['paths'] if j1 else '?'} -> {j3['paths']}")
    print('\nOVERALL:', 'PASS' if rc == 0 else 'SEE FAILURES ABOVE')
    return rc


if __name__ == '__main__':
    sys.exit(main())
