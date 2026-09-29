#!/usr/bin/env python
"""Round-6 §J harness: the transplanted FAFB real-data program (J1-J3).

Runs entirely on the manually downloaded FAFB v783 bundle — no NeuPrint or CAVE
traffic beyond what the local tables already hold.

    python t_fafb_j.py j1     pathfinding baseline + provenance re-stamp
    python t_fafb_j.py j2     hemisphere graph-cache key (same process, 2nd run)
    python t_fafb_j.py j3     reciprocal via the constructor field alone
    python t_fafb_j.py j123   all three in one process (J2 must follow J1)

Every [invariant] in the notes is asserted here; counts are only recorded, since
this host's FAFB copy may differ from the Mac reference (83/26 paths, 242/75
cone rows, 654/343 reciprocal rows).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
    else:
        # Fail loudly instead of walking on with a nonsense ancestor (a
        # drive root) that only surfaces later as confusing path errors.
        raise SystemExit(
            f'project root not found above {HERE} (no src/coana.py within '
            '5 ancestors, and no sibling drocat/ checkout)')

OUT = HERE.parent / 'j_runs'
for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DATASET = 'flywire_FAFB_v783'
SOURCE = ['s-LNv']
TARGET = ['DN1a']

FC_KWARGS = dict(
    dataset=DATASET,
    sourceNeurons=SOURCE,
    targetNeurons=TARGET,
    min_synapse_num=3,
    max_interlayer=2,
    drop_untyped=False,
    skip_bodyId=False,
    use_cache=True,
    pathfinding='StrongestFirst',
    keyword_in_path_to_remove=['None'],
    graph_edge_limit_bodyid=1000000,
    showfig=False,
    verbose_mode='simple',
)


def _snapshot(run_dir: Path) -> dict:
    """Everything a J row asserts on, from one run folder."""
    out = {'path': str(run_dir), 'files': sorted(
        str(p.relative_to(run_dir)) for p in run_dir.rglob('*') if p.is_file())}
    for name in ('parameters.txt', 'all_attributes.json', 'user_warning_notes.txt'):
        hits = [p for p in run_dir.rglob(name)]
        if hits:
            out[name] = hits[0].read_text(encoding='utf-8', errors='replace')
    for csv in sorted(run_dir.rglob('*allpaths_type.csv')):
        text = csv.read_text(encoding='utf-8', errors='replace').splitlines()
        out.setdefault('type_csv', {})[csv.name] = {
            'rows': max(len(text) - 1, 0), 'header': text[0] if text else ''}
    for csv in sorted(run_dir.rglob('*.csv')):
        if 'cone' in csv.name.lower() or 'pruned' in csv.name.lower():
            rows = csv.read_text(encoding='utf-8', errors='replace').splitlines()
            out.setdefault('cone_csv', {})[csv.name] = max(len(rows) - 1, 0)
    out['provenance_incomplete'] = any(
        '.provenance_incomplete' in p.name for p in run_dir.rglob('*'))
    out['find_reciprocal_dir'] = str(run_dir / 'find_reciprocal') \
        if (run_dir / 'find_reciprocal').is_dir() else None
    if out['find_reciprocal_dir']:
        out['reciprocal_rows'] = {
            p.name: max(len(p.read_text(encoding='utf-8', errors='replace')
                         .splitlines()) - 1, 0)
            for p in (run_dir / 'find_reciprocal').glob('*.csv')}
    return out


def _newest_run(before):
    candidates = [p for p in OUT.rglob('*/')
                  if p.is_dir() and p not in before and any(p.iterdir())]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _dirs():
    return {p for p in OUT.rglob('*/') if p.is_dir()} if OUT.exists() else set()


def _assert(label, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {label}' + (f' — {detail}' if detail else ''))
    return bool(ok)


def report(snapshot, tag):
    print(f'\n===== {tag}: run folder {snapshot["path"]}')
    print(f'  files: {len(snapshot["files"])}')
    params = snapshot.get('parameters.txt', '')
    notes = snapshot.get('user_warning_notes.txt', '')
    attrs_raw = snapshot.get('all_attributes.json', '')
    try:
        attrs = json.loads(attrs_raw) if attrs_raw else {}
    except json.JSONDecodeError:
        attrs = {}
    counts = {name: meta['rows'] for name, meta in snapshot.get('type_csv', {}).items()}
    print(f'  type-CSV path rows: {counts or "none found"}')
    print(f'  cone rows         : {snapshot.get("cone_csv", {}) or "no cone file named *cone*"}')
    tail = [ln for ln in params.splitlines()
            if 'applied_tau' in ln or 'edge_weight_floor' in ln
            or 'applied_threshold' in ln]
    print(f'  provenance lines  : {tail}')
    src = attrs.get('applied_threshold_source',
                    attrs.get('parameters', {}).get('applied_threshold_source')
                    if isinstance(attrs.get('parameters'), dict) else None)
    print(f'  applied_threshold_source: {src}')
    note_hits = {key: (key in notes) for key in
                 ('hop-budget pruning', 'natural tau', 'hemisphere',
                  'enrichment', 'find_reciprocal')}
    print(f'  note probes       : {note_hits}')
    print(f'  .provenance_incomplete present: {snapshot["provenance_incomplete"]}')
    if snapshot['find_reciprocal_dir']:
        print(f'  find_reciprocal/  : {snapshot["reciprocal_rows"]}')
    return {'counts': counts, 'notes': note_hits, 'src': src,
            'provenance_incomplete': snapshot['provenance_incomplete']}


def run_j1():
    from coana import FindNeuronConnection
    OUT.mkdir(exist_ok=True)
    before = _dirs()
    started = time.perf_counter()
    fc = FindNeuronConnection(**FC_KWARGS, output_dir=str(OUT / 'j1'))
    fc.InitializeNeuronInfo()
    fc.FindAllPath()
    elapsed = time.perf_counter() - started
    run = _newest_run(_dirs() - before) or OUT / 'j1'
    print(f'\nJ1 completed in {elapsed:.1f} s (Mac reference 90-120 s warm)')
    snap = _snapshot(run)
    summary = report(snap, 'J1')
    print('J1 [invariant] streamed type CSV complete '
          '(no path row dropped for a `None` substring):')
    dropped = [name for name, meta in snap.get('type_csv', {}).items()
               if 'None' in meta['header']]
    ok = _assert('no `None`-substring sentinel in the written header',
                 not dropped, str(dropped))
    ok &= _assert('no .provenance_incomplete marker',
                  not snap['provenance_incomplete'])
    ok &= _assert('applied_threshold_source == requested',
                  summary['src'] == 'requested', str(summary['src']))
    ok &= _assert('hop-budget + natural-tau notes present',
                  summary['notes']['hop-budget pruning']
                  and summary['notes']['natural tau'])
    ok &= _assert('no [hemisphere] block', not summary['notes']['hemisphere'])
    print(f'J1 overall: {"PASS" if ok else "FAIL"}')
    return snap


def run_j2(first):
    """Only hemisphere_filter differs — the graph cache must re-discover."""
    from coana import FindNeuronConnection
    before = _dirs()
    fc = FindNeuronConnection(**FC_KWARGS, hemisphere_filter='left',
                              output_dir=str(OUT / 'j2_left'))
    fc.InitializeNeuronInfo()
    fc.FindAllPath()
    run = _newest_run(_dirs() - before) or OUT / 'j2_left'
    snap = _snapshot(run)
    print('\n===== J2 (hemisphere_filter=left, same process)')
    left_counts = {n: m['rows'] for n, m in snap.get('type_csv', {}).items()}
    right_counts = {n: m['rows'] for n, m in first.get('type_csv', {}).items()}
    print(f'  J1 cone rows {first.get("cone_csv", {})} -> J2 {snap.get("cone_csv", {})}')
    print(f'  J1 paths     {right_counts} -> J2 {left_counts}')
    ok = True
    def total(d):
        return sum((v.get('rows') if isinstance(v, dict) else v)
                   for v in (d or {}).values())
    both = total(right_counts), total(left_counts)
    ok &= _assert('[invariant] the second run RE-DISCOVERED (counts shrank)',
                  both[1] < both[0], f'{both[0]} -> {both[1]}')
    r_hits = []
    for name, meta in snap.get('type_csv', {}).items():
        text = (Path(snap['path']) / name)
        body = text.read_text(encoding='utf-8', errors='replace')
        r_hits = [ln for ln in body.splitlines()[1:] if re.search(r'_R\b|_R_', ln)]
    ok &= _assert('zero _R type labels under the left filter',
                  not r_hits, f'{len(r_hits)} offending rows')
    notes = snap.get('user_warning_notes.txt', '')
    ok &= _assert('hemisphere note appears',
                  '[hemisphere]' in notes and 'hemisphere_filter=left' in notes)
    print(f'J2 overall: {"PASS" if ok else "FAIL"}  '
          f'(Mac reference cone 242->75, paths 83->26)')
    return snap


def run_j3():
    from coana import FindNeuronConnection
    before = _dirs()
    fc = FindNeuronConnection(**FC_KWARGS, find_reciprocal=True,
                              output_dir=str(OUT / 'j3_reciprocal'))
    fc.InitializeNeuronInfo()
    fc.FindAllPath()          # no method argument: the constructor field alone
    run = _newest_run(_dirs() - before) or OUT / 'j3_reciprocal'
    snap = _snapshot(run)
    print('\n===== J3 (find_reciprocal=True at construction, plain FindAllPath())')
    ok = _assert('[invariant] find_reciprocal/ subfolder exists',
                 snap['find_reciprocal_dir'] is not None)
    rows = snap.get('reciprocal_rows', {})
    ok &= _assert('both reciprocal CSVs populated',
                  bool(rows) and all(v > 0 for v in rows.values()), str(rows))
    ok &= _assert('- [enrichment] note present',
                  '[enrichment]' in snap.get('user_warning_notes.txt', ''))
    print(f'J3 overall: {"PASS" if ok else "FAIL"}  (Mac reference 654/343 rows)')
    return snap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['j1', 'j2', 'j3', 'j123'])
    args = parser.parse_args(argv)
    if args.action == 'j1':
        run_j1()
    elif args.action == 'j2':
        run_j2(run_j1())
    elif args.action == 'j3':
        run_j3()
    else:
        first = run_j1()
        run_j2(first)
        run_j3()
    return 0


if __name__ == '__main__':
    sys.exit(main())
