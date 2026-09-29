#!/usr/bin/env python
"""§J5/§J6: the FAFB + BANC edge-mode comparison and the fingerprint mismatch.

J5 runs the notes' exact edge-mode construction, then asserts the edge-mode
provenance invariant (applied == requested, tau null) and the export set.
J6 re-uses the SAME output folder in path mode with a different target query:
both datasets must log "written for a different query - ignoring it" and
re-derive, and the path-mode-only fingerprint sidecars must then appear.
"""

from __future__ import annotations

import json
import os
import shutil
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

OUT = HERE.parent / 'j5_runs'
for entry in (str(PROJECT / 'src'), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DATASETS = ['flywire_FAFB_v783', 'banc_v888']


def _show(label, run_root: Path):
    print(f'\n===== {label}: {run_root}')
    if not run_root.is_dir():
        print('  MISSING')
        return {}
    files = sorted(str(p.relative_to(run_root)) for p in run_root.rglob('*')
                   if p.is_file())
    print(f'  files: {len(files)}')
    out = {'root': run_root, 'files': files}

    cr = run_root / 'comparison_results'
    if cr.is_dir():
        print('  comparison_results/:')
        for p in sorted(cr.iterdir()):
            extra = ''
            if p.suffix == '.csv':
                lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
                extra = f'  rows={max(len(lines)-1,0)}'
                if p.name == 'unified_edge_comparison.csv' and lines:
                    print(f'      header: {lines[0][:300]}')
            print(f'      {p.name}{extra}')
            if p.name == 'unified_edge_comparison.csv':
                out['unified'] = lines

    for name in ('effective_thresholds.json', 'pathfinding_provenance.csv',
                 'comparison_report.html', 'query_resolution.csv'):
        hits = [p for p in run_root.rglob(name)]
        if hits:
            size = hits[0].stat().st_size
            print(f'  {name}: {size:,} bytes ({hits[0].parent.name}/)')
            out[name] = hits[0]
    ets = [p for p in run_root.rglob('effective_thresholds.json')]
    if ets:
        try:
            data = json.loads(ets[0].read_text(encoding='utf-8'))
            print('  effective_thresholds.json:')
            print('   ', json.dumps(data, indent=1)[:1400])
            out['et'] = data
        except Exception as exc:
            print('  effective_thresholds unreadable:', exc)

    side = sorted(run_root.rglob('connections_edge.fingerprint.json'))
    print(f'  fingerprint sidecars: {len(side)}')
    for p in side[:4]:
        print(f'      {p.relative_to(run_root)}')
    out['sidecars'] = side
    for p in run_root.rglob('auto_type_mapping*.csv'):
        print(f'  mapping export: {p.relative_to(run_root)}')
    return out


def check(label, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {label}' + (f' — {detail}' if detail else ''))
    return bool(ok)


def j5():
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    params = ComparisonParameters(
        datasets=DATASETS,
        source_neurons=['s-LNv.*'],
        target_neurons=['DN1a.*'],
        thresholds=[3],
        comparison_mode='edge',
        max_interlayer=2,
        output_folder=str(OUT),
        saveas='r6_j5_edge',
    )
    analyzer = ComparisonAnalyzer(params)
    analyzer.run_comparison()
    analyzer.export_results()
    run = OUT / 'r6_j5_edge'
    snap = _show('J5 edge mode', run)

    print('\n---- J5 assertions ----')
    rc = 0
    unified = snap.get('unified')
    rc |= not check('unified_edge_comparison.csv exists and is non-empty',
                    bool(unified) and len(unified) > 1,
                    f'{len(unified)-1 if unified else 0} rows (Mac ref ~188)')
    if unified:
        header = unified[0]
        rc |= not check('per-dataset weight + present columns and conservation',
                        all(k in header for k in ('conservation',))
                        and any('weight' in h for h in header.split(',')),
                        header[:160])
    et = snap.get('et')
    if et:
        rows = et if isinstance(et, list) else et.get('datasets', et)
        text = json.dumps(et)
        rc |= not check("edge-mode provenance: applied_threshold == 3 for each dataset",
                        text.count('"applied_threshold": 3') >= len(DATASETS),
                        f'{text.count(chr(34)+"applied_threshold"+chr(34)+": 3")} occurrences')
        tau_null = ('"tau": null' in text or '"tau": ""' in text
                    or '"tau": null' in text.lower() or 'tau' not in text)
        rc |= not check('edge-mode provenance: tau is null/empty', tau_null)
        rc |= not check('side_path_run block carries the side-run numbers',
                        'side_path_run' in text)
    else:
        rc |= not check('effective_thresholds.json present', False)
    rc |= not check('pathfinding_provenance.csv present',
                    'pathfinding_provenance.csv' in snap)
    html = snap.get('comparison_report.html')
    rc |= not check('comparison_report.html present and non-trivial',
                    html is not None and html.stat().st_size > 50_000,
                    f'{html.stat().st_size:,} bytes (Mac ref ~420 KB)' if html else '')
    rc |= not check('[invariant] no fingerprint sidecar in EDGE mode (path-mode only)',
                    not snap.get('sidecars'),
                    f'{len(snap.get("sidecars") or [])} found')
    return run, rc


def j6(first_run: Path):
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    log_path = HERE.parent / 'r6_j6_stdout.txt'
    handle = open(log_path, 'w', encoding='utf-8', errors='replace')
    saved = sys.stdout
    sys.stdout = handle
    try:
        params = ComparisonParameters(
            datasets=DATASETS,
            source_neurons=['s-LNv.*'],
            target_neurons=['DN2.*'],          # a DIFFERENT query
            thresholds=[3],
            comparison_mode='path',
            max_interlayer=2,
            output_folder=str(OUT),
            saveas='r6_j5_edge',               # reuse J5's folder
        )
        analyzer = ComparisonAnalyzer(params)
        analyzer.run_comparison()
        analyzer.export_results()
    finally:
        sys.stdout = saved
        handle.close()
    text = log_path.read_text(encoding='utf-8', errors='replace')
    print('\n===== J6: reused the edge-mode folder for a different path-mode query')
    hits = [ln.strip() for ln in text.splitlines()
            if 'different query' in ln or 'ignoring it' in ln]
    print(f'  "different query" lines found: {len(hits)}')
    for ln in hits[:8]:
        print(f'    {ln[:200]}')
    snap = _show('J6 path mode into the same folder', first_run)
    rc = 0
    per_ds = {ds: any(ds in ln for ln in hits) for ds in DATASETS}
    rc |= not check('[invariant] every dataset logs the fingerprint mismatch',
                    all(per_ds.values()) and len(hits) >= len(DATASETS),
                    str(per_ds))
    rc |= not check('path-mode run wrote the fingerprint sidecars',
                    len(snap.get('sidecars') or []) > 0,
                    f'{len(snap.get("sidecars") or [])} sidecars')
    for side in (snap.get('sidecars') or [])[:2]:
        print(f'    {side.name}: '
              f'{side.read_text(encoding="utf-8", errors="replace")[:400]}')
    return rc


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.chdir(PROJECT)
    run, rc = j5()
    rc |= j6(run)
    print('\nJ5+J6 OVERALL:', 'PASS' if rc == 0 else 'SEE FAILURES ABOVE')
    return 0 if rc == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
