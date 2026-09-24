#!/usr/bin/env python3
"""What differs between two TM VEV runs — the gate for any change that touches
scoring, caching or the candidate windows.

Usage
    python scripts/verify_tmvev_run_parity.py COMPARE RUN_A RUN_B
            [--expect-identical] [--precision 6] [--files NAME,NAME]
    python scripts/verify_tmvev_run_parity.py TABLE [DIR ...] [--scope PREFIX]

`COMPARE` prints, for two run folders (or one folder and one queue directory,
in which case the mode/label is matched out of its manifest):

* **cell diffs** — every exported CSV compared after normalising numbers to
  `--precision` decimals, plus the shape (rows/columns) when they differ. This
  is the exactness a caching change has to clear: a store that re-grades a
  vector in its 8th decimal shows up here, not in the verdicts, which stay
  identical while the numbers move. `--expect-identical` turns any diff into a
  non-zero exit, which is how a control run is certified.
* **verdict deltas** — the per-pair grade changes, keyed so a row that moved
  `verified -> borderline` is named rather than counted.
* **category transitions** — rows whose Rev 3.12 `category` differs, the
  symptom the drifting null backdrop produced on 2026-09-24.
* **the bar and the window** — the run null backdrop's p95, the candidate
  window's feed per band, and how many of A's borderline rows B lost.

`TABLE` prints one row per run folder across a directory (the newest
`local_data/` queue dirs by default), scoped to source types matching
`--scope`, so a change can be read across runs instead of pairwise.

Both subcommands read the FLAT pre-2026-09-19 layout too, and both are
read-only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from comparison.mapping_validation import RUN_FILE_LAYOUT  # noqa: E402

EVIDENCE = ('validation/validation_results.csv', 'validation/examinees.csv',
            'validation/pair_summary.csv', 'validation/deep_candidates.csv',
            'gap_fill/gap_fill_dedup.csv', 'gap_fill/gap_fill_levels.csv',
            'expansion/family_candidates.csv', 'expansion/relatives.csv',
            'expansion/source_candidates.csv',
            'expansion/out_map_expansion.csv')
POOLING = ('pooling/pooling_candidates.csv', 'pooling/pooling_pool.csv')


def find(run, rel):
    """A run file in the nested layout, falling back to the flat one."""
    root = Path(run)
    p = root / rel
    if p.exists():
        return p
    flat = root / Path(rel).name
    return flat if flat.exists() else None


def resolve(target, mode=None):
    """A run folder, or the run of a queue dir whose manifest names `mode`."""
    p = Path(target)
    if (p / 'parameters.json').exists():
        return p
    man = p / 'manifest.tsv'
    if man.exists():
        for line in man.read_text().splitlines()[1:]:
            f = line.split('\t')
            if len(f) > 4 and (mode is None or f[2] == mode) and f[4] \
                    and Path(f[4]).exists():
                return Path(f[4])
    sys.exit(f'no run folder in {target}' + (f' for mode {mode}' if mode
                                             else ''))


def norm(path, precision):
    df = pd.read_csv(path, dtype=str).fillna('')

    def fix(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return f'{f:.{precision}f}' if f == f else 'nan'
    return df.apply(lambda c: c.map(fix))


def mode_of(run):
    return json.loads((run / 'parameters.json').read_text('utf-8')).get(
        'validation_mode')


def label(run):
    par = json.loads((run / 'parameters.json').read_text('utf-8'))
    return (f'{str(par.get("target_dataset"))}/{mode_of(run)}'
            f'[{",".join(par.get("query_types") or ())}]')


# ------------------------------------------------------------------ compare
def compare_cells(a, b, files, precision):
    total = 0
    print(f'== cell diffs (numbers at {precision} dp)')
    for rel in files:
        pa, pb = find(a, rel), find(b, rel)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            print(f'  {rel:42} only in '
                  f'{"A" if pb is None else "B"}')
            total += 1
            continue
        A, B = norm(pa, precision), norm(pb, precision)
        if list(A.columns) != list(B.columns):
            print(f'  {rel:42} COLUMNS '
                  f'{sorted(set(A.columns) ^ set(B.columns))}')
            total += 1
            continue
        if A.shape != B.shape:
            print(f'  {rel:42} SHAPE {A.shape} vs {B.shape}')
            total += 1
            continue
        d = int((A != B).sum().sum())
        cols = [c for c in A.columns if (A[c] != B[c]).any()]
        note = f'{len(A)} rows' + (f'; {cols[:4]}' if d else '')
        print(f'  {rel:42} {d:>7}{note:>0}' if not d
              else f'  {rel:42} {d:>7}  {note}')
        total += d
    return total


def keyed(run, rel, key, col):
    p = find(run, rel)
    if p is None:
        return {}
    df = pd.read_csv(p, dtype=str).fillna('')
    if col not in df.columns:
        return {}
    k = df[key].agg('|'.join, axis=1)
    return dict(zip(k, df[col]))


def compare_semantics(a, b):
    print('\n== verdict deltas (validation_results.csv)')
    ka = keyed(a, 'validation/validation_results.csv',
               ['source_bodyId', 'target_bodyId'], 'verdict')
    kb = keyed(b, 'validation/validation_results.csv',
               ['source_bodyId', 'target_bodyId'], 'verdict')
    moved = [(k, ka[k], kb.get(k, '<gone>')) for k in ka
             if kb.get(k, ka[k]) != ka[k]]
    print(f'  {len(ka)} vs {len(kb)} pairs | verdict moved: {len(moved)}')
    for row in moved[:8]:
        print(f'    {row[0]}  {row[1]} -> {row[2]}')
    print(f'  counts A {Counter(ka.values())}')
    print(f'  counts B {Counter(kb.values())}')

    print('\n== category transitions (examinees.csv)')
    ca = keyed(a, 'validation/examinees.csv',
               ['source_bodyId', 'ahead_target_bodyId'], 'category')
    cb = keyed(b, 'validation/examinees.csv',
               ['source_bodyId', 'ahead_target_bodyId'], 'category')
    diff = [(k, ca[k], cb.get(k, '<gone>')) for k in ca
            if cb.get(k, ca[k]) != ca[k]]
    print(f'  {len(ca)} vs {len(cb)} rows | category moved: {len(diff)}')
    for row in diff[:8]:
        print(f'    {row[0]}  {row[1]!r} -> {row[2]!r}')

    print('\n== the null backdrop and the candidate window')
    for tag, run in (('A', a), ('B', b)):
        cal = json.loads(find(run, 'morphology_calibration.json')
                         .read_text('utf-8'))
        ps = pd.read_csv(find(run, 'validation/pair_summary.csv'),
                         dtype=str).fillna('')
        feed = pd.read_csv(find(run, 'validation/deep_candidates.csv'),
                           dtype=str).fillna('') \
            if find(run, 'validation/deep_candidates.csv') else \
            pd.DataFrame(columns=('candidate_source', 'source_bodyId',
                                  'ahead_target_bodyId'))
        tight = feed[feed.candidate_source == 'top_window']
        print(f'  {tag}: p95 {cal.get("track_a_null_bar")} · null sample '
              f'{pd.to_numeric(ps["null_sample"], errors="coerce").sum():.0f}'
              f' · window feed {len(feed)} rows ('
              f'{len(tight)} borderline)')
    fa = _feed_pairs(a)
    fb = _feed_pairs(b)
    if fa and fb:
        lost = {(s, t) for s, t, src in fa if src == 'top_window'} - \
               {(s, t) for s, t, src in fb if src == 'top_window'}
        print(f'  borderline rows in A missing from B: {len(lost)}'
              + (f' e.g. {sorted(lost)[:2]}' if lost else ''))


def _feed_pairs(run):
    p = find(run, 'validation/deep_candidates.csv')
    if p is None:
        return set()
    df = pd.read_csv(p, dtype=str).fillna('')
    return {(r.source_bodyId, r.ahead_target_bodyId, r.candidate_source)
            for r in df.itertuples(index=False)}


# -------------------------------------------------------------------- table
def table(dirs, scope):
    runs = []
    for raw in dirs:
        d = Path(raw)
        p = d if d.is_absolute() else REPO / d
        if (p / 'manifest.tsv').exists():
            for line in (p / 'manifest.tsv').read_text().splitlines()[1:]:
                f = line.split('\t')
                if len(f) > 4 and f[4] and Path(f[4]).exists():
                    runs.append(Path(f[4]))
        elif (p / 'parameters.json').exists():
            runs.append(p)
        else:
            runs.extend(sorted(p.glob('type-map-validation_*')))
    cols = ('mode', 'pairs', 'mapped', 'exam', 'bins', 'relatives',
            'levels', 'null p95', 'scenes', 'wall')
    print(f'{"run":26}' + ''.join(f'{c:>13}' for c in cols))
    for run in sorted(set(runs)):
        if mode_of(run) is None or not (run / 'parameters.json').exists():
            continue
        vr = (pd.read_csv(find(run, 'validation/validation_results.csv'),
                          dtype=str).fillna('')
              if find(run, 'validation/validation_results.csv') else None)
        ex = (pd.read_csv(find(run, 'validation/examinees.csv'),
                          dtype=str).fillna('')
              if find(run, 'validation/examinees.csv') else None)
        if scope and vr is not None and not vr.source_type.str.startswith(
                scope).any():
            continue
        cal = json.loads(find(run, 'morphology_calibration.json').read_text())
        lv, rl = (find(run, 'gap_fill/gap_fill_levels.csv'),
                  find(run, 'expansion/relatives.csv'))
        cells = [
            str(mode_of(run)),
            str(len(vr)) if vr is not None else '-',
            ('' if vr is None else str(int(vr.verdict.isin(
                ['matched', 'verified_strong', 'verified']).sum()))),
            str(len(ex)) if ex is not None else '-',
            ('' if ex is None else ','.join(
                f'{k or "·"}{v}' for k, v in
                sorted(Counter(ex.category).items(), key=lambda x: -x[1]))),
            str(len(pd.read_csv(rl, dtype=str))) if rl else '0',
            str(len(pd.read_csv(lv, dtype=str))) if lv else '0',
            f'{cal.get("track_a_null_bar"):.4f}'
            if cal.get('track_a_null_bar') is not None else '-',
            str(len(list((run / 'visualization').glob('plot-3d_*'))))
            if (run / 'visualization').exists() else '0',
            _wall(run)]
        print(f'{run.name[-25:]:26}'
              + ''.join(f'{c:>13}' for c in cells))


def _wall(run):
    """The queue's recorded wall seconds for this folder, when it has one."""
    for man in sorted(REPO.glob(f'local_data/*/manifest.tsv')):
        for line in man.read_text().splitlines()[1:]:
            f = line.split('\t')
            if len(f) > 6 and f[4] == str(run):
                return f'{int(f[6]) / 60:.1f} min'
    return '-'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compare', help='two runs, cell-by-cell and semantically')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--mode', default=None,
                   help='when a queue dir is given, pick that mode from it')
    c.add_argument('--mode-b', default=None)
    c.add_argument('--precision', type=int, default=6)
    c.add_argument('--files', default=None, help='comma-separated rel paths')
    c.add_argument('--expect-identical', action='store_true')
    t = sub.add_parser('table', help='one row per run folder')
    t.add_argument('dirs', nargs='*',
                   default=['local_data/mapping_validation'])
    t.add_argument('--scope', default=None,
                   help='only runs whose source types start with this')
    args = ap.parse_args(argv)
    if args.cmd == 'table':
        table(args.dirs or ['local_data/mapping_validation'], args.scope)
        return 0
    a = resolve(args.a, args.mode)
    b = resolve(args.b, args.mode_b or args.mode)
    print(f'A {label(a)}\nB {label(b)}')
    files = args.files.split(',') if args.files else \
        tuple(EVIDENCE + (POOLING if {mode_of(a), mode_of(b)} == {'pooling'}
                          else ()))
    total = compare_cells(a, b, files, args.precision)
    compare_semantics(a, b)
    if args.expect_identical:
        print(f'\n== {"IDENTICAL" if not total else f"DIFFERS ({total})"} ==')
        return 1 if total else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
