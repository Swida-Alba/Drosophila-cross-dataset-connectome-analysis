#!/usr/bin/env python3
"""Global same-name pair verification across datasets (Phase 5 asset).

For every type name native to BOTH datasets of an ordered pair (A->B),
report what the mapper's shared validity core says:

* confirm_unique    - mapped/bridged with a unique equivalence key equal to
                      the name itself (the identity is evidence-backed)
* among_candidates  - the identity is one of several targets
* conflict          - the mapper refuses (unscoped union / vote conflict)
* contradict_unique - unique evidence names a DIFFERENT counterpart
* silent            - no relation at all

plus a decision-level evidence census (``get_mapping_decision``):
confirmed / contradicted / none.  Reference numbers for MCNS v1.0 /
FAFB v783 / BANC v888 (plan-cross-dataset-query-resolution-samename-
taxonomy §1.4, rev 2): 37,594 directed pairs; 1,549 shadowed and 28
overridden pairs were fixed by the Phase 1b union-ranking change, so the
resolver-level census now shows confirm_unique for the formerly shadowed
pairs and conflict for the formerly overridden ones.

Usage:
    python3 scripts/VerifySameNamePairs.py [--workspace PATH] [--out DIR]

Writes ``same_name_audit.csv`` and prints the per-direction summary.
Requires the local dataset tables (datasets/<release>/) and the cached
mapper snapshot; skips gracefully when they are missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

UNTYPED = {'nan', 'none', 'unknown', ''}


def type_names(df, column: str = 'type') -> set:
    series = df[column].dropna().astype(str).str.strip()
    series = series[~series.str.lower().isin(UNTYPED) & (series != '')]
    return set(series)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=str(REPO_ROOT))
    parser.add_argument('--out', default=None,
                        help='Optional directory for same_name_audit.csv '
                             '(default: summary only; avoid writing under '
                             'datasets/ — it is mapper-fingerprinted)')
    args = parser.parse_args()

    import pandas as pd
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper
    from comparison.type_resolver import (
        MapperSnapshot, STATUS_BRIDGED, STATUS_CONFLICT, STATUS_MAPPED,
        STATUS_UNMAPPED, resolve_valid_targets,
    )

    tables = {
        'male-cns:v1.0': REPO_ROOT / 'datasets' / 'male-cns_v1_0'
        / 'male-cns_v1_0_allneurons_neuron_df.csv',
        'flywire_FAFB_v783': REPO_ROOT / 'datasets' / 'flywire_FAFB_v783'
        / 'flywire_FAFB_v783_allneurons_neuron_df.csv',
        'banc_v888': REPO_ROOT / 'datasets' / 'banc_v888'
        / 'banc_v888_allneurons_neuron_df.csv',
    }
    missing = [str(p) for p in tables.values() if not p.exists()]
    if missing:
        print('Local dataset tables missing, skipping:', '; '.join(missing))
        return 0

    sets = {}
    frames = {}
    for ds, path in tables.items():
        frames[ds] = pd.read_csv(path, low_memory=False)
        sets[ds] = type_names(frames[ds])
        print(f'{ds}: {len(sets[ds])} native types')

    mapper = CrossDatasetTypeMapper(workspace_path=args.workspace)
    if not mapper.load():
        print('Type mapper failed to load, skipping.')
        return 0
    snap = MapperSnapshot(mapper)
    alias_cache, bridge_cache = {}, {}

    rows = []
    started = time.time()
    decisions = {}
    for src in sets:
        for dst in sets:
            if src == dst:
                continue
            common = sorted(sets[src] & sets[dst])
            print(f'{src} -> {dst}: {len(common)} same-name pairs', flush=True)
            for name in common:
                try:
                    res = resolve_valid_targets(
                        mapper, name, src, dst, snapshot=snap,
                        alias_cache=alias_cache, bridge_cache=bridge_cache)
                except Exception as exc:  # noqa: BLE001
                    rows.append((src, dst, name, 'error', 'error',
                                 tuple(), None, ''))
                    continue
                eq = res.equivalence_key
                targets = tuple(sorted(str(t) for t in (res.target_types or ())))
                if (res.status in (STATUS_MAPPED, STATUS_BRIDGED)
                        and res.equivalence_key):
                    category = ('confirm_unique'
                                if str(eq) == name else 'contradict_unique')
                elif res.status == STATUS_CONFLICT:
                    category = 'conflict'
                elif res.status == STATUS_UNMAPPED:
                    category = 'silent'
                elif targets:
                    category = ('among_candidates' if name in targets
                                else 'excluded')
                else:
                    category = 'silent'
                # decision-level evidence census (Phase 1 annotation basis)
                try:
                    dec = decisions.setdefault(
                        (src, dst, name),
                        mapper.get_mapping_decision(name, src, dst))
                except Exception:
                    dec = {}
                d_target = dec.get('target_type')
                if dec.get('status') in (STATUS_MAPPED, STATUS_BRIDGED) \
                        and d_target:
                    evidence = ('confirmed' if str(d_target) == name
                                else 'contradicted')
                else:
                    evidence = 'none'
                rows.append((src, dst, name, res.status, category, targets,
                             eq, evidence))

    df = pd.DataFrame(rows, columns=[
        'src', 'dst', 'type', 'status', 'category', 'targets',
        'equivalence_key', 'decision_evidence'])
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'same_name_audit.csv'
        df.to_csv(out_path, index=False)
        print(f'\nWrote {len(df)} rows to {out_path} '
              f'({time.time() - started:.0f}s)')
    else:
        print(f'\n{len(df)} pairs audited ({time.time() - started:.0f}s); '
              f'pass --out DIR to write same_name_audit.csv')
    print('\nresolver categories per direction:')
    print(df.groupby(['src', 'dst', 'category']).size().unstack(
        fill_value=0).to_string())
    print('\ndecision evidence census:')
    print(df.decision_evidence.value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
