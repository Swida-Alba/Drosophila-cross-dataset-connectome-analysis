"""Real-data verification of the union type-coverage pass.

Re-runs the 2026-09-25 cross-dataset comparison (aMe12/aMe26/aMe9 ->
PPL101/PPL103 across male-cns:v1.0 + flywire_FAFB_v783 + banc_v888)
OFFLINE into a fresh folder from the original run's parameters.json, then
asserts the APL verdicts that motivated the feature:

- BANC: APL present (its aMe9 -> APL edge is weight 7 >= 3).
- MCNS + FAFB: APL resolves, neurons exist, but every aMe -> APL edge is
  weight 1-2 < threshold 3  ->  ``below_threshold`` with the max weight
  in the detail — NOT a bare absence.

Also checks the additive surfaces (status columns, txt section, HTML
annotations) and that the compared edge rows are unchanged.

Usage:
    python3 scripts/preview/type_coverage_realdata_check.py \
        [--source /path/to/original_run] [--out /path/to/fresh_folder]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

import pandas as pd  # noqa: E402

from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from comparison.comparison_parameters import ComparisonParameters  # noqa: E402

DEFAULT_SOURCE = Path(
    '/Users/apple/Local/connection_data/DROCAT_data/'
    'cross-dataset_aMe12_etc_to_PPL101_etc_MFB_20260925_191909')

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v888'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--out', default=str(
        REPO / 'local_data' / 'type_coverage_realdata_check'))
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    # The analyzer nests a timestamped cross-dataset_* run folder inside
    # output_folder; reuse the newest one instead of re-running.
    existing = sorted(out.glob('cross-dataset_*'), key=lambda p: p.stat().st_mtime) \
        if out.exists() else []
    if existing and (existing[-1] / 'comparison_report.html').exists():
        run_root = existing[-1]
        print(f'[[reusing exported run: {run_root}]]')
    else:
        payload = json.loads((source / 'parameters.json').read_text())
        params = ComparisonParameters.from_dict(payload)
        params.output_folder = str(out)
        params.cache_only = True
        params.force_API_fetching = False
        params.open_browser = False
        params.verbose = True

        analyzer = ComparisonAnalyzer(params, verbose=True)
        analyzer.run_comparison(skip_existing=True)
        analyzer.export_results()
        run_root = max(out.glob('cross-dataset_*'),
                       key=lambda p: p.stat().st_mtime)

    failures = []

    def check(name, condition, detail=''):
        status = 'PASS' if condition else 'FAIL'
        print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
        if not condition:
            failures.append(name)

    # ------------------------------------------------------------------
    # 1. type_resolution_union.csv: APL verdicts
    # ------------------------------------------------------------------
    print('\n== type_resolution_union.csv ==')
    union_path = run_root / 'comparison_results' / 'type_resolution_union.csv'
    if not union_path.exists():
        print('  [FAIL] type_resolution_union.csv missing')
        return 1
    union = pd.read_csv(union_path)
    # Applied threshold per (query, dataset) for the detail-text check:
    # the diagnosis reports "< applied threshold", which follows the
    # query's own threshold map (and its applied-floor grammar).
    comb = pd.read_csv(run_root / 'comparison_results' /
                       'threshold_combinations.csv')
    applied = {
        (row['query_id'], row['dataset']): int(row['applied_threshold'])
        for _, row in comb.iterrows()
    }
    for query_id in union['query_id'].unique():
        q = union[union['query_id'] == query_id]
        apl = q[q['type'] == 'APL'].set_index('dataset')
        if apl.empty:
            # The union only contains types that APPEARED in some dataset
            # for that query; at high thresholds APL appeared nowhere.
            print(f'  [SKIP] {query_id}: APL not in any dataset union '
                  '(appeared in no dataset at this threshold)')
            continue
        check(f'{query_id}: APL present in BANC',
              apl.loc[BANC, 'resolution_status'] == 'present')
        for ds in (MCNS, FAFB):
            entry = apl.loc[ds]
            detail = str(entry.get('detail', ''))
            check(
                f'{query_id}: APL below_threshold in {ds}',
                entry['resolution_status'] == 'below_threshold',
                detail)
            expected = applied.get((query_id, ds))
            check(
                f'{query_id}: APL detail names the weak source-side leg '
                f'in {ds}',
                detail == (f'max edge weight from path sources 2 '
                           f'< threshold {expected}'),
                f'expected applied threshold {expected}; detail: {detail}')

    # ------------------------------------------------------------------
    # 2. Edge presence matrix: status columns on the APL rows
    # ------------------------------------------------------------------
    print('\n== edge_presence_matrix_query_threshold_3.csv ==')
    pm_path = (run_root / 'comparison_results' /
               'edge_presence_matrix_query_threshold_3.csv')
    check('presence matrix exists', pm_path.exists())
    if pm_path.exists():
        pm = pd.read_csv(pm_path)
        safe = {ds: ds.replace(':', '_').replace('.', '_')
                for ds in (MCNS, FAFB, BANC)}
        row = pm[pm['edge_key'] == 'APL -> PPL101']
        check('APL -> PPL101 row exists', not row.empty)
        if not row.empty:
            row = row.iloc[0]
            check('BANC source_status present',
                  row[f"source_status_{safe[BANC]}"] == 'present')
            check('MCNS source_status below_threshold',
                  row[f"source_status_{safe[MCNS]}"] == 'below_threshold')
            check('FAFB source_status below_threshold',
                  row[f"source_status_{safe[FAFB]}"] == 'below_threshold')

    # ------------------------------------------------------------------
    # 3. Comparison rows unchanged vs the original run (additive only)
    # ------------------------------------------------------------------
    print('\n== additive-only check vs original run ==')
    old = pd.read_csv(source / 'comparison_results' /
                      'unified_edge_comparison.csv')
    new = pd.read_csv(run_root / 'comparison_results' /
                      'unified_edge_comparison.csv')
    check(f'row count unchanged ({len(old)} -> {len(new)})',
          len(old) == len(new))
    if len(old) == len(new):
        shared = [c for c in old.columns if c in new.columns]
        changed = []
        for col in shared:
            a = old[col].fillna('__NA__').astype(str)
            b = new[col].fillna('__NA__').astype(str)
            if not a.equals(b):
                changed.append(col)
        check('all original columns byte-stable', not changed,
              f'changed: {changed[:5]}')
        new_only = [c for c in new.columns if c not in old.columns]
        expected_new = {
            *(f'source_status_{s}' for s in safe.values()),
            *(f'target_status_{s}' for s in safe.values()),
        }
        check('new columns are exactly the status columns',
              set(new_only) == expected_new, f'got {sorted(new_only)}')

    # ------------------------------------------------------------------
    # 4. Report surfaces
    # ------------------------------------------------------------------
    print('\n== report surfaces ==')
    txt = (run_root / 'comparison_report.txt').read_text(encoding='utf-8')
    check('txt has TYPE COVERAGE section',
          'TYPE COVERAGE (UNION RESOLUTION)' in txt)
    check('txt names APL below threshold', 'APL:' in txt and
          'below threshold' in txt)

    html = (run_root / 'comparison_report.html').read_text(encoding='utf-8')
    check('html hover names APL below threshold',
          'APL: below threshold' in html)
    check('html has Type coverage card', 'Type coverage' in html)

    print('\n' + '=' * 60)
    if failures:
        print(f'{len(failures)} CHECK(S) FAILED:')
        for name in failures:
            print(f'  - {name}')
        return 1
    print('ALL CHECKS PASSED')
    print(f'Results in: {run_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
