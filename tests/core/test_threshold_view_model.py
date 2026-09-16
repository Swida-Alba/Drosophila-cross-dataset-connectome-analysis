"""Threshold view-model + comparability report (plan §2, §5A).

Covers the shared requested->applied accessors used by the folder export,
the report and the used-data writers, plus the comparability verdict that
warns when budget pruning leaves too few like-for-like thresholds.
"""

from comparison.comparison_analyzer import ComparisonAnalyzer


class _FakeParams:
    def __init__(self, datasets, thresholds_by_ds, thresholds=None):
        self._datasets = datasets
        self._thresholds_by_ds = thresholds_by_ds
        self.thresholds = thresholds if thresholds is not None else sorted(
            {t for ts in thresholds_by_ds.values() for t in ts})
        self.path_mode = 'all'
        self.pathfinding = 'StrongestFirst'

    def get_dataset_names(self):
        return list(self._datasets)

    def get_thresholds_for_dataset(self, ds):
        return list(self._thresholds_by_ds.get(ds, []))

    def get_threshold_queries(self):
        # Standard mode: one uniform row per requested threshold.
        rows = []
        for t in self.thresholds:
            rows.append({
                'id': f'threshold_{t}', 'label': f'N={t}',
                'thresholds': {ds: t for ds in self._datasets},
            })
        return rows


def _analyzer(datasets, thresholds_by_ds, meta, thresholds=None):
    a = object.__new__(ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda msg, *arg, **kw: None
    a.parameters = _FakeParams(datasets, thresholds_by_ds, thresholds)
    a._path_run_meta = dict(meta)
    return a


def test_get_applied_folder_prefers_orchestrator_folder():
    a = _analyzer(['banc_v888'], {'banc_v888': [3, 30]}, {
        ('banc_v888', 3): {'skipped': True, 'applied_folder': 6},
        ('banc_v888', 30): {'skipped': False, 'applied_folder': 30},
    })
    assert a.get_applied_folder('banc_v888', 3) == 6
    assert a.get_applied_folder('banc_v888', 30) == 30


def test_get_applied_thresholds_dedups():
    a = _analyzer(['male-cns:v1.0'], {'male-cns:v1.0': [3, 5, 10, 30]}, {
        ('male-cns:v1.0', 3): {'skipped': True, 'applied_folder': 19},
        ('male-cns:v1.0', 5): {'skipped': False, 'applied_folder': 19},
        ('male-cns:v1.0', 10): {'skipped': True, 'applied_folder': 19},
        ('male-cns:v1.0', 30): {'skipped': False, 'applied_folder': 30},
    })
    assert a.get_applied_thresholds('male-cns:v1.0') == [19, 30]


def test_threshold_view_marks_exact_and_aliased():
    a = _analyzer(['banc_v888'], {'banc_v888': [3, 30]}, {
        ('banc_v888', 3): {'skipped': True, 'applied_folder': 6,
                           'paths_complete': False},
        ('banc_v888', 30): {'skipped': False, 'applied_folder': 30,
                            'paths_complete': True},
    })
    aliased = a.get_threshold_view('banc_v888', 3)
    assert aliased['status'] == 'aliased' and aliased['is_exact'] is False
    ex = a.get_threshold_view('banc_v888', 30)
    assert ex['status'] == 'applied' and ex['is_exact'] is True


def test_comparability_warns_when_only_one_common_threshold():
    datasets = ['male-cns:v1.0', 'banc_v888', 'flywire_FAFB_v783']
    by_ds = {
        'male-cns:v1.0': [3, 10, 30],
        'banc_v888': [3, 5, 10, 30],
        'flywire_FAFB_v783': [3, 10, 30],
    }
    meta = {}
    applied = {
        'male-cns:v1.0': {3: 19, 10: 19, 30: 30},
        'banc_v888': {3: 6, 5: 6, 10: 10, 30: 30},
        'flywire_FAFB_v783': {3: 17, 10: 17, 30: 30},
    }
    skipped = {
        'male-cns:v1.0': {3: True, 10: True, 30: False},
        'banc_v888': {3: True, 5: True, 10: False, 30: False},
        'flywire_FAFB_v783': {3: True, 10: True, 30: False},
    }
    for ds, ts in by_ds.items():
        for t in ts:
            meta[(ds, t)] = {
                'skipped': skipped[ds][t],
                'applied_folder': applied[ds][t],
                'paths_complete': not skipped[ds][t],
            }
    a = _analyzer(datasets, by_ds, meta, thresholds=[3, 5, 10, 30])
    report = a.comparability_report()
    assert report['common_materialized'] == [30]
    assert report['warning_level'] == 'warning'
    assert any('30' in w for w in report['warnings'])
    assert report['materialized_thresholds']['male-cns:v1.0'] == [19, 30]


def test_comparability_critical_when_disjoint():
    datasets = ['a', 'b']
    by_ds = {'a': [3], 'b': [3]}
    meta = {
        ('a', 3): {'skipped': False, 'applied_folder': 5},
        ('b', 3): {'skipped': False, 'applied_folder': 7},
    }
    a = _analyzer(datasets, by_ds, meta, thresholds=[3])
    report = a.comparability_report()
    assert report['common_materialized'] == []
    assert report['warning_level'] == 'critical'


def test_comparability_no_warning_when_aligned():
    datasets = ['a', 'b']
    by_ds = {'a': [3, 5], 'b': [3, 5]}
    meta = {}
    for ds in datasets:
        for t in (3, 5):
            meta[(ds, t)] = {'skipped': False, 'applied_folder': t}
    a = _analyzer(datasets, by_ds, meta, thresholds=[3, 5])
    report = a.comparability_report()
    assert report['common_materialized'] == [3, 5]
    assert report['warning_level'] == 'none'


def test_threshold_queries_view_marks_fully_aliased():
    datasets = ['a', 'b']
    by_ds = {'a': [3, 30], 'b': [3, 30]}
    meta = {
        ('a', 3): {'skipped': True, 'applied_folder': 30},
        ('b', 3): {'skipped': True, 'applied_folder': 30},
        ('a', 30): {'skipped': False, 'applied_folder': 30},
        ('b', 30): {'skipped': False, 'applied_folder': 30},
    }
    a = _analyzer(datasets, by_ds, meta, thresholds=[3, 30])
    views = {v['id']: v for v in a.get_threshold_queries_view()}
    assert views['threshold_3']['is_fully_aliased'] is True
    assert views['threshold_30']['is_fully_aliased'] is False
    assert views['threshold_3']['applied_thresholds'] == {'a': 30, 'b': 30}
