"""Applied-folder reconcile: naming grammar, _skipped markers, note file.

Plan §4. The reconcile step is non-destructive: materialized folders are
renamed to the applied grammar, pruned requested thresholds get a marker
folder with a README, and each dataset gets APPLIED_THRESHOLDS.md.
"""

import json
import os

from comparison.comparison_analyzer import ComparisonAnalyzer
from comparison.comparison_parameters import ComparisonParameters


class _FakeParams:
    """Lightweight parameters double. The folder-name grammar DELEGATES to
    the real ``ComparisonParameters`` so the tests exercise production
    naming, not a re-implemented copy."""

    def __init__(self, tmp_path, datasets, thresholds_by_ds):
        self._datasets = datasets
        self._thresholds_by_ds = thresholds_by_ds
        self.thresholds = sorted(
            {t for ts in thresholds_by_ds.values() for t in ts})
        self.full_output_path = str(tmp_path)
        self.path_mode = 'all'
        self.pathfinding = 'StrongestFirst'
        self._applied_folder_lookup = {}

    def get_dataset_names(self):
        return list(self._datasets)

    def get_thresholds_for_dataset(self, ds):
        return list(self._thresholds_by_ds.get(ds, []))

    @property
    def dataset_data_path(self):
        return os.path.join(self.full_output_path, 'dataset_data')

    def _sanitize_name(self, name):
        return name.replace(':', '_').replace('.', '_')

    def applied_folder_name(self, applied, requested_thresholds,
                            is_floor=False):
        return ComparisonParameters.applied_folder_name(
            self, applied, requested_thresholds, is_floor=is_floor)

    def applied_folder_name_candidates(self, value):
        return ComparisonParameters.applied_folder_name_candidates(
            self, value)

    def skipped_folder_name(self, requested):
        return ComparisonParameters.skipped_folder_name(self, requested)

    def set_applied_folder_lookup(self, dataset, threshold, folder_name):
        self._applied_folder_lookup[(dataset, int(threshold))] = folder_name

    def get_dataset_output_path(self, dataset, threshold):
        safe = self._sanitize_name(dataset)
        name = self._applied_folder_lookup.get(
            (dataset, int(threshold)), f'minsyn_{int(threshold)}')
        return os.path.join(self.dataset_data_path, safe, name)

    def get_skipped_output_path(self, dataset, requested):
        return os.path.join(
            self.dataset_data_path, self._sanitize_name(dataset),
            f'minsyn_{int(requested)}_skipped')


def _make(tmp_path, meta, thresholds_by_ds, name='male-cns:v1.0'):
    a = object.__new__(ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda msg, *arg, **kw: None
    a.parameters = _FakeParams(tmp_path, [name], thresholds_by_ds)
    a._path_run_meta = dict(meta)
    return a


def _write_folder(base, name, applied, requested=None):
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'all_attributes.json'), 'w') as f:
        json.dump({'applied_threshold': applied,
                   'min_synapse_num': applied if requested is None
                   else requested,
                   'requested_threshold': requested if requested is not None
                   else applied}, f)
    return folder


def test_reconcile_renames_and_marks(tmp_path):
    ds = 'male-cns:v1.0'
    base = os.path.join(
        tmp_path, 'dataset_data', 'male-cns_v1_0')
    # Pre-existing plain folders for applied 19 (as requested 3) and 30.
    _write_folder(base, 'minsyn_3', 19)
    _write_folder(base, 'minsyn_30', 30)
    meta = {
        (ds, 3): {'skipped': True, 'applied_folder': 19,
                  'tau': 19.0, 'strongest_first_tau': 19.0,
                  'paths_complete': False, 'duplicate_of': 19},
        (ds, 5): {'skipped': False, 'applied_folder': 19,
                  'tau': 19.0, 'strongest_first_tau': 19.0,
                  'strongest_first_budget_bitten': True,
                  'tau_canonical': 19, 'strongest_dropped_bottleneck': 18.0,
                  'paths_complete': False},
        (ds, 10): {'skipped': True, 'applied_folder': 19,
                   'tau': 19.0, 'strongest_first_tau': 19.0,
                   'paths_complete': False, 'duplicate_of': 19},
        (ds, 30): {'skipped': False, 'applied_folder': 30,
                   'tau': 30.0, 'strongest_first_tau': 30.0,
                   'paths_complete': True},
    }
    a = _make(tmp_path, meta, {ds: [3, 5, 10, 30]})
    a._reconcile_applied_folders()

    # Data folder renamed to the applied grammar.
    assert os.path.isdir(os.path.join(base, 'minsyn_19_applied_floor'))
    assert not os.path.isdir(os.path.join(base, 'minsyn_3'))
    # requested 30 == applied 30 and not a floor -> bare name.
    assert os.path.isdir(os.path.join(base, 'minsyn_30'))

    # Skipped requested thresholds get markers with a README.
    for t in (3, 5, 10):
        marker = os.path.join(base, f'minsyn_{t}_skipped')
        assert os.path.isdir(marker), t
        readme = open(os.path.join(marker, 'README.txt')).read()
        assert 'Applied threshold' in readme
        assert '19' in readme

    # Note file lists data folders and markers.
    note = open(os.path.join(base, 'APPLIED_THRESHOLDS.md')).read()
    assert 'minsyn_19_applied_floor' in note
    assert 'minsyn_3_skipped' in note

    # Lookup routes requested thresholds to the materialized folder.
    assert a.parameters.get_dataset_output_path(ds, 3).endswith(
        'minsyn_19_applied_floor')
    assert a.parameters.get_dataset_output_path(ds, 30).endswith('minsyn_30')


def test_reconcile_exact_threshold_has_no_marker(tmp_path):
    ds = 'male-cns:v1.0'
    base = os.path.join(tmp_path, 'dataset_data', 'male-cns_v1_0')
    _write_folder(base, 'minsyn_30', 30)
    meta = {
        (ds, 30): {'skipped': False, 'applied_folder': 30,
                   'tau': 30.0, 'strongest_first_tau': 30.0,
                   'paths_complete': True},
    }
    a = _make(tmp_path, meta, {ds: [30]})
    a._reconcile_applied_folders()
    assert not os.path.isdir(os.path.join(base, 'minsyn_30_skipped'))
    assert os.path.isdir(os.path.join(base, 'minsyn_30'))


def test_reconcile_collapses_duplicate_data_folders(tmp_path):
    """Two data folders recording the SAME applied value collapse to one
    (the exact run kept) plus a marker for the redundant requested one."""
    ds = 'male-cns:v1.0'
    base = os.path.join(tmp_path, 'dataset_data', 'male-cns_v1_0')
    # `minsyn_5` (requested 5, applied 19) and `minsyn_19` (requested 19,
    # applied 19) both record applied 19.
    _write_folder(base, 'minsyn_5', 19, requested=5)
    _write_folder(base, 'minsyn_19', 19, requested=19)
    meta = {
        (ds, 5): {'skipped': True, 'applied_folder': 19, 'tau': 19.0,
                  'strongest_first_tau': 19.0, 'tau_canonical': 19,
                  'strongest_dropped_bottleneck': 18.0,
                  'strongest_first_budget_bitten': True,
                  'paths_complete': False, 'duplicate_of': 19},
        (ds, 19): {'skipped': False, 'applied_folder': 19, 'tau': 19.0,
                   'strongest_first_tau': 19.0, 'tau_canonical': 19,
                   'paths_complete': True},
    }
    a = _make(tmp_path, meta, {ds: [5, 19]})
    a._reconcile_applied_folders()

    names = sorted(os.listdir(base))
    data = [n for n in names if n.startswith('minsyn_')
            and not n.endswith('_skipped')]
    assert data == ['minsyn_19_applied_floor'], names
    assert 'minsyn_5_skipped' in names
    # requested 19 coincides with the floor it aliases to -> no marker.
    assert 'minsyn_19_skipped' not in names


def test_reconcile_normalizes_reenumerated_run_to_applied_folder(tmp_path):
    """A re-enumerated requested threshold whose set collapses onto an
    existing applied folder is normalized to skipped/duplicate_of and gets a
    marker, so the note, provenance row and disk agree."""
    ds = 'male-cns:v1.0'
    base = os.path.join(tmp_path, 'dataset_data', 'male-cns_v1_0')
    # Representative folder for applied 19, run requested 19.
    _write_folder(base, 'minsyn_19', 19, requested=19)
    # Re-enumerated run requested 5 that landed on applied 19.
    _write_folder(base, 'minsyn_5', 19, requested=5)
    meta = {
        (ds, 5): {'skipped': False, 'applied_folder': 5, 'tau': 19.0,
                  'strongest_first_tau': 19.0, 'tau_canonical': 19,
                  'strongest_dropped_bottleneck': 18.0,
                  'strongest_first_budget_bitten': True,
                  'paths_complete': False},
        (ds, 19): {'skipped': False, 'applied_folder': 19, 'tau': 19.0,
                   'strongest_first_tau': 19.0, 'paths_complete': True},
    }
    a = _make(tmp_path, meta, {ds: [5, 19]})
    a._reconcile_applied_folders()

    # req 19 is exact (own folder), req 5 aliases to 19.
    assert a._path_run_meta[(ds, 5)]['skipped'] is True
    assert a._path_run_meta[(ds, 5)]['duplicate_of'] == 19
    assert a._path_run_meta[(ds, 5)]['applied_folder'] == 19
    assert a._path_run_meta[(ds, 19)]['skipped'] is False
    assert os.path.isdir(os.path.join(base, 'minsyn_5_skipped'))
    assert not os.path.isdir(os.path.join(base, 'minsyn_19_skipped'))
    row5 = a._path_provenance_row(ds, 5)
    assert row5['applied_threshold'] == 19
    # Note reflects the alias (marker) and references the real folder.
    note = open(os.path.join(base, 'APPLIED_THRESHOLDS.md')).read()
    assert 'minsyn_5_skipped' in note
    assert 'data folders on disk' in note


def test_real_grammar_names():
    """The production grammar: a data folder is never a bare `minsyn_N`."""
    from comparison.comparison_parameters import ComparisonParameters
    p = object.__new__(ComparisonParameters)
    # genuine requested run -> bare name
    assert p.applied_folder_name(30, [3, 5, 10, 30]) == 'minsyn_30'
    # collapse floor -> always _applied_floor, even if the value coincides
    # with a requested level (no _equal_ variant).
    assert p.applied_folder_name(
        19, [3, 5, 10, 30], is_floor=True) == 'minsyn_19_applied_floor'
    assert p.applied_folder_name(
        10, [5, 8, 10, 15], is_floor=True) == 'minsyn_10_applied_floor'
    assert p.skipped_folder_name(5) == 'minsyn_5_skipped'
    cands = p.applied_folder_name_candidates(30)
    assert 'minsyn_30' in cands and 'minsyn_30_applied_floor' in cands


def test_resolve_folder_disk_aware_without_meta(tmp_path):
    """A later analysis (fresh process, empty _path_run_meta) still finds the
    new-grammar folders by reading each folder's requested_threshold."""
    ds = 'male-cns:v1.0'
    base = os.path.join(tmp_path, 'dataset_data', 'male-cns_v1_0')
    # Folders as reconcile would leave them: applied 19 (requested 5) and
    # applied 30 (requested 30, equal).
    _write_folder(base, 'minsyn_5', 19, requested=5)
    # rename to grammar name to mimic post-reconcile disk
    os.rename(os.path.join(base, 'minsyn_5'),
              os.path.join(base, 'minsyn_19_applied_floor'))
    _write_folder(base, 'minsyn_30', 30, requested=30)

    a = object.__new__(ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda *x, **k: None
    a.parameters = _FakeParams(tmp_path, [ds], {ds: [5, 30]})
    a._path_run_meta = {}  # fresh process: no in-memory provenance

    assert a._resolve_dataset_output_path(ds, 5).endswith(
        'minsyn_19_applied_floor')
    assert a._resolve_dataset_output_path(ds, 30).endswith('minsyn_30')


def test_coincident_requested_floor_is_skipped_silently(tmp_path):
    """User example: requested 5,8,10,15; the 5-run's applied floor is 10.
    10 coincides with the floor it aliases to -> served by
    `minsyn_10_applied_floor`, NO `minsyn_10_skipped`, and no bare
    `minsyn_10`. 5 and 8 alias to a different level -> markers."""
    ds = 'ds'
    base = os.path.join(tmp_path, 'dataset_data', 'ds')
    _write_folder(base, 'minsyn_10', 10, requested=5)   # floor from t=5
    _write_folder(base, 'minsyn_15', 15, requested=15)  # genuine run
    meta = {
        (ds, 5): {'skipped': True, 'applied_folder': 10, 'tau': 10.0,
                  'strongest_first_tau': 10.0, 'tau_canonical': 10,
                  'paths_complete': False, 'duplicate_of': 10},
        (ds, 8): {'skipped': True, 'applied_folder': 10, 'tau': 10.0,
                  'strongest_first_tau': 10.0, 'tau_canonical': 10,
                  'paths_complete': False, 'duplicate_of': 10},
        (ds, 10): {'skipped': True, 'applied_folder': 10, 'tau': 10.0,
                   'strongest_first_tau': 10.0, 'tau_canonical': 10,
                   'paths_complete': False, 'duplicate_of': 10},
        (ds, 15): {'skipped': False, 'applied_folder': 15, 'tau': 15.0,
                   'strongest_first_tau': 15.0, 'paths_complete': True},
    }
    a = _make(tmp_path, meta, {ds: [5, 8, 10, 15]}, name=ds)
    a._reconcile_applied_folders()

    names = sorted(os.listdir(base))
    data = [n for n in names if n.startswith('minsyn_')
            and not n.endswith('_skipped')]
    assert data == ['minsyn_10_applied_floor', 'minsyn_15'], names
    assert 'minsyn_10_skipped' not in names   # silent
    assert 'minsyn_5_skipped' in names
    assert 'minsyn_8_skipped' in names
    # requested 10 is not "skipped" (served by its floor folder).
    assert a._path_run_meta[(ds, 10)]['skipped'] is False
    assert a._path_run_meta[(ds, 5)]['skipped'] is True
    # resolution: 5 and 10 both map to the floor folder.
    assert a._resolve_dataset_output_path(ds, 5).endswith(
        'minsyn_10_applied_floor')
    assert a._resolve_dataset_output_path(ds, 10).endswith(
        'minsyn_10_applied_floor')
    assert a._resolve_dataset_output_path(ds, 15).endswith('minsyn_15')
