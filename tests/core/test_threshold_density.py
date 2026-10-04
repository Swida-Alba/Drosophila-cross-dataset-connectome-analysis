"""Tests for the auto threshold-density alignment (plan §6).

Pure curve math (no datasets) plus the persisted-array round trip and the
combination-row wiring. The real-data guards (§6 tests 4/5/5b) are marked
skip-by-default so the suite runs without the local caches.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from comparison.threshold_density import (  # noqa: E402
    align_horizontal,
    align_vertical,
    dataset_window,
    density_curve,
    density_denominator,
    normalized_edge_density,
    vertical_rows,
)


# --- 1. Curve monotonicity + exactness ------------------------------------

def test_curve_counts_match_bruteforce_and_are_monotone():
    rng = np.random.default_rng(7)
    bottlenecks = rng.integers(3, 40, size=500)
    edges = rng.integers(1, 60, size=2000)
    grid = list(range(3, 41))
    curve = density_curve(bottlenecks, edges, grid)

    assert np.all(np.diff(curve['path_count']) <= 0)
    assert np.all(np.diff(curve['edge_count']) <= 0)
    for i, t in enumerate(grid):
        assert curve['path_count'][i] == int((bottlenecks >= t).sum())
        assert curve['edge_count'][i] == int((edges >= t).sum())


# --- 3. Type-level caveat guard (never min_weight) -------------------------
# The curve is built from the raw arrays; a type-level SUM value would be
# orders of magnitude larger. Guard the units domain directly (test 4).

def test_units_guard_per_connection_not_type_sum():
    rng = np.random.default_rng(3)
    per_connection = rng.integers(1, 2592, size=1000)  # max ~2,591
    curve = density_curve([], per_connection, [1, 10, 100])
    assert int(per_connection.max()) < 10_000
    assert curve['edge_count'][0] == 1000


# --- 2/5. Measured ceiling & BANC-like empty path curve -------------------

def test_window_uses_measured_ceiling_and_flags_high_threshold():
    # BANC-like: 0 paths but plenty of edges.
    bns = np.array([], dtype=float)
    edges = np.array([4, 5, 5, 30, 37], dtype=int)
    meta = {'applied': 4, 'w_start': 4, 'w_star_measured': None,
            'strongest_retained_bottleneck': 37.0,
            'n_nodes': 100, 'n_annotated_nodes': 80, 'n_edges': 5}
    lo, hi = dataset_window(meta)
    assert lo == 4
    assert hi is None
    curve = density_curve(bns, edges, [4, 30, 37])
    assert list(curve['path_count']) == [0, 0, 0]
    assert list(curve['edge_count']) == [5, 2, 1]


def test_window_ceiling_is_max_bottleneck():
    meta = {'applied': 15, 'w_star_measured': 29.0}
    assert dataset_window(meta) == (15, 29)
    # w_start is raised to the physical floor of 3.
    assert dataset_window({'applied': 2, 'w_star_measured': 2.0}) == (3, 2)


# --- 7. Vertical grid formula ---------------------------------------------

def test_vertical_ladder_dedup_and_endpoints():
    windows = {'a': (15, 37), 'b': (15, 37)}
    ladder = align_vertical(windows, K=5)
    # endpoints retained; all inside the window; de-duplicated integers.
    assert ladder[0] == 15
    assert ladder[-1] == 37
    assert all(15 <= t <= 37 for t in ladder)
    assert len(ladder) == len(set(ladder))
    assert set(ladder) == {15, 16, 18, 20, 26, 37}


def test_vertical_degenerate_and_empty():
    assert align_vertical({'a': (20, 20), 'b': (10, 30)}) == [20]
    assert align_vertical({'a': (30, 20), 'b': (10, 30)}) == []


def test_vertical_rows_tag_mode_and_share_integer():
    rows = vertical_rows([15, 20], ['a', 'b'])
    assert [r['mode'] for r in rows] == ['vertical', 'vertical']
    assert rows[0]['thresholds'] == {'a': 15, 'b': 15}
    assert rows[1]['thresholds'] == {'a': 20, 'b': 20}


# --- 8. Horizontal combos --------------------------------------------------

def test_horizontal_levels_invert_monotone_curves():
    # a denser than b; both monotone non-increasing in t.
    curves = {
        'a': {'thresholds': [3, 4, 5, 6],
              'density': [100.0, 50.0, 20.0, 10.0]},
        'b': {'thresholds': [3, 4, 5, 6],
              'density': [40.0, 30.0, 15.0, 5.0]},
    }
    rows = align_horizontal(curves, levels=4)
    assert rows and all(set(r['thresholds']) == {'a', 'b'} for r in rows)
    # density levels are non-increasing from densest to sparsest endpoint.
    d_lo = max(c['density'][-1] for c in curves.values())
    d_hi = min(c['density'][0] for c in curves.values())
    assert d_lo == 10.0 and d_hi == 40.0
    for r in rows:
        assert d_lo - 1e-9 <= r['level_normalized'] <= d_hi + 1e-9


def test_horizontal_clamp_and_degenerate_single_level():
    curves = {
        'a': {'thresholds': [3, 4, 5], 'density': [10.0, 5.0, 1.0]},
        'b': {'thresholds': [3, 4, 5], 'density': [8.0, 4.0, 2.0]},
    }
    rows = align_horizontal(curves, levels=1)
    assert len(rows) == 1
    assert rows[0]['degenerate'] is True
    assert set(rows[0]['thresholds']) == {'a', 'b'}


def test_horizontal_empty_when_bands_do_not_overlap():
    curves = {
        'a': {'thresholds': [3, 4], 'density': [1.0, 0.5]},
        'b': {'thresholds': [3, 4], 'density': [100.0, 50.0]},
    }
    # a's ceiling density (0.5) > b's floor density (50) is impossible here
    # only if ranges are disjoint; construct that:
    curves['b']['density'] = [0.4, 0.2]  # b fully below a's floor
    assert align_horizontal(curves, levels=4) == []


# --- Normalizer / denominator ---------------------------------------------

def test_density_denominator_prefers_annotated_and_none_when_unknown():
    assert density_denominator({'n_annotated_nodes': 80, 'n_nodes': 100}) == 80
    assert density_denominator({'n_nodes': 100}) == 100
    assert density_denominator({}) is None
    assert density_denominator({}, n_source=10) == 10


def test_normalized_edge_density_variants_and_nan():
    counts = [10, 5]
    meta = {'n_annotated_nodes': 10, 'n_edges': 20}
    assert list(normalized_edge_density(counts, meta, 'per_node')) == [1.0, 0.5]
    assert list(normalized_edge_density(counts, meta, 'raw')) == [10.0, 5.0]
    assert list(normalized_edge_density(counts, meta, 'cone')) == [0.5, 0.25]
    out = normalized_edge_density(counts, {}, 'per_node')
    assert np.all(np.isnan(out))


# --- 6. Cost guard ---------------------------------------------------------

def test_curve_cost_is_independent_of_threshold_span():
    """Cost guard (plan test 6): no O(span) loop — counts are O(log P).

    The meaningful guarantee is that ``count(t)`` is a binary search, so a
    1M-element array over a 3000-point grid completes in well under a
    second (dominated by the one-time sort), not proportional to the array
    values.
    """
    import time
    rng = np.random.default_rng(1)
    edges = rng.integers(1, 3000, size=1_000_000)
    grid = list(range(1, 3001))
    t0 = time.perf_counter()
    curve = density_curve([], edges, grid)
    elapsed = time.perf_counter() - t0
    assert curve['edge_count'][0] == 1_000_000
    assert elapsed < 0.25


# --- 9. Combination round trip --------------------------------------------

def test_auto_rows_round_trip_through_comparison_parameters():
    from comparison.comparison_parameters import ComparisonParameters
    rows = [
        {'id': 'aligned_v_15', 'label': 'V t=15',
         'thresholds': {'ds_a': 15, 'ds_b': 15}},
        {'id': 'aligned_h_1', 'label': 'H d=0.5',
         'thresholds': {'ds_a': 20, 'ds_b': 12}},
    ]
    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'],
        source_neurons=['L2'], target_neurons=['clock'],
        thresholds=[3], threshold_mode='auto',
        threshold_dataset_order=['ds_a', 'ds_b'],
    )
    assert params.threshold_mode == 'auto'
    params.install_auto_combinations(rows)
    assert params.threshold_mode == 'combinations'
    assert params.threshold_auto is True
    queries = params.get_threshold_queries()
    assert len(queries) == 2
    assert queries[0]['thresholds'] == {'ds_a': 15, 'ds_b': 15}
    assert queries[1]['thresholds'] == {'ds_a': 20, 'ds_b': 12}
    assert params.get_thresholds_for_dataset('ds_a') == [15, 20]


def test_row_mode_plumbs_through_queries():
    """Vertical/horizontal origin is preserved on the installed rows and
    exposed by get_threshold_queries for the report's separation schema."""
    from comparison.comparison_parameters import ComparisonParameters
    import tempfile
    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['x'], target_neurons=['y'],
        thresholds=[3], threshold_mode='auto',
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=tempfile.mkdtemp(), verbose=False)
    params.install_auto_combinations([
        {'id': 'aligned_v_19', 'label': 'V t=19',
         'thresholds': {'ds_a': 19, 'ds_b': 19}},
        {'id': 'aligned_h_1', 'label': 'H d=0.5',
         'thresholds': {'ds_a': 20, 'ds_b': 12}},
    ])
    queries = params.get_threshold_queries()
    assert [q['row_mode'] for q in queries] == ['vertical', 'horizontal']
    assert params.threshold_combinations[0]['row_mode'] == 'vertical'
    assert params.threshold_combinations[1]['row_mode'] == 'horizontal'


def test_horizontal_labels_carry_explicit_thresholds():
    """Bootstrap horizontal labels embed the per-dataset thresholds."""
    import inspect
    import comparison.comparison_analyzer as ca
    src = inspect.getsource(ca.ComparisonAnalyzer._bootstrap_auto_mode)
    assert "'row_mode': 'vertical'" in src
    assert "'row_mode': 'horizontal'" in src
    assert '_ds_txt(' in src  # per-dataset threshold text in the label


def test_report_separates_vertical_and_horizontal_groups():
    """The query report partitions queries and renders group banners."""
    import inspect
    from comparison import html_report_generator as G
    src = inspect.getsource(G._generate_query_html_report)
    assert "_row_mode_of" in src
    assert "QUERY_GROUP_META" in src
    assert "_iter_grouped" in src
    assert "per-threshold analysis" in src
    assert "combination analysis" in src
    # presence matrices + networks render per group with distinct ids
    assert "networks-{_gmode or _gi}" in src
    assert "f'edge-matrices{_suffix}'" in src
    assert "rows=_grows, group_banner=_banner" in src
    nsrc = inspect.getsource(G._generate_networks_section)
    assert 'section_id' in nsrc


def test_auto_mode_requires_two_datasets():
    from comparison.comparison_parameters import ComparisonParameters
    with pytest.raises(ValueError):
        ComparisonParameters(
            datasets=['ds_a'], source_neurons=['L2'],
            target_neurons=['clock'], thresholds=[3],
            threshold_mode='auto')


# --- W* provenance fix (plan §3.3b T-W1 / T-W3) ---------------------------

def test_find_paths_core_reset_block_resets_wstar():
    """P4: _find_paths_core must reset strongest_retained_bottleneck."""
    import inspect
    import coana
    src = inspect.getsource(coana.FindNeuronConnection._find_paths_core)
    assert 'self.strongest_retained_bottleneck = None' in src
    # P1: the measured value is set from path_bottlenecks.
    assert 'max(path_bottlenecks) if path_bottlenecks else None' in src


def test_no_dead_widest_path_backward_call_in_core():
    """P3: the dead no-op recompute call must be gone."""
    import inspect
    import coana
    src = inspect.getsource(coana.FindNeuronConnection._find_paths_core)
    # The call site (assignment from the method) must not remain; the name
    # may still appear in the removal comment.
    assert '_w = self._widest_path_backward(' not in src
    assert '_vals = [_w[-1].get(s) for s in source_ID' not in src


def test_replay_slice_updates_wstar_from_survivors():
    """P1 (replay): _materialize_threshold sets W* from surviving paths."""
    import inspect
    import coana
    src = inspect.getsource(
        coana.FindNeuronConnection.FindAllPathMultiThreshold)
    assert 'slice_wstar' in src
    assert "capture['strongest_retained_bottleneck'] = slice_wstar" in src


# --- Phase D: auto bootstrap installs measured rows ------------------------

def test_bootstrap_auto_mode_installs_aligned_rows(monkeypatch):
    import tempfile
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['L2'],
        target_neurons=['clock'], thresholds=[3], threshold_mode='auto',
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=tempfile.mkdtemp(), verbose=False)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    analyzer.parameters.output_folder = None  # no disk caching in the test
    calls = []

    def _fake_run(ds, t, verbose_mode='simple'):
        calls.append((ds, t))
        return pd.DataFrame()

    monkeypatch.setattr(analyzer, 'run_path_analysis', _fake_run)
    monkeypatch.setattr(
        analyzer, '_build_density_curves',
        lambda: ({'ds_a': {'thresholds': [15, 20],
                           'path_count': [3, 1],
                           'edge_count': [10, 5],
                           'density': [1.0, 0.5]},
                  'ds_b': {'thresholds': [11, 15],
                           'path_count': [2, 1],
                           'edge_count': [8, 4],
                           'density': [0.8, 0.4]}},
                 {'ds_a': (15, 20), 'ds_b': (11, 15)},
                 {'ds_a': {}, 'ds_b': {}}))

    assert analyzer._bootstrap_auto_mode() is True
    assert calls == [('ds_a', 3), ('ds_b', 3)]
    assert params.threshold_mode == 'combinations'
    assert params.threshold_auto is True
    queries = params.get_threshold_queries()
    assert queries  # at least one aligned row
    # Every row must carry one threshold per dataset.
    for q in queries:
        assert set(q['thresholds']) == {'ds_a', 'ds_b'}


def test_bootstrap_falls_back_without_capture(monkeypatch):
    import tempfile
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['L2'],
        target_neurons=['clock'], thresholds=[3], threshold_mode='auto',
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=tempfile.mkdtemp(), verbose=False)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    analyzer.parameters.output_folder = None
    monkeypatch.setattr(analyzer, 'run_path_analysis',
                        lambda ds, t, verbose_mode='simple': pd.DataFrame())
    monkeypatch.setattr(analyzer, '_build_density_curves',
                        lambda: ({}, {}, {}))
    assert analyzer._bootstrap_auto_mode() is False
    assert params.thresholds == [3]  # restored
    status = analyzer._auto_bootstrap_status
    assert status['outcome'] == 'degraded'
    assert sorted(status['uncaptured']) == ['ds_a', 'ds_b']


# --- Auto-mode resilience: one failed dataset must not veto the schedule ---

def _three_ds_params(tmp):
    from comparison.comparison_parameters import ComparisonParameters
    return ComparisonParameters(
        datasets=['ds_a', 'ds_b', 'ds_c'], source_neurons=['L2'],
        target_neurons=['clock'], thresholds=[3], threshold_mode='auto',
        threshold_dataset_order=['ds_a', 'ds_b', 'ds_c'],
        output_folder=str(tmp), verbose=False)


def _two_point_curve(lo, hi, dense, sparse):
    return {'thresholds': [lo, hi], 'path_count': [10, 1],
            'edge_count': [10, 1], 'density': [dense, sparse]}


def _two_ds_curves():
    return ({'ds_a': _two_point_curve(3, 20, 2.0, 0.2),
             'ds_b': _two_point_curve(3, 18, 1.6, 0.16)},
            {'ds_a': (3, 20), 'ds_b': (3, 18)},
            {'ds_a': {}, 'ds_b': {}})


def _three_ds_curves():
    curves, windows, metas = _two_ds_curves()
    curves['ds_c'] = _two_point_curve(3, 16, 1.2, 0.12)
    windows['ds_c'] = (3, 16)
    metas['ds_c'] = {}
    return curves, windows, metas


def test_bootstrap_installs_verticals_when_one_dataset_fails(monkeypatch,
                                                             tmp_path):
    """A dataset that cannot be measured costs only itself (plan P3).

    The per-dataset loop used to share one try/except, so a raise in ANY
    dataset discarded the captures the others had already produced and the
    run silently degraded to the Standard schedule (the 2026-09-21
    regression: hemibrain's cache-marker write raised, three good
    measurements were thrown away, and nothing said so in the outputs).
    """
    from comparison.comparison_analyzer import ComparisonAnalyzer
    params = _three_ds_params(tmp_path)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    calls = []

    def _fake(ds, boot_t):
        calls.append((ds, boot_t))
        if ds == 'ds_c':
            raise RuntimeError('cache marker exploded')

    monkeypatch.setattr(analyzer, '_bootstrap_measure_floor', _fake)
    # ds_c yields no capture: the curve dict carries only ds_a / ds_b.
    curves, windows, metas = _two_ds_curves()
    monkeypatch.setattr(analyzer, '_build_density_curves',
                        lambda: (curves, windows, metas))

    assert analyzer._bootstrap_auto_mode() is True
    assert params.threshold_auto is True
    assert params.threshold_mode == 'combinations'
    queries = params.get_threshold_queries()
    assert queries
    # Vertical spine only — no density-matched row without every curve.
    assert all(str(q['id']).startswith('threshold=') for q in queries)
    # The schedule validator rejects a row missing a dataset, so the failed
    # one still carries a cell.
    for q in queries:
        assert set(q['thresholds']) == {'ds_a', 'ds_b', 'ds_c'}
    status = analyzer._auto_bootstrap_status
    assert status['outcome'] == 'verticals_only'
    assert status['uncaptured'] == ['ds_c']
    assert status['horizontal_rows'] == 0
    assert 'cache marker exploded' in status['failures']['ds_c']
    # The failure is retried once before the dataset is given up on.
    assert calls.count(('ds_c', 3)) == 2


def test_bootstrap_keeps_horizontals_when_every_dataset_captures(monkeypatch,
                                                                tmp_path):
    """The full-capture path is unchanged: density-matched rows still install."""
    from comparison.comparison_analyzer import ComparisonAnalyzer
    params = _three_ds_params(tmp_path)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    monkeypatch.setattr(analyzer, '_bootstrap_measure_floor',
                        lambda ds, boot_t: None)
    curves, windows, metas = _two_ds_curves()
    curves['ds_c'] = _two_point_curve(3, 16, 1.4, 0.14)
    windows['ds_c'] = (3, 16)
    metas['ds_c'] = {}
    monkeypatch.setattr(analyzer, '_build_density_curves',
                        lambda: (curves, windows, metas))

    assert analyzer._bootstrap_auto_mode() is True
    ids = [str(q['id']) for q in params.get_threshold_queries()]
    assert any(i.startswith('aligned_density=') for i in ids)
    status = analyzer._auto_bootstrap_status
    assert status['outcome'] == 'installed'
    assert status['uncaptured'] == []
    assert status['horizontal_rows'] >= 1


def test_auto_mode_degradation_reaches_user_warning_notes(tmp_path):
    """A degraded auto mode must be visible in the run folder (plan P4)."""
    from comparison.comparison_analyzer import ComparisonAnalyzer
    import pathlib
    params = _three_ds_params(tmp_path)
    run_dir = pathlib.Path(params.full_output_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    analyzer._append_auto_mode_status_notes({
        'outcome': 'degraded',
        'uncaptured': ['ds_c'],
        'failures': {'ds_c': 'RuntimeError: cache marker exploded'},
    })
    text = (run_dir / 'user_warning_notes.txt').read_text(encoding='utf-8')
    assert 'auto mode did NOT resolve' in text
    assert 'ds_c bootstrap measurement failed: RuntimeError' in text
    assert 'run_manifest.json' in text
    # An installed run writes nothing.
    analyzer._append_auto_mode_status_notes({'outcome': 'installed'})
    assert (run_dir / 'user_warning_notes.txt').read_text(
        encoding='utf-8') == text


def _bootstrap_with(monkeypatch, out_dir, failures=(), curves=None):
    """Run the real bootstrap with the per-dataset measurement stubbed out."""
    import pathlib
    from comparison.comparison_analyzer import ComparisonAnalyzer
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    params = _three_ds_params(out_dir)
    analyzer = ComparisonAnalyzer(params, verbose=False)

    def _fake(ds, boot_t):
        if ds in failures:
            raise RuntimeError(f'{ds} exploded')

    monkeypatch.setattr(analyzer, '_bootstrap_measure_floor', _fake)
    monkeypatch.setattr(analyzer, '_build_density_curves',
                        lambda: curves if curves is not None
                        else ({}, {}, {}))
    analyzer._bootstrap_auto_mode()
    return analyzer


def test_auto_mode_status_uses_one_schema_and_reaches_the_manifest(monkeypatch,
                                                                  tmp_path):
    """Every outcome publishes the same keys, and the manifest carries them.

    The warning note sends a reader to ``run_manifest.json ->
    auto_mode_status``, so a key that exists only for some outcomes breaks
    anything that reads the block generically; and without a bootstrap the
    key must be absent rather than null, because its presence is what marks a
    run as an auto-mode run.
    """
    import json
    import pathlib
    core = {'outcome', 'floor_threshold', 'requested_thresholds',
            'vertical_rows', 'horizontal_rows', 'uncaptured', 'failures',
            'reason'}
    degraded = _bootstrap_with(monkeypatch, tmp_path / 'degraded',
                               failures={'ds_a', 'ds_b', 'ds_c'})
    partial = _bootstrap_with(monkeypatch, tmp_path / 'partial',
                              failures={'ds_c'}, curves=_two_ds_curves())
    installed = _bootstrap_with(monkeypatch, tmp_path / 'installed',
                                curves=_three_ds_curves())
    statuses = [degraded._auto_bootstrap_status,
                partial._auto_bootstrap_status,
                installed._auto_bootstrap_status]
    assert [s['outcome'] for s in statuses] == [
        'degraded', 'verticals_only', 'installed']
    for status in statuses:
        assert set(status) == core
        assert status['floor_threshold'] == 3
    # `reason` carries the explanation the per-dataset failures cannot give.
    assert degraded._auto_bootstrap_status['reason']
    assert installed._auto_bootstrap_status['reason'] is None

    # A run that never bootstrapped (standard / custom-combination mode)
    # publishes no block at all: its presence is what marks an auto run.
    from comparison.comparison_analyzer import ComparisonAnalyzer
    out_dir = tmp_path / 'never-bootstrapped'
    out_dir.mkdir(parents=True, exist_ok=True)
    plain = ComparisonAnalyzer(_three_ds_params(out_dir), verbose=False)
    def _manifest_of(analyzer):
        out = pathlib.Path(analyzer.parameters.full_output_path)
        out.mkdir(parents=True, exist_ok=True)
        analyzer._write_run_manifest(str(out))
        return json.loads((out / 'run_manifest.json').read_text(
            encoding='utf-8'))

    assert 'auto_mode_status' not in _manifest_of(plain)
    manifest = _manifest_of(partial)
    assert manifest['auto_mode_status'] == partial._auto_bootstrap_status


def _marker_target(**overrides):
    """A bare FindNeuronConnection with the marker's collaborators stubbed."""
    import coana

    obj = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    obj.dataset = 'ds_x'
    obj._warn_notes = []
    obj.prints = []
    obj._vprint = lambda *a, **k: obj.prints.append(a)
    obj.calls = []
    obj._mark_neurons_as_cached = lambda *a, **k: obj.calls.append(a)
    for key, value in overrides.items():
        setattr(obj, key, value)
    return obj


def _printed(obj, fragment):
    return any(fragment in str(arg) for args in obj.prints for arg in args)


def test_cache_marker_failure_does_not_escape_the_fetch_path():
    """A marker write costs a later re-fetch, never the run in hand (P1)."""
    import inspect
    import pandas as pd
    import coana

    def _boom(*args, **kwargs):
        raise RuntimeError('neuron index write failed')

    obj = _marker_target(_mark_neurons_as_cached=_boom)
    frame = pd.DataFrame({'bodyId_pre': ['1', '2', '3'], 'weight': [1, 2, 3]})
    assert obj._mark_fetched_neurons_quietly(frame, ['1', '2', '3']) is None
    assert any('[cache markers]' in note for note in obj._warn_notes)
    assert 'completion cache markers failed' in obj._warn_notes[0]
    assert 'neuron index write failed' in obj._warn_notes[0]
    assert _printed(obj, 'Neuron-index cache markers failed')

    # The 2026-09-21 failure window was the slice itself, which the frame's
    # shape can break before any marker code runs (plan §6a item 6).
    obj = _marker_target()
    broken = pd.DataFrame({'bodyId_post': [1, 2, 3]})   # no bodyId_pre
    assert obj._mark_fetched_neurons_quietly(broken, ['1', '2']) is None
    assert obj.calls == []
    assert 'per-neuron slice behind cache markers failed' in obj._warn_notes[0]
    assert _printed(obj, 'Neuron-index cache slice failed')

    # Success marks exactly the sliced rows, on both engines.
    import polars as pl
    obj = _marker_target()
    pl_frame = pl.DataFrame({'bodyId_pre': ['1', '2', '9'], 'weight': [1, 2, 3]})
    assert obj._mark_fetched_neurons_quietly(pl_frame, ['1', '2', '3']) == 3
    marked_ids, marked_frame = obj.calls[0][0], obj.calls[0][1]
    assert marked_ids == ['1', '2', '3']
    assert sorted(marked_frame['bodyId_pre']) == ['1', '2']

    # The all-empty poisoning guard still skips rather than marks.
    obj = _marker_target()
    ids = [str(i) for i in range(60)]
    empty = pd.DataFrame({'bodyId_pre': pd.Series([], dtype=object)})
    assert obj._mark_fetched_neurons_quietly(empty, ids) == 0
    assert obj.calls == []
    assert obj._warn_notes == []

    src = inspect.getsource(coana.FindNeuronConnection)
    assert src.count(
        'self._mark_fetched_neurons_quietly(') == 2
    assert 'self._mark_neurons_as_cached(neurons_to_mark' not in src


# --- Regression guard: denominator is the PRUNED searched cone ------------
# The first real run computed N from the pre-prune discovery frontier
# (MCNS 170k / BANC 3.34M nodes), collapsing MCNS and FAFB to identical
# equal-density thresholds. N must come from the pruned graph frames.

def test_density_node_ids_come_from_pruned_frames_not_layer_neurons():
    import inspect
    import coana
    src = inspect.getsource(coana.FindNeuronConnection._find_paths_core)
    start = src.find('for _frame in graph_frames:\n                if _frame is None')
    assert start != -1, "density capture block not found"
    block = src[start:src.find('for _frame in graph_frames:', start + 10)]
    # The node set is built from the pruned frame endpoint columns...
    assert "_frame['bodyId_pre']" in block and "_frame['bodyId_post']" in block
    assert '_node_ids.update(_pre)' in block
    # ...and NOT from the discovery frontier.
    assert 'layer_neurons or []' not in block


def test_meta_reports_applied_edges_and_cone_edges_separately():
    import inspect
    import coana
    src = inspect.getsource(
        coana.FindNeuronConnection._persist_density_artifacts)
    assert "'n_edges': n_edges_applied" in src
    assert "'n_cone_edges': int(ew.size)" in src


def test_report_section_renders_in_both_modes():
    """The auto section has no vis-rel closure args (usable by both paths)."""
    import inspect
    from comparison import html_report_generator as G
    sig = inspect.signature(G._generate_auto_density_alignment_section)
    assert list(sig.parameters) == ['analyzer', 'dataset_names', 'nickname_map']
    # Both report shells must call it.
    qsrc = inspect.getsource(G._generate_query_html_report)
    assert '_generate_auto_density_alignment_section' in qsrc
    std = inspect.getsource(G.generate_html_report)
    assert 'include_auto_density' in std


# --- Classified cone capture: typed / untyped / debris ---------------------

def test_density_table_sets_classify_with_label_utils():
    import pandas as pd
    import coana
    tbl = pd.DataFrame({
        'bodyId': ['1', '2', '3', '4', '5'],
        'type': ['Tm1', 'Unknown', None, '720575941501912299', 'L2'],
    })
    cache = {'k': (0, tbl)}
    table_ids, typed_ids, known = coana._density_table_sets(cache)
    assert known and table_ids == {'1', '2', '3', '4', '5'}
    # 'Unknown', NaN and the digit-fallback are untyped; 'Tm1'/'L2' typed.
    assert typed_ids == {'1', '5'}
    cls = coana.classify_cone_edges(
        ['1', '1', '2', '9', '1'], ['2', '3', '3', '1', '5'],
        table_ids, typed_ids)
    assert cls[0] == coana.DENSITY_CLS_UNTYPED      # typed -> untyped
    assert cls[1] == coana.DENSITY_CLS_UNTYPED      # typed -> NaN-type (table)
    assert cls[2] == coana.DENSITY_CLS_UNTYPED      # untyped -> untyped
    assert cls[3] == coana.DENSITY_CLS_DEBRIS       # '9' absent from table
    assert cls[4] == coana.DENSITY_CLS_TYPED        # typed -> typed


def _write_classified_artifacts(params, per_ds):
    """Write classified density artifacts; per_ds: ds -> (bns, (w, cls), meta)."""
    import json
    import os
    import numpy as np
    for ds, (bns, (w, cls), meta) in per_ds.items():
        d = os.path.join(params.dataset_data_path,
                         params._sanitize_name(ds), '_density')
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'density_path_bottlenecks.npy'),
                np.asarray(bns, dtype=float))
        np.savez_compressed(os.path.join(d, 'density_edges.npz'),
                            weight=np.asarray(w, dtype=np.int32),
                            cls=np.asarray(cls, dtype=np.int8))
        json.dump(meta, open(os.path.join(d, 'density_meta.json'), 'w'))


def _pair_params(tmp, drop_untyped, threshold_mode='standard'):
    from comparison.comparison_parameters import ComparisonParameters
    return ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['x'], target_neurons=['y'],
        thresholds=[3], threshold_mode=threshold_mode,
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=str(tmp), drop_untyped=drop_untyped, verbose=False)


def test_basis_selection_typed_vs_all_but_debris(tmp_path):
    """drop_untyped=True -> typed-only E and N; False -> keep untyped, never debris."""
    import numpy as np
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer
    # ds_a cone: 4 edges = 2 typed, 1 untyped, 1 debris; 3 typed paths.
    per_ds = {
        'ds_a': ([10, 20, 30],
                 ([50, 60, 70, 80], [0, 0, 1, 2]),
                 {'applied': 10, 'w_start': 10, 'w_star_measured': 30.0,
                  'n_nodes': 10, 'n_nodes_typed': 4, 'n_nodes_untyped': 2,
                  'n_nodes_debris': 1, 'n_annotated_nodes': 4,
                  'denominator': 'typed_nodes_in_searched_graph',
                  'path_complete_from': 10, 'n_paths': 3, 'n_edges': 4,
                  'budget_bitten': False, 'paths_complete': True}),
        'ds_b': ([10, 20],
                 ([40, 55], [0, 0]),
                 {'applied': 10, 'w_start': 10, 'w_star_measured': 20.0,
                  'n_nodes': 5, 'n_nodes_typed': 5, 'n_nodes_untyped': 0,
                  'n_nodes_debris': 0, 'n_annotated_nodes': 5,
                  'denominator': 'typed_nodes_in_searched_graph',
                  'path_complete_from': 10, 'n_paths': 2, 'n_edges': 2,
                  'budget_bitten': False, 'paths_complete': True}),
    }
    for drop, want_basis, want_e10_a, want_n, want_e30 in (
            (True, 'bodyId_edges_typed', 2, 4, 2),
            (False, 'bodyId_edges_all_but_debris', 3, 6, 3)):
        params = _pair_params(tmp_path / str(drop), drop_untyped=drop)
        _write_classified_artifacts(params, per_ds)
        an = ComparisonAnalyzer(params, verbose=False)
        curves, windows, metas = an._build_density_curves()
        c = curves['ds_a']
        assert c['basis'] == want_basis
        assert c['edge_count'][0] == want_e10_a          # at t=10 (all kept)
        # Typed weights (50, 60) survive the whole window; the untyped
        # weight-70 edge joins them at t=30 only in all_but_debris.
        assert c['edge_count'][-1] == want_e30
        assert np.isclose(c['density'][0], want_e10_a / want_n)
    # ds_b has no untyped/debris: identical under both bases.
    params = _pair_params(tmp_path / "x", drop_untyped=True)
    _write_classified_artifacts(params, per_ds)
    curves, _, _ = ComparisonAnalyzer(params, verbose=False).\
        _build_density_curves()
    assert curves['ds_b']['basis'] == 'bodyId_edges_typed'
    assert curves['ds_b']['edge_count'][0] == 2


def test_export_in_standard_mode_writes_curves_not_aligned_rows(tmp_path):
    """All-mode curves: standard mode gets curves/windows, no aligned rows."""
    import os
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer
    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['x'], target_neurons=['y'],
        thresholds=[10], threshold_mode='standard',
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=str(tmp_path), drop_untyped=True, verbose=False)
    _write_classified_artifacts(params, {
        'ds_a': ([10, 20], ([50, 60, 70], [0, 0, 1]),
                 {'applied': 10, 'w_start': 10, 'w_star_measured': 20.0,
                  'n_nodes': 5, 'n_nodes_typed': 5, 'n_nodes_untyped': 1,
                  'n_nodes_debris': 0, 'path_complete_from': 10,
                  'n_paths': 2, 'n_edges': 3, 'budget_bitten': False,
                  'paths_complete': True,
                  'denominator': 'typed_nodes_in_searched_graph'}),
        'ds_b': ([10], ([40], [0]),
                 {'applied': 10, 'w_start': 10, 'w_star_measured': 10.0,
                  'n_nodes': 5, 'n_nodes_typed': 5, 'n_nodes_untyped': 0,
                  'n_nodes_debris': 0, 'path_complete_from': 10,
                  'n_paths': 1, 'n_edges': 1, 'budget_bitten': False,
                  'paths_complete': True,
                  'denominator': 'typed_nodes_in_searched_graph'}),
    })
    an = ComparisonAnalyzer(params, verbose=False)
    cr = os.path.join(params.full_output_path, 'comparison_results')
    os.makedirs(cr, exist_ok=True)
    an._export_density_alignment(cr)
    files = set(os.listdir(cr))
    assert 'density_curves.csv' in files
    assert 'density_windows.csv' in files
    assert 'density_alignment_best_matches.csv' not in files
    assert params.threshold_mode == 'standard'   # untouched
    # n_edges is all-class (3); n_edges_active_basis re-counts under the
    # typed basis (2) and matches density_curves.csv at w_start.
    win = pd.read_csv(os.path.join(cr, 'density_windows.csv'))
    row = win[win.dataset == 'ds_a'].iloc[0]
    assert int(row.n_edges) == 3 and int(row.n_edges_active_basis) == 2
    cur = pd.read_csv(os.path.join(cr, 'density_curves.csv'))
    e_at_wstart = cur[(cur.dataset == 'ds_a')
                      & (cur.threshold == int(row.w_start))].edge_count.iloc[0]
    assert int(e_at_wstart) == int(row.n_edges_active_basis)
    # capture flag is always-on at the delegated constructors
    import inspect
    src = inspect.getsource(ComparisonAnalyzer.run_path_analysis)
    assert 'capture_density=True' in src


def test_density_aligned_rows_partial_capture_emits_horizontal():
    """Plan C: one dataset without a density capture must not veto the
    horizontal rows — they cover the captured datasets and carry
    ``partial_datasets``; vertical rows keep the full shared spine."""
    from types import SimpleNamespace
    from comparison.comparison_analyzer import ComparisonAnalyzer

    def curve(lo, hi):
        ts = list(range(lo, hi + 1))
        return {
            'thresholds': ts,
            'density': [0.9 - 0.05 * (t - lo) for t in ts],
            'path_count': [100 // t for t in ts],
            'edge_count': [200 // t for t in ts],
            'basis': 'typed',
        }

    curves = {'d1': curve(2, 8), 'd2': curve(3, 6)}
    windows = {'d1': (2, 8), 'd2': (3, 6)}
    metas = {'d1': {}, 'd2': {}}
    fake = SimpleNamespace(
        parameters=SimpleNamespace(
            get_dataset_names=lambda: ['d1', 'd2', 'd3']))

    rows = ComparisonAnalyzer._density_aligned_rows(fake, curves, windows,
                                                    metas)
    horiz = [r for r in rows if 'horizontal' in (r.get('mode') or '')]
    assert horiz, 'partial capture must still emit horizontal rows'
    for r in horiz:
        assert set(r['thresholds']) == {'d1', 'd2'}
        assert r['partial_datasets'] == ['d3']
    vert = [r for r in rows if 'vertical' in (r.get('mode') or '')]
    assert vert, 'vertical spine keeps all datasets'
    for r in vert:
        assert set(r['thresholds']) == {'d1', 'd2', 'd3'}

    # Full coverage: no partial markers (regression).
    curves3 = dict(curves, d3=curve(2, 7))
    windows3 = dict(windows, d3=(2, 7))
    metas3 = dict(metas, d3={})
    rows_full = ComparisonAnalyzer._density_aligned_rows(
        fake, curves3, windows3, metas3)
    assert not any(r.get('partial_datasets') for r in rows_full)


def test_untyped_drop_note_names_the_query_cell_the_csv_uses(tmp_path):
    """The note's ``query_id=`` is the CSV cell, not a third id scheme.

    One threshold can serve both a vertical and a horizontal row, so the
    shared drop cell belongs to several queries. The CSV renders that as
    ``a;b``; the note joined with ``,``, which reads as one id containing a
    comma and cannot be matched back to a row of the file it points at.
    """
    from comparison.comparison_analyzer import ComparisonAnalyzer
    import pathlib
    params = _three_ds_params(tmp_path)
    params.threshold_mode = 'combinations'
    params.threshold_combinations = [
        {'id': 'threshold=3', 'label': 'threshold=3',
         'thresholds': {'ds_a': 3, 'ds_b': 3, 'ds_c': 3}},
        {'id': 'aligned_density=1.6', 'label': 'aligned_density=1.6',
         'thresholds': {'ds_a': 3, 'ds_b': 5, 'ds_c': 7}},
    ]
    analyzer = ComparisonAnalyzer(params, verbose=False)
    out_dir = pathlib.Path(analyzer.parameters.full_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzer._untyped_drop_stats = {
        ('ds_a', 3): {'rows': 12, 'neurons': 4, 'untyped_pre': 7,
                      'untyped_post': 5, 'total_rows': 100,
                      'fraction': 0.12},
    }
    analyzer._untyped_dropped_records = [pd.DataFrame({
        'dataset': ['ds_a', 'ds_a'], 'threshold': [3, 3],
        'type_pre': ['U', 'L2'], 'type_post': ['C', 'U'],
        'untyped_side': ['pre', 'post']})]
    analyzer._export_untyped_drop_records()

    csv_id = pd.read_csv(
        out_dir / 'comparison_results' / 'untyped_dropped_records.csv',
        dtype=str)['query_id'].unique().tolist()
    assert csv_id == ['threshold=3;aligned_density=1.6']

    lines = [ln for ln in (out_dir / 'user_warning_notes.txt').read_text(
        encoding='utf-8').splitlines() if '[untyped dropped]' in ln]
    # One drop cell -> one line, however many queries reuse it (the counts
    # are cell-level, so repeating them per query would read as double).
    assert len(lines) == 1
    assert 'query_id=threshold=3;aligned_density=1.6' in lines[0]
    # The embedded ';' carries no following space, so this line's own '; '
    # field separator still yields the whole id list as its last field.
    assert lines[0].split('; ')[-1] == (
        'query_id=threshold=3;aligned_density=1.6)')


# ---------------------------------------------------------------------------
# Ratio-basis float variants (plan Phase-3 remainder)
# ---------------------------------------------------------------------------
class TestRatioBasisFloatLadders:
    def _ratio_meta(self, applied, w_star, w_start=None):
        return {
            'weight_basis': 'connection_ratio',
            'applied': applied,
            'w_start': w_start,
            'w_star_measured': w_star,
        }

    def test_ratio_window_is_float_and_unfloored(self):
        from comparison.threshold_density import dataset_window
        lo, hi = dataset_window(self._ratio_meta(0.001, 0.05))
        assert lo == 0.001            # no synapse floor of 3
        assert hi == 0.05

    def test_synapse_window_stays_integer(self):
        from comparison.threshold_density import dataset_window
        lo, hi = dataset_window({'applied': 3, 'w_star_measured': 30})
        assert (lo, hi) == (3, 30) and isinstance(lo, int)

    def test_ratio_vertical_ladder_floats(self):
        from comparison.threshold_density import align_vertical
        windows = {'A': (0.001, 0.02), 'B': (0.002, 0.05)}
        ladder = align_vertical(windows, K=4)
        assert all(isinstance(t, float) for t in ladder)
        assert ladder[0] == 0.002 and ladder[-1] == 0.02  # floor/ceiling
        assert all(0.002 <= t <= 0.02 for t in ladder)

    def test_synapse_vertical_ladder_unchanged(self):
        from comparison.threshold_density import align_vertical
        ladder = align_vertical({'A': (3, 30), 'B': (5, 30)}, K=4)
        assert all(isinstance(t, int) for t in ladder)
        assert ladder[0] == 5 and ladder[-1] == 30

    def test_ratio_density_curve_float_grid(self):
        import numpy as np
        from comparison.threshold_density import density_curve
        ew = np.array([0.050, 0.020, 0.010, 0.004, 0.0012])
        grid = [0.0012, 0.004, 0.010, 0.020, 0.050]
        c = density_curve([], ew, grid)
        assert c['edge_count'].tolist() == [5, 4, 3, 2, 1]

    def test_ratio_horizontal_inverts_to_tiers(self):
        from comparison.threshold_density import align_horizontal
        curves = {
            'A': {'thresholds': [0.001, 0.004, 0.010, 0.020],
                  'density': [1.0, 0.8, 0.5, 0.2]},
            'B': {'thresholds': [0.002, 0.005, 0.012, 0.030],
                  'density': [0.9, 0.7, 0.45, 0.1]},
        }
        rows = align_horizontal(curves, levels=2, normalizer='raw')
        assert rows
        for r in rows:
            for ds, t in r['thresholds'].items():
                assert isinstance(t, float) or (
                    isinstance(t, int) and float(t).is_integer())
                assert t in curves[ds]['thresholds']

    def test_ratio_vertical_rows_carry_floats(self):
        from comparison.threshold_density import vertical_rows
        rows = vertical_rows([0.001, 0.004], ['A', 'B'])
        assert rows[0]['thresholds'] == {'A': 0.001, 'B': 0.001}


# ---------------------------------------------------------------------------
# Ratio query identity: float tiers must survive the whole query chain
# (get_threshold_queries / jobs / views / manifest / points). A bare int()
# truncates every sub-1 tier to 0 — verified corrupting the real AUTO+ratio
# run's threshold_view.json (queries showed thresholds 0, raw_run_key
# ".../minsyn_0"). Synapse runs must stay byte-identical.
# ---------------------------------------------------------------------------
class TestRatioQueryIdentity:
    def _ratio_params(self, **overrides):
        from comparison.comparison_parameters import ComparisonParameters
        kwargs = dict(
            datasets=['ds_a', 'ds_b'], source_neurons=['L2'],
            target_neurons=['clock'], weight_basis='connection_ratio',
            thresholds=[0.0011919],
            threshold_combinations=[{
                'id': 'threshold=0.0011919', 'label': 'spine',
                'row_mode': 'vertical',
                'thresholds': {'ds_a': 0.0011919, 'ds_b': 0.0011919},
            }],
            threshold_mode='combinations',
            threshold_dataset_order=['ds_a', 'ds_b'],
            output_folder=None, verbose=False)
        kwargs.update(overrides)
        return ComparisonParameters(**kwargs)

    def test_combination_queries_preserve_float_tiers(self):
        params = self._ratio_params()
        queries = params.get_threshold_queries()
        assert queries[0]['thresholds'] == {'ds_a': 0.0011919,
                                            'ds_b': 0.0011919}
        jobs = params.get_unique_threshold_jobs()
        assert set(jobs) == {('ds_a', 0.0011919), ('ds_b', 0.0011919)}

    def test_auto_installed_queries_preserve_float_tiers(self):
        params = self._ratio_params(
            threshold_mode='auto', threshold_combinations=None)
        params.install_auto_combinations([{
            'id': 'threshold=0.0011919', 'label': 'spine',
            'thresholds': {'ds_a': 0.0011919, 'ds_b': 0.0011919}}])
        assert params.threshold_mode == 'combinations'
        assert params.get_threshold_queries()[0]['thresholds'] == {
            'ds_a': 0.0011919, 'ds_b': 0.0011919}

    def test_standard_ratio_queries_preserve_float_tiers(self):
        params = self._ratio_params(
            threshold_mode='standard', threshold_combinations=None,
            thresholds=[0.001, 0.01])
        queries = params.get_threshold_queries()
        assert [q['thresholds']['ds_a'] for q in queries] == [0.001, 0.01]
        assert queries[0]['id'] == 'threshold_0.001'

    def test_legacy_dataset_thresholds_accept_float_tiers(self):
        params = self._ratio_params(
            threshold_mode='standard', threshold_combinations=None,
            dataset_thresholds={'ds_a': [0.001, 0.01], 'ds_b': [0.001]})
        assert params.threshold_mode == 'legacy_vertical'
        assert params.dataset_thresholds['ds_a'] == [0.001, 0.01]

    def test_synapse_query_identity_byte_identical(self):
        from comparison.comparison_parameters import ComparisonParameters
        params = ComparisonParameters(
            datasets=['ds_a', 'ds_b'], source_neurons=['L2'],
            target_neurons=['clock'], thresholds=[3, 5],
            threshold_dataset_order=['ds_a', 'ds_b'],
            output_folder=None, verbose=False)
        queries = params.get_threshold_queries()
        assert [q['id'] for q in queries] == ['threshold_3', 'threshold_5']
        assert queries[0]['thresholds'] == {'ds_a': 3, 'ds_b': 3}
        assert all(isinstance(v, int)
                   for q in queries for v in q['thresholds'].values())
        assert set(params.get_unique_threshold_jobs()) == {
            ('ds_a', 3), ('ds_b', 3), ('ds_a', 5), ('ds_b', 5)}

    def test_analyzer_threshold_view_preserves_floats(self):
        import tempfile
        from comparison.comparison_analyzer import ComparisonAnalyzer
        params = self._ratio_params()
        analyzer = ComparisonAnalyzer(params, verbose=False)
        analyzer.parameters.output_folder = None
        analyzer._path_run_meta[('ds_a', 0.0011919)] = {
            'requested_threshold': 0.0011919}
        view = analyzer.get_threshold_view('ds_a', 0.0011919)
        assert view['requested_threshold'] == 0.0011919
        assert view['applied_threshold'] == 0.0011919
        assert view['is_exact'] is True
        views = analyzer.get_threshold_queries_view()
        assert views[0]['applied_thresholds']['ds_a'] == 0.0011919

    def test_manifest_rows_use_ratio_folder_grammar(self):
        import tempfile
        from comparison.comparison_analyzer import ComparisonAnalyzer
        params = self._ratio_params()
        analyzer = ComparisonAnalyzer(params, verbose=False)
        analyzer.parameters.output_folder = None
        analyzer._path_run_meta[('ds_a', 0.0011919)] = {
            'requested_threshold': 0.0011919}
        analyzer._path_run_meta[('ds_b', 0.0011919)] = {
            'requested_threshold': 0.0011919}
        rows = analyzer._threshold_query_manifest_rows()
        by_ds = {r['dataset']: r for r in rows}
        assert by_ds['ds_a']['requested_threshold'] == 0.0011919
        assert by_ds['ds_a']['raw_run_key'] == 'ds_a/minratio_0_0011919'
        assert by_ds['ds_a']['applied_threshold'] == 0.0011919

    def test_comparison_points_preserve_float_tiers(self):
        from comparison.point_context import (
            ComparisonPoint, point_from_value, points_from_parameters)
        pts = points_from_parameters(self._ratio_params())
        assert pts[0].thresholds_by_dataset == {'ds_a': 0.0011919,
                                                'ds_b': 0.0011919}
        assert pts[0].file_stem.startswith('query_')  # combinations stem
        # Standard ratio point: float tier + minratio_ stem.
        params = self._ratio_params(
            threshold_mode='standard', threshold_combinations=None,
            thresholds=[0.001])
        std = points_from_parameters(params)
        assert std[0].thresholds_by_dataset == {'ds_a': 0.001, 'ds_b': 0.001}
        assert std[0].file_stem == 'minratio_0_001'
        assert std[0].raw_thresholds == [0.001]
        assert point_from_value(0.001, params).point_id == 'threshold_0.001'
        # Synapse standard point stays byte-identical.
        from comparison.comparison_parameters import ComparisonParameters
        syn = ComparisonParameters(
            datasets=['ds_a', 'ds_b'], thresholds=[3],
            output_folder=None, verbose=False)
        syn_pts = points_from_parameters(syn)
        assert syn_pts[0].file_stem == 'minsyn_3'
        assert syn_pts[0].raw_thresholds == [3]


def test_bootstrap_ratio_float_dedup_keeps_distinct_rows(monkeypatch):
    """AUTO under the ratio basis: vertical spine and horizontal rows sit on
    DISTINCT float tiers — an int() dedup key collapses them all to 0 and
    silently drops every row after the first (regression: the spine-only
    real run). The bootstrap must also enumerate at the float floor."""
    import tempfile
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['L2'],
        target_neurons=['clock'], thresholds=[0.001],
        threshold_mode='auto', weight_basis='connection_ratio',
        threshold_dataset_order=['ds_a', 'ds_b'],
        output_folder=tempfile.mkdtemp(), verbose=False)
    analyzer = ComparisonAnalyzer(params, verbose=False)
    analyzer.parameters.output_folder = None
    calls = []
    monkeypatch.setattr(
        analyzer, 'run_path_analysis',
        lambda ds, t, verbose_mode='simple': (
            calls.append((ds, t)), pd.DataFrame())[1])
    tiers = [0.0011919, 0.01, 0.1, 0.4]
    monkeypatch.setattr(
        analyzer, '_build_density_curves',
        lambda: (
            {'ds_a': {'thresholds': tiers, 'path_count': [9, 5, 2, 1],
                      'edge_count': [90, 50, 20, 10],
                      'density': [1.0, 0.5, 0.25, 0.1]},
             'ds_b': {'thresholds': tiers, 'path_count': [8, 4, 2, 1],
                      'edge_count': [80, 40, 16, 8],
                      'density': [0.8, 0.4, 0.2, 0.08]}},
            {'ds_a': (0.0011919, 0.4), 'ds_b': (0.0011919, 0.4)},
            {'ds_a': {}, 'ds_b': {}}))

    assert analyzer._bootstrap_auto_mode() is True
    # Bootstrap enumeration at the float floor, not max(1, ...) = 1.
    assert calls == [('ds_a', 0.001), ('ds_b', 0.001)]
    queries = params.get_threshold_queries()
    # Distinct float rows must ALL survive the dedup (int-collapse kept 1).
    assert len(queries) >= 2
    modes = {q.get('row_mode') for q in queries}
    assert 'vertical' in modes and 'horizontal' in modes
    seen_cells = {tuple(sorted(q['thresholds'].items())) for q in queries}
    assert len(seen_cells) == len(queries)  # no two rows share one cell
    for q in queries:
        for v in q['thresholds'].values():
            assert isinstance(v, float) and 0 < v <= 1


def test_ratio_density_loader_and_curves_roundtrip(tmp_path):
    """REGRESSION (real-run audit 2026-10-04): the artifact loader read
    density_edges.npz as int32 — under the ratio basis every sub-1 tier
    truncated to 0, the curve grids collapsed to one point, edge counts
    and densities read 0, and AUTO's horizontal rows silently vanished
    (only the window-driven vertical spine survived). The loader must
    follow the capture's weight_basis, and the curves/CSV must carry the
    float tiers."""
    import json
    import os
    import numpy as np
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer

    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['x'], target_neurons=['y'],
        thresholds=[0.01], threshold_mode='standard',
        threshold_dataset_order=['ds_a', 'ds_b'],
        weight_basis='connection_ratio',
        output_folder=str(tmp_path), drop_untyped=True, verbose=False)

    per_ds = {
        'ds_a': ([0.0105, 0.03],
                 ([0.010, 0.012, 0.02, 0.05], [0, 0, 1, 2]),
                 {'weight_basis': 'connection_ratio', 'applied': 0.01,
                  'w_start': 0.01, 'w_star_measured': 0.03,
                  'n_nodes': 10, 'n_nodes_typed': 4, 'n_nodes_untyped': 2,
                  'n_nodes_debris': 1, 'n_annotated_nodes': 4,
                  'denominator': 'typed_nodes_in_searched_graph',
                  'path_complete_from': 0.01, 'n_paths': 2, 'n_edges': 4,
                  'budget_bitten': False, 'paths_complete': True}),
        'ds_b': ([0.011, 0.02],
                 ([0.011, 0.015, 0.025], [0, 0, 0]),
                 {'weight_basis': 'connection_ratio', 'applied': 0.01,
                  'w_start': 0.01, 'w_star_measured': 0.02,
                  'n_nodes': 8, 'n_nodes_typed': 5, 'n_nodes_untyped': 0,
                  'n_nodes_debris': 0, 'n_annotated_nodes': 5,
                  'denominator': 'typed_nodes_in_searched_graph',
                  'path_complete_from': 0.01, 'n_paths': 2, 'n_edges': 3,
                  'budget_bitten': False, 'paths_complete': True}),
    }
    for ds, (bns, (w, cls), meta) in per_ds.items():
        d = os.path.join(params.dataset_data_path,
                         params._sanitize_name(ds), '_density')
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'density_path_bottlenecks.npy'),
                np.asarray(bns, dtype=float))
        np.savez_compressed(
            os.path.join(d, 'density_edges.npz'),
            weight=np.asarray(w, dtype=np.float64),
            cls=np.asarray(cls, dtype=np.int8))
        json.dump(meta, open(os.path.join(d, 'density_meta.json'), 'w'))

    analyzer = ComparisonAnalyzer(params, verbose=False)

    # Loader dtype follows the capture's basis.
    _meta, _bns, ew_a, _cls = analyzer._load_density_artifacts('ds_a')
    assert ew_a.dtype == np.float64 and ew_a.min() > 0

    curves, windows, metas = analyzer._build_density_curves()
    for ds in ('ds_a', 'ds_b'):
        c = curves[ds]
        # The distinct-tier ladder survived — NOT the int-collapsed
        # single point [0].
        assert len(c['thresholds']) >= 2
        assert all(isinstance(t, float) and t > 0 for t in c['thresholds'])
        assert c['edge_count'][0] > 0          # typed edges at w_start
        assert c['density'][0] > 0
    assert metas['ds_a']['n_edges_active_basis'] == 2   # 2 typed edges

    rows = analyzer._density_aligned_rows(curves, windows, metas)
    modes = {r.get('mode') for r in rows}
    assert 'vertical' in modes
    # With real curves the density-matched horizontal rows must appear
    # (the int32 loader produced zero curves and dropped them all).
    assert any('horizontal' in (m or '') for m in modes), modes

    # The exported CSV keeps the float tiers verbatim.
    cr = os.path.join(params.full_output_path, 'comparison_results')
    os.makedirs(cr, exist_ok=True)
    analyzer._export_density_alignment(cr)
    cur = pd.read_csv(os.path.join(cr, 'density_curves.csv'))
    a_rows = cur[cur.dataset == 'ds_a']
    assert (a_rows.threshold > 0).all()
    assert set(a_rows.threshold.round(6)) >= {0.010, 0.012}
    assert (a_rows.edge_count > 0).any()


def test_f7_extension_points_ratio_and_synapse():
    """F7 float analog: under the ratio basis the k × τ_ref ladder
    multiplies the FLOAT tier and respects the (0, 1] ceiling; the
    synapse integer ladder is unchanged."""
    from comparison.comparison_analyzer import ComparisonAnalyzer
    f7 = ComparisonAnalyzer._f7_extension_points
    pts = f7([0.002, 0.0018], 0.01, ratio_basis=True)
    assert len(pts) == 9
    assert all(abs(a - b) < 1e-12
               for a, b in zip(pts, [k * 0.002 for k in range(2, 11)]))
    assert all(isinstance(p, float) and 0 < p <= 1 for p in pts)
    # The (0, 1] domain cap: tau_ref 0.4 with asked-max 0.9 would allow
    # 1.2 on the 2x rule alone — the ceiling clips it.
    assert f7([0.4], 0.9, ratio_basis=True) == [0.8]
    # Float-tie clamp: a COMPLETE top-tier run's natural tau sits a hair
    # above its tier (0.0106 > 0.01); unclamped, 2 x 0.0106 overshoots
    # the 0.02 cap and the ladder dies. The clamp anchors at the tier.
    assert f7([0.00216, 0.0106], 0.01, ratio_basis=True) == [0.02]
    # Synapse behavior byte-identical (int ladder, 2x cap only).
    assert f7([19.0, 18.0], 50) == [38, 57, 76, 95]
    assert f7([], 10) == []
