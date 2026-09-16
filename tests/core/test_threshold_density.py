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
