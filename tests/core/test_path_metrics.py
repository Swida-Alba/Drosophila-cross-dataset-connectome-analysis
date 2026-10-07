#!/usr/bin/env python
"""
Regression tests for path probability / connection-ratio calculations.

Locks in the project-wide metric contract (see docs/core-features/ScoreCalculation_Guide.md):

- bodyId level:  connection_ratio = w(A->B) / total_incoming(B)  (GLOBAL denominator
  from ALL sources when supplied; local fallback otherwise)
- bodyId level:  traversal_probability = min(ratio / 0.3, 1);  block = 1 - prob
- type level:    ratio = sum(w over deduplicated bodyId pairs) / total_incoming(typeB);
                 prob  = aggregate_method over the deduplicated pairs (default 'product':
                 1 - prod(1 - p_pair), the reliability/OR model; 'average': weight-weighted
                 mean; 'ratio': min(ratio / 0.3, 1))
- path level:    path_prob = product(edge probs);  min_ratio = min(edge ratios);
                 min_weight = min(edge weights);  length = number of edges

The pandas (statvis.EnrichConnectionTable) and Polars
(statvis.EnrichConnectionTablePolars) implementations must agree.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statvis import EnrichConnectionTable  # noqa: E402
from statvis import build_path_dataframe_from_paths as sv_build  # noqa: E402
from statvis import EnrichConnectionTablePolars  # noqa: E402
from statvis import build_path_dataframe_from_paths_polars as svp_build  # noqa: E402


# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------

def _synthetic_conn():
    """BodyId-level table with pre-supplied GLOBAL connection ratios.

    Total incoming per post type (denominator of every ratio):
        X=30, Y=50, Z=10, W=100
    Includes a duplicate bodyId pair (1,10) with a large bogus weight (99) that
    must be dropped by deduplication.
    """
    rows = [
        # bodyId_pre, bodyId_post, type_pre, type_post, weight, connection_ratio
        (1, 10, 'A', 'X', 5, 5 / 30.0),
        (1, 11, 'A', 'X', 4, 4 / 30.0),
        (2, 10, 'A', 'X', 6, 6 / 30.0),
        (1, 10, 'A', 'X', 99, 99 / 30.0),   # duplicate pair -> must be deduped
        (3, 12, 'A', 'Y', 7, 7 / 50.0),
        (4, 12, 'B', 'Y', 9, 9 / 50.0),
        (1, 12, 'A', 'Y', 8, 8 / 50.0),
        (6, 12, 'X', 'Y', 3, 3 / 50.0),     # X->Y edge for a 2-hop path
        (5, 13, 'C', 'Z', 2, 2 / 10.0),
        (7, 14, 'D', 'W', 40, 40 / 100.0),  # ratio 0.4 > 0.3 -> prob capped at 1
    ]
    return pd.DataFrame(rows, columns=[
        'bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight', 'connection_ratio',
    ])


def _global_incoming():
    return pd.DataFrame({
        'type_post': ['X', 'Y', 'Z', 'W'],
        'total_incoming_weight': [30.0, 50.0, 10.0, 100.0],
    })


def _expected_type_metrics():
    """Hand-computed type-level expectations (global denominators).

    prob = product model: 1 - prod(1 - p_pair) over the deduplicated pairs,
    with p_pair = min(ratio_pair / 0.3, 1) from the supplied per-pair ratios:
      A->X: 1 - (4/9)(5/9)(1/3) = 223/243        A->Y: 1 - (8/15)(7/15) = 169/225
      B->Y: 1 - (1 - 0.6) = 0.6                 X->Y: 0.2
      C->Z: 2/3                                 D->W: 1.0 (pair capped)
    """
    return {
        ('A', 'X'): (15.0, 0.5, 223 / 243),
        ('A', 'Y'): (15.0, 0.3, 169 / 225),
        ('B', 'Y'): (9.0, 0.18, 0.6),
        ('X', 'Y'): (3.0, 0.06, 0.2),
        ('C', 'Z'): (2.0, 0.2, 2 / 3),
        ('D', 'W'): (40.0, 0.4, 1.0),          # pair capped at 1
    }


# ---------------------------------------------------------------------------
# BodyId-level metrics
# ---------------------------------------------------------------------------

class TestBodyIdLevelMetrics:
    def test_pandas_preserves_global_ratio_and_computes_prob(self):
        conn_df, _, _ = EnrichConnectionTable(
            _synthetic_conn(), traversal_probability_threshold=0,
            global_incoming_weights=_global_incoming(),
        )
        # bodyId_post is stringified by the enrichment; match as string
        row = conn_df[(conn_df['bodyId_pre'] == 1) & (conn_df['bodyId_post'] == '10')].iloc[0]
        assert row['connection_ratio'] == pytest.approx(5 / 30)
        assert row['traversal_probability'] == pytest.approx((5 / 30) / 0.3)
        assert row['block_probability'] == pytest.approx(1 - (5 / 30) / 0.3)

    def test_probability_capped_at_1(self):
        conn_df, _, _ = EnrichConnectionTable(
            _synthetic_conn(), traversal_probability_threshold=0,
            global_incoming_weights=_global_incoming(),
        )
        # D->W bodyId row has ratio 0.4 -> prob 1.333 -> capped to 1.0
        row = conn_df[(conn_df['bodyId_pre'] == 7) & (conn_df['bodyId_post'] == '14')].iloc[0]
        assert row['traversal_probability'] == pytest.approx(1.0)

    def test_polars_bodyid_level_matches_pandas(self):
        conn_df_pd, _, _ = EnrichConnectionTable(
            _synthetic_conn(), traversal_probability_threshold=0,
            global_incoming_weights=_global_incoming(),
        )
        conn_df_pl, _, _ = EnrichConnectionTablePolars(
            _synthetic_conn(), traversal_probability_threshold=0,
            global_incoming_weights=_global_incoming(),
        )
        # PROBABILITY SUSPENSION (2026-10-06): the polars engine no longer
        # exports traversal_probability/block_probability on the bodyId frame.
        assert 'traversal_probability' not in conn_df_pl.columns
        assert 'block_probability' not in conn_df_pl.columns
        pl_pd = conn_df_pl.to_pandas().sort_values(['bodyId_pre', 'bodyId_post']).reset_index(drop=True)
        pd_pd = conn_df_pd.sort_values(['bodyId_pre', 'bodyId_post']).reset_index(drop=True)
        for col in ['weight', 'connection_ratio']:
            pd.testing.assert_series_equal(
                pl_pd[col], pd_pd[col], check_dtype=False,
                obj=f'bodyId-level {col}',
            )


# ---------------------------------------------------------------------------
# Type-level aggregation
# ---------------------------------------------------------------------------

class TestTypeLevelAggregation:
    def _type_table(self, impl):
        if impl == 'pandas':
            _, conn_type, _ = EnrichConnectionTable(
                _synthetic_conn(), traversal_probability_threshold=0,
                global_incoming_weights=_global_incoming(),
            )
            return conn_type.rename(columns={
                'type_pre': 't_pre', 'type_post': 't_post'})
        _, conn_type, _ = EnrichConnectionTablePolars(
            _synthetic_conn(), traversal_probability_threshold=0,
            global_incoming_weights=_global_incoming(),
        )
        return conn_type.rename({
            'type_pre': 't_pre', 'type_post': 't_post'}).to_pandas()

    @pytest.mark.parametrize('impl', ['pandas', 'polars'])
    def test_global_ratio_and_probability(self, impl):
        ct = self._type_table(impl)
        # PROBABILITY SUSPENSION (2026-10-06): the suspended probability
        # columns are no longer exported (values no longer asserted); the
        # aggregate math itself is exercised through the F9 threshold filter
        # elsewhere.
        assert 'traversal_probability' not in ct.columns
        assert 'block_probability' not in ct.columns
        for (pre, post), (w, ratio, prob) in _expected_type_metrics().items():
            row = ct[(ct['t_pre'] == pre) & (ct['t_post'] == post)]
            assert len(row) == 1, f'missing type pair {pre}->{post} in {impl}'
            row = row.iloc[0]
            assert row['weight'] == pytest.approx(w), f'weight {pre}->{post}'
            assert row['connection_ratio'] == pytest.approx(ratio), f'ratio {pre}->{post}'

    @pytest.mark.parametrize('impl', ['pandas', 'polars'])
    def test_duplicate_bodyid_pair_is_deduplicated(self, impl):
        # The (1,10) pair appears twice (weight 5 and bogus 99); only the
        # first counts, so A->X weight must be 15, not 114.
        ct = self._type_table(impl)
        row = ct[(ct['t_pre'] == 'A') & (ct['t_post'] == 'X')].iloc[0]
        assert row['weight'] == pytest.approx(15.0)

    def test_pandas_matches_polars_with_global_weights(self):
        ct_pd = self._type_table('pandas').sort_values(['t_pre', 't_post']).reset_index(drop=True)
        ct_pl = self._type_table('polars').sort_values(['t_pre', 't_post']).reset_index(drop=True)
        for col in ['weight', 'connection_ratio']:
            pd.testing.assert_series_equal(
                ct_pl[col], ct_pd[col], check_dtype=False,
                obj=f'type-level {col}',
            )

    def test_local_fallback_matches_across_implementations(self):
        """Without global weights both implementations use the local
        denominator (connections in this table only) and still agree."""
        _, ct_pd, _ = EnrichConnectionTable(
            _synthetic_conn(), traversal_probability_threshold=0)
        _, ct_pl, _ = EnrichConnectionTablePolars(
            _synthetic_conn(), traversal_probability_threshold=0)
        ct_pd = ct_pd.sort_values(['type_pre', 'type_post']).reset_index(drop=True)
        ct_pl = ct_pl.sort(['type_pre', 'type_post']).to_pandas().reset_index(drop=True)
        for col in ['weight', 'connection_ratio']:
            pd.testing.assert_series_equal(
                ct_pl[col], ct_pd[col], check_dtype=False,
                obj=f'local-fallback {col}',
            )

    def test_missing_global_denominator_falls_back_to_local(self):
        """A post type absent from the global incoming table falls back to the
        LOCAL total over this table (never NaN/0) in both implementations, so
        paths through untyped/unlisted post neurons keep a positive ratio and
        probability."""
        conn = pd.DataFrame({
            'bodyId_pre': [1], 'bodyId_post': [20],
            'type_pre': ['E'], 'type_post': ['Q'],
            'weight': [5], 'connection_ratio': [np.nan],
        })
        global_incoming = pd.DataFrame({'type_post': ['X'], 'total_incoming_weight': [10.0]})
        _, ct_pd, _ = EnrichConnectionTable(
            conn.copy(), traversal_probability_threshold=0,
            global_incoming_weights=global_incoming)
        _, ct_pl, _ = EnrichConnectionTablePolars(
            conn.copy(), traversal_probability_threshold=0,
            global_incoming_weights=global_incoming)
        # Local fallback: the table's own total incoming to Q is 5 -> ratio 1.0
        assert ct_pd['connection_ratio'].iloc[0] == pytest.approx(1.0)
        assert ct_pl.to_pandas()['connection_ratio'].iloc[0] == pytest.approx(1.0)
        # PROBABILITY SUSPENSION (2026-10-06): probabilities are no longer
        # exported.
        assert 'traversal_probability' not in ct_pd.columns
        assert 'traversal_probability' not in ct_pl.columns

    @pytest.mark.parametrize('impl', ['pandas', 'polars'])
    def test_aggregate_method_param_still_accepted(self, impl):
        """'product'/'average'/'ratio' are accepted and still run; the
        suspended probability VALUES they used to export are no longer
        asserted (PROBABILITY SUSPENSION 2026-10-06) — the 'ratio' model's
        legacy min(ratio / 0.3, 1) math is exercised through the F9
        threshold filter in the pipeline tests."""
        if impl == 'pandas':
            _, ct_product, _ = EnrichConnectionTable(
                _synthetic_conn(), traversal_probability_threshold=0,
                aggregate_method='product', global_incoming_weights=_global_incoming())
            _, ct_ratio, _ = EnrichConnectionTable(
                _synthetic_conn(), traversal_probability_threshold=0,
                aggregate_method='ratio', global_incoming_weights=_global_incoming())
            cols = ['type_pre', 'type_post', 'weight', 'connection_ratio']
        else:
            _, ct_product, _ = EnrichConnectionTablePolars(
                _synthetic_conn(), traversal_probability_threshold=0,
                aggregate_method='product', global_incoming_weights=_global_incoming())
            _, ct_ratio, _ = EnrichConnectionTablePolars(
                _synthetic_conn(), traversal_probability_threshold=0,
                aggregate_method='ratio', global_incoming_weights=_global_incoming())
            ct_product = ct_product.to_pandas()
            ct_ratio = ct_ratio.to_pandas()
            cols = ['weight', 'connection_ratio']
        assert len(ct_product) == len(ct_ratio)
        # PROBABILITY SUSPENSION (2026-10-06): no probability columns in
        # either method's export.
        for ct in (ct_product, ct_ratio):
            assert 'traversal_probability' not in ct.columns
            assert 'block_probability' not in ct.columns
        # Both methods still emit the same schema
        for col in cols:
            assert col in ct_product.columns


# ---------------------------------------------------------------------------
# EnrichConnectionTable engine parity (unified entry point)
# ---------------------------------------------------------------------------

def _parity_conn():
    """Connection table with nt_type + custom groups and mixed NT on A->X
    (ACH twice vs GABA once -> mode must be ACH in both engines)."""
    rows = [
        (1, 10, 'A', 'X', 5, 'ACH', 'G1', 'GX', 5 / 30.0),
        (1, 11, 'A', 'X', 4, 'ACH', 'G1', 'GX', 4 / 30.0),
        (2, 10, 'A', 'X', 6, 'GABA', 'G1', 'GX', 6 / 30.0),
        (3, 12, 'A', 'Y', 7, 'ACH', 'G1', 'GY', 7 / 50.0),
        (4, 12, 'B', 'Y', 9, 'GLUT', 'G2', 'GY', 9 / 50.0),
        (5, 13, 'C', 'Z', 2, 'GABA', 'G3', 'GZ', 2 / 10.0),
    ]
    return pd.DataFrame(rows, columns=[
        'bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight', 'nt_type',
        'custom_group_pre', 'custom_group_post', 'connection_ratio'])


class TestEnrichmentEngineParity:
    @pytest.mark.parametrize('global_weights', [False, True])
    def test_type_level_parity(self, global_weights):
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = _parity_conn()
        gw = _global_incoming() if global_weights else None
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0,
                                global_incoming_weights=gw)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn),
                                                  traversal_probability_threshold=0,
                                                  global_incoming_weights=gw)
        ct_pl = ct_pl.to_pandas().sort_values(['type_pre', 'type_post']).reset_index(drop=True)
        ct_pd = ct_pd.sort_values(['type_pre', 'type_post']).reset_index(drop=True)
        assert set(ct_pl.columns) == set(ct_pd.columns), (
            f'type-level schema diverged: {sorted(ct_pl.columns)} vs {sorted(ct_pd.columns)}')
        for col in ct_pd.columns:
            pd.testing.assert_series_equal(ct_pl[col], ct_pd[col], check_dtype=False, obj=col)
        # nt_type uses mode on ties (ACH twice vs GABA once)
        row = ct_pd[(ct_pd['type_pre'] == 'A') & (ct_pd['type_post'] == 'X')].iloc[0]
        assert row['nt_type'] == 'ACH'

    def test_nt_type_tie_breaks_identically_across_engines(self):
        """A true mode tie (ACH x2 vs GABA x2) must resolve to the same value
        in both engines - lexicographically first (ACH) - regardless of row
        order (regression: order-dependent mode() tie-breaking diverged on
        ~5% of type pairs in large tables)."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        rows = [
            (1, 10, 'A', 'X', 5, 'GABA', 5 / 30.0),
            (1, 11, 'A', 'X', 4, 'ACH', 4 / 30.0),
            (2, 10, 'A', 'X', 6, 'ACH', 6 / 30.0),
            (2, 11, 'A', 'X', 3, 'GABA', 3 / 30.0),   # tie: ACH x2, GABA x2
        ]
        conn = pd.DataFrame(rows, columns=['bodyId_pre', 'bodyId_post', 'type_pre',
                                           'type_post', 'weight', 'nt_type', 'connection_ratio'])
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn),
                                                  traversal_probability_threshold=0)
        nt_pd = ct_pd[(ct_pd['type_pre'] == 'A') & (ct_pd['type_post'] == 'X')]['nt_type'].iloc[0]
        nt_pl = ct_pl.filter((pl.col('type_pre') == 'A') & (pl.col('type_post') == 'X'))['nt_type'].item()
        assert nt_pd == nt_pl == 'ACH'

    def test_group_level_parity_and_naming(self):
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = _parity_conn()
        _, _, cg_pd = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        _, _, cg_pl = EnrichConnectionTablePolars(pl.from_pandas(conn),
                                                  traversal_probability_threshold=0)
        assert cg_pd is not None and cg_pl is not None
        assert set(cg_pl.columns) == set(cg_pd.columns)
        # both engines use custom_group_pre/custom_group_post (unified schema)
        assert 'custom_group_pre' in cg_pd.columns and 'group_pre' not in cg_pd.columns
        cg_pl = cg_pl.to_pandas().sort_values(['custom_group_pre', 'custom_group_post']).reset_index(drop=True)
        cg_pd = cg_pd.sort_values(['custom_group_pre', 'custom_group_post']).reset_index(drop=True)
        for col in cg_pd.columns:
            pd.testing.assert_series_equal(cg_pl[col], cg_pd[col], check_dtype=False, obj=f'group {col}')

    def test_block_probability_missing_denominator_fallback(self):
        """Missing global denominator -> LOCAL fallback ratio (1.0 here) in
        BOTH engines (no more NaN/0 collapse). PROBABILITY SUSPENSION
        (2026-10-06): the probability columns are no longer exported, so the
        former prob-1.0/block-0.0 assertions became absence checks."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = pd.DataFrame({
            'bodyId_pre': [1], 'bodyId_post': [20],
            'type_pre': ['E'], 'type_post': ['Q'],
            'weight': [5], 'connection_ratio': [np.nan],
        })
        gw = pd.DataFrame({'type_post': ['X'], 'total_incoming_weight': [10.0]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0,
                                global_incoming_weights=gw)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn),
                                                  traversal_probability_threshold=0,
                                                  global_incoming_weights=gw)
        assert ct_pd['connection_ratio'].iloc[0] == pytest.approx(1.0)
        assert 'traversal_probability' not in ct_pd.columns
        assert 'block_probability' not in ct_pd.columns
        ct_pl = ct_pl.to_pandas()
        assert ct_pl['connection_ratio'].iloc[0] == pytest.approx(1.0)
        assert 'traversal_probability' not in ct_pl.columns
        assert 'block_probability' not in ct_pl.columns

    def test_bodyid_level_shared_schema(self):
        """BodyId-level outputs share every column; std_label_pre/post are
        polars-engine internals (the pandas engine overwrites type_pre/post
        with the mapped labels instead)."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = _parity_conn()
        conn_df_pd, _, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        conn_df_pl, _, _ = EnrichConnectionTablePolars(pl.from_pandas(conn),
                                                       traversal_probability_threshold=0)
        shared = set(conn_df_pd.columns) & set(conn_df_pl.columns)
        assert 'std_label_pre' in conn_df_pl.columns and 'std_label_pre' not in conn_df_pd.columns
        pl_pd = conn_df_pl.to_pandas().sort_values(['bodyId_pre', 'bodyId_post']).reset_index(drop=True)
        pd_pd = conn_df_pd.sort_values(['bodyId_pre', 'bodyId_post']).reset_index(drop=True)
        assert len(pl_pd) == len(pd_pd)
        for col in sorted(shared):
            # bodyId_pre may be int (pandas, no-local fallback) vs str (polars)
            left = pl_pd[col].astype(str) if col == 'bodyId_pre' else pl_pd[col]
            right = pd_pd[col].astype(str) if col == 'bodyId_pre' else pd_pd[col]
            pd.testing.assert_series_equal(left, right, check_dtype=False, obj=col)

    def test_unified_dispatcher_routes_by_input_and_engine(self):
        from statvis import EnrichConnectionTable
        conn = _parity_conn()
        # polars input -> polars engine
        _, ct, _ = EnrichConnectionTable(pl.from_pandas(conn), traversal_probability_threshold=0)
        assert isinstance(ct, pl.DataFrame)
        # pandas input -> pandas engine
        _, ct, _ = EnrichConnectionTable(conn.copy(), traversal_probability_threshold=0)
        assert isinstance(ct, pd.DataFrame)
        # forced engines convert the input frame
        _, ct, _ = EnrichConnectionTable(conn.copy(), traversal_probability_threshold=0, engine='polars')
        assert isinstance(ct, pl.DataFrame)
        _, ct, _ = EnrichConnectionTable(pl.from_pandas(conn), traversal_probability_threshold=0, engine='pandas')
        assert isinstance(ct, pd.DataFrame)

    def test_empty_inputs_return_typed_empty_outputs(self):
        """Regression: both engines used to raise KeyError/ColumnNotFound/
        InvalidOperation for empty or column-less tables."""
        from statvis import EnrichConnectionTable
        for frame in (pd.DataFrame(),
                      pl.DataFrame(),
                      pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']),
                      pl.DataFrame(schema={'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8, 'weight': pl.Int64})):
            conn_df, conn_type, conn_group = EnrichConnectionTable(frame, traversal_probability_threshold=0)
            assert len(conn_df) == 0 and len(conn_type) == 0 and conn_group is None

    def test_local_csv_reads_are_robust(self, tmp_path):
        """Regression: the pandas engine hard-coded index_col=0 and crashed on
        standard (bodyId-first) CSVs with KeyError 'bodyId', and on CSVs
        without a 'post' column. All three formats must work like Polars."""
        from statvis import EnrichConnectionTable
        neurons = pd.DataFrame({'bodyId': ['1', '2', '10', '11'],
                                'type': ['A', 'A', 'X', 'X'], 'post': [100, 90, 300, 280]})
        conn = pd.DataFrame({'bodyId_pre': [1, 2], 'bodyId_post': [10, 11],
                             'type_pre': ['A', 'A'], 'type_post': ['X', 'X'],
                             'weight': [5, 4], 'connection_ratio': [5 / 30, 4 / 30]})
        for name, write in [('ds_std', lambda p: neurons.to_csv(p, index=False)),
                            ('ds_legacy', lambda p: neurons.set_index('bodyId').to_csv(p)),
                            ('ds_nopost', lambda p: neurons.drop(columns=['post']).to_csv(p, index=False))]:
            d = tmp_path / 'datasets' / name
            d.mkdir(parents=True, exist_ok=True)
            write(d / f'{name}_allneurons_neuron_df.csv')
            conn_df, _, _ = EnrichConnectionTable(conn.copy(), traversal_probability_threshold=0,
                                                  dataset=f'{name.replace("_", ":")}', script_path=str(tmp_path))
            # post: real value for std/legacy, 0 for nopost (polars parity)
            expected = 0 if name == 'ds_nopost' else 300
            assert conn_df['post'].iloc[0] == expected, name

    def test_untyped_neurons_group_by_bodyid_in_both_engines(self):
        """Untyped neurons (null/empty type) must group per-bodyId in BOTH
        engines (Polars std_label semantics) - never lumped into one mixed
        'Unknown' group (regression of the pandas engine's old behavior)."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = pd.DataFrame({'bodyId_pre': [1, 2], 'bodyId_post': [10, 11],
                             'type_pre': [None, None], 'type_post': ['X', 'X'],
                             'weight': [5, 4], 'connection_ratio': [5 / 30, 4 / 30]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn), traversal_probability_threshold=0)
        pairs_pd = sorted(zip(list(ct_pd['type_pre']), list(ct_pd['type_post'])))
        pairs_pl = sorted(zip(list(ct_pl['type_pre']), list(ct_pl['type_post'])))
        assert pairs_pd == pairs_pl == [('1', 'X'), ('2', 'X')]
        assert 'Unknown' not in list(ct_pd['type_pre'])

    def test_pre_existing_unknown_type_is_preserved(self):
        """A literal 'Unknown' type in the input is a real type value and is
        preserved by both engines (only null/empty cells become per-bodyId
        groups)."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = pd.DataFrame({'bodyId_pre': [1, 2], 'bodyId_post': [10, 11],
                             'type_pre': ['Unknown', 'Unknown'], 'type_post': ['X', 'X'],
                             'weight': [5, 4], 'connection_ratio': [5 / 30, 4 / 30]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn), traversal_probability_threshold=0)
        pairs_pd = sorted(zip(list(ct_pd['type_pre']), list(ct_pd['type_post'])))
        pairs_pl = sorted(zip(list(ct_pl['type_pre']), list(ct_pl['type_post'])))
        assert pairs_pd == pairs_pl == [('Unknown', 'X')]

    def test_mixed_typed_and_untyped_parity(self):
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        conn = pd.DataFrame({'bodyId_pre': [1, 2, 3], 'bodyId_post': [10, 11, 12],
                             'type_pre': [None, 'A', 'A'], 'type_post': ['X', 'X', 'Y'],
                             'weight': [5, 4, 9], 'connection_ratio': [5 / 30, 4 / 30, 9 / 50]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn), traversal_probability_threshold=0)
        pairs_pd = sorted(zip(list(ct_pd['type_pre']), list(ct_pd['type_post'])))
        pairs_pl = sorted(zip(list(ct_pl['type_pre']), list(ct_pl['type_post'])))
        assert pairs_pd == pairs_pl == [('1', 'X'), ('A', 'X'), ('A', 'Y')]

    def test_label_mapper_parity(self, tmp_path):
        """With a real LabelMapper + local dataset, both engines must produce
        the same mapped type-level output."""
        from statvis import EnrichConnectionTable as pd_enrich
        from statvis import EnrichConnectionTablePolars
        from comparison.label_mapper import LabelMapper
        d = tmp_path / 'datasets' / 'ds_map'
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'bodyId': [1, 2, 10, 11, 20], 'type': ['A', 'A', 'X', 'X', 'M'],
                      'post': [100, 90, 300, 280, 500]}).to_csv(
            d / 'ds_map_allneurons_neuron_df.csv', index=False)
        lm = LabelMapper(source_mapping_dict={'ds:map': [[1, 2]]},
                         target_mapping_dict={'ds:map': [[10, 11]]},
                         intermediate_mapping_dict={'ds:map': [[20]]},
                         source_labels=['SRC'], target_labels=['TGT'], intermediate_labels=['MID'])
        conn = pd.DataFrame({'bodyId_pre': [1, 2, 20], 'bodyId_post': [10, 11, 10],
                             'type_pre': ['A', 'A', 'M'], 'type_post': ['X', 'X', 'X'],
                             'weight': [5, 4, 7], 'connection_ratio': [5 / 30, 4 / 30, 7 / 30]})
        _, ct_pd, _ = pd_enrich(conn.copy(), traversal_probability_threshold=0,
                                dataset='ds:map', script_path=str(tmp_path), label_mapper=lm)
        _, ct_pl, _ = EnrichConnectionTablePolars(pl.from_pandas(conn), traversal_probability_threshold=0,
                                                  dataset='ds:map', script_path=str(tmp_path), label_mapper=lm)
        pairs_pd = sorted(zip(ct_pd['type_pre'], ct_pd['type_post']))
        pairs_pl = sorted(zip(ct_pl['type_pre'].to_list(), ct_pl['type_post'].to_list()))
        assert pairs_pd == pairs_pl == [('MID', 'TGT'), ('SRC', 'TGT')]


# ---------------------------------------------------------------------------
# Unified builders (build_path_dataframe_from_paths / process_paths_streaming)
# ---------------------------------------------------------------------------

class TestUnifiedBuilders:
    def test_build_path_dataframe_dispatches_by_input_type(self):
        from statvis import build_path_dataframe_from_paths as sv_build
        conn = _enriched_type_table()          # polars
        paths = [['A', 'X', 'Y'], ['B', 'Y']]
        targets = ['Y']
        df_pl = sv_build(paths, conn, targets, real_layer_map=None, level='type')
        assert isinstance(df_pl, pl.DataFrame)
        df_pd = sv_build(paths, conn.to_pandas(), targets, real_layer_map=None, level='type')
        assert isinstance(df_pd, pd.DataFrame)
        assert len(df_pl) == len(df_pd)
        # forced engines convert the input frame
        df_pd2 = sv_build(paths, conn, targets, real_layer_map=None, level='type', engine='pandas')
        assert isinstance(df_pd2, pd.DataFrame)
        df_pl2 = sv_build(paths, conn.to_pandas(), targets, real_layer_map=None, level='type', engine='polars')
        assert isinstance(df_pl2, pl.DataFrame)

    def test_pandas_builder_nt_type_join_semantics(self):
        """The pandas engine's edge lookup aggregates nt_type as a '|'-joined
        sorted unique set of non-empty values ('Unknown' when none) - the
        semantics of the legacy _unique_nt_types helper, now computed with
        Polars."""
        from statvis import build_path_dataframe_from_paths as sv_build
        conn = pd.DataFrame({
            'bodyId_pre': ['1', '1', '3', '4', '5'],
            'bodyId_post': ['10', '11', '12', '12', '13'],
            'type_pre': ['A', 'A', 'B', 'B', 'C'],
            'type_post': ['X', 'X', 'Y', 'Y', 'Z'],
            'weight': [5, 4, 9, 1, 2],
            'traversal_probability': [0.5, 0.5, 0.8, 0.8, 0.9],
            'connection_ratio': [0.1, 0.1, 0.2, 0.2, 0.3],
            'nt_type': ['ACH', 'GABA', 'GABA', 'None', None],
        })
        out = sv_build([['A', 'X'], ['B', 'Y'], ['C', 'Z']], conn, ['X', 'Y', 'Z'],
                       real_layer_map=None, level='type')
        # A->X: sorted unique of {ACH, GABA}; B->Y: 'None' string dropped;
        # C->Z: only nulls -> 'Unknown' fallback
        assert out['nt_types'].tolist() == [['ACH|GABA'], ['GABA'], ['Unknown']]

    def test_process_paths_streaming_unified_delegates_to_polars(self, tmp_path):
        from statvis import process_paths_streaming as sv_stream
        from statvis import process_paths_streaming as svp_stream
        conn = _enriched_type_table()
        paths = [['A', 'X', 'Y'], ['B', 'Y']]
        out1 = tmp_path / 'sv_stream.csv'
        out2 = tmp_path / 'svp_stream.csv'
        n1 = sv_stream(iter(paths), conn, ['Y'], str(out1), verbose=False)
        n2 = svp_stream(iter(paths), conn, ['Y'], str(out2), verbose=False)
        assert n1 == n2 == 2
        assert out1.read_text() == out2.read_text()

    def test_csv_exports_keep_synapse_counts_integer_valued(self, tmp_path):
        """Counts stay integer-formatted while ratio/probability fields stay decimal."""
        from coana import FindNeuronConnection
        from statvis import process_paths_streaming as sv_stream

        # Exercise the shared edge-table CSV writer with an upstream Float64
        # weight column, which is the original 5.0/6.0 failure mode.
        finder = object.__new__(FindNeuronConnection)
        edge_path = tmp_path / 'edges.csv'
        edge_df = pl.DataFrame({
            'type_pre': ['A'],
            'type_post': ['X'],
            'weight': [5.0],
            'connection_ratio': [0.25],
            'traversal_probability': [0.75],
        })
        finder._save_df_to_csv_polars(edge_df, str(edge_path))
        edge_text = edge_path.read_text()
        assert '5.0' not in edge_text
        edge_read = pl.read_csv(edge_path)
        assert edge_read.schema['weight'] == pl.Int64
        assert edge_read['weight'].to_list() == [5]
        assert edge_read['connection_ratio'].to_list() == pytest.approx([0.25])

        # Exercise the streaming path writer, including the list-valued
        # ``weights`` field and scalar ``min_weight`` field.
        path_path = tmp_path / 'paths.csv'
        conn_float = _enriched_type_table().with_columns(
            pl.col('weight').cast(pl.Float64)
        )
        assert sv_stream(
            iter([['A', 'X', 'Y']]), conn_float, ['Y'], str(path_path), verbose=False
        ) == 1
        path_read = pl.read_csv(path_path)
        assert path_read['weights'].to_list() == ['[15, 3]']
        assert path_read['min_weight'].to_list() == [3]
        assert '15.0' not in path_path.read_text()


# ---------------------------------------------------------------------------
# Path-level metrics
# ---------------------------------------------------------------------------

def _enriched_type_table():
    """Type-level table from the Polars enrichment with global weights, with
    the type_pre/type_post columns the path builders expect."""
    _, conn_type, _ = EnrichConnectionTablePolars(
        _synthetic_conn(), traversal_probability_threshold=0,
        global_incoming_weights=_global_incoming(),
    )
    return conn_type


class TestPathLevelMetrics:
    def _build(self, builder, conn_data):
        paths = [['A', 'X', 'Y'], ['B', 'Y']]
        targets = ['Y']
        return builder(paths, conn_data, targets, real_layer_map=None, level='type')

    def test_hand_computed_path_metrics(self):
        """min_ratio/min_weight = weakest edge; length = number of edges.
        PROBABILITY SUSPENSION (2026-10-06): the per-hop probabilities and
        the product path_prob are no longer exported (formerly
        probabilities [223/243, 0.2] and path_prob (223/243)*0.2)."""
        df = self._build(sv_build, _enriched_type_table().to_pandas())
        assert 'probabilities' not in df.columns
        assert 'path_prob' not in df.columns
        row_2hop = df[df['path'] == 'A->X->Y'].iloc[0]
        assert row_2hop['weights'] == pytest.approx([15.0, 3.0])
        assert row_2hop['ratios'] == pytest.approx([0.5, 0.06])
        assert row_2hop['min_weight'] == pytest.approx(3.0)
        assert row_2hop['min_ratio'] == pytest.approx(0.06)
        assert row_2hop['length'] == 2
        row_1hop = df[df['path'] == 'B->Y'].iloc[0]
        assert row_1hop['min_ratio'] == pytest.approx(0.18)
        assert row_1hop['min_weight'] == pytest.approx(9.0)
        assert row_1hop['length'] == 1

    def test_pandas_and_polars_path_builders_agree(self):
        df_pd = self._build(sv_build, _enriched_type_table().to_pandas())
        df_pl = self._build(svp_build, _enriched_type_table())
        assert len(df_pl) == len(df_pd)
        df_pd = df_pd.sort_values('path').reset_index(drop=True)
        df_pl = df_pl.sort('path').to_pandas().reset_index(drop=True)
        for col in ['path', 'min_weight', 'min_ratio', 'length']:
            pd.testing.assert_series_equal(
                df_pl[col], df_pd[col], check_dtype=False, obj=f'path {col}',
            )
        # List-valued columns: pandas keeps lists, polars stringifies them
        for col in ('weights', 'ratios'):
            assert isinstance(df_pd[col].iloc[0], list)
            assert isinstance(df_pl[col].iloc[0], str)

    def test_zero_weight_edge_divergence_is_documented(self):
        """Polars drops paths with any zero-weight edge; pandas keeps them
        (coana filters them downstream after getAllPath - see
        docs/core-features/PathAnalysis_ZeroWeightFilter.md)."""
        conn_type = pd.DataFrame({
            'type_pre': ['X', 'X'], 'type_post': ['Y', 'Z'],
            'weight': [5, 0],
            'connection_ratio': [0.2, 0.0],
            'traversal_probability': [2 / 3, 0.0],
        })
        paths = [['X', 'Y'], ['X', 'Z']]
        df_pd = sv_build(paths, conn_type, ['Y', 'Z'], real_layer_map=None, level='type')
        df_pl = svp_build(paths, pl.from_pandas(conn_type), ['Y', 'Z'],
                          real_layer_map=None, level='type')
        assert len(df_pd) == 2                       # pandas keeps the zero-weight path
        assert df_pd[df_pd['path'] == 'X->Z']['min_weight'].iloc[0] == 0
        assert len(df_pl) == 1                       # polars filters it
        assert (df_pl.to_pandas()['min_weight'] > 0).all()
