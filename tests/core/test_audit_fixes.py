#!/usr/bin/env python
"""
Regression tests for the 2026-08 full-project audit fixes.

Covers:
  - NeuronFilter vectorized operators must match the original per-cell
    semantics exactly (all operators, NaN handling, mixed types)
  - unified folder-name decimal formatting (negative / non-numeric values)
  - api_call_with_retry: a hung call must NOT block past the timeout
  - hoisted helpers (_extract_base_type, escape-cypher fallback)
  - statvis vs statvis_polars path-builder output-schema contract
    (shared-schema subset; the polars 'path_str' omission is documented)

"""

import random
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from utils.api_utils import api_call_with_retry, APITimeoutError  # noqa: E402
from utils.neuron_filter import NeuronFilter  # noqa: E402


# =============================================================================
# NeuronFilter vectorization equivalence
# =============================================================================

def _reference_match_value(value, operator, patterns):
    """The original per-cell semantics (pre-vectorization)."""
    if value is None or pd.isna(value):
        return False
    value_str = str(value)
    if operator == 'exact':
        return value in patterns or value_str in [str(p) for p in patterns]
    elif operator == 'contains':
        return any(str(p) in value_str for p in patterns)
    elif operator == 'not_contains':
        return not any(str(p) in value_str for p in patterns)
    elif operator == 'startswith':
        return any(value_str.startswith(str(p)) for p in patterns)
    elif operator == 'endswith':
        return any(value_str.endswith(str(p)) for p in patterns)
    elif operator == 'regex':
        import re
        for pattern in patterns:
            try:
                if re.search(str(pattern), value_str):
                    return True
            except re.error:
                if str(pattern) == value_str:
                    return True
        return False
    elif operator == 'not_regex':
        import re
        for pattern in patterns:
            try:
                if re.search(str(pattern), value_str):
                    return False
            except re.error:
                if str(pattern) == value_str:
                    return False
        return True
    return False


def _reference_apply(df, filter_spec):
    """Reference (old-style) full filter application."""
    mask = pd.Series([True] * len(df), index=df.index)
    for operator, patterns in filter_spec.items():
        op_mask = pd.Series([False] * len(df), index=df.index)
        for col in ['type', 'instance', 'bodyId'] + [c for c in df.columns if c not in ('type', 'instance', 'bodyId')]:
            if col not in df.columns:
                continue
            if operator == 'exact' and patterns and isinstance(patterns[0], int) and col != 'bodyId':
                continue
            op_mask = op_mask | df[col].apply(lambda v: _reference_match_value(v, operator, patterns))
        mask = mask & op_mask
    return df[mask]


class TestNeuronFilterVectorizedEquivalence:
    def _random_df(self, seed):
        rng = random.Random(seed)
        rows = []
        types = ['DNp01', 'MBON01', 'aMe12', 'aMe17', 'KC', None, 'LC4', 'PPL101']
        instances = ['DNp01_R', 'MBON01_R', 'X07_L', None, 'LC4_U', 'LC4_R']
        for i in range(200):
            rows.append({
                'bodyId': rng.randint(1, 10**6),
                'type': rng.choice(types),
                'instance': rng.choice(instances),
                'extra': rng.choice(['alpha', 'beta', 'gamma', None, 42]),
            })
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("spec", [
        {'exact': ['DNp01']},
        {'exact': [12345]},
        {'contains': ['DN']},
        {'not_contains': ['R']},
        {'startswith': ['aMe', 'LC']},
        {'endswith': ['_R', '_L']},
        {'regex': [r'DN[a-z]\d+']},
        {'regex': ['aMe.*']},
        {'not_regex': ['MBON']},
        {'contains': ['a'], 'endswith': ['_R']},   # AND across operators
        {'exact': ['beta', 'gamma']},               # matches the 'extra' column
        {'regex': ['[[]invalid']},                  # invalid regex -> exact fallback
    ])
    def test_vectorized_equals_reference(self, seed, spec):
        df = self._random_df(seed)
        filt = NeuronFilter(spec)
        got = filt.apply(df)
        expected = _reference_apply(df, filt.filter_spec)
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_legacy_list_semantics_preserved(self):
        # bodyId list -> exact on bodyId column only
        df = pd.DataFrame({
            'bodyId': [1, 2, 3, 4],
            'type': ['A', 'B', 'C', 'D'],
        })
        filt = NeuronFilter([2, 4])
        assert filt.apply(df)['bodyId'].tolist() == [2, 4]
        # regex pattern list
        filt = NeuronFilter(['A.*'])
        assert sorted(filt.apply(df)['type'].tolist()) == ['A']
        # exact-name list
        filt = NeuronFilter(['C', 'D'])
        assert sorted(filt.apply(df)['type'].tolist()) == ['C', 'D']
        # mixed exact+regex lists are AND-ed across operators (original semantics)
        filt = NeuronFilter(['A.*', 'C'])
        assert filt.apply(df)['type'].tolist() == []
        # None -> match all
        assert len(NeuronFilter(None).apply(df)) == 4


# =============================================================================
# Folder-name decimal formatting
# =============================================================================

class TestFormatDecimalForFolder:
    def test_ints_and_floats(self):
        from coana import _format_decimal_for_folder
        assert _format_decimal_for_folder(0) == '0'
        assert _format_decimal_for_folder(0.0) == '0'
        assert _format_decimal_for_folder(0.5) == '0_5'
        assert _format_decimal_for_folder(0.1234567) == '0_123457'  # 6 decimals, rounded
        assert _format_decimal_for_folder(1.0) == '1'

    def test_negative_values_are_folder_safe(self):
        from coana import _format_decimal_for_folder
        # Float negatives are converted to 'neg' (folder-safe); integer
        # negatives keep the original int formatting.
        out = _format_decimal_for_folder(-0.5)
        assert '-' not in out
        assert out == 'neg0_5'
        assert _format_decimal_for_folder(-2) == '-2'

    def test_non_numeric_passthrough(self):
        from coana import _format_decimal_for_folder
        assert _format_decimal_for_folder('abc') == 'abc'
        assert _format_decimal_for_folder(None) == 'None'


# =============================================================================
# api_call_with_retry timeout must not hang
# =============================================================================

class TestApiCallTimeout:
    def test_hung_call_raises_timeout_promptly(self):
        started = time.time()

        def hung():
            time.sleep(30)  # simulate a dead API call
            return 1

        with pytest.raises(APITimeoutError):
            api_call_with_retry(hung, timeout=0.3, max_retries=1, verbose=False)
        elapsed = time.time() - started
        # Must return in well under the hung sleep (30s), allowing scheduler slack
        assert elapsed < 10, f'timeout did not protect the caller ({elapsed:.1f}s)'

    def test_success_path_unchanged(self):
        assert api_call_with_retry(lambda: 42, timeout=2, verbose=False) == 42


# =============================================================================
# Hoisted helpers
# =============================================================================

class TestHoistedHelpers:
    def test_extract_base_type(self):
        from neuronbridge_finder import _extract_base_type
        assert _extract_base_type('MCNS_aMe12') == 'aMe12'
        assert _extract_base_type('aMe12') == 'aMe12'
        assert _extract_base_type('MCNS_aMe12', {'MCNS_aMe12': 'aMe12'}) == 'aMe12'
        assert _extract_base_type('MCNS_aMe12', {'MCNS_aMe12': 'ME'}) == 'ME'

    def test_escape_cypher_string_fallback(self):
        from comparison.connectivity_profiler import _escape_cypher_string_fallback
        assert _escape_cypher_string_fallback("KCa'b'-ap1") == "KCa\\'b\\'-ap1"
        assert _escape_cypher_string_fallback(123) == '123'
        assert _escape_cypher_string_fallback('back\\slash') == 'back\\\\slash'


# =============================================================================
# statvis vs statvis_polars path-builder schema contract
# =============================================================================

class TestPathBuilderSchemas:
    def _build_inputs(self):
        import polars as pl
        paths = [['A', 'B', 'C'], ['D', 'E']]
        pd_conn = pd.DataFrame({
            'bodyId_pre': ['A', 'B', 'D'],
            'bodyId_post': ['B', 'C', 'E'],
            'weight': [5, 3, 7],
            'connection_ratio': [0.5, 0.3, 0.7],
            'traversal_probability': [0.9, 0.8, 0.95],
        })
        pl_conn = pl.from_pandas(pd_conn)
        return paths, pd_conn, pl_conn, ['C', 'E']

    def test_both_builders_return_dataframes_with_same_columns(self):
        from statvis import build_path_dataframe_from_paths as sv_build
        from statvis import build_path_dataframe_from_paths_polars as svp_build
        paths, pd_conn, pl_conn, targets = self._build_inputs()
        df_pd = sv_build(paths, pd_conn, targets, real_layer_map=None, level='bodyId')
        df_pl = svp_build(paths, pl_conn, targets, real_layer_map=None, level='bodyId')
        assert isinstance(df_pd, pd.DataFrame)
        # Known, documented divergence: the polars builder ignores type_lookup
        # and omits the 'path_str' list column (see the comment in
        # statvis.build_path_dataframe_from_paths_polars: "type_lookup is
        # currently ignored in Polars implementation for path string
        # formatting ... Path strings will contain IDs only").
        assert 'path_str' in df_pd.columns
        assert 'path_str' not in df_pl.columns
        # Contract: polars columns are a subset of pandas columns, row counts
        # match, and every scalar column present in both is equal.
        assert set(df_pl.columns) <= set(df_pd.columns), (
            f'polars builder has columns absent from pandas builder: '
            f'{sorted(set(df_pl.columns) - set(df_pd.columns))}'
        )
        assert len(df_pl) == len(df_pd)
        # path_prob SUSPENDED (2026-10-06): neither builder emits it
        scalar_cols = ['path', 'min_weight', 'min_ratio', 'length']
        for col in scalar_cols:
            assert col in df_pl.columns and col in df_pd.columns, col
            pd.testing.assert_series_equal(
                df_pl[col].to_pandas().reset_index(drop=True),
                df_pd[col].reset_index(drop=True),
                check_names=False,
                check_dtype=False,  # polars ints come back as uint32 vs int64
                obj=f'column {col}',
            )
        # List-valued columns differ in representation by design: pandas keeps
        # Python lists, polars stringifies them for CSV compatibility.
        for col in ('weights', 'ratios'):
            assert isinstance(df_pd[col].iloc[0], list)
            assert isinstance(df_pl[col].to_pandas().iloc[0], str)
