"""Integration tests: the MergePolicy rides the metrics alignment lane
(plan §3.3/§3.4) — synthesized LabelMapper merges group members BEFORE
aggregation; ungoverned types fall back to canonical_merge_key toward
the policy's anchor namespace; no policy = byte-identical legacy path.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.merge_policy import build_merge_policy  # noqa: E402
from comparison.metrics import ComparisonMetrics  # noqa: E402

from tests.core.test_merge_policy import (  # noqa: E402
    FAFB,
    MCNS,
    FakeMapper,
    MOTIVATING_RECORDS,
    build,
)


def _frame(pairs):
    return pd.DataFrame({
        'type_pre': [p[0] for p in pairs],
        'type_post': [p[1] for p in pairs],
        'weight': [p[2] for p in pairs],
    })


def test_alignment_merges_branch_members_into_one_row():
    """B1 merged-anchor semantics: FAFB's two branch members collapse
    into the anchor's ONE row; per-dataset weights sum, nothing
    double-counts."""
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    label_mapper = policy.synthesized_label_mapper()
    metrics = ComparisonMetrics()
    raw_results = {
        FAFB: {5: _frame([
            ('5th-LNv', 'SMP368', 3),
            ('LNd_CRY+_ITP+', 'SMP368', 1),
        ])},
        MCNS: {5: _frame([('5thsLNv_LNd6', 'SMP368', 10)])},
    }
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 5, FAFB: 5},
        label_mapper=label_mapper, type_mapper=None,
        merge_policy=policy,
    )
    # one row per merged edge; the FAFB branch weights are summed
    assert sorted(aligned.index) == ['5thsLNv_LNd6 -> SMP368']
    assert aligned.loc['5thsLNv_LNd6 -> SMP368', FAFB] == 4
    assert aligned.loc['5thsLNv_LNd6 -> SMP368', MCNS] == 10


def test_alignment_without_policy_is_unchanged():
    """No policy → the legacy path (raw names, no lane): the two FAFB
    branch rows stay separate."""
    metrics = ComparisonMetrics()
    raw_results = {
        FAFB: {5: _frame([
            ('5th-LNv', 'SMP368', 3),
            ('LNd_CRY+_ITP+', 'SMP368', 1),
        ])},
        MCNS: {5: _frame([('5thsLNv_LNd6', 'SMP368', 10)])},
    }
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 5, FAFB: 5},
        label_mapper=None, type_mapper=None,
    )
    assert sorted(aligned.index) == [
        '5th-LNv -> SMP368',
        '5thsLNv_LNd6 -> SMP368',
        'LNd_CRY+_ITP+ -> SMP368',
    ]


def test_ungoverned_types_fall_back_toward_anchor_namespace(monkeypatch):
    """§3.3: types the policy does not govern resolve via
    canonical_merge_key with target_dataset=policy.anchor_ds (B3's
    chip-dataset namespace) — never silently male-cns."""
    captured_targets = set()

    def fake_merge_key(mapper, type_name, source_dataset,
                       target_dataset=None, **kwargs):
        captured_targets.add(target_dataset)
        from comparison.type_resolver import MergeKey
        return MergeKey(key=str(type_name), status='native')

    monkeypatch.setattr(
        'comparison.type_resolver.canonical_merge_key', fake_merge_key)

    class IdentityDisplayMapper:
        def get_display_name(self, name, datasets):
            return str(name)

    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    metrics = ComparisonMetrics()
    raw_results = {
        FAFB: {5: _frame([('5th-LNv', 'Intermediate_Z', 1)])},
        MCNS: {5: _frame([('5thsLNv_LNd6', 'Intermediate_Z', 2)])},
    }
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 5, FAFB: 5},
        label_mapper=policy.synthesized_label_mapper(),
        type_mapper=IdentityDisplayMapper(),
        merge_policy=policy,
    )
    assert captured_targets == {policy.anchor_ds}
    # the ungoverned intermediate resolved identically on both sides, so
    # the cross-dataset edge keys agree
    assert any('Intermediate_Z' in name for name in aligned.index)


def test_display_pass_keeps_group_labels_verbatim():
    """B6: with a policy, the ≤10k display pass must NOT re-run
    get_display_name over group labels (the scoped cosmetic form must
    not reappear)."""
    class DisplayMapper:
        def get_display_name(self, name, datasets):
            return f'{name}(alt1/alt2)'

    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    metrics = ComparisonMetrics()
    raw_results = {
        FAFB: {5: _frame([('5th-LNv', 'SMP368', 1)])},
        MCNS: {5: _frame([('5thsLNv_LNd6', 'SMP368', 2)])},
    }
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 5, FAFB: 5},
        label_mapper=policy.synthesized_label_mapper(),
        type_mapper=DisplayMapper(),
        merge_policy=policy,
    )
    assert '5thsLNv_LNd6 -> SMP368' in aligned.index
    assert not any('(alt1/alt2)' in name for name in aligned.index)


def test_user_labels_pass_canonical_block_untouched():
    """User-first type resolution (Phase A): the canonical block must not
    re-resolve a USER label — that re-resolution manufactured the
    family-compound (`X(Y/Z)`) and dataset-qualified (`ds:X`) residual
    keys on the real clock run. Both datasets' rows merge under the
    verbatim user label instead."""

    class ConflictMapper:
        """Loaded mapper that declares every lookup a CONFLICT — the
        canonical fallback keys such names dataset-scoped (`ds:raw`),
        exactly the residual shape under regression."""
        _loaded = True
        last_load_error = None
        _neuron_df_path = None

        def _detect_type_source(self, raw):
            return MCNS

        def _get_type_mapping_key(self, ds):
            return str(ds)

        def get_mapping_decision(self, token, source_ds, target_ds,
                                 include_bridges=False, **kwargs):
            return {'status': 'conflict', 'target_type': None,
                    'target_types': [], 'conflicts': ['stub'],
                    'support': None, 'relationship': None}

        def get_display_name(self, name, datasets):
            return name

    from types import SimpleNamespace

    from comparison.comparison_analyzer import _UserFirstLabelMapper
    from comparison.label_mapper import LabelMapper

    user = LabelMapper(
        source_mapping_dict={FAFB: [["5th-LNv"]], MCNS: [["5th-LNv"]]},
        source_labels=["5th-LNv"],
    )
    wrapper = _UserFirstLabelMapper(user, None)
    metrics = ComparisonMetrics()
    raw_results = {
        FAFB: {3: _frame([('5th-LNv', 'DN1a', 4)])},
        MCNS: {3: _frame([('5th-LNv', 'DN1a', 9)])},
    }
    # A plain LabelMapper has no is_user_label — pass the wrapper, which
    # is what _policy_label_mapper produces for user-mapped runs.
    assert getattr(wrapper, 'is_user_label', None) is not None
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 3, FAFB: 3},
        label_mapper=wrapper,
        type_mapper=SimpleNamespace(
            _loaded=False,
            get_display_name=lambda name, datasets: name),
        merge_policy=None,
    )
    assert '5th-LNv -> DN1a' in aligned.index
    assert aligned.loc['5th-LNv -> DN1a', FAFB] == 4
    assert aligned.loc['5th-LNv -> DN1a', MCNS] == 9

    # Control: without the wrapper, the conflict mapper dataset-qualifies
    # the very same name on a NON-canonical source dataset (FAFB; MCNS is
    # the canonical namespace and would short-circuit as 'native') — the
    # residual class under regression.
    aligned_policy = metrics._align_results_for_threshold_map(
        {FAFB: {3: _frame([('5th-LNv', 'DN1a', 4)])}},
        [FAFB], {FAFB: 3},
        label_mapper=LabelMapper(),
        type_mapper=ConflictMapper(),
        merge_policy=None,
    )
    assert not any('5th-LNv -> DN1a' == name for name in aligned_policy.index)
    assert any(':5th-LNv' in name for name in aligned_policy.index)


def test_analyzer_tier0_identity_for_user_labels():
    """A3: the analyzer's canonical / path-merge / display lanes return a
    user label verbatim (tier 0) — no canonical re-resolution, no family
    compounds, no variant parentheses over a settled identity."""
    from types import SimpleNamespace

    from comparison.comparison_analyzer import ComparisonAnalyzer
    from comparison.label_mapper import LabelMapper

    user = LabelMapper(
        source_mapping_dict={FAFB: [["5th-LNv"]], MCNS: [["5th-LNv", "13531"]]},
        source_labels=["5th-LNv", "MCNS_5th"],
    )
    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    analyzer.label_mapper = user
    # The tier-0 check precedes every parameters touch: a bare namespace
    # proves the auto mapper is never consulted for a user label.
    analyzer.parameters = SimpleNamespace(
        auto_type_mapping=True, _auto_type_mapper=object())

    assert analyzer._user_label_governs('5th-LNv')
    assert analyzer._user_label_governs('5th-LNv_L')  # suffix-aware
    assert not analyzer._user_label_governs('5thsLNv_LNd6')
    assert analyzer._get_canonical_type('5th-LNv', MCNS) == '5th-LNv'
    assert analyzer._get_path_merge_type('5th-LNv', MCNS) == '5th-LNv'
    assert analyzer._get_display_type('5th-LNv') == '5th-LNv'
    # a raw (non-label) name is NOT governed — the type mapper's job
    assert not analyzer._user_label_governs('5thsLNv_LNd6')

    # without a user mapper there is no tier 0 at all
    bare = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    bare.label_mapper = LabelMapper()
    assert not bare._user_label_governs('5th-LNv')


def test_edge_lane_converges_split_families():
    """Phase R1: singleton per-chip policy groups must not leave split
    branches (FAFB R8 vs MCNS R8p) on separate edge rows — the family
    resolver (the analyzer's `_path_family_map`) converges them exactly
    like the path lane."""
    from types import SimpleNamespace

    class _SingletonPolicy:
        """Every name governed by its OWN group (the MF run's shape);
        group_by_label -> None = singleton, no cross-dataset members."""
        anchor_ds = None

        def key_for(self, dataset, t):
            return f'chip_{t}'

        def group_by_label(self, label):
            return None

        def is_group_label(self, label):
            return False

    families = {(FAFB, 'R8'): 'R8', (MCNS, 'R8p'): 'R8'}
    metrics = ComparisonMetrics()
    metrics._family_resolver = lambda ds, t: families.get((str(ds), str(t)))
    raw_results = {
        FAFB: {3: _frame([('R8', 'Dm9', 5)])},
        MCNS: {3: _frame([('R8p', 'Dm9', 7)])},
    }
    aligned = metrics._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 3, FAFB: 3},
        label_mapper=None,
        type_mapper=SimpleNamespace(
            _loaded=False, get_display_name=lambda name, datasets: name),
        merge_policy=_SingletonPolicy(),
    )
    assert 'R8 -> Dm9' in aligned.index
    assert aligned.loc['R8 -> Dm9', FAFB] == 5
    assert aligned.loc['R8 -> Dm9', MCNS] == 7

    # Control: without a family resolver the legacy behavior holds —
    # singleton key_for hits survive as the comparison keys and the rows
    # stay separate (the policy governs every name in this stub).
    legacy = ComparisonMetrics()
    aligned_legacy = legacy._align_results_for_threshold_map(
        raw_results, [MCNS, FAFB], {MCNS: 3, FAFB: 3},
        label_mapper=None,
        type_mapper=SimpleNamespace(
            _loaded=False, get_display_name=lambda name, datasets: name),
        merge_policy=_SingletonPolicy(),
    )
    assert 'chip_R8 -> chip_Dm9' in aligned_legacy.index
    assert 'chip_R8p -> chip_Dm9' in aligned_legacy.index
