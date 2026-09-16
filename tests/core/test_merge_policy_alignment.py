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
