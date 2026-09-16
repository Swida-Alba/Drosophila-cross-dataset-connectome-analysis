"""Unit tests for the query-anchored MergePolicy
(plan-query-anchored-cross-dataset-analysis.md).

Offline tests on a synthetic mapper replicating the §0 motivating rows —
no network, no caches.  Covers:

- B3 anchor selection, three cases (single_dataset / same_name / span)
- B1 merge granularity: "1"-side merge incl. the weak branch (Decision 9);
  leaf-anchored chip covers only its own branch
- span mode: parent keeps its own whole row; branches become leaf groups
- fan-in ownership: a shared leaf merges with NEITHER claimant
- chip-order invariance: permuted chips → identical key_map/ids/warnings
- synthesized LabelMapper lane (Decision 8) + user-mapper precedence
- topology dict JSON safety
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.merge_policy import (  # noqa: E402
    MergeGroup,
    MergePolicy,
    SPAN_WARNING,
    auto_only_edges,
    build_merge_policy,
    row_anchor_group,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v888'
DATASETS = [MCNS, FAFB, BANC]


def rec(token, per_ds, target_types_by_ds=None):
    """Synthetic per-dataset resolution records for one chip."""
    target_types_by_ds = target_types_by_ds or {}
    return {
        ds: {'token': token, 'dataset': ds, 'status': status,
             'target_types': list(target_types_by_ds.get(ds, [])),
             'method': '', 'confidence': 3}
        for ds, status in per_ds.items()
    }


class FakeMapper:
    """§0 motivating rows: MCNS ``5thsLNv_LNd6`` 1-to-N splits to FAFB
    {5th-LNv, LNd_CRY+_ITP+} and BANC {s-LNv_a, LNd_a, weak aMe24};
    clean FAFB→BANC pairings; the reverse FAFB→MCNS direction is
    evidence_only (leaf side)."""

    def has_native_type(self, token, dataset):
        return token in ('5thsLNv_LNd6', 'lLnk_ITP') and dataset == MCNS

    def get_mapping_decision(self, token, source_ds, target_ds):
        decision = {'target_type': None, 'target_types': [],
                    'status': 'unmapped', 'relationship': None,
                    'conflicts': [], 'support': None}
        key = (token, source_ds, target_ds)
        if key == ('5thsLNv_LNd6', MCNS, FAFB):
            decision.update(status='valid_split_evidence',
                            relationship='1-to-N',
                            target_types=['LNd_CRY+_ITP+', '5th-LNv'])
        elif key == ('5thsLNv_LNd6', MCNS, BANC):
            decision.update(status='valid_split_evidence',
                            relationship='1-to-N',
                            target_types=['LNd_a', 'aMe24', 's-LNv_a'])
        elif key == ('5th-LNv', FAFB, MCNS):
            decision.update(status='evidence_only',
                            target_types=['5thsLNv_LNd6'])
        elif key == ('LNd_CRY+_ITP+', FAFB, MCNS):
            decision.update(status='evidence_only',
                            target_types=['5thsLNv_LNd6'])
        elif key == ('5th-LNv', FAFB, BANC):
            decision.update(status='mapped', target_type='s-LNv_a',
                            target_types=['s-LNv_a'])
        elif key == ('LNd_CRY+_ITP+', FAFB, BANC):
            decision.update(status='mapped', target_type='LNd_a',
                            target_types=['LNd_a'])
        return decision

    def get_mapping_support(self, *args, **kwargs):
        return None

    def get_mapping_branches(self, *args, **kwargs):
        return []

    def get_mapping_conflicts(self, source_ds, target_ds, source_type=None):
        return []


MOTIVATING_RECORDS = {
    '5thsLNv_LNd6': rec('5thsLNv_LNd6', {
        MCNS: 'same_name_identity', FAFB: 'valid_split',
        BANC: 'valid_split'}),
    'SMP368': rec('SMP368', {
        MCNS: 'same_name_identity', FAFB: 'mapped', BANC: 'mapped'}),
}


def build(mapper, source_tokens, target_tokens, records, datasets=None,
          source_dataset=MCNS, attach_pools=False):
    return build_merge_policy(
        mapper,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        datasets=datasets if datasets is not None else DATASETS,
        source_dataset=source_dataset,
        attach_pools=attach_pools,
        records=records,
    )


# ----------------------------------------------------------------------
# B3: anchor selection, three cases
# ----------------------------------------------------------------------

def test_anchor_case_single_dataset():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    assert policy is not None
    assert policy.anchor_case == 'single_dataset'
    assert policy.anchor_ds == MCNS
    assert policy.warnings == []


def test_anchor_case_same_name_requires_native_everywhere():
    """B3 case 2 needs NATIVE presence in every dataset — a mapped RENAME
    into another dataset does not qualify (real-data finding 2026-09-15),
    and the same-name anchor stays None so ungoverned types keep the
    canonical fallback namespace."""
    records = {
        'LNd_b': rec('LNd_b', {ds: 'same_name_identity' for ds in DATASETS}),
        'LNd_c': rec('LNd_c', {ds: 'same_name_identity' for ds in DATASETS}),
    }
    policy = build(FakeMapper(), ['LNd_b'], ['LNd_c'], records)
    assert policy.anchor_case == 'same_name'
    assert policy.anchor_ds is None
    assert all('type granularity' not in w for w in policy.warnings)
    # same-name keys are identity everywhere
    for ds in DATASETS:
        assert policy.key_for(ds, 'LNd_b') == 'LNd_b'


def test_leaf_chip_clean_reverse_rename_keeps_parent_whole():
    """Real-data finding (2026-09-15): FAFB 5th-LNv renames CLEANLY into
    MCNS 5thsLNv_LNd6 while MCNS → FAFB is the 1-to-N conflict.  B1: the
    parent keeps its own whole row — it must not be absorbed into the
    leaf-anchored group via the reverse rename."""
    records = {
        '5th-LNv': rec('5th-LNv', {
            FAFB: 'same_name_identity', MCNS: 'mapped', BANC: 'mapped'},
            target_types_by_ds={MCNS: ['5thsLNv_LNd6'],
                                BANC: ['s-LNv_a']}),
        'SMP368': rec('SMP368', {ds: 'same_name_identity'
                                 for ds in DATASETS}),
    }

    class ReverseRenameMapper(FakeMapper):
        def get_mapping_conflicts(self, source_ds, target_ds,
                                  source_type=None):
            if (source_ds, target_ds, source_type) == (
                    MCNS, FAFB, '5thsLNv_LNd6'):
                conflict = type('C', (), {
                    'relationship': '1-to-N',
                    'target_types': {'5th-LNv', 'LNd_CRY+_ITP+'},
                    'source_dataset': MCNS,
                    'target_dataset': FAFB,
                    'source_type': '5thsLNv_LNd6',
                    'origin': '',
                })()
                return [conflict]
            return []

        def get_mapping_decision(self, token, source_ds, target_ds):
            key = (token, source_ds, target_ds)
            if key == ('5th-LNv', FAFB, MCNS):
                return {'status': 'mapped', 'target_type': '5thsLNv_LNd6',
                        'target_types': ['5thsLNv_LNd6'], 'conflicts': [],
                        'relationship': '1-to-1', 'support': None}
            if key == ('5th-LNv', FAFB, BANC):
                return {'status': 'mapped', 'target_type': 's-LNv_a',
                        'target_types': ['s-LNv_a'], 'conflicts': [],
                        'relationship': '1-to-1', 'support': None}
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': [], 'conflicts': [],
                    'relationship': None, 'support': None}

    policy = build(ReverseRenameMapper(), ['5th-LNv'], ['SMP368'], records)
    # anchored on the first chip's home namespace (FAFB), NOT same_name
    assert policy.anchor_case == 'single_dataset'
    assert policy.anchor_ds == FAFB
    group = policy.group_by_label('5th-LNv')
    assert group is not None
    # the clean reverse target is a split parent toward FAFB → excluded
    assert policy.key_for(MCNS, '5thsLNv_LNd6') is None
    assert MCNS not in group.members
    # the BANC pairing (not a split parent) still merges
    assert group.members[BANC] == ['s-LNv_a']


def test_anchor_case_resolution_beats_origin():
    """B3 case 1 is RESOLUTION-based: a FAFB-origin chip that maps cleanly
    into MCNS plus an MCNS chip anchor on MCNS — no span warning."""
    records = {
        'SMP368': rec('SMP368', {MCNS: 'same_name_identity'}),
        'FAFB_alias': rec('FAFB_alias', {
            FAFB: 'same_name_identity', MCNS: 'mapped'},
            target_types_by_ds={MCNS: ['SMP368']}),
    }

    class ResolvingMapper(FakeMapper):
        def get_mapping_decision(self, token, source_ds, target_ds):
            if (token, source_ds, target_ds) == ('FAFB_alias', FAFB, MCNS):
                return {'status': 'mapped', 'target_type': 'SMP368',
                        'target_types': ['SMP368'], 'conflicts': [],
                        'relationship': '1-to-1', 'support': None}
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': [], 'conflicts': [],
                    'relationship': None, 'support': None}

    policy = build(ResolvingMapper(), ['FAFB_alias'], ['SMP368'], records)
    assert policy.anchor_case == 'single_dataset'
    assert policy.anchor_ds == MCNS
    assert all('type granularity' not in w for w in policy.warnings)


def test_anchor_case_span_warning_and_leaves():
    records = dict(MOTIVATING_RECORDS)
    records['FAFB_only'] = rec('FAFB_only', {FAFB: 'same_name_identity'})
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['FAFB_only'], records)
    assert policy.anchor_case == 'span'
    assert policy.anchor_ds is None
    assert any('type granularity' in w for w in policy.warnings)
    by_kind = {}
    for group in policy.groups:
        by_kind.setdefault(group.kind, []).append(group)
    # parent keeps its own whole row; branches become separate leaf rows
    assert len(by_kind.get('parent', [])) == 1
    assert by_kind['parent'][0].members == {MCNS: ['5thsLNv_LNd6']}
    # three unified branch rows (one per inseparable chain); the anchor
    # label is deterministic (first seed = earliest-sorted dataset)
    branches = by_kind.get('branch', [])
    assert len(branches) == 3
    assert {g.label for g in branches} == {'LNd_a', 'aMe24', 's-LNv_a'}
    # each chain keeps its clean cross-dataset pairing
    chain_5th = next(g for g in branches
                     if g.members.get(FAFB) == ['5th-LNv'])
    assert chain_5th.members[BANC] == ['s-LNv_a']
    chain_lnd = next(g for g in branches
                     if g.members.get(FAFB) == ['LNd_CRY+_ITP+'])
    assert chain_lnd.members[BANC] == ['LNd_a']


def test_anchor_case_conflicted_same_name_disqualifies():
    """A conflicted same-name chip must NOT qualify for case 2 — its
    records carry a conflict status, so S(chip) != all datasets."""
    records = {
        'CB1011': rec('CB1011', {
            MCNS: 'same_name_identity', FAFB: 'conflict',
            BANC: 'same_name_identity'}),
        'LNd_b': rec('LNd_b', {ds: 'same_name_identity' for ds in DATASETS}),
    }
    policy = build(FakeMapper(), ['CB1011'], ['LNd_b'], records)
    assert policy.anchor_case != 'same_name'


# ----------------------------------------------------------------------
# B1: merge granularity
# ----------------------------------------------------------------------

def test_one_side_chip_merges_all_branches_incl_weak():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    group = policy.group_by_label('5thsLNv_LNd6')
    assert group.kind == 'merged_split'
    # MCNS parent + both FAFB leaves + all three BANC followers
    # (Decision 9: the weak n=1 auto-vote branch aMe24 stays in the group)
    assert sorted(group.members[FAFB]) == ['5th-LNv', 'LNd_CRY+_ITP+']
    assert sorted(group.members[BANC]) == ['LNd_a', 'aMe24', 's-LNv_a']
    assert policy.key_for(FAFB, '5th-LNv') == '5thsLNv_LNd6'
    assert policy.key_for(BANC, 'aMe24') == '5thsLNv_LNd6'
    # branch provenance chains survive for disclosure (B2)
    branch_chains = {
        frozenset((ds, tuple(names)) for ds, names in b.members.items())
        for b in group.branches
    }
    assert (frozenset({(FAFB, ('5th-LNv',)), (BANC, ('s-LNv_a',))})
            in branch_chains)
    assert (frozenset({(FAFB, ('LNd_CRY+_ITP+',)), (BANC, ('LNd_a',))})
            in branch_chains)


def test_leaf_anchored_chip_covers_only_own_branch():
    records = {
        '5th-LNv': rec('5th-LNv', {FAFB: 'same_name_identity'}),
        'SMP368': rec('SMP368', {MCNS: 'same_name_identity'}),
    }
    policy = build(FakeMapper(), ['5th-LNv'], ['SMP368'], records)
    group = policy.group_by_label('5th-LNv')
    assert group is not None
    # its own branch only: the clean BANC pairing joins; the conflicted
    # MCNS parent is NOT claimed
    assert set(group.members) == {FAFB, BANC}
    assert group.members[BANC] == ['s-LNv_a']
    # the parent is not in key_map: the canonical fallback keeps it a
    # dataset-scoped whole row (the residual container)
    assert policy.key_for(MCNS, '5thsLNv_LNd6') is None
    assert ('MCNS', '5thsLNv_LNd6') not in {
        (ds, name) for (ds, name) in policy.key_map}


# ----------------------------------------------------------------------
# Fan-in ownership + chip-order invariance
# ----------------------------------------------------------------------

class FanInMapper(FakeMapper):
    def get_mapping_decision(self, token, source_ds, target_ds):
        key = (token, source_ds, target_ds)
        if key == ('lLnk_ITP', MCNS, FAFB):
            return {'status': 'valid_split_evidence', 'relationship': '1-to-N',
                    'target_types': ['LNd_CRY+_ITP+'], 'target_type': None,
                    'conflicts': [], 'support': None}
        if key == ('lLnk_ITP', MCNS, BANC):
            return {'status': 'valid_split_evidence', 'relationship': '1-to-N',
                    'target_types': ['LNd_a'], 'target_type': None,
                    'conflicts': [], 'support': None}
        return super().get_mapping_decision(token, source_ds, target_ds)


FANIN_RECORDS = dict(MOTIVATING_RECORDS)
FANIN_RECORDS['lLnk_ITP'] = rec('lLnk_ITP', {
    MCNS: 'same_name_identity', FAFB: 'valid_split', BANC: 'valid_split'})


def test_fan_in_leaf_merges_with_neither_claimant():
    policy = build(FanInMapper(), ['5thsLNv_LNd6', 'lLnk_ITP'], ['SMP368'],
                   FANIN_RECORDS)
    # the shared leaf (and the follower it drags along) are keyed NOWHERE
    assert policy.key_for(FAFB, 'LNd_CRY+_ITP+') is None
    assert policy.key_for(BANC, 'LNd_a') is None
    # ... while the non-shared members merge normally
    assert policy.key_for(FAFB, '5th-LNv') == '5thsLNv_LNd6'
    assert policy.key_for(MCNS, 'lLnk_ITP') == 'lLnk_ITP'
    assert set(policy.fan_in) == {(FAFB, 'LNd_CRY+_ITP+'), (BANC, 'LNd_a')}
    assert any('merge fan-in' in w for w in policy.warnings)


def test_fan_in_warns_parent_row_is_partial():
    """The queried parent's row EXCLUDES the co-queried shared branch —
    the warning must state that explicitly (user requirement 2026-09-15)."""
    policy = build(FanInMapper(), ['5thsLNv_LNd6', 'lLnk_ITP'], ['SMP368'],
                   FANIN_RECORDS)
    explicit = [w for w in policy.warnings if 'does NOT include' in w]
    # one explicit line per (queried parent, shared key) pair
    assert len(explicit) == 4
    assert any("queried parent 5thsLNv_LNd6 does NOT include "
               "flywire_FAFB_v783 LNd_CRY+_ITP+" in w for w in explicit)
    assert any("queried parent lLnk_ITP does NOT include "
               "banc_v888 LNd_a" in w for w in explicit)
    assert all('PARTIAL by design' in w for w in explicit)
    # the explicit lines also reach the topology export
    assert any('does NOT include' in w
               for w in policy.topology_dict()['warnings'])


def test_chip_order_invariance():
    order_a = build(FanInMapper(), ['5thsLNv_LNd6', 'lLnk_ITP'], ['SMP368'],
                    FANIN_RECORDS)
    order_b = build(FanInMapper(), ['lLnk_ITP'], ['SMP368', '5thsLNv_LNd6'],
                    FANIN_RECORDS)
    assert order_a.key_map == order_b.key_map
    assert order_a.warnings == order_b.warnings
    assert [g.group_id for g in order_a.groups] == \
        [g.group_id for g in order_b.groups]
    assert order_a.fan_in == order_b.fan_in


# ----------------------------------------------------------------------
# Synthesized LabelMapper lane (Decision 8)
# ----------------------------------------------------------------------

def test_synthesized_label_mapper_merges_branch_members():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    label_mapper = policy.synthesized_label_mapper()
    assert label_mapper is not None
    frame = pd.DataFrame({
        'type_pre': ['5th-LNv', 'LNd_CRY+_ITP+', 'SMP368'],
        'type_post': ['SMP368', 'SMP368', '5th-LNv'],
        'weight': [3, 1, 2],
    })
    out = label_mapper.apply_to_dataframe(frame, FAFB)
    assert list(out['std_label_pre']) == [
        '5thsLNv_LNd6', '5thsLNv_LNd6', 'SMP368']
    assert list(out['std_label_post']) == [
        'SMP368', 'SMP368', '5thsLNv_LNd6']


def test_synthesized_label_mapper_identity_for_ungoverned_types():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    label_mapper = policy.synthesized_label_mapper()
    frame = pd.DataFrame({
        'type_pre': ['SomeOtherType'], 'type_post': ['5th-LNv'],
        'weight': [1],
    })
    out = label_mapper.apply_to_dataframe(frame, FAFB)
    assert list(out['std_label_pre']) == ['SomeOtherType']


def test_user_mapper_wins_by_construction():
    """A user-governed name is rewritten BEFORE the policy lane sees it;
    the policy is keyed on raw names, so the user label passes through
    untouched (the plan's 'user mappings win')."""
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    label_mapper = policy.synthesized_label_mapper()
    frame = pd.DataFrame({
        'type_pre': ['MyUserLabel'], 'type_post': ['SMP368'], 'weight': [1],
    })
    out = label_mapper.apply_to_dataframe(frame, FAFB)
    assert list(out['std_label_pre']) == ['MyUserLabel']


# ----------------------------------------------------------------------
# Policy plumbing
# ----------------------------------------------------------------------

def test_null_mapper_returns_none():
    assert build_merge_policy(
        None, source_tokens=['x'], target_tokens=['y'],
        datasets=DATASETS) is None


def test_topology_dict_is_json_safe():
    policy = build(FanInMapper(), ['5thsLNv_LNd6', 'lLnk_ITP'], ['SMP368'],
                   FANIN_RECORDS)
    top = policy.topology_dict()
    serialized = json.dumps(top)  # must not raise
    assert 'merged_split' in serialized
    assert any('merge fan-in' in w for w in top['warnings'])


def test_key_for_governed_vs_ungoverned():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    assert policy.key_for(FAFB, '5th-LNv') == '5thsLNv_LNd6'
    assert policy.key_for(FAFB, 'TotallyUnknown') is None  # fallback path
    assert policy.is_group_label('5thsLNv_LNd6')
    assert not policy.is_group_label('TotallyUnknown')
    assert policy.label_for_name('s-LNv_a') == '5thsLNv_LNd6'
    assert policy.label_for_name('Unknown') is None


# ----------------------------------------------------------------------
# anchor_group row rule (unambiguous: ALL endpoints in the SAME group)
# ----------------------------------------------------------------------

def test_row_anchor_group_all_endpoints_same_group():
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    assert row_anchor_group(policy, {
        MCNS: '5thsLNv_LNd6', BANC: 'aMe24'}) == '5thsLNv_LNd6'
    assert row_anchor_group(policy, {
        FAFB: '5th-LNv', BANC: 's-LNv_a'}) == '5thsLNv_LNd6'


def test_row_anchor_group_partial_membership_stays_blank():
    """A row that merely TOUCHES a group through one endpoint is NOT
    tagged — the column means 'this row IS a mapping of the group'."""
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   MOTIVATING_RECORDS)
    assert row_anchor_group(policy, {
        MCNS: 'LNd_b', BANC: 'LNd_a'}) == ''
    assert row_anchor_group(policy, {MCNS: '5thsLNv_LNd6'}) == \
        '5thsLNv_LNd6'
    assert row_anchor_group(policy, {}) == ''


def test_span_duplicate_group_deduped():
    """A leaf chip and a parent's unified branch can describe the SAME
    group in a span run — one group object survives."""
    records = {
        '5thsLNv_LNd6': rec('5thsLNv_LNd6', {
            MCNS: 'same_name_identity', FAFB: 'valid_split'}),
        '5th-LNv': rec('5th-LNv', {FAFB: 'same_name_identity'}),
    }

    class FafbOnlySplitMapper(FakeMapper):
        def get_mapping_decision(self, token, source_ds, target_ds):
            if (token, source_ds, target_ds) == (
                    '5thsLNv_LNd6', MCNS, FAFB):
                return {'status': 'valid_split_evidence',
                        'relationship': '1-to-N',
                        'target_types': ['LNd_CRY+_ITP+', '5th-LNv'],
                        'target_type': None, 'conflicts': [],
                        'support': None}
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': [], 'conflicts': [],
                    'relationship': None, 'support': None}

    policy = build(FafbOnlySplitMapper(), ['5thsLNv_LNd6', '5th-LNv'], [],
                   records)
    labels = [g.label for g in policy.groups]
    assert labels.count('5th-LNv') == 1
    # every label still maps to exactly one group object
    for label in labels:
        assert policy.group_by_label(label) is not None


def test_auto_only_branch_is_chain_terminal():
    """Real-data leak (run 050547 report, found 2026-09-15): the weak
    auto-only branch MCNS 5thsLNv_LNd6 -> BANC aMe24 chained through the
    curated aMe24=aMe24 identity and recruited the whole FAFB aMe24 type
    into the group.  Auto-only branch seeds must be CHAIN-TERMINAL: the
    branch stays (Decision 9, warned) but confined to its own dataset."""
    records = dict(MOTIVATING_RECORDS)

    class AutoOnlyBancMapper(FakeMapper):
        def get_mapping_decision(self, token, source_ds, target_ds):
            decision = super().get_mapping_decision(token, source_ds,
                                                    target_ds)
            if (token, source_ds, target_ds) == ('5thsLNv_LNd6', MCNS, BANC):
                # per-branch support: aMe24 is auto-only, LNd_a is
                # crosswalk-based (no votes), s-LNv_a is auto-only
                decision['support'] = {
                    'LNd_a': {'votes': {}, 'verified_votes': {},
                              'winner_derived_from_auto': False},
                    'aMe24': {'votes': {'5thsLNv_LNd6': 1},
                              'verified_votes': {},
                              'winner_derived_from_auto': True},
                    's-LNv_a': {'votes': {'5thsLNv_LNd6': 2},
                                'verified_votes': {},
                                'winner_derived_from_auto': True},
                }
            return decision

    policy = build(AutoOnlyBancMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                   records)
    group = policy.group_by_label('5thsLNv_LNd6')
    # the weak BANC branch stays in the group (Decision 9) …
    assert group.members[BANC] == ['LNd_a', 'aMe24', 's-LNv_a'] or \
        'aMe24' in group.members[BANC]
    # …but its auto-only chain must NOT recruit FAFB aMe24
    assert 'aMe24' not in group.members.get(FAFB, [])
    assert policy.key_for(FAFB, 'aMe24') is None
    # the aMe24 branch chain is confined to BANC
    ame_branch = next(b for b in group.branches if b.label == 'aMe24')
    assert set(ame_branch.members) == {BANC}
    # the curated FAFB-side branches chain normally (the LNd branch is
    # anchored on its BANC seed and carries the FAFB member)
    lnd_branch = next(b for b in group.branches
                      if b.members.get(FAFB) == ['LNd_CRY+_ITP+'])
    assert lnd_branch.members[BANC] == ['LNd_a']


def test_fan_in_prunes_claimant_group_members():
    """The shared fan-in leaf must leave the claimant groups' members and
    branches too, not just key_map — the display lane (group.members feeds
    the type-mapping table and topology) has to agree with the aggregation
    lane (plan E, 2026-09-15: CL317 showed inside both aMe26/aMe9 rows
    while the counts excluded it)."""
    policy = build(FanInMapper(), ['5thsLNv_LNd6', 'lLnk_ITP'], ['SMP368'],
                   FANIN_RECORDS)
    parent = policy.group_by_label('5thsLNv_LNd6')
    assert parent is not None
    for ds, names in parent.members.items():
        assert 'LNd_CRY+_ITP+' not in names, (ds, names)
    for branch in parent.branches:
        member_names = {n for names in branch.members.values()
                        for n in names}
        assert 'LNd_CRY+_ITP+' not in member_names, branch.members
    # the non-shared branch member still merges into the parent row
    assert policy.key_for(FAFB, '5th-LNv') == '5thsLNv_LNd6'
    assert any('5th-LNv' in names
               for names in parent.members.values())
    # and the topology export reflects the pruning as well
    top_group = next(g for g in policy.topology_dict()['groups']
                     if g['label'] == '5thsLNv_LNd6')
    for ds, names in top_group['members'].items():
        assert 'LNd_CRY+_ITP+' not in names
