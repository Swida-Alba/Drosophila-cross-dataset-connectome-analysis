"""Unit tests for the row-based split groups (plan Phase B).

The split is expressed ONLY through the MergePolicy key_map plus
per-branch disclosure data (linker-refined pools + vote provenance via
``get_mapping_branches``/``get_mapping_support``).  No bodyId frame is
rewritten anywhere; per-neuron connectivity partitioning is TM VEV
verification, display-only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.merge_policy import build_merge_policy  # noqa: E402

from tests.core.test_merge_policy import (  # noqa: E402
    BANC,
    DATASETS,
    FAFB,
    MCNS,
    FakeMapper,
    build,
    rec,
)

MOTIVATING_RECORDS = {
    '5thsLNv_LNd6': rec('5thsLNv_LNd6', {
        MCNS: 'same_name_identity', FAFB: 'valid_split',
        BANC: 'valid_split'}),
    'SMP368': rec('SMP368', {
        MCNS: 'same_name_identity', FAFB: 'mapped', BANC: 'mapped'}),
}


class PooledMapper(FakeMapper):
    """``get_mapping_branches`` honors ``pool_fn`` and returns one
    record per target with the pool fields the pipeline attaches."""

    def get_mapping_branches(self, source_type, source_dataset,
                             target_dataset, pool_fn=None, **kwargs):
        targets = {
            (MCNS, FAFB): ['LNd_CRY+_ITP+', '5th-LNv'],
            (MCNS, BANC): ['LNd_a', 'aMe24', 's-LNv_a'],
        }.get((source_dataset, target_dataset), [])
        records = []
        for target in targets:
            record = {
                'source_dataset': source_dataset,
                'source_type': source_type,
                'target_dataset': target_dataset,
                'target_type': target,
                'source_body_ids': None,
                'target_body_ids': None,
                'source_basis': None,
                'target_basis': None,
                'source_pool_size': None,
                'source_type_total': None,
                'target_pool_size': None,
                'target_type_total': None,
                'supported': None,
                'status': 'unpooled' if pool_fn is None else None,
            }
            if pool_fn is not None:
                pool = pool_fn(source_dataset, target_dataset, [],
                               source_type, target)
                record.update({
                    'source_body_ids': pool['source_body_ids'],
                    'target_body_ids': pool['target_body_ids'],
                    'source_basis': 'linker rows',
                    'target_basis': 'linker rows',
                    'source_pool_size': len(pool['source_body_ids']),
                    'target_pool_size': len(pool['target_body_ids']),
                    'supported': True,
                    'status': 'supported',
                })
            records.append(record)
        return records


def test_branch_pools_attached_via_row_based_pool_fn():
    seen_pools = []

    def pool_fn(source_dataset, target_dataset, linkers, source_type,
                foreign_type):
        seen_pools.append((source_type, foreign_type))
        return {'source_body_ids': [101, 102],
                'target_body_ids': [201]}
    policy = build_merge_policy(
        PooledMapper(), source_tokens=['5thsLNv_LNd6'],
        target_tokens=['SMP368'], datasets=DATASETS,
        source_dataset=MCNS, attach_pools=pool_fn,
        records=MOTIVATING_RECORDS)
    group = policy.group_by_label('5thsLNv_LNd6')
    assert group.branch_pools, 'branch pools must be attached'
    # pools were requested per (split parent, branch leaf) pair — the
    # branch anchors live in the branch dataset (BANC here), so the
    # bridge queries run in the MCNS→BANC direction
    assert ('5thsLNv_LNd6', 's-LNv_a') in seen_pools
    assert ('5thsLNv_LNd6', 'aMe24') in seen_pools
    assert ('5thsLNv_LNd6', 'LNd_a') in seen_pools
    # the disclosure payload carries sizes + basis + status
    for label, entry in group.branch_pools.items():
        assert 'support' in entry
        for pool in entry['pools']:
            assert pool['source_basis'] == 'linker rows'
            assert pool['source_pool_size'] == 2
            assert pool['status'] == 'supported'


def test_pools_are_disclosure_only_membership_unchanged():
    """Attaching pools must NOT change the merge map — evidence never
    becomes membership (mapper boundary)."""
    without = build(FakeMapper(), ['5thsLNv_LNd6'], ['SMP368'],
                    MOTIVATING_RECORDS, attach_pools=False)
    with_pools = build_merge_policy(
        PooledMapper(), source_tokens=['5thsLNv_LNd6'],
        target_tokens=['SMP368'], datasets=DATASETS,
        source_dataset=MCNS, attach_pools=True,
        records=MOTIVATING_RECORDS)
    assert without.key_map == with_pools.key_map
    assert without.groups[0].members == with_pools.groups[0].members


def test_topology_embeds_pool_disclosure():
    policy = build_merge_policy(
        PooledMapper(), source_tokens=['5thsLNv_LNd6'],
        target_tokens=['SMP368'], datasets=DATASETS,
        source_dataset=MCNS, attach_pools=True,
        records=MOTIVATING_RECORDS)
    top = policy.topology_dict()
    merged = top['groups'][0]
    assert merged['kind'] == 'merged_split'
    for branch in merged['branches']:
        assert 'pools' in branch
        assert branch['pools']['pools']


def test_parent_row_is_residual_container_in_span_mode():
    """B2 (row-based form): untied parent members stay on the parent's
    OWN row — the parent row IS the residual container; no per-neuron
    residual label is emitted."""
    records = dict(MOTIVATING_RECORDS)
    records['FAFB_only'] = rec('FAFB_only', {FAFB: 'same_name_identity'})
    policy = build(FakeMapper(), ['5thsLNv_LNd6'], ['FAFB_only'], records)
    parent = next(g for g in policy.groups if g.kind == 'parent')
    assert parent.label == '5thsLNv_LNd6'
    assert parent.members == {MCNS: ['5thsLNv_LNd6']}
    # the parent's own key is present (its row exists, whole)
    assert policy.key_for(MCNS, '5thsLNv_LNd6') == '5thsLNv_LNd6'
    # branches are separate rows keyed by their leaf labels
    for branch in [g for g in policy.groups if g.kind == 'branch']:
        assert policy.key_map.get(branch.anchor) == branch.label
