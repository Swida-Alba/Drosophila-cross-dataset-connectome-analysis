"""Unit tests for the B4 auto-only BANC warning feed
(plan-query-anchored-cross-dataset-analysis.md §B4).

Mappings stay VALID; ``auto_only_edges`` is the evidence-only feed the
analyzer turns into a consolidated ``[BANC auto labels]`` warning block.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.merge_policy import (  # noqa: E402
    _format_support,
    auto_only_edges,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v888'
DATASETS = [MCNS, FAFB, BANC]


def decision(status='mapped', target=None, targets=None, support=None):
    return {'status': status, 'target_type': target,
            'target_types': targets or ([target] if target else []),
            'conflicts': [], 'relationship': None, 'support': support}


class SupportMapper:
    """Decision map with one auto-only edge, one curated edge, and one
    evidence-less edge."""

    def get_mapping_decision(self, token, source_ds, target_ds):
        key = (token, source_ds, target_ds)
        if key == ('T_auto', MCNS, BANC):
            return decision(target='aMe24', support={
                'votes': {'auto:5thsLNv_LNd6': 1},
                'verified_votes': {},
                'winner_derived_from_auto': True,
            })
        if key == ('T_curated', MCNS, BANC):
            return decision(target='s-LNv_a', support={
                'votes': {'5thsLNv_LNd6': 2},
                'verified_votes': {'5thsLNv_LNd6': 2},
                'winner_derived_from_auto': False,
            })
        if key == ('T_none', MCNS, BANC):
            return decision(target='X_none', support={
                'votes': {}, 'verified_votes': {},
                'winner_derived_from_auto': False,
            })
        if key == ('T_split', MCNS, BANC):
            return decision(
                status='valid_split_evidence', targets=['A_branch',
                                                        'B_branch'],
                support={
                    'A_branch': {
                        'votes': {'auto:p': 3},
                        'verified_votes': {},
                        'winner_derived_from_auto': True,
                    },
                    'B_branch': {
                        'votes': {'q': 2},
                        'verified_votes': {'q': 2},
                        'winner_derived_from_auto': False,
                    },
                })
        return decision()


def test_auto_only_predicate():
    rows = auto_only_edges(SupportMapper(), ['T_auto', 'T_curated', 'T_none'],
                           DATASETS)
    flagged = {(r['source_type'], r['target_type']) for r in rows}
    assert ('T_auto', 'aMe24') in flagged
    assert ('T_curated', 's-LNv_a') not in flagged  # curated vote exists
    assert ('T_none', 'X_none') not in flagged      # no votes at all
    row = next(r for r in rows if r['source_type'] == 'T_auto')
    assert row['source_dataset'] == MCNS
    assert row['target_dataset'] == BANC
    assert row['total_votes'] == 1
    assert 'auto-transferred' in row['support_text']


def test_split_branch_support_is_scoped_per_branch():
    rows = auto_only_edges(SupportMapper(), ['T_split'], DATASETS)
    flagged = {(r['source_type'], r['target_type']) for r in rows}
    assert ('T_split', 'A_branch') in flagged      # auto-only branch
    assert ('T_split', 'B_branch') not in flagged  # curated branch


def test_rows_sorted_and_deduped():
    rows = auto_only_edges(SupportMapper(), ['T_auto', 'T_auto'], DATASETS)
    keys = [(r['source_dataset'], r['source_type'],
             r['target_dataset'], r['target_type']) for r in rows]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_format_support_text():
    assert _format_support(None) == ''
    auto_only = _format_support({
        'votes': {'auto:x': 1, 'auto:y': 2}, 'verified_votes': {},
        'auto_stripped_votes': {'x': 1, 'y': 2},
        'winner_derived_from_auto': True,
    })
    assert '3 label vote(s)' in auto_only
    assert 'auto-transferred labels only' in auto_only
    curated = _format_support({
        'votes': {'x': 2}, 'verified_votes': {'x': 2},
        'auto_stripped_votes': {}, 'winner_derived_from_auto': False,
    })
    assert 'curated votes' in curated
    assert 'auto-transferred' not in curated


def test_result_type_universe_is_respected():
    """Types outside the requested set contribute nothing."""
    rows = auto_only_edges(SupportMapper(), ['T_curated'], DATASETS)
    assert rows == []
