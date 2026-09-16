"""Plan R1 tests: hemisphere-suffix awareness in the type-level lanes.

With ``separate_hemispheres=True``, cached frames carry suffixed types
(``aMe26_L/_R``) while the merge policy / mapper key base names. The
metrics canonical-map must strip the suffix for the lookup and re-apply
it to the canonical key so L and R stay DISTINCT rows that still merge
across datasets. With the flag off (default) behavior is byte-identical.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.metrics import ComparisonMetrics  # noqa: E402
from utils.naming_utils import split_hemi_suffix  # noqa: E402


class FakePolicy:
    """Group-label keys for BASE names only — like the real merge policy,
    whose key_map keys unsuffixed type names."""

    anchor_ds = 'male-cns:v1.0'

    def __init__(self):
        self.key_map = {
            ('male-cns:v1.0', 'aMe26'): 'aMe26',
            ('flywire_FAFB_v783', 'aMe26'): 'aMe26',
            ('male-cns:v1.0', 's-LNv_a'): 's-LNv_a',
            ('flywire_FAFB_v783', '5th-LNv'): 's-LNv_a',
        }

    def key_for(self, dataset, type_name):
        return self.key_map.get((str(dataset), str(type_name)))


def _results():
    def frame(types):
        return pd.DataFrame([
            {'type_pre': p, 'type_post': q, 'weight': w}
            for p, q, w in types])

    return {
        'male-cns:v1.0': {3: frame([
            ('aMe26_L', 's-LNv_a', 10),
            ('aMe26_R', 's-LNv_a', 20),
            ('aMe26', 's-LNv_a', 5),
        ])},
        'flywire_FAFB_v783': {3: frame([
            ('aMe26_L', 's-LNv_a', 7),
            ('aMe26_R', 's-LNv_a', 3),
        ])},
    }


def test_hemi_aware_alignment_merges_across_datasets_keeps_sides():
    metrics = ComparisonMetrics()
    aligned = metrics._align_results_for_threshold_map(
        _results(), ['male-cns:v1.0', 'flywire_FAFB_v783'],
        {ds: 3 for ds in ('male-cns:v1.0', 'flywire_FAFB_v783')},
        label_mapper=None, type_mapper=None,
        merge_policy=FakePolicy(), hemi_aware=True)
    idx = set(aligned.index)
    # L rows merged ACROSS datasets (10 + 7), R rows merged (20 + 3),
    # base name kept separate from its suffixed variants
    assert 'aMe26_L -> s-LNv_a' in idx
    assert 'aMe26_R -> s-LNv_a' in idx
    assert 'aMe26 -> s-LNv_a' in idx
    assert aligned.loc['aMe26_L -> s-LNv_a', 'male-cns:v1.0'] == 10
    assert aligned.loc['aMe26_L -> s-LNv_a', 'flywire_FAFB_v783'] == 7
    assert aligned.loc['aMe26_R -> s-LNv_a', 'male-cns:v1.0'] == 20
    # suffixed types never merge into the base-name row
    assert aligned.loc['aMe26 -> s-LNv_a', 'male-cns:v1.0'] == 5
    assert aligned.loc['aMe26 -> s-LNv_a', 'flywire_FAFB_v783'] == 0


def test_non_hemi_aware_alignment_is_unchanged():
    metrics = ComparisonMetrics()
    aligned = metrics._align_results_for_threshold_map(
        _results(), ['male-cns:v1.0', 'flywire_FAFB_v783'],
        {ds: 3 for ds in ('male-cns:v1.0', 'flywire_FAFB_v783')},
        label_mapper=None, type_mapper=None,
        merge_policy=FakePolicy(), hemi_aware=False)
    idx = set(aligned.index)
    # legacy behavior: suffixed types are NOT resolved against the
    # base-name key_map — each stays a raw dataset-scoped row
    assert 'aMe26_L -> s-LNv_a' in idx
    assert 'aMe26_R -> s-LNv_a' in idx
    assert 'aMe26 -> s-LNv_a' in idx
    # per-dataset rows stay unmerged for suffixed types
    assert aligned.loc['aMe26_L -> s-LNv_a', 'male-cns:v1.0'] == 10
    assert aligned.loc['aMe26_L -> s-LNv_a', 'flywire_FAFB_v783'] == 7
    assert aligned.loc['aMe26_R -> s-LNv_a', 'flywire_FAFB_v783'] == 3


def test_split_hemi_suffix_shared_util():
    assert split_hemi_suffix('aMe26_L') == ('aMe26', '_L')
    assert split_hemi_suffix('aMe26_U') == ('aMe26', '_U')
    assert split_hemi_suffix('aMe26') == ('aMe26', '')
    assert split_hemi_suffix(42) == (42, '')


def test_label_mapper_uses_shared_split():
    from comparison.label_mapper import LabelMapper
    lm = LabelMapper(source_mapping_dict={'d': [['aMe26']]},
                     source_labels=['G'])
    # the static helper delegates to the shared implementation
    assert LabelMapper._split_hemi_suffix('aMe26_R') == ('aMe26', '_R')
    assert lm.get_label('d', 'aMe26_L') == 'G_L'
