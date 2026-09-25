"""Audit 2026-09-25 phase-6 fixes: the resolver status block.

RES-1  — the step-4 union keeps the decision's own verdict: an
         evidence-only relation with bridge ends stays evidence-only
         (never a licensed `mapped` equivalence), and the decision's
         target evidence joins the union.
RES-8  — the stale-claim native-realized branch is NOT a curated
         confirmation (`curated_identity=False`).
RES-11 — canonical_merge_key / expand_profile_types demote a stale claim
         to the raw name / `claimed` instead of merging under a phantom
         target no namespace carries.
RES-12 — `claimed` sits in the severity order (mapped must not absorb it).
RES-10 — `_member_targets` contributes only unique equivalences or
         licensed splits.
RES-9  — the tier-6 echo downgrade skips evidence-backed identities.
"""

import pytest

from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper
from comparison.type_resolver import (
    MapperSnapshot,
    canonical_merge_key,
    expand_profile_types,
    resolve_flow_status,
    resolve_valid_targets,
    _severity,
    _STATUS_SEVERITY,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'


def make_mapper():
    m = CrossDatasetTypeMapper.__new__(CrossDatasetTypeMapper)
    m._loaded = True
    m.include_suspects_in_targets = False
    m._type_mappings = {}
    m._conflicts = []
    m._mcns_v09_shared_names = set()
    m._bridge_provenance = {}
    m._fan_in_index = {}
    m._conflicted_links_index = None
    m._type_population_counts_cache = {}
    m._workspace_path = None
    m._type_namespaces = {}
    m._unsupported_dataset_warnings = set()
    m._stale_type_claims = {}
    m.last_load_error = None
    m.verbose = False
    return m


def chains(*ends):
    """One-link same-name derivation chains ending at each given value."""
    return [[{'dataset': MCNS, 'column': 'type', 'value': 'QX'},
             {'dataset': FAFB, 'column': 'type', 'value': e}]
            for e in ends]


class EvidenceOnlyMapper(CrossDatasetTypeMapper):
    """`QX` is a refused reverse N-to-1 toward FAFB whose members are
    {FA}, yet one same-name bridge chain reaches FA."""

    def __init__(self):
        vars(self).update(vars(make_mapper()))
        self._type_mappings = {MCNS: {'QX': {}}}

    def get_mapping_conflicts(self, source_dataset, target_dataset,
                              source_type=None):
        if source_type == 'QX':
            return []
        return []

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, *, include_bridges=True):
        if source_type == 'QX' and source_dataset == MCNS:
            return {'status': 'evidence_only', 'target_type': None,
                    'target_types': ['FA'], 'relationship': 'N-to-1',
                    'conflicts': [], 'support': None}
        return {'status': 'unmapped', 'target_type': None,
                'target_types': [], 'conflicts': [], 'support': None}

    def get_type_bridges(self, type_name, source_dataset, target_dataset,
                         max_bridges=0):
        if type_name == 'QX':
            return chains('FA')
        return []


class StaleClaimMapper(CrossDatasetTypeMapper):
    """`ST` maps to a phantom FAFB target the dataset does not carry."""

    def __init__(self):
        vars(self).update(vars(make_mapper()))
        self._type_mappings = {MCNS: {'ST': {FAFB: 'GHOST'}}}
        self._stale_type_claims = {
            (MCNS, FAFB): {'ST': {'GHOST'}}}

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, *, include_bridges=True):
        if source_type == 'ST':
            return {'status': 'mapped', 'target_type': 'GHOST',
                    'target_types': ['GHOST'], 'relationship': '1-to-1',
                    'conflicts': [], 'support': None}
        return {'status': 'unmapped', 'target_type': None,
                'target_types': [], 'conflicts': [], 'support': None}

    def get_type_bridges(self, type_name, source_dataset, target_dataset,
                         max_bridges=0):
        return []


def test_res1_evidence_only_with_bridge_end_stays_evidence_only():
    m = EvidenceOnlyMapper()
    res = resolve_valid_targets(m, 'QX', MCNS, FAFB)
    # the decision's refusal survives the bridge-end union
    assert res.status == 'evidence_only', res.status
    assert res.kind == 'one of N'
    # no licensed equivalence may appear
    assert res.equivalence_key is None
    # the decision's own evidence member joined the union
    assert 'FA' in res.target_types


def test_res8_native_realized_identity_is_not_curated():
    m = StaleClaimMapper()
    # the raw name is native in the target: the identity is realized
    m._type_namespaces = {FAFB: {'ST'}}
    m._dataset_types = {FAFB: {}}
    res = resolve_valid_targets(m, 'ST', MCNS, FAFB)
    assert res.status == 'mapped' and res.kind == 'same name'
    assert res.curated_identity is False


def test_res11_merge_key_demotes_stale_claim_to_raw():
    m = StaleClaimMapper()
    snap = MapperSnapshot(m)
    key = canonical_merge_key(m, 'ST', MCNS, FAFB, snapshot=snap)
    assert key.key == 'ST'
    assert key.status == 'claimed'


def test_res11_profile_expansion_keeps_stale_weight_local():
    m = StaleClaimMapper()
    exp = expand_profile_types(m, {'ST': 2.0}, MCNS, FAFB)
    assert 'ST' in exp.canonical
    assert exp.canonical['ST'] == 2.0
    assert exp.key_status.get('ST') == 'claimed'
    assert 'GHOST' not in exp.canonical


def test_res12_severity_orders_claimed_below_mapped():
    assert _severity('claimed') in range(len(_STATUS_SEVERITY))
    assert _severity('claimed') < _severity('mapped')


def test_res10_member_targets_needs_unique_or_split():
    from comparison.query_resolver import _member_targets

    class UnionMapper(CrossDatasetTypeMapper):
        """RES-1 behavior: evidence-only union renders as one-of-N
        evidence-only (never STATUS_MAPPED)."""
        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, *, include_bridges=True):
            return {'status': 'evidence_only', 'target_type': None,
                    'target_types': ['FB1', 'FB2'], 'relationship': 'N-to-1',
                    'conflicts': [], 'support': None}

        def get_type_bridges(self, *a, **k):
            return chains('FB1', 'FB2')

    assert _member_targets(UnionMapper(), 'QX', MCNS, FAFB) == []

    class UniqueMapper(CrossDatasetTypeMapper):
        def __init__(self):
            vars(self).update(vars(make_mapper()))
        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, *, include_bridges=True):
            return {'status': 'mapped', 'target_type': 'UNIQ',
                    'target_types': ['UNIQ'], 'relationship': '1-to-1',
                    'conflicts': [], 'support': None}

        def get_type_bridges(self, *a, **k):
            return []

    assert _member_targets(UniqueMapper(), 'ST', MCNS, FAFB) == ['UNIQ']


def test_res3_relabel_populates_target_types_and_res13_cache_key():
    """RES-3: a relabeled (unmapped→bridged/valid_split) flow carries the
    discovered end names as its target_types — flow_is_claimed excluded
    every relabeled flow while the panel's other half counted the same
    ends as mapped.  RES-13: bridge_end_count participates in the cache
    key — it is the only parameter that changes the relabel."""
    m = EvidenceOnlyMapper()   # get_mapping_decision: QX → evidence_only on MCNS
    m.get_mapping_decision = lambda t, s, tg, *, include_bridges=True: (
        {'status': 'unmapped', 'target_type': None, 'target_types': [],
         'conflicts': [], 'support': None})
    m.get_type_bridges = lambda *a, **k: []
    cache = {}
    s1, f1 = resolve_flow_status(m, 'QX', MCNS, FAFB,
                                 bridge_end_count=1,
                                 bridge_ends=('FA',), cache=cache)
    assert s1 == 'bridged'
    assert f1['target_types'] == ['FA']
    # same edge, different end count → NOT a cache hit on the first answer
    s2, f2 = resolve_flow_status(m, 'QX', MCNS, FAFB,
                                 bridge_end_count=2,
                                 bridge_ends=('FA', 'FB'), cache=cache)
    assert s2 == 'valid_split_evidence'
    assert f2['target_types'] == ['FA', 'FB']
    # a non-relabel keeps the decision's own target_types verbatim
    m2 = StaleClaimMapper()
    s3, f3 = resolve_flow_status(m2, 'ST', MCNS, FAFB,
                                 bridge_end_count=1, bridge_ends=('X',))
    assert s3 == 'mapped' and f3['target_types'] == ['GHOST']
