"""Standalone input-query resolver (plan §7E).

Verifies the token-class matrix: type tokens go through the shared validity
core, taxonomy values expand per dataset, bodyIds stay dataset-scoped, and
a bare same-name echo is surfaced as an explicit low-confidence fallback
rather than an invisible pass-through.
"""

from comparison import query_resolver as qr


class _Res:
    def __init__(self, status, kind='', equivalence_key=None,
                 target_types=()):
        self.status = status
        self.kind = kind
        self.equivalence_key = equivalence_key
        self.target_types = tuple(target_types)


class FakeMapper:
    """Minimal shared-resolver contract double."""

    _loaded = True

    def __init__(self, decisions, native=None):
        self._decisions = decisions
        self._native = native or {}

    def detect_type_source(self, token):
        return self._decisions.get(token, {}).get('source_ds')

    def has_native_type(self, token, ds):
        return token in self._native.get(ds, set())

    def get_alias_candidates(self, raw, targets, source_dataset=None):
        return {t: {'outcome': 'none', 'candidates': []} for t in targets}

    def get_type_bridges(self, *a, **k):
        return []

    def _get_type_mapping_key(self, d):
        return d

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, include_bridges=False):
        spec = self._decisions.get(source_type, {})
        per = spec.get('per', {}).get(target_dataset)
        if per == 'same_name':
            return {'status': 'mapped', 'source_type': source_type,
                    'target_type': source_type,
                    'target_types': [source_type],
                    'relationship': '1-to-1', 'conflicts': []}
        if per == 'conflict':
            return {'status': 'conflict', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': [{'origin': 'vote'}]}
        if per:
            return {'status': 'mapped', 'source_type': source_type,
                    'target_type': per, 'target_types': [per],
                    'relationship': '1-to-1', 'conflicts': []}
        return {'status': 'unmapped', 'source_type': source_type,
                'target_type': None, 'target_types': [],
                'relationship': None, 'conflicts': []}


def test_type_token_uses_mapped_evidence():
    mapper = FakeMapper({'aMe12': {'source_ds': 'banc', 'per': {
        'fafb': 'aMe12_fafb'}}})
    recs = qr.resolve_query_tokens(
        ['aMe12'], ['banc', 'fafb'], mapper)
    by_ds = {r['dataset']: r for r in recs}
    assert by_ds['fafb']['method'] == 'mapped_type'
    assert by_ds['fafb']['target_types'] == ['aMe12_fafb']
    assert by_ds['fafb']['confidence'] >= qr.CONFIDENCE[qr.STATUS_MAPPED]


def test_same_name_echo_is_flagged_low_confidence():
    mapper = FakeMapper({'aMe12': {'source_ds': 'banc',
                                   'per': {'fafb': 'same_name'}}})
    recs = qr.resolve_query_tokens(['aMe12'], ['banc', 'fafb'], mapper)
    fafb = [r for r in recs if r['dataset'] == 'fafb'][0]
    assert fafb['status'] == qr.STATUS_SAME_NAME
    assert fafb['method'] == 'same_name'
    assert fafb['confidence'] == qr.CONFIDENCE[qr.STATUS_SAME_NAME]
    assert 'same-name' in fafb['note'].lower() or \
        'tier 6' in fafb['note']


def test_body_id_is_dataset_scoped():
    recs = qr.resolve_query_tokens([12345], ['a', 'b'], None)
    assert all(r['method'] == 'body_id' for r in recs)
    assert all(r['matched_column'] == 'bodyId' for r in recs)


def test_pattern_passthrough():
    recs = qr.resolve_query_tokens(['aMe.*'], ['a'], None)
    assert recs[0]['method'] == 'pattern'
    assert recs[0]['matched_column'] is None


def test_taxonomy_value_expands_only_where_present():
    def taxonomy_resolver(token, ds):
        return ['clock_A', 'clock_B'] if ds == 'fafb' else None

    recs = qr.resolve_query_tokens(
        ['circadian_clock'], ['fafb', 'banc'], None,
        taxonomy_resolver=taxonomy_resolver)
    by_ds = {r['dataset']: r for r in recs}
    assert by_ds['fafb']['status'] == qr.STATUS_TAXONOMY
    assert by_ds['fafb']['target_types'] == ['clock_A', 'clock_B']
    assert by_ds['banc']['status'] == qr.STATUS_UNMAPPED


def test_group_label_detected():
    recs = qr.resolve_query_tokens(
        ['MyGroup'], ['a'], None, group_lookup=lambda t: t == 'MyGroup')
    assert recs[0]['status'] == qr.STATUS_GROUP
    assert recs[0]['method'] == 'group'


def test_mapper_unavailable_falls_back_to_same_name():
    recs = qr.resolve_query_tokens(['aMe12'], ['a'], None)
    assert recs[0]['status'] in (qr.STATUS_SAME_NAME, qr.STATUS_UNMAPPED)
    assert recs[0]['method'] in ('same_name', 'unmatched')


# --- Phase 1: same-name-first identity tier (plan §2.1) ---------------------

def test_native_same_name_becomes_identity_confirmed():
    mapper = FakeMapper(
        {'L2': {'source_ds': 'mcns', 'per': {'fafb': 'same_name'}}},
        native={'fafb': {'L2'}})
    recs = qr.resolve_query_tokens(['L2'], ['mcns', 'fafb'], mapper)
    by = {r['dataset']: r for r in recs}
    assert by['mcns']['status'] == qr.STATUS_SAME_NAME_IDENTITY
    assert by['mcns']['method'] == 'native_type'
    assert by['mcns']['confidence'] == 4
    assert by['fafb']['status'] == qr.STATUS_SAME_NAME_IDENTITY
    assert by['fafb']['evidence'] == qr.EVIDENCE_CONFIRMED
    assert by['fafb']['target_types'] == ['L2']


def test_identity_without_curated_relation_is_evidence_none():
    mapper = FakeMapper({'L2': {'source_ds': 'mcns', 'per': {}}},
                        native={'banc': {'L2'}})
    recs = qr.resolve_query_tokens(['L2'], ['mcns', 'banc'], mapper)
    banc = [r for r in recs if r['dataset'] == 'banc'][0]
    assert banc['status'] == qr.STATUS_SAME_NAME_IDENTITY
    assert banc['evidence'] == qr.EVIDENCE_NONE
    assert banc['target_types'] == ['L2']
    assert 'no curated relation' in banc['note']


def test_contradicted_identity_kept_with_counter_evidence():
    mapper = FakeMapper(
        {'Dm8a': {'source_ds': 'mcns', 'per': {'fafb': 'yDm8'}}},
        native={'fafb': {'Dm8a'}})
    recs = qr.resolve_query_tokens(['Dm8a'], ['mcns', 'fafb'], mapper)
    fafb = [r for r in recs if r['dataset'] == 'fafb'][0]
    assert fafb['status'] == qr.STATUS_SAME_NAME_IDENTITY
    assert fafb['evidence'] == qr.EVIDENCE_CONTRADICTED
    # the identity stays the query answer; the curated counterpart is shown
    assert fafb['target_types'] == ['Dm8a']
    assert 'yDm8' in fafb['evidence_chain']


def test_conflict_refuses_even_native_identity():
    mapper = FakeMapper(
        {'AN01': {'source_ds': 'banc', 'per': {'mcns': 'conflict'}}},
        native={'mcns': {'AN01'}})
    recs = qr.resolve_query_tokens(['AN01'], ['banc', 'mcns'], mapper)
    mcns = [r for r in recs if r['dataset'] == 'mcns'][0]
    assert mcns['status'] == qr.STATUS_CONFLICT


def test_non_native_echo_stays_fallback():
    mapper = FakeMapper(
        {'aMe12': {'source_ds': 'banc', 'per': {'fafb': 'same_name'}}},
        native={'fafb': set()})
    recs = qr.resolve_query_tokens(['aMe12'], ['banc', 'fafb'], mapper)
    fafb = [r for r in recs if r['dataset'] == 'fafb'][0]
    assert fafb['status'] == qr.STATUS_SAME_NAME
    assert fafb['confidence'] == qr.CONFIDENCE[qr.STATUS_SAME_NAME]


# --- Phase 2: Route A taxonomy member bridging (plan §2.2) ------------------

def test_taxonomy_members_bridge_into_other_datasets():
    mapper = FakeMapper({'clock_A': {'source_ds': 'a',
                                     'per': {'b': 'clock_A_b'}}})

    def taxonomy_resolver(token, ds):
        return ['clock_A', 'lonely'] if ds == 'a' else None

    recs = qr.resolve_query_tokens(
        ['circadian'], ['a', 'b'], mapper,
        taxonomy_resolver=taxonomy_resolver)
    by = {r['dataset']: r for r in recs}
    assert by['a']['status'] == qr.STATUS_TAXONOMY
    assert by['a']['target_types'] == ['clock_A', 'lonely']
    # Route A: 'b' gets the mapped union; the member without a counterpart
    # is not queried there but is counted in the note.
    assert by['b']['status'] == qr.STATUS_TAXONOMY_MAPPED
    assert by['b']['method'] == 'taxonomy_member_mapping'
    assert by['b']['target_types'] == ['clock_A_b']
    assert 'lonely' in by['b']['note']


def test_taxonomy_without_hits_stays_unmapped():
    recs = qr.resolve_query_tokens(
        ['nothing'], ['a', 'b'], None,
        taxonomy_resolver=lambda t, ds: None)
    assert all(r['status'] in (qr.STATUS_UNMAPPED, qr.STATUS_SAME_NAME)
               for r in recs)
