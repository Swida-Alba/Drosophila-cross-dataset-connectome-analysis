"""Contract tests for comparison.display_labels (§6.1 of the node-label issue)."""

import pytest

from comparison import display_labels as dl
from comparison.type_resolver import (
    STATUS_BRIDGED,
    STATUS_CLAIMED,
    STATUS_CONFLICT,
    STATUS_EVIDENCE_ONLY,
    STATUS_MAPPED,
    STATUS_UNMAPPED,
    STATUS_VALID_SPLIT,
    TypeResolution,
)

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'banc_v888'
HB = 'hemibrain:v1.2.1'


class FakeMapper:
    _loaded = True
    last_load_error = None
    _neuron_df_path = None

    def __init__(self, sources=None):
        self.sources = dict(sources or {})

    def _get_type_mapping_key(self, dataset):
        text = str(dataset).lower()
        if 'male-cns' in text:
            return MCNS
        if 'fafb' in text or 'flywire' in text:
            return FW
        if 'banc' in text:
            return BANC
        if 'hemibrain' in text:
            return HB
        return str(dataset)

    def _detect_type_source(self, name):
        return self.sources.get(str(name))

    def _get_dataset_short_codes(self, datasets):
        codes = []
        for dataset in datasets:
            key = self._get_type_mapping_key(dataset)
            codes.append({
                MCNS: 'M', FW: 'F', BANC: 'B', HB: 'H',
            }.get(key, str(dataset)[0].upper()))
        return codes

    def get_dataset_short_code(self, dataset):
        return self._get_dataset_short_codes([dataset])[0]


class FakeResolver:
    def __init__(self, table):
        self.table = dict(table)
        self.calls = []

    def __call__(self, mapper, source_type, source_dataset, target_dataset,
                 snapshot=None, **_kwargs):
        self.calls.append((source_type, source_dataset, target_dataset))
        step = self.table.get((str(source_type), str(target_dataset)))
        if step is None:
            return TypeResolution(
                status=STATUS_UNMAPPED, source_type=str(source_type),
                source_dataset=str(source_dataset or ''),
                target_dataset=str(target_dataset))
        status, targets = step
        return TypeResolution(
            status=status, source_type=str(source_type),
            source_dataset=str(source_dataset or ''),
            target_dataset=str(target_dataset),
            target_types=tuple(targets))


@pytest.fixture
def fake_resolver(monkeypatch):
    resolver = FakeResolver({})
    monkeypatch.setattr(dl, 'resolve_valid_targets', resolver)
    return resolver


def build(name, resolver, *, observed=None, mapper=None, **kwargs):
    return dl.build_node_display(
        name, [MCNS, FW, BANC], mapper=mapper or FakeMapper(),
        observed=observed, **kwargs)


def test_smp523_atomic_banc_compound(fake_resolver):
    fake_resolver.table[('SMP523', FW)] = (STATUS_MAPPED, ('SMP523',))
    fake_resolver.table[('SMP523', BANC)] = (
        STATUS_MAPPED, ('SMP523,SMP524',))
    display = build('SMP523', fake_resolver,
                    mapper=FakeMapper({'SMP523': MCNS}))

    assert display.label('network') == 'SMP523\n(SMP523,SMP524)'
    assert display.alternatives == ('SMP523,SMP524',)
    assert display.hover()['B'] == 'SMP523,SMP524'
    assert dl.format_hover_value(display.hover()['B']) == (
        '"SMP523,SMP524"')
    assert display.tier == 'canonical'
    assert display.decorated is True


def test_smp344_single_alternative(fake_resolver):
    fake_resolver.table[('SMP344', BANC)] = (STATUS_MAPPED, ('SMP344b',))
    display = build('SMP344', fake_resolver,
                    mapper=FakeMapper({'SMP344': MCNS}))

    assert display.label('network') == 'SMP344\n(SMP344b)'
    assert list(display.hover()) == ['M', 'B']


def test_observed_rename(fake_resolver):
    fake_resolver.table[('aDT4', FW)] = (STATUS_MAPPED, ('AC neuron',))
    fake_resolver.table[('aDT4', BANC)] = (STATUS_MAPPED, ('AC neuron',))
    display = build(
        'aDT4', fake_resolver, mapper=FakeMapper({'aDT4': MCNS}),
        observed={'aDT4': {FW: ['AC neuron'], BANC: ['AC neuron']}})

    assert display.label('network') == 'aDT4\n(AC neuron)'
    assert display.hover()['F'] == 'AC neuron'
    assert display.hover()['B'] == 'AC neuron'


def test_observed_and_mapper_names_mix(fake_resolver):
    fake_resolver.table[('Dm8a', FW)] = (STATUS_MAPPED, ('yDm8',))
    display = build(
        'Dm8a', fake_resolver, mapper=FakeMapper({'Dm8a': MCNS}),
        observed={'Dm8a': {FW: ['Dm8a', 'yDm8']}})

    assert display.label('network') == 'Dm8a\n(yDm8)'
    assert display.hover()['F'] == ['Dm8a', 'yDm8']
    assert dl.format_hover_value(display.hover()['F']) == 'Dm8a | yDm8'


def test_unmapped_dataset_is_absent(fake_resolver):
    fake_resolver.table[('aMe12', FW)] = (STATUS_MAPPED, ('aMe12',))
    display = build('aMe12', fake_resolver,
                    mapper=FakeMapper({'aMe12': MCNS}))

    assert display.label('network') == 'aMe12'
    assert display.alternatives == ()
    assert list(display.hover()) == ['M', 'F']


def test_slp249_rename(fake_resolver):
    fake_resolver.table[('SLP249', FW)] = (STATUS_MAPPED, ('APDN3',))
    display = build('SLP249', fake_resolver,
                    mapper=FakeMapper({'SLP249': MCNS}))

    assert display.label('network') == 'SLP249\n(APDN3)'
    assert list(display.hover()) == ['M', 'F']


def test_valid_split_keeps_all_targets(fake_resolver):
    fake_resolver.table[('CB2881', FW)] = (
        STATUS_VALID_SPLIT, ('CB2881', 'CB2881a', 'CB2881b'))
    display = build('CB2881', fake_resolver,
                    mapper=FakeMapper({'CB2881': MCNS}))

    assert display.label('network') == 'CB2881\n(CB2881a/CB2881b)'
    # the hover keeps the whole fan-out plus the status
    assert display.hover()['F'][:3] == ['CB2881', 'CB2881a', 'CB2881b']
    assert display.hover()['F'][-1] == f'({STATUS_VALID_SPLIT})'


def test_claimed_name_is_hover_only(fake_resolver):
    fake_resolver.table[('AMMC024', FW)] = (STATUS_CLAIMED, ('CB1038a',))
    display = build('AMMC024', fake_resolver,
                    mapper=FakeMapper({'AMMC024': MCNS}))

    assert display.label('network') == 'AMMC024'
    assert display.alternatives == ()
    assert display.hover()['F'] == f'CB1038a ({STATUS_CLAIMED})'


def test_unlicensed_statuses_do_not_decorate(fake_resolver):
    fake_resolver.table[('X1', FW)] = (STATUS_EVIDENCE_ONLY, ('FWx',))
    fake_resolver.table[('X1', BANC)] = (STATUS_CONFLICT, ())
    fake_resolver.table[('Y1', FW)] = (STATUS_BRIDGED, ('FWy',))
    fake_resolver.table[('Y1', BANC)] = (STATUS_UNMAPPED, ())
    mapper = FakeMapper({'X1': MCNS, 'Y1': MCNS})

    assert build('X1', fake_resolver, mapper=mapper).label() == 'X1'
    assert build('Y1', fake_resolver, mapper=mapper).label() == (
        'Y1\n(FWy)')


def test_primary_with_parentheses_keeps_boundary(fake_resolver):
    fake_resolver.table[('(PS023,PS024)a', FW)] = (
        STATUS_MAPPED, ('PS023a',))
    display = build('(PS023,PS024)a', fake_resolver,
                    mapper=FakeMapper({'(PS023,PS024)a': MCNS}))

    assert display.label('network') == '(PS023,PS024)a\n(PS023a)'
    assert display.label('inline') == '(PS023,PS024)a'
    assert 'inline_ambiguous' in display.warnings
    assert display.label('plain') == '(PS023,PS024)a'


def test_native_parenthetical_without_alternatives_is_plain(fake_resolver):
    display = build('PEN_a(PEN1)', fake_resolver,
                    mapper=FakeMapper({'PEN_a(PEN1)': MCNS}))

    assert display.label('network') == 'PEN_a(PEN1)'
    assert display.label('inline') == 'PEN_a(PEN1)'
    assert display.decorated is False


def test_verbatim_tier_zero(fake_resolver):
    display = build('CL125', fake_resolver, verbatim={'CL125'},
                    mapper=FakeMapper({'CL125': MCNS}))

    assert display.label('network') == 'CL125'
    assert display.tier == 'verbatim'
    assert display.decorated is False


def test_dataset_order_not_alphabetical(fake_resolver):
    fake_resolver.table[('AN05B102d', FW)] = (
        STATUS_MAPPED, ('AN_multi_69',))
    fake_resolver.table[('AN05B102d', BANC)] = (
        STATUS_MAPPED, ('ANXXX046',))
    display = build('AN05B102d', fake_resolver,
                    mapper=FakeMapper({'AN05B102d': MCNS}))

    assert display.label('network') == 'AN05B102d\n(AN_multi_69/ANXXX046)'
    assert list(display.hover()) == ['M', 'F', 'B']


def test_alternatives_never_repeat_primary(fake_resolver):
    fake_resolver.table[('Same1', FW)] = (STATUS_MAPPED, ('Same1',))
    display = build('Same1', fake_resolver,
                    mapper=FakeMapper({'Same1': MCNS}))

    assert display.primary == 'Same1'
    assert display.primary not in display.alternatives
    assert display.label('plain') == 'Same1'


def test_explicit_primary_only_when_licensed(fake_resolver):
    fake_resolver.table[('T1', FW)] = (STATUS_MAPPED, ('FW1',))
    mapper = FakeMapper({'T1': MCNS})

    licensed = build('T1', fake_resolver, mapper=mapper, primary='FW1')
    assert licensed.primary == 'FW1'
    assert licensed.label('inline') == 'FW1(T1)'

    rejected = build('T1', fake_resolver, mapper=mapper, primary='Nope')
    assert rejected.primary == 'T1'
    assert 'primary_not_mapped' in rejected.warnings


def test_inline_never_splits_unsafe_names(fake_resolver):
    fake_resolver.table[('pC1_2a', FW)] = (STATUS_MAPPED, ('pC1_2a/2b',))
    display = build('pC1_2a', fake_resolver,
                    mapper=FakeMapper({'pC1_2a': MCNS}))

    assert display.label('network') == 'pC1_2a\n(pC1_2a/2b)'
    assert display.label('inline') == 'pC1_2a'
    assert 'inline_ambiguous' in display.warnings


def test_parse_display_label_two_line():
    assert dl.parse_display_label('A\n(B/C)') == ('A', ('B', 'C'))
    assert dl.parse_display_label('(PS023,PS024)a\n(PS023a)') == (
        '(PS023,PS024)a', ('PS023a',))
    assert dl.parse_display_label('A(B/C)') == ('A', ('B', 'C'))
    assert dl.parse_display_label('PEN_a(PEN1)') == ('PEN_a', ('PEN1',))


def test_invert_edge_display_names():
    inverted = dl.invert_edge_display_names({
        'aMe12': {'aMe12': {MCNS}, 'MTe07': {FW}},
        'SMP523': {'SMP523,slim': {BANC}},
    })
    assert inverted == {
        'aMe12': {MCNS: ['aMe12'], FW: ['MTe07']},
        'SMP523': {BANC: ['SMP523,slim']},
    }


def test_canonical_key_stays_primary_over_unmapped_variant(fake_resolver):
    display = build(
        'SrcCanon', fake_resolver, mapper=FakeMapper(),
        observed={'SrcCanon': {MCNS: ['Src'], FW: ['Src']}})

    assert display.primary == 'SrcCanon'
    assert display.label('network') == 'SrcCanon\n(Src)'
    assert display.label('inline') == 'SrcCanon(Src)'


def test_inferred_source_does_not_create_identity(fake_resolver):
    display = build(
        'Canon2', fake_resolver, mapper=FakeMapper(),
        observed={'Canon2': {FW: ['Raw2']}})

    assert display.primary == 'Canon2'
    assert display.label('network') == 'Canon2\n(Raw2)'
    assert display.hover()['F'] == 'Raw2'


def test_merge_policy_group_is_verbatim(fake_resolver):
    class Group:
        members = {'M': ['a'], 'F': ['b']}

    class Policy:
        def group_by_label(self, label):
            return Group() if label == 'Group1' else None

        def names_by_dataset(self, label):
            return {'male-cns:v1.0': ['a'], 'flywire_FAFB_v783': ['b']}

    display = dl.build_node_display(
        'Group1', [MCNS, FW], mapper=FakeMapper(),
        merge_policy=Policy())

    assert display.label('network') == 'Group1'
    assert display.tier == 'verbatim'
    assert display.hover() == {'M': 'a', 'F': 'b'}
