"""Coverage tests for comparison.cross_dataset_type_mapper.

Hermetic: all mappers are built from synthetic neuron_df CSVs written to
pytest tmp_path. The real local neuron info file is only touched when it
exists (guarded by Path(...).exists()).
"""

import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

from comparison import cross_dataset_type_mapper as mapper_module
from comparison.cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    TypeMappingConflict,
    TypeMappingWarning,
    get_type_mapper,
)
from comparison.label_mapper import LabelMapper

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'banc_v626'
HB = 'hemibrain:v1.2.1'
MANC = 'manc:v1.0'

CSV_ROWS = (
    "bodyId,type,flywireType,hemibrainType,mancType\n"
    "1,aMe12,MTe07,aMe12,MN1\n"          # clean 1-to-1 everywhere
    "2,Same1,Same1,Same1,Same1\n"        # identical names in all datasets
    "3,SplitN,FWa,,\n"                   # 1-to-N: SplitN -> {FWa, FWb}
    "4,SplitN,FWb,,\n"
    "5,AggA,AggFW,,\n"                   # N-to-1: AggFW -> {AggA, AggB}
    "6,AggB,AggFW,,\n"
    "7,HBn1,HX,HBa,\n"                   # N-to-1 hemibrain HBa -> {HBn1, HBn2}
    "8,HBn2,HX,HBa,\n"
    "9,Mn1,MN1m,,MNx\n"                  # N-to-1 manc MNx -> {Mn1, Mn2}
    "10,Mn2,MN2m,,MNx\n"
    "11,Clash,ClashFw,,\n"               # for check_type_name_conflict
    "12,ClashFw,OtherFw,,\n"
)


@pytest.fixture
def csv_path(tmp_path):
    csv = tmp_path / 'neurons.csv'
    csv.write_text(CSV_ROWS, encoding='utf-8')
    return str(csv)


@pytest.fixture
def mapper(csv_path):
    m = CrossDatasetTypeMapper(neuron_df_path=csv_path, verbose=False)
    assert m.load() is True
    return m


# ---------------------------------------------------------------------------
# FlyWire additional Type(S) rename resolution
# ---------------------------------------------------------------------------
#
# FAFB carries an `additional_type(s)` column and BANC an
# `Alternative Cell Type(s)` column listing previous/alternative type names.
# A male-cns flywireType that is no longer a primary type in the target
# dataset must resolve to the current primary name (SLP249 -> APDN3).

MCNS_RENAME_ROWS = (
    "bodyId,type,flywireType,hemibrainType,mancType\n"
    "1,Mc249,SLP249,,\n"            # renamed in FAFB -> APDN3
    "2,McCB,CB1215,,\n"             # both resolve to LPN via one comma cell
    "3,McPV,PV7c11,,\n"
    "4,McOld,OldSplit,,\n"          # ambiguous: SplitA vs SplitB
    "5,McDirect,MTe07,,\n"          # primary passthrough
    "6,MDNx,MDN,,\n"                # FAFB renames MDN -> DNp50, BANC keeps MDN
    "7,Bonly1,BOnly,,\n"            # only BANC resolves it
    "8,McVS,\"VS1,VS2\",,\n"        # comma in crosswalk cell -> 1-to-N
)

FAFB_RENAME_TABLE = (
    "bodyId,type,instance,additional_type(s)\n"
    "f1,APDN3,APDN3_1,SLP249\n"
    "f2,LPN,LPN_1,\"CB1215, PV7c11\"\n"
    "f3,SplitA,SplitA_1,OldSplit\n"
    "f4,SplitB,SplitB_1,OldSplit\n"
    "f5,MTe07,MTe07_1,\n"
    "f6,DNp50,DNp50_1,MDN\n"
)

BANC_RENAME_TABLE = (
    "bodyId,type,instance,Alternative Cell Type(s),malecns_cell_type,fafb_cell_type\n"
    "b1,MDN,MDN_1,,MDNx,DNp50\n"
    "b2,SOMEB,SOMEB_1,BOnly,Bonly1,\n"
)


@pytest.fixture
def rename_mapper(tmp_path):
    mcns_csv = tmp_path / 'mcns.csv'
    mcns_csv.write_text(MCNS_RENAME_ROWS, encoding='utf-8')
    fafb_csv = tmp_path / 'fafb.csv'
    fafb_csv.write_text(FAFB_RENAME_TABLE, encoding='utf-8')
    banc_csv = tmp_path / 'banc.csv'
    banc_csv.write_text(BANC_RENAME_TABLE, encoding='utf-8')
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(mcns_csv),
        flywire_neuron_df_paths={
            'flywire_FAFB_v783': str(fafb_csv),
            'banc_v626': str(banc_csv),
        },
        verbose=False,
    )
    assert m.load() is True
    return m


def test_flywire_rename_resolves_to_primary_type(rename_mapper):
    # male-cns Mc249 -> FAFB APDN3 (its flywireType SLP249 only exists as
    # FAFB additional_type(s)); reverse direction stays unique here.
    assert rename_mapper.get_mapped_type('Mc249', MCNS, FW) == 'APDN3'
    assert rename_mapper.get_mapped_type('APDN3', FW, MCNS) == 'Mc249'
    assert rename_mapper.get_canonical_type('APDN3', FW) == 'Mc249'


def test_mapper_snapshot_fresh_workspace_first_run(tmp_path):
    """§startup (2026-09-10): a FRESH DROCAT workspace must handle the
    state snapshot gracefully at every stage — nothing initialized (no
    crash, no snapshot, load refuses), first dataset initialized (rebuild
    + snapshot written), and every later load restores from it without
    rewriting it."""
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    uninitialized = CrossDatasetTypeMapper(
        verbose=False, workspace_path=str(workspace))
    snap = uninitialized._mapper_snapshot_path()
    assert snap is not None
    assert not snap.exists()
    # nothing initialized yet: graceful refusal, no snapshot, no error
    assert uninitialized.load() is False
    assert 'neuron_df not found' in (uninitialized.last_load_error or '')

    # the user initializes the male-cns dataset (conventional layout)
    datasets = workspace / 'datasets' / 'male-cns_v1_0'
    datasets.mkdir(parents=True)
    csv = datasets / 'male-cns_v1_0_allneurons_neuron_df.csv'
    csv.write_text(CSV_ROWS, encoding='utf-8')

    first = CrossDatasetTypeMapper(
        neuron_df_path=str(csv), verbose=False,
        workspace_path=str(workspace))
    assert first.load() is True
    assert snap.exists()  # rebuilt once, snapshot written

    # the next process (isolated worker, app restart) restores from it
    second = CrossDatasetTypeMapper(
        neuron_df_path=str(csv), verbose=False,
        workspace_path=str(workspace))
    stamp = snap.stat().st_mtime_ns
    assert second.load() is True
    assert snap.stat().st_mtime_ns == stamp  # restored, not rebuilt
    assert second._loaded and not second.last_load_error
    # the restored mapper answers like the built one
    assert second.get_type_bridges('aMe12', MCNS, FW, max_bridges=0) == \
        first.get_type_bridges('aMe12', MCNS, FW, max_bridges=0)
    assert second.get_type_bridges('aMe12', MCNS, FW, max_bridges=0)


def test_flywire_rename_comma_separated_additional_cell(rename_mapper):
    # one FAFB additional cell lists two old names; both map to LPN
    assert rename_mapper.get_mapped_type('McCB', MCNS, FW) == 'LPN'
    assert rename_mapper.get_mapped_type('McPV', MCNS, FW) == 'LPN'
    # reverse is N-to-1 (two mcns types share LPN) -> not mapped
    assert rename_mapper.get_mapped_type('LPN', FW, MCNS) is None
    assert rename_mapper.is_n_to_1_type('LPN', FW) is True


def test_flywire_ambiguous_rename_is_conflict(rename_mapper):
    # OldSplit is listed as additional type of two FAFB primaries ->
    # cannot pick one, so no mapping and a 1-to-N conflict is recorded.
    assert rename_mapper.get_mapped_type('McOld', MCNS, FW) is None
    conflicts = rename_mapper.get_1_to_n_conflicts()
    assert any(
        c.source_type == 'McOld' and c.target_types == {'SplitA', 'SplitB'}
        for c in conflicts
    )


def test_flywire_primary_name_passthrough(rename_mapper):
    assert rename_mapper.get_mapped_type('McDirect', MCNS, FW) == 'MTe07'
    assert rename_mapper.get_mapped_type('MTe07', FW, MCNS) == 'McDirect'


def test_fafb_banc_namespaces_resolve_independently(rename_mapper):
    # FAFB renamed MDN -> DNp50; BANC's curated columns independently
    # identify the BANC MDN row from both MCNS and FAFB.
    assert rename_mapper.get_mapped_type('MDNx', MCNS, FW) == 'DNp50'
    assert rename_mapper.get_mapped_type('MDNx', MCNS, BANC) == 'MDN'
    # FAFB <-> BANC is direct label evidence, not a MCNS flywireType route.
    assert rename_mapper.get_mapped_type('DNp50', FW, BANC) == 'MDN'
    assert rename_mapper.get_mapped_type('MDN', BANC, FW) == 'DNp50'

    # The MCNS→FAFB crosswalk remains independent of the MCNS→BANC label;
    # BOnly is grounded by FAFB's own Alternative/primary table.
    assert rename_mapper.get_mapped_type('Bonly1', MCNS, BANC) == 'SOMEB'
    assert rename_mapper.get_mapped_type('Bonly1', MCNS, FW) == 'BOnly'


def test_comma_separated_crosswalk_cell_is_one_to_n(rename_mapper):
    # 'VS1,VS2' expands to two FAFB primaries -> conflict, no auto mapping
    assert rename_mapper.get_mapped_type('McVS', MCNS, FW) is None
    assert any(
        c.source_type == 'McVS' and c.target_types == {'VS1', 'VS2'}
        for c in rename_mapper.get_1_to_n_conflicts()
    )


def test_rename_display_name_uses_primary_name(rename_mapper):
    assert rename_mapper.get_display_name('Mc249', [MCNS, FW]) == 'Mc249(APDN3)'


def test_missing_flywire_table_disables_rename(tmp_path):
    mcns_csv = tmp_path / 'mcns.csv'
    mcns_csv.write_text(MCNS_RENAME_ROWS, encoding='utf-8')
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(mcns_csv), verbose=False)
    assert m.load() is True
    # without FAFB/BANC tables the crosswalk names pass through unchanged
    assert m.get_mapped_type('Mc249', MCNS, FW) == 'SLP249'
    # explicit None also disables resolution per namespace
    m2 = CrossDatasetTypeMapper(
        neuron_df_path=str(mcns_csv),
        flywire_neuron_df_paths={'flywire_FAFB_v783': None},
        verbose=False,
    )
    assert m2.load() is True
    assert m2.get_mapped_type('Mc249', MCNS, FW) == 'SLP249'
    assert m2._flywire_alt_to_primary.get('flywire_FAFB_v783') is None


def test_mapper_prefers_fresh_prepared_parquet_index(tmp_path):
    """Cold mapper loads use the prepared projection when it is current."""
    import polars as pl

    dataset_dir = tmp_path / 'datasets' / 'male-cns_v1_0'
    dataset_dir.mkdir(parents=True)
    source = dataset_dir / 'male-cns_v1_0_allneurons_neuron_df.csv'
    source.write_text(
        'bodyId,type,flywireType,hemibrainType,mancType\n'
        '1,CSVOnly,CSVOnly,,\n', encoding='utf-8')

    index_dir = tmp_path / 'neuron_indexes' / 'male-cns_v1_0'
    index_dir.mkdir(parents=True)
    pl.DataFrame({
        'bodyId': ['1'],
        'type': ['IndexOnly'],
        'flywireType': ['IndexOnly'],
        'hemibrainType': [''],
        'mancType': [''],
    }).write_parquet(index_dir / 'neuron_index.parquet')

    mapper = CrossDatasetTypeMapper(
        workspace_path=str(tmp_path), verbose=False)
    assert mapper.load() is True
    assert 'IndexOnly' in mapper._dataset_types[MCNS]
    assert 'CSVOnly' not in mapper._dataset_types[MCNS]


# ---------------------------------------------------------------------------
# User warning notes (expanded / N-to-1 / 1-to-N)
# ---------------------------------------------------------------------------

def test_build_user_warning_notes_expanded_rename(rename_mapper):
    notes = rename_mapper.build_user_warning_notes(['Mc249'], [MCNS, FW])
    # one expanded note + the double-check advice
    assert len(notes) == 2
    assert "'Mc249'" in notes[0] and "'APDN3'" in notes[0]
    assert 'FAFB' in notes[0]
    assert any('double check' in n.lower() for n in notes)


def test_build_user_warning_notes_n_to_1(rename_mapper):
    # McCB and McPV share the FAFB type LPN: one N-to-1 note (per conflict,
    # not per queried type), plus expanded notes and the advice.
    notes = rename_mapper.build_user_warning_notes(['McCB', 'McPV'], [MCNS, FW])
    n_to_1_notes = [n for n in notes if 'N-to-1' in n]
    assert len(n_to_1_notes) == 1
    assert 'McCB' in n_to_1_notes[0] and 'McPV' in n_to_1_notes[0]
    assert 'LPN' in n_to_1_notes[0]
    assert any('double check' in n.lower() for n in notes)


def test_build_user_warning_notes_one_to_n(rename_mapper):
    notes = rename_mapper.build_user_warning_notes(['McOld'], [MCNS, FW])
    # no expanded note (ambiguous split maps to nothing) ...
    assert not any('expanded' in n for n in notes)
    assert any(
        '1-to-N' in n and 'SplitA' in n and 'SplitB' in n
        and 'no automatic mapping' in n
        for n in notes
    )
    assert any('double check' in n.lower() for n in notes)


def test_build_user_warning_notes_empty_cases(rename_mapper):
    # identical name everywhere -> no notes
    assert rename_mapper.build_user_warning_notes(['Same1'], [MCNS, FW]) == []
    # unknown types, empty inputs, patterns, non-str -> no notes
    assert rename_mapper.build_user_warning_notes(['NoSuchType'], [MCNS, FW]) == []
    assert rename_mapper.build_user_warning_notes([], [MCNS, FW]) == []
    assert rename_mapper.build_user_warning_notes(['Mc249'], []) == []
    assert rename_mapper.build_user_warning_notes(['Agg*'], [MCNS, FW]) == []
    assert rename_mapper.build_user_warning_notes([123], [MCNS, FW]) == []


def test_build_user_warning_notes_unloaded_mapper(tmp_path):
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    assert m.build_user_warning_notes(['Mc249'], [MCNS, FW]) == []


# ---------------------------------------------------------------------------
# Alias candidates (expanded viewer search)
# ---------------------------------------------------------------------------

ALIAS_ROWS = (
    "bodyId,type,flywireType,hemibrainType,mancType\n"
    "21,Duo,Duo,,\n"
    "22,DuoB,DuoB,,\n"
)
ALIAS_FAFB_TABLE = (
    "bodyId,type,instance,additional_type(s)\n"
    "g1,Duo,Duo_1,DuoB\n"
)


@pytest.fixture
def alias_mapper(tmp_path):
    """Same-name candidate carrying the orthogonal aggregation annotation:
    FAFB 'Duo' is native and aggregates male-cns Duo + DuoB."""
    mcns_csv = tmp_path / 'mcns_alias.csv'
    mcns_csv.write_text(ALIAS_ROWS, encoding='utf-8')
    fafb_csv = tmp_path / 'fafb_alias.csv'
    fafb_csv.write_text(ALIAS_FAFB_TABLE, encoding='utf-8')
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(mcns_csv),
        flywire_neuron_df_paths={
            'flywire_FAFB_v783': str(fafb_csv),
            'banc_v626': None,
        },
        verbose=False,
    )
    assert m.load() is True
    return m


def test_get_alias_candidates_renamed(rename_mapper):
    res = rename_mapper.get_alias_candidates('Mc249', [MCNS, FW])
    assert res[MCNS]['outcome'] == 'matched'
    assert res[MCNS]['candidates'] == [
        {'name': 'Mc249', 'kind': 'same name', 'aggregates': None},
    ]
    assert res[FW]['outcome'] == 'matched'
    assert res[FW]['candidates'] == [
        {'name': 'APDN3', 'kind': 'renamed', 'aggregates': None},
    ]


def test_get_alias_candidates_renamed_with_aggregates(rename_mapper):
    # LPN receives McCB and McPV: the renamed candidate must warn that a
    # match by it also covers the sibling type.
    res = rename_mapper.get_alias_candidates('McCB', [MCNS, FW])
    assert res[FW]['candidates'] == [
        {'name': 'LPN', 'kind': 'renamed', 'aggregates': ['McCB', 'McPV']},
    ]


def test_get_alias_candidates_same_name_with_aggregates(alias_mapper):
    res = alias_mapper.get_alias_candidates('Duo', [MCNS, FW])
    assert res[MCNS]['candidates'] == [
        {'name': 'Duo', 'kind': 'same name', 'aggregates': None},
    ]
    assert res[FW]['candidates'] == [
        {'name': 'Duo', 'kind': 'same name', 'aggregates': ['Duo', 'DuoB']},
    ]
    # ...and the reverse: DuoB is renamed to Duo, with the same warning.
    res_b = alias_mapper.get_alias_candidates('DuoB', [MCNS, FW])
    assert res_b[FW]['candidates'] == [
        {'name': 'Duo', 'kind': 'renamed', 'aggregates': ['Duo', 'DuoB']},
    ]


def test_get_alias_candidates_reverse_split(rename_mapper):
    # Round-16: LPN's reverse reaches the WHOLE male-cns member pool as a
    # licensed type-pool split (the crosswalk's reverse direction) — the
    # old 'one of N' refusal shrank taxonomy Route A reaches (real case:
    # circadian FAFB->MCNS 52 of the documented 219).
    res = rename_mapper.get_alias_candidates('LPN', [FW, MCNS])
    assert res[FW]['candidates'] == [
        {'name': 'LPN', 'kind': 'same name', 'aggregates': ['McCB', 'McPV']},
    ]
    assert [c['name'] for c in res[MCNS]['candidates']] == ['McCB', 'McPV']
    assert all(c['kind'] == 'splits into' for c in res[MCNS]['candidates'])
    assert all(c['aggregates'] is None for c in res[MCNS]['candidates'])


def test_get_alias_candidates_splits_into(rename_mapper):
    res = rename_mapper.get_alias_candidates('McOld', [MCNS, FW])
    assert res[MCNS]['outcome'] == 'matched'
    assert res[FW]['candidates'] == [
        {'name': 'SplitA', 'kind': 'splits into', 'aggregates': None},
        {'name': 'SplitB', 'kind': 'splits into', 'aggregates': None},
    ]


def test_get_alias_candidates_outcomes(rename_mapper):
    # unknown name -> explicit per-dataset outcome
    res = rename_mapper.get_alias_candidates('NoSuchType', [MCNS, FW])
    assert res[MCNS]['outcome'] == 'no counterpart known'
    assert res[FW]['outcome'] == 'no counterpart known'
    # non-plain-name queries are not applicable
    for query in ('123', 'Agg*', '', 'a', None, 123):
        res = rename_mapper.get_alias_candidates(query, [MCNS, FW])
        assert all(o['outcome'] == 'not applicable' for o in res.values()), query


def test_get_alias_candidates_unloaded_mapper(tmp_path):
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    res = m.get_alias_candidates('Mc249', [MCNS, FW])
    assert all(o['outcome'] == 'mapper unavailable' for o in res.values())


def test_get_alias_candidates_kinds_exclusive(rename_mapper):
    # property: every candidate carries exactly one valid kind, names are
    # unique per dataset, and the orthogonal flag never changes the kind.
    queries = ['Mc249', 'McCB', 'McPV', 'McOld', 'APDN3', 'LPN', 'MTe07',
               'Same1', 'Duo', 'McDirect']
    for query in queries:
        res = rename_mapper.get_alias_candidates(query, [MCNS, FW])
        for outcome in res.values():
            names = [c['name'] for c in outcome['candidates']]
            assert len(names) == len(set(names)), (query, names)
            for cand in outcome['candidates']:
                assert cand['kind'] in CrossDatasetTypeMapper.ALIAS_KINDS
                assert sum(k == cand['kind'] for k in CrossDatasetTypeMapper.ALIAS_KINDS) == 1


def test_get_alias_candidates_hemisphere_suffix(rename_mapper):
    res = rename_mapper.get_alias_candidates('Mc249_L', [MCNS, FW])
    assert res[FW]['candidates'][0]['name'] == 'APDN3_L'
    assert res[FW]['candidates'][0]['kind'] == 'renamed'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_missing_file(tmp_path):
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'nope.csv'), verbose=False)
    assert m.load() is False
    assert m.get_mapped_type('aMe12', MCNS, FW) is None
    assert m.resolve_type_across_datasets('aMe12', [FW]) == {FW: None}


def test_load_error_path(tmp_path):
    # pointing at a directory makes pd.read_csv raise -> graceful False
    d = tmp_path / 'not_a_csv'
    d.mkdir()
    m = CrossDatasetTypeMapper(neuron_df_path=str(d), verbose=False)
    assert m.load() is False


def test_load_cached_and_force_reload(mapper):
    assert mapper.load() is True               # already loaded -> cached
    assert mapper.load(force_reload=True) is True


def test_build_type_mappings_without_df():
    m = CrossDatasetTypeMapper(neuron_df_path='/nonexistent/x.csv',
                               verbose=False)
    m._build_type_mappings()  # neuron_df is None -> no-op
    assert m._type_mappings == {}


def test_type_mapper_loads_v1_0_schema(tmp_path):
    csv = tmp_path / 'neurons.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,aMe12,MTe07,aMe12,\n",
        encoding='utf-8',
    )
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    assert 'male-cns:v1.0' in m._type_mappings


def test_real_neuron_info_guarded():
    real = CrossDatasetTypeMapper(verbose=False)
    if real._neuron_df_path and Path(real._neuron_df_path).exists():
        assert real.load() is True
    else:
        pytest.skip('real male-cns neuron info not available locally')


# ---------------------------------------------------------------------------
# Helpers: suffix split / normalization / mapping keys
# ---------------------------------------------------------------------------

def test_split_hemi_suffix():
    split = CrossDatasetTypeMapper._split_hemi_suffix
    assert split('aMe12_L') == ('aMe12', '_L')
    assert split('aMe12_R') == ('aMe12', '_R')
    assert split('aMe12_U') == ('aMe12', '_U')
    assert split('aMe12') == ('aMe12', '')
    assert split(None) == (None, '')  # non-str passes through


def test_normalize_dataset_name(mapper):
    norm = mapper._normalize_dataset_name
    assert norm('male_cns') == 'male-cns:v1.0'
    assert norm('male-cns:v0.9') == 'male-cns:v0.9'  # release preserved
    assert norm('banc') == 'banc_v626'
    assert norm('banc_v888') == 'banc_v888'
    assert norm('fafb') == 'flywire_FAFB_v783'
    assert norm('flywire') == 'flywire_FAFB_v783'
    assert norm('hemibrain') == 'hemibrain:v1.2.1'
    assert norm('manc') == 'manc:v1.0'
    assert norm('optic-lobe') == 'optic-lobe:v1.1'
    assert norm('unknown_ds') == 'unknown_ds'
    assert norm(None) is None


def test_get_type_mapping_key(mapper):
    key = mapper._get_type_mapping_key
    # §version control: releases are per-release namespaces — a banc_v888
    # selection must never resolve through v626 names, and male-cns v0.9
    # must not silently borrow the v1.0 crosswalk.
    assert key('male-cns:v0.9') == 'male-cns:v0.9'
    assert key('male-cns:v1.0') == 'male-cns:v1.0'
    assert key('flywire_FAFB_v783') == 'flywire_FAFB_v783'
    # BANC releases each keep their own namespace: they resolve renames
    # through their own "Alternative Cell Type(s)" column and own tables.
    assert key('banc_v626') == 'banc_v626'
    assert key('banc_v888') == 'banc_v888'
    assert key('hemibrain:v1.2.1') == 'hemibrain:v1.2.1'


def test_warn_if_unsupported_dataset(mapper, capsys):
    # not loaded -> silent early return
    unloaded = CrossDatasetTypeMapper(
        neuron_df_path='/nonexistent/x.csv', verbose=True)
    unloaded._warn_if_unsupported_dataset('hemibrain:v9.9')

    verbose_mapper = CrossDatasetTypeMapper(
        neuron_df_path=mapper._neuron_df_path, verbose=True)
    verbose_mapper.load()
    verbose_mapper._warn_if_unsupported_dataset('hemibrain:v9.9')
    out = capsys.readouterr().out
    assert 'No release-specific' in out
    # second call for same dataset -> warned only once
    verbose_mapper._warn_if_unsupported_dataset('hemibrain:v9.9')
    assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# get_mapped_type
# ---------------------------------------------------------------------------

def test_get_mapped_type_basic(mapper):
    assert mapper.get_mapped_type('aMe12', MCNS, FW) == 'MTe07'
    # MCNS flywireType no longer lands directly in BANC; a BANC row must
    # carry the corresponding malecns_cell_type label to bridge this pair.
    assert mapper.get_mapped_type('aMe12', MCNS, BANC) is None
    assert mapper.get_mapped_type('aMe12', MCNS, HB) == 'aMe12'
    assert mapper.get_mapped_type('aMe12', MCNS, MANC) == 'MN1'
    # reverse direction
    assert mapper.get_mapped_type('MTe07', FW, MCNS) == 'aMe12'
    # transitive mapping built from flywire side
    assert mapper.get_mapped_type('MTe07', FW, HB) == 'aMe12'
    assert mapper.get_mapped_type('MTe07', FW, MANC) == 'MN1'
    # unknown type
    assert mapper.get_mapped_type('NoSuchType', MCNS, FW) is None
    assert mapper.get_mapped_type('aMe12', 'unknown_ds', FW) is None


def test_get_mapped_type_same_namespace_and_suffix(mapper):
    # same schema namespace -> native name returned
    assert mapper.get_mapped_type('aMe12', MCNS, 'male-cns:v1.0') == 'aMe12'
    assert mapper.get_mapped_type('MTe07', FW, BANC) is None
    # §version control: male-cns v0.9 is its own native namespace.  With no
    # v0.9 table in this hermetic fixture, it remains unavailable rather than
    # borrowing v1.0 rows.
    assert mapper.get_mapped_type('aMe12', MCNS, 'male-cns:v0.9') is None
    # hemisphere suffix preserved on mapped name
    assert mapper.get_mapped_type('aMe12_L', MCNS, FW) == 'MTe07_L'
    assert mapper.get_mapped_type('aMe12_R', MCNS, FW) == 'MTe07_R'
    # suffix with no mapping -> None
    assert mapper.get_mapped_type('NoSuch_L', MCNS, FW) is None


# ---------------------------------------------------------------------------
# resolve / detect source
# ---------------------------------------------------------------------------

def test_resolve_type_across_datasets(mapper):
    result = mapper.resolve_type_across_datasets(
        'aMe12', [MCNS, FW, HB], source_dataset=MCNS)
    assert result[MCNS] == 'aMe12'
    assert result[FW] == 'MTe07'
    assert result[HB] == 'aMe12'

    # auto-detect source
    result_auto = mapper.resolve_type_across_datasets('aMe12', [FW])
    assert result_auto[FW] == 'MTe07'

    # flywire-side type auto-detected
    result_fw = mapper.resolve_type_across_datasets('MTe07', [MCNS])
    assert result_fw[MCNS] == 'aMe12'

    # unknown type -> all None
    result_none = mapper.resolve_type_across_datasets('Nothing', [FW, HB])
    assert result_none == {FW: None, HB: None}


def test_detect_type_source(mapper):
    assert mapper._detect_type_source('aMe12') == MCNS
    assert mapper._detect_type_source('MTe07') == FW
    assert mapper._detect_type_source('aMe12_L') == MCNS  # suffix stripped
    assert mapper._detect_type_source('Nothing') is None
    unloaded = CrossDatasetTypeMapper(
        neuron_df_path='/nonexistent/x.csv', verbose=False)
    assert unloaded._detect_type_source('aMe12') is None


# ---------------------------------------------------------------------------
# Display names / short codes
# ---------------------------------------------------------------------------

def test_get_display_name(mapper):
    assert mapper.get_display_name('aMe12', [MCNS, FW, HB]) == 'aMe12(MTe07)'
    # hemisphere suffix carried through
    assert mapper.get_display_name(
        'aMe12_L', [MCNS, FW, HB]) == 'aMe12_L(MTe07_L)'
    # all identical -> no parentheses
    assert mapper.get_display_name('Same1', [MCNS, FW, HB]) == 'Same1'
    # unknown type -> original name
    assert mapper.get_display_name('Unknown', [MCNS, FW]) == 'Unknown'


def test_get_display_name_with_dataset_info(mapper):
    display, info = mapper.get_display_name_with_dataset_info(
        'aMe12', [MCNS, FW, HB])
    assert display == 'aMe12(MTe07)'
    assert 'MTe07' in info.values()
    assert 'aMe12' in info.values()


def test_dataset_short_codes(mapper):
    assert mapper.get_dataset_short_code(MCNS) == 'M'
    assert mapper.get_dataset_short_code(FW) == 'F'
    assert mapper.get_dataset_short_code(HB) == 'H'
    assert mapper.get_dataset_short_code(MANC) == 'N'
    # unknown family -> first letter
    assert mapper.get_dataset_short_code('weird_ds') == 'W'

    # collision-aware codes for repeated families
    codes = mapper.get_dataset_short_code(
        MCNS, datasets=[MCNS, 'male-cns:v0.9'])
    codes2 = mapper.get_dataset_short_code(
        'male-cns:v0.9', datasets=[MCNS, 'male-cns:v0.9'])
    assert codes != codes2
    assert codes.startswith('M') and codes2.startswith('M')

    all_codes = mapper.get_all_dataset_short_codes([MCNS, FW])
    assert set(all_codes.keys()) == {'M', 'F'}
    assert all_codes['M'] == 'male-cns v1.0'


def test_dataset_full_names(mapper):
    assert mapper.get_dataset_full_name(MCNS) == 'male-cns v1.0'
    assert mapper.get_dataset_full_name(FW) == 'FlyWire FAFB v783'
    # unsupported release -> family + version
    assert mapper.get_dataset_full_name('male-cns:v0.9') == 'male-cns v0.9'
    assert mapper.get_dataset_full_name('banc') == 'BANC v626'
    assert mapper.get_dataset_full_name('manc:v9.9') == 'MANC v9.9'
    assert mapper.get_dataset_full_name('weird_ds') == 'weird_ds'


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------

def test_conflict_detection(mapper):
    n_to_1 = mapper.get_n_to_1_conflicts()
    one_to_n = mapper.get_1_to_n_conflicts()

    # AggFW (flywire) -> {AggA, AggB} in male-cns
    assert any(c.source_type == 'AggFW' and c.relationship == 'N-to-1'
               for c in n_to_1)
    assert any(c.source_type == 'HBa' for c in n_to_1)
    # Fixed: MANC reverse N-to-1 detection now exists alongside
    # flywire/hemibrain (MNx -> {Mn1, Mn2}).
    assert any(c.source_type == 'MNx' and c.relationship == 'N-to-1'
               for c in n_to_1)
    # SplitN (male-cns) -> {FWa, FWb}
    assert any(c.source_type == 'SplitN' and c.relationship == '1-to-N'
               and c.target_types == {'FWa', 'FWb'} for c in one_to_n)

    assert mapper.is_n_to_1_type('AggFW', FW) is True
    assert mapper.is_n_to_1_type('AggA', MCNS) is True
    assert mapper.is_n_to_1_type('MNx', MANC) is True
    assert mapper.is_n_to_1_type('Same1', MCNS) is False
    # n-to-1 types are NOT mapped (aggregation avoided)
    assert mapper.get_mapped_type('AggFW', FW, MCNS) is None
    assert mapper.get_mapped_type('SplitN', MCNS, FW) is None


def test_conflict_repr():
    c = TypeMappingConflict('ds1', 'ds2', 'T', {'A', 'B'}, 'N-to-1')
    assert 'N-to-1' in repr(c) and 'T' in repr(c)


def test_mapping_decision_labels_target_fan_in(mapper):
    """A mapped pair whose target also receives other source types reads
    N-to-1, not a hardcoded 1-to-1 (user 2026-09-10: 5th-LNv and
    LNd_CRY+_ITP+ both resolve to MCNS 5thsLNv_LNd6)."""
    agg = mapper.get_mapping_decision('AggA', MCNS, FW)
    assert agg['status'] == 'mapped'
    assert agg['target_type'] == 'AggFW'
    assert agg['relationship'] == 'N-to-1'
    # its sibling converging on the same target gets the same label
    sibling = mapper.get_mapping_decision('AggB', MCNS, FW)
    assert sibling['status'] == 'mapped'
    assert sibling['relationship'] == 'N-to-1'
    # unique pairs stay 1-to-1
    lone = mapper.get_mapping_decision('aMe12', MCNS, FW)
    assert lone['status'] == 'mapped'
    assert lone['relationship'] == '1-to-1'
    same = mapper.get_mapping_decision('Same1', MCNS, FW)
    assert same['status'] == 'mapped'
    assert same['relationship'] == '1-to-1'
    # the reverse query still goes through the conflict record untouched
    reverse = mapper.get_mapping_decision('AggFW', FW, MCNS)
    assert reverse['status'] == 'evidence_only'
    assert reverse['relationship'] == 'N-to-1'


def test_warn_if_conflicting(mapper):
    with pytest.warns(TypeMappingWarning):
        assert mapper.warn_if_conflicting('AggFW', [FW]) is True
    assert mapper.warn_if_conflicting('Same1', [MCNS, FW]) is False


def test_check_type_name_conflict(mapper):
    # ClashFw exists in both namespaces but maps to OtherFw in flywire
    conflict = mapper.check_type_name_conflict('ClashFw', [MCNS, FW])
    assert conflict == ('ClashFw', 'OtherFw', FW)

    # consistent mapping -> no conflict
    assert mapper.check_type_name_conflict('aMe12', [MCNS, HB]) is None
    # type present in a single namespace -> no conflict
    assert mapper.check_type_name_conflict('AggFW', [MCNS, FW]) is None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def test_export_mapping(tmp_path, mapper):
    out = tmp_path / 'mapping.csv'
    mapper.export_mapping(str(out))
    df = pd.read_csv(out)
    assert 'aMe12' in df[MCNS].values
    assert 'Same1' not in df[MCNS].values  # identical rows filtered

    out2 = tmp_path / 'filtered.csv'
    mapper.export_mapping(str(out2), filter_types={'aMe12_L'})
    df2 = pd.read_csv(out2)
    assert list(df2[MCNS]) == ['aMe12']

    out3 = tmp_path / 'subset.csv'
    mapper.export_mapping(str(out3), datasets=[MCNS, FW])
    df3 = pd.read_csv(out3)
    # Additive provenance column: dataset columns keep their positions,
    # mapping_origin records how each row was derived (crosswalk vs the
    # same-name / annotation-bridge overlay).
    assert list(df3.columns) == [MCNS, FW, 'mapping_origin',
                                'mapping_support']

    out4 = tmp_path / 'all.csv'
    mapper.export_mapping(str(out4), only_different=False)
    df4 = pd.read_csv(out4)
    assert 'Same1' in df4[MCNS].values

    # fewer than two datasets -> early return, no file
    out5 = tmp_path / 'never.csv'
    mapper.export_mapping(str(out5), datasets=[MCNS])
    assert not out5.exists()


def test_export_mapping_unloaded(tmp_path):
    m = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    with pytest.raises(RuntimeError):
        m.export_mapping(str(tmp_path / 'x.csv'))


def test_export_conflicts(tmp_path, mapper):
    out = tmp_path / 'conflicts.csv'
    mapper.export_conflicts(str(out))
    df = pd.read_csv(out)
    assert 'AggFW' in df['source_type'].values

    out2 = tmp_path / 'filtered_conflicts.csv'
    mapper.export_conflicts(str(out2), filter_types={'AggFW'})
    df2 = pd.read_csv(out2)
    assert set(df2['source_type']) == {'AggFW'}

    out3 = tmp_path / 'none.csv'
    mapper.export_conflicts(str(out3), filter_types={'NoSuchType'})
    assert not out3.exists()

    # mapper without conflicts
    clean_csv = tmp_path / 'clean.csv'
    clean_csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,aMe12,MTe07,,\n", encoding='utf-8')
    clean = CrossDatasetTypeMapper(
        neuron_df_path=str(clean_csv), verbose=False)
    clean.load()
    out4 = tmp_path / 'clean_conflicts.csv'
    clean.export_conflicts(str(out4))
    assert not out4.exists()


# ---------------------------------------------------------------------------
# LabelMapper conversion / canonical types
# ---------------------------------------------------------------------------

def test_to_label_mapper(mapper):
    lm = mapper.to_label_mapper(['aMe12'], [MCNS, FW], role='source')
    assert isinstance(lm, LabelMapper)
    assert lm.get_mapped_label('aMe12', FW) == 'MTe07'

    lm_target = mapper.to_label_mapper(['aMe12'], [MCNS, FW], role='target')
    assert isinstance(lm_target, LabelMapper)

    lm_inter = mapper.to_label_mapper(
        ['aMe12'], [MCNS, FW], role='intermediate')
    assert isinstance(lm_inter, LabelMapper)


def test_get_canonical_type(mapper):
    assert mapper.get_canonical_type('MTe07', FW) == 'aMe12'
    assert mapper.get_canonical_type('aMe12', MCNS) == 'aMe12'  # already mcns
    assert mapper.get_canonical_type('aMe12') == 'aMe12'  # auto-detected
    # empty / pattern / non-str pass through
    assert mapper.get_canonical_type('') == ''
    assert mapper.get_canonical_type('a.*b') == 'a.*b'
    assert mapper.get_canonical_type(None) is None
    # unknown type stays
    assert mapper.get_canonical_type('Nothing', FW) == 'Nothing'


def test_standardize_partner_types(mapper):
    out = mapper.standardize_partner_types(
        {'MTe07': 2.0, 'Same1': 1.0, '2hop:MTe07': 0.5, '': 1.0}, FW)
    assert out['aMe12'] == pytest.approx(2.0)
    assert out['Same1'] == pytest.approx(1.0)
    assert out['2hop:aMe12'] == pytest.approx(0.5)
    assert out[''] == pytest.approx(1.0)

    # weights merging into the same canonical type are summed
    summed = mapper.standardize_partner_types({'FWa': 1.0, 'FWb': 2.0}, FW)
    assert summed == {'SplitN': pytest.approx(3.0)}

    # male-cns namespace -> unchanged copy
    src = {'aMe12': 1.0}
    assert mapper.standardize_partner_types(src, MCNS) == src


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_get_source_target_mapping_summary(mapper):
    summary = mapper.get_source_target_mapping_summary(
        ['aMe12', 123, 'Agg*', 'Unknown', 'SplitN', 'AggFW'], [MCNS, FW])
    per = summary['per_dataset']
    assert per[FW]['aMe12'] == 'MTe07'
    assert per[MCNS]['aMe12'] == 'aMe12'
    assert per[FW][123] == 123              # non-str passthrough
    assert per[FW]['Agg*'] == 'Agg*'        # regex passthrough
    assert per[FW]['Unknown'] == 'Unknown'  # not found
    assert per[FW]['SplitN'] == 'SplitN'    # 1-to-N: no mapping -> as-is

    assert any(t == 'aMe12' for t, _ in summary['different_mappings'])
    assert any(t == 'AggFW' for t, *_ in summary['n_to_1_warnings'])
    assert any(t == 'SplitN' for t, *_ in summary['one_to_n_warnings'])


def test_get_intermediate_mapping_summary(mapper):
    summary = mapper.get_intermediate_mapping_summary(
        {'aMe12', 'AggFW', 'SplitN', 'nope', 123}, [MCNS, FW])
    assert summary['total_types'] == 5
    assert summary['mapped_count'] == 1      # only aMe12 -> MTe07
    assert summary['n_to_1_count'] == 1      # AggFW
    assert summary['one_to_n_count'] == 1    # SplitN


# ---------------------------------------------------------------------------
# Merge mappings (NeuronBridge-style prefixed types)
# ---------------------------------------------------------------------------

def test_get_merge_mapping_for_types(mapper):
    merge = mapper.get_merge_mapping_for_types(
        ['MCNS_aMe12', 'FAFB_MTe07'])
    assert merge['MCNS_aMe12'] == 'aMe12(MTe07)'
    assert merge['FAFB_MTe07'] == 'aMe12(MTe07)'

    # queried_name overrides the main display name
    merge_q = mapper.get_merge_mapping_for_types(
        ['MCNS_aMe12', 'FAFB_MTe07'], queried_name='Query')
    assert merge_q['MCNS_aMe12'] == 'Query(MTe07)'

    # no underscore -> passthrough; unknown prefix -> base type
    merge_misc = mapper.get_merge_mapping_for_types(
        ['Plain', 'XX_SomeType'])
    assert merge_misc['Plain'] == 'Plain'
    assert merge_misc['XX_SomeType'] == 'SomeType'

    # other prefixes resolve through their datasets
    merge_more = mapper.get_merge_mapping_for_types(
        ['BANC_Same1', 'HB_aMe12', 'MANC_MN1'], verbose=True)
    assert merge_more['BANC_Same1'] == 'Same1'
    # Fixed: MANC reverse mapping is built, so aMe12 now carries its MANC
    # alias and MN1 resolves to the male-cns canonical type.
    assert merge_more['HB_aMe12'] == 'aMe12(MN1)'
    assert merge_more['MANC_MN1'] == 'aMe12(MN1)'


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_get_type_mapper_singleton(tmp_path, monkeypatch):
    monkeypatch.setattr(mapper_module, '_global_type_mapper', None)
    ws = tmp_path
    ds_dir = ws / 'datasets' / 'male-cns_v1_0'
    ds_dir.mkdir(parents=True)
    (ds_dir / 'male-cns_v1_0_allneurons_neuron_df.csv').write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,aMe12,MTe07,,\n", encoding='utf-8')

    m1 = get_type_mapper(workspace_path=str(ws))
    assert m1._loaded is True
    assert 'male-cns:v1.0' in m1._type_mappings

    m2 = get_type_mapper()
    assert m2 is m1  # cached

    m3 = get_type_mapper(workspace_path=str(ws), force_reload=True)
    assert m3 is not m1

    monkeypatch.setattr(mapper_module, '_global_type_mapper', None)


def test_workspace_autodetect_path():
    # no workspace_path -> derived from module location (repo root)
    m = CrossDatasetTypeMapper(verbose=False)
    assert m._neuron_df_path.endswith(
        str(Path('datasets') / 'male-cns_v1_0' /
            'male-cns_v1_0_allneurons_neuron_df.csv'))


# ---------------------------------------------------------------------------
# Same-name-first within a fan-out (plan-samename-first-fanout-resolution)
# ---------------------------------------------------------------------------

# Row semantics for the same-name fixtures:
#  - aMe12 rows: a mcns cell that ALSO lists the same name in its crosswalk
#    but the fan-out carries OTHER candidate names too (same-name-in-fan-out).
#  - SMP520a/SMP520b: an a/b variant split — the source SMP520 is NOT among
#    the candidates (the exact-predicate do-NOT-fire guard).
SAME_NAME_CSV_ROWS = (
    "bodyId,type,flywireType,hemibrainType,mancType\n"
    # same-name fan-out into FAFB: candidates {aMe12, MTe07} include 'aMe12'
    "1,aMe12,\"aMe12,MTe07\",,\n"
    "2,aMe12,\"aMe12,MTe07\",,\n"
    # a/b variant split: candidates {Varianta, Variantb} — no 'Variant'
    "3,Variant,Varianta,,\n"
    "4,Variant,Variantb,,\n"
    # a decoy: the rival 'MTe07' has its OWN clean pair elsewhere
    "5,MTe07,MTe07,,\n"
)


@pytest.fixture
def same_name_mapper(tmp_path):
    csv = tmp_path / 'sn_neurons.csv'
    csv.write_text(SAME_NAME_CSV_ROWS, encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    return m


def test_same_name_first_selects_the_same_name_candidate(same_name_mapper):
    m = same_name_mapper
    snf = m.same_name_first_fires('aMe12', MCNS, FW)
    assert snf is not None, 'the same-name fan-out must be detected'
    assert snf['selected'] == 'aMe12'
    assert 'MTe07' in snf['rivals']
    assert snf['fires'] is True
    d = m.get_mapping_decision('aMe12', MCNS, FW)
    assert d['status'] == 'mapped'
    assert d['target_type'] == 'aMe12'
    assert d['relationship'] == 'suspects'
    assert d['suspects'] is True
    # DEFAULT: suspects stay OUT of target_types
    assert d['target_types'] == ['aMe12']
    assert 'MTe07' in d['fan_out_candidates']


def test_same_name_first_optin_includes_rivals(same_name_mapper):
    m = same_name_mapper
    m.include_suspects_in_targets = True
    try:
        d = m.get_mapping_decision('aMe12', MCNS, FW)
        assert d['status'] == 'mapped'
        assert d['target_types'] == ['aMe12', 'MTe07']
    finally:
        m.include_suspects_in_targets = False


def test_exact_predicate_never_fires_on_ab_variant_split(same_name_mapper):
    """§17.1: SMP520 -> {SMP520a, SMP520b} style must NOT fire — the exact
    base-name membership test, never a prefix/substring test."""
    m = same_name_mapper
    assert m.same_name_first_fires('Variant', MCNS, FW) is None
    d = m.get_mapping_decision('Variant', MCNS, FW)
    assert d['status'] != 'mapped'
    assert d.get('target_type') in (None,)
    assert d['relationship'] != 'suspects'


def test_rival_has_own_clean_pair_discriminator(same_name_mapper):
    m = same_name_mapper
    # MTe07 has its own clean same-name pair in FAFB
    assert m._rival_has_own_clean_pair('MTe07', MCNS, FW) is True


def test_same_name_conflict_detail_shape(same_name_mapper):
    m = same_name_mapper
    det = m.get_same_name_conflict_detail('aMe12', MCNS, FW)
    assert det is not None
    assert det['selected'] == 'aMe12'
    assert det['rivals'] == ['MTe07']
    row = det['rival_evidence'][0]
    assert row['rival'] == 'MTe07'
    assert row['rival_has_own_clean_pair'] is True
    # boundary-clean observation label (plan-ui-type-mapper-alignment §2.5):
    # no verdict words ("duplicate"/"convergence"), no claim that a gate ran.
    assert row['rival_pair_status'] == 'own_1to1_pair'
    assert 'suspected_duplicate' not in row


def test_suspects_export_roundtrip(same_name_mapper, tmp_path):
    # per-direction: both the MCNS->FAFB and FAFB->MCNS fan-outs fire
    m = same_name_mapper
    out = tmp_path / 'suspects.csv'
    n = m.export_suspects(str(out))
    assert n == 2
    df = pd.read_csv(out)
    fwd = df[(df['source_type'] == 'aMe12')
             & (df['target_dataset'] == FW)].iloc[0]
    assert fwd['selected'] == 'aMe12' and fwd['rival'] == 'MTe07'
    assert bool(fwd['rival_has_own_clean_pair']) is True
    assert fwd['rival_pair_status'] == 'own_1to1_pair'
    assert 'disposition' not in df.columns or 'selection_disposition' in df.columns
    assert fwd['custom_mapper_from'] == 'aMe12'
    assert fwd['custom_mapper_to'] == 'MTe07'


def test_conflicts_export_carries_same_name_and_multivalue_columns(
        same_name_mapper, tmp_path):
    m = same_name_mapper
    out = tmp_path / 'conflicts.csv'
    m.export_conflicts(str(out))
    df = pd.read_csv(out)
    for col in ('same_name_candidate', 'same_name_path',
                'same_name_disposition', 'multivalue_source', 'source_parts'):
        assert col in df.columns, col


def test_multivalue_accessors(tmp_path):
    """plan-type-column-multivalue-normalization Stage 1: a comma-joined
    release `type` cell is recorded and queryable, and STAYS atomic."""
    csv = tmp_path / 'mv.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,\"DNp51,DNpe019\",\"DNp51,DNpe019\",PS296,DNp51\n"
        "2,Plain1,,,\n", encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    assert m.is_multivalue_type('DNp51,DNpe019', MCNS) is True
    assert m.multivalue_parts('DNp51,DNpe019', MCNS) == ('DNp51', 'DNpe019')
    assert m.is_multivalue_type('Plain1', MCNS) is False
    # the joined cell stays a native atomic name (status quo pinned)
    assert m.has_native_type('DNp51,DNpe019', MCNS) is True
    assert m.has_native_type('DNp51', MCNS) is False


def test_same_name_first_summary_is_run_scoped(same_name_mapper):
    """The disclosure note counts must match the export scoping."""
    m = same_name_mapper
    counts = m.same_name_first_summary(
        datasets=[MCNS, FW], filter_types={'aMe12', 'MTe07'})
    assert counts['selected'] >= 1
    assert counts['rivals_exported'] >= 1
    # a type set that touches nothing ⇒ zero counts (not global numbers)
    empty = m.same_name_first_summary(
        datasets=[MCNS, FW], filter_types={'NoSuchType'})
    assert empty['selected'] == 0 and empty['rivals_exported'] == 0


def test_multivalue_summary_is_run_scoped(tmp_path):
    csv = tmp_path / 'mv2.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,\"A1,A2\",,,\n"
        "2,\"B1,B2\",,,\n", encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    assert m.multivalue_summary(datasets=[MCNS]) == {MCNS: 2}
    assert m.multivalue_summary(datasets=[FW]) == {}


def test_multivalue_parser_handles_parenthesized_cells():
    """A second release encoding: `(PLP191,PLP192)a` — the alternatives are
    INSIDE the parens, the variant suffix is not a candidate name, and the
    raw cell stays atomic.  Found 2026-09-18 while reviewing the multivalue
    marker (the naive comma split produced ('(PLP191', 'PLP192)a'))."""
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    parse = CrossDatasetTypeMapper._multivalue_cell_parts
    assert parse('(PLP191,PLP192)a') == ('PLP191', 'PLP192')
    assert parse('(PLP191,PLP192)b') == ('PLP191', 'PLP192')
    assert parse('(PS023,PS024)a') == ('PS023', 'PS024')
    # plain comma-joined form is unchanged
    assert parse('LAL173,LAL174') == ('LAL173', 'LAL174')
    # single names / empty / bodyId-paren forms are NOT multi-value
    assert parse('KCg-d') == ()
    assert parse('') == ()
    assert parse('(hb5813083315)') == ()


def test_multivalue_marker_covers_both_encodings(tmp_path):
    """Both encodings registered from a release table, and a bodyId-paren
    cell (never native) stays unflagged."""
    csv = tmp_path / 'mv3.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,\"(PLP191,PLP192)a\",,,\n"
        "2,\"LAL173,LAL174\",,,\n"
        "3,Plain2,,,\n", encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    assert m.multivalue_parts('(PLP191,PLP192)a', MCNS) == ('PLP191', 'PLP192')
    assert m.multivalue_parts('LAL173,LAL174', MCNS) == ('LAL173', 'LAL174')
    assert m.is_multivalue_type('Plain2', MCNS) is False
    assert m.multivalue_summary(datasets=[MCNS]) == {MCNS: 2}


# ---------------------------------------------------------------------------
# Parenthesized alternative groups in crosswalk cells (2026-09-18)
# ---------------------------------------------------------------------------

def test_split_type_cell_parenthesized_alternatives():
    """`(A,B)suffix` distributes the suffix to EACH alternative; the
    `auto:` provenance prefix is preserved per token.  Verified semantics:
    the FAFB counterpart of mcns AVLP346 is AVLP346a/b, and FAFB carries
    AVLP346a/AVLP346b but no plain AVLP346."""
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    split = CrossDatasetTypeMapper._split_type_cell
    assert split('(AVLP346,AVLP348)a') == ['AVLP346a', 'AVLP348a']
    assert split('(AVLP346,AVLP348)b') == ['AVLP346b', 'AVLP348b']
    assert split('(PLP191,PLP192)a') == ['PLP191a', 'PLP192a']
    assert split('auto:(PLP191,PLP192)a') == ['auto:PLP191a', 'auto:PLP192a']
    # plain lists and single names unchanged
    assert split('A, B,C') == ['A', 'B', 'C']
    assert split('KCg-d') == ['KCg-d']
    # parens WITHOUT an inner comma are part of the name
    assert split('PEN_a(PEN1)') == ['PEN_a(PEN1)']
    assert split('auto:PLP191') == ['auto:PLP191']


def test_crosswalk_parenthesized_cells_leave_no_garbage_keys(tmp_path):
    """The Polars fast paths must apply the same expansion: no fragment keys
    like '(AVLP346' / 'AVLP348)a' in the mapping tables, and the composite's
    alternatives resolve to real names."""
    csv = tmp_path / 'paren.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,AVLP346,\"(AVLP346,AVLP348)a\",,\n"
        "2,AVLP346,\"(AVLP346,AVLP348)b\",,\n"
        "3,PLP191,\"(PLP191,PLP192)a\",,\n", encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    fw = m._type_mappings.get(FW, {})
    garbage = [k for k in fw if k.startswith('(') or k.endswith((')a', ')b'))]
    assert garbage == [], garbage
    # the emitted tokens are the suffixed alternatives (unresolvable ones are
    # dropped by target-side resolution, exactly like a plain comma list)
    assert m._split_type_cell('(AVLP346,AVLP348)a') == [
        'AVLP346a', 'AVLP348a']


# ---------------------------------------------------------------------------
# BodyId-valued crosswalk cells (2026-09-18)
# ---------------------------------------------------------------------------

def test_bodyid_tokens_are_dropped_and_hybrids_keep_their_base_name():
    """A bodyId is not a type name: cells like ``hb1874217622`` /
    ``(hb5813083315)`` / ``(5901212906)`` are dropped instead of becoming
    pseudo-types; a NAME carrying a bodyId annotation
    (``PS279(hb1499087543)``) keeps its base name, because that base name
    exists natively in the target dataset."""
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    split = CrossDatasetTypeMapper._split_type_cell
    # pure bodyId forms (with/without hb, with/without parens, auto: prefix)
    for cell in ('hb1874217622', '(hb5813083315)', '(5901212906)',
                 '1343403608', '(hb487286529,hb517242832)', 'hb123456789.0',
                 'auto:(hb5813083315)', 'auto:hb5813083315'):
        assert split(cell) == [], cell
    # hybrid name(bodyId) keeps the base name (real cells from mcns)
    assert split('PS279(hb1499087543)') == ['PS279']
    assert split('SLP405_a(hb5813008928)') == ['SLP405_a']
    assert split('PVLP120(hb1599285725)') == ['PVLP120']
    # paren content that is NOT a bodyId stays part of the name
    assert split('PEN_a(PEN1)') == ['PEN_a(PEN1)']
    assert split('PEN_b(PEN2)') == ['PEN_b(PEN2)']


def test_bodyid_cells_do_not_become_mapping_keys(tmp_path):
    """End to end: a crosswalk cell holding a bodyId produces NO mapping key
    and NO decision target; the type simply stays unresolved (it can still
    resolve through another lane)."""
    csv = tmp_path / 'bodyid.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n"
        "1,RealA,RealA,hb1874217622,\n"       # bodyId in hemi cell
        "2,RealB,RealB,(hb2316502686),\n"     # parenthesized bodyId
        "3,PS279,PS279,PS279(hb1499087543),\n"  # hybrid -> base name
        "4,Li33,Li33,1343403608,\n",          # bare-digit flywire-ish cell
        encoding='utf-8')
    m = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    assert m.load() is True
    hemi = m._type_mappings.get(HB, {})
    assert [k for k in hemi if re.search(r'\d{5,}', str(k))] == []
    # the pure-bodyId rows resolve to nothing rather than to a pseudo-type
    assert m.get_mapping_decision('RealA', MCNS, HB)['target_type'] in (None, '')
    # the hybrid row resolves to its BASE NAME, not the annotated cell
    dec = m.get_mapping_decision('PS279', MCNS, HB)
    assert dec['target_type'] == 'PS279'


# ---------------------------------------------------------------------------
# build_type_coverage / dedupe_mirrored_pairs — same-name-first SUSPECTS
# (fan-out/suspects display round, plan Panel_fan-out_suspects_display).
# Both functions are pure over plain dicts, so these are hermetic.
# ---------------------------------------------------------------------------


def _sn_flow(src, tgt, s_type, f_type, suspects=False, count=2):
    return {
        'source_dataset': src, 'target_dataset': tgt,
        'source_type': s_type, 'foreign_type': f_type,
        'source_count': count, 'foreign_count': count + 1,
        'suspects': suspects,
    }


def test_build_type_coverage_propagates_suspects_both_directions():
    """A suspects-flagged flow marks its forward target row and the mirrored
    backward source row; unflagged rows keep empty/False defaults."""
    from comparison.mapping_visualization import build_type_coverage

    flows = [
        _sn_flow(MCNS, FW, 'aMe12', 'aMe12', suspects=True),
        _sn_flow(MCNS, FW, 'Plain', 'PlainT'),
    ]
    cov = build_type_coverage({(MCNS, FW): flows})
    fwd = {r['type']: r for r in cov['forward']}
    rev = {r['type']: r for r in cov['reverse']}
    # forward row: the queried type that fanned out via a suspects flow
    assert fwd['aMe12']['has_suspects'] is True
    assert fwd['aMe12']['suspect_targets'] == 'aMe12'
    assert fwd['aMe12']['suspect_count'] == 1
    # backward row: the receiving type arrived at via a suspects flow
    assert rev['aMe12']['has_suspects'] is True
    assert rev['aMe12']['suspect_sources'] == 'aMe12'
    assert rev['aMe12']['suspect_count'] == 1
    # an unflagged pair defaults to empty / False on BOTH sides
    for row in (fwd['Plain'], rev['PlainT']):
        assert row['has_suspects'] is False
        assert row['suspect_count'] == 0
        assert row.get('suspect_targets', row.get('suspect_sources')) == ''


def test_dedupe_mirrored_pairs_keeps_the_suspects_direction():
    """When the two mirrored directions otherwise tie, the suspects-carrying
    direction survives so the panel's marker is not silently dropped.  The
    suspects flow sits on the lexicographically SMALLER src_ds, so WITHOUT
    the new term the non-suspects (larger src_ds) flow would win."""
    from comparison.mapping_visualization import dedupe_mirrored_pairs

    # 'flywire_FAFB_v783' (FW) sorts before 'male-cns:v1.0' (MCNS), so on
    # the src_ds tie-break alone the MCNS/plain flow would win; only the
    # earlier suspects term can keep the FW/suspects direction.
    plain_flow = _sn_flow(MCNS, FW, 'aMe12', 'aMe12', suspects=False)
    sus_flow = _sn_flow(FW, MCNS, 'aMe12', 'aMe12', suspects=True)
    result = dedupe_mirrored_pairs(
        {(MCNS, FW): [plain_flow], (FW, MCNS): [sus_flow]}, ())
    survivors = [f for fl in result.values() for f in fl]
    # collapse to ONE flow (no added/removed pair, no moved count)
    assert len(survivors) == 1
    survivor, = survivors
    assert survivor['suspects'] is True
    assert survivor['source_dataset'] == FW
