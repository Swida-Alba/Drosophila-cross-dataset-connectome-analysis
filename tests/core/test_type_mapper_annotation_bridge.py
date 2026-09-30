"""Annotation-bridge tests for the cross-dataset type mapper.

Covers the designed FAFB <-> BANC bridge (plan
``_plan/plan-type-mapper-additional-type-bridge.md``):

* the cross-namespace annotation-value landing hop in the derivation
  walk (FAFB type -> FAFB additional_type(s) -> BANC Alternative Cell
  Type(s) -> BANC type), including the namespace-aware annotation-
  chaining guard and the landing/ping-pong linker rules;
* the production overlay: same-name identity + annotation-bridge
  candidates (1-to-1 -> mapping, 1-to-N -> conflict with provenance),
  crosswalk precedence, and untyped-sentinel exclusion;
* the mapping exports: ``mapping_origin`` column, bridge rows without a
  male-cns anchor, and release-key resolution in the conflicts export.

Walk-level tests run on a hermetic bare mapper (internal tables injected
directly); real-data acceptance tests skip when the local datasets are
absent.
"""

from pathlib import Path

import pandas as pd
import pytest

from comparison.cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    TypeMappingConflict,
    bridge_is_valid,
    standardize_bridge,
)

FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v626'
MCNS = 'male-cns:v1.0'
FAFB_RELEASE = 'flywire_FAFB_v783'
# §version control: releases are per-release namespaces — the banc_v626
# namespace is selected by its own release name (or the legacy flywire_BANC_*
# alias); banc_v888 resolves against ITS OWN tables and never sees these
# entries.
BANC_RELEASE = 'banc_v626'

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_TABLE = (REPO_ROOT / 'datasets' / 'male-cns_v1_0'
              / 'male-cns_v1_0_allneurons_neuron_df.csv')

CHAIN_KEYS = ('dataset', 'column', 'value')


def chain_key(chain):
    return tuple(
        tuple(hop[k] for k in CHAIN_KEYS) for hop in chain)


# ---------------------------------------------------------------------------
# Hermetic bare mapper: internal tables injected, no files touched
# ---------------------------------------------------------------------------

def _bare_mapper():
    m = CrossDatasetTypeMapper.__new__(CrossDatasetTypeMapper)
    m.verbose = False
    m._loaded = True
    m._dataset_types = {}
    m._flywire_primaries = {}
    m._flywire_alt_to_primary = {}
    m._flywire_annotation_primaries = {}
    m._flywire_primary_to_alts = {}
    m._type_mappings = {}
    m._reverse_mappings = {}
    m._conflicts = []
    m._n_to_1_types = {}
    m._bridge_provenance = {}
    m._unsupported_dataset_warnings = set()
    m._alias_n_to_1_cache = None
    m._crosswalk_parts_cache = None
    return m


def _seed_pair(m, *, fafb_primaries, banc_primaries,
               fafb_ann=None, banc_ann=None):
    """Seed a synthetic FAFB/BANC namespace pair.

    ``*_ann`` maps annotation value -> {primaries listing it}; the
    filtered alt table and the primary->alts inverse are derived from it
    exactly like ``_load_flywire_type_tables`` does.
    """
    m._flywire_primaries[FAFB] = set(fafb_primaries)
    m._flywire_primaries[BANC] = set(banc_primaries)
    for key, ann in ((FAFB, fafb_ann or {}), (BANC, banc_ann or {})):
        annotation = {v: set(ps) for v, ps in ann.items()}
        m._flywire_annotation_primaries[key] = annotation
        filtered = {}
        for value, primaries in annotation.items():
            if value in m._flywire_primaries[key]:
                continue  # a primary keeps its own identity
            for primary in primaries:
                filtered.setdefault(value, set()).add(primary)
        m._flywire_alt_to_primary[key] = filtered
        inverse = {}
        for value, primaries in annotation.items():
            for primary in primaries:
                inverse.setdefault(primary, set()).add(value)
        m._flywire_primary_to_alts[key] = inverse
    m._type_mappings = {
        'male-cns:v1.0': {},
        FAFB: {},
        BANC: {},
        'hemibrain:v1.2.1': {},
        'manc:v1.0': {},
        'manc:v1.2.1': {},
    }


# ---------------------------------------------------------------------------
# Walk layer: the composed cross-namespace annotation bridge
# ---------------------------------------------------------------------------

def test_composed_annotation_bridge_chain_is_derivable():
    """The designed 4-node chain: FAFB type -> FAFB additional_type(s)
    token -> BANC Alternative Cell Type(s) token -> BANC type.  The
    token is NOT a BANC primary, so only the cross-namespace landing hop
    completes the route."""
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1', 'P2'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    chains = m.get_type_bridges('T1', FAFB, BANC)
    expected = (
        (FAFB, 'type', 'T1'),
        (FAFB, 'additional_type(s)', 'V1'),
        (BANC, 'Alternative Cell Type(s)', 'V1'),
        (BANC, 'Alternative Cell Type(s)', 'P1'),
    )
    assert expected in [chain_key(c) for c in chains]
    composed = [c for c in chains if chain_key(c) == expected][0]
    # The landing + arrival legs share the token: the standardized
    # linkers must dedupe to one linker per annotation column, and the
    # raw consecutive hops differ in value (token -> primary).
    linkers = [l for l in standardize_bridge(composed, FAFB, BANC)
               if l['kind'] == 'linker']
    assert [(l['column'], l['value']) for l in linkers] == [
        ('additional_type(s)', 'V1'), ('Alternative Cell Type(s)', 'V1')]
    assert bridge_is_valid(composed, FAFB, BANC)


def test_landing_hop_skipped_when_token_is_a_primary():
    """When the token IS a primary in the other namespace, only the
    same-name identity hop connects it — no duplicate annotation-landing
    edge with a fabricated column label."""
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'V1'},
        fafb_ann={'V1': {'T1'}},
    )
    neighbors = m._name_neighbors(FAFB, 'V1')
    assert (BANC, 'V1', 'type', 'V1') in neighbors
    assert not any(
        nns == BANC and column == 'Alternative Cell Type(s)'
        for nns, _name, column, _via in neighbors)


def test_raw_pingpong_chain_still_rejected():
    """The genuine self-loop class (the same annotation hop twice on one
    dataset) stays invalid after the linker dedupe."""
    chain = [
        {'dataset': FAFB, 'column': 'type', 'value': 'T1'},
        {'dataset': FAFB, 'column': 'additional_type(s)', 'value': 'V1'},
        {'dataset': FAFB, 'column': 'additional_type(s)', 'value': 'V1',
         'via': 'V1'},
    ]
    assert not bridge_is_valid(chain, FAFB, BANC)


def test_reverse_bridge_banc_to_fafb():
    """Symmetry: a BANC primary's Alternative Cell Type(s) value that is
    a FAFB annotation token routes to the FAFB primaries annotated with
    it."""
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    chains = m.get_type_bridges('P1', BANC, FAFB)
    expected = (
        (BANC, 'type', 'P1'),
        (BANC, 'Alternative Cell Type(s)', 'V1'),
        (FAFB, 'additional_type(s)', 'V1'),
        (FAFB, 'additional_type(s)', 'T1'),
    )
    assert expected in [chain_key(c) for c in chains]


# ---------------------------------------------------------------------------
# ACT-glue suppression with backward-reciprocity rescue (user 2026-09-30,
# plan _plan/plan-type-mapper-act-lane-precedence-and-backward-reciprocity.md)
# ---------------------------------------------------------------------------

def _seed_glue_pair(m, *, reciprocate=False):
    """The synthetic s-CPDN3 pattern: FAFB T1 annotates token V1, V1 is a
    BANC primary, and BANC V1 + P2 share the ACT token W — so the walk can
    arrive at BANC V1 by same-name identity and depart through ACT onto the
    unrelated P2.  ``reciprocate`` adds P2's ACT naming T1 (a FAFB primary),
    giving the reverse direction a real bridge back to the source."""
    banc_ann = {'W': {'V1', 'P2'}}
    if reciprocate:
        banc_ann['T1'] = {'P2'}
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'V1', 'P2'},
        fafb_ann={'V1': {'T1'}},
        banc_ann=banc_ann,
    )


def test_act_glue_chain_suppressed_without_reciprocity():
    """A chain that same-name-arrives at a BANC primary and departs through
    ``Alternative Cell Type(s)`` transitivity is glue, not pair evidence —
    offered only when the reverse direction corroborates the pair."""
    m = _bare_mapper()
    _seed_glue_pair(m)
    ends = {c[-1]['value'] for c in m.get_type_bridges('T1', FAFB, BANC)}
    assert 'V1' in ends          # clean aT -> identity landing survives
    assert 'P2' not in ends      # the glue target is suppressed
    # the escape hatch used by the reciprocity probe still sees it
    raw = m.get_type_bridges(
        'T1', FAFB, BANC, _skip_glue_filter=True)
    assert 'P2' in {c[-1]['value'] for c in raw}


def test_act_glue_chain_rescued_when_reciprocal():
    """Reciprocal glue pairs survive and carry the ``reciprocal`` flag on
    the arrival hop (the DNp17 <-> DNpe054 class).  The pair may also
    carry an ordinary mirror chain — only the glue chain is flagged."""
    m = _bare_mapper()
    _seed_glue_pair(m, reciprocate=True)
    back = m.get_type_bridges('P2', BANC, FAFB)
    assert 'T1' in {c[-1]['value'] for c in back}
    chains = m.get_type_bridges('T1', FAFB, BANC)
    assert 'P2' in {c[-1]['value'] for c in chains}
    rescued = [c for c in chains
               if c[-1]['value'] == 'P2' and c[-1].get('reciprocal')]
    assert len(rescued) == 1


def test_reciprocal_flag_reaches_standardized_linkers():
    """The collapsed landing+arrival linker keeps the rescue flag so
    ``mapping_origin`` can disclose it."""
    m = _bare_mapper()
    _seed_glue_pair(m, reciprocate=True)
    rescued = [c for c in m.get_type_bridges('T1', FAFB, BANC)
               if c[-1]['value'] == 'P2' and c[-1].get('reciprocal')][0]
    linkers = standardize_bridge(rescued, FAFB, BANC)
    act = [l for l in linkers
           if l['column'] == 'Alternative Cell Type(s)']
    assert act and act[0].get('reciprocal') is True


def test_branch_record_linkers_carry_value_key():
    """get_mapping_branches' linker projection keeps ``value`` (and the
    rescue flag) — the per-bridge export's mapping_origin reads tokens
    from it (2026-09-30 review: the stripped projection made
    annotation-only rows export a bare ``mapped`` origin)."""
    from comparison.mapping_visualization import branches_to_per_bridge_rows
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    # the decision layer needs loader-built caches the bare mapper does
    # not seed; stub the parent decision — this test targets the linker
    # projection, not the decision machinery.
    m.get_mapping_decision = lambda *a, **k: {
        'status': 'mapped', 'target_types': ['P1'],
        'relationship': '1-to-1'}
    records = m.get_mapping_branches(
        'T1', FAFB, BANC, pool_fn=lambda *a, **k: None)
    assert records
    rows = branches_to_per_bridge_rows(records)
    origins = {r['mapping_origin'] for r in rows}
    assert any('annotation bridge via V1' in o for o in origins), origins
    for record in records:
        for linker in record['linkers']:
            assert linker.get('value'), linker


# ---------------------------------------------------------------------------
# Production overlay: same-name identity + annotation-bridge candidates
# ---------------------------------------------------------------------------

def test_overlay_same_name_identity_maps_shared_primaries():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'APDN3'},
        banc_primaries={'APDN3', 'CB1'},
    )
    m._apply_annotation_bridge_overlay()
    assert (m._type_mappings[FAFB]['APDN3'][BANC] == 'APDN3')
    assert (m._type_mappings[BANC]['APDN3'][FAFB] == 'APDN3')
    assert m._bridge_provenance[(FAFB, 'APDN3', BANC)]['kind'] == 'same name'


def test_overlay_annotation_bridge_one_to_one_maps():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    assert m._type_mappings[FAFB]['T1'][BANC] == 'P1'
    prov = m._bridge_provenance[(FAFB, 'T1', BANC)]
    assert prov['kind'] == 'annotation bridge'
    assert prov['target'] == 'P1'
    assert prov['via'] == ['V1']


def test_overlay_multi_candidate_becomes_conflict_not_guess():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1', 'P2', 'P3'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1', 'P2', 'P3'}},
    )
    m._apply_annotation_bridge_overlay()
    assert 'T1' not in m._type_mappings[FAFB]
    conflicts = [c for c in m._conflicts
                 if c.source_type == 'T1' and c.source_dataset == FAFB]
    assert len(conflicts) == 1
    assert conflicts[0].target_types == {'P1', 'P2', 'P3'}
    assert conflicts[0].relationship == '1-to-N'
    assert conflicts[0].origin == 'annotation bridge'
    assert m._bridge_provenance[(FAFB, 'T1', BANC)]['target'] is None


def test_overlay_crosswalk_resolution_wins():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    # The crosswalk already resolved this pair: the overlay adds nothing.
    m._type_mappings[FAFB]['T1'] = {BANC: 'P1'}
    m._apply_annotation_bridge_overlay()
    assert (FAFB, 'T1', BANC) not in m._bridge_provenance


def test_overlay_does_not_duplicate_existing_conflicts():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1', 'P2'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1', 'P2'}},
    )
    m._conflicts.append(TypeMappingConflict(
        source_dataset=FAFB, target_dataset=BANC, source_type='T1',
        target_types={'P1', 'P2'}, relationship='1-to-N',
    ))
    m._apply_annotation_bridge_overlay()
    matches = [c for c in m._conflicts
               if c.source_type == 'T1' and c.source_dataset == FAFB]
    assert len(matches) == 1  # the crosswalk one; provenance still recorded
    assert m._bridge_provenance[(FAFB, 'T1', BANC)]['target'] is None


def test_overlay_excludes_untyped_sentinels():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'Unknown', '123', 'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'Unknown', '123', 'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    # Unknown/numeric primaries are excluded as candidates; P1 remains
    # the sole candidate and becomes the mapping.
    assert m._type_mappings[FAFB]['T1'][BANC] == 'P1'
    all_targets = [
        t for c in m._conflicts for t in c.target_types]
    assert 'Unknown' not in all_targets
    assert '123' not in all_targets


def test_overlay_untyped_token_never_evidences():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'Unknown': {'T1'}},   # a useless annotation token
        banc_ann={'Unknown': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    assert 'T1' not in m._type_mappings[FAFB]
    assert not m._conflicts


def test_overlay_flows_through_get_mapped_type():
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'APDN3', 'T1'},
        banc_primaries={'APDN3', 'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    m._build_reverse_lookup()
    # Same-name identity (the probe's APDN3 case) and the annotation
    # bridge both resolve through the release name 'banc_v888'.
    assert m.get_mapped_type('APDN3', FAFB_RELEASE, BANC_RELEASE) == 'APDN3'
    assert m.get_mapped_type('T1', FAFB_RELEASE, BANC_RELEASE) == 'P1'


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def test_export_mapping_bridge_rows_and_origin_column(tmp_path):
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    out = tmp_path / 'mapping.csv'
    m.export_mapping(str(out), datasets=[FAFB_RELEASE, BANC_RELEASE])
    df = pd.read_csv(out)
    assert list(df.columns) == [FAFB_RELEASE, BANC_RELEASE,
                                'mapping_origin', 'mapping_support']
    row = df[(df[FAFB_RELEASE] == 'T1') & (df[BANC_RELEASE] == 'P1')]
    assert len(row) == 1
    assert 'annotation bridge' in row['mapping_origin'].iloc[0]
    assert 'V1' in row['mapping_origin'].iloc[0]


def test_export_mapping_same_name_row_respects_only_different(tmp_path):
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'APDN3'},
        banc_primaries={'APDN3'},
    )
    m._apply_annotation_bridge_overlay()
    out = tmp_path / 'diff.csv'
    m.export_mapping(str(out), datasets=[FAFB_RELEASE, BANC_RELEASE],
                     only_different=True)
    df = pd.read_csv(out)
    assert df.empty  # identical-name rows are not "different" mappings
    out2 = tmp_path / 'all.csv'
    m.export_mapping(str(out2), datasets=[FAFB_RELEASE, BANC_RELEASE],
                     only_different=False)
    df2 = pd.read_csv(out2)
    assert (df2[FAFB_RELEASE] == 'APDN3').any()
    assert 'same name' in df2['mapping_origin'].iloc[0]


def test_export_mapping_anchor_row_prevents_duplicate(tmp_path):
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._apply_annotation_bridge_overlay()
    # An mcns anchor row already carries both endpoints: the overlay must
    # not append a duplicate row.
    m._type_mappings[MCNS]['M1'] = {
        FAFB: 'T1', BANC: 'P1', 'hemibrain:v1.2.1': 'H1',
    }
    out = tmp_path / 'mapping.csv'
    m.export_mapping(str(out), datasets=[MCNS, FAFB_RELEASE, BANC_RELEASE])
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df['mapping_origin'].iloc[0] == 'crosswalk'
    assert df[MCNS].iloc[0] == 'M1'


def test_export_conflicts_resolves_release_keys(tmp_path):
    m = _bare_mapper()
    _seed_pair(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1', 'P2'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1', 'P2'}},
    )
    m._apply_annotation_bridge_overlay()
    out = tmp_path / 'conflicts.csv'
    # The run selected the LEGACY release spelling flywire_BANC_v626; the
    # conflict is keyed by the MAPPING namespace banc_v626 — it must
    # survive the release-name resolution and carry its origin.  (A
    # banc_v888-scoped export must NOT surface banc_v626 conflicts —
    # per-release namespaces, §version control.)
    m.export_conflicts(str(out), datasets=[FAFB_RELEASE, 'flywire_BANC_v626'])
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df['source_dataset'].iloc[0] == FAFB
    assert df['target_dataset'].iloc[0] == BANC
    assert df['origin'].iloc[0] == 'annotation bridge'


# ---------------------------------------------------------------------------
# Real-data acceptance (skip when the local datasets are absent)
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not MCNS_TABLE.exists(),
    reason='real male-cns v1.0 neuron table not available locally')


@pytest.fixture(scope='module')
def real_mapper():
    if not MCNS_TABLE.exists():
        pytest.skip('real datasets not available')
    m = CrossDatasetTypeMapper(verbose=False, workspace_path=str(REPO_ROOT))
    assert m.load() is True
    return m


@pytest.mark.skipif(
    not MCNS_TABLE.exists(),
    reason='real male-cns v1.0 neuron table not available locally')
class TestRealDataAcceptance:
    """The circadian-gap acceptance set (probe:
    a one-shot ``type_mapping_bridge_probe.py`` under local_data/, now gone:
    its findings are the assertions below."""

    def test_apdn3_same_name_resolves_production(self, real_mapper):
        assert real_mapper.get_mapped_type(
            'APDN3', FAFB_RELEASE, BANC_RELEASE) == 'APDN3'

    def test_slnv_still_resolves_via_crosswalk(self, real_mapper):
        assert real_mapper.get_mapped_type(
            's-LNv', FAFB_RELEASE, BANC_RELEASE) == 's-LNv_b'

    def test_scpdn3a_yields_annotation_candidates(self, real_mapper):
        # Derivation view: the composed bridge chains exist for the
        # shared annotation tokens (CB1770 is a BANC primary).
        chains = real_mapper.get_type_bridges(
            's-CPDN3A', FAFB_RELEASE, BANC_RELEASE)
        ends = {c[-1]['value'] for c in chains}
        assert {'CB1770', 'CB1791', 'SMP229'} <= ends
        # ACT-glue suppression (2026-09-30): the CB1791/CB3612 SMP220
        # cross-glue is gone — forward and backward now agree 1:1.
        assert 'CB3612' not in ends
        # The curated BANC FAFB labels independently expose three BANC
        # primaries.  They remain valid split evidence, but no single BANC
        # target is accepted merely because one direct chain is ranked first.
        mapped = real_mapper.get_mapped_type(
            's-CPDN3A', FAFB_RELEASE, BANC_RELEASE)
        assert mapped is None
        decision = real_mapper.get_mapping_decision(
            's-CPDN3A', FAFB_RELEASE, BANC_RELEASE)
        assert decision['status'] == 'valid_split_evidence'
        assert set(decision['target_types']) == {
            'CB1770', 'CB1791', 'SMP229'}
        provenance = real_mapper._bridge_provenance[
            (FAFB_RELEASE, 's-CPDN3A', BANC_RELEASE)]
        # The aggregate provenance key remains the annotation overlay; the
        # source-scoped conflict record above carries the BANC fan-out.
        assert provenance['kind'] == 'annotation bridge'
        assert set(provenance['via']) == {
            'CB1770', 'CB1791', 'SMP229'}

    def test_no_untyped_conflict_targets(self, real_mapper):
        # Scope to overlay-origin conflicts: legacy crosswalk conflicts
        # legitimately carry the BANC convention's 'Unknown' type label.
        for c in real_mapper._conflicts:
            if c.origin != 'annotation bridge':
                continue
            for t in c.target_types:
                assert not CrossDatasetTypeMapper._is_untyped_value(t), (
                    c.source_type, t)

    def test_scpdn3_mirror_side_of_the_cross_glue_also_gone(self, real_mapper):
        # s-CPDN3D's ends keep the curated fan-out but lose the CB1791
        # cross-glue; both directions of the pair space now agree.
        ends = {c[-1]['value'] for c in real_mapper.get_type_bridges(
            's-CPDN3D', FAFB_RELEASE, BANC_RELEASE)}
        assert 'CB3612' in ends
        assert 'CB1791' not in ends

    def test_backward_clock_routes_unchanged_by_suppression(
            self, real_mapper):
        # The reverse direction never had the glue (its arrival lands in
        # the source namespace); the fix must not disturb it.
        for banc_type, fafb_type in (('CB1791', 's-CPDN3A'),
                                     ('CB3612', 's-CPDN3D')):
            ends = {c[-1]['value'] for c in real_mapper.get_type_bridges(
                banc_type, BANC_RELEASE, FAFB_RELEASE)}
            assert ends == {fafb_type}

    def test_r8_act_hub_collapses_to_same_name(self, real_mapper):
        # R8's own ACT sub-type tokens (R7p/R8p/R8y/...) used to glue it
        # onto 26 unrelated BANC ends incl. m_NSC_DILP and MBON26.
        ends = {c[-1]['value'] for c in real_mapper.get_type_bridges(
            'R8', FAFB_RELEASE, BANC_RELEASE)}
        assert ends == {'R8'}

    def test_reciprocal_glue_pair_survives_with_flag(self, real_mapper):
        # DNp17 <-> DNpe054 is real mutual correspondence (the reverse
        # leg runs through FAFB additional_type(s) 'DNp16/17'): kept and
        # disclosed as reciprocal.
        from comparison.mapping_visualization import _pair_mapping_origin
        rescued = [
            c for c in real_mapper.get_type_bridges(
                'DNp17', FAFB_RELEASE, BANC_RELEASE)
            if c[-1]['value'] == 'DNpe054']
        assert len(rescued) == 1
        assert rescued[0][-1].get('reciprocal') is True
        linkers = standardize_bridge(rescued[0], FAFB_RELEASE, BANC_RELEASE)
        origin = _pair_mapping_origin(linkers, 'DNp17', 'DNpe054')
        assert origin == 'annotation bridge via DNht001 (reciprocal)'

    def test_per_bridge_export_origin_and_reciprocal(self, real_mapper,
                                                     tmp_path):
        # End-to-end through the export: annotation-lane rows name their
        # tokens, the rescued glue pair carries (reciprocal), and no row
        # falls back to a bare 'mapped' origin (2026-09-30 review fix).
        out = tmp_path / 'mapping_review.csv'
        n = real_mapper.export_mapping_per_bridge(
            str(out), source_types=['DNp17', 'LLPC2'],
            source_dataset=FAFB_RELEASE, target_datasets=[BANC_RELEASE],
            pool_fn=lambda *a, **k: None)
        assert n > 0
        df = pd.read_csv(out, dtype=str).fillna('')
        rescued = df[(df['source_type'] == 'DNp17')
                     & (df['target_type'] == 'DNpe054')]
        assert len(rescued) == 1
        assert rescued['mapping_origin'].iloc[0] == \
            'annotation bridge via DNht001 (reciprocal)'
        designed = df[(df['source_type'] == 'LLPC2')
                      & (df['target_type'] == 'LLPC1')]
        assert len(designed) >= 1
        assert designed['mapping_origin'].iloc[0].startswith(
            'annotation bridge via LLPC2')
        assert not (df['mapping_origin'] == 'mapped').any()


def test_scoped_pair_bridges_stay_two_namespace():
    """Direction-symmetric registry scoping (user directive): a FAFB ->
    male-cns bridge must not detour through BANC — neither namespace hops
    nor Alternative Cell Type(s) columns. The composed annotation chains
    belong to the FAFB <-> BANC pair only."""
    from comparison.cross_dataset_type_mapper import get_type_mapper

    if not MCNS_TABLE.exists():
        pytest.skip("real datasets not available")
    m = get_type_mapper()
    for t in ("APDN3", "s-LNv", "l-LNv", "DN1pA", "CL125", "MDN"):
        for chain in m.get_type_bridges(t, FAFB, "male-cns:v1.0"):
            for hop in chain[1:]:
                assert hop["dataset"] != BANC, (t, chain)
                assert hop["column"] != "Alternative Cell Type(s)", (t, chain)
    # the same holds for the reverse direction (symmetric scoping)
    for t in ("APDN3", "s-LNv", "DN1pA"):
        for chain in m.get_type_bridges(t, "male-cns:v1.0", FAFB):
            for hop in chain[1:]:
                assert hop["dataset"] != BANC, (t, chain)
