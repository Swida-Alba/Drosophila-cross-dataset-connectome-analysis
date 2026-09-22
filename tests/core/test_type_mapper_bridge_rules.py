"""General bridge-rule tests (user's 2026-09-06 algebra).

Rules under test:

  HEMI --hT-- MCNS;  MANC --mT-- MCNS
  MCNS --fT--aT-- FAFB | --fT-- | --aT--
  MCNS --mct-- BANC | --mct--
  FAFB --aT--ACT-- BANC | --aT-- | --ACT--
  HEMI --hemibrain_cell_type-- BANC
  MANC --manc_cell_type-- BANC
  + same-name identity everywhere.

  Connectors: MCNS connects the neuprint datasets through itself; BANC label
  columns are direct-only evidence and BANC is NEVER a connector. All bridges
  are bidirectional
  (the reverse travel direction reverses the hop order — the SAME bridge,
  not a flip) and a two-linker chain cannot flip its linker order.

Walk-level tests run on a hermetic bare mapper (internal tables injected
directly); real-data acceptance tests skip when the local datasets are
absent.  Probe: local_data/bridge_rules_probe.py.
"""

from pathlib import Path

import pytest

from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v626'
MCNS = 'male-cns:v1.0'
HEMI = 'hemibrain:v1.2.1'
MANC = 'manc:v1.0'

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_TABLE = (REPO_ROOT / 'datasets' / 'male-cns_v1_0'
              / 'male-cns_v1_0_allneurons_neuron_df.csv')

CHAIN_KEYS = ('dataset', 'column', 'value')


def chain_key(chain):
    return tuple(tuple(hop[k] for k in CHAIN_KEYS) for hop in chain)


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
    m._crosswalk_parts_cache = {}
    m._crosswalk_reverse_cache = {}
    m._banc_label_edges = {}
    m._banc_release_edges = {}
    m._mcns_v09_shared_names = set()
    return m


def _seed_flywire(m, fafb_primaries=(), banc_primaries=(),
                  fafb_ann=None, banc_ann=None):
    m._flywire_primaries[FAFB] = set(fafb_primaries)
    m._flywire_primaries[BANC] = set(banc_primaries)
    for key, ann in ((FAFB, fafb_ann or {}), (BANC, banc_ann or {})):
        annotation = {v: set(ps) for v, ps in ann.items()}
        m._flywire_annotation_primaries[key] = annotation
        filtered = {}
        for value, primaries in annotation.items():
            if value in m._flywire_primaries[key]:
                continue
            for primary in primaries:
                filtered.setdefault(value, set()).add(primary)
        m._flywire_alt_to_primary[key] = filtered
        inverse = {}
        for value, primaries in annotation.items():
            for primary in primaries:
                inverse.setdefault(primary, set()).add(value)
        m._flywire_primary_to_alts[key] = inverse


def routes(chains):
    out = []
    for chain in chains:
        route = []
        for hop in chain:
            if not route or route[-1] != hop['dataset']:
                route.append(hop['dataset'])
        out.append(tuple(route))
    return out


# ---------------------------------------------------------------------------
# Reverse crosswalk legs: every crosswalk pair is bidirectional
# ---------------------------------------------------------------------------

def test_reverse_crosswalk_leg_connects_renamed_hemi_pair():
    """HEMI --hT-- MCNS reversed: a renamed hemibrain type hops back to
    the male-cns primary whose hemibrainType cell names it."""
    m = _bare_mapper()
    m._dataset_types = {
        MCNS: {'M1': {'b1'}, 'ALIN4': {'b1'}},
        HEMI: {'lLN7': {'b1'}},
    }
    m._crosswalk_reverse_cache = {'hemibrainType': {'lLN7': {'ALIN4'}}}
    chains = m.get_type_bridges('lLN7', HEMI, MCNS)
    expected = (
        (HEMI, 'type', 'lLN7'),
        (MCNS, 'hemibrainType', 'ALIN4'),
    )
    assert expected in [chain_key(c) for c in chains]
    # and no chain leaves the pair's two namespaces
    assert all(set(r) <= {HEMI, MCNS} for r in routes(chains))


def test_reverse_flywire_crosswalk_leg_fafb_to_mcns():
    """MCNS --fT-- FAFB reversed: a FAFB name inside a male-cns flywireType
    cell hops back to the male-cns primaries (renamed pair, no same-name)."""
    m = _bare_mapper()
    _seed_flywire(m, fafb_primaries={'T1'})
    m._dataset_types = {MCNS: {'M1': {'b1'}}, FAFB: {'T1': {'b1'}}}
    m._crosswalk_reverse_cache = {'flywireType': {'T1': {'M1'}}}
    chains = m.get_type_bridges('T1', FAFB, MCNS)
    expected = (
        (FAFB, 'type', 'T1'),
        (MCNS, 'flywireType', 'M1'),
    )
    assert expected in [chain_key(c) for c in chains]


# ---------------------------------------------------------------------------
# Connector licensing
# ---------------------------------------------------------------------------

def test_banc_is_never_a_connector_for_mcns_fafb():
    """MCNS -- ... -- FAFB must not detour through BANC even when a BANC
    same-name type annotates the FAFB target."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'X', 'P1'},
        banc_ann={'X': {'P1'}},      # BANC rows typed X annotate P1
    )
    m._dataset_types = {
        MCNS: {'X': {'b1'}},         # same-name X in MCNS and BANC
        FAFB: {'T1': {'b1'}},
        BANC: {'X': {'b1'}, 'P1': {'b1'}},
    }
    chains = m.get_type_bridges('X', MCNS, FAFB)
    for chain in chains:
        for hop in chain[1:]:
            assert hop['dataset'] != BANC, chain
            assert hop['column'] != 'Alternative Cell Type(s)', chain


def test_mcns_banc_uses_the_direct_mct_label_bridge():
    """MCNS↔BANC uses ``malecns_cell_type`` directly, never fT→FAFB→ACT."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        banc_ann={'T1': {'P1'}},
    )
    m._dataset_types = {
        MCNS: {'M1': {'b1'}},
        FAFB: {'T1': {'b1'}},
        BANC: {'P1': {'b1'}},
    }
    m._banc_label_edges = {
        (MCNS, 'M1'): [
            (BANC, 'P1', 'malecns_cell_type', 'P1', BANC),
        ],
        (BANC, 'P1'): [
            (MCNS, 'M1', 'malecns_cell_type', 'M1', BANC),
        ],
    }
    m._crosswalk_parts_cache = {('flywireType', 'M1'): ['T1']}
    chains = m.get_type_bridges('M1', MCNS, BANC)
    assert chains
    assert routes(chains) == [(MCNS, BANC)]
    assert all(
        all(hop['column'] != 'flywireType' for hop in chain[1:])
        for chain in chains
    )
    assert any(
        any(hop['column'] == 'malecns_cell_type' for hop in chain[1:])
        for chain in chains
    )


def test_hemi_banc_uses_the_direct_curated_label_bridge():
    """HEMI↔BANC is direct ``hemibrain_cell_type`` evidence."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        banc_primaries={'P1'},
        banc_ann={'M1': {'P1'}},
    )
    m._dataset_types = {
        MCNS: {'M1': {'b1'}},
        HEMI: {'H1': {'b1'}},
        BANC: {'P1': {'b1'}},
    }
    m._banc_label_edges = {
        (HEMI, 'H1'): [
            (BANC, 'P1', 'hemibrain_cell_type', 'P1', BANC),
        ],
        (BANC, 'P1'): [
            (HEMI, 'H1', 'hemibrain_cell_type', 'H1', BANC),
        ],
    }
    m._crosswalk_reverse_cache = {'hemibrainType': {'H1': {'M1'}}}
    chains = m.get_type_bridges('H1', HEMI, BANC)
    assert chains
    assert routes(chains) == [(HEMI, BANC)]
    assert all(
        any(hop['column'] == 'hemibrain_cell_type' for hop in chain[1:])
        for chain in chains
    )


def test_fafb_banc_pair_has_no_mcns_detour():
    """FAFB <-> BANC is direct-only: MCNS is not a licensed connector for
    the flywire pair, so same-name echoes through MCNS are not offered."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1', 'T2'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._dataset_types = {
        MCNS: {'V1': {'b1'}},        # the token echoes as an MCNS type
        FAFB: {'T1': {'b1'}},
        BANC: {'P1': {'b1'}, 'T2': {'b1'}},
    }
    m._banc_label_edges = {
        (FAFB, 'T1'): [
            (BANC, 'P1', 'fafb_cell_type', 'P1', BANC),
        ],
        (BANC, 'P1'): [
            (FAFB, 'T1', 'fafb_cell_type', 'T1', BANC),
        ],
    }
    chains = m.get_type_bridges('T1', FAFB, BANC)
    assert chains
    assert all(set(r) <= {FAFB, BANC} for r in routes(chains))


def test_hemi_fafb_derives_through_two_crosswalk_linkers():
    """HEMI --hT-- MCNS --fT-- FAFB: the pair's MAXIMIZED linker chain.

    The composed route is the only linker derivation this licensed
    connector pair has (all its chains' linkers are these two), so the
    standardization must carry BOTH linker columns and every derived
    surface (preferred chain, bridge_linker_text -> the CSV
    ``bridge-<column>`` fields) must show them (§9.3)."""
    from comparison.cross_dataset_type_mapper import (
        bridge_linker_text,
        preferred_bridge_chain,
        standardize_bridge,
    )

    m = _bare_mapper()
    _seed_flywire(m, fafb_primaries={'F1'})
    m._dataset_types = {
        MCNS: {'M1': {'b1'}},
        HEMI: {'H1': {'b1'}},
        FAFB: {'F1': {'b1'}},
    }
    m._crosswalk_reverse_cache = {'hemibrainType': {'H1': {'M1'}}}
    m._crosswalk_parts_cache = {('flywireType', 'M1'): ['F1']}

    chains = m.get_type_bridges('H1', HEMI, FAFB)
    expected = (
        (HEMI, 'type', 'H1'),
        (MCNS, 'hemibrainType', 'M1'),
        (FAFB, 'flywireType', 'F1'),
    )
    assert expected in [chain_key(c) for c in chains]
    assert routes(chains) == [(HEMI, MCNS, FAFB)]

    best = preferred_bridge_chain(chains, HEMI, FAFB)
    linkers = standardize_bridge(best, HEMI, FAFB)
    assert [(l['column'], l['value']) for l in linkers] == [
        ('hemibrainType', 'M1'), ('flywireType', 'F1')]

    info = bridge_linker_text(chains, HEMI, FAFB, 'F1')
    assert {e['column'] for e in info['entries']} == {
        'hemibrainType', 'flywireType'}


# ---------------------------------------------------------------------------
# Order and untyped rules
# ---------------------------------------------------------------------------

def test_two_linker_composed_chain_hop_order_is_fixed():
    """FAFB --aT--ACT-- BANC forward: aT hop precedes the ACT legs; the
    reverse direction reverses the hop order (same bridge, not a flip)."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'P1'}},
    )
    m._dataset_types = {
        FAFB: {'T1': {'b1'}},
        BANC: {'P1': {'b1'}},
    }
    forward = m.get_type_bridges('T1', FAFB, BANC)
    expected = (
        (FAFB, 'type', 'T1'),
        (FAFB, 'additional_type(s)', 'V1'),
        (BANC, 'Alternative Cell Type(s)', 'V1'),
        (BANC, 'Alternative Cell Type(s)', 'P1'),
    )
    assert expected in [chain_key(c) for c in forward]
    reverse = m.get_type_bridges('P1', BANC, FAFB)
    expected_rev = (
        (BANC, 'type', 'P1'),
        (BANC, 'Alternative Cell Type(s)', 'V1'),
        (FAFB, 'additional_type(s)', 'V1'),
        (FAFB, 'additional_type(s)', 'T1'),
    )
    assert expected_rev in [chain_key(c) for c in reverse]


def test_untyped_names_never_become_bridge_targets():
    """The BANC 'Unknown' convention hub must not appear as a chain end."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        banc_primaries={'Unknown', '123', 'P1'},
        fafb_ann={'V1': {'T1'}},
        banc_ann={'V1': {'Unknown', '123', 'P1'}},
    )
    m._dataset_types = {
        FAFB: {'T1': {'b1'}},
        BANC: {'Unknown': {'b1'}, '123': {'b1'}, 'P1': {'b1'}},
    }
    chains = m.get_type_bridges('T1', FAFB, BANC)
    assert chains
    for chain in chains:
        assert not CrossDatasetTypeMapper._is_untyped_value(
            chain[-1]['value']), chain


def test_annotation_reverse_evidence_is_mcns_side_only():
    """MCNS --aT-- FAFB / MCNS --ACT-- BANC are male-cns-side forms: a
    hemibrain type named in FAFB annotation cells reaches FAFB only via
    the MCNS connector (hT leg), never by a direct annotation hop."""
    m = _bare_mapper()
    _seed_flywire(
        m,
        fafb_primaries={'T1'},
        fafb_ann={'H1': {'T1'}},
    )
    m._dataset_types = {
        MCNS: {'M1': {'b1'}},
        HEMI: {'H1': {'b1'}},
        FAFB: {'T1': {'b1'}},
    }
    chains = m.get_type_bridges('H1', HEMI, FAFB)
    for chain in chains:
        if len(chain) > 2:
            # any multi-hop chain must pass through MCNS
            assert (HEMI, MCNS, FAFB) in routes([chain]), chain


# ---------------------------------------------------------------------------
# Real-data acceptance (skip when the local datasets are absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MCNS_TABLE.exists(),
    reason='real male-cns v1.0 neuron table not available locally')
class TestRealBridgeRules:
    """Real-data acceptance for the general bridge rules."""

    @pytest.fixture(scope='class')
    def mapper(self):
        m = CrossDatasetTypeMapper(verbose=False,
                                   workspace_path=str(REPO_ROOT))
        assert m.load() is True
        return m

    def test_renamed_hemi_reaches_mcns(self, mapper):
        chains = mapper.get_type_bridges('lLN7', HEMI, MCNS)
        assert chains, 'renamed HEMI type must hop back via hT'
        assert all(c[-1]['dataset'] == MCNS for c in chains)
        assert any(
            any(h['column'] == 'hemibrainType' for h in c[1:])
            for c in chains)

    def test_hemi_banc_routes_stay_licensed(self, mapper):
        # Select a real HEMI label edge rather than assuming a particular
        # cell type is curated in every BANC release.
        source_type = None
        for (source_key, candidate), edges in mapper._banc_label_edges.items():
            if source_key != HEMI:
                continue
            if any(edge[0] == BANC for edge in edges):
                source_type = candidate
                break
        if source_type is None:
            pytest.skip('no grounded hemibrain_cell_type BANC edge locally')
        chains = mapper.get_type_bridges(source_type, HEMI, BANC)
        assert chains
        for chain in chains:
            for hop in chain[1:]:
                assert hop['dataset'] in (HEMI, BANC), chain
        assert any(
            (HEMI, BANC) == routes([c])[0] for c in chains)

    def test_no_untyped_chain_targets(self, mapper):
        for t in ('MDN', 'CL125', 'APDN3', 'DN1pA', 'lLN7', 'ADNM1 MN'):
            for src, tgt in ((MCNS, BANC), (BANC, MCNS), (FAFB, BANC),
                             (BANC, FAFB), (MCNS, HEMI), (HEMI, MCNS)):
                for chain in mapper.get_type_bridges(t, src, tgt):
                    assert not CrossDatasetTypeMapper._is_untyped_value(
                        chain[-1]['value']), (t, src, tgt, chain)


# ---------------------------------------------------------------------------
# BANC label-hop terminality (user 2026-09-09: the l-LNv -> BM_* fan-out)
# ---------------------------------------------------------------------------

def test_label_hop_and_primary_valued_alt_are_derivation_dead_ends():
    """The FAFB l-LNv -> BANC BM_* screenshot regression.

    Real shape (BANC v888): one l-LNv neuron carries curated
    malecns_cell_type 'BM_InOm', so l-LNv's Alternative Cell Type(s)
    lists 'BM_InOm' — ANOTHER BANC primary.  Composing the l-LNv label
    bridge with BANC-internal annotation hops used to derive
    l-LNv -> every primary whose cells list 'BM_InOm' (BM_InOm's 1,212
    neurons included) with zero supporting rows on the reached types.

    Two dead ends close this:

    * curated label hops (fafb_cell_type et al.) are derivation
      ENDPOINTS — nothing may follow them;
    * a primary's annotation listing another primary's name is a
      cross-reference, not a rename, so no primary->alt edge may hop to
      a primary-valued token.
    """
    m = _bare_mapper()
    m._dataset_types = {
        FAFB: {},
        BANC: {'l-LNv': {'b1'}, 'BM_InOm': {'b2'}, 'BM_MaPa': {'b3'}},
    }
    _seed_flywire(m, fafb_primaries=('l-LNv',),
                  banc_primaries=('l-LNv', 'BM_InOm', 'BM_MaPa'))
    # l-LNv(BANC)'s ACT lists 'BM_InOm' (another primary); the BM_* types'
    # own cells list 'BM_InOm' as well.
    m._flywire_primary_to_alts[BANC] = {'l-LNv': {'BM_InOm'}}
    m._flywire_alt_to_primary[BANC] = {'BM_InOm': {'l-LNv', 'BM_MaPa'}}
    # Curated label evidence: BANC l-LNv rows carry fafb_cell_type
    # 'l-LNv' — a direct, supported 1-hop bridge in both directions.
    m._banc_label_edges = {
        (BANC, 'l-LNv'): [(FAFB, 'l-LNv', 'fafb_cell_type', 'l-LNv', BANC)],
        (FAFB, 'l-LNv'): [(BANC, 'l-LNv', 'fafb_cell_type', 'l-LNv', BANC)],
    }

    forward = m.get_type_bridges('l-LNv', FAFB, BANC)
    assert {chain[-1]['value'] for chain in forward} == {'l-LNv'}
    label_chain = (
        (FAFB, 'type', 'l-LNv'),
        (BANC, 'fafb_cell_type', 'l-LNv'),
    )
    assert label_chain in {chain_key(c) for c in forward}

    reverse = m.get_type_bridges('l-LNv', BANC, FAFB)
    assert {chain[-1]['value'] for chain in reverse} == {'l-LNv'}

    # BM_MaPa has no route into FAFB: its own cells list 'BM_InOm', but
    # that token is a BANC primary, so the primary->alt hop is refused.
    assert m.get_type_bridges('BM_MaPa', BANC, FAFB) == []


def test_designed_fafb_at_to_banc_act_standard_survives():
    """The legit annotation standard keeps deriving: a FAFB primary's
    additional_type(s) token 'X' that is NOT itself a primary lands in
    BANC's ACT cells and reaches the BANC primaries annotated with X."""
    m = _bare_mapper()
    m._dataset_types = {FAFB: {}, BANC: {'Q1': {'b1'}, 'Q2': {'b2'}}}
    _seed_flywire(
        m, fafb_primaries=('P1',), banc_primaries=('Q1', 'Q2'),
        fafb_ann={'X': {'P1'}},           # P1's rows list old name 'X'
        banc_ann={'X': {'Q1', 'Q2'}},     # BANC ACT cells listing 'X'
    )
    chains = m.get_type_bridges('P1', FAFB, BANC)
    assert {chain[-1]['value'] for chain in chains} == {'Q1', 'Q2'}
    for chain in chains:
        columns = [h['column'] for h in chain[1:]]
        assert columns == ['additional_type(s)', 'Alternative Cell Type(s)',
                           'Alternative Cell Type(s)'], chain


def test_the_basis_vocabulary_separates_rows_from_name_assertions():
    """One definition, read by the resolver, the validation pipeline, the
    report and the scenes — a consumer that compares the basis with one
    literal silently misread every other row-backed basis."""
    from comparison.cross_dataset_type_mapper import (
        SOURCE_REFINING_BASES, basis_is_row_evidence)
    assert basis_is_row_evidence('linker rows')
    assert basis_is_row_evidence('release relation participants')
    assert basis_is_row_evidence('linker rows + alternative chains')
    for wide in ('full population', 'unmeasured', '', None,
                 'union of supported bridge pools'):
        assert not basis_is_row_evidence(wide)
    # a widened bucket is a union across chains, so it may not re-decide a
    # side — that stays the narrower question the swap guards on
    assert 'linker rows + alternative chains' not in SOURCE_REFINING_BASES


def test_the_source_side_refinement_is_disclosed_only_when_it_differs():
    """FAFB->BANC selects a target-side-only chain, so the panel and the
    visualization must say which chain the SOURCE pool actually came from
    (r23: 34 of 39 branches)."""
    from comparison.cross_dataset_type_mapper import source_side_refinement_note
    selected = [{'dataset': 'src', 'column': 'type', 'value': 'T1'},
                {'dataset': 'tgt', 'column': 'fafb_cell_type', 'value': 'X1'}]
    alt = [{'dataset': 'src', 'column': 'type', 'value': 'T1'},
           {'dataset': 'src', 'column': 'additional_type(s)', 'value': 'X1'},
           {'dataset': 'tgt', 'column': 'type', 'value': 'X1'}]
    assert source_side_refinement_note(None, 'src', 'tgt') == ''
    assert source_side_refinement_note(
        {'selected_chain': selected}, 'src', 'tgt') == ''
    note = source_side_refinement_note(
        {'selected_chain': selected,
         'source_side_refinement': {'chain': selected,
                                    'source_basis': 'linker rows'}},
        'src', 'tgt')
    assert note == ''                          # the winner already supplies it
    note = source_side_refinement_note(
        {'selected_chain': selected,
         'source_side_refinement': {'chain': alt,
                                    'source_basis': 'linker rows'}},
        'src', 'tgt')
    assert 'source side from another supported chain' in note
    assert 'additional_type(s)' in note and "'X1'" in note
    assert 'linker rows' in note
