"""Real-dataset validation for the cross-dataset type mapping.

The hermetic tests in ``test_type_mapper_coverage.py`` validate the mapping
logic on synthetic tables.  This module loads the *actual* local datasets
(male-cns v1.0, FAFB v783, standalone BANC v626) and validates the
additional Type(S) rename resolution end to end:

* male-cns ``SLP249`` (flywireType ``SLP249``) must resolve to FAFB ``APDN3``
  because FAFB v783 renamed those neurons and keeps ``SLP249`` only in its
  ``additional_type(s)`` column;
* every resolved mapping target must be a *current* primary type of the
  target dataset (or a name genuinely absent from it) - never a stale
  additional-only name;
* ambiguous renames (one old name listed under several primaries) and
  comma-separated crosswalk cells must produce conflicts, not guesses;
* FAFB and BANC rename independently, so their namespaces are checked
  separately.

All tests skip when the local datasets are not present.
"""

from pathlib import Path

import pandas as pd
import pytest

from comparison.cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    get_type_mapper,
    preferred_bridge_chain,
    standardize_bridge,
)
from comparison.mapping_visualization import build_type_coverage
from comparison.type_resolver import (
    STATUS_CLAIMED,
    STATUS_MAPPED,
    resolve_valid_targets,
)
from ui.neuron_index import (
    collect_native_type_matches,
    enrich_native_type_matches,
    load_cached_neuron_index,
    mapped_type_targets,
    pool_bridge_body_ids,
    resolve_prioritized_bridge_pool,
)

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'banc_v626'
HB = 'hemibrain:v1.2.1'

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_CSV = REPO_ROOT / 'datasets' / 'male-cns_v1_0' / 'male-cns_v1_0_allneurons_neuron_df.csv'
FAFB_CSV = REPO_ROOT / 'datasets' / 'flywire_FAFB_v783' / 'flywire_FAFB_v783_allneurons_neuron_df.csv'
BANC_CSV = REPO_ROOT / 'datasets' / 'banc_v626' / 'banc_v626_allneurons_neuron_df.csv'
BANC888_INDEX = (REPO_ROOT / 'neuron_indexes' / 'banc_v888'
                 / 'neuron_index.parquet')
BANC626_INDEX = (REPO_ROOT / 'neuron_indexes' / 'banc_v626'
                 / 'neuron_index.parquet')

pytestmark = pytest.mark.skipif(
    not (MCNS_CSV.exists() and FAFB_CSV.exists()),
    reason='real male-cns v1.0 / FAFB v783 neuron tables not available locally',
)

requires_banc = pytest.mark.skipif(
    not BANC_CSV.exists(),
    reason='real BANC v626 neuron table not available locally',
)

requires_banc_v888_index = pytest.mark.skipif(
    not BANC888_INDEX.exists(),
    reason='real BANC v888 cached neuron index not available locally',
)

requires_banc_release_indexes = pytest.mark.skipif(
    not (BANC626_INDEX.exists() and BANC888_INDEX.exists()),
    reason='real BANC v626/v888 cached neuron indexes not available locally',
)


@pytest.fixture(scope='module')
def mapper():
    m = CrossDatasetTypeMapper(verbose=False, workspace_path=str(REPO_ROOT))
    assert m.load() is True
    return m


@pytest.fixture(scope='module')
def fafb_names():
    """(primary, additional-only) type name sets recomputed from FAFB v783."""
    table = pd.read_csv(FAFB_CSV, usecols=['type', 'additional_type(s)'], low_memory=False)
    primaries = set(table['type'].dropna().astype(str).str.strip()) - {''}
    additional = set()
    for cell in table['additional_type(s)'].dropna().astype(str):
        additional.update(name.strip() for name in cell.split(',') if name.strip())
    return primaries, additional - primaries


@pytest.fixture(scope='module')
def banc_names():
    """(primary, additional-only) type name sets recomputed from BANC v626."""
    table = pd.read_csv(
        BANC_CSV, usecols=['type', 'Alternative Cell Type(s)'], low_memory=False)
    primaries = set(table['type'].dropna().astype(str).str.strip()) - {''}
    additional = set()
    for cell in table['Alternative Cell Type(s)'].dropna().astype(str):
        additional.update(name.strip() for name in cell.split(',') if name.strip())
    return primaries, additional - primaries


# ---------------------------------------------------------------------------
# Load integrity
# ---------------------------------------------------------------------------

def test_flywire_alt_tables_are_indexed(mapper):
    # SLP249 is additional-only in FAFB; APDN3 (its current primary) must
    # NOT be indexed as an additional-only name.
    assert 'SLP249' in mapper._flywire_alt_to_primary.get(FW, {})
    assert 'APDN3' not in mapper._flywire_alt_to_primary[FW]
    # observed: 1,208 FAFB / 642 BANC additional-only names; keep floors
    assert len(mapper._flywire_alt_to_primary[FW]) >= 700
    assert len(mapper._flywire_alt_to_primary[BANC]) >= 500


# ---------------------------------------------------------------------------
# The reported issue: male-cns SLP249 vs FAFB APDN3
# ---------------------------------------------------------------------------

def test_reported_issue_slp249_resolves_to_apdn3(mapper):
    # SLP249 is not a FAFB v783 primary type anymore; the male-cns crosswalk
    # name must resolve to the current FAFB primary name.
    assert mapper.get_mapped_type('SLP249', MCNS, FW) == 'APDN3'

    resolved = mapper.resolve_type_across_datasets(
        'SLP249', [MCNS, FW], source_dataset=MCNS)
    assert resolved == {MCNS: 'SLP249', FW: 'APDN3'}

    assert mapper.get_display_name('SLP249', [MCNS, FW]) == 'SLP249(APDN3)'
    _, hover = mapper.get_display_name_with_dataset_info(
        'SLP249', [MCNS, FW], source_dataset=MCNS)
    assert hover == {'M': 'SLP249', 'F': 'APDN3'}


def test_known_fafb_renames(mapper):
    expected = {
        'SLP249': 'APDN3',
        'CL125': 'APDN3',        # flywireType LMTe01
        'MDN': 'DNp50',
        'IPC': 'm_NSC_DILP',
        'mAL4A': 'mAL',          # flywireType mAL4
        'FS4B': 'FS4A',
        'FS4C': 'FS4A',
        'DNES1': 'l_NSC_unknown',
        'Hugin-RG': 'SEZ_NSC_Hugin',
        'CAPA': 'SEZ_NSC_CAPA',
        'ITP': 'l_NSC_ITP',
        'aSP22': 'DNa12',
        'TuBu04': 'TuBu03',
    }
    for mcns_type, fafb_type in expected.items():
        assert mapper.get_mapped_type(mcns_type, MCNS, FW) == fafb_type, mcns_type


@requires_banc_v888_index
def test_real_banc_label_bridges_use_independent_endpoint_coverage():
    """FAFB/BANC and MCNS/BANC labels retain independent side coverage.

    BANC v888 has six primary ``l-LNv`` rows.  Five carry the curated FAFB
    label and one carries the curated MCNS label.  The optional match columns
    must not reduce the unconstrained FAFB/MCNS endpoint populations or turn
    the type bridge into a bodyId-to-bodyId join.
    """
    fafb = 'flywire_FAFB_v783'
    banc = 'banc_v888'
    mcns = 'male-cns:v1.0'
    indexes = {
        ds: load_cached_neuron_index(ds, enrich=False)
        for ds in (fafb, banc, mcns)
    }
    assert all(indexes.values())

    fafb_linker = [{
        'column': 'fafb_cell_type', 'value': 'l-LNv', 'home': banc,
        'kind': 'linker',
    }]
    forward = pool_bridge_body_ids(
        fafb, banc, fafb_linker, 'l-LNv', 'l-LNv', indexes=indexes)
    assert forward['granularity'] == '8 to 5'
    assert len(forward['source_body_ids']) == 8
    assert len(forward['target_body_ids']) == 5
    assert forward['coverage'] == 'covered 5 of 6 (83.3%)'
    assert forward['source_coverage'] == 'covered 8 of 8 (100.0%)'
    assert forward['target_coverage'] == 'covered 5 of 6 (83.3%)'
    assert 'matched_body_ids' not in forward['per_linker'][0]

    mcns_linker = [{
        'column': 'malecns_cell_type', 'value': 'l-LNv', 'home': banc,
        'kind': 'linker',
    }]
    mcns_forward = pool_bridge_body_ids(
        mcns, banc, mcns_linker, 'l-LNv', 'l-LNv', indexes=indexes)
    # One curated row plus one normalized ``auto:l-LNv`` row is valid
    # evidence, so canonical-token matching covers two BANC bodyIds.  The
    # source side is still the full MCNS population; this is not a bodyId
    # pairing.
    assert mcns_forward['granularity'] == '8 to 2'
    assert mcns_forward['coverage'] == 'covered 2 of 6 (33.3%)'
    assert mcns_forward['source_coverage'] == 'covered 8 of 8 (100.0%)'
    assert mcns_forward['target_coverage'] == 'covered 2 of 6 (33.3%)'
    assert 'matched_body_ids' not in mcns_forward['per_linker'][0]


@requires_banc_release_indexes
def test_real_prioritized_pools_preserve_smp227_branch_evidence(mapper):
    """SMP227 keeps one selected branch and the supported alternatives.

    The MCNS type has six neurons.  Its FAFB bridge evidence is split across
    CB3763 (2 source bodies), CB3766 (1), and CB1449/CB2843 (3 shared source
    bodies).  The selected pool is intentionally narrower than the all-valid
    union; neither view is a bodyId-to-bodyId pairing.
    """
    from comparison.mapping_visualization import build_type_coverage

    indexes = {
        ds: load_cached_neuron_index(ds, enrich=False)
        for ds in (MCNS, FW)
    }
    chains = mapper.get_type_bridges('SMP227', MCNS, FW, max_bridges=0)
    expected = {
        's-CPDN3B': (2, 2, 3, 4, 2),
        's-CPDN3C': (3, 2, 3, 2, 1),
        's-CPDN3D': (3, 2, 3, 2, 1),
    }
    pools = {}
    flows = []
    for target_type, (selected_source, selected_target, all_source,
                      all_target, valid_count) in expected.items():
        pool = resolve_prioritized_bridge_pool(
            MCNS, FW, chains, 'SMP227', target_type, indexes=indexes)
        assert pool['resolution_status'] == 'supported'
        assert pool['selected_chain_rank'] == 1
        assert pool['valid_chain_count'] == valid_count
        assert len(pool['source_body_ids']) == selected_source
        assert len(pool['target_body_ids']) == selected_target
        assert len(pool['all_valid_source_body_ids']) == all_source
        assert len(pool['all_valid_target_body_ids']) == all_target
        assert not set(pool['source_body_ids']) & set(
            pool['target_body_ids'])
        selected_linkers = list(pool['per_linker'])
        assert selected_linkers
        assert all(linker['raw_value'] == linker['canonical_value']
                   for linker in selected_linkers)
        pools[(MCNS, FW, 'SMP227', target_type)] = pool
        flows.append({
            'source_dataset': MCNS,
            'target_dataset': FW,
            'source_type': 'SMP227',
            'foreign_type': target_type,
            'source_count': 6,
            'foreign_count': {
                's-CPDN3B': 25,
                's-CPDN3C': 32,
                's-CPDN3D': 37,
            }[target_type],
            'matched_origin': "type · 'SMP227'",
            'bridges': chains,
        })

    row = build_type_coverage({(MCNS, FW): flows}, pools)['forward'][0]
    assert row['relationship'] == '1-to-N'
    assert row['query_cov_selected'] == '5 of 6 (83.3%)'
    assert row['query_cov_all_valid'] == '6 of 6 (100.0%)'
    assert row['target_cov_selected'] == '6 of 94 (6.4%)'
    assert row['target_cov_all_valid'] == '8 of 94 (8.5%)'
    assert row['query_overlap_selected'] == 3
    assert row['query_overlap_all_valid'] == 3
    assert row['target_overlap_selected'] == 0
    assert row['target_overlap_all_valid'] == 0
    assert 'branch evidence is non-exclusive' in row['coverage_note']
    assert 'all-valid union includes supported alternative bridges' in row[
        'coverage_note']


@requires_banc_release_indexes
def test_real_mevplo2_banc_auto_label_is_counted_with_provenance(mapper):
    """Known ``auto:MeVPLo2`` labels are usable without erasing provenance."""
    indexes = {
        ds: load_cached_neuron_index(ds, enrich=False)
        for ds in (MCNS, BANC, 'banc_v888')
    }
    for banc, expected_target_count in ((BANC, 8), ('banc_v888', 9)):
        chains = mapper.get_type_bridges(
            'MeVPLo2', MCNS, banc, max_bridges=0)
        assert chains and chains[0][-1]['value'] == 'MTe07'
        pool = resolve_prioritized_bridge_pool(
            MCNS, banc, chains, 'MeVPLo2', 'MTe07',
            indexes={MCNS: indexes[MCNS], banc: indexes[banc]})
        assert pool['resolution_status'] == 'supported'
        assert pool['source_pool_size'] == 14
        assert pool['target_pool_size'] == expected_target_count
        linkers = list(pool['per_linker'])
        assert [(l['raw_value'], l['canonical_value']) for l in linkers] == [
            ('auto:MeVPLo2', 'MeVPLo2')]
        assert pool['all_valid_source_pool_size'] == 14
        assert pool['all_valid_target_pool_size'] == expected_target_count


def test_real_banc_conflicting_cell_type_vote_stays_unmapped(mapper):
    """Conflicting BANC labels stay blocked in both mapper/UI paths."""
    expected_targets = {'CB1011', 'CB3252', 'SMP227'}
    for banc in (BANC, 'banc_v888'):
        decision = mapper.get_mapping_decision(
            'CB1011', banc, MCNS)
        assert decision['status'] == 'conflict'
        assert set(decision['target_types']) == expected_targets
        assert mapper.get_mapped_type('CB1011', banc, MCNS) is None
        ui_result = mapped_type_targets(mapper, 'CB1011', banc, MCNS)
        assert ui_result['kind'] == 'conflict'
        assert ui_result['targets'] == []


@requires_banc_release_indexes
def test_real_banc_release_bridge_uses_root_body_relation():
    """BANC release bridges pool from root_626↔root_888, not type totals."""
    from ui.neuron_index import pool_bridge_body_ids

    left = 'banc_v626'
    right = 'banc_v888'
    indexes = {
        left: load_cached_neuron_index(left, enrich=False),
        right: load_cached_neuron_index(right, enrich=False),
    }
    linker = [{
        'column': 'banc_release_crosswalk', 'value': 'L5',
        'home': left, 'kind': 'linker',
    }]
    pool = pool_bridge_body_ids(
        left, right, linker, 'L5', 'L5', indexes=indexes)

    # The relation is not a complete type-total identity: it reaches 1,655
    # v888 L5 bodyIds out of 1,683, while retaining 1,651 v626 roots.
    assert pool['granularity'] == '1651 to 1655'
    assert pool['coverage'] == 'covered 1,655 of 1,683 (98.3%)'
    assert pool['source_coverage'] == 'covered 1,651 of 1,651 (100.0%)'
    assert pool['target_coverage'] == 'covered 1,655 of 1,683 (98.3%)'
    assert 'matched_body_ids' not in pool['per_linker'][0]


def test_ambiguous_renames_become_conflicts(mapper):
    # Each old name is listed under several FAFB primaries (a split), so no
    # single target can be chosen: no mapping, exact conflict recorded.
    expected_splits = {
        'SLP295': {'SLP295a', 'SLP295b'},
        'AOTU008': {'AOTU008a', 'AOTU008b', 'AOTU008c', 'AOTU008d'},
        'SMP206': {'SLP327a', 'SLP327b'},
        'SMP426': {'SLP402a', 'SLP402b', 'SLP402c'},
    }
    conflicts = mapper.get_1_to_n_conflicts()
    for mcns_type, targets in expected_splits.items():
        assert mapper.get_mapped_type(mcns_type, MCNS, FW) is None, mcns_type
        assert any(
            c.source_type == mcns_type
            and c.target_dataset == FW
            and c.target_types == targets
            for c in conflicts
        ), mcns_type


def test_comma_separated_crosswalk_cell_expands_to_conflict(mapper):
    # male-cns 'VS' lists eight flywire types in one cell -> 1-to-N conflict
    vs_targets = {f'VS{i}' for i in range(1, 9)}
    assert mapper.get_mapped_type('VS', MCNS, FW) is None
    assert any(
        c.source_type == 'VS'
        and c.target_dataset == FW
        and c.target_types == vs_targets
        for c in mapper.get_1_to_n_conflicts()
    )
    # The BANC overlay has its own MCNS-side labels.  They independently
    # expose a split (the BANC namespace omits VS6), but still must not make
    # one canonical BANC target up from the branch evidence.
    assert mapper.get_mapped_type('VS', MCNS, BANC) is None
    banc_decision = mapper.get_mapping_decision('VS', MCNS, BANC)
    assert banc_decision['status'] == 'valid_split_evidence'
    assert set(banc_decision['target_types']) == {
        'VS1', 'VS2', 'VS3', 'VS4', 'VS5', 'VS7', 'VS8'}


def test_comma_separated_hemibrain_cell_expands_to_conflict(mapper):
    # male-cns IPC: hemibrainType 'PI1,PI2,PI3' -> no single hemibrain target
    assert mapper.get_mapped_type('IPC', MCNS, HB) is None
    assert any(
        c.source_type == 'IPC'
        and c.target_dataset == HB
        and c.target_types == {'PI1', 'PI2', 'PI3'}
        for c in mapper.get_1_to_n_conflicts()
    )


# ---------------------------------------------------------------------------
# Global invariants over every mapping
# ---------------------------------------------------------------------------

def _iter_forward_targets(mapper, fw_key):
    for targets in mapper._type_mappings[MCNS].values():
        value = targets.get(fw_key)
        if value:
            yield value


def test_no_fafb_target_is_a_stale_name(mapper, fafb_names):
    # A mapping to an additional-only name is exactly the bug this fix
    # addresses: targets must be current primaries (or unknown names).
    _, additional_only = fafb_names
    stale = {v for v in _iter_forward_targets(mapper, FW) if v in additional_only}
    assert not stale


@requires_banc
def test_no_banc_target_is_a_stale_name(mapper, banc_names):
    _, additional_only = banc_names
    stale = {v for v in _iter_forward_targets(mapper, BANC) if v in additional_only}
    assert not stale


# ---------------------------------------------------------------------------
# §stale claims (2026-09-12): crosswalk names the target dataset does not
# actually carry (e.g. male-cns CB4091 flywireType CB4091, absent from FAFB
# v783) stay resolvable curated claims but never enter the target namespace
# nor license a mapped equivalence.
# ---------------------------------------------------------------------------

def test_stale_fafb_claims_are_registered_not_injected(mapper):
    claims = mapper._stale_type_claims.get((MCNS, FW), {})
    # the observed case, plus scale (hundreds of stale crosswalk names)
    assert claims.get('CB4091') == {'CB4091'}
    assert len(claims) >= 100
    # stale names never pollute the target namespace
    assert 'CB4091' not in mapper._dataset_types[FW]
    assert 'CB4091' not in mapper._flywire_primaries[FW]
    claimed_names = {name for names in claims.values() for name in names}
    assert not claimed_names & mapper._dataset_types[FW]
    # namespace ground truth answers existence post-load
    assert mapper.type_exists(FW, 'APDN3') is True
    assert mapper.type_exists(FW, 'CB4091') is False


def test_stale_claim_stays_resolvable_for_evidence_surfaces(mapper):
    # the curated claim keeps its get_mapped_type answer so the alias
    # viewer can show the crosswalk evidence — it is the resolver that
    # refuses to promote it to a mapping
    assert mapper.get_mapped_type('CB4091', MCNS, FW) == 'CB4091'


def test_stale_claim_resolves_as_claimed_not_mapped(mapper):
    res = resolve_valid_targets(mapper, 'CB4091', MCNS, FW)
    assert res.status == STATUS_CLAIMED
    assert res.target_types == ('CB4091',)
    assert res.equivalence_key is None
    # the UI adapter forwards the status so panels can render the claim
    ann = mapped_type_targets(mapper, 'CB4091', MCNS, FW)
    assert ann is not None
    assert ann.get('status') == STATUS_CLAIMED
    assert ann.get('targets') == ['CB4091']


def test_real_rename_still_resolves_as_mapped(mapper):
    # positive control: SLP249 -> APDN3 is a fulfilled curated rename
    res = resolve_valid_targets(mapper, 'SLP249', MCNS, FW)
    assert res.status == STATUS_MAPPED
    assert res.target_types == ('APDN3',)
    assert res.equivalence_key == 'APDN3'
    assert mapper.type_exists(FW, 'APDN3') is True


def test_every_fafb_target_is_primary_or_registered_claim(mapper, fafb_names):
    primaries, _ = fafb_names
    claims = mapper._stale_type_claims.get((MCNS, FW), {})
    claimed_names = {name for names in claims.values() for name in names}
    for value in _iter_forward_targets(mapper, FW):
        assert value in primaries or value in claimed_names


def test_rename_scale(mapper, fafb_names):
    # Observed: 89 male-cns types resolve to a different FAFB primary name.
    primaries, _ = fafb_names
    raw_by_type = (
        mapper._neuron_df.groupby('type')['flywireType']
        .apply(lambda s: {
            name.strip()
            for cell in s.dropna().astype(str)
            for name in cell.split(',') if name.strip()
        })
        .to_dict()
    )
    renamed = {
        mcns_type
        for mcns_type, targets in mapper._type_mappings[MCNS].items()
        if (v := targets.get(FW))
        and v in primaries
        and v not in raw_by_type.get(mcns_type, set())
    }
    assert len(renamed) >= 50
    assert 'SLP249' in renamed


# ---------------------------------------------------------------------------
# FAFB / BANC namespace independence
# ---------------------------------------------------------------------------

@requires_banc
def test_fafb_and_banc_namespaces_resolve_independently(mapper):
    # FAFB renamed MDN -> DNp50; BANC still uses MDN as its primary type.
    assert mapper.get_mapped_type('MDN', MCNS, FW) == 'DNp50'
    assert mapper.get_mapped_type('MDN', MCNS, BANC) == 'MDN'
    # ...and the reverse: FAFB keeps DNge036; the bucket-curated BANC
    # typing also uses DNge036 as its primary now (DNfl042 survives as an
    # alt alias on the same cells).
    assert mapper.get_mapped_type('DNge036', MCNS, FW) == 'DNge036'
    assert mapper.get_mapped_type('DNge036', MCNS, BANC) == 'DNge036'
    # The namespaces stay connected through the shared male-cns crosswalk.
    assert mapper.get_mapped_type('DNp50', FW, BANC) == 'MDN'
    assert mapper.get_mapped_type('MDN', BANC, FW) == 'DNp50'


# ---------------------------------------------------------------------------
# Reverse direction / canonicalization of renamed types
# ---------------------------------------------------------------------------

def test_reverse_of_merged_renames_is_flagged_n_to_1(mapper):
    # FAFB APDN3 receives SLP249, CL125 (LMTe01), SLP250 (LTe71) and PLP080;
    # aggregation is refused and flagged instead.
    assert mapper.get_mapped_type('APDN3', FW, MCNS) is None
    assert mapper.is_n_to_1_type('APDN3', FW)
    conflict = next(
        c for c in mapper.get_n_to_1_conflicts()
        if c.source_dataset == FW and c.source_type == 'APDN3'
    )
    assert conflict.target_types == {'SLP249', 'CL125', 'SLP250', 'PLP080'}

    # FS4A collects FS4B and FS4C (plus FS4A itself).
    assert mapper.is_n_to_1_type('FS4A', FW)
    fs4a = next(
        c for c in mapper.get_n_to_1_conflicts()
        if c.source_dataset == FW and c.source_type == 'FS4A'
    )
    assert {'FS4B', 'FS4C'} <= fs4a.target_types


def test_unique_reverse_rename_canonicalizes_partners(mapper):
    # DNp50 (FAFB) is only fed by male-cns MDN, so partner standardization
    # can safely canonicalize it.
    assert mapper.get_canonical_type('DNp50', FW) == 'MDN'
    assert not mapper.is_n_to_1_type('DNp50', FW)
    standardized = mapper.standardize_partner_types({'DNp50': 2.0}, FW)
    assert standardized == {'MDN': pytest.approx(2.0)}


def test_source_detection_uses_resolved_names(mapper):
    # APDN3 is registered as a FAFB-namespace type; SLP249 only as male-cns.
    assert mapper._detect_type_source('APDN3') == FW
    assert mapper._detect_type_source('SLP249') == MCNS
    assert 'APDN3' in mapper._dataset_types[FW]
    assert 'SLP249' not in mapper._dataset_types[FW]
    assert 'SLP249' in mapper._dataset_types[MCNS]


def test_rename_round_trip_consistency(mapper):
    # For every mapped male-cns type, canonicalizing its FAFB target either
    # returns the target itself (merged/N-to-1) or the source type (unique).
    for mcns_type, targets in mapper._type_mappings[MCNS].items():
        value = targets.get(FW)
        if not value:
            continue
        canonical = mapper.get_canonical_type(value, FW)
        assert canonical in (value, mcns_type), (mcns_type, value, canonical)


# ---------------------------------------------------------------------------
# Exports and global wiring
# ---------------------------------------------------------------------------

def test_export_mapping_includes_renames(mapper, tmp_path):
    out = tmp_path / 'mapping.csv'
    mapper.export_mapping(str(out), filter_types={'SLP249'})
    df = pd.read_csv(out)
    row = df[df[MCNS] == 'SLP249']
    assert not row.empty
    assert row.iloc[0][FW] == 'APDN3'


def test_global_type_mapper_singleton_resolves_renames(mapper, monkeypatch):
    from comparison import cross_dataset_type_mapper as mapper_module

    monkeypatch.setattr(mapper_module, '_global_type_mapper', mapper)
    assert get_type_mapper() is mapper
    assert get_type_mapper().get_mapped_type('SLP249', MCNS, FW) == 'APDN3'


# ---------------------------------------------------------------------------
# User warning notes (expanded / N-to-1 / 1-to-N)
# ---------------------------------------------------------------------------

def test_build_user_warning_notes_slp249_rename(mapper):
    notes = mapper.build_user_warning_notes(['SLP249'], [MCNS, FW])
    # expanded to the current FAFB primary name...
    assert any(
        'expanded' in n and "'SLP249'" in n and "'APDN3'" in n for n in notes)
    # ...and flagged N-to-1: FAFB APDN3 also collects CL125/SLP250/PLP080,
    # so the reverse (FAFB -> male-cns) aggregation is refused.
    n_to_1 = [n for n in notes if 'N-to-1' in n]
    assert n_to_1 and all(t in n_to_1[0] for t in ('CL125', 'PLP080', 'SLP250'))
    assert sum('double check' in n.lower() for n in notes) == 1


def test_build_user_warning_notes_apdn3_n_to_1(mapper):
    # FAFB APDN3 aggregates four male-cns types: flagged, never merged
    notes = mapper.build_user_warning_notes(['APDN3'], [FW, MCNS])
    n_to_1 = [n for n in notes if 'N-to-1' in n]
    assert n_to_1, notes
    assert all(t in n_to_1[0] for t in ('CL125', 'PLP080', 'SLP249', 'SLP250'))
    assert any('double check' in n.lower() for n in notes)


def test_build_user_warning_notes_known_rename_batch(mapper):
    # A queried batch mixes renames (expanded) and clean types; every
    # expected rename is reported and the advice line is present once.
    notes = mapper.build_user_warning_notes(
        ['SLP249', 'MDN', 'IPC', 'aMe12'], [MCNS, FW])
    joined = '\n'.join(notes)
    for old, new in (('SLP249', 'APDN3'), ('MDN', 'DNp50'), ('IPC', 'm_NSC_DILP')):
        assert f"'{old}'" in joined and f"'{new}'" in joined
    assert sum('double check' in n.lower() for n in notes) == 1


# ---------------------------------------------------------------------------
# Alias candidates (expanded viewer search) on real data
# ---------------------------------------------------------------------------

def _candidate(res, dataset, name):
    for cand in res[dataset]['candidates']:
        if cand['name'] == name:
            return cand
    return None


def test_get_alias_candidates_slp249_real(mapper):
    res = mapper.get_alias_candidates('SLP249', [MCNS, FW])
    assert res[MCNS]['outcome'] == 'matched'
    assert _candidate(res, MCNS, 'SLP249')['kind'] == 'same name'
    assert res[FW]['outcome'] == 'matched'
    renamed = _candidate(res, FW, 'APDN3')
    assert renamed['kind'] == 'renamed'
    # orthogonal aggregation annotation: FAFB APDN3 also covers the other
    # three male-cns types that were renamed into it.
    assert renamed['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']


def test_get_alias_candidates_apdn3_real(mapper):
    res = mapper.get_alias_candidates(
        'APDN3', [MCNS, FW, BANC])
    # native in both local releases, with the aggregation annotation on
    # the FAFB candidate (BANC keeps its own 1:1 annotation-free identity).
    fafb = _candidate(res, FW, 'APDN3')
    assert fafb['kind'] == 'same name'
    assert fafb['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']
    banc = _candidate(res, BANC, 'APDN3')
    assert banc['kind'] == 'same name'
    assert banc['aggregates'] is None
    # male-cns side: the refused reverse aggregation exposes the group.
    mcns = res[MCNS]
    assert mcns['outcome'] == 'matched'
    assert [c['name'] for c in mcns['candidates']] == [
        'CL125', 'PLP080', 'SLP249', 'SLP250']
    assert all(c['kind'] == 'one of N' for c in mcns['candidates'])


def test_get_alias_candidates_namespace_independence_real(mapper):
    res = mapper.get_alias_candidates('MDN', [MCNS, FW, BANC])
    assert _candidate(res, MCNS, 'MDN')['kind'] == 'same name'
    renamed = _candidate(res, FW, 'DNp50')
    assert renamed['kind'] == 'renamed'
    assert renamed['aggregates'] is None  # unique reverse (only male-cns MDN)
    assert _candidate(res, BANC, 'MDN')['kind'] == 'same name'

    # Bucket-curated BANC typing uses DNge036 as its primary now
    # (DNfl042 is an alt alias on the same cells), so the query resolves
    # as a same name in every namespace.
    res_d = mapper.get_alias_candidates('DNge036', [MCNS, FW, BANC])
    assert _candidate(res_d, FW, 'DNge036')['kind'] == 'same name'
    assert _candidate(res_d, BANC, 'DNge036')['kind'] == 'same name'


def test_get_alias_candidates_vs_splits_real(mapper):
    # male-cns 'VS' lists eight FAFB types in one crosswalk cell: the
    # candidates are the split targets, never a joined name.
    res = mapper.get_alias_candidates('VS', [MCNS, FW])
    fw_names = [c['name'] for c in res[FW]['candidates']]
    assert fw_names == [f'VS{i}' for i in range(1, 9)]
    assert all(c['kind'] == 'splits into' for c in res[FW]['candidates'])
    assert res[MCNS]['candidates'] == [
        {'name': 'VS', 'kind': 'same name', 'aggregates': None},
    ]


def test_get_alias_candidates_unknown_real(mapper):
    res = mapper.get_alias_candidates('NoRealType123', [MCNS, FW, BANC])
    assert all(
        res[ds]['outcome'] == 'no counterpart known'
        for ds in (MCNS, FW, BANC)
    )


def test_get_alias_candidates_kinds_exclusive_real(mapper):
    # property over a broad real-type batch: one valid kind per candidate,
    # unique names per dataset, aggregates never present on 'one of N' /
    # 'splits into' candidates (it is a candidate-side annotation).
    queries = [
        'SLP249', 'APDN3', 'MDN', 'DNp50', 'IPC', 'm_NSC_DILP', 'LPN',
        'MTe07', 'MeVPLo2', 'aMe12', 'VS', 'DNge036', 'DNfl042', 'TuBu03',
        'TuBu04', 'FS4A', 'FS4B', 'CL125', 'SLP250', 'PLP080', 'aMe12_L',
    ]
    for query in queries:
        res = mapper.get_alias_candidates(query, [MCNS, FW, BANC])
        for dataset, outcome in res.items():
            names = [c['name'] for c in outcome['candidates']]
            assert len(names) == len(set(names)), (query, dataset, names)
            for cand in outcome['candidates']:
                assert cand['kind'] in CrossDatasetTypeMapper.ALIAS_KINDS, (
                    query, dataset, cand)
                if cand['kind'] in ('one of N', 'splits into'):
                    assert cand['aggregates'] is None, (query, dataset, cand)


def test_alias_counts_match_expected_real(mapper):
    # the annotation lists real sibling types: SLP249 itself stays findable
    # in male-cns while FAFB APDN3 counts its four neurons.
    res = mapper.get_alias_candidates('SLP249', [MCNS, FW])
    assert 'SLP249' in res[MCNS]['candidates'][0]['name']
    assert res[FW]['candidates'][0]['name'] == 'APDN3'


def test_get_alias_candidates_combined_cell_name_real(mapper):
    # male-cns 'vDeltaB' is native here; in FAFB the old name 'vDeltaB' was
    # split into vDelta/vDeltaA (it now only survives inside combined
    # additional_type(s) cells) -> 'splits into' with both targets.
    res = mapper.get_alias_candidates('vDeltaB', [MCNS, FW])
    assert _candidate(res, MCNS, 'vDeltaB')['kind'] == 'same name'
    split_names = [c['name'] for c in res[FW]['candidates']]
    assert split_names == ['vDelta', 'vDeltaA']
    assert all(c['kind'] == 'splits into' for c in res[FW]['candidates'])
    assert all(c['aggregates'] is None for c in res[FW]['candidates'])

    # a literal comma-joined query is not a name: honest outcome everywhere.
    joined = mapper.get_alias_candidates(
        'vDeltaB, vDeltaC', [MCNS, FW])
    assert all(
        joined[ds]['outcome'] == 'no counterpart known'
        for ds in (MCNS, FW)
    )


# =============================================================================
# Round 1 of the bodyId-level bridge plan (_plan/plan-type-mapping-bodyid-bridge.md):
# standardized linker extraction, the at-most-two-linker theorem, and the
# BANC type-name routing through FAFB annotations into the male-cns crosswalk.
# =============================================================================

def test_standardize_bridge_real_anchors(mapper):
    """CL125 / PLP080 / SLP250 chains standardize to the registry linkers."""
    from comparison.cross_dataset_type_mapper import standardize_bridge

    # CL125: the full two-linker standard (flywireType -> additional_type(s))
    chains = mapper.get_type_bridges('CL125', MCNS, FW)
    two_linker = next(
        chain for chain in chains
        if [('flywireType', 'LMTe01'), ('additional_type(s)', 'LMTe01')]
        == [(l['column'], l['value'])
            for l in standardize_bridge(chain, MCNS, FW)
            if l['kind'] == 'linker'])
    linkers = standardize_bridge(two_linker, MCNS, FW)
    linker_only = [l for l in linkers if l['kind'] == 'linker']
    assert sorted((l['column'], l['value']) for l in linker_only) == [
        ('additional_type(s)', 'LMTe01'), ('flywireType', 'LMTe01')]
    assert not any(l['indirect'] for l in linkers)

    # PLP080: same two-linker shape with its own value — the literal
    # PLP080 -- PLP080 (additional_type(s)) -- APDN3 path (located by
    # signature: chain order varies with the pruned walk)
    chains_plp = mapper.get_type_bridges('PLP080', MCNS, FW)
    chain = next(
        c for c in chains_plp
        if [('flywireType', 'PLP080'), ('additional_type(s)', 'PLP080')]
        == [(l['column'], l['value'])
            for l in standardize_bridge(c, MCNS, FW)
            if l['kind'] == 'linker'])
    linker_only = [l for l in standardize_bridge(chain, MCNS, FW)
                   if l['kind'] == 'linker']
    assert [(l['column'], l['value']) for l in linker_only] == [
        ('flywireType', 'PLP080'), ('additional_type(s)', 'PLP080')]

    # SLP250: the preferred chain is the registry two-linker standard
    chain = preferred_bridge_chain(
        mapper.get_type_bridges('SLP250', MCNS, FW), MCNS, FW)
    linker_only = [l for l in standardize_bridge(chain, MCNS, FW)
                   if l['kind'] == 'linker']
    # the annotation linker's value is the CELL entry on APDN3 rows:
    # 'LTe71' (SLP250's own crosswalk name echoed in the annotation)
    assert [(l['column'], l['value']) for l in linker_only] == [
        ('flywireType', 'LTe71'), ('additional_type(s)', 'LTe71')]

    # MDN -> BANC uses BANC's curated MCNS label column.
    chain = mapper.get_type_bridges('MDN', MCNS, BANC)[0]
    linkers = [l for l in standardize_bridge(chain, MCNS, BANC)
               if l['kind'] == 'linker']
    assert [(l['column'], l['value']) for l in linkers] == [
        ('malecns_cell_type', 'MDN')]
    assert all(not l['indirect'] for l in linkers)

    # endpoints are always the two datasets' type identities
    assert chain[0]['column'] == 'type' and chain[0]['dataset'] == MCNS
    assert chain[-1]['dataset'] == BANC


def test_banc_type_names_route_through_direct_curated_labels(mapper):
    """A BANC type with an MCNS label maps directly into MCNS."""
    source_type = next(
        source_type
        for (source_key, source_type), edges
        in mapper._banc_label_edges.items()
        if source_key == BANC
        and any(edge[0] == MCNS and edge[2] == 'malecns_cell_type'
                for edge in edges)
    )
    chains = mapper.get_type_bridges(source_type, BANC, MCNS)
    assert chains
    assert all(chain[-1]['dataset'] == MCNS for chain in chains)
    assert any(
        any(hop['column'] == 'malecns_cell_type' for hop in chain[1:])
        for chain in chains)


def test_two_linker_cap_over_sample_matrix(mapper):
    """Every chain standardizes to at most two linker nodes (the theorem)."""
    pairs = [
        # registry pairs: the two-linker standard holds strictly
        (MCNS, FW), (MCNS, HB),
        # hub pairs (BANC routes through FAFB annotations): a third
        # annotation hop is legitimate and every linker is flagged indirect
        (MCNS, BANC), (BANC, MCNS), (FW, MCNS), (BANC, FW), (FW, BANC),
    ]
    types = ['MDN', 'CL125', 'APDN3', 'SLP249', 'aMe12', 'TmY9q', 'Dn3',
             'vDeltaB', 'LC10', 'DNp50']
    checked = 0
    for src, tgt in pairs:
        for type_name in types:
            for chain in mapper.get_type_bridges(type_name, src, tgt):
                linkers = standardize_bridge(chain, src, tgt)
                # The two-linker standard binds the DIRECT (registry) linkers;
                # deeper FAFB rename chains (e.g. vDeltaB -> vDeltaA ->
                # vDeltaL into hemibrain) are legitimate but every extra
                # linker is flagged indirect.
                direct = [l for l in linkers
                          if l['kind'] == 'linker' and not l['indirect']]
                assert len(direct) <= 2, (src, tgt, type_name, chain)
                checked += 1
    assert checked > 20, f'sample too small: {checked} chains'


def test_plp080_renamed_resolution_regression(mapper):
    """The FAFB-side search for PLP080 (additional-only there) resolves to
    APDN3 with the N-to-1 aggregates — preserved by the v2 changes."""
    res = mapper.get_alias_candidates('PLP080', [FW])
    info = res[FW]
    assert info['outcome'] == 'matched'
    assert [c['name'] for c in info['candidates']] == ['APDN3']
    assert info['candidates'][0]['kind'] == 'renamed'
    assert info['candidates'][0]['aggregates'] == [
        'CL125', 'PLP080', 'SLP249', 'SLP250']


def _entry(matches, dataset):
    for entry in matches:
        if entry['dataset'] == dataset:
            return entry
    return None


def test_bridge_linker_text_values_and_hub_note(mapper):
    """bridge_linker_text: values are included and deduplicated.

    MCNS→BANC now uses BANC's curated label columns directly, so the old
    FAFB annotation-hub detour must not appear in this path.
    """
    from comparison.cross_dataset_type_mapper import bridge_linker_text

    chains = mapper.get_type_bridges('CL125', MCNS, FW)
    info = bridge_linker_text(chains, MCNS, FW, 'APDN3')
    # The registry order keeps the crosswalk linker before the annotation
    # linker; values within one column are deterministic lexical order.
    assert info['text'] == (
        "flywireType 'LMTe01' + additional_type(s) 'CL125' "
        "+ additional_type(s) 'LMTe01'")
    assert [e['value'] for e in info['entries']] == [
        'LMTe01', 'CL125', 'LMTe01']
    assert not any(e['indirect'] for e in info['entries'])

    # The former CL125 -> LTe71 (BANC) hub chain is pure transitivity noise;
    # MCNS→BANC is direct-only under the curated BANC label policy.
    hub_chains = mapper.get_type_bridges('CL125', MCNS, BANC)
    assert not [c for c in hub_chains if c and c[-1]['value'] == 'LTe71']
    indirect_found = any(
        any(l['indirect'] for l in standardize_bridge(
            chain, MCNS, BANC))
        for type_name in ('l-LNv', 'DN1pA', 'CB3508')
        for chain in mapper.get_type_bridges(type_name, MCNS, BANC))
    assert not indirect_found


def test_mapping_sankey_vispath_backend(mapper):
    """The sankey uses the vispath backend: layered linker bands with
    pooled-count ribbons and the shared interactive control panel
    (user-adjustable node/edge colors)."""
    from comparison.mapping_visualization import (
        build_mapping_flows,
        build_mapping_sankey_paths,
        render_mapping_sankey_html,
    )
    from ui.neuron_index import (
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    assert entry
    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    flows = build_mapping_flows(
        [entry], MCNS,
        source_counts=count_types_in_index_map(index, entry))
    assert flows
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        pool = pool_bridge_body_ids(
            MCNS, FW, standardize_bridge(chain, MCNS, FW),
            flow['source_type'], flow['foreign_type'],
            indexes={MCNS: index, FW: foreign_index})
        pools[(flow['source_type'], flow['foreign_type'])] = pool

    rows = build_mapping_sankey_paths(flows, pools=pools)
    # pooled ribbons: each drawn chain contributes its pooled count per
    # hop — 4-bodyId bridges (CL125, SLP249) and 2-bodyId bridges
    # (SLP250, PLP080).  With direction-symmetric registry scoping the
    # composed BANC-landing twins are gone from MCNS~FAFB bridges: each
    # flow renders its two legitimate chains (the direct annotation row
    # and the 3-band crosswalk row), restoring the baseline 20 ribbons.
    assert sorted(w for _names, ws in rows for w in ws) == (
        [2] * 10 + [4] * 10)
    # fan-in regression: ribbons are the PAIR's pooled granularity
    # (min of the two sides), constant along the path — a shared target
    # must never flatten fan-in edges to one identical weight
    fan_in = [
        {'source_dataset': MCNS, 'target_dataset': FW, 'source_type': s,
         'foreign_type': 'APDN3', 'source_count': c, 'foreign_count': 12,
         'matched_origin': f"type · '{s}'", 'bridges': [[
             {'dataset': MCNS, 'column': 'type', 'value': s},
             {'dataset': FW, 'column': 'type', 'value': 'APDN3'}]]}
        for s, c in (('CL125', 8), ('PLP080', 2), ('SLP249', 4))]
    fan_rows = build_mapping_sankey_paths(fan_in)
    assert sorted(w for _n, ws in fan_rows for w in ws) == [2, 4, 8]
    # linker bands carry the column tag and the 4-char dataset codes
    assert any('LMTe01 · MCNS [flywireType]' in n
               for names, _ws in rows for n in names)
    assert all(any('· FAFB' in n for n in names) for names, _ws in rows)

    html = render_mapping_sankey_html(flows, pools=pools)
    assert html and 'sankey' in html.lower()
    # the shared control panel: adjustable node/edge colors
    assert 'Node Colors' in html and 'Edge Color' in html
    assert 'APDN3' in html
    # the flow cap trims rows (2 chains per flow maximum)
    assert len(build_mapping_sankey_paths(
        flows, pools=pools, max_flows=2)) <= 4


def count_types_in_index_map(index, entry):
    from ui.neuron_index import count_types_in_index
    return count_types_in_index(index, entry.get('mapped_type_names', []))


def test_analyzer_mapping_export_imports():
    """The Cross-Dataset run's mapping_sankey export imports cleanly
    (regression: the vispath-backend switch must not break the
    analyzer)."""
    import importlib

    module = importlib.import_module('comparison.mapping_visualization')
    assert hasattr(module, 'render_mapping_sankey_html')
    source = Path(  # the analyzer's lazy import names must all resolve
        Path(__file__).resolve().parents[2]
        / 'src' / 'comparison' / 'comparison_analyzer.py').read_text(
        encoding='utf-8')
    segment = source[source.index('render_mapping_sankey_html'):]
    assert 'write_mapping_network_html' in segment


def test_mapping_type_sankey_and_in_memory_renderers(mapper):
    """Type-level sankey bands (§9C.6) + the in-memory vispath renderers
    (no repository writes — temp-dir render, HTML string back)."""
    from comparison.mapping_visualization import (
        build_mapping_flows,
        build_mapping_sankey_paths,
        render_bridge_linker_html,
        render_mapping_network_html,
        render_mapping_sankey_html,
    )
    from ui.neuron_index import (
        count_types_in_index,
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    assert entry
    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    flows = build_mapping_flows(
        [entry], MCNS, source_counts=count_types_in_index(
            index, entry.get('mapped_type_names', [])))
    assert flows
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        pools[(flow['source_type'], flow['foreign_type'])] = (
            pool_bridge_body_ids(
                MCNS, FW, standardize_bridge(chain, MCNS, FW),
                flow['source_type'], flow['foreign_type'],
                indexes={MCNS: index, FW: foreign_index}))

    rows = build_mapping_sankey_paths(flows, pools=pools, variant="type")
    # two bands only: the 4 mapped sources -> the single FAFB target,
    # one aggregated hop per pair with the pooled granularity
    assert len(rows) == len(flows)
    assert all(len(names) == 2 for names, _ws in rows)
    assert sorted(w for _names, ws in rows for w in ws) == sorted(
        min(len(p['source_body_ids']), len(p['target_body_ids']))
        or 1 for p in pools.values())
    # the cap trims rows to one per flow
    assert len(build_mapping_sankey_paths(
        flows, pools=pools, variant="type", max_flows=1)) == 1
    html = render_mapping_sankey_html(
        flows, pools=pools, variant="type")
    assert html and 'Node Colors' in html

    # in-memory renderers: strings, correct renderer, no repo writes
    net_html = render_mapping_network_html(flows, pools=pools)
    assert net_html and 'cytoscape' in net_html.lower()
    # node hovers carry the pooled bodyId count next to the index
    # neuron count (user report)
    assert 'bodyIds' in net_html
    linker_html = render_bridge_linker_html(
        flows, source_dataset=MCNS, target_dataset=FW, pools=pools)
    assert linker_html and 'cytoscape' in linker_html.lower()
    assert 'flywireType' in linker_html
    assert 'pool ' in linker_html and 'bodyIds' in linker_html
    # per-dataset node groups (§13): every node is grouped and colored by
    # its dataset — quick-action buttons + color/opacity dropdown + group
    # ops cover the DATASETS (not the structural roles, which stay on the
    # hovers); the footer legend carries one color chip per dataset and
    # the dataset-code chips gain the matching color dot
    for html in (net_html, linker_html):
        assert "selectGroup('MCNS')" in html
        assert "selectGroup('FAFB')" in html
        assert '<option value="MCNS">MCNS Nodes</option>' in html
        assert 'const extraNodeGroups' in html
        assert 'legend-color' in html
        assert 'border-radius:50%' in html  # dataset-code color dots
        assert "selectGroup('linker')" not in html
        assert "selectGroup('entry')" not in html
    # the mapping preset is programmatic-only: listed (selected) ONLY in
    # the linker-path document, never in the dagre-rendered network
    assert 'value="mapping" selected' in linker_html
    assert 'value="mapping"' not in net_html
    # role survives on the hover via the node's 'role' data field
    assert "data.role" in net_html


def test_panel_chip_modes_and_origin_seeded_flows():
    """§12: chip resolution under the standard filter modes + the
    origin-seeded mapping (condition 1: the query lives where it
    matched) + the mirror dedupe."""
    from ui.neuron_index import (
        count_types_in_index,
        load_cached_neuron_index,
        resolve_type_matches,
    )
    from comparison.mapping_visualization import (
        dedupe_mirrored_pairs,
        origin_seeded_flows,
    )

    datasets = [MCNS, FW]
    indexes = {ds: load_cached_neuron_index(ds) for ds in datasets}

    # exact: the reported explosion — 'aMe2' resolves to the single
    # type in male-cns only (no aMe2_adpn/MeVPaMe2-style substring hits)
    resolved = resolve_type_matches(['aMe2'], 'exact', datasets, indexes)
    assert resolved['origins'] == {MCNS: ['aMe2']}
    assert not resolved['fallback_chips']

    # regex: the family in every dataset that has it (legacy 'aMe.*')
    resolved = resolve_type_matches(['aMe.*'], 'regex', datasets, indexes)
    assert len(resolved['origins'].get(MCNS, [])) > 5
    assert len(resolved['origins'].get(FW, [])) > 5

    # literal modes
    assert resolve_type_matches(
        ['aMe'], 'startswith', datasets, indexes)['origins']
    assert resolve_type_matches(
        ['e2'], 'contains', datasets, indexes)['origins']
    assert resolve_type_matches(
        ['2'], 'endswith', datasets, indexes)['origins']

    # zero-hit under the mode -> fallback chip + visible note
    resolved = resolve_type_matches(
        ['circadian'], 'exact', datasets, indexes)
    assert resolved['fallback_chips'] == ['circadian']
    assert resolved['notes'] and 'labels' in resolved['notes'][0]
    # invalid regex falls back too
    resolved = resolve_type_matches(['aMe('], 'regex', datasets, indexes)
    assert resolved['fallback_chips'] == ['aMe(']

    # condition 1: exact type in FAFB -> flows FROM FAFB to the others
    flows_m = origin_seeded_flows(FW, ['APDN3'], MCNS)
    flows_b = origin_seeded_flows(FW, ['APDN3'], BANC)
    assert flows_m and flows_b
    assert all(f['source_dataset'] == FW for f in flows_m + flows_b)
    assert all(f['origin_dataset'] == FW for f in flows_m + flows_b)
    assert all(f['origin_column'] == 'type' for f in flows_m + flows_b)
    assert all(f['origin_type'] == f['source_type']
               for f in flows_m + flows_b)
    assert {'CL125', 'PLP080', 'SLP249', 'SLP250'} <= {
        f['foreign_type'] for f in flows_m}
    assert any(f['foreign_type'] == 'APDN3' and f['source_type'] == 'APDN3'
               for f in flows_b)  # the same-name route to the BANC
    # supplied counts flow through to the flow dicts
    counts = count_types_in_index(indexes[FW], ['APDN3'])
    seeded = origin_seeded_flows(FW, ['APDN3'], MCNS,
                                 source_counts=counts)
    assert seeded and all(
        f['source_count'] == counts.get('APDN3', 0) for f in seeded)
    label_seeded = origin_seeded_flows(
        FW, ['APDN3'], MCNS,
        matched_origins={
            'APDN3': [{'column': 'cell_type',
                       'value': 'circadian_clock'}]})
    assert label_seeded and all(
        (f['origin_dataset'], f['origin_column'], f['origin_value']) ==
        (FW, 'cell_type', 'circadian_clock') for f in label_seeded)

    # mirror dedupe (type-pair granularity): the two directions of one
    # equivalence collapse to ONE flow — the origin-source direction
    # wins; a stripped same-name reverse loses to the linker-bearing
    # forward
    flow0 = flows_m[0]
    reverse = [dict(flow0, source_dataset=MCNS, target_dataset=FW,
                    source_type=flow0['foreign_type'],
                    foreign_type=flow0['source_type'], bridges=[])]
    kept = dedupe_mirrored_pairs(
        {(FW, MCNS): flows_m, (MCNS, FW): reverse}, [FW])
    assert set(kept) == {(FW, MCNS)}
    kept = dedupe_mirrored_pairs(
        {(FW, MCNS): flows_m, (MCNS, FW): reverse}, [FW, MCNS])
    assert set(kept) == {(FW, MCNS)}
    # asymmetric equivalences SURVIVE: a type pair only the reverse
    # direction found is kept, not swallowed by the collapsed direction
    # (the FAFB↔MCNS report: crosswalk evidence reads male-cns → FAFB,
    # yet the equivalence must render from the other side too)
    extra_rev = [dict(reverse[0], source_type='Zz', foreign_type='Yy')]
    kept = dedupe_mirrored_pairs(
        {(FW, MCNS): flows_m, (MCNS, FW): reverse + extra_rev}, [FW])
    pairs = {(f['source_type'], f['foreign_type'])
             for fl in kept.values() for f in fl}
    assert ('Zz', 'Yy') in pairs
    assert (flow0['source_type'], flow0['foreign_type']) in pairs
    # single-direction pairs pass through unchanged
    kept = dedupe_mirrored_pairs({(MCNS, FW): flows_m}, [])
    assert set(kept) == {(MCNS, FW)}


def test_pair_network_circadian_query_adds_no_node():
    """§14 real-index probe, re-cut 2026-09-26: the per-pair
    `Network (type-level)` export for the FAFB taxonomy query
    ``cell_type = circadian_clock`` plots the 21 FAFB source types and their
    male-cns counterparts — the chip that produced them is search
    provenance, not a mapping endpoint, so it renders nowhere on the
    canvas."""
    from comparison.mapping_visualization import (
        build_mapping_network_graph,
        origin_seeded_flows,
    )
    from ui.neuron_index import (
        count_types_in_index,
        resolve_type_matches,
    )

    res = resolve_type_matches(['circadian_clock'], 'exact', [MCNS, FW])
    origins = res['origins']
    assert set(origins) == {FW}
    counts = count_types_in_index(load_cached_neuron_index(FW), origins[FW])
    flows = origin_seeded_flows(
        FW, origins[FW], MCNS, source_counts=counts,
        matched_origins=res['origin_matches'].get(FW))
    assert flows
    graph = build_mapping_network_graph(flows)
    assert not [n for n in graph.nodes if str(n).startswith('E|')]
    assert not [n for n, d in graph.nodes(data=True)
                if d['node_type'] == 'entry']
    labels = {str(d.get('label')) for _, d in graph.nodes(data=True)}
    assert 'circadian_clock' not in labels
    # the origin side keeps layer 0 and the counterpart side layer 1
    assert any(n.startswith(f'0|{FW}|') for n in graph.nodes)
    pair_edges = list(graph.edges())
    assert pair_edges
    assert all(u.startswith(f'0|{FW}|') and v.startswith(f'1|{MCNS}|')
               for u, v in pair_edges)
    # the 21 types the entry used to summarise are all on the canvas
    assert len([n for n in graph.nodes if n.startswith(f'0|{FW}|')]) == 21



def test_reverse_context_smp227_incoming_families(mapper):
    """§12.2/§12.3 real-index probe: the dataset-wide incoming context for
    the s-CPDN3* receiving types lists the full incoming family (4/4/6
    MCNS types including SMP227) and materializes the per-target
    selected/all-valid unions that explain the 6 → 94 forward fan-out."""
    from comparison.mapping_visualization import build_type_coverage
    from ui.neuron_index import (
        _load_coverage_index,
        _load_cross_match_index,
        build_reverse_type_contexts,
    )

    targets = ['s-CPDN3B', 's-CPDN3C', 's-CPDN3D']
    indexes = {ds: _load_cross_match_index(ds) for ds in (MCNS, FW)}
    contexts = build_reverse_type_contexts(
        mapper, MCNS, FW, targets,
        source_index=indexes[MCNS], target_index=indexes[FW],
        coverage_indexes={ds: _load_coverage_index(ds)
                          for ds in (MCNS, FW)})
    expected_sources = {
        's-CPDN3B': ['CB1011', 'CB3252', 'SMP226', 'SMP227'],
        's-CPDN3C': ['SMP218', 'SMP220', 'SMP221', 'SMP227'],
        's-CPDN3D': ['CB3508', 'SMP219', 'SMP222', 'SMP223', 'SMP227',
                     'SMP232'],
    }
    expected = {
        # (population union, selected union, all-valid union, receiving
        #  population, selected target, all-valid target)
        's-CPDN3B': (26, 22, 23, 25, 23, 25),
        's-CPDN3C': (28, 19, 25, 32, 18, 24),
        's-CPDN3D': (36, 25, 33, 37, 24, 30),
    }
    for target in targets:
        ctx = contexts[target]
        assert [s['type'] for s in ctx['sources']] == expected_sources[target]
        assert 'SMP227' in [s['type'] for s in ctx['sources']]
        assert ctx['truncated'] is False
        pop, sel, all_valid, recv, sel_t, all_t = expected[target]
        assert ctx['source_population_total'] == pop
        assert len(ctx['selected_source_union_ids']) == sel
        assert len(ctx['all_valid_source_union_ids']) == all_valid
        assert ctx['receiving_count'] == recv
        assert len(ctx['selected_target_union_ids']) == sel_t
        assert len(ctx['all_valid_target_union_ids']) == all_t

    # the merged backward row carries the dataset-wide scope
    flows = []
    pools = {}
    for source in ('SMP227',):
        chains = mapper.get_type_bridges(source, MCNS, FW, max_bridges=0)
        by_end = {}
        for chain in chains:
            if chain and chain[-1].get('value'):
                by_end.setdefault(chain[-1]['value'], []).append(chain)
        for foreign, end_chains in by_end.items():
            flows.append({
                'source_dataset': MCNS, 'target_dataset': FW,
                'source_type': source, 'foreign_type': foreign,
                'source_count': 6, 'foreign_count': 0,
                'matched_origin': f"type · '{source}'",
                'bridges': end_chains[:2],
            })
    coverage = build_type_coverage(
        {(MCNS, FW): flows}, pools,
        reverse_contexts={(FW, r): contexts[r] for r in targets})
    rows = {r['type']: r for r in coverage['reverse']}
    for target in targets:
        row = rows[target]
        assert row['coverage_scope'] == 'dataset-wide incoming'
        assert row['relationship'] == '1-to-N'
        assert row['incoming_source_count'] == len(expected_sources[target])
        assert 'SMP227' in row['active_query_sources']
        assert 'active query: SMP227' in row['mapped_from']
        for name in expected_sources[target]:
            assert name in row['mapped_from']


def test_mapper_snapshot_round_trip(mapper):
    """§startup (2026-09-10): a fresh process restores the derived mapper
    state from the snapshot instead of rebuilding (~1s vs ~7s), with
    identical structures and answers; the snapshot version gate refuses
    mismatched payloads."""
    snap_path = mapper._mapper_snapshot_path()
    assert snap_path is not None
    # force a full rebuild once so the snapshot reflects the current
    # datasets regardless of prior runs
    assert mapper.load(force_reload=True) is True
    assert snap_path.exists()
    stamp = snap_path.stat().st_mtime_ns

    fresh = CrossDatasetTypeMapper(verbose=False,
                                   workspace_path=str(REPO_ROOT))
    assert fresh.load() is True
    # restore, not rebuild: the snapshot file was read, not rewritten
    assert snap_path.stat().st_mtime_ns == stamp
    assert fresh._loaded and fresh.last_load_error is None
    assert len(fresh._conflicts) == len(mapper._conflicts)
    assert fresh._type_mappings == mapper._type_mappings
    assert (fresh.get_type_bridges('SMP227', MCNS, FW, max_bridges=0)
            == mapper.get_type_bridges('SMP227', MCNS, FW, max_bridges=0))

    # a version mismatch refuses the payload and leaves the mapper
    # untouched for a full rebuild
    refusing = CrossDatasetTypeMapper(verbose=False,
                                      workspace_path=str(REPO_ROOT))
    refusing.MAPPER_SNAPSHOT_VERSION = mapper.MAPPER_SNAPSHOT_VERSION + 1
    assert refusing._restore_mapper_snapshot(snap_path) is False
    assert not refusing._loaded


def test_bridge_linker_text_warns_on_unverified_same_name():
    """A same-name pair whose only chain is bare name equality states
    'no metadata verification (please double check)'; a verified chain
    overrides the warning (§9F)."""
    from comparison.cross_dataset_type_mapper import bridge_linker_text

    chains = [[{"dataset": MCNS, "column": "type", "value": "X"},
               {"dataset": FW, "column": "type", "value": "X"}]]
    info = bridge_linker_text(chains, MCNS, FW, "X")
    assert info["text"] == ("same name — no metadata verification "
                            "(please double check)")

    # with a crosswalk-verification chain the warning is replaced by the
    # linker facts (same name + the verification)
    chains.append([{"dataset": MCNS, "column": "type", "value": "X"},
                   {"dataset": FW, "column": "flywireType", "value": "X"}])
    info = bridge_linker_text(chains, MCNS, FW, "X")
    assert info["text"] == ("same name + flywireType 'X'")


def test_dataset_abbreviations_and_version_suffixes():
    """4-char dataset names: FAFB (never FLYW), and family collisions
    (two male-cns or two BANC versions) get version suffixes."""
    from utils.naming_utils import dataset_abbrev, make_unique_dataset_labels

    assert dataset_abbrev('flywire_FAFB_v783') == 'FAFB'
    assert dataset_abbrev('banc_v888') == 'BANC'
    assert dataset_abbrev('banc_v626') == 'BANC'
    assert dataset_abbrev('male-cns:v1.0') == 'MCNS'
    assert dataset_abbrev('flywire') == 'FAFB'  # never the FLYW fallback

    labels = make_unique_dataset_labels([
        'male-cns:v1.0', 'male-cns:v0.9',
        'banc_v888', 'banc_v626',
        'flywire_FAFB_v783'])
    # BOTH colliding labels get versions — a bare MCNS would stay ambiguous
    assert labels == ['MCNS_v1_0', 'MCNS_v0_9', 'BANC_v888', 'BANC_v626',
                      'FAFB']


def test_linker_network_one_column_per_linker_and_legend(mapper):
    """§layout (user 2026-09-06): the linker network lays out ONE column
    per bridge linker COLUMN (MCNS type -> flywireType ->
    additional_type(s) -> FAFB type = four columns), never per-chain hop
    order, and both the linker network and the linker Sankey carry a
    color-keyed bridge-linker legend."""
    from comparison.mapping_visualization import (
        build_bridge_linker_graph,
        render_bridge_linker_html,
        render_mapping_sankey_html,
    )
    from comparison.mapping_visualization import origin_seeded_flows

    flows = []
    for type_name in ('CL125', 'PLP080', 'MDN', 'aMe12'):
        flows += origin_seeded_flows(
            'male-cns:v1.0', [type_name], FW)
    graph = build_bridge_linker_graph(
        flows, source_dataset=MCNS, target_dataset=FW)
    # canonical column order: the crosswalk linker first, the FAFB
    # annotation second — the four-column shape
    assert graph.graph['linker_columns'] == ['flywireType',
                                             'additional_type(s)']
    # every linker COLUMN occupies exactly one x layer (no mixing)
    xs = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'linker':
            column = str(node).split('|')[1]
            xs.setdefault(column, set()).add(data['position']['x'])
    assert set(xs) == {'flywireType', 'additional_type(s)'}
    assert all(len(v) == 1 for v in xs.values())
    assert xs['additional_type(s)'] != xs['flywireType']

    html = render_bridge_linker_html(
        flows, source_dataset=MCNS, target_dataset=FW)
    # §layout unification: the linker columns are header chips in the
    # unified CODE: full (count) format alongside the dataset chips —
    # no separate bottom legend box, no duplicated group chip.
    assert html and 'data-dataset="flywireType"' in html
    assert 'data-dataset="additional_type(s)"' in html
    assert 'data-dataset="FAFB"' in html and 'data-dataset="MCNS"' in html
    assert '(2)' in html and 'flywire_FAFB_v783' in html
    assert 'Bridge linkers' not in html

    sankey = render_mapping_sankey_html(flows, variant='linker')
    assert sankey and 'Bridge linkers' in sankey
    assert 'additional_type(s):' in sankey


def test_panel_and_viewer_share_mapped_type_backend(mapper):
    """§backend unification (user 2026-09-07): the cross-dataset panel's
    summary and the viewer's mapped view resolve through the SAME
    ``mapped_type_targets`` backend, and the panel counts each target's
    neurons ONCE (no per-flow multiplicity: the old Σ foreign_count
    turned 219 unique male-cns neurons into 243 via the shared
    s-LNv / 5thsLNv_LNd6 / SMP227 targets)."""
    from ui.neuron_index import (
        collect_native_type_matches,
        count_types_in_index,
        enrich_native_type_matches,
        mapped_type_targets,
        resolve_type_matches,
    )
    from ui.neuron_index import load_cached_neuron_index

    res = resolve_type_matches(['circadian_clock'], 'exact',
                               [MCNS, FW])
    origins = res['origins']
    assert set(origins) == {FW}
    assert res['origin_matches'][FW]
    assert all(
        any(record['column'] == 'cell_type'
            and record['value'] == 'circadian_clock'
            for record in records)
        for records in res['origin_matches'][FW].values())

    # shared-engine panel side
    idx = {MCNS: load_cached_neuron_index(MCNS), FW: load_cached_neuron_index(FW)}
    panel_targets = set()
    for target in (MCNS, FW):
        if target == FW:
            continue
        for otype in origins[FW]:
            ann = mapped_type_targets(mapper, otype, FW, target)
            if ann:
                panel_targets.update(ann['targets'])
    # viewer mapped-view side
    entries = collect_native_type_matches(
        MCNS, 'circadian_clock', datasets=[MCNS, FW], uncapped=True)
    enrich_native_type_matches(entries, MCNS)
    viewer_targets = set()
    for entry in entries:
        for label in (entry.get('labels_all') or []):
            for covered in (label.get('covered_all') or []):
                if covered.get('mapped'):
                    viewer_targets.update(
                        covered['mapped']['targets'])
    assert panel_targets == viewer_targets
    unique_neurons = sum(
        count_types_in_index(idx[MCNS], sorted(panel_targets)).values())
    # unique count, not the 243 per-flow multiplicity sum
    assert unique_neurons == 219, unique_neurons
    assert len(panel_targets) == 40


def _apdn3_bridge_flows_and_pools():
    """The user-reported APDN3 scenario (2026-09-07): the four male-cns
    types that bridge onto FAFB APDN3 (CL125, SLP249 4 neurons each;
    SLP250, PLP080 2 each), with per-pair bodyId pools attached."""
    from comparison.mapping_visualization import (
        dedupe_mirrored_pairs,
        origin_seeded_flows,
    )
    from ui.neuron_index import (
        count_types_in_index,
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    queries = ['CL125', 'SLP249', 'SLP250', 'PLP080']
    flows = origin_seeded_flows(
        MCNS, queries, FW,
        source_counts=count_types_in_index(index, queries))
    assert flows
    assert {f['foreign_type'] for f in flows} == {'APDN3'}
    foreign_counts = count_types_in_index(foreign_index, ['APDN3'])
    for flow in flows:
        flow['foreign_count'] = foreign_counts.get(
            flow['foreign_type'], 0)
    flows = dedupe_mirrored_pairs({(MCNS, FW): flows}, [MCNS])[(MCNS, FW)]
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        key = (flow['source_type'], flow['foreign_type'])
        if key not in pools:
            pools[key] = pool_bridge_body_ids(
                MCNS, FW, standardize_bridge(chain, MCNS, FW),
                flow['source_type'], flow['foreign_type'],
                indexes={MCNS: index, FW: foreign_index})
    return flows, pools


def test_pair_edge_weights_match_sankey_and_are_pair_specific():
    """User report: every network edge showed the same "12" (the source
    type's whole count duplicated onto each edge) and disagreed with the
    Sankey ribbon.  The network edge, the linker-path edge and the
    Sankey ribbon all draw the shared pair_flow_weight now — the pair's
    own granularity (4 / 2), never the shared target's 12."""
    from comparison.mapping_visualization import (
        build_bridge_linker_graph,
        build_mapping_network_graph,
        build_mapping_sankey_paths,
    )

    flows, pools = _apdn3_bridge_flows_and_pools()

    graph = build_mapping_network_graph(flows, pools=pools)
    network = {}
    for u, v, d in graph.edges(data=True):
        if u.startswith('0|'):
            network[(u.split('|', 2)[2], v.split('|', 2)[2])] = d['weight']
    sankey = {
        (names[0].split(' · ')[0], names[1].split(' · ')[0]): ws[0]
        for names, ws in build_mapping_sankey_paths(
            flows, pools=pools, variant='type')}
    assert network == sankey
    # pair-specific: 4-neuron sources carry 4, 2-neuron sources 2 —
    # never the shared 12-neuron target's count on every edge
    assert set(network.values()) == {2, 4}

    linker_graph = build_bridge_linker_graph(
        flows, source_dataset=MCNS, target_dataset=FW, pools=pools)
    final_weights = {d['weight'] for u, v, d in linker_graph.edges(
        data=True) if v.endswith('|APDN3')}
    assert final_weights == {2, 4}


def test_apdn3_node_hover_pools_union_across_pairs():
    """User report: the FAFB APDN3 node hovered "pool 4 bodyIds" next to
    "(12 neurons)".  Each male-cns counterpart bridges a DIFFERENT
    disjoint subset of APDN3's bodyIds (4/4/2/2); the hover count is the
    UNION across the pairs — 12 — not the largest single subset."""
    from comparison.mapping_visualization import build_mapping_network_graph

    flows, pools = _apdn3_bridge_flows_and_pools()
    graph = build_mapping_network_graph(flows, pools=pools)
    title = graph.nodes[f'1|{FW}|APDN3']['title']
    assert '(12 neurons)' in title
    assert 'pool 12 bodyIds' in title


def test_sankey_renders_no_parallel_duplicate_links():
    """User report: PLP080 → APDN3 drew TWO parallel ribbons because the
    pair's two derivation chains (direct annotation bridge + crosswalk
    chain) reached the shared band at different hop depths and the
    backend keyed its edges by layer.  Edge keys are layer-less now, so
    the same node pair merges into ONE link (max weight)."""
    import json
    import re

    from comparison.mapping_visualization import render_mapping_sankey_html

    flows, pools = _apdn3_bridge_flows_and_pools()
    html = render_mapping_sankey_html(flows, pools=pools)
    assert html
    # Anchor on the figure's own div id: the Sankey now EMBEDS plotly.js, and
    # the bundled library contains the string 'Plotly.newPlot(' itself, so a
    # first-occurrence search lands inside minified library code.
    div_id = re.search(r'<div id="([a-zA-Z0-9_\-]+)" class="plotly-graph-div"',
                       html).group(1)
    call = html.index('Plotly.newPlot(', html.index(f'"{div_id}"',
                                                    html.index(div_id)))
    start = html.index('[', call)
    figure = json.JSONDecoder().raw_decode(html[start:])[0][0]
    labels = figure['node']['label']
    links_by_pair = {}
    for s, t, v in zip(figure['link']['source'],
                       figure['link']['target'],
                       figure['link']['value']):
        links_by_pair.setdefault((labels[s], labels[t]), []).append(v)
    duplicates = {pair: values for pair, values in links_by_pair.items()
                  if len(values) > 1}
    assert not duplicates, duplicates


def test_network_html_edge_label_font_size_control():
    """User report: the on-edge weight labels were pinned to 9px while
    only the node labels had a size control.  The network now carries an
    'Edge Label Size' spinner, wired through the undo history."""
    from comparison.mapping_visualization import render_bridge_linker_html

    flows, pools = _apdn3_bridge_flows_and_pools()
    html = render_bridge_linker_html(
        flows, source_dataset=MCNS, target_dataset=FW, pools=pools)
    assert html
    assert 'id="edgeLabelSizeSlider"' in html
    assert 'function updateEdgeLabelFontSize' in html
    assert 'edgeLabelFontSize: globalEdgeLabelFontSize' in html
    assert "updateEdgeLabelFontSize(gs.edgeLabelFontSize)" in html


# ---------------------------------------------------------------------------
# Valid-mapper parity baseline (plan-valid-type-mapper-parity, Workstream A)
# ---------------------------------------------------------------------------

def test_resolver_rename_split_conflict_anchors():
    """The shared resolver reproduces the panel's validity policy on the
    three baseline anchors: unique rename, licensed split, and conflict."""
    from comparison.type_resolver import (
        STATUS_CONFLICT,
        STATUS_VALID_SPLIT,
        equivalence_key,
        expansion_targets,
        resolve_valid_targets,
    )

    mapper = get_type_mapper()
    assert mapper._loaded

    rename = resolve_valid_targets(mapper, 'MeVPLo2', MCNS, FW)
    assert rename.status == 'mapped'
    assert rename.target_types == ('MTe07',)
    assert equivalence_key(rename) == 'MTe07'

    split = resolve_valid_targets(mapper, 'VS', MCNS, FW)
    assert split.status == STATUS_VALID_SPLIT
    assert len(split.target_types) > 1
    assert equivalence_key(split) is None
    assert len(expansion_targets(split)) == len(split.target_types)

    conflict = resolve_valid_targets(mapper, 'CB1011', BANC, MCNS)
    assert conflict.status == STATUS_CONFLICT
    assert conflict.target_types == ()
    assert conflict.conflicts
    assert expansion_targets(conflict) == ()


def test_panel_and_resolver_parity():
    """Panel backend (mapped_type_targets) and core resolver agree on the
    anchors apart from UI-only fields."""
    from comparison.type_resolver import resolve_valid_targets

    mapper = get_type_mapper()
    for foreign, foreign_ds, selected_ds in [
            ('MeVPLo2', MCNS, FW), ('VS', MCNS, FW), ('CB1011', BANC, MCNS)]:
        panel = mapped_type_targets(mapper, foreign, foreign_ds, selected_ds)
        res = resolve_valid_targets(mapper, foreign, foreign_ds, selected_ds)
        assert panel is not None
        assert panel['kind'] == res.kind
        assert panel['targets'] == list(res.target_types)
        if res.status == 'mapped' and res.kind in ('renamed', 'same name'):
            assert 'status' not in panel
        else:
            assert panel.get('status') == res.status


def test_profile_expansion_follows_production_contract():
    """expand_profile_types returns canonical_type -> numeric weight (the
    production ``standardize_partner_types`` contract), maps the FAFB
    rename to its canonical MCNS key, and counts the unmapped raw-name
    fallback."""
    from comparison.type_resolver import expand_profile_types

    mapper = get_type_mapper()
    expansion = expand_profile_types(
        mapper, {'MTe07': 2.0, 'TotallyUnknownTypeZz': 1.0}, FW)
    assert expansion.canonical.get('MeVPLo2') == 2.0
    assert expansion.key_status.get('MeVPLo2') == 'mapped'
    assert expansion.canonical.get('TotallyUnknownTypeZz') == 1.0
    assert expansion.fallback_used is True
    assert all(isinstance(w, float) for w in expansion.canonical.values())


def test_split_expansion_distributes_weight_evenly():
    """A valid split distributes its weight across every licensed target
    (documented 'even' rule, no double counting)."""
    from comparison.type_resolver import expand_profile_types

    mapper = get_type_mapper()
    expansion = expand_profile_types(
        mapper, {'VS': 4.0}, MCNS, FW)
    split_targets = [k for k in expansion.canonical if k.startswith('VS')]
    assert len(split_targets) > 1
    assert expansion.key_status[split_targets[0]] == 'valid_split_evidence'
    assert sum(expansion.canonical[t] for t in split_targets) == pytest.approx(4.0)
    assert expansion.split_policy == 'even'


def test_compute_same_type_candidates_real_mapper():
    """Regression (plan §2.3 defect 1): the real mapper's canonical->weight
    contract must yield the renamed candidate.  The pre-fix helper unpacked
    the standardized dict backwards and returned an empty set here."""
    from comparison.profile_comparator import HomologFinder

    mapper = get_type_mapper()
    finder = HomologFinder.__new__(HomologFinder)
    out = finder._compute_same_type_candidates(
        ['MeVPLo2'], {101: 'MTe07', 102: 'Other'}, mapper, FW,
        source_dataset=MCNS)
    assert out == {101}


def test_compute_same_type_candidates_conflict_fails_closed():
    """A conflicted target type must not re-enter the candidate pool via
    its raw same-name when auto mapping is active (plan §2.4 defect 3)."""
    from comparison.profile_comparator import HomologFinder

    mapper = get_type_mapper()
    finder = HomologFinder.__new__(HomologFinder)
    out = finder._compute_same_type_candidates(
        ['CB1011'], {201: 'CB1011'}, mapper, BANC, source_dataset=MCNS)
    assert out == set()


def test_mapper_source_and_load_state_metadata():
    """The default ACTIVE mapper is the v1.0 table (v0.9 stays auxiliary /
    legacy data), and the metadata block records requested vs active
    state, source, version, and a clean load error."""
    from comparison.type_resolver import (
        auto_mapping_metadata,
        get_mapper_snapshot,
    )

    snapshot = get_mapper_snapshot(get_type_mapper())
    meta = auto_mapping_metadata(snapshot)
    assert meta['auto_type_mapping_requested'] is True
    assert meta['auto_type_mapping_active'] is True
    assert meta['auto_type_mapping_source'] == (
        'male-cns_v1_0/male-cns_v1_0_allneurons_neuron_df.csv')
    assert meta['auto_type_mapping_version_or_hash']
    assert meta['auto_type_mapping_load_error'] is None
    assert meta['mapping_policy_version']


def test_failed_mapper_load_is_observable(tmp_path):
    """A failed load is retryable and observable: ``last_load_error`` is
    recorded, the snapshot reports inactive, and resolutions fail closed
    with ``mapper_unavailable`` (plan Workstream E)."""
    from comparison.type_resolver import (
        STATUS_MAPPER_UNAVAILABLE,
        get_mapper_snapshot,
        resolve_valid_targets,
    )

    mapper = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    assert mapper.load() is False
    assert mapper.last_load_error

    snapshot = get_mapper_snapshot(mapper)
    assert snapshot.loaded is False
    assert snapshot.load_error
    res = resolve_valid_targets(
        mapper, 'MeVPLo2', MCNS, FW, snapshot=snapshot)
    assert res.status == STATUS_MAPPER_UNAVAILABLE


# ---------------------------------------------------------------------------
# Canonical merge keys (plan-unify-mapper-backends, Workstream B)
# ---------------------------------------------------------------------------

def test_canonical_merge_key_branches():
    """canonical_merge_key: licensed renames map to the canonical target,
    same-namespace stays native, conflicts become dataset-scoped, and
    unmapped types keep their raw name with a recorded fallback status."""
    from comparison.type_resolver import canonical_merge_key

    mapper = get_type_mapper()

    # licensed rename (MCNS source is native; FAFB source maps)
    assert canonical_merge_key(mapper, 'MeVPLo2', MCNS).key == 'MeVPLo2'
    mapped = canonical_merge_key(mapper, 'MTe07', FW)
    assert mapped.key == 'MeVPLo2' and mapped.status == 'mapped'

    # conflict: dataset-scoped key, no raw same-name merging
    conflict = canonical_merge_key(mapper, 'CB1011', BANC)
    assert conflict.key == f'{BANC}:CB1011'
    assert conflict.status == 'conflict'
    # ...and the MCNS-side raw name keeps its own (distinct) key
    assert canonical_merge_key(mapper, 'CB1011', MCNS).key == 'CB1011'

    # unmapped long-tail fallback
    fallback = canonical_merge_key(mapper, 'TotallyUnknownTypeZz', FW)
    assert fallback.key == 'TotallyUnknownTypeZz'
    assert fallback.status == 'unmapped'

    # split parity: FAFB branches arrive per-branch as mapped; the MCNS
    # split source itself stays native in the canonical namespace
    assert canonical_merge_key(mapper, 'VS1', FW).key == 'VS'
    native_split = canonical_merge_key(mapper, 'VS', MCNS)
    assert native_split.key == 'VS'


def test_canonical_merge_key_cache():
    from comparison.type_resolver import canonical_merge_key

    mapper = get_type_mapper()
    cache = {}
    first = canonical_merge_key(mapper, 'MTe07', FW, cache=cache)
    second = canonical_merge_key(mapper, 'MTe07', FW, cache=cache)
    assert first is second
    assert list(cache.keys()) == [('flywire_FAFB_v783', 'MTe07',
                                   'male-cns:v1.0')]


def test_same_namespace_and_flow_status_policy():
    """same_namespace + resolve_flow_status: the flow policy defined once —
    conflict markers, bridge-end relabeling, and namespace guards."""
    from comparison.type_resolver import (
        same_namespace, resolve_flow_status, STATUS_BRIDGED,
        STATUS_CONFLICT, STATUS_VALID_SPLIT,
    )

    mapper = get_type_mapper()
    assert same_namespace(mapper, MCNS, MCNS) is True
    assert same_namespace(mapper, MCNS, FW) is False

    # conflict marker (caller skips the flow)
    status, fields = resolve_flow_status(
        mapper, 'CB1011', BANC, MCNS, bridge_end_count=3)
    assert status == STATUS_CONFLICT and fields['conflicts']

    # unmapped + single bridge end -> bridged; N ends -> valid split
    status_one, _ = resolve_flow_status(
        mapper, 'TotallyUnknownTypeZz', MCNS, FW, bridge_end_count=1)
    assert status_one == STATUS_BRIDGED
    status_n, _ = resolve_flow_status(
        mapper, 'TotallyUnknownTypeZz', MCNS, FW, bridge_end_count=4)
    assert status_n == STATUS_VALID_SPLIT

    # mapped passthrough with scoped decision fields
    status_m, fields_m = resolve_flow_status(mapper, 'MeVPLo2', MCNS, FW)
    assert status_m == 'mapped'
    assert fields_m['target_types'] == ['MTe07']

def test_auto_mapping_metadata_basis_fields():
    """The metadata block self-describes its counting unit and keeps the
    partner-occurrence metric on a distinct key (P1.5)."""
    from comparison.profile_comparator import auto_mapping_result_metadata

    meta = auto_mapping_result_metadata(
        requested=True,
        type_mapper=get_type_mapper(),
        resolution_counts={'mapped': 5},
        partner_resolution_counts={'mapped': 900, 'unmapped': 4},
        raw_fallback_used=True,
    )
    assert meta['mapping_resolution_counts_basis'] == 'unique_type_resolutions'
    assert meta['mapping_resolution_counts_by_status'] == {'mapped': 5}
    assert meta['mapping_partner_type_resolutions_by_status'] == {
        'mapped': 900, 'unmapped': 4}
    assert meta['raw_fallback_used'] is True

    # disabled branch keeps the same key shape (empty counts), no crash
    disabled = auto_mapping_result_metadata(
        requested=False, type_mapper=None)
    assert disabled['mapping_resolution_counts_basis'] == 'unique_type_resolutions'
    assert disabled['mapping_resolution_counts_by_status'] == {}
    assert disabled['mapping_partner_type_resolutions_by_status'] == {}



# ---------------------------------------------------------------------------
# Phase 1b — curated unique relations survive the derivation union
# (plan-cross-dataset-query-resolution-samename-taxonomy §2.1b)
# ---------------------------------------------------------------------------

def test_curated_same_name_one_to_one_survives_bridge_union(mapper):
    """male-cns ``L2`` -> FAFB is a curated same-name 1-to-1, but FAFB's
    per-neuron ``additional_type(s)`` cells (``Mi13`` carries ``L2`` etc.)
    used to widen the union into a 9-candidate ``one of N`` with no unique
    answer.  The curated relation must be the equivalence result, with the
    alt-column ends demoted to display-only secondary targets."""
    from comparison.type_resolver import resolve_valid_targets

    res = resolve_valid_targets(mapper, 'L2', MCNS, FW)
    assert res.status == 'mapped'
    assert res.kind == 'same name'
    assert res.equivalence_key == 'L2'
    assert res.target_types == ('L2',)
    # the noisy alt-column ends stay visible as secondary display targets
    assert res.secondary_targets
    assert 'L2' not in res.secondary_targets


def test_stale_rename_defers_to_native_same_name(mapper):
    """§stale claims (2026-09-12), superseding the old
    ``test_curated_rename_wins_over_same_name_alias``: male-cns ``Dm8a``'s
    crosswalk names the FAFB counterpart ``yDm8``, but FAFB v783 carries
    neither a primary nor an additional ``yDm8`` — the curated claim is
    stale.  The realized native same-name wins the equivalence (query tier
    §2.1) and the unfulfilled claim is kept in the reason; a curated
    rename to a name the target actually carries still wins over the
    alias, as before."""
    from comparison.type_resolver import resolve_valid_targets

    res = resolve_valid_targets(mapper, 'Dm8a', MCNS, FW)
    assert res.status == 'mapped'
    assert res.kind == 'same name'
    assert res.equivalence_key == 'Dm8a'
    assert res.target_types == ('Dm8a',)
    assert 'yDm8' in (res.reason or '')
    assert mapper.type_exists(FW, 'yDm8') is False


def test_panel_adapter_sees_clean_same_name_for_l2(mapper):
    """The Type Mapping panel adapter reshapes the resolution; for L2 it now
    reports a plain same-name match instead of the one-of-N noise."""
    from ui.neuron_index import mapped_type_targets

    out = mapped_type_targets(mapper, 'L2', MCNS, FW)
    # ``curated_identity`` is part of the adapter contract since the UI
    # alignment round (plan-ui-type-mapper-alignment §5.1): the viewer must
    # tell a curated same-name relation from a bare echo, so the flag rides
    # on the annotation.  The rest of the shape is unchanged.
    assert out == {'kind': 'same name', 'targets': ['L2'],
                   'curated_identity': True}


def test_same_name_origin_split_confirmed_vs_echo(mapper, tmp_path):
    """Phase 1c (§2.1c): the per-bridge export labels an evidence-confirmed
    same-name pairing ``same name+evidence`` while a bare echo keeps
    ``same name``.  Additive split — no existing label value is renamed."""
    out = tmp_path / 'per_bridge.csv'
    mapper.export_mapping_per_bridge(
        str(out), source_types=['L2', 'Dm8a'], source_dataset=MCNS,
        target_datasets=[FW, BANC], pool_fn=lambda *a, **k: None)
    df = pd.read_csv(out)
    origins = {(r.source_type, r.target_dataset, r.target_type): r.mapping_origin
               for r in df.itertuples()}
    # BANC's own malecns_cell_type label confirms MCNS L2 = BANC L2
    assert ('L2', BANC, 'L2') in origins
    assert origins[('L2', BANC, 'L2')] == 'same name+evidence'
    # FAFB renames Dm8a to yDm8, so the FAFB Dm8a pairing stays a bare echo
    assert ('Dm8a', FW, 'Dm8a') in origins
    assert origins[('Dm8a', FW, 'Dm8a')] == 'same name'


def test_target_taxonomy_query_bridges_across_datasets(mapper):
    """Phase 2/3 worked example (plan §2.2/§2.3): the target token
    ``circadian_clock`` queries the 21 FAFB members natively and the mapped
    unions in MCNS/BANC — previously those two legs matched nothing and the
    comparison was silently empty there.  Non-taxonomy sources keep their
    identity names verbatim."""
    from comparison.comparison_parameters import ComparisonParameters

    p = ComparisonParameters(
        datasets=[MCNS, FW, BANC],
        source_neurons=['L2'], target_neurons=['circadian_clock'],
        auto_type_mapping=True)
    p._auto_type_mapper = mapper

    tgt_fafb = p.get_target_neurons_for_dataset(FW)
    assert len(tgt_fafb) == 21
    assert 'DN1a' in tgt_fafb and 'l-LNv' in tgt_fafb
    tgt_mcns = p.get_target_neurons_for_dataset(MCNS)
    assert 'DN1a' in tgt_mcns and 'LNd_b' in tgt_mcns and '5thsLNv_LNd6' in tgt_mcns
    tgt_banc = p.get_target_neurons_for_dataset(BANC)
    assert 's-LNv_b' in tgt_banc and 'aMe13' in tgt_banc

    # sources: identity names everywhere (query content unchanged)
    for ds in (MCNS, FW, BANC):
        assert p.get_source_neurons_for_dataset(ds) == ['L2']


# ---------------------------------------------------------------------------
# Curated-subsumes-annotation (user 2026-09-27): a one-linker
# ``additional_type(s)`` chain whose token a curated ``fafb_cell_type``
# chain already bridges to the SAME target is redundant — the linker
# network drew two PARALLEL one-linker paths (aT 'CB1770' beside fct
# 'CB1770').  aT-only pairs, composed aT→ACT routes, and same-target
# chains whose tokens disagree keep their chains.
# ---------------------------------------------------------------------------

BANC888 = 'banc_v888'


def test_curated_subsumes_same_token_annotation_landing(mapper):
    chains = mapper.get_type_bridges('s-CPDN3A', FW, BANC, max_bridges=0)
    assert chains
    by_end = {}
    for chain in chains:
        linkers = [l for l in standardize_bridge(chain, FW, BANC)
                   if l['kind'] == 'linker']
        by_end.setdefault(chain[-1]['value'], []).append(
            [l['column'] for l in linkers])
    # every fct-bridged target lost its parallel one-linker aT chain
    for end, shapes in by_end.items():
        if ['fafb_cell_type'] in shapes:
            assert ['additional_type(s)'] not in shapes, end
    # the curated bridges themselves are untouched
    assert any('fafb_cell_type' in s for shapes in by_end.values()
               for s in shapes)


def test_annotation_only_pair_keeps_its_at_chain(mapper):
    # CB3767 has NO fct chain from s-CPDN3C (the curated vote disagrees),
    # so its annotation-landing chain survives subsumption
    chains = mapper.get_type_bridges('s-CPDN3C', FW, BANC, max_bridges=0)
    shapes = {tuple(h['column'] for h in chain[1:]): chain[-1]['value']
              for chain in chains}
    assert shapes.get(('fafb_cell_type',)) != 'CB3767'
    assert ('additional_type(s)', 'type') in shapes
    assert shapes[('additional_type(s)', 'type')] == 'CB3767'


def test_composed_at_act_chain_survives_subsumption(mapper):
    # 4I1 -> FB4F_a travels the two-annotation-column standard; the fct
    # column has no chain there, so nothing is subsumed
    chains = mapper.get_type_bridges('4I1', FW, BANC, max_bridges=0)
    assert any(
        'Alternative Cell Type(s)' in tuple(h['column'] for h in chain[1:])
        for chain in chains)


def test_reverse_direction_subsumes_the_all_annotation_route(mapper):
    # BANC SMP537 -> FAFB: the curated label bridges token 'SMP537' to
    # DN1pD, so the independent all-aT route through the same token is
    # subsumed — the pair keeps exactly its curated chain
    chains = mapper.get_type_bridges('SMP537', BANC888, FW, max_bridges=0)
    assert chains
    assert all(chain[1]['column'] == 'fafb_cell_type' for chain in chains)
    assert chains[0][-1]['value'] == 'DN1pD'


# ---------------------------------------------------------------------------
# §token chaining (user 2026-09-27): a fafb_cell_type edge lands directly
# on the FAFB primary only when the cell token names it EXACTLY.  A
# rename-resolved winner (cell 'SMP537' -> 'DN1pD') terminates at the raw
# token node and the walk continues through FAFB's own
# additional_type(s) edge — the chain SHOWS the rename instead of hiding
# it in one hop.  Decision/provenance stay winner-resolved.
# ---------------------------------------------------------------------------

def test_token_chained_label_edge_for_renamed_winner(mapper):
    chains = mapper.get_type_bridges('SMP537', 'banc_v888', FW,
                                     max_bridges=0)
    assert chains, 'SMP537 -> FAFB must derive'
    shapes = {tuple(h['column'] for h in chain[1:]): chain for chain in
              chains}
    chained = shapes.get(('fafb_cell_type', 'additional_type(s)'))
    assert chained is not None, sorted(map(str, shapes))
    assert chained[-1]['value'] == 'DN1pD'
    # no direct one-hop fct->DN1pD chain survives for a renamed token
    assert ('fafb_cell_type',) not in shapes
    # the decision layer stays winner-resolved
    decision = mapper.get_mapping_decision('SMP537', 'banc_v888', FW)
    assert decision['status'] == 'mapped'
    assert decision['target_type'] == 'DN1pD'
    assert decision['support']['votes'] == {'DN1pD': 3}


def test_exact_match_label_edge_stays_direct(mapper):
    # Mi1's fct cell names the FAFB primary Mi1 exactly — one linker
    chains = mapper.get_type_bridges('Mi1', 'banc_v626', FW, max_bridges=0)
    shapes = [tuple(h['column'] for h in chain[1:]) for chain in chains]
    assert ('fafb_cell_type',) in shapes, shapes


def test_reverse_chained_route_is_at_then_fct(mapper):
    # FAFB DN1pD -> BANC: the aT hop reaches the token node, the fct hop
    # crosses into BANC (user form 3), for both SMP537 and SMP539
    chains = mapper.get_type_bridges('DN1pD', FW, 'banc_v888',
                                     max_bridges=0)
    shapes = {tuple(h['column'] for h in chain[1:]): chain[-1]['value']
              for chain in chains}
    assert shapes.get(
        ('additional_type(s)', 'fafb_cell_type')) in ('SMP537', 'SMP539')
    ends = {chain[-1]['value'] for chain in chains}
    assert {'SMP537', 'SMP539'} <= ends


def test_chained_pair_pools_linker_rows_on_both_sides(mapper):
    from ui.neuron_index import resolve_prioritized_bridge_pool
    chains = mapper.get_type_bridges('SMP537', 'banc_v888', FW,
                                     max_bridges=0)
    pool = resolve_prioritized_bridge_pool(
        'banc_v888', FW, chains, 'SMP537', 'DN1pD')
    assert pool['resolution_status'] == 'supported'
    assert pool['source_basis'] == 'linker rows'
    assert pool['target_basis'] == 'linker rows'
    # 3 BANC voters + the DN1pD rows whose aT cell carries SMP537
    assert pool['source_pool_size'] == 3
    assert pool['target_pool_size'] == 4
    assert pool['target_type_total'] == 8


def test_linker_columns_order_by_traversal_side(mapper):
    """§token chaining alignment (user 2026-09-27): sequential two-linker
    bridges must wire left-to-right in the linker network for EVERY pair,
    like the MCNS --flywireType--aT-- FAFB standard.  Columns order by
    traversal SIDE (source-home columns first, target-home last), so the
    FAFB --aT--> token --fct--> BANC chain puts additional_type(s) before
    fafb_cell_type while MCNS -> FAFB keeps flywireType first."""
    from comparison.mapping_visualization import (
        build_bridge_linker_graph, origin_seeded_flows)

    fwd_flows = origin_seeded_flows('male-cns:v1.0', ['CL125'], FW)
    graph = build_bridge_linker_graph(
        fwd_flows, source_dataset=MCNS, target_dataset=FW)
    assert graph.graph['linker_columns'] == [
        'flywireType', 'additional_type(s)']

    rev_flows = origin_seeded_flows(FW, ['DN1pD'], BANC888)
    graph = build_bridge_linker_graph(
        rev_flows, source_dataset=FW, target_dataset=BANC888)
    columns = graph.graph['linker_columns']
    assert 'additional_type(s)' in columns and 'fafb_cell_type' in columns
    assert columns.index('additional_type(s)') < columns.index(
        'fafb_cell_type')
    # every linker->linker edge flows left-to-right
    for src, dst in graph.edges():
        a = graph.nodes[src].get('node_type')
        b = graph.nodes[dst].get('node_type')
        if a == 'linker' and b == 'linker':
            assert (graph.nodes[dst]['position']['x']
                    > graph.nodes[src]['position']['x']), (src, dst)


# ---------------------------------------------------------------------------
# 2026-09-27 duality alignment: the three re-derivation sites (profile
# weight canonicalization, query expansion, Route A members) consult the
# BRIDGE-INCLUSIVE decision, so a bridge-derived fan-out (pure-ACT etc.)
# behaves identically to a crosswalk split — weight splits evenly, Route
# A returns all branches.  Also: curated subsumption compares CANONICAL
# tokens, so an auto:-spelled fct token subsumes its aT twin.
# ---------------------------------------------------------------------------

def test_profile_weight_splits_for_bridge_derived_fanout(mapper):
    from comparison.type_resolver import MapperSnapshot, expand_profile_types
    snap = MapperSnapshot(mapper)
    res = expand_profile_types(
        mapper, {'AN_GNG_161': 10.0}, FW, snapshot=snap,
        target_dataset=BANC)
    # the pure-ACT fan-out splits evenly across its two BANC targets
    # (pre-fix it stranded under the raw name with fallback_used=True)
    assert res.canonical == {'AN07B060': 5.0, 'AN18B053': 5.0}
    assert res.fallback_used is False
    # the crosswalk split is unchanged
    res2 = expand_profile_types(
        mapper, {'PLP120': 10.0}, FW, snapshot=snap, target_dataset=BANC)
    assert res2.canonical == {
        'PLP069': 5.0, 'PLP120,PLP145': 5.0}


def test_route_a_expansion_parity_for_fanout_and_split(mapper):
    from comparison.query_resolver import _member_targets
    fanout = _member_targets(mapper, 'AN_GNG_161', FW, BANC)
    split = _member_targets(mapper, 'PLP120', FW, BANC)
    assert fanout == ['AN07B060', 'AN18B053']
    assert split == ['PLP069', 'PLP120,PLP145']
