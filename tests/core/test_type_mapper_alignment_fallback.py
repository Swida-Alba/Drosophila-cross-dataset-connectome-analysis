"""The fafb_alignment_cell_type fallback lane (2026-09-27).

BANC publishes a second FAFB label column, ``fafb_alignment_cell_type``,
whose per-row agreement with the ``fafb_match`` bodyIds is weaker than the
curated ``fafb_cell_type`` (96.5% vs 99.2% whole-release).  The lane is
therefore fallback-only:

* it may fill a mapping ONLY where the curated pass yields no winner
  (keep-rule in ``record_votes``) — a curated winner keeps its provenance
  untouched, and a curated conflict stays fail-closed;
* its mappings carry the distinct kind
  ``cross-dataset cell type (alignment)`` and the evidence tier
  ``direct alignment label`` (below crosswalk, above auto);
* its own vote conflicts are diagnostics (``_banc_label_votes``), never
  ``TypeMappingConflict`` records.

Hermetic tests build synthetic tables with both label columns; real-data
tests pin the 2026-09-27 measurements on the local v626/v888 tables.
"""

from pathlib import Path

import pytest

from comparison.cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    linker_evidence_tier,
    prioritized_bridge_chains,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC626 = 'banc_v626'
BANC888 = 'banc_v888'
ALIGN = 'fafb_alignment_cell_type'

REPO_ROOT = Path(__file__).resolve().parents[2]
BANC_CSV = (REPO_ROOT / 'datasets' / 'banc_v626'
            / 'banc_v626_allneurons_neuron_df.csv')
BANC888_CSV = (REPO_ROOT / 'datasets' / 'banc_v888'
               / 'banc_v888_allneurons_neuron_df.csv')

requires_banc = pytest.mark.skipif(
    not BANC_CSV.exists(),
    reason='real BANC v626 neuron table not available locally',
)
requires_banc_v888 = pytest.mark.skipif(
    not BANC888_CSV.exists(),
    reason='real BANC v888 neuron table not available locally',
)


def _build_mapper(tmp_path: Path) -> CrossDatasetTypeMapper:
    v10 = tmp_path / 'mcns_v10.csv'
    v10.write_text(
        'bodyId,type,flywireType,hemibrainType,mancType\n'
        '1,Shared,FwShared,HbShared,MnShared\n',
        encoding='utf-8',
    )
    fafb = tmp_path / 'fafb.csv'
    fafb.write_text(
        'bodyId,type,additional_type(s)\n'
        'f1,FAlign,\n'
        'f2,FCurated,\n'
        'f3,FA,\n'
        'f4,FB,\n'
        'f5,AX,\n'
        'f6,AY,\n',
        encoding='utf-8',
    )
    # label columns: ACT, malecns_cell_type, malecns_match,
    # fafb_cell_type, fafb_alignment_cell_type, fafb_match
    banc626 = tmp_path / 'banc626.csv'
    banc626.write_text(
        'bodyId,type,Alternative Cell Type(s),malecns_cell_type,'
        'malecns_match,fafb_cell_type,fafb_alignment_cell_type,fafb_match\n'
        # curated empty, alignment clean -> alignment fills
        '6261,TFill,,,,,FAlign,f1\n'
        # curated winner; alignment AGREES -> must not touch provenance
        '6263,TCurated,,,,FCurated,FCurated,f2\n'
        # curated conflicted (FA vs FB), alignment clean -> stays conflicted
        '6264,TCflt,,,,FA,FA,\n'
        '6265,TCflt,,,,FB,FA,\n'
        # curated empty, alignment conflicted -> unmapped, NO record
        '6266,TAlignCflt,,,,,AX,\n'
        '6267,TAlignCflt,,,,,AY,\n',
        encoding='utf-8',
    )
    banc888 = tmp_path / 'banc888.csv'
    banc888.write_text(
        'bodyId,type,Alternative Cell Type(s),malecns_cell_type,'
        'malecns_match,fafb_cell_type,fafb_alignment_cell_type,fafb_match\n'
        '8881,T888,,,,,,\n',
        encoding='utf-8',
    )
    mapper = CrossDatasetTypeMapper(
        workspace_path=str(tmp_path),
        neuron_df_path=str(v10),
        flywire_neuron_df_paths={
            FAFB: str(fafb),
            BANC626: str(banc626),
            BANC888: str(banc888),
        },
        verbose=False,
    )
    assert mapper.load() is True
    return mapper


class TestAlignmentFallbackHermetic:
    def test_alignment_fills_where_curated_has_no_winner(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        decision = mapper.get_mapping_decision('TFill', BANC626, FAFB)
        assert decision['status'] == 'mapped'
        assert decision['target_type'] == 'FAlign'
        support = decision['support']
        assert support['column'] == ALIGN
        assert support['kind'] == 'cross-dataset cell type (alignment)'
        chains = mapper.get_type_bridges('TFill', BANC626, FAFB)
        assert any(hop['column'] == ALIGN
                   for chain in chains for hop in chain)

    def test_alignment_never_touches_a_curated_winner(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        decision = mapper.get_mapping_decision('TCurated', BANC626, FAFB)
        assert decision['status'] == 'mapped'
        assert decision['target_type'] == 'FCurated'
        assert decision['support']['column'] == 'fafb_cell_type'
        assert decision['support']['kind'] == 'cross-dataset cell type'

    def test_alignment_never_fills_a_curated_conflict(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        decision = mapper.get_mapping_decision('TCflt', BANC626, FAFB)
        assert decision['status'] == 'conflict'
        assert set(decision['target_types']) == {'FA', 'FB'}
        assert (BANC626, 'TCflt', FAFB) not in mapper._bridge_provenance \
            or mapper._bridge_provenance[
                (BANC626, 'TCflt', FAFB)].get('column') != ALIGN

    def test_alignment_conflicts_are_diagnostics_not_records(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        decision = mapper.get_mapping_decision('TAlignCflt', BANC626, FAFB)
        assert decision['status'] == 'unmapped'
        assert decision['conflicts'] == []
        votes = mapper._banc_label_votes.get((BANC626, ALIGN, 'TAlignCflt'))
        assert votes and set(votes['votes']) == {'AX', 'AY'}

    def test_alignment_tier_sits_between_crosswalk_and_auto(self):
        assert linker_evidence_tier(
            ALIGN, 'FAlign') == 'direct alignment label'
        assert linker_evidence_tier(
            'fafb_cell_type', 'X') == 'direct curated label'

    def test_alignment_support_renders_verified_wording(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        prov = mapper._bridge_provenance[(BANC626, 'TFill', FAFB)]
        text = mapper._format_bridge_support(prov)
        assert text.startswith('cross-dataset cell type (alignment):')
        assert '(verified 1)' in text  # fafb_match f1 backs the vote

    def test_alignment_fallback_rows_accessor(self, tmp_path):
        mapper = _build_mapper(tmp_path)
        rows = mapper.alignment_fallback_rows(datasets=[BANC626])
        assert [(r['source_type'], r['target_type']) for r in rows] == [
            ('TFill', 'FAlign')]
        assert rows[0]['total_votes'] == 1
        assert rows[0]['verified_votes'] == 1
        assert mapper.alignment_fallback_rows(
            filter_types={'TCurated'}, datasets=[BANC626]) == []

    def test_alignment_lane_needs_the_source_map(self, tmp_path):
        from comparison.cross_dataset_type_mapper import (
            BRIDGE_SOURCE_MAP, source_map_targets,
        )
        assert source_map_targets(BANC626, ALIGN) == {FAFB}
        assert source_map_targets(BANC888, ALIGN) == {FAFB}
        assert (BANC626, ALIGN) in BRIDGE_SOURCE_MAP


def test_alignment_fallback_notes_block(tmp_path):
    """The analyzer's [BANC alignment fallback] disclosure block.

    Driven with a stub mapper through the real
    ``ComparisonAnalyzer._write_merge_policy_warnings`` so the block
    wording, scoping, and state-prefix registration are exercised without
    a full comparison run.
    """
    from types import SimpleNamespace

    from comparison.comparison_analyzer import ComparisonAnalyzer

    rows = [{
        'source_dataset': 'banc_v626',
        'source_type': 'AVLP614',
        'target_dataset': FAFB,
        'target_type': 'CB1476',
        'total_votes': 1,
        'verified_votes': 1,
        'winner_votes': 1,
    }]

    class StubMapper:
        def alignment_fallback_rows(self, filter_types=None, datasets=None):
            return rows

        def same_name_first_summary(self, filter_types=None, datasets=None):
            return {}

        def multivalue_summary(self, datasets=None):
            return {}

        def get_mapping_decision(self, token, source_ds, target_ds):
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': [], 'conflicts': [], 'support': None}

    captured = {}
    stub = SimpleNamespace(
        parameters=SimpleNamespace(
            _auto_type_mapper=StubMapper(),
            full_output_path=str(tmp_path)),
        _log=lambda *a, **k: None,
        _append_user_warning_notes=(
            lambda path, blocks: captured.__setitem__('blocks', blocks)),
        _replace_state_notes=lambda blocks: blocks,
    )
    ComparisonAnalyzer._write_merge_policy_warnings(
        stub, SimpleNamespace(warnings=[]), ['AVLP614'],
        ['banc_v626', FAFB])
    blocks = captured['blocks']
    text = '\n'.join(blocks)
    assert '[BANC alignment fallback]' in text
    assert 'AVLP614 -> flywire_FAFB_v783 CB1476' in text
    assert '1/1 alignment vote(s), 1 fafb_match-verified' in text
    assert 'custom label mapper' in text


@requires_banc
class TestAlignmentFallbackRealData:
    """2026-09-27 measurements on the local BANC v626 table."""

    def _alignment_mapped(self, mapper, banc_key):
        rows = []
        for (key, column, banc_type), record in sorted(
                mapper._banc_label_votes.items()):
            if key != banc_key or column != ALIGN or not record['votes']:
                continue
            decision = mapper.get_mapping_decision(banc_type, banc_key, FAFB)
            support = decision.get('support') or {}
            if (decision['status'] == 'mapped'
                    and support.get('column') == ALIGN):
                rows.append((banc_type, decision['target_type']))
        return rows

    def test_real_alignment_lane_fills_eighteen_types(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        rows = self._alignment_mapped(mapper, BANC626)
        # 2026-09-27: 18 fct-empty BANC types gain an alignment mapping,
        # every one a genuinely cross-name mapping
        assert len(rows) == 18
        assert all(source != target for source, target in rows)
        assert ('AVLP614', 'CB1476') in rows

    def test_real_curated_winners_keep_their_lane(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        for banc_type, expected in (('ORN_VA1d', 'ORN_VA1v'),
                                    ('ORN_VA5', 'ORN_VA3'),
                                    ('ORN_VC4', 'ORN_VA2')):
            decision = mapper.get_mapping_decision(banc_type, BANC626, FAFB)
            assert decision['status'] == 'mapped', banc_type
            assert decision['target_type'] == expected, banc_type
            assert decision['support']['column'] == 'fafb_cell_type'
        # the curated-label chain outranks every other chain for ORN_VA1d
        chains = mapper.get_type_bridges('ORN_VA1d', BANC626, FAFB,
                                         max_bridges=0)
        ordered = prioritized_bridge_chains(chains, BANC626, FAFB)
        assert any(hop['column'] == 'fafb_cell_type' for hop in ordered[0])

    def test_real_curated_conflicts_stay_fail_closed(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        # Dm19: curated fct votes conflict (no dominant winner) — the
        # alignment lane (whose votes are unanimous Dm19) must not fill it.
        # Dm19 IS a FAFB primary, so a same-name identity mapping exists
        # (annotation overlay, pre-existing) — the assertion is that NO
        # alignment-lane provenance or edge appears for the pair.
        decision = mapper.get_mapping_decision('Dm19', BANC626, FAFB)
        assert decision['status'] == 'conflict'
        origins = {conflict['origin'] for conflict in decision['conflicts']}
        assert origins == {'cross-dataset cell type'}
        assert 'Dm19' in decision['target_types']
        prov = mapper._bridge_provenance.get((BANC626, 'Dm19', FAFB)) or {}
        assert prov.get('column') != ALIGN
        assert not any(edge[2] == ALIGN for edge in
                       mapper._banc_label_edges.get((BANC626, 'Dm19'), []))
        # the unanimous alignment votes remain visible as diagnostics
        votes = mapper._banc_label_votes.get((BANC626, ALIGN, 'Dm19'))
        assert votes and votes['votes'] == {'Dm19': 16}

    def test_real_alignment_chain_derives_both_directions(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        chains = mapper.get_type_bridges('AVLP614', BANC626, FAFB,
                                         max_bridges=0)
        assert chains
        assert any(hop['column'] == ALIGN
                   for chain in chains for hop in chain)
        reverse = mapper.get_type_bridges('CB1476', FAFB, BANC626,
                                          max_bridges=0)
        assert any(hop['column'] == ALIGN
                   for chain in reverse for hop in chain)

    def test_real_cb1011_mappings_unchanged(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        # toward MCNS the three-way curated conflict stays
        mcns = mapper.get_mapping_decision('CB1011', BANC626, MCNS)
        assert mcns['status'] == 'conflict'
        assert set(mcns['target_types']) == {'CB1011', 'CB3252', 'SMP227'}
        # toward FAFB the pre-existing curated rename route stays curated
        fafb = mapper.get_mapping_decision('CB1011', BANC626, FAFB)
        assert fafb['status'] == 'mapped'
        assert fafb['support']['column'] == 'fafb_cell_type'

    @requires_banc_v888
    def test_real_v888_lane_fills_eighteen_types(self):
        mapper = CrossDatasetTypeMapper(verbose=False)
        assert mapper.load()
        rows = self._alignment_mapped(mapper, BANC888)
        assert len(rows) == 18
        assert all(source != target for source, target in rows)
