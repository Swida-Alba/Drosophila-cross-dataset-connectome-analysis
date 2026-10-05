"""Real-data tests for the viewer's expanded cross-dataset alias search.

``collect_alias_matches`` powers the "See available neurons" viewer's
zero-hit panel: the search name is expanded through the auto type mapping
and every alias is counted in the other datasets' *cached* neuron indexes.
These tests need the local indexes (male-cns v1.0, FAFB v783, BANC v626 at
minimum) and are skipped when they are absent.
"""

from pathlib import Path

import pytest

from ui.neuron_index import (
    collect_alias_matches,
    collect_native_type_matches,
    enrich_native_type_matches,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'banc_v626'

REQUIRED_INDEXES = [
    REPO_ROOT / 'neuron_indexes' / dataset_to_folder / 'neuron_index.parquet'
    for dataset_to_folder in (
        'male-cns_v1_0', 'flywire_FAFB_v783', 'banc_v626',
    )
]

CROSSWALK_TABLE = (REPO_ROOT / 'datasets' / 'banc_v626' /
                   'banc_v626_allneurons_neuron_df.csv')

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED_INDEXES)
    or not CROSSWALK_TABLE.exists(),
    reason='cached neuron indexes (male-cns v1.0 / FAFB v783 / BANC v626) '
           'or the banc_v626 crosswalk table not available locally',
)


@pytest.fixture(scope='module')
def mapper():
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    m = CrossDatasetTypeMapper(verbose=False, workspace_path=str(REPO_ROOT))
    assert m.load() is True
    return m


def _entry(matches, dataset):
    for entry in matches:
        if entry['dataset'] == dataset:
            return entry
    return None


def _candidate(entry, name):
    for cand in entry['candidates']:
        if cand['name'] == name:
            return cand
    return None


def test_search_slp249_in_fafb_finds_rename_with_count():
    # The user-facing case: SLP249 does not exist in FAFB v783 (renamed to
    # APDN3); the expanded search must surface the alias with its count.
    matches = collect_alias_matches(FW, 'SLP249')
    fafb = _entry(matches, FW)
    assert fafb is not None and fafb['is_selected'] is True
    renamed = _candidate(fafb, 'APDN3')
    assert renamed is not None
    assert renamed['kind'] == 'renamed'
    # FAFB counts every APDN3 neuron (12), of which 4 carry the SLP249
    # additional-type alias.
    assert renamed['count'] == 12
    assert renamed['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']

    # the local alias still resolves in male-cns v1.0
    mcns = _entry(matches, MCNS)
    assert mcns is not None
    local = _candidate(mcns, 'SLP249')
    assert local is not None and local['kind'] == 'same name'
    assert local['count'] == 4


def test_search_apdn3_in_malecns_offers_group_members():
    # APDN3 aggregates four male-cns types; the selected dataset's entry
    # must offer each member as a searchable group (never merge silently).
    matches = collect_alias_matches(MCNS, 'APDN3')
    mcns = _entry(matches, MCNS)
    assert mcns is not None and mcns['is_selected'] is True
    counts = {c['name']: c['count'] for c in mcns['candidates']}
    # Round-16: the crosswalk's reverse reaches the WHOLE member pool —
    # a licensed type-pool split, no longer an arbitrary 'one of N' pick.
    assert all(c['kind'] == 'splits into' for c in mcns['candidates'])
    assert counts == {'SLP249': 4, 'CL125': 4, 'SLP250': 2, 'PLP080': 2}

    # both local releases know the name natively
    fafb = _entry(matches, FW)
    native = _candidate(fafb, 'APDN3')
    assert native['kind'] == 'same name'
    assert native['count'] == 12
    banc = _entry(matches, BANC)
    # The bucket's curated typing renamed two of the old APDN3 bodies
    # (CL085_a / CL083), so the native count dropped from 8 to 7.
    assert _candidate(banc, 'APDN3')['count'] == 7


def test_search_mdn_across_namespaces():
    # FAFB renamed MDN -> DNp50; BANC still uses MDN natively.
    matches = collect_alias_matches(MCNS, 'MDN')
    fafb = _entry(matches, FW)
    renamed = _candidate(fafb, 'DNp50')
    assert renamed['kind'] == 'renamed'
    assert renamed['count'] == 4
    banc = _entry(matches, BANC)
    same = _candidate(banc, 'MDN')
    assert same['kind'] == 'same name'
    assert same['count'] == 4


def test_selected_dataset_entry_is_sorted_first():
    matches = collect_alias_matches(MCNS, 'APDN3')
    assert matches[0]['dataset'] == MCNS
    assert matches[0]['is_selected'] is True


def test_guards_return_no_matches():
    # numeric (bodyId-style), single char, pattern, empty: the expansion
    # never fires for non-plain-name queries.
    for query in ('123', '7', 'a', 'SLP*', ''):
        assert collect_alias_matches(MCNS, query) == []


# ---------------------------------------------------------------------------
# Comma-joined additional_type(s) cells
# ---------------------------------------------------------------------------

def test_suggestion_pools_split_combined_additional_type_cells():
    from ui.type_suggestions import (
        dataset_aware_suggestions,
        get_dataset_pools,
    )

    pools = get_dataset_pools(FW)
    additional = [v for v, _ in pools['additional_type(s)']]
    # every part of 'vDeltaB, vDeltaC, ...' is offered individually ...
    assert 'vDeltaB' in additional
    assert 'vDeltaG' in additional
    # ... and no combined cell survives as a suggestion value.
    assert not any(',' in v for v in additional)

    suggestions = dataset_aware_suggestions('vDeltaB', [FW], 'auto')
    values = [str(e[0]) for e in suggestions]
    assert values and all(',' not in v for v in values)
    assert 'vDeltaB' in values


def test_viewer_match_groups_split_combined_cells():
    from ui.neuron_index import load_cached_neuron_index, query_neuron_index

    index = load_cached_neuron_index(FW)
    page = query_neuron_index(index, search='vDeltaB', page=1, page_size=10)
    # the neurons are still all found (substring over the joined cells)
    assert page.total == 315
    group_values = [str(g.get('match_value')) for g in page.match_groups]
    assert 'vDeltaB' in group_values
    # a matched value is always one real name, never a joined cell
    assert all(',' not in v for v in group_values)


# ---------------------------------------------------------------------------
# Native (mapper-free) type-name expansion
# ---------------------------------------------------------------------------

def test_native_expansion_finds_dn3_relatives_in_other_datasets():
    # DN3 does not exist anywhere as a standalone type; the expansion finds
    # the name-similar types natively — no auto type mapping involved.
    matches = collect_alias_matches  # noqa: F841  (sibling API sanity)
    from ui.neuron_index import collect_native_type_matches

    native = collect_native_type_matches(MCNS, 'DN3')
    by_ds = {e['dataset']: e for e in native}

    fafb = by_ds[FW]
    fafb_types = {c['name']: c['count'] for c in fafb['types']}
    assert fafb_types['APDN3'] == 12
    assert fafb_types['s-CPDN3A'] == 38
    assert fafb_types['l-CPDN3'] == 2
    assert not any(c['exact'] for c in fafb['types'])
    # sorted by neuron count (descending) within the non-exact tier
    counts = [c['count'] for c in fafb['types']]
    assert counts == sorted(counts, reverse=True)

    banc = by_ds[BANC]
    assert {c['name']: c['count'] for c in banc['types']}['APDN3'] == 7

    # datasets with nothing related are reported as absent (not in the list)
    assert MCNS not in by_ds          # selected dataset is not scanned
    assert 'hemibrain:v1.2.1' not in by_ds


def test_native_expansion_enriches_with_mapped_current_dataset_names():
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'DN3')
    enrich_native_type_matches(native, MCNS)

    by_ds = {e['dataset']: e for e in native}
    apdn3 = next(c for c in by_ds[FW]['types'] if c['name'] == 'APDN3')
    assert apdn3['mapped']['kind'] == 'splits into'
    assert apdn3['mapped']['targets'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']
    # l-CPDN3 maps uniquely to male-cns aMe13
    l_cpdn3 = next(c for c in by_ds[FW]['types'] if c['name'] == 'l-CPDN3')
    assert l_cpdn3['mapped'] == {'kind': 'renamed', 'targets': ['aMe13']}


def test_native_expansion_taxonomy_labels_map_covered_types():
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'circadian')
    enrich_native_type_matches(native, MCNS)
    by_ds = {e['dataset']: e for e in native}

    fafb_labels = {l['label']: l for l in by_ds[FW]['labels']}
    clock = fafb_labels['circadian_clock']
    assert clock['column'] == 'cell_type'
    covered = {t['name']: t for t in clock['types']}
    s_cpdn3a = covered['s-CPDN3A']
    assert s_cpdn3a['mapped']['kind'] == 'splits into'
    # taxonomy labels themselves get no annotation; their covered types do
    assert all(not l.get('mapped') for l in by_ds[FW]['labels'])

    banc_labels = {l['label']: l for l in by_ds[BANC]['labels']}
    neuron = banc_labels['circadian_neuron']
    assert neuron['column'] == 'Class'
    # covered types carry their mapping relation when one exists ...
    kinds = {(t.get('mapped') or {}).get('kind') for t in neuron['types']}
    # BANC fan-outs are evidence-only: the source type is still displayed,
    # but no single BANC target is selected as an accepted mapping.  This is
    # distinct from the valid one-of-N presentation used for ordinary
    # cross-dataset split evidence.
    assert {'same name', 'renamed', 'conflict'} <= kinds
    conflict_types = [
        t for t in neuron['types']
        if (t.get('mapped') or {}).get('kind') == 'conflict'
    ]
    assert conflict_types
    assert all(not (t.get('mapped') or {}).get('targets')
               for t in conflict_types)
    # ... and BANC v888 keeps conflict entries visible with an explicit
    # blocked mapping, so the user is led to inspect them in the other
    # dataset rather than seeing a fabricated target.
    v888 = {e['dataset']: e for e in native}['banc_v888']
    v888_labels = {l['label']: l for l in v888['labels']}
    assert any(
        (t.get('mapped') or {}).get('kind') == 'conflict'
        for t in v888_labels['circadian_neuron']['types']
    )


def test_zero_hit_matches_merges_native_and_mapped_tiers():
    from ui.neuron_index import collect_zero_hit_matches

    result = collect_zero_hit_matches(MCNS, 'DN3')
    # value_mapped/guidance stay empty without matched_values (zero-hit
    # queries carry no match groups to derive them from)
    assert set(result.keys()) == {'native', 'mapped', 'value_mapped',
                                  'guidance'}
    assert result['value_mapped'] == [] and result['guidance'] == []
    assert any(e['types'] for e in result['native'])
    # the mapper knows nothing about DN3: the mapped tier stays empty while
    # the native tier still delivers.
    assert not any(
        e['outcome'] == 'matched' and e['candidates'] for e in result['mapped']
    )


def test_native_expansion_is_mapper_free():
    """The native tier is a pure neuron-index search: its code path must not
    reference the auto type mapping at all."""
    import inspect

    import ui.neuron_index as ni

    for func in (ni.collect_native_type_matches, ni._native_type_matches,
                 ni._native_label_matches, ni._native_label_columns):
        source = inspect.getsource(func)
        # strip the docstring so prose mentions don't count as code usage
        if source.count('"""') >= 2:
            source = '"""'.join(source.split('"""')[2:])
        assert 'mapper' not in source.lower(), func.__name__
        assert 'comparison' not in source, func.__name__


def test_native_matches_preserve_written_case():
    # searching 'apdn3' (lowercase) must report the written form 'APDN3'
    from ui.neuron_index import collect_native_type_matches

    native = collect_native_type_matches(MCNS, 'apdn3', uncapped=True)
    fafb = next(e for e in native if e['dataset'] == FW)
    assert fafb['matched_written'] == 'APDN3'
    apdn3 = next(c for c in fafb['types'] if c['name'] == 'APDN3')
    assert apdn3['matched_written'] == 'APDN3'


def test_native_expansion_bounds_broad_query_all_lists():
    """A broad 2-character query keeps a ranked but finite enrichment feed.

    The query 'CB' matched ~23,000 names across the cached datasets and
    fed every one through the per-item mapper walk — long enough to starve
    the UI websocket (the CB2438 viewer crash).  Ordinary searches now cap
    the all-lists while the '+N more' counts stay truthful; whole real
    families still fit beneath the caps.
    """
    from ui.neuron_index import (
        NATIVE_COVERED_ALL_CAP,
        NATIVE_LABELS_ALL_CAP,
        NATIVE_TYPES_ALL_CAP,
        collect_native_type_matches,
    )

    native = collect_native_type_matches(MCNS, 'CB')
    assert native, 'expected broad CB matches in other cached datasets'
    for entry in native:
        assert len(entry['types_all']) <= NATIVE_TYPES_ALL_CAP
        assert len(entry['labels_all']) <= NATIVE_LABELS_ALL_CAP
        for label in entry['labels_all']:
            assert len(label['covered_all']) <= NATIVE_COVERED_ALL_CAP
        # the hidden tail counts stay truthful beyond the display caps
        assert entry['types_truncated'] >= max(
            0, len(entry['types_all']) - len(entry['types']))
        assert entry['labels_truncated'] >= max(
            0, len(entry['labels_all']) - len(entry['labels']))
    heavy = _entry(native, 'male-cns:v0.9')
    if heavy is not None:
        # v0.9 matches well over a thousand CB types; the cap must
        # actually engage for this broad query, ranked by coverage
        assert len(heavy['types_all']) == NATIVE_TYPES_ALL_CAP
        assert heavy['types_truncated'] > NATIVE_TYPES_ALL_CAP


def test_broad_query_uncapped_export_keeps_everything():
    """Only the explicit uncapped CSV export opts back out of the caps."""
    from ui.neuron_index import (
        NATIVE_TYPES_ALL_CAP,
        collect_native_type_matches,
    )

    native = collect_native_type_matches(MCNS, 'CB', uncapped=True)
    heavy = _entry(native, 'male-cns:v0.9')
    if heavy is not None:
        assert len(heavy['types_all']) > NATIVE_TYPES_ALL_CAP


def test_build_matches_csv_exports_uncapped_entries():
    import csv as csv_module
    import io

    from ui.neuron_index import build_matches_csv

    text = build_matches_csv(MCNS, 'DN3')
    rows = list(csv_module.DictReader(io.StringIO(text)))

    # FAFB's APDN3 entry: full count, one-of-N mapping, per-target origins
    fafb_apdn3 = [
        r for r in rows
        if r['dataset'] == FW and r['entry_kind'] == 'type'
        and r['name'] == 'APDN3'
    ]
    assert fafb_apdn3, rows
    assert fafb_apdn3[0]['neuron_count'] == '12'
    assert fafb_apdn3[0]['mapped_kind'] == 'splits into'
    assert fafb_apdn3[0]['mapped_to'] == 'CL125; PLP080; SLP249; SLP250'
    assert 'additional_type(s)' in fafb_apdn3[0]['map_used']

    # BANC's APDN3 entry exported as well (every dataset, nothing capped)
    banc_apdn3 = [
        r for r in rows
        if r['dataset'] == BANC and r['entry_kind'] == 'type'
        and r['name'] == 'APDN3'
    ]
    assert banc_apdn3 and banc_apdn3[0]['neuron_count'] == '7'


def test_enrich_attaches_mapped_type_names_per_block():
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'DN3')
    enrich_native_type_matches(native, MCNS)
    for entry in native:
        names = entry.get('mapped_type_names')
        assert isinstance(names, list) and names == sorted(names)
        # the set is exactly the union of all annotation targets over the
        # FULL covered lists (covered_all — the display cap must not
        # silently shrink the mapped-type set)
        targets = {
            target
            for c in entry.get('types', []) + [
                t for l in entry.get('labels', [])
                for t in (l.get('covered_all') or l.get('types', []))
            ]
            if c.get('mapped')
            for target in c['mapped']['targets']
        }
        assert set(names) == targets


def test_enrich_maps_full_label_coverage_beyond_display_cap():
    """The l-LNv/s-LNv regression: label covered types beyond the display
    cap (lowercase names sort last) must still transfer to the mapped
    type set of the mapped-type view."""
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'circadian_clock')  # capped
    enrich_native_type_matches(native, MCNS)
    entry = next(e for e in native if e['dataset'] == FW)
    label = entry['labels'][0]
    assert label['types_truncated'] > 0          # display keeps its cap
    assert len(label['covered_all']) > len(label['types'])
    mapped = entry['mapped_type_names']
    assert 'l-LNv' in mapped and 's-LNv' in mapped
    assert len(mapped) > len(label['types'])     # mapped beyond the cap


def test_mapping_origins_real(mapper):
    # male-cns SLP249 <-> FAFB APDN3: linked through FAFB's
    # additional_type(s) column (old name SLP249) and the flywireType
    # crosswalk.
    origins = mapper.get_mapping_origins('SLP249', 'APDN3', FW)
    assert {'source': 'additional_type(s)', 'via': 'SLP249'} in origins
    assert {'source': 'flywireType', 'via': 'SLP249'} in origins

    # same name, primary in both datasets: the type columns match directly
    # (plus the crosswalk carries the identical name).
    origins_same = mapper.get_mapping_origins('MDN', 'MDN', BANC)
    # Since 2026-09-09 flywireType never lands on BANC and MCNS is not a
    # licensed mid for MCNS->BANC, so the crosswalk descriptor is gone;
    # the pair is carried by BANC's curated malecns_cell_type label hop,
    # which now reports its own origin descriptor.
    assert origins_same == [{'source': 'malecns_cell_type', 'via': 'MDN'}]
    # unique rename: FAFB's additional_type(s) column carries the old name
    # 'aMe13' on l-CPDN3 rows, and the male-cns crosswalk carries it too.
    origins_x = mapper.get_mapping_origins('aMe13', 'l-CPDN3', FW)
    assert {'source': 'additional_type(s)', 'via': 'aMe13'} in origins_x
    assert {'source': 'flywireType', 'via': 'aMe13'} in origins_x

    # unknown pair -> nothing
    assert mapper.get_mapping_origins('NoSuchType', 'AlsoMissing', FW) == []


def test_query_neuron_index_types_include_filter():
    from ui.neuron_index import load_cached_neuron_index, query_neuron_index
    import polars as pl

    index = load_cached_neuron_index(MCNS)
    page = query_neuron_index(
        index, types_include=['SLP249', 'MDN'], page=1, page_size=500)
    types_seen = {r['type'] for r in page.rows}
    assert types_seen <= {'SLP249', 'MDN'}
    direct = pl.read_parquet(
        REPO_ROOT / 'neuron_indexes' / 'male-cns_v1_0' /
        'neuron_index.parquet')
    expected = direct.filter(
        pl.col('type').is_in(['SLP249', 'MDN'])).height
    assert page.total == expected == 8
    # pagination works within the filtered set
    page1 = query_neuron_index(
        index, types_include=['SLP249', 'MDN'], page=1, page_size=5)
    assert page1.total == expected
    assert len(page1.rows) == 5
    page2 = query_neuron_index(
        index, types_include=['SLP249', 'MDN'], page=2, page_size=5)
    assert len(page2.rows) == expected - 5
    # unknown types -> empty, not an error
    empty = query_neuron_index(index, types_include=['NoSuchType'], page=1)
    assert empty.total == 0 and empty.rows == []
    # the mapped view exposes type-level selection: one match group per
    # mapped type, with full member/body-id maps for the match panel
    groups = {g['match_value']: g for g in page.match_groups}
    assert set(groups) == {'SLP249', 'MDN'}
    assert groups['SLP249']['body_count'] == 4
    assert groups['SLP249']['match_column'] == 'type'
    assert len(page.match_group_members['SLP249']) == 4
    assert len(page.match_group_body_ids['MDN']) == 4
    # every row carries its own type as the matched value
    assert all(r['match_value'] == r['type'] for r in page.rows)


def test_mapping_visualizations_from_circadian_flows():
    from comparison.mapping_visualization import (
        build_bridge_texts,
        build_mapping_flows,
        build_mapping_network_graph,
        format_bridge,
    )
    from ui.neuron_index import (
        count_type_in_index,
        count_types_in_index,
        load_cached_neuron_index,
    )

    native = collect_native_type_matches(MCNS, 'circadian', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    flows = build_mapping_flows(native, MCNS)
    assert flows
    # name-similarity-only pairs (e.g. SMP532_b -> SMP532b) have no
    # metadata derivation and honestly carry no bridge chains.  Since
    # 2026-09-09 a few more pairs honestly derive chainless: the same-name
    # isolation for unsupported pairs and the primary->alt annotation hops
    # that walked BANC's cross-dataset label tokens are no longer derivable.
    chained = [f for f in flows if f['bridges']]
    assert len(chained) >= len(flows) - 9

    index = load_cached_neuron_index(MCNS)
    source_types = {f['source_type'] for f in flows}
    source_counts = count_types_in_index(index, source_types)
    assert source_counts
    # every flow's source_count is the mapped local type's own neuron
    # count in the current dataset (per-side correctness)
    flows = build_mapping_flows(native, MCNS, source_counts=source_counts)
    assert all(f['source_count'] == source_counts[f['source_type']]
               for f in flows)

    # Network: type-mapping graph, bridges hidden; dagre positions the
    # layers in the renderer — the graph itself carries no custom layer
    # map anymore
    graph = build_mapping_network_graph(flows)
    assert graph.number_of_nodes() > 10
    roles = {d.get('node_type') for _, d in graph.nodes(data=True)}
    assert {'source', 'target'} <= roles
    # 2026-09-26: the query chip is not a node — a type-level view plots
    # types, so the label that produced them stays in the tables and CSVs
    assert 'entry' not in roles
    assert not [n for n in graph.nodes if str(n).startswith('E|')]
    for node, data in graph.nodes(data=True):
        assert 'position' not in data and 'title' in data
        # rendered labels carry display values, no layer|dataset prefixes
        assert '|' not in str(data.get('label', ''))
        # bridge derivation never appears on node hover titles
        assert '[' not in str(data.get('title', ''))

    # the CL125 -> APDN3 mapping is a direct type-level edge; the
    # origin-side presentation draws it APDN3 (FAFB, layer 0) → CL125
    # (MCNS, layer 1), and its weight stays the SOURCE (CL125) type's
    # own neuron count, carried on source_count (correct neuron number)
    cl125_apdn3 = ('0|flywire_FAFB_v783|APDN3', '1|male-cns:v1.0|CL125')
    assert graph.has_edge(*cl125_apdn3)
    edge = graph.edges[cl125_apdn3]
    local = count_type_in_index(index, 'CL125')
    assert local and edge['weight'] == local == edge['source_count']
    assert edge['foreign_count'] > 0
    # §14 + origin-side fix, re-cut 2026-09-26: the origin side of these
    # native-match flows (the FOREIGN dataset, where the matched column
    # lives) is still presented LEFT (layer 0), so the rendered flow reads
    # origin types → searched types exactly like the panel's exports — only
    # now without the entry hub that used to be ranked in front of them.
    assert any(str(n).startswith('0|flywire_FAFB_v783|')
               for n in graph.nodes)
    assert not any(str(n).startswith('E|male-cns:v1.0|')
                   for n in graph.nodes)
    # bridge derivation lives exclusively on the pair edges (the few
    # name-similarity-only pairs are the honest exception)
    pair_edges = [(s, t, d) for s, t, d in graph.edges(data=True)
                  if s.startswith('0|')]
    assert pair_edges
    with_text = [d for _s, _t, d in pair_edges if d['bridge_texts']]
    # 2026-09-09: a few pairs honestly render chainless now — the
    # primary->alt annotation hops that walked BANC's cross-dataset label
    # tokens are no longer derivable (derivation noise).
    assert len(with_text) >= len(pair_edges) - 9

    # bridge text renders the full chain with type-identity endpoints,
    # names glued to their 4-char source, and the annotation hop carrying
    # its cell entry (LMTe01) rather than the reached type
    cl125_flow = next(f for f in chained if f['source_type'] == 'CL125')
    text = format_bridge(cl125_flow['bridges'][0])
    assert text.startswith('CL125[MCNS·type]')
    assert 'LMTe01[MCNS·flywireType]' in text
    assert 'LMTe01[FAFB·additional_type(s)]' in text
    assert text.endswith('APDN3[FAFB·type]')
    assert ' [' not in text
    # a same-name pair chain keeps per-namespace identities (never two
    # identical MCNS hops — the DN1pA hover duplication regression); no
    # uninvolved third dataset (BANC) is offered as a derivation for the
    # MCNS~FAFB pair (pure name-transitivity chains are pruned); and the
    # LEADING chain is the linker-verified one (flywireType/hemibrainType
    # corroboration), not the bare name-equality chain
    dn1pa_texts = [
        format_bridge(c)
        for f in chained if f['foreign_type'] == 'DN1pA'
        for c in f['bridges']]
    assert dn1pa_texts
    assert all(not t.startswith('DN1pA[MCNS·type] → DN1pA[MCNS·type]')
               for t in dn1pa_texts)
    assert all(not 'DN2' in t for t in dn1pa_texts)
    # pure name-transitivity through an uninvolved dataset (MCNS DN1pA →
    # BANC DN1pA → FAFB DN1pA) is pruned — no metadata-free echo chains
    assert not any(
        t == 'DN1pA[MCNS·type] → DN1pA[BANC·type] → DN1pA[FAFB·type]'
        for t in dn1pa_texts)
    # the LEADING chain of the same-name pair verifies it with a metadata
    # linker (flywireType corroboration), not the bare name-equality chain
    lead = format_bridge(next(
        f['bridges'][0] for f in chained
        if f['foreign_type'] == 'DN1pA' and f['source_type'] == 'DN1pA'))
    assert 'flywireType' in lead and lead.endswith('DN1pA[FAFB·type]'), lead
    for f in chained:
        if f['foreign_type'] == f['source_type'] and f['bridges']:
            texts = [format_bridge(c) for c in f['bridges']]
            # a same-name pair with ANY metadata-verification chain must
            # lead with it (bare name-equality chains may exist alongside;
            # pairs with no corroboration anywhere — e.g. CB3508 in BANC
            # — legitimately lead with the bare chain)
            verified = [t for t in texts
                        if '·type]' not in t.split('→')[1]]
            if verified:
                assert '·type]' not in texts[0].split('→')[1], texts[0]
    # multi-bridge pairs keep one formatted chain per entry
    multi = next((f for f in chained if len(f['bridges']) >= 2), None)
    if multi:
        texts = build_bridge_texts(multi['bridges'])
        assert len(texts) == 2 and all('[' in t for t in texts)


def test_matches_csv_writes_bridge_derivation():
    import csv as csv_mod
    import io as io_mod

    from ui.neuron_index import build_matches_csv

    csv_text = build_matches_csv(MCNS, 'apdn3')
    assert csv_text
    rows = list(csv_mod.reader(io_mod.StringIO(csv_text)))
    header = rows[0]
    # uniform schema: every logical row carries exactly the header fields
    assert {len(r) for r in rows} == {len(header)}
    for column in ('foreign_type', 'map_used', 'bridge-additional_type(s)',
                   'bridge-flywireType'):
        assert column in header
    f_type = header.index('foreign_type')
    f_used = header.index('map_used')
    f_bridge_add = header.index('bridge-additional_type(s)')
    f_bridge_fw = header.index('bridge-flywireType')
    # the CL125 -> flywireType 'LMTe01' -> APDN3 chain lands in map_used
    # (standardized linker text) and the bridge columns carry the values
    # (a row aggregates every route to the type: LMTe01; PLP080; ...)
    apdn3 = [r for r in rows[1:] if r[f_type] == 'APDN3']
    assert apdn3
    assert any(
        'flywireType' in r[f_used] and "'LMTe01'" in r[f_used]
        and 'LMTe01' in r[f_bridge_fw] and 'LMTe01' in r[f_bridge_add]
        for r in apdn3)
    # labels are exploded per covered type: one row per foreign type
    assert len({r[f_type] for r in rows[1:] if r[1] == 'label'}) >= 1


def test_pool_bridge_body_ids_apdn3_anchors():
    """The APDN3 bridges pool real bodyIds per linker at bodyId-level
    granularity: 4-to-4 / 4-to-4 / 2-to-2 / 2-to-2, covering all 12."""
    from comparison.cross_dataset_type_mapper import (
        get_type_mapper,
        preferred_bridge_chain,
        standardize_bridge,
    )
    from comparison.mapping_visualization import (
        build_bridge_linker_graph,
        build_mapping_flows,
    )
    from ui.neuron_index import (
        count_type_in_index,
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    # the FAFB APDN3 entry maps to the four male-cns counterpart types
    assert entry and entry.get('mapped_type_names') == [
        'CL125', 'PLP080', 'SLP249', 'SLP250']

    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    indexes = {MCNS: index, FW: foreign_index}
    mapper = get_type_mapper()
    pools = {}

    for source_type in entry['mapped_type_names']:
        chains = [c for c in mapper.get_type_bridges(source_type, MCNS, FW)
                  if c and c[-1]['value'] == 'APDN3']
        if not chains:
            continue
        # pool through the most representative (most direct) chain
        chain = preferred_bridge_chain(chains, MCNS, FW)
        linkers = standardize_bridge(chain, MCNS, FW)
        pool = pool_bridge_body_ids(
            MCNS, FW, linkers, source_type, 'APDN3', indexes=indexes)
        pools[(source_type, 'APDN3')] = pool

    assert len(pools) == 4  # CL125 / PLP080 / SLP249 / SLP250
    granularities = sorted(
        (src, pool['granularity']) for (src, _t), pool in pools.items())
    assert granularities == [
        ('CL125', '4 to 4'), ('PLP080', '2 to 2'),
        ('SLP249', '4 to 4'), ('SLP250', '2 to 2')]
    # total coverage: the four bridges cover all 12 APDN3 bodyIds
    assert sum(len(p['target_body_ids']) for p in pools.values()) == 12
    cl125 = pools[('CL125', 'APDN3')]
    assert cl125['coverage'] == 'covered 4 of 12 (33.3%)'
    # the source pool is the real male-cns bodyIds of CL125
    local = count_type_in_index(index, 'CL125')
    assert len(cl125['source_body_ids']) == local == 4
    # the comma cell 'LMTe01, CL125' matches per entry (no substring noise)
    assert cl125['per_linker'][0]['column'] == 'flywireType'
    assert cl125['per_linker'][1]['column'] == 'additional_type(s)'

    # the linker graph carries the pooled counts on the linker hovers
    flows = build_mapping_flows([entry], MCNS)
    graph = build_bridge_linker_graph(
        flows, source_dataset=MCNS, target_dataset=FW, pools=pools)
    linker_titles = [d['title'] for n, d in graph.nodes(data=True)
                     if d['node_type'] == 'linker']
    assert any('pool: 4 bodyIds' in t for t in linker_titles)


def test_linker_html_carries_column_colors(tmp_path):
    """The linker graph render colors each linker node by its matched
    column (per-node color attribute wins over the node-type palette)."""
    import json
    import re

    from comparison.cross_dataset_type_mapper import (
        get_type_mapper,
        preferred_bridge_chain,
        standardize_bridge,
    )
    from comparison.mapping_visualization import (
        build_mapping_flows,
        write_bridge_linker_html,
    )
    from ui.neuron_index import pool_bridge_body_ids

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    assert entry
    mapper = get_type_mapper()
    flows = build_mapping_flows([entry], MCNS)
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        linkers = standardize_bridge(chain, MCNS, FW)
        pools[(flow['source_type'], flow['foreign_type'])] = (
            pool_bridge_body_ids(
                MCNS, FW, linkers, flow['source_type'],
                flow['foreign_type']))

    out = tmp_path / "linker_paths.html"
    result = write_bridge_linker_html(
        flows, str(out), source_dataset=MCNS, target_dataset=FW,
        pools=pools, open_browser=False)
    assert result and out.exists()

    html = out.read_text()
    # Anchor on the renderer's own `const elements = {` assignment — the
    # offline-embedded vendor JS also contains `nodes:` tokens, and an
    # unanchored regex parses minified library code instead of the graph.
    match = re.search(
        r"const elements = \{\s*nodes:\s*(\[.*?\])\s*,\s*edges:",
        html, re.S)
    assert match, "elements JSON not found in the linker HTML"
    nodes = json.loads(match.group(1))
    by_role = {}
    by_group = {}
    for node in nodes:
        data = node["data"]
        by_role.setdefault(data.get("role") or data.get("node_type", ""),
                           []).append(data.get("color"))
        by_group.setdefault(data.get("group", ""), []).append(data)
    # per-dataset groups (§13): node_type carries the dataset code and
    # every node is tagged with it; the structural role lives in 'role'
    assert set(by_group) == {"MCNS", "FAFB"}
    assert all(d.get("node_type") in ("MCNS", "FAFB")
               for d in by_group["MCNS"] + by_group["FAFB"])
    # linkers colored per column (the explicit per-node color attribute
    # still wins over the group color); endpoints keep the type palette
    assert set(by_role.get("linker", [])) == {"#f59e0b", "#a855f7"}
    assert set(by_role.get("source", [])) == {"#2563eb"}   # MCNS group color
    assert set(by_role.get("target", [])) == {"#16a34a"}   # FAFB group color
    # bridge COLUMN names are never abbreviated (only dataset names use
    # the 4-char abbreviations).  The no-abbreviation check reads the
    # app's OWN payload — header/legend markup plus the elements JSON —
    # because the offline-embedded vendor JS coincidentally contains such
    # substrings (e.g. CoSE's TILING_PADDING).
    for column in ("flywireType", "additional_type(s)"):
        assert column in html
    app_block = re.search(r"const elements = \{.*?\};", html, re.S).group(0)
    app_visible = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    scope = app_visible + app_block
    assert "FLYW" not in scope and "ADDI" not in scope


def test_caps_are_display_only():
    """The display caps (16 type matches / 8 labels / 8 covered types) must
    never change the MAPPING state: capped collection + enrich produces
    exactly the same mapped types, covered lists, and flows as the
    uncapped collection used by the exports."""
    from comparison.mapping_visualization import build_mapping_flows, format_bridge
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    for query in ('circadian', 'circadian_clock', 'DN', 'aMe'):
        capped = collect_native_type_matches(MCNS, query)
        uncapped = collect_native_type_matches(MCNS, query, uncapped=True)
        enrich_native_type_matches(capped, MCNS)
        enrich_native_type_matches(uncapped, MCNS)

        # the display lists stay capped (with truncation flags)
        for entry in capped:
            assert len(entry['types']) <= 16
            assert len(entry['labels']) <= 8
            for label in entry['labels']:
                assert len(label['types']) <= 8
        # ... while the full lists are complete
        for entry_u in uncapped:
            entry_c = next(e for e in capped
                           if e['dataset'] == entry_u['dataset'])
            assert (entry_u.get('types_all') or entry_u.get('types')) == \
                entry_c.get('types_all')
            assert [l['label'] for l in
                    (entry_u.get('labels_all') or [])] == \
                [l['label'] for l in (entry_c.get('labels_all') or [])]
            assert [c['name'] for l in entry_c.get('labels_all', [])
                    for c in l['covered_all']] == \
                [c['name'] for l in entry_u.get('labels', [])
                 for c in l['covered_all']]

        # the mapping state converges: same mapped types, same flows
        flows_c = build_mapping_flows(capped, MCNS)
        flows_u = build_mapping_flows(uncapped, MCNS)
        pairs_c = sorted(
            (f['source_type'], f['target_dataset'], f['foreign_type'])
            for f in flows_c)
        pairs_u = sorted(
            (f['source_type'], f['target_dataset'], f['foreign_type'])
            for f in flows_u)
        assert pairs_c == pairs_u, query
        texts_c = sorted(format_bridge(c) for f in flows_c
                         for c in f['bridges'])
        texts_u = sorted(format_bridge(c) for f in flows_u
                         for c in f['bridges'])
        assert texts_c == texts_u, query


def test_asymmetric_same_name_carries_suggested_check_note():
    """User directive 2026-09-14: a same-name candidate whose population
    is an order of magnitude below the same name elsewhere carries the
    extreme-asymmetry suggested-check note (TmY18: 1367 MCNS vs 1 FAFB);
    symmetric names carry none."""
    matches = collect_alias_matches('flywire_FAFB_v783', 'TmY18')
    notes = []
    for entry in matches:
        for cand in entry['candidates']:
            if cand['kind'] == 'same name':
                if entry['dataset'] == 'male-cns:v1.0':
                    assert '1367' in (cand.get('asymmetry_note') or '')
                    notes.append(cand['asymmetry_note'])
    assert notes, 'expected the MCNS same-name candidate to carry the note'
    assert all('suggested check' in n for n in notes)
    # symmetric control: MDN carries no asymmetry note
    for entry in collect_alias_matches('banc_v888', 'MDN'):
        for cand in entry['candidates']:
            if cand['kind'] == 'same name':
                assert not cand.get('asymmetry_note')
