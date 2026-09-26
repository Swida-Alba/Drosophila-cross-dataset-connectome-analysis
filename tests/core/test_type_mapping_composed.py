"""Unit tests for the Round 2 mapping graph (spec §4/§6/§7).

Synthetic pair flows only — the mapping graph builder's layer naming,
per-component dataset ordering (§10.1), matched-type hovers (§10.2),
pooled label nodes, the scoping cap (§6) and the bridges CSV contract
(§7) are validated without touching the real datasets.
"""

from comparison.mapping_visualization import (
    build_bridges_csv,
    build_composed_mapping_graph,
    render_composed_mapping_html,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v626'


def _flow(s, st, t, tt, sc, fc, origin='type', linkers=True, **extra):
    hops = [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': 'male-cns:v1.0', 'column': 'flywireType', 'value': 'LMTe01'},
        {'dataset': t, 'column': 'type', 'value': tt},
    ] if linkers else [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': t, 'column': 'type', 'value': tt},
    ]
    flow = {'source_dataset': s, 'target_dataset': t, 'source_type': st,
            'foreign_type': tt, 'source_count': sc, 'foreign_count': fc,
            'matched_origin': origin, 'bridges': [hops]}
    flow.update(extra)
    return flow


def test_composed_graph_layers_roles_and_hovers():
    pair_flows = {
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4),
                       _flow(MCNS, 'T2', FAFB, 'T3', 2, 3)],
        (FAFB, BANC): [_flow(FAFB, 'T3', BANC, 'T3', 3, 5),
                       _flow(FAFB, 'T3', BANC, 'T4', 3, 1, linkers=False)],
        (BANC, MCNS): [_flow(BANC, 'T4', MCNS, 'T4', 1, 1)],
    }
    graph, meta = build_composed_mapping_graph(pair_flows)
    # one connected component of three datasets, deterministic order
    assert len(meta['components']) == 1
    order = meta['components'][0]['datasets']
    assert sorted(order) == sorted({MCNS, FAFB, BANC})
    # roles follow the component positions
    roles = {ds: ('source' if i == 0 else
                  'target' if i == len(order) - 1 else 'intermediate')
             for i, ds in enumerate(order)}
    for layer, ds in enumerate(order):
        nid = f'{layer}|{ds}|'
        nodes = [n for n in graph.nodes if str(n).startswith(nid)]
        assert nodes
        for n in nodes:
            assert graph.nodes[n]['node_type'] == roles[ds]
    # hovers list the cross-dataset matches (§10.2), own dataset excluded
    t2 = [n for n, d in graph.nodes(data=True) if d['label'] == 'T2'][0]
    assert 'matched: T3 (FAFB)' in graph.nodes[t2]['title']
    t4 = [n for n, d in graph.nodes(data=True) if d['label'] == 'T4'][0]
    title = graph.nodes[t4]['title']
    assert 'T3' in title and 'T4' in title
    # pair edges carry the maps-via texts
    assert any(d.get('bridge_texts') for _, _, d in graph.edges(data=True))


def test_composed_component_order_prefers_same_name():
    # A shares X and X2 with B (two same-name pairs); B shares only Y
    # with C — §10.1 puts A adjacent to B even though volumes tie.
    pair_flows = {
        ('ds_a', 'ds_b'): [_flow('ds_a', 'X', 'ds_b', 'X', 1, 1),
                           _flow('ds_a', 'X2', 'ds_b', 'X2', 1, 1)],
        ('ds_b', 'ds_c'): [_flow('ds_b', 'Y', 'ds_c', 'Y', 1, 1)],
    }
    _graph, meta = build_composed_mapping_graph(pair_flows)
    assert meta['components'][0]['datasets'] == ['ds_b', 'ds_a', 'ds_c']


def _circadian_pair_flows():
    """FAFB is the one query origin, MCNS and BANC its two targets — the
    1-source/2-target shape the composed view centres.  T1 fans out to
    several target types in both."""
    def _meta(origin_type, origin_count):
        return dict(origin="cell_type · 'circadian_clock'",
                    origin_dataset=FAFB, origin_column='cell_type',
                    origin_value='circadian_clock',
                    origin_label="cell_type · 'circadian_clock'",
                    origin_type=origin_type, origin_count=origin_count)
    return {
        (FAFB, MCNS): [
            _flow(FAFB, 'T1', MCNS, 'M1', 2, 2, **_meta('T1', 2)),
            _flow(FAFB, 'T1', MCNS, 'M2', 2, 3, **_meta('T1', 2)),
            _flow(FAFB, 'T2', MCNS, 'M1', 1, 3, **_meta('T2', 1)),
        ],
        (FAFB, BANC): [
            _flow(FAFB, 'T1', BANC, 'B1', 2, 4, **_meta('T1', 2)),
            _flow(FAFB, 'T2', BANC, 'B2', 1, 5, **_meta('T2', 1)),
        ],
    }


def test_composed_graph_plots_types_only():
    """2026-09-26: the chip that produced the mapping used to render as a
    hub node ('cell_type · circadian_clock') feeding the origin types, so the
    query itself read as one more mapping endpoint.  A type-granularity view
    plots types; the chip stays in the panel tables and the CSVs."""
    graph, _meta = build_composed_mapping_graph(_circadian_pair_flows())
    assert not [n for n in graph.nodes if str(n).startswith('E|')]
    assert not [n for n, d in graph.nodes(data=True)
                if d.get('node_type') == 'entry']
    labels = {str(d.get('label')) for _, d in graph.nodes(data=True)}
    assert 'circadian_clock' not in labels
    assert "cell_type · 'circadian_clock'" not in labels
    # the origin types the entry used to point at are still there, and the
    # target types they fan out to
    assert {'T1', 'T2', 'M1', 'M2', 'B1', 'B2'} <= labels


def test_composed_star_centres_the_origin_dataset():
    """One origin + two targets is not a chain.  Dagre's affinity order read
    it ``FAFB → BANC → MCNS``, which hides that BOTH targets came from ONE
    query; the star puts FAFB in the middle column and keeps its 'source'
    role (layer position must not demote it to 'intermediate')."""
    graph, meta = build_composed_mapping_graph(_circadian_pair_flows())
    assert meta['starred'] is True
    order = meta['components'][0]['datasets']
    assert order[1] == FAFB
    assert set(order) == {FAFB, MCNS, BANC}
    # equal origin-type coverage (T1, T2 reach both) → the dataset key
    # decides, so the layout is deterministic for a given result
    assert order == [BANC, FAFB, MCNS]
    for node, data in graph.nodes(data=True):
        layer = str(node).split('|')[0]
        assert data['node_type'] == (
            'source' if layer == '1' else 'target')
        # the star renders from explicit positions, one column per dataset
        assert data['position']['x'] == int(layer) * 380
        assert isinstance(data['position']['y'], (int, float))
    # each column is stacked with a row gap, not scattered by a force layout
    by_column: Dict[str, list] = {}
    for node, data in graph.nodes(data=True):
        by_column.setdefault(str(node).split('|')[0], []).append(
            data['position']['y'])
    for ys in by_column.values():
        assert len({round(y) for y in ys}) == len(ys)


def test_composed_star_flanks_follow_the_selection_order():
    """The denser half of the picture reads first: the target covering more
    origin types takes the left flank, and the caller's selection order
    breaks a tie."""
    pair_flows = {
        (FAFB, MCNS): [_flow(FAFB, 'T1', MCNS, 'M1', 2, 2),
                       _flow(FAFB, 'T2', MCNS, 'M2', 1, 1)],
        (FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'B1', 2, 4)],
    }
    _graph, meta = build_composed_mapping_graph(pair_flows)
    # MCNS covers two origin types, BANC one → MCNS left regardless of order
    assert meta['components'][0]['datasets'] == [MCNS, FAFB, BANC]
    # a 1-to-1 star (equal coverage) falls back to the selection order
    tied = {
        (FAFB, MCNS): [_flow(FAFB, 'T1', MCNS, 'M1', 2, 2)],
        (FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'B1', 2, 4)],
    }
    _graph, meta = build_composed_mapping_graph(
        tied, dataset_order=[FAFB, BANC, MCNS])
    assert meta['components'][0]['datasets'] == [BANC, FAFB, MCNS]


def test_composed_chain_is_not_a_star():
    """A target that is itself an origin (FAFB → BANC → MCNS) has no single
    centre, so §10.1's affinity order and dagre stay."""
    pair_flows = {
        (FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'T1', 1, 1)],
        (BANC, MCNS): [_flow(BANC, 'T1', MCNS, 'T1', 1, 1)],
    }
    graph, meta = build_composed_mapping_graph(pair_flows)
    assert meta['starred'] is False
    assert not any('position' in d for _, d in graph.nodes(data=True))
    assert meta['components'][0]['star'] is False



def _entry_flows(**overrides):
    """Three circadian_clock flows: T1→M1/M2, T2→M1 with origin metadata."""
    def _meta(origin_type, origin_count):
        base = dict(
            origin="cell_type · 'circadian_clock'",
            origin_dataset=FAFB, origin_column='cell_type',
            origin_value='circadian_clock',
            origin_label="cell_type · 'circadian_clock'",
            origin_type=origin_type, origin_count=origin_count)
        base.update(overrides)
        return base
    return [
        _flow(FAFB, 'T1', MCNS, 'M1', 2, 2, **_meta('T1', 2)),
        _flow(FAFB, 'T1', MCNS, 'M2', 2, 3, **_meta('T1', 2)),
        _flow(FAFB, 'T2', MCNS, 'M1', 1, 3, **_meta('T2', 1)),
    ]


def _no_entry_nodes(graph):
    """The 2026-09-26 contract: a type-level network plots TYPES.  The chip
    that produced them (``matched_origin``) is search provenance and lives in
    the panel tables and the CSVs, never on the canvas."""
    assert not [n for n in graph.nodes if str(n).startswith('E|')]
    assert not [n for n, d in graph.nodes(data=True)
                if d.get('node_type') == 'entry']
    assert not [d for _, d in graph.nodes(data=True)
                if 'circadian_clock' in str(d.get('label', ''))]


def test_pair_network_plots_types_only():
    """§14 as re-cut 2026-09-26: the per-pair type-level network keeps the
    origin-side types the query entry used to point at, but the entry itself
    is gone — and the pair edges are untouched."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(_entry_flows())
    _no_entry_nodes(graph)
    labels = {str(d.get('label')) for _, d in graph.nodes(data=True)}
    assert {'T1', 'T2', 'M1', 'M2'} <= labels
    # the real pair-mapping edges survive unchanged (T1→M1, T1→M2, T2→M1)
    assert {(u, v) for u, v in graph.edges()} == {
        (f'0|{FAFB}|T1', f'1|{MCNS}|M1'),
        (f'0|{FAFB}|T1', f'1|{MCNS}|M2'),
        (f'0|{FAFB}|T2', f'1|{MCNS}|M1')}
    assert any(graph[u][v]['bridge_texts'] for u, v in graph.edges())
    # the origin side stays LEFT
    assert all(str(n).startswith(f'0|{FAFB}|')
               for n in graph.nodes
               if str(n).endswith('|T1') or str(n).endswith('|T2'))


def test_pair_network_legacy_origin_string_adds_no_node():
    """A legacy flow carrying only the matched_origin DISPLAY string (no
    structured origin metadata) is the same case: types, no entry."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(
        [_flow(FAFB, 'T1', MCNS, 'M1', 2, 2,
               origin="cell_type · 'legacy_label'")])
    assert not [n for n in graph.nodes if str(n).startswith('E|')]
    assert {(u, v) for u, v in graph.edges()} == {
        (f'0|{FAFB}|T1', f'1|{MCNS}|M1')}


def test_pair_network_puts_the_origin_side_left():
    """Native-match flows (the viewer's expanded search, and the panel's
    fallback chips) run searched → foreign with the matched column on the
    FOREIGN side.  The entry node is gone, but the orientation rule that was
    introduced to rank it survives: the origin side renders LEFT, so the view
    reads FAFB types → searched MCNS types exactly like the panel's
    origin-seeded exports (user 2026-09-10 layout report)."""
    from comparison.mapping_visualization import build_mapping_network_graph

    def _native_flow(local, foreign_type, count, foreign_count):
        return _flow(
            MCNS, local, FAFB, foreign_type, count, foreign_count,
            origin="cell_type · 'circadian_clock'",
            origin_dataset=FAFB, origin_column='cell_type',
            origin_value='circadian_clock',
            origin_label="cell_type · 'circadian_clock'",
            origin_type=foreign_type, origin_count=foreign_count)

    graph = build_mapping_network_graph([
        _native_flow('M1', 's-CPDN3A', 3, 38),
        _native_flow('M2', 's-CPDN3C', 2, 32),
        _native_flow('M1', 's-CPDN3D', 3, 37),
    ])
    _no_entry_nodes(graph)
    assert {(u, v) for u, v in graph.edges()} == {
        (f'0|{FAFB}|s-CPDN3A', f'1|{MCNS}|M1'),
        (f'0|{FAFB}|s-CPDN3C', f'1|{MCNS}|M2'),
        (f'0|{FAFB}|s-CPDN3D', f'1|{MCNS}|M1')}
    for u, v in graph.edges():
        data = graph[u][v]
        assert data['source_dataset'] == MCNS
        assert data['target_dataset'] == FAFB


def test_pair_network_type_query_has_no_entry():
    """A `type`-column query creates no entry node (§14 guard)."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(
        [_flow(MCNS, 'SMP227', FAFB, 's-CPDN3B', 6, 25)])
    assert not any(d['node_type'] == 'entry'
                   for _, d in graph.nodes(data=True))


def test_composed_cap_hides_same_name_first():
    pair_flows = {}
    for i in range(20):
        a, b = f'S{i}', f'S{i}'
        pair_flows[(MCNS, FAFB)] = pair_flows.get((MCNS, FAFB), []) + [
            _flow(MCNS, a, FAFB, b, 1, 1, linkers=False)]
    pair_flows[(MCNS, FAFB)].append(_flow(MCNS, 'L1', FAFB, 'L1', 4, 4))
    pair_flows[(MCNS, FAFB)].append(_flow(MCNS, 'L2', FAFB, 'L3', 2, 2))
    graph, meta = build_composed_mapping_graph(pair_flows, node_cap=12)
    assert meta['hidden_same_name'] > 0
    assert any('same-name types hidden' in n for n in meta['notes'])
    # the cap holds and linker-bearing types survive
    assert graph.number_of_nodes() <= 12
    remaining_labels = {d['label'] for _, d in graph.nodes(data=True)}
    assert {'L1', 'L2', 'L3'} <= remaining_labels


def test_composed_render_html():
    pair_flows = {
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)],
    }
    html, meta = render_composed_mapping_html(pair_flows)
    assert html and 'cytoscape' in html.lower()
    assert '<title>Mapping graph</title>' in html
    empty, meta2 = render_composed_mapping_html({})
    assert empty is None


def test_bridges_csv_contract():
    flows = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4),
             _flow(MCNS, 'T2, X', FAFB, 'T3', 2, 3, linkers=False)]
    text = build_bridges_csv(flows, pools={
        ('T2, X', 'T3'): {'source_body_ids': [1, 2],
                          'target_body_ids': [3, 4, 5],
                          'source_coverage': 'covered 2 of 2 (100.0%)',
                          'target_coverage': 'covered 2 of 3 (66.7%)',
                          'source_type_body_ids': ['900', '901'],
                          'target_type_body_ids': ['400', '401', '402']}})
    lines = text.strip().splitlines()
    assert lines[0] == (
        'source_dataset,source_entry,matched_column,source_type,'
        'target_dataset,target_type,relationship,source_neurons,'
        'target_neurons,bridge,bridge_columns,mapping_origin,'
        'source_pool,source_total,target_pool,target_total,'
        'source_body_ids,target_body_ids,'
        'pool_coverage,pool_coverage_basis')
    # linker-bearing row: explicit endpoints, matched entry + column
    assert lines[1].startswith(
        'male-cns:v1.0,T1,type,T1,flywire_FAFB_v783,T1,1-to-1,4,4,')
    assert 'flywireType' in lines[1] and ',mapped,' in lines[1]
    # bare same-name row: no linker columns, same-name origin, pool
    # coverage + FULL per-type bodyId populations filled (quoting handles
    # the comma in the type name)
    assert '"T2, X"' in lines[2] and 'same name' in lines[2]
    assert 'source covered 2 of 2 (100.0%); target covered 2 of 3 (66.7%)' in lines[2]
    # bodyIds export as brace-wrapped comma-separated lists: ONE quoted CSV
    # field, so comma-containing values stay intact for spreadsheet readers.
    import csv as _csv
    import io as _io

    rows = list(_csv.reader(_io.StringIO(text)))
    assert len(rows[0]) == 20
    assert rows[2][16] == '{900, 901}'
    assert rows[2][17] == '{400, 401, 402}'
    # a pool without the per-type keys leaves the bodyId cells empty
    assert rows[1][16] == '' and rows[1][17] == ''

    taxonomy_text = build_bridges_csv([
        _flow(MCNS, 'T1', FAFB, 'T1', 4, 4,
              origin="cell_type · 'circadian_clock'")])
    taxonomy_reader = _csv.reader(_io.StringIO(taxonomy_text))
    next(taxonomy_reader)  # header
    taxonomy_row = next(taxonomy_reader)
    assert taxonomy_row[1:4] == ['circadian_clock', 'cell_type', 'T1']
    assert build_bridges_csv([]) is None


def test_bridges_csv_publishes_the_per_branch_out_map():
    """The extended export names the endpoint type's own neurons this row's
    bridge pool does NOT reach — the panel-side out-map the user asked for
    (2026-09-26).  A SET difference, so a `full population` basis reads 0, and
    an unresolvable pool stays an EMPTY cell rather than a false 0.  The
    historical 20-column legacy contract is untouched."""
    import csv as _csv
    import io as _io

    flows = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 5),
             _flow(MCNS, 'T2', FAFB, 'T2', 2, 2),
             _flow(MCNS, 'T3', FAFB, 'T3', 1, 1)]
    text = build_bridges_csv(flows, pools={
        ('T1', 'T1'): {'target_body_ids': [3, 4],
                       'target_type_body_ids': [3, 4, 5, 6, 7]},
        ('T2', 'T2'): {'target_body_ids': [8, 9],
                       'target_type_body_ids': [8, 9]},
    }, extended=True)
    rows = list(_csv.DictReader(_io.StringIO(text)))
    assert len(rows) == 3
    by_type = {r['source_type']: r for r in rows}
    assert by_type['T1']['target_out_map'] == '3'
    assert by_type['T1']['target_out_map_body_ids'] == '{5, 6, 7}'
    # a type pooled on its full population leaves nothing out of map
    assert by_type['T2']['target_out_map'] == '0'
    assert by_type['T2']['target_out_map_body_ids'] == '{}'
    # no pool is UNMEASURED, not zero — the two must not read alike
    assert by_type['T3']['target_out_map'] == ''
    assert by_type['T3']['target_out_map_body_ids'] == ''
    # legacy (default) form keeps its 20 columns exactly
    legacy = list(_csv.reader(_io.StringIO(build_bridges_csv(flows))))
    assert len(legacy[0]) == 20 and 'target_out_map' not in legacy[0]


def test_bridges_csv_relationship_reflects_fan_in():
    """Pair rows label cardinality from BOTH fan directions: two source
    types converging on one target read N-to-1 instead of two 1-to-1 rows
    (user 2026-09-10: 5th-LNv and LNd_CRY+_ITP+ both map to MCNS
    5thsLNv_LNd6); forward fan-out stays 1-to-N and both directions give
    N-to-N."""
    import csv as _csv
    import io as _io

    flows = [
        _flow(MCNS, 'T1', FAFB, 'T1', 4, 4, linkers=False),
        _flow(MCNS, '5th-LNv', FAFB, 'Shared', 2, 4, linkers=False),
        _flow(MCNS, 'LNd_CRY+_ITP+', FAFB, 'Shared', 2, 4, linkers=False),
        _flow(MCNS, 'Splitter', FAFB, 'SplitA', 3, 1, linkers=False),
        _flow(MCNS, 'Splitter', FAFB, 'SplitB', 3, 1, linkers=False),
        _flow(MCNS, 'MeshA', FAFB, 'MeshX', 1, 1, linkers=False),
        _flow(MCNS, 'MeshA', FAFB, 'MeshY', 1, 1, linkers=False),
        _flow(MCNS, 'MeshB', FAFB, 'MeshX', 1, 1, linkers=False),
        _flow(MCNS, 'MeshB', FAFB, 'MeshY', 1, 1, linkers=False),
    ]
    rows = list(_csv.reader(_io.StringIO(build_bridges_csv(flows))))
    relationship = {(r[3], r[5]): r[6] for r in rows[1:]}
    assert relationship[('T1', 'T1')] == '1-to-1'
    assert relationship[('5th-LNv', 'Shared')] == 'N-to-1'
    assert relationship[('LNd_CRY+_ITP+', 'Shared')] == 'N-to-1'
    assert relationship[('Splitter', 'SplitA')] == '1-to-N'
    assert relationship[('Splitter', 'SplitB')] == '1-to-N'
    assert relationship[('MeshA', 'MeshX')] == 'N-to-N'
    assert relationship[('MeshA', 'MeshY')] == 'N-to-N'
    assert relationship[('MeshB', 'MeshX')] == 'N-to-N'
    assert relationship[('MeshB', 'MeshY')] == 'N-to-N'


def test_coverage_forward_rows_label_converging_sources_n_to_1():
    """Forward coverage rows escalate to N-to-1 when the target also
    receives other queried types; pure fan-out keeps 1-to-N."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {
        (MCNS, FAFB): [
            _flow(MCNS, '5th-LNv', FAFB, 'Shared', 2, 4, linkers=False),
            _flow(MCNS, 'LNd_CRY+_ITP+', FAFB, 'Shared', 2, 4,
                  linkers=False),
            _flow(MCNS, 'T1', FAFB, 'T1', 4, 4, linkers=False),
            _flow(MCNS, 'Splitter', FAFB, 'SplitA', 3, 1, linkers=False),
            _flow(MCNS, 'Splitter', FAFB, 'SplitB', 3, 1, linkers=False),
        ],
    }
    coverage = build_type_coverage(pair_flows)
    relationship = {r['type']: r['relationship']
                    for r in coverage['forward']}
    assert relationship['5th-LNv'] == 'N-to-1'
    assert relationship['LNd_CRY+_ITP+'] == 'N-to-1'
    assert relationship['T1'] == '1-to-1'
    assert relationship['Splitter'] == '1-to-N'


def test_zero_count_endpoints_still_render_labeled():
    """A mapped type with 0 rows in the SELECTED table (the BANC v888
    DN1pE report: the name resolves in the mapper's v626-keyed BANC
    namespace but the selected v888 table has none) must still render
    as a labeled node — networkx would otherwise auto-create an
    attribute-less node on add_edge and the vispath renderer falls back
    to the raw ``<layer>|<dataset>|<type>`` id as the label."""
    graph, _meta = build_composed_mapping_graph({
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4,
                             linkers=False)],
        (BANC, FAFB): [_flow(BANC, 'DN1pE', FAFB, 'DN1pE', 0, 4,
                             linkers=False)],
    })
    for nid, data in graph.nodes(data=True):
        assert data.get('label'), nid
    banc_node = next(d for n, d in graph.nodes(data=True)
                     if BANC in n and d.get('label') == 'DN1pE')
    assert '(0 neurons)' in banc_node['title']
    assert 'matched: DN1pE (FAFB)' in banc_node['title']


def test_combined_bridges_csv_uniform_width():
    """All-pairs combined export: one fixed-width schema for every pair —
    the per-pair CSVs are plain header + rows concatenations (the old
    pivoted bridge-<column> fields needed a union-of-columns hack to
    avoid the Tablecruncher ragged-rows bug)."""
    import csv
    import io

    pair_a = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)]                # flywireType linker
    pair_b = [_flow(FAFB, 'T9', BANC, 'T9', 2, 2, linkers=False)]  # bare same-name

    headers = set()
    widths = set()
    for text in (build_bridges_csv(pair_a), build_bridges_csv(pair_b)):
        rows = list(csv.reader(io.StringIO(text)))
        headers.add(tuple(rows[0]))
        widths.update(len(r) for r in rows)
    assert len(headers) == 1
    assert widths == {20}
    assert headers.pop()[:2] == ('source_dataset', 'source_entry')


def test_extended_bridges_csv_exposes_selected_and_all_valid_scopes():
    """The UI export keeps selected evidence and supported alternatives distinct."""
    import csv
    import io

    flow = _flow(MCNS, 'T1', FAFB, 'T1', 4, 4)
    selected = [
        {'dataset': MCNS, 'column': 'type', 'value': 'T1'},
        {'dataset': MCNS, 'column': 'flywireType',
         'value': 'auto:LMTe01'},
        {'dataset': FAFB, 'column': 'type', 'value': 'T1'},
    ]
    alternate = [
        {'dataset': MCNS, 'column': 'type', 'value': 'T1'},
        {'dataset': MCNS, 'column': 'flywireType', 'value': 'LMTe02'},
        {'dataset': FAFB, 'column': 'type', 'value': 'T1'},
    ]
    flow['bridges'] = [selected, alternate]
    flow['mapping_status'] = 'valid_split_evidence'
    pool = {
        'selected_chain': selected,
        'valid_chains': [selected, alternate],
        'selected_chain_rank': 1,
        'valid_chain_count': 2,
        'source_body_ids': ['s1'],
        'target_body_ids': ['t1'],
        'source_type_body_ids': ['s1', 's2', 's3'],
        'target_type_body_ids': ['t1', 't2'],
        'all_valid_source_body_ids': ['s1', 's2'],
        'all_valid_target_body_ids': ['t1', 't2'],
        'source_pool_size': 1,
        'source_type_total': 3,
        'target_pool_size': 1,
        'target_type_total': 2,
        'all_valid_source_pool_size': 2,
        'all_valid_source_type_total': 3,
        'all_valid_target_pool_size': 2,
        'all_valid_target_type_total': 2,
        'source_basis': 'linker rows',
        'target_basis': 'full population',
        'all_valid_source_basis': 'union of supported bridge pools',
        'all_valid_target_basis': 'full population',
        'coverage_basis': 'independent endpoint pools; no bodyId pairing',
        'coverage_scope': 'selected bridge; all valid alternatives retained',
        'all_valid_source_overlap_count': 0,
        'all_valid_target_overlap_count': 0,
        'attempts': [
            {'rank': 0, 'supported': False, 'status': 'unsupported',
             'reason': 'target-side linker rows had no bodyIds'},
        ],
    }
    rows = list(csv.reader(io.StringIO(build_bridges_csv(
        [flow], pools={('T1', 'T1'): pool}, extended=True))))
    header = rows[0]
    row = dict(zip(header, rows[1]))
    assert len(header) == 37
    assert row['mapping_status'] == 'valid_split_evidence'
    assert row['selected_bridge_rank'] == '1'
    assert row['valid_bridge_count'] == '2'
    assert row['selected_linker_values'] == 'auto:LMTe01'
    assert row['selected_linker_canonical_values'] == 'LMTe01'
    assert row['all_valid_source_pool'] == '2'
    assert row['all_valid_target_pool'] == '2'
    assert row['coverage_scope'] == (
        'selected bridge; all valid alternatives retained')
    assert 'target-side linker rows had no bodyIds' in row[
        'unsupported_attempts']


def test_pair_flow_weight_shared_formula():
    """ONE per-pair weight formula for every artifact (user 2026-09-07).

    The network used to duplicate the SOURCE type's whole count onto
    every edge (all edges of a 12-neuron type showed "12"), the linker
    graph fell back foreign-first, and the Sankey used the pooled min —
    three artifacts, three numbers for the same pair."""
    from comparison.mapping_visualization import pair_flow_weight

    # no pool: min of the two sides (source-first fallback when one side
    # is unknown)
    assert pair_flow_weight(
        {'source_count': 12, 'foreign_count': 4}) == 4
    assert pair_flow_weight(
        {'source_count': 2, 'foreign_count': 12}) == 2
    assert pair_flow_weight({'source_count': 0, 'foreign_count': 12}) == 12
    assert pair_flow_weight({'foreign_count': 12}) == 12
    assert pair_flow_weight({}) == 1
    # pool present: the pooled bodyId granularity wins per side
    pool = {'source_body_ids': ['a', 'b'], 'target_body_ids': ['x']}
    assert pair_flow_weight(
        {'source_count': 12, 'foreign_count': 12}, pool) == 1
    # a one-sided pool falls back to that side's neuron count
    one_sided = {'source_body_ids': [], 'target_body_ids': ['x'] * 5}
    assert pair_flow_weight(
        {'source_count': 2, 'foreign_count': 12}, one_sided) == 2


def test_endpoint_pool_counts_union_not_max():
    """N-to-1 targets pool a DIFFERENT disjoint bodyId subset per
    counterpart — the hover count must be the UNION across the pairs,
    not the largest single subset (APDN3 showed "pool 4 bodyIds" next to
    "12 neurons" because CL125/SLP249 pool 4 each and PLP080/SLP250 2
    each)."""
    from comparison.mapping_visualization import _endpoint_pool_counts

    pools = {
        ('CL125', 'APDN3'): {
            'source_body_ids': ['a', 'b', 'c', 'd'],
            'target_body_ids': ['1', '2', '3', '4']},
        ('SLP249', 'APDN3'): {
            'source_body_ids': ['e', 'f', 'g', 'h'],
            'target_body_ids': ['5', '6', '7', '8']},
        ('PLP080', 'APDN3'): {
            'source_body_ids': ['i', 'j'],
            'target_body_ids': ['9', '10']},
        ('SLP250', 'APDN3'): {
            'source_body_ids': ['k', 'l'],
            'target_body_ids': ['11', '12']},
    }
    src, tgt = _endpoint_pool_counts(pools)
    assert tgt == {'APDN3': 12}  # union — the old max said 4
    assert src == {'CL125': 4, 'SLP249': 4, 'PLP080': 2, 'SLP250': 2}
    # overlapping pools count shared bodyIds once
    pools[('DN1', 'APDN3')] = {
        'source_body_ids': ['m'],
        'target_body_ids': ['12', '13']}
    _src, tgt = _endpoint_pool_counts(pools)
    assert tgt['APDN3'] == 13


def test_mapping_pools_are_scoped_by_dataset_direction_and_type():
    """The same type names in opposite directions must not swap coverage."""
    from comparison.mapping_visualization import (
        build_type_coverage,
        get_mapping_pool,
    )

    forward = _flow(FAFB, 'l-LNv', BANC, 'l-LNv', 8, 6)
    reverse = _flow(BANC, 'l-LNv', FAFB, 'l-LNv', 6, 8)
    pair_flows = {(FAFB, BANC): [forward], (BANC, FAFB): [reverse]}
    pools = {
        (FAFB, BANC, 'l-LNv', 'l-LNv'): {
            'source_body_ids': ['f1', 'f2'],
            'target_body_ids': ['b1'],
        },
        (BANC, FAFB, 'l-LNv', 'l-LNv'): {
            'source_body_ids': ['b1', 'b2', 'b3'],
            'target_body_ids': ['f1', 'f2', 'f3', 'f4'],
        },
    }

    assert get_mapping_pool(pools, forward)['source_body_ids'] == ['f1', 'f2']
    assert get_mapping_pool(pools, reverse)['source_body_ids'] == [
        'b1', 'b2', 'b3']

    coverage = build_type_coverage(pair_flows, pools)
    rows = {(row['dataset'], row['type']): row
            for row in coverage['forward']}
    assert rows[(FAFB, 'l-LNv')]['query_cov'] == '2 of 8 (25.0%)'
    assert rows[(FAFB, 'l-LNv')]['target_cov'] == '1 of 6 (16.7%)'
    assert rows[(BANC, 'l-LNv')]['query_cov'] == '3 of 6 (50.0%)'
    assert rows[(BANC, 'l-LNv')]['target_cov'] == '4 of 8 (50.0%)'


def test_type_coverage_forward_1_to_n_and_totals():
    """Forward coverage rows (user 2026-09-07): the queried type's own
    neuron count, its TOTAL mapped number (union of its per-pair pools),
    the relationship label, and per-side x-of-y coverage."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, 'APDN3', MCNS, 'CL125', 12, 4),
        _flow(FAFB, 'APDN3', MCNS, 'PLP080', 12, 2),
        _flow(FAFB, 'APDN3', MCNS, 'SLP249', 12, 4),
        _flow(FAFB, 'APDN3', MCNS, 'SLP250', 12, 2),
    ]}
    pools = {
        ('APDN3', 'CL125'): {'source_body_ids': ['1', '2', '3', '4'],
                             'target_body_ids': ['a', 'b', 'c', 'd']},
        ('APDN3', 'PLP080'): {'source_body_ids': ['5', '6'],
                              'target_body_ids': ['e', 'f']},
        # SLP249 / SLP250 have NO pool entry (unpoolable pair): they
        # contribute nothing to the unions
    }
    coverage = build_type_coverage(pair_flows, pools)
    forward = coverage['forward']
    assert len(forward) == 1
    row = forward[0]
    assert (row['type'], row['dataset'], row['count']) == (
        'APDN3', FAFB, 12)
    assert row['relationship'] == '1-to-N'
    assert row['maps_to'] == 'MCNS: CL125, PLP080, SLP249, SLP250'
    # total mapped number: union of the pooled source-side subsets
    assert row['query_cov'] == '6 of 12 (50.0%)'
    # target-side coverage sums only the pooled pairs
    assert row['target_cov'] == '6 of 6 (100.0%)'

    reverse = coverage['reverse']
    by_type = {r['type']: r for r in reverse}
    assert by_type['CL125']['relationship'] == '1-to-1'
    assert by_type['CL125']['source_cov'] == '4 of 12 (33.3%)'
    assert by_type['CL125']['target_cov'] == '4 of 4 (100.0%)'
    assert by_type['SLP249']['source_cov'] == 'not pooled'


def test_type_coverage_reverse_fanout_label_and_unions():
    """User report (circadian_clock): three FAFB types map onto ONE
    male-cns type.  §12.1 user decision (2026-09-09): relationship cells
    follow the ROW SUBJECT's fan-out, so the backward row reads 1-to-N
    (read from the receiving type back to its sources) — never 1-to-1 —
    and carries both sides' coverage (source-side union vs the receiving
    type's own population)."""
    from comparison.mapping_visualization import build_type_coverage

    sources = [('A', 5, ['s1', 's2', 's3', 's4', 's5']),
               ('B', 7, ['s6', 's7', 's8', 's9', 's10', 's11', 's12']),
               ('C', 2, ['s13', 's14'])]
    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, name, MCNS, 'SMP227', count, 4)
        for name, count, _ids in sources]}
    pools = {
        (name, 'SMP227'): {
            'source_body_ids': ids,
            'target_body_ids': ['t1', 't2', 't3', 't4'] if i == 0 else [],
        }
        for i, (name, _count, ids) in enumerate(sources)}
    reverse = build_type_coverage(pair_flows, pools)['reverse']
    assert len(reverse) == 1
    row = reverse[0]
    assert (row['type'], row['dataset'], row['count']) == (
        'SMP227', MCNS, 4)
    assert row['relationship'] == '1-to-N'
    assert row['sources'] == 3
    assert row['mapped_from'] == 'FAFB: A, B, C'
    # source side: 14 pooled of 14 queried; target side: the union (4)
    # of the receiving type's 4 bodyIds
    assert row['source_cov'] == '14 of 14 (100.0%)'
    assert row['target_cov'] == '4 of 4 (100.0%)'
    # fan-out rows sort first
    assert reverse[0]['relationship'] == '1-to-N'
    # query-scoped rows carry no dataset-wide fields
    assert row['coverage_scope'] == 'query'
    assert 'incoming_source_count' not in row


def test_type_coverage_reverse_dataset_wide_incoming_context():
    """§12.3: a reverse context upgrades the backward row to the
    dataset-wide incoming scope — the full incoming family in
    mapped_from (active query marked), the relationship from the FULL
    family, coverage cells re-measured with the incoming population
    union as the source denominator, and the query-scoped slice
    preserved on query_scope_* fields."""
    from comparison.mapping_visualization import build_type_coverage

    sources = [('A', 5, ['s1', 's2', 's3', 's4', 's5']),
               ('B', 7, ['s6', 's7', 's8', 's9', 's10', 's11', 's12'])]
    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, name, MCNS, 'SMP227', count, 4)
        for name, count, _ids in sources]}
    pools = {
        ('A', 'SMP227'): {
            'source_body_ids': ['s1', 's2', 's3'],
            'target_body_ids': ['t1', 't2'],
        },
    }
    # the dataset-wide family adds source 'C' (not part of the query)
    contexts = {(MCNS, 'SMP227'): {
        'source_dataset': FAFB,
        'target_dataset': MCNS,
        'receiving_type': 'SMP227',
        'receiving_count': 4,
        'sources': [
            {'type': 'A', 'count': 5, 'pooled': True,
             'selected_source_pool_size': 3, 'selected_target_pool_size': 2,
             'all_valid_source_pool_size': 3, 'all_valid_target_pool_size': 2},
            {'type': 'B', 'count': 7, 'pooled': False,
             'selected_source_pool_size': 0, 'selected_target_pool_size': 0,
             'all_valid_source_pool_size': 0, 'all_valid_target_pool_size': 0},
            {'type': 'C', 'count': 8, 'pooled': True,
             'selected_source_pool_size': 2, 'selected_target_pool_size': 1,
             'all_valid_source_pool_size': 4, 'all_valid_target_pool_size': 3},
        ],
        'incoming_source_count': 3,
        'truncated': False,
        'source_population_total': 20,
        'pooled': True,
        'selected_source_union_ids': ['s1', 's2', 's3', 'c1', 'c2'],
        'all_valid_source_union_ids': ['s1', 's2', 's3', 'c1', 'c2',
                                       'c3', 'c4'],
        'selected_target_union_ids': ['t1', 't2', 't3'],
        'all_valid_target_union_ids': ['t1', 't2', 't3', 't4'],
        'source_overlap_selected_ids': [],
        'source_overlap_all_valid_ids': [],
        'target_overlap_selected_ids': [],
        'target_overlap_all_valid_ids': [],
        'selected_source_measured': True,
        'selected_target_measured': True,
        'all_valid_source_measured': True,
        'all_valid_target_measured': True,
    }}
    row = build_type_coverage(
        pair_flows, pools, reverse_contexts=contexts)['reverse'][0]
    assert row['relationship'] == '1-to-N'
    assert row['coverage_scope'] == 'dataset-wide incoming'
    assert row['incoming_source_count'] == 3
    assert row['active_query_sources'] == ['A', 'B']
    assert row['truncated'] is False
    # full incoming family listed (it lives on the SOURCE dataset side);
    # the active query members marked
    assert row['mapped_from'] == (
        'FAFB: A, B, C — active query: A, B')
    # source denominator is the incoming population union (5 + 7 + 8)
    assert row['source_cov_selected'] == '5 of 20 (25.0%)'
    assert row['source_cov'] == '7 of 20 (35.0%)'
    # target side keeps the receiving population as the denominator
    assert row['target_cov_selected'] == '3 of 4 (75.0%)'
    assert row['target_cov'] == '4 of 4 (100.0%)'
    # the query-scoped slice survives on the row
    assert row['query_scope_relationship'] == '1-to-N'
    assert row['query_scope_mapped_from'] == 'FAFB: A, B'
    # query-scope denominator counts only the measured sources (A)
    assert row['query_scope_source_cov_selected'] == '3 of 5 (60.0%)'
    assert 'dataset-wide incoming: 3 source types' in row['coverage_note']


def test_format_coverage_states():
    """The shared one-side coverage formatter: thousands separators, a
    one-decimal share, and the measured-zero vs unmeasured distinction."""
    from comparison.mapping_visualization import format_coverage

    assert format_coverage(1655, 1683) == '1,655 of 1,683 (98.3%)'
    assert format_coverage(2, 8) == '2 of 8 (25.0%)'
    assert format_coverage(4, 4) == '4 of 4 (100.0%)'
    assert format_coverage(0, 168) == '0 of 168 (0.0%)'   # measured zero
    assert format_coverage(0, 0) == '0 of 0'              # degenerate
    assert format_coverage(3, None) == 'not measured'     # side unknown


def test_type_coverage_unmeasured_side_reads_not_measured():
    """A pool whose side's coverage index was unavailable must render
    'not measured' — never a fake '0 of n' measured zero."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, 'T1', MCNS, 'T9', 8, 12, linkers=False)]}
    pools = {
        ('T1', 'T9'): {
            'source_body_ids': [],
            'target_body_ids': [],
            'source_basis': 'unmeasured',
            'target_basis': 'linker rows',
        },
    }
    forward = build_type_coverage(pair_flows, pools)['forward'][0]
    assert forward['query_cov'] == 'not measured'
    # the target side measured a real zero: 0 of 12
    assert forward['target_cov'] == '0 of 12 (0.0%)'


def _star_flows(extra_target=None):
    """FAFB is the one origin; BANC and MCNS its targets.  ``extra_target``
    adds a third dataset FAFB maps into, which the Sankey cannot draw."""
    flows = {
        (FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'B1', 2, 4),
                       _flow(FAFB, 'T2', BANC, 'B2', 1, 5)],
        (FAFB, MCNS): [_flow(FAFB, 'T1', MCNS, 'M1', 2, 2),
                       _flow(FAFB, 'T1', MCNS, 'M2', 2, 3),
                       _flow(FAFB, 'T2', MCNS, 'M1', 1, 3)],
    }
    if extra_target:
        flows[(FAFB, extra_target)] = [
            _flow(FAFB, 'T1', extra_target, 'H1', 2, 2)]
    return flows


def test_composed_sankey_centres_the_origin_and_reverses_the_left_half():
    """2026-09-26: the cross-dataset Sankey draws one column per dataset with
    the query's origin in the middle.  Plotly ranks nodes topologically and
    ignores node.x when a link contradicts it, so the LEFT half's rows run
    target → origin — the drawing's direction only; the mapping stays an
    equivalence and the artifact's note says so."""
    from comparison.mapping_visualization import build_composed_sankey_paths

    rows, info = build_composed_sankey_paths(
        _star_flows(), dataset_order=[FAFB, BANC, MCNS])
    assert info['columns'] == [BANC, FAFB, MCNS]
    assert info['left'] == BANC and info['right'] == MCNS
    left = [names for names, _w in rows if names[0].endswith('· BANC')]
    right = [names for names, _w in rows if names[1].endswith('· MCNS')]
    assert left and right
    assert all(names[1].endswith('· FAFB') for names in left)
    assert all(names[0].endswith('· FAFB') for names in right)
    # every ribbon carries the pooled weight, one value per hop
    assert all(len(w) == len(names) - 1 for names, w in rows)
    # T1/T2 both reach BANC, so nothing falls into the wrong column
    assert info['slipped'] == []


def test_composed_sankey_names_the_types_that_would_slip_a_column():
    """An origin type with no counterpart in the LEFT target has no incoming
    ribbon, so plotly drops it into the leftmost column.  The builder cannot
    prevent that, so it must report it — and the flank rule already picked
    the denser target as the left, so this is the residual case."""
    from comparison.mapping_visualization import build_composed_sankey_paths

    pair_flows = {
        (FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'B1', 2, 4),
                       _flow(FAFB, 'T2', BANC, 'B2', 1, 5),
                       _flow(FAFB, 'T3', BANC, 'B3', 1, 5)],
        (FAFB, MCNS): [_flow(FAFB, 'T4', MCNS, 'M1', 2, 2)],
    }
    rows, info = build_composed_sankey_paths(pair_flows)
    # BANC covers three origin types, MCNS one → BANC takes the left flank,
    # and T4 (which only MCNS names) is the type that will sit beside it
    assert info['columns'] == [BANC, FAFB, MCNS]
    assert info['slipped'] == ['T4']


def test_composed_sankey_gate_is_one_origin_into_one_or_two_targets():
    """A third target has no column, and a chain has no centre — both fall
    back to the per-pair Sankeys rather than drawing something false."""
    from comparison.mapping_visualization import build_composed_sankey_paths

    rows, info = build_composed_sankey_paths(
        _star_flows(extra_target='hemibrain:v1.2.1'))
    assert rows == [] and info is None
    chain = {(FAFB, BANC): [_flow(FAFB, 'T1', BANC, 'T1', 1, 1)],
             (BANC, MCNS): [_flow(BANC, 'T1', MCNS, 'T1', 1, 1)]}
    rows, info = build_composed_sankey_paths(chain)
    assert rows == [] and info is None
    # one target is fine: two columns, no reversal needed
    rows, info = build_composed_sankey_paths(
        {(FAFB, MCNS): _star_flows()[(FAFB, MCNS)]})
    assert info['columns'] == [FAFB, MCNS] and info['left'] is None
    assert all(names[0].endswith('· FAFB') for names, _w in rows)


def test_composed_sankey_html_carries_its_own_wording_and_note():
    from comparison.mapping_visualization import render_composed_sankey_html

    html, info = render_composed_sankey_html(_star_flows())
    assert html and info
    assert 'Type mapping Sankey — BANC | FAFB | MCNS' in html
    assert 'Columns are datasets' in html
    assert 'Synapse' not in html


def test_extended_csv_header_matches_its_own_documentation():
    """The mapping CSV's fixed column set is documented by name in
    `docs/technical/AUTO_TYPE_MAPPING_IMPLEMENTATION.md`, and that list had
    silently fallen behind the code (18 of 35 columns, missing
    `selected_bridge`, `mapping_status`, the whole `all_valid_*` scope and
    `coverage_overlap` / `coverage_scope`).  A doc that names every column
    should not be able to drift from the header it names."""
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[2]
           / 'docs' / 'technical'
           / 'AUTO_TYPE_MAPPING_IMPLEMENTATION.md').read_text(encoding='utf-8')
    start = doc.index('FIXED column set for every pair')
    end = doc.index('so the all-pairs file', start)
    documented = [c.strip() for c in ''.join(
        re.findall(r'`([^`]+)`', doc[start:end])).split(',') if c.strip()]

    header = build_bridges_csv(
        [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)], extended=True).splitlines()[0]
    actual = header.split(',')
    assert actual == documented, (
        f'doc/code drift — only in the doc: {set(documented) - set(actual)}, '
        f'only in the header: {set(actual) - set(documented)}')
    # the legacy (non-extended) export stays a strict prefix-shaped subset
    legacy = build_bridges_csv(
        [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)]).splitlines()[0].split(',')
    assert set(legacy) <= set(actual)
