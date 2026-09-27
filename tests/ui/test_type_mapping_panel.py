"""UI tests for the Round 2 Type Mapping entrance (spec §3/§10).

Drives the real entrance button + dialog for male-cns v1.0 + FAFB v783
with NiceGUI's test client: the button is enabled when both datasets have
cached indexes, the global search composes the mapping across the
selection, and the mapping graph + per-pair cards + CSV actions appear.
Needs the local cached indexes; skipped when they are absent.
"""

from pathlib import Path

import pytest

import ui.history_store as hs
import ui.type_mapping_history as tmh
from ui.neuron_index import clear_neuron_index_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_INDEX = REPO_ROOT / 'neuron_indexes' / 'male-cns_v1_0' / 'neuron_index.parquet'
FAFB_INDEX = REPO_ROOT / 'neuron_indexes' / 'flywire_FAFB_v783' / 'neuron_index.parquet'
MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v888'

pytestmark = pytest.mark.skipif(
    not (MCNS_INDEX.exists() and FAFB_INDEX.exists()),
    reason='cached male-cns v1.0 / FAFB v783 neuron indexes not available locally',
)


@pytest.fixture
def panel_client(tmp_path, monkeypatch):
    from nicegui import Client, ui
    from nicegui.page import page

    # the panel's query box keeps its own history store; isolate both it
    # and the shared neuron history so tests never touch the real files
    monkeypatch.setattr(tmh, "_HISTORY_PATH",
                        tmp_path / "type_mapping_history.json")
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")

    clear_neuron_index_cache()
    # Warm the type mapper up front: its first load takes a minute and
    # would otherwise race the panel assertions below.
    from comparison.cross_dataset_type_mapper import get_type_mapper

    assert get_type_mapper().load() is True

    client = Client(page('/type-mapping-panel-test'))
    with client:
        selection = {'value': [MCNS, FAFB]}
        from ui.components.type_mapping_panel import create_type_mapping_entry
        button = create_type_mapping_entry(lambda: list(selection['value']))
    try:
        yield client, button, selection
    finally:
        clear_neuron_index_cache()


def _buttons(client):
    return [e for e in client.elements.values()
            if type(e).__name__ == 'Button']


def _invoke(handler, client, element):
    """Run a click handler, driving any returned coroutine to completion.

    The loading-notice search handler is async (run.io_bound off the
    event loop) — the results only exist after it finishes, so the
    assertions below need the coroutine fully awaited.
    """
    import asyncio
    import inspect

    args = {'sender': element.id, 'client': client, 'args': None}
    # NiceGUI wraps the user handler (lambda e: handle_event(user, e));
    # unwrap it so the click actually EXECUTES here instead of being
    # deferred without a running app loop.
    target = handler
    for cell in (getattr(handler, '__closure__', None) or ()): 
        candidate = cell.cell_contents
        if (callable(candidate) and candidate is not handler
                and 'nicegui' not in getattr(
                    candidate, '__module__', 'nicegui')):
            target = candidate
            break
    try:
        result = target(args)
    except TypeError:
        result = target()
    if inspect.isawaitable(result):
        # drive the coroutine inside the sender's parent slot: NiceGUI
        # resolves the client for ui.notify/UI creation from the current
        # TASK's slot stack, which a fresh loop's task does not carry
        slot = getattr(element, 'parent_slot', None)

        async def _drive():
            if slot is not None:
                with slot:
                    await result
            else:
                await result

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_drive())
        finally:
            loop.close()


def _click_button(client, text_part: str) -> bool:
    for element in client.elements.values():
        if type(element).__name__ != 'Button':
            continue
        if text_part not in str(getattr(element, 'text', '')):
            continue
        for listener in element._event_listeners.values():
            if listener.type == 'click' and listener.handler:
                _invoke(listener.handler, client, element)
                return True
    return False


def _labels(client):
    return [
        str(element.text)
        for element in client.elements.values()
        if isinstance(getattr(element, 'text', None), str)
        and getattr(element, 'text', '')
    ]


def test_button_enabled_with_two_cached_datasets(panel_client):
    client, button, selection = panel_client
    mapping_buttons = [b for b in _buttons(client)
                       if 'Type Mapping' in str(getattr(b, 'text', ''))]
    assert mapping_buttons, 'Type Mapping entrance button not found'
    assert mapping_buttons[0].enabled


def test_button_disabled_below_two_datasets():
    """No cached-index requirement met: the entrance stays disabled."""
    from nicegui import Client, ui
    from nicegui.page import page

    clear_neuron_index_cache()
    client = Client(page('/type-mapping-panel-test-empty'))
    with client:
        from ui.components.type_mapping_panel import create_type_mapping_entry
        create_type_mapping_entry(lambda: [])
    buttons = [b for b in _buttons(client)
               if 'Type Mapping' in str(getattr(b, 'text', ''))]
    assert buttons
    assert not buttons[0].enabled


def test_global_search_composes_the_selection(panel_client):
    client, button, _selection = panel_client
    search = button.search_container
    # the standard filter-mode control rides on the chip input (§12)
    assert getattr(search, "filter_mode", None) is not None
    search.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')
    labels = _labels(client)
    # the mapping graph + per-pair card for the selection appear; the
    # exact chip resolves in FAFB, so the origin-seeded pair runs
    # FAFB -> male-cns (§12 direction rule: the query lives where it
    # matched)
    assert any('Mapping graph (HTML)' in label for label in labels) or \
        any('Mapping graph' in b_text for b_text in
            [str(getattr(b, 'text', '')) for b in _buttons(client)])
    assert any('flywire_FAFB_v783 → male-cns:v1.0' in label
               for label in labels)
    # bidirectional type-coverage presentation (user 2026-09-07): a
    # TOP-LEVEL panel per dataset pair, with the pair named in its title
    assert any(
        'Type coverage — flywire_FAFB_v783 → male-cns:v1.0' in label
        and 'bidirectional' in label
        for label in labels)
    # 2026-09-09: the second view is "Backward" (not "Reverse"), and the
    # coverage columns carry SHORT `CODE · scope` labels with the full
    # dataset key and the selected/all-valid explanation on the header
    # tooltip (§12.4/§12.5)
    assert any(label.startswith('Backward —') for label in labels)
    assert not any(label.startswith('Reverse —') for label in labels)
    coverage_tables = [
        e for e in client.elements.values()
        if type(e).__name__ == 'Table'
        and any('· selected' in c.get('label', '')
                for c in e._props.get('columns', []))]
    assert coverage_tables, 'short coverage column labels missing'
    for table in coverage_tables:
        by_label = {c['label']: c for c in table._props['columns']}
        assert {'FAFB · selected', 'FAFB · all valid',
                'MCNS · selected', 'MCNS · all valid'} <= set(by_label)
        assert 'flywire_FAFB_v783 side (bodyIds)' in (
            by_label['FAFB · selected'].get('tooltip', ''))
        assert 'not a biological adjudication' in (
            by_label['FAFB · selected'].get('tooltip', ''))
        assert 'male-cns:v1.0 side (bodyIds)' in (
            by_label['MCNS · all valid'].get('tooltip', ''))
        assert 'not mutually exclusive' in (
            by_label['MCNS · all valid'].get('tooltip', ''))
    # §12.3: the backward rows carry the dataset-wide incoming scope —
    # the full incoming family with the active query marked
    backward_rows = [
        r for table in coverage_tables for r in table._props.get('rows', [])
        if 'mapped_from' in r]
    assert backward_rows
    assert any(r.get('coverage_scope') == 'dataset-wide incoming'
               for r in backward_rows)
    assert all(r.get('relationship') != '1-to-1'
               for r in backward_rows
               if int(r.get('incoming_source_count') or 1) > 1)
    # artifact + CSV actions of the per-pair card are present
    for action in ('Sankey (type-level)', 'Sankey (linker)',
                   'Network (type-level)', 'Network (linker)',
                   'Export mapping'):
        assert any(action in str(getattr(b, 'text', ''))
                   for b in _buttons(client)), action
    # the confirmed search is recorded in the panel's OWN history store —
    # never in the shared neuron-query history of the analysis tabs
    assert tmh.recent() == ['APDN3']
    assert tmh.datasets_of('APDN3') == sorted([MCNS, FAFB])
    assert hs.recent() == []


def test_failed_search_never_records_history(panel_client):
    """A zero-hit query matches no type, so nothing lands in the store."""
    client, button, _selection = panel_client
    search = button.search_container
    search.add_values(['zzz_no_such_type_zzz'])
    assert _click_button(client, 'Search mappings')
    assert tmh.recent() == []
    assert hs.recent() == []


def test_history_rows_show_dataset_and_column_hints(panel_client):
    """The panel's Recent list annotates rows like its suggestions do."""
    client, button, _selection = panel_client
    search = button.search_container
    search.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')

    # refocus the empty editor: the panel's own Recent list carries the
    # confirmed query with the suggestion-style gray hint
    focus = next(
        listener for listener in search.chip_input._event_listeners.values()
        if listener.type == 'focus'
    )
    search.chip_input._handle_event({'listener_id': focus.id, 'args': None})
    labels = _labels(client)
    assert 'APDN3' in labels
    # APDN3 lives in FAFB's type column only, so the hint is exactly the
    # matched column · dataset pair its suggestion row would show
    assert 'type · flywire_FAFB_v783' in labels


def test_empty_search_produces_no_results(panel_client):
    client, button, _selection = panel_client
    assert _click_button(client, 'Search mappings')
    # no mapping graph and no per-pair cards without a search
    assert not any('Mapping graph (HTML)' in str(getattr(b, 'text', ''))
                   for b in _buttons(client))
    assert not any('mapped pairs' in label for label in _labels(client))


def _summary_by_dataset(outcome):
    return {row['dataset']: row for row in outcome['summary']}


def test_cb4091_stale_crosswalk_claim_is_not_counted_as_mapped(panel_client):
    """2026-09-12: male-cns CB4091 carries a flywireType crosswalk claim
    (CB4091) that flywire_FAFB_v783 cannot fulfil — no such neurons exist
    there.  The claim must not inflate FAFB's mapped counts: only target
    types with actual neurons count as mapped, and the unfulfilled claim
    is reported on the orphan entry instead."""
    from ui.components.type_mapping_panel import _compute_type_mapping

    outcome = _compute_type_mapping(['CB4091'], [MCNS, FAFB], 'exact')
    rows = _summary_by_dataset(outcome)
    mcns, fafb = rows[MCNS], rows[FAFB]
    # the type lives (20 neurons) in male-cns only — its issued side is
    # the Matched types / Neurons pair (the old issued column was always
    # equal to Neurons and is gone)
    assert mcns['types'] == 1 and mcns['neurons'] == 20
    # nothing is realized: no flows, no received counterpart types
    assert not outcome['pair_flows']
    assert mcns['mapped_types'] == 0 and mcns['mapped_neurons'] == 0
    assert mcns['mapped'] == '0'
    # §three-tier (2026-09-27): the summary gains the reach/disclosure
    # fields — additive zeros here (no flows at all, so no evidence reach
    # beyond the empty claim either)
    assert fafb == {'dataset': FAFB, 'types': 0, 'neurons': 0,
                    'mapped_types': 0, 'mapped_neurons': 0, 'mapped': '0',
                    'reach_types': 0, 'reach_neurons': 0, 'reach': '0',
                    'disclosure_types': 0, 'disclosure_bodies': 0,
                    'disclosure_detail': [],
                    # an unfulfilled claim fabricates neither a mapped count
                    # NOR an out-map overhang — FAFB has no neurons of the
                    # type, so there is nothing left out of the map either
                    'out_map': 0,
                    'unmapped': 0}
    # the orphan explains WHY: the claim names a type FAFB does not have
    entries = outcome['orphans'].get((MCNS, FAFB)) or []
    assert [e['type'] for e in entries] == ['CB4091']
    assert entries[0]['count'] == 20
    assert 'CB4091' in entries[0]['claimed']


def test_slp249_rename_still_counts_as_mapped(panel_client):
    """Positive control for the realized-vs-claimed split: male-cns SLP249
    resolves to FAFB APDN3 (a real primary with neurons), so FAFB keeps a
    nonzero mapped count and no orphan is recorded."""
    from ui.components.type_mapping_panel import _compute_type_mapping

    outcome = _compute_type_mapping(['SLP249'], [MCNS, FAFB], 'exact')
    rows = _summary_by_dataset(outcome)
    mcns, fafb = rows[MCNS], rows[FAFB]
    assert mcns['types'] == 1 and mcns['neurons'] > 0
    assert fafb['mapped_types'] == 1 and fafb['mapped_neurons'] > 0
    # 1 received type → the combined cell is the plain neuron count
    assert fafb['mapped'] == str(fafb['mapped_neurons'])
    assert outcome['pair_flows']
    assert not outcome['orphans']


def test_orphan_claim_is_explained_in_the_expander(panel_client):
    """The rendered orphan line names the unfulfilled crosswalk claim, and
    the summary headers carry tooltips so the columns are self-explanatory."""
    client, button, _selection = panel_client
    search = button.search_container
    search.add_values(['CB4091'])
    assert _click_button(client, 'Search mappings')
    labels = _labels(client)
    assert any("auto-mapping claims 'CB4091' but flywire_FAFB_v783 "
               "has no such neurons" in label for label in labels)
    assert any(label.startswith(
        'Orphan types — no mapped counterpart (1)') for label in labels)
    summary_tables = [
        e for e in client.elements.values()
        if type(e).__name__ == 'Table'
        and any(c.get('label') == 'Mapped neurons'
                for c in e._props.get('columns', []))]
    assert summary_tables, 'per-dataset summary table missing'
    by_label = {c['label']: c
                for c in summary_tables[0]._props['columns']}
    assert 'no neurons here does not count' in by_label['Mapped neurons'][
        'tooltip']
    # the out-map column renders beside the claim it complements, and its own
    # hover says which population it is — the panel-side overhang, not the
    # validation pipeline's `family` bin (219 − 204 = 15 vs 11 rows)
    assert 'Out-map (in-map types)' in by_label
    out_map_tip = by_label['Out-map (in-map types)']['tooltip']
    assert '219' in out_map_tip and 'family' in out_map_tip
    assert 'does not reach' in out_map_tip
    assert 'no realized counterpart' in by_label['Unmapped (orphans)'][
        'tooltip']


# ---------------------------------------------------------------------------
# R4/R5: bodyId-level split resolution + per-type breakdown
# (plan-bodyid-level-granularity-in-type-mapper.md §3b.1/§3b.3)
# ---------------------------------------------------------------------------

def test_single_type_preview_has_no_per_type_expansion(panel_client):
    client, button, _selection = panel_client
    button.search_container.add_values(['SLP249'])
    assert _click_button(client, 'Search mappings')
    expansions = [e for e in client.elements.values()
                  if type(e).__name__ == 'Expansion']
    assert not any('Per-type breakdown' in str(getattr(e, 'text', ''))
                   for e in expansions)


def test_the_branch_bodyid_export_publishes_the_out_map(panel_client,
                                                        monkeypatch):
    """`mapping_branch_bodyids_*.csv` is assembled inside a closure no other
    test reached, so its header/row alignment was carried on trust. Capture
    the download and check the two out-map columns against the pool the row
    itself names — a count that disagrees with its own brace cell is the
    failure mode this pins."""
    import csv as _csv
    import io as _io

    from nicegui import ui

    client, button, _sel = panel_client
    captured = {}
    monkeypatch.setattr(
        ui.download, 'content',
        lambda text, name, mime='text/plain', **kw: captured.update(
            name=name, text=text))
    button.search_container.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')
    export = next((el for el in client.elements.values()
                   if type(el).__name__ == 'Button'
                   and 'Export branch bodyIds' in str(getattr(el, 'text', ''))),
                  None)
    assert export is not None, 'no branch-bodyId export button rendered'
    # click it IN THE BUTTON'S OWN SLOT: the handler ends in push_banner,
    # which needs a live slot — the generic `_click_button` helper only
    # re-enters a slot for async handlers, and a zero-arg lambda is not one
    with (export.parent_slot or client):
        for listener in export._event_listeners.values():
            if listener.type == 'click' and listener.handler:
                _invoke(listener.handler, client, export)
                break
    assert str(captured.get('name', '')).startswith(
        'mapping_branch_bodyids_'), captured.get('name')
    rows = list(_csv.DictReader(_io.StringIO(captured['text'])))
    assert rows, 'the branch export wrote no rows'
    assert {'target_out_map', 'target_out_map_body_ids'} <= set(rows[0])
    for r in rows:
        cell = (r['target_out_map_body_ids'] or '').strip()
        ids = [s for s in cell.strip('{}').split(', ') if cell and s]
        assert len(ids) == int(r['target_out_map']), r
        # the pools are int-keyed; a stray whitespace or dtype leak would
        # surface here as a cell that no longer parses
        assert all(s.isdigit() for s in ids), r


def test_multi_type_preview_shows_collapsed_per_type_expansion(panel_client):
    client, button, _selection = panel_client
    button.search_container.add_values(['5thsLNv_LNd6', 'SLP249'])
    assert _click_button(client, 'Search mappings')
    expansions = [e for e in client.elements.values()
                  if type(e).__name__ == 'Expansion']
    assert any('Per-type breakdown' in str(getattr(e, 'text', ''))
               for e in expansions)
    # the breakdown publishes the same column at its PER-BRANCH grain
    breakdown = [t for t in _tables(client)
                 if 'Out-map (in-map types)' in _column_labels(t)
                 and 'Matched type' in _column_labels(t)]
    assert breakdown, 'the per-type breakdown lacks the Out-map column'


def test_per_type_breakdown_rows_are_dataset_specific(panel_client):
    """2026-09-14: every breakdown row is one (matched type, target
    dataset) pair and its mapped counts are THAT target dataset's own
    claim set — never a sum across the target datasets.  The combined
    'mapped' cell reads '{N}({m} types)' from 2 mapped types up and the
    plain '{N}' for a single mapped type."""
    from ui.components.type_mapping_panel import (
        _compute_type_mapping,
        _format_mapped_neurons,
    )

    outcome = _compute_type_mapping(['SLP249'], [MCNS, FAFB], 'exact')
    rows = [r for r in outcome['summary_per_type']
            if r['type'] == 'SLP249']
    assert [(r['dataset'], r['target']) for r in rows] == [(MCNS, FAFB)]
    row = rows[0]
    assert row['mapped_neurons'] > 0
    assert row['mapped'] == str(row['mapped_neurons'])

    # CB2572 resolves both ways after the mirror dedupe (the split
    # MCNS→FAFB derivation direction, plus FAFB CB2572 → MCNS SMP352) —
    # the two rows must each carry their OWN target dataset's claim set
    outcome = _compute_type_mapping(['CB2572'], [MCNS, FAFB], 'exact')
    rows = outcome['summary_per_type']
    assert {(r['dataset'], r['target']) for r in rows} == {
        (MCNS, FAFB), (FAFB, MCNS)}
    by_target = {(r['dataset'], r['target']): r for r in rows}
    assert by_target[(MCNS, FAFB)]['mapped_neurons'] > 0
    # per-row format invariant: '{N}({m} types)' from 2 types up
    for r in rows:
        assert r['mapped'] == _format_mapped_neurons(
            r['mapped_neurons'], r['mapped_types'])


def test_format_mapped_neurons_combined_cell():
    """The combined mapped cell: type breadth shows from 2 types up."""
    from ui.components.type_mapping_panel import _format_mapped_neurons

    assert _format_mapped_neurons(204, 40) == '204(40 types)'
    assert _format_mapped_neurons(6, 2) == '6(2 types)'
    assert _format_mapped_neurons(9, 1) == '9'
    assert _format_mapped_neurons(0, 0) == '0'


# ---------------------------------------------------------------------------
# Suspects disclosure + boundary hygiene (plan-ui-type-mapper-alignment)
# ---------------------------------------------------------------------------

def test_flow_status_carries_same_name_first_fields():
    """§3: the panel's flow record must carry the mapper's own suspects
    verdict — the UI never re-derives it."""
    from comparison.cross_dataset_type_mapper import get_type_mapper
    from comparison.type_resolver import resolve_flow_status

    mapper = get_type_mapper()
    assert mapper.load() is True
    # a FIRED selection (broad, framing B)
    status, fields = resolve_flow_status(mapper, 'aMe9', MCNS, BANC)
    assert status == 'mapped'
    assert fields['suspects'] is True
    assert fields['suspect_rivals'], 'rivals must be carried for the expander'
    assert fields['same_name_first']['disposition'] == 'broad_selection'
    # a NON-fan-out pair keeps the fields at their empty defaults
    _s2, f2 = resolve_flow_status(mapper, 'CB4091', MCNS, FAFB)
    assert f2['suspects'] is False and f2['suspect_rivals'] == []


def test_held_pair_orphan_reason_is_boundary_clean():
    """§4.2: a HELD fan-out must explain itself; wording states the
    observation and never a verdict (no 'duplicate'/'confirmed')."""
    from comparison.cross_dataset_type_mapper import get_type_mapper
    from ui.components.type_mapping_panel import _held_same_name_reason

    mapper = get_type_mapper()
    assert mapper.load() is True
    reason = _held_same_name_reason(mapper, 'ORN_D', BANC, MCNS)
    assert reason, 'ORN_D is the gated_held flagship case'
    assert 'kept unmapped' in reason
    assert '1-to-1' in reason
    for banned in ('duplicate', 'confirmed', 'gated'):
        assert banned not in reason.lower(), banned


def test_multivalue_marker_format():
    """§4.3: the marker names the parts and nothing else."""
    from comparison.cross_dataset_type_mapper import get_type_mapper
    from ui.components.type_mapping_panel import _multivalue_marker

    mapper = get_type_mapper()
    assert mapper.load() is True
    mark = _multivalue_marker(mapper, 'LAL173,LAL174', BANC)
    assert mark.strip().startswith('🧩 multi')
    assert 'LAL173' in mark and 'LAL174' in mark
    assert _multivalue_marker(mapper, 'KCg-d', MCNS) == ''


def test_suspect_detail_rows_shape():
    """§4.1: the collapsed block's rows carry the per-rival facts, sourced
    from the mapper's evidence record (never re-derived).

    ``_suspect_detail_rows`` is a closure inside ``create_type_mapping_entry``
    (like its sibling renderers), so it is exercised through the mapper's
    evidence record the closure reads — the closure itself is covered by the
    panel_client tests above.
    """
    from comparison.cross_dataset_type_mapper import get_type_mapper

    mapper = get_type_mapper()
    assert mapper.load() is True
    detail = mapper.get_same_name_conflict_detail('aMe9', MCNS, BANC)
    assert detail is not None and detail['fires'] is True
    ev = detail['rival_evidence']
    assert ev and ev[0]['rival'] == 'aMe12'
    assert set(ev[0]) >= {'rival', 'rival_has_own_clean_pair',
                          'reverse_target', 'rival_pair_status',
                          'votes', 'verified_votes', 'auto_votes',
                          'population_source', 'population_target'}
    assert ev[0]['rival_pair_status'] in ('own_1to1_pair',
                                           'no_own_1to1_pair')
    # the boundary fix (D6): no verdict words anywhere in the record's keys
    assert 'suspected_duplicate' not in ev[0]


def test_panel_flows_carry_the_suspects_fields():
    """§4.1 regression: the panel's row marker / collapsed expander read the
    FLOW record, so `build_mapping_flows` must copy the same-name-first
    fields from the resolved decision.  (Found 2026-09-18: only
    status/relationship/target_types/conflicts were copied, so the marker
    could never fire despite the resolver carrying the flag.)"""
    from ui.components.type_mapping_panel import _compute_type_mapping

    out = _compute_type_mapping(['aMe9'], [MCNS, BANC], 'exact')
    flows = [f for fl in out['pair_flows'].values() for f in fl]
    fired = [f for f in flows if f.get('suspects')]
    assert fired, 'aMe9 -> banc must produce a suspects flow'
    f = fired[0]
    assert 'aMe12' in f['suspect_rivals']
    assert f['same_name_first']['disposition'] == 'broad_selection'
    assert f['mapping_relationship'] == 'suspects'
    # and a non-fan-out flow keeps the empty defaults (no false positives)
    other = [f for f in flows if not f.get('suspects')]
    for f in other:
        assert f['suspect_rivals'] == [] and not f['same_name_first']


def test_the_panel_publishes_the_out_map_of_the_received_types():
    """`Out-map (in-map types)` is the received types' own neurons that the
    claim set does NOT reach.  Recomputed here from the same pools the panel
    read, so the published number is pinned to its definition rather than to
    itself (user 2026-09-26).  Panel-side on purpose: with no morphology a
    candidate cannot close a hole, so this figure is at least as large as the
    validation run's `family` bin — 219 − 204 = 15 against 11 on
    circadian_clock → male-cns."""
    from comparison.mapping_visualization import (
        flow_is_claimed, mapping_pool_key)
    from ui.components.type_mapping_panel import _compute_type_mapping

    out = _compute_type_mapping(['SLP249'], [MCNS, FAFB], 'exact')
    assert out['summary'], 'the fixture query must resolve a mapping'
    for row in out['summary']:
        assert isinstance(row['out_map'], int) and row['out_map'] >= 0

    pop, claimed = {}, {}
    for (src, tgt), flows in out['pair_flows'].items():
        for f in flows:
            if not flow_is_claimed(f):
                continue
            pool = out['pools'].get(
                mapping_pool_key(src, tgt, f.get('source_type'),
                                 f.get('foreign_type'))) or {}
            tkey = (tgt, str(f.get('foreign_type') or ''))
            pop.setdefault(tkey, set()).update(
                str(b) for b in (pool.get('target_type_body_ids') or []))
            claimed.setdefault(tkey, set()).update(
                str(b) for b in (pool.get('target_body_ids') or []))
    assert pop, 'SLP249 must resolve at least one claimed endpoint type'

    by_ds = {}
    for (tgt, foreign), ids in pop.items():
        by_ds.setdefault(tgt, set()).update(
            ids - (claimed.get((tgt, foreign)) or set()))
    for row in out['summary']:
        assert row['out_map'] == len(by_ds.get(row['dataset'], set())), row
        # the claim set and the out-map are disjoint halves of the same
        # population: they can never both be counted as mapped
        assert row['out_map'] == 0 or row['mapped_neurons'] > 0


def test_the_circadian_clock_envelope_publishes_204_mapped_and_15_out_map():
    """The ratified envelope, read straight off the panel (user 2026-09-26):
    circadian_clock FAFB → male-cns claims 204 bodyIds and reports 15 of the
    received types' 219 neurons as out-of-map — the same 219 / 204 the
    validation run publishes in `set_coverage.json`, so panel and pipeline
    describe ONE claim set.  The pipeline's `family` bin is 11 rows on this
    query, because a morph-qualified candidate closes a hole there; the panel
    has no morphology, so it reports the whole overhang."""
    from ui.components.type_mapping_panel import _compute_type_mapping

    out = _compute_type_mapping(['circadian_clock'], [FAFB, MCNS], 'expand')
    row = {r['dataset']: r for r in out['summary']}
    target = row[MCNS]
    assert target['mapped_neurons'] == 204
    assert target['out_map'] == 15
    assert target['mapped_neurons'] + target['out_map'] == 219
    # the issuing side reports neither a claim nor an overhang
    assert row[FAFB]['mapped_neurons'] == 0 and row[FAFB]['out_map'] == 0
    # per-type rows are the PER-BRANCH grain: convergent sources can name the
    # same neuron twice, so they sum above the deduped union (17 vs 15)
    per_type = [r for r in out['summary_per_type'] if r['target'] == MCNS]
    assert sum(r['out_map'] for r in per_type) >= target['out_map']


@pytest.fixture
def suspects_panel_client(tmp_path, monkeypatch):
    """A panel over male-cns v1.0 + BANC v888 — the pair whose ``aMe9``
    query fires the same-name-first fan-out (suspects).  Mirrors
    ``panel_client`` but with the suspects-carrying selection."""
    from nicegui import Client
    from nicegui.page import page

    monkeypatch.setattr(tmh, "_HISTORY_PATH",
                        tmp_path / "type_mapping_history.json")
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
    clear_neuron_index_cache()
    from comparison.cross_dataset_type_mapper import get_type_mapper

    assert get_type_mapper().load() is True
    client = Client(page('/type-mapping-suspects-test'))
    with client:
        selection = {'value': [MCNS, BANC]}
        from ui.components.type_mapping_panel import create_type_mapping_entry
        button = create_type_mapping_entry(lambda: list(selection['value']))
    try:
        yield client, button, selection
    finally:
        clear_neuron_index_cache()


def _tables(client):
    return [e for e in client.elements.values()
            if type(e).__name__ == 'Table']


def _column_labels(table):
    return [c.get('label', '') for c in table._props.get('columns', [])]


def test_per_type_breakdown_carries_relationship_and_suspects():
    """The per-type breakdown row reports the pair's cardinality and its
    suspects count (fan-out/suspects display round)."""
    from ui.components.type_mapping_panel import _compute_type_mapping

    out = _compute_type_mapping(['aMe9'], [MCNS, BANC], 'exact')
    row = next(r for r in out['summary_per_type']
               if r['type'] == 'aMe9' and r['target'] == BANC)
    assert row['relationship'] == '1-to-N'
    assert int(row['suspects']) >= 1


def test_pair_card_gains_relationship_and_suspects_columns(
        suspects_panel_client):
    client, button, _sel = suspects_panel_client
    button.search_container.add_values(['aMe9'])
    assert _click_button(client, 'Search mappings')
    pair_cards = [t for t in _tables(client)
                  if 'Map used (per linker)' in _column_labels(t)]
    assert pair_cards, 'pair-card mapped-pairs table not found'
    for t in pair_cards:
        cols = _column_labels(t)
        assert 'Relationship' in cols and 'Suspects' in cols
    # the surviving suspects flow's row carries the badge cell
    cells = [str(v) for t in pair_cards
             for row in t._props.get('rows', []) for v in row.values()]
    assert any('⚠ suspects' in c for c in cells)


def test_coverage_and_suspects_blocks_render(suspects_panel_client):
    client, button, _sel = suspects_panel_client
    button.search_container.add_values(['aMe9'])
    assert _click_button(client, 'Search mappings')
    labels = _labels(client)
    # coverage tables gained a Suspects column
    cov = [t for t in _tables(client)
           if any('· selected' in c for c in _column_labels(t))]
    assert cov, 'coverage tables not found'
    for t in cov:
        assert 'Suspects' in _column_labels(t)
    # data-driven title: fan summary + suspect-pair count replace the old
    # hard-coded '(bidirectional, 1-to-N fan-out)'
    assert any('Type coverage —' in lbl and 'fan-out' in lbl
               and 'suspect pair(s)' in lbl for lbl in labels)
    # COLLAPSED per-rival suspects block (D1: never hover-only, never
    # expanded by default) titled by its (source -> target) type pair
    assert any(lbl.startswith('Suspects — ') for lbl in labels)



def test_suspects_badges_carry_a_hover_with_the_rival_evidence(
        suspects_panel_client):
    """2026-09-26: the badge was a dead end — its per-rival facts lived only
    in the collapsed block below the table, so reading what the warning
    meant meant hunting for the expander.  Every surface that renders a
    '⚠ suspects' badge now hovers too: the same-name-first explanation plus
    one line per rival candidate.  The collapsed block stays — a hover is
    never the only route to the evidence (D1)."""
    client, button, _sel = suspects_panel_client
    # two chips so the per-type breakdown renders (it is collapsed for a
    # single-type preview); aMe9 is the same-name-first fan-out
    button.search_container.add_values(['aMe9', 'aMe12'])
    assert _click_button(client, 'Search mappings')

    badge = [t for t in _tables(client) if 'Suspects' in _column_labels(t)]
    # pair card + coverage forward + coverage backward + per-type breakdown
    assert len(badge) == 4, [_column_labels(t) for t in badge]
    for t in badge:
        names = [k for k in t.slots if k.startswith('body-cell-suspects')]
        assert names, f'no suspects cell slot on {_column_labels(t)}'
        template = t.slots[names[0]].template
        assert 'q-tooltip' in template and 'suspects_tip' in template

    tips = [str(row.get('suspects_tip') or '') for t in badge
            for row in t._props.get('rows', [])]
    flagged = [tip for tip in tips if tip]
    assert len(flagged) >= 4, 'every flagged row carries its own hover'
    for tip in flagged:
        assert tip.startswith('same-name-first selection')
        assert 'own 1-to-1 pair' in tip and 'votes' in tip
        assert 'aMe12' in tip, 'the demoted rival is named'
    # a row without the badge never grows a hover
    assert any(not tip for tip in tips)
    # the collapsed block is still rendered beside the hover
    assert any(lbl.startswith('Suspects — ') for lbl in _labels(client))


def test_a_declined_rival_is_listed_but_never_counted_as_mapped(panel_client):
    """2026-09-22: the headline count and the pair list disagreed about what
    the mapping is.  FAFB ``APDN3`` fans out to three BANC types, and the
    same-name rule adopts ``APDN3 → APDN3`` (7 neurons) while declining
    ``→ LMTe01`` (4) and ``→ LTe71`` (1).  Summing all three pools made the
    panel claim 12 mapped neurons for a mapping the validator grades as 7 —
    and the same error at scale read 205 against the report's 198 on
    ``circadian_clock``.  The rivals stay visible as disclosure rows; only
    the adopted pair feeds the claim numbers."""
    from ui.components.type_mapping_panel import _compute_type_mapping

    outcome = _compute_type_mapping(['APDN3'], [FAFB, BANC], 'exact')
    ends = {f['foreign_type'] for f in outcome['pair_flows'][(FAFB, BANC)]}
    assert {'APDN3', 'LMTe01', 'LTe71'} <= ends        # still listed
    banc = _summary_by_dataset(outcome)[BANC]
    assert banc['mapped_neurons'] == 7 and banc['mapped_types'] == 1
    per_type = [r for r in outcome['summary_per_type']
                if r['type'] == 'APDN3' and r['target'] == BANC]
    assert per_type[0]['mapped_neurons'] == 7
    # the fan-out is still disclosed where it describes evidence, not claims
    assert per_type[0]['relationship'] == '1-to-N'


def test_composed_sankey_button_sits_beside_the_mapping_graph(panel_client):
    """2026-09-26: the composed export gained its own Sankey (one column per
    dataset, the query's origin in the middle) beside the network button.
    The network answers 'what connects to what'; the Sankey answers how much
    of each type crosses over."""
    client, button, _selection = panel_client
    button.search_container.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')
    texts = ([str(getattr(b, 'text', '')) for b in _buttons(client)]
             + _labels(client))
    assert any('Mapping sankey (HTML)' in t for t in texts), texts


def test_three_tier_summary_claim_reach_disclosure():
    """§three-tier readout (user 2026-09-27): the summary publishes the
    CLAIM set (pure claim basis — bodies AND types adopted-only), the
    EVIDENCE REACH (every flow's pools, disclosure ends included), and the
    disclosure detail with decline reasons — so no two surfaces can be
    mistaken for one number.  Anchored on circadian FAFB -> banc_v888:
    claim 198(39 types), reach 205(42 types), disclosure 3 types / 7
    bodies (LMTe01+LTe71 same-name-first rivals of APDN3; CB3767 fan-out
    branch not adopted), out-map 2."""
    from ui.components.type_mapping_panel import _compute_type_mapping
    outcome = _compute_type_mapping(
        ['circadian_clock'], ['flywire_FAFB_v783', 'banc_v888'], 'exact')
    row = next(r for r in outcome['summary']
               if r['dataset'] == 'banc_v888')
    assert row['mapped'] == '198(39 types)'
    assert row['reach'] == '205(42 types)'
    assert row['disclosure_types'] == 3
    assert row['disclosure_bodies'] == 7
    assert row['out_map'] == 2
    detail = {d['type']: d['reason'] for d in row['disclosure_detail']}
    assert detail.get('LMTe01') == (
        "same-name-first rival — 'APDN3' selected instead")
    assert detail.get('LTe71') == (
        "same-name-first rival — 'APDN3' selected instead")
    assert detail.get('CB3767') == (
        'fan-out branch not adopted by the decision '
        '(valid_split_evidence)')
    # MCNS keeps the ratified claim figures — claim == reach there
    outcome_m = _compute_type_mapping(
        ['circadian_clock'], ['flywire_FAFB_v783', 'male-cns:v1.0'],
        'exact')
    row_m = next(r for r in outcome_m['summary']
                 if r['dataset'] == 'male-cns:v1.0')
    assert row_m['mapped'] == '204(40 types)'
    assert row_m['reach'] == '204(40 types)'
    assert row_m['out_map'] == 15
