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
    assert fafb == {'dataset': FAFB, 'types': 0, 'neurons': 0,
                    'mapped_types': 0, 'mapped_neurons': 0, 'mapped': '0',
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


def test_multi_type_preview_shows_collapsed_per_type_expansion(panel_client):
    client, button, _selection = panel_client
    button.search_container.add_values(['5thsLNv_LNd6', 'SLP249'])
    assert _click_button(client, 'Search mappings')
    expansions = [e for e in client.elements.values()
                  if type(e).__name__ == 'Expansion']
    assert any('Per-type breakdown' in str(getattr(e, 'text', ''))
               for e in expansions)


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
