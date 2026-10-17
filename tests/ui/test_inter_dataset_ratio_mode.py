"""Cross-Dataset > Paths threshold-mode × connection-ratio integration.

The Threshold Basis selector swaps which chip editor is visible (each
basis keeps its own valid defaults — a synapse default like 3 is an
invalid ratio tier), the auto-mode hint names the basis-correct
bootstrap floor, and the payload collection is ratio-aware in EVERY
mode (auto included: float parse, PERCENT validation (0, 100], 0.1%
floor — the engine still receives fractions).
"""

import inspect


def _build_tab(client):
    from ui.tabs.inter_dataset import create_inter_dataset_tab
    create_inter_dataset_tab()
    elements = list(client.elements.values())
    controls = {
        el._props.get("label"): el
        for el in elements
        if getattr(el, "_props", {}).get("label")
    }
    editors = {
        el._props.get("data-threshold-editor"): el
        for el in elements
        if getattr(el, "_props", {}).get("data-threshold-editor")
    }
    labels = [
        getattr(el, "text", "") for el in elements
        if getattr(el, "text", "")
    ]
    return controls, editors, labels


def test_both_chip_editors_exist_with_basis_defaults():
    from nicegui import Client
    from nicegui.page import page
    with Client(page("/ratio-editors-exist")) as client:
        controls, editors, labels = _build_tab(client)
    assert "Synapse Thresholds" in controls
    assert "Connection-Ratio Thresholds (%)" in controls
    # Basis-correct defaults: [3, 5, 10] would all be invalid ratio tiers.
    assert controls["Synapse Thresholds"].value == [3, 5, 10]
    # Percent, like the single Min Connection Ratio (%) field on the other
    # three surfaces: 0.1% = the engine tier 0.001.
    assert controls["Connection-Ratio Thresholds (%)"].value == ['0.1', '0.5']
    # Default mode is Auto: chips belong to Standard/Custom combination,
    # so BOTH editors start hidden (the swap is pinned in the flip test).
    assert editors["synapse"].visible is False
    assert editors["ratio"].visible is False
    # The auto hint names the synapse floor at build time.
    assert any("Min Synapse Count 3" in t for t in labels)


def test_basis_flip_swaps_editor_visibility_and_hint_floor():
    from nicegui import Client
    from nicegui.page import page
    with Client(page("/ratio-basis-flip")) as client:
        controls, editors, _labels = _build_tab(client)
        # Chips only show in Standard mode: switch there first (via the
        # drive handle — buttons are not clickable in headless builds).
        editors["synapse"].set_threshold_mode("standard")
        assert editors["synapse"].visible is True
        assert editors["ratio"].visible is False
        controls["Threshold Basis"].value = "Connection ratio"
    # After the flip: the ratio editor is the visible one...
    assert editors["synapse"].visible is False
    assert editors["ratio"].visible is True
    # ...and the auto hint names the ratio bootstrap floor.
    texts = [
        getattr(el, "text", "") for el in client.elements.values()
        if getattr(el, "text", "")
    ]
    assert any("0.1%" in t and "floor" in t for t in texts)


def test_collection_is_ratio_aware_in_every_mode():
    """Source-level pins for the payload collection (the closure is not
    directly reachable from a Client build): auto mode parses floats,
    validates (0, 100] percent, and floors at 0.1% — [3] would be rejected
    by the
    backend before the bootstrap ever runs."""
    from ui.tabs import inter_dataset
    src = inspect.getsource(inter_dataset)
    # Auto mode: percent floor + float parse + (0,100] validation, and the
    # conversion happens ONCE, through the shared helper.
    # the floor is written in the UI unit and converted on the way out
    assert "[0.1] if ratio_mode else [3]" in src
    assert "_to_engine" in src
    assert "pc.percent_to_fraction(" in src
    assert "(float(value) if ratio_mode else int(value))" in src
    assert 'not (0 < value <= 100) for value in values' in src
    # The basis swap is registered and composes with the mode sync.
    assert "threshold_basis.on_value_change(lambda _e: _sync_threshold_basis())" in src
    assert "_apply_threshold_input_visibility()" in src
    # No stale single-editor references remain.
    assert "thresholds_input = " not in src
    assert "thresholds_input.get_value" not in src
    assert "thresholds_input.set_visibility" not in src


def test_coverage_controls_live_in_the_core_cards():
    """User placement decision (2026-10-04): the coverage early-stop
    selects belong to the CORE parameters card of both tabs, not the
    Advanced card."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.find_shortest import create_find_shortest_tab
    from ui.tabs.inter_dataset import create_inter_dataset_tab

    def in_card(el, card_id):
        ancestor = el
        while ancestor is not None:
            if ((getattr(ancestor, '_props', {}) or {}).get('id')
                    == card_id):
                return True
            parent_slot = getattr(ancestor, 'parent_slot', None)
            ancestor = parent_slot.parent if parent_slot is not None else None
        return False

    with Client(page("/fs-core-placement")) as c1:
        create_find_shortest_tab()
    placed = set()
    for el in c1.elements.values():
        label = (getattr(el, '_props', {}) or {}).get('label')
        if label in ('Source Coverage', 'Target Coverage',
                     'Custom Coverage %'):
            placed.add((label, in_card(el, 'card-findshortest-core')))
    assert placed and all(ok for _, ok in placed), placed

    with Client(page("/id-core-placement")) as c2:
        create_inter_dataset_tab()
    placed2 = set()
    for el in c2.elements.values():
        label = (getattr(el, '_props', {}) or {}).get('label')
        if label and 'Coverage' in str(label):
            placed2.add((label, in_card(el, 'card-interdataset-core')))
    assert placed2 and all(ok for _, ok in placed2), placed2


def _click(client, text_part):
    """Invoke the click handler of the first button whose text matches."""
    for element in client.elements.values():
        if type(element).__name__ != 'Button':
            continue
        if text_part not in str(getattr(element, 'text', '')):
            continue
        for listener in element._event_listeners.values():
            if listener.type == 'click' and listener.handler:
                listener.handler({'sender': element.id, 'client': client,
                                  'args': None})
                return
    raise AssertionError(f'no button matching {text_part!r}')


def _cells(client):
    return [el.value for el in client.elements.values()
            if (getattr(el, '_props', {}) or {}).get('placeholder')
            in ('positive integer', 'percent, e.g. 0.1')]


def _cell_hints(client):
    return {(getattr(el, '_props', {}) or {}).get('placeholder')
            for el in client.elements.values()
            if (getattr(el, '_props', {}) or {}).get('placeholder')
            in ('positive integer', 'percent, e.g. 0.1')}


def test_seeding_from_alignment_adopts_the_recorded_basis(tmp_path):
    """'Suggest from alignment' must read the SUGGESTION's basis, not the
    selector's (plan 2026-10-16 review).

    The alignment file's tiers come off the executed threshold folders:
    synapse COUNTS on the synapse basis, RAW FRACTIONS on the ratio basis.
    Seeding a ratio suggestion while the selector still said 'Synapse count'
    typed 0.05 into a synapse cell (and a synapse suggestion into a percent
    cell typed 5 as 5%), so the prefill silently changed units under the
    reader. The writer now records 'weight_basis' and the seed honours it,
    moving the selector with it.

    Wrong hypothesis: the pre-fix seed leaves the basis at 'Synapse count'
    and the cells at ['0.05', '0.005']."""
    import json

    from nicegui import Client
    from nicegui.page import page

    root = tmp_path / 'run'
    (root / 'comparison_results').mkdir(parents=True)
    (root / 'comparison_results'
     / 'suggested_threshold_combinations.json').write_text(json.dumps({
         'reference_dataset': 'dsA', 'tolerance': 0.1,
         'weight_basis': 'connection_ratio',
         'combinations': [{'id': 'combo_001', 'label': 'Aligned 1',
                           'thresholds': {'dsA': 0.05, 'dsB': 0.005}}],
     }), encoding='utf-8')

    with Client(page('/seed-basis')) as client:
        controls, _editors, _labels = _build_tab(client)
        datasets = [el for el in client.elements.values()
                    if str((getattr(el, '_props', {}) or {}).get('label', '')
                            ).startswith('Datasets to compare')]
        assert datasets, 'no dataset selector on the tab'
        controls['Output Directory'].value = str(root)
        datasets[0].value = ['dsA', 'dsB']
        assert controls['Threshold Basis'].value == 'Synapse count'

        _click(client, 'Suggest from alignment')

        assert controls['Threshold Basis'].value == 'Connection ratio', (
            'a ratio-basis suggestion must move the basis selector, or its '
            'numbers land in the wrong unit')
        assert _cells(client) == ['5.0', '0.5'], (
            f'the seeded cells must be PERCENT of the recorded fractions: '
            f'{_cells(client)}')
        # and the cells say what unit they are in
        assert _cell_hints(client) == {'percent, e.g. 0.1'}, _cell_hints(client)

    # and a synapse suggestion must NOT move a ratio-basis reader into counts
    (root / 'comparison_results'
     / 'suggested_threshold_combinations.json').write_text(json.dumps({
         'weight_basis': 'synapse',
         'combinations': [{'id': 'combo_002', 'label': 'Aligned 2',
                           'thresholds': {'dsA': 5, 'dsB': 12}}],
     }), encoding='utf-8')
    with Client(page('/seed-basis-synapse')) as client:
        controls, _editors, _labels = _build_tab(client)
        datasets = [el for el in client.elements.values()
                    if str((getattr(el, '_props', {}) or {}).get('label', '')
                            ).startswith('Datasets to compare')]
        controls['Output Directory'].value = str(root)
        datasets[0].value = ['dsA', 'dsB']
        controls['Threshold Basis'].value = 'Connection ratio'
        _click(client, 'Suggest from alignment')
        assert controls['Threshold Basis'].value == 'Synapse count'
        assert _cells(client) == ['5', '12'], _cells(client)
        assert _cell_hints(client) == {'positive integer'}, _cell_hints(client)


def test_the_alignment_file_records_the_basis_it_was_measured_in(tmp_path):
    """The writer half of the law: a suggestion with no recorded basis is
    unreadable on the fourth surface, and the folder grammar the tiers were
    read off is exactly what decides the unit."""
    import json
    import os

    import pandas as pd

    from src.comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)

    class _Params:
        weight_basis = 'connection_ratio'

        def get_dataset_names(self):
            return ['dsA', 'dsB']

    analyzer.parameters = _Params()
    analyzer._log = lambda *_a, **_k: None
    analyzer._suggested_combinations = None
    out = str(tmp_path / 'comparison_results')
    os.makedirs(out, exist_ok=True)
    best = pd.DataFrame([{
        'match_kind': 'anchor', 'reference_dataset': 'dsA',
        'anchor_threshold': 0.05, 'target_dataset': 'dsB', 'best_t': 0.005,
        'distance': 0.0}])
    analyzer._suggest_and_export_combinations(best, ['dsA', 'dsB'], out)
    payload = json.loads((tmp_path / 'comparison_results'
                          / 'suggested_threshold_combinations.json')
                         .read_text(encoding='utf-8'))
    assert payload['weight_basis'] == 'connection_ratio', payload
    assert payload['combinations'][0]['thresholds'] == {'dsA': 0.05,
                                                        'dsB': 0.005}, payload
