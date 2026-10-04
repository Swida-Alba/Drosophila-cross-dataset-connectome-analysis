"""Cross-Dataset > Paths threshold-mode × connection-ratio integration.

The Threshold Basis selector swaps which chip editor is visible (each
basis keeps its own valid defaults — a synapse default like 3 is an
invalid ratio tier), the auto-mode hint names the basis-correct
bootstrap floor, and the payload collection is ratio-aware in EVERY
mode (auto included: float parse, (0, 1] validation, 0.001 floor).
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
    assert "Connection-Ratio Thresholds" in controls
    # Basis-correct defaults: [3, 5, 10] would all be invalid ratio tiers.
    assert controls["Synapse Thresholds"].value == [3, 5, 10]
    assert controls["Connection-Ratio Thresholds"].value == ['0.001', '0.005']
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
    assert any("0.001" in t and "floor" in t for t in texts)


def test_collection_is_ratio_aware_in_every_mode():
    """Source-level pins for the payload collection (the closure is not
    directly reachable from a Client build): auto mode parses floats,
    validates (0, 1], and floors at 0.001 — [3] would be rejected by the
    backend before the bootstrap ever runs."""
    from ui.tabs import inter_dataset
    src = inspect.getsource(inter_dataset)
    # Auto mode: ratio floor + float parse + (0,1] validation.
    assert "[0.001] if ratio_mode else [3]" in src
    assert "(float(value) if ratio_mode else int(value))" in src
    assert 'not (0 < value <= 1) for value in values' in src
    # The basis swap is registered and composes with the mode sync.
    assert "threshold_basis.on_value_change(lambda _e: _sync_threshold_basis())" in src
    assert "_apply_threshold_input_visibility()" in src
    # No stale single-editor references remain.
    assert "thresholds_input = " not in src
    assert "thresholds_input.get_value" not in src
    assert "thresholds_input.set_visibility" not in src
