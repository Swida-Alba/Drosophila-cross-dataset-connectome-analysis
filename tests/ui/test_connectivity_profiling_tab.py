"""Focused wiring checks for the connectivity Comparison sub-tab."""

from pathlib import Path

from ui.tabs import connectivity as connectivity_tab
from ui.runner import ScriptRunner


def test_comparison_output_dir_prefers_selected_subtab_value(monkeypatch):
    fallback_calls = []

    def fallback(scope):
        fallback_calls.append(scope)
        return "/tmp/inherited-profiling"

    monkeypatch.setattr(connectivity_tab, "get_tab_output_dir", fallback)

    assert (
        connectivity_tab._resolve_comparison_output_dir(" /tmp/selected ")
        == "/tmp/selected"
    )
    assert fallback_calls == []
    assert (
        connectivity_tab._resolve_comparison_output_dir("")
        == "/tmp/inherited-profiling"
    )
    assert (
        connectivity_tab._resolve_comparison_output_dir("   ")
        == "/tmp/inherited-profiling"
    )
    assert fallback_calls == ["connectivity_profiling", "connectivity_profiling"]


def test_runner_resolves_comparison_output_marker(tmp_path):
    """The comparison completion log must populate the Output Files panel."""
    run_folder = Path(tmp_path) / "profiling_MCNS_aMe_20260814_175350"
    run_folder.mkdir()
    (run_folder / "report.html").write_text("<html></html>", encoding="utf-8")

    runner = ScriptRunner()
    runner._run_logs = [
        ("stdout", f"[ConnectivityProfileComparer] Output: {run_folder}"),
    ]

    assert runner._extract_output_folder(str(tmp_path)) == str(run_folder)
    assert runner._resolve_scan_dir(str(tmp_path)) == str(run_folder)


def _by_element_id(client, element_id):
    return next(
        element for element in client.elements.values()
        if getattr(element, "_props", {}).get("id") == element_id)


def _descendants(element):
    out = []
    stack = list(element.default_slot.children)
    while stack:
        child = stack.pop()
        out.append(child)
        stack.extend(child.default_slot.children)
    return out


def test_custom_group_mode_disables_the_query_input():
    """Parity with the morphology Comparison sub-tab: the query the custom
    group level does not use is disabled rather than left live-but-ignored,
    and it comes back when switching away.  The grouping card itself is
    ALWAYS visible — it is the only affordance for custom grouping and
    label mapping, and hiding it by default made the feature effectively
    undiscoverable."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.connectivity import create_connectivity_tab

    client = Client(page("/connectivity-comparison-query-disable"))
    with client:
        create_connectivity_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        query_field = _by_element_id(
            client, "field-connectivity-comparison-query")
        board = _by_element_id(client, "card-custom-group")

    def disabled():
        return [e for e in _descendants(query_field)
                if getattr(e, "_props", {}).get("disable")]

    assert board.visible is True, (
        "the Custom Groups (LabelMapper) card must be visible on tab open")
    assert disabled() == []
    aggregation.set_value("custom group")
    assert board.visible is True
    assert disabled(), "the query walker found no interactive child to disable"
    aggregation.set_value("bodyid")
    assert board.visible is True
    assert disabled() == []


def test_aggregation_select_lives_outside_advanced_expansion():
    """UX reveal: the Aggregation Level selector moved out of the Advanced
    Settings expansion into the Profile Construction card — it must not be
    reachable only through the expansion."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.connectivity import create_connectivity_tab

    client = Client(page("/connectivity-aggregation-placement"))
    with client:
        create_connectivity_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        advanced = _by_element_id(client, "card-connectivity-advanced")

    assert aggregation not in _descendants(advanced), (
        "the aggregation select is still buried in the Advanced expansion")


def test_custom_card_visible_in_type_mode_with_no_mapping():
    """The user-facing gap: opening the tab in the default 'type' mode with
    NO mapping loaded still shows the Custom Groups (LabelMapper) card —
    the grouping button, the tier-0 caption, and the partner-grouping
    toggle must be discoverable without touching Aggregation Level."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.connectivity import create_connectivity_tab

    client = Client(page("/connectivity-card-visible-by-default"))
    with client:
        create_connectivity_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        query_field = _by_element_id(
            client, "field-connectivity-comparison-query")
        board = _by_element_id(client, "card-custom-group")

    # default mode ('type'), empty board: the card is up and interactive
    assert board.visible is True
    assert aggregation.value == "type"
    assert not [e for e in _descendants(query_field)
                if getattr(e, "_props", {}).get("disable")], (
        "type mode keeps the query box live")

    # the card hosts the grouping affordance (button label carries the
    # preset name) — the tier-0 caption renders beside it
    assert any(
        "Custom Grouping Preset" in str(getattr(e, "text", ""))
        for e in _descendants(board)), (
        "the grouping preset button is not inside the always-visible card")
