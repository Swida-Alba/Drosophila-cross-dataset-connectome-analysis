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
    and it comes back when switching away."""
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

    assert board.visible is False
    assert disabled() == []
    aggregation.set_value("custom group")
    assert board.visible is True
    assert disabled(), "the query walker found no interactive child to disable"
    aggregation.set_value("bodyid")
    assert board.visible is False
    assert disabled() == []
