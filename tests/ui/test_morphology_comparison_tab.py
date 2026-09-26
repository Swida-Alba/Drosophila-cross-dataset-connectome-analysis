"""Focused wiring checks for the Morphology Comparison sub-tab."""

from pathlib import Path

from ui.runner import ScriptRunner


def test_runner_resolves_comparison_output_marker(tmp_path):
    """The morphology comparison completion log must populate the Output
    Files panel."""
    run_folder = (Path(tmp_path) /
                  "morphology_comparison_MCNS_aMe12_aMe10_20260901_120000")
    run_folder.mkdir()
    (run_folder / "report.html").write_text("<html></html>", encoding="utf-8")

    runner = ScriptRunner()
    runner._run_logs = [
        ("stdout",
         f"[MorphologyProfileComparer] Output: {run_folder}"),
    ]

    assert runner._extract_output_folder(str(tmp_path)) == str(run_folder)
    assert runner._resolve_scan_dir(str(tmp_path)) == str(run_folder)


def test_comparison_run_folder_prefix_matches_scan_regex():
    """The run-folder prefix must be whitelisted in the scan-dir fallback."""
    assert ScriptRunner._RUN_FOLDER_PREFIX_RE.match(
        "morphology_comparison_MCNS_aMe12_20260901_120000")


# ------------------------------------------------------- aggregation level UI
def _by_element_id(client, element_id):
    return next(
        element for element in client.elements.values()
        if getattr(element, "_props", {}).get("id") == element_id)


def test_aggregation_level_drops_bodyid_while_nblast_is_selected():
    """A row choice NBLAST cannot complete is not offered, and an already
    chosen one resets — a disabled choice must never reach the run record."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-aggregation"))
    with client:
        create_morphology_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        method = _by_element_id(client, "select-morph-comparison-method")

    assert "bodyid" in aggregation.options
    aggregation.set_value("bodyid")

    method.set_value("nblast")
    assert "bodyid" not in aggregation.options
    assert aggregation.value == "type"

    method.set_value("vector_v2")
    assert "bodyid" in aggregation.options
    assert aggregation.value == "type"


def test_custom_group_board_follows_the_aggregation_level():
    """The grouping board is shown only for the custom group level."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-custom-board"))
    with client:
        create_morphology_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        board = _by_element_id(client, "card-morph-custom-group")

    assert board.visible is False
    aggregation.set_value("custom group")
    assert board.visible is True
    aggregation.set_value("bodyid")
    assert board.visible is False


def test_intra_only_parameters_stay_hidden_for_two_datasets():
    """Aggregation Level and the board belong to the intra-dataset comparison:
    a cross run never reads them, so a stale bodyid/custom choice cannot ride
    into the cross-dataset constructor params."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-cross-visibility"))
    with client:
        create_morphology_tab()
        datasets = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Datasets to compare")
        aggregation = _by_element_id(client, "select-aggregation")
        intra_box = _by_element_id(client, "card-morph-comparison-intra")

    aggregation.set_value("bodyid")
    datasets.set_value(["male-cns:v1.0", "flywire_FAFB_v783"])
    # The whole intra block hides, so run_comparison's cross branch — which
    # never reads these widgets — cannot be reached with a stale choice.
    assert intra_box.visible is False
    datasets.set_value(["male-cns:v1.0"])
    assert intra_box.visible is True
    assert aggregation.value == "bodyid"


def _descendants(element):
    out = []
    stack = list(element.default_slot.children)
    while stack:
        child = stack.pop()
        out.append(child)
        stack.extend(child.default_slot.children)
    return out


def test_custom_group_mode_disables_the_ignored_query_input():
    """Custom rows come from the grouping board, so the query box is disabled
    rather than left live-but-ignored — and it comes back on switching away."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-query-disable"))
    with client:
        create_morphology_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        query_field = _by_element_id(client, "field-morph-comparison-query")

    def disabled():
        return [e for e in _descendants(query_field)
                if getattr(e, "_props", {}).get("disable")]

    assert disabled() == []
    aggregation.set_value("custom group")
    assert disabled(), "the query walker found no interactive child to disable"
    aggregation.set_value("type")
    assert disabled() == []
