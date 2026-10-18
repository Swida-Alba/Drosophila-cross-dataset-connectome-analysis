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

    method.set_value("both")
    assert "bodyid" not in aggregation.options
    assert aggregation.value == "type"

    method.set_value("vector_v2")
    assert "bodyid" in aggregation.options
    assert aggregation.value == "type"


def test_custom_group_board_always_visible():
    """The grouping board is ALWAYS visible: it is the only affordance for
    custom grouping and label mapping, and it serves every mode (intra
    'custom group' rows + the tier-0 label overlay on type/bodyid and
    cross-dataset runs). Hiding it by default made the feature effectively
    undiscoverable."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-custom-board"))
    with client:
        create_morphology_tab()
        aggregation = _by_element_id(client, "select-aggregation")
        board = _by_element_id(client, "card-morph-custom-group")

    assert board.visible is True, (
        "the Custom Groups (LabelMapper) card must be visible on tab open")
    aggregation.set_value("custom group")
    assert board.visible is True
    aggregation.set_value("bodyid")
    assert board.visible is True


def test_intra_parameters_stay_visible_for_two_datasets():
    """Full parity with Connectivity > Comparison: a multi-dataset run
    computes each selected volume's intra matrices AND the cross pairs, so
    the intra block must remain visible (it feeds the intra half)."""
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
    assert intra_box.visible is True
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


# ------------------------------------------------------- run-time level clamp
def test_multi_dataset_runs_keep_the_selected_intra_level():
    """The suite computes each volume's intra matrices at the selected level
    while the cross phase stays on native type rows; a stale value must still
    fall back to 'type' rather than reaching the run record."""
    from ui.tabs.morphology import _comparison_aggregation

    assert _comparison_aggregation(True, "custom group") == "custom group"
    assert _comparison_aggregation(True, "bodyid") == "bodyid"
    assert _comparison_aggregation(False, "custom group") == "custom group"
    assert _comparison_aggregation(False, "bodyid") == "bodyid"
    assert _comparison_aggregation(False, None) == "type"
    # A value the option list no longer offers (NBLAST reset) cannot ride along
    assert _comparison_aggregation(False, "not-a-level") == "type"


def test_multi_dataset_branch_calls_the_intra_plus_cross_suite():
    """A two-dataset run must go through the suite (intra + cross in one
    folder), not the cross-only backend, and must forward the intra controls
    that stay visible in cross mode."""
    source = (Path(__file__).resolve().parents[2]
              / "ui" / "tabs" / "morphology.py").read_text(encoding="utf-8")
    assert 'tool = "morphology_multi_comparison"' in source
    assert '"intra_fetch_online": bool(comparison_fetch.value)' in source
    assert '"intra_use_cache": bool(comparison_use_cache.value)' in source
    assert '"intra_visualize": viz_on' in source
    assert '"aggregation_level": _AGGREGATION_PARAM_VALUES[aggregation]' in source
    assert '"max_total_neurons": int(comparison_max_total.value)' in source


def test_custom_group_labels_read_every_preset_label(tmp_path):
    """A label with no members in the first selected dataset must still be
    sent to the cross phase, which resolves it per dataset and reports an
    explicit empty row."""
    import json

    from ui.tabs.morphology import _custom_group_labels

    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({
        "format": "drocat_custom_groups/v1",
        "source_mapping": {
            "custom_label": ["shared", "first-only", "second-only"],
            "male-cns:v1.0": [["aMe12"], ["aMe10"], []],
            "flywire_FAFB_v783": [["aMe12"], [], ["aMe10"]],
        },
    }), encoding="utf-8")
    assert _custom_group_labels(preset, "male-cns:v1.0") == [
        "shared", "first-only", "second-only"]
    assert _custom_group_labels(tmp_path / "missing.json", "x") == []


def test_the_ui_no_longer_refuses_a_single_neuron_entry():
    """The population gate lives in the backend, where it counts NEURONS: one
    type still yields its bodyId pairwise matrix, so the tab must not veto the
    entry before the run starts."""
    source = (Path(__file__).resolve().parents[2]
              / "ui" / "tabs" / "morphology.py").read_text(encoding="utf-8")
    assert "Please enter at least two neurons" not in source
    assert "at least one neuron, type, or pattern" in source


def test_aggregation_level_sits_in_advanced_settings():
    """Layout parity with the Connectivity tab: the level select lives inside
    the Advanced Settings card, and the grouping board stays ABOVE it (moving
    the board would break the one-advanced-card-last rule)."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-advanced-placement"))
    with client:
        create_morphology_tab()
        advanced = _by_element_id(client, "card-morphology-advanced")
        aggregation = _by_element_id(client, "select-aggregation")
        board = _by_element_id(client, "card-morph-custom-group")

    descendants = {id(el) for el in _descendants(advanced)}
    assert id(aggregation) in descendants, (
        "the level select is not inside the Advanced Settings card")
    assert id(board) not in descendants, (
        "the grouping board moved into the advanced card, which must stay the "
        "last child of the comparison column")


def test_intra_comparison_exposes_its_own_cache_switch():
    """Parity with the Connectivity tab: the cache choice is a control the user
    can see, not a default the tab reads silently on their behalf."""
    from nicegui import Client
    from nicegui.page import page

    from ui.tabs.morphology import create_morphology_tab

    client = Client(page("/morph-comparison-use-cache"))
    with client:
        create_morphology_tab()
        advanced = _by_element_id(client, "card-morphology-advanced")
        switches = [el for el in _descendants(advanced)
                    if getattr(el, "text", None) == "Use Cache"]

    assert len(switches) == 1, "expected exactly one Use Cache switch"
    source = (Path(__file__).resolve().parents[2]
              / "ui" / "tabs" / "morphology.py").read_text(encoding="utf-8")
    assert '"use_cache": bool(comparison_use_cache.value),' in source
