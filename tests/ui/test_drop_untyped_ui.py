"""Plan 2026-09-07 Phase C UI wiring: the checked-by-default
``Drop Untyped Neurons`` checkbox in both pathfinding tabs' Output Options,
its payload propagation, and the shared Default Settings entry.
"""

import pytest

from ui.config import DEFAULTS, DEFAULT_SETTING_SPECS, _coerce_user_default

PROJECT_ROOT_WEIGHTS = None  # placeholder to keep imports obvious


def _render_tab(module, route):
    from nicegui import Client
    from nicegui.page import page

    client = Client(page(route))
    with client:
        module()
    return client


def _checkbox_by_text(client, text):
    return [
        el for el in client.elements.values()
        if getattr(el, "text", "") == text
        and hasattr(el, "value")
    ]


@pytest.mark.parametrize("tab_module,route", [
    ("ui.tabs.find_path", "/drop-untyped-findpath"),
    ("ui.tabs.find_shortest", "/drop-untyped-findshortest"),
])
def test_checkbox_present_and_checked_by_default(tab_module, route,
                                                 monkeypatch):
    """The Output Options card shows the checkbox, checked by default
    (forced here by patching the saved-default lookup to True)."""
    import importlib

    mod = importlib.import_module(tab_module)
    monkeypatch.setattr(mod, "get_user_default",
                        lambda key: True if key == "drop_untyped" else
                        DEFAULTS.get(key))
    client = _render_tab(mod.create_find_path_tab
                         if "find_path" in tab_module
                         else mod.create_find_shortest_tab, route)

    boxes = _checkbox_by_text(client, "Drop Untyped Neurons")
    assert len(boxes) == 1, "checkbox missing from Output Options"
    assert boxes[0].value is True


@pytest.mark.parametrize("tab_module,route", [
    ("ui.tabs.find_path", "/drop-untyped-off-findpath"),
    ("ui.tabs.find_shortest", "/drop-untyped-off-findshortest"),
])
def test_checkbox_respects_saved_off_default(tab_module, route, monkeypatch):
    import importlib

    mod = importlib.import_module(tab_module)
    monkeypatch.setattr(mod, "get_user_default",
                        lambda key: False if key == "drop_untyped" else
                        DEFAULTS.get(key))
    client = _render_tab(mod.create_find_path_tab
                         if "find_path" in tab_module
                         else mod.create_find_shortest_tab, route)

    boxes = _checkbox_by_text(client, "Drop Untyped Neurons")
    assert len(boxes) == 1
    assert boxes[0].value is False


def test_payload_keys_forward_drop_untyped():
    """Both constructor payloads include ``drop_untyped`` wired from the
    checkbox (source-level wiring check)."""
    for tab_file in ("ui/tabs/find_path.py", "ui/tabs/find_shortest.py"):
        source = open(tab_file, encoding="utf-8").read()
        assert '"drop_untyped": drop_untyped.value' in source, tab_file
        assert 'checkbox_input(\n                    "Drop Untyped Neurons"' \
            in source or '"Drop Untyped Neurons"' in source, tab_file


def test_shared_default_setting_spec():
    """``drop_untyped`` is configurable in Settings -> Default Settings
    under Pathfinding & Output and shares the default with comparison."""
    spec = DEFAULT_SETTING_SPECS["drop_untyped"]
    assert spec["group"] == "pathfinding_output"
    assert spec["kind"] == "bool"
    assert DEFAULTS["drop_untyped"] is True
    # bool spec: real booleans coerce, truthy strings are rejected (the
    # saved override falls back to the built-in default)
    assert _coerce_user_default("drop_untyped", False) is False
    assert _coerce_user_default("drop_untyped", True) is True
    assert _coerce_user_default("drop_untyped", "false") is None


def test_inter_dataset_hint_mentions_shared_predicate_and_locations():
    source = open("ui/tabs/inter_dataset.py", encoding="utf-8").read()
    assert '"Drop Untyped Neurons"' in source
    assert "comparison_results/" in source
    assert "data_details/" in source
    assert '"drop_untyped": drop_untyped.value' in source


def test_inter_dataset_threshold_editor_uses_query_rows_in_core_parameters():
    """The cross-dataset editor exposes complete query rows, not schedules."""
    source = open("ui/tabs/inter_dataset.py", encoding="utf-8").read()
    assert 'section_header("Threshold Mode", "tune")' in source
    # Auto is the default, first-listed mode (no threshold chips needed);
    # Standard and Custom combination follow.
    assert 'threshold_mode_value = {"value": "auto"}' in source
    assert '("auto", "Auto")' in source
    assert '("standard", "Standard")' in source
    assert '("combinations", "Custom combination")' in source
    assert '"outline no-caps"' in source
    assert ('[0.001] if ratio_mode else [3]' in source
            and 'sorted(set(values))' in source)
    assert '"min-height: 3rem; font-size: 1.0rem; font-weight: 700;"' in source
    assert 'icon="delete_outline"' in source
    assert "Custom combination requires at least two selected" in source
    assert '"threshold_mode": threshold_mode' in source
    assert '"threshold_dataset_order": list(datasets)' in source
    assert '"threshold_combinations": threshold_combinations' in source
    assert "each row is one query" in source
    assert "per-dataset threshold" not in source
