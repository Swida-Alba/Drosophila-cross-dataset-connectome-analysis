"""Find Driver Lines tab: dataset-aware query box + coverage-aware selector.

Covers the plan
(`_plan/plan-nb-find-lines-dataset-aware-queries.md` §6) UI contract:

* the dataset input is a multi-select with an exclusive '(all)' indicator;
* the run payload sends the dataset list (or None for '(all)') and routes
  to the expanded tool with the expansion toggle on;
* history is recorded with per-value dataset provenance like the
  Cross-Dataset tab;
* coverage from the persisted snapshot disables unhosted datasets
  (advisory) and annotates the aligned releases;
* unavailable-dataset runs warn (notification path) but never block.
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import src.neuronbridge_coverage as nbc  # noqa: E402
import ui.history_store as hs  # noqa: E402
import ui.type_suggestions as ts  # noqa: E402
from ui.components.output_panel import OutputPanel  # noqa: E402
from ui.config import DATASETS  # noqa: E402
from ui.tabs.nb_find_lines import _KNOWN_COVERED_DEFAULT, _ALL_DATASETS  # noqa: E402


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
    return hs


@pytest.fixture
def coverage_isolated(tmp_path, monkeypatch):
    """Point the coverage snapshot store at a tmp dir; stub the background
    refresh so tab builds stay network-free."""
    root = tmp_path / "nb-cache"
    monkeypatch.setattr(nbc, "default_cache_root", lambda: root)
    monkeypatch.setattr(
        nbc, "refresh",
        lambda datasets, **kwargs: False)
    return root


def _write_snapshot(root, datasets_payload):
    path = root / nbc.SNAPSHOT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "nb_version": "v3_10_0",
        "created_at": "2026-09-15T00:00:00+00:00",
        "datasets": datasets_payload,
    }), encoding="utf-8")


def _mock_output_panel_run(monkeypatch, returncode=0):
    captured = []

    async def fake_run(self, runner, tool_name, constructor_params,
                       method_name, method_params=None, output_dir=None):
        captured.append({
            "tool_name": tool_name,
            "constructor_params": constructor_params,
            "method_params": method_params,
        })
        return {"returncode": returncode, "files": [], "duration": 0,
                "cancelled": False, "output_folder": None}

    monkeypatch.setattr(OutputPanel, "run", fake_run)
    monkeypatch.setattr(OutputPanel, "set_running", lambda self, value: None)
    monkeypatch.setattr(OutputPanel, "set_status",
                        lambda self, status, color="grey": None)
    monkeypatch.setattr(OutputPanel, "clear", lambda self: None)
    monkeypatch.setattr(OutputPanel, "show_files",
                        lambda self, files, output_dir=None: None)
    monkeypatch.setattr(OutputPanel, "log",
                        lambda self, message, level="stdout": None)
    return captured


def _chip_input(client, label):
    return next(
        element for element in client.elements.values()
        if getattr(element, "chip_input", None) is not None
        and element.chip_input._props.get("label") == label
    )


def _dataset_selector(client):
    return next(
        element for element in client.elements.values()
        if type(element).__name__ == "Select"
        and "Datasets" in str(element._props.get("label", ""))
    )


def _click_run(client, label):
    button = next(
        element for element in client.elements.values()
        if type(element).__name__ == "Button"
        and getattr(element, "text", "") == label
    )
    click = next(
        listener for listener in button._event_listeners.values()
        if listener.type == "click"
    )
    handler = click.handler.__closure__[0].cell_contents
    result = handler()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def _build_tab(url):
    client = Client(page(url))
    with client:
        from ui.tabs.nb_find_lines import create_nb_find_lines_tab
        create_nb_find_lines_tab()
    return client


# ---------------------------------------------------------------------------
# selector shape + defaults
# ---------------------------------------------------------------------------

class TestDatasetSelector:
    def test_multi_select_with_all_indicator_and_covered_default(
        self, isolated_history, coverage_isolated
    ):
        client = _build_tab("/nbfl-selector-default")
        selector = _dataset_selector(client)
        assert selector._props.get("multiple") is True
        assert _ALL_DATASETS in selector.options
        assert "(all) — search everywhere" == selector.options.get(_ALL_DATASETS)
        assert set(selector.value) == set(
            ds for ds in _KNOWN_COVERED_DEFAULT if ds in DATASETS)

    def test_release_upgrade_notice_never_renders_in_the_tab(
        self, isolated_history, coverage_isolated
    ):
        # NeuronBridge hosts male-cns:v0.9 data, so the selector's
        # "newer release" recommendation has nothing to offer here and
        # is opted out entirely.
        client = _build_tab("/nbfl-no-release-notice")
        assert not any(
            element._props.get("data-testid") == "dataset-release-notice"
            for element in client.elements.values()
        )

    def test_all_exclusivity_handler(self, isolated_history, coverage_isolated):
        client = _build_tab("/nbfl-all-exclusive")
        selector = _dataset_selector(client)

        # The exclusivity callback is registered via on_value_change and
        # normally fires from the websocket; invoke it directly here.
        handler = selector._change_handlers[-1]

        def _transition(value):
            """Sync the tracked previous selection, then apply the change."""
            handler(types.SimpleNamespace(args=None, sender=selector))
            selector.value = value
            handler(types.SimpleNamespace(args=None, sender=selector))

        # Establish '(all)' as the current selection, then pick a dataset:
        # the newest addition wins and '(all)' is dropped.
        selector.value = [_ALL_DATASETS]
        handler(types.SimpleNamespace(args=None, sender=selector))
        selector.value = [_ALL_DATASETS, "hemibrain:v1.2.1"]
        handler(types.SimpleNamespace(args=None, sender=selector))
        assert selector.value == ["hemibrain:v1.2.1"]

        # From a dataset selection, picking '(all)' drops the datasets.
        _transition(["hemibrain:v1.2.1", _ALL_DATASETS])
        assert selector.value == [_ALL_DATASETS]


# ---------------------------------------------------------------------------
# coverage-aware selector
# ---------------------------------------------------------------------------

class TestCoverageAwareSelector:
    def test_persisted_snapshot_disables_unhosted_and_notes_aligned(
        self, isolated_history, coverage_isolated
    ):
        _write_snapshot(coverage_isolated, {
            "banc_v626": {
                "status": "unavailable",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
            "male-cns:v1.0": {
                "status": "aligned",
                "hosted_version": "male-cns:v0.9",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        client = _build_tab("/nbfl-coverage-disable")
        selector = _dataset_selector(client)

        assert selector._drocat_nb_unavailable == ["banc_v626"]
        disable_prop = selector._props.get(":option-disable", "")
        assert "banc_v626" in disable_prop

        labels = [
            element.text for element in client.elements.values()
            if getattr(element, "text", "") and "NeuronBridge coverage" in str(getattr(element, "text"))
        ]
        assert labels, "coverage caption should be visible"
        assert "banc_v626" in labels[0]
        assert "male-cns:v1.0 → male-cns:v0.9" in labels[0]

    def test_missing_snapshot_disables_nothing(
        self, isolated_history, coverage_isolated
    ):
        client = _build_tab("/nbfl-coverage-empty")
        selector = _dataset_selector(client)
        assert getattr(selector, "_drocat_nb_unavailable", []) == []


# ---------------------------------------------------------------------------
# run payload + history
# ---------------------------------------------------------------------------

class TestRunPayloadAndHistory:
    def test_run_routes_to_expanded_tool_with_dataset_list(
        self, isolated_history, coverage_isolated, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        pools = {
            "hemibrain:v1.2.1": {"type": [("Tm4", "type")]},
            "male-cns:v1.0": {"type": [("Tm4", "type"), ("L2", "type")]},
        }
        monkeypatch.setattr(ts, "get_dataset_pools", lambda ds: pools.get(str(ds), {}))

        client = _build_tab("/nbfl-run-expanded")
        selector = _dataset_selector(client)
        selector.value = ["hemibrain:v1.2.1", "male-cns:v1.0"]
        _chip_input(client, "EM Neurons (bodyId, type, or instance)").add_values(
            ["Tm4"])
        _click_run(client, "Find Driver Lines")

        assert captured, "run should have been invoked"
        run_info = captured[0]
        assert run_info["tool_name"] == "nb_find_lines_expanded"
        params = run_info["method_params"]
        assert params["dataset"] == ["hemibrain:v1.2.1", "male-cns:v1.0"]
        assert params["expand_names"] is True
        assert params["coverage_datasets"] == DATASETS
        assert params["queries"] == ["Tm4"]
        # Full is the default output detail; the match-cache default comes
        # from Settings (off unless the user re-enables it).
        assert params["keep_per_match_csv"] is True
        assert params["cleanup_source_images"] is False
        assert params["compact_keep_last_n"] == 1
        assert run_info["constructor_params"]["use_cache"] is False

        # History: scoped to the selection, per-value provenance from the
        # exact local-pool membership (Tm4 resolves in both pools).
        assert hs.recent() == ["Tm4"]
        assert hs.datasets_of("Tm4") == [
            "hemibrain:v1.2.1", "male-cns:v1.0"]

    def test_all_run_sends_none_and_records_unscoped(
        self, isolated_history, coverage_isolated, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = _build_tab("/nbfl-run-all")
        selector = _dataset_selector(client)
        selector.value = [_ALL_DATASETS]
        # A pattern chip resolves in no local pool, so with '(all)' it must
        # stay unscoped (visible in every dataset's history list).  A
        # numeric chip, in contrast, resolves through the real bundled
        # bodyId pools and carries exactly those datasets.
        _chip_input(client, "EM Neurons (bodyId, type, or instance)").add_values(
            ["PPL.*"])
        _click_run(client, "Find Driver Lines")

        params = captured[0]["method_params"]
        assert params["dataset"] is None

        assert hs.recent() == ["PPL.*"]
        assert hs.datasets_of("PPL.*") == []

    def test_expansion_toggle_off_routes_to_the_plain_tool(
        self, isolated_history, coverage_isolated, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = _build_tab("/nbfl-run-plain")
        toggle = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("id") == "checkbox-nb-expand-names"
        )
        toggle.value = False
        selector = _dataset_selector(client)
        selector.value = ["hemibrain:v1.2.1"]
        _chip_input(client, "EM Neurons (bodyId, type, or instance)").add_values(
            ["Tm4"])
        _click_run(client, "Find Driver Lines")

        run_info = captured[0]
        assert run_info["tool_name"] == "nb_find_lines"
        assert "expand_names" not in run_info["method_params"]
        assert "coverage_datasets" not in run_info["method_params"]

    def test_compact_output_detail_flips_the_flags(
        self, isolated_history, coverage_isolated, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = _build_tab("/nbfl-run-compact")
        detail = next(
            element for element in client.elements.values()
            if type(element).__name__ == "Toggle"
        )
        detail.value = "compact"
        selector = _dataset_selector(client)
        selector.value = ["hemibrain:v1.2.1"]
        _chip_input(client, "EM Neurons (bodyId, type, or instance)").add_values(
            ["Tm4"])
        _click_run(client, "Find Driver Lines")

        params = captured[0]["method_params"]
        assert params["keep_per_match_csv"] is False
        assert params["cleanup_source_images"] is True


def test_settings_default_nb_use_cache_off_by_default():
    from ui.config import DEFAULTS, DEFAULT_SETTING_SPECS

    assert DEFAULTS["nb_use_cache"] is False
    spec = DEFAULT_SETTING_SPECS["nb_use_cache"]
    assert spec["kind"] == "bool"
    assert "NeuronBridge" in spec["label"]
