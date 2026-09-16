"""UI regression tests for subcaller skeleton-visualization defaults.

Similarity (Morphology and Connectivity Find Similar sub-tabs) and
NeuronBridge analysis panels default to line mode with the fast pipeline,
while the dedicated Skeleton tab keeps its independent tube/fast
configuration. Fetched skeletons cache by default (portable .swc.zst
shared cache), and the FlyWire branch of the cache-default logic keeps the
prepared mesh cache enabled even when line mode disables the NeuPrint
method selector.
"""

import sys
from pathlib import Path

import pytest
from nicegui import Client
from nicegui.page import page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ui.config as _cfg  # noqa: E402
from ui.components.skeleton_visualization_settings import (  # noqa: E402
    skeleton_visualization_settings,
)
from ui.tabs.connectivity import create_connectivity_tab  # noqa: E402
from ui.tabs.morphology import create_morphology_tab  # noqa: E402
from ui.tabs.nb_find_neuron import create_nb_find_neuron_tab  # noqa: E402
from ui.tabs.visualization import create_skeleton_tab  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_user_defaults(tmp_path, monkeypatch):
    """These tests assert built-in defaults; keep real saved overrides in
    ui/local_config.json from leaking into the assertions."""
    monkeypatch.setattr(_cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")


def _labeled(client):
    return {
        element._props.get("label"): element
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("label")
    }


def _checkbox_by_text(client, text):
    return next(
        element for element in client.elements.values()
        if getattr(element, "text", "") == text
    )


def _build(factory, url):
    client = Client(page(url))
    with client:
        factory()
    return client, _labeled(client)


def test_similarity_panels_default_to_line_with_fast_pipeline():
    """Both Find Similar editors start in line mode with the fast method."""
    for factory, url in (
        (create_morphology_tab, "/morphology-subcaller-line-defaults"),
        (create_connectivity_tab, "/connectivity-subcaller-line-defaults"),
    ):
        client, _by_label = _build(factory, url)

        modes = [
            element.value for element in client.elements.values()
            if getattr(element, "_props", {}).get("label") == "Skeleton Mode"
        ]
        methods = [
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label") == "Simplification Method"
        ]
        # The Morphology tab hosts two editors since the Comparison
        # sub-tab landed (Find Similar + Comparison); Connectivity keeps
        # one (Find Similar).  Every editor starts in line mode with the
        # fast method.
        expected = 2 if factory is create_morphology_tab else 1
        assert len(modes) == expected
        assert set(modes) == {"line"}
        assert [method.value for method in methods] == ["fast"] * expected
        # Line mode bypasses the pipeline selector.
        assert all(method.enabled is False for method in methods)


def test_neuronbridge_panel_defaults_to_line_with_fast_pipeline():
    client, by_label = _build(
        create_nb_find_neuron_tab, "/nb-subcaller-line-defaults")

    mode = by_label["Skeleton Mode"]
    method = by_label["Simplification Method"]
    assert mode.value == "line"
    assert method.value == "fast"
    assert method.enabled is False


def test_dedicated_skeleton_tab_keeps_independent_tube_default():
    """The Skeleton tab does not inherit the analysis line default."""
    client, by_label = _build(create_skeleton_tab, "/skeleton-tab-defaults")

    assert by_label["Skeleton Mode"].value == "tube"
    assert by_label["Simplification Method"].value == "fast"
    # NeuPrint tube mode leaves the method selector enabled.
    assert by_label["Simplification Method"].enabled is True


def test_explicit_tube_override_preserved():
    """Callers can still opt into tube mode via the shared component."""
    client, by_label = _build(
        lambda: skeleton_visualization_settings(default_skeleton_mode="tube"),
        "/subcaller-tube-override",
    )

    assert by_label["Skeleton Mode"].value == "tube"
    assert by_label["Simplification Method"].value == "fast"
    assert by_label["Simplification Method"].enabled is True


def test_flywire_keeps_prepared_mesh_cache_default_in_line_mode():
    """FAFB analysis renders keep the prepared mesh cache default even
    though line mode bypasses the method selector."""
    client, by_label = _build(
        lambda: skeleton_visualization_settings(
            default_skeleton_mode="line",
            dataset_provider=lambda: "flywire_FAFB_v783",
        ),
        "/subcaller-flywire-cache-default",
    )

    assert by_label["Skeleton Mode"].value == "line"
    assert by_label["Simplification Method"].value == "fast"
    assert by_label["Simplification Method"].enabled is False
    assert _checkbox_by_text(client, "Cache Neurons").value is True


def test_flywire_tube_keeps_method_selector_enabled():
    """Tube-mode FAFB renders expose fast/fine/artistic selection."""
    client, by_label = _build(
        lambda: skeleton_visualization_settings(
            default_skeleton_mode="tube",
            dataset_provider=lambda: "flywire_FAFB_v783",
        ),
        "/subcaller-flywire-tube-method",
    )

    assert by_label["Skeleton Mode"].value == "tube"
    assert by_label["Simplification Method"].enabled is True


def test_neuprint_fast_pipeline_caches_by_default():
    """Fetched skeletons persist as portable .swc.zst files in the shared
    cache, so even NeuPrint fast renders start with Cache Neurons on."""
    client, by_label = _build(
        lambda: skeleton_visualization_settings(
            default_skeleton_mode="tube",
            dataset_provider=lambda: "male-cns:v1.0",
        ),
        "/subcaller-neuprint-fast-cache-default",
    )

    assert by_label["Simplification Method"].value == "fast"
    assert _checkbox_by_text(client, "Cache Neurons").value is True
