"""Skeleton-tab wiring for the two new ``VisualizeSkeleton`` viewer knobs.

The backend gained ``freeze_view`` (the exported HTML viewer pins its 3D
scene axes so legend show/hide cannot rescale the framing) and
``plot_individuals(granularity=...)`` (one profile per legend entry, layer,
neuron type, or bodyId leaf).  These tests cover the front-end mirror: the
Settings defaults, the Skeleton tab controls, and the parameter dictionaries
the tab hands to the generated script.
"""

import asyncio
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from nicegui import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

import ui.history_store as hs  # noqa: E402
import ui.layer_style_store as layer_style_store  # noqa: E402
from ui.components.output_panel import OutputPanel  # noqa: E402
from ui.config import (  # noqa: E402
    DEFAULTS,
    DEFAULT_SETTING_SPECS,
    PROFILE_GRANULARITIES,
    PROFILE_GRANULARITY_CHOICES,
    get_user_default,
)


# ---------------------------------------------------------------------------
# Backend agreement
# ---------------------------------------------------------------------------

def test_profile_granularity_options_mirror_the_backend_constant():
    """The UI choice list must stay exactly the backend's legal values."""
    from visualize_skeleton import PROFILE_GRANULARITIES as backend_levels

    assert list(PROFILE_GRANULARITIES) == list(backend_levels)
    # PROFILE_GRANULARITY_CHOICES maps each backend token to its human label.
    assert set(PROFILE_GRANULARITY_CHOICES) == set(backend_levels)
    assert DEFAULTS["profile_granularity"] in backend_levels
    # The historical behaviour is the default, so an unchanged UI keeps it.
    assert DEFAULTS["profile_granularity"] == "legend"


def test_skeleton_knob_defaults_match_the_backend():
    """freeze_view and the granularity keyword keep the backend defaults."""
    import dataclasses
    import inspect

    from visualize_skeleton import VisualizeSkeleton

    fields = {f.name: f for f in dataclasses.fields(VisualizeSkeleton)}
    assert fields["freeze_view"].default is True
    assert DEFAULTS["freeze_view"] is fields["freeze_view"].default

    method_params = inspect.signature(VisualizeSkeleton.plot_individuals).parameters
    assert method_params["granularity"].default == "legend"


def test_new_knobs_are_registered_settings_defaults():
    """Both keys are Settings-registry defaults in the skeleton group."""
    for key in ("profile_granularity", "freeze_view"):
        assert key in DEFAULT_SETTING_SPECS, key
        spec = DEFAULT_SETTING_SPECS[key]
        assert spec["group"] == "skeleton_render", key
        assert key in DEFAULTS, key
        assert get_user_default(key) in (
            PROFILE_GRANULARITIES if key == "profile_granularity" else (True, False)
        )
    assert DEFAULT_SETTING_SPECS["profile_granularity"]["kind"] == "select"
    assert DEFAULT_SETTING_SPECS["profile_granularity"]["options"] == (
        PROFILE_GRANULARITIES)
    assert DEFAULT_SETTING_SPECS["freeze_view"]["kind"] == "bool"


# ---------------------------------------------------------------------------
# Skeleton tab controls
# ---------------------------------------------------------------------------

def _build_skeleton_tab():
    from ui.tabs.visualization import create_skeleton_tab

    client = Client(page(f"/skeleton-profile-options-{uuid.uuid4().hex}"))
    with client:
        create_skeleton_tab()
    return client


def _by_label(client, label):
    matches = [
        el for el in client.elements.values()
        if (getattr(el, "_props", None) or {}).get("label") == label
    ]
    assert len(matches) == 1, f"label {label!r}: {len(matches)} matches"
    return matches[0]


def _by_checkbox_text(client, label):
    """NiceGUI keeps a checkbox caption in ``text``, not in the ``label`` prop."""
    matches = [
        el for el in client.elements.values()
        if type(el).__name__ == "Checkbox" and getattr(el, "text", "") == label
    ]
    assert len(matches) == 1, f"checkbox {label!r}: {len(matches)} matches"
    return matches[0]


def test_skeleton_tab_exposes_granularity_and_freeze_view_controls(
    monkeypatch, tmp_path
):
    """The profiles block gains a human-labelled granularity select and the
    export block a freeze switch; both follow the persisted defaults."""
    monkeypatch.setattr(layer_style_store, "_store_dir", tmp_path / "tab_drafts")
    client = _build_skeleton_tab()

    granularity = _by_label(client, "Profile Granularity")
    assert granularity.value == get_user_default("profile_granularity")
    # NiceGUI dict options map the backend value to the human label, so the
    # value that reaches plot_individuals stays the backend token.
    assert granularity.options == PROFILE_GRANULARITY_CHOICES
    assert set(granularity.options) == set(PROFILE_GRANULARITIES)
    assert {
        option["label"] for option in granularity._props["options"]
    } == set(PROFILE_GRANULARITY_CHOICES.values())
    assert PROFILE_GRANULARITY_CHOICES["body"] == "Per bodyId"

    freeze = _by_checkbox_text(client, "Freeze 3D view when toggling the legend")
    assert freeze.value is get_user_default("freeze_view")

    texts = [
        getattr(el, "text", "")
        for el in client.elements.values()
        if getattr(el, "text", "")
    ]
    assert any("Profile Granularity" in t for t in texts), \
        "the individual-profile caption must describe the granularity choice"
    assert any("one profile per neuron" in t for t in texts), \
        "the granularity hint must warn about the per-bodyId file count"


# ---------------------------------------------------------------------------
# Generated-script parameter registration
# ---------------------------------------------------------------------------

def _capture_panel_run(monkeypatch):
    """Stub the panel runner and the UI methods its run path touches."""
    captured = []

    async def fake_run(self, runner, tool_name, constructor_params,
                       method_name, method_params=None, output_dir=None):
        captured.append((tool_name, constructor_params, method_params))
        return {"returncode": 1, "files": [], "duration": 0,
                "cancelled": False, "output_folder": None,
                "neuron_match": None}

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


def _run_skeleton_tab(client):
    """Seed one layer neuron and click Generate 3D Skeleton."""
    layer_box = next(
        el for el in client.elements.values()
        if getattr(el, "chip_input", None) is not None
        and el.chip_input._props.get("label") == "Layer neurons"
    )
    layer_box.add_values(["aMe12"])

    button = next(
        el for el in client.elements.values()
        if type(el).__name__ == "Button"
        and getattr(el, "text", "") == "Generate 3D Skeleton"
    )
    click = next(
        listener for listener in button._event_listeners.values()
        if listener.type == "click"
    )
    handler = click.handler.__closure__[0].cell_contents
    result = handler()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def test_skeleton_run_registers_both_knobs(monkeypatch, tmp_path):
    """freeze_view goes to the constructor dict (like legend_mode) and
    granularity to the method dict (like views / summary_format)."""
    monkeypatch.setattr(layer_style_store, "_store_dir", tmp_path / "tab_drafts")
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
    captured = _capture_panel_run(monkeypatch)
    client = _build_skeleton_tab()

    _by_label(client, "Profile Granularity").value = "body"
    _by_checkbox_text(
        client, "Freeze 3D view when toggling the legend").value = False
    _run_skeleton_tab(client)

    assert captured, "the tab never reached the runner"
    tool_name, constructor_params, method_params = captured[0]
    assert tool_name == "plot3d_skeleton"
    assert constructor_params["freeze_view"] is False
    assert "freeze_view" not in method_params
    assert method_params["granularity"] == "body"
    # The neighbouring knobs still ride along, so the new keys did not
    # displace the existing legend/profile wiring.
    assert constructor_params["legend_mode"] == get_user_default("legend_mode")
    assert method_params["summary_format"] == ["pdf"]

    # The default selection stays on the historical per-legend-entry level and
    # the pin stays on.
    default_client = _build_skeleton_tab()
    _run_skeleton_tab(default_client)
    _, default_constructor, default_method = captured[-1]
    assert default_constructor["freeze_view"] is True
    assert default_method["granularity"] == "legend"

