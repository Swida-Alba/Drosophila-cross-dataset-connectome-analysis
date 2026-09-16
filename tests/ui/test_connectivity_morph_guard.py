"""Morph qualification option guard on the Find Homolog panel.

Intra-dataset runs qualify too (identity chain), so the checkbox keys on
the target family only — while the mapping_ref mode, which needs mapper
branch pools from a cross-dataset pair, greys out when Target = Source.
"""

from nicegui import Client
from nicegui.page import page


def _build_tab() -> Client:
    from ui.tabs.connectivity import create_connectivity_tab

    client = Client(page("/morph-qualification-guard"))
    with client:
        create_connectivity_tab()
    return client


def _elements(client: Client):
    source = target = mode = qualify = warning = None
    for element in client.elements.values():
        label = getattr(element, "_props", {}).get("label")
        if label == "Source Dataset" and source is None:
            source = element
        elif label == "Target Dataset" and target is None:
            target = element
        elif getattr(element, "options", None) == {
            "null": "Null bar (per-query null p95 + offset)",
            "mapping_ref": "Mapping-referenced floor (floors v3)",
        }:
            mode = element
        elif (
            getattr(element, "text", "") == "Morph Qualification"
            and hasattr(element, "enabled")
        ):
            qualify = element
        elif "text-amber-8" in getattr(element, "_classes", []) and hasattr(
            element, "text"
        ):
            warning = element
    assert source is not None and target is not None
    assert mode is not None and qualify is not None and warning is not None
    return source, target, mode, qualify, warning


def test_guard_enables_intra_but_greys_mapping_ref(monkeypatch):
    monkeypatch.delenv("DROCAT_UI_PORT", raising=False)
    client = _build_tab()
    source, target, mode, qualify, warning = _elements(client)

    source.set_value("male-cns:v1.0")
    target.set_value("male-cns:v1.0")
    assert qualify.enabled, "intra-dataset runs qualify (identity chain)"
    assert not mode.enabled, "mapping_ref greys out when Target = Source"
    assert not warning.visible

    target.set_value("flywire_FAFB_v783")
    assert qualify.enabled
    assert mode.enabled, "mapping_ref enabled for a cross-dataset pair"


def test_guard_disables_qualification_for_unsupported_target(monkeypatch):
    monkeypatch.delenv("DROCAT_UI_PORT", raising=False)
    client = _build_tab()
    source, target, mode, qualify, warning = _elements(client)

    source.set_value("male-cns:v1.0")
    target.set_value("hemibrain:v1.2.1")
    assert not qualify.enabled
    assert not mode.enabled
    assert warning.visible
