"""UI regression tests for advisory newer-release selector notices."""

from nicegui import Client
from nicegui.page import page


def _buttons(client, prefix):
    return [
        element
        for element in client.elements.values()
        if str(getattr(element, "text", "")).startswith(prefix)
    ]


def _click(button):
    listener = next(iter(button._event_listeners.values()))
    listener.handler(None)


def test_single_selector_notice_requires_explicit_upgrade_action():
    from ui.components.common import dataset_selector

    client = Client(page("/dataset-release-notice-single"))
    with client:
        selector = dataset_selector(
            datasets=["male-cns:v0.9", "male-cns:v1.0"],
            default="male-cns:v0.9",
            show_local_status=False,
        )

    notice = selector._drocat_release_notice
    assert notice.visible is True
    assert notice._props["role"] == "status"
    assert notice._props["aria-live"] == "polite"
    assert _buttons(client, "Use male-cns:v1.0")
    assert selector.value == "male-cns:v0.9"

    _click(_buttons(client, "Use male-cns:v1.0")[0])
    assert selector.value == "male-cns:v1.0"
    assert notice.visible is False

    selector.set_value("male-cns:v0.9")
    assert notice.visible is True
    _click(_buttons(client, "Keep current")[0])
    assert notice.visible is False


def test_notice_is_non_actionable_when_newer_release_is_unavailable():
    from ui.components.common import dataset_selector

    client = Client(page("/dataset-release-notice-unavailable"))
    with client:
        selector = dataset_selector(
            datasets=["male-cns:v0.9"],
            default="male-cns:v0.9",
            show_local_status=False,
        )

    notice = selector._drocat_release_notice
    assert notice.visible is True
    assert not _buttons(client, "Use male-cns:v1.0")
    assert any(
        "not currently available" in str(getattr(element, "text", ""))
        for element in client.elements.values()
    )


def test_multi_selector_suppresses_notice_when_newer_release_is_selected():
    from ui.components.common import dataset_multi_selector

    client = Client(page("/dataset-release-notice-multi"))
    with client:
        selector = dataset_multi_selector(
            datasets=["male-cns:v0.9", "male-cns:v1.0", "manc:v1.0"],
            default=["male-cns:v0.9", "male-cns:v1.0"],
            show_local_status=False,
        )

    notice = selector._drocat_release_notice
    assert notice.visible is False
    selector.set_value(["male-cns:v0.9"])
    assert notice.visible is True


def test_selectors_can_opt_out_of_the_release_notice():
    from ui.components.common import dataset_multi_selector, dataset_selector

    client = Client(page("/dataset-release-notice-opt-out"))
    with client:
        single = dataset_selector(
            datasets=["male-cns:v0.9", "male-cns:v1.0"],
            default="male-cns:v0.9",
            show_local_status=False,
            show_release_notice=False,
        )
        multi = dataset_multi_selector(
            datasets=["male-cns:v0.9", "male-cns:v1.0"],
            default=["male-cns:v0.9"],
            show_local_status=False,
            show_release_notice=False,
        )

    for selector in (single, multi):
        assert not hasattr(selector, "_drocat_release_notice")
    assert not any(
        element._props.get("data-testid") == "dataset-release-notice"
        for element in client.elements.values()
    )
