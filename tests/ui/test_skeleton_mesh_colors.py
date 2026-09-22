"""The Skeleton tab's mesh-color trios reach the backend per half.

The tab used to send only ``brain_mesh_color``, so the nerve cord -- ticked on,
or only embedded hidden for a viewer to reveal from the legend tree -- kept the
auto blue whatever the user had picked for the outline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _El:
    """A NiceGUI element's only relevant surface here: ``.value``."""

    def __init__(self, value):
        self.value = value


def _compose(auto, color, opacity):
    from ui.tabs.visualization import mesh_color_backend_value
    return mesh_color_backend_value(_El(auto), _El(color), _El(opacity))


def test_auto_wins_while_the_auto_box_is_on():
    assert _compose(True, '#ff0000', 0.2) == 'auto'


def test_a_picked_color_carries_the_chosen_opacity():
    assert _compose(False, '#74A8D6', 0.04) == 'rgba(116, 168, 214, 0.04)'


def test_shorthand_hex_expands_and_solid_stays_hex():
    assert _compose(False, '#abc', 0.5) == 'rgba(170, 187, 204, 0.5)'
    assert _compose(False, '#abc', 1.0) == '#abc'


def test_an_unparseable_color_is_passed_through_untouched():
    # The backend validator owns that error; the UI must not invent a color.
    assert _compose(False, 'not-a-hex', 0.5) == 'not-a-hex'


def test_the_tab_builds_one_trio_per_half():
    """Both keys leave the tab, or the VNC half is still stuck on auto."""
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.visualization import create_skeleton_tab

    client = Client(page('/mesh-color-trios'))
    with client:
        create_skeleton_tab()
    labels = [
        getattr(el, '_props', {}).get('label')
        for el in client.elements.values()
        if getattr(el, '_props', {}).get('label')
    ]
    for label in ('Brain Mesh Color', 'Brain Opacity',
                  'VNC Mesh Color', 'VNC Opacity'):
        assert label in labels, f'{label} is missing from the Skeleton tab'
