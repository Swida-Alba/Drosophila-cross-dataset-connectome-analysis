"""The custom color panel must show the ramp the page actually draws.

Callers pass a color_scale to VisConnMatInteractive (the bodyId Jaccard and
vector_v2 pages pass white->theme and slate->white->theme). The panel used to
open on its legacy two-point defaults, so the reader saw purple swatches while
the heatmap was orange. These tests run the exported page and read the inputs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg.vispath import VisConnMatInteractive  # noqa: E402

from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402


def _export(tmp_path, name, scale, zmin, zmax):
    matrix = pd.DataFrame(np.eye(3) + 0.1,
                          index=["a", "b", "c"], columns=["a", "b", "c"])
    out = tmp_path / f"{name}.html"
    VisConnMatInteractive(
        matrix, str(out), title=name, showfig=False, verbose=False,
        inline_plotly=True, color_scale=scale, zmin=zmin, zmax=zmax)
    assert out.is_file()
    return out


@REQUIRES_CDP
def test_three_stop_scale_seeds_panel(tmp_path):
    scale = [[0.0, "#5B6B7F"], [0.5, "#FFFFFF"], [1.0, "#536CA3"]]
    out = _export(tmp_path, "three", scale, -1.0, 1.0)
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        assert page.wait(
            "return document.getElementById('use3PointScale').checked === true;")
        state = page.js(r"""
            return {
              checked: document.getElementById('use3PointScale').checked,
              two: document.getElementById('twoPointColors').style.display,
              three: document.getElementById('threePointColors').style.display,
              lo: document.getElementById('colorMin3').value.toLowerCase(),
              mid: document.getElementById('colorMid3').value.toLowerCase(),
              hi: document.getElementById('colorMax3').value.toLowerCase(),
              vlo: document.getElementById('valueMin3').value,
              vmid: document.getElementById('valueMid3').value,
              vhi: document.getElementById('valueMax3').value
            };
        """)
        assert state == {"checked": True, "two": "none", "three": "block",
                         "lo": "#5b6b7f", "mid": "#ffffff",
                         "hi": "#536ca3", "vlo": "-1", "vmid": "0",
                         "vhi": "1"}, state
        assert page.errors() == []


@REQUIRES_CDP
def test_two_stop_scale_seeds_panel(tmp_path):
    scale = [[0.0, "#FFFFFF"], [1.0, "#536CA3"]]
    out = _export(tmp_path, "two", scale, 0.0, 1.0)
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        assert page.wait(
            "return document.getElementById('colorMax').value.toLowerCase()"
            " === '#536ca3';")
        state = page.js(r"""
            return {
              checked: document.getElementById('use3PointScale').checked,
              two: document.getElementById('twoPointColors').style.display,
              three: document.getElementById('threePointColors').style.display,
              lo: document.getElementById('colorMin').value.toLowerCase(),
              hi: document.getElementById('colorMax').value.toLowerCase(),
              vlo: document.getElementById('valueMin2').value,
              vhi: document.getElementById('valueMax2').value
            };
        """)
        assert state == {"checked": False, "two": "block", "three": "none",
                         "lo": "#ffffff", "hi": "#536ca3", "vlo": "0",
                         "vhi": "1"}, state
        assert page.errors() == []
