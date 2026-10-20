"""Per-edge stroke pattern on the vispath network canvas.

Filed from `clock-production` for the one-interlayer conserved network: an
indirect connection must be readable as indirect, so its edges are drawn
dashed while a direct edge stays solid. The network renderer had a live
*manual* line-pattern control (it styles the selected edges and persists in a
saved layout) but no way for a caller to declare the pattern in the edge list,
so every generated page opened solid.

Design decisions these tests pin:

* the pattern is a **per-edge data channel** (`line_style` column,
  `solid`|`dashed`|`dotted`), not a page-wide switch: one page may mix an
  indirect edge with a direct one, and a future caller decides per row;
* an unknown value is dropped at load time rather than emitting a style
  Cytoscape does not know (it would silently render solid and read as a real
  choice);
* a page whose edge list carries no `line_style` column gains **no** rule and
  **no** edge-data key, so every existing page stays byte-identical.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/vispath/test_vispath_edge_line_style.py -q
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from vispath_pkg.vispath import VisualizePath  # noqa: E402


def _render(tmp_path, frame, name):
    vp = VisualizePath(path_file=frame, output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout="dagre")
    vp.base_filename = name
    vp.build_network()
    page = vp.create_network()
    return Path(page), Path(page).read_text(encoding="utf-8")


def _edge_payloads(html):
    match = re.search(r"const elements = \{\s*nodes:\s*(\[.*?\])\s*,"
                      r"\s*edges:\s*(\[.*?\])\s*\};", html, re.S)
    assert match, "embedded elements JSON not found"
    return json.loads(match.group(2))


def _frame(styles):
    return pd.DataFrame([
        {"source": src, "target": tgt, "weight": 10 + i,
         "ratio": 0.1, "ratio_adj": 0.2, "color": "#008048",
         **({"line_style": style} if style is not None else {})}
        for i, (src, tgt, style) in enumerate(zip(
            ["A", "R1", "C", "R2", "E"], ["R1", "B", "R2", "D", "F"],
            styles))
    ])


def test_dashed_and_dotted_strokes_are_declared_and_carried(tmp_path):
    page, html = _render(
        tmp_path, _frame(["dashed", "dashed", "dotted", "solid", None]),
        "stroke")
    assert page.is_file()
    for style in ("dashed", "dotted"):
        assert f'edge[line_style = "{style}"]' in html, style
        assert f"'line-style': '{style}'" in html, style
    payloads = {(e["data"]["source"], e["data"]["target"]): e["data"]
                for e in _edge_payloads(html)}
    assert payloads[("A", "R1")]["line_style"] == "dashed"
    assert payloads[("R1", "B")]["line_style"] == "dashed"
    assert payloads[("C", "R2")]["line_style"] == "dotted"
    # An explicit solid is carried (the export states the intent) but needs
    # no selector rule; a blank cell carries no key at all.
    assert payloads[("R2", "D")]["line_style"] == "solid"
    assert "line_style" not in payloads[("E", "F")]


def test_no_column_emits_no_style_machinery(tmp_path):
    _, html = _render(tmp_path, _frame([None, None, None, None, None]),
                      "plain")
    assert "line_style" not in html
    assert "edge[line_style" not in html


def test_unknown_pattern_is_dropped_not_styled(tmp_path):
    _, html = _render(tmp_path, _frame(["wavy", None, None, None]), "wavy")
    assert "edge[line_style" not in html
    payloads = {(e["data"]["source"], e["data"]["target"]): e["data"]
                for e in _edge_payloads(html)}
    assert "line_style" not in payloads[("A", "R1")]
