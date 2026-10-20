"""Heatmap pages in one family can share margins so equal canvases align."""

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from vispath_pkg.vispath import VisConnMatInteractive, heatmap_margins  # noqa: E402

LEFT = re.compile(r"const dynamicLeftMargin = (\d+);")
BOTTOM = re.compile(r"const dynamicBottomMargin = (\d+);")


def _margins(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return int(LEFT.search(text).group(1)), int(BOTTOM.search(text).group(1))


def _matrix(row_name, col_name, n=3):
    data = [[1.0 if i == j else 0.1 for j in range(n)] for i in range(n)]
    return pd.DataFrame(data,
                        index=[f"{row_name}{i}" for i in range(n)],
                        columns=[f"{col_name}{i}" for i in range(n)])


def _export(tmp_path, name, matrix, **kwargs):
    out = tmp_path / f"{name}.html"
    VisConnMatInteractive(matrix, str(out), title=name, showfig=False,
                          verbose=False, inline_plotly=False, **kwargs)
    assert out.is_file()
    return out


def test_margin_rule():
    assert heatmap_margins(["a"], ["b"]) == (120, 160)
    assert heatmap_margins(["x" * 30], ["y" * 30]) == (250, 270)
    assert heatmap_margins(["x" * 100], ["y" * 100]) == (400, 360)


def test_explicit_margins_reach_the_page(tmp_path):
    out = _export(tmp_path, "pinned", _matrix("row", "col"),
                  margin_left=285, margin_bottom=305)
    assert _margins(out) == (285, 305)


def test_sibling_pages_can_share_one_canvas(tmp_path):
    short = _matrix("r", "c")
    long = _matrix("row_" + "x" * 40, "col_" + "y" * 40)
    plain_short = _export(tmp_path, "plain_short", short)
    plain_long = _export(tmp_path, "plain_long", long)
    assert _margins(plain_short) != _margins(plain_long)

    shared = heatmap_margins(list(long.columns), list(long.index))
    a = _export(tmp_path, "shared_short", short,
                margin_left=shared[0], margin_bottom=shared[1])
    b = _export(tmp_path, "shared_long", long,
                margin_left=shared[0], margin_bottom=shared[1])
    assert _margins(a) == _margins(b) == shared
