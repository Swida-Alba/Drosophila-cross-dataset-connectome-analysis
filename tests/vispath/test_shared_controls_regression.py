"""Regression tests for the shared vispath controls backend.

Asserts the single-source-of-truth contract:
  - the shared JS block (vispath_pkg.shared_controls.SHARED_JS) is embedded
    exactly once in each generated HTML (network / Sankey / heatmap) and the
    template-local duplicates are gone,
  - the heatmap exposes both PNG and SVG export buttons,
  - data-derived strings (title, storage key, node labels) are escaped so they
    cannot break out of the inline script or inject markup.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from vispath_pkg.shared_controls import js_escape, html_escape, json_safe  # noqa: E402
from vispath_pkg.vispath import VisualizePath, VisConnMatInteractive  # noqa: E402

SHARED_MARKER = "Shared vispath controls (vispath_pkg.shared_controls)"
POISON_LABEL = "<img src=x onerror=alert(1)>S</img>"


def _network_sankey_htmls(tmp_path):
    df = pd.DataFrame(
        {
            "path_block": ["S>A>B>T", "S>X>Y", "D>T"],
            "weights": [[10, 20, 30], [5, 8], [3]],
        }
    )
    vp = VisualizePath(
        path_file=df,
        output_folder=str(tmp_path),
        showfig=False,
        verbose=False,
        network_layout="dagre",
    )
    net_path = vp.create_network()
    sankey_path = vp.create_sankey()
    return Path(net_path).read_text(encoding="utf-8"), Path(sankey_path).read_text(encoding="utf-8")


def _heatmap_html(tmp_path, title="Test Heatmap - O'Brien"):
    rng = np.random.default_rng(0)
    cmat = pd.DataFrame(
        rng.random((5, 5)),
        index=[f"N{i}" for i in range(5)],
        columns=[f"T{i}" for i in range(5)],
    )
    out = tmp_path / "heat.html"
    VisConnMatInteractive(cmat, str(out), title=title, showfig=False, verbose=False)
    return out.read_text(encoding="utf-8")


class TestSharedBlockSingleSourceOfTruth:
    def test_network_embeds_shared_block_once(self, tmp_path):
        net_html, _ = _network_sankey_htmls(tmp_path)
        assert net_html.count(SHARED_MARKER) == 1
        assert net_html.count("function isColorDark(color)") == 1
        assert net_html.count("function exportSVG()") == 1
        # multi-line labels re-merge into one <text> with <tspan> children
        assert net_html.count("function mergeMultilineLabels()") == 1
        assert net_html.count("mergeMultilineLabels();") == 1

    def test_sankey_embeds_shared_block_once(self, tmp_path):
        _, sankey_html = _network_sankey_htmls(tmp_path)
        assert sankey_html.count(SHARED_MARKER) == 1
        assert sankey_html.count("function isColorDark(color)") == 1
        assert sankey_html.count("function exportPNG()") == 1
        assert sankey_html.count("function mergeMultilineLabels()") == 1

    def test_heatmap_embeds_shared_block_once(self, tmp_path):
        heat_html = _heatmap_html(tmp_path)
        assert heat_html.count(SHARED_MARKER) == 1
        assert heat_html.count("function isColorDark(color)") == 1
        assert heat_html.count("function mergeMultilineLabels()") == 1

    def test_heatmap_has_both_export_buttons(self, tmp_path):
        heat_html = _heatmap_html(tmp_path)
        assert 'onclick="exportPNG()"' in heat_html
        assert 'onclick="exportSVG()"' in heat_html


class TestInjectionEscaping:
    def test_heatmap_title_escaped_into_js_string(self, tmp_path):
        heat_html = _heatmap_html(tmp_path, title="Test Heatmap - O'Brien")
        # Apostrophe must be escaped inside the JS string literal
        assert "const originalTitle = 'Test Heatmap - O\\'Brien';" in heat_html
        # HTML title uses entities
        assert "<title>Test Heatmap - O&#39;Brien</title>" in heat_html

    def test_heatmap_storage_key_escaped(self, tmp_path):
        heat_html = _heatmap_html(tmp_path)
        match = re.search(r"const storageKey = '([^']*)';", heat_html)
        assert match, "storageKey literal not found"
        assert "heatmap_settings_heat#" in match.group(1)

    def test_poisoned_node_label_does_not_break_out_of_script(self, tmp_path):
        df = pd.DataFrame(
            {
                "path_block": [f"S>{POISON_LABEL}>T"],
                "weights": [[5, 6]],
            }
        )
        vp = VisualizePath(
            path_file=df,
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
            network_layout="dagre",
        )
        net_path = vp.create_network()
        net_html = Path(net_path).read_text(encoding="utf-8")
        # The raw poison must never appear as markup/JS: JSON embedding escapes
        # '<' as \u003c, and render sites use escapeHtml(...).
        assert "<img src=x onerror" not in net_html
        assert "\\u003cimg" in net_html
        assert "escapeHtml(data.label)" in net_html

    def test_json_safe_escapes_script_terminators(self):
        payload = {"label": "</script><script>alert(1)</script>"}
        dumped = json_safe(payload)
        assert "</script>" not in dumped
        assert "\\u003c" in dumped
        assert (
            '{"label": "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e"}'
            == dumped
        )

    def test_js_escape_handles_quotes_and_angle_brackets(self):
        escaped = js_escape("O'Brien</script>&")
        assert "\\'" in escaped
        assert "</script>" not in escaped
        assert "\\u003c" in escaped and "\\u0026" in escaped

    def test_html_escape_entities(self):
        assert html_escape("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"


class TestHeatmapCallerKwargs:
    def test_color_scale_seeds_custom_colorscale(self, tmp_path):
        rng = np.random.default_rng(1)
        cmat = pd.DataFrame(rng.random((4, 4)))
        out = tmp_path / "custom_scale.html"
        VisConnMatInteractive(
            cmat,
            str(out),
            color_scale=[[0.0, "#ffffff"], [1.0, "#ff0000"]],
            showfig=False,
            verbose=False,
        )
        heat_html = out.read_text(encoding="utf-8")
        assert "let currentColorscale = 'Custom';" in heat_html
        assert '[[0.0, "#ffffff"], [1.0, "#ff0000"]]' in heat_html

    def test_metric_name_overrides_colorbar_label(self, tmp_path):
        rng = np.random.default_rng(2)
        cmat = pd.DataFrame(rng.random((4, 4)))
        out = tmp_path / "metric_name.html"
        VisConnMatInteractive(
            cmat, str(out), metric_name="Jaccard Similarity", showfig=False, verbose=False
        )
        heat_html = out.read_text(encoding="utf-8")
        assert "const metricNameOverride = 'Jaccard Similarity';" in heat_html
        assert "metricDisplayNames[metricType] = metricNameOverride;" in heat_html

    def test_nan_matrix_does_not_crash(self, tmp_path):
        cmat = pd.DataFrame([[0.5, np.nan], [np.nan, 0.25]])
        out = tmp_path / "nan_matrix.html"
        VisConnMatInteractive(cmat, str(out), showfig=False, verbose=False)
        heat_html = out.read_text(encoding="utf-8")
        # Non-finite cells render as an em dash in hover text; json.dumps
        # escapes it as \u2014 (ensure_ascii) inside the embedded JSON.
        assert "\\u2014" in heat_html


class TestPptSafeSvgExport:
    """Plotly renders the heatmap colormap to a rows×cols canvas and embeds
    it as a data-URL PNG inside the exported SVG. The export then:
      - vectorizes the cells (one <rect> per cell) for heatmaps with at most
        100 cells, keeping text and axis elements vector;
      - keeps the embedded pixel image for larger heatmaps so the colors stay
        crisp (no diffusion) after PowerPoint's Convert-to-Shape."""

    def test_heatmap_export_vectorizes_small_heatmaps(self, tmp_path):
        heat_html = _heatmap_html(tmp_path)
        assert "function vectorizeHeatmapCells" in heat_html
        assert 'class="heatmap-cells-vector"' in heat_html
        assert "shape-rendering" in heat_html
        # Export path routes through the vectorizer
        assert "vectorizeHeatmapCells(svgString, imgTag, pngUrl)" in heat_html
        # Text must stay text: the export never outlines labels
        assert "decodeURIComponent(dataUrl.split(',')[1])" in heat_html

    def test_heatmap_export_keeps_pixel_image_above_100_cells(self, tmp_path):
        heat_html = _heatmap_html(tmp_path)
        # The vectorizer bails out (keeping Plotly's embedded pixel image)
        # when the heatmap has more than 100 cells
        assert "rows * cols > 100" in heat_html

    def test_network_and_sankey_exports_do_not_rasterize(self, tmp_path):
        net_html, sankey_html = _network_sankey_htmls(tmp_path)
        # Cytoscape-svg draws nodes/edges/labels as real SVG elements
        # (exported through the shared vector backend, never rasterized)
        assert "exportCytoscapeToImage" in net_html
        assert "cyObj.svg" in net_html
        # Sankey goes through Plotly's vector pipeline (nodes/links are paths)
        assert "exportPlotlyToImage" in sankey_html
        assert "vectorizeHeatmapCells" not in sankey_html
