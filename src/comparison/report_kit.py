"""Shared tabbed-report and heatmap machinery for the comparison features.

Extracted verbatim from ``ConnectivityProfileComparer``
(``comparison.profile_comparator``) so the connectivity-profiling export and
the cross-dataset morphology comparison render identically: one stylesheet,
one tab controller, one Ward-clustered Plotly heatmap card format, one
standalone VisPath heatmap writer with the interactive-heatmap fallback.

Feature-specific vocabulary (metric sets, direction labels, CSV paths, hero
copy) stays with the caller.  Per-metric presentation is expressed as a
``MetricStyle`` so non-connectivity metrics plug in without touching this
module.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dependency in practice
    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

try:
    from .connectivity_profiler import progress_bars_disabled
except ImportError:  # pragma: no cover - direct src/ execution
    try:
        from connectivity_profiler import progress_bars_disabled
    except ImportError:
        def progress_bars_disabled(show_progress: bool = True,
                                   verbose: bool = True) -> bool:
            return not show_progress or not verbose


# Metrics rendered with the diverging blue-white-red [-1, 1] scale; every
# other metric uses the positive white-red [0, 1] scale.
METRIC_DIVERGING_KEYS = frozenset({'rank_corr', 'rank_corr_union'})

REPORT_POSITIVE_COLORSCALE = (
    (0.0, '#ffffff'),
    (0.1, '#fff5f0'),
    (0.25, '#fee0d2'),
    (0.4, '#fcbba1'),
    (0.55, '#fc9272'),
    (0.7, '#fb6a4a'),
    (0.85, '#de2d26'),
    (1.0, '#a50f15'),
)
REPORT_DIVERGING_COLORSCALE = (
    (0.0, '#053061'),
    (0.1, '#2166ac'),
    (0.2, '#4393c3'),
    (0.3, '#92c5de'),
    (0.4, '#d1e5f0'),
    (0.5, '#ffffff'),
    (0.6, '#fddbc7'),
    (0.7, '#f4a582'),
    (0.8, '#d6604d'),
    (0.9, '#b2182b'),
    (1.0, '#67001f'),
)


@dataclass(frozen=True)
class MetricStyle:
    """Per-metric presentation: label plus color scale and value range."""

    key: str
    display_name: str
    colorscale: Tuple[Tuple[float, str], ...]
    zmin: float
    zmax: float


def metric_style(key: str, display_name: Optional[str] = None) -> MetricStyle:
    """Default style for a metric key (profiling scale conventions)."""
    if display_name is None:
        display_name = str(key).replace('_', ' ').title()
    if key in METRIC_DIVERGING_KEYS:
        return MetricStyle(key, display_name,
                           REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)
    return MetricStyle(key, display_name,
                       REPORT_POSITIVE_COLORSCALE, 0.0, 1.0)


def cluster_heatmap_matrix(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Apply the same Ward/Euclidean ordering used by VisPath.

    VisPath clusters a finite copy of the matrix, replacing missing values
    with zero before calculating row and column Euclidean distances.  The
    report follows that ordering while retaining missing cells as blanks
    in the displayed Plotly heatmap.
    """
    numeric = matrix.apply(pd.to_numeric, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.empty:
        return numeric, False

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        finite = numeric.fillna(0.0).to_numpy(dtype=float)
        row_order = list(range(numeric.shape[0]))
        col_order = list(range(numeric.shape[1]))
        if finite.shape[0] > 1:
            row_order = leaves_list(
                linkage(pdist(finite, metric='euclidean'), method='ward')
            ).tolist()
        if finite.shape[1] > 1:
            col_order = leaves_list(
                linkage(pdist(finite.T, metric='euclidean'), method='ward')
            ).tolist()
        return numeric.iloc[row_order, col_order], True
    except (ImportError, ValueError, TypeError, FloatingPointError):
        return numeric, False


def plotly_heatmap_fragment(
    matrix: pd.DataFrame,
    title: str,
    style: MetricStyle,
    x_title: str,
    y_title: str,
    include_plotlyjs: bool = False,
    square_cells: bool = False,
) -> Tuple[Optional[str], bool]:
    """Render one clustered Plotly heatmap fragment without cell labels.

    ``square_cells`` is used for similarity matrices whose row and column
    dimensions represent the same set of items.
    """
    if matrix is None or matrix.empty:
        return None, False

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None, False

    ordered, clustered = cluster_heatmap_matrix(matrix)
    z = [
        [None if pd.isna(value) else float(value) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    x_labels = [str(value) for value in ordered.columns]
    y_labels = [str(value) for value in ordered.index]

    heatmap_kwargs = {
        'z': z,
        'x': x_labels,
        'y': y_labels,
        'type': 'heatmap',
        'colorscale': style.colorscale,
        'zmin': style.zmin,
        'zmax': style.zmax,
        'hoverongaps': False,
        'connectgaps': False,
        'colorbar': {
            'title': {'text': style.display_name},
            'thickness': 12,
            'len': 0.86,
        },
        'hovertemplate': (
            '<b>%{y}</b><br>%{x}<br>'
            f'{style.display_name}: %{{z:.3f}}'
            '<extra></extra>'
        ),
    }

    fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])
    max_label_length = max((len(label) for label in y_labels), default=12)
    left_margin = min(235, max(90, max_label_length * 5 + 22))
    matrix_dimension = max(len(y_labels), len(x_labels))
    row_height = 18 if len(y_labels) > 60 else 23
    if square_cells:
        # Two-column cards are wider than the previous three-column
        # layout. Give square intra-dataset matrices enough vertical
        # room before Plotly applies the equal-axis constraint.
        row_height = 18 if matrix_dimension > 60 else 27
    height = max(335, min(880, 185 + len(y_labels) * row_height))
    if square_cells:
        height = max(390, height)
    xaxis = {
        'title': {'text': x_title, 'font': {'size': 10}},
        'tickangle': -42,
        'automargin': True,
        'showgrid': False,
        'zeroline': False,
        'showline': False,
    }
    yaxis = {
        'title': {'text': y_title, 'font': {'size': 10}},
        'autorange': 'reversed',
        'automargin': True,
        'showgrid': False,
        'zeroline': False,
        'showline': False,
    }
    if square_cells:
        # Equal axis scaling makes each matrix cell a true square even
        # when the responsive report card is resized.
        yaxis.update({
            'scaleanchor': 'x',
            'scaleratio': 1,
            'constrain': 'domain',
        })
    fig.update_layout(
        template='plotly_white',
        title={
            'text': title,
            'x': 0.01,
            'xanchor': 'left',
            'font': {'size': 13, 'color': '#152238'},
        },
        height=height,
        margin={
            'l': left_margin,
            'r': 12,
            't': 58,
            'b': 110 if len(x_labels) > 1 else 68,
        },
        font={'family': 'Inter, Arial, sans-serif', 'size': 10, 'color': '#152238'},
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=xaxis,
        yaxis=yaxis,
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs='inline' if include_plotlyjs else False,
        config={
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        },
        default_width='100%',
    ), clustered


def report_css() -> str:
    """Return the self-contained report stylesheet."""
    return """<style>
:root {
  --ink: #152238;
  --muted: #637188;
  --line: #d9e2ee;
  --canvas: #f4f7fb;
  --surface: #ffffff;
  --surface-soft: #f8fafc;
  --accent: #2563eb;
  --accent-soft: #e8f0ff;
  --accent-dark: #1746a2;
  --success: #16756a;
  --shadow: 0 12px 30px rgba(27, 52, 84, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.45;
}
a { color: var(--accent-dark); text-decoration: none; }
a:hover { text-decoration: underline; }
.report-shell { max-width: 1480px; margin: 0 auto; padding: 34px 30px 56px; }
.report-hero {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: var(--shadow);
  padding: 28px 30px 24px;
  margin-bottom: 22px;
}
.report-kicker {
  color: var(--accent-dark);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.report-title { margin: 7px 0 5px; font-size: clamp(28px, 4vw, 42px); line-height: 1.08; }
.report-subtitle { color: var(--muted); margin: 0; max-width: 920px; font-size: 15px; }
.report-meta { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 21px; }
.meta-chip {
  display: flex; gap: 7px; align-items: baseline; flex-wrap: wrap;
  background: var(--surface-soft); border: 1px solid var(--line);
  border-radius: 999px; padding: 7px 12px; font-size: 12px;
}
.meta-chip span { color: var(--muted); font-weight: 700; }
.meta-chip strong { font-weight: 700; }
.report-note {
  color: var(--muted); font-size: 12px; margin: 18px 0 0;
  padding-top: 15px; border-top: 1px solid var(--line);
}
.tab-list {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin: 0 0 18px; padding: 5px;
  background: #e9eef6; border: 1px solid var(--line); border-radius: 12px;
}
.tab-button {
  appearance: none; border: 1px solid transparent; border-radius: 9px;
  background: transparent; color: var(--muted); cursor: pointer;
  font: inherit; font-size: 13px; font-weight: 750; padding: 10px 15px;
  transition: background .15s ease, color .15s ease, box-shadow .15s ease;
}
.tab-button:hover { color: var(--ink); background: rgba(255,255,255,.72); }
.tab-button.active {
  color: var(--accent-dark); background: var(--surface);
  border-color: #cbd9f2; box-shadow: 0 3px 9px rgba(44, 74, 120, .10);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.section-card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; box-shadow: 0 5px 16px rgba(27, 52, 84, .04);
  padding: 22px; margin-bottom: 20px;
}
.section-heading { margin: 0 0 4px; font-size: 21px; }
.section-summary { color: var(--muted); font-size: 13px; margin: 0 0 18px; }
.direction-intro { margin: 2px 0 16px; }
.direction-title { margin: 0; font-size: 18px; }
.direction-note { color: var(--muted); font-size: 12px; margin-top: 3px; }
.level-block { margin: 24px 0 28px; }
.level-block:first-child { margin-top: 0; }
.level-heading { display: flex; align-items: baseline; gap: 9px; margin-bottom: 10px; }
.level-heading h3 { margin: 0; font-size: 15px; }
.level-heading span { color: var(--muted); font-size: 12px; }
.metric-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 15px; align-items: stretch;
}
.metric-grid.single { grid-template-columns: minmax(0, 1fr); }
.metric-grid.single .plotly-graph-div { max-width: 1100px; }
.heatmap-card {
  min-width: 0; background: var(--surface); border: 1px solid var(--line);
  border-radius: 13px; box-shadow: 0 5px 15px rgba(27, 52, 84, .05);
  overflow: hidden;
}
.heatmap-card-head { padding: 13px 14px 9px; min-height: 76px; }
.heatmap-card-title { margin: 0; font-size: 14px; line-height: 1.25; }
.heatmap-card-subtitle { color: var(--muted); font-size: 11px; margin-top: 4px; }
.heatmap-links { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 8px; font-size: 11px; font-weight: 700; }
.heatmap-links a { white-space: nowrap; }
.heatmap-status {
  display: inline-block; color: var(--success); background: #e7f6f2;
  border-radius: 999px; padding: 2px 7px; font-size: 10px; font-weight: 800;
}
.heatmap-stage { border-top: 1px solid #edf1f6; padding: 2px 3px 0; min-height: 335px; }
.heatmap-stage .plotly-graph-div { width: 100% !important; }
.heatmap-empty { color: var(--muted); font-size: 12px; padding: 40px 16px; text-align: center; }
.detail-block { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 14px; }
.detail-block summary { cursor: pointer; color: var(--accent-dark); font-size: 13px; font-weight: 750; }
.detail-list { columns: 2; margin: 10px 0 0; padding-left: 20px; font-size: 12px; }
.mapping-table { border-collapse: collapse; font-size: 12px; margin-top: 12px; width: 100%; }
.mapping-table th, .mapping-table td { border: 1px solid var(--line); padding: 7px 9px; text-align: left; }
.mapping-table th { background: var(--surface-soft); color: var(--muted); font-weight: 750; }
.muted { color: var(--muted); font-size: 12px; }
@media (max-width: 700px) {
  .report-shell { padding: 17px 12px 34px; }
  .report-hero, .section-card { padding: 17px; border-radius: 14px; }
  .metric-grid { grid-template-columns: 1fr; }
  .detail-list { columns: 1; }
  .tab-button { flex: 1 1 auto; }
}
</style>"""


def report_script() -> str:
    """Return the small tab/resize controller used by report.html."""
    return """<script>
(function () {
  function resizePlots(panel) {
    if (!panel || !window.Plotly) return;
    panel.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
      try { window.Plotly.Plots.resize(plot); } catch (error) { /* no-op */ }
    });
  }

  document.querySelectorAll('[data-tab-button]').forEach(function (button) {
    button.addEventListener('click', function () {
      var group = button.getAttribute('data-tab-group');
      var target = button.getAttribute('data-tab-target');
      document.querySelectorAll('[data-tab-button][data-tab-group="' + group + '"]')
        .forEach(function (peer) {
          var active = peer === button;
          peer.classList.toggle('active', active);
          peer.setAttribute('aria-selected', active ? 'true' : 'false');
        });
      document.querySelectorAll('[data-tab-panel-group="' + group + '"]')
        .forEach(function (panel) {
          panel.classList.toggle('active', panel.id === target);
        });
      requestAnimationFrame(function () { resizePlots(document.getElementById(target)); });
    });
  });

  window.addEventListener('load', function () {
    document.querySelectorAll('.tab-panel.active').forEach(resizePlots);
  });
}());
</script>"""


def append_report_tab_group(
    lines: List[str],
    group_id: str,
    tabs: List[Tuple[str, str]],
    render_panel: Callable[[str, str], None],
    panel_class: str = 'tab-panel',
) -> None:
    """Append a reusable button/panel tab group to the report."""
    if not tabs:
        return
    lines.append(f"<div class='tab-list' role='tablist' data-tab-list='{group_id}'>")
    panel_ids = []
    for index, (key, label) in enumerate(tabs):
        panel_id = f'{group_id}-panel-{index}'
        panel_ids.append((key, panel_id))
        active = ' active' if index == 0 else ''
        selected = 'true' if index == 0 else 'false'
        lines.append(
            f"<button type='button' class='tab-button{active}' "
            f"data-tab-button data-tab-group='{group_id}' "
            f"data-tab-target='{panel_id}' aria-selected='{selected}' "
            f"role='tab'>{escape(str(label))}</button>"
        )
    lines.append('</div>')
    for index, (key, panel_id) in enumerate(panel_ids):
        active = ' active' if index == 0 else ''
        lines.append(
            f"<section id='{panel_id}' class='{panel_class}{active}' "
            f"data-tab-panel-group='{group_id}' data-tab-key='{escape(str(key), quote=True)}' "
            "role='tabpanel'>"
        )
        render_panel(key, panel_id)
        lines.append('</section>')


def append_report_heatmap(
    lines: List[str],
    output_path: Path,
    matrix: Optional[pd.DataFrame],
    heading: str,
    title: str,
    style: MetricStyle,
    x_title: str,
    y_title: str,
    csv_rel: Optional[str],
    vispath_rel: Optional[str],
    plotly_state: Dict[str, bool],
    square_cells: bool = False,
) -> None:
    """Append one report card with a Plotly heatmap and source links."""
    links = []
    if csv_rel:
        links.append(f"<a href='{escape(csv_rel, quote=True)}'>CSV</a>")
    if vispath_rel and (output_path / vispath_rel).exists():
        links.append(
            f"<a href='{escape(vispath_rel, quote=True)}' target='_blank' "
            "rel='noopener'>Open VisPath heatmap for editing</a>"
        )
    links_html = ' <span aria-hidden="true">·</span> '.join(links)

    lines.append("<article class='heatmap-card'>")
    lines.append("<header class='heatmap-card-head'>")
    lines.append(f"<h4 class='heatmap-card-title'>{escape(heading)}</h4>")
    if links_html:
        lines.append(f"<div class='heatmap-links'>{links_html}</div>")

    if matrix is None or matrix.empty:
        lines.append("<div class='heatmap-card-subtitle'>Not computed for this run</div>")
        lines.append('</header><div class="heatmap-empty">No matrix available.</div></article>')
        return

    fragment, clustered = plotly_heatmap_fragment(
        matrix=matrix,
        title=title,
        style=style,
        x_title=x_title,
        y_title=y_title,
        include_plotlyjs=plotly_state.get('include_plotlyjs', True),
        square_cells=square_cells,
    )
    status_text = 'Ward clustered' if clustered else 'Original order'
    lines.append(
        "<div class='heatmap-card-subtitle'><span class='heatmap-status'>"
        f"{status_text}</span> · hover cells for exact values</div>"
    )
    lines.append('</header><div class="heatmap-stage">')
    if fragment:
        plotly_state['include_plotlyjs'] = False
        if not clustered:
            lines.append(
                "<div class='heatmap-card-subtitle muted'>"
                "Ward ordering unavailable; original order shown.</div>"
            )
        lines.append(fragment)
    else:
        lines.append(
            "<div class='heatmap-empty'>Plotly is unavailable; use the CSV or "
            "VisPath editor link above.</div>"
        )
    lines.append('</div></article>')


def append_report_metric_grid(
    lines: List[str],
    output_path: Path,
    metric_matrices: Optional[Dict[str, pd.DataFrame]],
    title_prefix: str,
    csv_paths: Dict[str, str],
    vispath_paths: Dict[str, str],
    x_title: str,
    y_title: str,
    plotly_state: Dict[str, bool],
    styles: Optional[Dict[str, MetricStyle]] = None,
    square_cells: bool = False,
) -> None:
    """Append metric cards in a responsive two-column grid.

    ``styles`` maps metric key -> MetricStyle and fixes the card order;
    when omitted, default styles are derived per key on the fly.
    """
    metric_matrices = metric_matrices or {}
    lines.append("<div class='metric-grid'>")
    if styles:
        items = [(key, styles[key]) for key in styles]
    else:
        items = [(key, metric_style(key)) for key in metric_matrices]
    if len(items) == 1:
        # One-card grids (single-metric features like the morphology
        # comparison) span the full width instead of leaving half of the
        # two-column grid empty.
        lines[-1] = "<div class='metric-grid single'>"
    for metric, style in items:
        display = style.display_name
        append_report_heatmap(
            lines=lines,
            output_path=output_path,
            matrix=metric_matrices.get(metric),
            heading=display,
            title=f"{title_prefix} · {display}",
            style=style,
            x_title=x_title,
            y_title=y_title,
            csv_rel=csv_paths.get(metric),
            vispath_rel=vispath_paths.get(metric),
            plotly_state=plotly_state,
            square_cells=square_cells,
        )
    lines.append('</div>')


def generate_heatmaps_fallback(
    cards: Dict[str, Dict[str, pd.DataFrame]],
    viz_dir: Path,
    styles: Optional[Dict[str, MetricStyle]],
    *,
    filename_builder: Callable[[str, str], str],
    group_display: Optional[Callable[[str], str]] = None,
    fallback_title: Optional[Callable[[str, str, str], str]] = None,
    show_figures: bool = False,
    verbose: bool = True,
    saved_files: Optional[Dict[str, List[str]]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Fallback heatmap generation using the interactive_heatmap module."""
    if saved_files is None:
        saved_files = {'heatmaps_generated': []}
    if saved_files.get('heatmaps_generated') is None:
        saved_files['heatmaps_generated'] = []
    if group_display is None:
        group_display = str
    if fallback_title is None:
        fallback_title = lambda group, key, gd: (  # noqa: E731
            f"{style_for(styles, key).display_name} - {gd}")
    try:
        from .interactive_heatmap import generate_interactive_heatmap
    except ImportError:  # pragma: no cover - direct src/ execution
        from interactive_heatmap import generate_interactive_heatmap
    for group, key_matrices in cards.items():
        for key, matrix in key_matrices.items():
            if matrix is None:
                continue
            html_path = viz_dir / filename_builder(group, key)
            generate_interactive_heatmap(
                matrices_dict={key: matrix},
                filename=str(html_path),
                title=fallback_title(group, key, group_display(group)),
                showfig=show_figures,
                verbose=verbose
            )
            saved_files['heatmaps_generated'].append(str(html_path))
            if log:
                log(f"Generated heatmap (fallback): {html_path}")


def generate_standalone_heatmaps(
    cards: Dict[str, Dict[str, pd.DataFrame]],
    viz_dir: Path,
    styles: Optional[Dict[str, MetricStyle]],
    *,
    filename_builder: Callable[[str, str], str],
    group_display: Optional[Callable[[str], str]] = None,
    vispath_title: Optional[Callable[[str, str, str], str]] = None,
    fallback_title: Optional[Callable[[str, str, str], str]] = None,
    tqdm_desc: str = 'Generating heatmaps',
    show_figures: bool = False,
    verbose: bool = True,
    saved_files: Optional[Dict[str, List[str]]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Write one standalone interactive heatmap per (group, key) matrix.

    Mirrors the connectivity-profiler behavior: non-finite values are
    replaced with 0 for rendering only (the caller's CSVs keep the
    originals; the returned count lets the caller log it), VisPath's
    ``VisConnMatInteractive`` provides native clustering, and any failure
    re-renders everything through ``interactive_heatmap``.

    Args:
        cards: ``{group: {metric key: DataFrame}}`` matrices to render.
        viz_dir: Output directory (created as needed).
        styles: metric key -> MetricStyle (color scale, range, label).
        filename_builder: ``(group, key) -> filename`` for the HTML files.
        group_display: reader-facing label for a group (default: as-is).
        vispath_title: ``(group, key, group_display) -> title`` for the
            VisPath render; a generic default is used when omitted.
        fallback_title: same contract for the fallback renderer.
        tqdm_desc: progress-bar description.
        show_figures / verbose: renderer and progress-bar behavior.
        saved_files: dict whose ``'heatmaps_generated'`` list accumulates
            the written paths (created when omitted).
        log: optional sink for fallback notices.

    Returns:
        Number of non-finite cells replaced for rendering.
    """
    if saved_files is None:
        saved_files = {'heatmaps_generated': []}
    if group_display is None:
        group_display = str
    if vispath_title is None:
        vispath_title = lambda group, key, gd: (  # noqa: E731
            f"{style_for(styles, key).display_name} - {gd}")
    if fallback_title is None:
        fallback_title = vispath_title

    # Cross-dataset similarities can legitimately contain NaN when a
    # profile has no comparable partners.  Keep those values in the CSV
    # analysis output, but pass finite copies to renderers that format
    # every cell as an integer/float for hover text.
    render_cards: Dict[str, Dict[str, pd.DataFrame]] = {}
    replaced_nonfinite = 0
    for group, key_matrices in cards.items():
        render_cards[group] = {}
        for key, matrix in key_matrices.items():
            if matrix is None:
                render_cards[group][key] = matrix
                continue
            numeric = matrix.apply(pd.to_numeric, errors='coerce')
            values = numeric.to_numpy(dtype=float, copy=False)
            nonfinite = ~np.isfinite(values)
            replaced_nonfinite += int(nonfinite.sum())
            if nonfinite.any():
                numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            render_cards[group][key] = numeric

    viz_dir.mkdir(parents=True, exist_ok=True)
    if saved_files.get('heatmaps_generated') is None:
        saved_files['heatmaps_generated'] = []

    def _render_fallback() -> None:
        try:
            generate_heatmaps_fallback(
                render_cards, viz_dir, styles,
                filename_builder=filename_builder,
                group_display=group_display,
                fallback_title=fallback_title,
                show_figures=show_figures,
                verbose=verbose,
                saved_files=saved_files,
                log=log,
            )
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            if log:
                log(f"Error generating heatmaps: {exc}")

    try:
        # Import VisualizePath's heatmap function
        vispath_path = Path(__file__).parent.parent.parent / 'vispath-subproject' / 'src'
        if str(vispath_path) not in sys.path:
            sys.path.insert(0, str(vispath_path))

        from vispath_pkg.vispath import VisConnMatInteractive

        total_heatmaps = sum(
            1 for _group, key_matrices in render_cards.items()
            for _key, matrix in key_matrices.items()
            if matrix is not None
        )

        pbar = tqdm(total=total_heatmaps, desc=tqdm_desc,
                    disable=progress_bars_disabled(True, verbose))
        for group, key_matrices in render_cards.items():
            for key, matrix in key_matrices.items():
                if matrix is None:
                    continue
                style = style_for(styles, key)
                if styles is not None and key not in styles and log:
                    # A missing style silently defaults to the positive
                    # [0, 1] scale — flag it, since the render then clips
                    # out-of-range values (e.g. signed metrics).
                    log(f"Warning: no MetricStyle for card key '{key}'; "
                        "using the default positive scale.")
                html_path = viz_dir / filename_builder(group, key)
                VisConnMatInteractive(
                    cmat=matrix,
                    filename=str(html_path),
                    title=vispath_title(group, key, group_display(group)),
                    matrices_dict=None,
                    showfig=show_figures,
                    verbose=False,  # Suppress individual clustering messages
                    init_clustered=True,
                    color_scale=style.colorscale,
                    zmin=style.zmin,
                    zmax=style.zmax,
                    metric_name=style.display_name,
                )

                saved_files['heatmaps_generated'].append(str(html_path))
                pbar.update(1)

        pbar.close()

    except ImportError as e:
        if log:
            log(f"Warning: Could not import VisualizePath for heatmaps: {e}")
        _render_fallback()
    except Exception as e:
        if log:
            log(f"Warning: VisualizePath heatmap generation failed: {e}")
        _render_fallback()

    return replaced_nonfinite


def style_for(styles: Optional[Dict[str, MetricStyle]], key: str) -> MetricStyle:
    """Look up a MetricStyle, deriving the profiling default when absent."""
    if styles and key in styles:
        return styles[key]
    return metric_style(key)
