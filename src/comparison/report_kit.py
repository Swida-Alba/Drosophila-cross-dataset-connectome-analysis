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
    class _NullBar:
        """Stand-in for tqdm's kwargs-only call form (total=..., ...) —
        the previous bare-function shim raised TypeError at that call
        site and None.update() AttributeError after it."""

        def update(self, _n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else _NullBar()

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


def cluster_heatmap_matrix_with_linkage(
    matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply the same Ward/Euclidean ordering used by VisPath, keeping the linkages.

    VisPath clusters a finite copy of the matrix, replacing missing values
    with zero before calculating row and column Euclidean distances.  The
    report follows that ordering while retaining missing cells as blanks
    in the displayed Plotly heatmap.

    Returns ``(ordered, clustered, row_linkage, col_linkage)`` where the
    linkage matrices are the scipy Ward linkages behind the applied leaf
    orders (``None`` for axes with fewer than two labels or on failure) —
    the raw material for drawing dendrograms beside the heatmap.
    """
    numeric = matrix.apply(pd.to_numeric, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.empty:
        return numeric, False, None, None

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        finite = numeric.fillna(0.0).to_numpy(dtype=float)
        row_order = list(range(numeric.shape[0]))
        col_order = list(range(numeric.shape[1]))
        row_linkage: Optional[np.ndarray] = None
        col_linkage: Optional[np.ndarray] = None
        if finite.shape[0] > 1:
            row_linkage = linkage(pdist(finite, metric='euclidean'), method='ward')
            row_order = leaves_list(row_linkage).tolist()
        if finite.shape[1] > 1:
            col_linkage = linkage(pdist(finite.T, metric='euclidean'), method='ward')
            col_order = leaves_list(col_linkage).tolist()
        return numeric.iloc[row_order, col_order], True, row_linkage, col_linkage
    except (ImportError, ValueError, TypeError, FloatingPointError):
        return numeric, False, None, None


def cluster_heatmap_matrix(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Apply the same Ward/Euclidean ordering used by VisPath.

    VisPath clusters a finite copy of the matrix, replacing missing values
    with zero before calculating row and column Euclidean distances.  The
    report follows that ordering while retaining missing cells as blanks
    in the displayed Plotly heatmap.
    """
    ordered, clustered, _, _ = cluster_heatmap_matrix_with_linkage(matrix)
    return ordered, clustered


def _dendrogram_line_points(linkage_matrix: np.ndarray) -> Tuple[Optional[list], Optional[list]]:
    """Convert a scipy linkage matrix into dendrogram line-segment coordinates.

    Returns ``(positions, distances)`` lists with ``None`` separators between
    merge segments. Positions are leaf-slot indices (0..n-1) in the linkage's
    own leaf order — scipy spaces its icoord leaves at 5, 15, 25, ..., so the
    slots are ``(value - 5) / 10`` — and distances are merge heights.
    """
    try:
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    except ImportError:
        return None, None
    result = scipy_dendrogram(linkage_matrix, no_plot=True)
    positions: list = []
    distances: list = []
    for seg_positions, seg_distances in zip(result['icoord'], result['dcoord']):
        positions.extend((value - 5.0) / 10.0 for value in seg_positions)
        distances.extend(seg_distances)
        positions.append(None)
        distances.append(None)
    return positions, distances


def _thinned_ticks(labels: List[str], max_ticks: int = 45) -> Tuple[List[int], List[str]]:
    """(tickvals, ticktext) for the dendrogram's shared leaf-slot axis.

    Explicit per-leaf ticks keep the tree leaves and the axis labels on one
    grid, but drawing every label turns a large matrix's axes into an
    overlapping mass (a 200-row band is ~3px per label). Beyond
    ``max_ticks`` an even stride subsamples, keeping the first and last
    leaf so both ends stay anchored.
    """
    n = len(labels)
    if n <= max_ticks:
        return list(range(n)), list(labels)
    stride = -(-n // max_ticks)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx, [labels[i] for i in idx]


def plotly_heatmap_fragment(
    matrix: pd.DataFrame,
    title: str,
    style: MetricStyle,
    x_title: str,
    y_title: str,
    include_plotlyjs: bool = False,
    square_cells: bool = False,
    show_dendrogram: bool = True,
) -> Tuple[Optional[str], bool]:
    """Render one clustered Plotly heatmap fragment without cell labels.

    ``square_cells`` locks each matrix cell to a 1:1 aspect ratio:
    Plotly's ``scaleanchor`` is the exactness guarantee (it survives narrow
    metric cards, the estimated colorbar footprint, and responsive
    re-renders), while for small matrices (max dimension <= 30) an explicit
    figure width is additionally computed as the natural size target so a
    wide single-metric card does not leave the constrained domain centered
    far from the section header.

    When the matrix clustered successfully and ``show_dendrogram`` is set,
    mini-dendrograms for the applied Ward orderings are drawn beside the
    heatmap (column tree above, row tree right) within the same figure.
    """
    if matrix is None or matrix.empty:
        return None, False

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None, False

    ordered, clustered, row_linkage, col_linkage = cluster_heatmap_matrix_with_linkage(matrix)
    z = [
        [None if pd.isna(value) else float(value) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    x_labels = [str(value) for value in ordered.columns]
    y_labels = [str(value) for value in ordered.index]

    # Dendrogram overlay: line coordinates in leaf-slot / distance space.
    top_points = (None, None)
    right_points = (None, None)
    top_max = right_max = 0.0
    use_dendrogram = clustered and show_dendrogram
    if use_dendrogram:
        if col_linkage is not None:
            top_points = _dendrogram_line_points(col_linkage)
            top_max = float(np.max(col_linkage[:, 2])) if top_points[0] is not None else 0.0
        if row_linkage is not None:
            right_points = _dendrogram_line_points(row_linkage)
            right_max = float(np.max(row_linkage[:, 2])) if right_points[0] is not None else 0.0
        use_dendrogram = top_points[0] is not None or right_points[0] is not None

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
    if use_dendrogram:
        # Numeric axes so the dendrogram scatters share the heatmap's tick
        # space; labels move to ticktext and the hover text carries both ends.
        heatmap_kwargs['x'] = list(range(len(x_labels)))
        heatmap_kwargs['y'] = list(range(len(y_labels)))
        heatmap_kwargs['text'] = [
            [f'{row_label} · {col_label}' for col_label in x_labels]
            for row_label in y_labels
        ]
        heatmap_kwargs['hovertemplate'] = (
            '<b>%{text}</b><br>'
            f'{style.display_name}: %{{z:.3f}}'
            '<extra></extra>'
        )

    fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])
    max_label_length = max((len(label) for label in y_labels), default=12)
    left_margin = min(235, max(90, max_label_length * 5 + 22))
    matrix_dimension = max(len(y_labels), len(x_labels))
    row_height = 18 if len(y_labels) > 60 else 23
    # Square-cell strategy: the scaleanchor below is the exactness guarantee —
    # it locks cells 1:1 whatever the container does (a two-per-row metric
    # card is roughly half the width the explicit math assumes, the colorbar
    # footprint is only estimated, and a responsive re-render on window
    # resize would otherwise stretch the cells). The explicit width
    # computation stays as the NATURAL size target for small matrices: it
    # caps the plotly div so a wide single-metric card does not leave the
    # constrained domain centered far from the section header.
    n_cols = max(len(x_labels), 1)
    n_rows = max(len(y_labels), 1)
    colorbar_allow = 100
    square_explicit_width = False
    fig_width_px: Optional[int] = None
    dendro_band_px = 44
    top_band_px = dendro_band_px if top_points[0] is not None else 0
    right_band_px = dendro_band_px if right_points[0] is not None else 0
    if square_cells and matrix_dimension <= 30:
        row_height = 18 if matrix_dimension > 60 else 27
        # Compute cell size from BOTH dimensions so the height cap is
        # respected even for non-square matrices (e.g. 19 rows x 13 cols).
        width_limit = (1100 - left_margin - 12 - colorbar_allow - right_band_px) // n_cols
        height_limit = (880 - 168) // n_rows
        cell_px = max(30, min(80, width_limit, height_limit))
        fig_width_px = min(1100, left_margin + 12 + colorbar_allow + right_band_px + n_cols * cell_px)
        height = max(390, min(880, 168 + n_rows * cell_px))
        square_explicit_width = True
    elif square_cells:
        row_height = 18 if matrix_dimension > 60 else 27
        height = max(390, 185 + len(y_labels) * row_height)
        height = min(880, height)
    else:
        height = max(335, min(880, 185 + len(y_labels) * row_height))
    if top_band_px:
        # Keep the heatmap band at its computed size; the tree lives above it.
        height = min(924, height + top_band_px)
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
    if use_dendrogram:
        # Numeric axes so the dendrogram scatters share the heatmap's tick
        # space; labels move to ticktext. Large matrices thin the ticks —
        # an explicit per-leaf list would draw ~3px-pitch labels as an
        # overlapping mass.
        xaxis['tickvals'], xaxis['ticktext'] = _thinned_ticks(x_labels)
        yaxis['tickvals'], yaxis['ticktext'] = _thinned_ticks(y_labels)
    if square_cells:
        # Equal axis scaling makes each matrix cell a true square at any
        # container width, including responsive re-renders and the narrow
        # cards of a two-per-row metric grid. The explicit height above
        # keeps the plot area tall enough that the constrained domain stays
        # adjacent to the row labels.
        yaxis.update({
            'scaleanchor': 'x',
            'scaleratio': 1,
            'constrain': 'domain',
        })
    # Carve the tree bands out of the heatmap's plot area so leaf edges stay
    # flush with the heatmap: column tree above, row tree right. The domains
    # must be part of the update_layout call below — mutating these dicts
    # afterwards would not reach the figure.
    top_frac = (top_band_px / height) if top_band_px else 0.0
    # The right band's reserved fraction is nominal at a 1100px figure:
    # only the small-matrix square path pins an explicit width, so wider
    # matrices render at the card's fluid width and the band draws thinner
    # than the 44px constant there. Placement is corrected client-side by
    # report_script's alignDendrogramBands, which snaps the band's inner
    # edge onto the true cell edge whatever fraction was reserved.
    right_frac = right_band_px / float(fig_width_px or 1100)
    if top_frac:
        yaxis['domain'] = [0.0, 1.0 - top_frac]
    if right_frac:
        xaxis['domain'] = [0.0, 1.0 - right_frac]
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
    if use_dendrogram:
        dendro_line = {'color': '#8a8f98', 'width': 1}
        if top_points[0] is not None:
            fig.add_trace(go.Scatter(
                x=top_points[0], y=top_points[1], mode='lines',
                line=dendro_line, hoverinfo='skip', showlegend=False,
                xaxis='x', yaxis='y2'))
            fig.update_layout(yaxis2={
                'domain': [1.0 - top_frac, 1.0],
                'range': [0.0, top_max * 1.05],
                'visible': False, 'fixedrange': True,
            })
        if right_points[0] is not None:
            # Right band: x = merge heights (the band's depth axis), y = leaf
            # slots on the shared heatmap y axis. _dendrogram_line_points
            # returns (positions, distances), so the axes are swapped here.
            fig.add_trace(go.Scatter(
                x=right_points[1], y=right_points[0], mode='lines',
                line=dendro_line, hoverinfo='skip', showlegend=False,
                xaxis='x2', yaxis='y'))
            fig.update_layout(xaxis2={
                'domain': [1.0 - right_frac, 1.0],
                'range': [0.0, right_max * 1.05],
                'visible': False, 'fixedrange': True,
            })

    fragment = fig.to_html(
        full_html=False,
        include_plotlyjs='inline' if include_plotlyjs else False,
        config={
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        },
        default_width='100%',
    )
    if square_explicit_width and fig_width_px:
        # Cap the plotly div's container width so cells render as true
        # squares; the outer CSS rule `.heatmap-stage .plotly-graph-div
        # { width: 100% !important; }` still applies inside the wrapper.
        fragment = (f'<div class="heatmap-square-fit" '
                    f'style="max-width:{fig_width_px}px;margin:0 auto">'
                    f'{fragment}</div>')
    return fragment, clustered


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
/* Overview tables in the cross-dataset morphology report carry one column
   block per directed pair; keep headers on one line and let the surrounding
   overflow-x:auto container handle the horizontal scroll. */
.overview-table { width: auto; }
.overview-table th, .overview-table td { white-space: nowrap; }
.overview-table th { position: sticky; top: 0; z-index: 1; }
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
    """Return the small tab/resize controller used by report.html.

    Also re-aligns the dendrogram bands of clustered heatmap cards onto the
    heatmap's true cell edges after each render/resize/relayout. The
    square-cell constraint is allowed to shrink the y DOMAIN (constrain:
    'domain') or pad the ranges, so the band edge is mapped from the live
    ``_fullLayout`` domain — the same math the VisPath page uses, where it
    was verified against plotly directly.
    """
    return """<script>
(function () {
  function resizePlots(panel) {
    if (!panel || !window.Plotly) return;
    panel.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
      // resize resolves its re-constraint asynchronously; aligning before
      // it settles would read the pre-resize ranges and stay stale until
      // the next event.
      var done = function () { alignDendrogramBands(plot); hookPlot(plot); };
      try {
        var settled = window.Plotly.Plots.resize(plot);
        if (settled && typeof settled.then === 'function') {
          settled.then(done, done);
        } else {
          done();
        }
      } catch (error) { done(); }
    });
  }

  // Dendrogram bands must hug the heatmap's cell edges. The square-cell
  // scaleanchor reconciles the aspect ratio by padding the autoranged
  // ranges OR by shrinking the constrained axis's domain, so the band
  // edge is computed from the LIVE _fullLayout domain: map the
  // band-adjacent cell edge's data coordinate (-0.5 for the reversed y
  // top row, nX-0.5 for the right column) through the live domain. The
  // tree traces share the heatmap axes, so leaf positions stay put; only
  // the band domains move, and the outer edge keeps its full depth.
  function alignDendrogramBands(plot) {
    if (!window.Plotly || !plot || !plot._fullLayout) return;
    try {
      var full = plot._fullLayout;
      if (!full.xaxis2 && !full.yaxis2) return;
      var heat = plot.data && plot.data[0];
      if (!heat || heat.type !== 'heatmap') return;
      var update = {};
      if (full.yaxis2 && full.yaxis && full.yaxis.range &&
          full.yaxis.range[0] !== full.yaxis.range[1]) {
        var nY = (heat.y || []).length;
        if (nY > 1) {
          var ya = full.yaxis;
          var fracY = ((-0.5) - ya.range[0]) / (ya.range[1] - ya.range[0]);
          // frac == 1 is the common constrained case: the range stays
          // exact and plotly shrinks the domain instead.
          if (isFinite(fracY) && fracY > 0 && fracY <= 1) {
            var yd = ya.domain;
            var edgeY = yd[0] + fracY * (yd[1] - yd[0]);
            var curY = full.yaxis2.domain;
            if (Math.abs(curY[0] - edgeY) > 0.001) {
              update['yaxis2.domain'] = [edgeY, curY[1]];
            }
          }
        }
      }
      if (full.xaxis2 && full.xaxis && full.xaxis.range &&
          full.xaxis.range[0] !== full.xaxis.range[1]) {
        var nX = (heat.x || []).length;
        if (nX > 1) {
          var xa = full.xaxis;
          var fracX = ((nX - 0.5) - xa.range[0]) / (xa.range[1] - xa.range[0]);
          if (isFinite(fracX) && fracX > 0 && fracX <= 1) {
            var xd = xa.domain;
            var edgeX = xd[0] + fracX * (xd[1] - xd[0]);
            var curX = full.xaxis2.domain;
            if (Math.abs(curX[0] - edgeX) > 0.001) {
              update['xaxis2.domain'] = [edgeX, curX[1]];
            }
          }
        }
      }
      // A relayout with unchanged values would re-fire the relayout hook;
      // remember the last update and skip identical ones.
      var key = JSON.stringify(update);
      if ((update['yaxis2.domain'] || update['xaxis2.domain']) &&
          plot.__lastDendroAlign !== key) {
        plot.__lastDendroAlign = key;
        window.Plotly.relayout(plot, update);
      }
    } catch (error) { /* no-op */ }
  }

  // Re-run the alignment whenever a zoom/pan (or our own snap) relayouts
  // the plot; registered once per plot element.
  function hookPlot(plot) {
    if (plot.__dendroAlignHooked) return;
    plot.__dendroAlignHooked = true;
    try {
      plot.on('plotly_relayout', function () {
        setTimeout(function () { alignDendrogramBands(plot); }, 0);
      });
    } catch (error) { /* no-op */ }
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
    document.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
      setTimeout(function () {
        alignDendrogramBands(plot);
        hookPlot(plot);
      }, 0);
    });
    document.querySelectorAll('.tab-panel.active').forEach(resizePlots);
  });

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      document.querySelectorAll('.js-plotly-plot').forEach(alignDendrogramBands);
    }, 80);
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
            style = style_for(styles, key)
            generate_interactive_heatmap(
                matrices_dict={key: matrix},
                filename=str(html_path),
                title=fallback_title(group, key, group_display(group)),
                showfig=show_figures,
                verbose=verbose,
                # The fallback must honor the metric's colormap and fixed
                # domain (e.g. the diverging [-1, 1] similarity scale) —
                # without it the page defaults to sequential Viridis with
                # an auto-ranged axis and the negative half loses meaning.
                color_scale=style.colorscale,
                zmin=style.zmin,
                zmax=style.zmax,
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
    square_cells: bool = True,
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
        square_cells: open the rendered heatmaps with square cells locked
            (default True — all similarity heatmaps render square).

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
                    square_cells=square_cells,
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
