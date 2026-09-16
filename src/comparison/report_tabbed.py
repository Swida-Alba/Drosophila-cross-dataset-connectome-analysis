"""Tabbed comparison report — wraps the legacy generator's output.

Plan: ``_plan/plan-report-tabbed-layout.md``.

This module does NOT reimplement any analysis presentation.  It takes the
complete legacy report HTML (produced by
``html_report_generator.generate_html_report``, which stays untouched as
the reference implementation) and reorganizes it into a two-level tabbed
layout:

* **Page tabs** — the legacy top-level section blocks are split out by
  boundary-anchored scanning (section wrapper divs + section-header divs;
  the legacy HTML is treated as opaque tag soup — genuinely unbalanced —
  so spans are div-rebalanced instead of balance-scanned) and classified
  into seven page panes: Overview, Combos, Type Mapping, Plots, Matrices
  & Networks, Cross-views, Notes.
* **Combos page** — a per-query-row dashboard (B7 ids as labels,
  verticals first then density-matched): KPIs, involved-types card (the
  per-combo subset of the mapping table — combos involve anywhere from
  11 to 601 types), and deep links that activate the corresponding
  per-query inner tabs inside the heavy Matrices & Networks pages.
* **Plots page** — the legacy summary section's mixed charts (one chart
  blended ``threshold={N}`` and ``aligned_density={level}`` rows) are
  excised from Overview and REPLACED on the Plots page by charts split
  per analysis (per-threshold vs per-density), built from the run's
  ``comparison_report_used_data/*_by_query.csv`` files with numeric
  ordered x-axes.

Layout-only: every number, table, and CSV byte-identical to the legacy
report's data.
"""

from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOKEN = re.compile(
    r'<script\b[^>]*>|</script\s*>|<style\b[^>]*>|</style\s*>'
    r'|<div\b|</div>', re.IGNORECASE)

try:
    from .html_report_generator import _query_report_slug
except ImportError:  # direct src/ execution
    from html_report_generator import _query_report_slug


def _balanced_div_end(html: str, start: int) -> int:
    """Index just past the ``</div>`` that closes the ``<div>`` opening at
    ``start``.  Script/style bodies are skipped — legacy inline JS embeds
    div-like strings (vis/plotly payloads) that would otherwise unbalance
    the count and make one section swallow the rest of the page."""
    depth = 0
    pos = start
    in_block: Optional[str] = None
    for match in _TOKEN.finditer(html, start):
        token = match.group(0).lower()
        if in_block == 'script':
            if token.startswith('</script'):
                in_block = None
            continue
        if in_block == 'style':
            if token.startswith('</style'):
                in_block = None
            continue
        if token.startswith('<script'):
            in_block = 'script'
            pos = match.end()
            continue
        if token.startswith('<style'):
            in_block = 'style'
            pos = match.end()
            continue
        if token.startswith('<div'):
            depth += 1
            pos = match.end()
        else:
            depth -= 1
            pos = match.end()
            if depth == 0:
                return pos
    return pos


SECTION_HEADER_TEXT = re.compile(
    r'<div class="section-header">([^<]*)</div>', re.IGNORECASE)


WRAPPER_START = re.compile(r'<div[^>]*\bclass="section"[^>]*>',
                           re.IGNORECASE)


def _div_deficit(fragment: str) -> int:
    """Unclosed ``<div>`` count in a fragment (script/style bodies
    skipped).  The legacy report is tag soup — sections rely on the
    browser's implicit closing — so panes must be re-balanced to stay
    independent."""
    depth = 0
    in_block: Optional[str] = None
    for match in _TOKEN.finditer(fragment):
        token = match.group(0).lower()
        if in_block == 'script':
            if token.startswith('</script'):
                in_block = None
            continue
        if in_block == 'style':
            if token.startswith('</style'):
                in_block = None
            continue
        if token.startswith('<script'):
            in_block = 'script'
            continue
        if token.startswith('<style'):
            in_block = 'style'
            continue
        if token.startswith('<div'):
            depth += 1
        else:
            depth -= 1
    return max(0, depth)


def split_sections(body: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Split the legacy body into (preamble, [(header_text, section_html)],
    tail).

    The legacy HTML is tag soup (sections carry unclosed divs that the
    browser closes implicitly), so this does NOT balance-div scan.
    Section boundaries are anchored at the section WRAPPER divs
    (``<div id="…" class="section">``) and section-header divs; each span
    is then div-rebalanced (closing ``</div>`` appended) so page panes
    stay independent."""
    wrapper_bounds = [m.start() for m in WRAPPER_START.finditer(body)]
    header_bounds = [m.start() for m in SECTION_HEADER_TEXT.finditer(body)]
    bounds = sorted(set(wrapper_bounds) | set(header_bounds))
    # Drop header bounds that merely announce the wrapper right before
    # them (the wrapper's own section-header).
    merged: List[int] = []
    header_set = set(header_bounds)
    for bound in bounds:
        if (bound in header_set and merged
                and bound - merged[-1] <= 400):
            continue
        merged.append(bound)
    if not merged:
        return body, [], ''

    preamble = body[:merged[0]]
    tail = ''
    sections: List[Tuple[str, str]] = []
    for index, bound in enumerate(merged):
        end = merged[index + 1] if index + 1 < len(merged) else len(body)
        span = body[bound:end]
        header_match = SECTION_HEADER_TEXT.search(span)
        header = (header_match.group(1).strip()
                  if header_match else f'(section {index + 1})')
        deficit = _div_deficit(span)
        if deficit:
            span = span + '</div>' * deficit
        sections.append((header, span))
    return preamble, sections, tail


def excise_cards(section_html: str, title_keywords: Tuple[str, ...]):
    """Remove ``<div class="card"><h3>…keyword…</h3>…</div>`` blocks whose
    title matches AND whose payload (card + immediately following
    ``<script>`` — the legacy summary cards emit their Plotly script
    AFTER the card's closing div) contains a Plotly call."""
    kept: List[str] = []
    removed: List[str] = []
    pos = 0
    while True:
        match = re.search(r'<div class="card">\s*<h3>([^<]*)</h3>',
                          section_html[pos:])
        if match is None:
            kept.append(section_html[pos:])
            break
        card_start = pos + match.start()
        card_end = _balanced_div_end(section_html, card_start)
        card = section_html[card_start:card_end]
        title = match.group(1)
        if any(keyword.lower() in title.lower()
               for keyword in title_keywords):
            # include an immediately following script block, if any
            script_match = re.match(r'\s*<script>.*?</script>',
                                    section_html[card_end:], re.DOTALL)
            if script_match:
                card_end += script_match.end()
                card = section_html[card_start:card_end]
        if any(keyword.lower() in title.lower()
               for keyword in title_keywords) and 'Plotly.newPlot' in card:
            removed.append(card)
        else:
            kept.append(card)
        pos = card_end
    return ''.join(kept), removed


# ----------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------

PAGE_RULES = [
    ('types', ('Type Mapping', 'Resolution topology')),
    ('combos', ('Applied Thresholds', 'Bottleneck Provenance')),
    ('plots', ('Density curves', 'Threshold Alignment')),
    ('matrices', ('Edge Presence', 'Path Presence', 'Network Visualizations',
                  'Conservation')),
    ('crossviews', ('Hemisphere', 'Overlap', 'Statistics', 'Similarity')),
    ('overview', ('Summary', 'Neuron Counts', 'Query Resolution')),
]
PAGE_TITLES = [
    ('overview', 'Overview'),
    ('combos', 'Combos (per query)'),
    ('types', 'Type Mapping'),
    ('plots', 'Plots'),
    ('matrices', 'Matrices & Networks'),
    ('crossviews', 'Cross-views'),
]
PAGE_ORDER = [page for page, _ in PAGE_TITLES]


def classify(header_text: str) -> str:
    for page, keywords in PAGE_RULES:
        if any(keyword.lower() in header_text.lower() for keyword in keywords):
            return page
    return 'overview'


# ----------------------------------------------------------------------
# per-combo dashboard data
# ----------------------------------------------------------------------

def _query_rows(analyzer) -> List[Dict[str, Any]]:
    try:
        queries = analyzer.get_threshold_queries()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for query in queries:
        qid = str(query.get('id') or query.get('query_id') or '')
        mode = str(query.get('row_mode') or '').strip().lower()
        if not mode:
            if qid.startswith('threshold='):
                mode = 'vertical'
            elif qid.startswith('aligned_density='):
                mode = 'horizontal'
            elif qid.startswith('aligned_v'):
                mode = 'vertical'
            elif qid.startswith('aligned_h'):
                mode = 'horizontal'
        rows.append({'id': qid, 'label': str(query.get('label') or qid),
                     'mode': mode or 'other',
                     'thresholds': query.get('thresholds') or {}})
    return rows


def _presence_stats(run_dir: Path, slug: str) -> Dict[str, Any]:
    path = run_dir / 'comparison_results' / \
        f'edge_presence_matrix_query_{slug}.csv'
    if not path.exists():
        return {}
    try:
        import pandas as pd
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    types = set()
    for column in ('source_type', 'target_type'):
        if column in frame.columns:
            types.update(str(v) for v in frame[column].dropna()
                         if str(v) not in ('nan', ''))
    conserved = 0
    if 'conservation_count' in frame.columns:
        conserved = int((frame['conservation_count'] >= 2).sum())
    return {'edges': int(len(frame)), 'types': sorted(types),
            'conserved': conserved}


def _mapping_type_index(run_dir: Path) -> Dict[str, Dict[str, str]]:
    """type name -> {'anchor_group': …, 'auto_only': …} from the run's
    auto_type_mapping.csv (any endpoint of a row carries the row's
    flags)."""
    path = run_dir / 'auto_type_mapping.csv'
    index: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return index
    try:
        import pandas as pd
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return index
    if 'anchor_group' not in frame.columns:
        return index
    dataset_columns = [c for c in frame.columns
                       if c not in ('mapping_origin', 'mapping_support',
                                    'anchor_group', 'auto_only',
                                    'appearance_rank')]
    for _, row in frame.iterrows():
        anchor = '' if pd.isna(row.get('anchor_group')) else \
            str(row.get('anchor_group')).strip()
        auto = '' if pd.isna(row.get('auto_only')) else \
            str(row.get('auto_only')).strip()
        if not anchor and not auto:
            continue
        for ds in dataset_columns:
            value = row.get(ds)
            name = '' if pd.isna(value) else str(value).strip()
            if name and name not in index:
                index[name] = {'anchor_group': anchor, 'auto_only': auto}
    return index


def _combos_dashboard(analyzer, run_dir: Path, esc) -> str:
    """The per-query-row dashboard: KPI + involved-types card per combo,
    verticals first then density-matched, with deep links into the
    heavy pages' inner per-query tabs."""
    rows = _query_rows(analyzer)
    if not rows:
        return ''
    type_index = _mapping_type_index(run_dir)
    parts = ['<div class="card">',
             '<h3>Per-combo dashboard</h3>',
             '<p style="color:var(--secondary-color);font-size:0.9em;">',
             'One card per query row (verticals first, then '
             'density-matched). The involved-types card lists exactly the '
             'types this combo touches, with the mapping-table flags; '
             'buttons jump to the combo\'s panes in the heavy pages.</p>']
    for group in ('vertical', 'horizontal', 'other'):
        group_rows = [r for r in rows if r['mode'] == group]
        if not group_rows:
            continue
        if group != 'other':
            title = ('Per-threshold analysis (vertical rows)' if group == 'vertical'
                     else 'Density-matched analysis (horizontal rows)')
            parts.append(f'<h4>{esc(title)}</h4>')
        for row in group_rows:
            qid = row['id']
            slug = _query_report_slug(qid)
            stats = _presence_stats(run_dir, slug)
            n_types = len(stats.get('types', []))
            parts.append(
                f'<div style="border:1px solid var(--border-color);'
                f'border-radius:6px;padding:6px 10px;margin:8px 0;">'
                f'<strong>{esc(qid)}</strong> '
                f'<span style="color:var(--secondary-color);">'
                f'[{esc(row["mode"])}]</span> — '
                f'{esc(row["label"])}<br>'
                f'edges: {stats.get("edges", "—")} · '
                f'conserved(≥2): {stats.get("conserved", "—")} · '
                f'involved types: {n_types}')
            if stats.get('types'):
                items = []
                for type_name in stats['types'][:60]:
                    flags = type_index.get(type_name) or {}
                    badge = ' ⚠️' if flags.get('auto_only') else ''
                    items.append(f'{esc(type_name)}{badge}')
                more = (f' … (+{len(stats["types"]) - 60})'
                        if len(stats['types']) > 60 else '')
                shown = ', '.join(items) + more
                parts.append(
                    '<details><summary style="cursor:pointer;'
                    'font-size:0.85em;">involved types</summary>'
                    f'<div style="font-size:0.85em;">{shown}</div></details>')
            # json.dumps(qid) is double-quoted; escape it for the
            # double-quoted onclick attribute or the attribute truncates
            # mid-literal and the handler dies with a JS SyntaxError.
            qid_js = _html.escape(json.dumps(qid), quote=True)
            parts.append(
                '<div style="margin-top:4px;">'
                f'<button class="tab-btn" onclick="TAB.goto(\'matrices\', '
                f'{qid_js}, this)">matrices</button> '
                f'<button class="tab-btn" onclick="TAB.goto(\'matrices\', '
                f'{qid_js}, this, true)">networks</button> '
                '</div></div>')
    parts.append('</div>')
    return ''.join(parts)


# ----------------------------------------------------------------------
# split plots (per-threshold vs per-density), from used-data CSVs
# ----------------------------------------------------------------------

_METRICS = [
    ('edge_count_data_by_query.csv', 'edge_count', 'Edge counts'),
    ('total_weight_data_by_query.csv', 'total_weight', 'Total weight'),
    ('avg_ratio_data_by_query.csv', 'ratio', 'Avg weight ratio'),
    ('avg_prob_data_by_query.csv', 'prob', 'Avg traversal probability'),
]


def _metric_value_column(frame) -> Optional[str]:
    for candidate in ('count', 'total_weight', 'weight', 'ratio', 'prob',
                      'value'):
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        if column not in ('query_id', 'query_label', 'dataset', 'threshold'):
            return column
    return None


def _split_plots_html(run_dir: Path, esc) -> str:
    """Per-threshold and per-density sub-tabs of the four summary metrics,
    built from ``comparison_report_used_data/*_by_query.csv``."""
    used_dir = run_dir / 'comparison_report_used_data'
    charts = {'vertical': [], 'horizontal': []}
    for filename, value_hint, title in _METRICS:
        path = used_dir / filename
        if not path.exists():
            continue
        try:
            import pandas as pd
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        value_column = _metric_value_column(frame)
        if value_column is None or 'query_id' not in frame.columns:
            continue
        for group, marker in (('vertical', 'threshold='),
                              ('horizontal', 'aligned_density=')):
            points: Dict[Tuple[str, str], list] = {}
            for _, row in frame.iterrows():
                qid = str(row.get('query_id') or '')
                if not qid.startswith(marker):
                    continue
                try:
                    x = float(qid.split('=', 1)[1])
                except ValueError:
                    continue
                label = str(row.get('query_label') or qid)
                ds = str(row.get('dataset') or '')
                value = row.get(value_column)
                if value is None or pd.isna(value):
                    continue
                points.setdefault((label, ds), []).append(
                    (x, float(value), label))
            for (label, ds), values in sorted(points.items()):
                values.sort(key=lambda item: item[0])
                charts[group].append({
                    'title': f'{title} — {ds}',
                    'label': label,
                    'x': [v[0] for v in values],
                    'y': [v[1] for v in values],
                })
    if not any(charts.values()):
        return ''
    payload = json.dumps(charts, default=str)
    return (
        '<div class="card"><h3>Per-threshold analysis (vertical rows)</h3>'
        '<div id="split-plots-vertical" class="chart-container" '
        'style="min-height:320px;"></div></div>'
        '<div class="card"><h3>Density-matched analysis (horizontal rows)</h3>'
        '<div id="split-plots-horizontal" class="chart-container" '
        'style="min-height:320px;"></div></div>'
        '<script>(function() {'
        f'const groups = {payload};'
        'function draw(containerId, rows, xTitle) {'
        '  const byLabel = {}; rows.forEach(r => {'
        '    (byLabel[r.label] = byLabel[r.label] || {x: [], y: [], name: r.label + " — " + r.title});'
        '    byLabel[r.label].x.push(r.x); byLabel[r.label].y.push(r.y); });'
        '  const traces = Object.values(byLabel).map(t => ({'
        '    x: t.x, y: t.y, mode: "lines+markers", name: t.name}));'
        '  Plotly.newPlot(containerId, traces, {'
        '    xaxis: {title: xTitle}, yaxis: {title: "value"},'
        '    legend: {orientation: "h"}, responsive: true}, {responsive: true});}'
        'draw("split-plots-vertical", groups.vertical || [], "Min Synapse Count (N)");'
        'draw("split-plots-horizontal", groups.horizontal || [], "Density level E(t)/N");'
        '})();</script>')


# ----------------------------------------------------------------------
# page assembly
# ----------------------------------------------------------------------

def _excise_legacy_summary_charts(overview_html: str):
    """Remove the legacy mixed summary charts from the Overview pane and
    return (overview_html, legacy_chart_cards)."""
    return excise_cards(
        overview_html,
        ('Edge Counts', 'Traversal', 'Total Weight', 'Weight Ratio',
         'Avg Ratio', 'Across All Queries', 'Probability'))


def _excise_quick_navigation(preamble: str) -> str:
    """Remove the legacy Quick Navigation block (the page-tab nav replaces
    it).  The block is ``<div class="toc">…</div>`` (balanced scan)."""
    match = re.search(r'<div[^>]*class="[^"]*\btoc\b[^"]*"[^>]*>', preamble,
                      re.IGNORECASE)
    if match is None:
        return preamble
    end = _balanced_div_end(preamble, match.start())
    return preamble[:match.start()] + preamble[end:]


def build_tabbed_report(analyzer, legacy_html: str,
                        run_dir: Optional[str] = None) -> str:
    """Transform the legacy report HTML into the tabbed layout.

    ``legacy_html`` is the complete output of
    ``html_report_generator.generate_html_report`` (treated as opaque);
    ``analyzer`` supplies the query rows for the Combos dashboard.
    """
    esc = _html.escape
    head_match = re.match(r'^(.*?<body[^>]*>)', legacy_html,
                          re.IGNORECASE | re.DOTALL)
    head = head_match.group(1) if head_match else ''
    body = legacy_html[head_match.end():] if head_match else legacy_html

    preamble, sections, tail = split_sections(body)

    run_path = Path(run_dir) if run_dir else Path(
        getattr(getattr(analyzer, 'parameters', None), 'full_output_path',
                '.') or '.')

    pages: Dict[str, List[str]] = {page: [] for page in PAGE_ORDER}
    legacy_summary_cards: List[str] = []
    for header, section_html in sections:
        # The legacy TOC can ride inside any section's span (tag soup) —
        # excise it everywhere; the page-tab nav replaces it.
        section_html = _excise_quick_navigation(section_html)
        page = classify(header)
        if page == 'overview':
            section_html, removed = _excise_legacy_summary_charts(section_html)
            legacy_summary_cards.extend(removed)
        pages[page].append(section_html)

    # Overview: preamble (header/banners, minus the legacy Quick
    # Navigation which the tab nav replaces) + overview sections.  The
    # query report's summary cards are NOT wrapped in a .section div —
    # they land in the preamble, so the mixed-chart excision must run
    # here too (real-run finding 2026-09-15).  The preamble is also
    # div-rebalanced: its legacy banners carry unclosed divs that would
    # otherwise swallow the following panes into the Overview pane.
    preamble = _excise_quick_navigation(preamble)
    preamble, preamble_cards = _excise_legacy_summary_charts(preamble)
    legacy_summary_cards.extend(preamble_cards)
    deficit = _div_deficit(preamble)
    if deficit:
        preamble = preamble + '</div>' * deficit
    pages['overview'].insert(0, preamble)

    # Combos dashboard.
    combos_html = _combos_dashboard(analyzer, run_path, esc)
    if combos_html:
        pages['combos'].insert(0, combos_html)
    pages['combos'].insert(0, (
        '<div class="card"><h3>Applied Thresholds &amp; Bottleneck '
        'Provenance</h3><p style="color:var(--secondary-color);">Per-query '
        'provenance (applied thresholds, tau, budgets) — the per-query '
        'authoritative view. The matrices/networks pages carry the same '
        'query rows as their inner tabs.</p></div>'))

    # Plots: legacy density/alignment sections + the new split charts +
    # the excised legacy mixed charts (collapsed, for comparison).
    split_html = _split_plots_html(run_path, esc)
    if split_html:
        pages['plots'].insert(0, split_html)
    if legacy_summary_cards:
        pages['plots'].append(
            '<details class="card"><summary>Legacy combined charts '
            '(per-threshold and per-density mixed)</summary>'
            + ''.join(legacy_summary_cards) + '</details>')

    # (There is no Notes pane: every section span extends to the end of
    # the body, so an unmatched tail after the last section cannot exist
    # — the previous always-empty pane only ever rendered its fallback.)

    nav = ''.join(
        f'<button class="page-tab-btn" data-page="{page}" role="tab" '
        f'onclick="TAB.show(\'{page}\')">{esc(title)}</button>'
        for page, title in PAGE_TITLES)
    templates = []
    for page, title in PAGE_TITLES:
        content = ''.join(pages.get(page) or [])
        if not content.strip():
            content = ('<p class="card" style="color:var(--secondary-color);">'
                       'Nothing in this tab for this run.</p>')
        templates.append(
            f'<template id="tpl-{page}">{content}</template>')

    router = """
<style>
.page-tab-bar { display: flex; flex-wrap: wrap; gap: 4px; position: sticky;
  top: 0; background: var(--bg-color, #fff); z-index: 50; padding: 8px 4px;
  border-bottom: 2px solid var(--border-color, #ddd); }
.page-tab-btn { border: 1px solid var(--border-color, #ccc); background: #fff;
  padding: 6px 14px; border-radius: 16px; cursor: pointer; font-size: 0.92em; }
.page-tab-btn.active { background: #1d4ed8; color: #fff;
  border-color: #1d4ed8; font-weight: 600; }
</style>
<div class="page-tab-bar" role="tablist">__NAV__</div>
<div id="page-content"></div>
<template id="tpl-empty"><p class="card">Nothing in this tab.</p></template>
<script>
(function() {
  var TAB = {
    show: function(page, keepHash) {
      var tpl = document.getElementById('tpl-' + page);
      var container = document.getElementById('page-content');
      if (!tpl || !container) { return; }
      container.innerHTML = '';
      container.appendChild(tpl.content.cloneNode(true));
      document.querySelectorAll('.page-tab-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.page === page); });
      if (!keepHash) {
        try { history.replaceState(null, '', '#tab=' + page); } catch (e) {}
      }
      document.dispatchEvent(new CustomEvent('page:shown',
          {detail: {page: page}}));
    },
    goto: function(page, queryId, button, wantNetworks) {
      this.show(page, true);
      // The networks deep link targets the per-query tab inside the
      // network sections; plain matrices targets the first matching tab
      // anywhere on the page (heatmap sections come first).
      var root = document;
      if (wantNetworks) {
        root = document.querySelector('#page-content .networks-section')
            || document.getElementById('page-content') || document;
      }
      var target = null;
      root.querySelectorAll('button').forEach(function(b) {
        if (!target && b.textContent.trim() === queryId) { target = b; }
      });
      if (target) { try { target.click(); } catch (e) {} }
    }
  };
  window.TAB = TAB;
  function initial() {
    var hash = (location.hash || '').replace('#tab=', '');
    TAB.show(hash ? decodeURIComponent(hash) : 'overview', true);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initial);
  } else {
    initial();
  }
})();
</script>
"""
    router = router.replace('__NAV__', nav)

    return head + router + ''.join(templates) + tail
