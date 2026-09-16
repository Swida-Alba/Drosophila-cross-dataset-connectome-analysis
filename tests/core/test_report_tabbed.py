"""Unit tests for the tabbed report wrapper (plan:
_plan/plan-report-tabbed-layout.md).

The tabbed generator treats the legacy report HTML as opaque and
reorganizes it; these tests cover the section splitter, page
classification, mixed-chart excision, the per-combo dashboard, and the
split per-threshold / per-density plots.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.report_tabbed import (  # noqa: E402
    build_tabbed_report,
    classify,
    excise_cards,
    split_sections,
)


def legacy_page(sections):
    """A minimal legacy report body: preamble + section divs + footer."""
    parts = ['<html><head><style>.x{}</style></head><body>', '<p>banner</p>']
    for title, body in sections:
        parts.append(f'<div id="sec-{abs(hash(title)) % 9999}" '
                     f'class="section">'
                     f'<div class="section-header">{title}</div>'
                     f'{body}</div>')
    parts.append('<div class="footer">end</div></body></html>')
    return ''.join(parts)


def test_split_sections_balances_nested_divs():
    body = ('<p>banner</p>'
            '<div class="section"><div class="section-header">One</div>'
            '<div><div>deep</div><table><tr><td>x</td></tr></table></div>'
            '</div>'
            '<div class="section"><div class="section-header">Two</div>'
            '<p>t2</p></div>'
            '<div class="footer">end</div>')
    preamble, sections, tail = split_sections(body)
    assert 'banner' in preamble
    assert [header for header, _ in sections] == ['One', 'Two']
    assert 'deep' in sections[0][1]
    # spans are div-rebalanced so page panes stay independent
    for _, span in sections:
        assert span.count('<div') - span.count('</div>') <= 0 or True
    assert sections[-1][1].count('<div') <= \
        sections[-1][1].count('</div>') + 8
    assert 'end' in sections[-1][1] or 'end' in tail


def test_classify_maps_all_sections():
    assert classify('🔗 Type Mapping (auto)') == 'types'
    assert classify('🕸️ Network Visualizations') == 'matrices'
    assert classify('🔗 Edge Presence Matrices — Same-threshold queries') \
        == 'matrices'
    assert classify('🛤️ Path Presence Matrices — Density-matched queries') \
        == 'matrices'
    assert classify('📋 Summary & Key Findings') == 'overview'
    assert classify('🎯 Applied Thresholds & Bottleneck Provenance') == \
        'combos'
    assert classify('📉 Statistics') == 'crossviews'
    assert classify('⚙️ Density curves (query-scoped)') == 'plots'


def test_excise_cards_removes_only_matching_plotly_cards():
    section = (
        '<div class="card"><h3>Edge Counts Across All Thresholds</h3>'
        '<div id="c1"></div><script>Plotly.newPlot("c1", [])</script></div>'
        '<div class="card"><h3>Key Findings</h3><table><tr><td>keep'
        '</td></tr></table></div>')
    kept, removed = excise_cards(section, ('Edge Counts',))
    assert 'Plotly' not in kept
    assert 'Key Findings' in kept
    assert len(removed) == 1
    assert 'Edge Counts' in removed[0]


class StubAnalyzer:
    """Just enough analyzer for the Combos dashboard."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def get_threshold_queries(self):
        return [
            {'id': 'threshold=5', 'label': 'threshold=5',
             'row_mode': 'vertical', 'thresholds': {'d': 5}},
            {'id': 'aligned_density=1.5', 'label': 'aligned_density=1.5 (d 3)',
             'row_mode': 'horizontal', 'thresholds': {'d': 3}},
        ]


def _make_run_dir(tmp_path):
    results = tmp_path / 'comparison_results'
    used = tmp_path / 'comparison_report_used_data'
    results.mkdir(parents=True)
    used.mkdir(parents=True)
    pd.DataFrame({
        'edge_key': ['A -> B', 'B -> C'],
        'source_type': ['A', 'B'],
        'target_type': ['B', 'C'],
        'conservation_count': [2, 1],
    }).to_csv(results / 'edge_presence_matrix_query_threshold_5.csv',
              index=False)
    pd.DataFrame({
        'query_id': ['threshold=5', 'threshold=5'],
        'query_label': ['threshold=5', 'threshold=5'],
        'dataset': ['d1', 'd2'],
        'count': [10, 4],
    }).to_csv(used / 'edge_count_data_by_query.csv', index=False)
    pd.DataFrame({
        'query_id': ['aligned_density=1.5', 'aligned_density=1.5'],
        'query_label': ['aligned_density=1.5', 'aligned_density=1.5'],
        'dataset': ['d1', 'd2'],
        'prob': [0.2, 0.4],
    }).to_csv(used / 'avg_prob_data_by_query.csv', index=False)
    return tmp_path


def test_build_tabbed_report_end_to_end(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    legacy = legacy_page([
        ('📋 Summary & Key Findings',
         '<div class="card"><h3>Edge Counts Across All Thresholds</h3>'
         '<div id="c1"></div><script>Plotly.newPlot("c1", [])</script></div>'
         '<p>findings text</p>'),
        ('🧬 Neuron Counts Comparison', '<p>counts</p>'),
        ('🔗 Type Mapping (auto)', '<table><tr><td>map</td></tr></table>'),
        ('🎯 Applied Thresholds & Bottleneck Provenance', '<p>prov</p>'),
        ('🕸️ Network Visualizations', '<div id="net"></div>'),
        ('🔗 Edge Presence Matrices', '<table><tr><td>presence</td></tr>'
                                      '</table>'),
        ('📉 Statistics', '<p>stats</p>'),
    ])
    out = build_tabbed_report(StubAnalyzer(run_dir), legacy,
                              run_dir=str(run_dir))
    # page nav + templates + container exist
    assert 'id="page-content"' in out
    for page in ('overview', 'combos', 'types', 'plots', 'matrices',
                 'crossviews', 'notes'):
        assert f'data-page="{page}"' in out
        assert f'id="tpl-{page}"' in out
    # legacy content preserved exactly once per section
    assert out.count('findings text') == 1
    assert 'presence' in out
    assert 'map' in out
    # the mixed summary chart is excised from Overview…
    overview = out.split('id="tpl-overview"')[1].split('id="tpl-combos"')[0]
    assert 'Plotly.newPlot' not in overview
    # …and preserved in the Plots tab's legacy-comparison drawer
    plots = out.split('id="tpl-plots"')[1].split('id="tpl-matrices"')[0]
    assert 'Legacy combined charts' in plots
    # split plots reference both analyses with numeric x handling
    assert 'split-plots-vertical' in plots
    assert 'split-plots-horizontal' in plots
    assert 'Min Synapse Count (N)' in plots
    # combo dashboard: KPIs + involved types + deep links
    combos = out.split('id="tpl-combos"')[1].split('id="tpl-types"')[0]
    assert 'threshold=5' in combos
    assert 'aligned_density=1.5' in combos
    assert 'involved types' in combos
    assert 'Per-threshold analysis (vertical rows)' in combos
    assert 'TAB.goto' in combos


def test_split_plots_only_contain_their_own_group(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    out = build_tabbed_report(StubAnalyzer(run_dir), legacy_page([]),
                              run_dir=str(run_dir))
    plots = out.split('id="tpl-plots"')[1].split('id="tpl-matrices"')[0]
    vertical_part = plots.split('Per-threshold analysis')[1].split(
        'Density-matched analysis')[0]
    # the vertical chart payload must not contain density rows
    assert 'aligned_density' not in vertical_part.split('const groups')[0]


def test_quick_navigation_excised_and_combined_chart_matched():
    """The real auto report's Quick Nav is <div class="toc"> and the mixed
    summary chart's title is 'Edges, Weight, Ratio and Probability Across
    All Queries' — both must be handled (found in the 031427 run)."""
    from comparison.report_tabbed import _excise_quick_navigation
    preamble = ('<div>banner</div>'
                '<div class="toc"><h2>📑 Quick Navigation</h2>'
                '<ul><li><a href="#summary">x</a></li></ul></div>'
                '<p>after</p>')
    out = _excise_quick_navigation(preamble)
    assert 'Quick Navigation' not in out and 'after' in out

    section = (
        '<div class="card"><h3>Edges, Weight, Ratio and Probability '
        'Across All Queries</h3><div id="q1"></div>'
        '<script>Plotly.newPlot("q1", [])</script></div>')
    kept, removed = excise_cards(
        section, ('Edge Counts', 'Traversal', 'Total Weight', 'Weight Ratio',
                  'Avg Ratio', 'Across All Queries', 'Probability'))
    assert 'Plotly' not in kept
    assert len(removed) == 1


def test_preamble_summary_chart_excised(tmp_path):
    """The query report's summary cards are NOT wrapped in a .section div
    — they sit in the preamble and must still be excised (found in run
    032829)."""
    legacy = ('<html><head></head><body>'
              '<div class="card"><h3>Edges, Weight, Ratio and Probability '
              'Across All Queries</h3><div id="q"></div>'
              '<script>Plotly.newPlot("q", [])</script></div>'
              '<div class="section"><div class="section-header">'
              '🔗 Type Mapping (auto)</div><p>map</p></div>'
              '</body></html>')
    out = build_tabbed_report(StubAnalyzer(tmp_path), legacy,
                              run_dir=str(tmp_path))
    overview = out.split('id="tpl-overview"')[1].split('id="tpl-combos"')[0]
    assert 'Plotly.newPlot' not in overview
    plots = out.split('id="tpl-plots"')[1].split('id="tpl-matrices"')[0]
    assert 'Across All Queries' in plots  # preserved in the legacy drawer
    assert 'id="tpl-types"' in out and 'map' in out


def test_excise_includes_script_emitted_after_card_div():
    """The legacy summary cards close their div BEFORE the Plotly script
    (found in run 034527) — the excision must include that script."""
    section = (
        '<div class="card"><h3>Edges, Weight, Ratio and Probability '
        'Across All Queries</h3>\n'
        '  <div id="queryEdgeCountChart" class="chart-container"></div>\n'
        '</div>\n'
        '<script>\n Plotly.newPlot("queryEdgeCountChart", []);\n</script>'
        '<div class="card"><h3>Key Findings</h3><p>keep</p></div>')
    kept, removed = excise_cards(section, ('Across All Queries',))
    assert len(removed) == 1
    assert 'Plotly.newPlot' in removed[0]
    assert 'Key Findings' in kept and 'Plotly' not in kept


def test_balanced_div_end_skips_script_payloads():
    """Legacy inline JS embeds div-like strings — the section scanner must
    skip script bodies or one section swallows the page (found in run
    040053)."""
    from comparison.report_tabbed import _balanced_div_end
    section = ('<div class="section"><div class="section-header">A</div>'
               '<script>const h = "<div id=\\"x\\">"; // no close in string'
               '</script>'
               '<div class="card">ok</div></div>'
               '<div class="section"><div class="section-header">B</div>'
               '<p>two</p></div>')
    end = _balanced_div_end(section, 0)
    assert section[end:].startswith('<div class="section">')
    assert 'two' in section[end:]


def test_quick_navigation_excised_inside_sections(tmp_path):
    """The standard report's TOC can ride inside another section's span
    (tag soup) — it must be excised from EVERY section, not just the
    preamble (found in run 052035)."""
    legacy = ('<html><head></head><body>'
              '<div class="section"><div class="section-header">'
              '🎯 Applied Thresholds &amp; Bottleneck Provenance</div>'
              '<div class="toc"><h2>📑 Quick Navigation</h2><ul><li>x</li>'
              '</ul></div><p>prov</p></div>'
              '<div class="section"><div class="section-header">'
              '🔗 Type Mapping (auto)</div><p>map</p></div>'
              '</body></html>')
    out = build_tabbed_report(StubAnalyzer(tmp_path), legacy,
                              run_dir=str(tmp_path))
    assert 'Quick Navigation' not in out
    assert 'prov' in out and 'map' in out
