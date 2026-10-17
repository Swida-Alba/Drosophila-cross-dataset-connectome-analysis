"""The adjusted-ratio metric must name its own conditioning tier.

DROCAT audit F7.3: the metric dropdown read "Ratio (≥ N)" with a literal N
that was never filled in anywhere, so a reader could not tell which
threshold the denominator conditioned at — and the name said "Ratio", not
"Adj. Ratio", while every DROCAT table calls it the adjusted ratio.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'vispath-subproject' / 'src'))

from vispath_pkg.vispath import VisualizePath  # noqa: E402

VISPATH_SRC = (ROOT / 'vispath-subproject' / 'src' / 'vispath_pkg'
               / 'vispath.py')


def _df():
    return pd.DataFrame({
        'path_block': ['S>A>T', 'S>B>T'],
        'weights': [[10, 20], [5, 8]],
        'connection_ratios': [[0.01, 0.02], [0.003, 0.004]],
        'connection_ratios_adj': [[0.02, 0.04], [0.005, 0.007]],
    })


def _built(tmp_path, threshold):
    """The network page a run writes for this tier, plus the built object.

    The dead `out = tmp_path / 'net.html'` / `G = …` pair this replaced named a
    file the test never read and kept a build result nothing checked."""
    vp = VisualizePath(path_file=_df(), output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout='dagre',
                       adj_ratio_threshold=threshold)
    vp.build_network()
    vp.create_network()
    page = tmp_path / f'{vp.base_filename}_network.html'
    return vp, (page.read_text(encoding='utf-8') if page.exists() else None)


@pytest.mark.parametrize('raw, want_html, want_plain', [
    (5, 'Adj. Ratio (&ge; 5)', 'Adj. Ratio (≥ 5)'),
    (1, 'Adj. Ratio (&ge; 1)', 'Adj. Ratio (≥ 1)'),   # tier 1 is named too
    ('3', 'Adj. Ratio (&ge; 3)', 'Adj. Ratio (≥ 3)'),
    (3.0, 'Adj. Ratio (&ge; 3)', 'Adj. Ratio (≥ 3)'),
    (None, 'Adj. Ratio (threshold-conditioned)',
     'Adj. Ratio (threshold-conditioned)'),
    ('garbage', 'Adj. Ratio (threshold-conditioned)',
     'Adj. Ratio (threshold-conditioned)'),
])
def test_label_names_the_tier_it_conditions_at(raw, want_html, want_plain,
                                               tmp_path):
    vp = VisualizePath(path_file=_df(), output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       adj_ratio_threshold=raw)
    assert vp.adj_ratio_label == want_html
    assert vp.adj_ratio_label_plain == want_plain


def test_every_generated_page_carries_the_number(tmp_path):
    """Measured on the emitted HTML.

    The previous law pinned a call-site string
    (`metric_label_adj=self.adj_ratio_label_plain in src`), which broke on any
    refactor of that line while never showing that a page can omit the label."""
    _vp, html = _built(tmp_path, 5)
    assert html is not None, 'no network page was written'
    assert ('Adj. Ratio (&ge; 5)' in html
            or 'Adj. Ratio (≥ 5)' in html), (
        'the page does not name the tier its adjusted ratio was conditioned '
        'at')
    # the bare placeholder is gone from the emitted page, in every form
    for ghost in ('&ge; N)', '(>= N)', 'Ratio (&ge; N)'):
        assert ghost not in html, ghost


def test_the_batch_heatmap_entry_cannot_silently_drop_the_label():
    """`create_heatmap` (the entry DROCAT uses) hands the interactive builder
    the tiered label; `create_heatmaps` (the batch twin) does not — measured
    2026-10-16, it passes no `metric_label_adj` at all.

    That is a latent gap, not a live one: nothing in the repo calls the batch
    entry. So the law is the disjunction — either it passes the label, or it
    stays uncalled. The day a caller appears, this test says why its heatmap
    reads "Adj. Ratio" with no tier."""
    import subprocess

    src = VISPATH_SRC.read_text(encoding='utf-8')
    batch = src[src.index('def create_heatmaps('):
                src.index('def create_heatmap(')]
    passes_label = 'metric_label_adj=' in batch
    # the package's own docstring example is not a caller
    callers = subprocess.run(
        ['git', 'grep', '-l', r'\.create_heatmaps(', '--', 'src', 'ui',
         'scripts', 'tests'],
        cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    assert passes_label or not callers, (
        'create_heatmaps is now called and still drops the tier label: '
        f'{callers}')


def test_no_placeholder_survives_in_the_generator_source():
    """The law is source-wide: a label with an unfilled N cannot come back
    through a code path the page tests above do not reach (the heatmap's
    metric maps, the ribbon select, the sankey hover)."""
    src = VISPATH_SRC.read_text(encoding='utf-8')
    for ghost in ('&ge; N)', '(>= N)', '\\u2265 N)'):
        assert ghost not in src, ghost
    # the option's label is a parameter the caller supplies — never a string
    # one generator invents for itself
    assert 'metric_label_adj=None' in src


def _page_edges_of(text):
    """(source, target) -> edge data, read from the emitted page's own JSON."""
    import json
    i = text.find('[{"data": {"source":')
    assert i >= 0, text[:200]
    depth, j = 0, i
    while j < len(text):
        if text[j] == '[':
            depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0:
                break
        j += 1
    out = {}
    for row in json.loads(text[i:j + 1]):
        data = row.get('data', row)
        out[(str(data.get('source')), str(data.get('target')))] = data
    return out


def _trim_df():
    """A paths frame in the shape the ENGINE hands VisualizePath (the 'path'
    column plus stringified per-hop lists), which is the only shape whose
    build actually produces a graph — the module's other fixture feeds
    `path_block` and builds zero edges."""
    return pd.DataFrame([
        {'path': 'S->A->T', 'weights': '[10, 20]', 'ratios': '[0.01, 0.02]',
         'min_weight': 10, 'min_ratio': 0.01, 'length': 2, 'nt_types': '[]',
         'coverage': '[1/1, 1/1, 1/1]',
         'connection_ratios_adj': [0.02, 0.04]},
        {'path': 'S->B->T', 'weights': '[5, 8]', 'ratios': '[0.003, 0.004]',
         'min_weight': 5, 'min_ratio': 0.003, 'length': 2, 'nt_types': '[]',
         'coverage': '[1/1, 1/1, 1/1]',
         'connection_ratios_adj': [0.005, 0.007]},
    ])


def test_the_edge_limit_trim_keeps_the_pair_level_metrics(tmp_path):
    """A page trimmed to the Visualization Edge Limit must still carry Refilled
    and Adj. Ratio (DROCAT audit 2026-10-16, found on real data).

    The trim re-derives its edge records from the PATH ROWS — weight, ratio,
    probability — while the pair-level metrics live only on the graph the
    caller built. So every run whose edge count exceeded the limit drew a page
    where both toggles read 'N/A': measured on the cached MCNS run
    R3_complete_and (396 emitted pairs, limit 200), the page carried
    `refilled` for 0/200 edges and `ratio_adj` for 0/200, while the run's own
    data files and the un-trimmed graph had both. A run small enough to skip
    the trim showed them, which is why the unit tests — all of them below the
    limit — never saw it.

    Wrong hypothesis: a fixture that never trims would pass on the broken
    code, so the law also pins that the drawn page is smaller than the full
    graph."""
    refilled = {('S', 'A'): 11.0, ('A', 'T'): 22.0,
                ('S', 'B'): 5.0, ('B', 'T'): 8.0}
    adj = {('S', 'A'): 0.02, ('A', 'T'): 0.04,
           ('S', 'B'): 0.005, ('B', 'T'): 0.007}
    vp = VisualizePath(path_file=_trim_df(), output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout='dagre', edgeN_limit=2,
                       adj_ratio_threshold=5,
                       pair_extra_metrics={'refilled': refilled})
    vp.build_network()
    full = {(u, v): d for u, v, d in vp.G_network.edges(data=True)}
    assert len(full) == 4, (
        f'the fixture built {len(full)} edges, not the 4 the trim needs')
    vp.create_network()
    page = tmp_path / f'{vp.base_filename}_network.html'
    assert page.exists(), 'no page was written'
    drawn = _page_edges_of(page.read_text(encoding='utf-8'))
    assert len(drawn) < len(full), (
        f'the fixture did not trim ({len(drawn)} of {len(full)} edges) — '
        f'the law would pass on the broken code')
    for key, data in drawn.items():
        assert data.get('refilled') is not None, (
            f'{key}: the trimmed page lost Refilled (N/A on a run that has '
            f'the value {refilled.get(key)})')
        assert abs(data['refilled'] - refilled[key]) < 1e-9, (
            key, data['refilled'], refilled[key])
        assert data.get('ratio_adj') is not None, (
            f'{key}: the trimmed page lost the adjusted ratio')
        assert abs(data['ratio_adj'] - adj[key]) < 1e-12, (
            key, data['ratio_adj'], adj[key])


# ---------------------------------------------------------------------------
# metric_note: the one-line caption under the Connection-Metric select
# ---------------------------------------------------------------------------

_NOTE = 'Refilled and the adjusted ratio are measured per TYPE pair for this run'
_POISON = 'Refilled <b>&</b> ratio'


def _pages(tmp_path, **kwargs):
    """Every page that shows the Connection-Metric select, as emitted HTML."""
    vp = VisualizePath(path_file=_trim_df(), output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout='dagre', **kwargs)
    vp.build_network()
    net = Path(str(vp.create_network())).read_text(encoding='utf-8')
    sankey = Path(str(vp.create_sankey())).read_text(encoding='utf-8')
    return vp, net, sankey


def test_a_metric_note_is_rendered_under_the_select_on_every_page(tmp_path):
    """A disabled option is the page's only WORD about a missing metric unless
    the caller can add a sentence (DROCAT review 2026-10-17: a bodyId-level
    page read 'Refilled (N/A)' with nothing to say that this is by design)."""
    _vp, net, sankey = _pages(tmp_path, metric_note=_NOTE)
    for name, html in (('network', net), ('sankey', sankey)):
        assert html.count(_NOTE) == 1, (name, html.count(_NOTE))
        # …and it sits under the metric select, not somewhere on the page: the
        # caption is only an explanation if it is read with the control.
        i = html.index(_NOTE)
        j = html.rindex('</select>', 0, i)        # the select it follows
        assert i - j < 400, (name, i - j)
        open_i = html.rindex('<select', 0, j)     # …and which select that is
        assert 'id="metricSelect"' in html[open_i:j], (
            f'{name}: the caption follows a select that is not the '
            'Connection Metric one')


def test_a_page_with_no_note_says_nothing(tmp_path):
    """The default is silence, not a placeholder: every page the type-level
    run writes would otherwise carry an explanation of a gap it does not
    have. Wrong hypothesis: a template that always emitted the caption div
    would pass the test above."""
    vp, net, sankey = _pages(tmp_path)
    assert vp.metric_note == '' and vp.metric_note_html == ''
    for name, html in (('network', net), ('sankey', sankey)):
        assert _NOTE not in html, name
        assert 'opacity:0.72;margin-top:3px' not in html, (
            f'{name}: an empty caption container was still written')


def test_the_note_is_escaped_like_the_rest_of_the_page(tmp_path):
    """The caption is data the caller supplies, not trusted template text."""
    _vp, net, _sankey = _pages(tmp_path, metric_note=_POISON)
    assert 'Refilled &lt;b&gt;&amp;&lt;/b&gt; ratio' in net
    assert '<b>&</b>' not in net

