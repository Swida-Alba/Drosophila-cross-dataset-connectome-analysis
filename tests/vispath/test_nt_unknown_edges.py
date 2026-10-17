"""An edge with no transmitter must LOOK like it has no transmitter.

The network page colours edges by neurotransmitter with one Cytoscape rule per
vocabulary label (`edge[nt_type = "ACH"]` etc.). An unannotated pair matched no
rule and therefore kept the page's base link colour — on DROCAT runs that is a
light blue (rgba(100,150,240,.5), the Settings default), so "this pair carries
no transmitter information" was rendered in the same visual language as a real
transmitter colour (user report 2026-10-17, measured on
`find-network_FAFB_circadian_clock_w1r0_01_20261017_131605`: 130 of 148 edges
annotated, 18 drawn in the link colour).

Now the blank bucket gets its own rule in the palette's unknown gray, plus an
`Unknown Edges` entry in the group selector so the reader can recolour or hide
it — and it is emitted only when the page has real vocabulary to be gray
relative to, so a run carrying no transmitter column at all keeps its uniform
edge colour instead of a page that suddenly means "everything is unknown".
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'vispath-subproject' / 'src'))

from vispath_pkg.vispath import NT_COLORS, VisualizePath  # noqa: E402

UNKNOWN_GRAY = NT_COLORS['unknown'].lower()          # '#95A5A6'


def _page(tmp_path, nt_values, color_edges_by_nt=True):
    df = pd.DataFrame({
        'source': ['S', 'A', 'B'],
        'target': ['A', 'B', 'T'],
        'weight': [10, 20, 30],
        'nt_type': nt_values,
    })
    vp = VisualizePath(path_file=df, output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout='dagre',
                       color_edges_by_nt=color_edges_by_nt)
    vp.build_network()
    vp.create_network()
    page = Path(tmp_path) / f'{vp.base_filename}_network.html'
    assert page.exists(), 'no network page was written'
    return page.read_text(encoding='utf-8')


def _edge_payloads(html):
    start = html.find('[{"data": {"source":')
    assert start >= 0, 'no edge payload JSON on the page'
    depth, i = 0, start
    while i < len(html):
        if html[i] == '[':
            depth += 1
        elif html[i] == ']':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return [row.get('data', row) for row in json.loads(html[start:i + 1])]


def test_a_blank_transmitter_gets_the_unknown_gray(tmp_path):
    html = _page(tmp_path, ['acetylcholine', 'GLUT', ''])
    rule = html[html.index('edge[nt_type = "unknown"]'):]
    assert "'line-color': " in rule[:200], rule[:200]
    assert UNKNOWN_GRAY in rule[:200].lower(), rule[:200]
    assert '<option value="nt_unknown">Unknown Edges</option>' in html, (
        'the reader cannot recolour or hide the bucket the page invented')
    buckets = {str(e.get('nt_type')) for e in _edge_payloads(html)}
    assert buckets == {'acetylcholine', 'GLUT', 'unknown'}, buckets


def test_the_unknown_bucket_is_not_invented_for_a_run_with_no_vocabulary(
        tmp_path):
    """Wrong hypothesis: always emit the unknown rule. A run whose data carries
    no transmitter at all would then draw every edge gray, claiming a
    distinction it has no information to make — the payload still says
    'unknown', but no colour rule is written and the base edge colour stays."""
    html = _page(tmp_path, ['', '', ''])
    assert 'edge[nt_type = "unknown"]' not in html, html[:200]
    assert 'nt_unknown' not in html
    assert all(str(e.get('nt_type')) == 'unknown'
               for e in _edge_payloads(html))


def test_a_page_that_does_not_colour_by_transmitter_stays_unrulled(tmp_path):
    """`color_edges_by_nt` is opt-in: with it off the page draws the base link
    colour for every edge, writes no NT rule at all, and the payload keeps the
    raw column (blank stays blank — no bucket is imposed on a page that is not
    colouring by transmitter)."""
    html = _page(tmp_path, ['acetylcholine', 'GLUT', ''],
                 color_edges_by_nt=False)
    assert 'edge[nt_type = "unknown"]' not in html
    assert 'edge[nt_type = "acetylcholine"]' not in html
    by_edge = {(e['source'], e['target']): str(e.get('nt_type'))
               for e in _edge_payloads(html)}
    assert by_edge[('S', 'A')] == 'acetylcholine', by_edge
    assert '' in by_edge.values(), by_edge


def test_placeholder_spellings_share_one_bucket(tmp_path):
    """'None' / 'nan' / 'unknown' as CELL VALUES are no annotation, so they
    must not each become their own colour class."""
    html = _page(tmp_path, ['acetylcholine', 'None', 'nan'])
    for spelling in ('None', 'nan'):
        assert f'edge[nt_type = "{spelling}"]' not in html, spelling
    assert 'edge[nt_type = "unknown"]' in html
    buckets = {str(e.get('nt_type')) for e in _edge_payloads(html)}
    assert buckets == {'acetylcholine', 'unknown'}, buckets
