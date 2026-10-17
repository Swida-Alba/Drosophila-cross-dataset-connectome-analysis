"""The `conserved_network` producer's per-cell style wiring.

`clock-production`'s issue (§4.5) asks that the cross-dataset conservation claim
— a pair reported by every dataset, versus one reported by only some, plus the
BANC agreement tier — be readable on the matrix page instead of being averaged
into one weight. The producer already holds that record: `edge_labels` carries
one `'{nickname} wt'` entry per dataset that reports the pair. These laws pin
that the maps are read off that record and reach the page, and that nothing
changes when no dataset is asked for.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/core/test_conserved_network_cell_style.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from vispath_pkg import VisualizePath  # noqa: E402

FILL_SET = ['male-cns:v1.0', 'flywire_FAFB_v783', 'banc_v626']
NICKS = {'male-cns:v1.0': 'MCNS', 'flywire_FAFB_v783': 'FAFB', 'banc_v626': 'BANC'}

#: A four-pair record in the shape the producer builds it: `'{nickname} wt'`
#: per reporting dataset (and the ratio/nt siblings the hover carries, which the
#: map builder must ignore).
EDGE_LABELS = {
    ('5th-LNv', 'LPN'): {'MCNS wt': 3, 'MCNS ratio': 0.02, 'FAFB wt': 8.0,
                         'FAFB nt': 'ACh'},
    ('5th-LNv', 'APDN3'): {'MCNS wt': 106, 'FAFB wt': 116, 'BANC wt': 11},
    ('DN1a', 'PPL101'): {'FAFB wt': 40, 'BANC wt': 5},
    ('s-LNv', 'DPSVP036'): {'MCNS wt': 7},
}


@pytest.fixture
def analyzer():
    # the maps are a pure read of the record the producer already built, so the
    # law needs no comparison run — and no Neuprint client
    return ComparisonAnalyzer.__new__(ComparisonAnalyzer)


def test_fill_is_the_all_of_the_fill_set_claim_and_outline_is_any(analyzer):
    fill, outline, scale = analyzer._cell_style_maps_from_presence(
        EDGE_LABELS, FILL_SET, ['banc_v626'], NICKS)
    assert fill == {
        ('5th-LNv', 'LPN'): 0,        # MCNS+FAFB but no BANC
        ('5th-LNv', 'APDN3'): 1,      # all three
        ('DN1a', 'PPL101'): 0,        # FAFB+BANC, no MCNS
        ('s-LNv', 'DPSVP036'): 0,     # MCNS only
    }, fill
    assert outline == {('5th-LNv', 'APDN3'): True, ('DN1a', 'PPL101'): True}, outline
    # both tiers priced: an unpriced category is a refusal inside the renderer
    assert set(scale) == {0, 1} and scale[0] != scale[1], scale
    # and the conserved tier is not the page's unmapped grey, or "some datasets"
    # and "no record" would be the same colour
    assert scale[1].lower() != '#f2f2f2' and scale[0].lower() != '#f2f2f2', scale


def test_a_narrower_fill_set_grades_narrower(analyzer):
    fill, _o, _s = analyzer._cell_style_maps_from_presence(
        EDGE_LABELS, ['male-cns:v1.0', 'flywire_FAFB_v783'], None, NICKS)
    assert fill[('5th-LNv', 'LPN')] == 1, fill   # the article's FAFB∧MCNS tier
    assert fill[('s-LNv', 'DPSVP036')] == 0, fill


def test_the_short_code_spells_the_same_dataset(analyzer):
    by_name, _o1, _s1 = analyzer._cell_style_maps_from_presence(
        EDGE_LABELS, ['banc_v626'], None, NICKS)
    by_code, _o2, _s2 = analyzer._cell_style_maps_from_presence(
        EDGE_LABELS, ['BANC'], None, NICKS)
    assert by_name == by_code == {
        ('5th-LNv', 'APDN3'): 1, ('DN1a', 'PPL101'): 1,
        ('5th-LNv', 'LPN'): 0, ('s-LNv', 'DPSVP036'): 0}, (by_name, by_code)


def test_a_dataset_that_reports_nothing_here_raises_instead_of_greying(analyzer):
    """A silently unmatched release would colour EVERY cell as unconserved and
    read to the reader as an absence of conservation."""
    with pytest.raises(ValueError) as exc:
        analyzer._cell_style_maps_from_presence(
            EDGE_LABELS, ['hemibrain_v1_2_1'], None, NICKS)
    assert 'hemibrain_v1_2_1' in str(exc.value), str(exc.value)
    # the refusal names what IS here, so the caller can fix the spelling
    assert 'MCNS' in str(exc.value), str(exc.value)


def test_nothing_asked_is_nothing_passed(analyzer):
    assert analyzer._cell_style_maps_from_presence(
        EDGE_LABELS, None, None, NICKS) == ({}, {}, {})


# ---------------------------------------------------------------------------
# the create_heatmap surface the producer calls
# ---------------------------------------------------------------------------

PATH_FRAME = pd.DataFrame({
    'source': ['5th-LNv', '5th-LNv', 'DN1a'],
    'target': ['LPN', 'APDN3', 'PPL101'],
    'weight': [8, 116, 40],
    'synapses': [8, 116, 40],
    'type': ['direct', 'direct', 'direct'],
})


def _heatmap(tmp_path, **kwargs):
    vp = VisualizePath(path_file=PATH_FRAME.copy(), output_folder=str(tmp_path),
                       verbose=False, showfig=False)
    vp.build_network()
    return vp, vp.create_heatmap(**kwargs)


def test_create_heatmap_takes_the_maps_and_the_page_carries_them(tmp_path):
    fill = {('5th-LNv', 'LPN'): 1, ('5th-LNv', 'APDN3'): 0}
    outline = {('5th-LNv', 'APDN3'): True}
    _vp, page = _heatmap(tmp_path, cell_fill=fill, cell_outline=outline,
                         fill_scale={0: '#9aa0a6', 1: '#2171b5'})
    html = Path(page).read_text(encoding='utf-8')
    assert page.endswith('_heatmap.html'), page
    for token in ('cellStyleFields', 'cellFillSource', 'layout.shapes',
                  '"#2171b5"', '"#9aa0a6"'):
        assert token in html, token
    # keyed by label, and the pair nobody filled is still a cell
    assert '5th-LNv\\tLPN' in html, html[html.index('cellStyleFields'):][:400]


def test_create_heatmap_without_maps_writes_the_plain_page(tmp_path):
    _vp, page = _heatmap(tmp_path)
    html = Path(page).read_text(encoding='utf-8')
    for token in ('cellStyleFields', 'cellFillSource', 'layout.shapes'):
        assert token not in html, token
