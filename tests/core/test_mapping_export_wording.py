"""Export wording: a type-mapping artifact counts neurons, never synapses.

2026-09-26: the type mapper's Sankey hover read "Synapses: 9" and its metric
selector "Synapse Count", because the vendored vispath templates hard-coded
the pathway vocabulary — ``edge_weight_label='neurons'`` was already being
passed and simply dropped on the floor.  The same templates appended a
pathway hop layer to every Sankey node label ("s-CPDN3B · FAFB (L0)"), which
a mapping view — whose columns are DATASETS — states wrong, especially once a
half's ribbons are reversed to centre its source dataset.

These tests render from synthetic flows only.  The second half of the gate is
that a PATHWAY artifact keeps its original wording: the new knobs default to
it, so no caller that does not pass them changes.
"""

from comparison.mapping_visualization import (
    render_mapping_network_html,
    render_mapping_sankey_html,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'


def _flow(source_type, foreign_type, source_count=2, foreign_count=2):
    return {
        'source_dataset': FAFB, 'target_dataset': MCNS,
        'source_type': source_type, 'foreign_type': foreign_type,
        'source_count': source_count, 'foreign_count': foreign_count,
        'matched_origin': 'type',
        'bridges': [[
            {'dataset': FAFB, 'column': 'type', 'value': source_type},
            {'dataset': MCNS, 'column': 'type', 'value': foreign_type},
        ]],
    }


FLOWS = [_flow('s-LNv', 'sLNv'), _flow('APDN3', 'CL125', 4, 4)]


def _visible_text(html: str) -> str:
    """The artifact minus JS comment lines (vispath's own internals may keep
    saying 'synapse' where it describes the pathway scale)."""
    return '\n'.join(line for line in html.splitlines()
                     if not line.strip().startswith('//'))


def test_mapping_sankey_never_speaks_pathway():
    html = render_mapping_sankey_html(FLOWS, variant='type')
    assert html
    text = _visible_text(html)
    assert 'Synapse' not in text and 'synapse' not in text
    # no hop-layer suffix on node labels or hovers — the columns are datasets
    assert '(L0)' not in text and '(L1)' not in text
    assert 'Neuron count' in text
    assert 'Type mapping Sankey — FAFB → MCNS' in text
    # nodes are named by type and dataset, which is the whole point of the
    # column layout
    assert 's-LNv · FAFB' in text
    # the downloaded file's own tab title follows the artifact, not the
    # renderer's generic default
    assert '<title>Type mapping Sankey — FAFB → MCNS</title>' in text


def test_mapping_network_metric_selector_follows_the_unit():
    html = render_mapping_network_html(FLOWS)
    assert html
    text = _visible_text(html)
    assert 'Synapse Count' not in text
    assert 'Neuron count' in text
    # the layout dropdown pre-selects the layout actually rendered; the
    # template once shipped its Python conditional as HTML text
    assert '<option value="dagre" selected>' in html
    assert "{'selected' if" not in html


def test_pathway_sankey_keeps_its_own_wording():
    """The defaults are the pathway behaviour — an unset knob changes
    nothing for connectome artifacts."""
    import os
    import tempfile

    import pandas as pd

    from vispath_pkg.vispath import VisualizePath

    frame = pd.DataFrame({
        'path_block': ['A -> B -> C', 'A -> B -> D'],
        'weights': [[10, 5], [10, 7]],
    })
    with tempfile.TemporaryDirectory(prefix='vispath_pathway_') as tmp:
        visualizer = VisualizePath(path_file=frame, output_folder=tmp,
                                   showfig=False, verbose=False)
        path = visualizer.visualize_sankey()
        assert path and os.path.isfile(path)
        with open(path, encoding='utf-8') as handle:
            html = handle.read()
    text = _visible_text(html)
    assert 'Synapse Count' in text
    assert 'Sankey diagram of pathway connections' in text
    assert '(L1)' in text
