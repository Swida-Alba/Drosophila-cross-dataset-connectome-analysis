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

# ---------------------------------------------------------------------------
# The bare-instance contract: a NEW attribute on VisualizePath.__init__ that
# the vendored templates read must also exist in
# _RendererVisualizer._SAFE_FALLBACKS, because the mapping renderer builds the
# visualizer with __new__ (no __init__) and calls those templates directly.
# 2026-10-15: adding the adjusted-ratio label (vispath `Adj. Ratio (>= N)`,
# audit F7.3) broke 25 type-mapping tests that no pathfinding battery touches;
# this pin makes that failure immediate, names the attribute, and names the
# file to edit.
# ---------------------------------------------------------------------------

def test_bare_visualizer_supplies_every_attribute_the_templates_read():
    import ast
    import inspect
    import pathlib

    # The subproject path is set up by the mapping module itself (vispath is
    # optional), so resolve the module the same way the renderer does.
    from comparison.mapping_visualization import _import_vispath
    assert _import_vispath() is not None, 'vispath unavailable'
    import vispath_pkg.vispath as vispath_mod

    mapping_src = pathlib.Path(
        __file__).resolve().parents[2] / 'src/comparison' \
        / 'mapping_visualization.py'
    tree = ast.parse(mapping_src.read_text(encoding='utf-8'))
    fallbacks = set()
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == '_SAFE_FALLBACKS' for t in node.targets):
            fallbacks = {k.value for k in node.value.keys
                         if isinstance(k, ast.Constant)}
        # `visualizer.<attr> = ...` before the renderer call
        if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name) and node.value.id == 'visualizer':
            assigned.add(node.attr)
    assert fallbacks, '_SAFE_FALLBACKS not found — the contract moved'

    vtree = ast.parse(inspect.getsource(vispath_mod))
    cls = next(n for n in ast.walk(vtree) if isinstance(n, ast.ClassDef)
               and n.name == 'VisualizePath')
    # Names a bare instance resolves through the CLASS (methods, class-level
    # constants) need no fallback — only per-instance state does.
    class_level = {n.name for n in cls.body
                   if isinstance(n, ast.FunctionDef)}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            class_level |= {t.id for t in stmt.targets
                            if isinstance(t, ast.Name)}
    read = set()
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in (
                '_plot_cytoscape_network', '_create_sankey_html_with_controls'):
            for att in ast.walk(fn):
                if (isinstance(att, ast.Attribute)
                        and isinstance(att.value, ast.Name)
                        and att.value.id == 'self'
                        and isinstance(att.ctx, ast.Load)):
                    read.add(att.attr)
    assert read, 'the template methods were renamed — update this law'

    # __getattr__ resolves any __init__ PARAMETER name from its signature
    # default, so those need no map entry; a body-COMPUTED attribute has no
    # such fallback, and the map is the only place it can come from.
    init_params = set(
        inspect.signature(vispath_mod.VisualizePath.__init__).parameters)

    missing = sorted(read - fallbacks - assigned - class_level - init_params)
    assert not missing, (
        f'the bare mapping visualizer cannot resolve {missing}: add each to '
        f'_RendererVisualizer._SAFE_FALLBACKS in '
        'src/comparison/mapping_visualization.py (a real __init__ sets them, '
        'the __new__ path does not — `adj_ratio_label_plain` slipped through '
        'this hole on 2026-10-15 and broke 25 mapping tests)')
