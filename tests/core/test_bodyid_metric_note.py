"""The bodyId pages' metric caption is quoted only where it is true.

A bodyId-level Network/Heatmap/Sankey page disables Refilled and the adjusted
ratio: both are computed per TYPE pair, never per bodyId (the user ruling of
2026-10-10). The disabled option was the page's only word about that, so a
reader could not tell design from defect — the caption now says so (review
2026-10-17, 'yes. a note').

The caption's second sentence points at the type-level page beside the folder
and at `type_level_refill/`, so it is a claim about the FOLDER: it may only be
quoted when the run wrote those records and its type-level page really carries
them. Measured here: the guard's two halves on hand-built folders, the wiring
on whole offline runs, and the surface that must stay silent. The rendered
pages are `tests/vispath/test_metric_labels.py`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / 'src',
           PROJECT_ROOT / 'vispath-subproject' / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import coana  # noqa: E402
from tests.core.test_applied_threshold_provenance import _BUDGET_EDGES  # noqa: E402
from tests.core.test_coana_coverage import make_fc  # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402

NOTE = coana.FindNeuronConnection._BODYID_METRIC_NOTE
MIRROR_PAIRS = ('data_details', 'type_level_refill',
                'refill_full_type_pairs.csv')


def _install_recorder(monkeypatch):
    """Record every VisualizePath construction the run makes.

    Which CALLER hands which caption is the subject, not what a page renders,
    so the stand-in only has to survive the run's method calls.
    """
    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))
            self.G_network = None
            self.edge_limit_trimmed = False
            self.base_filename = 'x'

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(coana, 'VisualizePath', Recorder)
    return calls


def _is_bodyid(call):
    return str(call.get('output_folder', '')).endswith('bodyId_visualization')


def _write_mirror(folder, totals=(12.0,)):
    """The refill round's own output shape: pair rows carrying refilled_total."""
    out = Path(folder).joinpath(*MIRROR_PAIRS)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'type_pre': 'TS', 'type_post': 'TA',
                   'refilled_total': t} for t in totals]).to_csv(out,
                                                                 index=False)
    return out


def _fc(folder):
    fc, _logs = make_fc(allpath_folder=str(folder))
    return fc


@pytest.mark.parametrize('rows, want_note', [
    ((12.0,), True),          # the run published refilled totals
    ((None,), False),         # rows without a total are not a measurement
    (None, False),            # no refill round at all
])
def test_the_caption_is_quoted_exactly_when_the_run_wrote_the_totals(
        tmp_path, rows, want_note):
    """Both halves of the guard, on the folder the pages read."""
    if rows is not None:
        _write_mirror(tmp_path, totals=rows)
    else:
        tmp_path.mkdir(exist_ok=True)
    got = _fc(tmp_path)._bodyid_metric_note()
    assert (got == NOTE) is want_note, (got[:40] if got else got, want_note)
    assert got in (NOTE, ''), repr(got)


def test_the_direct_connection_surface_stays_silent(tmp_path, monkeypatch):
    """`FindDirectConnections` builds BOTH of its pages from a one-hop frame
    that carries no refilled total and no adjusted ratio, so even a folder with
    refill records beside it gets no caption: the note promises the page next
    door has the numbers, and here it does not. Wrong hypothesis: handing the
    note to every bodyId page regardless of its sibling."""
    calls = _install_recorder(monkeypatch)
    folder = tmp_path / 'direct'
    folder.mkdir()
    _write_mirror(folder)
    fc, _logs = make_fc(
        conn_type=pd.DataFrame({
            'type_pre': ['TS'], 'type_post': ['TA'], 'weight': [10],
            'connection_ratio': [0.4], 'traversal_probability': [0.6],
        }),
        conn_df=pd.DataFrame({
            'bodyId_pre': ['1'], 'bodyId_post': ['2'], 'weight': [10],
            'connection_ratio': [0.4], 'traversal_probability': [0.6],
            'type_pre': ['TS'], 'type_post': ['TA'],
        }),
        conn_group=pd.DataFrame(),
        direct_folder=str(folder), showfig=False, edgeN_limit=0,
        network_layout='hierarchical', output_format='csv',
        min_synapse_num=1,
    )
    fc.VisualizeDirectConnections_simple()
    bodyid = [c for c in calls if _is_bodyid(c)]
    assert bodyid, 'the fixture built no bodyId page to check'
    assert not any(c.get('metric_note') for c in calls), (
        'a direct-connection page promised totals its type-level page does '
        'not carry')


@pytest.mark.parametrize('finder', ['FindAllPath', 'FindShortestPath'])
def test_a_path_run_agrees_with_its_own_refill_records(finder, tmp_path,
                                                     monkeypatch):
    """The two FindPath-family bodyId sites, on a whole offline run: the page's
    caption is available exactly when the run's Refilled metric is, read here
    independently from the finished folder."""
    fc, _c, _logs = _make_pipeline_fc(monkeypatch, tmp_path, _BUDGET_EDGES,
                                      max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    fc.skip_bodyId = False
    monkeypatch.setattr(coana.FindNeuronConnection, '_ensure_neuprint_client',
                        lambda self: None)
    empty = pd.DataFrame({'bodyId': [], 'type': []})
    monkeypatch.setattr(coana, 'fetch_neurons', lambda *a, **k: (empty, ''))
    calls = _install_recorder(monkeypatch)   # after the harness's own stub
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    getattr(fc, finder)()
    run = Path(fc.allpath_folder)
    bodyid = [c for c in calls if _is_bodyid(c)]
    assert bodyid, f'{finder} built no bodyId-level page to check'
    has_totals = bool(fc._refilled_pair_metrics(run))
    for call in bodyid:
        assert bool(call.get('metric_note')) is has_totals, (
            f'{finder}: caption={call.get("metric_note")!r} while the run\'s '
            f'type-level Refilled metric is '
            f'{"available" if has_totals else "absent"}')
    assert not any(c.get('metric_note') for c in calls if not _is_bodyid(c))
