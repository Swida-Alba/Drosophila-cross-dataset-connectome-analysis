"""The type-level path export is FINALIZED, or the run says so.

`src_to_tgt_*_allpaths_type.csv` is the run's primary path table (the UI's
default view), and the finalize block reads it back, adds the per-node
Coverage column, re-scopes the realized per-hop Weights/Ratios, then re-writes
it STRONGEST-FIRST (length ascending, min_weight descending). The reload right
after it takes only the FIRST `pathN_to_show` rows for the HTML pages — so a
finalize that silently fails leaves the exported table in streaming order AND
the run's own pages showing the wrong "top paths".

Measured on the real cache (male-cns:v1.0, 2026-10-16 review round): three of
the four completed run folders exported an unsorted table, e.g. R2's
min_weight column opened 1, 2, 3, 1, 3, 7, 2, 1 across 631 rows. Cause: the
`_unresolvable_paths` counter was initialized inside the nested
`_patch_weights` and read in the enclosing scope, so every run that reached
the block raised NameError into an advisory `except` that only printed at
verbosity 'full'.

Wrong hypothesis tested by law 2: a file that happens to already be in
strongest-first order would pass a bare order check — so the forced-failure
run must be measurably OUT of order (positive control), and law 2 doubles as
the disclosure check.
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / 'src',
           PROJECT_ROOT / 'vispath-subproject' / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import coana  # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402

# Types: S->TS, A->TA, B->TB, C->TC, T->TT. The DFS enumerates the two 2-hop
# paths in neighbour order (S->B->T, bottleneck 3, before S->A->T, 6), so the
# streaming file is NOT already strongest-first and the order law has teeth.
_EDGES = [("S", "T", 12), ("S", "A", 8), ("A", "T", 6),
          ("S", "B", 8), ("B", "T", 3), ("A", "C", 9), ("C", "T", 7)]


def _run(monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, _calls, _logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _EDGES, max_interlayer=2)
    fc.FindAllPath()
    return Path(fc.allpath_folder)


def _rows(run_dir):
    import pandas as pd
    csv = next(Path(run_dir).glob('*_allpaths_type.csv'))
    frame = pd.read_csv(csv)
    return list(zip(frame['length'], frame['min_weight'])), frame, csv


def _force_finalize_failure(monkeypatch):
    """Make the finalize block raise, exactly where the run used to lose the
    sort: the read-back of the type-level table."""
    import polars as pl
    real = pl.read_csv

    def _boom(path, *a, **kw):
        if str(path).endswith('_allpaths_type.csv'):
            raise RuntimeError('forced finalize failure (test)')
        return real(path, *a, **kw)

    monkeypatch.setattr(pl, 'read_csv', _boom)


def test_the_type_table_is_exported_strongest_first(monkeypatch, tmp_path):
    rows, frame, csv = _rows(_run(monkeypatch, tmp_path))
    assert rows == sorted(rows, key=lambda k: (k[0], -k[1])), (
        f'{csv.name}: rows are in streaming order {rows}, the documented '
        f'order is {sorted(rows, key=lambda k: (k[0], -k[1]))} — the '
        f'finalize block did not complete')
    assert rows[0][0] == 1, f'1-hop path is not first: {rows}'
    assert frame['min_weight'].tolist()[:3] == [12, 6, 3], (
        f'not weakest-hop-descending inside the 2-hop block: {rows}')
    assert 'coverage' in frame.columns, (
        f'the Coverage column written by the same block is missing: '
        f'{list(frame.columns)}')
    notes = Path(csv.parent) / 'user_warning_notes.txt'
    text = notes.read_text(encoding='utf-8') if notes.exists() else ''
    assert '[type-path sort]' not in text, f'run disclosed a failure:\n{text}'


def test_a_failed_finalize_is_out_of_order_AND_the_run_says_so(monkeypatch,
                                                               tmp_path):
    _force_finalize_failure(monkeypatch)
    rows, _frame, csv = _rows(_run(monkeypatch, tmp_path))
    # positive control: without the finalize the export really is unordered,
    # so the passing law above is not scoring an accident of discovery order
    assert rows != sorted(rows, key=lambda k: (k[0], -k[1])), (
        f'the fixture is already strongest-first ({rows}) — the order law '
        f'would pass even with the finalize broken; pick edges whose DFS '
        f'order differs')
    notes = (csv.parent / 'user_warning_notes.txt').read_text(encoding='utf-8')
    assert '[type-path sort]' in notes, (
        'a failed finalize is invisible in the run folder:\n' + notes)
    assert 'path CONTENT is unaffected' in notes, notes


@pytest.mark.parametrize("rel", [
    'src/coana.py', 'src/statvis.py', 'src/paths_pair_report.py',
    'src/type_level_refill.py', 'src/route_plan.py',
    'src/connection_ratio_paths.py', 'ui/config.py',
    'ui/components/pathfinding_controls.py', 'ui/tabs/network.py',
    'ui/tabs/inter_dataset.py', 'ui/output_guide.py',
])
def test_the_engine_has_no_unbound_names(rel):
    """The class gate for this defect.

    pyflakes cannot report a hard 'undefined name' while a module carries
    `from neuprint import *` — it downgrades every free name to
    ImportStarUsage ('may be undefined, or defined from star imports'), which
    the earlier lint census grepped for 'undefined' and missed. That blind
    spot is how `_PRE`/`_POST` (the untyped-fetch path) and
    `_unresolvable_paths` (every run with type-level paths) survived to real
    data.
    """
    checker = pytest.importorskip('pyflakes.checker')
    messages = pytest.importorskip('pyflakes.messages')
    source = (PROJECT_ROOT / rel).read_text(encoding='utf-8')
    unbound = [m for m in checker.Checker(ast.parse(source),
                                         filename=rel).messages
               if isinstance(m, (messages.UndefinedName,
                                 messages.UndefinedLocal,
                                 messages.UndefinedExport,
                                 messages.ImportStarUsage))]
    assert not unbound, [
        f'{rel}:{m.lineno}: {m.message % m.message_args}' for m in unbound]
