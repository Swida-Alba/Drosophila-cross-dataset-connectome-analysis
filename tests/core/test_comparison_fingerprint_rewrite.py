"""Round-7 report, J6: after a fingerprint-mismatch rejection the cached
folder must be REWRITTEN — new ``connections_edge.csv`` AND a sidecar
naming the new query. The Windows round-7 run observed both datasets log
``written for a different query — ignoring it`` yet both sidecars ended
byte-identical to run 1's (0 changed); this pins the rewrite contract.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comparison import comparison_analyzer as ca  # noqa: E402
from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from comparison.comparison_parameters import ComparisonParameters  # noqa: E402


def _analyzer(target_pattern, output_folder, saveas):
    params = ComparisonParameters(
        datasets=['flywire_FAFB_v783', 'banc_v888'],
        source_neurons=['s-LNv.*'],
        target_neurons=[target_pattern],
        thresholds=[3],
        comparison_mode='path',
        max_interlayer=2,
        output_folder=output_folder,
        saveas=saveas,
    )
    return ComparisonAnalyzer(params, verbose=False)


def _stub_fnc(monkeypatch, analyzer, edge_rows):
    """Stub the per-dataset path analysis to a fixed edge frame."""

    def fake_run(dataset_name, threshold, **kwargs):
        return pd.DataFrame(edge_rows)

    monkeypatch.setattr(analyzer, 'run_path_analysis', fake_run)


def _fingerprint_of(dirpath):
    sidecar = Path(dirpath) / 'connections_edge.fingerprint.json'
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding='utf-8'))


def test_mismatch_run_rewrites_sidecar_with_new_query(tmp_path, monkeypatch):
    # The 2026-09-30 code audit closed the J6 third state: a definitive
    # fingerprint mismatch now returns None from _try_load_cached instead
    # of falling through to the same folder's unchecked legacy files, so
    # the re-run actually re-derives and the sidecar rewrite (or the
    # stale-pair removal) fires. Flipped from xfail(strict) to a hard
    # assertion by that fix.
    output_folder = str(tmp_path)
    rows = [{'bodyId_pre': 1, 'bodyId_post': 2, 'weight': 5}]

    # run 1: target DN1a.*
    a1 = _analyzer('DN1a.*', output_folder, 'j6_folder')
    _stub_fnc(monkeypatch, a1, rows)
    a1.run_comparison(skip_existing=True)

    sidecars = sorted(Path(output_folder).rglob(
        'connections_edge.fingerprint.json'))
    assert len(sidecars) == 2, (
        f'run 1 must stamp both datasets: {[str(s) for s in sidecars]}')
    fp1 = _fingerprint_of(sidecars[0].parent)

    # run 2: DIFFERENT target, same folder — the mismatch must fire AND
    # the rewrite must name the new query
    a2 = _analyzer('DN2.*', output_folder, 'j6_folder')
    _stub_fnc(monkeypatch, a2, rows)
    a2.run_comparison(skip_existing=True)

    # The binding contract: NO sidecar may still name the run-1 query.
    # Each dataset takes one of two honest outcomes — the sidecar is
    # rewritten to the new query (legacy save flow), or the stale pair is
    # removed outright (replay loader flow). The round-7 Windows failure
    # was the third state: stale pair surviving a mismatched re-run.
    sidecars2 = sorted(Path(output_folder).rglob(
        'connections_edge.fingerprint.json'))
    for sidecar in sidecars2:
        fp2 = _fingerprint_of(sidecar.parent)
        assert fp2['target_neurons'] != fp1['target_neurons'], (
            f'sidecar {sidecar} still names the run-1 query — the mismatch '
            'rejection re-derived the data but left the stale stamp '
            '(round-7 J6: 0 of 2 changed)')
    # and no orphaned stale pair: every remaining connections_edge.csv
    # sits beside a sidecar that names its query
    for csv in Path(output_folder).rglob('connections_edge.csv'):
        assert _fingerprint_of(csv.parent) is not None or \
            csv.stat().st_size == 0, f'stale unlabelled pair at {csv}'


def test_mismatch_verdict_is_caller_conditional(tmp_path, monkeypatch):
    """2026-09-30 verification round: the mismatch handling must serve BOTH
    callers. The PRE-derivation probe (remove_stale=False) returns None —
    a definitive mismatch must not serve the folder's unchecked legacy
    files, forcing re-derivation instead. The POST-replay loader
    (remove_stale=True) removes the stale pair and FALLS THROUGH — the
    replay has just rewritten paths.csv for the current query, and the
    blanket return-None made that lane return an empty frame (silently
    empty exports for a default-flow mismatched re-run)."""
    a = _analyzer('DN2.*', str(tmp_path), 'cond_folder')
    folder = (tmp_path / 'cond_folder' / 'dataset_data' / 'banc_v888'
              / 'minsyn_3')
    folder.mkdir(parents=True)
    # stale pair naming a different query + a legacy file with FRESH rows
    fresh = pd.DataFrame([{'bodyId_pre': 1, 'bodyId_post': 2, 'weight': 7}])
    fresh.to_csv(folder / 'connections_edge.csv', index=False)
    (folder / 'connections_edge.fingerprint.json').write_text(
        json.dumps({'target_neurons': ['OLD']}), encoding='utf-8')
    fresh.to_csv(folder / 'paths.csv', index=False)

    # probe: definitive miss, no removal
    assert a._try_load_cached('banc_v888', 3, remove_stale=False) is None
    assert (folder / 'connections_edge.csv').exists()

    # post-replay load: stale pair removed, fresh legacy rows served
    loaded = a._try_load_cached('banc_v888', 3, remove_stale=True)
    assert loaded is not None and len(loaded) == 1
    assert int(loaded.iloc[0]['weight']) == 7
    assert not (folder / 'connections_edge.csv').exists()
    assert not (folder / 'connections_edge.fingerprint.json').exists()
