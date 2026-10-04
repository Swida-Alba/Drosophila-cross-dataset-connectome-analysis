"""Shortest-path coverage early-stop (2026-10-04, user-approved design).

Two knobs — ``shortest_source_coverage`` / ``shortest_target_coverage`` —
stop backward discovery at a layer boundary once both sides' PER-TYPE
coverage requirements are met, instead of exhausting the depth bound on
per-pair completeness. Contract pinned here:

- measured per queried TYPE at each layer boundary (source covered once
  it reaches >= 1 target; target covered once >= 1 source reaches it);
- ``0.0`` = Any (>= 1 bodyId per type), a fraction = ceil(frac x N) per
  type, ``1.0`` = Full, ``None`` = legacy per-pair completeness;
- emitted pairs keep their EXACT per-pair minimum hop (subset of what a
  full-depth run finds — never a shorter-but-wrong hop count);
- a fired stop is disclosed (note + shortest_discovery_diagnostics
  .coverage_stop); an unmeetable requirement falls back to today's
  depth-cap behavior with identical outputs;
- the monolithic and store discovery twins stop identically.
"""

import inspect

import pytest

import coana
from tests.core.test_pathfinding import _PIPELINE_TYPES, _make_pipeline_fc
from tests.core.test_shortest_batched_discovery import (
    _SYNTH_EDGES, _SYNTH_SOURCES, _SYNTH_TARGETS,
)

# Extra universe for per-type strictness: TS1 gains a second member (S1b)
# that only connects at hop 3, and a type with a NEVER-reachable member
# (S5 of TS5 — no edges) for the fallback test.
_EXTRA_TYPES = {
    'S1b': 'TS1',   # second TS1 member, reached one hop later than S1
    'S5': 'TS5',    # enrolled but never connected (per-type 0 blocks)
}
_EXTRA_EDGES = [
    ('S1b', 'A', 6),   # S1b -> A -> T1 (same cone as S1, 1 hop deeper
                       # from every target that sees A at hop 2)
]
_EXTRA_SOURCES = ('S1', 'S1b', 'S2', 'S3')
# NOTE: S5 is deliberately NOT enrolled in the default universes; tests
# that need it pass source_ids including it explicitly.


@pytest.fixture()
def synth_types():
    saved = dict(_PIPELINE_TYPES)
    _PIPELINE_TYPES.update({
        'S1': 'TS1', 'S2': 'TS2', 'S3': 'TS3',
        'A': 'TA', 'B': 'TB', 'C': 'TC', 'X': 'TX', 'Y': 'TY', 'D': 'TD',
        'Z0': 'TZ', 'Z1': 'TZ', 'Z2': 'TZ', 'Z3': 'TZ',
        'T1': 'TT1', 'T2': 'TT2', 'T3': 'TT3', 'T4': 'TT4',
    })
    _PIPELINE_TYPES.update(_EXTRA_TYPES)
    yield _PIPELINE_TYPES
    _PIPELINE_TYPES.clear()
    _PIPELINE_TYPES.update(saved)


def _make_fc(monkeypatch, tmp_path, *, budget=None, edges=None,
             sources=None, targets=None, **kwargs):
    monkeypatch.setattr(
        coana.FindNeuronConnection, "_ensure_neuprint_client",
        lambda self: None)
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, edges or _SYNTH_EDGES + _EXTRA_EDGES,
        max_interlayer=3,
        source_ids=sources or _EXTRA_SOURCES,
        target_ids=targets or _SYNTH_TARGETS, **kwargs)
    fc.skip_bodyId = False
    if budget is not None:
        fc.discovery_batch_budget = budget
    return fc, fetch_calls, logs


def _bodyid_pairs_and_hops(fc):
    """(source, target, hop_count) for every emitted bodyId path."""
    import os
    path = os.path.join(
        fc.allpath_folder,
        f'{fc.source_fname}_to_{fc.target_fname}_allpaths_bodyId_paths.csv')
    import pandas as pd
    frame = pd.read_csv(path)
    out = set()
    for chain in frame['path'].astype(str):
        nodes = chain.split('->')
        out.add((nodes[0], nodes[-1], len(nodes) - 1))
    return out


def _attrs(fc):
    import json
    import os
    with open(os.path.join(fc.allpath_folder, 'all_attributes.json')) as f:
        return json.load(f)


def _notes_text(fc):
    import os
    path = os.path.join(fc.allpath_folder, 'user_warning_notes.txt')
    if not os.path.exists(path):
        return ''
    return open(path, encoding='utf-8').read()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_knobs_off_is_legacy(monkeypatch, tmp_path, synth_types):
    """None/None = no coverage gate: no record, no note, and the pair set
    matches a run made before the knobs existed (the existing equivalence
    suites guard byte-identity; here we pin the off-state)."""
    fc, _, _ = _make_fc(monkeypatch, tmp_path)
    fc.FindShortestPath()
    attrs = _attrs(fc)
    diag = attrs.get('shortest_discovery_diagnostics', {})
    assert diag.get('coverage_stop') is None
    assert 'shortest coverage stop' not in _notes_text(fc)


def test_full_target_stop_subset_and_exact(monkeypatch, tmp_path,
                                           synth_types):
    """Any source + Full target: the stop fires, the emitted pair set is a
    SUBSET of the legacy run's, and every emitted pair's hop count equals
    the legacy minimum (coverage scoping never distorts a pair's hop)."""
    fc_ref, _, _ = _make_fc(monkeypatch, tmp_path / 'ref')
    fc_ref.FindShortestPath()
    ref = _bodyid_pairs_and_hops(fc_ref)

    fc, _, _ = _make_fc(monkeypatch, tmp_path / 'cov')
    fc.shortest_source_coverage = 0.0
    fc.shortest_target_coverage = 1.0
    fc.FindShortestPath()
    got = _bodyid_pairs_and_hops(fc)

    stop = _attrs(fc)['shortest_discovery_diagnostics']['coverage_stop']
    assert stop is not None and stop['stopped_at_layer'] >= 1
    assert stop['requirements'] == {'source': 0.0, 'target': 1.0}
    assert 'shortest coverage stop' in _notes_text(fc)
    assert got <= ref
    # exactness: same pair -> same hop count
    ref_by_pair = {(s, t): h for s, t, h in ref}
    for s, t, h in got:
        assert ref_by_pair[(s, t)] == h


def test_any_any_stops_no_later_than_full_target(monkeypatch, tmp_path,
                                                 synth_types):
    """Any+Any is the loosest scope: its stop layer is <= the Any+Full
    layer and its pair set is a subset of the Any+Full run's."""
    def run(folder, src, tgt):
        fc, _, _ = _make_fc(monkeypatch, tmp_path / folder)
        fc.shortest_source_coverage = src
        fc.shortest_target_coverage = tgt
        fc.FindShortestPath()
        stop = _attrs(fc)['shortest_discovery_diagnostics']['coverage_stop']
        return stop['stopped_at_layer'], _bodyid_pairs_and_hops(fc)

    layer_any, pairs_any = run('anyany', 0.0, 0.0)
    layer_full, pairs_full = run('anyfull', 0.0, 1.0)
    assert layer_any <= layer_full
    assert pairs_any <= pairs_full


def test_half_source_per_type_strictness(monkeypatch, tmp_path,
                                         synth_types):
    """50% source: TS1 has 2 enrolled members (S1 at hop 2, S1b one hop
    deeper); the requirement is ceil(0.5 x 2) = 1 per type — met only
    once EVERY type has its share (a type still at 0 blocks the stop)."""
    fc, _, _ = _make_fc(monkeypatch, tmp_path)
    fc.shortest_source_coverage = 0.5
    fc.shortest_target_coverage = 0.0
    fc.FindShortestPath()
    stop = _attrs(fc)['shortest_discovery_diagnostics']['coverage_stop']
    assert stop is not None
    achieved = stop['achieved']['source']
    for type_name, value in achieved.items():
        n, total = value.split('/')
        assert int(n) >= 1, f'{type_name} still at 0 — stop must not fire'


def test_unmeetable_coverage_falls_back_to_depth(monkeypatch, tmp_path,
                                                 synth_types):
    """Full target with an enrolled-but-unreachable target type: the
    requirement can never be met, so the run behaves EXACTLY like legacy
    (no coverage_stop record, no note, identical pair set)."""
    unreachable = tuple(_SYNTH_TARGETS) + ('T9',)
    _PIPELINE_TYPES['T9'] = 'TT9'   # enrolled type with no edges at all

    fc_ref, _, _ = _make_fc(monkeypatch, tmp_path / 'ref',
                            targets=unreachable)
    fc_ref.FindShortestPath()

    fc, _, _ = _make_fc(monkeypatch, tmp_path / 'cov',
                        targets=unreachable)
    fc.shortest_source_coverage = None
    fc.shortest_target_coverage = 1.0
    fc.FindShortestPath()

    diag = _attrs(fc)['shortest_discovery_diagnostics']
    assert diag.get('coverage_stop') is None
    assert 'shortest coverage stop' not in _notes_text(fc)
    assert _bodyid_pairs_and_hops(fc) == _bodyid_pairs_and_hops(fc_ref)


def test_invalid_knob_refuses(monkeypatch, tmp_path, synth_types):
    """Out-of-range values are refused loudly (never silently coerced)."""
    fc, _, _ = _make_fc(monkeypatch, tmp_path)
    fc.shortest_target_coverage = 1.5
    with pytest.raises(ValueError):
        fc.FindShortestPath()


# ---------------------------------------------------------------------------
# Twin equivalence under the knob
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src,tgt", [(0.0, 1.0), (0.5, 0.5), (0.0, 0.0)])
def test_monolithic_store_equivalence(monkeypatch, tmp_path, synth_types,
                                      src, tgt):
    """Both discovery twins stop at the same layer with the same pair set
    under the same knobs."""
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_mono, _, _ = _make_fc(monkeypatch, tmp_path / 'mono', budget=0)
    fc_mono.shortest_source_coverage = src
    fc_mono.shortest_target_coverage = tgt
    fc_mono.FindShortestPath()

    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_batch, _, _ = _make_fc(monkeypatch, tmp_path / 'batch')
    fc_batch.shortest_source_coverage = src
    fc_batch.shortest_target_coverage = tgt
    fc_batch.FindShortestPath()

    stop_mono = _attrs(fc_mono)['shortest_discovery_diagnostics'][
        'coverage_stop']
    stop_batch = _attrs(fc_batch)['shortest_discovery_diagnostics'][
        'coverage_stop']
    assert (stop_mono or {}).get('stopped_at_layer') == \
        (stop_batch or {}).get('stopped_at_layer')
    assert _bodyid_pairs_and_hops(fc_mono) == _bodyid_pairs_and_hops(fc_batch)


# ---------------------------------------------------------------------------
# Cross-dataset threading (ComparisonParameters -> both delegate ctors)
# ---------------------------------------------------------------------------

def test_comparison_parameters_roundtrip():
    from comparison.comparison_parameters import ComparisonParameters
    params = ComparisonParameters(
        datasets=['ds_a', 'ds_b'], source_neurons=['x'],
        target_neurons=['y'], thresholds=[3],
        shortest_source_coverage=0.0, shortest_target_coverage=1.0,
        verbose=False)
    assert params.shortest_source_coverage == 0.0
    assert params.shortest_target_coverage == 1.0
    dumped = params.to_dict()
    assert dumped['shortest_source_coverage'] == 0.0
    assert dumped['shortest_target_coverage'] == 1.0


def test_delegate_constructors_receive_coverage_knobs(monkeypatch,
                                                       tmp_path):
    """Both FindNeuronConnection delegate constructions (run_path_analysis
    and the replay constructor) forward the knobs — kwargs stay identical
    across path modes."""
    from comparison.comparison_parameters import ComparisonParameters
    from comparison.comparison_analyzer import ComparisonAnalyzer
    import inspect
    src = inspect.getsource(ComparisonAnalyzer.run_path_analysis)
    assert 'shortest_source_coverage=' in src
    assert 'shortest_target_coverage=' in src
    # the replay/multi-threshold constructor too
    src2 = inspect.getsource(ComparisonAnalyzer)
    assert src2.count('shortest_target_coverage=') >= 2


def test_find_shortest_ui_payload_carries_coverage_levels():
    """The Find Shortest tab maps levels to engine fractions."""
    from ui.tabs import find_shortest
    src = inspect.getsource(find_shortest)
    assert '"shortest_source_coverage": _coverage_side_payload(' in src
    assert '"shortest_target_coverage": _coverage_side_payload(' in src
    assert '"Any", "25%", "50%", "75%", "Full",' in src
    assert '"Custom %"]' in src
    # defaults per the user: Any source + Full target
    assert '"Source Coverage", _COVERAGE_LEVELS, "Any"' in src
    assert '"Target Coverage", _COVERAGE_LEVELS, "Full"' in src


def test_inter_dataset_payload_gates_coverage_under_shortest():
    """The Inter-Dataset payload sends the fractions under shortest mode
    and None otherwise (the engine ignores them in 'all' mode)."""
    from ui.tabs import inter_dataset
    src = inspect.getsource(inter_dataset)
    assert '"shortest_source_coverage": (' in src
    assert "if path_mode.value == 'shortest' else None" in src
    # shortest-gated enable/disable
    assert 'shortest_source_coverage.set_enabled(_is_shortest)' in src
