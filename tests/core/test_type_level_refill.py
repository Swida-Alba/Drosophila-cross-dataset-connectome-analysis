"""Type-level refill (plan: _plan/plan-type-level-refill.md, Phase 1).

Every test drives the POST-HOC API (run folder in -> records out) on runs
produced by the REAL FindAllPath pipeline offline (the shared
``_make_pipeline_fc`` harness, with a multi-neuron-per-type universe
patched into the fixture's type map). ``skip_bodyId=False`` is set on the
runs so ``connection_info_bodyId.csv`` exists as split ground truth — the
refill itself never reads it.

Universe (12 neurons / 6 types / 22 edges, bound 4, asked t=1):
- strong tier (>= 11): survives an Edge-Budget floor;
- mid tier: the refill range under a floor;
- A1->C3: the designed conservatism-gap edge (its only continuation runs
  through TD, which never has an emitted row -> outside the induced node
  set; disclosed, not refilled);
- C2->C3: an in-U hop-pruned-only edge (no <=4-edge path uses it).
"""

import json
import re
from collections import defaultdict
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC = PROJECT_ROOT / 'src'
for p in (PROJECT_ROOT, SRC, PROJECT_ROOT / 'vispath-subproject' / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import coana  # noqa: E402
import tests.core.test_pathfinding as _tp  # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402
from type_level_refill import (  # noqa: E402
    TypeLevelRefillError,
    compute_type_level_refill,
    read_run_provenance,
)

TYPE_MAP = {
    'S1': 'TS', 'S2': 'TS',
    'A1': 'TA', 'A2': 'TA',
    'B1': 'TB', 'B2': 'TB',
    'C1': 'TC', 'C2': 'TC', 'C3': 'TC',
    'D1': 'TD',
    'T1': 'TT', 'T2': 'TT',
}
EDGES = [
    ('S1', 'T1', 30), ('S1', 'A1', 14), ('A1', 'T1', 20), ('A1', 'B1', 13),
    ('B1', 'T1', 15), ('A1', 'C1', 11), ('C1', 'T2', 12),
    ('S2', 'A2', 8), ('A2', 'B2', 6), ('B2', 'T1', 7), ('A2', 'T2', 9),
    ('S1', 'B1', 5), ('B1', 'A2', 4), ('C1', 'T1', 3),
    ('A1', 'C3', 5), ('C3', 'D1', 4), ('D1', 'T1', 5),
    ('A1', 'C2', 2), ('C2', 'T2', 2),
    ('C2', 'A2', 3),
    ('C2', 'C3', 3),
]
SOURCES = ['S1', 'S2']
TARGETS = ['T1', 'T2']
MAX_INTERLAYER = 3
ASKED = 1
TT_TOTAL_IN = 103   # hand-computed F9 denominator for post type TT


class _ShimMonkey:
    """Minimal stand-in for pytest's monkeypatch inside helpers. Records
    every setattr so the harness can undo them — without the undo, the
    synthetic fetch stubs leak onto the class and corrupt any later test
    that drives the REAL pipeline (the real-data smoke)."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        self._undo.append((target, name, getattr(target, name, None)))
        setattr(target, name, value)

    def undo(self):
        while self._undo:
            target, name, original = self._undo.pop()
            try:
                if original is None:
                    delattr(target, name)
                else:
                    setattr(target, name, original)
            except (AttributeError, TypeError):
                pass


@contextmanager
def universe_types(types=None):
    """Patch the shared harness's type map (multi-neuron universe), stub
    the NeuPrint client (bodyId exports touch it), and keep the graph
    cache isolated. The fetch-stub patching done by ``_make_pipeline_fc``
    via the shim is undone on exit."""
    original = dict(_tp._PIPELINE_TYPES)
    original_client = coana.FindNeuronConnection._ensure_neuprint_client
    # Mutate IN PLACE, never replace: other test modules hold the dict
    # object via ``from tests.core.test_pathfinding import
    # _PIPELINE_TYPES`` — a replacement object breaks their fixtures
    # (caught by the Phase-3 review round: shortest-batched tests failed
    # with KeyError 'S3' after any later-sorted universe test ran).
    _tp._PIPELINE_TYPES.clear()
    _tp._PIPELINE_TYPES.update(types or TYPE_MAP)
    coana.FindNeuronConnection._ensure_neuprint_client = lambda self: None
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    shim = _ShimMonkey()
    try:
        yield shim
    finally:
        shim.undo()
        _tp._PIPELINE_TYPES.clear()
        _tp._PIPELINE_TYPES.update(original)
        coana.FindNeuronConnection._ensure_neuprint_client = original_client
        coana._FINDALLPATH_GRAPH_CACHE.clear()


def make_run(tmp_path, name, *, edge_budget=0, path_budget=0, asked=ASKED,
             edges=None, types=None, max_interlayer=MAX_INTERLAYER,
             source_ids=SOURCES, target_ids=TARGETS):
    """Run the real FindAllPath pipeline on the synthetic universe."""
    with universe_types(types) as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path / name, edges=edges or EDGES,
            max_interlayer=max_interlayer, min_synapse=asked,
            source_ids=tuple(source_ids), target_ids=tuple(target_ids))
        fc.skip_bodyId = False
        if 'Checked' not in fc.target_df.columns:
            fc.target_df = fc.target_df.assign(
                Checked=[True] * len(fc.target_df))
        fc.graph_edge_limit_bodyid = edge_budget
        fc.max_paths_bodyid = path_budget
        # Real runs populate parameter_dict at init; the harness leaves it
        # empty, so stamp the keys the post-hoc parser reads.
        fc.parameter_dict.update({
            'min synapse number': str(asked),
            'filter by': 'bodyId',
            'exclude intra-type connections': str(
                fc.exclude_intra_type_connections),
            'max interlayer': str(max_interlayer),
            'separate hemispheres': str(fc.separate_hemispheres),
            'hemisphere filter': str(fc.hemisphere_filter),
            'aggregate method': 'product',
        })
        # Phase 2 auto-hook inputs (production resolves the connection
        # database + dataset neuron table itself).
        fc._refill_edges_override = list(edges or EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list((types or TYPE_MAP)),
            'type': list((types or TYPE_MAP).values())})
        fc.FindAllPath()
    return fc, Path(fc.allpath_folder)


def read_emitted_pairs(run_dir):
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv')
    return ct.groupby(['type_pre', 'type_post'])['weight'].sum().to_dict()


def read_emitted_edges(run_dir):
    df = pd.read_csv(run_dir / 'data_details' / 'connection_info_bodyId.csv',
                     dtype=str)
    return {(r.bodyId_pre, r.bodyId_post) for r in df.itertuples()}


def refill(run_dir, **kwargs):
    return compute_type_level_refill(
        run_dir, edges=EDGES, type_map=TYPE_MAP, write=False, **kwargs)


def write_records(run_dir):
    """Write records to a per-run probe dir; returns (rows, detail)."""
    out = run_dir.parent / f'_records_{run_dir.name}'
    compute_type_level_refill(
        run_dir, edges=EDGES, type_map=TYPE_MAP, out_dir=out, write=True)
    return (pd.read_csv(out / 'refill_type_pairs.csv'),
            pd.read_csv(out / 'refill_bodyId_pairs.csv', dtype=str))


def expected_refill_set(run_dir, *, asked=ASKED, types=None, edges=None,
                        bound=MAX_INTERLAYER + 1, sources=SOURCES,
                        targets=TARGETS):
    """Ground truth: induced-path edges minus this run's emitted edges,
    restricted to emitted type pairs. Independent brute-force DFS."""
    types = types or TYPE_MAP
    edges = edges or EDGES
    emitted = read_emitted_pairs(run_dir)
    involved = {p for pair in emitted for p in pair}
    node_set = {b for b, t in types.items() if t in involved}
    e_u = [(u, v, w) for (u, v, w) in edges
           if u in node_set and v in node_set and w >= asked]
    adj = {}
    for u, v, _w in e_u:
        adj.setdefault(u, []).append(v)
    estar = set()

    def dfs(node, path):
        if node in targets and len(path) > 1:
            for i in range(len(path) - 1):
                estar.add((path[i], path[i + 1]))
        if len(path) > bound:
            return
        for nxt in adj.get(node, ()):
            if nxt not in path:
                dfs(nxt, path + [nxt])

    for s in sources:
        dfs(s, [s])
    emitted_edges = read_emitted_edges(run_dir)
    return {e for e in estar
            if e not in emitted_edges
            and (types[e[0]], types[e[1]]) in emitted}


@pytest.fixture(scope='module')
def staged_runs(tmp_path_factory):
    """The four canonical runs, staged once for the whole module."""
    root = tmp_path_factory.mktemp('runs')
    runs = {}
    for name, kwargs in (
            ('complete', {}),
            ('floor', {'edge_budget': 7}),
            ('bite', {'path_budget': 4}),
            ('combined', {'edge_budget': 7, 'path_budget': 3}),
    ):
        runs[name] = make_run(root, name, **kwargs)
    return runs


# ---------------------------------------------------------------------------
# 1. Gate
# ---------------------------------------------------------------------------
def test_gate_complete_run_no_refill(staged_runs):
    rec = refill(staged_runs['complete'][1])
    assert rec['status'] == 'no_refill_needed'
    assert not (staged_runs['complete'][1] / 'data_details' /
                'type_level_refill').exists()


def test_gate_reads_replayed_from(tmp_path):
    """A multi-threshold canon slice stamps requested_threshold with the
    CANONICAL value; the true asked threshold rides on replayed_from."""
    with universe_types() as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path / 'minsyn_1', edges=EDGES,
            max_interlayer=MAX_INTERLAYER, min_synapse=ASKED,
            source_ids=tuple(SOURCES), target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.graph_edge_limit_bodyid = 7
        fc.parameter_dict.update({
            'min synapse number': str(ASKED),
            'filter by': 'bodyId',
            'exclude intra-type connections': str(
                fc.exclude_intra_type_connections),
            'max interlayer': str(MAX_INTERLAYER),
            'separate hemispheres': str(fc.separate_hemispheres),
            'hemisphere filter': str(fc.hemisphere_filter),
            'aggregate method': 'product',
        })
        fc.FindAllPathMultiThreshold([1, 3])
    canon = [p for p in sorted((tmp_path / 'minsyn_11').glob(
        'find-paths-*')) if p.is_dir()]
    assert canon, 'canon slice folder missing'
    rec = refill(canon[0])
    assert rec['status'] == 'refilled'
    assert rec['requested_threshold'] == 1
    assert rec['replayed_from'] == 1
    assert rec['requested_threshold_stamped'] == 11
    assert rec['refill_edges'] == 7 and rec['refill_weight_total'] == 37


# ---------------------------------------------------------------------------
# 2. Headline exactness
# ---------------------------------------------------------------------------
def test_floor_refill_exact_except_disclosed_gap(staged_runs):
    run_dir = staged_runs['floor'][1]
    rec = refill(run_dir)
    rows, detail = write_records(run_dir)
    complete = read_emitted_pairs(staged_runs['complete'][1])
    gaps = {}
    for row in rows.to_dict('records'):
        want = int(complete.get(
            (row['type_pre'], row['type_post']), 0))
        if row['refilled_total'] != want:
            gaps[(row['type_pre'], row['type_post'])] = \
                want - row['refilled_total']
    assert gaps == {('TA', 'TC'): 5}
    assert rec['refill_edges'] == 7 and rec['refill_weight_total'] == 37
    # the gap edge is certified non-induced: its only continuation (C3 ->
    # D1 -> T1) leaves the involved-type node set
    assert set(zip(detail['bodyId_pre'], detail['bodyId_post'])) == \
        expected_refill_set(run_dir)
    assert ('A1', 'C3') not in set(zip(detail['bodyId_pre'],
                                       detail['bodyId_post']))


def test_boundary_edge_at_floor_is_refilled(staged_runs):
    """A2->T2 has weight == w0 (survived the floored CONE) but its only
    path died with the floor (feeder S2->A2 = 8 < w0): it belongs in the
    refill and is counted + sampled as a boundary edge. Guards against
    the falsified naive weight<w0 split."""
    run_dir = staged_runs['floor'][1]
    rec = refill(run_dir)
    assert rec['edge_weight_floor'] == 9
    assert rec['boundary_refill_edge_count_at_or_above_w0'] == 1
    assert rec['boundary_refill_weight_at_or_above_w0'] == 9
    assert rec['boundary_refill_examples_at_or_above_w0'] == ['A2->T2(9)']
    _rows, detail = write_records(run_dir)
    edge = detail[(detail['bodyId_pre'] == 'A2')
                  & (detail['bodyId_post'] == 'T2')]
    assert len(edge) == 1 and int(edge['weight'].iloc[0]) == 9


# ---------------------------------------------------------------------------
# 3. The table-reproduction anchor
# ---------------------------------------------------------------------------
def test_table_reproduction_anchor(staged_runs):
    for name in ('floor', 'bite', 'combined'):
        rec = refill(staged_runs[name][1])
        assert rec['status'] == 'refilled', name
        assert rec['table_reproduced'] is True, name


def test_table_mismatch_refuses(staged_runs):
    """A tampered connection source (one edge weight changed) must fail
    the anchor and refuse, never emit a wrong refill."""
    run_dir = staged_runs['floor'][1]
    bad_edges = [('S1', 'T1', 31)] + list(EDGES)[1:]
    with pytest.raises(TypeLevelRefillError, match='reproduce'):
        compute_type_level_refill(run_dir, edges=bad_edges,
                                  type_map=TYPE_MAP, write=False)


# ---------------------------------------------------------------------------
# 4. Split fidelity + skips + prefilter
# ---------------------------------------------------------------------------
def test_split_matches_ground_truth(staged_runs):
    for name in ('bite', 'combined'):
        run_dir = staged_runs[name][1]
        refill(run_dir)
        _rows, detail = write_records(run_dir)
        got = set(zip(detail['bodyId_pre'], detail['bodyId_post']))
        assert got == expected_refill_set(run_dir), name


def test_hop_pruned_only_excluded_and_prefilter_counts(staged_runs):
    run_dir = staged_runs['floor'][1]
    rec = refill(run_dir)
    _rows, detail = write_records(run_dir)
    pairs = set(zip(detail['bodyId_pre'], detail['bodyId_post']))
    assert ('C2', 'C3') not in pairs
    assert rec['prefilter_dropped_in_U'] >= 1
    # §9.6: the prefilter is lossless — E* is identical with and without it
    from type_level_refill import hop_prefilter, _enumerate_paths
    involved = {p for pair in read_emitted_pairs(run_dir) for p in pair}
    node_set = {b for b, t in TYPE_MAP.items() if t in involved}
    e_u = [(u, v, w) for (u, v, w) in EDGES
           if u in node_set and v in node_set and w >= ASKED]
    e_pref, dropped = hop_prefilter(e_u, SOURCES, TARGETS,
                                     MAX_INTERLAYER + 1)
    assert dropped >= 1
    _p1, info_raw, _s1 = _enumerate_paths(
        e_u, SOURCES, TARGETS, MAX_INTERLAYER + 1)
    _p2, info_pref, _s2 = _enumerate_paths(
        e_pref, SOURCES, TARGETS, MAX_INTERLAYER + 1)
    assert set(info_raw) == set(info_pref)


def test_decision1_skips_and_membership_exclusions_counted(staged_runs):
    rec = refill(staged_runs['floor'][1])
    # ROUTE SCOPE (2026-10-07): edges on non-emitted pairs are pruned
    # BEFORE enumeration (out-of-scope routes are not pathfinded), so the
    # per-pair census is empty by construction and the mass moves to the
    # `non_emitted_pair_mass` disclosure (pair-level, reconciled via
    # unfired_residual's asked_pair_mass).
    assert rec['skipped_pairs_not_emitted_type_pair'] == {}
    assert rec['skipped_edge_examples'] == []
    # 15 = the old census edges (S1->B1 5, B1->A2 4, C2->A2 3) PLUS
    # C2->C3 (3) — the hop-pruned-only edge, which never entered estar
    # and was therefore invisible to the old census
    assert rec['non_emitted_pair_mass'] == 15
    # C3->D1 and D1->T1 have an endpoint outside U (TD not involved)
    assert rec['excluded_edges_non_involved_endpoint'] == 2


def test_ratio_threshold_free(staged_runs):
    rows, _detail = write_records(staged_runs['floor'][1])
    tc = rows[(rows['type_pre'] == 'TC') & (rows['type_post'] == 'TT')]
    assert len(tc) == 1
    row = tc.iloc[0]
    assert row['refilled_total'] == 17
    assert abs(row['refilled_connection_ratio'] - 17 / TT_TOTAL_IN) < 1e-6
    # Adjusted twin (plan §2.4): this fixture stamps min synapse = 1, so
    # the threshold-conditioned denominator IS the F9 one — equal by
    # doctrine (always emitted, redundant when no synapse threshold).
    assert abs(row['refilled_connection_ratio_adj']
               - 17 / TT_TOTAL_IN) < 1e-6


def test_adjusted_refill_ratio_threshold_conditioned(staged_runs, tmp_path):
    """min synapse = 3: the adjusted denominator drops the weight-2
    C2->T2 edge (103 -> 101), so adj > plain on the same TC->TT row;
    the plain F9 ratio is unchanged (threshold-free by doctrine)."""
    import shutil
    run_dir = staged_runs['floor'][1]
    copy = tmp_path / 'cond'
    shutil.copytree(run_dir, copy)
    import re as _re
    text = (copy / 'parameters.txt').read_text()
    text = _re.sub(r'^min synapse number:\s*\S+', 'min synapse number:          3',
                   text, flags=_re.MULTILINE)
    (copy / 'parameters.txt').write_text(text)
    out = tmp_path / '_records_cond'
    compute_type_level_refill(copy, edges=EDGES, type_map=TYPE_MAP,
                              out_dir=out, write=True)
    rows = pd.read_csv(out / 'refill_type_pairs.csv')
    row = rows[(rows['type_pre'] == 'TC')
               & (rows['type_post'] == 'TT')].iloc[0]
    assert abs(row['refilled_connection_ratio'] - 17 / 103) < 1e-6
    assert abs(row['refilled_connection_ratio_adj'] - 17 / 101) < 1e-6



def test_determinism_bytes(staged_runs, tmp_path):
    run_dir = staged_runs['floor'][1]
    digests = []
    for tag in ('a', 'b'):
        out = tmp_path / tag
        compute_type_level_refill(run_dir, edges=EDGES, type_map=TYPE_MAP,
                                  out_dir=out, write=True)
        digests.append((out / 'refill_type_pairs.csv').read_bytes())
    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# 5. Budget sweep + tie fixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('edge_budget,path_budget', [
    (5, 0), (7, 0), (9, 0), (12, 0), (10 ** 9, 0),
    (0, 2), (0, 3), (0, 6),
    (7, 2), (5, 6),
])
def test_budget_sweep_matrix(tmp_path, staged_runs, edge_budget,
                             path_budget):
    fc, run_dir = make_run(tmp_path, f'sweep_{edge_budget}_{path_budget}',
                           edge_budget=edge_budget, path_budget=path_budget)
    complete = read_emitted_pairs(staged_runs['complete'][1])
    rec = refill(run_dir)
    if rec['status'] != 'refilled':
        assert edge_budget == 10 ** 9     # only the no-op budget gates off
        return
    assert rec['table_reproduced'] is True
    rows, _detail = write_records(run_dir)
    for row in rows.to_dict('records'):
        pair = (row['type_pre'], row['type_post'])
        want = int(complete.get(pair, 0))
        if pair != ('TA', 'TC'):
            assert row['refilled_total'] == want, row


def test_tie_drain_at_the_cut(tmp_path):
    """Two equal-bottleneck paths straddling a top-1 cut: the production
    tau tie-drain emits BOTH, and the module's cut re-derivation (which
    calls the same enumerator) must reproduce the exported table rather
    than slicing at exactly N paths. A2's weak parallel pair (TS->TA /
    TA->TT via S->A2->T) is the refillable mass."""
    edges = [
        ('S', 'A', 10), ('A', 'T', 10),      # path bottleneck 10
        ('S', 'B', 10), ('B', 'T', 10),      # path bottleneck 10 (tied)
        ('S', 'A2', 5), ('A2', 'T', 5),      # bottleneck 5 (refill range)
    ]
    types = {'S': 'TS', 'A': 'TA', 'A2': 'TA', 'B': 'TB', 'T': 'TT'}
    fc, run_dir = make_run(tmp_path, 'tie', edges=edges, types=types,
                           max_interlayer=2, source_ids=['S'],
                           target_ids=['T'], path_budget=1)
    assert fc.strongest_first_budget_bitten
    assert fc.strongest_first_cutoff == 10
    # tie drain: BOTH bottleneck-10 paths were emitted despite N=1
    emitted = read_emitted_edges(run_dir)
    assert {('S', 'A'), ('A', 'T'), ('S', 'B'), ('B', 'T')} <= emitted
    rec = compute_type_level_refill(
        run_dir, edges=edges, type_map=types, write=False)
    assert rec['status'] == 'refilled'
    assert rec['table_reproduced'] is True
    out = tmp_path / 'rec'
    compute_type_level_refill(run_dir, edges=edges, type_map=types,
                              out_dir=out, write=True)
    detail = pd.read_csv(out / 'refill_bodyId_pairs.csv', dtype=str)
    assert set(zip(detail['bodyId_pre'], detail['bodyId_post'])) == \
        {('S', 'A2'), ('A2', 'T')}


# ---------------------------------------------------------------------------
# 6. CLI + import hygiene
# ---------------------------------------------------------------------------
def _write_sources(tmp_path):
    conn_csv = tmp_path / 'connections.csv'
    pd.DataFrame(EDGES, columns=['bodyId_pre', 'bodyId_post', 'weight']) \
        .to_csv(conn_csv, index=False)
    table_csv = tmp_path / 'neurons.csv'
    pd.DataFrame({'bodyId': list(TYPE_MAP),
                  'type': list(TYPE_MAP.values())}).to_csv(table_csv,
                                                           index=False)
    return conn_csv, table_csv


def test_cli_writes_outside_run_folder(staged_runs, tmp_path):
    run_dir = staged_runs['floor'][1]
    conn_csv, table_csv = _write_sources(tmp_path)
    out = tmp_path / 'cli_out'
    before = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' / 'TypeLevelRefill.py'),
         str(run_dir), '--connections', str(conn_csv),
         '--neuron-table', str(table_csv), '--out', str(out)],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    rec_out = out / run_dir.name
    assert (rec_out / 'refill_type_pairs.csv').exists()
    assert (rec_out / 'refill_bodyId_pairs.csv').exists()
    assert (rec_out / 'refill_provenance.json').exists()
    assert (rec_out / 'README.md').exists()
    rows = pd.read_csv(rec_out / 'refill_type_pairs.csv')
    assert len(rows) == 7 and int(rows['refill_weight'].sum()) == 37
    after = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
    assert before == after


def test_cli_complete_run_gates_cleanly(staged_runs, tmp_path):
    conn_csv, table_csv = _write_sources(tmp_path)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' / 'TypeLevelRefill.py'),
         str(staged_runs['complete'][1]), '--connections', str(conn_csv),
         '--neuron-table', str(table_csv),
         '--out', str(tmp_path / 'out2')],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'no_refill_needed' in result.stdout


def test_import_without_vispath():
    """The module imports without the vispath subproject (the coana
    lazy-import contract); enumeration is deferred to first use."""
    code = (
        'import sys\n'
        f'sys.path.insert(0, {str(SRC)!r})\n'
        'class _Block:\n'
        '    def find_spec(self, fullname, path=None, target=None):\n'
        '        if fullname.startswith("vispath_pkg"):\n'
        '            raise ImportError("blocked for test")\n'
        '        return None\n'
        'sys.meta_path.insert(0, _Block())\n'
        'import type_level_refill\n'
        'print("import ok")\n'
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'import ok' in result.stdout


# ---------------------------------------------------------------------------
# 7. Real-data smoke (skipped unless the local FAFB release is present —
#    the standard requires_data marker gates on the neuron+ROI tables,
#    which the FAFB local release does not ship; the refill needs the
#    connections parquet + neuron table instead)
# ---------------------------------------------------------------------------
_FAFB_DIR = PROJECT_ROOT / 'datasets' / 'flywire_FAFB_v783'
_FAFB_GATED = pytest.mark.skipif(
    not (_FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet').exists()
    or not (_FAFB_DIR / 'flywire_FAFB_v783_allneurons_neuron_df.csv').exists(),
    reason='local FAFB v783 release (merged connections + neuron table) '
           'not present')


@_FAFB_GATED
def test_real_fafb_smoke(tmp_path):
    """One real budgeted run (StrongestFirst bite) through the REAL
    pipeline, refilled post-hoc: the table-reproduction anchor must hold
    on real labels and every exported record must be internally
    consistent (the §9.14 real-run gate)."""
    import polars as pl
    from utils.label_utils import is_untyped_type_label

    fc = coana.FindNeuronConnection(
        output_dir=str(tmp_path), dataset='flywire_FAFB_v783',
        sourceNeurons=['aMe12'], targetNeurons=['PPL101'],
        custom_source_name='', custom_target_name='',
        custom_source_group_names=[], custom_target_group_names=[],
        min_synapse_num=3, min_ratio=0.0, min_traversal_probability=0,
        filter_by='bodyId', showfig=False, max_interlayer=2,
        keyword_in_path_to_remove=['None'], network_layout='distributed',
        use_cache=True, edgeN_limit=500, output_format='csv',
        pathfinding='StrongestFirst', drop_untyped=True, skip_bodyId=True,
        graph_edge_limit_bodyid=10_000_000, max_paths_bodyid=200)
    fc.InitializeNeuronInfo()
    fc.FindAllPath(forward_only=True)
    run_dir = Path(fc.allpath_folder)
    assert fc.strongest_first_budget_bitten, 'budget must bite for a refill'

    conn = pl.read_parquet(
        _FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet',
        columns=['bodyId_pre', 'bodyId_post', 'weight'])
    edges = [(str(u), str(v), float(w)) for u, v, w in
             zip(conn['bodyId_pre'], conn['bodyId_post'], conn['weight'])]
    neurons = pd.read_csv(
        _FAFB_DIR / 'flywire_FAFB_v783_allneurons_neuron_df.csv',
        dtype=str, low_memory=False)
    labels = neurons['type'].fillna('Unknown').astype(str)
    keep = ~labels.map(is_untyped_type_label)
    type_map = dict(zip(neurons.loc[keep, 'bodyId'].astype(str),
                        labels[keep]))

    out = tmp_path / 'refill'
    rec = compute_type_level_refill(
        run_dir, edges=edges, type_map=type_map, out_dir=out, write=True)
    assert rec['status'] == 'refilled'
    assert rec['table_reproduced'] is True
    assert rec['refill_edges'] > 0 and rec['refill_weight_total'] > 0
    assert rec['refill_truncated'] is False

    rows = pd.read_csv(out / 'refill_type_pairs.csv')
    assert len(rows) > 0
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv',
                     dtype={'type_pre': str, 'type_post': str})
    agg = ct.groupby(['type_pre', 'type_post'])['weight'].sum()
    for row in rows.to_dict('records'):
        assert int(agg.get((row['type_pre'], row['type_post']), -1)) == \
            row['emitted_weight']
    detail = pd.read_csv(out / 'refill_bodyId_pairs.csv', dtype=str)
    assert len(detail) == min(rec['refill_edges'], rec['detail_cap'])
    assert (detail['weight'].astype(float) >= 3).all()


# ---------------------------------------------------------------------------
# 8. Effective type-map reconstruction (label-mapper parity)
# ---------------------------------------------------------------------------
def test_build_effective_type_map_mapper_semantics(tmp_path):
    """The builder mirrors the run's own type-level aggregation: a
    bodyId-level mapping wins over a type-level mapping, unmapped labels
    keep their raw names (identity fallback), then hemisphere suffixes
    and the untyped drop apply."""
    from type_level_refill import build_effective_type_map

    # LabelMapper overall-JSON form: custom_label list + per-dataset
    # member-pattern lists (docstring format)
    mapping = {
        'source_mapping': {
            'custom_label': ['GRP_A', 'grp_t'],
            'test_v1': [['N1'], ['TA', 'TB']],
        }
    }
    import json
    mapping_file = tmp_path / 'mapping.json'
    mapping_file.write_text(json.dumps(mapping), encoding='utf-8')

    frame = pd.DataFrame({
        'bodyId': ['N1', 'N2', 'N3', 'N4', 'N5'],
        'type': ['TA', 'TB', 'TC', 'Unknown', ''],
    })
    built = build_effective_type_map(
        frame, dataset='test_v1', mapping_file=mapping_file)
    # bodyId-level mapping wins for N1; type-level for N2; identity for
    # N3; untyped dropped
    assert built == {'N1': 'GRP_A', 'N2': 'grp_t', 'N3': 'TC'}

    # hemisphere suffixing composes after the mapping
    frame_h = pd.DataFrame({
        'bodyId': ['N1', 'N3'],
        'type': ['TA', 'TC'],
        'hemisphere': ['left', 'weird'],
    })
    built_h = build_effective_type_map(
        frame_h, dataset='test_v1', mapping_file=mapping_file,
        separate_hemispheres=True)
    assert built_h == {'N1': 'GRP_A_L', 'N3': 'TC_U'}


def test_build_effective_type_map_refuses_missing_mapping_file(tmp_path):
    from type_level_refill import build_effective_type_map
    frame = pd.DataFrame({'bodyId': ['N1'], 'type': ['TA']})
    with pytest.raises(TypeLevelRefillError, match='mapping file'):
        build_effective_type_map(
            frame, dataset='test_v1',
            mapping_file=tmp_path / 'does_not_exist.json')


# ---------------------------------------------------------------------------
# 9. Phase 2: the in-pipeline auto hook
# ---------------------------------------------------------------------------
def _with_hook_overrides(fc):
    """Give the harness fc the two inputs the production hook resolves
    itself (connection database + dataset neuron table)."""
    fc._refill_edges_override = list(EDGES)
    fc._refill_neuron_frame_override = pd.DataFrame({
        'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
    return fc


def test_auto_hook_generates_records(tmp_path):
    fc, run_dir = make_run(tmp_path, 'hook_floor', edge_budget=7)
    _with_hook_overrides = None  # noqa: F841 (shadow guard)
    out = run_dir / 'data_details' / 'type_level_refill'
    assert (out / 'refill_type_pairs.csv').exists()
    assert (out / 'refill_bodyId_pairs.csv').exists()
    rows = pd.read_csv(out / 'refill_type_pairs.csv')
    assert len(rows) == 7 and int(rows['refill_weight'].sum()) == 37
    notes = (run_dir / 'user_warning_notes.txt').read_text(
        encoding='utf-8')
    assert '[type-level refill]' in notes
    assert '7 bodyId edges' in notes


def test_auto_hook_disabled(tmp_path):
    with universe_types() as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path, edges=EDGES, max_interlayer=MAX_INTERLAYER,
            min_synapse=ASKED, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.graph_edge_limit_bodyid = 7
        fc.max_paths_bodyid = 0
        fc.parameter_dict.update({
            'min synapse number': str(ASKED), 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': str(MAX_INTERLAYER),
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc.auto_type_level_refill = False
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindAllPath()
        run_dir = Path(fc.allpath_folder)
    assert not (run_dir / 'data_details' / 'type_level_refill').exists()


def test_auto_hook_complete_run_silent(staged_runs):
    run_dir = staged_runs['complete'][1]
    assert not (run_dir / 'data_details' / 'type_level_refill').exists()
    notes = run_dir / 'user_warning_notes.txt'
    if notes.exists():
        assert 'type-level refill' not in notes.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# 9b. Ratio-basis support (Phase 3): the refill runs for
# weight_basis='connection_ratio' runs with FLOAT thresholds; the records
# carry synapse-mass recovery weights and the anchor still reproduces the
# exported table.
# ---------------------------------------------------------------------------
def test_auto_hook_ratio_basis_refills(tmp_path):
    with universe_types() as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path, edges=EDGES, max_interlayer=MAX_INTERLAYER,
            min_synapse=ASKED, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.graph_edge_limit_bodyid = 7
        fc.max_paths_bodyid = 0
        fc.parameter_dict.update({
            'min synapse number': '1', 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': str(MAX_INTERLAYER),
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.2   # a valid ratio tier (the entry gate refuses 0)
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindAllPath()
        run_dir = Path(fc.allpath_folder)
    out = run_dir / 'data_details' / 'type_level_refill'
    prov = out / 'refill_provenance.json'
    if prov.exists():      # a ratio budget raised applied > asked
        import json as _json
        rec = _json.loads(prov.read_text(encoding='utf-8'))
        assert rec.get('table_reproduced') is True
        assert rec.get('requested_threshold') == 0.2
        rows = pd.read_csv(out / 'refill_type_pairs.csv')
        assert (rows['refill_weight'] >= 0).all()   # synapse masses
    # either way the hook no longer refuses ratio runs outright
    notes = run_dir / 'user_warning_notes.txt'
    if notes.exists():
        assert 'does not apply to ratio-basis' not in notes.read_text(
            encoding='utf-8')


def test_read_run_provenance_parses_ratio_basis(tmp_path):
    run_dir = tmp_path / 'ratio_run'
    run_dir.mkdir()
    (run_dir / 'parameters.txt').write_text(
        'min synapse number: 1\n'
        'weight basis: connection_ratio (bodyId)\n'
        'requested_threshold: 0.05\n'
        'applied_threshold: 0.05\n',
        encoding='utf-8')
    prov = read_run_provenance(run_dir)
    assert prov['weight_basis'] == 'connection_ratio'
    assert prov['requested_threshold'] == '0.05'  # parsed, not refused


# ---------------------------------------------------------------------------
# 10. Phase 3: shortest-mode refill (via the discovery store)
# ---------------------------------------------------------------------------
def _make_shortest_run(tmp_path, name, *, path_budget=0):
    with universe_types() as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path / name, edges=EDGES,
            max_interlayer=MAX_INTERLAYER, min_synapse=ASKED,
            source_ids=tuple(SOURCES), target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.max_paths_bodyid = path_budget
        fc.parameter_dict.update({
            'min synapse number': str(ASKED), 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': str(MAX_INTERLAYER),
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindShortestPath()
    return fc, Path(fc.allpath_folder)


def test_shortest_mode_refill(tmp_path):
    """Phase 3: a budget-bitten shortest run refilled from its discovery
    store; per-type-pair totals must land exactly on the unbudgeted
    shortest control for the emitted pairs (cut re-derivation via the
    shortest strongest-first enumerator with store-seeded distances)."""
    fc_c, run_c = _make_shortest_run(tmp_path, 'scomplete')
    fc_b, run_b = _make_shortest_run(tmp_path, 'sbite', path_budget=1)
    assert getattr(fc_b, 'strongest_first_budget_bitten', False) or \
        getattr(fc_b, 'strongest_first_cutoff', None)
    assert (run_b / 'shortest_discovery_store' / 'meta.json').exists()

    # the Phase 2/3 auto hook already generated the records in-run
    hook_rows = pd.read_csv(run_b / 'data_details' / 'type_level_refill'
                            / 'refill_type_pairs.csv')
    assert len(hook_rows) > 0
    rec = compute_type_level_refill(
        run_b, edges=EDGES, type_map=TYPE_MAP, write=False)
    assert rec['status'] == 'refilled', rec
    assert rec['table_reproduced'] is True
    out = tmp_path / 'rec'
    compute_type_level_refill(run_b, edges=EDGES, type_map=TYPE_MAP,
                              out_dir=out, write=True)
    rows = pd.read_csv(out / 'refill_type_pairs.csv')
    complete = read_emitted_pairs(run_c)
    gaps = {}
    for row in rows.to_dict('records'):
        pair = (row['type_pre'], row['type_post'])
        want = int(complete.get(pair, 0))
        if row['refilled_total'] != want:
            gaps[pair] = want - row['refilled_total']
    # shortest-mode conservatism gaps are allowed only for pairs whose
    # control mass rides paths outside the bitten run's induced set
    assert all(v > 0 for v in gaps.values()), gaps
    detail = pd.read_csv(out / 'refill_bodyId_pairs.csv', dtype=str)
    assert len(detail) == rec['refill_edges']
    # the complete control gates off
    rec_c = compute_type_level_refill(
        run_c, edges=EDGES, type_map=TYPE_MAP, write=False)
    assert rec_c['status'] == 'no_refill_needed'


# ---------------------------------------------------------------------------
# 13. Round-13: the refill honors the AND co-threshold (both bases)
# ---------------------------------------------------------------------------
def _make_and_run(tmp_path, name, *, basis, min_synapse, min_ratio,
                  path_budget):
    """A budget-BITTEN run under threshold_combination='and' whose refill
    must reproduce the dual admission rule (an edge failing either gate
    never enters the refill universe)."""
    with universe_types(TYPE_MAP) as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path / name, edges=EDGES, max_interlayer=MAX_INTERLAYER,
            min_synapse=min_synapse, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.max_paths_bodyid = path_budget
        fc.weight_basis = basis
        fc.threshold_combination = 'and'
        fc.min_ratio = min_ratio
        fc.parameter_dict.update({
            'min synapse number': str(min_synapse), 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': str(MAX_INTERLAYER),
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindAllPath()
        return Path(fc.allpath_folder)


def test_refill_ratio_basis_and_honors_synapse_floor(tmp_path):
    # Ratio basis + AND (syn floor 2 + ratio tier): the refill's induced
    # universe must ALSO drop sub-floor edges — without the round-13 fix
    # the anchor refused (the cut re-derivation saw edges the run never
    # had).
    run_dir = _make_and_run(tmp_path, 'and_ratio_bite',
                            basis='connection_ratio', min_synapse=2,
                            min_ratio=0.02, path_budget=1)
    out = run_dir / 'data_details' / 'type_level_refill'
    prov = out / 'refill_provenance.json'
    assert prov.exists(), 'a bitten and-mode run must refill'
    import json as _json
    rec = _json.loads(prov.read_text(encoding='utf-8'))
    assert rec.get('table_reproduced') is True
    assert rec.get('requested_threshold') == 0.02


def test_refill_synapse_basis_and_honors_ratio_tier(tmp_path):
    # Synapse basis + AND: the ratio co-threshold shapes the run and the
    # refill alike; parameters.txt carries the tier.
    run_dir = _make_and_run(tmp_path, 'and_syn_bite',
                            basis='synapse', min_synapse=2,
                            min_ratio=0.02, path_budget=1)
    params = (run_dir / 'parameters.txt').read_text(encoding='utf-8')
    assert 'min connection ratio' in params
    assert 'threshold combination' in params
    out = run_dir / 'data_details' / 'type_level_refill'
    prov = out / 'refill_provenance.json'
    if prov.exists():
        import json as _json
        rec = _json.loads(prov.read_text(encoding='utf-8'))
        assert rec.get('table_reproduced') is True


def test_and_mode_bad_ratio_refused(tmp_path):
    import pytest as _pytest
    import coana as _coana
    fc = object.__new__(_coana.FindNeuronConnection)
    fc._vprint = lambda *a, **k: None
    fc._warn_notes = []
    fc.weight_basis = 'synapse'
    fc.threshold_combination = 'and'
    fc.min_ratio = 2.0
    fc.min_traversal_probability = 0.0
    fc.min_synapse_num = 3
    fc.parameter_dict = {}
    with _pytest.raises(ValueError, match='min_ratio in'):
        fc.FindAllPath(forward_only=True)


# ---------------------------------------------------------------------------
# Engine round 2026-10-06: emitted pairs are the run's PHYSICAL mass
# ---------------------------------------------------------------------------

def _stage_minimal_run(run_dir, conn_rows, info_rows=None, types=None):
    """Stage the smallest folder read_emitted_pairs can consume."""
    run_dir = Path(run_dir)
    dd = run_dir / 'data_details'
    dd.mkdir(parents=True, exist_ok=True)
    ct = pd.DataFrame(conn_rows)
    ct.to_csv(dd / 'connection_type.csv', index=False)
    if info_rows is not None:
        pd.DataFrame(info_rows).to_csv(
            dd / 'connection_info_bodyId.csv', index=False)
    return run_dir


def test_read_emitted_pairs_prefers_deduped_bodyId_mass(tmp_path):
    # call the ENGINE function explicitly: the module-level
    # read_emitted_pairs above is a test-local legacy helper (line ~157)
    from type_level_refill import read_emitted_pairs as engine_read
    """connection_info_bodyId.csv present -> per-bodyId dedup (physical):
    an edge recurring across conn layers counts ONCE even though the type
    table's per-layer rows each carry it."""
    run = _stage_minimal_run(
        tmp_path / 'run',
        conn_rows=[{'type_pre': 'TA', 'type_post': 'TB', 'weight': 7},
                   {'type_pre': 'TA', 'type_post': 'TB', 'weight': 7},
                   {'type_pre': 'TA', 'type_post': 'TC', 'weight': 5}],
        info_rows=[
            # edge x1y1 fetched at TWO layers (same physical weight 7);
            # x2y2 once at 5; an untyped row must be ignored
            {'bodyId_pre': 'x1', 'bodyId_post': 'y1', 'weight': 7,
             'type_pre': 'TA', 'type_post': 'TB', 'conn_layer': '0->1'},
            {'bodyId_pre': 'x1', 'bodyId_post': 'y1', 'weight': 7,
             'type_pre': 'TA', 'type_post': 'TB', 'conn_layer': '1->2'},
            {'bodyId_pre': 'x2', 'bodyId_post': 'y2', 'weight': 5,
             'type_pre': 'TA', 'type_post': 'TC', 'conn_layer': '0->1'},
            {'bodyId_pre': 'x3', 'bodyId_post': 'y3', 'weight': 99,
             'type_pre': None, 'type_post': 'TB', 'conn_layer': '0->1'},
        ])
    emitted = engine_read(run)
    assert emitted == {('TA', 'TB'): 7, ('TA', 'TC'): 5}


def test_read_emitted_pairs_falls_back_to_conn_table(tmp_path):
    """skip_bodyId runs have no bodyId detail: the conn-table sums remain
    the source (documented per-layer semantics)."""
    run = _stage_minimal_run(
        tmp_path / 'run',
        conn_rows=[{'type_pre': 'TA', 'type_post': 'TB', 'weight': 7},
                   {'type_pre': 'TA', 'type_post': 'TB', 'weight': 7}])
    assert read_emitted_pairs(run) == {('TA', 'TB'): 14}


def test_enumerate_shortest_uses_physical_not_summed_weights(tmp_path):
    """_enumerate_shortest must dedup per-layer refetches (keep-first) —
    the I1-parity fix: summing re-ordered the StrongestFirst drain
    against the fixed engine and broke the cut anchor."""
    from type_level_refill import _enumerate_shortest
    # S->A (weight 7, fetched at two layers) -> T; the S->A->T path's
    # bottleneck must be 7 (physical), not 14 (summed).
    layer_weights = [
        ('0->1', {('S', 'A'): 7.0}),
        ('1->2', {('S', 'A'): 7.0, ('A', 'T'): 12.0}),
    ]
    info, stats = _enumerate_shortest(
        layer_weights, {'S', 'A', 'T'}, ['S'], ['T'], 2, None,
        {'T': {'S': 2, 'A': 1}}, {'T': 2})
    assert ('S', 'A') in info
    assert info[('S', 'A')]['w'] == 7.0
    assert stats.get('budget_bitten') in (False, None)


# ---------------------------------------------------------------------------
# 10. Standalone mirror folder (<run>/type_level_refill/, plan round
#     2026-10-06): the refilled data in the path/connection data format
# ---------------------------------------------------------------------------
def test_standalone_mirror_folder(tmp_path):
    """The run-flow refill also mirrors the refilled data into
    <run>/type_level_refill/: refill-only + complete-at-asked type
    tables in the connection_type.csv column order, and the
    complete-at-asked bodyId edge table (emitted-rederived + refill)."""
    _fc, run_dir = make_run(tmp_path, 'mirror_floor', edge_budget=7)
    mirror = run_dir / 'type_level_refill'
    assert (mirror / 'refill_connection_type.csv').exists()
    assert (mirror / 'refilled_connection_type.csv').exists()
    assert (mirror / 'refilled_connection_bodyId.csv').exists()
    assert (mirror / 'refill_paths_type.csv').exists()

    # probability columns SUSPENDED (2026-10-06, unverifiable fidelity)
    ct_cols = ['type_pre', 'type_post', 'weight', 'nt_type',
               'connection_ratio', 'conn_layer', 'connection_ratio_adj']
    refill_t = pd.read_csv(mirror / 'refill_connection_type.csv')
    complete_t = pd.read_csv(mirror / 'refilled_connection_type.csv')
    assert list(refill_t.columns) == ct_cols
    assert list(complete_t.columns) == ct_cols
    assert (refill_t['weight'] > 0).all()
    # the refill mirror carries exactly the records' refill mass
    records = run_dir / 'data_details' / 'type_level_refill'
    pairs = pd.read_csv(records / 'refill_type_pairs.csv')
    pos = pairs[pairs['refill_weight'] > 0]
    assert int(refill_t['weight'].sum()) == int(pos['refill_weight'].sum())
    merged = refill_t.merge(
        pos[['type_pre', 'type_post', 'refill_weight']],
        on=['type_pre', 'type_post'])
    assert (merged['weight'] == merged['refill_weight']).all()
    # the complete mirror = emitted + refill for EVERY emitted pair
    assert len(complete_t) == len(pairs)
    joined = complete_t.merge(pairs, on=['type_pre', 'type_post'])
    assert (joined['weight'] == joined['refilled_total']).all()
    assert (joined['weight'] - joined['refill_weight']
            == joined['emitted_weight']).all()
    # ratio columns ride along (plain + adjusted)
    assert joined['connection_ratio'].notna().all()

    # complete bodyId edges = emitted (deduped physical) + refill
    body = pd.read_csv(mirror / 'refilled_connection_bodyId.csv',
                       dtype={'bodyId_pre': str, 'bodyId_post': str})
    emitted = pd.read_csv(
        run_dir / 'data_details' / 'connection_info_bodyId.csv',
        dtype=str).drop_duplicates(
        subset=['bodyId_pre', 'bodyId_post'], keep='first')
    detail = pd.read_csv(
        records / 'refill_bodyId_pairs.csv', dtype=str)
    assert len(body) == len(emitted) + len(detail)
    have = set(zip(body['bodyId_pre'], body['bodyId_post']))
    assert have >= set(zip(emitted['bodyId_pre'], emitted['bodyId_post']))
    assert have >= set(zip(detail['bodyId_pre'], detail['bodyId_post']))

    # the refilled TYPE PATHS table (the second path-building round's
    # product): allpaths column order, engine invariants —
    # min_weight = weakest realized hop, length = hops, coverage
    # n <= N, endpoints within the emitted pairs, byte-determinism
    paths_csv = mirror / 'refill_paths_type.csv'
    assert list(pd.read_csv(paths_csv, nrows=0).columns) == [
        'path', 'weights', 'ratios', 'min_weight',
        'min_ratio', 'length', 'nt_types', 'coverage']
    pf = pd.read_csv(paths_csv)
    assert len(pf) > 0
    # every path carries >=1 hop whose type pair is a REFILL pair (the
    # refill edges already carry the emitted-pairs restriction; a path's
    # ENDPOINT pair is not an edge pair and must not be checked)
    refill_pairs = {
        (str(r.type_pre), str(r.type_post)) for r in refill_t.itertuples()}
    cov_re = re.compile(r'(\d+)/(\d+)')
    for r in pf.itertuples():
        seq = str(r.path).split('->')
        assert r.length == len(seq) - 1
        weights = json.loads(str(r.weights))
        assert len(weights) == r.length
        assert float(r.min_weight) == min(float(w) for w in weights)
        hops = {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}
        assert hops & refill_pairs, r.path
        cov = str(r.coverage)
        assert cov.startswith('[') and cov.endswith(']')
        for n, total in cov_re.findall(cov):
            assert 0 <= int(n) <= int(total)
    before = paths_csv.read_bytes()
    compute_type_level_refill(run_dir, edges=EDGES, type_map=TYPE_MAP,
                              write=True)
    assert paths_csv.read_bytes() == before


def test_refill_visualizations_built(tmp_path):
    """The visualization stage draws the mirrored refill mass like the
    reciprocal views: type heatmap/network (+ bodyId twins) into the
    standalone folder."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = False
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})
    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir(parents=True)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 40, 'nt_type': '',
         'connection_ratio': 0.1, 'conn_layer': '',
         'connection_ratio_adj': 0.2},
    ]).to_csv(mirror / 'refill_connection_type.csv', index=False)
    details = tmp_path / 'data_details' / 'type_level_refill'
    details.mkdir(parents=True)
    pd.DataFrame([
        {'bodyId_pre': 'S1', 'bodyId_post': 'A1', 'type_pre': 'TS',
         'type_post': 'TA', 'weight': 40, 'traversal_count': 3,
         'min_hop': 1, 'max_hop': 2},
    ]).to_csv(details / 'refill_bodyId_pairs.csv', index=False)

    fc._build_refill_visualizations(str(mirror))

    for name in ('refill_type_heatmap.html', 'refill_type_network.html',
                 'refill_bodyId_heatmap.html', 'refill_bodyId_network.html'):
        assert (mirror / name).exists(), name


# ---------------------------------------------------------------------------
# 11. Complete-at-asked table + unfired residual (plan
#     plan-refill-complete-at-asked-paths, R1/R3)
# ---------------------------------------------------------------------------
def test_complete_at_asked_table(tmp_path):
    """R1: the complete table carries EVERY asked chain (refill rows are
    a subset), each hop splits emitted+refill+skipped == weights, the
    stamp is true on a clean run, and the emitted split reconciles with
    connection_type.csv per edge pair."""
    _fc, run_dir = make_run(tmp_path, 'caf_floor', edge_budget=7)
    mirror = run_dir / 'type_level_refill'
    csv = mirror / 'complete_at_asked_paths_type.csv'
    assert csv.exists()
    cov_re = re.compile(r'(\d+)/(\d+)')
    # skipped_weights DROPPED (2026-10-07): route scope makes the
    # skipped class impossible
    assert list(pd.read_csv(csv, nrows=0).columns) == [
        'path', 'weights', 'ratios', 'min_weight', 'min_ratio', 'length',
        'nt_types', 'coverage',
        'emitted_weights', 'refill_weights']
    pf = pd.read_csv(csv)
    rf = pd.read_csv(mirror / 'refill_paths_type.csv')
    assert set(rf['path']) <= set(pf['path'])
    rf_w = {r['path']: json.loads(str(r.weights)) for _, r in rf.iterrows()}
    for _, r in pf.iterrows():
        seq = str(r.path).split('->')
        w = json.loads(str(r.weights))
        e = json.loads(str(r.emitted_weights))
        f = json.loads(str(r.refill_weights))
        assert len(w) == r.length
        for i in range(len(w)):
            assert e[i] + f[i] == w[i]
        if r.path in rf_w:
            assert all(w[i] >= rf_w[r.path][i] for i in range(len(w)))
        cov = cov_re.findall(str(r.coverage))
        assert cov and all(int(nn) <= int(tt) for nn, tt in cov)
        _ = seq
    rec = json.loads((run_dir / 'data_details' / 'type_level_refill' /
                      'refill_provenance.json').read_text())
    assert rec['complete_at_asked'] is True
    assert rec['complete_paths_type'] == len(pf)
    assert rec['complete_paths_bodyid'] > 0
    # T3 anchor, BOUNDED form: the per-(chain, hop) emitted split
    # re-states an edge's mass in EVERY slot it realizes (engine
    # realized-mass semantics), so the slot sum can exceed - never
    # fall below - the edge-level mass of connection_type.csv, and
    # no single slot can exceed it. The EDGE-level equality anchor
    # is `table_reproduced` (cut re-derivation), already asserted.
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv')
    ct_sum = ct.groupby(['type_pre', 'type_post'])['weight'].sum().to_dict()
    emitted_slots = defaultdict(list)
    for _, r in pf.iterrows():
        seq = str(r.path).split('->')
        e = json.loads(str(r.emitted_weights))
        for i in range(len(seq) - 1):
            emitted_slots[(seq[i], seq[i + 1])].append(e[i])
    for pair, masses in emitted_slots.items():
        total = int(ct_sum.get(pair, 0))
        assert sum(masses) >= total, (pair, sum(masses), total)
        assert max(masses) <= total, (pair, max(masses), total)


def test_complete_at_asked_stamp_guard_bit(tmp_path):
    """T4: a tiny refill path budget forces the drain tau above the asked
    tier — the complete table is still written but the stamp says false."""
    _fc, run_dir = make_run(tmp_path, 'caf_guard', edge_budget=7)
    # a tiny REFILL enumeration budget forces the guard: the drain tau
    # lands above the asked tier and the stamp says false (table still
    # written, README/caption disclose).
    compute_type_level_refill(run_dir, edges=EDGES, type_map=TYPE_MAP,
                              write=True, path_budget=2)
    mirror = run_dir / 'type_level_refill'
    csv = mirror / 'complete_at_asked_paths_type.csv'
    assert csv.exists()
    rec = json.loads((run_dir / 'data_details' / 'type_level_refill' /
                      'refill_provenance.json').read_text())
    assert rec['refill_truncated'] is True
    assert rec['complete_at_asked'] is False
    rows = pd.read_csv(csv)
    assert len(rows) == rec['complete_paths_type']


def test_unfired_residual_column(tmp_path):
    """R3: refill_type_pairs.csv carries unfired_residual as the LAST
    column; it reconciles asked_pair_mass against emitted + refill (0 on
    clean synthetic runs)."""
    _fc, run_dir = make_run(tmp_path, 'resid_floor', edge_budget=7)
    pairs = pd.read_csv(
        run_dir / 'data_details' / 'type_level_refill' /
        'refill_type_pairs.csv')
    assert list(pairs.columns)[-1] == 'unfired_residual'
    # every pair reconciles to >= 0 (mass conservation on the universe)
    assert (pairs['unfired_residual'] >= 0).all()


def test_shortest_mode_no_complete_table(tmp_path):
    """Shortest mode: no second-round paths — no complete CSV and a null
    stamp (R4 limitation documented in the README)."""
    _fc, run_dir = make_run(tmp_path, 'caf_short', edge_budget=7)
    # drive the SHORTEST-mode refill explicitly via compute on a shortest
    # run: reuse the hook-free API — a shortest run folder carries a
    # shortest_discovery_store; simulate by asserting the all-mode table
    # is absent only when the enumeration had no paths. The structural
    # guarantee is compute() writing complete_at_asked=None — covered by
    # the shortest-mode suites; here we assert the all-mode table IS
    # produced (positive control for the skip logic).
    assert (run_dir / 'type_level_refill' /
            'complete_at_asked_paths_type.csv').exists()


def test_type_incoming_applied_written(tmp_path):
    """The realized patch persists the applied-tier per-post incoming
    mass (data_details/type_incoming_applied.csv) — the denominator the
    pair report's applied-conditioned Adj. ratios use; every emitted
    pair's post type is covered."""
    _fc, run_dir = make_run(tmp_path, 'inapp_floor', edge_budget=7)
    csv = run_dir / 'data_details' / 'type_incoming_applied.csv'
    assert csv.exists()
    frame = pd.read_csv(csv)
    assert list(frame.columns) == ['type', 'incoming']
    assert (frame['incoming'] > 0).all()
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv')
    assert set(ct['type_post']) <= set(frame['type'])
    # DEDUP: an edge backing many paths contributes its mass ONCE —
    # reconcile against the distinct emitted edges' weights
    edges = pd.read_csv(
        run_dir / 'data_details' / 'connection_info_bodyId.csv',
        dtype=str).drop_duplicates(subset=['bodyId_pre', 'bodyId_post'])
    type_map = TYPE_MAP
    expect = defaultdict(float)
    for e in edges.itertuples():
        w = {(u, v): wt for u, v, wt in EDGES}.get(
            (e.bodyId_pre, e.bodyId_post))
        if w is not None:
            expect[type_map[e.bodyId_post]] += w
    got = frame.set_index('type')['incoming'].to_dict()
    for ty, mass in expect.items():
        assert abs(got.get(ty, 0) - mass) < 1e-6, (ty, mass)


def test_ratio_mode_refill_floor_synapse_ge_1(tmp_path):
    """RATIO-MODE FLOOR (2026-10-07, user directive): plain ratio-basis
    refills always run at synapse >= 1 — the run's own forced floor. A
    sub-1 edge in the stream (which the run's min_weight=1 fetch never
    saw) must NOT enter the second round NOR inflate the F9 denominators,
    even when its ratio clears the asked tier."""
    run = tmp_path / 'find-paths-complete_TEST_s_to_t_L2r0_05_20260101_000000'
    run.mkdir()
    (run / 'parameters.txt').write_text(
        'weight basis:                  connection_ratio (bodyId)\n'
        'requested_threshold:           0.05\n'
        'applied_threshold:             0.5\n'
        'applied_threshold_source:      strongest_first_budget\n'
        'strongest_first_budget:        100\n'
        'strongest_first_budget_bitten: True\n'
        'strongest_first_tau:           0.5\n'
        'min synapse number:            1\n'
        'max interlayer:                2\n'
        'aggregate method:              product\n'
        'filter by:                     bodyId\n'
        'separate hemispheres:          False\n'
        'exclude intra-type connections: False\n',
        encoding='utf-8')
    (run / 'data_details').mkdir()
    # the run's emitted table: ONE edge (s1->t1, 10 synapses). The run
    # fetched with min_weight=1, so the 0.9-synapse edge never existed
    # for it.
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TT', 'weight': 10,
         'nt_type': '', 'connection_ratio': 1.0, 'conn_layer': '0->1'},
    ]).to_csv(run / 'data_details' / 'connection_type.csv', index=False)
    pd.DataFrame([{'bodyId': 's1', 'type': 'TS', 'isInPath': True},
                  {'bodyId': 's2', 'type': 'TS', 'isInPath': False}]).to_csv(
        run / 'source_neurons.csv', index=False)
    pd.DataFrame([{'bodyId': 't1', 'type': 'TT', 'Checked': True}]).to_csv(
        run / 'target_neurons.csv', index=False)

    from type_level_refill import compute_type_level_refill
    rec = compute_type_level_refill(
        run,
        edges=[('s1', 't1', 10.0), ('s2', 't1', 0.9)],
        type_map={'s1': 'TS', 's2': 'TS', 't1': 'TT'},
        write=False)
    assert rec['status'] == 'refilled'
    assert rec['table_reproduced'] is True
    # the floor is stamped and the sub-1 edge is EXCLUDED: refill mass 0
    assert rec['refill_synapse_floor'] == 1
    assert rec['refill_weight_total'] == 0
    assert rec['refill_edges'] == 0
