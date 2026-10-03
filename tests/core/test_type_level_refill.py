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
    _tp._PIPELINE_TYPES = dict(types or TYPE_MAP)
    coana.FindNeuronConnection._ensure_neuprint_client = lambda self: None
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    shim = _ShimMonkey()
    try:
        yield shim
    finally:
        shim.undo()
        _tp._PIPELINE_TYPES = original
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
    # per-type-pair aggregate census (real-data lesson: the per-edge list
    # is unbounded; the census is bounded by the pair count)
    assert rec['skipped_pairs_not_emitted_type_pair'] == {
        'TS->TB': {'count': 1, 'weight': 5},
        'TB->TA': {'count': 1, 'weight': 4},
        'TC->TA': {'count': 1, 'weight': 3},
    }
    assert set(rec['skipped_edge_examples']) == \
        {'S1->B1(5)', 'B1->A2(4)', 'C2->A2(3)'}
    # C3->D1 and D1->T1 have an endpoint outside U (TD not involved)
    assert rec['excluded_edges_non_involved_endpoint'] == 2


def test_ratio_threshold_free(staged_runs):
    rows, _detail = write_records(staged_runs['floor'][1])
    tc = rows[(rows['type_pre'] == 'TC') & (rows['type_post'] == 'TT')]
    assert len(tc) == 1
    row = tc.iloc[0]
    assert row['refilled_total'] == 17
    assert abs(row['refilled_connection_ratio'] - 17 / TT_TOTAL_IN) < 1e-6


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
# 9b. Ratio-basis guard (plan-connection-ratio-pathfinding §16.1): the
# refill never fires for weight_basis='connection_ratio' runs, in-process
# or post-hoc. Dormant until the Phase-2 field exists.
# ---------------------------------------------------------------------------
def test_auto_hook_skips_ratio_basis(tmp_path):
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
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.2   # a valid ratio tier (the entry gate refuses 0)
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindAllPath()
        run_dir = Path(fc.allpath_folder)
    assert not (run_dir / 'data_details' / 'type_level_refill').exists()


def test_read_run_provenance_refuses_ratio_basis(tmp_path):
    run_dir = tmp_path / 'ratio_run'
    run_dir.mkdir()
    (run_dir / 'parameters.txt').write_text(
        'min synapse number: 1\n'
        'weight basis: connection_ratio (bodyId)\n'
        'requested_threshold: 0.05\n'
        'applied_threshold: 0.05\n',
        encoding='utf-8')
    with pytest.raises(TypeLevelRefillError, match='weight basis'):
        read_run_provenance(run_dir)


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
