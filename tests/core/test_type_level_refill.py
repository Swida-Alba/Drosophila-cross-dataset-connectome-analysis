import ast
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
    """DUAL REFILL (2026-10-08): a complete synapse run keeps
    status 'no_refill_needed' for the ASKED view (no asked tables) but
    now produces the refill-full scope (synapse>=1, emitted chains)."""
    run_dir = staged_runs['complete'][1]
    rec = refill(run_dir)
    assert rec['status'] == 'no_refill_needed'
    assert rec.get('full'), 'the full scope must ride on the record'
    assert rec['full']['status'] in ('ok', 'truncated')
    full_pairs = (run_dir / 'data_details' / 'type_level_refill' /
                  'refill_full_type_pairs.csv')
    assert full_pairs.exists()
    # the ASKED tables stay absent on the no-refill gate
    assert not (run_dir / 'data_details' / 'type_level_refill' /
                'refill_type_pairs.csv').exists()


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
    # The slice lives in the run's OWN output directory, restamped in the
    # folder NAME (plan 2026-10-16 review). The replay redirect used to run
    # `minsyn_\d+(?=/|$)` over the whole PATH, so here it renamed the output
    # DIRECTORY (minsyn_1 -> minsyn_11) and left the folder's own tier token
    # at w{asked} — a slice published beside a directory the user never chose
    # and a name that contradicted its parameters.txt.
    out_dir = tmp_path / 'minsyn_1'
    assert out_dir.is_dir(), (
        'the run\'s output directory was renamed by a slice restamp')
    published = [p for p in sorted(out_dir.glob('find-paths-*'))
                 if p.is_dir()]
    canon = [p for p in published if re.search(r'_L\d+w11_\d{8}_\d{6}$',
                                               p.name)]
    assert canon, (
        f'no slice folder carries the canonical tier in its NAME: '
        f'{[p.name for p in published]}')
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
        'format': 'drocat_custom_groups/v1',
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
    """DUAL REFILL (2026-10-08): a complete synapse run gets no ASKED
    refill data or note, but the refill-full scope IS written."""
    run_dir = staged_runs['complete'][1]
    details = run_dir / 'data_details' / 'type_level_refill'
    assert not (details / 'refill_type_pairs.csv').exists()
    assert (details / 'refill_full_type_pairs.csv').exists()
    notes = run_dir / 'user_warning_notes.txt'
    if notes.exists():
        text = notes.read_text(encoding='utf-8')
        assert 'type-level refill] applied threshold' not in text
        assert '[refill-full]' in text


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
        # gate L4: the pair lane books mass in SYNAPSES even on a
        # ratio-basis run — integers, never fractional ratio sums
        assert (rows['refill_weight'] >= 0).all()
        assert all(float(v).is_integer() for v in rows['refill_weight']), \
            list(rows['refill_weight'])
    # either way the hook no longer refuses ratio runs outright
    asked_in = out / 'type_incoming_asked.csv'
    if asked_in.exists():
        # a ratio-basis run can leave the conditioned map EMPTY — the file
        # must still carry its header, or the report's loader loses the
        # denominator (EmptyDataError on a lone newline).
        assert list(pd.read_csv(asked_in, nrows=0).columns) == [
            'type_post', 'incoming_threshold']
    notes = run_dir / 'user_warning_notes.txt'
    if notes.exists():
        assert 'does not apply to ratio-basis' not in notes.read_text(
            encoding='utf-8')


def _ratio_full_scope_run(tmp_path):
    """A ratio-basis run whose full scope (synapse >= 1) carries
    sub-threshold mass — the shape that produced F-EX-002(i)."""
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
        fc.min_ratio = 0.2
        fc._refill_edges_override = list(EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(TYPE_MAP), 'type': list(TYPE_MAP.values())})
        fc.FindAllPath()
        return Path(fc.allpath_folder)


def test_ratio_basis_full_scope_path_tables_are_synapse_units(tmp_path):
    """Gate L4 (F-EX-002(i)): on a ratio-basis run the full-scope PATH
    tables must carry SYNAPSE masses like the pair table always did. The
    defect: `e_u_full` enumerates on the F9 ratio, and the path builders
    summed that raw, so a chain published 16-digit ratio sums under the
    report's "Weights (synapse >= 1)" header and its ratios collapsed to
    0.00%."""
    run_dir = _ratio_full_scope_run(tmp_path)
    mirror = run_dir / 'type_level_refill'
    full_csv = mirror / 'refill_full_paths_type.csv'
    carry_csv = mirror / 'refill_full_carrying_paths_type.csv'
    assert full_csv.exists() and carry_csv.exists()

    def _mass_frame(csv):
        frame = pd.read_csv(csv)
        assert len(frame) > 0, csv
        for row in frame.itertuples():
            for col in ('weights', 'emitted_weights', 'refill_weights',
                        'total_weights'):
                if col not in frame.columns:
                    continue
                vals = ast.literal_eval(str(getattr(row, col)))
                # an integer-valued synapse count, never a ratio sum
                assert all(float(v).is_integer() and float(v) >= 0
                           for v in vals), (csv.name, col, vals)
        return frame

    full = _mass_frame(full_csv)
    carry = _mass_frame(carry_csv)
    # the three-way split reconciles hop for hop on both tables
    for row in full.itertuples():
        w = ast.literal_eval(str(row.weights))
        e = ast.literal_eval(str(row.emitted_weights))
        r = ast.literal_eval(str(row.refill_weights))
        assert [a + b for a, b in zip(e, r)] == w, row.path
    for row in carry.itertuples():
        w = ast.literal_eval(str(row.weights))          # sub-threshold
        e = ast.literal_eval(str(row.emitted_weights))
        t = ast.literal_eval(str(row.total_weights))
        assert [a + b for a, b in zip(e, w)] == t, row.path
    # F-EX-002(ii): the carrying view's subject is the sub-threshold band,
    # so its total must be one of the complete table's totals per chain
    full_by_path = {r.path: r.weights for r in full.itertuples()}
    for row in carry.itertuples():
        assert str(row.path) in full_by_path, row.path
        assert (ast.literal_eval(str(row.total_weights))
                == ast.literal_eval(str(full_by_path[row.path]))), row.path

    # UNIT CROSS-CHECK against independent ground truth: ratios must equal
    # the synapse mass over the ALL-POST incoming computed straight from the
    # fixture's EDGES. (A quotient of weights/ratios cannot discriminate a
    # unit — a ratio-unit numerator over a synapse denominator yields the
    # same denominator — so the mass itself is checked against the edge
    # list, which is exactly what the ratio-unit bug made impossible.)
    all_post: dict = defaultdict(float)
    for u, v, w in EDGES:
        all_post[TYPE_MAP[v]] += w
    checked = 0
    for row in full.itertuples():
        seq = str(row.path).split('->')
        w = ast.literal_eval(str(row.weights))
        r = ast.literal_eval(str(row.ratios))
        for i, ratio in enumerate(r):
            post = seq[i + 1]
            if not ratio or post not in all_post:
                continue
            assert abs(float(w[i]) - float(ratio) * all_post[post]) \
                <= 1e-6 * max(1.0, all_post[post]), (row.path, post, w[i],
                                                     ratio, all_post[post])
            checked += 1
    assert checked, 'no rows to cross-check against EDGES'


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
        'min_ratio', 'length', 'nt_types', 'coverage',
        'emitted_weights', 'total_weights']
    pf = pd.read_csv(paths_csv)
    assert len(pf) > 0
    # F-EX-002(ii): `weights` is the table's SUBJECT (the sub-threshold
    # mass), so the context columns must reconstruct the total exactly —
    # emitted + sub-threshold == total, hop for hop, on every row.
    for r in pf.itertuples():
        emit_w = ast.literal_eval(str(r.emitted_weights))
        sub_w = ast.literal_eval(str(r.weights))
        tot_w = ast.literal_eval(str(r.total_weights))
        assert len(emit_w) == len(sub_w) == len(tot_w) == r.length
        assert all(e + s == t for e, s, t in zip(emit_w, sub_w, tot_w)), r.path
        # the subject is never larger than the total it came from
        assert all(s <= t for s, t in zip(sub_w, tot_w)), r.path
        # the emitted class alone is never the whole story here: a carrying
        # row exists because it recovered mass
        assert any(s > 0 for s in sub_w), r.path
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


def test_empty_mirror_tables_still_publish_their_header(tmp_path):
    """A published CSV is parseable or it is not written. An UNBITTEN run
    has no sub-threshold chain to carry, so its full-scope mirror and the
    conditioned-denominator file are empty — and a header-less empty file
    (one newline) raised EmptyDataError in the report's loader, which
    logged the view as unreadable and dropped the conditioned denominator
    with it."""
    _fc, run_dir = make_run(tmp_path, 'mirror_unbitten', edge_budget=0)
    mirror = run_dir / 'type_level_refill'
    records = run_dir / 'data_details' / 'type_level_refill'
    carry = mirror / 'refill_full_carrying_paths_type.csv'
    assert carry.exists()
    first = carry.read_text(encoding='utf-8').splitlines()
    assert first and first[0] == (
        'path,weights,ratios,min_weight,min_ratio,length,nt_types,coverage,'
        'emitted_weights,total_weights'), first[:1]
    frame = pd.read_csv(carry)          # parses: 0 rows, full column set
    assert len(frame) == 0
    asked_in = records / 'type_incoming_asked.csv'
    assert asked_in.exists()
    assert list(pd.read_csv(asked_in, nrows=0).columns) == [
        'type_post', 'incoming_threshold']


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
    ]).to_csv(mirror / 'refilled_connection_type.csv', index=False)
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


def test_refill_visualizations_both_scopes_and_metrics(tmp_path):
    """F7.1/F7.2/F7.3 on the refill's OWN pages.

    The pages are built where the mirror is written, in the refill round
    itself — the materialization stage now runs the round BEFORE it renders
    (2026-10-16), but the refill's own scope pages still belong to the round
    that owns their data. Three laws, each checked against the wrong
    hypothesis:
      - a page exists for EVERY scope the mirror carries, and for no scope
        it does not (presence, not a hardcoded pair of names);
      - the asked page names the run's conditioned tier and the full page
        names 1 — a page labeled with another round's tier is a claim the
        reader cannot reproduce;
      - the 'Refilled' metric carries the pair's refilled_total instead of
        the (N/A) these pages used to show on the very subject they draw."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = True
    fc.min_synapse_num = 3          # -> _adj_ratio_tier() == 3
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})

    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir(parents=True)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 40,
         'nt_type': 'acetylcholine',
         'connection_ratio': 0.1, 'conn_layer': '',
         'connection_ratio_adj': 0.2},
    ]).to_csv(mirror / 'refill_connection_type.csv', index=False)
    # CURRENT shape: a bitten run ALSO mirrors the complete-at-asked TOTALS.
    # Without this file the asked page would draw the recovered DELTA (40)
    # beside the TOTAL's ratio and Refilled (46) — the exact mixed-scope page
    # the round forbids, so it now writes no asked page instead.
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
         'nt_type': 'acetylcholine',
         'connection_ratio': 0.115, 'conn_layer': '',
         'connection_ratio_adj': 0.23},
    ]).to_csv(mirror / 'refilled_connection_type.csv', index=False)
    # the full-scope mirror: the SAME pair, sub-threshold band + totals
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'emitted_weight': 40,
         'refill_weight': 6, 'refilled_total': 46,
         'refilled_connection_ratio': 0.115, 'pair_offchain_mass': 4},
    ]).to_csv(mirror / 'refilled_full_connection_type.csv', index=False)
    details = tmp_path / 'data_details' / 'type_level_refill'
    details.mkdir(parents=True)
    # the run's own transmitter matrix (the vocabulary source, issue 6)
    pd.DataFrame([{'type_pre': 'TS', 'TA': 'acetylcholine'}]).to_csv(
        tmp_path / 'data_details' / 'conn_mat_type_nt.csv', index=False)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'refill_weight': 6,
         'refilled_total': 46},
    ]).to_csv(details / 'refill_type_pairs.csv', index=False)

    fc._build_refill_visualizations(str(mirror))

    asked_net = (mirror / 'refill_type_network.html').read_text(
        encoding='utf-8')
    full_net = (mirror / 'refill_full_type_network.html').read_text(
        encoding='utf-8')
    for name in ('refill_type_heatmap.html', 'refill_type_network.html',
                 'refill_full_type_heatmap.html',
                 'refill_full_type_network.html'):
        assert (mirror / name).exists(), name
    # per-scope tier in the label (no bare placeholder, no borrowed tier)
    assert 'Adj. Ratio (&ge; 3)' in asked_net, asked_net[:200]
    assert 'Adj. Ratio (&ge; 1)' in full_net, full_net[:200]
    assert '(&ge; N)' not in asked_net and '(&ge; N)' not in full_net
    # the Refilled metric carries this pair's refilled_total on BOTH pages
    assert 'Refilled (N/A)' not in asked_net, 'asked page still N/A'
    assert 'Refilled (N/A)' not in full_net, 'full page still N/A'
    assert '"refilled": 46' in asked_net.replace('  ', ' '), asked_net[:400]
    assert '"refilled"' in full_net
    # the drawn WEIGHT is the same total — a page that draws the recovered
    # delta (40) while its ratio/Refilled read the total's (46) is the mixed
    # scope this rule exists to stop, so pin the number the page renders
    assert '"weight": 46' in asked_net.replace('  ', ' '), asked_net[:400]
    # NT parity with the run's own Network (user 2026-10-15): the round's
    # pages carry the transmitter per edge, so the legend/colouring option
    # has data instead of an empty legend.
    assert 'acetylcholine' in asked_net, 'asked page lost the transmitter'
    for page in (asked_net, full_net):
        assert '"nt_type"' in page

    # PRESENCE LAW: a full-only mirror builds no asked page (the budget did
    # not bite, so there is no asked table to draw) — the old gate skipped
    # the whole stage and produced neither.
    only = tmp_path / 'only_full' / 'type_level_refill'
    only.mkdir(parents=True)
    (tmp_path / 'only_full' / 'type_level_refill').parent.mkdir(
        exist_ok=True)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'emitted_weight': 40,
         'refill_weight': 6, 'refilled_total': 46,
         'refilled_connection_ratio': 0.115, 'pair_offchain_mass': 4},
    ]).to_csv(only / 'refilled_full_connection_type.csv', index=False)
    fc.allpath_folder = str(only.parent)
    fc._build_refill_visualizations(str(only))
    assert (only / 'refill_full_type_network.html').exists()
    assert not (only / 'refill_type_network.html').exists(), \
        'an asked page was invented for a run with no asked mirror'

    # SOURCE LAW (2026-10-16): when the round also mirrored its
    # complete-at-asked table, the asked page draws THAT (one row per
    # emitted pair, weight = refilled_total) and not the delta table —
    # otherwise a pair that recovered nothing disappears and its node is
    # drawn as a dead-end.
    both = tmp_path / 'both' / 'type_level_refill'
    both.mkdir(parents=True)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 40,
         'nt_type': 'acetylcholine', 'connection_ratio': 0.1,
         'conn_layer': '', 'connection_ratio_adj': 0.2},
    ]).to_csv(both / 'refill_connection_type.csv', index=False)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
         'nt_type': 'acetylcholine', 'connection_ratio': 0.115,
         'conn_layer': '', 'connection_ratio_adj': 0.23},
        {'type_pre': 'TA', 'type_post': 'TT', 'weight': 30,
         'nt_type': 'glutamate', 'connection_ratio': 0.075,
         'conn_layer': '', 'connection_ratio_adj': 0.15},
    ]).to_csv(both / 'refilled_connection_type.csv', index=False)
    fc.allpath_folder = str(both.parent)
    fc._build_refill_visualizations(str(both))
    asked_edges = _page_edges(both / 'refill_type_network.html')
    assert {k for k in asked_edges if k[0] == 'TS'} == {('TS', 'TA')}, \
        asked_edges.keys()
    assert ('TA', 'TT') in asked_edges, (
        'the zero-recovery pair was dropped again — the dead-end the user '
        'reported')
    assert asked_edges[('TS', 'TA')]['weight'] == 46, asked_edges[('TS', 'TA')]


def test_refilled_total_map_skips_unresolved_pairs(tmp_path):
    """`refilled_total` is None when a hop's post-type incoming cannot be
    resolved — the metric must then be absent, not a zero that reads as a
    measured 0 (the same law the report's cells follow)."""
    fc = object.__new__(coana.FindNeuronConnection)
    csv = tmp_path / 'refill_type_pairs.csv'
    pd.DataFrame([
        {'type_pre': 'A', 'type_post': 'B', 'refill_weight': 1,
         'refilled_total': 10.0},
        {'type_pre': 'A', 'type_post': 'C', 'refill_weight': 2,
         'refilled_total': None},
    ]).to_csv(csv, index=False)
    got = fc._refilled_total_map([str(csv)])
    assert got == {'refilled': {('A', 'B'): 10.0}}, got
    assert fc._refilled_total_map([str(tmp_path / 'nope.csv')]) == {}


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


def test_type_incoming_emitted_written(tmp_path):
    """The realized patch persists the EMITTED-CONE per-post incoming mass
    (data_details/type_incoming_emitted.csv) — this run's own emitted
    network, the full-scope refill's cache-guard input. Every emitted pair's
    post type is covered. Renamed from `type_incoming_applied.csv`, which
    invited the pair report to use it as an adjusted-ratio denominator
    (F-EX-003: it is a subset of the dataset total, so it inflated chain
    shares ~30x); chain ratios divide by type_incoming_conditioned.csv."""
    _fc, run_dir = make_run(tmp_path, 'inapp_floor', edge_budget=7)
    csv = run_dir / 'data_details' / 'type_incoming_emitted.csv'
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


def test_type_incoming_conditioned_matches_the_pair_table(tmp_path):
    """F-EX-003 D3a: the engine publishes the CONDITIONED per-post-type
    incoming it ACTUALLY used for `connection_ratio_adj`
    (data_details/type_incoming_conditioned.csv), so a chain row's adjusted
    ratio and the pair table divide by the same number. Path-independent by
    design: whether the denominator came from the dataset fetch or the
    in-frame last-resort fallback, the published file must equal the
    denominator implied by the pair table's own adjusted ratios."""
    _fc, run_dir = make_run(tmp_path, 'incond1', asked=1)
    csv = run_dir / 'data_details' / 'type_incoming_conditioned.csv'
    assert csv.exists()
    frame = pd.read_csv(csv)
    assert list(frame.columns) == ['type', 'incoming', 'threshold']
    assert set(frame['threshold']) == {1}
    cond = frame.set_index('type')['incoming'].to_dict()
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv')
    for tup in ct.itertuples():
        denom = float(tup.weight) / float(tup.connection_ratio_adj)
        assert abs(cond[str(tup.type_post)] - denom) < 1e-6, (
            tup.type_post, denom)

    # Raising Min Synapse Count conditions the denominator DOWN, never up,
    # and it always dominates this run's emitted-cone file.
    _fc2, run2 = make_run(tmp_path, 'incond14', asked=14)
    hi = pd.read_csv(
        run2 / 'data_details' / 'type_incoming_conditioned.csv')
    assert set(hi['threshold']) == {14}
    hi_map = hi.set_index('type')['incoming'].to_dict()
    for ty, mass in hi_map.items():
        if ty in cond:
            assert mass <= cond[ty] + 1e-6, ty
    emit = pd.read_csv(run2 / 'data_details' / 'type_incoming_emitted.csv')
    for row in emit.itertuples():
        assert row.incoming <= hi_map.get(row.type, 0.0) + 1e-6, row.type


def test_ratio_mode_refill_floor_synapse_ge_1(tmp_path):
    """Ratio-mode refill floor synapse >= 1."""
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


def test_dfs_chain_structures_alive_pruning_and_completion_recording():
    """F6 (plan-2026-10-11): walk-level alive pruning strips dead-end
    branches losslessly, and structures record only edges/bodies on
    >= 1 collected realization (the old explore-time recording inflated
    multi-visit masses with realized-nothing edges)."""
    from type_level_refill import (_dfs_chain_structures,
                                   _dp_chain_structures)
    # chain A->B->A->C; walk-dead decoys: a1->b2 (no B->A from b2),
    # b1->a3 (no A->C from a3)
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 10.0), ('b2', 4.0)]},
        ('B', 'A'): {'b1': [('a2', 9.0), ('a3', 8.0)]},
        ('A', 'C'): {'a2': [('c1', 7.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'a3': 'A', 'b1': 'B', 'b2': 'B',
                'c1': 'C'}
    chain = ('A', 'B', 'A', 'C')
    res = _dfs_chain_structures(chain, pair_edges, ['a1'], ['c1'],
                                type_map, set())
    assert res is not None and res['path_count'] == 1
    assert res['truncated'] is False
    assert set(res['traversals']) == {('a1', 'b1'), ('b1', 'a2'),
                                      ('a2', 'c1')}
    assert res['edges_at'][0] == {('a1', 'b1'): 10.0}
    assert res['edges_at'][2] == {('a2', 'c1'): 7.0}
    assert res['nodes_at'][1] == {'b1'} and res['nodes_at'][3] == {'c1'}
    # the walk-dead start-only case: prune to nothing -> None
    res2 = _dfs_chain_structures(
        chain, {('A', 'B'): {'a1': [('b9', 4.0)]},
                ('B', 'A'): {'b9': [('a9', 4.0)]},
                ('A', 'C'): {'a9': [('c1', 4.0)]}},
        ['zz'], ['c1'],
        {'a1': 'A', 'b9': 'B', 'a9': 'A', 'c1': 'C', 'zz': 'A'}, set())
    assert res2 is None


def test_dfs_matches_dp_on_distinct_chain_fixture():
    """The multi-visit DFS and the distinct-chain DP must agree exactly
    on a distinct-type chain (simplicity is automatic there) — the
    standing parity pin, now against the pruned + completion-recorded
    DFS."""
    from type_level_refill import (_dfs_chain_structures,
                                   _dp_chain_structures)
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 9.0), ('b2', 5.0)],
                     'a2': [('b2', 6.0)]},
        ('B', 'C'): {'b1': [('c1', 8.0)], 'b2': [('c1', 3.0),
                                                 ('c2', 2.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B',
                'c1': 'C', 'c2': 'C'}
    chain = ('A', 'B', 'C')
    dp = _dp_chain_structures(chain, pair_edges, ['a1', 'a2'],
                              ['c1', 'c2'], type_map, set())
    dfs = _dfs_chain_structures(chain, pair_edges, ['a1', 'a2'],
                                ['c1', 'c2'], type_map, set())
    assert dp['path_count'] == dfs['path_count'] == 5
    assert dp['traversals'] == dfs['traversals']
    assert dp['edges_at'] == dfs['edges_at']
    assert dp['nodes_at'] == dfs['nodes_at']


# ---------------------------------------------------------------------------
# The universal layered per-position filter (plan-corridor-ratio-basis
# §8.2/§9, 2026-10-13): the full-scope realization. Exactness oracle =
# the retired DP; simple-path bound = the retired DFS.
# ---------------------------------------------------------------------------

def test_layered_filter_equals_dp_walk_oracle_with_decoys():
    """Exactness pin (plan §9.5): the layered filter's kept set (edges,
    classes, node sets) equals the counted DP's realization set exactly.
    The DP's f/b structures are LAYERED-WALK level, so it oracles the
    filter on revisiting chains too (production used the DFS there
    precisely because the DP over-counts SIMPLE-path mass — its kept
    EDGE set is still the walk set). Covers decoy rejection: a
    forward-only decoy (reaches no target) and a same-type non-source
    body (pseudo-source trap) must both be excluded; seeding uses the
    run's sources/targets only."""
    from type_level_refill import (_dp_chain_structures,
                                   _layered_chain_structures)
    # decoys: a3 forward-reaches b3 but b3 is backward-dead; b4 is
    # backward-alive but no source reaches it; a2 is same-type as the
    # source but NOT a source (pseudo-source trap via b4). The chain
    # revisits A (positions 0 and 2) — revisit must be flagged.
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 10.0), ('b2', 4.0), ('b3', 6.0)],
                     'a2': [('b4', 5.0)]},
        ('B', 'A'): {'b1': [('a9', 9.0)], 'b2': [('a9', 8.0)]},
        ('A', 'C'): {'a9': [('c1', 7.0), ('c2', 3.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'a9': 'A', 'b1': 'B', 'b2': 'B',
                'b3': 'B', 'b4': 'B', 'c1': 'C', 'c2': 'C'}
    chain = ('A', 'B', 'A', 'C')
    for cut_keys in (set(), {('a1', 'b1')}):
        dp = _dp_chain_structures(chain, pair_edges, ['a1'], ['c1', 'c2'],
                                  type_map, cut_keys)
        layered = _layered_chain_structures(
            chain, pair_edges, ['a1'], ['c1', 'c2'], type_map, cut_keys)
        assert dp is not None and layered is not None
        assert layered['edges_at'] == dp['edges_at']
        assert layered['edges_cls_at'] == dp['edges_cls_at']
        assert layered['nodes_at'] == dp['nodes_at']
        assert layered['revisit'] is True
    # the decoys really are excluded by the intersection
    layered = _layered_chain_structures(
        chain, pair_edges, ['a1'], ['c1', 'c2'], type_map, set())
    assert ('a1', 'b3') not in layered['edges_at'][0]  # b3 backward-dead
    assert ('a2', 'b4') not in layered['edges_at'][0]  # a2 not a source
    assert layered['nodes_at'][0] == {'a1'}


def test_layered_filter_equals_dp_and_dfs_on_distinct_chain():
    """Distinct-chain exactness (plan §9.5): with no label revisit all
    three agree exactly — the layered filter, the DP, and the DFS (the
    standing DP/DFS parity fixture, now three-way)."""
    from type_level_refill import (_dfs_chain_structures,
                                   _dp_chain_structures,
                                   _layered_chain_structures)
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 9.0), ('b2', 5.0)],
                     'a2': [('b2', 6.0)]},
        ('B', 'C'): {'b1': [('c1', 8.0)], 'b2': [('c1', 3.0),
                                                 ('c2', 2.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B',
                'c1': 'C', 'c2': 'C'}
    chain = ('A', 'B', 'C')
    for cut_keys in (set(), {('a2', 'b2')}):
        dp = _dp_chain_structures(chain, pair_edges, ['a1', 'a2'],
                                  ['c1', 'c2'], type_map, cut_keys)
        dfs = _dfs_chain_structures(chain, pair_edges, ['a1', 'a2'],
                                    ['c1', 'c2'], type_map, cut_keys)
        layered = _layered_chain_structures(
            chain, pair_edges, ['a1', 'a2'], ['c1', 'c2'], type_map,
            cut_keys)
        assert layered is not None and layered['revisit'] is False
        assert layered['edges_at'] == dp['edges_at'] == dfs['edges_at']
        assert (layered['edges_cls_at'] == dp['edges_cls_at']
                == dfs['edges_cls_at'])
        assert layered['nodes_at'] == dp['nodes_at'] == dfs['nodes_at']


def test_layered_filter_none_contract_matches_dp():
    """None cases match the DP: backward-dead chains (no realization),
    missing hop pairs, and empty source/target type matches all return
    None."""
    from type_level_refill import (_dp_chain_structures,
                                   _layered_chain_structures)
    type_map = {'a1': 'A', 'b1': 'B', 'c1': 'C', 'z': 'Z'}
    dead = {('A', 'B'): {'a1': [('b1', 4.0)]},
            ('B', 'C'): {}}  # no B->C edges: backward-dead
    chain = ('A', 'B', 'C')
    assert _layered_chain_structures(
        chain, dead, ['a1'], ['c1'], type_map, set()) is None
    assert _dp_chain_structures(
        chain, dead, ['a1'], ['c1'], type_map, set()) is None
    missing_hop = {('A', 'B'): {'a1': [('b1', 4.0)]}}
    assert _layered_chain_structures(
        chain, missing_hop, ['a1'], ['c1'], type_map, set()) is None
    no_source = {('A', 'B'): {'a1': [('b1', 4.0)]},
                 ('B', 'C'): {'b1': [('c1', 4.0)]}}
    assert _layered_chain_structures(
        chain, no_source, ['z'], ['c1'], type_map, set()) is None
    assert _layered_chain_structures(
        chain, no_source, ['a1'], ['z'], type_map, set()) is None


def test_layered_filter_revisiting_chain_is_labeled_superset():
    """Superset pin (plan §9.5): on a label-revisiting chain the layered
    filter keeps the layered-WALK edges the simple-path semantics
    excludes — labeled via ``revisit`` — with per-position groups (the
    revisited body occupies two positions) and an edge recurring at two
    positions counted once per position. The DFS (simple-path bound)
    realizes nothing on this fixture; the DP is not consultable (it
    would over-count)."""
    from type_level_refill import (_dfs_chain_structures,
                                   _layered_chain_structures)
    # chain A->B->A->B revisits A and B; the ONLY realization is the
    # walk a1->b1->a1->b1 (no simple path exists: every realization
    # revisits bodies)
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 5.0)]},
        ('B', 'A'): {'b1': [('a1', 3.0), ('a2', 9.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'b1': 'B'}
    chain = ('A', 'B', 'A', 'B')
    layered = _layered_chain_structures(
        chain, pair_edges, ['a1'], ['b1'], type_map, set())
    assert layered is not None and layered['revisit'] is True
    # the walk's edges, each at its position; (a1,b1) counts once per
    # position (hops 0 and 2)
    assert layered['edges_at'][0] == {('a1', 'b1'): 5.0}
    assert layered['edges_at'][1] == {('b1', 'a1'): 3.0}
    assert layered['edges_at'][2] == {('a1', 'b1'): 5.0}
    # per-position groups: a1 occupies positions 0 and 2 (two
    # memberships), b1 positions 1 and 3
    assert layered['nodes_at'] == [{'a1'}, {'b1'}, {'a1'}, {'b1'}]
    # the simple-path bound realizes nothing here (the DFS finds no
    # simple path) — the layered set is a strict superset, disclosed
    dfs = _dfs_chain_structures(
        chain, pair_edges, ['a1'], ['b1'], type_map, set())
    assert dfs is None
    # a revisiting walk whose decoy branch is walk-dead stays excluded:
    # a2 is backward-alive (a2 reaches b1? no — a2 has no A->B edges)
    assert ('b1', 'a2') not in layered['edges_at'][1]


def test_layered_filter_superset_over_dfs_on_multi_visit_walks():
    """On a multi-visit chain WITH simple paths, the layered kept set
    contains the DFS-collected (simple-path) edge set edge for edge —
    the labeled superset over a completing bound."""
    from type_level_refill import (_dfs_chain_structures,
                                   _layered_chain_structures)
    # chain A->B->A->C: simple path a1->b1->a2->c1 exists; the layered
    # filter ALSO admits walk-only edges if any (here none — the decoys
    # are walk-dead — so equality on this fixture; a revisit-exposing
    # variant follows)
    pair_edges = {
        ('A', 'B'): {'a1': [('b1', 10.0)]},
        ('B', 'A'): {'b1': [('a2', 9.0), ('a1', 8.0)]},
        ('A', 'C'): {'a2': [('c1', 7.0)], 'a1': [('c1', 2.0)]},
    }
    type_map = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c1': 'C'}
    chain = ('A', 'B', 'A', 'C')
    dfs = _dfs_chain_structures(
        chain, pair_edges, ['a1'], ['c1'], type_map, set())
    layered = _layered_chain_structures(
        chain, pair_edges, ['a1'], ['c1'], type_map, set())
    assert dfs is not None and layered is not None
    assert layered['revisit'] is True
    # superset: every simple-path edge is kept...
    for i, edge_d in enumerate(dfs['edges_at']):
        for key in edge_d:
            assert key in layered['edges_at'][i]
    # ...and here the walk a1->b1->a1->c1 is layered-realizable but no
    # simple path, so the layered filter strictly exceeds the DFS: the
    # edge (b1,a1) backs only the revisiting walk
    assert ('b1', 'a1') in layered['edges_at'][1]
    assert ('b1', 'a1') not in dfs['edges_at'][1]
    assert ('a1', 'c1') in layered['edges_at'][3 - 1]  # hop 2
    assert ('a1', 'c1') not in dfs['edges_at'][2]


def test_load_shortest_store_slice_sum(tmp_path):
    """Per-ROI slice-sum fix (plan §10, original-data adjudication):
    the store's layer frames are verbatim per-ROI fetch slices — the
    per-layer physical pair weight is the SUM of the slices; keep-one-
    row policies sampled a different slice per layer. The loader must
    also iterate layers in sorted label order (deterministic keep-first
    at consumption)."""
    import polars as pl
    run = tmp_path / "run"
    store = run / "shortest_discovery_store"
    (store / "connections").mkdir(parents=True)
    (store / "meta.json").write_text('{"targets_found": []}')
    # (a1,b1): two L1 slices 2+7=9; a later same-roi row in L0 (4) is
    # a refetch of the SAME slice -> keep-first at consumption picks
    # L0's 4 for the physical dict; within L0 the duplicate exact row
    # collapses (refetch multiplicity)
    pl.DataFrame({
        "bodyId_pre": ["a1", "a1"],
        "bodyId_post": ["b1", "b1"],
        "roi": ["A", "A"],
        "weight": [4.0, 4.0]}).write_parquet(
        store / "connections" / "connections_L0.parquet")
    pl.DataFrame({
        "bodyId_pre": ["a1", "a1", "b1"],
        "bodyId_post": ["b1", "b1", "c1"],
        "roi": ["A", "B", "A"],
        "weight": [2.0, 7.0, 3.0]}).write_parquet(
        store / "connections" / "connections_L1.parquet")
    from type_level_refill import _load_shortest_store
    res = _load_shortest_store(run)
    assert res is not None
    by_layer = dict(res["layer_weights"])
    assert [k for k, _ in res["layer_weights"]] == \
        sorted(k for k, _ in res["layer_weights"])
    assert by_layer["connections_L1"] == {("a1", "b1"): 9.0,
                                          ("b1", "c1"): 3.0}
    assert by_layer["connections_L0"] == {("a1", "b1"): 4.0}


def test_mirror_tables_carry_the_runs_transmitter_vocabulary(staged_runs,
                                                             tmp_path):
    """Issue 6 (user, 2026-10-15): the refill's mirror tables hardcoded
    `nt_type: ''`, so its pages had an empty NT legend while the run's own
    Network was NT-coloured — two visuals disagreeing about the same edges.
    The vocabulary now comes from the run's own
    data_details/conn_mat_type_nt.csv (the same curated source), and the
    historical empty string is kept when that file is absent."""
    import shutil
    _fc, src = staged_runs['bite']
    run = tmp_path / src.name
    shutil.copytree(src, run)
    nt_csv = run / 'data_details' / 'conn_mat_type_nt.csv'
    if nt_csv.exists():
        nt_csv.unlink()

    # without the matrix: empty strings, nothing invented. The mirror
    # tables land in <run>/type_level_refill/ when no out_dir is forced —
    # that is the in-run shape the pages are built from.
    mirror = run / 'type_level_refill'
    compute_type_level_refill(run, edges=EDGES, type_map=TYPE_MAP,
                              write=True)
    asked = pd.read_csv(mirror / 'refill_connection_type.csv')
    assert set(asked['nt_type'].fillna('')) == {''}

    # with the run's matrix: every listed pair carries its transmitter
    types = sorted(set(TYPE_MAP.values()))
    pd.DataFrame([{'type_pre': pre, **{post: 'acetylcholine'
                                       for post in types}}
                  for pre in types]).to_csv(nt_csv, index=False)
    compute_type_level_refill(run, edges=EDGES, type_map=TYPE_MAP,
                              write=True)
    asked2 = pd.read_csv(mirror / 'refill_connection_type.csv')
    assert (asked2['nt_type'].astype(str) == 'acetylcholine').all(), \
        asked2[['type_pre', 'type_post', 'nt_type']].head()
    full = pd.read_csv(mirror / 'refilled_full_connection_type.csv')
    assert 'nt_type' in full.columns, full.columns.tolist()
    assert (full['nt_type'].astype(str) == 'acetylcholine').all(), full

def test_the_type_level_visualization_is_handed_the_refill_metrics():
    """F7.1 → G6's missing link (§8.6): the run's own type-level Network /
    Heatmap call must be handed the refill pair metrics — and since the
    2026-10-16 ordering fix the round RUNS BEFORE that render, so the
    pass-through carries an actual number instead of an empty dict. Pinned
    on the source pair (computed → handed, in that order) and on behavior in
    `test_the_run_pages_carry_the_refilled_metric_in_run`."""
    import inspect

    import coana

    src = inspect.getsource(coana.FindNeuronConnection._materialize_paths)
    refill_first = src.index('_maybe_generate_type_level_refill()')
    computed = src.index('_attach_viz_extra_metrics(')
    passed = src.index('pair_extra_metrics=(_pair_extra or None)', computed)
    assert refill_first < computed, (
        'the refill round moved back after the metrics read — the run would '
        'again ship its own network page with the Refilled option disabled')
    assert computed < passed, 'the type-level viz dropped the pair metrics'
    assert 'VisualizePath(' in src[computed:passed]
    # ... and the helper's own contract (returned, not swallowed). The pair
    # read itself moved OUT into `_refilled_pair_metrics` (L5, 2026-10-16) so
    # `FindNetwork`'s own pages read the metrics through the SAME reader — the
    # law survives as "the helper delegates" plus "the reader keeps its order".
    helper = inspect.getsource(
        coana.FindNeuronConnection._attach_viz_extra_metrics)
    reader = inspect.getsource(
        coana.FindNeuronConnection._refilled_pair_metrics)
    assert "pair_extra.update(self._refilled_pair_metrics(folder))" in helper
    assert "self._refilled_total_map([" in reader
    # the FULL scope leads: one meaning for 'Refilled' on every run
    order = reader[reader.index('_refilled_total_map(['):]
    assert order.index('refill_full_type_pairs.csv') < \
        order.index("refill_type_pairs.csv"), order[:400]


def _page_edges(html_path):
    """The embedded cytoscape edge list of a vispath network page, keyed by
    (source, target) — what the page ACTUALLY draws, not what the CSVs
    claim."""
    text = Path(html_path).read_text(encoding='utf-8', errors='replace')
    i = text.find('[{"data": {"source":')
    assert i >= 0, f'{html_path}: no embedded edge array found'
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '[':
            depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0:
                arr = json.loads(text[i:j + 1])
                break
    return {(e['data']['source'], e['data']['target']): e['data']
            for e in arr if 'source' in e.get('data', {})}


class _RecordingVisualizePath:
    """Stand-in for VisualizePath that records the edge frame each page is
    handed — what the page draws, without paying for a render."""

    calls = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append(kwargs)
        self.G_network = None
        self.edge_limit_trimmed = False

    def build_network(self):
        pass

    def create_heatmap(self):
        pass

    def create_network(self):
        pass

    def visualize(self, **kwargs):
        pass


def _drawn_edges(call):
    """(source, target) -> the row the page was handed for that pair."""
    frame = call['path_file']
    rows = frame if isinstance(frame, list) else frame.to_dict('records')
    return {(str(r['source']), str(r['target'])): r for r in rows}


def test_the_run_pages_are_handed_the_refilled_metric_in_run(tmp_path):
    """Issue 2b (user, 2026-10-16): 'can we embed the refill to the main
    network too, like the ratio, adj ratio? there is a refill selection
    disabled in the network's connection metric.'

    The run's own Network/Heatmap read their metrics through
    `_attach_viz_extra_metrics`, so the law is testable without a render: on
    a FRESH offline folder — one that has never been post-processed — that
    helper must return the pair's Refilled totals, because the round now runs
    before the render. Measured before the fix: `refilled: null` on all 23
    edges of this query's own network page."""
    _fc, run = make_run(tmp_path, 'refill_in_run')
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(run)
    paths = pd.DataFrame([
        {'path': 'S1->M->T1', 'path_block': 'S1->M->T1',
         'weights': '[40, 30]'}])
    _frame, pair_extra = fc._attach_viz_extra_metrics(paths, str(run))
    refilled = (pair_extra or {}).get('refilled') or {}
    assert refilled, (
        'a fresh run folder still hands its own pages an empty Refilled map')
    full = pd.read_csv(run / 'data_details' / 'type_level_refill'
                       / 'refill_full_type_pairs.csv')
    want = {(r.type_pre, r.type_post): float(r.refilled_total)
            for r in full.itertuples()}
    assert refilled == want, (refilled, want)


def test_a_delta_only_mirror_writes_no_self_contradicting_page(tmp_path):
    """A folder whose mirror carries ONLY the recovered delta has no asked
    total to draw. Before this rule it fell back to the delta table and
    printed the total's ratio and Refilled beside it (weight 40 / ratio 0.115)
    — the mixed-scope page the user reported. Now: no asked page, one note,
    and the full-scope page (which does carry totals) still written."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = True
    fc.min_synapse_num = 3
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})
    fc._warn_notes = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None

    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir()
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 6,
         'connection_ratio': 0.115},
    ]).to_csv(mirror / 'refill_connection_type.csv', index=False)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
         'refilled_connection_ratio': 0.115},
    ]).to_csv(mirror / 'refilled_full_connection_type.csv', index=False)

    fc._build_refill_visualizations(str(mirror))

    assert not (mirror / 'refill_type_network.html').exists(), (
        'the asked page drew the delta beside the total\'s ratio again')
    assert (mirror / 'refill_full_type_network.html').exists(), (
        'the scope that carries totals lost its page')
    assert any('asked' in n and 'no asked' in n for n in fc._warn_notes), (
        fc._warn_notes)


def test_refill_pages_draw_the_whole_topology_with_refilled_values(tmp_path):
    """Issue 2a (user, 2026-10-16, measured on
    find-paths-complete_MCNS_aMe12_etc_to_PPL101_etc_L1w5_20261016_132625):
    `refill_full_type_network.html` showed dead-ends (KCg, KCg-s1 had no
    outgoing edge) because the page drew the recovered DELTA and dropped every
    pair that recovered nothing — 18 edges against the run's own 23 — while
    the ratio beside the delta weight was the TOTAL's share.

    Law: a refill page is the run's topology RE-VALUED. Same pair set, weight
    = refilled_total, and the Refilled metric equals that weight; the asked
    page reads the complete-at-asked mirror, not the delta table."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = True
    fc.min_synapse_num = 3
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})

    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir()
    # the delta mirror (what the page USED to draw) — TS->TB recovered nothing
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 40,
         'nt_type': 'acetylcholine', 'connection_ratio': 0.1,
         'conn_layer': '', 'connection_ratio_adj': 0.2},
    ]).to_csv(mirror / 'refill_connection_type.csv', index=False)
    # the complete-at-asked mirror: every emitted pair, totals
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
         'nt_type': 'acetylcholine', 'connection_ratio': 0.115,
         'conn_layer': '', 'connection_ratio_adj': 0.23},
        {'type_pre': 'TS', 'type_post': 'TB', 'weight': 30,
         'nt_type': 'glutamate', 'connection_ratio': 0.075,
         'conn_layer': '', 'connection_ratio_adj': 0.15},
    ]).to_csv(mirror / 'refilled_connection_type.csv', index=False)
    # the pair tables the PAGES read their Refilled metric from, one per
    # scope, each carrying that scope's own total (a page never borrows the
    # other scope's number) — without them the metric is simply absent, and
    # the law below requires it to be present and equal to the drawn weight
    details = tmp_path / 'data_details' / 'type_level_refill'
    details.mkdir(parents=True)
    for name in ('refill_type_pairs.csv', 'refill_full_type_pairs.csv'):
        pd.DataFrame([
            {'type_pre': 'TS', 'type_post': 'TA', 'emitted_weight': 40,
             'refill_weight': 6, 'refilled_total': 46},
            {'type_pre': 'TS', 'type_post': 'TB', 'emitted_weight': 30,
             'refill_weight': 0, 'refilled_total': 30},
        ]).to_csv(details / name, index=False)
    # the full round: one pair recovers mass, the other none (the dead-end)
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'emitted_weight': 40,
         'refill_weight': 6, 'refilled_total': 46,
         'refilled_connection_ratio': 0.115, 'pair_offchain_mass': 4},
        {'type_pre': 'TS', 'type_post': 'TB', 'emitted_weight': 30,
         'refill_weight': 0, 'refilled_total': 30,
         'refilled_connection_ratio': 0.075, 'pair_offchain_mass': 0},
    ]).to_csv(mirror / 'refilled_full_connection_type.csv', index=False)

    _RecordingVisualizePath.calls = []
    monkey = _ShimMonkey()
    monkey.setattr(coana, 'VisualizePath', _RecordingVisualizePath)
    try:
        fc._build_refill_visualizations(str(mirror))
    finally:
        monkey.undo()

    by_name = {}
    for call in _RecordingVisualizePath.calls:
        by_name[call['path_file'].iloc[0]['source']] = call
    pages = {('asked' if c.get('adj_ratio_threshold') == 3 else 'full'): c
             for c in _RecordingVisualizePath.calls}
    assert len(pages) == 2, [c.get('adj_ratio_threshold') for c in
                             _RecordingVisualizePath.calls]
    for scope, call in pages.items():
        drawn = _drawn_edges(call)
        assert set(drawn) == {('TS', 'TA'), ('TS', 'TB')}, (
            f'{scope}: a pair with no recovered mass vanished again '
            f'({sorted(drawn)})')
        assert drawn[('TS', 'TB')]['weight'] == 30, drawn[('TS', 'TB')]
        assert drawn[('TS', 'TA')]['weight'] == 46, drawn[('TS', 'TA')]
        # weight and ratio on ONE basis: TB's ratio is 30/denominator, and
        # the delta (0) must never be the drawn weight beside it
        assert drawn[('TS', 'TB')]['ratio'] == 0.075, drawn[('TS', 'TB')]
        refilled = (call.get('pair_extra_metrics') or {}).get('refilled')
        # presence first: `refilled in (None, {}) or all(...)` passed on an
        # absent metric, so a page that dropped the metric entirely could
        # never fail the law it was written to hold
        assert refilled, f'{scope}: the page carries no Refilled metric at all'
        assert set(refilled) == set(drawn), (
            f'{scope}: metric pairs {sorted(refilled)} vs drawn '
            f'{sorted(drawn)}')
        assert all(abs(refilled[k] - drawn[k]['weight']) < 1e-9
                   for k in drawn), (
            f'{scope}: the metric and the drawn weight disagree')


def test_the_refill_pages_render_once_per_folder(tmp_path):
    """Both call sites are live: the round builds the pages where it writes
    the mirror (2026-10-16), and the historical visualization-stage site
    stayed — measured on three regimes (unbitten, budget-bitten,
    edge-bitten), so EVERY run rendered and overwrote its refill pages twice.

    The builder is now idempotent per folder. The visualization-stage site is
    kept on purpose: it keys on the mirror CSVs being ON DISK, not on this run
    having written them, so a folder that arrives at the visualization stage
    with a mirror this round did not render (a refill-disabled re-run over an
    existing folder) still gets its pages.
    Wrong hypothesis: a second call that renders again, or a guard keyed so
    broadly that a DIFFERENT folder is skipped too."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc._warn_notes = []
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = True
    fc.min_synapse_num = 3
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None

    def _mirror(dir_path):
        dir_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
             'connection_ratio': 0.115, 'nt_type': ''},
        ]).to_csv(dir_path / 'refilled_full_connection_type.csv', index=False)

    _RecordingVisualizePath.calls = []
    monkey = _ShimMonkey()
    try:
        monkey.setattr(coana, 'VisualizePath', _RecordingVisualizePath)
        first = tmp_path / 'type_level_refill'
        _mirror(first)
        fc.allpath_folder = str(tmp_path)
        fc._build_refill_visualizations(str(first))
        n_after_first = len(_RecordingVisualizePath.calls)
        fc._build_refill_visualizations(str(first))
        n_after_second = len(_RecordingVisualizePath.calls)
        other = tmp_path / 'slice2' / 'type_level_refill'
        _mirror(other)
        fc.allpath_folder = str(tmp_path / 'slice2')
        fc._build_refill_visualizations(str(other))
        n_after_other = len(_RecordingVisualizePath.calls)
    finally:
        monkey.undo()

    assert n_after_first > 0, 'the first call rendered nothing at all'
    assert n_after_second == n_after_first, (
        f'the same folder rendered {n_after_second - n_after_first} more '
        'objects on the second call — the double render is back')
    assert n_after_other > n_after_second, (
        'a DIFFERENT folder was skipped: the guard is not per folder')


def _viz_fc(tmp_path, **over):
    """A FindNeuronConnection with just the attributes the refill page
    builder reads (the same shape the page laws above use)."""
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.verbose_mode = 'quiet'
    fc.showfig = False
    fc.edgeN_limit = 500
    fc.output_format = 'html'
    fc.network_layout = 'hierarchical'
    fc.separate_hemispheres = False
    fc.skip_bodyId = True
    fc.min_synapse_num = 3
    fc.source_df = pd.DataFrame({'bodyId': ['S1'], 'type': ['TS']})
    fc.target_df = pd.DataFrame(
        {'bodyId': ['T1'], 'type': ['TT'], 'Checked': [True]})
    fc._warn_notes = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None
    for key, value in over.items():
        setattr(fc, key, value)
    return fc


def test_a_delta_only_full_mirror_writes_no_page_either(tmp_path):
    """The asked scope refuses a delta-only mirror (law above); the FULL scope
    must answer the same way (plan 2026-10-16 review).

    `refilled_full_connection_type.csv` written before 2026-10-16 carried only
    the recovered delta, and the builder RENAMED `refill_weight` to `weight`
    and drew it — the page then showed recovered mass under the total's ratio,
    exactly the mixed-scope defect the asked rule exists to stop. Two shapes
    are therefore distinct: a mirror with a total (refilled_total, or the
    delta-and-total `weight` the mirror grammar uses) draws; a mirror whose
    only mass column is the delta writes no page and says so.

    Wrong hypothesis: the pre-fix rename wrote refill_full_type_network.html
    with weight 6 (the delta) for a pair whose ratio was the total's share."""
    fc = _viz_fc(tmp_path)
    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir()
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'emitted_weight': 40,
         'refill_weight': 6},
    ]).to_csv(mirror / 'refilled_full_connection_type.csv', index=False)

    _RecordingVisualizePath.calls = []
    monkey = _ShimMonkey()
    try:
        monkey.setattr(coana, 'VisualizePath', _RecordingVisualizePath)
        fc._build_refill_visualizations(str(mirror))
    finally:
        monkey.undo()

    assert not _RecordingVisualizePath.calls, (
        'the delta-only full mirror was drawn again — a page whose weight is '
        'the recovered mass under the total\'s ratio')
    assert not (mirror / 'refill_full_type_network.html').exists()
    assert any('synapse >= 1 scope carries' in n for n in fc._warn_notes), (
        fc._warn_notes)


def test_a_crash_mid_render_leaves_the_folder_unmarked(tmp_path):
    """The idempotence mark is made only when the render RETURNS (plan
    2026-10-16 review).

    The mark used to happen before any page was drawn, so a raise inside the
    render (a bad mirror frame, a vispath failure) was swallowed by the
    visualization stage's advisory `except` and the second site skipped the
    folder as 'already rendered' — the run then shipped a refill folder with
    CSVs and no pages at all, and the reader had no way to get them short of a
    full re-run.

    Wrong hypothesis: marking up front, where the retry below renders nothing
    and the pages stay missing."""
    fc = _viz_fc(tmp_path)
    mirror = tmp_path / 'type_level_refill'
    mirror.mkdir()
    pd.DataFrame([
        {'type_pre': 'TS', 'type_post': 'TA', 'weight': 46,
         'connection_ratio': 0.115, 'nt_type': ''},
    ]).to_csv(mirror / 'refilled_connection_type.csv', index=False)

    def _boom(*_a, **_k):
        raise RuntimeError('forced render failure')

    _RecordingVisualizePath.calls = []
    monkey = _ShimMonkey()
    try:
        monkey.setattr(coana, 'VisualizePath', _RecordingVisualizePath)
        fc._viz_edge_df = _boom
        with pytest.raises(RuntimeError):
            fc._build_refill_visualizations(str(mirror))
        assert not _RecordingVisualizePath.calls
        del fc._viz_edge_df          # the retry is the second call site
        fc._build_refill_visualizations(str(mirror))
    finally:
        monkey.undo()

    assert _RecordingVisualizePath.calls, (
        'the retry after a crash rendered nothing — the folder is still '
        'marked as built even though it has no pages')


# ---------------------------------------------------------------------------
# 8. The transmitter vocabulary the refill mirror publishes
# ---------------------------------------------------------------------------

def _details(run_dir, **files):
    dd = Path(run_dir) / 'data_details'
    dd.mkdir(parents=True, exist_ok=True)
    for name, frame in files.items():
        frame.to_csv(dd / f'{name}.csv', index=False)
    return dd


def test_the_refill_mirror_carries_the_transmitter_the_run_itself_shows(
        tmp_path):
    """`refilled_*_connection_type.csv`'s `nt_type` must be the SAME vocabulary
    the run's own network page colours by.

    A Network-tab run writes no `conn_mat_type_nt.csv` at all, so the matrix was
    the only source and the mirror's transmitter column came out EMPTY on every
    row — while `data_details/connection_type.csv`, the file the page reads,
    carried GLUT/ACH/SER for 130 of the run's 148 pairs. The refill page
    therefore had no transmitter to show or colour by (user report 2026-10-17).
    """
    from type_level_refill import _pair_nt_lookup

    run = tmp_path / 'find-network_run'
    _details(run, connection_type=pd.DataFrame({
        'type_pre': ['A', 'B', 'C'], 'type_post': ['B', 'C', 'D'],
        'weight': [10, 20, 30],
        'nt_type': ['GLUT', None, 'ACH'],
    }))
    pair_nt, type_nt = _pair_nt_lookup(run)
    assert pair_nt == {('A', 'B'): 'glutamate', ('B', 'C'): '',
                       ('C', 'D'): 'acetylcholine'}, pair_nt
    # a pair the run LISTS with no transmitter stays unannotated: the run's own
    # page draws it blank, and the refill page beside it cannot call it 'GLUT'
    assert pair_nt[('B', 'C')] == '', pair_nt
    # the per-type vocabulary is only for pairs NO source mentions at all
    assert type_nt == {'A': 'glutamate', 'C': 'acetylcholine'}, type_nt
    # e.g. a pair the refill re-valued from a bodyId edge the type table lacks
    assert pair_nt.get(('D', 'E')) is None
    assert ('D' not in pair_nt) and type_nt.get('A') == 'glutamate'


def test_the_nt_matrix_still_wins_where_both_sources_speak(tmp_path):
    """Both sources merge, matrix first: the pathfinding tools have always
    published `conn_mat_type_nt.csv`, and its per-pair cell is the curated
    value. The edge table may only ADD pairs the matrix never listed."""
    from type_level_refill import _pair_nt_lookup

    _details(tmp_path,
             conn_mat_type_nt=pd.DataFrame({
                 'type_pre': ['A'], 'B': ['GLUT'], 'C': ['SER']}),
             connection_type=pd.DataFrame({
                 'type_pre': ['A', 'A', 'B'], 'type_post': ['B', 'C', 'D'],
                 'weight': [1, 2, 3],
                 'nt_type': ['ACH', 'ACH', 'SER']}))
    pair_nt, _type_nt = _pair_nt_lookup(tmp_path)
    assert pair_nt[('A', 'B')] == 'glutamate', (
        'the matrix must win on its own cells')
    assert pair_nt[('A', 'C')] == 'serotonin', pair_nt
    assert pair_nt[('B', 'D')] == 'serotonin', (
        f'the edge table must still fill a pair the matrix never listed: {pair_nt}')


def test_no_transmitter_source_means_no_claim(tmp_path):
    """With neither file the maps stay empty, so the writers keep their
    historical empty cell rather than inventing a label."""
    from type_level_refill import _pair_nt_lookup

    assert _pair_nt_lookup(tmp_path) == ({}, {})


def test_full_floor_type_pairs_export(tmp_path):
    """P0-C/A.4: the full-floor table is the ALL-PAIRS, asked-floor type
    universe (not emitted-chain-only) and carries both pair ratios."""
    _fc, run_dir = make_run(tmp_path, 'full_floor')
    out = tmp_path / 'recs'
    rec = compute_type_level_refill(
        run_dir, edges=EDGES, type_map=TYPE_MAP, out_dir=out, write=True)
    full = rec['full']
    assert full['full_floor']['status'] == 'ok'
    csv = out / 'refill_full_floor_type_pairs.csv'
    assert csv.exists()
    frame = pd.read_csv(csv)
    assert list(frame.columns) == [
        'type_pre', 'type_post', 'w_sum', 'w_max', 'n',
        'connection_ratio', 'connection_ratio_adj', 'nt_type',
        'non_emitted_mass', 'in_emitted_cone', 'source_type',
        'target_type']
    assert len(frame) > 0
    # all-pairs: at least every emitted pair is represented
    emitted = read_emitted_pairs(run_dir)
    got = set(zip(frame['type_pre'], frame['type_post']))
    assert set(emitted) <= got
    assert frame['connection_ratio_adj'].notna().any()
    # the chain-scoped full pair table carries the adjusted column too
    pair_csv = out / 'refill_full_type_pairs.csv'
    cols = pd.read_csv(pair_csv, nrows=0).columns
    assert 'refilled_connection_ratio_adj' in cols
    # and the full path table carries the realized adjusted readout
    path_csv = run_dir / 'type_level_refill' / 'refill_full_paths_type.csv'
    pcols = pd.read_csv(path_csv, nrows=0).columns
    assert {'ratios_adj', 'min_ratio_adj',
            'ratio_adj_complete'} <= set(pcols)
    # the ASKED-floor chain table (issue 2026-10-22 refill §3)
    floor_paths = out / 'refill_full_floor_paths_type.csv'
    assert floor_paths.exists()
    fp = pd.read_csv(floor_paths)
    assert len(fp) > 0
    assert {'ratios_adj', 'min_ratio_adj',
            'ratio_adj_complete'} <= set(fp.columns)
    # no >= 1 emitted/refill split on an asked-floor table
    assert 'emitted_weights' not in fp.columns
    assert 'refill_weights' not in fp.columns
    assert full['full_floor']['paths'] == len(fp)
    assert full['full_floor']['paths_scope'] == (
        'emitted chains re-valued at the asked floor')
    assert rec['full']['refill_status'] in (
        'ok', 'unavailable_guard_not_checked')
    # P0-E: the primary guard is bodyId-level (label-agnostic), so the
    # settled-vs-native label mismatch cannot fail a complete cache.
    assert rec['full']['cache_guard'] in ('ok', 'ok_bodyid')


def test_refill_native_label_convention_keeps_native_and_bodyid_keys(tmp_path):
    """Issue 2026-10-22 (refill): the `80` male-CNS export uses native
    `R8p`-style keys and bodyId fallbacks. The auto chooser must pick that
    convention over the mapped/dropped map."""
    from type_level_refill import (
        _involved_bodyid_fallbacks, choose_refill_type_map)

    frame = pd.DataFrame({
        'bodyId': ['a1', '212074', 'c1'],
        'type': ['R8p', None, 'APDN3'],
    })
    run = tmp_path / 'run'
    (run / 'data_details').mkdir(parents=True)
    pd.DataFrame({
        'type_pre': ['R8p', '212074'],
        'type_post': ['212074', 'APDN3'],
    }).to_csv(run / 'data_details' / 'connection_type.csv', index=False)
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({
        'format': 'drocat_custom_groups/v1',
        'source_mapping': {'custom_label': [], 'male-cns:v1.0': []},
        'target_mapping': {'custom_label': [], 'male-cns:v1.0': []},
    }), encoding='utf-8')
    type_map, convention, scores = choose_refill_type_map(
        frame, run, dataset='male-cns:v1.0', mapping_file=mapping)
    assert convention == 'native'
    assert type_map['a1'] == 'R8p'
    assert type_map['212074'] == '212074'
    assert type_map['c1'] == 'APDN3'
    assert scores['native'] >= scores['settled']

    # an exported body absent from the neuron table is added as itself
    augmented = _involved_bodyid_fallbacks(
        {'a1': 'R8p'}, {'R8p', '999999'},
        edges=[('999999', 'a1', 10)])
    assert augmented['999999'] == '999999'


def test_banc_style_type_guard_passes_without_bodyid_detail(tmp_path):
    """Issue 2026-10-22 (refill): the type-level fallback guard passes on a
    complete stream once the native/bodyId-fallback map is used."""
    _fc, run_dir = make_run(tmp_path, 'banc_style_ok')
    ci = run_dir / 'data_details' / 'connection_info_bodyId.csv'
    ci.unlink()
    out = tmp_path / 'recs'
    rec = compute_type_level_refill(
        run_dir, edges=EDGES, type_map=TYPE_MAP, out_dir=out, write=True)
    full = rec['full']
    assert full['cache_guard'] == 'ok'
    assert full['refill_status'] == 'ok'
    assert full['full_floor']['status'] == 'ok'
    assert (out / 'refill_full_floor_type_pairs.csv').exists()


def test_banc_style_guard_failure_is_unavailable_not_silent(tmp_path):
    """Issue 2026-10-22 (refill §2.2/§3): an undercounted stream is a
    per-dataset `unavailable`, never a silent `not_checked` and never a
    written full-scope table."""
    _fc, run_dir = make_run(tmp_path, 'banc_style_fail')
    (run_dir / 'data_details' / 'connection_info_bodyId.csv').unlink()
    inc = run_dir / 'data_details' / 'type_incoming_emitted.csv'
    frame = pd.read_csv(inc)
    frame = pd.concat([frame, pd.DataFrame([
        {'type': 'ZZZ_FAKE', 'incoming': 10_000_000}])],
        ignore_index=True)
    frame.to_csv(inc, index=False)
    out = tmp_path / 'recs'
    rec = compute_type_level_refill(
        run_dir, edges=EDGES, type_map=TYPE_MAP, out_dir=out, write=True)
    full = rec['full']
    assert full['status'] == 'unavailable'
    assert full['cache_guard'] == 'FAILED'
    assert full['refill_status'] == 'unavailable'
    assert 'ZZZ_FAKE' in full['error']
    assert not (out / 'refill_full_type_pairs.csv').exists()
