"""Phase 2 — the in-pipeline connection-ratio lane
(plan-connection-ratio-pathfinding §12-§16).

Keystone (§16.2 consistency pin): a real-pipeline ratio run
(``weight_basis='connection_ratio'``) must reproduce the standalone
lane's path SET with identical per-path ratio bottlenecks — same
division on both sides of the pipeline. Everything else pins the frame
surfaces: parameters/provenance stamps, folder grammar, the synapse
golden (the basis guards cost synapse runs nothing), the cache-key
basis segment, shortest's refusal, the Edge Budget ratio lane, and the
cross-dataset folder-token grammar.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC = PROJECT_ROOT / 'src'
for p in (PROJECT_ROOT, SRC, PROJECT_ROOT / 'vispath-subproject' / 'src',
          PROJECT_ROOT / 'tests' / 'core'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_type_level_refill import universe_types          # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402
from connection_ratio_paths import (                     # noqa: E402
    compute_incoming_totals,
    find_ratio_paths,
)
from coana import _findallpath_cache_key                   # noqa: E402

INT_EDGES = [
    ('S1', 'A', 60), ('S2', 'A', 30), ('S1', 'B', 20), ('S2', 'B', 35),
    ('A', 'C', 15), ('B', 'C', 30), ('S1', 'C', 40), ('A', 'D', 70),
    ('B', 'D', 10), ('C', 'T1', 40), ('D', 'T1', 25), ('E', 'T1', 30),
    ('C', 'T2', 15), ('D', 'T2', 45), ('E', 'T2', 35),
    ('B', 'E', 25), ('A', 'E', 15), ('D', 'E', 45),
    ('bg1', 'A', 10), ('bg2', 'B', 45), ('bg3', 'C', 15),
    ('bg4', 'D', 20), ('bg5', 'E', 15), ('bg6', 'T1', 5),
    ('bg7', 'T2', 5),
]
INT_TYPES = {'S1': 'SRC', 'S2': 'SRC', 'A': 'MID', 'B': 'MID',
             'C': 'INNER', 'D': 'INNER', 'E': 'INNER',
             'T1': 'SINK', 'T2': 'SINK'}
SOURCES = ['S1', 'S2']
TARGETS = ['T1', 'T2']


def _run(tmp_path, name, *, basis='synapse', t_r=0.2, edge_budget=0,
         path_budget=0, asked=1):
    with universe_types(INT_TYPES) as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path / name, edges=INT_EDGES, max_interlayer=3,
            min_synapse=asked, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        if 'Checked' not in fc.target_df.columns:
            fc.target_df = fc.target_df.assign(
                Checked=[True] * len(fc.target_df))
        fc.graph_edge_limit_bodyid = edge_budget
        fc.max_paths_bodyid = path_budget
        fc.weight_basis = basis
        fc.min_ratio = t_r
        fc.parameter_dict.update({
            'min synapse number': str(asked), 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': '3',
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc.FindAllPath()
        return fc, Path(fc.allpath_folder)


def _paths_frame(run_dir):
    return pd.read_csv(
        next(run_dir.glob('*_allpaths_bodyId_paths.csv')))


def _param(text, key):
    for line in text.splitlines():
        if line.split(':', 1)[0].strip() == key:
            return line.split(':', 1)[1].strip()
    return None


def _true_bn(path):
    from connection_ratio_paths import compute_incoming_totals
    totals = compute_incoming_totals(INT_EDGES)
    w = {(u, v): float(x) for u, v, x in INT_EDGES}
    ns = path.split('->')
    return round(min(w[(ns[i], ns[i + 1])] / totals[ns[i + 1]]
                     for i in range(len(ns) - 1)), 9)


# ---------------------------------------------------------------------------
# 1. The keystone: pipeline ratio run == standalone lane
# ---------------------------------------------------------------------------
def test_ratio_pipeline_matches_standalone_lane(tmp_path):
    ref = find_ratio_paths(INT_EDGES, SOURCES, TARGETS, min_ratio=0.2,
                           max_interlayer=3, type_map=None,
                           drop_untyped=False)
    _fc, run_dir = _run(tmp_path, 'keystone', basis='connection_ratio')
    paths = _paths_frame(run_dir)
    got = {(_true_bn(r.path), tuple(r.path.split('->')))
           for r in paths.itertuples()}
    want = {(round(p['ratio_bottleneck'], 9), tuple(p['nodes']))
            for p in ref['paths']}
    assert got == want
    # bottlenecks non-negative multiples of the threshold: every path
    # cleared t_r
    assert all(bn >= 0.2 for bn, _p in got)


# ---------------------------------------------------------------------------
# 2. Run artifacts: parameters/provenance/folder/notes/disclosure
# ---------------------------------------------------------------------------
def test_ratio_run_artifacts(tmp_path):
    _fc, run_dir = _run(tmp_path, 'artifacts', basis='connection_ratio')
    assert '_L3r0_2_' in run_dir.name
    text = (run_dir / 'parameters.txt').read_text(encoding='utf-8')
    assert _param(text, 'weight basis') == 'connection_ratio (bodyId)'
    assert _param(text, 'min connection ratio') == '0.2'
    assert _param(text, 'requested_threshold') == '0.2'
    assert _param(text, 'paths_complete') == 'True'
    attrs = __import__('json').loads(
        (run_dir / 'all_attributes.json').read_text(encoding='utf-8'))
    assert attrs.get('weight_basis') == 'connection_ratio'
    syn = run_dir / 'data_details' / 'ratio_synapse_map.csv'
    assert syn.exists()
    rows = pd.read_csv(syn)
    assert {'bodyId_post', 'total_incoming', 'implied_syn_cutoff',
            'kept_in_edges'} <= set(rows.columns)
    # every post's implied cutoff = max(1, ceil(0.2 * total))
    import math
    for r in rows.itertuples():
        assert r.implied_syn_cutoff == max(
            1, math.ceil(0.2 * r.total_incoming))
    notes = (run_dir / 'user_warning_notes.txt').read_text(
        encoding='utf-8')
    assert '[weight basis]' in notes
    assert '[ratio definition]' not in notes
    # the type-level refill never fires on a ratio run (§16.1 guard)
    assert not (run_dir / 'data_details' / 'type_level_refill').exists()


def test_ratio_bad_threshold_refused(tmp_path):
    with universe_types(INT_TYPES) as shim:
        fc, _c, _l = _make_pipeline_fc(
            shim, tmp_path, edges=INT_EDGES, max_interlayer=3,
            min_synapse=1, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 1.5
        with pytest.raises(ValueError, match='min_ratio'):
            fc.FindAllPath()


# ---------------------------------------------------------------------------
# 3. Synapse golden: the basis guards change nothing for synapse runs
# ---------------------------------------------------------------------------
def test_synapse_golden_explicit_vs_default(tmp_path):
    _fc, default_dir = _run(tmp_path, 'gold_default', asked=3)
    _fc2, explicit_dir = _run(tmp_path, 'gold_explicit', asked=3,
                              basis='synapse')
    for rel in ('data_details/connection_type.csv',
                'data_details/connection_info_bodyId.csv',
                'parameters.txt'):
        a = (default_dir / rel).read_bytes()
        b = (explicit_dir / rel).read_bytes()
        assert a == b, rel
    assert '_L3w3_' in default_dir.name


# ---------------------------------------------------------------------------
# 4. Graph-cache key carries the basis segment
# ---------------------------------------------------------------------------
def test_cache_key_basis_segment():
    kw = dict(dataset_safe='DS', source_ID=['a'], target_ID=['b'],
              max_interlayer=2, separate_hemispheres=False,
              filter_by='bodyId', min_ratio=0.0,
              min_traversal_probability=0.0,
              exclude_intra_type_connections=False)
    synapse = _findallpath_cache_key(**kw)
    ratio = _findallpath_cache_key(weight_basis='connection_ratio', **kw)
    assert synapse != ratio
    assert 'basis-connection_ratio' in ratio
    assert 'basis-synapse' in synapse


# ---------------------------------------------------------------------------
# 5. Edge Budget ratio lane: floor ≡ complete-at-tier
# ---------------------------------------------------------------------------
def test_ratio_edge_budget_floor(tmp_path):
    _fc, floored = _run(tmp_path, 'budgeted', basis='connection_ratio',
                        edge_budget=9)
    prov_txt = (floored / 'parameters.txt').read_text(encoding='utf-8')
    floor = float(_param(prov_txt, 'edge_weight_floor'))
    assert floor > 0.2 and floor <= 0.45
    assert f'{floor:g}' == _param(prov_txt, 'edge_weight_floor')
    got = {(_true_bn(r.path), tuple(r.path.split('->')))
           for r in _paths_frame(floored).itertuples()}
    _fc2, at_floor = _run(tmp_path, 'at_floor', basis='connection_ratio',
                          t_r=floor)
    want = {(_true_bn(r.path), tuple(r.path.split('->')))
            for r in _paths_frame(at_floor).itertuples()}
    assert got == want
    assert 'edge_budget' in (_param(prov_txt, 'applied_threshold_source')
                             or '')
    assert _param(prov_txt, 'paths_complete') == 'False'


# ---------------------------------------------------------------------------
# 6. Cross-dataset folder-token grammar (comparison_parameters)
# ---------------------------------------------------------------------------
def test_cross_dataset_threshold_tokens():
    from comparison.comparison_parameters import ComparisonParameters
    params = ComparisonParameters(
        datasets=['A', 'B'], source_neurons={}, target_neurons={})
    assert params._threshold_token(3) == 'minsyn_3'
    assert params.get_dataset_output_path('A', 3).endswith('minsyn_3')
    params.weight_basis = 'connection_ratio'
    assert params._threshold_token(0.0005) == 'minratio_0_0005'
    assert params._threshold_token(0.2) == 'minratio_0_2'
    assert params.get_dataset_output_path(
        'A', 0.0005).endswith('minratio_0_0005')
    cands = params.applied_folder_name_candidates(0.2)
    assert cands[0] == 'minratio_0_2'
    assert cands[1] == 'minratio_0_2_applied_floor'
    assert params.skipped_folder_name(0.2) == 'minratio_0_2_skipped'


# ---------------------------------------------------------------------------
# 7. Real-data gate (the §7.6 analogue): a real in-pipeline ratio run
# ---------------------------------------------------------------------------
_FAFB_DIR = PROJECT_ROOT / 'datasets' / 'flywire_FAFB_v783'
_FAFB_GATED = pytest.mark.skipif(
    not (_FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet').exists(),
    reason='local FAFB v783 release not present')


@_FAFB_GATED
def test_real_fafb_ratio_pipeline(tmp_path):
    """Real FAFB ratio run through the REAL pipeline: the exported
    min_ratio column must equal the independently recomputed F9
    bottleneck for every path (the §16.2 keystone on real data)."""
    import polars as pl
    from coana import FindNeuronConnection

    fc = FindNeuronConnection(
        output_dir=str(tmp_path), dataset='flywire_FAFB_v783',
        sourceNeurons=['aMe12'], targetNeurons=['PPL101'],
        custom_source_name='', custom_target_name='',
        custom_source_group_names=[], custom_target_group_names=[],
        min_synapse_num=1, min_ratio=0.001, min_traversal_probability=0,
        weight_basis='connection_ratio',
        filter_by='bodyId', showfig=False, max_interlayer=2,
        keyword_in_path_to_remove=['None'], network_layout='distributed',
        use_cache=True, edgeN_limit=500, output_format='csv',
        pathfinding='StrongestFirst', drop_untyped=True, skip_bodyId=False,
        max_paths_bodyid=0)
    fc.InitializeNeuronInfo()
    fc.FindAllPath(forward_only=True)
    run_dir = Path(fc.allpath_folder)
    assert '_L2r0_001_' in run_dir.name

    paths = pd.read_csv(next(run_dir.glob('*_allpaths_bodyId_paths.csv')))
    assert len(paths) > 0
    conn = pl.read_parquet(
        _FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet',
        columns=['bodyId_pre', 'bodyId_post', 'weight'])
    tot = conn.group_by('bodyId_post').agg(
        pl.col('weight').cast(pl.Float64).sum().alias('t'))
    totals = {str(k): v for k, v in
              zip(tot['bodyId_post'].to_list(), tot['t'].to_list())}
    w = {(str(u), str(v)): float(x) for u, v, x in
         zip(conn['bodyId_pre'], conn['bodyId_post'], conn['weight'])}
    for r in paths.itertuples():
        ns = r.path.split('->')
        bn = min(w[(ns[i], ns[i + 1])] / totals[ns[i + 1]]
                 for i in range(len(ns) - 1))
        assert abs(bn - float(r.min_ratio)) < 1e-9, r.path
    text = (run_dir / 'parameters.txt').read_text(encoding='utf-8')

    def param(key):
        for line in text.splitlines():
            if line.split(':', 1)[0].strip() == key:
                return line.split(':', 1)[1].strip()
        return None

    assert param('weight basis') == 'connection_ratio (bodyId)'
    assert param('requested_threshold') == '0.001'
    assert param('paths_complete') == 'True'
    assert (run_dir / 'data_details' / 'ratio_synapse_map.csv').exists()
    assert not (run_dir / 'data_details' / 'type_level_refill').exists()


# ---------------------------------------------------------------------------
# 8. Mediant guarantee end-to-end: type-level ratios clear t_r when every
#    bodyId pair does (per-pair INVOLVED-post denominators, §2 stage 8)
# ---------------------------------------------------------------------------
def test_ratio_type_ratios_clear_threshold(tmp_path):
    edges = INT_EDGES + [('bgx', 'bg8', 50)]   # bg8: typed MID, but no
    types = dict(INT_TYPES)                     # path ever emits into it
    types['bg8'] = 'MID'
    types['bgx'] = 'BG'
    with universe_types(types) as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path, edges=edges, max_interlayer=3,
            min_synapse=1, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        if 'Checked' not in fc.target_df.columns:
            fc.target_df = fc.target_df.assign(
                Checked=[True] * len(fc.target_df))
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.2
        fc.parameter_dict.update({
            'min synapse number': '1', 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': '3',
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc.FindAllPath()
        run_dir = Path(fc.allpath_folder)
    ct = pd.read_csv(run_dir / 'data_details' / 'connection_type.csv')
    assert len(ct) > 0
    offenders = ct[ct.connection_ratio < 0.2][
        ['type_pre', 'type_post', 'weight', 'connection_ratio']]
    assert offenders.empty, (
        f'type-level ratios below t_r (mediant violated):\n{offenders}')
    # the uninvolved member (bg8, 50 incoming) excluded: SRC->MID stays
    # the involved denominator (90+55=145 pre-bg8; with bg8 it would be 195)
    src_mid = ct[(ct.type_pre == 'SRC') & (ct.type_post == 'MID')]
    ratio = src_mid.connection_ratio.iloc[0]
    involved_den = src_mid.weight.sum() / ratio
    assert involved_den == pytest.approx(145.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 9. Phase 3: shortest-mode ratio lane
# ---------------------------------------------------------------------------
def _true_totals_frame(posts, min_weight=1, **_kwargs):
    totals = compute_incoming_totals(INT_EDGES)
    rows = [(str(p), totals.get(str(p), 0.0)) for p in posts]
    return pd.DataFrame(rows, columns=['bodyId_post',
                                       'total_incoming_weight'])


def test_shortest_ratio_lane(tmp_path):
    """Phase 3: FindShortestPath runs the ratio lane. The harness's
    backward-fetch stub bypasses the production fetch-level ratio filter,
    so this test runs the monolithic path and stubs the F9 global totals
    (what real runs read from the connection cache)."""
    import pandas as _pd

    types = dict(INT_TYPES)
    for i in range(1, 8):
        types[f'bg{i}'] = 'BG'   # fetched by the backward stub
    with universe_types(types) as shim:
        fc, _calls, _logs = _make_pipeline_fc(
            shim, tmp_path, edges=INT_EDGES, max_interlayer=3,
            min_synapse=1, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        if 'Checked' not in fc.target_df.columns:
            fc.target_df = fc.target_df.assign(
                Checked=[True] * len(fc.target_df))
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.2
        fc.discovery_batch_budget = 0     # monolithic path (the store path
        fc.target_batch_size = 0          # filters at the real fetch)
        fc._fetch_total_incoming_weight = _true_totals_frame
        fc.parameter_dict.update({
            'min synapse number': '1', 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': '3',
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        fc.FindShortestPath()
        run_dir = Path(fc.allpath_folder)
    assert '_L3r0_2_' in run_dir.name
    paths = _pd.read_csv(
        next(run_dir.glob('*_allpaths_bodyId_paths.csv')))
    assert len(paths) > 0
    bottlenecks = [(_true_bn(r.path), r.path) for r in paths.itertuples()]
    below = [b for b in bottlenecks if b[0] < 0.2]
    assert not below, below
    # shortest semantics: every emitted path is hop-minimal for its pair
    # (spot-check one known pair: S1 -> T1 via C is the 2-hop shortest)
    assert any(r.path == 'S1->C->T1' for r in paths.itertuples())
    text = (run_dir / 'parameters.txt').read_text(encoding='utf-8')
    assert 'weight basis' in text


# ---------------------------------------------------------------------------
# 10. Phase 3: replay with float ratio tiers
# ---------------------------------------------------------------------------
def test_ratio_replay_float_tiers(tmp_path):
    """FindAllPathMultiThreshold under the ratio lane: float tiers, slice
    materialization set-equal to fresh per-threshold runs, canonical
    collapse on a t0 budget bite, and r-suffix folder restamping."""
    import pandas as _pd

    def _make(shim, root):
        fc, _c, _l = _make_pipeline_fc(
            shim, root, edges=INT_EDGES, max_interlayer=3,
            min_synapse=1, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.skip_bodyId = False
        if 'Checked' not in fc.target_df.columns:
            fc.target_df = fc.target_df.assign(
                Checked=[True] * len(fc.target_df))
        fc.weight_basis = 'connection_ratio'
        fc.parameter_dict.update({
            'min synapse number': '1', 'filter by': 'bodyId',
            'exclude intra-type connections': 'False',
            'max interlayer': '3',
            'separate hemispheres': 'False', 'hemisphere filter': 'both',
            'aggregate method': 'product'})
        return fc

    types = dict(INT_TYPES)
    for i in range(1, 8):
        types[f'bg{i}'] = 'BG'
    with universe_types(types) as shim:
        fc = _make(shim, tmp_path / 'replay')
        fc.min_ratio = 0.1
        results = fc.FindAllPathMultiThreshold([0.1, 0.2, 0.3])
    root = Path(fc.allpath_folder).parent
    folders = sorted(root.glob('find-paths-complete_*'))
    names = [f.name for f in folders]
    # float r-suffix folders only (no minsyn_, no int collapse)
    assert all('_L3r' in n for n in names), names
    assert not any('minsyn' in n for n in names)
    # t0 collapsed onto its canonical (budget bit at t0=0.1)
    assert results[0.1]['skipped'] and results[0.1]['duplicate_of'] is not None
    canon = results[0.1]['duplicate_of']
    assert isinstance(canon, float) and canon > 0.1
    # slices set-equal to fresh per-threshold runs
    for t_r in (canon, 0.2, 0.3):
        with universe_types(types) as shim:
            ffc = _make(shim, tmp_path / f'fresh_{t_r}')
            ffc.min_ratio = t_r
            ffc.FindAllPath()
        fresh = Path(ffc.allpath_folder)
        want = {r.path for r in _pd.read_csv(
            next(fresh.glob('*_allpaths_bodyId_paths.csv'))).itertuples()}
        dec = str(t_r).replace('.', '_')
        match = [f for f in folders if f'r{dec}_' in f.name]
        assert match, (t_r, names)
        got = {r.path for r in _pd.read_csv(
            next(match[0].glob('*_allpaths_bodyId_paths.csv'))).itertuples()}
        assert got == want, (t_r, got ^ want)
    # per-slice provenance carries the replayed_from line + basis
    text = (match[0] / 'parameters.txt').read_text(encoding='utf-8')
    assert 'weight basis' in text and 'replayed_from' in text


def test_ratio_replay_refuses_bad_tiers(tmp_path):
    types = dict(INT_TYPES)
    for i in range(1, 8):
        types[f'bg{i}'] = 'BG'
    with universe_types(types) as shim:
        fc, _c, _l = _make_pipeline_fc(
            shim, tmp_path, edges=INT_EDGES, max_interlayer=3,
            min_synapse=1, source_ids=tuple(SOURCES),
            target_ids=tuple(TARGETS))
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.1
        with pytest.raises(ValueError, match='floats in'):
            fc.FindAllPathMultiThreshold([0.1, 5])
