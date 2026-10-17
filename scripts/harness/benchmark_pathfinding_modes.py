"""Benchmark cell runner for the three pathfinding modes.

Design: _plan/plan-2026-10-15-pathfinding-benchmark.md, decisions
D1 (^DN whole, DNES* included), D2 (one UNION target set), D3 (path budget
100,000), D4 (depths 0..3 first; 0..9 for m2/m3 later), D5 (pilot first).

Modes (plan section 1):
    m1   free        pathfinding='MemoizedDFS', max_paths_bodyid=None,
                     graph_edge_limit_bodyid=None -> trim_policy
                     'unbounded_complete' (coana.py:20214-20216, 20333-20342)
    m2a  budgeted   StrongestFirst + Edge Budget 1_000_000 + path budget
                     100_000, refill OFF
    m2b  + refill   m2a with auto_type_level_refill=True. The refill round
                     publishes counters but NO timers (type_level_refill.py),
                     so its cost is t(m2b) - t(m2a): the same query at the
                     same depth, so the difference IS the round.
    m3   corridor   corridor_mode='fast': the lane is the deliverable and
                     the Complete pipeline is skipped (coana.py:19289-19318).

Isolation: ONE SUBPROCESS PER CELL, so every engine cache
(_FINDALLPATH_GRAPH_CACHE, _FNC_CACHE, neuron caches) starts cold and
ru_maxrss is per-cell. A wall-clock cap (SIGALRM) makes a capped cell report
a LOWER BOUND plus a stop reason in the corridor lane's own style
(route_plan.py:509-539: capped/stalled/pop_capped = "emptiness unverified,
never an absence claim"), never a count.

usage (parent, runs the matrix and appends JSONL):
    python scripts/harness/benchmark_pathfinding_modes.py \
        --modes m2a,m2b,m3,m1 --depths 0,1,2,3 --cap 900 \
        --caps m1:0=300,m1:1=1800,m1:2=900,m1:3=900
single cell (child, spawned by the parent):
    python ... --cell m2a:3 --cap 900
"""
import argparse
import json
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

WORK = Path('/tmp/drocat-bench-20261015')
DATASET = 'male-cns:v1.0'
MIN_SYNAPSE = 3                       # the baseline threshold; also the lane's `asked`
EDGE_BUDGET = 1_000_000               # UI default
PATH_BUDGET = 100_000                 # decision D3
SRC_NAME = 'ORN53'
TGT_NAME = 'OlfPN_DN_LH_1103'

MODES = {
    'm1': dict(pathfinding='MemoizedDFS', max_paths_bodyid=None,
               graph_edge_limit_bodyid=None, auto_type_level_refill=False,
               corridor_mode='complete'),
    'm2a': dict(pathfinding='StrongestFirst', max_paths_bodyid=PATH_BUDGET,
                graph_edge_limit_bodyid=EDGE_BUDGET,
                auto_type_level_refill=False, corridor_mode='complete'),
    'm2b': dict(pathfinding='StrongestFirst', max_paths_bodyid=PATH_BUDGET,
                graph_edge_limit_bodyid=EDGE_BUDGET,
                auto_type_level_refill=True, corridor_mode='complete'),
    'm3': dict(pathfinding='StrongestFirst', max_paths_bodyid=PATH_BUDGET,
               graph_edge_limit_bodyid=EDGE_BUDGET,
               auto_type_level_refill=False, corridor_mode='fast'),
}


class WallClock(Exception):
    pass


def _alarm(signum, frame):
    raise WallClock('wall-clock cap reached')


COUNT_RE = [(re.compile(r'Total paths found:\s*([\d,]+)'), 'paths_bodyid'),
            (re.compile(r'Found and saved\s+([\d,]+)\s+type-level paths'),
             'paths_type')]
# The engine streams live progress with a running path counter, e.g.
#   "L3 Reconstruct: 348/2635 (13.2%) [08:46<57:41, 0.7it/s] | paths=453,348,660"
# A CAPPED cell has no final counts, so this is the only lower bound there is
# -- and its ETA is the engine's own extrapolation (linear in sources).
PROGRESS_RE = re.compile(
    r'^(?P<tag>\S+ Reconstruct): (?P<done>\d+)/(?P<total>\d+) '
    r'\((?P<pct>[\d.]+)%\) \[(?P<elapsed>[^<]*)<(?P<eta>[^,]*),[^\]]*\]'
    r'\s*\|\s*paths=(?P<paths>[\d,]+)')


def _last_progress(text):
    """The engine prints progress with '\r', so scan every carriage return."""
    last = None
    for line in re.split(r'[\r\n]', text):
        m = PROGRESS_RE.search(line.strip())
        if m:
            last = m
    if last is None:
        return {}
    g = last.groupdict()
    paths = int(g['paths'].replace(',', ''))
    done, total = int(g['done']), int(g['total'])
    out = {'progress_tag': g['tag'], 'progress_sources_done': done,
           'progress_sources_total': total, 'paths_lower_bound': paths,
           'progress_eta_engine': g['eta'].strip()}
    if done and total:
        out['paths_extrapolated_linear'] = int(round(paths * total / done))
    return out


def run_cell(mode, depth, cap, panel, out_dir):
    """Child: one engine run, then read back its own artifacts."""
    import re
    from coana import FindNeuronConnection          # noqa: E402
    src = panel['panel']['source']['types']
    tgt = panel['panel']['targets']['union_types']
    knobs = dict(MODES[mode])
    run_dir = out_dir / f'{mode}_L{depth}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'engine_stdout.log'
    record = {'mode': mode, 'max_interlayer': depth, 'hops_max': depth + 1,
              'panel_digest': panel['panel']['digest'],
              'min_synapse': MIN_SYNAPSE,
              'edge_budget': knobs.get('graph_edge_limit_bodyid'),
              'path_budget': knobs.get('max_paths_bodyid'),
              'corridor_mode': knobs['corridor_mode'],
              'refill': knobs['auto_type_level_refill'],
              'warm': 'cold-fresh-process', 'cap_s': cap,
              'run_dir': str(run_dir)}

    fc = FindNeuronConnection(
        output_dir=str(run_dir), dataset=DATASET,
        sourceNeurons=list(src), targetNeurons=list(tgt),
        custom_source_name=SRC_NAME, custom_target_name=TGT_NAME,
        custom_source_group_names=[], custom_target_group_names=[],
        min_synapse_num=MIN_SYNAPSE, min_ratio=0.0,
        min_traversal_probability=0, filter_by='bodyId', showfig=False,
        max_interlayer=depth, keyword_in_path_to_remove=['None'],
        use_cache=True, edgeN_limit=500, output_format='csv',
        skip_bodyId=True, drop_untyped=True, separate_hemispheres=False,
        hemisphere_filter='both', **knobs)

    t0 = time.perf_counter()
    status, note = 'complete', ''
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(cap)
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            sys.stdout = log
            try:
                fc.InitializeNeuronInfo()
                fc.FindAllPath(forward_only=True)
            finally:
                sys.stdout = sys.__stdout__
    except WallClock as exc:
        status, note = 'capped', str(exc)
    except Exception as exc:                              # noqa: BLE001
        status, note = 'error', f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    record['wall_s'] = round(time.perf_counter() - t0, 2)
    record['status'] = status
    if note:
        record['note'] = note
    # macOS reports ru_maxrss in BYTES (Linux: kilobytes); the existing
    # scripts/harness/*.py use /1024 and are 1024x off here.
    record['rss_peak_mb'] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 1)
    record.update(_collect(run_dir, log_path))
    record['estimate'] = _estimate(record, panel)
    (run_dir / 'cell.json').write_text(json.dumps(record, indent=2),
                                       encoding='utf-8')
    print(json.dumps(record))
    return 0 if status == 'complete' else (3 if status == 'capped' else 4)


def _collect(run_dir, log_path):
    """Structured read-back; every field names the artifact it came from."""
    out = {}
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
        for rx, key in COUNT_RE:
            m = rx.search(text)
            if m:
                out[key] = int(m.group(1).replace(',', ''))
        if 'paths_bodyid' not in out:
            # capped/aborted: fall back to the engine's live progress line
            out.update(_last_progress(text))
    except OSError:
        pass
    base = Path(run_dir)
    folders = [base] + [p for p in base.iterdir() if p.is_dir()]
    for f in folders:
        attrs = f / 'all_attributes.json'
        if attrs.exists() and 'phase_timers' not in out:
            try:
                a = json.loads(attrs.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            out['phase_timers'] = a.get('phase_timers') or {}
            for k in ('graph_pruning_record', 'trim_policy',
                      'strongest_first_tau', 'tau_canonical',
                      'strongest_dropped_bottleneck',
                      'strongest_retained_bottleneck',
                      'strongest_first_budget_bitten', 'paths_complete',
                      'strongest_first_budget', 'edge_weight_floor',
                      'edge_budget_landing', 'applied_threshold',
                      'requested_threshold', 'edgeN_limit_reached'):
                if k in a:
                    out[k] = a[k]
        rp = f / 'route_scope_paths' / 'route_provenance.json'
        if rp.exists():
            try:
                p = json.loads(rp.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            cls = p.get('classes') or {}
            out['lane'] = {
                k: p.get(k) for k in
                ('status', 'runtime_s', 'phases_s', 'time_capped',
                 'time_budget_s', 'asked', 'bound', 'validated_walks',
                 'scope_pairs', 'bodyId_paths', 'weight_basis',
                 'gap_partition', 'plan_hash')}
            stops = {}
            for v in cls.values():
                s = str((v or {}).get('stop'))
                stops[s] = stops.get(s, 0) + 1
            out['lane_classes'] = {
                'n_classes': len(cls), 'stops': stops,
                'ties_truncated': sum(
                    int((v or {}).get('ties_truncated') or 0)
                    for v in cls.values()),
                'used_max': max([int((v or {}).get('used') or 0)
                                 for v in cls.values()] or [0])}
        rf = (f / 'data_details' / 'type_level_refill'
              / 'refill_provenance.json')
        if rf.exists():
            try:
                r = json.loads(rf.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            full = r.get('full') or {}
            out['refill'] = {
                'status': r.get('status'),
                'table_reproduced': r.get('table_reproduced'),
                'asked_edges': r.get('refill_edges'),
                'asked_weight': r.get('refill_weight_total'),
                'full_chains': full.get('chains'),
                'full_edges': full.get('refill_edges'),
                'full_weight': full.get('refill_weight_total'),
                'full_revisit': full.get('chains_revisit'),
                'cache_guard': r.get('cache_guard')}
    return out


def _estimate(rec, panel):
    """A capped cell measures a PARTIAL run: lower bound + model total, and
    the rates the prediction table needs (plan section 3.3/6)."""
    ladder = {r['max_interlayer']: r for r in panel['edge_ladder']}
    row = ladder.get(rec['max_interlayer'], {})
    est = {'model_edges': row.get('cum_distinct_edges')}
    timers = rec.get('phase_timers') or {}
    n = rec.get('paths_bodyid')
    if n:
        if timers.get('enumeration_s'):
            est['rho_enum_paths_per_s'] = round(
                n / timers['enumeration_s'], 1)
        if timers.get('materialization_s'):
            est['rho_mat_paths_per_s'] = round(
                n / timers['materialization_s'], 1)
    if rec['status'] == 'capped':
        est['certified'] = False
        est['stop_reason'] = rec.get('note', 'wall-clock cap reached')
        est['paths_lower_bound'] = (rec.get('paths_bodyid')
                                    or rec.get('paths_lower_bound'))
        est['cap_overrun_s'] = round(
            (rec.get('wall_s') or 0) - (rec.get('cap_s') or 0), 1)
        exact = panel.get('exact_walks') or {}
        est['model_walks_hops_le_L+1'] = sum(
            int(exact.get(f'hop{h}', 0) or 0)
            for h in range(1, rec['max_interlayer'] + 2))
    return est


def _parse_caps(spec):
    """Keys are 'mode' or 'mode:depth', values follow '='. (Splitting on ':'
    first turned 'm1:0=300' into int('0=300') and killed a whole run.)"""
    caps = {}
    for item in (spec or '').split(','):
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        caps[key.strip()] = int(value)
    return caps


def parent(a):
    panel = json.loads(Path(a.panel).read_text(encoding='utf-8'))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch(exist_ok=True)
    depths = [int(x) for x in a.depths.split(',') if x != '']
    caps = _parse_caps(a.caps)
    for mode in a.modes.split(','):
        for d in depths:
            cap = caps.get(f'{mode}:{d}', caps.get(mode, a.cap))
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   '--cell', f'{mode}:{d}', '--cap', str(cap),
                   '--panel', str(a.panel), '--work', str(WORK)]
            t0 = time.time()
            print(f'>>> {mode} L={d} cap={cap}s', flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in (proc.stdout or '').splitlines()
                     if ln.startswith('{')]
            with out.open('a', encoding='utf-8') as fh:
                if lines:
                    try:
                        fh.write(json.dumps(json.loads(lines[-1])) + '\n')
                    except ValueError:
                        fh.write(json.dumps(
                            {'mode': mode, 'max_interlayer': d,
                             'status': 'error',
                             'note': 'unparsable child json',
                             'stderr': (proc.stderr or '')[-800:]}) + '\n')
                else:
                    fh.write(json.dumps(
                        {'mode': mode, 'max_interlayer': d,
                         'status': 'error',
                         'note': (proc.stderr or '')[-800:]}) + '\n')
            tail = (lines[-1][:220] if lines
                    else (proc.stderr or '')[-220:])
            print(f'    rc={proc.returncode} in {time.time() - t0:.0f}s {tail}',
                  flush=True)
    print(f'results -> {out}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=str(WORK / 'panel.json'))
    ap.add_argument('--modes', default='m2a,m2b,m3,m1')
    ap.add_argument('--depths', default='0,1,2,3')
    ap.add_argument('--cap', type=int, default=900)
    ap.add_argument('--caps', default='',
                    help='per-cell overrides: m1:1=1800,m1:2=900')
    ap.add_argument('--out', default=str(WORK / 'results.jsonl'))
    ap.add_argument('--cell', help='internal: MODE:DEPTH')
    ap.add_argument('--work', default=str(WORK))
    a = ap.parse_args()
    if a.cell:
        mode, depth = a.cell.split(':')
        return run_cell(mode, int(depth), a.cap,
                        json.loads(Path(a.panel).read_text(encoding='utf-8')),
                        Path(a.work) / 'runs')
    return parent(a)


if __name__ == '__main__':
    sys.exit(main())
