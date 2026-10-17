"""Panel + edge-only ladder for the pathfinding benchmark.

Design: _plan/plan-2026-10-15-pathfinding-benchmark.md (decisions D1 keep
^DN whole incl. DNES*, D2 one UNION target set, D3 path budget 100k, D4
depths 0..3 now, D5 pilot first).

Reads ONLY the local MCNS tables — the engine is never imported. This is
the "edge list, no pathfinding" pass, and it produces three things the
benchmark cannot proceed without:

1. the frozen panel. The engine's dict filter supports
   contains/startswith/endswith/regex/exact with AND but NO negation
   (coana.py:3438-3458), and the panel needs `contains PN` MINUS `LPN` and
   `^DN` MINUS the clock neurons — so the families are resolved here and
   handed to the runner as explicit type names, with a digest recorded.
2. the admitted-edge ladder by depth (distinct bodyId pairs within hops
   <= L+1 from the source set) — benchmark plot 2, series (a).
3. the EXACT walk-count anchors at hops 1..3 (each distinct walk counted
   once, by summing over intermediate tuples). These are the model's truth
   points; gate B3 re-derives them.

usage:
    python scripts/harness/pathfinding_benchmark_panel.py \
        [--min-weight 3] [--depths 0,1,2,3,4,5,6,7,8,9] [--out panel.json]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET = 'male-cns:v1.0'
NEURON_CSV = (ROOT / 'datasets' / 'male-cns_v1_0'
              / 'male-cns_v1_0_allneurons_neuron_df.csv')
CONN_PARQUET = ROOT / 'cache' / 'male-cns_v1_0' / 'connections.parquet'
# D1: ^DN is kept whole (DNES1-3 included). D2: the three target families
# form ONE union set.
DN_EXCLUDE = {'DN1a', 'DN1pA', 'DN1pB'}          # dorsal clock neurons
SOURCE_RULE = ('^ORN', re.compile(r'^ORN').match)
TARGET_RULES = {
    'PN_olfactory': lambda t: 'PN' in t and 'LPN' not in t,
    'DN_descending': lambda t: bool(re.match(r'^DN', t))
                               and t not in DN_EXCLUDE,
    'LH_lateral_horn': lambda t: bool(re.match(r'^LH', t)),
}


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(sorted(p)).encode('utf-8'))
    return h.hexdigest()[:16]


def resolve_panel():
    import pandas as pd
    nd = pd.read_csv(NEURON_CSV, usecols=['bodyId', 'type'], dtype=str)
    nd = nd.dropna(subset=['type'])
    types = nd['type'].astype(str)
    bodies = nd['bodyId'].astype(str)

    def expand(pred):
        keep = sorted({t for t in types.unique() if pred(t)})
        return keep, sorted(bid for bid, t in zip(bodies, types) if t in keep)

    src_types, src_bodies = expand(SOURCE_RULE[1])
    fam_types, fam_bodies, fam_sizes = {}, {}, {}
    for name, pred in TARGET_RULES.items():
        tt, bb = expand(pred)
        fam_types[name], fam_bodies[name] = tt, bb
        fam_sizes[name] = {'types': len(tt), 'bodies': len(bb)}
    tgt_types = sorted(set().union(*fam_types.values()))
    tgt_bodies = sorted(set().union(*fam_bodies.values()))
    overlaps = {}
    names = list(TARGET_RULES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(fam_types[names[i]]) & set(fam_types[names[j]])
            if inter:
                overlaps[f'{names[i]}&{names[j]}'] = sorted(inter)[:8]
    return {
        'dataset': DATASET, 'min_synapse': None,   # filled by main()
        'source': {'rule': SOURCE_RULE[0], 'types': src_types,
                   'bodies': len(src_bodies)},
        'targets': {'families': fam_sizes, 'union_types': tgt_types,
                    'union_bodies': len(tgt_bodies),
                    'family_type_overlaps': overlaps},
        'digest': sha(src_types, tgt_types),
        '_bodies_src': src_bodies, '_bodies_tgt': tgt_bodies,
    }


def edge_ladder(src_bodies, tgt_bodies, min_w, max_depth):
    """Distinct admitted bodyId pairs by depth + per-hop branching."""
    import polars as pl
    e = (pl.scan_parquet(CONN_PARQUET)
         .filter(pl.col('weight') >= min_w)
         .select(['bodyId_pre', 'bodyId_post'])
         .with_columns(pl.col('bodyId_pre').cast(pl.Utf8),
                       pl.col('bodyId_post').cast(pl.Utf8))
         .collect())
    adj = {}
    for p, q in zip(e['bodyId_pre'].to_list(), e['bodyId_post'].to_list()):
        adj.setdefault(p, []).append(q)
    tgt = set(tgt_bodies)
    rows = []
    level = {b for b in src_bodies if b in adj}
    seen = set(src_bodies) | level
    admitted = set()
    for L in range(0, max_depth + 1):
        t0 = time.time()
        layer = {(p, q) for p in level for q in adj.get(p, ())}
        new = layer - admitted
        admitted |= layer
        nxt = {q for _, q in layer} - seen
        rows.append({
            'max_interlayer': L, 'hops_max': L + 1,
            'layer_nodes': len(level),
            'edges_out_of_layer': len(layer),
            'new_distinct_edges': len(new),
            'cum_distinct_edges': len(admitted),
            'next_frontier': len(nxt),
            'targets_seen_so_far': len(seen & tgt),
            'seconds': round(time.time() - t0, 2),
        })
        if not nxt:
            level = set()
            rows[-1]['exhausted'] = True
            break
        seen |= nxt
        level = nxt
    return rows, e.height


def exact_walks(src_bodies, tgt_bodies, min_w, hops_exact=3):
    """EXACT walk counts by hop into the union target set (hops 1..3), plus
    the simplicity check at hop 2 (walks == simple paths iff no self-loop and
    sources/targets are disjoint)."""
    import polars as pl
    e = (pl.scan_parquet(CONN_PARQUET)
         .filter(pl.col('weight') >= min_w)
         .select(['bodyId_pre', 'bodyId_post'])
         .with_columns(pl.col('bodyId_pre').cast(pl.Utf8),
                       pl.col('bodyId_post').cast(pl.Utf8))
         .collect())
    srcs, tgts = sorted(set(src_bodies)), sorted(set(tgt_bodies))
    a = (e.filter(pl.col('bodyId_pre').is_in(srcs))
         .rename({'bodyId_pre': 's', 'bodyId_post': 'x'}).unique())
    b = e.rename({'bodyId_pre': 'x', 'bodyId_post': 't'}).unique()
    out = {}
    h1 = a.join(b, on='x', how='inner')                  # s->x and x->t
    out['hop1'] = int((a.filter(pl.col('x').is_in(tgts)).height))
    j2 = h1.filter(pl.col('t').is_in(tgts))
    out['hop2'] = int(j2.height)
    out['hop2_simple'] = int(j2.filter((pl.col('s') != pl.col('x'))
                                      & (pl.col('x') != pl.col('t'))
                                      & (pl.col('s') != pl.col('t'))).height)
    # hop 3: sum over (x,y) of #src->x * #y->tgt  (each (s,x,y,t) once)
    in_s = a.group_by('x').agg(pl.len().alias('n_s'))
    out_t = b.filter(pl.col('t').is_in(tgts)).group_by('x').agg(
        pl.len().alias('n_t'))
    xy = e.rename({'bodyId_pre': 'x', 'bodyId_post': 'y'}).join(
        in_s, on='x', how='inner')
    cand = xy.join(out_t.rename({'x': 'y', 'n_t': 'n_t'}), on='y',
                   how='inner')
    out['hop3'] = int((cand['n_s'] * cand['n_t']).sum()) if cand.height else 0
    out['method'] = ('polars pair-arithmetic; exact for walks (revisits '
                     'allowed). hop2_simple==hop2 proves the hop<=2 counts '
                     'are also exact simple-path counts for this panel.')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-weight', type=int, default=3)
    ap.add_argument('--depths', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--hops-exact', type=int, default=3)
    ap.add_argument('--out', default=str(Path('/tmp/drocat-bench-20261015')
                                         / 'panel.json'))
    a = ap.parse_args()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel = resolve_panel()
    panel['min_synapse'] = a.min_weight
    src = panel['_bodies_src']
    tgt = panel['_bodies_tgt']
    print(f"source: {panel['source']['rule']} "
          f"{len(panel['source']['types'])} types / {len(src):,} bodies")
    print(f"targets: union {len(panel['targets']['union_types'])} types / "
          f"{len(tgt):,} bodies  families={panel['targets']['families']}")
    if panel['targets']['family_type_overlaps']:
        print('  NOTE family type overlap:',
              panel['targets']['family_type_overlaps'])
    ladder, n_edges = edge_ladder(
        src, tgt, a.min_weight, max(int(x) for x in a.depths.split(',')))
    print(f'\nedges at weight>={a.min_weight}: {n_edges:,} pairs')
    print(f"{'L':>2} {'hops':>4} {'layer nodes':>12} {'new edges':>11} "
          f"{'cum distinct edges':>19} {'next frontier':>14}")
    for r in ladder:
        print(f"{r['max_interlayer']:>2} {r['hops_max']:>4} "
              f"{r['layer_nodes']:>12,} {r['new_distinct_edges']:>11,} "
              f"{r['cum_distinct_edges']:>19,} {r['next_frontier']:>14,}")
    walks = exact_walks(src, tgt, a.min_weight) if a.hops_exact >= 1 else {}
    print('\nexact walk counts into the union:',
          {k: v for k, v in walks.items() if k != 'method'})
    result = {'panel': {k: v for k, v in panel.items()
                        if not k.startswith('_')},
              'edges_total_at_min_weight': n_edges,
              'edge_ladder': ladder, 'exact_walks': walks}
    out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\npanel.json -> {out_path}')


if __name__ == '__main__':
    main()
