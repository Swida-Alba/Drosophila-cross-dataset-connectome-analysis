"""Standalone type-level connection-strength refill for budget-bitten
full-path runs (plan: ``_plan/plan-type-level-refill.md``, Phase 1).

When a FindAllPath ('all' mode) run's applied threshold exceeded the asked
threshold because the Edge Budget floored the discovery cone and/or the
StrongestFirst path budget bit, the exported type-level strength table
(``data_details/connection_type.csv``) counts only the bodyId pairs that
survived those cuts. This module computes the missing mass — the REFILL —
as standalone data records next to the run's own outputs, without touching
any existing file.

Semantics (user-ratified 2026-10-02; validated by the exploration round in
``local_data/refill-experiments/``):

1. **Refill only, no recovery** — only type pairs that already have an
   emitted row are refilled; other pairs are counted in the provenance
   census, never emitted.
2. **Hop-pruned-only edges excluded** — the refill counts bodyIds on
   PATHS, so a lossless distS/distT hop prefilter runs before enumeration
   (never removes a path edge; the criterion is documented in
   ``coana._hop_budget_pass_once``).
3. **Consistency guards** — the caller supplies the EFFECTIVE type map
   (label-mapped, hemisphere-suffixed exactly as the exported table shows)
   and the full connection source; the module additionally honors the
   run's ``exclude intra-type connections`` and refuses runs whose
   re-derived cut cannot reproduce the exported table
   (``table_reproduced`` — the hard post-hoc anchor).
4. **Re-exploration** — the refill re-runs pathfinding on the subgraph
   induced by all bodyIds of the involved types, at the asked threshold,
   with no budgets, toward ALL enrolled (Checked) targets. The emitted
   set is RE-DERIVED by enumerating that same induced graph at the run's
   effective cut (floor threshold w0, then the recorded StrongestFirst
   budget with its production tau tie-drain); refill = E*(asked) minus
   the re-derived emitted set. This is exact where a naive ``weight <
   w0`` split is not: a boundary edge with weight >= w0 whose only path
   died with the floor still belongs in the refill.

Post-hoc by design: everything is read from the run folder
(``parameters.txt`` provenance block, enrollment CSVs,
``connection_type.csv``) plus the caller-supplied connection source and
type map — the bodyId exports the pipeline suppresses by default are never
needed.

Public entry point: :func:`compute_type_level_refill`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

REFILL_FOLDER_NAME = 'type_level_refill'
DEFAULT_DETAIL_CAP = 50_000
DEFAULT_REFILL_PATH_BUDGET = 2_000_000
_DISCLOSURE_SAMPLE_CAP = 20   # per-edge examples in the provenance JSON
# PROB_RATIO_SCALE (0.3) SUSPENDED with the probability family
# (2026-10-06, unverifiable fidelity) — no consumer remains.

STATUS_NO_REFILL = 'no_refill_needed'
STATUS_NO_EMITTED = 'no_emitted_rows'
STATUS_REFILLED = 'refilled'

_PROVENANCE_KEYS = (
    'requested_threshold', 'applied_threshold', 'applied_threshold_source',
    'strongest_first_budget', 'strongest_first_budget_bitten',
    'strongest_first_tau', 'tau_canonical',
    'strongest_dropped_bottleneck', 'edge_budget', 'edge_budget_applied',
    'edge_budget_landing', 'edge_weight_floor',
    'strongest_retained_bottleneck', 'paths_complete', 'replayed_from',
)


class TypeLevelRefillError(Exception):
    """Refusal: the refill cannot be computed honestly for this run."""


# ---------------------------------------------------------------------------
# Run-folder readers
# ---------------------------------------------------------------------------
def _parse_scalar(raw: Optional[str]):
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() in ('n/a', 'none', 'not applied', 'not reached', ''):
        return None
    return raw


def _parse_float(raw: Optional[str]) -> Optional[float]:
    raw = _parse_scalar(raw)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_int(raw: Optional[str]) -> Optional[int]:
    raw = _parse_scalar(raw)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def read_run_provenance(run_dir) -> dict:
    """Parse the applied-threshold provenance from ``parameters.txt``.

    Replay slice folders write ``edge_weight_floor`` twice (the provenance
    block plus the replay writer's own line); all occurrences must agree,
    otherwise the run is ambiguous and refused.
    """
    run_dir = Path(run_dir)
    txt = run_dir / 'parameters.txt'
    if not txt.exists():
        raise TypeLevelRefillError(
            f'{run_dir}: no parameters.txt — not a pathfinding run folder '
            f'(or an unsupported layout).')
    text = txt.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'^weight basis:\s*(\S+)', text, re.MULTILINE)
    prov: Dict[str, str] = {}
    if m and m.group(1).strip().lower().startswith('connection_ratio'):
        prov['weight_basis'] = 'connection_ratio'
    for key in _PROVENANCE_KEYS:
        matches = re.findall(rf'^{key}:\s*(.+)$', text, re.MULTILINE)
        values = {_parse_scalar(m) for m in matches}
        values.discard(None)
        if len(values) > 1:
            raise TypeLevelRefillError(
                f'{run_dir}: parameters.txt carries contradictory '
                f'{key} values {sorted(values)} — refusing.')
        if values:
            prov[key] = values.pop()
    m = re.search(r'^max interlayer:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['max_interlayer'] = _parse_int(m.group(1))
    m = re.search(r'^min synapse number:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['min_synapse'] = _parse_int(m.group(1))
    # Round-13 AND mode: the co-threshold values ride parameters.txt so
    # the refill can reproduce the run's DUAL admission rule (an edge the
    # run never saw must not enter the refill universe — the anchor
    # would otherwise refuse, or worse, quietly disagree).
    m = re.search(r'^min connection ratio:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['min connection ratio'] = _parse_scalar(m.group(1))
    m = re.search(r'^threshold combination:\s*(.+)$', text, re.MULTILINE)
    if m:
        prov['threshold_combination'] = _parse_scalar(m.group(1))
    m = re.search(r'^filter by:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['filter_by'] = _parse_scalar(m.group(1))
    m = re.search(r'^separate hemispheres:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['separate_hemispheres'] = _parse_scalar(m.group(1))
    m = re.search(r'^exclude intra-type connections:\s*(\S+)', text,
                  re.MULTILINE)
    if m:
        prov['exclude_intra_type_connections'] = _parse_scalar(m.group(1))
    m = re.search(r'^aggregate method:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['aggregate_method'] = _parse_scalar(m.group(1))
    # The lossless/lossy pruning disclosure rides on all_attributes.json
    # (decision 2: hop-prune-only counts are disclosed, never refilled).
    attrs = run_dir / 'all_attributes.json'
    if attrs.exists():
        try:
            data = json.loads(attrs.read_text(encoding='utf-8'))
            record = data.get('graph_pruning_record')
            if isinstance(record, dict) and record:
                prov['graph_pruning_record'] = record
        except (OSError, ValueError):
            pass
    return prov


def read_enrollment(run_dir) -> Tuple[List[str], List[str]]:
    """Sources + enrolled (Checked) targets from the run's own enrollment
    CSVs (both carry ``bodyId``; the target table carries ``Checked``)."""
    run_dir = Path(run_dir)
    try:
        src = pd.read_csv(run_dir / 'source_neurons.csv', dtype=str)
        tgt = pd.read_csv(run_dir / 'target_neurons.csv', dtype=str)
    except FileNotFoundError as exc:
        raise TypeLevelRefillError(
            f'{run_dir}: missing enrollment CSVs ({exc.filename}).') from exc
    for frame, name in ((src, 'source_neurons.csv'),
                        (tgt, 'target_neurons.csv')):
        if 'bodyId' not in frame.columns:
            raise TypeLevelRefillError(
                f'{run_dir}: {name} lacks a bodyId column.')
    if 'Checked' not in tgt.columns:
        raise TypeLevelRefillError(
            f'{run_dir}: target_neurons.csv lacks a Checked column.')
    sources = src['bodyId'].dropna().tolist()
    targets = tgt.loc[
        tgt['Checked'].astype(str).str.lower().isin(('true', '1')),
        'bodyId'].dropna().tolist()
    return sources, targets


def _collapse_pair_weights(edges):
    """Per-ROI slice rows collapse to the pair's PHYSICAL mass (sum of
    slices). Hemibrain/male-cns caches store one row per (edge, ROI):
    passing the slices through as separate enumeration edges made the
    `edge_weight` dict keep only the LAST slice per pair, so the cut
    re-derivation under-counted exactly the per-ROI pairs and the
    anchor refused with an (approx-)2x mismatch pattern on real MCNS
    runs (L2->Dm19 34,346 vs 68,719 etc.). FAFB pair-level tables: one
    row per pair — no-op. Integer weights: sum is exact."""
    acc: Dict[Tuple[str, str], float] = {}
    for u, v, w in edges:
        key = (u, v)
        acc[key] = acc.get(key, 0.0) + float(w)
    # natural weights are integer-valued: keep the int type downstream
    # (the records' weight cells format from the raw tuple values)
    return [(u, v, int(w) if float(w).is_integer() else w)
            for (u, v), w in acc.items()]


def read_emitted_pairs(run_dir) -> Dict[Tuple[str, str], int]:
    """Per-type-pair emitted weight — the run's PHYSICAL edge mass.

    Engine round 2026-10-06: prefers ``connection_info_bodyId.csv`` and
    canonicalizes the pair rows before aggregating: EXACT duplicate rows
    collapse (refetch multiplicity — identical modulo the fetch-provenance
    ``conn_layer`` column the per-layer audit rows carry) and per-ROI
    slices sum to the pair's physical mass. In shortest runs the same
    physical edge is fetched once per backward BFS layer it recurs in,
    and ``connection_type.csv`` therefore counts it once per fetch
    (measured on a real FAFB run: the pair-level sums were 2.18x the
    physical mass; 519 of 582 type pairs inflated) — the exact-dup
    collapse restores the mass every other surface of the run reports
    and what the refill's own cut re-derivation (deduped per cut edge)
    must reproduce for the anchor check to hold; the per-ROI slice sum
    keeps multi-ROI pairs (hemibrain/male-cns caches) at full mass
    instead of one ROI's slice. In 'all' mode each pair lives in exactly
    one layer, so both readings coincide. Runs without bodyId detail
    (``skip_bodyId=True``) fall back to the conn-table sums; there the
    anchor semantics stay per-layer (disclosed)."""
    run_dir = Path(run_dir)
    info_path = run_dir / 'data_details' / 'connection_info_bodyId.csv'
    if info_path.exists():
        ci = pd.read_csv(
            info_path,
            dtype={'bodyId_pre': str, 'bodyId_post': str,
                   'type_pre': str, 'type_post': str})
        # EFFECTIVE-LABEL preference (2026-10-12, plan §13.5): a
        # label-mapper run exports the NATIVE types in type_pre/type_post
        # and the mapper's labels in std_label_pre/std_label_post — while
        # the cut re-derivation keys pairs through the EFFECTIVE
        # (mapper-applied) type map. Reading the native columns made
        # every custom-labeled pair mismatch ('re-derived 0 vs exported
        # N') and the refill refuse fail-closed. Prefer the std-label
        # columns when they carry values; fall back to the type columns
        # per row (unmapped bodies keep their native labels either way).
        if {'std_label_pre', 'std_label_post'} <= set(ci.columns):
            ci['type_pre'] = ci['std_label_pre'].where(
                ci['std_label_pre'].notna() & (ci['std_label_pre'] != ''),
                ci['type_pre'])
            ci['type_post'] = ci['std_label_post'].where(
                ci['std_label_post'].notna() & (ci['std_label_post'] != ''),
                ci['type_post'])
        if not ci.empty and {'bodyId_pre', 'bodyId_post',
                             'type_pre', 'type_post'} <= set(ci.columns):
            # canonical pair mass: exact dup collapse + per-pair sum (per-ROI caches store slices; pair-level caches no-op)
            # conn_layer is fetch provenance: refetched rows are exact
            # duplicates modulo it and collapse; per-ROI slice rows stay
            # distinct and sum in the groupby below.
            ci = ci.drop_duplicates(
                subset=[c for c in ci.columns if c != 'conn_layer'])
            ci = ci[ci['type_pre'].notna() & ci['type_post'].notna()]
            agg = ci.groupby(['type_pre', 'type_post'])['weight'].sum()
            return {tuple(k): int(v) for k, v in agg.items()}
    path = run_dir / 'data_details' / 'connection_type.csv'
    if not path.exists():
        raise TypeLevelRefillError(
            f'{run_dir}: no data_details/connection_type.csv — cannot '
            f'determine the emitted type pairs.')
    ct = pd.read_csv(path, dtype={'type_pre': str, 'type_post': str})
    if ct.empty or 'type_pre' not in ct.columns:
        return {}
    agg = ct.groupby(['type_pre', 'type_post'])['weight'].sum()
    return {tuple(k): int(v) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Graph machinery
# ---------------------------------------------------------------------------
def _load_fast_graph():
    """Lazy FastGraph import — the same contract as coana pathfinding
    (the module itself imports without the vispath subproject)."""
    from core.fast_graph import FastGraph
    return FastGraph


def _polars_stream_passes(edge_lf, type_map, emitted, node_set, *,
                          ratio_mode, asked, and_syn_floor, and_ratio_tier,
                          min_syn_active, min_syn, exclude_intra,
                          want_full=False):
    """Polars fast path for the all-mode stream work (PERF 2026-10-08):
    the accumulators (body_in / type_in / type_in_adj / F9 post_totals /
    AND post_totals) and the route-scoped induced admissions, computed by
    vectorized group-bys and semi-joins over the lazy edge frame instead
    of two Python passes over ~20M rows.

    Exactness: natural weights are integer-valued, so every accumulator
    sum is exact in float64 REGARDLESS of summation order — the dicts are
    value-identical to the Python passes. The only non-exact values are
    the F9 ratio divisions, which are pure functions of those exact
    integers, so tier comparisons are stable too.

    Returns dict with: body_in, type_in, type_in_adj, post_totals,
    and_post_totals, e_u (asked-tier induced), e_u_full (synapse>=1
    full-scope admission, [] unless want_full), syn_of, asked_pair_mass,
    non_emitted_pair_mass, non_involved_excluded."""
    import polars as pl
    tm = pl.DataFrame({'bid': [str(k) for k in type_map],
                       'typ': [type_map[k] for k in type_map]}).lazy()
    em = pl.DataFrame({'tpre': [p[0] for p in emitted],
                       'tpost': [p[1] for p in emitted]}).unique()
    em_lazy = em.lazy()
    ns = pl.DataFrame({'nid': sorted(node_set)})
    lf = (edge_lf.select(
              pl.col('bodyId_pre').cast(pl.Utf8).alias('u'),
              pl.col('bodyId_post').cast(pl.Utf8).alias('v'),
              pl.col('weight').cast(pl.Float64).alias('w'))
          # PER-ROI SLICE COLLAPSE (2026-10-10): hemibrain/male-cns caches
          # store one row per (edge, ROI) — tier filters must see the
          # pair's PHYSICAL weight, not one slice's share (a sub-threshold
          # slice of a supra-threshold pair would drop before any later
          # collapse and under-sum the pair).
          .group_by(['u', 'v']).agg(pl.col('w').sum())
          .join(tm.rename({'bid': 'u', 'typ': 'typ_pre'}), on='u',
                how='left')
          .join(tm.rename({'bid': 'v', 'typ': 'typ_post'}), on='v',
                how='left')
          .with_columns([
              pl.when(pl.col('typ_pre').is_null())
                .then(pl.concat_str(pl.lit('~'), pl.col('u')))
                .otherwise(pl.col('typ_pre')).alias('typ_pre'),
              pl.when(pl.col('typ_post').is_null())
                .then(pl.concat_str(pl.lit('~'), pl.col('v')))
                .otherwise(pl.col('typ_post')).alias('typ_post'),
          ]))

    # -- accumulators (the Python Pass A equivalent)
    body_in = {r[0]: r[1] for r in
               lf.group_by('u').agg(pl.col('w').sum()).collect().iter_rows()}
    type_in = {r[0]: r[1] for r in
               lf.group_by('typ_post').agg(pl.col('w').sum())
                 .collect().iter_rows()}
    type_in_adj = {}
    post_totals = {}
    and_post_totals = {}
    if ratio_mode:
        post_totals = {r[0]: r[1] for r in
                       lf.filter(pl.col('w') >= 1)
                         .group_by('v').agg(pl.col('w').sum())
                         .collect().iter_rows()}
    elif and_ratio_tier:
        and_post_totals = {r[0]: r[1] for r in
                           lf.group_by('v').agg(pl.col('w').sum())
                             .collect().iter_rows()}
    if (not ratio_mode) and min_syn_active:
        type_in_adj = {r[0]: r[1] for r in
                       lf.filter(pl.col('w') >= min_syn)
                         .group_by('typ_post').agg(pl.col('w').sum())
                         .collect().iter_rows()}

    # -- admissions. The Python pass's per-row order is: floor filter
    # (ratio) -> syn_of -> asked tier -> involved-node check (counted) ->
    # intra exclusion -> emitted-pair classification. Reproduce that
    # order: tier frame first (pre-node), then the node semi-joins.
    base = lf
    if ratio_mode:
        pt = (pl.DataFrame({'v': list(post_totals),
                             'denom': [post_totals[k]
                                        for k in post_totals]}).lazy())
        # Python semantics: a missing denominator means w_thr = 0.0
        # (excluded) — never w/0 = inf.
        base = base.join(pt, on='v', how='left').with_columns(
            pl.when(pl.col('denom') > 0)
              .then(pl.col('w') / pl.col('denom'))
              .otherwise(0.0).alias('ew'))
    else:
        if and_ratio_tier:
            apt = (pl.DataFrame(
                       {'v': list(and_post_totals),
                        'adenom': [and_post_totals[k]
                                   for k in and_post_totals]}).lazy())
            base = base.join(apt, on='v', how='left').with_columns(
                pl.when(pl.col('adenom') > 0)
                  .then(pl.col('w') / pl.col('adenom'))
                  .otherwise(0.0).alias('co_thr'))
            base = base.filter(pl.col('co_thr') >= and_ratio_tier)

    # syn_of (ratio): natural mass of every FLOOR-passing row keyed by
    # the bodyId edge — the whole-table floor scope, exactly like the
    # Python pass (which accumulates before the tier check). Integer
    # weights: exact regardless of summation order.
    syn_of = defaultdict(float)
    if ratio_mode:
        floor_frame = lf
        if and_syn_floor is not None:
            floor_frame = floor_frame.filter(pl.col('w') >= and_syn_floor)
        syn_of = defaultdict(float)
        for (u, v, w) in (floor_frame.group_by(['u', 'v'])
                          .agg(pl.col('w').sum()).collect().iter_rows()):
            syn_of[(u, v)] = w

    # tier frame (asked scope)
    tier = base
    if ratio_mode:
        if and_syn_floor is not None:
            tier = tier.filter(pl.col('w') >= and_syn_floor)
        tier = tier.filter(pl.col('ew') >= asked)
    else:
        tier = tier.with_columns(pl.col('w').alias('ew'))
        tier = tier.filter(pl.col('ew') >= asked)
    tier_collected = tier.collect()

    # involved-node drop counter (pre-semi-join, like the Python pass)
    ns_list = sorted(node_set)
    non_involved_excluded = int(
        tier_collected.filter(
            (~pl.col('u').is_in(ns_list)) | (~pl.col('v').is_in(ns_list)))
        .select(pl.len()).item())

    node_passing = (tier_collected
                    .join(ns.rename({'nid': 'u'}), on='u', how='semi')
                    .join(ns.rename({'nid': 'v'}), on='v', how='semi'))
    if exclude_intra:
        node_passing = node_passing.filter(
            pl.col('typ_pre') != pl.col('typ_post'))

    # emitted-pair classification of the tier-passing, node-passing,
    # intra-filtered mass (grouped: integer sums, order-free) — computed
    # BEFORE the route-scope semi-join, which only the ENUMERATION edges
    # (e_u) pass through; non-emitted mass is disclosed, not dropped.
    asked_pair_mass = defaultdict(float)
    non_emitted_pair_mass = 0.0
    for tpre, tpost, wsum in (node_passing
                              .group_by(['typ_pre', 'typ_post'])
                              .agg(pl.col('w').sum()).iter_rows()):
        pair = (tpre, tpost)
        if pair in emitted:
            asked_pair_mass[pair] += wsum
        else:
            non_emitted_pair_mass += wsum

    # ROUTE SCOPE (2026-10-07): edges on non-emitted pairs never enter
    # the enumeration graph.
    node_passing = node_passing.join(
        em, left_on=['typ_pre', 'typ_post'],
        right_on=['tpre', 'tpost'], how='semi')

    e_u = [(u, v, ew) for u, v, ew in
           node_passing.select(['u', 'v', 'ew']).iter_rows()]

    e_u_full = []
    if want_full:
        full_frame = base
        if ratio_mode:
            if and_syn_floor is not None:
                full_frame = full_frame.filter(
                    pl.col('w') >= and_syn_floor)
            # NO tier: the full-connection scope keeps every floor edge
            # (base already carries the F9-ratio `ew` enumeration weight)
        else:
            full_frame = full_frame.filter(pl.col('w') >= 1)
            full_frame = full_frame.with_columns(
                pl.col('w').alias('ew'))
        if exclude_intra:
            full_frame = full_frame.filter(
                pl.col('typ_pre') != pl.col('typ_post'))
        full_frame = (full_frame
                      .join(ns.lazy().rename({'nid': 'u'}), on='u',
                            how='semi')
                      .join(ns.lazy().rename({'nid': 'v'}), on='v',
                            how='semi')
                      .join(em_lazy, left_on=['typ_pre', 'typ_post'],
                            right_on=['tpre', 'tpost'], how='semi'))
        e_u_full = [(u, v, ew) for u, v, ew in
                    full_frame.select(['u', 'v', 'ew']).collect().iter_rows()]

    return {
        'body_in': body_in, 'type_in': type_in, 'type_in_adj': type_in_adj,
        'post_totals': post_totals, 'and_post_totals': and_post_totals,
        'e_u': e_u, 'e_u_full': e_u_full, 'syn_of': syn_of,
        'asked_pair_mass': asked_pair_mass,
        'non_emitted_pair_mass': non_emitted_pair_mass,
        'non_involved_excluded': non_involved_excluded,
    }


def hop_prefilter(edges: Sequence[Tuple[str, str, float]], sources,
                  targets, bound: int) -> Tuple[List, int]:
    """Lossless distS/distT hop-bound criterion on the induced edges:
    keep (u, v) iff distS(u) + 1 + distT(v) <= bound. Never removes an
    edge that lies on a simple source->target path within ``bound`` (the
    converse holds only for walks — coana._hop_budget_pass_once)."""
    adj: Dict[str, set] = defaultdict(set)
    radj: Dict[str, set] = defaultdict(set)
    for u, v, _w in edges:
        adj[u].add(v)
        radj[v].add(u)

    def bfs(starts, graph):
        dist = {}
        dq = deque()
        for s in starts:
            if s in graph:
                dist[s] = 0
                dq.append(s)
        while dq:
            u = dq.popleft()
            for v in graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    BIG = 10 ** 9
    ds = bfs(set(sources), adj)
    dt = bfs(set(targets), radj)
    kept, dropped = [], 0
    for u, v, w in edges:
        a, b = ds.get(u, BIG), dt.get(v, BIG)
        if a < BIG and b < BIG and a + 1 + b <= bound:
            kept.append((u, v, w))
        else:
            dropped += 1
    return kept, dropped


def _enumerate_paths(edges, sources, targets, bound, budget=None):
    """Enumerate simple source->target paths on ``edges`` with the
    PRODUCTION strongest-first engine (descending bottleneck, budget with
    tau tie-drain — identical emission semantics to the pipeline's
    budgeted 'all'-mode core). Returns (paths, edge info, stats) where
    edge info = {(u, v): {'w', 'traversals', 'hops'}} over path edges and
    paths = [(bottleneck, tuple(nodes))] in emission order."""
    FastGraph = _load_fast_graph()
    g = FastGraph()
    # PERF (2026-10-08): tuple constructor — the pandas DataFrame +
    # iterrows() round-trip was pure construction overhead.
    g.build_from_edges(edges)
    weight = {(u, v): w for (u, v, w) in edges}
    stats: Dict[str, object] = {}
    paths: List[Tuple[float, Tuple[str, ...]]] = []
    traversals: Dict[Tuple[str, str], int] = defaultdict(int)
    hops: Dict[Tuple[str, str], set] = defaultdict(set)
    seen_sources = {s for s in map(str, sources) if s in g}
    seen_targets = {t for t in map(str, targets) if t in g}
    gen = g.find_paths_strongest_first(
        sorted(seen_sources), sorted(seen_targets), bound,
        budget=budget, stats=stats, yield_bottleneck=True)
    for bn, raw in gen:
        # nodes are already the str ids the induced lists pushed
        nodes = tuple(raw)
        paths.append((bn, nodes))
        for i in range(len(nodes) - 1):
            key = (nodes[i], nodes[i + 1])
            traversals[key] += 1
            hops[key].add(i)
    info = {k: {'w': weight[k], 'traversals': traversals[k],
                'hops': sorted(hops[k])} for k in traversals}
    return paths, info, stats


def _layered_chain_structures(chain, pair_edges, sources, targets,
                              type_map, cut_keys):
    """LAYERED PER-POSITION FILTER for ONE emitted chain at the full
    scope — THE universal full-scope realization (plan-corridor-ratio-
    basis §8.2/§9, user directive 2026-10-13): forward body sets rooted
    at the run's sources and backward body sets rooted at the run's
    targets over the chain's own hop pairs, INTERSECTED per position
    (the intersection is load-bearing — a union would admit
    forward-only/backward-only decoys); a hop-i edge (u, v) is kept iff
    u survives at position i and v at position i+1. Per-hop mass = the
    kept edges' weight sum; the kept positions per edge are its hops.
    NO path or traversal counts exist (the estimator duality is
    dropped — reference-only data).

    Exactness keys on the chain's EFFECTIVE labels (the labels bodies
    resolve to — display-distinct labels that resolve alike are one
    label here). No label revisit -> the kept set equals the counted
    DP's realization set exactly (a body has one effective label, so a
    layered walk cannot revisit a body; probe/fixture-verified incl.
    decoy rejection). Label revisit -> the kept set is the LAYERED-WALK
    SUPERSET of the simple-path scope (disclosed via ``revisit``): a
    body may occupy two positions and contributes its edges at both; an
    edge recurring at two positions counts once per position.

    Canonical iteration throughout (sorted adjacency, sorted pres) so
    float mass sums are order-stable run to run.

    Returns None when the chain has no realization; else a dict with
    edges_cls_at (per-hop {'emitted'/'refill': {(u,v): w}}), edges_at
    (per-hop {(u,v): w} all classes), nodes_at (per-position body sets,
    fwd ∩ bwd), revisit (bool)."""
    h = len(chain) - 1
    if h < 1:
        return None
    start = {str(s) for s in sources
             if type_map.get(str(s)) == chain[0]}
    goal = {str(t) for t in targets
            if type_map.get(str(t)) == chain[-1]}
    if not start or not goal:
        return None
    hop_dicts = []
    for i in range(h):
        d = pair_edges.get((chain[i], chain[i + 1]))
        if not d:
            return None
        hop_dicts.append(
            {u: sorted(vs) for u, vs in d.items()})
    # forward per-position body sets (POST-only propagation)
    fwd = [set() for _ in range(h + 1)]
    fwd[0] = set(start)
    for i in range(h):
        d = hop_dicts[i]
        nxt = fwd[i + 1]
        for u in fwd[i]:
            for v, _w in d.get(u, ()):
                nxt.add(v)
    # backward per-position body sets (PRE-only propagation), rooted at
    # the FORWARD-REACHABLE targets only (a target no source reaches
    # realizes nothing — the counted DP roots its backward counts the
    # same way)
    bwd = [set() for _ in range(h + 1)]
    bwd[h] = {t for t in goal if t in fwd[h]}
    if not bwd[h]:
        return None
    for i in range(h - 1, -1, -1):
        d = hop_dicts[i]
        nxt = bwd[i + 1]
        prv = bwd[i]
        for u, vs in d.items():
            for v, _w in vs:
                if v in nxt:
                    prv.add(u)
                    break
    revisit = len(set(chain)) != len(chain)
    edges_cls_at = [{} for _ in range(h)]
    edges_at = [{} for _ in range(h)]
    for i in range(h):
        d = hop_dicts[i]
        cls_i = edges_cls_at[i]
        all_i = edges_at[i]
        fi = fwd[i]
        bi = bwd[i]
        fi1 = fwd[i + 1]
        bi1 = bwd[i + 1]
        for u in sorted(fi & bi):
            for v, w in d.get(u, ()):
                if v in fi1 and v in bi1:
                    key = (u, v)
                    cls = 'emitted' if key in cut_keys else 'refill'
                    cls_i.setdefault(cls, {})[key] = w
                    all_i[key] = w
    nodes_at = [fwd[i] & bwd[i] for i in range(h + 1)]
    return {'edges_cls_at': edges_cls_at, 'edges_at': edges_at,
            'nodes_at': nodes_at, 'revisit': revisit}


def _dp_chain_structures(chain, pair_edges, sources, targets, type_map,
                         cut_keys):
    """EXACT layered DP for ONE emitted chain at the full scope: forward
    path counts from the sources, backward counts to the targets — every
    edge/body on >= 1 realization is identified by f[i][u] * b[i+1][v]
    (or f[i][u] * b[i][u] for bodies), with EXACT per-edge traversal
    counts. No path materialization: cost is O(chain subgraph), and
    nothing truncates (2026-10-11: replaces the DFS collector whose
    200k cap under-counted the fat MCNS chains — 108/249 truncated).

    TEST ORACLE ONLY since 2026-10-13: the production full scope uses
    ``_layered_chain_structures`` (plan §9 — no counts, linear cost);
    this DP stays as the exactness oracle the layered filter is
    fixture-tested against.

    Returns None when the chain has no realization; else a dict with
    edges_cls_at (per-hop {'emitted'/'refill': {(u,v): w}}), edges_at
    (per-hop {(u,v): w} all classes), nodes_at (per-hop body sets),
    path_count, traversals {(u,v): count}."""
    h = len(chain) - 1
    if h < 1:
        return None
    start = {str(s) for s in sources
             if type_map.get(str(s)) == chain[0]}
    goal = {str(t) for t in targets
            if type_map.get(str(t)) == chain[-1]}
    if not start or not goal:
        return None
    hop_dicts = []
    for i in range(h):
        d = pair_edges.get((chain[i], chain[i + 1]))
        if not d:
            return None
        hop_dicts.append(d)
    # forward: f[i][body] = # realizations source -> body at layer i
    f = [defaultdict(int) for _ in range(h + 1)]
    for s in start:
        f[0][s] = 1
    for i in range(h):
        d = hop_dicts[i]
        for u, cnt in f[i].items():
            for v, _w in d.get(u, ()):
                f[i + 1][v] += cnt
    # backward: b[i][body] = # realizations body at layer i -> target
    b = [defaultdict(int) for _ in range(h + 1)]
    for t in goal:
        if f[h].get(t):
            b[h][t] = 1
    for i in range(h - 1, -1, -1):
        d = hop_dicts[i]
        for u, _cnt in f[i].items():
            acc = 0
            for v, _w in d.get(u, ()):
                acc += b[i + 1].get(v, 0)
            if acc:
                b[i][u] = acc
    path_count = sum(f[h].get(t, 0) for t in goal)
    if path_count == 0:
        return None
    edges_cls_at = [{} for _ in range(h)]
    edges_at = [{} for _ in range(h)]
    nodes_at = []
    traversals: Dict[Tuple[str, str], int] = {}
    for i in range(h):
        d = hop_dicts[i]
        wmap = {(u, v): w for u, v, w in
                ((u, v, wv) for u, vs in d.items() for v, wv in vs)}
        cls_i = edges_cls_at[i]
        all_i = edges_at[i]
        for u, cnt in f[i].items():
            for v, _w in d.get(u, ()):
                g = b[i + 1].get(v, 0)
                if not g:
                    continue  # never reaches a target on this chain
                key = (u, v)
                w = wmap[key]
                cls = 'emitted' if key in cut_keys else 'refill'
                cls_i.setdefault(cls, {})[key] = w
                all_i[key] = w
                traversals[key] = cnt * g
        edges_at[i] = all_i
    for i in range(h + 1):
        nodes_at.append({u for u, cnt in f[i].items()
                         if cnt and b[i].get(u)})
    return {'edges_cls_at': edges_cls_at, 'edges_at': edges_at,
            'nodes_at': nodes_at, 'path_count': path_count,
            'traversals': traversals}


def _dfs_chain_structures(chain, pair_edges, sources, targets, type_map,
                          cut_keys, cap=2_000_000):
    """SIMPLE-path structures for ONE MULTI-VISIT chain (a type appears
    twice in the chain, so body-level simplicity is NOT automatic): a
    visited-set DFS collecting the same per-chain structures the DP
    builds — classed per-hop edges, per-layer body sets, exact
    traversals — recorded ON COMPLETION so only edges/bodies on >= 1
    collected realization enter the structures. Walk-level alive
    pruning strips dead-end branches first (lossless: a simple path is
    a walk, so its prefixes/suffixes always satisfy walk reachability).
    Capped at `cap` paths (truncated=True discloses under-counted
    masses; the DP cannot serve these chains because layered walks may
    revisit a body across layers, which the engine's simple-path
    semantics excludes).

    TEST ORACLE ONLY since 2026-10-13: the production full scope uses
    ``_layered_chain_structures`` (plan §9); this DFS stays as the
    simple-path bound the layered filter's labeled superset is
    fixture-tested against.

    Returns None when the chain has no realization; else a dict with
    edges_cls_at / edges_at / nodes_at / path_count / traversals /
    truncated."""
    h = len(chain) - 1
    if h < 1:
        return None
    start = {str(s) for s in sources
             if type_map.get(str(s)) == chain[0]}
    goal = {str(t) for t in targets
            if type_map.get(str(t)) == chain[-1]}
    if not start or not goal:
        return None
    hop_dicts = []
    for i in range(h):
        d = pair_edges.get((chain[i], chain[i + 1]))
        if not d:
            return None
        hop_dicts.append(d)
    # ALIVE PRUNING (2026-10-11, plan F6): walk-level forward/backward
    # counts identify the edges/starts on >= 1 layered WALK. A simple
    # path is a walk, so pruning is LOSSLESS for the DFS — the dead-end
    # branches it strips otherwise burn the 2M path cap before real
    # paths are reached (the MCNS replay truncated 29/77 multi-visit
    # chains exactly there).
    fwd = [defaultdict(int) for _ in range(h + 1)]
    for s in start:
        fwd[0][s] = 1
    for i in range(h):
        d = hop_dicts[i]
        nxt = fwd[i + 1]
        for u, cnt in fwd[i].items():
            for v, _w in d.get(u, ()):
                nxt[v] += cnt
    bwd = [defaultdict(int) for _ in range(h + 1)]
    for t in goal:
        if fwd[h].get(t):
            bwd[h][t] = 1
    for i in range(h - 1, -1, -1):
        d = hop_dicts[i]
        for u in fwd[i]:
            acc = 0
            for v, _w in d.get(u, ()):
                acc += bwd[i + 1].get(v, 0)
            if acc:
                bwd[i][u] = acc
    hop_dicts = [
        {u: sorted((v, w) for v, w in vs if bwd[i + 1].get(v))
         for u, vs in d.items() if fwd[i].get(u) and bwd[i].get(u)}
        for i, d in enumerate(hop_dicts)]
    # sorted() per adjacency list pins the DFS COLLECTION order: the raw
    # lists inherit pair_edges_full's e_u_full iteration order, and polars
    # collect row order varies run to run. Complete chains are immune
    # (every simple path is collected regardless of order), but TRUNCATED
    # chains keep whichever paths the cap reached FIRST — without the
    # sort, the disclosed lower-bound edge set (and the full-scope mass)
    # wiggled ±0.005% between replays.
    start = {s for s in start if bwd[0].get(s)}
    if not start or not all(hop_dicts):
        return None
    edges_cls_at = [{} for _ in range(h)]
    edges_at = [{} for _ in range(h)]
    nodes_at = [set() for _ in range(h + 1)]
    traversals: Dict[Tuple[str, str], int] = defaultdict(int)
    state = {'count': 0, 'truncated': False}
    n = h

    def dfs(i, body, visited, path, wts):
        # ON-COMPLETION recording (2026-10-11, with the F6 pruning): the
        # structures must match the DP contract — edges/nodes/bodies on
        # >= 1 COLLECTED realization. The previous explore-time recording
        # also captured dead partial branches (edges whose subtree
        # completed nothing), inflating multi-visit chain masses with
        # mass that realized no path.
        if state['truncated']:
            return
        if i == n:
            if body in goal:
                state['count'] += 1
                if state['count'] >= cap:
                    state['truncated'] = True
                for j in range(n):
                    key = (path[j], path[j + 1])
                    cls = 'emitted' if key in cut_keys else 'refill'
                    edges_cls_at[j].setdefault(cls, {})[key] = wts[j]
                    edges_at[j][key] = wts[j]
                    traversals[key] += 1
                    nodes_at[j].add(path[j])
                nodes_at[n].add(path[n])
            return
        for v, w in hop_dicts[i].get(body, ()):
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            wts.append(w)
            dfs(i + 1, v, visited, path, wts)
            path.pop()
            wts.pop()
            visited.discard(v)
            if state['truncated']:
                return

    for s in sorted(start):
        if state['truncated']:
            break
        visited = {s}
        dfs(0, s, visited, [s], [])
    if state['count'] == 0:
        return None
    return {'edges_cls_at': edges_cls_at, 'edges_at': edges_at,
            'nodes_at': nodes_at, 'path_count': state['count'],
            'traversals': dict(traversals),
            'truncated': state['truncated']}


def build_effective_type_map(neuron_frame, *, dataset,
                             mapping_file=None, label_mapper=None,
                             separate_hemispheres=False) -> dict:
    """bodyId -> EFFECTIVE type label, mirroring the run's own type-level
    aggregation (statvis ``EnrichConnectionTable`` semantics):

    1. Label mapping first — a bodyId-level mapping wins when it differs
       from the raw id, else a type-level mapping when it differs from
       the raw type (``LabelMapper.get_label``; unmapped ids keep their
       raw labels through the identity fallback).
    2. Hemisphere suffix ``_L/_R/_U`` appended when the run separated
       hemispheres (coana ``_append_hemisphere_suffix_series`` rules:
       hemisphere > somaSide/soma side > rootSide columns, then the
       instance ``_L/_R`` suffix, defaulting to 'U'; labels that already
       carry a suffix are left alone).
    3. Untyped labels dropped (``utils.label_utils.is_untyped_type_label``)
       — they can never be involved labels.

    ``label_mapper`` (a live LabelMapper) wins over ``mapping_file``; a
    recorded mapping file that cannot be loaded raises
    :class:`TypeLevelRefillError` (refuse-on-doubt: a wrong label map
    would fail the table-reproduction anchor anyway, with a worse
    diagnostic).
    """
    import numpy as np

    frame = neuron_frame
    if 'bodyId' not in frame.columns:
        raise TypeLevelRefillError(
            'neuron table lacks a bodyId column.')
    if mapping_file is not None and label_mapper is None:
        try:
            from comparison.label_mapper import LabelMapper
            label_mapper = LabelMapper(overall_mapping_json=str(mapping_file))
        except Exception as exc:
            raise TypeLevelRefillError(
                f'cannot load the recorded mapping file '
                f'{mapping_file!r} ({exc}) — the refill cannot reproduce '
                f'the run\'s labels without it.') from exc

    body_ids = frame['bodyId'].astype(str)
    types = frame['type'].fillna('Unknown').astype(str) \
        if 'type' in frame.columns else pd.Series(
            ['Unknown'] * len(frame), index=frame.index)

    if label_mapper is not None:
        bid_map = {b: label_mapper.get_label(dataset, b)
                   for b in body_ids.unique()}
        mapped_body = body_ids.map(bid_map)
        mask_body = mapped_body.notna() & (mapped_body != body_ids)
        type_map_lut = {t: label_mapper.get_label(dataset, t)
                        for t in types[~mask_body].unique()}
        mapped_type = types.map(type_map_lut)
        mask_type = (~mask_body & (types != 'Unknown')
                     & (types != body_ids) & mapped_type.notna()
                     & (mapped_type != types))
        effective = types.copy()
        effective.loc[mask_body] = mapped_body[mask_body]
        effective.loc[mask_type] = mapped_type[mask_type]
    else:
        effective = types

    if separate_hemispheres:
        codes = _hemisphere_codes(frame)
        has_suffix = effective.str.endswith(('_L', '_R', '_U'))
        effective = effective.where(has_suffix, effective + '_' + codes)

    from utils.label_utils import is_untyped_type_label
    keep = ~effective.map(is_untyped_type_label)
    return dict(zip(body_ids[keep], effective[keep]))


def _hemisphere_codes(frame):
    """Hemisphere codes per the coana rules (vectorized mirror of
    ``_normalize_hemisphere_series`` + the instance-suffix fallback)."""
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    col = next((lowered[c] for c in
                ('hemisphere', 'soma side', 'somaside', 'rootside')
                if c in lowered), None)
    codes = pd.Series('U', index=frame.index, dtype=object)
    if col is not None:
        vals = frame[col].fillna('').astype(str).str.strip().str.lower()
        codes = vals.map(_HEMI_ALIASES).fillna('U')
    elif 'instance' in frame.columns:
        inst = frame['instance'].fillna('').astype(str)
        codes[inst.str.endswith('_R')] = 'R'
        codes[inst.str.endswith('_L')] = 'L'
    return codes


_HEMI_ALIASES = {
    'r': 'R', 'right': 'R', 'rhs': 'R', 'right hemisphere': 'R',
    'l': 'L', 'left': 'L', 'lhs': 'L', 'left hemisphere': 'L',
}


def filter_edges_by_hemisphere(edges, type_map, hemi_filter):
    """Fetch-time edge rule (coana ``_apply_hemisphere_suffix_to_conn_df``):
    'left'/'right' keep an edge only when BOTH endpoints are that side or
    unlabeled ('U'); 'both' keeps everything. The endpoint side is read
    from the (possibly suffixed) effective label."""
    if hemi_filter not in ('left', 'right'):
        return edges
    keep_code = hemi_filter[0].upper()

    def _code(body_id):
        label = type_map.get(str(body_id))
        if label is None:
            return 'U'
        return (label.rsplit('_', 1)[-1]
                if label.endswith(('_L', '_R', '_U')) else 'U')

    return (e for e in edges
            if _code(e[0]) in (keep_code, 'U')
            and _code(e[1]) in (keep_code, 'U'))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _load_shortest_store(run_dir: Path):
    """Load the shortest discovery store's structural pieces: per-layer
    connection frames (pair -> PER-LAYER PHYSICAL weight; the per-edge
    values are the pair's ROI-SLICE SUM within that layer — the fetch
    frames are verbatim per-ROI slices, and keep-one-row policies
    sampled a different slice per layer, manufacturing fake cross-layer
    conflicts; original-data adjudication 2026-10-12, plan §9), the
    store meta (targets found + hop limits) and the per-target distance
    maps. Cross-layer recurring edges dedup keep-first at consumption
    (``_enumerate_shortest``'s I1 physical dict) — after summation the
    layers agree exactly (measured 0 conflicts on the MCNS store)."""
    store_dir = run_dir / 'shortest_discovery_store'
    meta_file = store_dir / 'meta.json'
    if not meta_file.exists():
        return None
    try:
        import polars as pl
        from shortest_discovery_store import (
            load_store_meta,
            load_target_distances,
        )
        meta = load_store_meta(store_dir)
        layer_weights = []      # [(layer, {(pre, post): w})]
        conn_dir = store_dir / 'connections'
        merged = conn_dir / 'connections_all.parquet'

        def _layer_dict(frame: 'pl.DataFrame') -> dict:
            has_roi = 'roi' in frame.columns
            dedup_subset = (['bodyId_pre', 'bodyId_post', 'roi', 'weight']
                            if has_roi
                            else ['bodyId_pre', 'bodyId_post', 'weight'])
            sums = (frame.unique(subset=dedup_subset, keep='any')
                    .group_by(['bodyId_pre', 'bodyId_post'])
                    .agg(pl.col('weight').sum()))
            return {(str(pre), str(post)): float(w)
                    for pre, post, w in sums.iter_rows()}

        if merged.exists():
            frame = pl.read_parquet(merged)
            layer_col = ('conn_layer' if 'conn_layer' in frame.columns
                         else 'layer')
            groups = []
            for (layer,), group in frame.group_by([layer_col]):
                groups.append((str(layer), group))
            groups.sort(key=lambda kv: kv[0])   # deterministic keep-first
            for layer, group in groups:
                layer_weights.append((layer, _layer_dict(group)))
        else:
            for path in sorted(conn_dir.glob('*.parquet')):
                layer_weights.append(
                    (path.stem, _layer_dict(pl.read_parquet(path))))
        if not layer_weights:
            return None
        targets = list(meta.get('targets_found') or [])
        distances = load_target_distances(store_dir, targets)
        hop_limits = dict(meta.get('target_hop_limits') or {})
        return {
            'layer_weights': layer_weights,
            'targets': targets,
            'distances': distances,
            'hop_limits': hop_limits,
        }
    except Exception:
        return None


def _enumerate_shortest(layer_weights, node_set, sources, targets, bound,
                        budget, distances, hop_limits):
    """Shortest-mode re-exploration on the induced subgraph: per-layer
    weights are the pair's ROI-SLICE SUM (the loader collapses the
    fetched slices), deduped keep-first ACROSS layers — the I1 rule
    (2026-10-05) that summing the per-layer refetches multiplied
    weights by the fetch multiplicity, which re-ordered the
    StrongestFirst drain against the fixed engine and broke the cut
    anchor on runs with cross-layer recurring edges. After the slice
    summation the layers agree exactly (0 conflicts measured), so the
    keep-first choice is exact. Enumeration is seeded with the store's
    full-graph distance maps so per-pair min-hop semantics match the
    run."""
    FastGraph = _load_fast_graph()
    physical: Dict[Tuple[str, str], float] = {}
    for _layer, weights in layer_weights:
        for (u, v), w in weights.items():
            if u in node_set and v in node_set:
                physical.setdefault((u, v), w)
    g = FastGraph()
    g.build_from_edges(((u, v, w) for (u, v), w in physical.items()))
    stats: Dict[str, object] = {}
    target_list = sorted(t for t in targets if t in g)
    gen = g.find_paths_shortest_strongest_first(
        target_list, [s for s in map(str, sources) if s in g], bound,
        budget=budget, stats=stats, payload=False,
        target_cutoffs={t: hop_limits[t] for t in target_list
                        if t in hop_limits},
        target_distances={t: distances.get(t, {}) for t in target_list},
    )
    traversals: Dict[Tuple[str, str], int] = defaultdict(int)
    for path in gen:
        nodes = [str(n) for n in path]
        for i in range(len(nodes) - 1):
            traversals[(nodes[i], nodes[i + 1])] += 1
    weight_of = dict(physical)
    info = {k: {'w': weight_of[k], 'traversals': n, 'hops': [0]}
            for k, n in traversals.items()}
    return info, stats


def compute_type_level_refill(
    run_dir,
    *,
    edges: Optional[Sequence[Tuple[str, str, float]]] = None,
    type_map: Optional[Mapping[str, str]] = None,
    detail_cap: int = DEFAULT_DETAIL_CAP,
    path_budget: int = DEFAULT_REFILL_PATH_BUDGET,
    out_dir=None,
    write: bool = True,
    source_note: str = '',
    path_mode: Optional[str] = None,
    edge_lf=None,
    hemi_filter: str = 'both',
) -> dict:
    """Compute (and by default write) the type-level refill records for
    one FindAllPath ('all' mode) run folder — BOTH refill kinds (DUAL
    REFILL, 2026-10-08):

    - refill-asked: the second round at the run's own thresholding (the
      asked tier). Gate: synapse-basis runs need the applied threshold to
      exceed the asked one; ratio-basis runs always fill (non-bitten ones
      get the complete-at-asked view, status 'complete_at_asked', mass 0).
    - refill-full (rec['full']): the second round admitted at
      SYNAPSE >= 1 only — no ratio tier, no synapse tier — re-valuing the
      SAME emitted chain sequences (value change, zero structural
      change). Produced for EVERY all-mode run with emitted rows,
      restricted to the emitted chains (2026-10-11: bounded cost, no
      path budget, so the weak sub-threshold tail is never lost) and
      realized by the LAYERED PER-POSITION BODY-SET FILTER (2026-10-13,
      plan-corridor-ratio-basis §8.3/§9 — universal at every applied
      threshold, replacing the DP/DFS enumeration: forward source sets
      and backward target sets per position, intersected; per-hop mass
      = kept-edge weight sum; NO path/traversal counts). Exact on
      chains without an effective-label revisit; label-revisiting
      chains carry the layered-walk superset semantics, disclosed as
      ``chains_revisit`` (with examples) in the full block. The retired
      ``paths``/``bodyid_paths`` counters are null (no bodyId paths are
      materialized); nothing truncates (``chains_truncated`` = 0).
      Writes refill_full_type_pairs.csv + the mirror's
      refill_full_paths_type.csv / refill_full_carrying_paths_type.csv /
      refilled_full_connection_type.csv.

    Shortest mode produces the asked view only.

    Parameters
    ----------
    run_dir : run folder (``find-paths-complete_*``) with
        ``parameters.txt``, enrollment CSVs and
        ``data_details/connection_type.csv``.
    edges : the FULL connection source at natural weights —
        ``(bodyId_pre, bodyId_post, weight)`` triples covering every
        dataset connection (the F9 threshold-free denominators are
        computed from it at ``min_weight=1``).
    type_map : EFFECTIVE labels ``bodyId -> type`` exactly as the
        exported table shows them (label-mapped, hemisphere-suffixed by
        the caller; untyped ids should be absent or will never match).
    detail_cap : row cap for ``refill_bodyId_pairs.csv`` (deterministic
        weight-desc order); the type-pair summary is never capped.
    path_budget : enumeration guard for the asked-threshold
        re-exploration; a bitten guard is reported as
        ``refill_truncated`` (an honest partial refill, never a silent
        one). The cut enumeration that reproduces the exported table is
        never guarded — it enumerates exactly what the original run did.
    out_dir : output directory (defaults to
        ``<run_dir>/data_details/type_level_refill``).
    write : write the record files when True.

    Returns the provenance dict (also written as
    ``refill_provenance.json``). Raises :class:`TypeLevelRefillError` on
    any refusal (unsupported filters, unreadable artifacts, or a cut
    re-derivation that fails to reproduce the exported table).
    """
    run_dir = Path(run_dir)
    prov = read_run_provenance(run_dir)

    # Shortest mode (Phase 3): the target-rooted store supplies the layer
    # frames + distance maps; the edge budget never floors shortest mode,
    # so the only cut is the StrongestFirst budget drain.
    is_shortest = (path_mode == 'shortest'
                   or (path_mode is None
                       and run_dir.name.startswith('find-paths-shortest_')))
    store = _load_shortest_store(run_dir) if is_shortest else None
    if is_shortest and store is None:
        raise TypeLevelRefillError(
            f'{run_dir}: shortest-mode refill requires the run\'s '
            f'shortest_discovery_store/ (connections + node_distances) — '
            f'not present (retention pruned it, or a pre-batching run).')

    filter_by = prov.get('filter_by', 'bodyId')
    if filter_by and filter_by != 'bodyId':
        raise TypeLevelRefillError(
            f'{run_dir}: filter_by={filter_by!r} runs are out of scope '
            f'for the v1 refill (type-level filtering changes the '
            f'asked-threshold graph itself).')

    sources, targets = read_enrollment(run_dir)
    bound = (prov.get('max_interlayer')
             if prov.get('max_interlayer') is not None else None)
    if bound is None:
        raise TypeLevelRefillError(
            f'{run_dir}: parameters.txt lacks a readable max interlayer.')
    bound += 1

    ratio_mode = str(prov.get('weight_basis', '')) == 'connection_ratio'
    # The edge source may be a one-pass STREAM (the in-pipeline hook); the
    # ratio lane iterates it twice (denominator pre-pass + induced build)
    # and the anchor/denominator loops iterate it again — materialize
    # non-sequence sources once (the hook's own docstring says single
    # pass, no 20M-row materialization — the ratio lane trades that for
    # correctness; ~20M triples ≈ 2-3 GB, inside the replay-capture
    # budget class the pipeline already accepts).
    # PERF (2026-10-08): with a lazy polars edge frame the accumulators
    # and admissions run vectorized (`_polars_stream_passes`) and the
    # 20M-row tuple materialization is skipped entirely (the all-mode
    # flow never needs the tuples). The shortest lane still consumes the
    # tuple stream (per-layer conversion), so it materializes as before.
    use_polars = (edge_lf is not None and not is_shortest
                  and type_map is not None
                  and hemi_filter not in ('left', 'right'))
    if edges is not None and not isinstance(edges, (list, tuple)):
        edges = list(edges)
    if not use_polars and edges is None:
        raise TypeLevelRefillError(
            f'{run_dir}: neither edges nor edge_lf was supplied.')
    _parse_thr = _parse_float if ratio_mode else _parse_int
    asked = _parse_thr(prov.get('requested_threshold'))
    if asked is None:
        asked = _parse_thr(prov.get('min connection ratio')
                           if ratio_mode
                           else prov.get('min_synapse'))
    if asked is None:
        raise TypeLevelRefillError(
            f'{run_dir}: parameters.txt lacks requested_threshold.')
    # Replay canon folders stamp requested_threshold with the CANONICAL
    # value; the orchestrating request's true asked threshold rides on
    # the replayed_from line (validated in the exploration round).
    replayed_from = _parse_thr(prov.get('replayed_from'))
    slice_tier = None
    if replayed_from is not None and replayed_from < asked:
        slice_tier = asked          # the folder's own stamped tier
        asked = replayed_from       # the orchestrating request
    applied = _parse_thr(prov.get('applied_threshold'))
    applied = applied if applied is not None else asked
    w0 = _parse_thr(prov.get('edge_weight_floor'))
    budget_n = _parse_int(prov.get('strongest_first_budget'))
    bitten = prov.get('strongest_first_budget_bitten', 'False') == 'True'
    exclude_intra = prov.get(
        'exclude_intra_type_connections', 'False') == 'True'
    aggregate_method = prov.get('aggregate_method') or 'product'

    rec: dict = {
        'run_dir': str(run_dir),
        'requested_threshold': asked,
        'requested_threshold_stamped':
            _parse_thr(prov.get('requested_threshold')),
        'replayed_from': replayed_from,
        'applied_threshold': applied,
        'applied_threshold_source': prov.get('applied_threshold_source'),
        'edge_weight_floor': w0,
        'edge_budget_landing': _parse_int(prov.get('edge_budget_landing')),
        'strongest_first_budget': budget_n,
        'strongest_first_budget_bitten': bitten,
        'max_interlayer': prov.get('max_interlayer'),
        'separate_hemispheres': prov.get('separate_hemispheres'),
        'aggregate_method': aggregate_method,
        'graph_pruning_record': prov.get('graph_pruning_record'),
        'source_note': source_note,
        # basis disclosure (the report's cutoff labels are mode-aware):
        # on ratio runs the applied threshold is a RATIO and the mass
        # cutoff is the forced min synapse
        'weight_basis': ('connection_ratio' if ratio_mode else 'synapse'),
        'min_synapse': prov.get('min_synapse'),
    }

    # -- DUAL REFILL (2026-10-08): emitted rows + the induced graph feed
    # BOTH scopes — refill-asked (the gate below keeps its own semantics)
    # and refill-full (synapse >= 1, no tier; produced for EVERY run with
    # emitted rows, bitten or not, both modes).
    emitted = read_emitted_pairs(run_dir)
    if not emitted:
        rec['status'] = STATUS_NO_EMITTED
        if write:
            _write_records(run_dir, out_dir, rec, None, None, detail_cap)
        return rec
    # Non-bitten ratio runs (applied == asked): the second round runs at
    # the asked tier and recovers mass 0 — the status says
    # 'complete_at_asked' so consumers know the complete table is the
    # full asked-threshold set (the Refill tab still shows it).
    complete_only = ratio_mode and applied <= asked
    rec['status'] = STATUS_REFILLED
    involved = {p for pair in emitted for p in pair}
    node_set = {str(b) for b, t in type_map.items() if t in involved}

    syn_of: Dict[Tuple[str, str], float] = defaultdict(float)
    # DUAL REFILL: the full-scope admission list (all-mode only; the
    # shortest lane leaves it empty and skips the full scope).
    e_u_full: List[Tuple[str, str, float]] = []
    # R3 residual: raw physical >=asked mass on emitted pairs that
    # entered the induced graph. All-mode only — the shortest branch
    # leaves it empty and the pair rows say so (README).
    asked_pair_mass: Dict[Tuple[str, str], float] = defaultdict(float)
    non_emitted_pair_mass = 0.0
    residual_active = False
    _kept_cut_paths = None
    _cut_paths = []
    # Round-13 AND mode flag (shared by both lanes — the shortest branch
    # filters the store's layers with it, the all-mode branch filters the
    # streamed edges) + the RATIO-MODE FLOOR (2026-10-07, user directive):
    # PLAIN ratio mode always refills to synapse >= 1 — the run's own
    # forced floor — so the second round's universe is complete; "unless
    # specified independently" = AND mode's own min_synapse. Assigned on
    # BOTH lanes: the refill_synapse_floor provenance stamp below reads it
    # unconditionally under ratio_mode (the shortest lane used to hit an
    # UnboundLocalError here).
    _and_mode = str(prov.get('threshold_combination',
                             '')).strip().lower().startswith('and')
    if ratio_mode:
        and_syn_floor = (prov.get('min_synapse') if _and_mode else 1)
    else:
        and_syn_floor = None
    # -- stream accumulators, SHARED by both lanes. PERF (2026-10-08):
    # with a lazy polars edge frame the accumulators + admissions run
    # vectorized (`_polars_stream_passes`) — the previous form was 2-6
    # full Python passes over the ~20M-row stream (profiled: ~25s of the
    # 23s compute was function-body looping, 40M dict.get calls). The
    # tuple path keeps one fused accumulator pass + a per-lane admission
    # pass; every accumulator adds in stream order, and natural weights
    # are integer-valued, so both paths are value-identical.
    try:
        _min_syn = float(prov.get('min_synapse'))
    except (TypeError, ValueError):
        _min_syn = None
    _min_syn_active = bool(_min_syn and _min_syn > 1)
    and_ratio_tier = (_parse_float(prov.get('min connection ratio'))
                      if (_and_mode and not ratio_mode) else None)
    body_in: Dict[str, float] = defaultdict(float)
    type_in: Dict[str, float] = defaultdict(float)
    type_in_adj: Dict[str, float] = defaultdict(float)
    and_post_totals: Dict[str, float] = defaultdict(float)
    post_totals: Dict[str, float] = defaultdict(float)
    pol = None
    if use_polars:
        pol = _polars_stream_passes(
            edge_lf, type_map, emitted, node_set,
            ratio_mode=ratio_mode, asked=asked,
            and_syn_floor=and_syn_floor, and_ratio_tier=and_ratio_tier,
            min_syn_active=_min_syn_active, min_syn=_min_syn,
            exclude_intra=exclude_intra, want_full=True)
        body_in = pol['body_in']
        type_in = pol['type_in']
        type_in_adj = pol['type_in_adj']
    else:
        for u, v, w in edges:
            u_s, v_s = str(u), str(v)
            wf = float(w)
            body_in[u_s] += wf
            type_in[type_map.get(v_s, f'~{v}')] += wf
            if _min_syn_active and wf >= _min_syn:
                type_in_adj[type_map.get(v_s, f'~{v}')] += wf
            if ratio_mode:
                if wf >= 1:
                    # RATIO-MODE FLOOR (2026-10-07): the run itself forces
                    # min_synapse to 1 (or the AND-specified floor) and
                    # fetches with min_weight=1, so the F9 denominators sum
                    # only edges the run could ever see — sub-floor stream
                    # rows (w < 1) must not inflate them.
                    post_totals[v_s] += wf
            elif and_ratio_tier:
                and_post_totals[v_s] += wf
    def _produce_full_scope(chain_keys=None, cut_edge_keys=None):
        """REFILL-FULL (DUAL REFILL 2026-10-08): the second round on
        the route-scoped graph admitted at SYNAPSE >= 1 only — no
        ratio tier, no synapse tier — re-valuing the EMITTED chain
        sequences (value change, zero structural change; mass behind
        emitted pairs but off-chain stays disclosed via the asked
        view's unfired residual, never tabulated). Produced for
        EVERY all-mode run with emitted rows. Since 2026-10-13 the
        per-chain realization is the LAYERED PER-POSITION BODY-SET
        FILTER (``_layered_chain_structures`` — universal at every
        applied threshold, plan-corridor-ratio-basis §8.3/§9; no
        bodyId-path enumeration, no counts, no caps). Returns the
        'full' provenance block (the pair table is written alongside)."""
        if not e_u_full:
            return None
        # (the lossless hop prefilter is retired here: the per-chain
        # enumeration walks only the chains' own hop pairs, so a
        # whole-graph prefilter pass would be pure overhead)
        full_drop = 0
        weight_full = {(u, v): w for (u, v, w) in e_u_full}
        anchor_full = (
            (lambda uv: syn_of.get(uv, weight_full[uv]))
            if ratio_mode else (lambda uv: weight_full[uv]))
        # PER-CHAIN ENUMERATION (2026-10-11, user directive: the refill
        # is restricted to per-path chain and is budgeted): the full
        # scope re-values the EMITTED chains — enumerate each chain's
        # layered realizations exactly instead of walking the whole >=1
        # cone unbudgeted (the 2026-10-09 plain-budget rejection stands:
        # a count budget would cut the weak paths the scope exists for;
        # chain SCOPING bounds the cost without cutting them). The
        # collected set IS the chain scope the tables always filtered
        # to; off-chain mass stays disclosed via pair_offchain_mass.
        # (The earlier 10h+ CPU profile was this unbudgeted enumeration
        # on a fat MCNS query.)
        pair_edges_full: Dict[Tuple[str, str], Dict[str, list]] = {}
        for u, v, w in e_u_full:
            pair_edges_full.setdefault(
                (type_map.get(u, '~' + u), type_map.get(v, '~' + v)),
                {}).setdefault(u, []).append((v, w))
        if chain_keys is None:
            # no-asked-refill branch: the emitted paths ARE the chains
            # (cut == asked there; the re-derivation above already
            # produced them — no second enumeration). The cut-edge keys
            # come from the same re-derivation (refill_full_keys must
            # exclude the emitted edges or the emitted mass would be
            # double-counted as refill).
            chain_keys = {tuple(type_map[n] for n in nodes)
                          for _bn, nodes in (_emitted_paths or [])}
            cut_edge_keys = set(cut_info or ())
        # HYBRID per-chain enumeration: distinct-type chains via an
        # exact layered DP (no path materialization, no truncation);
        # multi-visit chains via the alive-pruned visited DFS (cap 2M,
        # disclosed). Forward/backward layered counts identify every
        # edge/body on >= 1 realization of the chain, with exact
        # per-edge traversals.
        cut_keys_f = set(cut_edge_keys or ())
        seq_edges_full: Dict[Tuple[str, ...], List[Dict]] = {}
        seq_nodes_full: Dict[Tuple[str, ...], List[set]] = {}
        seq_cls_full: Dict[Tuple[str, ...], List[Dict]] = {}
        # LAYERED PER-POSITION FILTER (2026-10-13, plan §9): every chain
        # through the same set-propagation pass — no per-chain dispatch,
        # no DP counts, no DFS, no 2M caps. Label-revisiting chains are
        # disclosed (the layered-walk superset semantics); everything
        # else is exact vs the retired DP's realization set.
        chains_revisit = 0
        revisit_examples: List[str] = []
        position_set_memberships = 0
        position_set_max = 0
        for chain in sorted(chain_keys):
            dp = _layered_chain_structures(
                chain, pair_edges_full, sources, targets, type_map,
                cut_keys_f)
            if dp is None:
                continue
            if dp['revisit']:
                chains_revisit += 1
                if len(revisit_examples) < 5:
                    revisit_examples.append('->'.join(chain))
            seq_cls_full[chain] = dp['edges_cls_at']
            seq_nodes_full[chain] = dp['nodes_at']
            for nodes_i in dp['nodes_at']:
                position_set_memberships += len(nodes_i)
                position_set_max = max(position_set_max, len(nodes_i))
            # carrying structures (refill view): chains with >= 1
            # refill-class (non-cut) edge on a realization
            if any('refill' in cls_at and cls_at['refill']
                   for cls_at in dp['edges_cls_at']):
                seq_edges_full[chain] = dp['edges_at']
        # estar_full edge info from the structures (kept positions)
        hops: Dict[Tuple[str, str], set] = defaultdict(set)
        for chain in seq_cls_full:
            for i, cls_at in enumerate(seq_cls_full[chain]):
                for _cls, edges in cls_at.items():
                    for (u, v) in edges:
                        hops[(u, v)].add(i)
        estar_full = {k: {'w': weight_full[k],
                          'hops': sorted(hops[k])}
                      for k in hops}
        refill_full_keys = set(estar_full) - set(cut_edge_keys or ())
        full_totals: Dict[Tuple[str, str], float] = defaultdict(float)
        # sorted() pins the float-addition order (polars row order
        # varies run to run; the pair totals must not wiggle in the
        # last ulp between replays)
        for uv in sorted(refill_full_keys):
            full_totals[(type_map[uv[0]], type_map[uv[1]])] += \
                anchor_full(uv)
        # OFF-CHAIN disclosure (2026-10-10 user-approved; 2026-10-11
        # wording per the per-chain enumeration): the full-scope
        # admitted (>= 1) mass on the pair whose edges fired on NO
        # CHAIN realization (per-path chain scope excludes them by
        # design — both never-fired edges and off-chain-fired edges)
        # — refilled_total is a chain-scoped value and this column
        # makes the gap self-describing:
        # raw_pair_total ~= emitted + refill + pair_offchain_mass
        # (+ edges outside the involved-types node set / hop bound).
        full_edge_keys = set(estar_full)
        offchain: Dict[Tuple[str, str], float] = defaultdict(float)
        # sorted() pins the float-addition order (see full_totals)
        for u, v, w in sorted(e_u_full):
            if (u, v) not in full_edge_keys:
                offchain[(type_map[u], type_map[v])] += (
                    syn_of.get((u, v),
                               weight_full.get((u, v), 0.0))
                    if ratio_mode
                    else w)
        rows_full = []
        for pair in sorted(emitted):
            emit_w = int(emitted[pair])
            ref_w = int(full_totals.get(pair, 0.0))
            denom = type_in.get(pair[1], 0.0)
            ratio = ((emit_w + ref_w) / denom) if denom > 0 else None
            rows_full.append({
                'type_pre': pair[0], 'type_post': pair[1],
                'emitted_weight': emit_w,
                'refill_weight': ref_w,
                'refilled_total': emit_w + ref_w,
                'refilled_connection_ratio': (
                    round(ratio, 6) if ratio is not None else None),
                'pair_offchain_mass': int(offchain.get(pair, 0.0)),
            })
        # DP-fed row assembly (same helpers the path-fed asked view
        # uses; type_n mirrors the aggregators' induced-membership N)
        type_n_full: Dict[str, int] = defaultdict(int)
        for b in node_set:
            label = type_map.get(str(b))
            if label is not None:
                type_n_full[label] += 1
        carrying_rows = _refill_rows_from_structures(
            seq_edges_full, seq_nodes_full, type_n_full, type_in)
        complete_rows_full, complete_stats_full = \
            _complete_rows_from_structures(
                seq_cls_full, seq_nodes_full, type_n_full, type_in)
        # kept-edge census (all classes incl. emitted-class): the
        # layered reference's own disclosures (plan §9.4)
        kept_mass_total = sum(
            anchor_full(uv) for uv in sorted(estar_full))
        block = {
            'status': 'ok',   # layered per-position filter (chain-scoped)
            'refill_star': True,
            'enumeration': 'none (layered per-position filtering '
                           'reference)',
            'chains': len(chain_keys),
            'chains_revisit': chains_revisit,
            'chains_revisit_examples': revisit_examples,
            'chains_truncated': 0,
            'induced_edges': len(e_u_full),
            'prefilter_dropped': full_drop,
            'paths': None,
            'paths_note': 'retired (the layered reference materializes '
                          'no bodyId paths)',
            'chains_carrying': len(carrying_rows),
            'chains_complete': complete_stats_full['chains'],
            'bodyid_paths': None,
            'refill_edges': len(refill_full_keys),
            'refill_weight_total': int(sum(
                full_totals.get(p, 0.0) for p in emitted)),
            'kept_edge_count': len(estar_full),
            'kept_mass_total': int(kept_mass_total),
            'position_set_memberships': position_set_memberships,
            'position_set_max': position_set_max,
            'truncated': False,
        }
        # FULL-SCOPE CACHE GUARD (2026-10-10, user-approved): the asked
        # scope is anchored by table_reproduced, but the full scope had
        # no check — an incomplete connection cache would silently
        # undercount it. Cheap cross-check: the stream-derived all-post
        # incoming must dominate the run's own exported applied-tier
        # incoming (type_incoming_applied.csv) by a sane margin — the
        # applied file sums a SUBSET of the same dataset's edges (every
        # emitted-network edge, each >= the applied cut). type_in <
        # applied_in is impossible under a complete cache.
        try:
            applied_csv = (run_dir / 'data_details'
                           / 'type_incoming_applied.csv')
            if applied_csv.exists():
                _applied = pd.read_csv(applied_csv)
                _bad = [
                    (str(r['type']), float(r['incoming']),
                     float(type_in.get(str(r['type']), 0.0)))
                    for _, r in _applied.iterrows()
                    if pd.notna(r['incoming'])
                    and float(r['incoming'])
                    > float(type_in.get(str(r['type']), 0.0)) + 1e-6]
                if _bad:
                    raise TypeLevelRefillError(
                        f'{run_dir}: full-scope cache guard FAILED — '
                        f'{len(_bad)} type(s) have a smaller stream-derived '
                        'all-post incoming than the run\'s own exported '
                        'applied-tier incoming (first: '
                        f'{_bad[0][0]}: stream {_bad[0][2]:g} < applied '
                        f'{_bad[0][1]:g}). The connection source appears '
                        'incomplete relative to the run; refusing to '
                        'write full-scope data that would silently '
                        'undercount it.')
                block['cache_guard'] = 'ok'
        except TypeLevelRefillError:
            raise
        except Exception:  # noqa: BLE001 — guard is best-effort
            pass
        if write:
            full_dir = (Path(out_dir) if out_dir is not None
                        else run_dir / 'data_details'
                        / 'type_level_refill')
            full_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows_full).to_csv(
                full_dir / 'refill_full_type_pairs.csv', index=False)
            # The >=threshold incoming per type (the Adj. ratios'
            # conditioned denominator at the asked tier) — exported so the
            # report computes CHAIN-derived adjusted ratios (realized hop
            # mass / this denominator) instead of pair-level constants.
            pd.DataFrame([
                {'type_post': k, 'incoming_threshold': v}
                for k, v in sorted(type_in_adj.items())]).to_csv(
                full_dir / 'type_incoming_asked.csv', index=False)
            if out_dir is None:
                mirror = run_dir / _MIRROR_FOLDER_NAME
                mirror.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(complete_rows_full).to_csv(
                    mirror / 'refill_full_paths_type.csv', index=False)
                pd.DataFrame(carrying_rows).to_csv(
                    mirror / 'refill_full_carrying_paths_type.csv',
                    index=False)
                pd.DataFrame(rows_full).to_csv(
                    mirror / 'refilled_full_connection_type.csv',
                    index=False)
        return block

    if is_shortest:
        # DUAL REFILL: shortest mode keeps the asked-view gate (the full
        # scope is an all-mode deliverable — the shortest lane has no
        # second-round paths table to re-value).
        if applied <= asked and not ratio_mode:
            rec['status'] = STATUS_NO_REFILL
            if write:
                _write_records(run_dir, out_dir, rec, None, None,
                               detail_cap)
            return rec
        # -- Phase 3: structure from the run's discovery store; `edges`
        # is only the F9 denominator source. Per-pair weights are the
        # per-layer sums (the exported table counts cross-layer
        # multiplicity exactly this way). Ratio lane: the store's
        # SYNPASE weights are converted to F9 ratios (w / all-post
        # incoming from the edge source) so the re-enumeration orders
        # and drains exactly like the ratio run did; the per-layer
        # synapse masses ride along in syn_of for the anchor.
        layer_weights = store['layer_weights']
        # Round-13 AND mode: the store's edges must respect the run's
        # co-threshold too (ratio basis + AND -> synapse floor; synapse
        # basis + AND -> ratio tier), exactly like the all-mode pass.
        if _and_mode:
            _and_floor = (prov.get('min_synapse') if ratio_mode else None)
            _and_tier = (_parse_float(prov.get('min connection ratio'))
                         if not ratio_mode else None)
            _and_tot: Dict[str, float] = defaultdict(float)
            if _and_tier:
                for _u, _v, _w in edges:
                    _and_tot[str(_v)] += float(_w)
            layer_weights = [
                (_layer,
                 {(u, v): w for (u, v), w in weights.items()
                  if not (_and_floor is not None and float(w) < _and_floor)
                  and not (_and_tier is not None and (
                      _and_tot.get(v, 0.0) <= 0
                      or float(w) / _and_tot[v] < _and_tier))})
                for _layer, weights in layer_weights
            ]
        if ratio_mode:
            post_tot: Dict[str, float] = defaultdict(float)
            for _u, _v, _w in edges:
                post_tot[str(_v)] += float(_w)
            converted = []
            for _layer, weights in layer_weights:
                conv_layer = {}
                for (u, v), w in weights.items():
                    denom = post_tot.get(v, 0.0)
                    conv_layer[(u, v)] = (
                        float(w) / denom if denom > 0 else 0.0)
                    # PHYSICAL anchor mass: keep the first LAYER's
                    # (slice-summed) weight. A pair recurring across BFS
                    # layers is one edge; the anchor must reproduce
                    # read_emitted_pairs' deduped mass, not the fetch
                    # multiplicity (2026-10-06). After the loader's
                    # slice summation the layers agree exactly, so the
                    # keep-first is exact.
                    if (u, v) not in syn_of:
                        syn_of[(u, v)] = float(w)
                converted.append((_layer, conv_layer))
            layer_weights = converted
        estar, ask_stats = _enumerate_shortest(
            layer_weights, node_set, sources, store['targets'],
            bound, path_budget, store['distances'], store['hop_limits'])
        refill_truncated = bool(ask_stats.get('budget_bitten'))
        _paths_asked = None
        cut_threshold = asked          # the edge budget never floors here
        cut_info, cut_stats = _enumerate_shortest(
            layer_weights, node_set, sources, store['targets'],
            bound, (budget_n if bitten else None),
            store['distances'], store['hop_limits'])
        e_u = []
        for _layer, weights in layer_weights:
            e_u.extend((u, v, w) for (u, v), w in weights.items()
                       if u in node_set and v in node_set)
        non_involved_excluded = sum(
            1 for _layer, weights in layer_weights
            for (u, v) in weights
            if u not in node_set or v not in node_set)
        prefilter_dropped = 0
        e_pref = e_u
    else:
        # -- node set U + induced edges at the asked threshold.
        # Ratio lane (Phase 3): the enumeration weights are the F9
        # connection ratios (computed here — connection_ratio_paths
        # imports this module, so no reverse import); the SYNPASE masses
        # ride along in syn_of for the anchor + refill aggregates (the
        # exported table's weight column stays synapse mass).
        e_u: List[Tuple[str, str, float]] = []
        # syn_of was declared above the mode branch (the shortest branch
        # fills it with per-layer multiplicity; the all-mode branch here)
        non_involved_excluded = 0
        # R3 residual accumulator: raw PHYSICAL mass of >=asked edges on
        # EMITTED type pairs that entered the induced graph — the union
        # the pair rows reconcile against (emitted + fired refill +
        # unfired off-route residual). All-mode only.
        residual_active = True
        # Round-13 AND mode: reproduce the run's DUAL admission rule in
        # the induced universe. ratio basis + AND -> the synapse floor;
        # synapse basis + AND -> the ratio co-threshold (F9 denominators
        # from the same edge source). (_and_mode/and_syn_floor and the
        # Pass A accumulators are hoisted above the lane branch.)
        # -- Pass B: the asked-tier admissions (the polars fast path
        # produced them inside `_polars_stream_passes`, together with the
        # full-scope admission list).
        e_u_full: List[Tuple[str, str, float]] = (
            list(pol['e_u_full']) if pol is not None else [])
        if pol is not None:
            e_u = pol['e_u']
            syn_of = pol['syn_of']
            asked_pair_mass = pol['asked_pair_mass']
            non_emitted_pair_mass = pol['non_emitted_pair_mass']
            non_involved_excluded = pol['non_involved_excluded']
        else:
            # PER-ROI SLICE COLLAPSE (2026-10-10): sum slice rows to the
            # pair's physical weight BEFORE any tier check (a
            # sub-threshold slice of a supra-threshold pair must not
            # drop). Shortest mode never reaches this branch (its
            # per-layer stream is slice-summed by the loader; the
            # cross-layer refetches dedup keep-first per engine I1,
            # never summed). Totals (Pass A
            # denominators) are collapse-order-invariant.
            if edges is not None:
                edges = _collapse_pair_weights(edges)
            for u, v, w in edges:
                u, v = str(u), str(v)
                # FULL-scope admission (DUAL REFILL 2026-10-08):
                # synapse >= 1 only — no ratio tier, no synapse tier, and
                # independent of the AND co-threshold. The enumeration
                # weight stays the mode's tier unit (F9 ratio / natural
                # synapse) for drain-order parity with the run.
                wf_full = float(w)
                if (wf_full >= 1 and u in node_set and v in node_set
                        and not (exclude_intra
                                 and type_map.get(u) == type_map.get(v))
                        and (type_map[u], type_map[v]) in emitted):
                    if ratio_mode:
                        _den_f = post_totals.get(v, 0.0)
                        e_u_full.append(
                            (u, v, (wf_full / _den_f) if _den_f > 0
                             else 0.0))
                    else:
                        e_u_full.append((u, v, wf_full))
                if ratio_mode:
                    w = float(w)
                    if and_syn_floor is not None and w < and_syn_floor:
                        continue
                    denom = post_totals.get(v, 0.0)
                    w_thr = (w / denom) if denom > 0 else 0.0
                    syn_of[(u, v)] += w
                else:
                    if and_ratio_tier:
                        _denom = and_post_totals.get(v, 0.0)
                        if _denom <= 0 or (float(w) / _denom) < and_ratio_tier:
                            continue
                    w_thr = w
                if w_thr < asked:
                    continue
                if u not in node_set or v not in node_set:
                    non_involved_excluded += 1
                    continue
                if exclude_intra and type_map.get(u) == type_map.get(v):
                    continue
                _pair = (type_map[u], type_map[v])
                if _pair in emitted:
                    asked_pair_mass[_pair] += float(w)
                else:
                    # ROUTE SCOPE (2026-10-07): the second round builds only
                    # the GIVEN (emitted) type routes — edges on non-emitted
                    # pairs never enter the enumeration graph, so out-of-scope
                    # routes are not pathfinded at all (not merely filtered
                    # after the fact). Their >=asked mass is disclosed at pair
                    # level via non_emitted_pair_mass / unfired_residual.
                    non_emitted_pair_mass += float(w)
                    continue
                e_u.append((u, v, w_thr))
        e_pref, prefilter_dropped = hop_prefilter(
            e_u, sources, targets, bound)

        # DUAL REFILL gate: the ASKED view keeps the synapse-run gate;
        # the full scope runs regardless and rides on the record.
        if applied <= asked and not ratio_mode:
            rec['status'] = STATUS_NO_REFILL
            # BUDGETED chain derivation (2026-10-11): the full scope
            # needs the emitted chains — one asked-tier enumeration with
            # the refill path budget (an UNBITTEN run's own set is
            # complete well under it; the old unbudgeted derivation is
            # what made the full scope unbounded on fat queries).
            _cut_paths_nr, cut_info_nr, _cut_stats_nr = _enumerate_paths(
                e_pref, sources, targets, bound, budget=path_budget)
            full_block = _produce_full_scope(
                chain_keys={tuple(type_map[n] for n in nodes)
                            for _bn, nodes in _cut_paths_nr},
                cut_edge_keys=set(cut_info_nr))
            if full_block is not None:
                rec['full'] = full_block
            if write:
                _write_records(run_dir, out_dir, rec, None, None,
                               detail_cap)
            return rec

        # -- E*(asked): re-exploration on the induced subgraph (no
        # budgets, guarded only by the honest truncation flag)
        _paths_asked, estar, ask_stats = _enumerate_paths(
            e_pref, sources, targets, bound, budget=path_budget)
        refill_truncated = bool(ask_stats.get('budget_bitten'))

        # -- re-derive the emitted set at the run's effective cut.
        # Replay slices (review round): the folder's set is the t0
        # DRAINED prefix further cut at its own tier — the stamps say
        # complete (bitten False) because the slice is complete AT its
        # tier, so the budget must come from the t0 stamp, and the
        # enumerated prefix is then restricted to bottleneck >= the
        # slice tier (identical for the canon folder, where the tier
        # sits at/below the drain tau).
        cut_threshold = max(asked, w0 or asked)
        if cut_threshold > asked:
            e_cut = [(u, v, w) for (u, v, w) in e_pref
                     if w >= cut_threshold]
        else:
            e_cut = e_pref
        _cut_paths, cut_info, cut_stats = _enumerate_paths(
            e_cut, sources, targets, bound,
            budget=(budget_n if (bitten or slice_tier is not None)
                    else None))
        if slice_tier is not None and _cut_paths:
            kept = [(bn, nodes) for bn, nodes in _cut_paths
                    if bn >= slice_tier]
            if len(kept) != len(_cut_paths):
                _w_cut = {(u, v): w for (u, v, w) in e_cut}
                _trav = defaultdict(int)
                _hops = defaultdict(set)
                for bn, nodes in kept:
                    for i in range(len(nodes) - 1):
                        key = (nodes[i], nodes[i + 1])
                        _trav[key] += 1
                        _hops[key].add(i)
                cut_info = {k: {'w': _w_cut[k], 'traversals': _trav[k],
                                'hops': sorted(_hops[k])}
                            for k in _trav}
                _kept_cut_paths = kept
    # The EMITTED bodyId paths (complete at the applied tier; replay
    # slices use their own kept prefix). Their type sequences are the
    # Pair Explorer's chain scope — both refill tables are restricted
    # to these chains (values change, structure does not).
    _emitted_paths = (_kept_cut_paths if _kept_cut_paths is not None
                      else _cut_paths)

    # -- anchor: the re-derived cut must reproduce the exported table
    edge_weight = {(u, v): w for (u, v, w) in e_pref}
    _anchor_w = (lambda uv: syn_of.get(uv, edge_weight[uv])) \
        if ratio_mode else (lambda uv: edge_weight[uv])
    rederived: Dict[Tuple[str, str], float] = defaultdict(float)
    for (u, v), meta in cut_info.items():
        rederived[(type_map[u], type_map[v])] += _anchor_w((u, v))
    mismatch = {
        f'{k[0]}->{k[1]}': (int(rederived.get(k, 0)), int(v))
        for k, v in emitted.items()
        if int(rederived.get(k, 0)) != int(v)}
    extra = {f'{k[0]}->{k[1]}': int(v)
             for k, v in rederived.items() if k not in emitted}
    # TIE-ORDER TOLERANCE (2026-10-10): a budget-BITTEN run's kept-path
    # membership at tau is order-dependent — the engine enumerated its
    # in-memory graph, the re-derivation enumerates the DB-built graph,
    # and large equal-bottleneck tie sets keep slightly different
    # members (measured on a real bitten MCNS run: all pairs present,
    # masses within +0.06..+0.6%, zero extra pairs). When the pair SETS
    # match exactly and every delta is within 1% of the exported mass,
    # the re-derivation IS the run's universe up to tie composition —
    # reproduce with a disclosure instead of refusing. Anything else
    # (missing/extra pairs, or a larger delta — labels/hemisphere/
    # filter drift) still refuses exactly as before.
    anchor_note = None
    bitten_now = bool(rec.get('budget_bitten')) or bitten
    if (mismatch and not extra and cut_info
            and bitten_now):
        worst = 0.0
        within = True
        d_total = 0
        exported_total = 0
        for k, (a, b) in mismatch.items():
            rel = abs(a - b) / b if b else 1.0
            worst = max(worst, rel)
            d_total += a - b
            if rel > 0.05:
                within = False
                break
        for k, v in emitted.items():
            exported_total += int(v)
        agg_rel = abs(d_total) / exported_total if exported_total else 1.0
        if within and agg_rel <= 0.01:
            anchor_note = (
                f'budget-tie tolerance: {len(mismatch)} pair mass(es) '
                f'differ (worst {worst * 100:.2f}%, aggregate '
                f'{agg_rel * 100:.3f}%) — equal-bottleneck tie membership '
                'is enumeration-order dependent between the engine graph '
                'and the DB-built re-derivation; the pair sets match '
                'exactly, so the universe is the run own data up to tie '
                'composition.')
            mismatch = {}
    if mismatch or extra or not cut_info:
        detail = '; '.join(
            f'{k}: re-derived {a} vs exported {b}'
            for k, (a, b) in sorted(mismatch.items())[:8])
        raise TypeLevelRefillError(
            f'{run_dir}: cut re-derivation failed to reproduce '
            f'connection_type.csv ({len(mismatch)} mismatched, '
            f'{len(extra)} extra pairs{"; " + detail if detail else ""}) '
            f'— the supplied edges/type map do not match the run '
            f'(labels, hemisphere suffixes, filters or dataset state).')
    rec['table_reproduced'] = True
    if anchor_note:
        rec['anchor_note'] = anchor_note

    # -- refill = E*(asked) - emitted_rederived, emitted pairs only
    refill_edges: Dict[Tuple[str, str], dict] = {}
    skipped_edges: List[Tuple[str, str, float]] = []
    boundary: List[Tuple[str, str, float]] = []
    for (u, v), meta in estar.items():
        pair = (type_map[u], type_map[v])
        if pair not in emitted:
            skipped_edges.append((u, v, meta['w']))
        elif (u, v) in cut_info:
            continue
        else:
            refill_edges[(u, v)] = meta
            if w0 is not None and meta['w'] >= w0:
                boundary.append((u, v, meta['w']))

    # -- per-type-pair aggregation + threshold-free F9 ratios.
    # Ratio lane: refill_weight is SYNPASE recovery mass (syn_of), the
    # per-edge pair ratio IS the enumeration weight, and body_in still
    # comes from the natural-weight edge source.
    totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for (u, v), meta in refill_edges.items():
        totals[(type_map[u], type_map[v])] += (
            syn_of.get((u, v), meta['w']) if ratio_mode else meta['w'])
    # body_in / type_in / type_in_adj are computed in the fused Pass A
    # above (single stream pass); _min_syn is hoisted there too.
    pair_ratio = {}
    for (u, v) in list(refill_edges) + list(cut_info):
        if ratio_mode:
            pair_ratio[(u, v)] = edge_weight[(u, v)]
        else:
            denom = body_in.get(v, 0.0)
            pair_ratio[(u, v)] = (
                edge_weight[(u, v)] / denom if denom > 0 else None)

    emitted_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for (u, v) in cut_info:
        emitted_pair_counts[(type_map[u], type_map[v])] += 1

    rows = []
    for pair in sorted(emitted):
        emit_w = emitted[pair]
        ref_w = totals.get(pair, 0.0)
        denom = type_in.get(pair[1], 0.0)
        type_ratio = ((emit_w + ref_w) / denom) if denom > 0 else None
        denom_adj = (type_in_adj.get(pair[1], 0.0)
                     if _min_syn_active else denom)
        type_ratio_adj = ((emit_w + ref_w) / denom_adj) \
            if denom_adj > 0 else None
        rows.append({
            'type_pre': pair[0], 'type_post': pair[1],
            'emitted_weight': int(emit_w),
            'refill_weight': int(ref_w),
            'refilled_total': int(emit_w + ref_w),
            'emitted_pair_count': emitted_pair_counts.get(pair, 0),
            'refill_pair_count': sum(
                1 for uv in refill_edges
                if (type_map[uv[0]], type_map[uv[1]]) == pair),
            'refilled_connection_ratio': (
                round(type_ratio, 6) if type_ratio is not None else None),
            'refilled_connection_ratio_adj': (
                round(type_ratio_adj, 6)
                if type_ratio_adj is not None else None),
            'split_status': ('rederived_floor+topn' if (w0 is not None
                                                         and bitten)
                             else 'rederived_floor' if w0 is not None
                             else 'rederived_topn'),
            # R3: >=asked mass on this emitted pair that entered the
            # induced graph but fired on NO chain (off-route mass +,
            # when the refill guard bit, guard-missed mass). Empty in
            # shortest mode (no second-round enumeration to reconcile).
            'unfired_residual': (
                int(asked_pair_mass.get(pair, 0) - emit_w - ref_w)
                if residual_active else ''),
        })

    ordered = sorted(refill_edges.items(),
                     key=lambda kv: (-_anchor_w(kv[0]), kv[0][0], kv[0][1]))
    capped = ordered[:max(0, int(detail_cap))]
    detail = [{
        'bodyId_pre': u, 'bodyId_post': v,
        'type_pre': type_map[u], 'type_post': type_map[v],
        'weight': _anchor_w((u, v)),
        'traversal_count': meta['traversals'],
        'min_hop': meta['hops'][0], 'max_hop': meta['hops'][-1],
    } for (u, v), meta in capped]

    # -- disclosure hygiene (real-data lesson: the per-edge boundary list
    # alone reached 40 KB on an FAFB query) — counts + capped examples +
    # per-type-pair aggregates, never unbounded edge dumps.
    def _edge_str(uv_w):
        u, v, w = uv_w
        return f'{u}->{v}({w:g})'

    def _cap_sample(items):
        ordered_sample = sorted(items, key=lambda t: (-t[2], t[0], t[1]))
        return [_edge_str(t) for t in ordered_sample[:_DISCLOSURE_SAMPLE_CAP]]

    skipped_by_pair: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {'count': 0, 'weight': 0.0})
    for u, v, w in skipped_edges:
        agg = skipped_by_pair[(type_map[u], type_map[v])]
        agg['count'] += 1
        agg['weight'] += w
    boundary_weight = sum(w for _u, _v, w in boundary)

    rec.update({
        'involved_labels': sorted(involved),
        'node_set_size': len(node_set),
        'induced_edges': len(e_u),
        'prefilter_dropped_in_U': prefilter_dropped,
        'estar_size': len(estar),
        'paths_asked': (len(_paths_asked) if _paths_asked is not None
                        else ask_stats.get('emitted')),
        'refill_path_budget': path_budget,
        'refill_truncated': refill_truncated,
        'cut_threshold': cut_threshold,
        'cut_paths_emitted': cut_stats.get('emitted'),
        'cut_tau': cut_stats.get('tau'),
        'emitted_rederived_size': len(cut_info),
        'table_reproduced': True,
        'refill_edges': len(refill_edges),
        # PHYSICAL synapse mass (the syn_of anchor in ratio mode; the
        # enumeration weight m['w'] is the F9 RATIO there — summing it
        # under a "synapses" label would be meaningless)
        'refill_weight_total': int(sum(
            _anchor_w(uv) for uv in refill_edges)),
        'detail_rows_written': len(detail),
        'detail_cap': int(detail_cap),
        'detail_capped': len(ordered) > len(capped),
        'boundary_refill_edge_count_at_or_above_w0': len(boundary),
        'boundary_refill_weight_at_or_above_w0': int(boundary_weight),
        'boundary_refill_examples_at_or_above_w0': _cap_sample(boundary),
        'excluded_edges_non_involved_endpoint': non_involved_excluded,
        'skipped_pairs_not_emitted_type_pair': {
            f'{k[0]}->{k[1]}': {'count': v['count'],
                                'weight': int(v['weight'])}
            for k, v in sorted(skipped_by_pair.items())},
        'skipped_edge_examples': _cap_sample(skipped_edges),
    })
    # ROUTE SCOPE disclosure: >=asked mass on non-emitted pairs pruned
    # from the second-round graph (out-of-scope routes are not built).
    rec['non_emitted_pair_mass'] = int(non_emitted_pair_mass)
    # RATIO-MODE FLOOR disclosure: the second round's synapse floor
    # (1 = the run's forced min_synapse; the AND-specified value under
    # 'and'; None on synapse-basis runs where `asked` is the floor).
    if ratio_mode:
        rec['refill_synapse_floor'] = float(and_syn_floor or 1)
    if write:
        # Standalone mirror folder: only for the DEFAULT in-run record
        # location — a custom --out dir means "leave the run folder
        # untouched" (the CLI byte-stability contract). Both statuses
        # fill: 'refilled' (cut mass) and 'complete_at_asked' (non-bitten
        # ratio runs — the complete view is the deliverable).
        build_mirrors = (rec.get('status') in ('refilled',
                                               'complete_at_asked')
                         and out_dir is None)
        if build_mirrors:
            # The refilled data in the path/connection data format.
            # Refill-only ratios reuse the pair's denominators (refill
            # mass / all-post incoming, and the threshold-conditioned
            # variant); the complete tables reuse the rows' refilled
            # columns verbatim.
            mirror_type_rows = []
            for r in rows:
                ref_w = int(r['refill_weight'])
                if ref_w <= 0:
                    continue
                denom = type_in.get(r['type_post'], 0.0)
                denom_adj = (type_in_adj.get(r['type_post'], 0.0)
                             if _min_syn_active else denom)
                plain = (ref_w / denom) if denom > 0 else None
                adj = (ref_w / denom_adj) if denom_adj > 0 else None
                mirror_type_rows.append({
                    'type_pre': r['type_pre'], 'type_post': r['type_post'],
                    'weight': ref_w, 'nt_type': '',
                    'connection_ratio': (round(plain, 6)
                                         if plain is not None else None),
                    'conn_layer': '',
                    'connection_ratio_adj': (round(adj, 6)
                                             if adj is not None else None),
                })
            complete_type_rows = [{
                'type_pre': r['type_pre'], 'type_post': r['type_post'],
                'weight': int(r['refilled_total']), 'nt_type': '',
                'connection_ratio': r['refilled_connection_ratio'],
                'conn_layer': '',
                'connection_ratio_adj': r['refilled_connection_ratio_adj'],
            } for r in rows]
            complete_bodyid_rows = []
            for (u, v), meta in list(cut_info.items()) + list(
                    refill_edges.items()):
                hops = meta.get('hops') or []
                ratio = pair_ratio.get((u, v))
                complete_bodyid_rows.append({
                    'bodyId_pre': u, 'bodyId_post': v,
                    'weight': _anchor_w((u, v)),
                    'roi': '', 'type_pre': type_map[u],
                    'type_post': type_map[v], 'instance_pre': '',
                    'instance_post': '', 'nt_type': '', 'nt_type_pre': '',
                    'custom_group_pre': '', 'custom_group_post': '',
                    'connection_ratio': (round(float(ratio), 6)
                                         if ratio is not None else None),
                    'synapse': '',
                    'traversal_count': meta.get('traversals'),
                    'min_hop': hops[0] if hops else None,
                    'max_hop': hops[-1] if hops else None,
                })
            complete_bodyid_rows.sort(
                key=lambda r: (-r['weight'], str(r['bodyId_pre']),
                               str(r['bodyId_post'])))
            # The second path-building round's product: the REFILLED
            # type-level paths (asked-threshold enumeration restricted to
            # paths carrying >=1 refill edge — the refill edges already
            # carry the emitted-pairs restriction).
            # The Pair Explorer's chain scope (emitted type sequences).
            cut_chain_keys = {
                tuple(type_map[n] for n in nodes)
                for _bn, nodes in (_emitted_paths or [])}
            path_rows, refill_bodyid_paths = _aggregate_refill_paths(
                _paths_asked, set(refill_edges.keys()), cut_chain_keys,
                type_map, node_set, _anchor_w, type_in)
            rec['refill_paths_type'] = len(path_rows)
            rec['refill_paths_bodyid'] = refill_bodyid_paths
            # R1: the COMPLETE-AT-ASKED view — the SAME chains re-valued
            # at the asked threshold (values change, structure does not).
            # All-mode only (the shortest branch leaves _paths_asked
            # None so the table is skipped and the stamp stays null).
            if _paths_asked:
                complete_rows, complete_stats = _aggregate_complete_paths(
                    _paths_asked, set(cut_info.keys()),
                    set(refill_edges.keys()), cut_chain_keys,
                    type_map, node_set, _anchor_w, type_in)
                rec['complete_at_asked'] = (not refill_truncated)
                rec['complete_paths_type'] = complete_stats['chains']
                rec['complete_paths_bodyid'] = complete_stats[
                    'bodyid_paths']
            else:
                complete_rows = None
                rec['complete_at_asked'] = None
            if complete_only:
                rec['status'] = 'complete_at_asked'
            rec['mirror_folder'] = _MIRROR_FOLDER_NAME
        # DUAL REFILL: the full scope rides on every all-mode record —
        # attached BEFORE the record write (fix 2026-10-09: the
        # provenance json serialized without it, so consumers never saw
        # the full block even though the files were written).
        full_block = _produce_full_scope(
            chain_keys={tuple(type_map[n] for n in nodes)
                        for _bn, nodes in (_emitted_paths or [])},
            cut_edge_keys=set(cut_info))
        if full_block is not None:
            rec['full'] = full_block
        _write_records(run_dir, out_dir, rec, rows, detail, detail_cap)
        if build_mirrors:
            _write_mirrors(run_dir, mirror_type_rows, complete_type_rows,
                           complete_bodyid_rows, path_rows,
                           complete_path_rows=complete_rows)
    return rec


# ---------------------------------------------------------------------------
# Record writers
# ---------------------------------------------------------------------------
_README_TEXT = """# Type-level refill records

Standalone refill of the type-level connection strength for a run whose
applied threshold exceeded the asked threshold (Edge Budget floor and/or
StrongestFirst path budget). Generated by `type_level_refill` (plan:
`_plan/plan-type-level-refill.md`); no run file was modified.

Files
-----
- `refill_type_pairs.csv` — one row per EMITTED type pair (zeros
  included). `refilled_total = emitted_weight + refill_weight` is the
  refilled strength; `refilled_connection_ratio` uses the threshold-free
  all-post denominator (F9) and `refilled_connection_ratio_adj` the
  threshold-conditioned one (edges at/above the run's Min Synapse Count;
  equals the plain ratio when no synapse threshold is active). Refill
  mass counts only bodyId pairs on
  simple source->target paths within the run's hop bound on the subgraph
  induced by the involved types' neurons — a certified lower bound of the
  full complete-at-asked refill (paths that leave the induced subgraph
  are disclosed in the provenance, not refilled).
- `refill_bodyId_pairs.csv` — the refilled bodyId pairs (capped,
  deterministic weight-desc order; the summary above is never capped).
- `type_incoming_asked.csv` — the >=threshold incoming per post type
  (the adjusted-ratio conditioned denominator at the asked tier,
  exported since 2026-10-10); the pair report's refill-asked Adj.
  ratios divide the chains' realized hop masses by it.
- `refill_provenance.json` — thresholds, budgets, node/edge counts,
  `table_reproduced` (the re-derived cut reproduced the exported
  `connection_type.csv` exactly — the consistency anchor), and the
  disclosure counters (`skipped_pairs_not_emitted_type_pair` census +
  capped examples — empty under the 2026-10-07 route scope, which
  discloses pruned mass as `non_emitted_pair_mass` instead), `excluded_edges_non_involved_endpoint`,
  `boundary_refill_edge_count/weight/examples_at_or_above_w0`,
  `refill_truncated`), plus the DUAL REFILL `full` block (2026-10-08).

Refill-full (DUAL REFILL, 2026-10-08)
-------------------------------------
The second kind: the second round admitted at SYNAPSE >= 1 ONLY (no
ratio tier, no synapse tier) — the full-connection view of the SAME
emitted chain sequences, produced for EVERY all-mode run with emitted
rows and restricted to the emitted chains (2026-10-11 — chain SCOPING
bounds the cost without cutting the weak sub-threshold paths this
scope exists for). Since 2026-10-13 each chain's realization is the
LAYERED PER-POSITION BODY-SET FILTER (plan-corridor-ratio-basis
§8.3/§9, universal at every applied threshold): forward body sets from
the run's sources and backward sets to its targets over the chain's
hop pairs, intersected per position; per-hop mass = the kept edges'
weight sum. NO bodyId paths are materialized (no traversal/path
counts anywhere). Chains WITHOUT an effective-label revisit are exact
(equal to the retired DP's realization set); label-REVISITING chains
carry the layered-walk superset semantics (a body may occupy two
positions and contributes its edges at both; an edge recurring at two
positions counts once per position) — counted in the provenance as
`full.chains_revisit` with up to five `full.chains_revisit_examples`.
The retired `full.paths`/`full.bodyid_paths` counters are null;
nothing truncates (`full.chains_truncated` = 0; the former per-chain
DP/DFS enumeration with its 2M-path caps is retired). Files:
- `refill_full_type_pairs.csv` (this folder) — per emitted pair:
  emitted / refill-full (sub-threshold mass) / refilled_total +
  `refilled_connection_ratio` (the plain full ratio; at floor 1 it is
  also the adjusted value) + `pair_offchain_mass` (the pair's
  floor-passing mass kept by NO chain's layered filter — the chain
  scope is a lower bound of the raw pair total by design,
  and this column makes the gap self-describing). The full scope is
  CACHE-GUARDED: the refill refuses when the stream-derived all-post
  incoming is smaller than the run's own exported applied-tier
  incoming (`type_incoming_applied.csv`) — an incomplete connection
  cache cannot silently undercount the full scope (stamped
  `full.cache_guard: 'ok'` in the provenance).
- mirror `refill_full_paths_type.csv` — every emitted chain re-valued
  at synapse >= 1 (complete at full).
- mirror `refill_full_carrying_paths_type.csv` — the chains that carry
  sub-threshold mass at full scope.
- mirror `refilled_full_connection_type.csv` — the full pair totals in
  connection-table format.

Join recipe
-----------
`emitted_weight` is the run's PHYSICAL edge mass per type pair: since
2026-10-06 it dedups `data_details/connection_info_bodyId.csv` by
`(bodyId_pre, bodyId_post)` (keep-first) before aggregating — in
shortest runs the same edge is fetched once per backward BFS layer it
recurs in, and summing `connection_type.csv`'s per-layer rows instead
multiplied the mass by that fetch multiplicity (measured 2.18x on a
real FAFB run). Runs without bodyId detail (`skip_bodyId=True`) fall
back to the conn-table sums (per-layer semantics, disclosed).

`unfired_residual` (refill_type_pairs.csv)
------------------------------------------
Every >=asked edge on an EMITTED type pair that entered the induced
graph must fire on some ENUMERATED asked-threshold path (any chain,
not only the Pair Explorer's chain scope — the pair totals and the
chain tables are intentionally different scopes); mass that fires on
no enumerated path (off-route geometry) is disclosed per pair as
RATIO-MODE FLOOR (2026-10-07): plain ratio-basis refills always run
at synapse >= 1 (the run's own forced min_synapse; the AND-specified
value under threshold_combination='and'), and the F9 denominators sum
only w >= 1 edges — matching the run's min_weight=1 fetch so the
second round's universe and ratios are complete. Stamped as
`refill_synapse_floor`.

`unfired_residual = asked_pair_mass - emitted_weight - refill_weight`
(0 on clean runs; nonzero = off-route mass +, when the refill guard
bit, guard-missed mass + pruned non-emitted-pair mass on the pair).
Empty in shortest mode (no second-round enumeration to reconcile
against).

Partially-finite ratio rows: a path row whose post type has zero F9
incoming mass renders a BLANK `ratios` list (per-hop shares need every
denominator); its `min_ratio` is the minimum over the FINITE hops
only. Zero occurrences on healthy data (every involved type has
incoming mass by construction).

Complete-at-asked table (`complete_at_asked_paths_type.csv`)
------------------------------------------------------------
The mirror folder also carries the COMPLETE-AT-ASKED view: the simple
source->target paths of the PAIR EXPLORER'S chain scope (ROUTE SCOPE,
2026-10-07: the second-round graph is pruned to emitted-pair edges, so
out-of-scope routes are never built) within the run's hop bound on the
induced subgraph at the ASKED threshold
(the emitted type routes — values change, structure does not; chains
realized only below the applied tier are excluded, and the second-round
GRAPH is pruned to emitted-pair edges so out-of-scope routes are never
pathfinded), aggregated into type chains with the same engine recipe as
`refill_paths_type.csv`. Each hop's realized mass is split two ways:
`emitted_weights` + `refill_weights` == `weights` (the route-scope
restriction prunes non-emitted-pair edges, so the former skipped class
cannot occur — the pruned mass is disclosed at pair level as
`non_emitted_pair_mass` / `unfired_residual`).
Completeness is a per-run EMPIRICAL property, stamped in
`refill_provenance.json` as `complete_at_asked` (true = the table is
the complete simple-path set at the asked tier; false = complete only
at the drain tau — the refill guard budget bit during the second
round; null = shortest mode, where the table is not produced). The
pair report's Refill tab defaults to this view.

Standalone mirror folder
------------------------
When generated inside the run flow, the refilled data is ALSO mirrored
into `<run_folder>/type_level_refill/` (parallel to `find_reciprocal/`)
in the path/connection data format, with heatmap/network visualizations
built by the run flow:

- `refill_connection_type.csv` — the REFILL mass only, in the
  `connection_type.csv` column order (`connection_ratio` = refill mass
  over the threshold-free all-post denominator, `connection_ratio_adj`
  the threshold-conditioned one; `nt_type`/`conn_layer`
  are not derivable for refilled edges and stay empty). Probability
  columns are SUSPENDED (2026-10-06, unverifiable fidelity).
- `refilled_connection_type.csv` — complete-at-asked: one row per
  emitted pair with `weight = refilled_total` (emitted + refill) and
  the pair's refilled ratios.
- `refilled_connection_bodyId.csv` — complete-at-asked bodyId edges
  (emitted-rederived + refill), the canonical connection-table columns
  plus `traversal_count`/`min_hop`/`max_hop`; enrichment columns the
  module cannot derive stay empty.
- `refill_paths_type.csv` — the SECOND path-building round's product:
  the refilled type-level paths (asked-threshold enumeration on the
  induced subgraph, restricted to paths carrying >=1 refill edge and to
  emitted pairs), in the exact `*_allpaths_type.csv` column order.
  Per-hop weights are the REALIZED masses (distinct edges realizing the
  hop position); per-hop ratios use the threshold-free all-post
  denominator; coverage `n/N` counts distinct bodyIds realizing the
  node in this path set over the type's membership in the induced node
  set (not the run's discovered-network N).
"""


_DETAIL_COLUMNS = (
    'bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight',
    'traversal_count', 'min_hop', 'max_hop')

# Standalone mirror folder (<run>/type_level_refill/, parallel to
# find_reciprocal/): the refilled data in the path/connection data
# format. Column orders mirror data_details/connection_type.csv and the
# canonical bodyId connection table (_PATH_CONN_KEEP_COLS in coana).
_MIRROR_FOLDER_NAME = 'type_level_refill'

# SUSPENDED (2026-10-06): the probability columns
# (traversal_probability/block_probability) are withheld — unverifiable
# fidelity. The connection-table mirrors keep the remaining
# connection_type.csv column order.
_TYPE_MIRROR_COLUMNS = (
    'type_pre', 'type_post', 'weight', 'nt_type', 'connection_ratio',
    'conn_layer', 'connection_ratio_adj')

_BODYID_MIRROR_COLUMNS = (
    'bodyId_pre', 'bodyId_post', 'weight', 'roi', 'type_pre', 'type_post',
    'instance_pre', 'instance_post', 'nt_type', 'nt_type_pre',
    'custom_group_pre', 'custom_group_post', 'connection_ratio',
    'synapse',
    'traversal_count', 'min_hop', 'max_hop')

# The refilled PATHS (the second path-building round's product) in the
# EXACT *_allpaths_type.csv column order, so the pair report's own
# loaders consume the file unchanged.
# SUSPENDED (2026-10-06): the probability columns (``probabilities``,
# ``path_prob``) are withheld — unverifiable fidelity. The order mirrors
# the allpaths schema minus those columns.
_PATHS_MIRROR_COLUMNS = (
    'path', 'weights', 'ratios', 'min_weight',
    'min_ratio', 'length', 'nt_types', 'coverage')


def _fmt_mass(value) -> str:
    return str(int(value)) if float(value).is_integer() else str(
        float(value))


def _refill_rows_from_structures(seq_edges, seq_nodes, type_n, type_in):
    """Row assembly for the REFILLED path tables from per-chain
    structures ({seq: per-hop {(u,v): anchor weight}}, {seq: per-hop
    body sets}) — shared by the path-fed asked view and the DP-fed
    full scope (2026-10-11)."""
    rows = []
    for seq in sorted(seq_edges,
                      key=lambda s: (s[0], s[-1], len(s), s)):
        edges_at = seq_edges[seq]
        nodes_at = seq_nodes[seq]
        weights = [sum(edge.values()) for edge in edges_at]
        ratios = []
        for i, post in enumerate(seq[1:]):
            denom = type_in.get(post, 0.0)
            ratios.append((weights[i] / denom) if denom > 0 else None)
        finite = [r for r in ratios if r is not None]
        coverage = '[' + ', '.join(
            f'{len(nodes_at[i])}/{type_n.get(label, 0)}'
            for i, label in enumerate(seq)) + ']'
        ratios_txt = ('[' + ', '.join(
            'None' if r is None else repr(r) for r in ratios) + ']'
            if ratios else '')
        rows.append({
            'path': '->'.join(seq),
            'weights': '[' + ', '.join(
                _fmt_mass(w) for w in weights) + ']',
            'ratios': ratios_txt,
            'min_weight': min(weights) if weights else 0,
            'min_ratio': (min(finite) if finite else None),
            'length': len(seq) - 1,
            'nt_types': '',
            'coverage': coverage,
        })
    rows.sort(key=lambda r: (-r['min_weight'], r['path']))
    return rows


def _complete_rows_from_structures(seq_cls, seq_nodes, type_n, type_in):
    """Row assembly for the COMPLETE-AT-ASKED path tables from per-chain
    classed structures — shared by the path-fed asked view and the
    DP-fed full scope (2026-10-11)."""
    rows = []
    for seq in sorted(seq_cls,
                      key=lambda s: (s[0], s[-1], len(s), s)):
        classes_at = seq_cls[seq]
        nodes_at = seq_nodes[seq]
        weights, emitted_w, refill_w = [], [], []
        ratios = []
        for i, post in enumerate(seq[1:]):
            per_cls = {cls: sum(edge.values())
                       for cls, edge in classes_at[i].items()}
            weights.append(sum(per_cls.values()))
            emitted_w.append(per_cls.get('emitted', 0.0))
            refill_w.append(per_cls.get('refill', 0.0))
            denom = type_in.get(post, 0.0)
            ratios.append((weights[-1] / denom) if denom > 0 else None)
        finite = [r for r in ratios if r is not None]
        coverage = '[' + ', '.join(
            f'{len(nodes_at[i])}/{type_n.get(label, 0)}'
            for i, label in enumerate(seq)) + ']'
        ratios_txt = ('[' + ', '.join(
            'None' if r is None else repr(r) for r in ratios) + ']'
            if ratios else '')

        def _arr(vals):
            return '[' + ', '.join(_fmt_mass(v) for v in vals) + ']'

        rows.append({
            'path': '->'.join(seq),
            'weights': _arr(weights),
            'ratios': ratios_txt,
            'min_weight': min(weights) if weights else 0,
            'min_ratio': (min(finite) if finite else None),
            'length': len(seq) - 1,
            'nt_types': '',
            'coverage': coverage,
            'emitted_weights': _arr(emitted_w),
            'refill_weights': _arr(refill_w),
        })
    rows.sort(key=lambda r: (-r['min_weight'], r['path']))
    stats = {'chains': len(rows), 'bodyid_paths': 0}
    return rows, stats


def _aggregate_refill_paths(paths_asked, refill_edge_keys, chain_keys,
                            type_map, node_set, anchor_w, type_in):
    """Aggregate the REFILLED bodyId paths — the asked-threshold
    enumeration on the involved-types induced subgraph (the second
    path-building round) restricted to paths carrying >=1 refill edge.
    The refill edges themselves were already restricted to EMITTED type
    pairs (refill-only semantics: non-emitted pairs are counted, not
    recovered), so this filter carries the restriction — a path's
    ENDPOINT pair must NOT be checked against the emitted set (emitted
    holds EDGE pairs from connection_type.csv; a source->target type
    pair generally has no direct edge).

    Engine recipe (coana rounds 9-11): per-hop REALIZED mass = the sum
    of the DISTINCT bodyId edges realizing that hop position; per-hop
    ratios = realized mass over the post type's full-membership incoming
    mass (the module's F9 ``type_in`` denominator —
    threshold-free all-post, matching the exported Ratios column;
    adjusted variants live in the pair tables, not the path tables); ``min_weight``/``min_ratio`` = the weakest hop;
    coverage ``n/N`` = distinct bodyIds realizing the node in THIS path
    set over the type's membership in the induced node set (not the
    run's discovered-network N — disclosed in the README). Probability
    columns (probabilities/path_prob) are SUSPENDED (2026-10-06,
    unverifiable fidelity) and are not emitted.
    Returns (rows in _PATHS_MIRROR_COLUMNS order, bodyId path count)."""
    type_n: Dict[str, int] = defaultdict(int)
    for b in node_set:
        label = type_map.get(str(b))
        if label is not None:
            type_n[label] += 1
    seq_edges: Dict[Tuple[str, ...], List[Dict[Tuple[str, str], float]]] = {}
    seq_nodes: Dict[Tuple[str, ...], List[set]] = {}
    bodyid_count = 0
    for item in (paths_asked or []):
        nodes = item[1]
        seq = tuple(type_map[n] for n in nodes)
        # chain scope: only chains the Pair Explorer already shows (the
        # emitted set) — the refill is a VALUE change on those chains,
        # never a structural one
        if seq not in chain_keys:
            continue
        carries_refill = any(
            (nodes[i], nodes[i + 1]) in refill_edge_keys
            for i in range(len(nodes) - 1))
        if not carries_refill:
            continue
        bodyid_count += 1
        edges_at = seq_edges.get(seq)
        if edges_at is None:
            edges_at = [{} for _ in range(len(seq) - 1)]
            seq_edges[seq] = edges_at
            seq_nodes[seq] = [set() for _ in range(len(seq))]
        nodes_at = seq_nodes[seq]
        for i in range(len(seq) - 1):
            key = (nodes[i], nodes[i + 1])
            edges_at[i][key] = anchor_w(key)
        for i, n in enumerate(nodes):
            nodes_at[i].add(n)
    rows = _refill_rows_from_structures(seq_edges, seq_nodes, type_n,
                                        type_in)
    return rows, bodyid_count


# SKIPPED column DROPPED (2026-10-07): the route-scope restriction
# prunes non-emitted-pair edges before enumeration, so the skipped class
# cannot occur — the split is emitted + refill == weights.
_COMPLETE_PATHS_COLUMNS = _PATHS_MIRROR_COLUMNS + (
    'emitted_weights', 'refill_weights')


def _aggregate_complete_paths(paths_asked, cut_keys, refill_keys,
                              chain_keys, type_map, node_set, anchor_w,
                              type_in):
    """Aggregate the COMPLETE-AT-ASKED path set — EVERY path in
    ``paths_asked`` (the asked-threshold enumeration on the involved-
    types induced subgraph), NO filter of any kind. In particular a
    path's ENDPOINT pair is never checked against the emitted set
    (emitted holds EDGE pairs from connection_type.csv; a source->target
    type pair generally has no direct edge — standing round-10 lesson).

    Same engine recipe as :func:`_aggregate_refill_paths` (realized
    per-hop masses over the F9 ``type_in`` denominator, threshold-
    conditioned ``type_in_adj`` variant, coverage over induced
    membership), with a two-way per-(chain, hop, edge) classification
    recorded at insertion: ``edge in cut_keys -> 'emitted'``, otherwise
    ``'refill'``. The former 'skipped' class CANNOT occur: the route-
    scope restriction prunes non-emitted-pair edges before enumeration,
    so every enumerated edge is cut or refill (the pruned mass is
    disclosed at pair level as ``non_emitted_pair_mass`` /
    ``unfired_residual``). Probability columns are SUSPENDED
    (2026-10-06) — not emitted.

    Returns (rows carrying _COMPLETE_PATHS_COLUMNS, stats_dict with
    ``chains``/``bodyid_paths``)."""
    type_n: Dict[str, int] = defaultdict(int)
    for b in node_set:
        label = type_map.get(str(b))
        if label is not None:
            type_n[label] += 1
    # per chain: per-hop edge dicts per class + per-node bodyId sets
    seq_cls: Dict[Tuple[str, ...], List[Dict[str, Dict[Tuple[str, str], float]]]] = {}
    seq_nodes: Dict[Tuple[str, ...], List[set]] = {}
    bodyid_count = 0
    for item in (paths_asked or []):
        nodes = item[1]
        seq = tuple(type_map[n] for n in nodes)
        # chain scope: the Pair Explorer's chains only — the complete view
        # re-values THOSE chains at the asked threshold; chains realized
        # only below the applied tier are structural additions and are
        # excluded (their mass stays disclosed via unfired_residual)
        if seq not in chain_keys:
            continue
        bodyid_count += 1
        classes_at = seq_cls.get(seq)
        if classes_at is None:
            classes_at = [{} for _ in range(len(seq) - 1)]
            seq_cls[seq] = classes_at
            seq_nodes[seq] = [set() for _ in range(len(seq))]
        nodes_at = seq_nodes[seq]
        for i in range(len(seq) - 1):
            key = (nodes[i], nodes[i + 1])
            w = anchor_w(key)
            # route scope: every enumerated edge is cut or refill
            cls = 'emitted' if key in cut_keys else 'refill'
            classes_at[i].setdefault(cls, {})
            classes_at[i][cls][key] = w
        for i, n in enumerate(nodes):
            nodes_at[i].add(n)
    rows, _stats = _complete_rows_from_structures(seq_cls, seq_nodes,
                                                  type_n, type_in)
    stats = {'chains': _stats['chains'],
             'bodyid_paths': bodyid_count}
    return rows, stats


def _write_mirrors(run_dir: Path, refill_type_rows, complete_type_rows,
                   complete_bodyid_rows, path_rows=None,
                   complete_path_rows=None):
    """Write the standalone mirror folder. The visualizations themselves
    are built later by the run flow's visualization stage (needs the
    VisualizePath context); the CLI regenerates the CSVs only."""
    out = Path(run_dir) / _MIRROR_FOLDER_NAME
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        refill_type_rows, columns=list(_TYPE_MIRROR_COLUMNS)).to_csv(
        out / 'refill_connection_type.csv', index=False)
    pd.DataFrame(
        complete_type_rows, columns=list(_TYPE_MIRROR_COLUMNS)).to_csv(
        out / 'refilled_connection_type.csv', index=False)
    pd.DataFrame(
        complete_bodyid_rows,
        columns=list(_BODYID_MIRROR_COLUMNS)).to_csv(
        out / 'refilled_connection_bodyId.csv', index=False)
    if path_rows is not None:
        pd.DataFrame(
            path_rows, columns=list(_PATHS_MIRROR_COLUMNS)).to_csv(
            out / 'refill_paths_type.csv', index=False)
    if complete_path_rows is not None:
        pd.DataFrame(
            complete_path_rows,
            columns=list(_COMPLETE_PATHS_COLUMNS)).to_csv(
            out / 'complete_at_asked_paths_type.csv', index=False)
    # parameters.csv completes the find_reciprocal-style mirror; the run
    # flow's hook writes it from the live parameter frame, the post-hoc
    # CLI copies the run's own data_details copy.
    run_params = Path(run_dir) / 'data_details' / 'parameters.csv'
    if run_params.exists():
        import shutil
        shutil.copyfile(run_params, out / 'parameters.csv')


def _write_records(run_dir: Path, out_dir, rec: dict, rows, detail,
                   detail_cap: int):
    out = Path(out_dir) if out_dir else (
        run_dir / 'data_details' / REFILL_FOLDER_NAME)
    out.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        pd.DataFrame(rows).to_csv(out / 'refill_type_pairs.csv', index=False)
        # Header-only when the refill is empty (never a zero-byte file).
        frame = (pd.DataFrame(detail, columns=list(_DETAIL_COLUMNS))
                 if not detail else pd.DataFrame(detail))
        frame.to_csv(out / 'refill_bodyId_pairs.csv', index=False)
        # (Re)write the README when missing OR predating a section — the
        # only-if-absent rule alone left older runs' READMEs stale when
        # the documented outputs grew (the standalone mirror folder).
        readme = out / 'README.md'
        if (not readme.exists()
                or 'Standalone mirror folder' not in readme.read_text(
                    encoding='utf-8')
                or 'ROUTE SCOPE' not in readme.read_text(
                    encoding='utf-8')
                or 'LAYERED PER-POSITION BODY-SET FILTER' not in
                readme.read_text(encoding='utf-8')):
            readme.write_text(_README_TEXT, encoding='utf-8')
    (out / 'refill_provenance.json').write_text(
        json.dumps(rec, indent=2, sort_keys=True, default=str),
        encoding='utf-8')
