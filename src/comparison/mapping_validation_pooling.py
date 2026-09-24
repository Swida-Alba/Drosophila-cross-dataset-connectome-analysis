"""TM VEV `pooling` mode — the unsupervised homolog finder, per source.

Normative design: ``_plan/plan-tmvev-pooling-mode.md`` (the mode) and
``_plan/plan-tmvev-pooling-tiers.md`` (the tier ladder, settled 2026-09-24 after
eight review rounds).  Where this pipeline's other modes ask "what else belongs
to the pool the type mapper already asserted" (supervised), pooling asks "what do
connectivity and morphology say is a homolog of the queried population, over the
whole opposite universe" and only THEN compares that answer with the mapper
(§4.5).  Its unit is the QUERIED SOURCE: every source keeps the top-N of each
chosen metric, and the tiers grade that finding.

The unsupervised property is load-bearing, so it is stated as an invariant:
**no function here may use a mapper claim to decide whether a row is a
candidate.**  The mapper is read after the gate, to label cells and leaf tokens.
A unit test enforces it by making the claim lookups raise.

The floors are FLAGS, not filters (user 2026-09-24).  ``pooling_jaccard_floor`` /
``pooling_rank_union_floor`` / ``pooling_window_mult`` are still evaluated and
published per row, and they remove nothing: measured on the landed run, the
``rank_union > 0.0`` floor was the sole reason 119 of 242 queried sources
reported nothing — a sign test on the data, not the volume control it was
documented as.  What bounds the scan now is the bar (``pooling_bar_metric`` x
``pooling_bar_top_n``) plus a per-source row cap, because ``rank_union``'s tie
mass makes a rank cut unbounded (17.1 rows/source at N=5 against jaccard's 5.1).
An earlier build also fitted the Jaccard floor to the dataset pair's own graded
evidence; that measured a quality question a volume knob cannot answer, so the
fit was deleted rather than tuned (plan-tmvev-pooling-mode.md §5).

Morphology is MANDATORY here: all three tiers are defined morph-qualified, so a
pooling run without it would publish connectivity-only rows under claim-shaped
names — see :func:`check_morphology_mandatory`.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

#: The mode string.  Deliberately NOT in ``VALIDATION_MODES`` / ``MODE_RANK``:
#: pooling is parallel to the nested restrictive<family<aggressive chain, and
#: adding it there would make every ``mode_at_least`` comparison admit it.
POOLING_MODE = 'pooling'

POOLING_CANDIDATES_CSV = 'pooling_candidates.csv'
POOLING_POOL_CSV = 'pooling_pool.csv'
POOLING_XVAL_JSON = 'pooling_cross_validation.json'

#: Pooling's own tier ladder, graded PER ROW.  `nominated` is deliberately not
#: called `candidates`: that word already names the supervised restrictive fill
#: bin (`EXPANSION_CATEGORIES`), and one report cannot carry two meanings in one
#: word.  The ladder is also NOT in `TIER_CATEGORIES`/`DEDUP_RANK` — pooling is
#: parallel to the nested chain, so its vocabulary must not widen the supervised
#: partition or the auditor's nesting checks.
POOLING_TIERS = ('matched', 'verified', 'nominated')
BAR_METRICS = ('either', 'jaccard', 'rank_union')
#: `rank_union` ranks tie at exactly 0 across hundreds of targets, so a rank cut
#: is not a row bound (measured: 17.1 rows/source at N=5, against jaccard's 5.1).
#: Each source therefore keeps this many x N rows in the chain order, and the
#: number the cap cut is published (`bar['rows_cut']`).
ROW_CAP_MULTIPLE = 2
#: The auto morph budget, in units per QUERIED SOURCE.  A unit is one pair the
#: last gate looks at (every tier-1 row, plus each target's chain-best row), so
#: the default bar measured ~2.3 per source on FAFB->male-cns; 3 leaves headroom
#: for a pair whose rank_union ties harder, and a cap that still bites reports
#: itself in `morph.capped` rather than silently shrinking the picture.
MORPH_UNITS_PER_SOURCE = 3
POOLING_SOURCES_CSV = 'pooling_sources.csv'


def bar_rank_of(jaccard_rank, rank_union_rank, metric: str = 'either') -> float:
    """The rank this row earned under the bar.

    `either` is the UNION of each metric's own top-N — the same reading the
    `matched` tier has always used — and never a merged best-rank ordering: a
    merged ordering spends the N slots on the two metrics' rank-1 rows (162 of
    242 sources have two distinct ones), which was measured to keep only 79 of
    the 118 targets the floors found where the union keeps 116.
    """
    if metric == 'jaccard':
        return float(jaccard_rank)
    if metric == 'rank_union':
        return float(rank_union_rank)
    return min(float(jaccard_rank), float(rank_union_rank))


def tier_of(*, bar_rank, rank_union, top_n, matched_ru_min) -> str:
    """The first-match tier of one row, BEFORE morphology qualifies it.

    `matched` keeps rank_union > `matched_ru_min` as a GRADE (it is the
    supervised `matched_ru_min`, so the two engines assert the same thing); it is
    not an admission test, and a row below it stays as `verified`.
    """
    if bar_rank is None:
        return ''
    if float(bar_rank) <= 1:
        ru = (None if rank_union is None
              else (np.nan if isinstance(rank_union, float) and np.isnan(rank_union)
                    else float(rank_union)))
        if ru is not None and not np.isnan(ru) and ru > float(matched_ru_min):
            return 'matched'
        return 'verified'
    if float(bar_rank) <= float(top_n):
        return 'nominated'
    return ''


def admit_by_bar(arrays: Dict[str, Any], *, source_bodyId: int,
                 source_type: str, metric: str, top_n: int,
                 floors: Dict[str, float], window: int) -> Tuple[List[Dict],
                                                                 int]:
    """One source's decision: which of its scan rows the bar admits.

    `arrays` is the scan frame for this source (`target_bid`, `jaccard`,
    `jaccard_rank`, `rank_union`, `rank_union_rank`).  The floors are EVALUATED
    and never applied: a row below one is flagged and kept, because the only
    thing that decides admission is the bar.  Returns the rows in chain order
    (jaccard desc -> rank_union desc -> bodyId, the same key every assignment
    and gap fill uses) and the number of admitted rows the row cap removed.
    """
    bids = np.asarray(arrays['target_bid'])
    jac = np.asarray(arrays['jaccard'], dtype=float)
    ru = np.asarray(arrays['rank_union'], dtype=float)
    jr = np.asarray(arrays['jaccard_rank'], dtype=float)
    rr = np.asarray(arrays['rank_union_rank'], dtype=float)
    ranks = np.minimum(jr, rr) if metric == 'either' else (
        jr if metric == 'jaccard' else rr)
    rows: List[Dict] = []
    for i in np.flatnonzero(ranks <= int(top_n)):
        rows.append({
            'source_bodyId': int(source_bodyId),
            'source_type': source_type,
            'target_bodyId': int(bids[i]),
            'jaccard': float(jac[i]), 'jaccard_rank': int(jr[i]),
            'rank_union': float(ru[i]), 'rank_union_rank': int(rr[i]),
            'bar_rank': int(ranks[i]), 'bar_metric': metric,
            'bar_top_n': int(top_n), 'window_size': int(window),
            # flags, not filters: each names the configured number it was
            # measured against, and none of them removed this row
            'below_jaccard_floor': bool(jac[i] <= float(floors['jaccard'])),
            'below_rank_union_floor': bool(ru[i] <= float(floors['rank_union'])),
            'outside_window': bool(jr[i] > window or rr[i] > window),
        })
    rows.sort(key=lambda r: (-r['jaccard'], -r['rank_union'],
                             r['target_bodyId']))
    cap = max(int(top_n) * ROW_CAP_MULTIPLE, 1)
    return rows[:cap], max(0, len(rows) - cap)


#: Post-hoc cells (§4.5), keyed on the target bodyId.
CELL_CONFIRMED = 'confirmed'          # pool ∩ mapper-refined target pool
CELL_POOL_MISS = 'pool_miss'          # the mapper never named this neuron
CELL_VERIFIED_ONLY = 'verified_only'  # the mapper's pair fails the abs bar
CELL_TYPE_MISS = 'type_miss'          # pool_miss, of an in-map type
CELL_TYPE_NEW = 'type_new'            # pool_miss, of a type outside the map


def seed_population(validator, query_types: Sequence[str]) -> List[int]:
    """Every source neuron the query names, by the SOURCE dataset's own
    annotation — the union over the query tokens, so the residue the branches
    do not claim is in the seed (plan §0.5).

    ``_bodyids_for`` is the pipeline's type-pool resolver; it is a dataset
    lookup, not a mapping claim: the same call stage 1 uses to build the
    populations the branches are then refined FROM.
    """
    out: List[int] = []
    seen = set()
    for token in query_types:
        for bid in validator._bodyids_for(str(token),
                                          validator.cfg.source_dataset) or []:
            try:
                b = int(bid)
            except (TypeError, ValueError):
                continue
            if b not in seen:
                seen.add(b)
                out.append(b)
    return sorted(out)


def window_sizes(source_types: Dict[int, str], mult: float) -> Dict[int, int]:
    """Per-source rank window: ``mult`` x the size of THAT SOURCE TYPE's
    queried population.

    The window is the query-scaled half of the gate, and it must scale to
    something the unsupervised run knows: the source type's own population.
    A branch's claimed pool is the supervised quantity, so it is not used.
    """
    counts: Dict[str, int] = {}
    for t in source_types.values():
        key = str(t or '')
        counts[key] = counts.get(key, 0) + 1
    return {bid: max(1, int(round(float(mult) * counts.get(str(tt or ''), 1))))
            for bid, tt in source_types.items()}


def connectivity_candidates(validator, seed: Sequence[int],
                            source_types: Dict[int, int],
                            windows: Dict[int, int], target_stats,
                            target_bids: Sequence[int],
                            jaccard_floor: float) -> Tuple[List[Dict], Dict]:
    """Scan the seed against the WHOLE target universe and admit each source's
    bar (plan-tmvev-pooling-tiers.md §2).

    The loop is the scan; the decision is :func:`admit_by_bar`, which is pure so
    the bar can be tested without a profiler.  `jaccard_floor` and the other two
    floor numbers are passed through as the FLAGS they now are.
    """
    from comparison.body_id_resolver import scan_source
    from comparison.mapping_validation import expanded_vector

    cfg = validator.cfg
    metric = str(cfg.pooling_bar_metric or 'either')
    if metric not in BAR_METRICS:
        raise ValueError(f'unknown pooling bar metric {metric!r}; expected one '
                         f'of {", ".join(BAR_METRICS)}')
    top_n = int(cfg.pooling_bar_top_n)
    if top_n < 1:
        raise ValueError('pooling_bar_top_n must be >= 1')
    floors = {'jaccard': float(jaccard_floor),
              'rank_union': float(cfg.pooling_rank_union_floor)}
    rows: List[Dict] = []
    cut = 0
    t0 = time.time()
    scored = positives = scanned = no_profile = 0
    for bid in seed:
        prof = validator.profiler.get_profile(bid, cfg.source_dataset)
        vec = expanded_vector(prof, validator.mapper) if prof else None
        if not vec:
            no_profile += 1
            continue
        fr = scan_source(vec, target_stats, list(target_bids))
        if fr is None or fr.empty:
            no_profile += 1
            continue
        scanned += 1
        scored += len(fr)
        jac_all = fr['jaccard'].to_numpy(dtype=float)
        positives += int(np.count_nonzero(np.isfinite(jac_all) & (jac_all > 0)))
        per_rows, per_cut = admit_by_bar(
            {'target_bid': fr['target_bid'].to_numpy(),
             'jaccard': jac_all,
             'jaccard_rank': fr['jaccard_rank'].to_numpy(dtype=float),
             'rank_union': fr['rank_union'].to_numpy(dtype=float),
             'rank_union_rank': fr['rank_union_rank'].to_numpy(dtype=float)},
            source_bodyId=int(bid),
            source_type=source_types.get(int(bid), ''),
            metric=metric, top_n=top_n, floors=floors,
            window=int(windows.get(int(bid), 1)))
        rows.extend(per_rows)
        cut += per_cut
    return rows, {'sources_scanned': scanned, 'sources_without_profile':
                  no_profile, 'rows_scored': scored,
                  'rows_positive': positives, 'rows_kept': len(rows),
                  'rows_cut_by_cap': cut,
                  'scan_s': round(time.time() - t0, 1)}


def mapper_reference(validator, val_rows: Iterable[Dict]) -> Dict[str, Any]:
    """The three supervised reference sets, read AFTER the gate (§4.5).

    ``C_type``  the target types the mapping asserts for this query;
    ``C_pool``  the bridge-refined target bodyId pools of the selected branches;
    ``C_pair``  the bodyId pairs this run graded ``matched``/``verified*``;
    ``C_src``   the SOURCE bodyIds a branch claims — the label behind
    ``source_claimed``, which is how a reader finds the neurons the mapper left
    out (the residue whose findings are this mode's most interesting harvest).

    Every one of these is read AFTER the gate.  None may decide admission or a
    tier: the unsupervised invariant is enforced by a unit test that makes the
    claim lookups raise.
    """
    c_type, c_pool, c_src = set(), set(), set()
    for pair in validator.pairs:
        if pair.target_type:
            c_type.add(str(pair.target_type))
        c_pool.update(int(b) for b in (pair.target_pool or []))
        c_src.update(int(b) for b in (pair.source_pool or []))
    c_pair: Dict[int, str] = {}
    for r in val_rows or []:
        verdict = str(r.get('verdict') or '')
        if verdict == 'matched' or verdict.startswith('verified'):
            try:
                c_pair[int(r.get('target_bodyId'))] = verdict
            except (TypeError, ValueError):
                continue
    return {'types': c_type, 'pools': c_pool, 'pairs': c_pair,
            'source_pools': c_src}


def _leaf_token(validator, target_type: str, in_map: bool) -> str:
    """The shared per-bodyId leaf token, reused verbatim so pooling cannot
    invent a second vocabulary for the same distinction."""
    from comparison.mapping_validation import (candidate_annotation,
                                               has_type_name)

    decision = validator._backward_decision(str(target_type or ''))
    mapped = decision.get('mapped')
    return candidate_annotation(
        target_type, [mapped] if mapped else [],
        has_type=has_type_name(target_type),
        home_real=bool(decision.get('home_real')),
        type_in_map=in_map)


def annotate(validator, rows: List[Dict], target_id2type: Dict[int, str],
             refs: Dict[str, Any]) -> List[Dict]:
    """Join the labels: type, leaf token, size, the tier, the tags, the
    post-hoc cell, and the mapper's own verdict for that target.

    The tier is computed here from the row's own bar rank, and nothing on this
    path reads a mapper value to decide it — `mapper_cell`, `map_tag` and
    `source_claimed` are labels joined after the grade.
    """
    from comparison.mapping_validation import has_type_name

    cfg = validator.cfg
    sizes = getattr(validator, '_target_sizes', None) or {}
    positive = np.sort(np.asarray([v for v in sizes.values()
                                   if v and v > 0], dtype=float))
    for r in rows:
        bid = int(r['target_bodyId'])
        ttype = str(target_id2type.get(bid, '') or '')
        has_type = has_type_name(ttype)
        r['target_type'] = ttype if has_type else ''
        r['in_scope'] = has_type
        r['leaf'] = _leaf_token(validator, ttype, has_type
                                and ttype in refs['types'])
        size = float(sizes.get(bid) or 0.0)
        r['size_nm3'] = size or None
        r['size_universe_percentile'] = (
            round(100.0 * float(np.searchsorted(positive, size,
                                                side='right')) / len(positive), 1)
            if size and positive.size else None)
        in_pool = bid in refs['pools']
        if in_pool:
            r['mapper_cell'] = CELL_CONFIRMED
        else:
            r['mapper_cell'] = (CELL_TYPE_MISS if has_type
                                and ttype in refs['types'] else CELL_TYPE_NEW)
        r['mapper_verdict'] = refs['pairs'].get(bid, '')
        # ---- the tier, graded on the row's own bar rank (never on a claim) ----
        r['tier'] = tier_of(bar_rank=r.get('bar_rank'),
                            rank_union=r.get('rank_union'),
                            top_n=int(cfg.pooling_bar_top_n),
                            matched_ru_min=cfg.matched_ru_min)
        # Which metrics put this row inside the bar.  A row only rank_union
        # reached is a weaker finding than one both metrics agree on, and that
        # is a LABEL: the retired `rank_union > 0` floor used to make this
        # disclosure by deleting the row.
        supported = [m for m, rk in (('jaccard', r.get('jaccard_rank')),
                                     ('rank_union', r.get('rank_union_rank')))
                     if rk is not None and int(rk) <= int(cfg.pooling_bar_top_n)]
        r['supported_by'] = '+'.join(supported)
        r['single_metric_support'] = len(supported) == 1
        # The map-structure grouping the report and the scenes roll up on.  The
        # LEAF keeps the ladder's shared `(out-map)` token (candidate_annotation
        # already prints it), so `family` names the group, not the leaf: two
        # words for one fact on one leaf is the drift the `matched`/`best`
        # rename exists to prevent.
        r['map_tag'] = ('untyped' if not has_type else
                        'in-map' if in_pool else
                        'family' if str(r.get('leaf', '')).endswith('(out-map)')
                        else 'foreign')
        r['source_claimed'] = int(r['source_bodyId']) in refs.get(
            'source_pools', set())
    return rows


def check_morphology_mandatory(cfg) -> None:
    """Refuse a pooling run that cannot qualify its own tiers.

    Every tier here is defined morph-qualified, so a run without the gate would
    publish connectivity-only rows under claim-shaped names — and the mode exists
    precisely because its findings are graded independently of the type mapper.
    The refusal mirrors `normalize_mode`, which already refuses `--mode pooling`
    alongside a widening flag.
    """
    if str(getattr(cfg, 'effective_mode', cfg.validation_mode)) != POOLING_MODE:
        return
    if not getattr(cfg, 'morph_enabled', True):
        raise ValueError(
            '--mode pooling requires morphology: its tiers are defined as '
            'morph-qualified, so without the gate they would be connectivity '
            'findings under claim-shaped names (the mode\'s own distinction). '
            'Drop --no-morphology, or run --mode restrictive for a structural '
            'pass.')


def _scoring_units(rows: List[Dict]) -> List[Dict]:
    """The pairs the hybrid scores: every tier-1 row, plus the chain-best row of
    every target that no tier-1 row reaches.

    A per-row tier needs a per-row verdict, so the rows that make a CLAIM (a
    source's own rank-1 under either metric) are always scored as themselves.
    The `nominated` rows of one target share the verdict made for that target's
    chain-best pair and say so (`morph_gate='shared'`, `verdict_for_pair`),
    because scoring all of them is 3.8x the work for a review bin.  Measured on
    the landed query: 408 + 138 = 546 units against today\'s 123.
    """
    chain = sorted(rows, key=lambda x: (-x['jaccard'], -x['rank_union'],
                                        int(x['target_bodyId']),
                                        int(x['source_bodyId'])))
    chosen: Dict[tuple, Dict] = {}
    owned: set = set()
    # 1. every tier-1 row is a claim about its own pair, so it is measured as
    #    its own pair
    for r in chain:
        if int(r.get('bar_rank') or 99) <= 1:
            chosen[(int(r['source_bodyId']), int(r['target_bodyId']))] = r
    # 2. plus the chain-best row of each target that no tier-1 row reached:
    #    that row owns the verdict every other row of the target borrows, and
    #    `chain` is already in chain order, so its first row for the target is
    #    the owner
    for r in chain:
        tgt = int(r['target_bodyId'])
        if tgt in owned:
            continue
        owned.add(tgt)
        chosen.setdefault((int(r['source_bodyId']), tgt), r)
    # claims first: a budget cut must never fall on a tier-1 row while a
    # `nominated` row still gets its look
    out = sorted(chosen.values(),
                 key=lambda r: (int(r.get('bar_rank') or 99), -r['jaccard'],
                                -r['rank_union'], int(r['target_bodyId']),
                                int(r['source_bodyId'])))
    return out


def morph_budget(cfg, seed_size: int) -> Tuple[int, str]:
    """The pass's unit budget, and WHERE it came from.

    `pooling_max_morph_targets` 0 means auto: :data:`MORPH_UNITS_PER_SOURCE` x
    the queried population, so a 58-source query is not budgeted like a
    242-source one.  A positive number is the user's own cap.  Either way the
    provenance is published, because "400 of 546 looked at" means nothing
    without the "why".
    """
    cap = int(getattr(cfg, 'pooling_max_morph_targets', 0) or 0)
    if cap > 0:
        return cap, f'configured pooling_max_morph_targets={cap}'
    return (MORPH_UNITS_PER_SOURCE * int(seed_size),
            f'auto: {MORPH_UNITS_PER_SOURCE} x {int(seed_size)} queried sources')


def apply_morph_gate(validator, rows: List[Dict],
                     seed_size: int = 0) -> Tuple[List[Dict], Dict]:
    """Morphology as the LAST gate, on the connectivity survivors only, through
    the Find-Homolog fast path (no NBLAST) with the branch-free, persisted
    ``mapping_ref`` bar (plan §3, §4.2).

    Advisory-safe by design: any failure is recorded per row and on the notes,
    never silently — a blank verdict would read like a rejection.

    The returned ``info`` is the run's morph RECORD and is published in
    `pooling_cross_validation.json`, so its keys have to keep the quantities
    apart: `attempted` is what the budget allowed to be looked at, `scored` is
    what the scorer returned a verdict for (`morph_gate='scored'`),
    `no_score` what it returned nothing for, `capped` the pairs never looked
    at, `qualified` the survivors of the bar, `error` a failed pass.  Keeping
    `attempted` apart from `scored` earned its keep twice on 2026-09-23: the
    field first collapsed them (a record of `attempted 8 / scored 8` whose rows
    held ONE score), and the separated ratio then exposed the cause — the shared
    scorer's mapping_ref rule "a branch-pool member was only fetched to anchor a
    bar, so it is not a candidate" was deleting the verdict of every candidate
    the mapper also claims, 35 of 41 rows on male-cns.  That is what
    `prune_pool_refs=False` below fixes; a low `scored` now means the scorer
    really had nothing for the pair, which is a missing measurement, never a
    rejection.
    """
    cfg = validator.cfg
    check_morphology_mandatory(cfg)
    info = {'attempted': 0, 'scored': 0, 'qualified': 0, 'capped': 0,
            'no_score': 0, 'shared': 0, 'error': ''}
    cap, info['budget'] = morph_budget(cfg, seed_size or len(rows))
    if not rows:
        return rows, info
    # one verdict per SCORING UNIT (see `_scoring_units`), and every other row
    # says which unit's verdict it is reading
    units = _scoring_units(rows)
    take, dropped = (units, []) if len(units) <= cap \
        else (units[:cap], units[cap:])
    # `units` is published because `attempted` and `capped` are a SPLIT of it,
    # not a sum that belongs to `attempted`: without it a reader cannot tell
    # how much of the bar the budget never looked at.
    info['units'] = len(units)
    info['attempted'] = len(take)
    info['capped'] = len(dropped)
    capped_pairs = {(int(r['source_bodyId']), int(r['target_bodyId']))
                    for r in dropped}
    # the target -> chain-best unit that a shared verdict may come from
    owner: Dict[int, tuple] = {}
    for r in take:
        owner.setdefault(int(r['target_bodyId']),
                         (int(r['source_bodyId']), int(r['target_bodyId'])))
    for r in rows:
        key = (int(r['source_bodyId']), int(r['target_bodyId']))
        r['morph_similarity'] = None
        r['morph_pool_ref'] = None
        r['morph_bar'] = None
        r['morph_bar_kind'] = ''
        r['morph_qualified'] = None
        r['verdict_for_pair'] = ''
        # every row is relabelled below; this is the value a row keeps only if
        # the pass never reached it, which is a bug, so make it loud
        r['morph_gate'] = 'unreached'
    pairs = [(int(r['source_bodyId']), int(r['target_bodyId'])) for r in take]
    try:
        from comparison.morph_cross_dataset import qualify_visualized_pairs

        mq = qualify_visualized_pairs(
            cfg.source_dataset, cfg.target_dataset, pairs,
            mode='mapping_ref',
            source_types={int(r['source_bodyId']): str(r['source_type'])
                          for r in take},
            # NOT the supervised convention. A pool member scored only to
            # anchor a bar is dropped there because its candidate set IS the
            # branch pool; here the candidate set is this run's own gate, so
            # dropping them would delete the verdict of every candidate the
            # mapper also claims — measured on 2026-09-23 as 35 of 41 rows
            # mislabelled `no-score` on male-cns (and 33 of 41 on BANC).
            # Grading stays honest either way: `native_scores` measures a
            # candidate against the OTHER refs (`r != tgt`).
            prune_pool_refs=False,
            log=validator.log)
    except Exception as exc:  # noqa: BLE001 — the gate is advisory, the run is not
        info['error'] = f'{type(exc).__name__}: {exc}'
        for r in rows:
            r['morph_gate'] = 'error'
        validator.log(f'[pooling/morph] qualification unavailable, rows stay '
                      f'unqualified: {info["error"]}')
        return rows, info
    verdicts: Dict[tuple, Dict] = {}
    for r in take:
        key = (int(r['source_bodyId']), int(r['target_bodyId']))
        rb = (mq.ref_bars or {}).get(key[0]) or {}
        score = mq.scores.get(key)
        # Publish the record the verdict was actually made from, exactly as the
        # supervised enrichment does: `morph_bar` is the BINDING bar of the kind
        # named in `morph_bar_kind`, and the native track's number rides along
        # in `morph_pool_ref`.  Reading the per-source NULL bar beside a
        # `native` verdict made a passing row look like a broken gate —
        # measured on 2026-09-24 as 14 of 41 scored rows whose published score
        # sat below their published bar while `morph_qualified` said yes.
        kind = rb.get('kind') or 'null_bar'
        verdicts[key] = {
            'morph_gate': ('no-score' if score is None else
                           ('scored' if mq.active else 'inactive')),
            'morph_similarity': score,
            'morph_pool_ref': (mq.native_scores or {}).get(key),
            'morph_bar_kind': kind,
            'morph_bar': (rb['native_floor'] if kind == 'native'
                          else rb.get('backup_floor') if kind == 'track_a'
                          else mq.bar(key[0])),
            'morph_qualified': mq.is_qualified(*key),
        }
        r.update(verdicts[key])
    for r in rows:
        key = (int(r['source_bodyId']), int(r['target_bodyId']))
        if key in verdicts:
            # this row's own measurement — including a second row that repeats
            # the same pair, which is the same measurement, not a borrowed one
            r.update(verdicts[key])
            continue
        src = owner.get(int(r['target_bodyId']))
        if src and src in verdicts:
            r.update(verdicts[src])
            # a borrowed verdict is not this row's measurement: name the pair it
            # came from, and mark the leaf so the scene cannot imply otherwise
            if r['morph_gate'] in ('scored', 'no-score'):
                r['morph_gate'] = 'shared'
            r['verdict_for_pair'] = f'{src[0]}->{src[1]}'
        elif key in capped_pairs:
            r['morph_gate'] = 'not-attempted-cap'
        else:
            r['morph_gate'] = 'no-verdict-for-target'
    # the record is about PAIRS the scorer measured, not rows: two rows sharing
    # one pair is one measurement reported twice
    info['scored'] = sum(1 for v in verdicts.values()
                         if v['morph_gate'] == 'scored')
    info['no_score'] = sum(1 for v in verdicts.values()
                           if v['morph_gate'] == 'no-score')
    info['shared'] = sum(1 for r in rows if r['morph_gate'] == 'shared')
    # unit-level, like attempted/scored: two rows reading one borrowed verdict
    # are ONE measurement that qualified, not two
    info['qualified'] = sum(1 for v in verdicts.values()
                            if v['morph_qualified'])
    info['warnings'] = list(mq.warnings or [])
    # What the persisted target-vector store did for this pass, published next
    # to the counts it changed: `loaded` is preparation this run did NOT pay
    # for, `stale_dropped` is geometry the store refused to reuse because the
    # skeleton behind it moved, `saved` is what the next run starts from.
    info['vector_cache'] = dict(getattr(mq, 'vector_cache', {}) or {})
    for w in info['warnings']:
        validator.log(f'[pooling/morph] {w}')
    return rows, info


def row_refused(r: Dict) -> bool:
    """True only when a verdict was MADE and it said no.

    A missing measurement is not a rejection — that is the rule the whole
    `morph_gate` vocabulary exists for (`no-score`, `not-attempted-cap`,
    `no-verdict-for-target`, `inactive`, `error` all stay).
    """
    return (str(r.get('morph_gate') or '') in ('scored', 'shared')
            and r.get('morph_qualified') is False)


def pool_by_target(rows: List[Dict]) -> List[Dict]:
    """One row per candidate target bodyId, on the ordering chain: Jaccard desc
    -> rank_union desc -> bodyId.

    The target axis is what the scenes and the `(dup)` tag group on, and it is
    also where a refusal is decided: a target stays in the pool while ANY of its
    admitted rows survives the bar, because another source may hold its own (and
    better) verdict for it.

    The row that REPRESENTS the target is therefore the chain-best row whose
    verdict SURVIVES, not the chain-best row full stop.  Both readings are
    defensible for "which source reached this target best", but only the first
    keeps the export recomputable: a row with `in_pool=True` beside
    `morph_qualified=False` tells a reader the gate let a refusal through, when
    what happened is that a different source's row carried it.  Measured on the
    2026-09-25 FAFB->BANC run, exactly 1 target of 208 had its chain-best row
    refused while another qualified (its `verified` row scored 0.803 over a
    native bar of 0.753 while the chain-best `nominated` row scored 0.729 under
    0.919), and the run auditor named it as a ledger contradiction.
    `n_rows_refused` still publishes how many of the target's rows the bar
    refused, and `tiers` still spans every admitting row, so nothing about the
    ladder is hidden by choosing a surviving representative.
    """
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(int(r['target_bodyId']), []).append(r)
    out: List[Dict] = []
    for bid, group in by.items():
        group.sort(key=lambda x: (-x['jaccard'], -x['rank_union'], bid))
        refused = [r for r in group if row_refused(r)]
        b = next((r for r in group if not row_refused(r)), group[0])
        out.append({
            'target_bodyId': bid, 'target_type': b['target_type'],
            'leaf': b['leaf'], 'best_source_bodyId': b['source_bodyId'],
            'best_source_type': b['source_type'],
            'jaccard': b['jaccard'], 'jaccard_rank': b['jaccard_rank'],
            'rank_union': b['rank_union'],
            'rank_union_rank': b['rank_union_rank'],
            'best_bar_rank': min(int(r.get('bar_rank') or 99) for r in group),
            'bar_metric': b.get('bar_metric'), 'bar_top_n': b.get('bar_top_n'),
            'window_size': b['window_size'],
            'size_nm3': b['size_nm3'],
            'size_universe_percentile': b['size_universe_percentile'],
            'in_scope': b['in_scope'], 'map_tag': b.get('map_tag'),
            'tiers': '+'.join(sorted({str(r.get('tier')) for r in group
                                      if r.get('tier')})),
            'n_sources': len(group), 'dup': len(group) - 1,
            'n_rows_refused': len(refused),
            'in_pool': len(refused) < len(group),
            # the morph fields exist only once that gate has run; a pool row
            # must not depend on the gate having been reached
            'morph_gate': b.get('morph_gate'),
            'morph_bar_kind': b.get('morph_bar_kind'),
            'morph_similarity': b.get('morph_similarity'),
            'morph_pool_ref': b.get('morph_pool_ref'),
            'morph_bar': b.get('morph_bar'),
            'morph_qualified': b.get('morph_qualified'),
            'verdict_for_pair': b.get('verdict_for_pair'),
            'mapper_cell': b['mapper_cell'],
            'mapper_verdict': b['mapper_verdict'],
        })
    out.sort(key=lambda x: (-(x['jaccard'] or 0.0), -(x['rank_union'] or 0.0),
                            x['target_bodyId']))
    return out


def mapper_only_rows(refs: Dict[str, Any], pool: List[Dict]) -> List[Dict]:
    """The targets the SUPERVISED path graded `matched`/`verified*` that this
    bar never admitted — as rows, not as a bare JSON count.

    `verified_only` used to be a number printed beside a table of row labels,
    which made two neighbouring facts read as the same kind of thing.  Here it is
    the same shape as every other target row, with `in_pool=False` and an empty
    bar rank, so one file answers "which target neurons does EITHER engine
    claim?".
    """
    admitted = {int(r['target_bodyId']) for r in pool}
    out: List[Dict] = []
    for bid, verdict in sorted((refs.get('pairs') or {}).items()):
        if int(bid) in admitted:
            continue
        out.append({'target_bodyId': int(bid), 'target_type': '',
                    'leaf': '', 'best_source_bodyId': '',
                    'best_source_type': '', 'jaccard': None,
                    'jaccard_rank': None, 'rank_union': None,
                    'rank_union_rank': None, 'best_bar_rank': None,
                    'bar_metric': '', 'bar_top_n': None, 'window_size': None,
                    'size_nm3': None, 'size_universe_percentile': None,
                    'in_scope': None, 'map_tag': 'in-map', 'tiers': '',
                    'n_sources': 0, 'dup': None, 'n_rows_refused': 0,
                    'in_pool': False, 'morph_gate': '', 'morph_bar_kind': '',
                    'morph_similarity': None, 'morph_pool_ref': None,
                    'morph_bar': None, 'morph_qualified': None,
                    'verdict_for_pair': '',
                    'mapper_cell': CELL_VERIFIED_ONLY,
                    'mapper_verdict': verdict})
    return out


def pool_by_source(rows: List[Dict], seed: Sequence[int],
                   refused_targets: set = frozenset()) -> List[Dict]:
    """One row per QUERIED SOURCE — the mode's own unit, and the table the
    headline reads.

    Every queried source appears, including the ones that found nothing and the
    ones whose findings morphology refused: a missing row would read as "not
    examined", which is a different claim than "examined, found nothing".
    """
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(int(r['source_bodyId']), []).append(r)
    out: List[Dict] = []
    for bid in sorted(int(s) for s in seed):
        group = sorted(by.get(bid, []),
                       key=lambda x: (int(x.get('bar_rank') or 99),
                                      -x['jaccard'], -x['rank_union'],
                                      int(x['target_bodyId'])))
        kept = [r for r in group
                if not row_refused(r)
                and int(r['target_bodyId']) not in refused_targets]
        head = (kept or group or [None])[0]
        src = group[0] if group else None
        out.append({
            'source_bodyId': bid,
            'source_type': (src or {}).get('source_type', '') if src else '',
            'n_admitted': len(group), 'n_in_pool': len(kept),
            'n_refused': len(group) - len(kept),
            'tier': (head or {}).get('tier', '') if head else '',
            'best_target_bodyId': (head or {}).get('target_bodyId')
            if head else None,
            'best_bar_rank': (head or {}).get('bar_rank') if head else None,
            'jaccard': (head or {}).get('jaccard') if head else None,
            'jaccard_rank': (head or {}).get('jaccard_rank') if head else None,
            'rank_union': (head or {}).get('rank_union') if head else None,
            'rank_union_rank': (head or {}).get('rank_union_rank')
            if head else None,
            'supported_by': (src or {}).get('supported_by', '') if src else '',
            'single_metric_support': (src or {}).get('single_metric_support')
            if src else None,
            'source_claimed': bool((src or {}).get('source_claimed'))
            if src else None,
            'morph_gate': (head or {}).get('morph_gate', '') if head else '',
            'morph_bar_kind': (head or {}).get('morph_bar_kind', '')
            if head else '',
            'morph_similarity': (head or {}).get('morph_similarity')
            if head else None,
            'morph_pool_ref': (head or {}).get('morph_pool_ref')
            if head else None,
            'morph_bar': (head or {}).get('morph_bar') if head else None,
            'morph_qualified': (head or {}).get('morph_qualified')
            if head else None,
            'verdict_for_pair': (head or {}).get('verdict_for_pair', '')
            if head else '',
            'no_finding': '' if group else 'no-admitted-target',
        })
    return out


def cross_validation(pool: List[Dict], rows: List[Dict], refs: Dict[str, Any],
                     stats: Dict, morph: Dict,
                     notes: List[str],
                     sources: List[Dict] | None = None) -> Dict[str, Any]:
    """The §4.5 comparison: cells, the containment numbers, and the honesty
    rules stated as data rather than as prose."""
    admitted = [p for p in pool if p.get('mapper_cell') != CELL_VERIFIED_ONLY]
    in_pool_ids = {int(p['target_bodyId']) for p in admitted
                   if p.get('in_pool')}
    in_pool = {int(p['target_bodyId']) for p in admitted
               if p.get('in_pool') and p['mapper_cell'] == CELL_CONFIRMED}
    miss = [p for p in admitted if p.get('in_pool')
            and p['mapper_cell'] in (CELL_TYPE_MISS, CELL_TYPE_NEW)]
    # derived from the supervised set itself, not from the rows the caller
    # happened to append: `mapper_only_rows` publishes exactly this set into
    # pooling_pool.csv, and the JSON must not depend on that step having run
    verified_only = sorted({int(b) for b in (refs.get('pairs') or {})}
                           - {int(p['target_bodyId']) for p in admitted})
    n_src = int(stats.get('seed', 0) or 0)
    per_source = (round(len(in_pool_ids) / n_src, 3) if n_src else None)

    def _by_type(cells):
        out: Dict[str, int] = {}
        for p in cells:
            out[p['target_type'] or 'untyped'] = \
                out.get(p['target_type'] or 'untyped', 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    def _flag(name):
        return sum(1 for r in rows if r.get(name))

    return {
        'universe_scanned': stats.get('universe'),
        # the comparison is only comparable between runs that scored against
        # the same store and mapper snapshot, so the note below names a record
        # this file actually carries (plan §4.3)
        'input_fingerprint': stats.get('fingerprint') or {},
        'seed': {'queried_sources': n_src,
                 'scanned': stats.get('sources_scanned', 0),
                 'without_profile': stats.get('sources_without_profile', 0),
                 # two different questions, both worth reading: how many
                 # sources found at least one homolog, and how many are the
                 # chain-best source of a pooled target
                 'sources_with_a_candidate':
                     len({int(r['source_bodyId']) for r in rows}),
                 'sources_with_a_claim':
                     sum(1 for s in (sources or []) if s.get('n_in_pool')),
                 'distinct_best_sources':
                     len({p['best_source_bodyId'] for p in admitted})},
        'bar': stats.get('bar', {}),
        'gate': stats.get('gate', {}),
        # the floors' disclosure: evaluated, published, and applied to nothing
        'floor_flags': {'below_jaccard_floor': _flag('below_jaccard_floor'),
                        'below_rank_union_floor':
                            _flag('below_rank_union_floor'),
                        'outside_window': _flag('outside_window'),
                        'rows': len(rows)},
        'tiers': {tier: sum(1 for r in rows if r.get('tier') == tier)
                  for tier in POOLING_TIERS},
        'pool_per_source': per_source,
        # the bar's own claim is "a pool close to the source count"; when it is
        # not, say so rather than leaving a reader to divide two numbers
        'pool_size_warning': (
            '' if per_source is None else
            '' if 0.5 <= per_source <= 2.0 else
            f'pool_per_source {per_source} has left [0.5, 2.0]: this bar is no '
            f'longer admitting about one target per queried source. Raise '
            f'pooling_bar_top_n or widen pooling_bar_metric.'),
        'cells': {
            CELL_CONFIRMED: len(in_pool),
            CELL_POOL_MISS: len(miss),
            CELL_TYPE_MISS: sum(1 for p in miss
                                if p['mapper_cell'] == CELL_TYPE_MISS),
            CELL_TYPE_NEW: sum(1 for p in miss
                               if p['mapper_cell'] == CELL_TYPE_NEW),
            CELL_VERIFIED_ONLY: len(verified_only),
        },
        'body_ids': {
            CELL_POOL_MISS: sorted(int(p['target_bodyId']) for p in miss),
            CELL_VERIFIED_ONLY: verified_only,
        },
        'pool_miss_by_type': _by_type(miss),
        'map_tags': _by_type([p for p in admitted if p.get('in_pool')]),
        'morph': morph,
        # §4.5 honesty rules, published where a reader cannot miss them:
        'reading_notes': [
            'No cell is a recall measure: the unsupervised seed is the queried '
            'population, the supervised C_pair exists only for the sources a '
            'branch claimed.',
            'verified_only is expected to be large: the supervised tier admits '
            'by pool membership, this gate admits by global rank. It is not a '
            'false-positive count. Its targets are exported as rows in '
            'pooling_pool.csv (in_pool=False), not only counted here.',
            'This comparison is only comparable across runs when the '
            'input_fingerprint (mapper snapshot + profile caches) matches.',
            'The floors are flags: floor_flags counts how many admitted rows '
            'sit below each configured number, and no row was removed for it.',
            'morph.dropped_targets are pooled targets EVERY admitting row of '
            'which the morphology bar refused; a target another source still '
            'holds qualified stays, and pooling_pool.csv names both counts.',
            'morph_gate=shared means the verdict was made for the pair named in '
            'verdict_for_pair, not for this row — a borrowed number is never '
            'this row\'s measurement.',
        ] + list(notes or []),
    }


def run_pooling(validator, *, target_stats, target_bids, target_id2type,
                val_rows) -> Dict[str, Any]:
    """The whole pass.  Returns ``{'candidates', 'pool', 'sources',
    'cross_validation'}``.

    Reads no pool, no branch bar and no mapper claim on the selection path; the
    seed comes from the source dataset's annotation and the bar is absolute.
    """
    cfg = validator.cfg
    check_morphology_mandatory(cfg)
    notes: List[str] = []
    # The floors are FLAGS now (plan-tmvev-pooling-tiers.md §2): each is still
    # evaluated against the configured number and published per row, and none of
    # them removes a candidate.  What bounds the scan is the bar and its cap.
    j_floor = float(cfg.pooling_jaccard_floor)
    ru_floor = float(cfg.pooling_rank_union_floor)
    gate = {'jaccard_floor': j_floor,
            'rank_union_floor': ru_floor,
            'window_mult': cfg.pooling_window_mult,
            'role': 'advisory flags — each is evaluated and published per row, '
                    'and none of them removes a candidate. The bar '
                    '(pooling_bar_metric x pooling_bar_top_n) is what decides '
                    'admission.'}
    bar = {'metric': str(cfg.pooling_bar_metric),
           'top_n': int(cfg.pooling_bar_top_n),
           'row_cap_multiple': ROW_CAP_MULTIPLE,
           'role': 'the admission rule: each source keeps the top-N of each '
                   'chosen metric, capped at this many x N rows per source'}
    seed = seed_population(validator, cfg.query_types)
    if not seed:
        notes.append('pooling: the queried population resolved to 0 neurons — '
                     'nothing scanned.')
        return {'candidates': [], 'pool': [], 'sources': [],
                'cross_validation': cross_validation(
                    [], [], {'types': set(), 'pools': set(), 'pairs': {},
                             'source_pools': set()},
                    {'gate': gate, 'bar': bar}, {}, notes)}
    # The source neurons' own type names: what the window flag scales to.  This
    # is the dataset's annotation, reached through the profile backend — the
    # same labels stage 2 reads for the target side.
    try:
        source_types = {int(k): str(v or '') for k, v in
                        (validator.profiler.get_types_for_bodyids(
                            list(seed), cfg.source_dataset) or {}).items()}
    except Exception as exc:  # noqa: BLE001
        notes.append(f'pooling: source type labels unavailable ({exc}); '
                     'the window falls back to 1 per source.')
        source_types = {}
    source_types = {int(b): source_types.get(int(b), '') for b in seed}
    windows = window_sizes(source_types, cfg.pooling_window_mult)
    rows, stats = connectivity_candidates(validator, seed, source_types,
                                         windows, target_stats, target_bids,
                                         j_floor)
    refs = mapper_reference(validator, val_rows)
    rows = annotate(validator, rows, target_id2type, refs)
    rows, morph = apply_morph_gate(validator, rows, seed_size=len(seed))
    # Morphology is the LAST GATE and the tier is per row: a target leaves the
    # pool only when EVERY row admitting it was refused (`pool_by_target`'s
    # in_pool), which is what lets one target be a claim for one source and
    # absent for another.
    pool_all = pool_by_target(rows)
    refused_targets = {int(p['target_bodyId']) for p in pool_all
                       if not p.get('in_pool')}
    sources = pool_by_source(rows, seed, refused_targets)
    pool = pool_all + mapper_only_rows(refs, pool_all)
    morph['gate_applied'] = True
    morph['mandatory'] = True
    morph['dropped_targets'] = len(refused_targets)
    # `morph['units']` (the record) is the number of scoring units the pass was
    # offered; `attempted`/`capped` are its split.
    stats.update({'seed': len(seed), 'universe': len(target_bids),
                  'fingerprint': dict(getattr(validator, 'input_fingerprint',
                                              None) or {}),
                  'gate': gate, 'bar': {**bar,
                                         'rows_cut':
                                             stats.get('rows_cut_by_cap', 0)}})
    xval = cross_validation(pool, rows, refs, stats, morph, notes,
                            sources=sources)
    validator.log(f"[pooling] {len(seed)} queried sources scanned "
                  f"{stats['rows_scored']:,} pairs -> {stats['rows_kept']} "
                  f"rows admitted by the bar "
                  f"({bar['metric']} x top-{bar['top_n']}, "
                  f"{stats.get('rows_cut_by_cap', 0)} cut by the "
                  f"{ROW_CAP_MULTIPLE}N row cap, morph budget "
                  f"{morph.get('budget')}); {len(pool_all)} distinct "
                  f"targets, {len(refused_targets)} refused by the morphology "
                  f"bar on every row, {xval['cells'][CELL_VERIFIED_ONLY]} "
                  f"mapper-only rows published alongside; "
                  f"pool/source {xval['pool_per_source']}, "
                  f"tiers {xval['tiers']}; morph {morph}")
    validator.log(f"[pooling] floors are flags now: jaccard > {j_floor:.4f}, "
                  f"rank_union > {ru_floor:.3f}, both ranks within "
                  f"{cfg.pooling_window_mult:.2f} x the source type's own "
                  f"population — {xval['floor_flags']} rows sit below them and "
                  f"none was removed for it")
    if xval['pool_size_warning']:
        validator.log(f"[pooling] ! {xval['pool_size_warning']}")
    return {'candidates': rows, 'pool': pool, 'sources': sources,
            'cross_validation': xval, 'stats': stats}
