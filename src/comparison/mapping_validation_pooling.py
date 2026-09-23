"""TM VEV `pooling` mode — the unsupervised candidate engine.

Normative design: ``_plan/plan-tmvev-pooling-mode.md``.  Where this pipeline's
other modes ask "what else belongs to the pool the type mapper already
asserted" (supervised), pooling asks "what do connectivity and morphology say is
a homolog of the queried population, over the whole opposite universe" and only
THEN compares that answer with the mapper (§4.5).

The unsupervised property is load-bearing, so it is stated as an invariant:
**no function here may use a mapper claim to decide whether a row is a
candidate.**  The mapper is read after the gate, to label cells and leaf tokens.
A unit test enforces it by making the claim lookups raise.

The floors are a **volume** control, not a data-quality claim (user
2026-09-23): ``pooling_jaccard_floor`` / ``pooling_rank_union_floor`` and the
window keep a scan from turning into thousands of candidates, and each is
published as the configured number it is.  An earlier build fitted the Jaccard
floor to the dataset pair's own graded evidence; that measured a quality
question the floor does not answer, so it was deleted rather than tuned — see
``_plan/plan-tmvev-pooling-mode.md`` §5.
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
    """Scan the seed against the WHOLE target universe and keep the rows that
    pass the absolute gate (plan §4.2): ``jaccard > J``, ``rank_union > RU``,
    and both metric ranks inside the source's window.

    ``jaccard_floor`` is the configured floor passed in, and it is the number
    the run gates on and publishes: a VOLUME guard-rail (plan §5), never a
    statement about whether a pair is a good homolog.

    Returns the surviving rows plus the scan stats (rows scored, positives).
    """
    from comparison.body_id_resolver import scan_source
    from comparison.mapping_validation import expanded_vector

    cfg = validator.cfg
    rows: List[Dict] = []
    t0 = time.time()
    scored = positives = scanned = 0
    for bid in seed:
        prof = validator.profiler.get_profile(bid, cfg.source_dataset)
        vec = expanded_vector(prof, validator.mapper) if prof else None
        if not vec:
            continue
        scanned += 1
        fr = scan_source(vec, target_stats, list(target_bids))
        if fr is None or fr.empty:
            continue
        jac = fr['jaccard'].to_numpy(dtype=float)
        ru = fr['rank_union'].to_numpy(dtype=float)
        jr = fr['jaccard_rank'].to_numpy(dtype=float)
        rr = fr['rank_union_rank'].to_numpy(dtype=float)
        scored += len(fr)
        positives += int(np.count_nonzero(np.isfinite(jac) & (jac > 0)))
        w = int(windows.get(int(bid), 1))
        keep = ((jac > float(jaccard_floor))
                & (ru > float(cfg.pooling_rank_union_floor))
                & (rr <= w) & (jr <= w))
        bids = fr['target_bid'].to_numpy()
        for i in np.flatnonzero(keep):
            rows.append({'source_bodyId': int(bid),
                         'source_type': source_types.get(int(bid), ''),
                         'target_bodyId': int(bids[i]),
                         'jaccard': float(jac[i]), 'jaccard_rank': int(jr[i]),
                         'rank_union': float(ru[i]),
                         'rank_union_rank': int(rr[i]), 'window_size': w})
    return rows, {'sources_scanned': scanned, 'rows_scored': scored,
                  'rows_positive': positives, 'rows_kept': len(rows),
                  'scan_s': round(time.time() - t0, 1)}


def mapper_reference(validator, val_rows: Iterable[Dict]) -> Dict[str, Any]:
    """The three supervised reference sets, read AFTER the gate (§4.5).

    ``C_type``  the target types the mapping asserts for this query;
    ``C_pool``  the bridge-refined target bodyId pools of the selected branches;
    ``C_pair``  the bodyId pairs this run graded ``matched``/``verified*``.
    """
    c_type, c_pool = set(), set()
    for pair in validator.pairs:
        if pair.target_type:
            c_type.add(str(pair.target_type))
        c_pool.update(int(b) for b in (pair.target_pool or []))
    c_pair: Dict[int, str] = {}
    for r in val_rows or []:
        verdict = str(r.get('verdict') or '')
        if verdict == 'matched' or verdict.startswith('verified'):
            try:
                c_pair[int(r.get('target_bodyId'))] = verdict
            except (TypeError, ValueError):
                continue
    return {'types': c_type, 'pools': c_pool, 'pairs': c_pair}


def _leaf_token(validator, target_type: str, in_map: bool) -> str:
    """The shared per-bodyId leaf token, reused verbatim so pooling cannot
    invent a second vocabulary for the same distinction."""
    from comparison.mapping_validation import candidate_annotation

    decision = validator._backward_decision(str(target_type or ''))
    mapped = decision.get('mapped')
    return candidate_annotation(
        target_type, [mapped] if mapped else [],
        has_type=bool(str(target_type or '').strip() not in ('', '?')),
        home_real=bool(decision.get('home_real')),
        type_in_map=in_map)


def annotate(validator, rows: List[Dict], target_id2type: Dict[int, str],
             refs: Dict[str, Any]) -> List[Dict]:
    """Join the labels: type, leaf token, size, the post-hoc cell, and the
    mapper's own verdict for that target."""
    sizes = getattr(validator, '_target_sizes', None) or {}
    positive = np.sort(np.asarray([v for v in sizes.values()
                                   if v and v > 0], dtype=float))
    for r in rows:
        bid = int(r['target_bodyId'])
        ttype = str(target_id2type.get(bid, '') or '')
        has_type = ttype.strip() not in ('', '?')
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
    return rows


def apply_morph_gate(validator, rows: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Morphology as the LAST gate, on the connectivity survivors only, through
    the Find-Homolog fast path (no NBLAST) with the branch-free, persisted
    ``mapping_ref`` bar (plan §3, §4.2).

    Advisory-safe by design: any failure is recorded per row and on the notes,
    never silently — a blank verdict would read like a rejection.

    The returned ``info`` is the run's morph RECORD and is published in
    `pooling_cross_validation.json`, so its keys have to keep the quantities
    apart: `attempted` is what the budget allowed to be looked at, `scored` is
    what the scorer returned a verdict for (`morph_gate='scored'`),
    `no_score` what it returned nothing for, `capped` the targets never looked
    at, `qualified` the survivors of the bar, `error` a failed pass.  Keeping
    `attempted` and `scored` apart earned its keep twice on 2026-09-23: the
    field first collapsed them (a record of `attempted 8 / scored 8` whose rows
    held ONE score), and the separated ratio then exposed the cause — the
    shared scorer's mapping_ref rule "a branch-pool member was only fetched to
    anchor a bar, so it is not a candidate" was deleting the verdict of every
    candidate the mapper also claims, 35 of 41 rows on male-cns.  That is what
    `prune_pool_refs=False` below fixes; a low `scored` now means the scorer
    really had nothing for the pair, which is a missing measurement, never a
    rejection.
    """
    cfg = validator.cfg
    info = {'attempted': 0, 'scored': 0, 'qualified': 0, 'capped': 0,
            'no_score': 0, 'error': ''}
    if not rows:
        return rows, info
    # `--no-morphology` is the run-wide "structural pass only" switch, and
    # pooling's own gate is morphology: honouring it is what lets a fast
    # pooling run exist without silently spending the fetch budget.
    if not cfg.pooling_morph_gate or not cfg.morph_enabled:
        for r in rows:
            r['morph_gate'] = 'disabled'
        return rows, info
    # one row per candidate target: the chain-best source carries the verdict
    best: Dict[int, Dict] = {}
    for r in sorted(rows, key=lambda x: (-x['jaccard'], -x['rank_union'],
                                         x['target_bodyId'])):
        best.setdefault(int(r['target_bodyId']), r)
    ordered = list(best.values())
    cap = int(cfg.pooling_max_morph_targets or 0)
    take, dropped = (ordered, []) if cap <= 0 or len(ordered) <= cap \
        else (ordered[:cap], ordered[cap:])
    info['attempted'] = len(take)
    info['capped'] = len(dropped)
    selected = {(int(r['source_bodyId']), int(r['target_bodyId']))
                for r in take}
    capped = {(int(r['source_bodyId']), int(r['target_bodyId']))
              for r in dropped}
    for r in rows:
        # three different absences, and none of them may read as a rejection:
        # this row is not the chain-best row of its target (the verdict lives
        # on that row), the budget refused to look at it, or the scorer had
        # no vector for the pair.
        key = (int(r['source_bodyId']), int(r['target_bodyId']))
        r['morph_gate'] = ('not-attempted-cap' if key in capped
                           else 'not-selected')
        r['morph_similarity'] = None
        r['morph_bar'] = None
        r['morph_bar_kind'] = ''
        r['morph_qualified'] = None
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
    for r in take:
        key = (int(r['source_bodyId']), int(r['target_bodyId']))
        rb = (mq.ref_bars or {}).get(key[0]) or {}
        score = mq.scores.get(key)
        r['morph_gate'] = ('no-score' if score is None else
                           ('scored' if mq.active else 'inactive'))
        r['morph_similarity'] = score
        r['morph_bar'] = mq.bar(key[0])
        r['morph_bar_kind'] = rb.get('kind') or 'null'
        r['morph_qualified'] = mq.is_qualified(*key)
    # `scored` counts PAIRS THE SCORER RETURNED a verdict for, not the pairs it
    # was asked about: an attempted target can still come back `no-score` (no
    # vector for it in this dataset pair's store), and reporting the attempt
    # count as the score count would hide exactly the degradation the row-level
    # `morph_gate` exists to name. `attempted` and `scored` are therefore both
    # published — and read them as the scorer's coverage of THIS run, never as
    # a rejection rate.
    info['scored'] = sum(1 for r in take if r['morph_gate'] == 'scored')
    info['no_score'] = sum(1 for r in take if r['morph_gate'] == 'no-score')
    info['qualified'] = sum(1 for r in take if r['morph_qualified'])
    info['warnings'] = list(mq.warnings or [])
    for w in info['warnings']:
        validator.log(f'[pooling/morph] {w}')
    return rows, info


def pool_by_target(rows: List[Dict]) -> List[Dict]:
    """One row per candidate target bodyId (the deduplicated pool), on the
    ordering chain: Jaccard desc -> rank_union desc -> bodyId."""
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(int(r['target_bodyId']), []).append(r)
    out: List[Dict] = []
    for bid, group in by.items():
        group.sort(key=lambda x: (-x['jaccard'], -x['rank_union'], bid))
        b = group[0]
        out.append({
            'target_bodyId': bid, 'target_type': b['target_type'],
            'leaf': b['leaf'], 'best_source_bodyId': b['source_bodyId'],
            'best_source_type': b['source_type'],
            'jaccard': b['jaccard'], 'jaccard_rank': b['jaccard_rank'],
            'rank_union': b['rank_union'], 'window_size': b['window_size'],
            'size_nm3': b['size_nm3'],
            'size_universe_percentile': b['size_universe_percentile'],
            'in_scope': b['in_scope'],
            'n_sources': len(group), 'dup': len(group) - 1,
            # the morph fields exist only once that gate has run; a pool row
            # must not depend on the gate having been reached
            'morph_gate': b.get('morph_gate'),
            'morph_similarity': b.get('morph_similarity'),
            'morph_bar': b.get('morph_bar'),
            'morph_qualified': b.get('morph_qualified'),
            'mapper_cell': b['mapper_cell'],
            'mapper_verdict': b['mapper_verdict'],
        })
    out.sort(key=lambda x: (-(x['jaccard'] or 0.0), -(x['rank_union'] or 0.0),
                            x['target_bodyId']))
    return out


def morph_refused(pool: List[Dict]) -> List[Dict]:
    """The pooled targets the LAST GATE said no to — the ones that leave the
    exported pool (decision 3, and the user's reading of it on 2026-09-23:
    morphology is a gate here, not a label).

    Only an explicit refusal removes a target. A row with no verdict
    (`no-score`, `not-attempted-cap`, `disabled`, `inactive`, `error`) stays,
    because a missing measurement is not a rejection — that is the rule the
    whole `morph_gate` vocabulary exists for. The count is published beside the
    pool so a shrunken pool is never read as a smaller harvest.
    """
    return [p for p in pool
            if p.get('morph_gate') == 'scored'
            and p.get('morph_qualified') is False]


def cross_validation(pool: List[Dict], rows: List[Dict], refs: Dict[str, Any],
                     stats: Dict, morph: Dict,
                     notes: List[str]) -> Dict[str, Any]:
    """The §4.5 comparison: cells, the containment numbers, and the honesty
    rules stated as data rather than as prose."""
    in_pool = {int(p['target_bodyId']) for p in pool
               if p['mapper_cell'] == CELL_CONFIRMED}
    miss = [p for p in pool if p['mapper_cell'] in (CELL_TYPE_MISS,
                                                    CELL_TYPE_NEW)]
    verified = refs['pairs']
    verified_only = sorted(set(int(b) for b in verified) - in_pool)
    def _by_type(cells):
        out: Dict[str, int] = {}
        for p in cells:
            out[p['target_type'] or 'untyped'] = \
                out.get(p['target_type'] or 'untyped', 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))
    return {
        'universe_scanned': stats.get('universe'),
        # the comparison is only comparable between runs that scored against
        # the same store and mapper snapshot, so the note below names a record
        # this file actually carries (plan §4.3)
        'input_fingerprint': stats.get('fingerprint') or {},
        'seed': {'queried_sources': stats.get('seed', 0),
                 'scanned': stats.get('sources_scanned', 0),
                 # two different questions, both worth reading: how many
                 # sources found at least one homolog, and how many are the
                 # chain-best source of a pooled target
                 'sources_with_a_candidate':
                     len({int(r['source_bodyId']) for r in rows}),
                 'distinct_best_sources':
                     len({p['best_source_bodyId'] for p in pool})},
        'gate': stats.get('gate', {}),
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
        'morph': morph,
        # §4.5 honesty rules, published where a reader cannot miss them:
        'reading_notes': [
            'No cell is a recall measure: the unsupervised seed is the queried '
            'population, the supervised C_pair exists only for the sources a '
            'branch claimed.',
            'verified_only is expected to be large: the supervised tier admits '
            'by pool membership, this gate admits by global rank. It is not a '
            'false-positive count.',
            'This comparison is only comparable across runs when the '
            'input_fingerprint (mapper snapshot + profile caches) matches.',
            'morph.dropped_targets are pooled targets the morphology bar '
            'refused: they left the pool and the scene root, and their rows '
            'stay in pooling_candidates.csv with the refusal on them.',
        ] + list(notes or []),
    }


def run_pooling(validator, *, target_stats, target_bids, target_id2type,
                val_rows) -> Dict[str, Any]:
    """The whole pass.  Returns ``{'candidates', 'pool', 'cross_validation'}``.

    Reads no pool, no branch bar and no mapper claim on the selection path;
    the seed comes from the source dataset's annotation and the floors are
    absolute.
    """
    cfg = validator.cfg
    notes: List[str] = []
    # The floors are a VOLUME control (plan §5, user 2026-09-23): their job is
    # to keep a scan from opening into thousands of candidates, not to say
    # whether a pair is a good homolog. So the configured number IS the number
    # that gated the run — published as one, with no fitted provenance.
    j_floor = float(cfg.pooling_jaccard_floor)
    gate = {'jaccard_floor': j_floor,
            'rank_union_floor': cfg.pooling_rank_union_floor,
            'window_mult': cfg.pooling_window_mult,
            'role': 'volume guard-rail — how wide connectivity may open, not a '
                    'data-quality claim about any pair'}
    seed = seed_population(validator, cfg.query_types)
    if not seed:
        notes.append('pooling: the queried population resolved to 0 neurons — '
                     'nothing scanned.')
        return {'candidates': [], 'pool': [],
                'cross_validation': cross_validation(
                    [], [], {'types': set(), 'pools': set(), 'pairs': {}},
                    {'gate': gate}, {}, notes)}
    # The source neurons' own type names: what the window scales to.  This is
    # the dataset's annotation, reached through the profile backend — the same
    # labels stage 2 reads for the target side.
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
    rows, morph = apply_morph_gate(validator, rows)
    # Morphology is the LAST GATE: an explicit refusal takes the target out of
    # the pool and the scene; "no verdict" never does (see `morph_refused`).
    pool_all = pool_by_target(rows)
    refused = {int(r['target_bodyId']) for r in morph_refused(pool_all)}
    pool = [r for r in pool_all if int(r['target_bodyId']) not in refused]
    morph['gate_applied'] = bool(cfg.pooling_morph_gate and cfg.morph_enabled)
    morph['dropped_targets'] = len(refused)
    stats.update({'seed': len(seed), 'universe': len(target_bids),
                  'fingerprint': dict(getattr(validator, 'input_fingerprint',
                                              None) or {}),
                  'gate': gate})
    xval = cross_validation(pool, rows, refs, stats, morph, notes)
    validator.log(f"[pooling] {len(seed)} queried sources scanned "
                  f"{stats['rows_scored']:,} pairs -> {stats['rows_kept']} "
                  f"rows; {len(pool_all)} connectivity-admitted targets, "
                  f"{len(refused)} refused by the morphology bar, "
                  f"{morph['no_score']} without a verdict kept and named "
                  f"-> {len(pool)} pooled "
                  f"({xval['cells'][CELL_POOL_MISS]} outside every mapper "
                  f"pool); morph {morph}")
    validator.log(f"[pooling] floors: jaccard > {j_floor:.4f}, rank_union > "
                  f"{cfg.pooling_rank_union_floor:.3f}, both ranks within "
                  f"{cfg.pooling_window_mult:.2f} x the source type's own "
                  f"population — volume guard-rails")
    return {'candidates': rows, 'pool': pool, 'cross_validation': xval,
            'stats': stats}
