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

One qualification is itself a design decision (plan §5): the Jaccard floor may be
*fitted* from the graded pairs of PAST runs of the same dataset pair
(:class:`JaccardFloorStore`).  That is dataset-level calibration, like the null
sample the morphology bars are built from — not this run's claims deciding this
run's rows.  The two are kept apart by ordering: the floor is read at the top of
the pass and the evidence is written at the bottom, so a run's own verdicts can
never move its own gate.  The effective floor and its provenance are published in
every export.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


#: Where a run's effective Jaccard floor came from.  Published, because the
#: number alone does not tell a reader whether it is a constant or a fit.
FLOOR_CONFIG = 'config-default'
FLOOR_FITTED = 'dataset-fitted'
FLOOR_THIN = 'dataset-fitted-thin-sample'


class JaccardFloorStore:
    """The per-dataset-pair evidence behind the pooling Jaccard floor (P4).

    One global floor is only safe in the low band: at J=0.20 the same rule
    rejects 33 % of BANC's verified rows, 3 % of male-cns's and 27 % of
    hemibrain's (plan §2.2), so a single constant quietly changes what each
    dataset's run means.  This store therefore holds, per (source, target)
    pair, the jaccard of every pair the SUPERVISED path graded
    ``matched``/``verified*`` in past runs, and the floor is fitted as
    ``min(configured, q05(that evidence))`` — the value at which this dataset
    pair's own verified rows start to be rejected.

    It lives beside the ``NullVectorStore`` sidecars for the same reason: an
    unpinned sample makes a bar drift (BANC's null-kind bar measured
    0.593 -> 0.235 between identical runs).  The drift that matters here is
    removed by ordering: the floor is READ at the start of a pass and the
    evidence is WRITTEN at the end, so no run can move its own gate with its
    own claims — which is what keeps `pooling` unsupervised.
    """

    MIN_N = 20          # mirrors `null_min_n`: below this, do not fit
    PERCENTILE = 5.0    # q05: the floor is a 5th-percentile statement
    CAP = 20000         # keep the LOW tail exactly; that is what q05 reads

    def __init__(self, source_dataset: str, target_dataset: str,
                 project_root: Optional[str] = None):
        from pathlib import Path
        from morphology import _dataset_folder
        from comparison.morph_cross_dataset import DEFAULT_PROJECT_ROOT
        self.source_dataset = str(source_dataset or '')
        self.target_dataset = str(target_dataset or '')
        root = Path(project_root or DEFAULT_PROJECT_ROOT)
        # the pool is scored against the TARGET universe, so the target owns
        # the file; the source is hashed into the name (one file per pair)
        tag = hashlib.sha1(self.source_dataset.encode()).hexdigest()[:12]
        self.path = (root / 'cache' / _dataset_folder(self.target_dataset)
                     / 'pooling' / f'jaccard_evidence_{tag}.json')

    def load(self) -> Dict[str, float]:
        try:
            with open(self.path, errors='ignore') as fh:
                raw = json.load(fh) or {}
            pairs = raw.get('pairs') or {}
            return {str(k): float(v) for k, v in pairs.items()}
        except (OSError, ValueError, TypeError):
            return {}

    def fit(self, configured: float) -> Tuple[float, Dict[str, Any]]:
        """The effective floor plus its provenance; never raises."""
        meta: Dict[str, Any] = {'source': FLOOR_CONFIG, 'configured':
                                float(configured), 'n': 0, 'q05': None,
                                'percentile': self.PERCENTILE,
                                'path': str(self.path)}
        try:
            values = np.asarray(sorted(self.load().values()), dtype=float)
        except (TypeError, ValueError):
            values = np.empty(0)
        values = values[np.isfinite(values)]
        meta['n'] = int(values.size)
        if values.size < self.MIN_N:
            meta['source'] = FLOOR_THIN
            meta['why'] = (f'{values.size} graded pair(s) on record, below '
                           f'min_n={self.MIN_N}: the configured floor stands')
            return float(configured), meta
        q = float(np.percentile(values, self.PERCENTILE))
        meta['q05'] = round(q, 6)
        floor = min(float(configured), max(0.0, q))
        meta['source'] = FLOOR_FITTED
        meta['clamped_by_config'] = floor == float(configured) and q >= float(
            configured)
        return floor, meta

    def observe(self, val_rows: Iterable[Dict]) -> int:
        """Add this run's supervised evidence; return the pairs newly recorded.

        Keyed by the pair, so re-running the same query contributes nothing
        twice.  Written atomically, and a write failure is a note, never an
        aborted run.
        """
        fresh: Dict[str, float] = {}
        for r in val_rows or []:
            verdict = str(r.get('verdict') or '')
            if verdict != 'matched' and not verdict.startswith('verified'):
                continue
            try:
                jac = float(r.get('jaccard'))
                key = (f"{int(r.get('source_bodyId'))}"
                       f":{int(r.get('target_bodyId'))}")
            except (TypeError, ValueError):
                continue
            if np.isfinite(jac) and jac > 0:
                fresh[key] = jac
        if not fresh:
            return 0
        merged = self.load()
        before = len(merged)
        merged.update(fresh)
        if len(merged) > self.CAP:
            # keep the lowest values: q05 is read off that tail, so trimming
            # there would be the one trim that could move the fitted floor
            keep = sorted(merged.items(), key=lambda kv: kv[1])[:self.CAP]
            merged = dict(keep)
        payload = {
            'schema': 1,
            'source_dataset': self.source_dataset,
            'target_dataset': self.target_dataset,
            'metric': 'jaccard of pairs the supervised path graded '
                      'matched/verified (advisory evidence for the pooling '
                      'floor — not a recall set)',
            'pairs': merged,
            'n': len(merged),
            'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.json.tmp')
            with open(tmp, 'w') as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            return 0
        return max(0, len(merged) - before)


def resolve_jaccard_floor(cfg, project_root: Optional[str] = None
                          ) -> Tuple[float, Dict[str, Any]]:
    """The floor this pass will gate on: fitted when evidence allows, else the
    configured constant.  Read once, at the top of the pass, and published.

    `project_root` follows the pipeline's convention — the tree the run lives in
    owns the store, so a scratch tree with its own frozen `cache/` clone fits
    against its own evidence and cannot move another tree's floor.
    """
    configured = float(cfg.pooling_jaccard_floor)
    if not getattr(cfg, 'pooling_floor_from_evidence', False):
        return configured, {'source': FLOOR_CONFIG, 'configured': configured,
                            'n': 0, 'q05': None,
                            'why': 'pooling_floor_from_evidence is off'}
    try:
        return JaccardFloorStore(cfg.source_dataset, cfg.target_dataset,
                                 project_root).fit(configured)
    except Exception as exc:  # noqa: BLE001 — a missing store is not a failure
        return configured, {'source': FLOOR_CONFIG, 'configured': configured,
                            'n': 0, 'q05': None,
                            'why': f'unavailable: {type(exc).__name__}: {exc}'}


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

    ``jaccard_floor`` is the RESOLVED floor (see `resolve_jaccard_floor`), not
    the config field: the run must gate on one number and publish that number.

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
    """
    cfg = validator.cfg
    info = {'attempted': 0, 'scored': 0, 'qualified': 0, 'capped': 0,
            'error': ''}
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
    info['scored'] = len(take) if mq.active else 0
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


def corroborate(run_dirs: Sequence[str], target_dataset: str) -> Dict[str, Any]:
    """Join this run's candidates against sibling pooling runs' pools.

    One run has one target dataset, so cross-target agreement cannot be
    computed inside a run: it is a join over the run folders of the same query
    (`--pooling-corroborate-with`).  A candidate bodyId means nothing across
    datasets, so the key is the (source bodyId, candidate target TYPE) pair —
    the only identity that transfers.  Advisory by decision: it ranks the
    review, it never gates.
    """
    out: Dict[str, Any] = {'status': 'not-computed', 'sibling_runs': 0,
                           'sibling_dirs': [],
                           'how': 'pass --pooling-corroborate-with <run dirs> '
                                  'to join the same query against other '
                                  'targets',
                           'by_pair': {}, 'unreadable': []}
    dirs = [str(d) for d in (run_dirs or []) if str(d or '').strip()]
    out['sibling_dirs'] = dirs
    for d in dirs:
        p = os.path.join(d, 'pooling', 'pooling_pool.csv')
        if not os.path.exists(p):
            p = os.path.join(d, 'pooling_pool.csv')
        if not os.path.exists(p):
            out['unreadable'].append(d)
            continue
        ds = _sibling_dataset(d)
        # read the whole sibling before merging anything: a file that breaks
        # half way must contribute a partial join to nobody
        found: Dict[Tuple[int, str], set] = {}
        try:
            with open(p, newline='', errors='ignore') as fh:
                for r in csv.DictReader(fh):
                    key = (int(r['best_source_bodyId']),
                           str(r.get('target_type') or ''))
                    found.setdefault(key, set()).add(ds)
        except (OSError, KeyError, ValueError, TypeError):
            out['unreadable'].append(d)
            continue
        for key, datasets in found.items():
            out['by_pair'].setdefault(key, set()).update(datasets)
    out['sibling_runs'] = len(dirs) - len(out['unreadable'])
    out['status'] = 'computed' if out['sibling_runs'] else 'not-computed'
    out['self_dataset'] = str(target_dataset or '')
    return out


def _sibling_dataset(run_dir: str) -> str:
    """A sibling run's TARGET dataset, which is what a corroboration count
    counts.  The folder name is a timestamped label, not a dataset."""
    try:
        with open(os.path.join(run_dir, 'parameters.json'),
                  errors='ignore') as fh:
            ds = str((json.load(fh) or {}).get('target_dataset') or '')
    except (OSError, ValueError):
        return os.path.basename(run_dir.rstrip('/'))
    return ds or os.path.basename(run_dir.rstrip('/'))


def _corroborated(corr: Dict[str, Any], source_bid: int,
                  target_type: str) -> Optional[int]:
    """How many target datasets put this (source, candidate type) pair in
    their pool, this run's own included.

    Blank, not 0, when the join was never asked for: a 0 would read as "no
    other target supports this pair", which is a claim about runs that were
    not run.
    """
    if corr.get('status') != 'computed':
        return None
    datasets = set(corr.get('by_pair', {}).get((int(source_bid),
                                                str(target_type or '')), ()))
    datasets.add(str(corr.get('self_dataset') or ''))
    datasets.discard('')
    return len(datasets)


def corroborate_pool(rows: List[Dict], pool: List[Dict],
                     corr: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the advisory count into every exported row and summarise it.

    The key is the source-side pair of each row, so a candidate the same
    source reaches under two targets is compared with itself — the identity
    that transfers across datasets is (source bodyId, target TYPE), never a
    target bodyId.
    """
    for r in rows:
        r['targets_corroborated'] = _corroborated(
            corr, int(r['source_bodyId']), r.get('target_type', ''))
    for p in pool:
        p['targets_corroborated'] = _corroborated(
            corr, int(p['best_source_bodyId']), p.get('target_type', ''))
    hist: Dict[str, int] = {}
    for p in pool:
        v = p['targets_corroborated']
        key = 'not-computed' if v is None else str(v)
        hist[key] = hist.get(key, 0) + 1
    return {
        'status': corr.get('status', 'not-computed'),
        'key': '(source bodyId of the queried dataset, candidate target '
               'TYPE name) — target bodyIds do not transfer across datasets',
        'sibling_runs': corr.get('sibling_runs', 0),
        'sibling_dirs': list(corr.get('sibling_dirs') or []),
        'unreadable': list(corr.get('unreadable') or []),
        'how': corr.get('how', ''),
        'pool_size_histogram': dict(sorted(hist.items(),
                                           key=lambda x: -x[1])),
        'advisory': 'ranks the review, never gates it (plan decision 6)',
    }


def cross_validation(pool: List[Dict], rows: List[Dict], refs: Dict[str, Any],
                     stats: Dict, morph: Dict, notes: List[str],
                     corroboration: Optional[Dict[str, Any]] = None,
                     ) -> Dict[str, Any]:
    """The §4.5 comparison: cells, the containment numbers, and the three
    honesty rules stated as data rather than as prose."""
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
        'corroboration': corroboration or {
            'status': 'not-computed',
            'how': 'pass --pooling-corroborate-with <run dirs> to join the '
                   'same query against other targets',
            'pool_size_histogram': {}},
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
        ] + list(notes or []),
    }


def run_pooling(validator, *, target_stats, target_bids, target_id2type,
                val_rows) -> Dict[str, Any]:
    """The whole pass.  Returns ``{'candidates', 'pool', 'cross_validation'}``.

    Reads no pool, no branch bar and no mapper claim on the selection path;
    the seed comes from the source dataset's annotation and the gate is
    absolute.
    """
    cfg = validator.cfg
    notes: List[str] = []
    # READ the fitted floor before anything is selected. The evidence store is
    # written at the END of this function, so this run's own claims cannot move
    # this run's gate — the unsupervised property depends on that ordering.
    j_floor, j_meta = resolve_jaccard_floor(
        cfg, getattr(validator, 'project_root', None))
    store = JaccardFloorStore(cfg.source_dataset, cfg.target_dataset,
                              getattr(validator, 'project_root', None))
    seed = seed_population(validator, cfg.query_types)
    if not seed:
        notes.append('pooling: the queried population resolved to 0 neurons — '
                     'nothing scanned.')
        return {'candidates': [], 'pool': [],
                'cross_validation': cross_validation(
                    [], [], {'types': set(), 'pools': set(), 'pairs': {}},
                    {'gate': {'jaccard_floor': j_floor,
                              'jaccard_floor_source': j_meta.get('source'),
                              'rank_union_floor': cfg.pooling_rank_union_floor,
                              'window_mult': cfg.pooling_window_mult}},
                    {}, notes)}
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
    pool = pool_by_target(rows)
    corr = corroborate(getattr(cfg, 'pooling_sibling_runs', None) or [],
                       cfg.target_dataset)
    corroboration = corroborate_pool(rows, pool, corr)
    gate = {'jaccard_floor': j_floor,
            'jaccard_floor_configured': cfg.pooling_jaccard_floor,
            'jaccard_floor_source': j_meta.get('source'),
            'jaccard_floor_evidence_n': j_meta.get('n'),
            'jaccard_floor_q05': j_meta.get('q05'),
            'rank_union_floor': cfg.pooling_rank_union_floor,
            'window_mult': cfg.pooling_window_mult}
    if j_meta.get('why'):
        gate['jaccard_floor_note'] = str(j_meta['why'])
    # WRITE the evidence last, and only from this run's supervised grades: the
    # next run's floor may learn from this one, this run's gate never does.
    try:
        added = store.observe(val_rows)
        gate['jaccard_floor_pairs_added'] = int(added)
    except Exception as exc:  # noqa: BLE001 — a store is never worth a failed pass
        notes.append(f'pooling: the J-floor evidence store could not be '
                     f'updated ({type(exc).__name__}: {exc}); the next run '
                     f'starts from what is already on record.')
    stats.update({'seed': len(seed), 'universe': len(target_bids),
                  'gate': gate})
    xval = cross_validation(pool, rows, refs, stats, morph, notes,
                            corroboration=corroboration)
    validator.log(f"[pooling] {len(seed)} queried sources scanned "
                  f"{stats['rows_scored']:,} pairs -> {stats['rows_kept']} "
                  f"rows, {len(pool)} candidate targets "
                  f"({xval['cells'][CELL_POOL_MISS]} outside every mapper "
                  f"pool); morph {morph}; corroboration "
                  f"{corroboration['status']} "
                  f"({corroboration['sibling_runs']} sibling run(s))")
    validator.log(f"[pooling] Jaccard floor {j_floor:.4f} "
                  f"({j_meta.get('source')}; configured "
                  f"{cfg.pooling_jaccard_floor}; "
                  f"{j_meta.get('n', 0)} graded pair(s) on record, "
                  f"q05={j_meta.get('q05')}"
                  + (f"; {j_meta['why']}" if j_meta.get('why') else '') + ")")
    return {'candidates': rows, 'pool': pool, 'cross_validation': xval,
            'stats': stats}
