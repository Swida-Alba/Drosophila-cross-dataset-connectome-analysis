"""bodyId-level resolution backend — ONE integrated, mapper-layer module.

Plan: ``_plan/plan-bodyid-level-granularity-in-type-mapper.md`` (Revision 5).

Ownership (Revision 3): every bodyId-level primitive and service lives
HERE; consumers import from this module and never re-implement:

- ``comparison.mapping_validation`` — the validation pipeline keeps only
  orchestration (verdicts, categories, gap fill, morphology, reports) and
  re-imports the moved names below; ``MappingValidator`` runs on a
  ``BodyIdResolver`` with its benchmark profiler injected.
- the informational UI surfaces: NONE since 2026-09-14 (the panel's
  connectivity split view was removed — the type mapper carries only
  row-based bridge evidence).  Any future UI for connectivity
  verification belongs to the validate-expand-visualize pipeline's own
  surfaces.

Two capability layers share this module:

- **pool-scoped resolution** (``BodyIdResolver.assign_bodyids``):
  verification/display machinery — NOT a mapper mapping mechanism (the
  type mapper's granularity is row-based bridge evidence only).  Profiles
  are built on demand ONLY for the requested neurons (per-neuron
  ``get_profile`` — ~40 ms/neuron after a one-time per-dataset
  connection-index build; NO bulk dataset profiling, NO global scans).
- **dataset-scale scan services** (``build_target_vectors`` /
  ``scan_source``): validation-only in usage — the whole-dataset
  machinery behind the validation pipeline's global verdicts.  The
  mapper-facing API never calls them.

bodyId integrity: FAFB **and BANC** bodyIds (~7.2e17) exceed float64's
exact range — ids stay Python ints end-to-end; DataFrame id columns stay
object dtype (see ``scan_source``).
"""

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison.connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    DataNotAvailableError,
    ProfilerConfig,
)
from comparison.profile_comparator import ProfileComparator

if TYPE_CHECKING:  # never imported at runtime (import-cycle guard)
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper


# ---------------------------------------------------------------------------
# Scoring — exact parity with ProfileComparator.batch_compare_cross_dataset
# (moved verbatim from mapping_validation; single implementation)
# ---------------------------------------------------------------------------

def expanded_vector(profile: ConnectivityProfile,
                    type_mapper=None) -> Dict[str, float]:
    """Standardized expanded-type vector ('both' direction), as production."""
    return ProfileComparator._get_expanded_types_standardized(
        profile, 'both', type_mapper)


def score_one_candidate(src_types: Dict[str, float],
                        tgt_types: Dict[str, float]) -> Optional[Dict]:
    """Reference scorer for one (source, target) expanded-vector pair.

    Reproduces the ``batch_compare_cross_dataset`` metric block exactly:
    jaccard over expanded-type sets; rank_union = Spearman over the union
    with missing = 0.0 (NaN when |U| < 3 or either weight list has a single
    distinct value); cosine and weighted_jaccard over the same union.
    Returns None for an empty target vector (production's all-NaN row can
    never rank).
    """
    if not tgt_types:
        return None
    return score_one_candidate_fast(_SideStats(src_types), _SideStats(tgt_types))


class _SideStats:
    """Precomputed per-vector quantities shared by the fast metrics."""

    __slots__ = ('vec', 'keys', 'sum1', 'norm')

    def __init__(self, vec: Dict[str, float]):
        self.vec = vec
        self.keys = set(vec)
        self.sum1 = float(sum(vec.values()))
        self.norm = math.sqrt(float(sum(v * v for v in vec.values())))


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1-based) — matches scipy.stats.rankdata."""
    sorter = np.argsort(x, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    arr_sorted = x[sorter]
    obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    am = a - a.mean()
    bm = b - b.mean()
    denom = math.sqrt(float((am * am).sum()) * float((bm * bm).sum()))
    return float((am * bm).sum()) / denom if denom > 0 else np.nan


def score_one_candidate_fast(src: _SideStats,
                             tgt: _SideStats) -> Optional[Dict]:
    """Fast exact scorer (see score_one_candidate for the contract)."""
    inter = src.keys & tgt.keys
    union = src.keys | tgt.keys
    jaccard = len(inter) / len(union) if union else 0.0

    cosine = 0.0
    if inter and src.norm > 0 and tgt.norm > 0:
        sv, tv = src.vec, tgt.vec
        dot = sum(sv[k] * tv[k] for k in inter)
        cosine = dot / (src.norm * tgt.norm)

    min_inter = max_inter = 0.0
    sum_inter_a = sum_inter_b = 0.0
    sv, tv = src.vec, tgt.vec
    for k in inter:
        a, b = sv[k], tv[k]
        sum_inter_a += a
        sum_inter_b += b
        min_inter += a if a < b else b
        max_inter += a if a > b else b
    w_union = (max_inter + (src.sum1 - sum_inter_a)
               + (tgt.sum1 - sum_inter_b))
    weighted_jaccard = min_inter / w_union if w_union > 0 else 0.0

    rank_union = np.nan
    if len(union) >= 3:
        a = np.array([sv.get(k, 0.0) for k in union])
        b = np.array([tv.get(k, 0.0) for k in union])
        # production NaN rule: either weight list with a single distinct
        # value carries no monotone information
        if len(set(a.tolist())) > 1 and len(set(b.tolist())) > 1:
            try:
                rank_union = _pearson(_rankdata_average(a),
                                      _rankdata_average(b))
            except Exception:
                rank_union = np.nan

    return {
        'jaccard': jaccard,
        'rank_union': rank_union,
        'cosine': cosine,
        'weighted_jaccard': weighted_jaccard,
        'shared_type_count': len(inter),
        'union_type_count': len(union),
    }


def passes_target_quality_gate(vec: Dict[str, float],
                               min_weight: Optional[float] = None,
                               min_partner_types: Optional[int] = None
                               ) -> bool:
    """Revision 3.5 Issue 5a: target-side profile-quality gate.

    A scan candidate must carry at least ``min_weight`` total expanded
    weight and at least ``min_partner_types`` distinct partner types;
    anything below the floors (e.g. a 3-synapse UNIDIRECTIONAL fragment)
    never enters a scan, so it cannot win a ranking on ordering noise.
    """
    if min_weight is None and min_partner_types is None:
        return True
    total = float(sum(v for v in vec.values()))
    if min_weight is not None and total < float(min_weight):
        return False
    if min_partner_types is not None and len(vec) < int(min_partner_types):
        return False
    return True


#: The bodyId-level ordering chain (plan-tmvev-jaccard-primary-bodyid-ranking
#: J3 + D-B): Jaccard desc, rank_union desc as the tie-break, then bodyId so
#: the order is TOTAL.  This is the answer to a measurement on run r16: 36%
#: of reverse scans have a DUPLICATED ``jaccard_rank`` inside their top-5
#: (0% for ``rank_union_rank``), because Jaccard is a ratio of small
#: integers — and the rank columns use ``method='min'``, so a tie block
#: shares one rank and any window taken on a rank column alone resolves by
#: whatever the row order happens to be.  Ordering on the scores instead
#: makes "top-N" mean N rows, reproducibly.  rank_union NaN (under 3 union
#: types) sorts LAST within a Jaccard tie, never first.
#:
#: The metric rank columns stay published unchanged: they are the EVIDENCE a
#: verdict is read off (the reciprocal grade's top-1 / top-3 windows, the
#: forward ladder's "rank 1 by either metric"), and a collapsed tie rank is
#: the honest statement of "these are indistinguishable by this metric".
_CHAIN = ['jaccard', 'rank_union', 'target_bid']
_CHAIN_ASC = [False, False, True]


def order_by_chain(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` in :data:`_CHAIN` order (Jaccard desc, rank_union desc,
    bodyId).  The public form of the chain, so the forward pipeline and the
    reverse pass take their "best" and their top-N windows from one
    definition instead of each re-spelling a rank-column sort."""
    return df.sort_values(_CHAIN, ascending=_CHAIN_ASC, na_position='last')


def _rank_key(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.inf
    return f if f == f else math.inf


def _neg_score_key(v) -> float:
    """Ascending key for a "higher is better" score; blank/NaN goes last."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.inf
    return -f if f == f else math.inf


def chain_key(row) -> Tuple[float, float, float, float, float]:
    """The single-row form of :data:`_CHAIN`, for code that compares rows
    rather than sorting a frame: ``min(rows, key=chain_key)`` is
    ``order_by_chain(df).iloc[0]``.

    Accepts a ``Series`` or a dict read back from a run CSV.  The rank
    columns lead because a ``method='min'`` tie block must resolve by the
    scores the ranks collapsed, not by inherited row order (J3); the scores
    then separate what a rank flattened, and bodyId last makes the order
    total.  The two agree with :func:`order_by_chain` because equal scores
    are what produce equal ranks in the first place."""
    return (_rank_key(row.get('jaccard_rank')),
            _rank_key(row.get('rank_union_rank')),
            _neg_score_key(row.get('jaccard')),
            _neg_score_key(row.get('rank_union')),
            _rank_key(row.get('target_bid')
                      if row.get('target_bid') is not None
                      else row.get('target_bodyId')))


def _ranks_squared_sum(values: np.ndarray) -> float:
    """Σ of the squared average-tie ranks of one weight vector.

    The sum of the ranks themselves is always ``n(n+1)/2`` whatever the tie
    structure, so Σρ² is the only per-vector statistic a closed form needs.
    """
    if values.size == 0:
        return 0.0
    return float(np.square(_rankdata_average(values)).sum())


def _disjoint_rank_union(p: int, r2_src: float, q: np.ndarray,
                         r2_tgt: np.ndarray) -> np.ndarray:
    """Exact ``rank_union`` for every pair (src, tgt) whose key sets are
    DISJOINT — that is, for every ``jaccard == 0`` row of a scan frame.

    For a disjoint pair the union is ``[src keys…, tgt keys…]``, so each
    side is the other's padding: the source weights read ``q + ρ`` over their
    own keys and the constant ``(q+1)/2`` over the target's, and vice versa
    (ρ, σ are the vectors' own average-tie ranks).  Pearson over that collapses
    to four scalars per side — the key count and Σρ² — because Σρ is
    ``p(p+1)/2`` whatever the tie structure is.

    This is an identity, not an approximation: the parity harness
    (``tests/core/test_mapping_validation_scan_parity.py``) checks it against
    :func:`score_one_candidate_fast` on real disjoint pairs.  It is valid only
    when every weight is STRICTLY POSITIVE (a zero weight would tie with the
    other side's padding) and the source side is non-empty; the caller gates
    both and falls back to the scorer otherwise.
    """
    q = q.astype(float)
    n = p + q
    tri_p = p * (p + 1) / 2.0
    tri_q = q * (q + 1) / 2.0
    total = p * q + tri_p + tri_q                      # ΣX == ΣY
    t2n = total * total / n
    sum_xy = ((p + 1) / 2.0 * (p * q + tri_p)
              + (q + 1) / 2.0 * (p * q + tri_q))
    sum_x2 = p * q * q + q * p * (p + 1) + r2_src + q * (q + 1) ** 2 / 4.0
    sum_y2 = q * p * p + p * q * (q + 1) + r2_tgt + p * (p + 1) ** 2 / 4.0
    cov = sum_xy - t2n
    var_x = sum_x2 - t2n
    var_y = sum_y2 - t2n
    out = np.full(n.shape, np.nan, dtype=float)
    ok = (n >= 3) & (var_x > 0) & (var_y > 0)
    out[ok] = cov[ok] / np.sqrt(var_x[ok] * var_y[ok])
    return out


class TargetScanIndex(dict):
    """``bid -> _SideStats`` plus the derived structures that turn a
    whole-universe scan into vector arithmetic.

    98 % of a scan frame's cost sits in pairs that share no expanded type at
    all: :func:`score_one_candidate_fast` still builds their union, ranks both
    sides and runs Pearson — for a row that can only ever read
    ``jaccard == 0``.  Measured on the real ``banc_v888`` universe (101,978
    targets), 99.3 % of the pairs are such rows and the scorer costs 3.64 s
    PER SOURCE NEURON on them.

    The index precomputes what the disjoint closed form
    (:func:`_disjoint_rank_union`) needs per target, plus one postings list
    per type key, so a scan reaches the ~0.5 % positive block directly
    through the postings and fills the rest of the frame arithmetically.

    What comes out is the SAME frame the per-pair loop builds — same rows in
    the same order, same columns, values bit-identical — so every consumer of
    a rank window (``metric == 1``, ``nsmallest(top_n)``, ``invaders_ahead``)
    and ``chain_pos`` reads exactly as before.  Plain dicts still work as
    they used to: the fast path is a property of the object, not of the call.
    """

    __slots__ = ('postings', 'q_arr', 'r2_arr', 'analytic_ok', 'row_of')

    def __init__(self, stats_by_bid: Dict[int, _SideStats]):
        super().__init__(stats_by_bid)
        postings: Dict[str, List[int]] = {}
        row_of: Dict[int, int] = {}
        qs: List[int] = []
        r2s: List[float] = []
        oks: List[bool] = []
        for bid, st in stats_by_bid.items():
            if st is None or not st.vec:
                continue                      # the reference loop skips these
            row_of[int(bid)] = len(qs)
            keys = list(st.keys)
            vals = np.array([st.vec[k] for k in keys], dtype=float)
            qs.append(len(keys))
            # a zero / NaN / negative weight breaks the padding assumption
            if not bool(np.isfinite(vals).all()) or bool((vals <= 0).any()):
                r2s.append(0.0)
                oks.append(False)
            else:
                r2s.append(_ranks_squared_sum(vals))
                oks.append(True)
            for k in keys:
                postings.setdefault(k, []).append(row_of[int(bid)])
        self.postings = {k: np.asarray(v, dtype=np.int64)
                         for k, v in postings.items()}
        self.q_arr = np.asarray(qs, dtype=np.int64)
        self.r2_arr = np.asarray(r2s, dtype=float)
        self.analytic_ok = np.asarray(oks, dtype=bool)
        self.row_of = row_of

    # -- the scan ----------------------------------------------------------

    def scan(self, src_vec: Dict[str, float],
             target_bids: Optional[List[int]] = None) -> pd.DataFrame:
        """One source against the indexed universe — the fast path of
        :func:`scan_source`, exact by construction (see this class's
        docstring for what keeps the frame identical)."""
        bids = target_bids if target_bids is not None else list(self)
        kept_bids: List[int] = []
        kept_stats: List[_SideStats] = []
        rows: List[int] = []
        for bid in bids:
            st = dict.get(self, bid)
            if st is None or not st.vec:
                continue
            kept_bids.append(bid)
            kept_stats.append(st)
            rows.append(self.row_of[int(bid)])
        if not rows:
            return pd.DataFrame()
        rows = np.asarray(rows, dtype=np.int64)
        n = len(rows)
        q = self.q_arr[rows]
        r2_tgt = self.r2_arr[rows]
        tgt_ok = self.analytic_ok[rows]

        src = _SideStats(src_vec)
        p = len(src.keys)
        union_n = p + q                       # |src ∪ tgt| while disjoint
        jac = np.zeros(n, dtype=float)
        shared = np.zeros(n, dtype=np.int64)
        pos_of = np.full(len(self.q_arr), -1, dtype=np.int64)
        pos_of[rows] = np.arange(n)
        hits = [self.postings[k] for k in src.keys if k in self.postings]
        if hits:
            loc = pos_of[np.concatenate(hits)]
            loc = loc[loc >= 0]
            if loc.size:
                uniq, counts = np.unique(loc, return_counts=True)
                shared[uniq] = counts
                union_n[uniq] = p + q[uniq] - counts
                jac[uniq] = counts / union_n[uniq]

        # rank_union: the closed form over the jaccard-zero block, the
        # reference scorer everywhere else (positives and any pair whose
        # weights break the padding assumption).
        ru = np.full(n, np.nan, dtype=float)
        vals = np.array([src.vec[k] for k in src.keys], dtype=float)
        src_ok = bool(p and np.isfinite(vals).all()
                      and not (vals <= 0).any())
        if src_ok:
            ru = _disjoint_rank_union(p, _ranks_squared_sum(vals), q, r2_tgt)
        positive = shared > 0
        need_ref = positive | ((~positive) & (~tgt_ok | (~src_ok)))
        # cosine / weighted_jaccard are structurally 0 on the disjoint block
        # (an empty intersection zeroes them in the scorer), so only these
        # rows can carry anything else.
        cos = np.zeros(n, dtype=float)
        wj = np.zeros(n, dtype=float)
        for i in np.nonzero(need_ref)[0]:
            m = score_one_candidate_fast(src, kept_stats[int(i)])
            ru[i] = m['rank_union']
            jac[i] = m['jaccard']
            union_n[i] = m['union_type_count']
            shared[i] = m['shared_type_count']
            cos[i] = m['cosine']
            wj[i] = m['weighted_jaccard']
        df = pd.DataFrame({
            'jaccard': jac,
            'rank_union': ru,
            'cosine': cos,
            'weighted_jaccard': wj,
            'shared_type_count': shared,
            'union_type_count': union_n,
        })
        df['target_bid'] = np.asarray(kept_bids, dtype=object)
        return _finish_frame(df)


def _finish_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The rank columns, the chain sort and ``chain_pos`` — one definition
    shared by the reference scan loop and :meth:`TargetScanIndex.scan`, so the
    two cannot drift apart."""
    if df.empty:
        return df
    # object dtype: mixed int/float frames upcast .iloc[0] rows to float64,
    # which silently corrupts FAFB bodyIds (> 2**53)
    df['target_bid'] = df['target_bid'].astype(object)
    df['rank_union_rank'] = df['rank_union'].rank(
        ascending=False, na_option='bottom', method='min')
    df['jaccard_rank'] = df['jaccard'].rank(
        ascending=False, na_option='bottom', method='min')
    df = df.sort_values(_CHAIN, ascending=_CHAIN_ASC, na_position='last')
    # 1-based position IN THE CHAIN, dense (unlike the rank columns, which a
    # tie block collapses): this is what a "top-N" window means from here on
    df['chain_pos'] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def scan_source(src_vec: Dict[str, float],
                target_stats: Dict[int, _SideStats],
                target_bids: Optional[List[int]] = None) -> pd.DataFrame:
    """Score one source vector against every target; rank globally.

    Validation-only scan service (dataset scale).  Returns a DataFrame
    (target_bid, metrics, rank_union_rank, jaccard_rank, chain_pos) sorted by
    the bodyId ordering chain :data:`_CHAIN`.  The two rank columns are
    competition-style metric EVIDENCE (ties share the better rank, NaN last)
    and are deliberately not the row order — see :data:`_CHAIN` for why.

    ``target_stats`` as built by :func:`prep_target_stats` is a
    :class:`TargetScanIndex`, which takes the vector path and returns the
    identical frame; a plain dict of ``_SideStats`` (tests, pool-scoped
    callers) still runs the per-pair loop below.
    """
    if isinstance(target_stats, TargetScanIndex):
        return target_stats.scan(src_vec, target_bids=target_bids)
    src = _SideStats(src_vec)
    rows = []
    bids = target_bids if target_bids is not None else list(target_stats)
    for tbid in bids:
        tgt = target_stats.get(tbid)
        if tgt is None or not tgt.vec:
            continue
        m = score_one_candidate_fast(src, tgt)
        m['target_bid'] = tbid
        rows.append(m)
    df = pd.DataFrame(rows)
    return _finish_frame(df)


def build_target_vectors(profiler: ConnectivityProfiler, dataset: str,
                         type_mapper=None, verbose: bool = True,
                         min_weight: Optional[float] = None,
                         min_partner_types: Optional[int] = None
                         ) -> Dict[int, Dict[str, float]]:
    """Expanded-type vector for EVERY cached bodyId profile of a dataset.

    Validation-only scan service (streams the dataset's whole profile
    parquet — never used by the pool-scoped resolver path).  Streams the
    profile-cache parquet in row batches and keeps only the expanded
    dicts (full ConnectivityProfile objects for ~140k neurons do not fit
    comfortably in memory).  Vector content matches what
    ``batch_compare_cross_dataset`` derives per candidate.  When
    ``min_weight`` / ``min_partner_types`` are set (Revision 3.5 Issue 5a),
    candidates below the profile-quality floors are excluded from the
    returned universe entirely.
    """
    import pyarrow.parquet as pq

    parquet_path = Path(profiler._get_cache_parquet_path(dataset))
    if not parquet_path.exists():
        raise FileNotFoundError(
            f'profile cache parquet not found for {dataset}: {parquet_path}')

    dict_cols = [
        'upstream_partners', 'downstream_partners',
        'untyped_upstream_bodyids', 'untyped_downstream_bodyids',
        'untyped_upstream_2hop', 'untyped_downstream_2hop',
    ]
    pf = pq.ParquetFile(str(parquet_path))
    keep_cols = [c for c in ['neuron_id'] + dict_cols
                 if c in pf.schema_arrow.names]

    vectors: Dict[int, Dict[str, float]] = {}
    gated = 0
    t0 = time.time()
    n_rows = 0
    for batch in pf.iter_batches(batch_size=20000, columns=keep_cols):
        cols = {name: batch.column(name).to_pylist() for name in keep_cols}
        for i in range(batch.num_rows):
            bid = int(cols['neuron_id'][i])  # int64 -> python int, exact
            up = json.loads(cols['upstream_partners'][i]) \
                if cols['upstream_partners'][i] else {}
            dn = json.loads(cols['downstream_partners'][i]) \
                if cols['downstream_partners'][i] else {}
            if not up and not dn:
                continue
            profile = ConnectivityProfile(
                neuron_id=bid, dataset=dataset,
                upstream_partners={str(k): float(v) for k, v in up.items()},
                downstream_partners={str(k): float(v) for k, v in dn.items()},
            )
            for col in dict_cols[2:]:
                raw = cols.get(col)
                if raw and raw[i]:
                    val = json.loads(raw[i]) or {}
                    setattr(profile, col,
                            {int(k): v for k, v in val.items()})
            vec = expanded_vector(profile, type_mapper)
            if not passes_target_quality_gate(vec, min_weight,
                                              min_partner_types):
                gated += 1
                continue
            vectors[bid] = vec
        n_rows += batch.num_rows
        if verbose:
            print(f'    [vectors] {n_rows} rows '
                  f'({time.time() - t0:.0f}s, {len(vectors)} scored)')
    if verbose:
        print(f'    [vectors] done: {len(vectors)} expanded vectors '
              f'from {n_rows} rows in {time.time() - t0:.0f}s'
              + (f' ({gated} below quality gate)' if gated else ''))
    return vectors


def prep_target_stats(vectors: Dict[int, Dict[str, float]]
                      ) -> TargetScanIndex:
    """The scan universe, indexed.

    Returns a :class:`TargetScanIndex` — still a plain ``bid -> _SideStats``
    mapping for every existing consumer — so a whole-universe scan costs
    milliseconds per source instead of seconds.
    """
    return TargetScanIndex(
        {bid: _SideStats(vec) for bid, vec in vectors.items() if vec})


# ---------------------------------------------------------------------------
# Reverse (target -> source) scan services — the TM VEV backward evidence
# ---------------------------------------------------------------------------

#: The four states of a backward label — the CSV token IS the display
#: label (user 2026-09-19: no token↔label translation layer).  The three
#: scanned verdicts grade HOW PROMINENTLY the member's own branch source
#: type ranks in the reverse scan; ``not-checked`` is the default so a run
#: with the pass disabled keeps byte-stable CSV headers.
BACKWARD_EVIDENCE_VALUES = ('high', 'medium', 'low', 'not-checked')

#: Columns the backward pass adds to an expansion row.  Connectivity only:
#: morphology is deliberately not re-scored here — the candidate bins already
#: carry their stage-5 morph columns and family/relative inherit theirs from
#: the branch's matched pool.
BACKWARD_COLUMNS = [
    'backward_evidence', 'backward_top1_source_bodyId',
    'backward_top1_source_type', 'backward_top1_in_branch',
    'backward_rank_union', 'backward_jaccard',
    'backward_shared_type_count', 'backward_union_type_count',
    'backward_rank_union_rank', 'backward_jaccard_rank',
    'backward_n_out_of_branch', 'backward_size_ratio',
    'backward_size_filtered', 'backward_thin_evidence', 'backward_topN',
    'backward_scanned_at',
    # The hit the GRADE actually rests on: the best-ranked source of the
    # claiming branch's own type, whichever ranking placed it there
    # (plan-tmvev-reciprocal-jaccard-sort-and-parity D1c).  Without these
    # the row is graded on evidence the cell never shows — measured on r15,
    # 20 of 49 high/medium rows displayed a top-1 of a different type.
    'backward_own_type_rank_source_bodyId',
    'backward_own_type_rank_source_type', 'backward_own_type_via',
    'backward_own_type_rank_union', 'backward_own_type_jaccard',
    'backward_own_type_shared_type_count',
    'backward_own_type_union_type_count',
    'backward_own_type_rank_union_rank', 'backward_own_type_jaccard_rank',
    'backward_own_type_thin_evidence',
]

#: The two metric EVIDENCE columns, in the order the reciprocal grade reads
#: them (a tie goes to the leading one, ``jaccard`` — J1).  Ordering of rows
#: is NOT here — it is :data:`_CHAIN`, which sorts on the scores so a tie
#: block cannot silently delegate the decision to row order.
_RANK_COLS = ['jaccard_rank', 'rank_union_rank']

#: A reverse hit whose two vectors share at most this many partner types.
#: ``rank_union`` ranks the union with missing types scored 0.0, so a
#: 2-shared-type pair CAN clear ``matched_ru_min`` on almost no evidence
#: (measured 2026-09-19: 209/819 random source pairs above 0.1 rest on <= 3
#: shared types).  The scorer already counts them
#: (:func:`score_one_candidate_fast`); this only publishes the count and
#: flags it.  NEVER a gate — see :data:`BACKWARD_COLUMNS`.
THIN_SHARED_TYPE_COUNT = 3


def _clean_num(v):
    """float or None (NaN/None collapse to None so CSVs stay readable)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clean_int(v):
    f = _clean_num(v)
    return None if f is None else int(f)


def _clean_type(v) -> str:
    s = '' if v is None else str(v)
    return '' if s in ('?', 'nan', 'None', 'Unknown') else s


def blank_backward_fields(scanned_at: str = 'disabled') -> Dict:
    """The no-evidence state for every :data:`BACKWARD_COLUMNS` field.

    ``scanned_at`` records WHY nothing was scored so a ``not-checked`` row
    is never mistaken for a negative result: ``disabled`` (the pass is off,
    the per-row default), ``cap`` (over ``backward_max_neurons`` /
    ``backward_per_branch_cap``), ``no_profile`` (no usable connectivity
    profile — absent, or a status the scan skips), ``error`` (the scan
    universe was unavailable, or this member's scan raised).  ``run`` is
    set by :func:`classify_backward_scan` when a grade was produced —
    including ``low``, which IS a graded negative: scanned, and the
    branch's own source type ranked outside every top-3."""
    out: Dict = {c: None for c in BACKWARD_COLUMNS}
    out['backward_evidence'] = 'not-checked'
    out['backward_size_filtered'] = False
    # advisory flags read "no warning", not "unknown", when nothing ran,
    # and the two serialized/label fields read "nothing there" as empty
    # rather than NaN — same convention as ``backward_topN``.
    out['backward_thin_evidence'] = False
    out['backward_own_type_thin_evidence'] = False
    out['backward_topN'] = ''
    out['backward_own_type_via'] = ''
    out['backward_scanned_at'] = scanned_at
    return out


def _with_chain_pos(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` in :data:`_CHAIN` order with a dense 1-based
    ``chain_pos`` column, adding it when the frame did not come from
    :func:`scan_source` (hand-built frames, subsets, older callers).

    Position, not rank, is what a "top-N" window means here: a rank column
    with ``method='min'`` collapses a tie block onto one value, so ``<= k``
    admits an unbounded set and ``nsmallest(k)`` delegates the choice to row
    order (plan-tmvev-jaccard-primary-bodyid-ranking D-B)."""
    if 'chain_pos' in df.columns:
        return df
    out = df.sort_values(_CHAIN, ascending=_CHAIN_ASC,
                         na_position='last').reset_index(drop=True)
    out['chain_pos'] = range(1, len(out) + 1)
    return out


def _best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """The chain-best row of one scan (Jaccard, then rank_union, then
    bodyId) — the hit the reciprocal top-1 column publishes (user
    2026-09-20, J1)."""
    usable = df[df['jaccard'].notna() | df['rank_union'].notna()]
    if usable.empty:
        return None
    return usable.sort_values(_CHAIN, ascending=_CHAIN_ASC,
                              na_position='last').iloc[0]


def serialize_backward_topN(df: Optional[pd.DataFrame], id2type=None,
                            top_n: int = 5, branch_pool=None) -> str:
    """The reverse neighbourhood as
    ``ru_rank|jac_rank|bid|type|ru|jaccard|in_branch`` records joined by
    ``;``, listed in :data:`_CHAIN` order (Jaccard first — user 2026-09-20).

    The report shows the top-1 in the cell and this list in the hover, so the
    whole neighbourhood travels in one CSV column (the same list-in-cell
    convention as ``same_name_rivals``).  Both rank columns travel because a
    hit can be Jaccard-1 while ranking poorly by rank_union, and a
    single-metric ``#`` column would hide that."""
    if df is None or df.empty or int(top_n or 0) <= 0:
        return ''
    pool = {int(b) for b in (branch_pool or [])}
    parts: List[str] = []
    head = df.sort_values(_CHAIN, ascending=_CHAIN_ASC,
                          na_position='last').head(int(top_n))
    for r in head.itertuples(index=False):
        try:
            bid = int(r.target_bid)
        except (TypeError, ValueError):
            continue
        ru = _clean_num(getattr(r, 'rank_union', None))
        jac = _clean_num(getattr(r, 'jaccard', None))
        parts.append('|'.join([
            str(_clean_int(getattr(r, 'rank_union_rank', None)) or ''),
            str(_clean_int(getattr(r, 'jaccard_rank', None)) or ''),
            str(bid),
            _clean_type((id2type or {}).get(bid)),
            '' if ru is None else f'{ru:.4f}',
            '' if jac is None else f'{jac:.4f}',
            '1' if bid in pool else '0',
        ]))
    return ';'.join(parts)


def topn_union_rows(df: Optional[pd.DataFrame], k: int = 3
                    ) -> Optional[pd.DataFrame]:
    """The rows :func:`serialize_topn_union` serializes: the top-``k`` by
    ``jaccard_rank`` UNION the top-``k`` by ``rank_union_rank``, deduped,
    in :data:`_CHAIN` order.  Rank ties beyond the rank column break by the
    chain so window membership is deterministic.  Callers that only need
    the bodyIds (reducing a scan frame before its types are known) use
    this; the serializer and the pipeline share one window definition."""
    if df is None or df.empty or int(k or 0) <= 0:
        return None if df is None else df.iloc[0:0]
    picked: set = set()
    for col in _RANK_COLS:
        sub = df[df[col].notna()]
        if sub.empty:
            continue
        for r in sub.sort_values(
                [col] + _CHAIN, ascending=[True] + _CHAIN_ASC,
                na_position='last').head(int(k)).itertuples(index=False):
            try:
                picked.add(int(r.target_bid))
            except (TypeError, ValueError):
                continue  # same NaN/None-bid tolerance as serialize_backward_topN
    if not picked:
        return df.iloc[0:0]
    return order_by_chain(df[df['target_bid'].isin(picked)])


def serialize_topn_union(df: Optional[pd.DataFrame], id2type=None,
                         k: int = 3, branch_pool=None) -> str:
    """The neighbourhood as the UNION of the top-``k`` of each rank
    column (user 2026-09-26, the homolog-panel hover payload).

    Same 7-field record format as :func:`serialize_backward_topN`
    (``ru_rank|jac_rank|bid|type|ru|jaccard|in_branch``, ``;``-joined), but
    the members come from :func:`topn_union_rows` — at most ``2*k``
    records, fewer when the two windows overlap — listed in
    :data:`_CHAIN` order so the chain-best hit stays first.  The chain's
    own top-N can hide a hit that is rank-1 on one metric while invisible
    on the other; the union shows both neighbourhoods without letting
    either metric alone define the window."""
    small = topn_union_rows(df, k)
    if small is None or small.empty:
        return ''
    pool = {int(b) for b in (branch_pool or [])}
    parts: List[str] = []
    for r in small.itertuples(index=False):
        bid = int(r.target_bid)
        ru = _clean_num(getattr(r, 'rank_union', None))
        jac = _clean_num(getattr(r, 'jaccard', None))
        parts.append('|'.join([
            str(_clean_int(getattr(r, 'rank_union_rank', None)) or ''),
            str(_clean_int(getattr(r, 'jaccard_rank', None)) or ''),
            str(bid),
            _clean_type((id2type or {}).get(bid)),
            '' if ru is None else f'{ru:.4f}',
            '' if jac is None else f'{jac:.4f}',
            '1' if bid in pool else '0',
        ]))
    return ';'.join(parts)


def _own_type_hit(usable: pd.DataFrame, id2type: Optional[Dict],
                  branch_source_type: str, top_k: int = 3) -> Optional[Dict]:
    """The best-ranked hit of the claiming branch's OWN source type, and
    which ranking put it there.

    Scans the top-``top_k`` window of BOTH rank columns and keeps the
    strongest same-type hit; ties prefer ``jaccard`` (the leading metric,
    :data:`_RANK_COLS`), so the result is deterministic.  Returns ``None``
    when no same-type hit ranks at all.  The evidence grade reads straight
    off it — ``high`` = rank 1, ``medium`` = rank 2..top_k, ``low`` =
    ``None`` — and a same-type hit counts wherever it lives: pool membership
    is context (``backward_top1_in_branch``), never a verdict, and there is
    NO score bar (user 2026-09-19).  Publishing the hit is what lets a reader
    see WHY a member reads ``high`` even when the globally best reverse hit
    is of another type (plan-tmvev-reciprocal-jaccard-sort-and-parity D1c)."""
    if not branch_source_type or usable is None or usable.empty:
        return None
    best = None                       # (rank, tie_pref, col, row)
    for tie, col in enumerate(_RANK_COLS):        # the two rank columns
        sub = usable[usable[col].notna()]
        if sub.empty:
            continue
        # POL-2: window on CHAIN POSITIONS, not the rank column — a
        # method='min' tie block at rank 1 with 4+ members kept only the
        # 3 chain-first rows under nsmallest, hiding a same-type member
        # sitting 4th inside the tie (the class classify_backward_scan
        # already fixed for backward_n_out_of_branch, D-B).
        ordered = sub.sort_values(col, kind='mergesort')
        in_top = ordered[col].rank(method='min') <= top_k
        for r in ordered[in_top].itertuples(index=False):
            try:
                bid = int(getattr(r, 'target_bid'))
            except (TypeError, ValueError):
                continue
            if (id2type or {}).get(bid) != branch_source_type:
                continue
            rk = _clean_num(getattr(r, col))
            if rk is None:
                continue
            if best is None or rk < best[0]:
                best = (rk, tie, col, r)
    if best is None:
        return None
    rk, _tie, col, row = best
    return {'rank': rk,
            'via': 'rank_union' if col == 'rank_union_rank' else 'jaccard',
            'row': row}


def classify_backward_scan(df: Optional[pd.DataFrame], *,
                           branch_pool=None, id2type=None,
                           sizes=None, pool_best_size: float = 0.0,
                           branch_source_type: str = '',
                           min_size_ratio: float = 0.1,
                           top_n: int = 5) -> Dict:
    """One reverse scan's grade for the scanned (target-dataset) neuron.

    The homolog-finding question, asked from the other side: "which source
    does this target prefer, and how prominently does its OWN branch's
    source type rank there?"  The grade is pure rank evidence — no score
    bar, no pool-membership gate (user 2026-09-19):

    * ``high``   — a hit of the branch's own source type is the top-1 by
      rank_union or by jaccard.
    * ``medium`` — such a hit sits within the top-3 of either ranking.
    * ``low``    — scanned, but the branch's source type ranked outside
      both top-3 windows (or nothing usable ranked at all).

    Pool membership and the spatial caliber are recorded as CONTEXT on the
    row (``backward_top1_in_branch``, ``backward_size_ratio`` /
    ``backward_size_filtered``) — advisory hints that never change the
    grade.  Alongside it the row publishes
    ``backward_shared_type_count`` / ``backward_union_type_count`` and a
    ``backward_thin_evidence`` flag (:data:`THIN_SHARED_TYPE_COUNT`) so a
    reader can see how much of the union a rank_union actually rests on —
    the flag never changes ``backward_evidence`` and never gates anything.

    The grade itself is also made legible: the winning hit from
    :func:`_own_type_hit` travels as the ``backward_own_type_*`` block (its
    bodyId, type, the ``_via`` ranking that surfaced it, both scores and
    both ranks, its own shared/union base + thin flag).  Without it a
    ``high`` whose displayed top-1 is of another type is an unexplained
    cell — and there is no such thing as an unexplained cell in an advisory
    report: ``low`` rows publish an empty block, so a grade is either
    backed by a visible hit or visibly has none.
    """
    out = blank_backward_fields('run')
    pool = {int(b) for b in (branch_pool or [])}
    if df is None or df.empty:
        out['backward_evidence'] = 'low'
        return out
    usable = _with_chain_pos(
        df[df['jaccard'].notna() | df['rank_union'].notna()])
    best = _best_row(usable)
    if best is None:
        out['backward_evidence'] = 'low'
        return out
    try:
        top1 = int(best['target_bid'])
    except (TypeError, ValueError):
        out['backward_evidence'] = 'low'
        return out
    ru = _clean_num(best.get('rank_union'))
    jac = _clean_num(best.get('jaccard'))
    shared = _clean_int(best.get('shared_type_count'))
    union = _clean_int(best.get('union_type_count'))
    in_branch = top1 in pool
    bid_size = _clean_num((sizes or {}).get(top1))
    pbs = _clean_num(pool_best_size) or 0.0
    ratio = None
    size_filtered = False
    if bid_size is not None and pbs > 0:
        ratio = bid_size / pbs
        size_filtered = ratio < float(min_size_ratio)
    # Out-of-branch competitors above the branch's OWN best-ranked source —
    # counted on CHAIN POSITIONS, not on a rank column (D-B): a rank tie block
    # collapses rivals onto the same rank as the pool member and understates
    # the count, and which rival is "above" would depend on row order.
    n_out = 0
    if pool:
        in_pool = usable['target_bid'].astype(int).isin(pool)
        pool_rows = usable[in_pool]
        own = _best_row(pool_rows)
        if own is not None:
            own_pos = _clean_int(own.get('chain_pos'))
            if own_pos is not None:
                n_out = int(len(usable[(usable['chain_pos'] < own_pos)
                                       & (~in_pool)]))
    # The grade is pure rank evidence: how prominently hits of the branch's
    # OWN source type rank.  Caliber and pool membership stay context —
    # advisory hints on the row, never a verdict (user 2026-09-19).  The
    # winning hit is published alongside it so a `high` that came from the
    # jaccard window is legible even though the displayed top-1 is the
    # rank_union winner (D1c).
    own = _own_type_hit(usable, id2type, branch_source_type)
    evidence = 'low' if own is None else (
        'high' if own['rank'] <= 1 else 'medium')
    own_fields: Dict = {}
    if own is not None:
        orow = own['row']
        try:
            own_bid: Optional[int] = int(getattr(orow, 'target_bid'))
        except (TypeError, ValueError):
            own_bid = None
        own_shared = _clean_int(getattr(orow, 'shared_type_count', None))
        own_fields = {
            'backward_own_type_rank_source_bodyId': own_bid,
            'backward_own_type_rank_source_type': _clean_type(
                (id2type or {}).get(own_bid) if own_bid is not None else None),
            'backward_own_type_via': own['via'],
            'backward_own_type_rank_union': _clean_num(
                getattr(orow, 'rank_union', None)),
            'backward_own_type_jaccard': _clean_num(
                getattr(orow, 'jaccard', None)),
            'backward_own_type_shared_type_count': own_shared,
            'backward_own_type_union_type_count': _clean_int(
                getattr(orow, 'union_type_count', None)),
            'backward_own_type_rank_union_rank': _clean_int(
                getattr(orow, 'rank_union_rank', None)),
            'backward_own_type_jaccard_rank': _clean_int(
                getattr(orow, 'jaccard_rank', None)),
            'backward_own_type_thin_evidence': bool(
                own_shared is not None
                and own_shared <= THIN_SHARED_TYPE_COUNT),
        }
    out.update({
        'backward_evidence': evidence,
        'backward_top1_source_bodyId': top1,
        'backward_top1_source_type': _clean_type(
            (id2type or {}).get(top1)),
        'backward_top1_in_branch': bool(in_branch),
        'backward_rank_union': ru,
        'backward_jaccard': jac,
        # how much evidence the numbers above rest on — the scorer counts
        # them anyway, publishing them is what lets a reader discount a
        # high rank_union built from 2 shared partners
        'backward_shared_type_count': shared,
        'backward_union_type_count': union,
        'backward_rank_union_rank': _clean_int(best.get('rank_union_rank')),
        'backward_jaccard_rank': _clean_int(best.get('jaccard_rank')),
        'backward_n_out_of_branch': n_out,
        'backward_size_ratio': ratio,
        'backward_size_filtered': bool(size_filtered),
        'backward_thin_evidence': bool(
            shared is not None and shared <= THIN_SHARED_TYPE_COUNT),
        'backward_topN': serialize_backward_topN(
            usable, id2type=id2type, top_n=top_n, branch_pool=pool),
        **own_fields,
    })
    return out


def reverse_source_column(df: Optional[pd.DataFrame], source_pool,
                          top_rows: int = 50) -> List[Dict]:
    """A pool target's column ranked over the WHOLE source universe.

    The true competitor set for ``categorize_pool_sources``: the forward scans
    can only ever see the branch's own sources, so out-of-branch rivals above a
    source are invisible there.  Returns entries in the shape
    ``{'source', 'ru', 'jac', 'in_pool'}`` ordered best-first in
    :data:`_CHAIN` order (Jaccard first; user 2026-09-20, D5), so a column and
    the neighbourhood printed for it read the same way.  The cut keeps every
    pool row regardless of position."""
    if df is None or df.empty:
        return []
    pool = {int(b) for b in (source_pool or [])}
    ranked = df.sort_values(_CHAIN, ascending=_CHAIN_ASC,
                            na_position='last')
    cut = int(top_rows)
    out: List[Dict] = []
    for r in ranked.itertuples(index=False):
        try:
            bid = int(r.target_bid)
        except (TypeError, ValueError):
            continue
        in_pool = bid in pool
        # Out-of-branch rivals past the cut are dropped, the branch's own pool
        # is not: a pool source ranked below it would otherwise lose its whole
        # column, which the caller reads as "no reverse evidence" rather than
        # "ranked low" -- and its competitor count would be wrong too.
        if len(out) >= cut and not in_pool:
            continue
        out.append({'source': bid,
                    'ru': _clean_num(getattr(r, 'rank_union', None)),
                    'jac': _clean_num(getattr(r, 'jaccard', None)),
                    'in_pool': in_pool})
    return out


# ---------------------------------------------------------------------------
# Evidence loaders (moved verbatim from mapping_validation)
# ---------------------------------------------------------------------------

def _column_key(name) -> str:
    """Lower-cased alphanumeric-only view of a table column name, so the
    datasets' own spellings fold together: ``size_nm``, ``Size (nm3)`` and
    ``Volume (nm^3)`` all reach the same gate (BANC v888 uses the last two;
    reading only ``size``/``size_nm`` silently disabled the spatial-caliber
    noise filter and the hemisphere trigger for that target)."""
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def load_caliber_map(dataset: str,
                     project_root=None) -> Dict[int, float]:
    """Spatial caliber per bodyId from the dataset's allneurons table.

    Primary noise-filter input (Rev 3.6: size/spatial arborization is a
    stronger, annotation-independent signal than connectivity heuristics).
    MCNS tables carry ``size``, FAFB ``size_nm``, BANC v888
    ``Volume (nm^3)`` — all nm³ and near-complete coverage (BANC: 162,643 of
    188,508 rows), so no skeleton fetch is needed.  Returns {} when the
    table is unavailable (the caliber gate then falls back to the
    expanded-weight ratio).
    """
    root = Path(project_root) if project_root else \
        Path(__file__).resolve().parents[2]
    try:
        from ..utils.naming_utils import canonical_dataset_name
    except ImportError:
        from utils.naming_utils import canonical_dataset_name
    # Canonicalize first: legacy aliases ('banc', 'flywire_BANC_v626', ...)
    # fold to their canonical folder name (review 2026-09-16).
    folder = canonical_dataset_name(str(dataset)).replace(
        ':', '_').replace('.', '_')
    base = root / 'datasets' / folder / f'{folder}_allneurons_neuron_df'
    try:
        if base.with_suffix('.parquet').exists():
            tdf = pd.read_parquet(base.with_suffix('.parquet'))
        elif base.with_suffix('.csv').exists():
            tdf = pd.read_csv(base.with_suffix('.csv'), low_memory=False)
        else:
            return {}
    except Exception:
        return {}
    keyed = {_column_key(c): c for c in tdf.columns}
    col = next((keyed[k] for k in ('sizenm', 'size', 'volumenm3', 'volume')
                if k in keyed), None)
    if col is None or 'bodyId' not in tdf.columns:
        return {}
    vals = pd.to_numeric(tdf[col], errors='coerce').fillna(0.0)
    return {int(b): float(v)
            for b, v in zip(tdf['bodyId'], vals) if pd.notna(b)}


def load_hemisphere_map(dataset: str,
                        project_root=None) -> Dict[int, str]:
    """bodyId -> 'L'/'R' ('?' when unknown) from the dataset's
    hemisphere/instance columns — the hemisphere-asymmetry trigger's
    side source.  MCNS/FAFB carry ``hemisphere``/``sides``/``side``;
    BANC v888 carries ``Soma side`` with 'left'/'right' (92% of rows).
    """
    root = Path(project_root) if project_root else \
        Path(__file__).resolve().parents[2]
    try:
        from ..utils.naming_utils import canonical_dataset_name
    except ImportError:
        from utils.naming_utils import canonical_dataset_name
    # Canonicalize first: legacy aliases ('banc', 'flywire_BANC_v626', ...)
    # fold to their canonical folder name (review 2026-09-16).
    folder = canonical_dataset_name(str(dataset)).replace(
        ':', '_').replace('.', '_')
    base = root / 'datasets' / folder / f'{folder}_allneurons_neuron_df'
    try:
        if base.with_suffix('.parquet').exists():
            tdf = pd.read_parquet(base.with_suffix('.parquet'))
        elif base.with_suffix('.csv').exists():
            tdf = pd.read_csv(base.with_suffix('.csv'), low_memory=False)
        else:
            return {}
    except Exception:
        return {}
    if 'bodyId' not in tdf.columns:
        return {}
    out: Dict[int, str] = {}
    keyed = {_column_key(c): c for c in tdf.columns}
    side_col = next((keyed[k] for k in ('hemisphere', 'hemispheres',
                                        'sides', 'side', 'somaside',
                                        'somahemisphere')
                     if k in keyed), None)
    # Columns are read BY LABEL, not through `itertuples` + `getattr`:
    # BANC v888 names the side column `Soma side`, which is not an
    # attribute name, so the attribute form could never see it (and a
    # renamed dataset column degrades to '?' silently).
    sides = ([str(v) for v in tdf[side_col].fillna('')]
             if side_col is not None else [''] * len(tdf))
    insts = ([str(v) for v in tdf['instance'].fillna('')]
             if 'instance' in tdf.columns else [''] * len(tdf))
    for bid, raw_side, inst in zip(tdf['bodyId'], sides, insts):
        try:
            bid = int(bid)
        except (TypeError, ValueError):
            continue
        v = raw_side.strip().upper()
        side = 'L' if v.startswith('L') else ('R' if v.startswith('R')
                                              else '?')
        if side == '?':
            if inst.endswith('_L'):
                side = 'L'
            elif inst.endswith('_R'):
                side = 'R'
        out[bid] = side
    return out


# ---------------------------------------------------------------------------
# The pool-scoped resolver (the mapper-facing capability)
# ---------------------------------------------------------------------------

class ProfilesUnavailable(RuntimeError):
    """A dataset's connection data is missing entirely.

    Raised instead of ANY degraded answer: bodyId-level assignment never
    silently falls back to a type-level result."""

    def __init__(self, dataset: str, reason: str = ''):
        self.dataset = dataset
        super().__init__(
            f'connectivity profiles unavailable for {dataset!r}: '
            f'{reason or "no connection cache"}')


@dataclass
class BodyIdResolverConfig:
    """Configuration for :class:`BodyIdResolver`.

    Profile construction mirrors ``MappingValidator``'s benchmark-frozen
    values so vectors match the existing caches and the validation parity
    holds.  The gates decide assignment confidence; every gate outcome is
    visible in ``AssignmentResult.flags`` / ``status_counts`` — never
    silently dropped.
    """
    # profile construction (benchmark-frozen: homolog_param_benchmark)
    top_k: int = 25
    top_m: int = 5
    min_synapse_threshold: int = 3
    include_untyped_partners: bool = True
    use_cache: bool = True
    # assignment gates
    min_score: float = 0.1         # absolute floor on the winning
                                   # rank_union; below -> unassigned
    min_margin: float = 0.02       # best minus second-best group; below
                                   # -> assigned best + 'low_confidence'
    require_positive: bool = True  # a non-positive ru win is ordering noise
    # same-hemisphere matching (plan R2.2 probe: the deciding evidence)
    # 'require': score only same-side members when the source side is
    #            known and some group offers them; groups whose same-side
    #            subset is empty are ineligible; if NO group offers a
    #            same-side candidate, fall back to full pools with the
    #            'side_fallback' flag.  Unknown sides never constrain
    #            ('side_unknown').
    # 'prefer':  a group is scored on its same-side members when it has
    #            them, else on its full pool.
    # 'off':     all members, sides ignored.
    side_matching: str = 'require'
    # optional spatial-caliber gate (validation's primary noise rule):
    # when True, a group's pool members below 0.1 x the group pool's best
    # caliber are excluded from scoring (no-op when the dataset table has
    # no size data).
    use_caliber: bool = False


@dataclass
class AssignmentResult:
    """Contract for the merge-integration phase (plan §6 — do not break).

    Exactly-one semantics: every input bodyId appears in exactly one of
    ``assignments`` (keyed by group id) or ``unassigned`` — never two.
    """
    assignments: Dict[int, str] = field(default_factory=dict)
    scores: Dict[int, Dict[str, Dict]] = field(default_factory=dict)
    unassigned: List[int] = field(default_factory=list)
    flags: Dict[int, List[str]] = field(default_factory=dict)
    group_stats: Dict[str, Dict] = field(default_factory=dict)
    status_counts: Dict[str, int] = field(default_factory=dict)

    def summary_line(self, groups: Dict[str, Dict[str, str]]) -> str:
        """One-line human summary, e.g.
        '5th-LNv ← 2 · LNd_CRY+_ITP+ ← 2 (0 unassigned · 2 low-confidence)'."""
        assigned_by_group = Counter(self.assignments.values())
        parts = ' · '.join(
            f'{gid} ← {assigned_by_group[gid]}'
            for gid in sorted(groups) if assigned_by_group.get(gid))
        n_low = sum(1 for fl in self.flags.values() if 'low_confidence' in fl)
        tail = (f' ({len(self.unassigned)} unassigned'
                + (f' · {n_low} low-confidence' if n_low else '') + ')')
        return (parts + tail) if parts else f'all unassigned{tail}'


class BodyIdResolver:
    """Pool-scoped bodyId-level resolution of 1-to-N type mapping splits.

    The resolver composes the type mapper (partner-type standardization)
    with the connectivity profiler (on-demand per-neuron profiles) and the
    production scorer.  It is intentionally pool-scoped: profiles are
    built only for the neurons being assigned and the branch pools being
    scored — seconds for the motivating cases, no bulk dataset profiling.
    """

    def __init__(self,
                 mapper: Optional['CrossDatasetTypeMapper'] = None,
                 profiler: Optional[ConnectivityProfiler] = None,
                 config: Optional[BodyIdResolverConfig] = None):
        self.mapper = mapper
        self._profiler = profiler
        self.config = config or BodyIdResolverConfig()
        self._side_maps: Dict[str, Dict[int, str]] = {}
        self._caliber_maps: Dict[str, Dict[int, float]] = {}
        if self.config.side_matching not in ('require', 'prefer', 'off'):
            raise ValueError(
                f"side_matching must be 'require', 'prefer' or 'off', "
                f"got {self.config.side_matching!r}")

    # -- services ----------------------------------------------------------

    @property
    def profiler(self) -> ConnectivityProfiler:
        """Lazy benchmark-frozen profiler (built on first use only)."""
        if self._profiler is None:
            self._profiler = ConnectivityProfiler(
                datasets=[], verbose=False,
                config=ProfilerConfig(
                    top_k_bodyid=self.config.top_k,
                    top_m_type=self.config.top_m,
                    min_synapse_threshold=self.config.min_synapse_threshold,
                    include_untyped_partners=(
                        self.config.include_untyped_partners),
                    use_cache=self.config.use_cache))
        return self._profiler

    def type_bodyid_pool(self, type_name: str, dataset: str) -> List[int]:
        """The resolved bodyId pool of one type (offline neuron tables).

        Sorted, de-duplicated Python ints; [] on lookup failure (callers
        treat an empty pool as 'type not present', mirroring the
        validation pipeline's ``_bodyids_for``)."""
        try:
            ids = self.profiler.get_bodyids_for_type(type_name, dataset) or []
        except Exception:
            return []
        return sorted({int(b) for b in ids})

    def neuron_profile(self, body_id: int,
                       dataset: str) -> ConnectivityProfile:
        """On-demand per-neuron profile (per-neuron caches apply).

        Raises :class:`ProfilesUnavailable` ONLY when the dataset's
        connection data is missing entirely — never a degraded answer."""
        try:
            return self.profiler.get_profile(int(body_id), dataset)
        except DataNotAvailableError as exc:
            raise ProfilesUnavailable(dataset, str(exc)) from exc

    def score_pair(self, src_vec: Dict[str, float],
                   tgt_vec: Dict[str, float]) -> Optional[Dict]:
        """The production per-pair scorer (exposed for parity checks)."""
        return score_one_candidate_fast(_SideStats(src_vec),
                                        _SideStats(tgt_vec))

    # -- internals -----------------------------------------------------------

    def _side_map(self, dataset: str) -> Dict[int, str]:
        smap = self._side_maps.get(dataset)
        if smap is None:
            smap = load_hemisphere_map(dataset)
            self._side_maps[dataset] = smap
        return smap

    def _caliber_map(self, dataset: str) -> Dict[int, float]:
        cmap = self._caliber_maps.get(dataset)
        if cmap is None:
            cmap = load_caliber_map(dataset)
            self._caliber_maps[dataset] = cmap
        return cmap

    # -- THE capability ------------------------------------------------------

    def assign_bodyids(self, body_ids, source_ds: str,
                       groups: Dict[str, Dict[str, str]], *,
                       pools: Optional[Dict[str, Dict[str, List[int]]]] = None
                       ) -> AssignmentResult:
        """Assign each source bodyId to exactly one branch — or flag it.

        ``body_ids``: the RESOLVED pool members of the conflicted parent
        type in ``source_ds``.  ``groups``: ordered
        ``{group_id: {target_ds: type_name}}`` — the N branches; every
        ``{target_ds: type}`` entry contributes that dataset's pool to the
        branch.  ``pools``: optional explicit
        ``{group_id: {target_ds: [ids]}}`` branch pools for callers that
        already refined them (linker-refined validation pools); explicit
        pools win over derivation.

        Branch score = max rank_union over the branch's eligible members
        (a branch asserts its best-matching member).  Decision: argmax by
        rank_union, jaccard tiebreak, group-id sort breaks remaining
        ties.  Gates (see :class:`BodyIdResolverConfig`): min_score floor,
        min_margin confidence, positivity, side matching.
        """
        cfg = self.config
        group_ids = sorted(groups)
        if not group_ids:
            raise ValueError('groups must contain at least one branch')

        # 1. normalize / derive the branch pools
        pool_map: Dict[str, Dict[str, List[int]]] = {}
        for gid in group_ids:
            entry = groups.get(gid) or {}
            explicit = (pools or {}).get(gid) or {}
            per_ds: Dict[str, List[int]] = {}
            for ds in sorted(entry):
                ids = explicit.get(ds)
                if ids is None:
                    ids = self.type_bodyid_pool(entry[ds], ds)
                per_ds[ds] = sorted({int(b) for b in (ids or [])})
            pool_map[gid] = per_ds

        # 2. optional caliber prefilter (validation's primary noise rule)
        caliber_dropped: Dict[str, int] = {}
        if cfg.use_caliber:
            for gid in group_ids:
                for ds in list(pool_map[gid]):
                    cmap = self._caliber_map(ds)
                    if not cmap:
                        continue
                    members = pool_map[gid][ds]
                    # POL-3: a bodyId ABSENT from the allneurons table
                    # (19% of both BANC tables read zero/NaN volume, which
                    # loads as 0.0) is not a TINY bodyId — the validation
                    # gate treats a missing size as 'no measurement' and
                    # keeps the member; the prefilter silently dropped
                    # ~19% of BANC branch members as if they were
                    # fragments.
                    sizes = [cmap.get(b) for b in members]
                    known = [s for s in sizes if s is not None]
                    best = max(known) if known else 0.0
                    if best <= 0:
                        continue
                    kept = [b for b, s in zip(members, sizes)
                            if s is None or s >= 0.1 * best]
                    dropped = len(members) - len(kept)
                    if dropped:
                        caliber_dropped[gid] = caliber_dropped.get(gid, 0) \
                            + dropped
                        pool_map[gid][ds] = kept

        result = AssignmentResult()
        for gid in group_ids:
            members_all = [(ds, b) for ds in sorted(pool_map[gid])
                           for b in pool_map[gid][ds]]
            side_cov: Counter = Counter()
            if cfg.side_matching != 'off':
                for ds, b in members_all:
                    side_cov[self._side_map(ds).get(b, '?')] += 1
            else:
                side_cov = Counter({'?': len(members_all)})
            result.group_stats[gid] = {
                'pool': {ds: list(pool_map[gid][ds])
                         for ds in sorted(pool_map[gid])},
                'n_scored': 0,
                'score_distribution': {},
                'side_coverage': dict(side_cov),
            }
            if caliber_dropped.get(gid):
                result.group_stats[gid]['caliber_filtered'] = \
                    caliber_dropped[gid]

        src_side_map: Dict[int, str] = {}
        if cfg.side_matching != 'off':
            src_side_map = self._side_map(source_ds)

        # per-call target profile cache: (ds, bid) -> ConnectivityProfile
        tgt_profiles: Dict[Tuple[str, int], Optional[ConnectivityProfile]] = {}

        def _target_profile(ds: str, tb: int) -> Optional[ConnectivityProfile]:
            key = (ds, tb)
            if key not in tgt_profiles:
                try:
                    tgt_profiles[key] = self.neuron_profile(tb, ds)
                except ProfilesUnavailable:
                    tgt_profiles[key] = None
            return tgt_profiles[key]

        def _side_of(ds: str, b: int) -> str:
            return self._side_map(ds).get(b, '?')

        status: Counter = Counter()

        for bid in sorted({int(b) for b in body_ids}):
            flags: List[str] = []
            # 3. source profile (dataset-level failure aborts loudly)
            prof = self.neuron_profile(bid, source_ds)
            if not prof.upstream_partners and not prof.downstream_partners:
                flags.append('no_profile')
                result.unassigned.append(bid)
                result.flags[bid] = flags
                status['unassigned_no_profile'] += 1
                result.scores[bid] = {}
                continue
            src_vec = expanded_vector(prof, self.mapper)
            src_stats = _SideStats(src_vec)
            src_side = src_side_map.get(bid, '?') \
                if cfg.side_matching != 'off' else '?'

            # 4. per-group eligible members under the side policy
            def _members(gid: str) -> List[Tuple[str, int]]:
                return [(ds, b) for ds in sorted(pool_map[gid])
                        for b in pool_map[gid][ds]]

            if cfg.side_matching == 'off':
                eligible = {gid: _members(gid) for gid in group_ids}
            elif src_side == '?':
                flags.append('side_unknown')
                eligible = {gid: _members(gid) for gid in group_ids}
            else:
                same = {gid: [(ds, b) for (ds, b) in _members(gid)
                              if _side_of(ds, b) == src_side]
                        for gid in group_ids}
                if cfg.side_matching == 'require':
                    if any(same.values()):
                        eligible = same
                    else:
                        flags.append('side_fallback')
                        eligible = {gid: _members(gid) for gid in group_ids}
                else:  # 'prefer'
                    eligible = {gid: (same[gid] or _members(gid))
                                for gid in group_ids}

            # 5. score each group: best member carries the branch
            scored: Dict[str, Dict] = {}
            target_unavailable = False
            for gid in group_ids:
                best_m = None
                best_key = None
                used: List[int] = []
                best_ds = None
                for ds, tb in eligible[gid]:
                    tprof = _target_profile(ds, tb)
                    if tprof is None:
                        target_unavailable = True
                        continue
                    if not tprof.upstream_partners \
                            and not tprof.downstream_partners:
                        continue
                    tgt_stats = _SideStats(expanded_vector(tprof, self.mapper))
                    if not tgt_stats.vec:
                        # POL-7: an empty target side is 'no pool', not a
                        # scoreable 0 — the reference scorer returns None
                        # for it and so does the contract here.
                        continue
                    m = score_one_candidate_fast(src_stats, tgt_stats)
                    if m is None:
                        continue
                    used.append(tb)
                    ru = m['rank_union']
                    ru_key = -math.inf if pd.isna(ru) else float(ru)
                    key = (ru_key, m['jaccard'], -tb)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_m = m
                        best_ds = ds
                entry = dict(best_m) if best_m else {}
                entry.setdefault('branch_score', None)
                entry['branch_pool_used'] = used
                if best_m:
                    entry['branch_score'] = float(best_m['rank_union']) \
                        if not pd.isna(best_m['rank_union']) else None
                    entry['best_member'] = -int(best_key[2])
                    entry['best_member_dataset'] = best_ds
                elif target_unavailable:
                    entry['profiles_unavailable'] = True
                result.scores.setdefault(bid, {})[gid] = entry
                if best_m:
                    scored[gid] = entry
                    result.group_stats[gid]['n_scored'] += 1

            # 6. decision + gates
            if not scored:
                if target_unavailable and all(
                        (not pool_map[gid])
                        or result.scores[bid][gid].get('profiles_unavailable')
                        for gid in group_ids):
                    first_ds = next(iter(groups[group_ids[0]]))
                    raise ProfilesUnavailable(
                        first_ds, 'branch target datasets have no '
                        'connection data')
                flags.append('no_pool')
                status['unassigned_no_pool'] += 1
                result.unassigned.append(bid)
                result.flags[bid] = flags
                continue

            ordered = sorted(
                scored,
                key=lambda gid: (
                    -(scored[gid]['branch_score']
                      if scored[gid]['branch_score'] is not None
                      else -math.inf),
                    -scored[gid]['jaccard'],
                    gid))
            best_gid = ordered[0]
            best = scored[best_gid]
            best_ru = best['branch_score']

            if cfg.require_positive and (
                    best_ru is None or best_ru <= 0):
                flags.append('non_positive')
                status['unassigned_non_positive'] += 1
                result.unassigned.append(bid)
                result.flags[bid] = flags
                continue
            if best_ru is None or best_ru < cfg.min_score:
                flags.append('low_score')
                status['unassigned_low_score'] += 1
                result.unassigned.append(bid)
                result.flags[bid] = flags
                continue

            result.assignments[bid] = best_gid
            if len(ordered) > 1:
                second_ru = scored[ordered[1]]['branch_score']
                margin = best_ru - (second_ru or 0.0)
                if margin < cfg.min_margin:
                    flags.append('low_confidence')
                    status['assigned_low_confidence'] += 1
                else:
                    status['assigned'] += 1
            else:
                status['assigned'] += 1
            if flags:
                result.flags[bid] = flags
            else:
                result.flags[bid] = []

        # 7. group score distributions + status summary
        for gid in group_ids:
            vals = [s['branch_score'] for s in
                    (result.scores[b].get(gid) for b in result.scores)
                    if s and s.get('branch_score') is not None]
            if vals:
                vals.sort()
                result.group_stats[gid]['score_distribution'] = {
                    'min': vals[0],
                    'median': vals[len(vals) // 2],
                    'max': vals[-1],
                }
        result.status_counts = dict(status)
        if flags_any := sum(1 for fl in result.flags.values() if fl):
            result.status_counts['flagged'] = flags_any
        return result
