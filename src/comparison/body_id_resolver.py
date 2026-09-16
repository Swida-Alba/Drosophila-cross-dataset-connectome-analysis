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


def scan_source(src_vec: Dict[str, float],
                target_stats: Dict[int, _SideStats],
                target_bids: Optional[List[int]] = None) -> pd.DataFrame:
    """Score one source vector against every target; rank globally.

    Validation-only scan service (dataset scale).  Returns a DataFrame
    (target_bid, metrics, rank_union_rank, jaccard_rank) sorted by
    rank_union_rank.  Ranks are competition-style (ties share the better
    rank, NaN last).
    """
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
    if df.empty:
        return df
    # object dtype: mixed int/float frames upcast .iloc[0] rows to float64,
    # which silently corrupts FAFB bodyIds (> 2**53)
    df['target_bid'] = df['target_bid'].astype(object)
    df['rank_union_rank'] = df['rank_union'].rank(
        ascending=False, na_option='bottom', method='min')
    df['jaccard_rank'] = df['jaccard'].rank(
        ascending=False, na_option='bottom', method='min')
    df = df.sort_values('rank_union_rank', na_position='last')
    return df.reset_index(drop=True)


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
                      ) -> Dict[int, _SideStats]:
    return {bid: _SideStats(vec) for bid, vec in vectors.items()
            if vec}


# ---------------------------------------------------------------------------
# Evidence loaders (moved verbatim from mapping_validation)
# ---------------------------------------------------------------------------

def load_caliber_map(dataset: str,
                     project_root=None) -> Dict[int, float]:
    """Spatial caliber per bodyId from the dataset's allneurons table.

    Primary noise-filter input (Rev 3.6: size/spatial arborization is a
    stronger, annotation-independent signal than connectivity heuristics).
    MCNS tables carry ``size``; FAFB carries ``size_nm`` — both ~100%
    coverage, so no skeleton fetch is needed.  Returns {} when the table
    is unavailable (the caliber gate then falls back to the
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
    col = next((c for c in ('size_nm', 'size') if c in tdf.columns), None)
    if col is None or 'bodyId' not in tdf.columns:
        return {}
    vals = pd.to_numeric(tdf[col], errors='coerce').fillna(0.0)
    return {int(b): float(v)
            for b, v in zip(tdf['bodyId'], vals) if pd.notna(b)}


def load_hemisphere_map(dataset: str,
                        project_root=None) -> Dict[int, str]:
    """bodyId -> 'L'/'R' ('?' when unknown) from the dataset's
    hemisphere/instance columns — the hemisphere-asymmetry trigger's
    side source."""
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
    out: Dict[int, str] = {}
    side_col = next((c for c in tdf.columns
                     if str(c).lower() in ('hemisphere', 'sides',
                                           'side')), None)
    for r in tdf.itertuples(index=False):
        try:
            bid = int(r.bodyId)
        except (TypeError, ValueError):
            continue
        side = '?'
        if side_col is not None:
            v = str(getattr(r, side_col) or '').strip().upper()
            if v.startswith('L'):
                side = 'L'
            elif v.startswith('R'):
                side = 'R'
        if side == '?' and 'instance' in tdf.columns:
            inst = str(getattr(r, 'instance', '') or '')
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
                    sizes = [cmap.get(b, 0.0) for b in members]
                    best = max(sizes) if sizes else 0.0
                    if best <= 0:
                        continue
                    kept = [b for b in members
                            if cmap.get(b, 0.0) >= 0.1 * best]
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
