"""Type-mapping validation pipeline (source -> target).

Plans: ``_plan/plan-type-mapping-validation-pipeline.md`` (Revisions
1–3.5) and ``_plan/plan-mapping-validation-rev311-chain-aware-pool-widening.md``
(§2b, the **Revision 3.12 category model** — normative).  The technical
report §4 and the user guide mirror §2b.

Stages
------
1. Live type mapping: resolve each query (a type or a coarse category such
   as ``circadian_clock``) into validated (source type -> target type) pairs
   with both sides' full bodyId pools via ``CrossDatasetTypeMapper``.
   Fail-closed: ``conflict`` decisions are excluded and reported.
2. bodyId-level validation: for every source neuron, score it against ALL
   target-dataset neurons on standardized expanded-type vectors (exact
   production formulas: ``ProfileComparator._get_expanded_types_standardized``
   + the ``batch_compare_cross_dataset`` metric block), rank every target
   globally under ``rank_union`` and ``jaccard``, apply the tiered verdict
   rule, assign 1:1 partners by mutual-best greedy, and report every
   non-pool neuron ranked ahead of the mapped partners.
3. Category partition (Revision 3.12 — :func:`classify_category` +
   :meth:`MappingValidator.finalize_categories`): every in-scope target
   gets EXACTLY ONE category by one ordered first-match —
   ``matched``/``verified``/``borderline``/``unmatched`` (the branch's
   in-map targets) > ``sibling`` (in-map target of another branch of the
   query) > ``candidates`` (out-of-map, connectivity- AND morph-qualified;
   the restrictive fill) > ``family`` (out-map bodyIds of the branch's
   target type) > ``relative`` (candidate-type mates outside the map) >
   ``suspicious`` (the aggressive-only deep window).  A
   connectivity-qualified suspect failing the morph rule is OUT OF SCOPE
   (``in_scope=False``, ``morph_failed=True``) — kept in the CSVs for
   reconciliation with a connectivity-only homolog search, never rendered.
   Modes nest: ``restrictive ⊆ family ⊆ aggressive``
   (``validation_mode`` / :func:`normalize_mode`); a shared neuron keeps
   the same category across modes.
4. Gap fill: ``gap = min(|P_S|, |P_T|) - M``; the restrictive fill counts
   ``candidates`` only, the family fill adds ``family``+``relative``.  The
   query-level dedup (``gap_fill_dedup.csv``) is bodyId-unique with
   precedence tier > sibling > candidates > family > relative >
   suspicious.  Proposals only — the mapping is never rewritten.
5. Morphology verification: vector_v2 + NBLAST via
   ``morphology.enrich_homolog_results`` on the pairs that matter, with
   per-run self-calibration (AUC separating verified_strong from
   suspicious) and a guard rail: morph gates ``verified_strong`` only when
   ``AUC >= morph_auc_floor``.  :func:`morph_qualified` (rule v3, the
   shared bar engine in :mod:`comparison.morph_bars`) is shared with the
   scene so the CSV and the picture cannot disagree.
5b. Profile pre-flight: before stage 2 the run checks the target dataset's
   typed universe against the bodyId profile cache and builds the missing
   profiles through the profiler backend (cache-first, resumable;
   ``--skip-profile-build`` opts out).  Progress lands in
   ``pipeline_progress.jsonl``.
5c. Out-map expansion: every unpaired source neuron (the scene's
   ``out-map query`` branch) is scanned against the full target universe;
   its top ``out_map_top_k`` typed non-in-map targets are morph-checked
   against the run null bar and exported to ``out_map_expansion.csv``.
   The layered gap-fill report ``gap_fill_levels.csv`` is assembled after
   the category partition (``build_gap_fill_levels``).
6. Visualization lives in ``comparison.mapping_validation_visualize``.

Performance notes
-----------------
- The per-candidate scorer works on precomputed target statistics; the
  union Spearman is computed with a numpy average-tie rankdata + Pearson,
  which is mathematically identical to ``scipy.stats.spearmanr`` on the
  same union vectors (unit-tested to 1e-9).
- Cosine norms are support-invariant (a vector is zero outside its own
  keys, which the union always contains), so norms are precomputed once.
- bodyId integrity: FAFB bodyIds (~7.2e17) exceed float64's exact range —
  ids stay Python ints end-to-end; row iteration uses ``itertuples`` /
  column arrays, never ``iterrows``.
"""

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison.connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    ProfilerConfig,
)
from comparison.cross_dataset_type_mapper import get_type_mapper
from comparison.morph_bars import (
    BarSet,
    candidate_qualified,
    compute_branch_bars,
    suspicious_qualified,
)
from comparison.profile_comparator import ProfileComparator
from comparison.body_id_resolver import (  # noqa: E402
    BodyIdResolver,
    BodyIdResolverConfig,
    _SideStats,
    _pearson,
    _rankdata_average,
    build_target_vectors,
    expanded_vector,
    load_caliber_map,
    load_hemisphere_map,
    passes_target_quality_gate,
    prep_target_stats,
    scan_source,
    score_one_candidate,
    score_one_candidate_fast,
)

SOURCE_STATUS_SKIP = {'NONE', 'ORPHAN'}
WEAK_STATUSES = {'RARE', 'UNIDIRECTIONAL', 'INCOMPLETE',
                 'INCOMPLETE_EXPANSION'}
ASSIGN_VERDICTS = {'verified_strong', 'verified', 'borderline'}
# Per-source candidate rows kept for gap fill: top-N by each metric's rank
# plus every pool member.  Full ~137k-row scans are not retained.
FILL_KEEP_TOP = 100


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MappingValidationConfig:
    source_dataset: str
    target_dataset: str
    query_types: List[str] = field(default_factory=list)
    # profile construction (benchmark-frozen: homolog_param_benchmark)
    top_k: int = 25
    top_m: int = 5
    min_synapse_threshold: int = 3
    include_untyped_partners: bool = True
    # verdict / gap knobs
    rank_top_k: int = 5
    # Revision 3.4: gap > gap_min always (the v1 '> 20% or > 5' rule is
    # retired — the strict ladder makes every fired gap actionable)
    gap_min: int = 1
    suspicious_per_source_cap: int = 20
    # Revision 3.3 category ladder
    verified_top_n: int = 2
    invader_borderline_max: int = 3
    matched_ru_min: float = 0.1
    # Revision 3.5 Issue 3c (R5 review): 0.5 x avg pool morph (thresholds
    # 0.31-0.37) qualified no invader in R5 while pool averages sit at
    # 0.62-0.74 — loosen to 0.25 (thresholds ~0.16-0.18). The chosen rule
    # is recorded per run in morphology_calibration.json.
    candidate_morph_factor: float = 0.25  # candidates: morph >= factor x avg pool morph
    # Revision 3.5 Issue 5: target-side profile-quality gate — scan
    # candidates below these floors are excluded at target-vector build
    # time (fragments such as MCNS 778042180: 1 partner, weight 3, won
    # rank_union top-1 for three sources on ordering noise alone).
    target_min_weight: float = 10.0
    target_min_partner_types: int = 2
    # Revision 3.5 Issue 5b: a rank_union-ahead row whose jaccard is far
    # below the best pool member's jaccard is ordering noise, regardless
    # of the rank_union sign.
    suspicious_jaccard_factor: float = 0.5
    # Revision 3.6: spatial caliber is the PRIMARY noise filter —
    # candidate spatial size below a fraction of the branch pool's best
    # is annotation-independent noise (293154: 14 M nm³ vs 2.4-2.7 B
    # nm³ pool, ratio 0.005). Caliber from the neuron-table metadata
    # (MCNS `size`, FAFB `size_nm`); fallback to the expanded-weight
    # ratio when size is missing.
    target_min_size_ratio: float = 0.1
    # Revision 3.6 secondary ordering-noise rule: a rank_union-ahead row
    # with a margin this small over the best pool member is a numerical
    # tie (293154's second row: +0.0013).
    suspicious_ru_margin: float = 0.02
    # Revision 3.7 Track B (all-native pool reference)
    pool_ref_cap: int = 6            # reference neurons per invader
    pool_ref_floor_margin: float = 0.05  # floor = native baseline - margin
    # Floors v3 (plan-unified-morph-qualification-bars): the Track-A offset Δ
    # — candidate bar = B_b − Δ, suspicious bar = B_b − k·Δ on the branch's
    # scored matched+verified pool Track-A baseline — and the suspicious
    # level k.  The native margin above stays the native-currency knob.
    morph_track_a_offset: float = 0.05
    morph_suspicious_level: int = 3
    # Plan I §1: build missing target-side bodyId profiles at stage 2
    # through the profiler backend (cache-first, resumable).  False keeps
    # the historical cache-only behavior.
    skip_profile_build: bool = False
    # Plan I follow-up: scan the out-map (unpaired) source neurons against
    # the full target universe and keep the top-k connectivity-ranked
    # candidates per source — the evidence layer behind the scene's
    # `out-map` branch (connectivity-only; morph bars do not apply).
    out_map_top_k: int = 10
    skip_out_map_expansion: bool = False
    # Revision 3.12 category model.  `validation_mode` is the single
    # source of truth; it is an ordered enum
    # restrictive < family < aggressive, and the modes NEST (a neuron's
    # category is mode-independent; later modes only admit more neurons):
    #   restrictive (default) - tier + sibling + candidates; minimal.
    #   family                - adds the family/relative bins (out-map
    #                           bodyIds of the in-map types, and the
    #                           type-mates of candidate types).
    #   aggressive            - adds the deep-window `suspicious` bin.
    # `aggressive_expansion` and `pool_widen` are legacy boolean aliases
    # resolved by `normalize_mode` (pool_widen -> family).  Pool widening
    # itself is RETIRED (Revision 3.12): family mode no longer touches the
    # validated pool, so the tier is identical across modes.
    validation_mode: str = 'restrictive'
    aggressive_expansion: bool = False
    pool_widen: bool = False
    relatives_cap: int = 12
    candidate_window: int = 25
    deep_cap: int = 10
    # Rev 3.9 Track-A bar calibration: the query-based morph threshold
    # is NULL-CALIBRATED per run — the p95 of the Track-A morph over
    # window rows with jaccard <= `null_jaccard_max` (provably unrelated
    # by connectivity), replacing the arbitrary factor x pooled-average
    # bar when enough null samples exist (null_min_n).
    null_jaccard_max: float = 0.05
    null_per_source_cap: int = 5
    null_min_n: int = 10
    null_percentile: float = 95.0
    # Revision 3.8: the gap trigger is retired as a gate — expansion is
    # unconditional (annotation errors/incompleteness are expected).
    # gap_min is kept for the informational gap stats only.
    # stage 5
    morph_enabled: bool = True
    morph_auc_floor: float = 0.65
    candidate_morph_cap: int = 20
    # stage 4
    visualize: bool = True
    max_scenes: int = 12
    neuron_alpha: float = 0.2   # global neuron opacity (backend default)
    # Revision 3.5 Issue 6c: debug self-check — after rendering, verify
    # each legend leaf's geometry bbox matches its labeled neuron's bbox.
    scene_selfcheck: bool = False
    # plumbing
    output_dir: Optional[str] = None
    run_label: str = 'run'
    use_cache: bool = True
    verbose: bool = True

    @property
    def effective_mode(self) -> str:
        """The resolved ordered validation mode (Revision 3.12).

        A single enum, not two flags: the legacy booleans are resolved to
        the most permissive mode they imply, so the reported mode always
        matches the behavior actually applied.
        """
        return normalize_mode(self.validation_mode,
                              aggressive_expansion=self.aggressive_expansion,
                              pool_widen=self.pool_widen)

    @property
    def mode_rank(self) -> int:
        return MODE_RANK[self.effective_mode]

    def mode_at_least(self, mode: str) -> bool:
        return self.mode_rank >= MODE_RANK[str(mode).lower()]


@dataclass
class TypePair:
    source_dataset: str
    source_type: str
    source_pool: List[int]
    target_dataset: str
    target_type: str
    target_pool: List[int]
    relationship: str = ''
    status: str = 'mapped'
    query: str = ''
    # Revision 2: bridge-resolved branch context
    selected_chain: List[Dict[str, str]] = field(default_factory=list)
    linkers: List[Dict[str, str]] = field(default_factory=list)
    pool_basis: str = 'full population'
    target_pool_basis: str = 'full population'
    source_type_total: int = 0
    target_type_total: int = 0
    parent_source_pool: List[int] = field(default_factory=list)
    parent_target_pool: List[int] = field(default_factory=list)
    branches_disjoint: Optional[bool] = None
    branch_annotation: str = ''
    branch_index: int = 0
    # Revision 3.11 chain-aware widening: alternative chains of the same
    # parent that reach the same target type contribute their members to
    # the branch pool (the linker-refined subset is a prioritization
    # artifact, not a membership boundary).
    pool_widen_added_sources: List[int] = field(default_factory=list)
    pool_widen_added_targets: List[int] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source_type, self.target_type)

    @property
    def linker_values(self) -> str:
        return '; '.join(str(l.get('raw_value', l.get('value', '')))
                         for l in self.linkers)

    @property
    def chain_text(self) -> str:
        return ' -> '.join(
            f"{h.get('dataset')}:{h.get('column')}={h.get('value')}"
            for h in self.selected_chain) if self.selected_chain else ''


def measure_branch_disjointness(pairs: List["TypePair"]) -> Optional[bool]:
    """Pairwise disjointness of the linker-refined source sub-pools across
    the branches of one parent source type.

    Returns None when it cannot be measured (single branch, or any branch
    falls back to the full population — the overlap is then meaningless).
    """
    refined = [p for p in pairs if p.pool_basis == 'linker rows']
    if len(pairs) < 2 or len(refined) != len(pairs):
        return None
    sets = [set(p.source_pool) for p in refined]
    flat = [b for s in sets for b in s]
    return len(set(flat)) == len(flat)


def annotate_pair_branches(pairs: List["TypePair"]) -> None:
    """Attach branch_index / branches_disjoint / branch_annotation to the
    branch pairs of one parent source type (Revision 2 vocabulary; see
    plan-type-mapper-fine-granularity-export.md §2)."""
    if not pairs:
        return
    for i, p in enumerate(sorted(
            pairs, key=lambda p: (-len(p.source_pool), p.target_type)), 1):
        p.branch_index = i
    disjoint = measure_branch_disjointness(pairs)
    targets = {p.target_type for p in pairs}
    all_linkered = all(p.pool_basis == 'linker rows' for p in pairs)
    terms = []
    if len(targets) > 1:
        terms.append('bifurcation')
        terms.append('resolved_by_linkers'
                     if (disjoint is True and all_linkered)
                     else 'unresolved_fanout')
    if disjoint is not None:
        terms.append('branches_disjoint' if disjoint
                     else 'branches_overlap')
    for p in pairs:
        p.branches_disjoint = disjoint
        p.branch_annotation = ';'.join(terms)


def compute_set_coverage(pairs: List["TypePair"],
                         per_pair_res: Dict,
                         fills: List[Dict],
                         evidence_rows: Optional[List[Dict]] = None) -> Dict:
    """Revision 3.10: set-level coverage of the source->target mapping.

    Per-branch gaps are diagnostics (they double-count cross-branch
    convergence); this answers the deliverable question directly:

    - FAFB side: of the queried source population
      (union of parent_source_pool), how many neurons are assigned
      (verified/borderline with a target), fill-proposed, or still
      unpaired — per type and in total.
    - MCNS side: of the mapped target set (union of
      parent_target_pool), how many neurons sit in a branch pool with
      which best category, how many are reached ONLY as out-of-pool
      candidates/proposals, and which are never claimed anywhere
      ('holes' — annotated-type neurons lost to linker refinement).
      "Reached" covers BOTH proposal routes: gap-fill rows and the
      invader-surfaced rows `finalize_categories` labeled
      ``candidates`` (pass the post-finalization sus/deep rows as
      ``evidence_rows``) — a bodyId the run claimed as a candidate is
      not a hole.
    """
    assigned_src: set = set()
    src_type_of: Dict[int, str] = {}
    assigned_tgt: set = set()
    pool_cat: Dict[int, str] = {}
    pool_typed: set = set()
    parent_src: Dict[str, set] = {}
    parent_tgt: Dict[str, set] = {}
    branch_pairs: set = set()

    for pair in pairs:
        parent_src.setdefault(pair.source_type, set()).update(
            int(b) for b in pair.parent_source_pool)
        parent_tgt.setdefault(pair.target_type, set()).update(
            int(b) for b in pair.parent_target_pool)
        res = per_pair_res.get(pair.key)
        if not res:
            continue
        for sid, tid in res.get('pairs', []):
            assigned_src.add(int(sid))
            assigned_tgt.add(int(tid))
            src_type_of[int(sid)] = pair.source_type
        for tbid, cat in (res.get('target_categories') or {}).items():
            bid = int(tbid)
            prev = pool_cat.get(bid)
            rank = {'matched': 0, 'verified': 1, 'borderline': 2,
                    'unmatched': 3}
            if prev is None or rank.get(cat, 9) < rank.get(prev, 9):
                pool_cat[bid] = cat
            pool_typed.add(bid)
        branch_pairs.add(pair.key)

    proposed_src: Dict[int, Dict] = {}
    reached_tgt: Dict[int, str] = {}
    for f in fills:
        if f.get('side') != 'source' or not f.get(
                'counts_toward_gap_fill', True):
            continue
        bid = int(f['bodyId'])
        proposed_src.setdefault(bid, {'count': 0})['count'] += 1
        src_type_of.setdefault(bid, str(f['source_type']))
        pbid = int(f['proposal_bodyId'])
        if pbid not in pool_cat:
            reached_tgt[pbid] = str(f.get('proposal_type') or '?')
    for r in evidence_rows or []:
        # Invader-surfaced candidates: the run's other claim route on a
        # mapped-population bodyId (morph-qualified, restrictive fill).
        if str(r.get('category') or '') != 'candidates' \
                or not r.get('in_scope', True):
            continue
        pbid = int(r['ahead_target_bodyId'])
        if pbid not in pool_cat:
            reached_tgt.setdefault(
                pbid, str(r.get('ahead_target_type') or '?'))

    fafb_types = {}
    for ptype, members in sorted(parent_src.items()):
        a = sum(1 for b in members if b in assigned_src)
        p = sum(1 for b in members
                if b not in assigned_src and b in proposed_src)
        fafb_types[ptype] = {
            'pool': len(members), 'assigned': a,
            'fill_proposed': p, 'unpaired_unproposed': len(members) - a - p}
    mcns_types = {}
    for ttype, members in sorted(parent_tgt.items()):
        in_pool = {b: pool_cat[b] for b in members if b in pool_cat}
        reached = {b for b in members if b in reached_tgt}
        holes = sorted(b for b in members
                       if b not in pool_cat and b not in reached)
        mcns_types[ttype] = {
            'mapped_population': len(members),
            'in_pool': len(in_pool),
            'in_pool_matched': sum(1 for c in in_pool.values()
                                   if c == 'matched'),
            'in_pool_verified': sum(1 for c in in_pool.values()
                                    if c == 'verified'),
            'in_pool_borderline': sum(1 for c in in_pool.values()
                                      if c == 'borderline'),
            'in_pool_unmatched': sum(1 for c in in_pool.values()
                                     if c == 'unmatched'),
            'reached_as_candidates_only': len(reached - set(in_pool)),
            'holes': len(holes),
            'hole_body_ids': holes}
    all_src = set().union(*parent_src.values()) if parent_src else set()
    all_tgt = set().union(*parent_tgt.values()) if parent_tgt else set()
    return {
        'fafb': {
            'total_queried': len(all_src),
            'assigned': len(assigned_src),
            'fill_proposed_only': len(set(proposed_src) - assigned_src),
            'unpaired_unproposed': len(
                all_src - assigned_src - set(proposed_src)),
            'per_type': fafb_types,
        },
        'mcns': {
            'mapped_target_set': len(all_tgt),
            'in_branch_pool': len(pool_cat),
            # family material (plan-same-name-fidelity-and-three-level-
            # coverage.md): in-map-TYPE bodyIds no branch pool claims —
            # the population overhang of the claim set (219 - 204 in the
            # circadian reference).  TM-EVE family/candidate material,
            # never a mapper failure.
            'family_material': sorted(all_tgt - set(pool_cat)),
            'reached_as_candidates_only': len(
                set(reached_tgt) - set(pool_cat)),
            'holes': len(all_tgt - set(pool_cat) - set(reached_tgt)),
            'per_type': mcns_types,
        },
        'branch_gap_note': ('per-branch gaps double-count cross-branch '
                            'convergence; the set-level numbers here '
                            'are the deliverable'),
    }


# ---------------------------------------------------------------------------
# Floors v3: the layered gap-fill report (plan-tmvev-profiler-integration-...)
# ---------------------------------------------------------------------------

GAP_FILL_LEVEL_BY_CATEGORY = {
    'candidates': None,   # resolved per row bar_kind -> high/medium/low
    'family': 'type_gated',
    'relative': 'advice',
}
BAR_KIND_TO_LEVEL = {'native': 'high', 'track_a_backup': 'medium',
                     'null': 'low'}
FILL_LEVELS = ('high', 'medium', 'low', 'type_gated', 'advice')


def build_gap_fill_levels(dedup_rows, evidence_rows, family_material=()):
    """The layered gap-fill report (floors v3): one row per non-tier
    bodyId in the query-level dedup, labeled with its confidence level.

    - ``high``   candidates passing the NATIVE m+v floor (bar_kind native)
    - ``medium`` candidates passing the Track-A backup floor (B_b - Δ)
    - ``low``    candidates passing only the run null bar
    - ``type_gated`` family (out-of-map instances of in-map types)
    - ``advice`` relative (candidate-type mates outside the map)

    ``family_material`` bodyIds are annotated as ``hole closer`` -- a
    candidate that closes a mapped-type hole is the highest-value fill.
    Tier rows (matched/...) are claims, not fills, and are not part of
    the report.  Returns ``(rows, counts)``.
    """
    fam_material = {int(b) for b in (family_material or [])}
    bars = {}
    for r in evidence_rows or []:
        bid = int(r.get('ahead_target_bodyId',
                        r.get('proposal_bodyId') or -1))
        bk = r.get('bar_kind')
        if bk and bid not in bars:
            bars[bid] = (bk, r.get('bar_value'))
    rows = []
    counts = {level: 0 for level in FILL_LEVELS}
    for d in dedup_rows or []:
        cat = str(d.get('dedup_category') or '')
        if cat not in GAP_FILL_LEVEL_BY_CATEGORY:
            continue          # tier rows are claims, not fills
        bid = int(d['target_bodyId'])
        if cat == 'candidates':
            kind, value = bars.get(bid, ('null', None))
            level = BAR_KIND_TO_LEVEL.get(kind, 'low')
            note = 'hole closer' if bid in fam_material else ''
            evidence, bar_value = kind, value
        else:
            level = GAP_FILL_LEVEL_BY_CATEGORY[cat]
            evidence = ('type_membership' if cat == 'family'
                        else 'candidate_type_mate')
            bar_value, note = None, ''
        counts[level] += 1
        rows.append({
            'level': level,
            'target_bodyId': bid,
            'target_type': d.get('target_type'),
            'dedup_category': cat,
            'evidence': evidence,
            'bar_value': bar_value,
            'dup': bool(d.get('dup')),
            'note': note,
        })
    rank = {level: i for i, level in enumerate(FILL_LEVELS)}
    rows.sort(key=lambda r: (rank[r['level']], r['target_bodyId']))
    return rows, counts


def compute_out_map_by_type(pairs):
    """{(query, source_type): sorted[bids]} — the UNCLAIMED source
    neurons: annotated bodyIds of each parent type that no branch pool
    claims (outside every refined ``source_pool`` — the pool-refinement
    residue).  These are the scene's ``out-map query`` branch: never
    scanned, never proposed, the part of the gap the expansion must
    explore.  In-pool unpaired sources are NOT here (they are claimed and
    already rendered as ``query``); skipped sources are in pools and also
    excluded naturally."""
    pools, claimed = {}, {}
    for pair in pairs:
        key = (pair.query, pair.source_type)
        pools.setdefault(key, set()).update(
            int(b) for b in pair.parent_source_pool)
        claimed.setdefault(key, set()).update(
            int(b) for b in pair.source_pool)
    out = {}
    for key, parent in pools.items():
        out[key] = sorted(parent - claimed.get(key, set()))
    return out


def load_source_type_counts(dataset: str,
                            project_root=None) -> Tuple[Dict[str, int],
                                                        Dict[str, int]]:
    """(type -> neuron count, additional_type token -> count) of the
    SOURCE dataset's table — the backward-home reality check (Rev 3.6):
    a mapper 'mapped' decision with a 0-count home (e.g. MCNS CB4091 →
    FAFB CB4091, 0 neurons) is hollow and must not earn a backward
    label.  Returns ({}, {}) when the table is unavailable.
    """
    root = Path(project_root) if project_root else \
        Path(__file__).resolve().parents[2]
    folder = str(dataset).replace(':', '_').replace('.', '_')
    base = root / 'datasets' / folder / f'{folder}_allneurons_neuron_df'
    try:
        if base.with_suffix('.parquet').exists():
            tdf = pd.read_parquet(base.with_suffix('.parquet'))
        elif base.with_suffix('.csv').exists():
            tdf = pd.read_csv(base.with_suffix('.csv'), low_memory=False)
        else:
            return {}, {}
    except Exception:
        return {}, {}
    type_counts: Dict[str, int] = {}
    if 'type' in tdf.columns:
        type_counts = tdf['type'].astype(str).value_counts().to_dict()
    add_counts: Dict[str, int] = {}
    add_col = next((c for c in tdf.columns
                    if c.lower().startswith('additional_type')), None)
    if add_col is not None:
        tokens: List[str] = []
        for v in tdf[add_col].astype(str):
            if v and v != 'nan':
                tokens.extend(t.strip() for t in v.split(',') if t.strip())
        add_counts = pd.Series(tokens).value_counts().to_dict()
    return type_counts, add_counts


def build_chain_index(pairs: List["TypePair"], mapper) -> Dict[
        str, Dict[str, Dict]]:
    """Per-parent bridge-chain index for collapsed-chain awareness
    (Rev 3.6): parent source type -> target type -> {'selected':
    selected-chain source linker value, 'alt': [non-selected linkers]}.

    An invader whose type is the final hop of a NON-selected chain of
    its own parent is the residue of a collapsed alternative chain
    (e.g. SMP223 via CB3612 under s-CPDN3D, where the CB3508 chain was
    selected) — mapping evidence, not an unexplained invader.
    """
    index: Dict[str, Dict[str, Dict]] = {}
    if mapper is None or not pairs:
        return index
    by_parent: Dict[str, List[TypePair]] = {}
    for p in pairs:
        by_parent.setdefault(p.source_type, []).append(p)
    for parent, plist in by_parent.items():
        src_ds, tgt_ds = plist[0].source_dataset, plist[0].target_dataset
        try:
            chains = mapper.get_type_bridges(parent, src_ds, tgt_ds,
                                             max_bridges=0) or []
        except Exception:
            continue
        selected: Dict[str, str] = {}
        for p in plist:
            sel = next((str(l.get('raw_value', l.get('value', '')))
                        for l in p.linkers
                        if l.get('home') == src_ds and l.get('raw_value')),
                       None)
            if sel:
                selected[p.target_type] = str(sel)
        entry = index.setdefault(parent, {})
        for chain in chains:
            hops = list(chain) if isinstance(chain, list) else []
            linker = next((str(h.get('value')) for h in hops
                           if str(h.get('column', '')).startswith(
                               'additional_type')), None)
            final = hops[-1].get('value') if hops else None
            if not linker or not final:
                continue
            rec = entry.setdefault(str(final),
                                   {'selected': selected.get(str(final)),
                                    'alt': []})
            if linker != rec['selected'] and \
                    linker not in rec['alt']:
                rec['alt'].append(linker)
    return index


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rank_or_inf(rank) -> float:
    return float(rank) if rank is not None and not pd.isna(rank) else np.inf


def _pair_sort_key(row) -> Tuple[float, float, float]:
    return (_rank_or_inf(row.get('rank_union_rank')),
            _rank_or_inf(row.get('jaccard_rank')),
            -(row['rank_union'] if row.get('rank_union') is not None
              and not pd.isna(row.get('rank_union')) else -9.0))


def _abbrev(dataset: str) -> str:
    return (dataset or 'ds').replace(':', '_').replace('.', '_')


def _write_csv(path: Path, rows: List[Dict]):
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.endswith('bodyId'):
            df[col] = df[col].astype('Int64')
    df.to_csv(path, index=False)


def _counter_text(key: str, rows: List[Dict]) -> str:
    return ', '.join(f'{k}={v}' for k, v in
                     Counter(r.get(key) for r in rows).most_common())


def _dedup_rows_by_bid(rows: List[Dict]) -> List[Dict]:
    """First row per (source_type, target_type, target bodyId).

    Used to union the enumerated ``family``/``relative`` members with the
    evidence rows carrying those categories without double-listing a bodyId
    in the same branch (Revision 3.12).
    """
    seen: set = set()
    out: List[Dict] = []
    for r in rows:
        key = (str(r.get('source_type')), str(r.get('target_type')),
               int(r.get('ahead_target_bodyId', -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def mutual_best_assignment(pool_set: set,
                           per_source: Dict[int, pd.DataFrame],
                           val_rows: Optional[List[Dict]] = None,
                           allowed_verdicts=None
                           ) -> List[Tuple[int, int]]:
    """Mutual-best greedy 1:1 within the mapped pool.

    Only sources with a confident verdict (verified_strong / verified /
    borderline by default — unmatched sources are NOT assigned; they flow
    into gap fill) participate.  Each side's best partner is decided by
    (rank_union rank, jaccard rank, rank_union score); a pair is assigned
    only when both sides agree.
    """
    allowed = allowed_verdicts or ASSIGN_VERDICTS
    verdict_by_src = {}
    for r in (val_rows or []):
        verdict_by_src[r['source_bodyId']] = r.get('verdict')

    def _pool_best(df: pd.DataFrame) -> Optional[pd.Series]:
        pool_df = df[df['target_bid'].isin(pool_set)]
        if pool_df.empty:
            return None
        return pool_df.sort_values(
            ['rank_union_rank', 'jaccard_rank', 'rank_union'],
            ascending=[True, True, False], na_position='last').iloc[0]

    src_best: Dict[int, pd.Series] = {}
    for sbid, df in per_source.items():
        if sbid in verdict_by_src and \
                verdict_by_src[sbid] not in allowed:
            continue
        row = _pool_best(df)
        if row is not None:
            src_best[sbid] = row
    tgt_best: Dict[int, Tuple[int, pd.Series]] = {}
    for sbid, row in src_best.items():
        t = int(row['target_bid'])
        cur = tgt_best.get(t)
        if cur is None or _pair_sort_key(row) < _pair_sort_key(cur[1]):
            tgt_best[t] = (sbid, row)
    assigned = []
    for t, (sbid, row) in tgt_best.items():
        if int(src_best[sbid]['target_bid']) == t:
            assigned.append((sbid, t))
    return assigned


def gap_check(n_src: int, n_tgt: int, matched: int,
              gap_min: int) -> Dict:
    smaller = min(n_src, n_tgt)
    gap = smaller - matched
    gr = (gap / smaller) if smaller else 0.0
    return {'gap': gap, 'gap_ratio': round(gr, 4),
            'gap_triggered': gap > gap_min}


def hemisphere_sides(pool: List[int], side_map: Dict[int, str]) -> Dict[str, int]:
    """Hemisphere group counts for a pool ('L'/'R', '?' when unknown)."""
    counts: Dict[str, int] = {'L': 0, 'R': 0, '?': 0}
    for b in pool:
        counts[side_map.get(int(b), '?')] = \
            counts.get(side_map.get(int(b), '?'), 0) + 1
    return counts


def hemisphere_gap_fired(pool_s: List[int], pool_t: List[int],
                         side_map_s: Dict[int, str],
                         side_map_t: Dict[int, str]) -> Tuple[bool, Dict]:
    """Hemisphere-asymmetry trigger (user 2026-09-12, corrected): every
    neuron has a hemisphere identity, so a pool should be L/R symmetric
    — an L != R imbalance in either pool fires the gap regardless of
    the arithmetic gap."""
    cs = hemisphere_sides(pool_s, side_map_s)
    ct = hemisphere_sides(pool_t, side_map_t)
    asymmetry = (cs.get('L', 0) != cs.get('R', 0)
                 or ct.get('L', 0) != ct.get('R', 0))
    return asymmetry, {'source_sides': cs, 'target_sides': ct,
                       'hemisphere_asymmetry': asymmetry}


POOL_CATEGORY_ORDER = ('matched', 'verified', 'borderline', 'unmatched')

# -- Revision 3.12: the branch/tier/sibling/candidates category model -------
# The categories are a partition (every in-scope target gets exactly one),
# decided by an ordered first-match.  See
# `_plan/plan-mapping-validation-rev311-chain-aware-pool-widening.md` §2b
# (S0–S8) and `docs/technical/...PIPELINE.md` §4 — those are normative.
VALIDATION_MODES = ('restrictive', 'family', 'aggressive')
MODE_RANK = {m: i for i, m in enumerate(VALIDATION_MODES)}

# Tier values (validated in-map targets) and expansion values.
TIER_CATEGORIES = ('matched', 'verified', 'borderline', 'unmatched')
EXPANSION_CATEGORIES = ('sibling', 'candidates', 'family', 'relative',
                        'suspicious')
# `paired_in_pool` is the category of a FILL row that pairs an unpaired
# neuron with an accepted pool member (either direction); it is a property
# of the fill table, not of a target neuron, so it never appears in the
# query-level dedup.
PAIRED_CATEGORY = 'paired_in_pool'
# Full display vocabulary (tier + expansion + the pairing category).
CATEGORY_VALUES = TIER_CATEGORIES + EXPANSION_CATEGORIES + (PAIRED_CATEGORY,)
# Tier strength for the query-level dedup rollup.
TIER_RANK = {c: i for i, c in enumerate(TIER_CATEGORIES)}
# Query-level dedup precedence (S6), higher wins:
# matched > verified > borderline > unmatched > sibling > candidates >
# family > relative > suspicious.  (Suspicious is the aggressive-only
# deep window, the lowest-confidence suggestion, so it rolls up last.)
DEDUP_RANK = {
    'matched': 9, 'verified': 8, 'borderline': 7, 'unmatched': 6,
    'sibling': 5, 'candidates': 4, 'family': 3, 'relative': 2,
    'suspicious': 1,
}


def normalize_mode(mode, aggressive_expansion: bool = False,
                   pool_widen: bool = False) -> str:
    """Resolve the single ordered validation mode (Revision 3.12).

    The mode string is the source of truth; the two legacy boolean flags
    are accepted as aliases and resolved to the MOST PERMISSIVE mode they
    imply (`pool_widen` -> family, `aggressive_expansion` -> aggressive),
    so a run can never be labelled family while aggressive behavior is
    active (the pre-3.12 precedence bug).
    """
    m = str(mode or 'restrictive').lower()
    if m not in MODE_RANK:
        m = 'restrictive'
    if pool_widen and MODE_RANK[m] < MODE_RANK['family']:
        m = 'family'
    if aggressive_expansion:
        m = 'aggressive'
    return m


def classify_category(*, target_bid, branch_pool, in_map, target_type,
                      branch_target_type, in_map_types, candidate_types,
                      connectivity_qualified, morph_ok, is_deep, tier,
                      mode, suspicious_morph_ok=None) -> Tuple[str, bool, bool]:
    """The per-branch partition (S2), one ordered first-match.

    Returns ``(category, in_scope, morph_failed)`` where:

    - ``category`` is one of :data:`CATEGORY_VALUES`, or ``''`` for an
      out-of-scope row (connectivity-qualified but morph-failed, or a
      cross-branch in-map target this branch has no evidence about);
    - ``in_scope`` is False for out-of-scope rows;
    - ``morph_failed`` is True for a row that failed the bar that would
      have admitted it: a connectivity-qualified suspect below the
      candidate bar, or (aggressive mode) a deep-window row below the
      suspicious bar.

    The order is part of the definition: tier -> sibling -> candidates ->
    family -> relative -> suspicious.  Each rule is evaluated on the
    complement of the earlier ones, and the last applicable rule is a
    residual, so the result is total and exclusive over the mode's scope.

    ``family`` is **branch-local** (user 2026-09-13): the out-map bodyIds
    whose type is THIS branch's target type.  The union over branches
    covers every out-map bodyId of the in-map types; a target type shared
    by several branches (N-to-1) repeats its family across them, which is
    what the query-level ``(dup)`` tag records.  ``relative`` is the
    type-mates of the branch's candidate types whose type is OUTSIDE the
    map, so family and relative are disjoint by type by construction.
    """
    b = int(target_bid)
    branch_pool = branch_pool or frozenset()
    in_map = in_map or frozenset()
    in_map_types = in_map_types or frozenset()
    candidate_types = candidate_types or frozenset()
    # 1. the validated in-map targets of THIS branch -> the tier.
    if b in branch_pool:
        return (tier or 'unmatched', True, False)
    # 2. an in-map target of the query in ANOTHER branch -> sibling
    #    (requires admission: connectivity-qualified AND morph-qualified).
    if b in in_map:
        if connectivity_qualified and morph_ok:
            return ('sibling', True, False)
        return ('', False, bool(connectivity_qualified and not morph_ok))
    # 3. an out-of-map suspect that clears the suspect bar -> candidates.
    if connectivity_qualified:
        if morph_ok:
            return ('candidates', True, False)
        return ('', False, True)
    # 4/5. family-mode residuals (ungated by qualification, gated by type
    #      membership).  family = THIS branch's target type; relative =
    #      candidate types outside the map.
    if mode in ('family', 'aggressive'):
        tt = None if target_type in (None, '', '?') else str(target_type)
        if tt is not None and branch_target_type is not None \
                and tt == str(branch_target_type):
            return ('family', True, False)
        if tt is not None and tt in candidate_types \
                and tt not in in_map_types:
            return ('relative', True, False)
    # 6. the aggressive-only deep window (floors v3 two-gate): a deep
    #    row is suspicious when it passes EITHER the qualified bar (it is
    #    then below-pool-best but fully morph-qualified) OR the loose
    #    suspicious bar; below both it stays out of scope, flagged as
    #    morph-failed (it failed the bar that would have admitted it).
    if is_deep and mode == 'aggressive':
        if morph_ok or suspicious_morph_ok:
            return ('suspicious', True, False)
        return ('', False, True)
    return ('', False, False)


def candidate_annotation(target_type, backward_types, has_type: bool,
                         home_real: bool = True,
                         type_in_map: bool = False) -> str:
    """The per-bodyId leaf token on an expansion leaf (S5).

    One token, ordered and mutually exclusive:

    - ``{target_type}(out-map)`` when the neuron's **type** is one of the
      mapping's in-map types — an unmapped bodyId of a type already in the
      map.  This is a **bodyId-level** out-of-map (the fill material):
      the type is mapped, this instance is not.
    - ``{target_type}>{src1_etc}`` when the type is NOT in-map but
      backward-maps with a real (non-hollow) source home.  This is a
      **type-level** statement: the neuron's type sits in another source's
      territory.
    - ``{target_type}(no_source)`` when the type is NOT in-map and has no
      usable backward route (including a hollow home — the mapper names a
      type whose source population is 0, e.g. CB4091).  A **type-level**
      out-of-map.
    - ``untyped`` when there is no type annotation at all.
    """
    if not has_type or target_type in (None, '', '?'):
        return 'untyped'
    t = str(target_type)
    if type_in_map:
        return f'{t}(out-map)'
    srcs = [str(s) for s in (backward_types or []) if s]
    if srcs and home_real:
        return f'{t}>{"_".join(srcs)}'
    return f'{t}(no_source)'


def dedup_category_rank(category: str) -> int:
    """Rollup rank for the query-level dedup (S6). Higher wins."""
    return DEDUP_RANK.get(str(category), 0)


def morph_qualified(row: Dict, bars) -> bool:
    """The morphology qualification rule v3 (floors v3), shared by the
    category classifier and the scene so they can never disagree.

    ``bars`` is the branch's
    :class:`~comparison.morph_bars.BarSet`: the native m+v floor is
    binding when the branch has >= 2 scored references (compares
    ``morph_pool_ref``); otherwise the Track-A backup floor ``B_b - Δ``
    (compares ``morph_v2_similarity``); otherwise the run null bar (also
    Track-A).  A branch with no basis admits nothing (fail-closed);
    morphology-disabled runs are treated as unqualified-but-admitted
    before this is called."""
    if bars is None:
        return False
    return candidate_qualified(bars, row.get('morph_pool_ref'),
                               row.get('morph_v2_similarity'))


def morph_qualified_suspicious(row: Dict, bars) -> bool:
    """The LOOSE aggressive-mode bar for the deep window: Track-A above
    ``B_b - k*Δ`` (or the run null p50 when the branch has no scored pool
    pairs).  Deep-window rows failing the candidate bar but passing this
    become ``suspicious``; below it they stay out of scope."""
    if bars is None:
        return False
    return suspicious_qualified(bars, row.get('morph_v2_similarity'))


def categorize_pool_targets(per_source: Dict[int, pd.DataFrame],
                            pool_set: set, top_n: int = 2,
                            invader_max: int = 3,
                            matched_ru_min: float = 0.1
                            ) -> Tuple[Dict[int, str], List[Dict]]:
    """Revision 3.3 ladder: categorize every in-pool neuron by its best
    evidence across the branch's sources (precedence top-down):

    matched    verified AND rank_union > matched_ru_min with its best source
    verified   top-1 of rank_union or top-1 of Jaccard for some source, OR
               continuous ordered top-N (default 2): the top-N of a
               source's ranking are ALL in-pool same-type neurons
    borderline not verified/matched, and at most ``invader_max``
               invading other-type neurons rank ahead of it
    unmatched  more invaders ahead (or no ranking at all)

    Returns ({target_bid: category}, per-target detail rows).
    """
    categories: Dict[int, str] = {}
    detail: List[Dict] = []
    verified: set = set()
    best: Dict[int, Dict] = {}
    for sbid, df in per_source.items():
        pool_df = df[df['target_bid'].isin(pool_set)]
        # top-1 of either metric (global rank 1, any sign)
        for metric in ('rank_union_rank', 'jaccard_rank'):
            for r in pool_df[pool_df[metric] == 1].itertuples(index=False):
                verified.add(int(r.target_bid))
        # continuous ordered top-N of either metric, all in-pool
        for metric in ('rank_union_rank', 'jaccard_rank'):
            heads = df[df[metric].notna()].nsmallest(top_n, metric)
            head_ids = [int(b) for b in heads['target_bid']]
            if len(head_ids) == top_n and all(
                    b in pool_set for b in head_ids):
                verified.update(head_ids)
        # best (lowest rank_union rank) row per pool target
        for tbid in pool_set:
            r = pool_df[pool_df['target_bid'] == tbid]
            if r.empty:
                continue
            row = r.sort_values('rank_union_rank',
                                na_position='last').iloc[0]
            cur = best.get(tbid)
            if cur is None or (row['rank_union_rank'] is not None
                               and not pd.isna(row['rank_union_rank'])
                               and (cur['ru_rank'] is None
                                    or pd.isna(cur['ru_rank'])
                                    or row['rank_union_rank']
                                    < cur['ru_rank'])):
                best[tbid] = {'source': sbid,
                              'ru': _f(row['rank_union']),
                              'ru_rank': _f(row['rank_union_rank']),
                              'ja_rank': _f(row['jaccard_rank'])}

    for tbid in sorted(pool_set):
        info = best.get(tbid) or {'source': None, 'ru': None,
                                  'ru_rank': None, 'ja_rank': None}
        if tbid in verified:
            ru = info.get('ru')
            if ru is not None and not pd.isna(ru) and ru > matched_ru_min:
                cat = 'matched'
            else:
                cat = 'verified'
        else:
            # invaders: non-pool neurons ranked ahead (best case over the
            # two metrics, across sources)
            inv_min = None
            for sbid, df in per_source.items():
                r = df[df['target_bid'] == tbid]
                if r.empty:
                    continue
                for metric in ('rank_union_rank', 'jaccard_rank'):
                    t_rank = r.iloc[0][metric]
                    if pd.isna(t_rank):
                        continue
                    inv = int(((~df['target_bid'].isin(pool_set))
                               & df[metric].notna()
                               & (df[metric] < t_rank)).sum())
                    inv_min = (inv if inv_min is None
                               else min(inv_min, inv))
            if inv_min is not None and inv_min <= invader_max:
                cat = 'borderline'
                info['invaders'] = inv_min
            else:
                cat = 'unmatched'
                info['invaders'] = inv_min
        categories[tbid] = cat
        detail.append({
            'target_bodyId': tbid,
            'category': cat,
            'best_source_bodyId': info.get('source'),
            'rank_union': info.get('ru'),
            'rank_union_rank': info.get('ru_rank'),
            'jaccard_rank': info.get('ja_rank'),
            'invaders_ahead': info.get('invaders'),
        })
    return categories, detail


def morph_auc(verified_scores: List[float],
              suspicious_scores: List[float]) -> Optional[float]:
    """P(verified morph score > suspicious morph score) via Mann-Whitney U."""
    from scipy.stats import mannwhitneyu
    v = [x for x in verified_scores if x is not None and not np.isnan(x)]
    s = [x for x in suspicious_scores if x is not None and not np.isnan(x)]
    if len(v) < 5 or len(s) < 5:
        return None
    try:
        res = mannwhitneyu(v, s, alternative='greater')
        return float(res.statistic) / (len(v) * len(s))
    except Exception:
        return None


def best_threshold(verified_scores: List[float],
                   suspicious_scores: List[float]) -> Optional[float]:
    """Threshold maximizing Youden J for 'verified above threshold'."""
    v = [x for x in verified_scores if x is not None and not np.isnan(x)]
    s = [x for x in suspicious_scores if x is not None and not np.isnan(x)]
    if not v or not s:
        return None
    pooled = sorted(v + s)
    cands = {(x + y) / 2 for x, y in zip(pooled, pooled[1:])}
    best_j, best_t = -1.0, None
    for t in cands:
        tpr = sum(x > t for x in v) / len(v)
        fpr = sum(x > t for x in s) / len(s)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t


# ---------------------------------------------------------------------------
# Backend progress monitor (Plan I §2: the contract a future UI tails)
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Run-scoped machine-readable progress stream.

    Appends one JSON object per event to ``pipeline_progress.jsonl`` in the
    run folder — the stable contract a future UI tails (Plan II).  The human
    log keeps its own lines; the JSONL is the machine truth.
    """

    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'pipeline_progress.jsonl'

    def emit(self, event, **fields):
        import json
        import time
        rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'event': event}
        rec.update(fields)
        try:
            with open(self.path, 'a') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MappingValidator:
    """Orchestrates the validation run and writes all outputs."""

    def __init__(self, cfg: MappingValidationConfig):
        self.cfg = cfg
        self.mapper = get_type_mapper()
        self.profiler = ConnectivityProfiler(
            datasets=[cfg.source_dataset, cfg.target_dataset],
            config=ProfilerConfig(
                top_k_bodyid=cfg.top_k, top_m_type=cfg.top_m,
                min_synapse_threshold=cfg.min_synapse_threshold,
                include_untyped_partners=cfg.include_untyped_partners,
                use_cache=cfg.use_cache),
            verbose=False)
        # Revision 3: the validator is a CONSUMER of the one integrated
        # bodyId backend — its benchmark profiler is injected so the run
        # holds exactly one profiler instance.
        self.resolver = BodyIdResolver(
            mapper=self.mapper, profiler=self.profiler,
            config=BodyIdResolverConfig(
                top_k=cfg.top_k, top_m=cfg.top_m,
                min_synapse_threshold=cfg.min_synapse_threshold,
                include_untyped_partners=cfg.include_untyped_partners,
                use_cache=cfg.use_cache))
        self.run_dir: Optional[Path] = None
        self.notes: List[str] = []
        self.pairs: List[TypePair] = []

    def log(self, msg=''):
        if self.cfg.verbose:
            print(msg, flush=True)
        self.notes.append(str(msg))

    # -- stage 1 ----------------------------------------------------------

    def _bodyids_for(self, name: str, dataset: str) -> List[int]:
        try:
            ids = self.resolver.type_bodyid_pool(name, dataset)
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! bodyId lookup failed for {name!r} in '
                     f'{dataset}: {exc}')
            return []
        return sorted(int(b) for b in ids)

    def resolve_type_pairs(self) -> List[TypePair]:
        """Expand queries (types or coarse categories) into branch pairs.

        Revision 2: each (source type -> target type) pair's pools are
        resolved through the bridge evidence
        (``ui.neuron_index.resolve_prioritized_bridge_pool``): the source
        pool is the selected chain's linker-refined subset, falling back
        to the full endpoint populations when no supported chain exists
        (same-name passes; basis flagged 'full population').  Branch
        pairs of one parent source type are then annotated with their
        disjointness and the shared plan vocabulary.
        """
        cfg = self.cfg
        pairs: List[TypePair] = []
        for query in cfg.query_types:
            query = query.strip()
            if not query:
                continue
            body_ids = self._bodyids_for(query, cfg.source_dataset)
            if body_ids:
                # A live pool means the query resolved (type, instance, or
                # coarse cell_type).  One pair per concrete type keeps
                # validation type-level.
                id2type = self.profiler.get_types_for_bodyids(
                    body_ids, cfg.source_dataset) or {}
                by_type: Dict[str, List[int]] = {}
                for bid in body_ids:
                    tname = id2type.get(bid)
                    if tname is None:
                        tname = id2type.get(str(bid))
                    by_type.setdefault(tname or '(untyped)', []).append(bid)
                self.log(f'  query {query!r}: {len(body_ids)} bodyIds in '
                         f'{len(by_type)} concrete source types')
            else:
                by_type = {query: []}
                self.log(f'  query {query!r}: no bodyId pool; trying as a '
                         'plain type name')
            for src_type, pool in sorted(by_type.items()):
                pairs.extend(self._pairs_for_type(src_type, pool, query))

        # Revision 2: refine pools through the bridge evidence, then
        # annotate branches per parent source type.
        by_parent: Dict[Tuple[str, str], List[TypePair]] = defaultdict(list)
        for pair in pairs:
            by_parent[(pair.query, pair.source_type)].append(pair)
        for (query, src_type), group in by_parent.items():
            annotate_pair_branches(group)
            for pair in group:
                if pair.pool_basis == 'linker rows':
                    self.log(
                        f'  branch {src_type} -> {pair.target_type}: '
                        f'pool {len(pair.source_pool)} of '
                        f'{pair.source_type_total} via '
                        f'[{pair.linker_values or "chain"}] '
                        f'(target {len(pair.target_pool)} of '
                        f'{pair.target_type_total})')
                else:
                    self.log(
                        f'  branch {src_type} -> {pair.target_type}: '
                        f'full population pool {len(pair.source_pool)} '
                        f'(no supported bridge chain)')
        return pairs

    def _refine_pair_branch(self, pair: TypePair) -> bool:
        """Replace the pair's pools with the selected bridge's refined
        subsets when a supported chain exists.  Returns True when the
        pair keeps a refined pool; fail-open to full pools otherwise
        (Revision 3.3: evidence_only branches without a supported chain
        are DROPPED by the caller instead — a wide fan-out must not be
        validated against full populations)."""
        from ui.neuron_index import resolve_prioritized_bridge_pool
        try:
            chains = self.mapper.get_type_bridges(
                pair.source_type, pair.source_dataset,
                pair.target_dataset, max_bridges=0)
            pool = resolve_prioritized_bridge_pool(
                pair.source_dataset, pair.target_dataset, chains,
                pair.source_type, pair.target_type)
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! branch pool resolution failed for '
                     f'{pair.source_type} -> {pair.target_type}: {exc}')
            return False
        if pool.get('resolution_status') != 'supported' \
                or not pool.get('selected_chain'):
            return False
        src_ids = [int(b) for b in (pool.get('source_body_ids') or [])]
        tgt_ids = [int(b) for b in (pool.get('target_body_ids') or [])]
        if not src_ids:
            return
        pair.source_pool = src_ids
        pair.target_pool = tgt_ids or pair.target_pool
        # Revision 3.12 (user 2026-09-13): chain-aware POOL WIDENING is
        # RETIRED.  Folding alternative-chain members into the validated
        # pool changed the tier across modes and erased the family bin
        # (the old open item 1).  Those members are now labelled `family`
        # by the category classifier instead, so the tier is identical
        # across restrictive/family/aggressive.  Base refinement (the
        # linker-refined pools) still runs unconditionally — see §3b.
        pair.selected_chain = [
            dict(hop) for hop in pool['selected_chain']]
        pair.linkers = [
            {
                'column': l.get('column', ''),
                'raw_value': l.get('raw_value', l.get('value', '')),
                'canonical_value': l.get('canonical_value',
                                         l.get('value', '')),
                'home': l.get('home', ''),
            }
            for l in (pool.get('per_linker') or [])
            if l.get('home') and l.get('column')
        ]
        pair.pool_basis = pool.get('source_basis') or 'linker rows'
        pair.target_pool_basis = pool.get('target_basis') or 'linker rows'
        if pair.pool_widen_added_sources or pair.pool_widen_added_targets:
            if pair.pool_basis == 'linker rows':
                pair.pool_basis = 'linker rows + alternative chains'
        pair.source_type_total = int(pool.get('source_type_total') or 0)
        pair.target_type_total = int(pool.get('target_type_total') or 0)
        return True

    def _pairs_for_type(self, src_type: str, pool: List[int],
                        query: str) -> List[TypePair]:
        cfg = self.cfg
        dec = self.mapper.get_mapping_decision(
            src_type, cfg.source_dataset, cfg.target_dataset)
        status = dec.get('status')
        if status in ('conflict', 'unmapped'):
            self.log(f'  - {src_type}: {status} — excluded (fail-closed)')
            return []
        targets = list(dec.get('target_types') or [])
        if not targets:
            self.log(f'  - {src_type}: status {status} with no target — '
                     'excluded')
            return []
        # Revision 3.3: the evidence_only fan-out guard (>4 targets) is
        # relaxed — branch resolution self-filters (unsupported branches
        # drop below), so wide fan-outs like s-CPDN3D (6 targets) are
        # tractable.  evidence_only branches that fail refinement are
        # dropped (never validated against full populations).
        if len(targets) > 1:
            self.log(f'  - {src_type}: {status} onto {len(targets)} targets')

        pairs = []
        for tgt_type in targets:
            tgt_pool = self._bodyids_for(tgt_type, cfg.target_dataset)
            if not tgt_pool:
                self.log(f'  - {src_type} -> {tgt_type}: empty target pool '
                         '— excluded')
                continue
            pair = TypePair(
                source_dataset=cfg.source_dataset,
                source_type=src_type,
                source_pool=sorted(int(b) for b in pool),
                target_dataset=cfg.target_dataset,
                target_type=tgt_type,
                target_pool=tgt_pool,
                relationship=dec.get('relationship') or '',
                status=status,
                query=query)
            pair.parent_source_pool = list(pair.source_pool)
            pair.parent_target_pool = list(pair.target_pool)
            if not self._refine_pair_branch(pair) \
                    and status == 'evidence_only':
                self.log(f'  - {src_type} -> {tgt_type}: evidence_only '
                         'without a supported bridge — dropped')
                continue
            pairs.append(pair)
        return pairs

    # -- stage 2 + 3 ------------------------------------------------------

    def validate_pair(self, pair: TypePair,
                      scans: Dict[int, pd.DataFrame],
                      target_id2type: Optional[Dict[int, str]] = None,
                      sizes: Optional[Dict[int, float]] = None,
                      weights: Optional[Dict[int, float]] = None,
                      source_sides: Optional[Dict[int, str]] = None,
                      target_sides: Optional[Dict[int, str]] = None
                      ) -> Dict[str, object]:
        """Global-rank validation for one type pair from cached scans.

        ``sizes`` / ``weights`` are the target-dataset caliber maps
        (spatial size per bodyId; expanded-vector total weight as the
        fallback caliber).  Returns {'rows', 'suspicious', 'noise',
        'pairs', 'summary', 'fills', 'per_source', ...}.
        """
        cfg = self.cfg
        pool_set = set(pair.target_pool)
        val_rows: List[Dict] = []
        sus_rows: List[Dict] = []
        noise_rows: List[Dict] = []
        per_source: Dict[int, pd.DataFrame] = {}
        pool_best_size = max((sizes.get(int(b), 0.0) for b in pool_set),
                             default=0.0) if sizes else 0.0
        pool_best_weight = max(
            (weights.get(int(b), 0.0) for b in pool_set),
            default=0.0) if weights else 0.0
        best_by_src: Dict[int, pd.Series] = {}

        for sbid in pair.source_pool:
            df = scans.get(sbid)
            sp = self.profiler.get_profile(sbid, pair.source_dataset)
            status_name = (sp.connectivity_status.name
                           if sp is not None else 'NO_PROFILE')
            if df is None or df.empty:
                val_rows.append(self._val_row(pair, sbid, 'skipped',
                                              status=status_name))
                continue

            pool_df = df[df['target_bid'].isin(pool_set)]
            if pool_df.empty:
                val_rows.append(self._val_row(
                    pair, sbid, 'unmatched', status=status_name,
                    flags='no_pool_member_in_cache'))
                continue
            best = pool_df.sort_values(
                ['rank_union_rank', 'jaccard_rank', 'rank_union'],
                ascending=[True, True, False], na_position='last').iloc[0]
            best_by_src[sbid] = best
            k = cfg.rank_top_k
            # Positivity policy (user 2026-09-11): only a POSITIVE
            # rank_union is strong evidence — a top-1 rank earned by a
            # less-negative score on a weak/incomplete profile carries no
            # weight (see suspicious_noise_diagnostic).  Jaccard is
            # sign-free and unaffected.
            best_ru = best.get('rank_union')
            ru_positive = (best_ru is not None
                           and not pd.isna(best_ru) and best_ru > 0)
            ru1 = best['rank_union_rank'] == 1 and ru_positive
            ja1 = best['jaccard_rank'] == 1
            flags = []
            if best['rank_union_rank'] == 1 and not ru_positive:
                flags.append('negative_rank_union_top1')
            if ru1 and ja1:
                verdict, which = 'verified_strong', 'both'
            elif ru1 or ja1:
                verdict = 'verified'
                which = '+'.join(n for n, top in
                                 (('rank_union', ru1), ('jaccard', ja1))
                                 if top)
            elif min(_rank_or_inf(best['rank_union_rank']),
                     _rank_or_inf(best['jaccard_rank'])) <= k:
                verdict, which = 'borderline', ''
            else:
                verdict, which = 'unmatched', ''

            # suspicious: non-pool neurons ranked ahead of the best pool
            # member under either metric (one row per metric).
            # Noise gates, in order (Rev 3.6 — spatial caliber PRIMARY,
            # connectivity heuristics secondary):
            #   1. positivity (rank_union rows): a non-positive ru win is
            #      ordering noise on a weak profile;
            #   2. jaccard sanity: ru win with jaccard far below the best
            #      pool member's;
            #   3. tie margin: ru win by less than suspicious_ru_margin
            #      over the pool best is a numerical tie;
            #   4. spatial caliber (both metrics): candidate size below
            #      target_min_size_ratio x the pool's best (metadata
            #      size; expanded-weight ratio fallback).
            # Dropped rows go to res['noise'] with reasons — nothing
            # silently disappears.
            ahead_ids: Dict[int, List[str]] = {}
            noise_count = 0
            tie_count = 0
            size_count = 0
            best_pool_jaccard = best.get('jaccard')
            for metric in ('rank_union', 'jaccard'):
                best_rank = best[f'{metric}_rank']
                if pd.isna(best_rank):
                    continue
                ahead = df[(~df['target_bid'].isin(pool_set))
                           & df[f'{metric}_rank'].notna()
                           & (df[f'{metric}_rank'] < best_rank)]
                ahead = ahead.nsmallest(cfg.suspicious_per_source_cap,
                                        f'{metric}_rank')
                for r in ahead.itertuples(index=False):
                    reasons: List[str] = []
                    if metric == 'rank_union':
                        # 1. positivity policy (user 2026-09-11): only a
                        # POSITIVE rank_union is evidence.
                        ru_ok = (r.rank_union is not None
                                 and not pd.isna(r.rank_union)
                                 and r.rank_union > 0)
                        if not ru_ok:
                            reasons.append('negative_rank_union')
                        # 2. jaccard sanity (Issue 5b).
                        elif best_pool_jaccard is not None \
                                and not pd.isna(best_pool_jaccard) \
                                and r.jaccard is not None \
                                and not pd.isna(r.jaccard) \
                                and r.jaccard < cfg.suspicious_jaccard_factor \
                                    * float(best_pool_jaccard):
                            reasons.append('jaccard_below_pool')
                        # 3. tie margin (Rev 3.6).
                        elif not pd.isna(best_ru) \
                                and not pd.isna(r.rank_union) \
                                and (float(r.rank_union)
                                     - float(best_ru)) \
                                < cfg.suspicious_ru_margin:
                            reasons.append('tie_margin')
                    # 4. spatial caliber (Rev 3.6, both metrics).
                    bid_i = int(r.target_bid)
                    size_ratio = None
                    if sizes and pool_best_size > 0:
                        a_size = sizes.get(bid_i)
                        if a_size is not None:
                            size_ratio = a_size / pool_best_size
                    if size_ratio is None and weights and pool_best_weight:
                        # fallback caliber: expanded-vector total weight
                        w = weights.get(bid_i)
                        if w is not None:
                            size_ratio = w / pool_best_weight
                    if size_ratio is not None \
                            and size_ratio < cfg.target_min_size_ratio:
                        reasons.append('spatial_caliber')
                    if reasons:
                        noise_count += 1
                        if 'tie_margin' in reasons:
                            tie_count += 1
                        if 'spatial_caliber' in reasons:
                            size_count += 1
                        noise_row = self._sus_row(
                            pair, sbid, r, best, metric,
                            getattr(r, 'rank_union_rank', None),
                            getattr(r, 'jaccard_rank', None), target_id2type)
                        noise_row.update({
                            'ahead_size': (sizes.get(bid_i)
                                           if sizes else None),
                            'pool_best_size': pool_best_size,
                            'size_ratio': (_f(size_ratio)
                                           if size_ratio is not None
                                           else None),
                            'noise_reason': ';'.join(reasons),
                        })
                        noise_rows.append(noise_row)
                        continue
                    ahead_ids.setdefault(bid_i, []).append(metric)
                    row = self._sus_row(
                        pair, sbid, r, best, metric,
                        getattr(r, 'rank_union_rank', None),
                        getattr(r, 'jaccard_rank', None), target_id2type)
                    a_size = sizes.get(bid_i) if sizes else None
                    row.update({
                        'ahead_size': a_size,
                        'pool_best_size': pool_best_size,
                        'size_ratio': (_f(size_ratio)
                                       if size_ratio is not None else None),
                        'size_filtered': False,
                    })
                    sus_rows.append(row)

            if noise_count:
                flags.append(
                    f'noise_filtered={noise_count}'
                    f'(neg_ru_or_jac={noise_count - tie_count - size_count},'
                    f'tie={tie_count},size={size_count})')
            if ahead_ids and verdict in ('verified_strong', 'verified',
                                         'borderline'):
                flags.append('suspicious_ahead')
            if status_name in WEAK_STATUSES:
                flags.append(f'weak_profile_{status_name}')
            val_rows.append(self._val_row(
                pair, sbid, verdict, status=status_name, best=best,
                metric=which, flags=';'.join(flags),
                suspicious_count=len(ahead_ids),
                noise_filtered=noise_count,
                size_filtered=size_count,
                tie_filtered=tie_count))

            # bounded per-source retention for gap fill + assignment
            keep = set(pool_set)
            for metric in ('rank_union', 'jaccard'):
                ranked = df[df[metric].notna()].nsmallest(
                    FILL_KEEP_TOP, f'{metric}_rank')
                keep.update(int(b) for b in ranked['target_bid'])
            per_source[sbid] = df[df['target_bid'].isin(keep)]

        # Revision 3.9 deep-window candidates: aggressive MODE only
        # (Rev 3.12 enum; the default restrictive run stays
        # invader-only + gap-fire-only).  Out-of-pool neurons
        # within the retained top-`candidate_window` per metric that are
        # NOT ahead of the pool best — the homologs a strong pool member
        # can hide just below it.  Same spatial-caliber gate; structural
        # + morph qualification happen later (annotate_invaders + 5).
        deep_rows: List[Dict] = []
        deep_per_src: Dict[int, int] = {}
        null_rows: List[Dict] = []
        deep_cap_null = 0
        seen_deep = {int(r['ahead_target_bodyId']) for r in sus_rows}
        if cfg.mode_at_least('aggressive'):
            for sbid, df in per_source.items():
                src_best = best_by_src.get(sbid)
                if src_best is None:
                    continue
                for metric in ('rank_union', 'jaccard'):
                    if deep_per_src.get(sbid, 0) >= cfg.deep_cap:
                        break
                    window = df[(~df['target_bid'].isin(pool_set))
                                & df[f'{metric}_rank'].notna()
                                & (df[f'{metric}_rank']
                                   <= cfg.candidate_window)]
                    for r in window.itertuples(index=False):
                        bid = int(r.target_bid)
                        if bid in seen_deep or bid in pool_set:
                            continue
                        if deep_per_src.get(sbid, 0) >= cfg.deep_cap:
                            break
                        size_ratio = None
                        if sizes and pool_best_size > 0:
                            a_size = sizes.get(bid)
                            if a_size is not None:
                                size_ratio = a_size / pool_best_size
                        if size_ratio is None and weights and pool_best_weight:
                            wt = weights.get(bid)
                            if wt is not None:
                                size_ratio = wt / pool_best_weight
                        if size_ratio is not None \
                                and size_ratio < cfg.target_min_size_ratio:
                            continue
                        deep_per_src[sbid] = deep_per_src.get(sbid, 0) + 1
                        seen_deep.add(bid)
                        row = self._sus_row(
                            pair, sbid, r, src_best, metric,
                            getattr(r, 'rank_union_rank', None),
                            getattr(r, 'jaccard_rank', None), target_id2type)
                        row.update({
                            'ahead_size': (sizes.get(bid) if sizes else None),
                            'pool_best_size': pool_best_size,
                            'size_ratio': (_f(size_ratio)
                                           if size_ratio is not None else None),
                            'size_filtered': False,
                            'candidate_source': 'deep_window',
                        })
                        deep_rows.append(row)
        # Rev 3.9 null sample for the Track-A bar calibration: window
        # rows with near-zero jaccard are provably unrelated by
        # connectivity — their Track-A morph distribution IS the
        # cross-dataset baseline (T1/R1-R6 live here).  Independent of
        # the deep window (works in the default mode); calibration data
        # only: never exported as candidates.
        seen_null = seen_deep
        null_per_src: Dict[int, int] = {}
        for sbid, df in per_source.items():
            src_best = best_by_src.get(sbid)
            if src_best is None:
                continue
            if null_per_src.get(sbid, 0) >= cfg.null_per_source_cap:
                continue
            nulldf = df[(~df['target_bid'].isin(pool_set))
                        & (df['jaccard'].notna())
                        & (df['jaccard'] <= cfg.null_jaccard_max)
                        & (~df['target_bid'].isin(seen_null))]
            for r in nulldf.itertuples(index=False):
                bid = int(r.target_bid)
                if null_per_src.get(sbid, 0) >= cfg.null_per_source_cap:
                    break
                if bid in seen_null or bid in pool_set:
                    continue
                null_per_src[sbid] = null_per_src.get(sbid, 0) + 1
                seen_null.add(bid)
                null_rows.append(self._sus_row(
                    pair, sbid, r, src_best, 'jaccard',
                    getattr(r, 'rank_union_rank', None),
                    getattr(r, 'jaccard_rank', None), target_id2type))

        assigned = mutual_best_assignment(pool_set, per_source, val_rows)
        summary = self._summary(pair, val_rows, assigned,
                                self.cfg.gap_min)
        # Hemisphere-asymmetry trigger (user 2026-09-12): pool neurons
        # should be L/R symmetric — an L != R imbalance in either pool
        # fires the gap even at arithmetic gap 0 (every neuron has a
        # hemisphere identity; asymmetry means a missing partner).
        hemi_fired = False
        if source_sides is not None and target_sides is not None:
            hemi_fired, hemi = hemisphere_gap_fired(
                pair.source_pool, pair.target_pool, source_sides,
                target_sides)
            summary['gap_triggered'] = bool(summary['gap_triggered']
                                            or hemi_fired)
            summary['hemisphere'] = hemi
        # Revision 3.8: in the DEFAULT scope fills are gap-gated (now
        # including the hemisphere-asymmetry trigger); in aggressive MODE
        # (Rev 3.12 enum) they are unconditional.
        fills = self._gap_fill(pair, per_source, assigned, val_rows,
                               pool_set, target_id2type,
                               sizes=sizes, weights=weights,
                               force_fire=hemi_fired)
        target_categories, pool_detail = categorize_pool_targets(
            per_source, pool_set, top_n=self.cfg.verified_top_n,
            invader_max=self.cfg.invader_borderline_max,
            matched_ru_min=self.cfg.matched_ru_min)
        for row in pool_detail:
            row.update({'query': pair.query,
                        'source_type': pair.source_type,
                        'target_type': pair.target_type,
                        'pool_basis': pair.pool_basis,
                        'branch_annotation': pair.branch_annotation})
            if sizes:
                row['size'] = sizes.get(int(row['target_bodyId']))
        summary['pool_best_size'] = pool_best_size
        # morph pairing for pool neurons: best source per target
        pool_pairs = [(d['best_source_bodyId'], d['target_bodyId'])
                      for d in pool_detail
                      if d.get('best_source_bodyId') is not None]
        summary['deep_candidates'] = len(deep_rows)
        summary['null_sample'] = len(null_rows)
        return {'rows': val_rows, 'suspicious': sus_rows,
                'noise': noise_rows, 'deep': deep_rows,
                'null': null_rows,
                'pairs': assigned, 'summary': summary, 'fills': fills,
                'per_source': per_source,
                'target_categories': target_categories,
                'pool_detail': pool_detail,
                '_pool_set': sorted(int(b) for b in pool_set),
                'pool_pairs': pool_pairs}

    @staticmethod
    def _val_row(pair: TypePair, sbid: int, verdict: str, status: str = '',
                 best: Optional[pd.Series] = None, metric: str = '',
                 flags: str = '', suspicious_count: int = 0,
                 noise_filtered: int = 0, size_filtered: int = 0,
                 tie_filtered: int = 0) -> Dict:
        row = {
            'query': pair.query,
            'source_dataset': pair.source_dataset,
            'source_type': pair.source_type,
            'target_dataset': pair.target_dataset,
            'target_type': pair.target_type,
            'mapping_status': pair.status,
            'relationship': pair.relationship,
            'pool_basis': pair.pool_basis,
            'branch_linker_values': pair.linker_values,
            'branch_annotation': pair.branch_annotation,
            'branches_disjoint': pair.branches_disjoint,
            'source_bodyId': sbid,
            'source_connectivity_status': status,
            'verdict': verdict,
            'metric_top1': metric,
            'flags': flags,
            'suspicious_count': suspicious_count,
            'suspicious_noise_filtered': noise_filtered,
            'suspicious_size_filtered': size_filtered,
            'suspicious_tie_filtered': tie_filtered,
            'target_bodyId': None,
            'rank_union': None, 'rank_union_rank': None,
            'jaccard': None, 'jaccard_rank': None,
            'cosine': None, 'weighted_jaccard': None,
        }
        if best is not None:
            row.update({
                'target_bodyId': int(best['target_bid']),
                'rank_union': _f(best.get('rank_union')),
                'rank_union_rank': _f(best.get('rank_union_rank')),
                'jaccard': _f(best.get('jaccard')),
                'jaccard_rank': _f(best.get('jaccard_rank')),
                'cosine': _f(best.get('cosine')),
                'weighted_jaccard': _f(best.get('weighted_jaccard')),
            })
        return row

    @staticmethod
    def _sus_row(pair: TypePair, sbid: int, r, best, metric: str,
                 ru_rank, ja_rank, target_id2type) -> Dict:
        return {
            'query': pair.query,
            'source_type': pair.source_type,
            'target_type': pair.target_type,
            'pool_basis': pair.pool_basis,
            'branch_linker_values': pair.linker_values,
            'branch_annotation': pair.branch_annotation,
            'source_bodyId': sbid,
            'ahead_target_bodyId': int(r.target_bid),
            'ahead_target_type': (target_id2type or {}).get(
                int(r.target_bid), '?'),
            'ahead_metric': metric,
            'ahead_rank': _f(ru_rank if metric == 'rank_union' else ja_rank),
            'ahead_rank_union': _f(r.rank_union),
            'ahead_jaccard': _f(r.jaccard),
            'best_pool_target_bodyId': int(best['target_bid']),
            'best_pool_rank': _f(best[metric + '_rank']
                                 if metric != 'rank_union'
                                 else best['rank_union_rank']),
            'best_pool_rank_union': _f(best['rank_union']),
            'best_pool_jaccard': _f(best['jaccard']),
        }

    @staticmethod
    def _summary(pair: TypePair, val_rows: List[Dict],
                 assigned: List[Tuple[int, int]], gap_min: int = 1) -> Dict:
        n_src, n_tgt = len(pair.source_pool), len(pair.target_pool)
        g = gap_check(n_src, n_tgt, len(assigned), gap_min=gap_min)
        return {
            'query': pair.query,
            'source_type': pair.source_type,
            'target_type': pair.target_type,
            'mapping_status': pair.status,
            'relationship': pair.relationship,
            'pool_basis': pair.pool_basis,
            'selected_chain': pair.chain_text,
            'branch_linker_values': pair.linker_values,
            'branch_annotation': pair.branch_annotation,
            'branches_disjoint': pair.branches_disjoint,
            'source_pool': n_src, 'target_pool': n_tgt,
            'pool_widen_added': len(pair.pool_widen_added_sources)
            + len(pair.pool_widen_added_targets),
            'source_type_total': pair.source_type_total,
            'target_type_total': pair.target_type_total,
            'matched': len(assigned),
            'verdict_verified_strong': sum(
                r['verdict'] == 'verified_strong' for r in val_rows),
            'verdict_verified': sum(
                r['verdict'] == 'verified' for r in val_rows),
            'verdict_borderline': sum(
                r['verdict'] == 'borderline' for r in val_rows),
            'verdict_unmatched': sum(
                r['verdict'] == 'unmatched' for r in val_rows),
            'verdict_skipped': sum(
                r['verdict'] == 'skipped' for r in val_rows),
            'suspicious_neurons': sum(
                r['suspicious_count'] > 0 for r in val_rows),
            'suspicious_noise_filtered': sum(
                r.get('suspicious_noise_filtered', 0) for r in val_rows),
            'suspicious_size_filtered': sum(
                r.get('suspicious_size_filtered', 0) for r in val_rows),
            'suspicious_tie_filtered': sum(
                r.get('suspicious_tie_filtered', 0) for r in val_rows),
            **g,
        }

    def _gap_fill(self, pair: TypePair, per_source: Dict[int, pd.DataFrame],
                  assigned: List[Tuple[int, int]], val_rows: List[Dict],
                  pool_set: set,
                  target_id2type: Optional[Dict[int, str]],
                  sizes: Optional[Dict[int, float]] = None,
                  weights: Optional[Dict[int, float]] = None,
                  force_fire: bool = False
                  ) -> List[Dict]:
        """Propose partners for unpaired neurons from the global ranking.

        Revision 3.8: expansion is UNCONDITIONAL — the gap trigger is
        retired as a gate because annotations carry unexpected errors
        and incompleteness; every unpaired neuron gets a proposal
        (evidence only, the mapping is never rewritten).  `gap` /
        `gap_ratio` / `gap_triggered` remain informational statistics
        in pair_summary.csv.

        Out-of-pool candidates pass the SAME spatial-caliber gate as the
        suspicious rows (Rev 3.6: a fragment proposed as a homolog is
        noise, e.g. MCNS 603004462 — 15 pre / 9 post, 2.4 M nm³ vs a
        ~4e8 nm³ pool — won rank_union for an unpaired source and used
        to flow straight into the scene).

        Source side: the best target neuron not already assigned, of ANY
        type (candidates limited to the retained top-100-per-metric + pool
        rows).  Target side: best unassigned source in the pool — the
        reverse global scan is not part of this direction's run.
        """
        if not self.cfg.mode_at_least('aggressive') and not force_fire:
            summary = self._summary(pair, val_rows, assigned,
                                    self.cfg.gap_min)
            if not summary['gap_triggered']:
                return []
        assigned_src = {s for s, _ in assigned}
        assigned_tgt = {t for _, t in assigned}
        row_by_src = {r['source_bodyId']: r for r in val_rows}
        pool_best_size = max((sizes.get(int(b), 0.0) for b in pool_set),
                             default=0.0) if sizes else 0.0
        proposals: List[Dict] = []

        def caliber_ok(bid: int) -> bool:
            if not sizes or not pool_best_size:
                return True
            a = sizes.get(int(bid))
            return a is None or pool_best_size <= 0 \
                or a / pool_best_size >= self.cfg.target_min_size_ratio

        for sbid, df in per_source.items():
            if sbid in assigned_src:
                continue
            cand = df[~df['target_bid'].isin(assigned_tgt)]
            if cand.empty:
                continue
            best = cand.sort_values(
                ['rank_union_rank', 'jaccard_rank', 'rank_union'],
                ascending=[True, True, False], na_position='last').iloc[0]
            tbid = int(best['target_bid'])
            in_pool = tbid in pool_set
            # spatial-caliber gate on out-of-pool candidates: fall through
            # to the next-best candidate above the caliber floor
            if not in_pool and not caliber_ok(tbid):
                cand = cand[cand['target_bid'].apply(
                    lambda b: int(b) in pool_set
                    or caliber_ok(int(b)))]
                if cand.empty:
                    continue
                best = cand.sort_values(
                    ['rank_union_rank', 'jaccard_rank', 'rank_union'],
                    ascending=[True, True, False],
                    na_position='last').iloc[0]
                tbid = int(best['target_bid'])
                in_pool = tbid in pool_set
            proposals.append({
                'query': pair.query,
                'source_type': pair.source_type,
                'target_type': pair.target_type,
                'side': 'source',
                'bodyId': sbid,
                'source_verdict': (row_by_src[sbid]['verdict']
                                   if sbid in row_by_src else ''),
                'proposal_bodyId': tbid,
                'fill_class': 'in_pool' if in_pool else 'out_of_pool',
                'proposal_type': (pair.target_type if in_pool else
                                  (target_id2type or {}).get(tbid, '?')),
                'rank_union': _f(best['rank_union']),
                'rank_union_rank': _f(best['rank_union_rank']),
                'jaccard': _f(best['jaccard']),
                'jaccard_rank': _f(best['jaccard_rank']),
            })
        # target-side: unassigned pool neurons -> best unassigned pool source
        for t in sorted(pool_set - assigned_tgt):
            cands = []
            for sbid, df in per_source.items():
                if sbid in assigned_src:
                    continue
                rows = df[df['target_bid'] == t]
                if not rows.empty:
                    cands.append((sbid, rows.iloc[0]))
            if not cands:
                continue
            sbid, r = min(cands, key=lambda cr: _pair_sort_key(cr[1]))
            proposals.append({
                'query': pair.query,
                'source_type': pair.source_type,
                'target_type': pair.target_type,
                'side': 'target',
                'bodyId': t,
                'source_verdict': '',
                'proposal_bodyId': sbid,
                'fill_class': 'in_pool',
                'proposal_type': pair.source_type,
                'rank_union': _f(r['rank_union']),
                'rank_union_rank': _f(r['rank_union_rank']),
                'jaccard': _f(r['jaccard']),
                'jaccard_rank': _f(r['jaccard_rank']),
            })
        return proposals

    # -- Revision 3.6: invader re-classification --------------------------

    def _backward_decision(self, atype: str) -> Dict:
        """Cached backward mapping + home reality check for one target
        type: {'status', 'mapped', 'home_count', 'home_real'}."""
        cache = getattr(self, '_backward_cache', None)
        if cache is None:
            cache = self._backward_cache = {}
        if atype in cache:
            return cache[atype]
        dec = {'status': '', 'mapped': None, 'home_count': 0,
               'home_real': False}
        if self.mapper is not None and atype and atype not in ('?',):
            try:
                d = self.mapper.get_mapping_decision(
                    atype, self.cfg.target_dataset, self.cfg.source_dataset)
                dec['status'] = str(d.get('status') or '')
                cands = [d.get('target_type')]
                cands += list(d.get('target_types') or [])
                mapped = [str(t) for t in dict.fromkeys(cands)
                          if t and str(t) != 'None']
                dec['mapped'] = '/'.join(mapped) or None
                type_counts = getattr(self, '_source_type_counts',
                                      {}) or {}
                add_counts = getattr(self, '_source_add_counts', {}) or {}
                counts = []
                for name in mapped:
                    n = int(type_counts.get(name, 0))
                    n += int(add_counts.get(name, 0))
                    counts.append(n)
                    dec['home_count'] = max(dec['home_count'], n)
                dec['home_real'] = any(n > 0 for n in counts)
            except Exception:
                pass
        cache[atype] = dec
        return dec

    def annotate_invaders(self, sus_rows: List[Dict],
                          fill_rows: List[Dict],
                          per_pair_res: Dict,
                          family_types: Optional[set] = None) -> None:
        """LEGACY (pre-3.12) structural classifier, retained for the
        compatibility `invader_class`/`invader_label` columns and the
        legacy bucket tests.  The CANONICAL classification is now
        :meth:`finalize_categories` (the S2 partition into
        matched/…/sibling/candidates/family/relative/suspicious), which
        runs after morphology.  This method still tags each row with the
        old evidence columns (`sibling_pool_of`, `backward_status`,
        `alt_chain_of_parent`, …) and the legacy counts.

        Rev 3.6 rule (unchanged here): sibling > alternate-chain >
        backward > unexplained; structural facts outrank morphology.
        """
        family_types = family_types or set()
        # sibling-branch pool membership (same parent, any branch)
        sibling_pools: Dict[int, List[Tuple[str, str, str]]] = {}
        for (src_type, tgt_type), res in per_pair_res.items():
            for tbid, cat in (res.get('target_categories')
                              or {}).items():
                sibling_pools.setdefault(int(tbid), []).append(
                    (str(src_type), str(tgt_type), str(cat)))
        chain_index = build_chain_index(self.pairs, self.mapper)

        def classify(bid: int, atype, own_branch: Tuple[str, str]) -> Dict:
            atype = None if atype is None or str(atype) in ('', 'nan', '?') \
                or (isinstance(atype, float) and atype != atype) \
                else str(atype)
            fields = {'invader_class': '', 'invader_label': None,
                      'sibling_pool_of': '', 'sibling_category': '',
                      'backward_status': '', 'backward_maps_to': '',
                      'fafb_home_count': None, 'backward_home_real': '',
                      'alt_chain_of_parent': '', 'in_query_family': '',
                      'same_type_residue': False}
            parent = own_branch[0]
            # Revision 3.11: a SAME-TYPE extra (ahead type == the branch
            # target type, outside the widened pool) is a residue of this
            # branch's own target — family/fill material, never a
            # cross-type invader.  Its qualification (and thus fill vs
            # family) is decided by the morph rule at render/export.
            if atype is not None and atype == str(own_branch[1]):
                fields.update({'invader_class': 'same-type',
                               'same_type_residue': True})
                return fields
            # Revision 3.6 refinement (user 2026-09-12): sibling = a pool
            # member of a sibling branch of the SAME parent only — members
            # of OTHER parents' branches fall through to the backward
            # classification (e.g. a verified s-CPDN3D->SMP222 member in
            # an s-CPDN3C scene is backward · s-CPDN3D, not sibling).
            sibs = [(s, t, c) for s, t, c in sibling_pools.get(bid, [])
                    if s == parent and (s, t) != own_branch]
            if sibs:
                best = sorted(sibs, key=lambda s: POOL_CATEGORY_ORDER.index(
                    s[2]) if s[2] in POOL_CATEGORY_ORDER else 99)[0]
                fields.update({
                    'invader_class': 'sibling',
                    'invader_label':
                        f'sibling · {best[2]} · {best[1]}',
                    'sibling_pool_of': ';'.join(sorted({f'{s}->{t}'
                                                        for s, t, _ in sibs})),
                    'sibling_category': ';'.join(
                        sorted({c for _, _, c in sibs})),
                })
                return fields
            if atype is None:
                fields['invader_class'] = 'untyped'
                return fields
            dec = self._backward_decision(atype)
            in_family = any(
                tok.strip() in family_types
                for tok in str(dec['mapped'] or '').split('/')) \
                if dec['mapped'] else False
            fields.update({
                'backward_status': dec['status'],
                'backward_maps_to': dec['mapped'] or '',
                'fafb_home_count': dec['home_count'],
                'backward_home_real': bool(dec['home_real']),
                'in_query_family': in_family,
            })
            alt = (chain_index.get(parent, {}).get(atype, {})
                   or {}).get('alt') or []
            if alt:
                fields['alt_chain_of_parent'] = ';'.join(sorted(alt))
            if dec['status'] in ('mapped', 'valid_split_evidence') \
                    and dec['home_real']:
                if alt:
                    fields['invader_class'] = 'alternate-chain'
                    fields['invader_label'] = (
                        f'alternate-chain · {";".join(sorted(alt))} '
                        f'→ {atype}')
                else:
                    fields['invader_class'] = 'backward'
                    fields['invader_label'] = f'backward · {dec["mapped"]}'
            elif dec['status'] in ('mapped', 'valid_split_evidence'):
                fields['invader_class'] = 'hollow-backward'
            else:
                fields['invader_class'] = 'unmapped'
            return fields

        for r in sus_rows:
            fields = classify(int(r['ahead_target_bodyId']),
                              r.get('ahead_target_type'),
                              (str(r['source_type']),
                               str(r['target_type'])))
            r.update(fields)
            r.setdefault('size_filtered', False)
        for f in fill_rows:
            # Revision 3.12 (fix D1): classify BOTH pairing directions.
            # A fill row is the PAIRED category regardless of side; the
            # old code skipped target-side rows, leaving their class
            # blank.  Target-side rows are always `in_pool` (created at
            # `_gap_fill`), so they take the same branch as source-side
            # in-pool rows.
            if f.get('fill_class') == 'in_pool':
                f.update({'invader_class': 'in-pool',
                          'invader_label': None,
                          'sibling_pool_of': '', 'sibling_category': '',
                          'backward_status': '', 'backward_maps_to': '',
                          'fafb_home_count': None,
                          'backward_home_real': '',
                          'alt_chain_of_parent': '',
                          'in_query_family': '',
                          'fill_scope_note': '',
                          'counts_toward_gap_fill': True})
                continue
            if f.get('side') != 'source':
                # target-side rows are always in_pool; anything else here
                # is unexpected — leave it for finalize_categories.
                continue
            fields = classify(int(f['proposal_bodyId']),
                              f.get('proposal_type'),
                              (str(f['source_type']),
                               str(f['target_type'])))
            f.update(fields)
            # User 2026-09-12 + Rev 3.10: fill accounting is class- and
            # scope-aware.  Sibling (cross-branch convergence), backward
            # (cross-type: in-family = convergence, out-of-family =
            # query-expansion advice) and different-target
            # alternate-chain NEVER count; only same-type
            # alternate-chain residues and unexplained candidates do.
            cls = fields['invader_class']
            if cls == 'same-type':
                # Rev 3.11: the only legitimate fills of the mapped type
                # -- qualification (morph) is finalized in stage 5
                counts = None
                note = 'same_type_residue'
            elif cls == 'sibling':
                counts = False
                note = 'cross_branch_convergence'
            elif cls == 'alternate-chain':
                # Rev 3.11: residues are mapped-set members (family) --
                # siblings, not fills
                counts = False
                note = ('same_type_residue' if
                        str(f.get('proposal_type')) ==
                        str(f['target_type'])
                        else 'cross_branch_convergence')
            elif cls == 'backward':
                counts = False
                in_family = bool(fields.get('in_query_family'))
                note = ('cross_branch_convergence' if in_family
                        else 'query_expansion_advice')
            elif cls == 'untyped':
                # Revision 3.12 (fix D4): an untyped suspect has no
                # annotation, so it can never advance coverage by itself;
                # only a morph-qualified `candidates` row counts.
                counts = False
                note = ''
            else:
                # hollow-backward / unmapped: connectivity-qualified but
                # the morph rule has not been evaluated yet -> None
                # (unknown); `finalize_categories` resolves it to
                # `counts_toward_gap_fill` iff the row becomes a candidate.
                counts = None
                note = ''
            f['counts_toward_gap_fill'] = counts
            f['fill_scope_note'] = note

    # -- Revision 3.12: the category partition ---------------------------

    def _bodyids_of_type(self, tname: str) -> List[int]:
        if not tname or str(tname) in ('?', 'nan', 'None'):
            return []
        try:
            return sorted(int(b) for b in (
                self.profiler.get_bodyids_for_type(
                    str(tname), self.cfg.target_dataset) or []))
        except Exception:  # noqa: BLE001
            return []

    def _leaf_token(self, row: Dict, tname) -> str:
        """The per-bodyId leaf token on an expansion leaf (S5), via
        :func:`candidate_annotation` with the ordered rule:

        - ``{T}(out-map)`` — the type is one of the mapping's in-map types
          (bodyId-level out-of-map; the fill material — includes every
          `family` member);
        - ``{T}>{src}`` — type NOT in-map but a real backward home;
        - ``{T}(no_source)`` — type NOT in-map with no usable route;
        - ``untyped`` — no type.

        Prefers the row's precomputed ``backward_maps_to`` /
        ``backward_home_real`` (set by ``annotate_invaders`` for invader
        and deep evidence rows); synthesised rows (enumerated `family` /
        `relative` members) carry none, so the type mapper is consulted
        directly.  The in-map-type check short-circuits both.
        """
        if tname in (None, '') or (
                isinstance(tname, float) and tname != tname) \
                or str(tname) == '?':
            return 'untyped'
        t = str(tname)
        if t in (getattr(self, '_in_map_types', None) or set()):
            return candidate_annotation(t, [], has_type=True,
                                        type_in_map=True)
        bm = (row or {}).get('backward_maps_to')
        home_real = bool((row or {}).get('backward_home_real'))
        srcs: List[str] = []
        if bm is not None and not (isinstance(bm, float) and pd.isna(bm)):
            srcs = [s for s in str(bm).split('/') if s]
        elif getattr(self, 'mapper', None) is not None:
            dec = self._backward_decision(t)
            if dec.get('status') in ('mapped', 'valid_split_evidence'):
                srcs = [s for s in str(dec.get('mapped') or '').split('/')
                        if s]
            home_real = bool(dec.get('home_real'))
        return candidate_annotation(t, srcs, has_type=True,
                                    home_real=home_real)

    def finalize_categories(self, per_pair_res: Dict, sus_rows: List[Dict],
                            deep_rows: List[Dict], fills: List[Dict],
                            pool_detail: List[Dict]) -> List[Dict]:
        """Compute the S2 category partition (Rev 3.12) and the query-level
        dedup rollup (S6) AFTER morphology, so qualification is known.

        Sets on every evidence row: ``category``, ``in_scope``,
        ``morph_failed``, ``candidate_annotation``,
        ``counts_toward_restrictive_fill``, ``counts_toward_family_fill``
        (and corrects the legacy ``counts_toward_gap_fill``).  Also sets
        ``_mapper_gap_types`` / ``_mapper_gap_untyped`` (the README
        mapper-gap report) from the FINALIZED annotations — candidate
        types with no backward route and untyped candidates.  Returns the
        query-level dedup rows for ``gap_fill_dedup.csv``.

        Design: `_plan/...rev311....md` §2b S2–S6 (normative).
        """
        cfg = self.cfg
        mode = cfg.effective_mode
        bars_by_key = getattr(self, '_branch_bars', None)
        if bars_by_key is None:
            # v2 compatibility (fixtures / morph-stage failure): synthesize
            # per-branch BarSets from the legacy fields with v2 precedence.
            from comparison.morph_bars import synthesize_barset
            floors = getattr(self, '_pool_ref_floors', {}) or {}
            thresholds = getattr(self, '_candidate_thresholds', {}) or {}
            null_bar = getattr(self, '_track_a_null_bar', None)
            all_keys = set(floors) | set(thresholds)
            bars_by_key = {k: synthesize_barset(native_floor=floors.get(k),
                                                track_a_bar=thresholds.get(k),
                                                null_bar=null_bar)
                           for k in all_keys}
            bars_by_key.setdefault(None, synthesize_barset(null_bar=null_bar))

        branch_pools: Dict[Tuple[str, str], set] = {}
        tiers: Dict[Tuple[str, str], Dict] = {}
        for key, res in per_pair_res.items():
            branch_pools[key] = {int(b) for b in (
                res.get('_pool_set') or [])}
            tiers[key] = res.get('target_categories') or {}
        in_map: set = set()
        for s in branch_pools.values():
            in_map |= s
        in_map_types = {str(t) for (_s, t) in branch_pools}
        self._in_map_ids = in_map
        self._in_map_types = in_map_types
        self._branch_pools = branch_pools

        def key_of(row) -> Tuple[str, str]:
            return (str(row['source_type']), str(row['target_type']))

        def mq(row) -> bool:
            # When morphology is disabled there is no qualification
            # evidence to gate on, so admission falls back to
            # connectivity alone (the fast structural pass).  This keeps
            # `--no-morphology` from blanking the whole expansion.
            if not cfg.morph_enabled:
                return True
            return morph_qualified(row, bars_by_key.get(
                key_of(row), bars_by_key.get(None)))

        def mq_suspicious(row) -> bool:
            # The LOOSE aggressive bar for deep-window rows (floors v3):
            # Track-A above B_b - k*Delta (or the run null p50).
            if not cfg.morph_enabled:
                return True
            return morph_qualified_suspicious(row, bars_by_key.get(
                key_of(row), bars_by_key.get(None)))

        # connectivity-qualified sets per branch: invaders + gap fires.
        invader_ids: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in sus_rows:
            invader_ids[key_of(r)].add(int(r['ahead_target_bodyId']))
        gapfire_ids: Dict[Tuple[str, str], set] = defaultdict(set)
        for f in fills:
            if f.get('side') == 'source' \
                    and f.get('fill_class') == 'out_of_pool':
                gapfire_ids[key_of(f)].add(int(f['proposal_bodyId']))
        deep_ids: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in deep_rows:
            deep_ids[key_of(r)].add(int(r['ahead_target_bodyId']))

        # Every connectivity-qualified evidence row (target-side view).
        evidence = []
        for r in sus_rows:
            evidence.append((r, int(r['ahead_target_bodyId']),
                             r.get('ahead_target_type')))
        for r in deep_rows:
            evidence.append((r, int(r['ahead_target_bodyId']),
                             r.get('ahead_target_type')))
        for f in fills:
            if f.get('side') == 'source' \
                    and f.get('fill_class') == 'out_of_pool':
                evidence.append((f, int(f['proposal_bodyId']),
                                 f.get('proposal_type')))

        # Phase A: classify; collect per-branch candidate types (rule 5).
        candidate_types: Dict[Tuple[str, str], set] = defaultdict(set)
        for row, bid, ttype in evidence:
            k = key_of(row)
            cq = (bid in invader_ids.get(k, set())
                  or bid in gapfire_ids.get(k, set()))
            tt = None if ttype in (None, '', '?') else str(ttype)
            cat, in_scope, mfail = classify_category(
                target_bid=bid, branch_pool=branch_pools.get(k),
                in_map=in_map, target_type=tt,
                branch_target_type=k[1], in_map_types=in_map_types,
                candidate_types=candidate_types.get(k),
                connectivity_qualified=cq, morph_ok=mq(row),
                suspicious_morph_ok=mq_suspicious(row),
                is_deep=(bid in deep_ids.get(k, set())),
                tier=tiers.get(k, {}).get(bid), mode=mode)
            row['category'] = cat
            row['in_scope'] = bool(in_scope)
            row['morph_failed'] = bool(mfail)
            if cat == 'candidates' and tt is not None:
                candidate_types[k].add(tt)

        labeled: Dict[Tuple[str, str], set] = defaultdict(set)
        for row, bid, _t in evidence:
            if str(row.get('category') or ''):
                labeled[key_of(row)].add(bid)

        # Phase B: residual bins (family / relative / suspicious) for the
        # DEEP rows only — they are the sole evidence rows that are not
        # connectivity-qualified, and rule 5 (relative) needs the complete
        # per-branch candidate-type set from phase A.  Rows already
        # categorized in phase A are left untouched (an out-of-scope row
        # must keep its morph_failed flag).
        for row, bid, ttype in evidence:
            k = key_of(row)
            if bid not in deep_ids.get(k, set()):
                continue
            if str(row.get('category') or ''):
                continue
            tt = None if ttype in (None, '', '?') else str(ttype)
            cat, in_scope, mfail = classify_category(
                target_bid=bid, branch_pool=branch_pools.get(k),
                in_map=in_map, target_type=tt,
                branch_target_type=k[1], in_map_types=in_map_types,
                candidate_types=candidate_types.get(k),
                connectivity_qualified=False, morph_ok=mq(row),
                suspicious_morph_ok=mq_suspicious(row),
                is_deep=True, tier=None, mode=mode)
            row['category'] = cat
            row['in_scope'] = bool(in_scope)
            row['morph_failed'] = bool(mfail)
            if str(cat):
                labeled[k].add(bid)

        # Pool (tier) rows.
        for d in pool_detail:
            k = (str(d['source_type']), str(d['target_type']))
            cat = tiers.get(k, {}).get(int(d['target_bodyId'])) or 'unmatched'
            d['category'] = cat
            d['in_scope'] = True
            d['morph_failed'] = False
            labeled[k].add(int(d['target_bodyId']))

        # Enumerate family and relative, both branch-local (S2): family is
        # THIS branch's target type; relative is THIS branch's candidate
        # types (outside the map).  Mode-gated.  The per-branch `labeled`
        # set keeps every residual exclusive.
        family_rows: List[Dict] = []
        relative_rows: List[Dict] = []
        if mode in ('family', 'aggressive'):
            fam_cache: Dict[str, List[int]] = {}
            rel_cache: Dict[str, List[int]] = {}
            for key, _pool in branch_pools.items():
                k_labeled = labeled.get(key, set())
                bt = key[1]
                if bt not in fam_cache:
                    fam_cache[bt] = self._bodyids_of_type(bt)
                for b in fam_cache[bt]:
                    if b in in_map or b in k_labeled \
                            or b in branch_pools.get(key, set()):
                        continue
                    k_labeled.add(b)
                    # family members are out-map bodyIds of the branch's
                    # OWN target type -> {T}(out-map) via `_leaf_token`.
                    fr = self._expansion_row(key, b, bt, 'family')
                    fr['candidate_annotation'] = self._leaf_token({}, bt)
                    family_rows.append(fr)
                # relative: this branch's candidate types outside the map
                for t in sorted(candidate_types.get(key, set())):
                    if t in in_map_types:
                        continue
                    if t not in rel_cache:
                        rel_cache[t] = self._bodyids_of_type(t)
                    for b in rel_cache[t]:
                        if b in in_map or b in k_labeled:
                            continue
                        k_labeled.add(b)
                        rr = self._expansion_row(key, b, t, 'relative')
                        rr['candidate_annotation'] = self._leaf_token({}, t)
                        relative_rows.append(rr)
        for r in family_rows + relative_rows:
            r['counts_toward_family_fill'] = True
            r['counts_toward_restrictive_fill'] = False
        self._family_rows = family_rows
        self._relative_rows = relative_rows

        # Finalize annotations + counts.
        for row, bid, ttype in evidence:
            bars = bars_by_key.get(key_of(row))
            row['bar_kind'] = bars.candidate_kind if bars else None
            row['bar_value'] = (bars.candidate_bar_value()
                                if bars else None)
            cat = str(row.get('category') or '')
            # Revision 3.12: every expansion leaf (family / candidates /
            # relative / suspicious) carries the per-bodyId token; the
            # order (out-map -> >src -> no_source -> untyped) is decided in
            # `_leaf_token`.
            if cat in ('family', 'candidates', 'relative', 'suspicious'):
                row['candidate_annotation'] = self._leaf_token(row, ttype)
            else:
                row.setdefault('candidate_annotation', '')
            if cat == 'candidates':
                row['counts_toward_restrictive_fill'] = True
                row['counts_toward_family_fill'] = True
                row['counts_toward_gap_fill'] = True
            else:
                row['counts_toward_restrictive_fill'] = False
                row['counts_toward_family_fill'] = cat in ('family',
                                                           'relative')
                row['counts_toward_gap_fill'] = False
        for d in pool_detail:
            d.setdefault('candidate_annotation', '')
            d['counts_toward_restrictive_fill'] = False
            d['counts_toward_family_fill'] = False
        # Fill rows: an IN-POOL fill pairs an unpaired neuron with an
        # accepted pool member (both directions, D1) — category
        # `paired_in_pool`; out-of-pool rows already classified above
        # carry their expansion category.
        for f in fills:
            f.setdefault('candidate_annotation', '')
            if f.get('fill_class') != 'in_pool':
                continue
            f['category'] = PAIRED_CATEGORY
            f['in_scope'] = True
            f['morph_failed'] = False
            f['counts_toward_restrictive_fill'] = True
            f['counts_toward_family_fill'] = True
            f['counts_toward_gap_fill'] = True

        # Rev 3.12 (plan open item 4): the mapper-gap report per TYPE —
        # candidate types with no backward mapping (annotation holes) and
        # the count of untyped candidates.  Runs AFTER the annotation
        # finalization above (it reads `candidate_annotation`).
        gap_types: Dict[str, int] = defaultdict(int)
        n_untyped = 0
        for row, _bid, ttype in evidence:
            if str(row.get('category') or '') != 'candidates':
                continue
            ann = str(row.get('candidate_annotation') or '')
            if ann == 'untyped' or ttype in (None, '', '?'):
                n_untyped += 1
            elif ann.endswith('(no_source)'):
                gap_types[str(ttype)] += 1
        self._mapper_gap_types = gap_types
        self._mapper_gap_untyped = n_untyped

        # Revision 3.12: propagate the query-level `dup` flag back onto the
        # per-branch rows (the plan's bodyId-level `(dup)` tag), so the
        # CSVs and the scene both show which expansion neurons recur across
        # branches.  Siblings are excluded by construction (never `dup`).
        dedup_rows = self._build_dedup_rows(evidence, pool_detail,
                                            family_rows, relative_rows)
        dup_by_bid = {int(r['target_bodyId']): bool(r['dup'])
                      for r in dedup_rows}
        for row, bid, _t in evidence:
            row['dup'] = dup_by_bid.get(bid, False)
        for d in pool_detail:
            d['dup'] = False
        for r in family_rows + relative_rows:
            r['dup'] = dup_by_bid.get(int(r['ahead_target_bodyId']), False)
        for f in fills:
            f['dup'] = dup_by_bid.get(
                int(f.get('proposal_bodyId') or -1), False)
        return dedup_rows

    def _expansion_row(self, key: Tuple[str, str], bid: int, tname: str,
                       category: str) -> Dict:
        src_type, tgt_type = key
        return {
            'query': next((p.query for p in self.pairs
                           if p.source_type == src_type
                           and p.target_type == tgt_type), ''),
            'source_type': src_type, 'target_type': tgt_type,
            'pool_basis': '', 'branch_linker_values': '',
            'branch_annotation': '', 'source_bodyId': None,
            'ahead_target_bodyId': int(bid),
            'ahead_target_type': tname,
            'ahead_metric': '', 'ahead_rank': None,
            'ahead_rank_union': None, 'ahead_jaccard': None,
            'best_pool_target_bodyId': None, 'best_pool_rank': None,
            'best_pool_rank_union': None, 'best_pool_jaccard': None,
            'ahead_size': None, 'pool_best_size': None,
            'size_ratio': None, 'size_filtered': False,
            'invader_class': '', 'invader_label': None,
            'category': category, 'in_scope': True, 'morph_failed': False,
            'candidate_annotation': '',
            'counts_toward_restrictive_fill': False,
            'counts_toward_family_fill': True,
            'candidate_source': category,
        }

    def _build_dedup_rows(self, evidence, pool_detail, family_rows,
                          relative_rows) -> List[Dict]:
        """S6: one bodyId-level row per query target, precedence by
        DEDUP_RANK (higher wins); `dup` set only for non-sibling repeats."""
        per_bid: Dict[int, Dict] = {}
        # `dup` counts DISTINCT BRANCHES, not rows: several source neurons
        # flagging the same target inside one branch is not a duplicate.
        branches: Dict[int, set] = defaultdict(set)

        def consider(bid, category, tname, src_type, tgt_type):
            if not str(category or ''):
                # out-of-scope / unlabelled rows are not part of the
                # query-level labeled rollup.
                return
            branches[bid].add((src_type, tgt_type))
            cur = per_bid.get(bid)
            if cur is None or dedup_category_rank(category) > \
                    dedup_category_rank(cur['dedup_category']):
                per_bid[bid] = {
                    'source_type': src_type, 'target_type': tgt_type,
                    'dedup_category': category,
                    'target_type_name': tname}

        for d in pool_detail:
            # An in-pool target's type IS the branch's target type.
            consider(int(d['target_bodyId']), d.get('category'),
                     str(d['target_type']), str(d['source_type']),
                     str(d['target_type']))
        for row, bid, tname in evidence:
            consider(bid, str(row.get('category') or ''),
                     str(tname or ''), str(row['source_type']),
                     str(row['target_type']))
        for r in list(family_rows) + list(relative_rows):
            consider(int(r['ahead_target_bodyId']), r['category'],
                     r['ahead_target_type'], r['source_type'],
                     r['target_type'])
        rows = []
        for bid, rec in per_bid.items():
            cat = rec['dedup_category'] or ''
            n_branches = len(branches[bid])
            # `dup` flags an EXPANSION-only repeat across branches: a
            # bodyId labelled in more than one branch whose rollup is
            # neither a tier nor a sibling.  In-map bodyIds
            # (tiers/siblings) are already mapped, so their repetition is
            # expected and never flagged (user 2026-09-13: "siblings are
            # always dup, so don't tag them").
            dup = n_branches > 1 and cat in (
                'candidates', 'family', 'relative', 'suspicious')
            rows.append({
                'target_bodyId': bid,
                'target_type': rec['target_type_name'],
                'dedup_category': cat,
                'n_branches': n_branches,
                'dup': bool(dup),
                'counts_toward_restrictive_fill': cat == 'candidates',
                'counts_toward_family_fill': cat in
                ('candidates', 'family', 'relative'),
            })
        rows.sort(key=lambda r: (-dedup_category_rank(r['dedup_category']),
                                 r['target_bodyId']))
        return rows

    # -- stage 5 ----------------------------------------------------------

    def run_morphology(self, val_rows: List[Dict], sus_rows: List[Dict],
                       fills: List[Dict],
                       pool_detail: Optional[List[Dict]] = None,
                       deep_rows: Optional[List[Dict]] = None,
                       null_rows: Optional[List[Dict]] = None) -> Dict:
        """Attach morph_v2_similarity / morph_nblast to the pairs that
        matter, self-calibrate, and (guard-rail permitting) gate
        verified_strong.  Never raises — morphology must not kill a run."""
        cfg = self.cfg
        out = {'calibrated': False, 'auc': None, 'threshold': None,
               'note': 'morphology disabled'}
        if not cfg.morph_enabled:
            return out
        pair_df = self._morph_pair_frame(val_rows, sus_rows, fills,
                                         deep_rows=deep_rows,
                                         null_rows=null_rows)
        if pair_df.empty:
            out['note'] = 'no pairs to score'
            return out
        try:
            from morphology import enrich_homolog_results
            enriched = enrich_homolog_results(
                pair_df, cfg.source_dataset, cfg.target_dataset,
                verbose=cfg.verbose)
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5] morphology enrichment failed: {exc}')
            out['note'] = f'enrichment failed: {exc}'
            return out

        score_col = 'morph_v2_similarity'
        if score_col not in enriched.columns:
            out['note'] = 'no morph columns returned'
            return out
        def _collect(col):
            vals = {}
            for r in enriched.itertuples(index=False):
                v = getattr(r, col, None)
                v = None if v is None or (isinstance(v, float)
                                          and np.isnan(v)) else float(v)
                vals[(int(r.source_bodyId), int(r.target_bodyId))] = v
            return vals
        scores = _collect(score_col)
        nblast = (_collect('morph_nblast')
                  if 'morph_nblast' in enriched.columns else {})

        # Rev 3.9: NULL-CALIBRATED Track-A bar.  The factor x
        # pooled-average threshold is arbitrary; the p95 of the Track-A
        # morph over jaccard<=null_jaccard_max window rows (provably
        # unrelated by connectivity) is the measured cross-dataset
        # baseline.  Fallback to the pooled-average bar when the null
        # sample is thin.
        null_morphs = [scores.get((int(r['source_bodyId']),
                                   int(r['ahead_target_bodyId'])))
                       for r in (null_rows or [])]
        null_morphs = [float(v) for v in null_morphs
                       if v is not None and not np.isnan(v)]
        if len(null_morphs) >= cfg.null_min_n:
            null_bar = float(np.percentile(null_morphs,
                                           cfg.null_percentile))
            null_bar_lo = float(np.median(null_morphs))
            self._track_a_null_bar = null_bar
            self._track_a_null_bar_lo = null_bar_lo
            out['track_a_null_bar'] = null_bar
            out['track_a_null_bar_lo'] = null_bar_lo
            out['track_a_null_n'] = len(null_morphs)
            out['track_a_null_source'] = (
                f'p{cfg.null_percentile:g} of jaccard<='
                f'{cfg.null_jaccard_max} window rows')
            self.log(f'[stage 5] Track-A null bar: p{cfg.null_percentile:g}'
                     f' = {null_bar:.3f} (n={len(null_morphs)} jaccard<='
                     f'{cfg.null_jaccard_max} rows)')
        else:
            self._track_a_null_bar = None
            self._track_a_null_bar_lo = None
            out['track_a_null_bar'] = None
            out['track_a_null_bar_lo'] = None
            out['track_a_null_n'] = len(null_morphs)
            self.log(f'[stage 5] Track-A null sample too thin '
                     f'(n={len(null_morphs)} < {cfg.null_min_n}); '
                     'falling back to the factor x pooled-average bar')

        # merge scores onto the pool detail rows, select the per-branch
        # reference tier (floors v3: matched+verified is THE reference tier
        # whenever it has >= 2 members — the matched-only preference is
        # deleted), and derive the per-branch admission bars from it.
        thresholds: Dict[Tuple[str, str], Optional[float]] = {}
        tiers: Dict[Tuple[str, str], Dict] = {}
        by_branch_rows: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        if pool_detail:
            for d in pool_detail:
                # best_source is None for pool members whose best evidence
                # is too weak to pair (BANC thin-annotation case) — the
                # pair simply has no Track-A score.
                bsid = d.get('best_source_bodyId')
                key = ((int(bsid), int(d['target_bodyId']))
                       if bsid is not None else None)
                d['morph_v2_similarity'] = scores.get(key) if key else None
                d['morph_nblast'] = nblast.get(key) if key else None
                by_branch_rows[(d['source_type'],
                                d['target_type'])].append(d)
            for key, rows in by_branch_rows.items():
                matched = [d for d in rows if d['category'] == 'matched']
                verified = [d for d in rows if d['category'] == 'verified']
                if len(matched) + len(verified) >= 2:
                    tier, ref_rows = ('matched+verified' if matched
                                      else 'verified-only',
                                      matched + verified)
                elif verified:
                    tier, ref_rows = ('verified (single ref)', verified)
                else:
                    tier, ref_rows = None, []
                tiers[key] = {'tier': tier, 'rows': ref_rows,
                              'ref_bids': [int(d['target_bodyId'])
                                           for d in ref_rows]}
            for key, t in tiers.items():
                vals = [d['morph_v2_similarity'] for d in t['rows']
                        if d.get('morph_v2_similarity') is not None]
                thresholds[key] = (self.cfg.candidate_morph_factor
                                   * float(np.mean(vals))) if vals else None
        self._candidate_thresholds = thresholds
        # Legacy admission rule text (floors v2) — superseded by the bar
        # engine below; kept because candidate_thresholds stays exported.
        out['candidate_morph_rule'] = (
            'LEGACY (floors v2): Track-A query morph >= factor x '
            'mean(reference-tier pool morph_v2); factor='
            f'{self.cfg.candidate_morph_factor}.  The v3 admission rule is '
            'the per-branch bar engine (see branch_bars / bar_params).')
        out['candidate_thresholds'] = {
            f'{s}->{t}': v for (s, t), v in sorted(thresholds.items())}
        pool_avg: Dict[Tuple[str, str], Optional[float]] = {}
        pool_avg_n: Dict[Tuple[str, str], int] = {}
        for (s, tt), tier in sorted(tiers.items()):
            vals = [d['morph_v2_similarity'] for d in tier['rows']
                    if d.get('morph_v2_similarity') is not None]
            pool_avg[(s, tt)] = (float(np.mean(vals)) if vals else None)
            pool_avg_n[(s, tt)] = len(vals)
        out['candidate_pool_avg_morph'] = {
            f'{s}->{tt}': v for (s, tt), v in sorted(pool_avg.items())}

        # ---- Revision 3.7 Track B: all-native pool reference --------
        # Invaders scored against their branch's matched+verified pool
        # natively (dataset-native v2 vector cache + native whitener) —
        # zero cross-dataset transforms, zero transform distortion.
        floors: Dict[Tuple[str, str], Optional[float]] = {}
        pool_ref_info: Dict[Tuple[str, str], Dict] = {}
        invader_scores: Dict[Tuple[Tuple[str, str], int],
                             Tuple[float, float]] = {}
        invaders_by_branch: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in list(sus_rows) + list(deep_rows or []):
            invaders_by_branch[(r['source_type'],
                                r['target_type'])].add(
                int(r['ahead_target_bodyId']))
        for f in fills:
            if f.get('side') == 'source' \
                    and f.get('fill_class') == 'out_of_pool':
                invaders_by_branch[(f['source_type'],
                                    f['target_type'])].add(
                    int(f['proposal_bodyId']))
        try:
            from morphology import (apply_whitening,
                                    find_similar_dataset_cache_v2,
                                    v2_similarity_matrix,
                                    DEFAULT_V2_BLOCK_WEIGHTS)
            ref_bids_all = sorted({b for t in tiers.values()
                                   for b in t['ref_bids']})
            need = sorted(set(ref_bids_all)
                          | {b for s in invaders_by_branch.values()
                             for b in s})
            if need:
                cache = find_similar_dataset_cache_v2(
                    cfg.target_dataset, verbose=False)
                X, ok, _ = cache.vectors_for(need, compute_missing=True)
                have = {bid: X[i] for i, bid in enumerate(need)
                        if ok[i]}
                wd = cache.load() or {}
                whiten = wd.get('whiten')
                if whiten is None:
                    raise RuntimeError('native whitener unavailable')

                def _sim(i_vec, j_vec):
                    wi = apply_whitening(whiten, i_vec.reshape(1, -1))
                    wj = apply_whitening(whiten, j_vec.reshape(1, -1))
                    # v2_similarity_matrix returns (scores, per-block);
                    # scores is a shape-(n_rows,) array.
                    return float(v2_similarity_matrix(
                        wi, wj.reshape(1, -1),
                        dict(DEFAULT_V2_BLOCK_WEIGHTS))[0][0])

                for key, t in tiers.items():
                    refs = [b for b in t['ref_bids'] if b in have]
                    baseline = floor = None
                    if len(refs) >= 2:
                        pair_sims = [_sim(have[a], have[b])
                                     for i, a in enumerate(refs)
                                     for b in refs[i + 1:]]
                        baseline = float(np.mean(pair_sims))
                        floor = baseline - cfg.pool_ref_floor_margin
                    floors[key] = floor
                    pool_ref_info[key] = {
                        'tier': t['tier'], 'baseline': baseline,
                        'floor': floor, 'n_refs': len(refs)}
                    for bid in invaders_by_branch.get(key, ()):
                        if bid not in have or not refs:
                            continue
                        sims = [_sim(have[bid], have[b])
                                for b in refs if b in have]
                        if sims:
                            invader_scores[(key, bid)] = (
                                max(sims), float(np.mean(sims)))
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5] track B vector-cache path unavailable '
                     f'({exc}); trying the render-frame fallback')
            try:
                invader_scores = self._track_b_render_fallback(
                    tiers, invaders_by_branch, invader_scores,
                    pool_ref_info, floors)
            except Exception as exc2:  # noqa: BLE001
                self.log(f'[stage 5] track B fallback failed: {exc2}')
        self._pool_ref_floors = floors
        for r in (list(sus_rows) + list(deep_rows or [])
                  + [f for f in fills if f.get('side') == 'source']):
            key = (r['source_type'], r['target_type'])
            bid = int(r.get('ahead_target_bodyId',
                            r.get('proposal_bodyId')))
            mx_mn = invader_scores.get((key, bid))
            r['morph_pool_ref'] = mx_mn[0] if mx_mn else None
            r['morph_pool_ref_mean'] = mx_mn[1] if mx_mn else None
            r['pool_ref_tier'] = pool_ref_info.get(key, {}).get('tier')
            key_a = (int(r.get('source_bodyId', r.get('bodyId'))), bid)
            r.setdefault('morph_v2_similarity', None)
            if r.get('morph_v2_similarity') is None:
                r['morph_v2_similarity'] = scores.get(key_a)
        out['pool_ref_tiers'] = {
            f'{s}->{t}': info.get('tier')
            for (s, t), info in sorted(pool_ref_info.items())}
        out['pool_ref_baselines'] = {
            f'{s}->{t}': info.get('baseline')
            for (s, t), info in sorted(pool_ref_info.items())}
        out['pool_ref_floors'] = {
            f'{s}->{t}': info.get('floor')
            for (s, t), info in sorted(pool_ref_info.items())}

        # ---- Floors v3: per-branch admission bars (plan-unified-
        # morph-qualification-bars).  One BarSet per branch drives EVERY
        # admission decision (candidates, siblings, aggressive suspicious)
        # through candidate_qualified / suspicious_qualified; native binds
        # when the m+v reference tier has >=2 scored members, the Track-A
        # backup (B_b − Δ) takes over below that, and the run null is the
        # last resort.  random_null_floor is None here: the pipeline does
        # not score a random sample (the clamp is available to consumers
        # that do, e.g. Find Homolog's per-source nulls).
        branch_bars: Dict[Tuple[str, str], BarSet] = {}
        for key in sorted(tiers):
            info = pool_ref_info.get(key, {})
            branch_bars[key] = compute_branch_bars(
                ref_pool_sim_mean=info.get('baseline'),
                pool_track_a_baseline=pool_avg.get(key),
                n_native_refs=info.get('n_refs', 0),
                n_scored_pool=pool_avg_n.get(key, 0),
                null_bar=getattr(self, '_track_a_null_bar', None),
                null_bar_lo=getattr(self, '_track_a_null_bar_lo', None),
                random_null_floor=None,
                track_a_offset=cfg.morph_track_a_offset,
                native_margin=cfg.pool_ref_floor_margin,
                suspicious_level=cfg.morph_suspicious_level)
        self._branch_bars = branch_bars
        out['bar_params'] = {
            'track_a_offset': cfg.morph_track_a_offset,
            'native_margin': cfg.pool_ref_floor_margin,
            'suspicious_level': cfg.morph_suspicious_level,
            'rule': 'floors v3 (plan-unified-morph-qualification-bars): '
                    'candidate = native m+v floor (>=2 refs) else '
                    f'B_b - Δ else null p{cfg.null_percentile:g}; suspicious '
                    '(aggressive deep window) = B_b - k*Δ else null p50'}
        out['branch_bars'] = {
            f'{s}->{t}': {
                'candidate_kind': b.candidate_kind,
                'native_floor': b.native_floor,
                'backup_floor': b.backup_floor,
                'suspicious_floor': b.suspicious_floor,
                'suspicious_kind': b.suspicious_kind,
                'pool_track_a_baseline': b.pool_track_a_baseline,
                'n_native_refs': b.n_native_refs,
                'n_scored_pool': b.n_scored_pool,
            } for (s, t), b in sorted(branch_bars.items())}
        try:
            from visualize_skeleton import dataset_render_space
            render_space = dataset_render_space(cfg.target_dataset)
        except Exception:  # noqa: BLE001
            render_space = f'render space of {cfg.target_dataset}'
        # Frame disclosure (Rev 3.7): both tracks score in TARGET
        # coordinates; Stage-4 scenes render in SOURCE coordinates.
        out['score_frame'] = {
            'track_a': (f'target render space ({render_space}); the '
                        'source neurons are transformed in'),
            'track_b': 'target native space; no transforms',
            'visualization': (f'source template ({cfg.source_dataset}); '
                              'targets bridged into source '
                              'coordinates — different frame from '
                              'the scores'),
        }
        out['size_ratio_cutoff'] = cfg.target_min_size_ratio
        out['suspicious_ru_margin'] = cfg.suspicious_ru_margin

        def _merge(row):
            sid = row.get('source_bodyId', row.get('bodyId'))
            tid = (row.get('target_bodyId')
                   if row.get('target_bodyId') is not None
                   else row.get('proposal_bodyId',
                                row.get('ahead_target_bodyId')))
            key = (int(sid), int(tid) if tid is not None else -1)
            row['morph_v2_similarity'] = scores.get(key)
            row['morph_nblast'] = nblast.get(key)
        for row in val_rows + fills + sus_rows:
            _merge(row)
        # finalize same-type fill counting (Rev 3.11): qualified by the
        # branch's morph rule (threshold or binding floor)
        for f in fills:
            if f.get('invader_class') != 'same-type':
                continue
            key = (f['source_type'], f['target_type'])
            thr = thresholds.get(key)
            floor = floors.get(key)
            m = f.get('morph_v2_similarity')
            pr = f.get('morph_pool_ref')
            q = False
            if thr is not None and m is not None and not np.isnan(m) \
                    and m >= thr:
                q = True
            if floor is not None and pr is not None and not np.isnan(pr) \
                    and pr >= floor:
                q = True
            f['counts_toward_gap_fill'] = bool(q)
            f['fill_qualified'] = bool(q)

        verified = [r['morph_v2_similarity'] for r in val_rows
                    if r['verdict'] == 'verified_strong'
                    and r.get('morph_v2_similarity') is not None]
        suspicious = [r['morph_v2_similarity'] for r in sus_rows
                      if r.get('morph_v2_similarity') is not None]
        auc = morph_auc(verified, suspicious)
        threshold = (best_threshold(verified, suspicious)
                     if auc is not None else None)
        out.update(auc=auc, threshold=threshold,
                   n_verified_scored=len(verified),
                   n_suspicious_scored=len(suspicious))
        if auc is not None and threshold is not None \
                and auc >= cfg.morph_auc_floor:
            out['calibrated'] = True
            out['note'] = 'gate ACTIVE'
            demoted = 0
            for row in val_rows:
                if row['verdict'] != 'verified_strong':
                    continue
                ms = row.get('morph_v2_similarity')
                if ms is not None and not np.isnan(ms) and ms < threshold:
                    row['verdict'] = 'verified'
                    row['flags'] = ';'.join(
                        filter(None, [row['flags'], 'morph_below_threshold']))
                    demoted += 1
            out['demoted'] = demoted
        else:
            out['note'] = (
                'gate INACTIVE (informational only): '
                + ('insufficient scored pairs' if auc is None else
                   f'AUC {auc:.3f} < floor {cfg.morph_auc_floor}'))
        return out

    def _track_b_render_fallback(self, tiers, invaders_by_branch,
                                 invader_scores, pool_ref_info, floors):
        """Track-B fallback (Rev 3.7): score invader-vs-reference pairs
        through compute_morph_similarity_vs_queries with the pool
        skeletons pre-transformed native->render (the render-frame
        whitener stage 5 already builds).  Bounded by pool_ref_cap and
        the invader universe; per-branch baselines/floors mirror the
        vector-cache path."""
        from morphology import (compute_morph_similarity_vs_queries,
                                _load_cached_skeleton_file,
                                fetch_skeleton_on_demand)
        from visualize_skeleton import (dataset_native_space,
                                        dataset_render_space,
                                        transform_neurons_to_space)
        import navis

        cfg = self.cfg
        native = dataset_native_space(cfg.target_dataset)
        render = dataset_render_space(cfg.target_dataset)

        def load(bids):
            out = {}
            skel_dir = (Path(__file__).resolve().parents[2] / 'cache' /
                        cfg.target_dataset.replace(':', '_')
                        .replace('.', '_') / 'skeletons' /
                        'raw_skeletons')
            for b in bids:
                try:
                    path = skel_dir / f'{int(b)}.swc.zst'
                    nrn = (_load_cached_skeleton_file(path)
                           if path.exists() else
                           fetch_skeleton_on_demand(cfg.target_dataset,
                                                    int(b)))
                    if nrn is not None:
                        out[int(b)] = nrn
                except Exception:  # noqa: BLE001
                    continue
            return out

        for key, t in tiers.items():
            ref_bids = [b for b in t['ref_bids']
                        ][:cfg.pool_ref_cap]
            refs_native = load(ref_bids)
            if len(refs_native) < 1:
                pool_ref_info[key] = {'tier': t['tier'], 'baseline': None,
                                      'floor': None,
                                      'n_refs': len(refs_native)}
                floors[key] = None
                continue
            ref_items = list(refs_native.items())
            if native != render:
                xformed = transform_neurons_to_space(
                    navis.NeuronList([n for _, n in ref_items]),
                    native, render, validate_bounds=False, verbose=False)
                # transform_neurons_to_space drops failures; track ids by
                # node-count match is unreliable — keep only full sets
                if len(xformed) != len(ref_items):
                    self.log('[stage 5] track B fallback: some reference '
                             'skeletons failed the render transform; '
                             'skipping the branch')
                    continue
                ref_items = [(b, n) for (b, _), n in
                             zip(ref_items, xformed)]
            ref_ids = [b for b, _ in ref_items]
            refs_render = [n for _, n in ref_items]
            inv = sorted(invaders_by_branch.get(key, ()))
            inv_native = load(inv)
            if not refs_render or not inv_native:
                continue
            pair_df = compute_morph_similarity_vs_queries(
                refs_render, sorted(inv_native),
                cfg.target_dataset, verbose=False, query_bids=ref_ids,
                source_dataset=cfg.target_dataset)
            sims = {}
            for r in pair_df.itertuples(index=False):
                try:
                    sims.setdefault(int(r.target_bodyId), {})[
                        int(r.source_bodyId)] = float(r.morph_v2_similarity)
                except (TypeError, ValueError):
                    continue
            # baseline: mean pairwise self-similarity among refs
            baseline = floor = None
            if len(ref_ids) >= 2:
                ps = [sims.get(b, {}).get(a)
                      for i, a in enumerate(ref_ids)
                      for b in ref_ids[i + 1:]]
                ps = [v for v in ps if v is not None
                      and not np.isnan(v)]
                if ps:
                    baseline = float(np.mean(ps))
                    floor = baseline - cfg.pool_ref_floor_margin
            floors[key] = floor
            pool_ref_info[key] = {'tier': t['tier'], 'baseline': baseline,
                                  'floor': floor, 'n_refs': len(ref_ids)}
            for bid in inv:
                vs = [v for v in sims.get(bid, {}).values()
                      if v is not None and not np.isnan(v)]
                if vs:
                    invader_scores[(key, bid)] = (max(vs),
                                                  float(np.mean(vs)))
        return invader_scores

    def _morph_pair_frame(self, val_rows, sus_rows, fills,
                          pool_pairs=None,
                          deep_rows: Optional[List[Dict]] = None,
                          null_rows: Optional[List[Dict]] = None
                          ) -> pd.DataFrame:
        cfg = self.cfg
        recs: List[Tuple[int, int, str]] = []
        seen = set()

        def add(sid, tid, kind):
            key = (int(sid), int(tid))
            if key not in seen and tid is not None:
                seen.add(key)
                recs.append((key[0], key[1], kind))

        for row in val_rows:
            if row['verdict'] in ('verified_strong', 'verified',
                                  'borderline') and \
                    row.get('target_bodyId') is not None:
                add(row['source_bodyId'], row['target_bodyId'], 'assigned')
        per_src: Dict[int, int] = Counter()
        for row in sus_rows:
            sid = int(row['source_bodyId'])
            if per_src[sid] < cfg.candidate_morph_cap:
                per_src[sid] += 1
                add(sid, int(row['ahead_target_bodyId']), 'suspicious')
        for row in (deep_rows or []):
            sid = int(row['source_bodyId'])
            if per_src[sid] < cfg.candidate_morph_cap:
                per_src[sid] += 1
                add(sid, int(row['ahead_target_bodyId']), 'deep')
        for row in (null_rows or []):
            add(row['source_bodyId'], row['ahead_target_bodyId'], 'null')
        for row in fills:
            add(row['bodyId'], row['proposal_bodyId'], 'fill')
        # pool neurons (for the candidate morph threshold: the average
        # verified+matched+borderline pool similarity)
        for sid, tid in (pool_pairs or []):
            add(sid, tid, 'pool')
        return pd.DataFrame(recs, columns=['source_bodyId',
                                           'target_bodyId', 'pair_kind'])

    # -- Revision 3.10: set-level coverage --------------------------------

    def _set_coverage_payload(self, coverage) -> Dict:
        return coverage

    # -- driver -----------------------------------------------------------

    # -- Plan I §1: stage-2 target-profile pre-flight --------------------

    NON_NEURON_STATUS = frozenset({'UNROOTED', 'TOO_SMALL',
                                   'GLIA,NOT_A_NEURON',
                                   'ALIGNMENT_CELL_TYPE'})

    def _typed_target_universe(self, dataset: str) -> List[int]:
        """Typed bodyId universe of a local dataset, from ONE neuron-table
        read (a per-type get_bodyids_for_type loop re-reads the table per
        type and is the known bottleneck)."""
        import pandas as pd
        from comparison.connectivity_profiler import canonical_dataset_name
        safe = (canonical_dataset_name(dataset)
                .replace(':', '_').replace('.', '_'))
        base = (Path(__file__).resolve().parents[2] / 'datasets' / safe)
        table = None
        for name in (f'{safe}_allneurons_neuron_df.parquet',
                     f'{safe}_allneurons_neuron_df.csv'):
            if (base / name).exists():
                table = base / name
                break
        if table is None:
            return []
        # Column tolerance: datasets ship different schemas (FAFB has no
        # `status` column; BANC/male-cns do).  Filter on what exists.
        if table.suffix == '.parquet':
            import pyarrow.parquet as pq
            cols = set(pq.ParquetFile(table).schema_arrow.names)
            use = [c for c in ('bodyId', 'type', 'status') if c in cols]
            nt = pd.read_parquet(table, columns=use)
        else:
            use = [c for c in ('bodyId', 'type', 'status')]
            nt = pd.read_csv(table, low_memory=False)
            nt = nt[[c for c in use if c in nt.columns]]
        nt['bodyId'] = nt['bodyId'].astype(int)
        keep = nt['type'].notna() & (nt['type'] != 'Unknown') \
            & (nt['type'] != '')
        if 'status' in nt.columns:
            keep = keep & (~nt['status'].isin(self.NON_NEURON_STATUS))
        return sorted(nt.loc[keep, 'bodyId'].tolist())

    def _preflight_target_profiles(self, dataset: str, universe=None,
                                   cached_ids=None) -> Dict[str, int]:
        """Build missing target-side bodyId profiles through the profiler
        backend (cache-first per bid, resumable; batch-save + consolidate).

        Returns ``{'universe', 'cached', 'built'}``.  Any failure is
        fail-open: the run continues against whatever the cache holds
        (the historical cache-only behavior)."""
        cfg = self.cfg
        if universe is None:
            universe = self._typed_target_universe(dataset)
        if not universe:
            self.log('[TMVEV] profile pre-flight: no typed universe '
                     f'resolvable for {dataset}; staying cache-only')
            return {'universe': 0, 'cached': 0, 'built': 0}
        if cached_ids is None:
            cache_df = self.profiler._load_cache_dataframe(dataset)
            cached_ids = set()
            if cache_df is not None and 'neuron_id' in cache_df.columns:
                cached_ids = {int(x) for x in cache_df['neuron_id']}
        missing = [b for b in universe if b not in set(cached_ids)]
        stats = {'universe': len(universe), 'cached': len(universe)
                 - len(missing), 'built': 0}
        self.log(f'[TMVEV] profile pre-flight ({dataset}): universe '
                 f'{len(universe)}, cached {stats["cached"]}, '
                 f'building {len(missing)}')
        self.progress.emit('profiles_progress', stage='2', dataset=dataset,
                           done=0, total=len(missing))
        if not missing:
            self.progress.emit('profiles_progress', stage='2',
                               dataset=dataset, done=0, total=0,
                               note='cache complete')
            return stats
        t0 = time.time()
        batch = {}
        built = 0
        for i, bid in enumerate(missing, 1):
            try:
                prof = self.profiler.get_profile(bid, dataset)
                if prof is not None:
                    batch[int(bid)] = prof
                    built += 1
            except Exception as exc:  # noqa: BLE001
                self.log(f'[TMVEV] profile build failed for {bid}: {exc}')
            if len(batch) >= 100:
                self.profiler._save_profiles_to_cache_batch(
                    batch, dataset, silent=True)
                batch.clear()
            if i % 500 == 0 or i == len(missing):
                self.log(f'[TMVEV] profile pre-flight: {i}/{len(missing)} '
                         f'({100 * i / len(missing):.1f}%) elapsed '
                         f'{time.time() - t0:.0f}s')
                self.progress.emit('profiles_progress', stage='2',
                                   dataset=dataset, done=i,
                                   total=len(missing))
        if batch:
            self.profiler._save_profiles_to_cache_batch(batch, dataset,
                                                        silent=True)
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception as exc:  # noqa: BLE001
            self.log(f'[TMVEV] profile consolidation failed: {exc}')
        stats['built'] = built
        self.progress.emit('profiles_progress', stage='2', dataset=dataset,
                           done=len(missing), total=len(missing),
                           note='complete')
        self.log(f'[TMVEV] profile pre-flight done: built {built} in '
                 f'{time.time() - t0:.0f}s')
        return stats

    # -- Plan I follow-up: out-map expansion -----------------------------

    def _expand_out_map_sources(self, out_map_by_type, in_map, target_stats,
                                target_bids, target_id2type, top_k):
        """Scan every out-map (unpaired) source neuron against the full
        target universe and keep the top-k connectivity-ranked targets that
        are NOT in-map claims.  Connectivity-only evidence: no morph bars
        apply at this stage (the scene layer and out_map_expansion.csv are
        exploratory surfaces, never fills)."""
        cfg = self.cfg
        rows = []
        t0 = time.time()
        total = sum(len(v) for v in out_map_by_type.values())
        done = 0
        for (query, src_type), bids in sorted(out_map_by_type.items()):
            for bid in bids:
                done += 1
                try:
                    sp = self.profiler.get_profile(bid, cfg.source_dataset)
                    if sp is None or sp.connectivity_status.name \
                            in SOURCE_STATUS_SKIP:
                        continue
                    df = scan_source(
                        expanded_vector(sp, self.mapper), target_stats,
                        target_bids)
                except Exception as exc:  # noqa: BLE001
                    self.log(f'[TMVEV] out-map expansion {bid}: {exc}')
                    continue
                if df is None or df.empty:
                    continue
                df = df.sort_values(['rank_union_rank', 'jaccard_rank'],
                                    na_position='last')
                kept = 0
                for r in df.itertuples(index=False):
                    tgt = int(r.target_bid)
                    if tgt in in_map:
                        continue          # in-map claims are not new finds
                    # Typed targets only: an untyped (NaN/'Unknown') neuron
                    # can never enter the mapping, so it is noise here —
                    # orphan fragments otherwise dominate the top ranks of
                    # saturated-type sources.
                    tt = target_id2type.get(tgt)
                    if (tt is None or str(tt) in ('?', 'nan', 'None',
                                                  'Unknown')
                            or (isinstance(tt, float) and tt != tt)):
                        continue
                    kept += 1
                    rows.append({
                        'query': query,
                        'source_type': src_type,
                        'source_bodyId': bid,
                        'target_bodyId': tgt,
                        'target_type': target_id2type.get(tgt, '?'),
                        'rank_union': _f(getattr(r, 'rank_union', None)),
                        'rank_union_rank': _f(getattr(r, 'rank_union_rank',
                                                      None)),
                        'jaccard': _f(getattr(r, 'jaccard', None)),
                        'jaccard_rank': _f(getattr(r, 'jaccard_rank', None)),
                        'in_map': False,
                    })
                    if kept >= top_k:
                        break
                if done % 25 == 0 or done == total:
                    self.log(f'[TMVEV] out-map expansion: {done}/{total} '
                             f'sources scanned, {len(rows)} candidate rows')
                    self.progress.emit('out_map_progress', stage='expansion',
                                       done=done, total=total)
        self.log(f'[TMVEV] out-map expansion done: {len(rows)} candidate '
                 f'row(s) for {total} source(s) in '
                 f'{time.time() - t0:.0f}s')
        # Morph qualification for the expansion candidates (Track-A): the
        # run null bar is the noise floor — exploratory pairs below it
        # FAIL (e.g. photoreceptor/orphan captures).  Bounded by the
        # per-source cap (top_k rows), NBLAST off.
        if rows and getattr(self, '_track_a_null_bar', None) is not None:
            try:
                import pandas as pd
                from morphology import enrich_homolog_results
                pair_df = pd.DataFrame(
                    [{'source_bodyId': r['source_bodyId'],
                      'target_bodyId': r['target_bodyId'],
                      'pair_kind': 'out_map'} for r in rows])
                enr = enrich_homolog_results(
                    pair_df, cfg.source_dataset, cfg.target_dataset,
                    verbose=False, compute_nblast=False)
                sc = {}
                for er in enr.itertuples(index=False):
                    v = getattr(er, 'morph_v2_similarity', None)
                    if v is not None and not (isinstance(v, float)
                                              and v != v):
                        sc[(int(er.source_bodyId),
                            int(er.target_bodyId))] = float(v)
                n_pass = 0
                for r in rows:
                    key = (int(r['source_bodyId']), int(r['target_bodyId']))
                    r['morph_v2_similarity'] = sc.get(key)
                    r['morph_qualified'] = bool(
                        r['morph_v2_similarity'] is not None
                        and r['morph_v2_similarity'] >= self._track_a_null_bar)
                    n_pass += r['morph_qualified']
                self.log(f'[TMVEV] out-map expansion morph check: '
                         f'{n_pass}/{len(rows)} pass the null bar '
                         f'({self._track_a_null_bar:.3f})')
                self.progress.emit('out_map_progress', stage='expansion',
                                   note=f'morph {n_pass}/{len(rows)} pass')
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[TMVEV] out-map morph check failed: {exc}')
                self.log(traceback.format_exc())
        return rows

    def run(self) -> Path:
        cfg = self.cfg
        t_start = time.time()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base = Path(cfg.output_dir) if cfg.output_dir else \
            Path(__file__).resolve().parents[2] / 'local_data' / \
            'mapping_validation'
        # The analysis ROOT folder carries the type-map prefix (the
        # scene folders inside keep their own plot-3d naming).
        self.run_dir = base / (
            f'type-map_{_abbrev(cfg.source_dataset)}_to_'
            f'{_abbrev(cfg.target_dataset)}'
            f'_{cfg.run_label}_{stamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressReporter(self.run_dir)
        self.progress.emit('run_start', source_dataset=cfg.source_dataset,
                           target_dataset=cfg.target_dataset,
                           query_types=list(cfg.query_types),
                           mode=cfg.effective_mode)

        self.progress.emit('stage_start', stage='1')
        self.log(f'[stage 1] resolving type pairs '
                 f'{cfg.source_dataset} -> {cfg.target_dataset}')
        self.pairs = self.resolve_type_pairs()
        if not self.pairs:
            self.log('no valid type pairs resolved; nothing to validate')
            self._write_outputs([], [], [], [], None, None, [], [], [])
            return self.run_dir

        self.progress.emit('stage_done', stage='1')
        self.progress.emit('stage_start', stage='2')
        self.log(f'[stage 2] building target expanded-type vectors '
                 f'({cfg.target_dataset})')
        if not cfg.skip_profile_build:
            try:
                self._preflight_target_profiles(cfg.target_dataset)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[TMVEV] profile pre-flight failed '
                         f'(continuing cache-only): {exc}')
                self.log(traceback.format_exc())
                self.progress.emit('warning', stage='2',
                                   note=f'profile pre-flight failed: {exc}')
        vectors = build_target_vectors(
            self.profiler, cfg.target_dataset, self.mapper, cfg.verbose,
            min_weight=cfg.target_min_weight,
            min_partner_types=cfg.target_min_partner_types)
        target_stats = prep_target_stats(vectors)
        target_bids = list(target_stats)
        target_id2type = self._target_types(target_bids)
        # Revision 3.6 evidence maps: spatial caliber (PRIMARY noise
        # filter) and the source type/additional_type counts (backward
        # home reality check).
        self._target_sizes = load_caliber_map(cfg.target_dataset)
        self._source_sizes = load_caliber_map(cfg.source_dataset)
        self._target_sides = load_hemisphere_map(cfg.target_dataset)
        self._source_sides = load_hemisphere_map(cfg.source_dataset)
        self._source_type_counts, self._source_add_counts = \
            load_source_type_counts(cfg.source_dataset)
        self._target_weights = {bid: s.sum1 for bid, s in
                                target_stats.items()}
        if self._target_sizes:
            self.log(f'[stage 2] caliber map: {len(self._target_sizes)} '
                     'target neurons (spatial size)')
        else:
            self.log('[stage 2] caliber map unavailable — spatial gate '
                     'falls back to the expanded-weight ratio')

        # group pairs by source type so each source neuron is scanned once
        by_src_type: Dict[str, List[TypePair]] = defaultdict(list)
        for pair in self.pairs:
            by_src_type[pair.source_type].append(pair)

        all_val_rows: List[Dict] = []
        all_sus_rows: List[Dict] = []
        all_noise_rows: List[Dict] = []
        all_deep_rows: List[Dict] = []
        all_null_rows: List[Dict] = []
        summaries: List[Dict] = []
        all_fills: List[Dict] = []
        all_pool_detail: List[Dict] = []
        per_pair_res = {}
        for n_type, (src_type, plist) in enumerate(
                sorted(by_src_type.items()), 1):
            # Revision 2: branches of one parent may hold DISJOINT source
            # sub-pools — scan the union once and share it.
            pool = sorted({b for p in plist for b in p.source_pool})
            self.log(f'[stage 2] source type {n_type}/'
                     f'{len(by_src_type)}: {src_type} — scanning '
                     f'{len(pool)} source neuron(s) against '
                     f'{len(target_bids)} targets '
                     f'({len(plist)} branches)')
            scans: Dict[int, pd.DataFrame] = {}
            t0 = time.time()
            for sbid in pool:
                sp = self.profiler.get_profile(sbid, cfg.source_dataset)
                if sp is None or \
                        sp.connectivity_status.name in SOURCE_STATUS_SKIP:
                    continue
                scans[sbid] = scan_source(
                    expanded_vector(sp, self.mapper), target_stats,
                    target_bids)
            self.log(f'    scans done in {time.time() - t0:.0f}s '
                     f'({len(scans)}/{len(pool)} valid)')
            for pair in plist:
                res = self.validate_pair(pair, scans, target_id2type,
                                         sizes=self._target_sizes,
                                         weights=self._target_weights,
                                         source_sides=self._source_sides,
                                         target_sides=self._target_sides)
                per_pair_res[pair.key] = res
                all_val_rows.extend(res['rows'])
                all_sus_rows.extend(res['suspicious'])
                all_noise_rows.extend(res['noise'])
                all_deep_rows.extend(res['deep'])
                all_null_rows.extend(res['null'])
                summaries.append(res['summary'])
                all_fills.extend(res['fills'])
                all_pool_detail.extend(res['pool_detail'])
                self.log(f'  pair {pair.source_type} -> '
                         f'{pair.target_type} [{pair.status}]: '
                         f'verdicts: '
                         f'{_counter_text("verdict", res["rows"])}; '
                         f'M={res["summary"]["matched"]}, '
                         f'gap={res["summary"]["gap"]} '
                         f'({res["summary"]["gap_ratio"]:.0%}), '
                         f'triggered={res["summary"]["gap_triggered"]}, '
                         f'suspicious rows={len(res["suspicious"])}, '
                         f'noise filtered={len(res["noise"])}')
            del scans

        # Revision 3.6: classify every invader BEFORE morph promotion —
        # structural facts (sibling pools, collapsed chains, real backward
        # homes) outrank morphology.  Deep-window candidates get the same
        # treatment (a deep sibling/backward member is structural, not a
        # candidate).
        family_types = {p.source_type for p in self.pairs}
        self.annotate_invaders(all_sus_rows + all_deep_rows, all_fills,
                               per_pair_res, family_types=family_types)

        # Revision 3.11 relatives: expand the type-mates of
        # hollow/unmapped candidate types (annotation-review targets --
        # the ranking never surfaced them, but their siblings are
        # Revision 3.12: relatives are now the `relative` CATEGORY bin
        # (candidate-type mates outside the map), computed by
        # `finalize_categories` after morphology.  relatives.csv is built
        # from `_relative_rows` below; the legacy `_relatives_by_type`
        # map is left empty so the scene no longer double-renders them.
        relatives_by_type: Dict[str, List[int]] = {}
        self._relatives_by_type = relatives_by_type

        morph_info = None
        if cfg.morph_enabled:
            self.log('[stage 5] morphology verification + self-calibration')
            try:
                morph_info = self.run_morphology(all_val_rows, all_sus_rows,
                                                 all_fills, all_pool_detail,
                                                 deep_rows=all_deep_rows,
                                                 null_rows=all_null_rows)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[stage 5] morphology stage failed: {exc}')
                self.log(traceback.format_exc())
                morph_info = {'note': f'stage failed: {exc}'}
            self.log(f'    {morph_info}')

        # Revision 3.12: compute the category partition + query-level
        # dedup AFTER morphology (qualification is now known).  This sets
        # `category`/`in_scope`/`morph_failed`/annotation/counts on every
        # evidence row and returns the dedup rows.
        dedup_rows: List[Dict] = []
        family_rows_out: List[Dict] = []
        relatives_out: List[Dict] = []
        try:
            dedup_rows = self.finalize_categories(
                per_pair_res, all_sus_rows, all_deep_rows, all_fills,
                all_pool_detail)
            # relatives.csv / family_candidates.csv must cover the WHOLE
            # bin, not just the enumerated members: a deep evidence row can
            # itself be classified `relative`/`family` (aggressive mode), so
            # union the enumerated rows with the evidence rows carrying
            # those categories (deduplicated per branch+bodyId).
            fam_evidence = [r for r in (all_deep_rows + all_sus_rows)
                            if str(r.get('category') or '') == 'family']
            rel_evidence = [r for r in (all_deep_rows + all_sus_rows)
                            if str(r.get('category') or '') == 'relative']
            family_rows_out = (_dedup_rows_by_bid(
                fam_evidence + list(getattr(self, '_family_rows', []) or [])))
            # Same contract as family_candidates.csv: full per-branch rows,
            # deduplicated per branch+bodyId (branch provenance, evidence
            # metrics and the `dup` tag all travel with the row).
            relatives_out = (_dedup_rows_by_bid(
                rel_evidence + list(getattr(self, '_relative_rows', []) or [])))
            self.log(f'[categories] mode={self.cfg.effective_mode}: '
                     + ', '.join(
                         f'{k or "out_of_scope"}={v}'
                         for k, v in Counter(
                             (r.get('category') or 'out_of_scope')
                             for r in all_sus_rows).most_common()))
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(f'[categories] failed: {exc}')
            self.log(traceback.format_exc())

        # Revision 3.10: set-level coverage — the deliverable for
        # "how much of the source→target mapping is validated, proposed,
        # and still missing" (per-branch gaps are diagnostics).  Computed
        # AFTER finalize_categories so the fills it counts carry the final
        # category-qualified `counts_toward_gap_fill` (Rev 3.12).
        coverage = None
        try:
            coverage = compute_set_coverage(self.pairs, per_pair_res,
                                            all_fills,
                                            evidence_rows=all_sus_rows
                                            + all_deep_rows)
            self._set_coverage = coverage
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(f'[set coverage] failed: {exc}')
            self.log(traceback.format_exc())
            self._set_coverage = None

        # Floors v3: the layered gap-fill report (one row per fill-level
        # bodyId; the summary also lands in set_coverage.json).
        gap_levels: List[Dict] = []
        try:
            gap_levels, gap_by_level = build_gap_fill_levels(
                dedup_rows,
                list(all_sus_rows) + list(all_deep_rows) + list(all_fills),
                (coverage or {}).get('mcns', {}).get('family_material', []))
            if coverage is not None:
                coverage['gap_fill_by_level'] = dict(gap_by_level)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(f'[gap fill levels] failed: {exc}')
            self.log(traceback.format_exc())

        # Plan I follow-up: out-map expansion — scan the unpaired source
        # neurons (the scene's out-map branch) for connectivity-ranked
        # candidates outside the in-map claims.
        out_map_rows: List[Dict] = []
        out_map_by_type = compute_out_map_by_type(self.pairs)
        n_out_map = sum(len(v) for v in out_map_by_type.values())
        if cfg.skip_out_map_expansion:
            self.log('[TMVEV] out-map expansion skipped '
                     '(--skip-out-map-expansion)')
        elif not n_out_map:
            self.log('[TMVEV] out-map expansion: no unclaimed sources — '
                     'every source neuron is claimed by a branch '
                     '(nothing to scan)')
        elif n_out_map:
            in_map = set()
            for res in per_pair_res.values():
                in_map |= {int(b) for b in (res.get('_pool_set') or [])}
            try:
                out_map_rows = self._expand_out_map_sources(
                    out_map_by_type, in_map, target_stats, target_bids,
                    target_id2type, cfg.out_map_top_k)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[TMVEV] out-map expansion failed: {exc}')
                self.log(traceback.format_exc())

        # The scene reads the expansion rows — expose them BEFORE stage 4.
        self._out_map_expansion_rows = out_map_rows
        self._out_map_by_type = out_map_by_type

        if cfg.visualize:
            try:
                from comparison.mapping_validation_visualize import \
                    render_pair_scenes
                render_pair_scenes(self, per_pair_res)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[stage 4] visualization failed: {exc}')
                self.log(traceback.format_exc())

        # Export contract (Rev 3.6/2c): source caliber travels with the
        # validation rows.
        for r in all_val_rows:
            r['source_size'] = (self._source_sizes or {}).get(
                int(r['source_bodyId']))
        self._write_outputs(all_val_rows, all_sus_rows, summaries, all_fills,
                            morph_info, all_pool_detail, all_noise_rows,
                            all_deep_rows, set_coverage=coverage,
                            dedup_rows=dedup_rows,
                            relatives=relatives_out,
                            family_rows=family_rows_out,
                            gap_levels=gap_levels,
                            out_map_rows=out_map_rows)
        self.progress.emit('run_done',
                           run_dir=str(self.run_dir),
                           elapsed_s=round(time.time() - t_start, 1))
        self.log(f'done in {time.time() - t_start:.0f}s -> {self.run_dir}')
        return self.run_dir

    def _target_types(self, target_bids: List[int]) -> Dict[int, str]:
        """Type label per target bodyId (for suspicious/fill reporting)."""
        id2type: Dict[int, str] = {}
        if not target_bids:
            return id2type
        try:
            got = self.profiler.get_types_for_bodyids(
                target_bids, self.cfg.target_dataset) or {}
            for k, v in got.items():
                id2type[int(k)] = v
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! target type labels unavailable: {exc}')
        return id2type

    # -- outputs ----------------------------------------------------------

    def _write_outputs(self, val_rows, sus_rows, summaries, fills,
                       morph_info, pool_detail, noise_rows=None,
                       deep_rows=None, set_coverage=None,
                       relatives=None, dedup_rows=None, family_rows=None,
                       gap_levels=None, out_map_rows=None):
        """Write the run's CSV/JSON exports (Revision 3.12).

        Evidence CSVs carry the category-partition columns: ``category`` /
        ``in_scope`` / ``morph_failed`` / ``candidate_annotation`` /
        ``counts_toward_restrictive_fill`` / ``counts_toward_family_fill``
        and ``dup``.  ``relatives`` / ``family_rows`` list the WHOLE
        ``relative`` / ``family`` bins (enumerated members ∪ evidence rows
        classified into them); ``dedup_rows`` is the query-level
        bodyId-unique rollup (``gap_fill_dedup.csv``).
        """
        rd = self.run_dir
        _write_csv(rd / 'validation_results.csv', val_rows)
        _write_csv(rd / 'suspicious_candidates.csv', sus_rows)
        _write_csv(rd / 'noise_filtered_candidates.csv',
                   noise_rows or [])
        _write_csv(rd / 'deep_candidates.csv', deep_rows or [])
        _write_csv(rd / 'relatives.csv', relatives or [])
        _write_csv(rd / 'family_candidates.csv', family_rows or [])
        _write_csv(rd / 'gap_fill_dedup.csv', dedup_rows or [])
        _write_csv(rd / 'gap_fill_levels.csv', gap_levels or [])
        _write_csv(rd / 'out_map_expansion.csv', out_map_rows or [])
        _write_csv(rd / 'pair_summary.csv', summaries)
        _write_csv(rd / 'gap_fill_proposals.csv', fills)
        _write_csv(rd / 'pool_categories.csv', pool_detail)
        _write_csv(rd / 'mapping_export.csv', self._mapping_export_rows())
        (rd / 'parameters.json').write_text(json.dumps({
            'source_dataset': self.cfg.source_dataset,
            'target_dataset': self.cfg.target_dataset,
            'query_types': self.cfg.query_types,
            'top_k': self.cfg.top_k, 'top_m': self.cfg.top_m,
            'min_synapse_threshold': self.cfg.min_synapse_threshold,
            'include_untyped_partners': self.cfg.include_untyped_partners,
            'rank_top_k': self.cfg.rank_top_k,
            'gap_min': self.cfg.gap_min,
            'gap_trigger_retired': True,
            'verified_top_n': self.cfg.verified_top_n,
            'invader_borderline_max': self.cfg.invader_borderline_max,
            'matched_ru_min': self.cfg.matched_ru_min,
            'candidate_morph_factor': self.cfg.candidate_morph_factor,
            'candidate_morph_cap': self.cfg.candidate_morph_cap,
            'target_min_weight': self.cfg.target_min_weight,
            'target_min_partner_types': self.cfg.target_min_partner_types,
            'suspicious_jaccard_factor': self.cfg.suspicious_jaccard_factor,
            'target_min_size_ratio': self.cfg.target_min_size_ratio,
            'suspicious_ru_margin': self.cfg.suspicious_ru_margin,
            'pool_ref_cap': self.cfg.pool_ref_cap,
            'pool_ref_floor_margin': self.cfg.pool_ref_floor_margin,
            'morph_track_a_offset': self.cfg.morph_track_a_offset,
            'morph_suspicious_level': self.cfg.morph_suspicious_level,
            # `validation_mode` is the resolved ordered enum; the legacy
            # boolean flags are reported as the mode's implications (not the
            # raw CLI input) so the record is self-consistent.
            'validation_mode': self.cfg.effective_mode,
            'mode_rank': self.cfg.mode_rank,
            'aggressive_expansion': self.cfg.mode_at_least('aggressive'),
            'pool_widen': False,   # retired in Rev 3.12
            'candidate_window': self.cfg.candidate_window,
            'deep_cap': self.cfg.deep_cap,
            'scene_selfcheck': self.cfg.scene_selfcheck,
            'morph_enabled': self.cfg.morph_enabled,
            'morph_auc_floor': self.cfg.morph_auc_floor,
            'suspicious_per_source_cap': self.cfg.suspicious_per_source_cap,
        }, indent=2))
        if morph_info:
            (rd / 'morphology_calibration.json').write_text(
                json.dumps(morph_info, indent=2, default=str))
        if set_coverage:
            (rd / 'set_coverage.json').write_text(
                json.dumps(set_coverage, indent=2, default=str))
        self._write_readme(summaries, morph_info, set_coverage)

    def _write_readme(self, summaries, morph_info, set_coverage=None):
        lines = [
            'Type-mapping validation run',
            '===========================',
            f'source: {self.cfg.source_dataset}  target: '
            f'{self.cfg.target_dataset}',
            f'queries: {", ".join(self.cfg.query_types)}',
            '',
            'Column glossary:',
            f'- validation mode: {self.cfg.effective_mode} '
            '(ordered enum restrictive < family < aggressive; modes',
            '  NEST — switching mode only admits more neurons, never',
            '  relabels one).',
            '- verdict: tiered rule — verified_strong (best pool member is',
            '  global top-1 under BOTH rank_union and jaccard), verified',
            f'  (top-1 under one), borderline (top-{self.cfg.rank_top_k}',
            '  window), unmatched.',
            '- category (Revision 3.12 partition; every in-scope target',
            '  gets EXACTLY one, decided in this order):',
            '    matched/verified/borderline/unmatched = the validated',
            '      in-map targets of THIS branch (tier; unmatched is the',
            '      else, no skipped);',
            '    sibling = an in-map target of the query in ANOTHER branch',
            '      that appears in this branch\'s expansion (connectivity-',
            '      and morph-qualified) — already mapped, never a fill;',
            '    candidates = out-of-map suspect, connectivity-qualified',
            '      (invader or gap fire) AND morph-qualified — the',
            '      restrictive fill;',
            '    family = out-map bodyIds of THIS branch\'s target type',
            '      (family/aggressive modes; type-gated, not morph-gated);',
            '    relative = type-mates of candidate types outside the map',
            '      (family/aggressive modes);',
            '    suspicious = the aggressive-only deep window.',
            '- candidate_annotation: the per-bodyId leaf token on every',
            '  expansion bin (family / candidates / relative / suspicious),',
            '  one ordered value:',
            '    {T}(out-map)  = the TYPE is an in-map type (bodyId-level',
            '                    out-of-map; every `family` member) — the',
            '                    fill material.  Wins over the tokens below.',
            '    {T}>{src}     = the type is NOT in-map but maps backward to',
            '                    a real source population (type-level).',
            '    {T}(no_source)= the type is NOT in-map with no usable',
            '                    backward route (hollow/absent; type-level).',
            '    untyped       = no type annotation.',
            '  (dup) is a standalone trailing tag.',
            '- in_scope / morph_failed: a connectivity-qualified suspect',
            '  that FAILS the morph rule is out of scope — exported with',
            '  in_scope=False, morph_failed=True, category blank, and',
            '  never rendered (this is the connectivity-only homolog',
            '  result, kept for reconciliation).',
            '- counts_toward_restrictive_fill = category is candidates;',
            '  counts_toward_family_fill = category is candidates/family/',
            '  relative.  The fill is a ranked process, not deterministic.',
            '- gap_fill_dedup.csv: query-level, one row per target bodyId',
            '  (precedence matched>verified>borderline>unmatched>sibling>',
            '  candidates>family>relative>suspicious); `dup` flags a',
            '  non-sibling bodyId labeled in more than one branch.',
            '- noise gates (rows moved to noise_filtered_candidates.csv',
            '  with noise_reason): spatial caliber <',
            f'  {self.cfg.target_min_size_ratio} x the branch pool best',
            '  (PRIMARY, from the neuron-table size — annotation-',
            '  independent), rank_union <= 0, jaccard < '
            f'{self.cfg.suspicious_jaccard_factor} x the pool best,',
            f'  rank_union margin < {self.cfg.suspicious_ru_margin}',
            '  (numerical tie).',
            '- candidate qualification (rule v3, floors): the NATIVE',
            '  matched+verified floor (mean pairwise reference sim -',
            '  native margin) is binding when the branch has >= 2 scored',
            '  references; otherwise the Track-A backup floor',
            '  B_b - Δ (Δ = morph_track_a_offset) on the branch scored',
            '  pool pairs; otherwise the run null p95.  The aggressive',
            '  deep window is admitted at B_b - k*Δ (null p50 fallback).',
            '  Bars and kinds are in morphology_calibration.json',
            '  (branch_bars / bar_params).',
            '- MORPHOLOGY SCORING FRAMES: both morph tracks score in the',
            '  TARGET dataset\'s coordinates — morph_v2_similarity/',
            '  morph_nblast (Track A) use the source transformed into',
            '  the target render space; morph_pool_ref (Track B, native',
            '  reference) is fully native with NO transforms. The scene',
            '  HTMLs render in the SOURCE dataset\'s coordinates',
            '  (targets bridged into the source template) — a different',
            '  frame from the scores; NBLAST is coordinate-sensitive,',
            '  so read the scene as anatomy, not as the scoring frame.',
            '- pool_ref tiers: the native reference set is matched-only',
            '  when a branch has >= 2 matched neurons; verified joins as',
            '  an explicitly-flagged compromise otherwise (see',
            '  pool_ref_tier in morphology_calibration.json).',
            '- gap: min(source_pool, target_pool) - matched; fills fire',
            f'  when gap > {self.cfg.gap_min} (default).',
            '  --mode aggressive adds the deep-window search (out-of-pool',
            '  homologs below the pool best) — off by default after the',
            '  r36e review showed over-expansion in finely identified',
            '  brain regions.',
            '- deep_candidates.csv (aggressive mode only):',
            '  out-of-pool neurons within the',
            f'  top-{self.cfg.candidate_window} per metric that rank',
            '  BELOW the pool best (Revision 3.8 deep window) — homologs',
            '  a strong pool member can hide.  Same gates as suspicious',
            '  rows; morph-qualified unexplained ones are `suspicious`.',
            '- pool_categories.csv: per in-map target, the tier category',
            '  (matched / verified / borderline / unmatched) with its',
            '  best-evidence metrics and size.',
            '- fills are proposals only (in_pool / out_of_pool tagged,',
            '  proposed for every unpaired neuron); out_of_pool fills',
            '  with their actual type are mapper-gap evidence. The',
            '  mapping is never rewritten.',
            '',
            'Run log:', *self.notes, '',
            'Pair summaries:',
        ]
        for s in summaries:
            lines.append(
                f"  {s['source_type']} -> {s['target_type']} "
                f"[{s['mapping_status']}/{s['pool_basis']}]: "
                f"pools {s['source_pool']}/{s['target_pool']}"
                f" (of {s.get('source_type_total', 0)}/"
                f"{s.get('target_type_total', 0)}), M={s['matched']}, "
                f"gap={s['gap']} ({s['gap_ratio']:.0%}), "
                f"triggered={s['gap_triggered']}")
        cov = set_coverage
        if cov:
            f, m = cov['fafb'], cov['mcns']
            lines += ['', 'SET-LEVEL COVERAGE (the deliverable; branch',
                      ' gaps double-count cross-branch convergence):',
                      f"  FAFB: {f['assigned']} of {f['total_queried']} "
                      "sources assigned;",
                      f"  +{f['fill_proposed_only']} fill-proposed only; "
                      f"{f['unpaired_unproposed']} unpaired without "
                      'proposal.',
                      f"  MCNS: mapped target set {m['mapped_target_set']};"
                      f" {m['in_branch_pool']} in branch pools;",
                      f"  {m['reached_as_candidates_only']} reached only "
                      'as candidates;',
                      f"  HOLES (never claimed): {m['holes']}",
                      f"  family material (in-map types, unclaimed by any"
                      f" branch pool): {len(m.get('family_material', []))}",
                      '  -> see set_coverage.json for per-type rollups, '
                      'the hole bodyIds, and family_material.',
                      '  NOTE: out-map candidates are listed in TWO files'
                      ' - suspicious_candidates.csv (category=candidates)'
                      ' and as proposal rows in gap_fill_proposals.csv;',
                      '  gap_fill_dedup.csv is the per-bodyId rollup.']
        gap_types = getattr(self, '_mapper_gap_types', None)
        if gap_types:
            lines += ['', 'Mapper-gap evidence (target types flagged in',
                      ' this run with NO backward mapping to the source',
                      ' dataset - candidate annotation holes; consider',
                      ' annotating or crosswalking them):']
            lines += [f'  {k}: {v} row(s)' for k, v in
                      sorted(gap_types.items(), key=lambda kv: -kv[1])]
        n_untyped = getattr(self, '_mapper_gap_untyped', 0)
        if n_untyped:
            lines.append(f'  (untyped): {n_untyped} row(s) - no type '
                         'annotation at all')
        if morph_info:
            lines += ['', 'Morphology calibration: ' +
                      json.dumps(morph_info, default=str)]
        (self.run_dir / 'README.txt').write_text('\n'.join(lines) + '\n')

    def _mapping_export_rows(self) -> List[Dict]:
        """Per-bridge mapping export (schema aligned with the per-bridge
        type-level export, plan-type-mapper-fine-granularity-export.md
        §3): one row per selected branch with the refined `{…}` pools,
        full populations in the parent_* columns."""
        rows = []
        for p in self.pairs:
            linkers = p.linkers
            linker_cols = '; '.join(str(l['column']) for l in linkers)
            linker_vals = p.linker_values

            def cell(ids):
                return '{' + ', '.join(str(b) for b in ids) + '}'

            rows.append({
                'source_dataset': p.source_dataset,
                'source_type': p.source_type,
                'target_dataset': p.target_dataset,
                'target_type': p.target_type,
                'relationship': p.relationship,
                'mapping_status': p.status,
                'query': p.query,
                'is_selected': True,
                'chain_rank': p.branch_index,
                'selected_bridge': p.chain_text,
                'bridge_linkers': (f'{linker_cols} = {linker_vals}'
                                   if linkers else ''),
                'selected_linker_values': linker_vals,
                'pool_basis': p.pool_basis,
                'target_pool_basis': p.target_pool_basis,
                'pool_widen_sources': len(p.pool_widen_added_sources),
                'pool_widen_targets': len(p.pool_widen_added_targets),
                'source_neurons': len(p.source_pool),
                'target_neurons': len(p.target_pool),
                'source_type_total': p.source_type_total,
                'target_type_total': p.target_type_total,
                'source_body_ids': cell(p.source_pool),
                'target_body_ids': cell(p.target_pool),
                'parent_source_body_ids': cell(p.parent_source_pool),
                'parent_target_body_ids': cell(p.parent_target_pool),
                'branch_index': p.branch_index,
                'branch_of': p.source_type,
                'branches_disjoint': ('' if p.branches_disjoint is None
                                      else p.branches_disjoint),
                'annotation': p.branch_annotation,
            })
        return rows


def _f(v):
    """NaN -> None so CSV cells stay empty rather than 'nan'."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except TypeError:
        return v
    return float(v)
