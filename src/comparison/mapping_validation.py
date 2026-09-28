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
   globally on BOTH metrics (``jaccard``, ``rank_union``) and order them by
   the bodyId chain — Jaccard first, rank_union as the tie-break
   (:func:`comparison.body_id_resolver.order_by_chain`); the two rank
   columns are published evidence, not the row order.  Apply the tiered
   verdict rule, assign 1:1 partners by mutual-best greedy on the same
   chain, and report every non-pool neuron ranked ahead of the mapped
   partners.
3. Category partition (Revision 3.12 — :func:`classify_category` +
   :meth:`MappingValidator.finalize_categories`): every in-scope target
   gets EXACTLY ONE category by one ordered first-match —
   ``matched``/``verified``/``borderline``/``unmatched`` (the branch's
   in-map targets) > ``sibling`` (in-map target of another branch of the
   query) > ``candidates`` (out-of-map, connectivity- AND morph-qualified;
   the restrictive fill.  Connectivity is either an invader beating the
   pool's best, a gap-fire proposal, or — family MODE and up — a neuron
   inside the candidate-discovery window, the top-``rank_top_k`` of each
   metric) > ``family`` (out-map bodyIds of the branch's
   target type) > ``relative`` (candidate-type mates outside the map) >
   ``examinees`` (the aggressive-only deep window; renamed from
   'suspicious' 2026-09-18 — the mapper's rival-suspects concept now owns
   that word).  A
   connectivity-qualified suspect failing the morph rule is OUT OF SCOPE
   (``in_scope=False``, ``morph_failed=True``) — kept in the CSVs for
   reconciliation with a connectivity-only homolog search, never rendered.
   Modes nest: ``restrictive ⊆ family ⊆ aggressive``
   (``validation_mode`` / :func:`normalize_mode`); a shared neuron keeps
   the same category across modes.  ``pooling`` is PARALLEL to that ladder,
   not a fourth rung: :mod:`comparison.mapping_validation_pooling` scans
   the whole queried population against the whole target universe, admits
   each source's top-N rows under its own bar, and joins the mapper's claims
   only afterwards.
4. Gap fill: ``gap = min(|P_S|, |P_T|) - M``; the restrictive fill counts
   ``candidates`` only, the family fill adds ``family``+``relative``.  The
   query-level dedup (``gap_fill_dedup.csv``) is bodyId-unique with
   precedence tier > sibling > candidates > family > relative >
   examinees.  Proposals only — the mapping is never rewritten.
5. Morphology verification: vector_v2 + NBLAST via
   ``morphology.enrich_homolog_results`` on the pairs that matter, with
   per-run self-calibration (AUC separating verified_strong from
   examinees) and a guard rail: morph gates ``verified_strong`` only when
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from comparison.connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    ProfilerConfig,
)
from comparison.cross_dataset_type_mapper import (
    SOURCE_REFINING_BASES,
    basis_is_row_evidence,
    get_type_mapper,
)
from comparison.morph_bars import (
    BarSet,
    candidate_qualified,
    compute_branch_bars,
    suspicious_qualified,
)
from comparison.profile_comparator import ProfileComparator
from flywire_ids import is_fafb_dataset
try:
    from utils.label_utils import UntypedLabelPolicy
except ImportError:  # direct src/ execution
    from src.utils.label_utils import UntypedLabelPolicy
from comparison.body_id_resolver import (  # noqa: E402
    BodyIdResolver,
    BodyIdResolverConfig,
    _SideStats,
    _pearson,
    _rankdata_average,
    _best_row,
    BACKWARD_COLUMNS,
    BACKWARD_EVIDENCE_VALUES,
    blank_backward_fields,
    build_target_vectors,
    chain_key,
    classify_backward_scan,
    expanded_vector,
    order_by_chain,
    load_caliber_map,
    load_hemisphere_map,
    passes_target_quality_gate,
    prep_target_stats,
    reverse_source_column,
    scan_source,
    score_one_candidate,
    score_one_candidate_fast,
    serialize_topn_union,
    topn_union_rows,
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
    # §full-map boundary guard (user 2026-09-28): the validation pipeline
    # consumes the mapper's CLAIM tier exclusively — a full-map (transitive
    # composition) scope exists only in the Type Mapping panel's display
    # layer and must NEVER reach validation.  Hard-refuse anything but the
    # curated scope so a future wiring mistake cannot silently validate
    # panel-composed pairs (plan-full-map-route-scope.md §6).
    route_scope: str = 'curated'
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
    #   restrictive (default) - tier + sibling + candidates; minimal, and
    #                           candidates come only from the invader bar.
    #   family                - adds the candidate-DISCOVERY window (the
    #                           top-`rank_top_k` of each metric) and the
    #                           family/relative bins (out-map bodyIds of
    #                           the in-map types, and the type-mates of
    #                           candidate types).
    #   aggressive            - widens the window to `candidate_window` and
    #                           adds the deep-window `examinees` bin.
    # `aggressive_expansion` and `pool_widen` are legacy boolean aliases
    # resolved by `normalize_mode` (pool_widen -> family).  Pool widening
    # itself is RETIRED (Revision 3.12): family mode no longer touches the
    # validated pool, so the tier is identical across modes.
    validation_mode: str = 'restrictive'
    aggressive_expansion: bool = False
    pool_widen: bool = False
    candidate_window: int = 25
    #: Per source and PER BAND: the borderline window (`rank_top_k`) and the
    #: deep window (`candidate_window`) each draw on a budget of this size, so
    #: widening the mode only ever adds rows (see `_scan_pair`'s window block).
    deep_cap: int = 10
    # ------------------------------------------------------------------
    # `pooling` mode (plan-tmvev-pooling-mode.md + plan-tmvev-pooling-tiers.md):
    # the UNSUPERVISED candidate engine, whose unit is the queried source.
    #
    # ADMISSION IS THE BAR: every source keeps the top-N of each chosen metric
    # (`pooling_bar_metric` x `pooling_bar_top_n`), no branch pool and no
    # pool-relative bar read on the selection path.
    #
    # The three FLOOR knobs no longer filter anything (user 2026-09-24).  Each is
    # still evaluated against its configured number and published per row
    # (`below_jaccard_floor` / `below_rank_union_floor` / `outside_window`), which
    # keeps the disclosure and drops the deletion: measured on the landed run,
    # `pooling_rank_union_floor = 0.0` was the SOLE reason 119 of 242 sources
    # reported nothing, and a sign test on every pair is not the volume control
    # these numbers were documented as.  An earlier build also fitted
    # `pooling_jaccard_floor` to the dataset pair's own graded evidence; that fit
    # was deleted rather than tuned.
    # ------------------------------------------------------------------
    pooling_jaccard_floor: float = 0.10
    pooling_rank_union_floor: float = 0.0
    pooling_window_mult: float = 2.0
    #: The admission bar.  `either` means the UNION of each metric's own top-N —
    #: not a merged best-rank ordering, which spends the N slots on the two
    #: metrics' rank-1 rows and was measured to keep only 79 of the 118 targets
    #: the floors found, where the union keeps 116.
    pooling_bar_metric: str = 'either'      # either | jaccard | rank_union
    pooling_bar_top_n: int = 3
    #: Morphology is MANDATORY here (every tier is defined morph-qualified, so a
    #: run without it would publish connectivity-only rows under claim-shaped
    #: names) and it is the LAST GATE: a row the bar refuses leaves the exported
    #: pool, counted as `morph['dropped_targets']`.  A row with NO verdict never
    #: leaves — a missing measurement is not a rejection.
    #: Morph is the LAST gate and the only network-bound step left in the
    #: path, so the pass is budgeted; rows past the budget say so in
    #: `morph_gate='not-attempted-cap'` rather than reading as rejections.
    #: 0 means AUTO: `MORPH_UNITS_PER_SOURCE` x the queried population, so the
    #: budget scales with the work a run was asked to do instead of with a
    #: constant nobody could check.  Measured on the default bar (either/3,
    #: 242 sources): the pass needs 546 units, and the old constant 400 cut
    #: 146 of them — which cost one tier-1 claim and left 115 rows without any
    #: verdict because the row that OWED their target its verdict was itself
    #: cut.  A positive number is an explicit cap.
    pooling_max_morph_targets: int = 0
    # ------------------------------------------------------------------
    # Stage 5d: backward (target -> source) homolog evidence for the
    # expansion bins.  ADVISORY ONLY — `backward_evidence` labels a row, it
    # never gates, never changes `category`, and never touches the
    # counts_toward_* flags.  Connectivity only: morphology is deliberately
    # not re-scored here (plan-tmvev-backward-expansion-evidence.md D5).
    # ------------------------------------------------------------------
    #: USER DECISION 2026-09-26: the reciprocal pass runs BY DEFAULT
    #: (it was opt-in `--backward-evidence` before; every certified round
    #: shipped an empty Reciprocal tab because nothing ever passed the
    #: flag).  Opt out with `--no-backward-evidence` / the UI checkbox.
    backward_evidence_enabled: bool = True
    #: Reverse hits kept per neuron (the report's hover label; the table
    #: cells show the top-1 and the branch-type hit, never the whole list).
    backward_top_n: int = 5
    #: Hard budget on dataset-scale reverse scans per run — one scan costs as
    #: much as a forward source scan (~4-7 s against a 140-180k universe).
    backward_max_neurons: int = 300
    #: Per-branch budget on labeled expansion members (a branch can still be
    #: labeled from a scan another branch paid for).
    backward_per_branch_cap: int = 40
    #: Reverse-scan the UNMATCHED validated pool targets too, so the
    #: `source-` status columns see out-of-branch competitors (they cannot
    #: from forward scans).  The matched/verified/borderline members are
    #: skipped: their forward pair already IS the symmetric evidence.
    backward_scan_pool_targets: bool = True
    skip_backward_pass: bool = False
    # Rev 3.9 Track-A bar calibration: the query-based morph threshold
    # is NULL-CALIBRATED per run — the p95 of the Track-A morph over
    # window rows with jaccard <= `null_jaccard_max` (provably unrelated
    # by connectivity), replacing the arbitrary factor x pooled-average
    # bar when enough null samples exist (null_min_n).  Selection is
    # deterministic (targets sorted by bodyId, see select_null_sample);
    # the bar is recomputed per run BY DESIGN — morph is an arbitrary
    # floor gate (connectivity ranks, the user verifies), so a small
    # sampled set is enough and there is no cross-run persistence
    # (user 2026-09-17).
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
    # The stage-5 skeleton pre-flight is the one network-bound block in the
    # run (measured 3.0 s per target fetched one-at-a-time: 1,401 s for 467
    # on banc_v888).  It therefore goes through the shared batched NeuPrint
    # fetcher with this many threads, and bounds the socket inactivity it
    # will tolerate — neuprint-python and navis both expose no per-request
    # timeout, so without this a single hung request stalls the stage.
    skeleton_fetch_workers: int = 8
    skeleton_fetch_timeout_s: int = 120
    # stage 4
    visualize: bool = True
    # 0 (default) renders one scene per parent type. A positive value caps
    # the list — branch review is the point of a run, so a silent cap that
    # leaves parents unrendered is a degradation, not a convenience; the
    # planner names every dropped parent in the log when it clamps.
    max_scenes: int = 0
    neuron_alpha: float = 0.2   # global neuron opacity (backend default)
    # Revision 3.5 Issue 6c: debug self-check — after rendering, verify
    # each legend leaf's geometry bbox matches its labeled neuron's bbox.
    scene_selfcheck: bool = False
    # The look of the stage-4 branch scenes, as two dicts rather than a dozen
    # fields: the UI's collapsed Advanced Visualization panel is a
    # ``VisualizeSkeleton``-keyed snapshot, and the scene colors are a
    # category -> color map.  Both default None, and the UI runner drops None
    # keys, so an unset run renders EXACTLY as it did before these existed.
    # ``render_pair_scenes`` merges each over its own pinned kwargs — the
    # legend tree, the coordinate template and the synapse skip are not
    # offered to the caller because the scene's correctness depends on them.
    scene_viz: Optional[Dict[str, Any]] = None
    scene_category_colors: Optional[Dict[str, str]] = None
    # plumbing
    # Same-name-first suspects verification (plan-tmvev-samename-first-
    # consumers.md P3): advisory connectivity check of the rival
    # candidates.  OPT-IN, default OFF — an off-run behaves exactly as
    # before (no extra scans, no suspects_verification.csv).
    verify_suspects: bool = False
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
    def mode_rank(self) -> Optional[int]:
        """The nested chain's rank, or None for `pooling`.

        None is the point: pooling has no rank in that ladder, and publishing
        it as such keeps a pooling run from being read as a wider restrictive
        run.
        """
        mode = self.effective_mode
        return None if mode == POOLING_MODE else MODE_RANK[mode]

    def mode_at_least(self, mode: str) -> bool:
        rank = self.mode_rank
        return False if rank is None else rank >= MODE_RANK[str(mode).lower()]


def scene_styling_record(cfg, log=None) -> Dict[str, Any]:
    """The scene look as the run ACTUALLY wore it, for `parameters.json`.

    Recording the raw config would understate the pages: `scene_viz` can carry
    the panel's "use the method default" `None` while the scenes rendered at
    0.95, and a category recolor propagates to its legacy aliases
    (`relative` -> `relatives`, `query` -> `out-map query`). So the same two
    helpers stage 4 uses resolve the record, which keeps the provenance and the
    render from ever disagreeing.  With scenes off there was no look to wear, so
    the raw config (usually `None`) stands.
    """
    raw = {'scene_viz': cfg.scene_viz,
           'scene_category_colors': cfg.scene_category_colors}
    if not cfg.visualize:
        return raw
    try:
        from comparison.mapping_validation_visualize import (
            resolve_scene_colors, scene_render_kwargs)
    except Exception as exc:  # noqa: BLE001 - provenance never fails a run
        if log:
            log(f'[export] scene styling record unavailable ({exc}); '
                'recording the raw config')
        return raw
    return {'scene_viz': scene_render_kwargs(cfg),
            'scene_category_colors': resolve_scene_colors(
                cfg.scene_category_colors)}


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
    # Which chain actually supplied the SOURCE pool.  Equal to
    # `selected_chain` except under the per-side basis below, where a
    # different supported chain for the same endpoint narrowed the source
    # side (plan-tmvev-jaccard-primary-bodyid-ranking.md §15.3).
    source_chain: List[Dict[str, str]] = field(default_factory=list)
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
    # Same-name-first provenance (plan-tmvev-samename-first-consumers.md
    # P1): the FIRED decision's {selected, rivals, path, disposition};
    # None for ordinary pairs.  Advisory marking only — the pair validates
    # exactly like any other mapped pair.
    same_name_first: Optional[dict] = None
    # §three-tier readout (user 2026-09-27): 'claim' (default — the pair
    # validates into the ordinary bins) or 'disclosure' (an end the mapper
    # decision declined but the evidence reaches; verified with the same
    # machinery into the SEPARATE disclosure bin — never the headline
    # counts, preserving panel parity).
    tier: str = 'claim'

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

    @property
    def source_chain_text(self) -> str:
        """The chain that supplied the SOURCE pool — equal to `chain_text`
        unless the per-side basis (§15.3) took the source side from another
        supported chain of the same endpoint."""
        return ' -> '.join(
            f"{h.get('dataset')}:{h.get('column')}={h.get('value')}"
            for h in self.source_chain) if self.source_chain else ''


def measure_branch_disjointness(pairs: List["TypePair"]) -> Optional[bool]:
    """Pairwise disjointness of the linker-refined source sub-pools across
    the branches of one parent source type.

    Returns None when it cannot be measured (single branch, or any branch
    falls back to the full population — the overlap is then meaningless).
    A side resolved through the release relation names neurons just like
    linker rows do, so both row-backed bases are measurable here.
    """
    refined = [p for p in pairs if basis_is_row_evidence(p.pool_basis)]
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
    all_linkered = all(basis_is_row_evidence(p.pool_basis) for p in pairs)
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
                         evidence_rows: Optional[List[Dict]] = None,
                         extra_counters: Optional[Dict] = None) -> Dict:
    """Revision 3.10: set-level coverage of the source->target mapping.

    Per-branch gaps are diagnostics (they double-count cross-branch
    convergence); this answers the deliverable question directly:

    - source side: of the queried source population
      (union of parent_source_pool), how many neurons are assigned
      (verified/borderline with a target), fill-proposed, or still
      unpaired — per type and in total.
    - target side: of the mapped target set (union of
      parent_target_pool), how many neurons sit in a branch pool with
      which best category, how many are reached ONLY as out-of-pool
      candidates/proposals, and which are never claimed anywhere
      ('holes' — the family material, i.e. out-map bodyIds of in-map
      types, that neither refinement nor a candidate reached).
      "Reached" covers BOTH proposal routes: gap-fill rows and the
      invader-surfaced rows `finalize_categories` labeled
      ``candidates`` (pass the post-finalization sus/deep rows as
      ``evidence_rows``) — a bodyId the run claimed as a candidate is
      not a hole.

    The two blocks are named ``source``/``target``, never after a
    concrete dataset: the same writer serves FAFB->male-cns and
    FAFB->BANC, and a `mcns` key on a BANC run is a lie the report
    then repeats. The dataset names are read from the pairs into
    ``source_dataset``/``target_dataset``.
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
        res = per_pair_res.get((pair.query,) + pair.key) \
            or per_pair_res.get(pair.key)
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

    src_types = {}
    for ptype, members in sorted(parent_src.items()):
        a = sum(1 for b in members if b in assigned_src)
        p = sum(1 for b in members
                if b not in assigned_src and b in proposed_src)
        src_types[ptype] = {
            'pool': len(members), 'assigned': a,
            'fill_proposed': p, 'unpaired_unproposed': len(members) - a - p}
    tgt_types = {}
    for ttype, members in sorted(parent_tgt.items()):
        in_pool = {b: pool_cat[b] for b in members if b in pool_cat}
        reached = {b for b in members if b in reached_tgt}
        holes = sorted(b for b in members
                       if b not in pool_cat and b not in reached)
        tgt_types[ttype] = {
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
    out = {
        'source_dataset': pairs[0].source_dataset if pairs else '',
        'target_dataset': pairs[0].target_dataset if pairs else '',
        'source': {
            'total_queried': len(all_src),
            'assigned': len(assigned_src),
            'fill_proposed_only': len(set(proposed_src) - assigned_src),
            'unpaired_unproposed': len(
                all_src - assigned_src - set(proposed_src)),
            'per_type': src_types,
        },
        'target': {
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
            'per_type': tgt_types,
        },
        'branch_gap_note': ('per-branch gaps double-count cross-branch '
                            'convergence; the set-level numbers here '
                            'are the deliverable'),
    }
    # Same-name-first consumers (plan-tmvev-samename-first-consumers.md):
    # additive counters, no schema break.  P1 marks the FIRED selections;
    # extra_counters carries the validator's P2/P4 accounting (held /
    # evidence-only fan-outs, multivalue cells).
    snf_pairs = [p for p in pairs if p.same_name_first]
    out['same_name_first_pairs'] = len(snf_pairs)
    out['same_name_first_types'] = len({p.source_type for p in snf_pairs})
    out.update(extra_counters or {})
    return out


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

# Stage 5 fetches the target skeletons it is about to score against (they
# otherwise arrive in stage 4, which runs too late — see
# MappingValidator._preflight_target_skeletons). The pair frame is already
# capped per source, but a wide aggressive run can still ask for tens of
# thousands of distinct targets, so the fetch is bounded and names what it
# leaves unscored.
MORPH_SKELETON_PREFLIGHT_CAP = 2000

# The pre-flight walks the shared batched fetcher in chunks so the neurons it
# returns (it persists AND hands back every skeleton it fetches) never all
# live at once: 2,000 target skeletons is multi-GB of TreeNeurons, and the
# pre-flight needs none of them in memory.
MORPH_SKELETON_FETCH_CHUNK = 128


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
        bev = str(d.get('backward_evidence') or '')
        if cat == 'candidates':
            kind, value = bars.get(bid, ('null', None))
            level = BAR_KIND_TO_LEVEL.get(kind, 'low')
            note = 'hole closer' if bid in fam_material else ''
            evidence, bar_value = kind, value
        else:
            level = GAP_FILL_LEVEL_BY_CATEGORY[cat]
            # The reverse finding is a second-direction CONNECTIVITY fact, so
            # it replaces the annotation-only provenance rather than joining
            # the morph-bar `level` ladder (plan D7).
            evidence = (f'backward_{bev}'
                        if bev and bev != 'not-checked'
                        else ('type_membership' if cat == 'family'
                              else 'candidate_type_mate'))
            bar_value, note = None, ''
        counts[level] += 1
        rows.append({
            'level': level,
            'target_bodyId': bid,
            'target_type': d.get('target_type'),
            'dedup_category': cat,
            'evidence': evidence,
            'bar_value': bar_value,
            'backward_evidence': bev,
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
    except Exception as exc:
        # TMV-5: with no source counts every backward invader reads
        # 'hollow-backward' instead of 'backward' — say so once.
        print(f'[TMVEV] ! source type counts unavailable ({exc}): '
              'backward invaders will classify hollow-backward', flush=True)
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


def _abbrev(dataset: str) -> str:
    return (dataset or 'ds').replace(':', '_').replace('.', '_')


def _short_name(dataset: str) -> str:
    """Short display nickname for run-folder names (user 2026-09-18:
    `type-map-validation_FAFB_to_MCNS_{stamp}`).  Mirrors the
    cross-dataset backend's `ComparisonParameters.get_display_nickname`
    (flywire_FAFB_v783 -> FAFB, male-cns:v1.0 -> MCNS, banc_v888 -> BANC,
    hemibrain:v1.2.1 -> HEMI); falls back to `_abbrev` when the nickname
    table is unavailable."""
    try:
        from comparison.comparison_parameters import ComparisonParameters
        return str(ComparisonParameters.get_display_nickname(
            ComparisonParameters, dataset))
    except Exception:  # noqa: BLE001
        return _abbrev(dataset)


def _store_identity(path) -> Optional[Dict[str, Any]]:
    """(bytes, mtime) of an input store, None when it is absent.

    A run's outputs are a function of the caches it read, so an old-code vs
    new-code A/B needs to tell 'the code changed' from 'the cache changed'
    without forensics: this is what `parameters.json` publishes."""
    try:
        st = Path(path).stat()
    except (OSError, TypeError):
        return None
    return {'path': str(path), 'bytes': int(st.st_size),
            'mtime_s': int(st.st_mtime)}


def _morph_store_identity(dataset: str,
                          project_root: Optional[str] = None) -> Dict[str, Any]:
    """What a dataset's morphology stores looked like when a verdict was made.

    The native track scores a candidate against its source's reference pool
    out of the target's V2 vector cache, and the Track-A scores come out of the
    rendered target-vector store — so a run's morphology record is a function of
    those files the same way the scan is a function of the profile caches.
    Before 2026-09-25 the fingerprint named only the latter, and a cold-vs-warm
    difference in a `morph_pool_ref` could not be attributed from the run folder
    (it cost 107 rows of one FAFB->BANC pooling run to find the cause).

    A skeleton store is reported as a COUNT plus its newest file: the store
    holds thousands of small files, so per-file identity would swamp the record
    while a count still says whether a run scored against a fuller store than
    its repeat.
    """
    out: Dict[str, Any] = {}
    try:
        from morphology import find_similar_dataset_cache_v2
        cache = find_similar_dataset_cache_v2(dataset,
                                              project_root=project_root,
                                              verbose=False)
    except Exception:  # noqa: BLE001 — bookkeeping never takes a stage down
        return out
    # Name what the cache object actually exposes: a unit-test double carries
    # none of these paths and a half-built cache carries some, and this record
    # must never be the reason a scored run fails.
    for key, label in (('parquet_path', 'vector_cache'),
                       ('pending_path', 'vector_pending'),
                       ('meta_path', 'vector_meta'),
                       ('whiten_path', 'whitener')):
        path = getattr(cache, key, None)
        if path is not None:
            identity = _store_identity(path)
            if identity:
                out[label] = identity
    newest = 0
    n_files = 0
    try:
        for path in cache._discover_skeleton_files():
            n_files += 1
            try:
                newest = max(newest, int(Path(str(path)).stat().st_mtime))
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    out['skeleton_files'] = n_files or None
    out['newest_skeleton_mtime_s'] = newest or None
    try:
        from comparison.morph_cross_dataset import TargetVectorStore
        out['target_vector_store'] = _store_identity(
            TargetVectorStore(dataset, project_root).path)
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in out.items() if v}


def _git_dirty() -> Optional[bool]:
    """Whether the run's code was NOT the committed tree.

    A run is attributed by `git_rev`, but a dirty worktree means that rev names
    a different code state than the one that scored — measured 2026-09-25, where
    the run certifying the `vectors_for` one-space fix printed the pre-fix rev
    because the fix was still uncommitted. None when git cannot answer.
    """
    try:
        import subprocess
        out = subprocess.run(['git', 'status', '--porcelain'],
                             cwd=str(Path(__file__).resolve().parents[2]),
                             capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001 (a source zip has no git)
        return None
    if out.returncode != 0:
        return None
    return bool((out.stdout or '').strip())


def _git_rev() -> Optional[str]:
    """Short HEAD rev of the tree this code was executed from."""
    try:
        import subprocess
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=str(Path(__file__).resolve().parents[2]),
                             capture_output=True, text=True, timeout=5)
        return (out.stdout or '').strip() or None
    except Exception:  # noqa: BLE001 (a source zip has no git)
        return None


# ---------------------------------------------------------------------------
# Run-folder layout (plan-tmvev-backward-expansion-evidence.md Part II)
# ---------------------------------------------------------------------------
#: Exported file -> subfolder inside the run dir (``''`` = root).  Root keeps
#: only the deliverables (report, user guide, README) and the parameter/meta
#: surface; the evidence CSVs are grouped by pipeline stage.  This is the ONE
#: registry writers and readers consult — a filename used to be hard-coded in
#: six places (the writer, the schema registry, the report readers,
#: ``ARTIFACT_LINES``, the run-guide patterns and the README start-here list).
RUN_FILE_LAYOUT: Dict[str, str] = {
    # stage-2/3 validation evidence
    'validation_results.csv': 'validation',
    'forward_matches.csv': 'validation',
    'pair_summary.csv': 'validation',
    'pool_categories.csv': 'validation',
    'examinees.csv': 'validation',
    'deep_candidates.csv': 'validation',
    'noise_filtered_candidates.csv': 'validation',
    # expansion bins + their reverse evidence
    'family_candidates.csv': 'expansion',
    'relatives.csv': 'expansion',
    'out_map_expansion.csv': 'expansion',
    'source_candidates.csv': 'expansion',
    'source_status.csv': 'expansion',
    'backward_matches.csv': 'expansion',
    'target_matches.csv': 'expansion',
    # gap-fill accounting
    'gap_fill_dedup.csv': 'gap_fill',
    'gap_fill_levels.csv': 'gap_fill',
    'gap_fill_proposals.csv': 'gap_fill',
    # pooling mode: the unsupervised engine and its post-hoc comparison
    'pooling_candidates.csv': 'pooling',
    'pooling_pool.csv': 'pooling',
    'pooling_sources.csv': 'pooling',
    'pooling_cross_validation.json': 'pooling',
    # mapping provenance
    'mapping_export.csv': 'mapping',
    'same_name_excluded.csv': 'mapping',
    'disclosure_evidence.csv': 'mapping',
    'suspects_verification.csv': 'mapping',
    # root: deliverables + parameter/meta
    'README.txt': '',
    'report.html': '',
    'parameters.json': '',
    'set_coverage.json': '',
    'morphology_calibration.json': '',
    'pipeline_progress.jsonl': '',
    'user_warning_notes.txt': '',
}


def run_file_path(run_dir, name: str, *, create_parent: bool = False) -> Path:
    """Resolve one run-dir file through :data:`RUN_FILE_LAYOUT`.

    Writers pass ``create_parent=True`` and always get the registry location.
    Readers get the LEGACY flat path when only that exists, so run folders
    written before the layout change keep reporting and regenerating — the
    same two-name fallback ``_read_examinees`` already applies to the
    pre-rename CSV."""
    sub = RUN_FILE_LAYOUT.get(name, '')
    new = Path(run_dir) / sub / name if sub else Path(run_dir) / name
    if create_parent:
        new.parent.mkdir(parents=True, exist_ok=True)
        return new
    if sub and not new.exists():
        flat = Path(run_dir) / name
        if flat.exists():
            return flat
    return new


# Header-only empty exports (user 2026-09-18): a run folder should never
# contain zero-byte CSVs — pd.read_csv raises EmptyDataError on them.  When
# `_write_outputs` has no rows for one of these files it writes the header
# line from this registry instead.  The registries mirror the CURRENT
# schemas; a schema change only needs to touch them if empty files must
# carry the new column (non-empty files are unaffected — DataFrame(rows)
# uses the row dicts' own keys).
_RUN_CSV_SCHEMAS: Dict[str, List[str]] = {
    'validation_results.csv': [
        'query', 'source_dataset', 'source_type', 'target_dataset',
        'target_type', 'mapping_status', 'relationship', 'same_name_first',
        'same_name_rivals', 'pool_basis', 'branch_linker_values',
        'branch_annotation', 'branches_disjoint', 'source_bodyId',
        'source_connectivity_status', 'verdict', 'metric_top1', 'flags',
        'suspicious_count', 'suspicious_noise_filtered',
        'suspicious_size_filtered', 'suspicious_tie_filtered',
        'target_bodyId', 'ru_top_target_bodyId',
        'rank_union', 'rank_union_rank', 'jaccard',
        'jaccard_rank', 'cosine', 'weighted_jaccard', 'morph_v2_similarity',
        'morph_nblast', 'source_size'],
    'examinees.csv': [
        'query', 'source_type', 'target_type', 'pool_basis',
        'branch_linker_values', 'branch_annotation', 'source_bodyId',
        'ahead_target_bodyId', 'ahead_target_type', 'ahead_metric',
        'ahead_rank', 'ahead_rank_union', 'ahead_jaccard',
        'best_pool_target_bodyId', 'best_pool_rank', 'best_pool_rank_union',
        'best_pool_jaccard', 'ahead_size', 'pool_best_size', 'size_ratio',
        'size_filtered', 'invader_class', 'invader_label', 'sibling_pool_of',
        'sibling_category', 'backward_status', 'backward_maps_to',
        'fafb_home_count', 'backward_home_real', 'alt_chain_of_parent',
        'in_query_family', 'same_type_residue', 'morph_pool_ref',
        'morph_pool_ref_mean', 'pool_ref_tier', 'morph_v2_similarity',
        'morph_nblast', 'category', 'in_scope', 'morph_failed', 'bar_kind',
        'bar_value', 'candidate_annotation',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'counts_toward_gap_fill', 'dup'],
    'noise_filtered_candidates.csv': [
        'query', 'source_type', 'target_type', 'pool_basis',
        'branch_linker_values', 'branch_annotation', 'source_bodyId',
        'ahead_target_bodyId', 'ahead_target_type', 'ahead_metric',
        'ahead_rank', 'ahead_rank_union', 'ahead_jaccard',
        'best_pool_target_bodyId', 'best_pool_rank', 'best_pool_rank_union',
        'best_pool_jaccard', 'ahead_size', 'pool_best_size', 'size_ratio',
        'noise_reason'],
    'deep_candidates.csv': [
        'query', 'source_type', 'target_type', 'pool_basis',
        'branch_linker_values', 'branch_annotation', 'source_bodyId',
        'ahead_target_bodyId', 'ahead_target_type', 'ahead_metric',
        'ahead_rank', 'ahead_rank_union', 'ahead_jaccard',
        'best_pool_target_bodyId', 'best_pool_rank', 'best_pool_rank_union',
        'best_pool_jaccard', 'ahead_size', 'pool_best_size', 'size_ratio',
        'size_filtered', 'invader_class', 'invader_label', 'category',
        'in_scope', 'morph_failed', 'candidate_annotation',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'candidate_source', 'dup'],
    'relatives.csv': [
        'query', 'source_type', 'target_type', 'pool_basis',
        'branch_linker_values', 'branch_annotation', 'source_bodyId',
        'ahead_target_bodyId', 'ahead_target_type', 'ahead_metric',
        'ahead_rank', 'ahead_rank_union', 'ahead_jaccard',
        'best_pool_target_bodyId', 'best_pool_rank', 'best_pool_rank_union',
        'best_pool_jaccard', 'ahead_size', 'pool_best_size', 'size_ratio',
        'size_filtered', 'invader_class', 'invader_label', 'category',
        'in_scope', 'morph_failed', 'candidate_annotation',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'candidate_source', 'dup'],
    'family_candidates.csv': [
        'query', 'source_type', 'target_type', 'pool_basis',
        'branch_linker_values', 'branch_annotation', 'source_bodyId',
        'ahead_target_bodyId', 'ahead_target_type', 'ahead_metric',
        'ahead_rank', 'ahead_rank_union', 'ahead_jaccard',
        'best_pool_target_bodyId', 'best_pool_rank', 'best_pool_rank_union',
        'best_pool_jaccard', 'ahead_size', 'pool_best_size', 'size_ratio',
        'size_filtered', 'invader_class', 'invader_label', 'category',
        'in_scope', 'morph_failed', 'candidate_annotation',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'candidate_source', 'dup'],
    'gap_fill_dedup.csv': [
        'target_bodyId', 'target_type', 'dedup_category', 'n_branches',
        'dup', 'counts_toward_restrictive_fill',
        'counts_toward_family_fill'],
    'gap_fill_levels.csv': [
        'level', 'target_bodyId', 'target_type', 'dedup_category',
        'evidence', 'bar_value', 'dup', 'note'],
    'gap_fill_proposals.csv': [
        'query', 'source_type', 'target_type', 'side', 'bodyId',
        'source_verdict', 'proposal_bodyId', 'fill_class',
        'proposal_type', 'rank_union', 'rank_union_rank', 'jaccard',
        'jaccard_rank', 'invader_class', 'invader_label',
        'sibling_pool_of', 'sibling_category', 'backward_status',
        'backward_maps_to', 'fafb_home_count', 'backward_home_real',
        'alt_chain_of_parent', 'in_query_family', 'fill_scope_note',
        'counts_toward_gap_fill', 'morph_pool_ref', 'morph_pool_ref_mean',
        'pool_ref_tier', 'morph_v2_similarity', 'morph_nblast',
        'candidate_annotation', 'category', 'in_scope', 'morph_failed',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'dup', 'same_type_residue', 'bar_kind', 'bar_value',
        'fill_qualified'],
    'pair_summary.csv': [
        'query', 'source_type', 'target_type', 'mapping_status',
        'relationship', 'same_name_first', 'same_name_rivals',
        'pool_basis', 'target_pool_basis', 'selected_chain',
        'source_chain', 'branch_linker_values',
        'branch_annotation', 'branches_disjoint', 'source_pool',
        'target_pool', 'pool_widen_added', 'source_type_total',
        'target_type_total', 'best', 'verdict_verified_strong',
        'verdict_verified', 'verdict_borderline', 'verdict_unmatched',
        'verdict_skipped', 'suspicious_neurons',
        'suspicious_noise_filtered', 'suspicious_size_filtered',
        'suspicious_tie_filtered', 'gap', 'gap_ratio', 'gap_triggered',
        'hemisphere', 'pool_best_size', 'deep_candidates', 'null_sample'],
    'pool_categories.csv': [
        'target_bodyId', 'category', 'best_source_bodyId', 'rank_union',
        'rank_union_rank', 'jaccard_rank', 'invaders_ahead', 'query',
        'source_type', 'target_type', 'pool_basis', 'branch_annotation',
        'size', 'morph_v2_similarity', 'morph_nblast', 'in_scope',
        'morph_failed', 'candidate_annotation',
        'counts_toward_restrictive_fill', 'counts_toward_family_fill',
        'dup'],
    'mapping_export.csv': [
        'source_dataset', 'source_type', 'target_dataset', 'target_type',
        'relationship', 'mapping_status', 'tier',
        'same_name_first',
        'same_name_rivals', 'query', 'is_selected', 'chain_rank',
        'selected_bridge', 'source_bridge', 'bridge_linkers',
        'selected_linker_values',
        'pool_basis', 'target_pool_basis', 'pool_widen_sources',
        'pool_widen_targets', 'source_neurons', 'target_neurons',
        'source_type_total', 'target_type_total', 'source_body_ids',
        'target_body_ids', 'parent_source_body_ids',
        'parent_target_body_ids', 'branch_index', 'branch_of',
        'branches_disjoint', 'annotation'],
    'source_status.csv': [
        'query', 'source_bodyId', 'source_type', 'branch_target_type',
        'pool_basis', 'status', 'col_rank', 'best_column_target',
        'best_pair_ru', 'n_competitors'],
    'same_name_excluded.csv': [
        'query', 'source_type', 'decision_status', 'disposition',
        'selected', 'n_rivals', 'rivals', 'reason'],
    # §three-tier delivery (user 2026-09-27): the ends the mapper decision
    # DECLINED but the derivation evidence reaches, with the decline
    # reason and — when --verify-suspects ran — the same-machinery
    # verification verdicts.  Advisory bin: never the headline counts.
    'disclosure_evidence.csv': [
        'query', 'source_type', 'target_type', 'decline_reason',
        'decision_status', 'verdict', 'rank_union'],
    'source_candidates.csv': [
        'source_bodyId', 'source_type', 'target_bodyId', 'target_type',
        'rank_union', 'jaccard', 'morph_v2_similarity', 'morph_bar',
        'morph_bar_kind', 'morph_qualified',
        'query', 'branch_source_type', 'branch_target_type', 'dup'],
    'out_map_expansion.csv': [
        'query', 'source_type', 'source_bodyId', 'target_bodyId',
        'target_type', 'rank_union', 'rank_union_rank', 'jaccard',
        'jaccard_rank', 'in_map', 'morph_v2_similarity', 'morph_bar',
        'morph_bar_kind', 'morph_qualified'],
    'backward_matches.csv': [
        'query', 'branch_source_type', 'branch_target_type',
        'member_bodyId', 'member_type', 'member_category', 'scan_role'],
    # Homolog · forward (user 2026-09-26): one row per appeared source
    # bodyId — the chain-best target plus the serialized top-3-rank_union
    # ∪ top-3-jaccard neighbourhood, captured while the stage-2 scan
    # frames are in memory.  Morph columns are display joins off the
    # artifacts that already scored the pair; nothing is re-scored here.
    'forward_matches.csv': [
        'source_bodyId', 'source_type', 'primary_target_bodyId',
        'primary_target_type', 'primary_jaccard', 'primary_rank_union',
        'primary_in_branch', 'forward_topN', 'n_scanned', 'scanned_at',
        'morph_v2_similarity', 'morph_pool_ref', 'morph_bar_kind'],
    # Homolog · backward (user 2026-09-26): one row per appeared target
    # bodyId scanned back against the whole source dataset (stage 5e, no
    # caps — the stage-5d caps belong to the evidence pass).  Same payload
    # rule, mirrored direction.
    'target_matches.csv': [
        'target_bodyId', 'target_type', 'pool_category', 'pool_branches',
        'primary_source_bodyId', 'primary_source_type', 'primary_jaccard',
        'primary_rank_union', 'primary_in_branch', 'backward_topN_union',
        'n_scanned', 'scanned_at', 'morph_v2_similarity', 'morph_pool_ref',
        'morph_bar_kind'],
    # one row per admitted PAIR: the bar's own decision, the flags the floors
    # left behind, the tier, and the morphology record that qualified it
    'pooling_candidates.csv': [
        'source_bodyId', 'source_type', 'target_bodyId', 'target_type',
        'jaccard', 'jaccard_rank', 'rank_union', 'rank_union_rank',
        'bar_rank', 'bar_metric', 'bar_top_n', 'tier', 'window_size',
        'below_jaccard_floor', 'below_rank_union_floor', 'outside_window',
        'supported_by', 'single_metric_support', 'map_tag', 'source_claimed',
        'in_scope', 'leaf', 'size_nm3',
        'size_universe_percentile', 'morph_gate', 'morph_bar_kind',
        'morph_similarity', 'morph_pool_ref', 'morph_bar',
        'morph_qualified', 'verdict_for_pair', 'mapper_cell',
        'mapper_verdict'],
    # one row per distinct TARGET: what the scenes and `(dup)` group on, plus
    # the mapper-only rows (in_pool=False) so one file answers "which target
    # neurons does either engine claim?"
    'pooling_pool.csv': [
        'target_bodyId', 'target_type', 'leaf', 'best_source_bodyId',
        'best_source_type', 'jaccard', 'jaccard_rank', 'rank_union',
        'rank_union_rank', 'best_bar_rank', 'bar_metric', 'bar_top_n',
        'window_size', 'size_nm3', 'size_universe_percentile', 'in_scope',
        'map_tag', 'tiers', 'n_sources', 'dup', 'n_rows_refused', 'in_pool',
        'morph_gate', 'morph_bar_kind', 'morph_similarity', 'morph_pool_ref',
        'morph_bar', 'morph_qualified', 'verdict_for_pair', 'mapper_cell',
        'mapper_verdict'],
    # one row per QUERIED SOURCE — the mode's own unit, and the denominator the
    # headline reads (a source that found nothing is named, never absent)
    'pooling_sources.csv': [
        'source_bodyId', 'source_type', 'n_admitted', 'n_in_pool',
        'n_refused', 'tier', 'best_target_bodyId', 'best_bar_rank',
        'jaccard', 'jaccard_rank', 'rank_union', 'rank_union_rank',
        'supported_by', 'single_metric_support', 'source_claimed',
        'morph_gate', 'morph_bar_kind', 'morph_similarity', 'morph_pool_ref',
        'morph_bar', 'morph_qualified', 'verdict_for_pair', 'no_finding'],
}

# The backward (target -> source) evidence columns ride the three expansion
# bins, the reverse-scan ledger and the bodyId-unique rollup.  ``BACKWARD_
# COLUMNS`` is the single source of truth for their order, so adding one there
# cannot desync an empty export's header from a populated one.  Connectivity
# only — no morphology is added here (plan-tmvev-backward-expansion-evidence.md
# D5).
for _bin_schema in ('examinees.csv', 'deep_candidates.csv', 'relatives.csv',
                    'family_candidates.csv', 'backward_matches.csv'):
    _RUN_CSV_SCHEMAS[_bin_schema] = (
        _RUN_CSV_SCHEMAS[_bin_schema] + BACKWARD_COLUMNS)
_RUN_CSV_SCHEMAS['gap_fill_dedup.csv'] = _RUN_CSV_SCHEMAS[
    'gap_fill_dedup.csv'] + [
        'backward_evidence', 'backward_top1_source_bodyId',
        'backward_top1_source_type', 'backward_top1_in_branch',
        'backward_shared_type_count', 'backward_n_out_of_branch',
        'backward_thin_evidence',
        'backward_own_type_rank_source_bodyId',
        'backward_own_type_rank_source_type', 'backward_own_type_via',
        'backward_own_type_shared_type_count',
        'backward_own_type_thin_evidence']
_RUN_CSV_SCHEMAS['gap_fill_levels.csv'] = _RUN_CSV_SCHEMAS[
    'gap_fill_levels.csv'] + ['backward_evidence']
# the opt-in rivals ledger rides the ordinary verdict-row shape plus the
# rival provenance columns added by the suspects pass.  Without a schema
# here an empty export is a bare newline (columns=None) — found on the
# r13 real-data run, where --verify-suspects had zero rivals to verify.
_RUN_CSV_SCHEMAS['suspects_verification.csv'] = (
    _RUN_CSV_SCHEMAS['validation_results.csv'] + [
        'rival_of', 'disposition', 'rival_has_own_clean_pair',
        'rival_reverse_target', 'rival_votes',
        'rival_population_source', 'rival_population_target'])
del _bin_schema


def _write_run_csv(run_dir: Path, name: str, rows: List[Dict]) -> None:
    """Write one run CSV at its :data:`RUN_FILE_LAYOUT` location with the
    registry's columns, so an empty export is a header-only file, never
    zero bytes."""
    _write_csv(run_file_path(run_dir, name, create_parent=True), rows,
               columns=_RUN_CSV_SCHEMAS.get(name))


def _write_csv(path: Path, rows: List[Dict],
               columns: Optional[List[str]] = None):
    # An empty rows list with known columns still writes the header —
    # a zero-byte file makes pd.read_csv raise EmptyDataError instead of
    # yielding an empty frame (review 2026-09-16).
    rows = rows or []
    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        if not col.endswith('bodyId'):
            continue
        # FAFB bodyIds are ~2**59, and a column that mixes them with blanks
        # has already been promoted to float64 by the DataFrame ctor — which
        # silently rounds (720575940623474019 -> ...474048).  Rebuild the
        # column from the raw row values so the int never sees a float.
        df[col] = pd.array([_int_or_na(r.get(col)) for r in rows],
                           dtype='Int64')
    df.to_csv(path, index=False)


def _int_or_na(v):
    """Raw row value as an ``Int64`` member, or NA for blank/NaN/garbage.

    Python ints, numpy ints and digit strings go through UNTOUCHED — routing
    a 2**59 bodyId through ``float`` is exactly the loss this guard exists to
    prevent, and a float64 that arrives here is already rounded upstream."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return pd.NA
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str) and v.strip().lstrip('-').isdigit():
        return int(v.strip())
    try:
        f = float(v)
    except (TypeError, ValueError):
        return pd.NA
    return pd.NA if f != f else int(f)


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


def select_null_sample(per_source: Dict[int, pd.DataFrame],
                       pool_set: set,
                       seen_null: set,
                       null_jaccard_max: float,
                       null_per_source_cap: int,
                       ) -> List[Tuple[int, int]]:
    """Deterministically pick the Rev 3.9 null-calibration sample.

    Per source, the candidates are the scanned targets OUTSIDE the pool
    with ``jaccard <= null_jaccard_max`` (provably unrelated by
    connectivity) that the invader feed did not already claim
    (``seen_null``).  The exclusion is deliberately MODE-INVARIANT: what
    this mode's candidate window consumed must not decide the backdrop,
    or the bar moves with the mode and re-grades rows that sit near it.
    Selection sorts by ``target_bid`` before the per-source cap, so the
    sample is a pure function of the scan rows — immune to
    DataFrame/scan order.

    Design stance (user 2026-09-17): the null bar gates an arbitrary
    floor — connectivity ranks, the user verifies — so a small sampled
    set recomputed per run is enough; there is deliberately NO cross-run
    persistence of the bar.

    Returns ``(source_bodyId, target_bid)`` pairs in selection order and
    updates ``seen_null`` in place.
    """
    per_src: Dict[int, int] = {}
    picked: List[Tuple[int, int]] = []
    for sbid in sorted(per_source):
        df = per_source[sbid]
        if per_src.get(sbid, 0) >= null_per_source_cap:
            continue
        nulldf = df[(~df['target_bid'].isin(pool_set))
                    & (df['jaccard'].notna())
                    & (df['jaccard'] <= null_jaccard_max)
                    & (~df['target_bid'].isin(seen_null))]
        for bid in sorted(int(b) for b in nulldf['target_bid'].tolist()):
            if per_src.get(sbid, 0) >= null_per_source_cap:
                break
            if bid in seen_null or bid in pool_set:
                continue
            per_src[sbid] = per_src.get(sbid, 0) + 1
            seen_null.add(bid)
            picked.append((sbid, bid))
    return picked


def mutual_best_assignment(pool_set: set,
                           per_source: Dict[int, pd.DataFrame],
                           val_rows: Optional[List[Dict]] = None,
                           allowed_verdicts=None
                           ) -> List[Tuple[int, int]]:
    """Mutual-best greedy 1:1 within the mapped pool.

    Only sources with a confident verdict (verified_strong / verified /
    borderline by default — unmatched sources are NOT assigned; they flow
    into gap fill) participate.  Each side's best partner is decided by the
    bodyId ordering chain (Jaccard first, rank_union as the tie-break); a
    pair is assigned only when both sides agree.
    """
    allowed = allowed_verdicts or ASSIGN_VERDICTS
    verdict_by_src = {}
    for r in (val_rows or []):
        verdict_by_src[r['source_bodyId']] = r.get('verdict')

    def _pool_best(df: pd.DataFrame) -> Optional[pd.Series]:
        pool_df = df[df['target_bid'].isin(pool_set)]
        if pool_df.empty:
            return None
        return order_by_chain(pool_df).iloc[0]

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
        if cur is None or chain_key(row) < chain_key(cur[1]):
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
#: `pooling` is deliberately NOT in `VALIDATION_MODES`/`MODE_RANK`: the nested
#: chain answers "what else belongs to the pool the mapper asserted", pooling
#: answers the unsupervised question, and putting it in the ladder would let
#: every `mode_at_least` comparison admit it.
POOLING_MODE = 'pooling'

# Tier values (validated in-map targets) and expansion values.
TIER_CATEGORIES = ('matched', 'verified', 'borderline', 'unmatched')
EXPANSION_CATEGORIES = ('sibling', 'candidates', 'family', 'relative',
                        'examinees')
# 'examinees' is the aggressive-only deep-window category, RENAMED from
# 'suspicious' (2026-09-18): the mapper's rival-suspects concept now owns
# that word.  Data-schema names that embed `suspicious_` (CSV columns,
# config fields, the res['suspicious'] key) keep their names for
# compatibility; only user-facing category values, filenames and labels
# changed.
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
# family > relative > examinees.  (Restored 2026-09-17: the precedence
# serves the GAP-FILL accounting — a bodyId proposed as a fill by one
# branch must count once.  It must not distort the family category,
# which describes map structure (the in-map minus map-covered
# remainder); the run report therefore renders family material complete
# from set_coverage and reconciles it against the dedup bins.)
DEDUP_RANK = {
    'matched': 9, 'verified': 8, 'borderline': 7, 'unmatched': 6,
    'sibling': 5, 'candidates': 4, 'family': 3, 'relative': 2,
    'examinees': 1,
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
    if m == POOLING_MODE:
        # `pooling` does not nest, so a legacy widening flag alongside it is a
        # contradiction, not a precedence question — say so and stop.
        if aggressive_expansion or pool_widen:
            raise ValueError(
                "--mode pooling is the unsupervised engine and does not nest "
                "with --aggressive-expansion / --pool-widen")
        return POOLING_MODE
    if m not in MODE_RANK:
        # A typo used to fall through to `restrictive`, i.e. the run reported
        # one mode while running another.
        raise ValueError(
            f'unknown validation mode {mode!r}; expected one of '
            f'{", ".join(VALIDATION_MODES)} or {POOLING_MODE}')
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
    family -> relative -> examinees.  Each rule is evaluated on the
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
        tt = str(target_type) if has_type_name(target_type) else None
        if tt is not None and branch_target_type is not None \
                and tt == str(branch_target_type):
            return ('family', True, False)
        if tt is not None and tt in candidate_types \
                and tt not in in_map_types:
            return ('relative', True, False)
    # 6. the aggressive-only deep window (floors v3 two-gate): a deep
    #    row becomes an examinee when it passes EITHER the qualified bar
    #    (it is then below-pool-best but fully morph-qualified) OR the
    #    loose suspicious bar; below both it stays out of scope, flagged
    #    as morph-failed (it failed the bar that would have admitted it).
    if is_deep and mode == 'aggressive':
        if morph_ok or suspicious_morph_ok:
            return ('examinees', True, False)
        return ('', False, True)
    return ('', False, False)


#: Type labels that are not labels.  A missing annotation reaches here as the
#: float NaN, whose ``str()`` is the four-character string ``'nan'`` — and that
#: string is truthy, so ``str(v or '')`` keeps it and an unannotated neuron reads
#: as its own type named "nan".  Measured on the landed pooling run: 6 of 393
#: rows exported ``target_type='nan'`` with ``in_scope=True`` and the leaf token
#: ``nan(no_source)``.  Every emptiness test in this mode goes through
#: :func:`has_type_name` for that reason.
UNSET_TYPE_LABELS = frozenset({'', '?', 'nan', 'NaN', 'NA', 'None', 'none'})


def has_type_name(value) -> bool:
    """True only when ``value`` is an actual type name."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip() not in UNSET_TYPE_LABELS


def type_name_of(values, bid, default: str = '?') -> str:
    """The type NAME of one bodyId out of a lookup dict, never the string 'nan'.

    ``target_id2type`` is built from a pandas column, so an unannotated neuron's
    entry is float ``nan`` and ``.get(bid, '?')`` returns it — the default only
    fires for a MISSING key.  Measured on the 2026-09-24 male-cns run: 178 rows
    of ``noise_filtered_candidates.csv`` and one ``gap_fill_proposals.csv`` row
    exported ``ahead_target_type`` / ``proposal_type`` as the literal 'nan',
    which is the hole :func:`has_type_name` closed on the pooling side and left
    open here.
    """
    value = (values or {}).get(int(bid))
    return value if has_type_name(value) else default


def check_pooling_target_supported(cfg) -> None:
    """Refuse a ``pooling`` run whose TARGET the morphology scorer cannot score.

    Pooling's tiers are defined morph-qualified, so the mode already refuses
    ``--no-morphology``.  The same invariant has a second half that refusal did
    not cover: a target outside the cross-dataset families (FAFB / male-cns /
    BANC) makes the pass raise *inside* stage P, where the engine records the
    error per row and keeps going — which is right for a transient scorer
    failure and wrong for a capability the run should never have started with.
    Measured on the 2026-09-25 FAFB->hemibrain run: 5.4 M pairs scanned, 1027
    rows admitted, every one ``morph_gate='error'``, and the run still published
    32 ``matched`` and 335 ``verified``.
    """
    from comparison.morph_cross_dataset import dataset_scope
    for ds in (cfg.source_dataset, cfg.target_dataset):
        scope = dataset_scope(str(ds))
        if not scope.get('ok'):
            raise ValueError(
                f"--mode pooling needs a morphology the target can actually "
                f"be scored with, and '{ds}' cannot: {scope.get('reason')} "
                "Run --mode restrictive/family/aggressive instead, or pick a "
                "FAFB / male-cns / BANC target.")


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
    if not has_type or not has_type_name(target_type):
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
    become ``examinees``; below it they stay out of scope."""
    if bars is None:
        return False
    return suspicious_qualified(bars, row.get('morph_v2_similarity'))


def morph_coverage_warning(scores: Dict[Tuple[int, int], Optional[float]],
                           target_dataset: str) -> Optional[str]:
    """Name the degradation when a morphology pass scores NOTHING it asked.

    An all-empty score map means the scorer never received a target skeleton,
    so every branch falls to the run null bar and the null bar has no sample
    either — the run then grades on connectivity while looking merely
    un-calibrated.  The stage-5 null line ("sample too thin, n=0") does not
    say that, and the scenes of such a run hold no target neurons at all.
    """
    if not scores or any(v is not None for v in scores.values()):
        return None
    return (f'morphology scored 0 of {len(scores)} requested pairs: no '
            f'{target_dataset} target skeleton was loadable (the scene '
            '"unavailable" lines in these notes name them). Every admission '
            'bar in this run is therefore the run null bar AND the null bar '
            'has no sample, so these verdicts are connectivity-only, not '
            'morphology-qualified.')


def fill_pair(row: Dict) -> Tuple[int, int]:
    """The ``(source_bodyId, target_bodyId)`` a gap-fill proposal asserts.

    A proposal is spelled from the side it FILLS: a ``source``-side row names
    the unpaired source in ``bodyId`` and its proposed target in
    ``proposal_bodyId``, and a ``target``-side row spells the same relation
    the other way round.  Reading the two columns in a fixed order asked
    Track A to render a target bodyId out of the SOURCE dataset, so no
    target-side fill ever scored (measured: 0 of 76 in the male-cns family
    baselines and 0 of 2 in the hemibrain probe, against 87 of 88
    source-side rows).
    """
    if row.get('side') == 'target':
        return int(row['proposal_bodyId']), int(row['bodyId'])
    return int(row['bodyId']), int(row['proposal_bodyId'])


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
        # best (chain-first) row per pool target: the source that carries
        # this target highest, Jaccard leading and rank_union as the
        # tie-break — the same key the assignment and gap fill use, so a
        # target's `best_source_bodyId` cannot disagree with its pair.
        for tbid in pool_set:
            r = pool_df[pool_df['target_bid'] == tbid]
            if r.empty:
                continue
            row = order_by_chain(r).iloc[0]
            cur = best.get(tbid)
            key = chain_key(row)
            if cur is None or key < cur['key']:
                best[tbid] = {'key': key,
                              'source': sbid,
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


def categorize_pool_sources(per_source: Dict[int, pd.DataFrame],
                            target_pool: set, source_pool: set,
                            top_n: int = 2, invader_max: int = 3,
                            matched_ru_min: float = 0.1,
                            reverse_columns: Optional[Dict[int, List[Dict]]]
                            = None
                            ) -> Tuple[Dict[int, str], List[Dict]]:
    """Backward mirror of `categorize_pool_targets` (plan
    plan-backward-source-status.md): read-only `source-` status per
    in-branch source bodyId, from the SAME bodyId-bodyId pair scores —
    the column view (per target, rank the sources) beside the forward row
    view (per source, rank the targets).

    ``target_pool`` defines the columns; ``source_pool`` defines WHO gets
    a status (the branch's own source pool — D-B7, in-branch sources
    only).  ``reverse_columns`` replaces a column with the ranking produced
    by a stage-5d REVERSE scan (target -> whole source universe): without it
    a column can only ever contain the branch's own sources, so no
    out-of-branch competitor can appear above one and ``n_competitors`` is
    structurally 0.

    source-matched    column-top-1 (either metric) of its own row-best
                      pool target AND pair rank_union > matched_ru_min
    source-verified   column-top-1 (either metric) of some pool target,
                      or a column's top-N sources are ALL in-pool
    source-borderline not column-top-1, but at most invader_max
                      OUT-OF-POOL sources rank above it in its best
                      column
    source-unmatched  more out-of-pool sources above / no ranked rows
                      into the pool

    ADVISORY ONLY (D-B11): the statuses are user-read interpretation —
    they never gate, never enter the dedup, and never rewrite the
    mapping. The targets remain the validated entities.

    Returns ({source_bid: status}, per-source detail rows).
    """
    # build the branch-local columns: per pool target, every source row
    columns: Dict[int, List[Dict]] = {t: [] for t in target_pool}
    for sbid, df in per_source.items():
        sub = df[df['target_bid'].isin(target_pool)]
        for r in sub.itertuples(index=False):
            ru = getattr(r, 'rank_union')
            jac = getattr(r, 'jaccard')
            ru_nan = ru is None or pd.isna(ru)
            jac_nan = jac is None or pd.isna(jac)
            if ru_nan and jac_nan:
                continue
            columns[int(r.target_bid)].append({
                'source': sbid,
                'ru': None if ru_nan else float(ru),
                'jac': None if jac_nan else float(jac),
                'in_pool': sbid in source_pool,
            })
    for t, entries in (reverse_columns or {}).items():
        t = int(t)
        if t in columns and entries:
            columns[t] = list(entries)

    def _or_min(v):
        """Score as a "higher is better" number; blank/NaN sinks."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return -np.inf
        return f if f == f else -np.inf

    def _order(entries, key):
        # deterministic: higher score first (both metrics: higher is
        # better; NaN sinks), then bodyId
        return sorted(entries,
                      key=lambda e: (e[key] is None,
                                     -(e[key] if e[key] is not None
                                       else 0.0),
                                     e['source']))

    def _order_chain(entries):
        """The column in :data:`_CHAIN` order — Jaccard first, rank_union
        breaking the tie (J3), source bodyId last."""
        return sorted(entries,
                      key=lambda e: (-_or_min(e['jac']), -_or_min(e['ru']),
                                     e['source']))

    col_top1_ru: set = set()
    col_top1_jac: set = set()
    topn_verified: set = set()
    for t, entries in columns.items():
        by_ru = _order(entries, 'ru')
        by_jac = _order(entries, 'jac')
        if by_ru:
            col_top1_ru.add(by_ru[0]['source'])
        if by_jac:
            col_top1_jac.add(by_jac[0]['source'])
        # the forward top-N mirror: a column whose top-N sources are ALL
        # in-pool admits those N sources as source-verified
        for ordered in (by_ru, by_jac):
            head = ordered[:top_n]
            if len(head) == top_n and all(e['in_pool'] for e in head):
                topn_verified.update(e['source'] for e in head)

    verified_sources = col_top1_ru | col_top1_jac | topn_verified

    statuses: Dict[int, str] = {}
    detail: List[Dict] = []
    for s in sorted(source_pool):
        # row-best pool target: the target this source prefers, read off the
        # chain (Jaccard first, rank_union breaking the tie)
        own_rows = [(t, e) for t, entries in columns.items()
                    for e in entries if e['source'] == s]
        row_best = (min(own_rows, key=lambda te: (
            -_or_min(te[1]['jac']), -_or_min(te[1]['ru']), te[0]))[0]
            if own_rows else None)
        best_col = None
        best_rank = None
        competitors = None
        best_ru = None
        for t, e in own_rows:
            by_chain = _order_chain(columns[t])
            pos = next(i for i, x in enumerate(by_chain, start=1)
                       if x['source'] == s)
            if best_rank is None or pos < best_rank:
                best_col = t
                best_rank = pos
                competitors = sum(1 for e2 in by_chain[:pos - 1]
                                  if not e2['in_pool'])
                best_ru = e['ru']
        if s in verified_sources and row_best is not None \
                and best_col == row_best and best_rank == 1 \
                and best_ru is not None and best_ru > matched_ru_min:
            statuses[s] = 'source-matched'
        elif s in verified_sources:
            statuses[s] = 'source-verified'
        elif competitors is not None and competitors <= invader_max:
            statuses[s] = 'source-borderline'
        else:
            statuses[s] = 'source-unmatched'
        detail.append({
            'source_bodyId': s,
            'status': statuses[s],
            'best_column_target': best_col,
            'col_rank': best_rank,
            'best_pair_ru': best_ru,
            'n_competitors': competitors,
        })
    return statuses, detail


def morph_auc(verified_scores: List[float],
              suspicious_scores: List[float]) -> Optional[float]:
    """P(verified morph score > examinee morph score) via Mann-Whitney U."""
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
            with open(self.path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def ensure_local_release_data(dataset: str, project_root=None) -> bool:
    """Prepare a local-release target the way every other entry point does.

    Round-6 finding F-P2: with an absent BANC target the pair resolver
    collapsed every pool to '(untyped)' and the run exited 0 having
    validated nothing, because nothing between the CLI and the resolver
    runs the public-bucket preparation that coana / visualize_skeleton /
    the converters all perform. This gate gives the TM VEV entry points
    the same behaviour: BANC targets auto-prepare on first use; a failed
    preparation refuses the run instead of resolving against a universe
    that does not exist. Returns True when the data is present/prepared.
    """
    from flywire_ids import is_banc_dataset
    if not is_banc_dataset(dataset):
        return True  # NeuPrint targets answer to the cache gates instead
    try:
        from BANC_file_converter import ensure_banc_data
    except ImportError:
        from .BANC_file_converter import ensure_banc_data  # type: ignore
    root = Path(project_root) if project_root else \
        Path(__file__).resolve().parents[2]
    dataset_dir = root / 'datasets'
    try:
        return bool(ensure_banc_data(dataset, dataset_dir))
    except Exception as exc:  # noqa: BLE001 — refusal, not a crash
        print(f'! [TMVEV] target preparation for {dataset} failed: {exc!r}')
        return False


class MappingValidator:
    """Orchestrates the validation run and writes all outputs."""

    def __init__(self, cfg: MappingValidationConfig):
        self.cfg = cfg
        if str(getattr(cfg, 'route_scope', 'curated')) != 'curated':
            # §full-map boundary (plan-full-map-route-scope.md §6): the
            # validation pipeline grades the mapper's CLAIM tier only.  A
            # full-map scope exists only in the Type Mapping panel's
            # display layer; hard-refuse it here so a wiring mistake can
            # never validate panel-composed pairs.
            raise ValueError(
                "route_scope must be 'curated' — full-map (transitive "
                "composition) results are panel-display-only and are not "
                "accepted by the validation pipeline")
        # fail before any stage runs, not after 30 minutes of scanning: a
        # pooling run without morphology cannot qualify its own tiers (the
        # CLI refuses it too; this is the backend/UI-payload path)
        if cfg.effective_mode == 'pooling' and not cfg.morph_enabled:
            from comparison.mapping_validation_pooling import (
                check_morphology_mandatory)
            check_morphology_mandatory(cfg)
        # ... and the same invariant has a second half: the TARGET must be one
        # the cross-dataset scorer can score at all.  A hemibrain pooling run
        # (2026-09-25) scanned 5.4 M pairs, published 32 `matched` and 335
        # `verified` claims, and had graded NONE of them — the morph pass raised
        # "not supported for cross-dataset morphology", recorded it per row as
        # `morph_gate='error'`, and the tiers stayed.
        if cfg.effective_mode == 'pooling':
            check_pooling_target_supported(cfg)
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
        # What the run actually scored against, published to parameters.json —
        # the scan's stores in _record_scan_universe(), the morphology stores in
        # _record_morph_stores().
        self.input_fingerprint: Dict[str, Any] = {}

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
        (same-name passes; basis flagged 'full population').  Per side, the
        source pool prefers ANY supported chain of the same endpoint that
        narrows it — the selected chain is the best single DERIVATION,
        which on FAFB->BANC is a target-side-only hop (§15.3).  Branch
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
                if body_ids and not any(id2type.values()):
                    # F-P2: an empty lookup is a TABLE/read failure, not a
                    # data-quality fact — say so before the '(untyped)'
                    # bucket reads like an accusation against the neurons.
                    self.log(
                        '! [resolution] type lookup returned nothing for '
                        f'{len(body_ids)} bodyIds of {query!r} in '
                        f'{cfg.source_dataset} (local neuron table '
                        'unreadable or absent?) — the "(untyped)" bucket '
                        'below names the LOOKUP GAP, not the data')
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
                if pair.pool_basis in SOURCE_REFINING_BASES:
                    self.log(
                        f'  branch {src_type} -> {pair.target_type}: '
                        f'pool {len(pair.source_pool)} of '
                        f'{pair.source_type_total} via '
                        f'[{pair.linker_values or "chain"}]'
                        + (' [source side from an alternative chain]'
                           if pair.source_chain != pair.selected_chain else
                           '')
                        + f' (target {len(pair.target_pool)} of '
                        f'{pair.target_type_total})')
                else:
                    # `full population` here is NOT always a missing chain:
                    # a supported chain can carry no linker column on the
                    # SOURCE side (the r22 FAFB->BANC chains resolve through
                    # the target-side `fafb_cell_type` hop only), so the
                    # target pool refines while the source pool stays wide.
                    self.log(
                        f'  branch {src_type} -> {pair.target_type}: '
                        f'{pair.pool_basis} pool {len(pair.source_pool)} '
                        + ('(no supported bridge chain)'
                           if not pair.selected_chain else
                           '(no supported chain narrows the source side)'))
        return pairs

    def _refine_pair_branch(self, pair: TypePair) -> bool:
        """Replace the pair's pools with the selected bridge's refined
        subsets when a supported chain exists — and only when that chain
        ENDS at the pair's own target type (the prioritizer otherwise
        falls back to any supported chain of the source type).  Returns
        True when the pair keeps a refined pool; fail-open to full pools
        otherwise (Revision 3.3: evidence_only branches without a supported
        bridge are DROPPED by the caller instead — a wide fan-out must not
        be validated against full populations)."""
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
        # The prioritizer falls back to ANY supported chain of the source
        # type when no chain ends at the requested target; a chain that
        # reaches a different type never narrowed THIS pair.  Accepting it
        # recorded provenance for a bridge that does not describe the pair
        # (44 MCNS->FAFB + 3 FAFB->hemibrain real cases, 2026-09-25) and
        # would substitute another type's neurons the day such a chain
        # resolves target rows — so fail open to the full pools instead.
        if str(pool['selected_chain'][-1].get('value') or '') \
                != str(pair.target_type):
            return False
        src_ids = [int(b) for b in (pool.get('source_body_ids') or [])]
        tgt_ids = [int(b) for b in (pool.get('target_body_ids') or [])]
        if not src_ids:
            return False
        per_linker = list(pool.get('per_linker') or [])
        pair.pool_basis = pool.get('source_basis') or 'linker rows'
        # TMV-7: a supported chain that resolves ZERO target bodyIds leaves
        # the pair on its full-population target pool — the basis must say
        # so (the exports claimed 'linker rows' over a full population).
        pair.target_pool_basis = (
            pool.get('target_basis') or 'linker rows') if tgt_ids \
            else 'full population'
        pair.source_chain = [dict(h) for h in pool['selected_chain']]
        # PER-SIDE BASIS (plan-tmvev-jaccard-primary-bodyid-ranking.md
        # §15.3): `selected_chain` is the best single DERIVATION, and the
        # prioritizer ranks by fewest linkers.  On FAFB->BANC that winner is
        # the 1-linker target-side hop `banc_v888/fafb_cell_type`, which
        # names the target neurons exactly but leaves the source pool at the
        # WHOLE source type (r22: 756 pool slots for 242 neurons, 39/39
        # branches wide), while 7 of that type's 13 supported chains carry
        # source-side `additional_type(s)` rows for the same endpoint.  Take
        # the source pool from the first chain that narrows it.  NARROWING
        # ONLY, never a union — the retired Rev 3.11 pool widening merged
        # alternative-chain members into the tier; this leaves every member
        # outside the pool to `compute_out_map_by_type` (the out-map
        # residue), so the tier stays mode-invariant as required.
        if pair.pool_basis not in SOURCE_REFINING_BASES:
            cand = pool.get('source_side_refinement') or {}
            cand_chain = [dict(h) for h in (cand.get('chain') or [])]
            cand_ids = [int(b) for b in (cand.get('source_body_ids') or [])]
            parent = {int(b) for b in pair.parent_source_pool}
            same_endpoint = bool(cand_chain) and \
                str(cand_chain[-1].get('value') or '') == pair.target_type
            if same_endpoint:
                if parent:
                    cand_ids = [b for b in cand_ids if b in parent]
                # A SUBSET only: the rule may narrow the source pool or
                # re-describe the same neurons with better evidence
                # (`linker rows` instead of `full population`), and may
                # never widen or replace membership.
                if cand_ids and set(cand_ids) <= set(src_ids):
                    src_ids = cand_ids
                    pair.pool_basis = (cand.get('source_basis')
                                       or 'linker rows')
                    pair.source_chain = cand_chain
                    per_linker = [
                        l for l in per_linker
                        if l.get('home') != pair.source_dataset] + [
                        l for l in (cand.get('per_linker') or [])
                        if l.get('home') == pair.source_dataset]
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
            for l in per_linker
            if l.get('home') and l.get('column')
        ]
        if pair.pool_widen_added_sources or pair.pool_widen_added_targets:
            if pair.pool_basis == 'linker rows':
                pair.pool_basis = 'linker rows + alternative chains'
        pair.source_type_total = int(pool.get('source_type_total') or 0)
        pair.target_type_total = int(pool.get('target_type_total') or 0)
        return True

    def _is_multivalue(self, name: str, dataset: str) -> bool:
        """P4: is this a comma-joined multi-value type cell?  Defensive
        against partial mapper instances (no is_multivalue_type)."""
        fn = getattr(self.mapper, 'is_multivalue_type', None)
        if fn is None or not name:
            return False
        try:
            return bool(fn(str(name), dataset))
        except Exception:  # noqa: BLE001
            return False

    def _record_same_name_excluded(self, src_type: str,
                                   snf: Dict, reason: str = 'same_name_fanout',
                                   status: str = '') -> None:
        """P2 accounting: a queried type whose same-name fan-out did NOT
        fire (gated_held / excluded_evidence_only), or a multivalue type
        cell (P4, reason='multivalue_cell').  Advisory record only — the
        type stays excluded exactly as before."""
        if not hasattr(self, '_same_name_excluded'):
            self._same_name_excluded = []
        self._same_name_excluded.append({
            'query': getattr(self, '_current_query', ''),
            'source_type': src_type,
            'decision_status': status,
            'disposition': snf.get('disposition') or '',
            'selected': snf.get('selected') or '',
            'n_rivals': len(snf.get('rivals') or []),
            'rivals': ';'.join(snf.get('rivals') or []),
            'reason': reason,
        })

    def _pairs_for_type(self, src_type: str, pool: List[int],
                        query: str) -> List[TypePair]:
        cfg = self.cfg
        self._current_query = query
        dec = self.mapper.get_mapping_decision(
            src_type, cfg.source_dataset, cfg.target_dataset)
        status = dec.get('status')
        snf = dec.get('same_name_first') or {}
        # §three-tier delivery (user 2026-09-27): record the DISCLOSURE
        # ends — bridge evidence the decision declined — regardless of
        # the status path below (conflict/unmapped types return early and
        # would otherwise lose them entirely).  Advisory only.
        for disc in (dec.get('disclosure_targets') or []):
            if not hasattr(self, '_disclosure_records'):
                self._disclosure_records = []
                self._disclosure_recorded = set()
            rkey = (str(query), str(src_type), str(disc.get('target')))
            if rkey in self._disclosure_recorded:
                continue
            self._disclosure_recorded.add(rkey)
            self._disclosure_records.append({
                'query': str(query),
                'source_type': str(src_type),
                'target_type': str(disc.get('target') or ''),
                'decline_reason': str(disc.get('reason') or ''),
                'decision_status': str(status),
            })
        if status in ('conflict', 'unmapped'):
            if self._is_multivalue(src_type, cfg.source_dataset):
                # P4: multi-value cells stay ATOMIC (locked decision) —
                # accounted, never split.
                self._record_same_name_excluded(
                    src_type, {'disposition': 'multivalue_cell'},
                    reason='multivalue_cell', status=str(status))
                self.log('! [same-name-first] '
                         f'{src_type}: multi-value type cell (kept '
                         'atomic) — excluded from validation')
            elif snf and not snf.get('fires', False):
                # P2: the same-name fan-out was HELD (gated_held) —
                # record it and say WHY the type is missing.
                self._record_same_name_excluded(src_type, snf,
                                                status=str(status))
                self.log('! [same-name-first] '
                         f'{src_type}: same-name fan-out held '
                         f'({snf.get("disposition")}) — selection '
                         f'{snf.get("selected")!r} withheld; '
                         f'{len(snf.get("rivals") or [])} rival(s) in '
                         'auto_type_mapping_suspects.csv; the run '
                         'continues without this type')
            else:
                self.log(f'  - {src_type}: {status} — excluded (fail-closed)')
            return []
        # P2: an evidence_only-path fan-out never fires (locked §0.0
        # disposition) — record it, then let the ordinary evidence_only
        # branch rules below decide pair survival.
        if snf and not snf.get('fires', False) \
                and snf.get('disposition') == 'excluded_evidence_only':
            self._record_same_name_excluded(src_type, snf,
                                            status=str(status))
            self.log('! [same-name-first] '
                     f'{src_type}: evidence-only fan-out (N-to-1 '
                     'convergence view) excluded by policy — rivals in '
                     'auto_type_mapping_suspects.csv')
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
            if self._is_multivalue(tgt_type, cfg.target_dataset):
                self.log(f'  - {tgt_type}: multi-value target type cell '
                         '(kept atomic) — skipped')
                if not hasattr(self, '_multivalue_target_skips'):
                    self._multivalue_target_skips = 0
                self._multivalue_target_skips += 1
                continue
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
            # P1: mark FIRED same-name-first selections (advisory only).
            if snf.get('fires'):
                pair.same_name_first = {
                    'selected': snf.get('selected'),
                    'rivals': list(snf.get('rivals') or []),
                    'path': snf.get('path'),
                    'disposition': snf.get('disposition'),
                }
            pair.parent_source_pool = list(pair.source_pool)
            pair.parent_target_pool = list(pair.target_pool)
            if not self._refine_pair_branch(pair) \
                    and status in ('evidence_only', 'valid_split_evidence'):
                # TMV-2 (2026-09-26): the Rev 3.3 rationale — a wide
                # fan-out must not be validated against full populations —
                # covers split branches too.  A split branch whose chain
                # is unsupported used to stay on the full population (and
                # lose its disjointness annotation silently) beside
                # linker-row verdicts for its siblings.
                self.log(f'  - {src_type} -> {tgt_type}: {status} '
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
        fallback caliber).  Returns {'rows', 'suspicious' (the examinee
        rows — the res key keeps its pre-rename name), 'noise',
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
            # J1/chain: the published row is the CHAIN-best pool member
            best = order_by_chain(pool_df).iloc[0]
            best_by_src[sbid] = best
            k = cfg.rank_top_k
            # D-A (plan-tmvev-jaccard-primary-bodyid-ranking): the ladder
            # asks the POOL, not one chosen row — "rank 1 by either metric"
            # is a set property, so switching the ordering key cannot move a
            # verdict.  Positivity stays a LABEL rule (J2): a non-positive
            # rank_union cannot carry the rank_union claim, but it never
            # drops the row (user 2026-09-11: only a POSITIVE rank_union is
            # strong evidence; jaccard is sign-free and unaffected).
            ru_tied = pool_df[pool_df['rank_union_rank'] == 1]
            ru_top = order_by_chain(ru_tied[ru_tied['rank_union'] > 0])
            ru1 = not ru_top.empty
            # r17's correction (plan §11, user 2026-09-20): the STRONG tier
            # additionally requires BOTH claims on the SAME pool member.
            # `*_rank == 1` is a competition (min) rank, so "∃ a rank_union
            # top-1" and "∃ a jaccard top-1" are satisfied by two different
            # neurons on 46 of r17's rows — which read verified_strong while
            # no single partner was agreed by both metrics.  Quantifying over
            # the members keeps the property ordering-invariant; requiring
            # one member to satisfy both keeps the tier's meaning.  A member
            # holding both claims is necessarily the chain best, so `best`
            # needs no second lookup: `verified_strong` always publishes the
            # record it rests on.
            strong = not pool_df[(pool_df['rank_union_rank'] == 1)
                                 & (pool_df['rank_union'] > 0)
                                 & (pool_df['jaccard_rank'] == 1)].empty
            # The same rule one tier down (user 2026-09-21): a rank-1 claim
            # lifts the row only when the PUBLISHED partner holds it.  The
            # jaccard side is equivalent either way (`best` is the pool's max
            # jaccard, so it shares any member's rank 1); the rank_union side
            # is not, and that is the point — a rank_union top-1 sitting on a
            # different neuron stays visible as `ru_top_target_bodyId`
            # evidence instead of becoming this row's verdict.  r18 measured
            # the effect on real data: 2 of 223 rows (the only two reading
            # `verified` on a rank_union claim their published partner does
            # not hold).  Ordering-invariance survives because the published
            # row is a function of the pool, not of a tie's accident (the
            # chain resolves ties on the scores).
            pub_ru1 = (float(best['rank_union_rank'] or 0) == 1
                       and float(best['rank_union'] or -1) > 0)
            pub_ja1 = float(best['jaccard_rank'] or 0) == 1
            # the pool's best position on each metric — what a rival has to
            # beat (D-B), and the row carrying the rank_union claim when
            # that is a different neuron from the published one
            pool_ru_rank = pool_df['rank_union_rank'].min()
            pool_ja_rank = pool_df['jaccard_rank'].min()
            # the same pool-wide reference for the tie-margin noise gate: a
            # rival "ties" the pool when it nearly beats the pool's BEST
            # rank_union, not the published row's
            pool_best_ru = pool_df['rank_union'].max()
            ru_claim_bid = (int(ru_top.iloc[0]['target_bid'])
                            if ru1 and not strong else None)
            flags = []
            if not ru1 and not ru_tied.empty:
                flags.append('negative_rank_union_top1')
            if strong:
                verdict, which = 'verified_strong', 'both'
            elif pub_ru1 or pub_ja1:
                verdict = 'verified'
                which = '+'.join(n for n, top in
                                 (('rank_union', pub_ru1),
                                  ('jaccard', pub_ja1))
                                 if top)
            elif min(_rank_or_inf(pool_ru_rank),
                     _rank_or_inf(pool_ja_rank)) <= k:
                verdict, which = 'borderline', ''
            else:
                verdict, which = 'unmatched', ''

            # examinees (sus_rows): non-pool neurons ranked ahead of the best pool
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
                # D-A/D-B: a rival is "ahead of the pool" when it beats the
                # pool's BEST position on that metric, not the published
                # row's — the published row is the chain (Jaccard) best and
                # its rank_union_rank can sit well below the pool's own best.
                best_rank = (pool_ru_rank if metric == 'rank_union'
                             else pool_ja_rank)
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
                        elif not pd.isna(pool_best_ru) \
                                and not pd.isna(r.rank_union) \
                                and (float(r.rank_union)
                                     - float(pool_best_ru)) \
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
                tie_filtered=tie_count, ru_top=ru_claim_bid))

            # bounded per-source retention for gap fill + assignment
            keep = set(pool_set)
            for metric in ('rank_union', 'jaccard'):
                ranked = df[df[metric].notna()].nsmallest(
                    FILL_KEEP_TOP, f'{metric}_rank')
                keep.update(int(b) for b in ranked['target_bid'])
            per_source[sbid] = df[df['target_bid'].isin(keep)]

        # Revision 3.9 deep window, widened into a candidate-DISCOVERY
        # window that family MODE reads too (user 2026-09-21).  Out-of-pool
        # neurons within the retained top-`k` of EACH metric that are not
        # ahead of the pool best — the homologs a strong pool member can
        # hide just below it.  Same spatial-caliber gate; structural + morph
        # qualification happen later (annotate_invaders + 5).
        #
        # Why a window and not the invader bar: the bar above answers "does
        # a non-pool neuron beat the pool's BEST position?", and for a
        # well-validated pool the answer is structurally NO — r18 published
        # global jaccard rank 1 on 190 of 223 rows, so `pool_ja_rank` = 1
        # closed that window on ~85% of source neurons.  Candidate TYPES rode
        # on that bar, and rule 5 of `classify_category` seeds the whole
        # `relative` bin from them, so the better a branch validated the less
        # it reported: r16 -> r18 examinees 168 -> 118, candidate types
        # 5 -> 3 (CB4091 and SMP223 dropped out), `relatives.csv` 39 rows
        # -> 1.  A window reads the top-k of jaccard AND of rank_union, so
        # one metric's top-1 cannot hide the other's candidates.
        # `--candidate-window` stays the aggressive window; family reads the
        # borderline window `--rank-top-k`; restrictive keeps its documented
        # invader-only feed.
        deep_rows: List[Dict] = []
        # TWO windows, TWO budgets.  The borderline window every family run
        # reads and the deep window only aggressive reads used to share ONE
        # per-source cap, so widening the mode spent that budget on deep rows
        # and never reached the tight rows the narrower mode had kept —
        # measured 2026-09-24 as 110 of 189 shared (source, target) pairs
        # displaced on male-cns and 262 of 408 on BANC, which then deleted their
        # `relative` rows (129 → 49 targets) and shrank the layered gap-fill
        # report (153 → 63 rows): the wider question reported LESS evidence.
        # The two bands now also run as two PASSES, because one per-source
        # budget per band is not enough on its own: a row's band is its rank in
        # the metric that surfaced it, and within a tie on the chain's leading
        # metric a deep-band row can be reached — and can spend the budget —
        # before the borderline row it is tied with.  Measured on the
        # post-fix 2026-09-24 male-cns run, 63 of family's 189 borderline pairs
        # still vanished that way.  Reading the borderline band first, with the
        # filter family uses, makes the displacement impossible rather than
        # unlikely: aggressive adds its deep band on top of what is left.
        tight_per_src: Dict[int, int] = {}
        deep_per_src: Dict[int, int] = {}
        null_rows: List[Dict] = []
        sus_targets = {int(r['ahead_target_bodyId']) for r in sus_rows}
        seen_deep = set(sus_targets)
        wide = cfg.mode_at_least('aggressive')

        def window_pass(k, budget, band, borderline):
            """One band's rows, in chain order, under one per-source budget."""
            for sbid, df in per_source.items():
                src_best = best_by_src.get(sbid)
                if src_best is None:
                    continue
                # The chain's leading metric fills the budget first (J1):
                # rank_union-first here let it consume the budget before the
                # Jaccard window was ever read.
                for metric in ('jaccard', 'rank_union'):
                    if budget.get(sbid, 0) >= cfg.deep_cap:
                        continue
                    window = df[(~df['target_bid'].isin(pool_set))
                                & df[f'{metric}_rank'].notna()
                                & (df[f'{metric}_rank'] <= k)]
                    # Read the window in chain order so WHICH rows survive a
                    # per-source budget never moves with the frame order.
                    window = order_by_chain(window)
                    for r in window.itertuples(index=False):
                        bid = int(r.target_bid)
                        if bid in seen_deep or bid in pool_set:
                            continue
                        rank = float(getattr(r, f'{metric}_rank'))
                        if borderline and rank > cfg.rank_top_k:
                            continue     # this pass reads the borderline band
                        if not borderline and rank <= cfg.rank_top_k:
                            continue     # the deep band never re-reads it
                        if budget.get(sbid, 0) >= cfg.deep_cap:
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
                        budget[sbid] = budget.get(sbid, 0) + 1
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
                            'candidate_source': band,
                        })
                        deep_rows.append(row)

        if cfg.mode_at_least('family'):
            window_pass(cfg.rank_top_k, tight_per_src, 'top_window', True)
            if wide:
                window_pass(cfg.candidate_window, deep_per_src,
                            'deep_window', False)
        # Rev 3.9 null sample for the Track-A bar calibration: window
        # rows with near-zero jaccard are provably unrelated by
        # connectivity — their Track-A morph distribution IS the
        # cross-dataset baseline (T1/R1-R6 live here).  Independent of
        # the deep window (works in the default mode); calibration data
        # only: never exported as candidates.  Selection is
        # deterministic (sorted by target bid, user 2026-09-17: a small
        # sampled set is enough for the arbitrary floor gate — no
        # cross-run persistence).
        # The backdrop excludes only what is MODE-INVARIANT: the invader feed
        # and the pool.  It used to share `seen_deep`, which accumulates this
        # mode's own window, so the eligible pool shrank as the mode widened and
        # the p95 moved with it (0.143561 / 0.143123 / 0.150580 on male-cns,
        # 2026-09-24) — enough to flip the category of any row sitting inside
        # that drift: one BANC pair holds `morph_v2_similarity` 0.2642521 in all
        # three runs and reads `candidates` / out-of-scope / `candidates` as its
        # bar moves 0.263224 → 0.264533 → 0.256532.  A candidate that is also
        # near-zero-connectivity belongs in the backdrop by the stated criterion,
        # so sharing the exclusion bought nothing and cost the invariant.
        seen_null = set(sus_targets)
        eligible = {s: df for s, df in per_source.items()
                    if best_by_src.get(s) is not None}
        for sbid, bid in select_null_sample(
                eligible, pool_set, seen_null,
                cfg.null_jaccard_max, cfg.null_per_source_cap):
            df = per_source[sbid]
            r = df[df['target_bid'] == bid].iloc[0]
            src_best = best_by_src.get(sbid)
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
                 tie_filtered: int = 0,
                 ru_top: Optional[int] = None) -> Dict:
        row = {
            'query': pair.query,
            'source_dataset': pair.source_dataset,
            'source_type': pair.source_type,
            'target_dataset': pair.target_dataset,
            'target_type': pair.target_type,
            'mapping_status': pair.status,
            'relationship': pair.relationship,
            'same_name_first': bool(pair.same_name_first),
            'same_name_rivals': ';'.join(
                pair.same_name_first.get('rivals') or [])
            if pair.same_name_first else '',
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
            # D-A traceability: `target_bodyId` is the CHAIN-best pool member
            # (Jaccard first), so when the verdict's rank_union claim is
            # carried by a DIFFERENT pool member it is named here rather than
            # left implicit — a verdict never rests on an unpublished record.
            'ru_top_target_bodyId': ru_top,
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
            'ahead_target_type': type_name_of(target_id2type, r.target_bid),
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
            'same_name_first': bool(pair.same_name_first),
            'same_name_rivals': ';'.join(
                pair.same_name_first.get('rivals') or [])
            if pair.same_name_first else '',
            'pool_basis': pair.pool_basis,
            'target_pool_basis': pair.target_pool_basis,
            'selected_chain': pair.chain_text,
            'source_chain': pair.source_chain_text,
            'branch_linker_values': pair.linker_values,
            'branch_annotation': pair.branch_annotation,
            'branches_disjoint': pair.branches_disjoint,
            'source_pool': n_src, 'target_pool': n_tgt,
            'pool_widen_added': len(pair.pool_widen_added_sources)
            + len(pair.pool_widen_added_targets),
            'source_type_total': pair.source_type_total,
            'target_type_total': pair.target_type_total,
            'best': len(assigned),
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

    def _resync_summary_verdicts(self, per_pair_res: Dict) -> None:
        """Recompute each pair summary's verdict counters from its final
        rows (TMV-3, matrix-measured 2026-09-25).

        Stage 5's AUC demotion rewrites ``row['verdict']`` AFTER
        ``_summary`` froze its counters, so ``pair_summary.csv`` reported
        pre-demotion counts while ``validation_results.csv`` carried the
        demoted rows (banc l-LNv -> l-LNv: summary strong=2/verified=6
        against 8/8 plain ``verified`` rows, ``demoted: 2``).  The summary
        dicts are shared with ``per_pair_res`` and its rows are the same
        objects stage 5 mutated, so this patches the counters in place —
        the Branches tab and the Validation tab then agree.
        """
        for res in per_pair_res.values():
            summary = (res or {}).get('summary') or {}
            rows = (res or {}).get('rows') or []
            if not summary or not rows:
                continue
            for verdict in ('verified_strong', 'verified', 'borderline',
                            'unmatched', 'skipped'):
                summary[f'verdict_{verdict}'] = sum(
                    r.get('verdict') == verdict for r in rows)

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
        examinee rows (Rev 3.6: a fragment proposed as a homolog is
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
            best = order_by_chain(cand).iloc[0]
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
                best = order_by_chain(cand).iloc[0]
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
                                  type_name_of(target_id2type, tbid)),
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
            sbid, r = min(cands, key=lambda cr: chain_key(cr[1]))
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
            except Exception as exc:
                # TMV-5: a silent mapper failure reads every invader as
                # 'unmapped' with zero trace — log the first one per run.
                if not getattr(self, '_backward_fail_logged', False):
                    self._backward_fail_logged = True
                    self.log(f'[backward] decision lookup failed for '
                             f'{atype!r} ({exc}); further failures stay '
                             'silent — invaders read unmapped')
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
        matched/…/sibling/candidates/family/relative/examinees), which
        runs after morphology.  This method still tags each row with the
        old evidence columns (`sibling_pool_of`, `backward_status`,
        `alt_chain_of_parent`, …) and the legacy counts.

        Rev 3.6 rule (unchanged here): sibling > alternate-chain >
        backward > unexplained; structural facts outrank morphology.
        """
        family_types = family_types or set()
        # sibling-branch pool membership (same parent, any branch)
        sibling_pools: Dict[int, List[Tuple[str, str, str]]] = {}
        for bkey, res in per_pair_res.items():
            src_type, tgt_type = bkey[-2], bkey[-1]
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
            if dec['status'] in ('mapped', 'bridged', 'valid_split_evidence') \
                    and dec['home_real']:
                if alt:
                    fields['invader_class'] = 'alternate-chain'
                    fields['invader_label'] = (
                        f'alternate-chain · {";".join(sorted(alt))} '
                        f'→ {atype}')
                else:
                    fields['invader_class'] = 'backward'
                    fields['invader_label'] = f'backward · {dec["mapped"]}'
            elif dec['status'] in ('mapped', 'bridged', 'valid_split_evidence'):
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
            if dec.get('status') in ('mapped', 'bridged', 'valid_split_evidence'):
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

        branch_pools: Dict[Tuple[str, str, str], set] = {}
        tiers: Dict[Tuple[str, str, str], Dict] = {}
        for key, res in per_pair_res.items():
            branch_pools[key] = {int(b) for b in (
                res.get('_pool_set') or [])}
            tiers[key] = res.get('target_categories') or {}
        # Branch keys are (query, source, target); fixture-driven
        # per_pair_res may still use the legacy (source, target) arity.
        # Pin the whole finalization to whichever arity is in use.
        legacy_branch_keys = any(len(k) == 2 for k in per_pair_res)
        in_map: set = set()
        for s in branch_pools.values():
            in_map |= s
        # Branch keys are (query, source, target); target is always last.
        in_map_types = {str(k[-1]) for k in branch_pools}
        self._in_map_ids = in_map
        self._in_map_types = in_map_types
        self._branch_pools = branch_pools

        def key_of(row) -> Tuple[str, ...]:
            # Rows carry their query (validate_pair stamps it on every
            # evidence/pool row); the query disambiguates branches that
            # resolve to the same concrete (source, target) type pair.
            full = (str(row.get('query') or ''), str(row['source_type']),
                    str(row['target_type']))
            return full[1:] if legacy_branch_keys else full

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
        # The candidate-discovery window (top-`rank_top_k` of each metric)
        # is connectivity evidence the same way an invader is: a real scan
        # row inside the review window.  It seeds `candidates`, and through
        # rule 5 of `classify_category` the whole `relative` bin — so these
        # rows must be qualified on connectivity, not merely residual.
        window_ids: Dict[Tuple[str, str], set] = defaultdict(set)
        for r in deep_rows:
            if str(r.get('candidate_source') or '') == 'top_window':
                window_ids[key_of(r)].add(int(r['ahead_target_bodyId']))

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
                  or bid in gapfire_ids.get(k, set())
                  or bid in window_ids.get(k, set()))
            tt = str(ttype) if has_type_name(ttype) else None
            cat, in_scope, mfail = classify_category(
                target_bid=bid, branch_pool=branch_pools.get(k),
                in_map=in_map, target_type=tt,
                branch_target_type=k[-1], in_map_types=in_map_types,
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

        # Phase B: residual bins (family / relative / examinees) for the
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
            tt = str(ttype) if has_type_name(ttype) else None
            cat, in_scope, mfail = classify_category(
                target_bid=bid, branch_pool=branch_pools.get(k),
                in_map=in_map, target_type=tt,
                branch_target_type=k[-1], in_map_types=in_map_types,
                candidate_types=candidate_types.get(k),
                connectivity_qualified=(bid in window_ids.get(k, set())),
                morph_ok=mq(row),
                suspicious_morph_ok=mq_suspicious(row),
                is_deep=True, tier=None, mode=mode)
            row['category'] = cat
            row['in_scope'] = bool(in_scope)
            row['morph_failed'] = bool(mfail)
            if str(cat):
                labeled[k].add(bid)

        # Pool (tier) rows.  `tiers` is keyed by the SAME branch arity as
        # per_pair_res — use key_of (NOT a hand-rolled (source, target)
        # tuple): the multi-query migration re-keyed tiers to
        # (query, source, target), and a 2-tuple lookup here silently
        # demoted every pool row to `unmatched` (found in the
        # 2026-09-17 reportcheck end-to-end run).
        for d in pool_detail:
            k = key_of(d)
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
                bt = key[-1]
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
            # relative / examinees) carries the per-bodyId token; the
            # order (out-map -> >src -> no_source -> untyped) is decided in
            # `_leaf_token`.
            if cat in ('family', 'candidates', 'relative', 'examinees'):
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
            if ann == 'untyped' or not has_type_name(ttype):
                n_untyped += 1
            elif ann.endswith('(no_source)'):
                gap_types[str(ttype)] += 1
        self._mapper_gap_types = gap_types
        self._mapper_gap_untyped = n_untyped

        # Revision 3.12: propagate the query-level `dup` flag back onto the
        # per-branch rows (the plan's bodyId-level `(dup)` tag), so the
        # CSVs and the scene both show which expansion neurons recur across
        # branches.  Siblings are excluded by construction (never `dup`).
        # Stash the partition inputs so stage 5d can relabel rows and REBUILD
        # this rollup without re-enumerating anything (`_backward_expansion_
        # pass`); the row dicts are shared, so an in-place label update is
        # picked up by the family/relative exports and the scenes.
        self._cat_evidence = evidence
        self._cat_pool_detail = pool_detail
        self._cat_family_rows = family_rows
        self._cat_relative_rows = relative_rows
        self._cat_per_pair_res = per_pair_res
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

    def _expansion_row(self, key: Tuple[str, ...], bid: int, tname: str,
                       category: str) -> Dict:
        if len(key) == 3:
            query, src_type, tgt_type = key
        else:  # legacy 2-tuple keys (fixtures)
            query, src_type, tgt_type = '', key[0], key[1]
        return {
            'query': query or next((p.query for p in self.pairs
                                    if p.source_type == src_type
                                    and p.target_type == tgt_type), ''),
            'source_type': src_type, 'target_type': tgt_type,
            'pool_basis': '', 'branch_linker_values': '',
            'branch_annotation': '', 'source_bodyId': None,
            'ahead_target_bodyId': int(bid),
            'ahead_target_type': tname if has_type_name(tname) else '?',
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
            # Connectivity-only placeholders; stage 5d overwrites them when
            # the backward pass is enabled (D5: no morphology here).
            **blank_backward_fields(),
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

        # Stage 5d carries the reverse grade onto the rollup so the layered
        # gap-fill report can split a bin by it.  The strongest label on a
        # bodyId wins (high > medium > low); `not-checked` never wins.
        rank_bev = {'low': 1, 'medium': 2, 'high': 3}
        bev_by_bid: Dict[int, Dict] = {}

        def note_bev(bid, row):
            lab = str(row.get('backward_evidence') or '')
            if lab not in rank_bev:
                return
            cur = bev_by_bid.get(int(bid))
            if cur is None or rank_bev[lab] > rank_bev.get(
                    str(cur.get('backward_evidence') or ''), 0):
                bev_by_bid[int(bid)] = row

        for row, bid, _t in evidence:
            note_bev(bid, row)
        for r in list(family_rows) + list(relative_rows):
            note_bev(r['ahead_target_bodyId'], r)
        for d in pool_detail:
            note_bev(d['target_bodyId'], d)

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
                'candidates', 'family', 'relative', 'examinees')
            bev = bev_by_bid.get(bid) or {}
            rows.append({
                'target_bodyId': bid,
                'target_type': rec['target_type_name'],
                'dedup_category': cat,
                'n_branches': n_branches,
                'dup': bool(dup),
                'counts_toward_restrictive_fill': cat == 'candidates',
                'counts_toward_family_fill': cat in
                ('candidates', 'family', 'relative'),
                'backward_evidence': bev.get('backward_evidence') or '',
                'backward_top1_source_bodyId':
                    bev.get('backward_top1_source_bodyId'),
                'backward_top1_source_type':
                    bev.get('backward_top1_source_type') or '',
                'backward_top1_in_branch':
                    bool(bev.get('backward_top1_in_branch')),
                'backward_shared_type_count':
                    bev.get('backward_shared_type_count'),
                'backward_n_out_of_branch':
                    bev.get('backward_n_out_of_branch'),
                'backward_thin_evidence':
                    bool(bev.get('backward_thin_evidence')),
                # the hit the winning grade rests on (D4: rides the rollup so
                # the gap-fill reader sees WHY a bodyId reads high)
                'backward_own_type_rank_source_bodyId':
                    bev.get('backward_own_type_rank_source_bodyId'),
                'backward_own_type_rank_source_type':
                    bev.get('backward_own_type_rank_source_type') or '',
                'backward_own_type_via':
                    bev.get('backward_own_type_via') or '',
                'backward_own_type_shared_type_count':
                    bev.get('backward_own_type_shared_type_count'),
                'backward_own_type_thin_evidence':
                    bool(bev.get('backward_own_type_thin_evidence')),
            })
        rows.sort(key=lambda r: (-dedup_category_rank(r['dedup_category']),
                                 r['target_bodyId']))
        return rows

    # -- stage 5 ----------------------------------------------------------

    def _preflight_target_skeletons(self, dataset: str, bids) -> Dict[str, int]:
        """Fetch the target skeletons this run is about to score against.

        Track A renders its targets out of the raw-skeleton cache, and
        ``load_skeleton`` never fetches — the fetching happens in the stage-4
        scenes, which run AFTER stage 5.  On a dataset with a cold cache that
        ordering scored 0 of 77 pairs while reading as ordinary thin-sample
        calibration (plan-tmvev-jaccard-primary-bodyid-ranking §17.2), so the
        run meant to grade morphology graded connectivity and the next run got
        different bars.  Pre-flighting here makes a dataset's first run behave
        like its second.

        Bounded by ``MORPH_SKELETON_PREFLIGHT_CAP`` and named in the log when
        it truncates: the pair frame of a wide aggressive run can ask for tens
        of thousands of targets, which is not a fetch budget.  The fetch goes
        through the shared batched fetcher (``cfg.skeleton_fetch_workers``
        threads over 64-body requests, in ``MORPH_SKELETON_FETCH_CHUNK``
        id-chunks) because this is the one network-latency-bound block in the
        run — fetching one at a time measured 3.0 s per target, 1,401 s for
        467.  Cache-first and resumable (each fetch persists into the raw
        store), and every failure is fail-open — the scorer then sees exactly
        what it used to see, and :func:`morph_coverage_warning` says so.  FAFB
        targets come from the healed-zip path, not this cache, so they are
        skipped.
        """
        cfg = self.cfg
        bids = sorted({int(b) for b in (bids or [])
                       if b is not None and str(b).strip()})
        stats = {'requested': len(bids), 'cached': 0, 'fetched': 0,
                 'failed': 0, 'skipped_cap': 0, 'unattempted': 0}
        if not bids or is_fafb_dataset(dataset):
            return stats
        try:
            from morphology import find_similar_raw_cache
            cache = find_similar_raw_cache(dataset, verbose=False)
            missing = [b for b in bids if cache.find_skeleton_file(b) is None]
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5] skeleton pre-flight unavailable: {exc}; '
                     'scoring against whatever the cache holds')
            return stats
        stats['cached'] = len(bids) - len(missing)
        if len(missing) > MORPH_SKELETON_PREFLIGHT_CAP:
            stats['skipped_cap'] = (len(missing)
                                    - MORPH_SKELETON_PREFLIGHT_CAP)
            self.log(f'[stage 5] ! skeleton pre-flight capped at '
                     f'{MORPH_SKELETON_PREFLIGHT_CAP} targets; '
                     f'{stats["skipped_cap"]} unfetched (their pairs stay '
                     'unscored, so their branches keep the null bar)')
            missing = missing[:MORPH_SKELETON_PREFLIGHT_CAP]
        self.log(f'[stage 5] skeleton pre-flight ({dataset}): '
                 f'{stats["cached"]}/{len(bids)} cached, fetching '
                 f'{len(missing)}')
        if not missing:
            return stats
        workers = max(1, min(int(getattr(cfg, 'skeleton_fetch_workers', 8) or 1),
                             len(missing)))
        timeout_s = max(0, int(getattr(cfg, 'skeleton_fetch_timeout_s', 120)
                               or 0))
        self.progress.emit('skeletons_progress', stage='5', dataset=dataset,
                           done=0, total=len(missing), workers=workers)
        t0 = time.time()
        done = 0
        # Neither neuprint-python nor navis exposes a per-request timeout, so
        # bound socket INACTIVITY for the duration of the fetch and restore it
        # afterwards: without this a single hung request stalls the stage
        # forever, which is also what the serial loop used to do.
        prev_timeout = None
        if timeout_s:
            import socket
            prev_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout_s)
        try:
            from morphology import fetch_skeletons_on_demand_batch
            for start in range(0, len(missing), MORPH_SKELETON_FETCH_CHUNK):
                chunk = missing[start:start + MORPH_SKELETON_FETCH_CHUNK]
                # The batcher persists AND returns every skeleton, so it runs
                # in chunks — the pre-flight needs the files on disk, not the
                # neurons in memory.
                got = fetch_skeletons_on_demand_batch(
                    dataset, chunk, persist=True, max_threads=workers,
                    raw_cache=cache, vector_cache=cache)
                done += len(chunk)
                stats['failed'] += len(chunk) - len(got or {})
                self.progress.emit('skeletons_progress', stage='5',
                                   dataset=dataset, done=done,
                                   total=len(missing))
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5] ! skeleton pre-flight stopped after '
                     f'{done}/{len(missing)}: {exc} — scoring against '
                     'whatever the cache holds')
            self.progress.emit('warning', stage='5',
                               note=f'skeleton pre-flight: {exc}')
        finally:
            if timeout_s:
                import socket
                socket.setdefaulttimeout(prev_timeout)
        stats['fetched'] = done - stats['failed']
        stats['unattempted'] = len(missing) - done
        self.progress.emit('skeletons_progress', stage='5', dataset=dataset,
                           done=done, total=len(missing),
                           fetched=stats['fetched'], failed=stats['failed'],
                           note='complete')
        self.log(f'[stage 5] skeleton pre-flight done: fetched '
                 f'{stats["fetched"]}, failed {stats["failed"]} in '
                 f'{time.time() - t0:.0f}s ({workers} threads)')
        return stats

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
        self._record_morph_stores()
        # Rev 3.12 fix (review 2026-09-16): the pool reference pairs must
        # be Track-A scored.  Without them the reference-tier mean (B_b)
        # and the floors-v3 bars are computed over a biased subset (or
        # none), and verified-only branches silently degrade from the
        # track_a_backup bar to the low null bar.
        pool_pairs = [
            (d['best_source_bodyId'], d['target_bodyId'])
            for d in (pool_detail or [])
            if d.get('best_source_bodyId') is not None] or None
        pair_df = self._morph_pair_frame(val_rows, sus_rows, fills,
                                         pool_pairs=pool_pairs,
                                         deep_rows=deep_rows,
                                         null_rows=null_rows)
        if pair_df.empty:
            out['note'] = 'no pairs to score'
            return out
        # Stage 5 must not depend on stage 4 having warmed the target's raw
        # skeleton cache: fetch what THIS pair frame asks for, now.
        self._preflight_target_skeletons(
            cfg.target_dataset, pair_df['target_bodyId'].tolist())
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
        # The grade publishes its own record: a run that scored nothing asked
        # for must say so, or its null bars read as ordinary thin-sample
        # calibration rather than as absent target geometry.
        out['morph_pairs_requested'] = len(scores)
        out['morph_pairs_scored'] = sum(1 for v in scores.values()
                                        if v is not None)
        coverage = morph_coverage_warning(scores, cfg.target_dataset)
        if coverage:
            self.log(f'[stage 5] ! {coverage}')
            out['morph_coverage_warning'] = coverage

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
        thresholds: Dict[Tuple[str, str, str], Optional[float]] = {}
        tiers: Dict[Tuple[str, str, str], Dict] = {}
        by_branch_rows: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
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
                # Branch keys carry the query: two queries can resolve the
                # same concrete (source_type, target_type) — their pools
                # and bars are distinct branches.
                by_branch_rows[(str(d.get('query') or ''),
                                d['source_type'],
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
        def _branch_label(key) -> str:
            # Query-qualified branch label; the prefix is omitted for the
            # empty (single-query / fixture) case to keep legacy keys.
            q, s, t = key
            return f'{s}->{t}' if not q else f'{q}|{s}->{t}'

        out['candidate_thresholds'] = {
            _branch_label(k): v
            for k, v in sorted(thresholds.items())}
        pool_avg: Dict[Tuple[str, str, str], Optional[float]] = {}
        pool_avg_n: Dict[Tuple[str, str, str], int] = {}
        for (q, s, tt), tier in sorted(tiers.items()):
            vals = [d['morph_v2_similarity'] for d in tier['rows']
                    if d.get('morph_v2_similarity') is not None]
            pool_avg[(q, s, tt)] = (float(np.mean(vals)) if vals else None)
            pool_avg_n[(q, s, tt)] = len(vals)
        out['candidate_pool_avg_morph'] = {
            _branch_label(k): v
            for k, v in sorted(pool_avg.items())}

        # ---- Revision 3.7 Track B: all-native pool reference --------
        # Invaders scored against their branch's matched+verified pool
        # natively (dataset-native v2 vector cache + native whitener) —
        # zero cross-dataset transforms, zero transform distortion.
        floors: Dict[Tuple[str, str, str], Optional[float]] = {}
        pool_ref_info: Dict[Tuple[str, str, str], Dict] = {}
        invader_scores: Dict[Tuple[Tuple[str, str, str], int],
                             Tuple[float, float]] = {}
        invaders_by_branch: Dict[Tuple[str, str, str], set] = defaultdict(set)
        for r in list(sus_rows) + list(deep_rows or []):
            invaders_by_branch[(str(r.get('query') or ''),
                                r['source_type'],
                                r['target_type'])].add(
                int(r['ahead_target_bodyId']))
        for f in fills:
            if f.get('side') == 'source' \
                    and f.get('fill_class') == 'out_of_pool':
                invaders_by_branch[(str(f.get('query') or ''),
                                    f['source_type'],
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
            key = (str(r.get('query') or ''),
                   r['source_type'], r['target_type'])
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
            _branch_label(k): info.get('tier')
            for k, info in sorted(pool_ref_info.items())}
        out['pool_ref_baselines'] = {
            _branch_label(k): info.get('baseline')
            for k, info in sorted(pool_ref_info.items())}
        out['pool_ref_floors'] = {
            _branch_label(k): info.get('floor')
            for k, info in sorted(pool_ref_info.items())}

        # ---- Floors v3: per-branch admission bars (plan-unified-
        # morph-qualification-bars).  One BarSet per branch drives EVERY
        # admission decision (candidates, siblings, aggressive examinees)
        # through candidate_qualified / suspicious_qualified; native binds
        # when the m+v reference tier has >=2 scored members, the Track-A
        # backup (B_b − Δ) takes over below that, and the run null is the
        # last resort.  random_null_floor is None here: the pipeline does
        # not score a random sample (the clamp is available to consumers
        # that do, e.g. Find Homolog's per-source nulls).
        branch_bars: Dict[Tuple[str, str, str], BarSet] = {}
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
            _branch_label(k): {
                'candidate_kind': b.candidate_kind,
                'native_floor': b.native_floor,
                'backup_floor': b.backup_floor,
                'suspicious_floor': b.suspicious_floor,
                'suspicious_kind': b.suspicious_kind,
                'pool_track_a_baseline': b.pool_track_a_baseline,
                'n_native_refs': b.n_native_refs,
                'n_scored_pool': b.n_scored_pool,
            } for k, b in sorted(branch_bars.items())}
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
            if 'source_bodyId' in row:
                sid = row['source_bodyId']
                tid = (row.get('target_bodyId')
                       if row.get('target_bodyId') is not None
                       else row.get('proposal_bodyId',
                                    row.get('ahead_target_bodyId')))
            else:
                # a gap-fill proposal names its pair by side, not by column
                sid, tid = fill_pair(row)
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
            key = (str(f.get('query') or ''),
                   f['source_type'], f['target_type'])
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
                add(sid, int(row['ahead_target_bodyId']), 'examinees')
        for row in (deep_rows or []):
            sid = int(row['source_bodyId'])
            if per_src[sid] < cfg.candidate_morph_cap:
                per_src[sid] += 1
                add(sid, int(row['ahead_target_bodyId']), 'deep')
        for row in (null_rows or []):
            add(row['source_bodyId'], row['ahead_target_bodyId'], 'null')
        for row in fills:
            add(*fill_pair(row), 'fill')
        # pool neurons (for the candidate morph threshold: the average
        # verified+matched+borderline pool similarity)
        for sid, tid in (pool_pairs or []):
            add(sid, tid, 'pool')
        return pd.DataFrame(recs, columns=['source_bodyId',
                                           'target_bodyId', 'pair_kind'])

    # -- Revision 3.10: set-level coverage --------------------------------

    def _set_coverage_payload(self, coverage) -> Dict:
        """set_coverage.json payload: the coverage dict plus the
        mapper-gap evidence (additive), so the evidence survives the
        slim README for the report generator's fallback chain."""
        if not coverage:
            return coverage
        payload = dict(coverage)
        counts = getattr(self, '_source_status_counts', None)
        if counts:
            src = dict(payload.get('source') or {})
            src['source_status'] = dict(counts)
            payload['source'] = src
        gap_types = getattr(self, '_mapper_gap_types', None)
        if gap_types:
            payload['mapper_gap'] = {
                'types': dict(gap_types),
                'untyped_rows': getattr(self, '_mapper_gap_untyped', 0)}
        # Stage 5d counters.  This is a NEW input, not a derivation: the
        # coverage block's `family_material` is a pool overhang
        # (parent_target_pool minus the branch pools), so it never sees the
        # enumerated family/relative rows the backward labels belong to.
        bev = getattr(self, '_backward_counters', None)
        if bev:
            tgt = dict(payload.get('target') or {})
            tgt['backward_evidence'] = dict(bev)
            payload['target'] = tgt
        return payload

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
        """Build the MISSING bodyId profiles for a dataset through the
        profiler backend — cache-first per bid, resumable, batch-save +
        consolidate.

        Returns ``{'universe', 'cached', 'built', 'below_k'}``.  Rows that
        exist but hold fewer partners than this run's ``top_k`` are counted,
        NOT rebuilt — 81% (FAFB) / 87% (MCNS) of them sit at k ≤ 5, where the
        profiler's ``k_used = min(max_k, n_rows)`` means the neuron simply has
        fewer partners than k, so a rebuild reproduces the same vector.  It is
        a sparsity note only, never a parity risk: the query path
        (:meth:`get_profile`) and the universe path (``build_target_vectors``)
        were verified to hand back identical vectors, and the scorer is exactly
        symmetric.  Any failure is fail-open: the run continues against
        whatever the cache holds (the historical cache-only behavior)."""
        cfg = self.cfg
        if universe is None:
            universe = self._typed_target_universe(dataset)
        if not universe:
            self.log('[TMVEV] profile pre-flight: no typed universe '
                     f'resolvable for {dataset}; staying cache-only')
            return {'universe': 0, 'cached': 0, 'built': 0, 'below_k': 0}
        cache_df = None
        if cached_ids is None:
            cache_df = self.profiler._load_cache_dataframe(dataset)
            cached_ids = set()
            if cache_df is not None and 'neuron_id' in cache_df.columns:
                cached_ids = {int(x) for x in cache_df['neuron_id']}
        missing = [b for b in universe if b not in set(cached_ids)]
        below_k = self._below_k_cache_ids(cache_df, universe)
        todo = sorted(set(missing))
        stats = {'universe': len(universe), 'cached': len(universe)
                 - len(todo), 'built': 0, 'below_k': len(below_k)}
        self.log(f'[TMVEV] profile pre-flight ({dataset}): universe '
                 f'{len(universe)}, cached {stats["cached"]}, '
                 f'building {len(todo)} ({len(below_k)} cached below k='
                 f'{cfg.top_k}, not rebuilt: degree-limited)')
        self.progress.emit('profiles_progress', stage='2', dataset=dataset,
                           done=0, total=len(todo))
        if not todo:
            self.progress.emit('profiles_progress', stage='2',
                               dataset=dataset, done=0, total=0,
                               below_k=stats['below_k'],
                               note='cache complete')
            return stats
        t0 = time.time()
        batch = {}
        built = 0
        for i, bid in enumerate(todo, 1):
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
            if i % 500 == 0 or i == len(todo):
                self.log(f'[TMVEV] profile pre-flight: {i}/{len(todo)} '
                         f'({100 * i / len(todo):.1f}%) elapsed '
                         f'{time.time() - t0:.0f}s')
                self.progress.emit('profiles_progress', stage='2',
                                   dataset=dataset, done=i,
                                   total=len(todo))
        if batch:
            self.profiler._save_profiles_to_cache_batch(batch, dataset,
                                                        silent=True)
        try:
            self.profiler.consolidate_profile_cache(dataset)
        except Exception as exc:  # noqa: BLE001
            self.log(f'[TMVEV] profile consolidation failed: {exc}')
        stats['built'] = built
        self.progress.emit('profiles_progress', stage='2', dataset=dataset,
                           done=len(todo), total=len(todo),
                           built=built, below_k=stats['below_k'],
                           note='complete')
        self.log(f'[TMVEV] profile pre-flight done: built {built} in '
                 f'{time.time() - t0:.0f}s')
        return stats

    def _below_k_cache_ids(self, cache_df, universe) -> List[int]:
        """Universe bodyIds whose cached profile holds fewer partners than
        the run's ``top_k`` — a SPARSITY DIAGNOSTIC, not a work list.

        This is NOT a parity risk.  ``get_profile`` (the query side) and
        ``build_target_vectors`` (the universe side) were checked against each
        other on real runs and hand back byte-identical vectors: rescoring
        every pair from the cache parquet reproduced the stored forward
        ``rank_union`` on 272/272 pairs (r9 + r11) with zero sub-k rows
        involved, and ``score_one_candidate_fast(a, b) == score(b, a)`` holds
        exactly, so direction cannot change a pair's score.

        Why the rows sit below k: 81% of the FAFB and 87% of the MCNS ones
        hold ``top_k_bodyid_used`` ≤ 5, where ``_process_connections`` sets
        ``k_used = min(max_k, n_rows)`` — the neuron has fewer partners than k.
        Rebuilding cannot raise that (measured: r9 rebuilt 6,883 rows in 903 s
        and r10 rebuilt the same 3,897 again in 512 s, both for no change), so
        the population is counted and reported
        (``source_vectors_below_k``) as a note on how much of the universe
        carries a short vector."""
        if cache_df is None or 'top_k_bodyid_used' not in cache_df.columns \
                or 'neuron_id' not in cache_df.columns:
            return []
        try:
            k = int(self.cfg.top_k)
            got = pd.to_numeric(
                cache_df.set_index('neuron_id')['top_k_bodyid_used'],
                errors='coerce')
            below = {int(b) for b in got.index[got.fillna(k) < k]}
            return sorted(b for b in set(universe) if b in below)
        except Exception:  # noqa: BLE001
            return []

    # -- Plan I follow-up: out-map expansion -----------------------------

    def _expand_out_map_sources(self, out_map_by_type, in_map, target_stats,
                                target_bids, target_id2type, top_k,
                                pool_owner=None):
        """Scan every out-map (unpaired) source neuron against the full
        target universe and keep the top-k connectivity-ranked targets that
        are NOT in-map claims.  Connectivity-only evidence: no morph bars
        apply at this stage (the scene layer and out_map_expansion.csv are
        exploratory surfaces, never fills).

        Source-candidates re-aim (plan-tmvev-samename-first-consumers.md
        §10, user option 2): the SAME scans also record the source's
        best-ranked hits onto IN-MAP branch pools — the D-B8 backward
        mirror of candidate admission (out-of-map source,
        connectivity-qualified onto pool(B), morph-checked against the
        run null bar, attributed to the branch owning the pool).
        ``pool_owner`` maps pool bodyId -> branch key.  Returns
        ``(rows, cand_rows)``; ``cand_rows`` still need their morph
        check before they are candidates."""
        cfg = self.cfg
        rows = []
        cand_rows = []
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
                df = order_by_chain(df)
                kept = 0
                kept_pool = 0
                for r in df.itertuples(index=False):
                    tgt = int(r.target_bid)
                    in_pool_owner = (pool_owner or {}).get(tgt)
                    if tgt in in_map and in_pool_owner is None:
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
                    base = {
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
                    }
                    if in_pool_owner is not None:
                        # D-B8 mirror: the unclaimed source ranks a branch
                        # pool member among its best hits.  Same per-source
                        # rank window as the expansion (rows are sorted).
                        if kept_pool < top_k:
                            kept_pool += 1
                            rec = dict(base)
                            rec['in_map'] = True
                            rec['branch_key'] = in_pool_owner
                            cand_rows.append(rec)
                        continue
                    if kept < top_k:
                        kept += 1
                        rows.append({**base, 'in_map': False})
                    # The non-in-map rows are capped, but in-pool hits
                    # further down the ranking still count — keep
                    # iterating until BOTH caps are closed.
                    if kept >= top_k and kept_pool >= top_k:
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
        # per-source cap (top_k rows), NBLAST off.  The in-pool
        # source-candidate rows are checked against the SAME bar (they
        # share the score map — one enrich pass covers both).
        morph_targets = rows + cand_rows
        if morph_targets and getattr(self, '_track_a_null_bar',
                                     None) is not None:
            try:
                import pandas as pd
                from morphology import enrich_homolog_results
                pair_df = pd.DataFrame(
                    [{'source_bodyId': r['source_bodyId'],
                      'target_bodyId': r['target_bodyId'],
                      'pair_kind': 'out_map'} for r in morph_targets])
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
                for r in morph_targets:
                    key = (int(r['source_bodyId']), int(r['target_bodyId']))
                    r['morph_v2_similarity'] = sc.get(key)
                    r['morph_qualified'] = bool(
                        r['morph_v2_similarity'] is not None
                        and r['morph_v2_similarity'] >= self._track_a_null_bar)
                    # the verdict's own record: `morph_qualified` is a
                    # comparison, and the number it was compared against lived
                    # only in `morphology_calibration.json` until 2026-09-25, so
                    # a reader could not recompute the ✓/✗ from the row. The
                    # kind is the supervised ladder's `null` (morph_bars), not
                    # pooling's `null_bar`: this bar is the run null backdrop.
                    r['morph_bar'] = self._track_a_null_bar
                    r['morph_bar_kind'] = 'null'
                    if not r.get('in_map'):
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
        return rows, cand_rows

    # -- homolog panels: forward capture (stage 2) + target pass (stage 5e)

    @staticmethod
    def _type_label(mapping, bid) -> str:
        """The homolog CSVs' type label for one bodyId: the resolved name,
        or the pipeline's own word ``untyped`` — never the profiler's raw
        ``'nan'``.  The local-table lookup stringifies a missing annotation
        as the literal ``'nan'``, and that string is truthy, so the
        ``or ''`` guards around these reads cannot catch it (measured:
        54/370 ``target_matches.csv`` rows read ``nan`` in the 2026-09-27
        clock-matrix run).  Same verdict as :func:`has_type_name` plus the
        :class:`UntypedLabelPolicy` sentinel vocabulary (case variants,
        hemi suffixes, digit fallbacks)."""
        value = (mapping or {}).get(int(bid))
        if value is None or not has_type_name(value) \
                or UntypedLabelPolicy.is_untyped(value):
            return 'untyped'
        return str(value)

    def _record_forward_matches(self, src_type, plist, scans,
                                target_id2type) -> None:
        """Homolog · forward payload capture (user 2026-09-26): while the
        stage-2 scan frames are in memory, keep per source bodyId the
        chain-best target plus the serialized top-3-rank_union ∪
        top-3-jaccard neighbourhood.  One row per pool source; a source
        the scan skipped (no usable profile) carries
        ``scanned_at='no_profile'`` so silence never reads as a negative.
        Type labels route through :meth:`_type_label` — ``untyped`` for
        an unannotated neuron, never the raw ``'nan'``.  Pure capture —
        the morph display join happens at write time."""
        if not hasattr(self, '_forward_match_rows'):
            self._forward_match_rows = []
            self._forward_seen: set = set()
        in_branch = {int(b) for p in plist for b in p.target_pool}
        for sbid in sorted({int(b) for p in plist for b in p.source_pool}):
            if sbid in self._forward_seen:
                continue
            self._forward_seen.add(sbid)
            row = {
                'source_bodyId': sbid,
                'source_type': src_type,
                'primary_target_bodyId': '',
                'primary_target_type': '',
                'primary_jaccard': None,
                'primary_rank_union': None,
                'primary_in_branch': '',
                'forward_topN': '',
                'n_scanned': 0,
                'scanned_at': 'no_profile',
                'morph_v2_similarity': None,
                'morph_pool_ref': None,
                'morph_bar_kind': '',
            }
            df = scans.get(sbid)
            if df is None or df.empty:
                self._forward_match_rows.append(row)
                continue
            row['n_scanned'] = int(len(df))
            row['scanned_at'] = 'run'
            row['forward_topN'] = serialize_topn_union(
                df, target_id2type, k=3, branch_pool=in_branch)
            best = _best_row(df)
            if best is not None:
                bid = int(best['target_bid'])
                row['primary_target_bodyId'] = bid
                row['primary_target_type'] = self._type_label(
                    target_id2type, bid)
                for f, col in (('primary_jaccard', 'jaccard'),
                               ('primary_rank_union', 'rank_union')):
                    v = best.get(col)
                    row[f] = None if pd.isna(v) else float(v)
                row['primary_in_branch'] = bool(bid in in_branch)
            self._forward_match_rows.append(row)

    def _appeared_target_bids(self, val_rows, sus_rows, deep_rows,
                              noise_rows, fills, pool_detail, out_map_rows,
                              dedup_rows, relatives, family_rows) -> List[int]:
        """Every target bodyId that APPEARED in the run — the stage-5e
        universe.  Sources never qualify (a fill row spells its pair
        through :func:`fill_pair`, so only the target side is taken)."""
        bids: Dict[int, None] = {}
        for rows, keys in (
                (val_rows, ('target_bodyId', 'ru_top_target_bodyId')),
                (sus_rows + deep_rows + noise_rows,
                 ('ahead_target_bodyId', 'best_pool_target_bodyId')),
                (pool_detail, ('target_bodyId',)),
                (out_map_rows, ('target_bodyId',)),
                (dedup_rows, ('target_bodyId', 'proposal_bodyId')),
                (relatives, ('ahead_target_bodyId',
                             'best_pool_target_bodyId')),
                (family_rows, ('ahead_target_bodyId',
                               'best_pool_target_bodyId')),
                (getattr(self, '_backward_matches_rows', None) or [],
                 ('member_bodyId',)),
                # pooling-engine targets: the supervised artifacts above
                # never name them, but a pooling run's pool is real
                # appeared-target material for the backward panel
                ((getattr(self, '_pooling', None) or {}).get('pool') or [],
                 ('target_bodyId',)),
        ):
            for r in rows or []:
                for k in keys:
                    try:
                        bids[int(r.get(k))] = None
                    except (TypeError, ValueError):
                        continue
        for r in fills or []:
            try:
                bids[fill_pair(r)[1]] = None
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(bids)

    def _target_match_pass(self, universe: List[int]) -> None:
        """Stage 5e (user 2026-09-26): scan every appeared target bodyId
        back against the WHOLE source dataset — the exact mirror of the
        stage-2 forward scan, feeding the report's Homolog · backward
        panel.  One row per target bodyId, no caps (the stage-5d caps
        belong to the evidence pass).  Type labels route through
        :meth:`_type_label` — ``untyped`` for an unannotated neuron.
        Advisory display data only: nothing downstream gates on it.
        Fail-open like the other advisory layers."""
        cfg = self.cfg
        if not universe:
            self._target_match_rows = []
            return
        self.log(f'[stage 5e] target matches: scanning {len(universe)} '
                 'appeared target bodyId(s) against the whole source '
                 'dataset')
        self.progress.emit('stage_start', stage='5e',
                           label='target matches', targets=len(universe))
        if not cfg.skip_profile_build:
            try:
                self._preflight_target_profiles(cfg.source_dataset)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[stage 5e] source profile pre-flight failed '
                         f'(continuing cache-only): {exc}')
        # type labels first: the lookup only needs the profiler, so even
        # the vector-build failure path below exports real labels
        tgt_id2type = self._bodyid_types(universe, cfg.target_dataset)
        try:
            vectors = build_target_vectors(
                self.profiler, cfg.source_dataset, self.mapper, cfg.verbose,
                min_weight=cfg.target_min_weight,
                min_partner_types=cfg.target_min_partner_types)
            source_stats = prep_target_stats(vectors)
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5e] source-universe vectors unavailable, '
                     f'target matches skipped: {exc}')
            self._target_match_rows = [{
                'target_bodyId': bid,
                'target_type': self._type_label(tgt_id2type, bid),
                'primary_source_bodyId': '', 'primary_source_type': '',
                'primary_jaccard': None, 'primary_rank_union': None,
                'primary_in_branch': '', 'backward_topN_union': '',
                'n_scanned': 0, 'scanned_at': 'error',
            } for bid in universe]
            self.progress.emit('stage_done', stage='5e',
                               label='target matches', rows=0)
            return
        del vectors
        source_bids = list(source_stats)
        in_branch = {int(b) for p in self.pairs for b in p.source_pool}
        reduced: Dict[int, object] = {}
        rows: List[Dict] = []
        t0 = time.time()
        for i, bid in enumerate(universe, 1):
            row = {
                'target_bodyId': bid,
                'target_type': self._type_label(tgt_id2type, bid),
                'primary_source_bodyId': '',
                'primary_source_type': '',
                'primary_jaccard': None,
                'primary_rank_union': None,
                'primary_in_branch': '',
                'backward_topN_union': '',
                'n_scanned': 0,
                'scanned_at': 'no_profile',
            }
            try:
                sp = self.profiler.get_profile(bid, cfg.target_dataset)
                df = None
                if sp is not None and sp.connectivity_status.name \
                        not in SOURCE_STATUS_SKIP:
                    df = scan_source(expanded_vector(sp, self.mapper),
                                     source_stats, source_bids)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[TMVEV] target scan {bid}: {exc}')
                row['scanned_at'] = 'error'
                df = None
            if df is not None and not df.empty:
                row['n_scanned'] = int(len(df))
                row['scanned_at'] = 'run'
                # keep only the union window until the types are
                # known — a full frame per target does not stay resident
                small = topn_union_rows(df, k=3)
                if small is not None and not small.empty:
                    reduced[bid] = small
                    best = small.iloc[0]
                    s_bid = int(best['target_bid'])
                    row['primary_source_bodyId'] = s_bid
                    row['primary_jaccard'] = (
                        None if pd.isna(best['jaccard'])
                        else float(best['jaccard']))
                    row['primary_rank_union'] = (
                        None if pd.isna(best['rank_union'])
                        else float(best['rank_union']))
                    row['primary_in_branch'] = bool(s_bid in in_branch)
            rows.append(row)
            if i % 50 == 0 or i == len(universe):
                self.log(f'[stage 5e] target scan {i}/{len(universe)} '
                         f'({time.time() - t0:.0f}s)')
                self.progress.emit('backward_progress', stage='5e',
                                   done=i, total=len(universe))
        del source_stats, source_bids
        wanted = sorted({int(r.target_bid) for small in reduced.values()
                         for r in small.itertuples(index=False)})
        src_id2type = self._bodyid_types(wanted, cfg.source_dataset)
        for row in rows:
            small = reduced.get(int(row['target_bodyId']))
            if small is None:
                continue
            row['backward_topN_union'] = serialize_topn_union(
                small, src_id2type, k=3, branch_pool=in_branch)
            if not row['primary_source_bodyId']:
                continue
            row['primary_source_type'] = self._type_label(
                src_id2type, int(row['primary_source_bodyId']))
        self._target_match_rows = rows
        self.log(f'[stage 5e] target matches: {len(reduced)} scanned / '
                 f'{len(universe)} universe '
                 f'({time.time() - t0:.0f}s)')
        self.progress.emit('stage_done', stage='5e',
                           label='target matches',
                           rows=len(reduced), total=len(universe))

    @staticmethod
    def _morph_pair_index(*row_groups) -> Dict[Tuple[int, int], Dict]:
        """(source_bodyId, target_bodyId) -> the morph values the run
        already exported for that pair, first non-empty value wins.  Fill
        rows spell their pair through :func:`fill_pair` and name their bar
        kind ``bar_kind``; both spellings normalize here so the homolog
        panels DISPLAY these joins and nothing is re-scored."""
        idx: Dict[Tuple[int, int], Dict] = {}

        def _absorb(rows, a_key, b_key, fields):
            for r in rows or []:
                try:
                    key = (int(r.get(a_key)), int(r.get(b_key)))
                except (TypeError, ValueError):
                    continue
                rec = idx.setdefault(key, {})
                for f in fields:
                    v = r.get(f)
                    if v is not None and v != '':
                        rec.setdefault(f, v)
                bk = rec.get('bar_kind')
                if bk is not None and 'morph_bar_kind' not in rec:
                    rec['morph_bar_kind'] = bk

        def _absorb_fill(rows):
            for r in rows or []:
                try:
                    key = fill_pair(r)
                except (KeyError, TypeError, ValueError):
                    continue
                rec = idx.setdefault(key, {})
                for f in ('morph_v2_similarity', 'morph_pool_ref',
                          'bar_kind', 'bar_value', 'fill_qualified'):
                    v = r.get(f)
                    if v is not None and v != '':
                        rec.setdefault(f, v)
                bk = rec.get('bar_kind')
                if bk is not None and 'morph_bar_kind' not in rec:
                    rec['morph_bar_kind'] = bk

        for rows in row_groups:
            if not rows:
                continue
            _absorb(rows, 'source_bodyId', 'target_bodyId',
                    ('morph_v2_similarity', 'morph_bar', 'morph_bar_kind',
                     'bar_kind'))
            _absorb(rows, 'source_bodyId', 'ahead_target_bodyId',
                    ('morph_v2_similarity', 'morph_pool_ref', 'bar_kind'))
            _absorb(rows, 'source_bodyId', 'best_pool_target_bodyId',
                    ('morph_v2_similarity', 'morph_pool_ref', 'bar_kind'))
            _absorb_fill(rows)
        return idx

    def _finalize_forward_match_rows(self, morph_idx) -> List[Dict]:
        """Attach the display joins to the stage-2 capture: the primary
        pair's morph values, read off the artifacts that already scored
        it."""
        rows = []
        for row in (getattr(self, '_forward_match_rows', None) or []):
            row = dict(row)
            rec = None
            try:
                rec = morph_idx.get((int(row['source_bodyId']),
                                     int(row['primary_target_bodyId'])))
            except (TypeError, ValueError):
                rec = None
            for f in ('morph_v2_similarity', 'morph_pool_ref',
                      'morph_bar_kind'):
                row[f] = (rec or {}).get(f)
            rows.append(row)
        return rows

    def _finalize_target_match_rows(self, pool_detail,
                                    morph_idx) -> List[Dict]:
        """Attach the pool category + claiming branches (every branch that
        claims the target) and the primary pair's morph values to the
        stage-5e rows."""
        cat_by_bid: Dict[int, List[str]] = {}
        branch_by_bid: Dict[int, List[str]] = {}
        for r in pool_detail or []:
            try:
                bid = int(r.get('target_bodyId'))
            except (TypeError, ValueError):
                continue
            c = str(r.get('category') or '')
            if c:
                cats = cat_by_bid.setdefault(bid, [])
                if c not in cats:
                    cats.append(c)
            b = (f"{r.get('source_type') or '?'}→"
                 f"{r.get('target_type') or '?'}")
            branches = branch_by_bid.setdefault(bid, [])
            if b not in branches:
                branches.append(b)
        rows = []
        for row in (getattr(self, '_target_match_rows', None) or []):
            row = dict(row)
            cats = cat_by_bid.get(int(row['target_bodyId'])) or []
            row['pool_category'] = ';'.join(cats)
            row['pool_branches'] = ';'.join(
                branch_by_bid.get(int(row['target_bodyId'])) or [])
            rec = None
            try:
                rec = morph_idx.get((int(row['primary_source_bodyId']),
                                     int(row['target_bodyId'])))
            except (TypeError, ValueError):
                rec = None
            for f in ('morph_v2_similarity', 'morph_pool_ref',
                      'morph_bar_kind'):
                row[f] = (rec or {}).get(f)
            rows.append(row)
        return rows

    # -- stage 5d: backward (target -> source) homolog evidence ------------

    def _backward_expansion_pass(self):
        """Reverse-scan the expansion bins so each labeled neuron answers
        "which SOURCE neuron do you prefer, and is it ours?".

        Runs after :meth:`finalize_categories` (that is where the family and
        relative members are first enumerated) and before the coverage /
        layered-fill rollups, then rebuilds the bodyId-unique dedup so the
        labels reach ``gap_fill_levels.csv``.

        Additive and ADVISORY: it writes ``backward_*`` columns and never
        touches ``category`` or any ``counts_toward_*`` flag, so the fill
        totals are identical to a run without it.  Connectivity only — no
        morphology is scored (plan D5).  It is not mode-gated: in restrictive
        mode the family / relative bins are simply empty, so the pass checks
        the ``candidates`` rows.  Fail-open like the other
        advisory layers: a failure leaves the rows ``unscanned`` (``error`` /
        ``no_profile`` in ``backward_scanned_at``) and never raises into
        :meth:`run`.

        The scanned set is the gap-fill bins (candidates / family /
        relative) plus the UNMATCHED pool members.  Matched / verified /
        borderline pool members are already mapped — the symmetric
        forward score is their evidence — so a reverse scan would only
        restate it (user 2026-09-19).
        """
        cfg = self.cfg
        if not cfg.backward_evidence_enabled or cfg.skip_backward_pass:
            return None
        evidence = getattr(self, '_cat_evidence', None)
        pool_detail = getattr(self, '_cat_pool_detail', None)
        family_rows = getattr(self, '_cat_family_rows', None) or []
        relative_rows = getattr(self, '_cat_relative_rows', None) or []
        per_pair_res = getattr(self, '_cat_per_pair_res', None) or {}
        if evidence is None or pool_detail is None:
            self.log('[stage 5d] backward evidence skipped: the category '
                     'partition stashed no inputs (finalize_categories '
                     'failed?)')
            return None

        pairs_by_key = {}
        for p in self.pairs:
            pairs_by_key[(p.query, p.source_type, p.target_type)] = p
            pairs_by_key.setdefault((p.source_type, p.target_type), p)

        def branch_key(row):
            k3 = (str(row.get('query') or ''), str(row['source_type']),
                  str(row['target_type']))
            return k3 if k3 in pairs_by_key else k3[1:]

        def branch_source_pool(key):
            p = pairs_by_key.get(tuple(key))
            return {int(b) for b in (p.source_pool if p else [])}

        target_branch = {}
        for key, res in per_pair_res.items():
            for t in (res.get('_pool_set') or []):
                target_branch[int(t)] = key

        members = {}

        def add(bid, key, role, rows, mtype, mcat):
            rec = members.setdefault(int(bid), {
                'roles': set(), 'rows_by_key': {},
                'member_type': '' if mtype is None else str(mtype),
                'member_category': '' if mcat is None else str(mcat)})
            rec['roles'].add(role)
            if key is not None:
                rec['rows_by_key'].setdefault(tuple(key), []).extend(
                    rows or [])

        # The bins the user asked to reverse-check: the gap-fill ladder
        # (candidates / family / relative) plus the unmatched pool
        # members below.  `sibling` / `examinees` / out-of-scope rows are
        # skipped — already mapped or noise — and so are matched /
        # verified / borderline pool members: they are already mapped,
        # and the symmetric forward score is their evidence (user
        # 2026-09-19).
        for row, bid, tname in evidence:
            cat = str(row.get('category') or '')
            if cat not in GAP_FILL_LEVEL_BY_CATEGORY:
                continue
            add(bid, branch_key(row), cat, [row], tname, cat)
        for r in list(family_rows) + list(relative_rows):
            add(r['ahead_target_bodyId'],
                (str(r.get('query') or ''), str(r['source_type']),
                 str(r['target_type'])),
                str(r['category']), [r], r.get('ahead_target_type'),
                r.get('category'))
        if cfg.backward_scan_pool_targets:
            for d in pool_detail:
                if str(d.get('category') or '') != 'unmatched':
                    continue
                add(int(d['target_bodyId']),
                    target_branch.get(int(d['target_bodyId'])),
                    'pool_target', [d],
                    d.get('target_type'), d.get('category'))

        # Budget: one reverse scan costs what a forward source scan costs
        # (~4-7 s against a 140-180k universe), so a bodyId is scanned ONCE
        # for all branches and the caps are enforced on distinct bodyIds.
        # The gap-fill bins go first — they are what this pass exists to
        # label; the unmatched pool members follow (measured on a real
        # run, an unordered budget spent 102 of 120 scans on the control
        # and capped out half the bin members).
        bids = sorted(members, key=lambda b: (
            0 if members[b]['roles'] - {'pool_target'} else 1, b))
        order, per_branch = [], Counter()
        budget = int(cfg.backward_max_neurons or 0) or len(bids)
        per_branch_cap = int(cfg.backward_per_branch_cap or 0) or len(bids)
        for bid in bids:
            if len(order) >= budget:
                break
            keys = list(members[bid]['rows_by_key']) or [None]
            if not any(per_branch[k] < per_branch_cap for k in keys):
                continue
            order.append(bid)
            for k in keys:
                per_branch[k] += 1
        order_set = set(order)
        for bid, rec in members.items():
            if bid in order_set:
                continue
            for rows in rec['rows_by_key'].values():
                for r in rows:
                    r.update(blank_backward_fields('cap'))
        self.log(f'[stage 5d] backward homolog evidence: {len(order)} '
                 f'distinct member(s) to reverse-scan, '
                 f'{len(bids) - len(order)} beyond the caps')
        if not order:
            return self._rebuild_dedup_rows()

        self.progress.emit('backward_progress', stage='backward',
                           done=0, total=len(order))
        refresh_stats = None
        if not cfg.skip_profile_build:
            # D9 of the plan: make sure every source neuron HAS a profile
            # before the universe can act as a scan target, and measure how
            # many of them are vector-truncated (counted, not rebuilt).
            try:
                refresh_stats = self._preflight_target_profiles(
                    cfg.source_dataset)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[stage 5d] source profile pre-flight failed '
                         f'(continuing cache-only): {exc}')
        try:
            vectors = build_target_vectors(
                self.profiler, cfg.source_dataset, self.mapper, cfg.verbose,
                min_weight=cfg.target_min_weight,
                min_partner_types=cfg.target_min_partner_types)
            source_stats = prep_target_stats(vectors)
            self._fingerprint()['scanned_source_universe'] = len(source_stats)
        except Exception as exc:  # noqa: BLE001
            self.log(f'[stage 5d] source-universe vectors unavailable, '
                     f'backward evidence skipped: {exc}')
            # the pass WAS asked to run, so the in-budget rows must say
            # `error`, not inherit the disabled-run default `disabled`
            for bid in order:
                for rows in members[bid]['rows_by_key'].values():
                    for r in rows:
                        r.update(blank_backward_fields('error'))
            return self._rebuild_dedup_rows()
        del vectors
        source_bids = list(source_stats)

        scanned = {}
        # why a member was never scanned — 'no_profile' (absent or a status
        # the scan skips) is silence; 'error' means the scan itself failed.
        # Both are `unscanned`; a scan that ranked nothing is `none`, which
        # is a negative result and must not be conflated with them.
        unscanned = {}
        t0 = time.time()
        for i, bid in enumerate(order, 1):
            try:
                sp = self.profiler.get_profile(bid, cfg.target_dataset)
                if sp is None or sp.connectivity_status.name \
                        in SOURCE_STATUS_SKIP:
                    unscanned[bid] = 'no_profile'
                    continue
                df = scan_source(expanded_vector(sp, self.mapper),
                                 source_stats, source_bids)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[TMVEV] backward scan {bid}: {exc}')
                unscanned[bid] = 'error'
                continue
            scanned[bid] = df
            if i % 25 == 0 or i == len(order):
                self.log(f'[stage 5d] backward scan {i}/{len(order)} '
                         f'({time.time() - t0:.0f}s)')
                self.progress.emit('backward_progress', stage='backward',
                                   done=i, total=len(order))
        del source_stats, source_bids

        # Source type names, resolved once for the recorded hits only — the
        # chain's top-N (what `serialize_backward_topN` publishes) PLUS the
        # top-3 of each rank column (what the branch-type grade reads), both
        # needed to name a hit.
        wanted = set()
        for df in scanned.values():
            if df is None or df.empty:
                continue
            head_n = max(cfg.backward_top_n, 3)
            for r in order_by_chain(df).head(head_n).itertuples(index=False):
                wanted.add(int(r.target_bid))
            for col in ('jaccard_rank', 'rank_union_rank'):
                for r in df.sort_values(col, na_position='last').head(
                        head_n).itertuples(index=False):
                    wanted.add(int(r.target_bid))
        src_id2type = self._bodyid_types(sorted(wanted), cfg.source_dataset)

        match_rows = []
        best_label = {}
        label_rank = {'low': 1, 'medium': 2, 'high': 3}
        reverse_columns = {}
        for bid in order:
            rec = members[bid]
            df = scanned.get(bid)
            for key, rows in rec['rows_by_key'].items():
                pool = branch_source_pool(key)
                verdict = classify_backward_scan(
                    df, branch_pool=pool, id2type=src_id2type,
                    sizes=self._source_sizes,
                    pool_best_size=max(
                        ((self._source_sizes or {}).get(b, 0.0)
                         for b in pool), default=0.0),
                    branch_source_type=key[-2],
                    min_size_ratio=cfg.target_min_size_ratio,
                    top_n=cfg.backward_top_n)
                if bid in unscanned:
                    # never scanned is not the same claim as a graded low
                    verdict['backward_scanned_at'] = unscanned[bid]
                    verdict['backward_evidence'] = 'not-checked'
                for r in rows:
                    r.update(verdict)
                lab = str(verdict.get('backward_evidence') or '')
                if lab not in label_rank:
                    continue
                cur = best_label.get(bid)
                if cur is None or label_rank[lab] > label_rank.get(cur, 0):
                    best_label[bid] = lab
                match_rows.append({
                    'query': key[0] if len(key) == 3 else '',
                    'branch_source_type': key[-2],
                    'branch_target_type': key[-1],
                    'member_bodyId': bid,
                    'member_type': rec['member_type'],
                    'member_category': rec['member_category'],
                    'scan_role': ','.join(sorted(rec['roles'])),
                    **verdict})
            if 'pool_target' in rec['roles'] and df is not None \
                    and not df.empty:
                key = next(iter(rec['rows_by_key']), None)
                col = reverse_source_column(
                    df, branch_source_pool(key) if key else set(),
                    top_rows=max(cfg.verified_top_n
                                 + cfg.invader_borderline_max + 2, 10))
                if col:
                    reverse_columns[bid] = col

        counts = Counter(best_label.values())
        self._backward_matches_rows = match_rows
        # the scene layer reads this to suffix a leaf with its reciprocal
        # verdict (mapping_validation_visualize._BACKWARD_LEAF_TAG)
        self._backward_label_by_bid = dict(best_label)
        self._backward_counters = {
            k: int(counts.get(k, 0)) for k in BACKWARD_EVIDENCE_VALUES}
        self._backward_counters['distinct_scanned'] = len(scanned)
        self._backward_counters['beyond_cap'] = len(bids) - len(order)
        if refresh_stats:
            # D9: how many source rows carry a vector built below this run's
            # k — counted, not rebuilt (see _below_k_cache_ids)
            self._backward_counters['source_vectors_below_k'] = int(
                refresh_stats.get('below_k') or 0)
        self._backward_counters['morph'] = 'not evaluated (connectivity-only)'
        self.log('[stage 5d] backward evidence: '
                 + ', '.join(f'{k}={v}' for k, v in
                             sorted(self._backward_counters.items()))
                 + f' in {time.time() - t0:.0f}s')
        self.progress.emit('backward_progress', stage='backward',
                           done=len(order), total=len(order),
                           note=f'{len(scanned)} scanned')
        if reverse_columns:
            self._reverse_columns = reverse_columns
            try:
                self._backward_source_pass(per_pair_res, [],
                                           reverse=reverse_columns)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[stage 5d] reverse-column refresh skipped: {exc}')
        return self._rebuild_dedup_rows()

    def _rebuild_dedup_rows(self) -> List[Dict]:
        """Re-run the bodyId-unique rollup after a label change (it is a pure
        function of the stashed partition inputs)."""
        return self._build_dedup_rows(
            getattr(self, '_cat_evidence', None) or [],
            getattr(self, '_cat_pool_detail', None) or [],
            getattr(self, '_cat_family_rows', None) or [],
            getattr(self, '_cat_relative_rows', None) or [])

    def _bodyid_types(self, bids: List[int],
                      dataset: str) -> Dict[int, str]:
        """Type label per bodyId of ANY dataset (``_target_types`` is the
        target-side convenience wrapper over this)."""
        out: Dict[int, str] = {}
        if not bids:
            return out
        try:
            got = self.profiler.get_types_for_bodyids(
                list(bids), dataset) or {}
            for k, v in got.items():
                out[int(k)] = v
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! type labels unavailable for {dataset}: {exc}')
        return out

    def _backward_source_pass(self, per_pair_res: Dict,
                              all_sus_rows: List[Dict],
                              reverse: Optional[Dict[int, List[Dict]]] = None
                              ) -> None:
        """Populate the advisory backward view (plan-
        backward-source-status.md): `source-` statuses per in-branch
        source (column view of the same pair scores).  The
        source-candidates regroup (foreign sources whose qualified
        sibling rows point into a branch pool) is populated separately,
        post-finalize, by ``_collect_source_candidates``."""
        all_source_status: List[Dict] = []
        target_branch: Dict[int, Tuple] = {}
        for key, res in per_pair_res.items():
            for t in (res.get('_pool_set') or []):
                target_branch[int(t)] = key
        seen_status: set = set()
        for key, res in per_pair_res.items():
            pool_set = {int(b) for b in (res.get('_pool_set') or [])}
            per_src = res.get('per_source')
            if not pool_set or per_src is None:
                continue
            pd0 = (res.get('pool_detail') or [{}])[0]
            q = str(pd0.get('query') or (key[0] if len(key) == 3 else ''))
            src_type = str(pd0.get('source_type')
                           or (key[1] if len(key) == 3 else key[0]))
            tgt_type = str(pd0.get('target_type') or key[-1])
            basis = str(pd0.get('pool_basis') or 'linker rows')
            statuses, detail = categorize_pool_sources(
                per_src, pool_set, set(per_src.keys()),
                top_n=self.cfg.verified_top_n,
                invader_max=self.cfg.invader_borderline_max,
                matched_ru_min=self.cfg.matched_ru_min,
                reverse_columns=(
                    {int(t): e for t, e in reverse.items()
                     if int(t) in pool_set} if reverse else None))
            for d in detail:
                sbid = int(d['source_bodyId'])
                if sbid in seen_status:
                    continue
                seen_status.add(sbid)
                all_source_status.append({
                    'query': q, 'source_bodyId': sbid,
                    'source_type': src_type,
                    'branch_target_type': tgt_type,
                    'pool_basis': basis, 'status': d['status'],
                    'col_rank': d['col_rank'],
                    'best_column_target': d['best_column_target'],
                    'best_pair_ru': d['best_pair_ru'],
                    'n_competitors': d['n_competitors'],
                })
        self._source_status_rows = all_source_status
        if all_source_status:
            counts = Counter(
                r['status'] for r in all_source_status)
            self._source_status_counts = dict(counts)
            self.log('[TMVEV] backward source status: '
                     + ', '.join(f'{k} {v}' for k, v in
                                 counts.most_common()))

    def _collect_source_candidates(self, cand_rows: List[Dict]) -> None:
        """Build the branch-attributed source-candidates from the
        out-map expansion's in-pool rows (plan
        `plan-tmvev-samename-first-consumers.md` §10 — user option 2).

        RE-AIMED semantics (replacing the sibling-row route, whose
        candidates were by construction other branches' query neurons):
        a source-candidate of branch B is an OUT-OF-MAP source (claimed
        by no branch) whose best-ranked scan hits land in pool(B) and
        pass the run null bar — the D-B8 backward mirror of candidate
        admission.  Sets `_source_candidates` + `_source_candidates_multi`
        for the scenes and `source_candidates.csv`."""
        by_branch: Dict[Tuple, List[Dict]] = {}
        for r in cand_rows or []:
            if not r.get('morph_qualified'):
                continue
            bkey = r.get('branch_key')
            if bkey is None:
                continue
            by_branch.setdefault(bkey, []).append({
                'source_bodyId': int(r['source_bodyId']),
                'source_type': r.get('source_type'),
                'target_bodyId': int(r['target_bodyId']),
                'target_type': r.get('target_type'),
                'rank_union': r.get('rank_union'),
                'jaccard': r.get('jaccard'),
                'morph_v2_similarity': r.get('morph_v2_similarity'),
                # every row here is qualified by construction (the filter
                # above), so the bar is the whole content of its ✓
                'morph_bar': r.get('morph_bar'),
                'morph_bar_kind': r.get('morph_bar_kind'),
                'morph_qualified': True,
            })
        cand_count: Counter = Counter()
        for rows in by_branch.values():
            for c in rows:
                cand_count[c['source_bodyId']] += 1
        self._source_candidates_multi = {
            bid for bid, n in cand_count.items() if n > 1}
        self._source_candidates = by_branch
        if by_branch:
            self.log('[TMVEV] source-candidates: '
                     + f"{sum(len(v) for v in by_branch.values())}"
                     + ' morph-qualified row(s) across '
                     + f"{len(by_branch)} target branch(es) from "
                     + f'{len(cand_count)} distinct out-of-map source(s)')

    # -- P3: advisory suspects verification (OPT-IN) ----------------------

    def _suspects_decision(self, src_type: str) -> Optional[Dict]:
        """The mapper's same-name-first decision for one source type
        (None when the mapper lacks the accessor or the type is not a
        same-name fan-out)."""
        fn = getattr(self.mapper, 'same_name_first_fires', None)
        if fn is None:
            return None
        try:
            return fn(src_type, self.cfg.source_dataset,
                      self.cfg.target_dataset)
        except Exception:  # noqa: BLE001
            return None

    def _verify_disclosure_targets(self, src_type: str, pool: List[int],
                                   scans: Dict[int, pd.DataFrame],
                                   decision: Optional[Dict],
                                   target_id2type=None, sizes=None,
                                   weights=None, source_sides=None,
                                   target_sides=None) -> None:
        """Verify the DISCLOSURE ends with the ordinary machinery.

        §three-tier (user 2026-09-27): the ends the mapper decision
        declined but the evidence reaches (decision['disclosure_targets'])
        get the SAME per-pair validation as claim pairs, routed into the
        SEPARATE disclosure accumulator — the verdicts annotate
        disclosure_evidence.csv and never touch the headline counts
        (panel parity preserved).  Advisory, same class as the suspects
        pass.
        """
        if not decision or not decision.get('disclosure_targets'):
            return
        if not hasattr(self, '_disclosure_verification'):
            self._disclosure_verification = {}
        cfg = self.cfg
        for disc in decision['disclosure_targets']:
            tgt_type = str(disc.get('target') or '')
            if not tgt_type:
                continue
            vkey = (str(self._current_query or src_type), str(src_type),
                    tgt_type)
            if vkey in self._disclosure_verification:
                continue
            tgt_pool = self._bodyids_for(tgt_type, cfg.target_dataset)
            if not tgt_pool:
                self.log(f'  [disclosure] {src_type}: {tgt_type!r} has no '
                         'target pool — recorded unverified')
                self._disclosure_verification[vkey] = {
                    'verdict': 'no_target_pool', 'rank_union': ''}
                continue
            dpair = TypePair(
                source_dataset=cfg.source_dataset,
                source_type=src_type,
                source_pool=sorted(int(b) for b in pool),
                target_dataset=cfg.target_dataset,
                target_type=tgt_type,
                target_pool=tgt_pool,
                relationship='disclosure',
                status='disclosure',
                tier='disclosure',
                query=str(self._current_query or src_type))
            try:
                res = self.validate_pair(
                    dpair, scans, target_id2type, sizes=sizes,
                    weights=weights, source_sides=source_sides,
                    target_sides=target_sides)
                rows = res['rows'] or []
                from collections import Counter as _C
                verdicts = _C(str(r.get('verdict') or '') for r in rows)
                best_ru = max(
                    (r.get('rank_union') for r in rows
                     if isinstance(r.get('rank_union'), (int, float))),
                    default='')
                self._disclosure_verification[vkey] = {
                    'verdict': '; '.join(
                        f'{v}×{n}' for v, n in sorted(verdicts.items()))
                    or 'no_rows',
                    'rank_union': best_ru,
                }
                self.log(f'  [disclosure] {src_type} -> {tgt_type} '
                         f"({disc.get('reason')}): "
                         f"{self._disclosure_verification[vkey]['verdict']}")
            except Exception as exc:  # noqa: BLE001 — advisory only
                self._disclosure_verification[vkey] = {
                    'verdict': f'verification_failed: {exc}', 'rank_union': ''}

    def _verify_suspects_for_type(self, src_type: str, pool: List[int],
                                  scans: Dict[int, pd.DataFrame],
                                  decision: Optional[Dict],
                                  target_id2type=None, sizes=None,
                                  weights=None, source_sides=None,
                                  target_sides=None) -> None:
        """Verify ONE source type's rival candidates against their own
        target pools with the ordinary validation machinery, routing the
        rows into the SEPARATE suspects accumulators.  Never touches the
        main validation outputs (advisory, P3)."""
        if not decision or not (decision.get('rivals') or []):
            return
        if not hasattr(self, '_suspects_verification_rows'):
            self._suspects_verification_rows = []
            self._suspects_verified_types = set()
        if src_type in self._suspects_verified_types:
            return
        self._suspects_verified_types.add(src_type)
        cfg = self.cfg
        # mapper-side per-rival evidence (clean own pair, votes, ...)
        rival_evidence = {}
        gd = getattr(self.mapper, 'get_same_name_conflict_detail', None)
        if gd is not None:
            try:
                detail = gd(src_type, cfg.source_dataset,
                            cfg.target_dataset)
                for r in (detail or {}).get('rival_evidence') or []:
                    rival_evidence[str(r.get('rival'))] = r
            except Exception:  # noqa: BLE001
                rival_evidence = {}
        status = str(decision.get('disposition') or '')
        for rival in decision.get('rivals') or []:
            tgt_pool = self._bodyids_for(rival, cfg.target_dataset)
            if not tgt_pool:
                self.log(f'  [suspects] {src_type}: rival {rival!r} has '
                         'no target pool — skipped')
                continue
            spair = TypePair(
                source_dataset=cfg.source_dataset,
                source_type=src_type,
                source_pool=sorted(int(b) for b in pool),
                target_dataset=cfg.target_dataset,
                target_type=rival,
                target_pool=tgt_pool,
                relationship='suspects',
                status=status,
                query=str(getattr(self, '_current_query', '')
                          or src_type))
            res = self.validate_pair(
                spair, scans, target_id2type, sizes=sizes,
                weights=weights, source_sides=source_sides,
                target_sides=target_sides)
            ev = rival_evidence.get(str(rival)) or {}
            for row in res['rows']:
                row['rival_of'] = decision.get('selected') or ''
                row['disposition'] = decision.get('disposition') or ''
                row['rival_has_own_clean_pair'] = bool(
                    ev.get('rival_has_own_clean_pair'))
                row['rival_reverse_target'] = ev.get('reverse_target') or ''
                row['rival_votes'] = ev.get('votes')
                # mapper-side population context (interpretation aid:
                # how many neurons carry the rival name on each side)
                row['rival_population_source'] = ev.get('population_source')
                row['rival_population_target'] = ev.get('population_target')
                self._suspects_verification_rows.append(row)
            self.log(f'  [suspects] {src_type}: rival {rival} — '
                     f'verdicts: '
                     f'{_counter_text("verdict", res["rows"])}')

    def _verify_suspects_for_excluded(self, target_id2type=None,
                                      sizes=None, weights=None,
                                      source_sides=None, target_sides=None,
                                      target_stats=None,
                                      target_bids=None) -> None:
        """P3 second hook: held / evidence-only types have no validated
        pairs, so their source pools were never scanned.  Build their
        scans and verify their rivals too (D-A: all queried fan-outs)."""
        cfg = self.cfg
        records = [r for r in (getattr(self, '_same_name_excluded', []) or [])
                   if r.get('disposition') in ('gated_held',
                                               'excluded_evidence_only')]
        for rec in records:
            src_type = str(rec.get('source_type') or '')
            if not src_type:
                continue
            decision = self._suspects_decision(src_type)
            if not decision:
                continue
            pool = self._bodyids_for(src_type, cfg.source_dataset)
            if not pool:
                continue
            self._current_query = str(rec.get('query') or src_type)
            scans: Dict[int, pd.DataFrame] = {}
            for sbid in pool:
                sp = self.profiler.get_profile(sbid, cfg.source_dataset)
                if sp is None or \
                        sp.connectivity_status.name in SOURCE_STATUS_SKIP:
                    continue
                scans[sbid] = scan_source(
                    expanded_vector(sp, self.mapper),
                    target_stats, target_bids)
            try:
                self._verify_suspects_for_type(
                    src_type, pool, scans, decision,
                    target_id2type=target_id2type, sizes=sizes,
                    weights=weights, source_sides=source_sides,
                    target_sides=target_sides)
            finally:
                del scans

    def _fingerprint(self) -> Dict[str, Any]:
        """The run's `input_fingerprint`, created on first write.

        Bookkeeping must never take a stage down: a validator built without
        ``__init__`` (the unit-test stubs) would otherwise raise an
        ``AttributeError`` here — and inside the backward pass's ``try`` that
        mislabels every in-budget row ``error``, i.e. "the scan failed", when
        nothing was even scanned.
        """
        fp = getattr(self, 'input_fingerprint', None)
        if fp is None:
            fp = self.input_fingerprint = {}
        return fp

    def _record_run_origin(self) -> None:
        """Say WHICH TREE produced this run before anything is scored.

        The fingerprint used to be written by the stage-2 scan, so a run that
        resolved no pairs at all — the target dataset has no local data, or
        the query is a lone held fan-out — reached `parameters.json` with no
        provenance whatsoever: no `git_rev`, no `git_dirty`, nothing to place
        the empty run. Measured twice on the MCNS->FAFB attempts of
        2026-09-25/26, both of which published an untraceable run folder.
        The runs that fail to start are the ones a reader most needs to
        locate, so the origin is recorded at stage 1."""
        self._fingerprint().update({
            'git_rev': _git_rev(),
            'git_dirty': _git_dirty(),
        })

    def _record_scan_universe(self, vectors, target_stats) -> None:
        """Publish the inputs the scan scored against.

        The scoring backend is a pure function of (query, target universe,
        caches), so an old-code vs new-code A/B is only meaningful on an
        identical store — and this block is what shows a reader, from the
        run folder alone, whether the two runs shared one or drifted (a
        concurrent profile merge moves the universe under a run)."""
        cfg = self.cfg
        fp = self._fingerprint()
        # Stage 1 already named the tree the run started on. If the worktree
        # moved since (a concurrent session commits into the live tree this
        # queue runs against), keep BOTH: the primary keys say what scored the
        # numbers, the `_at_start` pair says what the run believed at first.
        start_rev, start_dirty = fp.get('git_rev'), fp.get('git_dirty')
        rev, dirty = _git_rev(), _git_dirty()
        if start_rev is not None and (start_rev != rev
                                      or start_dirty != dirty):
            fp['git_rev_at_start'] = start_rev
            fp['git_dirty_at_start'] = start_dirty
            self.log(f'    ! the worktree moved during this run: started '
                     f'{start_rev}{"-dirty" if start_dirty else ""}, '
                     f'scanning against {rev}{"-dirty" if dirty else ""}')
        fp.update({
            'git_rev': rev,
            'git_dirty': dirty,
            'scanned_target_universe': len(target_stats),
            'target_vectors_built': len(vectors),
            'profile_cache_source': _store_identity(
                self.profiler._get_cache_parquet_path(cfg.source_dataset)),
            'profile_cache_target': _store_identity(
                self.profiler._get_cache_parquet_path(cfg.target_dataset)),
            'mapper_snapshot': _store_identity(
                getattr(self.mapper, '_mapper_snapshot_path',
                        lambda: None)()),
        })

    def _record_morph_stores(self, pass_name: str = 'supervised') -> None:
        """Name the morphology stores this run scores out of, KEYED BY PASS.

        Recorded when a morph pass runs, not at stage 2: the scan's
        fingerprint is taken before stage 5 fetches anything, and what
        decides a native verdict is the store as it stands at scoring time.
        A pooling run runs two passes (the supervised stage 5 and the
        pooling qualification); the old flat assignment let the second
        erase the first, so `parameters.json`'s `morph_stores` meant
        different things per mode — both passes are now named side by side,
        from the run-baseline store state (first observation; the identity
        walk stats every skeleton file, so a two-pass run walks each store
        once, not twice)."""
        cfg = self.cfg
        memo = getattr(self, '_morph_store_identity_memo', None)
        if memo is None:
            memo = self._morph_store_identity_memo = {}

        def _identity(dataset: str) -> Dict[str, Any]:
            if dataset not in memo:
                memo[dataset] = _morph_store_identity(dataset)
            return memo[dataset]

        self._fingerprint().setdefault('morph_stores', {})[pass_name] = {
            'target': _identity(cfg.target_dataset),
            'source': _identity(cfg.source_dataset),
        }

    def _suspects_only_pass(self) -> None:
        """P3 for a run whose EVERY queried type was fail-closed (e.g. a
        lone held fan-out): there are no pairs to validate, but the
        rivals still deserve their advisory verification.  Builds the
        minimal stage-2 target frame, then runs the excluded-types
        hook."""
        cfg = self.cfg
        self.log('[stage 2-lite] building target vectors for the '
                 'suspects pass (--verify-suspects)')
        if not cfg.skip_profile_build:
            try:
                self._preflight_target_profiles(cfg.target_dataset)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[TMVEV] profile pre-flight failed '
                         f'(continuing cache-only): {exc}')
        vectors = build_target_vectors(
            self.profiler, cfg.target_dataset, self.mapper, cfg.verbose,
            min_weight=cfg.target_min_weight,
            min_partner_types=cfg.target_min_partner_types)
        target_stats = prep_target_stats(vectors)
        target_bids = list(target_stats)
        self._record_scan_universe(vectors, target_stats)
        target_id2type = self._target_types(target_bids)
        self._target_sizes = load_caliber_map(cfg.target_dataset)
        self._source_sizes = load_caliber_map(cfg.source_dataset)
        self._target_sides = load_hemisphere_map(cfg.target_dataset)
        self._source_sides = load_hemisphere_map(cfg.source_dataset)
        self._source_type_counts, self._source_add_counts = \
            load_source_type_counts(cfg.source_dataset)
        self._target_weights = {bid: s.sum1 for bid, s in
                                target_stats.items()}
        self._verify_suspects_for_excluded(
            target_id2type=target_id2type,
            sizes=self._target_sizes, weights=self._target_weights,
            source_sides=self._source_sides,
            target_sides=self._target_sides,
            target_stats=target_stats, target_bids=target_bids)
        n_rows = len(getattr(self, '_suspects_verification_rows', []) or [])
        if n_rows:
            self.log(f'[TMVEV] suspects verification: {n_rows} rows '
                     '(no-pair run)')

    def run(self) -> Path:
        cfg = self.cfg
        t_start = time.time()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base = Path(cfg.output_dir) if cfg.output_dir else \
            Path(__file__).resolve().parents[2] / 'local_data' / \
            'mapping_validation'
        # The analysis ROOT folder carries the type-map-validation
        # prefix with the SHORT dataset nicknames (user 2026-09-18;
        # scene folders inside keep their own plot-3d naming).  The
        # --label stays in parameters.json / the report, not the name.
        self.run_dir = base / (
            f'type-map-validation_{_short_name(cfg.source_dataset)}_to_'
            f'{_short_name(cfg.target_dataset)}_{stamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressReporter(self.run_dir)
        self.progress.emit('run_start', source_dataset=cfg.source_dataset,
                           target_dataset=cfg.target_dataset,
                           query_types=list(cfg.query_types),
                           mode=cfg.effective_mode)

        # Record the tree before stage 1 can end the run: an empty run must
        # still be placeable.
        self._record_run_origin()

        self.progress.emit('stage_start', stage='1', label='resolve')
        self.log(f'[stage 1] resolving type pairs '
                 f'{cfg.source_dataset} -> {cfg.target_dataset}')
        self.pairs = self.resolve_type_pairs()
        if not self.pairs:
            self.log('! [resolution] no valid type pairs resolved; '
                     'nothing to validate — check that the TARGET '
                     f'dataset ({cfg.target_dataset}) has local data and '
                     'that the queried types exist on both sides')
            # P3: a no-pair run (e.g. a lone held fan-out) still verifies
            # its rivals when the pass is opted in.
            if cfg.verify_suspects:
                try:
                    self._suspects_only_pass()
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    self.log(f'[TMVEV] suspects verification (no-pair '
                             f'run) failed (advisory, skipped): {exc}')
                    self.log(traceback.format_exc())
            self._write_outputs([], [], [], [], None, None, [], [], [])
            return self.run_dir

        self.progress.emit('stage_done', stage='1', label='resolve')
        self.progress.emit('stage_start', stage='2', label='scans')
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
        self._record_scan_universe(vectors, target_stats)
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
            self.progress.emit('scan_progress', stage='2',
                               source_type=src_type,
                               done=n_type, total=len(by_src_type),
                               sources=len(pool), scanned=len(scans),
                               targets=len(target_bids),
                               elapsed_s=round(time.time() - t0, 1))
            self._record_forward_matches(src_type, plist, scans,
                                         target_id2type)
            for pair in plist:
                res = self.validate_pair(pair, scans, target_id2type,
                                         sizes=self._target_sizes,
                                         weights=self._target_weights,
                                         source_sides=self._source_sides,
                                         target_sides=self._target_sides)
                per_pair_res[(pair.query,) + pair.key] = res
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
                         f'best={res["summary"]["best"]}, '
                         f'gap={res["summary"]["gap"]} '
                         f'({res["summary"]["gap_ratio"]:.0%}), '
                         f'triggered={res["summary"]["gap_triggered"]}, '
                         f'examinee rows={len(res["suspicious"])}, '
                         f'noise filtered={len(res["noise"])}')
            # P3 (opt-in): verify this type's rival suspects against the
            # SAME scan frames — advisory, separate accumulators.
            if cfg.verify_suspects:
                try:
                    # TMV-4: stamp THIS type's query — stage 1 left
                    # `_current_query` at the last type it resolved, and
                    # the suspects rows inherited that stale value.
                    self._current_query = (
                        plist[0].query if plist else src_type)
                    self._verify_suspects_for_type(
                        src_type, pool, scans,
                        self._suspects_decision(src_type),
                        target_id2type=target_id2type,
                        sizes=self._target_sizes,
                        weights=self._target_weights,
                        source_sides=self._source_sides,
                        target_sides=self._target_sides)
                    # §three-tier: the disclosure ends of the SAME type
                    # get the same verification (advisory, separate bin)
                    self._verify_disclosure_targets(
                        src_type, pool, scans,
                        self.mapper.get_mapping_decision(
                            src_type, cfg.source_dataset,
                            cfg.target_dataset),
                        target_id2type=target_id2type,
                        sizes=self._target_sizes,
                        weights=self._target_weights,
                        source_sides=self._source_sides,
                        target_sides=self._target_sides)
                except Exception as exc:  # noqa: BLE001
                    self.log(f'  [suspects] verification failed for '
                             f'{src_type} (advisory, skipped): {exc}')
            del scans

        # P3 second hook (opt-in): held / evidence-only types were never
        # scanned (no validated pairs) — build their scans now so their
        # rivals get verified too (D-A: all queried fan-outs).
        if cfg.verify_suspects:
            try:
                self._verify_suspects_for_excluded(
                    target_id2type=target_id2type,
                    sizes=self._target_sizes,
                    weights=self._target_weights,
                    source_sides=self._source_sides,
                    target_sides=self._target_sides,
                    target_stats=target_stats, target_bids=target_bids)
            except Exception as exc:  # noqa: BLE001
                self.log(f'[TMVEV] suspects verification (excluded types) '
                         f'failed (advisory, skipped): {exc}')
            n_sus_ver = len(getattr(self, '_suspects_verification_rows',
                                    []) or [])
            n_sus_types = len(getattr(self, '_suspects_verified_types',
                                      set()) or set())
            if n_sus_ver:
                self.log(f'[TMVEV] suspects verification: {n_sus_ver} '
                         f'rows across {n_sus_types} source type(s)')

        # Backward source status (plan-backward-source-status.md): the
        # column view of the SAME pair scores — read-only `source-`
        # statuses per in-branch source (advisory, D-B11), plus the
        # source-candidates regroup (foreign sources whose qualified
        # sibling rows point into the branch pool). Additive: nothing
        # downstream consumes these for gating. FAIL-OPEN: any failure
        # here logs and skips — an advisory layer must never kill a run.
        try:
            self._backward_source_pass(per_pair_res, all_sus_rows)
        except Exception as exc:
            import traceback
            self.log(f'[TMVEV] backward source status failed '
                     f'(advisory, skipped): {exc}')
            self.log(traceback.format_exc())


        # Revision 3.6: classify every invader BEFORE morph promotion —
        # structural facts (sibling pools, collapsed chains, real backward
        # homes) outrank morphology.  Deep-window candidates get the same
        # treatment (a deep sibling/backward member is structural, not a
        # candidate).
        family_types = {p.source_type for p in self.pairs}
        self.annotate_invaders(all_sus_rows + all_deep_rows, all_fills,
                               per_pair_res, family_types=family_types)

        # Stage '2' opens right after stage 1 and spans the target-vector
        # build plus every per-source-type scan frame.  It used to close at
        # `run_done`, so 25 minutes of work sat inside one opaque stage and
        # cost could only be attributed from artifact mtimes; the blocks
        # below each get their own timed stage now.
        self.progress.emit('stage_done', stage='2', label='scans')

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
            self.progress.emit('stage_start', stage='5', label='morphology')
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
            self.progress.emit('stage_done', stage='5', label='morphology')
            # Stage 5 may have demoted verified_strong rows below the AUC
            # threshold — refresh the frozen summary counters so
            # pair_summary.csv matches validation_results.csv (TMV-3).
            self._resync_summary_verdicts(per_pair_res)

        # Revision 3.12: compute the category partition + query-level
        # dedup AFTER morphology (qualification is now known).  This sets
        # `category`/`in_scope`/`morph_failed`/annotation/counts on every
        # evidence row and returns the dedup rows.
        dedup_rows: List[Dict] = []
        family_rows_out: List[Dict] = []
        relatives_out: List[Dict] = []
        self.progress.emit('stage_start', stage='3', label='categories')
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
        self.progress.emit('stage_done', stage='3', label='categories')

        # Stage 5d: backward (target -> source) homolog evidence for the
        # candidates / family / relative bins.  ADVISORY — it labels rows and
        # rebuilds the dedup rollup; no category or counts_toward_* flag
        # changes, so the fill totals stay identical to a run without it.
        # FAIL-OPEN: an advisory layer must never kill a run.
        self.progress.emit('stage_start', stage='5d',
                           label='backward evidence')
        try:
            rebuilt = self._backward_expansion_pass()
            if rebuilt:
                dedup_rows = rebuilt
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(f'[stage 5d] backward evidence failed '
                     f'(advisory, skipped): {exc}')
            self.log(traceback.format_exc())
        self.progress.emit('stage_done', stage='5d',
                           label='backward evidence')

        # Revision 3.10: set-level coverage — the deliverable for
        # "how much of the source→target mapping is validated, proposed,
        # and still missing" (per-branch gaps are diagnostics).  Computed
        # AFTER finalize_categories so the fills it counts carry the final
        # category-qualified `counts_toward_gap_fill` (Rev 3.12).
        coverage = None
        self.progress.emit('stage_start', stage='3b',
                           label='coverage accounting')
        try:
            # P2/P4 accounting counters (advisory, additive): held /
            # evidence-only same-name fan-outs + multivalue cells.
            snf_excluded = getattr(self, '_same_name_excluded', []) or []
            extra = {}
            if snf_excluded:
                # TMV-8: a type reachable from TWO query tokens is recorded
                # twice — count TYPES, not records (the record list itself
                # keeps the per-query provenance).
                seen_types = {r.get('source_type') for r in snf_excluded}
                held = {r.get('source_type') for r in snf_excluded
                        if r.get('disposition') == 'gated_held'}
                excl = {r.get('source_type') for r in snf_excluded
                        if r.get('disposition') == 'excluded_evidence_only'}
                multi = {r.get('source_type') for r in snf_excluded
                         if r.get('reason') == 'multivalue_cell'}
                if held:
                    extra['same_name_first_held'] = len(held)
                if excl:
                    extra['same_name_first_excluded'] = len(excl)
                if multi:
                    extra['multivalue_types'] = len(multi)
                extra.setdefault('same_name_first_types_total',
                                 len(seen_types))
            if getattr(self, '_multivalue_target_skips', 0):
                extra['multivalue_target_types'] = int(
                    self._multivalue_target_skips)
            coverage = compute_set_coverage(self.pairs, per_pair_res,
                                            all_fills,
                                            evidence_rows=all_sus_rows
                                            + all_deep_rows,
                                            extra_counters=extra)
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
                (coverage or {}).get('target', {}).get('family_material', []))
            if coverage is not None:
                coverage['gap_fill_by_level'] = dict(gap_by_level)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(f'[gap fill levels] failed: {exc}')
            self.log(traceback.format_exc())
        self.progress.emit('stage_done', stage='3b',
                           label='coverage accounting')

        # `pooling` mode (plan-tmvev-pooling-mode.md): the unsupervised
        # candidate engine, on the universe stage 2 already built so nothing
        # re-scans the vectors.  Reachable only through `--mode pooling`.
        self._pooling = None
        if cfg.effective_mode == POOLING_MODE:
            self.progress.emit('stage_start', stage='P',
                               label='pooling (unsupervised)',
                               query_types=list(cfg.query_types))
            try:
                from comparison.mapping_validation_pooling import run_pooling
                self._pooling = run_pooling(
                    self, target_stats=target_stats, target_bids=target_bids,
                    target_id2type=target_id2type, val_rows=all_val_rows)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[stage P] pooling failed: {exc}')
                self.log(traceback.format_exc())
                self.progress.emit('warning', stage='P',
                                   note=f'pooling failed: {exc}')
                # a failed pass still writes its exports, carrying the reason:
                # a missing folder reads as "nothing found", which is a
                # different claim than "the pass did not run"
                self._pooling = {'candidates': [], 'pool': [],
                                 'cross_validation': {
                                     'error': f'{type(exc).__name__}: {exc}'}}
            _p = self._pooling or {}
            self.progress.emit(
                'stage_done', stage='P', label='pooling (unsupervised)',
                rows=len(_p.get('candidates') or []),
                pool=len(_p.get('pool') or []))

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
            pool_owner: Dict[int, tuple] = {}
            for key, res in per_pair_res.items():
                for b in (res.get('_pool_set') or []):
                    in_map.add(int(b))
                    pool_owner[int(b)] = key
            self.progress.emit('stage_start', stage='expansion',
                               label='out-map expansion',
                               unclaimed_sources=n_out_map)
            try:
                out_map_rows, cand_raw = self._expand_out_map_sources(
                    out_map_by_type, in_map, target_stats, target_bids,
                    target_id2type, cfg.out_map_top_k,
                    pool_owner=pool_owner)
                # Source-candidates, RE-AIMED (plan §10, user option 2):
                # out-of-map sources whose best-ranked hits land in a
                # branch pool, morph-qualified against the run null bar
                # (D-B8's mirror of candidate admission — now actually
                # foreign, unlike the sibling-row route).
                self._collect_source_candidates(cand_raw)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[TMVEV] out-map expansion failed: {exc}')
                self.log(traceback.format_exc())
            self.progress.emit('stage_done', stage='expansion',
                               label='out-map expansion',
                               rows=len(out_map_rows))

        # The scene reads the expansion rows — expose them BEFORE stage 4.
        self._out_map_expansion_rows = out_map_rows
        self._out_map_by_type = out_map_by_type

        # Stage 5e (user 2026-09-26): every appeared target bodyId scanned
        # back against the whole source dataset — the Homolog · backward
        # panel's payload.  Advisory display data only, fail-open.
        _tgt_universe = self._appeared_target_bids(
            all_val_rows, all_sus_rows, all_deep_rows, all_noise_rows,
            all_fills, all_pool_detail, out_map_rows, dedup_rows,
            relatives_out, family_rows_out)
        try:
            self._target_match_pass(_tgt_universe)
        except Exception as exc:
            import traceback
            self.log(f'[TMVEV] target match pass failed '
                     f'(advisory, skipped): {exc}')
            self.log(traceback.format_exc())

        if cfg.visualize:
            self.progress.emit('stage_start', stage='4', label='scenes')
            try:
                from comparison.mapping_validation_visualize import \
                    render_pair_scenes
                render_pair_scenes(self, per_pair_res)
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.log(f'[stage 4] visualization failed: {exc}')
                self.log(traceback.format_exc())
            self.progress.emit('stage_done', stage='4', label='scenes')

        # Export contract (Rev 3.6/2c): source caliber travels with the
        # validation rows.
        self.progress.emit('stage_start', stage='6', label='report')
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
        self.progress.emit('stage_done', stage='6', label='report')
        self.progress.emit('run_done',
                           run_dir=str(self.run_dir),
                           elapsed_s=round(time.time() - t_start, 1))
        self.log(f'done in {time.time() - t_start:.0f}s -> {self.run_dir}')
        # Refresh the slim README so its raw run log also carries the
        # closing lines (report written / done) — the copy written inside
        # `_write_outputs` predates them.  The report.html is refreshed
        # for the same reason (the run_done elapsed lands in its
        # header/timeline); notes append is idempotent.
        self._write_readme(summaries, morph_info, coverage)
        try:
            from comparison.mapping_validation_report import (
                collect_and_write as _report_refresh)
            _report_refresh(self.run_dir, log=self.log)
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! run report refresh failed: {exc}')
        return self.run_dir

    def _target_types(self, target_bids: List[int]) -> Dict[int, str]:
        """Type label per target bodyId (for examinee/fill reporting)."""
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
        # Every export goes through `_write_run_csv` so an empty result is
        # a header-only file (never zero bytes — user 2026-09-18).
        _write_run_csv(rd, 'validation_results.csv', val_rows)
        _write_run_csv(rd, 'examinees.csv', sus_rows)
        _write_run_csv(rd, 'noise_filtered_candidates.csv',
                       noise_rows or [])
        _write_run_csv(rd, 'deep_candidates.csv', deep_rows or [])
        _write_run_csv(rd, 'relatives.csv', relatives or [])
        _write_run_csv(rd, 'family_candidates.csv', family_rows or [])
        _write_run_csv(rd, 'gap_fill_dedup.csv', dedup_rows or [])
        _write_run_csv(rd, 'gap_fill_levels.csv', gap_levels or [])
        _write_run_csv(rd, 'source_status.csv',
                       getattr(self, '_source_status_rows', None) or [])
        # source-candidates flattened (branch attribution + the dup flag)
        # so the offline report reads the SAME rows the scene renders.
        _sc_rows = []
        for _bk, _rows in (getattr(self, '_source_candidates', {}) or {}).items():
            for _c in _rows:
                _row = dict(_c)
                _row['query'] = _bk[0] if isinstance(_bk, tuple) and len(_bk) >= 1 else ''
                _row['branch_source_type'] = _bk[1] if isinstance(_bk, tuple) and len(_bk) >= 2 else ''
                _row['branch_target_type'] = _bk[2] if isinstance(_bk, tuple) and len(_bk) >= 3 else ''
                _row['dup'] = int(_c['source_bodyId']) in (
                    getattr(self, '_source_candidates_multi', set()) or set())
                _sc_rows.append(_row)
        _write_run_csv(rd, 'source_candidates.csv', _sc_rows)
        # stage 5d: one row per (branch, member neuron) with the reverse
        # top-1 + the serialized top-N neighbourhood (advisory label only).
        _write_run_csv(rd, 'backward_matches.csv',
                       getattr(self, '_backward_matches_rows', None) or [])
        # Homolog panels (user 2026-09-26): the stage-2 forward capture
        # gets its morph display join here (morphology runs after stage 2);
        # the stage-5e rows get their pool categories + the same join.
        _morph_idx = self._morph_pair_index(val_rows, sus_rows, fills,
                                            out_map_rows, deep_rows)
        _write_run_csv(rd, 'forward_matches.csv',
                       self._finalize_forward_match_rows(_morph_idx))
        _write_run_csv(rd, 'target_matches.csv',
                       self._finalize_target_match_rows(pool_detail,
                                                        _morph_idx))
        _write_run_csv(rd, 'same_name_excluded.csv',
                       getattr(self, '_same_name_excluded', None) or [])
        # §three-tier delivery: the disclosure ends (declined but
        # evidence-reached), annotated with the advisory verification
        # verdicts when --verify-suspects ran.  Written only when any were
        # recorded — a run with no disclosure gains no file (the pooling
        # precedent).
        _disc_rows = list(getattr(self, '_disclosure_records', None) or [])
        if _disc_rows:
            _disc_ver = getattr(self, '_disclosure_verification',
                                None) or {}
            for row in _disc_rows:
                v = _disc_ver.get((str(row.get('query')),
                                   str(row.get('source_type')),
                                   str(row.get('target_type')))) or {}
                row['verdict'] = v.get('verdict', '')
                row['rank_union'] = v.get('rank_union', '')
            _write_run_csv(rd, 'disclosure_evidence.csv', _disc_rows)
        if getattr(self.cfg, 'verify_suspects', False):
            _write_run_csv(rd, 'suspects_verification.csv',
                           getattr(self, '_suspects_verification_rows', None)
                           or [])
        _write_run_csv(rd, 'out_map_expansion.csv', out_map_rows or [])
        # pooling mode: the unsupervised pool and its post-hoc comparison.
        # Written only when the pass ran — a non-pooling run gains no file.
        pooling = getattr(self, '_pooling', None) or {}
        if pooling:
            _write_run_csv(rd, 'pooling_candidates.csv',
                           pooling.get('candidates') or [])
            _write_run_csv(rd, 'pooling_pool.csv', pooling.get('pool') or [])
            _write_run_csv(rd, 'pooling_sources.csv',
                           pooling.get('sources') or [])
            run_file_path(rd, 'pooling_cross_validation.json',
                          create_parent=True).write_text(
                json.dumps(pooling.get('cross_validation') or {}, indent=2,
                           default=str),
                encoding='utf-8')
        _write_run_csv(rd, 'pair_summary.csv', summaries)
        _write_run_csv(rd, 'gap_fill_proposals.csv', fills)
        _write_run_csv(rd, 'pool_categories.csv', pool_detail)
        _write_run_csv(rd, 'mapping_export.csv', self._mapping_export_rows())
        run_file_path(rd, 'parameters.json',
                      create_parent=True).write_text(json.dumps({
            'source_dataset': self.cfg.source_dataset,
            'target_dataset': self.cfg.target_dataset,
            'query_types': self.cfg.query_types,
            # the folder name carries only the dataset nicknames + the
            # timestamp, so this is where a --label survives
            'run_label': self.cfg.run_label,
            # the inputs the scan scored against: an A/B of two code trees
            # is only meaningful on one frozen store, and this says whether
            # it was (see _record_scan_universe)
            'input_fingerprint': dict(sorted(
                (getattr(self, 'input_fingerprint', None) or {}).items())),
            'top_k': self.cfg.top_k, 'top_m': self.cfg.top_m,
            'min_synapse_threshold': self.cfg.min_synapse_threshold,
            'include_untyped_partners': self.cfg.include_untyped_partners,
            'rank_top_k': self.cfg.rank_top_k,
            'gap_min': self.cfg.gap_min,
            'gap_trigger_retired': True,
            'verify_suspects': self.cfg.verify_suspects,
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
            # `pooling` mode (read only when validation_mode == 'pooling'):
            # the admission bar, the floors it only flags, the window and the
            # morph budget.
            'pooling_jaccard_floor': self.cfg.pooling_jaccard_floor,
            'pooling_rank_union_floor': self.cfg.pooling_rank_union_floor,
            'pooling_window_mult': self.cfg.pooling_window_mult,
            'pooling_bar_metric': self.cfg.pooling_bar_metric,
            'pooling_bar_top_n': self.cfg.pooling_bar_top_n,
            'pooling_max_morph_targets': self.cfg.pooling_max_morph_targets,
            'max_scenes': self.cfg.max_scenes,
            # The Advanced Visualization panel is now the only writer of this
            # knob, and the scene look is only reproducible from provenance if
            # all three of its parts are recorded — opacity included.
            'neuron_alpha': self.cfg.neuron_alpha,
            'scene_selfcheck': self.cfg.scene_selfcheck,
            # The scene LOOK as this run actually wore it (see
            # scene_styling_record): the resolved styling kwargs and the full
            # effective category palette, so the legend swatches and mesh style
            # of the shipped pages are reproducible from provenance alone.
            **scene_styling_record(self.cfg, self.log),
            'morph_enabled': self.cfg.morph_enabled,
            'morph_auc_floor': self.cfg.morph_auc_floor,
            'suspicious_per_source_cap': self.cfg.suspicious_per_source_cap,
            # stage 5d (advisory; connectivity only — no morphology)
            'backward_evidence_enabled': self.cfg.backward_evidence_enabled,
            'skip_backward_pass': self.cfg.skip_backward_pass,
            'backward_top_n': self.cfg.backward_top_n,
            'backward_max_neurons': self.cfg.backward_max_neurons,
            'backward_per_branch_cap': self.cfg.backward_per_branch_cap,
            'backward_scan_pool_targets':
                self.cfg.backward_scan_pool_targets,
            'backward_evidence': dict(
                getattr(self, '_backward_counters', None) or {}),
        }, indent=2), encoding='utf-8')
        if morph_info:
            run_file_path(rd, 'morphology_calibration.json',
                          create_parent=True).write_text(
                json.dumps(morph_info, indent=2, default=str),
                encoding='utf-8')
        if set_coverage:
            run_file_path(rd, 'set_coverage.json',
                          create_parent=True).write_text(
                json.dumps(self._set_coverage_payload(set_coverage),
                           indent=2, default=str), encoding='utf-8')
        self._write_readme(summaries, morph_info, set_coverage)
        # per-run HTML report (report.html) + user warning notes —
        # fail-open, never blocks the run
        try:
            from comparison.mapping_validation_report import \
                finish_run_outputs
            finish_run_outputs(
                rd,
                mapper_gap_types=getattr(self, '_mapper_gap_types',
                                         None),
                mapper_gap_untyped=getattr(self,
                                           '_mapper_gap_untyped', 0),
                log=self.log)
        except Exception as exc:  # noqa: BLE001
            self.log(f'    ! run report failed: {exc}')

    def _write_readme(self, summaries, morph_info, set_coverage=None):
        """Slim README: directions + raw run log only.

        The analysis content (coverage levels, branches, fills,
        expansion bins, morph record, scenes, column glossary) lives in
        ``report.html``, generated right after this by
        ``mapping_validation_report``; ``summaries`` / ``morph_info`` /
        ``set_coverage`` stay in their own CSV/JSON artifacts.  Every
        ``!`` warning line reaches the user through the run log here AND
        the appended ``user_warning_notes.txt``.  Pure function of
        ``self.notes`` — ``run()`` writes it inside ``_write_outputs``
        and refreshes it once more after the closing log lines so the
        captured log is complete.
        """
        lines = [
            'Type-mapping validation run',
            '===========================',
            f'source: {self.cfg.source_dataset}  target: '
            f'{self.cfg.target_dataset}',
            f'queries: {", ".join(self.cfg.query_types)}',
            f'mode: {self.cfg.effective_mode}',
            f'label: {self.cfg.run_label}',
            '',
            'Start here:',
            '- report.html — the run report: headline + coverage levels '
            '(L1 claim / L2 provenance / L3 validation), branches,',
            '  fills, the Reciprocal tab (on --backward-evidence runs), '
            'the Homolog · forward / Homolog · backward tabs (one row '
            'per appeared bodyId, both directions), out-map expansion, '
            'morphology record, scenes; hover any term for its definition.',
            '- set_coverage.json — set-level coverage (per-type '
            'rollups, hole bodyIds, family_material).',
            '- gap_fill/gap_fill_dedup.csv — the bodyId-unique fill; '
            'gap_fill/gap_fill_levels.csv ranks it per branch.',
            '- validation/examinees.csv — expansion bins (Revision '
            '3.12 categories; the aggressive deep-window leaf was '
            'renamed from suspicious_candidates.csv); '
            'validation/noise_filtered_candidates.csv holds the '
            'gated rows.',
            '- mapping/same_name_excluded.csv — queried types whose '
            'same-name fan-out was held/excluded by the mapper, or '
            'multi-value type cells (kept atomic); advisory accounting, '
            'never a gate.',
            '- mapping/suspects_verification.csv (only with '
            '--verify-suspects) — advisory connectivity verification of '
            'the rivals listed in auto_type_mapping_suspects.csv, one '
            'row per source × rival with the ordinary verdict tiers; '
            'never merged into the validation counts.',
            '- expansion/source_status.csv — backward `source-` status '
            'per in-branch source (column view of the same pair '
            'scores); advisory, never a gate.',
            '- expansion/source_candidates.csv — out-of-map sources '
            'whose best-ranked hits reach a branch pool (null-bar '
            'morph-qualified); the scenes\' source-candidates roots '
            '(hidden by default).',
            '- expansion/backward_matches.csv (only with '
            '--backward-evidence) — each candidates/family/relative '
            'member and each UNMATCHED pool member scanned in REVERSE '
            'against the whole source universe: its top-1 source, '
            'whether that source is in the branch, how much of the '
            'union the score rests on '
            '(backward_shared_type_count / _union_type_count, with '
            'backward_thin_evidence at <= 3 shared types), the '
            'backward_own_type_* block naming the hit of the branch\'s '
            'OWN source type that the high/medium/low grade is actually '
            'computed from (including which ranking surfaced it: '
            '_via), and the serialized top-N listed in the report\'s '
            'order (jaccard first). Already-mapped members '
            '(matched/verified/borderline) are not re-scanned: their '
            'forward pair is the symmetric evidence. Connectivity-only '
            '(morphology is never re-scored here) and advisory: it '
            'labels the bins, it never changes a fill count.',
            '- expansion/out_map_expansion.csv — each unclaimed '
            'source\'s top-k typed non-in-map expansion candidates, '
            'morph-checked against the run null bar.',
            '- validation/forward_matches.csv — the Homolog · forward '
            'tab: one row per appeared source bodyId (assigned, '
            'fill-proposed, out-of-map or unpaired) with its chain-best '
            'target, the top-3 rank_union ∪ top-3 jaccard neighbourhood, '
            'and the primary pair\'s morph display join. `untyped` '
            'marks an unannotated neuron.',
            '- expansion/target_matches.csv — the Homolog · backward '
            'tab: every appeared target bodyId (pool, expansion-bin, '
            'out-map and proposal targets) scanned back against the '
            'WHOLE source dataset; `untyped` marks an unannotated '
            'neuron. Advisory display data only.',
            # a pooling run's own result is in none of the nested bins, so the
            # folder index has to point at it — an unlisted folder of exports
            # reads as "nothing was found" to anyone working from README.txt
            *([] if self.cfg.effective_mode != POOLING_MODE else [
                '- pooling/pooling_candidates.csv / pooling_pool.csv / '
                'pooling_sources.csv / '
                'pooling_cross_validation.json (this run, --mode pooling) — '
                'the UNSUPERVISED pool: every queried neuron scanned against '
                'the whole target universe, each source keeping its top-N rows '
                'under the admission bar (the Jaccard / rank_union floors and '
                'the window are published as ADVISORY flags and remove '
                'nothing), then the morphology bar as the last gate — a '
                'refusal leaves the pool and is counted in '
                '`morph.dropped_targets`, while a candidate with no verdict '
                'stays and is named; a pass that raised or graded nothing '
                'did NOT gate this pool (`morph.gate_applied` false — its '
                'tiers are unrefused, not passed); `morph.budget` says what '
                'priced the pass. '
                'The mapper is joined afterwards. Read the report\'s Pooling '
                'tab, whose Per-source block is the mode\'s own axis.']),
            '- validation/pair_summary.csv — per-branch pools / gap / '
            'verdicts; mapping/mapping_export.csv — the branch mapping '
            'with bodyId pools.',
            '- morphology_calibration.json — branch bars, run null '
            'bar, AUC gate, score frames.',
            '- visualization/ — one 3D scene per parent type (rendered '
            'in SOURCE coordinates; scores live in TARGET '
            'coordinates).',
            'Fills are proposals only; the mapping is never rewritten.',
            '',
            'Run log:', *self.notes, '',
        ]
        run_file_path(self.run_dir, 'README.txt',
                      create_parent=True).write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')

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
                'tier': getattr(p, 'tier', 'claim'),
                'same_name_first': bool(p.same_name_first),
                'same_name_rivals': ';'.join(
                    p.same_name_first.get('rivals') or [])
                if p.same_name_first else '',
                'query': p.query,
                'is_selected': True,
                'chain_rank': p.branch_index,
                'selected_bridge': p.chain_text,
                'source_bridge': p.source_chain_text,
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
