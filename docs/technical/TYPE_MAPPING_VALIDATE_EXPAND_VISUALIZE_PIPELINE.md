# Type-Mapping Validate-Expand-Visualize Pipeline — Technical Report

Module: `src/comparison/mapping_validation.py` (orchestrator),
`src/comparison/mapping_validation_visualize.py` (Stage-4 scenes),
`scripts/RunMappingValidation.py` (CLI).

Plans: `_plan/plan-type-mapping-validation-pipeline.md` (Revisions
1–3.5), `_plan/plan-mapping-validation-rev36-invader-reclassification.md`
(Revisions 3.6–3.9), and
`_plan/plan-mapping-validation-rev311-chain-aware-pool-widening.md` (§2b,
the category model — **normative**). Status: the pipeline is implemented
and verified on real data (FAFB v783 → male-cns v1.0; s-CPDN3C/D,
circadian_clock 242→204 — the in-map claims, of the in-map types' 219
populations, AN* gap sets). The **category taxonomy in §4 is
the design of record**; where the existing code disagrees, the design
governs and the code is to be brought into line (plan tasks T1–T12).

---

## 1. Purpose

The auto type mapper (`CrossDatasetTypeMapper`) produces **type-level**
matches with per-type bodyId pools, but nothing verifies the pairing at
**bodyId granularity**. This pipeline closes that loop:

1. **validate** — for every resolved branch, tier each in-map target
   bodyId by global cross-dataset connectivity evidence (§3);
2. **expand** — from the homolog finding, add the examinee neurons the
   type mapping missed, cross-referenced against the mapping structure
   and the morphology (§4, §6); stage 5d then asks the same finding the
   reverse question of the neurons it added, as advisory reciprocal
   evidence (§4.6a);
3. **visualize** — one 3D scene per parent mapping group with the full
   classification in the legend (§8).

The mapping is **never rewritten**: every proposal is evidence.

First-run direction (locked): source = FAFB v783, target = male-cns
v1.0. The machinery is direction-symmetric.

## 2. Stage 1 — branch resolution

Unit of validation is the **bridge-resolved branch** (linker-refined
source sub-pool ↔ target type), not the parent type pool:

- `get_mapping_decision(source_type, src_ds, tgt_ds)` enumerates
  candidate targets (fail-closed: `conflict`/`unmapped` excluded,
  `evidence_only` kept with all targets).
- `resolve_prioritized_bridge_pool` (`ui.neuron_index`) refines both
  pools through the selected bridge chain: the source pool is the
  linker-refined subset (basis "linker rows"); no-chain (same-name)
  pairs keep full populations (basis "full population").
  **Per-side basis (§15.3, r22):** the selected chain is the best single
  DERIVATION and the prioritizer ranks by fewest linkers, so on
  FAFB->BANC the winner is the 1-linker target-side hop
  `banc_v888/fafb_cell_type` — exact on the target, silent on the source,
  which left r22 validating every parent's WHOLE source population in
  every branch (756 pool slots for 242 neurons) while 7 of the type's 13
  supported chains carried source-side `additional_type(s)` rows. The
  source pool therefore prefers any supported chain of the SAME endpoint
  that narrows it (subset only — never a widening or a membership
  replacement, and never a union: the Rev 3.12 widening retirement
  stands), and `source_chain` records which one did. Measured on the
  r21/r22 branch sets: MCNS unchanged on all 43 branches (223 slots,
  residue 30 — byte-identical pools), BANC narrowed on 34 of 39 branches
  to 223 slots with 19 neurons left in the out-map residue, and 7 more
  keep their membership but are now honestly labelled "linker rows"
  instead of "full population".
  The basis vocabulary is owned by the type mapper
  (`cross_dataset_type_mapper.ROW_EVIDENCE_BASES` /
  `SOURCE_REFINING_BASES` / `basis_is_row_evidence`), because the resolver,
  the validation pipeline, the report's basis buckets, the scene branch
  labels and the branch-disjointness measure all read the same field — a
  consumer comparing it with one literal misread every other row-backed
  basis as a full population. The mapper UI is aligned with it too: the
  type-mapping panel and the mapping CSV's bridge hover name the chain that
  supplies the SOURCE pool when that is not the selected chain
  (`source_side_refinement_note`), so no view of one mapping claims a single
  bridge where two different chains serve the two sides.
  Ceiling, measured: the source side can only be named through FAFB's
  `additional_type(s)` (28,540 of 139,255 neurons), and FAFB never annotates
  a neuron with its OWN type name — so every same-name branch (APDN3 on BANC,
  DN1a / DN1pA / DN1pB / l-LNv on both) is structurally name-asserted and its
  wide pool is the correct terminal answer, not an unresolved gap (§15.7).
- Only the highest-ranked chain per target type survives
  prioritization; the collapsed alternatives are recorded as expansion
  evidence (§4, the `family`/`relative` bins).
- Branches of one parent are annotated with disjointness
  (`branches_disjoint` / `resolved_by_linkers`).

Example: FAFB `s-CPDN3D` (37 neurons) resolves into 6 branches —
CB3508 (7 via `additional_type=CB3508`), SMP219, SMP222 (4 via CB3612),
SMP223 (7 via CB3508), SMP227, SMP232.

## 3. Stage 2 — global scan and verdicts

For every source bodyId the standardized expanded-type vector is scored
against **every** target-dataset neuron with a cached profile
(~137 k for male-cns).  Since 2026-09-14 (plan
`plan-bodyid-level-granularity-in-type-mapper.md`, Revision 3) the
scoring backend lives in `comparison/body_id_resolver.py` —
`expanded_vector`, `score_one_candidate_fast`, `_SideStats`,
`scan_source`, `build_target_vectors`, `prep_target_stats` and the
quality/caliber/hemisphere loaders moved there verbatim, and
`mapping_validation` imports them back (single implementation;
`MappingValidator` runs on a `BodyIdResolver` with its benchmark
profiler injected).  The same backend also serves the
pool-scoped `assign_bodyids` resolution — on-demand profiles for the
resolved pool members only, no global scan — which backs THIS pipeline's
per-neuron split verification.  **Boundary (user, 2026-09-14): this
connectivity verification belongs to THIS pipeline — the type mapper
never uses it, and the former panel split view has been removed; any UI
for the verification is this pipeline's own:**

- **Exact production parity**: `expanded_vector` =
  `ProfileComparator._get_expanded_types_standardized(profile, 'both')`;
  `score_one_candidate_fast` reproduces the
  `batch_compare_cross_dataset` metric block (jaccard over expanded-type
  sets; rank_union = Pearson over average-tie rankdata of the union
  with missing = 0; NaN when either side is constant; cosine and
  weighted_jaccard on the same union). Unit-tested to 1e-9/1e-12
  against production.
- **The scan is indexed, and still exact.** A source neuron shares an
  expanded type with only ~0.5 % of a whole connectome (416–1,184 of
  banc_v888's 103,770 cached targets), yet scoring all of them pair-by-pair
  cost 3.64 s per source neuron — 98 % of it in `rank_union` work on rows
  that can only read `jaccard == 0`. `prep_target_stats` therefore returns a
  `TargetScanIndex`: a plain `bid → _SideStats` dict that additionally
  carries one postings list per type key and, per target, its key count and
  the squared sum of its average-tie ranks. `scan_source` reaches the
  positive block through the postings, scores those rows with the untouched
  `score_one_candidate_fast`, and fills the disjoint block arithmetically —
  including `rank_union`, whose padded-concatenation form is an identity, not
  an approximation (`_disjoint_rank_union`: each side's ranks are `q + ρ` over
  its own keys and `(q+1)/2` over the other's, so Pearson collapses to four
  scalars per side). A weight of zero breaks that reading, so such rows fall
  back to the scorer. The frame that comes out is the same frame — rows,
  order, columns, values — measured identical over 20 real sources × 103,770
  targets at 0.136 s vs 3.80 s per source
  (`tests/core/test_mapping_validation_scan_parity.py`), which is why no rank
  window, null sample or `chain_pos` consumer had to change.
- **The stage-5 skeleton pre-flight is batched and threaded.** It reaches
  the same store the stage-4 scenes use, so it is the run's only
  network-bound block; fetching one skeleton per request measured 3.0 s each
  (1,401 s for 467). It now goes through `fetch_skeletons_on_demand_batch`
  (64-body requests, `skeleton_fetch_workers` threads, default 8) in
  128-id chunks so the fetched neurons never all live in memory, bounded by
  `skeleton_fetch_timeout_s` of socket inactivity because neither
  neuprint-python nor navis exposes a per-request timeout. Cache-first,
  resumable and fail-open exactly as before.
- Ranks are competition-style (ties share the better rank, NaN last).
- **The ordering chain** (`body_id_resolver._CHAIN`, surfaced as
  `order_by_chain()` / `chain_key()`): Jaccard desc → rank_union desc as
  the tie-break → bodyId, applied at bodyId level everywhere a "best" or a
  "top-N" is taken — the published top-1, mutual-best assignment, a
  target's best source, gap-fill proposals, the out-map list and every
  hover. It sorts on the SCORES, not the rank columns, because a Jaccard
  tie block (36% of r16 reverse scans had one inside the top-5; 0% for
  rank_union) would otherwise delegate the choice to row order. Each scan
  also publishes `chain_pos`, a dense 1-based position, so "top-N" means N
  rows. The rank columns stay as verdict EVIDENCE; `rank_union` never
  filters a bodyId-level candidate (it keeps its positivity/margin gates,
  which are claims about rank_union's own scale).
- Each source is scanned ONCE per run and the scan is shared by all its
  branches. Retention: top-100 per metric + pool (`FILL_KEEP_TOP`).

**Source verdicts** (tiered, read off the POOL — quantified over its members,
so a label cannot move when the ordering key moves; the tiers stay
*nested*, verified ⊇ verified_strong):

- `verified_strong` — ONE pool member is global top-1 under BOTH rank_union
  and jaccard (rank_union must be positive — positivity policy). Both claims
  on one neuron is the whole point of the tier: an `∃ru-member` plus a separate
  `∃jaccard-member` reading is satisfied by two different partners (46 of r17's
  223 rows) and would say "agreed by both metrics" while naming nobody;
- `verified` — the PUBLISHED member is global top-1 under at least one metric
  (`metric_top1` records which). A rank-1 win sitting on a DIFFERENT pool
  member does not lift this row: it stays evidence in
  `ru_top_target_bodyId`. That published-partner rule is the second r17/r18
  correction — ∃-over-pool on the ru side alone moved 2 of r18's 223 rows;
- `borderline` — the pool's best member in the global top-`rank_top_k` (5);
- `unmatched` — outside the window; `skipped` — no/RARE-profile.

The same Jaccard-first convention now governs every bodyId-level surface,
including the cross-dataset ones: `ProfileComparator.direct_comparison` sorts
its result frames `jaccard` → `rank_union`, `find_homologs` defaults to
`similarity_metric='jaccard'`, and `ui/config.SIMILARITY_METRICS` lists
Jaccard first. Type/pooled ranking keeps `rank_union` (J5).

**1:1 assignment**: mutual-best greedy within the pool, confident verdicts
only, on the bodyId ordering
contest losers flow into gap fill.

## 4. The category model — one partition per branch

This section is normative: the implementation must follow it exactly. The
categories are a **partition** (every in-scope neuron gets exactly one)
defined by an ordered first-match. The full derivation and proofs live in
`_plan/plan-mapping-validation-rev311-chain-aware-pool-widening.md` §2b
(S0–S8); this is the operational summary.

### 4.1 Vocabulary

- **query** — the set of source types; each has a bodyId population
  (e.g. the FAFB `circadian_clock` query is 21 source types / 242 bodyIds).
- **in-map targets `IM`** — the target bodyIds the auto type mapping
  claims for the query: the **union of the refined branch pools** (204 in
  the circadian_clock example; `set_coverage.json` `in_branch_pool`), so
  `IM = ∪ IM_b`. **`IMT`** is their set of types (40 in the example).
  Distinct from IM: the 40 in-map types' full male-cns **populations**
  total 219 bodyIds (`mapped_target_set`) — the 15 unclaimed ones are
  bodyId-level out-of-map instances of in-map types (the `(out-map)` /
  `family` material), not members of IM.
- **`IM_b`** — `IM` restricted to branch `b`. A branch is one
  `(source_type, target_type, linker)` path, so a source type's `IM_b`
  values are partitioned across its branches.
- **expansion** — out-of-map suspects surfaced by the homolog finding.
  They may be in-map targets of another branch (`sibling`), out-map
  bodyIds of an in-map type (`family`), out-of-type suspects
  (`candidates`), or type-mates of those suspects (`relative`).
- **admitted** — connectivity-qualified **and** morph-qualified.
  Connectivity-qualified = an invader (ranked ahead of the branch pool
  best) or a gap fire.

### 4.2 The per-branch rule (ordered first-match; first true wins)

| # | category | criterion | mode |
| --- | --- | --- | --- |
| 1 | `matched` / `verified` / `borderline` / `unmatched` | `t ∈ IM_b` (nested tier; `unmatched` is the else) | all |
| 2 | `sibling` | `t ∈ IM \ IM_b` AND admitted | all |
| 3 | `candidates` | `t ∉ IM` AND connectivity-qualified (invader ∪ gap fire ∪, family+, the top-`rank_top_k` discovery window) AND morph-qualified | all |
| 4 | `family` | `type(t) == T_b` (THIS branch's target type) AND `t ∉ IM` AND not already labeled | family+ |
| 5 | `relative` | `type(t)` is a candidate type of THIS branch AND `type(t) ∉ IMT` AND not already labeled | family+ |
| 6 | `examinees` | deep-window AND morph-qualified AND not already labeled | aggressive |

**Totality and exclusivity** follow from the order: each criterion is
evaluated on the complement of the earlier ones, so the bins are pairwise
disjoint; and the last applicable bin is always a residual (`unmatched`
for the tier, `candidates` in restrictive, `family`/`relative`/`examinees`
as "not already labeled" residuals), so nothing is unlabelled. The
structural fact that matters most: `sibling ⇒ t ∈ IM` while
`candidates/family/relative ⇒ t ∉ IM`, so sibling and candidates can never
both match; and `candidates` precedes `family`, so a same-type out-map
neuron that clears the invader/gap bar is a candidate, never both.

### 4.3 What each category means

- **`matched` / `verified` / `borderline` / `unmatched`** — the validated
  in-map targets of *this* branch. `matched` = top-1 for some source with
  `rank_union > matched_ru_min` (0.1), the only asserted tier; `verified` =
  top-1 of one metric or the ordered top-N (default 2) all in-pool;
  `borderline` = ≤ `invader_borderline_max` (3) invaders ahead;
  `unmatched` = the else. There is **no `skipped` here** — `unmatched` is
  the final else. (`skipped` exists only on the source side, §3.)
- **`sibling`** — an in-map target of the *query* that belongs to a
  different branch but enters this branch's expansion. It is already in
  the map, so it is **never** a gap-fill candidate. Requires admission.
  Same-parent and cross-parent cases are treated alike.
- **`candidates`** — a suspected neuron **outside** `IM` that is both
  connectivity-qualified (invader or gap fire) and morph-qualified. This is
  the restrictive fill material. It carries a bodyId-level annotation
  (§4.4). A same-type out-map neuron that reaches this bar is a candidate;
  one that does not is `family`.
- **`family`** — every out-map bodyId whose type is **this branch's own
  target type** (`type(t) == T_b`; branch-local, per the user's ruling).
  Gated by **type membership only**, not by qualification; bounded by the
  in-map types' populations. In family and aggressive modes it is the
  residual of the already-labeled set. The union over branches covers
  every out-map bodyId of the in-map types.
- **`relative`** — the type-mates of this branch's candidate types,
  restricted to types **outside** the map (`type(t) ∉ IMT`). Same
  conceptual level as `family`; the `∉ IMT` clause makes the two
  **disjoint by type** — an in-map type's mates go to `family`, an
  out-of-map type's mates go to `relative`. Ungated by qualification.
- **`examinees`** — the aggressive-only deep window (renamed from
  `suspicious` 2026-09-18: the mapper's rival-suspects concept now owns
  that word): out-of-pool homologs
  ranked **below** the pool best, within `candidate_window` (25) per
  metric and `deep_cap` (10) per source **per band** — the borderline band
  and the deep band draw on budgets of their own, so widening the mode can
  only ADD rows. They used to share one budget, and the wide band spent it
  before the second metric was ever read: measured 2026-09-24 on
  `circadian_clock`, that displaced 110 of 189 borderline pairs on male-cns
  and 262 of 408 on BANC, and their `relative` rows and layered gap-fill
  entries with them (153 rows → 63), so the wider question reported LESS
  evidence. Rows inside `rank_top_k` (5) are tagged
  `candidate_source='top_window'` instead, are connectivity-qualified, and
  so land in `candidates` — the two bands are exactly the boundary between
  "advisory" and "fill material", and the tag travels with the row's own
  rank, never with the mode (modes nest).

### 4.4 BodyId-level annotation and legend suffixes

A category is never split into more categories by annotation; the
annotation is a **leaf token** on the bin, not the root (the four
expansion roots are bare, §8). It is applied to **every** expansion bin —
`candidates`, `relative`, `examinees`, `family` — and is a bodyId leaf
suffix in the scene (`{bodyId}_{token}_{side}`).

It is **one ordered, mutually exclusive value**:

| token | level | meaning | example leaf |
| --- | --- | --- | --- |
| `{target_type}(out-map)` | **bodyId** | the neuron's **type is one of the mapping's in-map types** — an unmapped bodyId of a type already in the map. This is the fill material (every `family` member, plus such a bodyId surfacing in another branch). It wins over the two type-level tokens below. | `65631_SMP220(out-map)_R` |
| `{target_type}>{src1_etc}` | type | the type is **not** in-map but backward-maps to a real source population | `75596_CB3252>s-CPDN3B_R` |
| `{target_type}(no_source)` | type | the type is **not** in-map and has no usable backward route (hollow/absent home, e.g. CB4091) | `58554_CB4091(no_source)_L` |
| `untyped` | — | no type annotation at all | `32796_untyped_R` |

The distinction is deliberate: `(out-map)` is a **bodyId-level** statement
*about a type the query already maps* ("this instance is missing"), whereas
`>{src}` and `(no_source)` are **type-level** statements about a foreign
type ("the type sits in another source's territory" / "the type is out of
the map"). The in-map-type check therefore short-circuits: an in-map type
is never described by a type-level token, even when that type also maps
backward. `(dup)` remains a **standalone** trailing tag on any expansion
leaf.

### 4.5 Mode nesting

The label of an admitted neuron is **independent of the mode**; modes
differ only in which expansion neurons are admitted. Hence, per branch:
**restrictive ⊆ family ⊆ aggressive**, with shared neurons keeping the
identical category, topology included. Restrictive admits the tier,
`sibling`, and `candidates`; family adds the discovery window and
`family` and `relative`; aggressive adds `examinees`. Two inspected
interactions: a same-type out-map neuron in the **deep band** (below
`rank_top_k`) lands in `family` (rule 4 precedes rule 6 — inside the
window it is connectivity-qualified, so rule 3 takes it as a
`candidate`), and a gap-fired target ranked below the pool best is a
`candidate` (rule 3 precedes rule 6).

### 4.5a Pooling: the parallel engine (`--mode pooling`)

`pooling` is deliberately absent from `VALIDATION_MODES` / `MODE_RANK`, so
every `mode_at_least` comparison keeps answering "not at all" for it and
adding a rung cannot silently make the nested bins admit pooling rows. A
pooling run therefore reports `validation_mode: pooling` with **no
`mode_rank`** in `parameters.json`, and `normalize_mode` raises on
`pooling` combined with a legacy widening flag (it does not nest, so that
pair is a contradiction rather than a precedence question).

The engine (`comparison/mapping_validation_pooling.py`, stage `P`, runs
after the category partition and before the out-map expansion) answers a
different question. The nested modes ask *what else belongs to the pool the
type mapper already asserted*; pooling asks *what do connectivity and
morphology say is a homolog of the queried population, over the whole
opposite universe*, and compares that answer with the mapper afterwards.
The unsupervised property is the invariant, and it is enforced by a test
that makes the claim lookups raise:

- **seed** — every neuron the query names by the SOURCE dataset's own
  annotation (`_bodyids_for`), so the residue the branches do not claim is
  inside it. A branch-pool seed would make the mode supervised and is not
  offered.
- **gate** — absolute, never pool-relative: `jaccard >` floor, `rank_union >`
  floor (0), and both metric ranks inside `window_mult` × the size of THAT
  SOURCE TYPE's queried population. The window scales to the quantity an
  unsupervised run knows; a branch's claimed pool is the supervised one.
  These floors are a VOLUME guard-rail, not a data-quality claim (user
  2026-09-23): they decide how wide the connectivity scan may open, and the
  configured numbers are exactly what gated the run. An earlier build fitted
  the Jaccard floor per (source, target) pair to the graded rows past runs
  recorded (`min(configured, q05 of that pair's `matched`/`verified`
  jaccards)`, read from a `cache/{target}/pooling/` evidence store before the
  scan); the fit was DELETED rather than tuned, because the measurement that
  motivated it — 0.10 rejecting 0 % of one pair's verified rows and 36.6 % of
  another's at 0.20 — describes how much of the pool survives, which is
  precisely the volume question the floor is allowed to answer, and nothing
  about whether a pair is a homolog.
- **morphology last, as a gate** — the Find-Homolog fast path (no NBLAST)
  with the branch-free persisted `mapping_ref` bar, applied to the
  connectivity survivors only, under a budget. A scored candidate BELOW its
  bar leaves the exported pool and the scene root (`morph_refused` →
  `morph.dropped_targets`); its rows stay in `pooling_candidates.csv`, so a
  refusal stays auditable instead of becoming a silent absence. Nothing else
  removes a target: a row that was not looked at says which of the three
  absences it is (`not-selected`, `not-attempted-cap`, `no-score`), because a
  blank would read as a rejection. A verdict publishes the pair it was made
  from, in four columns that name one another: `morph_bar_kind` is the binding
  rule (`native` / `track_a` / `null_bar`), `morph_bar` that rule's value, and
  the score applied to it is `morph_pool_ref` for a native row and
  `morph_similarity` otherwise. They were not always one another's: the bar
  column carried the per-source NULL bar under a `native` kind, which made 14
  of 41 scored rows on the 2026-09-24 male-cns run read as a score below its
  own bar with a ✓ beside it. The gate had graded the right pair; the export
  could not show it, and a report a reader cannot recompute from is not a
  record. The run publishes that as a record, not a
  ratio of convenience — `attempted` (what the budget allowed
  to be looked at) · `scored` (what came back with a value) · `qualified` ·
  `no_score` · `capped` · `gate_applied` · `dropped_targets` · `error` ·
  `warnings` — because `scored` can sit far below `attempted` (one BANC run
  recorded `attempted 8 / scored 8` while
  its own rows said 1 `scored` and 7 `no-score`), and a pool that
  was mostly unscored is a missing measurement rather than a morphologically
  cleared one. Whenever the two differ the caveat is stated from ONE builder
  (`_pooling_warning_line`) in the Pooling tab, the Log tab and
  `user_warning_notes.txt`. That recorded collapse had a cause, and the first
  explanation given for it was wrong: the pairs DO all score. The shared
  scorer's `mapping_ref` rule "a branch-pool member was fetched only to
  anchor a bar, so it is not a candidate" was deleting the verdict of every
  candidate the mapper also claims (35/41 rows on male-cns, 33/41 on BANC);
  `qualify_visualized_pairs(..., prune_pool_refs=False)` makes that
  caller-side choice explicit, and grading stays honest because
  `native_scores` never counts a candidate against itself.
- **mapper joined post-hoc** — `confirmed` / `type_miss` / `type_new` /
  `verified_only`, with `reading_notes` published beside them: no cell is a
  recall measure (the two engines seed from different sets), and a large
  `verified_only` is expected (pool membership vs global rank), not a
  false-positive count.
- **advisory only** — a pooling candidate never enters or leaves its pool
  because of the mapper or of a size column (`size_nm3` is published with its
  universe percentile, and nm³ medians differ by orders of magnitude across
  datasets, so it is a label, not a bar). Nor does its target's TYPE NAME
  corroborate anything: the type is the target dataset's own annotation,
  reported for review and for the `mapper_cell` comparison, and there is no
  cross-dataset agreement column (an earlier `targets_corroborated` join was
  deleted — pooling neglects the type name by design, so counting how many
  other datasets also ignored it in the same direction graded nothing).
- **surfaces** — the run's whole result reads in the report's **Pooling** tab
  (the gate as the volume statement it is, the four cells, the harvest by
  target type, the morph record including what the bar refused, then one row
  per pooled target); the tab names
  the STORES the cells were scored against (§P7a's `input_fingerprint`: git
  rev, target universe, mapper snapshot), because the scores
  and the comparison both read stores that can move, so a cell is only
  comparable with another run that read the same ones — and it says so when a
  run predates the fingerprint. In a scene it
  is the `pooling · {source type}` legend root (plum `#7b4173`), hosted by
  the parent group of the source that reached each target best — a pool row
  has no branch, so that type is its only scene address, and a type no branch
  group covers is named in the run log instead of rendering nothing. The UI's
  mode row offers a fourth **Pooling** button with its own gate card
  (`card-tmvev-pooling`), which is an entrance to the mode, not a widening
  of the ladder.

### 4.6 Query-level dedup

A bodyId can appear in several branches (N-to-1 gives one target type
several source parents). Export a bodyId-level deduplicated result with
precedence **`tier > sibling > candidates > family > relative`** (within
the tier `matched > verified > borderline > unmatched`). The precedence
serves the gap-fill accounting; the family category itself is reported
complete via `family_material` (set_coverage) and reconciled against the
dedup bins in the per-run report. A
`(dup)` flag marks **non-sibling** bodyIds labeled in more than one
branch — siblings are duplicated by definition and are never flagged. The
flag is written back onto every per-branch row (`dup` column) and
surfaces in the scene as a ` (dup)` suffix on the category root (e.g.
`candidates · CB4091 (no_source) (dup)`). The mapper already
deduplicates each branch's in-map set; the dedup pass is a safeguard.

### 4.6a Reciprocal homolog evidence (stage 5d, advisory)

The forward pipeline answers "which target does each source neuron
prefer?". **Stage 5d** asks the mirror question of the neurons the
expansion proposed: each member of the `candidates` / `family` /
`relative` bins — plus, by default, the UNMATCHED validated pool targets
(matched / verified / borderline are already mapped; the symmetric
forward score is their evidence) — is reverse-scanned against the
**whole SOURCE
universe** with the same homolog-finding scorer (§3's
`expanded_vector` / `scan_source`), and asked "which SOURCE neuron do you
prefer, and is it the one our branch claims?". It runs after
`finalize_categories` (where `family`/`relative` are first enumerated)
and before the coverage / layered-fill rollups, then rebuilds the
bodyId-unique dedup so the labels reach `gap_fill/gap_fill_levels.csv`.

Verdict vocabulary (`comparison/body_id_resolver.py`:
`BACKWARD_EVIDENCE_VALUES`, `BACKWARD_COLUMNS`, `blank_backward_fields`,
`classify_backward_scan`, `_own_type_hit`, `serialize_backward_topN`,
`THIN_SHARED_TYPE_COUNT`, `reverse_source_column`), over
`backward_evidence ∈ high | medium | low | not-checked`:

The three scanned grades measure how prominently hits of the claiming
branch's **OWN source type** rank in the reverse scan
(`_own_type_hit`, which also publishes the `backward_own_type_*` block) —
pure rank evidence: no score bar, no
pool-membership gate. Such a hit counts wherever it lives;
`backward_top1_in_branch` records the separate pool-membership fact as
context on the row, never as the verdict.

- **`high`** — such a hit is the **top-1** by `rank_union_rank` or by
  `jaccard_rank`.
- **`medium`** — such a hit sits within the **top-3** of either ranking.
- **`low`** — outside both top-3 windows, or nothing usable ranked at all.
  There is no separate "nothing found" state: a scan that ranked nothing
  grades `low`, so a `low` is a **graded negative, not silence**.
- **`not-checked`** — not scanned: the pass is off, the neuron was over the
  per-run / per-branch budget, it is an already-mapped (matched / verified /
  borderline) pool member rather than gap-fill material, or it has no
  usable profile (missing, untyped or skipped — that case is *never*
  reported as `low`, which claims the scan ran).
  `backward_scanned_at` records WHY (`disabled` / `cap` / `run` /
  `no_profile` / `error`), so an unchecked row is never read as a negative
  result.

**Two hard invariants, both deliberate.**

1. **Connectivity only** — no morphology is re-scored here: candidates
   are already morph-qualified, and family/relative members are
   morph-similar to the query or to those candidates. The pass counters
   state it (`morph = 'not evaluated (connectivity-only)'`).
2. **Advisory** — the pass never changes a `category`, any
   `counts_toward_*` flag, or a fill `level`; the `high / medium / low /
   type_gated / advice` ladder stays the morph-bar strength ordering
   (§6). The reverse fact rides `gap_fill/gap_fill_levels.csv`'s
   `evidence` column (as `backward_high` / `backward_medium` /
   `backward_low`) beside its own `backward_evidence` column — on the
   `family` / `relative` / `unmatched` rows, since a `candidates` row keeps
   its morph-bar-kind evidence and carries the grade in `backward_evidence`
   alone. The
   run's fill totals stay identical to a run without the pass. It is
   fail-open like the other advisory layers: a failure leaves the rows
   `not-checked`.

**The evidence base is published, not inferred.** `rank_union` ranks the
*union* of the two partner-type vectors and enters a type one side lacks as
0.0, so the value alone cannot say whether it was computed over 1 shared
partner type or 20 — and a high `rank_union` CAN rest on almost
nothing (of 819 random source pairs above 0.1, 209 rest on ≤ 3 shared types;
in the r9 run none of the 46 in-branch reverse hits did, median 16). The
scorer already counts both numbers, so every backward row carries
`backward_shared_type_count` / `backward_union_type_count` plus
`backward_thin_evidence` (True at ≤ `THIN_SHARED_TYPE_COUNT` = 3). Nothing
else moves: no score, no bar, no `backward_evidence` verdict, no fill total.
Padding short vectors with placeholder partners was measured and rejected —
per-neuron distinct pads leave cosine exactly unchanged, act as a crude
length penalty that demotes nothing (0 of those 819 fall below 0.1), and
shared-name pads inflate jaccard 0.30 → 0.80, while these two counts say the
same thing directly (plan §17 / §17b).

Knobs (all in `parameters.json`): `backward_evidence_enabled` (default
**False**; CLI `--backward-evidence`), `backward_top_n` (5, the number of
reverse hits serialized into `backward_topN`), `backward_max_neurons`
(300 — the per-run budget of dataset-scale scans),
`backward_per_branch_cap` (40), `backward_scan_pool_targets` (True; CLI
`--no-backward-pool-targets`), `skip_backward_pass` (False; CLI
`--skip-backward-pass`). One reverse scan costs what a forward source
scan costs (§10), so a bodyId is scanned **once for all branches** and
everything past the caps stays `not-checked`. The budget is spent in role
order — `candidates` / `family` / `relative` members first, the unmatched
pool targets only with what is left. (While the pool scan still included
matched / verified / borderline controls, an unordered 120-scan run spent
102 scans on them and capped out half of the three bins; since
2026-09-19 those controls are not scanned at all — the symmetric forward
score is their evidence.)

Side effect that matters for §4.6b: reverse-scanning pool members (the
unmatched ones since 2026-09-19) yields each scanned pool target's
**column ranked over the whole source
universe**, and that is fed back into `categorize_pool_sources` (new
`reverse_columns` parameter). Before stage 5d, `n_competitors` was
structurally 0 — a forward column can only ever contain the branch's own
sources — so `source-borderline` / `source-unmatched` were unreachable;
out-of-branch rivals above a source are now real.
`reverse_source_column` lists that column in **jaccard order** since
2026-09-20, matching the reciprocal list; the downstream top-1 judgements
re-sort internally by each metric and so are unaffected, but the column is
truncated to `max(verified_top_n + invader_borderline_max + 2, 10)` rows of
out-of-pool rivals, and *that* cut now keeps the jaccard-strongest rivals
rather than the rank_union-strongest. `n_competitors` counts can therefore
shift across this boundary — read `source-` statuses as same-run evidence,
not cross-run.

Precondition (P0, amended): a neuron's expanded vector must not depend on
which ROLE it plays. It does not — measured, not assumed: rescoring every
r9 + r11 pair straight from the cache parquet reproduced the stored forward
`rank_union` on 272/272 pairs (`<1e-12`), no scored pair involved a sub-k row,
and `score_one_candidate_fast(a, b) == score(b, a)` holds exactly. So
`_preflight_target_profiles` guarantees only that a profile EXISTS (missing
ids get built); rows whose `top_k_bodyid_used` is below the run's `top_k` are
**counted, not rebuilt** (`_below_k_cache_ids`, pre-flight stats key
`below_k`, and `source_vectors_below_k` in the stage-5d counters →
`set_coverage.json` + a Reciprocal-tab line). It is a SPARSITY statement:
3,897 of 139,255 FAFB and 5,265 of 176,422 MCNS rows hold fewer partner
types than k=25, and 81% / 87% of those sit at ≤ 5 because
`_process_connections` sets `k_used = min(max_k, n_rows)`. Rebuilding cannot
converge (r9 rebuilt 6,883, r10 rebuilt the same 3,897 again, ~2,300 s per
run for no change), which is why the refresh was cut back to a counter.

Exported: `expansion/backward_matches.csv` (one row per (branch, member)
with the reverse top-1, its metric values + ranks, the shared/union type
counts, `backward_n_out_of_branch`, the advisory size-caliber pair, the
`thin` flag, the `backward_own_type_*` block naming the branch-type hit the
grade rests on (bodyId, type, `backward_own_type_via`, its scores/ranks and
its own thin flag) and the serialized `backward_topN` neighbourhood listed
in jaccard order) plus the
`backward_*` columns riding
`validation/examinees.csv`, `validation/deep_candidates.csv`,
`expansion/family_candidates.csv`, `expansion/relatives.csv` (the full
`BACKWARD_COLUMNS`), `gap_fill/gap_fill_dedup.csv` (a 12-column rollup
subset: the grade, the top-1 triple, the shared count, `n_out_of_branch`,
the thin flag and the own-type explanation WITHOUT its scores, ranks or
union count), and `gap_fill/gap_fill_levels.csv` (`backward_evidence`
alone).
Surfaces: the report's **Reciprocal** tab (§9) and the scenes' leaf
suffixes (§8).

### 4.6b Backward source status (advisory)

The same bodyId-bodyId pair scores power a **column view**: for each pool
target, rank its sources; for each in-branch source, its best column
standing maps to `source-matched` (column-top-1 of its own row-best
target, pair ru > matched_ru_min), `source-verified` (column-top-1
elsewhere, or a column whose top-N sources are all in-branch),
`source-borderline` (≤ invader_max out-of-branch sources above), or
`source-unmatched`. `source-candidates` = OUT-OF-MAP sources (claimed by no branch) whose
best-ranked scan hits land in the branch pool AND pass the run null bar
— the D-B8 backward mirror of candidate admission, attributed to the
branch owning the pool and visualized as a scene root (RE-AIMED
2026-09-18, user option 2: the first implementation derived candidates
from sibling rows, which are other branches' query neurons by
construction; the route now scans the out-of-map sources inside the
out-map expansion and morph-checks the in-pool hits against the run
null bar). Exported as `source_candidates.csv` (cross-branch
convergence remains visible via the sibling category). **Advisory only** — the
targets remain the validated entities; statuses never gate and never
enter the dedup. Exported as `source_status.csv` +
`set_coverage.source.source_status`; rendered in the report's Backward tab
(plan `plan-backward-source-status.md`).  When stage 5d ran (§4.6a) its
reverse scans feed these columns (`categorize_pool_sources(reverse_columns=…)`):
a forward-only column can hold just the branch's own sources, so
`n_competitors` is structurally 0 there and `source-borderline` /
`source-unmatched` become reachable only once the whole-source-universe
columns are in play.

### 4.6c Same-name-first consumers (advisory)

The mapper's same-name-first rule (within a fan-out, the candidate
carrying the source's own base name is SELECTED; the rivals become
disclosure-only suspects — `plan-samename-first-fanout-resolution.md`
§0.0) reshapes what this pipeline receives:

- **Fired selections** arrive as ordinary `mapped` pairs. They are
  MARKED, never gated: `TypePair.same_name_first` carries
  `{selected, rivals, path, disposition}`; validation rows and pair
  summaries gain the advisory `same_name_first` / `same_name_rivals`
  columns; `set_coverage.json` gains `same_name_first_pairs` /
  `same_name_first_types`; the report marks such pairs ⟡ in Branches
  and adds a Coverage-tab accounting card (also derived from
  `same_name_excluded.csv` so a no-pair run renders it too).
- **Held / evidence-only fan-outs** stay excluded (fail-closed), but are
  now ACCOUNTED: the improved log line names the disposition and the
  suspects CSV; `same_name_excluded.csv` records one row per type;
  `set_coverage.json` gains `same_name_first_held` /
  `same_name_first_excluded`.
- **Suspects verification (OPT-IN, `--verify-suspects`, default OFF)**:
  each rival of every queried same-name fan-out (fired, held, or
  excluded) is validated against its own target pool with the ordinary
  tier machinery, using the same scan frames (fired types) or a fresh
  scan pass (held/excluded types). Rows land ONLY in
  `suspects_verification.csv` and the report's Suspects tab — never in
  the validation counts, fills, dedup, or scenes. Advisory: a rival that
  verifies well is a candidate annotation, not a mapping; inclusion goes
  through the custom label mapper.
- **Multi-value type cells** (comma-joined `type` annotations, kept
  atomic by the mapper) are accounted, never split:
  `multivalue_types` / `multivalue_target_types` counters, rows in
  `same_name_excluded.csv` under `reason='multivalue_cell'`.
- A fired decision in the REVERSE direction also improves the backward
  home-reality check (`_backward_decision`): a type whose reverse
  mapping used to be `conflict` can now be `mapped` (plan
  `plan-tmvev-samename-first-consumers.md`).

### 4.7 Gap-fill semantics

The fill is a **process**, not a deterministic result: the members are
ranked suggestions for the user to judge. Restrictive fill = `candidates`
only (connectivity **and** morph qualified) — the calibrated number.
Family fill = `candidates + family + relative`, an over-inclusive superset
that may exceed the numeric gap.

### 4.8 Out-of-scope rows (connectivity-only, morph-failed)

The partition §4.2 is over **in-scope** rows: the in-map targets plus the
expansion suspects the mode admits, where admission requires **both**
connectivity-qualification and morph-qualification. A
connectivity-qualified suspect that fails the morph rule is **not a
category** — it is out of scope. It is still **exported** in
`examinees.csv` (renamed from `suspicious_candidates.csv`) with
`in_scope=False`, `morph_failed=True`,
`category=''`, because it is exactly what a **connectivity-only homolog
search** returns, so the validation run can be reconciled against one.
Out-of-scope rows are never rendered. Note the in-scope boundary can be
mode-relative: a morph-failed same-type residue is out of scope in
restrictive mode but is `family` in family mode (rule 4 is ungated by
qualification), which does not break the mode nesting (§4.5).

## 5. Noise gates (rows failing them move to
`noise_filtered_candidates.csv` with `noise_reason`)

In order — **spatial caliber is PRIMARY, connectivity heuristics
secondary** (user directive: size/arborization are
annotation-independent physical facts):

1. **Spatial caliber** (both metrics): candidate `size` (neuron-table
   metadata; MCNS `size`, FAFB `size_nm`; ~100% coverage) below
   `target_min_size_ratio` (0.1) × the branch pool's best → dropped.
   VALIDATED on production data: the universe ratio distribution is
   p01=0.018 / p05=0.090 / median 0.505; the known fragments measure
   0.004–0.025 (universe p0.1–p1.5), the smallest real pool member
   0.605 (p57) — the 0.1 cutoff sits at universe p5.5 in a sparse zone
   with 20×/6× margins. Fallback caliber: expanded-vector total weight.
2. **Positivity** (rank_union rows): a non-positive ru win is ordering
   noise on a weak profile.
3. **Jaccard sanity**: ru win with jaccard < 0.5 × the best pool
   member's jaccard.
4. **Tie margin**: ru win by < `suspicious_ru_margin` (0.02) over the
   pool best is a numerical tie (counter `suspicious_tie_filtered`).

The absolute prefilter (`target_min_weight` ≥ 10,
`target_min_partner_types` ≥ 2) excludes sub-threshold profiles at
target-vector build time.

## 6. Gap fill and expansion scope (Revisions 3.8–3.12)

- `gap = min(|P_S|, |P_T|) − M`; stats are informational
  (`gap`/`gap_ratio`/`gap_triggered` in `pair_summary.csv`).
- **Hemisphere-asymmetry trigger**: every neuron has a hemisphere
  identity, so pool neurons should be L/R symmetric — an `L != R`
  imbalance in either pool fires the gap even at arithmetic gap 0
  (`hemisphere` block in `pair_summary.csv`).
- **Both physical gates read the neuron table by DATASET-SPIELING**, so the
  column keys are matched on an alphanumeric-normalized name and read by
  label: MCNS `size`, FAFB `size_nm`, BANC v888 `Volume (nm^3)` for caliber;
  `hemisphere`/`sides`/`side` for sides, plus BANC's `Soma side`
  ('left'/'right', 92% of rows).  Before that, a dataset whose columns were
  spelled differently got `{}` / all-`'?'` back and BOTH gates ran silently
  disabled — and `itertuples` + `getattr` could not have addressed a spaced
  column name at all.  Cross-dataset medians differ by orders of magnitude
  (BANC 4.6e9 vs MCNS 1.8e8 nm³) and that is harmless: the caliber test is a
  ratio inside one dataset.
- **Modes** (one nested enum `restrictive < family < aggressive`; §4.5):
  `pooling` is a fourth CLI value and is NOT a rung of this enum (§4.5a).
  - **restrictive** (default): tier + `sibling` + `candidates`
    (invaders ∪ gap fires, morph-qualified). Minimal expansion.
  - **family**: adds the **candidate-discovery window** — the retained
    top-`rank_top_k` (5) of EACH metric, per source, on its own
    `deep_cap` (10) budget, spatial-caliber gated — which is connectivity
    evidence and so admits `candidates` (rule 3), and adds `family` (all out-map bodyIds of
    in-map types) and `relative` (candidate-type mates outside the map).
    The last two are ungated by qualification and bounded by type
    membership.
    Why the window is not the invader bar: an invader must beat the pool's
    BEST position on a metric, and a well-validated pool holds the global
    rank 1 on both (r18: jaccard rank 1 published on 190 of 223 rows), so
    that window is empty exactly where the branch looks strongest.
    Candidate TYPES ride on the feed and rule 5 seeds `relative` from them,
    so r16 → r18 the better a branch validated the less it reported:
    examinees 168 → 118, candidate types 5 → 3 (CB4091, SMP223 dropped),
    `relatives.csv` 39 rows → 1. Reading the top-k of both metrics is what
    stops one metric's top-1 hiding the other's candidates.
  - **aggressive**: widens that window to `candidate_window` (25) and adds
    the **deep-window** `examinees` rows for the band beyond `rank_top_k` —
    out-of-pool neurons ranked BELOW the pool best, exported to
    `deep_candidates.csv`. Rolled back from default after the
    r36e review (pulls in other clock-neuron types plus R1-R6/marginal
    neurons).
- **Fill accounting** (restrictive): only `candidates` count — they are
  connectivity- and morph-qualified. `sibling` never counts (already in
  the map); `family`/`relative` never count in restrictive mode.
- **Set-level coverage** (`set_coverage.json` + report §1): the
  deliverable for "how much of the mapping is validated/proposed/
  missing" — FAFB rollup (assigned / fill-proposed / unpaired) over the
  queried population, MCNS rollup over the mapped target set (in-pool by
  best tier, reached-as-candidates-only, and the explicit **holes** list =
  family material (out-map bodyIds of in-map types) claimed by no branch
  pool, no counted proposal, AND no morph-qualified `candidates` row — an
  invader-surfaced candidate is
  a claim, so it closes the hole).
  Out-of-pool candidates pass the same spatial-caliber gate.
  Out-of-map candidates are enumerated in TWO exports —
  `examinees.csv` (category `candidates`) and as proposal
  rows in `gap_fill_proposals.csv`; `gap_fill_dedup.csv` is the
  per-bodyId rollup.  `set_coverage.json` also lists `family_material`:
  the in-map-type bodyIds no branch pool claims (the population
  overhang of the claim set — plan-same-name-fidelity-and-three-level-
  coverage.md).
- **Out-map expansion (Plan I follow-up)**: every **unclaimed source
  neuron** (the scene's `out-map query` branch — annotated bodyIds no
  branch pool claims, i.e. the pool-refinement residue) is scanned
  against the full target universe and its top `out_map_top_k` (default
  10) **typed** targets **outside the in-map claims** are exported to
  `out_map_expansion.csv` (untyped/orphan neurons are filtered — they
  can never enter the mapping and otherwise dominate the top ranks of
  saturated types) —
  connectivity-only evidence (no morph bars), exploratory, never fills.
  The kept pairs are then **morph-checked** (Track-A vs the run null
  bar — `morph_v2_similarity` / `morph_qualified` columns; present only
  when the run null bar was calibrated — with `--no-morphology` or a
  thin null sample the columns are absent): failing rows
  (photoreceptor/orphan captures) stay in the CSV; the scene renders
  morph-passing targets beside their source neurons as
  `out-map candidates · {type}`.
- All proposals are **evidence only** — the mapping is never rewritten.
- **Fill accounting is QUERY-SCOPE-RELATIVE**: a cross-type candidate in
  an s-CPDN3C branch is not counted toward that branch's gap when the
  neuron's own type (`s-CPDN3D`) is also queried — its own branch
  accounts for it (as sibling/tier). Querying the family gives the
  complete accounting; single-type queries report scope-relative fills
  plus expansion advice.

- **Stage-2 profile pre-flight (Plan I)**: before the target-vector build,
  the run checks the target dataset's typed universe (one neuron-table read;
  type present and not `Unknown`, non-neuron statuses excluded) against the
  bodyId profile cache and **builds the missing profiles through the same
  profiler backend** (cache-first per bid, batch-save every 100, one
  consolidation; ~125 neurons/s measured on banc_v888).  Progress is
  machine-readable: the run folder carries **`pipeline_progress.jsonl`**
  (`ts/event/stage/done/total/pct` events — `run_start`, `stage_start`,
  `profiles_progress`, `stage_done`, `warning`, `run_done`), the contract a
  future UI tails.  Every stage the run opens now also closes: the
  `stage_start`/`stage_done` pairs are `1` resolve, `2` scans (with one
  `scan_progress` line per source type), `5` morphology, `3` categories,
  `5d` backward evidence, `3b` coverage accounting, `expansion` out-map,
  `4` scenes and `6` report, each carrying a `label` the Log tab prints
  with its duration.  Before that, a 25-minute block sat inside one
  never-closed stage and cost could only be attributed from artifact
  mtimes.  `--skip-profile-build` restores the historical
  cache-only, fail-closed behavior.
- **Layered gap-fill report**: `gap_fill_levels.csv` — one row per
  non-tier, non-sibling bodyId (siblings are claims, not fill proposals,
  so they are excluded from the report) with its confidence `level`: `high` (native m+v floor) /
  `medium` (Track-A backup `B_b − Δ`) / `low` (run null bar) for
  candidates, `type_gated` (family), `advice` (relative); hole-closing
  candidates are annotated. `set_coverage.json` mirrors the levels in
  `gap_fill_by_level`. The level stays the morph-bar strength ordering:
  stage 5d's reverse label rides the row's `evidence` /
  `backward_evidence` columns, never the level (§4.6a).
- **Sibling layers start hidden**: in family/aggressive scenes the
  `sibling` group renders with its legend row present but the traces off —
  one eye click restores them (siblings are in-map members already shown
  by their own branches; the expansion roots stay the focus).

## 7. Morphology — two tracks, one binding rule (Rev 3.7)

Both tracks score in the **target dataset's coordinates**; the Stage-4
scenes render in the **source** coordinates (targets bridged into the
source template). The frame difference is disclosed in
`morphology_calibration.json` (`score_frame`) and the report's Morph tab.

- **Track A (cross, query-based)**: the source skeletons are
  transformed into the target render space
  (FLYWIRE → JRCFIB2022M via `enrich_homolog_results` internals) and
  scored per (source, target) pair: `morph_v2_similarity` (production
  vector_v2, ZCA-whitened per-block cosine, identical to Find Similar)
  and `morph_nblast` (forward, normalized). Scored pairs: assigned
  verdicts, expansion rows (`candidates`/`examinees`, ≤
  `candidate_morph_cap` = 20/source), deep-window rows, pooled fills,
  and pool pairs.
- **Track B (all-native, pool reference)**: the branch's reference set
  — **matched+verified whenever the two together have ≥ 2 members;
  verified-only otherwise** (`pool_ref_tier`: `matched+verified` |
  `verified-only` | `verified (single ref)`) — is scored against every
  expansion candidate **natively** (dataset-native v2 vector
  cache + native whitener; zero transforms). Exports
  `morph_pool_ref` (max) / `morph_pool_ref_mean`.

**Both tracks read geometry from disk, and the run says what it could not get.**
Track A renders each target out of the raw-skeleton cache
(`cache/<target>/skeletons/raw_skeletons`) and Track B out of that dataset's
v2 vector cache, while the fetching used to happen only in stage 4 — **after**
stage 5, so a cold dataset scored nothing (`n_scored_pool: 0` on every branch,
`track_a_null_n: 0`, every bar the `null` kind) while looking like an ordinary
thin-sample fallback. `_preflight_target_skeletons` now fetches exactly the
targets this run's pair frame asks for, before scoring: cache-first, resumable,
emitting `skeletons_progress`, bounded by `MORPH_SKELETON_PREFLIGHT_CAP = 2000`
and fail-open, and it names what it fetched, what it missed, and what the cap
left unscored (`[stage 5] skeleton pre-flight (ds): X/Y cached, fetching Z`). A wide
aggressive frame can ask for more targets than a fetch budget should cover, so
the truncation is logged rather than scoring less quietly. `run_morphology`
additionally records `morph_pairs_requested` / `morph_pairs_scored` in
`morphology_calibration.json` and logs
`[stage 5] ! morphology scored 0 of N requested pairs` when the pass is empty
(`morph_coverage_warning`). Measured on hemibrain: 0/77 without the pre-flight;
19/77 for the identical invocation once that run's scenes had cached 19
skeletons; **75/77** with the pre-flight (54/56 fetched), which also derived
the render-space artifacts and produced a real null bar (`p95 = 0.319, n = 55`)
where every earlier hemibrain run had `n = 0`. Dataset names are normalized at
the NeuPrint boundary (`flywire_ids.neuprint_dataset_name`: the local folder
spelling `hemibrain_v1_2_1` is not a dataset the server knows, and it resolves
happily against every local table, which is why r25 lost all its target
neurons silently). The pair frame asks each gap-fill proposal for its pair
through `fill_pair`, because a proposal is spelled from the SIDE it fills
(`bodyId` is a source neuron on a `source` row and a target neuron on a
`target` row): reading `bodyId` / `proposal_bodyId` in one fixed order sent
every target-side proposal to Track A with the datasets swapped, which could
not score (0 of 76 in the male-cns family baselines, against 87 of 88
source-side rows) and cost one doomed skeleton fetch per swapped row. With the
orientation fixed the same hemibrain pair scores **76 of 76** requested pairs
with the null bar unchanged (`p95 = 0.319, n = 55`), and two consecutive runs
are byte-equal on every graded value.

**Qualification rule v3 — one bar engine, two currencies**
(`comparison/morph_bars.py`; `morph_qualified` / `morph_qualified_suspicious`,
used by the category classifier and the scene): each branch gets a
`BarSet` in `morphology_calibration.json` (`branch_bars` / `bar_params`),
and every admission compares the currency its bar kind names.

- **candidate bar** (restrictive and aggressive):
  1. **native floor, BINDING** when the branch's matched+verified
     reference tier has ≥ 2 scored members: `pool_ref ≥ mean pairwise
     reference self-similarity − pool_ref_floor_margin`. Track A never
     overrides it (the T1 leak: query morph 0.226 ≥ 0.176 but pool-ref
     0.013 vs floor 0.83).
  2. else the **Track-A backup floor**: `morph_v2 ≥ B_b −
     morph_track_a_offset`, where `B_b` is the mean Track-A morph of the
     branch's scored m+v pool pairs — a calibrated replacement for the
     retired arbitrary `0.25 × pooled average`.
  3. else the **run null bar** (`null_p95` of the Track-A morph over
     jaccard ≤ 0.05 window rows; `morph_track_a_offset` applies as an
     offset there too via the config default). The sample excludes only
     what is MODE-INVARIANT — the invader feed and the pool — so one dataset
     pair yields one backdrop in every mode. It used to exclude the mode's
     own candidate window as well, which moved the p95 across the ladder
     (0.143561 / 0.143123 / 0.150580, `circadian_clock` → male-cns
     2026-09-24) and re-graded any row sitting inside that drift: one BANC
     pair holds `morph_v2` 0.2642521 in all three runs and read `candidates`
     / out of scope / `candidates` as its bar moved. A near-zero-connectivity
     row that a window also reaches belongs in the backdrop by the stated
     criterion, so the old exclusion bought nothing and cost the invariant.
- **suspicious bar** (aggressive deep window only):
  `morph_v2 ≥ B_b − morph_suspicious_level × morph_track_a_offset`
  (defaults k = 3), falling back to the null **p50** when the branch has
  no scored pool pairs. A deep row passing the candidate bar is
  admitted by the normal rules; between the bars it is `examinees`;
  below the suspicious bar it stays out of scope (`morph_failed`).
  Invader rows failing the candidate bar stay out of scope in every
  mode.
- `bar_kind` ('native' | 'track_a_backup' | 'null' | 'track_a_suspicious'
  | 'null_lo') travels on every evidence row and the calibration so an
  admission is always reconstructible. The per-run null bar no longer
  gates candidates on branches with a pool baseline — the 80536
  scope-flip class is eliminated (branch-local determinism).

**AUC guard rail (unchanged)**: the Track-A score gates
`verified_strong` (demotes to `verified`) only when the run's
self-calibration AUC (verified_strong vs examinees) ≥ `morph_auc_floor`
(0.65); otherwise scores stay informational. Every run so far: AUC
0.52–0.59 → gate INACTIVE.

## 8. Scene bucketing and rendering (Stage 4)

One scene per parent mapping group, in the source template, with the
`drocatLegend` tree. The tree is built from the §4 categories (not by
parsing label prefixes); leaves, in order:

`max_scenes` defaults to **0 = one scene per parent**, because branch
review is the point of the run: the old default of 12 silently dropped the
smallest-pool parents, so a 43-branch / 21-parent circadian run shipped 9
parents with no scene at all. A positive value still clamps (largest
source pool first) and then names every dropped parent in the log, and the
report's Branches tab states `N of M parent types have a rendered scene`
and marks the affected rows — a dash must never be ambiguous between "the
cap dropped this" and "there was nothing to render".

```
branch: {source_type} → {target_type} · {linker_signature}
├── query · {source_type} (n)                  the query set
├── matched · {target_type} (n)                tier
├── verified · {target_type} (n)               tier
├── borderline · {target_type} (n)             tier
├── unmatched · {target_type} (n)              tier
├── sibling · {category} (n)                   sibling (no (dup) tag)
├── candidates (n)                             candidates
│   └── {bodyId}_{token}_{side} [ (dup)]
├── family (n)                                 family   [family/aggressive]
│   └── {bodyId}_{token}_{side} [ (dup)]
├── relative (n)                               relative [family/aggressive]
│   └── {bodyId}_{token}_{side} [ (dup)]
└── examinees (n)                             examinees [aggressive]
    └── {bodyId}_{token}_{side} [ (dup)]
```
Plus, per parent type, one extra branch group renders the **source-side
gap**: `out-map query · {source_type}` — the annotated FAFB neurons of
the type that received no assigned partner (never scanned, or scanned
without a verdict pair). It exists so the source gap can be compared
against `candidates` directly; skipped sources (no connectivity
profile) keep their separate `unassigned` layer.

The four expansion categories (`candidates` / `family` / `relative` /
`examinees`) are **ONE root each**; the bodyId-level detail rides on the
leaf:

- Every expansion leaf carries the ordered per-row token (§4.4):
  `{T}(out-map)` (the type is an in-map type — bodyId-level out-of-map),
  else `{T}>{src}` (foreign type with a real backward home), else
  `{T}(no_source)` (foreign type, no route), else `untyped`.
- `(dup)` is a **standalone trailing tag** (on any of the four) marking a
  bodyId that recurs across branches.
- On a stage-5d run, a scanned `candidates` / `family` / `relative` leaf
  additionally suffixes with its reciprocal grade: `· high` (the branch's
  own source type is the top-1 by rank_union or jaccard) / `· medium`
  (within a top-3 of either) / `· low` (outside both top-3 windows). A
  not-checked member keeps a **bare leaf** — absence is never drawn as a
  negative (§4.6a). The grade is pure rank evidence (user 2026-09-19): no
  score bar, no pool-membership gate — `backward_top1_in_branch` and the
  size-ratio caliber stay on the row as context.

Leaves inside a root are sorted by `type + suffix` (not by bodyId), and
each root always renders even when it holds a single leaf.

Any bin may carry the `untyped` suffix (§4.4). Bins are drawn only when
non-empty. `family`/`relative` appear only in family/aggressive mode;
`examinees` only in aggressive mode.
- Scenes render in the source dataset's RENDER template space (the
  visualization backend's template target): FAFB/BANC are their native
  frames, male-cns/hemibrain/manc are bridged native → render
  (`JRCFIB2022Mraw → JRCFIB2022M` etc.) — BOTH sides.  Sources and
  targets are always delivered in one frame; the 2026-09-18 MCNS→BANC
  misplacement (raw-space neurons on a nanometre mesh) was exactly this
  frame mismatch.

- **Every rendered member passes the morph rule** (default scope
  included); structural labels only name the bin. The one deliberate
  render-only omission: a structural member (`sibling`) that fails
  qualification keeps its CSV label but is not drawn.
- Colors (category-keyed): query blue, matched cyan, verified green,
  borderline gold, unmatched grey, sibling pink, candidates orange,
  family light green, relative olive, examinees red.
- The collapsible legend panel is content-width, shrinking for short
  labels and capping at 420px (or the viewport width, whichever is
  smaller); a custom horizontal scrollbar appears when a row overflows.
- Trace identity: `nrn.name = str(bodyId)` at load; the overlay
  metadata rows are reindexed to the layer's bid order (the R5
  61430/50274 mislabelling root cause); per-bodyId hover templates;
  optional `--scene-selfcheck` verifies each legend leaf's geometry
  against its neuron's bbox (undoing the FAFB tilt rotation before
  comparing).

## 8b. Modes — behavioral summary

| | restrictive (default) | family | aggressive |
| --- | --- | --- | --- |
| tier | linker-refined `IM_b` | identical | identical |
| sibling | yes | yes | yes |
| candidates | invaders ∪ gap fires | + top-`rank_top_k` window | same |
| family / relative | no | yes | yes |
| deep-window examinees | no | no | yes |
| nesting | ⊆ family | ⊆ aggressive | superset |

The mode is recorded in `parameters.json` (`validation_mode`). Modes are
one ordered enum; a shared neuron's category is identical across modes
(§4.5). `pooling` is a fourth `--mode` value and is NOT in that enum: it
runs a parallel unsupervised engine whose rows live in `pooling/` and never
enter the bins above (§4.5a).

## 9. Exports

The new taxonomy (§4) is exported additively: a primary `category` column
plus the annotations, with the legacy `invader_class`/`invader_label`
columns retained for compatibility. The table lists FILE NAMES; §9.1
gives the subfolder each one lives in.

| file | content |
| --- | --- |
| `mapping_export.csv` | per-bridge record: refined `{…}` pools, selected chain (`selected_bridge`) and the chain that named the source neurons (`source_bridge`), linkers, per-side pool basis, parent context |
| `validation_results.csv` | per source bodyId: verdict, ranks + scores, connectivity flags, `source_size`, per-source noise counters |
| `examinees.csv` (was `suspicious_candidates.csv`) | expansion rows with metrics, caliber columns (`ahead_size`, `pool_best_size`, `size_ratio`), `category` + `candidate_annotation` + `dup`, legacy classification columns, Track-A/B morph, `pool_ref_tier` |
| `noise_filtered_candidates.csv` | every gate-dropped row with `noise_reason` (spatial_caliber, tie_margin, negative_rank_union, jaccard_below_pool) |
| `deep_candidates.csv` | candidate-window rows ranked below the pool best, tagged `candidate_source`: `'top_window'` (the `rank_top_k` band, family+ — connectivity evidence, so rule 3 admits them as `candidates`) or `'deep_window'` (the aggressive-only wider band, the `examinees` bin) |
| `gap_fill_proposals.csv` | proposals with `fill_class` (in/out of pool), `category`, `counts_toward_restrictive_fill`, `counts_toward_family_fill` |
| `family_candidates.csv` | the whole `family` bin — enumerated members ∪ evidence rows classified `family`, per branch+bodyId (family/aggressive modes) |
| `backward_matches.csv` | stage 5d (`--backward-evidence` only): one row per (branch, scanned member) of `candidates`/`family`/`relative` (+ the UNMATCHED pool members) — `member_bodyId`/`member_type`/`member_category`/`scan_role`, then `backward_evidence` with the reverse top-1 (`backward_top1_source_bodyId`/`_type`/`_in_branch`), the two metric values + ranks, the evidence base (`backward_shared_type_count`/`_union_type_count`, `backward_thin_evidence` at ≤3 shared), the `backward_own_type_*` block (the branch-type hit the grade rests on, with `backward_own_type_via`), `backward_n_out_of_branch`, the caliber pair, and the serialized `backward_topN` neighbourhood in jaccard order; advisory, connectivity only (§4.6a) |
| `gap_fill_dedup.csv` | query-level bodyId dedup with `dedup_category` (precedence §4.6), `dup`, and on stage-5d runs the 12-column reciprocal rollup of the neuron's strongest branch (§4.6a) |
| `set_coverage.json` | set-level coverage in two ROLE-named blocks — `source` (assigned/proposed/unpaired rollup) and `target` (in-pool/candidates/holes per type) — labelled by the top-level `source_dataset`/`target_dataset`, plus `family_material` (in-map-type bodyIds no branch pool claims — the 219−204 population overhang) and `mapper_gap` (types with no backward mapping) |
| `relatives.csv` | the whole `relative` bin (type-mates of candidate types, ∪ evidence rows classified `relative`), per branch+bodyId |
| `pool_categories.csv` | tier + metrics + `size` per in-map target |
| `pair_summary.csv` | per branch: pools, best (the mutual-best 1:1 pair count; renamed from `matched`), gap (informational), verdict/noise counters, `pool_best_size`, and the provenance pair `selected_chain` / `source_chain` (the chain that resolved the target pool vs the one that named the source neurons — equal except under the per-side basis) |
| `pooling_candidates.csv` / `pooling_pool.csv` / `pooling_cross_validation.json` | `--mode pooling` only (§4.5a): every (source, target) pair that passed the absolute gate, deduplicated to one row per candidate target on the ordering chain, and the post-hoc comparison with the mapper's claim sets (`confirmed` / `type_miss` / `type_new` / `verified_only`, with the `reading_notes` that say which cells are not recall measures). `morph_gate` keeps the three absences apart (`not-selected` / `not-attempted-cap` / `no-score`), and only an explicit `scored` below the bar removes a target from `pooling_pool.csv` (its rows stay in `pooling_candidates.csv`). Each pooling row publishes the whole record of its verdict: `morph_bar_kind` names the binding rule, `morph_bar` is that rule's value, and the score it graded is `morph_pool_ref` for a native row and `morph_similarity` otherwise, so `morph_qualified` is recomputable from the row beside it (§4.5a). Its `gate` block publishes the configured floors with their stated role (`jaccard_floor`, `rank_union_floor`, `window_mult`, `role` = volume guard-rail); its `morph` block is the RECORD the last gate left behind (`attempted` / `scored` / `qualified` / `no_score` / `capped` / `gate_applied` / `dropped_targets` / `error` / `warnings` / `vector_cache` (`loaded` / `stale_dropped` / `saved` — the target-vector store's ledger: a stored 256-dim vector plus its hemisphere lets a repeat run skip 0.412 s of skeleton load, render transform and vectorization per neuron, and rows are keyed on their skeleton's `(mtime_ns, size)` so a healed skeleton is recomputed rather than reused — a cache that could re-grade a pair would be a different instrument, not a faster one), where `scored` ≤ `attempted` by construction — the ratio is published, and a shortfall reaches `user_warning_notes.txt`), and `input_fingerprint` names the stores the scores came from (git rev, target universe, mapper snapshot) so cells are only compared across runs that read the same ones. The store keeps its vectors at full precision and stamps `vector_dtype`, and `load()` refuses a file stamped otherwise as a unit: a `float32` store was measured re-grading a score in its 8th decimal, and a cache that changes a verdict is a different instrument rather than a faster one. The cold/warm pair on the frozen harness came out byte-identical (0 differing cells across both pooling CSVs, same 40-target pool and same single refusal) with stage `P` at 132 s cold and 56 s warm |
| `parameters.json` | every knob incl. `validation_mode`, cutoffs, the null-calibration knobs (`null_jaccard_max`, `null_per_source_cap`, `null_min_n`, `null_percentile`), `out_map_top_k`, the pooling knobs (`pooling_jaccard_floor`, `pooling_rank_union_floor`, `pooling_window_mult`, `pooling_morph_gate`, `pooling_max_morph_targets`), the stage-5d knobs (`backward_evidence_enabled`, `backward_top_n`, `backward_max_neurons`, `backward_per_branch_cap`, `backward_scan_pool_targets`, `skip_backward_pass`) + the `backward_evidence` counter block, and the stage skip flags |
| `morphology_calibration.json` | per-branch thresholds, `pool_ref_tier`/`baselines`/`floors`, `track_a_null_bar`/`n`, `score_frame`, AUC gate record |
| `report.html` | the per-run report: headline + three coverage levels (L1 claim / L2 provenance / L3 validation), branches (Mapped = the sources carrying a mapping verdict, with the mutual-best pair count beside it; the displayed `gap` is measured against Mapped, while `pair_summary.csv` keeps the stricter pair-based gap — hover either cell for both), fills (with the per-row `reciprocal` column, the headline count and a `Reverse evidence by bin` split — its own axis, never a level), the **Reciprocal** tab (stage 5d: one row per neuron, jaccard-ordered, the branch-type hit beside the rank_union top-1, top-N on hover), out-map expansion, backward source status, the **Pooling** tab (§4.5a: the gate as the volume guard-rail it is, the four comparison cells, the harvest by target type, the morph record including what the bar refused, and one row per pooled target — a pooling run's result reads nowhere else, because the nested tabs are empty by construction there), morphology record, scenes, file index (paths as THIS run wrote them, so a pre-layout folder lists no subfolders); hover-glossary on every term; `backward_progress` events timeline the pass in the Log tab; regenerable via `python -m comparison.mapping_validation_report <run_dir>` |
| `README.txt` | slim directions (what file is what) + full run log — the analysis content moved into `report.html` |
| `user_warning_notes.txt` | bracketed-tag warning lines appended by the report writer (self-check, null-sample, mapper-gap, `[reciprocal]` own-source-type top-3 counts) — the `[reciprocal]` line is also quoted verbatim in `report.html`'s Warnings section as a derived advisory |
| `visualization/*.html` | tree-legend scenes per parent group |

### 9.1 Run-folder layout

Since 2026-09-19 the evidence CSVs are grouped by pipeline stage; the run
root keeps only the deliverables and the parameter/meta surface
(`visualization/` was already a subfolder):

```
{run dir}/
├── report.html / README.txt / _UserGuide_please_read_me.*
├── parameters.json / set_coverage.json / morphology_calibration.json
├── pipeline_progress.jsonl / user_warning_notes.txt
├── validation/   validation_results · pair_summary · pool_categories
│                 · examinees · deep_candidates · noise_filtered_candidates
├── expansion/    family_candidates · relatives · out_map_expansion
│                 · source_candidates · source_status · backward_matches
├── gap_fill/     gap_fill_dedup · gap_fill_levels · gap_fill_proposals
├── mapping/      mapping_export · same_name_excluded · suspects_verification
├── pooling/      pooling_candidates · pooling_pool · pooling_cross_validation
│                 (`--mode pooling` only — absent from every other run)
└── visualization/plot-3d_{ABBREV}_branches_{query}_{ts}/*.html
```
(CSV names shown without their `.csv` suffix; every evidence file lives
in exactly ONE category subfolder, and the root holds ONLY the listed
deliverables + parameter/meta.)

`RUN_FILE_LAYOUT` + `run_file_path()` in `mapping_validation.py` are the
ONE registry: writers resolve with `create_parent=True`, readers fall back
to the pre-2026-09-19 FLAT path when only that exists, so an OLD run
folder still regenerates its report untouched (a filename used to be
hard-coded in six places). `utils/naming_utils.RUN_FOLDER_PREFIXES` lists
`type-map-validation` as a run-folder prefix (the storage inventory had
mistaken these folders for nested run folders), and `storage_inventory.py`
registers the tool as all-deliverable (no re-downloadable source data, so
nothing is prunable).

## 10. Performance (measured, FAFB → male-cns, warm caches)

| scope | wall time |
| --- | --- |
| APDN3 (12 sources, 4 branches) | ~130 s |
| s-CPDN3C/D (49 sources, 10 branches) | ~330 s default; ~550 s with the deep window; ~420 s with null calibration |
| circadian_clock (242→219, 43 branches, 21 scenes) | ~2,220 s |

Dominant costs: per-source global scans (~4–7 s/source scalar numpy),
Track-A NBLAST (~0.2–1 s/pair), scene rendering (~1 s/neuron).
Track B is near-free after the first vector-cache warm-up (computed
vectors persist in `cache/{ds}/find_similar/morphology/`).
Stage 5d costs about one forward source scan per member: the scorer and
the universe size are the same on both axes (the r8 reference run's
out-map expansion measured 207 s for 30 forward scans of the 172 k-neuron
MCNS universe), and its equally large source universe builds once in ≈ 5
s. That is why it is capped (`backward_max_neurons` /
`backward_per_branch_cap`) and why a neuron is scanned ONCE for all the
branches that claim it (§4.6a).

## 11. Known limitations

- Scalar scan (~3 k pairs/s); full-dataset validation wants the
  vectorized sparse-rank scorer.
- The MCNS↔FAFB landmark bridge renders matched pairs contralaterally
  (opposite x-to-side conventions; needs an anatomical ground-truth
  call — mirror MCNS x, fix the landmark registration, or document).
  OPEN.
- Track B's render-frame fallback (`_track_b_render_fallback`) exists for the
  vector-cache-miss case, but no test exercises it; no NBLAST variant
  (deferred).
- The WIDE candidate window (6..`candidate_window`) is scan-noise-dominated in
  the small-FAFB-type regime (cross-type neighbors outnumber same-type
  expansion ~25:1) — one more reason that band stays opt-in, and why
  family mode reads only the top-`rank_top_k` of it.
