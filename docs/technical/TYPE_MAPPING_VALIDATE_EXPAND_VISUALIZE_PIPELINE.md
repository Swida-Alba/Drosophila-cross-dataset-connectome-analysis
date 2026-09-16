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
2. **expand** — from the homolog finding, add the suspicious neurons the
   type mapping missed, cross-referenced against the mapping structure
   and the morphology (§4, §6);
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
- Ranks are competition-style (ties share the better rank, NaN last).
- Each source is scanned ONCE per run and the scan is shared by all its
  branches. Retention: top-100 per metric + pool (`FILL_KEEP_TOP`).

**Source verdicts** (tiered, on the best pool member):

- `verified_strong` — global top-1 under BOTH rank_union and jaccard
  (rank_union must be positive — positivity policy);
- `verified` — top-1 under one metric (recorded; disagreement flagged);
- `borderline` — best pool member in the global top-`rank_top_k` (5);
- `unmatched` — outside the window; `skipped` — no/RARE-profile.

**1:1 assignment**: mutual-best greedy within the pool
(rank_union rank → jaccard rank → ru score), confident verdicts only;
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
| 3 | `candidates` | `t ∉ IM` AND connectivity-qualified AND morph-qualified | all |
| 4 | `family` | `type(t) == T_b` (THIS branch's target type) AND `t ∉ IM` AND not already labeled | family+ |
| 5 | `relative` | `type(t)` is a candidate type of THIS branch AND `type(t) ∉ IMT` AND not already labeled | family+ |
| 6 | `suspicious` | deep-window AND morph-qualified AND not already labeled | aggressive |

**Totality and exclusivity** follow from the order: each criterion is
evaluated on the complement of the earlier ones, so the bins are pairwise
disjoint; and the last applicable bin is always a residual (`unmatched`
for the tier, `candidates` in restrictive, `family`/`relative`/`suspicious`
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
- **`suspicious`** — the aggressive-only deep window: out-of-pool homologs
  ranked **below** the pool best, within `candidate_window` (25) per
  metric and `deep_cap` (10) per source, morph-qualified, and neither an
  invader nor a gap fire.

### 4.4 BodyId-level annotation and legend suffixes

A category is never split into more categories by annotation; the
annotation is a **leaf token** on the bin, not the root (the four
expansion roots are bare, §8). It is applied to **every** expansion bin —
`candidates`, `relative`, `suspicious`, `family` — and is a bodyId leaf
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
`sibling`, and `candidates`; family adds `family` and `relative`;
aggressive adds `suspicious`. Two inspected interactions: a same-type
out-map neuron in the deep window lands in `family` (rule 4 precedes rule
6), and a gap-fired target ranked below the pool best is a `candidate`
(rule 3 precedes rule 6).

### 4.6 Query-level dedup

A bodyId can appear in several branches (N-to-1 gives one target type
several source parents). Export a bodyId-level deduplicated result with
precedence **`tier > sibling > candidates > family > relative`** (within
the tier `matched > verified > borderline > unmatched`). A `(dup)` flag
marks **non-sibling** bodyIds labeled in more than one branch — siblings
are duplicated by definition and are never flagged. The flag is written
back onto every per-branch row (`dup` column) and surfaces in the scene as
a ` (dup)` suffix on the category root (e.g. `candidates · CB4091
(no_source) (dup)`). The mapper already deduplicates each branch's in-map
set; the dedup pass is a safeguard.

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
`suspicious_candidates.csv` with `in_scope=False`, `morph_failed=True`,
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
- **Modes** (one nested enum `restrictive < family < aggressive`; §4.5):
  - **restrictive** (default): tier + `sibling` + `candidates`
    (invaders ∪ gap fires, morph-qualified). Minimal expansion.
  - **family**: adds `family` (all out-map bodyIds of in-map types) and
    `relative` (candidate-type mates outside the map). Both are ungated by
    qualification and bounded by type membership.
  - **aggressive**: adds the **deep-window** `suspicious` rows —
    out-of-pool neurons within the retained top-`candidate_window` (25)
    per metric ranked BELOW the pool best, `deep_cap` (10) per source,
    exported to `deep_candidates.csv`. Rolled back from default after the
    r36e review (pulls in other clock-neuron types plus R1-R6/marginal
    neurons).
- **Fill accounting** (restrictive): only `candidates` count — they are
  connectivity- and morph-qualified. `sibling` never counts (already in
  the map); `family`/`relative` never count in restrictive mode.
- **Set-level coverage** (`set_coverage.json` + README section): the
  deliverable for "how much of the mapping is validated/proposed/
  missing" — FAFB rollup (assigned / fill-proposed / unpaired) over the
  queried population, MCNS rollup over the mapped target set (in-pool by
  best tier, reached-as-candidates-only, and the explicit **holes** list =
  mapped-set neurons claimed by no branch pool, no counted proposal, AND
  no morph-qualified `candidates` row — an invader-surfaced candidate is
  a claim, so it closes the hole).
  Out-of-pool candidates pass the same spatial-caliber gate.
  Out-of-map candidates are enumerated in TWO exports —
  `suspicious_candidates.csv` (category `candidates`) and as proposal
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
  future UI tails.  `--skip-profile-build` restores the historical
  cache-only, fail-closed behavior.
- **Layered gap-fill report**: `gap_fill_levels.csv` — one row per
  non-tier, non-sibling bodyId (siblings are claims, not fill proposals,
  so they are excluded from the report) with its confidence `level`: `high` (native m+v floor) /
  `medium` (Track-A backup `B_b − Δ`) / `low` (run null bar) for
  candidates, `type_gated` (family), `advice` (relative); hole-closing
  candidates are annotated. `set_coverage.json` mirrors the levels in
  `gap_fill_by_level`.
- **Sibling layers start hidden**: in family/aggressive scenes the
  `sibling` group renders with its legend row present but the traces off —
  one eye click restores them (siblings are in-map members already shown
  by their own branches; the expansion roots stay the focus).

## 7. Morphology — two tracks, one binding rule (Rev 3.7)

Both tracks score in the **target dataset's coordinates**; the Stage-4
scenes render in the **source** coordinates (targets bridged into the
source template). The frame difference is disclosed in
`morphology_calibration.json` (`score_frame`) and the README.

- **Track A (cross, query-based)**: the source skeletons are
  transformed into the target render space
  (FLYWIRE → JRCFIB2022M via `enrich_homolog_results` internals) and
  scored per (source, target) pair: `morph_v2_similarity` (production
  vector_v2, ZCA-whitened per-block cosine, identical to Find Similar)
  and `morph_nblast` (forward, normalized). Scored pairs: assigned
  verdicts, expansion rows (`candidates`/`suspicious`, ≤
  `candidate_morph_cap` = 20/source), deep-window rows, pooled fills,
  and pool pairs.
- **Track B (all-native, pool reference)**: the branch's reference set
  — **matched+verified whenever the two together have ≥ 2 members;
  verified-only otherwise** (`pool_ref_tier`: `matched+verified` |
  `verified-only` | `verified (single ref)`) — is scored against every
  expansion candidate **natively** (dataset-native v2 vector
  cache + native whitener; zero transforms). Exports
  `morph_pool_ref` (max) / `morph_pool_ref_mean`.

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
     offset there too via the config default).
- **suspicious bar** (aggressive deep window only):
  `morph_v2 ≥ B_b − morph_suspicious_level × morph_track_a_offset`
  (defaults k = 3), falling back to the null **p50** when the branch has
  no scored pool pairs. A deep row passing the candidate bar is
  admitted by the normal rules; between the bars it is `suspicious`;
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
self-calibration AUC (verified_strong vs suspicious) ≥ `morph_auc_floor`
(0.65); otherwise scores stay informational. Every run so far: AUC
0.52–0.59 → gate INACTIVE.

## 8. Scene bucketing and rendering (Stage 4)

One scene per parent mapping group, in the source template, with the
`drocatLegend` tree. The tree is built from the §4 categories (not by
parsing label prefixes); leaves, in order:

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
└── suspicious (n)                             suspicious [aggressive]
    └── {bodyId}_{token}_{side} [ (dup)]
```
Plus, per parent type, one extra branch group renders the **source-side
gap**: `out-map query · {source_type}` — the annotated FAFB neurons of
the type that received no assigned partner (never scanned, or scanned
without a verdict pair). It exists so the source gap can be compared
against `candidates` directly; skipped sources (no connectivity
profile) keep their separate `unassigned` layer.

The four expansion categories (`candidates` / `family` / `relative` /
`suspicious`) are **ONE root each**; the bodyId-level detail rides on the
leaf:

- Every expansion leaf carries the ordered per-row token (§4.4):
  `{T}(out-map)` (the type is an in-map type — bodyId-level out-of-map),
  else `{T}>{src}` (foreign type with a real backward home), else
  `{T}(no_source)` (foreign type, no route), else `untyped`.
- `(dup)` is a **standalone trailing tag** (on any of the four) marking a
  bodyId that recurs across branches.

Leaves inside a root are sorted by `type + suffix` (not by bodyId), and
each root always renders even when it holds a single leaf.

Any bin may carry the `untyped` suffix (§4.4). Bins are drawn only when
non-empty. `family`/`relative` appear only in family/aggressive mode;
`suspicious` only in aggressive mode.

- **Every rendered member passes the morph rule** (default scope
  included); structural labels only name the bin. The one deliberate
  render-only omission: a structural member (`sibling`) that fails
  qualification keeps its CSV label but is not drawn.
- Colors (category-keyed): query blue, matched cyan, verified green,
  borderline gold, unmatched grey, sibling pink, candidates orange,
  family light green, relative olive, suspicious red.
- The collapsible legend panel is content-width, shrinking for short
  labels and capping at 420px (or the viewport width, whichever is
  smaller); a custom horizontal scrollbar appears when a row overflows.
- Trace identity: `nrn.name = str(bodyId)` at load; the overlay
  metadata rows are reindexed to the layer's bid order (the R5
  61430/50274 mislabelling root cause); per-bodyId hover templates;
  optional `--scene-selfcheck` verifies each legend leaf's geometry
  against its neuron's bbox (undoing the FAFB tilt rotation before
  comparing).

## 8b. Three modes — behavioral summary

| | restrictive (default) | family | aggressive |
| --- | --- | --- | --- |
| tier | linker-refined `IM_b` | identical | identical |
| sibling | yes | yes | yes |
| candidates | invaders ∪ gap fires | same | same |
| family / relative | no | yes | yes |
| deep-window suspicious | no | no | yes |
| nesting | ⊆ family | ⊆ aggressive | superset |

The mode is recorded in `parameters.json` (`validation_mode`). Modes are
one ordered enum; a shared neuron's category is identical across modes
(§4.5).

## 9. Exports

The new taxonomy (§4) is exported additively: a primary `category` column
plus the annotations, with the legacy `invader_class`/`invader_label`
columns retained for compatibility.

| file | content |
| --- | --- |
| `mapping_export.csv` | per-bridge record: refined `{…}` pools, selected chain, linkers, pool basis, parent context |
| `validation_results.csv` | per source bodyId: verdict, ranks + scores, connectivity flags, `source_size`, per-source noise counters |
| `suspicious_candidates.csv` | expansion rows with metrics, caliber columns (`ahead_size`, `pool_best_size`, `size_ratio`), `category` + `candidate_annotation` + `dup`, legacy classification columns, Track-A/B morph, `pool_ref_tier` |
| `noise_filtered_candidates.csv` | every gate-dropped row with `noise_reason` (spatial_caliber, tie_margin, negative_rank_union, jaccard_below_pool) |
| `deep_candidates.csv` | deep-window `suspicious` rows (aggressive only) with `candidate_source='deep_window'` |
| `gap_fill_proposals.csv` | proposals with `fill_class` (in/out of pool), `category`, `counts_toward_restrictive_fill`, `counts_toward_family_fill` |
| `family_candidates.csv` | the whole `family` bin — enumerated members ∪ evidence rows classified `family`, per branch+bodyId (family/aggressive modes) |
| `gap_fill_dedup.csv` | query-level bodyId dedup with `dedup_category` (precedence §4.6) and `dup` |
| `set_coverage.json` | set-level coverage: FAFB assigned/proposed/unpaired rollup, MCNS in-pool/candidates/holes per type, plus `family_material` (in-map-type bodyIds no branch pool claims — the 219−204 population overhang) |
| `relatives.csv` | the whole `relative` bin (type-mates of candidate types, ∪ evidence rows classified `relative`), per branch+bodyId |
| `pool_categories.csv` | tier + metrics + `size` per in-map target |
| `pair_summary.csv` | per branch: pools, M, gap (informational), verdict/noise counters, `pool_best_size` |
| `parameters.json` | every knob incl. `validation_mode`, cutoffs, the null-calibration knobs (`null_jaccard_max`, `null_per_source_cap`, `null_min_n`, `null_percentile`), `out_map_top_k`, and the stage skip flags |
| `morphology_calibration.json` | per-branch thresholds, `pool_ref_tier`/`baselines`/`floors`, `track_a_null_bar`/`n`, `score_frame`, AUC gate record |
| `README.txt` | glossary (verdicts, categories, noise gates, morph frames, pool-ref tiers) + full run log |
| `visualization/*.html` | tree-legend scenes per parent group |

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

## 11. Known limitations

- Scalar scan (~3 k pairs/s); full-dataset validation wants the
  vectorized sparse-rank scorer.
- The MCNS↔FAFB landmark bridge renders matched pairs contralaterally
  (opposite x-to-side conventions; needs an anatomical ground-truth
  call — mirror MCNS x, fix the landmark registration, or document).
  OPEN.
- Track B has no render-frame fallback (vector-cache path only) and no
  NBLAST variant (deferred).
- The deep window is scan-noise-dominated in the small-FAFB-type regime
  (cross-type neighbors outnumber same-type expansion ~25:1) — one more
  reason it is opt-in.
