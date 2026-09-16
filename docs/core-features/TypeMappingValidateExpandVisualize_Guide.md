# Type-Mapping Validate-Expand-Visualize — User Guide

Validate that an automatic type mapping holds at the **single-neuron
level**, review every disagreement in a 3D scene, and collect
gap-fill/candidate proposals — without ever rewriting the mapping.

- Module: Comparison → type-mapping validation pipeline
- Script: `scripts/RunMappingValidation.py`
- Technical details: [`docs/technical/TYPE_MAPPING_VALIDATE_EXPAND_VISUALIZE_PIPELINE.md`](../technical/TYPE_MAPPING_VALIDATE_EXPAND_VISUALIZE_PIPELINE.md)

---

## 1. Quick start

```bash
# validate two FAFB types against male-cns (branch-resolved)
python scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types s-CPDN3C,s-CPDN3D \
    --label my_first_run --scene-selfcheck

# a whole coarse cell type (242 neurons → 219 targets)
python scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types circadian_clock --label circadian --max-scenes 21
```

Results land in `local_data/mapping_validation/{src}_to_{tgt}_{label}_{timestamp}/`.
Runtime: ~2 min for a small type pair, ~6 min for a 50-neuron family,
~35 min for the full circadian clock (warm caches).

## 2. Reading the results

### 2.1 The 3D scenes (`visualization/*.html`)

One scene per parent type, skeleton lines in the **source brain
template** (targets are bridged in). The legend tree, top to bottom:

| legend root | meaning |
| --- | --- |
| `query · {source type}` (blue) | your source neurons |
| `matched · {target type}` (cyan) | in-map target supported by top evidence **and** a clearly positive correlation — the only tier presented as *asserted* |
| `verified · {target type}` (green) | top-ranked by one metric but below the correlation bar — strong, review |
| `borderline · {target type}` (gold) | a few outside neurons rank ahead |
| `unmatched · {target type}` (grey) | many outside neurons rank ahead (the final tier) |
| `sibling` (pink) | neurons **already mapped in another branch** of your query that show up here — cross-branch convergence, never homolog candidates |
| `candidates` (orange) | suspected neurons the mapping missed: **outside the map**, connectivity-qualified, morph-qualified. One root; each bodyId leaf carries `{type}(out-map)` / `{type}>{src}` / `{type}(no_source)` / `untyped`, plus `(dup)` when it recurs across branches |
| `family` (light green) | out-map bodyIds whose type is one of your in-map types — the same annotated family, not re-validated. One root; each bodyId leaf carries `{type}(out-map)`, plus `(dup)` **[family mode]** |
| `relative` (olive) | type-mates of candidate types for types **outside** the map — annotation-review targets; each leaf carries `{type}>{src}` / `{type}(no_source)` / `untyped` **[family mode]** |
| `suspicious` (red) | deep-window homologs below the pool best — lowest confidence; each leaf carries `{type}(out-map)` / `{type}>{src}` / `{type}(no_source)` / `untyped`, plus `(dup)` **[aggressive mode]** |
| `out-map query · {type}` (blue) | your source neurons of the type that **no branch pool claims** — the unplaced residue of the source-side gap; compare them with `candidates` to judge the fill |
| `out-map candidates · {type}` (light blue) | the top connectivity-ranked targets found by scanning those unpaired sources (no morph bars; exploratory) |

Every expansion leaf carries **one** token, decided in this order:

- `{type}(out-map)` — the type is one of **your in-map types**, so this is
  an unmapped *bodyId* of a type already in the map (the fill material;
  every `family` member). This is a **bodyId-level** statement and takes
  precedence, so an in-map type is never described by a lower token even
  when the type also maps backward.
- `{type}>{src}` — the type is **not** in-map but maps backward to a real
  source population (**type-level**).
- `{type}(no_source)` — the type is **not** in-map and has no usable
  backward route (**type-level**; e.g. CB4091).
- `untyped` — no type annotation.

Within the `candidates` / `family` / `relative` / `suspicious` roots the
bodyId leaves are sorted by `type + suffix` (not bodyId), and each root
always renders even with a single leaf.

The `untyped` suffix can appear on any bin, not only candidates. Every
category root is collapsible; double-click a row to isolate it. Hover any
skeleton line to see its **bodyId**.

**What the bins mean as a set.** The categories are a partition: every
in-scope neuron falls into exactly one, decided in the order
matched/verified/borderline/unmatched → sibling → candidates → family →
relative → suspicious. The bins nest by mode: restrictive has the tier,
`sibling` and `candidates`; family adds `family` and `relative`;
aggressive adds `suspicious`. A neuron keeps the same bin across modes —
switching mode only reveals more neurons, never relabels one.

> **Frames note:** the scenes draw in the *source* brain coordinates,
> while the morphological scores are computed in the *target* dataset's
> coordinates (Track A transforms the source in; Track B is fully
> native). Read the scene as anatomy, not as the scoring frame.

### 2.2 What "morph-qualified" means

A neuron qualifies when it passes the per-branch bar (floors v3 — one
bar engine, two currencies; exact bars per branch in
`morphology_calibration.json` → `branch_bars`):

- **preferred**: its native similarity to the branch's accepted
  (`matched`+`verified`) reference neurons clears the **native floor** —
  the reference set's mean self-similarity minus a margin, measured in
  the target dataset with no cross-dataset distortion
  (`morph_pool_ref ≥ floor`; needs ≥ 2 scored references);
- **fallback** (branch has fewer than 2 scored references): its
  similarity to your transformed query neuron clears the **Track-A
  backup floor** — the query's mean similarity to the branch's scored
  pool pairs minus an offset (`B_b − Δ`, the `morph_track_a_offset`
  parameter);
- **last resort** (no scored pool pairs either): the run's measured
  noise level — a percentile of provably-unrelated pairs
  (`track_a_null_bar`).

In **aggressive mode** the deep window (neurons ranked below the pool
best) gets a second, looser bar — the pool baseline minus
`k × Δ` — between the candidate bar and noise; those rows are labeled
`suspicious`, never fills.

### 2.2b The layered gap-fill report

`gap_fill_levels.csv` is the fill answer at a glance — one row per
non-claim bodyId, sorted by confidence:

| level | meaning |
| --- | --- |
| `high` | candidate passed the native matched+verified floor |
| `medium` | candidate passed the Track-A backup floor (`B_b − Δ`) |
| `low` | candidate passed only the run null bar |
| `type_gated` | family — an out-of-map instance of an in-map type (no individual evidence yet) |
| `advice` | relative — a candidate-type mate worth crosswalking |

`set_coverage.json` mirrors the level counts (`gap_fill_by_level`), and
candidates that close a mapped-type hole carry a `hole closer` note.
Every run also writes `pipeline_progress.jsonl` — machine-readable
stage/profile progress (the hook a UI wires to).

Both numbers are in every CSV row and summarized in
`morphology_calibration.json`.

### 2.3 What is filtered as noise

Rows that fail a gate move to `noise_filtered_candidates.csv` with a
`noise_reason` — nothing is silently deleted:

| reason | gate |
| --- | --- |
| `spatial_caliber` | candidate is < 10% of the branch pool's largest neuron (a fragment) — the primary filter |
| `negative_rank_union` | "wins" only by a non-positive correlation |
| `jaccard_below_pool` | wins on correlation but shares far fewer partners than the pool member |
| `tie_margin` | beats the pool member by < 0.02 — a numerical tie |

## 2.4 The CSVs

| file | question it answers |
| --- | --- |
| `pair_summary.csv` | per branch: pool sizes, matched count, gap (informational), verdict/noise counters |
| `validation_results.csv` | per source neuron: verdict, global ranks, scores |
| `suspicious_candidates.csv` | every expansion row with its `category` + `candidate_annotation` and both morph tracks |
| `deep_candidates.csv` | deep-window rows (`suspicious`, aggressive mode only) |
| `noise_filtered_candidates.csv` | every dropped row and why |
| `gap_fill_proposals.csv` | proposed partners for unpaired neurons (evidence only), with `counts_toward_restrictive_fill` / `counts_toward_family_fill` |
| `family_candidates.csv` | the whole `family` bin (enumerated members ∪ evidence rows classified `family`) **[family mode]** |
| `gap_fill_dedup.csv` | query-level bodyId dedup of the fill (the true filled-gap list), with the `dup` tag |
| `pool_categories.csv` | the matched/verified/borderline/unmatched tier per in-map target |
| `relatives.csv` | the whole `relative` bin (type-mates of candidate types ∪ evidence rows classified `relative`) |
| `mapping_export.csv` | the per-bridge mapping record (refined pools + linkers) |
| `morphology_calibration.json` | per-branch qualification bars + reference tiers + scoring frames |

## 3. Decision policy (what the pipeline will and will not do)

- The mapping is **never rewritten**. Fill/candidate proposals are
  evidence with full provenance.
- Only `matched` neurons are *asserted*; everything else is a review
  tier or a proposal.
- Every gate decision is reconstructible from the run folder alone —
  dropped rows keep their reasons.

## 3b. The categories, in detail

Each branch is classified by one ordered rule (first match wins), so every
in-scope neuron gets **exactly one** category. The order encodes the
priority: a validated mapping beats an explanation, which beats a
suggestion.

1. **The tier — `matched` / `verified` / `borderline` / `unmatched`.**
   The in-map targets of *this* branch, i.e. the neurons the type mapping
   itself put here. `matched` is the asserted tier (top evidence under
   both metrics, positive correlation). `verified` is strong but
   imperfect (top under one metric). `borderline` has a few outside
   neurons ahead. `unmatched` is the fallback. There is no fifth tier:
   `unmatched` always catches the rest.
2. **`sibling`.** An in-map target of your **query** that belongs to a
   *different* branch but appears in this branch's expansion. It is
   already mapped somewhere, so it is not a missing partner — it is
   cross-branch convergence, and it never counts toward the fill.
3. **`candidates`.** A suspected neuron **outside the map** that is
   connectivity-qualified (an invader ahead of the pool best, or a gap
   fire) **and** morph-qualified. This is the restrictive fill material —
   the neurons worth reviewing as genuinely missing partners. The legend
   has ONE `candidates` root; each **bodyId leaf** carries the ordered
   token `{type}(out-map)` (an unmapped bodyId of an in-map type) /
   `{type}>{src}` (foreign type with a real backward home) /
   `{type}(no_source)` (foreign type, no route) / `untyped`. A standalone
   `(dup)` tag is appended to leaves whose bodyId recurs across branches.
4. **`family`.** In *family mode*: every out-map bodyId whose type is
   **this branch's own target type** (branch-local). Gate is type
   membership, not morphology, so this is the whole annotated family
   rather than a homolog call. It cannot explode — it is bounded by the
   populations of the types the mapping already names. One `family` root;
   each leaf carries `{type}(out-map)`.
5. **`relative`.** In *family mode*: the type-mates of candidate types,
   restricted to types **outside** the map. Where `family` covers your
   in-map types, `relative` covers the neighborhoods of the suspects
   themselves. The two are disjoint by type. One `relative` root; each
   leaf carries `{type}>{src}` / `{type}(no_source)` / `untyped`.
6. **`suspicious`.** In *aggressive mode*: the deep window — out-of-pool
   homologs ranked *below* the pool best. Lowest confidence, reviewed
   last. One `suspicious` root; each leaf carries `{type}(out-map)` /
   `{type}>{src}` / `{type}(no_source)` / `untyped`, plus `(dup)` when the
   bodyId recurs.

**Leaf ordering.** Within a `candidates` / `family` / `relative` /
`suspicious` root the bodyId leaves are sorted by `type + suffix` (e.g.
`CB1011>s-CPDN3B` before `SMP217(no_source)` before `SMP217(no_source)
(dup)`), not by bodyId, so the same target type groups together. The root
row always renders, even when it holds a single leaf.

Two sums matter. The **restrictive fill** counts `candidates` only. The
**family fill** counts `candidates + family + relative`, an
over-inclusive superset that may exceed the raw numeric gap (219→242 need
not yield exactly 23). Both are exported; the fill is a ranked process,
not a deterministic result.

At the query level, one bodyId can legitimately appear in several branches
(N-to-1 mappings repeat a target type across source types). The dedup
export gives each bodyId one row, choosing in the order
`tier > sibling > candidates > family > relative`, and flags non-sibling
repeats with `dup`. The same `dup` flag is written onto every per-branch
CSV row, and the scene marks those roots with a ` (dup)` suffix (e.g.
`suspicious · SMP217 (dup)`) so you can see which expansion neurons recur
across branches.

**Connectivity-only rows are kept, not shown.** A suspect that ranks ahead
but *fails* the morphological qualification is not a category — but it is
still exported, flagged `in_scope=False` / `morph_failed=True`. The reason
is reconciliation: these are exactly the rows a **connectivity-only
homolog search** would return, so you can compare the pipeline's output
against one. They never appear in the scene.

## 4. Expansion modes

The three modes **nest**: `restrictive ⊆ family ⊆ aggressive`. Each adds
neurons; it never relabels one. Pick the smallest mode that answers your
question.

| mode | flag | adds | behavior |
| --- | --- | --- | --- |
| **restrictive** (default) | — | — | the validated tier + `sibling` + `candidates` (invaders and gap fires that pass the morph rule). Minimal expansion |
| **family** | `--mode family` | `family`, `relative` | also surfaces every out-map bodyId of your in-map types (`family`) and the type-mates of candidate types (`relative`). Ungated by qualification — bounded by the types themselves |
| **aggressive** | `--mode aggressive` | `suspicious` | everything in family, plus the deep window (out-of-pool homologs ranked *below* the pool best). Over-expansion prone — review carefully |

The mode is recorded in `parameters.json` (`validation_mode`). Only
`candidates` count toward the restrictive gap fill; `family` and
`relative` are broader, lower-confidence suggestions.

## 5. Options you may actually use

| flag | effect |
| --- | --- |
| `--types` (required) | source types or a coarse `cell_type` (e.g. `circadian_clock`) |
| `--scene-selfcheck` | verify every legend leaf's geometry matches its neuron (recommended) |
| `--max-scenes N` | cap rendered scenes (default 12) |
| `--target-min-size-ratio` | fragment bar (default 0.1 × pool best) |
| `--suspicious-ru-margin` | numerical-tie rule (default 0.02) |
| `--no-morphology` | skip stage 5 (fast structural pass) |
| `--mode {restrictive\|family\|aggressive}` | expansion mode (§4); default restrictive |
| `--neuron-alpha`, `--max-scenes`, `--quiet` | rendering/verbosity |

## 6. Interpreting common outcomes

- **`sibling` roots dominate the scene** — normal: adjacent branches of
  the same source type share ranking neighbors, and those neighbors are
  already mapped. They never fill the gap.
- **A `candidates` root with many leaves** — neurons the mapping missed
  that are connectivity- and morph-qualified: the highest-value review
  targets. Read each leaf's token (`{type}>{src}` / `{type}(no_source)` /
  `untyped`) to see whether the type maps back.
- **A `family` root with many members** — family mode: every out-map
  bodyId of your in-map types. These are the same annotated family,
  *not* re-validated — use them to see the full type population, not as
  homolog calls. Each leaf shows the raw type.
- **A `relative` root** — family mode: type-mates of candidate types for
  types outside the map. Context for reviewing a candidate, not a
  proposal itself; leaves carry the `{type}>{src}` / `(no_source)` token.
- **A `suspicious` root** — aggressive mode: deep-window homologs below
  the pool best; the lowest confidence, and a known over-expansion
  signal in finely identified brain regions. Leaves carry the same
  `{type}>{src}` / `(no_source)` token plus `(dup)` when a bodyId recurs.
- **`sibling` proposals don't fill the gap** — a proposed partner already
  mapped in another branch is cross-branch convergence; it is excluded
  from the fill count.
- **Hemisphere asymmetry fires the gap** — every neuron has a hemisphere
  identity, so an L/R imbalance in either pool triggers fill review even
  when the arithmetic gap is 0 (see the `hemisphere` block in
  `pair_summary.csv`).
- **`set_coverage.json` — the bottom-line answer.** Branch gaps
  double-count cross-branch convergence; this file rolls coverage up to
  the set level: how many of your source neurons are assigned / only
  fill-proposed / unpaired, and which target neurons of the mapped set
  are **holes** (never claimed by any branch pool, proposal, or
  morph-qualified candidate — the true unmapped residues, listed by
  bodyId).
- **`gap_fill_dedup.csv` — the deduplicated fill.** A single bodyId can
  appear in several branches (N-to-1 mappings); this file gives the
  query-level, bodyId-unique fill list and flags cross-branch duplicates
  with `dup`. Siblings are never flagged — they are duplicates by
  definition.
- **`gap` is informational.** Fill proposals are the ranked suggestions;
  only `candidates` count toward the restrictive fill, while family mode
  reports the broader `candidates + family + relative` fill.
- **Fill numbers are query-scope-relative.** A single-type query reports
  its cross-type neighbors as expansion advice instead of counting them
  as fills; querying the whole family lets those neurons be accounted by
  their own branches. To validate a family honestly, query the family —
  or use per-type queries and read the expansion advice as the pointer to
  the next query.
