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

# a whole coarse cell type (242 FAFB neurons → 204 MCNS neurons map-covered)
python scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types circadian_clock --label circadian

# review the fill with the advisory reciprocal evidence (stage 5d, §2.2c)
python scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types s-CPDN3C,s-CPDN3D --mode family --backward-evidence
```

Results land in
`{output_dir}/type-map-validation_{SRC}_to_{TGT}_{timestamp}/` (short
dataset nicknames, `YYYYMMDD_HHMMSS` stamp; the `--label` is recorded in
`parameters.json` and the report rather than the folder name).
Runtime: ~2 min for a small type pair, ~6 min for a 50-neuron family,
~35 min for the full circadian clock (warm caches).

## 1b. Run from the UI

The pipeline has a first-class UI tab: **Cross-Dataset › Type Validation**
(`ui/tabs/type_validation.py`, tool key `type_mapping_validation`). Pick a
Source and Target dataset, add one or more source types / coarse `cell_type`
categories as chips, choose the mode, then toggle the optional stages. The tab
produces the exact same run folder as the CLI (see §2.4b). A short offline pass
is: **Cache-only profiles** checked + **Morphology** off + **3D scenes** off —
that keeps the run on local tables (no NeuPrint/CAVE).

The form speaks the **dataclass field names**, not the CLI flags, so the
mapping below is the contract (UI control → `MappingValidationConfig` field →
CLI flag). Several flags are renamed or negated, and some fields have no CLI
flag at all — the tab is their only entrance.

| UI control | Config field | CLI flag |
|---|---|---|
| Source / Target dataset | `source_dataset` / `target_dataset` | `--source` / `--target` |
| Query chips | `query_types` (comma-split) | `--types` |
| Run Label | `run_label` | `--label` |
| Output directory | `output_dir` | `--output-dir` |
| Validation Mode (Restrictive/Family/Aggressive/Pooling) | `validation_mode` | `--mode` |
| Pooling gate card (visible only in mode Pooling): bar metric / bar depth (top-N per metric) / Jaccard floor (advisory flag) / window multiplier (advisory flag) / morph budget | `pooling_bar_metric` / `pooling_bar_top_n` / `pooling_jaccard_floor` / `pooling_window_mult` / `pooling_max_morph_targets` | `--pooling-bar-metric` / `--pooling-bar-top-n` / `--pooling-jaccard-floor` / `--pooling-window-mult` / `--pooling-max-morph-targets` |
| Morphology verification | `morph_enabled` | `--no-morphology` (negated) |
| 3D review scenes | `visualize` | `--no-visualize` (negated) |
| Backward (reciprocal) evidence | `backward_evidence_enabled` | `--backward-evidence` |
| Backward scans unmatched pool members | `backward_scan_pool_targets` | `--no-backward-pool-targets` (negated) |
| Suspicious per-source cap | `suspicious_per_source_cap` | `--suspicious-cap` |
| Cache-only profiles | `skip_profile_build` | `--skip-profile-build` |
| Skeleton-fetch threads | `skeleton_fetch_workers` | `--skeleton-fetch-workers` |
| Skeleton-fetch timeout (s) | `skeleton_fetch_timeout_s` | `--skeleton-fetch-timeout` |
| Skip out-map expansion | `skip_out_map_expansion` | `--skip-out-map-expansion` |
| (Advanced) various thresholds | see §5 | various `--…` flags |

No CLI flag (tab / dataclass only): `top_k`, `top_m`, `min_synapse_threshold`,
`include_untyped_partners`, `null_jaccard_max`, `null_per_source_cap`,
`null_min_n`, `null_percentile`, `candidate_morph_cap`, `use_cache`. The first
four plus `use_cache` are driven from the **Settings** defaults (`top_k`,
`top_m`, `min_synapse_num`, `use_cache`); the `null_*` and
`candidate_morph_cap` live in the tab's **Advanced** card. `verbose=True` is
always sent so the report/README log carries every stage banner.

## 2. Reading the results

### 2.0 The per-run report (`report.html`)

Every run writes ONE self-contained `report.html` into the run folder —
start there. It assembles the headline (e.g. **242 source neurons →
204 male-cns:v1.0 neurons map-covered**), the three coverage levels
(L1 claim / L2 provenance / L3 validation), the branch table, the
target-side holes, the fill proposals with per-row provenance, the
**Reciprocal** tab (stage 5d, only on `--backward-evidence` runs — §2.2c),
the out-map expansion, the backward `source-` view, the **Pooling** tab (the
whole result of a `--mode pooling` run — §4), the morphology record
(with the null-sample advisory when null-kind bars are in play), the
scene gallery, and a file index.
Hover any dotted term — or any table header, which explains its own column
and any tolerance it reads against — for its definition. Every `!` log line is
reproduced verbatim in its Warnings section, and the same warnings are
appended to `user_warning_notes.txt`. `README.txt` stays slim
(directions + the raw run log). Any past run can be regenerated:

```
python -m comparison.mapping_validation_report <run_dir>
```

### 2.1 The 3D scenes (`visualization/*.html`)

One scene per parent type, skeleton lines in the **source brain
template** (targets are bridged in). By default every parent type in the
run gets a scene (`--max-scenes 0`); a positive value keeps the largest
pools and names each dropped parent in the run log and in the report's
Branches tab. The legend tree, top to bottom:

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
| `examinees` (red) | deep-window homologs below the pool best (renamed from `suspicious` — the mapper's rival-suspects concept now owns that word) — lowest confidence; each leaf carries `{type}(out-map)` / `{type}>{src}` / `{type}(no_source)` / `untyped`, plus `(dup)` **[aggressive mode]** |
| `out-map query · {type}` (blue) | your source neurons of the type that **no branch pool claims** — the unplaced residue of the source-side gap; compare them with `candidates` to judge the fill |
| `out-map candidates · {type}` (light blue) | the top connectivity-ranked targets found by scanning those unpaired sources (no morph bars; exploratory) |
| `pooling · {source type}` (plum — `#7b4173`) **[pooling mode]** | the unsupervised pool whose best source has this type — one leaf per candidate target, ordered by the exported leaf token and tagged `{mapper_cell} · {tiers} · morph ✓/✗` (`confirmed` / `type_miss` / `type_new`; the tier set that reached the target; the verdict where the morph gate scored it). The root is drawn from the rows the export marks `in_pool=True`, so a target the morphology bar refused on every row is already gone from it — nothing is re-gated for the picture, and the scene is exactly that subset of `pooling/pooling_pool.csv` (which keeps the refused rows, with `in_pool=False`, so the refusal stays countable). A pool whose best source has a type no branch group covers has no host scene and is named in the run log. Report: Pooling tab |
| `source-candidates · {target type}` (brown — Category10 `#8c564d`; hidden by default, one eye click restores) | **out-of-map sources** (claimed by no branch) whose best-ranked connectivity hits land in this branch's targets and pass the run null bar — the backward mirror of candidate admission; advisory only, never a fill. The leaf's type suffix is the source's own type; `(dup)` marks sources that are candidates for several branches. Report: Backward tab |

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
- On a stage 5d (`--backward-evidence`) run, a scanned `candidates` /
  `family` / `relative` leaf ends with one more tag: `· high`,
  `· medium` or `· low` (§2.2c). A member the pass did
  not scan keeps a **bare leaf** — no tag means "not checked", never
  "failed".

Within the `candidates` / `family` / `relative` / `examinees` roots the
bodyId leaves are sorted by `type + suffix` (not bodyId), and each root
always renders even with a single leaf.

The `untyped` suffix can appear on any bin, not only candidates. Every
category root is collapsible; double-click a row to isolate it. Hover any
skeleton line to see its **bodyId**.

**What the bins mean as a set.** The categories are a partition: every
in-scope neuron falls into exactly one, decided in the order
matched/verified/borderline/unmatched → sibling → candidates → family →
relative → examinees. The bins nest by mode: restrictive has the tier,
`sibling` and `candidates`; family adds `family` and `relative`;
aggressive adds `examinees`. A neuron keeps the same bin across modes —
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

In **aggressive mode** the wider window (neurons ranked below the pool
best and beyond `rank_top_k`) gets a second, looser bar — the pool baseline minus
`k × Δ` — between the candidate bar and noise; those rows are labeled
`examinees`, never fills.

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

### 2.2c Reciprocal homolog evidence (stage 5d, opt-in)

The fill answers "who could belong here?". **Stage 5d** asks the neurons
it proposed the uncomfortable reverse question: *you were found by
branch B — but when we scan YOU back against the entire source dataset,
do you prefer branch B's neurons too?* Each `candidates` / `family` /
`relative` member is scored against the **whole source universe** with
the same homolog finder (unmatched pool targets are scanned too;
matched / verified / borderline are not — the symmetric forward score
is their evidence). It is opt-in (`--backward-evidence`, default **off**)
and it is **advisory**: it never moves a neuron between bins, never
changes a fill level or a count, and **no morphology is re-scored** —
candidates are already morph-qualified, and family / relative members
are morph-similar to the query or to those candidates.

The CSV token IS the display label (no translation layer). The three
scanned grades measure how prominently the member's OWN branch source
type ranks in the reverse scan — pure rank evidence, no score bar, no
pool-membership gate:

| label = CSV value | what it says |
| --- | --- |
| `high` | a hit of the branch's own source type is the **top-1** reverse hit by rank_union or by jaccard — wherever that hit lives |
| `medium` | the branch's own source type appears within the **top-3** of either ranking (but is not a top-1) |
| `low` | scanned, but the branch's source type ranked outside both top-3 windows (or nothing usable ranked) — **a graded negative, not an error** |
| `not-checked`, or a blank cell | `not-checked` = the pass is off, the member was over the per-run / per-branch budget, or it has no usable profile to scan (`backward_scanned_at` says which). A BLANK cell means the row is not reverse-scan material at all — only `candidates` / `family` / `relative` and the UNMATCHED pool members (the in-map control) are enumerated, and matched / verified / borderline members never are. Both display as an explicit dash so absence is never read as failure |

**Reading the Reciprocal tab.** One row per scanned **neuron** (a member
claimed by several branches is graded per branch and listed once, with
every claiming branch and its grade in the `branch` cell), grouped by
bin and then by member type, each group ordered by the **branch-type
hit's jaccard** (descending) — the evidence the badge is about, so a neuron
whose own type never ranks (every `low`) sinks below the rows that have a
hit rather than floating up on some unrelated top-1's score. The badge is
that neuron's STRONGEST branch grade. Two hit columns sit
side by side, because they answer different questions:
**branch-type hit** is the one the badge rests on — the best-ranked source
of the claiming branch's OWN type, with the rank it reached and *which*
ranking placed it there (`by rank_union` / `by jaccard`);
**top-1 source** is the globally best hit on the ordering chain (Jaccard
first, rank_union breaking a Jaccard tie, bodyId last)
(source bodyId · its source type · `this branch` / `elsewhere` ·
rank_union). They are different neurons whenever the branch's type wins on
jaccard alone, which is why a row can read `high` while its top-1 sits
`elsewhere`. **Hover the top-1** for the top-N mini table (both ranks,
source bodyId, source type, rank_union, jaccard, in-branch flag). The
**shared/union types** column says what the top-1's score was computed
over: `rank_union` ranks the union of the two partner vectors and scores a
type one side lacks as 0.0, so a `2/9` and a `20/40` score of 0.42 are not
the same claim — the `thin` marker flags the ≤3-shared cases for the
reader, and nothing else (it never changes the verdict, a bar or a fill;
the branch-type hit carries its own marker, since that is the evidence the
grade stands on).
The tab
header gives the per-bin counts and the budget note (how many members
were scanned of how many eligible), and the **Fill** tab carries the same
fact three ways: a `reciprocal` column per row, a headline count, and a
`Reverse evidence by bin` split (`family 8 scanned → high 5 · low 3`)
that sits beside the level line because it is a different axis.
A scanned leaf in a
3D scene carries the matching `· high` / `· medium` / `· low` tag
(§2.1), and the Warnings section states
`[reciprocal] N/M scanned gap-fill member(s) rank their own branch source
type in a top-3 — advisory provenance to review, never a rejection`.

**When to use it.** Whenever the fill itself is the question — a
`low` grade is the cheapest way to spot a member whose own branch source
type reaches neither top-3 in its reverse scan, and a high `reciprocal`
share is the strongest support a type-gated `family` row can get without
morphology.
Skip it when you only need the tier: one reverse scan costs what a
forward source scan costs, so a large query is bounded by
`--backward-max-neurons` (300) and `--backward-per-branch-cap` (40).

### 2.3 What is filtered as noise

Rows that fail a gate move to `noise_filtered_candidates.csv` with a
`noise_reason` — nothing is silently deleted:

| reason | gate |
| --- | --- |
| `spatial_caliber` | candidate is < 10% of the branch pool's largest neuron (a fragment) — the primary filter |
| `negative_rank_union` | "wins" only by a non-positive correlation |
| `jaccard_below_pool` | wins on correlation but shares far fewer partners than the pool member |
| `tie_margin` | beats the pool member by < 0.02 — a numerical tie |

### 2.3b Backward `source-` statuses (advisory)

The same bodyId-bodyId pair scores also grade the **source** side (the
column view: per target, rank its sources). Every in-branch source gets a
read-only status — `source-matched` (column-best of its own top target,
above the matched bar), `source-verified` (column-best of another target),
`source-borderline` (few out-of-branch sources rank above it),
`source-unmatched` (dominated by out-of-branch sources). These are for
your reading only: an unpaired source is usually a *column runner-up of an
already matched/verified target* (population surplus + N-to-1
convergence), not a mapping failure. They never gate and never enter the
dedup. See the report's Backward tab and `expansion/source_status.csv`.

On a stage 5d run these statuses get sharper: reverse-scanning the pool
members (the unmatched ones since 2026-09-19) ranks each target's column
over the WHOLE source universe, so out-of-branch rivals above a source
become real. Without that, a column can only ever hold the branch's own
sources and
`n_competitors` is structurally 0 — so the rival-above reading behind
`source-borderline` / `source-unmatched` only becomes reachable with the
pass on.

## 2.4 The CSVs

| file | question it answers |
| --- | --- |
| `validation/pair_summary.csv` | per branch: pool sizes, `best` (mutual-best 1:1 pairs), gap (informational), verdict/noise counters. The report's Branches tab reads **Mapped** = verified_strong+verified+borderline and measures its displayed gap against that, because a source can carry a verdict without being paired; both numbers hover side by side |
| `validation/validation_results.csv` | per source neuron: verdict, global ranks, scores |
| `validation/examinees.csv` (was `suspicious_candidates.csv`) | every expansion row with its `category` + `candidate_annotation`, both morph tracks, and the `backward_*` columns (§2.2c) |
| `mapping/same_name_excluded.csv` | queried types whose same-name fan-out was held/excluded, or multi-value cells — advisory accounting |
| `mapping/suspects_verification.csv` | opt-in (`--verify-suspects`): rival-suspect connectivity verification — advisory |
| `validation/deep_candidates.csv` | candidate-window rows below the pool best — the top-`rank_top_k` band (`candidates`, family mode and up) and the wider aggressive band (`examinees`) |
| `validation/noise_filtered_candidates.csv` | every dropped row and why |
| `gap_fill/gap_fill_proposals.csv` | proposed partners for unpaired neurons (evidence only), with `counts_toward_restrictive_fill` / `counts_toward_family_fill`; `side` says whether `bodyId` is the source or the target neuron of the pair |
| `gap_fill/gap_fill_levels.csv` | the layered fill (§2.2b) + each row's `backward_evidence` |
| `expansion/family_candidates.csv` | the whole `family` bin (enumerated members ∪ evidence rows classified `family`) **[family mode]** |
| `expansion/backward_matches.csv` | stage 5d only (`--backward-evidence`): one row per (branch, scanned neuron) — reverse top-1, the `backward_own_type_*` hit the grade rests on, and the serialized top-N neighbourhood (§2.2c) |
| `expansion/source_status.csv` | the backward `source-` statuses (§2.3b) |
| `expansion/out_map_expansion.csv` | each unclaimed source neuron's top connectivity-ranked targets (exploratory, never a fill) |
| `gap_fill/gap_fill_dedup.csv` | query-level bodyId dedup of the fill (the true filled-gap list), with the `dup` tag and, on stage-5d runs, the reciprocal rollup of the neuron's strongest branch (§2.2c) |
| `validation/pool_categories.csv` | the matched/verified/borderline/unmatched tier per in-map target |
| `expansion/relatives.csv` | the whole `relative` bin (type-mates of candidate types ∪ evidence rows classified `relative`) |
| `mapping/mapping_export.csv` | the per-bridge mapping record (refined pools + linkers) |
| `morphology_calibration.json` (root) | per-branch qualification bars + reference tiers + scoring frames |

### 2.4b Where the files live

Since 2026-09-19 the evidence CSVs are grouped by pipeline stage; the run
folder's root keeps only the deliverables and the parameter/meta surface:

```
{run folder}/
├── report.html · README.txt · _UserGuide_please_read_me.*
├── parameters.json · set_coverage.json · morphology_calibration.json
├── pipeline_progress.jsonl · user_warning_notes.txt
├── validation/    stage 2/3 evidence (verdicts, pools, examinees)
├── expansion/     the bins + their reciprocal evidence (backward_matches)
├── gap_fill/      the fill accounting
├── mapping/       mapping provenance
└── visualization/ the 3D scenes (§2.1)
```

`README.txt`'s start-here list and the report's file index use these
paths. A folder written **before** 2026-09-19 keeps its flat layout and
still opens and regenerates — every reader tries the subfolder first and
falls back to the root.

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
   connectivity-qualified (an invader ahead of the pool best, a gap fire,
   or — *family mode and up* — a neuron inside the candidate-discovery
   window, the top `rank_top_k` of **each** metric) **and**
   morph-qualified. This is the restrictive fill material —
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
6. **`examinees`.** In *aggressive mode*: the deep window — out-of-pool
   homologs ranked *below* the pool best. Lowest confidence, reviewed
   last. One `examinees` root; each leaf carries `{type}(out-map)` /
   `{type}>{src}` / `{type}(no_source)` / `untyped`, plus `(dup)` when the
   bodyId recurs.

**Leaf ordering.** Within a `candidates` / `family` / `relative` /
`examinees` root the bodyId leaves are sorted by `type + suffix` (e.g.
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
`examinees · SMP217 (dup)`) so you can see which expansion neurons recur
across branches.

**Connectivity-only rows are kept, not shown.** A suspect that ranks ahead
but *fails* the morphological qualification is not a category — but it is
still exported, flagged `in_scope=False` / `morph_failed=True`. The reason
is reconciliation: these are exactly the rows a **connectivity-only
homolog search** would return, so you can compare the pipeline's output
against one. They never appear in the scene.

**A fill member can be asked back.** On a `--backward-evidence` run every
`candidates` / `family` / `relative` member also carries a *reciprocal*
label — which source neuron it prefers when IT is scanned against the
whole source universe (§2.2c). Advisory by construction: it never moves a
neuron between the bins above, never changes either sum, and scores no
morphology.

## 4. Expansion modes

The three modes **nest**: `restrictive ⊆ family ⊆ aggressive`. Each adds
neurons; it never relabels one. Pick the smallest mode that answers your
question. A fourth value, `pooling`, sits OUTSIDE that ladder and is
described after the table.

| mode | flag | adds | behavior |
| --- | --- | --- | --- |
| **restrictive** (default) | — | — | the validated tier + `sibling` + `candidates` (invaders and gap fires that pass the morph rule). Minimal expansion |
| **family** | `--mode family` | discovery window, `family`, `relative` | reads the top `rank_top_k` of each metric as candidate evidence, then surfaces every out-map bodyId of your in-map types (`family`) and the type-mates of candidate types (`relative`). The last two are ungated by qualification — bounded by the types themselves |
| **aggressive** | `--mode aggressive` | `examinees` | everything in family, with the window widened to `candidate_window` (25); the band beyond `rank_top_k` is labelled `examinees` (out-of-pool homologs ranked *below* the pool best). Over-expansion prone — review carefully |
| **pooling** | `--mode pooling` | an independent UNSUPERVISED pool (`pooling/`) | NOT a rung of this ladder: every neuron the query names is scanned against the WHOLE target universe, and each source then keeps its top-N rows under the admission bar (`pooling_bar_metric` × `pooling_bar_top_n`, default `either`/3, where `either` is the union of both metrics' own top-N); the Jaccard / rank_union floors and the window survive only as **advisory flags** published per row. Morphology then gates those connectivity survivors through the Find-Homolog fast path (no NBLAST), as the LAST GATE, a true one and a MANDATORY one (`--no-morphology` with `--mode pooling` is a usage error): a candidate scored below its bar leaves the exported pool. Nothing reads a branch pool or a mapper claim, so this is a homolog *finding*, not a pool *refinement*: the mapper's claims are joined afterwards into `pooling_cross_validation.json` (`confirmed` / `type_miss` / `type_new` / `verified_only`), and `parameters.json` carries `validation_mode: pooling` with **`mode_rank: null`** (the key is always published; a pooling run simply has no rung on the ladder). The nested bins above are still produced, from the mapper's side of the run |

Three readings of `pooling` keep it honest. `verified_only` is expected to be
large: the supervised tier admits by pool membership while this gate admits
by global rank, so it is not a false-positive count. And no cell is a recall
measure — the unsupervised seed is the queried population, whereas the
supervised pair set exists only for the sources a branch claimed. And the
morph summary is a RECORD, not a ratio of convenience: `attempted` is what
the budget allowed to be looked at and `scored` is what the scorer returned a
verdict for, so `scored` can sit far below `attempted` (one BANC run recorded
`attempted 8 / scored 8` while its own rows said 1 `scored` and 7
`no-score`). Read `scored/attempted`, never `attempted` alone — a pool
that was mostly unscored is a missing measurement, not a morphologically
cleared one, and when the two differ the Pooling tab, the Log tab and
`user_warning_notes.txt` all say so from one builder. `no-score` means the
scorer returned NO VALUE for the pair, never "this neuron is not in the
vector store": the collapsed `scored` counts behind that example came from
the shared `mapping_ref` scorer pruning every branch-pool member as a
bar-anchor rather than a candidate, which deletes the verdict of every
candidate the mapper also claims. Pooling passes `prune_pool_refs=False`
(`comparison/morph_cross_dataset.py::qualify_visualized_pairs`) because its
candidate set is its own gate, not a branch pool; a low `scored` now means a
missing measurement.

**Morphology is the last gate, and a gate that bites.** A candidate whose
chain-best row is `morph_gate='scored'` with `morph_qualified` false leaves
`pooling_pool.csv` AND its scene's `pooling · {source type}` root; its
per-pair rows stay in `pooling_candidates.csv`, so the refusal is auditable
rather than simply absent. The count is published as `morph.dropped_targets`
beside `morph.gate_applied`, and the run log states it (`[pooling] … N refused
by the morphology bar …`) — a shrunken pool must never read as a smaller
harvest. Only an explicit refusal removes a target: `no-score`,
`not-attempted-cap`, `shared`, `disabled`, `inactive` and `error` all
stay in the pool and are named as what they are. `shared` is the honest edge of
the hybrid: the pass scores one verdict per **scoring unit** (every tier-1 row,
plus each target's chain-best row), and a `nominated` row that reads another
pair's number says which one in `verdict_for_pair` — a borrowed number is never
this row's own measurement.

**Read a verdict from the four columns that made it.** `morph_bar_kind` names
the binding rule (`native` / `track_a` / `null_bar`), `morph_bar` is that rule's
value, and the score it was applied to is `morph_pool_ref` for a native row and
`morph_similarity` otherwise — so `morph_qualified` is always recomputable from
the row it sits on, and the Pooling tab's morphology cell prints exactly that
pair with its kind. An earlier build published the per-source null bar under a
`native` kind label, which made 14 of 41 scored rows read as a score below its
own bar with a ✓ beside it: the gate had graded the right pair, the export could
not show it.

**The bar admits; the floors only flag.** Admission is per source: each queried
neuron keeps the top-N rows of `pooling_bar_metric` (`either` | `jaccard` |
`rank_union`, default `either`) at depth `pooling_bar_top_n` (default 3), where
`either` is the **union of both metrics' own top-N** — not a merged best-rank
ordering, which spends the slots on the two metrics' rank-1 rows and loses the
candidates the union recovers. A `2N` row cap bounds the tie mass, and
`bar.rows_cut` says what it cut. `pooling_cross_validation.json`'s `bar` block is
that rule as run.

The floors are the demoted part of the old gate, and they are still published:
`pooling_jaccard_floor` (0.10), `pooling_rank_union_floor` (0) and
`pooling_window_mult` (2.0) are measured against every admitted row and exported
as `below_jaccard_floor` / `below_rank_union_floor` / `outside_window`, with the
counts in `floor_flags`, and **no row is removed for missing one**. They were
filters until 2026-09-24, when the rank_union floor — sitting at its own default
`0`, which is only a sign test — was measured starving 119 of 242 queried sources
of any candidate at all. A connectivity threshold says how wide the scan opened,
never whether a pair is a homolog; that is the morphology gate's job, which is
why it is mandatory here.
An earlier build also fitted the Jaccard floor per dataset pair — the minimum of
the configured value and the q05 of that pair's own graded
`matched`/`verified` jaccards, read from a persisted evidence store before the
scan — and the fit is DELETED, not tuned:
it measured how much of a pair's graded evidence a floor rejects, which is
exactly the pool-volume question a volume knob is allowed to answer and
exactly not the quality question its provenance line implied. The measurement
that motivated it stays as the reason, historically: the same floor rejects
very different shares of two dataset pairs' graded rows (measured over the
runs on record, `J=0.20` rejected 2.1 % of FAFB→male-cns (241 pairs), 36.6 %
of FAFB→BANC (164) and 20.0 % of FAFB→hemibrain (145), where the shipped 0.10
rejected none of them) — which is a statement about how many candidates each
dataset pair yields, not about which of them are good homologs.

**The `target_type` a candidate carries corroborates nothing, and no
cross-dataset column exists.** An earlier build could count, across sibling
runs of the same query against other targets, how many of them put the same
(source bodyId, candidate target TYPE) pair in their pool, and published it as
`targets_corroborated` (`--pooling-corroborate-with`). Both are DELETED: in
pooling the type name is an advisory label the mode deliberately neglects —
admission is by connectivity rank and the target's own annotation is only
joined afterwards to name the `mapper_cell` — so counting how many other
datasets neglected it the same way is not corroboration, and grades nothing.
There is now no cross-dataset agreement column at all; `pooling_pool.csv` and
`pooling_candidates.csv` are read on their own rows, per dataset pair.

The UI offers `pooling` as a fourth **Pooling** button with its own gate card
(§1b) because the tab must be able to START the mode — not because the mode
is wider: `ui/tabs/type_validation.py:MODE_OPTIONS` is
`VALIDATION_MODES + ['pooling']`, pinned beside `POOLING_MODE not in
MODE_RANK` by a test. The result reads in the report's **Pooling** tab (§2.0)
and, inside a scene, as the `pooling · {source type}` legend root (§2.1). The
tab's *Scored against* line names the stores the cells were measured in
(`input_fingerprint`: git rev, target universe, mapper snapshot) — cells from
runs that read different stores are not the same measurement, and a run that
predates the fingerprint says so instead of implying a comparison it cannot
support. The run's `README.txt` "Start here" list names `pooling/` for a
pooling run.

The mode is recorded in `parameters.json` (`validation_mode`, plus
`mode_rank` for the three nested ones — a `pooling` run publishes
`validation_mode: pooling` and NO `mode_rank`, because there is no rung to
be ranked on). Only
`candidates` count toward the restrictive gap fill; `family` and
`relative` are broader, lower-confidence suggestions.

## 5. Options you may actually use

| flag | effect |
| --- | --- |
| `--types` (required) | source types or a coarse `cell_type` (e.g. `circadian_clock`) |
| `--scene-selfcheck` | verify every legend leaf's geometry matches its neuron (recommended) |
| `--max-scenes N` | cap rendered scenes; default **0** renders one scene per parent type (§2.1) |
| `--target-min-size-ratio` | fragment bar (default 0.1 × pool best) |
| `--suspicious-ru-margin` | numerical-tie rule (default 0.02) |
| `--no-morphology` | skip stage 5 (fast structural pass) |
| `--mode {restrictive\|family\|aggressive}` | expansion mode (§4); default restrictive |
| `--mode pooling` | the parallel unsupervised engine (§4) — writes `pooling/` and joins the mapper afterwards; it does not accept a widening flag (`--aggressive-expansion` with it is a usage error) |
| `--pooling-bar-metric {either\|jaccard\|rank_union}` / `--pooling-bar-top-n N` | **the admission bar** (default `either` / `3`): each queried source keeps the top-N rows of its chosen metric, and `either` is the UNION of both metrics' own top-N — never a merged best-rank ordering, which spends the slots on the two metrics' rank-1 rows and loses the candidates the union recovers |
| `--pooling-jaccard-floor` / `--pooling-rank-union-floor` | **advisory flags, not filters** (default 0.10 / 0): every admitted row is measured against them and published as `below_jaccard_floor` / `below_rank_union_floor`, and nothing is removed for missing one. They were filters until the rank_union one alone was measured to starve 119 of 242 queried sources |
| `--pooling-window-mult` | the window the `outside_window` flag measures against = this × the source type's own queried population (default 2.0). Advisory: it removed nothing even when it was a filter |
| `--pooling-max-morph-targets N` | morph budget in **scoring units** (one network-bound step each; a unit is a tier-1 row, or the chain-best row that owes a target its verdict). Default **0 = auto: 3 × the number of queried source neurons**, so a 58-source query is not budgeted like a 242-source one. Rows past the budget are labelled `morph_gate=not-attempted-cap`, never blank, never read as rejections, and `morph.capped` / `morph.budget` say how many and under which rule |
| `--backward-evidence` | stage 5d: reverse (target → source) homolog evidence on the `candidates` / `family` / `relative` bins (§2.2c) — **advisory, connectivity-only, default OFF** |
| `--skip-backward-pass` | force-skip stage 5d even when `--backward-evidence` is set |
| `--backward-top-n N` | reverse hits kept per neuron — the hover list in the Reciprocal tab (default 5) |
| `--backward-max-neurons N` | per-run budget of dataset-scale reverse scans (default 300) |
| `--backward-per-branch-cap N` | max expansion members labeled per branch (default 40) |
| `--no-backward-pool-targets` | do not reverse-scan the unmatched pool targets (matched / verified / borderline are never scanned — the symmetric forward score is their evidence) — then the `source-` statuses keep no pool-derived out-of-branch rivals (§2.3b) |
| `--neuron-alpha`, `--quiet` | rendering/verbosity |

Turn `--backward-evidence` on when the *fill* is the question — it is the
only surface that tells you whether a proposed member wants this branch
back. Leave it off for a pure tier check: a reverse scan costs what a
forward source scan costs, so on a large query most members simply stay
`not-checked` once the caps bind (the Reciprocal tab says how many were
eligible). `--backward-max-neurons` / `--backward-per-branch-cap` trade
coverage against wall time; `--backward-top-n` only changes the hover
detail, not a verdict.

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
- **An `examinees` root** — aggressive mode: deep-window homologs below
  the pool best; the lowest confidence, and a known over-expansion
  signal in finely identified brain regions. Leaves carry the same
  `{type}>{src}` / `(no_source)` token plus `(dup)` when a bodyId recurs.
- **A `low`/`medium` member in the Reciprocal tab** — the branch's own
  source type ranks weakly (or not at all) in the member's reverse scan
  (§2.2c). Read the top-N hover for who actually wins and where it
  lives; nothing in the pipeline changes because of the label.
- **Rows reading *not checked*** — the reciprocal pass was off, or the
  member sat beyond `--backward-max-neurons` /
  `--backward-per-branch-cap`. Raise the caps when the reciprocal
  coverage is the deliverable.
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
  fill-proposed / unpaired, and which target neurons of the mapped types'
  populations are **holes** (out-map bodyIds of in-map types — family
  material — never claimed by any branch pool, proposal, or
  morph-qualified candidate; the true unmapped residues, listed by
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
