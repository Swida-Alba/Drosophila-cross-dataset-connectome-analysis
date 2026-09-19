---
name: type-mapping-validation
description: "Run and interpret the DROCAT type-mapping validation pipeline — bodyId-level validation of cross-dataset type mappings (FAFB ↔ male-cns), with branch-resolved pools, the Rev 3.12 category partition (tier / sibling / candidates / family / relative / examinees — 'examinees' was renamed from 'suspicious' 2026-09-18; the mapper's rival-suspects concept owns that word now), per-bodyId leaf tokens ((out-map) / >src / (no_source) / untyped), the opt-in stage-5d reciprocal homolog evidence (high / medium / low / not-checked — advisory, connectivity-only, graded on how prominently the member's own branch source type ranks), nested modes (restrictive / family / aggressive), two-track morphology, 3D review scenes, and gap-fill proposals. WHEN: \"validate type mapping\", \"mapping validation\", \"check bodyId mapping\", \"run RunMappingValidation\", \"gap fill proposals\", \"type mapping candidates\", \"cross-dataset mapping check\", \"s-CPDN3 validation\", \"circadian clock mapping validation\", \"reciprocal homolog evidence\", \"backward evidence\"."
---

# DROCAT Type-Mapping Validation

Validate an automatic cross-dataset type mapping at **bodyId granularity**,
expand the highly suspected neurons the mapping missed, and render 3D
review scenes — the mapping itself is **never rewritten**. All commands
are shell commands executed from the **repo root**.

## 0. Quick start (from a fresh machine)

```bash
# 1. get the repo (the pipeline needs its datasets/ tables and cache/ stores)
git clone https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git
cd Drosophila-cross-dataset-connectome-analysis

# 2. Python env with navis/flybrains/flywire (see the DROCAT install skill)
PY=python3

# 3. validate two FAFB types against male-cns
$PY scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types s-CPDN3C,s-CPDN3D \
    --label my_first_run --scene-selfcheck
```

The pipeline cannot run from a copy: it reads `datasets/` (neuron tables)
and `cache/` (connectivity profiles). If the repo is not the current
directory, `cd` to it first; everything below is repo-relative.

## 1. Run a validation

```bash
$PY scripts/RunMappingValidation.py \
    --source flywire_FAFB_v783 --target male-cns:v1.0 \
    --types <TYPE_A,TYPE_B | coarse_cell_type> \
    --label <short_label> \
    [--mode restrictive|family|aggressive] \
    [--backward-evidence] \
    [--scene-selfcheck]
```

- `--types` accepts concrete types (`s-CPDN3C,s-CPDN3D`, `APDN3`) or a
  coarse `cell_type` (`circadian_clock`).
- ALWAYS pass `--scene-selfcheck` (verifies legend-leaf geometry against
  the neuron bbox; flags mislabeled renders).
- Add `--backward-evidence` when the **fill** is the question (§3): it
  runs the homolog finding in reverse over the `candidates` / `family` /
  `relative` members. Advisory only, connectivity only, default OFF.
- Long runs: launch in the background and poll the log. Expected wall
  times (warm caches): small pair ~2–3 min; 50-neuron family ~6 min;
  `circadian_clock` (242 sources) ~35–40 min. A stage-5d run adds roughly
  one forward scan's worth of work per member it actually scans (~7 s at
  FAFB↔MCNS scale; the r8 out-map expansion measured 207 s / 30 forward
  scans against the same-size universe) — the same price as a forward
  source scan, which is why the caps exist.
- Results land in `{output_dir}/type-map-validation_{SRC}_to_{TGT}_{ts}/`
  with SHORT dataset nicknames (`FAFB`, `MCNS`, `BANC`) and a
  `YYYYMMDD_HHMMSS` stamp — the `--label` is recorded in
  `parameters.json` / the report, not in the folder name. Default
  `output_dir` is `local_data/mapping_validation/`. Since
  2026-09-19 the evidence CSVs sit in stage subfolders — `validation/`,
  `expansion/`, `gap_fill/`, `mapping/` — and the root keeps only
  `report.html`, `README.txt`, `_UserGuide_please_read_me.*`,
  `parameters.json`, `set_coverage.json`, `morphology_calibration.json`,
  `pipeline_progress.jsonl`, `user_warning_notes.txt` (`visualization/`
  was already a subfolder). Readers fall back to the old FLAT layout, so
  pre-2026-09-19 folders still open and regenerate.

## 2. Read the outputs (always in this order)

1. `report.html` — the per-run report: headline + the three coverage
   levels (L1 claim / L2 provenance / L3 validation), branches, fills
   (with their per-row `reciprocal` column), the Reciprocal tab (stage
   5d, opt-in runs), out-map expansion, morphology record, scenes, file
   index. Hover any
   dotted term for its definition; every `!` log line is reproduced
   verbatim in its Warnings section, which also quotes the stage-5d
   `[reciprocal]` advisory (the Fill tab's `Reverse evidence by bin` line
   is its per-bin form). Regenerable for any past run:
   `python -m comparison.mapping_validation_report <run_dir>`.
2. `README.txt` — slim directions (what file is what) + the raw run log
   (scan the log lines starting with `!` for self-check failures or
   gate warnings). The old glossary / pair-summaries / coverage
   sections moved into `report.html`; `user_warning_notes.txt` mirrors
   the warnings in bracketed-tag lines.
3. `validation/pair_summary.csv` — per branch: pools, matched `M`, `gap`
   (informational), verdict/noise counters.
4. `validation/examinees.csv` (renamed from `suspicious_candidates.csv`) — the
   expansion rows. Key columns:
   `category` (the Rev 3.12 bin — see §3), `in_scope` / `morph_failed`
   (out-of-scope rows are connectivity-only, kept for reconciliation),
   `candidate_annotation` (`{T}(out-map)` / `{T}>{src}` /
   `{T}(no_source)` / `untyped`),
   `counts_toward_restrictive_fill` / `counts_toward_family_fill`,
   `ahead_size` / `pool_best_size` / `size_ratio`,
   `morph_v2_similarity` (Track A, query-based), `morph_pool_ref`
   (Track B, native pool reference), `pool_ref_tier`, and on a stage-5d
   run the `backward_*` columns (§3). The legacy
   `invader_class` / `invader_label` columns are retained for
   compatibility.
5. `validation/noise_filtered_candidates.csv` — dropped rows with `noise_reason`.
6. `validation/deep_candidates.csv` — only when `--mode aggressive`.
7. `gap_fill/gap_fill_proposals.csv` — proposals with `fill_class`
   (`in_pool`/`out_of_pool`), `category`, and the fill-count columns.
8. `gap_fill/gap_fill_dedup.csv` — query-level, one row per target bodyId:
   `dedup_category`, `n_branches`, `dup`, and the reverse verdict
   (`backward_evidence` + top-1). This is the deduplicated fill
   (the real gap-fill list). `gap_fill/gap_fill_levels.csv` is the same
   fill per branch with its confidence `level`; on a stage-5d run the
   reverse fact rides its `evidence` column (`backward_high` /
   `backward_medium` / `backward_low`) — never the level.
9. `expansion/family_candidates.csv` / `expansion/relatives.csv` — the whole `family` /
   `relative` bin (family/aggressive modes): enumerated members ∪
   evidence rows classified into those bins.
10. `morphology_calibration.json` — per-branch `candidate_thresholds`,
    `pool_ref_tiers`/`baselines`/`floors`, `track_a_null_bar`,
    `score_frame`, AUC gate record.
11. `expansion/source_status.csv` — backward `source-` status (matched / verified /
    borderline / unmatched, `source-` prefixed) per in-branch source,
    with its column rank and best pair. ADVISORY: column view of the same
    pair scores, for user reading only — never a gate.
11b. `expansion/source_candidates.csv` — OUT-OF-MAP sources (claimed by no
    branch) whose best-ranked hits land in a branch pool and pass the
    run null bar — the backward mirror of candidate admission.
    Advisory; renders as the scenes' `source-candidates` roots
    (re-aimed 2026-09-18: the earlier sibling-row derivation showed
    other branches' query neurons, which the sibling category already
    covers).
11c. `expansion/backward_matches.csv` — only with `--backward-evidence`:
    one row per (branch, scanned neuron) — the `candidates` / `family` /
    `relative` members first, then the UNMATCHED validated pool targets
    (`scan_role=pool_target`; matched / verified / borderline are never
    scanned — the symmetric forward score is their evidence) —
    with the reverse top-1 (`backward_top1_source_bodyId` / `_type` /
    `_in_branch`), the metrics + ranks, the evidence base
    (`backward_shared_type_count` / `_union_type_count` +
    `backward_thin_evidence`, see below), `backward_n_out_of_branch`, and
    the serialized `backward_topN` neighbourhood the report hovers (§3).
12. `visualization/*.html` — one 3D scene per parent type.

## 3. Interpretation rules (hard-won; do not improvise)

- **The category partition (Rev 3.12) — one ordered first-match per
  branch, mode-independent**: tier (`matched` > `verified` >
  `borderline` > `unmatched`) > `sibling` (in-map target of the query in
  another branch) > `candidates` (out-of-map, connectivity- AND
  morph-qualified — the restrictive fill) > `family` (out-map bodyIds of
  THIS branch's target type) > `relative` (candidate-type mates outside
  the map) > `examinees` (aggressive-only deep window). A target gets
  exactly one; the modes NEST.
- **One ordered per-bodyId leaf token on every expansion bin**:
  `{T}(out-map)` = the type is an in-map type, so this is an unmapped
  bodyId of a type already in the map (bodyId-level; every `family`
  member; takes precedence). Else `{T}>{src}` = foreign type with a real
  backward home (type-level). Else `{T}(no_source)` = foreign type with no
  usable route incl. hollow homes (type-level). Else `untyped`. The old
  `backward`/`hollow-backward`/`unmapped` classes are the compatibility
  mirror in `invader_class`.
- **Out-of-scope rows**: a connectivity-qualified suspect that fails the
  morph rule is `in_scope=False` / `morph_failed=True`, category blank,
  never rendered — it is the connectivity-only homolog-finding result.
- **Every rendered member passed the morph rule**: binding native
  pool-ref floor (`pool_ref >= floor`) when the branch has one, else
  the null-calibrated Track-A bar (`track_a_null_bar` = p95 of
  jaccard<=0.05 rows). Failures stay in the CSVs.
- **Hemisphere asymmetry fires the gap**: an L != R imbalance in any
  pool fires fill review even at arithmetic gap 0. `sibling` never
  counts; only `candidates` count toward the restrictive fill
  (`counts_toward_restrictive_fill`); `family`+`relative` join the
  family fill (`counts_toward_family_fill`).
- **`sibling` is not a fill**: an in-map target of another branch of the
  query (connectivity + morph qualified), already mapped.
- **`set_coverage.json` is the deliverable for "how much gap is
  filled"** — set-level FAFB assigned/proposed/unpaired and MCNS
  in-pool/candidates/holes (hole bodyIds listed per type); use
  `gap_fill_dedup.csv` for the bodyId-unique fill.
- **Fill accounting is query-scope-relative**: single-type queries
  report cross-type neighbors as expansion advice (never as fills);
  family/full-set queries account them via their own branches. Recommend
  per-type or family queries when the user asks "how much of the gap is
  filled".
- **`matched` is the only asserted tier**; verified/borderline are
  review tiers; all proposals are evidence — the mapping is never
  rewritten.
- **Backward `source-` statuses are advisory** (column view of the same
  pair scores; plan `plan-backward-source-status.md`): an unpaired
  source is typically a column runner-up of an already matched/verified
  target (population surplus + N-to-1 convergence), NOT a mapping
  failure. They never gate and never enter the dedup.
  `source_candidates.csv` lists OUT-OF-MAP sources whose best-ranked
  hits reach a branch pool (null-bar morph-qualified) — the true
  foreign-candidate mirror; in-branch sources never appear (they carry
  the `source-` statuses), and cross-branch convergence lives in the
  sibling category.
- **Stage 5d reciprocal evidence is ADVISORY and connectivity-only**
  (`--backward-evidence`, default OFF): every `candidates` / `family` /
  `relative` member — plus, by default, the UNMATCHED validated pool
  targets (matched / verified / borderline are never scanned; the
  symmetric forward score is their evidence) — is reverse-scanned against
  the WHOLE source
  universe with the same homolog scorer (`_backward_expansion_pass`, run
  AFTER `finalize_categories`). The CSV token IS the display label, and
  the three scanned grades measure how prominently the member's OWN
  branch source type ranks in the reverse scan (pure rank evidence — no
  score bar, no pool gate): `high` = top-1 by rank_union or jaccard;
  `medium` = within a top-3 of either; `low` = outside both top-3
  windows (a graded negative); `not-checked` = in-scope member not
  scanned — pass off, or beyond `backward_max_neurons` (300) /
  `backward_per_branch_cap` (40), with `backward_scanned_at`
  saying why (`disabled` / `cap` / `run` / `no_profile` / `error`). No
  morphology is re-scored — the counters state
  `morph = 'not evaluated (connectivity-only)'` — and the pass never
  changes `category`, any `counts_toward_*` flag, or a fill `level` (the
  reverse fact rides `gap_fill_levels.csv`'s `evidence` column; the
  `high/medium/low/type_gated/advice` ladder stays the morph-bar
  ordering). Scenes suffix a scanned leaf
  with `· high` / `· medium` / `· low`; a
  not-checked member keeps a bare leaf. Report: the **Reciprocal** tab
  shows ONLY the top-1 per member, with the full top-N in the hover.
- **Stage 5d changes the backward `source-` columns, not the fill**: the
  pool-member reverse scans (unmatched pool targets since 2026-09-19)
  yield each scanned pool target's column ranked over
  the whole source universe and are fed into `categorize_pool_sources`
  (`reverse_columns`). Before the pass, a forward column could only ever
  contain the branch's own sources, so `n_competitors` was structurally 0
  and `source-borderline` / `source-unmatched` were unreachable; an ON
  run therefore ranks a wider competitor set than an OFF run.
- **Direction cannot change a pair's score**: `score_one_candidate_fast(a, b)
  == score(b, a)` exactly (set + rank operations over the union), and the
  query path (`get_profile`) and the universe path (`build_target_vectors`)
  hand back identical vectors — rescoring every r9 + r11 pair from the cache
  parquet reproduced the stored forward `rank_union` on 272/272 pairs. So the
  pre-flight only builds MISSING profiles; rows whose cached
  `top_k_bodyid_used` is below the run's `top_k` are counted, not rebuilt
  (stats key `below_k`, counter `source_vectors_below_k` →
  `set_coverage.json` + a Reciprocal-tab line). That count is a **sparsity**
  note (81% of the FAFB and 87% of the MCNS below-k rows hold ≤ 5 partner
  types: `k_used = min(max_k, n_rows)`); refreshing them cost ~2,300 s per run
  and changed nothing, so never re-add the rebuild.
- **`rank_union` has a floor, not a zero point**: two vectors that share NO
  partner type score ≈ **-0.79 … -0.86** (measured; the floor rises with
  `|len(a) - len(b)|`), because missing keys enter the union as 0.0. A random
  FAFB×MCNS pair therefore sits at median ≈ -0.78, so `matched_ru_min = 0.1`
  is a "far above the disjoint floor" bar, not "10% correlated". Ranking
  *within* one scan is unaffected (the query vector is fixed, so the floor is
  near-constant across candidates); comparing `rank_union` ACROSS pairs is
  not meaningful without the correction `(ru - floor) / (1 - floor)`.
- **Every backward row states its evidence base**: `backward_shared_type_count` /
  `backward_union_type_count` plus `backward_thin_evidence` (top-1 resting on
  ≤ `THIN_SHARED_TYPE_COUNT` = 3 shared types — a reader's note, it gates
  nothing; not-checked rows carry `None` counts / False). Measured 2026-09-19:
  209 of 819 random pairs with `rank_union` above 0.1 rest on ≤3 shared types,
  0 of r9's 46 real in-branch hits. Padding short vectors was tested and
  rejected (cosine unchanged, shared-name pads inflate mean jaccard 0.30 → 0.80).
- **Frames**: scenes render in the source dataset's RENDER coordinates
  (FAFB/BANC native; male-cns/hemibrain raw→render bridged — sources
  AND targets share one frame); both morph tracks
  score in TARGET coordinates. Never read scene geometry as the
  scoring frame.
- **Untyped neurons**: `untyped` is a leaf token on whatever bin the
  neuron earns (a `candidates` leaf reads `{bodyId}_untyped`), never a
  peer category. Known noise populations (R1-R6 photoreceptors etc.) are
  structurally labeled by real crosswalks, not homolog claims.
- **Legend shape (Rev 3.12)**: the scene has ONE root per expansion
  category (`candidates` / `examinees` / `family` / `relative`); every
  bodyId LEAF — in all four — carries ONE ordered token (`{T}(out-map)` /
  `{T}>{src}` / `{T}(no_source)` / `untyped`) plus a standalone `(dup)`
  tag. `(out-map)` = an unmapped bodyId of an in-map type (bodyId-level,
  wins); the other two are type-level. Leaves under a root are sorted by
  `type + suffix`, not bodyId. `sibling` stays a single counted root. The
  legend panel is content-width up to 420px.

## 4. Scope switches (THREE NESTED MODES)

The modes are one ordered enum `restrictive < family < aggressive`; they
NEST — a neuron keeps the same category across modes, and each mode only
admits more neurons.

- **restrictive (DEFAULT)**: tier + `sibling` + `candidates`. Minimal
  expansion.
- **`--mode family`**: adds `family` (out-map bodyIds of each branch's
  target type) and `relative` (candidate-type mates). Expected: more of
  the type population surfaces; the tier is unchanged.
- **`--mode aggressive`** (legacy alias `--aggressive-expansion`): adds
  the deep-window `examinees` bin. Over-expansion prone (r36e review) —
  flag it clearly in any report.
- **`--verify-suspects`** (default OFF): advisory connectivity check of
  each queried same-name fan-out's rival candidates →
  `suspects_verification.csv` + the report Suspects tab. A rival that
  verifies well is a candidate ANNOTATION, not a mapping — inclusion
  goes through the custom label mapper. Held/excluded fan-outs land in
  `same_name_excluded.csv` (always written, advisory accounting).
- `--no-morphology`: fast structural pass (skips Track A/B); admission
  then falls back to connectivity-only.
- **Stage-5d switches** (all inert without `--backward-evidence`;
  defaults in `parameters.json`): `--backward-top-n` (5 — hits kept per
  neuron, the report's hover list), `--backward-max-neurons` (300 — the
  per-run budget of dataset-scale reverse scans),
  `--backward-per-branch-cap` (40), `--no-backward-pool-targets` (skip
  the unmatched pool targets — matched / verified / borderline are never
  scanned — which also leaves the `source-` statuses without
  pool-derived out-of-branch competitors), `--skip-backward-pass`
  (force-off).
  A neuron is scanned ONCE for all the branches that claim it, and the
  budget goes to the `candidates` / `family` / `relative` members FIRST —
  the pool-target control only gets what is left (unordered, a real run
  spent 102 of 120 scans on the control and capped out half the bins).
  Default
  OFF costs nothing and keeps the CSV headers stable (rows read
  `not-checked`).

Note (Rev 3.12): chain-aware POOL widening is RETIRED — family mode no
longer touches the tier; the former widening members are the `family`
bin. `parameters.json` records `validation_mode` / `mode_rank`.

## 5. Reporting checklist

When presenting results to the user:

0. `report.html` assembles items 1–6 of this checklist per run — open
   it first, then drill into the CSVs below when a number needs
   scrutiny.
1. Run folder path + self-check status (must be "all legend leaves
   match their neuron geometry" per scene).
2. Branch table: pools, M, gap, examinee/noise counts.
3. Category distribution (`examinees.csv` → `category`
   value counts), the out-of-scope count, and what it says
   (sibling-dominated = cross-branch convergence; a `candidates ...
   (no_source)` cluster = source-annotation gap).
4. Gap-fill coverage: the `gap_fill/gap_fill_dedup.csv` bodyId-unique fill by
   `dedup_category`, how many dedup rows are `dup`, and the
   `set_coverage.json` holes. On a stage-5d run also report the
   Reciprocal tab's per-bin `high` / `medium` / `low` counts WITH
   the budget line (scanned of eligible) — never present `not checked`
   as a negative result.
5. Morphology record: AUC (expect INACTIVE below 0.65), per-branch
   thresholds/floors/tiers, null bar.
6. Any `! self-check` or `! scene` log lines verbatim, plus the
   `[reciprocal]` own-branch-source-type top-3 line when the pass ran.

## 6. Tests and references

- Unit tests:
  `$PY -m pytest tests/core/test_mapping_validation.py tests/core/test_mapping_validation_r35.py tests/core/test_mapping_validation_backward_evidence.py tests/core/test_visualize_skeleton_legend_tree.py -q`
- Normative design: `_plan/plan-mapping-validation-rev311-chain-aware-pool-widening.md`
  §2b (the **Rev 3.12 category model**). Earlier revisions:
  `_plan/plan-type-mapping-validation-pipeline.md` (Rev 1–3.5),
  `_plan/plan-mapping-validation-rev36-invader-reclassification.md`
  (Rev 3.6–3.9).
- Technical report: `docs/technical/TYPE_MAPPING_VALIDATE_EXPAND_VISUALIZE_PIPELINE.md`
  (§4 is the category model).
- User guide: `docs/core-features/TypeMappingValidateExpandVisualize_Guide.md`.
- Export guide: `docs/OUTPUT_FILES.md` §9.

## 7. Failure playbook

| symptom | cause | action |
| --- | --- | --- |
| `profile cache parquet not found` | cold dataset | build profiles first (`ConnectivityProfiling.py`) or pick another target |
| scenes missing / `scene ... failed` in log | rendering error | check the traceback in the log; the CSVs are still valid |
| `! {root}: {bid} unavailable` | skeleton not cached and the dataset API is unreachable | the neuron stays in the CSVs but gets no scene leaf — expected offline; re-run when the API/cache is available |
| `Track-A null sample too thin` | small run | expected; Track-A falls back to the pooled-average bar |
| `track B (native pool reference) unavailable` | vector cache failure | Track B skipped; candidates fall back to the Track-A bar (thinner evidence — note it in the report) |
| type lookups return 0 for a NeuPrint dataset | API unreachable | `get_bodyids_for_type` / `get_types_for_bodyids` are offline-first — ensure the dataset has a repo-local neuron table under `datasets/` |
| self-check `!` lines | legend/render mismatch | treat as a bug: capture the lines verbatim and investigate before trusting the scene |
| `[stage 5d] backward evidence failed (advisory, skipped)` / `source-universe vectors unavailable` | the reverse pass could not build or scan the source universe | advisory layer only — bins, fills and levels are unaffected; every row stays `not-checked`. Re-run with `--skip-backward-pass` if only the tier is needed |
