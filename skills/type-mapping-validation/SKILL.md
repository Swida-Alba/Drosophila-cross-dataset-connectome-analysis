---
name: type-mapping-validation
description: "Run and interpret the DROCAT type-mapping validation pipeline — bodyId-level validation of cross-dataset type mappings (FAFB ↔ male-cns), with branch-resolved pools, the Rev 3.12 category partition (tier / sibling / candidates / family / relative / examinees — 'examinees' was renamed from 'suspicious' 2026-09-18; the mapper's rival-suspects concept owns that word now) and per-bodyId leaf tokens ((out-map) / >src / (no_source) / untyped), nested modes (restrictive / family / aggressive), two-track morphology, 3D review scenes, and gap-fill proposals. WHEN: \"validate type mapping\", \"mapping validation\", \"check bodyId mapping\", \"run RunMappingValidation\", \"gap fill proposals\", \"type mapping candidates\", \"cross-dataset mapping check\", \"s-CPDN3 validation\", \"circadian clock mapping validation\"."
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
    [--scene-selfcheck]
```

- `--types` accepts concrete types (`s-CPDN3C,s-CPDN3D`, `APDN3`) or a
  coarse `cell_type` (`circadian_clock`).
- ALWAYS pass `--scene-selfcheck` (verifies legend-leaf geometry against
  the neuron bbox; flags mislabeled renders).
- Long runs: launch in the background and poll the log. Expected wall
  times (warm caches): small pair ~2–3 min; 50-neuron family ~6 min;
  `circadian_clock` (242 sources) ~35–40 min.
- Results land in
  `local_data/mapping_validation/{src}_to_{tgt}_{label}_{ts}/`.

## 2. Read the outputs (always in this order)

1. `report.html` — the per-run report: headline + the three coverage
   levels (L1 claim / L2 provenance / L3 validation), branches, fills,
   out-map expansion, morphology record, scenes, file index. Hover any
   dotted term for its definition; every `!` log line is reproduced
   verbatim in its Warnings section. Regenerable for any past run:
   `python -m comparison.mapping_validation_report <run_dir>`.
2. `README.txt` — slim directions (what file is what) + the raw run log
   (scan the log lines starting with `!` for self-check failures or
   gate warnings). The old glossary / pair-summaries / coverage
   sections moved into `report.html`; `user_warning_notes.txt` mirrors
   the warnings in bracketed-tag lines.
3. `pair_summary.csv` — per branch: pools, matched `M`, `gap`
   (informational), verdict/noise counters.
4. `examinees.csv` (renamed from `suspicious_candidates.csv`) — the
   expansion rows. Key columns:
   `category` (the Rev 3.12 bin — see §3), `in_scope` / `morph_failed`
   (out-of-scope rows are connectivity-only, kept for reconciliation),
   `candidate_annotation` (`{T}(out-map)` / `{T}>{src}` /
   `{T}(no_source)` / `untyped`),
   `counts_toward_restrictive_fill` / `counts_toward_family_fill`,
   `ahead_size` / `pool_best_size` / `size_ratio`,
   `morph_v2_similarity` (Track A, query-based), `morph_pool_ref`
   (Track B, native pool reference), `pool_ref_tier`. The legacy
   `invader_class` / `invader_label` columns are retained for
   compatibility.
5. `noise_filtered_candidates.csv` — dropped rows with `noise_reason`.
6. `deep_candidates.csv` — only when `--mode aggressive`.
7. `gap_fill_proposals.csv` — proposals with `fill_class`
   (`in_pool`/`out_of_pool`), `category`, and the fill-count columns.
8. `gap_fill_dedup.csv` — query-level, one row per target bodyId:
   `dedup_category`, `n_branches`, `dup`. This is the deduplicated fill
   (the real gap-fill list).
9. `family_candidates.csv` / `relatives.csv` — the whole `family` /
   `relative` bin (family/aggressive modes): enumerated members ∪
   evidence rows classified into those bins.
10. `morphology_calibration.json` — per-branch `candidate_thresholds`,
    `pool_ref_tiers`/`baselines`/`floors`, `track_a_null_bar`,
    `score_frame`, AUC gate record.
11. `source_status.csv` — backward `source-` status (matched / verified /
    borderline / unmatched, `source-` prefixed) per in-branch source,
    with its column rank and best pair. ADVISORY: column view of the same
    pair scores, for user reading only — never a gate.
11b. `source_candidates.csv` — OUT-OF-MAP sources (claimed by no
    branch) whose best-ranked hits land in a branch pool and pass the
    run null bar — the backward mirror of candidate admission.
    Advisory; renders as the scenes' `source-candidates` roots
    (re-aimed 2026-09-18: the earlier sibling-row derivation showed
    other branches' query neurons, which the sibling category already
    covers).
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
4. Gap-fill coverage: the `gap_fill_dedup.csv` bodyId-unique fill by
   `dedup_category`, how many dedup rows are `dup`, and the
   `set_coverage.json` holes.
5. Morphology record: AUC (expect INACTIVE below 0.65), per-branch
   thresholds/floors/tiers, null bar.
6. Any `! self-check` or `! scene` log lines verbatim.

## 6. Tests and references

- Unit tests:
  `$PY -m pytest tests/core/test_mapping_validation.py tests/core/test_mapping_validation_r35.py tests/core/test_visualize_skeleton_legend_tree.py -q`
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
