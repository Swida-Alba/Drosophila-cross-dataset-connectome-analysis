---
name: type-mapping-validation
description: "Run and interpret the DROCAT type-mapping validation pipeline — bodyId-level validation of cross-dataset type mappings (FAFB ↔ male-cns), with branch-resolved pools, the Rev 3.12 category partition (tier / sibling / candidates / family / relative / examinees — 'examinees' was renamed from 'suspicious' 2026-09-18; the mapper's rival-suspects concept owns that word now), per-bodyId leaf tokens ((out-map) / >src / (no_source) / untyped), the stage-5d reciprocal homolog evidence (DEFAULT ON since 2026-09-26 — opt out with --no-backward-evidence / the UI checkbox) (high / medium / low / not-checked — advisory, connectivity-only, graded on how prominently the member's own branch source type ranks), nested modes (restrictive / family / aggressive) plus the PARALLEL unsupervised `pooling` mode (absolute volume floors, morphology as the last gate, the mapper joined only afterwards), two-track morphology with per-branch bar kinds (native / Track-A backup / null) and a persisted target-vector store, 3D review scenes, and gap-fill proposals. WHEN: \"validate type mapping\", \"mapping validation\", \"check bodyId mapping\", \"run RunMappingValidation\", \"gap fill proposals\", \"type mapping candidates\", \"cross-dataset mapping check\", \"s-CPDN3 validation\", \"circadian clock mapping validation\", \"reciprocal homolog evidence\", \"backward evidence\", \"pooling mode\", \"unsupervised homolog scan\"."
---

# DROCAT Type-Mapping Validation

Validate an automatic cross-dataset type mapping at **bodyId granularity**,
expand the highly suspected neurons the mapping missed, and render 3D
review scenes — the mapping itself is **never rewritten**. All commands
are shell commands executed from the **repo root**.

## 0. Quick start (from a fresh machine)

> **Prefer the UI?** The same pipeline runs from the **Cross-Dataset › Type
> Validation** tab (`ui/tabs/type_validation.py`, tool key
> `type_mapping_validation`) — pick source/target, add type or coarse
> `cell_type` chips, choose the mode, toggle the stages. It drives this exact
> `MappingValidator` and writes the identical run folder. The tab speaks the
> dataclass *field* names; the CLI↔UI mapping is documented in the user guide
> §1b. Script usage is below.

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
    [--mode restrictive|family|aggressive|pooling] \
    [--no-backward-evidence] \
    [--scene-selfcheck]
```

- `--types` accepts concrete types (`s-CPDN3C,s-CPDN3D`, `APDN3`) or a
  coarse `cell_type` (`circadian_clock`).
- ALWAYS pass `--scene-selfcheck` (verifies legend-leaf geometry against
  the neuron bbox; flags mislabeled renders).
- The reciprocal pass runs by default (user 2026-09-26); add
  `--no-backward-evidence` to skip it when the **fill** is not the
  question (§3): it
  runs the homolog finding in reverse over the `candidates` / `family` /
  `relative` members. Advisory only, connectivity only, default OFF.
- Floors-v3 knobs: `--morph-track-a-offset` (Δ, default 0.05 — candidate
  bar = branch Track-A pool baseline − Δ; suspicious bar = baseline − k·Δ),
  `--morph-suspicious-level` (k, default 3, aggressive deep window),
  `--pool-ref-floor-margin` (the native floor's margin below the pool's mean
  pairwise similarity), `--out-map-top-k` (10 — out-map expansion keeps the
  top-k connectivity-ranked typed targets per unpaired source),
  `--max-scenes` (0 = one scene per parent type), `--skip-out-map-expansion`,
  `--skip-profile-build`, `--output-dir` (relocates the run folder, e.g. a
  shared data root), and `--skeleton-fetch-workers` /
  `--skeleton-fetch-timeout` (default 8 threads / 120 s of socket
  inactivity), which bound the stage-5 skeleton pre-flight — the run's only
  network-bound block. A run's `parameters.json` spells every knob by its
  config FIELD name (`morph_track_a_offset`, `morph_suspicious_level`,
  `pool_ref_floor_margin`, `out_map_top_k`, `max_scenes`), so the flag list
  and that file are the same knobs in two notations.
  Pooling's own: `--pooling-bar-metric` (`either|jaccard|rank_union`, the
  admission bar), `--pooling-bar-top-n` (3), `--pooling-jaccard-floor`,
  `--pooling-rank-union-floor`, `--pooling-window-mult` (the last three are
  ADVISORY flags now) and `--pooling-max-morph-targets` (0 = auto: 3 scoring
  units per queried source). Morphology is mandatory in this mode, so
  `--no-morphology` beside `--mode pooling` is a usage error, as is a target outside FAFB / male-cns / BANC.
- Stage 2 pre-flights the TARGET profile cache: typed neurons missing
  from the cache are built through the profiler backend (cache-first,
  resumable, ~125 neurons/s measured on banc_v888; fail-open to
  cache-only on any error). `pipeline_progress.jsonl` in the run folder
  carries machine-readable progress — the backend log a UI tails.
- The out-map expansion (on by default) scans every unpaired source
  neuron against the full target universe and exports the top-k TYPED
  non-in-map candidates to `expansion/out_map_expansion.csv`
  (connectivity-ranked, then morph-checked vs the run null bar — the row
  carries `morph_bar` + `morph_bar_kind` beside the score, so its
  `morph_qualified` ✓/✗ recomputes from the file).
- Long runs: launch in the background and poll the log. Expected wall
  times (warm caches): small pair ~2–3 min; 50-neuron family ~6 min;
  `circadian_clock` (242 sources) ~35–40 min. A stage-5d run adds roughly
  one forward scan's worth of work per member it actually scans (~7 s at
  FAFB↔MCNS scale; the r8 out-map expansion measured 207 s / 30 forward
  scans against the same-size universe) — the same price as a forward
  source scan, which is why the caps exist.
- Read the cost from `pipeline_progress.jsonl`, not from wall-clock
  folklore — every stage closes with its duration. Per-block costs
  measured on this tree: the whole target universe is expanded and indexed
  in ~5 s; the scan loop runs at ~0.14 s per source neuron (28× the
  per-pair scorer it replaced, on a 103,770-target universe); stage 5's
  skeleton pre-flight pays ~3 s per COLD target and 0 s warm; scenes are
  ~10–27 s per parent type and are the first thing to budget for once the
  caches are warm; a morph pass costs ~4 s per attempted candidate target
  COLD and about half of that warm, because the persisted target-vector
  store (§4b) skips skeleton load + render transform + vectorization
  (0.412 s measured per neuron).
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

## 1b. Run from the UI

The **Cross-Dataset › Type Validation** tab runs this same pipeline (tool key
`type_mapping_validation`; run folders still `type-map-validation_*`). Defaults
there: source = the Settings default dataset, **target = flywire_FAFB_v783**
(deliberately independent of the Settings default target, which belongs to
Connectivity → Find Homolog). Mode, morphology / scenes / backward toggles and
the advanced gates mirror `MappingValidationConfig`. **Max Scenes defaults to 0
= one scene per parent type**; a positive value clamps to the largest pools and
then names every dropped parent in the run log and in the report's Branches
tab, because a silently uncapped-out parent has no review scene at all.

**How the scenes look is a config field, not a code edit.** The collapsed
**Advanced Visualization** card (the shared
`ui/components/skeleton_visualization_settings.py` panel, scoped) sends
`scene_viz` (skeleton mode / background / simplification / export) and
`scene_category_colors` (one color per legend category); `neuron_alpha` is
written by its Neuron Opacity control and stays a first-class field with one
owner. The color rows are the Skeleton layer editor's own cell shape:
`ui/components/palette_picker.py::category_color_editor` renders one swatch
preview per category and opens ONE shared `color_picker_popup` (Bokeh palettes,
color grid, opt-in alpha) through a pending-category pointer — so 14 rows do not
build 14 dialogs, and the popup's DOM id is derived from the editor label. An
explicit alpha overrides the global opacity for that category alone.
The dataclass defaults are `None` and the UI runner prunes `None` keys,
so a CLI run without the scene flags renders exactly as it did before they
existed. Stage 4 merges them in
`resolve_scene_colors` / `scene_render_kwargs`, which refuse `legend_mode`,
`brain_mesh`, `skip_synapse` and the layer identity — the tree legend, the
coordinate frame and the injected layers are what the scene IS — and resolve the
panel's `None` "method default" simplification here, because custom (injected)
layers bypass the renderer's fetch-time default and a raw `None` raises. The same
guard covers a caller that asks for `tube` with NO fraction at all (the renderer's
own default is `None`): it is given the method default, because the alternative is
stage 4 swallowing the exception and shipping a run with zero scenes. The CLI
takes the same two objects as `--scene-viz-json` / `--scene-colors-json`.
`parameters.json` records the look through `scene_styling_record`, which runs the
SAME two helpers — so the provenance names what the pages wore (resolved
fraction, full merged palette) rather than an echo of what was sent; with scenes
off there was no look to wear and the raw config is recorded. A run's legend is
therefore reproducible from its own
provenance. Recoloring stays PER CATEGORY across the whole run (the
`COLOR_ALIASES` map keeps `relatives` / `fill` / `out-map query` in step) —
comparability across scenes is the point of the palette.

## 2. Read the outputs (always in this order)

Three shipped tools do the mechanical part of this list — run them first, then
read by hand (the report is still the thing to look AT, and `report.html` opens
in a browser, not in a terminal):

```bash
# schemas vs the registry, the ladder, the partition, the pooling ledger, the
# report's structure, the per-stage cost table — for run folders or queue dirs
$PY scripts/verify_tmvev_run_exports.py local_data/<queue-dir> [more…]
# what differs between two runs (a before/after certification): cell diffs at
# 6 dp, verdict deltas, category transitions, the backdrop and window feed
$PY scripts/verify_tmvev_run_parity.py compare <runA> <runB> [--expect-identical]
$PY scripts/verify_tmvev_run_parity.py table local_data/<queue-dir>
# the cold/warm gate any morphometry cache must pass, on a private clone
scripts/maintenance/verify_tmvev_frozen_gate.sh [--target male-cns:v1.0] \
    [--types l-LNv,APDN3,s-CPDN3A]
# a whole matrix, sequentially, with a manifest, waiting for an idle machine
scripts/maintenance/run_tmvev_matrix.sh --targets "male-cns:v1.0,banc_v888" \
    --modes "restrictive,family,aggressive,pooling" --types circadian_clock
```

`local_data/` holds DATA — run folders, the frozen harness clone, logs. Tooling
lives in `scripts/` beside `RunMappingValidation.py`; a script left in
`local_data/` is a script that gets deleted with the data it was written
against — `local_data/` is disposable by design.

1. `report.html` — the per-run report: headline + the three coverage
   levels (L1 claim / L2 provenance / L3 validation), branches, fills
   (with their per-row `reciprocal` column), the Reciprocal tab (stage
   5d, opt-in runs), the **Homolog · forward** / **Homolog · backward**
   tabs (per-bodyId match sheets, user 2026-09-26 — one row per appeared
   source / target bodyId with the chain-best primary match, the union of
   the top-3 rank_union and top-3 jaccard neighbourhood on hover, the
   allocation the run's artifacts give the bodyId, and the morph
   qualification of the pair displayed when the run scored it), out-map
   expansion, morphology record, scenes, file
   index. Every table renders ALL rows in one scrollable table (the
   20-row "Show all" catch-all is gone, user 2026-09-26). The **Scenes**
   tab prints the palette the run actually wore
   (`_scene_palette_html`, one chip per `COLOR_EDITABLE_CATEGORIES` bin, each
   recolored bin naming the default it replaced, no block when nothing was
   recorded) — read it before concluding a scene's colors are the defaults. Hover any
   dotted term — or any table header, which explains its own column — for
   its definition; every `!` log line is reproduced
   verbatim in its Warnings section, which also quotes the stage-5d
   `[reciprocal]` advisory (the Fill tab's `Reverse evidence by bin` line
   is its per-bin form). Regenerable for any past run:
   `python -m comparison.mapping_validation_report <run_dir>` (runs that
   predate the homolog panels show those two tabs as absent-artifact
   notes until the pipeline rewrites them).
2. `README.txt` — slim directions (what file is what) + the raw run log
   (scan the log lines starting with `!` for self-check failures or
   gate warnings). The old glossary / pair-summaries / coverage
   sections moved into `report.html`; `user_warning_notes.txt` mirrors
   the warnings in bracketed-tag lines.
3. `validation/pair_summary.csv` — per branch: pools, `best` (the
   mutual-best 1:1 pair count; renamed from `matched` on 2026-09-24, because
   the tier `matched` is a different quantity), `gap` = smaller pool − `best`
   (informational),
   verdict/noise counters. The report's Branches tab shows **Mapped** =
   verified_strong + verified + borderline and measures its own gap
   against Mapped, since a source can carry a verdict without being
   paired; hovering either cell gives both numbers.
4. `validation/examinees.csv` (renamed from `suspicious_candidates.csv`) — the
   per-pair pool-edge expansion rows (`examinee rows=` in the pair log).
   NOT the `examinees` bin: that category lives in `deep_candidates.csv`
   (`deep_window` rows binned `examinees`, first present in family); this
   file's set is mode-invariant. Key columns:
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
6. `validation/deep_candidates.csv` — candidate-window rows below the pool
   best: the top-`rank_top_k` band in family/aggressive modes
   (`candidate_source='top_window'`), the wider band in aggressive
   (`'deep_window'`). Empty in restrictive.
7. `gap_fill/gap_fill_proposals.csv` — proposals with `fill_class`
   (`in_pool`/`out_of_pool`), `category`, and the fill-count columns.
   `side` says which dataset `bodyId` belongs to: a `source` row names the
   unpaired source with `proposal_bodyId` as its target, a `target` row
   spells the same relation the other way round (Track A resolves both
   through `fill_pair`, so either side can carry `morph_v2_similarity`).
8. `gap_fill/gap_fill_dedup.csv` — query-level, one row per target bodyId:
   `dedup_category`, `n_branches`, `dup`, and the reverse verdict
   (`backward_evidence` + top-1). This is the deduplicated fill
   (the real gap-fill list). `gap_fill/gap_fill_levels.csv` is the same
   fill per branch with its confidence `level`; on a stage-5d run the
   reverse fact rides its `evidence` column (`backward_high` /
   `backward_medium` / `backward_low`) — never the level. Only on
   `family` / `relative` / `unmatched` rows: a `candidates` row keeps its
   bar-kind `evidence` and carries the grade in `backward_evidence` alone.
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
11c. `expansion/backward_matches.csv` — the default pass writes it;
    one row per (branch, scanned neuron) — the `candidates` / `family` /
    `relative` members first, then the UNMATCHED validated pool targets
    (`scan_role=pool_target`; matched / verified / borderline are never
    scanned — the symmetric forward score is their evidence) —
    with the reverse top-1 (`backward_top1_source_bodyId` / `_type` /
    `_in_branch`), the metrics + ranks, the evidence base
    (`backward_shared_type_count` / `_union_type_count` +
    `backward_thin_evidence`, see below), `backward_n_out_of_branch`, the
    `backward_own_type_*` block (the branch-type hit the grade rests on,
    with `backward_own_type_via` naming the ranking that placed it there),
    and the serialized `backward_topN` neighbourhood the report hovers,
    listed in jaccard order (§3).
11d. `validation/forward_matches.csv` — the **Homolog · forward** panel's
    data (user 2026-09-26): ONE row per appeared source bodyId — assigned,
    fill-proposed, out-of-map or unpaired alike — with the published
    chain-best target (`primary_*`), the serialized `forward_topN`
    neighbourhood, `n_scanned`, and `scanned_at` (`run` / `no_profile` /
    `error`; silence with a reason, never a negative). Captured during
    stage 2 while the scan frames are in memory; the `morph_*` columns are
    display joins off the artifacts that already scored that pair — the
    report re-derives ✓/✗ offline from the branch bars, nothing is
    re-scored.
11e. `expansion/target_matches.csv` — the **Homolog · backward** panel's
    data (stage 5e, user 2026-09-26): ONE row per appeared target bodyId
    (pool members — INCLUDING matched / verified / borderline — plus
    expansion / out-map / proposal targets) scanned back against the WHOLE
    source dataset, the exact mirror of the forward panel. Carries
    `pool_category` + `pool_branches` (every branch claiming the target),
    the chain-best `primary_source_*`, the serialized
    `backward_topN_union` payload, `n_scanned` / `scanned_at`. No caps —
    the stage-5d caps belong to the evidence pass. Advisory display data
    only: nothing downstream gates on it.
11f. `pooling/pooling_candidates.csv` / `pooling_pool.csv` /
    `pooling_sources.csv` / `pooling_cross_validation.json` — `--mode pooling`
    only, three axes: every (source, target) pair the BAR admitted (with its
    `tier`, `bar_rank`, the three advisory flags and `verdict_for_pair`), the
    same pool deduped to one row per candidate TARGET on the ordering chain
    AFTER the morphology gate (`in_pool`, `n_rows_refused`, `tiers`, plus the
    mapper-only `verified_only` rows), one row per QUERIED SOURCE (its
    chain-best finding, `n_admitted` / `n_in_pool` / `n_refused`, and
    `no_finding` — a source that found nothing is named, never absent), and the
    post-hoc comparison with the mapper (`confirmed` / `type_miss` /
    `type_new` / `verified_only`, the `bar` block as run, `floor_flags`,
    `tiers`, `pool_per_source` with its band, the morph record with its
    `units` / `budget` / `capped` / `gate_applied` / `dropped_targets` /
    `vector_cache` (the target-vector
    store's ledger), the `reading_notes`).  `gate_applied` claims the PASS:
    true only when `scored > 0` — a pass that raised or graded nothing did
    not gate the pool, and its tiers are then unrefused, not passed.
    `morph_gate` names its absences apart, and only an explicit
    `scored`-below-bar refusal leaves the pool (the refused pair rows stay in
    `pooling_candidates.csv`); there is no cross-dataset agreement column
    (§4b).
12. `visualization/*.html` — one 3D scene per parent type.

## 3. Interpretation rules (hard-won; do not improvise)

- **The category partition (Rev 3.12) — one ordered first-match per
  branch, mode-independent**: tier (`matched` > `verified` >
  `borderline` > `unmatched`) > `sibling` (in-map target of the query in
  another branch) > `candidates` (out-of-map, connectivity- AND
  morph-qualified — connectivity being an invader, a gap fire, or in
  family+ the top-`rank_top_k` discovery window; the restrictive fill) > `family` (out-map bodyIds of
  THIS branch's target type) > `relative` (candidate-type mates outside
  the map) > `examinees` (aggressive-only deep window). A target gets
  exactly one; the modes NEST.
- **"The modes NEST" is a guarantee the code keeps, not a reading
  convention** — a wider mode may only ADD rows, and a row that survives may
  not change category. Two things used to break it (both measured on the
  2026-09-24 `circadian_clock` ladder, both fixed): the borderline and deep
  candidate windows drew on ONE per-source `deep_cap`, so aggressive spent it
  before its second metric was read and displaced 110/189 (male-cns) and
  262/408 (BANC) of family's rows — taking their `relative` bin and their
  `gap_fill_levels.csv` rows with them (153 → 63); and the null backdrop
  excluded the mode's own window, so its p95 moved across the ladder and
  re-graded any row near the bar. Each band now has its own budget, and the
  backdrop excludes only the invader feed and the pool. Verify on any new
  dataset pair: `deep_candidates`' `top_window` pairs, `relatives` targets
  and the level counts must be non-decreasing along the ladder, the
  `track_a_null_bar` must be identical in all three runs, and no shared row's
  `category` may move.
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
- **Pool basis is decided PER SIDE**: `selected_chain` is the best single
  derivation and may be a target-side-only hop (FAFB->BANC resolves through
  `banc_v888/fafb_cell_type`), which refines the target pool exactly while
  the source pool would stay at the whole type population; the source pool
  therefore takes any supported chain of the SAME endpoint that NARROWS it
  (`source_chain` records which). Subset only — never a widening, never a
  union, so the tier stays mode-invariant. Where no chain names the source
  neurons (the name-asserted types: DN1a / DN1pA / DN1pB / l-LNv) the pool
  legitimately stays `full population` — FAFB annotates neurons with OTHER
  datasets' names and never with its own `type`, so a same-name branch has no
  row-level source evidence by construction. The basis vocabulary is
  `cross_dataset_type_mapper.basis_is_row_evidence`, shared by the resolver,
  the report's basis buckets, the scene branch labels and the disjointness
  measure (a consumer that compares the basis with one literal misreads every
  other row-backed basis), and the type-mapping panel / mapping-CSV hover name
  the source-supplying chain too, so no two views of one mapping claim
  different bridges. The Coverage tab's `Source claim envelope` line states
  how many queried sources sit in a branch pool and how many are the out-map
  residue; a residue of 0 over a wide pool means nothing was left outside,
  not that nothing is missing.
- **Every rendered member passed the morph rule (floors v3 — one bar engine
  per branch, published in `morphology_calibration.json` → `branch_bars`)**:
  the binding native matched+verified floor (`pool_ref >= mean pairwise
  reference sim - native margin`, kind `native`) when the branch has >= 2
  scored refs; else the Track-A backup floor (kind `track_a_backup`,
  `morph_v2 >= B_b - Δ` where Δ is §1's `--morph-track-a-offset`);
  else the run null bar (kind `null`, `track_a_null_bar` = p95 of the
  jaccard<=0.05 rows). Aggressive adds the loose deep-window bar
  (`B_b - k*Δ`, k = `--morph-suspicious-level`, with the null p50
  `track_a_null_bar_lo` as fallback): between the two bars a deep row is an
  `examinee`, below both it stays out of scope. Failures stay in the CSVs;
  `bar_params` records Δ, k and the native margin for the run, and
  `candidate_kind` / `suspicious_kind` say which currency each bar speaks —
  a `native` bar compares `morph_pool_ref`, every Track-A bar compares
  `morph_v2_similarity`, so never read one against the other.
  NOTE the two engines spell their kinds differently: the per-BRANCH
  `branch_bars` engine (supervised bins) uses `native` / `track_a_backup` /
  `null`, while the per-SOURCE mapping-ref engine that POOLING grades with
  uses `native` / `track_a` / `null_bar`. One report can therefore print both
  spellings; match on the engine, not on the word.
- **One ordering, one key**: every bodyId-level "best" and "top-N" — the
  published target, the mutual-best pairing, a target's best source,
  gap-fill proposals, the out-map list, the reverse top-1, and the
  cross-dataset candidate tables — comes from the ordering chain
  (`body_id_resolver.order_by_chain` / `chain_key`): jaccard first,
  `rank_union` breaking a jaccard tie, bodyId last. The `*_rank` columns are
  the EVIDENCE the verdicts read off (competition style, so a jaccard tie
  block shares one rank), never the row order; each scan publishes a dense
  `chain_pos` so "top-N" means N rows. A rank-1 claim lifts only the row that
  PUBLISHES the winning partner: `verified_strong` needs one member with both
  claims, `verified` needs one on the published member, and a `rank_union`
  top-1 sitting on a different pool member stays evidence in
  `ru_top_target_bodyId` (r17 measured 46 rows on two half-claims, r18 2 more
  on an unpublished win). `rank_union` does not filter a bodyId candidate —
  it keeps its own positivity / margin gates and the `matched` bar.
- **Hemisphere asymmetry fires the gap**: an L != R imbalance in any
  pool fires fill review even at arithmetic gap 0. `sibling` never
  counts; only `candidates` count toward the restrictive fill
  (`counts_toward_restrictive_fill`); `family`+`relative` join the
  family fill (`counts_toward_family_fill`).
- **`sibling` is not a fill**: an in-map target of another branch of the
  query (connectivity + morph qualified), already mapped.
- **`set_coverage.json` is the deliverable for "how much gap is
  filled"** — set-level `source` assigned/proposed/unpaired and `target`
  in-pool/candidates/holes. The two blocks are named by ROLE, never by a
  hard-coded dataset; `source_dataset`/`target_dataset` say which (hole
  bodyIds listed per type); use
  `gap_fill_dedup.csv` for the bodyId-unique fill. A **hole** is an
  out-map bodyId of an in-map type — the `family` bin's population minus
  whatever a fill candidate reached — claimed by no branch pool, no
  counted proposal and no morph-qualified `candidates` row (a candidate
  is a claim, so it closes the hole). Per type the identity holds:
  `mapped_population − in_pool = reached_as_candidates_only + holes`.
- **Fill accounting is query-scope-relative**: single-type queries
  report cross-type neighbors as expansion advice (never as fills);
  family/full-set queries account them via their own branches. Recommend
  per-type or family queries when the user asks "how much of the gap is
  filled".
- **The panel and this pipeline publish ONE claim set**: the type-mapping
  panel's *Mapped neurons* counts only the pairs the decision adopted
  (`foreign_type ∈ mapping_target_types` — the same field that builds the
  branches here). Declined same-name rivals and unadopted
  `valid_split_evidence` fan-outs stay listed in the panel as disclosure
  rows and never enter the count; before this rule `circadian_clock →
  banc_v888` read 205 in the panel against 198 here.
- **The panel's `Out-map (in-map types)` column is NOT the `family` bin**: it
  is the received types' own populations minus the bridge-claim set (a real
  set difference, so a `full population` basis reads 0), published per
  dataset as the deduped union and per matched type as a PER-RECEIVED-TYPE
  figure summed over that row's accepted target types (a neuron an adopted
  branch reaches anywhere in that type counts as claimed).
  On `circadian_clock → male-cns:v1.0` it reads **15** (219 − 204, matching
  `set_coverage.json`'s `mapped_target_set` / `in_branch_pool`) where
  `expansion/family_candidates.csv` holds **11** rows — the pipeline lets a
  morph-qualified candidate close a hole and the panel has no morphology —
  and the per-branch rows sum to 17 because convergent sources name the same
  neuron twice. Coverage evidence, never a second mapping. Its CSV twins are
  the mapping export's `target_out_map` / `target_out_map_body_ids`
  (extended form only; the legacy 20-column contract is untouched).
- **Every parent type gets a scene by default** (`max_scenes = 0`): branch
  review is the point of the run, and the old default of 12 silently dropped
  the smallest-pool parents (9 of 21 on a circadian run). A positive cap
  clamps largest-pool-first and names each dropped parent in the run log and
  in the report's Branches tab, so a missing scene is always a disclosure,
  never a silence.
  **Viewing on macOS**: the scene HTMLs carry a kernel-set `com.apple.provenance` attribute (the plotly/kaleido writer subprocess creates them), and Chrome may answer `ERR_ACCESS_DENIED` on `file://` links from the report even though the files are intact and OS-readable — macOS TCC applies the creating subprocess's restrictions to the reader. The scenes are fine (the run's scene self-checks verify legend/geometry): serve the run folder instead — `python3 -m http.server 8791` inside the run dir, then open `http://127.0.0.1:8791/report.html` (relative scene links work) — or open the report in Safari, or grant Chrome Full Disk Access. The attribute cannot be stripped (`xattr -d` no-ops) and inode rewrites do not clear it.

- **`matched` is the only asserted tier**; verified/borderline are
  review tiers; all proposals are evidence — the mapping is never
  rewritten.
- **Backward `source-` statuses are advisory** (column view of the same
  pair scores; plan `_plan/checked/plan-backward-source-status.md`): an unpaired
  source is typically a column runner-up of an already matched/verified
  target (population surplus + N-to-1 convergence), NOT a mapping
  failure. They never gate and never enter the dedup.
  `source_candidates.csv` lists OUT-OF-MAP sources whose best-ranked
  hits reach a branch pool (null-bar morph-qualified) — the true
  foreign-candidate mirror; in-branch sources never appear (they carry
  the `source-` statuses), and cross-branch convergence lives in the
  sibling category.
- **Stage 5d reciprocal evidence is ADVISORY and connectivity-only**
  (default ON since 2026-09-26): every `candidates` / `family` /
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
  shows ONE ROW PER NEURON, ordered by the branch-type hit's jaccard (a
  row with no hit — every `low` — sinks below the rows that have one), with
  the branch-type hit
  (the grade's own evidence) beside the chain-best (Jaccard-first) top-1,
  and the full top-N in the hover.
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

## 3b. Mapping result quality — three coverage levels

The same run answers three different coverage questions. ALWAYS state
which level a number comes from (reference: the circadian_clock FAFB →
male-cns family run; a newer run's numbers supersede these, the LEVELS do
not):

| level | criterion | FAFB (of 242) | MCNS (of 219) |
| --- | --- | --- | --- |
| **L1 branch claim** | bodyId in a branch's refined source/target pool | 212 examined | 204 claimed |
| **L2 row-based bridge evidence** | a crosswalk ROW individually names the neuron; pooled-identity (same-name full-population) claims are name-asserted, row-less | 188 row-backed + 24 name-asserted | 180 + 24 |
| **L3 validation evidence** | bodyId-level connectivity+morph: `best` (mutual-best pairs) / asserted `matched` tier | 103 paired | 52 asserted |

- L1 is the mapper's claim set (IM): `242→204` is the claim envelope;
  204/219 is claim coverage of the in-map types' populations (the 15
  remainder are `family` material).
- L2 provenance splits the claim: **row-backed** (a crosswalk row names
  the bodyId via its `additional_type` / `flywireType` value) vs
  **name-asserted** (pooled identity: same-name pairs claim full
  populations with no per-bodyId rows — DN1a / DN1pA / DN1pB / l-LNv;
  s-LNv resolves via rows instead). This is why a wide `full population`
  pool is not a missing bridge (§3's per-side basis rule).
- L3 is the grading: `matched` is the only asserted tier;
  verified/verified_strong/borderline are review; assignments are
  **mutual-best 1:1** — N-to-1 convergence (8 l-LNv sources → 3 MCNS
  targets) leaves in-pool sources unpaired even at 100% pool coverage.
  Those unpaired are the scene's `out-map` review material only when they
  are OUTSIDE the pools (the 30); in-pool unpaired stay in `query` + tier
  layers.
- **Panel vs TM VEV**: the cross-dataset panel's "coverage 8/8 100%" is
  L1/L2 type-POOL coverage; TM VEV reports L3 bodyId pairing (l-LNv:
  panel 8/8, M=3). Both true at their own level. (`I-LNv` vs `l-LNv` is
  an ell/capital-I display artifact — FAFB has only `l-LNv`, 8 neurons.)
- **Bar kinds grade L3 admission per branch** (`branch_bars`): `native`
  floor (>= 2 scored refs) > `track_a_backup` (`B_b − Δ`) > `null` (run
  null bar). The null-kind bar is SAMPLE-DEPENDENT: it moved
  0.593→0.235 between identical runs (scored null rows 72→346) as the
  skeleton cache GREW. Since 2026-09-24 the backdrop no longer moves with
  the MODE (§3's mode-invariant exclusion), so any remaining run-to-run
  drift in a null-kind branch verdict is cache growth or a healed
  skeleton — read `input_fingerprint` before comparing two runs' bars.
- **A native score used to move between a run and its warm repeat, and that
  was a bug, not cache growth**: `SkeletonVectorCache.vectors_for` handed back
  cached rows STANDARDIZED by the cache's mean/std and rows it computed during
  the same call RAW, so a first run whitened a cold neuron against its pool in
  two spaces. Measured 2026-09-25 on FAFB→BANC pooling: 107 of 1266 rows and
  16 `in_pool` decisions moved; re-feeding exactly the ids the call had to
  compute reproduced every moved value to ~1e-16 (and the target-vector store
  was ruled out by A/B — arms with and without it agree to 16 digits). Fixed:
  one call returns one space (`space='standardized'` default, `space='raw'`
  for a caller that standardizes itself), the cold==warm invariant is pinned by
  `test_vectors_for_returns_one_space_cold_matches_warm`, and
  `input_fingerprint.morph_stores` now names the vector cache, whitener and
  skeleton store a verdict was read out of, KEYED BY PASS (`supervised`, and
  `pooling` in a pooling run — both name the same run-baseline store).
  **Read `morph_stores` before blaming a `morph_pool_ref` difference on the
  code.**

## 3c. Gap-fill estimation — under different confidence levels

`gap_fill/gap_fill_levels.csv` assigns every non-claim bodyId a confidence
level; the fill estimate = the level table + the source-side residue.
Estimation recipe (family mode recommended for gap questions):

1. **Claim coverage**: `set_coverage.json` → `target.in_branch_pool` /
   `mapped_target_set` (reference: 204/219 = 93%).
2. **Evidence fill**: `gap_fill_levels.csv` `high` + `medium` (+ `low`) —
   the morph-qualified candidates, one level per bar kind; `hole closer`
   notes flag candidates that close mapped-type holes (reference: 15 high
   + 1 medium = 16, of which 5 are hole closers). `set_coverage.json`'s
   `gap_fill_by_level` mirrors the table.
3. **Expansion advice**: `expansion/out_map_expansion.csv` —
   connectivity-ranked typed targets for the unclaimed sources (top
   `--out-map-top-k`, default 10 per source), where `morph_qualified`
   marks passes vs the run null bar — `morph_bar` (kind `null`) prints the
   threshold beside the score, so a ✓ is checkable without opening
   `morphology_calibration.json` (reference: R1-R6 fail 528/530, CB4091
   pass 132/132 — the column is what separates photoreceptor/orphan noise
   from same-family targets). A scene renders only the passing ones, under
   its `out-map candidates · {type}` branch beside the `out-map query ·
   {type}` residue.
4. **Source-side residue**: `set_coverage.source` assigned / proposed /
   unpaired — split SATURATED types (target populations smaller than the
   source's: annotation-bound, not fillable) from non-saturated types
   (evidence-bound).

Reference totals (circadian_clock): 16 candidates (15 high + 1 medium) +
10 family + 17 relatives = <= +43 beyond the 204 claims. For BANC: 3
candidates (all native), claims 198 of a 200-type-population denominator
(out-map 2, holes 0 — every BANC run on disk reads `in_branch_pool: 198`;
the 205 that once appeared here was the pre-`1e377c8` panel headline, which
also summed the 7 bodyIds of declined rivals and unadopted split fan-outs) —
the BANC gap is
annotation-bound, and BANC Track-A cannot confirm same-name pairs
(positives score below the null; the calibration artifacts that showed it
were kept under `local_data/morph-qualification-inspection/` and are no
longer there).

## 4. Scope switches (THREE NESTED MODES + ONE PARALLEL)

The modes are one ordered enum `restrictive < family < aggressive`; they
NEST — a neuron keeps the same category across modes, and each mode only
admits more neurons. `pooling` is a fourth `--mode` value that is NOT in
that enum (see below): it does not admit more of these bins, it answers a
different question alongside them.

- **restrictive (DEFAULT)**: tier + `sibling` + `candidates`. Minimal
  expansion.
- **`--mode family`**: adds the candidate-discovery window (the top
  `rank_top_k` of each metric, per source), `family` (out-map bodyIds of
  each branch's target type) and `relative` (candidate-type mates).
  Expected: more of the type population surfaces; the tier is unchanged.
  The window is what keeps `candidates`/`relative` populated when the pool
  already holds both global rank-1s — an invader cannot beat rank 1, so
  the bar alone would empty the feed at the strongest branches.
- **`--mode aggressive`** (legacy alias `--aggressive-expansion`): adds
  the deep-window `examinees` bin. Over-expansion prone (r36e review) —
  flag it clearly in any report. The deep band draws on its OWN per-source
  `deep_cap` budget, so adding it never removes the borderline rows a family
  run reported (§3: the modes nest).
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
  the unmatched pool members only get what is left (that ordering is
  enforced in code; measured before it existed, an unordered budget spent
  102 of 120 scans on the control and capped out half the bins).
  Default
  OFF costs nothing and keeps the CSV headers stable (rows read
  `not-checked`).

Note (Rev 3.12): chain-aware POOL widening is RETIRED — family mode no
longer touches the tier; the former widening members are the `family`
bin. `parameters.json` records `validation_mode` / `mode_rank`.

## 4b. Pooling mode (`--mode pooling`) — the parallel unsupervised engine

The nested ladder asks *what else belongs to the pool the type mapper
already asserted*. Pooling asks *what do connectivity and morphology say
is a homolog of the queried population, over the whole opposite universe*,
and only then compares that answer with the mapper. It is a fourth mode
VALUE, NOT a wider rung: it is absent from `VALIDATION_MODES` /
`MODE_RANK`, so a pooling run publishes `validation_mode: pooling` with
`mode_rank: null` (the key is always written; the mode simply has no rung), and `--mode pooling` with a widening flag is a usage error
(it does not nest, so that pair is a contradiction, not a precedence
question). It writes `pooling/` beside the nested bins and changes none of
them.

- **Its own three tiers, per ROW** (`pooling_candidates.csv`/`pooling_sources.csv`
  column `tier`, first-match down, and `pooling_pool.csv` carries the SET of tiers
  its admitting rows reached, e.g. `matched+nominated`): `matched` (the row is
  some metric's top-1 AND its `rank_union` clears `matched_ru_min`), `verified`
  (top-1 on either metric), `nominated` (ranks 2..N inside the bar). All three
  are morph-qualified by definition, which is why morphology is mandatory.
  `nominated` is POOLING-ONLY — it is in no `TIER_CATEGORIES`,
  `EXPANSION_CATEGORIES` or `DEDUP_RANK`, so it cannot widen the supervised
  partition — and pooling has no `borderline` and no `relative`: the deep window
  and the type-mate bins belong to the nested modes. A pooling `matched` is
  therefore NOT the mapper's asserted tier, despite the shared name.
- **The UI can start it; that does not make it a rung**: the mode row of
  **Cross-Dataset › Type Validation** is TWO CARDS ON ONE ROW — the nested
  ladder's three buttons in `card-tmvev-mode` ("Validation Mode") and
  `Pooling` alone in `card-tmvev-mode-pooling` ("Parallel mode") beside them,
  because the seam between the cards is what says the fourth is not one more
  rung. Selecting it reveals its own gate card (`card-tmvev-pooling` — its
  controls are the admission bar
  (metric + top-N depth), the two advisory floors, the window multiplier and the
  morph budget: there is no floor-fit checkbox, no corroboration option, and no
  checkbox turning morphology off — the gate is mandatory), and
  `ui/tabs/type_validation.py:MODE_OPTIONS` is `VALIDATION_MODES +
  ['pooling']` — pinned beside `POOLING_MODE not in MODE_RANK` by
  `tests/ui/test_type_validation_tab.py` (which also pins
  `LADDER_MODES == VALIDATION_MODES` and the 3 + 1 card split). The CLI route
  (`scripts/RunMappingValidation.py --mode pooling`) sends the same config.
- **Run it**: `--mode pooling [--pooling-bar-metric either|jaccard|rank_union]
  [--pooling-bar-top-n 3] [--pooling-jaccard-floor 0.10]
  [--pooling-rank-union-floor 0] [--pooling-window-mult 2.0]
  [--pooling-max-morph-targets 0]` (`--no-morphology` beside `--mode pooling`
  is a usage error, and so is a TARGET the morph scorer cannot score:
  `dataset_scope()` admits only FAFB / male-cns / BANC, so hemibrain is refused
  before stage 1 — a run that got past it published 32 `matched` and 335
  `verified` rows over 0 graded verdicts, because the pass raised inside stage
  P, the engine recorded the error per row, and the tiers stayed).
- **Admission is the BAR, per source (never pool-relative)**: each queried
  source keeps the top-N rows of `pooling_bar_metric` at depth
  `pooling_bar_top_n` (default `either`/3). `either` is the **UNION of each
  metric's own top-N**, never a merged best-rank ordering — a merged order
  spends the slots on the two metrics' rank-1 rows (162 of 242 sources have two
  distinct ones) and was measured keeping 79 of the 118 targets the old gate
  found where the union keeps 116. `rank_union` ties at exactly 0 across
  hundreds of targets, so a rank cut is not a row bound (17.1 rows/source at
  N=5 against jaccard's 5.1): each source holds at most `2N` rows in the chain
  order and `bar.rows_cut` says what the cap cut. The seed is every neuron the
  query names by the SOURCE dataset's own annotation, so the residue no branch
  claims is inside it. `pooling_cross_validation.json → bar` is
  `{metric, top_n, row_cap_multiple, rows_cut, role}`.
- **The floors are FLAGS now, and they remove nothing**: `jaccard >` floor,
  `rank_union >` floor and `window_mult × the size of that source neuron's own
  type population` are still evaluated per admitted row and exported as
  `below_jaccard_floor` / `below_rank_union_floor` / `outside_window`, with the
  counts in `floor_flags` and the configured numbers in `gate` (whose `role`
  line says they flag rather than filter). They were filters until 2026-09-24:
  the rank_union floor at its own default `0` is only a sign test, and that
  alone starved 119 of 242 queried sources — which is why RU is never a real
  bar (user, round 5). An earlier build ALSO fitted the Jaccard floor per
  dataset pair (`pooling_floor_from_evidence`,
  `--no-pooling-floor-from-evidence`, a persisted
  `cache/{target}/pooling/jaccard_evidence_{sha1(source)}.json`, and the
  `jaccard_floor_*` gate keys); all of that is DELETED (Decision
  2026-09-23, user: "the floor is only a safeguard to avoid explosion in the
  finding, not a data quality guard"). What motivated the fit stays as the
  reason it went: the same floor rejects very different shares of two dataset
  pairs' graded rows (0.20 rejected 2.1 % of male-cns's 241 graded pairs,
  36.6 % of BANC's 164, 20.0 % of hemibrain's 145; 0.10 rejected none) — but
  that measures POOL VOLUME, which is all a connectivity threshold may answer.
  To widen or narrow the pool, move the BAR (`--pooling-bar-top-n`,
  `--pooling-bar-metric`) and read `pool_per_source` back; the floors are for
  reading the run, not for driving it.
- **Morphology last, and it is a GATE**: the Find-Homolog fast path (no
  NBLAST) with the branch-free persisted `mapping_ref` bar, on the
  connectivity survivors only, under a budget counted in SCORING UNITS (a unit
  is a tier-1 row, or the chain-best row that owes a target its verdict). The
  default budget is AUTO: `3 ×` the queried source population
  (`--pooling-max-morph-targets 0`), because the constant 400 an earlier build
  used cut 146 of the 546 units the default bar needed on 242 sources — which
  cost one tier-1 claim and left 115 rows with no verdict at all. Either way
  `morph.budget` publishes where the number came from, beside `units` /
  `attempted` / `capped`, so `capped` is always checkable against a rule
  (measured need ≈ 2.3 units per source). A candidate whose chain-best
  row is `morph_gate='scored'` with `morph_qualified` false leaves
  `pooling_pool.csv` AND the scene's `pooling · {source type}` root; its
  per-pair rows stay in `pooling_candidates.csv` so the refusal is auditable,
  and the count is published as `morph.dropped_targets` beside
  `morph.gate_applied`, in the log too (`[pooling] … N refused by the
  morphology bar …`). Only an explicit refusal removes a target:
  `not-selected` is GONE: the hybrid scores one verdict per SCORING UNIT, so a
  row that is not its target's chain-best row reads that unit's verdict and says
  so — `shared` with the pair named in `verdict_for_pair`. A borrowed number is
  never this row's own measurement, and it is not a rejection either.
  `not-attempted-cap` (the budget refused to look), `no-score` (the scorer
  returned no value for the pair), `disabled` / `inactive` / `error` all stay
  in the pool and are named as such — none of them is a rejection, and a
  shrunken pool is never read as a smaller harvest.
  `pooling_cross_validation.json → morph` is the RECORD, and its counts do
  not mean the same thing: `attempted` is what the budget allowed to be
  looked at, `scored` is what came back with a verdict (`scored ≤ attempted`;
  one BANC run recorded `scored 8` over rows that said 1 `scored` and 7
  `no-score`), `qualified` is what cleared the bar.
  Report `scored/attempted`, never `attempted` alone — a mostly-unscored
  pool is a missing measurement, not a morphologically cleared one, and the
  report's Pooling tab plus `user_warning_notes.txt` both say so from one
  builder (`[pooling] … scored 4/8 attempted targets …`). `no-score` means
  the scorer had nothing FOR THAT PAIR, never "the neuron is not in the
  vector store": the collapsed `scored` counts behind those examples came
  from the shared `mapping_ref` scorer's pool-reference pruning (it dropped
  every branch-pool member as a bar-anchor, i.e. every candidate the mapper
  also claims — 35 of 41 rows on male-cns), which pooling now disables with
  `prune_pool_refs=False` in
  `comparison/morph_cross_dataset.py::qualify_visualized_pairs`.
- **A pooling verdict publishes the four columns that made it**:
  `morph_bar_kind` is the binding rule (`native` / `track_a` / `null_bar`),
  `morph_bar` is that rule's value, and the score it graded is
  `morph_pool_ref` for a native row and `morph_similarity` otherwise — so
  `morph_qualified` is always recomputable from the row beside it, and the
  Pooling tab's morphology cell prints exactly that pair with its kind. Do
  not read `morph_similarity` as the number a native row was gated on: it is
  the Track-A score, kept for the pair's own record. An earlier build
  published the per-source null bar under a `native` kind, which made 14 of
  41 scored rows read as a score below its own bar with a ✓ beside it — the
  gate had graded the right pair and the export could not show it.
- **Target-vector store** (`morph.vector_cache`): preparing one target neuron
  costs 0.412 s (skeleton load + render transform + vectorize), so the
  render-space vectors persist in
  `cache/<target>/find_similar/morphology/cross_dataset_targetvec_<space>.npz`
  and a repeat run over the same dataset pair reads them instead. Two rules
  make that safe rather than fast-and-wrong: a stored vector is reused only
  together with a known hemisphere (otherwise the render still happens, so
  `pair_side` cannot change), and every row is keyed on its skeleton's
  `(mtime_ns, size)`, so a healed or re-fetched skeleton is recomputed, never
  reused — a cache that could re-grade a pair is a different instrument, not a
  faster one. And rows are stored at FULL precision with a `vector_dtype`
  stamp: a `float32` store moved a score in its 8th decimal (162 differing
  cells on the parity gate), so `load()` refuses a file not stamped `float64`
  as a unit rather than upcasting its rows. Measured on the frozen harness,
  cold vs warm with the same query: **0 differing cells** across
  `pooling_candidates.csv` (139 rows) and `pooling_pool.csv` (40 rows), the
  same 40-target pool and the same single refusal, and stage `P` 132 s → 56 s.
  `loaded` / `stale_dropped` / `saved` / `targets` say which a run did (the
  Pooling tab prints them as *reused · computed · store rows*, because
  `saved` is the file's row count, not what this run prepared), and a cold
  run reports `loaded: 0`. The gate is
  `scripts/maintenance/verify_tmvev_frozen_gate.sh` (it drives a private
  `local_data/tmvev-harness/` clone, which is data and stays there).
- **Post-hoc cells** (`pooling/pooling_cross_validation.json`):
  `confirmed` (sits in a refined target pool), `type_miss` / `type_new`
  (the harvest: the mapper never named this neuron), `verified_only`.
  Two rules when you report them: `verified_only` is EXPECTED to be large
  (the supervised tier admits by pool membership, this gate by global
  rank) and is not a false-positive count; and no cell is a recall
  measure, because the two engines seed from different sets.
- **Advisory columns**: `size_nm3` / `size_universe_percentile` are labels,
  not bars (the nm³ medians differ by orders of magnitude between datasets).
  Neither ever gates a row. `target_type` is the TARGET dataset's own
  annotation, shown for review and used to name `mapper_cell` — the mode
  neglects it for admission, so treat it as a label too. There is no
  cross-dataset agreement column: `targets_corroborated` and
  `--pooling-corroborate-with` were DELETED (Decision 2026-09-23, user) —
  counting how many other TARGET datasets put the same (source bodyId,
  candidate target TYPE) pair in their pool counted how many neglected the
  type name the same way, which is not corroboration and grades nothing.
- **Where it reads**: `report.html`'s **Pooling** tab carries the whole
  result (the gate and the volume-role line, the four cells, the harvest by
  target type, the morph record with its `gate applied` / refused count,
  then one row per pooled target) — the
  nested tabs are empty on a pooling run BY CONSTRUCTION, so quoting "0
  candidates" from them is wrong. The tab's *Scored against* line names the
  stores the cells came from (`input_fingerprint`: git rev, target universe,
  mapper snapshot, and the morphology stores a native verdict was read out of —
  the target's V2 vector cache, whitener and skeleton count); before comparing
  two runs' cells, read it — cells from runs that read different stores are not
  the same measurement. The run's
  `README.txt` "Start here" list points at `pooling/` for a pooling run. In
  a scene the same pool is the
  `pooling · {source type}` legend root (plum), hosted by the parent group of
  the source that reached each target best; each leaf is tagged
  `{mapper_cell} · {tiers} · morph ✓/✗`. The root is drawn from the rows the
  export marks `in_pool=True`, so a target the bar refused on EVERY row is out
  of the picture — but it stays a row of `pooling_pool.csv` with
  `in_pool=False`, because the refusal is a count the file owes and the scene
  does not. A pool whose best source has a type no branch group covers cannot
  be hosted and is named by a `!` line.
  **The scene answers on the target axis by design** (user, 2026-09-25): a
  per-source node would need a fourth legend level for every scene in the
  product, and grouping by each row's own source multiplies the leaves ~4x
  (1045 pairs vs 222 pooled targets measured) at ≈0.45 MB of page HTML per
  leaf. The source axis is `pooling_sources.csv` and the report's
  **Per source** block, where the counts reconcile.
- A stage-P failure never aborts the run: it lands in
  `pooling_cross_validation.json` as `{"error": …}` and the CSVs come out
  empty.

## 5. Reporting checklist

When presenting results to the user:

0. `report.html` assembles items 1–6 of this checklist per run — open
   it first, then drill into the CSVs below when a number needs
   scrutiny.
1. Run folder path + self-check status (must be "all legend leaves
   match their neuron geometry" per scene).
2. Branch table: pools, Mapped (verdict-carrying sources) beside the
   mutual-best pair count, gap, examinee/noise counts.
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
| `! {root}: {bid} unavailable` | skeleton not cached and the dataset API is unreachable | the neuron stays in the CSVs but gets no scene leaf — expected offline; re-run when the API/cache is available. (A NeuPrint dataset's *folder* spelling used to cause this on every target: `hemibrain_v1_2_1` is a local namespace, the server only knows `hemibrain:v1.2.1`; remote fetches normalize that now, so both spellings run.) |
| `[stage 5] ! morphology scored 0 of N requested pairs` | no target skeleton was loadable for that dataset | that run's bars all degrade to the `null` kind with `track_a_null_n: 0` and its verdicts are connectivity-only — never report it as a morphology-qualified baseline. Stage 5 now pre-flights the pair frame's own targets (`[stage 5] skeleton pre-flight (ds): X/Y cached, fetching Z`); a `failed` count or the `capped at 2000` line says which pairs stay unscored (network/cache limit, not a scoring bug) |
| `Track-A null sample too thin (n=k)` | small run | expected; Track-A falls back to the pooled-average bar. `n=0` is NOT this — see the line above |
| `track B (native pool reference) unavailable` | vector cache failure | Track B skipped; candidates fall back to the Track-A bar (thinner evidence — note it in the report) |
| type lookups return 0 for a NeuPrint dataset | API unreachable | `get_bodyids_for_type` / `get_types_for_bodyids` are offline-first — ensure the dataset has a repo-local neuron table under `datasets/` |
| self-check `!` lines | legend/render mismatch | treat as a bug: capture the lines verbatim and investigate before trusting the scene |
| `[stage 5d] backward evidence failed (advisory, skipped)` / `source-universe vectors unavailable` | the reverse pass could not build or scan the source universe | advisory layer only — bins, fills and levels are unaffected; every row stays `not-checked`. Re-run with `--skip-backward-pass` if only the tier is needed |
| null-kind branch bars drift between runs (e.g. 0.593→0.235) | the run null bar is the p95 of the scored jaccard-window null rows, and the scored subset grows as the skeleton cache fills | expected cache-dependence (the MODE-dependence was removed 2026-09-24 — one backdrop now serves every rung, so a bar that moves between two runs of the SAME mode means the cache moved). Prefer branches with native/backup floors for admission decisions, and pin the null sample per dataset (future work) |
| expansion top hits are R1-R6 / orphan NaN-type neurons | unpaired sources in saturated types grab dense-profile fragments | the typed-only filter removes NaN types; R1-R6 fail the morph check (median ≈ −0.02), so `morph_qualified=False` keeps them out of the scene — treat them as noise evidence, never as candidates |
