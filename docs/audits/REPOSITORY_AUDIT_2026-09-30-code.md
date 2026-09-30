# Audit report — DROCAT comprehensive code audit (src/)

- Audit date: 2026-09-30
- Project root: `/Users/apple/Documents/GitHub/DROCAT-Drosophila-connectome-analysis-toolkit`
- Report: `docs/audits/REPOSITORY_AUDIT_2026-09-30-code.md`
- Status: `complete` (fix round executed in the same session — §Fix record)
- Verification round: a post-fix re-audit (three agents) of the eight fix commits found two P1 regressions the fixes themselves introduced and one phantom fix, all repaired before push — see §Verification round.
- Scope: the code layer end-to-end — `src/coana.py`, `src/statvis.py`, `src/morphology*.py`, `src/visualize_skeleton.py` + viewer caches, `src/comparison/*` (engine, profiler, mapper/resolver/metrics/morph), the FAFB/BANC converters + download layer, `src/utils/*`. Companion surfaces (docstrings inside these files, the UserGuide export glossary, backend-skill module references) updated in lock-step where the code fixes changed claims. Tree at HEAD `ced3da4` (start), fixes through `df2e14e` plus this record's later amendments.
- Method: `skills/repository-audit` workflow. Six parallel read-only inspection passes (two waves of three agents), each hunting undefined-name/wrong-variable defects, silent exception swallowing on data paths, cache/contract breaks, keying discipline, logic defects, in-file docstring drift, and dead code; lead-auditor line-level verification of every P1 and the risky P2s before fixing; targeted suites per cluster; full staged battery at the end.
- Prior audits: 2026-09-25 full-repo (all fixed through 4e12c44); 2026-09-30 UI/docs/report round (same day, all fixed through ced3da4). This round found no recurrence of the 09-25 P1 classes (pyflakes-with-wildcard-resolution reported zero undefined names in every audited file).

## Executive summary

**29 findings: 4 P1, 12 P2, 13 P3** (register rows; the original summary under-counted by bundling F-CP-002 into F-CP-001's narrative and two P3 rows carry multi-site fixes). The three P1s: a query-fingerprint mismatch that fell through to unchecked legacy files and served the previous query's edges as this run's results; `build_connectivity_profile_cache` — the entry point the tool itself prints in its error message — crashing with `TypeError` on every call (invalid config kwargs, a nonexistent `get_all_types` method, and its coana caller binding the config to the `datasets` parameter); and `_render_scenes` rebinding its own `members` parameter, silently dropping every 3D overlay scene on cross-dataset morph runs with ≥3 datasets or ≥2 queried types.

The dominant P2 patterns: **contract violations the code's own comments name** (the F9 all-post ratio denominator missed in the exported matrices; the `check_extrusions` default a WIP landing reverted against its own comment), **silently-lossy paths** (WebDriver profile exports without the filename dedupe both parallel loops apply; the http_get restart appending offset-0 bytes onto the old prefix on a range-ignoring 200; a corrupt main profile cache deleted instead of quarantined; fabricated top-k provenance passing the tier-1 cache gate), and **id-canonicalization gaps** (the FAFB connections converter skipping the 9-digit short-id expansion; the polars canonicalizer completing before stripping).

### Findings register

| ID | Sev | Area | Finding | Disposition |
|---|---|---|---|---|
| F-CA-001 | P1 | comparison_analyzer | Fingerprint mismatch falls through to unchecked legacy files (paths.csv / connection_info_*) from the same folder | fixed (`return None`) |
| F-CP-001 | P1 | connectivity_profiler | `build_connectivity_profile_cache` always TypeError (cache_dir/verbose kwargs); coana caller binds config to `datasets` | fixed |
| F-CP-002 | P1 | connectivity_profiler | `get_all_types` does not exist (3 sites) — homolog finders + default build crash | fixed → `list_types` |
| F-CA-002 | P2 | coana | Matrix ratio denominators use `min_weight=self.min_synapse_num` — the one pre-F9 site left | fixed (`min_weight=1`) |
| F-MO-001 | P2 | morphology | `check_extrusions` default reverted to True against its own comment (FAFB fetch cost) | fixed (False) |
| F-VS-001 | P2 | visualize_skeleton | WebDriver profile export lacks filename dedupe → silent profile loss | fixed |
| F-VS-002 | P2 | visualize_skeleton | `all_pairs` UnboundLocalError on FAFB branch when table read returns None (latent) | fixed (hoisted) |
| F-CA-003 | P2 | comparison_analyzer | Path exporters use plain folder builder → silently empty on re-export after grammar renames | fixed (`_resolve_dataset_output_path`) |
| F-CA-004 | P2 | comparison_analyzer | `_matrix_stores` persists across invocations, leaking prior types into matrices | fixed (fresh per call) |
| F-CP-003 | P2 | connectivity_profiler | First profile failure leaves `use_cache=False` for the whole build | fixed (finally) |
| F-CP-004 | P2 | connectivity_profiler | Corrupt main cache deleted, not quarantined (policy contradiction, data loss) | fixed (quarantine) |
| F-CP-005 | P2 | connectivity_profiler | Aggregates fabricate top_k=20/5 provenance → tier-1 gate accepts truncated sets | fixed (member max) |
| F-DP-001 | P2 | FAFB converter | Connections converter skips 9-digit short-id expansion (never joins tables) | fixed (shared expr) |
| F-DP-002 | P2 | FAFB converter | Import fallback retries the identical absolute import | fixed (relative) |
| F-DP-003 | P2 | banc_public_data | http_get restart corruption on IncompleteRead of a range-ignoring 200 | fixed (clear first) |
| F-MM-001 | P1 | morph_cross_dataset | `_render_scenes` rebinds `members` param → all overlay scenes lost (≥3 datasets / ≥2 types) | fixed (renamed local) |
| F-CA-005 | P3 | comparison_analyzer | Untyped-drop records duplicate on re-runs | fixed (reset) |
| F-CA-006 | P3 | comparison_analyzer | Path presence matrix True/0 mixed typing; dead mid-threshold default writes | fixed (int; dropped) |
| F-CA-007 | P3 | comparison_analyzer | Hemisphere filter creates missing dataset columns as NaN | fixed (guarded) |
| F-CA-008 | P3 | comparison_analyzer/profiler | Stale label_mapper comments; consolidate(None) docstring overstates | fixed (text) |
| F-CS-001..004 | P3 | coana/statvis | Hop docstring 3-tuple; density-dir per-query claim; engine-schema parity claim; dead `PrintROIHierarchy` | fixed |
| F-MO-002..009 | P3 | morphology | NBLAST step label unreachable; FAFB-bundle comment vs value; custom-group self-pair 1.0; dead `_read_fafb_zip_skeleton` + non-V2 cache helper; `n_per_type` docstring; `.swc.zst` sibling scan | fixed |
| F-VS-003..009 | P3 | viewer/caches | `_temp_main_figure` warning leak; identical if/else; headless docstring; layer-CSV docstring; dead per-point color branch; nb-cache layout docstring | fixed |
| F-DP-004..011 | P3 | converters/utils | Non-atomic post counts; existence gate; PID-only temps ×4; Content-Length KeyError; canonicalize order; dead BANC arm; `banc-v888` alias; console docstring | fixed |
| F-MM-002..012 | P3 | mapper/morph | `taxonomy_mapped` status; metrics docstring lies; tie-safe claim; NetSimile range; rank docstring drift ×3; stats double-count; bridge-search hoist; None-floor guard; partner-failure sentinel; Jaccard convention divergence; dead locals | fixed |
| F-CA-009 | P3 | comparison_analyzer | Standard vs combination similarity engines render different metric sets under one section name | documented (audit + docstrings; behavior unchanged — numbers feed published comparisons) |
| F-CS-005/006 | P3 | coana/statvis | `real_layer_map` dead parameter chain (+ misleading progress log); `edges_in_paths_with_layer` dead param (replay passes wrong shape) | **deferred** — removal spans 3 signatures + ~150 lines |
| F-VS-010 | P3 | visualize_skeleton | ~140-line dead NeuPrint mesh-cache block + retired helpers | **deferred** — mechanical but wide (tests describe helpers as retired) |
| F-CA-010 | P3 | comparison_analyzer | `_generate_combination_html_content` orphaned (~140 unreachable lines); `suggest_threshold_combinations` zero-caller claim unverified against the OUTPUT_FILES row | **deferred** — verify writers before deleting |

## Validation record

- Per-cluster targeted suites green after each commit: profiler 89 + analyzer 142, coana/statvis pathfinding 311+117+138, morphology 379+162, viewer 494+9, converters/utils 106+53+63, mapper/morph 227+47+165+33. pyflakes: zero new warnings (only the pre-existing baseline classes).
- First full-battery run caught 4 regressions from the fix round itself, all repaired before the record was finalized: the fafb_utils dead-arm deletion removed two names its own error message still used (restored as FAFB-only bindings); banc_public_data used `temp_sibling` without importing it; and the J6 `xfail(strict)` test XPASSed — because the P1 fingerprint fix makes the mismatch re-run actually re-derive and rewrite, the round-7 open contract was flipped to a hard assertion (it passes).
- Final full staged battery after ALL repairs, including the verification round (2026-09-30 21:32): **3/3 stages clean — core 5,166 passed / 1 skipped, ui 1,092 passed, docs-and-misc 176 passed** (over the pre-round baseline: +1 is the J6 contract flipped to a hard assertion, +2 are the verification round's regression pins).
- Bug-documenting tests rewritten to pin the FIXED behavior (`test_build_connectivity_profile_cache_config_bug`, homolog-finder `list_types` patches, the FakeProfiler `datasets` kwarg, the parquet gate test, the cache-factory test).

## Residual uncertainty

- `suggest_threshold_combinations`/`suggested_threshold_combinations.json`: OUTPUT_FILES documents the file as written; before deleting the "zero-caller" function, its writer path needs tracing (deferred with F-CA-010).
- The deferred dead-code blocks (F-CS-005/006, F-VS-010) are behavior-neutral; removing them is cleanup, not correctness.

## Verification round (pre-push, same session)

Three verification agents re-audited the eight fix commits hunk-by-hunk. Findings, all repaired before push:

1. **P1 (introduced by F-CA-001's fix)**: the blanket mismatch `return None` broke the DEFAULT replay-loader lane — the FNC replay rewrites `paths.csv` for the new query BEFORE the load, so there the fall-through was load-bearing; the blanket None made a mismatched same-folder re-run silently produce EMPTY exports. The miss is now caller-conditional: the pre-derivation probe (`remove_stale=False`) returns None (the original P1 stays fixed); the post-replay loader (`remove_stale=True`) removes the stale pair and falls through to the freshly written files. Pinned by `test_mismatch_verdict_is_caller_conditional`.
2. **P1 (introduced by F-CA-004's fix)**: the `_matrix_stores` reset landed at PAIR-loop depth, so with ≥3 datasets the exported matrices kept only the last dataset pair. The reset is hoisted above the pair loop (fresh per call, original store shape).
3. **P2 (phantom fix)**: F-CA-006's presence-matrix changes (int presence; dropped dead mid-threshold default writes) were lost when their patch script aborted before writing — the commit message and this record claimed them. Applied for real now and verified on disk; the pinning test updated.
4. Low items: the dead `is None` guard after the `list_types` swap (`if not neuron_types` with a message); the polars short-id gate narrowed to ASCII `[0-9]` (exact mirror of the python check); the dead `c_val` assignment after the per-point branch removal; the two mapper dead locals actually removed this time (`disagreement_rows` — whose first botched cut broke the file and was repaired — and `has_canonical_as_base`); `CrossDatasetComparison_Guide.md`'s NetSimile range and tie-safe claims synced; a dash-fold alias pin added; this record's counting and end-hash corrected.
5. Residual (accepted): stale-pair removal is best-effort — a crash between removal and the replay's rewrite leaves run-1's legacy files loadable by a later resume's probe if `connections_edge.csv` is gone (narrow window; pre-fix the same folder served stale data immediately).
