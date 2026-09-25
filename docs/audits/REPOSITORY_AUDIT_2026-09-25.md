# Audit report — DROCAT (Drosophila Connectome Analysis Toolkit)

- Audit date: 2026-09-25
- Project root: `/Users/apple/Documents/GitHub/DROCAT-Drosophila-connectome-analysis-toolkit`
- Report: `docs/audits/REPOSITORY_AUDIT_2026-09-25.md`
- Status: `complete`
- Scope: full audit of the tracked tree (777 git-tracked files, branch `v4.5.0` at `eebf4d1`) — source (`src/`), scripts (`scripts/`), UI (`ui/`), tests (`tests/`), docs (`README.md` + `docs/`), skills (`skills/`, 5 skills), configuration & packaging (`pyproject.toml`, `requirements*.txt`, `pyrightconfig.json`, `config.json`, `.gitignore`, launchers). Worktree audited as-is: 11 files carry pre-existing uncommitted WIP edits (see §Residual uncertainty); these were recorded via `git status --short` before reading and were **not** modified.
- Exclusions (generated/local data, not audited): `cache/`, `local_data/`, `datasets/`, `outputs/`, `neuron_indexes/` (app-owned, except the tracked seeds), `archive/` (historical; audited only for launcher/installer reachability and obviously stale links), `morphology_benchmark_NBLAST_vector/`, `homolog_param_benchmark/`, `preliminary_ROI_sensitive_segmentation/`, `background_research/`, `_plan/` (untracked planning docs), `vispath-subproject/` (audited lightly; own packaging), external URLs.
- Method: `skills/repository-audit` skill workflow (inventory → code → docs → UI/exports → skills → traceability → validation). Six parallel read-only inspection passes + lead-auditor line-level verification of every P1/P2 claim.

## Executive summary

The repository is in substantially better shape than its size suggests: no committed secrets (committed `config.json` is a clean template; token-bearing `config_local.json` is correctly gitignored), all 41 standalone scripts' imports resolve, the full test suite collects cleanly (6,105 tests, 0 collection errors), the NiceGUI run/error plumbing is solid, and the TM VEV skill's 800 lines of operational claims verify against the tree point-for-point.

However, the audit found **58 issues: 5 P1, 17 P2, 36 P3**. The two most serious patterns:

1. **Silently swallowed NameErrors disabling documented behavior (2× P1).** `src/morphology.py:383` calls a function that is never imported (`normalize_flywire_body_ids`), and `src/comparison/morph_cross_dataset.py:2304` reads a loop variable `b` that does not exist in scope. Both are wrapped in `except Exception: return {}` / `pass`, so FAFB soma anchoring and bridged-scene neuron naming/tagging never execute — with tests asserting the broken behavior in the first case. `pyflakes` independently confirms both undefined names.
2. **A silent data-loss path and a packaging gap (P1 ×2 + P1 config).** The pandas fallback of the profile-cache consolidation drops an unreadable main cache without logging, overwrites it with batch-only rows, then deletes the batches (`src/comparison/connectivity_profiler.py:1664-1692`); and `pyproject.toml`'s `py-modules` list omits 19 top-level `src/` modules that packaged modules import unconditionally (`coana.py:80`), so a non-editable `pip install drocat` wheel fails at `import coana`.

Documentation drift is the largest bucket by count (broken links, one badly fence-unbalanced hub README that swallows three headings, a deleted verification script still given as a copyable command in three docs, and an output-column contract that contradicts the writer). The UI has one genuine stale-control bug (Network tab sends a greyed-out hemisphere filter to the backend, which applies it independently). Two skills share a broken token-import snippet. No P0 (security/data-destruction/blocking) findings.

### Priority findings (P1 + P2)

| ID | Severity | Confidence | Status | Area | Finding |
|---|---|---|---|---|---|
| F-CODE-001 | P1 | confirmed | open | Code | FAFB soma-position lookup always returns `{}`: undefined `normalize_flywire_body_ids` swallowed by `except Exception` |
| F-CODE-002 | P1 | confirmed | open | Code | Bridged-scene loop references undefined variable `b`; NameError swallowed, so neuron-name uniquification and source-dataset tagging never run |
| F-CODE-003 | P1 | confirmed | open | Code | `eval()` on data-derived path strings in hemisphere-conserved path filter |
| F-CODE-004 | P1 | confirmed | open | Code | Profile-cache consolidation (pandas fallback) silently discards an unreadable main cache, overwrites it, then deletes batch files |
| F-CONFIG-001 | P1 | confirmed | open | Config/packaging | `pyproject.toml` `py-modules` omits 19 top-level `src/` modules; non-editable install breaks at `import coana` |
| F-CODE-005 | P2 | confirmed | open | Code | Dead `_report_cache_fallback` referencing an undefined global — the documented silent-downgrade diagnostic never ships |
| F-CODE-006 | P2 | confirmed | open | Code | Diagonal type-block indexes desync from member list when a member lacks a matrix row (wrong-variable bug) |
| F-CODE-007 | P2 | confirmed | open | Code | NeuronBridge cache-load future exceptions swallowed; bodyIds silently vanish from match tables |
| F-CODE-008 | P2 | confirmed | open | Code | 63 bare `except:` clauses across `src/`, several on data-critical export paths |
| F-DOC-001 | P2 | confirmed | open | Docs | Three intra-doc links missing `../` prefix resolve to nonexistent siblings |
| F-DOC-002 | P2 | confirmed | open | Docs | `docs/core-features/README.md` unbalanced code fences: 3 real headings swallowed, 2 hub anchors dead, rendering mangled |
| F-DOC-003 | P2 | confirmed | open | Docs | `OUTPUT_FILES.md` claims `sim_roi` "no longer occurs"; `morphology.py` still writes it |
| F-DOC-004 | P2 | confirmed | open | Docs | Copyable commands reference deleted `scripts/PlotPath_TestNegatives.py` (known since 2026-09-07, unfixed) |
| F-DOC-005 | P2 | confirmed | open | Docs | Harness scripts referenced at `scripts/…` but live in `scripts/harness/` |
| F-DOC-006 | P2 | confirmed | open | Docs | Deleted test `tests/test_colabeling_similarity.py` cited as "complete example" |
| F-DOC-007 | P2 | confirmed | open | Docs | TROUBLESHOOTING TOC anchor `#flywire-data-download-issues` targets a renamed section |
| F-SCRIPT-001 | P2 | confirmed | open | Export/scripts | `verify_tmvev_run_exports.py` no-arg default audits stale `tmvev-clock-*` queues, not "the newest queue directory" |
| F-SCRIPT-002 | P2 | confirmed | open | Export/scripts | Tracked `scripts/preview/` one-off dev scripts hardcode a personal `/Users/apple/...` run folder |
| F-UI-001 | P2 | confirmed | open | UI | Network tab: disabled hemisphere controls keep stale values; greyed-out `hemisphere_filter` silently restricts backend results |
| F-SKILL-001 | P2 | confirmed | open | Skills | `drocat-usage` settings recipe imports nonexistent `get_access_token`/`get_cave_token` |
| F-SKILL-002 | P2 | confirmed | open | Skills | `drocat-backend` util-support reference has the same broken import snippet |
| F-CONFIG-002 | P2 | confirmed | open | Config/packaging | `requirements.txt` vs `pyproject` dependency drift (viz deps demoted to extra; 10 packages range- vs exact-pinned) |

### Minor findings (P3)

| ID | Area | Finding (confidence) |
|---|---|---|
| F-CODE-009 | Code | Dead local `source_candidates` + docstring promises a regroup the function never computes (`mapping_validation.py:5156-5166`, WIP file) (confirmed) |
| F-CODE-010 | Code | `src/plotting/` is an empty dead package (confirmed) |
| F-CODE-011 | Code | Pooling `gate_applied` can be `True` on a borrowed `no-score` verdict — the "graded nothing" case the new WIP comment excludes (`mapping_validation_pooling.py:952-954`) (inferred) |
| F-CODE-012 | Code | `vectors_for(space=...)` accepts any non-`"raw"` string as `"standardized"` silently (`morphology.py:3537-3581`) (confirmed) |
| F-CODE-013 | Code | `list(set(...))` process-dependent ordering feeds reciprocal-connection exports (`coana.py:17426`) (likely) |
| F-CODE-014 | Code | Annotation `"zipfile.ZipFile"` references never-imported `zipfile` (`morphology.py:1929`) (confirmed) |
| F-DOC-008 | Docs | Dead TOC anchors: `ScoreCalculation_Guide.md:13`, `VECTOR_V2_PIPELINE.md:58`, archive self-anchor (confirmed) |
| F-DOC-009 | Docs | Stale doc-to-doc refs in active docs: `HEATMAP_OPTIMIZATION_QUICKREF.md:133-134`, `SKELETON_DATA_PIPELINE.md:20-21`, `PATHFINDING_OPTIMIZATION_SUMMARY.md:162`, `VISPATH_IMPLEMENTATION_SUMMARY.md:20-22,245-247`, `SIMPLE_FORMAT_IMPLEMENTATION.md:297`, `CUSTOM_HEATMAP_ORDERING*.md` → deleted `scripts/test_custom_ordering.py` (confirmed) |
| F-DOC-010 | Docs | Deleted tests cited: `test_heatmap_clustering.py` (2 docs), `test_enhanced_edgelist.py` (confirmed) |
| F-DOC-011 | Docs | Dated `docs/UI_REDESIGN_AND_BACKEND_REPORT.md` references removed modules (`ui/tabs/find_homologs.py`, `statvis_polars.py`, `src/core/cache_manager.py`) — candidate for `docs/archive/` (confirmed) |
| F-DOC-012 | Docs | `OUTPUT_FILES.md` has two sections numbered 10; "§10" cross-references ambiguous (confirmed) |
| F-DOC-013 | Docs | `docs/core-features/README.md` duplicates the Connectivity Profiler section (confirmed) |
| F-DOC-014 | Docs | Seven overlapping CacheSystem docs (v1/v3/v4/proposal) linked side-by-side without deprecation markers (likely) |
| F-SCRIPT-003 | Scripts | Mac/Windows launcher parity drift: force-kill vs SIGTERM, busy-port autoreopen absent on Windows, wrong-Python handling, version fallback, repair cap (confirmed) |
| F-SCRIPT-004 | Scripts | `wait_then_full_suite.py` pins personal interpreter path + version-locked competitor regex; usage omits `--stages` (confirmed) |
| F-SCRIPT-005 | Scripts | `stage_own_hunks.py` argparse metavar (`PATH:SUBSTR`) contradicts actual `=` separator (confirmed) |
| F-SCRIPT-006 | Scripts | `generate_dataset_metadata.py` ImportError fallback duplicates the try-branch verbatim (confirmed) |
| F-SCRIPT-007 | Scripts | TM VEV auditor WIP: empty `{}` morph-store target can spuriously fail Track-A-only runs; `graded` rebound int→dict in one function (likely) |
| F-SCRIPT-008 | Scripts | Dead one-off: `scripts/harness/rerun_nb_colabel_timeout.py` (self-dated 2026-09-01); `flywire_fafb_v783_skeleton_recompression.py` judgment call (confirmed unreferenced) |
| F-SCRIPT-009 | Scripts | `preliminary_roi_sensitive_segmentation.py` writes a new repo-root directory (mitigated: gitignored) (confirmed) |
| F-UI-002 | UI | Disabled "Qualification mode" submits stale `mapping_ref` contradicting its tooltip (`connectivity.py:258-280,491`) (confirmed) |
| F-UI-003 | UI | `_export_matches_csv` reports any failure as "No matched entries to export" (`neuron_index_viewer.py:1558-1574`) (confirmed) |
| F-UI-004 | UI | `open_file`/`open_folder` swallow all failures; post-run Open buttons can no-op silently (`runner.py:1217-1257`) (confirmed) |
| F-UI-005 | UI | `mapping_store._save_all` is the only store without atomic write; crash mid-save loses all presets (`mapping_store.py:50-59`) (confirmed) |
| F-UI-006 | UI | Refresh recovery replays at most last 500 log lines despite full JSONL on disk (`run_state.py:303,504`) (confirmed) |
| F-UI-007 | UI | 2 s activity poll does synchronous manifest scans on the event loop per client (`app.py:3064`, `run_state.py:220-237`) (confirmed; severity unmeasured) |
| F-UI-008 | UI | Comparison sub-tab hides `use_cache` while sibling Find Homolog panel on the same page exposes it (`connectivity.py:589` vs `:153-156`) (confirmed) |
| F-UI-009 | UI | Provenance/threshold notices wrapped in `except Exception: pass` — incl. the only signal for empty-but-successful runs (`inter_dataset.py:751-765`, `type_validation.py:545-567`) (confirmed) |
| F-UI-010 | UI | Suggestion collectors swallow enrichment failures with no logging (`neuron_index.py:4908-4945,5168-5180`) (confirmed) |
| F-SKILL-003 | Skills | `drocat-usage` type-validation.md claims CLI sets "43 of the 53 fields"; actual 51 of 61 (confirmed) |
| F-SKILL-004 | Skills | `drocat-backend` util-support names nonexistent `apply_filter_mode` helper (confirmed) |
| F-SKILL-005 | Skills | TM VEV skill cites `plan-backward-source-status.md` pathlessly; it lives only under `_plan/checked/` (confirmed) |
| F-CONFIG-003 | Config | `pyproject.toml:125` comment claims `src/vispath.py` exists (it lives in the subproject) (confirmed) |
| F-CONFIG-004 | Config | No CI definition anywhere despite 6,105 tests (confirmed) |
| F-CONFIG-005 | Config | ~40 MB of parquet seeds tracked (documented deliberate policy in `.gitignore:21-36`; noted, appears accepted) (confirmed) |
| F-CONFIG-006 | Config | `pyrightconfig.json` includes untracked machine-local `archive/scripts_local` (confirmed) |

## Inventory and source-of-truth map

| Artifact family | Paths inspected | Source of truth | Derived/consumer artifacts | Coverage/notes |
|---|---|---|---|---|
| Code | `src/` 81 modules (33 top-level + `comparison/`, `utils/`, `core/`, `plotting/`) | `src/` modules | `ui/` tabs, `scripts/`, reports/exports under `outputs/`–`local_data/` | Full pyflakes/py_compile sweep + pattern sweeps + WIP-diff read; deep-read of flagged paths |
| Tests | `tests/` 252 files (6,105 tests) | `tests/` | pytest | Collect-only run (0 errors); marker & skip audit; no test-content diffing beyond cited cases |
| Documentation | `README.md` + `docs/` 185 md | `src/`/`scripts/`/`ui/` = observed truth | skills, users | 893 relative links + 245 repo-path references existence-checked; 15+ spot-checks of claims vs code |
| UI/HTML | `ui/` (app.py, 14 tabs, components, stores, runner) | `ui/` + `src/` | NiceGUI-rendered panels, run logs/manifests | Tab→handler→src call-path tracing for all 14 tool tabs |
| Templates/exports | run-report writers (`mapping_validation_report.py`, `report_kit.py`), `skills/repository-audit/assets/templates/` | writers + templates | HTML reports/CSVs under `local_data/` (excluded) | Writer↔doc↔skill cross-checks; exported files themselves excluded as local data |
| Skills | `skills/` 5 skills | `SKILL.md` + refs vs repo reality | agents, README pointers | All refs/scripts/assets/frontmatter checked; priority cross-checks vs CLI flags & export filenames |
| Config/build | `pyproject.toml`, `requirements*.txt`, `pyrightconfig.json`, `config.json`, `.gitignore`, launchers, `archive/install/` | pyproject + installers | wheels, conda env, launchers | Packaging diff vs `src/` listing; pin diff vs requirements; secret scan of tracked files |

## Audit sections

### Section 1 — Scope and inventory

- Inspected: repo root, `.gitignore`, `pyproject.toml`, tracked-file census (`git ls-files`, 777 files), untracked data dirs sized and classified.
- Evidence: `git status --short` (11 pre-existing WIP files recorded, preserved); dir census (`cache/` 19,980 files / `local_data/` 29,010 — untracked, excluded).
- Findings added/updated: F-CONFIG-005, F-CONFIG-006 (inventory-adjacent), F-SCRIPT-009.
- Unmatched information: none.
- Limitations: untracked data volumes excluded by design.
- Next: code audit.

### Audit checkpoint — code & executable behavior — 2026-09-25

- Inspected: all 81 `src/` modules via py_compile + pyflakes sweep; bare-except/TODO/hardcoded-path/eval/RNG/set-ordering pattern sweeps; close read of the 4 WIP `src/` files and their callers; repo-wide reference checks for dead-code claims.
- Evidence: `pyflakes src/morphology.py` → `383:17: undefined name 'normalize_flywire_body_ids'`; `pyflakes src/comparison/morph_cross_dataset.py` → `2304:43: undefined name 'b'`; line extracts at `coana.py:18210-18235`, `connectivity_profiler.py:1655-1695`, `neuron_search.py:1245-1269`, `morphology_comparison.py:500-519`, `neuronbridge_finder.py:4356-4374`; 63 bare `except:` sites enumerated.
- Findings added/updated: F-CODE-001…008, F-CODE-009…014 (F-CODE-001/002 personally re-verified via pyflakes + import/scope checks).
- Unmatched information: CODE_UNDOCUMENTED — `_report_cache_fallback`'s promised diagnostic exists nowhere (F-CODE-005); VALIDATION_GAP — tests assert the broken soma-lookup behavior instead of the contract (`tests/core/test_morphology_coverage.py:96-99`).
- Limitations: F-CODE-011 inferred from WIP code reading (not reproduced); F-CODE-013's end-to-end CSV ordering unverified.
- Next: documentation audit.

### Audit checkpoint — documentation & instructions — 2026-09-25

- Inspected: `README.md` + 185 `docs/` files; 893 relative links/anchors and 245 repo-path references existence-checked; 15+ claim-vs-code spot checks (quick start, output contracts, TM VEV defaults, rename hygiene).
- Evidence: fence-parity computation on `docs/core-features/README.md` (35 markers → unbalanced; headings at lines 158/334/448 inside fences); link extracts at `CrossDatasetComparison_Guide.md:317,367`, `NeuronBridge_Guide.md:677` (targets verified missing); `OUTPUT_FILES.md:380` vs `morphology.py:7222,8108` (sim_roi); `TROUBLESHOOTING.md:14` vs heading at `:437`; `find scripts -name PlotPath_TestNegatives.py` → nothing.
- Findings added/updated: F-DOC-001…007 (P2), F-DOC-008…014 (P3). No P1 doc findings: quick-start/install claims all verify (port 8080, env name, Python 3.10–3.11, class locations, output folder prefixes, TM VEV defaults, examinees rename).
- Unmatched information: DOC_NOT_IMPLEMENTED (deleted-script commands F-DOC-004), DEFAULT/NAME drift (F-DOC-003 sim_roi, F-DOC-005 paths), plus 13 archive-only stale refs (by design, not filed).
- Limitations: external URLs not audited; superseded-vs-current cache docs not content-diffed (F-DOC-014 kept "likely").
- Next: scripts & entry points.

### Audit checkpoint — scripts, launchers & entry points — 2026-09-25

- Inspected: all 41 `scripts/*.py` import contracts, argparse contracts, output-path hygiene; both launchers; `archive/install/`; `vispath-subproject` lightly; WIP `scripts/verify_tmvev_run_exports.py` audited as-is.
- Evidence: every script's imported src symbol verified to exist (0 broken imports); `RunMappingValidation.py` kwargs verified 1:1 against `MappingValidationConfig`; glob extract at `verify_tmvev_run_exports.py:848-850` vs `local_data/tmvev-*` listing (newest queues do not match `tmvev-clock-*`); launcher divergences at `mac_DROCAT.command:400-423,502-524` vs `windows_DROCAT.bat:272,166,68-72,31-33,103-104`.
- Findings added/updated: F-SCRIPT-001…009.
- Unmatched information: six unreferenced-but-living harness scripts (documented nowhere — noted, not filed as defects).
- Limitations: harness scripts' runtime behavior not executed; launcher drift severity judged from code only.
- Next: UI audit.

### Audit checkpoint — UI-linked & exported artifacts — 2026-09-25

- Inspected: `ui/app.py`, 14 tab modules, components, stores, runner; every tab's `constructor_params`/`method_params` keys traced to src constructor signatures; store persistence patterns; error-handling paths.
- Evidence: `ui/tabs/network.py:138-148` (disable-without-reset) vs `find_path.py:196-203` (reset+disable sibling guard) vs `coana.py:1886-1895` (filter applied independently of `separate_hemispheres`); `mapping_store.py:50-59` vs atomic-write invariant at `history_store.py:6`; `run_state.py:303,504`; `runner.py:1217-1257`; `connectivity.py:217-225,258-280,487-493`.
- Findings added/updated: F-UI-001…010.
- Unmatched information: UI_NOT_PROPAGATED (F-UI-001, F-UI-002); backend param with no UI exposure on an internally-inconsistent page (F-UI-008).
- Limitations: no live browser session; all findings from code-path tracing. F-UI-007 severity not measured under load.
- Next: skills audit.

### Audit checkpoint — skills & reusable instructions — 2026-09-25

- Inspected: all 5 `skills/*` — frontmatter, every relative reference, scripts (ast.parse), assets/templates, `agents/openai.yaml`; priority cross-checks: TM VEV SKILL.md flags/exports vs `RunMappingValidation.py` + `mapping_validation.py`; drocat-usage recipes vs scripts/; drocat-install vs `archive/install/install.sh`; drocat-backend module claims vs `src/`.
- Evidence: `grep '^def |^class |^token_manager' src/utils/token_manager.py` → only `TokenManager` + singleton (no `get_access_token`/`get_cave_token`), vs `skills/drocat-usage/tabs/settings.md:86` and `skills/drocat-backend/references/util-support.md:10`; config field counts via AST (61 fields, 51 set) vs `type-validation.md:18` claim.
- Findings added/updated: F-SKILL-001…005.
- Unmatched information: SKILL_DRIFT (broken import snippets ×2, stale counts/names); TM VEV skill otherwise fully verified against the tree including WIP state.
- Limitations: skill scripts parsed, not executed.
- Next: config/packaging/tests, then traceability & synthesis.

### Audit checkpoint — configuration, packaging & tests — 2026-09-25

- Inspected: `pyproject.toml` (py-modules vs `src/` listing), `requirements.txt`/`requirements-windows.txt` pin diff, `pyrightconfig.json`, `config.json`/`config_local.json` secret posture, tracked-file hygiene, pytest markers/skips, CI presence.
- Evidence: module-set diff (14 listed vs 33 top-level; 19 missing incl. `connection_map`, `neuron_search`, `morphology`); `src/coana.py:80` unconditional `from connection_map import ThresholdedConnectionMap`; `git ls-files config.json` + content (clean template, empty token strings); `git grep` JWT/Bearer scan over tracked files (placeholders only); `python3 -m pytest --collect-only -q -p no:cacheprovider` → 6,105 collected, 0 errors; requirements-windows pin set byte-identical to requirements.
- Findings added/updated: F-CONFIG-001…006.
- Unmatched information: none beyond findings (no secrets, no junk tracked, markers fully registered, skips principled).
- Limitations: full test suite not executed (collect-only; runtime pass out of audit scope); wheel build not performed (reasoned from setuptools flat-module semantics).
- Next: synthesis (this report).

## Finding details

### F-CODE-001 — FAFB soma-position lookup always returns `{}` (undefined name swallowed)

- Severity: P1 | Confidence: confirmed | Status: open | Area: Code
- Statement: `_load_flywire_soma_positions` filters results through `set(normalize_flywire_body_ids(body_ids))` at `src/morphology.py:383`, but only the singular `normalize_flywire_body_id` is imported (lines 92/102). The NameError is swallowed by the function-wide `except Exception: return {}` (lines 395-396), so the explicit-bodyIds path always returns empty.
- Evidence: `pyflakes src/morphology.py` → `383:17: undefined name 'normalize_flywire_body_ids'`; callers pass populated lists at `src/morphology.py:5348` (feeds `soma_pos=` at 5479/5493/5529) and `:5741`; the defect is documented and asserted by `tests/core/test_morphology_coverage.py:96-99` and `:2152-2155` ("BUG REPORTED … always swallows a NameError and returns {}").
- Impact: soma anchoring for FlyWire/FAFB skeleton fetches silently never happens; a known bug is enshrined by tests instead of fixed.
- Affected artifacts: `src/morphology.py`, `src/flywire_ids.py` (defines the function), `tests/core/test_morphology_coverage.py`.
- Reproduction: `python3 -m pyflakes src/morphology.py`; or call `_load_flywire_soma_positions(dataset, root, ['12345'])` on a dataset with a soma table — returns `{}`.
- Fixation proposal: import the plural normalizer (or map singular over the list); narrow the `except` to the IO/parsing it guards; flip the tests to assert the working contract.
- Validation: existing coverage tests after fix; a fixture with a soma CSV and requested ids asserting non-empty result.

### F-CODE-002 — Bridged-scene loop uses undefined `b`; guarded behavior never runs

- Severity: P1 | Confidence: confirmed | Status: open | Area: Code
- Statement: in `src/comparison/morph_cross_dataset.py:2296-2307` (`_render_scenes` bridged layers), the `for n in neurons:` loop assigns `n.name = f'…{int(b)}'` and `n._drocat_source_dataset = ds`, but `b` is only ever bound inside comprehensions (lines 2286, 2292), which do not leak in Python 3. Every iteration raises NameError, hidden by `except Exception: pass` — so neither the per-member name uniquification (the documented fix for legend collapse / layer-loop crashes, per the inline comment) nor the source-dataset tagging executes.
- Evidence: `pyflakes src/comparison/morph_cross_dataset.py` → `2304:43: undefined name 'b'`; scope scan of lines 2200-2296 shows no other `b` binding.
- Impact: bridged scenes silently run without the crash-guard the comment promises (legend-leaf collapse / index-slip risk) and without per-member dataset tagging.
- Affected artifacts: `src/comparison/morph_cross_dataset.py`.
- Reproduction: pyflakes line above; any bridged-scene render exercises the path with the exception firing per neuron.
- Fixation proposal: iterate `for n, b in zip(neurons, shown)`-style pairing or capture the id when building `neurons`; remove the blanket `except`.
- Validation: render a bridged scene and assert each neuron's name carries its bodyId suffix.

### F-CODE-003 — `eval()` on data-derived path strings

- Severity: P1 | Confidence: confirmed (code path) / likely (exploitability) | Status: open | Area: Code
- Statement: `path_has_unconserved_edge` at `src/coana.py:18227-18229` parses the `path_str` column with `eval(path_str_val)` under a bare `except` fallback. The string originates from persisted paths DataFrames (parquet/CSV round-trips).
- Evidence: `sed -n '18210,18235p' src/coana.py` (quoted in checkpoint); reachable when `keep_only_hemisphere_conserved_connections and separate_hemispheres` (`:18212`).
- Impact: arbitrary code execution if any upstream table carries a crafted string; robustness hazard independent of adversarial input.
- Affected artifacts: `src/coana.py`.
- Reproduction: set the two flags; feed a paths frame whose `path_str` cell is `"__import__('os').system('true')"` — executes.
- Fixation proposal: replace with `ast.literal_eval` with the existing `'->'`-split fallback.
- Validation: unit test with list-literal, arrow-joined, and malicious strings.

### F-CODE-004 — Profile-cache consolidation can silently destroy the main cache

- Severity: P1 | Confidence: confirmed (mechanism) / likely (occurrence) | Status: open | Area: Code
- Statement: in the pandas fallback of `src/comparison/connectivity_profiler.py:1664-1692`, an unreadable main `connectivity_profiles.parquet` is skipped via `except: pass`, the combined (batch-only) frame then overwrites the main cache (`:1683`), and the batch files are deleted (`:1688-1692`) — permanent silent loss of all previously consolidated profiles. The polars primary path (`:1585-1590`) at least logs the rebuild but still overwrites and deletes.
- Evidence: line extract in checkpoint.
- Impact: data loss under exactly the crash/partial-write conditions that batch files exist for; no log line in the fallback path.
- Affected artifacts: `src/comparison/connectivity_profiler.py`; consumers of the profile cache.
- Reproduction: write a truncated parquet to the main-cache path, create one valid batch file, run consolidation with `delete_after=True` — main cache is replaced by batch rows only.
- Fixation proposal: quarantine (rename) the unreadable main cache and abort consolidation (or require an explicit `--rebuild`) instead of dropping it silently; never delete batches after a degraded merge.
- Validation: fixture reproducing the corrupt-main-cache scenario asserting the old file is preserved/quarantined.

### F-CONFIG-001 — Installed wheel omits 19 top-level `src/` modules

- Severity: P1 | Confidence: confirmed | Status: open | Area: Config/packaging
- Statement: `[tool.setuptools] py-modules` (`pyproject.toml:128-143`) lists 14 modules; `src/` has 33 top-level modules. Missing 19: `banc_public_data`, `build_seed_indexes`, `connection_map`, `fafb_bundle`, `flywire_mesh_cache`, `morphology`, `morphology_comparison`, `neuron_index_builder`, `neuron_search`, `neuronbridge_coverage`, `neuronbridge_output_policy`, `neuronbridge_query_expansion`, `roi_screening`, `roi_sensitive_segmentation`, `skeleton_provenance`, `skeleton_simplification`, `storage_inventory`, `synapse_cache`, `visualization_options`. `packages.find` only rescues directories, not flat files.
- Evidence: module-set diff (checkpoint 6); `src/coana.py:80` unconditionally imports `connection_map` at module top; `src/statvis.py:464,577,2463` import `neuron_search`; `src/visualize_skeleton.py:233,254,12481` import `skeleton_simplification`/`synapse_cache`/`skeleton_provenance`.
- Impact: a non-editable `pip install drocat` yields `import coana` → `ModuleNotFoundError` at import time. Only repo-checkout + editable-install workflows (the documented paths) work.
- Affected artifacts: `pyproject.toml`; wheel/pip consumers.
- Reproduction: `python3 -m pip wheel . -w /tmp/wheeltest --no-deps && unzip -l` — the 19 modules are absent.
- Fixation proposal: replace the hand-maintained list with an explicit generated enumeration (or move flat modules into a package) so the manifest cannot drift from `src/`.
- Validation: build a wheel, install into a clean venv, `import coana, statvis, visualize_skeleton`.

### F-CODE-005 — Dead fallback-diagnostics helper with undefined global

- Severity: P2 | Confidence: confirmed | Status: open | Area: Code
- Statement: `src/neuron_search.py:1245-1269` `_report_cache_fallback` is never called anywhere, and references module-global `_FALLBACK_DIAGNOSTICS_EMITTED` that is never defined — calling it would NameError. Its documented purpose (log why a silent downgrade causes full-table scans) is therefore delivered nowhere.
- Evidence: repo-wide rg over `src/`, `scripts/`, `ui/`, `tests/`, `skills/` — zero callers, zero other references to the global.
- Impact: observability gap that the codebase already wrote the fix for but never wired.
- Affected artifacts: `src/neuron_search.py`.
- Reproduction: rg evidence; not executed.
- Fixation proposal: either wire the helper into the cache-fallback path or delete it; define the global if wired.
- Validation: trigger a cache fallback and assert a log line appears.

### F-CODE-006 — Diagonal type-block index/member desync

- Severity: P2 | Confidence: confirmed (logic) / likely (trigger) | Status: open | Area: Code
- Statement: `src/morphology_comparison.py:500-519` builds `ids` from filtered positions (`pos`) but indexes `members[a][ii]` with the same subscript (`:516`), so when any member of a type lacks a matrix row, `_pair_ok` filters the wrong pairs and `body_matrix[ids[ii], ids[jj]]` attributes cells to the wrong neurons — silently.
- Evidence: line extract; no guard ties `ids` and `members[a]` lengths.
- Impact: wrong cohesion/cross-type means in partial-cache runs; no error raised.
- Affected artifacts: `src/morphology_comparison.py` (type-block diagnostics).
- Reproduction: fixture where one member of a type is missing from the similarity matrix.
- Fixation proposal: iterate members and positions together (e.g. build `(member, id)` pairs) or assert parity.
- Validation: regression test with a missing member asserting correct pairing.

### F-CODE-007 — Cache-load failures silently drop bodyIds

- Severity: P2 | Confidence: confirmed (pattern) | Status: open | Area: Code
- Statement: both `as_completed` loops over parquet-load futures at `src/neuronbridge_finder.py:4356-4374` swallow every exception (`except Exception: pass`); bodyIds whose cache files fail to load vanish from the loaded set with no log.
- Evidence: line extract.
- Impact: incomplete match tables presented as complete.
- Affected artifacts: `src/neuronbridge_finder.py`.
- Reproduction: make one cache parquet unreadable (chmod/ truncate) and run a batch load.
- Fixation proposal: count and log failures; surface a warning with the missing bodyIds.
- Validation: fixture with one corrupt cache file asserting a warning naming the id.

### F-CODE-008 — Systemic bare `except:` on data paths

- Severity: P2 | Confidence: confirmed (sites) / varies (per-site impact) | Status: open | Area: Code
- Statement: 63 bare `except:` clauses across `src/` (worst: `comparison_analyzer` 15, `connectivity_profiler` 11, `visualize_skeleton` 9, `profile_comparator` 9). They also catch `KeyboardInterrupt`/`SystemExit`. Data-critical exemplars: `src/comparison/html_report_generator.py:2846-2847` turns any aggregation error into `avg_prob = 0.0` exported into `used_data/edge_count_data.csv`; `connectivity_profiler.py:1929-1932` parses malformed JSON rows to `None`.
- Evidence: rg census + cited lines.
- Impact: error masking in exports; unkillable-looking loops during interrupts.
- Affected artifacts: the 63 sites (census available on request).
- Reproduction: rg `except:` census.
- Fixation proposal: sweep to `except Exception` at minimum; add logging on the data-export exemplars first.
- Validation: targeted tests on the two exemplars.

### F-DOC-001 — Three intra-doc links missing `../`

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: `docs/core-features/CrossDatasetComparison_Guide.md:317` and `:367` link `[AUTO_TYPE_MAPPING](AUTO_TYPE_MAPPING.md)` and `docs/core-features/NeuronBridge_Guide.md:677` links `[Output Files](OUTPUT_FILES.md)` — all resolve to nonexistent files inside `core-features/`; the real files are `docs/AUTO_TYPE_MAPPING.md` and `docs/OUTPUT_FILES.md`.
- Evidence: `ls` of both targets (one exists at parent, one not in core-features).
- Impact: broken navigation for readers of two feature guides.
- Reproduction: open the three links.
- Fixation proposal: prefix `../`.
- Validation: link checker re-run.

### F-DOC-002 — `docs/core-features/README.md` unbalanced code fences swallow three headings

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: the file has 35 fence markers (odd) and ends inside an open fence (last block opens at line 594). Blocks opened earlier swallow the real headings `## ✨ Connectivity Profiler` (158), `## Path Finding` (334), `## Filtering` (448). Hub links `docs/README.md:161` (`#path-finding`) and `:213` (`#filtering`) therefore point at anchors GitHub never generates, and the page renders with a giant stray code block.
- Evidence: fence-parity computation (checkpoint 3); fence counts 7/17/21 before the three heading lines (all odd = inside a fence).
- Impact: mangled rendering of the feature hub plus two dead anchors.
- Affected artifacts: `docs/core-features/README.md`, `docs/README.md`.
- Reproduction: open the page on GitHub/any renderer; or rerun the parity script.
- Fixation proposal: locate the unbalanced fences (stray triple-backtick lines) and repair; verify anchors regenerate.
- Validation: parity script reports balanced; `#path-finding`/`#filtering` anchors exist.

### F-DOC-003 — `sim_roi` documented as removed but still written

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs / export schema
- Statement: `docs/OUTPUT_FILES.md:380` states the removed `sim_topology`/`sim_roi` columns "no longer occur"; `src/morphology.py:7222` still writes `row["sim_roi"]` and `:8108` still orders it in `results.csv`.
- Evidence: line extracts (checkpoint 3 verification batch).
- Impact: documented output contract incomplete; consumers/agents relying on it mishandle the column.
- Affected artifacts: `docs/OUTPUT_FILES.md`, `src/morphology.py`.
- Reproduction: any morphology-similarity run with ROI blocks present emits `sim_roi`.
- Fixation proposal: either delete the column (code) or restore it in the doc — pick one truth.
- Validation: run output vs doc column list diff.

### F-DOC-004 — Deleted verification script still given as a copyable command

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: `docs/technical/NEGATIVE_VALUES_QUICKREF.md:79`, `docs/visualizations/NETWORK_BLANK_FIX.md:70`, and `docs/technical/NEGATIVE_VALUES_IMPLEMENTATION.md:149` instruct `python scripts/PlotPath_TestNegatives.py`; no such file exists anywhere.
- Evidence: `find` over repo; previously flagged in `docs/audits/DROCAT_DOCUMENTATION_TRACEABILITY_AUDIT_2026-09-07.md:186` and still unfixed.
- Impact: documented verification command fails outright; repeat-offender drift.
- Affected artifacts: the three docs.
- Reproduction: run the command.
- Fixation proposal: repoint to the current equivalent harness or remove the commands.
- Validation: link/path checker.

### F-DOC-005 — Harness scripts referenced at pre-move paths

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: `docs/technical/PATHFINDING_PIPELINE.md:735` cites `scripts/verify_budget_fit_pruning.py`, `scripts/compare_budget_fit_real_data.py`, `scripts/verify_production_real_data.py`; they live at `scripts/harness/…`.
- Evidence: path existence check.
- Impact: copyable paths 404.
- Fixation proposal: prefix `harness/`.
- Validation: path checker.

### F-DOC-006 — Deleted test cited as "complete example"

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: `docs/COLABELING_SIMILARITY_METHODS.md:195` points to `tests/test_colabeling_similarity.py`; the current test is `tests/core/test_neuronbridge_colabeling.py`.
- Evidence: existence check.
- Impact: readers land on a missing file.
- Fixation proposal: update the path.
- Validation: path checker.

### F-DOC-007 — Dead TROUBLESHOOTING TOC anchor

- Severity: P2 | Confidence: confirmed | Status: open | Area: Docs
- Statement: `docs/TROUBLESHOOTING.md:14` links `#flywire-data-download-issues`; the section is now `## FAFB and BANC Release Preparation Issues` (line 437).
- Evidence: heading scan.
- Impact: TOC entry jumps nowhere.
- Fixation proposal: update the anchor/slug.
- Validation: anchor check.

### F-SCRIPT-001 — TM VEV auditor default audits a stale queue

- Severity: P2 | Confidence: confirmed | Status: open | Area: Export/scripts
- Statement: `scripts/verify_tmvev_run_exports.py:8-9` (docstring) promises "with no PATH the newest queue directory under `local_data/` is used", but `:848-850` globs only `tmvev-clock-*`. Current `local_data/` newest queues (`tmvev-banc-warm-20260925_1139`, `tmvev-cert-20260925_0147`, matrix queues `tmvev-matrix-*` from `run_tmvev_matrix.sh:42`) do not match; bare invocation audits `tmvev-clock-*` only.
- Evidence: glob extract + directory listing (verification batch 3).
- Impact: a bare run prints a clean verdict for the wrong (stale) queue — misleading certification.
- Affected artifacts: `scripts/verify_tmvev_run_exports.py` (WIP file), `scripts/maintenance/run_tmvev_matrix.sh`.
- Reproduction: run the script with no args; observe which queue is picked.
- Fixation proposal: glob `tmvev-*` or narrow the docstring; consider requiring an explicit path.
- Validation: no-arg run against the current `local_data/` picks the newest queue.

### F-SCRIPT-002 — Tracked preview scripts hardcode a personal run folder

- Severity: P2 | Confidence: confirmed | Status: open | Area: Export/scripts
- Statement: `scripts/preview/label_mapper_realdata_test.py:24`, `rerun_mfbh_hemi.py:20`, `reexport_mfbh_report.py:22`, `rerun_mfbh_real.py:22` hardcode `/Users/apple/Local/connection_data/DROCAT_data/cross-dataset_…_20260915_213447`. All six `scripts/preview/` files are git-tracked and referenced nowhere (docs/ui/skills/tests).
- Evidence: rg census + `git ls-files scripts/preview`.
- Impact: unrunnable on any other machine; dead weight in a public repo; one script (`kit_equivalence_sample.py:19`) additionally crashes barefoot on `sys.argv[1]`.
- Fixation proposal: move to `archive/scripts_local/` (the repo's own convention) or delete.
- Validation: `git ls-files` clean of preview/.

### F-UI-001 — Network tab sends stale, greyed-out hemisphere filter

- Severity: P2 | Confidence: confirmed | Status: open | Area: UI
- Statement: `ui/tabs/network.py:138-148` disables the hemisphere controls when "Hemisphere-aware" is off but never resets values (sibling tabs `find_path.py:196-203`, `find_shortest.py:233-241`, `inter_dataset.py:506-513` explicitly reset and carry a comment saying "a greyed-out True never reaches the backend"). `run_network` then submits `hemisphere_filter: hemi_filter.value` (`:238`), and the backend applies it independently of `separate_hemispheres` (`src/coana.py:1886-1895` runs before the `separate_hemispheres` branch).
- Evidence: line extracts in verification batch 2.
- Impact: a stale 'left'/'right' selection silently restricts the queried network while the control displays disabled.
- Affected artifacts: `ui/tabs/network.py`, `src/coana.py` (contract context).
- Reproduction: pick Hemisphere=left, toggle Hemisphere-aware off, run — payload still carries `hemisphere_filter: 'left'`.
- Fixation proposal: mirror the sibling tabs' reset-and-disable guard.
- Validation: e2e payload assertion; visual check of generated script kwargs.

### F-SKILL-001 / F-SKILL-002 — Broken token-import snippets in two skills

- Severity: P2 | Confidence: confirmed | Status: open | Area: Skills
- Statement: `skills/drocat-usage/tabs/settings.md:86` and `skills/drocat-backend/references/util-support.md:10` instruct `from src.utils.token_manager import get_access_token, get_cave_token`; `src/utils/token_manager.py` defines only `class TokenManager` and the `token_manager` singleton — no such module-level functions exist anywhere.
- Evidence: symbol grep (verification batch 3); real callers use `from utils.token_manager import token_manager`.
- Impact: following either snippet verbatim raises ImportError at the exact step meant to verify credentials.
- Affected artifacts: the two skill files.
- Reproduction: run the snippet.
- Fixation proposal: replace with the singleton usage (`token_manager.get_neuprint_token()` etc.).
- Validation: snippet executes.

### F-CONFIG-002 — requirements.txt vs pyproject dependency drift

- Severity: P2 | Confidence: confirmed | Status: open | Area: Config/packaging
- Statement: `open3d`/`k3d`/`ipywidgets` are core in `requirements.txt:81-96` (open3d "REQUIRED for accurate mesh simplification") but only in the `viz` extra in `pyproject.toml:104-108`; 10 packages are exact-pinned in requirements but range-pinned in pyproject (selenium, webdriver-manager, boto3, caveclient, cloud-volume, fast-simplification, psutil, xlsxwriter, packaging, xarray); 4 transitive pins exist only in requirements.
- Evidence: pin-set diff (checkpoint 6).
- Impact: `pip install drocat` resolves a different, untested stack than the documented requirements workflow — and misses open3d (degraded mesh simplification). No hard unsatisfiable conflict today.
- Affected artifacts: `pyproject.toml`, `requirements.txt`.
- Reproduction: `pip install drocat` vs `pip install -r requirements.txt` in clean venvs; diff `pip freeze`.
- Fixation proposal: generate one list from the other or add a consistency test.
- Validation: automated pin-diff check in CI (see F-CONFIG-004).

### P3 finding details

Each P3 in the table above carries its evidence inline (file:line + verification method). Reproduction for all is "open the cited lines"; proposals are the obvious minimal edit (fix path/anchor/count, delete dead code, add reset guard, add atomic write, add logging, mark superseded docs, renumber sections, add CI). Statuses: open. Two warrant nuance: **F-CODE-011** is an edge case in a WIP file (borrowed `no-score` verdicts relabeled `shared` can publish `gate_applied=True` — inferred, needs a fixture before fixing); **F-SCRIPT-007**'s false-positive risk (empty `{}` morph-store target failing Track-A-only runs) is likely but not reproduced, and its `graded` int→dict rebind is cosmetic.

## Traceability gaps

| Trace ID | Behavior/contract | Source | Test | UI/API | Instructions | Template/schema | Export | State | Gap/note | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| TR-001 | FAFB soma positions anchor fetched skeletons | `src/morphology.py:382-396` (+`:5348`) | `tests/core/test_morphology_coverage.py:96-99` (asserts broken behavior) | skeleton pull path | skeleton pipeline docs | n/a | skeleton cache metadata | conflicting | test encodes the defect as contract | F-CODE-001 |
| TR-002 | Bridged-scene per-member naming + dataset tagging | `src/comparison/morph_cross_dataset.py:2296-2307` | none found | Find Homolog/scene render | scene docs describe uniquified names | n/a | branches_*.html | conflicting | code comment documents behavior that never runs | F-CODE-002 |
| TR-003 | Hemisphere-conserved path filtering | `src/coana.py:18212-18230` | none found | Network/Path tabs (flags exist) | OUTPUT_FILES §pathfinding | n/a | paths CSV | partial | no focused test; eval-based parsing | F-CODE-003 |
| TR-004 | Profile-cache consolidation preserves history | `src/comparison/connectivity_profiler.py:1585-1692` | none found for corrupt-main case | Connectivity Profiler tab | CacheSystem_v4 docs | parquet schema | connectivity_profiles.parquet | partial | silent-loss path undocumented | F-CODE-004 |
| TR-005 | `pip install drocat` yields importable package | `pyproject.toml:128-148` | none found | n/a | README/drocat-install (editable-only) | wheel manifest | wheel | conflicting | 19 modules missing from manifest | F-CONFIG-001 |
| TR-006 | Morphology `results.csv` column contract | `src/morphology.py:8108` (+`:7222`) | column-order tests exist for tail list | Morphology tab | `docs/OUTPUT_FILES.md:380` | CSV header | results.csv | conflicting | sim_roi documented removed, still emitted | F-DOC-003 |
| TR-007 | TM VEV run-export verification | `scripts/verify_tmvev_run_exports.py:848-850` | self-checking | n/a | skill §verify + script docstring | manifest.tsv (7 cols) | local_data queues | conflicting | docstring says newest queue; glob says clock-only | F-SCRIPT-001 |
| TR-008 | Greyed-out dependent controls never reach backend | `ui/tabs/network.py:138-148` | none found for network tab | Network tab | sibling-tab code comments state the invariant | n/a | generated run script | unmatched | network.py is the one tab violating it | F-UI-001 |
| TR-009 | Token access recipe | `src/utils/token_manager.py:3,311` | token tests exist | Settings tab | `skills/drocat-usage/tabs/settings.md:86`; `skills/drocat-backend/references/util-support.md:10` | n/a | n/a | conflicting | snippets import nonexistent names | F-SKILL-001/002 |
| TR-010 | Docs hub navigation to feature anchors | `docs/README.md:161,213` | n/a | n/a | `docs/core-features/README.md:158,334,448` | n/a | rendered GitHub page | conflicting | headings swallowed by fences; anchors dead | F-DOC-002 |

Unmatched-information labels used: CODE_UNDOCUMENTED (TR-002 tagging), DOC_NOT_IMPLEMENTED (F-DOC-004), UI_NOT_PROPAGATED (TR-008, F-UI-002), EXPORT_UNDOCUMENTED/SCHEMA_DRIFT (TR-006), NAME_OR_PATH_DRIFT (F-DOC-005/006/009/010, F-SKILL-004), DEFAULT_CONFLICT (F-CONFIG-002), VALIDATION_GAP (TR-003/004), OBSERVABILITY_GAP (F-CODE-005/007/008, F-UI-003/004/009/010), GENERATOR_DRIFT (TR-007), SKILL_DRIFT (F-SKILL-001/002/003/005).

## Validation record

| Date | Command/check/render | Result | Scope/fixture | Limitations |
|---|---|---|---|---|
| 2026-09-25 | `git status --short` (pre-audit) | 11 modified WIP files recorded & preserved | worktree | snapshot only |
| 2026-09-25 | `python3 -m pytest --collect-only -q -p no:cacheprovider` | 6,105 tests collected, 0 collection errors | full `tests/` | collect-only; no execution |
| 2026-09-25 | `python3 -m pyflakes src/morphology.py src/comparison/morph_cross_dataset.py` | `383:17 undefined name 'normalize_flywire_body_ids'`; `2304:43 undefined name 'b'` | 2 flagged modules | spot-run of full sweep already done by code pass |
| 2026-09-25 | fence-parity script on `docs/core-features/README.md` | 35 markers (odd) → unbalanced; headings 158/334/448 inside fences | 1 file | heading-slug rendering inferred from GitHub rules |
| 2026-09-25 | relative-link & repo-path existence checks (agent-run, read-only) | 893 links + 245 path refs checked; broken ones filed | README + docs/ | external URLs excluded; anchor checks targeted |
| 2026-09-25 | `ast.parse` of all 5 skills' scripts; YAML validity | 5/5 parse; all YAML valid | skills/ | scripts not executed |
| 2026-09-25 | pyproject module diff vs `src/` listing; requirements pin diff | 19 modules missing from py-modules; pin drift catalogued | packaging files | wheel not actually built (reasoned from setuptools semantics) |
| 2026-09-25 | `git grep` secret scan (JWT/Bearer/token patterns) over tracked files | no credentials found; `config.json` clean template; placeholders only in TROUBLESHOOTING docs | tracked tree | pattern-based, not exhaustive |
| 2026-09-25 | lead-auditor `sed`/`nl` line verification | every P1/P2 claim re-verified at cited lines | all 22 P1/P2 findings | — |

## Residual uncertainty and blocked checks

- **WIP context**: 11 files carry uncommitted changes (pooling/morphology/report/docs/skill). Findings inside them (F-CODE-011, F-SCRIPT-007, F-DOC-003's doc side is WIP-edited) reflect the worktree, not HEAD; they may be mid-fix.
- **F-CODE-011** (pooling `gate_applied` on borrowed no-score verdicts) is inferred from code reading; needs a fixture to confirm.
- **F-SCRIPT-007**'s spurious-failure mode (empty `{}` morph-store identity on Track-A-only pooling runs) is reasoned, not reproduced.
- **F-CODE-013**: whether the set-ordering nondeterminism survives to final CSV row order is unverified downstream.
- **F-UI-007** severity is unmeasured (no loaded UI session during audit).
- Full test suite was collected but not executed; runtime pass/fail not re-established by this audit.
- External URLs, `archive/` historical docs (stale by design), and untracked data directories were not audited.
- On-disk-only hygiene notes (not tracked, not filed as findings): `skills/**/.DS_Store`, `skills/drocat-install/scripts/__pycache__/`, `src/.mimosa/`, stale egg-info dirs in `src/`.

## Audit completion statement

Audit complete across all six surfaces (code, docs, scripts/entry points, UI, skills, config/packaging/tests) with traceability and validation records above. **No implementation code, tests, UI, documentation sources, templates, skills, or generated outputs were modified by this audit** — the only file written is this report itself. All fixation proposals above are proposals, not changes. Pre-existing worktree changes (11 files) are untouched.

No implementation code was modified by this audit unless explicitly stated above.

---

## Fix round — 2026-09-25 (same day, post-plan)

The approved fix plan (8 waves) was executed the same day. All statuses below
supersede the `open` markers in the tables above; the original rows are kept
as history. Commits live on `v4.5.0` starting at `c80c285`.

### Disposition of all 58 findings

| Disposition | Findings | Evidence |
|---|---|---|
| fixed, verified (tests) | F-CODE-001 (4 soma tests flipped to the working contract), F-CODE-002 (`_tag_scene_members` test), F-CODE-003 (5 parser tests incl. non-executing crafted input), F-CODE-004 (corrupt-cache fixtures, both engines), F-CODE-006 (contralateral-only NaN fixture), F-CODE-007 (failure-reporting test), F-UI-001 (e2e: select resets + arrives disabled) | commit `c80c285` (wave 1), `721c024`* (wave 3) |
| fixed, verified (checks) | F-CONFIG-001 (wheel built + 11-module import smoke from the wheel), F-CONFIG-002 (pin checker green), F-CONFIG-003, F-CONFIG-004 (CI gates validated locally: undefined-name gate, packaging listing gate, 354-test slice, YAML parse), F-CONFIG-006 | `9b0d0a9`* (wave 2), `f83620a` (pins), `d1ce548` (pyright), `553a7f5` (CI) |
| fixed (docs/skills, checker-verified) | F-DOC-001…F-DOC-014 (fence parity 34/balanced, zero headings inside fences, `check_docs_links.py` green over 183 files), F-SKILL-001/002 (snippets executed), F-SKILL-003 (recounted 51/61), F-SKILL-004/005 | `7a94990` (wave 5), `553a7f5` (stragglers) |
| fixed, pending commit (rides with the owner's in-flight WIP hunks in the same files — not separable by hunk surgery) | F-SCRIPT-001 + F-SCRIPT-007 (`verify_tmvev_run_exports.py`: `tmvev-*` glob + native-verdict guard + `graded` rename), F-CODE-011 (pooling `gate_applied` now `scored > 0`, regression test added), F-CODE-012 (`vectors_for` space validation + test) | worktree edits on top of the WIP; regression tests added in `tests/core/test_mapping_validation_pooling.py` and `tests/core/test_morphology.py` |
| fixed (mechanical sweep) | F-CODE-008 — 58 remaining bare `except:` across 11 modules narrowed to `except Exception:` (audit counted 63; 5 were already fixed by waves 1/3 edits); the two data-critical exemplars log | wave-3 commit |
| premise corrected, fixed by removal | F-CODE-005 — the audit's "diagnostic never delivered" premise was wrong: `coana._record_cache_fallback_notes` consumes `info['cache_fallback_reason']` and reports per-reason to console + `user_warning_notes.txt`; the orphan was a leftover print-path duplicate of that mechanism (the module "intentionally prints nothing"). Dead duplicate removed instead of wiring a second path. | `src/neuron_search.py` deletion in wave 3 |
| fixed (P3 batch) | F-CODE-009/010/013/014, F-UI-002…F-UI-010 (full UI suite: 1,023 passed after the changes), F-SCRIPT-002/003/004/005/006/008, F-SKILL-003/004/005 | `0625ba3`, `ca95109`, `07379ca`, `d1ce548` |
| accepted-behavior, no change | F-CONFIG-005 (tracked parquet seeds are a documented `.gitignore` policy), F-SCRIPT-009 (`preliminary_ROI_sensitive_segmentation/` output dir is explicitly gitignored) | — |

\* wave-1/2/3 commit hashes: see `git log --oneline c80c285..553a7f5`; the
WIP-overlap files (`verify_tmvev_run_exports.py`, `mapping_validation_pooling.py`,
`morphology.py`'s `vectors_for`/`zipfile` hunks, `test_morphology.py`,
`test_mapping_validation_pooling.py`) remain uncommitted alongside the owner's
pre-existing WIP, untouched by every commit (temp-index hunk surgery for the
disjoint files; in-region edits left to ride for the overlapping ones).

### Audit-commit map (fixes only)

| Commit | Waves / findings |
|---|---|
| `c80c285` fix(core) | F-CODE-001…004 |
| wave-2 fix(packaging) | F-CONFIG-001, F-CONFIG-003 |
| wave-3 fix(core) | F-CODE-005 (premise corrected), 006, 007, 008 |
| wave-4 fix(ui) / fix(skills) / build(deps) / chore(scripts) | F-UI-001; F-SKILL-001/002; F-CONFIG-002 + pin checker; F-SCRIPT-002 |
| `7a94990` docs | F-DOC-001…014, F-SKILL-003/004/005 |
| `0625ba3` chore(core) | F-CODE-009/010/013 (+ F-SCRIPT-008's file removal, absorbed here) |
| `ca95109` chore(ui) | F-UI-002…010 |
| `07379ca` chore(scripts) | F-SCRIPT-003…006 |
| `d1ce548` chore(config) | F-CONFIG-006 |
| `553a7f5` ci | F-CONFIG-004 + docs checker + 3 checker-surfaced stragglers |

### Addendum — new observation from the fix round

- **Wheel dependency gap (P3, unfiled as code change)**: `coana.py` imports
  `vispath_pkg` unconditionally, but the drocat wheel does not contain it and
  `pyproject` does not declare the vispath subproject as a dependency — a
  non-editable install needs `vispath-subproject` installed separately. Same
  class as F-CONFIG-001; recorded here rather than expanded mid-round.

### Closeout checkpoint — fix round — 2026-09-25

- Inspected: all 58 findings re-touched; every fix verified by test, checker,
  or execution as listed above.
- Validation commands and results: see the tables above; headline runs —
  full UI suite 1,023 passed (post-UI-changes); curated CI slice 354 passed;
  `pytest --collect-only` 6,105/0 errors; wheel import smoke 11/11 from the
  wheel; `check_docs_links.py` green (183 files); `check_pin_consistency.py`
  green; pyflakes undefined-name gate green. The full `tests/core` battery:
  **4,911 passed, 1 skipped, 0 failed** (914 s) — combined with the UI
  suite's 1,023 passed, the whole post-fix battery is green.
- Residual: the four WIP-adjacent fixes await the owner's next commit (their
  tests are in place and green); the Windows launcher changes are
  code-reviewed but not executed on Windows; CI has not yet run on a real
  runner (all gates validated locally).
- No implementation changes beyond the fixes described here; the owner's WIP
  hunks were preserved unstaged throughout (verified by hunk-count diff after
  every surgery).
