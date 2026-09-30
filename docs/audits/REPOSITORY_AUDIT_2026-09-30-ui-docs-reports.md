# Audit report — DROCAT UI functionalities, docstrings, docs, UI-linked instructions, UserGuide export, report generation

- Audit date: 2026-09-30
- Project root: `/Users/apple/Documents/GitHub/DROCAT-Drosophila-connectome-analysis-toolkit`
- Report: `docs/audits/REPOSITORY_AUDIT_2026-09-30-ui-docs-reports.md`
- Status: `complete` (fix round executed in the same session — see §Fix record)
- Scope: the UI layer end-to-end — `ui/tabs/*`, `ui/components/*`, `ui/app.py`, `ui/output_guide.py` (the UserGuide/output-guide export), the four report.html builders (`report_kit` consumers), and every doc surface claiming their behavior: `docs/ui_guides/*.html`, `docs/OUTPUT_FILES.md`, `docs/core-features/*`, `docs/visualizations/HEATMAP_CLUSTERING*`, `skills/drocat-usage/tabs/*`, in-file docstrings/hints. Tree at HEAD `6896062`, clean.
- Method: `skills/repository-audit` workflow, scoped. Four parallel read-only inspection passes (analysis tabs · viz/infra tabs + editors · output-guide/UserGuide export · report-generation chain) following UI control → payload → writer → doc chains; lead-auditor line-level verification of every P1/P2 claim before recording.
- Exclusions: `local_data/` run folders were read only as evidence for claimed output patterns; no visual/browser verification this round (JS verified by node --check in the prior round).

## Executive summary

**36 findings: 2 P1, 14 P2, 20 P3.** No data-destroying or crashing defects. The two P1s are documentation asserting the opposite of the code: the core-features guide says Backward (reciprocal) evidence defaults OFF when everything (CLI, config, UI, sibling docs) defaults it ON — misstating the cost of a default run — and the output guide + OUTPUT_FILES attribute `SCENE_FAILED.txt` to the 3D-Skeleton viewer, which never writes it (only the TM VEV scene builder does), leaving Skeleton-tab mid-render failures exactly the ambiguous partial folders both docs promise the marker prevents.

The dominant pattern is **contract drift after recent feature rounds**: contracts flipped or added in the last three weeks (backward-evidence default, Route scope selector, match-panel select-all, taxonomy/instance query lanes, v2.2 similarity columns, TM VEV's 14-tab report, Ward-everywhere clustering) whose guide/skill/spec updates landed only partially. Second pattern: **the output guide's file-pattern/column lists lagging the writers** (a dead fnmatch pattern that silently drops the 3D-scene entry from every exported guide; ten v2.2 metric columns absent from both the spec and the glossary; six real inter-dataset artifacts with no contract row anywhere). Third: **two genuine behavior gaps** the docs already claim — Auto threshold mode's missing ≥2-dataset gate, and Shortest Paths dropping the default `None` keyword filter that Complete Paths sends.

### Findings register

| ID | Sev | Area | Finding | Fix disposition |
|---|---|---|---|---|
| F-UI-001 | P1 | Docs | Core-features guide: `--backward-evidence` "default OFF" contradicts code/UI/sibling docs (default ON since 2026-09-26) | fixed (doc) |
| F-UI-002 | P1 | Guide/writer | `SCENE_FAILED.txt` documented as a 3D-Skeleton viewer output; only TM VEV writes it; TM VEV spec has no entry for it | fixed (spec + docs) |
| F-UI-003 | P2 | UI/docs | Auto threshold mode: "requires ≥2 datasets" claimed, no gate in UI or backend | fixed (UI gate) |
| F-UI-004 | P2 | UI | Shortest Paths sends `keywords=[]` (no `['None']` fallback) — diverges from Complete Paths + docs | fixed (code) |
| F-UI-005 | P2 | UI/docs | Edge-list CSV import silently discards expanded columns (nt_type, ratio, probability, …) on re-export | fixed (import warning + doc) |
| F-UI-006 | P2 | UI/docs | "Never-loses-edits" autosave claims omit the 5-filled-rows + Draft-Name gates (both editors, caption, skeleton.html) | fixed (qualified) |
| F-UI-007 | P2 | Docs | Route scope (Curated/Full map) selector undocumented anywhere | fixed (guide + skill) |
| F-UI-008 | P2 | Guide | Output guide: similarity-matrix column lists omit the ten v2.2 columns; no glossary entries | fixed (spec + glossary) |
| F-UI-009 | P2 | Guide | Output guide: TM VEV report "12 tabs" — writer renders 14 (Homolog ·fwd/·bwd unnamed) | fixed |
| F-UI-010 | P2 | Guide | Dead spec pattern `plot-3d_*/` never matches a file (3D-scene entry dropped from every exported guide) | fixed → `plot-3d_*/**` |
| F-UI-011 | P2 | Guide | Morph results.csv column list asserts conditional columns unconditionally; hides `type_coverage` | fixed |
| F-UI-012 | P2 | Guide/docs | `conserved_paths/`, `type_resolution_topology.json`, `type_appearance_order.csv` written but in neither contract | fixed (spec + OUTPUT_FILES) |
| F-UI-013 | P2 | Docs | OUTPUT_FILES §6d claims `max_scenes` in cross-morph parameters.json — key never written (TM VEV-only) | fixed (clause removed) |
| F-UI-014 | P2 | Report | Profiling hero "Six similarity metrics" vs 5 rendered/its own "5 metrics" heading | fixed |
| F-UI-015 | P2 | Report | TM VEV hero points at a "Mapping tab" that does not exist (card lives in Coverage) | fixed |
| F-UI-016 | P2 | Docs | HEATMAP_CLUSTERING docs describe average linkage + cite statvis.py; every renderer is Ward (page selector aside) | fixed (docs rewritten) |
| F-UI-017 | P3 | UI | Morphology Comparison query hint omits taxonomy/instance lanes | fixed |
| F-UI-018 | P3 | UI | Use-Cache hint points at a Find Similar control that does not exist (lives in Settings) | fixed |
| F-UI-019 | P3 | Docstring | connectivity.py module docstring limits Morph Qualification to cross-dataset | fixed |
| F-UI-020 | P3 | Skill | connectivity-profiling skill: top_k=15 (UI 25), direction vocab "input/output" (backend upstream/downstream) | fixed |
| F-UI-021 | P3 | Skill | inter-dataset skill: Edge Budget 0 (UI 1,000,000), max_workers None (UI 4) | fixed |
| F-UI-022 | P3 | Docs | Match-panel tri-state select-all undocumented | fixed |
| F-UI-023 | P3 | Docs | nb_find_lines.html: "Flat Folder Structure" control does not exist | fixed (clause removed) |
| F-UI-024 | P3 | Docs | nb_find_lines.html: Datasets default "hosted releases" vs code `(all)` | fixed |
| F-UI-025 | P3 | Skill | nb-find-lines skill defaults drift (image formats/types, simple_mode, bg color) | fixed |
| F-UI-026 | P3 | Skill | network.md omits `drop_untyped` from the FindNetwork constructor the UI sends | fixed |
| F-UI-027 | P3 | UI | Compact keep-last-N tooltip omits that plain Compact runs ignore it | fixed |
| F-UI-028 | P3 | Guide | `auto_type_mapping_per_bridge.csv` in OUTPUT_FILES but absent from guide spec | fixed |
| F-UI-029 | P3 | Guide | sensitivity/unified_summary column lists omit the files' payload columns | fixed |
| F-UI-030 | P3 | Guide | unified_edge_comparison query columns are combination-mode-only, unqualified | fixed |
| F-UI-031 | P3 | Guide | FlyLight patterns miss flat-structure root images | fixed |
| F-UI-032 | P3 | Guide | Homolog spec omits `results/shuffle_test.json` + `by_type/` layout | fixed |
| F-UI-033 | P3 | Guide | `connections_edge.fingerprint.json` sidecar unnamed in the guide's dataset_data entry | fixed |
| F-UI-034 | P3 | Docs | OUTPUT_FILES TM VEV report row under-enumerates tabs (5 missing); guide omits Suspects | fixed |
| F-UI-035 | P3 | Docs | OUTPUT_FILES omits `heatmap_intra_bodyid_{direction}_{metric}.html` | fixed |
| F-UI-036 | P3 | Docs | heatmap stem/tab wording omits custom-group level; metric list says `rank_union` (key is `rank_corr_union`) | fixed |

## Validation record

- Every P1/P2 finding re-verified by direct read before recording (backward default in CLI/config/UI; `SCENE_FAILED` writer grep; fnmatch probe `plot-3d_*/` → False; keywords payload lines; auto-mode gate absence; metrics.py column writers vs zero guide mentions).
- Post-fix targeted suites, all green: output-guide consistency (55 incl. the new pins — every added spec column has a glossary entry, enforced by `test_all_spec_columns_exist_in_glossary`), editors (77 incl. the new dropped-columns behavioral test), UI tab suites (107), report/skill suites (169: profile-comparator coverage, mapping-validation report, drocat-usage skill). `check_docs_links.py` OK (184 files).
- Full staged battery after the fix commits (2026-09-30 15:47): 3/3 stages clean — core 5,163 passed / 1 skipped / 1 xfailed, ui 1,092 passed, docs-and-misc 176 passed.

## Fix record (same session, user-expanded scope)

Three commits, landed in this order:
1. **UI behavior fixes** — Shortest Paths sends the `['None']` keyword default like Complete Paths (F-UI-004); Auto threshold mode refuses a one-dataset run per the documented contract (F-UI-003); edge-CSV import names the columns it cannot round-trip instead of silently dropping them (F-UI-005, + behavioral test); Compact keep-last-N tooltip states the expanded-run scope (F-UI-027); morphology hints gain the taxonomy/instance lanes and the correct Use-Cache location (F-UI-017/018); connectivity module docstring drops the cross-dataset-only claim (F-UI-019); profiling hero says Five metrics (F-UI-014); TM VEV hero points at the Coverage tab (F-UI-015); both editors' never-lose-edits claims state the 5-rows + Draft-Name gates (F-UI-006).
2. **Output-guide/UserGuide contract fixes** — the ten v2.2 similarity columns in both spec entries + 19 new glossary entries (F-UI-008); TM VEV report 14 tabs with the Homolog tabs named (F-UI-009); dead `plot-3d_*/` pattern → `plot-3d_*/**` (F-UI-010); results.csv column list matches the writer's conditional layout incl. `type_coverage` (F-UI-011); new entries for `conserved_paths/`, `type_resolution_topology.json`, `type_appearance_order.csv`, `auto_type_mapping_per_bridge.csv`, `results/shuffle_test.json`, `by_type/**`, and the fingerprint sidecar (F-UI-012/028/032/033); sensitivity/unified_summary payload columns (F-UI-029); unified_edge combination-mode qualifier (F-UI-030); FlyLight patterns match flat structure (F-UI-031); `SCENE_FAILED.txt` moved to the TM VEV spec with the viewer-not-marked caveat (F-UI-002) + OUTPUT_FILES §3 corrected.
3. **Docs/skills lock-step + this record** — backward-evidence default ON in the core-features guide (F-UI-001, the P1); Route scope documented (F-UI-007); match-panel tri-state select-all documented (F-UI-022); skill parameter blocks synced to UI defaults and backend vocab (F-UI-020/021/025/026); nb_find_lines guide drops the nonexistent Flat Folder Structure control and fixes the Datasets default (F-UI-023/024); skeleton/network guides carry the autosave gates and the import-drop note; OUTPUT_FILES rows fixed (max_scenes clause, fourteen tabs, `heatmap_intra_bodyid_*`, `heatmap_{type|bodyid|group}` stems, `rank_corr_union` key, three new inter-dataset artifacts) (F-UI-013/034/035/036/012); HEATMAP_CLUSTERING docs rewritten to the Ward-everywhere reality with statvis.py references corrected (F-UI-016).

## Residual uncertainty

- No browser-level verification of rendered reports/guides this round; HTML claims were verified textually. `data-row-id` fallthrough and Quasar attribute forwarding were verified structurally in the prior round.
- The Skeleton-tab mid-render failure path still writes no `SCENE_FAILED.txt` (documented as such now); implementing a viewer-side marker is a feature decision left open.
