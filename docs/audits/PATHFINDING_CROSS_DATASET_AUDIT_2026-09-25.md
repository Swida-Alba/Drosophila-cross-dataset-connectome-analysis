# Audit report — pathfinding & cross-dataset analysis (+ fix-round review)

- Audit date: 2026-09-25 (second pass, after the repository-wide audit and its fix round)
- Project root: `/Users/apple/Documents/GitHub/DROCAT-Drosophila-connectome-analysis-toolkit`
- Report: `docs/audits/PATHFINDING_CROSS_DATASET_AUDIT_2026-09-25.md`
- Status: `complete`
- Scope: (1) review of the fix-round commits `c80c285..553a7f5` for regressions, focused on their pathfinding/cross-dataset hunks; (2) careful audit of the pathfinding subsystem (`src/coana.py` FindAllPath/FindShortestPath/FindNetwork machinery, budgets/pruning/thresholds/caching, `vispath-subproject` enumerators); (3) careful audit of cross-dataset pathfinding and analysis (inter-dataset mode, `comparison_analyzer`, `profile_comparator`, threshold provenance, ID-space handling).
- Method: three parallel read-only inspection passes + lead-auditor line-level verification of every P1 and the reviewed-fix residuals. No file was modified; commands were read/execute-only probes.
- Companion report: `REPOSITORY_AUDIT_2026-09-25.md` (58 findings; fix round disposition).

## Executive summary

**The fix round holds up.** All 13 commits reviewed hunk-by-hunk: the path-node parser is a strict improvement over `eval` (one theoretical new exception class escapes its `except` clause — see F-RV-001), the `sorted()` change is behavior-preserving (both downstream fetches are total, order-free; verified no caps), all 58 except-narrowing sites wrap ordinary library calls (no BaseException-reliant teardown found), the quarantine/tmp-rename logic is sound (one same-second collision nit), the type-matrix rewrite is value-identical when no member is missing, the Network-tab reset guard fires only on the disable transition, and the docs renumber left no stale cross-references (all OUTPUT_FILES-anchored references target §9).

**The focused subsystem audit found 4 new P1s** — two where recorded provenance or cached state silently diverges from what the run actually did (streaming keyword sentinel; graph-cache key missing `hemisphere_filter`), and two in cross-dataset analysis where an intra-dataset metric or an under-keyed cache produces wrong numbers without warning (bare-bodyId scoring across two connectomes' ID spaces; profile cache ignoring `min_synapse_threshold`). Six P2s and ten P3s follow. The core enumeration machinery itself verified clean and in places impressively so: lossless hop-budget pruning, the edge-budget fit/floor order, both shipped enumerators (`find_paths_strongest_first`, `find_paths_shortest_strongest_first`), per-dataset threshold application, and the cross-dataset type-level alignment (no bare-bodyId joins) all check out with evidence.

### Priority findings

| ID | Severity | Confidence | Status | Area | Finding |
|---|---|---|---|---|---|
| F-PF-001 | P1 | confirmed (mechanism; data impact conditional) | open | Pathfinding | Streaming type-path export receives the RAW `keyword_in_path_to_remove` ('None' sentinel default) — bypasses `_normalized_keyword_filter()`, silently drops paths whose type label contains the substring "None" |
| F-PF-002 | P1 | confirmed | open | Pathfinding | `hemisphere_filter` missing from the FindAllPath graph-cache key — same-process runs that change it reuse a stale graph with no re-check (the key's own docstring states the rule it violates) |
| F-XD-001 | P1 | confirmed (mechanism); reachability = strict mode | open | Cross-dataset | `direct_comparison(strict)` on type names delegates cross-dataset pairs to `combined_score_intra_dataset` — cosine/jaccard over bare bodyId intersections across two connectomes' ID spaces |
| F-XD-002 | P1 | confirmed | open | Cross-dataset | Profile cache validity gates only on `top_k`; `min_synapse_threshold` (and fuzzy/untyped config) not keyed — `_get_config_hash` exists for exactly this and is never called (dead) |
| F-PF-003 | P2 | likely (arithmetic; not executed) | open | Pathfinding | `fit_edge_budget` can IndexError at the virtual bracket boundary (`distinct[hi_idx]` with `hi_idx == len(distinct)`) in a rare all-tiers-fit + probe-exhaustion regime → hard crash, no marker |
| F-PF-004 | P2 | confirmed | open | Pathfinding | `user_warning_notes.txt` claims hemisphere-conserved filtering ran whenever the flag is set — all four filter gates also require `separate_hemispheres`; (True, False) silently skips the filter while the note asserts it |
| F-PF-005 | P2 | confirmed | open | Pathfinding | `_write_run_metadata` post-enumeration re-stamp is `except Exception: pass` — a failed re-write leaves the pre-enumeration snapshot (no tau/floor) silently |
| F-XD-003 | P2 | confirmed | open | Cross-dataset | `_try_load_cached` accepts any non-empty `connections_edge.csv` at `(dataset, threshold)` — no query fingerprint; folder reuse with `skip_existing=True` (the default) can silently load another query's (or mode's) edges |
| F-XD-004 | P2 | likely | open | Cross-dataset | Edge-mode `effective_thresholds.json` reports the side-effect path runs' tau/budget provenance, not the `weight >= requested` filter the edge data actually got |
| F-XD-005 | P2 | confirmed | open | Cross-dataset | `direct_comparison`'s loose path calls `batch_compare_cross_dataset` without `type_mapper` (no such parameter exists) — partner names compared raw, unlike every other cross-dataset surface |
| F-XD-006 | P2 | confirmed | open | Cross-dataset | Homolog build path swallows cache pre-warm failures and per-bodyId build errors (`except: pass`) — a systematically broken cache yields an empty/partial result, not an error |

### Minor findings (P3)

| ID | Area | Finding (confidence) |
|---|---|---|
| F-PF-006 | Pathfinding | find_reciprocal fallback guards on `'conn_layers' in locals()` but the parameter is `all_connections` — fallback is dead; reciprocal export silently empty when the graph extract finds nothing (confirmed) |
| F-PF-007 | Pathfinding | Type-path export order is set-iteration (PYTHONHASHSEED) until the post-sort; polars sort tie order not guaranteed stable; a failed sort leaves generator order with no marker (confirmed) |
| F-PF-008 | Pathfinding | `FindAllPathMultiThreshold` leaves `max_paths_bodyid=1,000,000` and the last threshold mutated on the instance — sequential API use inherits them (confirmed) |
| F-PF-009 | Pathfinding | Shortest-mode graph-cache reuse/extension block is unreachable; `_findallpath_cache_key` docstring promises shortest entries are "extended" — they never are (conservative behavior, misleading docs) (confirmed) |
| F-PF-010 | Pathfinding | `hemisphere_filter='left'/'right'` + conserved-edges empties everything by construction (mirror edges removed at fetch); undocumented interaction (likely) |
| F-XD-007 | Cross-dataset | `comparability_report` intersects only non-empty materialized sets — an empty dataset is excluded, so "comparable across all datasets" can print with one absent (confirmed) |
| F-XD-008 | Cross-dataset | No guard against two dataset spellings sanitizing to the same folder (`hemibrain:v1.2.1` vs `hemibrain_v1_2_1`) — colliding `dataset_data/` folders and caches, separate columns (confirmed, exotic input) |
| F-XD-009 | Cross-dataset | Query-report mini-network colors nodes by edge polarity, not query roles (a queried target rendered red when it appears as an edge source) (confirmed, presentation) |
| F-XD-010 | Cross-dataset | Explicit empty per-dataset threshold override `[]` falls through to the global list (`if override:`) (confirmed, intent ambiguous) |
| F-RV-001 | Fix-round residual | `_parse_path_nodes`'s `except (ValueError, SyntaxError)` is not exhaustive: `ast.literal_eval('{[1]:2}')` raises TypeError (probe-confirmed on this Python 3.11.5) — no path writer produces such cells, but broadening to `except Exception` costs one word (confirmed) |
| F-RV-002 | Fix-round residual | Quarantine filename has no same-second collision guard; POSIX rename silently overwrites a prior `.corrupt-<ts>` recovery copy (confirmed, requires two corruptions within 1 s) |
| F-RV-003 | Fix-round residual | `connectivity_profiler` json.loads narrowed to `(TypeError, ValueError)` — a RecursionError on pathological nesting now propagates (theoretical); leftover `.parquet.tmp`/`.corrupt-*` files are invisible to `storage_inventory` (cosmetic) |

## Review of the fix round (c80c285..553a7f5) — verdicts

1. **coana path-node parser (`_parse_path_nodes`/`_path_has_unconserved_edge`)**: net improvement. Input-shape matrix probed: identical for list/tuple literals, malformed literals, empty/whitespace, bytes passthrough, numpy str; **better** for bare globals that eval would resolve (`'os'`, `'np'` — old code crashed on `len(module)`), expressions (the security fix), and ellipsis/lambda; **worse only** for unhashable-literal cells (F-RV-001) and parse-time MemoryError. Crash surface shrank overall: the old apply path already crashed on TypeError from eval results.
2. **`sorted(set(layer_nodes))`**: behavior-preserving. `_fetch_direct_connections_for_nodes` batches the full list (1000/batch, set-membership filter); `_fetch_neurons_local_or_api` is `isin`-based — both total and order-free; no `head()`/`[:N]` caps found on these paths. Only batch boundaries/output ordering became deterministic (the intent).
3. **58-site except-narrowing sweep**: clean. Every changed try wraps ordinary library calls (neuprint fetch, parquet/csv IO, literal_eval, spearmanr, tqdm, unlink/rmdir, SIGALRM teardown whose raisers are ValueError/RuntimeError — all `Exception`). No `queue.task_done`, no `sys.exit`, no generator-close, no asyncio cancellation inside any changed try. Two adjacent intentional fixes verified (cache-load failure reporting; `_tag_scene_members`).
4. **Profiler consolidation rewrite**: quarantine+abort flow correct and regression-tested both engines; tmp+rename filenames verified (`connectivity_profiles.parquet.tmp`, not matched by any reader glob); residual = same-second collision (F-RV-002).
5. **`_type_level_matrix` rewrite**: value-identical when all members are present (element-for-element pairing proof); `_pair_ok` with empty sides returns True for every pair, unchanged.
6. **Network-tab hemisphere guard**: reset lives only in the disable branch, triggered solely by the checkbox transition and page build; nothing else writes those elements; e2e-covered.
7. **Docs in range**: harness/ paths exist; renumber verified with a repo-wide §-reference grep (all OUTPUT_FILES-anchored refs are §9/§9a-bis and still correct); sim_roi wording matches the writer. One pre-existing cosmetic: NeuronBridge subsections still numbered 9a/9a-bis under (now) §10.

## Finding details (P1/P2)

### F-PF-001 — streaming keyword sentinel bypass
- Statement: `sv.process_paths_streaming(..., keyword_in_path_to_remove=self.keyword_in_path_to_remove, ...)` (`src/coana.py:18051`) passes the attribute raw; the class default is the sentinel `'None'` (`coana.py:3334`) and the UI injects `['None']` when no keyword is entered (`ui/tabs/find_path.py:331`). `_normalized_keyword_filter()` (`coana.py:12988-12999`) exists precisely to neutralize this ("must NEVER reach path_filter as a literal keyword") and IS used by the in-memory filters (`coana.py:18207`, `:18269`) — the streaming writer never goes through it, and `statvis.process_batch_polars` applies any truthy value literally (`src/statvis.py:7994-8006`: `str.contains(kw, literal=True)`).
- Impact: with the default sentinel, every type path whose label contains the substring "None" is silently excluded from `..._allpaths_type.csv` and misfiled into the excluded CSV — the streamed canonical export and the in-memory/visualization frames can disagree. Cross-dataset comparison delegates runs with `drop_untyped=False` (untyped labels reachable), where such labels are plausible.
- Reproduction: default-config run with any node label containing "None"; compare streamed CSV row count vs the in-memory filter result.
- Proposal: pass `self._normalized_keyword_filter()` at `coana.py:18051` (streaming writer treats None as no-op) and add a fixture test.

### F-PF-002 — graph-cache key missing `hemisphere_filter`
- Statement: `_findallpath_cache_key` (`src/coana.py:545-584`) encodes `separate_hemispheres` (hemi_flag) but not `hemisphere_filter`; the fetch-level filter (`coana.py:2059-2071`) drops the mirror hemisphere's edges BEFORE the `if not self.separate_hemispheres: return` early-out, i.e. it changes the cached edge set in both modes. The key builder's own docstring ("must include every parameter that changes which edges the fetch returns ... hemisphere mode") states the violated rule.
- Impact: in any same-process sequence (notebook, script, sequential API calls) that changes `hemisphere_filter`, the second run reuses the first run's cached `'all_connections'` tables with no hemisphere re-check → wrong-hemisphere results, no warning. The UI escapes only because every tool run is a fresh subprocess.
- Reproduction: same instance, two FindAllPath calls differing only in `hemisphere_filter`; second reuses cache (add a `use_graph_cache` probe).
- Proposal: add `hemisphere_filter` to the key; while there, consider label-mapper identity (secondary gap — mapping runs at fetch time and feeds `drop_untyped`).
- Validation: two-run test asserting distinct cache keys / distinct graphs.

### F-XD-001 — intra-dataset bodyId metric used across two connectomes
- Statement: `direct_comparison` (explicitly cross-dataset; called per dataset pair by `ComparisonAnalyzer.connectivity_profile_comparison`, `comparison_analyzer.py:11244-11264`) in strict mode with type-name inputs delegates to `ProfileComparator.compare_types_bodyid_core(profiler, types, dataset_a, dataset_b, ...)` (`profile_comparator.py:2108-2117`), which scores each pair with `combined_score_intra_dataset` (`:3234`) — cosine/weighted-jaccard over bare-string-bodyId partner vectors with `shared_bodyids = set(a) & set(b)` (`:1287-1322`) — and gates on `len(set(partner_set_a) & set(partner_set_b))` (`:3245-3252`).
- Impact: when the same integer exists in both ID spaces (FAFB↔BANC share the FlyWire root-id family; manc↔male-cns share small ints), unrelated neurons count as shared partners and inflate the exported `direct_comparison_*.csv` scores; for magnitude-disjoint pairings the same code yields ~0 — "no similarity" where the honest answer is "not comparable in this metric". The metric's own name says intra.
- Reproduction: strict-mode direct comparison of one type present in two datasets whose bodyId ranges overlap.
- Proposal: route cross-dataset calls through the type-level standardizing comparator (as loose mode does), or intersect partners by resolved type names; at minimum, fail closed with a "metric not defined across ID spaces" marker.
- Validation: synthetic two-dataset fixture with colliding bodyIds asserting no inflation.

### F-XD-002 — profile cache ignores `min_synapse_threshold`
- Statement: cache validity gates only on `profile.top_k_bodyid_used >= required_k` (`connectivity_profiler.py:1924`, `:1948`); profiles are built with `weight >= min_syn` (`:2734`) and the serialized row records no threshold (`to_dict`, `:505-534`). `_get_config_hash` (`:1492-1507`) hashes exactly the missing fields (min threshold, untyped/fuzzy toggles) and has exactly one grep hit — its definition. The knob is exposed per run in the UI and forwarded into `ProfilerConfig` (`ui/tabs/connectivity.py:149-152`, `:490`; `profile_comparator.py:2764-2770`).
- Impact: changing Min Synapse Threshold (or the untyped/fuzzy toggles) silently reuses profiles built under the previous setting — homolog/similarity scores computed under the wrong filter.
- Reproduction: build profiles at min_syn=3; rebuild at min_syn=50 same dataset; assert reuse (cache hit log) despite different content.
- Proposal: store the config hash (the dead helper) in the row and gate on equality.
- Validation: two-config test asserting a rebuild.

### F-PF-003 — `fit_edge_budget` IndexError at the virtual boundary
- Statement: `hi_idx = len(distinct)` as a virtual bracket (`coana.py:1327-1329`); if bisection exhausts `max_probes` while every probed tier fit and `lo_idx < len(distinct)-1`, `stats['landing'] = distinct[hi_idx]` (`:1383-1384`) indexes past the end. Requires ≥ ~2^5 distinct weight tiers whose closures all fit while `E₀ > cap` (heavy-tailed cones) — the doc's own measured fit case used 7 of 8 probes.
- Impact: hard crash of FindAllPath, no output folder/marker. Confidence: likely (arithmetic reconstructed; not executed).
- Proposal: clamp `hi_idx` to `len(distinct)-1` for the landing read (landing = last real tier).

### F-PF-004 — provenance note claims a filter that was skipped
- Statement: the note is written whenever `keep_only_hemisphere_conserved_connections` is truthy (`coana.py:13617-13621`); all four filter gates also require `self.separate_hemispheres` (`:12196`, `:12732`, `:17691`, `:18241`). With (True, False) — reachable via API/config; the UI force-clears — no conservation filtering runs anywhere, and the note asserts it did.
- Impact: `user_warning_notes.txt`/parameters claim conserved-only output that is unfiltered.
- Proposal: gate the note identically, or run the type-level filter regardless (pick the contract).

### F-PF-005 — metadata re-stamp can silently fail
- Statement: `_write_run_metadata`'s post-enumeration rewrite of `all_attributes.json`/`parameters.txt` is `except Exception: pass` (`coana.py:15268-15298`); failure leaves the pre-enumeration snapshot (no final tau/floor/applied-threshold block) with no marker. Related: `density_meta.json` pointer write `:15173` and keep-larger-cone guard `:15044` are also `pass`-on-error.
- Proposal: log + a `.provenance_incomplete` marker on failure.

### F-XD-003 — cached comparison load without a query fingerprint
- Statement: `_try_load_cached` (`comparison_analyzer.py:4544-4603`) accepts any non-empty `connections_edge.csv` under `dataset_data/<safe>/minsyn_<N>`; the writer (`:4605-4621`) persists path-mode output under the same name ("edge mode cached version"). Nothing validates source/target sets, `max_interlayer`, mode, or pathfinding identity. `run_comparison(skip_existing=True)` is the default.
- Impact: reusing a `saveas` folder after changing the query (or switching modes) silently loads the previous query's edges as the result; a stale path-mode file outranks fresh edge-mode aggregation.
- Proposal: stamp a query fingerprint (source/target digests + mode + interlayer) in the folder and validate on load.

### F-XD-004 — edge-mode provenance describes the wrong run
- Statement: in edge mode the compared data is exactly `weight >= requested` (`comparison_analyzer.py:3928`), but the exported `effective_thresholds.json`/banner rows come from `_path_provenance_row` of the side-effect path runs (step 4 runs the path tool "for output consistency only", `:3894-3902`).
- Impact: a budget-bitten side run labels the dataset's applied threshold above requested although the edge data contains every edge ≥ requested; `comparability_report` keys off the same values.
- Proposal: edge mode should export its own `applied = requested` provenance row.

### F-XD-005 — `direct_comparison` bypasses the shared type resolver
- Statement: `direct_comparison` has no `type_mapper` parameter; its loose path calls `batch_compare_cross_dataset(...)` without one (`profile_comparator.py:2211-2216`), while HomologFinder and the main metrics alignment resolve partner names through the shared resolver for the same run.
- Impact: cross-dataset similarity in the `connectivity_profile_comparison/` export systematically under-scores renamed partners — inconsistent with every adjacent surface.
- Proposal: thread the analyzer's resolver into the call (it is available at the call site).

### F-XD-006 — swallowed failures can empty homolog results
- Statement: pre-warm failures `pass` (`profile_comparator.py:3746-3754`); `ensure_data_available(raise_on_missing=True)` is downgraded to a log (`:3791-3794`); the per-bodyId build loop swallows every exception (`:3818-3839`). A systematically broken cache therefore yields "Built 0 profiles" and an empty/partial result rather than an error. (`comparison_analyzer.py:6077-6080` can likewise drop `query_resolution.csv` silently.)
- Proposal: count and report build failures; abort when the build success rate is zero.

## Verified clean (evidence-checked highlights)

- **Enumerators**: `find_paths_strongest_first` pop bound = exact best completion (targets hold W=∞), τ_canonical = w2+1 tracked at both bite points, deterministic; `find_paths_shortest_strongest_first` per-target DAG + maximim DP correct, k-way lazy merge, no reordering under bite; `find_paths_shortest_backward` exists but is genuinely uncalled by the pipeline (docs agree).
- **Budgets/pruning**: hop-budget keep-rule and fixpoint iteration lossless; edge-budget floor order (lossless prune → fit → single lossless re-closure) correct; shortest never floored; empty-probe and declined paths honest.
- **Threshold provenance**: per-dataset folders and comparison exports consume the one `applied_threshold_provenance` formula; replay per-slice finalization correct; `max_paths_bodyid 0→1,000,000` enforced at all three sites and is a query-wide cap (docs consistent).
- **Cross-dataset joins**: conn assembly, type lookups, and FNC caches are single-dataset; aligned comparison matrices are type-level via label/type mapper + merge policy with untyped fallback dropped by default — no bare-bodyId cross-dataset joins outside F-XD-001.
- **Caches (cross-dataset)**: `_FNC_CACHE` dataset-scoped; profile disk paths canonicalized; graph-cache key includes source/target digests + depth + drop_untyped + filter level (modulo F-PF-002).
- **Hemisphere across datasets**: per-dataset suffixing, symmetric mirror zeroing, mapper strips/re-appends suffixes; the report UI disables the mirror toggle without separate_hemispheres.
- **Report machinery**: network-section DOM ids unique at every site; the auto-mode two-section IIFE state fix present.
- Six PATHFINDING_PIPELINE.md claims spot-checked accurate (max_passes/max_probes, UI defaults, auto→1M, no target early-stop in 'all' mode, shortest-backward unused, per-target hop limits).

## Validation record

| Date | Check | Result | Scope |
|---|---|---|---|
| 2026-09-25 | `git diff c80c285~1..553a7f5` hunk review (lead auditor) | sweep diffs are pure except-line changes (31 sites counted); coana hunks exactly as described | 13 commits |
| 2026-09-25 | `ast.literal_eval('{[1]:2}')` probe | raises TypeError (escapes `(ValueError, SyntaxError)`) — F-RV-001 confirmed | local Python 3.11.5 |
| 2026-09-25 | line verification of every P1/P2 claim (sentinel call site + statvis application + normalizer; cache-key fields + fetch filter ordering; strict-mode delegation + intra metric + overlap gate; config-hash grep = 1 hit + validity gate; note vs 4 gates; type_mapper absence; `_try_load_cached` + writer) | all confirmed at cited lines | this report |
| 2026-09-25 | downstream-cap check for `sorted()` change | both fetch functions total/order-free; no caps | reciprocal path |

## Residual uncertainty

- F-PF-003 is arithmetic-reconstructed, not executed (needs a ≥32-tier synthetic cone).
- F-PF-001's data impact requires labels containing the substring "None" (plausible with `drop_untyped=False`; not demonstrated on a real dataset).
- F-XD-001's real-world reachability depends on strict mode being selected (loose is the default verification mode).
- F-XD-004 medium-high confidence: mechanism verified; no edge-mode-specific guard found, but a contrary guard could exist outside the traced span.
- The user's live WIP (pooling/morphology, growing during this audit) was not part of this scope.

## Audit completion statement

Read-only audit; no implementation code, tests, docs, or outputs were modified. All fixation directions above are proposals. Earlier fix-round commits were reviewed but not altered.

---

## Fix round — 2026-09-25 (same day)

Commits `1cc62aa` (fix(pathfinding)) and `bb99268` (fix(comparison)); 18 new/updated tests. Statuses below supersede the `open` markers above.

| Disposition | Findings | Evidence |
|---|---|---|
| fixed, verified | F-PF-001 (normalized filter at the streaming call + `_keyword_filter_batch` seam, both tested), F-PF-002 (`hemisphere_filter` in the key, distinctness test), F-PF-004 (IGNORED note + combo warning, file-level test), F-PF-005 (loud failure + `.provenance_incomplete` marker, test), F-PF-006 (dead `'conn_layers' in locals()` fallback → `all_connections`), F-PF-007 (sorted derivation, order-stability test), F-PF-008 (knob snapshot/restore at all three returns), F-PF-009 (docstring corrected), F-PF-010 (runtime warning), F-XD-001 (same-dataset gate + fall-through, both directions tested; the old test that encoded the colliding-partner scenario updated), F-XD-002 (config_hash in rows + both tiers gated + write-site stamps; legacy rows accepted; test), F-XD-003 (fingerprint sidecar written+validated; legacy accepted; test), F-XD-004 (edge-mode provenance neutralized with `side_path_run` preservation; test), F-XD-005 (`type_mapper` threaded from both analyzer call sites), F-XD-006 (build failures counted/reported; total-failure abort — None-results stay normal), F-XD-007 (empty sets participate in the intersection), F-XD-008 (ValueError on sanitized-name collision; test), F-RV-001 (except Exception + unhashable-literal test), F-RV-002 (same-second collision suffix) | commits; targeted suites 151+448+117+100+15 green |
| downgraded after deeper analysis | F-PF-003 — closures under `prune_layers_hop_budget` are provably monotone (a weaker mask's closed cone is a superset), and the one-shot landing sits at the fit boundary, so the gallop's first probe past the landing already meets a real unfitting tier; the virtual-bracket + probe-exhaustion crash appears **unreachable**. The clamp is kept as a defensive guard with smoke tests (no reproduction exists). Original confidence "likely" was too generous. | fix-round analysis in commit 1cc62aa |
| premise corrected, no change | F-XD-009 — `_query_report_network_card`/`_query_report_overlap_table` have **zero callers** anywhere (dead code); the polarity-coloring issue is unreachable. Recorded; not modified. | repo-wide grep |
| documented, no behavior change | F-XD-010 — `dataset_thresholds` is a deprecated legacy field; the `[]` fall-through is documented in the field docstring rather than changing legacy replay semantics. | comparison_parameters.py field doc |

### Validation record (fix round)

| Date | Check | Result |
|---|---|---|
| 2026-09-25 | targeted suites: pathfinding (hop-budget, edge-budget-floor, budget-fit+shortest, pathfinding, find-network, applied-threshold-provenance, path-node-parsing, new audit-2 file) | 151 passed |
| 2026-09-25 | targeted suites: statvis, profile-comparator, connectivity-profiler, comparison-analyzer, label-mapper | 448+117 passed (after fixing the new abort's None-handling and updating the strict-types contract test) |
| 2026-09-25 | targeted suites: comparison-parameters, threshold-alignment, duplicate-threshold-skipping, applied-threshold banners/provenance | 100 passed |
| 2026-09-25 | hang diagnosis: `direct_comparison` offline tests must stub `coana.FindNeuronConnection` (its prewarm closure always builds a live FNC → NeuPrint fetch with no token hangs) | faulthandler stack dump; tests stubbed |
| 2026-09-25 | full `tests/core` + `tests/ui` battery | **core 4,931 passed / 1 skipped / 0 failed (983 s); UI 1,023 passed / 0 failed (825 s)** — 5,954 tests green post-fix |

No implementation changes beyond the fixes described; the owner's live WIP
files (docs/skills/UI pass) were not touched — both commits contain only the
six src files and four test files of this round.

---

## Verification round — 2026-09-25 (real data, exports, doc alignment)

Commits `c133596` + `b17a839` (docs/alignment; one docstring fix). Real-data
runs over the local FAFB v783 + BANC v888 bundles:

| Check | Result |
|---|---|
| **A. FindAllPath ×2** (s-LNv→DN1a, FAFB, same process, `hemisphere_filter` both→left) | Both complete (~90 s each). Run A: 242-row pruned cone, 83 type paths; run B: **75-row cone, 26 paths, zero `_R` labels** — the graph cache did NOT leak across the hemisphere change (F-PF-002 end-to-end). Final provenance re-stamp landed in both (tau 3, `applied=requested`, no `.provenance_incomplete`). Notes carry the honest pruning/tau lines; reciprocal exports populated (654 bodyId / 343 type rows); streamed type CSV clean (no sentinel artifacts). |
| **B. tmvev auditor** on the real `tmvev-cert-20260925_0147` queue | Tool works on real data; its 9 FAILs are pre-fix-run states (FAFB→hemibrain ungraded-pass scenario + bar-column schema drift) inside the owner's in-flight TM VEV workstream — not re-litigated here. |
| **C. Profile cache** (real 139,255-row FAFB cache, scratch copy) | Legacy rows accepted; a rebuilt profile's batch file carries `config_hash`; memory + disk tiers gate on it; config restore re-serves (F-XD-002 end-to-end). |
| **D. Comparison** (edge + path mode, FAFB+BANC, threshold 3) | Edge mode: `effective_thresholds.json` + `pathfinding_provenance.csv` rows report `applied=3=requested`, tau null (F-XD-004). Path mode: every `dataset_data/<ds>/minsyn_3/` carries `connections_edge.csv` + the fingerprint sidecar; a same-folder different-query rerun logs "written for a different query — ignoring it" for BOTH datasets and re-derives (F-XD-003). Full `export_results()` set (100+ files) inspected: unified edge table (188 rows), summary, report.html (423 KB), query_resolution, auto-mapping artifacts all present and well-formed. |

Issues found and fixed this round: the F-XD-010 field-docstring disposition
had been reported but never edited (now actually written); one more
`src/vispath.py` historical reference; the fingerprint doc sentence now
names its writer (path mode) and points edge-mode readers at
`edge_mode_data/`. Two test-script bugs on my side (missed
`InitializeNeuronInfo()` — the edit-to-run template calls it explicitly;
missed the canonical `export_results()` step) — no product defects.

Post-change tests: comparison-parameters + audit-2 suites green (58);
docs link checker green (183 files). A full battery was not re-run — this
round's only code change is a docstring (inert); everything behavioral
landed before the previous full battery.

---

## Follow-up round — 2026-09-25 (user directives on the four open questions)

| Directive | Action | Evidence |
|---|---|---|
| cert queue: don't touch | untouched | — |
| find_reciprocal: dig deeper, test, fix properly | The trap was deeper than reported: `_find_paths_core` already contained the None-keyed defer-to-field branch (with a comment stating the intent) — the public wrappers' `bool = False` defaults made it dead code, and the UI runner carried an explicit workaround naming the trap. A second symptom surfaced while digging: `user_warning_notes` read the FIELD, so an explicit method-level False override of a True field would claim enrichment that did not run. Fixed at all four signatures (`Optional[bool] = None`), run-truth flag added (`_find_reciprocal_ran`), note keys on execution; UI workaround retired. Failing-first: 4 of 6 tests red before the fix. Real-data verified: FAFB s-LNv→DN1a with constructor-field-only produces the `find_reciprocal/` subfolder + note. | commits `c7744a6`; `tests/core/test_find_reciprocal_deferral.py` (7) |
| vispath: make it better | Structural: PEP 562 lazy resolution for `VisualizePath` (coana) and `FastGraph`/`DiGraph` (the `core.fast_graph` shim) — the wheel addendum is closed. `import coana`/`statvis` now succeed without the subproject; enumeration/visualization raise on first use with the exact install command. Wheel-validated both ways in a scratch venv; subprocess test pins the contract; INSTALLATION.md updated. | commit `3c92694` |
| artifacts: remove | two FAFB run folders + `/tmp/drocat_realdat*` + wheel scratch removed | — |

Correction during this round: the first lazy-import implementation used PEP
562 module __getattr__, which does not serve bare in-module global
references — 'G = FastGraph()' inside coana raised NameError, caught by the
full battery as 53 pipeline-test failures. Replaced with guarded imports +
call-raising placeholder classes (identical behavior on every access path);
all 53 recovered.

Post-change validation: deferral + import-affected suites green; FULL
battery green — core 4,939 passed / 1 skipped / 0 failed (939 s), UI 1,023
passed / 0 failed (828 s).
