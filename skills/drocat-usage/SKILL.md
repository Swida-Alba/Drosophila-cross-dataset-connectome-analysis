---
name: drocat-usage
description: Layer-1 tab-matched direct analyses for DROCAT v4.5.0. Run, explain, debug, and safely execute the exact backend tools each UI tab drives, without reopening the NiceGUI UI. Use when an agent needs to reproduce a specific UI tab's scientific analysis end-to-end — pathfinding (complete/shortest/network), Connectivity tab (find similar via connectivity profiles, profile comparison), Morphology tab (find similar via skeleton shape), cross-dataset comparison, NeuronBridge (lines/neurons/co-labeling), FlyLight download, 3D skeleton / path visualization, or dataset & cache management — from a direct Python call using the tab's own parameters.
---

# DROCAT Direct Analysis — Layer 1 (Tab-Matched)

This is the **tab-matched** layer: for each UI tab it reproduces the exact backend
tool that tab invokes, using the same parameters the UI builds. Use it when a
request maps to one of the UI tabs and you want the UI-plugin-equivalent analysis
run as a direct script. For free-form composition of backend functions across
modules, use the [Layer-2 backend skill](../drocat-backend/SKILL.md). For
installation/environment repair, use the [install skill](../drocat-install/SKILL.md).

Keep the UI out of the execution path: build a focused script or run the tool
directly, execute it in the versioned environment, inspect the generated
artifacts, and report the exact parameters and output paths.

## Completion contract

Unless the user explicitly asks for a plan or diagnosis only, finish the
requested operation end-to-end in the current agent session:

1. select the matching tab recipe (or chain two recipes);
2. prepare the smallest safe run with the tab's parameters;
3. execute it with the requested parameters;
4. validate exit status and generated artifacts; and
5. report the command, output paths, counts, warnings, and any scientific
   limitation.

Do not stop after reading source files, creating a run script, or printing a
command. Ask the user only for a required credential, scientific choice, or
approval; otherwise continue until the task is complete. If a real blocker
remains, state the exact blocker and the last completed step.

## Shared runtime

1. Confirm the repository root and branch before running anything:

   ```bash
   cd /path/to/repo
   git branch --show-current
   git status --short
   ```

   Use branch `v4.5.0`. If the checkout is elsewhere, resolve that path first; do
   not clone a second copy over an existing project.

2. Use Python 3.11 in the `drocat-4.5.0` environment. Prefer
   `conda run -n drocat-4.5.0 ...` for a reproducible non-interactive command;
   an already activated environment is also acceptable. Set
   `PYTHONNOUSERSITE=1` so user-site packages cannot shadow the pinned
   environment.

3. Run a script through the shared launcher so template relative paths resolve
   from the script's own directory and `PYTHONPATH` is set for `src/`, the repo
   root, and `vispath-subproject/src`:

   ```bash
   python skills/drocat-usage/scripts/run_direct.py \
     --conda-env drocat-4.5.0 \
     --script scripts/FindPath.py
   ```

   Use `--dry-run` to inspect the resolved command and working directory. The
   files under `scripts/` are configuration templates, not a stable CLI — copy a
   template into `archive/scripts_local/` (or patch an explicitly requested
   source file) before a real scientific run.

4. Never print, commit, or paste `config.json`, API keys, cookies, or full
   authentication environment variables. Read credentials through the repository
   token manager or the environment. Ask before changing tokens, downloading
   large datasets, or deleting/replacing result folders.

5. Preserve the example templates. Before a substantial edit, copy the script to a
   run-specific file or create a new script in an explicitly named output
   directory. Use `apply_patch`/`SearchReplace` for code changes, then compile the
   changed file and run the smallest relevant test.

6. Use absolute output paths when possible. Keep each run isolated, record the
   dataset, source/target queries, filters, cache mode, and output directory, and
   inspect the generated CSV/HTML/JSON before claiming success.

## Choose the tab recipe

Each tab recipe in [`tabs/`](tabs/) follows the same shape: the `tool_key`, the
backend import/class/method, the exact parameters the UI builds, the method call,
expected outputs, and a runnable launcher command.

| UI Tab | Tab recipe | Backend tool_key → class.method |
| --- | --- | --- |
| Complete Paths | [find-path.md](tabs/find-path.md) | `find_path` → `FindNeuronConnection.FindAllPath` |
| Shortest Paths | [find-shortest.md](tabs/find-shortest.md) | `find_shortest` → `FindNeuronConnection.FindShortestPath` |
| Find Network | [network.md](tabs/network.md) | `find_network` → `FindNeuronConnection.FindNetwork` |
| Connectivity · Comparison | [connectivity-profiling.md](tabs/connectivity-profiling.md) | `connectivity_profiling` → `ConnectivityProfileComparer.run` |
| Connectivity · Find Similar | [find-homologs.md](tabs/find-homologs.md) | `find_homologs` → `HomologFinder.find_homologs_multi` (cross-dataset homologs; Target = Source for intra-dataset similar search) |
| Morphology · Find Similar | [find-similar.md](tabs/find-similar.md) | `find_similar_morphology` → `MorphologyComparer.find_similar` |
| Morphology · Comparison | [morphology-comparison.md](tabs/morphology-comparison.md) | `morphology_comparison` → `MorphologyProfileComparer.run` |
| Cross-Dataset Comparison | [inter-dataset.md](tabs/inter-dataset.md) | `inter_dataset` → `ComparisonParameters` + `ComparisonAnalyzer.run_comparison` |
| Find Driver Lines | [nb-find-lines.md](tabs/nb-find-lines.md) | `nb_find_lines` → `NeuronBridgeFinder.find_lines_batch`; `nb_find_lines_expanded` → `ExpandedLineFinder.run` (coverage-routed cross-dataset name expansion) |
| Find EM Neurons | [nb-find-neuron.md](tabs/nb-find-neuron.md) | `nb_find_neuron` → `NeuronBridgeFinder.find_neurons_batch` |
| Co-Labeling Analysis | [nb-colabel.md](tabs/nb-colabel.md) | `nb_colabel` → `NeuronBridgeFinder.analyze_colabeling` |
| FlyLight Image Download | [flylight.md](tabs/flylight.md) | `flylight_download` → `FlyLightDownloader.download` |
| 3D Skeleton + Path Network | [visualization.md](tabs/visualization.md) | `plot3d_skeleton` → `VisualizeSkeleton.plot_neurons`; `plot_path` → `VisualizePath.visualize` |
| Settings / Datasets & Cache | [settings.md](tabs/settings.md) | dataset_service, `SkeletonPuller`, cache builders |

## Standard agent workflow

1. **Clarify the analysis**: dataset (NeuPrint vs FAFB/BANC local), source/target
   queries, direct vs multi-hop and max layers, thresholds and filters, desired
   output format (CSV/XLSX/HTML/images/PDF/PPTX), output folder, and whether cache
   may be used. If underspecified, make the smallest safe assumption and state it
   before any remote query.
2. **Open the matching tab recipe** in [`tabs/`](tabs/), not the whole repo.
3. **Prepare** the focused run script from the recipe's parameters (explicit
   values, absolute output path, `showfig=False` for unattended runs).
4. **Execute** via the launcher and capture the exit code and last log lines.
5. **Inspect** the generated files (rows/HTML content) — a zero exit code alone is
   not enough.
6. **Report** the exact run command, parameters that matter, output folder,
   row/file counts, warnings, and any follow-up visualization.

## Direct-script safety and repair

- Validate dataset spelling against the configured dataset list before an API
  call. NeuPrint names include versions (`hemibrain:v1.2.1`, `male-cns:v0.9`);
  FAFB uses `flywire_FAFB_v783`; standalone BANC uses `banc_v888` or
  `banc_v626`.
- Keep `use_cache=True` for repeat work. Use `cache_only=True` only after checking
  cache coverage.
- Prefer CSV for large intermediate tables; use Excel only when the user needs
  workbook sheets.
- Limit the run with `edgeN_limit`/`max_paths_bodyid`/`graph_edge_limit_bodyid`
  before generating an interactive HTML.
- For 3D renders start with `skeleton_mode="line"` and disable exports during a
  first smoke test; use tube/WebDriver/PDF/video only after the HTML is valid.
- For PlotPath pass `path_file` from a completed FindPath result; use
  `generate_empty_network=True` and `path_file=None` only for an empty editable
  Cytoscape canvas.
- When modifying a backend, reproduce the bug with the smallest local fixture,
  patch the narrowest function, run `python -m py_compile` and the relevant tests,
  then rerun the original command.
- Do not "fix" a failing run by silently lowering thresholds, switching datasets,
  or disabling validation. Explain the trade-off and ask if it affects the
  scientific interpretation.

## Chaining tabs

For workflows that need more than one function (e.g. FindPath → PlotPath, or
similar-neuron → 3D skeleton), see [references/combinations.md](references/combinations.md).

## References

- [Tab recipes](tabs/) — one per UI tab
- [Tool catalog and backend contracts](references/tool-catalog.md)
- [Workflow recipes](references/workflow-recipes.md)
- [Cross-tab combinations](references/combinations.md)
- [Datasets, authentication, and output files](references/datasets-and-auth.md)
