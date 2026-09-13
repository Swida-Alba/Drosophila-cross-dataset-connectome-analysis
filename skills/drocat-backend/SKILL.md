---
name: drocat-backend
description: Layer-2 backend module usage for DROCAT v4.5.0. Use the backend function blocks and modules directly for more flexible combinations than a single UI tab — composing FindNeuronConnection, MorphologyComparer, comparison (ComparisonParameters/ComparisonAnalyzer, ConnectivityProfileComparer, HomologFinder), NeuronBridgeFinder, FlyLightDownloader, VisualizeSkeleton, VisualizePath, and supporting utilities into custom analyses. Use when a task needs functions from several tabs, a non-tab scientific workflow, or direct control over module-level parameters. For a single tab's exact analysis, use the layer-1 drocat-usage skill; for installation, use drocat-install.
---

# DROCAT Backend Modules — Layer 2

This is the **backend function** layer. It documents the composable building
blocks the UI tabs use, so you can build a custom analysis that is not limited to
one tab. Every class below is a stable programmatic API; the methods and
parameters are the ones the UI scripts actually call. Use it when you need to
chain functions across tabs or set parameters the UI does not expose.

For a tab-identical single analysis, use
[Layer 1 `drocat-usage`](../drocat-usage/SKILL.md). For installation/environment
repair, use [the install skill](../drocat-install/SKILL.md).

## Completion contract

Unless the user explicitly asks for a plan or diagnosis only, finish the
requested operation end-to-end in the current agent session: prepare the smallest
safe composition, execute it in the versioned environment, validate the artifacts,
and report the command, output paths, counts, warnings, and any scientific
limitation. Ask the user only for a required credential, scientific choice, or
approval; otherwise continue until the task is complete.

## Shared runtime

1. Confirm the repository root and branch (`v4.5.0`). Do not clone a second copy
   over an existing project.
2. Use Python 3.11 in `drocat-4.5.0`; prefer `conda run -n drocat-4.5.0 ...` and
   set `PYTHONNOUSERSITE=1`.
3. Set `PYTHONPATH` for `src/`, the repo root, and `vispath-subproject/src`. The
   layer-2 launcher resolves the repo root automatically and does this for a
   script:

   ```bash
   python skills/drocat-backend/scripts/run_module.py \
     --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Module_<date>.py
   ```

   The layer-1 launcher also works:
   `python skills/drocat-usage/scripts/run_direct.py --conda-env drocat-4.5.0 --script ...`.
   For an inline probe, run directly with `PYTHONPATH` set
   (`PYTHONNOUSERSITE=1 PYTHONPATH=src:vispath-subproject/src:. conda run -n drocat-4.5.0 python -c ...`).
4. Never print or commit tokens/`config.json`. Read credentials through the
   repository token manager. Ask before large downloads or deleting result folders.
5. Preserve templates; put custom code in a run-specific file under
   `archive/scripts_local/` or an explicitly named output dir.

## Choose a module

| Module | Class(es) / entry points | Typical use |
| --- | --- | --- |
| [coana-connectivity](modules/coana-connectivity.md) | `FindNeuronConnection` | direct edges, all/shortest paths, network, connection cache |
| [morphology-similarity](modules/morphology-similarity.md) | `MorphologyComparer`, `SkeletonVectorCache` | morphological similarity, skeleton vector cache |
| [comparison](modules/comparison.md) | `ComparisonParameters`, `ComparisonAnalyzer`, `quick_compare`, `CrossDatasetTypeMapper`, `LabelMapper` | cross-dataset comparison, type mapping |
| [profile-comparator](modules/profile-comparator.md) | `ConnectivityProfileComparer`, `HomologFinder`, `ProfileComparator` | connectivity profiles, homologs, profile similarity |
| [morph-cross-dataset](modules/morph-cross-dataset.md) | `CrossDatasetMorphComparer`, morph qualification helpers | cross-dataset morphology comparison, null-calibrated homolog qualification, BANC bootstrap |
| [neuronbridge](modules/neuronbridge.md) | `NeuronBridgeFinder` | EM↔LM lines, neurons, co-labeling |
| [flylight](modules/flylight.md) | `FlyLightDownloader` | FlyLight image/metadata download |
| [visualize-skeleton](modules/visualize-skeleton.md) | `VisualizeSkeleton` | 3D morphology, ROI meshes, exports |
| [vispath](modules/vispath.md) | `VisualizePath` | interactive network/Sankey/heatmap, empty editable canvas |
| [util-support](references/util-support.md) | `token_manager`, `neuron_filter`, `flywire_readiness`, `cache_manager`, `roi_screening` | data prep, token/readiness helpers |

## Verify a signature before composing

Do not guess a backend signature — probe it at import time:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src:vispath-subproject/src:. conda run -n drocat-4.5.0 python -c \
  'import inspect; from coana import FindNeuronConnection as C; print(inspect.signature(C)); print(inspect.signature(C.FindAllPath))'
```

## Composition guidance

- Compose by passing each step's output artifact into the next (e.g. a
  `FindNeuronConnection`/comparison CSV into `VisualizePath(path_file=...)`).
- Keep `showfig=False` and bound the run (`edgeN_limit`, `max_paths_bodyid`,
  `graph_edge_limit_bodyid`) until the output is validated.
- When combining across datasets, keep type mapping/labels consistent
  (`use_auto_type_mapping` / `ComparisonParameters.auto_type_mapping` /
  `LabelMapper` / `CrossDatasetTypeMapper`) — see [comparison](modules/comparison.md).
- For offline/deprecated datasets use `cache_only=True` only after checking cache
  coverage; never silently switch datasets to make a run pass.

## References

- [Module index (all functions & signatures)](references/module-index.md)
- [Supporting utilities](references/util-support.md)
- [Cross-module composition recipes](references/combinations.md)
