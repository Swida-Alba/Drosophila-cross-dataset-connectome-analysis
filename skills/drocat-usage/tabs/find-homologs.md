# Connectivity · Find Homolog (find_homologs)

Reproduce the **Connectivity tab → Find Homolog sub-tab** as a direct backend
call. With different `source_dataset` / `target_dataset` this is the
cross-dataset homolog search; with `target_dataset == source_dataset` it is
the within-dataset (intra-dataset) similar-neuron search. Both run the same
engine on connectivity profiles (fast adjacency search or the slower
comprehensive search).

## Backend contract

- **tool_key:** `find_homologs`
- **import:** `from comparison.profile_comparator import HomologFinder`
- **class:** `HomologFinder` (var `finder`)
- **method:** `finder.find_homologs_multi(**method_params)` with
  `method_params = {"use_fast": True}` (UI default); `use_fast=False` runs the
  comprehensive search. Add `"morph_qualify": True` (requires
  `visualize_skeleton=True`) to score the visualized top-N against the
  transformed query with vector_v2 and gate them by a per-query null bar —
  works cross-dataset and intra-dataset alike on supported targets
  (FAFB / male-cns:v1.0 / BANC). Optional: `"morph_bar_offset"` (0.0),
  `"morph_mode"` (`"null"` default | `"mapping_ref"` — mapping_ref needs a
  cross-dataset pair, same-dataset runs fall back to the null bar),
  `"morph_level"` (95).

## Parameters the UI builds

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    source_dataset="male-cns:v0.9",
    target_dataset="hemibrain:v1.2.1",   # == source_dataset for intra-dataset search
    output_dir="/absolute/output/homologs",
    top_n=100,
    top_k=25,
    top_m=5,
    similarity_metric="jaccard",
    vector_prefiltering=True,
    include_untyped_partners=False,     # expand 2-hop (untyped) partners
    visualize_skeleton=False,
    visualize_top_n=0,
    visualization_settings={},          # skeleton visualization options (only if visualize_skeleton)
    min_synapse_threshold=3,
    use_cache=True,
    saveas="",
    use_auto_type_mapping=True,
    ensure_cache_complete=False,
)
# The UI passes `source` (a list) per run:
finder = HomologFinder(..., source=["aMe12"], saveas="")
results = finder.find_homologs_multi(use_fast=True)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Homologs_<date>.py
```

For multiple source queries, the UI loops `source` and gives each query its own
`saveas` suffix.

## Outputs

- bodyId/type homolog tables (CSV/XLSX) and summaries; intra-dataset runs add
  `results/intra_type_results.csv`.
- With morph qualification: morph columns (`morph_v2`, `morph_bar`,
  `morph_bar_kind`, `morph_qualified`, ...) on the visualized
  `results/bodyid_results.csv` rows, `results/morph_qualification.json`
  provenance, and below-bar candidates excluded from the rendered scenes.

## Notes

- `use_auto_type_mapping=True` resolves cross-dataset names through the shared
  validity-aware resolver (`comparison/type_resolver.py`): licensed renames map,
  valid splits expand, conflicts fail closed (never matched by raw same-name),
  unmapped names fall back to raw. Set it explicitly when names differ between
  datasets (irrelevant when Target = Source). The run writes
  `auto_type_mapping.json` with the mapper provenance and per-status counts.
- `visualize_skeleton=True` requires a valid `visualization_settings` dict.
- Start with `use_fast=True` (adjacency expansion); escalate to the slower
  comprehensive search only when it is insufficient.
- The former "Connectivity similarity" UI mode was the same engine with
  Target = Source; `min_shared_partners` / `vector_prune_fraction` remain
  available programmatically to widen the candidate pool.
