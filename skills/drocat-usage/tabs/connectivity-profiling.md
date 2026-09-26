# Connectivity · Comparison (connectivity_profiling)

Reproduce the **Connectivity tab → Comparison sub-tab** as a direct backend
call. Builds connectivity profiles for a query set and compares them across
one or more datasets.

## Backend contract

- **tool_key:** `connectivity_profiling`
- **import:** `from comparison.profile_comparator import ConnectivityProfileComparer`
- **class:** `ConnectivityProfileComparer` (var `comparer`)
- **method:** `comparer.run()`

## Parameters the UI builds

```python
from comparison.profile_comparator import ConnectivityProfileComparer

comparer = ConnectivityProfileComparer(
    query=["aMe12", "aMe10"],           # ignored at aggregation_level="custom",
                                       # where custom_mapping_file supplies the rows
    datasets=["male-cns:v0.9", "hemibrain:v1.2.1"],
    output_dir="/absolute/output/profiles",
    top_k=15,
    top_m=5,
    min_synapse_threshold=3,
    direction="both",                   # input, output, or both
    generate_heatmaps=True,
    show_figures=False,
    verbose=True,
    use_cache=True,
    aggregation_level="type",           # "type" | "bodyid" | "custom"
    skip_bodyId_level=False,
    ensure_cache_complete=False,
)
# for aggregation_level="custom", add: comparer = ConnectivityProfileComparer(..., custom_mapping_file="/path/to/mapping.json")

results = comparer.run()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Profile_<date>.py
```

## Outputs

- Similarity matrices, heatmaps (when `generate_heatmaps=True`), and profile
  comparison tables under the output folder.

## Notes

- `aggregation_level="custom"` requires `custom_mapping_file`.
- `ensure_cache_complete=True` forces a full cache; use it deliberately.
- The profiling resolver resolves the query before comparing; a completed run
  means the queried chips resolved in the dataset.
- **Auto type mapping is ON by default** (`use_auto_type_mapping=True`):
  with 2+ datasets, query names resolve per dataset through the shared
  validity-aware resolver (`comparison/type_resolver.py`). A unique rename
  (MeVPLo2 ↔ MTe07) maps automatically; a valid split (MCNS `VS` → FAFB
  `VS1`…`VS8`) expands to every branch; a conflicted type is dropped for
  the dataset it conflicts toward (fail closed, recorded in the run's
  mapping-resolution metadata); unmapped names are used as-is and counted
  as raw fallback. Pass `use_auto_type_mapping=False` to disable and
  compare raw names.
- Explicit dataset-keyed query dicts (`{'dataset_a': [...], ...}`) keep
  their dataset-local literal semantics — they are not second-guessed by
  the mapper.
- Every saved run's `parameters.json` carries an `auto_type_mapping_*`
  metadata block (requested vs active mapper, v1.0 source table, version,
  load error, per-status resolution counts on the `unique_type_resolutions`
  basis, the separate occurrence-basis partner metric, and the raw-fallback
  flag).
