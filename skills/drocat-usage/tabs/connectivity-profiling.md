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

One dataset writes a level folder **only for the levels that ran**:

- `type_level/results/type_similarity_{metric}_{direction}.csv` +
  `visualization/heatmap_type_*.html` — pooled type profiles (`type`). The
  `custom` level writes the same matrix as
  `group_level/results/group_similarity_*.csv` + `heatmap_group_*.html`,
  because its axes are groups: the folder names what it holds.
- `bodyid_level/results/bodyid_similarity_{metric}_{direction}.csv` and
  `type_avg_bodyid_similarity_{metric}_{direction}.csv` +
  `visualization/heatmap_bodyid_*.html`, `heatmap_type_avg_*.html`.
- `profiles/individual/*_profile.json`, and `profiles/aggregated/` at the
  pooled levels.
- `report.html`, `parameters.json` (with `row_kind` and `levels_computed`),
  `README.txt` listing only the folders written.

At `aggregation_level="bodyid"` the compared rows ARE individual neurons, so
the main matrices land in `bodyid_level/` and there is no `type_level/` and no
`profiles/aggregated/` — nothing was pooled. The `type_avg_bodyid_*` matrix is
folded from those same pair scores (verified cell-for-cell against the
re-scored path), so a bodyId run still yields a per-type view.

Two or more datasets use the `intra_dataset/` + `cross_dataset/` layout.

## Notes

- **The population gate counts neurons, not rows.** One type with two members
  is a legitimate run; a query resolving to a single neuron raises
  `ValueError` (it does not return an error dict — the runner discards return
  values, so a silent refusal would show as Completed with no files).
- `skip_bodyId_level` cannot empty a one-row run while it stays cheap: that
  row's pooled cell is itself pooled against itself (1.0 on every metric), so
  the bodyId pass is kept despite the setting — unless the population passes the
  same 1000-bodyId budget the `auto` rule enforces, since the pair loop is
  quadratic. Past it the skip stands and the log says what to change.
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
