# Network (find_network)

Reproduce the **Network** UI tab as a direct backend call. `FindNetwork`
uses the queried group as **both source and target** (mutual connections), then
enriches it FindAllPath-style.

## Backend contract

- **tool_key:** `find_network`
- **import:** `from coana import FindNeuronConnection`
- **class:** `FindNeuronConnection` (var `fc`)
- **init:** `fc.InitializeNeuronInfo()`
- **method:** `fc.FindNetwork()`

## Parameters the UI builds

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12", "aMe10"],   # the queried set (used as source AND target)
    targetNeurons=["aMe12", "aMe10"],
    output_dir="/absolute/output/network",
    weight_basis="synapse",             # "synapse" | "connection_ratio"
    threshold_combination="or",         # "or" (one threshold governs) | "and" (both gates)
    min_synapse_num=3,                  # ratio 'or' mode: backend floors to 1
    min_ratio=0.0,                      # ratio mode: the threshold (e.g. 0.01 = 1%)
    min_traversal_probability=0.0,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    drop_untyped=True,                 # drop edges whose endpoint type is untyped (the UI default)
    edgeN_limit=500,
    output_format="csv",
    skip_bodyId=True,
    custom_source_name="",               # name for the queried group
    custom_target_name="",               # optional target-side label
    cache_only=False,
    saveas="",
    separate_hemispheres=False,
    hemisphere_filter="all",
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
)
# optional: fc constructor param custom_mapping_file for a custom grouping/mapping JSON

fc.InitializeNeuronInfo()
fc.FindNetwork()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_FindNetwork_<date>.py
```

## Outputs

- Enriched network (FindAllPath-style) tables and summaries; optionally an
  interactive Cytoscape HTML when `edgeN_limit` keeps it small.
- `<src>_to_<tgt>_allpaths_type.csv` — the 1-hop path-compatible table;
  the runner's post-run hook keys on it, so every network run also gets
  `path_report.html` + `paths_pair_breakdown/` (the Pair Explorer) exactly
  like the pathfinding tools.
- `source_neurons.csv` / `target_neurons.csv` — enrollment (bodyId, type,
  Checked) consumed by the type-level refill.
- `parameters.txt` carries the provenance stamps (`weight basis`,
  `requested_threshold` / `applied_threshold`) the pair report and the
  refill read.

## Notes

- `FindNetwork` initializes the queried neuron set before looking for connections;
  a zero-match query is a different failure than an empty result.
- Use `skip_bodyId=True` first, then enable bodyId-level work incrementally.
- Ratio basis (`weight_basis="connection_ratio"`): the threshold rides
  `min_ratio` and the backend floors `min_synapse_num` to 1 ('or' mode;
  `"and"` co-thresholds on both scales). A network is direct-connections-only
  (`max_interlayer` is recorded as 0).
- The type-level refill is ALWAYS produced for network runs: the
  full-scope records (every emitted chain re-valued at synapse ≥ 1) land
  in `data_details/type_level_refill/` + the `type_level_refill/` mirror.
  Ratio-basis runs additionally carry complete-at-asked records; a
  synapse-basis network never bites a budget, so it has no refill-asked
  mass.
