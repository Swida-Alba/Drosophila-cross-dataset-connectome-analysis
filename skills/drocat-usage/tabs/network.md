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
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    drop_untyped=True,                 # drop edges whose endpoint type is untyped (the UI default)
    edgeN_limit=500,
    output_format="csv",
    skip_bodyId=True,
    custom_source_name="",               # name for the queried group
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

## Notes

- `FindNetwork` initializes the queried neuron set before looking for connections;
  a zero-match query is a different failure than an empty result.
- Use `skip_bodyId=True` first, then enable bodyId-level work incrementally.
