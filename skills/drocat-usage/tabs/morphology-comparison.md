# Morphology · Comparison (morphology_comparison)

Reproduce the **Morphology tab → Comparison sub-tab** as a direct backend
call: an intra-dataset N×N morphology comparison of the queried neurons
(types, bodyIds, or patterns). `aggregation_level` picks what a matrix row
is — a type, one neuron, or a custom group. The bodyId × bodyId matrix is
always the scored primitive; the aggregate over types (or groups) is derived
from it and skipped when the comparison is already at bodyId level.

## Backend contract

- **tool_key:** `morphology_comparison`
- **import:** `from morphology_comparison import MorphologyProfileComparer`
- **class:** `MorphologyProfileComparer` (var `comparer`)
- **method:** `comparer.run()`

## Parameters the UI builds

```python
from morphology_comparison import MorphologyProfileComparer

comparer = MorphologyProfileComparer(
    dataset="male-cns:v1.0",             # ONE dataset (intra-dataset only; no BANC)
    query=["aMe12", "aMe10", "aMe.*"],   # types, bodyIds, or regex patterns
    method="vector_v2",                  # "vector_v2" (default) | "nblast"
    aggregation_level="type",            # "type" | "bodyid" | "custom group"
    custom_mapping_file=None,            # LabelMapper preset; forces "custom"
    max_members_per_type=25,             # members kept per type / per group
    max_total_neurons=200,               # safety cap (truncates; NBLAST refuses > 30)
    fetch_online=True,                   # pull missing skeletons via API (NeuPrint SWC / FAFB bundle→CAVE)
    output_dir="/absolute/output/morph_cmp",
    saveas="",
    generate_heatmaps=True,
    show_figures=False,
    verbose=True,
    n_workers=8,
    use_cache=True,
)
results = comparer.run()
```

`results` carries `aggregation_level`, `rows_compared` and
`neurons_compared` beside `output_folder` / `files`.

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_MorphCompare_<date>.py
```

## Outputs

- `type_level/type_similarity_{method}.csv` — type×type matrix, diagonal =
  intra-type cohesion. Written at the **type level only**.
- `group_level/group_similarity_{method}.csv` — the same aggregate at the
  **custom group level**, with group labels as axes (named for its axes, not
  filed under `type_level/`).
- `bodyid_level/bodyid_similarity_{method}.csv` — every individual pair;
  always written, and the run's only matrix at the bodyId level.
- `members.csv` — resolved population (`row` / `type` / `bodyId` / `instance`
  / status), status being `compared` / `no vector` / `no dotprops`.
- `visualization/heatmap_*.html`, `report.html`, `parameters.json`,
  `README.txt`.

## Notes

- **Row semantics follow `aggregation_level`.** `type`: each queried type is
  one row, and a bodyId query resolves to its type. `bodyid`: every
  individual neuron is its own row, so two neurons of the same type can be
  compared — at the type level they fold into one row and the run refuses.
  `custom`: rows are the source-side groups of `custom_mapping_file` and the
  query is ignored. Patterns always expand (to types, or to their neurons).
- **A bodyId named in the query is never dropped** by `max_members_per_type`
  or `max_total_neurons`; those caps only ever remove other members. The
  `row` column tells you which matrix row a neuron fed; `type` stays its real
  type.
- The UI hides the bodyId level while NBLAST is selected (see the cap below),
  so the combination cannot be requested from the tab; a direct backend call
  can still ask for it and gets the refusal.
- **Intra-dataset only**: scores live in one dataset's coordinate space
  against that dataset's caches. Cross-dataset comparison belongs to the
  connectivity side (`HomologFinder` / `ConnectivityProfileComparer`).
- `method="nblast"` scores every neuron pair, so it refuses any population
  over 30 **neurons** — the capped population counts, which is why
  `max_total_neurons=20` over a 50-neuron query runs instead of refusing.
  It needs local raw skeletons (online fetch only happens through
  `MorphologyComparer`'s shared dotprops pipeline) and excludes contralateral
  pairs from aggregate means.
- `vector_v2` uses the per-dataset `SkeletonVectorCacheV2`; missing members
  are fetched online by default (`fetch_online=True`) through the same
  skeleton pipeline as Find Similar (NeuPrint raw SWC; FAFB healed zip
  → CAVE/local-fix fallback) and persist into the shared cache. Set
  `fetch_online=False` for a strictly offline comparison.
- Patterns (`aMe.*`) expand against the dataset's type names; an exact type
  name always wins over pattern interpretation (some type names contain
  metacharacters, e.g. `PPL1*`).
