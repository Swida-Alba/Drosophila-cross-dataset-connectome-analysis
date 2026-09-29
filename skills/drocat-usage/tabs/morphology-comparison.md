# Morphology · Comparison (morphology_comparison)

Reproduce the **Morphology tab → Comparison sub-tab** as a direct backend
call: an intra-dataset N×N morphology comparison of the queried neurons
(types, taxonomy labels, bodyIds, instance names, or patterns).
`aggregation_level` picks what a matrix row
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
    dataset="male-cns:v1.0",             # ONE dataset (intra-dataset; BANC runs with a warning)
    query=["aMe12", "aMe10", "aMe.*"],   # types, taxonomy labels, bodyIds, instance names, or patterns
    method="vector_v2",                  # "vector_v2" (default) | "nblast"
    aggregation_level="type",            # "type" | "bodyid" | "custom group"
    custom_mapping_file=None,            # LabelMapper preset; forces "custom"
    max_members_per_type=25,             # members kept per type / per group
    max_total_neurons=200,               # safety cap (truncates; NBLAST warns past 30)
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
- `user_warning_notes.txt` — written only when something needs
  disclosing: taxonomy-label expansions, instance-name matches, member /
  total caps, NBLAST cost warnings, tokens nothing matched, and the BANC
  provisional-scores caveat. Rendered by the run guide.
- `visualization/heatmap_*.html`, `report.html`, `parameters.json`,
  `README.txt`. Heatmap pages carry interactive dendrograms, and the
  report's heatmap cards embed mini-dendrograms of the same Ward order;
  every heatmap's cells render square (1:1) at any window size.

## Notes

- **Row semantics follow `aggregation_level`.** `type`: each queried type is
  one row, and a bodyId query resolves to its type. `bodyid`: every
  individual neuron is its own row, so two neurons of the same type can be
  compared as themselves — the thing the type level folds away. `custom`:
  rows are the source-side groups of `custom_mapping_file` and the query is
  ignored. Patterns always expand (to types, or to their neurons).
- **The gate is two NEURONS in scope, not two rows.** One type runs at every
  level: at the type level its single aggregate cell is that type's cohesion
  (mean pairwise score, not 1.0) and `bodyid_level/` carries the pairwise
  detail; at the bodyId level its neurons are the rows. A query resolving to
  one neuron raises, and the message names what collapsed it.
- **A bodyId named in the query is never dropped** by `max_members_per_type`
  or `max_total_neurons`; those caps only ever remove other members. The
  `row` column tells you which matrix row a neuron fed; `type` stays its real
  type.
- The UI hides the bodyId level while NBLAST is selected (each row would
  be one neuron, so the per-type cap cannot bound the population), so the
  combination cannot be requested from the tab; a direct backend call can
  still ask for it and runs.
- **Intra-dataset only**: scores live in one dataset's coordinate space
  against that dataset's caches. Cross-dataset comparison belongs to the
  connectivity side (`HomologFinder` / `ConnectivityProfileComparer`).
- `method="nblast"` scores every neuron pair twice, so populations past
  30 scored **neurons** warn that the run may take very long — they are
  never refused (a `max_total_neurons=20` cap over a 50-neuron query stays
  under the bound and stays silent). It needs local raw skeletons (FAFB
  resolves through the healed zip / CAVE pipeline; BANC through the public
  SWC chain) and excludes contralateral pairs from aggregate means.
  Both similarity metrics span [-1, 1] (negative = below chance), so all
  heatmaps — report cards, VisPath pages, and the plotly fallback —
  render on the diverging scale over the full [-1, 1] domain, every run.
- `vector_v2` uses the per-dataset `SkeletonVectorCacheV2`; missing members
  are fetched online by default (`fetch_online=True`) through the same
  skeleton pipeline as Find Similar (NeuPrint raw SWC; FAFB healed zip
  → CAVE/local-fix fallback) and persist into the shared cache. Set
  `fetch_online=False` for a strictly offline comparison.
- Query tokens resolve in order: exact type name → bodyId → taxonomy label
  (a cell_type / class value such as FAFB's `circadian_clock` expands into
  one row per member type) → instance name (those neurons are pinned, like
  bodyId queries) → regex pattern. Patterns (`aMe.*`) expand against the
  dataset's type names; an exact type name always wins over pattern
  interpretation (some type names contain metacharacters, e.g. `PPL1*`).
  Every expansion, pin, cap, and unmatched token is disclosed in the run
  log and `user_warning_notes.txt`.
- **BANC runs with an explicit caveat**: the public L2/full skeleton
  products' vector quality is still unvalidated, so treat BANC scores as
  provisional (the caveat is recorded in `user_warning_notes.txt`).
  Cross-dataset BANC comparisons carry their own experimental banner.
