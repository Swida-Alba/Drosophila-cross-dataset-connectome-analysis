# Morphology · Find Similar (find_similar_morphology)

Reproduce the **Morphology tab → Find Similar sub-tab** as a direct backend
call: a query-vs-all morphological similarity search **within one dataset**
(intra-dataset only — BANC is excluded because morphological similarity on
its public L2/full skeletons is deferred pending vector-quality validation;
BANC 3D visualization works). For connectivity-based similar search
(including cross-dataset homolog search), see [find-homologs.md](find-homologs.md).

## Backend contract

- **tool_key:** `find_similar_morphology`
- **import:** `from morphology import MorphologyComparer`
- **class:** `MorphologyComparer` (var `comparer`)
- **method:** `comparer.find_similar()`

## Parameters the UI builds

```python
from morphology import MorphologyComparer

comparer = MorphologyComparer(
    dataset="male-cns:v0.9",
    level="auto",                        # query level (UI default)
    method="vector_v2",                  # "vector_v2" (default) | "nblast"
    candidate_cap=500,
    candidate_source="auto",             # "auto" | "profile" | "combined" | "cache" | "roi"
    roi_filter=None,                     # or ["EB", "LH", "AL"]
    visualize_top_n=10,                  # UI default; 0 disables rendering
    visualize_by="bodyId",
    visualization_settings={},           # skeleton settings when visualizing
    output_dir="/absolute/output/similar_morph",
    saveas="",
    verbose=True,
    n_workers=8,
    use_cache=True,
    cache_fetched_skeletons=True,
)
# The UI passes `query` per query (loop for multiple queries):
comparer = MorphologyComparer(..., query="aMe12")
results = comparer.find_similar()
```

## Run

```bash
# morphology (BANC excluded; one dataset only)
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_SimilarMorph_<date>.py
```

## Outputs

- BodyId-level `results.csv` + type-level `type_summary.csv`; optional
  query-plus-top-N 3D skeleton HTML.

## Notes

- Multiple queries loop the `query` field; each query gets its own run folder.
- `candidate_source` options are `["auto", "roi", "combined", "profile", "cache"]`;
  `"profile"`/`"combined"`/`"cache"` read the connection cache, `"roi"` uses ROI
  screening. Ensure cache coverage first or set `cache_fetched_skeletons` accordingly.
- There is no cross-dataset morphology comparison: scores are computed in one
  dataset's coordinate space against that dataset's caches.
