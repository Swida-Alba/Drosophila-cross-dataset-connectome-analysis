# Type Validation (TM VEV) (type_mapping_validation)

Reproduce the **Type Validation** tab (UI, in the **Cross-Dataset** group) as a
direct backend call. It runs the TM VEV pipeline — `MappingValidator` over a
`MappingValidationConfig` — to validate a source→target type mapping at bodyId
granularity, expand suspected candidates, and render 3D review scenes. Output is
proposals only; the mapping is never rewritten.

## Backend contract

- **tool_key:** `type_mapping_validation`
- **import:** `from comparison.mapping_validation import MappingValidationConfig, MappingValidator`
- **wrap:** build `MappingValidationConfig(**field_kwargs)`, then
  `MappingValidator(cfg).run() -> Path` (the returned run folder). The UI
  generator additionally installs a `log` wrapper that announces the run folder
  and emits `[DROCAT][progress]` steps; a direct script call does not need it.
- **class:** `MappingValidator` (var unused; `run()` returns the folder)
- **CLI:** `scripts/RunMappingValidation.py` (sets 51 of the 61 fields; see
  the flag↔field table in the user guide §1b).

## The payload speaks FIELD names, not CLI flags

The dataclass is a plain `@dataclass`: an unknown kwarg raises `TypeError`.
The tab always sends the field names below (never the flag names, which differ
and several are negated). Required: `source_dataset`, `target_dataset`;
`query_types` defaults to `[]` so the UI enforces non-empty itself.

```python
from comparison.mapping_validation import (
    MappingValidationConfig, MappingValidator)

cfg = MappingValidationConfig(
    source_dataset="flywire_FAFB_v783",
    target_dataset="male-cns:v1.0",
    query_types=["T5c"],                 # a type or a coarse cell_type value
    validation_mode="restrictive",       # restrictive | family | aggressive,
                                         # or "pooling" — a PARALLEL unsupervised
                                         # scan of the whole target universe (not
                                         # a wider rung; writes pooling/ beside
                                         # the nested bins; its gate card appears
                                         # only when the mode is pooling)
    aggressive_expansion=False,          # legacy alias; mode carries the enum
    pool_widen=False,                    # retired (Rev 3.12)
    morph_enabled=True,                  # CLI --no-morphology negates
    visualize=True,                      # CLI --no-visualize negates
    backward_evidence_enabled=False,     # CLI --backward-evidence (default off)
    skip_out_map_expansion=False,
    skip_profile_build=False,            # True => cache-only profiles (offline control)
    use_cache=True,                      # False => force refetch (NOT offline)
    output_dir="/abs/output",            # always pass explicitly (see folder note)
    run_label="T5c-familial",            # stays in parameters.json, not the folder name
    verbose=True,
    # fields with no CLI flag (UI/Settings only): top_k, top_m,
    # min_synapse_threshold, include_untyped_partners, null_* , candidate_morph_cap
)
run_dir = MappingValidator(cfg).run()
print(run_dir)
```

## Flags / semantics worth knowing

- **Offline:** there is no `cache_only` field on this pipeline. The offline
  control is `skip_profile_build=True`, plus `morph_enabled=False` and
  `visualize=False` (stage-2 profile pre-flight, stage-5 morph vectors, and
  stage-4 skeleton loads are what reach NeuPrint/CAVE).
- **Coarse categories:** a query chip may be a `cell_type` value
  (e.g. `circadian_clock`); the run resolves it via the prioritized column
  search (`search_columns='auto'`), then validates per concrete type. Coarse
  queries fan out to large pools and are slow.
- **No pair guard:** an unsupported dataset pair is not an error — it resolves
  zero branches and writes an essentially empty run folder. Treat "no type
  pairs resolved" as a distinct outcome, not a silent success.
- **Scene cap:** `max_scenes` defaults to **0 = one scene per parent type**, so a
  coarse category query renders (and costs) every scene it resolves. A positive
  value caps the count and names the parents it dropped; prefer splitting a
  large query across runs over raising a cap that is already off.
- **Run folder:** always `output_dir/type-map-validation_{SRCNICK}_to_{TGTNICK}_{ts}`.
  If `output_dir` is omitted it lands in `local_data/mapping_validation/` (a
  grandchild), which weakens the runner's direct-child discovery — the UI always
  passes `output_dir` explicitly.

## Outputs

`report.html`, `README.txt`, `parameters.json`, `set_coverage.json`,
`pipeline_progress.jsonl`, plus `validation/`, `expansion/`, `gap_fill/`,
`mapping/` CSVs and `visualization/plot-3d_*/branches_*.html` scenes. See
[the user guide](../../../docs/core-features/TypeMappingValidateExpandVisualize_Guide.md)
and [OUTPUT_FILES §9](../../../docs/OUTPUT_FILES.md).
