# neuronbridge_finder — NeuronBridgeFinder

Module `src/neuronbridge_finder.py`. A single dataclass `NeuronBridgeFinder`
carries out EM↔LM mapping: find GAL4/Split-GAL4 driver lines for EM neurons (EM →
LM), find candidate EM neurons for LM lines (LM → EM), and analyze line
co-labeling. **No token is required** for the NeuronBridge lookup.

## Constructor (dataclass defaults)

```python
from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(
    verbose=True,
    separate_splitgal4=False,
    region=None,                        # filter by brain region
    max_workers=4,
)
```

## Key methods

| Method | Direction | Purpose |
| --- | --- | --- |
| `find_lines_batch(queries=..., dataset=..., output_dir=..., match_type=..., download_images=None, download_img_for_top_n_lines=None, summary_format=None, sort_by=..., image_formats=..., image_types=..., max_download_images_per_line=None, flylight_category=None, simple_mode=False, organize_by_region=False, pdf_images_per_page=..., summary_background_color=...)` | EM → LM | Ranked driver lines for EM neuron queries. |
| `find_neurons_batch(line_names=..., output_dir=..., match_type=..., top_n=..., min_score=..., visualize_top_n=..., generate_individual_profiles=None, visualize_by=..., visualization_settings=..., sort_by=..., pdf_images_per_page=..., background_color=...)` | LM → EM | Ranked candidate EM neurons for line names. |
| `analyze_colabeling(lines=..., match_type=..., output_dir=..., similarity_methods=..., generate_report=..., visualize=..., visualize_top_n=..., top_n_neurons=..., min_score=..., min_type_avg_score=..., sort_by=..., background_color=..., pdf_images_per_page=..., datasets_to_visualize=..., visualize_by=..., visualization_settings=...)` | LM ↔ LM | Co-labeling overlap/similarity/specificity + heatmaps. |

```python
# EM → LM
finder.find_lines_batch(queries=["aMe12"], dataset="male-cns:v0.9",
                        output_dir="/abs/output/nb_lines", match_type="both")

# LM → EM
finder.find_neurons_batch(line_names=["SS00001"], output_dir="/abs/output/nb_neuron",
                          match_type="both", top_n=10)

# Co-labeling
finder.analyze_colabeling(lines=["SS00001", "SS00002"], output_dir="/abs/output/nb_colabel",
                          similarity_methods=["jaccard"], generate_report=True)
```

## Optional visualization methods

- `visualize_colabeling_matrix(...)`, `visualize_expression_matrix(...)`,
  `visualize_expression_matrix_merged(...)`, `visualize_labeling_distribution(...)`,
  `visualize_colabeling_distribution(...)` — standalone heatmaps/plots.

## Output-detail policy (`src/neuronbridge_output_policy.py`)

Shared prune pass behind the **Output detail: Full / Compact** control on
all three NeuronBridge tabs. `prune_find_lines_run` / `prune_find_neurons_run`
/ `prune_colabel_run(output_path, keep_per_match_csv, ...)` remove the
bodyId-level source-data tables after that run's summaries/report exist and
`images/` after the PDF/PPTX artifact exists (artifact-existence is the
success check — the generator swallows failures). Idempotent; audits every
removal in the run's `cleanup_audit.json`. A missing `output_path` (unset, or
a folder that does not exist) returns an empty audit and creates nothing: the
writer never mkdirs, because stringifying a `None` used to yield the relative
path `None` and leave that folder wherever the process ran. The finder methods
(`find_lines_batch`, `find_neurons_batch`, `analyze_colabeling`) accept
`keep_per_match_csv` / `cleanup_source_images` and call it before returning.

## Match cache default

`NeuronBridgeFinder.use_cache` defaults to **False** (Settings → NeuronBridge
Match Cache re-enables it per installation): NB queries are large and rarely
reused, so the per-body match tables would accumulate without paying off.
The flag gates both cache reads and writes.

## Coverage + expansion companion modules

### `src/neuronbridge_coverage.py`

Advisory dataset-coverage snapshots for the NeuronBridge tab. There is no
name/library index in the NeuronBridge bucket, so coverage is measured by
probing `metadata/by_body/{id}.json` for a small sample of typed bodyIds from
each dataset's LOCAL table:

```python
import neuronbridge_coverage as nbc

snapshot = nbc.refresh(["male-cns:v1.0", "banc_v626"])  # network, ~seconds
snapshot = nbc.load_snapshot()                # persisted, no network
snapshot.coverage_of("male-cns:v1.0").status  # exact | aligned | unavailable | unknown
snapshot.covered_datasets(["male-cns:v1.0"])  # {'male-cns:v1.0': 'male-cns:v0.9'}
nbc.warnings_for(["banc_v626"], snapshot)     # advisory warning strings
```

Contract: `unknown` never disables; `unavailable` only warns — a stale or
failed refresh must never lock out a dataset that works. Snapshot persisted
at `cache/neuronbridge/coverage_snapshot.json`, per-dataset 7-day TTL keyed
on NeuronBridge's `current.txt` version.

### `src/neuronbridge_query_expansion.py`

`ExpandedLineFinder` composes an untouched `NeuronBridgeFinder`: per query
chip, resolve cross-dataset equivalents with the row-based type resolver
(`resolve_valid_targets` → `expansion_targets`), route at the hosted release
(`male-cns:v1.0` → `v0.9` via the mapper's curated shared-name alias), and run
one `find_lines_batch` per chip. Writes `expansion_map.csv`,
`expansion_summary.json`, and `user_warning_notes.txt` (coverage notes
carry a `coverage:` prefix) into one
`NB-find-lines-expanded_*` run folder. Mapper-unavailable or coverage-unknown
degrades to the plain query path. See `skills/drocat-usage/tabs/nb-find-lines.md`.

## Notes

- `download_images` toggles image download (`"neuronbridge"`, `"flylight"`,
  `"both"`); `max_download_images_per_line` bounds it. Start without downloads.
- `simple_mode=True` speeds broad searches; `organize_by_region=True` groups the
  report by region.
- `match_type` is the NeuronBridge matching algorithm, one of `"cds"`, `"pppm"`, `"both"`.
- A 3D skeleton view (`visualize_top_n>0`) needs `visualization_settings`.
