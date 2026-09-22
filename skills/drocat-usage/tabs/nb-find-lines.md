# Find Driver Lines (nb_find_lines / nb_find_lines_expanded)

Reproduce the **Find Driver Lines** UI tab (EM → LM). Uses `NeuronBridgeFinder`
to map EM neurons to GAL4/Split-GAL4 driver lines. No token required.

The tab is dataset-aware: the query box shares the Cross-Dataset tab's
auto-suggestion and history (gray hints like `type · male-cns:v1.0`, history
rows tagged with the datasets where the value actually resolves; the history
list is click-to-toggle on the focused empty field — click hides, click shows
again; a suggestion pick adds the chip, clears the typed text and keeps that
query's rows listed — in either list a name already added stays, tinted and
ticked, and clicking a ticked row takes its chip back out, the same as the chip's
own `x` (a history row's `×` is the other action: it prunes the history, not the
query) — so entries
can be handled one after another), the dataset
input is a **multi-select** with an exclusive `(all)` indicator, datasets
NeuronBridge does not host are **disabled** (advisory — runs still proceed with
a warning), and an **Expand query names across datasets** toggle (default on)
routes each query chip through the cross-dataset type mapper into every
NeuronBridge-hosted release before searching.

## Backend contract

- **tool_key:** `nb_find_lines` (plain) / `nb_find_lines_expanded` (expansion)
- **import:**
  `from neuronbridge_finder import NeuronBridgeFinder` /
  `from neuronbridge_query_expansion import ExpandedLineFinder`
- **class:** `NeuronBridgeFinder` (var `finder`) / `ExpandedLineFinder` (var `finder`)
- **method:** `finder.find_lines_batch(**method_params)` /
  `finder.run(**method_params)`

## Parameters the UI builds

```python
from neuronbridge_query_expansion import ExpandedLineFinder

finder = ExpandedLineFinder(
    verbose=True,
    separate_splitgal4=True,
    region="Brain",                     # filter by brain region
    max_workers=8,
)
result = finder.run(
    queries=["DNp01", "Tm4"],           # EM neuron queries (names or bodyIds)
    dataset=["male-cns:v1.0", "flywire_FAFB_v783"],   # None = search everywhere
    expand_names=True,                  # mapper expansion + coverage routing
    coverage_datasets=[                 # full selectable list, drives the
        "male-cns:v1.0", "flywire_FAFB_v783"],        # coverage snapshot
    output_dir="/absolute/output/nb_lines",
    match_type="cds",                   # "cds" | "pppm" | "both"
    keep_per_match_csv=True,            # False = Compact: drop {query}_lines.csv after summarization
    cleanup_source_images=False,        # True = Compact: drop images/ after the PDF/PPTX exists
    download_images=None,               # "neuronbridge" | "flylight" | "both" | None
    download_img_for_top_n_lines=None,
    summary_format=None,                # ["pdf"], ["pptx"], or None
    sort_by="max",
    image_formats=["png"],
    image_types=["cdm"],
    max_download_images_per_line=None,
    flylight_category=None,
    simple_mode=False,
    organize_by_region=False,
    pdf_images_per_page=(3, 2),
    summary_background_color="#ffffff",
)
```

`expand_names=False` (or omitting the `coverage_*` params) makes `run`
delegate to the plain `NeuronBridgeFinder.find_lines_batch` unchanged.

## How expansion + coverage routing works

1. A coverage snapshot (`src/neuronbridge_coverage.py`, persisted at
   `cache/neuronbridge/coverage_snapshot.json`) classifies every selectable
   dataset against NeuronBridge's hosted EM libraries by probing
   `metadata/by_body/{id}.json` for a few typed local bodyIds. Verdicts:
   `exact` (hosted at that version), `aligned` (hosted at another release,
   e.g. `male-cns:v1.0` → `v0.9`, `manc:v1.2.3` → `v1.2.1`), `unavailable`
   (BANC, optic-lobe), `unknown` (offline / no local table — never disabled).
2. Each name chip expands via the row-based type resolver
   (`type_resolver.resolve_valid_targets` → `expansion_targets`) into every
   covered release's namespace; splits expand to all branches, conflicts to
   nothing, unmapped keeps the raw name.
3. One `find_lines_batch` call per chip; every result row carries the
   original chip as `source_query`. As of v3_10_0 the hosted set is
   hemibrain:v1.2.1, male-cns:v0.9, manc:v1.2.1 and FlyWire FAFB v783.

## Outputs

Everything lands in ONE per-run folder (`NB-find-lines-expanded_*`) — flat
for single-chip runs, one `chip_{query}/` folder per chip for multi-chip
runs:

- `expansion_map.csv` — chip → expanded name → hosted release → mapping
  status/kind;
- `expansion_summary.json` — the original selection, the coverage routing
  (selected vs covered vs unavailable datasets), the output-detail mode and
  the cleanup audit;
- `user_warning_notes.txt` — advisory notes for unavailable datasets
  (`coverage:` prefix), the same convention as the other NB tools;
- per-chip Find Lines outputs: `{query}_lines.csv`, `line_summary.csv`,
  `gal4_lexa_summary.csv` / `split_gal4_summary.csv` (Split-GAL4
  separation), `parameters.json`, optional `images/` and PDF/PPTX
  summaries;
- `cleanup_audit.json` — Compact runs: every removed path and the
  reclaimed bytes.

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_NBLines_<date>.py
```

## Notes

- `download_images` drives image download: `"neuronbridge"`, `"flylight"`,
  `"both"`, or `None`. Start without downloads and a small query.
- Family-wide FAFB types (e.g. `Tm4`) resolve to hundreds of bodies and can
  produce very large per-query CSVs; query narrowly first.
- Disabled datasets are advisory only: a run that still targets one (saved
  state, `(all)`) warns via notification and `user_warning_notes.txt` and
  proceeds — the coverage snapshot must never lock out a dataset that works.
- The NeuronBridge match cache is OFF by default (`use_cache=False`;
  Settings → NeuronBridge Match Cache re-enables it): NB queries are large
  and rarely reused, so caching would grow without paying off. **Compact**
  output detail (keep_per_match_csv=False / cleanup_source_images=True)
  keeps the newest `compact_keep_last_n` expanded runs' bodyId-level match
  tables (default 1 — the latest query stays inspectable), prunes older
  Compact runs' tables, and removes images after the PDF/PPTX is written —
  under the default-off cache pruned tables regenerate only by re-running
  the query. Full/unknown-mode runs are never swept by the window.
