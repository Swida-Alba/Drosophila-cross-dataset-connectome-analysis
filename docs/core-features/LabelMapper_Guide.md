# LabelMapper Guide

The `LabelMapper` class provides a centralized mechanism for standardizing neuron identifiers across different datasets. This is crucial for cross-dataset comparison analysis where naming conventions may vary (e.g., `aMe12` in hemibrain vs `aMe12-like` in male-cns).

## Key Features

*   **Centralized Mapping**: Manages mapping logic across all datasets in one place.
*   **Flexible Input Formats**: Supports both CSV and JSON formats.
*   **Asymmetric Mapping**: Handles cases where one dataset has more neurons for a given type than another.
*   **Group-Based Mapping**: Maps multiple neurons from different datasets to a single "standard label" (group).
*   **Auto-Generation**: Can automatically generate standard labels if not provided.

## Input Formats and Example Files

The toolkit provides several example files in `examples/data/` to demonstrate how to structure your mapping data.

### 1. CSV Format
The CSV format is the most straightforward way to define mappings. It requires a `custom_label` column and one column per dataset.

**Example (`examples/data/example_source_mapping.csv`):**
```csv
custom_label,hemibrain:v1.2.1,male-cns:v0.9,flywire_FAFB_v783
aMe12,aMe12,aMe12,720575940610453042
MBON14,MBON14.*_R,MBON14.*_R,MBON14
MBON06,MBON06.*_R,MBON06.*_R,MBON06
PPL101,PPL101,PPL101,720575940621886666
```
*   **custom_label**: The standardized name used in your analysis.
*   **Dataset Columns**: The specific neuron IDs or regex patterns for each dataset.
*   **Regex Support**: You can use regex patterns (e.g., `MBON14.*_R`) to match multiple neurons.

### 2. JSON Format
The JSON format allows for defining source, target, and intermediate mappings in a single file and supports nested structures.

A preset file must declare which grammar it uses through the top-level `format`
key (`drocat_custom_groups/v1`, the value the grouping board exports). The
backend refuses a preset without it, or with a grammar it does not read, rather
than accepting it on shape alone — and a comparison run does not degrade that
refusal to raw names, it stops. In-memory `source_mapping_dict` payloads and CSV
role files make no such claim and are not gated.

**Example (`examples/data/example_label_mapping.json`):**
```json
{
  "format": "drocat_custom_groups/v1",
  "source_mapping": {
    "custom_label": ["aMe12"],
    "hemibrain:v1.2.1": [["aMe12"]],
    "male-cns:v0.9": [["aMe12"]],
    "flywire_FAFB_v783": [["720575940610453042"]]
  },
  "target_mapping": {
    "custom_label": ["KCg-s1"],
    "hemibrain:v1.2.1": [["KCg-s1"]],
    "male-cns:v0.9": [["KCg-s1"]],
    "flywire_FAFB_v783": [["KCg-s1"]]
  },
  "intermediate_mapping": {
    "custom_label": ["LPLC2"],
    "hemibrain:v1.2.1": [["LPLC2"]],
    "male-cns:v0.9": [["LPLC2"]],
    "flywire_FAFB_v783": [["LPLC2"]]
  }
}
```

## Usage Examples

### Initialization from Files

```python
from comparison.label_mapper import LabelMapper

# Using a single unified JSON file (Recommended)
# Note: ONLY JSON format is supported for overall_mapping_json
mapper = LabelMapper(
    overall_mapping_json='examples/data/example_label_mapping.json'
)

# Using separate CSV files for source and target
mapper_csv = LabelMapper(
    source_mapping_file='examples/data/example_source_mapping.csv',
    target_mapping_file='examples/data/example_target_mapping.csv'
)
```

### Initialization from Dictionaries

```python
from comparison.label_mapper import LabelMapper

mapper = LabelMapper(
    source_mapping_dict={
        'hemibrain:v1.2.1': [['aMe12', 'aMe12_R']],
        'male-cns:v0.9': [['aMe12', 'aMe12-like']]
    },
    source_labels=['aMe12_grp']
)
```

### Retrieving Standard Labels

```python
# Get the standard label for a neuron in a specific dataset
std_label = mapper.get_std_label(
    dataset='hemibrain:v1.2.1', 
    neuron_id='aMe12', 
    group_type='source'
)
print(std_label) # Output: aMe12_grp1
```

### Retrieving Neurons by Standard Label

```python
# Get all neurons associated with a standard label in a specific dataset
neurons = mapper.get_neurons_by_label(
    std_label='aMe12_grp1', 
    dataset='male-cns:v0.9', 
    group_type='source'
)
print(neurons) # Output: ['aMe12', 'aMe12-like', 'aMe12_variant']
```

## Integration with ComparisonParameters

`LabelMapper` is the core component for defining neuron identity in `ComparisonParameters`. It allows you to abstract away dataset-specific naming differences.

### Using Example Files in Comparison

Here is how you would set up a comparison using the provided example files:

```python
from comparison.comparison_parameters import ComparisonParameters
from comparison.label_mapper import LabelMapper

# 1. Initialize the mapper with your mapping files
mapper = LabelMapper(
    source_mapping_file='examples/data/example_source_mapping.csv',
    target_mapping_file='examples/data/example_target_mapping.csv'
)

# 2. Create ComparisonParameters
# The mapper handles the translation of 'aMe12' and 'KCg-s1' to dataset-specific IDs
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    source_neurons=mapper,  # Pass the mapper object here
    target_neurons=mapper,  # Pass the mapper object here (or a list if no mapping needed)
    
    # Optional: Manually specify which standard labels to use from the mapper
    # If omitted, it might use all available in the mapper or require manual specification depending on implementation
    # source_labels=['aMe12'], 
    # target_labels=['KCg-s1'] 
)

# 3. Run the comparison
# The analyzer will use the mapper to look up 'aMe12' in hemibrain, male-cns, etc.
```

### Why use LabelMapper in ComparisonParameters?

*   **Consistency**: Ensures you are comparing "apples to apples" across datasets.
*   **Simplicity**: Your analysis code only deals with standard labels (e.g., `aMe12`), not dataset-specific IDs (e.g., `720575940610453042`).
*   **Scalability**: Easily add new datasets by just updating the CSV mapping file, without changing your analysis code.


## UI Workflow (v4.5.0)

The web UI builds custom mappings **inline on each tab** — no scripting needed:

1. **Open the tab's mapping board.** The **Cross-Dataset Comparison** page
   mounts it as **Custom Mapping**; **Complete Paths**, **Network** and
   **Shortest Paths** mount it as **Custom Grouping**, and **Connectivity's
   Comparison sub-tab** as **Custom Grouping Preset** (the Settings tab's
   **Custom Type Mappings** card offers the same editor). The shared board has
   one row per custom group, one aligned member input per selected dataset,
   outlined chips, a history menu of previously used groups, and an
   available-neuron viewer.
2. **Define your groups and run.** At run time the board is validated,
   exported to the canonical LabelMapper JSON
   (`cache/user_mappings/_inline/<tab>_<timestamp>.json`), recorded in the
   group history, and passed to the run — as `overall_mapping_json` on the
   Cross-Dataset page, where it acts as a mapping **overlay**: your explicit
   source/target neuron queries stay, and only neurons matching the mapping
   are renamed to their custom groups (unmapped neurons keep their
   identifiers) — or as `custom_mapping_file` on the path/network tabs, where
   mapped neurons are grouped at type level via `EnrichConnectionTable`
   (custom_group columns).

   **Residual neurons of collided native types** (2026-10-08): when a
   native type's population is only PARTIALLY governed — some of its
   neurons are mapper members and others are not — or the type name is
   itself one of the mapper's labels, the type resolver star-tags the
   unenrolled neurons: they are renamed to `T*` (the native type with a
   trailing `*`) so they can never silently merge into the governed
   population or into a settled label of the same spelling. `T*` keys are
   terminal (dataset-scoped, never resolved further) and disclosed in
   `user_warning_notes.txt`. Cross-dataset name coincidences are
   dataset-scoped and unaffected (a FAFB intermediate natively typed
   `SMP540` keeps its raw name even when BANC's enrolled `SMP540` is
   retyped).
3. **Reuse a mapping later** via the board's **Load…** (parses any mapping
   JSON back onto the board) or the history menu.
4. **Optional — save a named copy in Settings.** The Settings tab's
   **Custom Type Mappings** card adds an outlined **Save Mapping** action:
   it saves the stable `Custom Mapping` preset to
   `cache/user_mappings.json` (gitignored) and exports its JSON. Saving does
   **not** attach the preset to future runs — no tab auto-loads saved
   presets; bring the JSON back onto a board with **Load…** when you want it.

The `Custom Mapping` preset can be overwritten by saving an updated board;
runs always consume the exported file, so results are reproducible
(`label_map.json` / `parameters.json` record the mapping used).

## Display Role: Matrix Axes and Label Views (2026-10-11)

Beyond defining comparison rows, a loaded preset acts as the **tier-0
identity overlay** on every comparison surface — a label IS the identity
and is never re-resolved through the auto type mapper:

- **BodyId axes** (Connectivity ▸ Comparison, Morphology ▸ Comparison,
  Cross-Dataset Morphology): a preset-resolved bodyId reads
  `{bodyId}_{label}({native})` — e.g. `13531_5th-LNv(5thsLNv_LNd6_L)` —
  with the legacy suffix, side marker included, inside the parentheses.
  **Dedup rule**: when the label already IS the native name the
  parenthetical is dropped (`{bodyId}_{label}`), and when only the side
  marker differs the side is kept (`{bodyId}_{label}_{L|R}`, e.g.
  `11999_l-LNv_L`). The bodyId stays the token before the first `_` (axis
  parsers key on the head); unmapped members keep the legacy form
  byte-identically, so a no-preset run is unchanged.
- **Cross-Dataset Morphology** writes `morph_labeltype_matrix.csv` beside
  the native `morph_type_matrix.csv` (which stays native-keyed): one row
  per standard label, multi-label native types expanded, unmapped members
  keeping their native name. `overview.csv` gains a `label` column
  ('/'-joined for double-claim types).
- **Morphology Comparison** `members.csv` gains a `label` column beside
  the native `type` — per-bodyId resolution, because a native type can
  settle to two labels while a bodyId resolves to exactly one.
- **Connectivity ▸ Comparison** additionally accepts the preset as the
  ROW definition (`custom_mapping_file`, forces the custom level) or as
  the rows-unchanged tier-0 overlay (`label_mapping_file`), and can
  collapse the partner/feature space through the groups
  (`group_partners_by_mapping`). Report headers disclose the state
  (`User labels N (tier 0)` / `Partner grouping ON`).

Shared plumbing: `utils.label_utils.build_body_id_label(..., label_of=)`
formats the axes; `comparison.label_mapper.axis_label_resolver(mapper)` /
`load_label_mapper(file, log)` resolve a bodyId's standard label (the
mapper's identity fallback is deliberately NOT a resolution — only
bodyIds that genuinely carry a label upgrade).
