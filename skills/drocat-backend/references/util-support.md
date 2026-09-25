# Supporting utilities

Backend helpers used throughout the modules and tabs. Open [`module-index.md`](module-index.md)
for the main classes; use these when a task needs data prep, token/readiness checks,
or local-file setup.

## token_manager (`src/utils/token_manager.py`)

```python
from utils.token_manager import token_manager  # never print the values
token_manager.get_neuprint_token()    # NeuPrint
token_manager.get_token("cave")       # CAVE (FlyWire)
token_manager.get_auto_token(prefer_type="cave")  # config/env resolution
```

- Precedence (tokens live in `config.json` — wins per key; the gitignored
  `config_local.json` fills entries left empty). Never read `token_info*.txt`
  (removed). `config_local.json` is never auto-created.
- `NEUPRINT_TOKEN` from <https://neuprint.janelia.org/account>;
  `CAVE_TOKEN` (FlyWire only) from <https://codex.flywire.ai/auth_token>.

## neuron_filter (`src/utils/neuron_filter.py`)

```python
from src.utils.neuron_filter import ...   # apply filter-by mode (bodyId / type) and regex
```

Used by the pathfinding tabs to convert raw chips into resolved queries
(`apply_filter_mode`). Check the module for the exact helpers.

## flywire_readiness (`src/utils/flywire_readiness.py`)

```python
from src.utils.flywire_readiness import (
    flywire_skeleton_readiness, require_flywire_skeleton_access,
    flywire_manual_skeleton_instruction, print_download_instructions,
)
from src.flywire_ids import is_fafb_dataset, is_banc_dataset
```

Detects whether local FAFB files or standalone BANC release files exist and whether the skeleton source is
available. FAFB needs converted local files; BANC uses its public release bucket and local prepared tables; a missing local file and
a missing token are different failures.

## Cache locations + storage utility (`src/storage_inventory.py`)

`cache/` holds downloaded data and is safe to clear (the app refetches);
`neuron_indexes/` is a persistent "system files" directory never cleared by
`cache/`-cleanup. Programmatic scan/deletion goes through
`src/storage_inventory.py` (`scan_caches`, `scan_run_folders`,
`delete_paths`) — the same engine behind the Settings → Storage card.
`delete_paths` refuses protected paths (`cache/user_mappings/` LabelMapper
presets, availability/coverage snapshots, per-dataset
`available_rois.json`, the storage audit) and anything it cannot
classify; the incoming-connections class always removes
`incoming_connections.parquet` together with its
`incoming_complete.json` completion state. Run folders are identified by
a tool prefix plus the embedded `_YYYYMMDD_HHMMSS` timestamp
(`utils.naming_utils.is_run_folder_name`).

## roi_screening (`src/roi_screening.py`)

```python
from src.roi_screening import ...        # ROI-sensitive screening helpers
```

Backing for ROI-filtered candidate screening in morphology / profiling.

## Connection/profile caches (`src/build_connection_cache.py`, `src/build_connectivity_profile_cache.py`)

```bash
python src/build_connection_cache.py <dataset>
python src/build_connectivity_profile_cache.py <dataset>
```

Build the connection cache (used by the `candidate_source` `"profile"`/`"combined"`/`"cache"`
modes and by `use_cache`) and the connectivity-profile cache (used by
profiling/similar tools).

## FAFB/BANC release conversion (`src/FAFB_file_converter.py`, `src/BANC_file_converter.py`)

Converts raw Codex downloads under `datasets/<dataset>/downloads/` to the
`<dataset>_allneurons_neuron_df.parquet` + `<dataset>_merged_connections.parquet`
that the analysis modules require. Validation is dataset-specific — see the
datasets-and-auth reference.
