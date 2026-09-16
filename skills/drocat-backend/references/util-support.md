# Supporting utilities

Backend helpers used throughout the modules and tabs. Open [`module-index.md`](module-index.md)
for the main classes; use these when a task needs data prep, token/readiness checks,
or local-file setup.

## token_manager (`src/utils/token_manager.py`)

```python
from src.utils.token_manager import get_access_token, get_cave_token
# never print the returned values
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

## Cache locations (no manager module)

There is no cache-manager helper to import; caches are plain paths under the
project root. `cache/` holds downloaded data and is safe to clear (the app
refetches); `neuron_indexes/` is a persistent "system files" directory never
cleared by `cache/`-cleanup.

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
