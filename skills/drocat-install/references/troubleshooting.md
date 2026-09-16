# DROCAT Install — Troubleshooting

Use this after the one-click installer fails or produces an inconsistent
environment. Check items in order; a single root cause usually explains several
symptoms.

## Environment / version conflicts

| Symptom | Root cause | Resolution |
| --- | --- | --- |
| "env exists with wrong Python" / env not reused | `drocat-4.5.0` already exists with a different Python. The installers never modify/delete an env with the wrong Python. | Let the installer create the next free name (`drocat-4.5.0-2`, `-3`, …). The launchers resolve the first usable env. Do not manually delete the conflicting env. |
| Re-run installs but the UI still uses the old env | The launcher prefers `drocat-4.5.0`; an earlier failed run left a stale `-2`/`-3` env on `PATH`. | Confirm the env actually used via `config_local.json` `envs."4.5.0"` after a run; or remove the stale env with user approval. |
| A custom env name is ignored | The `envs` entry is version-specific (`envs."4.5.0"`), only consulted for the **current** `APP_VERSION`. | Set it for the exact version being installed; an empty entry means "auto-create/use the default". |
| Environment recreated instead of reused | `drocat-4.5.0` exists but with the wrong Python. | Same as the first row — a `-2`/`-3` env is created on purpose. |
| "pip check" fails after install | A dependency was upgraded or a user-site package leaked in. | Re-run the installer. Ensure `PYTHONNOUSERSITE=1` is set before pip runs (the installers do this). Never upgrade individual pinned packages. |
| Versions appear installed but imports fail | Metadata-only wheel from a failed PEP 517 sdist build (e.g. `img2pdf`/`asciitree`), cached as "success". | The installer detects this and retries once after clearing the pip wheel cache. If the retry also fails, it aborts — clear the wheel cache and re-run. |
| `neuronbridge-python` conflict | The legacy distribution pins Pydantic 2.9, conflicting with NiceGUI. | Do **not** install `neuronbridge-python`. The installer removes it when repairing an older versioned env. |

## Clone and download failures

| Symptom | Root cause | Resolution |
| --- | --- | --- |
| `git clone` fails with `HTTP2 framing layer` or `Failed to connect to github.com port 443` while other sites load | Transient GitHub-side outage (observed while PyPI/Anaconda stayed reachable). `git` does not retry on its own. | Retry the clone a few times spaced 30-60 s; if it persists, `git config --global http.version HTTP/1.1` or clone over SSH. The repository is ~270 MB. |
| Miniconda download is very slow or gets interrupted | CDN throttling; a single-shot download restarts from byte 0 on every failure. | The installers download into the gitignored `cache/miniconda/` with automatic retry and resume (`curl -C -` on macOS/Linux, `curl.exe -C -` on Windows). Re-running the installer resumes the partial file; a completed-but-uninstalled file is re-downloaded fresh automatically. |
| `error: resolution-too-deep` during `[3/5]` | pip resolver backtracking amplified by a truncated metadata response — not a network error. | A warm-cache retry usually succeeds; the installer retries automatically and reports the actual failure class. The `pymupdf` / `python-pptx` / `reportlab` pins are exact, so a persistent failure means the pinned set itself is unsolvable — tighten `requirements.txt`. |

## Installation environment prerequisites

- **Python 3.10-3.11, 3.11 recommended** (matplotlib 3.10.0 requires >= 3.10).
  The installers create a Python 3.11 env, so the system Python does not matter.
- **Conda missing:** install Miniconda from <https://docs.conda.io/miniconda.html>
  after user approval (`install.sh` / `install.ps1` / `install.bat` do this
  automatically).
- **`PYTHONNOUSERSITE=1`** must be exported before pip so packages in
  `~/.local/lib/python3.11/site-packages` are never treated as already installed.
  The installers and launchers set this automatically.
- **Network:** dependency download and token checks need network access. In a
  sandboxed environment, request escalation for network commands.

## Platform notes

### macOS
- Miniconda at `~/miniconda3` or Anaconda at `~/anaconda3`; both are
  auto-detected by `install.sh` and `mac_DROCAT.command`.
- `mac_DROCAT.command` (double-click) repairs/creates the env through the same
  installer, then launches the UI.
- Matplotlib "font cache" warnings on first import are benign.
- Open the UI with `python ui/app.py`; use `DROCAT_UI_SHOW=0` for headless.

### Windows
- Use `install.ps1` (PowerShell, `-ExecutionPolicy Bypass`) or `install.bat`.
- Do not install the upstream `neuronbridge-python`. DROCAT ships its own API
  client, so the Windows install has no `memray`, Ray, or conflicting Pydantic
  requirement.
- Installers and launchers use `conda run`; shell initialization is not required.
- Non-UTF-8 console (GBK/cp1252): direct library/script use used to die with
  `UnicodeEncodeError: 'gbk' codec can't encode ...`. The library now
  reconfigures its stdio to UTF-8 (errors="replace") at import and
  `windows_DROCAT.bat` sets `PYTHONIOENCODING=utf-8`. Older revisions: set
  the variable manually.
- Cache-Only runs refuse an incomplete cache ("cache is incomplete ..." with
  per-neuron recorded-vs-cached counts) instead of silently returning partial
  results. Complete the cache with a normal server run first, or pass
  `allow_incomplete_cache=True` to accept stamped partial output.

### Linux
- Same flow as macOS (`bash archive/install/install.sh`, `./mac_DROCAT.command`).
- `xdg-open` opens output folders; install `xdg-utils` if missing.
- If `tkinter` is unavailable, native folder pickers fall back to typing paths
  (the UI still works).

## Tokens

- Tokens live in `config.json` (wins per key — a GitHub-pulled copy edits it
  directly); the gitignored `config_local.json` only fills entries left empty
  there. `token_info.txt` / `token_info_local.txt` were removed and are never
  read. `config_local.json` is never auto-created.
- Placeholders: `config.json` ships empty token values and wins per key; real
  tokens can go there directly or in the gitignored `config_local.json`.
- **`NEUPRINT_TOKEN`** from <https://neuprint.janelia.org/account>;
  **`CAVE_TOKEN`** (FlyWire only) from <https://codex.flywire.ai/auth_token>.
  Ask the user; never invent or reuse tokens without permission.
- Token mismatch with the API usually means the token host/application
  credentials differ from the `NEUPRINT_APPLICATION_CREDENTIALS` for that
  dataset — align `config.json` with the dataset's credential profile.

## Data / runtime

- `datasets/` and `cache/` are created automatically; the first query downloads
  the full neuron table (needs token + network, can take minutes).
- FlyWire FAFB requires manually downloaded local files (Settings tab guide);
  BANC auto-prepares from its public bucket (network only, no token).
- UI port 8080 busy: set `DROCAT_UI_PORT` (for example
  `DROCAT_UI_PORT=8081 ./mac_DROCAT.command`) or stop the other process.
- 3D PNG/video exports use Chrome + WebDriver (Kaleido fallback). Install Chrome
  if exports fail.
- Neuron indexes are persistent "system files" under `neuron_indexes/` (not
  `cache/`): `male-cns:v1.0`, `flywire_FAFB_v783`, and `banc_v888` ship
  committed seed indexes; other datasets get their index on first pull. Clearing
  `cache/` never removes them. Refresh the bundled seeds with
  `python src/build_seed_indexes.py`.

## Verification recap

`skills/drocat-install/scripts/verify_install.py` checks:

1. Project layout (installers, UI, backend modules, vispath subproject).
2. Python version (3.10-3.11).
3. Core imports (`numpy`, `pandas`, `polars`, `scipy`, `matplotlib`, `seaborn`,
   `plotly`, `networkx`, `openpyxl`, `xlrd`, `xlsxwriter`, `bokeh`, `requests`,
   `jinja2`, `PIL`, `pydantic`, `neuprint`, `nicegui`, `tqdm`, `cv2`,
   `reportlab`, `pptx`, `img2pdf`).
4. Bundled NeuronBridge client import plus optional integrations (warn-only):
   `caveclient`, `cloudvolume`, `navis`, `flybrains`, `selenium`,
   `webdriver_manager`, `fitz`, `psutil`, `trimesh`.
5. Installed versions against the platform and UI requirement manifests.
6. `pip check` dependency consistency.
7. Atomic-write round trip: a real disk write through the shared
   `write_parquet_atomic` helper (the path that broke BANC/FAFB table
   preparation on Windows with `[Errno 9] Bad file descriptor`).
8. Token check reading `config_local.json` then `config.json` for a
   non-placeholder `NEUPRINT_TOKEN` (advisory unless `--require-token`).
9. UI package import (`ui.app`).
