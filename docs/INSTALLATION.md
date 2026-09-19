# Installation Guide

DROCAT v4.5.0 needs Python 3.10-3.11 and a conda environment. The one-click
installers handle everything: conda discovery (installing Miniconda if
missing), a fresh `drocat-4.5.0` environment (Python 3.11), pinned
dependencies, `pip check`, and a bundled verifier.

## 1. One-click install (recommended)

| Platform | Install + Launch |
| --- | --- |
| macOS | Double-click `mac_DROCAT.command` (installs on first run) |
| Linux | `./mac_DROCAT.command` |
| Windows | Double-click `windows_DROCAT.bat` (installs on first run) |

Standalone installers (usually only needed for repairs or scripting): `bash archive/install/install.sh` (macOS/Linux), `powershell -ExecutionPolicy Bypass -File archive/install/install.ps1` (Windows).

The UI opens at <http://127.0.0.1:8080>.

The launchers are self-healing: if the environment is missing or inconsistent
(imports fail or `pip check` reports conflicts), they run the installer
automatically before starting the UI.

## 2. Environment policy

- Preferred env: **`drocat-4.5.0`** (Python 3.11). Re-running the installer
  updates it in place.
- If that name is taken by a non-3.11 env, it is left untouched and a suffix
  is used: `drocat-4.5.0-2`, `drocat-4.5.0-3`, ... Launchers resolve the same
  way. Legacy `drocat` envs are never modified.
- **Custom env name (version-specific):** put the env name under the matching
  release key in `config.json` (`envs` section) - on a GitHub-pulled copy
  this is the file to edit, and it wins per key. The gitignored
  `config_local.json` is the developer-specific fallback: it is only
  consulted when the `config.json` entry is empty. The entry is only
  consulted for *that* release, so upgrading DROCAT never reuses an older
  release's custom environment. An empty value means default auto-find:

  ```json
  {
    "tokens": { "neuprint": "", "cave": "" },
    "envs": { "4.5.0": "my-custom-env" }
  }
  ```

  If the custom env exists with Python 3.11 it is reused; if it does not
  exist it is created; if it exists with a different Python the installers
  abort with a clear error (never silently switching envs or rewriting the
  config).
- **Auto-fill:** an empty `envs."4.5.0"` entry means "create/use the default
  versioned env automatically". When `config_local.json` already exists, the
  installers and launchers write the environment they actually used back into
  it (never `config.json`), so after the first install the fallback entry holds
  the concrete env name (e.g. `drocat-4.5.0`) and every script resolves it
  directly - while the `config.json` entry stays empty. The auto-fill only
  edits an existing file: it never creates `config_local.json` (the UI Settings
  tab creates it when you save tokens). Editing `config.json` always wins.
- All installers/launchers set `PYTHONNOUSERSITE=1`, so packages from a
  user-level Python cannot contaminate the environment.
- Do **not** install `neuronbridge-python` — DROCAT bundles its own client
  (`src/neuronbridge_client.py`) to avoid that package's incompatible
  Pydantic pin and heavy dependency tree.

## 3. Manual install

```bash
conda create -n drocat-4.5.0 python=3.11 -y
conda activate drocat-4.5.0
export PYTHONNOUSERSITE=1                      # PowerShell: $env:PYTHONNOUSERSITE="1"

pip install -r requirements.txt -r ui/requirements.txt            # Windows: requirements-windows.txt
pip install -e . --no-deps
python -m pip check
python skills/drocat-install/scripts/verify_install.py --project .
```

## 4. Authentication

NeuPrint datasets need a **NeuPrint token** (required). The **CAVE token is
optional** — it is only needed for FlyWire FAFB *online* fetching; local
converted FlyWire tables work without it. Tokens live in the project configs:
`config.json` (wins per key; the file a GitHub-pulled copy edits directly)
and the gitignored `config_local.json` (developer-specific fallback for
empty entries). `token_info.txt` / `token_info_local.txt`
were removed and are never read. Two equivalent ways:

1. **UI Settings tab** — paste the tokens there after launching; the tab
   reminds you when tokens are missing. Saving writes `config_local.json`
   and keeps its `envs` section intact (the committed `config.json` stays
   clean).
2. **Config file** — edit `config.json` directly; a GitHub-pulled copy is
   yours to modify. Git users who want to keep tokens out of version control
   can instead copy the template to the gitignored override and edit that -
   it only applies to entries left empty in `config.json`:

```bash
cp config.json config_local.json     # gitignored developer fallback
```

```json
{
  "tokens": {
    "neuprint": "your_actual_neuprint_token",
    "cave": "your_actual_cave_token"
  },
  "envs": {
    "4.5.0": ""
  }
}
```

An empty token value means "not configured"; the environment variables
(`NEUPRINT_TOKEN`, `CAVE_TOKEN`) remain a fallback for script mode.
`config_local.json` is NOT created automatically - it is an optional,
developer-specific file. Create it manually when you need local overrides:
`cp config.json config_local.json` (the file is gitignored). `config.json`
itself always ships with the repository.

- NeuPrint token: <https://neuprint.janelia.org/account>
- CAVE token: <https://codex.flywire.ai/auth_token>
- FlyWire FAFB also needs its local data files in
  `datasets/flywire_FAFB_v783/downloads/` — see the Settings tab guide.
  BANC (`banc_v626`/`banc_v888`) needs no manual downloads: it auto-prepares
  from its public release bucket on first use (network only, no token).
- Strict verification for a configured workstation:
  `python skills/drocat-install/scripts/verify_install.py --project . --require-token`

### 4.1 What the first run downloads

Installation verifies packages, not dataset readiness. Expect a one-time
wait on the first real query:

- **NeuPrint datasets** (hemibrain, male-cns, optic-lobe, ...): the first
  query downloads the full neuron tables into `datasets/<dataset>/` (needs
  the NeuPrint token) and builds the connection cache incrementally as you
  query. Later runs reuse it.
- **BANC** (`banc_v626`/`banc_v888`): the neuron meta and connection tables
  are pulled from the public release bucket automatically — no token.
- **Cache-Only (offline) runs** require a *verifiably complete* pre-built
  cache. The run verifies cache coverage two ways: per-neuron recorded vs
  actually-cached (deduplicated) connection counts, and the whole-cache
  distinct-connection count against `cache/<dataset>/cache_manifest.json`.
  Caches built before this verification existed — and caches copied in from
  a backup or a colleague — have no manifest, so the run cannot tell
  "complete" from "truncated" and refuses. Fix it without a network:
  **Settings → Storage → "Certify local cache"**, or
  `python scripts/maintenance/record_cache_baseline.py --dataset <id>`.
  Certification records the manifest only after the per-neuron check passes,
  and an online run still records the baseline by itself. Nothing is ever
  run against a cache that is known to be incomplete.

## 5. Agent-assisted install

### 5.1 Agent-assisted install

In any coding agent, ask:

> Install DROCAT on this machine, verify it works, and report the final environment and UI status.

The agent follows the bundled
[`drocat-install` skill](../skills/drocat-install/SKILL.md). The skill both
installs the project and installs the analysis skills with it, so after the
install the agent already has them (no fetch is required).

To fetch the release first:

```bash
git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
./mac_DROCAT.command
```

For agent-driven analysis *without* the UI, use the checked-in skills:
[`drocat-usage`](../skills/drocat-usage/SKILL.md) (Layer 1, a recipe for every analysis panel)
and [`drocat-backend`](../skills/drocat-backend/SKILL.md) (Layer 2, backend
module composition). For bodyId-level type-mapping validation and
gap-fill review, use
[`type-mapping-validation`](../skills/type-mapping-validation/SKILL.md).

### 5.2 Running scripts directly

The skill's launcher keeps the script's relative paths correct and adds the
repository modules to `PYTHONPATH`:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script scripts/FindPath.py \
  --dry-run
```

Remove `--dry-run` to execute. The main direct entry points are:

| Goal | Script |
| --- | --- |
| Direct one-hop edges | `scripts/FindDirect.py` |
| Multi-hop pathways | `scripts/FindPath.py` |
| Path/Sankey/network HTML | `scripts/PlotPath.py` |
| 3D morphology | `scripts/plot3dSkeleton.py` |
| Cross-dataset comparison | `scripts/InterDatasetComparator.py` |
| Homolog finding | `scripts/FindHomologs.py` |
| Connectivity profiles | `scripts/ConnectivityProfiling.py` |
| EM → LM lines | `scripts/NeuronBridge_FindLines.py` |
| LM → EM neurons | `scripts/NeuronBridge_FindNeuron.py` |
| Co-labeling | `scripts/NeuronBridge_Colabel.py` |

Start with small cached queries, `showfig=False`, CSV output, and a unique
output folder. Inspect CSV schemas and HTML existence before increasing hop
counts, enabling bodyId-level work, or exporting images/video.

### 5.3 Safety checklist

- Never paste `NEUPRINT_TOKEN`, `CAVE_TOKEN`, or the contents of
  `config.json` into a prompt, patch, log, or report.
- Ask before remote queries that may download large datasets or spend API/data
  budget.
- Keep `use_cache=True`; use `cache_only=True` only when cache coverage is
  known.
- Do not silently change dataset, thresholds, pathfinding algorithm, or mesh
  transforms to make a run pass.
- Preserve the example scripts; make changes in a copied run script or an
  explicitly requested source file.
- Keep outputs under `local_data/agent_runs/` or another user-approved folder.

For the complete operation catalog and recipes, see
[`skills/drocat-usage/SKILL.md`](../skills/drocat-usage/SKILL.md).

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `pip check` fails / dependency conflict | Re-run the one-click installer; it repairs the env. The release pins `chardet==5.2.0` (Requests/CloudVolume compatibility). |
| Port 8080 busy | Running `mac_DROCAT.command` / `windows_DROCAT.bat` interactively now asks what to do: **[1]** start on a new port, **[2]** kill the existing DROCAT process and restart on the same port (non-DROCAT processes are never auto-killed), **[3]** cancel. Manual override: `DROCAT_UI_PORT=8081 ./mac_DROCAT.command` (Windows: `set DROCAT_UI_PORT=8081 && windows_DROCAT.bat`). |
| PNG/video export fails | Install Google Chrome (WebGL/ChromeDriver). Kaleido is the fallback. |
| Native folder picker missing | Type the path directly. Linux may need `tkinter` / `xdg-utils`. |

## 7. Standalone VisPath

The VisPath subproject (`vispath-subproject/`) reuses the same environment.
The UI loads it from `vispath-subproject/src` directly — no separate install
needed. To use it as a library:

```bash
cd vispath-subproject && pip install -e .
```
