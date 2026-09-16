---
name: drocat-install
description: Auto-install the DROCAT Drosophila connectome analysis toolkit (drocat) on macOS, Linux, or Windows. Use when the user asks to install, set up, repair, or prepare DROCAT (e.g. "install DROCAT", "set up the connectome toolkit", "fix my DROCAT install", "fresh machine setup"). Invokes the bundled one-click installer to create the conda environment, install dependencies, configure tokens, verify the installation, and launch the web UI, and focuses on troubleshooting version/environment conflicts and custom installation requirements.
---

# DROCAT Install (Layer 0)

## Overview

The installation is fully scripted by the repository's one-click installer. This
skill's job is to **run that installer and verify the result**, then troubleshoot
anything the installer itself cannot resolve. It does not re-implement the
install step by step.

## Completion contract

Unless the user asks for instructions only, complete the installation or repair
in the current agent session. Do not stop after cloning, showing a command, or
starting an installer: wait for the installer and verifier to finish, confirm the
final environment/UI status, and report the exact result. Ask only for required
tokens, an unclear repository path, or an approval that the user must provide.

## Workflow

### 1. Resolve the repository and version

Pull DROCAT v4.5.0 if it is not already present (works in any agent that can run
shell commands):

```bash
git clone --branch v4.5.0 https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git drocat
cd drocat
```

If a checkout already exists, use the branch instead:

```bash
git fetch origin v4.5.0
git checkout v4.5.0
```

- If the user already has the repository locally, reuse it (ask if the path is
  not obvious) and skip the clone.
- A transient GitHub outage can fail the clone itself (`HTTP2 framing layer`,
  connect timeouts while other hosts stay reachable): retry the clone a few
  times spaced 30-60 s; if it persists, set
  `git config --global http.version HTTP/1.1` or clone over SSH. The
  repository is ~270 MB.
- Confirm the required files exist: `mac_DROCAT.command` / `windows_DROCAT.bat`
  (launchers), `archive/install/install.sh` (macOS/Linux), `archive/install/install.ps1` /
  `archive/install/install.bat` (Windows), `ui/app.py`, `src/coana.py`,
  `requirements.txt`.

### 2. Run the one-click installer directly

Call the bundled installer and let it build the environment. Do not recreate its
steps manually.

- **macOS / Linux:** `bash archive/install/install.sh`
  (creates/reuses `drocat-4.5.0` Python 3.11, installs both pinned requirement
  files, installs DROCAT in editable mode, runs `pip check`, runs the bundled
  verifier, and asks for NeuPrint/CAVE tokens on interactive terminals).
- **Windows:** `powershell -ExecutionPolicy Bypass -File archive/install/install.ps1`
  or `archive/install/install.bat`.
- **Self-healing launcher (any platform):** `./mac_DROCAT.command` (macOS/Linux)
  or `windows_DROCAT.bat` (Windows) invokes the same installer when the environment is
  missing or inconsistent, then launches the UI.

The installers are idempotent: an existing `drocat-4.5.0` environment with the
right Python is reused and updated in place. Treat a failed dependency
installation or a failed `pip check` as an installation failure — do not continue
with a partially inconsistent environment.

### 3. Verify

Run the bundled verifier with the environment Python:

```bash
conda activate drocat-4.5.0
python skills/drocat-install/scripts/verify_install.py --project /path/to/repo
```

Required checks: Python 3.10-3.11, project layout, every installed version
against the pinned manifests, dependency metadata (`pip check`), core imports
(including the bundled `neuronbridge_client`), and UI import. Token configuration
is advisory during installation; use `--require-token` when validating a
configured workstation. The verifier needs no network.

### 4. Launch the UI

- `./mac_DROCAT.command` (macOS/Linux) or `windows_DROCAT.bat` (Windows), or
  `conda activate drocat-4.5.0 && python ui/app.py`.
- Confirm the server responds: `curl -s http://127.0.0.1:8080/` should contain
  `DROCAT`.
- Override the host/port with `DROCAT_UI_HOST`, `DROCAT_UI_PORT`, or
  `DROCAT_UI_SHOW=0` for a headless launch. First run: datasets auto-download on
  first query (requires token + network). FlyWire FAFB additionally requires
  manually downloaded data files (the Settings tab has the guide); BANC
  (`banc_v626`/`banc_v888`) auto-prepares from its public bucket — no token,
  no manual downloads.

## After installation: analysis skills are already local

Installing the repository also installs the analysis skills under `skills/`, so
the agent already has them — no fetch is required:

- **Layer 1 — [drocat-usage](../drocat-usage/SKILL.md):** agent-assisted,
  tab-matched direct-script analyses (pathfinding, comparison, NeuronBridge,
  FlyLight, visualization, dataset settings) that mirror each UI tab's backend
  call.
- **Layer 2 — [drocat-backend](../drocat-backend/SKILL.md):** flexible usage of
  the backend modules and function blocks for custom combinations.
- **[type-mapping-validation](../type-mapping-validation/SKILL.md):** bodyId-level
  validation of an auto cross-dataset type mapping (category partition, nested
  modes, gap-fill review scenes) via `scripts/RunMappingValidation.py`.

Keep the UI closed for unattended runs and use `showfig=False` until the output
has been validated.

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for the
environment/version conflict matrix and platform notes, and
[references/custom-installation.md](references/custom-installation.md) for all
supported custom-installation variants (custom env name, custom Python, custom
paths, offline install, token-only flows).
