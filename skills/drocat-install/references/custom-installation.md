# DROCAT Install — Custom Installation

The one-click installer covers the default case. This reference covers every
supported custom-installation variant. All of them still go through the bundled
installer; nothing below re-implements the install.

## Custom conda environment name

Pin a per-version env name in `config.json` (`envs` section — config.json wins
per key; it is the file a GitHub-pulled copy edits directly). The gitignored
`config_local.json` is the developer-specific fallback and is never auto-created:

```json
{ "tokens": { "neuprint": "", "cave": "" }, "envs": { "4.5.0": "my-custom-env" } }
```

Rules:

- The `envs` entry is **version-specific** — it is only consulted for the current
  `APP_VERSION`, so upgrading never reuses an older release's custom environment.
- If the custom env exists with Python 3.11 it is reused; if missing it is
  created; if it exists with a different Python the installers abort with a clear
  error (never silently switch envs or rewrite the config).
- An **empty** `envs."4.5.0"` means "create/use the default versioned env".
  After the environment is selected, the installers **and** launchers write the
  actual env name back into `config_local.json`; the committed `config.json` is
  never rewritten.
- The launchers (`mac_DROCAT.command` / `windows_DROCAT.bat`) resolve the same
  override first.

To force a fresh env even when `drocat-4.5.0` exists, set a new custom name in
`envs."4.5.0"` (or delete the wrong-Python env with the user's approval).

## Custom Python version

Default is Python 3.11 (matplotlib 3.10.0 requires >= 3.10; validated through
3.11). If you must use a specific interpreter:

- The installer always creates a Python 3.11 env. To pin a different interpreter
  for the UI, run `conda create -n <env> python=<ver>` yourself, install the two
  pinned requirement files plus the project in editable mode, run `pip check`,
  then set `envs."4.5.0"` to that env name.
- Keep the interpreter's Python within 3.10-3.11; `verify_install.py` rejects
  other versions.
- Do not use the system Python (non-conda) for the UI — the pinned dependency set
  is only reproducible in a clean conda env.

## Custom repository location / moving the project

- The installer and launchers are path-agnostic: they resolve the repository root
  from their own location (`archive/install/install.sh` walks up two levels;
  `run_direct.py` accepts `--repo`).
- `verify_install.py` accepts `--project /path/to/repo`.
- Cached data is relative to the repository root; if you move the repo, copy
  `datasets/`, `cache/`, `neuron_indexes/`, and `config_local.json` with it, or
  re-download on demand.

## Offline / proxy install

- Run the installer once on a networked machine to populate the conda package
  cache, then set `CONDA_OFFLINE=true` and `PIP_NO_INDEX=1` for the offline run.
- Keep `PYTHONNOUSERSITE=1`. If a wheel build fails offline, pre-download the
  sdists and place them in the pip cache before running.
- Dependency download and token checks need network; request escalation inside a
  sandbox.

## Token-only flow (repair, no re-install)

When dependencies are already installed, verify and configure tokens without a
full re-install:

```bash
conda activate drocat-4.5.0
python skills/drocat-install/scripts/verify_install.py --project . --require-token
# then ensure tokens are set in config.json (wins per key) or config_local.json
```

- Tokens: `NEUPRINT_TOKEN` from <https://neuprint.janelia.org/account>;
  `CAVE_TOKEN` (FlyWire only) from <https://codex.flywire.ai/auth_token>. Ask the
  user; never invent or reuse tokens without permission.

## Headless / server install

- Launch without a browser: `DROCAT_UI_SHOW=0 python ui/app.py`.
- Set `DROCAT_UI_HOST` and `DROCAT_UI_PORT` for a non-default bind (port 8080 is
  default). If the port is busy the launcher lists the running DROCAT instances
  and offers a new port or stopping them all, or you can set the env var to a
  free port.
- Everything (analysis scripts, caches, tokens) works without the UI; the
  analysis skills (`drocat-usage` Layer 1, `drocat-backend` Layer 2) run in the
  same `drocat-4.5.0` environment.

## Install in a restricted sandbox

- Request escalation for network commands (dependency download, token checks).
- Run `verify_install.py` offline — it needs no network.
- If the sandbox blocks conda activation, use `conda run -n drocat-4.5.0 ...` or
  the absolute env Python.
