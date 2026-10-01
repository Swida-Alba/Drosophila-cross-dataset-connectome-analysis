# *Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.11](https://img.shields.io/badge/Python-3.10--3.11-3776AB.svg)](https://www.python.org/downloads/)

DROCAT is a Python toolkit for analyzing and visualizing connectome data from **all NeuPrint databases, FAFB, and standalone BANC releases** — type-based pathfinding, interactive network visualizations with neurotransmitter grouping, 3D neuron morphology rendering, cross-dataset comparison, and EM↔LM driver line mapping (NeuronBridge). Everything is available both through a web UI and as standalone scripts.

> [!TIP]
> 🤖 **Agent-assisted:** ask your AI agent to run the bundled
> [`drocat-install`](skills/drocat-install/SKILL.md) skill — it installs all
> dependencies, configures tokens, verifies the installation, and launches
> the web UI for you. Installing the project also installs the analysis skills,
> so the agent already has them afterward (no fetch is needed):
> [`drocat-usage`](skills/drocat-usage/SKILL.md) for tab-matched script analyses
> (a recipe for every analysis panel), [`drocat-backend`](skills/drocat-backend/SKILL.md)
> for flexible composition of backend modules, and
> [`type-mapping-validation`](skills/type-mapping-validation/SKILL.md)
> for bodyId-level type-mapping validation and gap-fill review. New to
> agents? Start with the
> [agent-assisted install section](docs/INSTALLATION.md#5-agent-assisted-install).

---

## Table of Contents

- [*Drosophila* Connectome Analysis Toolkit (DROCAT) v4.5.0](#drosophila-connectome-analysis-toolkit-drocat-v450)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Quick Start](#quick-start)
  - [Documentation](#documentation)
  - [Supported Datasets](#supported-datasets)
    - [FAFB + standalone BANC local releases (3)](#fafb--standalone-banc-local-releases-3)
  - [What's New in v4.5.0](#whats-new-in-v450)
  - [Releases](#releases)
  - [Contributing](#contributing)
  - [License](#license)
  - [Support](#support)

---

## Key Features

| Feature | Details |
| --- | --- |
| **Dataset Support** | NeuPrint (hemibrain, male-cns, optic-lobe, manc) + FAFB + standalone BANC releases, inter-dataset analysis |
| **EM↔LM Mapping** | NeuronBridge integration for GAL4/Split-GAL4 driver line discovery |
| **Visualization** | 3D skeletons, interactive networks, Sankey diagrams, heatmaps |
| **Similarity Tabs** | Connectivity (find similar + comparison, cross-dataset capable) and Morphology (find similar, intra-dataset) with connectivity-expanded candidates, ROI filtering, full-morphology downloads. Morphology find similar defers BANC pending skeleton vector-quality validation; the morphology comparison itself runs on BANC with a provisional-scores warning (BANC 3D visualization and pathfinding are supported) |
| **Analysis** | Multi-hop pathfinding, cross-dataset comparison, hemisphere-aware analysis |
| **Performance** | Substantially faster repeat queries through local caching and Polars acceleration (local releases avoid the network round-trips of API-backed access) |

---

## Quick Start

**Requirements:** conda (auto-installed if missing) and internet access on first run.

**Option 1 — One-click install & launch.** After cloning the repository, launch DROCAT with the bundled launcher:

| Platform | Command |
| --- | --- |
| macOS | double-click `mac_DROCAT.command` |
| Linux | run `./mac_DROCAT.command` in a terminal |
| Windows | double-click `windows_DROCAT.bat` |

On first run it creates the versioned `drocat-4.5.0` Python 3.11 environment (via the bundled installer in `archive/install/`; Python 3.10–3.11 is supported), installs the pinned dependencies, runs `pip check`, verifies the installation, and opens the web UI at **http://127.0.0.1:8080**. Later runs are self-healing: a missing or inconsistent environment is repaired automatically before starting. If the port is busy — or other DROCAT instances are already listening — the launcher lists them (port, PID, command) and offers to start on a new port, stop them all and restart, or cancel.

**Option 2 — Agent-assisted install.** Copy the following prompt to your AI agent and let it finish cloning the repo, installing, verifying, and launching DROCAT:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md and follow it to finish cloning the repo, installing, verifying, and launching DROCAT on this machine.

For script analysis without the UI, the agent uses the checked-in analysis skills
(with the UI closed): [`drocat-usage`](skills/drocat-usage/SKILL.md) for
one-tab analyses, [`drocat-backend`](skills/drocat-backend/SKILL.md) for
flexible backend composition, and
[`type-mapping-validation`](skills/type-mapping-validation/SKILL.md) for
bodyId-level type-mapping validation. They are part of the repository, so an
installed agent has them — no fetch is required.

**Manual launch** (after installation) — double-click `mac_DROCAT.command` (macOS / Linux) or `windows_DROCAT.bat` (Windows), or from a terminal:

```bash
conda activate drocat-4.5.0 && python ui/app.py
```

Every UI panel links to its own instruction guide (see [docs/ui_guides/README.html](docs/ui_guides/README.html)).

📖 **[Full Installation Guide](docs/INSTALLATION.md)** — installer details, manual setup, environment policy, token configuration (NeuPrint / CAVE), and agent-assisted install.

---

## Documentation

**Start here:**

| Guide | Description |
| --- | --- |
| **[Quick Start](docs/QUICK_START.md)** | First-time setup and basic examples |
| **[Installation](docs/INSTALLATION.md)** | One-click, agent-assisted, and manual install + token configuration |
| **[Script Examples](docs/core-features/ScriptExamples_Guide.md)** | Copy-paste code for pathfinding, comparison, NeuronBridge |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |
| **[Documentation Hub](docs/README.md)** | Full documentation index |

**Feature guides:**

| Feature | Guide | Script |
| --- | --- | --- |
| **Basic Usage** | [Basic Usage Guide](docs/core-features/BasicUsage_Guide.md) | `FindDirect.py`, `FindPath.py` |
| **Score Calculations** | [Score Calculation Guide](docs/core-features/ScoreCalculation_Guide.md) | All pathfinding scripts |
| **EM↔LM Mapping** | [NeuronBridge Guide](docs/core-features/NeuronBridge_Guide.md) | `NeuronBridge_FindLines.py` |
| **FlyLight Imagery** | [FlyLight Guide](docs/core-features/FlyLight_Guide.md) | `FlyLight_fetcher.py` |
| **Cross-Dataset** | [Comparison Guide](docs/core-features/CrossDatasetComparison_Guide.md) | `InterDatasetComparator.py` |
| **Homolog Finding** | [Homolog Guide](docs/core-features/HomologFinding_Guide.md) | `FindHomologs.py` (Connectivity tab → Find Homolog) |
| **3D Visualization** | [3D Skeleton Guide](docs/visualizations/3D_Skeleton_Guide.md) | `plot3dSkeleton.py` |
| **Path Visualization** | [Interaction Guide](docs/visualizations/VisualizePath_Interaction_Guide.md) | `PlotPath.py` |
| **Web UI Panels** | [docs/ui_guides/README.html](docs/ui_guides/README.html) | All web UI panels |
| **Output Files** | [Output Files Reference](docs/OUTPUT_FILES.md) | File formats |

---

## Supported Datasets

All NeuPrint server datasets are supported (verified against `api.neuprint.janelia.org`), plus the FAFB and standalone BANC local releases. NeuPrint datasets are fetched automatically; FAFB uses local Codex files — auto-downloadable from the Settings tab with a FlyWire Codex token (skeletons also fetch lazily per neuron) — plus an advanced optional CAVE fallback (config-file only), while BANC uses its public release bucket with no CAVE token (see the Settings tab).

<details>
<summary><b>NeuPrint (11 datasets)</b> — male-cns, hemibrain, optic-lobe, manc, fib19, mushroombody</summary>

| Dataset | Description (from NeuPrint server) |
| --- | --- |
| `male-cns:v1.0` | Complete MaleCNS connectome (Janelia FlyEM + Cambridge) — latest |
| `male-cns:v0.9` | Complete MaleCNS connectome |
| `hemibrain:v1.2.1` | Adult female brain reconstruction (central complex + surrounding neuropils) |
| `hemibrain:v1.1` | Older hemibrain release |
| `optic-lobe:v1.1` | Drosophila optic lobe (right lobe, ~50k neurons, subset of MaleCNS) |
| `optic-lobe:v1.0.1` | Fly optic lobe reconstruction (~50k neurons) |
| `manc:v1.2.3` | MANC connectome — latest |
| `manc:v1.2.1` | MANC connectome |
| `manc:v1.0` | MANC connectome (original) |
| `fib19:v1.0` | Partial reconstruction of the fly medulla / lobula / lobula plate |
| `mushroombody` | Fly alpha lobe in the mushroom body (983 neurons) |

</details>

### FAFB + standalone BANC local releases (3)

| Dataset | Description |
| --- | --- |
| `flywire_FAFB_v783` | Female Adult Fly Brain (FAFB v783, 139,255 neurons) |
| `banc_v888` | Brain and Nerve Cord (BANC v888; standalone public-bucket tables + skeletons) |
| `banc_v626` | Brain and Nerve Cord, older (BANC v626; standalone public-bucket tables + skeletons) |

> BANC (Brain And Nerve Cord) is a standalone source analyzed from its own public release bucket as `banc_v888`/`banc_v626` — neuron metadata and connections prepare automatically (~134 MB once, no login or CAVE token), and 3D skeletons are fetched on demand and cached, so no bulk skeleton download is required. The NeuPrint server metadata also lists a hidden `banc:v888` entry, but it is not queryable through the NeuPrint API; the supported BANC path is the standalone local release above, not the NeuPrint-backed one.

📖 **[FAFB + BANC Setup Guide](docs/FLYWIRE_USAGE.md)** · **[Available ROI Meshes](docs/AVAILABLE_ROIS.md)** · **[BANC Integration Guide](docs/BANC_INTEGRATION.md)**

---

## What's New in v4.5.0

- **Script-first analysis with coding agents** — run pathfinding, comparison, NeuronBridge, FlyLight, homolog, profile, PlotPath, and 3D skeleton scripts without the UI, via the [`drocat-usage`](skills/drocat-usage/SKILL.md) skill and its `run_direct.py` launcher.
- **Cross-Dataset group** — the connection-comparison tab moved into its own top-level **Cross-Dataset** group as **Paths**, alongside the new **Type Validation** tab that runs the TM VEV pipeline (`MappingValidator`) from the UI: bodyId-level validation of a source→target type mapping, candidate expansion, and 3D review scenes (see [type_validation.html](docs/ui_guides/type_validation.html)).
- **Local FAFB + standalone BANC dataset support** — local-first FAFB caching and public-bucket BANC caching so repeated local-release queries avoid network round-trips entirely; Polars-backed matrix/CSV steps measured 10-100x faster in the [December 2025 benchmarks](docs/technical/PERFORMANCE_OPTIMIZATIONS_DEC2025.md) ([FAFB Integration](docs/FAFB_INTEGRATION.md), [BANC Integration](docs/BANC_INTEGRATION.md)).
- **NT visualization & grouping** — neurotransmitter edge groups, custom groups, export/import ([Network Features](docs/visualizations/VisualizePath_Network_Features.md)).
- **Similarity tab reorganization** — the Similarity group is now two main tabs, each with a similarity-search / Comparison sub-tab pair (Find Homolog on Connectivity, Find Similar on Morphology):
    - **Connectivity**: Find Homolog is a homolog search across datasets (or within one dataset via Target = Source); comparison is multi-dataset connectivity profiling.
    - **Morphology**: find similar is an intra-dataset vector/NBLAST search; comparison is an N×N morphology comparison (one dataset) that switches to cross-dataset mode when two or more datasets are selected. An Aggregation Level picks what a row is — a type, one neuron, or a custom group — and the bodyId matrix is always the scored primitive, with heatmaps and a report (`vector_v2` or NBLAST scoring; NBLAST warns past 30 neurons and proceeds).
    - The old "Connectivity similarity" mode is folded into Find Homolog (it was intra-dataset homolog finding under another name); see the [Connectivity](docs/ui_guides/connectivity.html), [Morphology](docs/ui_guides/morphology.html), and [Comparison](docs/ui_guides/morphology_comparison.html) guides for connectivity-expanded candidates, ROI filtering, dual result tables, and the resumable full-morphology download.
- **Palette editor** — drag-and-drop reordering of discrete palette colors, a range slider applied directly to the displayed palette, a reset button beside the preview, and lateral range labels.
- **3D Skeleton reorganization** — independent card blocks for general appearance, neuron colors, synapse colors, and brain-region ROIs, with hemisphere-aware options.

📖 **[Full changelog](docs/README.md#v450-changelog)** · **[Agent-assisted install](docs/INSTALLATION.md#5-agent-assisted-install)**

---

## Releases

DROCAT ships one versioned branch per release (`v4.5.0`, `v4.4.5`, …), each mirrored as a Git tag and a [GitHub Release](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases). The default branch tracks the current release, and every release pins its own conda environment name (`drocat-<version>`), so multiple versions can coexist on one machine.

| Release | Date | Highlights |
| --- | --- | --- |
| **[v4.5.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/tree/v4.5.0)** — current | August 2026 | Script-first agent analysis, local FAFB + standalone BANC support, Similarity tab reorganization, NT grouping, palette editor, 3D skeleton reorganization (see [What's New](#whats-new-in-v450)) |
| [v4.4.5](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v4.4.5) — latest published | 2026-08-04 | Agent-assisted direct runs |
| [v4.4.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v4.4.0) | 2026-01-10 | Local FAFB + standalone BANC datasets, priority-based neuron search, NT visualization |
| [v4.3.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v4.3.0) | 2025-12-23 | NeuronBridge integration with region filtering, FlyLight |
| [v4.2.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v4.2.0) | 2025-12-14 | FAFB download guidance and fixes |
| [v4.0.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v4.0.0) | 2025-11-24 | Modular v4 rearchitecture |

Earlier releases (v4.4.1–v4.4.3, v4.3.1–v4.3.3, v4.2.1, [v3.2.1](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v3.2.1), [v3.2](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v3.2), [v3.0](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v3.0), [V3.1](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/V3.1), and the [v2.1 beta](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases/tag/v2.1-beta)) are on the [Releases page](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/releases); every version is also available as a [Git tag](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/tags).

**Get a specific release:**

```bash
git clone https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git
cd Drosophila-cross-dataset-connectome-analysis
git checkout v4.4.5   # a tagged release; omit to stay on the current default branch
```

Consolidated per-release notes live in the [changelog](docs/README.md#v450-changelog).

---

## Contributing

Contributions are welcome — please open an issue or PR on [GitHub](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis).

## License

MIT License — see [LICENSE](LICENSE).

## Support

- **[Documentation](docs/README.md)** — full index
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — common issues
- **[Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)** — bug reports and feature requests

When reporting an issue, include: Python version and OS, the full error traceback, and minimal code to reproduce.
