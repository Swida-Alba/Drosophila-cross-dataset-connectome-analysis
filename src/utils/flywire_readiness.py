"""Readiness checks for FAFB and standalone BANC morphology workflows.

The connectivity tables and morphology sources are different resources. FAFB
can use either its downloaded local skeleton bundle or the CAVE API. BANC uses
the public release bucket and never uses FlyWire Codex or a CAVE token. Keep
those rules in one small, dependency-light module so UI-facing scripts and
library entry points agree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

try:
    from .naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name

# Re-exported for compatibility (the canonical definitions live in
# flywire_ids).
try:
    from ..flywire_ids import is_banc_dataset, is_fafb_dataset
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from flywire_ids import is_banc_dataset, is_fafb_dataset


class FlyWireSkeletonAccessError(RuntimeError):
    """Raised when a FAFB morphology workflow cannot obtain skeletons."""


def dataset_folder(dataset: object) -> str:
    """Map a dataset identifier to the repository folder convention."""

    return canonical_dataset_name(dataset).replace(":", "_").replace(".", "_")


def flywire_manual_skeleton_instruction(
    dataset: object, dataset_dir: str | Path | None = None,
) -> str:
    """Return the manual-download instruction for FAFB skeleton bundles.

    Bulk skeleton downloads are disabled for FAFB: its healed skeleton bundle
    is a large one-time download from FlyWire Codex. BANC is handled by the
    early standalone public-bucket branch below.
    """

    dataset_name = str(dataset or "")
    key = "banc" if is_banc_dataset(dataset_name) else "fafb"
    converter = "BANC_file_converter" if key == "banc" else "FAFB_file_converter"
    folder = dataset_folder(dataset_name)
    if dataset_dir is None:
        root = Path(__file__).resolve().parents[2]
        dataset_dir = root / "datasets" / folder
    download_dir = Path(dataset_dir) / "downloads"
    if key == "banc":
        return (
            f"Download All Skeletons is not needed for BANC ('{dataset_name}'):\n"
            "  skeletons fetch on demand from the public BANC release bucket "
            "(banc_public_data.fetch_banc_swc)\n"
            f"  and cache locally under cache/{folder}/skeletons/"
            "raw_skeletons/ during visualization."
        )
    bundle = "the FAFB skeleton bundle (sk_lod1_783_healed.zip) from"
    return (
        f"Download All Skeletons is disabled for FAFB datasets "
        f"('{dataset_name}'); skeletons must be downloaded manually:\n"
        f"  1. Download {bundle} https://codex.flywire.ai/api/download?dataset={key}\n"
        f"  2. Save the download into {download_dir}\n"
        f"  3. Run the one-time converter: python src/{converter}.py\n"
        "Alternatively, save a FlyWire Codex API token (Settings > API "
        "Tokens) and use Settings > FAFB Data Downloads: skeletons fetch "
        "lazily per neuron from the Codex server during visualization, or "
        "the full bundle downloads as level 3.\n"
        "The converter moves the bundle into the dataset folder and builds "
        "the local tables; on-demand CAVE fetches for individual missing "
        "skeletons still work during visualization."
    )


def print_download_instructions(
    dataset: object,
    dataset_dir: str | Path | None = None,
) -> None:
    """Print source-specific preparation instructions for missing local tables.

    FAFB uses a one-time Codex download and converter. BANC uses its public
    release bucket and does not display a Codex URL, request a CAVE token, or
    require manual skeleton downloads.
    """

    dataset_name = str(dataset or "")
    key = "banc" if is_banc_dataset(dataset_name) else "fafb"
    converter = "BANC_file_converter" if key == "banc" else "FAFB_file_converter"
    if dataset_dir is None:
        root = Path(__file__).resolve().parents[2]
        dataset_dir = root / "datasets" / dataset_folder(dataset_name)
    download_dir = Path(dataset_dir) / "downloads"
    if key == "banc":
        print()
        print("=" * 70)
        print("BANC PUBLIC RELEASE — AUTOMATIC PREPARATION")
        print(f"Local BANC tables for '{dataset_name}' were not found.")
        print("DROCAT prepares metadata and connections from the public BANC")
        print("release bucket; no FlyWire/Codex login and no CAVE token are needed.")
        print("Skeletons are fetched on demand from banc_public_gcs and cached")
        print(f"as .swc.zst files under cache/{dataset_folder(dataset_name)}/skeletons/.")
        print("Run the BANC converter only when preparing a local table bundle:")
        print(f"  python src/{converter}.py")
        print("=" * 70)
        return

    required = (
        "neurons.csv.gz + connections_princeton.csv.gz"
        if key == "banc"
        else "classification.csv.gz + connections_princeton_no_threshold.csv.gz"
    )
    print()
    print("=" * 70)
    print(f"MISSING LOCAL FILES — ONE-TIME DOWNLOAD + CONVERSION REQUIRED")
    print(f"Local tables for '{dataset_name}' were not found. DROCAT queries need the")
    print("raw FAFB downloads converted into local parquet tables first; this is a")
    print("ONE-TIME preparation step (re-runs reuse the converted tables):")
    if key == "fafb":
        print()
        print("  Automatic alternative: save a FlyWire Codex API token")
        print("  (Settings > API Tokens) and use Settings > FAFB Data")
        print("  Downloads (levels: necessary data / synapses / skeleton")
        print("  bundle).")
    print()
    print(f"  1. Download the required files from:")
    print(f"     https://codex.flywire.ai/api/download?dataset={key}")
    print(f"     Required: {required}")
    print(f"  2. Save the downloads into:")
    print(f"     {download_dir}")
    print(f"  3. Run the one-time converter:")
    print(f"     python src/{converter}.py")
    print()
    if key == "fafb":
        print("  Optional FAFB extras (names/coordinates/neurons/cell_stats/")
        print("  consolidated_cell_types .csv.gz, synapse table, sk_lod1_783_healed.zip)")
        print("  can be saved into the same folder for enriched neurons, synapse")
        print("  markers and skeleton visualization.")
    print("=" * 70)


def _configured_token_from_config(project_root: Path, key: str) -> Optional[str]:
    """Read a token from config.json (primary) or config_local.json."""
    import json

    for filename in ("config.json", "config_local.json"):
        config_path = project_root / filename
        try:
            # utf-8-sig tolerates the UTF-8 BOM that Windows editors
            # prepend to saved JSON files.
            data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        section = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(section, dict):
            continue
        value = section.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value and not value.startswith("YOUR_"):
                return value
    return None


def _configured_cave_token(project_root: Path) -> Optional[str]:
    """Read a CAVE token without ever returning it to a caller-facing log.

    config.json (project config) is checked first; the environment is the
    fallback, matching the project's token manager.  A ``YOUR_*`` placeholder
    is treated as unconfigured.
    """

    config_value = _configured_token_from_config(project_root, "cave")
    if config_value:
        return config_value
    environment_value = os.environ.get("CAVE_TOKEN", "").strip()
    return (
        environment_value
        if environment_value and not environment_value.startswith("YOUR_")
        else None
    )


def _configured_codex_token(project_root: Path) -> Optional[str]:
    """Read the FlyWire Codex token (automatic FAFB downloads, lazy fetch)."""

    config_value = _configured_token_from_config(project_root, "flywire_codex")
    if config_value:
        return config_value
    environment_value = os.environ.get("FLYWIRE_CODEX_TOKEN", "").strip()
    return (
        environment_value
        if environment_value and not environment_value.startswith("YOUR_")
        else None
    )


def _first_existing(paths: list[Path]) -> Optional[Path]:
    """Return the first existing file or populated directory in *paths*."""

    cache_patterns = ("*.pkl", "*.pkl.zst", "*.swc", "*.swc.gz", "*.swc.zst")
    for path in paths:
        if path.is_file():
            return path
        if path.is_dir():
            try:
                for pattern in cache_patterns:
                    if next(path.rglob(pattern), None) is not None:
                        return path
            except OSError:
                continue
    return None


def local_fafb_skeleton_source(
    dataset: object,
    project_root: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate a usable local FAFB skeleton/geometry source.

    The converter's ZIP/parquet outputs are the preferred sources. Existing
    API and canonical skeleton pickle/SWC caches also count as local
    preparation: they allow a repeat run to proceed without making another
    CAVE request.
    """

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    folder = dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    cache_dir = root / "cache" / folder

    source = _first_existing([
        dataset_dir / f"{folder}_skeletons.zip",
        dataset_dir / "sk_lod1_783_healed.zip",
        dataset_dir / "downloads" / "sk_lod1_783_healed.zip",
        dataset_dir / f"{folder}_skeletons.parquet",
        cache_dir / "API_cache" / "skeletons",
        cache_dir / "skeletons",
        cache_dir / "meshes",
    ])
    if source is not None:
        return source

    # Be permissive about a converter-produced skeleton filename while still
    # requiring it to be a file in the selected dataset directory.
    try:
        for path in sorted(dataset_dir.glob("*skeleton*.zip")):
            if path.is_file():
                return path
        for path in sorted(dataset_dir.glob("*skeleton*.parquet")):
            if path.is_file():
                return path
    except OSError:
        pass
    return None


def _banc_public_source(root: Path, folder: str) -> str:
    """BANC skeletons always come from the public release bucket."""
    cache_dir = root / "cache" / folder / "skeletons"
    suffix = " (cached)" if _first_existing([cache_dir]) else ""
    return f"banc_public_gcs{suffix}"


def flywire_skeleton_readiness(
    dataset: object,
    project_root: Optional[str | Path] = None,
) -> dict:
    """Return non-secret readiness information for a morphology workflow."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    banc = is_banc_dataset(dataset)
    fafb = is_fafb_dataset(dataset)
    if banc:
        # BANC skeletons fetch on demand from the public release bucket —
        # no local bundle and no CAVE token is ever required.
        local_source = _banc_public_source(root, dataset_folder(dataset))
    elif fafb:
        local_source = local_fafb_skeleton_source(dataset, root)
    else:
        local_source = None
    # BANC is deliberately independent of CAVE/Codex. Do not surface an
    # ambient FAFB token as BANC readiness, because BANC never consumes it.
    cave_configured = False if banc else bool(_configured_cave_token(root))
    codex_configured = (
        False if banc else bool(_configured_codex_token(root)))
    # BANC is always ready (public bucket); FAFB needs local data, the
    # FlyWire Codex token (lazy per-neuron fetch), or a CAVE token;
    # NeuPrint datasets need neither.
    ready = ((not fafb) or local_source is not None or codex_configured
             or cave_configured)
    return {
        "dataset": str(dataset or ""),
        "is_banc": banc,
        "is_fafb": fafb,
        "local_skeletons": local_source is not None,
        "local_source": str(local_source) if local_source is not None else None,
        "cave_token": cave_configured,
        "flywire_codex": codex_configured,
        "ready": ready,
    }


def require_flywire_skeleton_access(
    dataset: object,
    project_root: Optional[str | Path] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Validate access for a local-release morphology workflow.

    BANC is always accepted: skeletons fetch on demand from the public
    release bucket and no CAVE token is involved.  FAFB is accepted when
    local skeleton data, the FlyWire Codex token (lazy per-neuron fetch),
    or a CAVE token is available.  When all are absent, the log includes
    the local preparation and token setup instructions before raising a
    clear exception.
    """

    status = flywire_skeleton_readiness(dataset, project_root)
    if not status["is_banc"] and not status["is_fafb"]:
        return status

    dataset_name = status["dataset"]
    if status["is_banc"]:
        log(
            f"[DROCAT][dataset-guard] BANC skeleton access ready: skeletons "
            f"fetch on demand from the public release bucket "
            f"(banc_public_gcs) and cache locally as .swc.zst. No CAVE "
            "token is involved."
        )
        return status

    if status["local_skeletons"]:
        source = status["local_source"] or "local FAFB skeleton cache"
        if status["cave_token"]:
            log(
                f"[DROCAT][dataset-guard] FAFB skeleton access ready: local "
                f"source found ({source}); CAVE API fallback is configured "
                "and will be attempted for missing or extrusion-affected "
                "skeletons."
            )
        else:
            log(
                f"[DROCAT][dataset-guard] FAFB skeleton access ready: local "
                f"source found ({source}). CAVE_TOKEN is not configured, so "
                "CAVE fallback is disabled."
            )
        return status

    if status["cave_token"]:
        log(
            "[DROCAT][dataset-guard] FAFB local skeletons were not found; "
            "CAVE_TOKEN is configured, so missing skeletons will be fetched "
            "through the CAVE API."
        )
        log(
            "Settings > FAFB Data Downloads (Skeleton bundle; needs the "
            "FlyWire Codex token) automates the bundle download below."
        )
        log(
            "For repeatable/offline runs, download sk_lod1_783_healed.zip "
            "from https://codex.flywire.ai/api/download?dataset=fafb, place "
            f"it in datasets/{dataset_name}/downloads/, then run "
            "python src/FAFB_file_converter.py (one-time preparation: the "
            "converter turns the raw downloads into the local tables used by "
            "all workflows; re-runs reuse them)."
        )
        return status

    if status.get("flywire_codex"):
        log(
            "[DROCAT][dataset-guard] FAFB skeleton access ready: no local "
            "skeleton data yet, but the FlyWire Codex token is configured "
            "— skeletons fetch lazily per neuron from the Codex server "
            "during visualization and cache into the local overlay "
            "(CAVE fallback is disabled)."
        )
        log(
            "For repeatable/offline runs, Settings > FAFB Data Downloads "
            "(level 3, Skeleton bundle) downloads the full local bundle."
        )
        return status

    message = (
        f"Skeleton workflow blocked for {dataset_name}: no local FAFB "
        "skeleton source was found and no FlyWire Codex or CAVE token "
        "is configured."
    )
    log(f"[DROCAT][dataset-guard] BLOCKED: {message}")
    log(
        "Prepare local FAFB skeletons: download "
        "sk_lod1_783_healed.zip from "
        "https://codex.flywire.ai/api/download?dataset=fafb, place it in "
        f"datasets/{dataset_name}/downloads/, then run "
        "python src/FAFB_file_converter.py (one-time preparation; the "
        "converter also builds the neuron/connection tables FAFB queries "
        "require)."
    )
    log(
        "Or configure CAVE_TOKEN in config.json (or the environment) "
        "using https://codex.flywire.ai/auth_token and rerun. The converted "
        "local neuron/connection tables are still required for FAFB queries."
    )
    log(
        "A FlyWire Codex API token (Settings > API Tokens) also unblocks "
        "skeletons: they fetch lazily per neuron from the Codex server "
        "during visualization, and Settings > FAFB Data Downloads offers "
        "the full bundle."
    )
    raise FlyWireSkeletonAccessError(message)
