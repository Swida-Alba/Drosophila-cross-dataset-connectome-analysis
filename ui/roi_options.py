"""Dataset-backed ROI option lists used by the skeleton visualization UI.

NeuPrint calls the non-overlapping ROI set ``primaryRois``.  DROCAT's local
dataset sidecars store that same set as ``roi_coverage.roi_list``.  The
available-ROI inventory is broader: it can contain sub-primary regions and,
for some datasets, super-level/container names as well.

The helpers in this module deliberately read local files only.  The UI should
not make a network request just to populate a select control; a visualization
run can still refresh the backend cache when a requested mesh is missing.

A fresh clone has no ``datasets/`` or ``cache/`` sidecars (both are gitignored
and the installer never downloads them), so the local read would otherwise
fall back to the hardcoded primary list and the "Include sub-primary ROIs"
checkbox would add nothing.  ``data/roi_catalogs.json`` ships a tracked
snapshot of the primary + available inventories (regenerate it with
``scripts/maintenance/build_roi_catalog.py`` on a machine that has the data).
It is the LAST fallback: live ``cache/``/``datasets/`` data always wins.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder
from src.flywire_ids import is_fafb_dataset


ROI_MODE_PRIMARY = "primary"
ROI_MODE_ALL = "all available"

_HEMISPHERE_SUFFIX = re.compile(r"\((?:L|R)\)$")

#: Tracked ROI snapshot shipped with the UI, resolved relative to this module
#: so it works from any working directory (and from a fresh clone).
BUNDLED_ROI_CATALOG = Path(__file__).resolve().parent / "data" / "roi_catalogs.json"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    """Return non-empty string values once, preserving their input order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        text = str(value).strip()
        if not text or text == "NotPrimary" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _candidate_folders(dataset: str) -> list[str]:
    """Return local dataset folders in the same order as mesh resolution."""

    dataset_text = str(dataset or "").strip()
    folders: list[str] = []
    # FAFB ROI meshes are sourced from male-cns.  Prefer v0.9 because
    # it is the established transformed-mesh cache, then try v1.0.  BANC has
    # its own public region_outlines catalog instead.
    if is_fafb_dataset(dataset_text):
        folders.extend(("male-cns_v0_9", "male-cns_v1_0"))
    if dataset_text:
        folder = dataset_to_folder(dataset_text)
        if folder and folder not in folders:
            folders.append(folder)
    return folders


def _metadata_paths(project_root: Path, folder: str) -> list[Path]:
    """Find metadata sidecars for one local dataset folder."""

    dataset_dir = project_root / "datasets" / folder
    if not dataset_dir.is_dir():
        return []

    paths: list[Path] = []
    preferred = dataset_dir / f"{folder}_metadata.json"
    if preferred.is_file():
        paths.append(preferred)
    for path in sorted(dataset_dir.glob("*metadata*.json")):
        if path not in paths:
            paths.append(path)
    return paths


def _read_metadata(project_root: Path, folders: Iterable[str]) -> list[dict[str, Any]]:
    """Read valid metadata objects from candidate folders."""

    result: list[dict[str, Any]] = []
    for folder in folders:
        for path in _metadata_paths(project_root, folder):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def _hierarchy_names(value: Any) -> list[str]:
    """Flatten names from a NeuPrint-style ``roiHierarchy`` object."""

    names: list[str] = []
    if isinstance(value, dict):
        name = value.get("name")
        if name is not None:
            names.append(str(name))
        children = value.get("children")
        if isinstance(children, list):
            for child in children:
                names.extend(_hierarchy_names(child))
    elif isinstance(value, list):
        for item in value:
            names.extend(_hierarchy_names(item))
    return names


def _metadata_primary_rois(metadata: dict[str, Any]) -> list[str]:
    """Extract the authoritative primary ROI list from a sidecar."""

    # ``primaryRois`` is the native NeuPrint metadata field.  Pulled DROCAT
    # sidecars project it into roi_coverage.roi_list, so support both forms.
    candidates = metadata.get("primaryRois")
    if not candidates:
        candidates = metadata.get("primary_rois")
    if not candidates:
        coverage = metadata.get("roi_coverage") or {}
        if isinstance(coverage, dict):
            candidates = coverage.get("roi_list")
    return _unique_strings(candidates if isinstance(candidates, list) else [])


def _metadata_available_rois(metadata: dict[str, Any]) -> list[str]:
    """Extract an all-ROI inventory when a metadata object contains one."""

    values: list[Any] = []
    for key in ("allRois", "all_rois", "availableRois", "available_rois"):
        candidate = metadata.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)

    roi_info = metadata.get("roiInfo")
    if isinstance(roi_info, dict):
        values.extend(roi_info.keys())

    hierarchy = metadata.get("roiHierarchy")
    if hierarchy is not None:
        values.extend(_hierarchy_names(hierarchy))

    coverage = metadata.get("roi_coverage") or {}
    if isinstance(coverage, dict):
        values.extend(coverage.get("roi_list") or [])
        values.extend((coverage.get("neuron_counts_per_roi") or {}).keys())

    return _unique_strings(values)


def _cached_available_rois(project_root: Path, folders: Iterable[str]) -> list[str]:
    """Load the first non-empty cached all-ROI inventory."""

    for folder in folders:
        path = project_root / "cache" / folder / "available_rois.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(value, list):
            values = _unique_strings(value)
            if values:
                return values
    return []


def _roi_count_file_rois(project_root: Path, folders: Iterable[str]) -> list[str]:
    """Read ROI names from local ROI-count tables as a cache fallback."""

    for folder in folders:
        dataset_dir = project_root / "datasets" / folder
        files = sorted(dataset_dir.glob("*_roi_count_df.parquet"))
        files += sorted(dataset_dir.glob("*_roi_count_df.csv"))
        for path in files:
            try:
                if path.suffix == ".parquet":
                    import polars as pl

                    values = (
                        pl.scan_parquet(str(path))
                        .select("roi")
                        .unique()
                        .collect()
                        .get_column("roi")
                        .to_list()
                    )
                else:
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        values = [row.get("roi") for row in csv.DictReader(handle)]
            except (OSError, ValueError, KeyError, ImportError):
                continue
            names = _unique_strings(values)
            if names:
                # Total order (casefold, then exact) so case-equal names like
                # AL(R)/aL(R) keep a stable order across runs.
                return sorted(names, key=lambda s: (s.casefold(), s))
    return []


def _local_roi_catalog(
    project_root: Path, folders: Iterable[str]
) -> dict[str, Any]:
    """Read the primary/available inventories from local sidecars only.

    ``has_live_inventory`` records whether a backend ``available_rois.json``
    cache or a ROI-count table supplied the available list; when it did, that
    live inventory is authoritative and the bundled snapshot must not merge
    into it.
    """

    metadata = _read_metadata(project_root, folders)
    primary: list[str] = []
    metadata_available: list[str] = []
    for item in metadata:
        if not primary:
            primary = _metadata_primary_rois(item)
        if not metadata_available:
            metadata_available = _metadata_available_rois(item)
        if primary and metadata_available:
            break

    cached_available = _cached_available_rois(project_root, folders)
    table_available: list[str] = []
    if not cached_available:
        table_available = _roi_count_file_rois(project_root, folders)

    available = _unique_strings(
        cached_available or table_available or metadata_available or primary
    )
    # A stale/incomplete cache must not hide a primary name loaded from the
    # authoritative sidecar.
    available = _unique_strings([*available, *primary])
    return {
        "primary": primary,
        "available": available,
        "has_live_inventory": bool(cached_available or table_available),
    }


def _bundled_roi_catalog(folders: Iterable[str]) -> dict[str, list[str]] | None:
    """Return the first shipped snapshot entry for the candidate folders."""

    try:
        data = json.loads(BUNDLED_ROI_CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    for folder in folders:
        entry = data.get(folder)
        if not isinstance(entry, dict):
            continue
        primary = _unique_strings(entry.get("primary") or [])
        available = _unique_strings(entry.get("available") or [])
        if primary or available:
            return {"primary": primary, "available": available or primary}
    return None


def load_roi_catalog(
    dataset: str,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, list[str]]:
    """Return ``primary`` and ``available`` ROI names for *dataset*.

    Primary names come from ``datasets/<dataset>/*metadata*`` first.  The
    available inventory prefers the backend's ``available_rois.json`` cache,
    then metadata/hierarchy fields, then local ROI-count tables.  When no
    local sidecar exists (a fresh install), the tracked ``roi_catalogs.json``
    snapshot supplies both lists so sub-primary ROIs are still selectable;
    a live cache/table always takes precedence over the snapshot.
    """

    root = Path(project_root)
    folders = _candidate_folders(dataset)
    local = _local_roi_catalog(root, folders)
    primary = local["primary"]
    available = local["available"]

    bundled = None
    if not primary or not local["has_live_inventory"]:
        # Read the shipped snapshot only when local data cannot answer the
        # request on its own (fresh install, or primary-only sidecars).
        bundled = _bundled_roi_catalog(folders)
    if bundled is not None:
        if not primary:
            primary = list(bundled["primary"])
        if not local["has_live_inventory"]:
            # No authoritative live inventory: restore the snapshot's broader
            # available list (it carries the sub-primary names) unless the
            # local metadata already supplies a wider one.
            bundled_available = bundled["available"]
            if len(bundled_available) > len(available):
                available = _unique_strings([*bundled_available, *primary])
        available = _unique_strings([*available, *primary])

    return {"primary": primary, "available": available}


def _without_hemisphere_suffix(name: str) -> str:
    return _HEMISPHERE_SUFFIX.sub("", name)


def _display_names(names: Iterable[str], include_lr: bool) -> list[str]:
    """Format primary-mode names with or without bilateral variants."""

    values = _unique_strings(names)
    if include_lr:
        return values
    return _unique_strings(_without_hemisphere_suffix(name) for name in values)


def roi_options_for_mode(
    dataset: str,
    mode: str = ROI_MODE_PRIMARY,
    *,
    include_lr: bool = False,
    include_subprimary: bool = False,
    fallback: Iterable[str] = (),
    project_root: Path | str = PROJECT_ROOT,
) -> list[str]:
    """Build the select options for the requested ROI mode.

    ``primary`` mode uses the metadata primary set and optionally adds
    non-primary entries.  ``all available`` intentionally returns exact names
    from the inventory, because collapsing L/R names would no longer be an
    all-available list.
    """

    catalog = load_roi_catalog(dataset, project_root=project_root)
    return roi_options_from_catalog(
        catalog,
        mode,
        include_lr=include_lr,
        include_subprimary=include_subprimary,
        fallback=fallback,
    )


def roi_options_from_catalog(
    catalog: dict[str, list[str]],
    mode: str = ROI_MODE_PRIMARY,
    *,
    include_lr: bool = False,
    include_subprimary: bool = False,
    fallback: Iterable[str] = (),
) -> list[str]:
    """Build options from an already loaded ROI catalog."""

    primary = catalog["primary"] or _unique_strings(fallback)
    available = catalog["available"] or primary

    if str(mode or "").strip().lower() in {"all", ROI_MODE_ALL}:
        return _unique_strings(available)

    values = list(primary)
    if include_subprimary:
        primary_set = set(primary)
        values.extend(name for name in available if name not in primary_set)
    return _display_names(values, include_lr)


__all__ = [
    "ROI_MODE_ALL",
    "ROI_MODE_PRIMARY",
    "load_roi_catalog",
    "roi_options_from_catalog",
    "roi_options_for_mode",
]
