"""Storage inventory and path-scoped deletion (Settings → Storage).

Implements ``_plan/plan-settings-storage-utility.md``: scan the cache
tree and the exported run folders, and delete only through explicit,
user-driven actions whose every path is validated here first.

Module rules:

* **No UI imports** — the UI layer reads its own config (``cache_only``,
  pull state, run manager) and passes the resulting flags in.
* Deletion is **path-scoped**: every path is checked against the
  protected list and the class/table classifiers before anything is
  touched. A path that is not a known class file/dir, a NeuronBridge
  cache entry, a neuron-index entry, or a classified run folder is
  REFUSED — a UI bug must never be able to delete user data or a
  shared root through this module.
* The incoming-connection state file is deleted together with its
  parquet (never left behind, never offered alone).
* Every removal is audited: run-folder prunes append to the folder's
  ``cleanup_audit.json`` (the output-policy contract); whole-folder
  deletes and cache clears append to ``cache/storage_audit.json``.
* Nothing runs by itself: no timers, no startup sweeps, no age rules.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "cache"
INDEX_ROOT = PROJECT_ROOT / "neuron_indexes"
STORAGE_AUDIT_FILENAME = "storage_audit.json"

try:  # src on sys.path (app runtime) or laid bare (unit tests)
    from utils.naming_utils import is_run_folder_name, run_folder_timestamp
except ImportError:  # pragma: no cover - repo-root import layout
    from src.utils.naming_utils import (  # type: ignore
        is_run_folder_name,
        run_folder_timestamp,
    )

try:  # reuses the output-policy prune passes and audit contract
    from neuronbridge_output_policy import (
        AUDIT_FILENAME as RUN_AUDIT_FILENAME,
        prune_colabel_run,
        prune_find_lines_run,
        prune_find_neurons_run,
    )
except ImportError:  # pragma: no cover - repo-root import layout
    from src.neuronbridge_output_policy import (  # type: ignore
        AUDIT_FILENAME as RUN_AUDIT_FILENAME,
        prune_colabel_run,
        prune_find_lines_run,
        prune_find_neurons_run,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_bytes(num: float) -> str:
    """Human-readable byte size (used by the storage tables)."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


# ---------------------------------------------------------------------------
# Protected paths — never offered, never accepted by delete_paths.
# ---------------------------------------------------------------------------


def protected_paths(cache_root: Optional[Path] = None) -> List[Path]:
    """Every path the storage utility must never delete.

    ``cache/user_mappings/`` holds user-authored LabelMapper presets with
    no rebuild path (created lazily on first save, so its absence proves
    nothing). ``storage_audit.json`` is this module's own audit — a sweep
    must never eat its own record.
    """
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    prot = [
        cache_root / "dataset_availability.json",
        cache_root / "neuronbridge" / "coverage_snapshot.json",
        cache_root / "user_mappings",
        cache_root / STORAGE_AUDIT_FILENAME,
    ]
    if cache_root.is_dir():
        for entry in sorted(cache_root.iterdir()):
            if entry.is_dir() and (entry / "available_rois.json").is_file():
                prot.append(entry / "available_rois.json")
    return prot


def is_protected(path: os.PathLike | str,
                 cache_root: Optional[Path] = None) -> bool:
    path = Path(path)
    for prot in protected_paths(cache_root):
        try:
            prot_resolved = prot.resolve()
            resolved = path.resolve()
        except OSError:  # pragma: no cover - unreadable path
            continue
        if resolved == prot_resolved or prot_resolved in resolved.parents:
            return True
    return False


# ---------------------------------------------------------------------------
# Inventory items
# ---------------------------------------------------------------------------


@dataclass
class CacheItem:
    """One deletable cache unit (class × dataset)."""

    key: str
    cls: str                  # machine class id ('connections', ...)
    label: str                # display label
    dataset: str              # dataset folder name ('' = app-level)
    paths: List[Path] = field(default_factory=list)  # exact deletable set
    size_bytes: int = 0
    last_used: float = 0.0    # max member mtime (0.0 when nothing exists)
    rebuild_note: str = ""
    warnings: List[str] = field(default_factory=list)
    default_checked: bool = False
    deleter: str = "paths"    # 'paths' | 'nb:id_to_lines' | 'nb:image_cache'
    open_path: Optional[str] = None  # folder the UI 'open dir' shows


@dataclass
class RunFolderItem:
    """One classified per-run output folder."""

    path: Path
    name: str
    tool: str                 # matched registry key ('' = unregistered)
    timestamp: str            # 'YYYYMMDD_HHMMSS' ('' = no stamp in name)
    size_bytes: int = 0
    source_bytes: int = -1    # -1 = unregistered tool: no class split
    deliverable_bytes: int = -1
    registered: bool = False
    compact_opted: bool = False

    @property
    def open_path(self) -> str:
        return str(self.path)


# Per-dataset cache classes. Each entry maps a class id to member names
# relative to ``cache/<dataset>/``; 'dir:' entries are whole trees, plain
# entries are files. The incoming pair is ONE class so the completion
# state file can never be offered (or left) alone.
_CLASS_MEMBERS = {
    "connections": [
        "connections.parquet",
        "connections.parquet.src",
        "neuron_index_state.parquet",
        "dir:_batch_files",
    ],
    "incoming": [
        "incoming_connections.parquet",
        "incoming_complete.json",
    ],
    "profiles": [
        "connectivity_profiles.parquet",
        "dir:_profile_batch_files",
    ],
    "skeletons": ["dir:skeletons"],
    "synapses": ["dir:synapses"],
    "meshes": ["dir:meshes", "dir:meshes_transformed"],
    "find_similar": ["dir:find_similar"],
    "morphology": ["dir:morphology"],
    "derived_misc": [
        "banc_id_crosswalk.parquet",
        "extrusion_check_results.parquet",
        "region_name_map.json",
        "dir:API_cache",
    ],
}

_CLASS_LABELS = {
    "connections": "Connection cache",
    "incoming": "Incoming connections (+ completion state)",
    "profiles": "Connectivity profiles",
    "skeletons": "Skeleton cache",
    "synapses": "Synapse cache",
    "meshes": "Meshes (+ transformed)",
    "find_similar": "Find-similar vectors",
    "morphology": "Morphology (dotprops)",
    "derived_misc": "Derived misc (crosswalk, extrusion memo)",
    "nb_match_tables": "NeuronBridge match tables",
    "nb_image_cache": "NeuronBridge image cache",
    "type_mapper_snapshot": "Type-mapper snapshot",
    "neuron_index": "Neuron index",
}

_REBUILD_NOTES = {
    "neuprint": {
        "connections": "NeuPrint server refetch — expensive",
        "incoming": "NeuPrint server refetch — expensive",
    },
    "fafb": {
        "connections": "Offline re-derivation from the local merged table "
                       "(CAVE network only when local tables are absent)",
        "incoming": "Offline re-derivation from the local merged table",
    },
    "banc": {
        "connections": "Automatic offline rebuild from the local merged "
                       "table (runs on next use)",
        "incoming": "Rebuilt alongside the offline connection rebuild",
    },
}


def _dataset_family(dataset_dir_name: str) -> str:
    lowered = dataset_dir_name.lower()
    if lowered.startswith("banc"):
        return "banc"
    if lowered.startswith("flywire_fafb") or lowered.startswith("fafb"):
        return "fafb"
    return "neuprint"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for member in path.rglob("*"):
            if member.is_file():
                try:
                    total += member.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _paths_stats(paths: List[Path]) -> Tuple[List[Path], int, float]:
    """(existing paths, total bytes, max mtime) for a member set."""
    existing: List[Path] = []
    total = 0
    newest = 0.0
    for path in paths:
        try:
            if path.is_dir():
                size = _dir_size(path)
                newest = max(newest, path.stat().st_mtime)
            elif path.is_file():
                size = path.stat().st_size
                newest = max(newest, path.stat().st_mtime)
            else:
                continue
        except OSError:
            continue
        existing.append(path)
        total += size
    return existing, total, newest


def _git_tracked_files(project_root: Optional[Path] = None) -> set:
    """Repo-relative paths of git-tracked files under neuron_indexes
    (empty when git is unavailable, e.g. in sandboxed test trees)."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--", "neuron_indexes"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def scan_caches(
    cache_root: Optional[Path] = None,
    index_root: Optional[Path] = None,
) -> List[CacheItem]:
    """Inventory the cache tree as (class × dataset) rows."""
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    index_root = Path(index_root) if index_root else INDEX_ROOT
    items: List[CacheItem] = []
    if cache_root.is_dir():
        cache_entries = sorted(
            entry for entry in cache_root.iterdir() if entry.is_dir())
    else:
        cache_entries = []

    tracked = _git_tracked_files()

    for entry in cache_entries:
        if entry.name == "neuronbridge":
            items.extend(_scan_neuronbridge(entry))
            continue
        for cls, members in _CLASS_MEMBERS.items():
            member_paths = [
                entry / name.split("dir:", 1)[1]
                if name.startswith("dir:") else entry / name
                for name in members
            ]
            existing, total, newest = _paths_stats(member_paths)
            if not existing or total == 0:
                continue
            family = _dataset_family(entry.name)
            note = _REBUILD_NOTES.get(family, {}).get(
                cls, "Locally rebuildable / re-derivable")
            items.append(CacheItem(
                key=f"{cls}:{entry.name}",
                cls=cls,
                label=_CLASS_LABELS.get(cls, cls),
                dataset=entry.name,
                paths=existing,
                size_bytes=total,
                last_used=newest,
                rebuild_note=note,
                open_path=str(entry),
            ))

    if index_root.is_dir():
        for entry in sorted(index_root.iterdir()):
            rel = f"neuron_indexes/{entry.name}"
            if entry.is_dir():
                existing, total, newest = _paths_stats([entry])
                if not existing:
                    continue
                warnings = []
                rel_members = [
                    f"neuron_indexes/{p.relative_to(index_root)}"
                    for p in entry.rglob("*") if p.is_file()
                ]
                if any(rel_member in tracked
                       for rel_member in rel_members):
                    warnings.append(
                        "git-tracked (ships with the repo) — deletion "
                        "shows in git status")
                warnings.append(
                    "only rebuildable offline when the dataset table is "
                    "pulled; otherwise the bundled index is the only "
                    "local presence")
                items.append(CacheItem(
                    key=f"neuron_index:{entry.name}",
                    cls="neuron_index",
                    label=f"{_CLASS_LABELS['neuron_index']} ({entry.name})",
                    dataset=entry.name,
                    paths=[entry],
                    size_bytes=total,
                    last_used=newest,
                    rebuild_note="Offline rebuild from the local dataset "
                                 "table (fast) — needs a metadata pull "
                                 "for not-pulled datasets",
                    warnings=warnings,
                    open_path=str(entry),
                ))
            elif entry.is_file() and entry.name.startswith(
                    "type_mapper_snapshot_v"):
                _, total, newest = _paths_stats([entry])
                items.append(CacheItem(
                    key=f"type_mapper_snapshot:{entry.name}",
                    cls="type_mapper_snapshot",
                    label=f"{_CLASS_LABELS['type_mapper_snapshot']} "
                          f"({entry.name})",
                    dataset="",
                    paths=[entry],
                    size_bytes=total,
                    last_used=newest,
                    rebuild_note="Rebuilt on the next cross-dataset type "
                                 "mapping load",
                    open_path=str(index_root),
                ))
        manifest = index_root / "manifest.json"
        if manifest.is_file():
            warnings = (["git-tracked (ships with the repo)"]
                        if "neuron_indexes/manifest.json" in tracked else [])
            items.append(CacheItem(
                key="neuron_index:manifest",
                cls="neuron_index",
                label="Neuron index manifest",
                dataset="",
                paths=[manifest],
                size_bytes=manifest.stat().st_size,
                last_used=manifest.stat().st_mtime,
                rebuild_note="Rewritten by the seed-index builder",
                warnings=warnings,
                open_path=str(index_root),
            ))
    return items


def _scan_neuronbridge(entry: Path) -> List[CacheItem]:
    """NeuronBridge match tables + image cache (two rows, dataset '')."""
    parquet_root = entry / "parquet"
    id_paths: List[Path] = []
    image_paths: List[Path] = []
    if parquet_root.is_dir():
        for version_dir in sorted(
                p for p in parquet_root.iterdir() if p.is_dir()):
            id_dir = version_dir / "id_to_lines"
            if id_dir.is_dir():
                id_paths.extend(sorted(id_dir.glob("*.parquet")))
            for sub in sorted(
                    p for p in version_dir.iterdir() if p.is_dir()):
                img_dir = sub / "image_cache"
                if img_dir.is_dir():
                    image_paths.extend(sorted(img_dir.glob("*.parquet")))
                mapping = sub / "line_image_mapping.json"
                if mapping.is_file():
                    image_paths.append(mapping)
    # Legacy layouts beside the parquet/ tree.
    id_paths.extend(sorted(entry.glob("id_to_lines_*.csv")))
    id_paths.extend(sorted(entry.glob("id_to_lines_*.parquet")))
    legacy_img = entry / "image_cache"
    if legacy_img.is_dir():
        image_paths.extend(sorted(legacy_img.glob("*.csv")))
        mapping = legacy_img / "line_image_mapping.json"
        if mapping.is_file():
            image_paths.append(mapping)

    items: List[CacheItem] = []
    for cls, paths, note, deleter in (
        ("nb_match_tables", id_paths,
         "Re-query against the NeuronBridge API", "nb:id_to_lines"),
        ("nb_image_cache", image_paths,
         "Re-download images from NeuronBridge/FlyLight", "nb:image_cache"),
    ):
        existing, total, newest = _paths_stats(paths)
        if not existing:
            continue
        items.append(CacheItem(
            key=cls,
            cls=cls,
            label=_CLASS_LABELS[cls],
            dataset="",
            paths=existing,
            size_bytes=total,
            last_used=newest,
            rebuild_note=note,
            deleter=deleter,
            open_path=str(entry),
        ))
    return items


# ---------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------

# Per-tool source-data registry. Source members are the prunable class;
# deliverables are what a Compact prune keeps; anything else is 'other'.
# Tools without an entry are Delete-folder-only. Patterns are fnmatch on
# the folder-relative posix path (fnmatch '*' crosses '/').
_RUN_TOOL_REGISTRY = {
    "NB-find-lines": {
        "prune": "find_lines",
        "source_files": ["*_lines.csv"],
        "source_dirs": ["images", "chip_*/images"],
        "deliverable_files": [
            "line_summary.csv", "chip_*/line_summary.csv",
            "images_summary.pdf", "chip_*/images_summary.pdf",
            "images_summary.pptx", "chip_*/images_summary.pptx",
            "expansion_*.csv", "expansion_*.json",
            "chip_*/expansion_*.csv",
        ],
    },
    "NB-find-neurons": {
        "prune": "find_neurons",
        "source_files": ["all_neurons.csv", "*_neurons.csv"],
        "source_dirs": [],
        "deliverable_files": [
            "*_types.csv", "*_type_mapped.csv",
        ],
    },
    "NB-colabeling": {
        "prune": "colabel",
        "source_files": ["distribution_data_by_neuron.csv"],
        "source_dirs": ["line_labeled_neurons"],
        "deliverable_files": [
            "colabeling_matrix_*.csv", "colabeling_report.html",
        ],
    },
}

_COMMON_DELIVERABLE_FILES = [
    "parameters.json", "parameters.txt", "README.txt",
    "_UserGuide_please_read_me*", "user_warning_notes.txt",
    "cleanup_audit.json", "run_guide*",
]


def _classify_run_member(tool: str, rel_posix: str) -> str:
    """'source' | 'deliverable' | 'other' for one FILE of a run folder."""
    reg = _RUN_TOOL_REGISTRY.get(tool)
    if reg is None:
        return "other"
    for pattern in reg["source_files"]:
        if fnmatch.fnmatch(rel_posix, pattern):
            return "source"
    for pattern in reg["source_dirs"]:
        if fnmatch.fnmatch(rel_posix, pattern) or \
                fnmatch.fnmatch(rel_posix, f"{pattern}/*"):
            return "source"
    for pattern in list(reg.get("deliverable_files", [])) + \
            _COMMON_DELIVERABLE_FILES:
        if fnmatch.fnmatch(rel_posix, pattern):
            return "deliverable"
    return "other"


def _tool_for_run(name: str) -> str:
    """Longest registered prefix matching the run-folder name."""
    best = ""
    for tool in _RUN_TOOL_REGISTRY:
        if name.startswith(tool) and len(tool) > len(best):
            best = tool
    return best


def scan_run_folders(
    output_roots: Iterable[os.PathLike | str],
    *,
    max_depth: int = 3,
) -> List[RunFolderItem]:
    """Inventory timestamped tool run folders under the given roots.

    A folder qualifies only with a known prefix AND the embedded
    ``_YYYYMMDD_HHMMSS`` timestamp; prefix-but-no-timestamp directories
    (shared roots such as ``morph_cross_dataset/``) and unknown folders
    are descended into (bounded), never listed themselves. Matched run
    folders are not descended into, so a nested scene folder (e.g.
    ``plot-3d_*`` inside a morphology run) is never double-listed.
    """
    found: dict = {}
    for root in output_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, _filenames in os.walk(root):
            current = Path(dirpath)
            if len(current.parts) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            descend: List[str] = []
            for child in dirnames:
                if is_run_folder_name(child):
                    found.setdefault(child, current / child)
                else:
                    descend.append(child)
            dirnames[:] = descend
    return [_describe_run_folder(path, name)
            for name, path in sorted(found.items())]


def _describe_run_folder(path: Path, name: str) -> RunFolderItem:
    tool = _tool_for_run(name)
    item = RunFolderItem(
        path=path,
        name=name,
        tool=tool,
        timestamp=run_folder_timestamp(name),
        registered=bool(tool),
    )
    source = deliverable = other = 0
    try:
        for member in path.rglob("*"):
            try:
                if member.is_dir():
                    continue
                size = member.stat().st_size
            except OSError:
                continue
            rel = member.relative_to(path).as_posix()
            kind = _classify_run_member(tool, rel)
            if kind == "source":
                source += size
            elif kind == "deliverable":
                deliverable += size
            else:
                other += size
    except OSError:
        pass
    item.size_bytes = source + deliverable + other
    if item.registered:
        item.source_bytes = source
        item.deliverable_bytes = deliverable
    item.compact_opted = _run_folder_was_compact(path)
    return item


def _run_folder_was_compact(folder: Path) -> bool:
    """Compact opt-in marker: cleanup audit or parameters.json flag."""
    try:
        if (folder / RUN_AUDIT_FILENAME).is_file():
            return True
        payload = json.loads(
            (folder / "parameters.json").read_text(encoding="utf-8"))
        params = payload.get("function_params") or {}
        return params.get("keep_per_match_csv") is False
    except (OSError, ValueError):
        return False


def prune_run_folder(
    path: os.PathLike | str,
    *,
    cache_root: Optional[Path] = None,
) -> dict:
    """Run the per-tool Compact prune on one run folder (user-driven).

    Runs regardless of the folder's original Full/Compact mode: the user
    is explicitly choosing this now. Unregistered tools and non-run
    folders are refused.
    """
    folder = Path(path)
    tool = _tool_for_run(folder.name)
    if not is_run_folder_name(folder.name):
        raise ValueError(f"refusing to prune a non-run folder: {folder}")
    if not tool:
        raise ValueError(
            f"refusing to prune an unregistered run folder: {folder}")
    if is_protected(folder, cache_root):
        raise ValueError(f"refusing to prune a protected path: {folder}")
    if tool == "NB-find-lines":
        return prune_find_lines_run(
            str(folder), keep_per_match_csv=False,
            cleanup_source_images=True)
    if tool == "NB-find-neurons":
        return prune_find_neurons_run(str(folder), keep_per_match_csv=False)
    if tool == "NB-colabeling":
        return prune_colabel_run(str(folder), keep_per_match_csv=False)
    raise ValueError(f"no prune registry entry for tool prefix: {tool}")


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

_CLASS_DIR_NAMES = {
    name.split("dir:", 1)[1]
    for members in _CLASS_MEMBERS.values()
    for name in members if name.startswith("dir:")
}
_CLASS_FILE_NAMES = {
    name for members in _CLASS_MEMBERS.values()
    for name in members if not name.startswith("dir:")
}


def _remove_file(path: Path) -> Optional[dict]:
    try:
        size = path.stat().st_size
        path.unlink()
        return {"path": str(path), "bytes": size, "kind": "file"}
    except OSError:
        return None


def _remove_tree(path: Path) -> Optional[dict]:
    try:
        size = _dir_size(path)
        shutil.rmtree(path)
        return {"path": str(path), "bytes": size, "kind": "directory"}
    except OSError:
        return None


def _within(path: Path, root: Path) -> bool:
    try:
        resolved_root = root.resolve()
        return path == resolved_root or resolved_root in path.parents
    except OSError:
        return False


def _nb_cache_kind(path: Path, cache_root: Path) -> Optional[str]:
    """Map a path under cache/neuronbridge to its clear_cache type."""
    try:
        rel = path.relative_to(cache_root / "neuronbridge")
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "parquet" and parts[2] == "id_to_lines":
        return "id_to_lines"
    if "image_cache" in parts:
        return "image_cache"
    if len(parts) >= 2 and parts[0] == "parquet" and \
            parts[-1] == "line_image_mapping.json":
        return "image_cache"
    if rel.name.startswith("id_to_lines_"):
        return "id_to_lines"
    return None


def _remove_index_path(resolved: Path, index_root: Path) -> Optional[dict]:
    """Validate + remove one neuron_indexes entry (not the root itself)."""
    if resolved == index_root.resolve():
        return None
    try:
        rel_parent = resolved.parent.relative_to(index_root)
    except ValueError:
        return None
    at_root = len(rel_parent.parts) == 0
    if resolved.is_dir() and at_root:
        return _remove_tree(resolved)  # neuron_indexes/<dataset>/
    if resolved.is_file() and at_root:
        if resolved.name.startswith("type_mapper_snapshot_v") and \
                resolved.suffix == ".pkl":
            return _remove_file(resolved)
        if resolved.name == "manifest.json":
            return _remove_file(resolved)
        return None
    if resolved.is_file() and len(rel_parent.parts) == 1 and \
            resolved.suffix == ".parquet":
        return _remove_file(resolved)  # per-dataset index/search member
    return None


def _default_nb_finder_factory():
    from neuronbridge_finder import NeuronBridgeFinder
    return NeuronBridgeFinder(use_cache=True, verbose=False)


def delete_paths(
    paths: Iterable[os.PathLike | str],
    *,
    cache_root: Optional[Path] = None,
    index_root: Optional[Path] = None,
    audit_into: Optional[os.PathLike | str] = None,
    nb_finder_factory: Optional[Callable[[], object]] = None,
) -> dict:
    """Delete validated paths; everything else is refused.

    Every path must be one of: a known class file/dir under a dataset
    cache folder, a NeuronBridge cache file, a neuron-index entry, or a
    classified run folder. Protected paths are refused unconditionally.
    Removals merge into ``audit_into`` (default
    ``cache/storage_audit.json``); run-folder removals are recorded
    there too, since their own folder no longer exists.

    Returns ``{removed, bytes_reclaimed, refused, nb_cleared, audit}``.
    """
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    index_root = Path(index_root) if index_root else INDEX_ROOT
    audit_target = Path(audit_into) if audit_into else (
        cache_root / STORAGE_AUDIT_FILENAME)
    nb_factory = nb_finder_factory or _default_nb_finder_factory

    removed: List[dict] = []
    refused: List[dict] = []
    nb_types: set = set()
    nb_expected: dict = {}

    # Pairing guard: deleting the incoming parquet implies its completion
    # state file; the state file alone is only deletable once orphaned.
    requested: List[Path] = [Path(p) for p in paths]
    for path in list(requested):
        if path.name == "incoming_connections.parquet":
            state = path.parent / "incoming_complete.json"
            if state not in requested and state.exists():
                requested.append(state)

    for path in requested:
        try:
            resolved = path.resolve()
        except OSError:
            refused.append({"path": str(path), "reason": "unresolvable"})
            continue
        if is_protected(resolved, cache_root):
            refused.append({"path": str(path), "reason": "protected path"})
            continue

        in_cache = _within(resolved, cache_root)
        in_index = _within(resolved, index_root)

        if not in_cache and not in_index:
            # Only classified run folders may live outside the roots.
            if resolved.is_dir() and is_run_folder_name(resolved.name):
                entry = _remove_tree(resolved)
                if entry:
                    removed.append(entry)
                else:
                    refused.append(
                        {"path": str(path), "reason": "deletion failed"})
                continue
            refused.append(
                {"path": str(path), "reason": "outside the storage roots"})
            continue

        if in_index:
            entry = _remove_index_path(resolved, index_root)
            if entry:
                removed.append(entry)
            else:
                refused.append({
                    "path": str(path),
                    "reason": "not a classifiable neuron-index entry"})
            continue

        # Under the cache root.
        if _within(resolved, cache_root / "neuronbridge"):
            kind = _nb_cache_kind(resolved, cache_root)
            if kind is None or not resolved.is_file():
                refused.append({
                    "path": str(path),
                    "reason": "not a classifiable NeuronBridge cache file"})
                continue
            nb_types.add(kind)
            try:
                size = resolved.stat().st_size
            except OSError:
                size = 0
            nb_expected.setdefault(kind, []).append(
                {"path": str(resolved), "bytes": size, "kind": "file"})
            continue

        parent = resolved.parent
        if resolved.is_dir() and resolved.name in _CLASS_DIR_NAMES and \
                parent != cache_root:
            entry = _remove_tree(resolved)
            if entry:
                removed.append(entry)
            else:
                refused.append(
                    {"path": str(path), "reason": "deletion failed"})
            continue
        if resolved.is_file() and resolved.name in _CLASS_FILE_NAMES:
            if resolved.name == "incoming_complete.json":
                if (parent / "incoming_connections.parquet").exists():
                    # The completion state must never be deleted alone
                    # while its parquet exists: complete posts would
                    # never refetch → silent data loss.
                    refused.append({
                        "path": str(path),
                        "reason": "incoming completion state is paired "
                                  "with its parquet — delete the "
                                  "'incoming' class instead"})
                    continue
            entry = _remove_file(resolved)
            if entry:
                removed.append(entry)
            else:
                refused.append(
                    {"path": str(path), "reason": "deletion failed"})
            continue
        refused.append(
            {"path": str(path), "reason": "not a classifiable cache path"})

    # NeuronBridge classes go through the finder's own clearer (one call
    # per type) so its cache index stays coherent with the deletions.
    for kind in sorted(nb_types):
        try:
            finder = nb_factory()
            finder.clear_cache(kind)
        except Exception as exc:
            refused.append({
                "path": f"neuronbridge:{kind}",
                "reason": f"clear_cache failed: {exc}"})
            continue
        removed.extend(entry for entry in nb_expected.get(kind, [])
                       if not os.path.exists(entry["path"]))

    bytes_reclaimed = sum(int(entry["bytes"]) for entry in removed)
    audit = _merge_storage_audit(audit_target, removed, bytes_reclaimed)
    return {
        "removed": removed,
        "bytes_reclaimed": bytes_reclaimed,
        "refused": refused,
        "nb_cleared": sorted(nb_types),
        "audit": audit,
    }


def _merge_storage_audit(audit_path: Path, removed: List[dict],
                         bytes_reclaimed: int) -> dict:
    """Append a pass to the storage-level audit file (output-policy
    contract, file form). Empty passes leave no file behind."""
    existing: dict = {}
    if audit_path.is_file():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = payload
        except (OSError, ValueError):
            existing = {}
    prior_removed = list(existing.get("removed") or [])
    prior_bytes = int(existing.get("bytes_reclaimed") or 0)
    if not removed and not prior_removed and not audit_path.exists():
        return {"removed": [], "bytes_reclaimed": 0, "total_removed": 0,
                "passes": 0}
    audit = {
        "removed": prior_removed + removed,
        "bytes_reclaimed": prior_bytes + bytes_reclaimed,
        "passes": int(existing.get("passes") or 0) + 1,
        "updated_at": _now(),
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_path.with_name(
            f".{audit_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, audit_path)
    except OSError:
        pass
    return {
        "removed": removed,
        "bytes_reclaimed_this_pass": bytes_reclaimed,
        "bytes_reclaimed": audit["bytes_reclaimed"],
        "total_removed": len(audit["removed"]),
        "passes": audit["passes"],
    }
