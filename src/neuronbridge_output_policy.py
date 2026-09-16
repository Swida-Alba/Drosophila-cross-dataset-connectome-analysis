"""Output-detail policy for the NeuronBridge tools (Full / Compact).

Implements `_plan/plan-nb-find-lines-output-modes.md` §3: one shared,
idempotent prune pass that removes the bodyId-level "source data only"
exports after a run's summaries/report exist, and the downloaded images
after the integrated PDF/PPTX was written.

Contract:

* **Full (default)** — `keep_per_match_csv=True, cleanup_source_images=
  False`: nothing is ever removed.
* **Compact** — both flags flipped: per-match tables are deleted once their
  summary exists, `images/` once the PDF/PPTX artifact exists on disk.
  `_generate_image_summaries` swallows failures, so "summary succeeded" is
  detected as the artifact existing — never as an exception check.
* The pass is idempotent (missing files are skipped), so an orchestrator
  wrapping finder calls can re-run it safely.
* Every removal is audited: `cleanup_audit.json` in the run folder lists the
  removed paths and reclaimed bytes (merged across passes).

Under the default-off NeuronBridge match cache (D1 of the plan), a deleted
per-match table regenerates only by re-running the query against the
NeuronBridge API — the doc text next to the UI knob says exactly that.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AUDIT_FILENAME = "cleanup_audit.json"

FIND_LINES_SUMMARY = "line_summary.csv"
FIND_LINES_IMAGE_SUMMARIES = ("images_summary.pdf", "images_summary.pptx")
RUN_FOLDER_PREFIX = "NB-find-lines"  # covers plain and -expanded runs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _has_content(path: Path) -> bool:
    """True when the artifact exists AND is non-empty.

    A zero-byte summary or PDF must never license deletion — it usually
    means a writer crashed mid-write, and Compact would then destroy the
    source data for nothing.
    """
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _merge_audit(output_path: str, removed: List[Dict[str, Any]],
                 bytes_reclaimed: int,
                 force_write: bool = False) -> Dict[str, Any]:
    """Merge this pass into the run's cleanup audit file and return the
    combined audit.  ``removed`` in the RETURN is this pass's removals only;
    the persisted file accumulates across passes.  Compact passes force a
    write even when nothing was removed: the audit is the marker later
    retention sweeps use to classify the run — without it, an imageless
    Compact run would be "unknown mode" and never reclaimed."""
    path = Path(output_path) / AUDIT_FILENAME
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing = payload
        except (OSError, ValueError):
            existing = {}
    prior_removed = list(existing.get("removed") or [])
    prior_bytes = int(existing.get("bytes_reclaimed") or 0)
    # A Full pass (nothing removed, no prior audit) leaves no file behind —
    # an empty audit on a keep-everything run would just be noise.
    if not removed and not prior_removed and not force_write \
            and not path.exists():
        return {
            "removed": [],
            "bytes_reclaimed_this_pass": 0,
            "bytes_reclaimed": 0,
            "total_removed": 0,
            "passes": 0,
        }
    audit = {
        "removed": prior_removed + removed,
        "bytes_reclaimed": prior_bytes + bytes_reclaimed,
        "passes": int(existing.get("passes") or 0) + 1,
        "updated_at": _now(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass
    return {
        "removed": removed,
        "bytes_reclaimed_this_pass": bytes_reclaimed,
        "bytes_reclaimed": audit["bytes_reclaimed"],
        "total_removed": len(audit["removed"]),
        "passes": audit["passes"],
    }


def _glob_files(output_path: str, patterns: Iterable[str]) -> List[Path]:
    """Top-level (non-recursive) glob matches, sorted for stable audits."""
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(
            Path(p) for p in sorted(glob.glob(os.path.join(output_path, pattern)))
        )
    return sorted(set(matches))


def _remove_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        size = _file_size(path)
        path.unlink()
        return {"path": str(path), "bytes": size, "kind": "file"}
    except OSError:
        return None


def _remove_tree(path: Path) -> Optional[Dict[str, Any]]:
    try:
        size = sum(_file_size(p) for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path)
        return {"path": str(path), "bytes": size, "kind": "directory"}
    except OSError:
        return None


def _apply(
    output_path: str,
    *,
    keep_per_match_csv: bool,
    cleanup_source_images: bool,
    match_table_patterns: Iterable[str],
    images_dirs: Iterable[str],
    summary_artifacts: Iterable[str],
    match_condition_met: bool,
) -> Dict[str, Any]:
    """Shared prune pass; see module docstring for the contract."""
    output_path = str(output_path)
    removed: List[Dict[str, Any]] = []

    if not keep_per_match_csv and match_condition_met:
        for path in _glob_files(output_path, match_table_patterns):
            entry = _remove_file(path)
            if entry:
                removed.append(entry)

    if cleanup_source_images:
        artifact_exists = any(
            (Path(output_path) / artifact).exists()
            for artifact in summary_artifacts
        )
        if artifact_exists:
            for images_dir in images_dirs:
                dir_path = Path(output_path) / images_dir
                if dir_path.is_dir():
                    entry = _remove_tree(dir_path)
                    if entry:
                        removed.append(entry)

    bytes_reclaimed = sum(int(entry["bytes"]) for entry in removed)
    return _merge_audit(output_path, removed, bytes_reclaimed,
                        force_write=not keep_per_match_csv)


def prune_find_lines_run(
    output_path: str,
    keep_per_match_csv: bool = True,
    cleanup_source_images: bool = False,
) -> Dict[str, Any]:
    """Compact prune for a Find Lines run folder (plain or per-chip).

    Handles both layouts: flat runs (single-chip expanded, plain) and
    multi-chip expanded runs (`chip_{query}/` folders).  In each base the
    per-query match tables (`{query}_lines.csv`) go once that base's
    `line_summary.csv` exists, and `images/` goes once the PDF/PPTX contact
    sheet exists.  Safe to call on folders without those files: everything
    is skipped.
    """
    output_path = str(output_path)
    root = Path(output_path)
    # Flat base first, then one base per chip folder.
    bases = [root] + sorted(
        d for d in root.glob("chip_*") if d.is_dir()
    )

    removed: List[Dict[str, Any]] = []
    for base in bases:
        summary_ok = _has_content(base / FIND_LINES_SUMMARY)
        if not keep_per_match_csv and summary_ok:
            for path in _glob_files(str(base), ["*_lines.csv"]):
                entry = _remove_file(path)
                if entry:
                    removed.append(entry)

        if cleanup_source_images:
            artifact_ok = any(
                _has_content(base / artifact)
                for artifact in FIND_LINES_IMAGE_SUMMARIES
            )
            images_dir = base / "images"
            if artifact_ok and images_dir.is_dir():
                entry = _remove_tree(images_dir)
                if entry:
                    removed.append(entry)

    bytes_reclaimed = sum(int(entry["bytes"]) for entry in removed)
    return _merge_audit(output_path, removed, bytes_reclaimed,
                        force_write=not keep_per_match_csv)


def prune_find_neurons_run(
    output_path: str,
    keep_per_match_csv: bool = True,
) -> Dict[str, Any]:
    """Compact prune for a Find EM Neurons run folder.

    Removes the bodyId-level tables (`all_neurons.csv`, per-line
    `{line}_neurons.csv`, and `by_dataset/*_neurons.csv`) once the type
    aggregates exist.  Type summaries, `{line}_type_mapped.csv`, the
    distribution plot, and `plot-3d_*` folders are deliverables and stay.
    There is no image payload in this mode.
    """
    output_path = str(output_path)
    by_dataset = Path(output_path) / "by_dataset"
    types_exist = bool(_glob_files(output_path, ["*_types.csv"])) or (
        by_dataset.is_dir()
        and bool(_glob_files(str(by_dataset), ["*_types.csv"]))
    )
    types_have_content = any(
        _has_content(p)
        for p in _glob_files(output_path, ["*_types.csv"])
        + _glob_files(str(by_dataset), ["*_types.csv"])
    ) if types_exist else False
    return _apply(
        output_path,
        keep_per_match_csv=keep_per_match_csv,
        cleanup_source_images=False,
        match_table_patterns=[
            "all_neurons.csv",
            "*_neurons.csv",
            "by_dataset/*_neurons.csv",
        ],
        images_dirs=(),
        summary_artifacts=(),
        match_condition_met=types_exist and types_have_content,
    )


def prune_colabel_run(
    output_path: str,
    keep_per_match_csv: bool = True,
) -> Dict[str, Any]:
    """Compact prune for a Co-Labeling run folder.

    Removes `line_labeled_neurons/` (the per-line × per-dataset row-level
    tables) and `distribution_data_by_neuron.csv` once the similarity
    matrices or the report exist.  Matrices, expression data, by-type
    distributions, and the report are deliverables and stay.
    """
    output_path = str(output_path)
    deliverable_exists = (
        any(_has_content(p)
            for p in _glob_files(output_path, ["colabeling_matrix_*.csv"]))
        or _has_content(Path(output_path) / "colabeling_report.html")
    )

    removed: List[Dict[str, Any]] = []
    if not keep_per_match_csv and deliverable_exists:
        labeled_dir = Path(output_path) / "line_labeled_neurons"
        if labeled_dir.is_dir():
            entry = _remove_tree(labeled_dir)
            if entry:
                removed.append(entry)
        for path in _glob_files(output_path, ["distribution_data_by_neuron.csv"]):
            entry = _remove_file(path)
            if entry:
                removed.append(entry)

    bytes_reclaimed = sum(int(entry["bytes"]) for entry in removed)
    return _merge_audit(output_path, removed, bytes_reclaimed,
                        force_write=not keep_per_match_csv)


def _run_sort_key(folder: Path) -> str:
    """Newest-first sort key from the run folder's embedded timestamp
    (`..._YYYYMMDD_HHMMSS`); falls back to the folder mtime."""
    match = re.search(r"_(\d{8}_\d{6})$", folder.name)
    if match:
        return match.group(1)
    try:
        return datetime.fromtimestamp(
            folder.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    except OSError:
        return ""


def _run_was_compact(folder: Path) -> bool:
    """Whether a run opted into Compact (its match tables are fair game
    for retention pruning).  Recorded via the run's parameters.json flag
    or the presence of a cleanup audit; unknown mode means never touch."""
    if (folder / AUDIT_FILENAME).exists():
        return True
    try:
        payload = json.loads(
            (folder / "parameters.json").read_text(encoding="utf-8"))
        function_params = payload.get("function_params") or {}
        return function_params.get("keep_per_match_csv") is False
    except (OSError, ValueError):
        return False


def enforce_match_table_retention(
    output_dir: str,
    current_run: str,
    keep_last_n: int,
) -> Dict[str, Any]:
    """Rolling match-table window across Find Lines runs (belt-and-braces
    for Compact, plan `_plan/plan-nb-find-lines-output-modes.md` §5).

    The newest ``keep_last_n`` run folders (by embedded timestamp, the
    in-progress run included) KEEP their bodyId-level match tables; older
    sibling runs that opted into Compact get them pruned.  Full runs and
    runs of unknown mode are never touched — retention only ever sweeps
    data whose owner already asked for Compact.  Each pruned run carries
    its own merged `cleanup_audit.json`.

    ``keep_last_n == 0`` means no window (Compact deletes immediately and
    only in the current run).  The per-pass removal list is returned; the
    caller mirrors it into its own provenance records.
    """
    if keep_last_n is None or keep_last_n <= 0:
        return {"kept_window": 0, "pruned_runs": [],
                "removed": [], "bytes_reclaimed": 0}

    root = Path(output_dir)
    runs = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(RUN_FOLDER_PREFIX)
    ]
    runs.sort(key=_run_sort_key, reverse=True)  # newest first

    keep = set(runs[:max(keep_last_n, 0)])
    current = Path(current_run)
    if current.exists():
        keep.add(current)  # the in-progress run is never swept here

    result: Dict[str, Any] = {
        "kept_window": min(keep_last_n, len(runs)),
        "pruned_runs": [],
        "removed": [],
        "bytes_reclaimed": 0,
    }
    for folder in runs:
        if folder in keep:
            continue
        if not _run_was_compact(folder):
            continue
        pass_audit = prune_find_lines_run(
            str(folder), keep_per_match_csv=False,
            cleanup_source_images=False,
        )
        if pass_audit["removed"]:
            result["pruned_runs"].append(folder.name)
            result["removed"].append({
                "run": folder.name,
                "removed": pass_audit["removed"],
            })
            result["bytes_reclaimed"] += pass_audit["bytes_reclaimed_this_pass"]
    return result
