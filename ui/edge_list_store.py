"""Auto-saving edge-list draft store for the network edge-list editor.

Drafts live in ``local_data/edge_list_drafts/`` (gitignored) as one CSV per
draft plus a JSON metadata sidecar:

    my_network.csv         header: source,target,weight[,color][,extended columns]
    my_network.meta.json   {name, slug, created_at, updated_at, dirty, row_count}

Every edit is flushed atomically (temp file + ``os.replace``), so a draft
survives even if the UI process or its port dies mid-session. The ``dirty``
flag stays True until the user explicitly exports the edge list, which lets a
restarted app remind the user about edited-but-not-exported drafts.

The CSV layout matches VisualizePath's edge-list format exactly:
``source``/``target``/``weight`` are required; ``color``, ``source_group`` /
``target_group`` (node groups) and ``edge info`` / ``source info`` /
``target info`` (hover labels, ``{key:val; ...}`` cells) are written only
when at least one row uses them. A draft file can be passed to PlotPath as
``path_file`` without conversion.
"""
import csv
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT

# Columns the editor can hold, in order. ``color``, the group columns and
# the hover-info columns are only written to the draft/export CSV when at
# least one row carries a non-empty value (VisualizePath reads them by
# name and treats them as optional metadata columns).
EDGE_COLUMNS = (
    "source", "target", "weight", "color",
    "source_group", "target_group",
    "edge info", "source info", "target info",
)
REQUIRED_COLUMNS = ("source", "target", "weight")
# Columns hidden in the editor's Basic column set (shown in Full mode).
EXTENDED_COLUMNS = EDGE_COLUMNS[4:]

_store_dir = PROJECT_ROOT / "local_data" / "edge_list_drafts"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Paths & naming
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Reduce a draft name to a safe file slug (empty string if unusable)."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return slug[:80]


def _csv_path(slug: str) -> Path:
    return _store_dir / f"{slug}.csv"


def _meta_path(slug: str) -> Path:
    return _store_dir / f"{slug}.meta.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text crash-safely: temp file in the same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Row validation / normalization
# ---------------------------------------------------------------------------

def normalize_rows(rows: List[dict]) -> List[dict]:
    """Coerce rows to the canonical edge-list shape.

    Keeps the four known columns as strings, drops nothing; missing keys
    become empty strings so the editor can hold half-filled rows.
    """
    normalized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        normalized.append({col: str(row.get(col, "") or "").strip() for col in EDGE_COLUMNS})
    return normalized


def validate_rows(rows: List[dict]) -> List[str]:
    """Return human-readable errors (empty list when the edge list is
    runnable). Half-filled rows are reported by 1-based index."""
    errors = []
    for i, row in enumerate(rows or [], start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: not a mapping")
            continue
        source = str(row.get("source", "") or "").strip()
        target = str(row.get("target", "") or "").strip()
        weight = str(row.get("weight", "") or "").strip()
        if not source and not target and not weight:
            continue  # completely empty rows are ignored, not invalid
        if not source:
            errors.append(f"Row {i}: missing source")
        if not target:
            errors.append(f"Row {i}: missing target")
        if not weight:
            errors.append(f"Row {i}: missing weight")
        else:
            try:
                value = float(weight)
                if value < 0:
                    errors.append(f"Row {i}: weight must be >= 0")
            except ValueError:
                errors.append(f"Row {i}: weight '{weight}' is not a number")
    return errors


def complete_rows(rows: List[dict]) -> List[dict]:
    """Rows usable as a network input (all required fields filled)."""
    return [
        row for row in normalize_rows(rows)
        if all(row[col] for col in REQUIRED_COLUMNS)
    ]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _read_meta(slug: str) -> Optional[dict]:
    try:
        meta = json.loads(_meta_path(slug).read_text(encoding="utf-8"))
        if isinstance(meta, dict) and meta.get("name"):
            return meta
    except (OSError, ValueError):
        pass
    return None


def _write_meta(slug: str, meta: dict) -> None:
    _atomic_write_text(_meta_path(slug), json.dumps(meta, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_draft(name: str, rows: List[dict], dirty: bool = True) -> Optional[str]:
    """Auto-save an edge list under *name*; returns the slug or None.

    The CSV and its metadata are written atomically. When a draft was loaded
    under a different name, call :func:`delete_draft` on the old name to get
    rename semantics (the editor does this).
    """
    slug = sanitize_name(name)
    if not slug:
        return None
    rows = normalize_rows(rows)
    stored_rows = [row for row in rows if not is_empty_row(row)]
    with _lock:
        existing = _read_meta(slug)
        now = _now_iso()
        meta = {
            "name": (name or "").strip(),
            "slug": slug,
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "dirty": bool(dirty),
            "row_count": len(stored_rows),
        }
        try:
            _atomic_write_text(_csv_path(slug), rows_to_csv(rows))
            _write_meta(slug, meta)
        except OSError:
            return None
    return slug


def _csv_quote(value: str) -> str:
    if any(ch in value for ch in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def is_empty_row(row: dict) -> bool:
    """True when a normalized row holds no value in any column."""
    return not any(str(row.get(col, "") or "").strip() for col in EDGE_COLUMNS)


def rows_to_csv(rows: List[dict]) -> str:
    """Serialize rows in the PlotPath edge-list format.

    Fully-empty rows (editor scaffolding) are dropped, and each optional
    column is written only when at least one row uses it.
    """
    normalized = [row for row in normalize_rows(rows) if not is_empty_row(row)]
    columns = list(REQUIRED_COLUMNS)
    for col in EDGE_COLUMNS[len(REQUIRED_COLUMNS):]:
        if any(row[col] for row in normalized):
            columns.append(col)
    lines = [",".join(columns)]
    for row in normalized:
        lines.append(",".join(_csv_quote(row[col]) for col in columns))
    return "\n".join(lines) + "\n"


def load_rows_from_csv_text(text: str) -> List[dict]:
    """Parse CSV text (the editor's or the backend's expanded export format)
    into normalized rows; unknown columns are ignored, missing ones filled."""
    import io
    reader = csv.DictReader(io.StringIO(text))
    return normalize_rows(
        {col: (record.get(col) or "") for col in EDGE_COLUMNS}
        for record in reader
    )


def load_draft(name: str) -> Optional[List[dict]]:
    """Load a draft's rows (normalized), or None when the draft is absent."""
    slug = sanitize_name(name)
    if not slug:
        return None
    path = _csv_path(slug)
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {col: (record.get(col) or "").strip() for col in EDGE_COLUMNS}
                for record in reader
            ]
        return rows
    except (OSError, csv.Error, UnicodeDecodeError, ValueError):
        return None


def get_meta(name: str) -> Optional[dict]:
    """Return a copy of the draft metadata, or None."""
    slug = sanitize_name(name)
    if not slug:
        return None
    with _lock:
        meta = _read_meta(slug)
    return dict(meta) if meta else None


def list_drafts() -> List[dict]:
    """Metadata of every draft, newest update first."""
    drafts = []
    with _lock:
        if _store_dir.exists():
            for meta_file in _store_dir.glob("*.meta.json"):
                slug = meta_file.name[: -len(".meta.json")]
                meta = _read_meta(slug)
                # Skip orphan metadata whose CSV is gone.
                if meta and _csv_path(slug).exists():
                    drafts.append(meta)
    drafts.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return drafts


def pending_drafts() -> List[dict]:
    """Drafts with unsaved (dirty) edits - the recovery reminder list."""
    return [meta for meta in list_drafts() if meta.get("dirty")]


def set_dirty(name: str, dirty: bool) -> bool:
    """Flip the dirty flag (e.g. False after a successful export)."""
    slug = sanitize_name(name)
    if not slug:
        return False
    with _lock:
        meta = _read_meta(slug)
        if meta is None:
            return False
        meta["dirty"] = bool(dirty)
        meta["updated_at"] = _now_iso()
        try:
            _write_meta(slug, meta)
        except OSError:
            return False
    return True


def mark_exported(name: str) -> bool:
    """Mark a draft as exported: no pending unsaved changes."""
    return set_dirty(name, False)


def mark_dirty(name: str) -> bool:
    """Mark a draft as having pending unsaved changes."""
    return set_dirty(name, True)


def draft_csv_path(name: str) -> Optional[str]:
    """Absolute path of the draft CSV (usable as PlotPath path_file)."""
    slug = sanitize_name(name)
    if not slug:
        return None
    path = _csv_path(slug)
    return str(path) if path.exists() else None


def delete_draft(name: str) -> bool:
    """Remove a draft's CSV and metadata. Returns success."""
    slug = sanitize_name(name)
    if not slug:
        return False
    with _lock:
        existed = _csv_path(slug).exists() or _meta_path(slug).exists()
        try:
            _csv_path(slug).unlink(missing_ok=True)
            _meta_path(slug).unlink(missing_ok=True)
        except OSError:
            return False
    return existed
