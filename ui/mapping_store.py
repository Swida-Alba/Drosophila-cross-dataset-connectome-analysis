"""Custom type-mapping store: persistent, named LabelMapper presets.

Presets live in ``cache/user_mappings/user_mappings.json`` (gitignored) and use
LabelMapper's native JSON schema:

    {
        "name": "aMe12 orthologs",
        "description": "...",
        "source_mapping": {
            "custom_label": ["grp1", "grp2"],
            "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]],
            "male-cns:v0.9": [["aMe12"], ["aMe12-like"]]
        },
        "target_mapping": {...},
        "intermediate_mapping": {...}
    }

For a run, the selected preset is exported to a dedicated JSON file
(``cache/user_mappings/<slug>.json``) that only contains the mapping keys,
so it can be passed straight to ``LabelMapper(overall_mapping_json=...)``.
"""
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT

MAPPING_SIDES = ("source_mapping", "target_mapping", "intermediate_mapping")
_LABEL_KEYS = ("custom_label", "std_label")

_store_dir = PROJECT_ROOT / "cache" / "user_mappings"
_store_file = _store_dir / "user_mappings.json"
_lock = threading.Lock()


def _load_all() -> dict:
    """Load the whole store (presets + active marker)."""
    try:
        if _store_file.exists():
            data = json.loads(_store_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("mappings"), list):
                return data
    except (OSError, ValueError):
        pass
    return {"mappings": [], "active": None}


def _save_all(data: dict) -> bool:
    """Persist the whole store atomically; returns success.

    A crash mid-write must not truncate every saved preset away (the other
    stores' invariant: tmp + replace, see history_store).
    """
    tmp = _store_file.with_suffix(f".tmp{os.getpid()}")
    try:
        _store_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, _store_file)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _find_index(mappings: List[dict], name: str) -> int:
    for i, m in enumerate(mappings):
        if m.get("name") == name:
            return i
    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_mappings() -> List[str]:
    """Return preset names (sorted)."""
    with _lock:
        return sorted(m.get("name", "") for m in _load_all()["mappings"] if m.get("name"))


def get_mapping(name: str) -> Optional[dict]:
    """Return a preset (with name/description/mapping sides), or None."""
    with _lock:
        for m in _load_all()["mappings"]:
            if m.get("name") == name:
                return dict(m)
    return None


def save_mapping(name: str, mapping_data: dict, description: str = "") -> bool:
    """Create or update a preset.

    *mapping_data* holds the LabelMapper sides (``source_mapping`` etc.).
    Returns success (also False when the name is invalid).
    """
    name = (name or "").strip()
    if not name:
        return False
    preset = {
        "name": name,
        "description": (description or "").strip(),
        **{side: mapping_data.get(side) for side in MAPPING_SIDES if mapping_data.get(side)},
    }
    with _lock:
        data = _load_all()
        idx = _find_index(data["mappings"], name)
        if idx >= 0:
            data["mappings"][idx] = preset
        else:
            data["mappings"].append(preset)
        if not _save_all(data):
            return False
    return _export_mapping_file(name, preset) is not None


def delete_mapping(name: str) -> bool:
    """Remove a preset (and its exported file). Returns success."""
    with _lock:
        data = _load_all()
        idx = _find_index(data["mappings"], name)
        if idx < 0:
            return False
        data["mappings"].pop(idx)
        if data.get("active") == name:
            data["active"] = None
        if not _save_all(data):
            return False
    _remove_exported_file(name)
    return True


def rename_mapping(old_name: str, new_name: str) -> bool:
    """Rename a preset; the active marker follows. Returns success."""
    new_name = (new_name or "").strip()
    if not new_name or new_name == old_name:
        return False
    with _lock:
        data = _load_all()
        idx = _find_index(data["mappings"], old_name)
        if idx < 0 or _find_index(data["mappings"], new_name) >= 0:
            return False
        data["mappings"][idx]["name"] = new_name
        if data.get("active") == old_name:
            data["active"] = new_name
        if not _save_all(data):
            return False
        preset = data["mappings"][idx]
    _remove_exported_file(old_name)
    return _export_mapping_file(new_name, preset) is not None


def get_active_mapping() -> Optional[str]:
    """Return the active preset name, or None."""
    with _lock:
        data = _load_all()
        active = data.get("active")
        if active is not None and _find_index(data["mappings"], active) >= 0:
            return active
    return None


def set_active_mapping(name: Optional[str]) -> bool:
    """Set (or clear, with None) the active preset. Returns success."""
    with _lock:
        data = _load_all()
        if name is not None and _find_index(data["mappings"], name) < 0:
            return False
        data["active"] = name
        return _save_all(data)


def mapping_file_path(name: str) -> Optional[str]:
    """Path of the exported JSON consumed by runs, or None if absent."""
    path = _export_path(name)
    return str(path) if path.exists() else None


def _export_path(name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "mapping"
    return _store_dir / f"{slug}.json"


def _export_mapping_file(name: str, preset: dict) -> Optional[Path]:
    """Write only the mapping keys to the preset's export file."""
    payload = {side: preset.get(side) for side in MAPPING_SIDES if preset.get(side)}
    try:
        _store_dir.mkdir(parents=True, exist_ok=True)
        path = _export_path(name)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except OSError:
        return None


def _remove_exported_file(name: str) -> None:
    try:
        _export_path(name).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_mapping(mapping_data: dict) -> List[str]:
    """Validate LabelMapper-side structures; return a list of error strings
    (empty when valid)."""
    errors = []
    for side in MAPPING_SIDES:
        data = mapping_data.get(side)
        if not data:
            continue
        if not isinstance(data, dict):
            errors.append(f"{side}: must be an object")
            continue
        label_key = next((k for k in _LABEL_KEYS if k in data), None)
        if label_key is None:
            errors.append(f"{side}: missing '{_LABEL_KEYS[0]}' group-name list")
            continue
        labels = data[label_key]
        if not isinstance(labels, list) or not labels:
            errors.append(f"{side}: '{label_key}' must be a non-empty list")
            continue
        if len(set(labels)) != len(labels):
            errors.append(f"{side}: group names must be unique")
        for ds, groups in data.items():
            if ds in _LABEL_KEYS:
                continue
            if not isinstance(groups, list) or len(groups) != len(labels):
                errors.append(
                    f"{side}: dataset '{ds}' must have {len(labels)} groups "
                    f"(one per group name), got {len(groups) if isinstance(groups, list) else 'invalid'}"
                )
                continue
            for gi, group in enumerate(groups):
                if not isinstance(group, list) or not all(isinstance(x, str) and x.strip() for x in group):
                    errors.append(f"{side}: dataset '{ds}' group {gi + 1} must be a list of neuron identifiers")
    return errors
