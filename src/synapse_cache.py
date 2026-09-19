"""Persistent cache primitives for visualization synapse queries.

The visualization cache has two useful granularities:

* pair files (``{pre}_{post}.parquet``) are ideal for exact inter-layer
  requests and incremental resume;
* query files (``queries/{hash}.parquet``) are ideal for connector/site
  requests where one side of the query is open-ended.

Keeping both granularities in the *synapse* namespace is important: connector
sites are derived from synapse rows and must not depend on the neuron/SWC
cache.  Query metadata makes a cache hit auditable and prevents a file from a
different dataset or query contract being silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


SYNAPSE_CACHE_SCHEMA_VERSION = 2
# Query rows are filtered to the exact requested body-ID sets before they are
# persisted.  Bump this when that contract changes so older broad-query files
# cannot be reused as exact layer/site results.
SYNAPSE_QUERY_SCHEMA_VERSION = 2
PAIR_COLUMNS = (
    "bodyId_pre", "bodyId_post",
    "x_pre", "y_pre", "z_pre",
    "x_post", "y_post", "z_post",
)


def dataset_folder(dataset: str) -> str:
    """Return the filesystem-safe dataset namespace used by DROCAT."""

    return str(dataset).replace(":", "_").replace(".", "_")


def _json_value(value):
    """Convert criteria-like values into deterministic JSON data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # numpy scalar/list values frequently appear in body-ID criteria built
    # from a DataFrame.  Normalize them before hashing rather than falling
    # back to a process-specific repr.
    try:
        if hasattr(value, "item"):
            return _json_value(value.item())
        if hasattr(value, "tolist"):
            return _json_value(value.tolist())
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)

    # NeuronCriteria/SynapseCriteria expose their useful fields as regular
    # attributes, but also carry a client object.  Keep only stable public
    # fields and fall back to repr for small test/custom criteria objects.
    attrs = {}
    for name in (
        "bodyId", "bodyIds", "type", "instance", "status", "roi",
        "inputRois", "outputRois", "primaryRois", "confidence",
        "min_weight", "weight", "pre", "post",
    ):
        try:
            candidate = getattr(value, name)
        except Exception:
            continue
        if candidate is not None and not callable(candidate):
            attrs[name] = _json_value(candidate)
    if attrs:
        return {
            "class": type(value).__name__,
            "attrs": attrs,
        }
    return {
        "class": type(value).__name__,
        "repr": repr(value),
    }


def stable_query_spec(spec: Mapping) -> dict:
    """Return a JSON-safe, deterministic query specification."""

    return _json_value(dict(spec))


def query_key(spec: Mapping) -> str:
    """Hash a query specification into a portable cache filename."""

    payload = json.dumps(
        stable_query_spec(spec),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_json(data: Mapping, path: Path) -> None:
    """Write a small JSON sidecar atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _pair_key(pre_id, post_id) -> str:
    return f"{str(pre_id)}\t{str(post_id)}"


class SynapseCache:
    """Dataset-scoped pair/query cache for connector data."""

    def __init__(self, dataset: str, project_root: str, enabled: bool = True):
        self.dataset = str(dataset)
        self.project_root = Path(project_root)
        self.enabled = bool(enabled)
        self.root = (
            self.project_root / "cache" / dataset_folder(self.dataset) / "synapses"
        )
        self.query_dir = self.root / "queries"
        self.manifest_path = self.root / "manifest.json"
        self._query_frames: dict[str, pd.DataFrame] = {}
        self._pair_frames: dict[str, pd.DataFrame] = {}
        self._manifest_cache = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _read_manifest(self) -> dict:
        if self._manifest_cache is not None:
            return self._manifest_cache
        if not self.manifest_path.exists():
            self._manifest_cache = {
                "schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                "dataset": self.dataset,
                "pairs": {},
            }
            return self._manifest_cache
        try:
            data = json.loads(self.manifest_path.read_text())
            if (
                data.get("schema_version") != SYNAPSE_CACHE_SCHEMA_VERSION
                or data.get("dataset") != self.dataset
            ):
                data = {
                    "schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                    "dataset": self.dataset,
                    "pairs": {},
                }
            data.setdefault("pairs", {})
            self._manifest_cache = data
        except Exception:
            self._manifest_cache = {
                "schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                "dataset": self.dataset,
                "pairs": {},
            }
        return self._manifest_cache

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        manifest = self._read_manifest()
        manifest["updated_at"] = self._now()
        _atomic_json(manifest, self.manifest_path)

    def pair_path(self, pre_id, post_id) -> Path:
        return self.root / f"{pre_id}_{post_id}.parquet"

    @staticmethod
    def query_key(spec: Mapping) -> str:
        return query_key(spec)

    def query_path(self, spec: Mapping) -> Path:
        return self.query_dir / f"{query_key(spec)}.parquet"

    def query_meta_path(self, spec: Mapping) -> Path:
        return self.query_dir / f"{query_key(spec)}.json"

    @staticmethod
    def _valid_frame(frame: pd.DataFrame | None) -> bool:
        if frame is None or not isinstance(frame, pd.DataFrame):
            return False
        if frame.empty:
            return True
        return set(PAIR_COLUMNS).issubset(frame.columns)

    def load_query(self, spec: Mapping) -> pd.DataFrame | None:
        """Load an exact query result, or ``None`` on a cache miss."""

        if not self.enabled:
            return None
        key = query_key(spec)
        if key in self._query_frames:
            return self._query_frames[key].copy()
        path = self.query_path(spec)
        meta_path = self.query_meta_path(spec)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            normalized = stable_query_spec(spec)
            if (
                meta.get("schema_version") != SYNAPSE_QUERY_SCHEMA_VERSION
                or meta.get("dataset") != self.dataset
                or meta.get("query_key") != key
                or meta.get("spec") != normalized
            ):
                return None
            frame = pd.read_parquet(path)
            if not self._valid_frame(frame):
                return None
        except Exception:
            return None
        self._query_frames[key] = frame.copy()
        return frame

    def save_query(self, spec: Mapping, frame: pd.DataFrame | None,
                   source: str = "unknown") -> None:
        """Persist an exact query result and its provenance sidecar."""

        if not self.enabled:
            return
        normalized = stable_query_spec(spec)
        key = query_key(normalized)
        if frame is None:
            frame = pd.DataFrame(columns=list(PAIR_COLUMNS))
        elif not isinstance(frame, pd.DataFrame):
            return
        if not self._valid_frame(frame):
            return
        frame = frame.copy()
        _atomic_parquet(frame, self.query_path(normalized))
        _atomic_json(
            {
                "schema_version": SYNAPSE_QUERY_SCHEMA_VERSION,
                "cache_schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                "dataset": self.dataset,
                "query_key": key,
                "spec": normalized,
                "source": str(source),
                "row_count": int(len(frame)),
                "created_at": self._now(),
                "columns": list(frame.columns),
            },
            self.query_meta_path(normalized),
        )
        self._query_frames[key] = frame.copy()

    def load_pairs(self, source_ids: Iterable, target_ids: Iterable):
        """Return ``(rows, missing_pairs)`` for exact pair requests.

        Empty results recorded in the manifest count as known pairs without
        creating a zero-column parquet file. Existing legacy empty pair files
        remain readable.
        """

        if not self.enabled:
            pairs = [(source, target)
                 for source in source_ids for target in target_ids]
            return None, pairs
        sources = sorted({str(value) for value in source_ids})
        targets = sorted({str(value) for value in target_ids})
        manifest = self._read_manifest().get("pairs", {})
        frames = []
        missing = []
        for pre_id in sources:
            for post_id in targets:
                key = _pair_key(pre_id, post_id)
                cached = self._pair_frames.get(key)
                if cached is not None:
                    if not cached.empty:
                        frames.append(cached.copy())
                    continue
                entry = manifest.get(key) or {}
                if entry.get("empty") and not entry.get("file"):
                    self._pair_frames[key] = pd.DataFrame(columns=list(PAIR_COLUMNS))
                    continue
                path = self.root / entry.get("file", f"{pre_id}_{post_id}.parquet")
                if not path.exists():
                    missing.append((pre_id, post_id))
                    continue
                try:
                    frame = pd.read_parquet(path)
                    if not self._valid_frame(frame):
                        missing.append((pre_id, post_id))
                        continue
                except Exception:
                    missing.append((pre_id, post_id))
                    continue
                self._pair_frames[key] = frame.copy()
                if not frame.empty:
                    frames.append(frame)
        result = pd.concat(frames, ignore_index=True) if frames else None
        return result, missing

    def save_pairs(self, frame: pd.DataFrame | None,
                   attempted_pairs: Iterable[tuple] | None = None) -> None:
        """Persist non-empty pair rows and index empty attempted pairs."""

        if not self.enabled:
            return
        manifest = self._read_manifest()
        pairs = manifest.setdefault("pairs", {})
        saved = set()
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            if not self._valid_frame(frame):
                return
            grouped = frame.groupby(["bodyId_pre", "bodyId_post"], dropna=False)
            for (pre_id, post_id), group in grouped:
                pre = str(pre_id)
                post = str(post_id)
                key = _pair_key(pre, post)
                path = self.pair_path(pre, post)
                _atomic_parquet(group.reset_index(drop=True), path)
                self._pair_frames[key] = group.reset_index(drop=True).copy()
                pairs[key] = {
                    "file": path.name,
                    "empty": False,
                    "row_count": int(len(group)),
                    "schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                    "updated_at": self._now(),
                }
                saved.add((pre, post))
        for pre_id, post_id in attempted_pairs or ():
            pre, post = str(pre_id), str(post_id)
            if (pre, post) in saved:
                continue
            key = _pair_key(pre, post)
            pairs[key] = {
                "file": None,
                "empty": True,
                "row_count": 0,
                "schema_version": SYNAPSE_CACHE_SCHEMA_VERSION,
                "updated_at": self._now(),
            }
            self._pair_frames[key] = pd.DataFrame(columns=list(PAIR_COLUMNS))
        self._write_manifest()
