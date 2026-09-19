"""Compact, versioned Parquet storage for NeuronBridge cache records.

The public Finder API still returns pandas DataFrames.  This module only owns
the on-disk representation, which keeps the cache code small and makes the
schema explicit:

* id-to-line records are stored once per canonical body/dataset/match key;
* image records are stored once per LM image and algorithm;
* derived ``both`` and line-level result tables are not written by the cache
  path.

The cache is intentionally Parquet-only.  Older CSV files are not read or
migrated, because they do not carry enough NeuronBridge-version information to
be safely reused after a cache reset.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    from .flywire_ids import is_local_connectome_dataset, normalize_flywire_body_id
except ImportError:
    from flywire_ids import is_local_connectome_dataset, normalize_flywire_body_id


PARQUET_CACHE_VERSION = 1

# Keep the fields consumed by the Finder and downstream result/export code.
# The old image cache also persisted dataset_folder, neuronType and
# neuronInstance, but those are derivable enrichment fields and are not read
# after _enrich_match_with_dataset_info has run.
IMAGE_COLUMNS = [
    "bodyId",
    "score",
    "image_id",
    "lm_sample",
    "match_type",
    "dataset",
    "library",
    "type",
    "instance",
    "status",
]

ID_COLUMNS = ["line", "library", "score", "image_id", "match_type"]


def _safe_component(value: Any) -> str:
    """Return a filesystem-safe cache-key component."""

    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def _empty_string_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Fill missing string columns without changing numeric score columns."""

    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    return frame


class NeuronBridgeParquetCache:
    """Read/write the new NeuronBridge Parquet cache layout.

    Layout::

        <cache_root>/parquet/<neuronbridge-version>/
            manifest.json
            id_to_lines/<canonical-key>.parquet
            image_cache/<match_type>_<lm_sample>.parquet

    Files are replaced atomically after a complete write.  A per-instance
    lock protects read/merge/write operations when Finder workers process LM
    images concurrently.
    """

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        version: Optional[str] = None,
    ):
        self.version = str(version) if version else None
        version_dir = f"v_{_safe_component(version)}" if version else ""
        self.root = Path(cache_root) / "parquet" / version_dir
        self.id_dir = self.root / "id_to_lines"
        self.image_dir = self.root / "image_cache"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()

    def ensure_manifest(self, neuronbridge_version: Optional[str] = None) -> Path:
        """Create/update the small format manifest and return its path."""

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            manifest: dict[str, Any] = {}
            if self.manifest_path.exists():
                try:
                    manifest = json.loads(self.manifest_path.read_text())
                except Exception:
                    manifest = {}
            manifest.update(
                {
                    "format": "parquet",
                    "schema_version": PARQUET_CACHE_VERSION,
                    "updated_at": now,
                }
            )
            version = neuronbridge_version or self.version
            if version:
                manifest["neuronbridge_version"] = str(version)
            manifest.setdefault("created_at", now)
            temporary = self.manifest_path.with_name(
                f".{self.manifest_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.manifest_path)
        return self.manifest_path

    def id_path(self, identifier: str) -> Path:
        """Return the canonical Parquet path for an id-to-lines key."""

        return self.id_dir / f"id_to_lines_{_safe_component(identifier)}.parquet"

    def image_path(self, image_id: str, match_type: str) -> Path:
        """Return the canonical Parquet path for an image/algorithm pair."""

        return self.image_dir / f"{_safe_component(match_type)}_{_safe_component(image_id)}.parquet"

    def load_id(self, identifier: str) -> Optional[pd.DataFrame]:
        """Load one id-to-lines table, or ``None`` if it does not exist."""

        path = self.id_path(identifier)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def load_image(self, image_id: str, match_type: str) -> Optional[pd.DataFrame]:
        """Load one image table, or ``None`` if it does not exist."""

        path = self.image_path(image_id, match_type)
        if not path.exists():
            return None
        try:
            # Normalize on read as well as on write so caches created before
            # the string-ID contract cannot reintroduce numeric FlyWire IDs.
            return self.normalize_image(pd.read_parquet(path))
        except Exception:
            return None

    @staticmethod
    def normalize_id(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize an id result while retaining optional rank columns."""

        if frame is None or frame.empty:
            return frame.copy() if frame is not None else pd.DataFrame(columns=ID_COLUMNS)
        frame = _empty_string_columns(frame, ["line", "library", "image_id", "match_type"])
        if "score" not in frame.columns:
            frame["score"] = 0.0
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
        ordered = [column for column in ID_COLUMNS if column in frame.columns]
        optional = [
            "cds_score",
            "pppm_score",
            "cds_rank",
            "pppm_rank",
            "combined_rank",
        ]
        ordered.extend([column for column in optional if column in frame.columns])
        frame = frame[ordered].copy()
        edge_key = ["line", "image_id", "match_type"]
        frame = frame.sort_values("score", ascending=False, kind="stable").drop_duplicates(
            subset=edge_key, keep="first"
        )
        # Preserve the public id_to_lines ordering after a cache reload.
        return frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)

    @staticmethod
    def normalize_image(frame: pd.DataFrame) -> pd.DataFrame:
        """Project image records onto the compact, stable image schema."""

        if frame is None or frame.empty:
            return pd.DataFrame(columns=IMAGE_COLUMNS)
        # Do not stringify bodyId before validating it: converting an unsafe
        # FlyWire float to text first would preserve the already-rounded value
        # and make the precision loss impossible to detect.
        frame = _empty_string_columns(
            frame,
            [
                "image_id",
                "lm_sample",
                "match_type",
                "dataset",
                "library",
                "type",
                "instance",
                "status",
            ],
        )
        if "bodyId" not in frame.columns:
            frame["bodyId"] = ""
        frame["bodyId"] = [
            normalize_flywire_body_id(value, field="bodyId")
            if is_local_connectome_dataset(dataset)
            else str(value)
            for value, dataset in zip(frame["bodyId"], frame["dataset"])
        ]
        if "score" not in frame.columns:
            frame["score"] = 0.0
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
        # Body IDs are identifiers, not quantities.  Keeping them as strings
        # avoids integer inference changing between files or platforms.
        frame["bodyId"] = frame["bodyId"].astype("string")
        frame = frame[IMAGE_COLUMNS].copy()

        # A repeated API lookup can return the same edge more than once.  Keep
        # the best score for the stable edge key and discard exact duplicates.
        edge_key = ["bodyId", "dataset", "image_id", "lm_sample", "match_type"]
        frame = frame.sort_values("score", kind="stable").drop_duplicates(
            subset=edge_key, keep="last"
        )
        return frame.reset_index(drop=True)

    def save_id(self, identifier: str, frame: pd.DataFrame) -> Optional[Path]:
        """Write an id table with stable column order and Zstandard compression."""

        frame = self.normalize_id(frame)
        if frame.empty:
            return None
        with self._lock:
            existing = self.load_id(identifier)
            if existing is not None and not existing.empty:
                frame = self.normalize_id(pd.concat([existing, frame], ignore_index=True))
            return self._write_atomic(frame, self.id_path(identifier))

    def save_image(
        self,
        image_id: str,
        match_type: str,
        frame: pd.DataFrame,
    ) -> Optional[Path]:
        """Merge and write one image table, deduplicated by stable edge key."""

        frame = self.normalize_image(frame)
        if frame.empty:
            return None
        path = self.image_path(image_id, match_type)
        with self._lock:
            existing = self.load_image(image_id, match_type)
            if existing is not None and not existing.empty:
                frame = self.normalize_image(pd.concat([existing, frame], ignore_index=True))
            return self._write_atomic(frame, path)

    def _write_atomic(self, frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def iter_parquet_files(self) -> list[Path]:
        """List new cache files for diagnostics and cleanup."""

        if not self.root.exists():
            return []
        return [path for path in self.root.rglob("*.parquet") if path.is_file()]
