"""Shared provenance contract for compressed-SWC skeleton caches.

Every skeleton cache file (``.swc.zst`` / ``.swc.gz``) may carry two header
comment lines prepended before the SWC body::

    # DROCAT simpl: 90
    # DROCAT source: banc_gcs_full

``simpl`` records the on-disk simplification level (percent of nodes removed;
absent/legacy files are raw, 0) so a later load can re-level.  ``source``
records the producing pipeline (CAVE mesh skeletonization, a local extrusion
fix, a BANC public-bucket product, ...).  Both parsers scan only the first
few lines and tolerate bytes or str input.

This module is the single implementation of that contract.  It is a leaf
module: it imports nothing from the rest of ``src`` so any module (morphology,
cave_data_fetcher, banc_public_data, visualize_skeleton) can use it without an
import cycle.  Callers that historically exposed the same helpers under a
private name keep that name as a thin alias.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

# Header line recording the on-disk simplification level.
SIMPLIFICATION_HEADER = "DROCAT simpl:"
# Header line recording the producing pipeline.
SOURCE_HEADER = "DROCAT source:"

# Number of leading lines scanned for the header pairs (navis metadata lines
# may follow, and legacy files may start with a few unrelated comments).
_HEADER_SCAN_LINES = 8

# ---------------------------------------------------------------------------
# Source tags (the values written after ``# DROCAT source:``)
# ---------------------------------------------------------------------------
CAVE_MESH_WAVEFRONT = "cave_mesh_wavefront"
LOCAL_EXTRUSION_FIX = "local_extrusion_fix"
BANC_GCS_L2 = "banc_gcs_l2"
BANC_GCS_FULL = "banc_gcs_full"
BANC_GCS_PCG_UM_X1000 = "banc_gcs_pcg_um_x1000"
NEUPRINT_FETCH = "neuprint.fetch_skeleton"
FAFB_CODEX_HEALED = "fafb_codex_healed"


class SkeletonProvenance(NamedTuple):
    """Parsed provenance of one compressed-SWC cache file."""

    simplification: int
    source: str


def _as_text(text) -> str:
    if isinstance(text, bytes):
        return text.decode("utf-8", "replace")
    return text


def _header_lines(text) -> list:
    """Stripped comment payloads of the leading header lines."""
    lines = []
    for line in _as_text(text).splitlines()[:_HEADER_SCAN_LINES]:
        line = line.strip()
        if line.startswith("#"):
            line = line[1:].strip()
        lines.append(line)
    return lines


def read_stored_simplification(text) -> int:
    """Parse ``# DROCAT simpl: N`` (0 = raw; headerless/legacy files = 0)."""
    for line in _header_lines(text):
        if line.startswith(SIMPLIFICATION_HEADER):
            try:
                return int(line[len(SIMPLIFICATION_HEADER):].strip())
            except ValueError:
                return 0
    return 0


def read_stored_source(text) -> str:
    """Parse ``# DROCAT source: NAME`` (``''`` when absent)."""
    for line in _header_lines(text):
        if line.startswith(SOURCE_HEADER):
            return line[len(SOURCE_HEADER):].strip()
    return ""


def parse_provenance(text) -> SkeletonProvenance:
    """Both header values in one scan."""
    return SkeletonProvenance(read_stored_simplification(text),
                              read_stored_source(text))


def resolution_from_source(source: str) -> str:
    """BANC resolution class implied by a provenance tag.

    ``banc_gcs_full`` -> ``'full'``; every other tag (``banc_gcs_l2``,
    ``banc_gcs_pcg_um_x1000``, a headerless legacy entry) reads as the
    coarse ``'l2'``.  Non-BANC tags are irrelevant to this classification.
    """
    return "full" if str(source or "").endswith("_full") else "l2"


def is_repaired_source(source: str) -> bool:
    """Whether a provenance tag denotes an extrusion-repair replacement."""
    return str(source or "") in (CAVE_MESH_WAVEFRONT, LOCAL_EXTRUSION_FIX)


def source_of(neuron) -> str:
    """The ``_drocat_source`` provenance attached to a loaded neuron."""
    return getattr(neuron, "_drocat_source", "") or ""


def write_compressed_swc_zst(path, text: bytes,
                             simplification: int = 0) -> None:
    """Atomically write compressed SWC as zstd-19 with a recorded level.

    ``text`` is the raw SWC body (bytes); the level header is prepended here.
    This is the text-level writer used by the cache stores that already have
    serialized bytes.  The neuron-level writer in ``morphology`` (which
    applies downsampling and ``navis.write_swc``) is a separate concern.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_out = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        payload = b"# DROCAT simpl: %d\n" % int(simplification) + text
        if zstd is None:
            raise ImportError(
                "zstandard is required to write .swc.zst caches")
        blob = zstd.ZstdCompressor(
            level=19, write_content_size=True).compress(payload)
        temp_out.write_bytes(blob)
        os.replace(temp_out, path)
    finally:
        temp_out.unlink(missing_ok=True)


def text_writer_available() -> bool:
    """Whether zstd is importable (the writer needs it)."""
    return zstd is not None


def make_source_line(source: str) -> bytes:
    """The ``# DROCAT source: NAME\\n`` header bytes prepended to SWC bodies."""
    return f"# {SOURCE_HEADER} {source}\n".encode("ascii")


# ---------------------------------------------------------------------------
# Raw-store layout (shared by the NeuPrint and BANC raw skeleton caches)
# ---------------------------------------------------------------------------
# FAFB does NOT write here: it serves from the healed zip (plus the
# codex_skeleton_cache overlay for lazily fetched skeletons) and the
# cave_skeletons / extrusion_fixes repair stores.
RAW_SKELETON_DIRNAME = "raw_skeletons"


def raw_skeleton_store_dir(project_root, dataset_folder) -> Path:
    """``<root>/cache/<folder>/skeletons/raw_skeletons`` for a dataset.

    ``dataset_folder`` must already be the normalized cache folder (the
    callers canonicalize BANC/FAFB aliases before reaching here).
    """
    return (Path(project_root) / "cache" / str(dataset_folder)
            / "skeletons" / RAW_SKELETON_DIRNAME)


def raw_skeleton_cache_path(project_root, dataset_folder, body_id) -> Path:
    """Canonical ``{bodyId}.swc.zst`` path in the shared raw store."""
    return raw_skeleton_store_dir(project_root, dataset_folder) / \
        f"{body_id}.swc.zst"
