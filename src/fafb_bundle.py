"""FAFB healed skeleton bundle — appendable columnar zstd container.

``sk_lod1_783_healed.zst`` (magic ``DRCB1SWC``) stores the healed SWC
skeletons of the FlyWire FAFB v783 release as zstd-19 frames of
column-major numeric blocks (~8 MiB each,
28 B/node: f32 x/y/z/radius, i32 node_id/type/parent_id).  A sorted neuron
index (bodyId -> block + row) gives fast per-id random access, and the
container is *appendable*: lazy conversion appends new blocks and rewrites
the index + footer at the end of the file, so a bulk ``pack`` and on-demand
per-skeleton conversion produce the exact same format and can be mixed.

The application serves skeletons from the healed ZIP directly (zip-only
mode): ``open_bundle`` resolves the ZIP and returns a bundle with
``bundle_path=None`` that never creates, appends to, or requires a
``.zst`` container — with the ``codex_skeleton_cache.zip`` overlay layered
on top: lazily fetched Codex SWCs (and their locally pruned repairs) are
cached there via ``write_overlay_members``, win over the base source per
id, and never mutate the pristine release files. Provenance for overlay
members lives in ``codex_skeleton_cache.json`` because this reader strips
SWC comment lines when rebuilding text.  The columnar format and its
tooling (``pack|verify|info|compact|append``) remain available for manual
maintenance; an existing ``.zst`` is opened read-only only when no ZIP
is present.

Precision policy: coordinates are stored as IEEE float32 — max absolute
error <= 0.06 nm at the corpus maximum (the source text quantizes to 1-4
decimals); node/type/parent exact; radius stored as f32 (exact for the
integer-valued corpus, fractional radii preserved).  The packer asserts
max round-trip error < 0.1 nm and refuses to build otherwise.

Run:  python -m src.fafb_bundle info|pack|verify|compact|append
"""
import array
import io
import json
import mmap
import os
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

MAGIC = b"DRCB1SWC"
FOOTER_MAGIC = b"DRCB1FTR"
FORMAT_VERSION = 2
HEADER_LEN = 1024                     # fixed padded header region (rewritten at pack end)

DEFAULT_LEVEL = 19
BLOCK_BYTES = 8 * 1024 * 1024         # ~8 MiB uncompressed columnar payload per block
NODE_BYTES = 28                       # 4*4 + 4*3 (node/type/parent i32, x/y/z/radius f32)
MAX_COORD_ERROR_NM = 0.1              # packer gate: float32 round-trip error limit
COMPACT_THRESHOLD_ENTRIES = 10_000    # lazy path: compact ZIP after this many converts
COMPACT_THRESHOLD_BYTES = 512 * 1024 * 1024

# footer: magic[8] u64 b_off u64 b_len u64 n_off u64 n_len
#         u32 n_blocks u32 n_neurons u32 crc_b u32 crc_n u32 version  (60 B)
_FOOTER = struct.Struct("<8sQQQQIIIII")
_BLOCK_ENTRY = struct.Struct("<QIII")          # 20 B: off, comp, raw, n_neurons
_NEURON_ENTRY = struct.Struct("<QII")          # 16 B: bodyId, block, row


class BundleCorruptError(IOError):
    """Raised when the bundle index/footer fails its CRC or layout checks."""


def _require_zstd():
    if zstd is None:
        raise ImportError("zstandard is required for .zst skeleton bundles")


def _parse_swc_rows(content: bytes) -> List[Tuple[int, int, float, float, float, float, int]]:
    """Parse SWC text bytes into (node_id, type, x, y, z, radius, parent) rows."""
    rows = []
    for line in content.splitlines():
        if not line or line[:1] == b"#":
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append((
            int(parts[0]), int(parts[1]),
            float(parts[2]), float(parts[3]), float(parts[4]),
            float(parts[5]), int(parts[6]),
        ))
    return rows


def _swc_text_from_rows(rows) -> str:
    """Rebuild canonical SWC text (coords at full float32 round-trip precision)."""
    lines = [
        f"{node_id} {typ} {x:.9g} {y:.9g} {z:.9g} {radius:.9g} {parent}"
        for node_id, typ, x, y, z, radius, parent in rows
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _rows_round_trip_error(rows) -> float:
    """Max |float32(v) - v| over the float columns of the rows (nm)."""
    worst = 0.0
    for row in rows:
        for value in (row[2], row[3], row[4], row[5]):
            f32 = float(struct.unpack("<f", struct.pack("<f", value))[0])
            worst = max(worst, abs(f32 - value))
    return worst


# ---------------------------------------------------------------------------
# Block (de)serialization — column-major numeric payload
# ---------------------------------------------------------------------------

def _pack_block(neurons: List[Tuple[int, List]]) -> Tuple[bytes, int, List[Tuple[int, int]]]:
    """Serialize one block: (payload, n_nodes, [(body_id, row_offset), ...]).

    ``neurons`` is a list of (body_id, rows) pairs; neurons are never split
    across blocks (callers flush between neurons).
    """
    row_offsets = array.array("I")
    node_ids = array.array("i")
    types = array.array("i")
    xs = array.array("f")
    ys = array.array("f")
    zs = array.array("f")
    radii = array.array("f")
    parents = array.array("i")
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for body_id, rows in neurons:
        row_offsets.append(cursor)
        offsets.append((int(body_id), len(offsets)))  # neuron ordinal in block
        for node_id, typ, x, y, z, radius, parent in rows:
            node_ids.append(node_id)
            types.append(typ)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            radii.append(radius)
            parents.append(parent)
            cursor += 1
    row_offsets.append(cursor)
    payload = b"".join((
        row_offsets.tobytes(),
        node_ids.tobytes(), types.tobytes(),
        xs.tobytes(), ys.tobytes(), zs.tobytes(),
        radii.tobytes(), parents.tobytes(),
    ))
    return payload, cursor, offsets


def _unpack_block(payload: bytes, n_neurons: int, n_nodes: int):
    """Unpack a block payload into (row_offsets, {column: array})."""
    pos = 0
    row_offsets = array.array("I")
    row_offsets.frombytes(payload[pos:pos + 4 * (n_neurons + 1)])
    pos += 4 * (n_neurons + 1)
    columns: Dict[str, array.array] = {}
    for name, typecode in (("node_id", "i"), ("type", "i"),
                           ("x", "f"), ("y", "f"), ("z", "f"),
                           ("radius", "f"), ("parent_id", "i")):
        col = array.array(typecode)
        col.frombytes(payload[pos:pos + 4 * n_nodes])
        pos += 4 * n_nodes
        columns[name] = col
    return row_offsets, columns


# ---------------------------------------------------------------------------
# Writer / index helpers
# ---------------------------------------------------------------------------

def _header_json(max_error: float = 0.0) -> bytes:
    header = {
        "level": DEFAULT_LEVEL,
        "block_bytes": BLOCK_BYTES,
        "node_bytes": NODE_BYTES,
        "radius_type": "f32",
        "max_coord_error_nm": max_error,
    }
    raw = json.dumps(header, sort_keys=True).encode("utf-8")
    return raw.ljust(HEADER_LEN, b" ")


def _write_header(handle, max_error: float = 0.0) -> None:
    handle.write(MAGIC)
    handle.write(struct.pack("<II", FORMAT_VERSION, HEADER_LEN))
    handle.write(_header_json(max_error))


def _read_header(handle) -> dict:
    handle.seek(0)
    magic = handle.read(len(MAGIC))
    if magic != MAGIC:
        raise BundleCorruptError("bad bundle magic")
    version, header_len = struct.unpack("<II", handle.read(8))
    if version != FORMAT_VERSION:
        raise BundleCorruptError(f"unsupported bundle version {version}")
    raw = handle.read(header_len)
    return json.loads(raw.rstrip(b" ").decode("utf-8"))


def _write_footer(handle, block_off, block_len, neuron_off, neuron_len,
                  n_blocks, n_neurons, crc_block, crc_neuron) -> None:
    handle.write(_FOOTER.pack(
        FOOTER_MAGIC, block_off, block_len, neuron_off, neuron_len,
        n_blocks, n_neurons, crc_block, crc_neuron, FORMAT_VERSION))


def _read_footer(handle) -> dict:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    if size < _FOOTER.size + len(MAGIC):
        raise BundleCorruptError(f"bundle too small: {size} bytes")
    handle.seek(size - _FOOTER.size)
    (magic, b_off, b_len, n_off, n_len,
     n_blocks, n_neurons, crc_b, crc_n, version) = _FOOTER.unpack(handle.read(_FOOTER.size))
    if magic != FOOTER_MAGIC:
        raise BundleCorruptError("bad footer magic")
    if version != FORMAT_VERSION:
        raise BundleCorruptError(f"unsupported bundle version {version}")
    return {
        "block_off": b_off, "block_len": b_len, "neuron_off": n_off,
        "neuron_len": n_len, "n_blocks": n_blocks, "n_neurons": n_neurons,
        "crc_block": crc_b, "crc_neuron": crc_n,
    }


def _read_indexes(handle, footer: dict):
    """Load block table + sorted neuron index; verify CRCs."""
    handle.seek(footer["block_off"])
    block_raw = handle.read(footer["block_len"])
    handle.seek(footer["neuron_off"])
    neuron_raw = handle.read(footer["neuron_len"])
    if zlib.crc32(block_raw) & 0xFFFFFFFF != footer["crc_block"]:
        raise BundleCorruptError("block index CRC mismatch")
    if zlib.crc32(neuron_raw) & 0xFFFFFFFF != footer["crc_neuron"]:
        raise BundleCorruptError("neuron index CRC mismatch")
    block_idx = [
        _BLOCK_ENTRY.unpack_from(block_raw, i * _BLOCK_ENTRY.size)
        for i in range(footer["n_blocks"])
    ]
    neuron_idx = [
        _NEURON_ENTRY.unpack_from(neuron_raw, i * _NEURON_ENTRY.size)
        for i in range(footer["n_neurons"])
    ]
    return block_idx, neuron_idx


def _build_indexes(block_table, neuron_entries) -> Tuple[bytes, bytes]:
    block_raw = b"".join(
        _BLOCK_ENTRY.pack(off, comp, raw, n_neurons)
        for off, comp, raw, n_neurons in block_table)
    neuron_raw = b"".join(
        _NEURON_ENTRY.pack(body_id, block, row)
        for body_id, block, row in sorted(neuron_entries, key=lambda e: e[0]))
    return block_raw, neuron_raw


def _compress_frame(payload: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    _require_zstd()
    return zstd.ZstdCompressor(level=level, write_content_size=True).compress(payload)


def _decompress_frame(frame: bytes, max_output_size: int) -> bytes:
    _require_zstd()
    return zstd.ZstdDecompressor().decompress(frame, max_output_size=max_output_size)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class FAFBSkeletonBundle:
    """Appendable columnar reader with optional ZIP fallback + lazy conversion.

    Thread-safe for readers; appends are guarded by a single-writer file
    lock.  When ``zip_path`` is given and ``lazy_convert`` is true, ids
    missing from the container are served from the ZIP and converted into
    the bundle (buffered into ~8 MiB blocks, flushed on ``flush``/``close``
    and when the buffer fills).  ``overlay_path`` adds the Codex
    skeleton-cache overlay: members win over every other source for their
    ids (see the module docstring).
    """

    def __init__(self, bundle_path, zip_path: Optional[Path] = None,
                 lazy_convert: bool = False, level: int = DEFAULT_LEVEL,
                 block_bytes: int = BLOCK_BYTES,
                 overlay_path: Optional[Path] = None):
        """``bundle_path=None`` selects zip-only mode: the ZIP is served
        directly and no container file is created or appended to.
        ``overlay_path`` adds the Codex skeleton-cache overlay: a small zip
        of ``{bodyId}.swc`` members (lazily fetched from FlyWire Codex, or
        locally pruned repairs) that wins over the base ZIP/container for
        the ids it contains; the base files stay byte-untouched."""
        self.bundle_path = Path(bundle_path) if bundle_path else None
        self.zip_path = Path(zip_path) if zip_path else None
        self.overlay_path = Path(overlay_path) if overlay_path else None
        self.lazy_convert = bool(lazy_convert)
        self.level = int(level)
        self.block_bytes = int(block_bytes)
        self._handle = None
        self._mm = None
        self._mm_len = -1        # mapped length (mmap.size() fstats on Unix)
        self._footer = None
        self._block_idx: List[Tuple[int, int, int, int]] = []
        self._neuron_idx: List[Tuple[int, int, int]] = []
        self._id_set_cache = None
        self._last_size = -1
        self._zip_handle = None
        self._zip_names = None
        self._overlay_handle = None
        self._overlay_names = None
        self._pending: List[Tuple[int, List]] = []
        self._pending_ids = set()
        self._pending_nodes = 0
        self._converted_since_compact = 0
        self._converted_bytes_since_compact = 0
        self._write_lock = __import__("threading").Lock()

    # ---------------- open / index ----------------

    def _ensure_open(self):
        if self.bundle_path is None or self._handle is not None:
            return
        if not self.bundle_path.exists():
            self.bundle_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bundle_path, "wb") as fh:
                _write_header(fh)
                # empty bundle: blocks start right after the fixed header
                _write_footer(fh, 16 + HEADER_LEN, 0, 0, 0, 0, 0, 0, 0)
        self._handle = open(self.bundle_path, "r+b")
        self._mm = mmap.mmap(self._handle.fileno(), 0)
        self._mm_len = os.path.getsize(self.bundle_path)
        self._reload_index(force=True)

    def _reload_index(self, force: bool = False):
        if self._handle is None:
            self._ensure_open()
            return
        size = os.path.getsize(self.bundle_path)
        if not force and size == self._last_size:
            return
        # the mmap was sized at open time; grow it with the file.  Note that
        # mmap.size() re-fstats the fd on Unix and reports the *current* file
        # size, so the mapped extent is tracked explicitly in _mm_len.
        if self._mm is not None and size > self._mm_len:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = mmap.mmap(self._handle.fileno(), 0)
            self._mm_len = size
        self._last_size = size
        self._footer = _read_footer(self._handle)
        self._block_idx, self._neuron_idx = _read_indexes(self._handle, self._footer)
        self._id_set_cache = None

    def _ids(self) -> set:
        self._ensure_open()
        if self._id_set_cache is None:
            self._id_set_cache = {body_id for body_id, _, _ in self._neuron_idx}
        return self._id_set_cache

    def _zip(self) -> Optional[zipfile.ZipFile]:
        if self.zip_path is None or not self.zip_path.exists():
            return None
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.zip_path, "r")
            self._zip_names = {
                int(name[:-4]) for name in self._zip_handle.namelist()
                if name.endswith(".swc")
            }
        return self._zip_handle

    def _zip_ids(self) -> set:
        self._zip()
        return set(self._zip_names or ())

    def _overlay_zip(self) -> Optional[zipfile.ZipFile]:
        if self.overlay_path is None or not self.overlay_path.exists():
            return None
        if self._overlay_handle is None:
            self._overlay_handle = zipfile.ZipFile(self.overlay_path, "r")
            self._overlay_names = {
                int(name[:-4]) for name in self._overlay_handle.namelist()
                if name.endswith(".swc")
            }
        return self._overlay_handle

    def _overlay_ids(self) -> set:
        self._overlay_zip()
        return set(self._overlay_names or ())

    # ---------------- public API ----------------

    def get(self, body_id) -> Optional[str]:
        """SWC text for a body id (overlay first; then .zst; then ZIP)."""
        body_id = int(body_id)
        overlay = self._overlay_zip()
        if overlay is not None and body_id in (self._overlay_names or ()):
            rows = _parse_swc_rows(overlay.read(f"{body_id}.swc"))
            return _swc_text_from_rows(rows)
        self._ensure_open()
        with self._write_lock:
            self._reload_index()
            text = self._read_from_index(body_id)
        if text is not None:
            return text
        z = self._zip()
        if z is None or body_id not in (self._zip_names or ()):
            return None
        rows = _parse_swc_rows(z.read(f"{body_id}.swc"))
        text = _swc_text_from_rows(rows)
        if self.lazy_convert:
            self._buffer_convert(body_id, rows)
        return text

    def _read_from_index(self, body_id: int) -> Optional[str]:
        lo, hi = 0, len(self._neuron_idx)
        while lo < hi:
            mid = (lo + hi) // 2
            bid, block, row = self._neuron_idx[mid]
            if bid < body_id:
                lo = mid + 1
            elif bid > body_id:
                hi = mid
            else:
                return self._text_at(block, row)
        return None

    def _text_at(self, block_idx: int, row: int) -> str:
        off, comp, raw, n_neurons = self._block_idx[block_idx]
        payload = _decompress_frame(
            self._mm[off:off + comp], max_output_size=raw)
        n_nodes = (raw - 4 * (n_neurons + 1)) // NODE_BYTES
        row_offsets, columns = _unpack_block(payload, n_neurons, n_nodes)
        rows = [
            (columns["node_id"][i], columns["type"][i],
             columns["x"][i], columns["y"][i], columns["z"][i],
             columns["radius"][i], columns["parent_id"][i])
            for i in range(row_offsets[row], row_offsets[row + 1])
        ]
        return _swc_text_from_rows(rows)

    def contains(self, body_id) -> bool:
        body_id = int(body_id)
        if body_id in self._overlay_ids():
            return True
        self._ensure_open()
        with self._write_lock:
            self._reload_index()
        if body_id in self._ids():
            return True
        return body_id in self._zip_ids() if self._zip() else False

    def ids(self) -> set:
        self._ensure_open()
        with self._write_lock:
            self._reload_index()
        ids = set(self._ids())
        ids.update(self._overlay_ids())
        ids.update(self._zip_ids())
        return ids

    def count(self) -> int:
        return len(self.ids())

    def bundle_count(self) -> int:
        """Neurons stored in the .zst container itself (excludes the ZIP)."""
        self._ensure_open()
        with self._write_lock:
            self._reload_index()
        return len(self._ids())

    def __len__(self) -> int:
        return self.count()

    def iter_texts(self):
        """Bulk sequential text iteration: bundle blocks, then ZIP leftovers."""
        self._ensure_open()
        with self._write_lock:
            self._reload_index()
        for body_id, block, row in self._neuron_idx:
            yield body_id, self._text_at(block, row)
        z = self._zip()
        if z is None:
            return
        converted = self._ids()
        for body_id in sorted(self._zip_names or ()):
            if body_id in converted:
                continue
            rows = _parse_swc_rows(z.read(f"{body_id}.swc"))
            text = _swc_text_from_rows(rows)
            if self.lazy_convert:
                self._buffer_convert(body_id, rows)
            yield body_id, text

    # ---------------- lazy conversion ----------------

    def _buffer_convert(self, body_id: int, rows: List):
        if body_id in self._ids() or body_id in self._pending_ids:
            return
        if self._pending_nodes + len(rows) >= self.block_bytes // NODE_BYTES:
            self.flush()
        self._pending.append((body_id, rows))
        self._pending_ids.add(body_id)
        self._pending_nodes += len(rows)
        self._converted_since_compact += 1
        self._converted_bytes_since_compact += sum(
            len(str(v)) for row in rows for v in row)
        if (self.zip_path is not None
                and (self._converted_since_compact >= COMPACT_THRESHOLD_ENTRIES
                     or self._converted_bytes_since_compact >= COMPACT_THRESHOLD_BYTES)):
            self.flush()
            self.compact_zip(best_effort=True)

    def flush(self):
        """Append any buffered converted neurons as one block (locked)."""
        with self._write_lock:
            if not self._pending or self.bundle_path is None:
                return
            pending, self._pending = self._pending, []
            self._pending_ids = set()
            self._pending_nodes = 0
            self._append_neurons_locked(pending)

    def _append_neurons_locked(self, neurons: List[Tuple[int, List]]):
        payload, n_nodes, offsets = _pack_block(neurons)
        frame = _compress_frame(payload, self.level)
        with _bundle_file_lock(self.bundle_path):
            self._reload_index(force=True)
            block_off = self._footer["block_off"]
            self._handle.seek(block_off)
            self._handle.write(frame)
            new_block_off = block_off + len(frame)
            block_table = list(self._block_idx) + [
                (block_off, len(frame), len(payload), len(neurons))]
            neuron_entries = list(self._neuron_idx) + [
                (body_id, len(self._block_idx), row)
                for body_id, row in offsets]
            block_raw, neuron_raw = _build_indexes(block_table, neuron_entries)
            self._handle.seek(new_block_off)
            self._handle.write(block_raw)
            self._handle.write(neuron_raw)
            _write_footer(
                self._handle, new_block_off, len(block_raw),
                new_block_off + len(block_raw), len(neuron_raw),
                len(block_table), len(neuron_entries),
                zlib.crc32(block_raw) & 0xFFFFFFFF,
                zlib.crc32(neuron_raw) & 0xFFFFFFFF)
            self._handle.flush()
        # refresh the in-memory index so subsequent reads see the append
        self._last_size = -1
        self._reload_index(force=True)

    # ---------------- compaction ----------------

    def compact_zip(self, best_effort: bool = False) -> int:
        """Physically drop converted entries from the ZIP (verbatim copy)."""
        if self.zip_path is None or not self.zip_path.exists():
            return 0
        self._ensure_open()
        with self._write_lock:
            self._reload_index(force=True)
            converted = self._ids()
        if not converted:
            return 0
        self._converted_since_compact = 0
        self._converted_bytes_since_compact = 0
        # Windows cannot atomically replace a file that this process still
        # holds open: the lazy ZIP handle opened by an earlier read/append
        # causes os.replace() in _compact_zip_verbatim to fail with a sharing
        # violation (PermissionError WinError 5). Drop the handle first;
        # _zip() reopens lazily on the next use.
        if self._zip_handle is not None:
            try:
                self._zip_handle.close()
            except Exception:
                pass
            self._zip_handle = None
            self._zip_names = None
        return _compact_zip_verbatim(self.zip_path, converted, best_effort=best_effort)

    # ---------------- lifecycle ----------------

    def close(self):
        try:
            self.flush()
        except Exception:
            pass
        for handle in (self._zip_handle, self._overlay_handle, self._handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
        self._zip_handle = self._handle = self._mm = None
        self._mm_len = -1


class _bundle_file_lock:
    """Single-writer advisory lock for appends (POSIX flock, best effort)."""

    def __init__(self, bundle_path: Path):
        self._path = Path(str(bundle_path) + ".lock")
        self._fd = None

    def __enter__(self):
        try:
            import fcntl
            self._fd = open(self._path, "a+")
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except Exception:
            self._fd = None
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fd.close()
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Bulk packer (parallel over entry shards)
# ---------------------------------------------------------------------------

def _pack_shard(args) -> List[dict]:
    """Worker: parse ZIP entries [start, end) and build compressed blocks."""
    zip_path, start, end, level, block_bytes = args
    blocks: List[dict] = []
    current: List[Tuple[int, List]] = []
    current_nodes = 0
    max_error = 0.0
    max_nodes = block_bytes // NODE_BYTES

    def flush_block():
        nonlocal current, current_nodes
        if not current:
            return
        payload, n_nodes, offsets = _pack_block(current)
        blocks.append({
            "frame": _compress_frame(payload, level),
            "raw_len": len(payload), "entries": offsets,
        })
        current = []
        current_nodes = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        for name in names[start:end]:
            if not name.endswith(".swc"):
                continue
            rows = _parse_swc_rows(z.read(name))
            if not rows:
                continue
            max_error = max(max_error, _rows_round_trip_error(rows))
            if current_nodes + len(rows) > max_nodes:
                flush_block()
            current.append((int(name[:-4]), rows))
            current_nodes += len(rows)
    flush_block()
    return {"blocks": blocks, "max_error": max_error}


def pack(zip_path, out_path, level: int = DEFAULT_LEVEL,
         block_bytes: int = BLOCK_BYTES, n_workers: int = 0,
         progress_callback=None) -> Dict[str, object]:
    """Bulk conversion: stream the healed ZIP into the columnar bundle."""
    _require_zstd()
    zip_path = Path(zip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.endswith(".swc")]
    n_entries = len(names)
    workers = int(n_workers) if int(n_workers) > 0 else min(
        8, os.cpu_count() or 1, max(1, n_entries // 1000))
    shard = max(1, -(-n_entries // workers))
    tasks = [
        (str(zip_path), start, min(start + shard, n_entries), int(level),
         int(block_bytes))
        for start in range(0, n_entries, shard)
    ]

    # R7-2: temp-sibling naming so an interrupted pack is reclaimable by
    # remove_stale_temp_files instead of a permanent naked .tmp orphan.
    try:
        from .utils.parquet_utils import (
            remove_stale_temp_files, temp_sibling)
    except ImportError:
        from utils.parquet_utils import remove_stale_temp_files, temp_sibling
    remove_stale_temp_files(out_path, 'bundle-pack')
    # temp_sibling returns str by contract; this function calls Path methods
    # on it (unlink(missing_ok=True) on the reject path).
    tmp_path = Path(temp_sibling(out_path, 'bundle-pack'))
    max_error = 0.0
    block_table: List[Tuple[int, int, int, int]] = []
    neuron_entries: List[Tuple[int, int, int]] = []
    with open(tmp_path, "wb") as out:
        _write_header(out)
        if workers > 1:
            import multiprocessing as mp
            ctx = (mp.get_context("fork")
                   if hasattr(mp, "get_context")
                   and "fork" in mp.get_all_start_methods()
                   else mp.get_context("spawn"))
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                for future in [ex.submit(_pack_shard, task) for task in tasks]:
                    result = future.result()
                    max_error = max(max_error, result["max_error"])
                    for block in result["blocks"]:
                        off = out.tell()
                        out.write(block["frame"])
                        block_table.append((
                            off, len(block["frame"]), block["raw_len"],
                            len(block["entries"])))
                        neuron_entries.extend(
                            (body_id, len(block_table) - 1, row)
                            for body_id, row in block["entries"])
                    if progress_callback:
                        progress_callback(len(block_table), len(tasks))
        else:
            for task in tasks:
                result = _pack_shard(task)
                max_error = max(max_error, result["max_error"])
                for block in result["blocks"]:
                    off = out.tell()
                    out.write(block["frame"])
                    block_table.append((
                        off, len(block["frame"]), block["raw_len"],
                        len(block["entries"])))
                    neuron_entries.extend(
                        (body_id, len(block_table) - 1, row)
                        for body_id, row in block["entries"])
                if progress_callback:
                    progress_callback(len(block_table), len(tasks))

        if max_error >= MAX_COORD_ERROR_NM:
            out.close()
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"float32 round-trip error {max_error:.4f} nm exceeds the "
                f"{MAX_COORD_ERROR_NM} nm gate; refusing to build")
        block_raw, neuron_raw = _build_indexes(block_table, neuron_entries)
        block_off = out.tell()
        out.write(block_raw)
        out.write(neuron_raw)
        _write_footer(
            out, block_off, len(block_raw), block_off + len(block_raw),
            len(neuron_raw), len(block_table), len(neuron_entries),
            zlib.crc32(block_raw) & 0xFFFFFFFF,
            zlib.crc32(neuron_raw) & 0xFFFFFFFF)
        out.flush()
        # rewrite the fixed-size header with the measured precision
        out.seek(0)
        _write_header(out, max_error=max_error)
        out.flush()
    os.replace(tmp_path, out_path)
    return {
        "entries": n_entries,
        "blocks": len(block_table),
        "neurons": len(neuron_entries),
        "max_coord_error_nm": max_error,
        "bytes": os.path.getsize(out_path),
    }


# ---------------------------------------------------------------------------
# Bulk lazy append + verbatim ZIP compaction
# ---------------------------------------------------------------------------

def append_entries(zip_path, bundle_path, ids) -> int:
    """Convert the given ids from the ZIP into the bundle (bulk lazy mode)."""
    bundle = FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                lazy_convert=False)
    try:
        bundle._ensure_open()
        with bundle._write_lock:
            bundle._reload_index(force=True)
            converted = bundle._ids()
        with zipfile.ZipFile(zip_path, "r") as z:
            names = set(z.namelist())
            appended = 0
            for body_id in ids:
                body_id = int(body_id)
                if body_id in converted or f"{body_id}.swc" not in names:
                    continue
                rows = _parse_swc_rows(z.read(f"{body_id}.swc"))
                if not rows:
                    continue
                bundle._pending.append((body_id, rows))
                bundle._pending_nodes += len(rows)
                appended += 1
                if bundle._pending_nodes >= bundle.block_bytes // NODE_BYTES:
                    bundle.flush()
        bundle.flush()
        return appended
    finally:
        bundle.close()


def _compact_zip_verbatim(zip_path: Path, converted: set,
                          best_effort: bool = False) -> int:
    """Rewrite the ZIP keeping only entries not in ``converted``.

    Copies raw local headers and compressed bytes verbatim (no
    recompression), then writes a fresh central directory + EOCD.  Atomic
    temp + rename; returns the number of entries removed.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zin:
            infos = zin.infolist()
            keep = [info for info in infos
                    if info.filename.endswith(".swc")
                    and int(info.filename[:-4]) not in converted]
            removed = len(infos) - len(keep)
            if removed == 0:
                return 0
            keep.sort(key=lambda info: info.header_offset)
            tmp_path = zip_path.with_name(zip_path.name + ".compact.tmp")
            with open(tmp_path, "wb") as out, open(zip_path, "rb") as src:
                new_offsets = {}
                for info in keep:
                    new_offsets[id(info)] = out.tell()
                    src.seek(info.header_offset)
                    local = src.read(30)
                    name_len, extra_len = struct.unpack_from("<HH", local, 26)
                    # local header + filename + extra, then the raw data
                    src.seek(info.header_offset)
                    out.write(src.read(30 + name_len + extra_len))
                    out.write(src.read(info.compress_size))
                central = []
                for info in keep:
                    name = info.filename.encode("utf-8")
                    extra = info.extra or b""
                    comment = info.comment or b""
                    dt = info.date_time
                    dosdate = ((dt[0] - 1980) << 9) | (dt[1] << 5) | dt[2]
                    dostime = (dt[3] << 11) | (dt[4] << 5) | (dt[5] // 2)
                    central.append(struct.pack(
                        "<4s4B4HL2L5H2L",
                        b"PK\x01\x02",
                        max(20, info.create_version), info.create_system,
                        max(20, info.extract_version), info.reserved,
                        info.flag_bits, info.compress_type, dostime, dosdate,
                        info.CRC & 0xFFFFFFFF, info.compress_size,
                        info.file_size, len(name), len(extra), len(comment),
                        0, info.internal_attr & 0xFFFF,
                        info.external_attr & 0xFFFFFFFF, new_offsets[id(info)])
                        + name + extra + comment)
                cd_offset = out.tell()
                cd_data = b"".join(central)
                out.write(cd_data)
                out.write(struct.pack(
                    "<4s4H2LH", b"PK\x05\x06", 0, 0, len(central),
                    len(central), len(cd_data), cd_offset, 0))
                out.flush()
        # The ZipFile and the raw read handle above are both closed by now:
        # on Windows os.replace() cannot replace a file this process still
        # holds open (sharing violation), so it must run outside the with.
        # Windows Defender/indexer may briefly scan the freshly written tmp
        # zip; retry with a growing backoff (total ~14 s) before giving up.
        last_error = None
        for attempt in range(10):
            try:
                os.replace(tmp_path, zip_path)
                break
            except PermissionError as error:
                last_error = error
                if attempt < 9:
                    time.sleep(0.25 * (attempt + 1))
        else:
            raise last_error
        return removed
    except Exception:
        if best_effort:
            return 0
        raise


# ---------------------------------------------------------------------------
# Verify + info
# ---------------------------------------------------------------------------

def verify(bundle_path, zip_path=None, sample: int = 200) -> Dict[str, object]:
    """Re-read sampled ids and compare parsed nodes against the ZIP source."""
    import random
    import navis

    bundle = FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                lazy_convert=False)
    zf = zipfile.ZipFile(zip_path, "r") if zip_path else None
    zip_names = set(zf.namelist()) if zf is not None else set()
    try:
        ids = sorted(bundle.ids())
        picked = random.Random(42).sample(ids, min(sample, len(ids)))
        errors = 0
        max_delta = 0.0
        for body_id in picked:
            text = bundle.get(body_id)
            if text is None:
                errors += 1
                continue
            nrn = navis.read_swc(io.StringIO(text))
            if zf is not None:
                # ids already converted out of the ZIP are bundle-internal
                # consistency checks only (no ZIP source to compare)
                if f"{body_id}.swc" not in zip_names:
                    continue
                src = zf.read(f"{body_id}.swc")
                src_nrn = navis.read_swc(
                    io.StringIO(src.decode("utf-8", "replace")))
                nrn_nodes = nrn.nodes.reset_index(drop=True)
                src_nodes = src_nrn.nodes.reset_index(drop=True)
                if len(nrn_nodes) != len(src_nodes):
                    errors += 1
                    continue
                merged = nrn_nodes.merge(
                    src_nodes, on="node_id", suffixes=("", "_src"))
                for col in ("x", "y", "z", "radius"):
                    delta = (merged[col] - merged[col + "_src"]).abs()
                    max_delta = max(max_delta, float(delta.max()))
                    if (delta > 0.1).any():
                        errors += 1
                for col in ("parent_id", "type"):
                    if (merged[col] != merged[col + "_src"]).any():
                        errors += 1
        return {
            "ids": len(ids), "sampled": len(picked), "errors": errors,
            "max_node_delta_nm": max_delta, "ok": errors == 0,
        }
    finally:
        if zf is not None:
            zf.close()
        bundle.close()


def info(bundle_path, zip_path=None) -> Dict[str, object]:
    """Entry counts and sizes for the bundle and (optionally) the ZIP."""
    bundle_path = Path(bundle_path)
    result: Dict[str, object] = {"bundle_path": str(bundle_path)}
    if bundle_path.exists():
        bundle = FAFBSkeletonBundle(bundle_path, zip_path=zip_path,
                                    lazy_convert=False)
        try:
            result["bundle_bytes"] = os.path.getsize(bundle_path)
            result["bundle_neurons"] = bundle.bundle_count()
            result["total"] = bundle.count()  # union with ZIP leftovers
        finally:
            bundle.close()
    if zip_path is not None and Path(zip_path).exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            result["zip_entries"] = sum(
                1 for n in z.namelist() if n.endswith(".swc"))
        result["zip_bytes"] = os.path.getsize(zip_path)
        result["zip_path"] = str(zip_path)
    converted = int(result.get("bundle_neurons", 0))
    total = int(result.get("total", 0)) or converted + int(
        result.get("zip_entries", 0))
    if total:
        result["converted_percent"] = round(converted / total * 100, 1)
    return result


# ---------------------------------------------------------------------------
# Resolution + CLI
# ---------------------------------------------------------------------------

# Codex skeleton-cache overlay: lazily fetched FlyWire Codex SWCs (and
# locally pruned repairs) land here instead of mutating the pristine healed
# bundle. Members win over the base source; provenance lives in the JSON
# manifest because the bundle reader strips SWC comment lines.
OVERLAY_SKELETON_CACHE_NAME = "codex_skeleton_cache.zip"
OVERLAY_MANIFEST_NAME = "codex_skeleton_cache.json"


def overlay_paths(data_dir) -> Tuple[Path, Path]:
    data_dir = Path(data_dir)
    return (data_dir / OVERLAY_SKELETON_CACHE_NAME,
            data_dir / OVERLAY_MANIFEST_NAME)


def read_overlay_manifest(data_dir) -> Dict[str, dict]:
    _, manifest_path = overlay_paths(data_dir)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_overlay_members(data_dir, members, origin: str) -> Path:
    """Write/replace ``{bodyId}.swc`` members in the overlay (atomic).

    ``members`` maps body id -> SWC text.  The overlay and its manifest are
    rewritten through temp siblings so an interrupted write never leaves a
    partial zip; the base healed bundle is never touched.
    """
    data_dir = Path(data_dir)
    overlay_path, manifest_path = overlay_paths(data_dir)
    existing: Dict[str, bytes] = {}
    if overlay_path.exists():
        with zipfile.ZipFile(overlay_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".swc"):
                    existing[name] = zf.read(name)
    for body_id, text in members.items():
        existing[f"{body_id}.swc"] = text.encode("utf-8")
    manifest = read_overlay_manifest(data_dir)
    prior_keys = set(manifest)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for body_id in members:
        manifest[str(body_id)] = {
            "origin": origin,
            "updated_at": now,
            "replaced": str(body_id) in prior_keys,
        }
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = overlay_path.with_name(f".{overlay_path.name}.tmp")
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(existing):
                zf.writestr(name, existing[name])
        os.replace(temp_zip, overlay_path)
    finally:
        temp_zip.unlink(missing_ok=True)
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temp_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True),
                                 encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
    finally:
        temp_manifest.unlink(missing_ok=True)
    return overlay_path


def open_bundle(data_dir) -> Optional[FAFBSkeletonBundle]:
    """Resolve the FAFB skeleton source: healed ZIP served directly.

    Zip-only mode: no ``.zst`` container is created or appended to.  An
    existing ``.zst`` is opened read-only only when no ZIP is available,
    so .zst-only setups keep working.  A Codex skeleton-cache overlay
    (``codex_skeleton_cache.zip``) is attached when present; its members
    win over the base source and the base files stay byte-untouched.
    """
    data_dir = Path(data_dir)
    dataset_name = data_dir.name
    bundle_candidates = (
        data_dir / "sk_lod1_783_healed.zst",
        data_dir / f"{dataset_name}_skeletons.zst",
    )
    zip_candidates = (
        data_dir / "sk_lod1_783_healed.zip",
        data_dir / f"{dataset_name}_skeletons.zip",
        data_dir / "downloads" / "sk_lod1_783_healed.zip",
    )
    overlay_path = data_dir / OVERLAY_SKELETON_CACHE_NAME
    overlay = overlay_path if overlay_path.is_file() else None
    zip_path = next((p for p in zip_candidates if p.is_file()), None)
    if zip_path is not None:
        return FAFBSkeletonBundle(None, zip_path=zip_path, lazy_convert=False,
                                  overlay_path=overlay)
    for candidate in bundle_candidates:
        if candidate.is_file():
            return FAFBSkeletonBundle(candidate, zip_path=None,
                                      lazy_convert=False,
                                      overlay_path=overlay)
    if overlay is not None:
        # Overlay only (no base bundle yet): still serve what was fetched.
        return FAFBSkeletonBundle(None, zip_path=None, lazy_convert=False,
                                  overlay_path=overlay)
    return None


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.fafb_bundle",
        description="FAFB healed skeleton bundle (.zst) tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pack = sub.add_parser("pack", help="bulk-convert the healed ZIP to .zst")
    p_pack.add_argument("zip")
    p_pack.add_argument("out")
    p_pack.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    p_pack.add_argument("--block-bytes", type=int, default=BLOCK_BYTES)
    p_pack.add_argument("--workers", type=int, default=0)
    p_pack.add_argument("--delete-source", action="store_true",
                        help="delete the ZIP only after verify passes")

    p_verify = sub.add_parser("verify", help="round-trip checks vs the ZIP")
    p_verify.add_argument("bundle")
    p_verify.add_argument("--zip", default=None)
    p_verify.add_argument("--sample", type=int, default=200)

    p_info = sub.add_parser("info", help="entry counts and sizes")
    p_info.add_argument("bundle")
    p_info.add_argument("--zip", default=None)

    p_compact = sub.add_parser(
        "compact", help="physically drop converted entries from the ZIP")
    p_compact.add_argument("zip")
    p_compact.add_argument("bundle")

    p_append = sub.add_parser(
        "append", help="convert specific ids from ZIP to bundle")
    p_append.add_argument("zip")
    p_append.add_argument("bundle")
    p_append.add_argument("ids", nargs="+")

    args = parser.parse_args()
    if args.command == "pack":
        stats = pack(args.zip, args.out, level=args.level,
                     block_bytes=args.block_bytes, n_workers=args.workers)
        print(json.dumps(stats, indent=2))
        if args.delete_source:
            result = verify(args.out, zip_path=args.zip, sample=200)
            if result["ok"]:
                Path(args.zip).unlink()
                print("source ZIP deleted after successful verify")
            else:
                print(f"verify failed: {result}; ZIP kept")
                return 1
    elif args.command == "verify":
        result = verify(args.bundle, zip_path=args.zip, sample=args.sample)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    elif args.command == "info":
        print(json.dumps(info(args.bundle, zip_path=args.zip), indent=2))
    elif args.command == "compact":
        bundle = FAFBSkeletonBundle(args.bundle, zip_path=args.zip,
                                    lazy_convert=False)
        try:
            removed = bundle.compact_zip()
        finally:
            bundle.close()
        print(f"removed {removed} converted entries")
    elif args.command == "append":
        appended = append_entries(args.zip, args.bundle, args.ids)
        print(f"appended {appended} entries")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
