"""Lossless Parquet re-encoding for the stored synapse tables.

The synapse tables are the largest files DROCAT keeps on disk.  Their
storage layout can be improved without touching a single value: Parquet
encodings (DELTA_BINARY_PACKED, BYTE_STREAM_SPLIT, DELTA_LENGTH_BYTE_ARRAY)
and conservative dtype narrowing are decoded transparently by every
reader, so a streaming rewrite that applies them is a pure byte reclaim.

``reencode_parquet_lossless`` applies a caller-supplied profile in one
streaming pass (optionally dropping dead columns at the same time), with
every narrowing decision guarded by footer statistics so a cast is only
taken when the data provably fits.  The rewrite goes to a temporary
sibling file and atomically replaces the original only after the complete
copy exists and its footer verifies, so an interrupted run never damages
the source; a footer key-value marker makes the pass idempotent.

The atomic write is centralized in ``write_parquet_atomic`` (write temp →
fsync → ``os.replace`` → fsync the directory) so every caller shares one
implementation, and stale temps are reclaimed when their writer is provably
gone — on every platform, including Windows — so an interrupted run's
leftovers do not sit around until a grace period expires.  A writer that is
still alive is never touched, however old its temp looks; only a temp whose
PID has demonstrably been recycled is reclaimed that way.
"""

# Round-6 finding F-D1: this module EMITS the non-ASCII status glyphs
# (✓ / ⚠️) but never installed the UTF-8 stdio guard its siblings do, so a
# SUCCESSFUL, already-committed re-encode aborted with UnicodeEncodeError on
# a cp936 console whenever the entry point (e.g. FAFB_file_converter or
# banc_public_data as __main__) did not import coana first. Installing the
# guard here covers every entry point; it is idempotent and never raises.
try:
    from .console_encoding import ensure_utf8_stdio
except ImportError:
    try:
        from utils.console_encoding import ensure_utf8_stdio
    except ImportError:
        ensure_utf8_stdio = None
if ensure_utf8_stdio is not None:
    ensure_utf8_stdio()

import os
import time

LOSSLESS_MARKER_KEY = b"DROCAT.lossless"
LOSSLESS_MARKER_VALUE = b"v1"

# Only read at call time by `_process_alive`; a module constant so the
# Windows branch is testable without mutating `os.name` process-wide.
_POSIX = os.name == "posix"

# When writer liveness cannot be determined at all, a temp older than this is
# reclaimed as a crash leftover.  A *live* writer is exempt (see
# ``_temp_is_stale``), so this only ever decides about unanswerable cases.
DEFAULT_TEMP_MAX_AGE_SECONDS = 6 * 60 * 60

# Hard integer bounds for the stats guard of `casts`.
_INT_DTYPE_BOUNDS = {
    "int8": (-2**7, 2**7 - 1),
    "int16": (-2**15, 2**15 - 1),
    "int32": (-2**31, 2**31 - 1),
    "int64": (-2**63, 2**63 - 1),
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
}


def temp_sibling(path, kind):
    """Return the atomic-rewrite temp path for *path*.

    Single source of truth for the naming scheme: a hidden sibling
    ``.{name}.{kind}.<pid>.tmp`` that ``remove_stale_temp_files`` matches.
    """
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    return os.path.join(directory, f".{name}.{kind}.{os.getpid()}.tmp")


def _temp_pid(entry, prefix):
    """PID encoded in a ``{prefix}<pid>.tmp`` name, or None if it is not one."""
    if not (entry.startswith(prefix) and entry.endswith(".tmp")):
        return None
    try:
        return int(entry[len(prefix):-len(".tmp")])
    except ValueError:
        return None


def _process_alive(pid):
    """True/False when writer liveness is knowable, None when it is not.

    POSIX ``kill(pid, 0)`` probes without signalling: a missing process
    raises ``ProcessLookupError`` and a foreign owner raises
    ``PermissionError``.  Windows has no ``kill``, so ``psutil`` answers the
    same question; without it (or off-platform) liveness is unknown and the
    age rule decides.
    """
    if _POSIX:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.pid_exists(pid)
    except OSError:
        return None


def _pid_recycled(pid, path):
    """True when process *pid* was born after *path* was last written.

    A live PID is not proof of a live *writer*: reuse hands a dead writer's
    temp to an unrelated process, which would otherwise protect the garbage
    forever now that age alone no longer reclaims it.  False when psutil is
    missing or the question cannot be answered, which keeps the temp.
    """
    try:
        import psutil
    except ImportError:
        return False
    try:
        return psutil.Process(pid).create_time() > os.path.getmtime(path)
    except (ValueError, OSError, psutil.Error):
        return False


def _temp_is_stale(path, entry_pid, max_age_seconds):
    if entry_pid == os.getpid():
        return False  # our own in-progress temp
    alive = _process_alive(entry_pid)
    if alive is False:
        return True
    if alive is True:
        # Live, but only when it can be the writer: an in-progress temp is
        # never aged out, so PID recycling is the one way to release it.
        return _pid_recycled(entry_pid, path)
    if max_age_seconds is not None:
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        if age > max_age_seconds:
            return True
    # Liveness unknown with no age evidence: keep it.
    return False


def remove_stale_temp_files(path, kind, max_age_seconds=DEFAULT_TEMP_MAX_AGE_SECONDS):
    """Delete leftover ``.{name}.{kind}.<pid>.tmp`` siblings of *path*.

    An interrupted run (crash, SIGKILL) can never unlink its temporary
    file, so before starting a new rewrite the siblings left behind are
    removed.  A sibling is reclaimed only when its writer is provably
    dead or the file is older than *max_age_seconds*; the current
    process's own temp is never touched, and a concurrent preparation's
    in-progress temp is preserved.
    """
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    prefix = f".{name}.{kind}."
    try:
        entries = os.listdir(directory)
    except OSError:
        return
    for entry in entries:
        entry_pid = _temp_pid(entry, prefix)
        if entry_pid is None:
            continue
        candidate = os.path.join(directory, entry)
        if _temp_is_stale(candidate, entry_pid, max_age_seconds):
            try:
                os.unlink(candidate)
            except OSError:
                pass


def _declared_data_extent(metadata):
    """Largest byte offset any column chunk declares it occupies.

    Uses only the footer metadata already parsed by the caller, so this
    costs no data-page reads.
    """
    extent = 0
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for column in range(row_group.num_columns):
            chunk = row_group.column(column)
            starts = [
                offset for offset in (
                    getattr(chunk, "data_page_offset", None),
                    getattr(chunk, "dictionary_page_offset", None),
                ) if offset and offset > 0
            ]
            if not starts:
                continue
            total = getattr(chunk, "total_compressed_size", 0) or 0
            extent = max(extent, min(starts) + total)
    return extent


def parquet_readable(path):
    """True when *path* parses as a complete, non-empty parquet file.

    Only reads the footer, so this is cheap enough to run before every
    reuse.  A file truncated by an interrupted write has no valid footer,
    and an empty table is never a valid release, so both fail here and
    callers rebuild instead of trusting them.

    The footer can also parse while the body was lost — e.g. a partial
    copy that happened to keep the tail, leaving column-chunk offsets
    pointing past EOF.  That is caught here by requiring the file to be
    at least as long as the data the footer declares, without reading any
    data pages.  In-place corruption of *equal* length is not detectable
    from metadata and would still surface on a page read.
    """
    try:
        import pyarrow.parquet as pq

        with pq.ParquetFile(path) as reader:
            metadata = reader.metadata
        if metadata.num_row_groups < 1 or metadata.num_rows <= 0:
            return False
        extent = _declared_data_extent(metadata)
        if extent and os.path.getsize(path) < extent:
            return False
        return True
    except Exception:
        return False


def parquet_is_reusable(path):
    """True when *path* is a complete parquet file that may be reused.

    A file that exists but is truncated/unreadable (an interrupted write
    from before the atomic writers landed) is removed so the caller
    rebuilds it rather than trusting ``os.path.exists``.  Missing files
    return False without side effects.
    """
    if not os.path.exists(path):
        return False
    if parquet_readable(path):
        return True
    try:
        os.remove(path)
    except OSError:
        pass
    return False


def parquet_lossless_marker(path):
    """True when *path* carries the lossless-encoding footer marker."""
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata or {}
        return metadata.get(LOSSLESS_MARKER_KEY) == LOSSLESS_MARKER_VALUE
    except Exception:
        return False


def atomic_replace(temp, final):
    """Durably move *temp* onto *final*.

    ``os.replace`` makes the swap atomic against crashes, but without an
    fsync the OS may still hold the data in its buffer, so a power loss
    can leave *final* partial or zero-length.  Flush the file, replace,
    then best-effort flush the directory entry so the rename survives too
    (directory fsync is a no-op/unsupported on some platforms).
    """
    # The flush handle must be writable: os.fsync on a read-only
    # descriptor raises [Errno 9] on Windows.  A failed flush is
    # non-fatal -- os.replace below is what makes the swap atomic.
    try:
        with open(temp, "r+b") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(temp, final)
    try:
        dir_fd = os.open(os.path.dirname(os.path.abspath(final)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def write_file_atomic(final_path, write_fn, kind="build",
                      max_age_seconds=DEFAULT_TEMP_MAX_AGE_SECONDS):
    """Write a file durably and atomically.

    ``write_fn(temp_path)`` must write a complete file to the given path
    (any format).  Stale temps are reclaimed first, the write goes to a
    temp sibling, and the result replaces *final_path* through
    ``atomic_replace``.  On any failure the temp file is removed and
    *final_path* is left untouched, so an interrupted write never leaves
    a truncated file that a later ``os.path.exists`` would trust.
    """
    remove_stale_temp_files(final_path, kind, max_age_seconds)
    temp = temp_sibling(final_path, kind)
    try:
        write_fn(temp)
        atomic_replace(temp, final_path)
    except Exception:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return final_path


# Parquet callers read best with the format-specific name.
write_parquet_atomic = write_file_atomic


def _column_bounds(parquet_file, column):
    """Global (min, max) of *column* from footer statistics, else None."""
    names = parquet_file.schema_arrow.names
    if column not in names:
        return None
    index = names.index(column)
    lo = hi = None
    for group in range(parquet_file.metadata.num_row_groups):
        stats = parquet_file.metadata.row_group(group).column(index).statistics
        if stats is None or not stats.has_min_max:
            return None
        lo = stats.min if lo is None else min(lo, stats.min)
        hi = stats.max if hi is None else max(hi, stats.max)
    return lo, hi


def _resolved_casts(parquet_file, schema, keep, casts):
    """Keep only the casts whose columns provably fit the target dtype."""
    import pyarrow as pa

    resolved = {}
    for column, target in (casts or {}).items():
        if column not in keep:
            continue
        bounds = _INT_DTYPE_BOUNDS.get(str(target))
        if bounds is None:
            continue
        if not pa.types.is_integer(schema.field(column).type):
            continue  # narrowing is only defined here for integer sources
        observed = _column_bounds(parquet_file, column)
        if observed is None:
            continue  # statistics missing: never gamble on a narrowing
        lo, hi = observed
        if lo is None or hi is None:
            continue
        if lo < bounds[0] or hi > bounds[1]:
            continue
        resolved[column] = target
    return resolved


def _encoding_fits(arrow_type, encoding):
    """Whether *encoding* is valid for *arrow_type*.

    Applying an incompatible encoding aborts the whole write, so profile
    entries that do not fit the actual column type are dropped instead
    (e.g. a string-only encoding listed for a dataset generation that
    stored the same ids as integers).
    """
    import pyarrow as pa

    if encoding == "DELTA_LENGTH_BYTE_ARRAY":
        return pa.types.is_string(arrow_type) or pa.types.is_binary(
            arrow_type) or pa.types.is_large_string(arrow_type)
    if encoding == "DELTA_BINARY_PACKED":
        return pa.types.is_integer(arrow_type)
    if encoding == "BYTE_STREAM_SPLIT":
        return (pa.types.is_integer(arrow_type)
                or pa.types.is_floating(arrow_type))
    return True  # unknown encodings are left to the writer to validate


def _resolved_encodings(schema, keep, encodings):
    """Keep only the encoding entries that fit their column's type."""
    resolved = {}
    for column, encoding in (encodings or {}).items():
        if column not in keep:
            continue
        if _encoding_fits(schema.field(column).type, encoding):
            resolved[column] = encoding
    return resolved


def reencode_parquet_lossless(path, profile, progress_callback=None):
    """Rewrite the parquet at *path* with a lossless storage profile.

    The profile may name ``drop_columns`` (applied in the same pass),
    stats-guarded integer ``casts`` ({column: 'int32', ...}),
    per-column ``encodings``, ``use_dictionary``, ``compression`` and
    ``compression_level``.  Values are preserved exactly: encodings are
    transparent on read and narrowing happens only when the footer
    statistics prove every value fits.

    Returns True when the file was rewritten, False when it already
    carries the lossless footer marker (or had nothing left to keep), or
    when the rewrite failed — the original is never destroyed on failure.
    """
    try:
        remove_stale_temp_files(path, "compact")
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        if (schema.metadata or {}).get(
                LOSSLESS_MARKER_KEY) == LOSSLESS_MARKER_VALUE:
            return False

        drop = [n for n in profile.get("drop_columns", ())
                if n in schema.names]
        keep = [n for n in schema.names if n not in drop]
        if not keep:
            return False

        original_size = os.path.getsize(path)
        temp = temp_sibling(path, "compact")
        try:
            # Both readers are scoped, not just tidy: on Windows an open
            # handle on the swap destination makes ``os.replace`` fail with
            # WinError 32, and one on the temp makes the failure-path unlink
            # fail too, so a silently skipped re-encode also leaked a temp.
            with pq.ParquetFile(path) as parquet_file:
                source_rows = parquet_file.metadata.num_rows
                resolved_casts = _resolved_casts(
                    parquet_file, schema, keep, profile.get("casts"))

                writer_fields = [schema.field(n) for n in keep]
                for column, target in resolved_casts.items():
                    writer_fields[keep.index(column)] = pa.field(
                        column, pa.type_for_alias(target))
                writer_schema = pa.schema(
                    writer_fields, metadata={
                        **(schema.metadata or {}),
                        LOSSLESS_MARKER_KEY: LOSSLESS_MARKER_VALUE,
                    })

                writer_kwargs = {
                    key: profile[key]
                    for key in ("compression", "compression_level",
                                "use_dictionary", "column_encoding")
                    if key in profile
                }
                if "column_encoding" in writer_kwargs:
                    # Drop entries for absent columns and encodings that do
                    # not fit the column's type (either aborts the whole write).
                    writer_kwargs["column_encoding"] = _resolved_encodings(
                        schema, keep, writer_kwargs["column_encoding"])
                    if not writer_kwargs["column_encoding"]:
                        writer_kwargs.pop("column_encoding")

                n_groups = parquet_file.metadata.num_row_groups
                with pq.ParquetWriter(temp, writer_schema,
                                      **writer_kwargs) as writer:
                    for group in range(n_groups):
                        table = parquet_file.read_row_group(
                            group, columns=keep)
                        for column, target in resolved_casts.items():
                            index = table.schema.get_field_index(column)
                            table = table.set_column(
                                index, table.schema.field(index).name,
                                table.column(index).cast(
                                    pa.type_for_alias(target)))
                        writer.write_table(table)
                        if progress_callback is not None:
                            progress_callback(group + 1, n_groups)
            # Gate before the swap: the copy must parse and carry every
            # source row.
            with pq.ParquetFile(temp) as written:
                if written.metadata.num_rows != source_rows:
                    raise OSError(
                        f"row count mismatch after re-encode "
                        f"({written.metadata.num_rows} != {source_rows})")
            atomic_replace(temp, path)
        except Exception:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

        reclaimed_gb = (original_size - os.path.getsize(path)) / 1e9
        print(f"  ✓ Losslessly re-encoded {os.path.basename(path)} "
              f"({reclaimed_gb:+.2f} GB).")
        return True
    except Exception as exc:
        print(f"  ⚠️ Lossless re-encode skipped for "
              f"{os.path.basename(path)}: {exc}")
        return False
