"""Cached neuron-index loading and querying for the UI viewer.

The auto-suggestion backend and the available-neurons viewer intentionally
share the same local index boundary: a viewer is available only when
``neuron_indexes/<dataset>/neuron_index.parquet`` exists.  That app-owned
"system files" directory persists across ``cache/`` cleanups - bundled
datasets ship committed seeds there and the pull pipeline builds every other
dataset's index into the same place.  The viewer never serves the raw dataset
file to the browser.  The index is built from the materialized projection of
the prepared local neuron table; an older/partial index can still be enriched
from that table to fill blank ``type``/``instance`` values.
"""

from __future__ import annotations

import atexit
import asyncio
import html
import logging
import multiprocessing
import re
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import date, datetime, time
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)
from .dataset_service import dataset_to_folder
from .search_logic import (
    SearchStage,
    is_numeric_search,
    normalize_search_operator,
    normalize_search_text,
    ordered_search_columns,
    polars_body_id_guard,
    polars_display_expression,
    polars_match_column_expression,
    polars_match_expression,
    search_plan,
)

try:
    from src.neuron_index_builder import (
        build_search_cache_frame,
        dataset_identifier_from_folder,
        is_search_cache_compatible,
        metadata_candidates as builder_metadata_candidates,
        metadata_columns,
        ordered_projection_columns,
        read_metadata_projection,
        search_cache_path,
        viewer_search_columns,
    )
except ImportError:
    from neuron_index_builder import (
        build_search_cache_frame,
        dataset_identifier_from_folder,
        is_search_cache_compatible,
        metadata_candidates as builder_metadata_candidates,
        metadata_columns,
        ordered_projection_columns,
        read_metadata_projection,
        search_cache_path,
        viewer_search_columns,
    )


@dataclass(frozen=True)
class CachedNeuronIndex:
    """A cached neuron index plus the path used to load it."""

    dataset: str
    path: Path
    frame: Any  # polars.DataFrame; kept Any so importing the UI does not require Polars eagerly
    columns: Tuple[str, ...]
    enriched: bool = False
    search_frame: Any = None  # compact presorted Polars search sidecar


@dataclass(frozen=True)
class NeuronIndexPage:
    """One server-side page of a filtered/sorted neuron index."""

    rows: List[Dict[str, Any]]
    total: int
    page: int
    pages: int
    page_size: int
    sort_by: str
    descending: bool
    match_groups: List[Dict[str, Any]] = field(default_factory=list)
    match_group_members: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    match_group_body_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Related match names share at least one exact result row.  The primary
    # values are the canonical query tokens for a selected related group.
    match_group_related: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    match_group_primary: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    focus_page: Optional[int] = None
    # Every matching ``__neuron_key`` -> user-facing bodyId across all pages.
    # Populated only when ``include_all_keys`` is requested, so normal paged
    # browsing never pays to materialise the full result set.
    all_keys: Dict[str, str] = field(default_factory=dict)


# The largest local indexes are only a few megabytes as Parquet but are read
# by several UI clients.  Keep one process-local copy and invalidate it when
# either the cache index or its optional metadata table changes.
_INDEX_CACHE: Dict[Tuple, CachedNeuronIndex] = {}
_INDEX_LOAD_LOCK = threading.RLock()

# Cross-dataset expansion is intentionally off the NiceGUI event loop, but
# that alone does not make it cheap: a cold expansion may initialize the type
# mapper and scan several large Parquet projections.  More than one viewer
# (or a stale request from the same viewer) can otherwise run that work at
# once and retain several independent Polars/Pandas working sets.  Keep this
# process-wide and re-entrant so callers can serialize a scan while also
# allowing the public collector below to use the same guard.
_CROSS_DATASET_SCAN_LOCK = threading.RLock()
CROSS_SCAN_SUPERSEDED = object()

# A cold type-mapper load is CPU- and memory-heavy even when called from a
# NiceGUI worker thread: pandas parsing and the mapper's Python graph build
# still contend with the event loop in the UI process.  Keep a separate,
# explicitly-spawned process for this workload.  One worker is intentional:
# two cold mapper instances can otherwise multiply the peak RSS and recreate
# the connection loss this guard is meant to prevent.  The worker stays alive
# after its first scan, so later searches reuse its warm mapper.
_CROSS_DATASET_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_CROSS_DATASET_PROCESS_POOL_LOCK = threading.Lock()
_CROSS_DATASET_PROCESS_ASYNC_LOCKS = weakref.WeakKeyDictionary()
_CROSS_DATASET_PROCESS_ASYNC_LOCKS_LOCK = threading.Lock()


def _cross_dataset_process_pool() -> ProcessPoolExecutor:
    """Return the one-process pool used by cold cross-dataset scans."""
    global _CROSS_DATASET_PROCESS_POOL
    with _CROSS_DATASET_PROCESS_POOL_LOCK:
        if _CROSS_DATASET_PROCESS_POOL is None:
            # ``spawn`` keeps the UI process's large Polars/Pandas allocations
            # out of the worker.  It also avoids forking a live asyncio/
            # websocket process, which is unsafe on Unix.
            context = multiprocessing.get_context("spawn")
            _CROSS_DATASET_PROCESS_POOL = ProcessPoolExecutor(
                max_workers=1, mp_context=context)
        return _CROSS_DATASET_PROCESS_POOL


def shutdown_cross_dataset_process_pool() -> None:
    """Stop the dedicated mapper worker during app shutdown or test cleanup."""
    global _CROSS_DATASET_PROCESS_POOL
    with _CROSS_DATASET_PROCESS_POOL_LOCK:
        pool = _CROSS_DATASET_PROCESS_POOL
        _CROSS_DATASET_PROCESS_POOL = None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown_cross_dataset_process_pool)


def _cross_dataset_process_async_lock() -> asyncio.Lock:
    """Get the single-flight lock belonging to the current event loop."""
    loop = asyncio.get_running_loop()
    with _CROSS_DATASET_PROCESS_ASYNC_LOCKS_LOCK:
        lock = _CROSS_DATASET_PROCESS_ASYNC_LOCKS.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _CROSS_DATASET_PROCESS_ASYNC_LOCKS[loop] = lock
        return lock


async def run_cross_dataset_scan_in_process(
    callback,
    *args,
    is_current=None,
    **kwargs,
):
    """Run one picklable cross-dataset callback outside the UI process.

    The callback must be a module-level function and return only picklable
    data.  The async lock prevents the viewer and Type Mapping preview from
    submitting competing cold scans, while ``is_current`` lets a queued
    request disappear before its process allocates mapper/index state.
    """
    lock = _cross_dataset_process_async_lock()
    async with lock:
        if is_current is not None and not is_current():
            return CROSS_SCAN_SUPERSEDED
        loop = asyncio.get_running_loop()
        pool = _cross_dataset_process_pool()
        try:
            return await loop.run_in_executor(
                pool, partial(callback, *args, **kwargs))
        except BrokenProcessPool:
            # Do not fall back to a UI-process thread after the worker dies:
            # that would reintroduce the websocket starvation.  The caller
            # renders a recoverable error and a later search creates a fresh
            # worker.
            shutdown_cross_dataset_process_pool()
            raise


def is_type_mapper_loaded() -> bool:
    """Return mapper readiness without triggering its lazy initialization."""
    try:
        from comparison import cross_dataset_type_mapper

        mapper = getattr(cross_dataset_type_mapper,
                         "_global_type_mapper", None)
        return bool(mapper is not None and getattr(mapper, "_loaded", False))
    except Exception:
        return False


def run_serialized_cross_dataset_scan(
    callback,
    *args,
    is_current=None,
    **kwargs,
):
    """Run one cross-dataset expansion under the process-wide single flight.

    ``is_current`` is an optional cheap predicate owned by the UI caller.  It
    is evaluated *after* acquiring the lock, so a queued stale request exits
    without initializing another mapper/index working set.  The callback must
    be UI-free because it normally runs in a worker thread.
    """
    with _CROSS_DATASET_SCAN_LOCK:
        if is_current is not None and not is_current():
            return CROSS_SCAN_SUPERSEDED
        return callback(*args, **kwargs)


def clear_neuron_index_cache() -> None:
    """Clear the process-local viewer cache (primarily useful for tests)."""
    with _INDEX_LOAD_LOCK:
        _INDEX_CACHE.clear()
        alias_cache = globals().get("_ALIAS_INDEX_CACHE")
        if alias_cache is not None:
            alias_cache.clear()
        release_cache = globals().get("_banc_release_body_pairs")
        if release_cache is not None:
            release_cache.cache_clear()


def neuron_index_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the app-owned neuron-index path for *dataset*.

    The index is a persistent "system file" outside ``cache/``: shipped seeds
    and pull-built indexes share this location, so clearing the cache never
    removes the metadata behind auto-suggestions and the viewer.
    """
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "neuron_indexes"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index.parquet"


def neuron_index_state_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the optional progress sidecar for a cached index."""
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "cache"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index_state.parquet"


def _metadata_candidates(dataset: str, datasets_dir: Optional[Path]) -> List[Path]:
    """Find generated metadata using the shared index-prep discovery rules."""
    if datasets_dir is None:
        datasets_dir = PROJECT_ROOT / "datasets"
    return builder_metadata_candidates(str(dataset).strip(), Path(datasets_dir))


def _metadata_signature(dataset: str, datasets_dir: Optional[Path]) -> Optional[Tuple[str, int]]:
    candidates = _metadata_candidates(dataset, datasets_dir)
    if not candidates:
        return None
    path = candidates[0]
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return None


def _is_blank(expression):
    """Return a Polars expression identifying null/empty display values."""
    import polars as pl

    return (
        expression.is_null()
        | (
            expression.cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            == ""
        )
    )


def _read_metadata_table(path: Path):
    """Read the materialized projection from a generated local neuron table."""
    import polars as pl

    try:
        frame = read_metadata_projection(path)
    except Exception:
        return None
    if "bodyId" not in frame.columns:
        return None
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
    )
    rename = {
        column: f"__metadata_{column}"
        for column in frame.columns
        if column != "bodyId"
    }
    return frame.rename(rename).unique(subset=["bodyId"], keep="first")


def _enrich_identifiers(frame, dataset: str, datasets_dir: Optional[Path]):
    """Fill blank type/instance values from local generated metadata."""
    import polars as pl

    if "bodyId" not in frame.columns:
        frame = frame.with_columns(pl.lit("").alias("bodyId"))

    display_columns = ["bodyId", "type", "instance"]
    expressions = []
    for column in display_columns:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )
        else:
            expressions.append(pl.lit("").alias(column))
    frame = frame.with_columns(expressions)

    metadata_signature = _metadata_signature(dataset, datasets_dir)
    if metadata_signature is None:
        return frame, False

    metadata_file = Path(metadata_signature[0])
    # A freshly generated rich index already contains the same projection.
    # Avoid opening the hundreds-of-MiB CSV again just to fill values it has.
    try:
        import polars as pl
        source_columns = set(metadata_columns(metadata_file))
        if (
            source_columns.issubset(set(frame.columns))
            and {"downstream_complete", "last_fetched", "connection_count"}
            .issubset(set(frame.columns))
        ):
            return frame, False
    except Exception:
        pass

    metadata = _read_metadata_table(metadata_file)
    if metadata is None:
        return frame, False

    frame = frame.join(metadata, on="bodyId", how="left")
    for metadata_column in metadata.columns:
        if not metadata_column.startswith("__metadata_"):
            continue
        column = metadata_column.removeprefix("__metadata_")
        if column not in frame.columns:
            frame = frame.with_columns(
                pl.col(metadata_column).alias(column)
            ).drop(metadata_column)
            continue
        metadata_column = f"__metadata_{column}"
        frame = frame.with_columns(
            pl.when(_is_blank(pl.col(column)))
            .then(pl.col(metadata_column).cast(pl.Utf8, strict=False).fill_null(""))
            .otherwise(pl.col(column))
            .alias(column)
        ).drop(metadata_column)
    return frame, True


def _load_cached_neuron_index(
    dataset: str,
    *,
    cache_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    enrich: bool = True,
) -> CachedNeuronIndex:
    """Load a cached neuron index for display.

    Raises:
        FileNotFoundError: when the cached ``neuron_index.parquet`` is absent.
        ValueError: when the file cannot be read as a usable table.
    """
    import polars as pl

    dataset = str(dataset or "").strip()
    if not dataset:
        raise FileNotFoundError("No dataset was selected")

    path = neuron_index_path(dataset, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    metadata_signature = _metadata_signature(dataset, datasets_dir) if enrich else None
    state_path = neuron_index_state_path(dataset, cache_dir)
    search_path = search_cache_path(path)
    try:
        state_signature = (
            (str(state_path), state_path.stat().st_mtime_ns)
            if state_path.is_file()
            else (str(state_path), None)
        )
        signature = (
            str(path),
            path.stat().st_mtime_ns,
            metadata_signature,
            state_signature,
            (
                str(search_path),
                search_path.stat().st_mtime_ns,
            ) if search_path.is_file() else (str(search_path), None),
            bool(enrich),
        )
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    if signature in _INDEX_CACHE:
        return _INDEX_CACHE[signature]

    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Could not read cached neuron index: {exc}") from exc

    # Overlay progress written by an in-flight connection pull.  The sidecar
    # contains only cache flags/counts and never changes the metadata columns.
    if state_path.is_file():
        try:
            state = pl.read_parquet(state_path)
            if "bodyId" in state.columns and "bodyId" in frame.columns:
                frame = frame.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                state = state.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                frame = frame.join(state, on="bodyId", how="left", suffix="__state")
                for column in ("downstream_complete", "last_fetched", "connection_count"):
                    state_column = f"{column}__state"
                    if state_column not in frame.columns:
                        continue
                    if column not in frame.columns:
                        frame = frame.with_columns(
                            pl.col(state_column).alias(column)
                        )
                    else:
                        frame = frame.with_columns(
                            pl.when(pl.col(state_column).is_not_null())
                            .then(pl.col(state_column))
                            .otherwise(pl.col(column))
                            .alias(column)
                        )
                    frame = frame.drop(state_column)
        except Exception:
            # A partially written sidecar must not make the viewer unusable;
            # the canonical index remains readable on its own.
            pass

    frame, enriched = _enrich_identifiers(frame, dataset, datasets_dir) if enrich else (frame, False)
    if not frame.columns:
        raise ValueError("The cached neuron index has no columns")

    # Legacy indexes may have been enriched from a source table after load;
    # apply the same order as newly generated caches before exposing columns.
    frame = frame.select(ordered_projection_columns(frame.columns))

    # Keep the schema stable and JSON-friendly for the UI.  Index values are
    # displayed as strings for bodyId/type/instance so large IDs are never
    # rounded by browser JavaScript.
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
        if "bodyId" in frame.columns
        else pl.lit("").alias("bodyId")
    )
    for column in ("type", "instance"):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )

    # Prefer the materialized compact sidecar. Older caches and test fixtures
    # transparently receive the same structure in memory, so the matcher has
    # one implementation and can be upgraded without rewriting the full
    # metadata parquet from the UI process.
    search_frame = None
    try:
        if search_path.is_file():
            search_frame = pl.read_parquet(search_path)
            if not is_search_cache_compatible(search_frame, frame.columns):
                search_frame = None
    except Exception:
        search_frame = None
    if search_frame is None:
        search_frame = build_search_cache_frame(frame)

    result = CachedNeuronIndex(
        dataset=dataset,
        path=path,
        frame=frame,
        columns=tuple(frame.columns),
        enriched=enriched,
        search_frame=search_frame,
    )
    _INDEX_CACHE[signature] = result
    # Remove old versions of this path so a rebuilt index does not accumulate
    # unbounded DataFrames in a long-running UI process.
    for old_key in list(_INDEX_CACHE):
        if old_key != signature and old_key[0] == str(path):
            _INDEX_CACHE.pop(old_key, None)
    return result


def load_cached_neuron_index(
    dataset: str,
    *,
    cache_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    enrich: bool = True,
) -> CachedNeuronIndex:
    """Load one cached index while preventing duplicate concurrent loads.

    A viewer opens the full metadata frame once, but a cross-dataset worker
    can request the same frame at the same time while resolving value-driven
    matches.  Serializing the cache boundary guarantees that both callers
    share the same ``CachedNeuronIndex`` instead of briefly retaining two
    multi-gigabyte Polars frames.
    """
    with _INDEX_LOAD_LOCK:
        return _load_cached_neuron_index(
            dataset,
            cache_dir=cache_dir,
            datasets_dir=datasets_dir,
            enrich=enrich,
        )


def _match_expression(frame, columns: List[str], text: str, mode: str):
    """Return the Polars expression for one shared search stage."""
    return polars_match_expression(frame, columns, text, mode)


def _match_column_expression(frame, column: str, text: str, mode: str):
    """Return the boolean expression for one column in a search stage."""
    return polars_match_column_expression(frame, column, text, mode)


def _match_hit_lists(
    frame,
    columns: List[str],
    text: str,
    mode: str,
    *,
    suppress_instance_if_type: bool = False,
):
    """Return ordered lists of matched columns and their displayed values.

    The primary match remains in ``match_column_key``/``match_value`` for
    compatibility with the pinned hint. These lists preserve every field that
    matched the same row, allowing the table to highlight a prefix hit and a
    secondary substring hit simultaneously. ``suppress_instance_if_type`` is
    retained for call-site compatibility, but suppression belongs only to the
    deduplicated match-panel lists; row-level hit lists must keep instances.
    """
    import polars as pl

    available = [column for column in columns if column in frame.columns]
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    if not available:
        return empty, empty

    needle = normalize_search_text(text)
    body_guard = _body_id_guard(frame, available, needle, mode)
    key_items = []
    value_items = []
    for column in available:
        matched = _match_column_expression(frame, column, needle, mode) & body_guard
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        key_items.append(
            pl.when(matched).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        value_items.append(
            pl.when(matched).then(display_value).otherwise(pl.lit(""))
        )

    def compact(items):
        return pl.concat_list(items).list.eval(
            pl.element().filter(pl.element() != "")
        )

    return compact(key_items), compact(value_items)


def _contains_expression(frame, columns: List[str], text: str):
    """Case-insensitive substring expression for an explicit column filter."""
    return _match_expression(frame, columns, text, "substring")


def _display_expression(column: str, frame):
    """Return a safe string expression for a retained metadata column.

    Body IDs are always matched and displayed as text.  Numeric-looking body
    IDs are normalized so a parquet reader that inferred ``123.0`` cannot
    produce a suggestion/query value that does not verify against the cached
    identifier ``123``.
    """
    return polars_display_expression(column)


def _body_id_guard(frame, columns: List[str], text: str, mode: str):
    """Prevent numeric searches from matching a non-bodyId display field."""
    return polars_body_id_guard(frame, columns, text)


def _ordered_match_columns(columns: List[str]) -> List[str]:
    """Return only the viewer's identity/taxonomy search scope.

    The full cache remains available for display and explicit column filters,
    but global viewer search must not scan operational, measurement, notes, or
    other arbitrary metadata. Keeping this list small also avoids repeatedly
    decoding large non-search fields for every keystroke.
    """
    return ordered_search_columns(columns)


def _normalize_filter_operator(value: str) -> str:
    """Normalize the viewer's explicit column-filter operator."""
    return normalize_search_operator(value)


def _highlight_text_html(value: Any, needle: str, mode: str) -> Optional[str]:
    """Return escaped cell text with only the matched spans marked.

    The table keeps its normal white/blue cell background and uses the cell
    outline for hit ownership.  This helper deliberately returns escaped HTML
    so the viewer can render a small ``<mark>`` around the matching characters
    without exposing metadata values through ``v-html``.
    """
    text = "" if value is None else str(value)
    needle = normalize_search_text(needle)
    if not text or not needle:
        return None

    spans: List[Tuple[int, int]] = []
    if mode == "prefix":
        if text.startswith(needle):
            spans = [(0, len(needle))]
    elif mode == "suffix":
        if text.endswith(needle):
            spans = [(len(text) - len(needle), len(text))]
    elif mode == "exact":
        if text == needle:
            spans = [(0, len(text))]
    elif mode == "regex":
        try:
            spans = [match.span() for match in re.finditer(needle, text)]
        except re.error:
            spans = []
    elif mode == "global":
        # A global row can have been admitted by either stage. Preserve the
        # strict-prefix visual cue when that exact prefix exists; substring
        # rows use the same case-insensitive rule as the search backend.
        if text.startswith(needle):
            spans = [(0, len(needle))]
        else:
            mode = "substring"

    if mode in {"substring", "contains"} and not spans:
        folded_text = text.casefold()
        folded_needle = needle.casefold()
        start = 0
        while folded_needle:
            position = folded_text.find(folded_needle, start)
            if position < 0:
                break
            spans.append((position, position + len(needle)))
            start = position + max(1, len(needle))

    if not spans:
        return None

    # Normalize overlapping/adjacent spans before escaping the source text.
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if not merged:
        return None

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(text[cursor:start], quote=False))
        parts.append(
            '<mark class="drocat-neuron-match-text">'
            f"{html.escape(text[start:end], quote=False)}"
            "</mark>"
        )
        cursor = end
    parts.append(html.escape(text[cursor:], quote=False))
    return "".join(parts)


def _highlighted_cells(
    row: Dict[str, Any],
    needle: str,
    mode: str,
    columns: List[str],
) -> Dict[str, str]:
    """Build safe HTML only for the searchable cells hit by this row."""
    hit_columns: List[str] = []
    for column in (
        *(row.get("match_column_keys") or []),
        *(row.get("secondary_match_column_keys") or []),
    ):
        column = str(column or "").strip()
        if column and column in columns and column not in hit_columns:
            hit_columns.append(column)
    if not hit_columns:
        column = str(row.get("match_column_key") or "").strip()
        if column in columns:
            hit_columns.append(column)

    highlighted: Dict[str, str] = {}
    for column in hit_columns:
        marked = _highlight_text_html(row.get(column), needle, mode)
        if marked is not None:
            highlighted[column] = marked
    return highlighted


def _presorted_search_matches(
    source,
    search_cache,
    columns: List[str],
    text: str,
    *,
    include_substrings: bool = True,
):
    """Return rows in canonical match order using the presorted sidecar.

    The sidecar is sorted by column priority and value.  For each column we
    therefore append its strict-prefix slice, then its substring-only slice,
    and remove row keys already claimed by a higher-priority column.  No
    result sort or full metadata-column scan is needed for this path.

    ``None`` means the requested mode is not representable by this cache
    (regex, suffix, and exact searches continue through the general matcher).
    """
    import polars as pl

    if search_cache is None or not columns:
        return None
    needle = normalize_search_text(text)
    if not needle:
        return None

    # The sidecar stores source row ordinals as a list.  The two hit frames
    # returned by this function are the exploded form used by the downstream
    # joins.  Keep that schema even when a valid query has no matching value;
    # returning ``search_cache.head(0)`` here leaves only ``__neuron_rows``
    # and causes the later aggregation to look for a missing
    # ``__neuron_row`` column.
    # Do not derive this schema with ``head(0).explode(...)``.  Polars keeps
    # an empty list column un-exploded in some releases, which leaves
    # ``__neuron_rows`` in the result and makes the later group/aggregation
    # fail with ``ColumnNotFoundError: __neuron_row`` on a valid no-hit query.
    empty_hits = pl.DataFrame(
        {
            "__neuron_row": pl.Series([], dtype=pl.UInt32),
            "search_column": pl.Series([], dtype=pl.Utf8),
            "search_priority": pl.Series([], dtype=pl.UInt16),
            "search_value": pl.Series([], dtype=pl.Utf8),
        }
    )
    empty_candidates = source.filter(pl.lit(False)).with_columns(
        pl.lit(0).alias("__match_priority"),
        pl.lit(0).alias("__match_kind_priority"),
        pl.lit("").alias("__candidate_column"),
        pl.lit("").alias("__candidate_value"),
    )

    available_columns = set(search_cache["search_column"].unique().to_list())
    ordered = [column for column in columns if column in available_columns]
    if not ordered:
        return empty_candidates, empty_hits, empty_hits
    numeric = is_numeric_search(needle)

    def _explode_combined_cells(column_frame):
        """Split comma-joined cells into one searchable name per part.

        FAFB/BANC release datasets pack several alternative type names into one
        ``additional_type(s)`` cell (``'vDeltaB, vDeltaC, ...'``).  The
        joined string is not a real neuron name, so match and report each
        part individually — otherwise match groups (and any query value a
        user selects from them) would carry the raw joined cell.
        """
        if not column_frame.height:
            return column_frame
        if not column_frame["search_value"].str.contains(",").any():
            return column_frame
        return (
            column_frame
            .with_columns(pl.col("search_value").str.split(","))
            .explode("search_value")
            .with_columns(pl.col("search_value").str.strip_chars())
            .filter(pl.col("search_value") != "")
            .with_columns(
                pl.col("search_value").str.to_lowercase().alias("search_value_folded")
            )
        )

    claimed = None
    chunks = []
    hit_parts = []
    for priority, column in enumerate(ordered):
        if numeric and column != "bodyId":
            continue
        column_frame = _explode_combined_cells(
            search_cache.filter(pl.col("search_column") == column)
        )
        if numeric:
            column_frame = column_frame.filter(
                pl.col("search_value").str.contains(r"^\d+$")
            )
        prefix_values = column_frame.filter(
            pl.col("search_value").str.starts_with(needle)
        )
        raw_prefix = (
            prefix_values
            .explode("__neuron_rows")
            .rename({"__neuron_rows": "__neuron_row"})
            if prefix_values.height else column_frame.head(0)
        )
        prefix_keys = raw_prefix.select("__neuron_row").unique(
            subset=["__neuron_row"], maintain_order=True
        ) if raw_prefix.height else None
        if raw_prefix.height:
            hit_parts.append(raw_prefix.select(
                "__neuron_row", "search_column", "search_priority", "search_value"
            ))
        prefix = raw_prefix
        if claimed is not None and prefix.height:
            prefix = prefix.join(claimed, on="__neuron_row", how="anti")
        prefix = prefix.with_columns(
            pl.lit(priority).alias("__candidate_priority"),
            pl.lit(0).alias("__candidate_kind"),
        )
        raw_substring_values = column_frame.filter(
            pl.col("search_value_folded").str.contains(
                needle.casefold(), literal=True
            )
        ) if include_substrings else column_frame.head(0)
        if prefix_values.height and raw_substring_values.height:
            raw_substring_values = raw_substring_values.join(
                prefix_values.select("search_value"),
                on="search_value",
                how="anti",
            )
        raw_substring = (
            raw_substring_values
            .explode("__neuron_rows")
            .rename({"__neuron_rows": "__neuron_row"})
            if raw_substring_values.height else column_frame.head(0)
        )
        if prefix_keys is not None and raw_substring.height:
            raw_substring = raw_substring.join(
                prefix_keys, on="__neuron_row", how="anti"
            )
        if raw_substring.height:
            hit_parts.append(raw_substring.select(
                "__neuron_row", "search_column", "search_priority", "search_value"
            ))
        substring = raw_substring
        excluded = claimed
        if prefix_keys is not None:
            excluded = prefix_keys if excluded is None else pl.concat(
                [excluded, prefix_keys], how="vertical"
            ).unique(subset=["__neuron_row"], maintain_order=True)
        if excluded is not None and substring.height:
            substring = substring.join(excluded, on="__neuron_row", how="anti")
        substring = substring.with_columns(
            pl.lit(priority).alias("__candidate_priority"),
            pl.lit(1).alias("__candidate_kind"),
        )

        chunk_parts = [part for part in (prefix, substring) if part.height]
        if not chunk_parts:
            continue
        chunk = pl.concat(chunk_parts, how="vertical_relaxed")
        # One searchable value exists per row/column, but keep this guard for
        # legacy sidecars built before that invariant was enforced.
        chunk = chunk.unique(subset=["__neuron_row"], maintain_order=True)
        chunks.append(chunk)
        keys = chunk.select("__neuron_row")
        claimed = keys if claimed is None else pl.concat(
            [claimed, keys], how="vertical"
        ).unique(subset=["__neuron_row"], maintain_order=True)

    if not chunks:
        return empty_candidates, empty_hits, empty_hits
    candidate = pl.concat(chunks, how="vertical_relaxed").select(
        "__neuron_row",
        pl.col("__candidate_priority").alias("__match_priority"),
        pl.col("__candidate_kind").alias("__match_kind_priority"),
        pl.col("search_column").alias("__candidate_column"),
        pl.col("search_value").alias("__candidate_value"),
    )
    raw_hits = pl.concat(hit_parts, how="vertical_relaxed")
    all_hits = raw_hits
    type_keys = all_hits.filter(
        pl.col("search_column") == "type"
    ).select("__neuron_row").unique(
        subset=["__neuron_row"], maintain_order=True
    )
    if type_keys.height:
        all_hits = all_hits.filter(
            ~(
                (pl.col("search_column") == "instance")
                & pl.col("__neuron_row").is_in(
                    type_keys["__neuron_row"].implode()
                )
            )
        )
    return source.join(
        candidate,
        on="__neuron_row",
        how="inner",
        maintain_order="right",
    ), all_hits, raw_hits


def _presorted_match_groups(
    filtered,
    hit_entries,
    search_columns: List[str],
    json_value,
    membership_entries=None,
):
    """Build the match panel from compact searchable-cell hits.

    This is the broad-query companion to :func:`_presorted_search_matches`.
    It keeps full metadata out of the group pass, but retains exact row and
    body membership for selection and focus.  ``None`` requests the legacy
    path (for example, when a caller supplies a hand-built frame without the
    sidecar hit columns).
    """
    import polars as pl

    required = {
        "__neuron_row", "search_column", "search_priority", "search_value",
    }
    if hit_entries is None or not required.issubset(set(hit_entries.columns)):
        return None
    if filtered.is_empty() or hit_entries.is_empty():
        return ([], {}, {}, {}, {})

    row_columns = [
        "__neuron_row", "__neuron_key", "bodyId", "match_column",
        "match_column_key", "match_value", "__match_priority",
        "__match_kind_priority", "__candidate_column", "__candidate_value",
    ]
    row_columns = [column for column in row_columns if column in filtered.columns]
    rows = filtered.select(row_columns)
    hits = hit_entries.join(
        rows.select("__neuron_row"), on="__neuron_row", how="inner"
    )
    if hits.is_empty():
        return ([], {}, {}, {}, {})
    hits = hits.join(rows, on="__neuron_row", how="inner", maintain_order="left")
    hits = hits.with_columns(
        (
            (pl.col("search_column") == pl.col("__candidate_column"))
            & (pl.col("search_value") == pl.col("__candidate_value"))
        ).cast(pl.Int8).alias("__is_primary_hit"),
        pl.col("search_value").str.to_lowercase().alias("__match_norm"),
    )
    # The first hit for a group is chosen after primary status, canonical
    # column priority, mode, and source row. The group itself is then sorted
    # by bodyId/type/instance/taxonomy priority and prefix-vs-substring.
    hits = hits.sort(
        [
            "search_value", "__is_primary_hit", "search_priority",
            "__match_kind_priority", "__neuron_row",
        ],
        descending=[False, True, False, False, False],
    )
    summary = hits.group_by("search_value", maintain_order=True).agg(
        pl.col("search_column").first().alias("__group_column"),
        pl.col("search_priority").first().alias("__group_priority"),
        pl.col("__match_kind_priority").first().alias("__group_kind"),
        pl.col("__is_primary_hit").max().alias("__group_is_primary"),
        pl.col("__neuron_key").unique(maintain_order=True).alias("__members"),
        pl.col("bodyId").unique(maintain_order=True).alias("__body_ids"),
    ).sort(
        ["__group_priority", "__group_kind", "search_value"],
        descending=[False, False, False],
    )

    # Case-insensitive exact-name membership mirrors the normal viewer
    # selection rule, while the group key preserves the visible spelling.
    # Membership is column-specific: if a value is a type, selecting it must
    # select the rows whose *type* is that value, not unrelated rows where
    # the same spelling happens to occur in hemibrainType/flywireType.
    # Use raw hits so a secondary group still retains every row sharing that
    # exact secondary value.
    membership_hits = membership_entries if membership_entries is not None else hit_entries
    membership_hits = membership_hits.join(
        rows.select("__neuron_row"), on="__neuron_row", how="inner"
    ).join(rows, on="__neuron_row", how="inner", maintain_order="left")
    membership_hits = membership_hits.with_columns(
        pl.col("search_value").str.to_lowercase().alias("__match_norm")
    )
    membership_map = {}
    if not membership_hits.is_empty():
        grouped_members = membership_hits.group_by(
            ["search_column", "__match_norm"], maintain_order=True
        ).agg(
            pl.col("__neuron_key").unique(maintain_order=True).alias("__members"),
            pl.col("bodyId").unique(maintain_order=True).alias("__body_ids"),
        )
        membership_map = {
            (
                str(raw.get("search_column") or ""),
                str(raw.get("__match_norm") or ""),
            ): (
                [str(value) for value in (raw.get("__members") or [])],
                [str(value or "") for value in (raw.get("__body_ids") or [])],
            )
            for raw in grouped_members.to_dicts()
        }

    match_groups: List[Dict[str, Any]] = []
    group_members: Dict[str, Tuple[str, ...]] = {}
    group_body_ids: Dict[str, Tuple[str, ...]] = {}
    group_related_sets: Dict[str, set[str]] = {}
    group_order: Dict[str, Tuple[int, int, str]] = {}
    group_rows: Dict[str, List[str]] = {}
    for raw in summary.to_dicts():
        value = str(json_value(raw.get("search_value")) or "").strip()
        if not value:
            continue
        column = str(raw.get("__group_column") or "")
        try:
            priority = int(raw.get("__group_priority") or len(search_columns))
        except (TypeError, ValueError):
            priority = len(search_columns)
        try:
            kind = int(raw.get("__group_kind") or 1)
        except (TypeError, ValueError):
            kind = 1
        members, body_ids = membership_map.get(
            (column, value.casefold()),
            ([str(item) for item in (raw.get("__members") or [])],
             [str(item or "") for item in (raw.get("__body_ids") or [])]),
        )
        members = tuple(member for member in members if member)
        body_ids = tuple(
            item[:-2] if item.endswith(".0") and item[:-2].isdigit() else item
            for item in body_ids if item
        )
        role = "primary" if int(raw.get("__group_is_primary") or 0) else "secondary"
        match_groups.append({
            "__match_group_key": value,
            "match_column": column,
            "match_column_key": column,
            "match_value": value,
            "body_count": len(body_ids),
            "match_role": role,
            "first_body_id": body_ids[0] if body_ids else "",
        })
        group_members[value] = members
        group_body_ids[value] = body_ids
        group_order[value] = (priority, kind, value)
        group_related_sets[value] = {value}
        group_rows[value] = list(members)

    group_rank = {
        str(group["__match_group_key"]): position
        for position, group in enumerate(match_groups)
    }
    # Link each row's canonical primary value only to its own secondary
    # values.  A value can be primary on one row and secondary on another;
    # that must not merge two independent primary entries.  In particular,
    # male-cns rows can have type=aMe17e and hemibrainType=aMe17a while other
    # rows have type=aMe17a: selecting either type must stay independent.
    primary_values = {
        str(json_value(raw.get("search_value")) or "").strip()
        for raw in summary.to_dicts()
        if int(raw.get("__group_is_primary") or 0)
    }
    row_hits = hits.select(
        "__neuron_row", "__candidate_value", "search_value"
    ).unique(maintain_order=True)
    owner_values: Dict[str, set[str]] = {}
    for raw in row_hits.to_dicts():
        primary = str(json_value(raw.get("__candidate_value")) or "").strip()
        secondary = str(json_value(raw.get("search_value")) or "").strip()
        if (
            not primary
            or not secondary
            or primary == secondary
            or primary not in group_rank
            or secondary not in group_rank
            or secondary in primary_values
        ):
            continue
        group_related_sets[primary].add(secondary)
        group_related_sets[secondary].add(primary)
        owner_values.setdefault(secondary, set()).add(primary)

    match_group_related = {
        key: tuple(sorted(values, key=lambda value: group_rank.get(value, len(group_rank))))
        for key, values in group_related_sets.items()
    }
    # Keep the relationship local to one primary/secondary bundle.  There is
    # deliberately no transitive component collapse: two primary names that
    # happen to share a secondary metadata spelling remain separate query
    # entries.
    primary_by_value = {}
    for key in group_rank:
        if key in primary_values:
            primary_by_value[key] = (key,)
            continue
        owners = sorted(
            owner_values.get(key, ()),
            key=lambda value: group_rank.get(value, len(group_rank)),
        )
        primary_by_value[key] = tuple(owners or (key,))

    return (
        _order_match_groups_with_secondaries(match_groups, primary_by_value),
        group_members,
        group_body_ids,
        match_group_related,
        primary_by_value,
    )


def _order_match_groups_with_secondaries(
    match_groups: List[Dict[str, Any]],
    match_group_primary: Dict[str, Tuple[str, ...]],
) -> List[Dict[str, Any]]:
    """Place each secondary match directly after its owning primary.

    Match groups are initially ordered for search relevance.  That order is
    useful for choosing the first page, but it can leave a taxonomy alias far
    away from the name that owns it.  The panel is easier to read when a
    secondary is rendered as an accessory row immediately below its primary.
    The relationship map remains unchanged, so selection continues to lock
    the whole primary/secondary bundle together.

    A secondary can technically be shared by more than one primary.  Display
    it after the first owner in the existing priority order and leave the
    relationship map responsible for the synchronized selection semantics.
    Any orphaned secondary is retained in its original position at the end
    rather than being dropped.
    """
    if not match_groups:
        return []

    by_key = {
        str(group.get("__match_group_key") or group.get("match_value") or ""): group
        for group in match_groups
    }
    secondary_by_owner: Dict[str, List[Dict[str, Any]]] = {}
    attached_secondary_keys: set[str] = set()
    for group in match_groups:
        if str(group.get("match_role") or "") != "secondary":
            continue
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        owners = match_group_primary.get(key, ())
        owner = next(
            (
                str(candidate or "")
                for candidate in owners
                if str(candidate or "") in by_key
                and str(by_key[str(candidate)].get("match_role") or "") == "primary"
            ),
            "",
        )
        if owner:
            secondary_by_owner.setdefault(owner, []).append(group)
            attached_secondary_keys.add(key)

    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in match_groups:
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        if key in seen:
            continue
        if (
            str(group.get("match_role") or "") == "secondary"
            and key in attached_secondary_keys
        ):
            continue
        ordered.append(group)
        seen.add(key)
        for secondary in secondary_by_owner.get(key, ()):
            secondary_key = str(
                secondary.get("__match_group_key")
                or secondary.get("match_value")
                or ""
            )
            if secondary_key not in seen:
                ordered.append(secondary)
                seen.add(secondary_key)

    # Preserve every group even when malformed/legacy cache data has a
    # missing relationship entry or a duplicate display key.
    for group in match_groups:
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        if key not in seen:
            ordered.append(group)
            seen.add(key)
    return ordered


def _match_metadata(
    frame,
    columns: List[str],
    text: str,
    stage: SearchStage | None = None,
):
    """Build match-priority and display metadata expressions for a query.

    The first matching column in the ordered scope wins.  For bodyId matches,
    the displayed hint mirrors auto-suggestion: show the corresponding
    instance when one exists, otherwise show ``bodyId``.  The match key and
    value are kept separately so the viewer can highlight the actual source
    cell while showing the compact hint in its pinned info columns.
    """
    import polars as pl

    ordered = [
        column
        for column in (stage.columns if stage is not None else _ordered_match_columns(columns))
        if column in frame.columns
    ]
    if not ordered:
        empty = pl.lit("")
        return pl.lit(0), empty, empty, empty

    needle = normalize_search_text(text)
    priority = pl.lit(len(ordered))
    match_column = pl.lit("")
    match_column_key = pl.lit("")
    match_value = pl.lit("")
    for rank, column in reversed(list(enumerate(ordered))):
        display_value = _display_expression(column, frame).cast(
            pl.Utf8, strict=False
        ).fill_null("")
        mode = stage.mode if stage is not None else "substring"
        if mode == "prefix":
            matched = display_value.str.starts_with(needle)
        elif mode == "suffix":
            matched = display_value.str.ends_with(needle)
        elif mode == "exact":
            matched = display_value == needle
        elif mode == "regex":
            import re

            try:
                re.compile(needle)
            except re.error:
                matched = pl.lit(False)
            else:
                matched = display_value.str.contains(needle, literal=False)
        else:
            matched = display_value.str.to_lowercase().str.contains(
                needle.casefold(), literal=True
            )
        if column == "bodyId" and is_numeric_search(needle):
            matched = matched & display_value.str.contains(r"^\d+$")
        if column == "bodyId" and "instance" in frame.columns:
            instance = (
                pl.col("instance")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            hint = pl.when(instance != "").then(instance).otherwise(pl.lit("bodyId"))
        else:
            hint = pl.lit(column)
        priority = pl.when(matched).then(pl.lit(rank)).otherwise(priority)
        match_column = pl.when(matched).then(hint).otherwise(match_column)
        match_column_key = pl.when(matched).then(pl.lit(column)).otherwise(match_column_key)
        match_value = pl.when(matched).then(display_value).otherwise(match_value)
    return priority, match_column, match_column_key, match_value


def _global_match_metadata(frame, columns: List[str], text: str):
    """Build canonical match metadata for the full viewer search.

    The viewer returns the union of prefix and substring matches, so the
    stage that admitted a row is not necessarily the row's best match.  For
    example, ``MeVPaMe2`` can contain ``aMe`` in ``type`` while ``aMe19a`` is
    a strict prefix in ``flywireType``.  The canonical column order must win
    first; prefix-vs-substring is only the tie-breaker inside that column.

    Returns expressions for the primary match, its match mode, all useful
    hits, and the lower-priority secondary hits.  Instance hits are omitted
    from the deduplicated match panel whenever the row has a type hit, because
    the type already identifies the neuron more precisely.  They remain in
    the row-level hit list so the corresponding instance cell can still be
    outlined and its matching characters highlighted.
    """
    import polars as pl

    ordered = [column for column in columns if column in frame.columns]
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    if not ordered:
        return (
            pl.lit(0), pl.lit(1), pl.lit(""), pl.lit(""), pl.lit(""),
            empty, empty, empty, empty,
        )

    needle = normalize_search_text(text)
    numeric = is_numeric_search(needle)
    body_guard = (
        _body_id_guard(frame, ["bodyId"], needle, "substring")
        if numeric and "bodyId" in frame.columns
        else pl.lit(True)
    )
    column_matches = {}
    type_matches = pl.lit(False)
    for column in ordered:
        allowed = (not numeric) or column == "bodyId"
        guard = body_guard if allowed else pl.lit(False)
        prefix = (
            _match_column_expression(frame, column, needle, "prefix") & guard
        )
        substring = (
            _match_column_expression(frame, column, needle, "substring") & guard
        )
        column_matches[column] = (prefix, substring, prefix | substring)
        if column == "type":
            type_matches = prefix | substring

    # Reverse assignment makes the first column in canonical order win.  A
    # prefix and substring match in the same column are resolved by the mode
    # expression, with strict prefix ranked first.
    priority = pl.lit(len(ordered))
    kind_priority = pl.lit(1)
    match_column = pl.lit("")
    match_column_key = pl.lit("")
    match_value = pl.lit("")
    for rank, column in reversed(list(enumerate(ordered))):
        prefix, _substring, matched = column_matches[column]
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        if column == "bodyId" and "instance" in frame.columns:
            instance = (
                _display_expression("instance", frame)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            hint = pl.when(instance != "").then(instance).otherwise(pl.lit("bodyId"))
        else:
            hint = pl.lit(column)
        priority = pl.when(matched).then(pl.lit(rank)).otherwise(priority)
        kind_priority = pl.when(matched).then(
            pl.when(prefix).then(pl.lit(0)).otherwise(pl.lit(1))
        ).otherwise(kind_priority)
        match_column = pl.when(matched).then(hint).otherwise(match_column)
        match_column_key = pl.when(matched).then(
            pl.lit(column)
        ).otherwise(match_column_key)
        match_value = pl.when(matched).then(display_value).otherwise(match_value)

    hit_column_items = []
    hit_value_items = []
    secondary_column_items = []
    secondary_value_items = []
    for rank, column in enumerate(ordered):
        _prefix, _substring, matched = column_matches[column]
        all_matched = matched
        if column == "instance":
            panel_matched = matched & ~type_matches
        else:
            panel_matched = matched
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        hit_column_items.append(
            pl.when(all_matched).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        hit_value_items.append(
            pl.when(all_matched).then(display_value).otherwise(pl.lit(""))
        )
        # The primary priority is row-dependent.  Keep only lower-priority
        # fields in the secondary lists so the match panel can label them
        # without exposing an instance already covered by type.
        lower_priority = panel_matched & (pl.lit(rank) > priority)
        secondary_column_items.append(
            pl.when(lower_priority).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        secondary_value_items.append(
            pl.when(lower_priority).then(display_value).otherwise(pl.lit(""))
        )

    def compact(items):
        if not items:
            return empty
        return pl.concat_list(items).list.eval(
            pl.element().filter(pl.element() != "")
        ).list.unique(maintain_order=True)

    return (
        priority,
        kind_priority,
        match_column,
        match_column_key,
        match_value,
        compact(hit_column_items),
        compact(hit_value_items),
        compact(secondary_column_items),
        compact(secondary_value_items),
    )


def _collect_all_keys(source, json_value) -> Dict[str, str]:
    """Map every matching ``__neuron_key`` to its user-facing bodyId.

    Used by the select-all action, which needs the complete result set across
    all pages without sending full row data to the browser.  The bodyId is
    normalised exactly like the viewer's per-row helper so that selecting all
    rows and selecting a single row resolve to the same query-safe value.
    """
    keys: Dict[str, str] = {}
    for key, raw_body_id in source.select(["__neuron_key", "bodyId"]).iter_rows():
        body_id = str(json_value(raw_body_id) or "").strip()
        if not body_id:
            keys[str(key or "")] = ""
            continue
        integer, dot, fraction = body_id.partition(".")
        if dot and integer.isdigit() and fraction and set(fraction) == {"0"}:
            body_id = integer
        keys[str(key or "")] = body_id
    return keys


def query_neuron_index(
    index: CachedNeuronIndex,
    *,
    search: str = "",
    search_column: Optional[str] = None,
    search_operator: str = "contains",
    filter_column: Optional[str] = None,
    filter_text: str = "",
    filter_operator: str = "contains",
    types_include: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    descending: bool = False,
    page: int = 1,
    page_size: int = 50,
    focus_key: Optional[str] = None,
    include_all_keys: bool = False,
    include_all_rows: bool = False,
    prefix_only_search: bool = False,
) -> NeuronIndexPage:
    """Filter, sort, and page a cached index without sending all rows to JS.

    Global search follows the shared matcher: every strict case-sensitive
    prefix match is returned first, followed by substring-only matches. This
    means a query such as ``aMe`` also returns ``MeVPaMe*`` values while
    keeping true prefixes at the top. Numeric input is verified against the
    real bodyId column only. An explicit column filter may target any retained
    metadata column and applies its selected operator directly — "contains"
    matches anywhere in that column with no starts-with display staging. With
    no selected search column, the global prefix-first behavior is used. The
    legacy ``filter_column``/``filter_text`` pair remains supported
    as an additional AND restriction for callers that still use it. When a
    query is present and no explicit sort is supplied, rows are grouped by
    matched-column priority (bodyId → type → instance → taxonomy), then by
    strict prefix versus substring, and finally matched value. The result
    also contains deduplicated matched-value groups plus primary/secondary
    relationships for the viewer's selection panel.

    ``types_include`` restricts the result to rows whose ``type`` is one of
    the given values (exact, case-sensitive). When provided it is the primary
    filter and the text search is skipped — the mapped-type view uses it to
    show the neurons of a set of mapped type names.

    ``include_all_rows`` skips the pagination slice and returns every row
    matching the current query (same filter, sort, and grouping) as one
    full page — the matched-rows CSV export uses it.

    ``prefix_only_search`` applies only to the default global search and keeps
    the strict prefix stage without expanding it to the potentially enormous
    case-insensitive substring stage. The viewer uses this bounded mode for
    an explicitly forced one-character query such as ``R``.
    """
    import polars as pl

    page_size = max(1, min(int(page_size), 500))
    frame = index.frame.with_row_index("__neuron_row")
    columns = list(index.columns)
    # The visible bodyId is the user-facing identity. The private key keeps
    # table selection safe even if a legacy index accidentally contains a
    # duplicate or blank bodyId.
    frame = frame.with_columns(
        pl.concat_str(
            [
                _display_expression("bodyId", frame)
                if "bodyId" in frame.columns
                else pl.lit(""),
                pl.lit("::"),
                pl.col("__neuron_row").cast(pl.Utf8),
            ]
        ).alias("__neuron_key")
    )
    filtered = frame

    include_types = [str(t) for t in (types_include or []) if str(t)]
    types_filter_active = bool(include_types) and "type" in filtered.columns
    if types_filter_active:
        filtered = filtered.filter(pl.col("type").is_in(include_types))

    search_text = normalize_search_text(search)
    search_columns = _ordered_match_columns(columns)
    search_column = str(search_column or "").strip()
    search_mode = _normalize_filter_operator(search_operator)
    filter_text = normalize_search_text(filter_text)
    filter_column = str(filter_column or "").strip()
    filter_mode = _normalize_filter_operator(filter_operator)
    if search_column in filtered.columns:
        search_target_columns = [search_column]
    else:
        search_target_columns = []
    if filter_column in {"__all__", "All columns", "all searchable fields"}:
        filter_columns = search_columns
    elif filter_column in filtered.columns:
        filter_columns = [filter_column]
    else:
        filter_columns = []
    active_column_filter = bool(filter_text and filter_columns)
    if active_column_filter:
        filtered = filtered.filter(
            _match_expression(filtered, filter_columns, filter_text, filter_mode)
        )

    def all_viewer_matches(source, text: str, *, include_substrings: bool = True):
        """Return prefix rows followed by substring-only rows.

        The inline suggestion menu intentionally stops at the first useful
        stage. The full viewer has room for the broader result set, so it
        unions both stages and marks the row kind for stable priority sorting.
        """
        stages = search_plan(
            text,
            columns,
            "auto",
            all_prefix_matches=True,
        )
        if not stages:
            return source.filter(pl.lit(False)), None, None
        prefix_stage = stages[0]
        substring_stage = (
            stages[1]
            if include_substrings and len(stages) > 1
            else None
        )
        prefix_rows = source.filter(
            _match_expression(source, list(prefix_stage.columns), text, "prefix")
        ).with_columns(pl.lit(0).alias("__match_kind_priority"))
        if substring_stage is None:
            return prefix_rows, prefix_stage, None
        substring_rows = source.filter(
            _match_expression(source, list(substring_stage.columns), text, "substring")
        )
        if prefix_rows.height:
            substring_rows = substring_rows.join(
                prefix_rows.select("__neuron_key").unique(),
                on="__neuron_key",
                how="anti",
            )
        substring_rows = substring_rows.with_columns(
            pl.lit(1).alias("__match_kind_priority")
        )
        if not prefix_rows.height:
            return substring_rows, prefix_stage, substring_stage
        if not substring_rows.height:
            return prefix_rows, prefix_stage, substring_stage
        return (
            pl.concat([prefix_rows, substring_rows], how="vertical"),
            prefix_stage,
            substring_stage,
        )

    if types_filter_active:
        # The type set is the primary filter (mapped-type view); the text
        # search is intentionally not applied on top of it.
        search_text = ""
        filter_text = ""
        active_column_filter = False

    scoped_search = bool(search_text and search_target_columns)
    match_text = search_text or (filter_text if active_column_filter else "")
    match_stage: SearchStage | None = None
    staged_search = False
    presorted_search = False
    fast_hit_entries = None
    fast_membership_entries = None
    if scoped_search:
        match_stage = SearchStage(search_mode, tuple(search_target_columns))
        # A targeted search applies its selected operator directly. "Contains"
        # matches anywhere in the column with no starts-with display staging;
        # users who want prefixes choose the "prefix" mode explicitly.
        filtered = filtered.filter(
            _match_expression(
                filtered,
                search_target_columns,
                search_text,
                search_mode,
            )
        )
    elif search_text:
        # The default viewer search can use the compact sidecar. It is
        # already ordered by the exact match priority used by the UI, so the
        # first page is available without sorting the full metadata frame.
        fast_matches = None
        if not active_column_filter:
            fast_matches = _presorted_search_matches(
                filtered,
                index.search_frame,
                search_columns,
                search_text,
                include_substrings=not prefix_only_search,
            )
        if fast_matches is not None:
            filtered, fast_hit_entries, fast_membership_entries = fast_matches
            presorted_search = True
        else:
            filtered, _, _ = all_viewer_matches(
                filtered,
                search_text,
                include_substrings=not prefix_only_search,
            )
        staged_search = True
    elif active_column_filter:
        match_stage = SearchStage(filter_mode, tuple(filter_columns))

    if match_text:
        empty_hit_list = pl.lit([], dtype=pl.List(pl.Utf8))
        global_search = bool(search_text and not scoped_search)
        if global_search:
            if fast_hit_entries is not None:
                # The sidecar already knows every eligible cell. Aggregate
                # those narrow rows instead of re-evaluating every search
                # expression against the full metadata table.
                # ``fast_hit_entries`` is the panel projection: it omits an
                # instance value when the row already has a type match.  The
                # raw membership entries retain that value for the table's
                # cell-level outline/highlight.
                row_hit_entries = (
                    fast_membership_entries
                    if fast_membership_entries is not None
                    else fast_hit_entries
                )
                hit_lists = row_hit_entries.group_by(
                    "__neuron_row", maintain_order=True
                ).agg(
                    pl.col("search_column").alias("__fast_hit_columns"),
                    pl.col("search_value").alias("__fast_hit_values"),
                )
                secondary_entries = fast_hit_entries.join(
                    filtered.select(["__neuron_row", "__match_priority"]),
                    on="__neuron_row",
                    how="inner",
                ).filter(
                    pl.col("search_priority") > pl.col("__match_priority")
                )
                secondary_lists = secondary_entries.group_by(
                    "__neuron_row", maintain_order=True
                ).agg(
                    pl.col("search_column").alias("__fast_secondary_columns"),
                    pl.col("search_value").alias("__fast_secondary_values"),
                ) if secondary_entries.height else None
                filtered = filtered.join(
                    hit_lists,
                    on="__neuron_row",
                    how="left",
                    maintain_order="left",
                )
                if secondary_lists is not None:
                    filtered = filtered.join(
                        secondary_lists,
                        on="__neuron_row",
                        how="left",
                        maintain_order="left",
                    )
                else:
                    filtered = filtered.with_columns(
                        empty_hit_list.alias("__fast_secondary_columns"),
                        empty_hit_list.alias("__fast_secondary_values"),
                    )
                filtered = filtered.with_columns(
                    pl.when(pl.col("__fast_secondary_columns").is_null())
                    .then(empty_hit_list)
                    .otherwise(pl.col("__fast_secondary_columns"))
                    .alias("__fast_secondary_columns"),
                    pl.when(pl.col("__fast_secondary_values").is_null())
                    .then(empty_hit_list)
                    .otherwise(pl.col("__fast_secondary_values"))
                    .alias("__fast_secondary_values"),
                )
                match_priority = pl.col("__match_priority")
                global_kind_priority = pl.col("__match_kind_priority")
                match_column_key = pl.col("__candidate_column")
                match_value = pl.col("__candidate_value")
                match_column = pl.when(
                    pl.col("__candidate_column") == "bodyId"
                ).then(
                    pl.when(
                        _display_expression("instance", filtered)
                        .cast(pl.Utf8, strict=False)
                        .fill_null("")
                        .str.strip_chars() != ""
                    ).then(
                        _display_expression("instance", filtered)
                        .cast(pl.Utf8, strict=False)
                        .fill_null("")
                        .str.strip_chars()
                    ).otherwise(pl.lit("bodyId"))
                ).otherwise(pl.col("__candidate_column"))
                all_hit_columns = pl.col("__fast_hit_columns")
                all_hit_values = pl.col("__fast_hit_values")
                secondary_hit_columns = pl.col("__fast_secondary_columns")
                secondary_hit_values = pl.col("__fast_secondary_values")
                prefix_hit_columns = all_hit_columns
                prefix_hit_values = all_hit_values
                substring_hit_columns = empty_hit_list
                substring_hit_values = empty_hit_list
                secondary_metadata = (
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_values.list.first().fill_null(""),
                )
            else:
                (
                    match_priority,
                    global_kind_priority,
                    match_column,
                    match_column_key,
                    match_value,
                    all_hit_columns,
                    all_hit_values,
                    secondary_hit_columns,
                    secondary_hit_values,
                ) = _global_match_metadata(filtered, search_columns, match_text)
                # Global search already has one canonical, column-ordered hit
                # list. The common metadata block below can materialize it
                # directly without reconstructing separate stage lists.
                prefix_hit_columns = all_hit_columns
                prefix_hit_values = all_hit_values
                substring_hit_columns = empty_hit_list
                substring_hit_values = empty_hit_list
                secondary_metadata = (
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_values.list.first().fill_null(""),
                )
        else:
            match_priority, match_column, match_column_key, match_value = _match_metadata(
                filtered, columns, match_text, stage=match_stage
            )
            prefix_hit_columns, prefix_hit_values = _match_hit_lists(
                filtered,
                list(match_stage.columns) if match_stage is not None else [],
                match_text,
                match_stage.mode if match_stage is not None else "substring",
                suppress_instance_if_type=False,
            )
            substring_hit_columns = empty_hit_list
            substring_hit_values = empty_hit_list
            secondary_hit_columns = empty_hit_list
            secondary_hit_values = empty_hit_list
            secondary_metadata = (pl.lit(""), pl.lit(""))

        if not global_search:
            filtered = filtered.with_columns(
                prefix_hit_columns.alias("__prefix_match_columns"),
                prefix_hit_values.alias("__prefix_match_values"),
                substring_hit_columns.alias("__substring_match_columns"),
                substring_hit_values.alias("__substring_match_values"),
            )
            all_hit_columns = pl.concat_list(
                [pl.col("__prefix_match_columns"), pl.col("__substring_match_columns")]
            ).list.unique(maintain_order=True)
            all_hit_values = pl.concat_list(
                [pl.col("__prefix_match_values"), pl.col("__substring_match_values")]
            ).list.unique(maintain_order=True)
        secondary_condition = (
            pl.lit(True)
            if global_search
            else pl.lit(False)
        )
        secondary_hit_columns = pl.when(secondary_condition).then(
            secondary_hit_columns
        ).otherwise(empty_hit_list)
        secondary_hit_values = pl.when(secondary_condition).then(
            secondary_hit_values
        ).otherwise(empty_hit_list)
        filtered = filtered.with_columns(
            match_priority.alias("__match_priority"),
            match_column.alias("match_column"),
            match_column_key.alias("match_column_key"),
            match_value.alias("match_value"),
            all_hit_columns.alias("match_column_keys"),
            all_hit_values.alias("match_values"),
            secondary_hit_columns.alias("secondary_match_column_keys"),
            secondary_hit_values.alias("secondary_match_values"),
            (
                global_kind_priority
                if global_search
                else pl.lit(0)
            ).alias("__match_kind_priority"),
            pl.lit(len(search_columns)).alias("__secondary_match_priority"),
            (
                secondary_metadata[0]
                if global_search
                else pl.lit("")
            ).alias("__secondary_match_column"),
            (
                secondary_metadata[1]
                if global_search
                else pl.lit("")
            ).alias("__secondary_match_column_key"),
            (
                secondary_metadata[2]
                if global_search
                else pl.lit("")
            ).alias("__secondary_match_value"),
        )
    else:
        if types_filter_active:
            # Mapped-type view: each row's own type is its matched value, so
            # the match panel lists the mapped types and type-level
            # selection works exactly like a normal multi-group search.
            type_value = (
                pl.col("type").cast(pl.Utf8, strict=False).fill_null("")
            )
            filtered = filtered.with_columns(
                pl.lit(len(search_columns)).alias("__match_priority"),
                pl.lit(1).alias("__match_kind_priority"),
                pl.lit("type").alias("match_column"),
                pl.lit("type").alias("match_column_key"),
                type_value.alias("match_value"),
                pl.lit(["type"], dtype=pl.List(pl.Utf8)).alias(
                    "match_column_keys"
                ),
                pl.concat_list([type_value]).alias("match_values"),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias(
                    "secondary_match_column_keys"
                ),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias(
                    "secondary_match_values"
                ),
            )
        else:
            filtered = filtered.with_columns(
                pl.lit(len(search_columns)).alias("__match_priority"),
                pl.lit(1).alias("__match_kind_priority"),
                pl.lit("").alias("match_column"),
                pl.lit("").alias("match_column_key"),
                pl.lit("").alias("match_value"),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias("match_column_keys"),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias("match_values"),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias("secondary_match_column_keys"),
                pl.lit([], dtype=pl.List(pl.Utf8)).alias("secondary_match_values"),
            )

    manual_match_value_sort = sort_by == "__match_value__"
    manual_sort = manual_match_value_sort or sort_by in columns
    selected_sort = "match_value" if manual_match_value_sort else (
        sort_by if manual_sort else ("bodyId" if "bodyId" in columns else columns[0])
    )
    sort_columns = []
    sort_directions = []
    if match_text and (not manual_sort or manual_match_value_sort):
        # Keep each match-column subset together in bodyId → type → instance →
        # taxonomy order. Within each column subset, strict prefixes precede
        # case-insensitive substring matches, then values are alphabetical.
        # This keeps a type substring such as MeVPaMe2 ahead of lower-priority
        # *type/taxonomy prefixes while still behind aMe* type prefixes.
        sort_columns.extend(
            ("__match_priority", "__match_kind_priority", "match_value")
        )
        sort_directions.extend(
            (
                False,
                False,
                bool(descending) if manual_match_value_sort else False,
            )
        )
    else:
        if selected_sort == "bodyId":
            # Body IDs are stored as strings in the cache to preserve large
            # values in the browser.  Sort them numerically so 10001 follows
            # 9999 instead of appearing between 10000 and 100011.
            filtered = filtered.with_columns(
                pl.col(selected_sort).cast(pl.UInt64, strict=False).alias("__body_id_sort")
            )
            sort_columns.append("__body_id_sort")
        else:
            sort_columns.append(selected_sort)
        sort_directions.append(bool(descending))
        if match_text:
            # An explicit sort column is primary; matching priority only
            # resolves rows with the same selected-column value.
            sort_columns.append("__match_priority")
            sort_directions.append(False)
    if not (
        presorted_search
        and match_text
        and not manual_sort
    ):
        try:
            try:
                filtered = filtered.sort(
                    sort_columns,
                    descending=sort_directions,
                    nulls_last=[True] * len(sort_columns),
                )
            except TypeError:  # compatibility with older Polars releases
                filtered = filtered.sort(sort_columns, descending=sort_directions)
        except TypeError:  # compatibility with older Polars releases
            filtered = filtered.sort(sort_columns, descending=sort_directions)

    # A match-group click should reveal the first corresponding metadata row.
    # Compute the page after the server-side sort so the UI does not need to
    # download or scan the full table in Python. This is optional and only
    # runs for an explicit focus request.
    focus_page = None
    if focus_key:
        focus_positions = (
            filtered
            .select("__neuron_key")
            .with_row_index("__sorted_position")
            .filter(pl.col("__neuron_key") == str(focus_key))
            .select("__sorted_position")
            .to_series()
            .to_list()
        )
        if focus_positions:
            focus_page = int(focus_positions[0]) // page_size + 1

    total = int(filtered.height)
    output_columns = [
        "__neuron_key",
        "match_column",
        "match_value",
        "match_column_key",
        "match_column_keys",
        "match_values",
        "secondary_match_column_keys",
        "secondary_match_values",
        *columns,
    ]
    if include_all_rows:
        # The matched-rows export consumes the complete filtered result set;
        # paging metadata collapses to a single full page.
        pages = 1
        current_page = 1
        rows = filtered.select(output_columns).to_dicts()
    else:
        pages = max(1, (total + page_size - 1) // page_size)
        current_page = max(1, min(int(page or 1), pages))
        rows = (
            filtered
            .select(output_columns)
            .slice((current_page - 1) * page_size, page_size)
            .to_dicts()
        )

    def json_value(value):
        if value is None:
            return ""
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return value

    safe_rows = [
        {column: json_value(value) for column, value in row.items()}
        for row in rows
    ]

    def highlight_mode() -> str:
        if search_text and not scoped_search:
            return "global"
        if search_text:
            return search_mode
        return filter_mode if active_column_filter else "global"

    if match_text:
        mode_for_highlight = highlight_mode()
        for row in safe_rows:
            row["__highlighted_cells"] = _highlighted_cells(
                row,
                match_text,
                mode_for_highlight,
                columns,
            )
    else:
        for row in safe_rows:
            row["__highlighted_cells"] = {}

    all_keys: Dict[str, str] = {}
    if include_all_keys:
        all_keys = _collect_all_keys(filtered, json_value)

    # Broad global searches can build their complete match panel directly
    # from the compact sidecar. This avoids re-evaluating every searchable
    # expression against the full metadata frame and, importantly, avoids a
    # Python pass over every wide row before the first page is rendered.
    if fast_hit_entries is not None:
        fast_groups = _presorted_match_groups(
            filtered,
            fast_hit_entries,
            search_columns,
            json_value,
            membership_entries=fast_membership_entries,
        )
        if fast_groups is not None:
            (
                fast_match_groups,
                fast_group_members,
                fast_group_body_ids,
                fast_group_related,
                fast_group_primary,
            ) = fast_groups
            return NeuronIndexPage(
                rows=safe_rows,
                total=total,
                page=current_page,
                pages=pages,
                page_size=page_size,
                sort_by=(
                    "match_value"
                    if match_text and (not manual_sort or manual_match_value_sort)
                    else selected_sort
                ),
                descending=(
                    bool(descending)
                    if match_text and manual_match_value_sort
                    else (False if match_text and not manual_sort else bool(descending))
                ),
                match_groups=fast_match_groups,
                match_group_members={
                    key: tuple(values)
                    for key, values in fast_group_members.items()
                },
                match_group_body_ids={
                    key: tuple(values)
                    for key, values in fast_group_body_ids.items()
                },
                match_group_related=fast_group_related,
                match_group_primary=fast_group_primary,
                focus_page=focus_page,
                all_keys=all_keys,
            )

    # Deduplicate by the exact matched name. A selection of one group means
    # “all rows sharing this name”, regardless of whether the name was found
    # in type, instance, class, or another useful taxonomy field.
    match_groups: List[Dict[str, Any]] = []
    group_members: Dict[str, List[str]] = {}
    group_body_ids: Dict[str, List[str]] = {}
    group_member_sets: Dict[str, set[str]] = {}
    group_body_id_sets: Dict[str, set[str]] = {}
    group_related_sets: Dict[str, set[str]] = {}
    group_primary_sets: Dict[str, set[str]] = {}
    group_index: Dict[str, Dict[str, Any]] = {}
    group_columns = [
        "__neuron_key",
        "bodyId",
        "match_column",
        "match_column_key",
        "match_value",
    ]
    if staged_search:
        group_columns.extend(
            [
                "__match_kind_priority",
                "__match_priority",
                "__secondary_match_column",
                "__secondary_match_column_key",
                "__secondary_match_value",
                "__secondary_match_priority",
                "secondary_match_column_keys",
                "secondary_match_values",
            ]
        )
    group_source = filtered.select(group_columns).to_dicts()
    ordered_filtered_keys = [
        str(json_value(raw.get("__neuron_key")) or "").strip()
        for raw in group_source
    ]
    filtered_key_order = {
        key: position for position, key in enumerate(ordered_filtered_keys)
    }
    group_order: Dict[str, Tuple[int, int, str]] = {}
    for raw in group_source:
        candidates = [
            (
                raw.get("match_column"),
                raw.get("match_column_key"),
                raw.get("match_value"),
                raw.get("__match_kind_priority", 1),
                raw.get("__match_priority", len(search_columns)),
                "primary",
            )
        ]
        if search_text and not scoped_search:
            primary_value = str(json_value(raw.get("match_value")) or "").strip()
            primary_column_key = str(raw.get("match_column_key") or "").strip()
            secondary_column_keys = [
                str(value or "").strip()
                for value in (raw.get("secondary_match_column_keys") or [])
            ]
            secondary_values = [
                str(json_value(value) or "").strip()
                for value in (raw.get("secondary_match_values") or [])
            ]
            secondary_candidates = list(zip(secondary_column_keys, secondary_values))
            if not secondary_candidates:
                # Keep compatibility with older cached rows that only carry
                # the original single-secondary fields.
                secondary_candidates = [
                    (
                        str(raw.get("__secondary_match_column_key") or "").strip(),
                        str(json_value(raw.get("__secondary_match_value")) or "").strip(),
                    )
                ]
            for secondary_column_key, secondary_value in secondary_candidates:
                if not secondary_value:
                    continue
                if (
                    secondary_value == primary_value
                    and secondary_column_key == primary_column_key
                ):
                    continue
                try:
                    secondary_priority = search_columns.index(secondary_column_key)
                except ValueError:
                    secondary_priority = len(search_columns)
                candidates.append(
                    (
                        secondary_column_key,
                        secondary_column_key,
                        secondary_value,
                        1,
                        secondary_priority,
                        "secondary",
                    )
                )

        member_key = str(json_value(raw.get("__neuron_key")) or "")
        primary_candidate_value = str(
            json_value(raw.get("match_value")) or ""
        ).strip()
        row_candidate_values = []
        for candidate in candidates:
            candidate_value = str(json_value(candidate[2]) or "").strip()
            if candidate_value and candidate_value not in row_candidate_values:
                row_candidate_values.append(candidate_value)
        body_id = json_value(raw.get("bodyId"))
        body_id = str(body_id or "").strip()
        if body_id.endswith(".0") and body_id[:-2].isdigit():
            body_id = body_id[:-2]
        for column, column_key, raw_value, raw_kind, raw_priority, match_role in candidates:
            value = str(json_value(raw_value) or "").strip()
            if not value:
                continue
            key = value
            try:
                kind_priority = int(raw_kind)
            except (TypeError, ValueError):
                kind_priority = 1
            try:
                column_priority = int(raw_priority)
            except (TypeError, ValueError):
                column_priority = len(search_columns)
            candidate_order = (column_priority, kind_priority, value)
            if key not in group_index:
                group_index[key] = {
                    "__match_group_key": key,
                    "match_column": str(column or ""),
                    "match_column_key": str(column_key or ""),
                    "match_value": value,
                    "body_count": 0,
                    "match_role": match_role,
                }
                match_groups.append(group_index[key])
                group_members[key] = []
                group_body_ids[key] = []
                group_member_sets[key] = set()
                group_body_id_sets[key] = set()
                # Self-included like the presorted path: a group with no
                # primary/secondary relations must still resolve to itself,
                # otherwise match-panel selection cannot remember it.
                group_related_sets[key] = {key}
                group_primary_sets[key] = set()
                group_order[key] = candidate_order
            elif candidate_order < group_order[key]:
                group_order[key] = candidate_order
                group_index[key]["match_column"] = str(column or "")
                group_index[key]["match_column_key"] = str(column_key or "")
                group_index[key]["match_role"] = match_role
            if member_key and member_key not in group_member_sets[key]:
                group_members[key].append(member_key)
                group_member_sets[key].add(member_key)
                group_index[key]["body_count"] += 1
            if body_id and body_id not in group_body_id_sets[key]:
                group_body_ids[key].append(body_id)
                group_body_id_sets[key].add(body_id)
            # Keep the row's candidate values for the direct primary/secondary
            # relationship pass below.  Do not union every value into one
            # connected component: a name can be primary on one row and a
            # secondary taxonomy value on another row.
            if primary_candidate_value:
                group_primary_sets[key].add(primary_candidate_value)

    match_groups.sort(
        key=lambda group: group_order.get(
            str(group.get("__match_group_key") or ""),
            (len(search_columns), 1, str(group.get("match_value") or "")),
        )
    )
    group_rank = {
        str(group.get("__match_group_key") or ""): position
        for position, group in enumerate(match_groups)
    }
    primary_values = {
        key for key, group in group_index.items()
        if group.get("match_role") == "primary"
    }
    owner_values: Dict[str, set[str]] = {}
    for key, owners in group_primary_sets.items():
        if key not in group_rank:
            continue
        for owner in owners:
            owner = str(owner or "").strip()
            if (
                not owner
                or owner == key
                or owner not in group_rank
                or key in primary_values
            ):
                continue
            group_related_sets.setdefault(owner, {owner}).add(key)
            group_related_sets.setdefault(key, {key}).add(owner)
            owner_values.setdefault(key, set()).add(owner)

    match_group_related = {
        key: tuple(
            sorted(
                group_related_sets.get(key, {key}),
                key=lambda value: group_rank.get(
                    value, len(group_rank)
                ),
            )
        )
        for key in group_rank
    }
    # Keep each primary value independent. A pure secondary value points back
    # to its owning primary value; a value that is itself primary does not
    # become an alias for another primary just because it also appears in a
    # taxonomy column on that row.
    match_group_primary = {}
    for key in group_rank:
        if key in primary_values:
            match_group_primary[key] = (key,)
        else:
            owners = sorted(
                owner_values.get(key, ()),
                key=lambda value: group_rank.get(value, len(group_rank)),
            )
            match_group_primary[key] = tuple(owners or (key,))

    # Keep the relevance order for primary names, but render each secondary
    # taxonomy match as an accessory row immediately below its owner.
    match_groups = _order_match_groups_with_secondaries(
        match_groups,
        match_group_primary,
    )

    # A row's displayed match is intentionally only its highest-priority
    # match, but selecting a name should include rows where that same name is
    # present in another searched field as well. Build that membership map
    # once from the active shared stage instead of making the UI guess from a
    # page-sized subset.
    if match_text and match_groups:
        # A strict type stage determines which rows are returned, but a
        # matched-name selection should still cover the same name in the other
        # searchable identity/taxonomy fields. Numeric input remains bodyId
        # only so a number in a taxonomy field can never bypass the safeguard.
        if scoped_search:
            membership_columns = list(search_target_columns)
        elif search_text:
            membership_columns = (
                ["bodyId"] if is_numeric_search(match_text) else search_columns
            )
        elif active_column_filter:
            membership_columns = list(match_stage.columns)
        else:
            membership_columns = search_columns
        scope_columns = [
            column for column in membership_columns if column in filtered.columns
        ]
        # A broad one-character prefix can match most of a large index. Do
        # not materialize every searchable cell as Python dictionaries here:
        # that blocks NiceGUI's event loop long enough for the browser socket
        # to disconnect. Polars performs the same exact-name membership join
        # column-wise and only returns the compact groups needed by selection.
        group_names_by_column: Dict[str, set[str]] = {}
        for group in match_groups:
            column = str(group.get("match_column_key", "") or "").strip()
            value = str(group.get("match_value", "") or "").strip()
            if column in scope_columns and value:
                group_names_by_column.setdefault(column, set()).add(
                    value.casefold()
                )
        same_name_members: Dict[Tuple[str, str], set[str]] = {}
        for column in scope_columns:
            group_norms = sorted(group_names_by_column.get(column, set()))
            if not group_norms:
                continue
            display_value = (
                _display_expression(column, filtered)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            column_matches = (
                filtered
                .select(
                    pl.col("__neuron_key"),
                    display_value.alias("__match_name"),
                )
                .with_columns(
                    pl.col("__match_name")
                    .str.to_lowercase()
                    .alias("__match_norm")
                )
                .filter(pl.col("__match_norm").is_in(group_norms))
            )
            grouped = column_matches.group_by("__match_norm").agg(
                pl.col("__neuron_key").alias("__members")
            )
            for raw in grouped.to_dicts():
                name = str(raw.get("__match_norm") or "")
                members = {
                    str(member)
                    for member in (raw.get("__members") or [])
                    if str(member)
                }
                if name and members:
                    same_name_members[(column, name)] = members

        body_ids_by_key = {}
        for key, raw_body_id in filtered.select(
            ["__neuron_key", "bodyId"]
        ).iter_rows():
            body_id = str(json_value(raw_body_id) or "").strip()
            if body_id.endswith(".0") and body_id[:-2].isdigit():
                body_id = body_id[:-2]
            body_ids_by_key[str(key or "")] = body_id
        for group in match_groups:
            value = str(group.get("match_value", "") or "").strip()
            column = str(group.get("match_column_key", "") or "").strip()
            members = same_name_members.get((column, value.casefold()))
            if members:
                ordered_members = sorted(
                    members,
                    key=lambda key: filtered_key_order.get(
                        key, len(filtered_key_order)
                    ),
                )
                group_members[value] = ordered_members
                group["body_count"] = len(members)
                group_body_ids[value] = [
                    body_ids_by_key[key]
                    for key in ordered_members
                    if body_ids_by_key.get(key)
                ]

    for group in match_groups:
        group_key = str(group.get("__match_group_key") or "")
        body_ids = group_body_ids.get(group_key, [])
        group["first_body_id"] = body_ids[0] if body_ids else ""

    return NeuronIndexPage(
        rows=safe_rows,
        total=total,
        page=current_page,
        pages=pages,
        page_size=page_size,
        sort_by=(
            "match_value"
            if match_text and (not manual_sort or manual_match_value_sort)
            else selected_sort
        ),
        descending=(
            bool(descending)
            if match_text and manual_match_value_sort
            else (False if match_text and not manual_sort else bool(descending))
        ),
        match_groups=match_groups,
        match_group_members={
            key: tuple(members) for key, members in group_members.items()
        },
        match_group_body_ids={
            key: tuple(body_ids) for key, body_ids in group_body_ids.items()
        },
        match_group_related=match_group_related,
        match_group_primary=match_group_primary,
        focus_page=focus_page,
        all_keys=all_keys,
    )


def query_match_group_subtypes(
    index: CachedNeuronIndex,
    member_keys,
    *,
    type_column: str = "type",
    limit: int = 500,
) -> Dict[str, Any]:
    """Return the distinct type values inside one match group's member rows.

    The viewer calls this lazily when a coarse taxonomy entry (for example a
    ``cell_type`` value such as ``circadian_clock``) is expanded, so a broad
    query never pays the subtype fan-out for every group. Member keys are the
    private ``bodyId::<row ordinal>`` keys used by the match panel; the
    ordinals are positions in the cached frame and stay stable for the
    lifetime of the loaded index, so filtering by them reproduces exactly the
    rows behind the clicked entry.

    The result is sorted by type name (case-insensitive). ``body_ids`` and
    ``member_keys`` per subtype are the exact selection payloads for the
    viewer; the client only receives the compact display fields.
    """
    import polars as pl

    empty = {"subtypes": [], "total_types": 0, "truncated": False}
    ordinals: List[int] = []
    key_by_ordinal: Dict[int, str] = {}
    for raw in member_keys or ():
        key = str(raw or "").strip()
        if not key:
            continue
        suffix = key.rpartition("::")[2]
        try:
            ordinal = int(suffix)
        except ValueError:
            continue
        if ordinal < 0 or ordinal in key_by_ordinal:
            continue
        ordinals.append(ordinal)
        key_by_ordinal[ordinal] = key
    if not ordinals or type_column not in index.columns:
        return empty

    frame = index.frame.with_row_index("__neuron_row").filter(
        pl.col("__neuron_row").is_in(
            pl.Series(ordinals, dtype=pl.UInt32).implode()
        )
    )
    frame = frame.with_columns(
        pl.col(type_column)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .alias("__subtype_value")
    ).filter(pl.col("__subtype_value") != "")
    if frame.is_empty():
        return empty

    grouped = frame.group_by("__subtype_value").agg(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias(
            "__subtype_body_ids"
        ),
        pl.col("__neuron_row").alias("__subtype_rows"),
    )
    subtypes: List[Dict[str, Any]] = []
    for raw in grouped.to_dicts():
        value = str(raw.get("__subtype_value") or "")
        keys: List[str] = []
        for ordinal in raw.get("__subtype_rows") or ():
            key = key_by_ordinal.get(int(ordinal))
            if key and key not in keys:
                keys.append(key)
        body_ids: List[str] = []
        for raw_body_id in raw.get("__subtype_body_ids") or ():
            body_id = str(raw_body_id or "").strip()
            if body_id.endswith(".0") and body_id[:-2].isdigit():
                body_id = body_id[:-2]
            if body_id and body_id not in body_ids:
                body_ids.append(body_id)
        subtypes.append({
            "match_value": value,
            "body_count": len(body_ids),
            "first_body_id": body_ids[0] if body_ids else "",
            "body_ids": tuple(body_ids),
            "member_keys": tuple(keys),
        })
    subtypes.sort(
        key=lambda item: (item["match_value"].casefold(), item["match_value"])
    )
    total_types = len(subtypes)
    return {
        "subtypes": subtypes[: max(0, int(limit))],
        "total_types": total_types,
        "truncated": total_types > int(limit),
    }


# ---------------------------------------------------------------------------
# Cross-dataset alias matches (expanded search for zero-hit viewer queries)
# ---------------------------------------------------------------------------

# Loaded indexes for alias lookups, keyed by (dataset, index mtime) so a
# Settings-side force rebuild invalidates the cache naturally.
_ALIAS_INDEX_CACHE: Dict[Tuple[str, int], "CachedNeuronIndex"] = {}


def datasets_with_cached_indexes(cache_dir: Optional[Path] = None) -> List[str]:
    """Return every known or locally indexed dataset.

    Keep the configured list first for stable UI ordering, then discover
    additional index folders so a newly pulled release participates in
    cross-dataset search without requiring a code/config update.
    """
    from .config import DATASETS

    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "neuron_indexes"
    datasets = [
        dataset
        for dataset in DATASETS
        if neuron_index_path(dataset, cache_dir).is_file()
    ]
    if not root.is_dir():
        return datasets
    for folder in sorted(root.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or not (folder / "neuron_index.parquet").is_file():
            continue
        dataset = dataset_identifier_from_folder(folder.name)
        if dataset and dataset not in datasets:
            datasets.append(dataset)
    return datasets


def count_type_in_index(index: "CachedNeuronIndex", type_name: str) -> Optional[int]:
    """Exact neuron count for one ``type`` value in a cached index."""
    if index is None or "type" not in index.frame.columns:
        return None
    import polars as pl

    search_frame = getattr(index, "search_frame", None)
    if search_frame is not None and {
            "search_column", "search_value", "__neuron_rows"}.issubset(
                set(search_frame.columns)):
        rows = search_frame.filter(
            (pl.col("search_column") == "type")
            & (pl.col("search_value") == str(type_name))
        )
        if rows.is_empty():
            return 0
        return int(rows.select(
            pl.col("__neuron_rows").list.len().sum()).item())

    return int(index.frame.filter(pl.col("type") == type_name).height)


def count_types_in_index(index: "CachedNeuronIndex",
                         type_names) -> Dict[str, int]:
    """Exact neuron counts for several ``type`` values in one query.

    The mapping-visualization flows use this for the source side, so an
    edge shows each current-dataset type's own neuron count instead of
    the foreign type's count.
    """
    names = sorted({str(n) for n in (type_names or ()) if str(n)})
    if index is None or not names or "type" not in index.frame.columns:
        return {}
    import polars as pl

    search_frame = getattr(index, "search_frame", None)
    if search_frame is not None and {
            "search_column", "search_value", "__neuron_rows"}.issubset(
                set(search_frame.columns)):
        rows = (
            search_frame
            .filter(
                (pl.col("search_column") == "type")
                & pl.col("search_value").is_in(names)
            )
            .with_columns(
                pl.col("__neuron_rows").list.len().alias("__count"))
            .group_by("search_value")
            .agg(pl.col("__count").sum())
            .to_dicts()
        )
        return {
            str(row["search_value"]): int(row["__count"])
            for row in rows
        }

    rows = (
        index.frame
        .filter(pl.col("type").is_in(names))
        .group_by("type")
        .len()
        .to_dicts()
    )
    return {str(row["type"]): int(row["len"]) for row in rows}


def mapped_csv_extras(rows, provenance, foreign_dataset: str) -> tuple:
    """Per-row mapping columns for the mapped-view CSV export (§9.3).

    ``provenance`` is the mapped-view state (target type → provenance
    entries built from the standardized linkers).  Returns
    ``(fieldnames, extras)`` aligned with ``rows``: the foreign dataset,
    the matched foreign type(s), the matched column(s), and one
    ``bridge-<column>`` cell per standardized linker column — deduped
    values in chain order, indirect (hub-route) values carrying the
    ``via`` note, empty when the row's type has no bridge through that
    column.  Pure: no index access, UI-free.
    """
    import re

    linker_columns: List[str] = []
    seen_columns = set()
    for entries in (provenance or {}).values():
        for entry in entries or []:
            for origin in entry.get("origins") or []:
                if origin.get("kind") not in (None, "linker"):
                    continue  # 'same name' markers carry no bridge column
                column = str(origin.get("column", ""))
                if column and column not in seen_columns:
                    seen_columns.add(column)
                    linker_columns.append(column)
    linker_columns.sort()
    fieldnames = (["foreign_dataset", "foreign_type(s)",
                   "matched column(s)"]
                  + [f"bridge-{column}" for column in linker_columns])

    extras: List[Dict[str, str]] = []
    for row in rows:
        entries = ((provenance or {}).get(str(row.get("type", "")))
                   or [])
        foreign_types = sorted({str(e.get("foreign_type", ""))
                                for e in entries
                                if e.get("foreign_type")})
        matched = sorted({str(e.get("matched", ""))
                          for e in entries if e.get("matched")})
        by_column: Dict[str, List[str]] = {
            column: [] for column in linker_columns}
        for entry in entries:
            for origin in entry.get("origins") or []:
                if origin.get("kind") not in (None, "linker"):
                    continue  # 'same name' markers carry no bridge column
                column = str(origin.get("column", ""))
                if column not in by_column:
                    continue
                value = str(origin.get("value", ""))
                if origin.get("indirect"):
                    hub = re.search(r"\(via ([^)]+)\)",
                                    str(origin.get("text", "")))
                    value += (f" (via {hub.group(1)})" if hub
                              else " (indirect)")
                if value not in by_column[column]:
                    by_column[column].append(value)
        extra = {
            "foreign_dataset": foreign_dataset or "",
            "foreign_type(s)": "; ".join(foreign_types),
            "matched column(s)": "; ".join(matched),
        }
        for column in linker_columns:
            extra[f"bridge-{column}"] = "; ".join(by_column[column])
        extras.append(extra)
    return fieldnames, extras


def _cell_contains(column_expr, value: str):
    """Match a comma-separated annotation cell against one value.

    ``additional_type(s)``-style cells list several names separated by
    ','; the match is exact per entry after splitting and stripping.
    """
    import polars as pl

    # ``auto:`` is a provenance tier used by BANC label columns, not a
    # different type namespace.  The mapper keeps the raw cell for
    # provenance; coverage matching uses the same canonical token so an
    # auto-derived bridge can still reach the independently indexed rows.
    from comparison.cross_dataset_type_mapper import canonical_linker_token

    canonical = canonical_linker_token(value).casefold()
    tokens = (
        column_expr.cast(pl.Utf8)
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
    )
    return tokens.list.eval(
        pl.when(pl.element().str.to_lowercase().str.starts_with("auto:"))
        .then(pl.element().str.slice(5).str.strip_chars())
        .otherwise(pl.element())
        .str.to_lowercase()
    ).list.contains(canonical)


@lru_cache(maxsize=1)
def _banc_release_body_pairs() -> Tuple[Tuple[str, str], ...]:
    """Load the complete BANC v626↔v888 root relation once.

    The relation is bodyId evidence, not a type-name crosswalk.  Keep every
    row (including repeated roots) because one v626 root can legitimately
    correspond to several v888 roots.  An unavailable relation returns an
    empty tuple so ordinary type/annotation pooling remains usable.
    """
    import polars as pl

    candidates = (
        PROJECT_ROOT / "datasets" / "banc_v888" / "downloads"
        / "banc_888_meta.feather",
        PROJECT_ROOT / "datasets" / "banc_v626" / "downloads"
        / "banc_888_meta.feather",
        PROJECT_ROOT / "compiled_data" / "banc_v888"
        / "banc_888_meta.feather",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()),
                None)
    if path is None:
        return ()
    try:
        relation = pl.read_ipc(
            path, columns=["root_626", "root_888"], memory_map=False)
    except Exception:
        return ()
    pairs: List[Tuple[str, str]] = []
    for left, right in relation.iter_rows():
        left_tokens = _match_body_id_tokens(left)
        right_tokens = _match_body_id_tokens(right)
        if left_tokens and right_tokens:
            pairs.append((left_tokens[0], right_tokens[0]))
    return tuple(pairs)


def _banc_release_pairs_for_direction(
        source_dataset: str, target_dataset: str
) -> Tuple[Tuple[str, str], ...]:
    """Return release body pairs oriented source → target."""
    relation = _banc_release_body_pairs()
    if (source_dataset, target_dataset) == ("banc_v626", "banc_v888"):
        return relation
    if (source_dataset, target_dataset) == ("banc_v888", "banc_v626"):
        return tuple((right, left) for left, right in relation)
    return ()


def _match_body_id_tokens(value: Any) -> List[str]:
    """Normalize one metadata match cell into candidate bodyId strings."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return []
    tokens = re.split(r"[,;|\s]+", text)
    result: List[str] = []
    for token in tokens:
        token = token.strip().strip("'")
        if not token or token.casefold() in {"nan", "none", "null"}:
            continue
        # Some CSV readers render integer-valued identifiers as ``123.0``;
        # remove only that harmless suffix and never coerce large IDs through
        # floating point (which would lose precision).
        if re.fullmatch(r"-?\d+\.0", token):
            token = token[:-2]
        if token not in result:
            result.append(token)
    return result


def _type_body_id_set(index: Optional["CachedNeuronIndex"],
                      type_name: str) -> set:
    """Return endpoint bodyIds for exact type validation."""
    if index is None or "type" not in index.frame.columns:
        return set()
    import polars as pl

    frame = index.frame.filter(pl.col("type") == type_name)
    if "bodyId" not in frame.columns:
        return set()
    return {
        str(body_id).strip()
        for body_id in frame.get_column("bodyId").to_list()
        if str(body_id).strip()
    }


def pool_bridge_body_ids(source_dataset: str, target_dataset: str,
                         linkers, source_type: str, foreign_type: str,
                         *, indexes: Optional[Dict[str,
                                                   "CachedNeuronIndex"]]
                         = None) -> Dict[str, Any]:
    """One-side bodyId pools per standardized linker of a bridge.

    ``linkers`` are ``standardize_bridge`` entries (``column``/``value``/
    ``home``; same-name pass entries are ignored). Each linker pools the
    bodyIds on its home side: the rows of that side's endpoint type whose
    linker column carries the linker value (comma-split exact match).
    There is intentionally no bodyId-to-bodyId join here.  A side without
    an independent linker uses the full endpoint type population, while a
    side with a linker reports only the rows carrying that side's evidence.
    Optional BANC match columns remain mapper provenance, not coverage
    constraints.  The BANC release relation may provide participant pools
    for release-coverage diagnostics, but its individual pairs are never
    exposed as matched neurons.

    Returns the independent per-side pools plus:

    * ``source_coverage`` / ``target_coverage`` — human-readable
      ``covered <pool> of <total> (<pct>)`` strings (shared formatter;
      ``coverage`` stays the historical target-side alias, ``""`` when
      that side could not be measured);
    * ``source_pool_size`` / ``source_type_total`` / ``target_pool_size``
      / ``target_type_total`` — the same numbers machine-readable (the
      totals are ``None`` for an unmeasurable side);
    * ``source_basis`` / ``target_basis`` — WHICH pool state produced the
      numbers: ``"linker rows"`` (measured subset), ``"full population"``
      (unconstrained side), ``"release relation participants"``, or
      ``"unmeasured"`` (coverage index unavailable);
    * ``source_type_body_ids`` / ``target_type_body_ids`` — the FULL
      population of each mapped type in its own dataset (sorted; NOT the
      linker-filtered pool subset, and never a cross-dataset bodyId
      pairing);
    * ``granularity`` (``"n to m"``) and ``coverage_basis``.
    """
    import polars as pl

    datasets = {source_dataset, target_dataset}
    loaded = dict(indexes or {})
    for ds in datasets:
        if ds not in loaded:
            # Body IDs are needed here only to report coverage for an already
            # resolved type-level edge.  Do not silently materialize the full
            # wide viewer index when a caller omits an explicit projection.
            loaded[ds] = _load_coverage_index(ds)

    side_type = {source_dataset: source_type, target_dataset: foreign_type}

    def _resolve_home(column: str, claimed_home: str) -> str:
        """The dataset whose index physically holds this linker column.

        Chain hops attribute the crosswalk hop (``flywireType``) to the
        foreign dataset although the physical column lives in male-cns;
        trust the pair registry first, then plain column presence.
        """
        from comparison.cross_dataset_type_mapper import BRIDGE_STANDARD

        registry = (
            BRIDGE_STANDARD.get((source_dataset, target_dataset))
            or BRIDGE_STANDARD.get((target_dataset, source_dataset), ())
        )
        for reg_column, reg_home in registry:
            if reg_column == column:
                return reg_home
        if (claimed_home in datasets and loaded.get(claimed_home) is not None
                and column in loaded[claimed_home].frame.columns):
            return claimed_home
        source_has = (loaded.get(source_dataset) is not None
                      and column in loaded[source_dataset].frame.columns)
        target_has = (loaded.get(target_dataset) is not None
                      and column in loaded[target_dataset].frame.columns)
        if source_has and not target_has:
            return source_dataset
        if target_has and not source_has:
            return target_dataset
        return claimed_home

    per_linker = []
    source_pool: List[str] = []
    target_pool: List[str] = []
    had_target_linker = False
    had_source_linker = False

    for linker in linkers or []:
        if linker.get("kind") != "linker":
            continue
        column = str(linker.get("column", ""))
        value = str(linker.get("value", ""))
        home = _resolve_home(column, str(linker.get("home", "")))
        if not column or not value or home not in datasets:
            continue
        index = loaded.get(home)
        if column == "banc_release_crosswalk":
            # This is a relation-backed virtual linker: its evidence lives
            # in banc_888_meta.feather rather than in a cell column of either
            # neuron index.  Resolve it after all ordinary linkers so the
            # complete type-pair relation can be filtered on both endpoints.
            per_linker.append({
                "column": column, "value": value, "home": home,
                "body_ids": [], "coverage_basis": "release relation",
                "raw_value": linker.get("raw_value", value),
                "canonical_value": linker.get(
                    "canonical_value", value),
                "evidence_tier": linker.get("evidence_tier", ""),
            })
            continue
        if index is None or column not in index.frame.columns:
            continue
        endpoint_type = side_type.get(home, "")
        if not endpoint_type:
            continue
        frame = index.frame.filter(pl.col("type") == endpoint_type)
        frame = frame.filter(_cell_contains(pl.col(column), value))
        body_ids = list(dict.fromkeys(
            str(b) for b in frame.select("bodyId").to_series().to_list()
            if str(b).strip()))
        per_linker.append({
            "column": column, "value": value, "home": home,
            "body_ids": body_ids,
            "coverage_basis": "home-side linker rows",
            "raw_value": linker.get("raw_value", value),
            "canonical_value": linker.get(
                "canonical_value", value),
            "evidence_tier": linker.get("evidence_tier", ""),
            "indirect": bool(linker.get("indirect", False)),
        })
        if home == source_dataset:
            # An existing linker with no matching endpoint rows is an
            # observed zero-coverage result. Keep that empty pool instead of
            # treating the side as unconstrained and substituting the whole
            # type population.
            had_source_linker = True
            source_pool.extend(body_ids)
        elif home == target_dataset:
            had_target_linker = True
            target_pool.extend(body_ids)

    release_entries = [
        entry for entry in per_linker
        if entry.get("column") == "banc_release_crosswalk"]
    release_pairs = _banc_release_pairs_for_direction(
        source_dataset, target_dataset)
    release_pools = False
    if release_entries and release_pairs:
        source_type_ids = _type_body_id_set(
            loaded.get(source_dataset), source_type)
        target_type_ids = _type_body_id_set(
            loaded.get(target_dataset), foreign_type)
        valid_pairs = [
            (source_id, target_id)
            for source_id, target_id in release_pairs
            if source_id in source_type_ids and target_id in target_type_ids
        ]
        if valid_pairs:
            release_pools = True
            source_pool = list(dict.fromkeys(
                source_id for source_id, _ in valid_pairs))
            target_pool = list(dict.fromkeys(
                target_id for _, target_id in valid_pairs))
            had_source_linker = True
            had_target_linker = True
            for entry in release_entries:
                home = entry.get("home")
                if home == target_dataset:
                    entry["body_ids"] = target_pool
                else:
                    entry["home"] = source_dataset
                    entry["body_ids"] = source_pool
                entry["relation_rows"] = len(valid_pairs)
                entry["coverage_basis"] = (
                    "release relation participant coverage; no bodyId pairing")

    source_pool = list(dict.fromkeys(source_pool))
    target_pool = list(dict.fromkeys(target_pool))

    # Crosswalk-arrival chains (e.g. [DN1pB·type, flywireType 'DN1pB'])
    # and bare same-name chains have NO per-side linker on one or both
    # sides: the crosswalk cell (or the name equality itself) names that
    # side's type identity, so the pool on an unconstrained side is the
    # FULL endpoint type's bodyIds — not an empty pool.
    if not had_target_linker:
        target_index = loaded.get(target_dataset)
        if target_index is not None and "type" in target_index.frame.columns:
            frame = target_index.frame.filter(
                pl.col("type") == foreign_type)
            target_pool = [
                str(b) for b in frame.select("bodyId").to_series().to_list()
                if str(b).strip()
            ]
    if not had_source_linker:
        source_index = loaded.get(source_dataset)
        if source_index is not None and "type" in source_index.frame.columns:
            frame = source_index.frame.filter(
                pl.col("type") == source_type)
            source_pool = [
                str(b) for b in frame.select("bodyId").to_series().to_list()
                if str(b).strip()
            ]

    granularity = f"{len(source_pool)} to {len(target_pool)}"
    source_total = count_type_in_index(loaded.get(source_dataset), source_type)
    target_total = count_type_in_index(loaded.get(target_dataset), foreign_type)
    # Shared formatter: thousands separators + the share in parentheses,
    # matching the rendered coverage cells (user 2026-09-09).
    from comparison.mapping_visualization import format_coverage

    source_coverage = (
        f"covered {format_coverage(len(source_pool), source_total)}"
        if source_total is not None else "")
    target_coverage = (
        f"covered {format_coverage(len(target_pool), target_total)}"
        if target_total is not None else "")
    def _side_basis(had_linker: bool, dataset: str) -> str:
        # One of four pool states per side: the release relation supplied
        # participants, a linker measured a subset, the unconstrained
        # fallback used the full endpoint type population, or the side's
        # coverage index was unavailable so nothing was measured at all.
        if release_pools:
            return "release relation participants"
        if had_linker:
            return "linker rows"
        if loaded.get(dataset) is None:
            return "unmeasured"
        return "full population"

    # The FULL per-type populations (user 2026-09-09): every bodyId of the
    # mapped source/target type in its own dataset — independent per side,
    # never a cross-dataset bodyId pairing.  Unlike the pools above these
    # are not filtered by linker evidence; empty when the side's coverage
    # index is unavailable.
    source_type_body_ids = sorted(_type_body_id_set(
        loaded.get(source_dataset), source_type))
    target_type_body_ids = sorted(_type_body_id_set(
        loaded.get(target_dataset), foreign_type))

    return {
        "per_linker": per_linker,
        "source_body_ids": source_pool,
        "target_body_ids": target_pool,
        "granularity": granularity,
        # ``coverage`` is retained as the historical target-side alias.
        "coverage": target_coverage,
        "source_coverage": source_coverage,
        "target_coverage": target_coverage,
        # Numeric pool/total pairs for machine-readable exports and the
        # basis-aware coverage cells (None total = side not measurable).
        "source_pool_size": len(source_pool),
        "source_type_total": source_total,
        "target_pool_size": len(target_pool),
        "target_type_total": target_total,
        # Full per-type populations for the mapping CSV export.
        "source_type_body_ids": source_type_body_ids,
        "target_type_body_ids": target_type_body_ids,
        "source_basis": _side_basis(had_source_linker, source_dataset),
        "target_basis": _side_basis(had_target_linker, target_dataset),
        "coverage_basis": "independent endpoint pools; no bodyId pairing",
    }


def chain_is_supported(pool: Dict[str, Any], target_dataset: str,
                       source_dataset: Optional[str] = None) -> bool:
    """False when a pooled bridge chain carries no measured evidence.

    A chain whose standardized linkers home on one endpoint yet every one
    pooled zero rows on that endpoint derives nothing observable.  Such a
    chain would render a mapping whose own coverage reads ``0 of N`` —
    dropped by the pooling callers as a safety net on top of the mapper's
    derivation licensing.  The historical two-argument call remains
    target-side-only for compatibility; the shared resolver supplies
    ``source_dataset`` and checks both endpoint sides.
    """
    if not pool:
        return True
    target_linkers = [
        linker for linker in (pool.get("per_linker") or [])
        if linker.get("home") == target_dataset]
    if target_linkers and pool.get("target_basis") != "unmeasured":
        if not any(linker.get("body_ids") for linker in target_linkers):
            return False
    if source_dataset is None:
        return True
    source_linkers = [
        linker for linker in (pool.get("per_linker") or [])
        if linker.get("home") == source_dataset]
    if source_linkers and pool.get("source_basis") != "unmeasured":
        if not any(linker.get("body_ids") for linker in source_linkers):
            return False
    return True


def resolve_prioritized_bridge_pool(
        source_dataset: str,
        target_dataset: str,
        chains,
        source_type: str,
        foreign_type: str,
        *,
        indexes: Optional[Dict[str, "CachedNeuronIndex"]] = None,
        max_alternatives: int = 8,
) -> Dict[str, Any]:
    """Resolve and pool bridge candidates in evidence-priority order.

    Every candidate is attempted independently, in the deterministic order
    from :func:`prioritized_bridge_chains`.  The first supported candidate is
    the selected bridge used for the legacy top-level pool fields and visual
    weights.  Later supported candidates are retained as valid alternatives;
    their independent endpoint pools are unioned into the explicit
    ``all_valid_*`` fields.  A failed candidate never contributes coverage.

    This is intentionally a type-evidence/coverage boundary: it never joins
    bodyIds across datasets and never uses population size to choose a type
    target.  The selected-versus-all-valid distinction lets the UI explain a
    split such as MCNS SMP227 → FAFB s-CPDN3B/C/D without pretending that the
    three branches are mutually exclusive neurons.
    """
    from comparison.cross_dataset_type_mapper import (
        prioritized_bridge_chains,
        standardize_bridge,
    )

    candidates = [
        chain for chain in (chains or [])
        if chain and (not foreign_type
                      or chain[-1].get("value") == foreign_type)
    ]
    if not candidates:
        candidates = [chain for chain in (chains or []) if chain]
    ordered = prioritized_bridge_chains(
        candidates, source_dataset, target_dataset)
    if max_alternatives and max_alternatives > 0:
        ordered = ordered[:max_alternatives]

    attempts: List[Dict[str, Any]] = []
    valid: List[Tuple[int, Any, Dict[str, Any]]] = []
    selected: Optional[Tuple[int, Any, Dict[str, Any]]] = None
    for rank, chain in enumerate(ordered, start=1):
        linkers = standardize_bridge(chain, source_dataset, target_dataset)
        try:
            pool = pool_bridge_body_ids(
                source_dataset, target_dataset, linkers,
                source_type, foreign_type, indexes=indexes)
        except Exception as exc:
            attempts.append({
                "rank": rank,
                "chain": chain,
                "linkers": linkers,
                "supported": False,
                "status": "error",
                "reason": str(exc),
            })
            continue
        supported = chain_is_supported(
            pool, target_dataset, source_dataset=source_dataset)
        target_linkers = [
            linker for linker in (pool.get("per_linker") or [])
            if linker.get("home") == target_dataset]
        source_linkers = [
            linker for linker in (pool.get("per_linker") or [])
            if linker.get("home") == source_dataset]
        unsupported_sides = []
        if (target_linkers and pool.get("target_basis") != "unmeasured"
                and not any(l.get("body_ids") for l in target_linkers)):
            unsupported_sides.append("target")
        if (source_linkers and pool.get("source_basis") != "unmeasured"
                and not any(l.get("body_ids") for l in source_linkers)):
            unsupported_sides.append("source")
        attempt = {
            "rank": rank,
            "chain": chain,
            "linkers": linkers,
            "supported": supported,
            "status": "supported" if supported else "unsupported",
            "reason": (
                ", ".join(
                    f"{side}-side linker rows had no bodyIds"
                    for side in unsupported_sides)
                if not supported else ""),
            "unsupported_sides": unsupported_sides,
            "source_basis": pool.get("source_basis"),
            "target_basis": pool.get("target_basis"),
            "linker_values": [
                {
                    "column": linker.get("column", ""),
                    "raw_value": linker.get(
                        "raw_value", linker.get("value", "")),
                    "canonical_value": linker.get(
                        "canonical_value", linker.get("value", "")),
                    "home": linker.get("home", ""),
                }
                for linker in linkers
                if linker.get("kind") == "linker"
            ],
            "source_pool_size": pool.get("source_pool_size"),
            "target_pool_size": pool.get("target_pool_size"),
        }
        attempts.append(attempt)
        if not supported:
            continue
        record = (rank, chain, pool)
        valid.append(record)
        if selected is None:
            selected = record

    if selected is None:
        return {
            "resolution_status": (
                "no_supported_bridge" if ordered else "no_bridge_candidates"),
            "selected_chain": None,
            "selected_chain_rank": None,
            "valid_chains": [],
            "alternative_chains": [],
            "valid_chain_ranks": [],
            "valid_chain_count": 0,
            "fallback_used": False,
            "attempts": attempts,
            "source_body_ids": [],
            "target_body_ids": [],
            "source_pool_size": 0,
            "target_pool_size": 0,
            "coverage_scope": "no supported bridge",
            "failure_reason": (
                attempts[-1].get("reason", "") if attempts else ""),
        }

    selected_rank, selected_chain, selected_pool = selected
    result = dict(selected_pool)
    result["per_linker"] = list(selected_pool.get("per_linker") or [])
    result.update({
        "selected_chain": selected_chain,
        "selected_chain_rank": selected_rank,
        "selected_bridge": selected_chain,
        "fallback_used": selected_rank > 1,
        "fallback_reason": (
            "; ".join(
                attempt.get("reason", "") for attempt in attempts
                if attempt.get("rank", 0) < selected_rank
                and attempt.get("reason"))
            if selected_rank > 1 else ""),
        "selected_source_body_ids": list(
            selected_pool.get("source_body_ids") or []),
        "selected_target_body_ids": list(
            selected_pool.get("target_body_ids") or []),
        "selected_source_pool_size": selected_pool.get(
            "source_pool_size", len(selected_pool.get("source_body_ids") or [])),
        "selected_target_pool_size": selected_pool.get(
            "target_pool_size", len(selected_pool.get("target_body_ids") or [])),
        "valid_chains": [chain for _rank, chain, _pool in valid],
        "alternative_chains": [
            chain for rank, chain, _pool in valid if rank != selected_rank],
        "valid_chain_ranks": [rank for rank, _chain, _pool in valid],
        "valid_chain_count": len(valid),
        "attempts": attempts,
        "resolution_status": "supported",
        "coverage_scope": "selected bridge; all valid alternatives retained",
    })

    def union_ids(side: str) -> List[str]:
        values = set()
        field = f"{side}_body_ids"
        for _rank, _chain, pool in valid:
            values.update(str(body_id) for body_id in pool.get(field) or [])
        return sorted(values)

    all_source = union_ids("source")
    all_target = union_ids("target")
    result.update({
        "all_valid_source_body_ids": all_source,
        "all_valid_target_body_ids": all_target,
        "all_valid_source_pool_size": len(all_source),
        "all_valid_target_pool_size": len(all_target),
        "all_valid_source_type_total": next(
            (pool.get("source_type_total") for _rank, _chain, pool in valid
             if pool.get("source_type_total") is not None), None),
        "all_valid_target_type_total": next(
            (pool.get("target_type_total") for _rank, _chain, pool in valid
             if pool.get("target_type_total") is not None), None),
    })

    def union_basis(side: str) -> str:
        bases = [pool.get(f"{side}_basis") for _rank, _chain, pool in valid]
        measurable = [basis for basis in bases if basis != "unmeasured"]
        if not measurable:
            return "unmeasured"
        if all(basis == "full population" for basis in measurable):
            return "full population"
        return "union of supported bridge pools"

    result["all_valid_source_basis"] = union_basis("source")
    result["all_valid_target_basis"] = union_basis("target")

    # Retain per-linker unions for inspection/export without replacing the
    # selected chain's per-linker counts used by existing visual edges.
    linker_union: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rank, _chain, pool in valid:
        for linker in pool.get("per_linker") or []:
            marker = (str(linker.get("column", "")),
                      str(linker.get("value", "")),
                      str(linker.get("home", "")))
            entry = linker_union.setdefault(marker, {
                "column": linker.get("column", ""),
                "value": linker.get("value", ""),
                "raw_value": linker.get(
                    "raw_value", linker.get("value", "")),
                "canonical_value": linker.get(
                    "canonical_value", linker.get("value", "")),
                "home": linker.get("home", ""),
                "body_ids": [],
                "coverage_basis": linker.get("coverage_basis", ""),
                "evidence_tier": linker.get("evidence_tier", ""),
                "indirect": bool(linker.get("indirect", False)),
                "valid_chain_ranks": [],
            })
            entry["body_ids"] = sorted(set(entry["body_ids"]) | {
                str(body_id) for body_id in linker.get("body_ids") or []})
            entry["valid_chain_ranks"].append(rank)
            if linker.get("relation_rows") is not None:
                entry["relation_rows"] = max(
                    int(entry.get("relation_rows") or 0),
                    int(linker.get("relation_rows") or 0))
    result["all_valid_per_linker"] = list(linker_union.values())
    from collections import Counter

    def overlap_ids(side: str) -> List[str]:
        counts = Counter()
        field = f"{side}_body_ids"
        for _rank, _chain, pool in valid:
            counts.update({str(body_id) for body_id in pool.get(field) or []})
        return sorted(body_id for body_id, count in counts.items()
                      if count > 1)

    result["all_valid_source_overlap_body_ids"] = overlap_ids("source")
    result["all_valid_target_overlap_body_ids"] = overlap_ids("target")
    result["all_valid_source_overlap_count"] = len(
        result["all_valid_source_overlap_body_ids"])
    result["all_valid_target_overlap_count"] = len(
        result["all_valid_target_overlap_body_ids"])
    result["valid_chain_metadata"] = [
        {
            "rank": rank,
            "selected": rank == selected_rank,
            "source_pool_size": pool.get("source_pool_size"),
            "target_pool_size": pool.get("target_pool_size"),
            "source_basis": pool.get("source_basis"),
            "target_basis": pool.get("target_basis"),
            "linkers": [
                {
                    "column": linker.get("column", ""),
                    "raw_value": linker.get(
                        "raw_value", linker.get("value", "")),
                    "canonical_value": linker.get(
                        "canonical_value", linker.get("value", "")),
                    "home": linker.get("home", ""),
                }
                for linker in standardize_bridge(
                    chain, source_dataset, target_dataset)
                if linker.get("kind") == "linker"
            ],
        }
        for rank, chain, pool in valid
    ]
    return result


def build_reverse_type_contexts(
        mapper,
        source_dataset: str,
        target_dataset: str,
        receiving_types,
        *,
        source_index: Optional["CachedNeuronIndex"] = None,
        target_index: Optional["CachedNeuronIndex"] = None,
        coverage_indexes: Optional[Dict[str, Any]] = None,
        warm_pools: Optional[Dict[tuple, Dict[str, Any]]] = None,
        max_sources: int = 24,
) -> Dict[str, Dict[str, Any]]:
    """Dataset-wide incoming context for the backward coverage rows (§12.3).

    For each receiving type, expand the direction-scoped set of source
    types in ``source_dataset`` with a valid bridge onto it (the mapper's
    ``incoming_type_names``), materialize every incoming pair's
    prioritized bridge pool with :func:`resolve_prioritized_bridge_pool`,
    and accumulate the per-receiving-type unions the backward rows
    display: the selected/all-valid source-side unions over ALL incoming
    types (denominator: the incoming types' population union), the
    target-side unions on the receiving type itself, and the per-branch
    overlaps.  Pure with respect to the UI — no NiceGUI state is touched.

    ``warm_pools`` is read (never written) for pools the forward search
    already resolved, so a query member's pair is not resolved twice.
    Fresh resolutions go into a context-local cache only, so the forward
    artifacts' pool set stays exactly what the forward flows resolved.
    ``max_sources`` bounds how many incoming sources are LISTED per
    receiving type; ``incoming_source_count`` and ``truncated`` stay
    honest about the full incoming family.
    """
    from comparison.mapping_visualization import mapping_pool_key

    coverage_indexes = {
        ds: idx for ds, idx in (coverage_indexes or {}).items()
        if idx is not None}
    local_pools: Dict[tuple, Dict[str, Any]] = {}
    counts_cache: Dict[str, Dict[str, int]] = {}
    contexts: Dict[str, Dict[str, Any]] = {}

    def _count(dataset: str, name: str) -> int:
        cache = counts_cache.setdefault(dataset, {})
        if name not in cache:
            index = (source_index if dataset == source_dataset
                     else target_index)
            cache[name] = (int(count_types_in_index(index, [name])
                               .get(name, 0))
                           if index is not None else 0)
        return cache[name]

    def _pool(name: str, receiving: str) -> Optional[Dict[str, Any]]:
        key = mapping_pool_key(
            source_dataset, target_dataset, name, receiving)
        for cache in (local_pools, warm_pools or {}):
            pool = cache.get(key)
            if pool is not None:
                return pool
        try:
            chains = [
                chain for chain in mapper.get_type_bridges(
                    name, source_dataset, target_dataset, max_bridges=0)
                if chain and chain[-1].get("value") == receiving]
        except Exception:
            chains = []
        if not chains:
            return None
        pool = resolve_prioritized_bridge_pool(
            source_dataset, target_dataset, chains, name, receiving,
            indexes=coverage_indexes)
        if pool.get("resolution_status") == "supported":
            local_pools[key] = pool
            return pool
        return None

    for raw_receiving in receiving_types or []:
        receiving = str(raw_receiving or "").strip()
        if not receiving:
            continue
        try:
            verified, discovery_truncated = mapper.incoming_type_names(
                source_dataset, target_dataset, receiving)
        except Exception:
            logger.warning(
                "reverse context discovery failed for %r (%s → %s)",
                receiving, source_dataset, target_dataset, exc_info=True)
            continue
        truncated = bool(discovery_truncated) or len(verified) > max_sources
        listed = verified[:max_sources] if max_sources else verified

        members: List[Dict[str, Any]] = []
        sel_src: set = set()
        all_src: set = set()
        sel_tgt: set = set()
        all_tgt: set = set()
        sel_src_branches: List[set] = []
        all_src_branches: List[set] = []
        sel_tgt_branches: List[set] = []
        all_tgt_branches: List[set] = []
        sel_src_measured = all_src_measured = False
        sel_tgt_measured = all_tgt_measured = False
        pooled_any = False
        for name in listed:
            sel_s: set = set()
            all_s: set = set()
            sel_t: set = set()
            all_t: set = set()
            record = {
                "type": name,
                "count": _count(source_dataset, name),
                "pooled": False,
                "selected_source_pool_size": 0,
                "selected_target_pool_size": 0,
                "all_valid_source_pool_size": 0,
                "all_valid_target_pool_size": 0,
            }
            pool = _pool(name, receiving)
            if pool and pool.get("resolution_status") == "supported":
                pooled_any = True
                sel_s = set(pool.get("source_body_ids") or [])
                sel_t = set(pool.get("target_body_ids") or [])
                all_s = set(
                    pool.get("all_valid_source_body_ids")
                    if pool.get("all_valid_source_body_ids") is not None
                    else pool.get("source_body_ids") or [])
                all_t = set(
                    pool.get("all_valid_target_body_ids")
                    if pool.get("all_valid_target_body_ids") is not None
                    else pool.get("target_body_ids") or [])
                s_measured = pool.get("source_basis") != "unmeasured"
                t_measured = pool.get("target_basis") != "unmeasured"
                s_all_measured = pool.get(
                    "all_valid_source_basis",
                    pool.get("source_basis")) != "unmeasured"
                t_all_measured = pool.get(
                    "all_valid_target_basis",
                    pool.get("target_basis")) != "unmeasured"
                record.update({
                    "pooled": True,
                    "selected_source_pool_size": len(sel_s),
                    "selected_target_pool_size": len(sel_t),
                    "all_valid_source_pool_size": len(all_s),
                    "all_valid_target_pool_size": len(all_t),
                })
                if s_measured:
                    sel_src |= sel_s
                    sel_src_branches.append(sel_s)
                    sel_src_measured = True
                if t_measured:
                    sel_tgt |= sel_t
                    sel_tgt_branches.append(sel_t)
                    sel_tgt_measured = True
                if s_all_measured:
                    all_src |= all_s
                    all_src_branches.append(all_s)
                    all_src_measured = True
                if t_all_measured:
                    all_tgt |= all_t
                    all_tgt_branches.append(all_t)
                    all_tgt_measured = True
            members.append(record)

        def _overlap(branch_lists: List[set]) -> set:
            from collections import Counter

            counts = Counter()
            for branch in branch_lists:
                counts.update(branch)
            return {body_id for body_id, count in counts.items()
                    if count > 1}

        contexts[receiving] = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "receiving_type": receiving,
            "receiving_count": _count(target_dataset, receiving),
            "sources": members,
            "incoming_source_count": len(verified),
            "truncated": truncated,
            "source_population_total": sum(
                m["count"] for m in members),
            "pooled": pooled_any,
            "selected_source_union_ids": sorted(sel_src),
            "all_valid_source_union_ids": sorted(all_src),
            "selected_target_union_ids": sorted(sel_tgt),
            "all_valid_target_union_ids": sorted(all_tgt),
            "source_overlap_selected_ids": sorted(_overlap(sel_src_branches)),
            "source_overlap_all_valid_ids": sorted(_overlap(all_src_branches)),
            "target_overlap_selected_ids": sorted(_overlap(sel_tgt_branches)),
            "target_overlap_all_valid_ids": sorted(_overlap(all_tgt_branches)),
            "selected_source_measured": sel_src_measured,
            "selected_target_measured": sel_tgt_measured,
            "all_valid_source_measured": all_src_measured,
            "all_valid_target_measured": all_tgt_measured,
        }
    return contexts


def _load_alias_index(dataset: str) -> Optional["CachedNeuronIndex"]:
    """Load (and memoize) the cached index used for alias counting."""
    path = neuron_index_path(dataset)
    if not path.is_file():
        return None
    with _INDEX_LOAD_LOCK:
        key = (dataset, path.stat().st_mtime_ns)
        cached = _ALIAS_INDEX_CACHE.get(key)
        if cached is None:
            try:
                cached = load_cached_neuron_index(dataset)
            except Exception:
                return None
            _ALIAS_INDEX_CACHE[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Native cross-dataset type-name expansion (mapper-free search + enrichment)
# ---------------------------------------------------------------------------

# Caps for the native expansion.  Generous enough to keep reasonable entries
# visible; anything beyond is summarized behind a "+N more" note.
# pairs matched by name similarity with NO derivation chain, and
# same-name pairs without any metadata verification, carry this
# warning wherever their provenance text renders (§9F)
NO_DERIVATION_TEXT = "no derivation chain — please double check"
NATIVE_TYPE_MATCH_CAP = 16
NATIVE_LABEL_MATCH_CAP = 8
NATIVE_LABEL_TYPES_CAP = 8

# Broad ordinary searches (a 2-character query can match thousands of
# names) previously retained every native match in ``types_all`` /
# ``covered_all`` and enriched each item through the type mapping walk.
# The query 'CB' on male-cns v1.0 alone fed ~23,000 items into that walk
# and held a warm scan for over a minute — long enough to starve the UI
# websocket.  Keep a ranked but finite tail instead: the "+N more" counts
# stay truthful, whole real families still fit (the largest real label
# coverage and the 'DN' type families are well beneath these bounds), and
# only the explicit uncapped CSV export opts back out.
NATIVE_TYPES_ALL_CAP = 500
NATIVE_LABELS_ALL_CAP = 500
NATIVE_COVERED_ALL_CAP = 96

# Caps for the value-driven mapper fallback: how many matched (column,
# value) pairs, local types, and displayed foreign types per dataset one
# lookup may consume.
VALUE_MATCH_PAIR_CAP = 8
VALUE_MATCH_TYPE_CAP = 64
VALUE_MATCH_FOREIGN_CAP = 24

# Column names (normalized: casefold, non-alphanumerics removed) that carry
# taxonomy labels in the shipped dataset indexes.
_NATIVE_LABEL_COLUMNS = {
    "class", "subclass", "superclass", "cellclass", "celltype", "group",
}


def _native_label_columns(index: "CachedNeuronIndex") -> List[str]:
    """Taxonomy columns of one index, in a stable priority order."""
    priority = ("class", "subclass", "superclass",
                "cellclass", "celltype", "group")
    found = []
    # Cross-dataset scans may keep only the ``type`` column in ``frame`` and
    # answer taxonomy lookups from the compact search sidecar.  Derive the
    # available label names from that sidecar in its stored priority order;
    # falling back to the frame keeps old/test indexes working unchanged.
    columns = list(index.frame.columns)
    search_frame = getattr(index, "search_frame", None)
    if search_frame is not None and {
            "search_column", "search_priority"}.issubset(
                set(search_frame.columns)):
        try:
            columns = [
                str(row["search_column"])
                for row in (
                    search_frame
                    .select(["search_column", "search_priority"])
                    .unique(subset=["search_column"], maintain_order=True)
                    .sort("search_priority")
                    .to_dicts()
                )
            ]
        except Exception:
            columns = list(index.frame.columns)
    for column in columns:
        norm = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        if norm in _NATIVE_LABEL_COLUMNS:
            found.append((priority.index(norm), column))
    found.sort()
    return [column for _, column in found]


def _load_cross_match_index(dataset: str) -> Optional["CachedNeuronIndex"]:
    """Load only the columns needed for native cross-dataset matching.

    The available-neurons table needs the full cached index, but a native
    cross-dataset scan only inspects ``type`` and a small set of taxonomy
    labels.  Reusing ``_load_alias_index`` here used to materialize every
    metadata column and its search sidecar for every cached dataset.  That
    duplicated the selected table's memory footprint during a mapping search
    and could take down the NiceGUI websocket before results were rendered.

    When a compatible ``neuron_index_search.parquet`` sidecar exists, native
    matching uses that distinct-value index and loads only the ``type`` column
    for the small row-ordinal-to-type join used by label coverage.  The old
    projected-column path remains the fallback for legacy indexes without a
    sidecar.

    This helper intentionally does not memoize the projected frame.  The
    caller keeps only the compact match summaries, allowing the projected
    frames to be reclaimed before mapper enrichment starts.
    """
    import polars as pl

    path = neuron_index_path(dataset)
    if not path.is_file():
        return None
    try:
        schema = pl.read_parquet_schema(path)
        columns = list(schema)
        search_path = search_cache_path(path)
        if "type" in columns and search_path.is_file():
            sidecar_schema = pl.read_parquet_schema(search_path)
            required_sidecar = {
                "__neuron_rows", "search_column", "search_priority",
                "search_value", "search_value_folded",
            }
            expected_search_columns = viewer_search_columns(columns)
            sidecar_compatible = required_sidecar.issubset(
                set(sidecar_schema))
            if sidecar_compatible:
                try:
                    priority_rows = (
                        pl.scan_parquet(search_path)
                        .select(["search_column", "search_priority"])
                        .unique(subset=["search_column"],
                                maintain_order=True)
                        .sort("search_priority")
                        .collect()
                        .to_dicts()
                    )
                    actual_search_columns = [
                        str(row.get("search_column") or "")
                        for row in priority_rows
                    ]
                    actual_priorities = [
                        int(row.get("search_priority"))
                        for row in priority_rows
                    ]
                    sidecar_compatible = (
                        actual_search_columns == expected_search_columns
                        and actual_priorities == list(
                            range(len(expected_search_columns)))
                    )
                except Exception:
                    sidecar_compatible = False
            if sidecar_compatible:
                native_columns = [
                    column for column in columns
                    if column == "type"
                    or re.sub(r"[^a-z0-9]", "", str(column).casefold())
                    in _NATIVE_LABEL_COLUMNS
                ]
                # Predicate-push the sidecar read and discard unrelated
                # searchable metadata (bodyId, instance, and arbitrary type
                # fields).  The full sidecar is only a few MB on disk but its
                # list column can be tens of MB when decoded.
                search_frame = (
                    pl.scan_parquet(search_path)
                    .filter(pl.col("search_column").is_in(native_columns))
                    .collect()
                )
                # ``__neuron_rows`` in the sidecar are ordinals into this
                # exact parquet row order.  No bodyId or wide metadata is
                # needed for native type/label matching.
                frame = pl.read_parquet(path, columns=["type"])
                return CachedNeuronIndex(
                    dataset=str(dataset),
                    path=path,
                    frame=frame,
                    columns=("type",),
                    enriched=False,
                    search_frame=search_frame,
                )
        needed = [
            column
            for column in columns
            if column == "type"
            or re.sub(r"[^a-z0-9]", "", str(column).casefold())
            in _NATIVE_LABEL_COLUMNS
        ]
        if not needed:
            return None
        frame = pl.read_parquet(path, columns=needed)
    except Exception:
        return None
    return CachedNeuronIndex(
        dataset=str(dataset),
        path=path,
        frame=frame,
        columns=tuple(frame.columns),
        enriched=False,
        search_frame=None,
    )


def _load_coverage_index(dataset: str) -> Optional["CachedNeuronIndex"]:
    """Load the small bodyId/linker projection used for coverage only.

    Type mapping must not use body IDs to decide which type names map.  The
    preview still reports useful ``m-of-n`` coverage, though, so it needs a
    narrow view of the endpoint rows.  Keep that view separate from both the
    full available-neurons index and the type/taxonomy projection above.

    The projection deliberately includes only the physical bridge columns
    consumed by :func:`pool_bridge_body_ids`; optional bodyId match columns
    are not needed for coverage.
    It is not placed in ``_INDEX_CACHE``: coverage is an annotation for one
    mapping request, not a second full viewer cache.
    """
    import polars as pl

    path = neuron_index_path(dataset)
    if not path.is_file():
        return None
    coverage_columns = {
        "bodyId", "type",
        # MCNS crosswalks and endpoint-side annotation columns.
        "flywireType", "additional_type(s)", "Alternative Cell Type(s)",
        "hemibrainType", "mancType",
        # BANC's curated label bridges.
        "fafb_cell_type", "malecns_cell_type", "manc_cell_type",
        "hemibrain_cell_type",
    }
    try:
        schema = pl.read_parquet_schema(path)
        needed = [column for column in schema if column in coverage_columns]
        if not {"bodyId", "type"}.issubset(needed):
            return None
        frame = pl.read_parquet(path, columns=needed)
    except Exception:
        return None
    return CachedNeuronIndex(
        dataset=str(dataset),
        path=path,
        frame=frame,
        columns=tuple(frame.columns),
        enriched=False,
        search_frame=None,
    )


def _native_type_matches(
    index: "CachedNeuronIndex",
    needle: str,
    cap: int,
    *,
    prefix_only: bool = False,
):
    """Type matches in one index, exact matches ranked first.

    ``prefix_only`` keeps a deliberately short search bounded by matching
    only names that start with the query.  The normal path remains a
    case-insensitive substring search.

    Returns ``(matches, truncated)`` where matches are
    ``{'name', 'count', 'exact', 'matched_written'}`` dicts sorted by
    exact-flag, count (descending), then name.  ``matched_written`` is the
    matched substring in the type's written case (``'apdn3'`` → ``'APDN3'``).
    """
    import polars as pl

    if "type" not in index.frame.columns:
        return [], 0
    folded_needle = needle.casefold()
    search_frame = getattr(index, "search_frame", None)
    grouped_frame = None
    if search_frame is not None and {
            "search_column", "search_value", "search_value_folded",
            "__neuron_rows"}.issubset(set(search_frame.columns)):
        type_rows = search_frame.filter(pl.col("search_column") == "type")
        if not type_rows.is_empty():
            folded = pl.col("search_value_folded").cast(
                pl.Utf8, strict=False)
            match = (
                folded.str.starts_with(folded_needle)
                if prefix_only
                else folded.str.contains(folded_needle, literal=True)
            )
            hits = type_rows.filter(folded.is_not_null() & match)
            if not hits.is_empty():
                # The sidecar has one row per distinct written type and keeps
                # the matching source ordinals as a list, so counting the
                # list length is the indexed equivalent of grouping the wide
                # metadata table by ``type``.
                grouped_frame = hits.select(
                    pl.col("search_value").alias("type"),
                    pl.col("__neuron_rows").list.len().alias("len"),
                )
            else:
                return [], 0
    if grouped_frame is None:
        folded = pl.col("type").cast(pl.Utf8, strict=False).str.to_lowercase()
        match = (
            folded.str.starts_with(folded_needle)
            if prefix_only
            else folded.str.contains(folded_needle, literal=True)
        )
        hits = index.frame.filter(folded.is_not_null() & match)
        if hits.is_empty():
            return [], 0
        grouped_frame = hits.group_by("type").len()
    total_types = grouped_frame.height
    bounded_prefix = bool(prefix_only and cap < 10 ** 9)
    if bounded_prefix:
        # A deliberate one-character search is the safety valve for broad
        # prefixes.  Rank in Polars and trim before converting to Python
        # dicts; sorting/materializing every R* type defeats that bound.
        grouped_frame = (
            grouped_frame
            .with_columns(
                pl.col("type").cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .eq(folded_needle)
                .cast(pl.Int8)
                .alias("__exact")
            )
            .sort(
                ["__exact", "len", "type"],
                descending=[True, True, False],
            )
            .head(max(0, cap))
            .drop("__exact")
        )
    else:
        grouped_frame = grouped_frame.sort("type")
    grouped = grouped_frame.to_dicts()
    matches = []
    for row in grouped:
        name = str(row["type"])
        if not name:
            continue
        matches.append({
            "name": name,
            "count": int(row["len"]),
            "exact": name.casefold() == needle.casefold(),
            # The written form of the matched entry itself (e.g. searching
            # 'apdn3' reports the type 'APDN3', not the typed substring).
            "matched_written": name,
        })
    matches.sort(key=lambda m: (not m["exact"], -m["count"], m["name"].casefold()))
    if bounded_prefix:
        return matches, max(0, total_types - len(matches))
    return matches[:cap], max(0, len(matches) - cap)


def _native_label_matches_from_sidecar(
    index: "CachedNeuronIndex",
    needle: str,
    cap: int,
    types_cap: int,
    *,
    prefix_only: bool = False,
    covered_all_cap: int = 10 ** 9,
):
    """Search taxonomy values from the compact distinct-value sidecar.

    The sidecar stores source row ordinals per value.  Only rows belonging
    to the few displayed labels are looked up in the narrow ``type`` frame, so a
    broad one-character prefix never scans or materializes the wide index.
    ``covered_all_cap`` bounds each label's hidden covered tail for ordinary
    searches (bounded prefix mode caps at ``types_cap`` instead).
    """
    import polars as pl

    search_frame = index.search_frame
    folded_needle = needle.casefold()
    matches = []
    bounded_prefix = bool(prefix_only and cap < 10 ** 9)
    total_matching_labels = 0
    type_by_row = None
    if "type" in index.frame.columns:
        type_by_row = index.frame.with_row_index("__neuron_row").select(
            ["__neuron_row", "type"])

    for column in _native_label_columns(index):
        label_rows = search_frame.filter(
            pl.col("search_column") == column)
        if label_rows.is_empty():
            continue
        folded = pl.col("search_value_folded").cast(
            pl.Utf8, strict=False)
        match = (
            folded.str.starts_with(folded_needle)
            if prefix_only
            else folded.str.contains(folded_needle, literal=True)
        )
        labels_frame = (
            label_rows
            .filter(folded.is_not_null() & match)
            .with_columns(
                pl.col("__neuron_rows").list.len().alias("len"))
            .select(["search_value", "__neuron_rows", "len"])
        )
        if labels_frame.is_empty():
            continue
        total_matching_labels += labels_frame.height
        if bounded_prefix:
            labels_frame = labels_frame.sort(
                ["len", "search_value"], descending=[True, False]
            ).head(max(0, cap))
        else:
            labels_frame = labels_frame.sort(
                ["len", "search_value"], descending=[True, False])

        for label, row_ids, count in labels_frame.iter_rows():
            label = str(label or "")
            if not label or folded_needle not in label.casefold():
                continue
            covered = []
            total_types = 0
            if type_by_row is not None and row_ids:
                types_frame = type_by_row.filter(
                    pl.col("__neuron_row").is_in(row_ids))
                type_groups_frame = types_frame.group_by("type").len()
                total_types = type_groups_frame.height
                type_groups = (
                    type_groups_frame
                    .sort(["len", "type"], descending=[True, False])
                    .head(
                        max(0, types_cap) if bounded_prefix
                        else max(0, covered_all_cap))
                    .to_dicts()
                )
                covered = [
                    {"name": str(group["type"]),
                     "count": int(group["len"])}
                    for group in type_groups if group["type"]
                ]
            matches.append({
                "label": label,
                "column": column,
                "count": int(count),
                "matched_written": label,
                "covered_all": covered,
                "types": covered[:types_cap],
                "types_truncated": max(0, total_types - types_cap),
            })
    matches.sort(key=lambda item: -item["count"])
    if bounded_prefix:
        visible_labels = min(max(0, cap), len(matches))
        return matches[:cap], max(
            0, total_matching_labels - visible_labels)
    return matches[:cap], max(0, len(matches) - cap)


def _native_label_matches(
    index: "CachedNeuronIndex",
    needle: str,
    cap: int,
    types_cap: int,
    *,
    prefix_only: bool = False,
    covered_all_cap: int = 10 ** 9,
):
    """Taxonomy-label matches in one index with their covered types.

    Returns ``(matches, truncated)`` where matches are
    ``{'label', 'column', 'count', 'types': [{'name', 'count'}],
    'types_truncated'}`` sorted by count (descending).
    ``covered_all_cap`` bounds each label's hidden covered tail for ordinary
    searches (bounded prefix mode caps at ``types_cap`` instead).
    """
    import polars as pl

    search_frame = getattr(index, "search_frame", None)
    if search_frame is not None and {
            "__neuron_rows", "search_column", "search_value",
            "search_value_folded"}.issubset(set(search_frame.columns)):
        return _native_label_matches_from_sidecar(
            index, needle, cap, types_cap, prefix_only=prefix_only,
            covered_all_cap=covered_all_cap)

    folded_needle = needle.casefold()
    matches = []
    truncated_labels = 0
    bounded_prefix = bool(prefix_only and cap < 10 ** 9)
    total_matching_labels = 0
    for column in _native_label_columns(index):
        if column not in index.frame.columns:
            continue
        folded_col = pl.col(column).cast(pl.Utf8, strict=False).str.to_lowercase()
        match = (
            folded_col.str.starts_with(folded_needle)
            if prefix_only
            else folded_col.str.contains(folded_needle, literal=True)
        )
        label_rows = index.frame.filter(folded_col.is_not_null() & match)
        if label_rows.is_empty():
            continue
        # Ties broken by the label name: parallel group_by does not
        # guarantee an order, and the display cap must keep the same
        # labels on every run.
        labels_frame = label_rows.group_by(column).len()
        if bounded_prefix:
            # The global display cap is safe to apply per column: every
            # label that could survive the global top-cap is in the top cap
            # of the column where it occurs.  Keep the bound in Polars so a
            # broad prefix never creates a Python object for every label.
            total_matching_labels += labels_frame.height
            labels_frame = (
                labels_frame
                .sort(["len", column], descending=[True, False])
                .head(max(0, cap))
            )
        else:
            labels_frame = labels_frame.sort(
                ["len", column], descending=[True, False]
            )
        labels = labels_frame.to_dicts()
        for row in labels:
            label = str(row[column])
            if folded_needle not in label.casefold():
                continue
            matched_written = label
            types_frame = label_rows.filter(
                pl.col(column) == row[column]
            )
            # Same tie-break policy as the labels: parallel group_by is
            # unordered, and the covered-types cap must be reproducible.
            total_types = 0
            if "type" in types_frame.columns:
                type_groups_frame = types_frame.group_by("type").len()
                total_types = type_groups_frame.height
                type_groups = (
                    type_groups_frame
                    .sort(["len", "type"], descending=[True, False])
                    .head(
                        max(0, types_cap) if bounded_prefix
                        else max(0, covered_all_cap))
                    .to_dicts()
                )
            else:
                type_groups = []
            covered = [
                {"name": str(g["type"]), "count": int(g["len"])}
                for g in type_groups if g["type"]
            ]
            matches.append({
                "label": label,
                "column": column,
                "count": int(row["len"]),
                "matched_written": matched_written,
                # Ordinary searches retain the FULL covered list
                # (``covered_all``) so enrichment transfers every covered
                # type.  Bounded one-character searches intentionally keep
                # only the capped prefix subset here; retaining the hidden
                # tail would recreate the resource spike this mode exists
                # to prevent.
                "covered_all": covered,
                "types": covered[:types_cap],
                "types_truncated": max(0, total_types - types_cap),
            })
    matches.sort(key=lambda m: -m["count"])
    if bounded_prefix:
        visible_labels = min(max(0, cap), len(matches))
        truncated_labels = max(0, total_matching_labels - visible_labels)
    else:
        truncated_labels = max(0, len(matches) - cap)
    return matches[:cap], truncated_labels


def _type_match_expression(chip: str, mode: str):
    """Polars expression matching a lowercased ``type`` column (§12).

    Modes follow the ``apply_filter_mode`` vocabulary (exact / startswith
    / endswith / contains / regex); matching is case-insensitive by
    lowering both sides — the written-case names are recovered from the
    index rows.  Regex chips translate bare ``*`` wildcards (the legacy
    list format) to ``.*`` while keeping explicit ``.*`` meta sequences
    intact, so mixed patterns like ``A*B.*`` translate the wildcard only.
    Returns None for an invalid regex so the caller can fall back to the
    staged native search.
    """
    mode = str(mode or "exact").strip().lower().replace(" ", "")
    aliases = {"startswith": "startswith", "startwith": "startswith",
               "endswith": "endswith", "endwith": "endswith",
               "contains": "contains", "regex": "regex"}
    mode = aliases.get(mode, "exact")
    import polars as pl

    def _folded():
        return pl.col("type").cast(pl.Utf8, strict=False).str.to_lowercase()

    # `.lower()` on both sides (not casefold): polars lowers the column
    # with to_lowercase, so a casefolded chip (ß → ss) would never match
    # its lowercased counterpart.
    needle = chip.lower()

    if mode == "exact":
        return _folded() == needle
    if mode == "startswith":
        return _folded().str.starts_with(needle)
    if mode == "endswith":
        return _folded().str.ends_with(needle)
    if mode == "contains":
        return _folded().str.contains(needle, literal=True)
    # regex: bare '*' wildcards translate like the legacy list format
    # ('*' not already preceded by a '.'); the pattern is lowered to
    # match the lowered column (documented case-insensitive semantics)
    pattern = re.sub(r"(?<!\.)\*", ".*", chip)
    pattern = pattern.lower()
    try:
        re.compile(pattern)
    except re.error:
        return None
    return _folded().str.contains(pattern, literal=False)


def resolve_type_matches(queries, mode: str, datasets,
                         indexes: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Resolve panel chips to written type names per dataset (§12).

    Every chip is matched against each selected dataset's cached index
    ``type`` column under the active filter mode (see
    ``_type_match_expression``).  Returns::

        {'origins': {dataset: [written type names]},
         'origin_matches': {dataset: {type: [{column, value}]}},
         'fallback_chips': [chips that matched NOTHING anywhere],
         'notes': [human-readable resolution notes]}

    The fallback chips go through the staged native search (substring
    types + taxonomy labels → pooled nodes) so coarse queries and the
    label-pooling feature keep working under explicit modes.
    """
    import polars as pl

    indexes = dict(indexes or {})
    origins: Dict[str, List[str]] = {ds: [] for ds in datasets}
    origin_matches: Dict[str, Dict[str, set]] = {}
    fallback_chips: List[str] = []
    notes: List[str] = []

    def _record_origin(dataset: str, type_name: str, column: str,
                       value: str) -> None:
        type_name = str(type_name or '').strip()
        column = str(column or '').strip()
        value = str(value or '').strip()
        if not type_name or not column:
            return
        origin_matches.setdefault(dataset, {}).setdefault(
            type_name, set()).add((column, value or type_name))

    def _label_matches_mode(label: str, chip: str, mode: str) -> bool:
        """Apply the panel's filter mode to one taxonomy label."""
        lab = label.casefold()
        chip_l = chip.casefold()
        if mode == "exact":
            return lab == chip_l
        if mode == "startswith":
            return lab.startswith(chip_l)
        if mode == "endswith":
            return lab.endswith(chip_l)
        if mode == "regex":
            try:
                return re.search(chip, label, flags=re.IGNORECASE) is not None
            except re.error:
                return False
        return chip_l in lab  # contains

    for chip in [str(q).strip() for q in (queries or []) if str(q).strip()]:
        expr = _type_match_expression(chip, mode)
        matched_any = False
        if expr is not None:
            for ds in datasets:
                index = indexes.get(ds) or load_cached_neuron_index(ds)
                if index is None or "type" not in index.frame.columns:
                    continue
                folded = pl.col("type").cast(
                    pl.Utf8, strict=False).str.to_lowercase()
                hits = index.frame.filter(
                    folded.is_not_null() & expr)
                # dedupe INSIDE polars — a wide `contains` can match tens
                # of thousands of rows and to_dicts materializes each one
                names = sorted({
                    str(row["type"])
                    for row in hits.select("type").unique().to_dicts()
                    if row["type"]})
                if names:
                    matched_any = True
                    origins[ds].extend(names)
                    for name in names:
                        _record_origin(ds, name, "type", name)
        if not matched_any:
            # Taxonomy-scope resolution (H2-1): a chip that names a
            # taxonomy VALUE (e.g. FAFB cell_type 'circadian_clock')
            # expands to the member types of the pooled rows — the same
            # label columns the staged native sweep searches, resolved
            # DIRECTLY so the mapping runs from the member types (H2-4:
            # one resolution feeding every downstream count).
            for ds in datasets:
                index = indexes.get(ds) or load_cached_neuron_index(ds)
                if index is None:
                    continue
                labels, _trunc = _native_label_matches(
                    index, chip, 10 ** 9, 10 ** 9)
                member: Dict[str, int] = {}
                for m in labels:
                    label = str(m.get("label", ""))
                    if not _label_matches_mode(label, chip, mode):
                        continue
                    label_column = str(m.get("column", "") or "")
                    for t in m.get("types", []):
                        name = str(t.get("name", ""))
                        if name:
                            member[name] = int(t.get("count", 0))
                            _record_origin(ds, name, label_column, label)
                if member:
                    matched_any = True
                    origins[ds].extend(sorted(member))
                    label_columns = sorted({
                        column for name in member
                        for column, _value in origin_matches.get(
                            ds, {}).get(name, set())
                    })
                    label_values = sorted({
                        value for name in member
                        for _column, value in origin_matches.get(
                            ds, {}).get(name, set())
                    })
                    notes.append(
                        f"'{chip}' resolved via taxonomy column "
                        f"'{', '.join(label_columns)}' in {ds} (label "
                        f"'{', '.join(label_values)}'): {len(member)} types, "
                        f"{sum(member.values())} neurons")
        if not matched_any:
            fallback_chips.append(chip)
            notes.append(
                f"'{chip}' matched no type under the active filter — "
                "resolved via labels/substring (please double check)")
    return {
        "origins": {ds: sorted(set(v))
                    for ds, v in origins.items() if v},
        "origin_matches": {
            ds: {
                type_name: [
                    {"column": column, "value": value}
                    for column, value in sorted(records)
                ]
                for type_name, records in sorted(by_type.items())
            }
            for ds, by_type in sorted(origin_matches.items())
        },
        "fallback_chips": fallback_chips,
        "notes": notes,
    }


def collect_native_type_matches(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
    *,
    uncapped: bool = False,
    prefix_only_search: bool = False,
    should_abort: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    """Native type-name expansion for a zero-hit viewer search.

    Mapper-free by design: the search text is matched as a case-insensitive
    substring against the ``type`` column and the taxonomy label columns of
    every *other* locally cached dataset's index.  With
    ``prefix_only_search=True``, only case-insensitive starts-with matches are
    returned; this is the bounded mode used for deliberate one-character
    searches.  Results are name-similar entries, not mapped equivalences, and
    stay strictly informational.  With ``uncapped=True`` every match is
    returned (used by the CSV export).  ``should_abort`` is polled between
    datasets so a superseded scan abandons the remaining parquet loads.
    """
    search = str(search or "").strip()
    if (
        not search
        or search.isdigit()
        or "*" in search
        or len(search) < (1 if prefix_only_search else 2)
    ):
        return []
    if datasets is None:
        datasets = datasets_with_cached_indexes()
    datasets = [
        ds for ds in datasets
        if ds != dataset and neuron_index_path(ds).is_file()
    ]

    matches = []
    type_cap = 10 ** 9 if uncapped else NATIVE_TYPE_MATCH_CAP
    label_cap = 10 ** 9 if uncapped else NATIVE_LABEL_MATCH_CAP
    types_cap = 10 ** 9 if uncapped else NATIVE_LABEL_TYPES_CAP
    for ds in datasets:
        if should_abort is not None and should_abort():
            return matches
        index = _load_cross_match_index(ds)
        if index is None:
            continue
        # Normal searches keep complete lists in ``*_all`` because the
        # mapped-type view and CSV export consume them.  A forced one-letter
        # search is different: it is explicitly the bounded performance
        # mode, so do not materialize/map the hidden tail just to discard it
        # at render time.  An explicit uncapped CSV request still opts into
        # the complete prefix result.
        bounded_prefix = bool(prefix_only_search and not uncapped)
        if uncapped or bounded_prefix:
            # Explicit uncapped export keeps every match; a deliberate
            # bounded one-character search keeps only the display-sized
            # prefix lists (``type_cap`` etc. are the finite display caps).
            match_cap = type_cap
            label_match_cap = label_cap
            covered_all_cap = types_cap
        else:
            # Ordinary searches keep a ranked but finite all-list so a
            # broad query cannot feed tens of thousands of items into the
            # per-item mapping enrichment.
            match_cap = NATIVE_TYPES_ALL_CAP
            label_match_cap = NATIVE_LABELS_ALL_CAP
            covered_all_cap = NATIVE_COVERED_ALL_CAP
        types_all, types_truncated = _native_type_matches(
            index, search, match_cap, prefix_only=prefix_only_search
        )
        labels_all, labels_truncated = _native_label_matches(
            index,
            search,
            label_match_cap,
            types_cap,
            prefix_only=prefix_only_search,
            covered_all_cap=covered_all_cap,
        )
        types = types_all[:type_cap]
        labels = labels_all[:label_cap]
        # The helpers' counts hide everything beyond the all-list cap, and
        # the display slice hides the tail of the kept list; both are
        # hidden from the panel, so report their sum.
        types_truncated = int(types_truncated) + max(
            0, len(types_all) - len(types))
        labels_truncated = int(labels_truncated) + max(
            0, len(labels_all) - len(labels))
        if types or labels:
            matched_written = ""
            for cand in types_all:
                if cand.get("exact"):
                    matched_written = cand["matched_written"]
                    break
            if not matched_written and types_all:
                matched_written = types_all[0]["matched_written"]
            matches.append({
                "dataset": ds,
                "is_selected": False,
                "types": types,
                "types_all": types_all,
                "types_truncated": types_truncated,
                "labels": labels,
                "labels_all": labels_all,
                "labels_truncated": labels_truncated,
                "matched_written": matched_written,
            })
    return matches


def mapped_type_targets(mapper, foreign_type: str, foreign_ds: str,
                        selected_ds: str,
                        alias_cache: Optional[Dict[Any,
                                                   Optional[Dict[str, Any]]]] = None,
                        bridge_cache: Optional[Dict[Tuple[str, str, str],
                                                      List[List[Dict[str, str]]]]] = None,
                        ) -> Optional[Dict[str, Any]]:
    """Canonical mapped-target resolution — the UI adapter.

    The core validity-aware decision lives in the comparison layer
    (``comparison.type_resolver.resolve_valid_targets``), shared with
    homolog finding and connectivity-profile comparison; this wrapper only
    reshapes the typed :class:`TypeResolution` into the response shape this
    module's callers consume (``{'kind', 'targets'[, 'status', ...]}`` or
    ``None``).  Both the 'See available neurons' auto-initiated type
    mapping (the viewer's mapped view via ``enrich_native_type_matches``)
    and the cross-dataset tab's Type Mapping panel (its summary via
    ``_compute``) resolve every foreign type through THIS function.
    """
    from comparison.type_resolver import (
        STATUS_CONFLICT, STATUS_MAPPER_UNAVAILABLE, STATUS_MAPPED,
        STATUS_UNMAPPED, resolve_valid_targets,
    )

    res = resolve_valid_targets(
        mapper, foreign_type, foreign_ds, selected_ds,
        alias_cache=alias_cache, bridge_cache=bridge_cache)
    if res.status == STATUS_MAPPER_UNAVAILABLE:
        return None
    if res.status == STATUS_CONFLICT:
        out = {
            'kind': 'conflict',
            'targets': [],
            'status': 'conflict',
            'source_dataset': res.source_dataset,
            'target_dataset': res.target_dataset,
            'conflicts': [dict(c) for c in res.conflicts],
        }
        # A HELD same-name fan-out is a conflict with a story: carry the
        # mapper's verdict so the viewer can explain WHY nothing was
        # selected instead of leaving a bare conflict (plan-ui-type-mapper-
        # alignment §5.2).  Disclosure only.
        if res.suspect_rivals:
            out['suspect_rivals'] = list(res.suspect_rivals)
        return out
    if res.status == STATUS_UNMAPPED and not res.target_types:
        return None
    result: Dict[str, Any] = {'kind': res.kind, 'targets': list(res.target_types)}
    if res.status != STATUS_MAPPED or res.kind in ('one of N', 'splits into'):
        result['status'] = res.status
    # Same-name-first disclosure (plan §5.1): a deliberate SELECTION must be
    # distinguishable from a bare same-name echo, and a curated identity from
    # both — the mapper's own flags, never re-derived here.
    if getattr(res, 'suspects', False):
        result['suspects'] = True
        result['suspect_rivals'] = list(res.suspect_rivals or ())
    if res.kind == 'same name':
        result['curated_identity'] = bool(res.curated_identity)
    return result


def enrich_native_type_matches(
    native_matches: List[Dict[str, Any]],
    selected_dataset: str,
    *,
    should_abort: Optional[Callable[[], bool]] = None,
) -> None:
    """Annotate native type matches with their mapped names, in place.

    For every matched foreign type, the shared backend
    (``mapped_type_targets``) resolves what that type corresponds to in
    the *selected* dataset (unique rename, same name, the members of a
    refused N-to-1 aggregation, or the bridge ends).  Types with no
    counterpart keep ``annotation=None`` and stay visible unmapped, so
    the user is still led to inspect them in the other dataset.  Mapper
    failures simply leave everything unannotated.

    ``should_abort`` lets a superseded scan abandon the walk between
    entries and labels; the caller discards the partial result anyway.
    """
    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper

        mapper = get_type_mapper()
    except Exception:
        return
    if mapper is None or not getattr(mapper, "_loaded", False):
        return

    cache: Dict[str, Optional[Dict[str, Any]]] = {}
    # A single expansion often sees the same type once as a native type and
    # again under one or more taxonomy labels.  Keep graph walks keyed by
    # direction as well as name; mapper results are immutable for the life of
    # this worker and repeating the walk was a major R1 memory/time multiplier.
    bridge_cache: Dict[Tuple[str, str, str], List[List[Dict[str, str]]]] = {}

    def _annotation_for(foreign_type: str, foreign_ds: str) \
            -> Optional[Dict[str, Any]]:
        return mapped_type_targets(
            mapper, foreign_type, foreign_ds, selected_dataset,
            alias_cache=cache, bridge_cache=bridge_cache)

    for entry in native_matches:
        if should_abort is not None and should_abort():
            return
        foreign_ds = entry.get("dataset", "")
        mapped_names = set()
        # Complete lists for ordinary searches (or the deliberately bounded
        # prefix lists for a forced one-character search): the display cap
        # must never shrink the mapped-type set (the l-LNv/DN1a regression
        # class).
        types_iter = entry.get("types_all") or entry.get("types", [])
        labels_iter = entry.get("labels_all") or entry.get("labels", [])

        def _map_used(local_target: str, foreign_type: str) -> str:
            """Human-readable 'map used' text for one mapped pair.

            Same standardized-linker source as the provenance hover and
            the CSV bridge columns (§9B.3 unification) — one consistent
            derivation text across datasets and surfaces.
            """
            try:
                from comparison.cross_dataset_type_mapper import (
                    bridge_linker_text,
                )

                bridge_key = (
                    str(local_target), str(selected_dataset), str(foreign_ds))
                if bridge_key not in bridge_cache:
                    try:
                        bridge_cache[bridge_key] = mapper.get_type_bridges(
                            local_target, selected_dataset, foreign_ds,
                            max_bridges=8)
                    except Exception:
                        bridge_cache[bridge_key] = []
                chains = bridge_cache[bridge_key]
                info = bridge_linker_text(
                    chains, selected_dataset, foreign_ds, foreign_type)
            except Exception:
                return NO_DERIVATION_TEXT
            return info["text"] or NO_DERIVATION_TEXT

        for cand in types_iter:
            cand["mapped"] = _annotation_for(cand["name"], foreign_ds)
            if cand["mapped"]:
                mapped_names.update(cand["mapped"]["targets"])
                cand["map_used"] = "; ".join(
                    f"{target}: {_map_used(target, cand['name'])}"
                    for target in cand["mapped"]["targets"]
                )
                cand["bridges_by_target"] = {
                    bridge_target: bridge_cache.get(
                        (str(bridge_target), str(selected_dataset),
                         str(foreign_ds)), [])
                    for bridge_target in cand["mapped"]["targets"]
                }
        for label in labels_iter:
            # Map the FULL covered list (covered_all): the display-capped
            # ``types`` would silently drop tail types (lowercase names
            # sort last) from the mapped-type view.
            if should_abort is not None and should_abort():
                return
            for covered in (label.get("covered_all")
                            or label.get("types", [])):
                covered["mapped"] = _annotation_for(
                    covered["name"], foreign_ds)
                if covered["mapped"]:
                    mapped_names.update(covered["mapped"]["targets"])
                    covered["map_used"] = "; ".join(
                        f"{bridge_target}: {_map_used(bridge_target, covered['name'])}"
                        for bridge_target in covered["mapped"]["targets"]
                    )
                    covered["bridges_by_target"] = {
                        bridge_target: bridge_cache.get(
                            (str(bridge_target), str(selected_dataset),
                             str(foreign_ds)), [])
                        for bridge_target in covered["mapped"]["targets"]
                    }
        # Current-dataset type names this block's matches map to — the
        # mapped-type view's equivalent search set.
        entry["mapped_type_names"] = sorted(mapped_names)


def _same_name_asymmetry_note(mapper, name: str,
                              datasets: List[str]) -> str:
    """Population-asymmetry note for same-name candidates (user
    2026-09-14): when the same type name's populations across the queried
    datasets that HAVE it differ by an order of magnitude, return the
    suggested-check text ('' otherwise).  Bookkeeping only — never a
    mapping decision."""
    try:
        counts = {ds: mapper.get_type_population(name, ds)
                  for ds in datasets}
    except Exception:
        return ""
    present = {ds: n for ds, n in counts.items() if n > 0}
    if len(present) < 2:
        return ""
    hi_ds = max(present, key=present.get)
    lo_ds = min(present, key=present.get)
    if present[lo_ds] * 10 > present[hi_ds]:
        return ""
    return (f"extreme population asymmetry across datasets "
            f"({present[lo_ds]} in {lo_ds} vs {present[hi_ds]} in "
            f"{hi_ds}; suggested check)")


def collect_alias_matches(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Expand a zero-hit viewer search into cross-dataset alias matches.

    The search name is resolved through the cross-dataset type mapper into
    per-dataset alias candidates (kind + orthogonal aggregation annotation,
    see ``CrossDatasetTypeMapper.get_alias_candidates``), then each alias is
    counted in that dataset's cached neuron index.  Only locally cached
    indexes are consulted — nothing is fetched.

    The result is strictly informational: callers must never merge these
    rows into the selected dataset's table or selection, because the bodyIds
    belong to a different dataset.
    """
    search = str(search or "").strip()
    if not search or search.isdigit() or "*" in search or len(search) < 2:
        return []
    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper

        mapper = get_type_mapper()
    except Exception:
        return []
    if mapper is None or not getattr(mapper, "_loaded", False):
        return []

    if datasets is None:
        datasets = datasets_with_cached_indexes()
    datasets = [ds for ds in datasets if neuron_index_path(ds).is_file()]
    if not datasets:
        return []
    try:
        outcomes = mapper.get_alias_candidates(search, datasets)
    except Exception:
        return []

    matches: List[Dict[str, Any]] = []
    for ds in datasets:
        info = outcomes.get(ds) or {}
        outcome = info.get("outcome")
        if outcome in (None, "not applicable", "mapper unavailable"):
            continue
        entry: Dict[str, Any] = {
            "dataset": ds,
            "is_selected": ds == dataset,
            "outcome": outcome,
            "candidates": [],
        }
        if outcome == "matched":
            # Alias counts only inspect the exact ``type`` column.  Keep the
            # cross-dataset lookup projection-only; loading the full display
            # index here can duplicate hundreds of megabytes per dataset.
            index = _load_cross_match_index(ds)
            if index is None:
                continue
            # Same-name disclosure for the alias list (plan-ui-type-mapper-
            # alignment §5.1/§5.3).  ONE scoped decision per (search -> ds)
            # pair: the mapper's own verdict, never re-derived.  Disclosure
            # only — never a merge, never a selection change.
            _decision: Dict[str, Any] = {}
            try:
                _decision = mapper.get_mapping_decision(
                    search, dataset, ds, include_bridges=False) or {}
            except Exception:
                _decision = {}
            _fired = bool(_decision.get('suspects'))
            _curated = bool(_decision.get('status') == 'mapped'
                            and _decision.get('target_type'))
            for cand in info.get("candidates", []):
                _cand_extra: Dict[str, Any] = {}
                if cand["kind"] == "same name":
                    if _fired:
                        _cand_extra["suspects"] = True
                        _cand_extra["suspect_rivals"] = list(
                            (_decision.get('same_name_first') or {}
                             ).get('rivals') or ())
                    elif _curated:
                        # The scoped decision resolves this pair — the alias
                        # list must not call that "no metadata verification".
                        _cand_extra["curated_identity"] = True
                entry["candidates"].append({
                    "name": cand["name"],
                    "kind": cand["kind"],
                    "aggregates": cand.get("aggregates"),
                    "count": count_type_in_index(index, cand["name"]),
                    **_cand_extra,
                })
            # Same-name fidelity flag (user 2026-09-14): a same-name
            # candidate whose population is an order of magnitude below
            # the same name in another queried dataset is flagged as a
            # suggested check — carried evidence, never a decision.
            same = [c for c in entry["candidates"]
                    if c["kind"] == "same name"]
            if same:
                note = _same_name_asymmetry_note(
                    mapper, search, datasets)
                if note:
                    for c in same:
                        c["asymmetry_note"] = note
        matches.append(entry)

    matches.sort(key=lambda entry: not entry["is_selected"])
    return matches


def collect_zero_hit_matches(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
    matched_values: Optional[List[tuple]] = None,
    *,
    prefix_only_search: bool = False,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Expansion tiers for a viewer search's cross-dataset panel.

    ``native``: mapper-free, name-similar type and taxonomy-label matches
    from the other datasets' cached indexes (see
    :func:`collect_native_type_matches`), enriched in place with mapped
    current-dataset names where the type mapper knows them.

    ``mapped``: the auto-type-mapping alias candidates for the query itself
    (see :func:`collect_alias_matches`).

    ``value_mapped``: mapper-driven counterparts of the search's OWN
    matched values (see :func:`collect_value_mapped_matches`) — only
    computed when ``matched_values`` (the ``(column, value)`` pairs of
    the current search's match groups) is supplied.  This is the fallback
    for non-type column entries (a ``cell_type`` value, for example)
    whose literal string appears nowhere in the other datasets.

    ``guidance``: datasets checked alongside ``value_mapped`` that have no
    automatic mapping (cached but unmapped, or metadata not downloaded).

    All tiers are strictly informational.

    ``prefix_only_search`` is used for an explicitly submitted
    one-character query.  It applies the same starts-with-only guard to the
    native cross-dataset tier so that enabling this panel cannot re-expand a
    bounded local search into a broad substring scan.

    ``should_abort`` is a UI-owned predicate polled between enrichment
    steps; a superseded scan abandons quickly instead of finishing its
    full walk.  It must stay un-picklable-safe: the spawned-process
    entry point (:func:`collect_zero_hit_matches_in_process`) never
    receives it, and its process isolation already bounds stale work.
    """
    # Keep the entire expansion single-flight.  The native projection is
    # bounded, but mapper enrichment and value-driven fallback can still hold
    # large temporary graph/index structures; overlapping requests were the
    # path to the connection disappearing under R1 and other short queries.
    return run_serialized_cross_dataset_scan(
        _collect_zero_hit_matches,
        dataset,
        search,
        datasets,
        matched_values,
        prefix_only_search=prefix_only_search,
        should_abort=should_abort,
    )


def collect_zero_hit_matches_in_process(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
    matched_values: Optional[List[tuple]] = None,
    *,
    prefix_only_search: bool = False,
) -> Dict[str, Any]:
    """Pickle-safe process entry point for the viewer's mapper expansion.

    Keep this as a module-level function so ``spawn`` can import it without
    serializing a UI closure.  The process-local mapper and index caches stay
    in the dedicated worker and never compete with the websocket process.
    """
    return _collect_zero_hit_matches(
        dataset,
        search,
        datasets,
        matched_values,
        prefix_only_search=prefix_only_search,
    )


def _collect_zero_hit_matches(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
    matched_values: Optional[List[tuple]] = None,
    *,
    prefix_only_search: bool = False,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Implementation for :func:`collect_zero_hit_matches`.

    Kept separate so the public collector can enforce the single-flight
    boundary without making the worker callback acquire a non-reentrant lock.
    """
    if datasets is None:
        datasets = datasets_with_cached_indexes()
    native = collect_native_type_matches(
        dataset,
        search,
        datasets,
        prefix_only_search=prefix_only_search,
        should_abort=should_abort,
    )
    try:
        enrich_native_type_matches(
            native, dataset, should_abort=should_abort)
    except Exception:
        pass
    if should_abort is not None and should_abort():
        return {"native": [], "mapped": [], "value_mapped": [],
                "guidance": []}
    try:
        mapped = collect_alias_matches(dataset, search, datasets)
    except Exception:
        mapped = []
    value_mapped: List[Dict[str, Any]] = []
    guidance: List[Dict[str, Any]] = []
    if matched_values and not (
            should_abort is not None and should_abort()):
        try:
            value_mapped, guidance = collect_value_mapped_matches(
                dataset, matched_values, datasets)
        except Exception:
            value_mapped, guidance = [], []
        try:
            enrich_native_type_matches(
                value_mapped, dataset, should_abort=should_abort)
        except Exception:
            pass
    return {
        "native": native,
        "mapped": mapped,
        "value_mapped": value_mapped,
        "guidance": guidance,
    }


def collect_value_mapped_matches(
    dataset: str,
    matched_values: List[tuple],
    datasets: Optional[List[str]] = None,
) -> tuple:
    """Mapper-driven counterparts of the current search's matched values.

    ``matched_values`` carries ``(column, value)`` pairs from the
    viewer's match groups — entries of non-type columns (a ``cell_type``
    or ``class`` value, for example) that are dataset-specific: their
    literal string appears in no other dataset, so the name-similar scan
    and the query-alias lookup both come up empty.  The local neurons
    carrying those values contribute their ``type`` names, and the
    cross-dataset type mapper resolves those into every other cached
    dataset's types (BANC v888, male-cns:v1.0, and FAFB carry default
    mappings).

    Returns ``(blocks, guidance)``.  ``blocks`` are native-tier-shaped
    entries (dataset, types, types_all, ...) so the viewer renders them
    beside the name-similar tier and the mapped-type view, provenance,
    and CSV reuse them unchanged; run them through
    :func:`enrich_native_type_matches` for the mapped annotations.
    ``guidance`` lists the remaining datasets as ``{'dataset', 'cached'}``
    — no automatic mapping exists for them, so the viewer points the user
    at the Cross-Dataset tab's type mapping panel (and, when not cached,
    at downloading the metadata first).
    """
    import polars as pl

    pairs: List[tuple] = []
    seen_pairs = set()
    for column, value in (matched_values or []):
        column = str(column or "").strip()
        value = str(value or "").strip()
        if not column or not value:
            continue
        normalized_column = re.sub(r"[^a-z0-9]", "", column.casefold())
        if normalized_column in {"bodyid", "type"}:
            # bodyIds are dataset-specific, while type values already have a
            # native cross-dataset tier.  This fallback is specifically for
            # metadata/taxonomy values whose literal text is not itself a
            # type query; remapping type groups here duplicates the mapper
            # graph walk and was the R1 amplification path.
            continue
        key = (column.casefold(), value.casefold())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pairs.append((column, value))
        if len(pairs) >= VALUE_MATCH_PAIR_CAP:
            break
    if not pairs:
        return [], []

    index = _load_alias_index(dataset)
    if index is None or "type" not in index.frame.columns:
        return [], []
    frame = index.frame

    # The matched values identify a local neuron set; its type names are
    # what the type mapper can translate.
    local_types: Dict[str, int] = {}
    for column, value in pairs:
        if column not in frame.columns:
            continue
        column_expr = (
            pl.col(column).cast(pl.Utf8, strict=False)
            .fill_null("").str.strip_chars()
        )
        rows = frame.filter(column_expr == value)
        if rows.is_empty():
            continue
        for row in rows.group_by("type").len().to_dicts():
            name = str(row["type"] or "").strip()
            if name:
                local_types[name] = local_types.get(name, 0) + int(row["len"])
    if not local_types:
        return [], []
    ranked = sorted(local_types.items(),
                    key=lambda item: (-item[1], item[0].casefold()))
    local_types = dict(ranked[:VALUE_MATCH_TYPE_CAP])

    if datasets is None:
        datasets = datasets_with_cached_indexes()

    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper

        mapper = get_type_mapper()
    except Exception:
        mapper = None

    blocks: List[Dict[str, Any]] = []
    mapped_datasets: List[str] = []
    for ds in datasets:
        if ds == dataset:
            continue
        foreign: Dict[str, set] = {}
        if mapper is not None:
            for local_type in local_types:
                try:
                    res = mapper.get_alias_candidates(local_type, [ds])
                    info = res.get(ds) or {}
                    if info.get("outcome") == "matched":
                        for cand in info.get("candidates") or []:
                            name = str(cand.get("name") or "").strip()
                            if name:
                                foreign.setdefault(name, set()).add(local_type)
                except Exception:
                    pass
                try:
                    chains = mapper.get_type_bridges(local_type, dataset, ds)
                    for chain in chains or []:
                        end = str((chain[-1] or {}).get("value") or "").strip()
                        if end:
                            foreign.setdefault(end, set()).add(local_type)
                except Exception:
                    pass
        if not foreign:
            continue
        names = sorted(foreign, key=lambda name: (name.casefold(), name))
        # Only exact type counts are needed to render value-driven matches.
        # The selected dataset remains on its full index above because it is
        # also used to resolve the matched value into local types; foreign
        # datasets must stay projection-only to avoid a memory spike.
        foreign_index = _load_cross_match_index(ds)
        types_all = [
            {
                "name": name,
                "count": count_type_in_index(foreign_index, name) or 0,
                "exact": False,
                "matched_written": name,
            }
            for name in names
        ]
        blocks.append({
            "dataset": ds,
            "is_selected": False,
            "types": types_all[:VALUE_MATCH_FOREIGN_CAP],
            "types_all": types_all,
            "types_truncated": max(0, len(types_all) - VALUE_MATCH_FOREIGN_CAP),
            "labels": [],
            "labels_all": [],
            "labels_truncated": 0,
            "matched_written": pairs[0][1],
            "value_source": pairs[0][0],
        })
        mapped_datasets.append(ds)

    guidance = [
        {"dataset": ds, "cached": neuron_index_path(ds).is_file()}
        for ds in datasets
        if ds != dataset and ds not in mapped_datasets
    ]
    return blocks, guidance


def _bridge_cells(bridges_by_target, selected_dataset: str,
                  foreign_ds: str, foreign_type: str) -> Dict[str, List[str]]:
    """``bridge-<column>`` cell values for one matched entry (§9.3).

    Chains are filtered to the entry's foreign type (a chain that walks
    on to another primary — e.g. DN1pA → DN2 — describes a DIFFERENT
    pair), standardized, and deduped per linker column; indirect
    (hub-route) values carry the via note.
    """
    import re

    from comparison.cross_dataset_type_mapper import bridge_linker_text

    cells: Dict[str, List[str]] = {}
    for chains in (bridges_by_target or {}).values():
        usable = [c for c in (chains or [])
                  if c and c[-1].get("value") == foreign_type]
        if not usable:
            continue
        info = bridge_linker_text(
            usable, selected_dataset, foreign_ds, foreign_type)
        for entry in info["entries"]:
            if entry.get("kind") not in (None, "linker"):
                continue  # 'same name' markers carry no bridge column
            column = f"bridge-{entry['column']}"
            value = str(entry["value"])
            if entry.get("indirect"):
                hub = re.search(r"\(via ([^)]+)\)",
                                str(entry.get("text", "")))
                value += (f" (via {hub.group(1)})" if hub
                          else " (indirect)")
            bucket = cells.setdefault(column, [])
            if value not in bucket:
                bucket.append(value)
    return cells


def build_matches_csv(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
    matched_values: Optional[List[tuple]] = None,
    *,
    prefix_only_search: bool = False,
) -> str:
    """Build the CSV text of every matched entry for a zero-hit search.

    The uncapped expansion (native type matches, taxonomy-label matches
    with their covered types, and the query-alias candidates) is
    exported in full — nothing hidden behind the panel's display caps.

    Uniform schema (§9.3): one row per matched entry per foreign type —
    labels are EXPLODED per covered type — with the standardized bridge
    as named ``bridge-<column>`` columns.  Every row carries exactly the
    same fields (readers like Tablecruncher reject ragged rows).

    Returns the CSV text; empty string when there is nothing to export.
    """
    import csv
    import io

    if datasets is None:
        datasets = datasets_with_cached_indexes()
    native = collect_native_type_matches(
        dataset,
        search,
        datasets,
        uncapped=True,
        prefix_only_search=prefix_only_search,
    )
    enrich_native_type_matches(native, dataset)
    if matched_values:
        try:
            value_blocks, _guidance = collect_value_mapped_matches(
                dataset, matched_values, datasets)
            enrich_native_type_matches(value_blocks, dataset)
            native = native + value_blocks
        except Exception:
            pass
    try:
        mapped = collect_alias_matches(dataset, search, datasets)
    except Exception:
        mapped = []

    base_fields = [
        "dataset", "entry_kind", "matched_column", "name",
        "foreign_type", "neuron_count", "mapped_kind", "mapped_to",
        "map_used",
    ]
    rows: List[Dict[str, Any]] = []
    bridge_columns: List[str] = []
    seen_bridge_columns = set()

    def _row(entry_kind, matched_column, name, foreign_type, count,
             ann, map_used, cells):
        row = {
            "dataset": ds,
            "entry_kind": entry_kind,
            "matched_column": matched_column,
            "name": name,
            "foreign_type": foreign_type,
            "neuron_count": count,
            "mapped_kind": (ann or {}).get("kind", ""),
            "mapped_to": "; ".join((ann or {}).get("targets", [])),
            "map_used": map_used or "",
        }
        for column, values in cells.items():
            if column not in seen_bridge_columns:
                seen_bridge_columns.add(column)
                bridge_columns.append(column)
            row[column] = "; ".join(values)
        rows.append(row)

    for entry in native:
        ds = entry["dataset"]
        for cand in (entry.get("types_all") or entry.get("types", [])):
            cells = _bridge_cells(
                cand.get("bridges_by_target"), dataset,
                entry["dataset"], cand["name"])
            _row("type", entry.get("value_source") or "type",
                 cand["name"], cand["name"],
                 cand["count"], cand.get("mapped"),
                 cand.get("map_used", ""), cells)
        for label in (entry.get("labels_all") or entry.get("labels", [])):
            # exploded per covered type — the label total is the panel's
            # business; the CSV is the per-type mapping table
            for covered in (label.get("covered_all")
                            or label.get("types", [])):
                cells = _bridge_cells(
                    covered.get("bridges_by_target"), dataset,
                    entry["dataset"], covered["name"])
                _row("label", label["column"], label["label"],
                     covered["name"], covered.get("count", ""),
                     covered.get("mapped"), covered.get("map_used", ""),
                     cells)
    for entry in mapped:
        ds = entry["dataset"]
        for cand in entry.get("candidates", []):
            _row("query alias", "auto type mapping", cand["name"], "",
                 cand.get("count", ""),
                 {"kind": cand.get("kind", ""),
                  "targets": cand.get("aggregates", []) or []},
                 "", {})

    if not rows:
        return ""
    fieldnames = base_fields + sorted(bridge_columns)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
