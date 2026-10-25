"""Build the local neuron index used by the UI and cache layer.

The pulled ``*_allneurons_neuron_df.csv`` file remains the authoritative
dataset metadata.  This module creates a typed, columnar copy for local
search and display.  All source metadata columns are retained, including
numeric fields.  Large serialized ROI annotations are excluded because they
make the local viewer unnecessarily large and slow; the pulled source table
remains the authoritative copy of those annotations.  Accidental index
columns and cache-state fields owned by the connection cache are also
excluded from the source projection.

Cache completion state is kept by :class:`FindNeuronConnection`; this module
only owns metadata-source discovery and the list of columns that make up the
search projection so the UI and backend use the same scope.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional


# These are not source metadata fields.  The viewer can retain and display
# serialized values, but these generated/bookkeeping fields are owned by the
# connection cache and are added back at the end of the materialized index.
SEARCH_EXCLUDED_COLUMNS = frozenset({
    "downstream_complete",
    "last_fetched",
    "connection_count",
    "unnamed: 0",
    "",
})

# ROI annotations are serialized lists/maps rather than compact metadata.
# The pull pipeline drops them from the local ``*_neuron_df.csv`` releases
# (``roiInfo``/``inputRois``/``outputRois``, all preserved long-form in
# ``*_roi_count_df.csv``), but legacy files may still carry them, and they
# grow into multi-hundred-kilobyte strings. Keep the source CSV untouched,
# but omit every ROI-named field from the index used by suggestions and the
# rendered viewer. Matching and display remain fast while ordinary numeric
# and string metadata stays available in its source order.
LARGE_SERIALIZED_COLUMN_MARKERS = ("roi",)

# FlyWire releases have historically used several spellings for their stable
# neuron identifier.  The cache has one canonical name (``bodyId``), while
# source metadata can use any of these aliases.
_BODY_ID_ALIASES = (
    "bodyid", "body_id", "rootid", "root_id", "flywireid", "flywire_id",
)

# These are cache fields rather than source metadata.  ``post`` is also a
# source column in most releases, so it stays in its source position when it
# exists there; the other three fields are appended as cache state.
CACHE_STATE_COLUMNS = (
    "downstream_complete", "last_fetched", "connection_count",
)
OPERATIONAL_COLUMNS = (
    "post", *CACHE_STATE_COLUMNS,
)

# Compact sidecar used by the viewer's prefix/substring search path.
SEARCH_CACHE_FILENAME = "neuron_index_search.parquet"

# App-owned directory that holds the neuron index and its search sidecar for
# every dataset.  These are persistent "system files": the bundled datasets'
# seeds are committed here, and the pull pipeline builds every other dataset's
# index into the same directory - so clearing ``cache/`` never removes the
# metadata behind auto-suggestions and the available-neurons viewer.
NEURON_INDEX_DIR_NAME = "neuron_indexes"
NEURON_INDEX_FILENAME = "neuron_index.parquet"

# Search priority after the three identity columns.  Keep this explicit rather
# than relying on lexical column order: the same order is consumed by the
# viewer, auto-suggestions, and the analysis resolver.  Source columns that do
# not occur in a particular dataset simply drop out of the resulting list.
TAXONOMY_PRIORITY_COLUMNS = ("class", "subclass", "superclass")

# A pulled dataset folder normally ends in ``_v1_2_3``.  Keep local-release
# identifiers such as ``flywire_FAFB_v783`` and ``banc_v888`` untouched: those
# names are public release IDs rather than NeuPrint's ``family:vX.Y`` form.
_VERSIONED_DATASET_FOLDER_RE = re.compile(
    r"^(?P<prefix>.+)_v(?P<version>\d+(?:_\d+)*)$",
    re.IGNORECASE,
)

# ``metadata_candidates`` has a schema-validated fallback for a future pull
# whose filename does not follow the current ``*_neuron_df`` convention.  Do
# not inspect obvious non-neuron artifacts as possible metadata tables.
_NON_METADATA_FILENAME_MARKERS = (
    "connection", "synapse", "roi", "skeleton", "mesh", "manifest",
    "index", "summary",
)


def _normalized_column_name(column: str) -> str:
    """Normalize a metadata name for case/spacing/punctuation comparisons."""
    return re.sub(r"[^a-z0-9]", "", str(column).casefold())


def is_large_serialized_metadata_column(column: str) -> bool:
    """Whether a source field is a serialized, oversized metadata payload."""
    normalized = _normalized_column_name(column)
    return any(marker in normalized for marker in LARGE_SERIALIZED_COLUMN_MARKERS)


def body_id_column(columns: Iterable[str]) -> Optional[str]:
    """Find the source column that identifies one neuron.

    ``bodyId`` is the canonical output name.  ``root_id``/``rootId`` and
    other FlyWire spellings are accepted so a new local release does not
    silently produce an index with blank identifiers.
    """
    names = [str(column) for column in columns]
    for name in names:
        if name == "bodyId":
            return name
    normalized = {_normalized_column_name(alias) for alias in _BODY_ID_ALIASES}
    for name in names:
        if _normalized_column_name(name) in normalized:
            return name
    return None


def is_priority_metadata_column(column: str) -> bool:
    """Whether a field is useful for type-oriented search priority.

    The viewer still displays every retained string field.  This predicate is
    deliberately narrower: it keeps suggestion expansion focused on canonical
    type/class taxonomy instead of values such as confidence scores, counts,
    coordinates, notes, or arbitrary annotations.  A field such as
    ``celltypePredictedNt`` remains useful, while similarly named measurement
    fields such as ``celltypePredictedNtConfidence`` do not.
    """
    normalized = _normalized_column_name(column)
    if any(token in normalized for token in (
        "confidence", "score", "count", "prediction",
    )):
        return False
    return (
        "type" in normalized
        or normalized == "class"
        or normalized == "superclass"
        or normalized.endswith("class")
    )


def priority_metadata_columns(columns: Iterable[str]) -> List[str]:
    """Order type/class metadata columns for display and matching.

    Cross-dataset names are promoted in the requested order, followed by
    other type fields, then the inner taxonomy fields (class, subclass,
    superclass). Ties retain source order so two releases with different
    metadata schemas remain stable.
    """
    names = [str(column) for column in columns]

    def rank(column: str) -> int:
        normalized = _normalized_column_name(column)
        explicit = {
            "flywiretype": 0,
            "hemibraintype": 1,
            "manctype": 2,
        }
        if normalized in explicit:
            return explicit[normalized]
        if "type" in normalized:
            return 3
        if normalized in TAXONOMY_PRIORITY_COLUMNS:
            return 4 + TAXONOMY_PRIORITY_COLUMNS.index(normalized)
        return 7

    selected = [
        column for column in names
        if column not in {"bodyId", "type", "instance"}
        and is_priority_metadata_column(column)
    ]
    return [
        column for _, column in sorted(
            enumerate(selected), key=lambda item: (rank(item[1]), item[0])
        )
    ]


def ordered_projection_columns(columns: Iterable[str]) -> List[str]:
    """Return the canonical cache/viewer order for a projected schema.

    Identity fields come first, followed by the type/class taxonomy used for
    suggestion expansion.  Remaining retained metadata stays visible after
    that group in its original source order.  Cache state fields are kept at
    the end; ``post`` is not moved because it is part of the original
    metadata order.
    """
    names = [str(column) for column in columns]
    identity = ["bodyId", "type", "instance"]
    cache_state = list(CACHE_STATE_COLUMNS)
    priority = priority_metadata_columns(names)
    return [
        *[column for column in identity if column in names],
        *[column for column in priority if column in names],
        *[
            column for column in names
            if column not in identity
            and column not in priority
            and column not in cache_state
        ],
        *[column for column in cache_state if column in names],
    ]


def viewer_search_columns(columns: Iterable[str]) -> List[str]:
    """Return the canonical columns indexed by the viewer search cache."""
    names = [str(column) for column in columns]
    result: List[str] = []
    for column in ("bodyId", "type", "instance"):
        if column in names and column not in result:
            result.append(column)
    for column in priority_metadata_columns(names):
        if column in names and column not in result:
            result.append(column)
    return result


def search_cache_path(index_path: Path) -> Path:
    """Return the compact searchable sidecar next to a neuron index."""
    index_path = Path(index_path)
    return index_path.with_name(SEARCH_CACHE_FILENAME)


def system_neuron_index_path(dataset: str, index_dir: Path) -> Path:
    """Return the app-owned neuron-index path for *dataset*.

    The index lives outside ``cache/`` so an intentional cache cleanup never
    removes the metadata behind auto-suggestions and the viewer.
    """
    return Path(index_dir) / dataset_folder(dataset) / NEURON_INDEX_FILENAME


def migrate_legacy_neuron_index(dataset, *, cache_dir, index_dir) -> bool:
    """Move a legacy ``cache/`` index into the app-owned directory once.

    Older releases stored ``neuron_index.parquet`` and its search sidecar
    inside ``cache/<dataset>/``.  When the app-owned location is still empty,
    move both files over so existing pull state (completion flags) survives
    the layout change without a re-pull.  Returns True when a migration
    happened; move failures surface as OSError to the caller.
    """
    target = system_neuron_index_path(dataset, index_dir)
    if target.is_file():
        return False
    legacy = Path(cache_dir) / dataset_folder(dataset) / NEURON_INDEX_FILENAME
    if not legacy.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(legacy, target)
    sidecar = search_cache_path(legacy)
    if sidecar.is_file():
        os.replace(sidecar, search_cache_path(target))
    return True


def migrate_legacy_neuron_indexes(cache_dir: Path, index_dir: Path) -> List[str]:
    """Migrate every legacy cache/ index into the app-owned directory.

    Returns the dataset folders that were moved.  Used once at UI startup so
    existing installations upgrade in place without a re-pull; each failed
    move is skipped and left for the pull pipeline to rebuild.
    """
    migrated: List[str] = []
    cache_root = Path(cache_dir)
    if not cache_root.is_dir():
        return migrated
    for folder in sorted(
        path.name for path in cache_root.iterdir() if path.is_dir()
    ):
        if not (cache_root / folder / NEURON_INDEX_FILENAME).is_file():
            continue
        try:
            if migrate_legacy_neuron_index(
                folder, cache_dir=cache_root, index_dir=index_dir
            ):
                migrated.append(folder)
        except OSError:
            continue
    return migrated


def is_search_cache_compatible(search_frame, source_columns: Iterable[str]) -> bool:
    """Check that a search sidecar covers the current canonical projection.

    Search sidecars are deliberately version-light parquet files.  A cache can
    therefore outlive a metadata projection rebuild, especially when a new
    FlyWire release adds a ``*Type`` or taxonomy field.  Checking only the
    sidecar schema is not enough in that case: the file may be readable while
    silently omitting a newly promoted search column.  Keep this validation in
    the builder module so the UI and analysis resolver apply the same rule.
    """
    required = {
        "__neuron_rows", "search_column", "search_priority", "search_value",
        "search_value_folded",
    }
    if search_frame is None or not required.issubset(set(search_frame.columns)):
        return False

    expected = viewer_search_columns(source_columns)
    try:
        priority_rows = (
            search_frame
            .select(["search_column", "search_priority"])
            .unique(subset=["search_column"], maintain_order=True)
            .sort("search_priority")
            .to_dicts()
        )
    except Exception:
        return False

    expected_priority = {column: position for position, column in enumerate(expected)}
    seen = set()
    for row in priority_rows:
        column = str(row.get("search_column") or "")
        if column in seen or column not in expected_priority:
            return False
        seen.add(column)
        try:
            priority = int(row.get("search_priority"))
        except (TypeError, ValueError):
            return False
        if priority != expected_priority[column]:
            return False

    return (
        [str(row.get("search_column") or "") for row in priority_rows]
        == expected
        and [int(row.get("search_priority")) for row in priority_rows]
        == list(range(len(expected)))
    )


def _search_display_expression(frame, column: str):
    """Return the string representation used by the shared search matcher."""
    import polars as pl

    expression = pl.col(column).cast(pl.Utf8, strict=False).fill_null("")
    if column == "bodyId":
        expression = expression.str.strip_chars().str.replace(r"\.0+$", "")
    return expression


def build_search_cache_frame(frame, columns: Optional[Iterable[str]] = None):
    """Build the ordered, compact search sidecar from a Polars frame.

    The sidecar has one row per non-empty searchable value, with the source
    row ordinals stored as a compact list. It also keeps one empty marker row
    for a projected column whose values are all blank. Those tiny markers
    make the sidecar schema self-describing, so an older sidecar cannot hide a
    newly added type/taxonomy field. Its input order is the query order for
    the viewer: canonical column priority, then strict value order. A query
    can therefore filter a small distinct-value table and explode only the
    matching row lists; it never sorts the full metadata frame.

    ``__neuron_rows`` contains the stable source-row ordinals used to join the
    sidecar back to the complete metadata table. It avoids copying every
    retained metadata column into this cache and avoids repeating a value once
    per neuron.
    """
    import polars as pl

    if columns is None:
        columns = viewer_search_columns(frame.columns)
    ordered = [column for column in columns if column in frame.columns]
    if not ordered:
        return pl.DataFrame(
            schema={
                "__neuron_rows": pl.List(pl.UInt32),
                "search_column": pl.Utf8,
                "search_priority": pl.UInt16,
                "search_value": pl.Utf8,
                "search_value_folded": pl.Utf8,
            }
        )

    source = frame.with_row_index("__neuron_row")
    parts = []
    for priority, column in enumerate(ordered):
        display = _search_display_expression(source, column)
        parts.append(
            source.select(
                pl.col("__neuron_row").cast(pl.UInt32, strict=False),
                pl.lit(column).alias("search_column"),
                pl.lit(priority).cast(pl.UInt16).alias("search_priority"),
                display.alias("search_value"),
            ).filter(pl.col("search_value").str.strip_chars() != "")
        )
    result = pl.concat(parts, how="vertical_relaxed").with_columns(
        pl.col("search_value").str.to_lowercase().alias("search_value_folded")
    )
    result = (
        result.group_by(
            [
                "search_column", "search_priority", "search_value",
                "search_value_folded",
            ],
            maintain_order=True,
        )
        .agg(
            pl.col("__neuron_row")
            .cast(pl.UInt32, strict=False)
            .unique(maintain_order=True)
            .alias("__neuron_rows")
        )
        .sort(["search_priority", "search_value"])
    )
    present = set(result.get_column("search_column").to_list())
    missing = [
        (priority, column)
        for priority, column in enumerate(ordered)
        if column not in present
    ]
    if missing:
        markers = pl.DataFrame({
            "search_column": [column for _, column in missing],
            "search_priority": pl.Series(
                "search_priority",
                [priority for priority, _ in missing],
                dtype=pl.UInt16,
            ),
            "search_value": ["" for _ in missing],
            "search_value_folded": ["" for _ in missing],
            "__neuron_rows": pl.Series(
                "__neuron_rows",
                [[] for _ in missing],
                dtype=pl.List(pl.UInt32),
            ),
        })
        result = pl.concat([result, markers], how="vertical_relaxed")
    return result.sort(["search_priority", "search_value"])


try:
    from .flywire_ids import dataset_folder as _canonical_dataset_folder
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from flywire_ids import dataset_folder as _canonical_dataset_folder


def dataset_folder(dataset: str) -> str:
    """Return the repository-safe folder name for a dataset identifier.

    Delegates to the canonicalizer so legacy aliases (``flywire_BANC_v888``)
    resolve to the same folder as seeds/caches/converter tables.
    """
    return _canonical_dataset_folder(dataset)


def metadata_candidates(dataset: str, datasets_dir: Path) -> List[Path]:
    """Return local neuron metadata files in preferred read order.

    Current pulls use deterministic ``*_allneurons_neuron_df``/``*_neuron_df``
    names and those always win.  The final schema-validated fallback keeps
    future releases discoverable when a provider changes only the filename:
    a candidate must be a direct CSV/Parquet file with a recognized body-ID
    column and must not look like a connection, ROI, or other derived table.
    """
    folder = Path(datasets_dir) / dataset_folder(dataset)
    if not folder.is_dir():
        return []

    safe = folder.name
    exact = [
        folder / f"{safe}_allneurons_neuron_df.csv",
        folder / f"{safe}_neuron_df.csv",
        folder / f"{safe}_allneurons_neuron_df.parquet",
        folder / f"{safe}_neuron_df.parquet",
    ]
    discovered = sorted(
        [
            path
            for pattern in (
                "*_allneurons_neuron_df.csv",
                "*_neuron_df.csv",
                "*_allneurons_neuron_df.parquet",
                "*_neuron_df.parquet",
            )
            for path in folder.glob(pattern)
        ],
        key=lambda path: path.name,
    )
    result: List[Path] = []
    for path in (*exact, *discovered):
        if not path.is_file() or path in result:
            continue
        try:
            has_body_id = body_id_column(_metadata_schema(path).names()) is not None
        except Exception:
            has_body_id = False
        if has_body_id:
            result.append(path)

    # Be tolerant of a future provider-specific filename while avoiding a
    # full data read.  ``_metadata_schema`` uses Polars' lazy reader, so this
    # only inspects headers/Parquet schemas during dataset discovery.
    fallback = []
    for path in sorted(folder.iterdir(), key=lambda candidate: candidate.name):
        if (
            not path.is_file()
            or path in result
            or path.suffix.lower() not in {".csv", ".parquet"}
        ):
            continue
        normalized_name = _normalized_column_name(path.stem)
        if any(
            marker in normalized_name
            for marker in _NON_METADATA_FILENAME_MARKERS
        ):
            continue
        try:
            if body_id_column(_metadata_schema(path).names()) is not None:
                fallback.append(path)
        except Exception:
            # A corrupt/unreadable artifact should not make a valid sibling
            # table disappear from the candidate list.
            continue
    result.extend(fallback)
    return result


def metadata_path(dataset: str, datasets_dir: Path) -> Optional[Path]:
    """Return the first usable local neuron metadata file, if any."""
    candidates = metadata_candidates(dataset, datasets_dir)
    return candidates[0] if candidates else None


def dataset_identifier_from_folder(folder: str) -> str:
    """Convert a local dataset folder into the identifier used by the app.

    NeuPrint folders encode dots as underscores (for example
    ``male-cns_v1_0``), while FlyWire/BANC release folders are already public
    identifiers and must remain unchanged.  Unknown/unversioned folder names
    are returned verbatim so a new provider can still be prepared without a
    new hard-coded mapping table.
    """
    name = Path(str(folder or "")).name.strip()
    if not name:
        return ""
    normalized = name.casefold()
    if normalized.startswith(("flywire_", "banc_")):
        return name
    match = _VERSIONED_DATASET_FOLDER_RE.match(name)
    if match is None:
        return name
    version = match.group("version").replace("_", ".")
    return f'{match.group("prefix")}:v{version}'


def discover_metadata_datasets(datasets_dir: Path) -> List[str]:
    """Discover every local dataset that has a usable neuron table.

    This is intentionally filesystem/schema based rather than tied to the
    UI's current dataset list.  It lets the preparation script handle newly
    pulled releases and datasets that are no longer available from a remote
    service but whose metadata remains on disk.
    """
    root = Path(datasets_dir)
    if not root.is_dir():
        return []

    datasets: List[str] = []
    seen = set()
    for folder in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        if folder.name.startswith("."):
            continue
        if not metadata_candidates(folder.name, root):
            continue
        dataset = dataset_identifier_from_folder(folder.name)
        if dataset and dataset not in seen:
            datasets.append(dataset)
            seen.add(dataset)
    return datasets


def metadata_columns(path: Path) -> List[str]:
    """Return canonical columns retained by the local projection."""
    source_columns = _metadata_source_columns(path)
    source_body_id = body_id_column(source_columns)
    return [
        "bodyId" if column == source_body_id else column
        for column in source_columns
    ]


def searchable_columns(columns: Iterable[str]) -> List[str]:
    """Return source columns retained in the local metadata projection."""
    result: List[str] = []
    seen = set()
    for column in columns:
        name = str(column)
        if (
            name.casefold() in SEARCH_EXCLUDED_COLUMNS
            or is_large_serialized_metadata_column(name)
            or name in seen
        ):
            continue
        result.append(name)
        seen.add(name)
    return result


def _metadata_schema(path: Path):
    """Read only the source schema needed to choose projection columns."""
    import polars as pl

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path).collect_schema()
    return pl.scan_csv(
        path,
        infer_schema_length=1000,
        ignore_errors=True,
        try_parse_dates=False,
    ).collect_schema()


def _metadata_source_columns(path: Path) -> List[str]:
    """Return all source metadata columns in their original order.

    The source fields omitted are accidental CSV index columns, cache state
    columns, and large serialized ROI payloads. Keeping the source order here
    means the later projection can promote only the useful identity/taxonomy
    fields without reordering the rest of the dataset's metadata for
    readability.
    """
    schema = _metadata_schema(path)
    names = list(schema.names())
    source_body_id = body_id_column(names)
    if source_body_id is None:
        return []

    return [
        name for name in names
        if (
            str(name).casefold() not in SEARCH_EXCLUDED_COLUMNS
            and not is_large_serialized_metadata_column(name)
        )
    ]


def read_metadata_projection(path: Path):
    """Read all retained source metadata as a Polars DataFrame.

    The caller adds cache bookkeeping fields and writes the result to the
    cache index. Only the identity/taxonomy group is reordered; every other
    retained source column and value is preserved. ROI payloads have already
    been excluded by :func:`_metadata_source_columns`.
    """
    import polars as pl

    path = Path(path)
    source_columns = _metadata_source_columns(path)
    source_body_id = body_id_column(source_columns)
    if source_body_id is None:
        raise ValueError(f"Neuron metadata has no body ID column: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pl.read_parquet(path, columns=source_columns)
    else:
        schema_overrides = {source_body_id: pl.Utf8}
        frame = pl.read_csv(
            path,
            columns=source_columns,
            # Body IDs can exceed signed 64-bit and JavaScript's safe integer
            # range.  Read them as text before any projection/cast so Polars
            # never turns an out-of-range value into null.
            schema_overrides=schema_overrides,
            infer_schema_length=1000,
            ignore_errors=True,
            try_parse_dates=False,
        )

    if source_body_id != "bodyId":
        frame = frame.rename({source_body_id: "bodyId"})

    # Large FlyWire IDs must remain exact.  The UI also uses strings so the
    # browser never rounds a value beyond JavaScript's safe integer range.
    frame = frame.with_columns(
        pl.col("bodyId")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .alias("bodyId")
    )
    for column in ("type", "instance"):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )
        else:
            frame = frame.with_columns(pl.lit("").alias(column))
    return frame.select(ordered_projection_columns(frame.columns))
