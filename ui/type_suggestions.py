"""Dataset type-name pools for the neuron-input auto-suggest.

Suggestion entries are ``(value, hint)`` pairs so the dropdown can render the
matched name with a gray hint of the searched column (for bodyId matches the
hint is the corresponding instance). Pools are read from LOCAL dataset files
only — ``neuron_indexes/<dataset>/neuron_index.parquet`` first, supplemented
by the ``datasets/<dataset>`` neuron tables when the index is incomplete —
and cached per (dataset, file mtime); the server is never contacted, so a
dataset that has not been pulled yet simply yields no suggestions (except the
bundled datasets whose indexes ship with the repository).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder, folder_to_dataset
from .search_logic import (
    filter_candidate_entries as _shared_filter_candidate_entries,
    match_search_pools,
)

try:
    from src.neuron_index_builder import (
        metadata_candidates,
        metadata_columns,
        priority_metadata_columns,
        read_metadata_projection,
        viewer_search_columns,
    )
except ImportError:
    from neuron_index_builder import (
        metadata_candidates,
        metadata_columns,
        priority_metadata_columns,
        read_metadata_projection,
        viewer_search_columns,
    )

# (value, hint) — the hint is the searched column name, except for bodyId
# matches where it is the corresponding instance.
Entry = Tuple[str, str]

_DATASETS_DIR = PROJECT_ROOT / "datasets"
_INDEX_DIR = PROJECT_ROOT / "neuron_indexes"

_POOL_CACHE: Dict[tuple, Dict[str, List[Entry]]] = {}

# Columns whose distinct values become suggestion pools (hint = column name).
_TYPE_COLUMNS = ("type", "instance")
_EMPTY_POOLS: Dict[str, List[Entry]] = {}


def _valid_text(value) -> Optional[str]:
    """Return a useful display/search string, or ``None`` for a null value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {
        "nan", "none", "null", "<na>", "<null>",
    }:
        return None
    return text


def _body_id_entries(body_ids, instances) -> List[Entry]:
    """Build bodyId suggestions with the corresponding instance as hint."""
    body_entries = []
    for body_id, instance in zip(body_ids, instances):
        body_id_text = _valid_text(body_id)
        if body_id_text is None:
            continue
        # Parquet files inferred from spreadsheet exports can represent an
        # integer identifier as ``123.0``. Normalize only that safe suffix so
        # selecting the suggestion verifies against the canonical bodyId.
        if re.fullmatch(r"\d+\.0+", body_id_text):
            body_id_text = body_id_text.split(".", 1)[0]
        hint = _valid_text(instance) or "bodyId"
        body_entries.append((body_id_text, hint))
    return sorted(set(body_entries))


def _clean(values) -> List[str]:
    """Deduplicated, sorted, non-empty string values from a column."""
    seen = set()
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "<na>", "<null>"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def _clean_split(values) -> List[str]:
    """Like :func:`_clean`, but also splits comma-joined cells.

    Datasets such as FlyWire FAFB store several alternative type names in
    one ``additional_type(s)`` cell (``'vDeltaB, vDeltaC, ...'``).  A
    combined cell is never a real neuron name, so every part becomes its
    own suggestion entry instead of offering the raw joined string.
    """
    expanded: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v)
        if "," in s:
            expanded.extend(part.strip() for part in s.split(","))
        else:
            expanded.append(s)
    return _clean(expanded)


def _index_pools(folder: str) -> Optional[Dict[str, List[Entry]]]:
    """Pools from neuron_indexes/<folder>/neuron_index.parquet (None when absent)."""
    index = _INDEX_DIR / folder / "neuron_index.parquet"
    if not index.exists():
        return None
    try:
        import polars as pl

        schema_columns = pl.scan_parquet(index).collect_schema().names()
        selected_columns = [
            column for column in (
                "bodyId", "type", "instance",
                *priority_metadata_columns(schema_columns),
            )
            if column in schema_columns
        ]
        frame = pl.read_parquet(
            index,
            columns=list(dict.fromkeys(selected_columns)),
        )
        cols = set(frame.columns)
        pools: Dict[str, List[Entry]] = {}
        for col in _TYPE_COLUMNS:
            if col not in cols:
                continue
            values = frame[col].drop_nulls().to_list()
            pools[col] = [(v, col) for v in _clean_split(values)]
        # Auto-suggestion expands beyond the canonical type column only into
        # type/class taxonomy fields.  The viewer still displays every
        # retained string field and searches it when explicitly requested.
        for col in priority_metadata_columns(frame.columns):
            if col not in cols:
                continue
            values = frame[col].cast(pl.Utf8, strict=False).to_list()
            pools[col] = [(v, col) for v in _clean_split(values)]
        # bodyId pool: string-form ids, hint = the corresponding instance.
        if "bodyId" in cols:
            bid_col = frame["bodyId"].cast(pl.Utf8, strict=False)
            inst_col = frame["instance"].cast(pl.Utf8, strict=False) \
                if "instance" in cols else None
            pools["bodyId"] = _body_id_entries(
                bid_col.to_list(),
                inst_col.to_list() if inst_col is not None
                else [None] * len(bid_col),
            )
        return pools
    except Exception:
        return None


def _index_has_search_projection(folder: str) -> bool:
    """Whether the index already contains the pulled metadata fields."""
    index = _INDEX_DIR / folder / "neuron_index.parquet"
    if not index.exists():
        return False
    try:
        import polars as pl

        columns = set(pl.scan_parquet(index).collect_schema().names())
        tables = _metadata_tables(folder)
        if tables:
            # The pulled metadata is authoritative.  If it is newer than the
            # index, use it until the normal pull pipeline rebuilds the index;
            # otherwise a newly pulled type could remain invisible to input
            # suggestions.
            if index.stat().st_mtime_ns < tables[0].stat().st_mtime_ns:
                return False
            expected = viewer_search_columns(metadata_columns(tables[0]))
            # An identity-only index may still cover every column name while
            # containing only a partial set of rows.  Keep the table fallback
            # for that legacy shape; generated rich indexes have at least one
            # promoted taxonomy field.
            return (
                len(expected) > 3
                and set(expected).issubset(columns)
            )

        # With no local source table to compare, only a canonical search
        # projection can be trusted.  Arbitrary operational columns do not
        # make a legacy connection index rich.
        return len(viewer_search_columns(columns)) > 3
    except Exception:
        return False


def _metadata_tables(folder: str) -> List[Path]:
    """Find pulled neuron metadata using the shared builder ordering."""
    return metadata_candidates(folder_to_dataset(folder), _DATASETS_DIR)


def _table_pools(folder: str) -> Optional[Dict[str, List[Entry]]]:
    """Pools from the datasets/<folder> neuron tables (None when absent)."""
    ds_dir = _DATASETS_DIR / folder
    if not ds_dir.exists():
        return None
    candidates = _metadata_tables(folder)
    if not candidates:
        return None
    for table in candidates:
        try:
            import polars as pl
            frame = read_metadata_projection(table)
            cols = set(frame.columns)
            if not {"bodyId", "type"}.issubset(cols):
                continue
            pools: Dict[str, List[Entry]] = {}
            for col in _TYPE_COLUMNS:
                if col in cols:
                    values = frame[col].cast(pl.Utf8, strict=False).to_list()
                    pools[col] = [(v, col) for v in _clean_split(values)]
            for col in priority_metadata_columns(frame.columns):
                if col in pools or col in ("bodyId", "type", "instance"):
                    continue
                values = frame[col].cast(pl.Utf8, strict=False).to_list()
                pools[col] = [(v, col) for v in _clean_split(values)]
            if "bodyId" in cols:
                body_ids = frame["bodyId"].cast(pl.Utf8, strict=False).to_list()
                inst_series = (
                    frame["instance"].cast(pl.Utf8, strict=False).to_list()
                    if "instance" in cols else [None] * len(body_ids)
                )
                pools["bodyId"] = _body_id_entries(
                    body_ids,
                    inst_series,
                )
            return pools
        except Exception:
            continue
    return None


def _merge_pools(
    primary: Dict[str, List[Entry]],
    fallback: Dict[str, List[Entry]],
) -> Dict[str, List[Entry]]:
    """Supplement a cache pool with values from the local neuron table.

    The cache remains first: duplicate values keep its ordering and hint.
    Missing values from the table are appended so a partially populated
    connection index cannot hide valid names. BodyId hints are upgraded when
    the cache only has the generic ``bodyId`` label but the table knows the
    corresponding instance.
    """
    merged: Dict[str, List[Entry]] = {}
    for column in dict.fromkeys((*primary.keys(), *fallback.keys())):
        primary_entries = list(primary.get(column, []))
        fallback_entries = list(fallback.get(column, []))
        fallback_by_value = {value: hint for value, hint in fallback_entries}
        entries: List[Entry] = []
        seen = set()
        for value, hint in primary_entries:
            if (column == "bodyId" and hint == "bodyId"
                    and fallback_by_value.get(value)
                    and fallback_by_value[value] != "bodyId"):
                hint = fallback_by_value[value]
            if value not in seen:
                entries.append((value, hint))
                seen.add(value)
        for value, hint in fallback_entries:
            if value not in seen:
                entries.append((value, hint))
                seen.add(value)
        if entries:
            merged[column] = entries
    return merged


def _folder_pools(folder: str) -> Dict[str, List[Entry]]:
    # Cache key covers every local source file's mtime so a rebuilt index
    # or an updated dataset table invalidates the pools.
    sources = []
    index = _INDEX_DIR / folder / "neuron_index.parquet"
    if index.exists():
        sources.append((str(index), index.stat().st_mtime_ns))
    tables = _metadata_tables(folder)
    for p in tables:
        sources.append((str(p), p.stat().st_mtime_ns))
    key = ("pools", folder, tuple(sorted(sources)))
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    index_pools = _index_pools(folder)
    if index_pools is not None and _index_has_search_projection(folder):
        # The generated rich index already contains the pulled metadata; do
        # not scan a 500+ MiB CSV again just to rebuild identical value pools.
        pools = index_pools
    else:
        table_pools = _table_pools(folder)
        # A legacy connection cache may be complete for bodyIds but only
        # partial for names. Supplement it instead of letting it mask the
        # full local table; otherwise valid names such as ``aMe12`` disappear.
        if index_pools is not None and table_pools is not None:
            source_is_newer = bool(
                index.exists()
                and tables
                and index.stat().st_mtime_ns < tables[0].stat().st_mtime_ns
                and len(viewer_search_columns(metadata_columns(tables[0]))) > 3
            )
            # Once the pulled table changes, it is authoritative for both
            # additions and removals.  Merging an old index would preserve
            # deleted names and make suggestions disagree with the source.
            pools = table_pools if source_is_newer else _merge_pools(
                index_pools, table_pools
            )
        elif index_pools is not None:
            pools = index_pools
        else:
            pools = table_pools
    if pools is None:
        pools = _EMPTY_POOLS
    _POOL_CACHE[key] = pools
    return pools


def get_dataset_pools(dataset: str) -> Dict[str, List[Entry]]:
    """``{column: [(value, hint), ...]}`` for one dataset; {} when nothing
    is available locally (dataset not pulled / no local tables)."""
    if not dataset:
        return _EMPTY_POOLS
    return _folder_pools(dataset_to_folder(str(dataset)))


def suggestion_pool(datasets: Sequence[str]) -> Dict[str, List[Entry]]:
    """Union of per-dataset pools (cross-dataset tab): same column keys,
    values deduplicated across datasets."""
    merged: Dict[str, List[Entry]] = {}
    for ds in datasets:
        for col, entries in get_dataset_pools(ds).items():
            seen = {v for v, _ in merged.get(col, [])}
            merged[col] = merged.get(col, []) + [
                e for e in entries if e[0] not in seen
            ]
    return merged


def dataset_suggestions(
    text: str,
    dataset: Optional[str],
    search_columns: str = "auto",
    *,
    limit: Optional[int] = None,
) -> List[Entry]:
    """Resolve an input's suggestions from one dataset's shared pool."""
    return match_suggestions(
        text,
        get_dataset_pools(dataset or ""),
        search_columns,
        limit=limit,
    )


def datasets_suggestions(
    text: str,
    datasets: Sequence[str],
    search_columns: str = "auto",
    *,
    limit: Optional[int] = None,
) -> List[Entry]:
    """Resolve suggestions across selected datasets with shared semantics."""
    return match_suggestions(
        text,
        suggestion_pool(list(datasets or [])),
        search_columns,
        limit=limit,
    )


def dataset_aware_suggestions(
    text: str,
    datasets: Sequence[str],
    search_columns: str = "auto",
    *,
    limit: Optional[int] = None,
) -> List[Entry]:
    """Dataset-aware suggestions for the cross-dataset tab.

    Same staged matching semantics as :func:`datasets_suggestions`, but the
    gray hint carries both the matched column and the dataset it came from
    (``type · male-cns:v1.0``), restricted to the given (selected) datasets
    — the suggestion-list counterpart of the history list's dataset tags. A
    value matched in several datasets lists every matching dataset, keeping
    per-dataset categories when they differ
    (``type · male-cns:v1.0 · instance · hemibrain:v1.2.1``).
    """
    merged: Dict[str, List[Tuple[str, str]]] = {}
    for dataset in datasets:
        if not dataset:
            continue
        for value, hint in match_suggestions(
            text, get_dataset_pools(dataset), search_columns, limit=None
        ):
            merged.setdefault(value, []).append(
                (str(hint), str(dataset))
            )

    entries: List[Entry] = []
    for value, pairs in merged.items():
        # Group identical hints into one segment ("type · ds1, ds2") while
        # preserving the dataset order of first appearance.
        grouped: Dict[str, List[str]] = {}
        for hint, dataset in pairs:
            grouped.setdefault(hint, []).append(dataset)
        hint = " · ".join(
            f"{hint} · {', '.join(ds_list)}"
            for hint, ds_list in grouped.items()
        )
        entries.append((value, hint))
    if limit is not None:
        entries = entries[:limit]
    return entries


def _membership_pairs(
    values: Sequence[str],
    datasets: Sequence[str],
) -> Tuple[List[str], Dict[str, List[Tuple[str, str]]]]:
    """Exact local-pool membership scan shared by the history hint strings
    and the per-value resolving-datasets map.

    Returns the deduplicated ``wanted`` values and, for every value found in
    at least one pool, its ``(column_hint, dataset)`` pairs in the caller's
    dataset order. Membership uses the same staged priority as suggestions:
    a type match hides same-value matches in the other columns (suggestions
    stop at the first non-empty stage), and a numeric value resolves through
    the bodyId pool to its cached instance.
    """
    wanted: List[str] = []
    seen: set = set()
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            wanted.append(text)

    # value -> [(hint, dataset)] in the caller's dataset order.
    pairs: Dict[str, List[Tuple[str, str]]] = {}
    for dataset in datasets or []:
        if not dataset:
            continue
        pools = get_dataset_pools(str(dataset))
        if not pools:
            continue
        # Per-column membership sets, built once per dataset for the whole
        # batch of history rows (the pools themselves are already cached).
        columns = [
            column for column in viewer_search_columns(pools.keys())
            if column != "bodyId"
        ]
        sets = {
            column: {v for v, _ in pools.get(column, [])}
            for column in columns
        }
        body_ids: Optional[Dict[str, str]] = None
        for value in wanted:
            if re.fullmatch(r"\d+(?:\.0+)?", value):
                if body_ids is None:
                    body_ids = {
                        v: hint for v, hint in pools.get("bodyId", [])
                    }
                key = (value.split(".", 1)[0]
                       if re.fullmatch(r"\d+\.0+", value) else value)
                hint = body_ids.get(key)
                if hint:
                    pairs.setdefault(value, []).append(
                        (str(hint), str(dataset)))
                continue
            if value in sets.get("type", set()):
                pairs.setdefault(value, []).append(("type", str(dataset)))
                continue
            for column in columns:
                if column != "type" and value in sets[column]:
                    pairs.setdefault(value, []).append(
                        (column, str(dataset)))
    return wanted, pairs


def dataset_aware_history_hints(
    values: Sequence[str],
    datasets: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Suggestion-style gray hints for query-history rows.

    The cross-dataset counterpart of :func:`dataset_aware_suggestions` for
    COMPLETE stored queries: each value gets the same gray hint its
    auto-suggestion row shows — the matched column and the dataset it came
    from (``type · male-cns:v1.0``), with identical columns grouped across
    datasets (``type · ds1, ds2``) and differing columns listed per dataset
    (``type · ds1 · instance · ds2``). Membership is exact (see
    :func:`_membership_pairs`). Values found in no local pool map to
    ``None`` so their rows stay unannotated — history also keeps regex
    patterns that no pool contains.
    """
    wanted, pairs = _membership_pairs(values, datasets)
    hints: Dict[str, Optional[str]] = {value: None for value in wanted}
    for value, entries in pairs.items():
        grouped: Dict[str, List[str]] = {}
        for hint, dataset in entries:
            grouped.setdefault(hint, []).append(dataset)
        hints[value] = " · ".join(
            f"{hint} · {', '.join(ds_list)}"
            for hint, ds_list in grouped.items()
        )
    return hints


def datasets_resolving(
    values: Sequence[str],
    datasets: Sequence[str],
) -> Dict[str, List[str]]:
    """Datasets whose local pools contain each value, in caller order.

    The badge counterpart of :func:`dataset_aware_history_hints`: exact
    membership (see :func:`_membership_pairs`) per value, deduplicated while
    keeping the caller's dataset order. Values found in no local pool —
    regex patterns, or names living only in datasets without local tables —
    are absent from the mapping, so callers can distinguish "resolves here"
    from "no local evidence".
    """
    _wanted, pairs = _membership_pairs(values, datasets)
    resolved: Dict[str, List[str]] = {}
    for value, entries in pairs.items():
        seen_datasets: set = set()
        ordered: List[str] = []
        for _hint, dataset in entries:
            if dataset not in seen_datasets:
                seen_datasets.add(dataset)
                ordered.append(dataset)
        resolved[value] = ordered
    return resolved


def filter_candidate_entries(
    text: str,
    candidates: Sequence[Entry],
) -> List[Entry]:
    """Narrow an existing candidate list with a case-sensitive prefix.

    This is intentionally smaller than :func:`match_suggestions`: it is the
    fast path used while a user appends characters to the same query. The
    caller must fall back to the full matcher when the query is not a strict
    continuation or this list produces no matches.
    """
    return _shared_filter_candidate_entries(text, candidates)


def match_suggestions(
    text: str,
    pools: Dict[str, List[Entry]],
    search_columns: str = "auto",
    limit: Optional[int] = 50,
    *,
    all_prefix_matches: bool = False,
) -> List[Entry]:
    """Match ``text`` against the pools with the backend's column priority.

    Matching follows :func:`ui.search_logic.search_plan` so the viewer and
    source/target input use exactly the same staged behavior:

    1. strict, case-sensitive prefixes;
    2. only when every applicable prefix stage is empty, a
       case-insensitive substring fallback.

    Numeric input is restricted to real bodyId values. String input tries the
    type prefix first and expands to bodyId/instance/taxonomy only when no
    type prefix exists. Set ``all_prefix_matches`` for a full-viewer-style
    result that includes every strict-prefix column at once, ordered by the
    same canonical priority. Explicit scopes search that column only.

    ``limit=None`` returns the complete candidate pool. The UI providers use
    this mode because the input component applies its own display limit after
    each continuation is narrowed; limiting here would make names outside
    the first page impossible to reach by typing more characters.
    """
    return match_search_pools(
        text,
        pools,
        search_columns,
        limit,
        all_prefix_matches=all_prefix_matches,
    )


def entry_hint(value: str, pools: Dict[str, List[Entry]]) -> Optional[str]:
    """Column of a value in the pools (type/instance/bodyId priority), for
    labeling history entries; None when the value is not in any pool."""
    for col in ("type", "instance", "bodyId"):
        if any(v == str(value) for v, _ in pools.get(col, [])):
            return col
    return None
