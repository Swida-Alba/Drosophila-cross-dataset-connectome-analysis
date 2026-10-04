"""Union type-resolution coverage for cross-dataset comparisons.

Alignment only ever sees the types a dataset's own pathfinding recruited;
a type that another dataset surfaced is silently zero after the outer
join.  This module closes that gap: it takes the UNION of types that
appeared in any dataset's aligned results for a query, resolves each type
into EVERY dataset through the shared type mapper, and classifies the
absent ones so the exports/report can say WHY a dataset shows no weight:

- ``present``           the type appears in this dataset's searched graph
- ``below_threshold``   neurons exist and edges to in-graph partners
                        exist, but all below the applied threshold
- ``not_recruited``     a >= threshold edge to an in-graph partner exists
                        but pathfinding/budget did not include the type
- ``no_edges``          neurons exist, but no edges to in-graph partners
- ``not_in_dataset``    the resolved names have no neurons in the dataset
- ``unmapped`` / ``conflict`` / ``evidence_only`` / ``claimed`` /
  ``mapper_unavailable``  the resolver's own verdicts, surfaced verbatim
- ``resolved_absent``   resolved, but the diagnosis inputs (neuron table /
                        connection cache / searched-graph list) were
                        unavailable — reported rather than silently 0

Every read here is local and read-only (dataset neuron table, cached
connections parquet, the run's own ``neurons_included.csv``); a missing
input degrades the status, it never raises — the same policy as the
mapper itself ("a broken mapper must never block a comparison").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .type_resolver import (
    STATUS_CLAIMED,
    STATUS_CONFLICT,
    STATUS_EVIDENCE_ONLY,
    STATUS_MAPPER_UNAVAILABLE,
    STATUS_UNMAPPED,
    expansion_targets,
    resolve_valid_targets,
)

STATUS_PRESENT = 'present'
STATUS_BELOW_THRESHOLD = 'below_threshold'
STATUS_NOT_RECRUITED = 'not_recruited'
STATUS_NO_EDGES = 'no_edges'
STATUS_NOT_IN_DATASET = 'not_in_dataset'
STATUS_RESOLVED_ABSENT = 'resolved_absent'

# Short human-facing labels for report text (underscores read poorly).
STATUS_LABELS = {
    STATUS_PRESENT: 'present',
    STATUS_BELOW_THRESHOLD: 'below threshold',
    STATUS_NOT_RECRUITED: 'not recruited',
    STATUS_NO_EDGES: 'no edges',
    STATUS_NOT_IN_DATASET: 'not in dataset',
    STATUS_RESOLVED_ABSENT: 'resolved, absent',
    STATUS_UNMAPPED: 'unmapped',
    STATUS_CONFLICT: 'conflict',
    STATUS_EVIDENCE_ONLY: 'evidence only',
    STATUS_CLAIMED: 'claimed',
    STATUS_MAPPER_UNAVAILABLE: 'mapper unavailable',
}

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

_NEURON_TABLE_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_CONNECTION_CACHE: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
_IN_GRAPH_CACHE: Dict[str, Optional[Set[str]]] = {}


@dataclass(frozen=True)
class TypeCoverageEntry:
    """One (type, dataset) resolution record for one query."""

    type: str
    dataset: str
    present: bool
    resolved_type: str
    status: str
    detail: str = ''

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


# ---------------------------------------------------------------------------
# Local read-only data access (module-level so tests can monkeypatch)
# ---------------------------------------------------------------------------

def _safe_dataset_name(dataset: str) -> str:
    return str(dataset).replace(':', '_').replace('.', '_')


def _bid_series_as_strings(series: pd.Series) -> pd.Series:
    """bodyId column as strings unless it already is (the cache parquet
    stores bodyIds as strings; only convert foreign dtypes)."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series
    return series.astype(str)


def load_neuron_table(project_root: str, dataset: str) -> Optional[pd.DataFrame]:
    """The dataset's local all-neuron table (type -> bodyId facts)."""
    key = (str(project_root), _safe_dataset_name(dataset))
    if key in _NEURON_TABLE_CACHE:
        return _NEURON_TABLE_CACHE[key]
    root = Path(project_root)
    safe = key[1]
    candidates = [
        root / 'datasets' / safe / f'{safe}_allneurons_neuron_df.parquet',
        root / 'datasets' / safe / f'{safe}_allneurons_neuron_df.csv',
        root / 'datasets' / safe / f'{safe}_neurons.parquet',
        root / 'datasets' / safe / f'{safe}_neurons.csv',
        root / 'datasets' / safe / 'neurons.parquet',
        root / 'datasets' / safe / 'neurons.csv',
        root / 'neuron_indexes' / safe / 'neuron_index.parquet',
    ]
    frame = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = (
                pd.read_parquet(path)
                if path.suffix == '.parquet'
                else pd.read_csv(path, low_memory=False)
            )
            break
        except Exception:  # noqa: BLE001 - degrade, never block the export
            frame = None
    _NEURON_TABLE_CACHE[key] = frame
    return frame


def load_connection_frame(project_root: str, dataset: str) -> Optional[pd.DataFrame]:
    """The cached connections parquet reduced to the three needed columns.

    bodyId columns are kept in their stored dtype when already string-like
    (the cache parquet convention); otherwise they are coerced to str so
    membership tests against ``str(bodyId)`` sets are exact.
    """
    key = (str(project_root), _safe_dataset_name(dataset))
    if key in _CONNECTION_CACHE:
        return _CONNECTION_CACHE[key]
    path = Path(project_root) / 'cache' / key[1] / 'connections.parquet'
    frame = None
    if path.exists():
        try:
            frame = pd.read_parquet(
                path, columns=['bodyId_pre', 'bodyId_post', 'weight'])
            if len(frame):
                frame = frame.assign(
                    bodyId_pre=_bid_series_as_strings(frame['bodyId_pre']),
                    bodyId_post=_bid_series_as_strings(frame['bodyId_post']),
                )
        except Exception:  # noqa: BLE001
            try:
                frame = pd.read_parquet(path)
                for col in ('bodyId_pre', 'bodyId_post'):
                    if col in frame.columns:
                        frame[col] = _bid_series_as_strings(frame[col])
                frame = frame[['bodyId_pre', 'bodyId_post', 'weight']]
            except Exception:  # noqa: BLE001
                frame = None
    _CONNECTION_CACHE[key] = frame
    return frame


def load_in_graph_roles(folder: str) -> Optional[Dict[str, Set[str]]]:
    """Searched-graph bodyIds by role (the run's ``neurons_included.csv``).

    Returns ``{'source': {...}, 'intermediate': {...}, 'target': {...}}``
    or ``None`` when the file is unavailable.  The role split matters for
    the absence diagnosis: a type's ENTRY leg is its edge from the path
    sources — a huge edge from an in-graph intermediate (e.g. KC -> APL)
    says nothing about why pathfinding never entered the type.
    """
    key = str(folder)
    if key in _IN_GRAPH_CACHE:
        return _IN_GRAPH_CACHE[key]
    path = os.path.join(folder, 'data_details', 'neurons_included.csv')
    roles: Optional[Dict[str, Set[str]]] = None
    if folder and os.path.exists(path):
        try:
            frame = pd.read_csv(path, usecols=['group', 'bodyId'])
            roles = {}
            for group, ids in frame.groupby('group', sort=False):
                roles[str(group)] = {str(bid) for bid in ids['bodyId']}
        except Exception:  # noqa: BLE001
            roles = None
    _IN_GRAPH_CACHE[key] = roles
    return roles


def clear_caches() -> None:
    """Drop memoized frames (used by tests and long-lived processes)."""
    _NEURON_TABLE_CACHE.clear()
    _CONNECTION_CACHE.clear()
    _IN_GRAPH_CACHE.clear()


# ---------------------------------------------------------------------------
# Name/existence helpers
# ---------------------------------------------------------------------------

def _type_bodyids(type_map: Dict[str, Set[str]], type_name: str) -> Set[str]:
    """bodyIds typed ``type_name`` (exact cell match; hemi-suffix fallback).

    Multi-value type cells stay atomic — a comma-joined annotation is never
    split into candidate names (same convention as the type mapper).
    """
    ids = type_map.get(str(type_name))
    if ids:
        return ids
    base = str(type_name)
    for suffix in ('_L', '_R', '_U'):
        if base.endswith(suffix):
            return type_map.get(base[: -len(suffix)]) or set()
    return set()


def _build_type_map(frame: Optional[pd.DataFrame]) -> Dict[str, Set[str]]:
    """type cell -> set(bodyId strings), built once per dataset frame."""
    if frame is None or 'type' not in frame.columns or 'bodyId' not in frame.columns:
        return {}
    worked = pd.DataFrame({
        'type': frame['type'].astype(str),
        'bodyId': frame['bodyId'].astype(str),
    })
    return {
        str(name): set(group['bodyId'])
        for name, group in worked.groupby('type', sort=False)
    }


def _max_touching_weight(conn: Optional[pd.DataFrame],
                         in_graph: Optional[Set[str]],
                         bids: Set[str],
                         directed: bool = False) -> Optional[int]:
    """Max weight of edges between ``in_graph`` and ``bids``.

    ``directed`` restricts to the entry direction (pre ∈ in_graph →
    post ∈ bids) — the semantics of a path-search entry leg; the
    symmetric form also counts candidate→partner edges (e.g. an APL
    feedback edge), which say nothing about recruitment.
    ``None`` = diagnosis inputs unavailable; ``-1`` = no such edges.
    """
    if conn is None or in_graph is None or not bids:
        return None
    pre_in = conn['bodyId_pre'].isin(in_graph) & conn['bodyId_post'].isin(bids)
    if directed:
        mask = pre_in
    else:
        mask = pre_in | (
            conn['bodyId_post'].isin(in_graph) & conn['bodyId_pre'].isin(bids))
    sub = conn[mask]
    if sub.empty:
        return -1
    return int(sub['weight'].max())


# ---------------------------------------------------------------------------
# Coverage build
# ---------------------------------------------------------------------------

def _split_edge_key(edge_key: Any) -> Tuple[str, str]:
    text = str(edge_key)
    if ' -> ' in text:
        source, target = text.split(' -> ', 1)
        return source, target
    return text, ''


def build_query_type_coverage(analyzer, query: Dict[str, Any]
                              ) -> Dict[Tuple[str, str], TypeCoverageEntry]:
    """Resolve the union of a query's appeared types into every dataset."""
    try:
        aligned = analyzer.get_aligned_data_for_query(query)
    except Exception:  # noqa: BLE001
        return {}
    if aligned is None or aligned.empty:
        return {}

    dataset_names = [
        d for d in analyzer.parameters.get_dataset_names()
        if d in aligned.columns
    ]
    if not dataset_names:
        return {}

    union_types: List[str] = []
    seen_types: Set[str] = set()
    for edge_key in aligned.index:
        for name in _split_edge_key(edge_key):
            if name and name not in seen_types:
                seen_types.add(name)
                union_types.append(name)

    present: Dict[str, Set[str]] = {d: set() for d in dataset_names}
    for dataset in dataset_names:
        for edge_key, weight in aligned[dataset].items():
            try:
                if pd.isna(weight) or weight <= 0:
                    continue
            except TypeError:  # non-numeric weight
                continue
            source, target = _split_edge_key(edge_key)
            if source:
                present[dataset].add(source)
            if target:
                present[dataset].add(target)

    mapper = getattr(analyzer.parameters, '_auto_type_mapper', None)
    snapshot = getattr(analyzer, '_mapper_snapshot', None)
    requested_map = query.get('thresholds') or {}
    # The below-threshold / not-recruited split compares raw SYNAPSE edge
    # weights against the applied threshold — meaningless under the
    # connection-ratio basis (the frame's weight column is a synapse count,
    # the tier is a per-post fraction). Degrade the absence diagnosis
    # honestly instead of truncating tiers to 0 and mislabeling statuses.
    ratio_basis = (getattr(analyzer.parameters, 'weight_basis', 'synapse')
                   == 'connection_ratio')

    coverage: Dict[Tuple[str, str], TypeCoverageEntry] = {}

    for dataset in dataset_names:
        # Per-dataset facts, built once for this query's whole union.
        neuron_table = load_neuron_table(PROJECT_ROOT, dataset)
        type_map = _build_type_map(neuron_table)
        conn = load_connection_frame(PROJECT_ROOT, dataset)

        requested = requested_map.get(dataset)
        applied_int: Optional[int] = None
        if requested is not None:
            try:
                applied_int = int(requested)
            except (TypeError, ValueError):
                applied_int = None
            try:
                applied = analyzer._path_provenance_row(
                    dataset, int(requested)).get('applied_threshold')
                applied_int = int(applied)
            except Exception:  # noqa: BLE001
                pass

        folder = None
        if requested is not None:
            try:
                folder = analyzer.parameters.get_dataset_output_path(
                    dataset, int(requested))
            except Exception:  # noqa: BLE001
                folder = None
        roles = load_in_graph_roles(folder) if folder else None

        # Pre-filter the connection frame once for every candidate bodyId
        # of this query (the full frame is tens of millions of rows).
        pending: List[Tuple[str, str, Set[str]]] = []  # (type, resolved, bids)
        for type_name in union_types:
            if type_name in present[dataset]:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=True,
                    resolved_type=type_name, status=STATUS_PRESENT,
                    detail="appeared in this dataset's searched graph",
                )
                continue

            # Resolve the type into THIS dataset's namespace.  Canonical
            # names can carry merged-display alternates — "CB0937(CB2577)"
            # — which are a rendering construct the mapper and the neuron
            # tables never contain; resolution and same-name existence
            # checks use the base name (the hover/tooltip convention).
            resolve_name = str(type_name).split('(')[0].strip() or type_name
            candidates = [resolve_name]
            try:
                resolution = resolve_valid_targets(
                    mapper, resolve_name, None, dataset, snapshot=snapshot)
            except Exception:  # noqa: BLE001
                resolution = None
            if resolution is not None:
                licensed = list(expansion_targets(resolution))
                if licensed:
                    candidates = licensed
                elif resolution.status in (
                        STATUS_CONFLICT, STATUS_EVIDENCE_ONLY, STATUS_CLAIMED):
                    # Fail-closed verdicts surface verbatim.
                    coverage[(type_name, dataset)] = TypeCoverageEntry(
                        type=type_name, dataset=dataset, present=False,
                        resolved_type='', status=resolution.status,
                        detail=resolution.reason or '',
                    )
                    continue

            resolved_display = '/'.join(candidates)
            if neuron_table is None:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type='', status=STATUS_RESOLVED_ABSENT,
                    detail='dataset neuron table unavailable locally',
                )
                continue

            bids: Set[str] = set()
            for name in candidates:
                bids |= _type_bodyids(type_map, name)
            if not bids:
                if resolution is not None and (
                        resolution.status == STATUS_UNMAPPED):
                    status, detail = STATUS_UNMAPPED, (
                        'no mapping relation and no same-name neurons')
                elif resolution is not None and (
                        resolution.status == STATUS_MAPPER_UNAVAILABLE):
                    status, detail = STATUS_MAPPER_UNAVAILABLE, (
                        'mapper unavailable; no same-name neurons')
                else:
                    status, detail = STATUS_NOT_IN_DATASET, (
                        f'no neurons typed {resolved_display}')
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type=resolved_display, status=status,
                    detail=detail,
                )
                continue
            pending.append((type_name, resolved_display, bids))

        if not pending:
            continue
        if conn is not None:
            bid_union: Set[str] = set()
            for _, _, bids in pending:
                bid_union |= bids
            try:
                conn_sub = conn[
                    conn['bodyId_pre'].isin(bid_union)
                    | conn['bodyId_post'].isin(bid_union)
                ]
            except Exception:  # noqa: BLE001
                conn_sub = None
        else:
            conn_sub = None

        for type_name, resolved_display, bids in pending:
            # Role-aware diagnosis: the ENTRY leg into a path search is an
            # edge from a path-source neuron.  A type can carry huge edges
            # from in-graph intermediates (e.g. KC -> APL) yet never be
            # entered because its source-side leg is below threshold.
            in_graph_all: Optional[Set[str]] = (
                set().union(*roles.values()) if roles else None)
            src_ids: Optional[Set[str]] = (
                (roles.get('source') or set()) if roles else None)
            overall_max = _max_touching_weight(
                conn_sub, in_graph_all, bids)
            src_max = (
                _max_touching_weight(conn_sub, src_ids, bids, directed=True)
                if src_ids else None)

            def _detail(weight: int, partners: str) -> str:
                return (
                    f'max edge weight from {partners} {weight} '
                    f'< threshold {applied_int}')

            if ratio_basis or overall_max is None or applied_int is None:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type=resolved_display,
                    status=STATUS_RESOLVED_ABSENT,
                    detail=('below-threshold diagnosis compares synapse '
                            'edge weights and stays unavailable under '
                            'the connection-ratio basis'
                            if ratio_basis else
                            'diagnosis unavailable (no connection cache '
                            'or searched-graph list)'),
                )
            elif overall_max < 0:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type=resolved_display, status=STATUS_NO_EDGES,
                    detail='no edges to the searched graph at any weight',
                )
            elif src_max is not None and src_max >= 0:
                if src_max < applied_int:
                    coverage[(type_name, dataset)] = TypeCoverageEntry(
                        type=type_name, dataset=dataset, present=False,
                        resolved_type=resolved_display,
                        status=STATUS_BELOW_THRESHOLD,
                        detail=_detail(src_max, 'path sources'),
                    )
                else:
                    coverage[(type_name, dataset)] = TypeCoverageEntry(
                        type=type_name, dataset=dataset, present=False,
                        resolved_type=resolved_display,
                        status=STATUS_NOT_RECRUITED,
                        detail=(f'edge weight {src_max} >= threshold '
                                f'{applied_int} from path sources but '
                                f'not in searched graph'),
                    )
            elif overall_max < applied_int:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type=resolved_display,
                    status=STATUS_BELOW_THRESHOLD,
                    detail=_detail(overall_max, 'searched-graph partners'),
                )
            else:
                coverage[(type_name, dataset)] = TypeCoverageEntry(
                    type=type_name, dataset=dataset, present=False,
                    resolved_type=resolved_display,
                    status=STATUS_NOT_RECRUITED,
                    detail=(f'edge weight {overall_max} >= threshold '
                            f'{applied_int} from searched-graph partners '
                            f'but not in searched graph'),
                )
    return coverage


def build_type_coverage(analyzer) -> Dict[str, Dict[Tuple[str, str],
                                                    TypeCoverageEntry]]:
    """Coverage for every threshold query of the run, keyed by query id."""
    out: Dict[str, Dict[Tuple[str, str], TypeCoverageEntry]] = {}
    for query in analyzer.get_threshold_queries():
        query_id = query.get('id') or query.get('query_id')
        if not query_id:
            continue
        out[str(query_id)] = build_query_type_coverage(analyzer, query)
    return out


def coverage_rows(analyzer, coverage: Dict[str, Dict[Tuple[str, str],
                                                     TypeCoverageEntry]]
                  ) -> List[Dict[str, Any]]:
    """Long-form rows for ``type_resolution_union.csv``."""
    rows = []
    for query in analyzer.get_threshold_queries():
        query_id = str(query.get('id') or query.get('query_id') or '')
        query_label = query.get('label', query_id)
        entries = coverage.get(query_id) or {}
        if not entries:
            continue
        for (type_name, dataset), entry in sorted(
                entries.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            rows.append({
                'query_id': query_id,
                'query_label': query_label,
                'type': type_name,
                'dataset': dataset,
                'present': bool(entry.present),
                'resolved_type': entry.resolved_type,
                'resolution_status': entry.status,
                'detail': entry.detail,
            })
    return rows


def absence_note(coverage: Dict[Tuple[str, str], TypeCoverageEntry],
                 dataset: str, *type_names: str) -> str:
    """Why an EDGE is absent in ``dataset``: the failing endpoint(s).

    Returns '' when coverage has nothing to say; names each non-present
    endpoint, and falls back to 'edge not in run' when the endpoint types
    themselves are present but this particular edge did not survive.
    """
    if not coverage:
        return ''
    notes = []
    saw_any = False
    for name in type_names:
        base = str(name).split('(')[0]
        entry = coverage.get((base, dataset))
        if entry is None:
            continue
        saw_any = True
        if not entry.present:
            note = f'{base}: {entry.status_label}'
            if entry.detail and entry.status in (
                    STATUS_BELOW_THRESHOLD, STATUS_NOT_RECRUITED):
                note += f' ({entry.detail})'
            notes.append(note)
    if notes:
        return '; '.join(notes)
    if saw_any:
        return 'edge not in run'
    return ''


def node_coverage_line(coverage: Dict[Tuple[str, str], TypeCoverageEntry],
                       label: str, datasets: Iterable[str],
                       nickname_for) -> str:
    """One-line per-dataset absence summary for a node tooltip."""
    if not coverage:
        return ''
    base = str(label).split('(')[0]
    parts = []
    for dataset in datasets:
        entry = coverage.get((base, dataset))
        if entry is not None and not entry.present:
            parts.append(f'{nickname_for(dataset)} {entry.status_label}')
    if not parts:
        return ''
    return 'Coverage: ' + '; '.join(parts)
