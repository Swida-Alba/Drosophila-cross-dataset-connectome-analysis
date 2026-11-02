"""One structured node-label model for every ``A(B)`` display surface.

Labels are presentation; node ids stay canonical.  A node label is built
once as :class:`NodeDisplay` and rendered through :meth:`NodeDisplay.label`
in one of three styles:

``network``
    ``primary`` or ``primary\\n(alt/alt)`` — the vispath canvas form.  The
    newline is the primary/alternative boundary, so a primary that itself
    contains ``(``/``)`` stays unambiguous.
``inline``
    ``primary(alt/alt)`` when no name contains ``(``/``)``/``/``; otherwise
    the plain primary (the alternatives remain in the hover).
``plain``
    the primary only — ids, keys, matching, sorting.

Names are **atomic**: a comma is part of a native type name
(``SMP523,SMP524`` is one BANC type), never a list separator.  Lists cross
this API as tuples/lists and are never joined into a name.

The observed variants of a run (``metrics.last_edge_display_names``) are
the authority for names that actually occurred; mapper-only names follow the
:mod:`comparison.type_resolver` status policy (mapped/bridged/valid split
are licensed, claims/conflicts are hover evidence only).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .type_resolver import (
        STATUS_BRIDGED,
        STATUS_CLAIMED,
        STATUS_CONFLICT,
        STATUS_EVIDENCE_ONLY,
        STATUS_MAPPED,
        STATUS_UNMAPPED,
        STATUS_VALID_SPLIT,
        MapperSnapshot,
        is_residual_key,
        resolve_valid_targets,
    )
except ImportError:  # pragma: no cover - direct package imports
    from type_resolver import (
        STATUS_BRIDGED,
        STATUS_CLAIMED,
        STATUS_CONFLICT,
        STATUS_EVIDENCE_ONLY,
        STATUS_MAPPED,
        STATUS_UNMAPPED,
        STATUS_VALID_SPLIT,
        MapperSnapshot,
        is_residual_key,
        resolve_valid_targets,
    )

__all__ = [
    'DATASET_ORDER', 'NAME_LIST_SEP', 'ALT_SEP',
    'DatasetNames', 'NodeDisplay',
    'build_node_display', 'format_label', 'format_hover_value',
    'parse_display_label', 'invert_edge_display_names',
    'order_datasets', 'dataset_codes',
]

#: Default dataset order for alternatives and hover lines.
DATASET_ORDER = ('male-cns:v1.0', 'flywire_FAFB_v783', 'banc_v888',
                 'hemibrain:v1.2.1', 'manc:v1.0')
#: Separator inside one dataset's hover value.
NAME_LIST_SEP = ' | '
#: Separator between alternatives in a rendered label.
ALT_SEP = '/'

_LICENSED_STATUSES = frozenset({
    STATUS_MAPPED, STATUS_BRIDGED, STATUS_VALID_SPLIT})
#: Statuses whose names are shown in the hover with a per-name annotation.
_HOVER_STATUS_NOTES = frozenset({
    STATUS_CLAIMED, STATUS_EVIDENCE_ONLY, STATUS_CONFLICT})

_INLINE_UNSAFE = ('(', ')', ALT_SEP)

_LABEL_MAPPER_LABELS: 'weakref.WeakKeyDictionary[Any, frozenset]' = (
    weakref.WeakKeyDictionary())


@dataclass(frozen=True)
class DatasetNames:
    """Names carried by one selected dataset for one node.

    ``names`` includes hover evidence (a claimed target is listed) even when
    the name is not licensed for the label.  ``observed_names`` keeps the
    per-name provenance internally: those names decorate unconditionally.
    """

    dataset: str
    code: str
    names: Tuple[str, ...]
    status: str
    observed: bool
    observed_names: Tuple[str, ...] = ()

    def licensed_names(self) -> Tuple[str, ...]:
        """Names allowed in the alternative list (and thus in a label)."""
        if self.status in _LICENSED_STATUSES:
            return self.names
        return self.observed_names


@dataclass(frozen=True)
class NodeDisplay:
    """Structured presentation of one canonical node."""

    node_id: str
    primary: str
    primary_dataset: Optional[str]
    alternatives: Tuple[str, ...]
    names_by_dataset: Tuple[DatasetNames, ...]
    tier: str
    decorated: bool
    warnings: Tuple[str, ...] = ()

    def label(self, style: str = 'network') -> str:
        return format_label(self, style)

    def hover(self) -> Dict[str, Any]:
        """``{code: name}`` or ``{code: [name, ...]}`` in dataset order."""
        out: Dict[str, Any] = {}
        for entry in self.names_by_dataset:
            value = _hover_value(entry)
            if value:
                out[entry.code] = value
        return out

    def all_names(self) -> Tuple[str, ...]:
        seen = []
        for entry in self.names_by_dataset:
            for name in entry.names:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)


# --- formatting -----------------------------------------------------------

def format_label(display: NodeDisplay, style: str = 'network') -> str:
    """Render one :class:`NodeDisplay` in ``network``/``inline``/``plain``."""
    if style not in ('network', 'inline', 'plain'):
        raise ValueError(f'unknown label style: {style!r}')
    if style == 'plain' or not display.decorated:
        return display.primary
    alts = ALT_SEP.join(display.alternatives)
    if style == 'network':
        return f'{display.primary}\n({alts})'
    unsafe = any(ch in display.primary for ch in _INLINE_UNSAFE)
    unsafe = unsafe or any(
        any(ch in alt for ch in _INLINE_UNSAFE)
        for alt in display.alternatives)
    if unsafe:
        return display.primary
    return f'{display.primary}({alts})'


def format_hover_value(value) -> str:
    """Render one hover value the way the network canvas shows it."""
    if isinstance(value, (list, tuple)):
        return NAME_LIST_SEP.join(_quote_hover_name(str(v)) for v in value)
    return _quote_hover_name(str(value))


def _quote_hover_name(name: str) -> str:
    """Quote a native name containing the list separator or a comma."""
    if ',' in name or NAME_LIST_SEP in name:
        return f'"{name}"'
    return name


def _hover_value(entry: DatasetNames) -> Any:
    if not entry.names:
        return None
    observed = set(entry.observed_names)
    rendered: List[str] = []
    for name in entry.names:
        if name not in observed and entry.status in _HOVER_STATUS_NOTES:
            rendered.append(f'{name} ({entry.status})')
        else:
            rendered.append(name)
    if (entry.status == STATUS_VALID_SPLIT
            and any(name not in observed for name in entry.names)):
        rendered.append(f'({entry.status})')
    if len(rendered) == 1:
        return rendered[0]
    return rendered


# --- compatibility parsing -------------------------------------------------

def parse_display_label(text) -> Tuple[str, Tuple[str, ...]]:
    """Best-effort split of a *legacy* rendered label.

    The new model never parses a label: ids and keys are canonical.  This
    helper exists for old saved strings only.  A two-line label splits at the
    first newline and returns the primary unchanged (including its own
    parentheses); a one-line label splits its trailing ``(...)`` group.
    """
    import re

    raw = str(text if text is not None else '')
    if '\n' in raw:
        primary, rest = raw.split('\n', 1)
        return primary.strip(), _parse_alt_group(rest)
    match = re.search(r'\(([^()]*)\)\s*$', raw)
    if match and raw[:match.start()].strip():
        return raw[:match.start()].strip(), _parse_alt_group(
            raw[match.start():])
    return raw.strip(), ()


def _parse_alt_group(text: str) -> Tuple[str, ...]:
    chunk = str(text).strip()
    if not (chunk.startswith('(') and chunk.endswith(')')):
        return ()
    inner = chunk[1:-1].strip()
    if not inner:
        return ()
    return tuple(part.strip() for part in inner.split(ALT_SEP)
                 if part.strip())


def invert_edge_display_names(variants) -> Dict[str, Dict[str, List[str]]]:
    """``{canonical: {raw: {datasets}}}`` -> ``{canonical: {dataset: [raw]}}``.

    ``metrics.last_edge_display_names`` stores the run-observed variants;
    the display builder needs them grouped by dataset, in first-seen order.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for canonical, by_name in (variants or {}).items():
        per_dataset: Dict[str, List[str]] = {}
        for raw_name, datasets in (by_name or {}).items():
            for dataset in datasets or ():
                per_dataset.setdefault(str(dataset), []).append(
                    str(raw_name))
        out[str(canonical)] = {
            ds: list(dict.fromkeys(names))
            for ds, names in per_dataset.items()}
    return out


# --- dataset helpers -------------------------------------------------------

def _label_mapper_labels(label_mapper) -> frozenset:
    if label_mapper is None or getattr(label_mapper, 'is_empty', True):
        return frozenset()
    cached = _LABEL_MAPPER_LABELS.get(label_mapper)
    if cached is not None:
        return cached
    labels = set()
    for role in ('source', 'target', 'intermediate'):
        try:
            labels.update(str(x) for x in label_mapper.get_all_std_labels(role))
        except Exception:  # noqa: BLE001 — role simply absent
            pass
    frozen = frozenset(labels)
    try:
        _LABEL_MAPPER_LABELS[label_mapper] = frozen
    except TypeError:  # pragma: no cover - unhashable custom mapper
        pass
    return frozen


def _mapping_key(dataset, mapper) -> str:
    if mapper is not None:
        try:
            return str(mapper._get_type_mapping_key(dataset))
        except Exception:  # noqa: BLE001 — mapper without the helper
            pass
    return str(dataset)


def _same_namespace(a, b, mapper) -> bool:
    if a is None or b is None:
        return False
    return _mapping_key(a, mapper) == _mapping_key(b, mapper)


def _dataset_rank(dataset, mapper) -> int:
    key = _mapping_key(dataset, mapper).lower()
    if 'male-cns' in key or 'male_cns' in key:
        return 0
    if 'fafb' in key or 'flywire' in key:
        return 1
    if 'banc' in key:
        return 2
    if 'hemibrain' in key:
        return 3
    if 'manc' in key:
        return 4
    return 5


def _ordered_datasets(datasets, mapper) -> List[str]:
    unique: List[str] = []
    for dataset in datasets:
        if dataset not in unique:
            unique.append(dataset)
    indexed = list(enumerate(unique))
    indexed.sort(key=lambda item: (_dataset_rank(item[1], mapper), item[0]))
    return [dataset for _, dataset in indexed]


def order_datasets(datasets, mapper=None) -> List[str]:
    """Selected datasets in the contract order (MCNS/F/B/H/MANC, then caller)."""
    return _ordered_datasets(datasets or (), mapper)


def dataset_codes(datasets, mapper=None, code_map=None) -> List[str]:
    """Collision-qualified short codes for ``datasets``."""
    return _codes_for(list(datasets or ()), mapper, code_map)


def _codes_for(datasets, mapper, code_map) -> List[str]:
    if code_map is not None:
        out = []
        for dataset in datasets:
            code = None
            for key in (dataset, _mapping_key(dataset, mapper)):
                try:
                    code = code_map.get(key)
                except AttributeError:  # a sequence, not a dict
                    code = None
                if code:
                    break
            if code is None and isinstance(code_map, (list, tuple)):
                try:
                    code = code_map[datasets.index(dataset)]
                except (ValueError, IndexError):
                    code = None
            out.append(str(code) if code else _fallback_code(dataset, mapper))
        return out
    if mapper is not None:
        try:
            codes = list(mapper._get_dataset_short_codes(datasets))
            if len(codes) == len(datasets):
                return [str(code) for code in codes]
        except Exception:  # noqa: BLE001 — mapper without the helper
            pass
    return [_fallback_code(dataset, mapper) for dataset in datasets]


def _fallback_code(dataset, mapper) -> str:
    if mapper is not None:
        try:
            return str(mapper.get_dataset_short_code(dataset))
        except Exception:  # noqa: BLE001
            pass
    text = str(dataset).strip()
    return text[0].upper() if text else '?'


def _split_hemi(name: str) -> Tuple[str, str]:
    try:
        from ..utils.naming_utils import split_hemi_suffix
    except ImportError:  # pragma: no cover - direct package imports
        from utils.naming_utils import split_hemi_suffix
    return split_hemi_suffix(str(name))


def _observed_for(observed_node, dataset, mapper) -> List[str]:
    if not observed_node:
        return []
    if dataset in observed_node:
        return [str(name) for name in observed_node[dataset]]
    for key, names in observed_node.items():
        if _same_namespace(key, dataset, mapper):
            return [str(name) for name in names]
    return []


def _merged_names(*groups: Iterable[str]) -> List[str]:
    out: List[str] = []
    for group in groups:
        for name in group or ():
            text = str(name)
            if text and text not in out:
                out.append(text)
    return out


def _resolve(mapper, snapshot, base_name, source_dataset, target_dataset):
    if mapper is None:
        return None
    return resolve_valid_targets(
        mapper, base_name, source_dataset, target_dataset,
        snapshot=snapshot)


# --- construction ----------------------------------------------------------

def build_node_display(
    node,
    datasets,
    *,
    mapper=None,
    label_mapper=None,
    merge_policy=None,
    observed=None,
    verbatim=(),
    primary=None,
    source_dataset=None,
    code_map=None,
    snapshot=None,
) -> NodeDisplay:
    """Build the single display model for one canonical node.

    ``observed`` is ``{node: {dataset: [names]}}`` — normally
    :func:`invert_edge_display_names` of ``metrics.last_edge_display_names``.
    ``verbatim`` names (tier-0 labels, resolved source/target queries) keep
    the node id as primary and never gain a parenthetical.
    """
    node_id = str(node)
    # Compatibility: the legacy display wrappers lazy-loaded the mapper.
    if mapper is not None and not getattr(mapper, '_loaded', False):
        loader = getattr(mapper, 'load', None)
        if callable(loader):
            try:
                loader()
            except Exception:  # noqa: BLE001 - display falls back to ids
                pass
    selected = _ordered_datasets(datasets or (), mapper)
    codes = _codes_for(selected, mapper, code_map)
    code_by_dataset = dict(zip(selected, codes))
    base_name, hemi_suffix = _split_hemi(node_id)
    warnings: List[str] = []

    observed_by_node = (observed or {}).get(node_id)
    if observed_by_node is None:
        observed_by_node = (observed or {}).get(base_name) or {}
    explicit_source = source_dataset is not None
    detected_source = None
    if mapper is not None:
        try:
            detected_source = mapper._detect_type_source(base_name)
        except Exception:  # noqa: BLE001
            detected_source = None
    if source_dataset is None:
        if detected_source is not None:
            source_dataset = detected_source
        elif observed_by_node:
            dataset_hits = [
                ds for ds, names in observed_by_node.items() if names]
            if len(dataset_hits) == 1:
                source_dataset = dataset_hits[0]
    # Identity is only trusted when the source namespace is known (explicit
    # or mapper-detected).  A source merely inferred from one observed raw
    # variant must not turn that variant into the dataset's identity name.
    source_known = bool(
        source_dataset is not None
        and (explicit_source or detected_source is not None))

    snap = snapshot
    if snap is None and mapper is not None:
        snap = MapperSnapshot(mapper)

    verbatim_set = {str(value) for value in (verbatim or ())}
    governed = (
        node_id in verbatim_set
        or base_name in verbatim_set
        or base_name in _label_mapper_labels(label_mapper)
        or node_id in _label_mapper_labels(label_mapper))
    if merge_policy is not None and not governed:
        try:
            governed = (
                merge_policy.group_by_label(node_id) is not None
                or merge_policy.group_by_label(base_name) is not None)
        except Exception:  # noqa: BLE001
            governed = False

    merge_names_by_dataset: Dict[str, List[str]] = {}
    if merge_policy is not None:
        for label in (node_id, base_name):
            try:
                names = merge_policy.names_by_dataset(label) or {}
            except Exception:  # noqa: BLE001
                names = {}
            if names:
                merge_names_by_dataset = {
                    str(ds): [str(n) for n in values]
                    for ds, values in names.items()}
                break

    entries: List[DatasetNames] = []
    for dataset in selected:
        observed_names = _observed_for(observed_by_node, dataset, mapper)
        merge_names: List[str] = []
        for key, names in merge_names_by_dataset.items():
            if _same_namespace(key, dataset, mapper):
                merge_names = list(names)
                break
        identity_name = (
            node_id
            if (_same_namespace(dataset, source_dataset, mapper)
                and (source_known
                     or node_id in observed_names
                     or node_id in merge_names))
            else None)

        mapper_names: List[str] = []
        status = STATUS_MAPPED if identity_name else STATUS_UNMAPPED
        if identity_name is None and mapper is not None:
            resolution = _resolve(
                mapper, snap, base_name, source_dataset, dataset)
            if resolution is not None:
                status = resolution.status
                if status in _LICENSED_STATUSES or status in (
                        STATUS_CLAIMED, STATUS_EVIDENCE_ONLY):
                    mapper_names = [
                        f'{target}{hemi_suffix}'
                        for target in resolution.target_types]

        names = _merged_names(
            merge_names, observed_names,
            [identity_name] if identity_name else (), mapper_names)
        observed_all = _merged_names(merge_names, observed_names)
        entries.append(DatasetNames(
            dataset=str(dataset),
            code=code_by_dataset.get(dataset, _fallback_code(dataset, mapper)),
            names=tuple(names),
            status=status,
            observed=bool(observed_all),
            observed_names=tuple(observed_all),
        ))

    all_names = _merged_names(*(entry.names for entry in entries))
    licensed_names = _merged_names(
        *(entry.licensed_names() for entry in entries))
    primary_dataset: Optional[str] = None
    explicit = str(primary) if primary is not None else None

    residual = is_residual_key(base_name)
    if governed or residual:
        chosen = node_id
        tier = 'verbatim' if governed else 'residual'
        if node_id in all_names:
            primary_dataset = next(
                entry.dataset for entry in entries if node_id in entry.names)
    elif explicit and explicit in licensed_names:
        chosen = explicit
        tier = 'native'
        primary_dataset = next(
            entry.dataset for entry in entries if explicit in entry.names)
    else:
        if explicit:
            warnings.append('primary_not_mapped')
        mcns_entry = next(
            (entry for entry in entries
             if _mapping_key(entry.dataset, mapper).startswith('male-cns')),
            None)
        # A Male-CNS primary must be a licensed canonical name (or the node
        # id itself).  An observed raw variant in the MCNS dataset is hover
        # evidence — it must not flip the direction of a canonical key.
        mcns_names: Tuple[str, ...] = ()
        if mcns_entry is not None:
            if mcns_entry.status in _LICENSED_STATUSES:
                mcns_names = mcns_entry.licensed_names()
            elif node_id in mcns_entry.names:
                mcns_names = (node_id,)
        if mcns_names:
            chosen = mcns_names[0]
            primary_dataset = mcns_entry.dataset
            tier = 'canonical'
        else:
            chosen = node_id
            primary_dataset = (
                next((entry.dataset for entry in entries
                      if node_id in entry.names), None))
            tier = 'native'

    alternatives: List[str] = []
    if not (governed or residual):
        for entry in entries:
            for name in entry.licensed_names():
                if name == chosen or name in alternatives:
                    continue
                alternatives.append(name)

    if alternatives and (any(ch in chosen for ch in _INLINE_UNSAFE)
                         or any(any(ch in alt for ch in _INLINE_UNSAFE)
                                for alt in alternatives)):
        warnings.append('inline_ambiguous')

    return NodeDisplay(
        node_id=node_id,
        primary=chosen,
        primary_dataset=primary_dataset,
        alternatives=tuple(alternatives),
        names_by_dataset=tuple(entries),
        tier=tier,
        decorated=bool(alternatives),
        warnings=tuple(dict.fromkeys(warnings)),
    )
