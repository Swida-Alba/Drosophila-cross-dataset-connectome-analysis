"""Standalone input-query resolver (plan §7E).

Maps each raw query token (a type name, a taxonomy-column value, a bodyId,
a custom-group label, or a wildcard/regex pattern) to each called dataset
through the SAME backend the auto type mapper uses, emitting a structured
record with method, status, confidence and evidence — instead of the old
implicit same-name pass-through.

The previous behaviour (``ComparisonParameters._resolve_neurons_with_auto_mapping``)
used only ``resolve_one_target`` and, when that returned nothing, silently
re-used the raw token for every dataset. That makes a bare name echo
(``aMe12``) indistinguishable from a high-confidence crosswalk rename, and
the mapper itself ranks same-name as its LOWEST evidence tier (6). This
module surfaces that distinction so the report and the user can see it.

Read-only w.r.t. the mapper: it only calls the shared resolver APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Resolver statuses (superset of type_resolver's, plus non-type methods).
STATUS_MAPPED = 'mapped'
STATUS_BRIDGED = 'bridged'
STATUS_VALID_SPLIT = 'valid_split'
STATUS_EVIDENCE_ONLY = 'evidence_only'
STATUS_CONFLICT = 'conflict'
STATUS_UNMAPPED = 'unmapped'
STATUS_SAME_NAME = 'same_name_fallback'
STATUS_SAME_NAME_IDENTITY = 'same_name_identity'
STATUS_TAXONOMY = 'taxonomy'
STATUS_TAXONOMY_MAPPED = 'taxonomy_mapped'
STATUS_BODY_ID = 'body_id'
STATUS_GROUP = 'group'
STATUS_PATTERN = 'pattern'

# Evidence annotation for same-name identity records (plan §2.1): what does
# the curated relation say about the identity itself?
EVIDENCE_CONFIRMED = 'confirmed'
EVIDENCE_CONTRADICTED = 'contradicted'
EVIDENCE_NONE = 'none'

# Confidence tiers (higher = more trustworthy). Mirrors the mapper's
# evidence tiers: curated label > crosswalk > ... > same name.
CONFIDENCE = {
    'group': 5,
    STATUS_BODY_ID: 5,
    'native_type': 4,
    STATUS_MAPPED: 4,
    STATUS_BRIDGED: 3,
    STATUS_VALID_SPLIT: 3,
    STATUS_TAXONOMY: 3,
    STATUS_SAME_NAME_IDENTITY: 3,
    STATUS_TAXONOMY_MAPPED: 2,
    STATUS_EVIDENCE_ONLY: 2,
    STATUS_PATTERN: 2,
    STATUS_SAME_NAME: 1,
    STATUS_CONFLICT: 0,
    STATUS_UNMAPPED: 0,
}


@dataclass
class QueryResolution:
    """Resolution of one raw token into one dataset."""
    token: str
    dataset: str
    role: str = 'source'
    status: str = STATUS_UNMAPPED
    method: str = 'unmatched'
    target_types: List[str] = field(default_factory=list)
    matched_column: Optional[str] = None
    evidence_chain: Optional[str] = None
    confidence: int = 0
    note: str = ''
    evidence: str = ''

    def as_dict(self) -> Dict[str, Any]:
        return {
            'token': self.token,
            'dataset': self.dataset,
            'role': self.role,
            'status': self.status,
            'method': self.method,
            'target_types': list(self.target_types),
            'matched_column': self.matched_column,
            'evidence_chain': self.evidence_chain,
            'confidence': self.confidence,
            'note': self.note,
            'evidence': self.evidence,
        }


def _is_body_id(token) -> bool:
    if isinstance(token, int):
        return True
    return isinstance(token, str) and token.strip().isdigit()


def _is_pattern(token) -> bool:
    if not isinstance(token, str):
        return False
    return '*' in token or '.*' in token


def _classify(token, mapper) -> str:
    if _is_body_id(token):
        return STATUS_BODY_ID
    if _is_pattern(token):
        return STATUS_PATTERN
    # A token that names a known type in some dataset is a type token;
    # otherwise it may be a taxonomy-column value.
    try:
        if mapper is not None and mapper.detect_type_source(str(token)):
            return 'type'
    except Exception:
        pass
    return STATUS_TAXONOMY


def resolve_query_tokens(
    tokens: Sequence,
    datasets: Sequence[str],
    mapper,
    *,
    role: str = 'source',
    source_dataset: Optional[str] = None,
    taxonomy_resolver=None,
    group_lookup=None,
) -> List[Dict[str, Any]]:
    """Resolve every token against every dataset into structured records.

    ``taxonomy_resolver(token, dataset) -> list[str] | None`` expands a
    taxonomy-column value to member types for one dataset (optional).
    ``group_lookup(token) -> bool`` reports whether a token is a custom
    group label (optional).
    """
    records: List[QueryResolution] = []

    for token in tokens or []:
        kind = _classify(token, mapper)

        if kind == STATUS_BODY_ID:
            for ds in datasets:
                records.append(QueryResolution(
                    token=str(token), dataset=ds, role=role,
                    status=STATUS_BODY_ID, method='body_id',
                    matched_column='bodyId', confidence=CONFIDENCE[STATUS_BODY_ID],
                    note='Dataset-scoped identifier; applies only to its home '
                         'dataset.'))
            continue

        if kind == STATUS_PATTERN:
            for ds in datasets:
                records.append(QueryResolution(
                    token=str(token), dataset=ds, role=role,
                    status=STATUS_PATTERN, method='pattern',
                    confidence=CONFIDENCE[STATUS_PATTERN],
                    note='Wildcard/regex pattern passed through to each '
                         "dataset's identity search."))
            continue

        if group_lookup is not None:
            try:
                if group_lookup(token):
                    for ds in datasets:
                        records.append(QueryResolution(
                            token=str(token), dataset=ds, role=role,
                            status=STATUS_GROUP, method='group',
                            confidence=CONFIDENCE['group'],
                            note='Custom group label expanded per dataset.'))
                    continue
            except Exception:
                pass

        if kind == STATUS_TAXONOMY and taxonomy_resolver is not None:
            expansions: Dict[str, List[str]] = {}
            for ds in datasets:
                try:
                    members = taxonomy_resolver(str(token), ds)
                except Exception:
                    members = None
                if members:
                    expansions[ds] = sorted({str(m) for m in members})
            if expansions:
                hit_datasets = list(expansions)
                all_members = sorted({m for members in expansions.values()
                                      for m in members})
                for ds in datasets:
                    if ds in expansions:
                        records.append(QueryResolution(
                            token=str(token), dataset=ds, role=role,
                            status=STATUS_TAXONOMY, method='taxonomy_column',
                            target_types=expansions[ds],
                            confidence=CONFIDENCE[STATUS_TAXONOMY],
                            note='Taxonomy-column value expanded to member '
                                 'types.'))
                        continue
                    # Route A: no native taxonomy hit — bridge the concept
                    # by mapping every member type through the auto mapper
                    # into this dataset and querying the union.
                    union: set = set()
                    unmapped: List[str] = []
                    for hit_ds, members in expansions.items():
                        for member in members:
                            targets = _member_targets(
                                mapper, member, hit_ds, ds)
                            if targets:
                                union.update(targets)
                            else:
                                unmapped.append(f'{member} ({hit_ds})')
                    if union:
                        note = (f'Mapped via {len(all_members)} member '
                                f'types expanded in '
                                f'{", ".join(hit_datasets)}.')
                        if unmapped:
                            shown = ', '.join(sorted(unmapped)[:5])
                            more = ('…' if len(unmapped) > 5 else '')
                            note += (f' {len(unmapped)} members had no '
                                     f'counterpart here: {shown}{more}')
                        records.append(QueryResolution(
                            token=str(token), dataset=ds, role=role,
                            status=STATUS_TAXONOMY_MAPPED,
                            method='taxonomy_member_mapping',
                            target_types=sorted(union),
                            confidence=CONFIDENCE[STATUS_TAXONOMY_MAPPED],
                            note=note))
                    else:
                        records.append(QueryResolution(
                            token=str(token), dataset=ds, role=role,
                            status=STATUS_UNMAPPED, method='unmatched',
                            note='Taxonomy value not present in this '
                                 'dataset and no member mapped.'))
                continue

        # Type token (or taxonomy with no resolver): shared validity core.
        records.extend(_resolve_type_token(
            str(token), datasets, mapper,
            role=role, source_dataset=source_dataset))

    return [r.as_dict() for r in records]


def _resolve_type_token(token, datasets, mapper, *, role,
                        source_dataset) -> List[QueryResolution]:
    from .type_resolver import (
        STATUS_BRIDGED as TR_BRIDGED, STATUS_CONFLICT as TR_CONFLICT,
        STATUS_EVIDENCE_ONLY as TR_EVIDENCE, STATUS_MAPPED as TR_MAPPED,
        STATUS_UNMAPPED as TR_UNMAPPED, STATUS_VALID_SPLIT as TR_SPLIT,
        resolve_valid_targets,
    )

    # The resolver core strips internally; normalize here too so identity
    # detection and equivalence comparison match the stripped answer.
    token = str(token).strip()

    src = source_dataset
    # When the source namespace was DETECTED (not caller-specified), the
    # token is by construction a native type of it.
    src_detected = False
    if not src and mapper is not None:
        try:
            src = mapper.detect_type_source(token)
            src_detected = src is not None
        except Exception:
            src = None

    out: List[QueryResolution] = []
    for ds in datasets:
        if mapper is None:
            out.append(_same_name(token, ds, role,
                                  'Type mapper unavailable.'))
            continue
        try:
            res = resolve_valid_targets(mapper, token, src, ds)
        except Exception as e:
            out.append(QueryResolution(
                token=token, dataset=ds, role=role, status=STATUS_UNMAPPED,
                method='unmatched', note=f'resolver error: {e}'))
            continue

        # A mapper conflict refuses first — even when the name is native in
        # this dataset (generic numbered false friends, §1.4 of the plan).
        if res.status == TR_CONFLICT:
            out.append(QueryResolution(
                token=token, dataset=ds, role=role, status=STATUS_CONFLICT,
                method='conflict', confidence=0,
                note='Unresolved mapping conflict; no automatic target.'))
            continue

        # Same-name-first query tier (plan §2.1): when the token is a native
        # type of this dataset, the identity is the first-choice QUERY
        # answer — the curated relation only annotates it (confirmed /
        # contradicted / none).  A unique *derivation* answer (bridge end
        # != the name) still wins; a unique CURATED different-name relation
        # does NOT replace the identity, it is displayed as counter-evidence.
        try:
            native = (
                (ds == src
                 and (src_detected or src is None
                      or _has_native_type(mapper, token, ds)))
                or (src is not None and ds != src
                    and (_same_native_namespace(mapper, src, ds)
                         or _has_native_type(mapper, token, ds))))
        except Exception:
            native = ds == src
        if native:
            out.append(_identity_record(
                token, ds, role, mapper, src, res))
            continue

        kind = str(getattr(res, 'kind', '') or '')
        if res.status in (TR_MAPPED, TR_BRIDGED) and res.equivalence_key:
            # Cross-dataset resolution that returns the SAME name is a
            # name echo (mapper tier 6 / bare bridge), not evidence-based
            # equivalence — surface it as an explicit fallback.  RES-9:
            # except a FIRED same-name-first selection (`suspects`), which
            # is the mapper's deliberate pick, not an echo.  A curated
            # rename never reaches this branch (its target differs from
            # the token), and a name-only identity — curated flag or not —
            # stays flagged (the pinned echo contract).
            if (str(res.equivalence_key) == token
                    and not getattr(res, 'suspects', False)):
                out.append(_same_name(
                    token, ds, role,
                    'Same-name echo only (mapper tier 6); low confidence.'))
            else:
                status = (STATUS_MAPPED if res.status == TR_MAPPED
                          else STATUS_BRIDGED)
                out.append(QueryResolution(
                    token=token, dataset=ds, role=role, status=status,
                    method='mapped_type',
                    target_types=[str(res.equivalence_key)],
                    evidence_chain=kind,
                    confidence=CONFIDENCE[status],
                    note=f'resolved via {kind or "mapping"}'))
            continue
        if res.status == TR_SPLIT and res.target_types:
            out.append(QueryResolution(
                token=token, dataset=ds, role=role, status=STATUS_VALID_SPLIT,
                method='mapped_type',
                target_types=[str(t) for t in res.target_types],
                confidence=CONFIDENCE[STATUS_VALID_SPLIT],
                note='valid split: all licensed branches'))
            continue
        # Mapped/bridged but not a unique target ("one of N"): keep the
        # candidate targets as a licensed VALID_SPLIT-style expansion.
        # (2026-09-27 duality alignment: candidate discovery licenses a
        # bridge-derived fan-out — the same edge reads valid_split_evidence
        # on the decision surface — so its branches are as defensible a
        # query as a crosswalk split's.  Demoting them to evidence_only
        # made `drop_unresolved` silently drop bridge-derived partners
        # while keeping crosswalk splits.)
        if res.status in (TR_MAPPED, TR_BRIDGED) and len(
                res.target_types or ()) > 1:
            out.append(QueryResolution(
                token=token, dataset=ds, role=role,
                status=STATUS_VALID_SPLIT, method='mapped_type',
                target_types=[str(t) for t in res.target_types],
                evidence_chain=kind,
                confidence=CONFIDENCE[STATUS_VALID_SPLIT],
                note='one of N fan-out: all licensed branches'))
            continue
        if res.status == TR_EVIDENCE:
            out.append(QueryResolution(
                token=token, dataset=ds, role=role,
                status=STATUS_EVIDENCE_ONLY, method='mapped_type',
                target_types=[str(t) for t in (res.target_types or [])],
                confidence=CONFIDENCE[STATUS_EVIDENCE_ONLY],
                note='evidence-only relation; no unique target'))
            continue
        if res.status == TR_UNMAPPED:
            # Unknown to the mapper: the raw token may still be a native
            # name that the identity search can match — flag it honestly.
            out.append(_same_name(token, ds, role,
                                  'No mapper relation; raw-name pass-through.'))
            continue
        out.append(QueryResolution(
            token=token, dataset=ds, role=role, status=STATUS_UNMAPPED,
            method='unmatched', note=f'unresolved ({res.status})'))
    return out


def _has_native_type(mapper, token, ds) -> bool:
    """Native-presence check against the mapper's per-dataset type sets."""
    has = getattr(mapper, 'has_native_type', None)
    if callable(has):
        try:
            return bool(has(str(token), ds))
        except Exception:
            return False
    # Older mapper builds: fall back to the type-count accessor.
    try:
        return bool(mapper.get_type_neuron_count(str(token), ds) > 0)
    except Exception:
        return False


def _identity_record(token, ds, role, mapper, src, res) -> QueryResolution:
    """Same-name-first identity record with curated-evidence annotation.

    Called only when the caller's native check passed (detected source
    namespace, same mapping namespace, or verified native presence)."""
    if ds == src:
        return QueryResolution(
            token=token, dataset=ds, role=role,
            status=STATUS_SAME_NAME_IDENTITY, method='native_type',
            target_types=[token], matched_column='type',
            evidence=EVIDENCE_CONFIRMED, confidence=4,
            note='native type of the source dataset')
    dec = _curated_decision(mapper, token, str(src), ds)
    d_status = str(dec.get('status') or '')
    d_target = dec.get('target_type')
    if d_status in ('mapped', 'bridged') and d_target:
        if str(d_target) == token:
            return QueryResolution(
                token=token, dataset=ds, role=role,
                status=STATUS_SAME_NAME_IDENTITY,
                method='same_name_identity',
                target_types=[token], matched_column='type',
                evidence_chain=str(dec.get('relationship') or ''),
                evidence=EVIDENCE_CONFIRMED,
                confidence=CONFIDENCE[STATUS_SAME_NAME_IDENTITY],
                note='native same-name type; cross-dataset relation '
                     'confirms the identity')
        return QueryResolution(
            token=token, dataset=ds, role=role,
            status=STATUS_SAME_NAME_IDENTITY, method='same_name_identity',
            target_types=[token], matched_column='type',
            evidence_chain=f'curated counterpart: {d_target}',
            evidence=EVIDENCE_CONTRADICTED,
            confidence=CONFIDENCE[STATUS_SAME_NAME_IDENTITY],
            note='native same-name type; cross-dataset relation names '
                 f'another counterpart ({d_target}) — identity kept for '
                 'the query, counter-evidence shown')
    from .type_resolver import (
        STATUS_BRIDGED as TR_BRIDGED,
    )
    if (res.status == TR_BRIDGED and res.equivalence_key
            and str(res.equivalence_key) != token):
        # A unique derivation-derived answer: derivation evidence wins for
        # queries too (no curated relation exists to annotate the identity).
        return QueryResolution(
            token=token, dataset=ds, role=role, status=STATUS_MAPPED,
            method='mapped_type', target_types=[str(res.equivalence_key)],
            evidence_chain=str(getattr(res, 'kind', '') or 'bridged'),
            confidence=CONFIDENCE[STATUS_MAPPED],
            note='resolved via unique derivation bridge')
    note = ('native same-name type; no curated relation (echo verified only '
            'against the dataset type table)')
    targets = [str(t) for t in (res.target_types or ())]
    if len(targets) > 1 and token in targets:
        note += f'; identity among {len(targets)} derivation candidates'
    return QueryResolution(
        token=token, dataset=ds, role=role,
        status=STATUS_SAME_NAME_IDENTITY, method='same_name_identity',
        target_types=[token], matched_column='type',
        evidence=EVIDENCE_NONE,
        confidence=CONFIDENCE[STATUS_SAME_NAME_IDENTITY],
        note=note)


def _curated_decision(mapper, token, src, ds) -> Dict[str, Any]:
    """Scoped policy decision for the identity annotation.

    The mapper's scoped decision (default, derivation chains included)
    refuses to promote a bare name echo to ``mapped`` — so a ``mapped``
    answer here means real tabular/label evidence confirms the identity,
    and its absence means the identity rests on the name alone.
    """
    try:
        return mapper.get_mapping_decision(token, src, ds) or {}
    except Exception:
        return {}


def _member_targets(mapper, member, hit_ds, ds) -> List[str]:
    """Map one taxonomy member type into the target dataset (Route A).

    Only licensed targets count: a member with no counterpart (unmapped)
    or an evidence-only one-of-N relation contributes nothing — the union
    must stay a defensible query, and misses are reported in the note.
    """
    from .type_resolver import (
        STATUS_BRIDGED, STATUS_MAPPED, STATUS_VALID_SPLIT,
        resolve_valid_targets,
    )
    try:
        res = resolve_valid_targets(mapper, member, hit_ds, ds)
    except Exception:
        return []
    # RES-10 as amended 2026-09-27: a UNIQUE equivalence, a licensed
    # split, OR a multi-target mapped fan-out (candidate discovery
    # licenses it — the decision surface reads the same edge
    # valid_split_evidence) contributes its branches; single-end bridged
    # partners map to their one end.  Only evidence-only/unmapped unions
    # contribute nothing.
    #
    # Round-17 (user-approved): the contributed branches are the DECISION-
    # ADOPTED ones. resolve_valid_targets unions declined derivation-bridge
    # ends into a VALID_SPLIT (type_resolver.py:578), which admitted
    # disclosure-only ends into the delegate target set (real case:
    # s-CPDN3C -> banc admitted CB3767 x2 via a branch_not_adopted
    # disclosure — BANC resolved 202 where the ratified panel claim is
    # 198+2=200). Intersecting with the scoped decision's adopted targets
    # aligns the delegate with the claim; MCNS's 40/219 union is fully
    # adopted (verified) and unaffected. Declined ends remain visible as
    # panel suspects/disclosure rows.
    if res.status in (STATUS_VALID_SPLIT, STATUS_MAPPED) and len(
            res.target_types or ()) > 1:
        targets = [str(t) for t in (res.target_types or ())]
        try:
            decision = mapper.get_mapping_decision(member, hit_ds, ds)
            adopted = {str(t) for t in (
                decision.get('target_types') or [])}
            if adopted:
                targets = [t for t in targets if t in adopted]
        except Exception:
            pass
        return targets
    if res.status == STATUS_MAPPED and res.equivalence_key is not None:
        return [str(res.equivalence_key)]
    if res.status == STATUS_BRIDGED:
        if len(res.target_types or ()) == 1:
            return [str(res.target_types[0])]
        return [str(t) for t in (res.target_types or ())]
    return []


try:
    from utils.label_utils import UntypedLabelPolicy
except ImportError:  # pragma: no cover - direct src/ execution
    from src.utils.label_utils import UntypedLabelPolicy


class DatasetTaxonomyResolver:
    """Expand taxonomy-column values to member type names per dataset.

    Reads the same local neuron tables the type mapper indexes and scans
    the taxonomy columns the viewer search covers (``viewer_search_columns``
    minus the identity columns bodyId/type/instance and the cross-dataset
    type columns flywireType/hemibrainType/mancType).  A value matches a
    cell EQUAL to the token; the member types are the distinct ``type``
    values of the matching rows.

    This is the native half of taxonomy query resolution (plan §2.2): the
    token ``circadian_clock`` is a FAFB ``cell_type`` value and expands to
    its 21 member types here, while datasets without that value answer
    ``None`` and the resolver bridges them via member mapping (Route A).
    """

    _TABLES = {
        'male-cns:v1.0': ('male-cns_v1_0',
                          'male-cns_v1_0_allneurons_neuron_df.csv'),
        'male-cns:v0.9': ('male-cns_v0_9',
                          'male-cns_v0_9_allneurons_neuron_df.csv'),
        'flywire_FAFB_v783': ('flywire_FAFB_v783',
                              'flywire_FAFB_v783_allneurons_neuron_df.csv'),
        'banc_v626': ('banc_v626', 'banc_v626_allneurons_neuron_df.csv'),
        'banc_v888': ('banc_v888', 'banc_v888_allneurons_neuron_df.csv'),
    }
    _EXCLUDED = {'bodyid', 'type', 'instance',
                 'flywiretype', 'hemibraintype', 'manctype'}
    # Shared untyped sentinels (UntypedLabelPolicy) + '' for the
    # membership test; plan-untyped-labels-and-drop-hardening 4.1.
    _UNTYPED = UntypedLabelPolicy.SENTINELS | {''}

    def __init__(self, mapper=None, workspace_path=None,
                 include_cross_dataset_type_columns: bool = False):
        self._mapper = mapper
        self._workspace = workspace_path
        # The cross-dataset flow keeps flywireType/hemibrainType/mancType out
        # of the scan (those names are bridged by the mapper, not matched
        # natively); an intra-dataset comparer wants them, matching the
        # connectivity profiler's get_types_for_label column list.
        self._include_cross = bool(include_cross_dataset_type_columns)
        self._tables: Dict[str, Any] = {}
        self._frames: Dict[str, Any] = {}
        self._cache: Dict[str, Dict[str, Optional[List[str]]]] = {}

    def _root(self):
        if self._workspace:
            return Path(self._workspace)
        root = getattr(self._mapper, '_workspace_path', None) \
            if self._mapper is not None else None
        return Path(root) if root else None

    def _excluded(self) -> set:
        if self._include_cross:
            return self._EXCLUDED - {'flywiretype', 'hemibraintype',
                                     'manctype'}
        return self._EXCLUDED

    def _table(self, dataset):
        """(path, scan_columns) for one dataset, or None when unavailable."""
        if dataset in self._tables:
            return self._tables[dataset]
        entry = self._TABLES.get(str(dataset))
        root = self._root()
        result = None
        if entry and root is not None:
            path = root / 'datasets' / entry[0] / entry[1]
            if path.exists():
                result = self._scan_table(path)
        if result is None and root is not None:
            # Generic fallback: the same allneurons table
            # morphology._load_neuron_type_map reads, named by the dataset's
            # canonical folder — so any local dataset resolves, not just the
            # five the cross-dataset flow tabulates.
            result = self._generic_table(str(dataset), root)
        self._tables[dataset] = result
        return result

    def _generic_table(self, dataset: str, root: Path):
        """(path, scan_columns) from the canonical allneurons table, or None."""
        try:
            from utils.naming_utils import canonical_dataset_name
        except ImportError:  # pragma: no cover - src laid bare on sys.path
            from naming_utils import canonical_dataset_name
        folder = (canonical_dataset_name(dataset)
                  .replace(':', '_').replace('.', '_'))
        dataset_dir = root / 'datasets' / folder
        for name in (f'{folder}_allneurons_neuron_df.parquet',
                     f'{folder}_allneurons_neuron_df.csv'):
            path = dataset_dir / name
            if path.exists():
                return self._scan_table(path)
        return None

    def _scan_table(self, path: Path):
        """(path, scan_columns) for one existing table, or None."""
        try:
            from neuron_index_builder import viewer_search_columns
            header = self._table_header(path)
            if 'type' not in header:
                return None
            scan = [c for c in viewer_search_columns(header)
                    if c.strip().lower() not in self._excluded()]
            return (path, scan) if scan else None
        except Exception:
            return None

    @staticmethod
    def _table_header(path: Path) -> List[str]:
        if path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema_arrow.names)
        return list(pd.read_csv(path, nrows=0).columns)

    def _frame(self, dataset):
        """(type column frame, scan columns) for one dataset, or None.

        The reduced frame (type + scanned taxonomy columns only) is loaded
        once per dataset and reused across tokens.
        """
        if dataset in self._frames:
            return self._frames[dataset]
        table = self._table(dataset)
        result = None
        if table is not None:
            path, scan = table
            if scan:
                try:
                    keep = sorted(set(scan) | {'type'})
                    if path.suffix.lower() == '.parquet':
                        frame = pd.read_parquet(path, columns=keep)
                    else:
                        frame = pd.read_csv(path, usecols=keep,
                                            low_memory=False)
                    result = (frame, scan)
                except Exception:
                    result = None
        self._frames[dataset] = result
        return result

    def resolve(self, token, dataset) -> Optional[List[str]]:
        """Member types for one taxonomy value in one dataset (None = no hit)."""
        token = str(token or '').strip()
        dataset = str(dataset or '')
        if not token:
            return None
        if token.lower() in self._UNTYPED:
            # A NaN-cell stringified must never match as a taxonomy value.
            return None
        loaded = self._frame(dataset)
        if loaded is None:
            return None
        frame, scan = loaded
        ds_cache = self._cache.setdefault(dataset, {})
        if token in ds_cache:
            return ds_cache[token]
        result: Optional[List[str]] = None
        mask = None
        for col in scan:
            values = frame[col].astype(str).str.strip()
            m = values.eq(token)
            mask = m if mask is None else (mask | m)
        if mask is not None and bool(mask.any()):
            types = (frame.loc[mask, 'type'].dropna()
                     .astype(str).str.strip())
            types = sorted(
                t for t in types.unique()
                if t and t.lower() not in self._UNTYPED)
            result = types or None
        ds_cache[token] = result
        return result


def _same_native_namespace(mapper, src, ds) -> bool:
    """True when two datasets share one mapping namespace (e.g. male-cns
    v0.9/v1.0), where an identical name is a genuine native identity."""
    if mapper is None:
        return str(src) == str(ds)
    try:
        return (mapper._get_type_mapping_key(src)
                == mapper._get_type_mapping_key(ds))
    except Exception:
        return str(src) == str(ds)


def _same_name(token, dataset, role, note) -> QueryResolution:
    return QueryResolution(
        token=token, dataset=dataset, role=role,
        status=STATUS_SAME_NAME, method='same_name',
        target_types=[token], confidence=CONFIDENCE[STATUS_SAME_NAME],
        matched_column='type', note=note)
