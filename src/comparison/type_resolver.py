"""Validity-aware cross-dataset type resolution — THE shared backend.

One canonical resolution contract for every cross-dataset consumer: the
Type Mapping panel, the neuron-index viewer, homolog finding, and
connectivity-profile comparison all read their mapping decisions from this
module.  The comparison layer must never import a UI module, and no
consumer may invent its own interpretation of ``None``, an empty target
list, or a split.

The policy source is ``CrossDatasetTypeMapper.get_mapping_decision``; this
module wraps it in

* :class:`MapperSnapshot` — one immutable load-state record per analysis
  run (source, version token, load error) plus a per-run decision cache;
* :func:`resolve_valid_targets` — the validity-aware target resolution
  (union of curated/alias mapping, crosswalk splits, and derivation-bridge
  ends) that backs ``ui.neuron_index.mapped_type_targets``;
* :func:`expand_profile_types` — status-aware profile canonicalization for
  scoring, which knows WHICH mapping status produced each canonical key;
* :func:`resolve_one_target` — the compatibility adapter for the legacy
  single-target APIs (``get_mapped_type`` /
  ``resolve_type_across_datasets``).

Status vocabulary (the one canonical set)::

    mapped | bridged | valid_split_evidence | evidence_only |
    conflict | unmapped | mapper_unavailable

Analysis policy (see ``equivalence_key`` / ``expansion_targets``):

* ``mapped`` / ``bridged`` with exactly one target license a unique
  equivalence key.  ``bridged`` is bridge-DERIVED evidence — it keeps its
  own status and provenance and is never silently rewritten to curated
  ``mapped``.
* ``valid_split_evidence`` licenses expansion to ALL its targets (never a
  single arbitrary branch).
* ``evidence_only`` is display/diagnostics only; it contributes no
  automatic equivalence or expansion unless a caller explicitly opts in.
* ``conflict`` fails closed: no automatic target, no raw same-name
  equivalence.
* ``unmapped`` keeps the raw name as an explicitly-counted long-tail
  fallback in profile/candidate expansion.  Unlike a conflict, no evidence
  says the raw names disagree; the fallback is always visible in the
  expansion record and in result metadata.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .cross_dataset_type_mapper import CrossDatasetTypeMapper, get_type_mapper

__all__ = [
    'STATUS_MAPPED', 'STATUS_BRIDGED', 'STATUS_VALID_SPLIT',
    'STATUS_EVIDENCE_ONLY', 'STATUS_CONFLICT', 'STATUS_UNMAPPED',
    'STATUS_MAPPER_UNAVAILABLE', 'STATUS_CLAIMED',
    'MAPPING_POLICY_VERSION',
    'MapperSnapshot', 'TypeResolution', 'ProfileExpansion', 'MergeKey',
    'get_mapper_snapshot', 'resolve_valid_targets', 'expand_profile_types',
    'resolve_one_target', 'canonical_merge_key', 'same_namespace',
    'resolve_flow_status', 'equivalence_key', 'expansion_targets',
    'auto_mapping_metadata',
]

# --- canonical status vocabulary -----------------------------------------
STATUS_MAPPED = 'mapped'
STATUS_BRIDGED = 'bridged'
STATUS_VALID_SPLIT = 'valid_split_evidence'
STATUS_EVIDENCE_ONLY = 'evidence_only'
STATUS_CONFLICT = 'conflict'
STATUS_UNMAPPED = 'unmapped'
STATUS_MAPPER_UNAVAILABLE = 'mapper_unavailable'
#: The curated relation names a target the target dataset does not actually
#: carry (§stale claims 2026-09-12): the claim stays visible as evidence on
#: the resolution, but no equivalence is licensed and expansion follows it
#: nowhere.
STATUS_CLAIMED = 'claimed'

#: Version of the equivalence/expansion policy implemented here.  Recorded
#: in result metadata so saved analyses can be audited against the policy
#: that produced them.
MAPPING_POLICY_VERSION = '1'

#: Statuses that license a unique automatic equivalence key.
_UNIQUE_EQUIVALENCE_STATUSES = frozenset({STATUS_MAPPED, STATUS_BRIDGED})

#: Most-restrictive-first order used to merge statuses when several raw
#: types contribute to one canonical key.
_STATUS_SEVERITY = (
    STATUS_CONFLICT,
    # claims license nothing (a target the dataset does not carry must not
    # be absorbed by a mapped label merging over it)
    STATUS_CLAIMED,
    STATUS_EVIDENCE_ONLY,
    STATUS_VALID_SPLIT,
    STATUS_UNMAPPED,
    STATUS_BRIDGED,
    STATUS_MAPPED,
)

CANONICAL_NAMESPACE = 'male-cns:v1.0'


def _severity(status: str) -> int:
    try:
        return _STATUS_SEVERITY.index(status)
    except ValueError:
        return len(_STATUS_SEVERITY)


def _merge_status(a: str, b: str) -> str:
    return a if _severity(a) <= _severity(b) else b


# --- mapper snapshot ------------------------------------------------------

class MapperSnapshot:
    """One mapper + load-state record shared by a whole analysis run.

    Obtained once at the start of an operation and passed through candidate
    discovery, query mapping, scoring, and result metadata, so every step
    observes the same load state and reuses one decision cache.  The
    snapshot does not copy the mapper's pandas indexes; it only records
    identity, provenance, and caches derived lookups.
    """

    def __init__(self, mapper: Optional[CrossDatasetTypeMapper],
                 requested: bool = True):
        self.mapper = mapper
        self.requested = requested
        self._decision_cache: Dict[Tuple[str, str, str, bool], Dict[str, Any]] = {}
        self.refresh()

    def refresh(self) -> None:
        mapper = self.mapper
        self.loaded = bool(mapper is not None
                           and getattr(mapper, '_loaded', False))
        self.source_path: Optional[str] = None
        self.source: Optional[str] = None
        self.version: Optional[str] = None
        self.load_error: Optional[str] = getattr(
            mapper, 'last_load_error', None) if mapper else None
        if mapper is not None:
            path = getattr(mapper, '_neuron_df_path', None)
            if path:
                self.source_path = str(path)
                # Record the dataset-relative portion so metadata stays
                # portable across machines.
                parts = Path(path).parts
                self.source = '/'.join(parts[-2:]) if len(parts) >= 2 else str(path)
                try:
                    st = os.stat(path)
                    self.version = f'{st.st_mtime_ns}:{st.st_size}'
                except OSError:
                    self.version = None
        if self.loaded:
            self.load_error = None

    @property
    def active(self) -> bool:
        """Requested auto mapping AND a loaded mapper."""
        return self.requested and self.loaded

    def decision(self, source_type: str, source_dataset: str,
                 target_dataset: str, include_bridges: bool = False,
                 ) -> Dict[str, Any]:
        """Cached :meth:`get_mapping_decision` for the run."""
        mapper = self.mapper
        if mapper is None:
            return {'status': STATUS_MAPPER_UNAVAILABLE, 'target_type': None,
                    'target_types': [], 'conflicts': [], 'relationship': None}
        key = (str(source_type), str(source_dataset), str(target_dataset),
               bool(include_bridges))
        cached = self._decision_cache.get(key)
        if cached is None:
            try:
                cached = mapper.get_mapping_decision(
                    source_type, source_dataset, target_dataset,
                    include_bridges=include_bridges)
            except Exception as exc:  # mapper API failure != unmapped data
                cached = {'status': STATUS_MAPPER_UNAVAILABLE,
                          'target_type': None, 'target_types': [],
                          'conflicts': [], 'relationship': None,
                          'error': str(exc)}
            self._decision_cache[key] = cached
        return cached

    def metadata(self) -> Dict[str, Any]:
        return {
            'requested': self.requested,
            'active': self.active,
            'loaded': self.loaded,
            'source': self.source,
            'source_path': self.source_path,
            'version': self.version,
            'load_error': self.load_error,
            'policy_version': MAPPING_POLICY_VERSION,
        }


def get_mapper_snapshot(
    mapper: Optional[CrossDatasetTypeMapper] = None,
    *,
    workspace_path: Optional[str] = None,
    force_reload: bool = False,
    requested: bool = True,
) -> MapperSnapshot:
    """Snapshot the process-wide mapper (or the given one) for one run."""
    if mapper is None:
        try:
            mapper = get_type_mapper(workspace_path=workspace_path,
                                     force_reload=force_reload)
        except Exception:
            mapper = None
    return MapperSnapshot(mapper, requested=requested)


# --- target resolution ----------------------------------------------------

@dataclass(frozen=True)
class TypeResolution:
    """Immutable, status/evidence-aware resolution of one type-level edge.

    ``status`` uses the canonical vocabulary of this module.  ``kind``
    carries the panel/viewer display vocabulary so the UI adapter can
    preserve its response shape.  ``target_types`` is ordered and unique.
    """

    status: str
    source_type: str
    source_dataset: str
    target_dataset: str
    target_types: Tuple[str, ...] = ()
    kind: str = 'unmapped'
    equivalence_key: Optional[str] = None
    evidence: Tuple[Tuple[Dict[str, str], ...], ...] = ()
    conflicts: Tuple[Dict[str, Any], ...] = ()
    mapper_source: Optional[str] = None
    mapper_version: Optional[str] = None
    mapper_loaded: bool = False
    fallback_used: bool = False
    reason: str = ''
    secondary_targets: Tuple[str, ...] = ()
    # True when the unique answer comes from the curated decision itself
    # (Phase 1b): a tabular/label relation confirms this exact pairing, so
    # consumers must not treat it as a tier-6 name echo.
    curated_identity: bool = False
    # Same-name-first selection (plan-samename-first-fanout-resolution):
    # True when this resolution IS the mapper's rule-driven selection of a
    # fan-out's own-name candidate.  ``suspect_rivals`` lists the demoted
    # rival candidates (disclosure only — never defaults, never targets).
    # Carried so display surfaces can distinguish a deliberate selection
    # from a bare same-name echo WITHOUT re-deriving the decision; the
    # mapper owns detection, consumers only display it.
    suspects: bool = False
    suspect_rivals: Tuple[str, ...] = ()

    @property
    def direction(self) -> str:
        return f'{self.source_dataset}->{self.target_dataset}'

    @property
    def expansion_key(self) -> Optional[str]:
        """Alias of :attr:`equivalence_key` for expansion contexts."""
        return self.equivalence_key


def equivalence_key(resolution: TypeResolution) -> Optional[str]:
    """The single automatic equivalence key for this edge, if licensed.

    Only a unique target under a ``mapped``/``bridged`` status licenses
    one.  Splits, conflicts, evidence-only relations, and unmapped types
    return ``None`` — callers must not fall back to the raw name for a
    conflict merely because the names are equal.
    """
    if (resolution.status in _UNIQUE_EQUIVALENCE_STATUSES
            and len(resolution.target_types) == 1):
        return resolution.target_types[0]
    return None


def expansion_targets(resolution: TypeResolution) -> Tuple[str, ...]:
    """Target-local names this edge licenses for candidate/profile expansion.

    * mapped/bridged/valid splits: all their targets.
    * evidence_only: none (display/diagnostics only by default).
    * claimed: none — a crosswalk claim the target namespace cannot
      fulfil must not seed candidate/profile expansion (§stale claims).
    * conflict / mapper unavailable: none — fail closed.
    * unmapped: the raw name itself, the explicitly-counted long-tail
      fallback (no evidence says the raw names disagree).
    """
    if resolution.status == STATUS_MAPPER_UNAVAILABLE:
        return ()
    if resolution.status == STATUS_CONFLICT:
        return ()
    if resolution.status == STATUS_EVIDENCE_ONLY:
        return ()
    if resolution.status == STATUS_CLAIMED:
        return ()
    if resolution.status == STATUS_UNMAPPED:
        return (resolution.source_type,)
    return resolution.target_types


def resolve_valid_targets(
    mapper: Optional[CrossDatasetTypeMapper],
    source_type: str,
    source_dataset: Optional[str],
    target_dataset: str,
    *,
    snapshot: Optional[MapperSnapshot] = None,
    alias_cache: Optional[Dict[Any, Any]] = None,
    bridge_cache: Optional[Dict[Tuple[str, str, str],
                                list]] = None,
) -> TypeResolution:
    """Resolve one type-level edge with full validity policy.

    This is the shared lower-level implementation behind the panel/viewer
    (``mapped_type_targets``), homolog finding, and profile comparison.
    It unions the curated/alias resolution (``get_alias_candidates``) with
    the derivation-bridge ends (``get_type_bridges``) and scopes the whole
    result through ``get_mapping_decision`` so a conflict can never be
    promoted to a mapped target by a same-name bridge chain.

    ``source_dataset=None`` auto-detects the source namespace from the
    type name (the legacy ``resolve_type_across_datasets`` behavior); when
    detection fails the result is ``unmapped`` with the reason attached.
    """
    snap = snapshot if snapshot is not None else MapperSnapshot(mapper)
    meta = snap.metadata()

    raw = str(source_type or '').strip()
    if not raw:
        return TypeResolution(
            status=STATUS_UNMAPPED, kind='unmapped',
            source_type='', source_dataset=str(source_dataset or ''),
            target_dataset=str(target_dataset),
            mapper_source=meta['source'], mapper_version=meta['version'],
            mapper_loaded=meta['loaded'],
            reason='empty source type')

    if mapper is None or not snap.loaded:
        return TypeResolution(
            status=STATUS_MAPPER_UNAVAILABLE, kind='mapper unavailable',
            source_type=raw, source_dataset=str(source_dataset or ''),
            target_dataset=str(target_dataset),
            mapper_source=meta['source'], mapper_version=meta['version'],
            mapper_loaded=meta['loaded'],
            reason='type mapper is not loaded')

    if not source_dataset:
        try:
            source_dataset = mapper._detect_type_source(raw)
        except Exception:
            source_dataset = None
        if not source_dataset:
            return TypeResolution(
                status=STATUS_UNMAPPED, kind='unmapped',
                source_type=raw, source_dataset='',
                target_dataset=str(target_dataset),
                mapper_source=meta['source'], mapper_version=meta['version'],
                mapper_loaded=meta['loaded'],
                reason='source namespace unknown for type')

    base = dict(
        source_type=raw,
        source_dataset=str(source_dataset),
        target_dataset=str(target_dataset),
        mapper_source=meta['source'],
        mapper_version=meta['version'],
        mapper_loaded=meta['loaded'],
    )

    # Same namespace (including the shared male-cns v0.9/v1.0 type
    # namespace via mapping-key equality): the raw name is native.
    try:
        same_ns = (mapper._get_type_mapping_key(source_dataset)
                   == mapper._get_type_mapping_key(target_dataset))
    except Exception:
        same_ns = str(source_dataset) == str(target_dataset)
    if same_ns:
        return TypeResolution(
            status=STATUS_MAPPED, kind='same name',
            target_types=(raw,), equivalence_key=raw,
            reason='same namespace', **base)

    # 1) Curated decision WITHOUT bridge derivation: conflict must win over
    #    any same-name bridge chain (BANC CB1011 class).
    decision = snap.decision(raw, source_dataset, target_dataset,
                             include_bridges=False)
    # Same-name-first disclosure (plan-samename-first-fanout-resolution):
    # carry the mapper's own verdict on this resolution so display surfaces
    # can tell a deliberate SELECTION from a bare same-name echo without
    # re-deriving the decision.  ``suspects`` is True only when the rule
    # fired (status mapped); a held pair keeps its ordinary status but still
    # carries ``suspect_rivals``, which is what lets the UI explain WHY it
    # was held.  Disclosure only — never a target, never a default.
    base['suspects'] = bool(decision.get('suspects'))
    base['suspect_rivals'] = tuple(
        decision.get('same_name_first', {}).get('rivals') or ())
    if decision['status'] == STATUS_MAPPER_UNAVAILABLE:
        return TypeResolution(
            status=STATUS_MAPPER_UNAVAILABLE, kind='mapper unavailable',
            reason=decision.get('error', 'mapper decision failed'), **base)
    conflicts = tuple(decision.get('conflicts') or ())
    if decision['status'] == STATUS_CONFLICT:
        return TypeResolution(
            status=STATUS_CONFLICT, kind='conflict',
            conflicts=conflicts,
            reason='unresolved mapping conflict; no automatic target',
            **base)

    # 1b) A unique curated relation IS the equivalence answer (Phase 1b of
    #     plan-cross-dataset-query-resolution-samename-taxonomy).  The noisy
    #     derivation union (alt-column bridge ends) may widen the DISPLAY
    #     via ``secondary_targets`` but never replaces a unique curated
    #     answer, and a same-name alias candidate never overrides a curated
    #     rename.  Before this branch, FAFB ``additional_type(s)`` bridge
    #     ends demoted a clean 1-to-1 (MCNS ``L2`` -> FAFB ``L2``) to a
    #     9-candidate ``one of N`` and a same-name alias confirmed ``Dm8a``
    #     over the curated rename to ``yDm8``.
    d_target = decision.get('target_type')
    if (decision['status'] == STATUS_MAPPED and d_target
            and len(decision.get('target_types') or ()) == 1):
        d_target = str(d_target)
        bridge_key = (raw, str(source_dataset), str(target_dataset))
        if bridge_cache is not None and bridge_key in bridge_cache:
            chains = bridge_cache[bridge_key]
        else:
            try:
                chains = mapper.get_type_bridges(
                    raw, source_dataset, target_dataset, max_bridges=8)
            except Exception:
                chains = []
            if bridge_cache is not None:
                bridge_cache[bridge_key] = chains
        chains = chains or []
        evidence = tuple(
            tuple(dict(step) for step in chain) for chain in chains)
        ends = {str(c[-1]['value']) for c in chains if c and c[-1].get('value')}
        kind = 'same name' if d_target == raw else 'renamed'
        # §stale claims: a unique curated answer naming a type the target
        # dataset does not carry is evidence, not an equivalence.  When the
        # raw name is native in the target namespace the realized same-name
        # equivalence wins (query tier §2.1) and the unfulfilled claim is
        # kept as the reason; otherwise the claim is reported without
        # licensing expansion into an empty namespace.
        if _is_stale_claim(mapper, raw, source_dataset, target_dataset,
                           d_target):
            if (d_target != raw
                    and _target_has_native(mapper, target_dataset, raw)):
                return TypeResolution(
                    status=STATUS_MAPPED, kind='same name',
                    target_types=(raw,),
                    equivalence_key=raw,
                    secondary_targets=tuple(sorted(ends - {raw})),
                    evidence=evidence,
                    # RES-8: the curated relation names a DIFFERENT type
                    # (that is why the claim is stale) — it cannot confirm
                    # this identity.  A verified-native identity, not a
                    # curated confirmation.
                    curated_identity=False,
                    reason=(f'curated claim {d_target!r} is absent from the '
                            'target dataset; native same-name realized'),
                    **base)
            return TypeResolution(
                status=STATUS_CLAIMED, kind=kind,
                target_types=(d_target,),
                secondary_targets=tuple(sorted(ends - {d_target})),
                evidence=evidence,
                curated_identity=True,
                reason='crosswalk claim; target dataset carries no such type',
                **base)
        return TypeResolution(
            status=STATUS_MAPPED, kind=kind,
            target_types=(d_target,),
            equivalence_key=d_target,
            secondary_targets=tuple(sorted(ends - {d_target})),
            evidence=evidence,
            curated_identity=True,
            reason=decision.get('relationship') or 'curated unique relation',
            **base)

    # 2) Alias/candidates half (kind + split/evidence statuses).
    alias_key = (raw, str(source_dataset), str(target_dataset))
    if alias_cache is not None and alias_key in alias_cache:
        ann = alias_cache[alias_key]
    else:
        ann = _alias_annotation(mapper, raw, source_dataset, target_dataset)
        if alias_cache is not None:
            alias_cache[alias_key] = ann

    # 3) Bridge ends half.
    bridge_key = (raw, str(source_dataset), str(target_dataset))
    if bridge_cache is not None and bridge_key in bridge_cache:
        chains = bridge_cache[bridge_key]
    else:
        try:
            chains = mapper.get_type_bridges(
                raw, source_dataset, target_dataset, max_bridges=8)
        except Exception:
            chains = []
        if bridge_cache is not None:
            bridge_cache[bridge_key] = chains
    chains = chains or []
    evidence = tuple(tuple(dict(step) for step in chain) for chain in chains)
    ends = {str(c[-1]['value']) for c in chains if c and c[-1].get('value')}

    # 4) Union, preserving mapped_type_targets semantics exactly.
    if not ends:
        if ann:
            kind = ann['kind']
            targets = tuple(ann['targets'])
            decision_status = decision.get('status')
            status = ann.get('status') or (
                decision_status if decision_status in {
                    STATUS_EVIDENCE_ONLY, STATUS_VALID_SPLIT}
                else STATUS_MAPPED)
            eq = targets[0] if (
                status in _UNIQUE_EQUIVALENCE_STATUSES
                and len(targets) == 1) else None
            # §stale claims: a single alias candidate whose name the target
            # dataset does not carry is reported as a claim, not a mapping.
            if (status == STATUS_MAPPED and len(targets) == 1
                    and _is_stale_claim(mapper, raw, source_dataset,
                                        target_dataset, targets[0])):
                status = STATUS_CLAIMED
                eq = None
            return TypeResolution(
                status=status, kind=kind, target_types=targets,
                equivalence_key=eq, reason='alias/curated resolution',
                **base)
        # No alias annotation: fall back to the scoped decision status.
        status = decision['status']
        targets = tuple(decision.get('target_types') or ())
        single = decision.get('target_type')
        kind = {
            STATUS_MAPPED: 'renamed',
            STATUS_VALID_SPLIT: 'splits into',
            STATUS_EVIDENCE_ONLY: 'one of N',
        }.get(status, status)
        eq = single if status == STATUS_MAPPED and single else None
        return TypeResolution(
            status=status, kind=kind, target_types=targets,
            equivalence_key=eq,
            reason=decision.get('relationship')
            or 'no scoped relation available',
            **base)

    targets = set(ends)
    if ann:
        targets.update(ann.get('targets') or ())
    # The decision's own targets belong in the union when the decision
    # itself licensed them as evidence: a valid-split's branches or an
    # evidence-only convergence's members are exactly what the bridge ends
    # should be read beside (audit RES-1: the union used to drop them, so
    # an evidence-only relation with one coinciding bridge end collapsed
    # into a unique "mapped" equivalence — a refusal resurrected).
    if decision.get('status') in (STATUS_VALID_SPLIT, STATUS_EVIDENCE_ONLY):
        targets.update(str(t) for t in (decision.get('target_types') or ()))
    targets = tuple(sorted(targets))
    if len(targets) == 1 and ann:
        kind = ann['kind']
        status = ann.get('status') or STATUS_MAPPED
        # §stale claims applies here too: a single alias candidate the
        # target dataset does not carry is a claim, not a mapping.
        if (status == STATUS_MAPPED
                and _is_stale_claim(mapper, raw, source_dataset,
                                    target_dataset, targets[0])):
            status = STATUS_CLAIMED
        return TypeResolution(
            status=status, kind=kind, target_types=targets,
            equivalence_key=targets[0]
            if status in _UNIQUE_EQUIVALENCE_STATUSES else None,
            reason='alias/curated resolution', **base)
    split_evidence = bool(
        ann and ann.get('kind') == 'splits into') or bool(
            decision.get('status') == STATUS_VALID_SPLIT)
    evidence_only = bool(decision.get('status') == STATUS_EVIDENCE_ONLY)
    if split_evidence and len(targets) > 1:
        kind, status = 'splits into', STATUS_VALID_SPLIT
    elif evidence_only:
        # the decision refused a canonical target; bridge ends widen the
        # review set but never promote it to a licensed mapping — at any
        # arity (one coinciding end included, audit RES-1)
        kind, status = 'one of N', STATUS_EVIDENCE_ONLY
    elif len(targets) > 1:
        kind, status = 'one of N', STATUS_MAPPED
    else:
        kind, status = 'bridged', STATUS_BRIDGED
    return TypeResolution(
        status=status, kind=kind, target_types=targets,
        equivalence_key=targets[0]
        if status in _UNIQUE_EQUIVALENCE_STATUSES and len(targets) == 1
        else None,
        evidence=evidence,
        reason='bridge-derived resolution', **base)


def _is_stale_claim(mapper: Optional[CrossDatasetTypeMapper],
                    source_type: str, source_dataset: str,
                    target_dataset: str, target_type: str) -> bool:
    """Whether the curated claim's target is absent from the target dataset.

    §stale claims (2026-09-12): consults the mapper's stale-claim registry
    (crosswalk names that passed through resolution unchanged and do not
    exist in the target namespace).  Mappers without the registry — older
    snapshots, custom fakes — fail open to ``False``.
    """
    checker = getattr(mapper, 'is_stale_claim', None)
    if not callable(checker):
        return False
    try:
        return bool(checker(source_type, source_dataset, target_dataset,
                            target_type))
    except Exception:
        return False


def _target_has_native(mapper: Optional[CrossDatasetTypeMapper],
                       dataset: str, name: str) -> bool:
    """True only when the mapper's namespace ground truth confirms ``name``.

    An unknown namespace (``type_exists`` returning ``None``) does not
    license the native same-name fallback — without ground truth a stale
    claim must not be quietly re-pointed at an unverified identity.
    """
    checker = getattr(mapper, 'type_exists', None)
    if not callable(checker):
        return False
    try:
        return checker(dataset, name) is True
    except Exception:
        return False


def _alias_annotation(mapper: CrossDatasetTypeMapper, raw: str,
                      source_dataset: str, target_dataset: str,
                      ) -> Optional[Dict[str, Any]]:
    """Alias/curated half of the resolution (mapped_type_targets contract)."""
    try:
        res = mapper.get_alias_candidates(
            raw, [target_dataset], source_dataset=source_dataset)
        info = res.get(target_dataset) or {}
        candidates = info.get('candidates', [])
        if info.get('outcome') != 'matched' or not candidates:
            return None
        if any(c['kind'] == 'splits into' for c in candidates):
            return {
                'kind': 'splits into',
                'targets': sorted({
                    c['name'] for c in candidates
                    if c['kind'] == 'splits into'}),
                'status': STATUS_VALID_SPLIT,
            }
        if any(c['kind'] == 'one of N' for c in candidates):
            return {
                'kind': 'one of N',
                'targets': sorted(
                    c['name'] for c in candidates if c['kind'] == 'one of N'),
            }
        cand = candidates[0]
        return {'kind': cand['kind'], 'targets': [cand['name']]}
    except Exception:
        return None


# --- compatibility adapter for the legacy one-target APIs -----------------

def resolve_one_target(
    mapper: Optional[CrossDatasetTypeMapper],
    source_type: str,
    source_dataset: str,
    target_dataset: str,
    *,
    snapshot: Optional[MapperSnapshot] = None,
) -> Optional[str]:
    """Unique valid target for one edge, else ``None``.

    Compatibility adapter for legacy single-target callers
    (``get_mapped_type`` / ``resolve_type_across_datasets``): only an
    unambiguous one-target case under a ``mapped``/``bridged`` status
    returns a name.  Splits, evidence-only relations, conflicts, unmapped
    types, and an unavailable mapper return ``None`` — never an arbitrary
    branch of a split.
    """
    res = resolve_valid_targets(mapper, source_type, source_dataset,
                                target_dataset, snapshot=snapshot)
    if res.status in _UNIQUE_EQUIVALENCE_STATUSES:
        return res.equivalence_key
    return None


# --- merge-oriented canonicalization --------------------------------------

def same_namespace(mapper: Optional[CrossDatasetTypeMapper],
                   ds_a: str, ds_b: str) -> bool:
    """True when two datasets share one mapping namespace (mapping-key
    equality, e.g. male-cns v0.9/v1.0), with a plain string-equality
    fallback when the mapper cannot answer.  Replaces direct
    ``_get_type_mapping_key`` access by flow builders and reports."""
    if mapper is None:
        return str(ds_a) == str(ds_b)
    try:
        return (mapper._get_type_mapping_key(ds_a)
                == mapper._get_type_mapping_key(ds_b))
    except Exception:
        return str(ds_a) == str(ds_b)


def resolve_flow_status(
    mapper: Optional[CrossDatasetTypeMapper],
    type_name: str,
    source_dataset: str,
    target_dataset: str,
    *,
    bridge_end_count: int = 0,
    snapshot: Optional[MapperSnapshot] = None,
    cache: Optional[Dict] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The flow-builder status policy, defined once (shared resolver).

    Mirrors the scoped decision the mapping flows consume: a conflict
    yields ``('conflict', decision)`` and the caller skips the flow; an
    ``unmapped`` edge is relabeled ``valid_split_evidence`` when the
    caller's bridge discovery found MORE THAN ONE end (a licensed split)
    and ``bridged`` for a single end; every other status passes through.
    Returns ``(status, fields)`` where ``fields`` carries ``relationship``,
    ``target_types``, and ``conflicts`` from the scoped decision.
    """
    snap = snapshot if snapshot is not None else MapperSnapshot(mapper)
    cache_key = (str(source_dataset), str(type_name), str(target_dataset))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if mapper is None or not snap.loaded:
        result = (STATUS_MAPPER_UNAVAILABLE,
                  {'relationship': None, 'target_types': [],
                   'conflicts': [], 'suspects': False,
                   'suspect_rivals': [], 'same_name_first': None})
        if cache is not None:
            cache[cache_key] = result
        return result

    decision = snap.decision(type_name, source_dataset, target_dataset,
                             include_bridges=False)
    status = decision.get('status', STATUS_UNMAPPED)
    if status == STATUS_UNMAPPED:
        status = (STATUS_VALID_SPLIT if bridge_end_count > 1
                  else STATUS_BRIDGED)
    result = (status,
              {'relationship': decision.get('relationship'),
               'target_types': list(decision.get('target_types') or []),
               'conflicts': list(decision.get('conflicts') or []),
               # Same-name-first disclosure for the UI flow record
               # (plan-ui-type-mapper-alignment §3): the selection flag and
               # the demoted rivals, so the panel can render the collapsed
               # suspects block without re-deriving the decision.
               'suspects': bool(decision.get('suspects')),
               'suspect_rivals': list(
                   decision.get('same_name_first', {}).get('rivals') or []),
               'same_name_first': decision.get('same_name_first')})
    if cache is not None:
        cache[cache_key] = result
    return result


@dataclass(frozen=True)
class MergeKey:
    """Canonical merge key for one type plus the mapping status that
    produced it.  ``key`` is what cross-dataset path/edge merging groups
    by; ``status`` records why (``native`` for same-namespace identity)."""

    key: str
    status: str


def canonical_merge_key(
    mapper: Optional[CrossDatasetTypeMapper],
    type_name: str,
    source_dataset: str,
    target_dataset: Optional[str] = None,
    *,
    snapshot: Optional[MapperSnapshot] = None,
    cache: Optional[Dict] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> MergeKey:
    """Canonical key for merging one type across datasets, with status.

    The merge-oriented counterpart of the mapper's raw
    ``get_canonical_type``: same default target namespace (male-cns v1.0)
    and identical keys for every licensed mapping, but status-blind
    fallbacks become explicit:

    * mapped/bridged → the unique canonical target;
    * conflict → ``f'{source_dataset}:{raw}'`` — a conflicted type is kept
      DATASET-SCOPED so two datasets' same-named rows can never merge under
      a name the panel refuses to map (the BANC CB1011 ↔ MCNS CB1011 class);
    * valid splits / evidence-only / unmapped / unavailable mapper → the
      raw name, with the status recorded so callers can count fallbacks.

    ``on_status`` (optional) is invoked with the resolved status exactly
    ONCE per unique resolution — i.e. only when the answer is computed, not
    on a ``cache`` hit — so callers can count resolutions on a
    unique-``(source_dataset, type_name, target)`` basis without
    over-counting repeated lookups of the same type (path rows, edges,
    shared query items).  Pass a ``cache`` for the dedupe to hold across
    calls; without one every call is a fresh resolution.
    """
    raw = str(type_name or '').strip()
    target_ds = str(target_dataset) if target_dataset else CANONICAL_NAMESPACE
    cache_key = (str(source_dataset), raw, target_ds)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    def _cached(key: str, status: str) -> MergeKey:
        result = MergeKey(key=key, status=status)
        if cache is not None:
            cache[cache_key] = result
        if on_status is not None:
            on_status(status)
        return result

    snap = snapshot if snapshot is not None else MapperSnapshot(mapper)
    if mapper is None or not snap.loaded:
        return _cached(raw, STATUS_MAPPER_UNAVAILABLE)
    if not raw:
        return _cached(raw, STATUS_UNMAPPED)

    resolved_source = str(source_dataset)
    if not resolved_source:
        # Auto-detect the source namespace from the type name (mirrors
        # resolve_valid_targets); unknown -> raw long-tail fallback.
        try:
            detected = mapper._detect_type_source(raw)
        except Exception:
            detected = None
        if not detected:
            return _cached(raw, STATUS_UNMAPPED)
        resolved_source = str(detected)

    try:
        same_ns = (mapper._get_type_mapping_key(resolved_source)
                   == mapper._get_type_mapping_key(target_ds))
    except Exception:
        same_ns = resolved_source == str(target_ds)
    if same_ns:
        return _cached(raw, 'native')

    decision = snap.decision(raw, resolved_source, target_ds,
                             include_bridges=False)
    status = decision.get('status', STATUS_UNMAPPED)
    if status == STATUS_MAPPED and decision.get('target_type'):
        target = str(decision['target_type'])
        if _is_stale_claim(mapper, raw, resolved_source, target_ds, target):
            # RES-11: a claim the target namespace cannot fulfil must not
            # become the MERGE KEY either — two datasets would merge under
            # a name neither carries (the resolver demotes the same edge
            # to `claimed`); the raw name keeps the rows dataset-local.
            return _cached(raw, STATUS_CLAIMED)
        return _cached(target, STATUS_MAPPED)
    if status == STATUS_CONFLICT:
        return _cached(f'{resolved_source}:{raw}', STATUS_CONFLICT)
    return _cached(raw, status)


# --- profile-key expansion ------------------------------------------------

@dataclass(frozen=True)
class ProfileExpansion:
    """Status-aware canonicalization of one dataset-local profile.

    ``canonical`` maps comparison-namespace keys to accumulated weights.
    ``key_status`` records WHICH mapping status produced each key (the
    most restrictive status among its contributors), ``excluded`` records
    raw types deliberately kept out of automatic comparison with the
    reason, and ``status_counts`` aggregates the whole profile for
    result metadata.
    """

    canonical: Dict[str, float]
    key_status: Dict[str, str]
    excluded: Dict[str, str]
    status_counts: Dict[str, int]
    source_dataset: str
    target_dataset: str
    fallback_used: bool
    split_policy: str
    mapper_source: Optional[str] = None
    mapper_version: Optional[str] = None


def expand_profile_types(
    mapper: Optional[CrossDatasetTypeMapper],
    partner_types: Dict[str, float],
    source_dataset: str,
    target_dataset: Optional[str] = None,
    *,
    snapshot: Optional[MapperSnapshot] = None,
    decision_cache: Optional[Dict] = None,
    allow_evidence_only: bool = False,
) -> ProfileExpansion:
    """Convert a dataset-local profile into the comparison namespace.

    This is the status-aware replacement for raw
    ``standardize_partner_types`` consumption.  Per contributor type:

    * ``mapped`` — weight moves to the unique canonical target;
    * ``valid_split_evidence`` — weight is distributed EVENLY across all
      licensed targets (``split_policy='even'``: total mass preserved, no
      arbitrary branch, no double counting);
    * ``bridged`` — same as mapped (unique bridge-derived target);
    * ``evidence_only`` — excluded by default; distributed like a split
      only when ``allow_evidence_only=True``;
    * ``conflict`` — excluded; never compared by raw same-name;
    * ``unmapped`` — raw name kept as the long-tail fallback and counted.

    ``target_dataset=None`` canonicalizes toward the mapper's canonical
    namespace (male-cns v1.0), matching
    ``standardize_partner_types`` semantics for every licensed status.
    """
    snap = snapshot if snapshot is not None else MapperSnapshot(mapper)
    meta = snap.metadata()
    target_ds = str(target_dataset) if target_dataset else CANONICAL_NAMESPACE

    canonical: Dict[str, float] = {}
    key_status: Dict[str, str] = {}
    excluded: Dict[str, str] = {}
    status_counts: Counter = Counter()
    fallback_used = False

    def _cache_get(raw: str, src: str, tgt: str) -> Dict[str, Any]:
        ck = (src, raw, tgt)
        hit = decision_cache.get(ck) if decision_cache is not None else None
        if hit is None:
            hit = snap.decision(raw, src, tgt, include_bridges=False)
            if decision_cache is not None:
                decision_cache[ck] = hit
        return hit

    def _add(key: str, weight: float, status: str) -> None:
        if not key:
            return
        canonical[key] = canonical.get(key, 0.0) + weight
        prev = key_status.get(key)
        key_status[key] = status if prev is None else _merge_status(prev, status)

    def _emit(type_name: str, weight: float, prefix: str) -> None:
        nonlocal fallback_used
        base_key = type_name[len(prefix):] if prefix and (
            type_name.startswith(prefix)) else type_name
        if not isinstance(base_key, str) or not base_key:
            # Mirror standardize_partner_types: keep invalid keys as-is.
            _add(type_name, weight, STATUS_UNMAPPED)
            fallback_used = True
            status_counts[STATUS_UNMAPPED] += 1
            return
        if mapper is None or not snap.loaded:
            excluded[base_key] = STATUS_MAPPER_UNAVAILABLE
            status_counts[STATUS_MAPPER_UNAVAILABLE] += 1
            return
        try:
            same_ns = (mapper._get_type_mapping_key(source_dataset)
                       == mapper._get_type_mapping_key(target_ds))
        except Exception:
            same_ns = str(source_dataset) == str(target_ds)
        if same_ns:
            _add(prefix + base_key, weight, STATUS_MAPPED)
            status_counts[STATUS_MAPPED] += 1
            return
        d = _cache_get(base_key, source_dataset, target_ds)
        status = d.get('status', STATUS_UNMAPPED)
        if status == STATUS_MAPPED:
            tgt = d.get('target_type')
            if tgt and _is_stale_claim(mapper, base_key, source_dataset,
                                       target_ds, str(tgt)):
                # RES-11: profile weight must not land on a canonical key
                # no namespace carries — keep the raw name dataset-local
                # (the merge-key lane demotes the same edge to `claimed`).
                _add(prefix + base_key, weight, STATUS_CLAIMED)
                status_counts[STATUS_CLAIMED] += 1
                return
            if tgt:
                _add(prefix + str(tgt), weight, STATUS_MAPPED)
            else:
                _add(prefix + base_key, weight, STATUS_UNMAPPED)
                fallback_used = True
            status_counts[STATUS_MAPPED] += 1
        elif status == STATUS_VALID_SPLIT:
            tgts = [str(t) for t in (d.get('target_types') or ())
                    if t]
            if tgts:
                share = weight / len(tgts)
                for t in tgts:
                    _add(prefix + t, share, STATUS_VALID_SPLIT)
            else:
                _add(prefix + base_key, weight, STATUS_UNMAPPED)
                fallback_used = True
            status_counts[STATUS_VALID_SPLIT] += 1
        elif status == STATUS_EVIDENCE_ONLY:
            if allow_evidence_only:
                tgts = [str(t) for t in (d.get('target_types') or ())
                        if t]
                if tgts:
                    share = weight / len(tgts)
                    for t in tgts:
                        _add(prefix + t, share, STATUS_EVIDENCE_ONLY)
                else:
                    _add(prefix + base_key, weight, STATUS_UNMAPPED)
                    fallback_used = True
            else:
                excluded[base_key] = (
                    STATUS_EVIDENCE_ONLY
                    + ' (display-only; no automatic equivalence)')
                # RES-19: an exclusion is not a raw fallback — nothing was
                # re-keyed, so `raw_fallback_used` must not claim one.
            status_counts[STATUS_EVIDENCE_ONLY] += 1
        elif status == STATUS_CONFLICT:
            excluded[base_key] = STATUS_CONFLICT
            status_counts[STATUS_CONFLICT] += 1
        else:  # unmapped (or bridge-only: profile keys use no bridge walks)
            _add(prefix + base_key, weight, STATUS_UNMAPPED)
            fallback_used = True
            status_counts[STATUS_UNMAPPED] += 1

    for raw, weight in (partner_types or {}).items():
        if isinstance(raw, str) and raw.startswith('2hop:'):
            _emit(raw[5:], weight, '2hop:')
        else:
            _emit(raw if isinstance(raw, str) else str(raw), weight, '')

    return ProfileExpansion(
        canonical=canonical,
        key_status=key_status,
        excluded=excluded,
        status_counts=dict(status_counts),
        source_dataset=str(source_dataset),
        target_dataset=target_ds,
        fallback_used=fallback_used,
        split_policy='even',
        mapper_source=meta['source'],
        mapper_version=meta['version'],
    )


# --- result metadata ------------------------------------------------------

def auto_mapping_metadata(
    snapshot: MapperSnapshot,
    resolution_counts: Optional[Dict[str, int]] = None,
    raw_fallback_used: bool = False,
    partner_resolution_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """The standard auto-type-mapping metadata block for saved results.

    ``resolution_counts`` is the primary metric, on a
    ``mapping_resolution_counts_basis`` of ``'unique_type_resolutions'`` —
    one count per distinct ``(source_dataset, type[, target])`` resolver
    input, deduped so repeated lookups of one type are not over-counted.
    ``partner_resolution_counts`` (when supplied) is a SECOND, distinctly
    labeled metric on an occurrence basis — one count per contributor-type
    occurrence inside a canonicalized profile — and is never conflated with
    the primary counts.
    """
    meta = snapshot.metadata()
    meta.update({
        'auto_type_mapping_requested': snapshot.requested,
        'auto_type_mapping_active': snapshot.active,
        'auto_type_mapping_status': (
            'active' if snapshot.active
            else ('load_failed' if snapshot.requested else 'disabled')),
        'auto_type_mapping_source': meta.get('source'),
        'auto_type_mapping_version_or_hash': meta.get('version'),
        'auto_type_mapping_load_error': meta.get('load_error'),
        'mapping_policy_version': MAPPING_POLICY_VERSION,
        'mapping_resolution_counts_by_status': dict(resolution_counts or {}),
        'mapping_resolution_counts_basis': 'unique_type_resolutions',
        'mapping_partner_type_resolutions_by_status': dict(
            partner_resolution_counts or {}),
        'raw_fallback_used': bool(raw_fallback_used),
    })
    return meta
