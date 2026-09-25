"""Query-anchored merge policy for cross-dataset comparison runs.

Plan: ``_plan/plan-query-anchored-cross-dataset-analysis.md``.

A :class:`MergePolicy` is built once per comparison run from the query
chips and owns THE merge map: aligned-frame keys become group labels
instead of canonical-namespace keys.  Row-based only — the mapper
carries the bridge evidence (``get_mapping_decision`` /
``get_mapping_support`` / ``get_mapping_branches``); nothing here gates
or verifies a mapping, and no bodyId frame is ever rewritten.

Decisions encoded here (user, 2026-09-13/14):

* **B3 anchor selection, three cases** — all chips resolving into one
  dataset's naming anchor on that dataset (resolution-based, not
  origin-based); all-clean-same-name runs use the shared naming with no
  span warning; mixed runs warn and resolve conflicted groups to their
  minimal inseparable leaves.
* **B1 merge granularity** — a chip on the "1" side of a 1-to-N merges
  ALL its branches into one row (Decision 9: every branch, no vote
  threshold); a leaf-anchored chip covers only its own branch and the
  "1"-side parent keeps its own whole row.
* **Fan-in ownership** — a (dataset, type) claimed by two different
  groups keeps its dataset-scoped raw key and merges with NEITHER
  claimant (``[merge fan-in]`` warning).  First-group-wins is rejected
  as chip-order dependent; per-claimant frame copies are rejected as
  forbidden per-neuron partitioning.
* **Chip-order invariance** — chips are deduped and processed in sorted
  order, group ids are assigned after a canonical sort, warnings are
  deduped and sorted; permuting the chips cannot change any output.

The policy is a pure CONSUMER of the mapper surfaces; the bodyId
connectivity backend (``body_id_resolver.py``) is NOT consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # package-relative imports (coana runtime)
    from .label_mapper import LabelMapper
    from .query_resolver import (
        STATUS_BRIDGED,
        STATUS_MAPPED,
        STATUS_SAME_NAME_IDENTITY,
        resolve_query_tokens,
    )
except ImportError:  # pragma: no cover - direct script imports
    from label_mapper import LabelMapper
    from query_resolver import (
        STATUS_BRIDGED,
        STATUS_MAPPED,
        STATUS_SAME_NAME_IDENTITY,
        resolve_query_tokens,
    )


SPAN_WARNING = (
    '[type granularity] Query chips span multiple datasets; cross-dataset '
    'type granularity is ambiguous for intermediate types (1-to-N branches '
    'cannot be merged unambiguously). Recommend querying all chips from one '
    'dataset. Proceeding with minimal inseparable types (the branch leaves '
    'of every 1-to-N; shared parents keep their own whole row).'
)


@dataclass
class MergeGroup:
    """One merge group: a label plus the (dataset, raw type) members that
    collapse onto it.  ``branches`` carries the per-branch chains of a
    1-to-N (provenance/disclosure only — never extra merge keys)."""

    group_id: str = ''
    anchor: Tuple[str, str] = ('', '')
    label: str = ''
    kind: str = 'clean'  # clean | same_name | merged_split | leaf | parent
    members: Dict[str, List[str]] = field(default_factory=dict)
    branches: List['MergeGroup'] = field(default_factory=list)
    branch_pools: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_member(self, dataset: str, type_name: str) -> None:
        names = self.members.setdefault(dataset, [])
        if type_name not in names:
            names.append(type_name)

    def member_pairs(self):
        for dataset, names in sorted(self.members.items()):
            for name in names:
                yield dataset, name


@dataclass
class MergePolicy:
    """Per-run merge map.  ``key_for`` is the single lookup the metrics
    alignment consults; ``None`` means "not governed — fall back to
    ``canonical_merge_key`` toward the anchor namespace"."""

    anchor_ds: Optional[str] = None
    anchor_case: str = 'span'  # single_dataset | same_name | span
    groups: List[MergeGroup] = field(default_factory=list)
    key_map: Dict[Tuple[str, str], str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    fan_in: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    labels: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    def key_for(self, dataset: str, type_name: str) -> Optional[str]:
        return self.key_map.get((str(dataset), str(type_name)))

    def is_group_label(self, name: str) -> bool:
        return name in self.labels

    def group_of(self, dataset: str, type_name: str) -> Optional[MergeGroup]:
        label = self.key_map.get((str(dataset), str(type_name)))
        if label is None:
            return None
        for group in self.groups:
            if group.label == label:
                return group
        return None

    def group_by_label(self, label: str) -> Optional[MergeGroup]:
        for group in self.groups:
            if group.label == label:
                return group
        return None

    def label_for_name(self, name: str) -> Optional[str]:
        """Group label for a raw type name in ANY dataset (name-keyed
        convenience lookup for report code that no longer knows the
        dataset).  Deterministic: the key_map is built in canonical order,
        so duplicate names across datasets resolve identically under any
        chip permutation."""
        for (dataset, key_name), label in self.key_map.items():
            if key_name == name:
                return label
        return None

    def names_by_dataset(self, label: str) -> Dict[str, List[str]]:
        """Per-dataset raw names of a group (B6 hover) — ``{dataset:
        [names]}`` sorted, for the group label's own row."""
        group = self.group_by_label(label)
        if group is None:
            return {}
        return {
            dataset: sorted(names)
            for dataset, names in sorted(group.members.items())
        }

    def summary_line(self) -> str:
        if self.anchor_case == 'span':
            head = 'span-chip run: minimal inseparable leaves'
        else:
            head = f'anchored on {self.anchor_ds} ({self.anchor_case})'
        n_split = sum(1 for g in self.groups if g.branches)
        return (f'{head}; {len(self.groups)} group(s), '
                f'{len(self.key_map)} keyed type(s), '
                f'{n_split} split group(s), '
                f'{len(self.fan_in)} fan-in key(s)')

    # ------------------------------------------------------------------
    def synthesized_label_mapper(self) -> Optional[LabelMapper]:
        """Materialize raw→group-label as a LabelMapper riding the tested
        ``apply_to_dataframe`` → ``std_label_*`` lane (Decision 8).  The
        mapper is SEPARATE from the user's; user-mapped names never match
        these raw-keyed entries, so user mappings win by construction."""
        label_map: Dict[str, Dict[str, List[str]]] = {}
        for (dataset, raw), label in sorted(self.key_map.items()):
            if raw == label:
                continue
            label_map.setdefault(label, {}).setdefault(dataset, []).append(raw)
        if not label_map:
            return None
        source_labels = sorted(label_map)
        datasets = sorted({
            dataset
            for per_ds in label_map.values()
            for dataset in per_ds
        })
        source_mapping_dict = {
            dataset: [
                sorted(label_map[label].get(dataset, []))
                for label in source_labels
            ]
            for dataset in datasets
        }
        return LabelMapper(
            source_mapping_dict=source_mapping_dict,
            source_labels=source_labels,
        )

    # ------------------------------------------------------------------
    def topology_dict(self) -> Dict[str, Any]:
        """JSON-safe resolution topology (B5) — exported as
        ``type_resolution_topology.json`` next to ``auto_type_mapping.json``."""

        def _group_dict(group: MergeGroup) -> Dict[str, Any]:
            out: Dict[str, Any] = {
                'group_id': group.group_id,
                'anchor': {'dataset': group.anchor[0],
                           'type': group.anchor[1]},
                'label': group.label,
                'kind': group.kind,
                'members': {ds: sorted(names)
                            for ds, names in sorted(group.members.items())},
            }
            if group.branches:
                out['branches'] = [
                    {'label': b.label,
                     'members': {ds: sorted(names)
                                 for ds, names in sorted(b.members.items())},
                     'pools': group.branch_pools.get(b.label)}
                    for b in group.branches
                ]
            return out

        return {
            'anchor_case': self.anchor_case,
            'anchor_ds': self.anchor_ds,
            'summary': self.summary_line(),
            'warnings': list(self.warnings),
            'fan_in': {
                f'{ds}:{name}': sorted(labels)
                for (ds, name), labels in sorted(self.fan_in.items())
            },
            'groups': [_group_dict(g) for g in self.groups],
        }


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def _resolves_into(record: Dict[str, Any]) -> bool:
    """The chip lands in this dataset's naming: native identity, or a
    unique mapped/bridged target (B3's ``S(chip)`` membership test)."""
    status = str(record.get('status') or '')
    if status == STATUS_SAME_NAME_IDENTITY:
        return True
    if status in (STATUS_MAPPED, STATUS_BRIDGED) and record.get('target_types'):
        return True
    return False


def _is_native(record: Dict[str, Any]) -> bool:
    return str(record.get('status') or '') == STATUS_SAME_NAME_IDENTITY


def _decision(mapper, source_type: str, source_ds: str, target_ds: str):
    """The policy's decision probe — BRIDGES INCLUDED BY DESIGN (POL-6,
    resolved as documentation 2026-09-26).

    The 2026-09-25 audit proposed include_bridges=False to match the
    canonical merge fallback.  The test suite says otherwise: bridge-derived
    branch membership is the DESIGNED behavior this policy exists for
    (17+ tests pin it — e.g. the 5thsLNv->LNd6 linker chain that motivated
    the per-side basis), with the auto-only chain-terminal and fan-in/
    global-contest lanes as its guardrails.  "Byte-identical to the
    canonical fallback" holds for UNGOVERNED types (which never reach this
    probe), not for governed chips.  The morph cross-dataset ref pools
    make the same choice, consistently with TM VEV validating bridged
    pairs forward."""
    try:
        return mapper.get_mapping_decision(source_type, source_ds, target_ds)
    except Exception:  # noqa: BLE001 — decision failures mean "no relation"
        return {'status': 'unmapped', 'target_type': None,
                'target_types': [], 'conflicts': [], 'support': None}


def _split_branches(decision: Dict[str, Any]) -> List[str]:
    """Sorted branch names when the decision is a 1-to-N rooted at the
    queried type (crosswalk split or conflicted split — Decision 9 keeps
    every branch, no vote threshold)."""
    status = str(decision.get('status') or '')
    targets = sorted(str(t) for t in (decision.get('target_types') or []))
    if not targets:
        return []
    if status == 'valid_split_evidence':
        return targets
    if status == 'conflict':
        for conflict in decision.get('conflicts') or []:
            if conflict.get('relationship') == '1-to-N':
                return sorted(str(t) for t in (conflict.get('target_types')
                                               or targets))
    return []


def _branch_chain(mapper, branch: str, branch_home: str, datasets: List[str],
                  exclude: Set[str]) -> Tuple[Tuple[str, str],
                                              Dict[str, str]]:
    """Follow ONE branch's clean pairings outward from its home dataset.

    The branch lives in ``branch_home`` (the dataset the 1-to-N split
    points into); from there, unique mapped/bridged targets extend the
    chain dataset by dataset in canonical order.  Returns the branch's
    anchor pair (the N-side leaf) plus its per-dataset chain."""
    chain: Dict[str, str] = {branch_home: branch}
    current, current_ds = branch, branch_home
    for ds in sorted(datasets):
        if ds in exclude or ds in chain:
            continue
        decision = _decision(mapper, current, current_ds, ds)
        if (decision.get('status') in ('mapped', 'bridged')
                and decision.get('target_type')):
            current = str(decision['target_type'])
            chain[ds] = current
            current_ds = ds
    return (branch_home, branch), chain


def _is_split_parent(mapper, target_type: str, target_ds: str,
                     anchor_ds: str) -> bool:
    """True when a clean mapping target is itself the "1"-side of a 1-to-N
    toward the anchor namespace.  B1: such a parent keeps its own whole
    row — it must not be absorbed into a leaf-anchored group even when
    the REVERSE direction is a clean rename (real-data finding 2026-09-15:
    FAFB ``5th-LNv`` → MCNS ``5thsLNv_LNd6`` is a licensed rename while
    MCNS → FAFB is the 1-to-N conflict)."""
    try:
        conflicts = mapper.get_mapping_conflicts(
            target_ds, anchor_ds, target_type)
    except Exception:  # noqa: BLE001
        return False
    return any(
        getattr(conflict, 'relationship', None) == '1-to-N'
        for conflict in (conflicts or [])
    )


def _unify_chains(seeds: List[Tuple[Tuple[str, str], Dict[str, str]]]):
    """Merge seed chains that share any (dataset, name) member.  The
    surviving anchor is the FIRST seed's (deterministic: seeds arrive in
    dataset/branch-sorted order)."""
    unified: List[List[Any]] = []  # [anchor_pair, chain_dict]
    for anchor, chain in seeds:
        overlap = [
            existing for existing in unified
            if any((ds, name) in existing[1].items()
                   for ds, name in chain.items())
        ]
        if overlap:
            base = overlap[0]
            for other in overlap[1:]:
                base[1].update(other[1])
                unified.remove(other)
            base[1].update(chain)
            continue
        unified.append([anchor, dict(chain)])
    return unified


def _format_support(support: Optional[Dict[str, Any]]) -> str:
    if not support:
        return ''
    votes = support.get('votes') or {}
    verified = support.get('verified_votes') or {}
    auto = support.get('auto_stripped_votes') or {}
    parts: List[str] = []
    if verified:
        curated = ', '.join(f'{k} ({v})' for k, v in sorted(verified.items())
                            if v)
        if curated:
            parts.append(f'curated votes: {curated}')
    if votes:
        total = sum(int(v or 0) for v in votes.values())
        parts.append(f'{total} label vote(s)')
    if not verified and (auto or support.get('winner_derived_from_auto')):
        parts.append('auto-transferred labels only')
    return '; '.join(parts)


def build_merge_policy(
    mapper,
    *,
    source_tokens,
    target_tokens,
    datasets: List[str],
    source_dataset: Optional[str] = None,
    log=None,
    attach_pools=True,
    records: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Optional[MergePolicy]:
    """Build the per-run MergePolicy, or ``None`` when there is nothing to
    govern (no mapper, or no usable chips).  Purely additive: a ``None``
    return keeps every downstream path byte-identical to the canonical
    merge fallback.

    ``attach_pools``: ``True`` binds the comparison layer's default
    ``pool_bridge_body_ids`` lazily (UI import at call time); a callable
    is used directly as the ``pool_fn`` (tests / alternate pool sources);
    ``False`` skips pool disclosure entirely.

    ``records`` optionally injects pre-resolved per-chip/per-dataset
    records (``{token: {dataset: QueryResolution-dict}}``) — used by
    tests; by default the chips are resolved through
    ``resolve_query_tokens`` exactly like ``resolve_query_inputs``."""
    if mapper is None:
        return None
    tokens = sorted({
        str(token).strip()
        for token in list(source_tokens or []) + list(target_tokens or [])
        if str(token) and str(token).strip()
    })
    if not tokens:
        return None
    datasets = [str(ds) for ds in datasets or []]
    if len(datasets) < 2:
        return None

    def _log(message: str) -> None:
        if log is not None:
            log(message)

    # ---- resolve every chip against every dataset (canonical order) ----
    records = records if records is not None else {}
    if not records:
        try:
            flat = resolve_query_tokens(
                tokens, datasets, mapper,
                role='source', source_dataset=source_dataset)
        except Exception as exc:  # noqa: BLE001 — resolution failure = fallback
            _log(f'Merge policy: query resolution failed ({exc}); '
                 'falling back to the canonical merge')
            return None
        for record in flat:
            as_dict = (
                record.as_dict() if hasattr(record, 'as_dict')
                else dict(record))
            records.setdefault(as_dict['token'], {})[
                as_dict['dataset']] = as_dict

    # ---- B3: per-chip naming sets and the anchor case ------------------
    naming_sets: Dict[str, Set[str]] = {}
    native_sets: Dict[str, Set[str]] = {}
    for token in tokens:
        per_ds = records.get(token) or {}
        naming_sets[token] = {
            ds for ds in datasets if _resolves_into(per_ds.get(ds) or {})
        }
        native_sets[token] = {
            ds for ds in datasets if _is_native(per_ds.get(ds) or {})
        }
    selected = set(datasets)
    if all(native_sets[token] >= selected for token in tokens):
        # TRUE same-name run (B3 case 2): every chip is NATIVE in every
        # dataset — the naming is the shared name itself.  A mapped RENAME
        # does not qualify (real-data finding 2026-09-15: FAFB 5th-LNv
        # renames cleanly into MCNS).  anchor_ds stays None so the
        # fallback for UNGOVERNED types keeps the canonical namespace —
        # a same-name run makes no claim about non-chip types, and the
        # anchor pick is NOT unobservable for them.
        anchor_case = 'same_name'
        anchor_ds = None
    else:
        intersection = set.intersection(
            *(naming_sets[token] for token in tokens)) if tokens else set()
        if not intersection:
            anchor_case = 'span'
            anchor_ds = None
        else:
            # Case 1: anchor on the FIRST chip's home namespace when it
            # lies in the common naming set (canonical chip order keeps
            # this order-invariant); else the canonical pick from the
            # intersection.
            anchor_case = 'single_dataset'
            first_home = None
            for token in tokens:
                if native_sets[token]:
                    first_home = (
                        source_dataset
                        if source_dataset in native_sets[token]
                        else sorted(native_sets[token])[0])
                    break
            anchor_ds = (first_home if first_home in intersection
                         else sorted(intersection)[0])

    policy = MergePolicy(
        anchor_ds=anchor_ds,
        anchor_case=anchor_case,
    )
    warnings: List[str] = list(policy.warnings)
    if anchor_case == 'span':
        warnings.append(SPAN_WARNING)

    # ---- B1: derive one group per chip ---------------------------------
    groups: List[MergeGroup] = []
    claims: Dict[Tuple[str, str], Dict[str, str]] = {}

    def _claim(pair: Tuple[str, str], label: str) -> None:
        claims.setdefault(pair, {})[label] = label

    for token in tokens:
        per_ds = records.get(token) or {}
        non_type = all(
            str((per_ds.get(ds) or {}).get('status') or '')
            in ('body_id', 'pattern', 'group', 'taxonomy')
            for ds in datasets
        ) if per_ds else False
        if non_type:
            continue  # body ids / patterns / groups are not type chips
        native = sorted(ds for ds in datasets if _is_native(per_ds.get(ds) or {}))
        if native:
            home = (source_dataset if source_dataset in native
                    else native[0])
            kind = 'same_name' if len(native) > 1 else 'clean'
        elif naming_sets[token]:
            home = sorted(naming_sets[token])[0]
            kind = 'leaf'
        else:
            warnings.append(
                f'[unresolved chip] {token} resolves in no selected '
                'dataset; left to the canonical merge fallback')
            continue

        group = MergeGroup(
            anchor=(home, token),
            label=token,
            kind=kind,
        )
        group.add_member(home, token)
        _claim((home, token), token)
        # A chip native in several datasets (same-name identity) claims
        # its raw name in EACH of them — the name IS the same type there.
        for ds in native:
            group.add_member(ds, token)
            _claim((ds, token), token)

        seed_chains: List[Tuple[Tuple[str, str], Dict[str, str]]] = []
        for ds in sorted(datasets):
            if ds == home:
                continue
            decision = _decision(mapper, token, home, ds)
            branches = _split_branches(decision)
            if branches:
                branch_support = decision.get('support') or {}
                for branch in branches:
                    # Auto-only branch seeds are CHAIN-TERMINAL (real-data
                    # leak, 2026-09-15): a branch licensed only by
                    # auto-transferred labels stays confined to its own
                    # dataset node.  Extending it through downstream
                    # identities recruited unrelated same-named cell types
                    # (the FAFB aMe24 case: 1 auto label on 1 of 2 BANC
                    # neurons claimed the whole FAFB aMe24 type via the
                    # curated aMe24=aMe24 identity).  The branch itself
                    # stays valid and warned (Decision 9 / B4); the strong
                    # side's chains still cover the pairings.
                    support = branch_support.get(branch)
                    auto_only = bool(
                        support and support.get('votes')
                        and not support.get('verified_votes'))
                    if auto_only:
                        seed_chains.append(
                            ((ds, branch), {ds: branch}))
                        continue
                    seed_chains.append(
                        _branch_chain(mapper, branch, ds, datasets,
                                      exclude={home}))
            elif (decision.get('status') in ('mapped', 'bridged')
                    and decision.get('target_type')):
                target_name = str(decision['target_type'])
                if _is_split_parent(mapper, target_name, ds, home):
                    # B1: a clean target that is itself the "1"-side of a
                    # 1-to-N toward the chip's namespace keeps its own
                    # whole row (the reverse rename must not absorb the
                    # parent into the leaf group).
                    continue
                group.add_member(ds, target_name)
                _claim((ds, target_name), token)
            # evidence_only / unmapped / plain vote conflict: the chip has
            # no claim in this dataset (leaf-anchored semantics).

        unified = _unify_chains(seed_chains)
        if unified:
            group.kind = ('merged_split' if anchor_case != 'span'
                          else 'parent')
            for leaf_anchor, chain in unified:
                branch_group = MergeGroup(
                    anchor=leaf_anchor,
                    label=leaf_anchor[1],
                    kind='branch',
                    members={ds: [name]
                             for ds, name in sorted(chain.items())},
                )
                group.branches.append(branch_group)

        if anchor_case != 'span':
            # merged_split: every branch member folds into the anchor's
            # ONE row (B1 "1"-side semantics).
            for branch_group in group.branches:
                for ds, name in branch_group.member_pairs():
                    group.add_member(ds, name)
                    _claim((ds, name), token)
        else:
            # span: the parent keeps its own row; each branch becomes its
            # own leaf group (minimal inseparable leaves).
            for branch_group in group.branches:
                for ds, name in branch_group.member_pairs():
                    _claim((ds, name), branch_group.label)
                groups.append(branch_group)

        groups.append(group)

    # ---- Phase B: branch pools + vote support (row-based disclosure) ----
    pool_fn = None
    if attach_pools is True:
        try:
            from .mapping_visualization import default_branch_pool_fn
            pool_fn = default_branch_pool_fn()
        except Exception:  # noqa: BLE001 — pools are optional evidence
            pool_fn = None
    elif callable(attach_pools):
        pool_fn = attach_pools
    if pool_fn is not None:
        _attach_branch_pools(mapper, groups, pool_fn)

    # ---- fan-in ownership: a key claimed by two labels merges with
    # NEITHER (kept raw; `[merge fan-in]` warning) ------------------------
    fan_in: Dict[Tuple[str, str], List[str]] = {}
    for pair, labels in sorted(claims.items()):
        if len(labels) > 1:
            fan_in[pair] = sorted(labels)
    if fan_in:
        for (ds, name), labels in sorted(fan_in.items()):
            warnings.append(
                f'[merge fan-in] {ds} {name} is claimed by '
                f'{" and ".join(labels)}; it merges with neither claimant '
                'and stays its own row (custom label mapper can override)')
        # Explicit parent-partial note: when one of the claimants is a
        # QUERIED parent chip (a merged group with branches), that
        # parent's row EXCLUDES the shared branch — the user must see
        # that the parent row is partial by design.
        parent_chip_labels = {
            group.label for group in groups
            if group.branches and group.kind in ('merged_split', 'parent')
        }
        for (ds, name), labels in sorted(fan_in.items()):
            for label in labels:
                if label in parent_chip_labels:
                    others = ' and '.join(
                        other for other in labels if other != label)
                    warnings.append(
                        f'[merge fan-in] queried parent {label} does NOT '
                        f'include {ds} {name} (claimed by {others}): the '
                        'parent row is PARTIAL by design — the shared '
                        'branch is presented separately at its own '
                        'minimal level. Query the parent alone to merge '
                        'it, or use the custom label mapper.')

        # Fan-in pruning (display lane): key_map below already keeps the
        # shared key out of the aggregation — the claimant groups'
        # members/branches must agree, or the type-mapping table and the
        # topology show the shared type inside BOTH parent rows while the
        # counts exclude it (real-data finding 2026-09-15: CL317 appeared
        # inside the aMe26 and aMe9 rows although it merges with neither).
        if fan_in:
            for group in groups:
                kept_branches = []
                for branch in group.branches:
                    for ds in list(branch.members):
                        kept = [
                            n for n in branch.members[ds]
                            if (ds, n) not in fan_in
                            or group.label not in fan_in[(ds, n)]]
                        if kept:
                            branch.members[ds] = kept
                        else:
                            del branch.members[ds]
                    if branch.members:
                        kept_branches.append(branch)
                    else:
                        group.branch_pools.pop(branch.label, None)
                group.branches = kept_branches
                for ds in list(group.members):
                    kept = [
                        n for n in group.members[ds]
                        if (ds, n) not in fan_in
                        or group.label not in fan_in[(ds, n)]]
                    if kept:
                        group.members[ds] = kept
                    else:
                        del group.members[ds]

    # Globally contested branches (data-scoped fan-in, same lane): a
    # branch whose name is ALSO a 1-to-N target of a DIFFERENT source
    # type in the parent's home namespace is ambiguous in the mapper's
    # crosswalk regardless of this run's chips — the run-scoped check
    # above only fires when the rival parent is queried too (real-data
    # finding 2026-09-16: banc CL317 is a 1-to-N target of BOTH aMe26
    # and aMe9, so every aMe26-without-aMe9 run re-folded CL317 into
    # the aMe26 row and key_map).  Same treatment as fan-in: the
    # branch leaves the parent row AND the key map, so display and
    # counts agree; the branch name stays its own raw row downstream.
    conflicts_fn = getattr(mapper, 'get_mapping_conflicts', None)
    if conflicts_fn is not None and datasets:
        for group in groups:
            if not group.branches:
                continue
            home = group.anchor[0]
            if not home:
                continue
            # rivals per branch label: other home-namespace source
            # types whose 1-to-N conflicts list the branch name.
            contested: Dict[str, Set[str]] = {}
            # POL-9: one crosswalk scan per (home, ds) — the old loop
            # re-scanned per BRANCH per dataset (O(branches x datasets)
            # where O(datasets) suffices).
            pair_conflicts_by_ds: Dict[str, list] = {}
            for ds in datasets:
                try:
                    pair_conflicts_by_ds[ds] = conflicts_fn(home, ds) or []
                except Exception:  # noqa: BLE001 — no mapper = no contest
                    pair_conflicts_by_ds[ds] = []
            for branch in group.branches:
                branch_name = str(branch.label)
                rivals: Set[str] = set()
                for ds in datasets:
                    for conflict in pair_conflicts_by_ds[ds]:
                        if str(getattr(conflict, 'relationship', '')) \
                                != '1-to-N':
                            continue
                        if str(getattr(conflict, 'source_type', '')) \
                                == group.label:
                            continue
                        if branch_name in {
                                str(t) for t in
                                (getattr(conflict, 'target_types', None)
                                 or [])}:
                            rivals.add(str(conflict.source_type))
                if rivals:
                    contested[branch_name] = rivals
            if not contested:
                continue
            for branch_name, rivals in sorted(contested.items()):
                warnings.append(
                    f'[merge fan-in] {home} {branch_name} is also a '
                    f'1-to-N target of {" and ".join(sorted(f"{home} {r}" for r in rivals))} '
                    f'in the type-mapper crosswalk (global contest — '
                    f'the rival parent is not queried in this run); it '
                    f'merges with neither claimant and stays out of '
                    f'the {group.label} row')
            kept_branches = []
            # POL-4 (2026-09-26, resolved for the name scope): the
            # 2026-09-25 audit proposed (dataset, name)-scoped pruning,
            # but the pinned contract says the contested NAME leaves the
            # group EVERYWHERE — a branch's members legitimately live in
            # datasets other than `home` (the branch chains fold them in),
            # and a branch is a group-specific concept: an "unrelated
            # same-named member" would have arrived through this group's
            # own claims, which is exactly the ambiguity the contest
            # flags.  (test_globally_contested_branch_stays_out_of_parent_row)
            for branch in group.branches:
                if str(branch.label) in contested:
                    group.branch_pools.pop(branch.label, None)
                    continue
                kept_branches.append(branch)
            group.branches = kept_branches
            for ds in list(group.members):
                kept = [n for n in group.members[ds]
                        if str(n) not in contested]
                if kept:
                    group.members[ds] = kept
                else:
                    del group.members[ds]

    # ---- canonical group ids + key map ----------------------------------
    # POL-8: the fan-in / global-contest prunes can leave a group with
    # NO members — an empty group still received an id, a label in
    # policy.labels and a topology row, so group_by_label could resolve
    # to a group that keys nothing.
    groups = [g for g in groups if any(g.members.values())]
    groups.sort(key=lambda g: (g.anchor[0], g.anchor[1], g.label))
    # Deduplicate identical groups (same label AND same member set): in a
    # span run a leaf chip's own group and a parent's unified branch can
    # come out identical — keep the first (chip group), drop the shadow.
    deduped: List[MergeGroup] = []
    seen_groups: Set[Tuple[str, frozenset]] = set()
    for group in groups:
        identity = (
            group.label,
            frozenset((ds, tuple(names))
                      for ds, names in group.members.items()),
        )
        if identity in seen_groups:
            continue
        seen_groups.add(identity)
        deduped.append(group)
    groups = deduped
    for index, group in enumerate(groups, start=1):
        group.group_id = f'g{index:03d}'
    for group in groups:
        for ds, name in group.member_pairs():
            if (ds, name) in fan_in:
                continue
            policy.key_map[(ds, name)] = group.label
        # identity entries: a label already written by the synthesized
        # lane re-keys to itself at the canonical-map stage.  POL-10: when
        # a DIFFERENT group pre-empted the pair, say so — key_for silently
        # resolving this group's label into a rival's row (blanking its own
        # mapping row) was undiscoverable.
        for ds in sorted(group.members):
            pre = policy.key_map.get((ds, group.label))
            if pre is not None and pre != group.label:
                warnings.append(
                    f'[merge key] {ds} {group.label}: the name is already '
                    f'keyed under group {pre!r}; label_for_name resolves '
                    f'there (custom label mapper can override)')
                continue
            policy.key_map[(ds, group.label)] = group.label

    policy.groups = groups
    policy.fan_in = fan_in
    policy.warnings = sorted(set(warnings))
    policy.labels = {group.label for group in groups}
    return policy


# ----------------------------------------------------------------------
# Mapping-row annotation (auto_type_mapping.csv columns)
# ----------------------------------------------------------------------

def row_anchor_group(policy: 'MergePolicy',
                     values_by_dataset: Dict[str, str]) -> str:
    """The ``anchor_group`` value for one mapping-table row.

    UNAMBIGUOUS RULE: the row is tagged with a group label only when
    EVERY non-empty endpoint type in the row resolves — via the policy's
    own key map — to that SAME group.  A row that merely touches a group
    through one endpoint (e.g. ``LNd_b -> LNd_a`` where only ``LNd_a``
    is a member) stays blank, so the column always means "this row IS a
    mapping of that group", never "this row is adjacent to it"."""
    label: Optional[str] = None
    for dataset, name in values_by_dataset.items():
        if not name:
            continue
        row_label = policy.key_for(dataset, name)
        if row_label is None:
            return ''
        if label is None:
            label = row_label
        elif label != row_label:
            return ''
    return label or ''


def _attach_branch_pools(mapper, groups: List[MergeGroup],
                         pool_fn) -> None:
    """Row-based branch pools + vote provenance per split branch (B2).

    Uses ``get_mapping_branches`` with the given ``pool_fn``; failures
    degrade to support-only disclosure — pools are evidence, never
    membership."""
    for group in groups:
        if not group.branches:
            continue
        parent_ds, parent_type = group.anchor
        for branch in group.branches:
            leaf_ds, leaf_type = branch.anchor
            if leaf_ds == parent_ds:
                continue
            entry: Dict[str, Any] = {
                'support': _format_support(_support_for(
                    mapper, parent_type, parent_ds, leaf_type, leaf_ds)),
                'pools': [],
            }
            try:
                records = mapper.get_mapping_branches(
                    parent_type, parent_ds, leaf_ds, pool_fn=pool_fn) or []
            except Exception:  # noqa: BLE001
                records = []
            for record in records:
                if str(record.get('target_type') or '') != leaf_type:
                    continue
                entry['pools'].append({
                    'source_basis': record.get('source_basis'),
                    'target_basis': record.get('target_basis'),
                    'source_pool_size': record.get('source_pool_size'),
                    'source_type_total': record.get('source_type_total'),
                    'target_pool_size': record.get('target_pool_size'),
                    'target_type_total': record.get('target_type_total'),
                    'supported': record.get('supported'),
                    'status': record.get('status'),
                })
            group.branch_pools[branch.label] = entry


def _support_for(mapper, source_type, source_ds, target_type, target_ds):
    try:
        return mapper.get_mapping_support(
            source_type, source_ds, target_type, target_ds)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# B4: auto-only BANC mapping edges (evidence-only warning feed)
# ----------------------------------------------------------------------

def auto_only_edges(mapper, types, datasets) -> List[Dict[str, Any]]:
    """Mapping edges whose evidence rests on auto-transferred labels only
    (``votes`` present, ``verified_votes`` empty,
    ``winner_derived_from_auto``) — the B4 warning feed.  Mappings stay
    VALID; the custom label mapper is the removal path."""
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for source_type in sorted({str(t) for t in types or []}):
        for source_ds in sorted(datasets or []):
            for target_ds in sorted(datasets or []):
                if source_ds == target_ds:
                    continue
                decision = _decision(mapper, source_type, source_ds,
                                     target_ds)
                status = str(decision.get('status') or '')
                pairs: List[Tuple[str, Any]] = []
                if decision.get('target_type'):
                    pairs.append((str(decision['target_type']),
                                  decision.get('support')))
                if status == 'valid_split_evidence':
                    branch_supports = decision.get('support') or {}
                    for branch in decision.get('target_types') or []:
                        pairs.append((str(branch), branch_supports.get(branch)))
                for target_type, support in pairs:
                    if not support:
                        continue
                    votes = support.get('votes') or {}
                    verified = support.get('verified_votes') or {}
                    if not votes or verified:
                        continue
                    # POL-5: same condition as the build path — a branch
                    # whose votes are manual-but-unverified is ALSO
                    # chain-terminal and belongs in this disclosure.
                    key = (source_ds, source_type, target_ds, target_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        'source_dataset': source_ds,
                        'source_type': source_type,
                        'target_dataset': target_ds,
                        'target_type': target_type,
                        'votes': dict(votes),
                        'total_votes': sum(int(v or 0)
                                           for v in votes.values()),
                        'support_text': _format_support(support),
                    })
    rows.sort(key=lambda row: (row['source_dataset'], row['source_type'],
                               row['target_dataset'], row['target_type']))
    return rows
