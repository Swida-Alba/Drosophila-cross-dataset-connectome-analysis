"""Round 2 — the standalone Type Mapping entrance (composed global view).

"Type Mapping" beside the Cross-Dataset tab's dataset selector opens a
preview dialog that maps the search across EVERY ordered pair of the
selected datasets (the global search) and renders one composed
N-dataset type-level graph plus per-pair cards and CSV exports.
Strictly informational — nothing leaks into the analysis selection
(spec: _plan/plan-type-mapping-round2-entrance-composed-view.md).

The composed view exports two canvases: **Mapping graph (HTML)** (the
cytoscape network) and **Mapping sankey (HTML)** (one column per dataset,
the query's origin centred, offered for up to two target datasets).  Both
plot types only — the query chip is provenance, not a node — and when one
origin dataset maps into exactly two targets the origin takes the middle
column with a target on each flank.

A `⚠ suspects (N)` badge in any table explains itself TWO ways: hover the
badge for the same-name-first note plus one line per rival, or read the
collapsed **Suspects** block below the table (the hover is additive — the
block stays the no-hover route).

The panel's bodyId-level data is ROW-BASED BRIDGE EVIDENCE only —
per-branch linker-refined bodyId pools and label-vote provenance
(``mapping_support`` / ``support_*`` export columns; ``Export branch
bodyIds`` download).  Connectivity similarity is NOT part of the type
mapper: per-neuron split verification lives in the
validate-expand-visualize pipeline (``comparison.mapping_validation``).
A collapsed per-type breakdown expansion appears for multi-type
previews — one row per (matched type × target dataset) so the mapped
counts are dataset-specific, never summed across datasets, with query
chip provenance (user 2026-09-14).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Tuple

from nicegui import ui

from .banner import push_banner
from .common import neuron_list_input

logger = logging.getLogger(__name__)

COMPOSED_NODE_CAP = 80

# Quasar table cells render nowrap by default, so a long Map used /
# Maps-to cell stretches its table until the later columns overflow the
# viewport.  These column helpers cap the width and force wrapping so
# ALL columns stay visible and long values continue on the next line
# (user 2026-09-07).
_WRAP = "white-space: normal; overflow-wrap: anywhere;"

# §12.5: the four trailing coverage columns carry short visible labels;
# the full dataset key and the selected/all-valid scope explanation ride
# on each column's `tooltip` key, rendered by this header-cell slot.
_HEADER_TOOLTIP_SLOT = (
    '<q-th :props="props">'
    '<q-tooltip v-if="props.col.tooltip" anchor="top middle" '
    'self="bottom middle" max-width="26rem" '
    'style="white-space: normal">{{ props.col.tooltip }}'
    '</q-tooltip>'
    '{{ props.col.label }}'
    '</q-th>'
)


def _col(name: str, label: str, field: str, *,
         max_w=None, min_w=None, tooltip=None) -> Dict[str, Any]:
    """A QTable column def with wrapping, an optional width cap, and an
    optional header tooltip (rendered by ``_HEADER_TOOLTIP_SLOT``)."""
    bounds = ((f"max-width: {max_w}px;" if max_w else "")
              + (f"min-width: {min_w}px;" if min_w else ""))
    column = {
        "name": name, "label": label, "field": field, "align": "left",
        "headerStyle": _WRAP + bounds,
        "style": _WRAP + bounds,
    }
    if tooltip:
        column["tooltip"] = tooltip
    return column


def _format_mapped_neurons(neurons: int, types: int) -> str:
    """The combined mapped cell: '{N}({m} types)' from 2 mapped types
    up; a single (or zero) mapped type renders as just '{N}'."""
    if types >= 2:
        return f"{neurons}({types} types)"
    return str(neurons)


# ---------------------------------------------------------------------------
# §full-map route scope (plan-full-map-route-scope.md; user 2026-09-28;
# RE-RATIFIED 2026-10-01 as a full parallel mode)
# ---------------------------------------------------------------------------
# Full map is now a COMPLETE second mode beside Curated: the composition
# engine lives in the mapper (compose_full_map_bridges — ALL datasets
# licensed as connectors), get_type_bridges/get_mapping_decision carry a
# route_scope keyword, the panel data path (flows, tiers, exports,
# visualizations) runs on the mode's route universe, and the validation
# pipeline accepts the mode through MappingValidationConfig.route_scope.
# Curated stays the default and is byte-identical to the 2026-09-28
# behavior; claim/reach figures in FULL mode legitimately differ (the
# mode is wholesale).


def compose_full_map_reach(mapper, src_ds: str, tgt_ds: str,
                           source_types: List[str]) -> Dict[str, Any]:
    """Compose the FULL-MAP reach for one queried pair.

    Thin back-compat wrapper (2026-10-01 re-ratification): the
    composition engine now lives in the mapper
    (``CrossDatasetTypeMapper.compose_full_map_bridges`` — the same
    engine behind ``get_type_bridges(route_scope='full')`` and the
    full-mode decisions), licensed over ALL datasets as connectors.  The
    historical return contract is preserved:

    * ``ends`` / ``routes`` — every composed target end and the routes
      reaching it (``direct`` chain lists + ``transitive`` route
      records carrying ``source_type``/``mid``/``via_type``/``leg_a``/
      ``leg_b``);
    * ``direct_ends`` / ``transitive_only_ends`` / ``corroborated``;
    * ``route_conflicts`` — one entry per (mid type, mid pair) whose
      transitive routes license DIFFERENT ends (the union shows both;
      the marker makes the disagreement visible);
    * ``mids`` — the candidate intermediates consulted.
    """
    composed = mapper.compose_full_map_bridges(source_types, src_ds, tgt_ds)
    routes_out: Dict[str, Dict[str, list]] = {}
    transitive_only: List[Dict[str, Any]] = []
    direct_ends = set(composed.get('direct_ends') or [])
    for end in composed.get('ends', []):
        bucket = composed.get('chains_by_end', {}).get(end, {})
        records = list(composed.get('routes', {}).get(end, []))
        routes_out[end] = {
            'direct': [list(h) for h in bucket.get('direct', [])],
            'transitive': records,
        }
        if records and end not in direct_ends:
            transitive_only.extend(records)
    return {
        'ends': list(composed.get('ends', [])),
        'direct_ends': list(composed.get('direct_ends', [])),
        'routes': {end: {k: v for k, v in rts.items() if v}
                   for end, rts in routes_out.items()},
        'transitive_only': transitive_only,
        'transitive_only_ends': list(
            composed.get('transitive_only_ends', [])),
        'corroborated': list(composed.get('corroborated', [])),
        'route_conflicts': list(composed.get('route_conflicts', [])),
        'mids': list(composed.get('mids', [])),
    }


def _pair_card_label(src: str, tgt: str, flows) -> str:
    """The pair-card heading: the count is the ADOPTED pairs.

    Disclosure rows — a same-name rival the decision declined, a split
    fan-out it did not adopt — stay listed in the card, but they are
    counted separately so the heading cannot read as a wider mapping than
    the one the analysis (and the TM VEV report) uses.
    """
    from comparison.mapping_visualization import flow_is_claimed

    claimed = sum(1 for f in flows if flow_is_claimed(f))
    label = f"{src} → {tgt} · {claimed} mapped pairs"
    extra = len(flows) - claimed
    return label + (f" (+{extra} disclosure rows)" if extra else "")


def _pool_mapping_pair(flows, src, tgt, indexes, pools) -> None:
    """Pool bodyIds for one type-mapping pair.

    BodyIds are coverage evidence only.  The type-level mapping has already
    been resolved by the mapper; this helper merely computes the independent
    endpoint pools used to report ``m of n`` coverage in the rendered card.
    """
    from ..neuron_index import resolve_prioritized_bridge_pool
    from comparison.mapping_visualization import mapping_pool_key

    for flow in flows:
        chains = [c for c in (flow.get("bridges") or [])
                  if c and c[-1].get("value") == flow.get("foreign_type")]
        chains = chains or flow.get("bridges") or []
        key = mapping_pool_key(
            src, tgt, flow.get("source_type"), flow.get("foreign_type"))
        if key in pools:
            continue
        index_kw = {
            dataset: index for dataset, index in (
                (src, indexes.get(src)), (tgt, indexes.get(tgt)))
            if index is not None
        }
        result = resolve_prioritized_bridge_pool(
            src, tgt, chains,
            flow.get("source_type"), flow.get("foreign_type"),
            indexes=index_kw)
        for attempt in result.get("attempts") or []:
            if attempt.get("status") in {"unsupported", "error"}:
                linker_text = "; ".join(
                    f"{item.get('column', '')}="
                    f"{item.get('raw_value', '')}"
                    f"[{item.get('canonical_value', '')}]"
                    for item in attempt.get("linker_values") or [])
                logger.warning(
                    "unsupported bridge chain dropped for %r (%s → %s; "
                    "rank %s; linkers %s; source pool %s; target pool %s; "
                    "%s)", key, src, tgt, attempt.get("rank"),
                    linker_text or "none", attempt.get("source_pool_size"),
                    attempt.get("target_pool_size"),
                    attempt.get("reason") or attempt.get("status"))
        if result.get("resolution_status") != "supported":
            continue
        pools[key] = result


def _multivalue_marker(mapper, type_name: str, dataset: str) -> str:
    """`🧩 multi (a|b)` marker when the release's own `type` cell lists
    several candidate names, or "" when the name is a normal single name.

    Display-only (plan-type-column-multivalue-normalization Stage 1 /
    plan-ui-type-mapper-alignment §4.3): the cell stays an atomic type name;
    the marker only tells the user the name is composite in the release.
    """
    try:
        parts = mapper.multivalue_parts(type_name, dataset)
    except Exception:
        parts = None
    if not parts:
        return ""
    return f" 🧩 multi ({'|'.join(str(p) for p in parts)})"


# The Suspects badge's hover text: the explanation the column header
# carries, then one line per rival candidate.  The collapsed Suspects
# blocks below the tables stay exactly as they were — a hover is never the
# only route to the evidence (plan-ui-type-mapper-alignment §4.1/D1).
_SUSPECTS_EXPLANATION = (
    "same-name-first selection: the fan-out's candidate set contained the "
    "source type's own name, so that candidate was selected and the rival "
    "candidate(s) below were NOT. An observation about the crosswalk, not a "
    "biological verdict — rivals are never merged. Add a custom label "
    "mapping if one belongs to your analysis.")

# Quasar's per-column cell slot is named body-cell-<column name>.  The badge
# keeps its visible text; the hover covers the whole cell.
_SUSPECTS_CELL_SLOT = (
    '<q-td :props="props">'
    '{{ props.value }}'
    '<q-tooltip v-if="props.value && props.row.suspects_tip" '
    'class="drocat-map-cell-tooltip" anchor="top middle" '
    'self="bottom middle">{{ props.row.suspects_tip }}</q-tooltip>'
    '</q-td>'
)


def _suspects_tip(groups) -> str:
    """One Suspects hover block, or "" when nothing is flagged.

    ``groups`` is ``[(heading, rival_rows), ...]`` where ``rival_rows`` are
    the rows ``_suspects_by_pair`` builds.  The heading names the type pair a
    rival belongs to, so the surfaces that aggregate several suspect pairs
    into one row (both coverage tables, the per-type breakdown) still read
    one rival per line.
    """
    lines = [f"{heading}{row['rival']} — own 1-to-1 pair {row['own_pair']}, "
             f"votes {row['votes']}, reverse {row['reverse']}, status "
             f"{row['status']}, neurons src/tgt {row['counts']}"
             for heading, rows in groups for row in (rows or [])]
    if not lines:
        return ""
    return "\n".join([_SUSPECTS_EXPLANATION, *lines])


_HELD_REASON_CACHE: Dict[Tuple[int, str, str, str], str] = {}


def _held_same_name_reason(mapper, type_name: str, origin: str,
                           target: str) -> str:
    """Why a same-name fan-out was NOT selected, or "" when not applicable.

    Reads the mapper's own verdict (``same_name_first_fires``); used to
    explain an orphan that is really a HELD pair.  Observation-only wording:
    the mapper never verifies, so this states which rivals lack a 1-to-1
    pairing of their own — never a verdict about them.  Verdicts are cached
    per (mapper id, type, origin, target) — the probe behind this runs the
    unindexed conflict scan, and the panel calls it per orphan per target.
    """
    key = (id(mapper), type_name, origin, target)
    hit = _HELD_REASON_CACHE.get(key)
    if hit is not None:
        return hit
    result = _held_same_name_reason_uncached(
        mapper, type_name, origin, target)
    _HELD_REASON_CACHE[key] = result
    return result


def _held_same_name_reason_uncached(mapper, type_name: str, origin: str,
                                    target: str) -> str:
    try:
        fired = mapper.same_name_first_fires(type_name, origin, target)
    except Exception:
        return ""
    if not fired or fired.get("disposition") != "gated_held":
        return ""
    rivals = list(fired.get("rivals") or ())
    n_no_pair = 0
    try:
        for rival in rivals:
            if not mapper._rival_has_own_clean_pair(rival, origin, target):
                n_no_pair += 1
    except Exception:
        n_no_pair = 0
    if rivals and n_no_pair:
        return (f"kept unmapped: {n_no_pair} of {len(rivals)} rival "
                f"candidate(s) have no 1-to-1 pair of their own, so the "
                f"same-name candidate '{fired.get('selected')}' was not "
                f"selected")
    return (f"kept unmapped: the same-name candidate "
            f"'{fired.get('selected')}' was not selected")


def _describe_orphan(entry: Dict[str, Any], target: str) -> str:
    """One orphan summary line: the type, its neuron count, and any
    crosswalk claim the target dataset cannot fulfil."""
    text = f"{entry['type']} ({entry['count']})"
    claimed = [c for c in (entry.get("claimed") or []) if c]
    if claimed:
        quoted = ", ".join(f"'{c}'" for c in claimed)
        text += (f" — auto-mapping claims {quoted} but {target} "
                 f"has no such neurons")
    held = str(entry.get("held_reason") or "")
    if held:
        text += f" — {held}"
    return text


def _compute_type_mapping(queries, datasets, mode,
                          route_scope: str = 'curated') -> Dict[str, Any]:
    """Compute the global type mapping without touching NiceGUI state.

    This is deliberately a module-level, pickle-safe callback.  A cold
    mapper load uses it in a dedicated spawned process so pandas parsing and
    the Python graph build cannot starve the websocket event loop.

    ``route_scope`` (§full-map, user 2026-09-28): ``'curated'`` (default —
    today's behavior, unchanged) or ``'full'`` — the panel-layer
    composition of the transitive reach through candidate intermediate
    datasets (:func:`compose_full_map_reach`).  The composition NEVER
    touches the mapper's derivation/decision surfaces: claim-tier figures
    are identical in both scopes, and the mapper->TM VEV interface cannot
    see the full-map view by construction.
    """
    if route_scope not in ('curated', 'full'):
        raise ValueError(f"unknown route_scope: {route_scope!r}")
    from ..neuron_index import (
        collect_native_type_matches,
        count_types_in_index,
        enrich_native_type_matches,
        _load_cross_match_index,
        _load_coverage_index,
        mapped_type_targets,
        resolve_type_matches,
    )
    from comparison.cross_dataset_type_mapper import get_type_mapper
    from comparison.mapping_visualization import (
        build_mapping_flows,
        dedupe_mirrored_pairs,
        origin_seeded_flows,
        render_composed_mapping_html,
        _pair_relationship,
    )

    try:
        mapper = get_type_mapper()
    except Exception:
        mapper = None

    # Type mapping is a type-level operation. Keep bodyId/linker columns out
    # of the resolver and its counts; they are only needed later to annotate
    # a rendered type edge with coverage.
    indexes = {ds: _load_cross_match_index(ds) for ds in datasets}
    if any(indexes.get(ds) is None for ds in datasets):
        raise RuntimeError(
            "one or more selected type indexes could not be read")

    # Coverage is intentionally separate from type matching. These compact
    # projections contain only bodyId, type, and linker columns used to
    # report an m-of-n coverage subset.
    coverage_indexes: Dict[str, Any] = {}

    def _coverage_indexes_for(src: str, tgt: str) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        for dataset in (src, tgt):
            if dataset not in coverage_indexes:
                coverage_indexes[dataset] = _load_coverage_index(dataset)
            index = coverage_indexes[dataset]
            if index is not None:
                selected[dataset] = index
        return selected

    # §12: resolve every chip under the ACTIVE MATCH MODE (exact /
    # startswith / contains / endswith / regex) against each selected
    # dataset's type column; no-hit chips fall back to the staged native
    # search (labels) below.
    resolved = resolve_type_matches(queries, mode, datasets, indexes)
    origins = resolved["origins"]
    origin_matches = resolved.get("origin_matches", {})
    notes = list(resolved["notes"])

    pair_flows: Dict[tuple, list] = {}
    pools: Dict[tuple, Dict[str, Any]] = {}

    # Explicit modes: map FROM each origin dataset (the query lives where it
    # matched) INTO every other selected dataset.
    for origin in sorted(origins):
        o_types = origins[origin]
        source_counts = count_types_in_index(indexes[origin], o_types)
        for target in datasets:
            if target == origin:
                continue
            flows = origin_seeded_flows(
                origin, o_types, target, source_counts=source_counts,
                matched_origins=origin_matches.get(origin),
                route_scope=route_scope)
            if not flows:
                continue
            ends = sorted({f["foreign_type"] for f in flows})
            f_counts = count_types_in_index(indexes[target], ends)
            for flow in flows:
                flow["foreign_count"] = f_counts.get(
                    flow["foreign_type"], 0)
            pair_flows[(origin, target)] = \
                pair_flows.get((origin, target), []) + flows
            _pool_mapping_pair(
                flows, origin, target,
                _coverage_indexes_for(origin, target), pools)

    # Zero-hit fallback chips: the staged native sweep (substring types +
    # taxonomy labels → pooled nodes).
    for chip in resolved["fallback_chips"]:
        for ds in datasets:
            index = indexes[ds]
            entries_by_foreign: Dict[str, list] = {}
            for entry in collect_native_type_matches(
                    ds, chip, datasets=datasets, uncapped=True):
                foreign = entry.get("dataset", "")
                if foreign == ds or foreign not in datasets:
                    continue
                entries_by_foreign.setdefault(foreign, []).append(entry)
            flat = [e for entries in entries_by_foreign.values()
                    for e in entries]
            if flat:
                enrich_native_type_matches(flat, ds, route_scope=route_scope)
            for foreign, entries in entries_by_foreign.items():
                source_counts = count_types_in_index(index, [
                    t for e in entries
                    for t in e.get("mapped_type_names", [])])
                flows = build_mapping_flows(
                    entries, ds, source_counts=source_counts,
                    route_scope=route_scope)
                if not flows:
                    continue
                pair_flows[(ds, foreign)] = \
                    pair_flows.get((ds, foreign), []) + flows
                _pool_mapping_pair(
                    flows, ds, foreign,
                    _coverage_indexes_for(ds, foreign), pools)

    # One canonical entry per unordered pair (§12 mirror dedupe).
    pair_flows = dedupe_mirrored_pairs(pair_flows, origins.keys())

    # Ratified claim-set accounting (plan-same-name-fidelity-and-three-
    # level-coverage.md): the mapping's bodyId-level claim is the union of
    # the branches' resolved target pools — NOT the full populations of the
    # received types (those are the reference denominator; the population
    # overhang is TM-EVE family material).
    #
    # Only flows the scoped decision ADOPTED contribute (``flow_is_claimed``).
    # The pair list also carries disclosure rows — a same-name rival the
    # mapper declined, a valid-split fan-out it did not adopt — and counting
    # their pools made the panel and the TM VEV report publish two different
    # mappings for one query (205 vs 198 on circadian_clock → BANC).
    from comparison.mapping_visualization import (
        flow_is_claimed, mapping_pool_key)
    claimed_by_ds: Dict[str, set] = {}
    # Per (source dataset, source type, TARGET dataset): the per-type
    # breakdown reports dataset-specific claim sets — never a sum across
    # the target datasets (user 2026-09-14).
    claimed_by_type: Dict[Tuple[str, str, str], set] = {}
    # OUT-MAP (user 2026-09-26): the received types' own neurons that the
    # claim set does NOT reach.  Both halves are bodyId SETS — the pool's
    # `target_type_body_ids` (the full population of the endpoint type in its
    # OWN dataset) minus the claim — so the count is a set difference and can
    # never be a sum of per-type counts that a multivalue cell would
    # double-count.  A `full population` basis claims its whole type, so it
    # contributes 0 by construction.
    #
    # This is the PANEL's figure and deliberately NOT the validation
    # pipeline's `family` bin.  TM VEV bins out-map rows per BRANCH and lets a
    # morph-qualified candidate CLOSE a hole; on circadian_clock → MCNS the
    # panel therefore reads 219 − 204 = 15 where family_candidates.csv holds
    # 11.  Every surface that shows this number says which population it is.
    pop_by_target_type: Dict[Tuple[str, str], set] = {}
    claimed_for_target_type: Dict[Tuple[str, str], set] = {}
    # §three-tier readout (user 2026-09-27): alongside the claim tier,
    # accumulate the EVIDENCE-REACH tier — every flow's pools, including
    # the DISCLOSURE rows the decision declined (a same-name rival, a
    # vote-declined annotation pair).  The reach tier is what the CSVs'
    # all-valid scopes and the coverage tables already publish; the strip
    # now states both so the readout cannot be mistaken for one number.
    reach_by_ds: Dict[str, set] = {}
    reach_by_type: Dict[Tuple[str, str, str], set] = {}
    reach_types_by_ds: Dict[str, set] = {}
    claim_types_by_ds: Dict[str, set] = {}
    disclosure_by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for (src, tgt), flows in pair_flows.items():
        for f in flows:
            claimed = flow_is_claimed(f)
            key = mapping_pool_key(src, tgt, f.get("source_type"),
                                   f.get("foreign_type"))
            pool = pools.get(key)
            tb = (pool or {}).get("target_body_ids") or []
            tkey = (tgt, str(f.get("foreign_type") or ""))
            if claimed:
                # str-normalized on both sides: the same ids the card and
                # the CSV compare, and it cannot raise on a local release's
                # id spelling
                pop_by_target_type.setdefault(tkey, set()).update(
                    str(b) for b in ((pool or {}).get("target_type_body_ids")
                                     or []))
                claimed_for_target_type.setdefault(tkey, set()).update(
                    str(b) for b in tb)
                if not tb:
                    continue
                claim_types_by_ds.setdefault(tgt, set()).add(
                    str(f.get("foreign_type") or ""))
                claimed_by_ds.setdefault(tgt, set()).update(
                    int(b) for b in tb)
                claimed_by_type.setdefault(
                    (src, f.get("source_type"), tgt), set()).update(
                    int(b) for b in tb)
            # reach tier: adopted and disclosure alike
            if f.get("foreign_type"):
                reach_types_by_ds.setdefault(tgt, set()).add(
                    str(f["foreign_type"]))
            if not tb:
                continue
            reach_by_ds.setdefault(tgt, set()).update(int(b) for b in tb)
            reach_by_type.setdefault(
                (src, f.get("source_type"), tgt), set()).update(
                int(b) for b in tb)
            if not claimed:
                snf = f.get("same_name_first") or {}
                if f.get("suspects") and snf.get("selected"):
                    reason = (f"same-name-first rival — "
                              f"'{snf['selected']}' selected instead")
                elif (f.get("mapping_status") == "conflict"):
                    reason = "conflict — fail-closed"
                else:
                    reason = ("fan-out branch not adopted by the decision "
                              f"({f.get('mapping_status') or 'evidence'})")
                disclosure_by_ds.setdefault(tgt, []).append({
                    "type": str(f.get("foreign_type") or ""),
                    "bodies": len({int(b) for b in tb}),
                    "reason": reason,
                })

    def _out_map_ids(target_dataset: str, foreign_type: str) -> set:
        """The out-map bodyId set of one received type in one dataset."""
        tkey = (target_dataset, str(foreign_type or ""))
        return (pop_by_target_type.get(tkey) or set()) - (
            claimed_for_target_type.get(tkey) or set())

    # Per dataset the union over its received types; equal to
    # (union of received-type populations) − (claim union), because every
    # claim is a subset of its own type's population.
    out_map_by_ds: Dict[str, set] = {}
    for (tgt, foreign) in pop_by_target_type:
        out_map_by_ds.setdefault(tgt, set()).update(
            _out_map_ids(tgt, foreign))

    # §12.3: dataset-wide incoming context for the backward coverage
    # table — only for receiving types already present in the result,
    # bounded, and cached per pair.  Explanatory evidence only; it never
    # changes the mapper's accepted flows or the forward rows.
    reverse_contexts: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    if mapper is not None and getattr(mapper, "_loaded", False):
        from ..neuron_index import build_reverse_type_contexts
        for (src, tgt), flows in pair_flows.items():
            receiving = sorted({f.get("foreign_type", "") for f in flows
                                if f.get("foreign_type")})
            if not receiving:
                continue
            try:
                reverse_contexts[(src, tgt)] = build_reverse_type_contexts(
                    mapper, src, tgt, receiving,
                    source_index=indexes.get(src),
                    target_index=indexes.get(tgt),
                    coverage_indexes=_coverage_indexes_for(src, tgt),
                    warm_pools=pools)
            except Exception:
                logger.warning(
                    "reverse coverage context failed for %s → %s",
                    src, tgt, exc_info=True)

    # W3 orphans: queried/expanded types with NO mapped counterpart in a
    # specific target dataset stay visible.  ``claimed`` is filled in below
    # when the resolver names a counterpart the target dataset does not have.
    orphans: Dict[tuple, List[Dict[str, Any]]] = {}
    orphan_by_key: Dict[tuple, Dict[str, Any]] = {}
    for origin in sorted(origins):
        o_types = origins[origin]
        flowed_sources = {
            f.get("source_type")
            for (s, _t), fl in pair_flows.items() if s == origin
            for f in fl
        }
        for target in datasets:
            if target == origin:
                continue
            missing = [t for t in o_types if t not in flowed_sources]
            if not missing:
                continue
            counts = count_types_in_index(indexes[origin], missing)
            orphans[(origin, target)] = []
            for t in missing:
                entry = {"dataset": origin, "type": t,
                         "count": counts.get(t, 0), "target": target,
                         "claimed": []}
                # Same-name-first HELD pairs land here as bare orphans
                # (build_mapping_flows skips conflicts).  Record WHY so the
                # orphan line can explain it instead of implying the type
                # simply has no counterpart (plan-ui-type-mapper-alignment
                # §4.2; boundary-clean wording — an observation, no verdict).
                reason = _held_same_name_reason(mapper, t, origin, target)
                if reason:
                    entry["held_reason"] = reason
                orphans[(origin, target)].append(entry)
                orphan_by_key[(origin, target, t)] = entry

    # Per-dataset summary strip.  Mapped neurons count each target type once;
    # the old per-flow sum double-counted shared targets.  A resolver target
    # only counts as mapped when the target dataset actually has neurons of
    # that type — a crosswalk claim naming an absent type is reported on the
    # orphan entry instead of inflating the target's mapped counts.
    # Resolver-half orphan accounting: a resolver claim naming a type the
    # target dataset lacks registers an orphan so the claim renders.  (The
    # resolver's PRESENT licensed targets no longer feed the summary's
    # mapped figures — since the 2026-09-27 three-tier readout the Mapped
    # cell is pure claim basis and the resolver-licensed-but-declined
    # names surface through the Evidence reach column instead.)
    if mapper is not None and getattr(mapper, "_loaded", False):
        # One cache pair per compute: the resolver memoizes alias
        # candidates and bridge chains per (type, source, target), so a
        # broad query reuses evidence instead of re-walking the bridge
        # graph for every chip match.
        alias_cache: Dict[Any, Optional[Dict[str, Any]]] = {}
        bridge_cache: Dict[Any, List] = {}
        for origin in sorted(origins):
            for target in datasets:
                if target == origin:
                    continue
                for otype in origins[origin]:
                    ann = mapped_type_targets(
                        mapper, otype, origin, target,
                        alias_cache=alias_cache, bridge_cache=bridge_cache)
                    targets = [t for t in ((ann or {}).get("targets") or [])
                               if t]
                    if not targets:
                        continue
                    t_counts = count_types_in_index(indexes[target], targets)
                    absent = [t for t in targets if not t_counts.get(t, 0)]
                    if absent:
                        entry = orphan_by_key.get((origin, target, otype))
                        if entry is None:
                            # The type has flows into this target (so the
                            # no-flow rule above skipped it), but the
                            # resolver still claims a counterpart the
                            # target dataset lacks.  Register a real
                            # orphan entry so the claim renders — a bare
                            # detached dict would silently drop it.
                            entry = {"dataset": origin, "type": otype,
                                     "count": count_types_in_index(
                                         indexes[origin],
                                         [otype]).get(otype, 0),
                                     "target": target, "claimed": []}
                            orphans.setdefault(
                                (origin, target), []).append(entry)
                            orphan_by_key[(origin, target, otype)] = entry
                        entry["claimed"].extend(absent)

    summary = []
    for ds in datasets:
        matched = origins.get(ds, [])
        neurons = (sum(count_types_in_index(
            indexes[ds], matched).values()) if matched else 0)
        # §three-tier (2026-09-27): the MAPPED cell is PURE claim basis —
        # its type count is the adopted types whose pools actually claimed
        # bodyIds, so bodies and types describe one and the same set (the
        # old count mixed in resolver-licensed names the decision
        # declined).  The reach tier carries the full evidence touch.
        claim_types = claim_types_by_ds.get(ds, set())
        recv_neurons = len(claimed_by_ds.get(ds, set()))
        reach_counts = (count_types_in_index(
            indexes[ds], sorted(reach_types_by_ds.get(ds, set())))
            if reach_types_by_ds.get(ds) else {})
        reach_present = {t for t, c in reach_counts.items() if c}
        reach_neurons = len(reach_by_ds.get(ds, set()))
        disclosure = disclosure_by_ds.get(ds, [])
        # exclusive disclosure = reach − claim (bodies the evidence touches
        # but the decision did not adopt); the per-flow detail list carries
        # the decline reasons for the hover
        disclosure_bodies = reach_neurons - recv_neurons
        disclosure_types = len(reach_present - claim_types)
        out_map = len(out_map_by_ds.get(ds, set()))
        unmapped = len({e["type"] for (s, _t), v in orphans.items()
                        if s == ds for e in v})
        summary.append({
            "dataset": ds,
            "types": len(matched),
            "neurons": neurons,
            "mapped_types": len(claim_types),
            "mapped_neurons": recv_neurons,
            "mapped": _format_mapped_neurons(recv_neurons,
                                             len(claim_types)),
            "reach_types": len(reach_present),
            "reach_neurons": reach_neurons,
            "reach": _format_mapped_neurons(reach_neurons,
                                            len(reach_present)),
            "disclosure_types": disclosure_types,
            "disclosure_bodies": disclosure_bodies,
            "disclosure_detail": disclosure,
            "out_map": out_map,
            "unmapped": unmapped,
        })

    # R5: the same metrics per (matched type, TARGET dataset) for the
    # collapsed per-type breakdown — the mapped counts are specific to
    # the row's target dataset, never summed across datasets (user
    # 2026-09-14).  Derived from the structures above — no new mapper
    # walks, no profiling.  ``query`` carries the chip provenance when
    # the resolver recorded the match (multi-type expansions).
    chip_of: Dict[tuple, str] = {}
    for ds_m, type_matches in (origin_matches or {}).items():
        for t_name, matches in (type_matches or {}).items():
            for m in (matches or [])[:1]:
                v = str((m or {}).get("value") or "").strip("'")
                if v:
                    chip_of[(ds_m, t_name)] = v
    summary_per_type: List[Dict[str, Any]] = []
    # Per-pair sources-by-target map so a row's relationship can read N-to-1
    # when several queried types converge on one target (built once per pair).
    _s_by_t: Dict[tuple, Dict[str, set]] = {}
    for (ds, tgt), fl in pair_flows.items():
        m = _s_by_t.setdefault((ds, tgt), {})
        for f in fl:
            m.setdefault(str(f.get("foreign_type") or ""), set()).add(
                str(f.get("source_type") or ""))
    for ds in datasets:
        matched = origins.get(ds, [])
        ds_counts = (count_types_in_index(indexes[ds], matched)
                     if matched else {})
        for t in sorted(matched):
            for tgt in datasets:
                if tgt == ds:
                    continue
                tgt_flows = [f for f in pair_flows.get((ds, tgt), [])
                             if f.get("source_type") == t]
                ftypes = sorted({f.get("foreign_type") for f in tgt_flows
                                 if f.get("foreign_type")})
                # The row's MAPPED columns count adopted ends only (the fan-
                # out above stays visible in the pair cards and the
                # relationship cell, which describe evidence, not claims).
                claim_ftypes = sorted({
                    f.get("foreign_type") for f in tgt_flows
                    if f.get("foreign_type") and flow_is_claimed(f)})
                tgt_counts = (count_types_in_index(indexes[tgt], ftypes)
                              if ftypes else {})
                present_types = [tt for tt in claim_ftypes
                                 if tgt_counts.get(tt)]
                claimed = claimed_by_type.get((ds, t, tgt), set())
                orphaned = sum(1 for e in orphans.get((ds, tgt), [])
                               if e.get("type") == t)
                _pair_map = _s_by_t.get((ds, tgt), {})
                relationship = _pair_relationship(
                    len(ftypes),
                    max((len(_pair_map.get(ft, ())) for ft in ftypes),
                        default=0))
                suspect_n = sum(1 for f in tgt_flows if f.get("suspects"))
                # Per-type out-map for this row: each accepted target type's
                # population minus the claims reaching THAT TYPE (a union
                # across the adopted branches that share it, so under N-to-1
                # convergence this is narrower than a per-branch read — the
                # validation pipeline bins `family` per BRANCH). Summed over
                # the row's accepted types, which is why these rows can total
                # more than the deduped dataset figure (17 vs 15).
                out_map_n = sum(len(_out_map_ids(tgt, ft))
                                for ft in claim_ftypes)
                # §three-tier: the row's REACH figures — all flows' pools
                # toward this row's target dataset, disclosure ends
                # included (the same claim-vs-reach pairing as the strip)
                reach_ids = reach_by_type.get((ds, t, tgt), set())
                reach_ftypes = [tt for tt in ftypes if tgt_counts.get(tt)]
                summary_per_type.append({
                    "dataset": ds,
                    "type": t,
                    "query": chip_of.get((ds, t), ""),
                    "neurons": ds_counts.get(t, 0),
                    "target": tgt,
                    "relationship": relationship,
                    "suspects": suspect_n,
                    "mapped_types": len(present_types),
                    "mapped_neurons": len(claimed),
                    "mapped": _format_mapped_neurons(
                        len(claimed), len(present_types)),
                    "reach_types": len(reach_ftypes),
                    "reach_neurons": len(reach_ids),
                    "reach": _format_mapped_neurons(
                        len(reach_ids), len(reach_ftypes)),
                    "out_map": out_map_n,
                    "unmapped": orphaned,
                })

    html, meta = render_composed_mapping_html(
        pair_flows, pools=pools, node_cap=COMPOSED_NODE_CAP,
        # the selection order breaks a star layout's flank tie
        dataset_order=datasets)
    meta = dict(meta or {})
    meta["notes"] = notes + list(meta.get("notes", []))

    # §full-map route scope (user 2026-09-28; RE-RATIFIED 2026-10-01 as a
    # full parallel mode): the flows above were already seeded from the
    # mode's route universe (origin_seeded_flows / enrich carry
    # route_scope); this block composes the per-pair summary the strip
    # cell, the breakdown column and the full-map CSV render from.
    full_map: Dict[tuple, Dict[str, Any]] = {}
    if route_scope == 'full' and mapper is not None \
            and getattr(mapper, "_loaded", False):
        for (src, tgt), flows in pair_flows.items():
            source_types = sorted({f.get('source_type') for f in flows
                                   if f.get('source_type')})
            if not source_types:
                continue
            try:
                full_map[(src, tgt)] = compose_full_map_reach(
                    mapper, src, tgt, source_types)
            except Exception as exc:  # noqa: BLE001 — advisory view
                logger.warning(
                    "full-map composition failed for %s -> %s: %s",
                    src, tgt, exc)
            else:
                info = full_map[(src, tgt)]
                _to = len(info.get('transitive_only_ends', []))
                meta["notes"].append(
                    f"[full map] {src} -> {tgt}: composed reach "
                    f"{len(info.get('ends', []))} ends "
                    f"(+{_to} transitive-only via "
                    f"{', '.join(info.get('mids') or [])}) — full-map "
                    "mode; the validation pipeline accepts this scope "
                    "gated (MappingValidationConfig.route_scope).")
        # post-fill the per-type breakdown rows with what full mode ADDS
        # for each source type (matched surfaces): the transitive-only
        # ends attributed to the row's source type, their connectors, and
        # the cross-mid conflict count.
        for row in summary_per_type:
            pair_key = (row.get('dataset'), row.get('target'))
            info = full_map.get(pair_key)
            if not info:
                continue
            src_type = str(row.get('type') or '')
            transitive_ends = set()
            mids_for_type = set()
            for end in info.get('transitive_only_ends', []):
                for r in (info.get('routes', {}).get(end) or {}).get(
                        'transitive', []):
                    if r.get('source_type') == src_type:
                        transitive_ends.add(end)
                        if r.get('mid'):
                            mids_for_type.add(r['mid'])
            conflicts_for_type = [
                c for c in (info.get('route_conflicts') or [])
                if c.get('source_type') == src_type]
            row['fullmap_reach'] = len(transitive_ends) + int(
                row.get('reach_types') or 0)
            row['fullmap_delta'] = len(transitive_ends)
            row['fullmap_mids'] = sorted(mids_for_type)
            row['fullmap_conflicts'] = len(conflicts_for_type)
            row['fullmap_cell'] = (
                f"{row['fullmap_reach']} (+{row['fullmap_delta']} via "
                f"{', '.join(row['fullmap_mids'])})"
                + (f" ⚠{row['fullmap_conflicts']}"
                   if row['fullmap_conflicts'] else ''))
        # strip cells (matched surfaces): per received dataset, the
        # composed end count and the transitive-only delta across the
        # pairs pointing into it — an ADDITIVE labeled cell; the ratified
        # Evidence reach cell above stays untouched.
        for row in summary:
            tgt_ds = row.get('dataset')
            ends = set()
            transitive = set()
            mids = set()
            for (src, tgt), info in full_map.items():
                if tgt != tgt_ds:
                    continue
                ends.update(info.get('ends', []))
                transitive.update(info.get('transitive_only_ends', []))
                mids.update(info.get('mids') or [])
            row['fullmap'] = (
                f"{len(ends)} (+{len(transitive)} via "
                f"{', '.join(sorted(mids))})") if ends else ''

    return {"pair_flows": pair_flows, "pools": pools, "meta": meta,
            "composed": html, "datasets": datasets,
            "summary": summary, "summary_per_type": summary_per_type,
            "queries": [str(q) for q in (queries or [])],
            "orphans": orphans,
            "reverse_contexts": reverse_contexts,
            "route_scope": route_scope,
            "full_map": full_map}


def create_type_mapping_entry(get_datasets: Callable[[], list]):
    """The entrance button + preview dialog (spec §3).

    Returns the button element; the tab wires ``refresh_state()`` into
    the dataset selector so the disabled state follows the selection
    (disabled with a tooltip until >= 2 selected datasets have cached
    neuron indexes).
    """
    from ..neuron_index import neuron_index_path

    state: Dict[str, Any] = {"pair_flows": {}, "pools": {}, "meta": {},
                             "composed": None, "datasets": [],
                             "summary": [], "orphans": {},
                             "reverse_contexts": {},
                             # (source dataset, target dataset) -> per-rival
                             # rows, rebuilt on every search by _apply
                             "suspect_cache": {}}

    def _ready() -> bool:
        datasets = list(get_datasets() or [])
        return len(datasets) >= 2 and all(
            neuron_index_path(ds).is_file() for ds in datasets)

    # Same fixed window as the 'See available neurons' viewer: the backdrop
    # never dismisses it, the card is viewport-bounded with internal scroll,
    # and the corner 'x' is the only way out.
    dialog = ui.dialog().props("persistent")
    with dialog, \
            ui.card().classes(
                "w-[min(98vw,1800px)] max-w-none drocat-neuron-viewer-card"
            ):
        with ui.row().classes(
            "w-full items-center justify-between gap-2 drocat-neuron-dialog-header"
        ):
            ui.label("Auto type mapping preview").classes(
                "text-h6 drocat-neuron-dialog-title")
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense")
        with ui.column().classes("w-full gap-2 drocat-neuron-viewer-content"):
            ui.label(
                "Preview of the auto type mapping the cross-dataset analysis "
                "will use — informational only, please double check."
            ).classes("text-caption drocat-muted")
            def _suggest(text: str):
                """Dataset-aware suggestions — the SAME staged semantics as
                the tab's query boxes: types first, widening to the metadata
                columns when no type matched, so typing 'circadian' offers
                'circadian_clock' (FAFB) and 'circadian_neuron' (BANC) here
                too."""
                from ..type_suggestions import dataset_aware_suggestions

                return dataset_aware_suggestions(
                    text, list(get_datasets() or []), "auto", limit=None)

            search_action: Dict[str, Any] = {}

            def _add_search_action() -> None:
                # The Search button lives IN the input row (right of the
                # Match-by select) so query box, filter and action share
                # one toolbar row (user 2026-09-07). self-stretch keeps it
                # at the row's height; lazy dispatch: _run_click is defined
                # below the dialog build (NiceGUI schedules the returned
                # coroutine as a task).
                search_action["button"] = ui.button(
                    "Search mappings", icon="search",
                    on_click=lambda: _run_click()
                ).classes("self-stretch")

            search = neuron_list_input(
                label="Types to map",
                placeholder="e.g. APDN3, aMe.* — one query per chip",
                unit_label="query",
                show_upload=False,
                suggestions=_suggest,
                # Same height as the query box, and the Search button
                # slots in right of it — the chip input narrows to make
                # room (user 2026-09-07).
                filter_dense=False,
                input_actions=_add_search_action,
                # Standalone history: the panel's Recent/Frequent list reads
                # and writes its own store, so panel searches never mix with
                # the analysis tabs' neuron-query history.
                history_kind="type_mapping",
                # History rows annotated like suggestion rows: the matched
                # column and dataset(s) ("type · flywire_FAFB_v783") from
                # the current selection's local pools.
                history_hint_datasets=lambda: list(get_datasets() or []),
                hint="One query per chip. The match modes match the standard "
                     "query (exact / starts with / contains / ends with / "
                     "regex); dataset-aware type suggestions from the "
                     "selected datasets appear as you type. This box keeps "
                     "its own query history, separate from the tabs.",
            ).classes("w-full")
            search_btn = search_action["button"]
            # §full-map route scope (user 2026-09-28): Curated (default) =
            # today's behavior; Full map composes the transitive reach
            # through candidate intermediate datasets — PANEL-LAYER ONLY
            # (claim tier and the mapper->TM VEV interface are untouched).
            route_scope_state: Dict[str, str] = {"value": "curated"}
            ui.select(
                ["curated", "full"],
                value="curated",
                label="Route scope",
                on_change=lambda e: route_scope_state.__setitem__(
                    "value", str(e.value or "curated")),
            ).props("outlined dense").classes(
                "drocat-select min-w-[130px]"
            ).tooltip(
                "Curated: the licensed derivation routes (default). "
                "Full map: additionally compose transitive reach through "
                "candidate intermediate datasets — evidence/display only, "
                "the claim tier and the validation pipeline are never "
                "extended by it.")
            # Loading notice: painted BEFORE the heavy search leaves the
            # event loop (the first search warms a large index and can take
            # a minute or two — silent freezing looked like a hang).
            loading_row = ui.row().classes("items-center gap-2")
            with loading_row:
                ui.spinner("dots", size="md", color="primary")
                ui.label("Loading the selected datasets' type indexes and "
                         "mapping — the first search can take a minute or "
                         "two. Please wait …").classes(
                    "text-caption drocat-muted")
            loading_row.set_visibility(False)
            notes_label = ui.label("").classes("text-caption drocat-muted")
            results = ui.column().classes("w-full")

    def _deliver(html: str, name: str) -> None:
        ui.download.content(html, name, "text/html")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _deliver_composed_sankey(stamp: str) -> None:
        """The cross-dataset Sankey: one column per dataset, the query's
        origin in the middle.  A Sankey ribbon has to start somewhere, so
        this exists only for one origin into one or two targets — beyond
        that the per-pair Sankey buttons in the pair cards are the route."""
        from comparison.mapping_visualization import (
            render_composed_sankey_html)

        html, _info = render_composed_sankey_html(
            state["pair_flows"], pools=state["pools"],
            dataset_order=state.get("datasets") or [])
        if not html:
            ui.notify("The composed Sankey covers one source dataset and up "
                      "to two target datasets — for this selection use a "
                      "pair card's 'Sankey (type-level)'.", type="info")
            return
        _deliver(html, f"mapping_sankey_{stamp}.html")

    def _deliver_flows(src: str, tgt: str, flows, pools,
                       kind: str, variant: str, stamp: str) -> None:
        from comparison.mapping_visualization import (
            render_bridge_linker_html,
            render_mapping_network_html,
            render_mapping_sankey_html,
        )
        foreign = tgt.replace(":", "_")
        name = f"mapping_{kind}_{variant}_{foreign}_{stamp}.html"
        orphans = (state.get("orphans") or {}).get((src, tgt))
        if kind == "sankey":
            html = render_mapping_sankey_html(flows, pools=pools,
                                              variant=variant,
                                              orphans=orphans)
        elif variant == "linker":
            html = render_bridge_linker_html(
                flows, source_dataset=src, target_dataset=tgt, pools=pools)
        else:
            html = render_mapping_network_html(flows, pools=pools,
                                               orphans=orphans)
        if not html:
            ui.notify("Nothing to visualize.", type="info")
            return
        _deliver(html, name)

    def _deliver_pair_csv(src: str, tgt: str, flows, pools,
                          stamp: str) -> None:
        from comparison.mapping_visualization import build_bridges_csv

        text = build_bridges_csv(flows, pools=pools, extended=True)
        if not text:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"mapping_{src.replace(':', '_')}_{tgt.replace(':', '_')}_{stamp}.csv"
        ui.download.content(text, name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _deliver_fullmap_csv(src: str, tgt: str, stamp: str) -> None:
        """Full-map mode (2026-10-01): one row per composed route —
        source type, connector (mid), via type, both leg chains, the
        composed end, its class, and the cross-mid conflicts — so the
        transitive universe is auditable beside the curated CSVs."""
        import csv as _csv
        import io

        info = (state.get("full_map") or {}).get((src, tgt)) or {}
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["source_dataset", "source_type", "connector",
                    "via_type", "class", "target_dataset", "target_type",
                    "leg_a", "leg_b"])

        def _chain_text(chain) -> str:
            return " -> ".join(
                f"{h.get('dataset')}:{h.get('column')}={h.get('value')}"
                for h in (chain or []))

        direct_ends = set(info.get("direct_ends") or [])
        rows_written = 0
        for end in info.get("ends", []):
            bucket = (info.get("routes") or {}).get(end) or {}
            klass = ("corroborated" if end in direct_ends
                     else "transitive_only")
            if bucket.get("direct") and not bucket.get("transitive"):
                continue  # curated-only end: the ordinary CSVs carry it
            for r in bucket.get("transitive", []):
                for leg_a in r.get("leg_a") or [None]:
                    w.writerow([
                        src, r.get("source_type", ""), r.get("mid", ""),
                        r.get("via_type", ""), klass, tgt, end,
                        _chain_text(leg_a),
                        _chain_text((r.get("leg_b") or [None])[0]),
                    ])
                    rows_written += 1
        for c in info.get("route_conflicts", []):
            w.writerow([
                src, c.get("source_type", ""),
                f"{c.get('mid_a')}|{c.get('mid_b')}", c.get("source_type",
                                                            ""),
                "route_conflict", tgt,
                f"{','.join(c.get('ends_a') or [])} VS "
                f"{','.join(c.get('ends_b') or [])}",
                "", ""])
            rows_written += 1
        if not rows_written:
            ui.notify("No composed transitive routes for this pair.",
                      type="info")
            return
        name = (f"mapping_fullmap_{src.replace(':', '_')}_"
                f"{tgt.replace(':', '_')}_{stamp}.csv")
        ui.download.content(buf.getvalue(), name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Full-map composed routes (evidence; connectors licensed per "
            "the full-map mode).")

    def _deliver_branch_bodyids(src: str, tgt: str, flows, pools,
                                stamp: str) -> None:
        """Row-based bodyId-level export: per-branch bridge-resolved
        pools (linker rows / full population) + vote provenance + the
        per-branch out-map of the target type — the mapper's bodyId-level
        data for this pair.  Evidence only: the out-map is the panel-side
        figure (pool-only claim), never the validation run's `family` bin."""
        import csv as _csv
        import io

        from comparison.mapping_visualization import (
            get_mapping_pool,
            mapping_pool_key,
        )

        rows: List[list] = []
        seen = set()
        for flow in flows:
            key = mapping_pool_key(src, tgt, flow.get("source_type"),
                                   flow.get("foreign_type"))
            if key in seen:
                continue
            seen.add(key)
            pool = get_mapping_pool(pools, flow)
            if not pool:
                continue
            # Out-map: the endpoint type's own neurons this branch's pool
            # does not reach (same definition as the mapping CSV's
            # `target_out_map`; 0 when the basis is the full population).
            _claimed_target = {str(b) for b in
                               (pool.get("target_body_ids") or [])}
            out_map_ids = [b for b in
                           (pool.get("target_type_body_ids") or [])
                           if str(b) not in _claimed_target]
            rows.append([
                src, tgt,
                flow.get("source_type"), flow.get("foreign_type"),
                pool.get("source_basis") or "",
                pool.get("target_basis") or "",
                pool.get("source_pool_size", ""),
                pool.get("target_pool_size", ""),
                "{" + ", ".join(str(b) for b in
                                (pool.get("source_body_ids") or [])) + "}"
                if pool.get("source_body_ids") is not None else "",
                "{" + ", ".join(str(b) for b in
                                (pool.get("target_body_ids") or [])) + "}"
                if pool.get("target_body_ids") is not None else "",
                len(out_map_ids),
                "{" + ", ".join(str(b) for b in out_map_ids) + "}",
            ])
        if not rows:
            ui.notify("No bridge-resolved bodyId pools to export.",
                      type="info")
            return
        buffer = io.StringIO()
        writer = _csv.writer(buffer, quoting=_csv.QUOTE_MINIMAL)
        writer.writerow(["source_dataset", "target_dataset",
                         "source_type", "target_type",
                         "source_basis", "target_basis",
                         "source_pool_size", "target_pool_size",
                         "source_body_ids", "target_body_ids",
                         "target_out_map", "target_out_map_body_ids"])
        writer.writerows(rows)
        name = (f"mapping_branch_bodyids_{src.replace(':', '_')}"
                f"_{tgt.replace(':', '_')}_{stamp}.csv")
        ui.download.content(buffer.getvalue(), name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _suspects_by_pair(flows, src_ds: str, tgt_ds: str) -> dict:
        """Per-rival rows for the collapsed suspects blocks, GROUPED by the
        surviving flow's own (source type → target type) pair (plan §4.1/D1;
        fan-out/suspects display round).

        Reads the mapper's own evidence record
        (``get_same_name_conflict_detail``) for every flow carrying the
        same-name-first flag, so the UI displays the mapper's facts without
        re-deriving anything.  The group key is the FLOW's direction — the
        same direction ``dedupe_mirrored_pairs`` kept and ``build_type_coverage``
        marks — so the badge on a row and its details block always agree.
        Display-only.

        Memoized per dataset pair (``state['suspect_cache']``, cleared on
        every search): the pair card, both coverage tables and the per-type
        breakdown all ask for the same evidence, and the mapper probe behind
        it is not free.
        """
        cache = state.setdefault("suspect_cache", {})
        if (src_ds, tgt_ds) in cache:
            return cache[(src_ds, tgt_ds)]
        out: Dict[tuple, list] = _suspects_by_pair_uncached(
            flows, src_ds, tgt_ds)
        cache[(src_ds, tgt_ds)] = out
        return out

    def _suspects_by_pair_uncached(flows, src_ds: str, tgt_ds: str) -> dict:
        try:
            from comparison.cross_dataset_type_mapper import get_type_mapper
            mapper = get_type_mapper()
        except Exception:
            return {}
        out: Dict[tuple, list] = {}
        seen = set()
        for flow in flows or []:
            if not flow.get("suspects"):
                continue
            s_type = str(flow.get("source_type") or "")
            f_type = str(flow.get("foreign_type") or "")
            key = (s_type, f_type)
            if key in seen or not (s_type and f_type):
                continue
            seen.add(key)
            try:
                detail = mapper.get_same_name_conflict_detail(
                    s_type, src_ds, tgt_ds)
            except Exception:
                detail = None
            rows: list = []
            if not detail:
                for rival in (flow.get("suspect_rivals") or []):
                    rows.append({
                        "rival": rival, "own_pair": "—", "votes": "—",
                        "reverse": "—", "status": "—", "counts": "—",
                    })
            else:
                for ev in detail.get("rival_evidence", []):
                    verified = ev.get("verified_votes") or {}
                    auto = ev.get("auto_votes") or {}
                    curated_n = sum(int(v or 0) for v in verified.values())
                    auto_n = sum(int(v or 0) for v in auto.values())
                    rows.append({
                        "rival": ev.get("rival", ""),
                        "own_pair": ("yes"
                                     if ev.get("rival_has_own_clean_pair")
                                     else "no"),
                        "votes": (f"{curated_n} curated / {auto_n} auto"
                                  if (curated_n or auto_n) else "—"),
                        "reverse": ev.get("reverse_target") or "—",
                        "status": ev.get("rival_pair_status", "—"),
                        "counts": (f"{ev.get('population_source') or 0}/"
                                   f"{ev.get('population_target') or 0}"),
                    })
            if rows:
                out[key] = rows
        return out

    def _suspect_details_blocks(rows_by_pair: dict) -> None:
        """Render one COLLAPSED expander per suspect-flagged type pair (D1:
        never hover-only, never expanded by default).  Shared by the pair
        card and the coverage panel so the per-rival facts render once per
        surface.  Body = the six-column rival table."""
        for (s_type, f_type), rows in sorted((rows_by_pair or {}).items()):
            if not rows:
                continue
            with ui.expansion(
                    f"Suspects — {s_type} → {f_type}: {len(rows)} rival "
                    f"candidate(s) not selected by same-name-first",
                    icon="warning").classes("w-full"):
                ui.label(
                    "These rival names were in the fan-out's candidate set "
                    "but were NOT selected (the same-name candidate was). "
                    "`own 1-to-1 pair` = the crosswalk also pairs this "
                    "rival's own name 1-to-1 in this direction — a separate "
                    "pairing observation, not a verdict. Rivals are never "
                    "merged; add a custom label mapping if one belongs to "
                    "your analysis."
                ).classes("text-caption drocat-muted")
                ui.table(
                    columns=[
                        _col("rival", "Rival candidate", "rival",
                             min_w=120),
                        _col("own_pair", "Own 1-to-1 pair", "own_pair",
                             min_w=110,
                             tooltip="The crosswalk also pairs this rival's "
                                     "own name 1-to-1 in this direction — a "
                                     "separate pairing observation, not a "
                                     "verdict."),
                        _col("votes", "Votes (curated/auto)", "votes",
                             min_w=130),
                        _col("reverse", "Reverse target", "reverse",
                             min_w=110),
                        _col("status", "Rival pair status", "status",
                             min_w=140),
                        _col("counts", "Neurons (src/tgt)", "counts",
                             min_w=120),
                    ],
                    rows=rows,
                ).classes("w-full").add_slot("header-cell",
                                             _HEADER_TOOLTIP_SLOT)

    def _pair_card(src: str, tgt: str, flows, pools: dict) -> None:
        from comparison.cross_dataset_type_mapper import (
            bridge_linker_text, get_type_mapper, source_side_refinement_note,
        )
        from comparison.mapping_visualization import (
            _pair_relationship,
            flow_is_claimed,
            format_pool_side,
            get_mapping_pool,
        )
        from utils.naming_utils import dataset_abbrev

        try:
            mapper = get_type_mapper()
        except Exception:
            mapper = None

        stamp = time.strftime("%Y%m%d_%H%M%S")
        src_code = dataset_abbrev(src) or src
        tgt_code = dataset_abbrev(tgt) or tgt
        # The mapper's per-rival evidence for this pair — it drives both the
        # badge hovers and the collapsed blocks below the table (memoized).
        suspect_rows = _suspects_by_pair(flows, src, tgt)
        # Per-flow cardinality over this card's own flows — the SAME
        # fan/fan-in counts the all-pairs CSV records (build_bridges_csv),
        # so the card's Relationship column and the export never disagree.
        targets_by_source: Dict[str, set] = {}
        sources_by_target: Dict[str, set] = {}
        for flow in flows:
            targets_by_source.setdefault(
                str(flow.get("source_type", "")), set()).add(
                str(flow.get("foreign_type", "")))
            sources_by_target.setdefault(
                str(flow.get("foreign_type", "")), set()).add(
                str(flow.get("source_type", "")))
        rows = []
        for flow in flows:
            s_type = flow.get("source_type", "")
            f_type = flow.get("foreign_type", "")
            info = bridge_linker_text(flow.get("bridges") or [], src, tgt,
                                      f_type)
            pool = get_mapping_pool(pools, flow)
            s_total = int(flow.get("source_count") or 0)
            t_total = int(flow.get("foreign_count") or 0)
            # each linker annotated with its own pooled bodyId count on
            # its home side (user 2026-09-07: "each linker corresponding
            # neuron number")
            per_linker = {(l.get("column"), l.get("value")): l
                          for l in (pool.get("per_linker") or [])}
            parts = []
            for entry in info["entries"]:
                text = entry["text"]
                linker = per_linker.get((entry["column"], entry["value"]))
                if linker and linker.get("body_ids"):
                    text += (f" · {len(linker['body_ids']):,} "
                             f"{dataset_abbrev(linker.get('home', '')) or '?'}"
                             " bodyIds")
                parts.append(text)
            selected_chain = pool.get("selected_chain") if pool else None
            selected_text = ""
            if selected_chain:
                selected_text = bridge_linker_text(
                    [selected_chain], src, tgt, f_type).get("text") or ""
                side_note = source_side_refinement_note(pool, src, tgt)
                if side_note:
                    selected_text = f"{selected_text} — {side_note}"
            all_valid_chains = (pool.get("valid_chains") if pool else None)
            all_valid_text = ""
            if all_valid_chains:
                all_valid_text = bridge_linker_text(
                    all_valid_chains, src, tgt, f_type).get("text") or ""
            if selected_text and all_valid_text and (
                    all_valid_text != selected_text):
                map_used = (f"selected: {selected_text}; "
                            f"all valid evidence: {all_valid_text}")
            else:
                map_used = selected_text or (" + ".join(parts)
                                             if parts else (info["text"] or "—"))
            if flow.get("mapping_status") == "valid_split_evidence":
                map_used = ("valid 1-to-N — branches non-exclusive, no "
                            f"single target; {map_used}")
            elif flow.get("mapping_status") == "conflict":
                map_used = "unresolved conflict — no automatic target; " \
                           f"{map_used}"
            # basis-aware per-side cells (user 2026-09-09): a measured
            # subset, the unconstrained full population, an unmeasurable
            # side and a missing pool must never look alike
            if pool:
                selected_cov = " · ".join((
                    format_pool_side(src_code, pool, "source"),
                    format_pool_side(tgt_code, pool, "target")))
                all_valid_cov = " · ".join((
                    format_pool_side(src_code, pool, "source", "all_valid"),
                    format_pool_side(tgt_code, pool, "target", "all_valid")))
                cov = selected_cov
                if all_valid_cov != selected_cov:
                    cov += f" · all-valid union: {all_valid_cov}"
            else:
                cov = "not pooled"
            if flow.get("mapping_status") == "valid_split_evidence":
                cov = "branch evidence (non-exclusive); " + cov
            # Same-name-first selection (plan-ui-type-mapper-alignment §4.1):
            # a short inline marker here; the rival details go into the
            # COLLAPSED block below the table (user directive: details
            # collapsed, never a hover-only tooltip) AND onto the badge's
            # hover, which is additive — the block stays the no-hover route.
            if flow.get("suspects"):
                _n_riv = len(flow.get("suspect_rivals") or [])
                map_used = (f"same-name-first selection — same-name "
                            f"candidate chosen over {_n_riv} rival "
                            f"candidate(s); {map_used}")
            # A rival the decision declined, or a split fan-out it did not
            # adopt, is disclosure: its pool never enters the Mapped-neurons
            # claim set, so the row must say so instead of reading as a claim.
            if not flow_is_claimed(flow):
                map_used = ("not adopted by the decision — evidence only, "
                            f"not counted as mapped; {map_used}")
            # Per-row cardinality + suspects badge (fan-out/suspects display
            # round): the Relationship column mirrors the coverage tables and
            # the CSV; the Suspects badge is the row's marker, its per-rival
            # details render in the collapsed block below the table.
            relationship = _pair_relationship(
                len(targets_by_source.get(s_type, ())),
                len(sources_by_target.get(f_type, ())))
            suspects_cell = ""
            suspects_tip = ""
            if flow.get("suspects"):
                suspects_cell = (f"⚠ suspects "
                                 f"({len(flow.get('suspect_rivals') or [])})")
                suspects_tip = _suspects_tip(
                    [("", suspect_rows.get((str(s_type), str(f_type)))
                       or [])])
            # Out-map for THIS branch: the endpoint type's own neurons that
            # this row's pool does not reach (per branch, so split branches
            # are non-exclusive and can each report the same neuron).  Only
            # an adopted row is a claim, so a disclosure row says so instead
            # of quoting a number the Mapped column never counted.
            if not pool:
                out_map_cell = "not pooled"
            elif not flow_is_claimed(flow):
                out_map_cell = "not adopted"
            else:
                out_map_cell = len(
                    {str(b) for b in (pool.get("target_type_body_ids") or [])}
                    - {str(b) for b in (pool.get("target_body_ids") or [])})
            rows.append([
                s_type + _multivalue_marker(mapper, s_type, src),
                f_type + _multivalue_marker(mapper, f_type, tgt),
                f"{s_total} {src_code} → {t_total} {tgt_code}",
                relationship, suspects_cell, map_used, cov, out_map_cell,
                suspects_tip,
            ])
        # Quasar table cells nowrap by default: cap the long columns and
        # force wrapping so every column stays visible (user 2026-09-07)
        _table = ui.table(
            columns=[
                _col("name", f"Type ({src_code})", "name", max_w=170),
                _col("foreign", f"Mapped to ({tgt_code})", "foreign",
                     max_w=170),
                _col("counts", "Neurons", "counts", min_w=150),
                _col("relationship", "Relationship", "relationship",
                     min_w=100,
                     tooltip="Cardinality over this pair's accepted "
                             "mappings: 1-to-N fans out to several targets, "
                             "N-to-1 converges from several sources."),
                _col("suspects", "Suspects", "suspects", min_w=110,
                     tooltip="Same-name-first selection: rival candidates "
                             "were in the fan-out but were NOT selected — "
                             "hover the badge for the per-rival evidence, or "
                             "expand the Suspects block below."),
                _col("map_used", "Map used (per linker)", "map_used",
                     max_w=440),
                _col("cov", "Pool coverage (bodyIds)", "cov", min_w=210),
                _col("out_map", "Out-map (this type)", "out_map",
                     min_w=120,
                     tooltip="Neurons of the mapped target type that THIS "
                             "branch's pool does not reach — per branch, so "
                             "split branches are non-exclusive and may quote "
                             "the same neuron twice. Panel-side evidence: "
                             "the claim is the bridge pool, so a type pooled "
                             "on its full population reads 0 and no "
                             "morphology closes holes here (that is what "
                             "makes this figure larger than the validation "
                             "run's `family` bin)."),
            ],
            rows=[dict(zip(("name", "foreign", "counts", "relationship",
                             "suspects", "map_used", "cov", "out_map",
                             "suspects_tip"), r))
                  for r in rows],
        ).classes("w-full")
        # add_slot returns the Slot element, so the two slots are separate
        # calls rather than a chain.
        _table.add_slot("header-cell", _HEADER_TOOLTIP_SLOT)
        _table.add_slot("body-cell-suspects", _SUSPECTS_CELL_SLOT)
        # COLLAPSED suspects details (plan-ui-type-mapper-alignment §4.1/D1;
        # fan-out/suspects display round): one expander per suspect-flagged
        # type pair, one row per demoted rival, carrying the crosswalk
        # observation + votes + the pair-level disposition.  Never a
        # hover-only tooltip; never expanded by default.
        _suspect_details_blocks(suspect_rows)
        with ui.row().classes("flex-wrap"):
            ui.button("Sankey (type-level)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "sankey", "type", stamp))
            ui.button("Sankey (linker)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "sankey", "linker", stamp))
            ui.button("Network (type-level)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "type", stamp))
            ui.button("Network (linker)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "linker", stamp))
            ui.button("Export mapping",
                      on_click=lambda: _deliver_pair_csv(
                          src, tgt, flows, pools, stamp))
            ui.button("Export branch bodyIds",
                      on_click=lambda: _deliver_branch_bodyids(
                          src, tgt, flows, pools, stamp))
            if (state.get("route_scope") == "full"
                    and (state.get("full_map") or {}).get((src, tgt))):
                ui.button("Export full-map routes (CSV)",
                          on_click=lambda: _deliver_fullmap_csv(
                              src, tgt, stamp)).props("outline")

    def _coverage_panel(src: str, tgt: str, flows, pools: dict,
                        contexts: dict) -> None:
        """Bidirectional type-level coverage for ONE dataset pair.

        Rendered at the results' top level (user 2026-09-07: outside
        the dataset-pair card again, with the pair named in the title):
        forward = each queried type's total mapped number (1-to-N
        visible), backward = each receiving type and the sources that
        map onto it, read from the receiving type back to its sources
        (§12.1 user decision: several sources read `1-to-N`, never
        `1-to-1`).  Rows with a dataset-wide incoming context (§12.3)
        list the FULL incoming family with the active query marked, so
        the table explains why one queried type's few neurons fan out to
        a large target population.  Coverage columns carry short
        `CODE · selected` / `CODE · all valid` labels with the full
        dataset key and scope explanation on the header tooltip (§12.5).
        """
        from utils.naming_utils import dataset_abbrev
        from comparison.mapping_visualization import (
            build_type_coverage,
            mapping_pool_key,
        )

        src_code = dataset_abbrev(src) or src
        tgt_code = dataset_abbrev(tgt) or tgt

        # §12.4 scope wording — shared by the four header tooltips
        _SELECTED_TIP = ("Primary supported bridge chain (deterministic "
                         "evidence order) — not a biological adjudication.")
        _ALL_VALID_TIP = ("Union of bodyIds from every supported candidate "
                          "chain; branches are not mutually exclusive.")

        # Coverage columns are named by the FULL dataset key (user
        # 2026-09-09): 'source/target side' read as flipped in the
        # backward view, and abbreviations collide for the two BANC
        # releases (both "BANC") — the full key lives on the tooltip.
        pair_pools = {}
        for flow in flows:
            key = mapping_pool_key(
                src, tgt, flow.get("source_type"), flow.get("foreign_type"))
            if key in pools:
                pair_pools[key] = pools[key]
        coverage = build_type_coverage(
            {(src, tgt): flows}, pair_pools,
            reverse_contexts={(tgt, str(name)): ctx
                              for name, ctx in (contexts or {}).items()})
        forward = coverage.get("forward") or []
        backward = coverage.get("reverse") or []
        if not (forward or backward):
            return
        # Same-name-first suspects: the per-rival evidence grouped by the
        # surviving flow's (source → target) pair — drives both the row
        # badges and the collapsed detail blocks below the tables.
        suspect_map = _suspects_by_pair(flows, src, tgt)
        # The mapper's conflict record is keyed by (SOURCE type, direction),
        # so a source type flagged on several targets repeats the same rival
        # rows under each target key.  A forward/per-type hover lists them
        # once.
        rivals_by_source: Dict[str, list] = {}
        for (st, _ft), riv in suspect_map.items():
            rivals_by_source.setdefault(st, riv)
        # Format each row's suspects badge cell (query-scoped, from the
        # coverage row's suspect_targets / suspect_sources fields) plus its
        # hover: a coverage row aggregates the suspect pairs of ONE queried
        # (or receiving) type, so the rival lines are prefixed with the type
        # pair they belong to.
        for row in forward:
            names = [n for n in str(row.get("suspect_targets") or "").split(",")
                     if n.strip()]
            n = int(row.get("suspect_count") or len(names))
            row["suspects_cell"] = (f"⚠ suspects ({n}): "
                                    f"{', '.join(names)}" if n else "")
            row["suspects_tip"] = _suspects_tip(
                [("", rivals_by_source.get(str(row.get("type") or ""))
                   or [])]) if n else ""
        for row in backward:
            names = [n for n in str(row.get("suspect_sources") or "").split(",")
                     if n.strip()]
            n = int(row.get("suspect_count") or len(names))
            row["suspects_cell"] = (f"⚠ suspects ({n}): "
                                    f"{', '.join(names)}" if n else "")
            row["suspects_tip"] = _suspects_tip(
                [(f"← {st} · {src_code}: ", rows_)
                 for (st, ft), rows_ in sorted(suspect_map.items())
                 if ft == str(row.get("type") or "")]) if n else ""
        # Data-driven fan-out summary for the expansion title (replaces the
        # old hard-coded "1-to-N fan-out"): read from the forward rows' own
        # relationship values.
        rels = {str(r.get("relationship") or "") for r in forward}
        if rels & {"1-to-N", "N-to-N"}:
            fan_summary = "1-to-N fan-out"
        elif "N-to-1" in rels:
            fan_summary = "N-to-1 fan-in"
        else:
            fan_summary = "all 1-to-1"
        if suspect_map:
            fan_summary += f" · ⚠ {len(suspect_map)} suspect pair(s)"
        _legend = (
            "Forward: per queried type - neurons, mapped targets, and "
            "per-side bodyId coverage.\n"
            "Backward: per receiving type - the source types mapping onto "
            "it (several sources = 1-to-N); rows marked 'dataset-wide "
            "incoming' list every incoming source type (context, not "
            "canonical acceptance).\n"
            "Coverage: selected = primary bridge chain; all valid = union "
            "of every supported chain.  States: not pooled / not measured "
            "/ measured zero.  No bodyId-to-bodyId pairing is inferred.\n"
            "same-name-first selection: a fan-out whose candidate set "
            "contained the source type's own name — that candidate was "
            "selected and the other (rival) candidates were NOT; a "
            "'⚠ suspects' badge marks the row, hovering it lists the "
            "per-rival evidence, and the collapsed 'Suspects' block below "
            "lists the same facts.\n"
            "🧩 multi-value: the release annotates several candidate names "
            "in ONE `type` cell (e.g. 'LAL173,LAL174'); the cell is kept as "
            "one atomic type name.")
        with ui.expansion(
                f"Type coverage — {src} → {tgt} "
                f"(bidirectional, {fan_summary})",
                icon="swap_vert").classes("w-full") \
                .tooltip(_legend):
            _fwd_table = ui.table(
                columns=[
                    _col("type", "Queried type", "type", max_w=170),
                    _col("dataset", "Dataset", "dataset", max_w=180),
                    _col("count", "Neurons", "count", min_w=90),
                    _col("maps_to", "Maps to", "maps_to", max_w=480),
                    _col("relationship", "Relationship",
                         "relationship", min_w=110),
                    _col("suspects_cell", "Suspects", "suspects_cell",
                         min_w=140,
                         tooltip="Same-name-first selection: rival "
                                 "candidates were in the fan-out but were "
                                 "NOT selected — hover the badge for the "
                                 "per-rival evidence, or expand the Suspects "
                                 "block below."),
                    _col("coverage_note", "Coverage interpretation",
                         "coverage_note", max_w=480),
                    _col("query_cov_selected", f"{src_code} · selected",
                         "query_cov_selected", min_w=135,
                         tooltip=f"{src} side (bodyIds) — selected "
                                 f"bridge. {_SELECTED_TIP}"),
                    _col("query_cov", f"{src_code} · all valid",
                         "query_cov", min_w=145,
                         tooltip=f"{src} side (bodyIds) — all-valid "
                                 f"union. {_ALL_VALID_TIP}"),
                    _col("target_cov_selected", f"{tgt_code} · selected",
                         "target_cov_selected", min_w=135,
                         tooltip=f"{tgt} side (bodyIds) — selected "
                                 f"bridge. {_SELECTED_TIP}"),
                    _col("target_cov", f"{tgt_code} · all valid",
                         "target_cov", min_w=145,
                         tooltip=f"{tgt} side (bodyIds) — all-valid "
                                 f"union. {_ALL_VALID_TIP}"),
                ],
                rows=forward,
            ).classes("w-full")
            _fwd_table.add_slot("header-cell", _HEADER_TOOLTIP_SLOT)
            _fwd_table.add_slot("body-cell-suspects_cell",
                                _SUSPECTS_CELL_SLOT)
            ui.label(
                "Backward — per receiving type: the source types mapping "
                "onto it (several sources = 1-to-N).  Rows marked "
                "`dataset-wide incoming` list every incoming source type "
                "(the active query marked) — context, not canonical "
                "acceptance.  Same coverage states as forward; split "
                "branches are non-exclusive; no bodyId pairing."
            ).classes("text-caption drocat-muted")
            _bwd_table = ui.table(
                columns=[
                    _col("type", "Receiving type", "type", max_w=170),
                    _col("dataset", "Dataset", "dataset", max_w=180),
                    _col("count", "Neurons", "count", min_w=90),
                    _col("mapped_from", "Mapped from", "mapped_from",
                         max_w=480),
                    _col("relationship", "Relationship",
                         "relationship", min_w=110),
                    _col("suspects_cell", "Suspects", "suspects_cell",
                         min_w=140,
                         tooltip="Same-name-first selection: rival "
                                 "candidates were in the fan-out but were "
                                 "NOT selected — hover the badge for the "
                                 "per-rival evidence, or expand the Suspects "
                                 "block below."),
                    _col("coverage_note", "Coverage interpretation",
                         "coverage_note", max_w=480),
                    _col("source_cov_selected", f"{src_code} · selected",
                         "source_cov_selected", min_w=135,
                         tooltip=f"{src} side (bodyIds) — selected "
                                 f"bridge. {_SELECTED_TIP}"),
                    _col("source_cov", f"{src_code} · all valid",
                         "source_cov", min_w=145,
                         tooltip=f"{src} side (bodyIds) — all-valid "
                                 f"union. {_ALL_VALID_TIP}"),
                    _col("target_cov_selected", f"{tgt_code} · selected",
                         "target_cov_selected", min_w=135,
                         tooltip=f"{tgt} side (bodyIds) — selected "
                                 f"bridge. {_SELECTED_TIP}"),
                    _col("target_cov", f"{tgt_code} · all valid",
                         "target_cov", min_w=145,
                         tooltip=f"{tgt} side (bodyIds) — all-valid "
                                 f"union. {_ALL_VALID_TIP}"),
                ],
                rows=backward,
            ).classes("w-full")
            _bwd_table.add_slot("header-cell", _HEADER_TOOLTIP_SLOT)
            _bwd_table.add_slot("body-cell-suspects_cell",
                                _SUSPECTS_CELL_SLOT)
            # COLLAPSED per-rival suspect details for this pair's flagged
            # type pairs — rendered ONCE (shared renderer with the pair
            # card), never a hover-only tooltip, never expanded by default.
            _suspect_details_blocks(suspect_map)

    def _render_results() -> None:
        results.clear()
        pair_flows = state["pair_flows"]
        pools = state["pools"]
        with results:
            composed = state.get("composed")
            if composed:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    ui.button("Mapping graph (HTML)", icon="account_tree",
                              on_click=lambda: _deliver(
                                  composed,
                                  f"mapping_graph_{stamp}.html"))
                    ui.button("Mapping sankey (HTML)",
                              on_click=lambda: _deliver_composed_sankey(stamp))
                    ui.button("Export mapping — all pairs (CSV)",
                              on_click=lambda: _deliver_combined_csv(stamp))
            else:
                ui.label("No mapping graph (nothing mapped across the "
                         "selected datasets).").classes(
                    "text-caption drocat-muted")
            for note in state["meta"].get("notes", []):
                ui.label(note).classes("text-caption drocat-muted")
            summary = state.get("summary") or []
            if summary:
                ui.table(
                    columns=[
                        {"name": "dataset", "label": "Dataset",
                         "field": "dataset", "align": "left",
                         "tooltip": "One row per selected dataset."},
                        {"name": "types", "label": "Matched types",
                         "field": "types", "align": "left",
                         "tooltip": "Search terms that matched type names "
                                    "in this dataset."},
                        {"name": "neurons", "label": "Neurons",
                         "field": "neurons", "align": "left",
                         "tooltip": "Neurons of the matched types in this "
                                    "dataset."},
                        {"name": "mapped", "label": "Mapped neurons",
                         "field": "mapped", "align": "left",
                         "tooltip": "The CLAIM set: branch-claimed bodyIds "
                                    "received INTO this dataset (union of "
                                    "the ADOPTED branches' resolved pools), "
                                    "with '(k types)' from 2 distinct "
                                    "adopted types up. Pure claim basis "
                                    "(2026-09-27): only pairs the scoped "
                                    "decision ADOPTED count — a same-name "
                                    "rival it declined, or a fan-out branch "
                                    "it did not adopt, is listed in the "
                                    "pair table as a disclosure row and "
                                    "shows under Evidence reach instead. "
                                    "The received types' full populations "
                                    "are the reference denominator. A "
                                    "claimed type that has no neurons here "
                                    "does not count — it is listed under "
                                    "orphans. A dataset that only issues "
                                    "the query receives 0 — its issued "
                                    "side is Matched types / Neurons."},
                        {"name": "reach", "label": "Evidence reach (all flows)",
                         "field": "reach", "align": "left",
                         "tooltip": "The REACH tier: every flow's pools "
                                    "unioned — the ADOPTED claim set PLUS "
                                    "the disclosure ends the decision "
                                    "declined (a same-name rival, a "
                                    "vote-declined annotation pair). This "
                                    "is the same scope the CSVs' all-valid "
                                    "columns and the coverage tables "
                                    "publish. claim ≤ reach always; the "
                                    "difference is the disclosure "
                                    "material, listed per type with its "
                                    "decline reason in the pair table's "
                                    "'not adopted' rows."},
                        {"name": "out_map",
                         "label": "Out-map (in-map types)",
                         "field": "out_map", "align": "left",
                         "tooltip": "Neurons of the received (in-map) "
                                    "types in THIS dataset that the claim "
                                    "set above does not reach — the "
                                    "union of those types' own populations "
                                    "minus the branch-claimed bodyIds, so a "
                                    "type claimed on its full-population "
                                    "basis contributes 0. This is the "
                                    "PANEL's figure: the validation "
                                    "pipeline bins the same material per "
                                    "BRANCH and lets a morph-qualified "
                                    "candidate close a hole, so on "
                                    "circadian_clock → male-cns it reports "
                                    "15 out-map here (219 − 204) while "
                                    "family_candidates.csv holds 11 rows. "
                                    "Evidence about coverage, never a "
                                    "mapping claim."},
                        {"name": "unmapped", "label": "Unmapped (orphans)",
                         "field": "unmapped", "align": "left",
                         "tooltip": "Matched types here with no realized "
                                    "counterpart in another selected "
                                    "dataset — OR held back by same-name-"
                                    "first: a same-name candidate existed, "
                                    "but not every rival candidate has its "
                                    "own 1-to-1 pairing, so none was "
                                    "selected (see the Orphan types list for "
                                    "the per-row reason)."},
                    ] + ([{
                        "name": "fullmap",
                        "label": "Full-map ends (transitive via connectors)",
                        "field": "fullmap", "align": "left",
                        "tooltip": "FULL-MAP MODE ONLY (2026-10-01): the "
                                   "composed reach through ALL licensed "
                                   "connector datasets — direct ends plus "
                                   "transitive-only ends '(+K via <mids>)'. "
                                   "An additive disclosure cell; the "
                                   "Evidence reach cell keeps its curated "
                                   "definition. The validation pipeline "
                                   "accepts this scope gated "
                                   "(route_scope='full')."}]
                        if (state.get("route_scope") == "full") else []),
                    rows=summary,
                ).classes("w-full").add_slot("header-cell",
                                             _HEADER_TOOLTIP_SLOT)
            else:
                ui.label("No mappings found for the search across the "
                         "selected datasets.").classes(
                    "text-caption drocat-muted")
            # R5: the per-type breakdown — collapsed, and only for
            # multi-type previews (a single-type exact preview looks
            # exactly like before)
            per_type = state.get("summary_per_type") or []
            multi_type = (len(state.get("queries") or []) > 1
                          or any((r.get("types") or 0) > 1
                                 for r in (summary or [])))
            if per_type and multi_type:
                for r in per_type:
                    _n = int(r.get("suspects") or 0)
                    r["suspects_cell"] = (f"⚠ suspects ({_n})" if _n else "")
                    r["suspects_tip"] = ""
                    if _n:
                        # The row is one (matched type × target dataset): its
                        # hover lists that source type's rival rows once (the
                        # mapper keys the conflict record by source type and
                        # direction, not by target) — the same evidence the
                        # pair's Suspects block renders.
                        _ds = str(r.get("dataset") or "")
                        _tgt = str(r.get("target") or "")
                        _riv: Dict[str, list] = {}
                        for (st, _ft), rows_ in _suspects_by_pair(
                                pair_flows.get((_ds, _tgt), []),
                                _ds, _tgt).items():
                            _riv.setdefault(st, rows_)
                        r["suspects_tip"] = _suspects_tip(
                            [("", _riv.get(str(r.get("type") or ""))
                               or [])])
                with ui.expansion(
                        f"Per-type breakdown — {len(per_type)} rows "
                        f"(matched type × target dataset)",
                        icon="format_list_bulleted").classes("w-full"):
                    _pt_table = ui.table(
                        columns=[
                            _col("dataset", "Dataset", "dataset",
                                 max_w=180),
                            _col("type", "Matched type", "type",
                                 max_w=170),
                            _col("query", "Query chip", "query",
                                 max_w=140),
                            _col("neurons", "Neurons", "neurons",
                                 min_w=90),
                            _col("target", "Mapped into", "target",
                                 max_w=180,
                                 tooltip="The target dataset this row's "
                                         "mapping lands in — its mapped "
                                         "counts are specific to THAT "
                                         "dataset, never summed across "
                                         "datasets."),
                            _col("relationship", "Relationship",
                                 "relationship", min_w=100,
                                 tooltip="Cardinality of this matched type's "
                                         "mapping into the row's target "
                                         "dataset."),
                            _col("suspects_cell", "Suspects", "suspects_cell",
                                 min_w=120,
                                 tooltip="Same-name-first selection: rival "
                                         "candidates were in the fan-out but "
                                         "were NOT selected — hover the badge "
                                         "for the per-rival evidence; the "
                                         "pair's Suspects block lists the "
                                         "same facts."),
                            _col("mapped", "Mapped neurons", "mapped",
                                 min_w=120,
                                 tooltip="Branch-claimed bodyIds received "
                                         "into this row's target dataset "
                                         "(the claim set), with '(k "
                                         "types)' from 2 distinct "
                                         "received types up."),
                            _col("reach", "Evidence reach", "reach",
                                 min_w=120,
                                 tooltip="All flows' pools toward this "
                                         "row's target dataset, "
                                         "disclosure ends included — "
                                         "the reach tier (same scope "
                                         "as the strip's Evidence "
                                         "reach column)."),
                            *([_col(
                                "fullmap_cell", "Full-map reach",
                                "fullmap_cell", min_w=150,
                                tooltip="FULL-MAP MODE ONLY (2026-10-01): "
                                        "composed ends for this source "
                                        "type through ALL licensed "
                                        "connectors — curated reach plus "
                                        "the transitive-only delta "
                                        "'(+K via <mids>)'; ⚠ marks "
                                        "source types where two "
                                        "connectors license DIFFERENT "
                                        "end sets (route conflict — the "
                                        "full list rides the full-map "
                                        "CSV).")]
                              if (state.get("route_scope") == "full")
                              else []),
                            _col("out_map", "Out-map (in-map types)",
                                 "out_map", min_w=120,
                                 tooltip="Neurons of THIS row's mapped "
                                         "target type(s) that the claim does "
                                         "not reach — computed PER RECEIVED "
                                         "TARGET TYPE (a neuron an adopted "
                                         "branch reaches anywhere in that "
                                         "type counts as claimed) and summed "
                                         "over the row's accepted target "
                                         "types. Two reasons this is not the "
                                         "dataset row's number: convergent "
                                         "sources can report the same "
                                         "neuron twice across rows (on "
                                         "circadian_clock → male-cns these "
                                         "rows sum to 17 where the deduped "
                                         "union above is 15), and with no "
                                         "morphology a candidate cannot "
                                         "close a hole, so this can exceed "
                                         "the run's family count (11 there, "
                                         "binned per branch — so match this "
                                         "to `family` only per type, not "
                                         "across a convergent pair)."),
                            _col("unmapped", "Unmapped (orphans)",
                                 "unmapped", min_w=100),
                        ],
                        rows=per_type,
                    ).classes("w-full")
                    _pt_table.add_slot("header-cell", _HEADER_TOOLTIP_SLOT)
                    _pt_table.add_slot("body-cell-suspects_cell",
                                       _SUSPECTS_CELL_SLOT)
            orphan_all = state.get("orphans") or {}
            if orphan_all:
                orphan_count = len({
                    (e["dataset"], e["type"])
                    for v in orphan_all.values() for e in v})
                with ui.expansion(
                        f"Orphan types — no mapped counterpart "
                        f"({orphan_count})",
                        icon="link_off").classes("w-full"):
                    for (src, tgt), entries in sorted(orphan_all.items()):
                        names = ", ".join(
                            _describe_orphan(e, tgt) for e in entries)
                        ui.label(
                            f"{src} → {tgt}: {names}").classes(
                            "text-caption drocat-muted")
            for (src, tgt), flows in sorted(
                    pair_flows.items(),
                    key=lambda kv: (-len(kv[1]), kv[0])):
                _coverage_panel(
                    src, tgt, flows, pools,
                    (state.get("reverse_contexts") or {}).get((src, tgt))
                    or {})
                _label = _pair_card_label(src, tgt, flows)
                _fm = (state.get("full_map") or {}).get((src, tgt)) or {}
                if state.get("route_scope") == "full" and _fm:
                    _to = len(_fm.get("transitive_only_ends") or [])
                    _mids = ', '.join(_fm.get('mids') or [])
                    _label += (f" · full map: +{_to} transitive-only"
                               f" via {_mids}" if _to else
                               " · full map: no transitive-only ends")
                with ui.expansion(
                        _label,
                        icon="compare_arrows").classes("w-full"):
                    _pair_card(src, tgt, flows, pools)
            if not pair_flows:
                ui.label("No mappings found for the search across the "
                         "selected datasets.").classes(
                    "text-caption drocat-muted")

    def _deliver_combined_csv(stamp: str) -> None:
        from comparison.mapping_visualization import build_bridges_csv

        pair_flows = state["pair_flows"]
        pools = state["pools"]
        # One fixed column set for every pair, so the all-pairs file is a
        # plain header + rows concatenation (the old per-pair pivoted
        # bridge-<column> fields needed a union-of-columns hack to keep
        parts: List[str] = []
        for (src, tgt), flows in sorted(pair_flows.items()):
            text = build_bridges_csv(flows, pools=pools, extended=True)
            if not text:
                continue
            lines = text.splitlines()
            if parts:
                lines = lines[1:]  # shared header already written
            parts.append("\n".join(lines))
        if not parts:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"mapping_all_pairs_{stamp}.csv"
        ui.download.content("\n".join(parts), name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _set_loading(on: bool) -> None:
        """Show/hide the loading notice and freeze the Search button."""
        loading_row.set_visibility(on)
        if on:
            search_btn.disable()
        else:
            search_btn.enable()

    async def _run_click() -> None:
        """Paint the loading notice FIRST, then search off the event loop.

        Warm searches use the existing serialized worker-thread path.  A
        cold mapper load uses the dedicated spawned process so pandas and the
        Python graph build cannot starve the websocket event loop.
        """
        queries, datasets, mode = _collect_queries()
        if not queries:
            ui.notify("Enter a type search first.", type="warning")
            return
        if len(datasets) < 2:
            ui.notify("Select at least 2 datasets with cached neuron "
                      "indexes.", type="warning")
            return
        _set_loading(True)
        try:
            # yield once so the browser paints the notice before the
            # heavy search starts
            await asyncio.sleep(0.05)
            from ..neuron_index import (
                is_type_mapper_loaded,
                run_cross_dataset_scan_in_process,
                run_serialized_cross_dataset_scan,
            )
            if not is_type_mapper_loaded():
                outcome = await run_cross_dataset_scan_in_process(
                    _compute_type_mapping, queries, datasets, mode,
                    route_scope=route_scope_state.get("value", "curated"))
            else:
                loop = asyncio.get_running_loop()
                outcome = await loop.run_in_executor(
                    None,
                    lambda: run_serialized_cross_dataset_scan(
                        _compute, queries, datasets, mode),
                )
            _apply(outcome)
            _record_panel_history(queries, datasets, outcome)
        except Exception:
            logger.exception(
                "type mapping search failed (queries=%r, datasets=%r)",
                queries, datasets)
            ui.notify("The type mapping search failed — please double "
                      "check the query and try again.", type="negative")
        finally:
            _set_loading(False)

    def _record_panel_history(queries, datasets, outcome) -> None:
        """Record the searched chips in the panel's OWN history store.

        Only confirmed hits are recorded, mirroring the shared neuron
        history's rule: at least one queried type must have matched a
        dataset (the per-dataset summary's matched-type count). Failed or
        zero-hit searches never pollute the Recent/Frequent list.
        """
        matched = sum(int(row.get("types") or 0)
                      for row in (outcome.get("summary") or []))
        if not matched:
            return
        try:
            from ..type_mapping_history import record as _record_history

            _record_history([str(q) for q in queries],
                            datasets=[str(d) for d in datasets])
        except Exception:
            # history is a convenience, never an error
            logger.warning("type mapping history record failed",
                           exc_info=True)

    def _collect_queries() -> tuple:
        mode, chips = search.get_value()
        queries = [str(q).strip() for q in chips if str(q).strip()]
        datasets = [d for d in (get_datasets() or [])
                    if neuron_index_path(d).is_file()]
        return queries, datasets, mode

    def _compute(queries, datasets, mode) -> Dict[str, Any]:
        """Compatibility wrapper for the shared process-safe computation."""
        return _compute_type_mapping(
            queries, datasets, mode,
            route_scope=route_scope_state.get("value", "curated"))

    def _apply(outcome: Dict[str, Any]) -> None:
        """Apply the search outcome and render results (event loop)."""
        state.update(**outcome)
        # The suspects evidence memo is per search: the flows it was built
        # from are gone.
        state.get("suspect_cache", {}).clear()
        notes_label.set_text(
            "   ".join(outcome["meta"].get("notes", [])))
        _render_results()

    button = ui.button("Type Mapping", icon="hub", on_click=dialog.open)

    def refresh_state() -> None:
        if _ready():
            button.enable()
            button.tooltip("Preview the auto type mapping across the "
                           "selected datasets (informational only)")
        else:
            button.disable()
            button.tooltip("Select at least 2 datasets with cached neuron "
                           "indexes to preview the type mapping")

    button.search_container = search  # type: ignore[attr-defined]
    button.refresh_state = refresh_state  # type: ignore[attr-defined]
    refresh_state()
    return button
