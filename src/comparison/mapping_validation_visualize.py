"""Stage-4 visualization for the type-mapping validation pipeline.

Revision 3.12 is the current category model (plan
plan-mapping-validation-rev311-chain-aware-pool-widening.md §2b; technical
report §4).  The scene reads the EXPORTED ``category`` /
``candidate_annotation`` / ``in_scope`` columns — it never re-parses label
prefixes — so the legend and the CSVs cannot disagree.

- One scene per PARENT mapping group (a source type with all its bridge
  branches), rendered in the SOURCE dataset's RENDER template (locked
  first-run direction: source=FAFB -> FLYWIRE brain; sources are bridged
  native -> render when those differ — male-cns/hemibrain raw-voxel
  frames — and target neurons are bridged in via
  ``transform_neurons_to_space``).
- ONE legend group per BRANCH, named ``{src} → {tgt} · {linker}``.  The
  hierarchy is branch group -> category root -> bodyId leaf, with roots:
    ``query · {src_type}``                  (all branch source neurons)
    ``matched`` / ``verified`` / ``borderline`` / ``unmatched`` · {tgt}
    ``sibling · {N} members``               (in-map targets of other
                                             branches of the query)
    ``family`` / ``candidates`` / ``relative`` / ``examinees`` (one root
                                             each; every bodyId leaf carries
                                             the qualified token
                                             ``{T}(out-map)`` / ``{T}>{src}``
                                             / ``{T}(no_source)`` / ``untyped``)
    ``pooling · {src_type}``            (``pooling`` mode only: the
                                             unsupervised pool whose best
                                             source has this parent type;
                                             the leaf token rides with the
                                             post-hoc ``mapper_cell`` — and
                                             the morph verdict where the
                                             gate scored it — as the tag)
  A standalone ``(dup)`` tag on a leaf marks a bodyId recurring across
  branches.  Leaves inside a root are sorted by ``type + suffix``, and a
  bare-category root always renders even with a single leaf.
  Query roots always render ahead of the category roots (layer order); the
  expansion bins follow in the order sibling > candidates > family >
  relative > examinees.
- Only in-scope rows render: a connectivity-qualified suspect failing the
  morph rule carries ``in_scope=False`` / ``morph_failed=True`` and stays
  in the CSVs (the connectivity-only homolog-finding result).
- ``untyped`` is a suffix on any bin, not a peer category.
- Colors: each category has one fixed color across branches and types
  (query blue, matched cyan, verified green, borderline gold, unmatched
  grey, sibling pink, candidates orange, family light green, relatives
  olive, examinees red).
- ``legend_mode='tree'``: the drocat legend panel builds branch ->
  category -> bodyId from the ``drocatLegend`` meta tags, so the row-to-
  trace mapping can never drift from the figure.  Targets whose skeleton
  cannot be fetched are logged (``! {root}: {bid} unavailable``) and
  remain in the CSVs; unavailable neurons never silently vanish.
- Revision 3.5 Issue 6: loaded neurons are renamed to ``str(bodyId)`` so
  navis trace names match the backend trace-identity resolver exactly
  (no positional fallback slips); a neuron renders in exactly ONE
  expansion bucket per branch; an optional debug self-check
  (``cfg.scene_selfcheck``) verifies each legend leaf's geometry against
  its neuron's loaded bbox.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from comparison.body_id_resolver import chain_key
from comparison.cross_dataset_type_mapper import basis_is_row_evidence
from comparison.mapping_validation import DEDUP_RANK, TIER_CATEGORIES

# Color = f(category), unified across branches and types, high contrast
# (user directive 2026-09-12: neurons under the same category share one
# color even across different types; query keeps its own fixed color).
# Alpha-free hex: the backend applies the global ``neuron_alpha``.
CATEGORY_COLORS = {
    'query': '#1f77b4',        # blue    - the FAFB source neurons
    'out-map query': '#1f77b4',  # blue   - FAFB sources without a partner
    'out-map candidates': '#aec7e8',  # light blue - their found targets
    'matched': '#17becf',      # cyan    - verified AND ru > matched_ru_min
    'verified': '#2ca02c',     # green   - top-1 (ru*/jaccard) or top-N same-type
    'borderline': '#e6ab02',   # gold    - <= invader_borderline_max invaders
    'unmatched': '#7f7f7f',    # grey    - more invaders ahead
    'candidates': '#ff7f0e',   # orange  - out-of-pool, morph-qualified
    'fill': '#9467bd',         # purple  - candidates promoted on gap
    'examinees': '#d62728',   # red     - ahead invaders without morph support
                              # (category renamed from 'suspicious')
    'sibling': '#e377c2',      # pink    - pool member of a sibling branch
    'backward': '#8c564b',     # brown   - type maps backward to the query
    'source-candidates': '#8c564d',  # brown (bokeh.palettes Category10[5])
                                     # - re-aimed out-of-map candidates; user
                                     # color adjustment of D-B12's #6baed6
    'family': '#98df8a',       # light green - same-type extra, unqualified
    'relative': '#bcbd22',     # olive   - type-mates of candidate types
    'relatives': '#bcbd22',    # olive   - legacy alias (pre-3.12 root name)
    'pooling': '#7b4173',      # plum    - `pooling` mode's unsupervised
                               # pool: parallel to the ladder, so it wears
                               # no colour the ladder's bins own
                               # (distinct from candidates orange and
                               # sibling pink)
}
UNASSIGNED_COLOR = '#7f7f7f'  # grey

# Single-bucket rule per invader neuron (Revision 3.6): the LOWEST
# priority value wins when a neuron qualifies for several buckets —
# structural facts (sibling pool membership, backward/alternate-chain
# evidence) outrank morphological qualification (Revision 3.6
# inversion; the old fill=candidates>sibling order is retired, and the
# `fill` root is merged into `candidates` per Revision 3.8).
INVADER_BUCKET_PRIORITY = {
    'sibling': 0,
    'backward': 1,
    'alternate-chain': 1,
    'fill': 2,
    'candidates': 2,
    'family': 3,
    'examinees': 4,
}

# Revision 3.12 render order for the new S2 categories (lower = earlier).
# Tier roots are added separately (before the expansion bins); this covers
# only the expansion bins.
CATEGORY_RENDER_PRIORITY = {
    'sibling': 0,
    'candidates': 1,
    'family': 2,
    'relative': 3,
    'examinees': 4,
}


def _linker_sig(pair) -> str:
    """Deduplicated linker values of the selected chain ('direct' if none)."""
    vals: List[str] = []
    for l in pair.linkers:
        v = str(l.get('raw_value', l.get('value', '')))
        if v and v not in vals:
            vals.append(v)
    return '/'.join(vals) if vals else 'direct'


def invader_bucket_key(ahead_bid: int, ahead_type: str,
                       sibling_index: Dict[int, Tuple[str, str]],
                       backward_lookup) -> str:
    """Rev 3.6 fallback classification (rows without precomputed
    fields).  Collapsed root keys (user 2026-09-12): sibling is ONE
    root per branch; backward roots carry the exact mapped type as
    suffix (never linker/additional_type names)."""
    hit = sibling_index.get(int(ahead_bid))
    if hit is not None:
        return 'sibling'
    mapped = backward_lookup(ahead_type) if backward_lookup else None
    if mapped:
        return f'backward · {mapped}'
    return 'examinees'


def bucket_root_label(key: str, types_by_bid: Dict[int, str]) -> str:
    """Collapsed roots.  Revision 3.12 (S8): each expansion category is ONE
    root; the bodyId-level detail rides on the leaf labels, not the root.
    Only the legacy `` · ``-suffixed keys (compatibility callers) pass
    through verbatim; a bare category key returns itself."""
    prefix = key.split(' · ')[0]
    if prefix in ('candidates', 'examinees', 'family', 'relative'):
        if ' · ' in key:
            # legacy/verbatim key (pre-merge callers) — keep as-is
            return key
        return prefix
    if prefix == 'sibling' and key == 'sibling':
        n = len(types_by_bid)
        return f'sibling · {n} member{"" if n == 1 else "s"}'
    if ' · ' in key:
        return key
    if key == 'examinees · untyped':
        return key
    types = sorted({str(t) for t in types_by_bid.values()
                    if t and t != 'untyped'})
    if not types and 'untyped' in {str(t) for t in types_by_bid.values()}:
        return 'examinees · untyped'
    if len(types) <= 1:
        return f'{prefix} · {types[0] if types else "?"}'
    return f'{prefix} · {len(types)} types'


def ensure_bucket(buckets: Dict[str, Dict], bucket_order: List[str],
                  key: str) -> Dict:
    """Get-or-create a bucket record.

    Revision 3.12 record shape: ``ids`` + ``types`` (raw target type per
    bodyId) + the leaf detail ``leaf_token`` / ``tags`` / ``sort_key``.
    """
    if key not in buckets:
        buckets[key] = {'ids': set(), 'types': {}, 'leaf_token': {},
                        'tags': {}, 'sort_key': {}}
        bucket_order.append(key)
    return buckets[key]


def assign_invader(claimed: Dict[int, Tuple[int, str]],
                   buckets: Dict[str, Dict], bucket_order: List[str],
                   bid: int, atype: str, key: str) -> None:
    """Single-bucket rule per invader neuron (Revision 3.5 Issue 6b).

    LEGACY (pre-3.12) helper: maintained for the older bucket tests.  The
    Revision 3.12 render path buckets by the exported ``category`` in
    :func:`build_category_buckets`, which enforces the same single-bucket
    rule directly.  Lowest priority value wins; earlier higher-priority
    claims are moved.
    """
    bid = int(bid)
    prio = INVADER_BUCKET_PRIORITY[key.split(' · ')[0]]
    cur = claimed.get(bid)
    if cur is not None:
        if cur[0] <= prio:
            return
        buckets[cur[1]]['ids'].discard(bid)
        buckets[cur[1]]['types'].pop(bid, None)
    claimed[bid] = (prio, key)
    rec = ensure_bucket(buckets, bucket_order, key)
    rec['ids'].add(bid)
    rec['types'][bid] = atype


def build_sibling_index(branch_list, own_pair) -> Dict[int,
                                                        Tuple[str, str]]:
    """Revision 3.5 Issue 2: pool categories of the SIBLING branches of
    the same parent (same query + source_type), excluding the branch
    being rendered: target_bid -> (category there, sibling target type).
    """
    sibling_index: Dict[int, Tuple[str, str]] = {}
    for other_pair, other_res in branch_list:
        if other_pair.key == own_pair.key:
            continue
        for tbid, cat in (other_res.get('target_categories')
                          or {}).items():
            sibling_index.setdefault(
                int(tbid), (str(cat), other_pair.target_type))
    return sibling_index


def make_backward_lookup(validator, cfg, cache: Dict[str, Optional[str]]):
    """Revision 3.5 Issue 2: cached backward type-mapper lookups — an
    invader's target-dataset type mapped toward the source dataset via
    ``get_mapping_decision`` ('mapped' / 'valid_split_evidence' only,
    fail-closed otherwise)."""
    def backward_lookup(ahead_type):
        if (not ahead_type or ahead_type == '?'
                or (isinstance(ahead_type, float)
                    and ahead_type != ahead_type)
                or validator.mapper is None):
            return None
        if ahead_type not in cache:
            mapped = None
            try:
                dec = validator.mapper.get_mapping_decision(
                    ahead_type, cfg.target_dataset, cfg.source_dataset)
                if dec.get('status') in ('mapped', 'valid_split_evidence'):
                    cands = [dec.get('target_type')]
                    cands += list(dec.get('target_types') or [])
                    vals = list(dict.fromkeys(
                        str(t) for t in cands if t))
                    mapped = '/'.join(vals) or None
            except Exception:  # noqa: BLE001
                mapped = None
            cache[ahead_type] = mapped
        return cache[ahead_type]
    return backward_lookup


def build_category_buckets(res, validator, pair, suspicious_cap: int
                           ) -> Tuple[Dict[str, Dict], List[str]]:
    """Revision 3.12: bucket the branch's expansion rows by their exported
    ``category`` (the S2 partition), not by re-parsing label prefixes.

    The scene reads exactly what the CSV exported, so the picture and the
    table cannot disagree.  A row is rendered only when it is in scope
    (``in_scope`` True and a non-empty ``category``); out-of-scope rows
    (connectivity-qualified but morph-failed) stay CSV-only.

    Each expansion category is ONE root (``candidates`` / ``examinees`` /
    ``family`` / ``relative``); the bodyId-level detail — the qualified
    ordered token (``{T}(out-map)`` / ``{T}>{src}`` /
    ``{T}(no_source)`` / ``untyped``) and the standalone ``(dup)`` tag —
    rides on the LEAF label.  A record carries:
    ``ids``, ``types`` (bid -> raw target type), ``leaf_token`` (bid ->
    qualified token), ``tags`` (bid -> '(dup)' or '') and ``sort_key``
    (bid -> 'token tag', the leaf ordering key).

    Returns (buckets {key: {...}}, insertion-ordered keys).
    """
    def _cat(r):
        return str(r.get('category') or '')

    def _dup(r):
        return '(dup)' if r.get('dup') else ''

    buckets: Dict[str, Dict] = {}
    bucket_order: List[str] = []
    claimed: Dict[int, str] = {}

    def add(r) -> None:
        if not r.get('in_scope', True):
            return
        cat = _cat(r)
        if not cat or cat in TIER_CATEGORIES:
            return
        bid = int(r.get('ahead_target_bodyId',
                        r.get('proposal_bodyId')))
        raw_type = str(r.get('ahead_target_type',
                             r.get('proposal_type')) or '?')
        # Every expansion bin carries the ordered leaf token
        # {T}(out-map) / {T}>{src} / {T}(no_source) / untyped.
        token = str(r.get('candidate_annotation') or '').strip() or raw_type
        tag = _dup(r)
        cur = claimed.get(bid)
        # single bucket per neuron per branch, category precedence
        if cur is not None and DEDUP_RANK.get(_cat_of_key(cur), 0) >= \
                DEDUP_RANK.get(cat, 0):
            return
        if cur is not None and cat != _cat_of_key(cur):
            buckets[cur]['ids'].discard(bid)
            for m in ('types', 'leaf_token', 'tags', 'sort_key'):
                buckets[cur][m].pop(bid, None)
        claimed[bid] = cat
        rec = ensure_bucket(buckets, bucket_order, cat)
        rec['ids'].add(bid)
        rec['types'][bid] = raw_type
        rec['leaf_token'][bid] = token
        rec['tags'][bid] = tag
        rec['sort_key'][bid] = f'{token} {tag}'.strip()

    for r in res.get('suspicious', []):
        add(r)
    for r in res.get('deep', []):
        add(r)
    for f in res.get('fills', []):
        if f.get('side') == 'source' \
                and f.get('fill_class') == 'out_of_pool':
            add(f)
    return buckets, bucket_order


#: Stage 5d leaf suffix (plan-tmvev-backward-expansion-evidence D8): the
#: reciprocal verdict rides the legend leaf so the picture and
#: ``expansion/backward_matches.csv`` state the same thing.  ``not-checked``
#: deliberately has NO entry: an unchecked member keeps a bare leaf rather
#: than implying a negative.
_BACKWARD_LEAF_TAG = {'high': '· high',
                      'medium': '· medium',
                      'low': '· low'}


def _cat_of_key(key: str) -> str:
    return key.split(' · ')[0]


def build_invader_buckets(res, suspicious_cap: int, threshold,
                          floors: Optional[Dict] = None,
                          pair_key: Optional[Tuple[str, str]] = None,
                          sibling_index: Optional[Dict[int,
                                                       Tuple[str, str]]] = None,
                          backward_lookup=None,
                          null_bar: Optional[float] = None,
                          branch_target_type: Optional[str] = None
                          ) -> Tuple[Dict[str, Dict], List[str]]:
    """Revision 3.6/3.8: bucket every out-of-pool invader of one branch.

    EVERY rendered neuron passes the morph rule (user 2026-09-12,
    default included): structural bins (sibling/backward/
    alternate-chain) only name the bucket — they never bypass
    qualification, and failures stay in the CSVs (a morph-failing
    sibling is still structurally explainable, so it does not demote
    to examinees either).  Rule v2 with the binding native floor:
    `pool_ref >= floor` when the branch has one, otherwise
    `query morph >= bar` (null-calibrated when available, else the
    factor x pooled-average threshold).

    Rows carry the precomputed classification from
    ``MappingValidator.annotate_invaders`` (structural facts outrank
    morphology): sibling / alternate-chain / backward labels are used
    verbatim; only UNEXPLAINED invaders (hollow-backward, unmapped) are
    eligible for morph-qualified `candidates` promotion — qualification
    rule v2 `query_or_pool` (Rev 3.7): query-based morph >= threshold OR
    native pool_ref >= floor.  Untyped invaders stay `examinees ·
    untyped` (never promoted, never `?`).  Out-of-pool fill proposals
    (proposed unconditionally per Rev 3.8) merge into the same buckets.

    Returns (buckets {key: {'ids': set, 'types': {bid: type}}},
    insertion-ordered bucket keys).
    """
    def pd_isna(v):
        return v is None or (isinstance(v, float) and v != v)

    def type_label(v):
        # CSV round-trips can turn missing labels into NaN
        return '?' if v is None or pd_isna(v) else str(v)

    floor = floors.get(pair_key) if floors and pair_key else None
    bar_a = null_bar if null_bar is not None else threshold

    def morph_ok(r):
        # Rule v2 (user 2026-09-12): the native pool-ref floor is
        # BINDING when the branch has one — the cross-dataset
        # Track-A score has a heavy-tailed baseline (T1: query 0.226
        # >= 0.176 but pool-ref 0.013 vs floor 0.83), so Track A never
        # overrides it.  Track A (null-calibrated when possible)
        # qualifies only branches with no usable native floor.
        pr = r.get('morph_pool_ref')
        if floor is not None:
            return (pr is not None and not pd_isna(pr) and pr >= floor)
        m = r.get('morph_v2_similarity')
        return (bar_a is not None and m is not None and not pd_isna(m)
                and m >= bar_a)

    def bucket_key(r):
        cls = r.get('invader_class')
        if cls == 'untyped':
            # Rev 3.6 untyped policy: explicit review root, never promoted
            return 'examinees · untyped'
        # Rev 3.11 (user 2026-09-13): SAME-TYPE extras (ahead/proposal
        # type == the branch target type, outside the widened pool --
        # their annotation group has no chain to the parent) are the
        # ONLY fills of the mapped type ("the qualified ones").
        # Unqualified same-type extras get the explicit
        # `family · {type}` label -- family by annotation, unvalidated
        # by evidence; both render.
        atype = type_label(r.get('proposal_type',
                                 r.get('ahead_target_type')))
        if cls == 'same-type' or (
                branch_target_type is not None
                and atype == str(branch_target_type)):
            return 'fill' if morph_ok(r) else 'family'
        # EVERY rendered STRUCTURAL member must pass the morph rule
        # (user 2026-09-12, default included): sibling/backward labels
        # only name the bin — they never bypass qualification; failures
        # stay in the CSVs (a morph-failing sibling is still structurally
        # explainable, so it does not demote to examinees either).
        if cls in ('sibling', 'backward') and not morph_ok(r):
            return None
        # Rev 3.11 (user 2026-09-13): alternate-chain residues are
        # mapped-set members — they are SIBLINGS (family), not fills.
        # `backward` is reserved for OUT-OF-SCOPE mapped types (home
        # outside the queried population; linker detail stays in
        # alt_chain_of_parent).
        if cls == 'sibling' or cls == 'alternate-chain':
            return 'sibling'
        if cls == 'backward':
            if r.get('in_query_family'):
                return 'sibling'
            mapped = r.get('backward_maps_to')
            if mapped and not (isinstance(mapped, float)
                               and mapped != mapped):
                return f'backward · {mapped}'
        bid = int(r.get('ahead_target_bodyId',
                        r.get('proposal_bodyId')))
        atype = type_label(r.get('ahead_target_type',
                                 r.get('proposal_type')))
        if sibling_index is not None or backward_lookup is not None:
            key = invader_bucket_key(bid, atype, sibling_index or {},
                                     backward_lookup)
            if key != 'examinees':
                return key
        # unexplained: morph-qualified -> candidates; otherwise the
        # examinees review root (the "ranked ahead, failed
        # qualification" signal survives by design)
        if morph_ok(r):
            return 'candidates'
        return 'examinees'

    buckets: Dict[str, Dict] = {}
    bucket_order: List[str] = []
    claimed: Dict[int, Tuple[int, str]] = {}
    per_src: Dict[int, int] = {}
    for r in res['suspicious']:
        sid = int(r['source_bodyId'])
        if per_src.get(sid, 0) >= suspicious_cap:
            continue
        per_src[sid] = per_src.get(sid, 0) + 1
        key = bucket_key(r)
        if key is None:
            continue
        assign_invader(claimed, buckets, bucket_order,
                       int(r['ahead_target_bodyId']),
                       type_label(r.get('ahead_target_type')), key)
    # Revision 3.8 deep-window candidates: out-of-pool homologs that
    # rank BELOW the pool best (not invaders) join the same buckets —
    # structural labels respected, morph-qualified unexplained ones
    # become `candidates`; unqualified deep rows stay in
    # deep_candidates.csv only (examinees remains an ahead-only
    # concept).
    for r in res.get('deep', []):
        key = bucket_key(r)
        if key is None or key in ('examinees', 'examinees · untyped'):
            continue
        assign_invader(claimed, buckets, bucket_order,
                       int(r['ahead_target_bodyId']),
                       type_label(r.get('ahead_target_type')), key)
    # Revision 3.8: out-of-pool fill proposals are unconditional and
    # merge into the same buckets (candidates when morph-qualified and
    # unexplained; sibling/backward labels respected).  Untyped
    # proposals NEVER render (Rev 3.6 untyped policy) — they stay in
    # gap_fill_proposals.csv as evidence.
    for f in res['fills']:
        if f.get('side') != 'source' \
                or f.get('fill_class') != 'out_of_pool':
            continue
        key = bucket_key(f)
        if key is None or key in ('examinees', 'examinees · untyped'):
            continue
        assign_invader(claimed, buckets, bucket_order,
                       int(f['proposal_bodyId']),
                       type_label(f.get('proposal_type')), key)
    return buckets, bucket_order


def neuron_bbox(nrn) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(lo, hi) xyz extent of a navis neuron (tree nodes or mesh verts)."""
    pts = None
    try:
        pts = nrn.nodes[['x', 'y', 'z']].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        pts = None
    if pts is None or not len(pts):
        verts = getattr(nrn, 'vertices', None)
        if verts is not None:
            try:
                pts = np.asarray(verts, dtype=float)[:, :3]
            except Exception:  # noqa: BLE001
                pts = None
    if pts is None or not len(pts):
        return None
    return pts.min(axis=0), pts.max(axis=0)


def check_scene_identities(viz, scene_bboxes: Dict[int,
                           Tuple[np.ndarray, np.ndarray]],
                           undo_matrix: Optional[np.ndarray] = None
                           ) -> List[Tuple[str, int, str]]:
    """Revision 3.5 Issue 6c debug self-check: every legend leaf's
    skeleton geometry must sit inside its own labeled neuron's bbox.

    Catches trace-identity slips like the R5 scene where 50274's geometry
    rendered under 61430's legend item.  Only ``mode='lines'`` traces are
    checked (soma-mesh companions can legitimately poke outside the node
    bbox by their radius).  ``undo_matrix`` maps trace coordinates back
    to raw space before comparison — the backend applies the FAFB/FLYWIRE
    tilt correction (a rigid rotation about the template center) to every
    rendered layer, so the loaded bboxes must be compared in raw space.
    Returns (item, labeled_bid, reason) triples.
    """
    problems: List[Tuple[str, int, str]] = []
    for tr in getattr(getattr(viz, 'fig_3d', None), 'data', []) or []:
        if str(getattr(tr, 'mode', '')) != 'lines':
            continue
        meta = getattr(tr, 'meta', None) or {}
        lg = meta.get('drocatLegend')
        if not isinstance(lg, dict) or lg.get('kind') != 'neuron':
            continue
        item = str(lg.get('item') or '')
        m = re.match(r'^(\d+)', item)
        if not m:
            continue
        bid = int(m.group(1))
        bbox = scene_bboxes.get(bid)
        if bbox is None:
            problems.append((item, bid, 'no loaded bbox for the label'))
            continue
        try:
            pts = np.stack([np.asarray(tr.x, dtype=float),
                            np.asarray(tr.y, dtype=float),
                            np.asarray(tr.z, dtype=float)], axis=1)
        except Exception:  # noqa: BLE001
            continue
        pts = pts[~np.isnan(pts).any(axis=1)]
        if not len(pts):
            continue
        if undo_matrix is not None:
            try:
                pts = (undo_matrix @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            except Exception:  # noqa: BLE001
                pass
        lo, hi = bbox
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        slack = 0.02 * diag
        overflow = float(max(
            np.max(lo - pts.min(0)), np.max(pts.max(0) - hi), 0.0))
        if overflow > slack:
            center = pts.mean(axis=0)
            best_bid, best_d = min(
                ((b, float(np.linalg.norm((bl + bh) / 2 - center)))
                 for b, (bl, bh) in scene_bboxes.items()),
                key=lambda kv: kv[1])
            problems.append((item, bid,
                             'geometry outside the labeled neuron bbox by '
                             f'{overflow:.0f} units (slack {slack:.0f}); '
                             f'nearest neuron {best_bid} '
                             f'(center distance {best_d:.0f})'))
    return problems


def pool_rows_by_host(pool_rows) -> Dict[str, List[Dict]]:
    """The ``pooling`` pool grouped by the scene that may host it: the TYPE
    of each candidate target's best source.  A pool row belongs to no branch,
    so its source's type is its only scene address — one row per target means
    one host, and a type no branch group covers renders no layer (the render
    pass names those rows rather than dropping them quietly).
    """
    hosts: Dict[str, List[Dict]] = defaultdict(list)
    for r in pool_rows or []:
        hosts[str(r.get('best_source_type') or '')].append(r)
    return hosts


def compute_out_map_sources(branch_list) -> List[int]:
    """The source-side gap of one parent type: annotated FAFB bodyIds
    that NO branch pool claims (outside every refined ``source_pool`` —
    the pool-refinement residue, never scanned, never proposed).  Sorted
    ascending for deterministic rendering."""
    parent_pool, claimed = set(), set()
    for pair, _res in branch_list:
        parent_pool |= {int(b) for b in pair.parent_source_pool}
        claimed |= {int(b) for b in pair.source_pool}
    return sorted(parent_pool - claimed)


def plan_scene_parents(parents: Dict[Tuple[str, str], List],
                       max_scenes: int) -> Tuple[List, List[str]]:
    """Order the parent scene groups and apply the ``max_scenes`` cap.

    Returns ``(kept, dropped)``: the ordered
    ``((query, source_type), branches)`` items and the NAMES of the parent
    types the cap removed. A cap that fires must name them — a Branches-tab
    row with no scene is otherwise indistinguishable from a branch that had
    nothing to review. ``max_scenes <= 0`` caps nothing (the default: every
    parent gets a review scene).
    """
    scenes = sorted(parents.items(),
                    key=lambda kv: -sum(len(p.source_pool)
                                        for p, _ in kv[1]))
    if 0 < max_scenes < len(scenes):
        # An item is ((query, source_type), branches) — the parent TYPE is the
        # SECOND element of the item's KEY, not the payload.
        return (scenes[:max_scenes],
                [key[1] for key, _ in scenes[max_scenes:]])
    return scenes, []


def render_pair_scenes(validator, per_pair_res: Dict) -> None:
    """Render one branch-structured scene per parent mapping group."""
    import navis

    cfg = validator.cfg
    viz_dir = validator.run_dir / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)
    from visualize_skeleton import (VisualizeSkeleton,
                                    dataset_native_space,
                                    dataset_render_space,
                                    transform_neurons_to_space)
    from morphology import _load_cached_skeleton_file

    # The backend renders injected overlay layers AS-IS in its template
    # space (visualize_skeleton's template target = the RENDER space), so
    # the scene must deliver EVERY neuron — sources and bridged targets —
    # in the source dataset's RENDER space.  FAFB/BANC are identity
    # (native == render); male-cns/hemibrain/manc raw-voxel frames differ
    # from their render frames (the 2026-09-18 MCNS→BANC misplacement:
    # raw-space neurons drawn on the JRCFIB2022M-nanometre mesh).
    src_native = dataset_native_space(cfg.source_dataset)
    src_space = dataset_render_space(cfg.source_dataset)
    tgt_native = dataset_native_space(cfg.target_dataset)
    tgt_render = dataset_render_space(cfg.target_dataset)
    needs_transform = src_space != tgt_render
    src_needs_bridge = src_native != src_space
    validator.log(
        f'[stage 4] scenes in {cfg.source_dataset} template ({src_space}); '
        f'target {tgt_native} -> {tgt_render}'
        if needs_transform else
        f'[stage 4] scenes in {src_space} (no transform needed)')

    def bridge_to_scene_space(neurons):
        """Bridge SOURCE-dataset neurons into the scene's render space
        (identity when native == render).  Fail-open: unbridgeable
        neurons are dropped with the helper's own warning."""
        if not neurons or not src_needs_bridge:
            return neurons
        xformed = transform_neurons_to_space(
            navis.NeuronList(neurons), src_native, src_space,
            validate_bounds=True, verbose=False)
        if isinstance(xformed, navis.NeuronList):
            xformed = list(xformed)
        return xformed or []

    def bridge_targets_to_scene_space(raw):
        """TARGET neurons into the scene's render space, as
        ``(neurons, dropped)``.  The dropped count matters: a neuron the
        bounds validation refuses must be reported, never silently lost."""
        if not raw or not needs_transform:
            return list(raw), 0
        xformed = transform_neurons_to_space(
            navis.NeuronList(raw), tgt_native, src_space,
            validate_bounds=True, verbose=False)
        if isinstance(xformed, navis.NeuronList):
            xformed = list(xformed)
        xformed = xformed or []
        return xformed, len(raw) - len(xformed)

    # group pairs into parent mapping groups (query, source_type)
    parents: Dict[Tuple[str, str], List] = {}
    for pair in validator.pairs:
        # per_pair_res is keyed by (query, source_type, target_type);
        # fall back to the bare pair.key for legacy/fixture dicts.
        res = per_pair_res.get((pair.query,) + pair.key) \
            or per_pair_res.get(pair.key)
        if res:
            parents.setdefault((pair.query, pair.source_type),
                               []).append((pair, res))
    scenes, dropped = plan_scene_parents(parents, cfg.max_scenes)
    if dropped:
        # Cap-and-warn (user decision 2026-09-19): the cap must not fire
        # silently. Routing through validator.log puts the line on stdout AND
        # in notes/README.txt, where the UI tab surfaces it post-run.  The
        # dropped parents are named because a Branches-tab row without a
        # scene is otherwise indistinguishable from a rendering failure.
        validator.log(
            f'[stage 4] scene cap: rendering {cfg.max_scenes} of '
            f'{len(scenes) + len(dropped)} candidate branch scenes; '
            f"not rendered: {', '.join(dropped)} — set Max Scenes to 0 to "
            'render every parent, or split the query into separate runs')

    skel_dir = (Path(validator.profiler.cache_dir) /
                cfg.target_dataset.replace(':', '_').replace('.', '_') /
                'skeletons' / 'raw_skeletons')

    # `pooling` mode (plan-tmvev-pooling-mode.md): the unsupervised pool is
    # hosted by the parent group of the source that reached each target BEST
    # — a pool row has no branch, so the source type is its only scene
    # address.  A source type no branch covers has no scene to host it, and
    # that must be said rather than left as a missing layer.
    pooling_hosts = pool_rows_by_host(
        (getattr(validator, '_pooling', None) or {}).get('pool'))
    unhosted = sorted(set(pooling_hosts) - {k[1] for k, _ in scenes})
    if unhosted:
        validator.log(
            f'[stage 4] ! pooling: '
            f'{sum(len(pooling_hosts[t]) for t in unhosted)} pooled '
            f'target(s) have no scene — their best source has type '
            f"{', '.join(unhosted)}, which no branch group covers. "
            'pooling/pooling_pool.csv holds every one of them.')

    def load_target_neurons(bids: List[int]):
        neurons, dropped = [], []
        for bid in bids:
            try:
                path = skel_dir / f'{bid}.swc.zst'
                if path.exists():
                    nrn = _load_cached_skeleton_file(path)
                else:
                    from morphology import fetch_skeleton_on_demand
                    nrn = fetch_skeleton_on_demand(cfg.target_dataset, bid)
            except Exception as exc:  # noqa: BLE001
                dropped.append((bid, str(exc)))
                continue
            if nrn is None:
                dropped.append((bid, 'no skeleton'))
                continue
            try:
                nrn.id = int(bid)
                # Revision 3.5 Issue 6a: navis names plotly traces after
                # the neuron name; exact unique names keep the backend
                # trace-identity resolver on name matching instead of the
                # positional fallback that rendered 50274's geometry
                # under 61430's legend item in the R5 scene.
                nrn.name = str(int(bid))
                nrn._drocat_source_dataset = cfg.target_dataset
            except Exception:  # noqa: BLE001
                pass
            neurons.append(nrn)
        return neurons, dropped

    def load_query_neurons(bids: List[int]):
        try:
            from morphology import _resolve_fafb_skeleton_trees
            trees = _resolve_fafb_skeleton_trees(
                cfg.source_dataset, [int(b) for b in bids])
            neurons = []
            for b in bids:
                if b in trees:
                    trees[b]._drocat_source_dataset = cfg.source_dataset
                    try:
                        trees[b].name = str(int(b))
                    except Exception:  # noqa: BLE001
                        pass
                    neurons.append(trees[b])
            dropped = [(b, 'no skeleton') for b in bids if b not in trees]
            # The scene renders in the source RENDER space; raw cached
            # skeletons are in the dataset's native frame.
            neurons = bridge_to_scene_space(neurons)
            return neurons, dropped
        except Exception as exc:  # noqa: BLE001
            return [], [(b, str(exc)) for b in bids]

    for si, ((query, src_type), branch_list) in enumerate(scenes, 1):
        validator.log(f'[stage 4] scene {si}/{len(scenes)}: {src_type} '
                      f'({len(branch_list)} branches) [{query}]')
        try:
            entries: List[Tuple[str, object, str]] = []  # (group, neurons, color)
            group_names: List[str] = []

            type_overrides: Dict[int, str] = {}
            # Revision 3.12 leaf detail: the standalone (dup) tag per
            # bodyId, and the bare-category roots that must always render
            # (forceRoot) even when they hold a single leaf.
            tag_overrides: Dict[int, str] = {}
            # Leaf-level TYPE override — set ONLY for the expansion bins, so
            # it never leaks a category root into a query/tier leaf (unlike
            # the root-level `type_overrides`).
            leaf_type_overrides: Dict[int, str] = {}
            # Per-leaf ORDER key (type + suffix) so expansion leaves sort by
            # target type, not bodyId.
            sort_overrides: Dict[int, str] = {}
            force_roots: set = set()
            # Issue 6c: loaded (post-transform) bbox per bodyId in scene
            # space, for the optional legend-leaf self-check.
            scene_bboxes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

            def add_layer(group, neurons, color, legend_types, category,
                          leaf_types=None, leaf_tags=None, leaf_sorts=None,
                          default_off=False):
                # unique layer name; ' :: ' splits it from the merged
                # legend group (backend splits on ' :: ')
                layer_name = f'{group} :: {category}'
                for n, lt in zip(neurons, legend_types):
                    # Root stamp (the legend group row).
                    try:
                        n._drocat_legend_type = lt
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        type_overrides[int(n.id)] = lt
                    except Exception:  # noqa: BLE001
                        pass
                    if leaf_types is not None:
                        # Revision 3.12 per-leaf detail, stamped on the
                        # NEURON OBJECT (the backend reads these by
                        # attribute; the int-keyed maps are a fallback):
                        # the ordered type override ({T}(out-map) /
                        # {T}>{src} / {T}(no_source) / untyped), (dup)
                        # tag, and the type-first order key.
                        try:
                            n._drocat_legend_type_override = leaf_types.get(
                                int(n.id))
                            tg = (leaf_tags or {}).get(int(n.id)) or ''
                            if tg:
                                n._drocat_legend_leaf_tag = tg
                            sk = (leaf_sorts or {}).get(int(n.id)) or ''
                            if sk:
                                n._drocat_legend_sort_key = sk
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        bb = neuron_bbox(n)
                    except Exception:  # noqa: BLE001
                        bb = None
                    if bb is not None:
                        try:
                            scene_bboxes[int(n.id)] = bb
                        except Exception:  # noqa: BLE001
                            pass
                if default_off:
                    # Plan I §4: sibling layers start hidden (legend row
                    # present, eye off).  Stamped on the FINAL neuron
                    # objects (post-transform) so the trace loop reads it.
                    for n in neurons:
                        try:
                            n._drocat_legend_default_off = True
                        except Exception:  # noqa: BLE001
                            pass
                if leaf_types is not None:
                    for bid, ltv in leaf_types.items():
                        type_overrides[int(bid)] = ltv
                        leaf_type_overrides[int(bid)] = ltv
                    for bid, tg in (leaf_tags or {}).items():
                        if tg:
                            tag_overrides[int(bid)] = tg
                    for bid, sk in (leaf_sorts or {}).items():
                        if sk:
                            sort_overrides[int(bid)] = sk
                    # forceRoot: a bare-category root must always render,
                    # even when it holds a single leaf.
                    force_roots.add(str(category))
                entries.append((layer_name, navis.NeuronList(neurons),
                                color))
                group_names.append(layer_name)

            for bi, (pair, res) in enumerate(branch_list, 0):
                sig = _linker_sig(pair)
                group = (f'{src_type} → {pair.target_type} · {sig}'
                         if basis_is_row_evidence(pair.pool_basis) else
                         f'{src_type} → {pair.target_type} · full pool')

                # query (source) neurons — first layer of the branch
                live_src, skipped_src = [], []
                for r in res['rows']:
                    (live_src if r['verdict'] != 'skipped'
                     else skipped_src).append(int(r['source_bodyId']))
                if live_src:
                    qn, qdrop = load_query_neurons(live_src)
                    for bid, why in qdrop:
                        validator.log(f'    ! query {bid} skipped ({why})')
                    if qn:
                        add_layer(group, qn, CATEGORY_COLORS['query'],
                                  [f'query · {src_type}'] * len(qn),
                                  category='query')
                if skipped_src:
                    from morphology import _resolve_fafb_skeleton_trees
                    try:
                        trees = _resolve_fafb_skeleton_trees(
                            cfg.source_dataset, skipped_src)
                        sn = [trees[b] for b in skipped_src if b in trees]
                    except Exception:  # noqa: BLE001
                        sn = []
                    sn = bridge_to_scene_space(sn)
                    if sn:
                        add_layer(group, sn, UNASSIGNED_COLOR,
                                  [f'query · {src_type} · unassigned']
                                  * len(sn), category='unassigned')

                # pool neurons by Revision 3.3 ladder category
                categories: Dict[str, List[int]] = defaultdict(list)
                for tbid, cat in res['target_categories'].items():
                    categories[cat].append(int(tbid))
                cat_colors = {c: CATEGORY_COLORS.get(
                    c, CATEGORY_COLORS['unmatched'])
                    for c in ('matched', 'verified', 'borderline',
                              'unmatched')}
                for cat in ('matched', 'verified', 'borderline',
                            'unmatched'):
                    ids = sorted(set(categories.get(cat, [])))
                    if not ids:
                        continue
                    root = f'{cat} · {pair.target_type}'
                    raw, dropped = load_target_neurons(ids)
                    for bid, why in dropped:
                        validator.log(f'    ! {root}: {bid} unavailable '
                                      f'({why}) — no leaf, see CSVs')
                    if not raw:
                        continue
                    neurons = raw
                    if needs_transform:
                        xformed = transform_neurons_to_space(
                            navis.NeuronList(raw), tgt_native, src_space,
                            validate_bounds=True, verbose=False)
                        if isinstance(xformed, navis.NeuronList):
                            xformed = list(xformed)
                        if len(raw) - len(xformed or []):
                            validator.log(
                                f'    ! {root}: '
                                f'{len(raw) - len(xformed or [])} '
                                'neuron(s) dropped by bounds validation')
                        if not xformed:
                            continue
                        neurons = xformed
                    add_layer(group, neurons, cat_colors[cat],
                              [root] * len(neurons), category=cat)

                # Backward source-candidates (plan-
                # backward-source-status.md D-B8/D-B12, RE-AIMED per plan
                # `-samename-first-consumers.md` §10, user option 2):
                # OUT-OF-MAP sources (claimed by no branch) whose
                # best-ranked scan hits land in THIS branch's pool and
                # pass the run null bar — the true mirror of candidate
                # admission.  Advisory only (D-B11).  Leaf type suffix =
                # the source's own type (provenance); `(dup)` marks
                # sources that are candidates for several branches.
                bkey = (query,) + pair.key
                cand_rows = (getattr(validator, '_source_candidates',
                                     {}) or {}).get(bkey) or []
                if cand_rows:
                    by_src = {}
                    for c in cand_rows:
                        by_src.setdefault(int(c['source_bodyId']), c)
                    cids = sorted(by_src)
                    cn, cd = load_query_neurons(cids)
                    for bid, why in cd:
                        validator.log(f'    ! source-candidate {bid} '
                                      f'skipped ({why})')
                    if cn:
                        multi = getattr(validator,
                                        '_source_candidates_multi',
                                        set()) or set()
                        root = f'source-candidates · {pair.target_type}'
                        leaf_types = {}
                        leaf_tags = {}
                        leaf_sorts = {}
                        for n in cn:
                            c = by_src.get(int(n.id)) or {}
                            stype = str(c.get('source_type') or '?')
                            leaf_types[int(n.id)] = stype
                            tag = '(dup)' if int(n.id) in multi else ''
                            if tag:
                                leaf_tags[int(n.id)] = tag
                            leaf_sorts[int(n.id)] = \
                                f'{stype} {tag}'.strip()
                        add_layer(group, cn,
                                  CATEGORY_COLORS['source-candidates'],
                                  [root] * len(cn),
                                  category='source-candidates',
                                  leaf_types=leaf_types,
                                  leaf_tags=leaf_tags or None,
                                  leaf_sorts=leaf_sorts,
                                  default_off=True)

                # -----------------------------------------------------------------
                # Revision 3.12: expansion bins are the EXPORTED categories
                # (S2 partition) — the scene reads `category` /
                # `candidate_annotation` / `in_scope`, so the legend and
                # the CSV cannot disagree.  In-scope rows only; out-of-
                # scope (morph-failed) rows stay CSV-only.  One root per
                # category (+ annotation for candidates); one bucket per
                # neuron per branch.
                buckets, bucket_order = build_category_buckets(
                    res, validator, pair,
                    cfg.suspicious_per_source_cap)

                # Family/relative members are enumerated per branch (they
                # are not evidence rows); fold the validator's cached rows
                # for THIS branch into the same buckets, carrying the
                # ordered leaf token ({T}(out-map) for family members —
                # unmapped bodyIds of the in-map type) and (dup).
                for cat_name, rows_attr in (('family', '_family_rows'),
                                            ('relative', '_relative_rows')):
                    for r in (getattr(validator, rows_attr, []) or []):
                        if str(r.get('source_type')) != pair.source_type \
                                or str(r.get('target_type')) != pair.target_type:
                            continue
                        bid = int(r['ahead_target_bodyId'])
                        rec = ensure_bucket(buckets, bucket_order, cat_name)
                        tname = str(r.get('ahead_target_type') or '?')
                        token = str(
                            r.get('candidate_annotation') or '').strip() \
                            or tname
                        tag = '(dup)' if r.get('dup') else ''
                        rec['ids'].add(bid)
                        rec['types'][bid] = tname
                        rec['leaf_token'][bid] = token
                        rec['tags'][bid] = tag
                        rec['sort_key'][bid] = f'{token} {tag}'.strip()

                # Stage 5d (plan-tmvev-backward-expansion-evidence): ride the
                # reciprocal verdict on the leaf tag so the picture and
                # expansion/backward_matches.csv cannot disagree.  Unscanned
                # members (pass off, beyond cap) keep a bare leaf.
                bev_labels = (getattr(validator, '_backward_label_by_bid',
                                      None) or {})
                if bev_labels:
                    for key in bucket_order:
                        cat_name = _cat_of_key(key)
                        if cat_name not in ('candidates', 'family',
                                            'relative'):
                            continue
                        rec = buckets[key]
                        for bid in rec['ids']:
                            suffix = _BACKWARD_LEAF_TAG.get(
                                bev_labels.get(int(bid)) or '')
                            if not suffix:
                                continue
                            tag = f"{rec['tags'].get(bid) or ''} " \
                                  f"{suffix}".strip()
                            rec['tags'][bid] = tag
                            rec['sort_key'][bid] = (
                                f"{rec['leaf_token'].get(bid, '')} "
                                f"{tag}").strip()

                def add_invader_layers():
                    for key in sorted(bucket_order, key=lambda k: (
                            CATEGORY_RENDER_PRIORITY.get(
                                _cat_of_key(k), 50), k)):
                        rec = buckets[key]
                        ids = sorted(rec['ids'])
                        if not ids:
                            continue
                        prefix = _cat_of_key(key)
                        color = CATEGORY_COLORS.get(
                            prefix, CATEGORY_COLORS['examinees'])
                        root = bucket_root_label(key, rec['types'])
                        raw, dropped = load_target_neurons(ids)
                        for bid, why in dropped:
                            validator.log(f'    ! {root}: {bid} '
                                          f'unavailable ({why})')
                        if not raw:
                            continue
                        neurons = raw
                        if needs_transform:
                            xformed = transform_neurons_to_space(
                                navis.NeuronList(raw), tgt_native,
                                src_space, validate_bounds=True,
                                verbose=False)
                            if isinstance(xformed, navis.NeuronList):
                                xformed = list(xformed)
                            if not xformed:
                                continue
                            neurons = xformed
                        # Per-leaf detail (Revision 3.12): the qualified type
                        # token and the standalone (dup) tag ride on the
                        # bodyId leaf; the root stays bare.
                        leaf_types = {int(n.id): rec['leaf_token'].get(
                            int(n.id), rec['types'].get(int(n.id), '?'))
                            for n in neurons}
                        leaf_tags = {int(n.id): rec['tags'].get(int(n.id), '')
                                     for n in neurons}
                        leaf_sorts = {int(n.id): rec['sort_key'].get(
                            int(n.id), '') for n in neurons}
                        add_layer(group, neurons, color,
                                  [root] * len(neurons), category=root,
                                  leaf_types=leaf_types,
                                  leaf_tags=leaf_tags,
                                  leaf_sorts=leaf_sorts,
                                  default_off=(prefix == 'sibling'))

                add_invader_layers()

            # Plan I follow-up: the source-side gap as its own branch —
            # annotated FAFB query neurons of this type that received no
            # assigned partner (never scanned, or scanned without a
            # verdict pair).  Rendered next to the branches so the gap can
            # be compared against `candidates` directly.
            out_map = (getattr(validator, '_out_map_by_type', {}) or {}) \
                .get((query, src_type), [])
            if not out_map:
                out_map = compute_out_map_sources(branch_list)
            if out_map:
                on, on_drop = load_query_neurons(out_map)
                for bid, why in on_drop:
                    validator.log(f'    ! out-map query {bid} '
                                  f'unavailable ({why})')
                if on:
                    label = f'out-map query · {src_type}'
                    for n in on:
                        type_overrides[int(n.id)] = label
                    add_layer(f'{src_type} · out-map query', on,
                              CATEGORY_COLORS['out-map query'],
                              [label] * len(on), category='out-map query')
                # The expansion's found targets for this type: top-k
                # connectivity-ranked BANC/MCNS neurons outside the in-map
                # claims, one layer beside their source neurons.
                exp_rows = [r for r in
                            (getattr(validator,
                                     '_out_map_expansion_rows', None)
                             or [])
                            if r.get('query') == query
                            and r.get('source_type') == src_type
                            and r.get('morph_qualified') is not False]
                if exp_rows:
                    best = {}
                    for r in sorted(exp_rows, key=chain_key):
                        best.setdefault(int(r['target_bodyId']), r)
                    cand_ids = sorted(best)
                    raw, dropped = load_target_neurons(cand_ids)
                    for bid, why in dropped:
                        validator.log(f'    ! out-map candidates: {bid} '
                                      f'unavailable ({why})')
                    if raw:
                        neurons = raw
                        if needs_transform:
                            xformed = transform_neurons_to_space(
                                navis.NeuronList(raw), tgt_native,
                                src_space, validate_bounds=True,
                                verbose=False)
                            if isinstance(xformed, navis.NeuronList):
                                xformed = list(xformed)
                            if not xformed:
                                neurons = []
                            else:
                                neurons = xformed
                        if neurons:
                            c_label = f'out-map candidates · {src_type}'
                            for n in neurons:
                                type_overrides[int(n.id)] = c_label
                            add_layer(f'{src_type} · out-map candidates',
                                      neurons,
                                      CATEGORY_COLORS['out-map candidates'],
                                      [c_label] * len(neurons),
                                      category='out-map candidates')

            # `pooling` mode: the unsupervised pool for THIS parent type,
            # one layer beside the branches it was compared with.  Nothing
            # is re-gated for the picture — the leaf carries the exported
            # leaf token plus the post-hoc mapper cell (and the morph
            # verdict where the gate scored it), so the scene and
            # `pooling/pooling_pool.csv` cannot disagree.
            pool_rows = pooling_hosts.get(src_type) or []
            if pool_rows:
                by_bid = {}
                for r in pool_rows:
                    by_bid.setdefault(int(r['target_bodyId']), r)
                raw, drop = load_target_neurons(sorted(by_bid))
                for bid, why in drop:
                    validator.log(f'    ! pooling: {bid} unavailable '
                                  f'({why}) — no leaf, see the CSV')
                neurons, lost = bridge_targets_to_scene_space(raw)
                if lost:
                    validator.log(f'    ! pooling: {lost} neuron(s) dropped '
                                  'by bounds validation')
                if neurons:
                    root = f'pooling · {src_type}'
                    p_types, p_tags, p_sorts = {}, {}, {}
                    for n in neurons:
                        r = by_bid.get(int(n.id)) or {}
                        token = str(r.get('leaf') or r.get('target_type')
                                    or '?')
                        bits = [str(r.get('mapper_cell') or '')]
                        if r.get('morph_gate') == 'scored':
                            bits.append('morph ✓'
                                        if r.get('morph_qualified')
                                        else 'morph ✗')
                        tag = ' · '.join(b for b in bits if b)
                        p_types[int(n.id)] = token
                        if tag:
                            p_tags[int(n.id)] = tag
                        p_sorts[int(n.id)] = f'{token} {tag}'.strip()
                    add_layer(f'{src_type} · pooling', neurons,
                              CATEGORY_COLORS['pooling'],
                              [root] * len(neurons), category=root,
                              leaf_types=p_types, leaf_tags=p_tags or None,
                              leaf_sorts=p_sorts)

            if not entries:
                validator.log('    nothing to render for this parent')
                continue

            viz = VisualizeSkeleton(
                dataset=cfg.source_dataset,
                neuron_layers=[],   # all-custom layers (query + matched)
                custom_neurons=[(g, n) for g, n, _ in entries],
                custom_layer_names=group_names,
                neuron_colors=[c for _, _, c in entries],
                brain_mesh='native',
                skeleton_mode='line',
                legend_mode='tree',
                neuron_alpha=cfg.neuron_alpha,
                # pair scenes carry no connectivity data; synapse plotting
                # also indexes per-layer color tuples sized at init and is
                # not expanded for the custom layers injected post-init
                skip_synapse=True,
                output_dir=str(viz_dir),
                saveas=f'branches_{src_type}',
                verbose=False,
                # a batch run renders up to max_scenes pages; opening each
                # one in the browser is noise — the report's Scenes tab
                # links them
                show_fig=False,
            )
            viz._drocat_expand_roots = True   # Plan I §4: roots start expanded
            viz._drocat_legend_type_overrides = dict(type_overrides)
            viz._drocat_legend_leaf_type_overrides = dict(leaf_type_overrides)
            viz._drocat_legend_tag_overrides = dict(tag_overrides)
            viz._drocat_legend_sort_overrides = dict(sort_overrides)
            viz._drocat_legend_force_roots = set(force_roots)
            viz.plot_neurons()
            validator.log(f'    scene written: {viz.save_folder}')
            if getattr(cfg, 'scene_selfcheck', False):
                # Revision 3.5 Issue 6c debug self-check: each legend
                # leaf's geometry must sit on its own labeled neuron.
                # The backend warps every rendered layer with the FAFB
                # tilt correction (rigid rotation) — undo it so the
                # trace coordinates line up with the raw loaded bboxes.
                try:
                    undo = np.linalg.inv(
                        viz._get_fafb_tilt_correction_matrix())
                except Exception:  # noqa: BLE001
                    undo = None
                problems = check_scene_identities(viz, scene_bboxes,
                                                  undo_matrix=undo)
                if problems:
                    for item, bid, why in problems:
                        validator.log(f'    ! self-check [{src_type}]: '
                                      f'leaf {item} (bodyId {bid}): {why}')
                else:
                    validator.log(f'    self-check [{src_type}]: all '
                                  'legend leaves match their neuron '
                                  'geometry')
        except Exception as exc:  # noqa: BLE001
            import traceback
            validator.log(f'    ! scene {src_type} failed: {exc}')
            validator.log(traceback.format_exc())
