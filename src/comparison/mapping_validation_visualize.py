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
                                             the leaf tag rides with the
                                             post-hoc ``mapper_cell``, the
                                             row's own tier, the morph verdict
                                             where the gate scored it, and a
                                             trailing ``(shared)`` when that
                                             verdict belongs to another pair —
                                             see ``pool_leaf_tag``)
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
  olive, examinees red).  Those are DEFAULTS: ``cfg.scene_category_colors``
  merges over them per category (``resolve_scene_colors``), so a recolor
  still moves the whole bin everywhere at once and never one layer.
- ``legend_mode='tree'``: the drocat legend panel builds branch ->
  category -> bodyId from the ``drocatLegend`` meta tags, so the row-to-
  trace mapping can never drift from the figure.  Targets whose skeleton
  cannot be fetched are logged (``! {root}: {bid} unavailable``) and
  remain in the CSVs; unavailable neurons never silently vanish.
- ONE skeleton loader for every lane and dataset kind
  (``_load_scene_skeletons``): FAFB/BANC via
  ``load_local_release_skeletons`` with the extrusion check ON (CAVE
  centerlines — never the prepared mesh), NeuPrint via the dataset's own
  raw-skeleton cache + on-demand fetch.  A centerline guard refuses any
  non-TreeNeuron as a dropped id, so ``skeleton_mode='line'`` always
  means line geometry (the 2026-09-29 reverse MCNS→FAFB run served
  mesh-only FAFB targets and lost its whole query layer through the old
  per-side loaders; record:
  ``_plan/plan-tmvev-reverse-scene-loader-defects.md``).
- A scene whose QUERY layer is empty while the branch scanned sources is
  a FAILURE: ``EmptyQueryLayerError`` + ``SCENE_FAILED.txt`` in a
  synthetic ``plot-3d_branches_*`` folder the report's Scenes tab reads,
  and nothing rendered.  Partial query drops still render and are named.
- Revision 3.5 Issue 6: loaded neurons are renamed to ``str(bodyId)`` so
  navis trace names match the backend trace-identity resolver exactly
  (no positional fallback slips); a neuron renders in exactly ONE
  expansion bucket per branch; an optional debug self-check
  (``cfg.scene_selfcheck``) verifies each legend leaf's geometry against
  its neuron's loaded bbox AND — ``check_scene_population`` — that every
  expected id per layer was planted and rendered, with a geometry census
  flagging mesh-only leaves under ``skeleton_mode='line'``.
"""

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from comparison.body_id_resolver import chain_key
from comparison.cross_dataset_type_mapper import basis_is_row_evidence
from comparison.mapping_validation import DEDUP_RANK, TIER_CATEGORIES
from utils.color_utils import standardize_color
from visualization_options import default_analysis_skeleton_mesh_simplification

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
    'unassigned': '#7f7f7f',   # grey    - queried source with no verdict pair
}
# Kept as a name because the scene code and the docs both refer to it; it is
# the same entry the per-category color editor paints, so an override reaches
# the unassigned query neurons too.
UNASSIGNED_COLOR = CATEGORY_COLORS['unassigned']

# Category keys the UI may recolor, and the legacy aliases that must follow
# their canonical key so a recolor never leaves a bin behind wearing the old
# color. `out-map query` is the same source population as `query`, so it
# follows it and is not offered separately.
# `fill` / `backward` / `relatives` are NOT offered: the pre-3.12 bucket
# builder that could produce them is deleted, and the live partition
# (`CATEGORY_VALUES`) contains none of the three. They stay in
# CATEGORY_COLORS as a three-line hedge so a dict still carrying the old root
# names resolves to its historical color instead of falling back to examinees
# red. `backward` and `alternate-chain` remain live as the `invader_class`
# DATA value — only their scene color keys are unreachable.
COLOR_EDITABLE_CATEGORIES = (
    'query', 'unassigned', 'matched', 'verified', 'borderline', 'unmatched',
    'sibling', 'candidates', 'family', 'relative', 'examinees',
    'source-candidates', 'out-map candidates', 'pooling',
)
COLOR_ALIASES = {'candidates': ('fill',), 'relative': ('relatives',),
                 'query': ('out-map query',)}


def resolve_scene_colors(overrides: Optional[Dict[str, str]],
                         log=None) -> Dict[str, str]:
    """Merge a caller's category -> color map over :data:`CATEGORY_COLORS`.

    Validation happens here, at the boundary, using the renderer's own color
    parser: a value it refuses keeps the pipeline default and logs one line.
    Stage 4 already swallows a raised exception, so an unchecked bad color
    would silently cost the run EVERY scene rather than one category.

    A recolor propagates to :data:`COLOR_ALIASES` so a legacy key can never
    render a stale color while its canonical bin was edited.
    """
    out = dict(CATEGORY_COLORS)
    for cat, raw in sorted((overrides or {}).items()):
        cat = str(cat)
        value = str(raw or '').strip()
        if not value:
            continue
        try:
            standardize_color(value)
        except (TypeError, ValueError) as exc:  # noqa: BLE001
            if log:
                log(f'    ! scene color {cat!r} rejected ({exc}); keeping '
                    f'{out.get(cat, "the pipeline default")}')
            continue
        if cat not in CATEGORY_COLORS:
            # A typo is otherwise silent: the value parses, merges into the map,
            # and is never looked up, so the bin keeps its default color and
            # nothing says the edit was wasted.  Rejected unconditionally, not
            # only when a log is attached — the provenance record calls this
            # without one, and an accepted typo would then appear in
            # parameters.json as a color the run never wore.
            if log:
                log(f'    ! scene color {cat!r} is not a scene category; '
                    'ignored (see CATEGORY_COLORS)')
            continue
        out[cat] = value
        for alias in COLOR_ALIASES.get(cat, ()):
            out[alias] = value
    return out


#: Stage-4 kwargs the scene's correctness depends on, so a caller cannot
#: rewrite them: ``legend_mode='tree'`` is what carries the branch -> category
#: -> bodyId hierarchy (and the ``_drocat_*`` override dicts are read by that
#: panel); ``brain_mesh='native'`` IS the coordinate frame every neuron was
#: delivered in (the 2026-09-18 MCNS->BANC misplacement was a template switch);
#: ``skip_synapse`` is forced because pair scenes carry no connectivity and the
#: per-layer synapse color tuples are sized before the custom layers are
#: injected; and the layer/color/output identity belongs to the planner.
SCENE_PINNED_KWARGS = ('legend_mode', 'brain_mesh', 'skip_synapse',
                       'custom_neurons', 'custom_layer_names', 'neuron_colors',
                       'neuron_layers', 'dataset', 'output_dir', 'saveas',
                       'folder_prefix', 'include_timestamp')
#: Keys a caller may hold a widget for but which are not scene-render kwargs:
#: the tab's own Max scenes owns the scene count, ``neuron_alpha`` is a
#: first-class config field (one owner, not two), and the rest are panel
#: bookkeeping or features these scenes do not have.
SCENE_DROPPED_KEYS = ('visualize_top_n', 'visualize_by', 'neuron_alpha',
                      'output_format', 'show_soma', 'mesh_roi', 'roi_colors',
                      'synapse_colors', 'use_default_simplification')
#: The renderer's own default tube pipeline (``VisualizeSkeleton.
#: neuprint_skeleton_pipeline``), which is what decided the simplification
#: default before any of these knobs existed. Kept as a name so the fallback in
#: :func:`scene_render_kwargs` cannot silently drift to morphology's 'fine'.
_DEFAULT_SCENE_PIPELINE = 'fast'


def scene_render_kwargs(cfg, log=None) -> Dict[str, Any]:
    """Resolve ``cfg.scene_viz`` into kwargs a stage-4 scene will accept.

    Mirrors how the morphology analysis renders merge the same panel
    (src/morphology.py, the ``viz_kwargs`` loop): iterate, skip what the
    pipeline owns, then let the caller's values stand.
    """
    out: Dict[str, Any] = {}
    for key, value in sorted((cfg.scene_viz or {}).items()):
        if key in SCENE_PINNED_KWARGS or key in SCENE_DROPPED_KEYS:
            continue
        out[str(key)] = value
    # The shared panel's "use the method default" arrives as None. It must be
    # resolved HERE, not at the renderer: custom (injected) layers bypass the
    # fetch-time default path entirely, and a None that reaches
    # _effective_render_simplification raises — which stage 4 swallows by
    # losing every scene in the run.  The fallback pipeline is the RENDERER's
    # own default ('fast' -> 0.90), not morphology's 'fine': before this panel
    # existed a scene passed neither key, so 'fast' is what was actually in
    # force, and an untouched run must keep wearing it.
    pipeline = str(out.get('neuprint_skeleton_pipeline')
                   or _DEFAULT_SCENE_PIPELINE)
    if ('skeleton_mesh_simplification' in out
            and out['skeleton_mesh_simplification'] is None):
        out['skeleton_mesh_simplification'] = (
            default_analysis_skeleton_mesh_simplification(
                cfg.source_dataset, pipeline))
    elif ('skeleton_mesh_simplification' not in out
            and str(out.get('skeleton_mode') or '').strip().lower() == 'tube'):
        # A caller that asks for tube WITHOUT a fraction (the CLI example in
        # scripts/RunMappingValidation.py does exactly that) would otherwise
        # inherit the renderer's own ``None`` default and raise — the same
        # failure reached by a different door. Line mode never builds a neuron
        # mesh, so it is left alone and stays byte-identical.
        out['skeleton_mesh_simplification'] = (
            default_analysis_skeleton_mesh_simplification(
                cfg.source_dataset, pipeline))
    if out and log:
        log(f'[stage 4] scene styling from the caller: '
            f'{", ".join(sorted(out))}')
    return out


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


def cap_bucket_per_type(rec: Dict, cap: int) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    """Scene-only render cap (user 2026-09-29): keep at most ``cap``
    members per target type, in the bucket's own deterministic leaf order
    ((sort_key, bodyId) — the order the leaves would have rendered in
    anyway), and prune the record's fields to the kept set.  Returns
    ``(kept_total, {type: (kept, total)})``; an empty stats dict means the
    bucket was at or under ``cap`` (or ``cap <= 0``) and is untouched.
    Rendering-only: the CSVs keep every row."""
    if cap <= 0 or len(rec['ids']) <= cap:
        return len(rec['ids']), {}
    by_type: Dict[str, List[int]] = {}
    for bid in rec['ids']:
        by_type.setdefault(str(rec['types'].get(bid) or '?'), []).append(bid)
    keep: set = set()
    stats: Dict[str, Tuple[int, int]] = {}
    for tpe, bids in sorted(by_type.items()):
        ordered = sorted(bids, key=lambda b: (rec['sort_key'].get(b, ''), b))
        keep.update(ordered[:cap])
        stats[tpe] = (min(len(ordered), cap), len(ordered))
    if len(keep) < len(rec['ids']):
        rec['ids'] = keep
        for field in ('types', 'leaf_token', 'tags', 'sort_key'):
            rec[field] = {b: v for b, v in rec[field].items() if b in keep}
    return len(keep), stats


def relative_cap_note(kept: int, total: int, cap: int) -> str:
    """The legend-visible disclosure a capped relative bucket carries on
    its root label."""
    return f'rendered {kept:,} of {total:,} · {cap}/type'


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


class EmptyQueryLayerError(RuntimeError):
    """A scene's queried source neurons all failed to load a skeleton.

    Raised into the scene-failure marker by ``render_pair_scenes``: a
    scene whose query layer is empty is a FAILURE, not a smaller picture
    (plan-tmvev-reverse-scene-loader-defects.md fix 2 — the 2026-09-29
    reverse run rendered 21 target-only scenes while all 204 queried
    male-cns sources dropped silently, and every gate stayed green).
    """


def _scene_folder_slug(name: str) -> str:
    """Folder-safe slug for a synthetic failed-scene folder name (the
    report's scene-failure reader only needs the ``plot-3d_`` prefix)."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'scene'


def _is_tree_neuron(nrn) -> bool:
    """True for objects carrying a centerline (navis TreeNeuron — or a
    test double shaped like one).  A prepared mesh has ``vertices`` but
    no ``nodes``; under ``skeleton_mode='line'`` it must never be plotted,
    so the scene loader refuses it as dropped."""
    return nrn is not None and hasattr(nrn, 'nodes')


def _raw_skeleton_cache_dir(cache_dir, dataset: str) -> Path:
    """The dataset's own raw-skeleton store:
    ``<cache>/<dataset_folder>/skeletons/raw_skeletons/``."""
    return (Path(cache_dir) /
            dataset.replace(':', '_').replace('.', '_') /
            'skeletons' / 'raw_skeletons')


def _load_scene_skeletons(dataset: str, bids, cache_dir=None, log=None
                          ) -> Tuple[List, List[Tuple[int, str]]]:
    """Scene skeleton loader for EVERY dataset kind — the one loader both
    the query and the target layers go through.

    Three measured defects this closes (all on the 2026-09-29 MCNS→FAFB
    run, ``plan-tmvev-reverse-scene-loader-defects.md``):

    - the query layer went through ``_resolve_fafb_skeleton_trees``,
      which returns ``{}`` for any non-FAFB dataset — all 204 queried
      male-cns sources silently vanished from the reverse scenes;
    - the target layer probed ``raw_skeletons/`` (0 files for FAFB — its
      centerlines live in ``cave_skeletons/``) and fell back to
      ``fetch_skeleton_on_demand``, whose FAFB branch returns a prepared
      MESH: no centerline, served under ``Skeleton Mode: line``;
    - neither path ever passed ``check_extrusions=True``, so drawn FAFB
      geometry was not extrusion-refined (the picture and the morph
      score were not guaranteed to come from the same skeleton).

    Dispatch: FAFB/BANC → ``morphology.load_local_release_skeletons``
    with ``check_extrusions=True`` (repair caches → raw cache → healed
    zip → extrusion check/repair → token-gated CAVE; BANC from its
    public-release bucket).  NeuPrint → the dataset's own
    ``raw_skeletons`` cache, then ``fetch_skeleton_on_demand`` (its
    NeuPrint branch returns TreeNeurons and persists there).  Anything
    that is not a centerline-bearing neuron is refused as dropped —
    never plotted.  ``_resolve_fafb_skeleton_trees``'s non-FAFB early
    return stays untouched: it is load-bearing for build-time
    vectorization, where ``check_extrusions=False`` is deliberate.

    Returns ``(neurons, dropped)``; neurons are stamped ``id`` / ``name``
    (bodyId — the backend's trace-identity resolver matches on name) and
    ``_drocat_source_dataset``; ``dropped`` is ``(bodyId, why)`` pairs.
    """
    from morphology import (_load_cached_skeleton_file,
                            fetch_skeleton_on_demand)
    from flywire_ids import is_local_connectome_dataset

    def _stamp(nrn, bid: int) -> None:
        try:
            nrn.id = int(bid)
            nrn.name = str(int(bid))
            nrn._drocat_source_dataset = dataset
        except Exception:  # noqa: BLE001
            pass

    ids = [int(b) for b in bids]
    neurons, dropped = [], []
    if is_local_connectome_dataset(dataset):
        try:
            from morphology import load_local_release_skeletons
            trees = load_local_release_skeletons(
                dataset, ids, log=log, check_extrusions=True)
        except Exception as exc:  # noqa: BLE001
            return [], [(b, str(exc)) for b in ids]
        for b in ids:
            nrn = trees.get(b)
            if nrn is None:
                dropped.append((b, 'no skeleton'))
            elif not _is_tree_neuron(nrn):
                dropped.append((b, 'non-skeleton object refused'))
            else:
                _stamp(nrn, b)
                neurons.append(nrn)
    else:
        skel_dir = _raw_skeleton_cache_dir(cache_dir, dataset)
        for b in ids:
            try:
                path = skel_dir / f'{b}.swc.zst'
                nrn = (_load_cached_skeleton_file(path)
                       if path.exists() else None)
                if nrn is None:
                    nrn = fetch_skeleton_on_demand(dataset, b)
            except Exception as exc:  # noqa: BLE001
                dropped.append((b, str(exc)))
                continue
            if nrn is None:
                dropped.append((b, 'no skeleton'))
            elif not _is_tree_neuron(nrn):
                dropped.append((b, 'non-skeleton object refused'))
            else:
                _stamp(nrn, b)
                neurons.append(nrn)
    return neurons, dropped


def _neuron_leaf_census(viz) -> Dict[int, Dict]:
    """Per-bodyId census of the rendered neuron-kind legend leaves: how
    many traces carry the leaf and whether any of them is a centerline
    (``mode='lines'``).  Hidden (default-off) layers keep their traces in
    the payload (``visible=False``), so they are censused like the rest."""
    census: Dict[int, Dict] = {}
    for tr in getattr(getattr(viz, 'fig_3d', None), 'data', []) or []:
        meta = getattr(tr, 'meta', None) or {}
        lg = meta.get('drocatLegend')
        if not isinstance(lg, dict) or lg.get('kind') != 'neuron':
            continue
        m = re.match(r'^(\d+)', str(lg.get('item') or ''))
        if not m:
            continue
        rec = census.setdefault(int(m.group(1)),
                                {'item': lg.get('item'), 'traces': 0,
                                 'lines': False})
        rec['traces'] += 1
        if str(getattr(tr, 'mode', '')) == 'lines':
            rec['lines'] = True
    return census


def check_scene_population(viz, expected_by_layer: Dict[str, set],
                           planted_by_layer: Dict[str, set],
                           skeleton_mode=None) -> List[str]:
    """Population self-check: every EXPECTED id must be planted, and every
    planted id must be rendered.

    ``check_scene_identities`` compares plotted geometry against legend
    leaves, both derived from the same load pass — a layer that never
    loaded produces neither traces nor bboxes and is invisible to it.
    That is the gate blind spot that let the 2026-09-29 reverse run ship
    21 target-only scenes ("40/40 self-checks passed") with all 204
    queried sources missing.  This check compares:

    - per layer, expected ids (captured at the loader call sites) against
      planted ids (what ``add_layer`` actually received);
    - planted ids against the rendered neuron leaves in the figure
      payload — a planted neuron the backend dropped is still missing;
    - a geometry census: under ``skeleton_mode='line'`` a neuron leaf
      whose payload holds no ``lines`` trace is mesh-only, i.e. the
      declared mode and the served geometry disagree.

    Returns problem strings; empty means the population is complete.
    """
    problems: List[str] = []

    def _fmt(bids: List[int]) -> str:
        shown = ', '.join(str(b) for b in bids[:8])
        if len(bids) > 8:
            shown += f' … (+{len(bids) - 8} more)'
        return shown

    for layer in sorted(set(expected_by_layer) | set(planted_by_layer)):
        exp = expected_by_layer.get(layer) or set()
        got = planted_by_layer.get(layer) or set()
        missing = sorted(exp - got)
        if missing:
            problems.append(f'layer {layer}: {len(got)}/{len(exp)} '
                            f'plotted, missing: {_fmt(missing)}')
        extra = sorted(got - exp)
        if extra:
            problems.append(f'layer {layer}: {len(extra)} plotted id(s) '
                            f'not in the expected set: {_fmt(extra)}')
    census = _neuron_leaf_census(viz)
    planted_all: set = set()
    for ids in planted_by_layer.values():
        planted_all |= ids
    ghost = sorted(planted_all - set(census))
    if ghost:
        problems.append(f'{len(ghost)} planted neuron(s) rendered no '
                        f'trace: {_fmt(ghost)}')
    if str(skeleton_mode or '') == 'line':
        mesh_only = [rec for rec in census.values()
                     if rec['traces'] and not rec['lines']]
        if mesh_only:
            names = ', '.join(str(rec['item']) for rec in mesh_only[:6])
            if len(mesh_only) > 6:
                names += f' … (+{len(mesh_only) - 6} more)'
            problems.append(
                f'{len(mesh_only)} neuron leaf/leaves carry no centerline '
                f'trace under skeleton_mode=line (mesh-only): {names}')
    return problems


def _planted_query_ids(planted_by_layer: Dict[str, set]) -> set:
    """Ids planted in the query/unassigned layers — the keys ending in
    ``:: query`` / ``:: unassigned``.  ``:: out-map query`` is a different
    population (the unclaimed-source gap) and does not match."""
    planted: set = set()
    for key, pids in planted_by_layer.items():
        if key.endswith(':: query') or key.endswith(':: unassigned'):
            planted |= pids
    return planted


def pool_rows_by_host(pool_rows) -> Dict[str, List[Dict]]:
    """The ``pooling`` pool grouped by the scene that may host it: the TYPE
    of each candidate target's best source.  A pool row belongs to no branch,
    so its source's type is its only scene address — one row per target means
    one host, and a type no branch group covers renders no layer (the render
    pass names those rows rather than dropping them quietly).

    Rows the morphology bar refused on every admitting row (`in_pool=False`) are
    not in the pool and are not drawn, and the count is reported so a shrunken
    scene cannot read as a smaller harvest; so are the mapper-only rows, which
    answer a different question (what the SUPERVISED path claims) and live in
    `pooling_pool.csv` alone.
    """
    hosts: Dict[str, List[Dict]] = defaultdict(list)
    for r in pool_rows or []:
        if r.get('in_pool') is False:
            continue
        hosts[str(r.get('best_source_type') or '')].append(r)
    return hosts


def pool_leaf_tag(row: Dict) -> str:
    """The pooling leaf's tag: the post-hoc mapper cell, the tiers that reached
    this target, the morphology verdict where the gate scored it, and
    `(shared)` LAST.

    The tiers belong here because the scene is the only place a candidate is
    seen without a column header: a leaf some source called `matched` asserts a
    homolog and one that is only ever `nominated` is review material, and the
    picture must not let them look alike. A target is reached by several rows,
    so `pooling_pool.csv` carries the set (`matched+nominated`), and that is
    what the leaf shows.

    Measured on the 2026-09-24 male-cns run: all 222 drawn leaves carry a tier
    and a verdict, and NONE of them reads `(shared)` — `pool_by_target` draws
    each target's chain-best row, which is by construction a row the gate
    scored for that pair. The 424 `shared` rows that belong to drawn targets are
    its NON-best rows, which live in `pooling_candidates.csv` and never reach
    the scene, so a borrowed number is visible in the CSV and not in the
    picture. The branch stays because the tag's contract is any pooling row,
    and a leaf that DID borrow would otherwise print the number as its own.
    """
    bits = [str(row.get('mapper_cell') or '')]
    tiers = str(row.get('tiers') or '')
    if tiers:
        bits.append(tiers)
    gate = str(row.get('morph_gate') or '')
    if gate in ('scored', 'shared'):
        mark = 'morph ✓' if row.get('morph_qualified') else 'morph ✗'
        # attached to the verdict with a space, not joined as its own bit: the
        # marker qualifies THAT number, and `morph ✗ · (shared)` would read as
        # a second, unrelated tag on the neuron.
        bits.append(f'{mark} (shared)' if gate == 'shared' else mark)
    return ' · '.join(b for b in bits if b)


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


def _write_scene_failure_marker(folder, src_type, exc, tb) -> bool:
    """Say so IN the folder when a scene dies part-way through.

    ``VisualizeSkeleton`` creates its save folder before the figure is written,
    so a failed render leaves a directory holding ``parameters.txt`` and the
    layer CSVs but no HTML and no manifest — on disk that is
    indistinguishable from a scene that succeeded. Measured on the 2026-09-24
    male-cns family run: 5 of its 21 parent scenes were exactly such empty
    folders, and nothing downstream noticed.

    Also used by the empty-query-layer gate for scenes that never reach the
    constructor at all: there ``folder`` is a synthetic
    ``plot-3d_branches_*`` directory created here, holding nothing but the
    marker — the report's scene-failure reader only needs the folder name
    to start with ``plot-3d_`` and the marker to carry ``parent type:`` /
    ``error:`` lines.

    Returns whether a marker was written. An unwritable folder is not worth
    raising over: the run log already carries the error and its traceback.
    """
    if not folder:
        return False
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / 'SCENE_FAILED.txt').write_text(
            'this scene did not render\n'
            f'parent type: {src_type}\n'
            f'error: {type(exc).__name__}: {exc}\n'
            'No branches_*.html, no PNG and no visualization_manifest.json '
            'was written; everything else in this folder is a partial '
            'artifact of the attempt.\n\n'
            f'{tb}\n', encoding='utf-8')
    except OSError:
        return False
    return True


def render_pair_scenes(validator, per_pair_res: Dict) -> None:
    """Render one branch-structured scene per parent mapping group.

    Every layer loads its skeletons through ``_load_scene_skeletons``
    (dataset-general, TreeNeuron-guaranteed, extrusion-checked on
    FAFB/BANC).  A parent whose queried sources all fail to load is a
    failure (``SCENE_FAILED.txt``, nothing rendered); with
    ``cfg.scene_selfcheck`` on, each rendered scene also runs the
    population check (expected-vs-planted-vs-rendered per layer, geometry
    census) beside the leaf-identity check.
    """
    import navis

    cfg = validator.cfg
    viz_dir = validator.run_dir / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)
    from visualize_skeleton import (VisualizeSkeleton,
                                    dataset_native_space,
                                    dataset_render_space,
                                    transform_neurons_to_space)

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

    # Scene look. Colors are validated at the boundary (resolve_scene_colors:
    # one rejected value costs one category, not the run); the kwargs are
    # already filtered against what the pipeline owns.
    colors = resolve_scene_colors(cfg.scene_category_colors, validator.log)
    scene_kwargs = scene_render_kwargs(cfg, validator.log)

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

    # ONE loader for both layers (plan-tmvev-reverse-scene-loader-defects
    # fix 1): the query path's `_resolve_fafb_skeleton_trees` returned {}
    # for any non-FAFB source (all 204 male-cns queries vanished from the
    # 2026-09-29 reverse scenes), and the target path's
    # `fetch_skeleton_on_demand` fallback serves prepared MESHES for FAFB
    # — no centerline under `Skeleton Mode: line`, and never
    # extrusion-checked.  `_load_scene_skeletons` routes every dataset
    # kind to a TreeNeuron guarantee instead.
    def load_target_neurons(bids: List[int]):
        return _load_scene_skeletons(
            cfg.target_dataset, bids,
            cache_dir=validator.profiler.cache_dir, log=validator.log)

    def load_query_neurons(bids: List[int]):
        neurons, dropped = _load_scene_skeletons(
            cfg.source_dataset, bids,
            cache_dir=validator.profiler.cache_dir, log=validator.log)
        # The scene renders in the source RENDER space; raw cached
        # skeletons are in the dataset's native frame.
        neurons = bridge_to_scene_space(neurons)
        return neurons, dropped

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
            # Population self-check bookkeeping (fix 3): per layer, the ids
            # the loaders were ASKED for (expected, keyed at each call
            # site) and the ids `add_layer` actually received (planted).
            # A layer that never loaded has no traces and no bboxes, so
            # `check_scene_identities` alone cannot see it.
            expected_by_layer: Dict[str, set] = {}
            planted_by_layer: Dict[str, set] = {}
            # The branch's queried sources across ALL its branches — the
            # population an empty-query-layer failure is measured against.
            scene_query_expected: set = set()

            def note_expected(group, category, bids):
                rec = expected_by_layer.setdefault(
                    f'{group} :: {category}', set())
                rec.update(int(b) for b in bids)

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
                # Population self-check: what this layer actually received.
                layer_planted = planted_by_layer.setdefault(layer_name, set())
                for n in neurons:
                    try:
                        layer_planted.add(int(n.id))
                    except Exception:  # noqa: BLE001
                        pass

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
                scene_query_expected.update(live_src)
                scene_query_expected.update(skipped_src)
                note_expected(group, 'query', live_src)
                note_expected(group, 'unassigned', skipped_src)
                if live_src:
                    qn, qdrop = load_query_neurons(live_src)
                    for bid, why in qdrop:
                        validator.log(f'    ! query {bid} skipped ({why})')
                    if qn:
                        add_layer(group, qn, colors['query'],
                                  [f'query · {src_type}'] * len(qn),
                                  category='query')
                if skipped_src:
                    sn, sdrop = load_query_neurons(skipped_src)
                    for bid, why in sdrop:
                        validator.log(f'    ! query {bid} skipped '
                                      f'(unassigned lane: {why})')
                    if sn:
                        add_layer(group, sn, colors['unassigned'],
                                  [f'query · {src_type} · unassigned']
                                  * len(sn), category='unassigned')

                # pool neurons by Revision 3.3 ladder category
                categories: Dict[str, List[int]] = defaultdict(list)
                for tbid, cat in res['target_categories'].items():
                    categories[cat].append(int(tbid))
                cat_colors = {c: colors.get(
                    c, colors['unmatched'])
                    for c in ('matched', 'verified', 'borderline',
                              'unmatched')}
                for cat in ('matched', 'verified', 'borderline',
                            'unmatched'):
                    ids = sorted(set(categories.get(cat, [])))
                    if not ids:
                        continue
                    root = f'{cat} · {pair.target_type}'
                    note_expected(group, cat, ids)
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
                    note_expected(group, 'source-candidates', cids)
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
                                  colors['source-candidates'],
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
                                  f" {suffix}".strip()
                            rec['tags'][bid] = tag
                            rec['sort_key'][bid] = (
                                f"{rec['leaf_token'].get(bid, '')} "
                                f"{tag}").strip()

                # Scene-only render cap (user 2026-09-29): a wide mode can
                # enumerate thousands of relatives for one branch (BANC
                # family s-CPDN3B->CB3252: 2,730 rows, KCg-m 1,455) and
                # every one became a 3D leaf + legend entry — the
                # 0.4-0.8 GB/run scenes.  Render at most
                # `scene_relative_cap_per_type` leaves per target type, in
                # the bucket's own leaf order; relatives.csv keeps every
                # row and the drop is disclosed on the root label + here.
                cap = max(0, int(getattr(
                    cfg, 'scene_relative_cap_per_type', 50) or 0))
                for key in bucket_order:
                    if _cat_of_key(key) != 'relative':
                        continue
                    rec = buckets[key]
                    total = len(rec['ids'])
                    kept, stats = cap_bucket_per_type(rec, cap)
                    if kept < total:
                        rec['cap_note'] = relative_cap_note(
                            kept, total, cap)
                        over = [f'{t} {k}/{n}'
                                for t, (k, n) in sorted(
                                    stats.items(),
                                    key=lambda kv: -(kv[1][1] - kv[1][0]))
                                if k < n]
                        top = ', '.join(over[:4])
                        validator.log(
                            f'    · relative leaves capped at {cap}/type: '
                            f'rendered {kept:,} of {total:,}'
                            + (f' ({top})' if top else ''))

                def add_invader_layers():
                    for key in sorted(bucket_order, key=lambda k: (
                            CATEGORY_RENDER_PRIORITY.get(
                                _cat_of_key(k), 50), k)):
                        rec = buckets[key]
                        ids = sorted(rec['ids'])
                        if not ids:
                            continue
                        prefix = _cat_of_key(key)
                        color = colors.get(
                            prefix, colors['examinees'])
                        root = bucket_root_label(key, rec['types'])
                        if rec.get('cap_note'):
                            root = f"{root} · {rec['cap_note']}"
                        # `rec['ids']` is already the post-cap render set
                        # (cap_bucket_per_type pruned it) — that is the
                        # population this layer PROMISES on its root label.
                        note_expected(group, root, ids)
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
                note_expected(f'{src_type} · out-map query',
                              'out-map query', out_map)
                on, on_drop = load_query_neurons(out_map)
                for bid, why in on_drop:
                    validator.log(f'    ! out-map query {bid} '
                                  f'unavailable ({why})')
                if on:
                    label = f'out-map query · {src_type}'
                    for n in on:
                        type_overrides[int(n.id)] = label
                    add_layer(f'{src_type} · out-map query', on,
                              colors['out-map query'],
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
                    note_expected(f'{src_type} · out-map candidates',
                                  'out-map candidates', cand_ids)
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
                                      colors['out-map candidates'],
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
                note_expected(f'{src_type} · pooling',
                              f'pooling · {src_type}', sorted(by_bid))
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
                        tag = pool_leaf_tag(r)
                        p_types[int(n.id)] = token
                        if tag:
                            p_tags[int(n.id)] = tag
                        p_sorts[int(n.id)] = f'{token} {tag}'.strip()
                    add_layer(f'{src_type} · pooling', neurons,
                              colors['pooling'],
                              [root] * len(neurons), category=root,
                              leaf_types=p_types, leaf_tags=p_tags or None,
                              leaf_sorts=p_sorts)

            # A scene whose QUERY layer is empty is a failure, not a
            # smaller picture (fix 2).  The 2026-09-29 reverse run
            # rendered 21 target-only scenes while all 204 queried
            # sources dropped silently — on disk indistinguishable from a
            # correct scene.  Partial query drops still render (the
            # population self-check below names them); a fully empty
            # query layer writes the SCENE_FAILED marker the report's
            # Scenes tab already reads, and nothing is rendered.
            planted_query = _planted_query_ids(planted_by_layer)
            if scene_query_expected and not planted_query:
                why = (f'{len(scene_query_expected)} queried source '
                       'neuron(s) resolved to no skeleton — the query '
                       'layer is empty')
                validator.log(f'    ! scene {src_type} failed: {why}')
                folder = Path(viz_dir) / (
                    'plot-3d_branches_' + _scene_folder_slug(src_type)
                    + '_' + time.strftime('%Y%m%d_%H%M%S'))
                _write_scene_failure_marker(folder, src_type,
                                            EmptyQueryLayerError(why), '')
                continue

            if not entries:
                validator.log('    nothing to render for this parent')
                continue

            # Built as a dict, not as `**scene_kwargs` after the explicit
            # keywords: a caller-supplied `skeleton_mode` would then be passed
            # twice and raise "got multiple values for keyword argument"
            # (caught by the 2026-09-26 real-data scene run).
            viz_kwargs = dict(
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
            # The caller's panel values, last: they are already filtered
            # against SCENE_PINNED_KWARGS / SCENE_DROPPED_KEYS, so what lands
            # here is only what a scene may legitimately be talked into
            # (line/tube, background, export, simplification).
            viz_kwargs.update(scene_kwargs)
            viz = VisualizeSkeleton(**viz_kwargs)
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
                # Population self-check (fix 3): the leaf-vs-geometry
                # check above is blind to an ABSENT population — expected
                # vs planted per layer, planted vs rendered traces, and a
                # geometry census under skeleton_mode=line (fix 4's
                # detector: a declared `line` scene serving mesh-only
                # neurons is a documentation lie).
                pop_problems = check_scene_population(
                    viz, expected_by_layer, planted_by_layer,
                    skeleton_mode=viz_kwargs.get('skeleton_mode'))
                if pop_problems:
                    for why in pop_problems:
                        validator.log(f'    ! self-check [{src_type}]: '
                                      f'{why}')
                else:
                    validator.log(
                        f'    self-check [{src_type}]: population '
                        f'complete — {len(planted_by_layer)} layer(s), '
                        f'{sum(len(v) for v in planted_by_layer.values())}'
                        f' neuron(s) plotted, skeleton_mode='
                        f"{viz_kwargs.get('skeleton_mode')}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            validator.log(f'    ! scene {src_type} failed: {exc}')
            tb = traceback.format_exc()
            validator.log(tb)
            # `viz` is bound inside the guarded block, so a constructor failure
            # leaves it unbound — and no folder to mark, since the constructor
            # is what makes it.
            failed_viz = locals().get('viz')
            _write_scene_failure_marker(getattr(failed_viz, 'save_folder', None),
                                        src_type, exc, tb)
