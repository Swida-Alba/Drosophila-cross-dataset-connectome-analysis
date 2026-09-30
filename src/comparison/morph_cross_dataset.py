"""Cross-dataset morphology: shared scoring helper + UI comparison backend.

Implements two features on top of the production cross-dataset scorer
(``morphology.compute_morph_similarity_vs_queries``, vector_v2 only —
NBLAST is never used cross-dataset):

- **Morph qualification** for Find Homolog
  (``_plan/plan-morph-qualification-find-homolog.md``): after the
  connectivity ranking, the top-N *visualized* candidates are scored
  against the transformed query and gated by a per-query null bar
  (p95 of seeded random target neurons). Failures are excluded from the
  3D scene; the results tables keep every row.
- **Cross-Dataset morphology comparison**
  (``_plan/plan-ui-cross-dataset-morph-comparison.md``): pure comparison
  of the *queried* neurons across FAFB / male-cns / BANC — pairwise
  vector_v2 matrices per dataset pair plus a null baseline reference.

Dataset scope (plan §2): only FAFB, male-cns v1.0 and BANC participate.
Every allowed pair is ≤2 bridging transforms; anything else is refused
with an explicit reason (the repo's ≤2-hop accuracy guard). BANC carries
a standing reliability warning (mixed skeleton sources) and needs a
one-time morphology-cache bootstrap (``ensure_population_artifacts``)
before it can score with the production whitened scorer.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NULL_K_DEFAULT = 200
# Floors v3 (plan-unified-morph-qualification-bars): mapping_ref-mode Track-A
# offset Δ and native floor margin.  The null-mode offset stays the user's
# `bar_offset` slider; these govern the mapping-referenced rungs.
MAPPING_REF_TRACK_A_OFFSET = 0.05
MAPPING_REF_NATIVE_MARGIN = 0.05
NULL_MIN_N = 10
BAR_OFFSET_MAX = 0.2
MAX_TRANSFORM_HOPS = 2
BOOTSTRAP_SAMPLE_DEFAULT = 600
BOOTSTRAP_MIN_ROWS = 24

try:  # package import (runner scripts / tests)
    from .cross_dataset_type_mapper import get_type_mapper
    from .type_resolver import (MapperSnapshot, expansion_targets,
                                resolve_valid_targets)
    from . import report_kit
except ImportError:  # direct src/ execution
    try:
        from cross_dataset_type_mapper import get_type_mapper
        from type_resolver import (MapperSnapshot, expansion_targets,
                                   resolve_valid_targets)
        import report_kit
    except ImportError:  # mapper optional: feature degrades to same-name
        get_type_mapper = None
        MapperSnapshot = None
        resolve_valid_targets = None
        expansion_targets = None

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# The one morphology metric rendered by the comparison report. vector_v2 is
# a whitened cosine that can go negative, so it uses the shared kit's
# diverging scale (blue = negative, white = 0, red = positive) instead of
# clipping negatives like the old green [0, 1] card.
_MORPH_V2_STYLE = report_kit.MetricStyle(
    'morph_v2', 'Vector v2', report_kit.REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)
_LEVEL_DISPLAY = {'type': 'Type level', 'bodyid': 'BodyId level'}


def _body_id_label(dataset: str, bid, fallback_type: str) -> str:
    """Tree-legend display label ('{bodyId}_{instance}' / '{bodyId}_{type}_{L|R}').

    Uses the connectivity-profiler's dataset-backed label helper; degrades
    to '{bodyId}_{type}' when that import or its label maps are unavailable.
    """
    try:
        from .profile_comparator import _body_id_display_label
    except ImportError:  # pragma: no cover - direct src/ execution
        try:
            from profile_comparator import _body_id_display_label
        except ImportError:
            return f'{bid}_{fallback_type}'
    return _body_id_display_label(dataset, bid, fallback_type=fallback_type)


def _logline(log, message: str) -> None:
    if log is not None:
        log(message)
    else:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Dataset scope and pair availability
# ---------------------------------------------------------------------------

def _dataset_family(dataset: str) -> str:
    """FAFB | MCNS | BANC | '' (unsupported) — mirrors flywire_ids predicates."""
    d = str(dataset or '').strip().lower()
    if not d:
        return ''
    if 'banc' in d:
        return 'BANC'
    if 'fafb' in d or ('flywire' in d and 'banc' not in d):
        return 'FAFB'
    if 'male-cns' in d or 'malecns' in d or 'optic' in d:
        return 'MCNS'
    return ''


def dataset_spaces(dataset: str) -> Tuple[str, str]:
    """(native_space, render_space) via the canonical visualize_skeleton map."""
    from visualize_skeleton import dataset_native_space, dataset_render_space
    return dataset_native_space(dataset), dataset_render_space(dataset)


def dataset_scope(dataset: str) -> Dict[str, Any]:
    """Validate one dataset for cross-dataset morphology.

    Returns ``{ok, family, native_space, render_space, warnings, reason}``.
    """
    family = _dataset_family(dataset)
    scope = {'ok': False, 'family': family, 'native_space': '',
             'render_space': '', 'warnings': [], 'reason': ''}
    if not family:
        scope['reason'] = (
            f"Dataset '{dataset}' is not supported for cross-dataset "
            "morphology. Only FAFB, male-cns and BANC participate (no "
            "bridging transform within 2 hops reaches other datasets).")
        return scope
    try:
        native, render = dataset_spaces(dataset)
    except Exception as exc:  # noqa: BLE001
        scope['reason'] = f"No known template space for '{dataset}': {exc}"
        return scope
    scope['native_space'] = native
    scope['render_space'] = render
    d = str(dataset or '').strip().lower()
    if family == 'MCNS' and 'male-cns' not in d:
        # optic-lobe and other male-CNS-space datasets have no vector_v2
        # population artifacts and no cached skeleton population to
        # bootstrap them from — refuse instead of silently degrading.
        scope['ok'] = False
        scope['reason'] = (
            f"Dataset '{dataset}' has no vector_v2 morphology population "
            "artifacts and cannot be bootstrapped offline; cross-dataset "
            "morphology is limited to male-cns:v1.0 within the male-CNS "
            "family.")
        return scope
    if family == 'MCNS' and 'v0.9' in d:
        scope['ok'] = False
        scope['reason'] = (
            "male-cns:v0.9 lacks vector_v2 population artifacts; use "
            "male-cns:v1.0 for cross-dataset morphology.")
        return scope
    if family == 'BANC':
        scope['warnings'].append(
            'BANC morphology is experimental: public skeleton products mix '
            'L2 / full / µm sources, so scores are less reliable than '
            'FAFB / male-cns comparisons.')
        if not population_artifacts_ready(dataset):
            scope['warnings'].append(
                'BANC morphology population artifacts are missing; they '
                'will be bootstrapped from the locally cached skeletons '
                'before scoring (one-time).')
    scope['ok'] = True
    return scope


def _bridging_chain(source_space: str, target_space: str
                    ) -> Tuple[int, str]:
    """(transform count, chain string) of the shortest bridging path."""
    try:
        import flybrains
        flybrains.register_transforms()
        from navis.transforms import registry as registry
        path, _seq = registry.shortest_bridging_seq(source_space, target_space)
    except Exception as exc:  # noqa: BLE001
        return -1, f'unavailable ({exc})'
    return max(len(path) - 1, 0), ' -> '.join(path)


def check_pair_availability(source_dataset: str, target_dataset: str
                            ) -> Dict[str, Any]:
    """Whether source->target scoring is allowed under the ≤2-hop guard.

    Returns ``{ok, hops, chain, reason}``.
    """
    out = {'ok': False, 'hops': -1, 'chain': '', 'reason': ''}
    if str(source_dataset) == str(target_dataset):
        out['ok'] = True
        out['hops'] = 0
        out['chain'] = 'identity'
        return out
    src = dataset_scope(source_dataset)
    tgt = dataset_scope(target_dataset)
    if not src['ok']:
        out['reason'] = src['reason']
        return out
    if not tgt['ok']:
        out['reason'] = tgt['reason']
        return out
    if src['native_space'] == tgt['render_space']:
        out['ok'] = True
        out['hops'] = 0
        out['chain'] = 'identity'
        return out
    hops, chain = _bridging_chain(src['native_space'], tgt['render_space'])
    out['hops'] = hops
    out['chain'] = chain
    if hops < 0:
        out['reason'] = (f'No bridging transform {src["native_space"]} -> '
                         f'{tgt["render_space"]}: {chain}')
    elif hops > MAX_TRANSFORM_HOPS:
        out['reason'] = (f'{source_dataset} -> {target_dataset} needs '
                         f'{hops} transforms ({chain}) — beyond the 2-hop '
                         'accuracy guard; comparison refused.')
    else:
        out['ok'] = True
    return out


def validate_cross_dataset_sets(datasets: Sequence[str]) -> List[Dict[str, Any]]:
    """Scope every selected dataset + every ordered pair; raise on refusal.

    Returns the per-dataset scope dicts (input order). Raises ``ValueError``
    with a user-facing message when any dataset or pair is not allowed.
    """
    names = [str(d) for d in (datasets or []) if str(d or '').strip()]
    if len(names) < 1:
        raise ValueError('Select at least one dataset.')
    if len(names) == 1:
        raise ValueError('Select at least two datasets for a cross-dataset '
                         'comparison.')
    scopes = {}
    for name in names:
        scope = dataset_scope(name)
        if not scope['ok']:
            raise ValueError(scope['reason'])
        scopes[name] = scope
    for a in names:
        for b in names:
            if a == b:
                continue
            check = check_pair_availability(a, b)
            if not check['ok']:
                raise ValueError(check['reason'])
    return [scopes[name] for name in names]


# ---------------------------------------------------------------------------
# Neuron universe, seeded null sampling
# ---------------------------------------------------------------------------

_UNIVERSE_CACHE: Dict[Tuple[str, str], List[int]] = {}
_SKELETON_ID_CACHE: Dict[Tuple[str, str], set] = {}


def _skeleton_cached_ids(dataset: str,
                         project_root: Optional[str] = None) -> set:
    """bodyIds whose raw skeletons are locally cached (memoized)."""
    key = (dataset, str(project_root or DEFAULT_PROJECT_ROOT))
    if key in _SKELETON_ID_CACHE:
        return _SKELETON_ID_CACHE[key]
    ids: set = set()
    try:
        from morphology import find_similar_dataset_cache_v2
        files = find_similar_dataset_cache_v2(
            dataset, project_root=key[1], verbose=False) \
            ._discover_skeleton_files()
        for path in files:
            m = re.match(r'(\d+)', Path(str(path)).name)
            if m:
                ids.add(int(m.group(1)))
    except Exception:  # noqa: BLE001
        pass
    _SKELETON_ID_CACHE[key] = ids
    return ids


def dataset_bodyids(dataset: str,
                    project_root: Optional[str] = None) -> List[int]:
    """The dataset's bodyId universe: vector-cache rows first, neuron index
    fallback (prefers ids with local material for cheap scoring)."""
    root = Path(project_root or DEFAULT_PROJECT_ROOT)
    key = (dataset, str(root))
    if key in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE[key]
    universe: List[int] = []
    try:
        from morphology import find_similar_dataset_cache_v2
        data = find_similar_dataset_cache_v2(
            dataset, project_root=str(root), verbose=False).load()
        if data is not None and data.get('bodyIds') is not None:
            bids = []
            for b in list(data['bodyIds']):
                try:
                    bids.append(int(b))
                except (TypeError, ValueError):
                    continue
            if bids:
                universe = sorted(set(bids))
    except Exception:  # noqa: BLE001
        pass
    if not universe:
        safe = str(dataset).replace(':', '_').replace('.', '_')
        for path in (root / 'neuron_indexes' / safe /
                     'neuron_index.parquet',
                     root / 'datasets' / safe /
                     f'{safe}_allneurons_neuron_df.parquet'):
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                if 'bodyId' not in df.columns:
                    continue
                bids = pd.to_numeric(df['bodyId'], errors='coerce') \
                    .dropna().astype('int64')
                universe = sorted(set(int(b) for b in bids))
                break
            except Exception:  # noqa: BLE001
                continue
    _UNIVERSE_CACHE[key] = universe
    return universe


def _null_seed(dataset: str) -> int:
    # Dataset-level seed: every query and every run shares one null sample
    # so skeleton fetches / transforms / vectorizations amortize.
    return int(zlib.crc32(f'drocat-morph-null|{dataset}'.encode('utf-8')))


def null_sample(dataset: str, k: int = NULL_K_DEFAULT,
                exclude: Iterable[int] = (),
                project_root: Optional[str] = None) -> List[int]:
    """Deterministic random bodyId sample of the target dataset.

    The seed derives from the dataset name alone (not the run), so the same
    K neurons are reused across queries and restarts. ``exclude`` removes
    the scored candidates; exclusions are replaced by the next-ranked
    random draws, keeping the sample stable for a given candidate set.
    """
    universe = [int(b) for b in dataset_bodyids(dataset, project_root)]
    if not universe:
        return []
    # Offline cost control: when enough null neurons have locally cached
    # skeletons, sample only from those — scoring a cross-space target
    # needs the skeleton (native -> render transform), and fetching 200
    # skeletons online per run would defeat the cost model.
    cached_ids = _skeleton_cached_ids(dataset, project_root)
    if cached_ids:
        narrowed = [b for b in universe if b in cached_ids]
        if len(narrowed) >= max(int(k), 0):
            universe = narrowed
    excluded = set(int(b) for b in exclude if b is not None)
    # Sample from the full universe order; skipping the excluded candidates
    # keeps the shared selection stable as candidate sets change.
    order = np.random.default_rng(_null_seed(dataset)).permutation(
        len(universe))
    picked: List[int] = []
    seen = set()
    for idx in order:
        bid = int(universe[int(idx)])
        if bid in excluded or bid in seen:
            continue
        seen.add(bid)
        picked.append(bid)
        if len(picked) >= max(int(k), 0):
            break
    return picked


# ---------------------------------------------------------------------------
# Skeleton fetch + transform (query side)
# ---------------------------------------------------------------------------

def fetch_source_skeletons(dataset: str, body_ids: Sequence[int],
                           project_root: Optional[str] = None,
                           log=None,
                           allow_fetch: bool = True) -> Dict[int, Any]:
    """Raw source-space skeletons keyed by bodyId (offline-first).

    Local raw cache first; online fetches only when ``allow_fetch`` is on
    (the unified contract: a strict-offline run must never trigger the
    declined network access, it just gets fewer members with notes).

    The FAFB local release sources are network-free (repair caches, raw
    cache, healed zip), so they are served for strict-offline runs too —
    with the CAVE extrusion pass skipped. BANC's public-release stage
    fetches online and therefore stays behind ``allow_fetch``.
    """
    bids = [int(b) for b in dict.fromkeys(int(b) for b in body_ids or [])]
    if not bids:
        return {}
    root = str(Path(project_root or DEFAULT_PROJECT_ROOT))
    family = _dataset_family(dataset)
    # FAFB release reading is offline-safe without the extrusion pass;
    # BANC would hit the public bucket, so it keeps the online gate.
    use_release = family in ('FAFB', 'BANC') and (
        allow_fetch or family == 'FAFB')
    if use_release:
        try:
            from morphology import load_local_release_skeletons
            got = load_local_release_skeletons(
                dataset, bids, project_root=root, log=None,
                check_extrusions=allow_fetch)
            got = {int(k): v for k, v in (got or {}).items()}
            missing = [b for b in bids if b not in got]
            if missing:
                _logline(log, f'  {len(missing)} source skeleton(s) '
                         f'unavailable locally in {dataset}')
            return got
        except Exception as exc:  # noqa: BLE001
            _logline(log, f'  Local release loader unavailable ({exc}); '
                     'falling back to the raw cache.')
    got: Dict[int, Any] = {}
    try:
        from morphology import (find_similar_raw_cache,
                                fetch_skeleton_on_demand)
        raw_cache = find_similar_raw_cache(dataset, project_root=root,
                                           verbose=False)
        for b in bids:
            try:
                n = raw_cache.load_skeleton(int(b))
            except Exception:  # noqa: BLE001
                n = None
            if n is None and allow_fetch:
                try:
                    n = fetch_skeleton_on_demand(dataset, int(b),
                                                 project_root=root)
                except Exception:  # noqa: BLE001
                    n = None
            if n is not None:
                got[int(b)] = n
    except Exception as exc:  # noqa: BLE001
        _logline(log, f'  Skeleton fetch failed for {dataset}: {exc}')
    return got


def transform_queries(source_dataset: str, target_dataset: str,
                      skeletons: Dict[int, Any],
                      validate_bounds: bool = False,
                      log=None) -> Tuple[List[Any], List[int]]:
    """Transform source neurons into the target render space.

    Returns ``(neurons, bids)`` — aligned; neurons dropped by the transform
    (or with an unreadable bodyId) are excluded and reported.
    """
    if not skeletons:
        return [], []
    from visualize_skeleton import (dataset_native_space,
                                    dataset_render_space,
                                    transform_neurons_to_space)
    native = dataset_native_space(source_dataset)
    render = dataset_render_space(target_dataset)
    items = []
    for bid, neuron in skeletons.items():
        if neuron is None:
            continue
        try:
            neuron.name = f'query_{int(bid)}'
        except Exception:  # noqa: BLE001
            pass
        items.append((int(bid), neuron))
    if not items:
        return [], []
    import navis
    neurons = [n for _b, n in items]
    if native == render:
        transformed = list(navis.NeuronList(neurons))
    else:
        transformed = transform_neurons_to_space(
            navis.NeuronList(neurons), native, render,
            validate_bounds=validate_bounds, verbose=False)
    out_neurons: List[Any] = []
    out_bids: List[int] = []
    for xf in transformed or []:
        name = str(getattr(xf, 'name', '') or '')
        match = re.fullmatch(r'query_(\d+)', name)
        if not match:
            continue
        out_neurons.append(xf)
        out_bids.append(int(match.group(1)))
    dropped = len(items) - len(out_bids)
    if dropped:
        _logline(log, f'  {dropped} query neuron(s) dropped by the '
                 f'{native} -> {render} transform')
    return out_neurons, out_bids


# ---------------------------------------------------------------------------
# Scoring + null-vector sidecar
# ---------------------------------------------------------------------------

def score_pairs(source_dataset: str, target_dataset: str,
                query_neurons: Sequence[Any], query_bids: Sequence[int],
                target_bids: Sequence[int],
                project_root: Optional[str] = None,
                vector_cache: Optional[Dict[int, Any]] = None,
                side_cache: Optional[Dict[int, str]] = None,
                verbose: bool = False) -> pd.DataFrame:
    """Production vector_v2 scores for (query, target) pairs — the shared
    scoring path. Thin wrapper over
    ``morphology.compute_morph_similarity_vs_queries(compute_nblast=False)``
    with an optional persistent target-vector cache. ``side_cache`` is that
    cache's hemisphere column: a stored vector lets the scorer skip the target
    render only when the side is answerable too, so the two travel together."""
    if not len(query_neurons) or not len(target_bids):
        return pd.DataFrame()
    from morphology import compute_morph_similarity_vs_queries
    return compute_morph_similarity_vs_queries(
        list(query_neurons), [int(b) for b in target_bids], target_dataset,
        project_root=str(Path(project_root or DEFAULT_PROJECT_ROOT)),
        compute_nblast=False, verbose=verbose,
        query_bids=[int(b) for b in query_bids],
        source_dataset=source_dataset,
        target_vector_cache=vector_cache,
        target_side_cache=side_cache,
    )


def _scoring_bounds(target_dataset: str,
                    project_root: Optional[str] = None) -> Optional[Any]:
    """The spatial bounds the scorer uses for this target's frame."""
    try:
        from morphology import render_v2_artifacts
        art = render_v2_artifacts(target_dataset,
                                  project_root=str(
                                      Path(project_root
                                           or DEFAULT_PROJECT_ROOT)),
                                  verbose=False)
        if art is not None:
            return art.get('bounds')
    except Exception:  # noqa: BLE001
        pass
    try:
        from morphology import find_similar_dataset_cache_v2
        data = find_similar_dataset_cache_v2(
            target_dataset, project_root=str(
                Path(project_root or DEFAULT_PROJECT_ROOT)),
            verbose=False).load()
        bounds = ((data or {}).get('meta') or {}).get('spatial_bounds')
        if isinstance(bounds, list) and len(bounds) == 2:
            return np.asarray(bounds, dtype=float)
    except Exception:  # noqa: BLE001
        pass
    return None


class TargetVectorStore:
    """Persisted render-space vectors for one target dataset, shared by every
    caller that scores against it — the null sample AND the candidates.

    A small npz sidecar per (dataset, render space) so a cross-space target
    (male-cns) does not re-transform and re-vectorize the same neurons on every
    run: measured at 0.412 s per neuron, which is most of what a morph
    qualification costs. Rows carry each neuron's hemisphere, because a stored
    vector only lets the scorer skip the RENDER if the side is answerable
    without it (see `compute_morph_similarity_vs_queries`).

    Invalidated in layers. A whole-file signature of the population bounds, the
    render space and the V2 vector-cache version drops everything the moment
    the frame changes; so does a `vector_dtype` that is not full precision,
    since a truncated row would re-grade the score instead of only speeding it
    up. Per bodyId, the backing skeleton's `(mtime_ns, size)`
    drops a row whose geometry was healed or re-fetched underneath the store —
    without that, a run would score the old neuron and the report would claim
    the new one. Rows stored with no resolvable path (a FAFB target served from
    the release bundle) fall back on the file signature alone, exactly as
    before: the file's `stats` say how many rows were dropped as stale, so the
    gap is visible rather than assumed away.
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None):
        self.dataset = dataset
        self.root = Path(project_root or DEFAULT_PROJECT_ROOT)
        from morphology import _dataset_folder
        self.dir = (self.root / 'cache' / _dataset_folder(dataset)
                    / 'find_similar' / 'morphology')
        try:
            from visualize_skeleton import dataset_render_space
            self.space = dataset_render_space(dataset)
        except Exception:  # noqa: BLE001
            self.space = 'native'
        self.path = self.dir / f'cross_dataset_targetvec_{self.space}.npz'
        #: what this store did: rows loaded, rows dropped as stale, rows saved
        self.stats: Dict[str, int] = {'loaded': 0, 'stale_dropped': 0,
                                      'saved': 0}
        self._raw_cache = None
        self._raw_cache_tried = False

    def _bounds_signature(self) -> str:
        # Signature includes the V2 vector-cache version: sidecar vectors
        # must share the vectorization basis of the live cache (raw since
        # the raw-basis flip); a cache-version bump invalidates them.
        from morphology import VECTOR_CACHE_V2_VERSION
        tag = f"raw-basis|v2cache={VECTOR_CACHE_V2_VERSION}".encode()
        bounds = _scoring_bounds(self.dataset, str(self.root))
        if bounds is None:
            return hashlib.sha1(tag).hexdigest()[:16]
        return hashlib.sha1(np.asarray(bounds, dtype=float).tobytes()
                            + tag).hexdigest()[:16]

    def _provenance(self, bid: int) -> Tuple[int, int]:
        """`(mtime_ns, size)` of the skeleton this row was computed from, or
        `(-1, -1)` when no file backs it — the value load() treats as
        unverifiable rather than as a mismatch."""
        if not self._raw_cache_tried:
            self._raw_cache_tried = True
            try:
                from morphology import is_fafb_dataset
                if not is_fafb_dataset(self.dataset):
                    from morphology import find_similar_raw_cache
                    self._raw_cache = find_similar_raw_cache(
                        self.dataset, project_root=str(self.root),
                        verbose=False)
            except Exception:  # noqa: BLE001
                self._raw_cache = None
        if self._raw_cache is None:
            return (-1, -1)
        try:
            p = self._raw_cache.find_skeleton_file(int(bid))
            if p is None:
                return (-1, -1)
            st = Path(p).stat()
            return (int(st.st_mtime_ns), int(st.st_size))
        except Exception:  # noqa: BLE001
            return (-1, -1)

    def load(self) -> Tuple[Dict[int, Any], Dict[int, str]]:
        """`(vectors, sides)` for the rows still valid in this frame."""
        vectors: Dict[int, Any] = {}
        sides: Dict[int, str] = {}
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if str(z['space']) != self.space:
                    return vectors, sides
                if str(z['bounds_sig']) != self._bounds_signature():
                    return vectors, sides
                # A float32 sidecar truncates the vector, and upcasting the
                # truncated row would re-grade every score it serves. The
                # store is a speed-up, never a re-grading, so a file not
                # written at full precision is refused whole.
                if str(z['vector_dtype']) != 'float64':
                    return vectors, sides
                bids = np.asarray(z['bodyIds'])
                mat = np.asarray(z['matrix'], dtype=float)
                stored_side = np.asarray(z['sides']) if 'sides' in z else None
                mt = (np.asarray(z['mtimes'], dtype=np.int64)
                      if 'mtimes' in z else None)
                sz = (np.asarray(z['sizes'], dtype=np.int64)
                      if 'sizes' in z else None)
        except Exception:  # noqa: BLE001 - no file, unreadable, legacy shape
            return vectors, sides
        for i, b in enumerate(bids):
            key = int(b)
            if stored_side is None or mt is None or sz is None:
                # a store written before rows carried a side or provenance is
                # not a partial answer: it cannot prove the geometry current
                self.stats['stale_dropped'] += 1
                continue
            if int(mt[i]) >= 0:
                now = self._provenance(key)
                if now != (int(mt[i]), int(sz[i])):
                    self.stats['stale_dropped'] += 1
                    continue
            vectors[key] = mat[i]
            side = str(stored_side[i] or '')
            if side:
                sides[key] = side
        self.stats['loaded'] = len(vectors)
        return vectors, sides

    def update(self, vectors: Dict[int, Any],
               sides: Optional[Dict[int, str]] = None) -> None:
        rows = {int(b): np.asarray(v, dtype=float)
                for b, v in (vectors or {}).items()
                if v is not None and len(v)}
        if len(rows) < 8:
            return  # not worth a file
        try:
            from morphology import VECTOR_V2_DIM
            self.dir.mkdir(parents=True, exist_ok=True)
            existing: Dict[int, Any] = {}
            existing_sides: Dict[int, str] = {}
            if self.path.exists():
                self.stats = {'loaded': 0, 'stale_dropped': 0}
                existing, existing_sides = self.load()
            existing = {k: v for k, v in existing.items()
                        if np.asarray(v).shape == (VECTOR_V2_DIM,)}
            existing.update({k: v for k, v in rows.items()
                             if v.shape == (VECTOR_V2_DIM,)})
            merged_sides = dict(existing_sides)
            merged_sides.update({int(k): str(v) for k, v in (sides or {}).items()
                                 if v})
            bids = np.array(sorted(existing), dtype=np.int64)
            mat = np.vstack([np.asarray(existing[int(b)], dtype=float)
                              for b in bids])
            side_arr = np.array([merged_sides.get(int(b), '') for b in bids],
                                dtype='U8')
            prov = [self._provenance(int(b)) for b in bids]
            mt = np.array([p[0] for p in prov], dtype=np.int64)
            sz = np.array([p[1] for p in prov], dtype=np.int64)
            tmp = self.path.with_suffix('.npz.tmp')
            with open(tmp, 'wb') as fh:
                np.savez(fh, bodyIds=bids, matrix=mat, sides=side_arr,
                         mtimes=mt, sizes=sz,
                         space=np.array(self.space),
                         vector_dtype=np.array(mat.dtype.name),
                         bounds_sig=np.array(self._bounds_signature()))
            tmp.replace(self.path)
            self.stats['saved'] = int(len(bids))
        except Exception:  # noqa: BLE001
            pass


def null_baselines(source_dataset: str, target_dataset: str,
                   query_neurons: Sequence[Any], query_bids: Sequence[int],
                   candidate_bids: Sequence[int], null_k: int = NULL_K_DEFAULT,
                   project_root: Optional[str] = None,
                   vector_cache: Optional[Dict[int, Any]] = None,
                   side_cache: Optional[Dict[int, str]] = None,
                   level: int = 95,
                   exclude: Sequence[int] = (),
                   log=None) -> Tuple[Dict[int, Dict[str, float]], List[int]]:
    """Score the shared null sample against every query; per-query stats.

    Returns ``({source_bid: {p95, median, std, mean, n}}, sample_bids)``.
    The null sample is deterministic per (dataset, k, candidates). ``exclude``
    removes extra bodyIds from the sample on top of the candidates — the query
    bids for intra-dataset runs, where the queries live in the target
    universe and could otherwise draw themselves into the null.

    Persistence is the CALLER's business: it seeds ``vector_cache`` /
    ``side_cache`` from a `TargetVectorStore` and writes them back once, so the
    null sample and the candidate scoring share one read and one write instead
    of each paying for the sidecar.
    """
    sample = null_sample(target_dataset, k=null_k,
                         exclude=list(candidate_bids) + list(exclude),
                         project_root=project_root)
    stats: Dict[int, Dict[str, float]] = {}
    if not sample:
        return stats, []
    if vector_cache is None:
        vector_cache = {}
    if side_cache is None:
        side_cache = {}
    df = score_pairs(source_dataset, target_dataset, query_neurons,
                     query_bids, sample, project_root=project_root,
                     vector_cache=vector_cache, side_cache=side_cache)
    sample_set = set(int(b) for b in sample)
    if df is not None and not df.empty:
        for src_bid, group in df.groupby('source_bodyId'):
            vals = [float(v) for v in group.loc[
                group['target_bodyId'].map(lambda t: int(t) in sample_set),
                'morph_v2_similarity']
                if pd.notna(v) and np.isfinite(v)]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            stats[int(src_bid)] = {
                'p95': float(np.percentile(arr, 95)),
                'bar_p': float(np.percentile(arr, level)),
                'median': float(np.median(arr)),
                'mean': float(arr.mean()),
                'std': float(arr.std() if len(arr) > 1 else 0.0),
                'n': int(len(arr)),
            }
    return stats, sample


# ---------------------------------------------------------------------------
# Morph qualification (Find Homolog)
# ---------------------------------------------------------------------------

@dataclass
class MorphQualification:
    """Result of the pooled morph-qualification pass for one run.

    ``mode`` selects the bar family: ``'null'`` (default — per-source null
    percentile + offset) or ``'mapping_ref'`` (floors v3: the mapper's
    target-side branch pools as the matched+verified reference set —
    native pool floor binding when the pool has >= 2 members, Track-A
    backup floor ``B_b - Δ`` otherwise, per-source null fallback).
    ``level`` is the null percentile of the bar (default 95 = the
    historical null p95).
    """

    source_dataset: str = ''
    target_dataset: str = ''
    null_k: int = NULL_K_DEFAULT
    bar_offset: float = 0.0
    mode: str = 'null'
    level: int = 95
    scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    null_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    active: bool = False
    # mapping_ref mode: per-source bar provenance
    # {source_bid: {'kind': 'native'|'track_a'|None, 'native_floor',
    #               'backup_floor', 'B_b', 'n_refs'}}
    ref_bars: Dict[int, Dict[str, object]] = field(default_factory=dict)
    # mapping_ref mode: per-pair native candidate evidence
    # {(source_bid, target_bid): max native sim to the source's pool}
    native_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # what the persisted target-vector store did for this pass:
    # {'loaded', 'stale_dropped', 'saved', 'targets'} — empty when the caller
    # turned it off. A speed-up that is not reported is indistinguishable from
    # a run that recomputed everything.
    vector_cache: Dict[str, int] = field(default_factory=dict)

    def bar(self, source_bid: int) -> Optional[float]:
        stats = self.null_stats.get(int(source_bid))
        if not stats:
            return None
        return stats.get('bar_p', stats.get('p95')) + float(self.bar_offset)

    def is_qualified(self, source_bid: int, target_bid: int) -> Optional[bool]:
        score = self.scores.get((int(source_bid), int(target_bid)))
        if score is None or not np.isfinite(score):
            return None
        if self.mode == 'mapping_ref':
            rb = self.ref_bars.get(int(source_bid))
            if not rb or rb.get('kind') is None:
                return self._null_qualified(source_bid, target_bid, score)
            if rb['kind'] == 'native':
                nv = self.native_scores.get((int(source_bid),
                                             int(target_bid)))
                if nv is None or not np.isfinite(nv):
                    return None
                return bool(nv >= rb['native_floor'])
            return bool(score >= rb['backup_floor'])
        return self._null_qualified(source_bid, target_bid, score)

    def _null_qualified(self, source_bid: int, target_bid: int,
                        score: float) -> Optional[bool]:
        bar = self.bar(source_bid)
        if bar is None:
            return None
        return bool(score >= bar)

    def z_score(self, source_bid: int, target_bid: int) -> Optional[float]:
        score = self.scores.get((int(source_bid), int(target_bid)))
        stats = self.null_stats.get(int(source_bid))
        if score is None or not stats:
            return None
        std = stats.get('std') or 0.0
        if std <= 0:
            return None
        return float((score - stats['median']) / std)

    def qualified_pairs(self) -> List[Tuple[int, int]]:
        return [pair for pair, ok in self._pair_flags().items() if ok]

    def _pair_flags(self) -> Dict[Tuple[int, int], Optional[bool]]:
        return {pair: self.is_qualified(*pair) for pair in self.scores}


def select_visualized_pairs(results_df: pd.DataFrame,
                            top_n: int,
                            exclude_targets=None,
                            metric: Optional[str] = None) -> List[Tuple[int, int]]:
    """(source_bodyId, target_bodyId) pairs the visualization would render.

    Mirrors ``_visualize_homolog_candidates``' bodyId-level selection:
    per-source top-N by the run's sort ``metric`` (Jaccard-led fallback, the
    same chain the exported CSV uses — hard-coding rank_union here made the
    pooled pre-scoring cover a DIFFERENT set than the scene rendered), then a
    first-wins dedupe on the target across sources. Keeping the two in
    lockstep is what makes the pooled pre-scoring cover exactly the rendered
    set. ``exclude_targets`` mirrors
    the same-dataset scene rule — rows whose target is a query neuron (they
    render as the query layer, not as candidates) drop out before the
    top-N selection.
    """
    if results_df is None or results_df.empty:
        return []
    if 'target_bodyId' not in results_df.columns:
        return []
    candidate = results_df
    if exclude_targets:
        def _excluded(value) -> bool:
            try:
                if pd.isna(value):
                    return False
                return int(value) in exclude_targets
            except (TypeError, ValueError):
                return False
        candidate = candidate.loc[~candidate['target_bodyId'].map(_excluded)]
        if candidate.empty:
            return []
    if top_n and top_n > 0:
        if 'source_bodyId' in candidate.columns:
            per_source = []
            try:
                from .profile_comparator import pick_visualization_metric
            except ImportError:      # direct src/ execution
                from profile_comparator import pick_visualization_metric
            for _key, group in candidate.groupby('source_bodyId'):
                _vc = pick_visualization_metric(metric, group.columns)
                if _vc:
                    group = group.sort_values(_vc, ascending=False,
                                              na_position='last')
                per_source.append(group.head(top_n))
            top = (pd.concat(per_source, ignore_index=True)
                   if per_source else candidate.iloc[0:0])
        elif 'rank_corr' in candidate.columns:
            top = candidate.nlargest(top_n, 'rank_corr')
        else:
            top = candidate.head(top_n)
        top = top.drop_duplicates(subset=['target_bodyId'], keep='first')
    else:
        top = candidate

    def _int(value) -> Optional[int]:
        try:
            if pd.isna(value):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    pairs: List[Tuple[int, int]] = []
    for _idx, row in top.iterrows():
        tgt = _int(row.get('target_bodyId'))
        if tgt is None:
            continue
        src = _int(row.get('source_bodyId')) if 'source_bodyId' in top.columns \
            else None
        pairs.append((src if src is not None else 0, tgt))
    return pairs


def merge_morph_columns(df: pd.DataFrame,
                        qualification: Optional[MorphQualification]
                        ) -> pd.DataFrame:
    """Annotate result rows with the qualification columns.

    Only rows whose (source_bodyId, target_bodyId) pair was scored get
    values; everything else stays empty. No-op (unchanged frame) when the
    qualification is inactive.
    """
    if df is None or df.empty:
        return df
    if qualification is None or not qualification.scores:
        return df
    out = df.copy()

    def _key(row) -> Optional[Tuple[int, int]]:
        try:
            if pd.isna(row.get('source_bodyId')) or \
                    pd.isna(row.get('target_bodyId')):
                return None
            return (int(row['source_bodyId']), int(row['target_bodyId']))
        except (TypeError, ValueError):
            return None

    keys = [_key(row) for _idx, row in out.iterrows()]
    out['morph_v2'] = [qualification.scores.get(k) if k else None
                       for k in keys]
    out['morph_null_p95'] = [
        (qualification.null_stats.get(k[0], {}) or {}).get('p95')
        if k and k[0] in qualification.null_stats else None for k in keys]
    out['morph_z'] = [qualification.z_score(*k) if k else None for k in keys]

    def _bar_kind(k):
        if not k:
            return None
        if qualification.mode == 'mapping_ref':
            rb = qualification.ref_bars.get(k[0])
            if rb and rb.get('kind') in ('native', 'track_a'):
                return rb['kind']
            return 'null_bar'
        # Not 'null': that token round-trips as NaN through the default
        # pandas NA parsing of the exported CSVs.
        return 'null_bar'

    def _bar_value(k):
        if not k:
            return None
        if qualification.mode == 'mapping_ref':
            rb = qualification.ref_bars.get(k[0])
            if not rb:
                return None
            return (rb['native_floor'] if rb.get('kind') == 'native'
                    else rb.get('backup_floor'))
        return qualification.bar(k[0])

    out['morph_bar_kind'] = [_bar_kind(k) for k in keys]
    out['morph_bar'] = [_bar_value(k) for k in keys]
    out['morph_null_level'] = [
        qualification.level if k and k[0] in qualification.null_stats
        else None for k in keys]
    out['morph_qualified'] = [qualification.is_qualified(*k) if k else None
                              for k in keys]
    return out


def filter_qualified_top_matches(top_matches: pd.DataFrame,
                                 qualification: Optional[MorphQualification]
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop scene rows whose pair failed the morph bar (scene exclusion).

    Returns ``(filtered, excluded_rows)`` — ``excluded_rows`` is the
    DataFrame of dropped rows (empty when nothing was dropped) so callers
    can report the pairs and exclude now-empty types from the type-level
    scene. Rows without qualification data (never scored) pass through.
    """
    if top_matches is None or top_matches.empty:
        return top_matches, top_matches.iloc[0:0]
    if qualification is None or not qualification.scores:
        return top_matches, top_matches.iloc[0:0]

    def _pair(row) -> Optional[Tuple[int, int]]:
        try:
            if pd.isna(row.get('source_bodyId')) or \
                    pd.isna(row.get('target_bodyId')):
                return None
            return (int(row['source_bodyId']), int(row['target_bodyId']))
        except (TypeError, ValueError):
            return None

    keep: List[bool] = []
    for _idx, row in top_matches.iterrows():
        pair = _pair(row)
        ok = qualification.is_qualified(*pair) if pair else None
        keep.append(ok is not False)
    if all(keep):
        return top_matches, top_matches.iloc[0:0]
    excluded = top_matches.loc[[not flag for flag in keep]]
    return top_matches.loc[keep], excluded




def _mapper_ref_pools(source_types: Dict[int, str], source_dataset: str,
                      target_dataset: str,
                      log=None) -> Dict[str, List[int]]:
    """{source_type: [target bodyIds]} — the mapper's target-side branch
    pools for the queried source types (the matched+verified reference
    analog available without validation).  Per source type: the mapping
    decision's target types, each refined through the prioritized bridge
    pool; the union is the source type's reference pool.  Silent empty on
    any resolution failure (the caller falls back to the null bar)."""
    if not source_types:
        return {}
    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper
        from ui.neuron_index import resolve_prioritized_bridge_pool
    except Exception as exc:  # noqa: BLE001
        _logline(log, f'[morph-qualify] mapping_ref unavailable: {exc}')
        return {}
    try:
        mapper = get_type_mapper()
        pools: Dict[str, List[int]] = {}
        for stype in sorted(set(source_types.values())):
            if not stype or stype in ('?', 'nan', 'None'):
                continue
            try:
                # POL-6 (resolved as documentation 2026-09-26): bridges
                # stay INCLUDED in the ref pools on purpose — the pool is
                # 'what a mapping could validate against', and TM VEV
                # validates bridged pairs forward; excluding bridge-only
                # ends would let the bar lag the validation it prices.
                dec = mapper.get_mapping_decision(stype, source_dataset,
                                                  target_dataset)
                targets = list(dec.get('target_types') or [])
                refs: List[int] = []
                # Bridge chains do not depend on the target type — derive
                # once per branch (a 1-to-N fan-out used to pay N full
                # searches).
                chains = mapper.get_type_bridges(
                    stype, source_dataset, target_dataset,
                    max_bridges=0)
                for ttype in targets:
                    pool = resolve_prioritized_bridge_pool(
                        source_dataset, target_dataset, chains, stype, ttype)
                    if pool.get('resolution_status') == 'supported':
                        refs.extend(int(b) for b in
                                    (pool.get('target_body_ids') or []))
                if refs:
                    pools[stype] = sorted(set(refs))
                    _logline(log, f'[morph-qualify] mapping_ref: {stype} -> '
                              f'pool {len(pools[stype])} via mapper branches')
            except Exception:  # noqa: BLE001
                continue
        return pools
    except Exception as exc:  # noqa: BLE001
        _logline(log, f'[morph-qualify] mapping_ref pools failed: {exc}')
        return {}


def _finalize_mapping_ref_bars(mq: "MorphQualification",
                               src_refs: Dict[int, List[int]],
                               score_df, log=None) -> None:
    """Compute per-source mapping_ref bars + native candidate evidence.

    Per source: native floor = mean pairwise native sim among its pool refs
    (binding when >= 2 refs have native vectors); otherwise the Track-A
    backup floor ``B_b - Δ`` from the scored (source -> pool) pairs; a
    source with neither keeps the per-source null bar.  Also records each
    scored pair's max native similarity to the source's pool (the native
    candidate evidence the native rung compares)."""
    from morphology import (apply_whitening, find_similar_dataset_cache_v2,
                            v2_similarity_matrix, DEFAULT_V2_BLOCK_WEIGHTS)
    ids = sorted({b for refs in src_refs.values() for b in refs}
                 | {t for (_s, t) in mq.scores})
    cache = find_similar_dataset_cache_v2(mq.target_dataset, verbose=False)
    X, ok, _ = cache.vectors_for(ids, compute_missing=True)
    have = {bid: X[i] for i, bid in enumerate(ids) if ok[i]}
    wd = cache.load() or {}
    whiten = wd.get('whiten')

    def nsim(a, b):
        wa = apply_whitening(whiten, have[a].reshape(1, -1))
        wb = apply_whitening(whiten, have[b].reshape(1, -1))
        return float(v2_similarity_matrix(
            wa, wb.reshape(1, -1), dict(DEFAULT_V2_BLOCK_WEIGHTS))[0][0])

    for src, refs in src_refs.items():
        avail = [r for r in refs if r in have]
        rb = {'kind': None, 'native_floor': None, 'backup_floor': None,
              'B_b': None, 'n_refs': len(avail)}
        if len(avail) >= 2 and whiten is not None:
            sims = [nsim(avail[i], avail[j])
                    for i in range(len(avail))
                    for j in range(i + 1, len(avail))]
            rb['native_floor'] = (float(np.mean(sims))
                                  - MAPPING_REF_NATIVE_MARGIN)
            rb['kind'] = 'native'
        pool_set = set(refs)
        ta = []
        if score_df is not None and not score_df.empty:
            for _idx, row in score_df.iterrows():
                try:
                    if int(row['source_bodyId']) != int(src):
                        continue
                    if int(row['target_bodyId']) in pool_set:
                        v = row.get('morph_v2_similarity')
                        if v is not None and pd.notna(v) and np.isfinite(v):
                            ta.append(float(v))
                except (TypeError, ValueError):
                    continue
        if ta:
            rb['B_b'] = float(np.mean(ta))
            if rb['kind'] is None:
                rb['kind'] = 'track_a'
                rb['backup_floor'] = rb['B_b'] - MAPPING_REF_TRACK_A_OFFSET
        mq.ref_bars[int(src)] = rb
        _logline(log, f'[morph-qualify] mapping_ref source {src}: kind='
                 f'{rb["kind"]} refs={len(avail)} B_b='
                 f'{rb["B_b"] is not None and round(rb["B_b"], 3)}')
    # native candidate evidence per scored pair
    for (src, tgt) in list(mq.scores):
        refs = src_refs.get(src)
        if not refs or tgt not in have or whiten is None:
            continue
        sims = [nsim(tgt, r) for r in refs if r in have and r != tgt]
        if sims:
            mq.native_scores[(src, tgt)] = max(sims)


def qualify_visualized_pairs(
        source_dataset: str, target_dataset: str,
        pairs: Sequence[Tuple[int, int]],
        null_k: int = NULL_K_DEFAULT,
        project_root: Optional[str] = None,
        bar_offset: float = 0.0,
        mode: str = 'null',
        level: int = 95,
        source_types: Optional[Dict[int, str]] = None,
        prune_pool_refs: bool = True,
        use_vector_store: bool = True,
        log=None) -> MorphQualification:
    """Score the visualized (source, target) pairs + shared null sample.

    One scoring call covers the union of candidates and the null sample so
    target transforms/vectorizations are computed once. Same-dataset
    (intra) runs go through the identity chain; their query bids are
    excluded from the null sample since the queries live in the target
    universe. Raises ValueError on scope violations; returns an inactive
    MorphQualification (with a warning) when scoring yields nothing usable.

    ``mode='mapping_ref'`` (floors v3) replaces the per-source null bar
    with the mapper-referenced ladder wherever the pair's source type
    resolves to mapper branch pools: native pool floor (>= 2 refs)
    binding, Track-A backup floor ``B_b - Δ``, per-source null fallback.
    ``source_types`` maps source bodyId -> source type name (required for
    mapping_ref; built by the caller from the results rows).

    ``prune_pool_refs`` (default True) is the mapping_ref convention that a
    branch pool member which was only fetched to ANCHOR a bar is not a
    candidate of its own pool, so its verdict is dropped. That is right for a
    caller whose candidate set is the branch pool (the supervised ladder,
    Find Homolog) and wrong for one whose candidates are its own list — where
    keeping the default silently deletes the verdict of every candidate the
    mapper also claims. Pass False for the latter: the grading stays honest
    because `native_scores` never counts the candidate against itself
    (`r != tgt`), so a pool member is measured against the OTHER refs, exactly
    like an outsider. (Found on 2026-09-23: `pooling` lost 35 of its 41
    verdicts this way and labelled them `no-score`.)

    ``use_vector_store`` (default True) shares the persisted
    `TargetVectorStore` of this target across the null sample and the
    candidates, so a repeat run over the same dataset pair does not re-render
    and re-vectorize neurons it has already measured. What it loaded, dropped
    as stale and saved rides on `MorphQualification.vector_cache`, so a speedup
    is reported rather than felt.
    """
    mq = MorphQualification(source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            null_k=int(null_k), bar_offset=float(bar_offset),
                            mode=str(mode or 'null'), level=int(level))
    source_types = dict(source_types or {})
    check = check_pair_availability(source_dataset, target_dataset)
    if not check['ok']:
        raise ValueError(check['reason'])
    scope = dataset_scope(target_dataset)
    mq.warnings.extend(scope['warnings'])
    # Same-dataset pairs short-circuit the pair check to identity before
    # any scope validation, so enforce the target scope here — artifact-
    # less datasets (male-cns:v0.9, optic-lobe) refuse with the clear
    # reason instead of failing deep in the scorer.
    if not scope['ok']:
        raise ValueError(scope['reason'])

    unique_sources, unique_targets = [], []
    for src, tgt in pairs or []:
        if src is None or tgt is None:
            continue
        if int(src) not in unique_sources:
            unique_sources.append(int(src))
        if int(tgt) not in unique_targets:
            unique_targets.append(int(tgt))
    if not unique_sources or not unique_targets:
        mq.warnings.append('No visualized pairs to qualify.')
        return mq

    _logline(log, f'[morph-qualify] fetching {len(unique_sources)} query '
             f'skeleton(s) from {source_dataset}')
    skeletons = fetch_source_skeletons(source_dataset, unique_sources,
                                       project_root=project_root, log=log)
    neurons, bids = transform_queries(source_dataset, target_dataset,
                                      skeletons, log=log)
    if not neurons:
        mq.warnings.append('No query skeletons could be transformed into '
                           f'{target_dataset} render space; qualification '
                           'inactive.')
        return mq

    # Unified skeleton acquisition: score the visualized targets only
    # after they are in the shared raw cache (per-neuron isolation; uncached
    # targets would otherwise silently miss their morph verdict).
    # mapping_ref: resolve the mapper's target-side branch pools per source
    # type and make sure every pool member is fetched + scored (they back
    # the native floor and the B_b baseline).
    src_refs: Dict[int, List[int]] = {}
    if mq.mode == 'mapping_ref':
        ref_pools = _mapper_ref_pools(source_types, source_dataset,
                                      target_dataset, log=log)
        for src in unique_sources:
            refs = ref_pools.get(source_types.get(src))
            if refs:
                src_refs[src] = list(refs)
        if not src_refs:
            mq.warnings.append('mapping_ref mode: no source type resolved '
                               'to mapper branch pools; falling back to '
                               'the null bar for every source.')
    pool_bids = sorted({b for refs in src_refs.values() for b in refs})
    if pool_bids:
        unique_targets = unique_targets + [b for b in pool_bids
                                           if b not in unique_targets]
    missing_targets = [b for b in unique_targets
                       if b not in _skeleton_cached_ids(
                           target_dataset, project_root)]
    if missing_targets:
        _logline(log, f'[morph-qualify] fetching {len(missing_targets)} '
                      'target skeleton(s) not locally cached')
        fetch_source_skeletons(target_dataset, missing_targets,
                               project_root=project_root, log=log)

    vector_cache: Dict[int, Any] = {}
    side_cache: Dict[int, str] = {}
    # ONE store for the null sample and the candidates, written once after both.
    # It is what the 0.412 s/neuron preparation is for: a repeat run over the
    # same dataset pair reaches the same targets and must not re-pay for them.
    store = (TargetVectorStore(target_dataset, project_root)
             if use_vector_store else None)
    if store is not None:
        v, s = store.load()
        vector_cache.update(v)
        side_cache.update(s)
    stats, sample = null_baselines(
        source_dataset, target_dataset, neurons, bids, unique_targets,
        null_k=null_k, project_root=project_root, vector_cache=vector_cache,
        side_cache=side_cache,
        level=level, exclude=bids, log=log)
    mq.null_stats = stats
    if not stats:
        mq.warnings.append('Null sample produced no usable scores; '
                           'qualification inactive.')
        return mq
    thin = [b for b, s in stats.items() if s.get('n', 0) < NULL_MIN_N]
    if thin:
        mq.warnings.append(
            f'Null sample thin (n<{NULL_MIN_N}) for {len(thin)} query '
            f'neuron(s); bars are noisy.')

    targets_all = sorted(set(unique_targets)
                         | {int(b) for b in sample})
    df = score_pairs(source_dataset, target_dataset, neurons, bids,
                     targets_all, project_root=project_root,
                     vector_cache=vector_cache, side_cache=side_cache)
    if store is not None:
        # written before the emptiness checks, because whatever this call did
        # measure is worth keeping; `stats` rides on to the run's record
        store.update(vector_cache, side_cache)
        mq.vector_cache = dict(store.stats)
        mq.vector_cache['targets'] = len(vector_cache)
    if df is None or df.empty:
        mq.warnings.append('Candidate scoring produced no rows; '
                           'qualification inactive.')
        return mq
    sample_set = set(int(b) for b in sample)
    # score_pairs covers the full unique-sources x unique-targets product;
    # only the requested (visualized) pairs carry a qualification verdict.
    requested = {(int(s), int(t)) for s, t in pairs or []}
    for _idx, row in df.iterrows():
        try:
            src, tgt = int(row['source_bodyId']), int(row['target_bodyId'])
        except (TypeError, ValueError):
            continue
        if tgt in sample_set:
            continue
        if (src, tgt) not in requested:
            continue
        val = row.get('morph_v2_similarity')
        if val is None or pd.isna(val):
            continue
        mq.scores[(src, tgt)] = float(val)

    if mq.mode == 'mapping_ref' and src_refs:
        _finalize_mapping_ref_bars(mq, src_refs, df, log=log)
    pool_bid_set = set(pool_bids)
    for (src, tgt), score in list(mq.scores.items()):
        if prune_pool_refs and tgt in pool_bid_set:
            # Pool members were scored only to anchor B_b / native floors —
            # they are references, never candidates of their own pool.
            del mq.scores[(src, tgt)]
            continue
        if mq.is_qualified(src, tgt) is None:
            # No bar for this source (unscored null / no mapping_ref
            # basis): drop the pair.
            del mq.scores[(src, tgt)]
    n_qualified = sum(1 for pair in mq.scores if mq.is_qualified(*pair))
    mq.active = bool(mq.scores)
    _logline(log, f'[morph-qualify] scored {len(mq.scores)} visualized '
             f'pair(s); {n_qualified} above the bar '
             f'(mode={mq.mode}, level={mq.level}, '
             f'offset {mq.bar_offset:+.2f}); null n='
             f'{sorted({s["n"] for s in stats.values()})}')
    return mq


# ---------------------------------------------------------------------------
# Population-artifact bootstrap (BANC, male-cns v0.9)
# ---------------------------------------------------------------------------

def population_artifacts_ready(dataset: str,
                               project_root: Optional[str] = None) -> bool:
    try:
        from morphology import find_similar_dataset_cache_v2
        data = find_similar_dataset_cache_v2(
            dataset,
            project_root=str(Path(project_root or DEFAULT_PROJECT_ROOT)),
            verbose=False).load()
        meta = (data or {}).get('meta') or {}
        return bool(data is not None and data.get('raw') is not None
                    and meta.get('mean') is not None
                    and meta.get('spatial_bounds'))
    except Exception:  # noqa: BLE001
        return False


def ensure_population_artifacts(
        dataset: str, sample_k: int = BOOTSTRAP_SAMPLE_DEFAULT,
        project_root: Optional[str] = None,
        log=None) -> Dict[str, Any]:
    """One-time offline build of the vector_v2 population artifacts.

    Samples locally cached skeletons (seeded, deterministic), vectorizes
    them with the cache's own row recipe, and persists the vector parquet +
    ``meta_v2.json`` + whitener sidecar so ``find_similar_dataset_cache_v2
    (...).load()`` succeeds — production scorer parity, no NeuPrint or
    network dependency. Returns ``{status: ready|ready-existing|insufficient,
    n_rows, ...}``.
    """
    root = str(Path(project_root or DEFAULT_PROJECT_ROOT))
    from morphology import (find_similar_dataset_cache_v2,
                            _load_cached_skeleton_file, _neuron_points,
                            _load_neuron_type_map)
    cache_v2 = find_similar_dataset_cache_v2(dataset, project_root=root,
                                             verbose=False)
    if cache_v2.load() is not None:
        return {'status': 'ready-existing', 'n_rows': None, 'dataset': dataset}

    files = sorted(cache_v2._discover_skeleton_files())
    if len(files) < BOOTSTRAP_MIN_ROWS:
        return {'status': 'insufficient', 'n_rows': 0, 'dataset': dataset,
                'reason': f'only {len(files)} locally cached skeletons '
                          f'(< {BOOTSTRAP_MIN_ROWS})'}
    rng = np.random.default_rng(_null_seed(f'bootstrap|{dataset}'))
    if len(files) > sample_k:
        pick = rng.choice(len(files), int(sample_k), replace=False)
        files = [files[int(i)] for i in sorted(pick)]

    _logline(log, f'[bootstrap {dataset}] vectorizing {len(files)} cached '
             'skeleton(s) (one-time population-artifact build)')
    # Pass 1: population bounds from the sampled skeletons.
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    loaded: List[Tuple[int, Any]] = []
    for path in files:
        try:
            neuron = _load_cached_skeleton_file(path)
        except Exception:  # noqa: BLE001
            continue
        if neuron is None:
            continue
        pts = _neuron_points(neuron)
        if not len(pts):
            continue
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
        bid = int(Path(str(path)).stem.split('.')[0])
        loaded.append((bid, neuron))
    if len(loaded) < BOOTSTRAP_MIN_ROWS or not np.isfinite(lo).all():
        return {'status': 'insufficient', 'n_rows': len(loaded),
                'dataset': dataset,
                'reason': 'usable skeletons below the minimum'}
    pad = 0.02 * np.maximum(hi - lo, 0.0)
    bounds = np.vstack([lo - pad, hi + pad])
    cache_v2._spatial_bounds = bounds
    cache_v2._bounds_resolved = True

    # Pass 2: vectorize with the cache's own recipe (raw basis + v2 features).
    rows: List[Tuple[int, List[float], str]] = []
    for bid, neuron in loaded:
        row = cache_v2._in_memory_vector_row(bid, neuron)
        if row is not None:
            rows.append((int(row[0]), row[1], str(row[2])))
    if len(rows) < BOOTSTRAP_MIN_ROWS:
        return {'status': 'insufficient', 'n_rows': len(rows),
                'dataset': dataset,
                'reason': 'vectorizable skeletons below the minimum'}

    raw = np.asarray([r[1] for r in rows], dtype=float)
    mean = raw.mean(axis=0)
    std = np.where(raw.std(axis=0) <= 0, 1.0, raw.std(axis=0))
    cache_v2.meta_path.parent.mkdir(parents=True, exist_ok=True)
    cache_v2._write_meta({'mean': mean.tolist(), 'std': std.tolist()},
                         n_rows=len(rows), rep='skeleton')

    local_release = _dataset_family(dataset) in ('FAFB', 'BANC')
    type_map, instance_map = ({}, {})
    try:
        type_map, instance_map = _load_neuron_type_map(dataset, root)
    except Exception:  # noqa: BLE001
        pass
    feature_cols = cache_v2._feature_columns()
    records = []
    for bid, vector, rep in rows:
        rec = {'bodyId': str(bid) if local_release else bid, 'rep': rep}
        for i, name in enumerate(feature_cols):
            rec[name] = float(vector[i])
        lookup = (str(bid) if local_release else bid)
        rec['type'] = (type_map or {}).get(lookup, '') if type_map else ''
        rec['instance'] = (instance_map or {}).get(lookup, '') \
            if instance_map else ''
        records.append(rec)
    df = pd.DataFrame(records)
    tmp = cache_v2.parquet_path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp)
    tmp.replace(cache_v2.parquet_path)

    # Whitener sidecar (fitted on standardized rows, like every load()).
    X_std = (raw - mean) / std
    W = cache_v2._whitener(X_std)

    data = cache_v2.load()
    ok = data is not None and data.get('raw') is not None
    _logline(log, f'[bootstrap {dataset}] artifacts written: '
             f'{len(rows)} rows, whiten={"fitted" if W.shape[0] > 1 and abs(W - np.eye(W.shape[0])).max() > 0 else "identity"} '
             f'-> {"ready" if ok else "FAILED verification"}')
    return {'status': 'ready' if ok else 'failed', 'n_rows': len(rows),
            'dataset': dataset}


# ---------------------------------------------------------------------------
# Cross-dataset comparison backend (Similarity -> Morphology -> Cross-Dataset)
# ---------------------------------------------------------------------------

def _safe_name(text: str, limit: int = 40) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_'
                   for ch in str(text))
    return safe.strip('_')[:limit] or 'neuron'


def _tag_scene_members(members, type_name, abbrev, dataset):
    """Name bridged-scene members uniquely and tag their source dataset.

    Each entry of ``members`` is a ``(body_id, neuron)`` pair; the returned
    list preserves that order.  Unique per-member name: navis uniquifies
    duplicate names inside a NeuronList, which defeats the plotly
    trace-identity resolution (legend leaves collapse and indices can slip
    past the neuron count, crashing the layer loop).
    """
    named = []
    for body_id, neuron in members:
        neuron.name = (f'{_safe_name(type_name, 20)}_{abbrev}'
                       f'_{int(body_id)}')
        neuron._drocat_source_dataset = dataset
        named.append(neuron)
    return named


def _dataset_abbrev(dataset: str) -> str:
    family = _dataset_family(dataset)
    if family == 'FAFB':
        return 'FAFB'
    if family == 'BANC':
        # Disambiguate the two BANC releases (banc_v626 / banc_v888).
        version = str(dataset).split('_')[-1]
        return f'BANC{version[:4]}' if version else 'BANC'
    return 'MCNS'


class CrossDatasetMorphComparer:
    """Compare the queried neurons' morphology across datasets.

    Pure comparison of the queried neurons (type-resolved per dataset,
    members capped) — pairwise vector_v2 per dataset pair in the target's
    render space, a seeded null baseline per pair, and overlay scenes in a
    chosen reference template. The export mirrors the connectivity-
    profiling layout: per-pair ``results/`` with the type × type mean
    matrix, the bodyId × bodyId score matrix, the long-form scores CSV and
    the null reference; per-pair ``visualization/`` with interactive
    VisPath heatmaps; and a tabbed ``report.html`` built on the shared
    ``report_kit`` (pair tabs → Type/BodyId level tabs, Ward-clustered
    cards, embedded Plotly so it renders offline).
    """

    def __init__(self, datasets: Optional[List[str]] = None,
                 query: Any = None, output_dir: Optional[str] = None,
                 saveas: Optional[str] = None,
                 max_members_per_type: int = 25,
                 null_k: int = NULL_K_DEFAULT,
                 reference_template: Optional[str] = None,
                 scene_members_per_type: int = 3,
                 visualize: bool = True,
                 generate_heatmaps: bool = True,
                 fetch_online: bool = True,
                 token: str = '',
                 use_auto_type_mapping: bool = True,
                 use_cache: bool = True,
                 verbose: bool = True,
                 n_workers: int = 8,
                 project_root: Optional[str] = None):
        self.datasets = [str(d) for d in (datasets or [])]
        self.query = query
        self.output_dir = output_dir
        self.saveas = saveas
        self.max_members_per_type = int(max_members_per_type)
        self.null_k = int(null_k)
        self.reference_template = reference_template
        self.scene_members_per_type = max(1, int(scene_members_per_type))
        self.visualize = bool(visualize)
        self.generate_heatmaps = bool(generate_heatmaps)
        self.fetch_online = bool(fetch_online)
        self.token = token or ''
        self.use_auto_type_mapping = bool(use_auto_type_mapping)
        self.use_cache = bool(use_cache)
        self.verbose = bool(verbose)
        self.n_workers = int(n_workers)
        self.project_root = str(Path(project_root or DEFAULT_PROJECT_ROOT))
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ log
    def _log(self, message: str) -> None:
        if self.verbose:
            print(str(message), flush=True)

    # ------------------------------------------------------------ resolution
    def _load_neuron_frame(self, dataset: str) -> Optional[pd.DataFrame]:
        root = Path(self.project_root)
        safe = str(dataset).replace(':', '_').replace('.', '_')
        candidates = [
            root / 'datasets' / safe / f'{safe}_allneurons_neuron_df.parquet',
            root / 'datasets' / safe / f'{safe}_allneurons_neuron_df.csv',
            root / 'datasets' / safe / f'{safe}_neurons.parquet',
            root / 'datasets' / safe / f'{safe}_neurons.csv',
            root / 'datasets' / safe / 'neurons.parquet',
            root / 'datasets' / safe / 'neurons.csv',
            root / 'neuron_indexes' / safe / 'neuron_index.parquet',
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                return (pd.read_parquet(path) if path.suffix == '.parquet'
                        else pd.read_csv(path, low_memory=False))
            except Exception as exc:  # noqa: BLE001
                self._log(f'Warning: could not read {path.name}: {exc}')
        return None

    def _get_type_mapper(self):
        """The shared process-wide CrossDatasetTypeMapper, or None.

        With the option off, or when the mapper cannot load, resolution
        degrades to same-name/pattern (the pre-mapper behavior) — a broken
        mapper must never block a comparison.
        """
        if not self.use_auto_type_mapping or get_type_mapper is None:
            return None
        try:
            mapper = get_type_mapper()
        except Exception as exc:  # noqa: BLE001
            self._log(f'Auto type mapping unavailable: {exc}')
            return None
        if mapper is None or not getattr(mapper, '_loaded', False):
            return None
        return mapper

    def _mapper_search_names(self, token: str, dataset: str,
                             mapper, snapshot, alias_cache, bridge_cache
                             ) -> Tuple[Optional[List[str]], List[str]]:
        """Validated target type names for one token in one dataset.

        Returns ``(names, notes)``: ``names=None`` means "no mapper verdict
        — fall through to the raw local path" (patterns, bodyIds, mapper
        off/absent). The policy mirrors homolog finding's resolver contract:
        mapped / bridged / valid-split targets resolve; a CONFLICT fails
        closed (no members, same-name suppressed); unmapped and
        evidence-only/claimed fall back to the raw name with a note.
        """
        notes: List[str] = []
        if mapper is None or resolve_valid_targets is None:
            return None, notes
        text = str(token or '').strip()
        if not text or text.isdigit() or ('*' in text):
            return None, notes  # bodyId / pattern tokens stay dataset-local
        try:
            resolution = resolve_valid_targets(
                mapper, text, None, dataset, snapshot=snapshot,
                alias_cache=alias_cache, bridge_cache=bridge_cache)
        except Exception as exc:  # noqa: BLE001
            self._log(f'Warning: mapper resolution for {text!r} failed: {exc}')
            return None, notes
        status = getattr(resolution, 'status', '')
        if status == 'conflict':
            notes.append(
                f'{text}: mapping conflict in {dataset} '
                f'({getattr(resolution, "reason", "") or "unresolved"}); '
                'no automatic target (fail closed)')
            return [], notes
        names = list(expansion_targets(resolution)) if expansion_targets is not None else []
        if not names and status == 'evidence_only' and (
                getattr(resolution, 'target_types', ()) or ()):
            # RES-1 follow-up (2026-09-26): the resolver now keeps an
            # evidence-only union UNLICENSED (no equivalence key), but
            # member ENUMERATION here may still use its convergence
            # members — this is the aggregation answer ('APDN3 in MCNS is
            # carried by these N types'), disclosed as such and never a
            # unique mapping.  claimed stays no-members (the dataset does
            # not carry the claimed name at all).
            names = [str(t) for t in resolution.target_types]
            notes.append(
                f'{text}: one of N in {dataset} '
                f'({"/".join(names)}) — aggregation members, not a '
                'unique mapping')
        if not names:
            # claimed / anything else without targets: a verdict the
            # target dataset cannot fulfil — no expansion, explicit note
            # (not silently empty).
            claimed = tuple(getattr(resolution, 'target_types', ()) or ())
            notes.append(
                f'{text}: {"claim" if status == "claimed" else status} '
                f'{"/".join(claimed) or "(none)"} is not carried by '
                f'{dataset}; no members')
            return [], notes
        if names != [text]:
            note = f'{text} → {"/".join(names)} ({dataset})'
            secondary = tuple(getattr(resolution, 'secondary_targets', ())
                              or ())
            if secondary:
                note += f' (secondary: {", ".join(secondary)})'
            notes.append(note)
        return names, notes

    def _resolve_members(self, tokens: Sequence[str],
                         dataset: str) -> Tuple[Dict[str, List[int]],
                                                List[str]]:
        """{type: [bodyIds]} for each query token in one dataset.

        Non-numeric type tokens resolve through the shared validity-aware
        CrossDatasetTypeMapper (curated renames and valid splits resolve;
        conflicts fail closed; unmapped fall back to the raw name) — the
        same contract as the connectivity Comparison sub-tab. Patterns and
        bodyIds stay dataset-local. Tokens with no members in this dataset
        are reported in the notes list — explicit empty rows, never silent
        omissions.
        """
        from neuron_search import resolve_dataframe_query
        frame = self._load_neuron_frame(dataset)
        members: Dict[str, List[int]] = {}
        notes: List[str] = []
        if frame is None:
            return members, [f'{dataset}: no neuron table available locally']
        if 'bodyId' not in frame.columns:
            return members, [f'{dataset}: neuron table has no bodyId column']

        def _bid(value) -> Optional[int]:
            try:
                if value is None or (isinstance(value, float)
                                     and pd.isna(value)):
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        type_lookup: Dict[str, str] = {}
        if 'type' in frame.columns:
            for bid_val, type_val in zip(frame['bodyId'], frame['type']):
                key = _bid(bid_val)
                if key is None:
                    continue
                type_lookup.setdefault(str(key), str(type_val or '').strip())

        mapper = self._get_type_mapper()
        snapshot = (MapperSnapshot(mapper)
                    if mapper is not None and MapperSnapshot is not None
                    else None)
        alias_cache: Dict[Any, Any] = {}
        bridge_cache: Dict[Tuple[str, str, str], list] = {}

        for token in tokens:
            text = str(token or '').strip()
            if not text:
                continue
            search_names, mapper_notes = self._mapper_search_names(
                text, dataset, mapper, snapshot, alias_cache, bridge_cache)
            notes.extend(mapper_notes)
            if search_names is not None and not search_names:
                continue   # fail-closed / unfulfilled claim: note recorded

            token_body_ids: List[int] = []
            label_default = text
            if search_names is None:
                # Raw local path: bodyIds, patterns, same-name, or mapper
                # off/unavailable.
                try:
                    hits, _info = resolve_dataframe_query(
                        frame, text, search_columns='auto')
                except Exception as exc:  # noqa: BLE001
                    hits = []
                    self._log(f'Warning: resolving {text!r} in {dataset} '
                              f'failed: {exc}')
                for hit in hits:
                    bid = _bid(hit)
                    if bid is not None:
                        token_body_ids.append(bid)
            else:
                # Mapped names: exact type-column lookups (the mapper
                # licensed these names; nothing else).
                label_default = search_names[0]
                for name in search_names:
                    try:
                        hits, _info = resolve_dataframe_query(
                            frame, name, search_columns='type')
                    except Exception as exc:  # noqa: BLE001
                        hits = []
                        self._log(f'Warning: resolving mapped {name!r} in '
                                  f'{dataset} failed: {exc}')
                    for hit in hits:
                        bid = _bid(hit)
                        if bid is not None:
                            token_body_ids.append(bid)

            token_body_ids = sorted(dict.fromkeys(token_body_ids))
            if not token_body_ids:
                notes.append(f'{text}: 0 members in {dataset}')
                continue
            capped = token_body_ids[:self.max_members_per_type]
            if len(token_body_ids) > len(capped):
                notes.append(f'{text}: capped to {len(capped)} of '
                             f'{len(token_body_ids)} members in {dataset}')
            by_type: Dict[str, List[int]] = {}
            for bid in capped:
                type_name = type_lookup.get(str(bid)) or label_default
                by_type.setdefault(type_name, []).append(bid)
            for type_name, bids in by_type.items():
                members.setdefault(type_name, [])
                for bid in bids:
                    if bid not in members[type_name]:
                        members[type_name].append(bid)
        return members, notes

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        tokens = self._query_tokens()
        if not tokens:
            raise ValueError('Enter at least one query neuron (type or '
                             'pattern).')
        scopes = validate_cross_dataset_sets(self.datasets)
        datasets = [s for s in self.datasets]
        for scope in scopes:
            self.warnings.extend(scope['warnings'])
        # FAFB's L/R annotation is inverted relative to the other datasets
        # in the ORIGINAL source data (male-cns + BANC agree with each
        # other; FLYWIRE-native x runs opposite). Surface it once per run:
        # side labels do not transfer across datasets here, so same-type
        # partners pair across sides. Scores are unaffected — the morphology
        # features are side-invariant — and nothing is flipped in either
        # direction.
        if (any(_dataset_family(name) == 'FAFB' for name in datasets)
                and any(_dataset_family(name) != 'FAFB' for name in datasets)):
            self.warnings.append(
                'FAFB L/R annotation is opposite to the other datasets in '
                'the original data: same-type cross-dataset partners pair '
                'across sides (L <-> R). Scores are unaffected - the '
                'morphology features are side-invariant and nothing was '
                'flipped.')
        # BANC / stale-artifact bootstrap (offline; one-time per dataset).
        for name in datasets:
            scope = dataset_scope(name)
            if scope['ok'] and not population_artifacts_ready(
                    name, self.project_root):
                result = ensure_population_artifacts(
                    name, project_root=self.project_root, log=self._log)
                if result['status'] == 'insufficient':
                    self.warnings.append(
                        f'{name}: population artifacts missing and the '
                        f'bootstrap found too few cached skeletons '
                        f'({result.get("reason")}); scores for this dataset '
                        'use the degraded un-whitened scorer.')

        run_path = self._output_path(tokens)
        run_path.mkdir(parents=True, exist_ok=True)
        # Per-run abbreviations (same-family datasets get suffixes so pair
        # folders and overview columns never collide).
        self._abbrevs: Dict[str, str] = {}
        used: Dict[str, int] = {}
        for name in datasets:
            ab = _dataset_abbrev(name)
            if ab in used:
                used[ab] += 1
                self._abbrevs[name] = f"{ab}{used[ab]}"
            else:
                used[ab] = 1
                self._abbrevs[name] = ab
        self._log(f'📁 Output folder: {run_path}')

        members: Dict[str, Dict[str, List[int]]] = {}
        notes: List[str] = []
        for name in datasets:
            got, resolved_notes = self._resolve_members(tokens, name)
            members[name] = got
            notes.extend(resolved_notes)

        pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        skeletons_cache: Dict[str, Dict[int, Any]] = {}
        for a in datasets:
            src_members = members.get(a) or {}
            if not src_members:
                continue
            if a not in skeletons_cache:
                bids = sorted({b for bids_ in src_members.values()
                               for b in bids_})
                self._log(f'[{a}] fetching {len(bids)} query skeleton(s)')
                skeletons_cache[a] = fetch_source_skeletons(
                    a, bids, project_root=self.project_root, log=self._log,
                    allow_fetch=self.fetch_online)
            for b in datasets:
                if a == b:
                    continue
                tgt_members = members.get(b) or {}
                if not tgt_members:
                    continue
                self._log(f'--- pair {a} -> {b} ---')
                pairs[(a, b)] = self._score_pair(
                    a, src_members, b, tgt_members, skeletons_cache[a])

        overview = self._overview_frame(tokens, datasets, members, pairs)
        files = self._write_outputs(run_path, tokens, datasets, members,
                                    pairs, overview, notes)
        if self.generate_heatmaps:
            try:
                files.extend(self._write_heatmaps(run_path, pairs))
            except Exception as exc:  # noqa: BLE001
                self._log(f'heatmap generation failed (comparison kept): '
                          f'{exc}')
        scenes = []
        if self.visualize:
            try:
                scenes = self._render_scenes(run_path, datasets, members)
                files.extend(scenes)
            except Exception as exc:  # noqa: BLE001
                # Kept fail-soft: a scene problem must never lose the
                # comparison itself. (The 2026-09-16 bridged-layer crash —
                # overlay neurons re-fetched through NeuPrint, then a
                # legend-index overrun — is fixed in visualize_skeleton.)
                self._log(f'3D visualization failed (comparison kept): {exc}')
        files = self._write_report(run_path, tokens, datasets, members,
                                   pairs, overview, notes, scenes,
                                   existing=files)

        return {'output_folder': str(run_path),
                'files': [{'path': str(p)} for p in files],
                'warnings': list(self.warnings)}

    # ------------------------------------------------------------ internals
    def _query_tokens(self) -> List[str]:
        value = self.query
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        out: List[str] = []
        for item in value:
            text = str(item or '').strip()
            if text and text not in out:
                out.append(text)
        return out

    def _output_path(self, tokens: Sequence[str]) -> Path:
        output_dir = Path(self.output_dir) if self.output_dir \
            else Path(self.project_root) / 'local_data' / 'morph_cross_dataset'
        if self.saveas:
            name = self.saveas
        else:
            ds = '_'.join(_dataset_abbrev(d) for d in self.datasets)
            name = (f"morph_cross_{ds}_{_safe_name('_'.join(tokens), 50)}"
                    f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return output_dir / name

    def _score_pair(self, source: str, src_members: Dict[str, List[int]],
                    target: str, tgt_members: Dict[str, List[int]],
                    skeletons: Dict[int, Any]) -> Dict[str, Any]:
        query_neurons, query_bids = transform_queries(
            source, target, skeletons, log=self._log)
        pair_info: Dict[str, Any] = {
            'source': source, 'target': target,
            'src_members': src_members, 'tgt_members': tgt_members,
            'n_queries': len(query_neurons), 'baseline': {}, 'null_n': 0,
        }
        if not query_neurons:
            pair_info['error'] = (f'no {source} skeleton could be '
                                  f'transformed into {target} render space')
            return pair_info
        tgt_bids = sorted({b for bids_ in tgt_members.values() for b in bids_})
        bid_to_type = {b: t for t, bids_ in tgt_members.items() for b in bids_}
        src_bid_type = {b: t for t, bids_ in src_members.items() for b in bids_}

        # The scorer never fetches TARGET skeletons on demand; without a
        # pre-fetch, members outside the local cache would silently drop
        # out of the matrix (NaN cells). Local sources (raw cache, FAFB
        # release bundle) load even for strict-offline runs; only true
        # online fetches are gated by fetch_online.
        missing = [b for b in tgt_bids
                   if b not in _skeleton_cached_ids(
                       target, self.project_root)]
        if missing:
            if self.fetch_online:
                self._log(f'[{target}] fetching {len(missing)} target '
                          'skeleton(s) not locally cached')
            else:
                self._log(f'[{target}] loading {len(missing)} target '
                          'skeleton(s) from local sources')
            fetch_source_skeletons(target, missing,
                                   project_root=self.project_root,
                                   log=self._log,
                                   allow_fetch=self.fetch_online)

        vector_cache: Dict[int, Any] = {}
        side_cache: Dict[int, str] = {}
        store = (TargetVectorStore(target, self.project_root)
                 if self.use_cache else None)
        if store is not None:
            v, s = store.load()
            vector_cache.update(v)
            side_cache.update(s)
        stats, sample = null_baselines(
            source, target, query_neurons, query_bids, tgt_bids,
            null_k=self.null_k, project_root=self.project_root,
            vector_cache=vector_cache, side_cache=side_cache,
            log=self._log)
        del sample
        pair_info['baseline'] = stats

        # Score only the candidate targets here; the null rows were already
        # computed inside null_baselines against the shared vector cache.
        df = score_pairs(source, target, query_neurons, query_bids,
                         tgt_bids, project_root=self.project_root,
                         vector_cache=vector_cache, side_cache=side_cache)
        if store is not None:
            store.update(vector_cache, side_cache)
        rows = []
        if df is not None and not df.empty:
            for _idx, row in df.iterrows():
                try:
                    src, tgt = int(row['source_bodyId']), \
                        int(row['target_bodyId'])
                except (TypeError, ValueError):
                    continue
                val = row.get('morph_v2_similarity')
                if val is None or pd.isna(val):
                    continue
                stats_for_src = stats.get(src) or {}
                bar = (stats_for_src.get('p95')
                       if stats_for_src else None)
                rows.append({
                    'source_type': src_bid_type.get(src, ''),
                    'source_bodyId': src,
                    'target_type': bid_to_type.get(tgt, ''),
                    'target_bodyId': tgt,
                    'morph_v2': float(val),
                    'null_p95': bar,
                    'above_baseline': (bool(float(val) >= bar)
                                       if bar is not None else None),
                })
        pair_info['rows'] = pd.DataFrame(rows)
        type_matrix = pd.DataFrame(index=sorted(src_members),
                                   columns=sorted(tgt_members),
                                   dtype=float)
        if rows:
            rdf = pair_info['rows']
            for src_type in type_matrix.index:
                for tgt_type in type_matrix.columns:
                    sel = rdf[(rdf['source_type'] == src_type)
                              & (rdf['target_type'] == tgt_type)]
                    type_matrix.loc[src_type, tgt_type] = (
                        round(float(sel['morph_v2'].mean()), 4)
                        if not sel.empty else np.nan)
        pair_info['type_matrix'] = type_matrix
        pair_info['bodyid_matrix'] = self._member_matrix(
            pair_info['rows'], source, src_members, target, tgt_members)
        self._log(f'[{source} -> {target}] scored {len(rows)} pair(s); '
                  f'baseline p95 range: '
                  f'{min((s["p95"] for s in stats.values()), default=float("nan")):.3f}'
                  f'..{max((s["p95"] for s in stats.values()), default=float("nan")):.3f}')
        return pair_info

    def _member_matrix(self, rows: pd.DataFrame, source: str,
                       src_members: Dict[str, List[int]], target: str,
                       tgt_members: Dict[str, List[int]]) -> pd.DataFrame:
        """BodyId x bodyId morph_v2 matrix for one dataset pair.

        The bodyId-level counterpart of ``type_matrix``: rows are the source
        members, columns the target members, values the raw vector_v2 score
        (NaN where a pair was unscoreable). Axes use the tree-legend labels
        ('{bodyId}_{instance}' or '{bodyId}_{type}_{L|R}'), ordered by type
        then bodyId like the connectivity-profiler's bodyId matrices.
        """
        if rows is None or rows.empty:
            return pd.DataFrame()

        def axis(dataset: str, members: Dict[str, List[int]]
                 ) -> Tuple[List[int], List[str]]:
            bids: List[int] = []
            labels: List[str] = []
            for type_name in sorted(members):
                for bid in sorted(members[type_name]):
                    bid = int(bid)
                    bids.append(bid)
                    labels.append(_body_id_label(dataset, bid, type_name))
            return bids, labels

        src_bids, src_labels = axis(source, src_members)
        tgt_bids, tgt_labels = axis(target, tgt_members)
        src_map = dict(zip(src_bids, src_labels))
        tgt_map = dict(zip(tgt_bids, tgt_labels))
        long = rows.drop_duplicates(['source_bodyId', 'target_bodyId'])
        long = long.assign(
            _src=long['source_bodyId'].map(src_map),
            _tgt=long['target_bodyId'].map(tgt_map))
        matrix = long.pivot(index='_src', columns='_tgt',
                            values='morph_v2')
        return matrix.reindex(index=src_labels, columns=tgt_labels)

    def _overview_frame(self, tokens: Sequence[str], datasets: List[str],
                        members: Dict[str, Dict[str, List[int]]],
                        pairs: Dict[Tuple[str, str], Dict[str, Any]]
                        ) -> pd.DataFrame:
        queried_types: List[str] = []
        for name in datasets:
            for type_name in (members.get(name) or {}):
                if type_name not in queried_types:
                    queried_types.append(type_name)
        # One column block per dataset pair: the queried type's BEST
        # target-type cell (its mean over the member pairs) with the target
        # type's name, plus the pair's baseline. A row mean over all target
        # types would dilute the signal with unrelated cells.
        overview = pd.DataFrame({'queried_type': queried_types})
        for (a, b), info in pairs.items():
            col = (f'{self._abbrevs.get(a, _dataset_abbrev(a))}→'
                   f'{self._abbrevs.get(b, _dataset_abbrev(b))}')
            best, best_tgt = {}, {}
            matrix = info.get('type_matrix')
            if matrix is not None and not matrix.empty:
                for src_type in matrix.index:
                    row = matrix.loc[src_type].dropna()
                    if len(row):
                        tgt_type = row.astype(float).idxmax()
                        best[src_type] = round(float(row[tgt_type]), 4)
                        best_tgt[src_type] = tgt_type
            overview[col] = [best.get(t, np.nan) for t in queried_types]
            overview[f'{col} best_target'] = [
                best_tgt.get(t, '') for t in queried_types]
            baselines = [s['p95'] for s in
                         (info.get('baseline') or {}).values()]
            overview[f'{col} baseline_p95'] = [
                round(float(np.mean(baselines)), 4) if baselines else np.nan
            ] * len(queried_types)
        return overview

    # --------------------------------------------------------------- output
    def _write_outputs(self, run_path: Path, tokens: Sequence[str],
                       datasets: List[str],
                       members: Dict[str, Dict[str, List[int]]],
                       pairs: Dict[Tuple[str, str], Dict[str, Any]],
                       overview: pd.DataFrame, notes: List[str]
                       ) -> List[Path]:
        files: List[Path] = []
        for (a, b), info in pairs.items():
            pair_dir = (run_path / f'{self._abbrevs.get(a, _dataset_abbrev(a))}'
                        f'_to_{self._abbrevs.get(b, _dataset_abbrev(b))}'
                        / 'results')
            pair_dir.mkdir(parents=True, exist_ok=True)
            rows = info.get('rows')
            if rows is not None and not rows.empty:
                csv = pair_dir / 'morph_bodyid_scores.csv'
                rows.sort_values(['source_type', 'morph_v2'],
                                 ascending=[True, False],
                                 na_position='last').to_csv(csv, index=False)
            matrix = info.get('type_matrix')
            if matrix is not None and not matrix.empty:
                csv = pair_dir / 'morph_type_matrix.csv'
                matrix.to_csv(csv)
            bodyid = info.get('bodyid_matrix')
            if bodyid is not None and not bodyid.empty:
                csv = pair_dir / 'morph_bodyid_matrix.csv'
                bodyid.to_csv(csv)
            baseline = {'source': a, 'target': b,
                        'null_k': self.null_k,
                        'per_query': {str(k): v for k, v in
                                      (info.get('baseline') or {}).items()},
                        }
            (pair_dir / 'null_baseline.json').write_text(
                json.dumps(baseline, indent=2), encoding='utf-8')
            for path in pair_dir.iterdir():
                if path.is_file():
                    files.append(path)
        if not overview.empty:
            csv = run_path / 'overview.csv'
            overview.to_csv(csv, index=False)
            files.append(csv)
        member_rows = []
        for type_name in self._queried_type_order(datasets, members):
            row = {'queried_type': type_name}
            for name in datasets:
                row[self._abbrevs.get(name, _dataset_abbrev(name))] = len(
                    (members.get(name) or {}).get(type_name) or [])
            member_rows.append(row)
        if member_rows:
            csv = run_path / 'members_summary.csv'
            pd.DataFrame(member_rows).to_csv(csv, index=False)
            files.append(csv)
        parameters = {
            'queries': list(tokens),
            'datasets': list(datasets),
            'max_members_per_type': self.max_members_per_type,
            'null_k': self.null_k,
            'reference_template': self.reference_template,
            'scene_members_per_type': self.scene_members_per_type,
            'visualize': self.visualize,
            'generate_heatmaps': self.generate_heatmaps,
            'scoring': 'vector_v2 (block-weighted whitened cosine; '
                       'no NBLAST cross-dataset)',
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'warnings': list(self.warnings),
            'resolution_notes': list(notes),
        }
        (run_path / 'parameters.json').write_text(
            json.dumps(parameters, indent=2), encoding='utf-8')
        files.append(run_path / 'parameters.json')
        (run_path / 'README.txt').write_text(self._readme_text(
            tokens, datasets, pairs, notes), encoding='utf-8')
        files.append(run_path / 'README.txt')
        return files

    def _readme_text(self, tokens, datasets, pairs, notes) -> str:
        lines = [
            '=' * 70,
            '  CROSS-DATASET MORPHOLOGY COMPARISON',
            '=' * 70,
            '',
            f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
            f'Queries: {", ".join(tokens)}',
            f'Datasets: {", ".join(datasets)}',
            f'Scoring: vector_v2 in each target render space '
            f'(no NBLAST cross-dataset)',
            '',
            '  OUTPUT STRUCTURE',
            '  <SRC>_to_<TGT>/            per dataset pair',
            '    ├── results/',
            '    │   ├── morph_type_matrix.csv      type x type mean matrix',
            '    │   ├── morph_bodyid_matrix.csv    bodyId x bodyId vector_v2',
            '    │   ├── morph_bodyid_scores.csv    long-form scores + null flag',
            '    │   └── null_baseline.json         seeded null (p95/median)',
            '    └── visualization/',
            '        └── heatmap_morph_<SRC>_to_<TGT>_{level}.html',
            '                                    interactive VisPath heatmaps',
            '  plot-3d_*/                 overlay scenes (reference template)',
            '  report.html                tabbed report (pair tabs, Type/BodyId',
            '                             levels, Ward-clustered heatmaps)',
            '  overview.csv               queried type x pair: best target cell',
            '  members_summary.csv        compared member counts per type/dataset',
            '',
        ]
        if self.warnings:
            lines.append('  WARNINGS')
            lines.extend(f'  - {w}' for w in self.warnings)
            lines.append('')
        if notes:
            lines.append('  RESOLUTION NOTES')
            lines.extend(f'  - {n}' for n in notes)
        return '\n'.join(lines) + '\n'

    # ------------------------------------------------------------- scenes
    def _pair_slug(self, a: str, b: str) -> str:
        return (f'{self._abbrevs.get(a, _dataset_abbrev(a))}'
                f'_to_{self._abbrevs.get(b, _dataset_abbrev(b))}')

    def _write_heatmaps(self, run_path: Path,
                        pairs: Dict[Tuple[str, str], Dict[str, Any]]
                        ) -> List[Path]:
        """Standalone interactive VisPath heatmaps per pair (type + bodyId).

        Same writer as the connectivity-profiling export (shared report_kit):
        Ward-clustered, diverging vector_v2 scale, interactive-heatmap
        fallback when VisPath is unavailable. One kit call per (pair,
        level): the kit's card keys are METRIC keys, so the level rides in
        the closures while every call uses the ``morph_v2`` style.
        """
        styles = {'morph_v2': _MORPH_V2_STYLE}
        saved: Dict[str, List[str]] = {'heatmaps_generated': []}
        for (a, b), info in pairs.items():
            if info.get('error'):
                continue
            slug = self._pair_slug(a, b)
            pair_title = slug.replace('_to_', ' → ')
            for level in ('type', 'bodyid'):
                matrix = (info.get('type_matrix') if level == 'type'
                          else info.get('bodyid_matrix'))
                if matrix is None or matrix.empty:
                    continue
                report_kit.generate_standalone_heatmaps(
                    {slug: {'morph_v2': matrix}},
                    run_path / slug / 'visualization', styles,
                    filename_builder=lambda group, key, _lv=level:
                        f'heatmap_morph_{group}_{_lv}.html',
                    group_display=lambda group: group.replace('_to_', ' → '),
                    vispath_title=lambda group, key, gd, _lv=level:
                        f'{_MORPH_V2_STYLE.display_name} — {gd} · '
                        f'{_LEVEL_DISPLAY.get(_lv, _lv)}',
                    fallback_title=lambda group, key, gd, _lv=level:
                        f'Cross-Dataset Morphology - {gd} - '
                        f'{_LEVEL_DISPLAY.get(_lv, _lv)}',
                    tqdm_desc=f'Generating {pair_title} '
                              f'{_LEVEL_DISPLAY.get(level, level)} heatmap',
                    show_figures=False, verbose=self.verbose,
                    saved_files=saved, log=self._log)
        return [Path(p) for p in saved['heatmaps_generated']]

    def _render_scenes(self, run_path: Path, datasets: List[str],
                       members: Dict[str, Dict[str, List[int]]]
                       ) -> List[Path]:
        """One overlay scene per run: a layer per (queried type, dataset).

        Same-name layers from two datasets sit side by side, and so do
        renamed homolog pairs (APDN3@FAFB next to SLP249@MCNS).
        Reference-template layers load natively; every other dataset's
        members are bridged into the reference render space.
        """
        from visualize_skeleton import VisualizeSkeleton, \
            dataset_native_space, dataset_render_space, \
            transform_neurons_to_space
        import navis

        reference = self.reference_template or datasets[0]
        if reference not in datasets:
            reference = datasets[0]
        ref_render = dataset_render_space(reference)
        neuron_layers: List[List[int]] = []
        layer_names: List[str] = []
        custom_neurons: List[Tuple[str, List[Any]]] = []
        for type_name in self._queried_type_order(datasets, members):
            for ds in datasets:
                bids = (members.get(ds) or {}).get(type_name) or []
                shown = bids[:self.scene_members_per_type]
                if not shown:
                    continue
                abbrev = self._abbrevs.get(ds, _dataset_abbrev(ds))
                label = f'{_safe_name(type_name, 28)}@{abbrev}_x{len(shown)}'
                if ds == reference:
                    neuron_layers.append([int(b) for b in shown])
                    layer_names.append(label)
                    continue
                skeletons = fetch_source_skeletons(
                    ds, shown, project_root=self.project_root, log=self._log,
                    allow_fetch=self.fetch_online)
                # Local only — the `members` PARAMETER (the per-dataset
                # type map) must survive for the next dataset/type
                # iteration; rebinding it crashed every run with >= 3
                # datasets or >= 2 queried types, silently dropping all
                # overlay scenes.
                scene_members = [(int(b), skeletons[int(b)]) for b in shown
                                 if int(b) in skeletons]
                if not scene_members:
                    continue
                neurons = _tag_scene_members(scene_members, type_name, abbrev, ds)
                native = dataset_native_space(ds)
                if native == ref_render:
                    custom_neurons.append((label, neurons))
                    continue
                xf = transform_neurons_to_space(
                    navis.NeuronList(neurons), native, ref_render,
                    validate_bounds=True, verbose=False)
                if xf:
                    custom_neurons.append((label, list(xf)))
        if not neuron_layers and not custom_neurons:
            self._log('3D visualization skipped: no compared neurons.')
            return []

        viz_kwargs: Dict[str, Any] = {
            # Empty lists (never None): VisualizeSkeleton inserts the
            # custom overlay layers into neuron_layers in place.
            'dataset': reference,
            'output_dir': str(run_path),
            'neuron_layers': neuron_layers,
            'custom_layer_names': layer_names,
            'custom_neurons': custom_neurons,
            'saveas': 'crossmorph_' + _safe_name('_'.join(
                self._query_tokens()), 40),
            'include_timestamp': False,
            'skip_synapse': True,
            'skeleton_mode': 'line',
            'legend_mode': 'tree',
            'brain_mesh': 'native',
            'export_views': False,
            'show_fig': False,
            # Unified cache-first behavior: render reference layers from
            # the shared raw cache; only true misses go online.
            'cache_neurons': True,
            'verbose': 'simple',
        }
        files: List[Path] = []
        # VisualizeSkeleton may take ownership of (and mutate) the layer
        # lists — capture the counts for the log before plot_neurons().
        n_native, n_bridged = len(neuron_layers), len(custom_neurons)
        try:
            vs = VisualizeSkeleton(**viz_kwargs)
            vs.plot_neurons()
            self._log(f'3D overlay scene rendered: {n_native} '
                      f'native + {n_bridged} bridged layer(s) in '
                      f'the {reference} render space')
        except Exception as exc:  # noqa: BLE001
            self._log(f'3D visualization failed (comparison kept): {exc}')
            return []
        for folder in sorted(run_path.glob('plot-3d_*'),
                             key=lambda p: p.stat().st_mtime,
                             reverse=True):
            files.extend(sorted(folder.glob('*.html')))
        return files

    def _queried_type_order(self, datasets: List[str],
                            members: Dict[str, Dict[str, List[int]]]
                            ) -> List[str]:
        order: List[str] = []
        for name in datasets:
            for type_name in (members.get(name) or {}):
                if type_name not in order:
                    order.append(type_name)
        return order

    # -------------------------------------------------------------- report
    def _write_report(self, run_path: Path, tokens: Sequence[str],
                      datasets: List[str],
                      members: Dict[str, Dict[str, List[int]]],
                      pairs: Dict[Tuple[str, str], Dict[str, Any]],
                      overview: pd.DataFrame, notes: List[str],
                      scenes: List[Path],
                      existing: Optional[List[Path]] = None) -> List[Path]:
        """Tabbed report on the shared report_kit (same format as the
        connectivity-profiling export): hero header, a horizontally
        scrollable overview table with a frame-asymmetry disclosure, one
        tab per dataset pair, Type/BodyId level tabs, Ward-clustered
        heatmaps (square cells via explicit-width sizing for small
        matrices) with CSV and VisPath editor links, member/resolution
        details, overlay scenes.  Plotly.js is embedded, so the report
        renders offline.
        """
        from html import escape

        def chip(label: str, value: str) -> str:
            return (f"<div class='meta-chip'><span>{escape(label)}</span>"
                    f"<strong>{escape(value)}</strong></div>")

        reference = self.reference_template or datasets[0]
        lines = [
            '<!DOCTYPE html>',
            "<html><head><meta charset='utf-8'>",
            '<title>Cross-Dataset Morphology Comparison</title>',
            report_kit.report_css(),
            '</head><body><main class="report-shell">',
            '<header class="report-hero">',
            '<div class="report-kicker">DROCAT · Cross-dataset morphology</div>',
            '<h1 class="report-title">Cross-dataset morphology comparison</h1>',
            '<p class="report-subtitle">Production vector_v2 scored in each '
            'target render space (no NBLAST cross-dataset). Scores are '
            'comparable within a pair tab, not across tabs: each dataset '
            'pair is scored in its target frame. Use the VisPath editor '
            'links to change clustering; hover cells for exact values.</p>',
            '<div class="report-meta">',
            chip('Datasets', ' · '.join(datasets)),
            chip('Queries', ', '.join(tokens)),
            chip('Null k', str(self.null_k)),
            chip('Reference template', str(reference)),
            '</div>',
            f"<p class='report-note'>Generated {datetime.now():%Y-%m-%d %H:%M:%S} · "
            'negative vector_v2 cells are genuinely negative (whitened '
            'cosine) and render blue on the diverging scale.</p>',
            '</header>',
        ]
        if self.warnings:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">Warnings</h2><ul>'
                + ''.join(f'<li>{escape(w)}</li>' for w in self.warnings)
                + '</ul></div>')

        if not overview.empty:
            # The overview frame carries one column block per directed pair
            # (3 sub-columns each), so the table grows quadratically with
            # the dataset count and can overflow the section-card. Wrap it
            # in a horizontally scrollable container (mirrors the members
            # details block below) and disclose the frame asymmetry so
            # readers don't compare A→B and B→A scores as if they were
            # mirrored measurements of the same quantity.
            lines.append('<div class="section-card">'
                         '<h2 class="section-heading">Overview — best '
                         'target-type match per queried type</h2>'
                         '<p class="section-summary">Each column block is '
                         'scored in its TARGET dataset\u2019s render space '
                         'with the target\u2019s vector_v2 whitening basis, '
                         'so A\u2192B and B\u2192A are not symmetric by '
                         'construction \u2014 they measure morphological '
                         'similarity in two different coordinate frames. '
                         'Baseline p95 also shifts with the basis; compare '
                         'a score only against its own column\u2019s '
                         'baseline.</p>')
            lines.append('<div style="overflow-x:auto">')
            lines.append(overview.to_html(index=False, na_rep='\u2014', border=0,
                                          classes='mapping-table overview-table'))
            lines.append('</div>')
            lines.append('</div>')

        plotly_state = {'include_plotlyjs': True}

        def render_level(pair_dir: str, info: Dict[str, Any],
                         level: str, _level_panel_id: str) -> None:
            level_label = _LEVEL_DISPLAY[level]
            matrix = (info.get('type_matrix') if level == 'type'
                      else info.get('bodyid_matrix'))
            matrix = matrix if matrix is not None else pd.DataFrame()
            if matrix.empty:
                lines.append(
                    "<div class='heatmap-empty'>This level was not computed "
                    'for this pair.</div>')
                return
            rows = info.get('rows')
            n_above = (int(rows['above_baseline'].fillna(False).sum())
                       if rows is not None and not rows.empty else 0)
            n_pairs = len(rows) if rows is not None and not rows.empty else 0
            baseline = info.get('baseline') or {}
            note = (f'{n_pairs} member pairs · {n_above} above the null p95'
                    if rows is not None and not rows.empty else
                    'no scored member pairs')
            if baseline:
                p95s = [s['p95'] for s in baseline.values()]
                note += (f' · null baseline p95 {min(p95s):.3f}–{max(p95s):.3f}'
                         f' (seeded random targets, k≈{self.null_k})')
            lines.append(
                f"<div class='direction-intro'><h3 class='direction-title'>"
                f'{escape(level_label)}</h3>'
                f"<div class='direction-note'>{escape(note)}</div></div>")
            report_kit.append_report_metric_grid(
                lines, run_path, {'morph_v2': matrix},
                f'{pair_dir.replace("_to_", " → ")} · {level_label}',
                {'morph_v2': f'{pair_dir}/results/'
                                  f'morph_{level}_matrix.csv'},
                {'morph_v2': f'{pair_dir}/visualization/'
                                  f'heatmap_morph_{pair_dir}_{level}.html'},
                'Target neuron' if level == 'bodyid' else 'Target type',
                'Source neuron' if level == 'bodyid' else 'Source type',
                plotly_state,
                styles={'morph_v2': _MORPH_V2_STYLE},
                square_cells=True,
            )

        def render_pair(key: str, panel_id: str) -> None:
            info = self._pair_by_slug.get(key) or {}
            a, b = self._slug_pairs[key]
            pair_title = key.replace('_to_', ' → ')
            lines.append(
                '<div class="section-card">'
                f"<h2 class='section-heading'>{escape(pair_title)}</h2>"
                '<p class="section-summary">vector_v2 in the '
                f'{escape(str(b))} render space; the mirrored direction '
                'has its own tab.</p>')
            if info.get('error'):
                lines.append(
                    f"<p class='heatmap-empty'>{escape(str(info['error']))}</p>"
                    '</div>')
                return
            report_kit.append_report_tab_group(
                lines, f'{panel_id}-levels',
                [(level, _LEVEL_DISPLAY[level])
                 for level in ('type', 'bodyid')],
                lambda level, level_pid, _d=key, _i=info:
                    render_level(_d, _i, level, level_pid),
                panel_class='tab-panel direction-panel')
            lines.append('</div>')

        self._pair_by_slug: Dict[str, Dict[str, Any]] = {}
        self._slug_pairs: Dict[str, Tuple[str, str]] = {}
        pair_tabs = []
        for (a, b) in pairs:
            slug = self._pair_slug(a, b)
            self._pair_by_slug[slug] = pairs[(a, b)]
            self._slug_pairs[slug] = (a, b)
            pair_tabs.append((slug, slug.replace('_to_', ' → ')))
        if pair_tabs:
            report_kit.append_report_tab_group(
                lines, 'morph-pairs', pair_tabs, render_pair)
        else:
            lines.append("<div class='heatmap-empty'>No dataset pairs "
                         'scored for this run.</div>')

        # --- compared members + resolution notes -------------------------
        detail_bits = []
        if any((members.get(ds) or {}) for ds in datasets):
            rows = ['<tr><th>queried type</th>'
                    + ''.join(f'<th>{escape(self._abbrevs.get(ds, _dataset_abbrev(ds)))}</th>'
                              for ds in datasets)
                    + '</tr>']
            for type_name in self._queried_type_order(datasets, members):
                rows.append(
                    f'<tr><td>{escape(type_name)}</td>'
                    + ''.join(
                        f"<td>{len((members.get(ds) or {}).get(type_name) or [])}</td>"
                        for ds in datasets)
                    + '</tr>')
            detail_bits.append(
                '<details class="detail-block"><summary>Compared members '
                'per dataset</summary><div style="overflow-x:auto">'
                "<table class='mapping-table'><thead>" + ''.join(rows)
                + '</thead></table></div></details>')
        if notes:
            detail_bits.append(
                '<details class="detail-block"><summary>Resolution '
                'notes</summary><ul class="detail-list">'
                + ''.join(f'<li>{escape(n)}</li>' for n in notes)
                + '</ul></details>')
        if detail_bits:
            lines.append('<div class="section-card">' + ''.join(detail_bits)
                         + '</div>')

        if scenes:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">'
                'Overlay scenes</h2><ul class="detail-list">'
                + ''.join(f"<li><a href='{escape(p.parent.name)}/"
                          f"{escape(p.name)}'>"
                          f'{escape(str(Path(p.parent.name) / p.name))}</a></li>'
                          for p in scenes)
                + f'</ul><p class="muted">Scene frame: {escape(reference)} '
                'render space — different frame from the scores (frame '
                'disclosure).</p></div>')

        lines.extend(['</main>', report_kit.report_script(),
                      '</body></html>'])
        report = run_path / 'report.html'
        report.write_text('\n'.join(lines), encoding='utf-8')
        files = list(existing or [])
        files.append(report)
        self._log(f'report written: {report}')
        return files
