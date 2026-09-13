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
NULL_MIN_N = 10
BAR_OFFSET_MAX = 0.2
MAX_TRANSFORM_HOPS = 2
BOOTSTRAP_SAMPLE_DEFAULT = 600
BOOTSTRAP_MIN_ROWS = 24

try:  # package import (runner scripts / tests)
    from .cross_dataset_type_mapper import get_type_mapper
    from .type_resolver import (MapperSnapshot, expansion_targets,
                                resolve_valid_targets)
except ImportError:  # direct src/ execution
    try:
        from cross_dataset_type_mapper import get_type_mapper
        from type_resolver import (MapperSnapshot, expansion_targets,
                                   resolve_valid_targets)
    except ImportError:  # mapper optional: feature degrades to same-name
        get_type_mapper = None
        MapperSnapshot = None
        resolve_valid_targets = None
        expansion_targets = None

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


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
    """
    bids = [int(b) for b in dict.fromkeys(int(b) for b in body_ids or [])]
    if not bids:
        return {}
    root = str(Path(project_root or DEFAULT_PROJECT_ROOT))
    try:
        from morphology import load_local_release_skeletons
        if allow_fetch and _dataset_family(dataset) in ('FAFB', 'BANC'):
            got = load_local_release_skeletons(dataset, bids, project_root=root,
                                               log=None)
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
                verbose: bool = False) -> pd.DataFrame:
    """Production vector_v2 scores for (query, target) pairs — the shared
    scoring path. Thin wrapper over
    ``morphology.compute_morph_similarity_vs_queries(compute_nblast=False)``
    with an optional persistent target-vector cache."""
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


class NullVectorStore:
    """Persistent render-space vectors for the shared null sample.

    A small npz sidecar per (dataset, render space) so cross-space targets
    (male-cns) do not re-transform + re-vectorize the same K neurons on
    every run. Invalidated when the population bounds change.
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None):
        self.dataset = dataset
        self.root = Path(project_root or DEFAULT_PROJECT_ROOT)
        folder = str(dataset).replace(':', '_').replace('.', '_')
        from morphology import _dataset_folder
        self.dir = (self.root / 'cache' / _dataset_folder(dataset)
                    / 'find_similar' / 'morphology')
        del folder
        try:
            from visualize_skeleton import dataset_render_space
            self.space = dataset_render_space(dataset)
        except Exception:  # noqa: BLE001
            self.space = 'native'
        self.path = self.dir / f'cross_dataset_nullvec_{self.space}.npz'

    def _bounds_signature(self) -> str:
        bounds = _scoring_bounds(self.dataset, str(self.root))
        if bounds is None:
            return 'none'
        return hashlib.sha1(np.asarray(bounds, dtype=float).tobytes()
                            ).hexdigest()[:16]

    def load(self) -> Dict[int, Any]:
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if str(z['space']) != self.space:
                    return {}
                if str(z['bounds_sig']) != self._bounds_signature():
                    return {}
                bids = np.asarray(z['bodyIds'])
                mat = np.asarray(z['matrix'], dtype=float)
                return {int(b): mat[i] for i, b in enumerate(bids)}
        except Exception:  # noqa: BLE001
            return {}

    def update(self, vectors: Dict[int, Any]) -> None:
        rows = {int(b): np.asarray(v, dtype=float)
                for b, v in (vectors or {}).items()
                if v is not None and len(v)}
        if len(rows) < 8:
            return  # not worth a file
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            bids = np.array(sorted(rows), dtype=np.int64)
            mat = np.vstack([rows[int(b)] for b in bids]).astype(np.float32)
            tmp = self.path.with_suffix('.npz.tmp')
            with open(tmp, 'wb') as fh:
                np.savez(fh, bodyIds=bids, matrix=mat,
                         space=np.array(self.space),
                         bounds_sig=np.array(self._bounds_signature()))
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001
            pass


def null_baselines(source_dataset: str, target_dataset: str,
                   query_neurons: Sequence[Any], query_bids: Sequence[int],
                   candidate_bids: Sequence[int], null_k: int = NULL_K_DEFAULT,
                   project_root: Optional[str] = None,
                   vector_cache: Optional[Dict[int, Any]] = None,
                   log=None) -> Tuple[Dict[int, Dict[str, float]], List[int]]:
    """Score the shared null sample against every query; per-query stats.

    Returns ``({source_bid: {p95, median, std, mean, n}}, sample_bids)``.
    The null sample is deterministic per (dataset, k, candidates) and its
    vectors persist in the NullVectorStore sidecar.
    """
    sample = null_sample(target_dataset, k=null_k,
                         exclude=candidate_bids, project_root=project_root)
    stats: Dict[int, Dict[str, float]] = {}
    if not sample:
        return stats, []
    if vector_cache is None:
        vector_cache = {}
    store = NullVectorStore(target_dataset, project_root)
    if not vector_cache:
        vector_cache.update(store.load())
    df = score_pairs(source_dataset, target_dataset, query_neurons,
                     query_bids, sample, project_root=project_root,
                     vector_cache=vector_cache)
    store.update(vector_cache)
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
    """Result of the pooled morph-qualification pass for one run."""

    source_dataset: str = ''
    target_dataset: str = ''
    null_k: int = NULL_K_DEFAULT
    bar_offset: float = 0.0
    scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    null_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    active: bool = False

    def bar(self, source_bid: int) -> Optional[float]:
        stats = self.null_stats.get(int(source_bid))
        if not stats:
            return None
        return stats['p95'] + float(self.bar_offset)

    def is_qualified(self, source_bid: int, target_bid: int) -> Optional[bool]:
        score = self.scores.get((int(source_bid), int(target_bid)))
        if score is None or not np.isfinite(score):
            return None
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
                            top_n: int) -> List[Tuple[int, int]]:
    """(source_bodyId, target_bodyId) pairs the visualization would render.

    Mirrors ``_visualize_homolog_candidates``' bodyId-level selection:
    per-source rank_union top-N, then a first-wins dedupe on the target
    across sources. Keeping the two in lockstep is what makes the pooled
    pre-scoring cover exactly the rendered set.
    """
    if results_df is None or results_df.empty:
        return []
    if 'target_bodyId' not in results_df.columns:
        return []
    candidate = results_df
    if top_n and top_n > 0:
        if 'source_bodyId' in candidate.columns:
            per_source = []
            for _key, group in candidate.groupby('source_bodyId'):
                if 'rank_union' in group.columns:
                    group = group.sort_values('rank_union', ascending=False,
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


def qualify_visualized_pairs(
        source_dataset: str, target_dataset: str,
        pairs: Sequence[Tuple[int, int]],
        null_k: int = NULL_K_DEFAULT,
        project_root: Optional[str] = None,
        bar_offset: float = 0.0,
        log=None) -> MorphQualification:
    """Score the visualized (source, target) pairs + shared null sample.

    One scoring call covers the union of candidates and the null sample so
    target transforms/vectorizations are computed once. Raises ValueError
    on scope violations; returns an inactive MorphQualification (with a
    warning) when scoring yields nothing usable.
    """
    mq = MorphQualification(source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            null_k=int(null_k), bar_offset=float(bar_offset))
    check = check_pair_availability(source_dataset, target_dataset)
    if not check['ok']:
        raise ValueError(check['reason'])
    scope = dataset_scope(target_dataset)
    mq.warnings.extend(scope['warnings'])

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
    missing_targets = [b for b in unique_targets
                       if b not in _skeleton_cached_ids(
                           target_dataset, project_root)]
    if missing_targets:
        _logline(log, f'[morph-qualify] fetching {len(missing_targets)} '
                      'target skeleton(s) not locally cached')
        fetch_source_skeletons(target_dataset, missing_targets,
                               project_root=project_root, log=log)

    vector_cache: Dict[int, Any] = {}
    stats, sample = null_baselines(
        source_dataset, target_dataset, neurons, bids, unique_targets,
        null_k=null_k, project_root=project_root, vector_cache=vector_cache,
        log=log)
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
                     vector_cache=vector_cache)
    if df is None or df.empty:
        mq.warnings.append('Candidate scoring produced no rows; '
                           'qualification inactive.')
        return mq
    sample_set = set(int(b) for b in sample)
    for _idx, row in df.iterrows():
        try:
            src, tgt = int(row['source_bodyId']), int(row['target_bodyId'])
        except (TypeError, ValueError):
            continue
        if tgt in sample_set:
            continue
        val = row.get('morph_v2_similarity')
        if val is None or pd.isna(val):
            continue
        mq.scores[(src, tgt)] = float(val)

    for (src, tgt), score in list(mq.scores.items()):
        if mq.is_qualified(src, tgt) is None:
            # No null bar for this source (unscored null): drop the pair.
            del mq.scores[(src, tgt)]
    n_qualified = sum(1 for pair in mq.scores if mq.is_qualified(*pair))
    mq.active = bool(mq.scores)
    _logline(log, f'[morph-qualify] scored {len(mq.scores)} visualized '
             f'pair(s); {n_qualified} above the null p95 bar '
             f'(offset {mq.bar_offset:+.2f}); null n='
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

    # Pass 2: vectorize with the cache's own recipe (relevel + v2 features).
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
    render space, a seeded null baseline per pair, overlay scenes in a
    chosen reference template, and a self-contained report.html.
    """

    def __init__(self, datasets: Optional[List[str]] = None,
                 query: Any = None, output_dir: Optional[str] = None,
                 saveas: Optional[str] = None,
                 max_members_per_type: int = 25,
                 null_k: int = NULL_K_DEFAULT,
                 reference_template: Optional[str] = None,
                 scene_members_per_type: int = 3,
                 visualize: bool = True,
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
        names = list(expansion_targets(resolution))             if expansion_targets is not None else []
        if not names:
            # claimed / evidence_only: a verdict the target dataset cannot
            # fulfil — no expansion, explicit note (not silently empty).
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
        scenes = []
        if self.visualize:
            try:
                scenes = self._render_scenes(run_path, datasets, members)
                files.extend(scenes)
            except Exception as exc:  # noqa: BLE001
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
        candidate_set = set(tgt_bids)

        # The scorer never fetches TARGET skeletons on demand; without a
        # pre-fetch, members outside the local cache would silently drop
        # out of the matrix (NaN cells).
        if self.fetch_online:
            missing = [b for b in tgt_bids
                       if b not in _skeleton_cached_ids(
                           target, self.project_root)]
            if missing and self.fetch_online:
                self._log(f'[{target}] fetching {len(missing)} target '
                          'skeleton(s) not locally cached')
                fetch_source_skeletons(target, missing,
                                       project_root=self.project_root,
                                       log=self._log)

        vector_cache: Dict[int, Any] = {}
        if self.use_cache:
            vector_cache.update(NullVectorStore(
                target, self.project_root).load())
        stats, sample = null_baselines(
            source, target, query_neurons, query_bids, tgt_bids,
            null_k=self.null_k, project_root=self.project_root,
            vector_cache=vector_cache, log=self._log)
        del sample
        pair_info['baseline'] = stats

        # Score only the candidate targets here; the null rows were already
        # computed inside null_baselines against the shared vector cache.
        df = score_pairs(source, target, query_neurons, query_bids,
                         tgt_bids, project_root=self.project_root,
                         vector_cache=vector_cache)
        if self.use_cache:
            NullVectorStore(target, self.project_root).update(vector_cache)
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
        self._log(f'[{source} -> {target}] scored {len(rows)} pair(s); '
                  f'baseline p95 range: '
                  f'{min((s["p95"] for s in stats.values()), default=float("nan")):.3f}'
                  f'..{max((s["p95"] for s in stats.values()), default=float("nan")):.3f}')
        return pair_info

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
                        f'_to_{self._abbrevs.get(b, _dataset_abbrev(b))}')
            pair_dir.mkdir(parents=True, exist_ok=True)
            rows = info.get('rows')
            if rows is not None and not rows.empty:
                csv = pair_dir / 'bodyid_scores.csv'
                rows.sort_values(['source_type', 'morph_v2'],
                                 ascending=[True, False],
                                 na_position='last').to_csv(csv, index=False)
            matrix = info.get('type_matrix')
            if matrix is not None and not matrix.empty:
                csv = pair_dir / 'type_matrix.csv'
                matrix.to_csv(csv)
            baseline = {'source': a, 'target': b,
                        'null_k': self.null_k,
                        'per_query': {str(k): v for k, v in
                                      (info.get('baseline') or {}).items()},
                        }
            (pair_dir / 'null_baseline.json').write_text(
                json.dumps(baseline, indent=2))
            for path in pair_dir.iterdir():
                if path.is_file():
                    files.append(path)
        if not overview.empty:
            csv = run_path / 'overview.csv'
            overview.to_csv(csv, index=False)
            files.append(csv)
        parameters = {
            'queries': list(tokens),
            'datasets': list(datasets),
            'max_members_per_type': self.max_members_per_type,
            'null_k': self.null_k,
            'reference_template': self.reference_template,
            'scene_members_per_type': self.scene_members_per_type,
            'visualize': self.visualize,
            'scoring': 'vector_v2 (block-weighted whitened cosine; '
                       'no NBLAST cross-dataset)',
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'warnings': list(self.warnings),
            'resolution_notes': list(notes),
        }
        (run_path / 'parameters.json').write_text(
            json.dumps(parameters, indent=2))
        files.append(run_path / 'parameters.json')
        (run_path / 'README.txt').write_text(self._readme_text(
            tokens, datasets, pairs, notes))
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
            '    ├── bodyid_scores.csv    member-level vector_v2 + baseline flag',
            '    ├── type_matrix.csv      type x type mean matrix',
            '    └── null_baseline.json   seeded null reference (p95/median)',
            '  plot-3d_*/                 overlay scenes (reference template)',
            '  report.html                full report',
            '  overview.csv               queried type x pair: best target-type cell',
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
                neurons = [skeletons[int(b)] for b in shown
                           if int(b) in skeletons]
                if not neurons:
                    continue
                for n in neurons:
                    try:
                        n.name = f'{_safe_name(type_name, 20)}_{abbrev}'
                        n._drocat_source_dataset = ds
                    except Exception:  # noqa: BLE001
                        pass
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
            'legend_mode': 'layer',
            'brain_mesh': 'template',
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
        esc = lambda s: (str(s).replace('&', '&amp;').replace('<', '&lt;')
                         .replace('>', '&gt;'))
        parts: List[str] = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Cross-Dataset Morphology Comparison</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1d1d1f;background:#fafafa}
h1{font-size:1.5rem} h2{font-size:1.15rem;margin-top:1.6rem}
.card{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:16px 20px;margin:14px 0}
table{border-collapse:collapse;margin:8px 0;font-size:.9rem}
th,td{border:1px solid #d9d9de;padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
.warn{background:#fff7e0;border:1px solid #f0c36d;border-radius:8px;padding:10px 14px;margin:10px 0}
.note{color:#666;font-size:.85rem}
a{color:#0b66d0}
</style></head><body>""",
                           '<h1>Cross-Dataset Morphology Comparison</h1>',
                           f'<p class="note">Generated '
                           f'{datetime.now():%Y-%m-%d %H:%M:%S} · queries: '
                           f'{esc(", ".join(tokens))} · datasets: '
                           f'{esc(", ".join(datasets))}<br>'
                           'Scoring: production vector_v2 in each target '
                           'render space (no NBLAST cross-dataset). '
                           'Scores are comparable within a column, not '
                           'across columns: each dataset pair is scored in '
                           'its target frame.</p>']
        if self.warnings:
            parts.append('<div class="warn"><b>Warnings</b><ul>' + ''.join(
                f'<li>{esc(w)}</li>' for w in self.warnings) + '</ul></div>')

        if not overview.empty:
            parts.append('<h2>Overview — best target-type match per queried type</h2>')
            parts.append('<div class="card">')
            parts.append(overview.to_html(index=False, na_rep='—',
                                          border=0, classes='ov'))
            parts.append('</div>')

        for (a, b), info in pairs.items():
            abbrev = (f'{self._abbrevs.get(a, _dataset_abbrev(a))} → '
                      f'{self._abbrevs.get(b, _dataset_abbrev(b))}')
            parts.append(f'<h2>{esc(abbrev)}</h2><div class="card">')
            if info.get('error'):
                parts.append(f'<p class="warn">{esc(info["error"])}</p></div>')
                continue
            baseline = info.get('baseline') or {}
            if baseline:
                p95s = [s['p95'] for s in baseline.values()]
                med = [s['median'] for s in baseline.values()]
                ns = [s['n'] for s in baseline.values()]
                parts.append(
                    f'<p class="note">Null baseline (seeded random targets, '
                    f'k≈{max(ns) if ns else 0}): p95 '
                    f'{min(p95s):.3f}–{max(p95s):.3f}, median '
                    f'{min(med):.3f}–{max(med):.3f}. Candidate scores above '
                    'the p95 line are above the unrelated-pair noise '
                    'floor.</p>')
            matrix = info.get('type_matrix')
            if matrix is not None and not matrix.empty:
                parts.append('<p><b>Type × type mean vector_v2</b></p>')
                parts.append(matrix.to_html(na_rep='—', border=0,
                                            float_format=lambda v: f'{v:.3f}'))
            rows = info.get('rows')
            if rows is not None and not rows.empty:
                n_above = int(rows['above_baseline'].fillna(False).sum())
                parts.append(f'<p>Member pairs: {len(rows)} · above the '
                             f'baseline p95: {n_above} · '
                             f'<a href="{esc(abbrev.replace(" → ", "_to_"))}'
                             '/bodyid_scores.csv">bodyid_scores.csv</a></p>')
            parts.append('</div>')

        if scenes:
            parts.append('<h2>Overlay scenes</h2><div class="card"><ul>' +
                         ''.join(f'<li><a href="{esc(p.name)}">'
                                 f'{esc(Path(p.parent.name) / p.name)}</a></li>'
                                 for p in scenes) + '</ul>'
                         f'<p class="note">Scene frame: '
                         f'{esc(self.reference_template or datasets[0])} '
                         'render space — different frame from the scores '
                         '(frame disclosure).</p></div>')
        if notes:
            parts.append('<h2>Resolution notes</h2><div class="card"><ul>' +
                         ''.join(f'<li>{esc(n)}</li>' for n in notes) +
                         '</ul></div>')
        parts.append('</body></html>')
        report = run_path / 'report.html'
        report.write_text('\n'.join(parts))
        files = list(existing or [])
        files.append(report)
        self._log(f'report written: {report}')
        return files
