"""MorphologyProfileComparer - intra-dataset morphology comparison.

Given ONE dataset and 2+ queried neurons (types, bodyIds, or regex
patterns), computes the N×N morphological similarity matrix at bodyId level
and an aggregated type-level matrix, then writes CSV matrices, interactive
heatmaps, and a report — the morphology analogue of the Connectivity tab's
Comparison sub-tab (``ConnectivityProfileComparer``).

Two scoring methods:
- ``vector_v2`` (default): the production similarity of Find Similar —
  per-block cosine (shape/spatial 0.30/0.70) on standardized + ZCA-whitened
  vectors from the per-dataset ``SkeletonVectorCacheV2``.
- ``nblast``: canonical normalized NBLAST on raw-skeleton dotprops. Both
  orientations of every pair are scored and averaged (the forward score is
  asymmetric); type means exclude contralateral pairs (mirror arbors score
  at chance), matching Find Similar's ipsilateral-only aggregation.

Intra-dataset ONLY: vectors and NBLAST dotprops are scored in one dataset's
coordinate space against that dataset's caches. Cross-dataset workflows
belong to the connectivity side.

Output folder (under ``output_dir``)::

    morphology_comparison_{DATASET}_{query}_{ts}/
      parameters.json
      README.txt
      members.csv                       # resolved type/bodyId provenance
      type_level/type_similarity_{method}.csv
      bodyid_level/bodyid_similarity_{method}.csv
      visualization/heatmap_type_{method}.html
      visualization/heatmap_bodyid_{method}.html
      report.html
"""

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from morphology import (
        DEFAULT_V2_BLOCK_WEIGHTS,
        TYPE_RENDER_MEMBER_CAP,
        MorphologyComparer,
        NEUPRINT_FETCH_MAX_THREADS,
        VECTOR_BASIS_RAW,
        _canonical_dataset_body_id,
        _dataset_folder,
        _dataset_soma_side_map,
        _import_visualizer,
        _load_neuron_type_map,
        _neuron_rep,
        apply_whitening,
        fetch_skeletons_on_demand_batch,
        find_similar_dataset_cache_v2,
        load_local_release_skeletons,
        v2_pairwise_matrix,
    )
except ImportError:  # direct src/ execution
    from morphology import (  # type: ignore
        DEFAULT_V2_BLOCK_WEIGHTS,
        TYPE_RENDER_MEMBER_CAP,
        MorphologyComparer,
        NEUPRINT_FETCH_MAX_THREADS,
        VECTOR_BASIS_RAW,
        _canonical_dataset_body_id,
        _dataset_folder,
        _dataset_soma_side_map,
        _import_visualizer,
        _load_neuron_type_map,
        _neuron_rep,
        apply_whitening,
        fetch_skeletons_on_demand_batch,
        find_similar_dataset_cache_v2,
        load_local_release_skeletons,
        v2_pairwise_matrix,
    )

try:
    from utils.label_utils import body_id_label_map
except ImportError:  # pragma: no cover - direct src/ execution
    from label_utils import body_id_label_map  # type: ignore

try:
    from visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:  # pragma: no cover - direct src/ execution
    try:
        from src.visualization_options import (
            default_analysis_skeleton_mesh_simplification)  # type: ignore
    except ImportError:
        def default_analysis_skeleton_mesh_simplification(dataset, pipeline):
            return None

try:
    from flywire_ids import is_banc_dataset, is_fafb_dataset
except ImportError:  # pragma: no cover - direct src/ execution
    from flywire_ids import (  # type: ignore
        is_banc_dataset, is_fafb_dataset)

try:
    from utils.naming_utils import dataset_abbrev
except ImportError:  # pragma: no cover
    def dataset_abbrev(dataset: str) -> str:
        return str(dataset or "").split(":")[0].replace("_", "")[:5].upper()

try:
    from comparison import report_kit
except ImportError:  # pragma: no cover - direct src/ execution
    import report_kit


# Regex metacharacters that mark a query token as a pattern rather than an
# exact type name (mirrors the UI filter modes: 'aMe.*', '.*KC.*', ...).
_PATTERN_CHARS = set("*?[](){}|^$.+\\")

# NBLAST builds a dotprop per neuron and scores every pair twice: O(N²)
# skeleton loads dominate quickly, so the method is hard-capped.
NBLAST_MAX_NEURONS = 30

_METHOD_LABELS = {
    "vector_v2": "Vector (spatial, vector_v2)",
    "nblast": "NBLAST",
}


def _looks_like_pattern(token: str) -> bool:
    return any(ch in _PATTERN_CHARS for ch in str(token))


def _safe_name(value: str, limit: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return safe[:limit] or "query"


class MorphologyProfileComparer:
    """Compare the morphology of already-identified neurons within ONE dataset.

    The query is a single list of neurons; the result is one N×N matrix.
    Rows are neuron types: each queried type contributes one row (diagonal =
    intra-type cohesion, the mean pairwise score among its members), and the
    type-level entry is the mean over the cross-member bodyId pairs. The
    bodyId-level matrix carries every individual pair.
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        query: Optional[Union[str, int, List[Union[str, int]]]] = None,
        method: str = "vector_v2",
        max_members_per_type: int = 25,
        max_total_neurons: int = 200,
        fetch_online: bool = True,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        generate_heatmaps: bool = True,
        show_figures: bool = False,
        use_cache: bool = True,
        verbose: bool = True,
        n_workers: int = 8,
        project_root: Optional[str] = None,
        visualize: bool = False,
        visualization_settings: Optional[Dict[str, object]] = None,
    ):
        self.dataset = dataset
        self.query = query
        self.method = str(method).lower()
        self.max_members_per_type = max(1, int(max_members_per_type))
        self.max_total_neurons = max(2, int(max_total_neurons))
        self.fetch_online = bool(fetch_online)
        self.output_dir = output_dir
        self.saveas = str(saveas or "").strip()
        self.generate_heatmaps = bool(generate_heatmaps)
        self.show_figures = bool(show_figures)
        self.use_cache = bool(use_cache)
        self.verbose = bool(verbose)
        self.n_workers = max(1, int(n_workers))
        self.visualize = bool(visualize)
        self.visualization_settings = dict(visualization_settings or {})
        self.project_root = (
            Path(project_root) if project_root
            else Path(__file__).parent.parent
        )
        if self.method not in ("vector_v2", "nblast"):
            raise ValueError(
                f"Invalid method: {self.method} (vector_v2|nblast)")
        if is_banc_dataset(self.dataset):
            raise ValueError(
                "BANC morphological comparison is deferred: the public "
                "L2/full skeletons still need vector-quality validation. "
                "3D skeleton visualization for BANC is available.")

    # ------------------------------------------------------------------ log
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MorphologyProfileComparer] {msg}", flush=True)

    def _body_id(self, value):
        return _canonical_dataset_body_id(self.dataset, value)

    # ------------------------------------------------------------- resolution
    def _resolve_members(self) -> "Dict[str, List[object]]":
        """Resolve the query into {type: [member bodyIds]} (input order).

        Numeric tokens are bodyIds (resolved to their type via the dataset
        neuron table); non-numeric tokens are exact type names or regex
        patterns expanded against the dataset's type names. Members are
        capped per type and in total.
        """
        tokens: List[Union[str, int]] = []
        if self.query is None:
            tokens = []
        elif isinstance(self.query, (str, int)):
            tokens = [self.query]
        else:
            tokens = list(self.query)
        tokens = [t for t in tokens if str(t).strip()]
        if not tokens:
            raise ValueError("Please provide at least two neurons to compare.")

        type_map, instance_map = _load_neuron_type_map(
            self.dataset, str(self.project_root))
        if not type_map:
            raise ValueError(
                f"Dataset '{self.dataset}' has no local neuron table; pull "
                "the dataset first (Settings → Dataset Cache).")
        # Cached for the bodyId display labels and the members table so the
        # neuron table is read once per run.
        self._type_map = type_map
        self._instance_map = instance_map
        all_types = sorted({str(t or "").strip() for t in type_map.values()
                            if str(t or "").strip()})

        members: Dict[str, List[object]] = {}
        warning_missing_types: List[str] = []

        def _add_type(type_name: str) -> None:
            if type_name in members:
                return
            ids = [bid for bid, t in type_map.items()
                   if str(t or "").strip() == type_name]
            if not ids:
                return
            ids.sort(key=lambda b: str(b))
            if len(ids) > self.max_members_per_type:
                self._log(
                    f"{type_name}: capping {len(ids)} members to "
                    f"{self.max_members_per_type} (max_members_per_type).")
                ids = ids[: self.max_members_per_type]
            members[type_name] = ids

        for token in tokens:
            text = str(token).strip()
            # Exact type names win over pattern interpretation (some dataset
            # type names contain regex metacharacters, e.g. "PPL1*").
            if text in all_types:
                _add_type(text)
                continue
            if text.isdigit():
                bid = self._body_id(text)
                type_name = str(type_map.get(bid, "") or "").strip()
                if not type_name:
                    raise ValueError(
                        f"bodyId {text} was not found in dataset "
                        f"'{self.dataset}'.")
                _add_type(type_name)
                continue
            if _looks_like_pattern(text):
                try:
                    rx = re.compile(text)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid query pattern '{text}': {exc}")
                matched = [t for t in all_types if rx.fullmatch(t)]
                if not matched:
                    warning_missing_types.append(text)
                    continue
                for type_name in matched:
                    _add_type(type_name)
                continue
            warning_missing_types.append(text)

        if warning_missing_types:
            self._log(
                "No dataset types matched: "
                + ", ".join(map(str, warning_missing_types)))
        # ``members`` is already in first-occurrence query order (dicts
        # preserve insertion order).

        if len(members) < 2:
            raise ValueError(
                "Morphology comparison needs at least two resolved types; "
                f"query resolved {len(members)}.")
        total = sum(len(v) for v in members.values())
        if total > self.max_total_neurons and self.method == "nblast":
            raise ValueError(
                f"NBLAST comparison is capped at {NBLAST_MAX_NEURONS} total "
                f"neurons (got {total}); reduce the query or the member cap.")
        if total > self.max_total_neurons:
            self._log(
                f"Query resolves to {total} neurons; truncating to "
                f"{self.max_total_neurons} (max_total_neurons).")
            kept: Dict[str, List[object]] = {}
            budget = self.max_total_neurons
            for type_name, ids in members.items():
                if budget <= 0:
                    break
                take = ids[:budget]
                kept[type_name] = take
                budget -= len(take)
            members = kept
        return members

    # ----------------------------------------------------------------- vector
    def _fetch_missing_vectors(self, cache, missing_ids: List[object]) -> int:
        """Fetch skeletons online for ``missing_ids`` and vectorize them into
        the cache — the same contract as Find Similar's cache-direct search.

        NeuPrint datasets go through the shared batch fetch (raw SWC staged
        + persisted into the shared skeleton cache); FAFB goes through
        ``load_local_release_skeletons`` (raw cache → healed FAFB zip →
        CAVE fallback); BANC goes through the shared batch fetch too — its
        branch resolves each body via the official public-bucket SWC chain
        (``fetch_banc_swc``), never the FAFB CAVE machinery, which has
        no BANC products. Fetched neurons are re-vectorized with the
        cache's own vectorizer so rows land in the cache's exact schema.
        Returns the number of neurons vectorized."""
        self._log(
            f"Vector cache miss: fetching {len(missing_ids)} skeleton(s) "
            "online.")
        if is_fafb_dataset(self.dataset):
            neurons = load_local_release_skeletons(
                self.dataset, [int(self._body_id(b)) for b in missing_ids],
                project_root=str(self.project_root), log=self._log)
        else:
            neurons = fetch_skeletons_on_demand_batch(
                self.dataset, missing_ids,
                project_root=str(self.project_root),
                persist=True,
                level=VECTOR_BASIS_RAW,
                max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                                max(1, int(self.n_workers))),
                raw_cache=cache,
                vector_cache=None,
            )
        rows = []
        for bid, neuron in (neurons or {}).items():
            try:
                if _neuron_rep(neuron) != "skeleton":
                    continue
                _, vec = cache._vectorize_neuron(neuron)
                rows.append((self._body_id(bid), vec, "skeleton"))
            except Exception as exc:
                # Glitchy fetches (empty/partial SWC) are skipped from the
                # vector cache, mirroring Find Similar's behavior.
                self._log(f"vectorization skipped for {bid}: {exc}")
        if rows:
            cache.append_vectors(rows, vector_basis=cache._default_basis())
        self._log(
            f"Fetched + vectorized {len(rows)}/{len(missing_ids)} "
            "missing neuron(s).")
        return len(rows)

    def _vector_matrix(self, all_ids: List[object]) -> np.ndarray:
        """Whitened vector rows for ``all_ids`` (order-preserving).

        Local cache rows come first; missing neurons are fetched through the
        API when ``fetch_online`` is on (default), persisting into the
        shared skeleton + vector caches so one comparison warms every later
        run. Still-missing neurons stay NaN and are reported via
        ``members.csv``.
        """
        cache = find_similar_dataset_cache_v2(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=self.verbose)
        canonical = [self._body_id(b) for b in all_ids]
        cache.vectors_for(canonical, compute_missing=self.use_cache)
        data = cache.load()
        index = (self._cache_index(data) if data is not None else {})
        missing = [b for b in canonical if b not in index]
        if missing and self.fetch_online:
            self._fetch_missing_vectors(cache, missing)
            data = cache.load()
            index = (self._cache_index(data) if data is not None else {})
        if data is None:
            raise ValueError(
                f"No morphology vector cache for '{self.dataset}' and no "
                "skeletons available locally or online. Run Find Similar "
                "once or download skeletons (Settings → Dataset Cache).")
        X = data["X"]
        rows = np.full((len(canonical), X.shape[1]), np.nan)
        for out_i, bid in enumerate(canonical):
            src_i = index.get(bid, -1)
            if src_i >= 0:
                rows[out_i] = X[src_i]
        W = data.get("whiten")
        if W is not None and getattr(W, "size", 0):
            valid = ~np.isnan(rows[:, 0])
            if valid.any():
                rows[valid] = apply_whitening(W, rows[valid])
        return rows

    def _cache_index(self, data: dict) -> dict:
        """bodyId → row index for a loaded cache snapshot."""
        index = {}
        for i, bid in enumerate(data["bodyIds"]):
            try:
                index[self._body_id(bid)] = i
            except (TypeError, ValueError):
                index[bid] = i
        return index

    # ----------------------------------------------------------------- nblast
    def _nblast_matrix(self, all_ids: List[object]) -> Tuple[np.ndarray, List[object]]:
        """Symmetric normalized-NBLAST matrix (mean of both orientations)."""
        total = len(all_ids)
        if total > NBLAST_MAX_NEURONS:
            raise ValueError(
                f"NBLAST comparison is capped at {NBLAST_MAX_NEURONS} total "
                f"neurons (got {total}).")
        helper = MorphologyComparer(
            dataset=self.dataset, method="nblast",
            verbose=self.verbose, n_workers=self.n_workers,
            project_root=str(self.project_root))
        dps = helper._dotprops_for_ids(
            [self._body_id(b) for b in all_ids],
            desc="Building comparison dotprops")
        kept = [bid for bid in all_ids
                if dps.get(int(self._body_id(bid))) is not None]
        dropped = len(all_ids) - len(kept)
        if dropped:
            self._log(f"NBLAST: {dropped} neuron(s) without dotprops dropped.")
        if len(kept) < 2:
            raise ValueError(
                "Fewer than two neurons produced NBLAST dotprops; cannot "
                "compare.")

        from navis.nbl.nblast_funcs import NBlaster
        nb = NBlaster(use_alpha=False, normalized=True, progress=False)
        idx = {}
        for bid in kept:
            dp = dps[int(self._body_id(bid))]
            idx[bid] = nb.append(dp, self_hit=nb.calc_self_hit(dp))

        n = len(kept)
        matrix = np.full((n, n), np.nan)
        for i in range(n):
            matrix[i, i] = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    fwd = float(nb.single_query_target(
                        idx[kept[i]], idx[kept[j]], scores="forward"))
                except Exception:
                    fwd = float("nan")
                try:
                    rev = float(nb.single_query_target(
                        idx[kept[j]], idx[kept[i]], scores="forward"))
                except Exception:
                    rev = float("nan")
                vals = [v for v in (fwd, rev) if math.isfinite(v)]
                score = float(np.mean(vals)) if vals else float("nan")
                matrix[i, j] = matrix[j, i] = score
        return matrix, kept

    # ------------------------------------------------------------ aggregation
    def _type_level_matrix(self, body_matrix: np.ndarray,
                           labels: List[object],
                           members: Dict[str, List[object]]) -> pd.DataFrame:
        """Type×type means over cross-member blocks; diagonal = cohesion.

        NBLAST type means exclude contralateral member pairs (mirror arbors
        score at chance); unknown/midline sides are always kept.
        """
        sides: Dict[object, str] = {}
        if self.method == "nblast":
            raw = _dataset_soma_side_map(
                self.dataset, str(self.project_root))
            sides = {self._body_id(b): {"left": "L", "right": "R"}.get(name, "")
                     for b, name in (raw or {}).items()}

        def _pair_ok(a: object, b: object) -> bool:
            if self.method != "nblast":
                return True
            sa, sb = sides.get(a, ""), sides.get(b, "")
            if sa in ("L", "R") and sb in ("L", "R") and sa != sb:
                return False
            return True

        def _block_mean(pairs: List[float]) -> float:
            finite = [v for v in pairs if v is not None
                      and np.isfinite(v)]
            return float(np.mean(finite)) if finite else np.nan

        pos = {bid: i for i, bid in enumerate(labels)}
        types = list(members.keys())
        out = pd.DataFrame(np.nan, index=types, columns=types, dtype=float)
        for a in types:
            for b in types:
                if a == b:
                    # Keep member identity and matrix index locked together:
                    # members absent from the matrix must not shift the
                    # subscripts (the old code indexed members[a][ii] with a
                    # filtered-list ii, misattributing pairs).
                    pairs = [(self._body_id(x), pos[self._body_id(x)])
                             for x in members[a]
                             if self._body_id(x) in pos]
                    if len(pairs) <= 1:
                        out.loc[a, a] = 1.0 if pairs else np.nan
                        continue
                    vals = []
                    for ii in range(len(pairs)):
                        for jj in range(len(pairs)):
                            if ii == jj:
                                continue
                            xa, xb = pairs[ii][0], pairs[jj][0]
                            if not _pair_ok(xa, xb):
                                continue
                            vals.append(body_matrix[pairs[ii][1],
                                                    pairs[jj][1]])
                    out.loc[a, a] = _block_mean(vals)
                elif pd.isna(out.loc[a, b]):
                    vals = []
                    for xa in members[a]:
                        for xb in members[b]:
                            if self._body_id(xa) not in pos \
                                    or self._body_id(xb) not in pos:
                                continue
                            if not _pair_ok(xa, xb):
                                continue
                            vals.append(
                                body_matrix[pos[self._body_id(xa)],
                                            pos[self._body_id(xb)]])
                    val = _block_mean(vals)
                    out.loc[a, b] = out.loc[b, a] = val
        return out

    # ------------------------------------------------------------------ files
    def _output_path(self, query_name: str) -> Path:
        base = (Path(self.output_dir) if self.output_dir
                else self.project_root / "local_data" / "morphology_comparison")
        base.mkdir(parents=True, exist_ok=True)
        if self.saveas:
            name = self.saveas
        else:
            name = (f"morphology_comparison_{dataset_abbrev(self.dataset)}"
                    f"_{_safe_name(query_name)}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return base / name

    def _metric_style(self) -> "report_kit.MetricStyle":
        """Report/heatmap presentation per scoring method.

        vector_v2 is a whitened cosine that can go negative, so it uses
        the shared kit's diverging [-1, 1] scale; NBLAST similarity is
        treated as a [0, 1] score on the positive scale (matching the
        historical rendering).
        """
        if self.method == "vector_v2":
            return report_kit.MetricStyle(
                "morph_similarity", "Vector v2",
                report_kit.REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)
        return report_kit.MetricStyle(
            "morph_similarity", "NBLAST similarity",
            report_kit.REPORT_POSITIVE_COLORSCALE, 0.0, 1.0)

    def _write_heatmaps(self, matrices: Dict[str, pd.DataFrame],
                        viz_dir: Path) -> List[str]:
        """Standalone interactive VisPath heatmaps (shared report_kit).

        Card keys are the METRIC key (the kit looks styles up by it); the
        level rides in the closures so the exported filenames stay
        ``heatmap_{level}_{method}.html``. VisPath renders with native
        clustering; any failure re-renders through the plotly fallback.
        """
        style = self._metric_style()
        styles = {style.key: style}
        method_label = _METHOD_LABELS.get(self.method, self.method)
        level_labels = {"type": "Type level", "bodyid": "BodyId level"}
        saved: Dict[str, List[str]] = {"heatmaps_generated": []}
        for level in ("type", "bodyid"):
            matrix = matrices.get(level)
            if matrix is None or matrix.empty:
                continue
            report_kit.generate_standalone_heatmaps(
                {self.dataset: {style.key: matrix}},
                viz_dir, styles,
                filename_builder=lambda group, key, _lv=level:
                    f"heatmap_{_lv}_{self.method}.html",
                group_display=lambda group: str(group),
                vispath_title=lambda group, key, gd, _lv=level:
                    f"{method_label} — {gd} · "
                    f"{level_labels.get(_lv, _lv)}",
                fallback_title=lambda group, key, gd, _lv=level:
                    f"Morphology Comparison - {_lv}",
                tqdm_desc=f"Generating {level_labels.get(level, level)} "
                          "heatmaps",
                show_figures=self.show_figures, verbose=self.verbose,
                saved_files=saved, log=self._log)
        return saved["heatmaps_generated"]

    def _write_report(self, report_path: Path,
                      matrices: Dict[str, pd.DataFrame],
                      csv_links: Dict[str, str],
                      params: Dict[str, object]) -> None:
        """Tabbed report on the shared report_kit (the same generator
        family as the connectivity-profiling and cross-dataset morphology
        reports): hero header, Type/BodyId level tabs with Ward-clustered
        heatmap cards (CSV + VisPath editor links), compared-members and
        parameter details, and the 3D scene link. Plotly.js is embedded,
        so the report renders offline.
        """
        from html import escape

        member_rows = params.pop("_member_rows", [])
        plot3d_link = str(params.pop("_plot3d_link", "") or "")
        method_label = _METHOD_LABELS.get(self.method, self.method)
        style = self._metric_style()
        compared = sum(1 for m in member_rows
                       if m.get("status") == "compared")

        def chip(label: str, value: str) -> str:
            return (f"<div class='meta-chip'><span>{escape(label)}</span>"
                    f"<strong>{escape(value)}</strong></div>")

        lines = [
            '<!DOCTYPE html>',
            "<html><head><meta charset='utf-8'>",
            '<title>Morphology Comparison Report</title>',
            report_kit.report_css(),
            '</head><body><main class="report-shell">',
            '<header class="report-hero">',
            '<div class="report-kicker">DROCAT · Intra-dataset '
            'morphology</div>',
            '<h1 class="report-title">Morphology comparison</h1>',
            '<p class="report-subtitle">N×N similarity of the queried '
            f'neurons in {escape(str(self.dataset))} '
            f'({escape(method_label)}). Each queried type contributes one '
            'type-level row — the diagonal is the intra-type cohesion and '
            'each entry the mean over the cross-member bodyId pairs; the '
            'bodyId level carries every individual pair. Use the VisPath '
            'editor links to change clustering; hover cells for exact '
            'values.</p>',
            '<div class="report-meta">',
            chip('Dataset', str(self.dataset)),
            chip('Method', method_label),
            chip('Types compared', str(params.get("types_compared", "—"))),
            chip('Neurons compared',
                 f"{compared}/{len(member_rows)}"),
            '</div>',
        ]
        scale_note = ('negative vector_v2 cells are genuinely negative '
                      '(whitened cosine) and render blue on the diverging '
                      'scale.' if self.method == "vector_v2" else
                      'NBLAST similarity renders on the positive [0, 1] '
                      'scale.')
        lines.append(
            f"<p class='report-note'>Generated "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} · {scale_note}</p>")
        lines.append('</header>')

        plotly_state = {'include_plotlyjs': True}
        level_labels = {"type": "Type level", "bodyid": "BodyId level"}

        def render_level(level: str, _level_panel_id: str) -> None:
            matrix = matrices.get(level)
            matrix = matrix if matrix is not None else pd.DataFrame()
            if matrix.empty:
                lines.append(
                    "<div class='heatmap-empty'>This level was not "
                    'computed for this run.</div>')
                return
            scored = int(matrix.notna().sum().sum())
            note = (f'{matrix.shape[0]}×{matrix.shape[1]} · {scored} '
                    'scored cells')
            if level == 'type':
                note += ' · diagonal = intra-type cohesion'
            lines.append(
                f"<div class='direction-intro'><h3 class='direction-title'>"
                f'{escape(level_labels.get(level, level))}</h3>'
                f"<div class='direction-note'>{escape(note)}</div></div>")
            report_kit.append_report_metric_grid(
                lines, report_path.parent, {style.key: matrix},
                f'Intra-dataset · {level_labels.get(level, level)}',
                {style.key: csv_links.get(level, "")},
                {style.key: f'visualization/heatmap_{level}_'
                            f'{self.method}.html'},
                'Neuron' if level == 'bodyid' else 'Type',
                'Neuron' if level == 'bodyid' else 'Type',
                plotly_state,
                styles={style.key: style},
                square_cells=True,
            )

        report_kit.append_report_tab_group(
            lines, 'morph-levels',
            [(level, level_labels[level]) for level in ('type', 'bodyid')],
            render_level,
            panel_class='tab-panel direction-panel')

        # --- compared members + parameters details -----------------------
        detail_bits = []
        if member_rows:
            rows = "".join(
                f"<tr><td>{escape(str(m['type']))}</td>"
                f"<td>{escape(str(m['bodyId']))}</td>"
                f"<td>{escape(str(m.get('instance', '')))}</td>"
                f"<td>{escape(str(m.get('status', '')))}</td></tr>"
                for m in member_rows)
            detail_bits.append(
                '<details class="detail-block"><summary>Compared '
                'neurons</summary><div style="overflow-x:auto">'
                "<table class='mapping-table'><thead><tr><th>type</th>"
                '<th>bodyId</th><th>instance</th><th>status</th></tr>'
                f'</thead><tbody>{rows}</tbody></table></div></details>')
        if params:
            rows = "".join(
                f'<tr><th>{escape(str(k))}</th>'
                f'<td>{escape(str(v))}</td></tr>'
                for k, v in sorted(params.items()))
            detail_bits.append(
                '<details class="detail-block"><summary>Parameters'
                '</summary><div style="overflow-x:auto">'
                f"<table class='mapping-table'><tbody>{rows}</tbody>"
                '</table></div></details>')
        if detail_bits:
            lines.append('<div class="section-card">' + ''.join(detail_bits)
                         + '</div>')

        if plot3d_link:
            lines.append(
                '<div class="section-card"><h2 class="section-heading">'
                '3D skeleton visualization</h2>'
                f"<p><a href='{escape(plot3d_link)}'>Open plot-3d scene"
                '</a> — one layer per compared type; the legend tree '
                'lists every bodyId leaf.</p></div>')

        lines.extend(['</main>', report_kit.report_script(),
                      '</body></html>'])
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        self._log(f'report written: {report_path}')

    def _display_labels(self, body_ids: List[object]) -> List[str]:
        """Tree-legend display labels ('{bodyId}_{instance}' or
        '{bodyId}_{type}_{L|R}') for the bodyId-level matrix axes."""
        canon_ids = [self._body_id(b) for b in body_ids]
        label_map = body_id_label_map(
            self.dataset, canon_ids, str(self.project_root),
            type_map=getattr(self, "_type_map", None),
            instance_map=getattr(self, "_instance_map", None))
        return [label_map.get(str(bid), str(bid)) for bid in canon_ids]

    def _offline_render_filter(
            self, members: Dict[str, List[object]]
    ) -> Tuple[Dict[str, List[object]], List[str]]:
        """Drop members whose raw skeletons are not locally cached.

        Only applies when ``fetch_online=False`` and the dataset is
        NeuPrint-hosted: VisualizeSkeleton would otherwise attempt the
        declined online fetch for the missing members and abort the whole
        scene. FAFB/BANC resolve offline through the healed bundle /
        public bucket, so no filtering is needed there. Returns the
        filtered members plus human-readable skip notes.
        """
        if self.fetch_online:
            return members, []
        if is_fafb_dataset(self.dataset) or is_banc_dataset(self.dataset):
            return members, []
        try:
            from morphology import find_similar_raw_cache
            raw_cache = find_similar_raw_cache(
                self.dataset, project_root=str(self.project_root),
                verbose=False)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Raw cache unavailable for the offline scene: {exc}")
            return members, []
        filtered: Dict[str, List[object]] = {}
        skipped: List[str] = []
        for type_name, ids in members.items():
            kept = []
            for bid in ids:
                try:
                    neuron = raw_cache.load_skeleton(int(self._body_id(bid)))
                except Exception:  # noqa: BLE001
                    neuron = None
                # Mirror VisualizeSkeleton's render-cache policy: only
                # level-0 raw files are renderable sources; legacy simp90
                # files would abort the scene when they cannot be
                # refreshed online.
                level = 0
                if neuron is not None:
                    try:
                        level = max(0, int(
                            getattr(neuron, '_drocat_simplification', 0)
                            or 0))
                    except (TypeError, ValueError):
                        level = 0
                ok = neuron is not None and level == 0
                (kept if ok else skipped).append(
                    bid if ok else f"{type_name}:{self._body_id(bid)}")
            if kept:
                filtered[type_name] = kept
        if skipped:
            self._log(
                f"Offline scene: skipping {len(skipped)} member(s) whose "
                f"skeletons are not cached (fetch_online=False); turn on "
                f"'Fetch Missing Skeletons Online' to include them.")
        return filtered, skipped

    def _visualize_members(self, output_path: Path,
                           members: Dict[str, List[object]]
                           ) -> Optional[str]:
        """Render the compared neurons as one skeleton layer per type.

        Mirrors Find Similar's 3D scene (line skeletons, interactive
        tree legend, template brain, no synapses); each compared type contributes one
        layer capped at ``TYPE_RENDER_MEMBER_CAP`` members for the render
        only. With ``fetch_online=False`` the scene is strictly offline:
        members without locally cached skeletons are skipped (reported)
        instead of triggering a doomed online fetch that would abort the
        whole scene. Returns the report-relative scene link, or None when
        the renderer is unavailable or fails — a visualization problem
        never fails the comparison.
        """
        try:
            VisualizeSkeleton = _import_visualizer()
        except Exception as exc:
            self._log(f"3D visualization failed (comparison kept): {exc}")
            return None
        if VisualizeSkeleton is None:
            self._log(
                "3D visualization skipped: visualize_skeleton unavailable.")
            return None

        # Unified offline behavior: with fetch_online=False, only members
        # whose raw skeletons are locally cached enter the scene (the
        # cross-dataset mode follows the same cache-first rule).
        members, _offline_skipped = self._offline_render_filter(members)
        if not members:
            self._log(
                "3D visualization skipped: no locally cached member "
                "skeletons (fetch_online=False). Turn on 'Fetch Missing "
                "Skeletons Online' to fetch and render them.")
            return None

        layers: List[List[object]] = []
        names: List[str] = []
        notes: List[str] = []
        for rank, (type_name, ids) in enumerate(members.items(), start=1):
            canon_ids = [self._body_id(b) for b in ids]
            shown = canon_ids[:TYPE_RENDER_MEMBER_CAP]
            layers.append(shown)
            safe_type = _safe_name(type_name, 40)
            names.append(f"t{rank}_{safe_type}_x{len(shown)}")
            if len(canon_ids) > len(shown):
                notes.append(
                    f"t{rank}_{safe_type}: showing {len(shown)} of "
                    f"{len(canon_ids)} members of type '{type_name}' "
                    f"(per-layer render cap {TYPE_RENDER_MEMBER_CAP})")
        if not layers:
            self._log("3D visualization skipped: no compared neurons.")
            return None

        self._log(
            f"3D visualization: {len(layers)} compared type layer(s), "
            f"{sum(len(layer) for layer in layers)} neurons.")

        settings = dict(self.visualization_settings)
        pipeline = str(
            settings.get("neuprint_skeleton_pipeline", "fine") or "fine"
        ).strip().lower()
        local_release = is_fafb_dataset(self.dataset) or is_banc_dataset(
            self.dataset)
        saveas = _dataset_folder(self.dataset)
        before = {p.name for p in output_path.iterdir() if p.is_dir()}
        viz_kwargs: Dict[str, object] = {
            "dataset": self.dataset,
            "output_dir": str(output_path),
            "neuron_layers": layers,
            "custom_layer_names": names,
            "saveas": saveas,
            "include_timestamp": False,
            "skip_synapse": True,
            # Analysis visualizations default to the light-weight line
            # representation, like Find Similar.
            "skeleton_mode": settings.get("skeleton_mode", "line"),
            "legend_mode": "tree",
            "brain_mesh": "native",
            "export_views": False,
            "show_fig": False,
            "cache_neurons": (
                True if local_release
                else pipeline not in {"fast", "direct", "artistic",
                                      "fine_opt1"}),
            "verbose": "simple",
            "layer_sample_notes": notes or None,
        }
        # The settings panel carries the same keyword names as
        # VisualizeSkeleton; ranking controls belong to the caller.
        for key, value in settings.items():
            if key in {"visualize_top_n", "visualize_by",
                       "use_default_simplification"}:
                continue
            if key == "mesh_color" and value == "auto":
                continue
            # 'tree' is the default; an explicit settings value wins so
            # the user can pick 'layer' or 'tree' either way.
            viz_kwargs[key] = value
        if viz_kwargs.get("skeleton_mesh_simplification") is None:
            viz_kwargs["skeleton_mesh_simplification"] = (
                default_analysis_skeleton_mesh_simplification(
                    self.dataset, pipeline))

        try:
            vs = VisualizeSkeleton(**viz_kwargs)
            vs.plot_neurons()
        except Exception as exc:
            self._log(f"3D visualization failed (comparison kept): {exc}")
            return None
        scene_link = self._scene_link(output_path, before, saveas)
        if scene_link:
            self._log(f"3D visualization saved to: {output_path / scene_link}")
        return scene_link

    @staticmethod
    def _scene_link(output_path: Path, before: set,
                    saveas: str) -> Optional[str]:
        """Report-relative path of the freshly rendered 3D scene.

        VisualizeSkeleton owns the folder naming (plot-3d_{abbrev}_{stem}),
        so the link is discovered by diffing the run folder: prefer the new
        plot-3d directory's '{saveas}.html', then any top-level HTML.
        """
        new_dirs = [p for p in output_path.iterdir()
                    if p.is_dir() and p.name.startswith("plot-3d_")]
        candidates = [p for p in new_dirs if p.name not in before] or new_dirs
        for folder in sorted(candidates, key=lambda p: p.stat().st_mtime,
                             reverse=True):
            preferred = folder / f"{saveas}.html"
            if preferred.exists():
                return f"{folder.name}/{preferred.name}"
            htmls = sorted(folder.glob("*.html"))
            if htmls:
                return f"{folder.name}/{htmls[0].name}"
        return None

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, object]:
        started = time.time()
        members = self._resolve_members()
        all_ids: List[object] = []
        for ids in members.values():
            all_ids.extend(ids)
        total = len(all_ids)
        self._log(
            f"Comparing {len(members)} types / {total} neurons in "
            f"{self.dataset} ({self.method}).")

        if self.method == "nblast":
            kept_matrix, kept_ids = self._nblast_matrix(all_ids)
            kept_set = {self._body_id(k) for k in kept_ids}
            status_by_id = {
                self._body_id(b): ("compared" if self._body_id(b) in kept_set
                                   else "no dotprops")
                for b in all_ids}
            # Scatter the kept-only matrix back into the full id order so
            # labels keep their original positions (dropped ids = NaN).
            body_matrix = np.full((len(all_ids), len(all_ids)), np.nan)
            kept_pos = [i for i, b in enumerate(all_ids)
                        if self._body_id(b) in kept_set]
            for out_i, src_i in enumerate(kept_pos):
                body_matrix[src_i, kept_pos] = kept_matrix[out_i]
        else:
            X = self._vector_matrix(all_ids)
            valid = ~np.isnan(X[:, 0])
            status_by_id = {
                self._body_id(b): ("compared" if ok else "no vector")
                for b, ok in zip(all_ids, valid)}
            if int(valid.sum()) < 2:
                raise ValueError(
                    "Fewer than two queried neurons have vectors (skeletons "
                    "not available locally). Download skeletons or run Find "
                    "Similar once to warm the cache.")
            # Score valid rows only, then scatter back into a full matrix so
            # labels keep their original order.
            body_matrix = np.full((len(all_ids), len(all_ids)), np.nan)
            idxs = [i for i, ok in enumerate(valid) if ok]
            sub = v2_pairwise_matrix(X[idxs], DEFAULT_V2_BLOCK_WEIGHTS)
            for out_i, src_i in enumerate(idxs):
                body_matrix[src_i, idxs] = sub[out_i]
            for i, ok in enumerate(valid):
                if ok:
                    body_matrix[i, i] = 1.0
            kept_ids = [b for b, ok in zip(all_ids, valid) if ok]

        labels = [self._body_id(b) for b in all_ids]
        body_df = pd.DataFrame(body_matrix, index=labels, columns=labels)
        type_df = self._type_level_matrix(body_matrix, labels, members)
        # BodyId rows read as '{bodyId}_{instance}' or
        # '{bodyId}_{type}_{L|R}' (the tree-legend rule). The type-level
        # aggregation above resolves members by canonical bodyId, so the
        # display relabel happens only after it has run.
        display_labels = self._display_labels(all_ids)
        body_df.index = display_labels
        body_df.columns = display_labels

        output_path = self._output_path(
            "_".join(str(t) for t in list(members.keys())[:4]))
        (output_path / "type_level").mkdir(parents=True, exist_ok=True)
        (output_path / "bodyid_level").mkdir(parents=True, exist_ok=True)

        type_df.to_csv(output_path / "type_level"
                       / f"type_similarity_{self.method}.csv")
        body_df.to_csv(output_path / "bodyid_level"
                       / f"bodyid_similarity_{self.method}.csv")

        type_map = getattr(self, "_type_map", {}) or {}
        instance_map = getattr(self, "_instance_map", {}) or {}
        member_rows: List[Dict[str, object]] = []
        for type_name, ids in members.items():
            for bid in ids:
                canon = self._body_id(bid)
                member_rows.append({
                    "type": type_name,
                    "bodyId": canon,
                    "instance": str(instance_map.get(canon, "") or ""),
                    "status": status_by_id.get(canon, "compared"),
                })
        pd.DataFrame(member_rows).to_csv(
            output_path / "members.csv", index=False)

        matrices = {
            "type": type_df,
            "bodyid": body_df,
        }
        heatmap_files: List[str] = []
        if self.generate_heatmaps:
            heatmap_files = self._write_heatmaps(
                matrices, output_path / "visualization")

        plot3d_link: Optional[str] = None
        if self.visualize:
            plot3d_link = self._visualize_members(output_path, members)

        params = {
            "dataset": self.dataset,
            "query": [str(q) for q in
                      (self.query if isinstance(self.query, list)
                       else [self.query])],
            "method": self.method,
            "max_members_per_type": self.max_members_per_type,
            "max_total_neurons": self.max_total_neurons,
            "fetch_online": self.fetch_online,
            "types_compared": len(members),
            "neurons_compared": int(sum(
                1 for m in member_rows if m["status"] == "compared")),
            "intra_dataset_only": True,
            "generate_heatmaps": self.generate_heatmaps,
            "visualize": self.visualize,
            "duration_s": round(time.time() - started, 1),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        report_params = dict(params)
        report_params["_member_rows"] = member_rows
        if plot3d_link:
            report_params["_plot3d_link"] = plot3d_link
        (output_path / "parameters.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8")
        self._write_report(
            output_path / "report.html", matrices,
            csv_links={
                "type": f"type_level/type_similarity_{self.method}.csv",
                "bodyid": f"bodyid_level/bodyid_similarity_{self.method}.csv",
            },
            params=report_params)
        (output_path / "README.txt").write_text(
            self._readme_text(output_path, heatmap_files),
            encoding="utf-8")

        self._log(f"Output: {output_path}")
        return {
            "output_folder": str(output_path),
            "types_compared": len(members),
            "neurons_compared": params["neurons_compared"],
            "files": [str(p) for p in sorted(output_path.rglob("*"))
                      if p.is_file()],
        }

    def _readme_text(self, output_path: Path,
                     heatmap_files: List[str]) -> str:
        return f"""MORPHOLOGY COMPARISON — {self.dataset}
Generated {datetime.now().isoformat(timespec='seconds')}

Intra-dataset morphology comparison (method: {_METHOD_LABELS.get(self.method, self.method)}).
Type-level entry = mean over the cross-member bodyId pairs; the diagonal is
the type's intra-type cohesion (mean pairwise among its own members). The
bodyId-level matrix carries every individual pair.

Output layout:
  parameters.json                              run parameters
  members.csv                                  resolved type/bodyId provenance
  type_level/type_similarity_{self.method}.csv   type×type matrix
  bodyid_level/bodyid_similarity_{self.method}.csv  bodyId×bodyId matrix
  visualization/heatmap_*.html                 interactive heatmaps
  plot-3d_<dataset>/                           3D skeleton scene (when enabled;
                                               one layer per compared type)
  report.html                                  summary report

bodyId-level rows/axes are labeled '{{bodyId}}_{{instance}}' (NeuPrint-style
datasets) or '{{bodyId}}_{{type}}_L/_R' (FAFB/BANC); members.csv maps every
label back to its raw bodyId.

Morphological comparison is intra-dataset only: skeletons are scored in one
dataset's coordinate space against that dataset's caches. Select two or
more datasets in the Comparison sub-tab to switch to the cross-dataset
comparison (morph_cross_dataset.CrossDatasetMorphComparer) instead.
"""
