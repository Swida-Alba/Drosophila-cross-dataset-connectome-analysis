"""MorphologyProfileComparer - intra-dataset morphology comparison.

Given ONE dataset and 2+ queried neurons (types, taxonomy labels such as a
cell_type value, bodyIds, instance names, or regex patterns), computes the
N×N morphological similarity matrix at bodyId level
plus, unless the comparison IS at bodyId level, an aggregated matrix over
them, then writes CSV matrices, interactive heatmaps, and a report — the
morphology analogue of the Connectivity tab's Comparison sub-tab
(``ConnectivityProfileComparer``). ``aggregation_level`` picks the row
granularity: ``type`` (default), ``bodyid`` (one row per neuron) or
``custom`` (one row per LabelMapper group).

Two scoring methods:
- ``vector_v2`` (default): the production similarity of Find Similar —
  per-block cosine (shape/spatial 0.30/0.70) on standardized + ZCA-whitened
  vectors from the per-dataset ``SkeletonVectorCacheV2``.
- ``nblast``: canonical normalized NBLAST on raw-skeleton dotprops. Both
  orientations of every pair are scored and averaged (the forward score is
  asymmetric); aggregate means exclude contralateral pairs (mirror arbors
  score at chance), matching Find Similar's ipsilateral-only aggregation.

Intra-dataset ONLY: vectors and NBLAST dotprops are scored in one dataset's
coordinate space against that dataset's caches. Cross-dataset workflows
belong to the connectivity side.

Output folder (under ``output_dir``)::

    morphology_comparison_{DATASET}_{query}_{ts}/
      parameters.json
      README.txt
      members.csv                       # row / type / bodyId provenance
      type_level/type_similarity_{method}.csv      # group_level/group_similarity_…
                                                   # at the custom group level;
                                                   # absent at the bodyId level
      bodyid_level/bodyid_similarity_{method}.csv
      visualization/heatmap_type_{method}.html     # heatmap_group_… / absent
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
    from utils.naming_utils import (
        dataset_abbrev, load_labelmapper_source_groups)
except ImportError:  # pragma: no cover
    def dataset_abbrev(dataset: str) -> str:
        return str(dataset or "").split(":")[0].replace("_", "")[:5].upper()
    from naming_utils import load_labelmapper_source_groups

try:
    from comparison import report_kit
except ImportError:  # pragma: no cover - direct src/ execution
    import report_kit

try:
    from comparison.query_resolver import DatasetTaxonomyResolver
except ImportError:  # pragma: no cover - direct src/ execution
    from query_resolver import DatasetTaxonomyResolver


# Regex metacharacters that mark a query token as a pattern rather than an
# exact type name (mirrors the UI match modes: 'aMe.*', '.*KC.*', ...).
_PATTERN_CHARS = set("*?[](){}|^$.+\\")

# NBLAST builds a dotprop per neuron and scores every pair twice: O(N²)
# skeleton loads dominate quickly. Past this population the run WARNS
# (user 2026-09-29: disclose the cost, never refuse it) — the bound is the
# scored population, so a max_total_neurons cap below it is honored first.
NBLAST_WARN_NEURONS = 30

_METHOD_LABELS = {
    "vector_v2": "Vector (spatial, vector_v2)",
    "nblast": "NBLAST",
}

# Row granularity of the comparison matrix. "custom group" is accepted as an
# alias so the raw UI label works, mirroring the connectivity comparer.
_AGGREGATION_ALIASES = {
    "type": "type",
    "bodyid": "bodyid",
    "custom": "custom",
    "custom group": "custom",
}
_AGGREGATION_DEFAULT = "type"

# How a matrix row is described in the report, per level.
_ROW_KIND_LABELS = {"type": "type", "bodyid": "neuron", "custom": "group"}


def _normalize_aggregation_level(value) -> str:
    """`type` | `bodyid` | `custom`; empty/None takes the default.

    Unlike the connectivity comparer, an unknown value raises instead of
    silently becoming "type": a run whose rows are not what the user selected
    is worse than a refused run. The strictness is deliberately local —
    connectivity's fallthrough is pinned by its own coverage test.
    """
    if value is None or not str(value).strip():
        return _AGGREGATION_DEFAULT
    key = str(value).strip().lower()
    if key not in _AGGREGATION_ALIASES:
        raise ValueError(
            f"Invalid aggregation_level: {value!r} "
            "(type|bodyid|custom group)")
    return _AGGREGATION_ALIASES[key]


def _looks_like_pattern(token: str) -> bool:
    return any(ch in _PATTERN_CHARS for ch in str(token))


def _safe_name(value: str, limit: int = 60) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return safe[:limit] or "query"


class MorphologyProfileComparer:
    """Compare the morphology of already-identified neurons within ONE dataset.

    The query is a single list of neurons; the result is one N×N matrix whose
    row granularity ``aggregation_level`` selects:

    - ``type`` — each queried type is one row (diagonal = intra-type cohesion,
      the mean pairwise score among its members), and the entry is the mean
      over the cross-member bodyId pairs. The bodyId-level matrix, which
      carries every individual pair, is written alongside it.
    - ``bodyid`` — every individual neuron is its own row, so a queried
      bodyId is compared as itself rather than folded into its type. The
      bodyId matrix IS the comparison, so no type-level aggregate is written.
    - ``custom`` — rows are the source-side groups of a LabelMapper preset
      (``custom_mapping_file``); the query list is ignored.

    Each query token resolves in the same order the connectivity Comparison
    sub-tab uses: exact type name, bodyId, taxonomy label (a cell_type /
    class / subclass value such as FAFB's ``circadian_clock`` expands into
    one row per member type), instance name, and finally the regex-pattern
    interpretation.

    The bodyId × bodyId matrix is always the scored primitive: the other
    matrices are block means over it.

    A run needs at least two NEURONS in scope, not two rows: one type with
    several members answers "how similar are these neurons to each other", so
    it is a legitimate comparison rather than a refusal.
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        query: Optional[Union[str, int, List[Union[str, int]]]] = None,
        method: str = "vector_v2",
        aggregation_level: str = _AGGREGATION_DEFAULT,
        custom_mapping_file: Optional[str] = None,
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
        self.aggregation_level = _normalize_aggregation_level(aggregation_level)
        self.custom_mapping_file = str(custom_mapping_file or "").strip()
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
        # Resolution disclosures for user_warning_notes.txt: expansions,
        # instance matches, caps, and tokens nothing matched.
        self._resolution_notes: List[str] = []
        if self.method not in ("vector_v2", "nblast"):
            raise ValueError(
                f"Invalid method: {self.method} (vector_v2|nblast)")
        if is_banc_dataset(self.dataset):
            # User 2026-09-29: BANC comparison runs, with the caveat
            # disclosed — never gated. Scores come from the public L2/full
            # skeleton products whose vector quality is still unvalidated.
            self._note(
                "BANC morphology comparison runs on the public L2/full "
                "skeleton products, whose vector quality is still "
                "unvalidated: treat scores as provisional and spot-check "
                "surprising pairs in the 3D scene.")
        if self.custom_mapping_file and self.aggregation_level != "custom":
            self._log("custom_mapping_file given: aggregation level "
                      "switched to 'custom'.")
            self.aggregation_level = "custom"

    # ------------------------------------------------------------------ log
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MorphologyProfileComparer] {msg}", flush=True)

    def _note(self, msg: str) -> None:
        """Log a resolution disclosure AND keep it for user_warning_notes.txt
        (the same file the connectivity comparison writes; the run guide
        renders it)."""
        self._resolution_notes.append(str(msg))
        self._log(str(msg))

    def _body_id(self, value):
        return _canonical_dataset_body_id(self.dataset, value)

    # ------------------------------------------------------------- resolution
    def _resolve_members(self) -> "Dict[str, List[object]]":
        """Resolve the comparison population into {row label: [member bodyIds]}.

        The row label follows ``aggregation_level``: a type name, a bodyId
        display label, or a custom-group label. Values are always bodyIds, so
        the block-mean aggregation and the 3D layering work unchanged at every
        level. Neurons named directly in the query are *pinned*: no per-row cap
        and no total cap may drop them.
        """
        tokens = self._query_tokens()
        if not tokens and self.aggregation_level != "custom":
            raise ValueError(
                "Morphology comparison needs a query: enter one or more neuron "
                "types, bodyIds, or a pattern.")
        # Argument validation first: a missing preset must surface even when
        # the dataset has no local neuron table (fresh clones, CI).
        if (self.aggregation_level == "custom"
                and not self.custom_mapping_file):
            raise ValueError(
                "The custom group level needs a grouping preset "
                "(custom_mapping_file).")

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
        self._type_names = sorted(
            {str(t or "").strip() for t in type_map.values()
             if str(t or "").strip()})
        # Lazy per-run lookups for the taxonomy-label and instance-name
        # token kinds (both read the same local tables the type map uses).
        self._taxonomy_resolver: Optional[DatasetTaxonomyResolver] = None
        self._instance_lookup: Optional[Dict[str, List[object]]] = None

        self._pinned: set = set()
        if self.aggregation_level == "custom":
            members = self._custom_rows()
        else:
            members = self._query_rows(tokens)
            if self.aggregation_level == "bodyid":
                members = self._relabel_neuron_rows(members)

        self._check_population(members)
        return self._apply_total_cap(members)

    def _query_tokens(self) -> List[Union[str, int]]:
        """The query as an ordered list of non-blank tokens."""
        if self.query is None:
            tokens: List[Union[str, int]] = []
        elif isinstance(self.query, (str, int)):
            tokens = [self.query]
        else:
            tokens = list(self.query)
        return [t for t in tokens if str(t).strip()]

    def _type_member_ids(self, type_name: str) -> List[object]:
        """Every bodyId of one dataset type, in the stable string order."""
        ids = [bid for bid, t in self._type_map.items()
               if str(t or "").strip() == type_name]
        ids.sort(key=lambda b: str(b))
        return ids

    def _taxonomy_types(self, text: str) -> Optional[List[str]]:
        """Member type names when the token equals a taxonomy-column value
        (cell_type / class / subclass / cross-dataset type columns), else
        None — the connectivity comparison's first string lane.

        The shared ``DatasetTaxonomyResolver`` scans the same local neuron
        table the type map comes from; it is built lazily because that table
        is read a second time with the taxonomy columns included. Any failure
        degrades to None so the token falls through to the pattern lane.
        """
        if self._taxonomy_resolver is None:
            self._taxonomy_resolver = DatasetTaxonomyResolver(
                workspace_path=str(self.project_root),
                include_cross_dataset_type_columns=True)
        try:
            return self._taxonomy_resolver.resolve(text, self.dataset)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Warning: taxonomy lookup for {text!r} failed: {exc}")
            return None

    def _instance_member_ids(self, text: str) -> List[object]:
        """BodyIds whose instance name equals the token (exact, stripped)."""
        if self._instance_lookup is None:
            lookup: Dict[str, List[object]] = {}
            for bid, inst in self._instance_map.items():
                name = str(inst or "").strip()
                if name:
                    lookup.setdefault(name, []).append(bid)
            self._instance_lookup = lookup
        return self._instance_lookup.get(text, [])

    def _cap_members(self, ids: List[object], label: str) -> List[object]:
        """Cap one row's members, never at the expense of a pinned neuron.

        The cap bounds the UNPINNED members: with two or more queried neurons
        in one row and a small cap, truncating `pinned + rest` would evict a
        neuron the query named — the very defect this comparison fixes.
        """
        limit = self.max_members_per_type
        if len(ids) <= limit:
            return list(ids)
        forced = [b for b in ids if self._body_id(b) in self._pinned]
        rest = [b for b in ids if self._body_id(b) not in self._pinned]
        keep_rest = max(0, limit - len(forced))
        if len(forced) > limit:
            self._note(
                f"{label}: keeping all {len(forced)} queried neurons although "
                f"max_members_per_type is {limit}.")
        else:
            self._note(
                f"{label}: capping {len(ids)} members to {limit} "
                "(max_members_per_type).")
        return forced + rest[:keep_rest]

    def _query_rows(self, tokens: List[Union[str, int]]) -> "Dict[str, List[object]]":
        """Rows for the type and bodyId levels, keyed by canonical bodyId or
        type name (bodyId keys are relabelled once, in bulk, afterwards)."""
        rows: Dict[str, List[object]] = {}
        missing: List[str] = []

        def _add_ids(target_key: str, ids: List[object]) -> None:
            merged = rows.get(target_key, [])
            seen = {self._body_id(b) for b in merged}
            for bid in ids:
                canon = self._body_id(bid)
                if canon in seen:
                    continue
                seen.add(canon)
                merged.append(bid)
            rows[target_key] = self._cap_members(merged, target_key)

        def _add_type(type_name: str) -> None:
            ids = self._type_member_ids(type_name)
            if not ids:
                return
            if self.aggregation_level == "bodyid":
                # One row per neuron; the per-type cap bounds how many rows a
                # queried type contributes.
                for bid in self._cap_members(ids, type_name):
                    rows.setdefault(str(self._body_id(bid)), [bid])
                return
            _add_ids(type_name, ids)

        for token in tokens:
            text = str(token).strip()
            # Exact type names win over pattern interpretation (some dataset
            # type names contain regex metacharacters, e.g. "PPL1*").
            if text in self._type_names:
                _add_type(text)
                continue
            if text.isdigit():
                bid = self._body_id(text)
                type_name = str(self._type_map.get(bid, "") or "").strip()
                if not type_name:
                    raise ValueError(
                        f"bodyId {text} was not found in dataset "
                        f"'{self.dataset}'.")
                # The neuron the user asked for is protected from every cap.
                self._pinned.add(bid)
                if self.aggregation_level == "bodyid":
                    rows.setdefault(str(bid), [bid])
                    continue
                _add_ids(type_name, [bid] + self._type_member_ids(type_name))
                continue
            # Taxonomy label: a cell_type / class value (e.g. FAFB's
            # 'circadian_clock') expands into one row per member type. Like
            # exact type names, this exact-cell lane must run before the
            # pattern interpretation — and it mirrors the connectivity
            # comparison's first string lane.
            expanded = self._taxonomy_types(text)
            if expanded:
                shown = ", ".join(expanded[:8])
                if len(expanded) > 8:
                    shown += ", …"
                self._note(
                    f"'{text}' is a taxonomy label: expanded to "
                    f"{len(expanded)} dataset type(s) ({shown}).")
                for type_name in expanded:
                    _add_type(type_name)
                continue
            # Instance name: names specific neurons directly, so they are
            # pinned like bodyId queries and fold into their types at the
            # type level.
            inst_ids = self._instance_member_ids(text)
            if inst_ids:
                self._note(
                    f"'{text}' resolved to {len(inst_ids)} neuron(s) by "
                    "instance name.")
                for bid in inst_ids:
                    self._pinned.add(self._body_id(bid))
                if self.aggregation_level == "bodyid":
                    for bid in inst_ids:
                        rows.setdefault(str(self._body_id(bid)), [bid])
                    continue
                by_type: Dict[str, List[object]] = {}
                for bid in inst_ids:
                    type_name = str(self._type_map.get(
                        self._body_id(bid), "") or "").strip()
                    if type_name:
                        by_type.setdefault(type_name, []).append(bid)
                    else:
                        rows.setdefault(str(self._body_id(bid)), [bid])
                for type_name, bids in by_type.items():
                    _add_ids(type_name,
                             bids + self._type_member_ids(type_name))
                continue
            if _looks_like_pattern(text):
                try:
                    rx = re.compile(text)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid query pattern '{text}': {exc}")
                matched = [t for t in self._type_names if rx.fullmatch(t)]
                if not matched:
                    missing.append(text)
                    continue
                for type_name in matched:
                    _add_type(type_name)
                continue
            missing.append(text)

        if missing:
            self._note(
                "No dataset types matched: " + ", ".join(map(str, missing)))
        # ``rows`` is already in first-occurrence query order (dicts preserve
        # insertion order).
        return rows

    def _relabel_neuron_rows(self, rows: "Dict[str, List[object]]"):
        """Swap canonical-bodyId keys for the tree-legend display labels.

        Labels come from one bulk lookup; two neurons that would share a label
        get the type appended rather than merging into one row.
        """
        ids = [bid for members in rows.values() for bid in members]
        labels = self._display_labels(ids)
        out: Dict[str, List[object]] = {}
        for bid, label in zip(ids, labels):
            key = str(label)
            if key in out:
                key = f"{key}_{self._type_map.get(self._body_id(bid), '')}"
            out[key] = [bid]
        return out

    def _custom_rows(self) -> "Dict[str, List[object]]":
        """Rows from the grouping preset's source-side groups (query ignored).

        Group members are type names or bodyIds, both resolved against this
        dataset; a shared parser reads the preset so the two comparison tabs
        cannot disagree about what a group is.
        """
        if not self.custom_mapping_file:
            raise ValueError(
                "The custom group level needs a grouping preset "
                "(custom_mapping_file).")
        groups, names, _side = load_labelmapper_source_groups(
            self.custom_mapping_file, self.dataset, log=self._log)
        if not names:
            raise ValueError(
                f"No custom groups with members in '{self.dataset}' — check "
                "the grouping board's source mapping.")
        rows: Dict[str, List[object]] = {}
        for name, members in zip(names, groups):
            ids: List[object] = []
            for value in members:
                text = str(value).strip()
                if text.isdigit():
                    ids.append(self._body_id(text))
                elif text in self._type_names:
                    ids.extend(self._type_member_ids(text))
                else:
                    self._note(f"{name}: '{text}' is neither a type nor a "
                               f"bodyId in {self.dataset}; skipped")
            ordered = list(dict.fromkeys(self._body_id(b) for b in ids))
            if not ordered:
                self._note(f"{name}: no members in {self.dataset}; skipped")
                continue
            rows[name] = self._cap_members(ordered, name)
        return rows

    def _check_population(self, members: "Dict[str, List[object]]") -> None:
        """Neuron-count gate and NBLAST cost warning, before any scoring
        work starts.

        The gate counts NEURONS, not rows: one type with several members is a
        legitimate comparison. Its aggregate cell is that type's cohesion and
        its bodyId matrix holds the pairwise detail, so a second row was never
        what made the run worth doing.
        """
        total = sum(len(v) for v in members.values())
        if total < 2:
            kind = self._row_axis_label(
                'bodyid' if self.aggregation_level == 'bodyid' else 'type')
            hint = ""
            if self.aggregation_level == "bodyid" and self.max_members_per_type < 2:
                hint = (" 'Max Members per Type' is below 2, which leaves every "
                        "type at most one neuron to compare.")
            raise ValueError(
                f"Morphology comparison needs at least two neurons: the query "
                f"resolved {total} neuron(s) in {len(members)} {kind} row(s) at "
                f"the {self.aggregation_level} level.{hint}")
        if len(members) == 1:
            note = (f"Single {self._row_axis_label('type')} row "
                    f"'{next(iter(members))}': the aggregate matrix reports that "
                    "row's own cohesion, and the bodyId matrix carries the "
                    "pairwise detail between its neurons.")
            if self.aggregation_level == "type" and self._pinned:
                note += (" bodyId queries fold into their type at this level; set"
                         " Aggregation Level to 'bodyid' to keep the neurons as"
                         " separate rows.")
            self._log(note)
        # NBLAST scores every neuron pair twice, so large populations are
        # SLOW — disclosed, not refused (user 2026-09-29). The bound is the
        # population that would actually be scored — min(total,
        # max_total_neurons) — not the raw query size, so the warning names
        # the capped size a small max_total_neurons produces.
        effective = min(total, self.max_total_neurons)
        if self.method == "nblast" and effective > NBLAST_WARN_NEURONS:
            self._note(
                f"NBLAST comparison scores every neuron pair twice: "
                f"{effective} neurons is past the {NBLAST_WARN_NEURONS}-"
                "neuron comfort bound, so this run may take a very long "
                "time. Reduce the query or the per-type member cap"
                + (" (at the bodyId level each row is one neuron)"
                   if self.aggregation_level == "bodyid" else "")
                + ", or use the vector method if it is too slow.")

    def _apply_total_cap(self, members: "Dict[str, List[object]]"):
        """Bound the population — but never by dropping a queried neuron.

        Queried bodyIds are reserved first; the remaining budget fills with
        unpinned members in row order. If the reservation alone passes the cap
        the cap is exceeded deliberately and said out loud, because silently
        evicting a neuron the user named is the defect this comparison fixes.
        """
        total = sum(len(v) for v in members.values())
        if total <= self.max_total_neurons:
            return members
        limit = self.max_total_neurons
        kept: Dict[str, List[object]] = {label: [] for label in members}
        for label, ids in members.items():
            for bid in ids:
                if self._body_id(bid) in self._pinned:
                    kept[label].append(bid)
        reserved = sum(len(v) for v in kept.values())
        if reserved > limit:
            self._note(
                f"max_total_neurons={limit} is below the "
                f"{reserved} neuron(s) the query named; keeping every named "
                "neuron.")
        budget = max(0, limit - reserved)
        if total - reserved > budget:
            self._note(
                f"Query resolves to {total} neurons; truncating the "
                f"unpinned members to {limit} (max_total_neurons).")
        for label, ids in members.items():
            for bid in ids:
                if budget <= 0:
                    break
                if self._body_id(bid) in self._pinned:
                    continue
                kept[label].append(bid)
                budget -= 1
            if budget <= 0:
                break
        dropped = total - sum(len(v) for v in kept.values())
        if dropped:
            self._log(f"{dropped} unpinned neuron(s) dropped by "
                      "max_total_neurons.")
        return {label: ids for label, ids in kept.items() if ids}

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
        helper = MorphologyComparer(
            dataset=self.dataset, method="nblast",
            verbose=self.verbose, n_workers=self.n_workers,
            project_root=str(self.project_root))
        dps = helper._dotprops_for_ids(
            [self._body_id(b) for b in all_ids],
            desc="Building comparison dotprops")
        # dps is keyed by dataset-canonical ids (string bodyIds on FlyWire) —
        # the int() lookups this used before silently missed EVERY FlyWire
        # entry, so NBLAST compared nothing there.
        kept = [bid for bid in all_ids
                if dps.get(self._body_id(bid)) is not None]
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
            dp = dps[self._body_id(bid)]
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

    # --------------------------------------------------------- level wording
    def _level_display_labels(self) -> Dict[str, str]:
        """Titles for the two matrix panels.

        The internal level keys stay ``type``/``bodyid`` so filenames,
        ``csv_links`` and heatmap names are level-independent code; only the
        reader-facing wording moves with the aggregation level.
        """
        aggregate = ("Group level" if self.aggregation_level == "custom"
                     else "Type level")
        return {"type": aggregate, "bodyid": "BodyId level"}

    def _row_axis_label(self, level: str) -> str:
        """Axis title for one matrix panel."""
        if level == "bodyid":
            return "Neuron"
        return "Group" if self.aggregation_level == "custom" else "Type"

    def _aggregate_name(self) -> tuple:
        """``(folder, csv_stem, heatmap_stem)`` for the aggregate matrix.

        The internal level key stays ``"type"`` for the report plumbing, but
        the on-disk names describe the axes: a group×group matrix lives in
        ``group_level/``, not in a folder that claims types. Connectivity
        files bodyId-labelled matrices under ``type_level/``; this module
        deliberately does not do that to groups either.
        """
        if self.aggregation_level == "custom":
            return "group_level", "group_similarity", "group"
        return "type_level", "type_similarity", "type"

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
        """Report/heatmap presentation for this run's similarity metric.

        Both similarity metrics span [-1, 1] — a whitened cosine
        (vector_v2) and normalized NBLAST both genuinely score below zero
        (less similar than chance, verified at -0.88 on real FAFB pairs) —
        so both render on the shared kit's diverging blue-white-red scale
        over the full [-1, 1] domain, every run, whether or not this
        particular matrix happens to contain negative cells. A run-conditional
        scale would give the same metric different colormaps across runs.
        """
        return report_kit.MetricStyle(
            "morph_similarity",
            "Vector v2" if self.method == "vector_v2" else "NBLAST similarity",
            report_kit.REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)

    def _write_heatmaps(self, matrices: Dict[str, pd.DataFrame],
                        viz_dir: Path) -> List[str]:
        """Standalone interactive VisPath heatmaps (shared report_kit).

        Card keys are the METRIC key (the kit looks styles up by it); the
        level rides in the closures so the exported filenames stay
        ``heatmap_{level}_{method}.html``, with the aggregate panel named for
        its axes (``heatmap_group_*`` at the custom group level). VisPath
        renders with native clustering; any failure re-renders through the
        plotly fallback.
        """
        style = self._metric_style()
        styles = {style.key: style}
        method_label = _METHOD_LABELS.get(self.method, self.method)
        level_labels = self._level_display_labels()
        agg_heatmap = self._aggregate_name()[2]
        saved: Dict[str, List[str]] = {"heatmaps_generated": []}
        for level in ("type", "bodyid"):
            matrix = matrices.get(level)
            if matrix is None or matrix.empty:
                continue
            stem = agg_heatmap if level == "type" else level
            report_kit.generate_standalone_heatmaps(
                {self.dataset: {style.key: matrix}},
                viz_dir, styles,
                filename_builder=lambda group, key, _st=stem:
                    f"heatmap_{_st}_{self.method}.html",
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
        reports): hero header, one tab per matrix level this run computed
        (a bodyId-level run therefore shows the BodyId tab alone), each with
        Ward-clustered heatmap cards (CSV + VisPath editor links),
        compared-members and parameter details, and the 3D scene link.
        Plotly.js is embedded, so the report renders offline.
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

        row_kind = _ROW_KIND_LABELS[self.aggregation_level]
        level_blurb = {
            "type": (
                "Each queried type contributes one row — the diagonal is the "
                "intra-type cohesion and each entry the mean over the "
                "cross-member bodyId pairs; the bodyId level carries every "
                "individual pair."),
            "bodyid": (
                "Every individual neuron is its own row, so the bodyId matrix "
                "is the comparison itself and no type-level aggregate is "
                "written."),
            "custom": (
                "Each custom group is one row — the diagonal is the "
                "intra-group cohesion and each entry the mean over the "
                "cross-member bodyId pairs; the bodyId level carries every "
                "individual pair."),
        }[self.aggregation_level]
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
            f'({escape(method_label)} · {escape(self.aggregation_level)} '
            f'level). {escape(level_blurb)} Use the VisPath editor links to '
            'change clustering; hover cells for exact values.</p>',
            '<div class="report-meta">',
            chip('Dataset', str(self.dataset)),
            chip('Method', method_label),
            chip('Aggregation level', self.aggregation_level),
            chip(f'{row_kind.capitalize()} rows compared',
                 str(params.get("rows_compared", "—"))),
            chip('Neurons compared',
                 f"{compared}/{len(member_rows)}"),
            '</div>',
        ]
        metric_name = "vector_v2" if self.method == "vector_v2" else "NBLAST"
        scale_note = (f'both similarity metrics span [-1, 1] — negative means '
                      f'less similar than chance — so {metric_name} cells '
                      'render on the diverging blue-white-red scale over the '
                      'full [-1, 1] domain.')
        lines.append(
            f"<p class='report-note'>Generated "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} · {scale_note}</p>")
        lines.append('</header>')

        plotly_state = {'include_plotlyjs': True}
        level_labels = self._level_display_labels()
        agg_heatmap = self._aggregate_name()[2]

        def render_level(level: str, _level_panel_id: str) -> None:
            matrix = matrices[level]
            scored = int(matrix.notna().sum().sum())
            note = (f'{matrix.shape[0]}×{matrix.shape[1]} · {scored} '
                    'scored cells')
            if level == 'type':
                note += f' · diagonal = intra-{row_kind} cohesion'
            axis = self._row_axis_label(level)
            lines.append(
                f"<div class='direction-intro'><h3 class='direction-title'>"
                f'{escape(level_labels.get(level, level))}</h3>'
                f"<div class='direction-note'>{escape(note)}</div></div>")
            report_kit.append_report_metric_grid(
                lines, report_path.parent, {style.key: matrix},
                f'Intra-dataset · {level_labels.get(level, level)}',
                {style.key: csv_links.get(level, "")},
                {style.key: f'visualization/heatmap_'
                            f'{agg_heatmap if level == "type" else level}'
                            f'_{self.method}.html'},
                axis,
                axis,
                plotly_state,
                styles={style.key: style},
                square_cells=True,
            )

        # Only the levels this run actually computed get a tab: an
        # always-present "Type level" tab reading "not computed" invites the
        # reader to look for a missing input rather than a deliberate level.
        computed_levels = [
            (level, level_labels[level]) for level in ('type', 'bodyid')
            if matrices.get(level) is not None and not matrices[level].empty
        ]
        report_kit.append_report_tab_group(
            lines, 'morph-levels', computed_levels, render_level,
            panel_class='tab-panel direction-panel')

        # --- compared members + parameters details -----------------------
        detail_bits = []
        if member_rows:
            rows = "".join(
                f"<tr><td>{escape(str(m.get('row', '')))}</td>"
                f"<td>{escape(str(m['type']))}</td>"
                f"<td>{escape(str(m['bodyId']))}</td>"
                f"<td>{escape(str(m.get('instance', '')))}</td>"
                f"<td>{escape(str(m.get('status', '')))}</td></tr>"
                for m in member_rows)
            detail_bits.append(
                '<details class="detail-block"><summary>Compared '
                'neurons</summary><div style="overflow-x:auto">'
                "<table class='mapping-table'><thead><tr><th>row</th>"
                '<th>type</th><th>bodyId</th><th>instance</th>'
                '<th>status</th></tr>'
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
                '</a> — one layer per compared matrix row; the legend tree '
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
        """Render the compared neurons as one skeleton layer per matrix row.

        Mirrors Find Similar's 3D scene (line skeletons, interactive
        tree legend, template brain, no synapses); each compared row
        contributes one
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
        for rank, (row_label, ids) in enumerate(members.items(), start=1):
            canon_ids = [self._body_id(b) for b in ids]
            shown = canon_ids[:TYPE_RENDER_MEMBER_CAP]
            layers.append(shown)
            safe_type = _safe_name(row_label, 40)
            names.append(f"t{rank}_{safe_type}_x{len(shown)}")
            if len(canon_ids) > len(shown):
                notes.append(
                    f"t{rank}_{safe_type}: showing {len(shown)} of "
                    f"{len(canon_ids)} members of row '{row_label}' "
                    f"(per-layer render cap {TYPE_RENDER_MEMBER_CAP})")
        if not layers:
            self._log("3D visualization skipped: no compared neurons.")
            return None

        self._log(
            f"3D visualization: {len(layers)} compared row layer(s), "
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
            f"Comparing {len(members)} "
            f"{_ROW_KIND_LABELS[self.aggregation_level]} row(s) / {total} "
            f"neurons in {self.dataset} ({self.method}, "
            f"{self.aggregation_level} level).")

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
        # At the bodyId level the body matrix IS the comparison, so the block
        # mean is neither computed nor written — connectivity files its bodyId
        # rows under the type-level name; this one drops the level instead.
        aggregate_df = (
            None if self.aggregation_level == "bodyid"
            else self._type_level_matrix(body_matrix, labels, members))
        # BodyId rows read as '{bodyId}_{instance}' or
        # '{bodyId}_{type}_{L|R}' (the tree-legend rule). The aggregation above
        # resolves members by canonical bodyId, so the display relabel happens
        # only after it has run.
        display_labels = self._display_labels(all_ids)
        body_df.index = display_labels
        body_df.columns = display_labels

        # Name the run folder from what the user queried. At the bodyId level
        # the row labels are '{bodyId}_{instance}' display labels, and four of
        # them overflow the 60-char name budget with no information the query
        # did not already carry.
        name_source = ([str(t) for t in self._query_tokens()]
                       if self.aggregation_level == "bodyid"
                       else list(members.keys()))
        output_path = self._output_path(
            "_".join(str(t) for t in name_source[:4]))
        agg_dir, agg_stem, _agg_heatmap_stem = self._aggregate_name()
        (output_path / "bodyid_level").mkdir(parents=True, exist_ok=True)
        if aggregate_df is not None:
            (output_path / agg_dir).mkdir(parents=True, exist_ok=True)
            aggregate_df.to_csv(
                output_path / agg_dir
                / f"{agg_stem}_{self.method}.csv")
        body_df.to_csv(output_path / "bodyid_level"
                       / f"bodyid_similarity_{self.method}.csv")

        type_map = getattr(self, "_type_map", {}) or {}
        instance_map = getattr(self, "_instance_map", {}) or {}
        member_rows: List[Dict[str, object]] = []
        for row_label, ids in members.items():
            for bid in ids:
                canon = self._body_id(bid)
                member_rows.append({
                    # `row` is the matrix row this neuron belongs to: a type
                    # name, this neuron's own display label, or a group label.
                    # `type` stays the real type at every level, so neither
                    # column lies about the other.
                    "row": row_label,
                    "type": str(type_map.get(canon, "") or ""),
                    "bodyId": canon,
                    "instance": str(instance_map.get(canon, "") or ""),
                    "status": status_by_id.get(canon, "compared"),
                })
        pd.DataFrame(member_rows).to_csv(
            output_path / "members.csv", index=False)

        matrices: Dict[str, pd.DataFrame] = {"bodyid": body_df}
        if aggregate_df is not None:
            matrices["type"] = aggregate_df
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
            "aggregation_level": self.aggregation_level,
            "max_members_per_type": self.max_members_per_type,
            "max_total_neurons": self.max_total_neurons,
            "fetch_online": self.fetch_online,
            "rows_compared": len(members),
            "neurons_compared": int(sum(
                1 for m in member_rows if m["status"] == "compared")),
            "intra_dataset_only": True,
            "generate_heatmaps": self.generate_heatmaps,
            "visualize": self.visualize,
            "duration_s": round(time.time() - started, 1),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self.custom_mapping_file:
            params["custom_mapping_file"] = self.custom_mapping_file
        report_params = dict(params)
        report_params["_member_rows"] = member_rows
        if plot3d_link:
            report_params["_plot3d_link"] = plot3d_link
        (output_path / "parameters.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8")
        self._write_warning_notes(output_path)
        csv_links = {
            "bodyid": f"bodyid_level/bodyid_similarity_{self.method}.csv",
        }
        if aggregate_df is not None:
            csv_links["type"] = f"{agg_dir}/{agg_stem}_{self.method}.csv"
        self._write_report(
            output_path / "report.html", matrices,
            csv_links=csv_links,
            params=report_params)
        (output_path / "README.txt").write_text(
            self._readme_text(output_path, heatmap_files),
            encoding="utf-8")

        self._log(f"Output: {output_path}")
        return {
            "output_folder": str(output_path),
            "aggregation_level": self.aggregation_level,
            "rows_compared": len(members),
            "neurons_compared": params["neurons_compared"],
            "files": [str(p) for p in sorted(output_path.rglob("*"))
                      if p.is_file()],
        }

    def _write_warning_notes(self, output_path: Path) -> None:
        """user_warning_notes.txt (parity with the connectivity comparison):
        resolution disclosures — taxonomy expansions, instance matches, caps,
        and tokens nothing matched. The run guide renders this file when
        present, so a query that resolved through an expansion is disclosed
        next to the results it produced."""
        if not self._resolution_notes:
            return
        lines = ["User warning notes", "==================", "",
                 "How this run's query resolved (expansions, caps, and "
                 "tokens nothing matched):", ""]
        lines.extend(f"- {note}" for note in self._resolution_notes)
        (output_path / "user_warning_notes.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def _readme_text(self, output_path: Path,
                     heatmap_files: List[str]) -> str:
        level = self.aggregation_level
        aggregate = "group" if level == "custom" else "type"
        if level == "bodyid":
            semantics = (
                "Rows are individual neurons, so the bodyId matrix IS the "
                "comparison; no aggregate-level matrix is written for this "
                "run.")
        else:
            semantics = (
                f"{aggregate.capitalize()}-level entry = mean over the "
                "cross-member bodyId pairs; the diagonal is the "
                f"{aggregate}'s intra-{aggregate} cohesion (mean pairwise "
                "among its own members). The bodyId-level matrix carries "
                "every individual pair.")
        agg_dir, agg_stem, _agg_heatmap = self._aggregate_name()
        layout = [
            "  parameters.json",
            "  members.csv                    row / type / bodyId provenance",
        ]
        if (output_path / "user_warning_notes.txt").exists():
            layout.append(
                "  user_warning_notes.txt         how the query resolved "
                "(expansions / caps / unmatched tokens)")
        if level != "bodyid":
            layout.append(
                f"  {agg_dir}/{agg_stem}_{self.method}.csv"
                f"          {aggregate}×{aggregate} matrix")
        layout += [
            f"  bodyid_level/bodyid_similarity_{self.method}.csv"
            "     bodyId×bodyId matrix",
            "  visualization/heatmap_*.html   interactive heatmaps",
            "  plot-3d_<dataset>/             3D skeleton scene (when enabled;"
            " one layer per compared row)",
            "  report.html                    summary report",
        ]
        return f"""MORPHOLOGY COMPARISON — {self.dataset}
Generated {datetime.now().isoformat(timespec='seconds')}

Intra-dataset morphology comparison (method: {_METHOD_LABELS.get(self.method, self.method)},
aggregation level: {level}).
{semantics}

Output layout:
{chr(10).join(layout)}

bodyId-level rows/axes are labeled '{{bodyId}}_{{instance}}' (NeuPrint-style
datasets) or '{{bodyId}}_{{type}}_L/_R' (FAFB/BANC); members.csv maps every
label back to its raw bodyId.

Morphological comparison is intra-dataset only: skeletons are scored in one
dataset's coordinate space against that dataset's caches. Select two or
more datasets in the Comparison sub-tab to switch to the cross-dataset
comparison (morph_cross_dataset.CrossDatasetMorphComparer) instead.
"""
