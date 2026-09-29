"""
Morphological similarity for DROCAT.

Two comparison backends are provided:

- **Vector-based** (default, fast): each neuron is reduced to a fixed vector
  of ~24 L-Measure-style morphometrics plus a 100-dim persistence vector
  (navis). Vectors are cached per dataset as a single parquet file
  (``cache/{dataset}/find_similar/morphology/skeleton_vectors.parquet``) and
  queried with cosine / Pearson similarity. Raw skeleton files live in the
  shared dataset skeleton cache at
  ``cache/{dataset}/skeletons/raw_skeletons/`` as portable ``.swc.zst``
  files (zstd-19, simplification level recorded in the header; legacy
  ``.swc.gz`` remains readable). FAFB CAVE replacement trees live separately
  in ``cache/{dataset}/skeletons/cave_skeletons/`` and are selected for
  extrusion-repaired bodies. The former ``find_similar/raw_skeletons``
  directory remains a read-only migration fallback.
- **NBLAST** (navis implementation of Costa et al. 2016): the canonical
  pairwise morphology score. It uses the same raw skeleton/vector cache for
  prefiltering, while dotprops are rebuilt on demand for the vector-
  prefiltered candidate set of the current query.

Also provides ``enrich_homolog_results``, which attaches vector-based
morphological similarity scores to already-ranked homolog finding results
(post-search, result rows only).
"""

import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import navis
from tqdm import tqdm

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

from statvis import getNeurons

# Shared compressed-SWC provenance contract (headers, parsers).
try:
    from . import skeleton_provenance as _provenance
except ImportError:  # pragma: no cover - src laid bare on sys.path
    import skeleton_provenance as _provenance

try:
    from .roi_screening import (
        RoiProfileStore, RoiScreeningUnavailable, backfill_dataset_metadata,
        load_primary_rois, roi_count_table_path,
    )
except ImportError:
    from roi_screening import (
        RoiProfileStore, RoiScreeningUnavailable, backfill_dataset_metadata,
        load_primary_rois, roi_count_table_path,
    )

try:
    from .utils.flywire_readiness import (
        FlyWireSkeletonAccessError,
        flywire_manual_skeleton_instruction,
        is_fafb_dataset,
        require_flywire_skeleton_access,
    )
except ImportError:
    from utils.flywire_readiness import (
        FlyWireSkeletonAccessError,
        flywire_manual_skeleton_instruction,
        is_fafb_dataset,
        require_flywire_skeleton_access,
    )

try:
    from .flywire_ids import (
        body_id_to_api_int,
        is_banc_dataset,
        is_flywire_dataset,
        is_local_connectome_dataset,
        neuprint_dataset_name,
        normalize_flywire_body_id,
        normalize_flywire_body_ids,
        normalize_flywire_id_columns,
    )
except ImportError:
    from flywire_ids import (
        body_id_to_api_int,
        is_banc_dataset,
        is_flywire_dataset,
        is_local_connectome_dataset,
        neuprint_dataset_name,
        normalize_flywire_body_id,
        normalize_flywire_body_ids,
        normalize_flywire_id_columns,
    )

try:
    from .flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
    )
except ImportError:
    from flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
    )

try:
    from .visualization_options import default_analysis_skeleton_mesh_simplification
except ImportError:
    from visualization_options import default_analysis_skeleton_mesh_simplification

# Feature columns (morphometrics part of the vector). The full per-neuron
# vector is these 24 features + 100 persistence dimensions.
MORPHOMETRIC_FEATURES: List[str] = [
    "cable_length", "n_nodes", "n_branch", "n_leaf", "n_root", "n_primary",
    "max_branch_order", "mean_edge_length",
    "bbox_x", "bbox_y", "bbox_z", "bbox_diagonal",
    "max_path_length", "mean_path_length", "std_path_length", "tortuosity",
    "soma_radius", "strahler_max", "strahler_mean",
    "leaf_density", "branch_density", "bbox_xy_ratio", "path_ratio", "edge_cv",
]
PERSISTENCE_DIM = 100
VECTOR_DIM = len(MORPHOMETRIC_FEATURES) + PERSISTENCE_DIM

VECTOR_CACHE_VERSION = 1
RAW_SKELETON_CACHE_VERSION = 2

# =============================================================================
# V2 spatial vectorization (opt-in method="vector_v2")
# =============================================================================
# The V1 124-dim vector is purely shape-statistical: two neurons with the
# same global size/branching but arborized in different brain regions score
# identically.  The V2 schema appends two blocks to the SAME V1 features
# (nothing is removed, so "vector" results remain reproducible):
#   shape extras — cheap arbor-geometry scalars the V1 features omit
#                  (node-radius stats, branch angles, Strahler cable
#                  fractions);
#   spatial      — an arbor ellipsoid (PCA of node coordinates: centroid,
#                  spread, principal axis, anisotropy), a cable-mass
#                  histogram over a POPULATION-FIXED bounding box, and two
#                  1-D cable-mass profiles (radial from the centroid and
#                  midline-distance), so absolute position in the shared
#                  template space is preserved.
# A third, ROI-expansion block (Hellinger pre/post synapse fractions over
# the primary ROIs) is composed at runtime from ``RoiProfileStore`` when the
# dataset provides one — FAFB runs simply omit it. The former
# 24-dim Laplacian topology block was removed in cache schema v4: its
# eigensolver dominated vectorization time with an intermittent stall
# (seconds-to-minutes per neuron) while its 10% weight measurably changed
# nothing in the user-facing ranking (see the morphology benchmark).
SPATIAL_ELLIPSOID_FEATURES: List[str] = [
    "sp_com_x", "sp_com_y", "sp_com_z",
    "sp_ax1_len", "sp_ax2_len", "sp_ax3_len",
    "sp_ax1_x", "sp_ax1_y", "sp_ax1_z",
    "sp_anisotropy", "sp_flatness", "sp_radius",
]
SPATIAL_HIST_BINS = (6, 4, 4)          # x/y/z over the population bbox
SPATIAL_HIST_DIM = SPATIAL_HIST_BINS[0] * SPATIAL_HIST_BINS[1] * SPATIAL_HIST_BINS[2]
# Cheap shape extras (skeletons only; zeros for mesh-representation caches,
# which carry no radius/branch structure). Computed from the node table in
# O(k) — radius stats, sibling-edge branch angles at branch points, and
# cable-length fractions by the Strahler order already computed for
# strahler_max/strahler_mean.
SHAPE_EXTRA_FEATURES: List[str] = [
    "sx_radius_mean", "sx_radius_std", "sx_radius_max", "sx_radius_leaf_mean",
    "sx_branch_angle_mean", "sx_branch_angle_std",
    "sx_strahler_frac_1", "sx_strahler_frac_ge4",
]
SHAPE_EXTRA_DIM = len(SHAPE_EXTRA_FEATURES)
# 1-D cable-mass profiles over the population bbox: radial distance from
# the arbor centroid (rp, scale-free shape of the distribution) and
# absolute distance from the brain midline (md, region identity along the
# lateral axis). Both L1-normalized then Hellinger-transformed like the
# 96-bin histogram.
SPATIAL_PROFILE_RADIAL_BINS = 8
SPATIAL_PROFILE_MIDLINE_BINS = 8
SPATIAL_PROFILE_DIM = SPATIAL_PROFILE_RADIAL_BINS + SPATIAL_PROFILE_MIDLINE_BINS
SPATIAL_EXTRA_FEATURES: List[str] = (
    [f"rp_{i}" for i in range(SPATIAL_PROFILE_RADIAL_BINS)]
    + [f"md_{i}" for i in range(SPATIAL_PROFILE_MIDLINE_BINS)]
)
# FAFB bundle population seeding: cache-direct FlyWire searches only see
# locally vectorized neurons, so a representative random sample of the
# whole-brain skeleton bundle is vectorized once (and cached) whenever the
# population falls below this target.
# FAFB bundle population seeding is DISABLED by default: the search is
# candidate-list-first (connectivity screen -> fetch -> vectorize), so the
# cache no longer defines the search population. Set a positive target to
# opt back into a pre-seeded exploratory pool for cache-direct browsing.
FAFB_BUNDLE_SAMPLE_TARGET = 1500   # raw-basis whole-brain sample: the V2
# cache seeds itself from the healed zip so identity-space runs (Find
# Similar / cross-dataset targets) have population artifacts and a
# candidate pool entirely offline.
SPATIAL_BLOCK_DIM = len(SPATIAL_ELLIPSOID_FEATURES) + SPATIAL_HIST_DIM \
    + SPATIAL_PROFILE_DIM
SHAPE_BLOCK_DIM = VECTOR_DIM + SHAPE_EXTRA_DIM
VECTOR_V2_DIM = SHAPE_BLOCK_DIM + SPATIAL_BLOCK_DIM   # 132+124 = 256

# Column slices of the fixed V2 schema (the optional ROI block is appended
# beyond the fixed width and carried separately at scoring time).
V2_SHAPE_SLICE = (0, SHAPE_BLOCK_DIM)
V2_SPATIAL_SLICE = (SHAPE_BLOCK_DIM, VECTOR_V2_DIM)
# The 96-bin histogram's position inside the spatial block (after the 12
# ellipsoid features; the radial/midline profiles follow it).
SPATIAL_HIST_SLICE = (V2_SPATIAL_SLICE[0] + len(SPATIAL_ELLIPSOID_FEATURES),
                      V2_SPATIAL_SLICE[0] + len(SPATIAL_ELLIPSOID_FEATURES)
                      + SPATIAL_HIST_DIM)

# Per-block weights of the V2 score: sum(weight * cosine_block). When the
# ROI block is available its weight is added and ALL weights are renor-
# malized, so the defaults below describe the ROI-less (FAFB) case.
# The topology block was removed entirely in schema v4 (cost without
# ranking effect); ``v2_block_weights`` accepts only shape/spatial keys.
# Spatial-heavy 3:7 (unbiased-sample weight sweep 2026-08-30: +0.016 MRR
# over 5:5, Wilcoxon p = 0.039, AP@50 peak).
DEFAULT_V2_BLOCK_WEIGHTS = {"shape": 0.30, "spatial": 0.70}

# Two-pass type reevaluation (vector_v2): after the first scoring pass, the
# remaining (screen-ranked) members of the top-ranked candidate types join
# the pool and every type is re-aggregated on real coverage — a type's
# score must not be decided by 1-2 lucky screen survivors. The coverage
# weighting itself is continuous: similarity * sqrt(type_coverage), see
# ``_coverage_factor``.
DEFAULT_EXPAND_TOP_TYPES = 20
DEFAULT_EXPAND_PER_TYPE = 10

# Minimum standardized rows required to fit a population ZCA whitener; below
# this the V2 comparison falls back to z-scored per-block cosine.
MIN_ROWS_FOR_WHITENING = 64

VECTOR_CACHE_V2_VERSION = 6   # 6 = raw skeleton basis (simplification is visualization-only): vectors are computed from raw trees, never releveled to simp90; 5 = bbox_xy_ratio capped at 1e6 (planar arbors with zero y-span produced 1e13-scale features that dominated the population stats); 4 = topology block removed, +8 sx_* +16 spatial profile dims (rp_/md_)

# Bump when the whitening fit changes so persisted matrices are refit.
WHITEN_FIT_VERSION = 3

# Step-progress totals reported to the web UI during a similarity run
# (see the [DROCAT][progress] event protocol in ui/components/output_panel.py).
PROFILE_FIRST_TOTAL_STEPS = 6   # candidate-screen-first (NeuPrint) pipeline
CACHE_DIRECT_TOTAL_STEPS = 4    # vector-cache-direct (FlyWire) pipeline

# Candidate sources that discover a bounded pool before morphology scoring:
# 'profile' (connection-cache shared partners), 'roi' (primary-ROI
# distribution cosine with bilateral mirroring) and 'combined' (both).
CANDIDATE_SCREEN_SOURCES = ("profile", "roi", "combined")

# Members per type sampled for NBLAST type-level means. A scoring detail,
# not a candidate-list knob: pool size is controlled by ``candidate_cap``
# alone. Type-level 3D visualization layers use their own, larger cap
# (``TYPE_RENDER_MEMBER_CAP``).
TYPE_MEMBER_SAMPLE_CAP = 5

# Members per type sampled for type-level 3D visualization layers. Each
# result layer renders at most this many neurons; when a type has more,
# the render page carries an in-page truncation warning.
TYPE_RENDER_MEMBER_CAP = 20

# Maximum number of cached skeletons sampled for population standardization
# statistics when a dataset has no vector cache (see ``population_stats``).
POPULATION_STATS_SAMPLE = 3000

# A dataset whose skeleton cache holds fewer than this many neurons cannot
# estimate stable population statistics on its own; ``population_stats``
# then borrows them from a version sibling (e.g. male-cns:v1.0 <- v0.9).
MIN_POPULATION_STATS_SKELETONS = 300

# Vectorization level ("basis"). Every morphology vector cache is built from
# the raw skeleton returned by the source API. Visualization simplification is
# a render-time concern and never changes the on-disk skeleton representation.
# ``VECTOR_BASIS_SIMP90`` remains as a compatibility label for older callers;
# it is not a valid persisted skeleton-cache level anymore.
VECTOR_BASIS_RAW = "raw"
VECTOR_BASIS_SIMP90 = "simp90"
SKELETON_CACHE_LEVEL = VECTOR_BASIS_SIMP90   # legacy compatibility label
SKELETON_DOWNSAMPLE_FACTOR = 10             # legacy compatibility constant

# On-disk NeuPrint skeleton simplification: percent of nodes removed, 0-90.
# 90 = the canonical "simp90" cache (keep ~10% of nodes); 0 = raw. Every
# compressed-SWC cache file records its level in a ``# DROCAT simpl: N``
# header line so later loads can re-simplify to a different target level
# (only ever coarser; detail cannot be restored).
# Header lines and their parsers come from the shared provenance contract.
# ``# DROCAT simpl: N`` records the on-disk simplification level; the optional
# ``# DROCAT source: NAME`` line (e.g. ``banc_gcs_full``) records the fetch
# pipeline and is parsed on load so reloads keep the fetch-time metadata.
SIMPLIFICATION_HEADER = _provenance.SIMPLIFICATION_HEADER
SOURCE_HEADER = _provenance.SOURCE_HEADER
DEFAULT_SIMPLIFICATION = 90
MAX_SIMPLIFICATION = 90

# Keep the online NeuPrint fetch policy shared by visualization and similarity
# workflows.  The request is batched at the application boundary, while
# navis/NeuPrint performs the individual SWC requests in a small worker pool.
# This avoids one client/query setup per neuron without creating an unbounded
# nested thread pool.
NEUPRINT_FETCH_BATCH_SIZE = 64
NEUPRINT_FETCH_MAX_THREADS = 3


# =============================================================================
# Feature extraction
# =============================================================================

def _dataset_folder(dataset: str) -> str:
    """Map a dataset name to its cache folder (hemibrain:v1.2.1 -> hemibrain_v1_2_1)."""
    try:
        from utils.naming_utils import canonical_dataset_name
    except ImportError:  # pragma: no cover - src laid bare on sys.path
        from utils.naming_utils import canonical_dataset_name
    return canonical_dataset_name(dataset).replace(":", "_").replace(".", "_")


def _canonical_dataset_body_id(dataset: str, body_id):
    """Keep FlyWire IDs exact while retaining NeuPrint's integer contract."""

    if is_local_connectome_dataset(dataset):
        return normalize_flywire_body_id(body_id)
    return int(body_id)


def _api_dataset_body_id(dataset: str, body_id) -> int:
    """Convert a dataset body ID only at a numeric third-party API boundary."""

    if is_local_connectome_dataset(dataset):
        return body_id_to_api_int(body_id)
    return int(body_id)


def _load_flywire_soma_positions(
        dataset: str, root: Path,
        body_ids: Optional[List[Union[int, str]]] = None,
        ) -> Dict[str, np.ndarray]:
    """Load optional FAFB soma coordinates from the local neuron table."""
    folder = _dataset_folder(dataset)
    candidates = [
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.parquet",
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv",
    ]
    table = next((path for path in candidates if path.exists()), None)
    if table is None:
        return {}
    try:
        if table.suffix.lower() == ".parquet":
            columns = pd.read_parquet(table).columns.tolist()
            pos_col = next(
                (name for name in (
                    "position", "soma_position", "soma_pos", "somaLocation")
                 if name in columns),
                None,
            )
            if pos_col is None:
                return {}
            frame = pd.read_parquet(table, columns=["bodyId", pos_col])
        else:
            header = pd.read_csv(table, nrows=0).columns.tolist()
            pos_col = next(
                (name for name in (
                    "position", "soma_position", "soma_pos", "somaLocation")
                 if name in header),
                None,
            )
            if pos_col is None:
                return {}
            frame = pd.read_csv(table, usecols=["bodyId", pos_col])
    except Exception:
        # The soma table is optional: an unreadable one degrades to no
        # anchoring instead of failing the whole skeleton fetch.  Only the
        # table IO lives here — filtering bugs below must surface.
        return {}
    requested = (
        set(normalize_flywire_body_ids(body_ids))
        if body_ids is not None else None
    )
    out = {}
    for body_id, value in zip(frame["bodyId"], frame[pos_col]):
        key = normalize_flywire_body_id(body_id)
        if requested is not None and key not in requested:
            continue
        position = parse_soma_position(value)
        if position is not None:
            out[key] = position
    return out


def _has_local_dataset_presence(dataset: str, root: Path) -> bool:
    """Whether the dataset has real local data beyond a shipped index seed.

    The bundled neuron-index seeds only support search surfaces (auto-suggest,
    the available-neurons viewer, name resolution).  Skeleton workflows must
    not treat a seed as the authoritative neuron list of a pulled dataset:
    the index fallback is therefore only used when prepared tables, cached
    connections, or cached skeletons show the dataset is genuinely local, so
    a fresh clone never triggers bulk fetches from a seed alone.
    """
    folder = _dataset_folder(dataset)
    dataset_dir = root / "datasets" / folder
    if dataset_dir.is_dir() and any(dataset_dir.glob("*_neuron_df.*")):
        return True
    cache_dir = root / "cache" / folder
    if (cache_dir / "connections.parquet").exists():
        return True
    skeletons = cache_dir / "skeletons"
    if skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in skeletons.rglob("*")):
        return True
    raw_skeletons = cache_dir / "skeletons" / "raw_skeletons"
    if raw_skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in raw_skeletons.rglob("*")):
        return True
    meshes = cache_dir / "meshes"
    if meshes.is_dir() and any(
            p.suffix == ".pkl" or str(p).endswith(".pkl.zst")
            for p in meshes.rglob("*")):
        return True
    legacy_raw_skeletons = cache_dir / "find_similar" / "raw_skeletons"
    if legacy_raw_skeletons.is_dir() and any(
            p.suffix == ".pkl"
            or str(p).endswith((".pkl.zst", ".swc.gz", ".swc.zst"))
            for p in legacy_raw_skeletons.rglob("*")):
        return True
    return False


def compute_morphometrics(neuron) -> Dict[str, float]:
    """Compute the curated L-Measure-style morphometric feature set.

    All features are computed directly from ``TreeNeuron.nodes`` (vectorized
    numpy where possible) so results are deterministic and fast; navis is only
    used for the persistence vector (see ``compute_persistence_vector``).
    """
    df = neuron.nodes
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    parent = df["parent_id"].to_numpy(dtype=np.int64)
    node_ids = df["node_id"].to_numpy(dtype=np.int64)
    has_radius = "radius" in df.columns
    radius = df["radius"].to_numpy(dtype=float) if has_radius else np.zeros(len(df))
    n = len(df)

    # Root detection must use "parent not in the node set": SWC parent ids are
    # 0-based in navis tables, so parent_id == 0 is a VALID edge (not a root).
    is_root = ~np.isin(parent, node_ids)

    # Parent index lookup (robust to unsorted node ids).
    id_to_idx = {int(i): k for k, i in enumerate(node_ids)}
    pidx = np.array([id_to_idx.get(int(p), -1) for p in parent], dtype=np.int64)
    has_parent = ~is_root

    # Edge lengths (parent -> node).
    seg = coords[has_parent] - coords[pidx[has_parent]]
    edge_len = np.linalg.norm(seg, axis=1)
    edge_len_by_child = np.zeros(n)
    edge_len_by_child[has_parent] = edge_len
    cable = float(edge_len.sum())

    # Child counts.
    child_count = np.bincount(pidx[has_parent], minlength=n)
    n_branch = int((child_count >= 2).sum())
    n_leaf = int((child_count == 0).sum())
    n_root = int(is_root.sum())
    n_primary = int(is_root[pidx[has_parent]].sum()) if has_parent.any() else 0

    # Tree traversal: depth, path length from root, root id per node.
    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        if has_parent[i]:
            children[pidx[i]].append(i)
    roots = np.where(is_root)[0].tolist()
    depth = np.zeros(n)
    path_len = np.zeros(n)
    root_of = np.zeros(n, dtype=np.int64)
    stack = list(roots)
    for r in roots:
        root_of[r] = r
    while stack:
        i = stack.pop()
        for c in children[i]:
            depth[c] = depth[i] + 1
            path_len[c] = path_len[i] + edge_len_by_child[c]
            root_of[c] = root_of[i]
            stack.append(c)

    # Leaf statistics (tips).
    leaves = np.where(child_count == 0)[0]
    if leaves.size:
        straight = np.linalg.norm(
            coords[leaves] - coords[root_of[leaves]], axis=1
        )
        tortuosity = float(np.mean(
            path_len[leaves] / np.maximum(straight, 1e-9)
        ))
        mean_path = float(path_len[leaves].mean())
        std_path = float(path_len[leaves].std())
    else:
        tortuosity, mean_path, std_path = 1.0, 0.0, 0.0

    # Strahler order via reverse depth processing. A node with a single
    # child inherits the child's order; with >= 2 children it increments only
    # when the two highest child orders are equal.
    strahler = np.ones(n)
    for i in sorted(range(n), key=lambda k: -depth[k]):
        vals = [strahler[c] for c in children[i]]
        if not vals:
            strahler[i] = 1
        elif len(vals) == 1:
            strahler[i] = vals[0]
        else:
            top2 = sorted(vals, reverse=True)[:2]
            strahler[i] = top2[0] + 1 if top2[0] == top2[1] else top2[0]

    # Soma radius (0 when absent).
    soma_radius = 0.0
    try:
        soma = neuron.soma
        if soma is not None:
            soma_ids = [soma] if np.isscalar(soma) else list(soma)
            radii = [float(radius[id_to_idx[int(s)]]) for s in soma_ids if int(s) in id_to_idx]
            if radii:
                soma_radius = float(np.mean(radii))
    except Exception:
        pass

    bbox = np.ptp(coords, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))
    mean_edge = float(edge_len.mean()) if edge_len.size else 0.0
    edge_std = float(edge_len.std()) if edge_len.size else 0.0

    features = {
        "cable_length": cable,
        "n_nodes": float(n),
        "n_branch": float(n_branch),
        "n_leaf": float(n_leaf),
        "n_root": float(n_root),
        "n_primary": float(n_primary),
        "max_branch_order": float(depth.max()) if n else 0.0,
        "mean_edge_length": mean_edge,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_diagonal": bbox_diag,
        "max_path_length": float(path_len.max()) if n else 0.0,
        "mean_path_length": mean_path,
        "std_path_length": std_path,
        "tortuosity": tortuosity,
        "soma_radius": soma_radius,
        "strahler_max": float(strahler.max()) if n else 0.0,
        "strahler_mean": float(strahler.mean()) if n else 0.0,
        "leaf_density": n_leaf / cable if cable > 0 else 0.0,
        "branch_density": n_branch / cable if cable > 0 else 0.0,
        "bbox_xy_ratio": min(float(bbox[0]) / max(float(bbox[1]), 1e-9),
                       1e6),  # degenerate flat arbors: bounded, not astronomical
        "path_ratio": (float(path_len.max()) if n else 0.0) / max(bbox_diag, 1e-9),
        "edge_cv": edge_std / max(mean_edge, 1e-9),
    }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in features.items()}


def compute_persistence_vector(neuron) -> np.ndarray:
    """Return the 100-dim persistence vector (navis), zeros on failure."""
    try:
        pv = navis.morpho.persistence_vectors(neuron, samples=PERSISTENCE_DIM)
        arr = np.asarray(pv[0]).ravel()  # primary root vector
        if arr.size != PERSISTENCE_DIM:
            return np.zeros(PERSISTENCE_DIM)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    except Exception:
        return np.zeros(PERSISTENCE_DIM)


# =============================================================================
# V2 spatial/topological feature extraction (method="vector_v2")
# =============================================================================

def _neuron_points(neuron) -> np.ndarray:
    """Node (skeleton) or vertex (mesh) coordinates as an (n, 3) array."""
    if hasattr(neuron, "nodes") and neuron.nodes is not None and len(neuron.nodes):
        return np.asarray(neuron.nodes[["x", "y", "z"]].to_numpy(), dtype=float)
    if hasattr(neuron, "vertices") and neuron.vertices is not None \
            and len(neuron.vertices):
        return np.asarray(neuron.vertices, dtype=float)
    return np.zeros((0, 3), dtype=float)


def _sign_stable(vec: np.ndarray) -> np.ndarray:
    """Flip a vector so its largest-magnitude component is positive.

    Eigenvectors have arbitrary sign; this makes the stored axis deterministic
    for a given geometry (comparisons must not depend on LAPACK sign choices).
    """
    v = np.asarray(vec, dtype=float).ravel()
    if not v.size or not np.isfinite(v).all():
        return np.zeros_like(v)
    return -v if v[np.argmax(np.abs(v))] < 0 else v


def compute_spatial_ellipsoid(neuron) -> np.ndarray:
    """12-dim arbor-spread descriptor from the PCA of neuron coordinates.

    [centroid(3), axis spreads sqrt(eigenvalues)(3), principal axis(3),
    anisotropy, flatness, radius(rms distance to the centroid)]. Captures
    WHERE the arbor sits in the template space and HOW it expands — the
    expansion signal V1 lacks. Zeros when no coordinates are available.
    """
    pts = _neuron_points(neuron)
    out = np.zeros(len(SPATIAL_ELLIPSOID_FEATURES))
    if len(pts) < 3:
        return out
    com = pts.mean(axis=0)
    centered = pts - com
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return out
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)[::-1]          # descending: a1 >= a2 >= a3
    evals = evals[order]
    evecs = evecs[:, order]
    spread = np.sqrt(evals)                   # units of the template space
    a1, a2, a3 = (float(s) for s in spread)
    axis1 = _sign_stable(evecs[:, 0])
    radius = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    out[:] = [
        com[0], com[1], com[2],
        a1, a2, a3,
        axis1[0], axis1[1], axis1[2],
        (a1 - a2) / a1 if a1 > 0 else 0.0,
        (a2 - a3) / a1 if a1 > 0 else 0.0,
        radius,
    ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _neuron_edges(neuron):
    """(coords, edge index pairs) of the skeleton graph.

    Mirrors ``compute_morphometrics``: the parent column is ``parent_id``
    and roots are parents that are not themselves nodes (SWC parent ids are
    0-based, so ``parent_id == 0`` can be a VALID edge)."""
    df = neuron.nodes
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    node_ids = df["node_id"].to_numpy()
    parent = df["parent_id"].to_numpy(dtype=np.int64)
    id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
    edges = []
    for i, p in enumerate(parent):
        j = id_to_idx.get(int(p), -1)
        if j < 0:
            continue   # root (or dangling parent): no edge
        edges.append((j, i))
    return coords, edges


def _cable_mass_points(neuron) -> Tuple[np.ndarray, np.ndarray]:
    """Edge midpoints and lengths carrying the neuron's cable mass.

    Skeletons weight each parent->child edge by its length (mass = cable);
    meshes fall back to unit-weight vertex counts (surface sampling).
    """
    if hasattr(neuron, "nodes") and neuron.nodes is not None \
            and len(neuron.nodes) and "parent_id" in neuron.nodes.columns:
        coords, edges = _neuron_edges(neuron)
        if not edges:
            return coords, np.ones(len(coords))
        mids = np.asarray([(coords[a] + coords[b]) / 2.0 for a, b in edges])
        lens = np.asarray([float(np.linalg.norm(coords[a] - coords[b]))
                           for a, b in edges])
        lens[lens <= 0] = 1.0
        return mids, lens
    pts = _neuron_points(neuron)
    return pts, np.ones(len(pts))


def compute_spatial_histogram_abs(neuron,
                                  bounds: Optional[np.ndarray] = None
                                  ) -> np.ndarray:
    """Cable-mass histogram over a POPULATION-FIXED bounding box.

    ``bounds`` is [lo(3), hi(3)] shared by every neuron of a dataset (cache
    meta), so bin k means the same brain region for all neurons — unlike a
    per-neuron normalized grid, absolute position is preserved. Weights are
    cable length (skeletons) or unit vertex mass (meshes), L1-normalized and
    sqrt-transformed (Hellinger) for cosine. Zeros when bounds are unknown.
    """
    hist = np.zeros(SPATIAL_HIST_DIM)
    if bounds is None:
        return hist
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3):
        return hist
    mids, weights = _cable_mass_points(neuron)
    if not len(mids):
        return hist
    lo, hi = bounds[0], bounds[1]
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    total = float(weights.sum())
    if total <= 0:
        return hist
    frac = np.clip((mids - lo) / span, 0.0, 1.0 - 1e-9)
    idx = np.floor(frac * np.asarray(SPATIAL_HIST_BINS)).astype(np.int64)
    idx = np.clip(idx, 0, np.asarray(SPATIAL_HIST_BINS) - 1)
    flat = (idx[:, 0] * SPATIAL_HIST_BINS[1]
            + idx[:, 1]) * SPATIAL_HIST_BINS[2] + idx[:, 2]
    hist = np.bincount(flat, weights=weights / total,
                       minlength=SPATIAL_HIST_DIM)
    return np.sqrt(np.maximum(hist, 0.0))


def compute_shape_extras(neuron) -> np.ndarray:
    """8 cheap arbor-geometry extras (skeletons; zeros for meshes).

    Complements the curated V1 morphometrics with node-radius statistics
    (the V1 feature set only touches radius via ``soma_radius``),
    sibling-edge branch angles at branch points, and cable-length
    fractions by the Strahler order computed here with the same
    reverse-depth idiom as ``compute_morphometrics``. All O(k) from the
    node table; meshes carry no radius/branch structure and return zeros
    (same convention as ``soma_radius``).
    """
    out = np.zeros(SHAPE_EXTRA_DIM)
    if not (hasattr(neuron, "nodes") and neuron.nodes is not None
            and len(neuron.nodes)
            and "parent_id" in neuron.nodes.columns):
        return out
    df = neuron.nodes
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    parent = df["parent_id"].to_numpy(dtype=np.int64)
    node_ids = df["node_id"].to_numpy(dtype=np.int64)
    n = len(df)
    radius = (df["radius"].to_numpy(dtype=float)
              if "radius" in df.columns else np.zeros(n))
    radius = np.where(np.isfinite(radius) & (radius > 0), radius, 0.0)

    is_root = ~np.isin(parent, node_ids)
    id_to_idx = {int(i): k for k, i in enumerate(node_ids)}
    pidx = np.array([id_to_idx.get(int(p), -1) for p in parent],
                    dtype=np.int64)
    has_parent = ~is_root

    # radius statistics (0 when the source carries no radii)
    pos = radius > 0
    if pos.any():
        r = radius[pos]
        out[0] = float(r.mean())
        out[1] = float(r.std())
        out[2] = float(r.max())

    child_count = np.bincount(pidx[has_parent], minlength=n) \
        if has_parent.any() else np.zeros(n, dtype=np.int64)
    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        if has_parent[i]:
            children[pidx[i]].append(i)

    # mean radius at leaf nodes (tips are often behaviorally distinct)
    leaves = np.where(child_count == 0)[0]
    leaf_r = radius[leaves]
    leaf_r = leaf_r[leaf_r > 0]
    if leaf_r.size:
        out[3] = float(leaf_r.mean())

    # branch angles: at each branch node, the angle between the unit
    # vectors to each pair of children
    angles: List[float] = []
    for i in np.where(child_count >= 2)[0]:
        vecs = coords[children[i]] - coords[i]
        norms = np.linalg.norm(vecs, axis=1)
        keep = norms > 1e-9
        if keep.sum() < 2:
            continue
        units = vecs[keep] / norms[keep][:, None]
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                cos = float(np.clip(units[a] @ units[b], -1.0, 1.0))
                angles.append(np.degrees(np.arccos(cos)))
    if angles:
        out[4] = float(np.mean(angles))
        out[5] = float(np.std(angles))

    # cable-length fractions by Strahler order (edge Strahler = child's)
    depth = np.zeros(n)
    stack = [i for i in range(n) if is_root[i]]
    while stack:
        i = stack.pop()
        for c in children[i]:
            depth[c] = depth[i] + 1
            stack.append(c)
    strahler = np.ones(n)
    for i in sorted(range(n), key=lambda k: -depth[k]):
        vals = [strahler[c] for c in children[i]]
        if not vals:
            strahler[i] = 1
        elif len(vals) == 1:
            strahler[i] = vals[0]
        else:
            top2 = sorted(vals, reverse=True)[:2]
            strahler[i] = top2[0] + 1 if top2[0] == top2[1] else top2[0]
    seg = coords[has_parent] - coords[pidx[has_parent]]
    edge_len = np.linalg.norm(seg, axis=1)
    child_str = strahler[np.where(has_parent)[0]]
    total = float(edge_len.sum())
    if total > 0:
        out[6] = float(edge_len[child_str == 1].sum() / total)
        out[7] = float(edge_len[child_str >= 4].sum() / total)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_spatial_profile_extras(neuron,
                                   bounds: Optional[np.ndarray] = None
                                   ) -> np.ndarray:
    """16-dim 1-D cable-mass profiles (see ``SPATIAL_EXTRA_FEATURES``).

    ``rp_0..7``: radial distance from the arbor centroid in 8 equal bins
    over [0, max distance] — a scale-free shape-of-distribution summary
    complementing the absolute-position histogram. ``md_0..7``: absolute
    distance from the brain midline (x mid of the population bbox) in 8
    equal bins over [0, x span] — region identity along the lateral axis.
    Weights are cable length (skeletons) or unit vertex mass (meshes),
    L1-normalized then Hellinger-transformed like the 96-bin histogram.
    Zeros when bounds are unknown.
    """
    out = np.zeros(SPATIAL_PROFILE_DIM)
    if bounds is None:
        return out
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3):
        return out
    mids, weights = _cable_mass_points(neuron)
    if not len(mids):
        return out
    total = float(weights.sum())
    if total <= 0:
        return out
    centroid = mids.mean(axis=0)
    dist = np.linalg.norm(mids - centroid, axis=1)
    r_max = float(dist.max())
    if r_max > 0:
        frac = np.clip(dist / r_max, 0.0, 1.0 - 1e-9)
        idx = np.floor(frac * SPATIAL_PROFILE_RADIAL_BINS).astype(np.int64)
        prof = np.bincount(idx, weights=weights / total,
                           minlength=SPATIAL_PROFILE_RADIAL_BINS)
        out[:SPATIAL_PROFILE_RADIAL_BINS] = np.sqrt(
            np.maximum(prof, 0.0))
    mid_x = 0.5 * (float(bounds[0][0]) + float(bounds[1][0]))
    span_x = float(bounds[1][0] - bounds[0][0])
    if span_x > 0:
        dmid = np.abs(mids[:, 0] - mid_x)
        frac = np.clip(dmid / span_x, 0.0, 1.0 - 1e-9)
        idx = np.floor(frac * SPATIAL_PROFILE_MIDLINE_BINS).astype(np.int64)
        prof = np.bincount(idx, weights=weights / total,
                           minlength=SPATIAL_PROFILE_MIDLINE_BINS)
        out[SPATIAL_PROFILE_RADIAL_BINS:] = np.sqrt(
            np.maximum(prof, 0.0))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _lateral_reflected(neuron, mid_x: float):
    """View of ``neuron`` reflected about the x = mid_x plane.

    Returns a lightweight object whose node/vertex table carries the
    reflected x coordinates; the input neuron is never mutated. The V1
    shape features are reflection-invariant, so only the spatial features
    are computed from this view."""
    if hasattr(neuron, "nodes") and neuron.nodes is not None:
        import types
        nodes = neuron.nodes.copy()
        nodes["x"] = 2.0 * mid_x - nodes["x"]
        return types.SimpleNamespace(nodes=nodes)
    if hasattr(neuron, "vertices") and neuron.vertices is not None:
        import types
        verts = np.asarray(neuron.vertices, dtype=float).copy()
        verts[:, 0] = 2.0 * mid_x - verts[:, 0]
        return types.SimpleNamespace(vertices=verts)
    return neuron


def vectorize_neuron_v2(neuron,
                        spatial_bounds: Optional[np.ndarray] = None,
                        lateral_normalize: bool = False
                        ) -> Tuple[Dict[str, float], np.ndarray]:
    """Return (24-feature dict, 256-dim V2 vector).

    The first 124 dims are the V1 vector (identical schema and values); the
    appended 8 shape-extra dims carry arbor geometry the V1 features omit
    (radius stats, branch angles, Strahler cable fractions); the appended
    124 spatial dims carry position (ellipsoid + population-bbox
    histogram + radial/midline cable-mass profiles). ``spatial_bounds``
    comes from the cache's population estimate; without it the histogram
    and midline-profile blocks are zeros (ellipsoid/radial still apply).

    ``lateral_normalize`` reflects right-hemisphere arbors (centroid x past
    the bounds midline) onto the left BEFORE the spatial features are
    computed. Stored this way, every neuron of a dataset is represented on
    one hemisphere, so type-level aggregation can no longer average L and R
    positions into a midline blur; L/R homologs compare directly. The V1
    prefix and shape extras are reflection-invariant and unaffected, and
    the radial/midline profiles are mirror-invariant too.
    """
    morph, v1 = vectorize_neuron(neuron)
    spatial_view = neuron
    if lateral_normalize and spatial_bounds is not None:
        mid_x = 0.5 * (float(spatial_bounds[0][0]) + float(spatial_bounds[1][0]))
        pts = _neuron_points(neuron)
        if len(pts) and float(np.mean(pts[:, 0])) > mid_x:
            spatial_view = _lateral_reflected(neuron, mid_x)
    vector = np.concatenate([
        v1,
        compute_shape_extras(neuron),
        compute_spatial_ellipsoid(spatial_view),
        compute_spatial_histogram_abs(spatial_view, spatial_bounds),
        compute_spatial_profile_extras(spatial_view, spatial_bounds),
    ])
    if vector.shape[0] != VECTOR_V2_DIM:   # defensive: never mis-slice blocks
        raise ValueError(
            f"V2 vectorization produced {vector.shape[0]} dims, "
            f"expected {VECTOR_V2_DIM}")
    return morph, vector


# =============================================================================
# Mesh-based features (for datasets cached as MeshNeurons, e.g. FAFB)
# =============================================================================

def compute_mesh_morphometrics(mesh) -> Dict[str, float]:
    """24 morphometrics computed directly from a navis MeshNeuron.

    Uses the same feature names as the skeleton set so both neuron types
    share one vector schema (the shape block differs: persistence for
    skeletons, a spatial histogram for meshes).
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(vertices)

    # Triangle geometry.
    tri = vertices[faces]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    edge_lens = np.concatenate([
        np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1),
    ])
    tri_area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    area = float(tri_area.sum())

    # Bounding box.
    bbox = np.ptp(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))

    # Centroid statistics.
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)

    features = {
        "cable_length": float(edge_lens.sum()),  # total edge length (mesh analog)
        "n_nodes": float(n),
        "n_branch": float(len(faces)),
        "n_leaf": float(len(np.unique(faces))),
        "n_root": 0.0,
        "n_primary": 0.0,
        "max_branch_order": 0.0,
        "mean_edge_length": float(edge_lens.mean()) if edge_lens.size else 0.0,
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_diagonal": bbox_diag,
        "max_path_length": float(radii.max()) if n else 0.0,
        "mean_path_length": float(radii.mean()) if n else 0.0,
        "std_path_length": float(radii.std()) if n else 0.0,
        "tortuosity": 1.0,
        "soma_radius": 0.0,
        "strahler_max": 0.0,
        "strahler_mean": 0.0,
        "leaf_density": area / max(bbox_diag**2, 1e-9),  # surface compactness
        "branch_density": len(faces) / max(area, 1e-9),  # face density
        "bbox_xy_ratio": min(float(bbox[0]) / max(float(bbox[1]), 1e-9),
                       1e6),  # degenerate flat arbors: bounded, not astronomical
        "path_ratio": float(radii.max()) / max(bbox_diag, 1e-9) if n else 0.0,
        "edge_cv": (float(edge_lens.std()) / max(float(edge_lens.mean()), 1e-9))
        if edge_lens.size else 0.0,
    }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in features.items()}


def compute_spatial_histogram(mesh, bins: Tuple[int, int, int] = (5, 5, 4)) -> np.ndarray:
    """Coarse 3D shape descriptor: vertex counts in a normalized grid.

    Default 5x5x4 = 100 bins, matching the persistence-vector dimension so
    skeleton and mesh vectors share one schema.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not len(vertices):
        return np.zeros(PERSISTENCE_DIM)
    lo = vertices.min(axis=0)
    span = np.ptp(vertices, axis=0)
    span = np.where(span > 0, span, 1.0)
    norm = (vertices - lo) / span
    idx = np.floor(norm * np.array(bins)).astype(int)
    idx = np.clip(idx, 0, np.array(bins) - 1)
    flat = idx[:, 0] * (bins[1] * bins[2]) + idx[:, 1] * bins[2] + idx[:, 2]
    hist = np.bincount(flat, minlength=bins[0] * bins[1] * bins[2]).astype(float)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def vectorize_neuron(neuron) -> Tuple[Dict[str, float], np.ndarray]:
    """Return (24-feature dict, full 124-dim vector).

    Dispatches on the neuron type: TreeNeurons use skeleton morphometrics +
    a persistence vector; MeshNeurons (FlyWire bulk caches) use mesh
    morphometrics + a 100-bin spatial histogram. Both share one schema.
    """
    if hasattr(neuron, "nodes") and neuron.nodes is not None and len(neuron.nodes):
        morph = compute_morphometrics(neuron)
        shape = compute_persistence_vector(neuron)
    elif hasattr(neuron, "vertices") and neuron.vertices is not None and len(neuron.vertices):
        morph = compute_mesh_morphometrics(neuron)
        shape = compute_spatial_histogram(neuron)
    else:
        raise ValueError(
            "Unsupported neuron type for vectorization "
            f"({type(neuron).__name__}): expected a skeleton or a mesh."
        )
    vector = np.concatenate([
        np.array([morph[f] for f in MORPHOMETRIC_FEATURES], dtype=float),
        shape,
    ])
    return morph, vector


def _neuron_rep(neuron) -> str:
    """Representation of a neuron: 'skeleton' or 'mesh' ('' otherwise).

    Skeletons and meshes produce different feature semantics in the shared
    124-dim schema, so a comparison must never mix the two (nor two
    simplification levels of the same kind).
    """
    if hasattr(neuron, "nodes") and neuron.nodes is not None and len(neuron.nodes):
        return "skeleton"
    if hasattr(neuron, "vertices") and neuron.vertices is not None and len(neuron.vertices):
        return "mesh"
    return ""


def _skeleton_body_id(path) -> int:
    """Extract a bodyId from a cached skeleton filename.

    The cache accepts historical ``.pkl`` files, compressed mesh pickles
    (``{bodyId}.pkl.zst``), and the portable compressed-SWC forms
    (``{bodyId}.swc.zst`` / ``{bodyId}.swc.gz``). Keeping this parser in one
    place prevents ``Path.stem`` from turning ``123.swc.gz`` or
    ``123.pkl.zst`` into an invalid bodyId during parallel cache builds.
    """
    name = Path(path).name
    if name.endswith(".swc.zst"):
        name = name[:-len(".swc.zst")]
    elif name.endswith(".swc.gz"):
        name = name[:-len(".swc.gz")]
    elif name.endswith(".pkl.zst"):
        name = name[:-len(".pkl.zst")]
    elif name.endswith(".pkl"):
        name = name[:-len(".pkl")]
    return int(name)


def _load_cached_skeleton_file(path, target_simplification: Optional[int] = None):
    """Load a legacy pickle, compressed mesh pickle, or compressed SWC.

    Compressed-SWC files record their simplification level
    (``# DROCAT simpl: N``); when ``target_simplification`` is given and is
    coarser than the stored level, the neuron is re-simplified to that
    target on load.  The stored level is attached as
    ``neuron._drocat_simplification`` (headerless legacy files read as raw,
    level 0).
    """
    import io

    path = Path(path)
    stored = 0
    source = ""
    if str(path).endswith(".swc.zst"):
        if zstd is None:
            raise ImportError(
                "zstandard is required to read .swc.zst caches")
        with path.open("rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read()
        stored = _read_stored_simplification(content)
        source = _read_stored_source(content)
        neuron = navis.read_swc(io.StringIO(
            content.decode("utf-8", "replace")))
        try:
            neuron.id = _skeleton_body_id(path)
        except Exception:
            pass
        # SWC has no units field. Local skeleton caches use nanometers, so
        # restore that convention after a compressed-SWC round trip.
        try:
            neuron.units = "nm"
        except Exception:
            pass
    elif str(path).endswith(".swc.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            content = handle.read()
        stored = _read_stored_simplification(content)
        source = _read_stored_source(content)
        neuron = navis.read_swc(io.StringIO(content))
        try:
            neuron.id = _skeleton_body_id(path)
        except Exception:
            pass
        try:
            neuron.units = "nm"
        except Exception:
            pass
    elif str(path).endswith(".pkl.zst"):
        if zstd is None:
            raise ImportError(
                "zstandard is required to read FlyWire .pkl.zst caches")
        with path.open("rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                return pickle.load(reader)
    else:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if target_simplification is not None:
        neuron = _relevel_for_target(neuron, stored, target_simplification)
    try:
        neuron._drocat_simplification = stored
    except Exception:
        pass
    if source:
        try:
            neuron._drocat_source = source
        except Exception:
            pass
    return neuron


def _simplification_factor(simplification: int) -> int:
    """Downsample factor for a simplification level (percent removed, 0-90).

    90 -> factor 10 (keep ~10% of nodes), 50 -> 2, 0 -> 1 (raw).  Values
    outside 0..90 raise ``ValueError``; the factor is floored to an int
    (the canonical downsample helper floors too).
    """
    try:
        simplification = int(simplification)
    except (TypeError, ValueError):
        raise ValueError(
            f"simplification must be an integer in 0..{MAX_SIMPLIFICATION} "
            f"(percent of nodes removed); got {simplification!r}")
    if not 0 <= simplification <= MAX_SIMPLIFICATION:
        raise ValueError(
            f"simplification must be an integer in 0..{MAX_SIMPLIFICATION} "
            f"(percent of nodes removed); got {simplification!r}")
    if simplification == 0:
        return 1
    return max(1, 100 // (100 - simplification))


def _read_stored_simplification(text) -> int:
    """Parse the simplification level recorded in a compressed-SWC header.

    Header line format: ``# DROCAT simpl: 50``. Headerless files (legacy
    gzip cache, pickles) default to raw (level 0).
    """
    return _provenance.read_stored_simplification(text)


def _read_stored_source(text) -> str:
    """Parse the source provenance recorded in a compressed-SWC header.

    Header line format: ``# DROCAT source: banc_gcs_full``.  Files without
    a source line (most datasets, legacy caches) return ``''``.
    """
    return _provenance.read_stored_source(text)


def _relevel_for_target(neuron, stored: int, target: int):
    """Re-simplify a cached neuron from its stored level to a coarser target.

    Only ever simplifies further (``target > stored``); requesting more
    detail than the file holds returns the neuron unchanged (the caller may
    fetch raw online instead).  Factor = (100-stored)/(100-target): a 50%
    file re-leveled to 90% keeps 20% of its remaining nodes ("simplify by
    80%").  Never mutates the input.
    """
    stored = max(0, int(stored))
    _simplification_factor(target)  # validate 0..90
    target = int(target)
    # MeshNeurons and anything without a node table are never re-leveled:
    # the simplification pipeline is NeuPrint/TreeNeuron-only.
    if not hasattr(neuron, "nodes"):
        return neuron
    if target <= stored or target <= 0 or stored >= MAX_SIMPLIFICATION:
        return neuron
    factor = (100 - stored) / (100 - target)
    if factor <= 1:
        return neuron
    return _downsample_for_cache(neuron, factor)


def _write_compressed_skeleton(path, neuron,
                               simplification: Optional[int] = DEFAULT_SIMPLIFICATION,
                               codec: str = "zst",
                               codec_level: int = 19) -> None:
    """Shared simplify + compress pipeline for the on-disk skeleton cache.

    Every raw-SWC cache write routes through this function so the recorded
    simplification level and the compression format stay consistent.

    - ``simplification`` int 0-90: deterministically simplifies the neuron
      to that level (percent removed) before writing; ``0`` = raw.
      ``0`` records the level already attached to the neuron (a tree
      loaded from a simplified file is never relabeled as raw) — the
      on-disk cache stores raw skeletons; simplification is applied at
      visualization/render time, never at cache time.
    - ``simplification=None``: writes the neuron as-is and records the level
      already attached to it (``neuron._drocat_simplification``; absent = 0)
      - used by lazy migrations that must not re-simplify.
    - ``codec_level``: zstd compression level (default 19).  The transient
      ``_temp_cache`` staging writer uses level 3: measured ~2.8x faster than
      level 19 (Step 0), which lets the staging stage keep up with the fetch
      rate; the standard loader decompresses any level.

    The level is recorded as the first SWC header line
    (``# DROCAT simpl: N``) and the payload is atomically written as
    ``.swc.zst`` (zstd-19) or the legacy ``.swc.gz`` (gzip-6) form.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if simplification is None or simplification == 0:
        stored = getattr(neuron, "_drocat_simplification", 0) or 0
        neuron_out = neuron
    else:
        factor = _simplification_factor(simplification)
        stored = int(simplification)
        neuron_out = (_downsample_for_cache(neuron, factor)
                      if simplification > 0 else neuron)
    temp_swc = None
    temp_out = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tempfile.NamedTemporaryFile(
                suffix=".swc", dir=str(path.parent), delete=False) as handle:
            temp_swc = Path(handle.name)
        navis.write_swc(neuron_out, temp_swc, write_meta=True)
        payload = temp_swc.read_bytes()
        # Record the level in the header so later loads can re-level.
        payload = b"# DROCAT simpl: %d\n" % int(stored) + payload
        if codec == "zst":
            if zstd is None:
                raise ImportError(
                    "zstandard is required to write .swc.zst caches")
            blob = zstd.ZstdCompressor(
                level=codec_level, write_content_size=True).compress(payload)
        else:
            blob = gzip.compress(payload, compresslevel=6, mtime=0)
        temp_out.write_bytes(blob)
        os.replace(temp_out, path)
    finally:
        if temp_swc is not None:
            temp_swc.unlink(missing_ok=True)
        temp_out.unlink(missing_ok=True)


def _write_compressed_swc(path, neuron) -> None:
    """Legacy raw gzip SWC writer (level 0, gzip-6).

    Retained for older callers/tests; new writes go through
    :func:`_write_compressed_skeleton` (``.swc.zst``, recorded level).
    """
    _write_compressed_skeleton(path, neuron, simplification=0, codec="gz")


def _vectorize_one_file(path: str) -> Optional[Tuple[int, List[float], List[float], str]]:
    """Module-level worker for parallel cache builds (picklable).

    Returns None for un-vectorizable files (corrupt or unexpected types)
    so a single bad file cannot break the whole build. The 4th element is
    the neuron representation ('skeleton' | 'mesh')."""
    try:
        neuron = _load_cached_skeleton_file(path)
        # Per-file level guard: only raw (level 0) on-disk skeletons are
        # vectorized into the raw-basis cache; simplified files are skipped
        # (their vectors come from the vector cache / raw fetches instead).
        if getattr(neuron, "_drocat_simplification", 0) != 0:
            return None
        morph, vector = vectorize_neuron(neuron)
        rep = _neuron_rep(neuron)
    except Exception:
        return None
    body_id = _skeleton_body_id(path)
    # Shape block (persistence / spatial histogram) = the tail after the
    # morphometric block.
    return (body_id, [morph[f] for f in MORPHOMETRIC_FEATURES],
            vector[len(MORPHOMETRIC_FEATURES):].tolist(), rep)


# ---------------------------------------------------------------------------
# Healed-bundle workers (FAFB full-dataset vectorization)
# ---------------------------------------------------------------------------
# The FAFB healed zip ({bodyId}.swc entries) is the full skeleton source
# for FAFB v783: the local pickle cache holds meshes, which is the wrong
# representation for the vector cache. Workers open the zip once per
# process; the central-directory read is the expensive part, per-id reads
# are cheap, and nothing is written back (zip-only mode).

_FAFB_WORKER_BUNDLE = None


def _init_fafb_zip_worker(source_path: Optional[str],
                          zip_path: Optional[str] = None):
    """Per-worker initializer: open the healed skeleton source once.

    ``source_path`` is a legacy .zst container (opened read-only) when no
    zip exists; zip-only mode passes None and serves straight from the ZIP.
    """
    global _FAFB_WORKER_BUNDLE
    from fafb_bundle import FAFBSkeletonBundle

    _FAFB_WORKER_BUNDLE = FAFBSkeletonBundle(
        source_path, zip_path=zip_path, lazy_convert=False)


def _vectorize_one_swc(body_id: int
                       ) -> Optional[Tuple[int, List[float], List[float], str]]:
    """Module-level worker: vectorize one healed-zip skeleton."""
    global _FAFB_WORKER_BUNDLE
    import io

    try:
        content = _FAFB_WORKER_BUNDLE.get(int(body_id))
        if content is None:
            return None
        neuron = navis.read_swc(io.StringIO(content))
        neuron.units = "nm"
        morph, vector = vectorize_neuron(neuron)
    except Exception:
        return None
    return (int(body_id), [morph[f] for f in MORPHOMETRIC_FEATURES],
            vector[len(MORPHOMETRIC_FEATURES):].tolist(), "skeleton")


# V2 workers (method="vector_v2"): same sources, full 256-dim schema. The
# population spatial bounds travel through the pool initializer / argument
# so every worker bins the histogram identically.
_V2_WORKER_BOUNDS = None


def _init_v2_swc_worker(source_path: str, zip_path: Optional[str], bounds):
    global _FAFB_WORKER_BUNDLE, _V2_WORKER_BOUNDS
    _init_fafb_zip_worker(source_path, zip_path)
    _V2_WORKER_BOUNDS = bounds


def _vectorize_one_file_v2(path: str,
                           bounds=None
                           ) -> Optional[Tuple[int, List[float], str]]:
    """Module-level V2 worker for one cached skeleton/mesh file."""
    global _V2_WORKER_BOUNDS
    try:
        neuron = _load_cached_skeleton_file(path)
        # Raw basis: only level-0 (raw) trees are vectorized; simplified
        # files are skipped (their bodies are re-fetched raw on demand).
        if (getattr(neuron, "_drocat_simplification", 0) or 0) != 0:
            return None
        used_bounds = bounds if bounds is not None else _V2_WORKER_BOUNDS
        _, vector = vectorize_neuron_v2(neuron, used_bounds,
                                        lateral_normalize=True)
        rep = _neuron_rep(neuron)
    except Exception:
        return None
    return (_skeleton_body_id(path), vector.tolist(), rep)


def _vectorize_one_swc_v2(body_id: int) -> Optional[Tuple[int, List[float], str]]:
    """Module-level V2 worker for one healed-zip skeleton."""
    global _FAFB_WORKER_BUNDLE, _V2_WORKER_BOUNDS
    import io

    try:
        content = _FAFB_WORKER_BUNDLE.get(int(body_id))
        if content is None:
            return None
        neuron = navis.read_swc(io.StringIO(content))
        neuron.units = "nm"
        _, vector = vectorize_neuron_v2(neuron, _V2_WORKER_BOUNDS,
                                        lateral_normalize=True)
    except Exception:
        return None
    return (int(body_id), vector.tolist(), "skeleton")


def _fafb_bundle(dataset: str,
                 project_root: Optional[str] = None):
    """Healed-zip reader for FAFB v783 (zip-only; no ``.zst`` is created).

    Returns a :class:`fafb_bundle.FAFBSkeletonBundle` when the healed zip
    exists, else None.  A legacy ``.zst`` container is opened read-only
    only when no zip is present.
    """
    from fafb_bundle import open_bundle as _open_fafb_bundle

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    return _open_fafb_bundle(root / "datasets" / folder)


def _bundle_tree_neuron(bundle, body_id: int):
    """TreeNeuron for a healed-zip body id."""
    import io

    text = bundle.get(int(body_id))
    if text is None:
        return None
    nrn = navis.read_swc(io.StringIO(text))
    nrn.units = "nm"
    nrn.id = int(body_id)
    nrn.name = str(int(body_id))
    return nrn


def _extrusion_fix_fetcher(dataset: str, root: str):
    """Token-free fetcher for the ``extrusion_fixes`` local repair store."""
    try:
        from cave_data_fetcher import CAVEDataFetcher
        return CAVEDataFetcher(
            dataset=_dataset_folder(dataset), cave_token="",
            cache_enabled=True, project_root=str(root), verbose=False)
    except Exception:
        return None


def _flywire_cave_skeletons(dataset: str, body_ids,
                            project_root: Optional[str] = None,
                            log=None, denoise_twigs: Optional[float] = None
                            ) -> Dict[int, object]:
    """FAFB-only CAVE fallback: mesh -> wavefront skeletonize -> cache.

    Returns ``{int(body_id): TreeNeuron}`` for the ids that resolved. Only
    the skeletonized tree is cached in the dedicated ``cave_skeletons``
    replacement store; the raw mesh itself is not cached here, so the
    morphology chain never depends on the prepared mesh cache nor writes
    into ``raw_skeletons``.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    say = log or (lambda _message: None)
    ids = sorted({int(b) for b in body_ids})
    if not ids:
        return {}

    # A previously repaired CAVE tree is a complete local source. Probe it
    # before requiring a token so missing-from-bundle bodies can also be
    # served offline on later runs.
    out: Dict[int, object] = _load_cave_cached_skeletons(
        dataset, ids, str(root), log=say)
    ids = [bid for bid in ids if bid not in out]
    if not ids:
        return out

    try:
        from utils.flywire_readiness import flywire_skeleton_readiness
        cave_token = bool(
            flywire_skeleton_readiness(dataset, root).get("cave_token"))
    except Exception:
        cave_token = False
    if not cave_token:
        say(f"FAFB skeletons: {len(ids)} body id(s) unavailable locally "
            "and CAVE_TOKEN is not configured: "
            f"{ids[:5]}{'...' if len(ids) > 5 else ''}")
        return out
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(
        dataset=_dataset_folder(dataset), project_root=str(root),
        verbose=False,
    )
    for bid in ids:
        try:
            neuron = fetcher.fetch_skeleton(
                body_id_to_api_int(bid), use_cache=True,
                denoise_twigs=denoise_twigs)
        except Exception as exc:
            neuron = None
            say(f"FAFB skeletons: CAVE fetch failed for {bid}: {exc}")
        if neuron is not None:
            out[bid] = neuron
    unresolved = [b for b in ids if b not in out]
    if unresolved:
        say(f"FAFB skeletons: {len(unresolved)} body id(s) unavailable "
            f"from cache, bundle, and CAVE: "
            f"{unresolved[:5]}{'...' if len(unresolved) > 5 else ''}")
    return out


def _load_cave_cached_skeletons(
        dataset: str, body_ids, project_root: Optional[str] = None,
        log=None) -> Dict[int, object]:
    """Load CAVE replacement trees without a token or network access.

    CAVE replacements are deliberately kept separate from the raw healed
    bundle mirror.  This probe is used by morphology workflows before their
    token-gated fallback so a previously repaired body remains usable offline.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    say = log or (lambda _message: None)
    ids = sorted({int(b) for b in body_ids})
    if not ids:
        return {}

    try:
        from cave_data_fetcher import CAVEDataFetcher
    except ImportError:
        try:
            from .cave_data_fetcher import CAVEDataFetcher
        except ImportError:
            return {}

    try:
        # An empty token is intentional: this is a strictly local probe and
        # must remain usable when the CAVE API is not configured.
        fetcher = CAVEDataFetcher(
            dataset=_dataset_folder(dataset), cave_token="",
            cache_enabled=True, project_root=str(root), verbose=False,
        )
    except Exception as exc:
        say(f"FAFB skeletons: CAVE cache probe unavailable ({exc})")
        return {}

    out: Dict[int, object] = {}
    for bid in ids:
        try:
            neuron = fetcher.load_cached_skeleton(body_id_to_api_int(bid))
        except Exception:
            neuron = None
        if isinstance(neuron, navis.TreeNeuron):
            neuron.id = body_id_to_api_int(bid)
            neuron.name = str(bid)
            neuron.units = "nm"
            out[bid] = neuron
    return out


def load_local_release_skeletons(dataset: str, body_ids,
                                 project_root: Optional[str] = None,
                                 log=None, check_extrusions: bool = True,
                                 denoise_twigs: Optional[float] = None
                                 ) -> Dict[int, object]:
    """Load FAFB/BANC local-release raw skeletons through one pipeline.

    FAFB priority per body id:

        1. previously repaired trees, served network-free from the local
           repair caches before the zip is ever read: ``api_repaired``
           from ``cave_skeletons``, ``local_fallback`` from
           ``cave_skeletons`` then ``extrusion_fixes``,
        2. the shared raw skeleton cache (legacy frozen reads; the loader
           never writes new entries),
        3. the healed FAFB zip (offline full-dataset source),
        4. the extrusion check on the tree sources — one-time per body
           id, recorded in ``extrusion_check_results.parquet``; flagged
           neurons are replaced through the CAVE API (cached in
           ``cave_skeletons``) and, when CAVE is unavailable, by a locally
           pruned fix persisted to ``extrusion_fixes``,
        5. the token-gated CAVE fallback for everything still missing:
           the CAVE mesh is skeletonized (wavefront) into a TreeNeuron and
           cached into the dedicated ``cave_skeletons`` replacement store.

    The prepared mesh cache is never consulted: the morphology comparison
    is TreeNeuron-native (vector_v2 vectorization and NBLAST dotprops),
    and this loader guarantees every returned neuron is a skeleton.

    BANC skips the FAFB zip/extrusion/CAVE stages and resolves missing
    entries directly from the public BANC release bucket. NeuPrint datasets
    are not handled here (they use
    :func:`fetch_skeleton_on_demand`). Returns ``{int(body_id): neuron}``
    for every id that resolved; ids unavailable everywhere are absent.
    """
    ids = sorted({int(b) for b in body_ids})
    if not ids:
        return {}
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    say = log or (lambda _message: None)
    fafb = is_fafb_dataset(dataset)

    loaded: Dict[int, object] = {}
    pre_repaired: set = set()

    # 1. Previously repaired trees from the local repair caches (status-
    #    driven probe; the extruded zip tree is never loaded for them).
    if fafb:
        try:
            from cave_data_fetcher import load_repaired_skeletons
            repaired = load_repaired_skeletons(
                dataset, ids, project_root=str(root), log=say)
        except Exception as exc:
            repaired = {}
            say(f"FAFB skeletons: repair-cache probe failed ({exc}).")
        if repaired:
            loaded.update(repaired)
            pre_repaired = set(repaired)

    # 2. Shared raw cache (legacy frozen reads).
    try:
        raw_cache = find_similar_raw_cache(
            dataset, project_root=str(root), verbose=False)
    except Exception:
        raw_cache = None
    missing: List[int] = [bid for bid in ids if bid not in loaded]
    if raw_cache is not None:
        for bid in missing:
            try:
                neuron = raw_cache.load_skeleton(bid)
            except Exception:
                neuron = None
            if neuron is not None:
                loaded[bid] = neuron
        missing = [bid for bid in missing if bid not in loaded]

    # 2. Standalone BANC public-release source.  BANC never falls through to
    # the FAFB zip or the CAVE compatibility path.
    if missing and is_banc_dataset(dataset):
        try:
            import banc_public_data
            for bid in missing:
                try:
                    neuron = banc_public_data.fetch_banc_swc(
                        dataset, bid, project_root=str(root), use_cache=True)
                except Exception as exc:
                    neuron = None
                    say(f"BANC skeleton fetch failed for {bid}: {exc}")
                if neuron is not None:
                    loaded[bid] = neuron
        except ImportError as exc:
            say(f"BANC public-release fetcher unavailable: {exc}")
        missing = [bid for bid in missing if bid not in loaded]

    # 3. Healed FAFB zip (offline full-dataset source; read-only).
    if missing and fafb:
        try:
            bundle = _fafb_bundle(dataset, str(root))
        except Exception:
            bundle = None
        if bundle is not None:
            still_missing: List[int] = []
            try:
                for bid in missing:
                    neuron = _bundle_tree_neuron(bundle, bid)
                    if neuron is None:
                        still_missing.append(bid)
                    else:
                        loaded[bid] = neuron
            finally:
                try:
                    bundle.close()
                except Exception:
                    pass
            bundle_hits = len(missing) - len(still_missing)
            say(f"FAFB skeletons: healed zip resolved {bundle_hits}/"
                f"{len(missing)} body id(s); {len(still_missing)} left for "
                "the CAVE fallback.")
            missing = still_missing

    # 4. Extrusion check on the tree sources (one-time per body id; results
    #    cached in extrusion_check_results.parquet). Flagged neurons are
    #    replaced through CAVE and cached in cave_skeletons; when CAVE is
    #    unavailable, a locally pruned fix is persisted to extrusion_fixes.
    #    Trees already served from the repair caches above are final.
    if check_extrusions and fafb and loaded:
        from fafb_utils import (
            EXTRUSION_REPAIR_API_FAILED,
            EXTRUSION_REPAIR_API_REPAIRED,
            EXTRUSION_REPAIR_LOCAL_FALLBACK,
            flag_extrusions,
            repair_extruded_skeleton,
            set_extrusion_repair_status,
        )
        folder = _dataset_folder(dataset)
        try:
            flagged = flag_extrusions(str(root), folder, loaded, log=say)
        except Exception as exc:
            flagged = []
            say(f"FlyWire skeletons: extrusion check failed ({exc}); "
                "serving the local trees unchecked.")
        to_replace = sorted(
            b for b in flagged if b in loaded and b not in pre_repaired)
        if to_replace:
            say(f"FlyWire skeletons: replacing {len(to_replace)} "
                "extrusion-flagged neuron(s) through the CAVE API.")
            replaced = _flywire_cave_skeletons(
                dataset, to_replace, str(root), log=say,
                denoise_twigs=denoise_twigs)
            fix_fetcher = _extrusion_fix_fetcher(dataset, str(root))
            updates = {}
            for bid in to_replace:
                if bid in replaced:
                    loaded[bid] = replaced[bid]
                    updates[bid] = EXTRUSION_REPAIR_API_REPAIRED
                    continue
                # CAVE unavailable: prune the extruded subtree locally and
                # persist the fix so later runs never redo the work.
                fixed = None
                if fix_fetcher is not None:
                    try:
                        fixed, fix_stats = repair_extruded_skeleton(
                            loaded[bid])
                    except Exception:
                        fixed, fix_stats = None, {}
                    if fixed is not None and not fix_stats.get("repaired"):
                        fixed = None
                    if fixed is not None and not (
                            fix_fetcher.save_extrusion_fix_skeleton(
                                bid, fixed)):
                        fixed = None
                if fixed is not None:
                    fixed.units = "nm"
                    loaded[bid] = fixed
                    updates[bid] = EXTRUSION_REPAIR_LOCAL_FALLBACK
                    say(f"FlyWire skeletons: local extrusion prune applied "
                        f"and cached for {bid}.")
                else:
                    # Keep the flagged local tree; status stays retryable.
                    updates[bid] = EXTRUSION_REPAIR_API_FAILED
            try:
                set_extrusion_repair_status(str(root), folder, updates)
            except Exception as exc:
                say(f"FlyWire skeletons: could not save extrusion repair "
                    f"status: {exc}")

    # 5. Token-gated CAVE fallback for missing FAFB skeletons only.
    if missing and fafb:
        replaced = _flywire_cave_skeletons(
            dataset, missing, str(root), log=say, denoise_twigs=denoise_twigs)
        loaded.update(replaced)
    return loaded


def _resolve_fafb_skeleton_trees(dataset: str, body_ids,
                                 project_root: Optional[str] = None,
                                 log=None,
                                 check_extrusions: bool = False
                                 ) -> Dict[int, object]:
    """Resolve FAFB skeletons through the shared local loader.

    Thin wrapper for callers that want the loader's build-time posture in
    one call: it routes through :func:`load_local_release_skeletons`
    (repair caches -> raw cache -> healed zip -> CAVE) and returns a plain
    ``{int: TreeNeuron}`` map.  (The on-demand fetchers delegate to the
    same loader for FAFB now; historically they served prepared meshes
    here, which is the defect class behind
    ``plan-tmvev-reverse-scene-loader-defects.md``.)

    ``check_extrusions`` defaults to False so build-time vectorization stays
    offline; the status-driven repair-cache stage in the loader still applies
    regardless.  Returns ``{}`` on any failure or for non-FAFB datasets.
    """
    if not is_fafb_dataset(dataset):
        return {}
    try:
        resolved = load_local_release_skeletons(
            dataset, body_ids, project_root=project_root, log=log,
            check_extrusions=check_extrusions)
    except Exception:
        return {}
    out: Dict[int, object] = {}
    for body_id, neuron in (resolved or {}).items():
        try:
            out[int(body_id)] = neuron
        except (TypeError, ValueError):
            continue
    return out


# Deprecated alias: the loader serves both local releases (FAFB and BANC),
# so the historical FAFB-only name is retained only for older importers.
load_flywire_skeletons_batch = load_local_release_skeletons


def _import_visualizer():
    """Lazily import the VisualizeSkeleton class (heavy module; never loaded
    unless a run actually renders). Returns None when unavailable."""
    try:
        from visualize_skeleton import VisualizeSkeleton
        return VisualizeSkeleton
    except Exception:
        return None


def _load_neuron_type_map(dataset: str, project_root: Optional[str] = None
                          ) -> Tuple[Dict[Union[int, str], str],
                                     Dict[Union[int, str], str]]:
    """bodyId -> type / instance maps for a dataset.

    Uses the allneurons neuron table (fullest coverage), falling back to the
    neuron index parquet. These are the same sources ``SkeletonVectorCache``
    merges into the vector cache, so type lookups work even for datasets that
    have no vector cache yet (e.g. male-cns v1.0 with cached skeletons only).
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    local_release = is_local_connectome_dataset(dataset)
    type_map: Dict[Union[int, str], str] = {}
    instance_map: Dict[Union[int, str], str] = {}

    dataset_dir = root / "datasets" / folder
    table_candidates = [
        dataset_dir / f"{folder}_allneurons_neuron_df.parquet",
        dataset_dir / f"{folder}_allneurons_neuron_df.csv",
    ]
    table_path = next((path for path in table_candidates if path.exists()), None)
    if table_path is not None:
        try:
            if table_path.suffix.lower() == ".parquet":
                tdf = pd.read_parquet(
                    table_path, columns=["bodyId", "type", "instance"]
                )
            else:
                tdf = pd.read_csv(
                    table_path,
                    usecols=["bodyId", "type", "instance"],
                    dtype={"bodyId": "string"} if local_release else None,
                )
            if local_release:
                normalize_flywire_id_columns(tdf, ["bodyId"])
            else:
                tdf["bodyId"] = tdf["bodyId"].astype(np.int64)
            type_map = dict(zip(tdf["bodyId"], tdf["type"].fillna("").astype(str)))
            instance_map = dict(zip(tdf["bodyId"], tdf["instance"].fillna("").astype(str)))
            return type_map, instance_map
        except Exception:
            pass

    index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
    if index_path.exists() and _has_local_dataset_presence(dataset, root):
        try:
            idx_df = pd.read_parquet(index_path, columns=["bodyId", "type", "instance"])
            if local_release:
                normalize_flywire_id_columns(idx_df, ["bodyId"])
            else:
                idx_df["bodyId"] = idx_df["bodyId"].astype(np.int64)
            type_map = dict(zip(idx_df["bodyId"], idx_df["type"].fillna("").astype(str)))
            instance_map = dict(zip(idx_df["bodyId"], idx_df["instance"].fillna("").astype(str)))
        except Exception:
            pass
    return type_map, instance_map


def _find_skeleton_file(dataset: str, body_id: int,
                        project_root: Optional[str] = None) -> Optional[Path]:
    """Locate a cached skeleton/mesh file for a bodyId.

    Searches the dataset's skeletons directory recursively (datasets such as
    FlyWire keep bulk downloads in nested subfolders) and returns the first
    match, or None.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    cache_dir = root / "cache" / _dataset_folder(dataset) / "skeletons"
    if not cache_dir.exists():
        return None
    for name in (f"{body_id}.swc.zst", f"{body_id}.pkl.zst",
                 f"{body_id}.pkl", f"{body_id}.swc.gz"):
        direct = cache_dir / name
        if direct.exists():
            return direct
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.pkl.zst")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.pkl")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.swc.zst")
        if "raw_skeletons" not in path.parts
    )
    if nested:
        return nested[0]
    nested = sorted(
        path for path in cache_dir.rglob(f"{body_id}.swc.gz")
        if "raw_skeletons" not in path.parts
    )
    return nested[0] if nested else None


def _fafb_skeleton_zip_path(dataset: str,
                            project_root: Optional[str] = None) -> Optional[Path]:
    """Path of the healed FAFB skeleton bundle (``{bodyId}.swc`` entries).

    The FlyWire local mesh cache cannot serve NBLAST (dotprops want
    skeletons); the official healed skeleton ZIP is the real skeleton source.
    Returns None when the dataset has no such bundle.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    base = root / "datasets" / folder
    for candidate in (
        base / "sk_lod1_783_healed.zip",
        base / "downloads" / "sk_lod1_783_healed.zip",
        base / f"{folder}_skeletons.zip",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_fafb_zip_skeleton(zfile: "zipfile.ZipFile",
                            body_id: int) -> Optional["navis.TreeNeuron"]:
    """Load one skeleton (``{bodyId}.swc``) from the healed FAFB bundle."""
    try:
        import io

        content = zfile.read(f"{int(body_id)}.swc").decode("utf-8")
        nrn = navis.read_swc(io.StringIO(content))
        nrn.units = "nm"
        nrn.id = int(body_id)
        nrn.name = str(int(body_id))
        return nrn
    except Exception:
        return None


def _skeleton_folder_level(dataset: str,
                           project_root: Optional[str] = None) -> str:
    """Read a legacy folder-level marker for compatibility only.

    New raw SWC cache paths do not use ``skeletons/.level``. Older vector
    cache/build callers may still inspect the marker, so a missing marker
    conservatively resolves to ``raw``.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    marker = root / "cache" / _dataset_folder(dataset) / "skeletons" / ".level"
    try:
        v = marker.read_text(encoding='utf-8').strip()
        if v in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
            return v
    except Exception:
        pass
    return VECTOR_BASIS_RAW


def _write_skeleton_level_marker(dataset: str,
                                 project_root: Optional[str] = None):
    """Legacy marker writer retained for old integrations and tests.

    Production fetch/pull paths never call this function: raw skeletons are
    represented by compressed SWC and have no folder-level simplification
    marker.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    marker = root / "cache" / _dataset_folder(dataset) / "skeletons" / ".level"
    try:
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(SKELETON_CACHE_LEVEL + "\n", encoding="utf-8")
    except Exception:
        pass


def _downsample_for_cache(
        neuron, downsampling_factor: int = SKELETON_DOWNSAMPLE_FACTOR
        ) -> "navis.TreeNeuron":
    """Deterministic simplification for the on-disk NeuPrint skeleton cache.

    ``navis.downsample_neuron(factor=10)`` keeps ~10% of nodes while
    preserving root/leaves/branchpoints — the canonical "90% simplified"
    skeleton. ``downsampling_factor=20`` is used by the Settings bulk-pull
    option for the separate 95% simplified skeleton cache. Falls back to the
    original neuron when downsampling fails.
    """
    try:
        # navis' default soma detection flags every node with radius >= 1
        # (neuprint radii are in nm) as soma; a whole-neuron "soma" makes
        # every node a downsample fix point and freezes the skeleton at
        # full resolution. Treat a multi-node soma as no soma.
        soma = getattr(neuron, "soma", None)
        if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
            neuron = neuron.copy()
            neuron.soma = None
        return navis.downsample_neuron(
            neuron, downsampling_factor=max(1, int(downsampling_factor)))
    except Exception:
        return neuron


# Per-(dataset, root) memo for legacy vector caches without a ``rep`` column.
_REP_MEMO: Dict[Tuple[str, str], str] = {}


def _infer_dataset_rep(dataset: str, project_root: Optional[str] = None) -> str:
    """Representation of a dataset's cached skeletons ('skeleton'|'mesh').

    Legacy vector caches predate the ``rep`` column; the representation is
    inferred once per (dataset, project) from the first cached pickle.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    key = (dataset, str(root))
    if key in _REP_MEMO:
        return _REP_MEMO[key]
    rep = ""
    try:
        files = SkeletonVectorCache(
            dataset, project_root=str(root), verbose=False
        )._discover_skeleton_files()
        if files:
            rep = _neuron_rep(_load_cached_skeleton_file(files[0]))
    except Exception:
        rep = ""
    _REP_MEMO[key] = rep
    return rep


# =============================================================================
# Similarity
# =============================================================================

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against rows of a matrix (standardized)."""
    q = np.asarray(query, dtype=float).ravel()
    m = np.asarray(matrix, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return np.zeros(len(m))
    rows = np.linalg.norm(m, axis=1)
    scores = m @ q / (qn * np.maximum(rows, 1e-12))
    return np.where(rows == 0, 0.0, scores)


def pearson_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation of one vector against rows of a matrix."""
    q = np.asarray(query, dtype=float).ravel() - np.asarray(query, dtype=float).mean()
    m = np.asarray(matrix, dtype=float)
    m = m - m.mean(axis=1, keepdims=True)
    return cosine_similarity_matrix(q, m)


def similarity_matrix(query: np.ndarray, matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Similarity of one vector against matrix rows for a metric."""
    if metric == "pearson":
        return pearson_similarity_matrix(query, matrix)
    return cosine_similarity_matrix(query, matrix)


def pairwise_similarity_matrix(matrix: np.ndarray,
                               metric: str = "cosine") -> np.ndarray:
    """Pairwise similarity for every row of ``matrix``.

    The one-query helper is convenient for a single search, but calling it
    once per member repeats the row norms (and the matrix multiply setup) for
    every member of a type.  Normalize the whole matrix once and use one
    matrix multiplication instead.  Zero vectors remain zero-similarity,
    matching :func:`cosine_similarity_matrix`.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array")
    if values.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    if metric == "pearson":
        values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    scores = normalized @ normalized.T
    zero = norms == 0
    if np.any(zero):
        scores[zero, :] = 0.0
        scores[:, zero] = 0.0
    return scores


# =============================================================================
# V2 comparison: ZCA whitening + block-weighted similarity
# =============================================================================

def fit_zca_whitener(X_std: np.ndarray,
                     relative_floor: float = 1e-2,
                     max_amplification: float = 10.0
                     ) -> np.ndarray:
    """Truncated ZCA whitening matrix from standardized population rows.

    W = V diag(s) Vᵀ over the covariance Σ = V Λ Vᵀ of the standardized
    population, with

        s_j = min(1/sqrt(λ_j), max_amplification)   if λ_j >= floor
        s_j = 1                                     otherwise
        floor = relative_floor * λ_max

    High-variance directions are compressed (s < 1) and mid-range ones
    whitened (s ≈ 1), which decorrelates the feature space so a few
    high-variance shape dims can no longer dominate the cosine. Directions
    whose variance is below ``relative_floor`` of the largest carry only
    population noise: they are passed through UNCHANGED (s = 1) instead of
    being amplified — the plain eps-regularized version scaled a
    near-constant direction by 1/sqrt(eps), and that shared constant then
    dominated every vector, collapsing all cosines toward 1. The
    ``max_amplification`` cap bounds the same pathology for mid-low
    directions. Use :func:`apply_whitening` on standardized rows.
    """
    values = np.asarray(X_std, dtype=float)
    d = values.shape[1] if values.ndim == 2 else 0
    if d == 0:
        return np.eye(0)
    sigma = np.cov(values, rowvar=False)
    sigma = np.atleast_2d(sigma)
    try:
        evals, evecs = np.linalg.eigh(sigma)
    except np.linalg.LinAlgError:
        return np.eye(d)
    evals = np.clip(evals, 0.0, None)
    lam_max = float(evals.max())
    floor = relative_floor * lam_max
    keep = evals >= floor
    scale = np.where(keep, 1.0 / np.sqrt(np.maximum(evals, 1e-12)), 1.0)
    scale = np.minimum(scale, max_amplification)
    W = (evecs * scale) @ evecs.T
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)


def apply_whitening(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Whiten rows: X @ Wᵀ (W symmetric, so row order is preserved)."""
    W = np.asarray(W, dtype=float)
    values = np.asarray(X, dtype=float)
    if W.size == 0 or values.shape[-1] != W.shape[0]:
        return values
    return values @ W.T


def _block_cosine_one(q: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine of one query block against the matrix rows' matching block.

    Returns (scores, valid): ``valid`` is False for rows whose block is all
    zeros on either side — those rows exclude the block from their weighted
    mean instead of being forced to 0 (a neuron with no arbor in a region
    must not be scored as "maximally dissimilar there", it just carries no
    evidence for that block).
    """
    q = np.asarray(q, dtype=float).ravel()
    m = np.asarray(m, dtype=float)
    qn = float(np.linalg.norm(q))
    rows_n = np.linalg.norm(m, axis=1)
    if qn <= 1e-12:
        return np.zeros(len(m)), rows_n <= 1e-12
    scores = m @ q / (qn * np.maximum(rows_n, 1e-12))
    valid = rows_n > 1e-12
    return np.where(valid, scores, 0.0), valid


def _mirror_spatial_block(block: np.ndarray) -> np.ndarray:
    """Mirror a spatial block across the brain midline (x-axis flip).

    The ellipsoid's x-signed features (centroid x, principal-axis x) flip
    sign; the histogram's x-bin order reverses. The radial and midline-
    distance profiles are mirror-invariant (distances don't change sign)
    and pass through untouched. Applied to standardized/whitened vectors
    this is an approximation of scoring in the mirrored space — exact when
    the population statistics are LR-symmetric, which a full-brain
    population approximately is (the ROI screen makes the same assumption).
    """
    values = np.asarray(block, dtype=float)
    single = values.ndim == 1
    m = np.atleast_2d(values).copy()
    n_ell = len(SPATIAL_ELLIPSOID_FEATURES)
    m[:, 0] = -m[:, 0]    # sp_com_x
    m[:, 6] = -m[:, 6]    # sp_ax1_x
    hist_end = n_ell + SPATIAL_HIST_DIM
    hist = m[:, n_ell:hist_end].reshape(-1, SPATIAL_HIST_BINS[0],
                                        SPATIAL_HIST_BINS[1],
                                        SPATIAL_HIST_BINS[2])
    m[:, n_ell:hist_end] = np.flip(hist, axis=1).reshape(len(m), -1)
    return m[0] if single else m


def v2_similarity_matrix(query: np.ndarray, matrix: np.ndarray,
                         block_weights: Dict[str, float],
                         spatial_overlap: Optional[Dict[str, np.ndarray]] = None,
                         query_index: Optional[int] = None
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Block-weighted V2 similarity of one query against matrix rows.

    Fixed blocks (shape / spatial) are column slices of the full V2 schema.
    Each block contributes ``weight * cosine_block`` over the rows where the
    block is defined on both sides; the total renormalizes by the weights
    actually used per row. (The former runtime-composed ROI-expansion
    ``extra_blocks`` were removed — ROI evidence is a candidate-selection
    signal only.)

    The spatial block additionally blends a **mass-overlap term**
    (50/50): ``sum(min(q_hist, candidate_hist))`` over the RAW Hellinger
    histograms supplied via ``spatial_overlap`` =
    ``{"members": (n_query, 96), "centroid": (96,), "pool": (n_pool, 96)}``.
    The stored histograms are L1-style proportions, so the intersection is
    the fraction of the query's arbor distribution the candidate actually
    shares — cosine alone is proportion-blind (a neuron with the same
    *shape* of distribution but far less arbor in the query's region scores
    identically). ``query_index`` selects the query member's histogram;
    without it the pooled centroid histogram is used. Returns
    (total scores, per-block scores dict).
    """
    q = np.asarray(query, dtype=float).ravel()
    m = np.asarray(matrix, dtype=float)
    n = len(m)
    total_w = np.zeros(n)
    total = np.zeros(n)
    per_block: Dict[str, np.ndarray] = {}

    def _accumulate(name, weight, q_block, m_block):
        nonlocal total_w, total
        scores, valid = _block_cosine_one(q_block, m_block)
        if name == "spatial" and spatial_overlap is not None:
            pool = spatial_overlap["pool"]
            qh = (spatial_overlap["members"][query_index]
                  if query_index is not None
                  else spatial_overlap["centroid"])
            if qh is not None and len(pool) == n:
                inter = np.minimum(qh, pool).sum(axis=1)
                scores = 0.5 * scores + 0.5 * inter
        per_block[name] = scores
        total += weight * np.where(valid, scores, 0.0)
        total_w += np.where(valid, weight, 0.0)

    # Block slices are clamped to the input width: a V1-width snapshot
    # (e.g. the warmed V1 counterpart cache) is scored by its shape prefix.
    for name, (a, b) in (("shape", V2_SHAPE_SLICE),
                         ("spatial", V2_SPATIAL_SLICE)):
        b_eff = min(b, q.shape[0])
        weight = float(block_weights.get(name, 0.0))
        if weight <= 0 or a >= b_eff:
            continue
        _accumulate(name, weight, q[a:b_eff], m[:, a:b_eff])
    result = np.divide(total, np.maximum(total_w, 1e-12),
                       out=np.zeros(n), where=total_w > 1e-12)
    return result, per_block


def v2_pairwise_matrix(matrix: np.ndarray,
                       block_weights: Dict[str, float]) -> np.ndarray:
    """Pairwise block-weighted similarity over the rows of ``matrix``.

    Used for intra-type reference values. Only the fixed schema blocks
    participate (the ROI-expansion block is aligned to one run's id order,
    not to an arbitrary row subset).
    """
    values = np.asarray(matrix, dtype=float)
    n = len(values)
    total_w = np.zeros((n, n))
    total = np.zeros((n, n))
    # Slice clamp, as in v2_similarity_matrix (V1-width snapshots are
    # scored by their shape prefix).
    for name, (a, b) in (("shape", V2_SHAPE_SLICE),
                         ("spatial", V2_SPATIAL_SLICE)):
        b_eff = min(b, values.shape[1])
        weight = float(block_weights.get(name, 0.0))
        if weight <= 0 or a >= b_eff:
            continue
        block = values[:, a:b_eff]
        norms = np.linalg.norm(block, axis=1)
        valid = norms > 1e-12
        normalized = block / np.maximum(norms[:, None], 1e-12)
        scores = normalized @ normalized.T
        v = valid[:, None] & valid[None, :]
        total += weight * np.where(v, scores, 0.0)
        total_w += np.where(v, weight, 0.0)
    return np.divide(total, np.maximum(total_w, 1e-12),
                     out=np.zeros((n, n)), where=total_w > 1e-12)


def _sorted_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Candidate frame sorted by ``_score`` descending (ties by bodyId).

    Every discovery screen hands out its candidates in this one canonical
    order so the scoring pool is unambiguously the first ``candidate_cap``
    rows, whatever the source mode."""
    if candidates is None or candidates.empty or "_score" not in candidates.columns:
        return candidates
    return candidates.sort_values(
        ["_score", "target_bodyId"], ascending=[False, True]
    ).reset_index(drop=True)


# =============================================================================
# Skeleton vector cache
# =============================================================================

# Amortized merge thresholds for the append-only vector cache (main +
# pending): pending rows are folded into the main parquet once either
# threshold is crossed, so appends stay O(batch) and the full-file
# read-modify-write happens rarely.
PENDING_MERGE_ROWS = 5000
PENDING_MERGE_APPENDS = 8


class SkeletonVectorCache:
    """Per-dataset cache of vectorized skeletons (parquet + meta.json).

    ``raw_only=True`` creates the shared raw NeuPrint skeleton cache used by
    visualization and morphology comparison. Its files live below
    ``cache/{dataset}/skeletons/raw_skeletons/``. ``representation="mesh"``
    creates the separate FAFB prepared-mesh cache; it never writes SWC and
    reads the former visualization mesh folder as a migration source. BANC
    uses the raw public-release SWC cache instead.
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None,
                 n_workers: int = 8, verbose: bool = True,
                 raw_only: bool = False, raw_format: str = "swc.zst",
                 representation: Optional[str] = None):
        self.dataset = dataset
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.n_workers = max(1, int(n_workers))
        self.verbose = verbose
        self.raw_only = bool(raw_only)
        self.representation = str(representation or "").strip().lower()
        if self.representation not in {"", "mesh", "skeleton"}:
            raise ValueError("representation must be 'mesh', 'skeleton', or None")
        if self.raw_only and self.representation == "mesh":
            raise ValueError("raw_only cache cannot use mesh representation")
        self.mesh_only = self.representation == "mesh"
        self._local_release_ids = is_local_connectome_dataset(dataset)
        # Raw skeleton caches are portable compressed SWC by default.  The
        # legacy non-raw cache may still be explicitly opened as pickle for
        # migration/mesh compatibility, but new raw callers converge here.
        raw_format = str(raw_format or "swc.zst").strip().lower()
        if raw_format not in {"pkl", "swc.gz", "swc.zst"}:
            raise ValueError(
                "raw_format must be 'pkl', 'swc.gz', or 'swc.zst'")
        self.raw_format = raw_format
        folder = _dataset_folder(dataset)
        if self.mesh_only:
            base = self.project_root / "cache" / folder
            self.morph_dir = base / "find_similar" / "morphology"
            mesh_key = flywire_mesh_cache_key(
                FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            )
            self.skeleton_dir = base / "meshes" / mesh_key
            self.legacy_skeleton_dir = base / "skeletons" / mesh_key
        elif self.raw_only:
            base = self.project_root / "cache" / folder
            self.morph_dir = base / "find_similar" / "morphology"
            self.skeleton_dir = _provenance.raw_skeleton_store_dir(
                self.project_root, folder)
            self.legacy_skeleton_dir = (
                base / "find_similar" / _provenance.RAW_SKELETON_DIRNAME
            )
        else:
            self.morph_dir = self.project_root / "cache" / folder / "morphology"
            self.skeleton_dir = self.project_root / "cache" / folder / "skeletons"
            self.legacy_skeleton_dir = None
        vector_name = "mesh_vectors.parquet" if self.mesh_only \
            else "skeleton_vectors.parquet"
        meta_name = "mesh_meta.json" if self.mesh_only else "meta.json"
        self.parquet_path = self.morph_dir / vector_name
        self.meta_path = self.morph_dir / meta_name
        # Append-only staging file: new rows land here (O(batch)) and are
        # folded into ``parquet_path`` by the amortized merge checkpoint.
        self.pending_path = self.morph_dir / (
            "mesh_vectors_pending.parquet" if self.mesh_only
            else "skeleton_vectors_pending.parquet"
        )
        # Raw NeuPrint SWCs are the shared morphology/render source.  Keep a
        # small provenance manifest beside them so a cleared/rebuilt cache is
        # self-describing and cannot be mistaken for a render-time mesh cache.
        self.skeleton_manifest_path = self.skeleton_dir / (
            "raw_skeleton_manifest.json" if self.raw_only
            else "skeleton_manifest.json"
        )

    def _canonical_body_id(self, body_id):
        """Canonical cache key: exact strings for FAFB/BANC, ints for NeuPrint."""

        if self._local_release_ids:
            return normalize_flywire_body_id(body_id)
        return int(body_id)

    # ------------------------------------------------------------------ paths
    def cache_exists(self) -> bool:
        # A pending file counts as cache existence: rows appended but not yet
        # merged are real data (a crash must never make them invisible or
        # trigger a rebuild that would drop them).
        return self.parquet_path.exists() or self.pending_path.exists()

    def _load_skeleton_manifest(self) -> dict:
        """Load the optional raw-skeleton provenance manifest."""
        if not self.raw_only or not self.skeleton_manifest_path.exists():
            return {
                "cache_schema_version": RAW_SKELETON_CACHE_VERSION,
                "dataset": self.dataset,
                "representation": "skeleton",
                "source": "neuprint.fetch_skeleton",
                "coordinate_units": "nm",
                "vector_basis": VECTOR_BASIS_RAW,
                "files": {},
            }
        try:
            manifest = json.loads(self.skeleton_manifest_path.read_text(encoding='utf-8'))
            if (
                manifest.get("cache_schema_version")
                != RAW_SKELETON_CACHE_VERSION
                or manifest.get("dataset") != self.dataset
                or manifest.get("representation") != "skeleton"
                or not isinstance(manifest.get("files", {}), dict)
            ):
                raise ValueError("raw skeleton manifest has incompatible provenance")
            manifest.setdefault("files", {})
            return manifest
        except Exception:
            # A malformed/old manifest must not make otherwise valid SWCs
            # unreadable; the next successful write repairs the metadata.
            return {
                "cache_schema_version": RAW_SKELETON_CACHE_VERSION,
                "dataset": self.dataset,
                "representation": "skeleton",
                "source": "neuprint.fetch_skeleton",
                "coordinate_units": "nm",
                "vector_basis": VECTOR_BASIS_RAW,
                "files": {},
            }

    def _write_skeleton_manifest(self, manifest: dict) -> None:
        """Atomically persist raw-skeleton provenance metadata."""
        if not self.raw_only:
            return
        self.skeleton_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.skeleton_manifest_path.with_name(
            f".{self.skeleton_manifest_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.skeleton_manifest_path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    # -------------------------------------------------- pending (append-only)
    @staticmethod
    def _atomic_parquet(frame, path) -> None:
        """Atomic parquet write (temp + os.replace): a crash never leaves a
        truncated main/pending file behind."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    @staticmethod
    def _dedupe_frames(main_df: pd.DataFrame,
                       pending_df: pd.DataFrame) -> pd.DataFrame:
        """Concat main + pending and dedupe by bodyId, first occurrence wins
        (main rows win over pending rows — matches the historical "existing
        rows win" semantics)."""
        if pending_df is None or pending_df.empty:
            return (main_df.reset_index(drop=True)
                    if main_df is not None and not main_df.empty else main_df)
        if main_df is None or main_df.empty:
            combined = pending_df
        else:
            combined = pd.concat([main_df, pending_df], ignore_index=True)
        if combined.empty or "bodyId" not in combined.columns:
            return combined
        return combined.drop_duplicates(
            subset=["bodyId"], keep="first").reset_index(drop=True)

    def _clear_pending(self) -> None:
        """Remove the pending file and reset its append counter."""
        try:
            if self.pending_path.exists():
                self.pending_path.unlink()
        except OSError:
            pass
        meta = self._load_meta() or {}
        if meta.get("pending_appends"):
            meta["pending_appends"] = 0
            try:
                self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except OSError:
                pass

    def _vector_row_count(self) -> int:
        """Rows in main + pending (informational counts)."""
        n = 0
        for path in (self.parquet_path, self.pending_path):
            if not path.exists():
                continue
            try:
                n += len(pd.read_parquet(path))
            except Exception:
                pass
        return n

    def _merge_pending(self) -> int:
        """Fold pending rows into the main parquet (deduped, first-wins) and
        clear pending.  Called by ``append_vectors`` once a threshold is
        crossed (amortized O(n)) and as the pull's final checkpoint.
        Returns the number of rows written to main (0 = nothing to merge)."""
        if not self.pending_path.exists():
            return 0
        try:
            pending_df = pd.read_parquet(self.pending_path)
        except Exception:
            pending_df = pd.DataFrame()
        if pending_df is None or pending_df.empty:
            self._clear_pending()
            return 0
        main_df = pd.DataFrame()
        if self.parquet_path.exists():
            try:
                main_df = pd.read_parquet(self.parquet_path)
            except Exception:
                main_df = pd.DataFrame()
        merged = self._dedupe_frames(main_df, pending_df)
        if merged.empty:
            self._clear_pending()
            return 0
        # Align the schema to the main file's columns (legacy mains may lack
        # newer columns), as the historical append did with keep_cols.
        if main_df is not None and not main_df.empty:
            keep = [c for c in main_df.columns]
            merged = merged[[c for c in keep if c in merged.columns]]
        merged = merged.sort_values("bodyId").reset_index(drop=True)
        self._atomic_parquet(merged, self.parquet_path)
        meta = self._load_meta() or {}
        if not meta.get("mean") and not merged.empty:
            mat = self._raw_matrix(merged)
            mean = mat.mean(axis=0).tolist()
            std = mat.std(axis=0).tolist()
            std = [s if s > 0 else 1.0 for s in std]
            meta["mean"] = mean
            meta["std"] = std
        meta["dataset"] = self.dataset
        meta["n_rows"] = len(merged)
        meta["pending_appends"] = 0
        meta["built_at"] = datetime.now().isoformat(timespec="seconds")
        if "rep" not in meta and "rep" in merged.columns and len(merged):
            meta["rep"] = str(merged["rep"].iloc[0])
        if "vector_basis" not in meta:
            meta["vector_basis"] = VECTOR_BASIS_RAW
        if self.raw_only and "raw_format" not in meta:
            meta["raw_format"] = self.raw_format
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._clear_pending()
        return len(merged)

    # -------------------------------------------------- temp staging
    def temp_cache_dir(self) -> Path:
        """Transient raw-skeleton staging directory (crash-resume).

        Files here are written as fetch batches arrive and deleted once each
        batch is fully persisted; stale files from a crash are reprocessed
        (never re-fetched) on the next run.  Lives OUTSIDE ``skeleton_dir``
        (``skeletons/_temp_cache`` vs ``skeletons/raw_skeletons``) so the
        cache discovery never treats staging files as cached skeletons.
        """
        return (self.project_root / "cache" / _dataset_folder(self.dataset)
                / "skeletons" / "_temp_cache")

    def write_temp_skeleton(self, body_id, neuron) -> Path:
        """Write a raw (level-0) skeleton into the staging dir.

        Uses the fast zstd-3 codec (measured ~2.8x faster than level 19 in
        Step 0) so the staging stage keeps up with the fetch rate even for
        the densest neurons; the standard loader decompresses any level.
        Atomic per file - a crash never leaves a partial staging entry.
        """
        path = self.temp_cache_dir() / f"{body_id}.swc.zst"
        _write_compressed_skeleton(path, neuron, simplification=0,
                                   codec_level=3)
        return path

    def delete_temp_skeletons(self, body_ids) -> None:
        """Remove staging entries for fully-persisted neurons (best-effort)."""
        temp_dir = self.temp_cache_dir()
        for body_id in body_ids:
            try:
                (temp_dir / f"{body_id}.swc.zst").unlink(missing_ok=True)
            except OSError:
                continue

    def _discover_skeleton_files(self) -> List[str]:
        """All cached skeleton/mesh files, including nested bulk folders."""
        directories = [self.skeleton_dir]
        if (self.raw_only or self.mesh_only) \
                and self.legacy_skeleton_dir is not None:
            directories.append(self.legacy_skeleton_dir)
        directories = [directory for directory in directories
                       if directory.exists()]
        if not directories:
            return []
        files = []
        for directory in directories:
            files.extend(directory.rglob("*.pkl.zst"))
            files.extend(directory.rglob("*.pkl"))
            files.extend(directory.rglob("*.swc.zst"))
            files.extend(directory.rglob("*.swc.gz"))
        if not self.raw_only:
            files = [path for path in files
                     if "raw_skeletons" not in path.parts]
        # Transient crash-resume staging files are never cached skeletons.
        files = [path for path in files if "_temp_cache" not in path.parts]
        preferred = {}
        for path in files:
            try:
                body_id = _skeleton_body_id(path)
            except (TypeError, ValueError):
                continue
            current = preferred.get(body_id)
            # Prefer the new shared path over the legacy Find Similar path,
            # then prefer the canonical compressed representation for the
            # cache type (zstd SWC for raw skeletons, Zstandard for meshes).
            if str(path).endswith(".swc.zst"):
                format_rank = 0
            elif str(path).endswith(".swc.gz"):
                format_rank = 1
            elif str(path).endswith(".pkl.zst"):
                format_rank = 2
            else:
                format_rank = 3
            path_rank = (
                0 if self.skeleton_dir in path.parents else 1,
                format_rank,
            )
            current_path = Path(current) if current is not None else None
            if current_path is not None and str(current_path).endswith(".swc.zst"):
                current_format_rank = 0
            elif current_path is not None and str(current_path).endswith(".swc.gz"):
                current_format_rank = 1
            elif current_path is not None and str(current_path).endswith(".pkl.zst"):
                current_format_rank = 2
            else:
                current_format_rank = 3
            current_rank = (
                0 if self.skeleton_dir in current_path.parents else 1,
                current_format_rank,
            ) if current_path is not None else None
            if current is None or path_rank < current_rank:
                preferred[body_id] = path
        return sorted(str(path) for path in preferred.values())

    def _cached_dir_listing(self, directory: Path):
        """Flat file names + subdirectory flag of a cache directory.

        Cached per directory mtime: bulk pulls write thousands of files, and
        a per-id rglob would rescan the whole tree on every cache miss
        (O(N^2) across a full-dataset pull).  Returns ``(names, has_subdirs)``.
        """
        cache = getattr(self, "_dir_listing_cache", None)
        if cache is None:
            cache = {}
            self._dir_listing_cache = cache
        try:
            mtime = directory.stat().st_mtime_ns
        except OSError:
            return set(), False
        entry = cache.get(directory)
        if entry is not None and entry[0] == mtime:
            return entry[1], entry[2]
        names = set()
        has_subdirs = False
        try:
            with os.scandir(directory) as iterator:
                for item in iterator:
                    if item.is_dir(follow_symlinks=False):
                        has_subdirs = True
                    else:
                        names.add(item.name)
        except OSError:
            pass
        cache[directory] = (mtime, names, has_subdirs)
        return names, has_subdirs

    def find_skeleton_file(self, body_id: Union[int, str]) -> Optional[Path]:
        """Find a skeleton belonging to this cache namespace.

        The raw cache must never fall back to the shared simp90 pickle files.
        Its canonical location is the shared dataset skeleton namespace;
        historical Find Similar raw files are accepted only as a migration
        fallback.
        """
        body_id = self._canonical_body_id(body_id)
        if self.raw_only or self.mesh_only:
            directories = [self.skeleton_dir]
            if self.legacy_skeleton_dir is not None:
                directories.append(self.legacy_skeleton_dir)
            # Prefer the new shared path. Within a mesh cache prefer the
            # canonical compressed pickle; within a raw cache prefer zstd SWC.
            if self.mesh_only:
                names = (
                    f"{body_id}.pkl.zst",
                    f"{body_id}.pkl",
                )
            else:
                names = (
                    f"{body_id}.swc.zst",
                    f"{body_id}.swc.gz",
                    f"{body_id}.pkl",
                    f"{body_id}.pkl.zst",
                )
            for directory in directories:
                if not directory.exists():
                    continue
                _, has_subdirs = self._cached_dir_listing(directory)
                for name in names:
                    direct = directory / name
                    if direct.exists():
                        return direct
                    # Nested bulk folders exist only when the directory has
                    # subdirectories; skip the per-id rglob otherwise (it
                    # rescans the whole tree on every cache miss).
                    if has_subdirs:
                        nested = sorted(directory.rglob(name))
                        if nested:
                            return nested[0]
            return None
        return _find_skeleton_file(
            self.dataset, body_id, project_root=str(self.project_root)
        )

    def load_skeleton(self, body_id: int,
                      simplification: Optional[int] = None):
        """Load one valid cached neuron from this cache namespace.

        ``simplification`` optionally requests a target level (percent of
        nodes removed); when it is coarser than the stored level a
        TreeNeuron is re-simplified on load.  ``None`` (default) returns the
        neuron at its stored level.  MeshNeurons are never re-leveled (the
        simplification pipeline is NeuPrint/TreeNeuron-only).
        """
        path = self.find_skeleton_file(body_id)
        if path is None:
            return None
        try:
            # Load at the STORED level first: a lazy migration below must
            # persist the as-stored neuron (with its recorded level), never a
            # re-leveled copy that would mismatch its header.
            neuron = _load_cached_skeleton_file(path)
            if self.mesh_only:
                allowed = ("MeshNeuron",)
            elif self.raw_only:
                allowed = ("TreeNeuron",)
            else:
                allowed = ("TreeNeuron", "MeshNeuron")
            if type(neuron).__name__ not in allowed:
                return None
            if (self.mesh_only
                    and (self.skeleton_dir not in path.parents
                         or str(path).endswith(".pkl"))):
                self.persist_skeletons({self._canonical_body_id(body_id): neuron})
            if (self.raw_only and self.raw_format in ("swc.gz", "swc.zst")
                    and (self.skeleton_dir not in path.parents
                         or not str(path).endswith(f".{self.raw_format}"))):
                # Migrate legacy Find Similar files, and any temporary raw
                # file, to the canonical compressed-SWC namespace on first
                # successful load. The old file is intentionally kept
                # recoverable as a non-destructive fallback, and the stored
                # simplification level is preserved (never re-simplified).
                self.persist_skeletons(
                    {self._canonical_body_id(body_id): neuron},
                    simplification=None)
            if simplification is not None \
                    and type(neuron).__name__ == "TreeNeuron":
                stored_level = getattr(neuron, "_drocat_simplification", 0)
                neuron = _relevel_for_target(
                    neuron, stored_level, simplification)
            return neuron
        except Exception:
            return None

    def persist_skeletons(self, neurons: Dict[Union[int, str], object],
                          simplification: Optional[int] = 0
                          ) -> int:
        """Persist neurons in this cache's skeleton namespace.

        Shared raw caches store the fetched TreeNeuron through the shared
        compress pipeline: ``simplification`` (percent of nodes removed,
        0-90) is applied and recorded in each ``.swc.zst`` file header.
        The default is ``0`` — raw skeletons are cached as-is; simplification
        is a visualization-time concern and must not enter the stored cache.
        ``simplification=None`` writes the neuron as-is, recording the level
        already attached to it (used by lazy migrations).  Legacy pickle
        files remain readable and selectable with ``raw_format='pkl'``. The
        legacy visualization cache continues to use its explicit
        downsampling path and does not call this helper.
        """
        if not neurons:
            return 0
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        manifest = self._load_skeleton_manifest() if self.raw_only else None
        manifest_changed = False
        mesh_cache = None
        if self.mesh_only:
            mesh_cache = FlyWireMeshCache(
                self.dataset,
                project_root=self.project_root,
                simplification=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
                soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
                soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            )
        for body_id, neuron in neurons.items():
            try:
                body_id = self._canonical_body_id(body_id)
                if self.mesh_only and _neuron_rep(neuron) != "mesh":
                    continue
                if self.raw_only and _neuron_rep(neuron) != "skeleton":
                    continue
                if self.mesh_only:
                    written += mesh_cache.save({body_id: neuron})
                elif self.raw_only and self.raw_format == "swc.zst":
                    path = self.skeleton_dir / f"{body_id}.swc.zst"
                    _write_compressed_skeleton(
                        path, neuron,
                        simplification=simplification)
                    written += 1
                    stored_level = (
                        getattr(neuron, "_drocat_simplification", 0)
                        if simplification is None else simplification
                    ) or 0
                elif self.raw_only and self.raw_format == "swc.gz":
                    path = self.skeleton_dir / f"{body_id}.swc.gz"
                    _write_compressed_swc(
                        path, neuron)
                    written += 1
                    stored_level = 0
                else:
                    path = self.skeleton_dir / f"{body_id}.pkl"
                    with open(path, "wb") as handle:
                        pickle.dump(neuron, handle)
                    written += 1
                    stored_level = getattr(neuron, "_drocat_simplification", 0) or 0
                if self.raw_only and manifest is not None:
                    manifest["files"][str(body_id)] = {
                        "file": path.name,
                        "simplification": int(stored_level),
                        "representation": "skeleton",
                        "source": "neuprint.fetch_skeleton",
                        "coordinate_units": "nm",
                        "vector_basis": VECTOR_BASIS_RAW,
                        "updated_at": datetime.now().isoformat(
                            timespec="seconds"),
                    }
                    manifest_changed = True
            except Exception:
                continue
        if manifest_changed and manifest is not None:
            self._write_skeleton_manifest(manifest)
        return written

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------ meta
    def _write_meta(self, stats: Dict[str, List[float]], n_rows: int,
                    rep: str = "", vector_basis: str = VECTOR_BASIS_RAW):
        meta = {
            "version": self._cache_version(),
            "dataset": self.dataset,
            "feature_columns": self._feature_columns(),
            "persistence_dim": PERSISTENCE_DIM,
            "n_rows": n_rows,
            "rep": rep,
            # Simplification level of the vectors in this cache ("raw" |
            # "simp90"); the cache holds ONE level and never mixes.
            "vector_basis": vector_basis,
            "raw_format": self.raw_format if self.raw_only else None,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "mean": stats["mean"],
            "std": stats["std"],
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_meta(self) -> Optional[dict]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding='utf-8'))
        except Exception:
            return None

    # ------------------------------------------------------------ build
    def build(self, fetch_missing: int = 0) -> Dict[str, int]:
        """Vectorize all cached skeletons (incremental) and persist.

        Reuses existing rows; optionally fetches up to ``fetch_missing``
        additional neurons (persisted to the skeleton cache first). For
        FAFB v783 the full-dataset source is the healed skeleton bundle
        (``{bodyId}.swc`` entries — the local pickle cache holds meshes,
        which is the wrong representation for the vector cache).
        """
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        # FAFB v783: the healed zip is the full skeleton source (served
        # directly; a legacy .zst is opened read-only when no zip exists).
        use_bundle = False
        bundle_source = None
        bundle_ids: List[Union[int, str]] = []
        if not self.mesh_only and is_fafb_dataset(self.dataset):
            try:
                bundle_source = _fafb_bundle(
                    self.dataset, str(self.project_root))
            except Exception:
                bundle_source = None
            if bundle_source is not None:
                bundle_ids = sorted(
                    {
                        normalize_flywire_body_id(b)
                        if self._local_release_ids else int(b)
                        for b in bundle_source.ids()
                    }
                )
                use_bundle = True

        existing: Dict[int, dict] = {}
        if self.parquet_path.exists():
            try:
                df_old = pd.read_parquet(self.parquet_path)
                if use_bundle:
                    # The bundle produces skeleton vectors; a mesh-based
                    # (or legacy, rep-less) cache is incompatible and must
                    # be rebuilt from scratch.
                    reps = (set(df_old["rep"].fillna("").astype(str))
                            if "rep" in df_old.columns else {"legacy"})
                    if not reps <= {"skeleton", ""}:
                        self._log(
                            "[SkeletonVectorCache] Existing FAFB vector cache "
                            "is mesh-based; rebuilding it from the healed "
                            "skeleton bundle.")
                        df_old = None
                if df_old is not None:
                    existing = {
                        self._canonical_body_id(r["bodyId"]): r
                        for r in df_old.to_dict("records")
                    }
            except Exception:
                existing = {}
        # Fold any pending (appended-but-unmerged) rows into the rebuild so
        # they are never dropped; the pending file is cleared once the new
        # main parquet is written below.
        if self.pending_path.exists():
            try:
                pending_df = pd.read_parquet(self.pending_path)
                for r in pending_df.to_dict("records"):
                    existing.setdefault(
                        self._canonical_body_id(r["bodyId"]), r)
            except Exception:
                pass

        # The cache holds ONE vectorization level (its "basis"). On-disk
        # skeletons are vectorized only when their simplification level
        # matches that basis: post-cleanup NeuPrint caches hold simp90
        # files while the basis is raw, so those files are skipped (their
        # vectors come from the vector cache / raw fetches instead).
        basis = (VECTOR_BASIS_RAW if self.raw_only else
                 ((self._load_meta() or {}).get("vector_basis")
                  or VECTOR_BASIS_RAW))
        folder_level = (VECTOR_BASIS_RAW if self.raw_only else
                         _skeleton_folder_level(self.dataset,
                                                 str(self.project_root)))

        # Candidate skeletons not yet vectorized.
        if use_bundle:
            files: List[str] = []
            pending = [
                self._canonical_body_id(b)
                for b in bundle_ids
                if self._canonical_body_id(b) not in existing
            ]
        else:
            files = self._discover_skeleton_files()
            files = [f for f in files if folder_level == basis]
            pending = [
                f for f in files
                if self._canonical_body_id(_skeleton_body_id(f)) not in existing
            ]

        # Optional on-demand fetch to extend coverage (cap applies).
        # fetch_skeleton_on_demand already vectorizes at fetch time (raw
        # basis) and persists the vector; the fetched row set is refreshed
        # below so those rows are not dropped by the merge.
        fetched_new = 0
        # In-memory rows for loader-resolved trees that are never written to
        # the raw store (FAFB zip/repair caches). Appended to ``rows`` below.
        extra_rows: List = []
        if fetch_missing and fetch_missing > 0:
            index_path = self.project_root / "neuron_indexes" / _dataset_folder(self.dataset) / "neuron_index.parquet"
            index: List[int] = []
            if index_path.exists() and _has_local_dataset_presence(
                self.dataset, Path(self.project_root)
            ):
                try:
                    idx_df = pd.read_parquet(index_path, columns=["bodyId"])
                    index = (
                        [normalize_flywire_body_id(b)
                         for b in idx_df["bodyId"].tolist()]
                        if self._local_release_ids else
                        [int(b) for b in idx_df["bodyId"].tolist()]
                    )
                except Exception:
                    index = []
            if index:
                have = {self._canonical_body_id(b) for b in list(existing)}
                if use_bundle:
                    have |= set(bundle_ids)
                else:
                    have |= {
                        self._canonical_body_id(_skeleton_body_id(f))
                        for f in files
                    }
                missing = [b for b in index if b not in have]
                if is_fafb_dataset(self.dataset):
                    # FAFB skeletons come from the local loader (repair
                    # caches / healed zip), never the mesh-returning batch
                    # fetcher. Vectorize the in-memory trees here; nothing is
                    # written into the raw store (zip-only pipeline).
                    fetched_trees = _resolve_fafb_skeleton_trees(
                        self.dataset, missing[:fetch_missing],
                        project_root=str(self.project_root),
                        check_extrusions=False)
                    extra_rows = [
                        row for row in (
                            self._in_memory_vector_row(bid, neuron)
                            for bid, neuron in fetched_trees.items())
                        if row is not None]
                    fetched_new = len(fetched_trees)
                else:
                    fetched_map = fetch_skeletons_on_demand_batch(
                        self.dataset,
                        missing[:fetch_missing],
                        project_root=str(self.project_root),
                        persist=True,
                        level=VECTOR_BASIS_RAW,
                        raw_cache=self if (self.raw_only or self.mesh_only) else None,
                        vector_cache=self if (self.raw_only or self.mesh_only) else None,
                    )
                    extra_rows = []
                    fetched_new = len(fetched_map)
            if fetched_new:
                # Re-discover after the fetches: they wrote new skeleton
                # files (and, in the real pipeline, already appended the
                # raw vectors). Refresh the row set and the pending files
                # so neither the fetched vectors nor the on-disk files are
                # dropped by the merge below.
                try:
                    df_old = pd.read_parquet(self.parquet_path)
                    existing = {
                        self._canonical_body_id(r["bodyId"]): r
                        for r in df_old.to_dict("records")
                    }
                except Exception:
                    pass
                if use_bundle:
                    pending = [
                        self._canonical_body_id(b)
                        for b in bundle_ids
                        if self._canonical_body_id(b) not in existing
                    ]
                else:
                    files = self._discover_skeleton_files()
                    files = [f for f in files if folder_level == basis]
                    pending = [
                        f for f in files
                        if self._canonical_body_id(_skeleton_body_id(f))
                        not in existing
                    ]

        rows = []
        if pending:
            started = time.time()
            source_label = ("healed zip skeletons" if use_bundle
                            else "skeletons")
            self._log(
                f"[SkeletonVectorCache] Vectorizing {len(pending)} {source_label} "
                f"({self.dataset})..."
            )
            if use_bundle:
                source_path = (str(bundle_source.bundle_path)
                               if bundle_source.bundle_path else None)
                zip_path = (str(bundle_source.zip_path)
                            if bundle_source.zip_path else None)
                if self.n_workers > 1:
                    try:
                        rows = self._vectorize_parallel_swc(
                            source_path, zip_path, pending)
                    except Exception:
                        rows = self._vectorize_swc_serial(
                            source_path, zip_path, pending)
                else:
                    rows = self._vectorize_swc_serial(
                        source_path, zip_path, pending)
            elif self.n_workers > 1:
                try:
                    rows = self._vectorize_parallel(pending)
                except Exception:
                    rows = [None] * len(pending)
                    for i, p in enumerate(pending):
                        rows[i] = _vectorize_one_file(p)
            else:
                rows = [_vectorize_one_file(p) for p in pending]
            elapsed = time.time() - started
            self._log(
                f"[SkeletonVectorCache] Vectorized {len(pending)} neurons in "
                f"{elapsed:.1f}s ({elapsed / max(len(pending), 1) * 1000:.1f} ms/neuron)"
            )
        if extra_rows:
            # Loader-resolved FAFB trees: appended after the file rows so the
            # representation guard and merge below treat them identically.
            rows = list(rows) + list(extra_rows)

        # A cache must hold ONE representation (skeleton vs mesh) and one
        # simplification level: vector features differ between the two, so
        # rows of any other representation are skipped (never mixed into
        # comparisons). The majority representation of the pending set wins.
        ok_rows = [r for r in rows if r is not None]
        rep = ""
        if ok_rows:
            from collections import Counter
            rep = ("skeleton" if self.raw_only else
                   Counter(r[3] for r in ok_rows).most_common(1)[0][0])
            foreign = sum(1 for r in ok_rows if r[3] != rep)
            if foreign:
                self._log(
                    f"[SkeletonVectorCache] Skipping {foreign} pickles of a "
                    f"different representation ({'mesh' if rep == 'skeleton' else 'skeleton'})"
                )
            rows = [r if r is None or r[3] == rep else None for r in rows]

        # Merge with existing rows (type/instance are refreshed from the
        # neuron index below, so drop any stale copies from previous builds).
        records = []
        for bid, rec in existing.items():
            row = {k: v for k, v in rec.items() if k not in ("type", "instance")}
            row["bodyId"] = self._canonical_body_id(bid)
            records.append(row)
        for row in rows:
            if row is None:
                continue
            bid, morph_vals, pv_vals, row_rep = row
            record = {
                "bodyId": self._canonical_body_id(bid),
                "rep": row_rep,
            }
            for name, val in zip(MORPHOMETRIC_FEATURES, morph_vals):
                record[name] = float(val)
            for i, val in enumerate(pv_vals):
                record[f"pv_{i}"] = float(val)
            records.append(record)

        if not records:
            self._log("[SkeletonVectorCache] No skeletons available to vectorize.")
            self._write_meta({"mean": [], "std": []}, 0, rep=rep,
                             vector_basis=basis)
            return {"rows": 0, "new": 0, "fetched": 0}

        df = pd.DataFrame(records)
        if self._local_release_ids:
            normalize_flywire_id_columns(df, ["bodyId"])
        df = df.sort_values("bodyId").reset_index(drop=True)

        # Attach type/instance (used by type-level aggregation and result
        # reporting). The allneurons neuron table has the fullest bodyId ->
        # type/instance coverage; the neuron index is the fallback.
        type_map, instance_map = _load_neuron_type_map(self.dataset, str(self.project_root))
        if type_map:
            df["type"] = df["bodyId"].map(type_map).fillna("")
            df["instance"] = df["bodyId"].map(instance_map or {}).fillna("")
        else:
            df["type"] = ""
            df["instance"] = ""

        self._atomic_parquet(df, self.parquet_path)

        # Z-score stats over the population.
        mat = self._raw_matrix(df)
        mean = mat.mean(axis=0).tolist()
        std = mat.std(axis=0).tolist()
        std = [s if s > 0 else 1.0 for s in std]
        self._write_meta({"mean": mean, "std": std}, len(df), rep=rep,
                         vector_basis=basis)
        # A full rebuild supersedes the append-only staging file.
        self._clear_pending()

        self._log(
            f"[SkeletonVectorCache] Cache ready: {len(df)} rows "
            f"({len(rows)} new, {fetched_new} fetched) -> {self.parquet_path}"
        )
        if bundle_source is not None:
            try:
                bundle_source.close()
            except Exception:
                pass
        return {"rows": len(df), "new": len(rows), "fetched": fetched_new}

    def _vectorize_parallel(self, files: List[str]) -> List[Tuple[int, List[float], List[float]]]:
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as ex:
            return list(ex.map(_vectorize_one_file, files, chunksize=16))

    def _vectorize_parallel_swc(self, source_path: str,
                                zip_path: Optional[str], bids: List[int]
                                ) -> List[Tuple[int, List[float], List[float]]]:
        """Vectorize healed-zip skeletons in a worker pool; each worker
        opens the zip (read-only; legacy .zst only when no zip exists)."""
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx,
                                 initializer=_init_fafb_zip_worker,
                                 initargs=(source_path, zip_path)) as ex:
            return list(ex.map(_vectorize_one_swc, bids, chunksize=16))

    def _vectorize_swc_serial(self, source_path: str,
                              zip_path: Optional[str], bids: List[int]
                              ) -> List[Tuple[int, List[float], List[float]]]:
        """Serial healed-zip vectorization (single-worker or fallback)."""
        global _FAFB_WORKER_BUNDLE
        _init_fafb_zip_worker(source_path, zip_path)
        try:
            return [_vectorize_one_swc(b) for b in bids]
        finally:
            if _FAFB_WORKER_BUNDLE is not None:
                _FAFB_WORKER_BUNDLE.close()
                _FAFB_WORKER_BUNDLE = None

    # ------------------------------------------------------------ load
    @staticmethod
    def _raw_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = MORPHOMETRIC_FEATURES + [f"pv_{i}" for i in range(PERSISTENCE_DIM)]
        return df[cols].to_numpy(dtype=float)

    def _feature_columns(self) -> List[str]:
        """Parquet feature columns of this cache's vector schema (V1 124)."""
        return MORPHOMETRIC_FEATURES + [f"pv_{i}" for i in range(PERSISTENCE_DIM)]

    def _cache_version(self) -> int:
        return VECTOR_CACHE_VERSION

    def _meta_extra(self) -> dict:
        """Extra meta keys stamped on append-created caches (schema hook)."""
        return {}

    def _default_basis(self) -> str:
        """Vector basis recorded when a cache is created (schema hook)."""
        return VECTOR_BASIS_RAW

    def _file_level_ok(self, neuron) -> bool:
        """Whether a locally cached skeleton may feed this cache's basis.

        V1 (raw basis): only level-0 files — simplified files are skipped
        (their vectors come from the vector cache / raw fetches instead).
        """
        return getattr(neuron, "_drocat_simplification", 0) == 0

    def _vector_dim(self) -> int:
        """Width of this cache's vectors (schema hook for the V2 cache)."""
        return VECTOR_DIM

    def _vectorize_neuron(self, neuron):
        """Vector-schema hook: (feature dict, flat vector) for one neuron."""
        return vectorize_neuron(neuron)

    def _in_memory_vector_row(self, body_id, neuron):
        """Build a cache row for an in-memory neuron (loader-resolved trees).

        Mirrors the V1 file worker's row shape
        ``(body_id, morph list, shape list, rep)`` so FAFB skeletons that are
        served from the zip / repair caches (never written to the raw store)
        can still be vectorized by ``build(fetch_missing>0)``. Returns None
        for unserializable neurons.
        """
        try:
            morph, vector = self._vectorize_neuron(neuron)
            rep = _neuron_rep(neuron)
        except Exception:
            return None
        return (int(body_id),
                [morph[f] for f in MORPHOMETRIC_FEATURES],
                vector[len(MORPHOMETRIC_FEATURES):].tolist(),
                rep)

    def load(self) -> Optional[dict]:
        """Load the cache: meta + raw df + standardized matrix + index arrays.

        ``rep`` carries each row's representation ('skeleton' | 'mesh');
        ``dataset_rep`` the cache's single representation (legacy caches
        without a ``rep`` column infer it from the first cached file).
        Main + pending rows are merged with a first-wins dedupe by bodyId, so
        crash-duplicate appends are never visible to consumers.
        """
        if not self.parquet_path.exists() and not self.pending_path.exists():
            return None
        df = pd.DataFrame()
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
            except Exception:
                df = pd.DataFrame()
        if self.pending_path.exists():
            try:
                pending_df = pd.read_parquet(self.pending_path)
                df = self._dedupe_frames(df, pending_df)
            except Exception:
                pass
        if df.empty:
            return None
        if self._local_release_ids:
            normalize_flywire_id_columns(df, ["bodyId"])
        meta = self._load_meta() or {}
        raw = self._raw_matrix(df)
        mean = np.asarray(meta.get("mean") or raw.mean(axis=0), dtype=float)
        std = np.asarray(meta.get("std") or raw.std(axis=0), dtype=float)
        std = np.where(std <= 0, 1.0, std)
        X = (raw - mean) / std
        reps = df.get("rep", pd.Series([""] * len(df))).fillna("").astype(str).tolist()
        if reps and reps[0]:
            from collections import Counter
            dataset_rep = Counter(reps).most_common(1)[0][0]
        else:
            dataset_rep = _infer_dataset_rep(self.dataset, self.project_root)
        body_ids = (
            df["bodyId"].astype("string").to_numpy(dtype=object)
            if self._local_release_ids else
            df["bodyId"].astype(np.int64).to_numpy()
        )
        return {
            "meta": meta,
            "df": df,
            "raw": raw,
            "X": X,
            # The statistics ``X`` was z-scored with, published so a row
            # computed later in the same query can join that same space
            # (``vectors_for``) instead of being scored raw beside it.
            "mean": mean,
            "std": std,
            "bodyIds": body_ids,
            "types": df.get("type", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
            "instances": df.get("instance", pd.Series([""] * len(df))).fillna("").astype(str).tolist(),
            "rep": reps,
            "dataset_rep": dataset_rep,
        }

    def ensure(self, fetch_missing: int = 0) -> dict:
        """Auto-build the cache on first query (lazy)."""
        if not self.cache_exists():
            self._log(f"[SkeletonVectorCache] Building vector cache for {self.dataset}...")
            return self.build(fetch_missing=fetch_missing)
        return {"rows": self._vector_row_count(), "new": 0, "fetched": 0}

    def coverage(self) -> Dict[str, int]:
        """Skeleton and vector counts for the dataset.

        For FAFB v783 the local skeleton count is the healed zip's entry
        count (the pickle cache holds meshes, not skeletons).
        """
        n_skeletons = len(self._discover_skeleton_files())
        if is_fafb_dataset(self.dataset):
            try:
                bundle = _fafb_bundle(self.dataset, str(self.project_root))
            except Exception:
                bundle = None
            if bundle is not None:
                try:
                    n_skeletons = bundle.count()
                finally:
                    bundle.close()
            else:
                zip_path = _fafb_skeleton_zip_path(
                    self.dataset, str(self.project_root))
                if zip_path is not None:
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, "r") as z:
                            n_skeletons = sum(
                                1 for n in z.namelist() if n.endswith(".swc"))
                    except Exception:
                        pass
        n_vectors = self._vector_row_count()
        return {"skeletons": n_skeletons, "vectors": n_vectors}

    # ------------------------------------------------------------ append
    def append_vectors(self, records: List[Tuple[int, np.ndarray, str]],
                       vector_basis: Optional[str] = None) -> int:
        """Persist freshly-computed vectors (raw feature rows) into the cache.

        ``vector_basis=None`` resolves to the cache's own default basis
        (raw for both caches since the raw-basis flip — V1 and the V2
        schema cache alike; simplification is visualization-only).

        Called when a vector was computed from a cached skeleton file or from
        an online-fetched skeleton that was NOT persisted: the VECTOR is
        stored so later queries reuse it without re-fetching or
        re-vectorizing, even though the original skeleton stays uncached.

        Append-only: the rows are written to the small ``*_pending.parquet``
        staging file (O(batch); atomic) and folded into the main parquet by
        the amortized merge checkpoint once a threshold is crossed.  Dedupe
        by bodyId is deferred to read/merge time (first occurrence wins), so
        crash-duplicate appends are never visible to consumers.  The cache's
        standardization statistics (meta mean/std) are left untouched, so the
        standardized space stays consistent across appends.  Rows of a
        representation different from the cache's are rejected (a cache
        holds ONE level), and rows whose ``vector_basis`` differs from the
        cache's basis are rejected too (a cache holds ONE simplification
        level).  Returns the number of rows appended to pending (may
        over-count by rows that dedupe against existing data at merge).
        """
        if not records:
            return 0
        vector_basis = vector_basis or self._default_basis()
        # Cross-process safety: UI runs execute in separate subprocesses, so
        # the pending read-append-write is guarded with an advisory file lock
        # (POSIX; best-effort elsewhere).
        lock_fd = None
        lock_path = self.parquet_path.with_suffix(".parquet.lock")
        try:
            import fcntl
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            lock_fd = None
        try:
            type_map, instance_map = _load_neuron_type_map(
                self.dataset, str(self.project_root)
            )
            rows_new = []
            seen = set()
            for bid, vec, rep in records:
                if self.raw_only and str(rep) != "skeleton":
                    continue
                bid = self._canonical_body_id(bid)
                if bid in seen:
                    continue
                seen.add(bid)
                row = {"bodyId": bid, "rep": rep}
                for i, name in enumerate(self._feature_columns()):
                    row[name] = float(vec[i])
                row["type"] = type_map.get(bid, "") if type_map else ""
                row["instance"] = (instance_map or {}).get(bid, "") if instance_map else ""
                rows_new.append(row)
            if not rows_new:
                return 0
            df_new = pd.DataFrame(rows_new)

            # A cache holds ONE basis and ONE representation; the meta
            # records them once they exist (build or first append).
            meta = self._load_meta() or {}
            cache_basis = meta.get("vector_basis") or VECTOR_BASIS_RAW
            if meta.get("vector_basis") and cache_basis != vector_basis:
                # Different simplification level: never mix (a cache
                # holds ONE basis).
                return 0
            cache_rep = meta.get("rep") or ""
            if cache_rep:
                df_new = df_new[df_new["rep"] == cache_rep]
            else:
                # Creating the cache: keep it homogeneous (majority
                # representation of this batch) and record the basis.
                from collections import Counter
                canonical = (Counter(df_new["rep"].astype(str))
                             .most_common(1)[0][0] if len(df_new) else "")
                if canonical:
                    df_new = df_new[df_new["rep"].astype(str) == canonical]
            if df_new.empty:
                return 0

            # Append to the pending staging file (O(pending + batch), atomic).
            pending_df = pd.DataFrame()
            if self.pending_path.exists():
                try:
                    pending_df = pd.read_parquet(self.pending_path)
                except Exception:
                    pending_df = pd.DataFrame()
            combined = self._dedupe_frames(pending_df, df_new)
            self._atomic_parquet(combined, self.pending_path)

            meta["dataset"] = self.dataset
            meta["n_rows"] = (self._vector_row_count()
                               if self.parquet_path.exists() else len(combined))
            meta["pending_appends"] = int(meta.get("pending_appends") or 0) + 1
            if not (meta.get("mean") and meta.get("std")):
                # Fix the standardization basis the first time a population
                # exists. Without it `load()` falls back to the CURRENT rows'
                # stats, so every later append re-bases the vectors it already
                # served and the same bodyId scores differently run to run.
                after = self.load()
                if after is not None:
                    meta["mean"] = [float(v) for v in after["mean"]]
                    meta["std"] = [float(v) for v in after["std"]]
            meta.setdefault("version", self._cache_version())
            meta.update(self._meta_extra())
            meta["built_at"] = datetime.now().isoformat(timespec="seconds")
            if "rep" not in meta and len(df_new) and "rep" in df_new.columns:
                meta["rep"] = str(df_new["rep"].iloc[0])
            # Record the basis when creating the cache; existing caches keep
            # their own basis (enforced by the check above).
            if "vector_basis" not in meta:
                meta["vector_basis"] = vector_basis
            if self.raw_only and "raw_format" not in meta:
                meta["raw_format"] = self.raw_format
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Amortized merge checkpoint: fold pending into main once either
            # threshold is crossed (read-modify-write of the whole main file
            # happens rarely, never per append).
            if (len(combined) >= PENDING_MERGE_ROWS
                    or int(meta.get("pending_appends") or 0)
                    >= PENDING_MERGE_APPENDS):
                self._merge_pending()
            return len(df_new)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_fd.close()

    # ------------------------------------------------------------ vectors_for
    def vectors_for(self, body_ids: List[int], compute_missing: bool = True,
                    space: str = "standardized"
                    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return vectors for bodyIds — every row in ONE space.

        ``space='standardized'`` (default) hands back rows z-scored by the
        cache's own mean/std, which is the space :func:`apply_whitening` and
        therefore every native-track scorer expects. ``space='raw'`` hands
        back the persisted feature rows as stored, for a caller that applies
        its own statistics.

        The one-space rule is the contract. Rows this call computes used to be
        returned RAW while cached rows came back STANDARDIZED, so a cold
        neuron was whitened and scored in a different space from the pool it
        was being graded against: measured 2026-09-25 on the FAFB->BANC pooling
        run, that moved 107 of 1266 exported rows and 16 pool decisions between
        the run and its warm repeat, and every one of those rows was
        reproducible to ~1e-16 by re-feeding exactly the ids the call had to
        compute. A call that finds no statistics to standardize with — no
        cache yet, or one whose meta predates the key — persists its rows,
        stamps the basis they are now stored under, and re-reads them, so its
        answer is the answer the NEXT call gives too. One space means one space
        ACROSS runs, not only inside a call.

        Rows missing from the cache are computed on the fly when a skeleton
        file of the cache's representation AND simplification level exists
        (``compute_missing=True``); files of a DIFFERENT representation
        (e.g. skeletons beside bulk meshes) or of a different level than
        the cache's ``vector_basis`` are skipped so comparisons never mix
        levels. Otherwise they are NaN rows. Never fetches from the server
        and never forces a full cache build. Returns (vectors, mask, reps)
        where ``reps`` carries each row's representation ('skeleton' |
        'mesh' | '').
        """
        body_ids = [self._canonical_body_id(b) for b in body_ids]
        if space not in ("standardized", "raw"):
            # a typo'd space name would otherwise fall through to the
            # standardized branch and silently return the wrong space
            raise ValueError(
                f"space must be 'standardized' or 'raw', not {space!r}")
        data = self.load()
        known: Dict[int, int] = {}
        X = np.zeros((0, self._vector_dim()))
        mu = sd = None
        dataset_rep = ""
        basis = self._default_basis()
        if data is not None:
            known = {
                self._canonical_body_id(b): i
                for i, b in enumerate(data["bodyIds"])
            }
            X = data["raw"] if space == "raw" else data["X"]
            dataset_rep = data.get("dataset_rep", "")
            basis = ((data.get("meta") or {}).get("vector_basis")
                     or self._default_basis())

        result = np.full((len(body_ids), self._vector_dim()), np.nan)
        reps = [""] * len(body_ids)
        computed: List[Tuple[int, np.ndarray, str]] = []
        for j, bid in enumerate(body_ids):
            if bid in known:
                result[j] = X[known[bid]]
                reps[j] = dataset_rep
                continue
            if compute_missing:
                # Level guard: on-disk skeletons are vectorized only when
                # their simplification level matches the cache's basis
                # (post-cleanup NeuPrint caches hold simp90 files while
                # the basis is raw -> never vectorized here).
                pkl = None
                if (self.raw_only or self.mesh_only or
                        _skeleton_folder_level(
                            self.dataset, str(self.project_root)) == basis):
                    pkl = self.find_skeleton_file(bid)
                if pkl is not None:
                    try:
                        neuron = _load_cached_skeleton_file(pkl)
                        # Per-file level guard: only files matching the
                        # cache's basis are vectorized (V1: level 0 only).
                        if not self._file_level_ok(neuron):
                            continue
                        row_rep = _neuron_rep(neuron)
                        if self.raw_only and row_rep != "skeleton":
                            continue
                        if self.mesh_only and row_rep != "mesh":
                            continue
                        if dataset_rep and row_rep != dataset_rep:
                            continue  # different representation: never mix
                        _, vec = self._vectorize_neuron(neuron)
                        # Persist the RAW row (that is what the parquet holds);
                        # hand back the row in the space this call promised.
                        computed.append((bid, vec, row_rep))
                        if space != "raw" and data is not None:
                            # `load()` always publishes the statistics it
                            # standardized `X` with, so the row computed here
                            # joins that same space.
                            vec = (vec - data["mean"]) / data["std"]
                        result[j] = vec
                        reps[j] = row_rep
                    except Exception:
                        result[j] = np.nan
        if computed:
            self.append_vectors(computed, vector_basis=basis)
            # The store's META is the pin, not `data["mean"]`: `load()`
            # ALWAYS publishes statistics (falling back to the current rows
            # when meta is silent), so the old guard `data.get("mean") is
            # None` was reachable only with no cache at all.  A LEGACY cache
            # — rows present, meta without mean/std — served its computed
            # rows standardized by N-row statistics while append_vectors
            # pinned N+M statistics into meta: one cold/warm divergence per
            # legacy store (the V1 Find-Similar path, 413 rows).  Re-read
            # whenever this call CREATED the basis or the pin: what a call
            # hands back has to be what the NEXT call hands back for the
            # same bodyIds, or a dataset's first run grades its own cold
            # neurons in one space and every later run in another (measured
            # 2026-09-25 on an empty cache: raw 33.31, 12, 1, 10 then
            # standardized 1, -1, 1, 1).
            if data is None or not ((data.get("meta") or {}).get("mean")):
                after = self.load()
                if after is not None:
                    index = {self._canonical_body_id(b): i
                             for i, b in enumerate(after["bodyIds"])}
                    proj = after["raw"] if space == "raw" else after["X"]
                    for j, bid in enumerate(body_ids):
                        i = index.get(bid)
                        if i is not None:
                            result[j] = proj[i]
        mask = ~np.isnan(result[:, 0])
        return result, mask, reps


def find_similar_raw_cache(dataset: str,
                           project_root: Optional[str] = None,
                           n_workers: int = 8,
                           verbose: bool = True,
                           raw_format: str = "swc.zst") -> SkeletonVectorCache:
    """Return the shared raw-skeleton/vector cache for a dataset.

    The historical function name is retained for compatibility with callers,
    but raw skeletons now live outside the Find Similar directory at
    ``cache/{dataset}/skeletons/raw_skeletons/``. New persisted raw skeletons
    use zstd-compressed SWC by default; pass ``raw_format='pkl'`` or
    ``raw_format='swc.gz'`` only for legacy compatibility.
    """
    return SkeletonVectorCache(
        dataset,
        project_root=project_root,
        n_workers=n_workers,
        verbose=verbose,
        raw_only=True,
        raw_format=raw_format,
    )


def find_similar_flywire_mesh_cache(
        dataset: str,
        project_root: Optional[str] = None,
        n_workers: int = 8,
        verbose: bool = True,
        ) -> SkeletonVectorCache:
    """Return the separate prepared-mesh/vector cache for FAFB only."""
    if not is_flywire_dataset(dataset):
        raise ValueError("FAFB mesh cache requires a FAFB dataset")
    return SkeletonVectorCache(
        dataset,
        project_root=project_root,
        n_workers=n_workers,
        verbose=verbose,
        representation="mesh",
    )


def find_similar_dataset_cache(
        dataset: str,
        project_root: Optional[str] = None,
        n_workers: int = 8,
        verbose: bool = True,
        ) -> SkeletonVectorCache:
    """Return the dataset-native vector/cache manager.

    NeuPrint and BANC own raw SWC managers; FAFB owns the prepared mesh
    manager. This boundary prevents a CAVE MeshNeuron from being serialized
    through a raw SWC writer.
    """
    if is_flywire_dataset(dataset):
        return find_similar_flywire_mesh_cache(
            dataset, project_root=project_root, n_workers=n_workers,
            verbose=verbose)
    return find_similar_raw_cache(
        dataset, project_root=project_root, n_workers=n_workers,
        verbose=verbose)


class SkeletonVectorCacheV2(SkeletonVectorCache):
    """V2 (spatial/topological) vector cache: same skeletons, richer schema.

    Shares the raw skeleton store with the V1 cache (one download serves
    both vector schemas — V1 rows also stay available for the vector-cache
    prefilter and ``enrich_homolog_results``) but keeps SEPARATE
    parquet/meta/pending files. Vectors are :data:`VECTOR_V2_DIM`-dim (V1's
    124 features + the spatial block; the topology block was removed in
    schema v4). Meta additionally records
    the population spatial bounds used by the histogram block; a ZCA
    whitening matrix fitted on the standardized population lives in a
    ``*_whiten_v2.npz`` sidecar (identity below
    :data:`MIN_ROWS_FOR_WHITENING` rows; reused while ``fit_version``
    matches; refit otherwise; deleted on a full ``build()``).
    """

    def __init__(self, dataset: str, project_root: Optional[str] = None,
                 n_workers: int = 8, verbose: bool = True,
                 raw_format: str = "swc.zst", representation: Optional[str] = None,
                 bundle_sample_target: int = FAFB_BUNDLE_SAMPLE_TARGET):
        super().__init__(dataset, project_root=project_root,
                         n_workers=n_workers, verbose=verbose,
                         raw_only=str(representation or "").lower() != "mesh",
                         raw_format=raw_format,
                         representation=representation)
        # FAFB only: when the vector population falls below this target,
        # build() seeds a representative random sample of the whole-brain
        # healed skeleton bundle so cache-direct searches have a meaningful
        # candidate pool.
        self.bundle_sample_target = max(0, int(bundle_sample_target))
        # Re-point every schema-bearing file at the *_v2 namespace; the
        # skeleton dirs (raw SWCs / meshes) stay shared with V1.
        suffix = "_v2"
        stem = "mesh" if self.mesh_only else "skeleton"
        self.parquet_path = self.morph_dir / f"{stem}__vectors{suffix}.parquet"
        self.meta_path = self.morph_dir / (
            f"mesh_meta{suffix}.json" if self.mesh_only else f"meta{suffix}.json")
        self.pending_path = self.morph_dir / f"{stem}__vectors{suffix}_pending.parquet"
        self.whiten_path = self.morph_dir / (
            f"mesh_whiten{suffix}.npz" if self.mesh_only else f"whiten{suffix}.npz")
        self._spatial_bounds: Optional[np.ndarray] = None
        self._bounds_resolved = False

    # ------------------------------------------------------- schema hooks
    @staticmethod
    def _raw_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = (MORPHOMETRIC_FEATURES + [f"pv_{i}" for i in range(PERSISTENCE_DIM)]
                + SHAPE_EXTRA_FEATURES
                + SPATIAL_ELLIPSOID_FEATURES
                + [f"sh_{i}" for i in range(SPATIAL_HIST_DIM)]
                + SPATIAL_EXTRA_FEATURES)
        return df[cols].to_numpy(dtype=float)

    def _vector_dim(self) -> int:
        return VECTOR_V2_DIM

    def _default_basis(self) -> str:
        # Raw skeleton basis (design contract): vectorization runs on raw
        # skeletons; simplification is a visualization-time concern and
        # never enters the vector cache. Files stored at a simplified
        # level are skipped (their bodies are re-fetched raw on demand).
        return VECTOR_BASIS_RAW

    def _file_level_ok(self, neuron) -> bool:
        # Only raw (level 0 / unstamped) trees join the raw basis; a
        # simp90-stamped file would silently shift this body onto a
        # different feature basis than the rest of the cache.
        return (getattr(neuron, "_drocat_simplification", 0) or 0) == 0

    def _feature_columns(self) -> List[str]:
        return (MORPHOMETRIC_FEATURES + [f"pv_{i}" for i in range(PERSISTENCE_DIM)]
                + SHAPE_EXTRA_FEATURES
                + SPATIAL_ELLIPSOID_FEATURES
                + [f"sh_{i}" for i in range(SPATIAL_HIST_DIM)]
                + SPATIAL_EXTRA_FEATURES)

    def _cache_version(self) -> int:
        return VECTOR_CACHE_V2_VERSION

    def _vectorize_neuron(self, neuron):
        # Raw basis: vectorize the stored skeleton as-is (the caller's
        # _file_level_ok gate keeps simplified files out).
        return vectorize_neuron_v2(neuron, self.spatial_bounds(),
                                   lateral_normalize=True)

    def _in_memory_vector_row(self, body_id, neuron):
        """V2 cache row for an in-memory (loader-resolved) tree.

        Raw basis: the loader-resolved tree is vectorized as-is. Returns
        None on failure.
        """
        try:
            _, vector = vectorize_neuron_v2(
                neuron, self.spatial_bounds(), lateral_normalize=True)
            rep = _neuron_rep(neuron)
        except Exception:
            return None
        return (int(body_id), vector.tolist(), rep)

    def _meta_extra(self) -> dict:
        """Extra meta keys stamped on every append (schema hook)."""
        return {"lateral_normalize": True}

    def _is_stale(self) -> bool:
        """Whether persisted V2 rows predate the current vector semantics.

        True when rows exist whose meta lacks the current schema version or
        the lateral-normalization marker: such rows must be re-vectorized,
        never mixed with newer ones."""
        meta = self._load_meta()
        if meta is None:
            return False   # nothing persisted: nothing to invalidate
        return (int(meta.get("version") or 0) != self._cache_version()
                or meta.get("lateral_normalize") is not True)

    # ------------------------------------------------------ spatial bounds
    def spatial_bounds(self) -> Optional[np.ndarray]:
        """Population bounding box for the histogram block: [lo(3), hi(3)].

        Estimated once from up to 200 locally cached skeletons and persisted
        in meta, so every neuron of the dataset is binned identically (2%
        padding; outliers clip at the edges). None when no skeletons are
        locally available — the histogram block is then zeros everywhere.
        """
        if self._bounds_resolved:
            return self._spatial_bounds
        self._bounds_resolved = True
        meta = self._load_meta() or {}
        stored = meta.get("spatial_bounds")
        if (isinstance(stored, list) and len(stored) == 2
                and all(len(row) == 3 for row in stored)):
            self._spatial_bounds = np.asarray(stored, dtype=float)
            return self._spatial_bounds
        files = self._discover_skeleton_files()
        if len(files) > 200:
            rng = np.random.default_rng(0)
            files = [files[i] for i in
                     rng.choice(len(files), 200, replace=False)]
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        seen = 0
        for path in files:
            try:
                neuron = _load_cached_skeleton_file(path)
                pts = _neuron_points(neuron)
            except Exception:
                continue
            if not len(pts):
                continue
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
            seen += 1
        if not seen:
            return None
        pad = 0.02 * np.maximum(hi - lo, 0.0)
        bounds = np.vstack([lo - pad, hi + pad])
        self._spatial_bounds = bounds
        try:
            meta = self._load_meta() or {}
            meta["spatial_bounds"] = bounds.tolist()
            # Stamp the schema identity: a bounds-only meta (no version /
            # lateral-normalization marker) reads as schema-stale on the
            # next load and would force a needless mid-search rebuild.
            meta.setdefault("version", self._cache_version())
            meta["lateral_normalize"] = True
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            pass
        return self._spatial_bounds

    def _write_meta(self, stats: Dict[str, List[float]], n_rows: int,
                    rep: str = "", vector_basis: str = VECTOR_BASIS_RAW):
        meta = {
            "version": self._cache_version(),
            "dataset": self.dataset,
            "feature_columns": self._feature_columns(),
            "persistence_dim": PERSISTENCE_DIM,
            "spatial_bounds": (self._spatial_bounds.tolist()
                               if self._spatial_bounds is not None
                               else (self._load_meta() or {}).get("spatial_bounds")),
            "lateral_normalize": True,
            "block_weights": dict(DEFAULT_V2_BLOCK_WEIGHTS),
            "n_rows": n_rows,
            "rep": rep,
            "vector_basis": VECTOR_BASIS_RAW,
            "raw_format": self.raw_format if self.raw_only else None,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "mean": stats["mean"],
            "std": stats["std"],
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- whitening
    def _whitener(self, X_std: np.ndarray) -> np.ndarray:
        """Load or fit the population ZCA whitener (identity below the
        minimum population). Fitted on standardized rows; callers apply it
        AFTER standardization. The sidecar records the fit version — a
        whitener saved by an older algorithm is refit, never reused."""
        d = X_std.shape[1] if X_std.ndim == 2 else VECTOR_V2_DIM
        if len(X_std) < MIN_ROWS_FOR_WHITENING:
            return np.eye(d)
        if self.whiten_path.exists():
            try:
                with np.load(self.whiten_path, allow_pickle=False) as data:
                    if int(data["fit_version"]) == WHITEN_FIT_VERSION:
                        W = data["W"]
                        if W.shape == (d, d):
                            return W
            except Exception:
                pass
        W = fit_zca_whitener(X_std)
        try:
            self.whiten_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.whiten_path.with_suffix(".npy.tmp")
            # File handle: np.savez would otherwise append ".npz" to the
            # temp name and break the atomic replace below.
            with open(tmp, "wb") as handle:
                np.savez(handle, W=W, fit_version=WHITEN_FIT_VERSION)
            tmp.replace(self.whiten_path)
        except OSError:
            pass
        return W

    def load(self) -> Optional[dict]:
        if self._is_stale():
            # Rows written under older vector semantics would silently mix
            # with new ones; treat the cache as absent so callers rebuild
            # (ensure() routes through build() for the same reason).
            return None
        data = super().load()
        if data is None:
            return None
        data["whiten"] = self._whitener(data["X"])
        return data

    def ensure(self, fetch_missing: int = 0) -> dict:
        if self._is_stale():
            self._log("[SkeletonVectorCacheV2] Cache predates the current "
                      "vector semantics; rebuilding.")
            # A stale basis must never be reused incrementally: drop the
            # old rows so build() re-vectorizes every body at the current
            # basis (raw since the raw-basis flip) instead of merging them
            # into the new cache.
            for path in (self.parquet_path, self.meta_path,
                         self.pending_path):
                try:
                    Path(path).unlink()
                except OSError:
                    pass
            return self.build(fetch_missing=fetch_missing)
        return super().ensure(fetch_missing=fetch_missing)

    def append_vectors(self, records: List[Tuple[int, np.ndarray, str]],
                       vector_basis: Optional[str] = None) -> int:
        """Append with a staleness guard.

        Rows of the CURRENT schema must never land in a cache whose meta
        predates it: load() refuses stale caches outright, so such rows
        would be invisible to every reader (the vectors silently recompute
        on every run) while still shaping the staging file. Rebuild first —
        offline, no fetches — so the append lands under current meta.
        """
        if records and self._is_stale():
            self._log("[SkeletonVectorCacheV2] Cache predates the current "
                      f"vector semantics; rebuilding before appending "
                      f"{len(records)} row(s).")
            self.build(fetch_missing=0)
        return super().append_vectors(records, vector_basis=vector_basis)

    # --------------------------------------------------------------- build
    def _vectorize_parallel_swc_v2(self, source_path: str,
                                   zip_path: Optional[str], bids: List[int],
                                   bounds) -> List[Tuple[int, List[float], str]]:
        """Bundle skeletons -> V2 vectors in a worker pool (FAFB sampling)."""
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx,
                                 initializer=_init_v2_swc_worker,
                                 initargs=(source_path, zip_path, bounds)) as ex:
            return list(ex.map(_vectorize_one_swc_v2, bids, chunksize=8))

    def _vectorize_swc_serial_v2(self, source_path: str,
                                 zip_path: Optional[str], bids: List[int],
                                 bounds) -> List[Tuple[int, List[float], str]]:
        global _FAFB_WORKER_BUNDLE, _V2_WORKER_BOUNDS
        _init_v2_swc_worker(source_path, zip_path, bounds)
        try:
            return [_vectorize_one_swc_v2(b) for b in bids]
        finally:
            if _FAFB_WORKER_BUNDLE is not None:
                _FAFB_WORKER_BUNDLE.close()
                _FAFB_WORKER_BUNDLE = None

    def _bundle_population_rows(self, need: int,
                                skip_ids: set) -> List[Tuple[int, List[float], str]]:
        """Vectorize a seeded random sample of the FAFB healed skeleton
        bundle into V2 rows. Bounds are (re-)estimated from a small
        coordinate pre-sample of the bundle so the whole brain shares one
        histogram binning; the estimate is persisted by ``_write_meta``."""
        bundle = _fafb_bundle(self.dataset, str(self.project_root))
        if bundle is None:
            return []
        try:
            import io
            all_ids = sorted(int(b) for b in bundle.ids())
            all_ids = [b for b in all_ids if b not in skip_ids]
            if not all_ids:
                return []
            rng = np.random.default_rng(0)
            pre = [int(b) for b in rng.choice(all_ids,
                                              size=min(60, len(all_ids)),
                                              replace=False)]
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for bid in pre:
                content = bundle.get(bid)
                if not content:
                    continue
                try:
                    nrn = navis.read_swc(io.StringIO(content))
                    pts = nrn.nodes[["x", "y", "z"]].to_numpy(dtype=float)
                except Exception:
                    continue
                lo = np.minimum(lo, pts.min(axis=0))
                hi = np.maximum(hi, pts.max(axis=0))
            if not np.isfinite(lo).all():
                return []
            pad = 0.02 * (hi - lo)
            bounds = np.vstack([lo - pad, hi + pad])
            self._spatial_bounds = bounds
            self._bounds_resolved = True
            rng.shuffle(all_ids)
            picks = sorted(all_ids[:need])
            self._log(f"[SkeletonVectorCacheV2] Seeding {len(picks)} bundle "
                      f"skeletons (whole-brain sample of {len(all_ids) + len(skip_ids)})...")
            source_path = (str(bundle.bundle_path)
                           if bundle.bundle_path else None)
            zip_path = str(bundle.zip_path) if bundle.zip_path else None
            if self.n_workers > 1:
                try:
                    rows = self._vectorize_parallel_swc_v2(
                        source_path, zip_path, picks, bounds.tolist())
                except Exception:
                    rows = self._vectorize_swc_serial_v2(
                        source_path, zip_path, picks, bounds)
            else:
                rows = self._vectorize_swc_serial_v2(
                    source_path, zip_path, picks, bounds)
            return [r for r in rows if r is not None]
        finally:
            try:
                bundle.close()
            except Exception:
                pass
    def build(self, fetch_missing: int = 0) -> Dict[str, int]:
        """Vectorize all cached skeletons into the V2 schema (incremental).

        Mirrors the V1 build's discovery/merge structure but vectorizes with
        :func:`vectorize_neuron_v2` and records the population spatial
        bounds in meta. Optional ``fetch_missing`` extends the skeleton
        store first (raw files land in the SHARED skeleton dir; the V1
        vector cache is warmed too, never the V2 files).
        """
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        fafb_fetch_rows: List = []
        if fetch_missing and fetch_missing > 0:
            index_path = (self.project_root / "neuron_indexes"
                          / _dataset_folder(self.dataset)
                          / "neuron_index.parquet")
            index: List[int] = []
            if index_path.exists() and _has_local_dataset_presence(
                    self.dataset, Path(self.project_root)):
                try:
                    idx_df = pd.read_parquet(index_path, columns=["bodyId"])
                    index = ([normalize_flywire_body_id(b)
                              for b in idx_df["bodyId"].tolist()]
                             if self._local_release_ids else
                             [int(b) for b in idx_df["bodyId"].tolist()])
                except Exception:
                    index = []
            if index:
                existing_ids = {
                    self._canonical_body_id(_skeleton_body_id(f))
                    for f in self._discover_skeleton_files()
                    if isinstance(_skeleton_body_id(f), (int, np.integer))
                    or self._local_release_ids
                }
                try:
                    df_old = pd.read_parquet(self.parquet_path)
                    existing_ids |= {
                        self._canonical_body_id(b) for b in df_old["bodyId"]}
                except Exception:
                    pass
                missing = [b for b in index
                           if self._canonical_body_id(b) not in existing_ids]
                if missing and is_fafb_dataset(self.dataset):
                    # FAFB resolves through the local loader (repair caches /
                    # healed zip); the batch fetcher returns meshes. Rows are
                    # built in memory below — the zip is never mirrored.
                    fafb_fetch_rows = [
                        row for row in (
                            self._in_memory_vector_row(bid, neuron)
                            for bid, neuron in _resolve_fafb_skeleton_trees(
                                self.dataset, missing[:fetch_missing],
                                project_root=str(self.project_root),
                                check_extrusions=False).items())
                        if row is not None]
                elif missing:
                    fetch_skeletons_on_demand_batch(
                        self.dataset, missing[:fetch_missing],
                        project_root=str(self.project_root),
                        persist=True, level=VECTOR_BASIS_RAW,
                        raw_cache=self,
                        vector_cache=None,
                    )

        files = self._discover_skeleton_files()
        existing: Dict[int, dict] = {}
        if self._is_stale():
            # Rows written under older vector semantics are never reused.
            self._log("[SkeletonVectorCacheV2] Existing rows are stale; "
                      "rebuilding from local skeletons.")
        elif self.parquet_path.exists():
            try:
                df_old = pd.read_parquet(self.parquet_path)
                existing = {self._canonical_body_id(r["bodyId"]): r
                            for r in df_old.to_dict("records")}
            except Exception:
                existing = {}
        if self.pending_path.exists() and not self._is_stale():
            try:
                for r in pd.read_parquet(self.pending_path).to_dict("records"):
                    existing.setdefault(self._canonical_body_id(r["bodyId"]), r)
            except Exception:
                pass
        pending = [f for f in files
                   if self._canonical_body_id(_skeleton_body_id(f))
                   not in existing]

        # FAFB whole-brain seeding needs to fall through even with an empty
        # raw store: the bundle sample derives its own population bounds.
        fafb_seeding = (is_fafb_dataset(self.dataset) and not self.mesh_only
                        and self.bundle_sample_target > 0)
        if (not pending and not fafb_fetch_rows and not fafb_seeding
                and not self.pending_path.exists()):
            self._log("[SkeletonVectorCacheV2] No skeletons available to "
                      "vectorize.")
            return {"rows": len(existing), "new": 0, "fetched": 0}
        # When pending staging rows exist, fall through even with no files
        # to vectorize: the write path below folds them into the main
        # parquet and recomputes the population statistics (the amortized
        # merge checkpoint); an early return here would leave the parquet
        # and its stats frozen at the first build's population.
        # Population bounds must exist BEFORE vectorization so every row of
        # the build shares one histogram binning.
        bounds = self.spatial_bounds()
        bounds_list = bounds.tolist() if bounds is not None else None
        self._log(f"[SkeletonVectorCacheV2] Vectorizing {len(pending)} "
                  f"skeletons ({self.dataset}, bounds="
                  f"{'estimated' if bounds is not None else 'unavailable'})...")
        started = time.time()
        rows: List[Optional[Tuple[int, List[float], str]]] = []
        if self.n_workers > 1:
            ctx = mp.get_context("fork") if hasattr(mp, "get_context") and \
                "fork" in mp.get_all_start_methods() else mp.get_context()
            try:
                with ProcessPoolExecutor(max_workers=self.n_workers,
                                         mp_context=ctx) as ex:
                    futures = [ex.submit(_vectorize_one_file_v2, f, bounds_list)
                               for f in pending]
                    rows = [f.result() for f in futures]
            except Exception:
                rows = [_vectorize_one_file_v2(f, bounds_list)
                        for f in pending]
        else:
            rows = [_vectorize_one_file_v2(f, bounds_list) for f in pending]
        if fafb_fetch_rows:
            # Loader-resolved FAFB trees (zip / repair caches) are vectorized
            # into the same population; they were never written to disk.
            rows.extend(fafb_fetch_rows)
        elapsed = time.time() - started
        self._log(f"[SkeletonVectorCacheV2] Vectorized {len(pending)} neurons "
                  f"in {elapsed:.1f}s "
                  f"({elapsed / max(len(pending), 1) * 1000:.1f} ms/neuron)")

        ok_rows = [r for r in rows if r is not None]
        # FAFB: the local raw_skeletons only hold neurons a user has already
        # touched. The healed skeleton bundle carries the WHOLE-BRAIN
        # population — seed a representative random sample (once, cached)
        # so cache-direct searches have a meaningful candidate pool.
        if (is_fafb_dataset(self.dataset) and not self.mesh_only
                and self.bundle_sample_target > 0):
            covered = {int(self._canonical_body_id(k)) for k in existing}
            covered |= {int(r[0]) for r in ok_rows}
            need = self.bundle_sample_target - len(existing) - len(ok_rows)
            if need > 0:
                t_bundle = time.time()
                bundle_rows = self._bundle_population_rows(need, covered)
                ok_rows.extend(bundle_rows)
                self._log(f"[SkeletonVectorCacheV2] Bundle sample: "
                          f"+{len(bundle_rows)} neurons in "
                          f"{time.time() - t_bundle:.1f}s "
                          f"(target {self.bundle_sample_target})")
        rep = ""
        if ok_rows:
            from collections import Counter
            rep = ("skeleton" if self.raw_only else
                   Counter(r[2] for r in ok_rows).most_common(1)[0][0])
            foreign = sum(1 for r in ok_rows if r[2] != rep)
            if foreign:
                self._log(f"[SkeletonVectorCacheV2] Skipping {foreign} files "
                          f"of a different representation than {rep}.")
            # Merge from ok_rows (file rows + in-memory rows such as the
            # FAFB bundle sample): filtering the original `rows` list here
            # silently dropped every in-memory row.
            rows = ok_rows

        records = []
        for bid, rec in existing.items():
            row = {k: v for k, v in rec.items() if k not in ("type", "instance")}
            row["bodyId"] = self._canonical_body_id(bid)
            records.append(row)
        columns = self._feature_columns()
        for row in rows:
            if row is None:
                continue
            bid, vec, row_rep = row
            record = {"bodyId": self._canonical_body_id(bid), "rep": row_rep}
            for name, val in zip(columns, vec):
                record[name] = float(val)
            records.append(record)
        if not records:
            self._log("[SkeletonVectorCacheV2] No usable skeletons.")
            return {"rows": 0, "new": 0, "fetched": 0}

        df = pd.DataFrame(records)
        if self._local_release_ids:
            normalize_flywire_id_columns(df, ["bodyId"])
        df = df.sort_values("bodyId").reset_index(drop=True)
        type_map, instance_map = _load_neuron_type_map(
            self.dataset, str(self.project_root))
        df["type"] = df["bodyId"].map(type_map).fillna("") if type_map else ""
        df["instance"] = (df["bodyId"].map(instance_map or {}).fillna("")
                          if instance_map else "")

        self._atomic_parquet(df, self.parquet_path)
        mat = self._raw_matrix(df)
        mean = mat.mean(axis=0).tolist()
        std = mat.std(axis=0).tolist()
        std = [s if s > 0 else 1.0 for s in std]
        self._write_meta({"mean": mean, "std": std}, len(df), rep=rep,
                         vector_basis=VECTOR_BASIS_RAW)
        # A fresh population invalidates the fitted whitener.
        try:
            self.whiten_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._clear_pending()
        self._log(f"[SkeletonVectorCacheV2] Cache ready: {len(df)} rows "
                  f"({len(ok_rows)} new) -> {self.parquet_path}")
        return {"rows": len(df), "new": len(ok_rows), "fetched": 0}


def find_similar_dataset_cache_v2(
        dataset: str,
        project_root: Optional[str] = None,
        n_workers: int = 8,
        verbose: bool = True,
        ) -> SkeletonVectorCacheV2:
    """Return the dataset-native V2 (spatial/topological) cache manager.

    FAFB uses a SKELETON-representation cache fed by the local healed
    skeleton bundle. Standalone BANC also uses a SKELETON cache, populated
    from its public-release SWCs. NeuPrint remains the default raw-SWC cache.
    """
    if is_fafb_dataset(dataset):
        return SkeletonVectorCacheV2(dataset, project_root=project_root,
                                     n_workers=n_workers, verbose=verbose,
                                     representation="skeleton")
    if is_banc_dataset(dataset):
        return SkeletonVectorCacheV2(dataset, project_root=project_root,
                                     n_workers=n_workers, verbose=verbose,
                                     representation="skeleton")
    return SkeletonVectorCacheV2(dataset, project_root=project_root,
                                 n_workers=n_workers, verbose=verbose)


def cache_fetched_skeleton_vectors(
        dataset: str, neurons, project_root: Optional[str] = None,
        vector_cache: Optional[SkeletonVectorCache] = None,
        progress_callback=None, progress_offset: int = 0,
        progress_total: Optional[int] = None, verbose: bool = False
        ) -> Dict[str, object]:
    """Vectorize and persist fresh NeuPrint/BANC skeletons or FAFB meshes.

    This is the cache transaction shared by the morphology batch fetcher and
    the visualizer's own NeuPrint batch loop.  The function returns only after
    the parquet append has completed, so a caller cannot enter layered
    rendering (or similarity scoring) while the vector cache is still being
    written.  ``progress_callback`` uses the same ``(done, total, message)``
    contract as the online skeleton fetch.

    Cache failures are reported explicitly and do not discard the fetched
    neurons: visualization remains usable even if a read-only cache directory
    is encountered. The returned ``cached`` count makes that condition
    observable to callers and tests.
    """
    if isinstance(neurons, dict):
        items = list(neurons.items())
    else:
        items = []
        for index, neuron in enumerate(list(neurons or [])):
            body_id = getattr(neuron, "id", None)
            if body_id is None:
                body_id = index
            items.append((body_id, neuron))

    total = int(progress_total if progress_total is not None
               else progress_offset + len(items))
    if progress_callback:
        progress_callback(
            progress_offset, total,
            f"Vectorizing fetched skeletons (0/{len(items)})")

    started = time.perf_counter()
    rows = []
    failures = 0
    for index, (body_id, neuron) in enumerate(items, start=1):
        try:
            rep = _neuron_rep(neuron)
            if rep not in {"skeleton", "mesh"}:
                raise TypeError(
                    f"expected TreeNeuron or MeshNeuron, got {type(neuron).__name__}")
            _, vector = vectorize_neuron(neuron)
            rows.append((
                _canonical_dataset_body_id(dataset, body_id),
                vector,
                rep,
            ))
        except Exception as exc:
            failures += 1
            if verbose:
                print(f"[morphology] vectorization skipped for {body_id}: {exc}")
        if progress_callback:
            progress_callback(
                progress_offset + index, total,
                f"Vectorizing fetched skeletons ({index}/{len(items)})")

    cached = 0
    cache_error = None
    if rows:
        if vector_cache is not None:
            target = vector_cache
        elif all(row[2] == "mesh" for row in rows):
            target = find_similar_flywire_mesh_cache(
                dataset, project_root=project_root, verbose=False)
        else:
            target = SkeletonVectorCache(
                dataset, project_root=project_root, verbose=False)
        try:
            cached = int(target.append_vectors(
                rows, vector_basis=VECTOR_BASIS_RAW))
        except Exception as exc:
            cache_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[morphology] vector cache write failed: {cache_error}")

    elapsed = time.perf_counter() - started
    if progress_callback:
        progress_callback(
            progress_offset + len(items), total,
            f"Vector cache complete ({cached}/{len(rows)} rows; "
            f"{elapsed:.2f}s)")
    return {
        "seen": len(items),
        "vectorized": len(rows),
        "cached": cached,
        "failures": failures,
        "elapsed": elapsed,
        "cache_error": cache_error,
    }


# =============================================================================
# Population standardization statistics
# =============================================================================

def _datasets_share_population(dataset: str, other: str,
                               root: Path) -> bool:
    """True when at least 30% of the smaller neuron index is in the other.

    Guards the version-sibling statistics fallback: two versions of one
    reconstruction (e.g. male-cns v0.9/v1.0) share their neurons, so the
    older release's population stats are a valid baseline for the newer.
    """

    def _ids(ds: str):
        p = root / "neuron_indexes" / _dataset_folder(ds) / "neuron_index.parquet"
        if not p.exists():
            return None
        try:
            import polars as pl
            return set(pl.read_parquet(p, columns=["bodyId"])["bodyId"].to_list())
        except Exception:
            return None
    a, b = _ids(dataset), _ids(other)
    if not a or not b:
        return False
    smaller, bigger = (a, b) if len(a) <= len(b) else (b, a)
    return len(smaller & bigger) / len(smaller) >= 0.3


def _sibling_skeleton_dirs(dataset: str, root: Path) -> List[Path]:
    """Skeleton-cache directories of version siblings sharing the population.

    Two versions of one reconstruction (e.g. male-cns v0.9/v1.0) contain the
    same neurons, so the sibling's cached skeletons extend the sample used
    for population standardization statistics. The dataset-folder convention
    is inverted (``male-cns_v1_0`` -> ``male-cns:v1.0``) and every candidate
    is verified against ``_datasets_share_population``.
    """
    name = dataset.split(":")[0] if ":" in dataset else dataset
    if not name:
        return []
    dirs: List[Path] = []
    cache_root = root / "cache"
    if not cache_root.is_dir():
        return []
    for folder in sorted(cache_root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(name + "_"):
            continue
        parts = folder.name.split("_")
        if len(parts) < 2:
            continue
        sibling = f"{parts[0]}:{parts[1]}" + (
            f".{'.'.join(parts[2:])}" if len(parts) > 2 else ""
        )
        if sibling == dataset or not _datasets_share_population(dataset, sibling, root):
            continue
        skel = folder / "skeletons"
        if skel.is_dir():
            dirs.append(skel)
    return dirs


def population_stats(dataset: str, project_root: Optional[str] = None,
                     max_sample: int = POPULATION_STATS_SAMPLE,
                     cache: Optional[SkeletonVectorCache] = None
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Population mean/std for a dataset's cached skeletons.

    Stable standardization statistics used when a dataset has no vector
    cache: pool-only statistics depend on the (connectivity-skewed) pool
    composition and distort the geometry between query and candidates. The
    stats are computed once from a bounded sample of cached skeletons and
    persisted under the selected cache namespace's
    ``morphology/population_stats.json`` for reuse. The legacy visualization
    cache may extend a sparse sample with a version sibling; the raw-only
    cache never borrows simplified/shared skeletons.
    Returns (None, None) when no statistics can be estimated.
    """
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    vc = cache or SkeletonVectorCache(dataset, project_root=str(root),
                                      verbose=False)
    stats_file = vc.morph_dir / "population_stats.json"
    if stats_file.exists():
        try:
            data = json.loads(stats_file.read_text(encoding='utf-8'))
            if (data.get("dataset") == dataset and data.get("dim") == VECTOR_DIM
                    and data.get("sample_cap") == max_sample
                    and int(data.get("n", 0)) >= MIN_POPULATION_STATS_SKELETONS):
                return (np.asarray(data["mean"], dtype=float),
                        np.asarray(data["std"], dtype=float))
        except Exception:
            pass

    files = vc._discover_skeleton_files()

    # Level guard: the statistics must match the vector basis (raw). Once
    # raw skeletons are replaced by the simplified cache (NeuPrint), the
    # on-disk sample can no longer be vectorized at the right level; fall
    # back to the vector cache's own raw meta stats, which were computed
    # from the same feature schema at fetch time.
    basis = (vc._load_meta() or {}).get("vector_basis") or VECTOR_BASIS_RAW
    if (not vc.raw_only and
            _skeleton_folder_level(dataset, str(root)) != basis):
        data = vc.load()
        if data is not None:
            meta = data.get("meta") or {}
            m = meta.get("mean")
            s = meta.get("std")
            if m is not None and s is not None:
                mm = np.asarray(m, dtype=float)
                ss = np.asarray(s, dtype=float)
                if mm.shape == (VECTOR_DIM,) and ss.shape == (VECTOR_DIM,):
                    return mm, ss
        return None, None

    # A raw comparison may have vector rows but no persisted raw files (for
    # example after an interrupted run). Its metadata is still a valid
    # raw-vector standardization baseline once the cache is large enough;
    # never borrow the shared visualization cache or a sibling's simplified
    # files.
    if vc.raw_only and len(files) == 0:
        data = vc.load()
        if data is not None and len(data["bodyIds"]) >= MIN_POPULATION_STATS_SKELETONS:
            meta = data.get("meta") or {}
            m, s = meta.get("mean"), meta.get("std")
            if m is not None and s is not None:
                mm, ss = np.asarray(m, dtype=float), np.asarray(s, dtype=float)
                if mm.shape == (VECTOR_DIM,) and ss.shape == (VECTOR_DIM,):
                    return mm, ss

    # FAFB: the raw store no longer mirrors the whole-brain bundle. When the
    # cache is too sparse for the meta fallback, sample the healed zip and
    # resolve through the local loader (repair caches / zip).
    if (is_fafb_dataset(dataset) and len(files) == 0
            and not vc.mesh_only):
        zip_ids: List[int] = []
        try:
            bundle = _fafb_bundle(dataset, str(root))
            if bundle is not None:
                zip_ids = sorted(int(b) for b in bundle.ids())
                bundle.close()
        except Exception:
            zip_ids = []
        if zip_ids:
            rng = np.random.default_rng(0)
            if len(zip_ids) > max_sample:
                zip_ids = [zip_ids[i] for i in
                           rng.choice(len(zip_ids), max_sample, replace=False)]
            trees = _resolve_fafb_skeleton_trees(
                dataset, zip_ids, project_root=str(root))
            rows = [row for row in (
                vc._in_memory_vector_row(bid, neuron)
                for bid, neuron in trees.items()) if row is not None]
            vecs = [np.concatenate([r[1], r[2]]) for r in rows]
            if len(vecs) >= MIN_POPULATION_STATS_SKELETONS:
                mat = np.asarray(vecs, dtype=float)
                mu = mat.mean(axis=0)
                sd = np.where(mat.std(axis=0) <= 0, 1.0, mat.std(axis=0))
                try:
                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    stats_file.write_text(json.dumps({
                        "dataset": dataset, "dim": VECTOR_DIM,
                        "n": len(vecs), "sample_cap": max_sample,
                        "mean": mu.tolist(), "std": sd.tolist()}), encoding="utf-8")
                except Exception:
                    pass
                return mu, sd
        return None, None

    # Too few cached skeletons for stable stats: sample from the version
    # sibling's cache instead — it contains the same neurons (shared
    # reconstruction, e.g. male-cns v1.0 <- v0.9), and the sparse local
    # cache may be morphologically skewed (e.g. one query's transient
    # fetches), which would bias the statistics. Only a LARGER sibling
    # cache is used (the sibling may itself be the sparse one).
    if len(files) < MIN_POPULATION_STATS_SKELETONS and not vc.raw_only:
        sibling_files: List[str] = []
        for skel_dir in _sibling_skeleton_dirs(dataset, root):
            sf = sorted(
                [str(p) for p in skel_dir.rglob("*.pkl")
                 if "raw_skeletons" not in p.parts]
                + [str(p) for p in skel_dir.rglob("*.pkl.zst")
                   if "raw_skeletons" not in p.parts]
                + [str(p) for p in skel_dir.rglob("*.swc.gz")
                   if "raw_skeletons" not in p.parts]
            )
            if len(sf) > len(sibling_files):
                sibling_files = sf
        if len(sibling_files) > len(files):
            files = sibling_files

    if not files:
        return None, None
    if len(files) > max_sample:
        rng = np.random.default_rng(0)
        files = [files[i] for i in
                 rng.choice(len(files), max_sample, replace=False)]
    try:
        rows = vc._vectorize_parallel(files)
        if len(rows) != len(files):
            # A broken worker pool can silently drop rows; recompute
            # sequentially so the statistics stay deterministic.
            raise ValueError("parallel vectorization incomplete")
    except Exception:
        rows = [_vectorize_one_file(p) for p in files]
    # One representation per dataset: mixed skeleton/mesh samples would
    # bias the statistics (different feature semantics in one schema).
    ok_rows = [r for r in rows if r is not None]
    if ok_rows:
        from collections import Counter
        canonical = Counter(r[3] for r in ok_rows).most_common(1)[0][0]
        ok_rows = [r for r in ok_rows if r[3] == canonical]
    vecs = [np.concatenate([r[1], r[2]]) for r in ok_rows]
    if not vecs:
        return None, None
    mat = np.asarray(vecs, dtype=float)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    try:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps({
            "dataset": dataset,
            "dim": VECTOR_DIM,
            "n": len(vecs),
            "sample_cap": max_sample,
            "mean": mu.tolist(),
            "std": sd.tolist(),
        }), encoding="utf-8")
    except Exception:
        pass
    return mu, sd


# =============================================================================
# On-demand skeleton fetching
# =============================================================================

def _fetch_neuprint_skeleton(dataset: str, body_id: int):
    """Fetch one skeleton from a NeuPrint dataset.

    ``neuprint.fetch_skeleton`` returns a skeleton DataFrame; convert it to a
    navis TreeNeuron before caching.
    """
    from neuprint import Client, fetch_skeleton, set_default_client
    try:
        from utils.token_manager import token_manager
        token = token_manager.get_neuprint_token()
    except Exception:
        token = ""
    client = Client("neuprint.janelia.org",
                    dataset=neuprint_dataset_name(dataset), token=token)
    set_default_client(client)
    df = fetch_skeleton(body_id)
    if df is None or len(df) == 0:
        return None
    try:
        nrn = navis.TreeNeuron(df)
        # navis' default soma detection flags every node with radius >= 1
        # (neuprint radii are in nm) as soma; a whole-neuron "soma" would
        # freeze the skeleton at full resolution during downsampling and
        # distort the soma_radius feature.
        soma = nrn.soma
        if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
            nrn.soma = None
        return nrn
    except Exception:
        return None


def _fetch_cave_skeleton(dataset: str, body_id: int,
                         project_root: Optional[str] = None,
                         use_cache: bool = True):
    """Legacy compatibility fetch; production FAFB uses ``_fetch_cave_mesh``."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(
        dataset=_dataset_folder(dataset), project_root=project_root,
        verbose=False,
    )
    return fetcher.fetch_skeleton(
        body_id_to_api_int(body_id), use_cache=use_cache,
        simplify_mesh=0.0, denoise_twigs=None,
    )


def _fetch_cave_mesh(dataset: str, body_id: int,
                     project_root: Optional[str] = None,
                     use_cache: bool = True,
                     soma_pos=None):
    """Fetch one prepared MeshNeuron from the FAFB/CAVE path."""
    from cave_data_fetcher import CAVEDataFetcher
    fetcher = CAVEDataFetcher(
        dataset=_dataset_folder(dataset), project_root=project_root,
        verbose=False,
    )
    return fetcher.fetch_fafb_mesh(
        body_id_to_api_int(body_id),
        use_cache=use_cache,
        simplify_mesh=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        soma_pos=soma_pos,
    )


def fetch_skeleton_on_demand(dataset: str, body_id: int,
                             project_root: Optional[str] = None,
                             persist: bool = True,
                             level: str = VECTOR_BASIS_RAW,
                             raw_cache: Optional[SkeletonVectorCache] = None,
                             vector_cache: Optional[SkeletonVectorCache] = None,
                             soma_pos=None,
                             simplification: int = 0,
                             banc_resolution: str = "l2"
                             ) -> Optional[object]:
    """Fetch one dataset-native neuron if missing.

    NeuPrint datasets use ``neuprint.fetch_skeleton`` and persist raw
    ``TreeNeuron`` objects through the shared compress pipeline
    (``simplification`` percent removed, default 0 — the cache stores raw
    skeletons, and simplification is a visualization-time concern,
    recorded in the ``.swc.zst`` header). FAFB resolves through
    :func:`load_local_release_skeletons` (repair caches -> raw cache ->
    healed zip -> extrusion check/repair -> CAVE skeletonization) and BANC
    fetches its public-release SWC directly — both ALWAYS come back as
    ``TreeNeuron`` skeletons.

    Vectorization always runs on the RAW fetched neuron and is persisted to
    the standalone vector cache BEFORE the simplified on-disk file is
    written, so the on-disk simplification level never affects vectors.

    ``persist=False`` is a compatibility escape hatch: it skips the
    write-back paths (the NeuPrint ``.swc.zst`` persist, the FAFB
    vector-cache write, BANC's public-bucket caching) but never forces a
    network round-trip when a local cache can serve.

    ``level`` is retained for compatibility with older callers, but is
    deliberately ignored after validation for NeuPrint; FAFB/BANC keep
    their release representations (level-0 raw skeletons).

    Callers that need a FAFB MESH (never a skeleton) must go through
    :func:`_fetch_cave_mesh` / ``CAVEDataFetcher.fetch_fafb_mesh``
    directly — this fetcher is skeleton-native for every dataset kind.
    """
    body_id = _canonical_dataset_body_id(dataset, body_id)
    level = str(level).lower()
    if level not in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
        raise ValueError(f"Invalid level: {level} (raw|simp90)")
    # None is never accepted here: 0 (raw) is the explicit escape.
    _simplification_factor(simplification)
    # The simplification pipeline is NeuPrint-only: FAFB/BANC local releases
    # fetch source representations and must never be re-leveled here.
    if is_local_connectome_dataset(dataset):
        simplification = 0
    root = Path(project_root) if project_root else Path(__file__).parent.parent

    if is_banc_dataset(dataset):
        # BANC fetches official SWCs from the public release bucket (the
        # CAVE graphene/mesh path below is FAFB-only).
        try:
            import banc_public_data
        except ImportError:
            return None
        neuron = banc_public_data.fetch_banc_swc(
            dataset, body_id, resolution=banc_resolution,
            project_root=str(root), use_cache=bool(persist))
        if neuron is not None and persist:
            cache_fetched_skeleton_vectors(
                dataset, {body_id: neuron}, project_root=str(root),
                vector_cache=vector_cache, verbose=False)
        return neuron

    if is_flywire_dataset(dataset):
        # FAFB resolves through the same release loader every other
        # comparison consumer uses (repair caches -> raw cache -> healed
        # zip -> extrusion check/repair -> CAVE skeletonization), so this
        # fetcher is skeleton-native for FAFB too.  History: this branch
        # used to return the CAVE prepared MESH, and a now-dead block
        # below it served skeletons behind an unconditional `return mesh`
        # — unreachable dead code, and the reason a mesh-where-skeleton
        # was possible at all (the 2026-09-29 reverse-scene defect class,
        # plan-tmvev-reverse-scene-loader-defects.md).  The mesh cache is
        # never consulted here; mesh consumers call _fetch_cave_mesh
        # directly.  `persist` / `soma_pos` are accepted for caller
        # compatibility: the loader owns FAFB caching and the extrusion
        # default (True).
        trees = load_local_release_skeletons(
            dataset, [body_id], project_root=str(root))
        neuron = trees.get(body_id)
        if neuron is None:
            # loader stages key by canonical or plain-int ids depending on
            # where the entry resolved
            try:
                neuron = trees.get(int(body_id))
            except (TypeError, ValueError):
                neuron = None
        if neuron is None:
            return None
        if persist:
            cache_fetched_skeleton_vectors(
                dataset, {body_id: neuron}, project_root=str(root),
                vector_cache=vector_cache, verbose=False)
        return neuron

    # Every fetch-level caller uses the same raw cache, regardless of the
    # legacy level value supplied by an older visualization caller.
    if raw_cache is None:
        try:
            raw_cache = find_similar_raw_cache(
                dataset, project_root=str(root), verbose=False,
            )
        except Exception:
            raw_cache = None

    if raw_cache is not None:
        cached = raw_cache.load_skeleton(body_id, simplification=simplification)
        if cached is not None:
            # A raw-file hit should also populate the raw vector cache when
            # a previous interrupted run left only the skeleton behind.
            cache_fetched_skeleton_vectors(
                dataset, {body_id: cached}, project_root=str(root),
                vector_cache=vector_cache or raw_cache, verbose=False,
            )
            return cached

    neuron = _fetch_neuprint_skeleton(
        dataset, _api_dataset_body_id(dataset, body_id)
    )

    if neuron is None:
        return None

    # Vectorize the RAW skeleton at fetch time and persist the vector before
    # any simplification: the vector cache is standalone and always raw-basis.
    cache_fetched_skeleton_vectors(
        dataset, {body_id: neuron}, project_root=str(root),
        vector_cache=vector_cache or raw_cache, verbose=False,
    )

    if persist and raw_cache is not None:
        raw_cache.persist_skeletons({body_id: neuron},
                                    simplification=simplification)
    return neuron


# Keep a compatibility seam for callers/tests that intentionally replace the
# singular fetcher (for example, an offline fixture). Normal production calls
# never take this branch; the optimized path below remains the default.
_SINGLE_FETCH_IMPLEMENTATION = fetch_skeleton_on_demand


def _fetch_neuprint_batch_with_progress(
        batch_ids, *, client, max_threads, on_neuron=None,
        missing_swc: str = "warn") -> list:
    """Fetch one batch of NeuPrint SWCs with a per-neuron completion hook.

    Vendored from ``navis.interfaces.neuprint.fetch_skeletons`` (navis 1.5.0)
    so every completed skeleton can be reported: the upstream wrapper only
    updates its own hidden tqdm counter, which never reaches the UI.  The
    behavior is intentionally identical to navis: one ``fetch_neurons``
    metadata query per batch (soma/instance/status), then a
    ``ThreadPoolExecutor`` with ``max_threads`` workers fetching the
    individual SWCs concurrently, ``missing_swc`` warn/skip handling, and
    the same TreeNeuron construction (``__fetch_skeleton``).

    ``on_neuron(done, batch_total)`` is called from the caller's thread for
    every completed future (successful or failed) - exactly the per-neuron
    cadence of navis' internal bar.  Returns the fetched TreeNeurons.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from navis.interfaces import neuprint as neu
    from navis.interfaces.neuprint import __fetch_skeleton

    meta, _roi_info = neu.fetch_neurons(
        neu.NeuronCriteria(bodyId=list(batch_ids)), client=client)
    if meta is None or meta.empty:
        return []
    # Make sure there is a somaLocation and somaRadius column (navis does
    # the same before the row iteration below).
    if "somaLocation" not in meta.columns:
        meta["somaLocation"] = None
    if "somaRadius" not in meta.columns:
        meta["somaRadius"] = None

    neurons = []
    done = 0
    batch_total = len(meta)
    with ThreadPoolExecutor(max_workers=max(1, int(max_threads))) as executor:
        futures = {}
        for row in meta.itertuples():
            future = executor.submit(
                __fetch_skeleton, row, client=client,
                with_synapses=False, missing_swc=missing_swc, heal=False)
            futures[future] = row.bodyId
        for future in as_completed(futures):
            done += 1
            try:
                neuron = future.result()
            except Exception as exc:  # navis prints and continues
                print(f"{futures[future]} generated an exception: {exc}")
                neuron = None
            if neuron is not None:
                neurons.append(neuron)
            if on_neuron is not None:
                on_neuron(done, batch_total)
    return neurons


def _normalize_fetched_neurons(dataset: str, neurons: Dict[Union[int, str], object]
                               ) -> Dict[Union[int, str], object]:
    """Normalize fetched neurons at the dataset boundary.

    Every fetch path is skeleton-native now (FAFB resolves through
    :func:`load_local_release_skeletons`), so this coerces non-TreeNeuron
    objects via ``navis.TreeNeuron`` and clears multi-node ``soma``
    (navis' default soma detection flags every radius >= 1 node, so a
    whole-neuron "soma" would freeze downsampling).  Returns the
    canonical-keyed mapping with the surviving neurons.
    """
    out: Dict[Union[int, str], object] = {}
    for fallback_id, neuron in neurons.items():
        try:
            if not isinstance(neuron, navis.TreeNeuron):
                neuron = navis.TreeNeuron(neuron)
            neuron.id = _api_dataset_body_id(dataset, fallback_id)
            soma = getattr(neuron, "soma", None)
            if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
                neuron.soma = None
            out[_canonical_dataset_body_id(dataset, fallback_id)] = neuron
        except Exception:
            continue
    return out


def _fetch_banc_skeleton_batch(dataset, missing, root, persist, requested,
                               loaded, progress_callback, cancel_event
                               ) -> Dict[int, object]:
    """BANC per-neuron SWCs from the public release bucket.

    ``fetch_banc_swc`` persists its own level-0 raw cache entry (its own
    three-step chain: 888 L2 -> 888 full -> v626 pcg), so this branch never
    touches a ``raw_cache`` object itself.
    """
    import banc_public_data

    out: Dict[int, object] = {}
    for bid in missing:
        if cancel_event is not None and cancel_event.is_set():
            break
        try:
            neuron = banc_public_data.fetch_banc_swc(
                dataset, bid, project_root=str(root),
                use_cache=bool(persist))
        except Exception:
            neuron = None
        if neuron is not None:
            out[_canonical_dataset_body_id(dataset, bid)] = neuron
        if progress_callback:
            done = min(len(requested), len(loaded) + len(out))
            progress_callback(
                done, len(requested),
                f"Fetching BANC skeletons ({done}/{len(requested)})")
    return out


def _fetch_neuprint_skeleton_batch(
        dataset, missing, root, persist, requested, loaded, raw_cache,
        client, batch_size, max_threads, progress_callback,
        cancel_event, temp_normalized, simplification) -> Dict[int, object]:
    """Batched NeuPrint SWCs with the crash-resume staging/persist pipeline.

    One NeuPrint client is shared by all bounded requests. Raw skeletons are
    staged to ``skeletons/_temp_cache`` (fast zstd-3) as each batch completes
    so a crash loses at most the in-flight batch; a standalone persist worker
    vectorizes the RAW neurons (appending rows BEFORE the simplified
    ``.swc.zst`` is written) and clears the staging entries. The final
    pending-row merge checkpoint runs here, so the caller treats this pull as
    pipelined and skips its own persistence pass.
    """
    from neuprint import Client, set_default_client
    from navis.interfaces import neuprint as neu

    if client is None:
        try:
            from utils.token_manager import token_manager
            token = token_manager.get_neuprint_token()
        except Exception:
            token = ""
        client = Client(
            "neuprint.janelia.org",
            dataset=neuprint_dataset_name(dataset), token=token)
    try:
        set_default_client(client)
    except Exception:
        pass

    effective_batch_size = max(1, int(batch_size))
    effective_threads = max(1, int(max_threads))
    total_missing = len(missing)
    total_batches = ((total_missing + effective_batch_size - 1)
                     // effective_batch_size)

    # Pipeline: raw staging + simplification/cache writing run on standalone
    # threads so the network fetch loop is never interrupted by CPU/disk
    # work. Unbounded queues keep the fetch loop from ever blocking (memory
    # is bounded by ``fetched_by_id``; the staging worker runs well above the
    # fetch rate, so its backlog stays ~0).
    import queue as _queue
    import threading
    staging_queue: "_queue.Queue" = _queue.Queue()
    persist_queue: "_queue.Queue" = _queue.Queue()
    persist_failures = {"n": 0}
    fetched_by_id: Dict[int, object] = {}

    def _staging_worker() -> None:
        """Stage raw skeletons to disk ASAP, then forward to persist.

        Existing staging files (crash leftovers being reprocessed) are
        skipped - they are already on disk.
        """
        while True:
            item = staging_queue.get()
            if item is None:
                staging_queue.task_done()
                break
            try:
                if persist and raw_cache is not None:
                    temp_dir = raw_cache.temp_cache_dir()
                    for body_id, neuron in item.items():
                        try:
                            target = temp_dir / f"{body_id}.swc.zst"
                            if not target.exists():
                                raw_cache.write_temp_skeleton(
                                    body_id, neuron)
                        except Exception:
                            continue
            except Exception:
                persist_failures["n"] += 1
            finally:
                persist_queue.put(item)
                staging_queue.task_done()

    def _persist_worker() -> None:
        processed = 0
        while True:
            item = persist_queue.get()
            if item is None:
                persist_queue.task_done()
                break
            try:
                processed += len(item)
                # Vectorize the RAW neurons first and append their rows
                # BEFORE the simplified file is written ("cached only after
                # vectorization"); the main + pending append makes per-batch
                # appends O(batch).
                rows = []
                for body_id, neuron in item.items():
                    try:
                        rep = _neuron_rep(neuron)
                        if rep in {"skeleton", "mesh"}:
                            _, vector = vectorize_neuron(neuron)
                            rows.append((
                                _canonical_dataset_body_id(
                                    dataset, body_id),
                                vector, rep,
                            ))
                    except Exception as exc:
                        # Glitchy fetches (empty/partial SWC) produce neurons
                        # without nodes; they are skipped from the vector
                        # cache, never from the result set.
                        print(
                            f"[morphology] vectorization skipped "
                            f"for {body_id}: {exc}"
                        )
                        continue
                # The NeuPrint path is never the FAFB mesh path, so raw
                # vector rows are always persisted here.
                if rows and raw_cache is not None:
                    try:
                        raw_cache.append_vectors(
                            rows, vector_basis=VECTOR_BASIS_RAW)
                    except Exception as exc:
                        print(
                            f"[morphology] vector append failed: "
                            f"{exc}"
                        )
                if persist and raw_cache is not None:
                    raw_cache.persist_skeletons(
                        item, simplification=simplification)
                    # The final skeleton is on disk: staging entries are no
                    # longer needed (resume-safe).
                    raw_cache.delete_temp_skeletons(list(item.keys()))
                if progress_callback:
                    progress_callback(
                        min(len(requested), processed), len(requested),
                        f"Vectorizing + caching skeletons "
                        f"({min(len(requested), processed)}/"
                        f"{len(requested)})")
            except Exception:
                persist_failures["n"] += 1
            finally:
                persist_queue.task_done()

    persist_thread = threading.Thread(
        target=_persist_worker, daemon=True,
        name=f"skeleton-persist-{_dataset_folder(dataset)}",
    )
    staging_thread = threading.Thread(
        target=_staging_worker, daemon=True,
        name=f"skeleton-staging-{_dataset_folder(dataset)}",
    )
    persist_thread.start()
    staging_thread.start()

    # Temp-pending neurons from a crashed previous run are already on disk:
    # route them through the pipeline without a network fetch (the staging
    # stage skips rewriting existing temp files).
    if temp_normalized:
        staging_queue.put(dict(temp_normalized))

    try:
        for batch_index, start in enumerate(
                range(0, total_missing, effective_batch_size), start=1):
            # Cooperative cancel: stop submitting new batches; the in-flight
            # one finishes and its results are persisted below (resume-safe).
            if cancel_event is not None and cancel_event.is_set():
                break
            batch_ids = missing[start:start + effective_batch_size]
            # Report the batch BEFORE the network call: the first batch can
            # take a minute or more, so the batch counter keeps the pull
            # visibly alive between the after-batch updates.
            done_so_far = min(len(requested), len(loaded) + start)
            if progress_callback:
                progress_callback(
                    done_so_far, len(requested),
                    f"Fetching skeletons - batch {batch_index}/"
                    f"{total_batches} ({done_so_far}/{len(requested)})")

            # Per-neuron progress: the vendored loop mirrors navis' own
            # parallel fetch but reports every completed skeleton.
            def _neuron_progress(done, _batch_total, _start=start,
                                 _batch_index=batch_index):
                if progress_callback:
                    current = min(
                        len(requested), len(loaded) + _start + done)
                    progress_callback(
                        current, len(requested),
                        f"Fetching skeletons - batch {_batch_index}/"
                        f"{total_batches} ({current}/{len(requested)})")

            try:
                batch_neurons = _fetch_neuprint_batch_with_progress(
                    batch_ids,
                    client=client,
                    max_threads=max(
                        1, min(effective_threads, len(batch_ids))),
                    on_neuron=_neuron_progress,
                )
            except Exception:
                # Fall back to the upstream wrapper if a future navis
                # release changes the vendored internals; only the per-batch
                # progress granularity is lost.
                batch_df = pd.DataFrame({"bodyId": batch_ids})
                result = neu.fetch_skeletons(
                    batch_df,
                    parallel=True,
                    max_threads=max(
                        1, min(effective_threads, len(batch_ids))),
                    missing_swc="warn",
                    client=client,
                )
                batch_neurons = list(result or [])
            batch_map = {}
            for index, neuron in enumerate(batch_neurons):
                neuron_id = getattr(neuron, "id", None)
                if neuron_id is None and index < len(batch_ids):
                    neuron_id = batch_ids[index]
                try:
                    batch_map[
                        _canonical_dataset_body_id(dataset, neuron_id)
                    ] = neuron
                except (TypeError, ValueError):
                    continue
            # Normalize at the batch boundary, stage the raw skeletons for
            # crash-resume, and hand the batch to the persist worker while
            # the fetch loop continues.
            normalized_batch = _normalize_fetched_neurons(
                dataset, batch_map)
            fetched_by_id.update(normalized_batch)
            staging_queue.put(normalized_batch)
            if progress_callback:
                done = min(len(requested), len(loaded) + len(fetched_by_id))
                progress_callback(
                    done, len(requested),
                    f"Fetching skeletons ({done}/{len(requested)})")
    finally:
        # Stop the workers only after the in-flight batch's results were
        # handed over; both workers drain their queues (staging every fetched
        # batch, persisting everything fetched, incl. after a cancel).
        staging_queue.put(None)
        staging_queue.join()
        persist_queue.put(None)
        persist_queue.join()

    # Final merge checkpoint folds any remaining pending rows into the main
    # parquet. The staged batches are already persisted; a failed merge only
    # delays folding (the next append/load merges them).
    if raw_cache is not None:
        try:
            raw_cache._merge_pending()
        except Exception as exc:
            print(f"[morphology] final vector merge failed: {exc}")
    return fetched_by_id


def fetch_skeletons_on_demand_batch(
        dataset: str, body_ids, project_root: Optional[str] = None,
        persist: bool = True, level: str = VECTOR_BASIS_RAW,
        batch_size: int = NEUPRINT_FETCH_BATCH_SIZE,
        max_threads: int = NEUPRINT_FETCH_MAX_THREADS,
        progress_callback=None, client=None,
        raw_cache: Optional[SkeletonVectorCache] = None,
        vector_cache: Optional[SkeletonVectorCache] = None,
        simplification: int = 0,
        cancel_event=None) -> Dict[int, object]:
    """Cache-aware online fetch for one dataset family.

    FAFB resolves in ONE call through :func:`load_local_release_skeletons`
    (repair caches -> raw cache -> healed zip -> extrusion check/repair ->
    CAVE skeletonization) and returns TreeNeuron skeletons — the same
    guarantee as the singular fetcher, without a per-id loop.  BANC and
    NeuPrint are thin dispatches over two independent per-source fetchers,
    which keep their own representations and storage:
    :func:`_fetch_neuprint_skeleton_batch` (batched TreeNeuron SWCs, staged
    and persisted through the shared simplify+compress pipeline) and
    :func:`_fetch_banc_skeleton_batch` (public-bucket TreeNeuron SWCs).
    History: FAFB used to dispatch to a CAVE prepared-mesh batch
    (``_fetch_fafb_mesh_batch``, MeshNeuron-only) while every production
    caller was gated to the loader — the mesh path was what made a
    mesh-where-skeleton possible at all (the 2026-09-29 reverse-scene
    defect class, plan-tmvev-reverse-scene-loader-defects.md).

    The surrounding shared cache transaction (raw-cache membership scan,
    crash-resume staging discovery, and the non-pipelined normalize +
    vectorize + persist pass) is dataset-agnostic and lives here so the
    per-source fetchers only own their fetch loop.

    Vectorization always runs on the RAW fetched neurons and is persisted to
    the standalone vector cache BEFORE the simplified on-disk files are
    written. Cached files are loaded with the requested target level, so a
    file stored at a lower level is re-simplified on read.

    ``progress_callback(done, total, message)`` is optional and is called
    at batch boundaries and, during the online NeuPrint fetch, once per
    completed skeleton (the per-neuron cadence of the terminal bar); the
    FAFB loader path reports one boundary callback. ``cancel_event``
    (threading.Event) is optional: when set, no new fetch batch is started
    and the already-fetched skeletons are still vectorized/persisted
    (resume-safe; the FAFB loader path is not interruptible mid-call).
    The returned mapping is keyed by canonical body ID (string for
    FlyWire, int otherwise) and contains only skeletons that were
    available or successfully fetched.
    """
    requested = []
    seen = set()
    if body_ids is not None:
        for body_id in body_ids:
            try:
                bid = _canonical_dataset_body_id(dataset, body_id)
            except (TypeError, ValueError):
                continue
            if bid not in seen:
                seen.add(bid)
                requested.append(bid)

    level = str(level).lower()
    if level not in (VECTOR_BASIS_RAW, VECTOR_BASIS_SIMP90):
        raise ValueError(f"Invalid level: {level} (raw|simp90)")
    # None is never accepted here: 0 (raw) is the explicit escape.
    _simplification_factor(simplification)
    # The simplification pipeline is NeuPrint-only: FAFB/BANC local releases
    # fetch raw source representations and must never be re-leveled here.
    if is_local_connectome_dataset(dataset):
        simplification = 0
    if not requested:
        return {}

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    flywire = is_flywire_dataset(dataset)
    banc = is_banc_dataset(dataset)
    if banc:
        # BANC SWCs cache raw (level 0) inside banc_public_data itself.
        simplification = 0
    if flywire:
        # FAFB resolves through the same release loader every other
        # comparison consumer uses — batch-native and skeleton-guaranteed.
        # `persist` / `level` / `simplification` are accepted for caller
        # compatibility: the loader owns FAFB caching, its stores are
        # durable (cave_skeletons / extrusion_fixes are never re-persisted
        # into a raw cache) and its extrusion default (True) applies.
        # A pre-set cancel_event still short-circuits here (no fetch is
        # started); the loader call itself is not interruptible mid-way.
        if cancel_event is not None and cancel_event.is_set():
            return {}
        trees = load_local_release_skeletons(
            dataset, requested, project_root=str(root))
        if trees and persist:
            cache_fetched_skeleton_vectors(
                dataset, trees, project_root=str(root),
                vector_cache=vector_cache, verbose=False)
        out: Dict[Union[int, str], object] = {}
        for bid in requested:
            neuron = trees.get(bid)
            if neuron is None:
                # loader stages key by canonical or plain-int ids depending
                # on where the entry resolved
                try:
                    neuron = trees.get(int(bid))
                except (TypeError, ValueError):
                    neuron = None
            if neuron is not None:
                out[bid] = neuron
        if progress_callback:
            progress_callback(len(out), len(requested),
                              "Skeletons ready (FAFB release loader)")
        return out
    loaded: Dict[int, object] = {}
    missing = []
    raw_lookup_cache = raw_cache
    # The pipelined NeuPrint path (fetch loop + standalone persist worker)
    # is enabled only for the batched NeuPrint branch below.
    pipeline = False
    if raw_lookup_cache is None:
        # Every NeuPrint caller gets the same first lookup into the shared
        # raw SWC cache. The same cache is also the raw-file write target.
        try:
            raw_lookup_cache = find_similar_raw_cache(
                dataset, project_root=str(root), verbose=False)
        except Exception:
            raw_lookup_cache = None
        if raw_cache is None:
            raw_cache = raw_lookup_cache

    if raw_lookup_cache is not None:
        # Cache membership is decided with ONE directory scan instead of one
        # per-id lookup: a full-dataset pull with an almost-empty cache used
        # to spend ~30 minutes re-scanning the cache directory (rglob per
        # missing id) before the first batch even completed.  Only ids that
        # ARE cached go through load_skeleton (which loads/migrates the
        # actual neuron); the rest go straight to the fetch list.  Caches
        # without a directory scanner (e.g. test fakes) keep the per-id
        # lookup semantics.
        cached_ids: set = set()
        scan_ok = False
        scanner = getattr(raw_lookup_cache, "_discover_skeleton_files", None)
        if callable(scanner):
            try:
                for path in scanner():
                    try:
                        cached_ids.add(
                            _canonical_dataset_body_id(
                                dataset, _skeleton_body_id(path)
                            )
                        )
                    except (TypeError, ValueError):
                        continue
                scan_ok = True
            except Exception:
                cached_ids = set()
        # Crash-resume staging: raw skeletons written by a previous
        # interrupted pull live in ``skeletons/_temp_cache``.  They are loaded
        # from disk (never re-fetched) and routed through the pipeline like
        # freshly-fetched batches; their temp entries are removed once the
        # final simplified files are persisted.
        temp_pending: Dict[int, object] = {}
        temp_dir_fn = getattr(raw_lookup_cache, "temp_cache_dir", None)
        if callable(temp_dir_fn):
            try:
                temp_dir = temp_dir_fn()
            except Exception:
                temp_dir = None
            if temp_dir is not None and temp_dir.is_dir():
                try:
                    for path in temp_dir.glob("*.swc.zst"):
                        try:
                            bid = _canonical_dataset_body_id(
                                dataset, _skeleton_body_id(path))
                        except (TypeError, ValueError):
                            continue
                        neuron = _load_cached_skeleton_file(str(path))
                        if neuron is not None:
                            temp_pending.setdefault(bid, neuron)
                except Exception:
                    temp_pending = {}
        for bid in requested:
            if bid in temp_pending:
                continue  # processed from temp by the pipeline below
            if scan_ok and bid not in cached_ids:
                missing.append(bid)
                continue
            neuron = raw_lookup_cache.load_skeleton(
                bid, simplification=simplification)
            if neuron is None:
                missing.append(bid)
            else:
                loaded[bid] = neuron
    else:
        # If cache construction failed, still fetch online; persistence is
        # best-effort and the in-memory result remains usable.
        missing = list(requested)
        temp_pending = {}

    if progress_callback:
        progress_callback(len(loaded), len(requested),
                          f"Neuron cache ({len(loaded)}/{len(requested)})")

    if missing or temp_pending:
        fetched_by_id: Dict[int, object] = {}
        # Normalize the temp-pending neurons once; they flow through the same
        # pipeline as fetched batches (vectorize + persist + temp cleanup)
        # without a network round-trip.
        temp_normalized: Dict[int, object] = {}
        if temp_pending:
            temp_normalized = _normalize_fetched_neurons(
                dataset, temp_pending)
            fetched_by_id.update(temp_normalized)
        if fetch_skeleton_on_demand is not _SINGLE_FETCH_IMPLEMENTATION:
            # Preserve an explicit singular-fetch override. This is useful
            # for offline clients and keeps downstream integrations that
            # monkeypatch the old public seam working while production uses
            # the batch API below.
            for bid in missing:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    neuron = fetch_skeleton_on_demand(
                        dataset, bid, project_root=str(root), persist=persist,
                        level=level, raw_cache=raw_cache,
                        vector_cache=vector_cache)
                except TypeError as exc:
                    # A few older integrations exposed the original helper
                    # without the newer cache/level keywords.  The batch
                    # phase below owns persistence, so old overrides remain
                    # compatible without changing the new raw-cache path.
                    if not any(k in str(exc) for k in
                               ("persist", "level", "raw_cache", "vector_cache",
                                "soma_pos")):
                        raise
                    try:
                        neuron = fetch_skeleton_on_demand(
                            dataset, bid, project_root=str(root),
                            persist=persist)
                    except TypeError as older_exc:
                        if not any(k in str(older_exc)
                                   for k in ("persist", "soma_pos")):
                            raise
                        try:
                            neuron = fetch_skeleton_on_demand(
                                dataset, bid, project_root=str(root),
                                persist=persist)
                        except TypeError as legacy_exc:
                            if "persist" not in str(legacy_exc):
                                raise
                            try:
                                neuron = fetch_skeleton_on_demand(
                                    dataset, bid,
                                    project_root=str(root))
                            except TypeError as root_exc:
                                if "project_root" not in str(root_exc):
                                    raise
                                neuron = fetch_skeleton_on_demand(
                                    dataset, bid)
                if neuron is not None:
                    fetched_by_id[
                        _canonical_dataset_body_id(dataset, bid)
                    ] = neuron
            if progress_callback:
                progress_callback(
                    len(loaded) + len(fetched_by_id), len(requested),
                    f"Fetching skeletons ({len(loaded) + len(fetched_by_id)}/"
                    f"{len(requested)})")
        elif banc:
            fetched_by_id.update(_fetch_banc_skeleton_batch(
                dataset, missing, root, persist, requested, loaded,
                progress_callback, cancel_event))
        else:
            fetched_by_id.update(_fetch_neuprint_skeleton_batch(
                dataset, missing, root, persist, requested, loaded,
                raw_cache, client, batch_size, max_threads,
                progress_callback, cancel_event, temp_normalized,
                simplification))
            pipeline = True
        if not pipeline:
            # Non-pipelined sources (BANC or a singular-fetch override)
            # normalize at the dataset boundary here — they become
            # TreeNeuron skeletons — and take the shared vectorize+persist
            # pass below. The pipelined NeuPrint path already did all of
            # this inside ``_fetch_neuprint_skeleton_batch``.
            for fallback_id, neuron in fetched_by_id.items():
                try:
                    if not isinstance(neuron, navis.TreeNeuron):
                        neuron = navis.TreeNeuron(neuron)
                    neuron.id = _api_dataset_body_id(dataset, fallback_id)
                    soma = getattr(neuron, "soma", None)
                    if soma is not None and hasattr(soma, "__len__") and len(soma) > 1:
                        neuron.soma = None
                    fetched_by_id[
                        _canonical_dataset_body_id(dataset, fallback_id)
                    ] = neuron
                except Exception:
                    continue

            # Vectorize and persist the raw vectors before the batch returns.
            # A single append keeps the vector cache consistent with the
            # fetched skeleton set and exposes the previously invisible
            # post-fetch wait.
            vector_stats = {"cache_error": None}
            if vector_cache is not None:
                # vector_cache=None (V1 vector path retired): the caller
                # vectorizes under the target schema locally, so the fetcher
                # must not write legacy rows into a cache it was not given.
                vector_stats = cache_fetched_skeleton_vectors(
                    dataset,
                    fetched_by_id,
                    project_root=str(root),
                    vector_cache=vector_cache or raw_cache,
                    progress_callback=progress_callback,
                    progress_offset=len(loaded),
                    progress_total=len(requested),
                    verbose=True,
                )
                if vector_stats.get("cache_error"):
                    print(
                        f"[morphology] vector cache incomplete for {dataset}: "
                        f"{vector_stats['cache_error']}"
                    )

            if persist and fetched_by_id and raw_cache is not None:
                raw_cache.persist_skeletons(fetched_by_id,
                                            simplification=simplification)
                # Temp-pending neurons (crashed previous run) are now fully
                # persisted: their staging entries are no longer needed.
                raw_cache.delete_temp_skeletons(list(fetched_by_id.keys()))

        loaded.update(fetched_by_id)

    if progress_callback:
        progress_callback(len(loaded), len(requested),
                          f"Skeletons ready ({len(loaded)}/{len(requested)})")

    return {bid: loaded[bid] for bid in requested if bid in loaded}


def download_all_skeletons(dataset: str, project_root: Optional[str] = None,
                           max_workers: int = 8, limit: Optional[int] = None,
                           progress_callback=None, cancel_event=None,
                           verbose: bool = True,
                           raw: bool = True, mode: Optional[str] = None,
                           raw_format: str = "swc.zst",
                           simplification: int = 0,
                           batch_size: int = NEUPRINT_FETCH_BATCH_SIZE
                           ) -> Dict[str, object]:
    """Download every missing skeleton of a dataset to the local cache.

    Mirrors the Settings-panel full dataset pull. NeuPrint pulls persist
    ``TreeNeuron`` objects through the shared compress pipeline in
    ``skeletons/raw_skeletons/{bodyId}.swc.zst``: the cache stores raw
    skeletons (``simplification`` percent of nodes removed, 0-90, default
    0 — simplification is a visualization-time concern), with the level
    recorded in each file header.

    FAFB is not supported: bulk skeleton downloads are disabled (the healed
    bundle is a large one-time manual download), and a
    ``FlyWireSkeletonAccessError`` with the manual Codex download
    instructions is raised instead. BANC pulls fetch per-neuron SWCs from
    the public release bucket into the raw skeleton cache (level 0).
    Visualization ``fast`` is not a pull mode for either representation.

    Vectorization always runs on the RAW fetched neurons first and is
    persisted to the standalone vector cache, so the downloaded on-disk
    simplification level never affects analysis vectors.

    ``mode`` and ``raw`` remain compatibility parameters for older callers;
    any accepted mode is normalized to ``"raw"``. NeuPrint requests use
    bounded online batches.
    ``raw_format`` and ``simplification`` are retained for compatibility
    and do not alter BANC raw-SWC caches.
    ``batch_size`` bounds the per-call NeuPrint request size: every call
    repeats a metadata query first, so larger batches amortize it (the
    default 64).  Progress is reported per completed skeleton, so the batch
    size no longer affects progress granularity.
    ``progress_callback(current, total, info)`` and
    ``cancel_event`` (threading.Event) drive the UI; ``limit`` bounds the
    download (tests / smoke runs). Returns a summary dict.
    """
    import threading

    _simplification_factor(simplification)  # validate 0..90 up front

    # Bulk skeleton downloads are disabled for FAFB: the healed zip is a
    # large one-time download that must be fetched manually from the FlyWire
    # Codex and placed by the converter. BANC pulls fetch per-neuron SWCs
    # from the public release bucket instead. On-demand fetches for
    # individual missing FAFB skeletons still work (the fetchers and the
    # release loader skeletonize through the CAVE path).
    if is_fafb_dataset(dataset):
        raise FlyWireSkeletonAccessError(
            flywire_manual_skeleton_instruction(dataset))

    require_flywire_skeleton_access(
        dataset,
        project_root=project_root,
        log=print if verbose else (lambda _message: None),
    )

    root = Path(project_root) if project_root else Path(__file__).parent.parent
    folder = _dataset_folder(dataset)
    requested_mode = str(mode or "raw").strip().lower()
    requested_mode = {
        "raw_skeletons": "raw", "fine95": "fine",
        "fine_skeletons": "fine", "simp90": "fast",
    }.get(requested_mode, requested_mode)
    if requested_mode not in {"raw", "fine", "fast"}:
        raise ValueError("mode must be 'raw', 'fine', or 'fast' (compatibility only)")
    # Pulling is representation-only. ``fast`` and ``fine`` are visualization
    # simplification choices and must never select a different disk format.
    mode = "raw"
    raw_cache = find_similar_raw_cache(
        dataset, project_root=str(root), n_workers=max_workers,
        verbose=verbose, raw_format=raw_format
    )
    skeleton_dir = raw_cache.skeleton_dir
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    # Index of all bodyIds. Read FAFB/BANC IDs as strings before any operation
    # that could route through pandas/Polars numeric inference.
    index: List[Union[int, str]] = []
    table_candidates = [
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.parquet",
        root / "datasets" / folder / f"{folder}_allneurons_neuron_df.csv",
    ]
    table_path = next((path for path in table_candidates if path.exists()), None)
    if table_path is not None:
        try:
            if table_path.suffix.lower() == ".parquet":
                index_frame = pd.read_parquet(table_path, columns=["bodyId"])
            else:
                index_frame = pd.read_csv(
                    table_path,
                    usecols=["bodyId"],
                    dtype={"bodyId": "string"}
                    if is_local_connectome_dataset(dataset) else None,
                )
            index = (
                [normalize_flywire_body_id(b)
                 for b in index_frame["bodyId"].tolist()]
                if is_local_connectome_dataset(dataset) else
                [int(b) for b in index_frame["bodyId"].tolist()]
            )
        except Exception:
            index = []
    if not index:
        index_path = root / "neuron_indexes" / folder / "neuron_index.parquet"
        if index_path.exists() and _has_local_dataset_presence(dataset, root):
            try:
                index_frame = pd.read_parquet(index_path, columns=["bodyId"])
                index = (
                    [normalize_flywire_body_id(b)
                     for b in index_frame["bodyId"].tolist()]
                    if is_local_connectome_dataset(dataset) else
                    [int(b) for b in index_frame["bodyId"].tolist()]
                )
            except Exception:
                index = []

    # Only files from the representation-specific cache count as available.
    existing_paths = [Path(path) for path in raw_cache._discover_skeleton_files()]
    existing = set()
    for path in existing_paths:
        try:
            existing.add(
                _canonical_dataset_body_id(dataset, _skeleton_body_id(path))
            )
        except (TypeError, ValueError):
            continue

    missing = [
        _canonical_dataset_body_id(dataset, b)
        for b in index
        if _canonical_dataset_body_id(dataset, b) not in existing
    ]
    if limit is not None:
        missing = missing[: int(limit)]
    total = len(missing)
    skipped_existing = len(existing)
    if total == 0:
        if verbose:
            print(f"[morphology] download_all_skeletons: "
                  f"{skipped_existing} skeletons already available locally "
                  f"({len(existing)} cached); nothing to fetch.")
        return {"total": 0, "fetched": 0, "skipped_existing": skipped_existing,
                "cancelled": False, "errors": 0, "mode": mode,
                "representation": "skeleton",
                "simplification": int(simplification)}

    if verbose:
        print(f"[morphology] download_all_skeletons: fetching {total} "
              f"skeletons ({dataset})...")
    if progress_callback:
        progress_callback(0, total, f"Fetching skeletons (0/{total})")

    fetched = 0
    errors = 0
    cancelled = False
    cancel_event = cancel_event or threading.Event()

    if cancel_event.is_set():
        cancelled = True
    else:
        def _batch_progress(done, batch_total, message):
            if progress_callback:
                progress_callback(min(done, total), total, message)

        try:
            fetched_map = fetch_skeletons_on_demand_batch(
                dataset,
                missing,
                project_root=str(root),
                persist=True,
                level=VECTOR_BASIS_RAW,
                batch_size=int(batch_size),
                max_threads=max_workers,
                progress_callback=_batch_progress,
                raw_cache=raw_cache,
                vector_cache=raw_cache,
                simplification=simplification,
                cancel_event=cancel_event,
            )
            fetched = len(fetched_map)
            errors = total - fetched
            cancelled = cancel_event.is_set()
        except Exception as exc:
            errors = total
            if verbose:
                print(f"[morphology] batched NeuPrint fetch failed: {exc}")
        finally:
            if progress_callback:
                progress_callback(
                    fetched + errors, total,
                    "Cancelled." if cancelled else "Finished.")

    summary = {
        "total": total,
        "fetched": fetched,
        "skipped_existing": skipped_existing,
        "cancelled": cancelled,
        "errors": errors,
        "mode": mode,
        "representation": "skeleton",
        "simplification": int(simplification),
    }
    if verbose:
        print(f"[morphology] download_all_skeletons: {fetched}/{total} "
              f"fetched, {errors} errors, cancelled={cancelled}, "
              f"simplification={simplification}")
    return summary


# =============================================================================
# MorphologyComparer (runner-facing)
# =============================================================================

def _coverage_factor(coverage: float) -> float:
    """Continuous coverage weight of a type statistic: sqrt(coverage).

    A gentle, monotone penalty with no thresholds: a fully covered type
    passes through untouched (factor 1), a quarter-sampled type keeps half
    its statistic, a barely-sampled type is strongly damped but never
    zeroed or excluded. Applies only when the population size is known.
    """
    if not np.isfinite(coverage):
        return 1.0   # population unknown: no basis to weight
    return float(np.sqrt(min(max(float(coverage), 0.0), 1.0)))


class MorphologyComparer:
    """Query-vs-all morphological similarity search for one dataset.

    The runner script constructs this with all user parameters and calls
    ``find_similar()``; module-level defaults are used when arguments are
    omitted.
    """

    def __init__(
        self,
        query: Optional[Union[str, int]] = None,
        dataset: Optional[str] = None,
        level: str = "auto",
        method: str = "vector_v2",
        metric: str = "cosine",
        # The two size knobs shared by EVERY candidate source mode:
        # ``candidate_cap`` bounds how many screen candidates enter the
        # morphology comparison (also the NBLAST prefilter of cache-direct
        # searches) and ``visualize_top_n`` bounds rendering only. Results
        # themselves are never truncated: every compared candidate is
        # returned and written.
        candidate_cap: int = 500,
        candidate_source: str = "auto",
        visualize_top_n: int = 0,
        visualize_by: str = "type",
        # Connection-cache screen thresholds ('profile' / 'combined' modes).
        min_weight: int = 3,
        min_shared_partners: int = 2,
        roi_filter: Optional[List[str]] = None,
        # V2 (method="vector_v2") knob: per-block score weights over the
        # fixed schema blocks. ROI evidence is a candidate-selection signal
        # only and deliberately does not participate in scoring.
        v2_block_weights: Optional[Dict[str, float]] = None,
        # Two-pass type reevaluation: after the first scoring pass, the
        # remaining members of the top `expand_top_types` candidate types
        # join the pool (up to `expand_per_type` screen-ranked members per
        # type) and the type aggregation re-runs on the fuller coverage.
        # 0 disables the expansion pass.
        expand_top_types: int = DEFAULT_EXPAND_TOP_TYPES,
        expand_per_type: int = DEFAULT_EXPAND_PER_TYPE,        # Coverage-aware type aggregation (V2): the type statistic is
        # scaled by the continuous factor sqrt(type_coverage) — gentle,
        # monotone, threshold-free. type_agg picks the statistic: "mean"
        # (typical member) or "max" (best member — "does the type contain
        # a similar neuron").
        type_agg: str = "mean",
        visualization_settings: Optional[Dict[str, object]] = None,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 8,
        use_cache: bool = True,
        # Retained as a compatibility keyword. Raw skeletons are now always
        # persisted as compressed SWC in the shared dataset skeleton cache.
        cache_fetched_skeletons: bool = True,
        # FAFB mesh-based extrusion detection is enforced at render time
        # (top-N results visualization, Skeleton tab). Similarity runs load
        # skeletons unchecked by default because the detector dominates the
        # FAFB fetch cost while extrusions are a small fraction of each tree.
        check_extrusions: bool = True,
        project_root: Optional[str] = None,
    ):
        self.query = query
        self.dataset = dataset
        self.level = str(level).lower()
        self.method = str(method).lower()
        self.metric = str(metric).lower()
        # V2 routes through the same search pipelines but swaps the cache
        # schema (SkeletonVectorCacheV2) and the scorer (block-weighted,
        # whitened); everything else (screening, pooling, output) is shared.
        self._is_v2 = self.method == "vector_v2"
        self.candidate_cap = int(candidate_cap)
        self.candidate_source = str(candidate_source).lower()
        # Lazily-built ROI profile store for 'roi'/'combined' screens.
        self._roi_store: Optional[RoiProfileStore] = None
        # Lazy bodyId->somaSide map (ipsi/contra NBLAST aggregation).
        self._soma_sides: Optional[Dict[int, str]] = None
        self.min_weight = int(min_weight)
        self.min_shared_partners = int(min_shared_partners)
        self.roi_filter = list(roi_filter) if roi_filter else None
        self.visualize_top_n = int(visualize_top_n)
        self.visualize_by = str(visualize_by).lower()
        self.visualization_settings = dict(visualization_settings or {})
        self.verbose = verbose
        self.n_workers = max(1, int(n_workers))
        self.use_cache = use_cache
        # Keep accepting the old keyword, but raw persistence is unconditional
        # so visualization, Find Similar, and Settings pulls share one source.
        self.cache_fetched_skeletons = True
        self.check_extrusions = bool(check_extrusions)
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent

        if self.level not in ("auto", "bodyid", "type"):
            raise ValueError(f"Invalid level: {self.level} (auto|bodyid|type)")
        if self.method not in ("vector_v2", "nblast"):
            raise ValueError(
                f"Invalid method: {self.method} (vector_v2|nblast)")
        weights = dict(DEFAULT_V2_BLOCK_WEIGHTS)
        for name, value in (v2_block_weights or {}).items():
            if name in weights:
                weights[name] = float(value)
        self._v2_weights = weights
        self.expand_top_types = int(expand_top_types)
        self.expand_per_type = int(expand_per_type)
        self.type_agg = str(type_agg).lower()
        if self.expand_top_types < 0 or self.expand_per_type < 0:
            raise ValueError(
                "expand_top_types/expand_per_type must be >= 0")
        if self.type_agg not in ("mean", "max"):
            raise ValueError(
                f"Invalid type_agg: {self.type_agg} (mean|max)")
        # Run-scoped V2 state: extra (ROI-expansion) blocks aligned to the
        # scoring pool, set by ``_prepare_v2_scoring``; the V1 counterpart
        # cache that online fetches warm with V1 vectors.
        # V2 run state: raw Hellinger histograms for the spatial-overlap
        # term — per query member, the pooled query centroid, and the pool.
        self._v2_spatial_overlap: Optional[Dict[str, np.ndarray]] = None
        # NBLAST runs: whether NBLAST scores actually replaced the vector
        # prefilter (False = dotprops unavailable, vector scores were kept).
        self.nblast_applied: Optional[bool] = None
        if self.metric not in ("cosine", "pearson"):
            raise ValueError(f"Invalid metric: {self.metric} (cosine|pearson)")
        if self.candidate_source not in ("auto", "roi", "combined", "profile",
                                         "cache"):
            raise ValueError(
                f"Invalid candidate_source: {self.candidate_source} "
                "(auto|roi|combined|profile|cache)"
            )
        if self.candidate_cap < 1:
            raise ValueError(
                f"Invalid candidate_cap: {self.candidate_cap} (>= 1)"
            )
        if self.min_shared_partners < 1:
            raise ValueError(
                f"Invalid min_shared_partners: {self.min_shared_partners} (>= 1)"
            )
        if self.visualize_by not in ("type", "bodyid"):
            raise ValueError(
                f"Invalid visualize_by: {self.visualize_by} (type|bodyId)"
            )

        if output_dir is None:
            self.output_dir = str(self.project_root / "outputs" / "similar")
        else:
            self.output_dir = output_dir
        self.saveas = saveas or None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line is a control event (determinate bar + step label in the
        results panel), not log output, and is only emitted when verbose.
        """
        if self.verbose:
            print(f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip(),
                  flush=True)

    # ------------------------------------------------------------ V2 scoring
    def _similarity_matrix(self, query: np.ndarray, matrix: np.ndarray,
                           pool_subset=None, query_index: Optional[int] = None) -> np.ndarray:
        """Whitened block-weighted V2 score. ``pool_subset`` (bool mask or
        index array) selects which pool rows the run-composed spatial
        mass-overlap histograms must be sliced to."""
        total, _ = self._similarity_matrix_with_blocks(
            query, matrix, pool_subset=pool_subset, query_index=query_index)
        return total

    def _similarity_matrix_with_blocks(self, query: np.ndarray,
                                       matrix: np.ndarray, pool_subset=None,
                                       query_index: Optional[int] = None):
        """As ``_similarity_matrix`` but also returns per-block scores."""
        total, per_block = v2_similarity_matrix(
            query, matrix, self._v2_weights,
            spatial_overlap=self._slice_spatial_overlap(pool_subset),
            query_index=query_index)
        return total, per_block

    def _pairwise_similarity(self, matrix: np.ndarray) -> np.ndarray:
        return v2_pairwise_matrix(matrix, self._v2_weights)

    def _vector_dim_for(self, cache_data: Optional[dict]) -> int:
        """Vector width of this run's cache (124 for V1, 256 for V2)."""
        if cache_data is not None and getattr(cache_data.get("X"), "ndim", 0) == 2:
            return int(cache_data["X"].shape[1])
        return VECTOR_V2_DIM if self._is_v2 else VECTOR_DIM

    def _maybe_roi_store(self) -> Optional[RoiProfileStore]:
        """The ROI store for the V2 expansion block, when it is cheaply
        available (NeuPrint datasets with an ROI table). Never raises."""
        if self._is_flywire():
            return None
        if self._roi_store is not None:
            return self._roi_store
        try:
            self._roi_store = RoiProfileStore(
                self.dataset, project_root=str(self.project_root),
                verbose=False, log=self._log,
            ).ensure()
            return self._roi_store
        except Exception:
            return None

    def _prepare_v2_scoring(self, cache_data: Optional[dict],
                            X_q: np.ndarray, X_c: np.ndarray,
                            mask_q: np.ndarray, mask_c: np.ndarray,
                            query_ids: List, pool_ids: List) -> None:
        """Whiten the standardized snapshot and build the spatial overlap.

        Runs once per search, after standardization and before any scoring:
        applies the population ZCA whitener to the query/candidate rows (and
        the cache snapshot, so intra-type fallbacks live in the same space),
        then aligns the Hellinger pre/post mass-overlap term to the pool.
        ROI evidence deliberately does NOT participate in scoring — it is a
        candidate-selection signal only (see ``_roi_candidates``).
        """
        W = (cache_data or {}).get("whiten")
        if W is None or not getattr(W, "size", 0):
            W = np.eye(X_q.shape[1] if X_q.ndim == 2 else VECTOR_V2_DIM)
        if cache_data is not None:
            cache_data["X"] = apply_whitening(W, cache_data["X"])
        if mask_q.any():
            X_q[mask_q] = apply_whitening(W, X_q[mask_q])
        if mask_c.any():
            X_c[mask_c] = apply_whitening(W, X_c[mask_c])

        self._v2_spatial_overlap = self._build_spatial_overlap(
            cache_data, query_ids, pool_ids)

    # ------------------------------------------------------ candidate source
    def _is_flywire(self) -> bool:
        """Compatibility predicate for local-release cache behavior.

        The historical method name is retained for callers, but its value is
        now supplied by the shared FAFB/BANC local-release predicate rather
        than an inline source-family string test.
        """
        return is_local_connectome_dataset(self.dataset)

    def _body_id(self, value):
        """Return the comparison-layer body-ID representation."""

        return _canonical_dataset_body_id(self.dataset, value)

    def _body_ids(self, values):
        """Canonicalize a body-ID iterable for this dataset."""

        return [self._body_id(value) for value in values]

    def _resolved_candidate_source(self) -> str:
        if self.candidate_source != "auto":
            return self.candidate_source
        # NeuPrint datasets screen candidates from the per-neuron primary-ROI
        # distributions: every neuron is reachable, while the connection-cache
        # screen can never find neurons without shared partners. FAFB and
        # standalone BANC have no ROI table; FAFB carries a merged connection cache,
        # so it runs the connectivity screen FIRST and only then fetches and
        # vectorizes the skeletons of the selected candidates — the vector
        # cache is a store, not the definition of the search population.
        # Datasets without a connection cache fall back to searching the
        # local vector population directly.
        if self._is_flywire():
            if is_fafb_dataset(self.dataset) and self._connection_cache_available():
                return "profile"
            return "cache"
        return "roi" if self._roi_screening_ready(allow_backfill=True) \
            else "profile"

    def _connection_cache_available(self) -> bool:
        """Cheap probe: does this dataset have a merged connection cache?"""
        return (Path(self.project_root) / "cache" / _dataset_folder(self.dataset)
                / "connections.parquet").exists()

    def _roi_screening_ready(self, allow_backfill: bool) -> bool:
        """Cheap local probe for the ROI screen's prerequisites.

        Checks the ROI-count table (parquet or CSV) and the metadata sidecar;
        when the sidecar is missing, ``allow_backfill`` fetches and saves it
        once through the dataset-preparation code (network). The partition
        validation of the ROI list happens later, at store build.
        """
        try:
            if not roi_count_table_path(
                    self.dataset, str(self.project_root)).exists():
                return False
            if load_primary_rois(self.dataset, str(self.project_root)):
                return True
            if allow_backfill:
                meta = backfill_dataset_metadata(
                    self.dataset, str(self.project_root), log=self._log)
                return bool(
                    meta and (meta.get("roi_coverage") or {}).get("roi_list")
                )
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------ query
    def _resolve_level(self, query_df: pd.DataFrame) -> str:
        """Resolve the result level for ``level='auto'``.

        A bodyId query (all-numeric input) yields bodyId-to-bodyId rows; a
        type query (type/pattern input resolving to one common type) yields
        type-to-type rows; mixed or multi-type lists fall back to bodyId."""
        if self.level != "auto":
            return self.level
        q = self.query
        if isinstance(q, (list, tuple)):
            all_digits = all(str(x).strip().isdigit() for x in q)
        else:
            all_digits = str(q).strip().isdigit()
        if all_digits:
            return "bodyid"
        types = {str(t) for t in query_df["type"].tolist() if str(t)}
        return "type" if len(types) == 1 else "bodyid"

    def _resolve_query(self) -> pd.DataFrame:
        if self.query is None:
            raise ValueError("No query neuron specified.")
        # getNeurons returns (neuron_df, roi_count_df, auto_name, criteria).
        # Some local loaders (FlyWire) only accept list queries, so retry with
        # a list-wrapped query when the scalar form fails or finds nothing.
        attempts = [self.query]
        if not isinstance(self.query, (list, tuple)):
            attempts.append([self.query])
        df = None
        for q in attempts:
            try:
                candidate = getNeurons(q, self.dataset, verbose=False, search_columns="auto")[0]
            except Exception:
                candidate = None
            if candidate is not None and len(candidate):
                df = candidate
                break
        if df is None or len(df) == 0:
            raise ValueError(f"Query '{self.query}' not found in {self.dataset}")
        out = df[["bodyId"]].copy()
        out["bodyId"] = self._body_ids(out["bodyId"].tolist())
        if not self._is_flywire():
            out["bodyId"] = out["bodyId"].astype(np.int64)
        out["type"] = df.get("type", pd.Series([""] * len(df))).fillna("").astype(str)
        out["instance"] = df.get("instance", pd.Series([""] * len(df))).fillna("").astype(str)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ run
    def find_similar(self) -> pd.DataFrame:
        """Run the similarity search and save results. Returns the results df."""
        if self.dataset is None:
            raise ValueError("No dataset specified.")
        source = self._resolved_candidate_source()
        # The step scheme depends on the pipeline: candidate-screen-first
        # (NeuPrint: roi / combined / profile) reports 6 steps,
        # vector-cache-direct (FlyWire) 4.
        total_steps = (PROFILE_FIRST_TOTAL_STEPS
                       if source in CANDIDATE_SCREEN_SOURCES
                       else CACHE_DIRECT_TOTAL_STEPS)
        self.resolved_candidate_source = source
        self._log(f"Morphological similarity: query={self.query} dataset={self.dataset} "
                  f"method={self.method} level={self.level} metric={self.metric} "
                  f"candidate_source={source}")

        # BANC similarity is deliberately deferred: find-similar on the
        # public L2/full skeletons needs its own vector-quality validation
        # before it can run.  Raise a clear deferred error instead of
        # silently reusing the FAFB pipeline.
        if is_banc_dataset(self.dataset):
            raise FlyWireSkeletonAccessError(
                "BANC morphological similarity is deferred: public "
                "L2/full skeletons still need vector-quality validation. "
                "3D skeleton visualization for BANC is available."
            )

        # FAFB may use its local skeleton bundle or CAVE, but must not
        # silently fall into a failed remote fetch when neither source is
        # configured.
        require_flywire_skeleton_access(
            self.dataset,
            project_root=self.project_root,
            log=self._log,
        )

        self._progress(1, total_steps, "Resolving query neuron")
        query_df = self._resolve_query()
        if self.level == "auto":
            self.level = self._resolve_level(query_df)
            self._log(f"Level auto-resolved to: {self.level} "
                      f"({'type-to-type' if self.level == 'type' else 'bodyId-to-bodyId'})")
        # The V2 skeleton-vector cache is the single search population for
        # both methods: vector_v2 scores it directly, and NBLAST needs
        # skeletons (the FlyWire local mesh cache cannot serve dotprops,
        # and the V1 vector path is retired). Its raw-skeleton store is
        # backed by the raw skeleton store / healed zip.
        cache = find_similar_dataset_cache_v2(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=self.verbose,
        )

        if source in CANDIDATE_SCREEN_SOURCES:
            bodyid_df, type_df = self._profile_first_search(query_df, cache,
                                                            source)
            if bodyid_df.empty and type_df.empty:
                # Connection-profile discovery can return nothing for neurons
                # with sparse connectivity; fall back to the vector cache when
                # one exists, otherwise surface a clear error.
                data = cache.load()
                if data is not None and len(data["bodyIds"]):
                    self._log(
                        "Profile-first found no candidates; falling back to "
                        "the vector cache."
                    )
                    # The fallback switches the whole run to the 4-step
                    # cache-direct protocol; later events (including the
                    # final one) must keep that total so the bar never
                    # regresses or jumps between step schemes.
                    total_steps = CACHE_DIRECT_TOTAL_STEPS
                    self._progress(2, total_steps,
                                   "Loading vector cache (fallback)")
                    self._progress(3, CACHE_DIRECT_TOTAL_STEPS,
                                   self._scoring_step_label())
                    if self._is_v2:
                        bodyid_df, type_df = self._vector_search(query_df, data)
                    else:
                        bodyid_df, type_df = self._nblast_search(query_df, data)
                else:
                    raise ValueError(
                        "Connection-profile search found candidates for "
                        f"{self.query} in {self.dataset} but none could be "
                        "scored: no cached skeletons were usable, transient "
                        "fetches returned nothing, and no vector cache exists "
                        "to fall back to. Check network/token access and the "
                        "skeleton cache, or build the vector cache first."
                    )
        else:
            # Cache-direct search (FlyWire bulk caches and explicit choice).
            # Build from raw local sources first, then fetch any missing query
            # skeletons online. Candidate skeletons remain bounded by the
            # available raw population; this avoids silently using the shared
            # visualization simp90 cache.
            self._progress(2, CACHE_DIRECT_TOTAL_STEPS, "Loading vector cache")
            cache.ensure(fetch_missing=0)
            data = cache.load()
            query_ids = self._body_ids(query_df["bodyId"].tolist())
            cached_ids = (set(self._body_ids(data["bodyIds"]))
                          if data is not None else set())
            missing_query = [bid for bid in query_ids if bid not in cached_ids]
            if missing_query:
                self._log(
                    f"Vector cache miss: fetching {len(missing_query)} raw "
                    "query skeleton(s) online."
                )
                if is_fafb_dataset(self.dataset):
                    # FAFB: the batch fetcher returns MESHES; resolve the
                    # query through the local loader (repair caches / zip).
                    fetched_query = _resolve_fafb_skeleton_trees(
                        self.dataset, missing_query,
                        project_root=str(self.project_root),
                        log=self._log)
                else:
                    fetched_query = fetch_skeletons_on_demand_batch(
                        self.dataset,
                        missing_query,
                        project_root=str(self.project_root),
                        persist=self.cache_fetched_skeletons,
                        level=VECTOR_BASIS_RAW,
                        max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                                        max(1, int(self.n_workers))),
                        raw_cache=cache,
                        vector_cache=None,
                    )
                # Keep the vector-mode contract even for integrations that
                # override the batch fetch seam and return neurons without
                # appending their own vector rows.
                rows = []
                for bid, neuron in (fetched_query or {}).items():
                    try:
                        if _neuron_rep(neuron) != "skeleton":
                            continue
                        _, vec = cache._vectorize_neuron(neuron)
                        rows.append((self._body_id(bid), vec, "skeleton"))
                    except Exception:
                        continue
                if rows:
                    cache.append_vectors(rows, vector_basis=cache._default_basis())
                data = cache.load()
            if data is None or len(data["bodyIds"]) == 0:
                raise ValueError(
                    f"No raw vectorized neurons for {self.dataset}. Fetch or "
                    "build the shared raw vector cache first."
                )
            self._progress(3, CACHE_DIRECT_TOTAL_STEPS, self._scoring_step_label())
            if self._is_v2:
                bodyid_df, type_df = self._vector_search(query_df, data)
            else:
                bodyid_df, type_df = self._nblast_search(query_df, data)

        # The returned/primary frame follows the level (type-to-type for type
        # queries, bodyId-to-bodyId otherwise); both files are always saved.
        results = type_df if self.level == "type" else bodyid_df
        if results.empty:
            self._log("No similar neurons found.")
            # Keep the query-only reference visualization useful even when a
            # search has no ranked matches.  The query and empty result tables
            # are still saved in the normal per-run folder.
            self._save_results(results, bodyid_df, type_df, query_df)
            self._visualize_top_results(results, query_df=query_df)
        else:
            self._save_results(results, bodyid_df, type_df, query_df)
            self._visualize_top_results(results, query_df=query_df)
        final_label = "Search finished (no similar neurons found)"
        if not results.empty:
            final_label = ("Saving results & visualization"
                           if self.visualize_top_n else "Saving results")
        self._progress(total_steps, total_steps, final_label)
        return results

    def _scoring_step_label(self) -> str:
        return "Scoring similarity (vector: shape + spatial)"
        if self.method == "nblast":
            return "Building dotprops & NBLAST scoring"
        return "Scoring similarity (vector: shape + spatial)"

    # ---------------------------------------------------- profile-first
    def _connection_cache_candidates(self, query_df: pd.DataFrame,
                                     min_weight: Optional[int] = None,
                                     min_shared_partners: Optional[int] = None,
                                     roi_filter: Optional[List[str]] = None,
                                     top_k: Optional[int] = None) -> pd.DataFrame:
        """Rank candidate neurons directly from the connection cache.

        Reads ``cache/{dataset}/connections.parquet``
        (bodyId_pre, bodyId_post, weight, roi) with polars and finds neurons
        sharing upstream/downstream partners with the query:

        1. the query's partner rows (``weight >= min_weight``; optionally
           restricted to ``roi_filter`` ROIs when the dataset has ROI data —
           ``"*"`` = non-empty ROIs only),
        2. shared-partner adjacency: candidates = neurons sharing at least
           ``min_shared_partners`` upstream/downstream partner neurons with
           the query,
        3. ranked by shared-partner count (descending), target types joined
           via the neuron type map.

        Lightweight (no profile building); same adjacency semantics as the
        homolog finding, direct from the cache. Returns a DataFrame with
        target_bodyId, shared_count, profile_similarity (normalized 0-1 by
        the max shared count) and target_type.
        """
        import polars as pl

        min_weight = self.min_weight if min_weight is None else int(min_weight)
        min_shared_partners \
            = self.min_shared_partners if min_shared_partners is None else int(min_shared_partners)
        roi_filter = self.roi_filter if roi_filter is None else roi_filter

        conn_path = (Path(self.project_root) / "cache"
                     / _dataset_folder(self.dataset) / "connections.parquet")
        if not conn_path.exists():
            self._log("Connection cache missing; no candidates discoverable.")
            return pd.DataFrame()

        # Lazy scan with projection/predicate pushdown: only the needed
        # columns are read, min_weight is applied inside the scan, and the
        # shared-partner counting uses hash semi/anti JOINS — is_in filters
        # over a materialized frame degrade badly on FlyWire-sized caches
        # (string bodyIds, ~20M rows).
        needed = ["bodyId_pre", "bodyId_post", "weight"]
        if roi_filter:
            needed.append("roi")
        conn = (pl.scan_parquet(conn_path)
                .select(needed)
                .filter(pl.col("weight") >= min_weight))
        if roi_filter:
            if roi_filter == ["*"] or roi_filter == "*":
                conn = conn.filter(pl.col("roi").is_not_null()
                                   & (pl.col("roi") != ""))
            else:
                conn = conn.filter(pl.col("roi").is_in(
                    pl.Series("roi_filter", roi_filter).implode()))

        query_ids = self._body_ids(query_df["bodyId"].tolist())
        id_dtype = pl.Utf8 if self._is_flywire() else pl.Int64
        try:
            # The laziness defers ALL file access to collect(); a corrupt or
            # truncated cache must still degrade to an empty candidate list
            # (readers contract), so the whole computation is guarded here.
            conn = conn.with_columns([
                pl.col("bodyId_pre").cast(id_dtype, strict=False),
                pl.col("bodyId_post").cast(id_dtype, strict=False),
            ]).drop_nulls(["bodyId_pre", "bodyId_post"])

            q_df = pl.DataFrame({"q": query_ids}).with_columns(
                pl.col("q").cast(id_dtype)).lazy()
            up = conn.join(q_df, left_on="bodyId_post", right_on="q",
                           how="semi")
            down = conn.join(q_df, left_on="bodyId_pre", right_on="q",
                             how="semi")

            def _shared(partner_col: str, candidate_col: str,
                        partners: "pl.LazyFrame") -> "pl.DataFrame":
                p_df = partners.select(pl.col(partner_col).unique())
                return (conn
                        .join(p_df, left_on=partner_col,
                              right_on=partner_col, how="semi")
                        .join(q_df, left_on=candidate_col, right_on="q",
                              how="anti")
                        .group_by([candidate_col, partner_col])
                        .agg(pl.len().alias("_w"))
                        .group_by(candidate_col)
                        .agg(pl.len().alias("n_shared"))
                        .collect())

            up_shared = _shared("bodyId_pre", "bodyId_post", up)
            down_shared = _shared("bodyId_post", "bodyId_pre", down)

            counts = None
            for part in (up_shared, down_shared):
                if part.height == 0:
                    continue
                part = part.rename({part.columns[0]: "candidate"})
                counts = part if counts is None else counts.vstack(part)
            if counts is None:
                return pd.DataFrame()
            counts = (counts
                      .group_by("candidate")
                      .agg(pl.col("n_shared").sum())
                      .filter(pl.col("n_shared") >= min_shared_partners)
                      .sort("n_shared", descending=True))
            if top_k:
                counts = counts.head(top_k)
        except Exception as exc:
            self._log(f"Connection cache unreadable: {exc}")
            return pd.DataFrame()

        out = pd.DataFrame({
            "target_bodyId": counts["candidate"].to_list(),
            "shared_count": counts["n_shared"].to_list(),
        })
        if out.empty:
            return out
        max_shared = float(out["shared_count"].max()) or 1.0
        out["profile_similarity"] = out["shared_count"] / max_shared
        type_map, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        out["target_type"] = out["target_bodyId"].map(
            lambda b: type_map.get(self._body_id(b), "")
        )
        return out

    def _roi_candidates(self, query_df: pd.DataFrame,
                         top_k: Optional[int] = None) -> pd.DataFrame:
        """Rank candidate neurons by primary-ROI distribution similarity.

        Uses ``RoiProfileStore`` (built once per dataset from the ROI-count
        CSV, cached under ``cache/{dataset}/morphology/roi_profiles.npz``):
        mirrored cosine of the input/output synapse distributions over the
        dataset's primary ROIs. Raises ``RoiScreeningUnavailable`` when the
        dataset lacks the ROI table or a usable primary-ROI list. Returns a
        DataFrame [target_bodyId, roi_similarity, target_type] sorted by
        similarity descending; ``top_k`` bounds it to the best K rows.
        """
        if self._roi_store is None:
            self._roi_store = RoiProfileStore(
                self.dataset, project_root=str(self.project_root),
                verbose=self.verbose, log=self._log,
            ).ensure()
        scores = self._roi_store.screen(
            [int(b) for b in query_df["bodyId"].tolist()], top_k=top_k
        )
        if scores.empty:
            return pd.DataFrame()
        type_map, _ = _load_neuron_type_map(self.dataset, str(self.project_root))
        out = pd.DataFrame({
            "target_bodyId": scores["bodyId"].astype(np.int64),
            "roi_similarity": scores["roi_similarity"].astype(float),
        })
        # Vectorized type lookup (the previous per-row lambda mapped the
        # full dataset population in Python).
        out["target_type"] = (
            pd.Series(out["target_bodyId"].values).map(type_map)
            .fillna("").astype(str)
        )
        return out

    def _discover_candidates(self, query_df: pd.DataFrame,
                             source: str) -> Tuple[pd.DataFrame, str]:
        """Rank candidate neurons for the scoring-pool selection.

        Returns the unified candidate frame and the source label recorded in
        the results. The frame carries ``target_bodyId`` / ``target_type`` /
        ``profile_similarity`` (connectivity evidence, 0-1) /
        ``roi_similarity`` (ROI-screen evidence, 0-1) and ``_score``, the
        per-candidate ranking value: shared-partner count for 'profile', ROI
        cosine for 'roi', and the mean of the max-normalized scores of both
        screens for 'combined'. The frame is ALWAYS sorted by ``_score``
        descending — the pool is simply its top ``candidate_cap`` rows.
        """
        if source == "profile":
            candidates = self._connection_cache_candidates(query_df)
            if not candidates.empty:
                candidates = candidates.copy()
                candidates["_score"] = candidates["shared_count"].astype(float)
                candidates["roi_similarity"] = np.nan
            return _sorted_candidates(candidates), "profile"

        roi = self._roi_candidates(
            query_df,
            # 'roi' mode pools exactly the top candidate_cap rows, so the
            # screen can return the partial ranking directly; 'combined'
            # needs the full list to merge with the connectivity screen.
            top_k=self.candidate_cap if source == "roi" else None,
        )   # raises when unavailable
        if source == "roi":
            roi = roi.copy()
            roi["_score"] = roi["roi_similarity"]
            roi["profile_similarity"] = np.nan
            return _sorted_candidates(roi), "roi"

        # combined: outer-merge both screens; a missing connection cache
        # degrades to the ROI screen (the reverse is not possible — the ROI
        # table is a hard requirement here).
        conn = self._connection_cache_candidates(query_df)
        if conn.empty:
            self._log("Connection cache missing; 'combined' candidate "
                      "discovery uses the ROI screen only.")
            roi = roi.copy()
            roi["_score"] = roi["roi_similarity"]
            roi["profile_similarity"] = np.nan
            return _sorted_candidates(roi), "combined"
        conn = conn.copy()
        conn["_score"] = conn["shared_count"] / max(
            1.0, float(conn["shared_count"].max()))
        roi = roi.copy()
        roi["_score"] = roi["roi_similarity"] / max(
            1e-9, float(roi["roi_similarity"].max()))
        merged = conn.merge(
            roi, on="target_bodyId", how="outer",
            suffixes=("", "_roi"),
        )
        # Both frames type their candidates from the same map; keep whichever
        # side has a non-empty label, then average the per-screen scores
        # (pandas mean skips the NaN of a screen that missed the neuron).
        merged["target_type"] = (
            merged["target_type"].replace("", np.nan)
            .fillna(merged["target_type_roi"].replace("", np.nan))
            .fillna("")
        )
        merged["_score"] = merged[["_score", "_score_roi"]].mean(axis=1)
        merged = merged.drop(columns=["target_type_roi", "_score_roi"])
        return _sorted_candidates(merged), "combined"

    def _profile_first_search(self, query_df: pd.DataFrame,
                              cache: SkeletonVectorCache,
                              source: str = "profile") -> pd.DataFrame:
        """Candidate-screen-first, then morphology on the capped pool.

        Candidate discovery ranks neurons by the selected screen
        (``_discover_candidates``: connection-cache shared partners, primary-
        ROI distribution cosine, or both) and returns them sorted; the
        scoring pool is simply the top ``candidate_cap`` candidates (query
        members are already excluded by the screens). Fetched skeletons are
        transient (used for the current comparison only), while their raw
        vectors are persisted for reuse; cached skeletons are reused. Every scored
        vector is standardized with ONE consistent set of statistics (cache
        meta, sample-based population stats, or pool stats as a last
        resort), and the whole comparison runs at ONE representation level
        (skeleton vs mesh — rows of any other representation are
        unscorable). ALL compared candidates are returned and written.
        """
        step2_label = {
            "profile": "connection cache",
            "roi": "ROI distribution screen",
            "combined": "ROI + connectivity screens",
        }.get(source, source)
        self._log(f"Step 2/6 — Discovering candidates: running {step2_label} "
                  "candidate discovery...")
        self._progress(2, PROFILE_FIRST_TOTAL_STEPS,
                       f"Discovering candidates ({step2_label})")
        try:
            candidates, source = self._discover_candidates(query_df, source)
        except RoiScreeningUnavailable as exc:
            # Auto-resolved ROI runs fall back to the connection-cache screen;
            # explicit roi/combined selections surface the preparation hint.
            if self.candidate_source != "auto":
                raise
            self._log(f"{exc}")
            self._log("Falling back to the connection-cache candidate screen.")
            candidates, source = self._discover_candidates(query_df, "profile")
        self.resolved_candidate_source = source
        if candidates.empty:
            self._log(f"Candidate discovery ({source}) returned no candidates.")
            return pd.DataFrame(), pd.DataFrame()

        # The scoring pool: the sorted candidate list truncated to the cap.
        # Untyped candidates stay in — they are comparable neurons, their
        # rows just carry an empty target_type.
        pool_ids = self._body_ids(candidates["target_bodyId"].tolist())
        pool_ids = pool_ids[: self.candidate_cap]
        n_screened = len(candidates)
        self._log(f"Step 3/6 — Selecting the scoring pool: top {len(pool_ids)} "
                  f"of {n_screened} candidates (cap {self.candidate_cap})")
        self._progress(3, PROFILE_FIRST_TOTAL_STEPS,
                       f"Selecting top {len(pool_ids)} candidates for scoring")

        # The exported profile_similarity column should carry connectivity
        # evidence whatever screen discovered the pool: when the ROI screen
        # ran alone, score the candidates' shared partners post-hoc (the
        # 'profile'/'combined' screens already populate the column).
        conn_map: Dict[int, float] = {}
        if source == "roi":
            try:
                conn = self._connection_cache_candidates(query_df)
            except Exception as exc:
                self._log(f"Profile-evidence backfill skipped: {exc}")
                conn = pd.DataFrame()
            if not conn.empty:
                conn_map = {
                    self._body_id(b): float(v)
                    for b, v in zip(conn["target_bodyId"],
                                    conn["profile_similarity"])
                    if pd.notna(v)
                }
                candidates = candidates.copy()
                candidates["profile_similarity"] = [
                    conn_map.get(self._body_id(b), np.nan)
                    for b in candidates["target_bodyId"]
                ]

        prof_by_id = {
            self._body_id(b): float(v)
            for b, v in zip(candidates["target_bodyId"], candidates["profile_similarity"])
            if np.isfinite(v)
        }
        # Expansion members are not part of the screen's candidate list but
        # share the same connectivity evidence: keep the full map for them.
        prof_by_id.update(conn_map)
        roi_by_id = {
            self._body_id(b): float(v)
            for b, v in zip(candidates["target_bodyId"], candidates["roi_similarity"])
            if np.isfinite(v)
        }

        self._log("Step 4/6 — Loading & vectorizing skeletons")
        self._progress(4, PROFILE_FIRST_TOTAL_STEPS, "Loading & vectorizing skeletons")
        query_ids = self._body_ids(query_df["bodyId"].tolist())

        cache_data = cache.load()
        cache_ids = (set(self._body_ids(cache_data["bodyIds"]))
                     if cache_data is not None else set())
        cache_rep = cache_data.get("dataset_rep", "") if cache_data is not None else ""
        cache_basis = (((cache_data.get("meta") or {}).get("vector_basis")
                        or cache._default_basis()) if cache_data is not None
                       else cache._default_basis())

        # Determine every online miss before fetching anything.  Query and
        # candidate IDs are intentionally combined: a query can also occur in
        # the candidate pool, and the optimized path must not issue two
        # separate online phases for the same comparison.
        all_load_ids = []
        seen_load_ids = set()
        for bid in query_ids + pool_ids:
            bid = self._body_id(bid)
            if bid not in seen_load_ids:
                seen_load_ids.add(bid)
                all_load_ids.append(bid)
        missing_ids = []
        for bid in all_load_ids:
            if bid in cache_ids:
                continue  # the vector cache is sufficient for screening
            pkl = cache.find_skeleton_file(bid)
            if pkl is not None:
                continue
            missing_ids.append(bid)

        def _report_fetch(done, total, message):
            self._progress(4, PROFILE_FIRST_TOTAL_STEPS, message)
            if total and (done == 0 or done >= total
                          or done % max(1, total // 10) == 0):
                self._log(f"Step 4/6 — {message}")

        fetch_cache = cache
        fetched_all: Dict[int, object] = {}
        if missing_ids:
            if self._is_v2 and is_fafb_dataset(self.dataset):
                # FAFB: the healed zip is LOCAL — fetch through the shared
                # FlyWire loader (repair caches -> raw cache -> zip ->
                # one-time extrusion check with CAVE replacement / local
                # fix) instead of the generic batch fetcher's per-neuron
                # CAVE mesh path, which takes minutes per skeleton on this
                # dataset and returns meshes a skeleton cache cannot store.
                # The loader does not mirror trees into ``raw_skeletons``:
                # the zip plus the repair caches are the durable sources,
                # and the vector rows below are appended from the in-memory
                # neurons, so no skeleton file needs to be written here.
                loaded = self._load_fafb_skeletons(missing_ids)
                fetched = {int(b): n for b, n in (loaded or {}).items()
                           if n is not None}
                fetched_all = {self._body_id(b): n for b, n in fetched.items()}
                self._log(f"FAFB: fetched {len(fetched_all)} of "
                          f"{len(missing_ids)} skeletons from the local "
                          "healed zip.")
            else:
                fetched_all = fetch_skeletons_on_demand_batch(
                    self.dataset,
                    missing_ids,
                    project_root=str(self.project_root),
                    persist=self.cache_fetched_skeletons,
                    level=VECTOR_BASIS_RAW,
                    max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                                    max(1, int(self.n_workers))),
                    progress_callback=_report_fetch,
                    # V2 (non-FAFB): the fetcher warms the V1 vector cache
                    # and persists raw SWCs into the SHARED skeleton dir;
                    # the V2 schema is always computed locally afterwards
                    # (a 124-dim fetcher row must never land in the V2
                    # parquet).
                    raw_cache=fetch_cache,
                    vector_cache=None,
                )

        query_neurons = {
            self._body_id(bid): fetched_all[self._body_id(bid)]
            for bid in query_ids if self._body_id(bid) in fetched_all
        }

        # The comparison's representation: the majority among the query
        # members (cache rows carry the cache's representation).
        from collections import Counter
        known_q = []
        for bid in query_ids:
            if self._body_id(bid) in cache_ids:
                known_q.append(cache_rep)
            elif self._body_id(bid) in query_neurons:
                known_q.append(_neuron_rep(query_neurons[self._body_id(bid)]))
        q_rep = Counter(r for r in known_q if r).most_common(1)[0][0] \
            if any(known_q) else ""
        if not q_rep:
            # No cached/fetched query member with a known representation
            # (e.g. all query rows computed from cache files): infer it from
            # the dataset's skeleton store.
            q_rep = _infer_dataset_rep(self.dataset, str(self.project_root))

        # The one combined fetch above supplies both sides of the comparison.
        # Filter only after q_rep is known so the representation guard remains
        # identical to the former query-then-pool implementation.
        pool_neurons = {
            self._body_id(bid): fetched_all[self._body_id(bid)]
            for bid in pool_ids if self._body_id(bid) in fetched_all
            and (not q_rep or _neuron_rep(fetched_all[self._body_id(bid)]) == q_rep)
        }
        self._log(f"Profile-first: {len(pool_ids)} pool neurons, "
                  f"{len(pool_neurons)} skeletons fetched (transient)")

        # Vectorize any in-memory fetches before the cache lookup.  The batch
        # fetcher normally appends these rows itself; this append is a
        # deduplicating compatibility path for singular-fetch overrides and
        # offline fixtures.  It also makes the subsequent snapshot complete.
        fetched_vectors: List[Tuple[Union[int, str], np.ndarray, str]] = []
        for bid, neuron in fetched_all.items():
            try:
                # The cache's schema hook: V1's vectorize_neuron, or V2 with
                # the population spatial bounds.
                _, vec = cache._vectorize_neuron(neuron)
                rep = _neuron_rep(neuron)
                if not q_rep or rep == q_rep:
                    fetched_vectors.append((self._body_id(bid), vec, rep))
            except Exception:
                continue
        if fetched_vectors:
            cache.append_vectors(fetched_vectors, vector_basis=cache_basis)

        # Resolve all local-file misses in one call, then reload the cache.
        # The reload is what makes the snapshot complete: a row this call
        # vectorized is persisted, and the snapshot below reads it back in the
        # cache's own space. ``space='raw'`` is asked for because the rows this
        # returns are only used as the transient fallback for an id that could
        # NOT be persisted, and the standardization block below supplies that
        # transform itself.
        pre_X, pre_mask, pre_reps = cache.vectors_for(
            all_load_ids, compute_missing=True, space='raw'
        )
        cache_data = cache.load()
        cache_ids = (set(self._body_ids(cache_data["bodyIds"]))
                     if cache_data is not None else set())
        cache_rep = (cache_data.get("dataset_rep", "")
                     if cache_data is not None else "")
        cache_basis = (((cache_data.get("meta") or {}).get("vector_basis")
                        or cache._default_basis()) if cache_data is not None
                       else cache._default_basis())
        vec_dim = self._vector_dim_for(cache_data)

        snapshot_rows = {}
        if cache_data is not None:
            snapshot_rows = {
                self._body_id(bid): i for i, bid in enumerate(cache_data["bodyIds"])
            }
        pre_rows = {self._body_id(bid): i for i, bid in enumerate(all_load_ids)}

        def _snapshot_vectors(ids):
            vectors = np.full((len(ids), vec_dim), np.nan)
            mask = np.zeros(len(ids), dtype=bool)
            reps = [""] * len(ids)
            for i, bid in enumerate(ids):
                bid = self._body_id(bid)
                row = snapshot_rows.get(bid)
                if row is not None and cache_data is not None:
                    candidate = cache_data["X"][row]
                    if np.isfinite(candidate).all():
                        vectors[i] = candidate
                        mask[i] = True
                        reps[i] = (cache_data["rep"][row]
                                   or cache_data.get("dataset_rep", ""))
                        continue
                # If persistence was unavailable, retain the raw vector
                # computed by this call as a last-resort transient row.  It
                # is intentionally marked as non-cache data below so the
                # standardization block applies the chosen transform.
                pre_row = pre_rows.get(bid)
                if pre_row is not None and pre_mask[pre_row]:
                    vectors[i] = pre_X[pre_row]
                    mask[i] = True
                    reps[i] = pre_reps[pre_row]
            return vectors, mask, reps

        X_q, mask_q, rep_q = _snapshot_vectors(query_ids)
        X_c, mask_c, rep_c = _snapshot_vectors(pool_ids)

        # ONE representation per comparison: skeletons and meshes (or two
        # simplification levels) produce different features in the shared
        # schema, so rows of any other representation than the query's are
        # unscorable and stay out of the masks.
        from collections import Counter
        known_q = [r for r in rep_q if r]
        q_rep = Counter(known_q).most_common(1)[0][0] if known_q else ""
        if q_rep:
            rep_q = np.array(rep_q)
            rep_c = np.array(rep_c)
            mask_q = mask_q & (rep_q == q_rep)
            mask_c = mask_c & (rep_c == q_rep)
        if not mask_q.any():
            raise ValueError("Could not vectorize the query neuron.")

        # Standardize EVERY scored vector with ONE consistent set of
        # statistics so cosine is scale-fair (raw morphometrics + shape are
        # on very different scales). Cache rows are already standardized
        # with the cache's own meta stats: they are left untouched when
        # those stats are used, and restored to raw (then re-standardized)
        # when the stats come from elsewhere. Freshly-computed rows always
        # receive the chosen transform. A small vector cache carries skewed,
        # unreliable stats (it is typically built from one query's transient
        # fetches), so its meta is ignored in favour of sample-based stats,
        # which extend the sample with a version sibling's skeletons.
        meta_mu = meta_sd = None
        if cache_data is not None:
            meta = cache_data.get("meta") or {}
            m = meta.get("mean")
            s = meta.get("std")
            if m is not None and s is not None:
                mm = np.asarray(m, dtype=float)
                ss = np.asarray(s, dtype=float)
                if mm.shape == (vec_dim,) and ss.shape == (vec_dim,):
                    meta_mu, meta_sd = mm, ss

        using_cache_stats = False
        if mask_q.any() or mask_c.any():
            if (meta_mu is not None
                    and len(cache_data["bodyIds"]) >= MIN_POPULATION_STATS_SKELETONS):
                mu, sd, using_cache_stats = meta_mu, meta_sd, True

            cache_q = (np.array([self._body_id(b) in cache_ids for b in query_ids])
                       & mask_q)
            cache_c = (np.array([self._body_id(b) in cache_ids for b in pool_ids])
                       & mask_c)
            if using_cache_stats:
                # Cache rows already use these stats; transform only the
                # freshly-computed rows (raw -> standardized).
                X_q[mask_q & ~cache_q] = (X_q[mask_q & ~cache_q] - mu) / sd
                X_c[mask_c & ~cache_c] = (X_c[mask_c & ~cache_c] - mu) / sd
            else:
                # Pool-computed statistics. Cache rows carry the cache's own
                # standardization: restore them to raw FIRST, then compute
                # mu/sd over the restored (all-raw) rows, so every row is
                # z-scored in one consistent frame. Computing the stats
                # before the restore mixed frames — with degenerate features
                # the meta-standardized and raw magnitudes differ by orders
                # of magnitude and every score was poisoned.
                if meta_mu is not None:
                    X_q[cache_q] = X_q[cache_q] * meta_sd + meta_mu
                    X_c[cache_c] = X_c[cache_c] * meta_sd + meta_mu
                all_rows = np.vstack([X_q[mask_q], X_c[mask_c]])
                mu = all_rows.mean(axis=0)
                sd = all_rows.std(axis=0)
                sd = np.where(sd <= 0, 1.0, sd)
                X_q[mask_q] = (X_q[mask_q] - mu) / sd
                X_c[mask_c] = (X_c[mask_c] - mu) / sd

        # Whiten the standardized snapshot BEFORE any scoring sees the
        # vectors (both methods: the snapshot is always the V2 schema).
        if mask_q.any() or mask_c.any():
            self._prepare_v2_scoring(cache_data, X_q, X_c, mask_q, mask_c,
                                     query_ids, pool_ids)

        q_vec = X_q[mask_q].mean(axis=0)
        keep = mask_c
        scores = np.full(len(pool_ids), np.nan)
        if keep.any():
            self._log(f"Step 5/6 — Scoring similarity ({self.method}) for "
                      f"{int(keep.sum())} usable candidates")
            self._progress(5, PROFILE_FIRST_TOTAL_STEPS,
                           self._scoring_step_label())
            scores[keep] = self._similarity_matrix(
                q_vec, X_c[keep], pool_subset=keep)

        query_type = ""
        if len(query_df):
            raw_query_type = query_df["type"].iloc[0]
            query_type = "" if pd.isna(raw_query_type) else str(raw_query_type).strip()
        query_ids_set = set(query_ids)
        # Type lookup: the full neuron table / index map, with the vector
        # cache's labels overriding for the neurons it covers (the cache is
        # freshest there). The old replace-all behaviour silently dropped
        # every pool neuron outside the cache from the type-level results.
        id_to_type, id_to_instance = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        if cache_data is not None:
            id_to_type.update(
                {self._body_id(b): t for b, t in zip(cache_data["bodyIds"], cache_data["types"])}
            )
            id_to_instance.update(
                {self._body_id(b): i for b, i in zip(cache_data["bodyIds"], cache_data["instances"])}
            )
        intra = float("nan")
        # Type queries are resolved to all members of the queried type. Use
        # those unified-standardized query vectors directly, rather than a
        # possibly stale cache snapshot loaded before fetched query vectors
        # were appended.
        if self.level == "type" and query_type and mask_q.any():
            query_ok = np.where(mask_q)[0]
            intra = self._intra_type_similarity(
                query_type,
                np.asarray(
                    [query_ids[i] for i in query_ok],
                    dtype=object if self._is_flywire() else np.int64,
                ),
                [query_type for _ in query_ok],
                X_q[query_ok],
            )
        elif using_cache_stats and cache_data is not None and len(cache_data["bodyIds"]):
            # BodyId queries still use the complete cached type population so
            # their same-type rows retain the established reference value.
            intra = self._intra_type_similarity(
                query_type, cache_data["bodyIds"], cache_data["types"],
                cache_data["X"],
            )
        if not np.isfinite(intra) and query_type and mask_q.any():
            ok = np.where(mask_q)[0]
            intra = self._intra_type_similarity(
                query_type,
                np.array(
                    [query_ids[i] for i in ok],
                    dtype=object if self._is_flywire() else np.int64,
                ),
                [str(query_df["type"].iloc[i]).strip() if i < len(query_df) else ""
                 for i in ok],
                X_q[ok],
            )

        # NBLAST (method="nblast"): score every scored candidate against the
        # query BEFORE aggregation, so the type-level table reflects NBLAST
        # instead of the vector prefilter (which only ordered the pool).
        nblast_scores: Dict[int, float] = {}
        if self.method == "nblast":
            # NBLAST needs skeletons, not vectors: score the WHOLE pool.
            # The keep mask only bounds the vector scorer — on FlyWire the
            # V1 mesh snapshot covers a small fraction of the pool, so
            # gating on it would silently leave most candidates unscored
            # (NaN). Unresolvable skeletons return None from
            # _dotprops_for_ids and are dropped there.
            nblast_scores = self._compute_nblast_scores(
                query_df,
                scored_pool_ids=[self._body_id(pool_ids[i])
                                 for i in range(len(pool_ids))],
                neurons={**query_neurons, **pool_neurons},
            )

        def _similarity_for(bid, vector_sim: float, is_same: bool) -> float:
            """NBLAST override: scored candidates take their NBLAST score;
            unscored same-type rows keep the vector reference, unscored
            cross-type rows drop to NaN (rank last)."""
            if not nblast_scores:
                return vector_sim
            n = nblast_scores.get(bid)
            if n is not None:
                return float(n)
            return vector_sim if is_same else float("nan")

        rows: List[Dict[str, object]] = []
        if self.level == "type":
            # Compare every resolved query member to every pool candidate.
            # Candidates whose vector is unavailable keep a NaN similarity
            # row so the written candidate list stays complete (they rank
            # last and are excluded from type aggregations and rendering).
            candidate_indices = [i for i, ok in enumerate(keep) if ok]
            query_indices = [i for i, ok in enumerate(mask_q) if ok]
            cand_pos = {ci: k for k, ci in enumerate(candidate_indices)}
            for query_i in query_indices:
                q_bid = self._body_id(query_ids[query_i])
                raw_q_type = query_df["type"].iloc[query_i]
                q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()
                q_scores = np.full(len(pool_ids), np.nan)
                q_blocks: Dict[str, np.ndarray] = {}
                if candidate_indices:
                    q_scores[candidate_indices], q_blocks = \
                        self._similarity_matrix_with_blocks(
                            X_q[query_i], X_c[candidate_indices],
                            pool_subset=candidate_indices,
                            query_index=query_i
                        )
                for candidate_i in range(len(pool_ids)):
                    bid = self._body_id(pool_ids[candidate_i])
                    if bid in query_ids_set:
                        continue
                    target_type = str(id_to_type.get(bid, "") or "").strip()
                    is_same = target_type == q_type if target_type else False
                    pos = cand_pos.get(candidate_i)
                    row = {
                        "source_bodyId": q_bid,
                        "source_type": q_type,
                        "target_bodyId": bid,
                        "target_type": target_type,
                        "target_instance": id_to_instance.get(bid, ""),
                        "profile_similarity": prof_by_id.get(bid, np.nan),
                        "roi_similarity": roi_by_id.get(bid, np.nan),
                        "similarity": _similarity_for(
                            bid, float(q_scores[candidate_i]), is_same),
                        "is_same_type": is_same,
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                        "candidate_source": source,
                    }
                    for name in ("shape", "spatial"):
                        if name in q_blocks and pos is not None:
                            row[f"sim_{name}"] = float(q_blocks[name][pos])
                    if "roi" in q_blocks and pos is not None:
                        row["sim_roi"] = float(q_blocks["roi"][pos])
                    rows.append(row)
            rows.extend(self._type_query_intra_rows(
                query_df, X_q, mask_q, intra, candidate_source=source
            ))
        else:
            # Every pool candidate gets a row: comparable ones carry their
            # similarity, the rest a NaN (sorted last, never rendered).
            for i, bid in enumerate(pool_ids):
                bid = self._body_id(bid)
                target_type = str(id_to_type.get(bid, "") or "").strip()
                if bid in query_ids_set:
                    continue
                is_same = target_type == query_type if target_type else False
                rows.append({
                    "source_bodyId": query_ids[0],
                    "source_type": query_type,
                    "target_bodyId": bid,
                    "target_type": target_type,
                    "target_instance": id_to_instance.get(bid, ""),
                    "profile_similarity": prof_by_id.get(bid, np.nan),
                    "roi_similarity": roi_by_id.get(bid, np.nan),
                    "similarity": _similarity_for(bid, float(scores[i]), is_same),
                    "is_same_type": is_same,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": source,
                })
        if not rows and not (self.level == "type" and np.isfinite(intra)):
            return pd.DataFrame(), pd.DataFrame()

        # V2 two-pass type reevaluation: the first pass identified candidate
        # types; their remaining (screen-ranked) members now join the pool
        # and are scored, so the aggregation below re-ranks on real
        # coverage instead of 1-2 lucky screen survivors.
        if self._is_v2 and self.level == "type" \
                and self.expand_top_types > 0 and np.isfinite(intra):
            pre_type_df = self._aggregate_type_rows(
                rows, query_type=query_type, intra=intra,
                query_type_count=self._type_member_count(
                    query_type, id_to_type, query_df=query_df),
                candidate_source=source,
            )
            self._expand_top_type_rows(
                rows, pre_type_df, query_df, query_ids, X_q, mask_q,
                id_to_type, id_to_instance, intra, source,
                cache=cache, cache_data=cache_data, mu=mu, sd=sd,
                scored_ids=set(self._body_ids(pool_ids)) | set(query_ids_set),
                pool_len=len(pool_ids), q_rep=q_rep, prof_by_id=prof_by_id,
            )

        # Connectivity-profile fallback: rows still without connectivity
        # evidence (threshold-screen misses and intra-type query pairs) are
        # scored with the Similarity>Connectivity machinery.
        if self._is_v2:
            self._fill_profile_similarity(rows, query_df)

        query_type_count = self._type_member_count(
            query_type, id_to_type, query_df=query_df
        )
        population_counts = None
        type_coverage = None
        if self._is_v2:
            from collections import Counter
            population_counts = {
                str(t or "").strip(): n
                for t, n in Counter(
                    str(v or "").strip() for v in id_to_type.values()
                ).items() if str(t or "").strip()
            }
            scored: Dict[str, set] = {}
            for r in rows:
                t = str(r.get("target_type", "") or "").strip()
                sim = r.get("similarity")
                if t and pd.notna(sim) and np.isfinite(sim):
                    scored.setdefault(t, set()).add(
                        self._body_id(r["target_bodyId"]))
            type_coverage = {
                t: (len(ids) / population_counts[t]
                    if population_counts.get(t) else np.nan)
                for t, ids in scored.items()
            }
        type_df = self._aggregate_type_rows(
            self._drop_contra_nblast_rows(rows),
            query_type=query_type,
            intra=intra,
            query_type_count=query_type_count,
            candidate_source=source,
            population_counts=population_counts,
        )

        # bodyId-level rows: top-N scored neurons.
        bodyid_df = self._bodyid_dataframe(rows, query_type=query_type,
                                           type_coverage=type_coverage)
        return bodyid_df, type_df

    def _fill_profile_similarity(self, rows: List[Dict[str, object]],
                                 query_df: pd.DataFrame) -> None:
        """Connectivity-profile fallback for rows without profile evidence.

        Rows whose profile_similarity is still NaN — threshold-screen misses
        and intra-type query pairs — are scored with the
        Similarity>Connectivity machinery: ``ConnectivityProfiler`` builds
        each involved neuron's partner-type profile from the shared local
        connection cache and ``ProfileComparator.weighted_cosine_similarity``
        scores the pairs. Best-effort: any failure keeps the NaN in place.
        """
        pending = [r for r in rows if pd.isna(r.get("profile_similarity"))]
        if not pending:
            return
        try:
            from comparison.connectivity_profiler import ConnectivityProfiler
            from comparison.profile_comparator import ProfileComparator
        except Exception as exc:
            self._log(f"Connectivity-profile fallback unavailable: {exc}")
            return
        neurons = sorted(
            {self._body_id(r["source_bodyId"]) for r in pending}
            | {self._body_id(r["target_bodyId"]) for r in pending},
            key=str,
        )
        try:
            profiler = ConnectivityProfiler(datasets=[self.dataset],
                                            verbose=False)
            profiles = profiler.get_profiles_batch(
                neurons, self.dataset, show_progress=False)
        except Exception as exc:
            self._log(f"Connectivity-profile fallback skipped: {exc}")
            return
        if not profiles:
            return
        filled = 0
        for r in pending:
            pa = profiles.get(self._body_id(r["source_bodyId"]))
            pb = profiles.get(self._body_id(r["target_bodyId"]))
            if pa is None or pb is None:
                continue
            try:
                sim = ProfileComparator.weighted_cosine_similarity(pa, pb)
            except Exception:
                continue
            r["profile_similarity"] = float(sim)
            filled += 1
        if filled:
            self._log(f"Profile evidence: filled {filled} rows from "
                      "connectivity profiles.")

    def _compute_nblast_scores(self, query_df: pd.DataFrame,
                               scored_pool_ids: List,
                               neurons: Dict[int, object]) -> Dict[int, float]:
        """Score the vector-prefiltered pool against the query with NBLAST.

        Runs BEFORE the type aggregation so the type-level table reflects
        NBLAST (the former post-aggregation refinement left type_summary
        showing the vector prefilter ranking). Returns {} when dotprops
        cannot be built — the vector scores are then kept, and
        ``nblast_applied`` is False so the fallback is observable.
        """
        self._progress(5, PROFILE_FIRST_TOTAL_STEPS, "Refining scores with NBLAST")
        query_ids_set = set(self._body_ids(query_df["bodyId"].tolist()))
        cand_ids = [b for b in scored_pool_ids if b not in query_ids_set]
        query_dp = self._dotprops_for_ids(
            self._body_ids(query_df["bodyId"].tolist()), neurons=neurons
        )
        if not query_dp:
            self._log("NBLAST: query dotprops unavailable; KEEPING VECTOR "
                      "SCORES (results are NOT NBLAST-refined).")
            self.nblast_applied = False
            return {}
        cand_dp = self._dotprops_for_ids(cand_ids, neurons=neurons)
        cand_dp = {b: dp for b, dp in cand_dp.items() if dp is not None}
        if not cand_dp:
            self._log("NBLAST: no candidate dotprops; KEEPING VECTOR SCORES "
                      "(results are NOT NBLAST-refined).")
            self.nblast_applied = False
            return {}
        self.nblast_applied = True
        return self._nblast_pairwise(query_dp, cand_dp, desc="NBLAST scoring")

    def _build_spatial_overlap(self, cache_data: Optional[dict],
                               query_ids: List,
                               pool_ids: List) -> Optional[Dict[str, np.ndarray]]:
        """Raw Hellinger histograms for the spatial-overlap term.

        Per query member, the pooled query centroid, and every pool
        candidate — all lateral-normalized at extraction, so intersections
        compare one hemisphere against one hemisphere. None when the raw
        rows are unavailable.
        """
        hist_lo, hist_hi = SPATIAL_HIST_SLICE
        if cache_data is None or cache_data.get("raw") is None:
            return None
        raw = np.asarray(cache_data["raw"], dtype=float)
        if raw.shape[1] < hist_hi:
            return None
        pos = {self._body_id(b): i for i, b in enumerate(cache_data["bodyIds"])}

        def _hists(ids: List) -> Optional[np.ndarray]:
            rows = [pos.get(self._body_id(b)) for b in ids]
            if any(r is None for r in rows):
                return None
            h = raw[np.asarray(rows, dtype=int), hist_lo:hist_hi]
            totals = h.sum(axis=1, keepdims=True)
            return h / np.maximum(totals, 1e-9)   # L1 proportions

        members = _hists(query_ids)
        pool = _hists(pool_ids)
        if members is None or pool is None:
            return None
        centroid = members.mean(axis=0)
        centroid = centroid / max(float(centroid.sum()), 1e-9)
        return {"members": members, "centroid": centroid, "pool": pool}

    def _slice_spatial_overlap(self, subset
                               ) -> Optional[Dict[str, np.ndarray]]:
        """Spatial-overlap pool matrix sliced to the rows a scoring call
        selected."""
        so = self._v2_spatial_overlap
        if so is None:
            return None
        pool = so["pool"]
        if subset is None:
            return so
        if isinstance(subset, np.ndarray) and subset.dtype == bool:
            sliced = pool[subset]
        else:
            sliced = pool[np.asarray(subset, dtype=int)]
        return {"members": so["members"], "centroid": so["centroid"],
                "pool": sliced}


    @staticmethod
    def _select_expansion_ids(top_types: List[str],
                              id_to_type: Dict[int, str],
                              scored_ids: set,
                              roi_rank: Dict[int, tuple],
                              per_type_cap: int, total_cap: int) -> List[int]:
        """Remaining members of the top-ranked types, in expansion order.

        Members already scored are skipped; the rest are prioritized by the
        ROI screen's full ranking (the strongest-overlap representatives of
        the type first), capped per type and in total. Pure/static so the
        selection policy is unit-testable in isolation.
        """
        members_by_type: Dict[str, List[int]] = {}
        wanted = {str(t or "").strip() for t in top_types if str(t or "").strip()}
        for bid, t in (id_to_type or {}).items():
            tt = str(t or "").strip()
            if tt in wanted:
                members_by_type.setdefault(tt, []).append(bid)
        out: List[int] = []
        for t in top_types:
            mems = [b for b in members_by_type.get(t, [])
                    if b not in scored_ids]
            mems.sort(key=lambda b: roi_rank.get(b, (1 << 30, b)))
            out.extend(mems[:max(0, int(per_type_cap))])
        return out[:max(0, int(total_cap))]

    def _expand_top_type_rows(self, rows: List[Dict[str, object]],
                              pre_type_df: pd.DataFrame,
                              query_df: pd.DataFrame,
                              query_ids: List,
                              X_q: np.ndarray, mask_q: np.ndarray,
                              id_to_type: Dict[int, str],
                              id_to_instance: Dict[int, str],
                              intra: float, source: str, *,
                              cache: SkeletonVectorCache,
                              cache_data: Optional[dict],
                              mu: np.ndarray, sd: np.ndarray,
                              scored_ids: set, pool_len: int,
                              q_rep: str,
                              prof_by_id: Optional[Dict[int, float]] = None) -> None:
        """Two-pass type reevaluation: score the top types' remaining members.

        Takes the first-pass top candidate types from ``pre_type_df``, pulls
        their unscored members into the pool (ROI-screen-ranked, bounded by
        ``expand_per_type``/``expand_top_types``), vectorizes them with the
        SAME standardization/whitening as the first pass, scores them against
        every query member, and appends their rows to ``rows`` (in place).
        The caller re-runs the aggregation afterwards; V1 never calls this.

        Fetch routing: FAFB loads skeletons through the healed-zip loader
        (``_load_fafb_skeletons``); NeuPrint keeps the batched SWC fetch.
        """
        inter = pre_type_df[~pre_type_df["is_intra_type"]] \
            if not pre_type_df.empty else pd.DataFrame()
        if inter.empty:
            return
        top_types = inter["target_type"].astype(str).tolist()[:self.expand_top_types]

        # Full screen ranking prioritizes each type's strongest-overlap
        # representatives first.
        roi_rank: Dict[int, tuple] = {}
        store = self._maybe_roi_store()
        if store is not None:
            try:
                full = store.screen([int(b) for b in query_ids], top_k=None)
                roi_rank = {int(b): (-float(s), int(b))
                            for b, s in zip(full["bodyId"],
                                            full["roi_similarity"])}
            except Exception:
                pass
        new_ids = self._select_expansion_ids(
            top_types, id_to_type, scored_ids, roi_rank,
            self.expand_per_type,
            self.expand_top_types * self.expand_per_type)
        if not new_ids:
            return

        cache_ids = set()
        if cache_data is not None:
            cache_ids = {self._body_id(b) for b in cache_data["bodyIds"]}
        missing = []
        for bid in new_ids:
            if bid in cache_ids:
                continue
            if cache.find_skeleton_file(bid) is not None:
                continue
            missing.append(bid)
        self._log(
            f"Type reevaluation: expanding the pool with "
            f"{len(new_ids)} members of the top {len(top_types)} candidate "
            f"types ({len(missing)} to fetch)...")
        self._progress(5, PROFILE_FIRST_TOTAL_STEPS,
                       f"Reevaluating top types (+{len(new_ids)} members)")
        fetch_cache = cache
        fetched_all: Dict[int, object] = {}
        if missing:
            if is_fafb_dataset(self.dataset):
                # FAFB: the healed zip is LOCAL — fetch through the shared
                # FlyWire loader instead of the generic batch fetcher, which
                # is mesh-native on this dataset and would leave expansion
                # members unscored. The zip and the repair caches are the
                # durable sources (no ``raw_skeletons`` mirror); the vector
                # rows are appended from the in-memory neurons below.
                loaded = self._load_fafb_skeletons(missing)
                fetched = {
                    int(b): n for b, n in (loaded or {}).items()
                    if n is not None
                }
                fetched_all = {
                    self._body_id(b): n for b, n in fetched.items()
                }
            else:
                # NeuPrint (and other datasets): batched SWC fetch.
                fetched_all = fetch_skeletons_on_demand_batch(
                    self.dataset, missing,
                    project_root=str(self.project_root),
                    persist=True, level=VECTOR_BASIS_RAW,
                    max_threads=min(NEUPRINT_FETCH_MAX_THREADS,
                                    max(1, int(self.n_workers))),
                    raw_cache=fetch_cache, vector_cache=None,
                )

        order: List[int] = []
        vec_by_id: Dict[int, np.ndarray] = {}
        appends: List[Tuple[int, np.ndarray, str]] = []
        cache_row: Dict[int, int] = {}
        if cache_data is not None:
            cache_row = {self._body_id(b): i
                         for i, b in enumerate(cache_data["bodyIds"])}
        for bid in new_ids:
            bid = self._body_id(bid)
            neuron = fetched_all.get(bid)
            if neuron is None:
                pkl = cache.find_skeleton_file(bid)
                if pkl is None:
                    # No in-memory neuron and no local file, but the id may
                    # still carry a cached vector row — the `missing` check
                    # above skips fetching exactly those ids. Reuse the raw
                    # row instead of silently dropping the member (without
                    # this, a member scores on the run that fetched it and
                    # vanishes from every later run until its skeleton file
                    # materializes).
                    row = cache_row.get(bid)
                    if row is None:
                        continue
                    vec = np.asarray(cache_data["raw"][row], dtype=float)
                    rep = (cache_data["rep"][row]
                           or cache_data.get("dataset_rep", ""))
                    if not np.isfinite(vec).all():
                        continue
                    if q_rep and rep != q_rep:
                        continue   # one representation per comparison
                    order.append(bid)
                    vec_by_id[bid] = vec
                    continue   # already cached: nothing to append
                neuron = _load_cached_skeleton_file(str(pkl))
            rep = _neuron_rep(neuron)
            if q_rep and rep != q_rep:
                continue   # one representation per comparison
            try:
                _, vec = cache._vectorize_neuron(neuron)
            except Exception:
                continue
            if not np.isfinite(vec).all():
                continue
            order.append(bid)
            vec_by_id[bid] = vec
            appends.append((bid, vec, rep))
        if appends:
            cache.append_vectors(appends, vector_basis=cache._default_basis())
        if not order:
            return

        # Same standardization + whitening as the first pass (mu/sd were
        # fixed there; meta stats never move during appends).
        dim = X_q.shape[1] if X_q.ndim == 2 else self._vector_dim_for(cache_data)
        X_new = np.full((len(order), dim), np.nan)
        for k, bid in enumerate(order):
            vec = vec_by_id[bid]
            if vec.shape[0] == dim:
                X_new[k] = (vec - mu) / sd
        W = (cache_data or {}).get("whiten")
        if W is not None and getattr(W, "size", 0):
            X_new = apply_whitening(W, X_new)

        # Extend the spatial-overlap histograms to cover the new
        # candidates.
        new_idx = np.arange(pool_len, pool_len + len(order))
        if self._v2_spatial_overlap is not None:
            hist_lo, hist_hi = SPATIAL_HIST_SLICE
            new_h = np.zeros((len(order), hist_hi - hist_lo))
            for k, bid in enumerate(order):
                vec = vec_by_id.get(bid)
                if vec is None:
                    continue
                h = np.asarray(vec[hist_lo:hist_hi], dtype=float)
                new_h[k] = h / max(float(h.sum()), 1e-9)
            so = self._v2_spatial_overlap
            self._v2_spatial_overlap = {
                "members": so["members"], "centroid": so["centroid"],
                "pool": np.vstack([so["pool"], new_h])}
        query_indices = [i for i, ok in enumerate(mask_q) if ok]
        id_to_type = dict(id_to_type)
        added_rows = 0
        for query_i in query_indices:
            q_bid = self._body_id(query_ids[query_i])
            raw_q_type = query_df["type"].iloc[query_i]
            q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()
            totals, blocks = self._similarity_matrix_with_blocks(
                X_q[query_i], X_new, pool_subset=new_idx, query_index=query_i)
            for k, bid in enumerate(order):
                if bid in scored_ids:
                    continue
                target_type = str(id_to_type.get(bid, "") or "").strip()
                row = {
                    "source_bodyId": q_bid,
                    "source_type": q_type,
                    "target_bodyId": bid,
                    "target_type": target_type,
                    "target_instance": id_to_instance.get(bid, ""),
                    "profile_similarity": (prof_by_id or {}).get(bid, np.nan),
                    "roi_similarity": (-roi_rank[bid][0]
                                       if bid in roi_rank else np.nan),
                    "similarity": float(totals[k]),
                    "is_same_type": target_type == q_type if target_type else False,
                    "intra_type_similarity": intra,
                    "method": self.method,
                    "metric": self.metric,
                    "candidate_source": source,
                    "pool_stage": "expansion",
                }
                for name in ("shape", "spatial", "roi"):
                    if name in blocks:
                        row[f"sim_{name}"] = float(blocks[name][k])
                rows.append(row)
                added_rows += 1
        self._log(f"Type reevaluation: scored {added_rows} expanded-member "
                  "rows; re-aggregating types.")

    def _nblast_pairwise(self, query_dps: Dict[int, object],
                         cand_dps: Dict[int, object],
                         desc: str = "NBLAST scoring") -> Dict[int, float]:
        """Score every query-candidate pair in-process with a per-individual
        progress bar.

        Mirrors ``navis.nblast`` defaults (forward, normalized, FCWB scoring)
        but scores pair-by-pair with ``navis.nbl.NBlaster`` so the bar can
        name the current candidate. This also avoids navis' process-pool
        startup (its spawn overhead dominates small candidate pools)."""
        from navis.nbl.nblast_funcs import NBlaster

        nb = NBlaster(use_alpha=False, normalized=True, progress=False)
        q_idx = {}
        for q_bid, q_dp in query_dps.items():
            q_idx[q_bid] = nb.append(q_dp, self_hit=nb.calc_self_hit(q_dp))
        t_idx = {}
        for t_bid, t_dp in cand_dps.items():
            t_idx[t_bid] = nb.append(t_dp, self_hit=nb.calc_self_hit(t_dp))

        total = len(q_idx) * len(t_idx)
        pbar = tqdm(total=total, desc=desc, unit="pair",
                    disable=not self.verbose, leave=False, file=sys.stdout)
        nblast_scores: Dict[int, float] = {}
        try:
            for q_bid, qi in q_idx.items():
                for t_bid, ti in t_idx.items():
                    pbar.set_postfix_str(f"{q_bid} x {t_bid}")
                    try:
                        val = float(nb.single_query_target(qi, ti,
                                                           scores='forward'))
                    except Exception:
                        val = float("nan")
                    if np.isfinite(val):
                        nblast_scores[t_bid] = max(
                            nblast_scores.get(t_bid, -np.inf), val)
                    pbar.update(1)
        finally:
            pbar.close()
        # Canonical id keys: _dotprops_for_ids keys its dict by plain ints,
        # while every consumer looks scores up by the dataset-canonical id
        # (string bodyIds on FlyWire — an int key would silently miss ALL
        # lookups there).
        return {self._body_id(b): v for b, v in nblast_scores.items()}

    # ------------------------------------------------------------------ vector
    def _intra_type_similarity(self, type_name: str, body_ids: np.ndarray,
                              types: List[str], X: np.ndarray) -> float:
        """Mean pairwise similarity among a type's members (vector-based).

        1.0 for a single member (trivially identical); NaN when the type has
        no members in the population."""
        if not type_name:
            return float("nan")
        idxs = [i for i, t in enumerate(types) if t == type_name]
        n = len(idxs)
        if n == 0:
            return float("nan")
        if n == 1:
            return 1.0
        sub = X[idxs]
        # Compute the whole pairwise matrix once.  The previous row-by-row
        # implementation recalculated all row norms for every member and was
        # unnecessarily slow for large types.
        pair = self._pairwise_similarity(sub)
        total = float(pair.sum()) - n  # drop the diagonal (self = 1)
        return total / (n * (n - 1))

    def _type_member_count(self, type_name: str, id_to_type: Dict[int, str],
                           query_df: Optional[pd.DataFrame] = None) -> int:
        """Count the bodyIds belonging to a type without relying on a cache.

        A profile-first run can load ``cache_data`` before it vectorizes the
        query members.  Counting from that snapshot therefore undercounts a
        queried type (and historically produced ``n_bodyids=0``).  The neuron
        index is the authoritative population; ``query_df`` is a final
        fallback for small/test datasets whose index is unavailable.
        """
        member_ids = set()
        for bid, t in (id_to_type or {}).items():
            if str(t or "").strip() == str(type_name or "").strip():
                try:
                    member_ids.add(self._body_id(bid))
                except (TypeError, ValueError):
                    continue
        if query_df is not None and not query_df.empty:
            for _, row in query_df.iterrows():
                if str(row.get("type", "") or "").strip() != str(type_name or "").strip():
                    continue
                try:
                    member_ids.add(self._body_id(row.get("bodyId")))
                except (TypeError, ValueError):
                    continue
        return len(member_ids)

    def _type_query_intra_rows(
        self,
        query_df: pd.DataFrame,
        X: np.ndarray,
        mask: np.ndarray,
        intra: float,
        candidate_source: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Return unique bodyId-level pairs within a type query.

        Type searches resolve to every bodyId in the queried type.  Keep the
        individual same-type comparisons in ``results.csv`` as well as the
        type-level reference row.  Emit one row per unordered pair, while
        orienting the rows so every member is represented as a source when
        there are at least three members.  The mean of these unique pairs
        matches the off-diagonal mean used by ``_intra_type_similarity``
        because the supported similarity metrics are symmetric.
        """
        if self.level != "type" or query_df is None or query_df.empty:
            return []

        records = []
        for i, (_, row) in enumerate(query_df.iterrows()):
            if i >= len(mask) or not bool(mask[i]):
                continue
            try:
                bid = self._body_id(row["bodyId"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_type = row.get("type", "")
            q_type = "" if pd.isna(raw_type) else str(raw_type).strip()
            records.append((i, bid, q_type, str(row.get("instance", "") or "")))

        rows: List[Dict[str, object]] = []
        pair_directions = {}
        n_records = len(records)
        # A cyclic orientation gives every member an outgoing row without
        # duplicating an unordered pair.  For two members there is only one
        # possible row, so both members are still represented across its
        # endpoints.
        if n_records >= 3:
            for i in range(n_records):
                j = (i + 1) % n_records
                pair_directions[tuple(sorted((i, j)))] = (i, j)
        for i in range(n_records):
            for j in range(i + 1, n_records):
                pair_directions.setdefault((i, j), (i, j))

        pair_matrix = self._pairwise_similarity(
            np.asarray([X[i] for i, _, _, _ in records], dtype=float),
        )
        for source_i, target_i in pair_directions.values():
            _, source_bid, source_type, _ = records[source_i]
            _, target_bid, target_type, target_instance = records[target_i]
            score = float(pair_matrix[source_i, target_i])
            row: Dict[str, object] = {
                "source_bodyId": source_bid,
                "source_type": source_type,
                "target_bodyId": target_bid,
                "target_type": target_type,
                "target_instance": target_instance,
                "profile_similarity": np.nan,
                "roi_similarity": np.nan,
                "similarity": score,
                "is_same_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            }
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            rows.append(row)
        return rows

    def _soma_side_map(self) -> Dict[int, str]:
        """bodyId -> 'L' | 'R' | '' for ipsi/contra classification.

        Delegates to the shared index reader (``somaSide`` on NeuPrint
        datasets, ``hemisphere`` on FlyWire), memoized per run. Neurons the
        index does not side (midline/unknown) map to '' so they are never
        side-filtered. Best-effort: any read failure means no filtering.
        """
        if self._soma_sides is not None:
            return self._soma_sides
        raw = _dataset_soma_side_map(self.dataset, str(self.project_root))
        sides = {bid: {"left": "L", "right": "R"}.get(name, "")
                 for bid, name in raw.items()}
        self._soma_sides = sides
        return sides

    def _drop_contra_nblast_rows(
            self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Rows pairing opposite known sides, dropped for NBLAST type means.

        The morphology benchmark measured NBLAST contralateral same-type
        pairs at chance (median score -0.03 vs 0.89 ipsilateral; partner
        RR 0.13 vs 0.84) — mirror arbors sit far apart in its absolute-
        coordinate matching. Type means over both sides therefore dilute
        the ipsilateral evidence with noise, so ``method="nblast"``
        aggregations exclude contralateral pairs. vector_v2 is
        mirror-invariant by lateral normalization and keeps them. Rows with
        an unknown or midline side on either end are always kept.
        """
        if self.method != "nblast" or not rows:
            return rows
        sides = self._soma_side_map()
        if not sides:
            return rows

        def _side_of(value) -> str:
            try:
                # Index keys are plain ints; canonical bodyIds on FlyWire
                # are numeric strings.
                return sides.get(int(self._body_id(value)), "")
            except (TypeError, ValueError):
                return ""

        out: List[Dict[str, object]] = []
        dropped = 0
        for row in rows:
            s = _side_of(row.get("source_bodyId"))
            t = _side_of(row.get("target_bodyId"))
            if s in ("L", "R") and t in ("L", "R") and s != t:
                dropped += 1
                continue
            out.append(row)
        if dropped:
            self._log(f"NBLAST: {dropped} contralateral pair rows excluded "
                      "from type means (mirror scores unreliable).")
        return out

    def _aggregate_type_rows(
        self,
        rows: List[Dict[str, object]],
        query_type: str,
        intra: float,
        query_type_count: int = 0,
        candidate_source: Optional[str] = None,
        population_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Aggregate bodyId rows while preserving the queried type count.

        For inter-type rows, ``n_bodyids`` counts unique target bodyIds even
        when a type query contributes one row per query source.  The queried
        type is a reference row, so its count is the population member count,
        not the number of pairwise rows.

        When ``population_counts`` is provided (V2), the type statistic
        (mean, or max under ``type_agg="max"``) is scaled by the continuous
        coverage factor ``sqrt(type_coverage)`` (:func:`_coverage_factor`)
        and a ``type_coverage`` column (scored members / population
        members) is added: a sparsely covered type's sample is a
        screen-selected slice of it and is gently damped, a fully covered
        type passes through untouched, and the weighting is continuous
        everywhere. The unweighted statistic is kept in
        ``similarity_raw``; ``similarity_max`` records the best member;
        per-block means (``sim_shape`` etc.) are averaged through when the
        rows carry them.
        """
        import collections

        grouped: Dict[str, List[Dict[str, object]]] = collections.defaultdict(list)
        for row in rows:
            target_type = str(row.get("target_type", "") or "").strip()
            if target_type:
                grouped[target_type].append(row)

        block_keys = [k for k in ("sim_shape", "sim_spatial",
                                  "sim_roi")
                      if any(k in r for r in rows)]

        def _type_stat(values: List[float]) -> float:
            return float(np.max(values)) if self.type_agg == "max" \
                else float(np.mean(values))

        agg_rows: List[Dict[str, object]] = []
        for target_type, subrows in grouped.items():
            values = [float(row["similarity"]) for row in subrows
                      if pd.notna(row.get("similarity"))]
            if not values:
                continue
            target_ids = {
                self._body_id(row["target_bodyId"])
                for row in subrows
                if row.get("target_bodyId") is not None
                and pd.notna(row.get("similarity"))
            }
            profile_values = [float(row["profile_similarity"])
                              for row in subrows
                              if pd.notna(row.get("profile_similarity"))]
            roi_values = [float(row["roi_similarity"])
                          for row in subrows
                          if pd.notna(row.get("roi_similarity"))]
            is_intra = target_type == str(query_type or "").strip()
            sim_raw = float(intra if is_intra and np.isfinite(intra)
                            else _type_stat(values))
            similarity = sim_raw
            type_coverage = np.nan
            if population_counts is not None and not is_intra:
                pop = int(population_counts.get(target_type, 0))
                type_coverage = (len(target_ids) / pop) if pop > 0 else np.nan
                if np.isfinite(type_coverage):
                    similarity = sim_raw * _coverage_factor(type_coverage)
            row = {
                "target_type": target_type,
                "similarity": float(similarity),
                "n_bodyids": int(query_type_count if is_intra and query_type_count
                                 else len(target_ids)),
                "profile_similarity": (
                    np.nan if is_intra else
                    (float(np.mean(profile_values)) if profile_values else np.nan)
                ),
                "roi_similarity": (
                    np.nan if is_intra else
                    (float(np.mean(roi_values)) if roi_values else np.nan)
                ),
                "is_intra_type": is_intra,
                "intra_type_similarity": intra if is_intra else float("nan"),
                "method": self.method,
                "metric": self.metric,
            }
            if population_counts is not None:
                row["similarity_raw"] = sim_raw
                row["similarity_max"] = (float(np.max(values))
                                         if not is_intra else np.nan)
                row["type_coverage"] = type_coverage
            for key in block_keys:
                vals = [float(r[key]) for r in subrows
                        if key in r and pd.notna(r.get(key))]
                row[key] = float(np.mean(vals)) if vals else np.nan
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            agg_rows.append(row)

        if (query_type and query_type not in grouped and np.isfinite(intra)):
            row = {
                "target_type": str(query_type).strip(),
                "similarity": intra,
                "n_bodyids": int(query_type_count),
                "profile_similarity": np.nan,
                "roi_similarity": np.nan,
                "is_intra_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            }
            if population_counts is not None:
                row["similarity_raw"] = float(intra)
                row["similarity_max"] = np.nan
                pop = int(population_counts.get(str(query_type).strip(), 0))
                row["type_coverage"] = (int(query_type_count) / pop
                                        if pop > 0 else np.nan)
            if candidate_source is not None:
                row["candidate_source"] = candidate_source
            agg_rows.append(row)

        agg_rows = sorted(
            agg_rows,
            key=lambda row: (
                not bool(row["is_intra_type"]),
                -float(row["similarity"]),
                str(row["target_type"]),
            ),
        )
        result = pd.DataFrame(agg_rows).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def _bodyid_dataframe(
        self, rows: List[Dict[str, object]], query_type: str = "",
        type_coverage: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Rank bodyId rows, retaining every compared candidate.

        All compared neurons are kept (no top-N truncation; the candidate cap
        already bounded the pool). Rows without a finite similarity
        (unscorable candidates) are kept but rank after every scored row. For
        type queries every resolved same-type pair is preserved so the query
        does not collapse to one source. ``type_coverage`` (V2) maps target
        type -> pool coverage and is attached per row so the exported
        bodyId-level table carries the same trust signal as the type-level
        aggregation.
        """
        if not rows:
            return pd.DataFrame()

        def _sim_key(row: Dict[str, object]) -> float:
            sim = row.get("similarity")
            value = float(sim)
            return -value if np.isfinite(value) else float("inf")

        if self.level == "type" and query_type:
            ordered = sorted(
                rows,
                key=lambda row: (
                    _sim_key(row),
                    self._body_id(row["source_bodyId"]),
                    self._body_id(row["target_bodyId"]),
                ),
            )
        else:
            ordered = sorted(
                rows,
                key=lambda row: (_sim_key(row), self._body_id(row["target_bodyId"])),
            )
        result = pd.DataFrame(ordered).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        if type_coverage:
            result["type_coverage"] = (
                result["target_type"].map(type_coverage).astype(float))
        # Canonical column order, mirroring type_summary.csv: target
        # identity, score and trust, evidence columns, same-type flag and
        # reference, method meta, per-block scores, then the query-side
        # identity and run annotations. Optional columns appear only when
        # present.
        base = ["target_bodyId", "target_type", "target_instance",
                "similarity", "type_coverage", "profile_similarity",
                "roi_similarity",
                "is_same_type", "intra_type_similarity", "method", "metric"]
        tail = ["sim_shape", "sim_spatial", "sim_roi",
                "source_bodyId", "source_type", "candidate_source",
                "pool_stage"]
        ordered_cols = [c for c in base + tail if c in result.columns]
        extra = [c for c in result.columns
                 if c not in ordered_cols and c != "rank"]
        result = result[["rank"] + ordered_cols + extra]
        return result

    def _vector_search(self, query_df: pd.DataFrame, data: dict
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Vector similarity over the cache population.

        Returns (bodyId-level df, type-level df): results.csv always holds
        the bodyId rows and type_summary.csv the type rows, whatever the
        query kind. Type queries score every resolved query member against
        every candidate member in the bodyId export; results.csv keeps
        ordered intra-type pairs. Inter-type summary rows average those
        pairs; the cache-direct query-type reference retains its centroid
        score contract."""
        body_ids = data["bodyIds"]
        types = data["types"]
        X = data["X"]
        if self._is_v2:
            # Cache-direct V2 scores in the whitened space (the screen-first
            # path whitens inside ``_prepare_v2_scoring`` instead). No
            # ROI-expansion block here: cache-direct runs carry no screen
            # alignment, and FAFB has no ROI table anyway.
            W = data.get("whiten")
            if W is not None and getattr(W, "size", 0):
                data["X"] = apply_whitening(W, X)
                X = data["X"]
            self._v2_spatial_overlap = None
        query_ids = self._body_ids(query_df["bodyId"].tolist())
        query_ids_set = set(query_ids)
        q_type = ""
        if len(query_df):
            raw_q_type = query_df["type"].iloc[0]
            q_type = "" if pd.isna(raw_q_type) else str(raw_q_type).strip()

        q_mask = np.isin(body_ids, query_ids)
        # Cache-direct type searches use every cached member of the queried
        # type; the type index below supplies the authoritative member count
        # even if a legacy query resolver returned only one member.
        intra = self._intra_type_similarity(q_type, body_ids, types, X)
        type_map, _ = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        query_type_count = self._type_member_count(
            q_type, type_map, query_df=query_df
        )

        # --- bodyId rows ---
        rows = []
        if self.level == "type":
            if not q_mask.any():
                return pd.DataFrame(), pd.DataFrame()
            query_pair_X = np.full(
                (len(query_df), X.shape[1]), np.nan, dtype=float
            )
            query_pair_mask = np.zeros(len(query_df), dtype=bool)
            for query_i, qrow in enumerate(query_df.itertuples(index=False)):
                q_bid = self._body_id(qrow.bodyId)
                q_idx = np.where(body_ids == q_bid)[0]
                if not len(q_idx):
                    continue
                q_vec = X[q_idx[0]]
                query_pair_X[query_i] = q_vec
                query_pair_mask[query_i] = True
                scores = self._similarity_matrix(q_vec, X)
                source_type = str(getattr(qrow, "type", q_type) or "").strip()
                for i, bid in enumerate(body_ids):
                    bid = self._body_id(bid)
                    if bid in query_ids_set:
                        continue
                    target_type = str(types[i] or "").strip()
                    rows.append({
                        "source_bodyId": q_bid,
                        "source_type": source_type,
                        "target_bodyId": bid,
                        "target_type": target_type,
                        "target_instance": data["instances"][i],
                        "similarity": float(scores[i]),
                        "is_same_type": target_type == source_type,
                        "intra_type_similarity": intra,
                        "method": self.method,
                        "metric": self.metric,
                    })
            rows.extend(self._type_query_intra_rows(
                query_df, query_pair_X, query_pair_mask, intra,
                candidate_source=None
            ))
        else:
            # Multi-query support: each query row is ranked independently
            # (self and any co-query neurons excluded from its rows).
            for _, qrow in query_df.iterrows():
                q_vec = self._vector_for_body_id(
                    self._body_id(qrow["bodyId"]), body_ids, X
                )
                if q_vec is None:
                    continue
                scores = self._similarity_matrix(q_vec, X)
                row_intra = self._intra_type_similarity(
                    qrow["type"], body_ids, types, X
                )
                for i, bid in enumerate(body_ids):
                    bid = self._body_id(bid)
                    if bid in query_ids_set:
                        continue
                    rows.append({
                        "source_bodyId": self._body_id(qrow["bodyId"]),
                        "source_type": str(qrow["type"] or "").strip(),
                        "target_bodyId": bid,
                        "target_type": str(types[i] or "").strip(),
                        "target_instance": data["instances"][i],
                        "similarity": float(scores[i]),
                        "is_same_type": types[i] == qrow["type"],
                        "intra_type_similarity": row_intra,
                        "method": self.method,
                        "metric": self.metric,
                    })
        bodyid_df = self._bodyid_dataframe(rows, query_type=q_type)

        # --- type rows ---
        if self.level == "type":
            # Aggregate inter-type scores over every query-member/candidate
            # pair.  Keep the established centroid score for the query-type
            # reference itself; ``intra_type_similarity`` remains the
            # off-diagonal member-pair reference.
            q_vec = X[q_mask].mean(axis=0) if q_mask.any() else None
            type_df = self._aggregate_type_rows(
                rows,
                query_type=q_type,
                intra=intra,
                query_type_count=(query_type_count or int(q_mask.sum())),
            )
            if q_vec is not None and q_type and not type_df.empty:
                centroid_scores = self._similarity_matrix(q_vec, X)
                query_type_mask = np.asarray(
                    [str(t or "").strip() == q_type for t in types],
                    dtype=bool,
                )
                if query_type_mask.any():
                    centroid_similarity = float(
                        np.mean(centroid_scores[query_type_mask])
                    )
                    type_df.loc[
                        type_df["target_type"] == q_type, "similarity"
                    ] = centroid_similarity
                    if query_type_count:
                        type_df.loc[
                            type_df["target_type"] == q_type, "n_bodyids"
                        ] = query_type_count
        else:
            # Keep the established centroid aggregation for bodyId queries'
            # per-type summary.
            q_vec = X[q_mask].mean(axis=0) if q_mask.any() else None
            type_df = pd.DataFrame()
            if q_vec is not None:
                scores = self._similarity_matrix(q_vec, X)
                type_df = self._aggregate_by_type(
                    body_ids, types, scores, query_type=q_type, X=X,
                )
        return bodyid_df, type_df

    def _vector_for_body_id(self, bid: int, body_ids: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
        idx = np.where(body_ids == bid)[0]
        if not len(idx):
            return None
        return X[idx[0]]

    def _aggregate_by_type(self, body_ids: np.ndarray, types: List[str], scores: np.ndarray,
                           query_type: str = "",
                           X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Aggregate bodyId scores to type level (mean of member scores).

        The query type itself is included as the first row (is_intra_type=True)
        so intra-type similarity data is part of the results. ``X``/``metric``
        are used for the pairwise intra-type similarity column."""
        import collections
        agg: Dict[str, List[float]] = collections.defaultdict(list)
        for i, t in enumerate(types):
            if not t:
                continue
            agg[t].append(float(scores[i]))
        rows = []
        for t, vals in agg.items():
            intra = float("nan")
            if X is not None and t == query_type:
                intra = self._intra_type_similarity(t, body_ids, types, X)
            rows.append({
                "target_type": t,
                "similarity": float(np.mean(vals)),
                "n_bodyids": len(vals),
                "is_intra_type": t == query_type,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": self.metric,
            })
        # The intra-type (query-type) row always ranks first as the reference.
        rows = sorted(rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
        results = pd.DataFrame(rows).reset_index(drop=True)
        results.insert(0, "rank", np.arange(1, len(results) + 1))
        return results

    # ------------------------------------------------------------------ nblast
    def _nblast_search(self, query_df: pd.DataFrame, data: dict
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        body_ids = data["bodyIds"]
        types = data["types"]
        X = data["X"]

        # Vector prefilter: top-K candidate bodyIds by cosine, excluding the
        # query itself. Same-type members stay in (they provide the intra-type
        # reference data at the type level).
        query_ids = set(self._body_ids(query_df["bodyId"].tolist()))
        candidate_mask = ~np.isin(body_ids, list(query_ids))

        scores = np.full(len(body_ids), -np.inf)
        for _, qrow in query_df.iterrows():
            q_vec = self._vector_for_body_id(
                self._body_id(qrow["bodyId"]), body_ids, X
            )
            if q_vec is None:
                continue
            s = self._similarity_matrix(q_vec, X)
            scores = np.maximum(scores, s)
        prefilter_idx = np.where(candidate_mask)[0]
        prefilter_idx = prefilter_idx[np.argsort(-scores[prefilter_idx])][: self.candidate_cap]

        if not len(prefilter_idx):
            self._log("NBLAST: no candidates survived the vector prefilter.")
            return pd.DataFrame(), pd.DataFrame()

        # Build dotprops for query + candidates in one load/fetch phase
        # (in microns; NEVER cached). Splitting after construction prevents a
        # query/candidate overlap from triggering two online batch requests.
        cand_ids = [self._body_id(body_ids[i]) for i in prefilter_idx]
        all_dotprop_ids = []
        seen_dotprop_ids = set()
        for bid in list(query_ids) + cand_ids:
            bid = self._body_id(bid)
            if bid not in seen_dotprop_ids:
                seen_dotprop_ids.add(bid)
                all_dotprop_ids.append(bid)
        all_dp = self._dotprops_for_ids(all_dotprop_ids)
        query_dp = {bid: all_dp.get(bid) for bid in query_ids
                    if all_dp.get(bid) is not None}
        if not query_dp:
            raise ValueError("NBLAST: could not build dotprops for the query neuron(s).")
        cand_dp = {bid: all_dp.get(bid) for bid in cand_ids}
        cand_dp = {bid: dp for bid, dp in cand_dp.items() if dp is not None}

        self._log(f"NBLAST: {len(query_dp)} query x {len(cand_dp)} candidates "
                  f"({self.candidate_cap} prefiltered)")
        nblast_scores = self._nblast_pairwise(
            query_dp, cand_dp, desc="NBLAST scoring")

        # Rank candidates by NBLAST score.
        ranked = sorted(nblast_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        id_to_type = {self._body_id(b): t for b, t in zip(body_ids, types)}
        id_to_inst = {
            self._body_id(b): i for b, i in zip(body_ids, data["instances"])
        }

        query_type = ""
        if len(query_df):
            raw_query_type = query_df["type"].iloc[0]
            query_type = "" if pd.isna(raw_query_type) else str(raw_query_type).strip()
        query_mask = np.isin(body_ids, list(query_ids))
        if self.level == "type" and query_mask.any():
            intra = self._intra_type_similarity(
                query_type,
                body_ids[query_mask],
                [query_type for _ in range(int(query_mask.sum()))],
                X[query_mask],
            )
        else:
            intra = self._intra_type_similarity(query_type, body_ids, types, X)

        type_map, _ = _load_neuron_type_map(
            self.dataset, str(self.project_root)
        )
        query_type_count = self._type_member_count(
            query_type, type_map, query_df=query_df
        )

        # --- bodyId rows (query members excluded) ---
        rows: List[Dict[str, object]] = []
        for bid, score in ranked:
            if bid in query_ids:
                continue
            t = str(id_to_type.get(bid, "") or "").strip()
            rows.append({
                "source_bodyId": self._body_id(query_df["bodyId"].iloc[0]),
                "source_type": query_type,
                "target_bodyId": bid,
                "target_type": t,
                "target_instance": id_to_inst.get(bid, ""),
                "similarity": float(score),
                "is_same_type": t == query_type,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": "nblast",
            })
        if self.level == "type" and query_mask.any():
            query_pair_X = np.full(
                (len(query_df), X.shape[1]), np.nan, dtype=float
            )
            query_pair_mask = np.zeros(len(query_df), dtype=bool)
            for query_i, qrow in enumerate(query_df.itertuples(index=False)):
                q_idx = np.where(body_ids == self._body_id(qrow.bodyId))[0]
                if len(q_idx):
                    query_pair_X[query_i] = X[q_idx[0]]
                    query_pair_mask[query_i] = True
            rows.extend(self._type_query_intra_rows(
                query_df, query_pair_X, query_pair_mask, intra,
                candidate_source=None,
            ))
        bodyid_df = self._bodyid_dataframe(rows, query_type=query_type)

        # --- type rows (per-type NBLAST means + intra reference) ---
        if self.level == "type":
            # Contralateral pair rows stay in results.csv (complete record)
            # but are excluded from the type means: NBLAST mirror scores are
            # unreliable (benchmark: chance-level), unlike vector_v2.
            type_df = self._aggregate_type_rows(
                self._drop_contra_nblast_rows(rows),
                query_type=query_type,
                intra=intra,
                query_type_count=(query_type_count or int(query_mask.sum())),
            )
            return bodyid_df, type_df

        import collections
        agg: Dict[str, List[float]] = collections.defaultdict(list)
        for bid, score in ranked:
            if bid in query_ids:
                continue
            t = id_to_type.get(bid, "")
            if t:
                agg[t].append(score)
        agg_rows = []
        for t, vals in agg.items():
            # Cap per-type pairs to the sampling cap for the mean.
            vals = sorted(vals, reverse=True)[: TYPE_MEMBER_SAMPLE_CAP]
            agg_rows.append({
                "target_type": t,
                "similarity": float(np.mean(vals)),
                "n_bodyids": len(vals),
                "is_intra_type": t == query_type,
                "intra_type_similarity": intra if t == query_type else float("nan"),
                "method": self.method,
                "metric": "nblast",
            })
        # All query-type members are the query itself and cannot be
        # candidates; inject the intra-type reference row from the vector
        # statistics so type queries always carry the intra data.
        if query_type and query_type not in agg and np.isfinite(intra):
            n_members = int(sum(1 for t in types if t == query_type))
            agg_rows.append({
                "target_type": query_type,
                "similarity": intra,
                "n_bodyids": n_members,
                "is_intra_type": True,
                "intra_type_similarity": intra,
                "method": self.method,
                "metric": "nblast",
            })
        agg_rows = sorted(agg_rows, key=lambda r: (not r["is_intra_type"], -r["similarity"], r["target_type"]))
        type_df = pd.DataFrame(agg_rows).reset_index(drop=True)
        type_df.insert(0, "rank", np.arange(1, len(type_df) + 1))
        return bodyid_df, type_df

    def _load_fafb_skeletons(self, body_ids: List[int],
                             check_extrusions: Optional[bool] = None
                             ) -> Dict[int, object]:
        """Load FAFB skeletons through the shared FlyWire pipeline.

        Priority: previously repaired trees from the local repair caches
        (``cave_skeletons`` / ``extrusion_fixes``, status-driven) -> shared
        raw ``.swc.zst`` cache (legacy frozen reads) -> healed zip ->
        one-time extrusion check with cached results. Flagged bodies are
        replaced through the token-gated CAVE skeletonization path (cached
        in ``cave_skeletons``) or, when CAVE is unavailable, a locally
        pruned fix persisted to ``extrusion_fixes``; missing bodies follow
        the same cache-first CAVE path. Every returned neuron is a
        TreeNeuron; the prepared mesh cache is never consulted.

        Kept as a method so callers and tests can override the seam.
        """
        check = self.check_extrusions if check_extrusions is None \
            else bool(check_extrusions)
        return load_local_release_skeletons(
            self.dataset, body_ids, project_root=str(self.project_root),
            log=self._log, check_extrusions=check)

    def _dotprops_for_ids(self, body_ids: List[int],
                          neurons: Optional[Dict[int, "navis.TreeNeuron"]] = None,
                          desc: str = "Building dotprops"
                          ) -> Dict[int, Optional["navis.core.dotprop.Dotprops"]]:
        """Load skeletons and build dotprops in microns.

        ``neurons`` supplies transient in-memory raw skeletons (profile-first
        fetches) so they are not re-fetched; anything missing is resolved
        through the shared raw cache and batched online fetch (raw skeletons
        are always persisted as compressed SWC). FAFB resolves its misses
        through the shared FAFB pipeline in ``load_local_release_skeletons``
        (repair caches -> raw cache / healed zip -> one-time extrusion check
        with CAVE replacement or local fix) and the loader never mirrors
        trees into ``raw_skeletons``; BANC and NeuPrint datasets fetch their
        misses through ``fetch_skeletons_on_demand_batch`` (BANC resolves
        each body through its public SWC chain)."""
        out: Dict[int, Optional[navis.core.dotprop.Dotprops]] = {}
        local_neurons: Dict[int, object] = {
            int(bid): neuron for bid, neuron in (neurons or {}).items()
            if _neuron_rep(neuron) == "skeleton"
        }

        raw_cache = find_similar_raw_cache(
            self.dataset, project_root=str(self.project_root),
            n_workers=self.n_workers, verbose=False,
        )

        # Keep raw vector rows and raw skeleton persistence in the same cache
        # transaction. This makes an NBLAST-first run useful to a later
        # vector-mode or visualization run. FAFB/BANC local-release datasets are
        # excluded: FAFB's shared loader persists at the stored level (a
        # second write at a different simplification level would churn the
        # same files), and BANC's batch fetch persists its own level-0 raw
        # entries inside fetch_banc_swc.
        def _cache_raw_neurons(mapping: Dict[int, object]) -> None:
            if self._is_flywire():
                return
            # Raw skeleton persistence only: V1 vector rows are retired, and
            # the V2 schema is always computed by the caller's cache.
            try:
                raw_cache.persist_skeletons(mapping)
            except Exception:
                pass

        # The shared raw cache has priority for every dataset.
        for body_id in body_ids:
            bid = int(body_id)
            if bid in local_neurons:
                continue
            cached = raw_cache.load_skeleton(bid)
            if cached is not None:
                local_neurons[bid] = cached

        # FAFB skeletons: resolve every non-transient id through the FAFB
        # pipeline once for the whole batch (the extrusion check is
        # cached, so repeated batches reuse earlier results). BANC must
        # NOT take this branch — it has no CAVE token, so the pipeline's
        # fallback would return nothing for it — and follows the NeuPrint
        # batch fetch below instead (mirrors the v2 missing-fetch gate).
        if is_fafb_dataset(self.dataset):
            fetched = self._load_fafb_skeletons(
                [int(b) for b in body_ids if int(b) not in local_neurons]
            )
            local_neurons.update(fetched)
        else:
            # Resolve raw-cache hits first, then issue one combined NeuPrint
            # fetch for all remaining dotprops. This covers cache-direct
            # NBLAST, where no profile-first in-memory map is available.
            missing_online = []
            for body_id in body_ids:
                bid = int(body_id)
                if bid in local_neurons:
                    continue
                missing_online.append(bid)
            if missing_online:
                fetched = fetch_skeletons_on_demand_batch(
                    self.dataset,
                    missing_online,
                    project_root=str(self.project_root),
                    persist=self.cache_fetched_skeletons,
                    level=VECTOR_BASIS_RAW,
                    raw_cache=raw_cache,
                    vector_cache=raw_cache,
                )
                local_neurons.update(fetched)
                # Compatibility with singular-fetch overrides that predate
                # the raw_cache/vector_cache keywords.
                _cache_raw_neurons(fetched)

        _cache_raw_neurons({
            int(b): n for b, n in local_neurons.items()
            if int(b) in {int(x) for x in body_ids}
        })

        pbar = tqdm(body_ids, desc=desc, unit="neuron",
                    disable=not self.verbose, leave=False, file=sys.stdout)
        try:
            for bid in pbar:
                bid = int(bid)
                pbar.set_postfix_str(f"{bid}")
                nrn = local_neurons.get(bid)
                if nrn is None:
                    # The FlyWire pipeline (repair caches / raw cache /
                    # healed zip -> extrusion check -> CAVE) already ran
                    # for this id; there is no other source to try.
                    out[bid] = None
                    continue
                try:
                    if _neuron_rep(nrn) != "skeleton":
                        out[bid] = None
                        continue
                    nrn_um = nrn / 1000.0  # nanometres -> microns (NBLAST requirement)
                    out[bid] = navis.make_dotprops(nrn_um, k=20)
                except Exception:
                    out[bid] = None
        finally:
            pbar.close()
        # Canonical keys: consumers look dotprops up by dataset-canonical id
        # (string bodyIds on FlyWire), not the raw-cache int keys — an int
        # key silently misses EVERY lookup there.
        return {self._body_id(b): dp for b, dp in out.items()}

    # ------------------------------------------------------------------ save
    def _save_results(self, results: pd.DataFrame, bodyid_df: pd.DataFrame,
                      type_df: pd.DataFrame, query_df: pd.DataFrame):
        """Save the run outputs: results.csv always holds the bodyId-level
        rows and type_summary.csv the type-level rows, whatever the query
        kind (mirrors the homolog finding outputs). ``results`` is the
        primary frame (type-to-type for type queries, bodyId-to-bodyId
        otherwise) recorded in the README."""
        query_label = str(self.query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.saveas or f"similar-morphology_{_dataset_folder(self.dataset)}_{query_label[:40]}_{timestamp}"
        run_dir = Path(self.output_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        bodyid_df.to_csv(run_dir / "results.csv", index=False)
        type_df.to_csv(run_dir / "type_summary.csv", index=False)
        params = {
            "query": str(self.query),
            "dataset": self.dataset,
            "level": self.level,
            "method": self.method,
            "metric": self.metric,
            **({"v2_block_weights": dict(self._v2_weights),
                "expand_top_types": self.expand_top_types,
                "expand_per_type": self.expand_per_type,
                "coverage_weight": "sqrt",
                "type_agg": self.type_agg}
               if self._is_v2 else {}),
            **({"nblast_applied": self.nblast_applied}
               if self.method == "nblast" else {}),
            "candidate_source": getattr(self, "resolved_candidate_source",
                                        self.candidate_source),
            "candidate_cap": self.candidate_cap,
            "note": ("results.csv/type_summary.csv contain EVERY compared "
                     "candidate (never truncated; bounded by candidate_cap); "
                     "visualize_top_n applies to visualization only"),
            "visualize_top_n": self.visualize_top_n,
            "visualize_by": self.visualize_by,
            "cache_raw_skeletons": self.cache_fetched_skeletons,
            "raw_skeleton_cache": str(
                find_similar_dataset_cache_v2(
                    self.dataset, project_root=str(self.project_root),
                    verbose=False,
                ).skeleton_dir
            ),
            "raw_vector_cache": str(
                find_similar_dataset_cache_v2(
                    self.dataset, project_root=str(self.project_root),
                    verbose=False,
                ).parquet_path
            ),
            "query_bodyIds": (
                self._body_ids(query_df["bodyId"].tolist())
                if self._is_flywire()
                else [int(b) for b in query_df["bodyId"].tolist()]
            ),
            "n_bodyid_rows": len(bodyid_df),
            "n_type_rows": len(type_df),
            "primary_level": self.level,
            "output_folder": str(run_dir),
        }
        readme = [f"DROCAT Find Similar Neurons (morphological)",
                  f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                  ""]
        for k, v in params.items():
            readme.append(f"{k}: {v}")
        (run_dir / "README.txt").write_text("\n".join(readme), encoding="utf-8")
        # The run-folder marker line must carry the path ONLY (the UI parses
        # the folder by splitting after the marker and checking isdir).
        self._log(f"Results saved to: {run_dir}")
        self._log(f"Saved {len(bodyid_df)} bodyId rows -> results.csv, "
                  f"{len(type_df)} type rows -> type_summary.csv")
        self.output_folder = str(run_dir)

    # ------------------------------------------------------------------ viz
    def _visualize_top_results(
        self,
        results: pd.DataFrame,
        query_df: Optional[pd.DataFrame] = None,
    ):
        """Render the query plus the top-N result neurons/types.

        The query is always the first visualization layer and does not consume
        the requested top-N result count. With ``visualize_by='type'``
        (default) each of the top-N distinct result types becomes one layer
        containing its member bodyIds (the result rows for bodyId-level
        searches, or the vector-cache members capped at ``n_per_type`` for
        type-level searches); with ``'bodyId'`` each top result row is one
        layer. The intra-type reference row is never rendered. Output goes to
        the same run folder (plot-3d_{dataset}/subfolder); a visualization
        failure is logged but never fails the similarity search.

        ``query_df`` is retained for compatibility with callers from older
        releases. It supplies the query bodyIds for the reference layer and is
        also used to remove query bodyIds if they are present in a supplied
        result frame.
        """
        if self.visualize_top_n <= 0:
            return
        if results is None:
            work = pd.DataFrame()
        else:
            work = results.copy()
        if work.empty and (query_df is None or query_df.empty):
            return
        VisualizeSkeleton = _import_visualizer()
        if VisualizeSkeleton is None:
            self._log("Visualization skipped: visualize_skeleton not available.")
            return

        if "is_intra_type" in work.columns:
            work = work[work["is_intra_type"] != True]  # noqa: E712
        # Unscorable candidates (NaN similarity) are never rendered.
        if "similarity" in work.columns:
            work = work[work["similarity"].notna()]

        layers: List[List[Union[int, str]]] = []
        names: List[str] = []

        def _body_ids(frame: pd.DataFrame) -> List[Union[int, str]]:
            values = frame.get("bodyId", pd.Series(dtype=object)).tolist()
            ids: List[Union[int, str]] = []
            for value in values:
                try:
                    if pd.isna(value):
                        continue
                    ids.append(self._body_id(value))
                except (TypeError, ValueError):
                    continue
            return list(dict.fromkeys(ids))

        def _body_id_in(value, ids: set) -> bool:
            try:
                return not pd.isna(value) and self._body_id(value) in ids
            except (TypeError, ValueError):
                return False

        query_body_ids = set()
        if query_df is not None and not query_df.empty:
            query_body_ids = set(_body_ids(query_df))

        # Keep the query visible as a reference layer. It is intentionally
        # added before selecting result layers so the first color and legend
        # entry identify the neuron that was searched.
        if query_body_ids:
            query_label = str(self.query)
            safe_query = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in query_label
            ).strip("_") or "neuron"
            query_members = list(dict.fromkeys(_body_ids(query_df)))
            layers.append(query_members)
            names.append(f"query_{safe_query}_x{len(query_members)}")

        # Be defensive if a caller supplies a frame that still contains the
        # query rows. Filtering before the top-N selection is what makes the
        # requested count mean "top N results", not "top N rows including the
        # query".
        if query_body_ids and "target_bodyId" in work.columns:
            work = work.loc[
                ~work["target_bodyId"].map(
                    lambda value: _body_id_in(value, query_body_ids)
                )
            ]
        if work.empty and not layers:
            self._log("Visualization skipped: no renderable query or results.")
            return

        capped_notes: List[str] = []
        if self.visualize_by == "type":
            seen: set = set()
            rank = 0
            for _, row in work.iterrows():
                t = str(row.get("target_type", "") or "")
                if not t or t in seen:
                    continue
                if self.level == "type" or "target_bodyId" not in work.columns:
                    # Type-level results carry no bodyIds: resolve the type's
                    # members from the vector cache (bounded to the render cap).
                    members, total_members = self._type_members_from_cache(t)
                else:
                    members = _body_ids(work.loc[
                        work["target_type"] == t
                    ].rename(columns={"target_bodyId": "bodyId"}))
                    total_members = len(members)
                members = [bid for bid in members if bid not in query_body_ids]
                if not members:
                    continue
                seen.add(t)
                rank += 1
                layers.append(members)
                names.append(f"r{rank}_{t}_x{len(members)}")
                if total_members > len(members):
                    capped_notes.append(
                        f"r{rank}_{t}: showing {len(members)} of "
                        f"{total_members} members of type '{t}' "
                        f"(per-layer render cap {TYPE_RENDER_MEMBER_CAP})")
                if rank >= self.visualize_top_n:
                    break
        else:
            rank = 0
            for _, row in work.iterrows():
                if rank >= self.visualize_top_n:
                    break
                try:
                    bid = self._body_id(row.get("target_bodyId"))
                except (TypeError, ValueError):
                    continue
                if bid in query_body_ids:
                    continue
                t = str(row.get("target_type", "") or "")
                rank += 1
                layers.append([bid])
                names.append(f"r{rank}_{t or f'unknown_{bid}'}_{bid}")

        if not layers:
            self._log("Visualization skipped: no renderable layers.")
            return

        self._log(
            f"3D visualization: including query + {max(0, len(layers) - 1)} "
            "result layer(s)"
        )

        run_dir = Path(getattr(self, "output_folder", "") or self.output_dir)
        try:
            pipeline = str(
                self.visualization_settings.get(
                    "neuprint_skeleton_pipeline", "fine"
                ) or "fine"
            ).strip().lower()
            viz_kwargs = {
                "dataset": self.dataset,
                "output_dir": str(run_dir),
                "neuron_layers": layers,
                "custom_layer_names": names,
                "saveas": _dataset_folder(self.dataset),
                "include_timestamp": False,
                "skip_synapse": True,
                # Analysis visualizations default to the light-weight line
                # representation. Callers can explicitly request tube/fine
                # rendering through visualization_settings.
                "skeleton_mode": self.visualization_settings.get(
                    "skeleton_mode", "line"
                ),
                "legend_mode": "layer" if self.visualize_by == "type" else "single",
                "brain_mesh": "native",
                "export_views": False,
                "show_fig": False,
                "cache_neurons": (
                    True if is_local_connectome_dataset(self.dataset)
                    else pipeline not in {
                        "fast", "direct", "artistic", "fine_opt1"
                    }
                ),
                "verbose": "simple",
                # Per-layer truncation notes for the renderer's in-page
                # banner (type layers showing a capped member sample).
                "layer_sample_notes": capped_notes or None,
            }
            # The panel contains the same keyword names as VisualizeSkeleton.
            # Ranking controls belong to MorphologyComparer, not the renderer.
            for key, value in self.visualization_settings.items():
                if key in {"visualize_top_n", "visualize_by", "use_default_simplification"}:
                    continue
                if key == "mesh_color" and value == "auto":
                    continue
                if key == "legend_mode":
                    # Derived from ``visualize_by`` above. Letting the shared
                    # panel's global preference (app default 'type') rewrite
                    # it collapsed per-bodyId legends into bare type names,
                    # exactly like the Find Homologs regression. The
                    # interactive 'tree' preference IS let through: its panel
                    # re-groups by the custom layer names.
                    if value != "tree":
                        continue
                viz_kwargs[key] = value

            # The shared panel returns None when its analysis default is
            # selected. Resolve that default at the analysis boundary so the
            # dedicated Skeleton tab can retain its own historical defaults.
            if viz_kwargs.get("skeleton_mesh_simplification") is None:
                viz_kwargs["skeleton_mesh_simplification"] = (
                    default_analysis_skeleton_mesh_simplification(
                        self.dataset, pipeline
                    )
                )

            vs = VisualizeSkeleton(
                **viz_kwargs,
            )
            vs.plot_neurons()
            viz_dir = run_dir / f"plot-3d_{_dataset_folder(self.dataset)}"
            self._log(f"3D visualization saved to: {viz_dir}")
        except Exception as ex:
            self._log(f"3D visualization failed (search results kept): {ex}")

    def _type_members_from_cache(self, type_name: str
                                 ) -> Tuple[List[Union[int, str]], int]:
        """Member bodyIds of a type for a 3D visualization layer, plus the
        total member count found before capping.

        Members come from the vector cache, then are topped up from the
        full neuron-table type map: the vector cache only holds ids that
        were vectorized at some point, so a type whose results report full
        coverage could still resolve to fewer members than it has (the
        render must not disagree with ``type_summary.csv``). Cached ids
        come first, neuron-table ids follow, and the list is capped to
        ``TYPE_RENDER_MEMBER_CAP`` so type-level renders stay bounded.
        Returning ``(members, total)`` lets the caller warn when the render
        shows only a sample of the type.
        """
        try:
            data = find_similar_dataset_cache_v2(
                self.dataset, project_root=str(self.project_root),
                n_workers=self.n_workers, verbose=False,
            ).load()
            members: List[Union[int, str]] = []
            if data is not None:
                members = [self._body_id(b)
                           for b, t in zip(data["bodyIds"], data["types"])
                           if t == type_name]
            # Top up from the neuron table / index (memoized per run).
            type_maps = getattr(self, "_type_members_maps", None)
            if type_maps is None:
                type_maps = _load_neuron_type_map(
                    self.dataset, str(self.project_root)
                )
                self._type_members_maps = type_maps
            type_map = type_maps[0]
            seen = set(members)
            for b, t in type_map.items():
                if t != type_name:
                    continue
                bid = self._body_id(b)
                if bid not in seen:
                    seen.add(bid)
                    members.append(bid)
            total = len(members)
            return members[: TYPE_RENDER_MEMBER_CAP], total
        except Exception:
            return [], 0


# =============================================================================
# Homolog results enrichment (post-search, vector-based)
# =============================================================================

RENDER_ARTIFACTS_SAMPLE = 200
_RENDER_V2_ARTIFACTS: Dict[Tuple[str, str], dict] = {}


def _dataset_soma_side_map(dataset: str,
                           project_root: Optional[str] = None
                           ) -> Dict[int, str]:
    """bodyId -> 'left' | 'right' | '' from the dataset's neuron index.

    Reads ``somaSide`` (NeuPrint) or ``hemisphere`` (FlyWire) once per call
    and normalizes both vocabularies. Midline and unknown values map to ''
    so those neurons are never side-filtered. Best-effort: any read failure
    means no side information.
    """
    sides: Dict[int, str] = {}
    try:
        import polars as pl
        p = (Path(project_root if project_root
                  else Path(__file__).resolve().parents[1])
             / "neuron_indexes" / _dataset_folder(dataset)
             / "neuron_index.parquet")
        if p.exists():
            schema = pl.read_parquet_schema(p)
            col = ("somaSide" if "somaSide" in schema
                   else "hemisphere" if "hemisphere" in schema else "")
            if col:
                df = pl.read_parquet(p, columns=["bodyId", col])
                norm = {"l": "left", "left": "left",
                        "r": "right", "right": "right"}
                for bid, s in zip(df["bodyId"].to_list(), df[col].to_list()):
                    try:
                        sides[int(bid)] = norm.get(
                            str(s or "").strip().lower(), "")
                    except (TypeError, ValueError):
                        continue
    except Exception:
        sides = {}
    return sides


def render_v2_artifacts(dataset: str, project_root: Optional[str] = None,
                        verbose: bool = False) -> Optional[dict]:
    """Population vector_v2 artifacts (bounds, μ/σ, ZCA whitener) for the
    dataset's RENDER space.

    The homolog enrichment scores transformed overlay neurons in the target
    scene's render space. The native cache artifacts (``meta_v2.json``) do
    not describe that frame: the transform rescales coordinates, so
    histograms binned against the native bbox clip into boundary bins and
    the lateral-normalization midline is wrong by the transform offset.
    This derives the render frame ONCE per dataset — transform a sample of
    the cached native population through the same
    ``transform_neurons_to_space`` pipeline, take the transformed point
    bbox (+2 % padding) as bounds, vectorize the transformed neurons
    against those bounds, and fit μ/σ + truncated ZCA whitener on the
    sample — then persists it in ``meta_v2_render.json`` /
    ``whiten_v2_render.npz`` beside the native cache artifacts.

    Returns ``{bounds, mean, std, whiten, space}`` or ``None`` when the
    dataset has no render transform (native == render — the native
    artifacts already describe the frame) or no usable cached skeletons.
    Memoized per process; the sidecar makes restarts free.
    """
    root = Path(project_root) if project_root \
        else Path(__file__).resolve().parents[1]
    try:
        from visualize_skeleton import (dataset_native_space,
                                        dataset_render_space)
        native_space = dataset_native_space(dataset)
        render_space = dataset_render_space(dataset)
    except Exception:
        return None
    if native_space == render_space:
        return None   # identity transform: the native artifacts ARE the frame
    key = (dataset, render_space, str(Path(root).resolve()))
    if key in _RENDER_V2_ARTIFACTS:
        return _RENDER_V2_ARTIFACTS[key]

    morph_dir = (Path(root) / "cache" / _dataset_folder(dataset)
                 / "find_similar" / "morphology")
    meta_path = morph_dir / "meta_v2_render.json"
    whiten_path = morph_dir / "whiten_v2_render.npz"

    def _load_sidecar() -> Optional[dict]:
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        if meta.get("render_space") != render_space:
            return None
        if int(meta.get("v2_cache_version", -1)) != VECTOR_CACHE_V2_VERSION:
            # Artifacts fit on a population vectorized at a different cache
            # basis (e.g. pre-raw-basis simp90 rows) are stale.
            return None
        bounds = (np.asarray(meta.get("spatial_bounds"), dtype=float)
                  if meta.get("spatial_bounds") is not None else None)
        mean = (np.asarray(meta.get("mean"), dtype=float)
                if meta.get("mean") is not None else None)
        std = (np.asarray(meta.get("std"), dtype=float)
               if meta.get("std") is not None else None)
        W = None
        if whiten_path.exists():
            try:
                z = np.load(whiten_path)
                if int(z["fit_version"]) == WHITEN_FIT_VERSION:
                    W = np.asarray(z["W"], dtype=float)
            except Exception:
                W = None
        if W is None or W.shape != (VECTOR_V2_DIM, VECTOR_V2_DIM):
            W = np.eye(VECTOR_V2_DIM)
        if (bounds is None or bounds.shape != (2, 3) or mean is None
                or std is None or mean.shape != (VECTOR_V2_DIM,)):
            return None
        return {"bounds": bounds, "mean": mean, "std": std, "whiten": W,
                "space": render_space}

    artifacts = _load_sidecar()
    if artifacts is None:
        # Derive: transform a sample of the cached native population.
        try:
            cache_v2 = find_similar_dataset_cache_v2(
                dataset, project_root=str(root), verbose=False)
            files = cache_v2._discover_skeleton_files()
        except Exception:
            return None
        from visualize_skeleton import transform_neurons_to_space
        transformed = []
        if not files and is_fafb_dataset(dataset):
            # FAFB raw store no longer holds bundle skeletons: sample the
            # healed zip and resolve through the loader instead.
            try:
                bundle = _fafb_bundle(dataset, str(root))
                zip_ids = sorted(int(b) for b in bundle.ids()) \
                    if bundle is not None else []
                if bundle is not None:
                    bundle.close()
            except Exception:
                zip_ids = []
            if not zip_ids:
                return None
            rng = np.random.default_rng(0)
            if len(zip_ids) > RENDER_ARTIFACTS_SAMPLE:
                zip_ids = [zip_ids[i] for i in rng.choice(
                    len(zip_ids), RENDER_ARTIFACTS_SAMPLE, replace=False)]
            trees = _resolve_fafb_skeleton_trees(
                dataset, zip_ids, project_root=root)
            for neuron in trees.values():
                try:
                    xf = transform_neurons_to_space(
                        [neuron], native_space, render_space,
                        validate_bounds=False)
                    if xf and xf[0] is not None:
                        transformed.append(xf[0])
                except Exception:
                    continue
        elif files:
            if len(files) > RENDER_ARTIFACTS_SAMPLE:
                rng = np.random.default_rng(0)
                files = [files[i] for i in
                         rng.choice(len(files), RENDER_ARTIFACTS_SAMPLE,
                                    replace=False)]
            for path in files:
                try:
                    neuron = _load_cached_skeleton_file(path)
                    if neuron is None:
                        continue
                    xf = transform_neurons_to_space(
                        [neuron], native_space, render_space,
                        validate_bounds=False)
                    if xf and xf[0] is not None:
                        transformed.append(xf[0])
                except Exception:
                    continue
        else:
            return None
        if len(transformed) < 2:
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for neuron in transformed:
            pts = _neuron_points(neuron)
            if len(pts):
                lo = np.minimum(lo, pts.min(axis=0))
                hi = np.maximum(hi, pts.max(axis=0))
        if not np.isfinite(lo).all():
            return None
        pad = 0.02 * np.maximum(hi - lo, 0.0)
        bounds = np.vstack([lo - pad, hi + pad])
        rows = []
        for neuron in transformed:
            try:
                _, vec = vectorize_neuron_v2(
                    neuron, spatial_bounds=bounds, lateral_normalize=True)
                rows.append(np.asarray(vec, dtype=float))
            except Exception:
                continue
        if len(rows) < 2:
            return None
        X = np.vstack(rows)
        mean = X.mean(axis=0)
        std = np.where(X.std(axis=0) <= 0, 1.0, X.std(axis=0))
        X_std = (X - mean) / std
        if len(X_std) >= MIN_ROWS_FOR_WHITENING:
            W = fit_zca_whitener(X_std)
        else:
            W = np.eye(VECTOR_V2_DIM)
        artifacts = {"bounds": bounds, "mean": mean, "std": std,
                     "whiten": W, "space": render_space}
        if verbose:
            print(f"[morphology] render-space artifacts derived for "
                  f"{dataset} ({render_space}; {len(X)} sampled neurons)",
                  flush=True)
        try:
            morph_dir.mkdir(parents=True, exist_ok=True)
            meta_out = {
                "render_space": render_space,
                "native_space": native_space,
                "v2_cache_version": VECTOR_CACHE_V2_VERSION,
                "sample_rows": int(len(X)),
                "spatial_bounds": bounds.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
            with open(whiten_path, "wb") as fh:
                np.savez(fh, W=W, fit_version=WHITEN_FIT_VERSION)
        except Exception:
            pass
    _RENDER_V2_ARTIFACTS[key] = artifacts
    if verbose:
        print(f"[morphology] render-space artifacts ready for {dataset} "
              f"({render_space})", flush=True)
    return artifacts


def compute_morph_similarity_vs_queries(
    query_neurons,
    target_bids,
    target_dataset: str,
    project_root: Optional[str] = None,
    compute_nblast: bool = True,
    verbose: bool = False,
    query_bids: Optional[List[int]] = None,
    source_dataset: Optional[str] = None,
    target_vector_cache: Optional[Dict[int, object]] = None,
    target_side_cache: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Pairwise morphological similarity of transformed query neurons vs
    target neurons, in the target scene's render space.

    Both sides are compared in the SAME frame: the query neurons arrive
    already transformed (``query_transformed_*`` overlay neurons), and the
    target raw skeletons are transformed from their native space into the
    render space here (e.g. male-cns raw voxels -> JRCFIB2022M).

    Scores per (source_bodyId, target_bodyId) pair:
      - ``morph_v2_similarity``: THE production vector_v2 score — per-block
        cosine (shape/spatial, ``DEFAULT_V2_BLOCK_WEIGHTS``) on the
        standardized + ZCA-whitened 256-dim features, identical to the
        Find Similar cache-direct scorer (spatial bounds from the target
        dataset's cache meta, lateral normalization on). Replaces the
        former raw ``morph_v2_cosine`` / ``morph_v2_pearson`` columns,
        whose full-vector cosine/Pearson on unstandardized features was
        magnitude-dominated and incomparably inflated.
      - ``morph_nblast``: forward, normalized NBLAST (navis NBlaster, k=20
        dotprops in microns).

    Failure-isolated per neuron: unavailable skeletons yield NaN; no
    exception escapes.

    ``target_vector_cache`` optionally maps bodyId -> raw 256-dim feature
    vector previously computed in THIS target frame (same bounds /
    lateral normalization). Hits skip the target re-vectorization and
    freshly computed vectors are written back, letting callers (the
    cross-dataset comparison / homolog morph qualification) persist the
    null-sample vectors across runs. Vectors must describe the same frame
    this call derives; the caller owns frame-keyed invalidation.

    ``target_side_cache`` maps bodyId -> 'left' / 'right' for the same frame
    and makes a cache hit complete: with a stored vector but no answerable
    side, the geometric fallback still has to RENDER the skeleton, so the
    skip is only taken when the side is known (from the neuron index or this
    cache). That keeps `pair_side` byte-identical either way, and it is
    written back for every neuron this call renders.
    """
    import navis

    rows = []
    if not query_neurons or not target_bids:
        return pd.DataFrame()

    root = project_root or str(Path(__file__).resolve().parents[1])
    try:
        from visualize_skeleton import (
            dataset_native_space, dataset_render_space,
            transform_neurons_to_space,
        )
        have_xform = True
    except Exception:
        have_xform = False

    # Unknown datasets (test fixtures, custom sets) -> identity transform.
    try:
        native_space = dataset_native_space(target_dataset)
        render_space = dataset_render_space(target_dataset)
    except ValueError:
        native_space = render_space = ''

    raw_target_cache = None
    fafb_target_trees: Dict[int, object] = {}
    if is_fafb_dataset(target_dataset):
        # FAFB skeletons are served from the healed zip / repair caches, not
        # the (frozen) raw store; resolve the whole target set once.
        fafb_target_trees = _resolve_fafb_skeleton_trees(
            target_dataset, list(target_bids), project_root=root)
    else:
        try:
            raw_target_cache = find_similar_raw_cache(
                target_dataset, project_root=root, verbose=False)
        except Exception:
            raw_target_cache = None

    def _render_target(bid: int):
        if is_fafb_dataset(target_dataset):
            n = fafb_target_trees.get(int(bid))
        elif raw_target_cache is None:
            return None
        else:
            n = raw_target_cache.load_skeleton(int(bid))
        if n is None:
            return None
        if have_xform and native_space != render_space:
            try:
                xf = transform_neurons_to_space(
                    [n], native_space, render_space, validate_bounds=False)
                return xf[0] if xf else None
            except Exception:
                return None
        return n

    from navis.nbl.nblast_funcs import NBlaster
    nb = NBlaster(use_alpha=False, normalized=True, progress=False)

    # Ipsilateral/contralateral classification: soma (or root) x relative
    # to the render template's midline (x-mid of the template bounding box).
    space_midline = {
        'FLYWIRE': (192200.0 + 853686.0) / 2.0,
        'JRCFIB2022M': 752704.0 / 2.0,
        'JRCFIB2018F': 275456.0 / 2.0,
        'MANC': (49184.0 + 342752.0) / 2.0,
    }
    midline_x = space_midline.get(render_space)

    def _soma_x(n) -> Optional[float]:
        # soma may be a Somaholder, a node-id ndarray, or missing entirely
        soma = getattr(n, 'soma', None)
        if soma is not None:
            try:
                return float(soma.x)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                rid = int(np.asarray(soma).ravel()[0])
                row = n.nodes[n.nodes.node_id == rid]
                if len(row):
                    return float(row['x'].iloc[0])
            except Exception:
                pass
        try:  # root node position as last resort
            rid = n.root[0]
            row = n.nodes[n.nodes.node_id == rid]
            return float(row['x'].iloc[0])
        except Exception:
            return None

    def _side(n) -> str:
        x = _soma_x(n)
        if x is None:
            return 'unknown'
        mid = midline_x
        if mid is None:  # unknown space: mid of the population x envelope
            xs = [_soma_x(o) for o in list(query_neurons) ]
            xs = [v for v in xs if v is not None]
            mid = (min(xs) + max(xs)) / 2.0 if xs else None
        if mid is None:
            return 'unknown'
        return 'right' if x >= mid else 'left'

    q_vecs = {}
    q_dps = {}
    q_sides: Dict[int, str] = {}

    # Key pairs by the query's SOURCE bodyId (explicit list preferred; the
    # cached SWC round-trip rewrites neuron .id, so .id is only a fallback).
    if query_bids and len(query_bids) == len(query_neurons):
        qids = [int(b) for b in query_bids]
    else:
        qids = []
        for q in query_neurons:
            try:
                qids.append(int(getattr(q, 'id', None)
                                or getattr(q, 'bodyId', None) or 0))
            except (TypeError, ValueError):
                qids.append(0)

    # Side classification: the neuron index (somaSide / hemisphere) is the
    # primary source of truth; the geometric render midline is only the
    # fallback for neurons the index does not list.
    side_maps: Dict[str, Dict[int, str]] = {}

    def _index_side(ds: str, bid) -> str:
        try:
            key = int(bid)
        except (TypeError, ValueError):
            return ''
        m = side_maps.get(ds)
        if m is None:
            m = side_maps[ds] = _dataset_soma_side_map(ds, str(root))
        # the shared reader already returns 'left' / 'right' / ''
        return m.get(key, '')

    q_vecs = {}
    q_dps = {}
    # Production scoring context. The scored neurons live in the target
    # scene's render space, so the population artifacts (bounds, μ/σ, ZCA
    # whitener) must describe THAT frame: cross-space runs use dedicated
    # render-space artifacts derived from a transformed population sample;
    # identity spaces (native == render, or unknown datasets) use the
    # native cache artifacts directly — exact Find Similar parity.
    # ensure(fetch_missing=0): a cache whose version predates the current
    # vector semantics (e.g. the raw-basis flip) is rebuilt from the local
    # raw skeletons instead of silently degrading this comparison.
    native_cache_data = None
    try:
        _v2_cache = find_similar_dataset_cache_v2(
            target_dataset, project_root=root, verbose=False)
        native_cache_data = _v2_cache.load()
        if native_cache_data is None:
            # Stale (old basis) or absent: rebuild from the local raw
            # skeletons, then load. ensure()'s summary dict is never used
            # as cache data.
            _v2_cache.ensure(fetch_missing=0)
            native_cache_data = _v2_cache.load()
    except Exception:
        native_cache_data = None

    def _native_artifacts():
        meta = (native_cache_data or {}).get('meta') or {}
        bounds = meta.get('spatial_bounds')
        bounds = (np.asarray(bounds, dtype=float)
                  if isinstance(bounds, list) and len(bounds) == 2
                  else None)
        mean = np.asarray(meta.get('mean'), dtype=float) \
            if meta.get('mean') is not None else None
        std = np.asarray(meta.get('std'), dtype=float) \
            if meta.get('std') is not None else None
        whiten = (native_cache_data or {}).get('whiten')
        return bounds, mean, std, whiten

    same_frame = (not have_xform) or native_space == render_space
    if same_frame:
        v2_bounds, v2_mean, v2_std, v2_whiten = _native_artifacts()
    else:
        art = render_v2_artifacts(target_dataset, project_root=root,
                                  verbose=verbose)
        if art is not None:
            v2_bounds = art['bounds']
            v2_mean = art['mean']
            v2_std = art['std']
            v2_whiten = art['whiten']
        else:
            if verbose:
                print('[morphology] render-space artifacts unavailable; '
                      'scoring with native cache artifacts (degraded '
                      'spatial block).')
            v2_bounds, v2_mean, v2_std, v2_whiten = _native_artifacts()

    native_rows: Dict[int, int] = {}
    if native_cache_data is not None:
        try:
            native_rows = {int(b): i for i, b in
                           enumerate(native_cache_data['bodyIds'])}
        except (TypeError, ValueError):
            native_rows = {}

    def _v2_similarity(qv: np.ndarray, tv: np.ndarray) -> float:
        """Production vector_v2 score for two RAW 256-dim feature vectors."""
        X = np.vstack([np.asarray(qv, dtype=float),
                       np.asarray(tv, dtype=float)])
        if (v2_mean is not None and v2_std is not None
                and v2_mean.shape == (X.shape[1],)
                and v2_std.shape == (X.shape[1],)):
            X = (X - v2_mean) / np.where(v2_std <= 0, 1.0, v2_std)
            if v2_whiten is not None and v2_whiten.shape == (X.shape[1],) * 2:
                X = apply_whitening(v2_whiten, X)
        else:
            # No cache population stats (unknown dataset): plain block
            # cosine on the raw vectors — closest computable proxy; do NOT
            # center (a pooled z-score degenerates for identical pairs).
            pass
        return float(v2_similarity_matrix(
            X[0], X[1].reshape(1, -1), dict(DEFAULT_V2_BLOCK_WEIGHTS))[0])

    q_vecs = {}
    q_dps = {}
    for qbid, q in zip(qids, query_neurons):
        if qbid in q_vecs:
            continue
        try:
            _, vec = vectorize_neuron_v2(q, spatial_bounds=v2_bounds,
                                         lateral_normalize=True)
            q_vecs[qbid] = np.asarray(vec, dtype=float)
        except Exception:
            pass
        try:
            q_dps[qbid] = navis.make_dotprops(
                navis.NeuronList([q])[0] / 1000.0, k=20)
        except Exception:
            pass
        q_sides[qbid] = ((_index_side(source_dataset, qbid)
                          if source_dataset else '')
                         or _side(q))

    out_rows = []
    t_vecs = {}
    t_dps = {}
    for tb in target_bids:
        tb = int(tb)
        # A stored vector PLUS an answerable side means this target needs no
        # skeleton at all: load + transform + vectorize measured 0.412 s per
        # neuron, re-paid on every run before the store existed. The side is
        # part of the hit condition on purpose — without it the geometric
        # fallback needs the neuron, and skipping the render would turn a real
        # `pair_side` into 'unknown'. NBLAST needs the geometry either way, so
        # that caller never takes the skip.
        t_side = (_index_side(target_dataset, tb)
                  or (target_side_cache or {}).get(tb) or '')
        cached = None if compute_nblast else (target_vector_cache or {}).get(tb)
        usable_hit = (cached is not None and t_side
                      and np.asarray(cached).ndim == 1
                      and len(cached) == VECTOR_V2_DIM)
        tn = None
        if usable_hit:
            t_vecs[tb] = np.asarray(cached, dtype=float)
        else:
            try:
                tn = _render_target(tb)
            except Exception:
                tn = None
            if tn is None:
                continue
            t_side = t_side or _side(tn)
            if target_side_cache is not None and t_side:
                # every rendered neuron teaches the store a side, which is what
                # lets the NEXT run skip the render
                target_side_cache[tb] = t_side
        if compute_nblast and tb not in t_dps:
            try:
                t_dps[tb] = navis.make_dotprops(
                    navis.NeuronList([tn])[0] / 1000.0, k=20)
            except Exception:
                pass
        if tb not in t_vecs and target_vector_cache is not None:
            cached_vec = target_vector_cache.get(tb)
            if (cached_vec is not None
                    and np.asarray(cached_vec).ndim == 1
                    and len(cached_vec) == VECTOR_V2_DIM):
                t_vecs[tb] = np.asarray(cached_vec, dtype=float)
        if tb not in t_vecs:
            # Identity space: the cached raw row IS the native-space vector —
            # exact Find Similar parity without re-vectorizing.
            if (same_frame and native_cache_data is not None
                    and native_cache_data.get('raw') is not None):
                try:
                    row_i = native_rows.get(tb)
                    if row_i is not None:
                        t_vecs[tb] = np.asarray(
                            native_cache_data['raw'][row_i], dtype=float)
                except Exception:
                    pass
        if tb not in t_vecs:
            try:
                _, vec = vectorize_neuron_v2(tn, spatial_bounds=v2_bounds,
                                             lateral_normalize=True)
                t_vecs[tb] = np.asarray(vec, dtype=float)
                if target_vector_cache is not None:
                    target_vector_cache[tb] = t_vecs[tb]
            except Exception:
                pass
        for qbid, qv in q_vecs.items():
            tv = t_vecs.get(tb)
            q_side = q_sides.get(qbid, 'unknown')
            row = {'source_bodyId': qbid, 'target_bodyId': tb,
                   'query_side': q_sides.get(qbid, 'unknown'),
                   'target_side': t_side,
                   'pair_side': ('ipsi' if q_side == t_side and t_side != 'unknown'
                                 else 'contra' if t_side != 'unknown' and q_side != 'unknown'
                                 else 'unknown'),
                   'morph_v2_similarity': np.nan,
                   'morph_nblast': np.nan}
            if tv is not None and len(qv) == len(tv):
                try:
                    row['morph_v2_similarity'] = _v2_similarity(qv, tv)
                except Exception:
                    pass
            if compute_nblast:
                qi = q_dps.get(qbid)
                ti = t_dps.get(tb)
                if qi is not None and ti is not None:
                    try:
                        q_idx = nb.append(qi, self_hit=nb.calc_self_hit(qi))
                        t_idx = nb.append(ti, self_hit=nb.calc_self_hit(ti))
                        v = float(nb.single_query_target(q_idx, t_idx,
                                                         scores='forward'))
                        if np.isfinite(v):
                            row['morph_nblast'] = v
                    except Exception:
                        pass
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def enrich_homolog_results(
    results_df: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
    project_root: Optional[str] = None,
    verbose: bool = True,
    query_neurons=None,
    compute_nblast: bool = True,
) -> pd.DataFrame:
    """Attach morphological similarity to homolog results.

    Computes, per (source_bodyId, target_bodyId) row, the similarity of the
    TRANSFORMED query neurons (``query_transformed_*`` overlay neurons, in
    the target render space) against the target neuron:
    ``morph_v2_similarity`` (the production vector_v2 score — standardized +
    ZCA-whitened per-block cosine, identical to Find Similar) and
    ``morph_nblast`` (forward, normalized NBLAST). Replaces the removed
    ``morph_cosine`` / ``morph_pearson`` and raw ``morph_v2_cosine`` /
    ``morph_v2_pearson`` columns, which lacked a consistent cross-dataset
    frame and were magnitude-inflated. Runs only on the final ranked result
    rows; NaN where skeletons are unavailable; no rows are dropped or
    re-ranked.
    """
    if results_df is None or results_df.empty:
        return results_df
    needed = {"source_bodyId", "target_bodyId"}
    if not needed.issubset(results_df.columns):
        return results_df

    results_df = results_df.copy()

    # Resolve the transformed query neurons when not supplied.
    if not query_neurons:
        try:
            from visualize_skeleton import (
                dataset_native_space, dataset_render_space,
                transform_neurons_to_space,
            )
            src_ids = [int(b) for b in
                       results_df['source_bodyId'].dropna().unique().tolist()]
            if is_fafb_dataset(source_dataset):
                # FAFB query trees live in the zip / repair caches, not the
                # frozen raw store.
                _fafb_src = _resolve_fafb_skeleton_trees(
                    source_dataset, src_ids, project_root=project_root)
                raw_src = [_fafb_src[b] for b in src_ids if b in _fafb_src]
            else:
                raw_cache = find_similar_raw_cache(
                    source_dataset, project_root=project_root, verbose=False)
                raw_src = [raw_cache.load_skeleton(int(b)) for b in src_ids]
                raw_src = [n for n in raw_src if n is not None]
            try:
                s_space = dataset_native_space(source_dataset)
                t_space = dataset_render_space(target_dataset)
            except ValueError:
                s_space = t_space = ''
            query_neurons = (transform_neurons_to_space(
                raw_src, s_space, t_space, validate_bounds=False)
                if s_space != t_space else list(raw_src))
        except Exception:
            query_neurons = None
    if not query_neurons:
        # Fallback: no transformed query available (unknown/bridge-less
        # spaces). Use the per-dataset vector-cache vectors for both sides —
        # the old enrichment behavior — under the new column names. NBLAST
        # needs skeletons in one shared space, so it stays NaN here.
        if verbose:
            print('[morphology] Transformed query unavailable; using cached '
                  'dataset vectors.')
        try:
            whiten_cache = {}
            for ds in {source_dataset, target_dataset}:
                try:
                    wd = find_similar_dataset_cache_v2(
                        ds, project_root=project_root, verbose=False).load()
                    whiten_cache[ds] = (wd or {}).get('whiten')
                except Exception:
                    whiten_cache[ds] = None
            src_ids = [
                _canonical_dataset_body_id(source_dataset, b)
                for b in results_df['source_bodyId'].tolist()
            ]
            tgt_ids = [
                _canonical_dataset_body_id(target_dataset, b)
                for b in results_df['target_bodyId'].tolist()
            ]
            results_df['source_bodyId'] = src_ids
            results_df['target_bodyId'] = tgt_ids
            src_X, src_ok, _ = find_similar_dataset_cache_v2(
                source_dataset, project_root=project_root,
                verbose=verbose).vectors_for(src_ids, compute_missing=True)
            tgt_X, tgt_ok, _ = find_similar_dataset_cache_v2(
                target_dataset, project_root=project_root,
                verbose=verbose).vectors_for(tgt_ids, compute_missing=True)
            cos = np.full(len(results_df), np.nan)
            for i in range(len(results_df)):
                if not (src_ok[i] and tgt_ok[i]):
                    continue
                # Production vector_v2 score: whiten each side with its own
                # dataset's cached whitener, then per-block weighted cosine.
                sw = whiten_cache[source_dataset]
                tw = whiten_cache[target_dataset]
                try:
                    s_w = apply_whitening(sw, src_X[i].reshape(1, -1))[0]
                    t_w = apply_whitening(tw, tgt_X[i].reshape(1, -1))[0]
                    cos[i] = float(v2_similarity_matrix(
                        s_w, t_w.reshape(1, -1),
                        dict(DEFAULT_V2_BLOCK_WEIGHTS))[0])
                except Exception:
                    continue
            results_df['morph_v2_similarity'] = cos
            results_df['morph_nblast'] = np.nan
            return results_df
        except Exception:
            if verbose:
                print('[morphology] Vector fallback failed.')
        for col in ('morph_v2_similarity', 'morph_nblast'):
            results_df[col] = np.nan
        return results_df

    target_bids = [int(b) for b in
                   results_df['target_bodyId'].dropna().unique().tolist()]
    src_unique = [int(b) for b in
                  results_df['source_bodyId'].dropna().unique().tolist()]
    try:
        pair_df = compute_morph_similarity_vs_queries(
            query_neurons, target_bids, target_dataset,
            project_root=project_root, compute_nblast=compute_nblast,
            verbose=verbose, query_bids=src_unique,
            source_dataset=source_dataset)
    except Exception:
        if verbose:
            print('[morphology] Enrichment skipped (pair scoring failed).')
        for col in ('morph_v2_similarity', 'morph_nblast'):
            results_df[col] = np.nan
        return results_df

    if pair_df.empty:
        for col in ('morph_v2_similarity', 'morph_nblast'):
            results_df[col] = np.nan
        return results_df

    pair_key = {}
    for r in pair_df.itertuples():
        try:
            pair_key[(int(r.source_bodyId), int(r.target_bodyId))] = r
        except (TypeError, ValueError):
            continue

    def _pair_val(sb, tb, col):
        try:
            r = pair_key.get((int(sb), int(tb)))
            v = float(getattr(r, col)) if r is not None else np.nan
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    for col in ('morph_v2_similarity', 'morph_nblast'):
        results_df[col] = [
            _pair_val(sb, tb, col)
            for sb, tb in zip(results_df['source_bodyId'],
                              results_df['target_bodyId'])
        ]
    side_key = {(int(r.source_bodyId), int(r.target_bodyId)): r.pair_side
                for r in pair_df.itertuples()}
    results_df['pair_side'] = [
        side_key.get((int(sb), int(tb)))
        for sb, tb in zip(results_df['source_bodyId'],
                          results_df['target_bodyId'])
    ]
    return results_df
