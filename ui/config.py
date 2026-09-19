"""
DROCAT UI Configuration
Datasets, defaults, and path settings.
"""

from pathlib import Path
import json
import os
import re
import subprocess
from copy import deepcopy

# Project root (parent of ui/)
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_data"
TOKEN_FILE = PROJECT_ROOT / "config_local.json"
LOCAL_CONFIG_FILE = PROJECT_ROOT / "ui" / "local_config.json"
TAB_OUTPUT_DIRS_KEY = "tab_output_dirs"

# mtime-keyed cache behind :func:`load_local_config` (see its docstring).
_LOCAL_CONFIG_CACHE: dict = {"mtime_ns": None, "data": {}}

try:
    from src.utils.naming_utils import canonical_dataset_name
except ImportError:  # src not importable; legacy names stay as-is
    def canonical_dataset_name(value):
        return str(value or "").strip()


def load_local_config() -> dict:
    """Load the user-editable local UI configuration (output dir, etc.).

    The file is read once per mtime change, not per call: the config is
    consulted on every input keystroke (auto-suggest/history toggles) and
    hundreds of times per page build (one ``get_user_default`` per widget),
    and a synchronous disk read per call stalls the NiceGUI event loop.
    Callers receive a fresh copy, so mutating the result never poisons the
    cache.
    """
    try:
        if LOCAL_CONFIG_FILE.exists():
            mtime_ns = LOCAL_CONFIG_FILE.stat().st_mtime_ns
            if _LOCAL_CONFIG_CACHE["mtime_ns"] == mtime_ns:
                return deepcopy(_LOCAL_CONFIG_CACHE["data"])
            # utf-8-sig tolerates the UTF-8 BOM that Windows editors
            # prepend to saved JSON files.
            data = json.loads(LOCAL_CONFIG_FILE.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                _LOCAL_CONFIG_CACHE["mtime_ns"] = mtime_ns
                _LOCAL_CONFIG_CACHE["data"] = data
                return deepcopy(data)
    except (OSError, ValueError):
        pass
    return {}


def save_local_config(config: dict) -> bool:
    """Persist the local UI configuration (and refresh the load cache)."""
    try:
        LOCAL_CONFIG_FILE.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _LOCAL_CONFIG_CACHE["mtime_ns"] = LOCAL_CONFIG_FILE.stat().st_mtime_ns
        # Store a copy: the caller keeps owning the dict it passed in.
        _LOCAL_CONFIG_CACHE["data"] = deepcopy(config)
        return True
    except OSError:
        _LOCAL_CONFIG_CACHE["mtime_ns"] = None
        return False


def get_default_output_dir() -> str:
    """Resolve the effective default output directory (user override wins)."""
    override = load_local_config().get("default_output_dir")
    if override and Path(override).is_absolute():
        return override
    return str(DEFAULT_OUTPUT_DIR)


def _resolve_output_path(value: str) -> str:
    """Normalize a user-entered output path without touching the filesystem."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def set_default_output_dir(value: str, create: bool = True) -> tuple:
    """Persist the UI default output directory permanently (across sessions).

    Relative paths are resolved against PROJECT_ROOT; the directory is
    created when *create* is true. Empty values clear the override so the
    project default is used again.

    Returns (saved: bool, effective_path: Optional[str]).
    """
    raw = (value or "").strip()
    if not raw:
        config = load_local_config()
        config.pop("default_output_dir", None)
        saved = save_local_config(config)
        return saved, str(DEFAULT_OUTPUT_DIR) if saved else None
    path = Path(_resolve_output_path(raw))
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, None
    config = load_local_config()
    config["default_output_dir"] = str(path)
    saved = save_local_config(config)
    return saved, str(path) if saved else None


def get_tab_output_override(scope: str | None) -> str | None:
    """Return a persisted tab-specific output directory, if one exists."""
    key = str(scope or "").strip()
    if not key:
        return None
    overrides = load_local_config().get(TAB_OUTPUT_DIRS_KEY, {})
    if not isinstance(overrides, dict):
        return None
    value = overrides.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_absolute() else None


def has_tab_output_override(scope: str | None) -> bool:
    """Whether *scope* has its own path instead of inheriting the default."""
    return get_tab_output_override(scope) is not None


def get_tab_output_dir(scope: str | None) -> str:
    """Resolve a tab path, falling back to the global output default."""
    return get_tab_output_override(scope) or get_default_output_dir()


def set_tab_output_dir(
    scope: str | None,
    value: str,
    create: bool = False,
) -> tuple:
    """Persist or clear one tab's output-directory override.

    An empty value removes the override and makes the tab inherit the global
    default again.  Tab overrides never modify ``default_output_dir``.
    """
    key = str(scope or "").strip()
    if not key:
        return False, None
    raw = (value or "").strip()
    config = load_local_config()
    overrides = config.get(TAB_OUTPUT_DIRS_KEY, {})
    if not isinstance(overrides, dict):
        overrides = {}
    if not raw:
        overrides.pop(key, None)
        if overrides:
            config[TAB_OUTPUT_DIRS_KEY] = overrides
        else:
            config.pop(TAB_OUTPUT_DIRS_KEY, None)
        saved = save_local_config(config)
        return saved, get_default_output_dir() if saved else None

    path = Path(_resolve_output_path(raw))
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, None
    overrides[key] = str(path)
    config[TAB_OUTPUT_DIRS_KEY] = overrides
    saved = save_local_config(config)
    return saved, str(path) if saved else None


def clear_tab_output_overrides() -> bool:
    """Clear every tab override so all tabs inherit the global default."""
    config = load_local_config()
    config.pop(TAB_OUTPUT_DIRS_KEY, None)
    return save_local_config(config)


STORAGE_SCAN_ROOTS_KEY = "storage_scan_roots"


def get_storage_scan_roots() -> list[str]:
    """Recorded output roots for the Settings → Storage scanner.

    Roots land here automatically when a tool run uses a non-default
    output directory (recorded by ``ui.runner``) or when the user adds
    one from the Storage card. The default output directory and the
    per-tab overrides are scanned regardless and are not duplicated
    here.
    """
    values = load_local_config().get(STORAGE_SCAN_ROOTS_KEY, [])
    if not isinstance(values, list):
        return []
    roots: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip()).expanduser()
        if path.is_absolute() and str(path) not in roots:
            roots.append(str(path))
    return roots


def add_storage_scan_root(path: str | os.PathLike) -> bool:
    """Record an output root for the Storage scanner (deduplicated)."""
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (OSError, ValueError):
        return False
    if not resolved:
        return False
    config = load_local_config()
    values = config.get(STORAGE_SCAN_ROOTS_KEY, [])
    if not isinstance(values, list):
        values = []
    roots: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            candidate = str(Path(value.strip()).expanduser().resolve())
        except (OSError, ValueError):
            continue
        if candidate not in roots:
            roots.append(candidate)
    if resolved in roots:
        return True
    roots.append(resolved)
    config[STORAGE_SCAN_ROOTS_KEY] = roots
    return save_local_config(config)


def remove_storage_scan_root(path: str | os.PathLike) -> bool:
    """Forget a recorded Storage-scanner output root."""
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (OSError, ValueError):
        return False
    config = load_local_config()
    values = config.get(STORAGE_SCAN_ROOTS_KEY, [])
    if not isinstance(values, list):
        return True
    kept: list[str] = []
    changed = False
    for value in values:
        if not isinstance(value, str):
            changed = True
            continue
        try:
            candidate = str(Path(value.strip()).expanduser().resolve())
        except (OSError, ValueError):
            changed = True
            continue
        if candidate == resolved:
            changed = True
            continue
        kept.append(value)
    config[STORAGE_SCAN_ROOTS_KEY] = kept
    if not changed:
        return True
    return save_local_config(config)


def get_auto_suggest_enabled() -> bool:
    """Whether type-ahead neuron-name suggestions are enabled (Settings).

    Independent of the query-history toggle: this drives only the live
    suggestions shown while typing in a neuron input (pathfinding boxes and the
    advanced layer editor cell dropdown).
    """
    return bool(load_local_config().get("auto_suggest_enabled", True))


def set_auto_suggest_enabled(enabled: bool) -> bool:
    """Persist the type-ahead suggestion toggle; applies immediately."""
    config = load_local_config()
    config["auto_suggest_enabled"] = bool(enabled)
    return save_local_config(config)


def get_show_history_enabled() -> bool:
    """Whether the Recent/Frequent query history is enabled (Settings).

    Independent of the type-ahead suggestion toggle: this drives only the
    history list shown when a neuron input's editor is empty (pathfinding
    boxes and the advanced layer editor cell dropdown).
    """
    return bool(load_local_config().get("show_history_enabled", True))


def set_show_history_enabled(enabled: bool) -> bool:
    """Persist the query-history toggle; applies immediately."""
    config = load_local_config()
    config["show_history_enabled"] = bool(enabled)
    return save_local_config(config)


USER_DEFAULTS_KEY = "user_defaults"


def get_user_defaults() -> dict:
    """Return the persisted user default overrides (Settings tab)."""
    overrides = load_local_config().get(USER_DEFAULTS_KEY, {})
    return overrides if isinstance(overrides, dict) else {}


def has_user_default(key: str) -> bool:
    """Whether the user saved a valid custom default for *key*.

    Invalid stored values coerce to nothing and behave like the built-in
    default, so they must not count as a saved override either.
    """
    overrides = get_user_defaults()
    if key not in overrides:
        return False
    return _coerce_user_default(key, overrides[key]) is not None


def _coerce_user_default(key: str, value):
    """Validate a stored override against its registry spec.

    Returns the coerced value, or None when the override is unknown or
    invalid (callers fall back to the built-in DEFAULTS value).
    """
    spec = DEFAULT_SETTING_SPECS.get(key)
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "bool":
        return value if isinstance(value, bool) else None
    if kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = value
        if kind == "int":
            if isinstance(number, float) and not number.is_integer():
                return None
            number = int(number)
        else:
            number = float(number)
        low, high = spec.get("min"), spec.get("max")
        if low is not None and number < low:
            return None
        if high is not None and number > high:
            return None
        return number
    if kind == "combo":
        # Dropdown options plus free text matching the backend's fold
        # notation ('real', any number, 'Nx real'); anything else is
        # rejected so an invalid saved override falls back to DEFAULTS.
        if not isinstance(value, str) or not value.strip():
            return None
        options = spec.get("options") or []
        text = value.strip().lower()
        if text in [str(o).lower() for o in options]:
            return value.strip()
        if key == "synapse_size" and is_valid_synapse_size(value):
            return value.strip()
        return None
    options = spec.get("options") or []
    if kind == "select" and options is DATASETS:
        # Legacy BANC spellings canonicalize so saved defaults survive the
        # flywire_BANC_* -> banc_* rename.
        value = canonical_dataset_name(str(value).strip())
    if key == "brain_mesh":
        # Legacy 'template'/'whole' selections map onto the renamed
        # native/fafb options so saved defaults survive the rename.
        value = normalize_brain_mesh_choice(value)
    return value if value in options else None


def get_user_default(key: str):
    """Resolve one default: a valid saved user override wins over DEFAULTS."""
    overrides = get_user_defaults()
    if key in overrides:
        coerced = _coerce_user_default(key, overrides[key])
        if coerced is not None:
            return coerced
    return DEFAULTS[key]


def set_user_default(key: str, value) -> bool:
    """Persist one validated default override; returns success."""
    coerced = _coerce_user_default(key, value)
    if coerced is None:
        return False
    config = load_local_config()
    overrides = config.get(USER_DEFAULTS_KEY, {})
    if not isinstance(overrides, dict):
        overrides = {}
    overrides[key] = coerced
    config[USER_DEFAULTS_KEY] = overrides
    return save_local_config(config)


def reset_user_default(key: str) -> bool:
    """Clear one saved default override so the built-in applies again."""
    config = load_local_config()
    overrides = config.get(USER_DEFAULTS_KEY, {})
    if not isinstance(overrides, dict) or key not in overrides:
        return True
    overrides.pop(key, None)
    if overrides:
        config[USER_DEFAULTS_KEY] = overrides
    else:
        config.pop(USER_DEFAULTS_KEY, None)
    return save_local_config(config)


def reset_user_defaults() -> bool:
    """Clear every saved default override (restore built-in DEFAULTS)."""
    config = load_local_config()
    config.pop(USER_DEFAULTS_KEY, None)
    return save_local_config(config)

# Available datasets (static fallback - use dataset_service for dynamic fetching)
DATASETS = [
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
    "flywire_FAFB_v783",
    "banc_v888",
    "banc_v626",
]

# NeuPrint server datasets (can be fetched dynamically via /api/dbmeta/datasets)
NEUPRINT_DATASETS = [
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
]

# FAFB release datasets (converted local files; CAVE is only relevant to
# explicit FAFB API/skeleton fallbacks).
FLYWIRE_DATASETS = [
    "flywire_FAFB_v783",
]

# Standalone BANC public-release datasets.  BANC preparation and skeleton
# fetches use the public release bucket and never require a CAVE token.
BANC_DATASETS = [
    "banc_v888",
    "banc_v626",
]

# Default parameter values
DEFAULTS = {
    "min_synapse_num": 3,
    "min_ratio": 0.0,
    "min_traversal_probability": 0.0,
    "max_interlayer": 2,
    "edgeN_limit": 500,
    "filter_by": "bodyId",
    "search_columns": "auto",
    "output_format": "csv",
    "pathfinding": "StrongestFirst",
    # Fix A: 0 = auto (StrongestFirst uses its internal 1M budget;
    # complete enumerators run unbounded). Explicit >0 = that budget.
    # Applies to FindAllPath and FindShortestPath (min-hop paths) alike.
    "max_paths_bodyid": 0,
    # Fix D (Edge Budget): after lossless pruning, cones exceeding this
    # many bodyId edges are floored just above the N-th strongest edge's
    # weight (w0 = w1 + 1). 0 = off.
    "graph_edge_limit_bodyid": 1000000,
    "network_layout": "distributed",
    "use_cache": True,
    "cache_only": False,
    # NeuronBridge match cache: OFF by default — NB queries are large and
    # rarely reused (unlike the connection-data caches), so the per-body
    # match tables would accumulate without ever paying off. Applies to
    # Find Driver Lines / Find EM Neurons / Co-Labeling.
    "nb_use_cache": False,
    # Cache defaults (Settings -> Default Settings). Connection and
    # skeleton caches both stay on by default: fetched skeletons persist as
    # portable .swc.zst files in the shared cache, so every tab benefits.
    "cache_neurons": True,
    "cache_synapses": True,
    "auto_type_mapping": True,
    "skip_bodyId": True,
    # Feature F: enumerate once at the lowest threshold, materialize the
    # rest from the bottleneck-annotated path set ('all' path mode only).
    "replay_paths": True,
    # Untyped-neuron drop: edges touching untyped neurons (Unknown /
    # bodyId-fallback labels) are removed from cross-dataset results;
    # dropped rows are exported and counted in user_warning_notes.
    "drop_untyped": True,
    "showfig_analysis": False,
    # Exported per-run user guide (_UserGuide_please_read_me.<ext>)
    "run_guide_format": "html",
    # Default dataset selections
    "default_dataset": "male-cns:v1.0",
    "default_target_dataset": "male-cns:v0.9",
    # 3D skeleton rendering defaults
    "skeleton_mode": "tube",
    "analysis_skeleton_mode": "line",
    "simplification_method": "fast",
    # Show Figure defaults OFF globally; only the Visualization > Skeleton
    # tab still opens the figure by default (unless the user saves an
    # explicit override in Settings).
    "show_fig_skeleton": False,
    "export_views": True,
    "legend_mode": "tree",
    # Individual profiles group per native legend entry (the historical
    # behaviour); 'layer'/'type'/'body' follow the tree legend's own levels.
    "profile_granularity": "legend",
    # The exported HTML viewer pins its 3D axes so legend show/hide cannot
    # rescale the scene (a Freeze/Fit button and the 'F' key restore it).
    "freeze_view": True,
    "background": "white",
    "brain_mesh": "native",
    "synapse_size": "1",
    "uniform_synapse_size": False,
    # Similarity / homolog search toggles
    "fast_search": True,
    "vector_prefilter": True,
    "expand_2hop": True,
    "top_k": 25,
    "top_m": 5,
    "similarity_metric": "jaccard",  # Sort By (homolog candidates): sorting only — all metrics are always computed. Jaccard is the conservative bodyId-level default (benchmark-best, bounded [0,1], never degenerate); see BENCHMARK_RESULTS.md §7.
    "top_n": 100,
    "match_algorithm": "cds",
    # NeuronBridge Find Neurons / Co-Labeling shared controls
    "nb_top_n": 50,
    "nb_min_score": 30000,
    "nb_min_type_avg_score": 10000,
    # Find Similar Neurons (morphological mode)
    "morph_level": "auto",
    "morph_method": "vector_v2",
    "candidate_source": "auto",
    "candidate_cap": 500,
    "morph_visualize_top_n": 10,
    "morph_visualize_by": "type",
}

# Pathfinding algorithms (names match the FastGraph implementations:
# StrongestFirst = budgeted best-first on path bottleneck (emits intact
# paths strongest-first; see find_paths_strongest_first),
# MemoizedDFS = memoized DFS forward, DFS = memoized DFS backward,
# MeetInMiddle = meet-in-the-middle)
PATHFINDING_ALGORITHMS = ["StrongestFirst", "Bidirectional", "DP", "MemoizedDFS", "MeetInMiddle", "DFS"]

# Columns searched when resolving source/target neuron names
SEARCH_COLUMNS = ["auto", "type", "instance", "bodyId"]

# Filter options
FILTER_OPTIONS = ["bodyId", "type"]

# Output formats
OUTPUT_FORMATS = ["csv", "xlsx"]

# Exported per-run user guide formats (_UserGuide_please_read_me.<ext>)
RUN_GUIDE_FORMATS = ["html", "txt", "markdown", "disabled"]

# Network layouts
NETWORK_LAYOUTS = ["distributed", "circular", "shell", "spring"]

# Similarity metrics
# rank_corr (shared-only Spearman) is intentionally excluded: it is fragile at
# bodyId level (<3 shared types or low variance -> NaN, MRR 0.34 at (15,5) per
# the homolog-parameter benchmark), and the UI default is rank_union, which
# dominates at both bodyId and pooled type level. See BENCHMARK_RESULTS.md §6c.
SIMILARITY_METRICS = ["rank_union", "jaccard", "cosine"]

# Match algorithms for NeuronBridge
MATCH_ALGORITHMS = ["cds", "pppm", "both"]

# Comparison modes
COMPARISON_MODES = ["path", "edge"]
PATH_MODES = ["all", "shortest"]

# Skeleton modes
SKELETON_MODES = ["tube", "line"]

# Individual-profile granularity levels (mirrors
# visualize_skeleton.PROFILE_GRANULARITIES): 'legend' keeps the historical
# one-profile-per-legend-entry behaviour, 'layer' / 'type' / 'body' follow the
# tree legend's own levels instead. The Settings registry stores the backend
# values; the Skeleton tab shows them through NiceGUI's value -> label dict.
PROFILE_GRANULARITIES = ["legend", "layer", "type", "body"]
PROFILE_GRANULARITY_CHOICES = {
    "legend": "Per legend entry",
    "layer": "Per layer",
    "type": "Per neuron type",
    "body": "Per bodyId",
}

# Synapse rendering modes (mirrors VisualizeSkeleton.synapse_mode).
# 'pre_post' renders the input/output SITES of the queried neurons rather
# than the paired inter-layer synapses.
SYNAPSE_MODE_OPTIONS = ["cone", "scatter", "sphere", "tetrahedron", "pre_post"]

# Top-level synapse view modes (Skeleton tab). 'skip' maps to
# VisualizeSkeleton.skip_synapse=True; 'pre-post sites' maps to
# synapse_mode='pre_post' with its own shape selector.
SYNAPSE_VIEW_MODES = ["synapse", "pre-post sites", "skip"]
# Marker-shape choices for 'pre-post sites' mode (maps to pre_post_scatter).
# Scatter is the default (and the recommended choice) because it keeps the
# exported HTML much smaller than solid cone/sphere meshes.
PRE_POST_SHAPES = ["solid (spheres + cones)", "scatter (circles + diamonds)"]
# Layer editor modes (Skeleton tab), shown as segmented buttons.
LAYER_EDITOR_MODES = ["Standard", "Advanced", "File upload"]
# Net-Viz canvas source modes, shown as the same segmented-button row.
NET_VIZ_SOURCE_MODES = ["Edge list editor", "Empty canvas", "File upload"]

# Brain-mesh option tokens + legacy normalization are canonical in
# src/utils/naming_utils and shared with the renderer.
try:
    from src.utils.naming_utils import (
        BRAIN_MESH_OPTIONS, normalize_brain_mesh_choice)
except ImportError:  # src not importable; keep the current option set
    BRAIN_MESH_OPTIONS = ["native", "BANC", "FAFB", "male-cns", "none"]

    def normalize_brain_mesh_choice(value) -> str:
        return str(value or "").strip().lower()

# Synapse size presets (screen-space pixels, 1-12, default 3); the UI combo
# box additionally accepts any typed integer in that range.
SYNAPSE_SIZE_OPTIONS = [str(value) for value in range(1, 13)]

# The Synapse Size control is a pixel value (1-12, default 3); validate the
# free-typed combo value before it reaches the backend. Only a bare integer in
# that range is accepted (no 'real' / 'Nx real' notation).
_SYNAPSE_SIZE_RE = re.compile(r'^([0-9]+)$')


def is_valid_synapse_size(value) -> bool:
    """Return True when *value* is a valid synapse size (an integer 1-12)."""
    if not isinstance(value, str):
        return False
    match = _SYNAPSE_SIZE_RE.match(value.strip())
    if not match:
        return False
    return 1 <= int(match.group(1)) <= 12

# Morphology similarity options (Morphology tab, Find Similar sub-tab)
MORPH_LEVEL_OPTIONS = ["auto", "bodyid", "type"]
MORPH_METHOD_OPTIONS = ["vector_v2", "nblast"]
CANDIDATE_SOURCE_OPTIONS = ["auto", "roi", "combined", "profile", "cache"]

# Simplification pipelines (NeuPrint tube rendering)
SIMPLIFICATION_METHODS = ["fast", "fine", "artistic"]

# On-disk skeleton cache simplification levels (percent of nodes removed,
# 0-90; mirrors morphology.DEFAULT_SIMPLIFICATION). 90 = the canonical
# "simp90" cache that keeps ~10% of nodes; 0 = raw.
CACHE_SIMPLIFICATION_OPTIONS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# User-configurable defaults rendered in the Settings tab. Each spec drives
# the Default Settings card and validates overrides saved to
# local_config.json under ``user_defaults``; the built-in fallback always
# comes from DEFAULTS.
DEFAULT_SETTING_GROUPS = [
    ("dataset_search", "Dataset & Search"),
    ("thresholds", "Connection Thresholds"),
    ("cache", "Cache & Data"),
    ("pathfinding_output", "Pathfinding & Output"),
    ("skeleton_render", "3D Skeleton Rendering"),
    ("similarity", "Similarity & Comparison Search"),
]

DEFAULT_SETTING_SPECS = {
    # --- Dataset & Search -------------------------------------------------
    "default_dataset": {
        "label": "Default Dataset",
        "group": "dataset_search",
        "kind": "select",
        "options": DATASETS,
        "hint": "Dataset preselected in every tool tab until changed.",
    },
    "default_target_dataset": {
        "label": "Default Similar-Search Target Dataset",
        "group": "dataset_search",
        "kind": "select",
        "options": DATASETS,
        "hint": "Target dataset preselected in Connectivity → Find Similar.",
    },
    "search_columns": {
        "label": "Search Columns",
        "group": "dataset_search",
        "kind": "select",
        "options": SEARCH_COLUMNS,
        "hint": "Columns searched when resolving neuron names.",
    },
    "filter_by": {
        "label": "Filter By",
        "group": "dataset_search",
        "kind": "select",
        "options": FILTER_OPTIONS,
        "hint": "'bodyId': individual neuron level. 'type': aggregate by type.",
    },
    # --- Connection Thresholds --------------------------------------------
    "min_synapse_num": {
        "label": "Min Synapse Count / Threshold",
        "group": "thresholds",
        "kind": "int",
        "min": 1,
        "max": 100,
        "step": 1,
        "hint": "Default minimum synapses for a connection to be included "
                "(pathfinding, network, skeleton tab, comparison, "
                "similar search).",
    },
    # F9: Min Connection Ratio / Min Traversal Probability are READOUT
    # columns now (weight / all-post incoming weight; ratio/0.3 capped) —
    # the Settings entrances were removed with the tab entrances. The
    # DEFAULTS keys survive for payload/back-compat.
    "edgeN_limit": {
        "label": "Visualization Edge Limit",
        "group": "thresholds",
        "kind": "int",
        "min": 10,
        "max": 5000,
        "step": 10,
        "hint": "Maximum edges drawn per visualization.",
    },
    # --- Cache & Data ------------------------------------------------------
    "use_cache": {
        "label": "Connection Cache (Use Cache)",
        "group": "cache",
        "kind": "bool",
        "hint": "Cache neuron/connection data locally for faster repeat runs.",
    },
    "cache_only": {
        "label": "Cache Only (Offline)",
        "group": "cache",
        "kind": "bool",
        "hint": "Never contact the server; requires a pre-built cache.",
    },
    "nb_use_cache": {
        "label": "NeuronBridge Match Cache",
        "group": "cache",
        "kind": "bool",
        "hint": "Cache NeuronBridge match tables across runs (Find Driver "
                "Lines / Find EM Neurons / Co-Labeling). Off by default: "
                "NB queries are large and rarely reused; re-running "
                "refetches. On, repeat queries resolve from the local "
                "match cache instead of the API.",
    },
    "cache_neurons": {
        "label": "Skeleton Cache (Cache Neurons)",
        "group": "cache",
        "kind": "bool",
        "hint": "Cache fetched skeletons as portable .swc.zst files in the "
                "shared cache for faster repeat renders (default on). "
                "Saving a value here disables the dataset-aware auto-flip.",
    },
    "cache_synapses": {
        "label": "Synapse Cache (Cache Synapses)",
        "group": "cache",
        "kind": "bool",
        "hint": "Cache fetched synapse data locally.",
    },
    "auto_type_mapping": {
        "label": "Auto Type Mapping",
        "group": "cache",
        "kind": "bool",
        "hint": "Standardize type names across datasets via male-cns v1.0.",
    },
    # --- Pathfinding & Output ----------------------------------------------
    "skip_bodyId": {
        "label": "Skip BodyId-Level Export",
        "group": "pathfinding_output",
        "kind": "bool",
        "hint": "Exclude bodyId-level results/tables; keep type-level output. "
                "Default on: type-level analysis (incl. cross-dataset type "
                "mapping) needs no bodyId exports, which can run to multiple "
                "GB per run (bodyId connMatrix CSVs, raw path lists).",
    },
    "drop_untyped": {
        "label": "Drop Untyped Neurons",
        "group": "pathfinding_output",
        "kind": "bool",
        "hint": "Neuron-label filter shared by Complete Paths, Shortest "
                "Paths, and Cross-Dataset Comparison: remove edges touching "
                "untyped neurons (empty / Unknown / bodyId-fallback labels). "
                "Dropped rows are exported (data_details/"
                "untyped_dropped_records.csv per pathfinding run; "
                "comparison_results/untyped_dropped_records.csv for "
                "comparison runs) and counted in user_warning_notes.txt.",
    },
    "replay_paths": {
        "label": "Replay Paths (single enumeration)",
        "group": "pathfinding_output",
        "kind": "bool",
        "hint": "Path mode 'all': enumerate ONCE at the lowest threshold and "
                "materialize every higher threshold from the bottleneck-"
                "annotated path set — identical outputs, no re-enumeration. "
                "Shortest mode is never replayed (min-hop sets are not "
                "nested across thresholds).",
    },
    "output_format": {
        "label": "Output Format",
        "group": "pathfinding_output",
        "kind": "select",
        "options": OUTPUT_FORMATS,
        "hint": "'csv': faster, smaller. 'xlsx': Excel format.",
    },
    "run_guide_format": {
        "label": "Run Guide Format",
        "group": "pathfinding_output",
        "kind": "select",
        "options": RUN_GUIDE_FORMATS,
        "hint": "Exported _UserGuide_please_read_me file in every run "
                "folder: 'html' (default), 'txt', 'markdown', or "
                "'disabled' to skip writing it.",
    },
    "network_layout": {
        "label": "Network Layout",
        "group": "pathfinding_output",
        "kind": "select",
        "options": NETWORK_LAYOUTS,
        "hint": "Layout algorithm for the HTML network visualization.",
    },
    "max_paths_bodyid": {
        "label": "Max Paths (BodyId)",
        "group": "pathfinding_output",
        "kind": "int",
        "min": 0,
        "max": 100000000,
        "step": 1000,
        "hint": "Path-output budget for StrongestFirst enumeration (bodyId "
                "level): when the search exceeds it, ALL paths above the "
                "achieved strength cutoff (tau) are kept and tau is "
                "reported. Filters the emitted paths only — the graph is "
                "not trimmed. 0 = auto (StrongestFirst: 1M budget).",
    },
    "graph_edge_limit_bodyid": {
        "label": "Edge Budget",
        "group": "pathfinding_output",
        "kind": "int",
        "min": 0,
        "max": 100000000,
        "step": 100000,
        "hint": "Graph filter ('all' path mode only): after the lossless "
                "prunes, discovery cones exceeding this many bodyId edges "
                "are floored just above the N-th strongest edge's weight "
                "(w0 = w1 + 1) — exactly equivalent to raising the "
                "threshold; the applied floor is reported as "
                "edge_weight_floor. Distinct from the drawing-only "
                "Visualization Edge Limit. 0 = off. Shortest mode is "
                "never floored.",
    },
    "showfig_analysis": {
        "label": "Show Figure (Analysis Tabs)",
        "group": "pathfinding_output",
        "kind": "bool",
        "hint": "Open the HTML visualization automatically after "
                "Complete/Shortest Paths runs.",
    },
    "max_interlayer": {
        "label": "Max Intermediate Layers (Complete Paths)",
        "group": "pathfinding_output",
        "kind": "int",
        "min": 0,
        "max": 10,
        "step": 1,
        "hint": "Default intermediate neuron layers for Complete Paths.",
    },
    # --- 3D Skeleton Rendering ---------------------------------------------
    "skeleton_mode": {
        "label": "Skeleton Mode (Skeleton tab)",
        "group": "skeleton_render",
        "kind": "select",
        "options": SKELETON_MODES,
        "hint": "'tube': detailed. 'line': fast, for many neurons.",
    },
    "analysis_skeleton_mode": {
        "label": "Skeleton Mode (analysis visualizations)",
        "group": "skeleton_render",
        "kind": "select",
        "options": SKELETON_MODES,
        "hint": "Skeleton mode for optional renders generated by analysis tabs.",
    },
    "simplification_method": {
        "label": "Simplification Method",
        "group": "skeleton_render",
        "kind": "select",
        "options": SIMPLIFICATION_METHODS,
        "hint": "NeuPrint tube rendering pipeline (fast / fine / artistic).",
    },
    "show_fig_skeleton": {
        "label": "Show Figure (Skeleton renders)",
        "group": "skeleton_render",
        "kind": "bool",
        "hint": "Default Show Figure state for skeleton renders. The "
                "Visualization > Skeleton tab ignores this built-in default "
                "(it opens the figure by default) unless you save an "
                "override; analysis tabs start unchecked.",
    },
    "export_views": {
        "label": "Export Views",
        "group": "skeleton_render",
        "kind": "bool",
        "hint": "Export PNG screenshots from 6 angles after rendering.",
    },
    "legend_mode": {
        "label": "Neuron Legend Mode",
        "group": "skeleton_render",
        "kind": "select",
        "options": ["layer", "type", "tree", "single"],
        "hint": "One legend entry per layer, type, or individual neuron. "
                "'tree' adds an expandable type -> neuron legend panel "
                "to the exported interactive HTML. Independent of the "
                "individual-profile granularity below.",
    },
    "profile_granularity": {
        "label": "Individual Profile Granularity",
        "group": "skeleton_render",
        "kind": "select",
        "options": PROFILE_GRANULARITIES,
        "hint": "Grouping for the exported individual profiles: 'legend' = "
                "one profile per legend entry (the historical behaviour), "
                "'layer' = per layer, 'type' = per neuron type, 'body' = per "
                "bodyId leaf (one profile per neuron; can be many files).",
    },
    "freeze_view": {
        "label": "Freeze 3D View",
        "group": "skeleton_render",
        "kind": "bool",
        "hint": "Pin the exported HTML viewer's 3D axes so showing or "
                "hiding traces in the legend cannot rescale the scene. A "
                "Freeze/Fit button and the 'F' key restore autoscaling.",
    },
    "background": {
        "label": "Background",
        "group": "skeleton_render",
        "kind": "select",
        "options": ["white", "black"],
        "hint": "Background color for the 3D scene and exports.",
    },
    "brain_mesh": {
        "label": "Brain Mesh",
        "group": "skeleton_render",
        "kind": "select",
        "options": BRAIN_MESH_OPTIONS,
        "hint": "Template for the scene: 'native' uses the dataset's own "
                "outline and coordinates; 'BANC'/'FAFB'/'male-cns' move the "
                "whole scene (neurons included) into that template's "
                "coordinates with its outline. 'none' hides the outline.",
    },
    "synapse_size": {
        "label": "Synapse Size",
        "group": "skeleton_render",
        "kind": "combo",
        "options": SYNAPSE_SIZE_OPTIONS,
        "hint": "Marker size in pixels (1-12, default 3); scatter uses it "
                "directly, mesh modes map 3 px = 1x the real distance.",
    },
    "uniform_synapse_size": {
        "label": "Uniform Synapse Size",
        "group": "skeleton_render",
        "kind": "bool",
        "hint": "Use the median pre→post distance for every synapse marker "
                "so all markers share one size.",
    },
    # --- Similarity & Comparison Search --------------------------------------
    "top_n": {
        "label": "Top N Candidates",
        "group": "similarity",
        "kind": "int",
        "min": 5,
        "max": 100,
        "step": 1,
        "hint": "Number of top candidates returned by similarity searches.",
    },
    "top_k": {
        "label": "Top K Partners",
        "group": "similarity",
        "kind": "int",
        "min": 5,
        "max": 50,
        "step": 1,
        "hint": "Top K partners per direction for profile construction.",
    },
    "top_m": {
        "label": "Min Types (M)",
        "group": "similarity",
        "kind": "int",
        "min": 3,
        "max": 20,
        "step": 1,
        "hint": "Minimum unique partner types in a connectivity profile.",
    },
    "similarity_metric": {
        "label": "Similarity Metric (Sort By)",
        "group": "similarity",
        "kind": "select",
        "options": SIMILARITY_METRICS,
        "hint": "Metric used for ordering candidate lists.",
    },
    "fast_search": {
        "label": "Fast Search",
        "group": "similarity",
        "kind": "bool",
        "hint": "Adjacency-expansion candidate discovery.",
    },
    "vector_prefilter": {
        "label": "Vector Pre-filtering",
        "group": "similarity",
        "kind": "bool",
        "hint": "Cosine pre-filter of candidates for speed.",
    },
    "expand_2hop": {
        "label": "2-Hop Expansion",
        "group": "similarity",
        "kind": "bool",
        "hint": "Include untyped partners via 2-hop typed partners.",
    },
    "candidate_cap": {
        "label": "Candidate Cap",
        "group": "similarity",
        "kind": "int",
        "min": 10,
        "max": 5000,
        "step": 10,
        "hint": "Maximum candidates entering morphological comparison.",
    },
    "candidate_source": {
        "label": "Candidate Source",
        "group": "similarity",
        "kind": "select",
        "options": CANDIDATE_SOURCE_OPTIONS,
        "hint": "Discovery pool for morphological similarity.",
    },
    "morph_level": {
        "label": "Morphology Level",
        "group": "similarity",
        "kind": "select",
        "options": MORPH_LEVEL_OPTIONS,
        "hint": "'auto' follows the query kind (type vs bodyId).",
    },
    "morph_method": {
        "label": "Morphology Method",
        "group": "similarity",
        "kind": "select",
        "options": MORPH_METHOD_OPTIONS,
        "hint": "'vector': fast morphometrics. 'nblast': canonical NBLAST.",
    },
    "match_algorithm": {
        "label": "NeuronBridge Algorithm",
        "group": "similarity",
        "kind": "select",
        "options": MATCH_ALGORITHMS,
        "hint": "'cds': Color Depth Search. 'pppm': Point Pattern. 'both'.",
    },
    "nb_top_n": {
        "label": "NeuronBridge Top N Matches Per Line",
        "group": "similarity",
        "kind": "int",
        "min": 1,
        "max": 2000,
        "step": 10,
        "hint": "Highest-scoring matches retrieved per driver line.",
    },
    "nb_min_score": {
        "label": "NeuronBridge Score Cutoff",
        "group": "similarity",
        "kind": "float",
        "min": 0,
        "max": 200000,
        "step": 1000,
        "hint": "Threshold for score-based NeuronBridge views.",
    },
    "nb_min_type_avg_score": {
        "label": "NeuronBridge Min Type Avg Score",
        "group": "similarity",
        "kind": "float",
        "min": 0,
        "max": 200000,
        "step": 1000,
        "hint": "Extra filter for co-labeling similarity matrices.",
    },
}

# App settings
APP_TITLE = "DROCAT - Connectome Analysis Toolkit"
APP_VERSION = "4.5.0"
APP_PORT = 8080
APP_HOST = "127.0.0.1"

# Keep external project links aligned with the checkout that is running the
# UI. Deployments without a .git directory fall back to the matching version
# branch, so a release build never links back to an unrelated `main` page.
GITHUB_REPOSITORY_URL = (
    "https://github.com/Swida-Alba/"
    "Drosophila-cross-dataset-connectome-analysis"
)


def _current_docs_branch() -> str:
    """Resolve the branch used by versioned GitHub documentation links."""
    configured = os.environ.get("DROCAT_DOCS_BRANCH", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        branch = result.stdout.strip()
        if branch:
            return branch
    except (OSError, subprocess.SubprocessError):
        pass
    return f"v{APP_VERSION}"


APP_DOCS_BRANCH = _current_docs_branch()
APP_GITHUB_URL = f"{GITHUB_REPOSITORY_URL}/tree/{APP_DOCS_BRANCH}"
APP_DOCS_URL = f"{GITHUB_REPOSITORY_URL}/blob/{APP_DOCS_BRANCH}/README.md"
